#!/usr/bin/env python3
"""
scripts/get_hists_nhits.py
--------------------------
RDataFrame-based histogram generator for SciFi and MuFilter hit distributions
and Upstream (US) QDC deposits across:
  - GENIE signal (charm -> direct muon)
  - GENIE indirect fake signal (charm -> hadrons -> muon)
  - GENIE background (everything else)
  - Muon simulation background (--input-pmu)
  - Collision data (--input-data)

Output: Unified ROOT file with TDirectory structure.
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
import yaml
import ROOT

# Add project root to sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import DataManager, load_trident_libraries

load_trident_libraries(_repo_root)


def declare_cpp_hit_metrics():
    """Declare fast C++ helper to compute hit occupancies and US total QDC."""
    ROOT.gInterpreter.Declare("""
#ifndef HIT_METRICS_HELPER_H
#define HIT_METRICS_HELPER_H

#include <TClonesArray.h>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include "Scifi.h"
#include "MuFilter.h"
#include "sndScifiHit.h"
#include "MuFilterHit.h"
#include "sndRecoTrack.h"

namespace HitMetrics {
    struct ChannelInfo {
        float z{0.0f};
        float pos{0.0f};
        bool is_vert{false};
        bool valid{false};
    };

    inline std::vector<ChannelInfo>& getScifiChannelVec() {
        static std::vector<ChannelInfo> vec;
        return vec;
    }

    inline void init_scifi_geometry() {
        Scifi* scifi = dynamic_cast<Scifi*>(gROOT->GetListOfGlobals()->FindObject("Scifi"));
        if (!scifi) return;
        auto& vec = getScifiChannelVec();
        if (vec.size() < 6000000) {
            vec.assign(6000000, ChannelInfo{});
        }
        TVector3 left, right;
        for (int s = 1; s <= 5; ++s) {
            for (int o = 0; o <= 1; ++o) {
                for (int m = 0; m <= 2; ++m) {
                    for (int sipm = 0; sipm <= 3; ++sipm) {
                        for (int ch = 0; ch < 128; ++ch) {
                            int detID = s * 1000000 + o * 100000 + m * 10000 + sipm * 1000 + ch;
                            scifi->GetSiPMPosition(detID, left, right);
                            ChannelInfo info;
                            info.z = 0.5f * (left.Z() + right.Z());
                            info.is_vert = (o == 1);
                            info.pos = info.is_vert ? 0.5f * (left.X() + right.X()) : 0.5f * (left.Y() + right.Y());
                            info.valid = true;
                            if (detID < (int)vec.size()) {
                                vec[detID] = info;
                            }
                        }
                    }
                }
            }
        }
    }

    struct Metrics {
        int n_scifi_hits;
        int n_ds_hits;
        int max_scifi_plane_hits;
        double us_total_qdc;
        int max_ds_plane_hits;
        int n_ds_h_hits;
        int n_ds_v_hits;
        double ds_min_max_ratio_all;
        double ds_min_max_ratio_h;
        double ds_min_max_ratio_v;
        double scifi_min_max_ratio_all;
        double scifi_min_max_ratio_h;
        double scifi_min_max_ratio_v;
        double us_qdc_min_max_ratio;
        double us_qdc_plane4_over_plane0;
        int n_scifi_v_hits_pm2cm;
        int n_scifi_h_hits_pm2cm;
        int n_scifi_all_hits_pm2cm;
        double scifi_trk_iso_all;
        double scifi_trk_iso_h;
        double scifi_trk_iso_v;
    };

    inline double safe_ratio(double min_val, double max_val) {
        if (max_val <= 0.0) return 0.0;
        double r = min_val / max_val;
        return (r >= 1.0) ? 0.999999 : ((r < 0.0) ? 0.0 : r);
    }

    inline Metrics compute(const TClonesArray& scifiHits, const TClonesArray& mufiHits, const TClonesArray& recoTracks) {
        Metrics m{0, 0, 0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0};

        int n_trks = recoTracks.GetEntries();
        const auto& ch_vec = getScifiChannelVec();
        int n_sf_h = 0;
        int n_sf_v = 0;
        int n_sf_iso_all = 0;
        int n_sf_iso_h = 0;
        int n_sf_iso_v = 0;

        // 1. SciFi Hits: 5 stations * 2 orientations (vert=0, horiz=1) -> 10 planes
        int scifi_plane_hits[10] = {0};
        int scifi_h_hits[5] = {0};
        int scifi_v_hits[5] = {0};
        int n_sf = scifiHits.GetEntries();
        for (int i = 0; i < n_sf; ++i) {
            auto* h = static_cast<sndScifiHit*>(scifiHits.At(i));
            if (!h || !h->isValid()) continue;
            m.n_scifi_hits++;
            int st = h->GetStation();
            bool is_vert = h->isVertical();
            int o = is_vert ? 0 : 1;
            if (st >= 1 && st <= 5) {
                int p_idx = (st - 1) * 2 + o;
                scifi_plane_hits[p_idx]++;
                if (is_vert) {
                    scifi_v_hits[st - 1]++;
                } else {
                    scifi_h_hits[st - 1]++;
                }
            }
            if (is_vert) {
                n_sf_v++;
            } else {
                n_sf_h++;
            }

            int detID = h->GetDetectorID();
            bool in_window = false;
            if (n_trks > 0 && detID < (int)ch_vec.size() && ch_vec[detID].valid) {
                const auto& ch = ch_vec[detID];
                for (int t = 0; t < n_trks; ++t) {
                    auto* trk = static_cast<sndRecoTrack*>(recoTracks.At(t));
                    if (!trk) continue;
                    TVector3 trkPos = trk->getPointAtZ(ch.z);
                    float dist = is_vert ? std::abs(trkPos.X() - ch.pos) : std::abs(trkPos.Y() - ch.pos);
                    if (dist <= 2.0f) {
                        in_window = true;
                        break;
                    }
                }
            }
            if (in_window) {
                n_sf_iso_all++;
                if (is_vert) {
                    n_sf_iso_v++;
                } else {
                    n_sf_iso_h++;
                }
            }
        }
        for (int p = 0; p < 10; ++p) {
            if (scifi_plane_hits[p] > m.max_scifi_plane_hits) {
                m.max_scifi_plane_hits = scifi_plane_hits[p];
            }
        }

        m.n_scifi_v_hits_pm2cm = n_sf_iso_v;
        m.n_scifi_h_hits_pm2cm = n_sf_iso_h;
        m.n_scifi_all_hits_pm2cm = n_sf_iso_all;
        m.scifi_trk_iso_all = (m.n_scifi_hits > 0) ? (static_cast<double>(n_sf_iso_all) / m.n_scifi_hits) : 0.0;
        m.scifi_trk_iso_h = (n_sf_h > 0) ? (static_cast<double>(n_sf_iso_h) / n_sf_h) : 0.0;
        m.scifi_trk_iso_v = (n_sf_v > 0) ? (static_cast<double>(n_sf_iso_v) / n_sf_v) : 0.0;

        // SciFi occupancy ratios
        int sf_all_min = scifi_plane_hits[0];
        int sf_all_max = scifi_plane_hits[0];
        for (int p = 1; p < 10; ++p) {
            if (scifi_plane_hits[p] < sf_all_min) sf_all_min = scifi_plane_hits[p];
            if (scifi_plane_hits[p] > sf_all_max) sf_all_max = scifi_plane_hits[p];
        }
        m.scifi_min_max_ratio_all = safe_ratio(sf_all_min, sf_all_max);

        int sf_h_min = scifi_h_hits[0];
        int sf_h_max = scifi_h_hits[0];
        for (int p = 1; p < 5; ++p) {
            if (scifi_h_hits[p] < sf_h_min) sf_h_min = scifi_h_hits[p];
            if (scifi_h_hits[p] > sf_h_max) sf_h_max = scifi_h_hits[p];
        }
        m.scifi_min_max_ratio_h = safe_ratio(sf_h_min, sf_h_max);

        int sf_v_min = scifi_v_hits[0];
        int sf_v_max = scifi_v_hits[0];
        for (int p = 1; p < 5; ++p) {
            if (scifi_v_hits[p] < sf_v_min) sf_v_min = scifi_v_hits[p];
            if (scifi_v_hits[p] > sf_v_max) sf_v_max = scifi_v_hits[p];
        }
        m.scifi_min_max_ratio_v = safe_ratio(sf_v_min, sf_v_max);

        // 2. MuFilter Hits: US (system 2) and DS (system 3)
        // US: 5 planes (0..4), all horizontal
        double us_plane_qdc[5] = {0.0};

        // DS has 3 horizontal planes (stations 0, 1, 2) and 4 vertical planes (stations 0, 1, 2, 3) -> 7 planes
        int ds_plane_hits[7] = {0};
        int ds_h_hits[3] = {0};
        int ds_v_hits[4] = {0};

        int n_mf = mufiHits.GetEntries();
        for (int i = 0; i < n_mf; ++i) {
            auto* h = static_cast<MuFilterHit*>(mufiHits.At(i));
            if (!h || !h->isValid()) continue;
            int sys = h->GetSystem();
            int det_id = h->GetDetectorID();
            int plane_idx = (det_id % 10000) / 1000;
            bool is_vert = h->isVertical();
            int o_idx = is_vert ? 1 : 0;

            if (sys == 2) {
                // Upstream: sum all SiPM QDC deposits using GetAllSignals()
                auto sigs = h->GetAllSignals();
                for (const auto& kv : sigs) {
                    m.us_total_qdc += kv.second;
                    if (plane_idx >= 0 && plane_idx < 5) {
                        us_plane_qdc[plane_idx] += kv.second;
                    }
                }
            } else if (sys == 3) {
                // Downstream
                m.n_ds_hits++;
                if (is_vert) {
                    m.n_ds_v_hits++;
                    if (plane_idx >= 0 && plane_idx < 4) {
                        ds_v_hits[plane_idx]++;
                    }
                } else {
                    m.n_ds_h_hits++;
                    if (plane_idx >= 0 && plane_idx < 3) {
                        ds_h_hits[plane_idx]++;
                    }
                }
                if (plane_idx >= 0 && plane_idx < 4) {
                    int global_ds_plane = (plane_idx < 3) ? (plane_idx * 2 + o_idx) : 6;
                    if (global_ds_plane >= 0 && global_ds_plane < 7) {
                        ds_plane_hits[global_ds_plane]++;
                    }
                }
            }
        }
        for (int p = 0; p < 7; ++p) {
            if (ds_plane_hits[p] > m.max_ds_plane_hits) {
                m.max_ds_plane_hits = ds_plane_hits[p];
            }
        }

        // DS occupancy ratios
        int ds_all_min = ds_plane_hits[0];
        int ds_all_max = ds_plane_hits[0];
        for (int p = 1; p < 7; ++p) {
            if (ds_plane_hits[p] < ds_all_min) ds_all_min = ds_plane_hits[p];
            if (ds_plane_hits[p] > ds_all_max) ds_all_max = ds_plane_hits[p];
        }
        m.ds_min_max_ratio_all = safe_ratio(ds_all_min, ds_all_max);

        int ds_h_min = ds_h_hits[0];
        int ds_h_max = ds_h_hits[0];
        for (int p = 1; p < 3; ++p) {
            if (ds_h_hits[p] < ds_h_min) ds_h_min = ds_h_hits[p];
            if (ds_h_hits[p] > ds_h_max) ds_h_max = ds_h_hits[p];
        }
        m.ds_min_max_ratio_h = safe_ratio(ds_h_min, ds_h_max);

        int ds_v_min = ds_v_hits[0];
        int ds_v_max = ds_v_hits[0];
        for (int p = 1; p < 4; ++p) {
            if (ds_v_hits[p] < ds_v_min) ds_v_min = ds_v_hits[p];
            if (ds_v_hits[p] > ds_v_max) ds_v_max = ds_v_hits[p];
        }
        m.ds_min_max_ratio_v = safe_ratio(ds_v_min, ds_v_max);

        // US plane QDC min/max ratio
        double qdc_min = us_plane_qdc[0];
        double qdc_max = us_plane_qdc[0];
        for (int p = 1; p < 5; ++p) {
            if (us_plane_qdc[p] < qdc_min) qdc_min = us_plane_qdc[p];
            if (us_plane_qdc[p] > qdc_max) qdc_max = us_plane_qdc[p];
        }
        m.us_qdc_min_max_ratio = safe_ratio(qdc_min, qdc_max);
        m.us_qdc_plane4_over_plane0 = (us_plane_qdc[0] > 0.0) ? (us_plane_qdc[4] / us_plane_qdc[0]) : 0.0;

        return m;
    }

    inline Metrics compute(const TClonesArray& scifiHits, const TClonesArray& mufiHits) {
        TClonesArray emptyArr("sndRecoTrack", 0);
        return compute(scifiHits, mufiHits, emptyArr);
    }
}

#endif // HIT_METRICS_HELPER_H
""")


def resolve_file_list(pattern: str) -> list[str]:
    """Resolve file pattern or wildcard path."""
    if "*" in pattern or "?" in pattern or "[" in pattern:
        files = sorted(glob.glob(pattern))
    elif os.path.isfile(pattern):
        files = [pattern]
    elif os.path.isdir(pattern):
        files = sorted(glob.glob(os.path.join(pattern, "*.root")))
    else:
        files = sorted(glob.glob(pattern))
    return [f for f in files if os.path.exists(f)]


def build_dataframe_with_friend(files: list[str], tree_name: str, friend_name: str = "mctruth"):
    """Construct a DataManager backed by TChain with friend mctruth tree."""
    ch = ROOT.TChain(tree_name)
    ch_mc = ROOT.TChain(friend_name) if friend_name else None

    for f in files:
        ch.Add(f)
        if ch_mc:
            ch_mc.Add(f)

    if ch_mc:
        ch.AddFriend(ch_mc)

    dm = DataManager(source=ch)
    return dm, dm.get_rdataframe()


def define_hit_metrics(df):
    """Defines hit metric columns on RDataFrame."""
    cols = [str(c) for c in df.GetColumnNames()]
    if "Reco_MuonTracks" in cols:
        df_defined = df.Define("hit_metrics", "HitMetrics::compute(Digi_ScifiHits, Digi_MuFilterHits, Reco_MuonTracks)")
    else:
        df_defined = df.Define("hit_metrics", "HitMetrics::compute(Digi_ScifiHits, Digi_MuFilterHits)")

    return (
        df_defined
          .Define("n_scifi_hits", "hit_metrics.n_scifi_hits")
          .Define("n_ds_hits", "hit_metrics.n_ds_hits")
          .Define("max_scifi_plane_hits", "hit_metrics.max_scifi_plane_hits")
          .Define("us_total_qdc", "hit_metrics.us_total_qdc")
          .Define("max_ds_plane_hits", "hit_metrics.max_ds_plane_hits")
          .Define("n_ds_h_hits", "hit_metrics.n_ds_h_hits")
          .Define("n_ds_v_hits", "hit_metrics.n_ds_v_hits")
          .Define("ds_min_max_ratio_all", "hit_metrics.ds_min_max_ratio_all")
          .Define("ds_min_max_ratio_h", "hit_metrics.ds_min_max_ratio_h")
          .Define("ds_min_max_ratio_v", "hit_metrics.ds_min_max_ratio_v")
          .Define("scifi_min_max_ratio_all", "hit_metrics.scifi_min_max_ratio_all")
          .Define("scifi_min_max_ratio_h", "hit_metrics.scifi_min_max_ratio_h")
          .Define("scifi_min_max_ratio_v", "hit_metrics.scifi_min_max_ratio_v")
          .Define("us_qdc_min_max_ratio", "hit_metrics.us_qdc_min_max_ratio")
          .Define("us_qdc_plane4_over_plane0", "hit_metrics.us_qdc_plane4_over_plane0")
          .Define("n_scifi_v_hits_pm2cm", "hit_metrics.n_scifi_v_hits_pm2cm")
          .Define("n_scifi_h_hits_pm2cm", "hit_metrics.n_scifi_h_hits_pm2cm")
          .Define("n_scifi_all_hits_pm2cm", "hit_metrics.n_scifi_all_hits_pm2cm")
          .Define("scifi_trk_iso_all", "hit_metrics.scifi_trk_iso_all")
          .Define("scifi_trk_iso_h", "hit_metrics.scifi_trk_iso_h")
          .Define("scifi_trk_iso_v", "hit_metrics.scifi_trk_iso_v")
    )


def book_category_histograms(df_node, hist_configs, prefix, weight_col=None):
    """Book 1D histograms from configuration on an RNode."""
    booked = {}
    for h_cfg in hist_configs:
        col = h_cfg["column"]
        name = f"h_{prefix}_{h_cfg['name']}"
        title = f"{prefix}: {h_cfg['title']}"
        bins = int(h_cfg["bins"])
        xmin = float(h_cfg["xmin"])
        xmax = float(h_cfg["xmax"])

        model = ROOT.RDF.TH1DModel(name, title, bins, xmin, xmax)
        if weight_col and weight_col in df_node.GetColumnNames():
            booked[name] = df_node.Histo1D(model, col, weight_col)
        else:
            booked[name] = df_node.Histo1D(model, col)
    return booked


def get_2d_pairs(hist_configs):
    """Generate all distinct variable pairs (x, y) for 2D correlation histograms."""
    pairs = []
    n = len(hist_configs)
    for i in range(n):
        for j in range(i + 1, n):
            cfg_x = hist_configs[i]
            cfg_y = hist_configs[j]
            # Ensure n_scifi_v_hits_pm2cm vs n_ds_v_hits puts n_scifi_v_hits_pm2cm on x-axis
            if cfg_y["name"] == "n_scifi_v_hits_pm2cm" and "ds" in cfg_x["name"]:
                cfg_x, cfg_y = cfg_y, cfg_x
            pairs.append((cfg_x, cfg_y))
    return pairs


def book_category_histograms_2d(df_node, pairs_2d, prefix, default_bins=35, weight_col=None):
    """Book 2D histograms from variable pairs on an RNode."""
    booked = {}
    for cfg_x, cfg_y in pairs_2d:
        col_x = cfg_x["column"]
        col_y = cfg_y["column"]
        var_x = cfg_x["name"]
        var_y = cfg_y["name"]
        pair_key = (var_x, var_y)
        name = f"h2_{prefix}_{var_x}_vs_{var_y}"

        # Extract axis labels from titles (format: "Title;X-axis;Y-axis")
        title_x = cfg_x["title"].split(";")[1] if ";" in cfg_x["title"] else var_x
        title_y = cfg_y["title"].split(";")[1] if ";" in cfg_y["title"] else var_y
        full_title = f"{prefix}: {title_y} vs {title_x};{title_x};{title_y}"

        bins_x = int(cfg_x.get("bins_2d", default_bins))
        xmin = float(cfg_x["xmin"])
        xmax = float(cfg_x["xmax"])

        bins_y = int(cfg_y.get("bins_2d", default_bins))
        ymin = float(cfg_y["xmin"])
        ymax = float(cfg_y["xmax"])

        model = ROOT.RDF.TH2DModel(name, full_title, bins_x, xmin, xmax, bins_y, ymin, ymax)
        if weight_col and weight_col in df_node.GetColumnNames():
            booked[pair_key] = df_node.Histo2D(model, col_x, col_y, weight_col)
        else:
            booked[pair_key] = df_node.Histo2D(model, col_x, col_y)
    return booked


def main():
    parser = argparse.ArgumentParser(
        description="Generate SciFi and MuFilter hit occupancy histograms using DataManager and RDataFrame."
    )
    parser.add_argument("-i", "--input-genie", required=True,
                        help="Input GENIE files or pattern (e.g. '/path/to/*/sndLHC.Genie-TGeant4*.root')")
    parser.add_argument("--input-pmu", default=None,
                        help="Optional input pmu2022 muon simulation files or pattern")
    parser.add_argument("--input-data", default=None,
                        help="Optional input collision data files or pattern")
    parser.add_argument("-o", "--output", default=None,
                        help="Output ROOT file (default from config)")
    parser.add_argument("-c", "--config", default=os.path.join(_repo_root, "config/hists_nhits_config.yaml"),
                        help="Path to YAML configuration file")
    parser.add_argument("-j", "--threads", type=int, default=None,
                        help="Number of worker threads (overrides config)")
    parser.add_argument("-n", "--max-events", type=int, default=None,
                        help="Maximum events to process per sample")
    parser.add_argument("-g", "--geo", default=None,
                        help="Path to geometry file (auto-detected if omitted)")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f_cfg:
        config = yaml.safe_load(f_cfg)

    threads = args.threads or config.get("performance", {}).get("threads", 1)
    if args.max_events:
        threads = 1  # Range is incompatible with ImplicitMT in ROOT
    if threads > 1:
        ROOT.EnableImplicitMT(threads)
        print(f"Enabled ROOT ImplicitMT with {threads} threads.")

    output_file = args.output or config.get("output", {}).get("default_file", "hists_nhits_distributions.root")
    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    declare_cpp_hit_metrics()

    # Determine and initialize geometry
    geo_path = args.geo
    if not geo_path:
        for c in [
            "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/geofile_full.Ntuple-TGeant4.root",
            "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root",
        ]:
            if os.path.exists(c):
                geo_path = c
                break

    if geo_path and os.path.exists(geo_path):
        import SndlhcGeo
        print(f"Initializing geometry from: {geo_path}")
        _geo_interface = SndlhcGeo.GeoInterface(geo_path)
        ROOT.HitMetrics.init_scifi_geometry()
        print("Initialized SciFi channel geometry lookup table.")
    else:
        print("Warning: Geometry file not found. SciFi track isolation window calculations will default to 0.")

    hist_cfgs = config.get("histograms", [])
    config_2d = config.get("histograms_2d", {})
    enable_2d = config_2d.get("enable_all_pairs", True)
    default_bins_2d = config_2d.get("default_bins", 35)
    pairs_2d = get_2d_pairs(hist_cfgs) if enable_2d else []

    booked_all = {}
    booked_all_2d = {}

    # -------------------------------------------------------------------------
    # 1. Process GENIE Sample
    # -------------------------------------------------------------------------
    genie_files = resolve_file_list(args.input_genie)
    if not genie_files:
        raise FileNotFoundError(f"No files matched GENIE pattern: {args.input_genie}")
    print(f"\nProcessing GENIE: found {len(genie_files)} files.")

    dm_genie, df_genie = build_dataframe_with_friend(genie_files, "cbmsim", "mctruth")
    if args.max_events:
        df_genie = df_genie.Range(args.max_events)

    df_genie = define_hit_metrics(df_genie)

    has_truth = "mctruth.is_direct_charm_decay" in df_genie.GetColumnNames()
    has_weight = "mctruth.mc_weight" in df_genie.GetColumnNames()
    weight_col = "mctruth.mc_weight" if has_weight else None

    if has_truth:
        df_genie_signal = df_genie.Filter("mctruth.is_direct_charm_decay == 1", "GENIE Direct Signal")
        df_genie_indirect = df_genie.Filter("mctruth.is_direct_charm_decay == 0 && mctruth.is_indirect_charm_decay == 1", "GENIE Indirect Signal")
        df_genie_bkg = df_genie.Filter("mctruth.is_direct_charm_decay == 0 && mctruth.is_indirect_charm_decay == 0", "GENIE Inclusive Background")

        booked_all[("genie/signal", "signal")] = book_category_histograms(df_genie_signal, hist_cfgs, "signal", weight_col)
        booked_all[("genie/indirect_fake", "indirect")] = book_category_histograms(df_genie_indirect, hist_cfgs, "indirect", weight_col)
        booked_all[("genie/background", "genie_bkg")] = book_category_histograms(df_genie_bkg, hist_cfgs, "genie_bkg", weight_col)

        if enable_2d:
            booked_all_2d[("genie/signal", "signal")] = book_category_histograms_2d(df_genie_signal, pairs_2d, "signal", default_bins_2d, weight_col)
            booked_all_2d[("genie/indirect_fake", "indirect")] = book_category_histograms_2d(df_genie_indirect, pairs_2d, "indirect", default_bins_2d, weight_col)
            booked_all_2d[("genie/background", "genie_bkg")] = book_category_histograms_2d(df_genie_bkg, pairs_2d, "genie_bkg", default_bins_2d, weight_col)
    else:
        print("Warning: 'mctruth' branch not found in GENIE sample; booking as inclusive.")
        booked_all[("genie/inclusive", "genie_inclusive")] = book_category_histograms(df_genie, hist_cfgs, "genie_inclusive", weight_col)
        if enable_2d:
            booked_all_2d[("genie/inclusive", "genie_inclusive")] = book_category_histograms_2d(df_genie, pairs_2d, "genie_inclusive", default_bins_2d, weight_col)

    # -------------------------------------------------------------------------
    # 2. Process PMU Sample (Optional)
    # -------------------------------------------------------------------------
    if args.input_pmu:
        pmu_files = resolve_file_list(args.input_pmu)
        if pmu_files:
            print(f"\nProcessing PMU2022: found {len(pmu_files)} files.")
            dm_pmu, df_pmu = build_dataframe_with_friend(pmu_files, "cbmsim", "mctruth")
            if args.max_events:
                df_pmu = df_pmu.Range(args.max_events)
            df_pmu = define_hit_metrics(df_pmu)

            pmu_weight_col = "mctruth.mc_weight" if "mctruth.mc_weight" in df_pmu.GetColumnNames() else None
            booked_all[("pmu2022", "pmu")] = book_category_histograms(df_pmu, hist_cfgs, "pmu", pmu_weight_col)
            if enable_2d:
                booked_all_2d[("pmu2022", "pmu")] = book_category_histograms_2d(df_pmu, pairs_2d, "pmu", default_bins_2d, pmu_weight_col)

    # -------------------------------------------------------------------------
    # 3. Process Collision Data Sample (Optional)
    # -------------------------------------------------------------------------
    if args.input_data:
        data_files = resolve_file_list(args.input_data)
        if data_files:
            print(f"\nProcessing Collision Data: found {len(data_files)} files.")
            # Data can be under cbmsim or rawConv
            first_f = ROOT.TFile.Open(data_files[0])
            tree_name = "rawConv" if (first_f and first_f.Get("rawConv")) else "cbmsim"
            if first_f:
                first_f.Close()

            dm_data, df_data = build_dataframe_with_friend(data_files, tree_name, "")
            if args.max_events:
                df_data = df_data.Range(args.max_events)
            df_data = define_hit_metrics(df_data)

            booked_all[("data", "data")] = book_category_histograms(df_data, hist_cfgs, "data")
            if enable_2d:
                booked_all_2d[("data", "data")] = book_category_histograms_2d(df_data, pairs_2d, "data", default_bins_2d)

    # -------------------------------------------------------------------------
    # 4. Trigger Event Loops & Write Output ROOT File with Structured TDirectories
    # -------------------------------------------------------------------------
    print("\nTriggering event loop evaluations...")
    # First extract all histograms into memory before opening/writing to the destination file
    materialized = {}
    for key, hist_dict in booked_all.items():
        materialized[key] = {name: h_res.GetValue() for name, h_res in hist_dict.items()}

    materialized_2d = {}
    for key, hist2d_dict in booked_all_2d.items():
        materialized_2d[key] = {pair: h_res.GetValue() for pair, h_res in hist2d_dict.items()}

    # Rebinning factors for signal histograms to handle low statistics
    signal_rebin_factors = {
        "n_scifi_hits": 4,
        "max_scifi_plane_hits": 2,
        "us_total_qdc": 2,
    }
    if ("genie/signal", "signal") in materialized:
        sig_dict = materialized[("genie/signal", "signal")]
        for h_name, h_obj in sig_dict.items():
            for var_key, factor in signal_rebin_factors.items():
                if h_name.endswith(f"_{var_key}"):
                    h_obj.Rebin(factor)
                    break

    print(f"\nWriting histograms to: {output_file}")
    out_file = ROOT.TFile.Open(output_file, "RECREATE")

    for (dir_path, _), hist_dict in materialized.items():
        # Create nested TDirectory structure
        current_dir = out_file
        for part in dir_path.split("/"):
            sub = current_dir.GetDirectory(part)
            if not sub:
                sub = current_dir.mkdir(part)
            current_dir = sub

        current_dir.cd()
        for name, h in hist_dict.items():
            current_dir.WriteTObject(h, name, "Overwrite")

    # -------------------------------------------------------------------------
    # 5. Build Normalized Overlay Comparison TCanvases
    # -------------------------------------------------------------------------
    print("\nGenerating normalized overlay comparison TCanvases...")
    d_canvases = out_file.GetDirectory("comparisons")
    if not d_canvases:
        d_canvases = out_file.mkdir("comparisons")

    # Categories to overlay: (internal_key, legend_label, line_color, line_width, fill_color, fill_style, is_marker)
    # Signal: blue, linewidth=4, no fill style
    # GENIE bkg: red/orange, linewidth=2, brighter subtle hatched fill styles
    # PMU: dark gray marker with error bars (no caps)
    overlay_specs = [
        (("genie/background", "genie_bkg"), "GENIE Other Bkg", ROOT.kRed + 1, 2, ROOT.kRed - 9, 3004, False),
        (("genie/indirect_fake", "indirect"), "GENIE Indirect Fake", ROOT.kOrange + 1, 2, ROOT.kOrange - 3, 3005, False),
        (("genie/inclusive", "genie_inclusive"), "GENIE Inclusive", ROOT.kRed + 1, 2, ROOT.kRed - 9, 3004, False),
        (("genie/signal", "signal"), "GENIE Direct Signal", ROOT.kBlue, 4, ROOT.kBlue, 0, False),
        (("pmu2022", "pmu"), "pmu2022 Muon Bkg", ROOT.kGray + 2, 2, ROOT.kGray + 2, 0, True),
    ]

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetEndErrorSize(0)

    for h_cfg in hist_cfgs:
        var_name = h_cfg["name"]
        canvas_name = f"c_{var_name}_normalized"
        canvas_title = f"Normalized Comparison: {h_cfg['title'].split(';')[0]}"

        c = ROOT.TCanvas(canvas_name, canvas_title, 900, 700)
        c.SetLeftMargin(0.12)
        c.SetRightMargin(0.06)
        c.SetTopMargin(0.08)
        c.SetBottomMargin(0.12)

        leg = ROOT.TLegend(0.55, 0.68, 0.90, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.035)

        h_clones = []
        max_val = 0.0

        for cat_key, label, lcolor, lwidth, fcolor, fstyle, is_marker in overlay_specs:
            if cat_key not in materialized:
                continue
            cat_dict = materialized[cat_key]
            # Match the histogram for this variable
            h_orig = None
            for h_name, h_obj in cat_dict.items():
                if h_name.endswith(f"_{var_name}"):
                    h_orig = h_obj
                    break
            if not h_orig or h_orig.GetEntries() == 0:
                continue

            h_norm = h_orig.Clone(f"{h_orig.GetName()}_norm")
            integral = h_norm.Integral()
            if integral > 0:
                h_norm.Scale(1.0 / integral, "width")

            h_norm.SetLineColor(lcolor)
            h_norm.SetLineWidth(lwidth)
            h_norm.SetLineStyle(1)

            if is_marker:
                # Marker with Poisson/binomial error bars without caps
                h_norm.SetMarkerStyle(20)
                h_norm.SetMarkerSize(0.8)
                h_norm.SetMarkerColor(lcolor)
                h_norm.SetFillStyle(0)
            else:
                h_norm.SetFillColor(fcolor)
                h_norm.SetFillStyle(fstyle)

            h_norm.GetYaxis().SetTitle("Probability density")
            h_norm.GetYaxis().SetTitleOffset(1.3)
            h_norm.GetXaxis().SetTitle(h_orig.GetXaxis().GetTitle())

            curr_max = h_norm.GetMaximum()
            if curr_max > max_val:
                max_val = curr_max

            h_clones.append((h_norm, label, is_marker, fstyle))

        if h_clones:
            for idx, (h_norm, label, is_marker, fstyle) in enumerate(h_clones):
                h_norm.SetMaximum(max_val * 1.30)
                h_norm.SetMinimum(0.0)
                if is_marker:
                    draw_opt = "E1 P" if idx == 0 else "E1 P SAME"
                    leg.AddEntry(h_norm, label, "PL")
                elif fstyle == 0:
                    draw_opt = "HIST" if idx == 0 else "HIST SAME"
                    leg.AddEntry(h_norm, label, "L")
                else:
                    draw_opt = "HIST" if idx == 0 else "HIST SAME"
                    leg.AddEntry(h_norm, label, "F")
                h_norm.Draw(draw_opt)

            leg.Draw()
            d_canvases.cd()
            c.Write()

    # -------------------------------------------------------------------------
    # 6. Write 2D Histograms under <category>/2d/
    # -------------------------------------------------------------------------
    if materialized_2d:
        print("\nWriting 2D correlation histograms...")
        for (dir_path, prefix), hist2d_dict in materialized_2d.items():
            current_dir = out_file
            for part in dir_path.split("/"):
                sub = current_dir.GetDirectory(part)
                if not sub:
                    sub = current_dir.mkdir(part)
                current_dir = sub
            d_2d = current_dir.GetDirectory("2d")
            if not d_2d:
                d_2d = current_dir.mkdir("2d")
            d_2d.cd()
            for (var_x, var_y), h2 in hist2d_dict.items():
                h2_name = f"h2_{prefix}_{var_x}_vs_{var_y}"
                d_2d.WriteTObject(h2, h2_name, "Overwrite")

    # -------------------------------------------------------------------------
    # 7. Build 2D Comparison TCanvases & Correlation Matrices
    # -------------------------------------------------------------------------
    if materialized_2d and pairs_2d:
        print("\nGenerating 2D comparison TCanvases...")
        d_canvases_2d = d_canvases.GetDirectory("2d")
        if not d_canvases_2d:
            d_canvases_2d = d_canvases.mkdir("2d")

        ROOT.gStyle.SetPalette(ROOT.kViridis)

        pads_spec = [
            (("genie/signal", "signal"), "GENIE Direct Signal", 1),
            (("genie/background", "genie_bkg"), "GENIE Other Bkg", 2),
            (("genie/indirect_fake", "indirect"), "GENIE Indirect Fake", 3),
            (("pmu2022", "pmu"), "pmu2022 Muon Bkg", 4),
        ]

        for cfg_x, cfg_y in pairs_2d:
            var_x = cfg_x["name"]
            var_y = cfg_y["name"]
            pair_key = (var_x, var_y)
            c2 = ROOT.TCanvas(f"c2d_{var_x}_vs_{var_y}", f"2D Comparison: {var_y} vs {var_x}", 1200, 950)
            c2.Divide(2, 2)

            has_any = False
            for cat_key, cat_title, pad_idx in pads_spec:
                if cat_key not in materialized_2d:
                    continue
                h2_orig = materialized_2d[cat_key].get(pair_key)
                if not h2_orig or h2_orig.GetEntries() == 0:
                    continue

                has_any = True
                pad = c2.cd(pad_idx)
                pad.SetLeftMargin(0.12)
                pad.SetRightMargin(0.14)
                pad.SetBottomMargin(0.12)
                pad.SetTopMargin(0.10)

                h2_draw = h2_orig.Clone(f"{h2_orig.GetName()}_pad")
                r = h2_draw.GetCorrelationFactor()
                h2_draw.SetTitle(f"{cat_title} (r = {r:+.2f})")
                h2_draw.Draw("COLZ")

            if has_any:
                d_canvases_2d.cd()
                c2.Write()
            ROOT.gROOT.GetListOfCanvases().Remove(c2)
            c2.Close()

        # Build summary correlation matrices
        print("\nGenerating correlation matrices...")
        num_vars = len(hist_cfgs)
        var_names = [c["name"] for c in hist_cfgs]

        for cat_key, cat_name in [
            (("genie/signal", "signal"), "signal"),
            (("genie/background", "genie_bkg"), "background"),
            (("pmu2022", "pmu"), "pmu2022"),
        ]:
            if cat_key not in materialized_2d:
                continue
            h_corr = ROOT.TH2D(f"h2_corr_matrix_{cat_name}", f"Correlation Matrix: {cat_name.upper()};Variable X;Variable Y",
                               num_vars, 0, num_vars, num_vars, 0, num_vars)
            for idx_v, vname in enumerate(var_names):
                h_corr.GetXaxis().SetBinLabel(idx_v + 1, vname)
                h_corr.GetYaxis().SetBinLabel(idx_v + 1, vname)

            for i in range(num_vars):
                h_corr.SetBinContent(i + 1, i + 1, 1.0)

            cat_dict = materialized_2d[cat_key]
            for cfg_x, cfg_y in pairs_2d:
                pair = (cfg_x["name"], cfg_y["name"])
                if pair in cat_dict:
                    h2 = cat_dict[pair]
                    if h2.GetEntries() > 0:
                        r = h2.GetCorrelationFactor()
                        ix = var_names.index(cfg_x["name"])
                        iy = var_names.index(cfg_y["name"])
                        h_corr.SetBinContent(ix + 1, iy + 1, r)
                        h_corr.SetBinContent(iy + 1, ix + 1, r)

            c_corr = ROOT.TCanvas(f"c_corr_matrix_{cat_name}", f"Correlation Matrix - {cat_name}", 1300, 1100)
            c_corr.SetLeftMargin(0.22)
            c_corr.SetBottomMargin(0.22)
            c_corr.SetRightMargin(0.14)
            c_corr.SetTopMargin(0.08)
            h_corr.SetMinimum(-1.0)
            h_corr.SetMaximum(1.0)
            h_corr.GetXaxis().LabelsOption("v")
            h_corr.Draw("COLZ")
            d_canvases.cd()
            c_corr.Write()
            ROOT.gROOT.GetListOfCanvases().Remove(c_corr)
            c_corr.Close()

        # Correlation Difference Matrix (Signal - Background)
        if ("genie/signal", "signal") in materialized_2d and ("genie/background", "genie_bkg") in materialized_2d:
            h_diff = ROOT.TH2D("h2_corr_matrix_diff_sig_minus_bkg", "Correlation Matrix Difference (Signal - Background);Variable X;Variable Y",
                               num_vars, 0, num_vars, num_vars, 0, num_vars)
            for idx_v, vname in enumerate(var_names):
                h_diff.GetXaxis().SetBinLabel(idx_v + 1, vname)
                h_diff.GetYaxis().SetBinLabel(idx_v + 1, vname)

            sig_dict = materialized_2d[("genie/signal", "signal")]
            bkg_dict = materialized_2d[("genie/background", "genie_bkg")]
            for cfg_x, cfg_y in pairs_2d:
                pair = (cfg_x["name"], cfg_y["name"])
                if pair in sig_dict and pair in bkg_dict:
                    h2_sig = sig_dict[pair]
                    h2_bkg = bkg_dict[pair]
                    if h2_sig.GetEntries() > 0 and h2_bkg.GetEntries() > 0:
                        delta_r = h2_sig.GetCorrelationFactor() - h2_bkg.GetCorrelationFactor()
                        ix = var_names.index(cfg_x["name"])
                        iy = var_names.index(cfg_y["name"])
                        h_diff.SetBinContent(ix + 1, iy + 1, delta_r)
                        h_diff.SetBinContent(iy + 1, ix + 1, delta_r)

            c_diff = ROOT.TCanvas("c_corr_matrix_diff_sig_minus_bkg", "Correlation Difference (Signal - Background)", 1300, 1100)
            c_diff.SetLeftMargin(0.22)
            c_diff.SetBottomMargin(0.22)
            c_diff.SetRightMargin(0.14)
            c_diff.SetTopMargin(0.08)
            h_diff.SetMinimum(-2.0)
            h_diff.SetMaximum(2.0)
            h_diff.GetXaxis().LabelsOption("v")
            h_diff.Draw("COLZ")
            d_canvases.cd()
            c_diff.Write()
            ROOT.gROOT.GetListOfCanvases().Remove(c_diff)
            c_diff.Close()

    out_file.Close()
    print("Histogram generation complete!")


if __name__ == "__main__":
    main()
