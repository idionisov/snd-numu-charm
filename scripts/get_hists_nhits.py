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
#include <algorithm>
#include <cmath>
#include "Scifi.h"
#include "MuFilter.h"
#include "sndScifiHit.h"
#include "MuFilterHit.h"

namespace HitMetrics {
    struct Metrics {
        int n_scifi_hits;
        int n_ds_hits;
        int max_scifi_plane_hits;
        double us_total_qdc;
        int max_ds_plane_hits;
    };

    inline Metrics compute(const TClonesArray& scifiHits, const TClonesArray& mufiHits) {
        Metrics m{0, 0, 0, 0.0, 0};

        // 1. SciFi Hits: 5 stations * 2 orientations (vert=0, horiz=1) -> 10 planes
        int scifi_plane_hits[10] = {0};
        int n_sf = scifiHits.GetEntries();
        for (int i = 0; i < n_sf; ++i) {
            auto* h = static_cast<sndScifiHit*>(scifiHits.At(i));
            if (!h || !h->isValid()) continue;
            m.n_scifi_hits++;
            int st = h->GetStation();
            int o = h->isVertical() ? 0 : 1;
            if (st >= 1 && st <= 5) {
                int p_idx = (st - 1) * 2 + o;
                scifi_plane_hits[p_idx]++;
            }
        }
        for (int p = 0; p < 10; ++p) {
            if (scifi_plane_hits[p] > m.max_scifi_plane_hits) {
                m.max_scifi_plane_hits = scifi_plane_hits[p];
            }
        }

        // 2. MuFilter Hits: US (system 2) and DS (system 3)
        // DS has stations 1-3 with 2 planes each (H/V) and station 4 with 1 plane (H) -> 7 planes
        int ds_plane_hits[7] = {0};
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
                }
            } else if (sys == 3) {
                // Downstream
                m.n_ds_hits++;
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

        return m;
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

    hist_cfgs = config.get("histograms", [])
    booked_all = {}

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

    df_genie = df_genie.Define("hit_metrics", "HitMetrics::compute(Digi_ScifiHits, Digi_MuFilterHits)") \
                       .Define("n_scifi_hits", "hit_metrics.n_scifi_hits") \
                       .Define("n_ds_hits", "hit_metrics.n_ds_hits") \
                       .Define("max_scifi_plane_hits", "hit_metrics.max_scifi_plane_hits") \
                       .Define("us_total_qdc", "hit_metrics.us_total_qdc") \
                       .Define("max_ds_plane_hits", "hit_metrics.max_ds_plane_hits")

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
    else:
        print("Warning: 'mctruth' branch not found in GENIE sample; booking as inclusive.")
        booked_all[("genie/inclusive", "genie_inclusive")] = book_category_histograms(df_genie, hist_cfgs, "genie_inclusive", weight_col)

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
            df_pmu = df_pmu.Define("hit_metrics", "HitMetrics::compute(Digi_ScifiHits, Digi_MuFilterHits)") \
                           .Define("n_scifi_hits", "hit_metrics.n_scifi_hits") \
                           .Define("n_ds_hits", "hit_metrics.n_ds_hits") \
                           .Define("max_scifi_plane_hits", "hit_metrics.max_scifi_plane_hits") \
                           .Define("us_total_qdc", "hit_metrics.us_total_qdc") \
                           .Define("max_ds_plane_hits", "hit_metrics.max_ds_plane_hits")

            pmu_weight_col = "mctruth.mc_weight" if "mctruth.mc_weight" in df_pmu.GetColumnNames() else None
            booked_all[("pmu2022", "pmu")] = book_category_histograms(df_pmu, hist_cfgs, "pmu", pmu_weight_col)

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
            df_data = df_data.Define("hit_metrics", "HitMetrics::compute(Digi_ScifiHits, Digi_MuFilterHits)") \
                             .Define("n_scifi_hits", "hit_metrics.n_scifi_hits") \
                             .Define("n_ds_hits", "hit_metrics.n_ds_hits") \
                             .Define("max_scifi_plane_hits", "hit_metrics.max_scifi_plane_hits") \
                             .Define("us_total_qdc", "hit_metrics.us_total_qdc") \
                             .Define("max_ds_plane_hits", "hit_metrics.max_ds_plane_hits")

            booked_all[("data", "data")] = book_category_histograms(df_data, hist_cfgs, "data")

    # -------------------------------------------------------------------------
    # 4. Trigger Event Loops & Write Output ROOT File with Structured TDirectories
    # -------------------------------------------------------------------------
    print("\nTriggering event loop evaluations...")
    # First extract all histograms into memory before opening/writing to the destination file
    materialized = {}
    for key, hist_dict in booked_all.items():
        materialized[key] = {name: h_res.GetValue() for name, h_res in hist_dict.items()}

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

    out_file.Close()
    print("Histogram generation complete!")


if __name__ == "__main__":
    main()
