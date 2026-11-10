#!/usr/bin/env python3
"""
scripts/qdc/qdc_distr.py
------------------------
Extract and compare comprehensive QDC and hit occupancy distributions across:
  1. Collision Data (--input-data)
  2. PMU Muon Background Simulation (--input-pmu)
  3. GENIE Neutrino Simulation (--input-genie)
     - Inclusive & Background
     - Charm Dimuon Signal (numu -> charm -> 2nd muon) with wider bins

Features:
  - System total, per plane (all planes filled together into one distribution),
    plane of max occupancy, and per hit QDC distributions for US, DS, and SciFi.
  - Total and per plane hit occupancy distributions for Veto, SciFi, US, and DS.
  - Veto delayed hits (> 300 ps after earliest SciFi hit).
  - Informative 2D correlation histograms for each category.
  - Hierarchical TDirectory organization in output ROOT file.
  - Superimposed normalized comparison TCanvases for all distributions
    (Data: markers with error bars, Sims: colored histograms, Signal: colored markers with error bars).
  - Both linear and log-Y scale comparison canvases.

Usage:
  python3 scripts/qdc/qdc_distr.py \
    --input-genie "/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root" \
    --input-pmu "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP_2MuTrks_truth.root" \
    --input-data "/eos/experiment/sndlhc/convertedData/physics/2023/run_006596/sndsw_raw-*.root" \
    --output "/path/to/output.root" \
    -j 8
"""

from __future__ import annotations

import os
import sys
import glob
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple

import ROOT

# Add project root to sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import load_trident_libraries

load_trident_libraries(_repo_root)


def declare_cpp_qdc_metrics():
    """Declare high-performance C++ helper to compute QDC and hit occupancy metrics."""
    ROOT.gInterpreter.Declare("""
#ifndef QDC_METRICS_HELPER_H
#define QDC_METRICS_HELPER_H

#include <TClonesArray.h>
#include <ROOT/RVec.hxx>
#include <vector>
#include <algorithm>
#include <cmath>
#include "sndScifiHit.h"
#include "MuFilterHit.h"

namespace QdcMetrics {

struct Metrics {
    // System total QDCs
    double us_total_qdc{0.0};
    double ds_total_qdc{0.0};
    double scifi_total_qdc{0.0};

    // System total hits
    int veto_total_hits{0};
    int veto_delayed_hits{0};
    int scifi_total_hits{0};
    int us_total_hits{0};
    int ds_total_hits{0};

    // Max plane QDCs (plane of max occupancy)
    double us_max_plane_qdc{0.0};
    double ds_max_plane_qdc{0.0};
    double scifi_max_plane_qdc{0.0};

    // Max plane hit counts
    int us_max_plane_hits{0};
    int ds_max_plane_hits{0};
    int scifi_max_plane_hits{0};

    // Per plane QDC (all planes in one collection)
    ROOT::RVec<double> us_plane_qdc;       // 5 planes
    ROOT::RVec<double> ds_plane_qdc;       // 7 planes
    ROOT::RVec<double> scifi_plane_qdc;    // 10 planes

    // Per plane hits (all planes in one collection)
    ROOT::RVec<int> us_plane_hits;         // 5 planes
    ROOT::RVec<int> ds_plane_hits;         // 7 planes
    ROOT::RVec<int> scifi_plane_hits;      // 10 planes

    // Per hit QDC
    ROOT::RVec<double> us_hit_qdc;
    ROOT::RVec<double> ds_hit_qdc;
    ROOT::RVec<double> scifi_hit_qdc;
};

inline Metrics compute(const TClonesArray& scifiHits, const TClonesArray& mufiHits, bool is_data = false) {
    Metrics m;
    m.us_plane_qdc.assign(5, 0.0);
    m.ds_plane_qdc.assign(7, 0.0);
    m.scifi_plane_qdc.assign(10, 0.0);

    m.us_plane_hits.assign(5, 0);
    m.ds_plane_hits.assign(7, 0);
    m.scifi_plane_hits.assign(10, 0);

    float min_scifi_t = 1e30f;
    bool has_scifi = false;

    // 1. Process SciFi Hits
    const int n_sf = scifiHits.GetEntriesFast();
    for (int i = 0; i < n_sf; ++i) {
        auto* h = static_cast<sndScifiHit*>(scifiHits.At(i));
        if (!h || !h->isValid()) continue;
        m.scifi_total_hits++;

        float t = h->GetTime();
        if (t < min_scifi_t) {
            min_scifi_t = t;
            has_scifi = true;
        }

        double qdc = static_cast<double>(h->GetSignal(0));
        if (qdc > 0.0) {
            m.scifi_total_qdc += qdc;
            m.scifi_hit_qdc.push_back(qdc);
        }

        int st = h->GetStation();
        bool is_vert = h->isVertical();
        int o = is_vert ? 0 : 1;
        if (st >= 1 && st <= 5) {
            int p_idx = (st - 1) * 2 + o;
            if (p_idx >= 0 && p_idx < 10) {
                m.scifi_plane_hits[p_idx]++;
                if (qdc > 0.0) {
                    m.scifi_plane_qdc[p_idx] += qdc;
                }
            }
        }
    }

    // SciFi max occupancy plane
    int max_sf_h = -1;
    int max_sf_p = 0;
    for (int p = 0; p < 10; ++p) {
        if (m.scifi_plane_hits[p] > max_sf_h) {
            max_sf_h = m.scifi_plane_hits[p];
            max_sf_p = p;
        }
    }
    m.scifi_max_plane_hits = (max_sf_h > 0) ? max_sf_h : 0;
    m.scifi_max_plane_qdc = m.scifi_plane_qdc[max_sf_p];

    // 2. Process MuFilter Hits
    const int n_mf = mufiHits.GetEntriesFast();
    const double tdc2ns = is_data ? 6.23768 : 1.0;

    for (int i = 0; i < n_mf; ++i) {
        auto* h = static_cast<MuFilterHit*>(mufiHits.At(i));
        if (!h || !h->isValid()) continue;
        int sys = h->GetSystem();
        int det_id = h->GetDetectorID();
        int plane_idx = (det_id % 10000) / 1000;
        bool is_vert = h->isVertical();
        int o_idx = is_vert ? 1 : 0;

        if (sys == 1) { // Veto
            m.veto_total_hits++;
            if (has_scifi) {
                float t0 = h->GetTime(0);
                float t1 = h->GetTime(1);
                float tv = 0.0f;
                if (is_vert) {
                    tv = (t0 > -900.0f) ? t0 : t1;
                } else {
                    if (t0 > -900.0f && t1 > -900.0f) tv = 0.5f * (t0 + t1);
                    else if (t0 > -900.0f) tv = t0;
                    else tv = t1;
                }
                if (tv > -900.0f) {
                    double dt = (tv - min_scifi_t) * tdc2ns;
                    if (dt > 0.3) { // > 300 ps (0.3 ns) latency relative to earliest SciFi
                        m.veto_delayed_hits++;
                    }
                }
            }
        } else if (sys == 2) { // Upstream (US)
            m.us_total_hits++;
            if (plane_idx >= 0 && plane_idx < 5) {
                m.us_plane_hits[plane_idx]++;
            }
            auto sigs = h->GetAllSignals();
            double hit_qdc = 0.0;
            for (const auto& kv : sigs) {
                hit_qdc += kv.second;
            }
            if (hit_qdc > 0.0) {
                m.us_total_qdc += hit_qdc;
                m.us_hit_qdc.push_back(hit_qdc);
                if (plane_idx >= 0 && plane_idx < 5) {
                    m.us_plane_qdc[plane_idx] += hit_qdc;
                }
            }
        } else if (sys == 3) { // Downstream (DS)
            m.ds_total_hits++;
            int global_ds_plane = (plane_idx < 3) ? (plane_idx * 2 + o_idx) : 6;
            if (global_ds_plane >= 0 && global_ds_plane < 7) {
                m.ds_plane_hits[global_ds_plane]++;
            }
            auto sigs = h->GetAllSignals();
            double hit_qdc = 0.0;
            for (const auto& kv : sigs) {
                hit_qdc += kv.second;
            }
            if (hit_qdc > 0.0) {
                m.ds_total_qdc += hit_qdc;
                m.ds_hit_qdc.push_back(hit_qdc);
                if (global_ds_plane >= 0 && global_ds_plane < 7) {
                    m.ds_plane_qdc[global_ds_plane] += hit_qdc;
                }
            }
        }
    }

    // US max occupancy plane
    int max_us_h = -1;
    int max_us_p = 0;
    for (int p = 0; p < 5; ++p) {
        if (m.us_plane_hits[p] > max_us_h) {
            max_us_h = m.us_plane_hits[p];
            max_us_p = p;
        }
    }
    m.us_max_plane_hits = (max_us_h > 0) ? max_us_h : 0;
    m.us_max_plane_qdc = m.us_plane_qdc[max_us_p];

    // DS max occupancy plane
    int max_ds_h = -1;
    int max_ds_p = 0;
    for (int p = 0; p < 7; ++p) {
        if (m.ds_plane_hits[p] > max_ds_h) {
            max_ds_h = m.ds_plane_hits[p];
            max_ds_p = p;
        }
    }
    m.ds_max_plane_hits = (max_ds_h > 0) ? max_ds_h : 0;
    m.ds_max_plane_qdc = m.ds_plane_qdc[max_ds_p];

    return m;
}

} // namespace QdcMetrics
#endif // QDC_METRICS_HELPER_H
""")


def resolve_file_list(pattern: str) -> List[str]:
    """Resolve file pattern, %s/%d partition template, comma-separated list, or wildcards."""
    if not pattern:
        return []
    if "%s" in pattern or "%d" in pattern:
        expanded = []
        for p in range(1, 1001):
            path = pattern % p
            if os.path.exists(path):
                expanded.append(path)
        return expanded
    if "," in pattern:
        files = []
        for part in pattern.split(","):
            files.extend(resolve_file_list(part.strip()))
        return sorted(list(set(files)))
    if "*" in pattern or "?" in pattern or "[" in pattern:
        files = sorted(glob.glob(pattern))
    elif os.path.isfile(pattern):
        files = [pattern]
    elif os.path.isdir(pattern):
        files = sorted(glob.glob(os.path.join(pattern, "*.root")))
    else:
        files = sorted(glob.glob(pattern))
    return [f for f in files if os.path.exists(f)]


def build_dataframe_with_friend(files: List[str], tree_name: str, friend_name: str = "mctruth"):
    """Construct a persistent TChain backing with friend mctruth tree, returning (ch, ch_mc, df)."""
    ch = ROOT.TChain(tree_name)
    ch_mc = ROOT.TChain(friend_name) if friend_name else None

    for f in files:
        ch.Add(f)
        if ch_mc:
            ch_mc.Add(f)

    if ch_mc:
        ch.AddFriend(ch_mc)

    df = ROOT.RDataFrame(ch)
    return ch, ch_mc, df


def define_qdc_metrics(df, is_data: bool = False):
    """Defines QDC and hit metric columns on RDataFrame."""
    is_data_str = "true" if is_data else "false"
    df_def = df.Define("qdc_metrics", f"QdcMetrics::compute(Digi_ScifiHits, Digi_MuFilterHits, {is_data_str})")

    return (
        df_def
        .Define("us_total_qdc", "qdc_metrics.us_total_qdc")
        .Define("ds_total_qdc", "qdc_metrics.ds_total_qdc")
        .Define("scifi_total_qdc", "qdc_metrics.scifi_total_qdc")
        .Define("veto_total_hits", "qdc_metrics.veto_total_hits")
        .Define("veto_delayed_hits", "qdc_metrics.veto_delayed_hits")
        .Define("scifi_total_hits", "qdc_metrics.scifi_total_hits")
        .Define("us_total_hits", "qdc_metrics.us_total_hits")
        .Define("ds_total_hits", "qdc_metrics.ds_total_hits")
        .Define("us_max_plane_qdc", "qdc_metrics.us_max_plane_qdc")
        .Define("ds_max_plane_qdc", "qdc_metrics.ds_max_plane_qdc")
        .Define("scifi_max_plane_qdc", "qdc_metrics.scifi_max_plane_qdc")
        .Define("us_max_plane_hits", "qdc_metrics.us_max_plane_hits")
        .Define("ds_max_plane_hits", "qdc_metrics.ds_max_plane_hits")
        .Define("scifi_max_plane_hits", "qdc_metrics.scifi_max_plane_hits")
        .Define("us_plane_qdc", "qdc_metrics.us_plane_qdc")
        .Define("ds_plane_qdc", "qdc_metrics.ds_plane_qdc")
        .Define("scifi_plane_qdc", "qdc_metrics.scifi_plane_qdc")
        .Define("us_plane_hits", "qdc_metrics.us_plane_hits")
        .Define("ds_plane_hits", "qdc_metrics.ds_plane_hits")
        .Define("scifi_plane_hits", "qdc_metrics.scifi_plane_hits")
        .Define("us_hit_qdc", "qdc_metrics.us_hit_qdc")
        .Define("ds_hit_qdc", "qdc_metrics.ds_hit_qdc")
        .Define("scifi_hit_qdc", "qdc_metrics.scifi_hit_qdc")
    )


# Histogram specifications
HIST_CONFIGS_1D = [
    # QDC distributions
    {
        "name": "us_total_qdc",
        "title": "Total Upstream (US) QDC;QDC_{tot}^{US} [a.u.];Events",
        "column": "us_total_qdc",
        "bins": 190,
        "xmin": 0.0,
        "xmax": 190000.0,
        "sig_rebin": 2,
    },
    {
        "name": "us_plane_qdc",
        "title": "US QDC per Plane (all planes);QDC_{plane}^{US} [a.u.];Entries",
        "column": "us_plane_qdc",
        "bins": 150,
        "xmin": 0.0,
        "xmax": 60000.0,
        "sig_rebin": 2,
    },
    {
        "name": "us_max_plane_qdc",
        "title": "US QDC in Plane of Max Occupancy;QDC_{max-occ plane}^{US} [a.u.];Events",
        "column": "us_max_plane_qdc",
        "bins": 150,
        "xmin": 0.0,
        "xmax": 60000.0,
        "sig_rebin": 2,
    },
    {
        "name": "us_hit_qdc",
        "title": "US QDC per Hit;QDC_{hit}^{US} [a.u.];Hits",
        "column": "us_hit_qdc",
        "bins": 150,
        "xmin": 0.0,
        "xmax": 5000.0,
        "sig_rebin": 2,
    },
    {
        "name": "ds_total_qdc",
        "title": "Total Downstream (DS) QDC;QDC_{tot}^{DS} [a.u.];Events",
        "column": "ds_total_qdc",
        "bins": 150,
        "xmin": 0.0,
        "xmax": 150000.0,
        "sig_rebin": 2,
    },
    {
        "name": "ds_plane_qdc",
        "title": "DS QDC per Plane (all planes);QDC_{plane}^{DS} [a.u.];Entries",
        "column": "ds_plane_qdc",
        "bins": 150,
        "xmin": 0.0,
        "xmax": 40000.0,
        "sig_rebin": 2,
    },
    {
        "name": "ds_max_plane_qdc",
        "title": "DS QDC in Plane of Max Occupancy;QDC_{max-occ plane}^{DS} [a.u.];Events",
        "column": "ds_max_plane_qdc",
        "bins": 150,
        "xmin": 0.0,
        "xmax": 40000.0,
        "sig_rebin": 2,
    },
    {
        "name": "ds_hit_qdc",
        "title": "DS QDC per Hit;QDC_{hit}^{DS} [a.u.];Hits",
        "column": "ds_hit_qdc",
        "bins": 150,
        "xmin": 0.0,
        "xmax": 3000.0,
        "sig_rebin": 2,
    },
    {
        "name": "scifi_total_qdc",
        "title": "Total SciFi QDC;QDC_{tot}^{SciFi} [a.u.];Events",
        "column": "scifi_total_qdc",
        "bins": 200,
        "xmin": 0.0,
        "xmax": 50000.0,
        "sig_rebin": 2,
    },
    {
        "name": "scifi_plane_qdc",
        "title": "SciFi QDC per Plane (all planes);QDC_{plane}^{SciFi} [a.u.];Entries",
        "column": "scifi_plane_qdc",
        "bins": 150,
        "xmin": 0.0,
        "xmax": 15000.0,
        "sig_rebin": 2,
    },
    {
        "name": "scifi_max_plane_qdc",
        "title": "SciFi QDC in Plane of Max Occupancy;QDC_{max-occ plane}^{SciFi} [a.u.];Events",
        "column": "scifi_max_plane_qdc",
        "bins": 150,
        "xmin": 0.0,
        "xmax": 15000.0,
        "sig_rebin": 2,
    },
    {
        "name": "scifi_hit_qdc",
        "title": "SciFi QDC per Hit;QDC_{hit}^{SciFi} [a.u.];Hits",
        "column": "scifi_hit_qdc",
        "bins": 100,
        "xmin": 0.0,
        "xmax": 100.0,
        "sig_rebin": 2,
    },

    # Number of hits distributions
    {
        "name": "veto_total_hits",
        "title": "Total Veto Hits;N_{hits}^{Veto};Events",
        "column": "veto_total_hits",
        "bins": 30,
        "xmin": 0.0,
        "xmax": 30.0,
        "sig_rebin": 1,
    },
    {
        "name": "veto_delayed_hits",
        "title": "Delayed Veto Hits (>300 ps after SciFi min);N_{hits}^{Veto delayed};Events",
        "column": "veto_delayed_hits",
        "bins": 30,
        "xmin": 0.0,
        "xmax": 30.0,
        "sig_rebin": 1,
    },
    {
        "name": "scifi_total_hits",
        "title": "Total SciFi Hits;N_{hits}^{SciFi};Events",
        "column": "scifi_total_hits",
        "bins": 200,
        "xmin": 0.0,
        "xmax": 2000.0,
        "sig_rebin": 4,
    },
    {
        "name": "scifi_plane_hits",
        "title": "SciFi Hits per Plane (all planes);N_{hits / plane}^{SciFi};Entries",
        "column": "scifi_plane_hits",
        "bins": 150,
        "xmin": 0.0,
        "xmax": 300.0,
        "sig_rebin": 2,
    },
    {
        "name": "scifi_max_plane_hits",
        "title": "SciFi Hits in Plane of Max Occupancy;N_{hits}^{max SciFi plane};Events",
        "column": "scifi_max_plane_hits",
        "bins": 150,
        "xmin": 0.0,
        "xmax": 400.0,
        "sig_rebin": 2,
    },
    {
        "name": "us_total_hits",
        "title": "Total US Hits;N_{hits}^{US};Events",
        "column": "us_total_hits",
        "bins": 100,
        "xmin": 0.0,
        "xmax": 100.0,
        "sig_rebin": 2,
    },
    {
        "name": "us_plane_hits",
        "title": "US Hits per Plane (all planes);N_{hits / plane}^{US};Entries",
        "column": "us_plane_hits",
        "bins": 50,
        "xmin": 0.0,
        "xmax": 50.0,
        "sig_rebin": 2,
    },
    {
        "name": "us_max_plane_hits",
        "title": "US Hits in Plane of Max Occupancy;N_{hits}^{max US plane};Events",
        "column": "us_max_plane_hits",
        "bins": 50,
        "xmin": 0.0,
        "xmax": 50.0,
        "sig_rebin": 2,
    },
    {
        "name": "ds_total_hits",
        "title": "Total DS Hits;N_{hits}^{DS};Events",
        "column": "ds_total_hits",
        "bins": 100,
        "xmin": 0.0,
        "xmax": 200.0,
        "sig_rebin": 2,
    },
    {
        "name": "ds_plane_hits",
        "title": "DS Hits per Plane (all planes);N_{hits / plane}^{DS};Entries",
        "column": "ds_plane_hits",
        "bins": 60,
        "xmin": 0.0,
        "xmax": 60.0,
        "sig_rebin": 2,
    },
    {
        "name": "ds_max_plane_hits",
        "title": "DS Hits in Plane of Max Occupancy;N_{hits}^{max DS plane};Events",
        "column": "ds_max_plane_hits",
        "bins": 60,
        "xmin": 0.0,
        "xmax": 60.0,
        "sig_rebin": 2,
    },
]

# Informative 2D histogram pairings
HIST_CONFIGS_2D = [
    ("scifi_total_hits", "us_total_qdc", 40, 0.0, 2000.0, 40, 0.0, 190000.0, "Total US QDC vs Total SciFi Hits;N_{hits}^{SciFi};QDC_{tot}^{US}"),
    ("us_total_qdc", "ds_total_qdc", 40, 0.0, 190000.0, 40, 0.0, 150000.0, "Total DS QDC vs Total US QDC;QDC_{tot}^{US};QDC_{tot}^{DS}"),
    ("us_total_hits", "us_total_qdc", 40, 0.0, 100.0, 40, 0.0, 190000.0, "Total US QDC vs Total US Hits;N_{hits}^{US};QDC_{tot}^{US}"),
    ("ds_total_hits", "ds_total_qdc", 40, 0.0, 200.0, 40, 0.0, 150000.0, "Total DS QDC vs Total DS Hits;N_{hits}^{DS};QDC_{tot}^{DS}"),
    ("scifi_total_hits", "scifi_total_qdc", 40, 0.0, 2000.0, 40, 0.0, 50000.0, "Total SciFi QDC vs Total SciFi Hits;N_{hits}^{SciFi};QDC_{tot}^{SciFi}"),
    ("scifi_total_hits", "ds_total_hits", 40, 0.0, 2000.0, 40, 0.0, 200.0, "Total DS Hits vs Total SciFi Hits;N_{hits}^{SciFi};N_{hits}^{DS}"),
    ("scifi_total_hits", "us_total_hits", 40, 0.0, 2000.0, 40, 0.0, 100.0, "Total US Hits vs Total SciFi Hits;N_{hits}^{SciFi};N_{hits}^{US}"),
    ("us_total_hits", "ds_total_hits", 40, 0.0, 100.0, 40, 0.0, 200.0, "Total DS Hits vs Total US Hits;N_{hits}^{US};N_{hits}^{DS}"),
    ("scifi_max_plane_hits", "us_max_plane_qdc", 40, 0.0, 400.0, 40, 0.0, 60000.0, "US Max Plane QDC vs SciFi Max Plane Hits;N_{hits}^{max SciFi plane};QDC_{max-occ plane}^{US}"),
    ("veto_total_hits", "veto_delayed_hits", 20, 0.0, 20.0, 20, 0.0, 20.0, "Delayed Veto Hits vs Total Veto Hits;N_{hits}^{Veto};N_{hits}^{Veto delayed}"),
]


def book_1d_histograms(df, prefix: str, weight_col: Optional[str] = None):
    """Book 1D histograms using RDataFrame."""
    booked = {}
    cols = [str(c) for c in df.GetColumnNames()]
    use_weight = bool(weight_col and weight_col in cols)

    for cfg in HIST_CONFIGS_1D:
        name = f"h_{prefix}_{cfg['name']}"
        title = f"{prefix}: {cfg['title']}"
        model = ROOT.RDF.TH1DModel(name, title, cfg["bins"], cfg["xmin"], cfg["xmax"])
        if use_weight:
            booked[cfg["name"]] = df.Histo1D(model, cfg["column"], weight_col)
        else:
            booked[cfg["name"]] = df.Histo1D(model, cfg["column"])
    return booked


def book_2d_histograms(df, prefix: str, weight_col: Optional[str] = None):
    """Book 2D histograms using RDataFrame."""
    booked = {}
    cols = [str(c) for c in df.GetColumnNames()]
    use_weight = bool(weight_col and weight_col in cols)

    for col_x, col_y, nbins_x, xmin, xmax, nbins_y, ymin, ymax, title in HIST_CONFIGS_2D:
        pair_name = f"{col_x}_vs_{col_y}"
        hist_name = f"h2_{prefix}_{pair_name}"
        full_title = f"{prefix}: {title}"
        model = ROOT.RDF.TH2DModel(hist_name, full_title, nbins_x, xmin, xmax, nbins_y, ymin, ymax)
        if use_weight:
            booked[pair_name] = df.Histo2D(model, col_x, col_y, weight_col)
        else:
            booked[pair_name] = df.Histo2D(model, col_x, col_y)
    return booked


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive QDC & Hit Occupancy distribution generator and overlay comparison tool."
    )
    parser.add_argument("--input-genie", default="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root",
                        help="Input GENIE files, pattern or %%s partition template")
    parser.add_argument("--input-pmu", default="/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP_2MuTrks_truth.root",
                        help="Input PMU muon simulation file or pattern")
    parser.add_argument("--input-data", default="/eos/experiment/sndlhc/convertedData/physics/2023/run_006596/sndsw_raw-*.root",
                        help="Input collision data files or pattern")
    parser.add_argument("-o", "--output", default="qdc_hit_distributions.root",
                        help="Output ROOT file path (default: qdc_hit_distributions.root)")
    parser.add_argument("-j", "--threads", type=int, default=4,
                        help="Number of worker threads for ROOT ImplicitMT (default: 4)")
    parser.add_argument("-n", "--max-events", type=int, default=None,
                        help="Maximum events to process per sample (for testing)")
    args = parser.parse_args()

    t_start = time.time()

    threads = args.threads
    if args.max_events:
        threads = 1  # Range is incompatible with ImplicitMT in ROOT
    if threads > 1:
        ROOT.EnableImplicitMT(threads)
        print(f"Enabled ROOT ImplicitMT with {threads} threads.")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    declare_cpp_qdc_metrics()

    booked_all_1d: Dict[Tuple[str, str], Any] = {}
    booked_all_2d: Dict[Tuple[str, str], Any] = {}
    chain_holders: List[Any] = []

    # -------------------------------------------------------------------------
    # 1. Collision Data Sample
    # -------------------------------------------------------------------------
    if args.input_data:
        data_files = resolve_file_list(args.input_data)
        if data_files:
            print(f"\nProcessing Collision Data: {len(data_files)} files found.")
            first_f = ROOT.TFile.Open(data_files[0])
            tree_name = "rawConv" if (first_f and first_f.Get("rawConv")) else "cbmsim"
            if first_f:
                first_f.Close()

            ch_d, ch_d_mc, df_data = build_dataframe_with_friend(data_files, tree_name, "")
            chain_holders.extend([ch_d, ch_d_mc])

            if args.max_events:
                df_data = df_data.Range(args.max_events)
            df_data = define_qdc_metrics(df_data, is_data=True)

            booked_all_1d[("data", "data")] = book_1d_histograms(df_data, "data")
            booked_all_2d[("data", "data")] = book_2d_histograms(df_data, "data")
        else:
            print(f"Warning: No collision data files found for {args.input_data}")

    # -------------------------------------------------------------------------
    # 2. PMU Muon Background Simulation Sample
    # -------------------------------------------------------------------------
    if args.input_pmu:
        pmu_files = resolve_file_list(args.input_pmu)
        if pmu_files:
            print(f"\nProcessing PMU Simulation: {len(pmu_files)} files found.")
            ch_p, ch_p_mc, df_pmu = build_dataframe_with_friend(pmu_files, "cbmsim", "mctruth")
            chain_holders.extend([ch_p, ch_p_mc])

            if args.max_events:
                df_pmu = df_pmu.Range(args.max_events)
            df_pmu = define_qdc_metrics(df_pmu, is_data=False)

            cols = [str(c) for c in df_pmu.GetColumnNames()]
            pmu_weight = "mctruth.mc_weight" if "mctruth.mc_weight" in cols else ("mc_weight" if "mc_weight" in cols else None)
            booked_all_1d[("pmu2022", "pmu")] = book_1d_histograms(df_pmu, "pmu", pmu_weight)
            booked_all_2d[("pmu2022", "pmu")] = book_2d_histograms(df_pmu, "pmu", pmu_weight)
        else:
            print(f"Warning: No PMU files found for {args.input_pmu}")

    # -------------------------------------------------------------------------
    # 3. GENIE Simulation Sample (Inclusive, Signal, and Background)
    # -------------------------------------------------------------------------
    if args.input_genie:
        genie_files = resolve_file_list(args.input_genie)
        if genie_files:
            print(f"\nProcessing GENIE Simulation: {len(genie_files)} files found.")
            ch_g, ch_g_mc, df_genie = build_dataframe_with_friend(genie_files, "cbmsim", "mctruth")
            chain_holders.extend([ch_g, ch_g_mc])

            if args.max_events:
                df_genie = df_genie.Range(args.max_events)
            df_genie = define_qdc_metrics(df_genie, is_data=False)

            cols = [str(c) for c in df_genie.GetColumnNames()]
            genie_weight = "mctruth.mc_weight" if "mctruth.mc_weight" in cols else ("mc_weight" if "mc_weight" in cols else None)

            # Signal definition: numu -> charm -> 2nd muon
            has_truth = "mctruth.charm_has_direct_muon" in cols
            if has_truth:
                sig_cut = "mctruth.is_cc == 1 && mctruth.has_charm == 1 && mctruth.charm_has_direct_muon == 1"
                bkg_cut = "!(mctruth.is_cc == 1 && mctruth.has_charm == 1 && mctruth.charm_has_direct_muon == 1)"

                df_genie_inc = df_genie
                df_genie_sig = df_genie.Filter(sig_cut, "GENIE Charm Dimuon Signal")
                df_genie_bkg = df_genie.Filter(bkg_cut, "GENIE Background")

                booked_all_1d[("genie/inclusive", "genie_inc")] = book_1d_histograms(df_genie_inc, "genie_inc", genie_weight)
                booked_all_2d[("genie/inclusive", "genie_inc")] = book_2d_histograms(df_genie_inc, "genie_inc", genie_weight)

                booked_all_1d[("genie/signal", "genie_sig")] = book_1d_histograms(df_genie_sig, "genie_sig", genie_weight)
                booked_all_2d[("genie/signal", "genie_sig")] = book_2d_histograms(df_genie_sig, "genie_sig", genie_weight)

                booked_all_1d[("genie/background", "genie_bkg")] = book_1d_histograms(df_genie_bkg, "genie_bkg", genie_weight)
                booked_all_2d[("genie/background", "genie_bkg")] = book_2d_histograms(df_genie_bkg, "genie_bkg", genie_weight)
            else:
                print("Warning: 'mctruth' branch not found in GENIE sample; booking as inclusive.")
                booked_all_1d[("genie/inclusive", "genie_inc")] = book_1d_histograms(df_genie, "genie_inc", genie_weight)
                booked_all_2d[("genie/inclusive", "genie_inc")] = book_2d_histograms(df_genie, "genie_inc", genie_weight)
        else:
            print(f"Warning: No GENIE files found for {args.input_genie}")

    # -------------------------------------------------------------------------
    # 4. Materialize Histograms & Apply Signal Rebinning
    # -------------------------------------------------------------------------
    print("\nTriggering event loop evaluations...")
    materialized_1d = {}
    for key, h_dict in booked_all_1d.items():
        materialized_1d[key] = {var_name: h_res.GetValue() for var_name, h_res in h_dict.items()}

    materialized_2d = {}
    for key, h2_dict in booked_all_2d.items():
        materialized_2d[key] = {pair_name: h2_res.GetValue() for pair_name, h2_res in h2_dict.items()}

    # Rebin signal histograms for low statistics
    sig_rebin_map = {cfg["name"]: cfg.get("sig_rebin", 1) for cfg in HIST_CONFIGS_1D}
    if ("genie/signal", "genie_sig") in materialized_1d:
        sig_dict = materialized_1d[("genie/signal", "genie_sig")]
        for var_name, h_obj in sig_dict.items():
            factor = sig_rebin_map.get(var_name, 1)
            if factor > 1:
                h_obj.Rebin(factor)

    # -------------------------------------------------------------------------
    # 5. Write ROOT File with Structured TDirectories
    # -------------------------------------------------------------------------
    print(f"\nWriting output to: {args.output}")
    out_file = ROOT.TFile.Open(args.output, "RECREATE")

    # Write 1D Histograms
    for (dir_path, _), hist_dict in materialized_1d.items():
        current_dir = out_file
        for part in dir_path.split("/"):
            sub = current_dir.GetDirectory(part)
            if not sub:
                sub = current_dir.mkdir(part)
            current_dir = sub

        current_dir.cd()
        for name, h in hist_dict.items():
            current_dir.WriteTObject(h, h.GetName(), "Overwrite")

    # Write 2D Histograms under <dir_path>/2d/
    for (dir_path, _), h2_dict in materialized_2d.items():
        sub_2d_path = f"{dir_path}/2d"
        current_dir = out_file
        for part in sub_2d_path.split("/"):
            sub = current_dir.GetDirectory(part)
            if not sub:
                sub = current_dir.mkdir(part)
            current_dir = sub

        current_dir.cd()
        for name, h2 in h2_dict.items():
            current_dir.WriteTObject(h2, h2.GetName(), "Overwrite")

    # -------------------------------------------------------------------------
    # 6. Generate Superimposed Normalized Comparison TCanvases
    # -------------------------------------------------------------------------
    print("\nGenerating normalized overlay comparison TCanvases...")
    d_canvases = out_file.GetDirectory("comparisons")
    if not d_canvases:
        d_canvases = out_file.mkdir("comparisons")

    d_canvases_log = d_canvases.GetDirectory("log")
    if not d_canvases_log:
        d_canvases_log = d_canvases.mkdir("log")

    # Overlay specifications:
    # (cat_key, legend_label, line_color, line_width, is_marker, marker_style, marker_size, marker_color)
    overlay_specs = [
        (("data", "data"), "Collision Data", ROOT.kBlack, 2, True, 20, 0.8, ROOT.kBlack),
        (("pmu2022", "pmu"), "pmu2022 Muon Sim", ROOT.kAzure + 1, 2, False, 0, 0.0, ROOT.kAzure + 1),
        (("genie/background", "genie_bkg"), "GENIE Other / Bkg", ROOT.kOrange + 7, 2, False, 0, 0.0, ROOT.kOrange + 7),
        (("genie/inclusive", "genie_inc"), "GENIE Inclusive", ROOT.kRed + 1, 2, False, 0, 0.0, ROOT.kRed + 1),
        (("genie/signal", "genie_sig"), "GENIE Charm Signal (#nu_{#mu}#rightarrow c#rightarrow 2^{nd}#mu)", ROOT.kMagenta + 2, 2, True, 21, 0.9, ROOT.kMagenta + 2),
    ]

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetEndErrorSize(0)

    for cfg in HIST_CONFIGS_1D:
        var_name = cfg["name"]
        canvas_name = f"c_{var_name}_normalized"
        canvas_title = f"Normalized Comparison: {cfg['title'].split(';')[0]}"

        # 6a. Linear Scale Canvas
        c = ROOT.TCanvas(canvas_name, canvas_title, 950, 700)
        c.SetLeftMargin(0.12)
        c.SetRightMargin(0.06)
        c.SetTopMargin(0.08)
        c.SetBottomMargin(0.12)

        leg = ROOT.TLegend(0.46, 0.68, 0.92, 0.89)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.033)

        h_clones = []
        max_val = 0.0
        min_pos_val = 1e30

        for cat_key, label, lcolor, lwidth, is_marker, mstyle, msize, mcolor in overlay_specs:
            if cat_key not in materialized_1d:
                continue
            cat_dict = materialized_1d[cat_key]
            if var_name not in cat_dict:
                continue
            h_orig = cat_dict[var_name]
            if not h_orig or h_orig.GetEntries() == 0:
                continue

            # Clone and normalize by unit area / bin width for consistent probability density
            h_norm = h_orig.Clone(f"{h_orig.GetName()}_norm")
            integral = h_norm.Integral()
            if integral > 0:
                h_norm.Scale(1.0 / integral, "width")

            h_norm.SetLineColor(lcolor)
            h_norm.SetLineWidth(lwidth)
            h_norm.SetLineStyle(1)

            if is_marker:
                h_norm.SetMarkerStyle(mstyle)
                h_norm.SetMarkerSize(msize)
                h_norm.SetMarkerColor(mcolor)
                h_norm.SetFillStyle(0)
            else:
                h_norm.SetFillStyle(0)

            h_norm.GetYaxis().SetTitle("Probability Density [1 / unit]")
            h_norm.GetYaxis().SetTitleOffset(1.3)
            h_norm.GetXaxis().SetTitle(h_orig.GetXaxis().GetTitle())

            curr_max = h_norm.GetMaximum()
            if curr_max > max_val:
                max_val = curr_max

            for b in range(1, h_norm.GetNbinsX() + 1):
                val = h_norm.GetBinContent(b)
                if val > 0.0 and val < min_pos_val:
                    min_pos_val = val

            h_clones.append((h_norm, label, is_marker))

        if h_clones:
            for idx, (h_norm, label, is_marker) in enumerate(h_clones):
                h_norm.SetMaximum(max_val * 1.35)
                h_norm.SetMinimum(0.0)
                if is_marker:
                    draw_opt = "E1 P" if idx == 0 else "E1 P SAME"
                    leg.AddEntry(h_norm, label, "PL")
                else:
                    draw_opt = "HIST" if idx == 0 else "HIST SAME"
                    leg.AddEntry(h_norm, label, "L")
                h_norm.Draw(draw_opt)

            leg.Draw()
            d_canvases.cd()
            c.Write()

            # 6b. Log-Y Scale Canvas
            c_log = ROOT.TCanvas(f"{canvas_name}_logy", f"{canvas_title} (Log Scale)", 950, 700)
            c_log.SetLeftMargin(0.12)
            c_log.SetRightMargin(0.06)
            c_log.SetTopMargin(0.08)
            c_log.SetBottomMargin(0.12)
            c_log.SetLogy(1)

            leg_log = ROOT.TLegend(0.46, 0.68, 0.92, 0.89)
            leg_log.SetBorderSize(0)
            leg_log.SetFillStyle(0)
            leg_log.SetTextFont(42)
            leg_log.SetTextSize(0.033)

            log_min = max(1e-7, min_pos_val * 0.5) if min_pos_val < 1e20 else 1e-6
            for idx, (h_norm, label, is_marker) in enumerate(h_clones):
                h_log_clone = h_norm.Clone(f"{h_norm.GetName()}_logclone")
                h_log_clone.SetMaximum(max_val * 5.0)
                h_log_clone.SetMinimum(log_min)
                if is_marker:
                    draw_opt = "E1 P" if idx == 0 else "E1 P SAME"
                    leg_log.AddEntry(h_log_clone, label, "PL")
                else:
                    draw_opt = "HIST" if idx == 0 else "HIST SAME"
                    leg_log.AddEntry(h_log_clone, label, "L")
                h_log_clone.Draw(draw_opt)

            leg_log.Draw()
            d_canvases_log.cd()
            c_log.Write()

    out_file.Close()
    elapsed = time.time() - t_start
    print(f"\nDone! Output saved to: {args.output} (Elapsed: {elapsed:.1f} s)")


if __name__ == "__main__":
    main()
