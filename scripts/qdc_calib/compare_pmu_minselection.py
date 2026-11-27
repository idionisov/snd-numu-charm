#!/usr/bin/env python3
"""
scripts/qdc_calib/compare_pmu_minselection.py
---------------------------------------------
Side-by-side comparison of Hit Occupancy distributions between:
  1. Real Collision Data from Run 5408 (LHC Stable Beams, IP1)
  2. 2022 PMU Monte Carlo without energy threshold correction (Raw digitized hits)
  3. 2022 PMU Monte Carlo with energy threshold correction (SiPM trigger qualification,
     including 0.9 ns rise time, Birks' light quenching, and calibrated k_SiPM)

Minimal Event Selection (Apples-to-Apples):
  - Requires physical detector activity: N_total_hits (Veto + US + DS) >= 3
  - Requires at least 1 hit in the MuFilter (US or DS >= 1) to eliminate pure noise / random trigger empties.
  - No through-going plane constraints, preserving stopping muons, wide angles, and showers.

Evaluates:
  - Subsystems: Veto Total, US Total, DS Total
  - Individual planes:
    * Veto Planes 0, 1
    * US Planes 0, 1, 2, 3, 4
    * DS Horizontal Stations 0, 1, 2
    * DS Vertical Stations 0, 1, 2, 3

Outputs:
  - ROOT file: output/pmu_vs_run5408_minselection.root
  - Publication-quality PNG figures in plots/
"""

from __future__ import annotations

import os
import sys
import glob
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Add repository root to path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import load_trident_libraries
load_trident_libraries(_repo_root)

import scripts.qdc_calib.min_pe as mpe
import ddfUtils.mpl_styling as dms

# Observable histogram specifications: (name, title, nbins, xmin, xmax)
HIST_SPECS = {
    # Subsystems
    "veto_total": ("Veto Total Hits", "N_{hits}^{Veto}", 16, -0.5, 15.5),
    "us_total":   ("US Total Hits", "N_{hits}^{US}", 26, -0.5, 25.5),
    "ds_total":   ("DS Total Hits", "N_{hits}^{DS}", 31, -0.5, 30.5),
    # Veto Planes
    "veto_pl0":   ("Veto Plane 0 Hits", "N_{hits}^{Veto 0}", 11, -0.5, 10.5),
    "veto_pl1":   ("Veto Plane 1 Hits", "N_{hits}^{Veto 1}", 11, -0.5, 10.5),
    # US Planes
    "us_pl0":     ("US Plane 0 Hits", "N_{hits}^{US 0}", 11, -0.5, 10.5),
    "us_pl1":     ("US Plane 1 Hits", "N_{hits}^{US 1}", 11, -0.5, 10.5),
    "us_pl2":     ("US Plane 2 Hits", "N_{hits}^{US 2}", 11, -0.5, 10.5),
    "us_pl3":     ("US Plane 3 Hits", "N_{hits}^{US 3}", 11, -0.5, 10.5),
    "us_pl4":     ("US Plane 4 Hits", "N_{hits}^{US 4}", 11, -0.5, 10.5),
    # DS Horizontal Planes
    "ds_h_pl0":   ("DS Station 0 H Hits", "N_{hits}^{DS-H 0}", 11, -0.5, 10.5),
    "ds_h_pl1":   ("DS Station 1 H Hits", "N_{hits}^{DS-H 1}", 11, -0.5, 10.5),
    "ds_h_pl2":   ("DS Station 2 H Hits", "N_{hits}^{DS-H 2}", 11, -0.5, 10.5),
    # DS Vertical Planes
    "ds_v_pl0":   ("DS Station 0 V Hits", "N_{hits}^{DS-V 0}", 11, -0.5, 10.5),
    "ds_v_pl1":   ("DS Station 1 V Hits", "N_{hits}^{DS-V 1}", 11, -0.5, 10.5),
    "ds_v_pl2":   ("DS Station 2 V Hits", "N_{hits}^{DS-V 2}", 11, -0.5, 10.5),
    "ds_v_pl3":   ("DS Station 3 V Hits", "N_{hits}^{DS-V 3}", 11, -0.5, 10.5),
}


# ==============================================================================
# C++ Fast Data Processor Declaration
# ==============================================================================
FAST_DATA_CPP = """
#ifndef FAST_DATA_PROCESSOR_MIN_H
#define FAST_DATA_PROCESSOR_MIN_H

#include "TTree.h"
#include "TClonesArray.h"
#include "MuFilterHit.h"
#include "SNDLHCEventHeader.h"
#include <vector>
#include <cmath>
#include <iostream>

struct DataHitsRecordMin {
    int veto_total;
    int us_total;
    int ds_total;
    int veto_pl0;
    int veto_pl1;
    int us_pl0;
    int us_pl1;
    int us_pl2;
    int us_pl3;
    int us_pl4;
    int ds_h_pl0;
    int ds_h_pl1;
    int ds_h_pl2;
    int ds_v_pl0;
    int ds_v_pl1;
    int ds_v_pl2;
    int ds_v_pl3;
};

struct DataExtractionResultMin {
    std::vector<DataHitsRecordMin> records;
    Long64_t n_scanned;
};

DataExtractionResultMin extract_run5408_muons_min(TTree* tree, Long64_t max_entries) {
    DataExtractionResultMin result;
    if (!tree) return result;

    SNDLHCEventHeader* header = nullptr;
    TClonesArray* mfHits = nullptr;

    tree->SetBranchAddress("EventHeader", &header);
    tree->SetBranchAddress("Digi_MuFilterHits", &mfHits);

    Long64_t n_entries = tree->GetEntries();
    if (max_entries > 0 && max_entries < n_entries) n_entries = max_entries;
    result.n_scanned = n_entries;

    double prev_t = -999999.0;

    for (Long64_t i = 0; i < n_entries; ++i) {
        tree->GetEntry(i);
        if (!header || !header->isIP1()) continue;
        int bm = header->GetBeamMode();
        if (bm != 11 && bm != 14) continue;
        double t_cur = header->GetEventTime();
        if (std::abs(t_cur - prev_t) <= 100.0) {
            prev_t = t_cur;
            continue;
        }
        prev_t = t_cur;

        if (!mfHits) continue;

        int n_mf = mfHits->GetEntriesFast();
        DataHitsRecordMin rec{};

        for (int j = 0; j < n_mf; ++j) {
            auto* h = static_cast<MuFilterHit*>(mfHits->At(j));
            if (!h || !h->isValid()) continue;

            int sys = h->GetSystem();
            int plane = h->GetPlane();
            bool is_vert = h->isVertical();

            if (sys == 1) { // Veto
                if (plane == 0 || plane == 1) {
                    rec.veto_total++;
                    if (plane == 0) rec.veto_pl0++;
                    else if (plane == 1) rec.veto_pl1++;
                }
            } else if (sys == 2) { // US
                rec.us_total++;
                if (plane == 0) rec.us_pl0++;
                else if (plane == 1) rec.us_pl1++;
                else if (plane == 2) rec.us_pl2++;
                else if (plane == 3) rec.us_pl3++;
                else if (plane == 4) rec.us_pl4++;
            } else if (sys == 3) { // DS
                rec.ds_total++;
                if (!is_vert) {
                    if (plane == 0) rec.ds_h_pl0++;
                    else if (plane == 1) rec.ds_h_pl1++;
                    else if (plane == 2) rec.ds_h_pl2++;
                } else {
                    if (plane == 0) rec.ds_v_pl0++;
                    else if (plane == 1) rec.ds_v_pl1++;
                    else if (plane == 2) rec.ds_v_pl2++;
                    else if (plane == 3) rec.ds_v_pl3++;
                }
            }
        }

        // Minimal selection criteria:
        // Total hits (Veto + US + DS) >= 3 AND at least 1 hit in MuFilter (US or DS)
        int total_hits = rec.veto_total + rec.us_total + rec.ds_total;
        if (total_hits >= 3 && (rec.us_total >= 1 || rec.ds_total >= 1)) {
            result.records.push_back(rec);
        }
    }
    return result;
}
#endif
"""

ROOT.gInterpreter.Declare(FAST_DATA_CPP)


def _data_file_worker(args_tuple: Tuple[str, int]) -> Tuple[List[Dict[str, int]], int]:
    """Worker to extract muons from a single raw data file."""
    fpath, max_ev = args_tuple
    f = ROOT.TFile.Open(fpath)
    if not f or f.IsZombie():
        return [], 0
    tree = f.Get("rawConv")
    if not tree:
        f.Close()
        return [], 0

    tree.SetCacheSize(64 * 1024 * 1024)
    tree.SetBranchStatus("*", 0)
    tree.SetBranchStatus("EventHeader*", 1)
    tree.SetBranchStatus("Digi_MuFilterHits*", 1)
    tree.AddBranchToCache("EventHeader", True)
    tree.AddBranchToCache("Digi_MuFilterHits", True)
    tree.StopCacheLearningPhase()

    res = ROOT.extract_run5408_muons_min(tree, max_ev)
    recs = []
    for rec in res.records:
        recs.append({
            "veto_total": rec.veto_total,
            "us_total": rec.us_total,
            "ds_total": rec.ds_total,
            "veto_pl0": rec.veto_pl0,
            "veto_pl1": rec.veto_pl1,
            "us_pl0": rec.us_pl0,
            "us_pl1": rec.us_pl1,
            "us_pl2": rec.us_pl2,
            "us_pl3": rec.us_pl3,
            "us_pl4": rec.us_pl4,
            "ds_h_pl0": rec.ds_h_pl0,
            "ds_h_pl1": rec.ds_h_pl1,
            "ds_h_pl2": rec.ds_h_pl2,
            "ds_v_pl0": rec.ds_v_pl0,
            "ds_v_pl1": rec.ds_v_pl1,
            "ds_v_pl2": rec.ds_v_pl2,
            "ds_v_pl3": rec.ds_v_pl3,
        })
    n_sc = int(res.n_scanned)
    f.Close()
    return recs, n_sc


def process_collision_data(files: List[str], max_events: int = 500000, n_workers: int = 8) -> Tuple[Dict[str, np.ndarray], int]:
    """Process collision data across files in parallel."""
    print(f"\n[Data] Processing Run 5408 from {len(files)} file(s), up to {max_events:,} total events ({n_workers} workers)...")
    t0 = time.time()

    per_file_max = (max_events + len(files) - 1) // len(files)
    tasks = [(fpath, per_file_max) for fpath in files]

    n_procs = min(n_workers, len(files))
    with mp.Pool(processes=n_procs) as pool:
        chunk_results = pool.map(_data_file_worker, tasks)

    all_records = []
    total_scanned = 0
    for chunk, n_sc in chunk_results:
        all_records.extend(chunk)
        total_scanned += n_sc

    elapsed = time.time() - t0
    print(f"[Data] Completed in {elapsed:.1f}s: Scanned {total_scanned:,} events, found {len(all_records):,} minimal-selection muons.")

    data_dict = {}
    for key in HIST_SPECS:
        data_dict[key] = np.array([r[key] for r in all_records], dtype=np.int32) if all_records else np.array([])
    return data_dict, total_scanned


# ==============================================================================
# MC Chunk Worker Routine
# ==============================================================================
def _mc_chunk_worker(args_tuple: Tuple[str, int, int, int]) -> Dict[str, Any]:
    """Worker function to process a slice of PMU MC events."""
    mc_file, start_ev, n_ev, worker_id = args_tuple
    f = ROOT.TFile.Open(mc_file)
    if not f or f.IsZombie():
        return {}

    tree = f.Get("cbmsim")
    if not tree:
        f.Close()
        return {}

    tree.SetCacheSize(32 * 1024 * 1024)
    tree.SetBranchStatus("*", 0)
    tree.SetBranchStatus("MCTrack*", 1)
    tree.SetBranchStatus("Digi_MuFilterHits*", 1)
    tree.SetBranchStatus("MuFilterPoint*", 1)
    tree.AddBranchToCache("MCTrack", True)
    tree.AddBranchToCache("Digi_MuFilterHits", True)
    tree.AddBranchToCache("MuFilterPoint", True)
    tree.StopCacheLearningPhase()

    calc = mpe.MinPE(year=2022)  # Automatically includes rise time, Birks, and re-extracted k_SiPM

    raw_data: Dict[str, List[float]] = {k: [] for k in HIST_SPECS}
    raw_data["weights"] = []

    corr_data: Dict[str, List[float]] = {k: [] for k in HIST_SPECS}
    corr_data["weights"] = []

    end_ev = min(start_ev + n_ev, tree.GetEntries())

    for ev_idx in range(start_ev, end_ev):
        tree.GetEntry(ev_idx)

        tracks = getattr(tree, "MCTrack", None)
        wt = 1.0
        if tracks and tracks.GetEntriesFast() > 0:
            wt = float(tracks.At(0).GetWeight()) * 8e5

        mf_hits = getattr(tree, "Digi_MuFilterHits", None)
        pts_branch = getattr(tree, "MuFilterPoint", None)

        # 1. PMU Raw: uses raw digitized hits
        if mf_hits and mf_hits.GetEntriesFast() > 0:
            raw_rec = {k: 0 for k in HIST_SPECS}

            for j in range(mf_hits.GetEntriesFast()):
                h = mf_hits.At(j)
                if not h or not h.isValid():
                    continue
                sys_id = h.GetSystem()
                plane = h.GetPlane()
                is_vert = h.isVertical()

                if sys_id == 1:
                    if plane in (0, 1):
                        raw_rec["veto_total"] += 1
                        if plane == 0: raw_rec["veto_pl0"] += 1
                        elif plane == 1: raw_rec["veto_pl1"] += 1
                elif sys_id == 2:
                    raw_rec["us_total"] += 1
                    if plane == 0: raw_rec["us_pl0"] += 1
                    elif plane == 1: raw_rec["us_pl1"] += 1
                    elif plane == 2: raw_rec["us_pl2"] += 1
                    elif plane == 3: raw_rec["us_pl3"] += 1
                    elif plane == 4: raw_rec["us_pl4"] += 1
                elif sys_id == 3:
                    raw_rec["ds_total"] += 1
                    if not is_vert:
                        if plane == 0: raw_rec["ds_h_pl0"] += 1
                        elif plane == 1: raw_rec["ds_h_pl1"] += 1
                        elif plane == 2: raw_rec["ds_h_pl2"] += 1
                    else:
                        if plane == 0: raw_rec["ds_v_pl0"] += 1
                        elif plane == 1: raw_rec["ds_v_pl1"] += 1
                        elif plane == 2: raw_rec["ds_v_pl2"] += 1
                        elif plane == 3: raw_rec["ds_v_pl3"] += 1

            raw_tot = raw_rec["veto_total"] + raw_rec["us_total"] + raw_rec["ds_total"]
            if raw_tot >= 3 and (raw_rec["us_total"] >= 1 or raw_rec["ds_total"] >= 1):
                for k in HIST_SPECS:
                    raw_data[k].append(raw_rec[k])
                raw_data["weights"].append(wt)

        # 2. PMU Corrected: evaluate threshold qualification per bar
        if pts_branch and pts_branch.GetEntriesFast() > 0:
            grouped = mpe.group_mcpoints_by_bar(pts_branch)
            corr_rec = {k: 0 for k in HIST_SPECS}

            for det_id, clusters in grouped.items():
                if not calc.is_valid_det_id(det_id):
                    continue
                for pts in clusters:
                    th_info = calc.get_lowest_trigger_threshold(pts, det_id=det_id, include_mirror=True)
                    if th_info["current_hit_qualified"]:
                        sys_id, plane, bar = calc.decode_detector_id(det_id)
                        is_vert = calc.is_vertical(det_id)

                        if sys_id == 1:
                            if plane in (0, 1):
                                corr_rec["veto_total"] += 1
                                if plane == 0: corr_rec["veto_pl0"] += 1
                                elif plane == 1: corr_rec["veto_pl1"] += 1
                        elif sys_id == 2:
                            corr_rec["us_total"] += 1
                            if plane == 0: corr_rec["us_pl0"] += 1
                            elif plane == 1: corr_rec["us_pl1"] += 1
                            elif plane == 2: corr_rec["us_pl2"] += 1
                            elif plane == 3: corr_rec["us_pl3"] += 1
                            elif plane == 4: corr_rec["us_pl4"] += 1
                        elif sys_id == 3:
                            corr_rec["ds_total"] += 1
                            if not is_vert:
                                if plane == 0: corr_rec["ds_h_pl0"] += 1
                                elif plane == 1: corr_rec["ds_h_pl1"] += 1
                                elif plane == 2: corr_rec["ds_h_pl2"] += 1
                            else:
                                if plane == 0: corr_rec["ds_v_pl0"] += 1
                                elif plane == 1: corr_rec["ds_v_pl1"] += 1
                                elif plane == 2: corr_rec["ds_v_pl2"] += 1
                                elif plane == 3: corr_rec["ds_v_pl3"] += 1

            corr_tot = corr_rec["veto_total"] + corr_rec["us_total"] + corr_rec["ds_total"]
            if corr_tot >= 3 and (corr_rec["us_total"] >= 1 or corr_rec["ds_total"] >= 1):
                for k in HIST_SPECS:
                    corr_data[k].append(corr_rec[k])
                corr_data["weights"].append(wt)

    f.Close()
    return {"raw": raw_data, "corr": corr_data, "n_scanned": end_ev - start_ev}


def process_pmu_simulation(mc_file: str, max_events: int = 60000, n_workers: int = 8) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """Process PMU Monte Carlo in parallel chunks."""
    print(f"\n[MC] Processing 2022 PMU Simulation ({max_events:,} events, {n_workers} workers)...")
    t0 = time.time()

    chunk_size = (max_events + n_workers - 1) // n_workers
    tasks = []
    for w in range(n_workers):
        start_ev = w * chunk_size
        if start_ev >= max_events:
            break
        n_ev = min(chunk_size, max_events - start_ev)
        tasks.append((mc_file, start_ev, n_ev, w))

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_mc_chunk_worker, tasks)

    merged_raw: Dict[str, np.ndarray] = {k: [] for k in HIST_SPECS}
    merged_raw["weights"] = []
    merged_corr: Dict[str, np.ndarray] = {k: [] for k in HIST_SPECS}
    merged_corr["weights"] = []
    total_mc_scanned = 0

    for res in results:
        if not res:
            continue
        for k in HIST_SPECS:
            merged_raw[k].extend(res["raw"][k])
            merged_corr[k].extend(res["corr"][k])
        merged_raw["weights"].extend(res["raw"]["weights"])
        merged_corr["weights"].extend(res["corr"]["weights"])
        total_mc_scanned += res.get("n_scanned", 0)

    for k in HIST_SPECS:
        merged_raw[k] = np.array(merged_raw[k], dtype=np.float64)
        merged_corr[k] = np.array(merged_corr[k], dtype=np.float64)
    merged_raw["weights"] = np.array(merged_raw["weights"], dtype=np.float64)
    merged_corr["weights"] = np.array(merged_corr["weights"], dtype=np.float64)

    elapsed = time.time() - t0
    print(f"[MC] Completed in {elapsed:.1f}s (Scanned {total_mc_scanned:,} events):")
    print(f"     Raw Events       : {len(merged_raw['weights']):,}")
    print(f"     Corrected Events : {len(merged_corr['weights']):,}")

    return merged_raw, merged_corr, total_mc_scanned


# ==============================================================================
# Plotting and Canvas Generation
# ==============================================================================
def plot_comparison_panels(
    data_dict: Dict[str, np.ndarray],
    mc_raw: Dict[str, Any],
    mc_corr: Dict[str, Any],
    keys: List[str],
    fig_title: str,
    out_png: str,
    ncols: int = 3,
    norm_mode: str = "lumi_data",
    lumi_info: Optional[Dict[str, float]] = None,
):
    """Generate multi-panel comparison figure with luminosity or area normalization and vertical error bars."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.0
    nrows = (len(keys) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows * 2, ncols,
        figsize=(5.5 * ncols, 3.8 * nrows),
        gridspec_kw={"height_ratios": [3, 1] * nrows},
        constrained_layout=True,
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif nrows == 1:
        axes = np.array([axes[0, :], axes[1, :]])
    elif ncols == 1:
        axes = axes.reshape(nrows * 2, 1)

    scale_mc = lumi_info["scale_mc_to_data"] if lumi_info else 1.0
    lumi_pb = (lumi_info["lumi_data"] * 1e3) if lumi_info else 1.80

    for idx, key in enumerate(keys):
        row = (idx // ncols) * 2
        col = idx % ncols
        ax_main = axes[row, col]
        ax_ratio = axes[row + 1, col]

        title, xlabel, nbins, xmin, xmax = HIST_SPECS[key]
        bins = np.linspace(xmin, xmax, nbins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = bins[1] - bins[0]

        d_vals = data_dict[key]
        raw_vals = mc_raw[key]
        raw_wts = mc_raw["weights"]
        corr_vals = mc_corr[key]
        corr_wts = mc_corr["weights"]

        h_data, _ = np.histogram(d_vals, bins=bins)
        h_raw, _ = np.histogram(raw_vals, bins=bins, weights=raw_wts)
        raw_w2, _ = np.histogram(raw_vals, bins=bins, weights=raw_wts**2)
        h_corr, _ = np.histogram(corr_vals, bins=bins, weights=corr_wts)
        corr_w2, _ = np.histogram(corr_vals, bins=bins, weights=corr_wts**2)

        if norm_mode == "lumi_data":
            # Matched to Data Integrated Luminosity
            y_data = h_data.astype(np.float64)
            y_data_err = np.sqrt(np.maximum(y_data, 0.0))

            y_raw = h_raw * scale_mc
            y_raw_err = np.sqrt(raw_w2) * scale_mc

            y_corr = h_corr * scale_mc
            y_corr_err = np.sqrt(corr_w2) * scale_mc

            y_axis_label = f"Events / bin (L = {lumi_pb:.2f} pb$^{{-1}}$)"
            data_label = f"Run 5408 Data ({lumi_pb:.2f} pb$^{{-1}}$)"
        else:
            # Area Normalized
            norm_d = np.sum(h_data) * bin_width if np.sum(h_data) > 0 else 1.0
            y_data = h_data / norm_d
            y_data_err = np.sqrt(h_data) / norm_d

            norm_raw = np.sum(h_raw) * bin_width if np.sum(h_raw) > 0 else 1.0
            y_raw = h_raw / norm_raw
            y_raw_err = np.sqrt(raw_w2) / norm_raw

            norm_corr = np.sum(h_corr) * bin_width if np.sum(h_corr) > 0 else 1.0
            y_corr = h_corr / norm_corr
            y_corr_err = np.sqrt(corr_w2) / norm_corr

            y_axis_label = "Normalized Entries / bin"
            data_label = f"Run 5408 Data (N={len(d_vals):,})"

        # Plot Main Panel
        # 1. Data Points
        ax_main.errorbar(
            bin_centers, y_data, yerr=y_data_err,
            fmt="o", color="black", ms=5, lw=1.5, capsize=2,
            label=data_label, zorder=10
        )

        # 2. PMU Sim Raw with Vertical Error Bars and step outline
        ax_main.errorbar(
            bin_centers - 0.08, y_raw, yerr=y_raw_err,
            fmt="s", color="#D9534F", mfc="none", ms=4.5, lw=1.2, capsize=2,
            label="PMU Sim Raw (Eth=0)", zorder=5
        )
        ax_main.step(
            bins[:-1], y_raw, where="post", color="#D9534F",
            linestyle=":", lw=1.4, alpha=0.85
        )

        # 3. PMU Sim Corrected with Vertical Error Bars, step outline, and shaded band
        ax_main.errorbar(
            bin_centers + 0.08, y_corr, yerr=y_corr_err,
            fmt="o", color="#0275D8", ms=4.5, lw=1.4, capsize=2,
            label="PMU Sim Corrected (Rise+Birks+k)", zorder=6
        )
        ax_main.step(
            bins[:-1], y_corr, where="post", color="#0275D8",
            linestyle="-", lw=1.8, alpha=0.9
        )
        ax_main.fill_between(
            bins[:-1], np.maximum(y_corr - y_corr_err, 0.0), y_corr + y_corr_err,
            step="post", color="#0275D8", alpha=0.18
        )

        ax_main.set_xlim(xmin, xmax)
        ax_main.set_yscale("log")
        y_max = max(np.max(y_data), np.max(y_raw), np.max(y_corr)) * 3.5 if len(y_data) > 0 else 1.0
        ax_main.set_ylim(max(0.5, y_max * 1e-4) if norm_mode == "lumi_data" else 2e-4, max(y_max, 10.0))
        ax_main.set_ylabel(y_axis_label, fontsize=10)
        ax_main.set_title(title, fontsize=11, fontweight="bold")
        ax_main.grid(True, which="both", linestyle=":", alpha=0.5)
        ax_main.tick_params(labelbottom=False)
        if idx == 0:
            ax_main.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5, loc="upper right")

        # Plot Ratio Panel
        mask_raw = (y_data > 0) & (y_raw > 0)
        mask_corr = (y_data > 0) & (y_corr > 0)

        r_raw = np.full(nbins, np.nan)
        r_raw_err = np.full(nbins, np.nan)
        r_raw[mask_raw] = y_raw[mask_raw] / y_data[mask_raw]
        r_raw_err[mask_raw] = r_raw[mask_raw] * np.sqrt(
            (y_data_err[mask_raw] / y_data[mask_raw])**2 + (y_raw_err[mask_raw] / y_raw[mask_raw])**2
        )

        r_corr = np.full(nbins, np.nan)
        r_corr_err = np.full(nbins, np.nan)
        r_corr[mask_corr] = y_corr[mask_corr] / y_data[mask_corr]
        r_corr_err[mask_corr] = r_corr[mask_corr] * np.sqrt(
            (y_data_err[mask_corr] / y_data[mask_corr])**2 + (y_corr_err[mask_corr] / y_corr[mask_corr])**2
        )

        ax_ratio.axhline(1.0, color="black", linestyle="--", lw=1.0)
        if np.any(mask_raw):
            ax_ratio.errorbar(
                bin_centers[mask_raw] - 0.08, r_raw[mask_raw], yerr=r_raw_err[mask_raw],
                fmt="s", color="#D9534F", mfc="none", ms=4.5, capsize=2, lw=1.2, label="Raw / Data"
            )
        if np.any(mask_corr):
            ax_ratio.errorbar(
                bin_centers[mask_corr] + 0.08, r_corr[mask_corr], yerr=r_corr_err[mask_corr],
                fmt="o", color="#0275D8", ms=5, capsize=2, lw=1.4, label="Corr / Data"
            )

        ax_ratio.set_xlim(xmin, xmax)
        ax_ratio.set_ylim(0.0, 3.5)
        ax_ratio.set_ylabel("MC / Data", fontsize=9, fontweight="bold")
        ax_ratio.set_xlabel(xlabel, fontsize=10, fontweight="bold")
        ax_ratio.grid(True, linestyle=":", alpha=0.55)
        ax_ratio.xaxis.set_minor_locator(AutoMinorLocator())
        if idx == 0:
            ax_ratio.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.0, loc="upper right")

    # Turn off unused axes
    for idx in range(len(keys), nrows * ncols):
        row = (idx // ncols) * 2
        col = idx % ncols
        axes[row, col].axis("off")
        axes[row + 1, col].axis("off")

    fig.suptitle(fig_title, fontsize=13, fontweight="bold")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_png}")


def save_root_output(
    data_dict: Dict[str, np.ndarray],
    mc_raw: Dict[str, Any],
    mc_corr: Dict[str, Any],
    out_root_path: str,
    lumi_info: Optional[Dict[str, float]] = None,
):
    """Write all comparison histograms and styled TCanvases to ROOT file with explicit vertical error bars."""
    out_file = ROOT.TFile(out_root_path, "RECREATE")

    dir_hists_lumi = out_file.mkdir("Histograms_Lumi")
    dir_canvases_lumi = out_file.mkdir("Canvases_Lumi")
    dir_hists_area = out_file.mkdir("Histograms_Area")
    dir_canvases_area = out_file.mkdir("Canvases_Area")

    scale_mc = lumi_info["scale_mc_to_data"] if lumi_info else 1.0
    lumi_pb = (lumi_info["lumi_data"] * 1e3) if lumi_info else 1.80

    for key, (title, xlabel, nbins, xmin, xmax) in HIST_SPECS.items():
        # --- 1. Luminosity Matched ---
        dir_hists_lumi.cd()
        h_dl = ROOT.TH1D(f"h_data_lumi_{key}", f"{title} - Run 5408 Data ({lumi_pb:.2f} pb^{{-1}});{xlabel};Events / bin", nbins, xmin, xmax)
        h_dl.Sumw2()
        for v in data_dict[key]:
            h_dl.Fill(v)

        h_rawl = ROOT.TH1D(f"h_raw_lumi_{key}", f"{title} - PMU Raw ({lumi_pb:.2f} pb^{{-1}});{xlabel};Events / bin", nbins, xmin, xmax)
        h_rawl.Sumw2()
        for v, w in zip(mc_raw[key], mc_raw["weights"]):
            h_rawl.Fill(v, w * scale_mc)

        h_corrl = ROOT.TH1D(f"h_corr_lumi_{key}", f"{title} - PMU Corrected ({lumi_pb:.2f} pb^{{-1}});{xlabel};Events / bin", nbins, xmin, xmax)
        h_corrl.Sumw2()
        for v, w in zip(mc_corr[key], mc_corr["weights"]):
            h_corrl.Fill(v, w * scale_mc)

        h_dl.Write()
        h_rawl.Write()
        h_corrl.Write()

        dir_canvases_lumi.cd()
        c_l = ROOT.TCanvas(f"c_lumi_{key}", f"Lumi Matched Comparison - {title}", 900, 750)
        p1l = ROOT.TPad("p1l", "main", 0.0, 0.3, 1.0, 1.0)
        p2l = ROOT.TPad("p2l", "ratio", 0.0, 0.0, 1.0, 0.3)
        p1l.SetBottomMargin(0.02)
        p1l.SetGrid()
        p1l.SetLogy()
        p2l.SetTopMargin(0.05)
        p2l.SetBottomMargin(0.28)
        p2l.SetGrid()
        p1l.Draw()
        p2l.Draw()

        p1l.cd()
        max_y = max(h_dl.GetMaximum(), h_rawl.GetMaximum(), h_corrl.GetMaximum()) * 3.5
        h_dl.SetMaximum(max_y)
        h_dl.SetMinimum(max(0.5, max_y * 1e-4))
        h_dl.SetMarkerStyle(20)
        h_dl.SetMarkerSize(1.0)
        h_dl.SetMarkerColor(ROOT.kBlack)
        h_dl.SetLineColor(ROOT.kBlack)
        h_dl.Draw("E1")

        h_rawl.SetMarkerStyle(25)
        h_rawl.SetMarkerSize(0.9)
        h_rawl.SetMarkerColor(ROOT.kRed + 1)
        h_rawl.SetLineColor(ROOT.kRed + 1)
        h_rawl.Draw("E1 SAME")

        h_corrl.SetMarkerStyle(20)
        h_corrl.SetMarkerSize(0.9)
        h_corrl.SetMarkerColor(ROOT.kAzure + 2)
        h_corrl.SetLineColor(ROOT.kAzure + 2)
        h_corrl.Draw("E1 SAME")

        leg_l = ROOT.TLegend(0.55, 0.70, 0.88, 0.88)
        leg_l.SetBorderSize(0)
        leg_l.SetFillStyle(0)
        leg_l.AddEntry(h_dl, f"Run 5408 Data ({lumi_pb:.2f} pb^{{-1}})", "lep")
        leg_l.AddEntry(h_rawl, f"PMU Raw (Eth=0)", "lep")
        leg_l.AddEntry(h_corrl, f"PMU Corrected (Rise+Birks+k)", "lep")
        leg_l.Draw()

        p2l.cd()
        h_rl_raw = h_rawl.Clone(f"h_r_raw_lumi_{key}")
        h_rl_raw.Divide(h_dl)
        h_rl_corr = h_corrl.Clone(f"h_r_corr_lumi_{key}")
        h_rl_corr.Divide(h_dl)

        h_rl_raw.SetTitle(f";{xlabel};MC / Data")
        h_rl_raw.GetYaxis().SetRangeUser(0.0, 3.5)
        h_rl_raw.SetMarkerStyle(25)
        h_rl_raw.SetMarkerSize(0.9)
        h_rl_raw.SetMarkerColor(ROOT.kRed + 1)
        h_rl_raw.SetLineColor(ROOT.kRed + 1)
        h_rl_raw.Draw("E1")

        h_rl_corr.SetMarkerStyle(20)
        h_rl_corr.SetMarkerSize(0.9)
        h_rl_corr.SetMarkerColor(ROOT.kAzure + 2)
        h_rl_corr.SetLineColor(ROOT.kAzure + 2)
        h_rl_corr.Draw("E1 SAME")
        c_l.Write()

        # --- 2. Area Normalized ---
        dir_hists_area.cd()
        h_da = ROOT.TH1D(f"h_data_area_{key}", f"{title} - Run 5408 Data (Area Norm);{xlabel};Normalized Entries", nbins, xmin, xmax)
        h_da.Sumw2()
        for v in data_dict[key]:
            h_da.Fill(v)
        if h_da.Integral() > 0:
            h_da.Scale(1.0 / h_da.Integral())

        h_rawa = ROOT.TH1D(f"h_raw_area_{key}", f"{title} - PMU Raw (Area Norm);{xlabel};Normalized Entries", nbins, xmin, xmax)
        h_rawa.Sumw2()
        for v, w in zip(mc_raw[key], mc_raw["weights"]):
            h_rawa.Fill(v, w)
        if h_rawa.Integral() > 0:
            h_rawa.Scale(1.0 / h_rawa.Integral())

        h_corra = ROOT.TH1D(f"h_corr_area_{key}", f"{title} - PMU Corrected (Area Norm);{xlabel};Normalized Entries", nbins, xmin, xmax)
        h_corra.Sumw2()
        for v, w in zip(mc_corr[key], mc_corr["weights"]):
            h_corra.Fill(v, w)
        if h_corra.Integral() > 0:
            h_corra.Scale(1.0 / h_corra.Integral())

        h_da.Write()
        h_rawa.Write()
        h_corra.Write()

        dir_canvases_area.cd()
        c_a = ROOT.TCanvas(f"c_area_{key}", f"Area Norm Comparison - {title}", 900, 750)
        p1a = ROOT.TPad("p1a", "main", 0.0, 0.3, 1.0, 1.0)
        p2a = ROOT.TPad("p2a", "ratio", 0.0, 0.0, 1.0, 0.3)
        p1a.SetBottomMargin(0.02)
        p1a.SetGrid()
        p1a.SetLogy()
        p2a.SetTopMargin(0.05)
        p2a.SetBottomMargin(0.28)
        p2a.SetGrid()
        p1a.Draw()
        p2a.Draw()

        p1a.cd()
        max_ya = max(h_da.GetMaximum(), h_rawa.GetMaximum(), h_corra.GetMaximum()) * 3.5
        h_da.SetMaximum(max_ya)
        h_da.SetMinimum(2e-4)
        h_da.SetMarkerStyle(20)
        h_da.SetMarkerColor(ROOT.kBlack)
        h_da.SetLineColor(ROOT.kBlack)
        h_da.Draw("E1")

        h_rawa.SetMarkerStyle(25)
        h_rawa.SetMarkerColor(ROOT.kRed + 1)
        h_rawa.SetLineColor(ROOT.kRed + 1)
        h_rawa.Draw("E1 SAME")

        h_corra.SetMarkerStyle(20)
        h_corra.SetMarkerColor(ROOT.kAzure + 2)
        h_corra.SetLineColor(ROOT.kAzure + 2)
        h_corra.Draw("E1 SAME")

        p2a.cd()
        h_ra_raw = h_rawa.Clone(f"h_r_raw_area_{key}")
        h_ra_raw.Divide(h_da)
        h_ra_corr = h_corra.Clone(f"h_r_corr_area_{key}")
        h_ra_corr.Divide(h_da)

        h_ra_raw.SetTitle(f";{xlabel};MC / Data")
        h_ra_raw.GetYaxis().SetRangeUser(0.0, 3.5)
        h_ra_raw.SetMarkerStyle(25)
        h_ra_raw.SetMarkerColor(ROOT.kRed + 1)
        h_ra_raw.SetLineColor(ROOT.kRed + 1)
        h_ra_raw.Draw("E1")

        h_ra_corr.SetMarkerStyle(20)
        h_ra_corr.SetMarkerColor(ROOT.kAzure + 2)
        h_ra_corr.SetLineColor(ROOT.kAzure + 2)
        h_ra_corr.Draw("E1 SAME")
        c_a.Write()

    out_file.Close()
    print(f"Saved ROOT file with lumi and area comparisons: {out_root_path}")


def main():
    parser = argparse.ArgumentParser(description="PMU vs Run 5408 Data with Minimal Selection")
    parser.add_argument("--data-files", nargs="+", default=[f"/eos/experiment/sndlhc/convertedData/physics/2022/run_005408/sndsw_raw-{i:04d}.root" for i in range(8)],
                        help="Data root file(s)")
    parser.add_argument("--mc-file", default="/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP_2MuTrks_truth.root",
                        help="PMU MC simulation root file")
    parser.add_argument("--max-events-data", type=int, default=400000, help="Max data events to process")
    parser.add_argument("--max-events-mc", type=int, default=60000, help="Max MC events to process")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("-o", "--output", default="output/pmu_vs_run5408_minselection.root", help="Output ROOT file")
    parser.add_argument("--plot-dir", default="plots", help="Directory to save figures")
    parser.add_argument("--cache-file", default="output/cache_pmu_minselection.npz", help="Cache NPZ file path")
    parser.add_argument("--force-reprocess", action="store_true", help="Force re-reading of raw data and MC files")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cache_path = args.cache_file
    if os.path.exists(cache_path) and not args.force_reprocess:
        print(f"\n[Cache] Loading cached minimal-selection events from {cache_path}...")
        loaded = np.load(cache_path, allow_pickle=True)
        data_dict = {k: loaded[f"data_{k}"] for k in HIST_SPECS}
        mc_raw = {k: loaded[f"raw_{k}"] for k in HIST_SPECS}
        mc_raw["weights"] = loaded["raw_weights"]
        mc_corr = {k: loaded[f"corr_{k}"] for k in HIST_SPECS}
        mc_corr["weights"] = loaded["corr_weights"]
        n_data_scanned = int(loaded["n_data_scanned"])
        n_mc_scanned = int(loaded["n_mc_scanned"])
        print(f"  Loaded {len(data_dict['us_total']):,} data events, {len(mc_raw['weights']):,} MC raw events, {len(mc_corr['weights']):,} MC corr events.")
    else:
        # 1. Process Data
        data_dict, n_data_scanned = process_collision_data(args.data_files, max_events=args.max_events_data, n_workers=args.workers)

        # 2. Process MC (Raw and Corrected)
        mc_raw, mc_corr, n_mc_scanned = process_pmu_simulation(args.mc_file, max_events=args.max_events_mc, n_workers=args.workers)

        # Save Cache
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        save_dict = {}
        for k in HIST_SPECS:
            save_dict[f"data_{k}"] = data_dict[k]
            save_dict[f"raw_{k}"] = mc_raw[k]
            save_dict[f"corr_{k}"] = mc_corr[k]
        save_dict["raw_weights"] = mc_raw["weights"]
        save_dict["corr_weights"] = mc_corr["weights"]
        save_dict["n_data_scanned"] = np.array(n_data_scanned)
        save_dict["n_mc_scanned"] = np.array(n_mc_scanned)
        np.savez_compressed(cache_path, **save_dict)
        print(f"[Cache] Saved events to {cache_path}")

    # Compute Luminosity Normalization Info
    TOTAL_RUN5408_EVENTS = 90174916
    TOTAL_RUN5408_LUMI_FB = 0.406569
    TOTAL_PMU_EVENTS = 1108804
    TOTAL_PMU_LUMI_FB = 1.0

    lumi_data_analyzed = (n_data_scanned / TOTAL_RUN5408_EVENTS) * TOTAL_RUN5408_LUMI_FB  # fb^-1
    lumi_mc_analyzed = (n_mc_scanned / TOTAL_PMU_EVENTS) * TOTAL_PMU_LUMI_FB              # fb^-1
    scale_mc_to_data = lumi_data_analyzed / lumi_mc_analyzed if lumi_mc_analyzed > 0 else 1.0

    lumi_info = {
        "lumi_data": lumi_data_analyzed,
        "lumi_mc": lumi_mc_analyzed,
        "scale_mc_to_data": scale_mc_to_data,
        "n_data_scanned": n_data_scanned,
        "n_mc_scanned": n_mc_scanned,
    }

    print("\n[Luminosity Normalization Summary]")
    print(f"  Run 5408 Data Scanned : {n_data_scanned:,} / {TOTAL_RUN5408_EVENTS:,} events")
    print(f"  Data Int. Luminosity  : {lumi_data_analyzed*1e3:.3f} pb^-1 ({lumi_data_analyzed:.6f} fb^-1)")
    print(f"  PMU MC Scanned        : {n_mc_scanned:,} / {TOTAL_PMU_EVENTS:,} events")
    print(f"  MC Int. Luminosity    : {lumi_mc_analyzed*1e3:.3f} pb^-1 ({lumi_mc_analyzed:.6f} fb^-1)")
    print(f"  MC -> Data Lumi Scale : {scale_mc_to_data:.6f}")

    # 3. Generate PNG Figures (Matched to Data Luminosity with Explicit Vertical Error Bars)
    subsystem_keys = ["veto_total", "us_total", "ds_total"]
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, subsystem_keys,
        f"Subsystems Hit Multiplicity: Run 5408 Data vs. 2022 PMU (Lumi = {lumi_data_analyzed*1e3:.2f} pb$^{{-1}}$, N_total >= 3, MuFilter >= 1)",
        os.path.join(args.plot_dir, "pmu_minselection_subsystems.png"),
        ncols=3, norm_mode="lumi_data", lumi_info=lumi_info,
    )

    veto_keys = ["veto_pl0", "veto_pl1"]
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, veto_keys,
        f"Veto Planes Hit Multiplicity: Run 5408 Data vs. 2022 PMU (Lumi = {lumi_data_analyzed*1e3:.2f} pb$^{{-1}}$, Minimal Selection)",
        os.path.join(args.plot_dir, "pmu_minselection_veto_planes.png"),
        ncols=2, norm_mode="lumi_data", lumi_info=lumi_info,
    )

    us_keys = [f"us_pl{p}" for p in range(5)]
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, us_keys,
        f"Upstream (US) Planes Hit Multiplicity: Run 5408 Data vs. 2022 PMU (Lumi = {lumi_data_analyzed*1e3:.2f} pb$^{{-1}}$, Minimal Selection)",
        os.path.join(args.plot_dir, "pmu_minselection_us_planes.png"),
        ncols=3, norm_mode="lumi_data", lumi_info=lumi_info,
    )

    ds_keys = ["ds_h_pl0", "ds_h_pl1", "ds_h_pl2", "ds_v_pl0", "ds_v_pl1", "ds_v_pl2", "ds_v_pl3"]
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, ds_keys,
        f"Downstream (DS) Planes Hit Multiplicity: Run 5408 Data vs. 2022 PMU (Lumi = {lumi_data_analyzed*1e3:.2f} pb$^{{-1}}$, Minimal Selection)",
        os.path.join(args.plot_dir, "pmu_minselection_ds_planes.png"),
        ncols=4, norm_mode="lumi_data", lumi_info=lumi_info,
    )

    # Area Normalized figures as well
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, subsystem_keys,
        "Subsystems Hit Multiplicity: Run 5408 Data vs. 2022 PMU (Area Normalized, Minimal Selection)",
        os.path.join(args.plot_dir, "pmu_minselection_subsystems_area.png"),
        ncols=3, norm_mode="area",
    )
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, us_keys,
        "US Planes Hit Multiplicity: Run 5408 Data vs. 2022 PMU (Area Normalized, Minimal Selection)",
        os.path.join(args.plot_dir, "pmu_minselection_us_planes_area.png"),
        ncols=3, norm_mode="area",
    )
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, ds_keys,
        "DS Planes Hit Multiplicity: Run 5408 Data vs. 2022 PMU (Area Normalized, Minimal Selection)",
        os.path.join(args.plot_dir, "pmu_minselection_ds_planes_area.png"),
        ncols=4, norm_mode="area",
    )
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, veto_keys,
        "Veto Planes Hit Multiplicity: Run 5408 Data vs. 2022 PMU (Area Normalized, Minimal Selection)",
        os.path.join(args.plot_dir, "pmu_minselection_veto_planes_area.png"),
        ncols=2, norm_mode="area",
    )

    # 4. Save ROOT histograms and canvases
    save_root_output(data_dict, mc_raw, mc_corr, args.output, lumi_info=lumi_info)

    # Copy to brain artifact directory
    art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"
    if os.path.exists(art_dir):
        os.system(f"cp -f {args.output} {art_dir}/")
        os.system(f"cp -f {args.plot_dir}/pmu_minselection_*.png {art_dir}/")


if __name__ == "__main__":
    main()
