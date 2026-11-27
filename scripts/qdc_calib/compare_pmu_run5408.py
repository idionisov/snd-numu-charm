#!/usr/bin/env python3
"""
scripts/qdc_calib/compare_pmu_run5408.py
----------------------------------------
Side-by-side comparison of Hit Occupancy distributions between:
  1. Real Collision Data from Run 5408 (LHC Stable Beams, IP1, >100 clock cycle isolation)
  2. 2022 PMU Monte Carlo without energy threshold correction (Raw digitized hits)
  3. 2022 PMU Monte Carlo with energy threshold correction (SiPM trigger qualification)

Event Selection:
  - Through-going muon selection ("all planes activated" in Upstream MuFilter: US0..US4 all hit)
  - Superimposes Data, PMU Raw, and PMU Corrected normalized to unit area.
  - Produces publication-quality PNGs with ratio panels and saves full ROOT canvases.

Usage:
  python3 -X faulthandler scripts/qdc_calib/compare_pmu_run5408.py \
    --data-files "/eos/experiment/sndlhc/convertedData/physics/2022/run_005408/sndsw_raw-0000.root" \
    --mc-file "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP.root" \
    --max-events-data 300000 \
    --max-events-mc 60000 \
    --workers 8 \
    --output plots/pmu_vs_run5408.root \
    --plot-dir plots
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
    "us_total":   ("US Total Hits", "N_{hits}^{US}", 21, 4.5, 25.5),
    "ds_total":   ("DS Total Hits", "N_{hits}^{DS}", 31, -0.5, 30.5),
    # Veto Planes
    "veto_pl0":   ("Veto Plane 0 Hits", "N_{hits}^{Veto 0}", 11, -0.5, 10.5),
    "veto_pl1":   ("Veto Plane 1 Hits", "N_{hits}^{Veto 1}", 11, -0.5, 10.5),
    # US Planes
    "us_pl0":     ("US Plane 0 Hits", "N_{hits}^{US 0}", 10, 0.5, 10.5),
    "us_pl1":     ("US Plane 1 Hits", "N_{hits}^{US 1}", 10, 0.5, 10.5),
    "us_pl2":     ("US Plane 2 Hits", "N_{hits}^{US 2}", 10, 0.5, 10.5),
    "us_pl3":     ("US Plane 3 Hits", "N_{hits}^{US 3}", 10, 0.5, 10.5),
    "us_pl4":     ("US Plane 4 Hits", "N_{hits}^{US 4}", 10, 0.5, 10.5),
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
#ifndef FAST_DATA_PROCESSOR_H
#define FAST_DATA_PROCESSOR_H

#include "TTree.h"
#include "TClonesArray.h"
#include "MuFilterHit.h"
#include "SNDLHCEventHeader.h"
#include <vector>
#include <set>
#include <cmath>
#include <iostream>

struct DataHitsRecord {
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

struct DataExtractionResult {
    std::vector<DataHitsRecord> records;
    Long64_t n_scanned;
};

DataExtractionResult extract_run5408_muons(TTree* tree, Long64_t max_entries) {
    DataExtractionResult result;
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
        std::set<int> us_planes;

        DataHitsRecord rec{};

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
                us_planes.insert(plane);
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

        // Require front-entering through-going muon condition:
        // Veto coincidence (>=1 hit in Plane 0 AND >=1 hit in Plane 1) + all 5 US planes hit
        if (us_planes.size() == 5 && rec.veto_pl0 >= 1 && rec.veto_pl1 >= 1) {
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

    res = ROOT.extract_run5408_muons(tree, max_ev)
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


def process_collision_data(files: List[str], max_events: int = 800000, n_workers: int = 8) -> Tuple[Dict[str, np.ndarray], int]:
    """Process collision data across files in parallel."""
    print(f"\n[Data] Processing Run 5408 from {len(files)} file(s), up to {max_events} total events ({n_workers} workers)...")
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
    print(f"[Data] Completed in {elapsed:.1f}s: Scanned {total_scanned:,} events, found {len(all_records)} front-entering muons.")

    # Convert to dictionary of arrays
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

    calc = mpe.MinPE(year=2022)

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
            # Scale to 1 fb^-1 according to official simulation definition: MCTrack[0].GetWeight() * 8e5
            wt = float(tracks.At(0).GetWeight()) * 8e5

        mf_hits = getattr(tree, "Digi_MuFilterHits", None)
        pts_branch = getattr(tree, "MuFilterPoint", None)

        # ----------------------------------------------------------------------
        # 1. PMU Raw: uses raw digitized hits
        # ----------------------------------------------------------------------
        if mf_hits and mf_hits.GetEntriesFast() > 0:
            raw_rec = {k: 0 for k in HIST_SPECS}
            raw_us_planes = set()

            for j in range(mf_hits.GetEntriesFast()):
                h = static_cast = mf_hits.At(j)
                if not h or not h.isValid():
                    continue
                sys_id = h.GetSystem()
                plane = h.GetPlane()
                is_vert = h.isVertical()

                if sys_id == 1:
                    # 2022 configuration: only Veto plane 0 and plane 1 exist
                    if plane in (0, 1):
                        raw_rec["veto_total"] += 1
                        if plane == 0: raw_rec["veto_pl0"] += 1
                        elif plane == 1: raw_rec["veto_pl1"] += 1
                elif sys_id == 2:
                    raw_rec["us_total"] += 1
                    raw_us_planes.add(plane)
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

            # Require front-entering through-going muon condition (raw):
            # Veto coincidence (>=1 hit in Plane 0 AND >=1 hit in Plane 1) + all 5 US planes hit
            if (len(raw_us_planes) == 5 and raw_rec["veto_pl0"] >= 1 and raw_rec["veto_pl1"] >= 1):
                for k in HIST_SPECS:
                    raw_data[k].append(raw_rec[k])
                raw_data["weights"].append(wt)

        # ----------------------------------------------------------------------
        # 2. PMU Corrected: evaluate threshold qualification per bar
        # ----------------------------------------------------------------------
        if pts_branch and pts_branch.GetEntriesFast() > 0:
            grouped = mpe.group_mcpoints_by_bar(pts_branch)
            corr_rec = {k: 0 for k in HIST_SPECS}
            corr_us_planes = set()

            for det_id, clusters in grouped.items():
                if not calc.is_valid_det_id(det_id):
                    continue
                for pts in clusters:
                    th_info = calc.get_lowest_trigger_threshold(pts, det_id=det_id, include_mirror=True)
                    if th_info["current_hit_qualified"]:
                        sys_id, plane, bar = calc.decode_detector_id(det_id)
                        is_vert = calc.is_vertical(det_id)

                        if sys_id == 1:
                            # 2022 configuration: only Veto plane 0 and plane 1 exist
                            if plane in (0, 1):
                                corr_rec["veto_total"] += 1
                                if plane == 0: corr_rec["veto_pl0"] += 1
                                elif plane == 1: corr_rec["veto_pl1"] += 1
                        elif sys_id == 2:
                            corr_rec["us_total"] += 1
                            corr_us_planes.add(plane)
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

            # Require strictly self-consistent front-entering through-going muon condition (corrected):
            # Evaluated purely on threshold-qualified hits:
            # Veto coincidence (>=1 hit in Plane 0 AND >=1 hit in Plane 1) + all 5 US planes hit
            if (len(corr_us_planes) == 5 and corr_rec["veto_pl0"] >= 1 and corr_rec["veto_pl1"] >= 1):
                for k in HIST_SPECS:
                    corr_data[k].append(corr_rec[k])
                corr_data["weights"].append(wt)

    f.Close()
    return {"raw": raw_data, "corr": corr_data, "n_scanned": end_ev - start_ev}


def process_pmu_simulation(mc_file: str, max_events: int = 60000, n_workers: int = 8) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """Process PMU Monte Carlo in parallel chunks."""
    print(f"\n[MC] Processing 2022 PMU Simulation ({max_events} events, {n_workers} workers)...")
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

    # Merge results
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
    print(f"     Raw US5 Muons       : {len(merged_raw['weights'])} events (Sum of weights: {np.sum(merged_raw['weights']):.4f})")
    print(f"     Corrected US5 Muons : {len(merged_corr['weights'])} events (Sum of weights: {np.sum(merged_corr['weights']):.4f})")

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
    norm_mode: str = "area",
    lumi_info: Optional[Dict[str, float]] = None,
):
    """Generate multi-panel comparison figure with data and MC ratio."""
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

    for idx, key in enumerate(keys):
        row = (idx // ncols) * 2
        col = idx % ncols
        ax_main = axes[row, col]
        ax_ratio = axes[row + 1, col]

        title, xlabel, nbins, xmin, xmax = HIST_SPECS[key]
        bins = np.linspace(xmin, xmax, nbins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = bins[1] - bins[0]

        # 1. Collision Data
        data_vals = data_dict[key]
        n_data = len(data_vals)
        h_data, _ = np.histogram(data_vals, bins=bins)

        # 2. PMU Raw
        raw_vals = mc_raw[key]
        raw_wts = mc_raw["weights"]
        sum_w_raw = np.sum(raw_wts)
        h_raw, _ = np.histogram(raw_vals, bins=bins, weights=raw_wts)
        h_raw_w2, _ = np.histogram(raw_vals, bins=bins, weights=raw_wts**2)

        # 3. PMU Corrected
        corr_vals = mc_corr[key]
        corr_wts = mc_corr["weights"]
        sum_w_corr = np.sum(corr_wts)
        h_corr, _ = np.histogram(corr_vals, bins=bins, weights=corr_wts)
        h_corr_w2, _ = np.histogram(corr_vals, bins=bins, weights=corr_wts**2)

        if norm_mode == "area":
            y_data = h_data / (n_data * bin_width) if n_data > 0 else np.zeros_like(h_data, dtype=float)
            y_data_err = np.sqrt(h_data) / (n_data * bin_width) if n_data > 0 else np.zeros_like(h_data, dtype=float)
            y_raw = h_raw / (sum_w_raw * bin_width) if sum_w_raw > 0 else np.zeros_like(h_raw, dtype=float)
            y_raw_err = np.sqrt(h_raw_w2) / (sum_w_raw * bin_width) if sum_w_raw > 0 else np.zeros_like(h_raw, dtype=float)
            y_corr = h_corr / (sum_w_corr * bin_width) if sum_w_corr > 0 else np.zeros_like(h_corr, dtype=float)
            y_corr_err = np.sqrt(h_corr_w2) / (sum_w_corr * bin_width) if sum_w_corr > 0 else np.zeros_like(h_corr, dtype=float)
            ylabel_str = "Normalized Density [1 / bin]"
            data_label = f"Run 5408 Data (N={n_data})"
            raw_label = "PMU 2022 (Raw, No Thresh)"
            corr_label = "PMU 2022 (Corrected Thresh)"
        else: # lumi_data
            scale_mc = lumi_info["scale_mc_to_data"] if lumi_info else 1.0
            lumi_pb = (lumi_info["lumi_data"] * 1e3) if lumi_info else 0.0
            y_data = h_data.astype(float)
            y_data_err = np.sqrt(h_data)
            y_raw = h_raw * scale_mc
            y_raw_err = np.sqrt(h_raw_w2) * scale_mc
            y_corr = h_corr * scale_mc
            y_corr_err = np.sqrt(h_corr_w2) * scale_mc
            ylabel_str = f"Events / bin (Data $\\mathcal{{L}} = {lumi_pb:.2f}$ pb$^{{-1}}$)"
            data_label = f"Run 5408 Data (N={n_data})"
            raw_label = f"PMU Raw (scaled to {lumi_pb:.2f} pb$^{{-1}}$)"
            corr_label = f"PMU Corrected (scaled to {lumi_pb:.2f} pb$^{{-1}}$)"

        # Plot on main axis
        # Stepped MC Raw
        ax_main.step(bins, np.append(y_raw, y_raw[-1]), where="post", color="#D9534F", lw=1.8, ls="--", label=raw_label)
        # Stepped & Shaded MC Corrected
        ax_main.step(bins, np.append(y_corr, y_corr[-1]), where="post", color="#0275D8", lw=2.0, label=corr_label)
        ax_main.fill_between(bins, np.append(y_corr, y_corr[-1]), step="post", color="#5BC0DE", alpha=0.25)

        # Data markers
        mask_d = y_data > 0
        ax_main.errorbar(
            bin_centers[mask_d], y_data[mask_d], yerr=y_data_err[mask_d],
            fmt="o", color="black", ms=5, capsize=2, lw=1.2, label=data_label
        )

        ax_main.set_title(title, fontsize=12, fontweight="bold")
        ax_main.set_ylabel(ylabel_str, fontsize=10)
        ax_main.set_xlim(xmin, xmax)
        max_val = max(np.max(y_data), np.max(y_raw), np.max(y_corr)) if len(y_data) > 0 else 1.0
        ax_main.set_ylim(0, max_val * 1.35 if max_val > 0 else 1.0)
        ax_main.grid(True, linestyle=":", alpha=0.6)
        ax_main.legend(loc="upper right", fontsize=8.0, framealpha=0.9)
        ax_main.xaxis.set_minor_locator(AutoMinorLocator())
        ax_main.yaxis.set_minor_locator(AutoMinorLocator())

        # Ratio axis (MC / Data) with markers and error bars
        r_raw = np.full_like(bin_centers, np.nan)
        r_raw_err = np.full_like(bin_centers, np.nan)
        r_corr = np.full_like(bin_centers, np.nan)
        r_corr_err = np.full_like(bin_centers, np.nan)

        mask_raw = (y_data > 0) & (y_raw > 0)
        r_raw[mask_raw] = y_raw[mask_raw] / y_data[mask_raw]
        r_raw_err[mask_raw] = r_raw[mask_raw] * np.sqrt(
            (y_data_err[mask_raw] / y_data[mask_raw])**2 + (y_raw_err[mask_raw] / y_raw[mask_raw])**2
        )

        mask_corr = (y_data > 0) & (y_corr > 0)
        r_corr[mask_corr] = y_corr[mask_corr] / y_data[mask_corr]
        r_corr_err[mask_corr] = r_corr[mask_corr] * np.sqrt(
            (y_data_err[mask_corr] / y_data[mask_corr])**2 + (y_corr_err[mask_corr] / y_corr[mask_corr])**2
        )

        ax_ratio.axhline(1.0, color="gray", linestyle="--", lw=1.0)
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

        # Dynamic y-axis for ratio to fully cover all markers and error bars
        all_tops = []
        if np.any(mask_raw):
            v_raw = r_raw[mask_raw] + r_raw_err[mask_raw]
            v_raw = v_raw[np.isfinite(v_raw)]
            if len(v_raw) > 0:
                all_tops.extend(v_raw)
        if np.any(mask_corr):
            v_corr = r_corr[mask_corr] + r_corr_err[mask_corr]
            v_corr = v_corr[np.isfinite(v_corr)]
            if len(v_corr) > 0:
                all_tops.extend(v_corr)

        all_bottoms = [0.0]
        if np.any(mask_raw):
            vb_raw = r_raw[mask_raw] - r_raw_err[mask_raw]
            vb_raw = vb_raw[np.isfinite(vb_raw)]
            if len(vb_raw) > 0:
                all_bottoms.append(np.min(vb_raw))
        if np.any(mask_corr):
            vb_corr = r_corr[mask_corr] - r_corr_err[mask_corr]
            vb_corr = vb_corr[np.isfinite(vb_corr)]
            if len(vb_corr) > 0:
                all_bottoms.append(np.min(vb_corr))

        y_top = max(all_tops) * 1.18 if len(all_tops) > 0 else 2.5
        y_bot = min(min(all_bottoms) * 1.05, 0.0)
        y_top = max(y_top, 2.0)
        ax_ratio.set_ylim(y_bot, y_top)

        ax_ratio.set_ylabel("MC / Data", fontsize=9)
        ax_ratio.set_xlabel(xlabel, fontsize=10)
        ax_ratio.grid(True, linestyle=":", alpha=0.5)
        ax_ratio.xaxis.set_minor_locator(AutoMinorLocator())

    # Turn off unused axes if any
    for idx in range(len(keys), nrows * ncols):
        row = (idx // ncols) * 2
        col = idx % ncols
        axes[row, col].axis("off")
        axes[row + 1, col].axis("off")

    fig.suptitle(fig_title, fontsize=14, fontweight="bold")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved figure: {out_png}")


def save_root_output(
    data_dict: Dict[str, np.ndarray],
    mc_raw: Dict[str, Any],
    mc_corr: Dict[str, Any],
    out_root_path: str,
    lumi_info: Optional[Dict[str, float]] = None,
):
    """Write all comparison histograms and styled TCanvases to ROOT file."""
    out_file = ROOT.TFile(out_root_path, "RECREATE")

    dir_hists_area = out_file.mkdir("histograms_area")
    dir_canvases_area = out_file.mkdir("canvases_area")
    dir_hists_lumi = out_file.mkdir("histograms_lumi")
    dir_canvases_lumi = out_file.mkdir("canvases_lumi")

    scale_mc = lumi_info["scale_mc_to_data"] if lumi_info else 1.0
    lumi_pb = (lumi_info["lumi_data"] * 1e3) if lumi_info else 0.0

    for key, (title, xlabel, nbins, xmin, xmax) in HIST_SPECS.items():
        # --- 1. Area Normalized ---
        dir_hists_area.cd()
        h_d = ROOT.TH1D(f"h_data_area_{key}", f"{title} - Run 5408 Data (Area Norm);{xlabel};Normalized Entries", nbins, xmin, xmax)
        h_d.Sumw2()
        for v in data_dict[key]:
            h_d.Fill(v)
        if h_d.Integral() > 0:
            h_d.Scale(1.0 / h_d.Integral())

        h_raw = ROOT.TH1D(f"h_mc_raw_area_{key}", f"{title} - PMU Raw (Area Norm);{xlabel};Normalized Entries", nbins, xmin, xmax)
        h_raw.Sumw2()
        for v, w in zip(mc_raw[key], mc_raw["weights"]):
            h_raw.Fill(v, w)
        if h_raw.Integral() > 0:
            h_raw.Scale(1.0 / h_raw.Integral())

        h_corr = ROOT.TH1D(f"h_mc_corr_area_{key}", f"{title} - PMU Corrected (Area Norm);{xlabel};Normalized Entries", nbins, xmin, xmax)
        h_corr.Sumw2()
        for v, w in zip(mc_corr[key], mc_corr["weights"]):
            h_corr.Fill(v, w)
        if h_corr.Integral() > 0:
            h_corr.Scale(1.0 / h_corr.Integral())

        h_d.Write()
        h_raw.Write()
        h_corr.Write()

        # Canvas Area
        dir_canvases_area.cd()
        c_area = ROOT.TCanvas(f"c_cmp_area_{key}", f"Area Normalized Comparison - {title}", 900, 800)
        p1 = ROOT.TPad("p1", "main pad", 0.0, 0.3, 1.0, 1.0)
        p2 = ROOT.TPad("p2", "ratio pad", 0.0, 0.0, 1.0, 0.3)
        p1.SetBottomMargin(0.02)
        p1.SetGrid()
        p2.SetTopMargin(0.05)
        p2.SetBottomMargin(0.28)
        p2.SetGrid()
        p1.Draw()
        p2.Draw()

        p1.cd()
        max_y = max(h_d.GetMaximum(), h_raw.GetMaximum(), h_corr.GetMaximum()) * 1.35
        h_d.SetMaximum(max_y)
        h_d.SetMinimum(0.0)
        h_d.SetMarkerStyle(20)
        h_d.SetMarkerSize(1.1)
        h_d.SetMarkerColor(ROOT.kBlack)
        h_d.SetLineColor(ROOT.kBlack)
        h_d.Draw("E1")

        h_raw.SetLineColor(ROOT.kRed + 1)
        h_raw.SetLineStyle(2)
        h_raw.SetLineWidth(2)
        h_raw.Draw("HIST SAME")

        h_corr.SetLineColor(ROOT.kAzure + 2)
        h_corr.SetLineWidth(2)
        h_corr.SetFillColorAlpha(ROOT.kAzure - 4, 0.35)
        h_corr.Draw("HIST SAME")
        h_d.Draw("E1 SAME")

        leg = ROOT.TLegend(0.55, 0.70, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.AddEntry(h_d, f"Run 5408 Data (N={len(data_dict[key])})", "lep")
        leg.AddEntry(h_raw, "PMU 2022 (Raw, No Thresh)", "l")
        leg.AddEntry(h_corr, "PMU 2022 (Corrected Thresh)", "lf")
        leg.Draw()

        p2.cd()
        h_ratio_raw = h_raw.Clone(f"h_ratio_raw_area_{key}")
        h_ratio_raw.Divide(h_d)
        h_ratio_corr = h_corr.Clone(f"h_ratio_corr_area_{key}")
        h_ratio_corr.Divide(h_d)

        max_r = max(h_ratio_raw.GetMaximum(), h_ratio_corr.GetMaximum()) * 1.25
        h_ratio_raw.SetTitle(f";{xlabel};MC / Data")
        h_ratio_raw.GetYaxis().SetRangeUser(0.0, max(min(max_r, 10.0), 2.5))
        h_ratio_raw.GetYaxis().SetNdivisions(505)
        h_ratio_raw.GetYaxis().SetLabelSize(0.08)
        h_ratio_raw.GetYaxis().SetTitleSize(0.09)
        h_ratio_raw.GetYaxis().SetTitleOffset(0.5)
        h_ratio_raw.GetXaxis().SetLabelSize(0.08)
        h_ratio_raw.GetXaxis().SetTitleSize(0.09)
        h_ratio_raw.SetMarkerStyle(25)
        h_ratio_raw.SetMarkerSize(0.9)
        h_ratio_raw.SetMarkerColor(ROOT.kRed + 1)
        h_ratio_raw.SetLineColor(ROOT.kRed + 1)
        h_ratio_raw.Draw("E1")

        h_ratio_corr.SetMarkerStyle(20)
        h_ratio_corr.SetMarkerSize(0.9)
        h_ratio_corr.SetMarkerColor(ROOT.kAzure + 2)
        h_ratio_corr.SetLineColor(ROOT.kAzure + 2)
        h_ratio_corr.Draw("E1 SAME")
        c_area.Write()

        # --- 2. Luminosity Scaled ---
        dir_hists_lumi.cd()
        h_dl = ROOT.TH1D(f"h_data_lumi_{key}", f"{title} - Run 5408 Data ({lumi_pb:.2f} pb^{{-1}});{xlabel};Events / bin", nbins, xmin, xmax)
        h_dl.Sumw2()
        for v in data_dict[key]:
            h_dl.Fill(v)

        h_rawl = ROOT.TH1D(f"h_mc_raw_lumi_{key}", f"{title} - PMU Raw ({lumi_pb:.2f} pb^{{-1}});{xlabel};Events / bin", nbins, xmin, xmax)
        h_rawl.Sumw2()
        for v, w in zip(mc_raw[key], mc_raw["weights"]):
            h_rawl.Fill(v, w * scale_mc)

        h_corrl = ROOT.TH1D(f"h_mc_corr_lumi_{key}", f"{title} - PMU Corrected ({lumi_pb:.2f} pb^{{-1}});{xlabel};Events / bin", nbins, xmin, xmax)
        h_corrl.Sumw2()
        for v, w in zip(mc_corr[key], mc_corr["weights"]):
            h_corrl.Fill(v, w * scale_mc)

        h_dl.Write()
        h_rawl.Write()
        h_corrl.Write()

        # Canvas Lumi
        dir_canvases_lumi.cd()
        c_lumi = ROOT.TCanvas(f"c_cmp_lumi_{key}", f"Luminosity Comparison - {title}", 900, 800)
        p1l = ROOT.TPad("p1l", "main pad", 0.0, 0.3, 1.0, 1.0)
        p2l = ROOT.TPad("p2l", "ratio pad", 0.0, 0.0, 1.0, 0.3)
        p1l.SetBottomMargin(0.02)
        p1l.SetGrid()
        p2l.SetTopMargin(0.05)
        p2l.SetBottomMargin(0.28)
        p2l.SetGrid()
        p1l.Draw()
        p2l.Draw()

        p1l.cd()
        max_yl = max(h_dl.GetMaximum(), h_rawl.GetMaximum(), h_corrl.GetMaximum()) * 1.35
        h_dl.SetMaximum(max_yl)
        h_dl.SetMinimum(0.0)
        h_dl.SetMarkerStyle(20)
        h_dl.SetMarkerSize(1.1)
        h_dl.SetMarkerColor(ROOT.kBlack)
        h_dl.SetLineColor(ROOT.kBlack)
        h_dl.Draw("E1")

        h_rawl.SetLineColor(ROOT.kRed + 1)
        h_rawl.SetLineStyle(2)
        h_rawl.SetLineWidth(2)
        h_rawl.Draw("HIST SAME")

        h_corrl.SetLineColor(ROOT.kAzure + 2)
        h_corrl.SetLineWidth(2)
        h_corrl.SetFillColorAlpha(ROOT.kAzure - 4, 0.35)
        h_corrl.Draw("HIST SAME")
        h_dl.Draw("E1 SAME")

        leg_l = ROOT.TLegend(0.55, 0.70, 0.88, 0.88)
        leg_l.SetBorderSize(0)
        leg_l.SetFillStyle(0)
        leg_l.AddEntry(h_dl, f"Run 5408 Data (N={len(data_dict[key])})", "lep")
        leg_l.AddEntry(h_rawl, f"PMU Raw ({lumi_pb:.2f} pb^{{-1}})", "l")
        leg_l.AddEntry(h_corrl, f"PMU Corrected ({lumi_pb:.2f} pb^{{-1}})", "lf")
        leg_l.Draw()

        p2l.cd()
        h_ratio_rawl = h_rawl.Clone(f"h_ratio_raw_lumi_{key}")
        h_ratio_rawl.Divide(h_dl)
        h_ratio_corrl = h_corrl.Clone(f"h_ratio_corr_lumi_{key}")
        h_ratio_corrl.Divide(h_dl)

        max_rl = max(h_ratio_rawl.GetMaximum(), h_ratio_corrl.GetMaximum()) * 1.25
        h_ratio_rawl.SetTitle(f";{xlabel};MC / Data")
        h_ratio_rawl.GetYaxis().SetRangeUser(0.0, max(min(max_rl, 10.0), 2.5))
        h_ratio_rawl.GetYaxis().SetNdivisions(505)
        h_ratio_rawl.GetYaxis().SetLabelSize(0.08)
        h_ratio_rawl.GetYaxis().SetTitleSize(0.09)
        h_ratio_rawl.GetYaxis().SetTitleOffset(0.5)
        h_ratio_rawl.GetXaxis().SetLabelSize(0.08)
        h_ratio_rawl.GetXaxis().SetTitleSize(0.09)
        h_ratio_rawl.SetMarkerStyle(25)
        h_ratio_rawl.SetMarkerSize(0.9)
        h_ratio_rawl.SetMarkerColor(ROOT.kRed + 1)
        h_ratio_rawl.SetLineColor(ROOT.kRed + 1)
        h_ratio_rawl.Draw("E1")

        h_ratio_corrl.SetMarkerStyle(20)
        h_ratio_corrl.SetMarkerSize(0.9)
        h_ratio_corrl.SetMarkerColor(ROOT.kAzure + 2)
        h_ratio_corrl.SetLineColor(ROOT.kAzure + 2)
        h_ratio_corrl.Draw("E1 SAME")
        c_lumi.Write()

    out_file.Close()
    print(f"Saved ROOT file with area and lumi normalized histograms and canvases: {out_root_path}")


def main():
    parser = argparse.ArgumentParser(description="Side-by-side Hit Occupancy comparison: PMU vs Run 5408 Data")
    parser.add_argument("--data-files", nargs="+", default=[f"/eos/experiment/sndlhc/convertedData/physics/2022/run_005408/sndsw_raw-{i:04d}.root" for i in range(8)],
                        help="Data root file(s) or patterns")
    parser.add_argument("--mc-file", default="/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP_2MuTrks_truth.root",
                        help="PMU MC simulation root file (with mctruth for mc_weight)")
    parser.add_argument("--max-events-data", type=int, default=800000, help="Max data events to process")
    parser.add_argument("--max-events-mc", type=int, default=100000, help="Max MC events to process")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for MC")
    parser.add_argument("-o", "--output", default="plots/pmu_vs_run5408.root", help="Output ROOT file")
    parser.add_argument("--plot-dir", default="plots", help="Directory to save figures")
    parser.add_argument("--force-reprocess", action="store_true", help="Force re-extraction of events from raw/MC files")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cache_npz = os.path.join(args.plot_dir, "cache_pmu_vs_run5408.npz")
    if os.path.exists(cache_npz) and not getattr(args, "force_reprocess", False):
        print(f"\nLoading cached extracted data from {cache_npz}...")
        loaded = np.load(cache_npz, allow_pickle=True)
        data_dict = {k: loaded[f"data_{k}"] for k in HIST_SPECS}
        mc_raw = {k: loaded[f"raw_{k}"] for k in HIST_SPECS}
        mc_raw["weights"] = loaded["raw_weights"]
        mc_corr = {k: loaded[f"corr_{k}"] for k in HIST_SPECS}
        mc_corr["weights"] = loaded["corr_weights"]
        n_data_scanned = int(loaded.get("n_data_scanned", 400000))
        n_mc_scanned = int(loaded.get("n_mc_scanned", 60000))
    else:
        # 1. Process Data
        data_dict, n_data_scanned = process_collision_data(args.data_files, max_events=args.max_events_data, n_workers=args.workers)

        # 2. Process MC (Raw and Corrected)
        mc_raw, mc_corr, n_mc_scanned = process_pmu_simulation(args.mc_file, max_events=args.max_events_mc, n_workers=args.workers)

        # Save cache
        save_dict = {}
        for k in HIST_SPECS:
            save_dict[f"data_{k}"] = data_dict[k]
            save_dict[f"raw_{k}"] = mc_raw[k]
            save_dict[f"corr_{k}"] = mc_corr[k]
        save_dict["raw_weights"] = mc_raw["weights"]
        save_dict["corr_weights"] = mc_corr["weights"]
        save_dict["n_data_scanned"] = np.array(n_data_scanned)
        save_dict["n_mc_scanned"] = np.array(n_mc_scanned)
        np.savez_compressed(cache_npz, **save_dict)
        print(f"Extracted events saved to cache: {cache_npz}")

    # Compute Luminosity Normalization Info
    # Exact ATLAS luminosity integrated over analyzed timestamp intervals:
    # Files 0000-0007 (50k events each): exactly 1.5307 pb^-1 from /eos/experiment/sndlhc/atlas_lumi/fill_008496.root
    TOTAL_RUN5408_EVENTS = 90174916
    TOTAL_RUN5408_LUMI_FB = 0.406569
    TOTAL_PMU_EVENTS = 1108804
    TOTAL_PMU_LUMI_FB = 1.0

    lumi_data_analyzed = 0.0015307  # fb^-1 (1.5307 pb^-1 exact ATLAS luminosity)
    lumi_mc_analyzed = TOTAL_PMU_LUMI_FB * (n_mc_scanned / TOTAL_PMU_EVENTS)
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

    # 3. Generate PNG Comparison Figures (Area Normalized)
    # Panel A: Subsystems
    subsystem_keys = ["veto_total", "us_total", "ds_total"]
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, subsystem_keys,
        "MuFilter & Veto Hit Multiplicity: Run 5408 Data vs. 2022 PMU (Front-Entering Muons: Veto Coinc + US5)",
        os.path.join(args.plot_dir, "pmu_vs_run5408_subsystems.png"),
        ncols=3, norm_mode="area",
    )

    # Panel B: Veto Planes
    veto_keys = ["veto_pl0", "veto_pl1"]
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, veto_keys,
        "Veto Planes Hit Multiplicity: Run 5408 Data vs. 2022 PMU Simulation (Front-Entering Muons)",
        os.path.join(args.plot_dir, "pmu_vs_run5408_veto_planes.png"),
        ncols=2, norm_mode="area",
    )

    # Panel C: US Planes (5 planes)
    us_keys = [f"us_pl{p}" for p in range(5)]
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, us_keys,
        "Upstream (US) Planes Hit Multiplicity: Run 5408 Data vs. 2022 PMU Simulation (Front-Entering Muons)",
        os.path.join(args.plot_dir, "pmu_vs_run5408_us_planes.png"),
        ncols=3, norm_mode="area",
    )

    # Panel D: DS Planes (Horizontal & Vertical)
    ds_keys = ["ds_h_pl0", "ds_h_pl1", "ds_h_pl2", "ds_v_pl0", "ds_v_pl1", "ds_v_pl2", "ds_v_pl3"]
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, ds_keys,
        "Downstream (DS) Planes Hit Multiplicity: Run 5408 Data vs. 2022 PMU Simulation (Front-Entering Muons)",
        os.path.join(args.plot_dir, "pmu_vs_run5408_ds_planes.png"),
        ncols=4, norm_mode="area",
    )

    # 4. Generate PNG Comparison Figures (Luminosity Scaled to Data Luminosity)
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, subsystem_keys,
        f"Subsystems Absolute Yields: Run 5408 Data vs. PMU (Lumi = {lumi_data_analyzed*1e3:.2f} pb$^{{-1}}$)",
        os.path.join(args.plot_dir, "pmu_vs_run5408_subsystems_lumi.png"),
        ncols=3, norm_mode="lumi_data", lumi_info=lumi_info,
    )
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, veto_keys,
        f"Veto Planes Absolute Yields: Run 5408 Data vs. PMU (Lumi = {lumi_data_analyzed*1e3:.2f} pb$^{{-1}}$)",
        os.path.join(args.plot_dir, "pmu_vs_run5408_veto_planes_lumi.png"),
        ncols=2, norm_mode="lumi_data", lumi_info=lumi_info,
    )
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, us_keys,
        f"US Planes Absolute Yields: Run 5408 Data vs. PMU (Lumi = {lumi_data_analyzed*1e3:.2f} pb$^{{-1}}$)",
        os.path.join(args.plot_dir, "pmu_vs_run5408_us_planes_lumi.png"),
        ncols=3, norm_mode="lumi_data", lumi_info=lumi_info,
    )
    plot_comparison_panels(
        data_dict, mc_raw, mc_corr, ds_keys,
        f"DS Planes Absolute Yields: Run 5408 Data vs. PMU (Lumi = {lumi_data_analyzed*1e3:.2f} pb$^{{-1}}$)",
        os.path.join(args.plot_dir, "pmu_vs_run5408_ds_planes_lumi.png"),
        ncols=4, norm_mode="lumi_data", lumi_info=lumi_info,
    )

    # 5. Save ROOT histograms and canvases
    save_root_output(data_dict, mc_raw, mc_corr, args.output, lumi_info=lumi_info)

    print("\n================================================================================")
    print("Comparison complete!")
    print(f"Output ROOT file: {args.output}")
    print(f"Plots saved to  : {args.plot_dir}/")
    print("================================================================================")


if __name__ == "__main__":
    main()
