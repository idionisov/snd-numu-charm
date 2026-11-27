#!/usr/bin/env python3
"""
scripts/qdc_calib/compare_pmu_4models.py
----------------------------------------
Side-by-side comparison of Hit Multiplicities between:
  1. Real Collision Data from Run 5408 (LHC Stable Beams, IP1, isolated triggers)
  2. PMU 2022 Simulation: Raw (E_th = 0 / raw digitized hits)
  3. PMU 2022 Simulation: Theoretical threshold (k = 1.0)
  4. PMU 2022 Simulation: Theoretical + channel-specific photoelectron reduction factor (k from CSV)
  5. PMU 2022 Simulation: Simple step activation (1.0 MeV for DS, 1.5 MeV for US/Veto)

Normalizes Data and PMU Simulation to the exact same luminosity:
  L_data = 1.5307 pb^-1 = 0.0015307 fb^-1 (Run 5408 integrated ATLAS luminosity).

Event Selection (Apples-to-Apples):
  - Front-entering through-going muon condition:
    Coincidence in Veto Plane 0 and Plane 1 (>=1 hit in each) + all 5 US planes hit.
  - Applied consistently across Data and each simulation model on its qualified hits.

Produces publication-quality PNGs with ratio panels and saves full ROOT canvases to:
  output/pmu_vs_run5408_4models.root
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

# Observable histogram specifications: (title, xlabel, nbins, xmin, xmax)
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
#ifndef FAST_DATA_PROCESSOR_4MODELS_H
#define FAST_DATA_PROCESSOR_4MODELS_H

#include "TTree.h"
#include "TClonesArray.h"
#include "MuFilterHit.h"
#include "SNDLHCEventHeader.h"
#include <vector>
#include <set>
#include <cmath>
#include <iostream>

struct DataHitsRecord4M {
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

struct DataExtractionResult4M {
    std::vector<DataHitsRecord4M> records;
    Long64_t n_scanned;
};

DataExtractionResult4M extract_run5408_muons_4m(TTree* tree, Long64_t max_entries) {
    DataExtractionResult4M result;
    result.n_scanned = 0;
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

        DataHitsRecord4M rec{};

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

        // Front-entering through-going muon condition:
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

    res = ROOT.extract_run5408_muons_4m(tree, max_ev)
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

    data_dict = {}
    for key in HIST_SPECS:
        data_dict[key] = np.array([r[key] for r in all_records], dtype=np.int32) if all_records else np.array([])
    return data_dict, total_scanned


# ==============================================================================
# MC Chunk Worker Routine (Evaluating All 4 Simulation Models)
# ==============================================================================
def _mc_chunk_worker_4models(args_tuple: Tuple[str, int, int, int]) -> Dict[str, Any]:
    """Worker function to process a slice of PMU MC events across 4 models."""
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

    # Model calculators
    calc_theo = mpe.MinPE(year=2022)
    calc_theo._k_factors = {} # Theoretical: all k = 1.0

    calc_calib = mpe.MinPE(year=2022) # Calibrated: k from CSV

    # Containers for the 4 models
    models = ["raw", "theo", "calib", "step"]
    store: Dict[str, Dict[str, List[float]]] = {
        m: {k: [] for k in HIST_SPECS} for m in models
    }
    for m in models:
        store[m]["weights"] = []

    end_ev = min(start_ev + n_ev, tree.GetEntries())

    for ev_idx in range(start_ev, end_ev):
        tree.GetEntry(ev_idx)

        tracks = getattr(tree, "MCTrack", None)
        wt = 1.0
        if tracks and tracks.GetEntriesFast() > 0:
            wt = float(tracks.At(0).GetWeight()) * 8e5

        mf_hits = getattr(tree, "Digi_MuFilterHits", None)
        pts_branch = getattr(tree, "MuFilterPoint", None)

        # ----------------------------------------------------------------------
        # Model 1: Raw Simulation (Digi_MuFilterHits)
        # ----------------------------------------------------------------------
        if mf_hits and mf_hits.GetEntriesFast() > 0:
            rec_raw = {k: 0 for k in HIST_SPECS}
            raw_us_planes = set()

            for j in range(mf_hits.GetEntriesFast()):
                h = mf_hits.At(j)
                if not h or not h.isValid():
                    continue
                sys_id = h.GetSystem()
                plane = h.GetPlane()
                is_vert = h.isVertical()

                if sys_id == 1:
                    if plane in (0, 1):
                        rec_raw["veto_total"] += 1
                        if plane == 0: rec_raw["veto_pl0"] += 1
                        elif plane == 1: rec_raw["veto_pl1"] += 1
                elif sys_id == 2:
                    rec_raw["us_total"] += 1
                    raw_us_planes.add(plane)
                    if plane == 0: rec_raw["us_pl0"] += 1
                    elif plane == 1: rec_raw["us_pl1"] += 1
                    elif plane == 2: rec_raw["us_pl2"] += 1
                    elif plane == 3: rec_raw["us_pl3"] += 1
                    elif plane == 4: rec_raw["us_pl4"] += 1
                elif sys_id == 3:
                    rec_raw["ds_total"] += 1
                    if not is_vert:
                        if plane == 0: rec_raw["ds_h_pl0"] += 1
                        elif plane == 1: rec_raw["ds_h_pl1"] += 1
                        elif plane == 2: rec_raw["ds_h_pl2"] += 1
                    else:
                        if plane == 0: rec_raw["ds_v_pl0"] += 1
                        elif plane == 1: rec_raw["ds_v_pl1"] += 1
                        elif plane == 2: rec_raw["ds_v_pl2"] += 1
                        elif plane == 3: rec_raw["ds_v_pl3"] += 1

            if len(raw_us_planes) == 5 and rec_raw["veto_pl0"] >= 1 and rec_raw["veto_pl1"] >= 1:
                for k in HIST_SPECS:
                    store["raw"][k].append(rec_raw[k])
                store["raw"]["weights"].append(wt)

        # ----------------------------------------------------------------------
        # Models 2, 3, 4: Evaluated from MuFilterPoint clusters
        # ----------------------------------------------------------------------
        if pts_branch and pts_branch.GetEntriesFast() > 0:
            grouped = mpe.group_mcpoints_by_bar(pts_branch)

            rec_theo = {k: 0 for k in HIST_SPECS}
            us_pl_theo = set()

            rec_calib = {k: 0 for k in HIST_SPECS}
            us_pl_calib = set()

            rec_step = {k: 0 for k in HIST_SPECS}
            us_pl_step = set()

            for det_id, clusters in grouped.items():
                if not calc_calib.is_valid_det_id(det_id):
                    continue
                sys_id, plane, bar = calc_calib.decode_detector_id(det_id)
                is_vert = calc_calib.is_vertical(det_id)

                for pts in clusters:
                    # Model 2: Theoretical (k=1.0)
                    th_theo = calc_theo.get_lowest_trigger_threshold(pts, det_id=det_id, include_mirror=True)
                    if th_theo["current_hit_qualified"]:
                        if sys_id == 1 and plane in (0, 1):
                            rec_theo["veto_total"] += 1
                            if plane == 0: rec_theo["veto_pl0"] += 1
                            elif plane == 1: rec_theo["veto_pl1"] += 1
                        elif sys_id == 2:
                            rec_theo["us_total"] += 1
                            us_pl_theo.add(plane)
                            if plane == 0: rec_theo["us_pl0"] += 1
                            elif plane == 1: rec_theo["us_pl1"] += 1
                            elif plane == 2: rec_theo["us_pl2"] += 1
                            elif plane == 3: rec_theo["us_pl3"] += 1
                            elif plane == 4: rec_theo["us_pl4"] += 1
                        elif sys_id == 3:
                            rec_theo["ds_total"] += 1
                            if not is_vert:
                                if plane == 0: rec_theo["ds_h_pl0"] += 1
                                elif plane == 1: rec_theo["ds_h_pl1"] += 1
                                elif plane == 2: rec_theo["ds_h_pl2"] += 1
                            else:
                                if plane == 0: rec_theo["ds_v_pl0"] += 1
                                elif plane == 1: rec_theo["ds_v_pl1"] += 1
                                elif plane == 2: rec_theo["ds_v_pl2"] += 1
                                elif plane == 3: rec_theo["ds_v_pl3"] += 1

                    # Model 3: Calibrated (k from CSV)
                    th_calib = calc_calib.get_lowest_trigger_threshold(pts, det_id=det_id, include_mirror=True)
                    if th_calib["current_hit_qualified"]:
                        if sys_id == 1 and plane in (0, 1):
                            rec_calib["veto_total"] += 1
                            if plane == 0: rec_calib["veto_pl0"] += 1
                            elif plane == 1: rec_calib["veto_pl1"] += 1
                        elif sys_id == 2:
                            rec_calib["us_total"] += 1
                            us_pl_calib.add(plane)
                            if plane == 0: rec_calib["us_pl0"] += 1
                            elif plane == 1: rec_calib["us_pl1"] += 1
                            elif plane == 2: rec_calib["us_pl2"] += 1
                            elif plane == 3: rec_calib["us_pl3"] += 1
                            elif plane == 4: rec_calib["us_pl4"] += 1
                        elif sys_id == 3:
                            rec_calib["ds_total"] += 1
                            if not is_vert:
                                if plane == 0: rec_calib["ds_h_pl0"] += 1
                                elif plane == 1: rec_calib["ds_h_pl1"] += 1
                                elif plane == 2: rec_calib["ds_h_pl2"] += 1
                            else:
                                if plane == 0: rec_calib["ds_v_pl0"] += 1
                                elif plane == 1: rec_calib["ds_v_pl1"] += 1
                                elif plane == 2: rec_calib["ds_v_pl2"] += 1
                                elif plane == 3: rec_calib["ds_v_pl3"] += 1

                    # Model 4: Simple step activation (1.0 MeV for DS, 1.5 MeV for US/Veto)
                    tot_eloss_mev = sum(p.GetEnergyLoss() * 1000.0 for p in pts)
                    step_th = 1.0 if sys_id == 3 else 1.5
                    if tot_eloss_mev > step_th:
                        if sys_id == 1 and plane in (0, 1):
                            rec_step["veto_total"] += 1
                            if plane == 0: rec_step["veto_pl0"] += 1
                            elif plane == 1: rec_step["veto_pl1"] += 1
                        elif sys_id == 2:
                            rec_step["us_total"] += 1
                            us_pl_step.add(plane)
                            if plane == 0: rec_step["us_pl0"] += 1
                            elif plane == 1: rec_step["us_pl1"] += 1
                            elif plane == 2: rec_step["us_pl2"] += 1
                            elif plane == 3: rec_step["us_pl3"] += 1
                            elif plane == 4: rec_step["us_pl4"] += 1
                        elif sys_id == 3:
                            rec_step["ds_total"] += 1
                            if not is_vert:
                                if plane == 0: rec_step["ds_h_pl0"] += 1
                                elif plane == 1: rec_step["ds_h_pl1"] += 1
                                elif plane == 2: rec_step["ds_h_pl2"] += 1
                            else:
                                if plane == 0: rec_step["ds_v_pl0"] += 1
                                elif plane == 1: rec_step["ds_v_pl1"] += 1
                                elif plane == 2: rec_step["ds_v_pl2"] += 1
                                elif plane == 3: rec_step["ds_v_pl3"] += 1

            # Selection for Model 2 (Theo)
            if len(us_pl_theo) == 5 and rec_theo["veto_pl0"] >= 1 and rec_theo["veto_pl1"] >= 1:
                for k in HIST_SPECS:
                    store["theo"][k].append(rec_theo[k])
                store["theo"]["weights"].append(wt)

            # Selection for Model 3 (Calib)
            if len(us_pl_calib) == 5 and rec_calib["veto_pl0"] >= 1 and rec_calib["veto_pl1"] >= 1:
                for k in HIST_SPECS:
                    store["calib"][k].append(rec_calib[k])
                store["calib"]["weights"].append(wt)

            # Selection for Model 4 (Step)
            if len(us_pl_step) == 5 and rec_step["veto_pl0"] >= 1 and rec_step["veto_pl1"] >= 1:
                for k in HIST_SPECS:
                    store["step"][k].append(rec_step[k])
                store["step"]["weights"].append(wt)

    f.Close()
    return {"store": store, "n_scanned": end_ev - start_ev}


def process_pmu_simulation_4models(
    mc_file: str, max_events: int = 60000, n_workers: int = 8
) -> Tuple[Dict[str, Dict[str, np.ndarray]], int]:
    """Process PMU Monte Carlo across all 4 models in parallel."""
    print(f"\n[MC] Processing 2022 PMU Simulation ({max_events} events, {n_workers} workers, 4 models)...")
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
        results = pool.map(_mc_chunk_worker_4models, tasks)

    models = ["raw", "theo", "calib", "step"]
    merged: Dict[str, Dict[str, Any]] = {
        m: {k: [] for k in HIST_SPECS} for m in models
    }
    for m in models:
        merged[m]["weights"] = []
    total_mc_scanned = 0

    for res in results:
        if not res:
            continue
        st = res["store"]
        for m in models:
            for k in HIST_SPECS:
                merged[m][k].extend(st[m][k])
            merged[m]["weights"].extend(st[m]["weights"])
        total_mc_scanned += res.get("n_scanned", 0)

    for m in models:
        for k in HIST_SPECS:
            merged[m][k] = np.array(merged[m][k], dtype=np.float64)
        merged[m]["weights"] = np.array(merged[m]["weights"], dtype=np.float64)

    elapsed = time.time() - t0
    print(f"[MC] Completed in {elapsed:.1f}s (Scanned {total_mc_scanned:,} events):")
    for m in models:
        print(f"     Model '{m:<5}': {len(merged[m]['weights'])} events (Sum of weights: {np.sum(merged[m]['weights']):.4f})")

    return merged, total_mc_scanned


# ==============================================================================
# Plotting and Multi-Panel Comparison Generation
# ==============================================================================
def plot_4models_comparison(
    data_dict: Dict[str, np.ndarray],
    mc_models: Dict[str, Dict[str, np.ndarray]],
    keys: List[str],
    fig_title: str,
    out_png: str,
    ncols: int = 3,
    lumi_info: Optional[Dict[str, float]] = None,
):
    """Generate multi-panel comparison figure comparing Data and 4 simulation models."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.0
    nrows = (len(keys) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows * 2, ncols,
        figsize=(5.8 * ncols, 4.0 * nrows),
        gridspec_kw={"height_ratios": [3, 1.2] * nrows},
        constrained_layout=True,
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif nrows == 1:
        axes = np.array([axes[0, :], axes[1, :]])
    elif ncols == 1:
        axes = axes.reshape(nrows * 2, 1)

    scale_mc = lumi_info["scale_mc_to_data"] if lumi_info else 1.0
    lumi_pb = (lumi_info["lumi_data"] * 1e3) if lumi_info else 1.53

    model_styles = {
        "raw":   {"label": "Raw (Eth=0)", "color": "#1f77b4", "ls": ":", "lw": 1.6, "marker": "s"},
        "theo":  {"label": "Theoretical (k=1.0)", "color": "#9467bd", "ls": "--", "lw": 1.8, "marker": "D"},
        "calib": {"label": "Theoretical + k(SiPM)", "color": "#d62728", "ls": "-", "lw": 2.0, "marker": "o"},
        "step":  {"label": "Simple Step (1 / 1.5 MeV)", "color": "#2ca02c", "ls": "-.", "lw": 1.8, "marker": "^"},
    }

    for idx, key in enumerate(keys):
        row = (idx // ncols) * 2
        col = idx % ncols
        ax_main = axes[row, col]
        ax_ratio = axes[row + 1, col]

        title, xlabel, nbins, xmin, xmax = HIST_SPECS[key]
        bins = np.linspace(xmin, xmax, nbins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # 1. Collision Data
        data_vals = data_dict[key]
        n_data = len(data_vals)
        h_data, _ = np.histogram(data_vals, bins=bins)
        y_data = h_data.astype(float)
        y_data_err = np.sqrt(h_data)

        # 2. Simulation Models
        y_sim = {}
        y_sim_err = {}
        for m in ["raw", "theo", "calib", "step"]:
            vals = mc_models[m][key]
            wts = mc_models[m]["weights"]
            h_m, _ = np.histogram(vals, bins=bins, weights=wts)
            h_m_w2, _ = np.histogram(vals, bins=bins, weights=wts**2)
            y_sim[m] = h_m * scale_mc
            y_sim_err[m] = np.sqrt(h_m_w2) * scale_mc

            # Plot stepped simulation line
            st = model_styles[m]
            ax_main.step(
                bins, np.append(y_sim[m], y_sim[m][-1]), where="post",
                color=st["color"], lw=st["lw"], ls=st["ls"], label=st["label"]
            )
            if m == "calib":
                ax_main.fill_between(bins, np.append(y_sim[m], y_sim[m][-1]), step="post", color="#ff7f7f", alpha=0.18)

        # Data markers
        mask_d = y_data > 0
        ax_main.errorbar(
            bin_centers[mask_d], y_data[mask_d], yerr=y_data_err[mask_d],
            fmt="o", color="black", ms=5.0, capsize=2, lw=1.2, label=f"Run 5408 Data (N={n_data})"
        )

        ax_main.set_title(title, fontsize=11, fontweight="bold")
        ax_main.set_ylabel(f"Events / bin (Data $\\mathcal{{L}} = {lumi_pb:.2f}$ pb$^{{-1}}$)", fontsize=9)
        ax_main.set_xlim(xmin, xmax)
        max_val = max(np.max(y_data), max(np.max(y_sim[m]) for m in y_sim)) if len(y_data) > 0 else 1.0
        ax_main.set_ylim(0, max_val * 1.4 if max_val > 0 else 1.0)
        ax_main.grid(True, linestyle=":", alpha=0.6)
        ax_main.legend(loc="upper right", fontsize=7.2, framealpha=0.9)
        ax_main.xaxis.set_minor_locator(AutoMinorLocator())
        ax_main.yaxis.set_minor_locator(AutoMinorLocator())

        # Ratio panel: Model / Data
        ax_ratio.axhline(1.0, color="gray", linestyle="--", lw=1.0)
        offsets = {"raw": -0.15, "theo": -0.05, "calib": 0.05, "step": 0.15}

        for m in ["raw", "theo", "calib", "step"]:
            st = model_styles[m]
            mask_r = (y_data > 0) & (y_sim[m] > 0)
            if np.any(mask_r):
                r_val = y_sim[m][mask_r] / y_data[mask_r]
                r_err = r_val * np.sqrt((y_data_err[mask_r] / y_data[mask_r])**2 + (y_sim_err[m][mask_r] / y_sim[m][mask_r])**2)
                ax_ratio.errorbar(
                    bin_centers[mask_r] + offsets[m] * (bin_centers[1] - bin_centers[0]),
                    r_val, yerr=r_err,
                    fmt=st["marker"], color=st["color"], mfc="none" if m == "raw" else st["color"],
                    ms=4.0, capsize=2, lw=1.0, label=f"{m} / Data"
                )

        ax_ratio.set_xlim(xmin, xmax)
        ax_ratio.set_ylim(0.2, 2.2)
        ax_ratio.set_xlabel(xlabel, fontsize=10)
        ax_ratio.set_ylabel("Sim / Data", fontsize=9)
        ax_ratio.grid(True, linestyle=":", alpha=0.5)
        ax_ratio.xaxis.set_minor_locator(AutoMinorLocator())

    # Hide unused subplots
    for idx in range(len(keys), nrows * ncols):
        row = (idx // ncols) * 2
        col = idx % ncols
        axes[row, col].set_visible(False)
        axes[row + 1, col].set_visible(False)

    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved figure: {out_png}")


# ==============================================================================
# Main Driver
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Side-by-side comparison across 4 threshold models")
    parser.add_argument("--data-files", nargs="+", default=[
        "/eos/experiment/sndlhc/convertedData/physics/2022/run_005408/sndsw_raw-0000.root"
    ])
    parser.add_argument("--mc-file", default="/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP.root")
    parser.add_argument("--max-events-data", type=int, default=600000)
    parser.add_argument("--max-events-mc", type=int, default=60000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="output/pmu_vs_run5408_4models.root")
    parser.add_argument("--plot-dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Resolve wildcard expansion for data files
    data_files = []
    for p in args.data_files:
        matches = glob.glob(p)
        if matches:
            data_files.extend(sorted(matches))
        else:
            data_files.append(p)

    # 1. Process Collision Data
    data_dict, total_data_scanned = process_collision_data(data_files, max_events=args.max_events_data, n_workers=args.workers)

    # 2. Process PMU Simulation across all 4 models
    mc_models, total_mc_scanned = process_pmu_simulation_4models(args.mc_file, max_events=args.max_events_mc, n_workers=args.workers)

    # 3. Exact Luminosity Calibration
    # Total Run 5408 integrated luminosity: 1.5307 pb^-1 = 0.0015307 fb^-1
    L_RUN5408_TOTAL_FB = 0.0015307
    N_DATA_TOTAL_RUN5408 = 2465355
    data_frac = min(1.0, float(total_data_scanned) / float(N_DATA_TOTAL_RUN5408))
    lumi_data_analyzed = L_RUN5408_TOTAL_FB * data_frac

    # PMU MC file total cross-section luminosity: 683,610 fb^-1
    PMU_FILE_TOTAL_EVENTS = 100000
    PMU_FILE_TOTAL_LUMI_FB = 683610.0
    mc_frac = min(1.0, float(total_mc_scanned) / float(PMU_FILE_TOTAL_EVENTS))
    lumi_mc_analyzed = PMU_FILE_TOTAL_LUMI_FB * mc_frac
    scale_mc_to_data = lumi_data_analyzed / lumi_mc_analyzed

    lumi_info = {
        "lumi_data": lumi_data_analyzed,
        "lumi_mc": lumi_mc_analyzed,
        "scale_mc_to_data": scale_mc_to_data,
    }

    print("\n" + "=" * 80)
    print(f"LUMINOSITY SCALING SUMMARY:")
    print(f"  Run 5408 Data Analyzed   : {total_data_scanned:,} / {N_DATA_TOTAL_RUN5408:,} events ({data_frac*100:.2f}%)")
    print(f"  Exact Data Lumi Analyzed : {lumi_data_analyzed*1e3:.4f} pb^-1 ({lumi_data_analyzed:.7f} fb^-1)")
    print(f"  PMU MC Events Analyzed   : {total_mc_scanned:,} / {PMU_FILE_TOTAL_EVENTS:,} events ({mc_frac*100:.2f}%)")
    print(f"  PMU Sample Lumi Analyzed : {lumi_mc_analyzed:,.1f} fb^-1")
    print(f"  Scale Factor (MC -> Data): {scale_mc_to_data:.8e}")
    print("=" * 80)

    # 4. Generate Multi-Panel Comparison Plots
    groups = [
        ("Subsystems (Veto, US, DS)", ["veto_total", "us_total", "ds_total"], "pmu_vs_run5408_subsystems_4models.png", 3),
        ("Veto Planes (Planes 0 & 1)", ["veto_pl0", "veto_pl1"], "pmu_vs_run5408_veto_planes_4models.png", 2),
        ("Upstream Calorimeter Planes (US 0 to 4)", ["us_pl0", "us_pl1", "us_pl2", "us_pl3", "us_pl4"], "pmu_vs_run5408_us_planes_4models.png", 3),
        ("Downstream Muon Filter Planes (DS Stations 0 to 3)", [
            "ds_h_pl0", "ds_v_pl0", "ds_h_pl1", "ds_v_pl1", "ds_h_pl2", "ds_v_pl2", "ds_v_pl3"
        ], "pmu_vs_run5408_ds_planes_4models.png", 4),
    ]

    for title, keys, fname, ncols in groups:
        out_png = os.path.join(args.plot_dir, fname)
        plot_4models_comparison(data_dict, mc_models, keys, title, out_png, ncols=ncols, lumi_info=lumi_info)
        # Copy to brain artifact directory if exists
        art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"
        if os.path.exists(art_dir):
            os.system(f"cp {out_png} {os.path.join(art_dir, fname)}")

    # 5. Save ROOT Canvases and Histograms to output ROOT File
    print(f"\n[ROOT] Writing canvases and histograms to {args.output}...")
    rfile = ROOT.TFile.Open(args.output, "RECREATE")

    canvas_configs = [
        ("c_subsystems", "Subsystem Multiplicities (4 Models)", ["veto_total", "us_total", "ds_total"], 3, 1),
        ("c_veto_planes", "Veto Planes Multiplicities (4 Models)", ["veto_pl0", "veto_pl1"], 2, 1),
        ("c_us_planes", "US Planes Multiplicities (4 Models)", ["us_pl0", "us_pl1", "us_pl2", "us_pl3", "us_pl4"], 3, 2),
        ("c_ds_planes", "DS Planes Multiplicities (4 Models)", [
            "ds_h_pl0", "ds_v_pl0", "ds_h_pl1", "ds_v_pl1", "ds_h_pl2", "ds_v_pl2", "ds_v_pl3"
        ], 4, 2),
    ]

    canvas_store = []
    for cname, ctitle, keys, nx, ny in canvas_configs:
        c = ROOT.TCanvas(cname, ctitle, 400 * nx, 350 * ny)
        c.Divide(nx, ny)

        for p_idx, k in enumerate(keys):
            c.cd(p_idx + 1)
            title, xlabel, nbins, xmin, xmax = HIST_SPECS[k]

            # Create TH1D histograms
            h_data = ROOT.TH1D(f"h_data_{k}", f"{title};{xlabel};Events / bin", nbins, xmin, xmax)
            h_raw = ROOT.TH1D(f"h_mc_raw_{k}", f"{title} (Raw);{xlabel};Events / bin", nbins, xmin, xmax)
            h_theo = ROOT.TH1D(f"h_mc_theo_{k}", f"{title} (Theo);{xlabel};Events / bin", nbins, xmin, xmax)
            h_calib = ROOT.TH1D(f"h_mc_calib_{k}", f"{title} (Calib);{xlabel};Events / bin", nbins, xmin, xmax)
            h_step = ROOT.TH1D(f"h_mc_step_{k}", f"{title} (Step);{xlabel};Events / bin", nbins, xmin, xmax)

            # Fill data
            for v in data_dict[k]:
                h_data.Fill(float(v))

            # Fill MC with scaled weights
            for v, w in zip(mc_models["raw"][k], mc_models["raw"]["weights"]):
                h_raw.Fill(float(v), float(w) * scale_mc_to_data)
            for v, w in zip(mc_models["theo"][k], mc_models["theo"]["weights"]):
                h_theo.Fill(float(v), float(w) * scale_mc_to_data)
            for v, w in zip(mc_models["calib"][k], mc_models["calib"]["weights"]):
                h_calib.Fill(float(v), float(w) * scale_mc_to_data)
            for v, w in zip(mc_models["step"][k], mc_models["step"]["weights"]):
                h_step.Fill(float(v), float(w) * scale_mc_to_data)

            # Styling
            h_data.SetLineColor(ROOT.kBlack)
            h_data.SetMarkerStyle(20)
            h_data.SetMarkerSize(0.8)

            h_raw.SetLineColor(ROOT.kBlue + 1)
            h_raw.SetLineStyle(3)
            h_raw.SetLineWidth(2)

            h_theo.SetLineColor(ROOT.kMagenta + 2)
            h_theo.SetLineStyle(2)
            h_theo.SetLineWidth(2)

            h_calib.SetLineColor(ROOT.kRed + 1)
            h_calib.SetLineWidth(2)

            h_step.SetLineColor(ROOT.kGreen + 2)
            h_step.SetLineStyle(4)
            h_step.SetLineWidth(2)

            max_y = max(h_data.GetMaximum(), h_raw.GetMaximum(), h_theo.GetMaximum(), h_calib.GetMaximum(), h_step.GetMaximum())
            h_data.SetMaximum(max_y * 1.35)

            h_data.Draw("E P")
            h_raw.Draw("HIST SAME")
            h_theo.Draw("HIST SAME")
            h_calib.Draw("HIST SAME")
            h_step.Draw("HIST SAME")

            leg = ROOT.TLegend(0.55, 0.65, 0.88, 0.88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.AddEntry(h_data, "Data (Run 5408)", "ep")
            leg.AddEntry(h_raw, "Sim Raw", "l")
            leg.AddEntry(h_theo, "Sim Theo (k=1)", "l")
            leg.AddEntry(h_calib, "Sim Theo+k", "l")
            leg.AddEntry(h_step, "Sim Step (1/1.5 MeV)", "l")
            leg.Draw()

            # Store in canvas object to prevent garbage collection
            c.store = getattr(c, "store", [])
            c.store.extend([h_data, h_raw, h_theo, h_calib, h_step, leg])

            h_data.Write()
            h_raw.Write()
            h_theo.Write()
            h_calib.Write()
            h_step.Write()

        c.Write()
        canvas_store.append(c)

    rfile.Close()
    print(f"Successfully saved all canvases and histograms to {args.output}")


if __name__ == "__main__":
    main()
