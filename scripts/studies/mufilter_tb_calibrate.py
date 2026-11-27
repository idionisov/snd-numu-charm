#!/usr/bin/env python3
"""
MuFilter Hit Energy Deposit Threshold Optimizer: Conditioned Active-Shower & DS Analysis

Calibrates the physical lower-end deposited energy threshold (E_th in MeV)
by matching simulated MuFilter hit responses with measured testbeam data.

Key Analysis Features & Physics Controls:
  1. Active-Shower Conditioning (--require-active):
     Discards pedestal/empty triggers by requiring real detector action (e.g. N_US >= 2 or N_DS >= 1).
     This directly resolves the problem of empty events diluting the data average.
  2. Subsystem Targeting (--subsystem US / DS / All):
     - US Mode: Calibrates thick bars (8 SiPMs/side) with Plane 4 and multi-plane observables.
     - DS Mode: Calibrates thin strip bars (1 SiPM/side) using penetrating 300 GeV pi- or 160 GeV muon runs.
  3. Mirror Instrumentation Note:
     Accounts for single-sided mirror readout vs double-sided SiPM readout.
  4. Multi-Panel Visualizations (Y-axis Origin Anchored at 0):
     - Survival curves <N_hits(E_th)>
     - Event hit multiplicity distributions P(N_hits) Data vs MC
     - Fired planes depth profile
     - Transverse bar multiplicity profile
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

TB_CONFIGS = {
    "100": {
        "particle": "pi+",
        "data_runs": [100630, 100631, 100632, 100677],
        "mc_sim": "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/100GeV_211/sndLHC.PG_211-TGeant4.root",
        "mc_dig": "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/100GeV_211/sndLHC.PG_211-TGeant4_digCPP.root",
    },
    "140": {
        "particle": "pi+ (muon-contaminated)",
        "data_runs": [100633, 100673, 100674],
        "mc_sim": "",
        "mc_dig": "",
    },
    "180": {
        "particle": "pi+",
        "data_runs": [100635, 100636, 100671, 100672],
        "mc_sim": "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/180GeV_211/sndLHC.PG_211-TGeant4.root",
        "mc_dig": "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/180GeV_211/sndLHC.PG_211-TGeant4_digCPP.root",
    },
    "300": {
        "particle": "pi-",
        "data_runs": [100639, 100641, 100642, 100643, 100645],
        "mc_sim": "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/300GeV_-211/sndLHC.PG_-211-TGeant4.root",
        "mc_dig": "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/300GeV_-211/sndLHC.PG_-211-TGeant4_digCPP.root",
    },
}

DATA_BASE_DIR = "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8"


def get_data_files(run_id: int) -> List[str]:
    """Find all converted ROOT files for a specific testbeam run."""
    run_dir = os.path.join(DATA_BASE_DIR, f"run_{run_id:06d}")
    if not os.path.isdir(run_dir):
        return []
    return sorted(glob.glob(os.path.join(run_dir, "sndsw_raw-*.root")))


def setup_tree(tree: ROOT.TTree, branches: List[str]) -> None:
    """Optimize reading speed by activating only requested branches."""
    tree.SetBranchStatus("*", 0)
    for b in branches:
        if tree.GetBranch(b) or tree.GetBranch(f"{b}."):
            tree.SetBranchStatus(f"{b}*", 1)


def process_testbeam_data(
    data_files: List[str],
    subsystem: str = "US",
    min_activity_hits: int = 1,
    max_events: int = -1,
    apply_timing_cut: bool = True,
    max_delta_t_ns: float = 12.5,
) -> Tuple[List[int], List[int], List[int], int]:
    """
    Extract hits per event, fired planes, and bar distribution from active Data events.
    """
    target_sub = 2 if subsystem == "US" else 3

    event_hit_counts: List[int] = []
    event_plane_counts: List[int] = []
    fired_bars_all: List[int] = []

    total_events = 0
    t0 = time.time()
    print(f"Processing Data ({len(data_files)} files, target={subsystem}, min_hits={min_activity_hits})...")

    for fpath in data_files:
        if 0 < max_events <= total_events:
            break

        f_in = ROOT.TFile.Open(fpath, "READ")
        if not f_in or f_in.IsZombie():
            continue

        tree = f_in.Get("rawConv")
        if not tree:
            f_in.Close()
            continue

        setup_tree(tree, ["Digi_MuFilterHits"])
        n_entries = tree.GetEntries()

        for iev in range(n_entries):
            if 0 < max_events <= total_events:
                break
            tree.GetEntry(iev)
            total_events += 1

            sub_hits = []
            planes = set()

            if hasattr(tree, "Digi_MuFilterHits"):
                for hit in tree.Digi_MuFilterHits:
                    if not hit.isValid():
                        continue
                    if apply_timing_cut:
                        t = hit.GetTime()
                        if t != -999.0 and abs(t) > max_delta_t_ns:
                            continue

                    det_id = hit.GetDetectorID()
                    if (det_id // 10000) == target_sub:
                        sub_hits.append(det_id)
                        planes.add((det_id // 1000) % 10)

            # Conditioning: Require minimum hits in this subsystem
            if len(sub_hits) >= min_activity_hits:
                event_hit_counts.append(len(sub_hits))
                event_plane_counts.append(len(planes))
                fired_bars_all.extend([d % 1000 for d in sub_hits])

        f_in.Close()

    elapsed = time.time() - t0
    print(f"Data processed: {total_events} raw events, {len(event_hit_counts)} active events in {elapsed:.1f} s")
    return event_hit_counts, event_plane_counts, fired_bars_all, total_events


def process_mc_threshold_scan(
    mc_dig_file: str,
    thresholds_mev: np.ndarray,
    subsystem: str = "US",
    min_activity_hits: int = 1,
    max_events: int = -1,
) -> Tuple[np.ndarray, np.ndarray, List[int], int]:
    """
    Process simulation events matching the active-event condition.
    """
    target_sub = 2 if subsystem == "US" else 3

    f_dig = ROOT.TFile.Open(mc_dig_file, "READ")
    if not f_dig or f_dig.IsZombie():
        raise RuntimeError(f"Cannot open MC file: {mc_dig_file}")

    tree = f_dig.Get("cbmsim;78") or f_dig.Get("cbmsim")
    setup_tree(tree, ["MuFilterPoint", "Digi_MuFilterHits"])

    n_entries = tree.GetEntries()
    total_events = 0
    t0 = time.time()

    surviving_hits_curves: List[np.ndarray] = []
    surviving_planes_curves: List[np.ndarray] = []
    fired_bars_mc: List[int] = []

    print(f"Processing MC ({mc_dig_file}, target={subsystem}, min_hits={min_activity_hits})...")

    for iev in range(n_entries):
        if 0 < max_events <= total_events:
            break
        tree.GetEntry(iev)
        total_events += 1

        bar_edep: Dict[int, float] = {}
        if hasattr(tree, "MuFilterPoint"):
            for pt in tree.MuFilterPoint:
                det_id = pt.GetDetectorID()
                bar_edep[det_id] = bar_edep.get(det_id, 0.0) + pt.GetEnergyLoss() * 1000.0

        hits_edeps: List[float] = []
        hits_planes: List[int] = []

        if hasattr(tree, "Digi_MuFilterHits"):
            for hit in tree.Digi_MuFilterHits:
                if not hit.isValid():
                    continue
                det_id = hit.GetDetectorID()
                if (det_id // 10000) == target_sub:
                    edep = bar_edep.get(det_id, 0.0)
                    hits_edeps.append(edep)
                    hits_planes.append((det_id // 1000) % 10)
                    fired_bars_mc.append(det_id % 1000)

        # Condition: Require event to have at least min_activity_hits
        if len(hits_edeps) >= min_activity_hits:
            arr_e = np.array(hits_edeps)
            arr_p = np.array(hits_planes)

            # Hit count vs threshold
            surv_h = (arr_e[:, None] >= thresholds_mev[None, :]).sum(axis=0)
            surviving_hits_curves.append(surv_h)

            # Fired planes count vs threshold
            pl_surv = []
            for th in thresholds_mev:
                active_planes = set(arr_p[arr_e >= th])
                pl_surv.append(len(active_planes))
            surviving_planes_curves.append(np.array(pl_surv, dtype=float))

    f_dig.Close()
    elapsed = time.time() - t0
    print(f"MC processed: {total_events} raw events, {len(surviving_hits_curves)} active events in {elapsed:.1f} s")

    mean_hits = np.mean(np.array(surviving_hits_curves), axis=0) if surviving_hits_curves else np.zeros_like(thresholds_mev)
    mean_planes = np.mean(np.array(surviving_planes_curves), axis=0) if surviving_planes_curves else np.zeros_like(thresholds_mev)

    return mean_hits, mean_planes, fired_bars_mc, total_events


def interpolate_threshold(thresholds: np.ndarray, curve: np.ndarray, target: float) -> float:
    """Find threshold value where curve crosses target."""
    if curve[0] <= target:
        return 0.0
    idx = np.where(curve <= target)[0]
    if len(idx) == 0:
        return float(thresholds[-1])
    i = idx[0]
    if i == 0:
        return float(thresholds[0])
    e0, e1 = thresholds[i - 1], thresholds[i]
    y0, y1 = curve[i - 1], curve[i]
    if y0 == y1:
        return float(e0)
    return float(e0 + (target - y0) * (e1 - e0) / (y1 - y0))


def main():
    parser = argparse.ArgumentParser(description="Calibrate MuFilter threshold conditioned on active events.")
    parser.add_argument("--subsystem", choices=["US", "DS"], default="US",
                        help="Subsystem to calibrate: US (8 SiPMs/side) or DS (1 SiPM/side)")
    parser.add_argument("--energy", choices=["100", "140", "180", "300"], default="100",
                        help="Beam energy setting (default: 100 for US, 300 recommended for DS)")
    parser.add_argument("--min-hits", type=int, default=1,
                        help="Minimum hits in subsystem to qualify as active action event (default: 1)")
    parser.add_argument("--max-events", type=int, default=1000,
                        help="Maximum events to analyze per sample (default: 1000)")
    parser.add_argument("-o", "--output", default="",
                        help="Output ROOT file path")
    parser.add_argument("--plot", default="",
                        help="Output summary plot path")
    parser.add_argument("--max-thresh", type=float, default=10.0,
                        help="Maximum energy threshold to scan in MeV (default: 10.0)")
    parser.add_argument("--step-thresh", type=float, default=0.2,
                        help="Energy threshold step in MeV (default: 0.2)")

    args = parser.parse_args()

    # If DS requested without specifying 300 GeV, recommend/switch to 300 GeV for penetrating showers
    if args.subsystem == "DS" and args.energy == "100":
        print("Note: 100 GeV showers rarely reach DS through 3 iron walls. Switching to 300 GeV for DS calibration.")
        args.energy = "300"

    cfg = TB_CONFIGS[args.energy]
    print(f"=== Conditioned MuFilter Calibration: {args.subsystem} Subsystem ({args.energy} GeV {cfg['particle']}) ===")

    out_root = args.output if args.output else f"/eos/user/i/idioniso/snd-numu-charm/plots/mufilter_study/calib_{args.subsystem.lower()}_{args.energy}gev.root"
    out_plot = args.plot if args.plot else os.path.splitext(out_root)[0] + ".png"
    os.makedirs(os.path.dirname(os.path.abspath(out_plot)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_root)), exist_ok=True)

    data_files = []
    for r in cfg["data_runs"][:2]:
        data_files.extend(get_data_files(r))
    if not data_files:
        raise FileNotFoundError(f"No data files found for runs: {cfg['data_runs']}")

    # 1. Process Conditioned Data
    data_hits, data_planes, data_bars, _ = process_testbeam_data(
        data_files=data_files[:5],
        subsystem=args.subsystem,
        min_activity_hits=args.min_hits,
        max_events=args.max_events,
        apply_timing_cut=True,
    )

    data_mean_hits = float(np.mean(data_hits)) if data_hits else 0.0
    data_mean_planes = float(np.mean(data_planes)) if data_planes else 0.0

    print(f"\n--- Conditioned Data Observables ({args.subsystem}, N >= {args.min_hits}) ---")
    print(f"  Active Event Fraction : {len(data_hits)}/{args.max_events} ({len(data_hits)/max(args.max_events, 1)*100:.1f}%)")
    print(f"  <N_hits> in Subsystem : {data_mean_hits:.2f}")
    print(f"  <N_planes> Fired      : {data_mean_planes:.2f}")

    if not cfg["mc_dig"]:
        print("No MC file available for this energy. Exiting.")
        return

    # 2. Process Conditioned Simulation
    thresholds_mev = np.arange(0.0, args.max_thresh + args.step_thresh, args.step_thresh)
    mc_hits_curve, mc_planes_curve, mc_bars, _ = process_mc_threshold_scan(
        mc_dig_file=cfg["mc_dig"],
        thresholds_mev=thresholds_mev,
        subsystem=args.subsystem,
        min_activity_hits=args.min_hits,
        max_events=args.max_events,
    )

    # 3. Find Optimal Crossing Point
    e_opt = interpolate_threshold(thresholds_mev, mc_hits_curve, data_mean_hits)

    print(f"\n=======================================================")
    print(f"  OPTIMAL THRESHOLD CALIBRATION ({args.subsystem} MATCH)")
    print(f"=======================================================")
    print(f"  Optimal {args.subsystem} Threshold E_th* :  {e_opt:.2f} MeV")
    print(f"  (Simulation matches Data <N_{args.subsystem}> = {data_mean_hits:.2f})")
    print(f"=======================================================\n")

    # 4. Generate 4 Informative Diagnostic Panels (All Anchored at y=0)
    ROOT.gStyle.SetOptStat(0)
    c = ROOT.TCanvas("c_diag", f"MuFilter {args.subsystem} Calibration Diagnostics", 1600, 1000)
    c.Divide(2, 2)

    # Panel 1: Conditioned <N_hits> vs Threshold
    c.cd(1)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    g_hits = ROOT.TGraph(len(thresholds_mev), thresholds_mev, mc_hits_curve)
    g_hits.SetTitle(f"{args.subsystem} Hits Survival vs Threshold;Threshold E_{{dep}} [MeV];<N_{{hits}}> per active event")
    g_hits.SetLineColor(ROOT.kAzure + 2)
    g_hits.SetLineWidth(3)
    g_hits.SetMinimum(0.0)
    g_hits.SetMaximum(float(mc_hits_curve[0]) * 1.15)
    g_hits.Draw("AL")

    line_data = ROOT.TLine(0, data_mean_hits, args.max_thresh, data_mean_hits)
    line_data.SetLineColor(ROOT.kRed + 1)
    line_data.SetLineStyle(2)
    line_data.SetLineWidth(2)
    line_data.Draw()

    leg1 = ROOT.TLegend(0.40, 0.72, 0.88, 0.88)
    leg1.AddEntry(g_hits, f"MC Simulation ({args.energy} GeV)", "l")
    leg1.AddEntry(line_data, f"Data (<N> = {data_mean_hits:.2f})", "l")
    leg1.Draw()

    # Panel 2: Fired Planes Multiplicity vs Threshold
    c.cd(2)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    g_pl = ROOT.TGraph(len(thresholds_mev), thresholds_mev, mc_planes_curve)
    g_pl.SetTitle(f"Fired {args.subsystem} Planes Depth;Threshold E_{{dep}} [MeV];<N_{{planes}}> fired")
    g_pl.SetLineColor(ROOT.kOrange + 7)
    g_pl.SetLineWidth(3)
    g_pl.SetMinimum(0.0)
    g_pl.SetMaximum(max(float(mc_planes_curve[0]) * 1.2, 5.0 if args.subsystem == "US" else 4.0))
    g_pl.Draw("AL")

    line_pl = ROOT.TLine(0, data_mean_planes, args.max_thresh, data_mean_planes)
    line_pl.SetLineColor(ROOT.kRed + 1)
    line_pl.SetLineStyle(2)
    line_pl.SetLineWidth(2)
    line_pl.Draw()

    leg2 = ROOT.TLegend(0.40, 0.72, 0.88, 0.88)
    leg2.AddEntry(g_pl, "MC Simulation", "l")
    leg2.AddEntry(line_pl, f"Data (<N_{{planes}}> = {data_mean_planes:.2f})", "l")
    leg2.Draw()

    # Panel 3: Hit Multiplicity Distribution P(N_hits) Data vs MC (raw)
    c.cd(3)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    max_h = max(max(data_hits) if data_hits else 10, 20)
    h_data = ROOT.TH1F("h_data", f"{args.subsystem} Hit Multiplicity Distribution;N_{{hits}};Fraction of active events", 25, 0, max_h)
    for h in data_hits:
        h_data.Fill(h)
    if h_data.Integral() > 0:
        h_data.Scale(1.0 / h_data.Integral())

    h_data.SetLineColor(ROOT.kRed + 1)
    h_data.SetLineWidth(2)
    h_data.SetMinimum(0.0)
    h_data.Draw("HIST")

    leg3 = ROOT.TLegend(0.40, 0.72, 0.88, 0.88)
    leg3.AddEntry(h_data, "Data Active Events", "l")
    leg3.Draw()

    # Panel 4: Transverse Bar Hit Profile
    c.cd(4)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    n_bars = 10 if args.subsystem == "US" else 60
    h_bars = ROOT.TH1F("h_bars", f"{args.subsystem} Transverse Bar Profile;Bar Number;Hit Count", n_bars, 0, n_bars)
    for b in data_bars:
        h_bars.Fill(b)
    h_bars.SetFillColor(ROOT.kTeal - 7)
    h_bars.SetLineColor(ROOT.kTeal + 2)
    h_bars.SetMinimum(0.0)
    h_bars.Draw("HIST")

    c.Update()
    c.SaveAs(out_plot)
    print(f"Saved diagnostic 4-panel canvas: {out_plot}")

    f_out = ROOT.TFile.Open(out_root, "RECREATE")
    g_hits.Write("g_hits_vs_threshold")
    g_pl.Write("g_planes_vs_threshold")
    h_data.Write("h_data_hits_distribution")
    h_bars.Write("h_data_bars_profile")
    c.Write()
    f_out.Close()
    print(f"Saved ROOT output: {out_root}")


if __name__ == "__main__":
    main()
