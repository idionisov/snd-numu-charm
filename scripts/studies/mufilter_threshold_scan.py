#!/usr/bin/env python3
"""
MuFilter Hit Energy Deposit Threshold Scan & Distribution Analysis

Studies the impact of applying a lower-end deposited energy threshold to
digitized MuFilterHits using Monte Carlo simulation truth (MuFilterPoint).

Features:
  - Associates MuFilterPoint deposited energy (in MeV) with each Digi_MuFilterHit.
  - Extracts energy deposit distributions separately for Veto, US, and DS.
  - Generates monotonically decreasing curves of <N_hits> vs E_dep_threshold
    and survival fraction curves S(E_th) = <N_hits(E_th)> / <N_hits(0)>.
  - Stores all histograms, TGraphs, and canvases in an output ROOT file and PNG.

Usage examples:
  python3 scripts/studies/mufilter_threshold_scan.py --preset genie -p "1-20" -o /eos/user/i/idioniso/snd-numu-charm/plots/mufilter_study/genie_thresh_scan.root
  python3 scripts/studies/mufilter_threshold_scan.py --preset direct -p "1-50" -o /eos/user/i/idioniso/snd-numu-charm/plots/mufilter_study/direct_thresh_scan.root
  python3 scripts/studies/mufilter_threshold_scan.py --preset pmu --max-events 5000 -o /eos/user/i/idioniso/snd-numu-charm/plots/mufilter_study/pmu_thresh_scan.root
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Ensure ROOT operates in batch mode
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

PRESETS = {
    "genie": "/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root",
    "direct": "/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonDirect.root",
    "indirect": "/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonIndirect.root",
    "pmu": "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP_2MuTrks_truth.root",
}

SUBSYSTEMS = {
    "Veto": 1,
    "US": 2,
    "DS": 3,
}


def parse_partition_spec(spec: str) -> List[str]:
    """Parse partition specifications like '1-10', '1,2,5', or '1'."""
    partitions = []
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start, end = int(start_str), int(end_str)
            partitions.extend([str(i) for i in range(start, end + 1)])
        elif token:
            partitions.append(token)
    return partitions


def resolve_input_files(pattern_or_preset: str, partitions: List[str]) -> List[str]:
    """Resolve file paths from preset / pattern and partitions."""
    matched_files = []
    if "%s" in pattern_or_preset:
        for p in partitions:
            fpath = pattern_or_preset % p
            if os.path.exists(fpath):
                matched_files.append(fpath)
    elif any(c in pattern_or_preset for c in ["*", "?", "["]):
        matched_files = sorted(glob.glob(pattern_or_preset))
    else:
        if os.path.exists(pattern_or_preset):
            matched_files = [pattern_or_preset]

    return matched_files


def setup_branch_filtering(tree: ROOT.TTree) -> None:
    """Optimize reading speed by disabling unneeded branches."""
    tree.SetBranchStatus("*", 0)
    tree.SetBranchStatus("MuFilterPoint*", 1)
    tree.SetBranchStatus("Digi_MuFilterHits*", 1)
    if tree.GetBranch("mc_weight"):
        tree.SetBranchStatus("mc_weight", 1)
    if tree.GetBranch("EventHeader."):
        tree.SetBranchStatus("EventHeader.*", 1)
    elif tree.GetBranch("EventHeader"):
        tree.SetBranchStatus("EventHeader*", 1)


def run_threshold_study(
    files: List[str],
    thresholds_mev: np.ndarray,
    max_events: int = -1,
) -> Tuple[Dict[str, List[float]], Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    """
    Process input simulation files and extract:
      1. Energy deposit values for each hit in Veto, US, DS, Total.
      2. Event hit counts vs threshold array for Veto, US, DS, Total.
    """
    # Raw hit energy lists per subsystem
    edep_values: Dict[str, List[float]] = {
        "Veto": [],
        "US": [],
        "DS": [],
        "Total": [],
    }

    # Hit counts per threshold: list of 1D arrays of length len(thresholds_mev)
    event_counts_unweighted: Dict[str, List[np.ndarray]] = {
        "Veto": [],
        "US": [],
        "DS": [],
        "Total": [],
    }
    event_counts_weighted: Dict[str, List[np.ndarray]] = {
        "Veto": [],
        "US": [],
        "DS": [],
        "Total": [],
    }
    event_weights: List[float] = []

    total_events_processed = 0
    t_start = time.time()

    print(f"Starting analysis over {len(files)} file(s)...")

    for f_idx, fpath in enumerate(files):
        if 0 < max_events <= total_events_processed:
            break

        f_in = ROOT.TFile.Open(fpath, "READ")
        if not f_in or f_in.IsZombie():
            print(f"Warning: Failed to open {fpath}")
            continue

        tree = f_in.Get("cbmsim")
        if not tree:
            f_in.Close()
            continue

        setup_branch_filtering(tree)
        n_entries = tree.GetEntries()

        for iev in range(n_entries):
            if 0 < max_events <= total_events_processed:
                break

            tree.GetEntry(iev)
            total_events_processed += 1

            # Event weight
            weight = 1.0
            if hasattr(tree, "mc_weight"):
                weight = float(tree.mc_weight)

            # 1. Sum true deposited energy per detector element bar (in MeV)
            # MuFilterPoint.GetEnergyLoss() is in GeV, convert to MeV (* 1000.0)
            bar_edep: Dict[int, float] = {}
            if hasattr(tree, "MuFilterPoint") and tree.MuFilterPoint:
                for pt in tree.MuFilterPoint:
                    det_id = pt.GetDetectorID()
                    bar_edep[det_id] = bar_edep.get(det_id, 0.0) + pt.GetEnergyLoss() * 1000.0

            # 2. Assign deposited energy to each digitized hit
            hit_energies: Dict[str, List[float]] = {
                "Veto": [],
                "US": [],
                "DS": [],
                "Total": [],
            }

            if hasattr(tree, "Digi_MuFilterHits") and tree.Digi_MuFilterHits:
                for hit in tree.Digi_MuFilterHits:
                    if not hit.isValid():
                        continue
                    det_id = hit.GetDetectorID()
                    subsys = det_id // 10000

                    edep = bar_edep.get(det_id, 0.0)
                    hit_energies["Total"].append(edep)
                    edep_values["Total"].append(edep)

                    if subsys == 1:
                        hit_energies["Veto"].append(edep)
                        edep_values["Veto"].append(edep)
                    elif subsys == 2:
                        hit_energies["US"].append(edep)
                        edep_values["US"].append(edep)
                    elif subsys == 3:
                        hit_energies["DS"].append(edep)
                        edep_values["DS"].append(edep)

            # 3. Vectorized threshold evaluation for this event
            event_weights.append(weight)
            for sub_key in ["Veto", "US", "DS", "Total"]:
                sub_e = np.array(hit_energies[sub_key])
                if len(sub_e) > 0:
                    # Matrix comparison: (n_hits, 1) >= (1, n_thresholds) -> sum over hits
                    surviving = (sub_e[:, None] >= thresholds_mev[None, :]).sum(axis=0)
                else:
                    surviving = np.zeros_like(thresholds_mev, dtype=float)

                event_counts_unweighted[sub_key].append(surviving)
                event_counts_weighted[sub_key].append(surviving * weight)

        f_in.Close()

        if (f_idx + 1) % 5 == 0 or (f_idx + 1) == len(files):
            elapsed = time.time() - t_start
            print(f"  Processed {f_idx + 1}/{len(files)} files ({total_events_processed} events) in {elapsed:.1f} s")

    # Aggregate mean curves and standard errors
    mean_unweighted: Dict[str, np.ndarray] = {}
    mean_weighted: Dict[str, np.ndarray] = {}

    w_arr = np.array(event_weights)
    sum_w = np.sum(w_arr) if len(w_arr) > 0 else 1.0

    for sub_key in ["Veto", "US", "DS", "Total"]:
        unw_mat = np.array(event_counts_unweighted[sub_key])  # shape: (N_events, N_thresh)
        wt_mat = np.array(event_counts_weighted[sub_key])    # shape: (N_events, N_thresh)

        if len(unw_mat) > 0:
            mean_unweighted[sub_key] = np.mean(unw_mat, axis=0)
            mean_weighted[sub_key] = np.sum(wt_mat, axis=0) / sum_w
        else:
            mean_unweighted[sub_key] = np.zeros_like(thresholds_mev)
            mean_weighted[sub_key] = np.zeros_like(thresholds_mev)

    return edep_values, mean_unweighted, mean_weighted, total_events_processed


def create_and_save_root_output(
    out_root_path: str,
    thresholds_mev: np.ndarray,
    edep_values: Dict[str, List[float]],
    mean_unw: Dict[str, np.ndarray],
    mean_wt: Dict[str, np.ndarray],
    total_events: int,
    plot_path: str = "",
) -> None:
    """Create ROOT histograms, TGraphs, and summary canvas, saving to out_root_path."""
    os.makedirs(os.path.dirname(os.path.abspath(out_root_path)), exist_ok=True)
    f_out = ROOT.TFile.Open(out_root_path, "RECREATE")

    # 1. Energy Deposit Distributions (Linear & Log-binned)
    d_edep = f_out.mkdir("energy_distributions")
    d_edep.cd()

    hist_cfgs = [
        ("Veto", "h_edep_veto", "Veto Bar Deposited Energy;Deposited Energy E_{dep} [MeV];Hits / 0.1 MeV", 200, 0.0, 20.0),
        ("US", "h_edep_us", "Upstream (US) Bar Deposited Energy;Deposited Energy E_{dep} [MeV];Hits / 0.25 MeV", 200, 0.0, 50.0),
        ("DS", "h_edep_ds", "Downstream (DS) Bar Deposited Energy;Deposited Energy E_{dep} [MeV];Hits / 0.1 MeV", 200, 0.0, 20.0),
        ("Total", "h_edep_all", "All MuFilter Bar Deposited Energy;Deposited Energy E_{dep} [MeV];Hits / 0.25 MeV", 200, 0.0, 50.0),
    ]

    hists_edep = {}
    for sub, h_name, h_title, n_bins, x_min, x_max in hist_cfgs:
        h = ROOT.TH1D(h_name, h_title, n_bins, x_min, x_max)
        vals = edep_values.get(sub, [])
        for v in vals:
            h.Fill(v)
        h.Write()
        hists_edep[sub] = h

    # 2. Monotonically Decreasing Graphs: <N_hits> vs Threshold
    d_curves = f_out.mkdir("threshold_scan")
    d_curves.cd()

    n_points = len(thresholds_mev)
    graphs_unw = {}
    graphs_wt = {}
    graphs_frac = {}

    colors = {
        "Veto": ROOT.kGreen + 2,
        "US": ROOT.kRed + 1,
        "DS": ROOT.kAzure + 2,
        "Total": ROOT.kBlack,
    }

    for sub in ["Veto", "US", "DS", "Total"]:
        unw = mean_unw[sub]
        wt = mean_wt[sub]

        g_u = ROOT.TGraph(n_points, thresholds_mev.astype(float), unw.astype(float))
        g_u.SetName(f"g_hits_vs_ethresh_{sub.lower()}")
        g_u.SetTitle(f"Average {sub} Hits vs Energy Threshold;Lower Energy Threshold E_{{th}} [MeV];#LT N_{{hits}} #GT")
        g_u.SetLineColor(colors[sub])
        g_u.SetLineWidth(2)
        g_u.Write()
        graphs_unw[sub] = g_u

        g_w = ROOT.TGraph(n_points, thresholds_mev.astype(float), wt.astype(float))
        g_w.SetName(f"g_hits_vs_ethresh_{sub.lower()}_weighted")
        g_w.SetTitle(f"MC-Weighted Average {sub} Hits vs Energy Threshold;Lower Energy Threshold E_{{th}} [MeV];#LT N_{{hits}} #GT (weighted)")
        g_w.SetLineColor(colors[sub])
        g_w.SetLineWidth(2)
        g_w.SetLineStyle(2)
        g_w.Write()
        graphs_wt[sub] = g_w

        # Normalized survival fraction S(E_th) = <N_hits(E_th)> / <N_hits(0)>
        base_val = unw[0] if unw[0] > 0 else 1.0
        frac = unw / base_val
        g_f = ROOT.TGraph(n_points, thresholds_mev.astype(float), frac.astype(float))
        g_f.SetName(f"g_frac_vs_ethresh_{sub.lower()}")
        g_f.SetTitle(f"Fraction of Surviving {sub} Hits;Lower Energy Threshold E_{{th}} [MeV];Survival Fraction S(E_{{th}})")
        g_f.SetLineColor(colors[sub])
        g_f.SetLineWidth(2)
        g_f.Write()
        graphs_frac[sub] = g_f

    # 3. Create Multi-Panel TCanvas Comparison
    f_out.cd()
    c_summary = ROOT.TCanvas("c_mufilter_threshold_scan", "MuFilter Threshold Scan & Energy Distributions", 1400, 900)
    c_summary.Divide(2, 2)

    # Pad 1: Energy Distributions
    c_summary.cd(1)
    ROOT.gPad.SetLogy(1)
    ROOT.gPad.SetGrid()
    leg1 = ROOT.TLegend(0.60, 0.70, 0.88, 0.88)
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)

    for idx, sub in enumerate(["US", "DS", "Veto"]):
        h = hists_edep[sub]
        h.SetLineColor(colors[sub])
        h.SetLineWidth(2)
        draw_opt = "HIST" if idx == 0 else "HIST SAME"
        h.Draw(draw_opt)
        leg1.AddEntry(h, sub, "l")
    leg1.Draw()

    # Pad 2: Average Hits vs Threshold (Unweighted)
    c_summary.cd(2)
    ROOT.gPad.SetGrid()
    leg2 = ROOT.TLegend(0.60, 0.68, 0.88, 0.88)
    leg2.SetBorderSize(0)
    leg2.SetFillStyle(0)

    mg2 = ROOT.TMultiGraph("mg_nhits", "Average Hits vs Threshold;E_{th} [MeV];#LT N_{hits} #GT")
    for sub in ["Total", "US", "DS", "Veto"]:
        mg2.Add(graphs_unw[sub])
        leg2.AddEntry(graphs_unw[sub], sub, "l")
    mg2.Draw("AL")
    leg2.Draw()

    # Pad 3: MC-Weighted Average Hits vs Threshold
    c_summary.cd(3)
    ROOT.gPad.SetGrid()
    leg3 = ROOT.TLegend(0.60, 0.68, 0.88, 0.88)
    leg3.SetBorderSize(0)
    leg3.SetFillStyle(0)

    mg3 = ROOT.TMultiGraph("mg_nhits_weighted", "MC-Weighted Average Hits vs Threshold;E_{th} [MeV];#LT N_{hits} #GT (weighted)")
    for sub in ["Total", "US", "DS", "Veto"]:
        mg3.Add(graphs_wt[sub])
        leg3.AddEntry(graphs_wt[sub], sub, "l")
    mg3.Draw("AL")
    leg3.Draw()

    # Pad 4: Surviving Hit Fraction (Normalized to E_th = 0)
    c_summary.cd(4)
    ROOT.gPad.SetGrid()
    leg4 = ROOT.TLegend(0.60, 0.68, 0.88, 0.88)
    leg4.SetBorderSize(0)
    leg4.SetFillStyle(0)

    mg4 = ROOT.TMultiGraph("mg_frac", "Surviving Hit Fraction vs Threshold;E_{th} [MeV];Survival Fraction S(E_{th})")
    for sub in ["Total", "US", "DS", "Veto"]:
        mg4.Add(graphs_frac[sub])
        leg4.AddEntry(graphs_frac[sub], sub, "l")
    mg4.Draw("AL")
    leg4.Draw()

    c_summary.Update()
    c_summary.Write()

    if plot_path:
        os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
        c_summary.SaveAs(plot_path)
        print(f"Saved summary canvas to: {plot_path}")

    f_out.Close()
    print(f"Saved ROOT output with histograms and graphs to: {out_root_path}")


def main():
    parser = argparse.ArgumentParser(description="Scan MuFilter Hit Energy Deposit Lower Threshold on Simulations.")
    parser.add_argument("-i", "--input", help="Direct input ROOT file or pattern with '%s' or glob")
    parser.add_argument("--preset", choices=["genie", "direct", "indirect", "pmu"], default="genie",
                        help="Simulation sample preset (default: genie)")
    parser.add_argument("-p", "--partitions", default="1-20",
                        help="Partition range for '%s' pattern (default: 1-20)")
    parser.add_argument("-o", "--output", default="mufilter_threshold_scan.root",
                        help="Output ROOT file path (default: mufilter_threshold_scan.root)")
    parser.add_argument("--plot", default="",
                        help="Output PNG/PDF plot path (default: same stem as -o with .png)")
    parser.add_argument("--max-events", type=int, default=-1,
                        help="Maximum events to analyze across files (default: -1 for all)")
    parser.add_argument("--max-thresh", type=float, default=15.0,
                        help="Maximum energy deposit threshold to scan in MeV (default: 15.0)")
    parser.add_argument("--step-thresh", type=float, default=0.2,
                        help="Energy deposit threshold step in MeV (default: 0.2)")

    args = parser.parse_args()

    # Determine input pattern
    input_pattern = args.input if args.input else PRESETS[args.preset]
    partitions = parse_partition_spec(args.partitions)
    files = resolve_input_files(input_pattern, partitions)

    if not files:
        print(f"Error: No matching ROOT files found for pattern: {input_pattern}")
        sys.exit(1)

    print(f"Found {len(files)} matching file(s) for study.")

    # Determine plot path if not explicitly provided
    plot_path = args.plot
    if not plot_path:
        base, _ = os.path.splitext(args.output)
        plot_path = f"{base}.png"

    thresholds_mev = np.arange(0.0, args.max_thresh + args.step_thresh, args.step_thresh)

    edep_vals, mean_unw, mean_wt, total_ev = run_threshold_study(
        files=files,
        thresholds_mev=thresholds_mev,
        max_events=args.max_events,
    )

    print("\n--- Summary at Threshold Extremes ---")
    for sub in ["Veto", "US", "DS", "Total"]:
        v0 = mean_unw[sub][0]
        v1 = mean_unw[sub][int(1.0 / args.step_thresh)] if 1.0 <= args.max_thresh else 0.0
        v2 = mean_unw[sub][int(2.0 / args.step_thresh)] if 2.0 <= args.max_thresh else 0.0
        print(f"  {sub:6s} | E_th = 0.0 MeV: {v0:.2f} hits | E_th = 1.0 MeV: {v1:.2f} hits (-{(1-v1/v0)*100:.1f}%) | E_th = 2.0 MeV: {v2:.2f} hits (-{(1-v2/v0)*100:.1f}%)")

    create_and_save_root_output(
        out_root_path=args.output,
        thresholds_mev=thresholds_mev,
        edep_values=edep_vals,
        mean_unw=mean_unw,
        mean_wt=mean_wt,
        total_events=total_ev,
        plot_path=plot_path,
    )


if __name__ == "__main__":
    main()
