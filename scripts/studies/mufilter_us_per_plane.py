#!/usr/bin/env python3
"""
MuFilter Testbeam Upstream (US) Plane-by-Plane Calibration & Detailed Comparison
Author: Antigravity Assistant

Performs individual plane-by-plane analysis for US Planes 0, 1, 2, 3, 4:
1. Dynamic threshold scan conditioned on active hits in that specific plane (N_plane >= 1).
2. Dynamic crossing threshold finding for each plane.
3. Overlapping hit multiplicity distributions comparing:
     - MC Raw (E_th = 0) in grey
     - MC Calibrated (E_th = E_th*) in solid blue
     - Real Data in red dashed line
4. Transverse bar profile (Bars 0-9) per plane.
5. Saves all figures and interactive ROOT TCanvases into calib_mufilter_tb_detailed.root.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

try:
    import ddfUtils.mpl_styling as dms
    dms.apply_style()
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def get_plane_data(
    data_files: List[str],
    max_events: int = 4000,
    max_delta_t_ns: float = 10.0,
) -> Dict[int, Dict[str, any]]:
    """Extract per-plane hit multiplicities and bar profiles from testbeam data."""
    print(f"Reading Data for US planes from {len(data_files)} files (max {max_events} events)...")
    t0 = time.time()

    plane_hits = {p: [] for p in range(5)}
    plane_bars = {p: [] for p in range(5)}
    total_events = 0

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

        tree.SetBranchStatus("*", 0)
        tree.SetBranchStatus("Digi_MuFilterHits*", 1)

        n_entries = tree.GetEntries()
        for iev in range(n_entries):
            if 0 < max_events <= total_events:
                break
            tree.GetEntry(iev)
            total_events += 1

            counts = {p: 0 for p in range(5)}
            bars_ev = {p: [] for p in range(5)}

            for hit in tree.Digi_MuFilterHits:
                if not hit.isValid():
                    continue
                t = hit.GetTime()
                if t != -999.0 and abs(t) > max_delta_t_ns:
                    continue

                det_id = hit.GetDetectorID()
                if det_id // 10000 == 2:  # US
                    pl = (det_id // 1000) % 10
                    bar = det_id % 1000
                    if pl in counts:
                        counts[pl] += 1
                        bars_ev[pl].append(bar)

            for p in range(5):
                plane_hits[p].append(counts[p])
                if counts[p] >= 1:
                    plane_bars[p].extend(bars_ev[p])

        f_in.Close()

    print(f"Data read: {total_events} events in {time.time()-t0:.1f} s.")

    res = {}
    for p in range(5):
        arr = np.array(plane_hits[p])
        active = arr[arr >= 1]
        res[p] = {
            "all_hits": arr,
            "active_hits": active,
            "mean_active": float(np.mean(active)) if len(active) > 0 else 0.0,
            "active_frac": len(active) / max(total_events, 1),
            "bars": plane_bars[p],
        }
    return res


def get_plane_mc(
    mc_file: str,
    max_events: int = 4000,
    scan_thresholds: np.ndarray = np.linspace(0.0, 50.0, 51),
) -> Dict[int, Dict[str, any]]:
    """Extract per-plane energy deposits from simulation and compute plane threshold scan curves."""
    print(f"Reading MC for US planes from {mc_file} (max {max_events} events)...")
    t0 = time.time()

    f_in = ROOT.TFile.Open(mc_file, "READ")
    if not f_in or f_in.IsZombie():
        raise RuntimeError(f"Cannot open MC file: {mc_file}")

    tree = f_in.Get("cbmsim;236") or f_in.Get("cbmsim")
    tree.SetBranchStatus("*", 0)
    tree.SetBranchStatus("MuFilterPoint*", 1)
    tree.SetBranchStatus("Digi_MuFilterHits*", 1)

    n_entries = tree.GetEntries()
    total_events = 0

    # Store per-event, per-plane hits as list of (bar, edep)
    events_plane_data = {p: [] for p in range(5)}

    for iev in range(n_entries):
        if 0 < max_events <= total_events:
            break
        tree.GetEntry(iev)
        total_events += 1

        bar_edep: Dict[int, float] = {}
        for pt in tree.MuFilterPoint:
            did = pt.GetDetectorID()
            bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss() * 1000.0

        ev_hits = {p: [] for p in range(5)}
        for hit in tree.Digi_MuFilterHits:
            if not hit.isValid():
                continue
            did = hit.GetDetectorID()
            if did // 10000 == 2:  # US
                pl = (did // 1000) % 10
                bar = did % 1000
                edep = bar_edep.get(did, 0.0)
                if pl in ev_hits:
                    ev_hits[pl].append((bar, edep))

        for p in range(5):
            events_plane_data[p].append(ev_hits[p])

    f_in.Close()
    print(f"MC read: {total_events} events in {time.time()-t0:.1f} s.")

    res = {}
    for p in range(5):
        # Scan curve conditioned on active events in plane p: N(th) >= 1
        curve = []
        for th in scan_thresholds:
            counts = [(np.array([e for _, e in ev]) >= th).sum() for ev in events_plane_data[p]]
            act = [c for c in counts if c >= 1]
            curve.append(float(np.mean(act)) if act else 0.0)

        res[p] = {
            "events": events_plane_data[p],
            "scan_thresholds": scan_thresholds,
            "scan_curve": np.array(curve),
        }
    return res


def interpolate_crossing(xs: np.ndarray, ys: np.ndarray, target: float) -> float:
    """Interpolate crossing point."""
    if target >= ys[0]:
        return float(xs[0])
    if target <= ys[-1]:
        return float(xs[-1])
    idx = np.where(ys <= target)[0][0]
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    if y1 == y0:
        return float(x0)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def main():
    parser = argparse.ArgumentParser(description="MuFilter US Plane-by-Plane Calibration")
    parser.add_argument("--max-events", type=int, default=4000, help="Max events to process per sample")
    parser.add_argument("-o", "--output-root", type=str, default="/eos/user/i/idioniso/snd-numu-charm/plots/mufilter_study/calib_mufilter_tb_detailed.root")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output_root))
    os.makedirs(out_dir, exist_ok=True)

    data_files = [
        "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100639/sndsw_raw-0000.root",
        "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100639/sndsw_raw-0001.root",
        "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100639/sndsw_raw-0002.root",
    ]
    mc_file = "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/300GeV_-211/sndLHC.PG_-211-TGeant4_digCPP.root"

    data_planes = get_plane_data(data_files, max_events=args.max_events)
    mc_planes = get_plane_mc(mc_file, max_events=args.max_events)

    optimal_thresholds = {}
    print("\n=======================================================")
    print("  UPSTREAM (US) PLANE-BY-PLANE CALIBRATION SUMMARY")
    print("=======================================================")
    for p in range(5):
        target = data_planes[p]["mean_active"]
        th_opt = interpolate_crossing(mc_planes[p]["scan_thresholds"], mc_planes[p]["scan_curve"], target)
        optimal_thresholds[p] = th_opt
        print(f"  Plane {p}: Data Active <N> = {target:.2f} hits -> Optimal E_th* = {th_opt:.2f} MeV")
    print("=======================================================\n")

    # Evaluate MC per plane at Raw (E_th = 0) and Calibrated (E_th = E_th*)
    mc_plane_calib = {}
    for p in range(5):
        th = optimal_thresholds[p]
        raw_counts = np.array([len(ev) for ev in mc_planes[p]["events"]])
        cal_counts = np.array([sum(1 for _, e in ev if e >= th) for ev in mc_planes[p]["events"]])

        raw_active = raw_counts[raw_counts >= 1]
        cal_active = cal_counts[cal_counts >= 1]

        # Bars at calibrated threshold
        cal_bars = []
        for ev in mc_planes[p]["events"]:
            for b, e in ev:
                if e >= th:
                    cal_bars.append(b)

        mc_plane_calib[p] = {
            "raw_active": raw_active,
            "cal_active": cal_active,
            "mean_cal": float(np.mean(cal_active)) if len(cal_active) > 0 else 0.0,
            "cal_bars": cal_bars,
        }

    # =========================================================================
    # PLOTTING: 2-Row Grid for US Planes (Row 1: Scans & Crossings, Row 2: Overlapping Multiplicities)
    # =========================================================================
    fig, axes = plt.subplots(2, 5, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.28, hspace=0.32)

    for p in range(5):
        # Top Row: Threshold Scan & Exact Crossing
        ax_top = axes[0, p]
        xs = mc_planes[p]["scan_thresholds"]
        ys = mc_planes[p]["scan_curve"]
        target = data_planes[p]["mean_active"]
        th_opt = optimal_thresholds[p]

        ax_top.plot(xs, ys, color="#1f77b4", lw=2.5, label="Simulation (MC)")
        ax_top.axhline(target, color="#d62728", ls="--", lw=2.0, label=f"Data Mean: {target:.2f}")
        ax_top.axvline(th_opt, color="#2ca02c", ls=":", lw=1.8, label=f"$E_{{th}}^* = {th_opt:.1f}$ MeV")
        ax_top.plot([th_opt], [target], marker="o", markersize=8, color="#d62728", markeredgecolor="black", zorder=5)

        ax_top.set_title(f"US Plane {p} Threshold Scan", fontsize=12, fontweight="bold")
        ax_top.set_xlabel(r"Threshold $E_{\rm dep}$ [MeV]", fontsize=10)
        ax_top.set_ylabel(r"$\langle N_{\rm hits}\rangle$ (active events)", fontsize=10)
        ax_top.set_ylim(bottom=0.0, top=max(float(ys[0]) * 1.15, target * 1.5))
        ax_top.set_xlim(left=0.0, right=xs[-1])
        ax_top.legend(loc="upper right", frameon=True, fontsize=8.5)
        ax_top.grid(True, alpha=0.3)

        # Bottom Row: Overlapping Hit Multiplicities
        ax_bot = axes[1, p]
        bins = np.arange(0.5, 11.5, 1.0)
        data_act = data_planes[p]["active_hits"]
        mc_raw_act = mc_plane_calib[p]["raw_active"]
        mc_cal_act = mc_plane_calib[p]["cal_active"]

        ax_bot.hist(
            mc_raw_act,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.30,
            color="#7f7f7f",
            edgecolor="#555555",
            lw=1.4,
            label=f"MC Raw (mean={np.mean(mc_raw_act):.2f})",
        )
        ax_bot.hist(
            mc_cal_act,
            bins=bins,
            density=True,
            histtype="step",
            color="#1f77b4",
            lw=2.4,
            label=f"MC Calib (mean={np.mean(mc_cal_act):.2f})",
        )
        ax_bot.hist(
            data_act,
            bins=bins,
            density=True,
            histtype="step",
            color="#d62728",
            lw=2.4,
            ls="--",
            label=f"Data Active (mean={target:.2f})",
        )

        ax_bot.set_title(f"US Plane {p} Hit Multiplicity", fontsize=12, fontweight="bold")
        ax_bot.set_xlabel(r"$N_{\rm hits}$ in Plane", fontsize=10)
        ax_bot.set_ylabel("Probability Density", fontsize=10)
        ax_bot.set_ylim(bottom=0.0)
        ax_bot.set_xlim(0.5, 10.5)
        ax_bot.legend(loc="upper right", frameon=True, fontsize=8.5)
        ax_bot.grid(True, alpha=0.3)

    out_png = os.path.join(out_dir, "calib_mufilter_us_per_plane.png")
    out_pdf = os.path.join(out_dir, "calib_mufilter_us_per_plane.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved US per-plane plots: {out_png}")

    # =========================================================================
    # APPEND TO ROOT FILE (calib_mufilter_tb_detailed.root)
    # =========================================================================
    f_root = ROOT.TFile.Open(args.output_root, "UPDATE")
    print(f"Updating ROOT file: {args.output_root} with US per-plane TCanvases and histograms...")

    c_us_planes = ROOT.TCanvas("c_us_planes_summary", "US Per-Plane Calibration Summary", 2000, 800)
    c_us_planes.Divide(5, 2)

    for p in range(5):
        # Top Pad: Scan
        c_us_planes.cd(p + 1)
        ROOT.gPad.SetLeftMargin(0.14)
        ROOT.gPad.SetBottomMargin(0.12)
        xs = mc_planes[p]["scan_thresholds"]
        ys = mc_planes[p]["scan_curve"]
        target = data_planes[p]["mean_active"]
        th_opt = optimal_thresholds[p]

        g_pl = ROOT.TGraph(len(xs), xs, ys)
        g_pl.SetName(f"g_us_scan_p{p}")
        g_pl.SetTitle(f"US Plane {p} Scan;E_{{dep}} [MeV];<N_{{hits}}>")
        g_pl.SetLineColor(ROOT.kAzure + 2)
        g_pl.SetLineWidth(3)
        g_pl.SetMinimum(0.0)
        g_pl.SetMaximum(float(ys[0]) * 1.15)
        g_pl.Draw("AL")

        l_data = ROOT.TLine(0, target, xs[-1], target)
        l_data.SetLineColor(ROOT.kRed + 1)
        l_data.SetLineStyle(2)
        l_data.SetLineWidth(2)
        l_data.Draw()

        l_th = ROOT.TLine(th_opt, 0, th_opt, target)
        l_th.SetLineColor(ROOT.kGreen + 2)
        l_th.SetLineStyle(3)
        l_th.SetLineWidth(2)
        l_th.Draw()

        leg_top = ROOT.TLegend(0.40, 0.70, 0.88, 0.88)
        leg_top.AddEntry(g_pl, "MC Scan", "l")
        leg_top.AddEntry(l_data, f"Data: {target:.2f}", "l")
        leg_top.AddEntry(l_th, f"E_{{th}}^*: {th_opt:.1f} MeV", "l")
        leg_top.Draw()

        # Bottom Pad: Multiplicity
        c_us_planes.cd(p + 6)
        ROOT.gPad.SetLeftMargin(0.14)
        ROOT.gPad.SetBottomMargin(0.12)

        h_raw = ROOT.TH1D(f"h_us_p{p}_mc_raw", f"US Plane {p} Raw;N_{{hits}};Normalized Events", 10, 0.5, 10.5)
        for x in mc_plane_calib[p]["raw_active"]: h_raw.Fill(x)
        if h_raw.Integral() > 0: h_raw.Scale(1.0 / h_raw.Integral())
        h_raw.SetFillColor(ROOT.kGray)
        h_raw.SetLineColor(ROOT.kGray + 2)
        h_raw.SetMinimum(0.0)
        h_raw.Draw("HIST")

        h_cal = ROOT.TH1D(f"h_us_p{p}_mc_calib", f"US Plane {p} Calib;N_{{hits}};Normalized Events", 10, 0.5, 10.5)
        for x in mc_plane_calib[p]["cal_active"]: h_cal.Fill(x)
        if h_cal.Integral() > 0: h_cal.Scale(1.0 / h_cal.Integral())
        h_cal.SetLineColor(ROOT.kAzure + 2)
        h_cal.SetLineWidth(3)
        h_cal.Draw("HIST SAME")

        h_data = ROOT.TH1D(f"h_us_p{p}_data", f"US Plane {p} Data;N_{{hits}};Normalized Events", 10, 0.5, 10.5)
        for x in data_planes[p]["active_hits"]: h_data.Fill(x)
        if h_data.Integral() > 0: h_data.Scale(1.0 / h_data.Integral())
        h_data.SetLineColor(ROOT.kRed + 1)
        h_data.SetLineStyle(2)
        h_data.SetLineWidth(3)
        h_data.Draw("HIST SAME")

        leg_bot = ROOT.TLegend(0.40, 0.70, 0.88, 0.88)
        leg_bot.AddEntry(h_raw, "MC Raw", "f")
        leg_bot.AddEntry(h_cal, f"MC Cal ({th_opt:.1f})", "l")
        leg_bot.AddEntry(h_data, "Data Active", "l")
        leg_bot.Draw()

        # Write individual objects
        g_pl.Write(f"g_us_scan_p{p}", ROOT.TObject.kOverwrite)
        h_raw.Write(f"h_us_p{p}_mc_raw", ROOT.TObject.kOverwrite)
        h_cal.Write(f"h_us_p{p}_mc_calib", ROOT.TObject.kOverwrite)
        h_data.Write(f"h_us_p{p}_data", ROOT.TObject.kOverwrite)

    c_us_planes.Update()
    c_us_planes.Write("c_us_planes_summary", ROOT.TObject.kOverwrite)
    f_root.Close()
    print(f"Successfully updated ROOT file with per-plane canvases: {args.output_root}")


if __name__ == "__main__":
    main()
