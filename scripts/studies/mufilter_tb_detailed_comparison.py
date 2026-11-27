#!/usr/bin/env python3
"""
MuFilter Testbeam Detailed Comparison: Data vs Simulation

Generates:
1. Dedicated Crossing Plot: MC <N_hits> vs Threshold E_dep crossing Data dotted line.
2. Overlapping Hit Multiplicity Distributions: Data vs MC for DS (at 1.29 MeV) and US.
3. Plane-by-plane Hit Multiplicities: Data vs MC across US planes (0 to 4).
4. DS Station 0 View Breakdown: Horizontal vs Vertical bar hit counts.

Author: Antigravity Assistant
"""

from __future__ import annotations

import argparse
import glob
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

# Custom utils styling
try:
    import ddfUtils.mpl_styling as dms
    dms.apply_style()
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def get_data_events(
    data_files: List[str],
    max_events: int = 1500,
    max_delta_t_ns: float = 10.0,
) -> Dict[str, any]:
    """Extract hit multiplicities per subsystem and per plane from testbeam data."""
    print(f"Reading Data from {len(data_files)} files (max {max_events} events)...")
    t0 = time.time()

    us_hits_all = []
    ds_hits_all = []
    us_plane_hits = {p: [] for p in range(5)}
    ds_horiz_hits = []
    ds_vert_hits = []

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

            n_us = 0
            n_ds = 0
            plane_counts = {p: 0 for p in range(5)}
            n_ds_h = 0
            n_ds_v = 0

            for hit in tree.Digi_MuFilterHits:
                if not hit.isValid():
                    continue
                t = hit.GetTime()
                if t != -999.0 and abs(t) > max_delta_t_ns:
                    continue

                det_id = hit.GetDetectorID()
                sub = det_id // 10000
                if sub == 2:  # US
                    n_us += 1
                    pl = (det_id // 1000) % 10
                    if pl in plane_counts:
                        plane_counts[pl] += 1
                elif sub == 3:  # DS
                    n_ds += 1
                    bar = det_id % 1000
                    if bar < 60:
                        n_ds_h += 1
                    else:
                        n_ds_v += 1

            us_hits_all.append(n_us)
            ds_hits_all.append(n_ds)
            for p in range(5):
                us_plane_hits[p].append(plane_counts[p])
            ds_horiz_hits.append(n_ds_h)
            ds_vert_hits.append(n_ds_v)

        f_in.Close()

    print(f"Data read: {total_events} events in {time.time()-t0:.1f} s.")
    return {
        "us_hits": np.array(us_hits_all),
        "ds_hits": np.array(ds_hits_all),
        "us_plane_hits": {p: np.array(us_plane_hits[p]) for p in range(5)},
        "ds_horiz_hits": np.array(ds_horiz_hits),
        "ds_vert_hits": np.array(ds_vert_hits),
    }


def get_mc_events(
    mc_file: str,
    max_events: int = 1500,
    ds_threshold_mev: float = 1.29,
    us_threshold_mev: float = 6.0,
    scan_thresholds: np.ndarray = np.linspace(0.0, 5.0, 51),
) -> Dict[str, any]:
    """Process simulation events: extract threshold scan curve and distributions at optimal thresholds."""
    print(f"Reading MC from {mc_file} (max {max_events} events)...")
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

    mc_ds_hits_calib = []
    mc_us_hits_calib = []
    mc_us_plane_calib = {p: [] for p in range(5)}
    mc_ds_horiz_calib = []
    mc_ds_vert_calib = []

    # Per-event DS hits vs threshold for the scan curve
    ds_hits_scan = []

    for iev in range(n_entries):
        if 0 < max_events <= total_events:
            break
        tree.GetEntry(iev)
        total_events += 1

        bar_edep: Dict[int, float] = {}
        for pt in tree.MuFilterPoint:
            did = pt.GetDetectorID()
            bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss() * 1000.0

        # Evaluate at calibrated threshold
        n_us = 0
        n_ds = 0
        plane_counts = {p: 0 for p in range(5)}
        n_ds_h = 0
        n_ds_v = 0
        ds_hits_edeps = []

        for hit in tree.Digi_MuFilterHits:
            if not hit.isValid():
                continue
            did = hit.GetDetectorID()
            sub = did // 10000
            edep = bar_edep.get(did, 0.0)

            if sub == 2:  # US
                if edep >= us_threshold_mev:
                    n_us += 1
                    pl = (did // 1000) % 10
                    if pl in plane_counts:
                        plane_counts[pl] += 1
            elif sub == 3:  # DS
                ds_hits_edeps.append(edep)
                if edep >= ds_threshold_mev:
                    n_ds += 1
                    bar = did % 1000
                    if bar < 60:
                        n_ds_h += 1
                    else:
                        n_ds_v += 1

        mc_us_hits_calib.append(n_us)
        mc_ds_hits_calib.append(n_ds)
        for p in range(5):
            mc_us_plane_calib[p].append(plane_counts[p])
        mc_ds_horiz_calib.append(n_ds_h)
        mc_ds_vert_calib.append(n_ds_v)

        # Threshold scan for DS (conditioned on raw hits >= 1)
        if len(ds_hits_edeps) >= 1:
            arr_e = np.array(ds_hits_edeps)
            surv = (arr_e[:, None] >= scan_thresholds[None, :]).sum(axis=0)
            ds_hits_scan.append(surv)

    f_in.Close()
    print(f"MC read: {total_events} events in {time.time()-t0:.1f} s.")

    scan_curve = np.mean(ds_hits_scan, axis=0) if ds_hits_scan else np.zeros_like(scan_thresholds)

    return {
        "scan_thresholds": scan_thresholds,
        "ds_scan_curve": scan_curve,
        "us_hits": np.array(mc_us_hits_calib),
        "ds_hits": np.array(mc_ds_hits_calib),
        "us_plane_hits": {p: np.array(mc_us_plane_calib[p]) for p in range(5)},
        "ds_horiz_hits": np.array(mc_ds_horiz_calib),
        "ds_vert_hits": np.array(mc_ds_vert_calib),
    }


def make_plots(
    data_res: Dict[str, any],
    mc_res: Dict[str, any],
    ds_th_opt: float = 1.29,
    us_th_opt: float = 6.0,
    out_prefix: str = "/eos/user/i/idioniso/snd-numu-charm/plots/mufilter_study/tb_detailed_comparison",
) -> None:
    """Generate high-quality multi-panel comparison figure."""
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)

    # Active DS events filter (N_DS >= 1)
    mask_ds_data = data_res["ds_hits"] >= 1
    data_ds_active = data_res["ds_hits"][mask_ds_data]
    data_ds_mean = float(np.mean(data_ds_active)) if len(data_ds_active) > 0 else 0.0

    mask_ds_mc = mc_res["ds_hits"] >= 1
    mc_ds_active = mc_res["ds_hits"][mask_ds_mc]
    mc_ds_mean = float(np.mean(mc_ds_active)) if len(mc_ds_active) > 0 else 0.0

    # Active US events filter (N_US >= 1)
    mask_us_data = data_res["us_hits"] >= 1
    data_us_active = data_res["us_hits"][mask_us_data]
    data_us_mean = float(np.mean(data_us_active)) if len(data_us_active) > 0 else 0.0

    mask_us_mc = mc_res["us_hits"] >= 1
    mc_us_active = mc_res["us_hits"][mask_us_mc]
    mc_us_mean = float(np.mean(mc_us_active)) if len(mc_us_active) > 0 else 0.0

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(wspace=0.25, hspace=0.30)

    # =========================================================================
    # Panel 1: DS Hit Survival Curve vs Threshold & Exact Crossing Point
    # =========================================================================
    ax1 = axes[0, 0]
    th_vals = mc_res["scan_thresholds"]
    curve = mc_res["ds_scan_curve"]

    ax1.plot(th_vals, curve, color="#1f77b4", lw=2.8, label=r"Simulation (MC $\langle N_{\rm DS}\rangle$)")
    ax1.axhline(data_ds_mean, color="#d62728", ls="--", lw=2.2, label=f"Data Mean: {data_ds_mean:.2f} hits")
    ax1.axvline(ds_th_opt, color="#2ca02c", ls=":", lw=2.0, label=f"Optimal $E_{{th}}^* = {ds_th_opt:.2f}$ MeV")
    ax1.plot([ds_th_opt], [data_ds_mean], marker="o", markersize=9, color="#d62728", markeredgecolor="black")

    ax1.set_title("Downstream (DS) Threshold Scan & Crossing", fontsize=14, fontweight="bold")
    ax1.set_xlabel(r"Energy Deposit Threshold $E_{\rm dep}$ [MeV]", fontsize=12)
    ax1.set_ylabel(r"$\langle N_{\rm hits}^{\rm DS}\rangle$ per active event", fontsize=12)
    ax1.set_ylim(bottom=0.0, top=max(float(curve[0]) * 1.15, data_ds_mean * 1.35))
    ax1.set_xlim(left=0.0, right=th_vals[-1])
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax1.annotate(
        f"Intersection:\n$E_{{th}}^* = {ds_th_opt:.2f}$ MeV\n$<N_{{hits}}> = {data_ds_mean:.2f}$",
        xy=(ds_th_opt, data_ds_mean),
        xytext=(ds_th_opt + 0.8, data_ds_mean + 1.8),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2ae", edgecolor="#b59410", alpha=0.9),
        fontsize=10,
        fontweight="bold",
    )

    # =========================================================================
    # Panel 2: DS Overlapping Hit Multiplicity Distribution (Data vs MC)
    # =========================================================================
    ax2 = axes[0, 1]
    bins_ds = np.arange(0.5, 35.5, 1.0)
    
    # Normalized histograms (density=True)
    ax2.hist(
        data_ds_active,
        bins=bins_ds,
        density=True,
        histtype="step",
        color="#d62728",
        lw=2.5,
        label=f"Data Active (mean={data_ds_mean:.2f})",
    )
    ax2.hist(
        mc_ds_active,
        bins=bins_ds,
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color="#1f77b4",
        edgecolor="#1f77b4",
        lw=1.8,
        label=f"MC ($E_{{th}}^* = {ds_th_opt:.2f}$ MeV, mean={mc_ds_mean:.2f})",
    )

    ax2.set_title("DS Hit Multiplicity Distribution (Active Events)", fontsize=14, fontweight="bold")
    ax2.set_xlabel(r"$N_{\rm hits}^{\rm DS}$ per event", fontsize=12)
    ax2.set_ylabel("Probability Density", fontsize=12)
    ax2.set_ylim(bottom=0.0)
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 3: US Plane-by-Plane Hit Multiplicity (Planes 0 to 4)
    # =========================================================================
    ax3 = axes[1, 0]
    planes = np.arange(5)
    data_pl_means = [float(np.mean(data_res["us_plane_hits"][p])) for p in planes]
    mc_pl_means = [float(np.mean(mc_res["us_plane_hits"][p])) for p in planes]

    width = 0.35
    ax3.bar(planes - width/2, data_pl_means, width=width, color="#d62728", alpha=0.85, label="Data (mean hits/plane)")
    ax3.bar(planes + width/2, mc_pl_means, width=width, color="#1f77b4", alpha=0.75, label=f"MC ($E_{{th}} = {us_th_opt:.1f}$ MeV)")

    ax3.set_title("US Plane Hit Multiplicity Comparison (Planes 0-4)", fontsize=14, fontweight="bold")
    ax3.set_xlabel("US Plane Index", fontsize=12)
    ax3.set_ylabel(r"$\langle N_{\rm hits}\rangle$ per event", fontsize=12)
    ax3.set_xticks(planes)
    ax3.set_xticklabels([f"Plane {p}" for p in planes])
    ax3.set_ylim(bottom=0.0, top=max(max(data_pl_means), max(mc_pl_means)) * 1.35)
    ax3.legend(loc="upper right", frameon=True, fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")

    # =========================================================================
    # Panel 4: DS Station 0 View Breakdown (Horizontal vs Vertical Bars)
    # =========================================================================
    ax4 = axes[1, 1]
    views = ["Horizontal (Bars 0-59)", "Vertical (Bars 60-119)"]
    data_view_means = [
        float(np.mean(data_res["ds_horiz_hits"][mask_ds_data])),
        float(np.mean(data_res["ds_vert_hits"][mask_ds_data])),
    ]
    mc_view_means = [
        float(np.mean(mc_res["ds_horiz_hits"][mask_ds_mc])),
        float(np.mean(mc_res["ds_vert_hits"][mask_ds_mc])),
    ]

    x_idx = np.arange(2)
    ax4.bar(x_idx - width/2, data_view_means, width=width, color="#d62728", alpha=0.85, label="Data Active")
    ax4.bar(x_idx + width/2, mc_view_means, width=width, color="#1f77b4", alpha=0.75, label=f"MC ($E_{{th}}^* = {ds_th_opt:.2f}$ MeV)")

    ax4.set_title("DS Station 0 View Breakdown (Active Events)", fontsize=14, fontweight="bold")
    ax4.set_xlabel("DS Readout Orientation", fontsize=12)
    ax4.set_ylabel(r"$\langle N_{\rm hits}\rangle$ per active event", fontsize=12)
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(views)
    ax4.set_ylim(bottom=0.0, top=max(max(data_view_means), max(mc_view_means)) * 1.35)
    ax4.legend(loc="upper right", frameon=True, fontsize=10)
    ax4.grid(True, alpha=0.3, axis="y")

    out_png = out_prefix + ".png"
    out_pdf = out_prefix + ".pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"Saved publication plot: {out_png}")
    print(f"Saved vector PDF: {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="MuFilter TB Detailed Comparison")
    parser.add_argument("--max-events", type=int, default=1500, help="Max events to process per sample")
    parser.add_argument("--ds-th", type=float, default=1.29, help="Calibrated DS threshold in MeV")
    parser.add_argument("--us-th", type=float, default=6.0, help="Calibrated US threshold in MeV")
    parser.add_argument("-o", "--output-prefix", type=str, default="/eos/user/i/idioniso/snd-numu-charm/plots/mufilter_study/tb_detailed_comparison")
    args = parser.parse_args()

    data_files = [
        "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100639/sndsw_raw-0000.root",
        "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100639/sndsw_raw-0001.root",
    ]
    mc_file = "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/300GeV_-211/sndLHC.PG_-211-TGeant4_digCPP.root"

    data_res = get_data_events(data_files, max_events=args.max_events)
    mc_res = get_mc_events(mc_file, max_events=args.max_events, ds_threshold_mev=args.ds_th, us_threshold_mev=args.us_th)

    make_plots(data_res, mc_res, ds_th_opt=args.ds_th, us_th_opt=args.us_th, out_prefix=args.output_prefix)


if __name__ == "__main__":
    main()
