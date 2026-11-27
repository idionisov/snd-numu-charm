#!/usr/bin/env python3
"""
Complete MuFilter Testbeam Calibration & Detailed Comparison
Author: Antigravity Assistant

Features:
- Dynamically conditioned threshold crossing for both Downstream (DS) and Upstream (US)
  where the active event condition (N >= 1) is consistently enforced at every threshold.
- Overlapping hit distributions comparing:
    * MC Raw (without threshold, E_th = 0) in grey
    * MC Calibrated (with threshold E_th*) as solid blue line
    * Data Active in red dashed line
- Subsystem and Plane/View breakdowns for both US and DS with consistent means.
- Stores all multi-panel TCanvases, TH1 histograms, and TGraphs directly into the output ROOT file.
- Saves publication PNG and vector PDF figures.
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

try:
    import ddfUtils.mpl_styling as dms
    dms.apply_style()
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def get_data_events(
    data_files: List[str],
    max_events: int = 4000,
    max_delta_t_ns: float = 10.0,
) -> Dict[str, any]:
    """Read data events and extract US/DS hits and plane/view multiplicities."""
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
    max_events: int = 4000,
    ds_scan_thresholds: np.ndarray = np.linspace(0.0, 5.0, 51),
    us_scan_thresholds: np.ndarray = np.linspace(0.0, 80.0, 81),
) -> Dict[str, any]:
    """Read MC events, extract bar energy deposits, and scan thresholds with consistent active conditioning."""
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

    events_us_data = []  # list of list of (plane, edep)
    events_ds_data = []  # list of list of (bar, edep)

    for iev in range(n_entries):
        if 0 < max_events <= total_events:
            break
        tree.GetEntry(iev)
        total_events += 1

        bar_edep: Dict[int, float] = {}
        for pt in tree.MuFilterPoint:
            did = pt.GetDetectorID()
            bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss() * 1000.0

        ev_us = []
        ev_ds = []

        for hit in tree.Digi_MuFilterHits:
            if not hit.isValid():
                continue
            did = hit.GetDetectorID()
            sub = did // 10000
            edep = bar_edep.get(did, 0.0)

            if sub == 2:  # US
                pl = (did // 1000) % 10
                ev_us.append((pl, edep))
            elif sub == 3:  # DS
                bar = did % 1000
                ev_ds.append((bar, edep))

        events_us_data.append(ev_us)
        events_ds_data.append(ev_ds)

    f_in.Close()
    print(f"MC read: {total_events} events in {time.time()-t0:.1f} s.")

    # Compute DS threshold scan curve conditioned on N(th) >= 1
    ds_curve = []
    for th in ds_scan_thresholds:
        counts = [(np.array([e for _, e in ev]) >= th).sum() for ev in events_ds_data]
        act = [c for c in counts if c >= 1]
        ds_curve.append(float(np.mean(act)) if act else 0.0)
    ds_curve = np.array(ds_curve)

    # Compute US threshold scan curve conditioned on N(th) >= 1
    us_curve = []
    for th in us_scan_thresholds:
        counts = [(np.array([e for _, e in ev]) >= th).sum() for ev in events_us_data]
        act = [c for c in counts if c >= 1]
        us_curve.append(float(np.mean(act)) if act else 0.0)
    us_curve = np.array(us_curve)

    return {
        "events_us": events_us_data,
        "events_ds": events_ds_data,
        "ds_scan_thresholds": ds_scan_thresholds,
        "ds_scan_curve": ds_curve,
        "us_scan_thresholds": us_scan_thresholds,
        "us_scan_curve": us_curve,
    }


def interpolate_crossing(xs: np.ndarray, ys: np.ndarray, target: float) -> float:
    """Find threshold where ys crosses target value."""
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


def evaluate_mc_at_threshold(events_data: List[List[Tuple[int, float]]], threshold: float) -> List[List[int]]:
    """Return surviving hit elements per event given an energy deposit threshold."""
    res = []
    for ev in events_data:
        surv = [elem for elem, edep in ev if edep >= threshold]
        res.append(surv)
    return res


def main():
    parser = argparse.ArgumentParser(description="Complete MuFilter TB Calibration & Comparison")
    parser.add_argument("--max-events", type=int, default=4000, help="Max events to process per sample")
    parser.add_argument("-o", "--output-root", type=str, default="/eos/user/i/idioniso/snd-numu-charm/plots/mufilter_study/calib_mufilter_tb_detailed.root")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output_root))
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Data
    data_files = [
        "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100639/sndsw_raw-0000.root",
        "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100639/sndsw_raw-0001.root",
        "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100639/sndsw_raw-0002.root",
    ]
    data_res = get_data_events(data_files, max_events=args.max_events)

    # Data Active selections
    mask_ds_data = data_res["ds_hits"] >= 1
    data_ds_active = data_res["ds_hits"][mask_ds_data]
    data_ds_mean = float(np.mean(data_ds_active)) if len(data_ds_active) > 0 else 0.0

    mask_us_data = data_res["us_hits"] >= 1
    data_us_active = data_res["us_hits"][mask_us_data]
    data_us_mean = float(np.mean(data_us_active)) if len(data_us_active) > 0 else 0.0

    print(f"\n--- Data Active Summary ---")
    print(f"  DS Active Events: {len(data_ds_active)}/{len(data_res['ds_hits'])}, <N_DS> = {data_ds_mean:.2f}")
    print(f"  US Active Events: {len(data_us_active)}/{len(data_res['us_hits'])}, <N_US> = {data_us_mean:.2f}")

    # 2. Load MC & Threshold Scans
    mc_file = "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/300GeV_-211/sndLHC.PG_-211-TGeant4_digCPP.root"
    mc_res = get_mc_events(mc_file, max_events=args.max_events)

    # Find Dynamic Crossing Points
    ds_th_opt = interpolate_crossing(mc_res["ds_scan_thresholds"], mc_res["ds_scan_curve"], data_ds_mean)
    us_th_opt = interpolate_crossing(mc_res["us_scan_thresholds"], mc_res["us_scan_curve"], data_us_mean)

    print(f"\n=======================================================")
    print(f"  OPTIMAL ENERGY DEPOSIT THRESHOLDS (CONSISTENT CONDITIONING)")
    print(f"=======================================================")
    print(f"  Optimal Downstream (DS) Threshold E_th* : {ds_th_opt:.2f} MeV (matches <N_DS> = {data_ds_mean:.2f})")
    print(f"  Optimal Upstream   (US) Threshold E_th* : {us_th_opt:.2f} MeV (matches <N_US> = {data_us_mean:.2f})")
    print(f"=======================================================\n")

    # 3. Evaluate MC at E_th = 0 (Raw) and E_th = E_th* (Calibrated)
    # DS
    mc_ds_raw_evs = evaluate_mc_at_threshold(mc_res["events_ds"], threshold=0.0)
    mc_ds_cal_evs = evaluate_mc_at_threshold(mc_res["events_ds"], threshold=ds_th_opt)

    mc_ds_raw_counts = np.array([len(ev) for ev in mc_ds_raw_evs])
    mc_ds_cal_counts = np.array([len(ev) for ev in mc_ds_cal_evs])

    mc_ds_raw_active = mc_ds_raw_counts[mc_ds_raw_counts >= 1]
    mc_ds_cal_active = mc_ds_cal_counts[mc_ds_cal_counts >= 1]
    mc_ds_cal_mean = float(np.mean(mc_ds_cal_active))

    # US
    mc_us_raw_evs = evaluate_mc_at_threshold(mc_res["events_us"], threshold=0.0)
    mc_us_cal_evs = evaluate_mc_at_threshold(mc_res["events_us"], threshold=us_th_opt)

    mc_us_raw_counts = np.array([len(ev) for ev in mc_us_raw_evs])
    mc_us_cal_counts = np.array([len(ev) for ev in mc_us_cal_evs])

    mc_us_raw_active = mc_us_raw_counts[mc_us_raw_counts >= 1]
    mc_us_cal_active = mc_us_cal_counts[mc_us_cal_counts >= 1]
    mc_us_cal_mean = float(np.mean(mc_us_cal_active))

    # Views for DS
    data_ds_views = [
        float(np.mean(data_res["ds_horiz_hits"][mask_ds_data])),
        float(np.mean(data_res["ds_vert_hits"][mask_ds_data])),
    ]
    mc_raw_h = [sum(1 for b in ev if b < 60) for ev in mc_ds_raw_evs if len(ev) >= 1]
    mc_raw_v = [sum(1 for b in ev if b >= 60) for ev in mc_ds_raw_evs if len(ev) >= 1]
    mc_cal_h = [sum(1 for b in ev if b < 60) for ev in mc_ds_cal_evs if len(ev) >= 1]
    mc_cal_v = [sum(1 for b in ev if b >= 60) for ev in mc_ds_cal_evs if len(ev) >= 1]

    mc_raw_views = [float(np.mean(mc_raw_h)), float(np.mean(mc_raw_v))]
    mc_cal_views = [float(np.mean(mc_cal_h)), float(np.mean(mc_cal_v))]

    # Planes for US
    planes = np.arange(5)
    data_us_pl_means = [float(np.mean(data_res["us_plane_hits"][p][mask_us_data])) for p in planes]
    mc_raw_pl = {p: [sum(1 for pl in ev if pl == p) for ev in mc_us_raw_evs if len(ev) >= 1] for p in planes}
    mc_cal_pl = {p: [sum(1 for pl in ev if pl == p) for ev in mc_us_cal_evs if len(ev) >= 1] for p in planes}

    mc_raw_pl_means = [float(np.mean(mc_raw_pl[p])) for p in planes]
    mc_cal_pl_means = [float(np.mean(mc_cal_pl[p])) for p in planes]

    # =========================================================================
    # MATPLOTLIB FIGURE 1: Downstream (DS)
    # =========================================================================
    fig_ds, axes_ds = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.28)

    # Panel 1: Scan & Crossing
    ax1 = axes_ds[0]
    th_ds = mc_res["ds_scan_thresholds"]
    crv_ds = mc_res["ds_scan_curve"]
    ax1.plot(th_ds, crv_ds, color="#1f77b4", lw=2.8, label=r"Simulation (MC $\langle N_{\rm DS}\rangle$)")
    ax1.axhline(data_ds_mean, color="#d62728", ls="--", lw=2.2, label=f"Data Mean: {data_ds_mean:.2f} hits")
    ax1.axvline(ds_th_opt, color="#2ca02c", ls=":", lw=2.0, label=f"Optimal $E_{{th}}^* = {ds_th_opt:.2f}$ MeV")
    ax1.plot([ds_th_opt], [data_ds_mean], marker="o", markersize=9, color="#d62728", markeredgecolor="black", zorder=5)
    ax1.set_title("DS Threshold Scan & Exact Crossing", fontsize=13, fontweight="bold")
    ax1.set_xlabel(r"Energy Deposit Threshold $E_{\rm dep}$ [MeV]", fontsize=11)
    ax1.set_ylabel(r"$\langle N_{\rm hits}^{\rm DS}\rangle$ per active event", fontsize=11)
    ax1.set_ylim(bottom=0.0, top=float(crv_ds[0]) * 1.15)
    ax1.set_xlim(left=0.0, right=th_ds[-1])
    ax1.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax1.grid(True, alpha=0.3)
    ax1.annotate(
        f"Intersection:\n$E_{{th}}^* = {ds_th_opt:.2f}$ MeV\n$<N_{{hits}}> = {data_ds_mean:.2f}$",
        xy=(ds_th_opt, data_ds_mean),
        xytext=(ds_th_opt + 0.8, data_ds_mean + 1.8),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2ae", edgecolor="#b59410", alpha=0.9),
        fontsize=9.5,
        fontweight="bold",
    )

    # Panel 2: Multiplicity Distribution
    ax2 = axes_ds[1]
    bins_ds = np.arange(0.5, 36.5, 1.0)
    ax2.hist(
        mc_ds_raw_active,
        bins=bins_ds,
        density=True,
        histtype="stepfilled",
        alpha=0.30,
        color="#7f7f7f",
        edgecolor="#555555",
        lw=1.5,
        label=f"MC Raw ($E_{{th}}=0$, mean={np.mean(mc_ds_raw_active):.2f})",
    )
    ax2.hist(
        mc_ds_cal_active,
        bins=bins_ds,
        density=True,
        histtype="step",
        color="#1f77b4",
        lw=2.5,
        label=f"MC Calib ($E_{{th}}^*={ds_th_opt:.2f}$ MeV, mean={mc_ds_cal_mean:.2f})",
    )
    ax2.hist(
        data_ds_active,
        bins=bins_ds,
        density=True,
        histtype="step",
        color="#d62728",
        lw=2.5,
        ls="--",
        label=f"Data Active (mean={data_ds_mean:.2f})",
    )
    ax2.set_title("DS Hit Multiplicity Distribution", fontsize=13, fontweight="bold")
    ax2.set_xlabel(r"$N_{\rm hits}^{\rm DS}$ per event", fontsize=11)
    ax2.set_ylabel("Probability Density", fontsize=11)
    ax2.set_ylim(bottom=0.0)
    ax2.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Views
    ax3 = axes_ds[2]
    views = ["Horizontal\n(Bars 0-59)", "Vertical\n(Bars 60-119)"]
    x_idx = np.arange(2)
    width = 0.26
    ax3.bar(x_idx - width, mc_raw_views, width=width, color="#7f7f7f", alpha=0.55, edgecolor="#444", label="MC Raw ($E_{{th}}=0$)")
    ax3.bar(x_idx, mc_cal_views, width=width, color="#1f77b4", alpha=0.85, edgecolor="#114b75", label=f"MC Calib ({ds_th_opt:.2f} MeV)")
    ax3.bar(x_idx + width, data_ds_views, width=width, color="#d62728", alpha=0.85, edgecolor="#8b0000", label="Data Active")
    ax3.set_title("DS Station 0 Readout Orientation", fontsize=13, fontweight="bold")
    ax3.set_ylabel(r"$\langle N_{\rm hits}\rangle$ per active event", fontsize=11)
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(views, fontsize=10)
    ax3.set_ylim(bottom=0.0, top=max(mc_raw_views) * 1.30)
    ax3.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax3.grid(True, alpha=0.3, axis="y")

    ds_plot_png = os.path.join(out_dir, "calib_mufilter_ds_detailed.png")
    ds_plot_pdf = os.path.join(out_dir, "calib_mufilter_ds_detailed.pdf")
    fig_ds.savefig(ds_plot_png, dpi=300, bbox_inches="tight")
    fig_ds.savefig(ds_plot_pdf, bbox_inches="tight")
    plt.close(fig_ds)
    print(f"Saved DS plots: {ds_plot_png}")

    # =========================================================================
    # MATPLOTLIB FIGURE 2: Upstream (US)
    # =========================================================================
    fig_us, axes_us = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.28)

    # Panel 1: US Scan & Crossing
    ax1_u = axes_us[0]
    th_us = mc_res["us_scan_thresholds"]
    crv_us = mc_res["us_scan_curve"]
    ax1_u.plot(th_us, crv_us, color="#1f77b4", lw=2.8, label=r"Simulation (MC $\langle N_{\rm US}\rangle$)")
    ax1_u.axhline(data_us_mean, color="#d62728", ls="--", lw=2.2, label=f"Data Mean: {data_us_mean:.2f} hits")
    ax1_u.axvline(us_th_opt, color="#2ca02c", ls=":", lw=2.0, label=f"Optimal $E_{{th}}^* = {us_th_opt:.2f}$ MeV")
    ax1_u.plot([us_th_opt], [data_us_mean], marker="o", markersize=9, color="#d62728", markeredgecolor="black", zorder=5)
    ax1_u.set_title("US Threshold Scan & Exact Crossing", fontsize=13, fontweight="bold")
    ax1_u.set_xlabel(r"Energy Deposit Threshold $E_{\rm dep}$ [MeV]", fontsize=11)
    ax1_u.set_ylabel(r"$\langle N_{\rm hits}^{\rm US}\rangle$ per active event", fontsize=11)
    ax1_u.set_ylim(bottom=0.0, top=float(crv_us[0]) * 1.15)
    ax1_u.set_xlim(left=0.0, right=th_us[-1])
    ax1_u.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax1_u.grid(True, alpha=0.3)
    ax1_u.annotate(
        f"Intersection:\n$E_{{th}}^* = {us_th_opt:.2f}$ MeV\n$<N_{{hits}}> = {data_us_mean:.2f}$",
        xy=(us_th_opt, data_us_mean),
        xytext=(us_th_opt + 4.0, data_us_mean + 4.0),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2ae", edgecolor="#b59410", alpha=0.9),
        fontsize=9.5,
        fontweight="bold",
    )

    # Panel 2: Multiplicity Distribution
    ax2_u = axes_us[1]
    bins_us = np.arange(0.5, 45.5, 1.0)
    ax2_u.hist(
        mc_us_raw_active,
        bins=bins_us,
        density=True,
        histtype="stepfilled",
        alpha=0.30,
        color="#7f7f7f",
        edgecolor="#555555",
        lw=1.5,
        label=f"MC Raw ($E_{{th}}=0$, mean={np.mean(mc_us_raw_active):.2f})",
    )
    ax2_u.hist(
        mc_us_cal_active,
        bins=bins_us,
        density=True,
        histtype="step",
        color="#1f77b4",
        lw=2.5,
        label=f"MC Calib ($E_{{th}}^*={us_th_opt:.2f}$ MeV, mean={mc_us_cal_mean:.2f})",
    )
    ax2_u.hist(
        data_us_active,
        bins=bins_us,
        density=True,
        histtype="step",
        color="#d62728",
        lw=2.5,
        ls="--",
        label=f"Data Active (mean={data_us_mean:.2f})",
    )
    ax2_u.set_title("US Hit Multiplicity Distribution", fontsize=13, fontweight="bold")
    ax2_u.set_xlabel(r"$N_{\rm hits}^{\rm US}$ per event", fontsize=11)
    ax2_u.set_ylabel("Probability Density", fontsize=11)
    ax2_u.set_ylim(bottom=0.0)
    ax2_u.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax2_u.grid(True, alpha=0.3)

    # Panel 3: Planes
    ax3_u = axes_us[2]
    ax3_u.bar(planes - width, mc_raw_pl_means, width=width, color="#7f7f7f", alpha=0.55, edgecolor="#444", label="MC Raw ($E_{{th}}=0$)")
    ax3_u.bar(planes, mc_cal_pl_means, width=width, color="#1f77b4", alpha=0.85, edgecolor="#114b75", label=f"MC Calib ({us_th_opt:.2f} MeV)")
    ax3_u.bar(planes + width, data_us_pl_means, width=width, color="#d62728", alpha=0.85, edgecolor="#8b0000", label="Data Active")
    ax3_u.set_title("US Plane Multiplicity (Planes 0-4)", fontsize=13, fontweight="bold")
    ax3_u.set_xlabel("US Plane Index", fontsize=11)
    ax3_u.set_ylabel(r"$\langle N_{\rm hits}\rangle$ per active event", fontsize=11)
    ax3_u.set_xticks(planes)
    ax3_u.set_xticklabels([f"Plane {p}" for p in planes], fontsize=10)
    ax3_u.set_ylim(bottom=0.0, top=max(mc_raw_pl_means) * 1.30)
    ax3_u.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax3_u.grid(True, alpha=0.3, axis="y")

    us_plot_png = os.path.join(out_dir, "calib_mufilter_us_detailed.png")
    us_plot_pdf = os.path.join(out_dir, "calib_mufilter_us_detailed.pdf")
    fig_us.savefig(us_plot_png, dpi=300, bbox_inches="tight")
    fig_us.savefig(us_plot_pdf, bbox_inches="tight")
    plt.close(fig_us)
    print(f"Saved US plots: {us_plot_png}")

    # =========================================================================
    # CREATE ROOT CANVASES & HISTOGRAMS TO STORE IN ROOT FILE
    # =========================================================================
    ROOT.gStyle.SetOptStat(0)
    f_root = ROOT.TFile.Open(args.output_root, "RECREATE")
    print(f"Writing complete TCanvases and histograms to ROOT file: {args.output_root}...")

    # --- DS ROOT Canvas ---
    c_ds = ROOT.TCanvas("c_ds_summary", "Downstream (DS) Testbeam Calibration Summary", 1800, 600)
    c_ds.Divide(3, 1)

    # Pad 1: Scan
    c_ds.cd(1)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    g_ds_scan = ROOT.TGraph(len(th_ds), th_ds, crv_ds)
    g_ds_scan.SetName("g_ds_scan")
    g_ds_scan.SetTitle("DS Threshold Scan;Threshold E_{dep} [MeV];<N_{hits}^{DS}> per active event")
    g_ds_scan.SetLineColor(ROOT.kAzure + 2)
    g_ds_scan.SetLineWidth(3)
    g_ds_scan.SetMinimum(0.0)
    g_ds_scan.SetMaximum(float(crv_ds[0]) * 1.15)
    g_ds_scan.Draw("AL")

    line_ds_data = ROOT.TLine(0, data_ds_mean, th_ds[-1], data_ds_mean)
    line_ds_data.SetLineColor(ROOT.kRed + 1)
    line_ds_data.SetLineStyle(2)
    line_ds_data.SetLineWidth(2)
    line_ds_data.Draw()

    line_ds_th = ROOT.TLine(ds_th_opt, 0, ds_th_opt, data_ds_mean)
    line_ds_th.SetLineColor(ROOT.kGreen + 2)
    line_ds_th.SetLineStyle(3)
    line_ds_th.SetLineWidth(2)
    line_ds_th.Draw()

    leg_ds1 = ROOT.TLegend(0.40, 0.70, 0.88, 0.88)
    leg_ds1.AddEntry(g_ds_scan, "Simulation (MC)", "l")
    leg_ds1.AddEntry(line_ds_data, f"Data Mean: {data_ds_mean:.2f}", "l")
    leg_ds1.AddEntry(line_ds_th, f"Optimal E_{{th}}^* = {ds_th_opt:.2f} MeV", "l")
    leg_ds1.Draw()

    # Pad 2: Hit Multiplicity
    c_ds.cd(2)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    h_ds_mc_raw = ROOT.TH1D("h_ds_hits_mc_raw", "DS Hit Multiplicity;N_{hits}^{DS};Normalized Events", 36, 0.5, 36.5)
    for x in mc_ds_raw_active: h_ds_mc_raw.Fill(x)
    if h_ds_mc_raw.Integral() > 0: h_ds_mc_raw.Scale(1.0 / h_ds_mc_raw.Integral())
    h_ds_mc_raw.SetFillColor(ROOT.kGray)
    h_ds_mc_raw.SetLineColor(ROOT.kGray + 2)
    h_ds_mc_raw.SetMinimum(0.0)
    h_ds_mc_raw.Draw("HIST")

    h_ds_mc_cal = ROOT.TH1D("h_ds_hits_mc_calib", f"DS Hits MC Calib;N_{{hits}}^{{DS}};Normalized Events", 36, 0.5, 36.5)
    for x in mc_ds_cal_active: h_ds_mc_cal.Fill(x)
    if h_ds_mc_cal.Integral() > 0: h_ds_mc_cal.Scale(1.0 / h_ds_mc_cal.Integral())
    h_ds_mc_cal.SetLineColor(ROOT.kAzure + 2)
    h_ds_mc_cal.SetLineWidth(3)
    h_ds_mc_cal.Draw("HIST SAME")

    h_ds_data = ROOT.TH1D("h_ds_hits_data", "DS Hits Data;N_{hits}^{DS};Normalized Events", 36, 0.5, 36.5)
    for x in data_ds_active: h_ds_data.Fill(x)
    if h_ds_data.Integral() > 0: h_ds_data.Scale(1.0 / h_ds_data.Integral())
    h_ds_data.SetLineColor(ROOT.kRed + 1)
    h_ds_data.SetLineStyle(2)
    h_ds_data.SetLineWidth(3)
    h_ds_data.Draw("HIST SAME")

    leg_ds2 = ROOT.TLegend(0.40, 0.70, 0.88, 0.88)
    leg_ds2.AddEntry(h_ds_mc_raw, f"MC Raw (mean={np.mean(mc_ds_raw_active):.2f})", "f")
    leg_ds2.AddEntry(h_ds_mc_cal, f"MC Calib (mean={mc_ds_cal_mean:.2f})", "l")
    leg_ds2.AddEntry(h_ds_data, f"Data (mean={data_ds_mean:.2f})", "l")
    leg_ds2.Draw()

    # Pad 3: Views
    c_ds.cd(3)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    h_ds_views_mc_raw = ROOT.TH1D("h_ds_views_mc_raw", "DS Station 0 Readout;Readout Orientation;<N_{hits}>", 2, 0, 2)
    h_ds_views_mc_raw.GetXaxis().SetBinLabel(1, "Horizontal")
    h_ds_views_mc_raw.GetXaxis().SetBinLabel(2, "Vertical")
    h_ds_views_mc_raw.SetBinContent(1, mc_raw_views[0])
    h_ds_views_mc_raw.SetBinContent(2, mc_raw_views[1])
    h_ds_views_mc_raw.SetFillColor(ROOT.kGray)
    h_ds_views_mc_raw.SetLineColor(ROOT.kGray + 2)
    h_ds_views_mc_raw.SetMinimum(0.0)
    h_ds_views_mc_raw.SetMaximum(max(mc_raw_views) * 1.35)
    h_ds_views_mc_raw.Draw("BAR")

    h_ds_views_mc_cal = ROOT.TH1D("h_ds_views_mc_calib", "DS Station 0 MC Calib;Orientation;<N_{hits}>", 2, 0, 2)
    h_ds_views_mc_cal.SetBinContent(1, mc_cal_views[0])
    h_ds_views_mc_cal.SetBinContent(2, mc_cal_views[1])
    h_ds_views_mc_cal.SetFillColor(ROOT.kAzure - 4)
    h_ds_views_mc_cal.SetLineColor(ROOT.kAzure + 2)
    h_ds_views_mc_cal.Draw("BAR SAME")

    h_ds_views_data = ROOT.TH1D("h_ds_views_data", "DS Station 0 Views Data;Orientation;<N_{hits}>", 2, 0, 2)
    h_ds_views_data.SetBinContent(1, data_ds_views[0])
    h_ds_views_data.SetBinContent(2, data_ds_views[1])
    h_ds_views_data.SetFillColor(ROOT.kRed - 7)
    h_ds_views_data.SetLineColor(ROOT.kRed + 1)
    h_ds_views_data.Draw("BAR SAME")

    leg_ds3 = ROOT.TLegend(0.40, 0.70, 0.88, 0.88)
    leg_ds3.AddEntry(h_ds_views_mc_raw, "MC Raw", "f")
    leg_ds3.AddEntry(h_ds_views_mc_cal, "MC Calib", "f")
    leg_ds3.AddEntry(h_ds_views_data, "Data Active", "f")
    leg_ds3.Draw()

    c_ds.Update()
    c_ds.Write()

    # --- US ROOT Canvas ---
    c_us = ROOT.TCanvas("c_us_summary", "Upstream (US) Testbeam Calibration Summary", 1800, 600)
    c_us.Divide(3, 1)

    # Pad 1: US Scan
    c_us.cd(1)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    g_us_scan = ROOT.TGraph(len(th_us), th_us, crv_us)
    g_us_scan.SetName("g_us_scan")
    g_us_scan.SetTitle("US Threshold Scan;Threshold E_{dep} [MeV];<N_{hits}^{US}> per active event")
    g_us_scan.SetLineColor(ROOT.kAzure + 2)
    g_us_scan.SetLineWidth(3)
    g_us_scan.SetMinimum(0.0)
    g_us_scan.SetMaximum(float(crv_us[0]) * 1.15)
    g_us_scan.Draw("AL")

    line_us_data = ROOT.TLine(0, data_us_mean, th_us[-1], data_us_mean)
    line_us_data.SetLineColor(ROOT.kRed + 1)
    line_us_data.SetLineStyle(2)
    line_us_data.SetLineWidth(2)
    line_us_data.Draw()

    line_us_th = ROOT.TLine(us_th_opt, 0, us_th_opt, data_us_mean)
    line_us_th.SetLineColor(ROOT.kGreen + 2)
    line_us_th.SetLineStyle(3)
    line_us_th.SetLineWidth(2)
    line_us_th.Draw()

    leg_us1 = ROOT.TLegend(0.40, 0.70, 0.88, 0.88)
    leg_us1.AddEntry(g_us_scan, "Simulation (MC)", "l")
    leg_us1.AddEntry(line_us_data, f"Data Mean: {data_us_mean:.2f}", "l")
    leg_us1.AddEntry(line_us_th, f"Optimal E_{{th}}^* = {us_th_opt:.2f} MeV", "l")
    leg_us1.Draw()

    # Pad 2: US Multiplicity
    c_us.cd(2)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    h_us_mc_raw = ROOT.TH1D("h_us_hits_mc_raw", "US Hit Multiplicity;N_{hits}^{US};Normalized Events", 45, 0.5, 45.5)
    for x in mc_us_raw_active: h_us_mc_raw.Fill(x)
    if h_us_mc_raw.Integral() > 0: h_us_mc_raw.Scale(1.0 / h_us_mc_raw.Integral())
    h_us_mc_raw.SetFillColor(ROOT.kGray)
    h_us_mc_raw.SetLineColor(ROOT.kGray + 2)
    h_us_mc_raw.SetMinimum(0.0)
    h_us_mc_raw.Draw("HIST")

    h_us_mc_cal = ROOT.TH1D("h_us_hits_mc_calib", f"US Hits MC Calib;N_{{hits}}^{{US}};Normalized Events", 45, 0.5, 45.5)
    for x in mc_us_cal_active: h_us_mc_cal.Fill(x)
    if h_us_mc_cal.Integral() > 0: h_us_mc_cal.Scale(1.0 / h_us_mc_cal.Integral())
    h_us_mc_cal.SetLineColor(ROOT.kAzure + 2)
    h_us_mc_cal.SetLineWidth(3)
    h_us_mc_cal.Draw("HIST SAME")

    h_us_data = ROOT.TH1D("h_us_hits_data", "US Hits Data;N_{hits}^{US};Normalized Events", 45, 0.5, 45.5)
    for x in data_us_active: h_us_data.Fill(x)
    if h_us_data.Integral() > 0: h_us_data.Scale(1.0 / h_us_data.Integral())
    h_us_data.SetLineColor(ROOT.kRed + 1)
    h_us_data.SetLineStyle(2)
    h_us_data.SetLineWidth(3)
    h_us_data.Draw("HIST SAME")

    leg_us2 = ROOT.TLegend(0.40, 0.70, 0.88, 0.88)
    leg_us2.AddEntry(h_us_mc_raw, f"MC Raw (mean={np.mean(mc_us_raw_active):.2f})", "f")
    leg_us2.AddEntry(h_us_mc_cal, f"MC Calib (mean={mc_us_cal_mean:.2f})", "l")
    leg_us2.AddEntry(h_us_data, f"Data (mean={data_us_mean:.2f})", "l")
    leg_us2.Draw()

    # Pad 3: US Planes
    c_us.cd(3)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    h_us_planes_mc_raw = ROOT.TH1D("h_us_planes_mc_raw", "US Planes Multiplicity;Plane Index;<N_{hits}>", 5, 0, 5)
    for p in range(5):
        h_us_planes_mc_raw.GetXaxis().SetBinLabel(p + 1, f"Plane {p}")
        h_us_planes_mc_raw.SetBinContent(p + 1, mc_raw_pl_means[p])
    h_us_planes_mc_raw.SetFillColor(ROOT.kGray)
    h_us_planes_mc_raw.SetLineColor(ROOT.kGray + 2)
    h_us_planes_mc_raw.SetMinimum(0.0)
    h_us_planes_mc_raw.SetMaximum(max(mc_raw_pl_means) * 1.35)
    h_us_planes_mc_raw.Draw("BAR")

    h_us_planes_mc_cal = ROOT.TH1D("h_us_planes_mc_calib", "US Planes MC Calib;Plane Index;<N_{hits}>", 5, 0, 5)
    for p in range(5):
        h_us_planes_mc_cal.SetBinContent(p + 1, mc_cal_pl_means[p])
    h_us_planes_mc_cal.SetFillColor(ROOT.kAzure - 4)
    h_us_planes_mc_cal.SetLineColor(ROOT.kAzure + 2)
    h_us_planes_mc_cal.Draw("BAR SAME")

    h_us_planes_data = ROOT.TH1D("h_us_planes_data", "US Planes Multiplicity Data;Plane Index;<N_{hits}>", 5, 0, 5)
    for p in range(5):
        h_us_planes_data.SetBinContent(p + 1, data_us_pl_means[p])
    h_us_planes_data.SetFillColor(ROOT.kRed - 7)
    h_us_planes_data.SetLineColor(ROOT.kRed + 1)
    h_us_planes_data.Draw("BAR SAME")

    leg_us3 = ROOT.TLegend(0.40, 0.70, 0.88, 0.88)
    leg_us3.AddEntry(h_us_planes_mc_raw, "MC Raw", "f")
    leg_us3.AddEntry(h_us_planes_mc_cal, "MC Calib", "f")
    leg_us3.AddEntry(h_us_planes_data, "Data Active", "f")
    leg_us3.Draw()

    c_us.Update()
    c_us.Write()

    # Also write standalone histograms
    g_ds_scan.Write()
    h_ds_data.Write()
    h_ds_mc_raw.Write()
    h_ds_mc_cal.Write()
    h_ds_views_data.Write()
    h_ds_views_mc_raw.Write()
    h_ds_views_mc_cal.Write()

    g_us_scan.Write()
    h_us_data.Write()
    h_us_mc_raw.Write()
    h_us_mc_cal.Write()
    h_us_planes_data.Write()
    h_us_planes_mc_raw.Write()
    h_us_planes_mc_cal.Write()

    f_root.Close()
    print(f"Saved complete ROOT output with TCanvases: {args.output_root}")


if __name__ == "__main__":
    main()
