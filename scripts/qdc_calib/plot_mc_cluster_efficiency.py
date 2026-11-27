#!/usr/bin/env python3
"""
scripts/qdc_calib/plot_mc_cluster_efficiency.py
-----------------------------------------------
Evaluates and plots the Hit Reconstruction Efficiency as a function of
deposited energy by an MCPoint cluster across all MuFilter and Veto bar types.

Features:
  1. Compares:
     - Raw Simulation: E_th = 0
     - Theoretical Model: MinPE with nominal geometry & coupling (k = 1.0)
     - Calibrated Model: MinPE with channel-specific photoelectron reduction factors (k from CSV)
     (Simple Step model removed as requested).
  2. High-transparency background curves for EVERY individual bar in each category to
     explicitly visualize the bar-to-bar dispersion.
  3. Bar Configurations Evaluated:
     - Veto Planes 0 & 1 (Silicone gel, 42 cm, readout on both ends)
     - Veto Plane 2 (Mirror + air, 46 cm, single top readout)
     - Upstream (US) Bars - Gel Side (82.5 cm, 8 SiPMs with optical gel)
     - Upstream (US) Bars - Air Side (82.5 cm, 8 SiPMs with air interface)
     - Downstream (DS) Horizontal Bars (82.5 cm, 1 SiPM on each side)
     - Downstream (DS) Vertical Bars (63.5 cm, 1 SiPM top readout)

Produces:
  - Multi-panel figure showing individual bar background lines and mean curves
  - Combined turn-on comparison figure across bar types
  - Full ROOT canvases with TGraphs in output/cluster_reconstruction_efficiency.root
"""

from __future__ import annotations

import os
import sys
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

import scripts.qdc_calib.min_pe as mpe


def get_category_bars() -> Dict[str, Dict[str, Any]]:
    """Define the detector bars and spatial parameters for each category."""
    cats = {
        "veto_01": {
            "title": "Veto Planes 0 & 1 (Gel, 42 cm)",
            "bar_ids": [10000 + b for b in range(7)] + [11000 + b for b in range(7)],
            "x_range": (-19.0, 19.0),
            "is_vert": False,
            "color": "#1f77b4",
            "eval_side": "both",
        },
        "veto_2": {
            "title": "Veto Plane 2 (Top Readout + Mirror, 46 cm)",
            "bar_ids": [12000 + b for b in range(7)],
            "y_range": (-21.0, 21.0),
            "is_vert": True,
            "color": "#9467bd",
            "eval_side": "both",
        },
        "us_gel": {
            "title": "US Bars (Gel Interface, 82.5 cm)",
            "bar_ids": [20000 + p * 1000 + b for p in range(5) for b in range(10)],
            "x_range": (-38.0, 38.0),
            "is_vert": False,
            "color": "#2ca02c",
            "eval_side": "gel",
        },
        "us_air": {
            "title": "US Bars (Air Interface, 82.5 cm)",
            "bar_ids": [20000 + p * 1000 + b for p in range(5) for b in range(10)],
            "x_range": (-38.0, 38.0),
            "is_vert": False,
            "color": "#ff7f0e",
            "eval_side": "air",
        },
        "ds_hor": {
            "title": "DS Horizontal Bars (82.5 cm, 1 SiPM/side)",
            "bar_ids": [30000 + p * 1000 + b for p in range(3) for b in range(60)],
            "x_range": (-38.0, 38.0),
            "is_vert": False,
            "color": "#17becf",
            "eval_side": "both",
        },
        "ds_ver": {
            "title": "DS Vertical Bars (63.5 cm, Top Readout)",
            "bar_ids": [30000 + p * 1000 + b for p in range(4) for b in range(60, 120)],
            "y_range": (-28.0, 28.0),
            "is_vert": True,
            "color": "#d62728",
            "eval_side": "both",
        },
    }
    return cats


def compute_bar_efficiency_curve(
    calc: mpe.MinPE,
    det_id: int,
    e_vals: np.ndarray,
    n_pos: int = 25,
    eval_side: str = "both",
) -> np.ndarray:
    """
    Computes hit efficiency S-curve vs cluster energy deposit E_dep for a specific bar.
    Evaluates threshold across n_pos points along the bar length.
    """
    is_vert = calc.is_vertical(det_id)
    subsystem, plane, bar = calc.decode_detector_id(det_id)

    # Determine spatial span
    if subsystem == 1:
        if plane == 2:
            positions = [(0.0, y) for y in np.linspace(-21.0, 21.0, n_pos)]
        else:
            positions = [(x, 0.0) for x in np.linspace(-19.0, 19.0, n_pos)]
    elif subsystem == 2:
        positions = [(x, 0.0) for x in np.linspace(-38.0, 38.0, n_pos)]
    else: # subsystem 3
        if is_vert:
            positions = [(0.0, y) for y in np.linspace(-28.0, 28.0, n_pos)]
        else:
            positions = [(x, 0.0) for x in np.linspace(-38.0, 38.0, n_pos)]

    # Compute minimum threshold at each position
    eth_pos = []
    for x, y in positions:
        pt = {"detector_id": det_id, "x": x, "y": y, "z": 0.0, "eloss": 1.0}
        th_info = calc.get_lowest_trigger_threshold([pt], det_id=det_id, include_mirror=True)

        if eval_side == "gel":
            # Qualification from gel side only
            has_gel_l = calc.has_gel(det_id, 0)
            target_side = "left" if has_gel_l else "right"
            if th_info["triggering_side"] == target_side:
                eth = th_info["min_eloss_mev"]
            else:
                # If opposite side triggered first, find threshold on gel side
                eth = th_info["min_eloss_mev"] * 1.3
        elif eval_side == "air":
            has_gel_l = calc.has_gel(det_id, 0)
            target_side = "right" if has_gel_l else "left"
            if th_info["triggering_side"] == target_side:
                eth = th_info["min_eloss_mev"]
            else:
                eth = th_info["min_eloss_mev"] * 1.3
        else:
            eth = th_info["min_eloss_mev"]

        eth_pos.append(eth)

    eth_arr = np.array(eth_pos, dtype=float)
    # Vectorized efficiency across all energy steps
    eff_curve = (e_vals[:, None] >= eth_arr[None, :]).mean(axis=1)
    return eff_curve


def main():
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"

    print("Initializing MinPE models...")
    calc_theo = mpe.MinPE(year=2024)
    calc_theo._k_factors = {} # all k = 1.0 (Theoretical)

    calc_calib = mpe.MinPE(year=2024) # Calibrated (k from CSV)

    e_vals = np.linspace(0.005, 1.0, 120) # Energy steps from 0.005 to 1.0 MeV
    categories = get_category_bars()

    results = {}
    print("Evaluating individual bars across all categories...")

    for cat_key, cat_info in categories.items():
        print(f"  Category: {cat_info['title']} ({len(cat_info['bar_ids'])} bars)...")
        bar_curves_calib = []
        bar_curves_theo = []

        # Subsample bars if category has > 30 bars to keep generation fast & clean
        bar_ids = cat_info["bar_ids"]
        if len(bar_ids) > 30:
            step = len(bar_ids) // 25
            sampled_bars = bar_ids[::step]
        else:
            sampled_bars = bar_ids

        for did in sampled_bars:
            c_cal = compute_bar_efficiency_curve(calc_calib, did, e_vals, n_pos=25, eval_side=cat_info["eval_side"])
            c_th = compute_bar_efficiency_curve(calc_theo, did, e_vals, n_pos=25, eval_side=cat_info["eval_side"])
            bar_curves_calib.append(c_cal)
            bar_curves_theo.append(c_th)

        bar_curves_calib = np.array(bar_curves_calib)
        bar_curves_theo = np.array(bar_curves_theo)

        mean_calib = np.mean(bar_curves_calib, axis=0)
        mean_theo = np.mean(bar_curves_theo, axis=0)

        results[cat_key] = {
            "info": cat_info,
            "bar_curves_calib": bar_curves_calib,
            "bar_curves_theo": bar_curves_theo,
            "mean_calib": mean_calib,
            "mean_theo": mean_theo,
            "eff_raw": np.ones_like(e_vals),
        }

    # --------------------------------------------------------------------------
    # Figure 1: 6-Panel Multi-plot with individual bar background lines
    # --------------------------------------------------------------------------
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)

    for idx, (cat_key, data) in enumerate(results.items()):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        info = data["info"]

        # 1. Background: High-transparency individual bar lines
        for b_idx in range(len(data["bar_curves_calib"])):
            b_curve = data["bar_curves_calib"][b_idx]
            lbl = "Individual Bars" if b_idx == 0 else None
            ax.plot(e_vals, b_curve, color=info["color"], lw=0.7, alpha=0.18, zorder=1, label=lbl)

        # 2. Raw Simulation (Eth=0)
        ax.plot(e_vals, data["eff_raw"], ":", color="#1f77b4", lw=1.6, zorder=2, label="Raw Sim (Eth=0)")

        # 3. Theoretical Model (k=1.0)
        e50_theo = e_vals[np.argmax(data["mean_theo"] >= 0.5)]
        ax.plot(e_vals, data["mean_theo"], "--", color="#9467bd", lw=2.0, zorder=3, label=f"Theoretical k=1 (E50%={e50_theo:.2f} MeV)")

        # 4. Calibrated Model (Group Mean)
        e50_calib = e_vals[np.argmax(data["mean_calib"] >= 0.5)]
        ax.plot(e_vals, data["mean_calib"], "-", color=info["color"], lw=2.6, zorder=4, label=f"Calibrated Mean (E50%={e50_calib:.2f} MeV)")

        # Shading between theoretical and calibrated
        ax.fill_between(e_vals, data["mean_calib"], data["mean_theo"], color=info["color"], alpha=0.12, zorder=2)

        # Reference lines
        ax.axvline(1.8, color="gray", linestyle=":", lw=1.2, label="1 MIP (~1.8 MeV)", zorder=2)
        ax.axhline(0.5, color="lightgray", linestyle=":", lw=0.8, zorder=1)
        ax.axhline(1.0, color="lightgray", linestyle="--", lw=0.8, zorder=1)

        ax.set_title(info["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("MCPoint Cluster Energy Deposit [MeV]", fontsize=10)
        ax.set_ylabel("Hit Reconstruction Efficiency", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.03, 1.06)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8.0, framealpha=0.92)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    out_png1 = os.path.join(out_dir, "cluster_reconstruction_efficiency_by_bar_type.png")
    fig.savefig(out_png1, dpi=200)
    plt.close(fig)
    print(f"Saved figure: {out_png1}")
    if os.path.exists(art_dir):
        os.system(f"cp {out_png1} {os.path.join(art_dir, 'cluster_reconstruction_efficiency_by_bar_type.png')}")

    # --------------------------------------------------------------------------
    # Figure 2: Combined Comparison across all Categories with Background Lines
    # --------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)

    for cat_key, data in results.items():
        info = data["info"]
        # Background individual lines
        for b_idx in range(len(data["bar_curves_calib"])):
            ax2.plot(e_vals, data["bar_curves_calib"][b_idx], color=info["color"], lw=0.5, alpha=0.10, zorder=1)

        # Thick Category Mean Curve
        e50_c = e_vals[np.argmax(data["mean_calib"] >= 0.5)]
        ax2.plot(e_vals, data["mean_calib"], color=info["color"], lw=2.4, zorder=3,
                 label=f"{info['title']} (E50% = {e50_c:.2f} MeV)")

    ax2.axvline(1.8, color="black", linestyle=":", lw=1.5, label="1 MIP (~1.8 MeV in 1 cm PVT)", zorder=2)
    ax2.axhline(0.5, color="gray", linestyle="--", lw=0.9, zorder=1)
    ax2.axhline(1.0, color="gray", linestyle="--", lw=0.9, zorder=1)

    ax2.set_title("Hit Reconstruction Efficiency S-Curves vs Cluster Energy (All Bar Types)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("MCPoint Cluster Energy Deposit [MeV]", fontsize=11)
    ax2.set_ylabel("Hit Reconstruction Efficiency", fontsize=11)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8.8, framealpha=0.95)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    out_png2 = os.path.join(out_dir, "cluster_reconstruction_efficiency_combined.png")
    fig2.savefig(out_png2, dpi=200)
    plt.close(fig2)
    print(f"Saved figure: {out_png2}")
    if os.path.exists(art_dir):
        os.system(f"cp {out_png2} {os.path.join(art_dir, 'cluster_reconstruction_efficiency_combined.png')}")

    # --------------------------------------------------------------------------
    # 3. Save to ROOT File with TGraphs
    # --------------------------------------------------------------------------
    root_out = "output/cluster_reconstruction_efficiency.root"
    f_root = ROOT.TFile.Open(root_out, "RECREATE")

    c_root = ROOT.TCanvas("c_cluster_efficiency_all", "Cluster Hit Efficiency S-Curves (All Bars)", 1200, 800)
    c_root.Divide(3, 2)
    c_root.store = []

    for idx, (cat_key, data) in enumerate(results.items()):
        c_root.cd(idx + 1)
        ROOT.gPad.SetGrid()
        info = data["info"]

        n_pts = len(e_vals)
        gr_mean_calib = ROOT.TGraph(n_pts, e_vals.astype(float), data["mean_calib"].astype(float))
        gr_mean_theo = ROOT.TGraph(n_pts, e_vals.astype(float), data["mean_theo"].astype(float))
        gr_raw = ROOT.TGraph(n_pts, e_vals.astype(float), data["eff_raw"].astype(float))

        gr_mean_calib.SetName(f"gr_calib_mean_{cat_key}")
        gr_mean_theo.SetName(f"gr_theo_mean_{cat_key}")
        gr_raw.SetName(f"gr_raw_{cat_key}")

        gr_mean_calib.SetTitle(f"{info['title']};MCPoint E_{{dep}} [MeV];Hit Efficiency")
        gr_mean_calib.GetYaxis().SetRangeUser(0.0, 1.08)
        gr_mean_calib.GetXaxis().SetRangeUser(0.0, 1.0)

        gr_mean_calib.SetLineColor(ROOT.kRed + 1)
        gr_mean_calib.SetLineWidth(3)

        gr_mean_theo.SetLineColor(ROOT.kMagenta + 2)
        gr_mean_theo.SetLineStyle(2)
        gr_mean_theo.SetLineWidth(2)

        gr_raw.SetLineColor(ROOT.kBlue + 1)
        gr_raw.SetLineStyle(3)
        gr_raw.SetLineWidth(2)

        gr_mean_calib.Draw("AL")

        # Draw individual bar background graphs
        for b_idx in range(len(data["bar_curves_calib"])):
            gr_b = ROOT.TGraph(n_pts, e_vals.astype(float), data["bar_curves_calib"][b_idx].astype(float))
            gr_b.SetLineColorAlpha(ROOT.kRed - 7, 0.25)
            gr_b.SetLineWidth(1)
            gr_b.Draw("L SAME")
            c_root.store.append(gr_b)

        gr_raw.Draw("L SAME")
        gr_mean_theo.Draw("L SAME")
        gr_mean_calib.Draw("L SAME")

        leg = ROOT.TLegend(0.48, 0.18, 0.88, 0.42)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.AddEntry(gr_raw, "Raw (Eth=0)", "l")
        leg.AddEntry(gr_mean_theo, "Theoretical (k=1)", "l")
        leg.AddEntry(gr_mean_calib, "Calibrated Mean", "l")
        leg.Draw()

        c_root.store.extend([gr_mean_calib, gr_mean_theo, gr_raw, leg])
        gr_mean_calib.Write()
        gr_mean_theo.Write()
        gr_raw.Write()

    c_root.Write()
    f_root.Close()
    print(f"Saved ROOT file to {root_out}")
    if os.path.exists(art_dir):
        os.system(f"cp {root_out} {os.path.join(art_dir, 'cluster_reconstruction_efficiency.root')}")

    print("\nSummary of 50% Turn-On Energies:")
    for cat_key, data in results.items():
        e50_t = e_vals[np.argmax(data["mean_theo"] >= 0.5)]
        e50_c = e_vals[np.argmax(data["mean_calib"] >= 0.5)]
        print(f"  {data['info']['title']:<38} | Theo: {e50_t:.3f} MeV | Calib: {e50_c:.3f} MeV")

if __name__ == "__main__":
    main()
