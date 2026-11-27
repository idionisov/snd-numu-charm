#!/usr/bin/env python3
"""
scripts/qdc_calib/plot_tb_sipm_reduction_factors.py
--------------------------------------------------
Visualizes the re-extracted testbeam SiPM photoelectron reduction factors (k)
across all instrumented channels, showing:
1. Channel-by-channel variation along US bar faces (Left vs Right)
2. Gel vs Air coupling distributions in Upstream stations
3. Downstream Station 1 (DS1-H and DS1-V) distributions
4. Full summary violin / box comparison

Outputs:
- plots/tb_sipm_reduction_factors_variation.png
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from min_pe import MinPE
calc = MinPE(is_testbeam=True)

def main():
    csv_file = "config/sipm_photoelectron_reduction_factors_testbeam.csv"
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"

    df = pd.read_csv(csv_file)
    # Filter active large SiPMs (k > 0)
    df = df[df["k"] > 0.0].copy()

    df["subsystem"] = df["detector_id"] // 10000
    df["plane"] = (df["detector_id"] // 1000) % 10
    df["bar"] = df["detector_id"] % 1000

    def classify_tb_channel(r):
        sub = int(r["subsystem"])
        did = int(r["detector_id"])
        ch = int(r["sipm_number"])
        if sub == 2:
            return "US Gel" if calc.has_gel(did, ch) else "US Air"
        elif sub == 3:
            return "DS1-V" if int(r["bar"]) >= 60 else "DS1-H"
        return "Other"

    df["category"] = df.apply(classify_tb_channel, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), constrained_layout=True)

    # -------------------------------------------------------------------------
    # Panel 1: US Gel vs Air Distribution
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    bins = np.linspace(0.2, 1.05, 35)
    us_gel = df[df["category"] == "US Gel"]["k"].values
    us_air = df[df["category"] == "US Air"]["k"].values

    ax.hist(us_gel, bins=bins, density=True, histtype="stepfilled", color="#2ca02c", alpha=0.55, label=f"US Gel (N={len(us_gel)}, Mean={np.mean(us_gel):.3f} $\\pm$ {np.std(us_gel):.3f})")
    ax.hist(us_gel, bins=bins, density=True, histtype="step", color="#2ca02c", lw=2.0)
    ax.hist(us_air, bins=bins, density=True, histtype="stepfilled", color="#ff7f0e", alpha=0.45, label=f"US Air (N={len(us_air)}, Mean={np.mean(us_air):.3f} $\\pm$ {np.std(us_air):.3f})")
    ax.hist(us_air, bins=bins, density=True, histtype="step", color="#ff7f0e", lw=2.0)
    ax.set_title("Upstream (US): Optical Gel vs. Air Coupling", fontsize=12, fontweight="bold")
    ax.set_xlabel("Photoelectron Reduction Factor $k$", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9.5, loc="upper left")

    # -------------------------------------------------------------------------
    # Panel 2: Downstream (DS1-H vs DS1-V)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    ds_h = df[df["category"] == "DS1-H"]["k"].values
    ds_v = df[df["category"] == "DS1-V"]["k"].values

    ax.hist(ds_h, bins=bins, density=True, histtype="stepfilled", color="#00acc1", alpha=0.55, label=f"DS1-H (N={len(ds_h)}, Mean={np.mean(ds_h):.3f} $\\pm$ {np.std(ds_h):.3f})")
    ax.hist(ds_h, bins=bins, density=True, histtype="step", color="#00acc1", lw=2.0)
    ax.hist(ds_v, bins=bins, density=True, histtype="stepfilled", color="#3949ab", alpha=0.45, label=f"DS1-V (N={len(ds_v)}, Mean={np.mean(ds_v):.3f} $\\pm$ {np.std(ds_v):.3f})")
    ax.hist(ds_v, bins=bins, density=True, histtype="step", color="#3949ab", lw=2.0)
    ax.set_title("Downstream (DS1): Horizontal vs. Vertical Bars", fontsize=12, fontweight="bold")
    ax.set_xlabel("Photoelectron Reduction Factor $k$", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9.5, loc="upper left")

    # -------------------------------------------------------------------------
    # Panel 3: Channel-by-Channel Profile across 8 SiPM positions (US)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    # Channels 0..7 on a side (0, 1, 3, 4, 6, 7 are large SiPMs)
    large_channels = [0, 1, 3, 4, 6, 7]
    pos_data = [df[df["subsystem"] == 2][(df["sipm_number"] % 8) == ch]["k"].values for ch in large_channels]

    bplot = ax.boxplot(pos_data, positions=range(len(large_channels)), patch_artist=True, widths=0.55,
                       boxprops=dict(facecolor="#90caf9", color="#1565c0", alpha=0.7),
                       medianprops=dict(color="black", lw=2.0))
    ax.set_title("Channel Position Uniformity Across US SiPM Array", fontsize=12, fontweight="bold")
    ax.set_xlabel("SiPM Channel Position (Mod 8)", fontsize=11)
    ax.set_ylabel("Photoelectron Reduction Factor $k$", fontsize=11)
    ax.set_xticks(range(len(large_channels)))
    ax.set_xticklabels([f"Ch {ch}" for ch in large_channels])
    ax.set_ylim(0.2, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    # -------------------------------------------------------------------------
    # Panel 4: Violin Plot Across All 4 Testbeam Categories
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    categories = ["US Gel", "US Air", "DS1-H", "DS1-V"]
    colors = ["#2ca02c", "#ff7f0e", "#00acc1", "#3949ab"]
    v_data = [df[df["category"] == c]["k"].values for c in categories]

    parts = ax.violinplot(v_data, positions=range(len(categories)), showmeans=True, showmedians=False, widths=0.65)
    for idx, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[idx])
        pc.set_edgecolor(colors[idx])
        pc.set_alpha(0.65)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(2.0)
    parts["cbars"].set_color("#555555")
    parts["cmins"].set_color("#555555")
    parts["cmaxes"].set_color("#555555")

    for idx, c in enumerate(categories):
        m_val = np.mean(df[df["category"] == c]["k"].values)
        ax.text(idx, 0.22, f"Mean:\n{m_val:.3f}", ha="center", fontsize=9.5, fontweight="bold", color=colors[idx])

    ax.set_title("Testbeam Reduction Factor Comparison Across Subsystems", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Photoelectron Reduction Factor $k$", fontsize=11)
    ax.set_ylim(0.18, 1.08)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    fig.suptitle("Testbeam H8 SiPM Photoelectron Reduction Factor ($k = N_{\\rm pe}^{\\rm data} / N_{\\rm pe}^{\\rm th}$) Variations", fontsize=14, fontweight="bold")

    out_file = os.path.join(plot_dir, "tb_sipm_reduction_factors_variation.png")
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    os.system(f"cp {out_file} {os.path.join(art_dir, 'tb_sipm_reduction_factors_variation.png')}")
    plt.close()
    print(f"[SUCCESS] Generated: {out_file}")

if __name__ == "__main__":
    main()
