#!/usr/bin/env python3
"""
Plot SiPM Photoelectron Reduction Factor (k) Variations
-------------------------------------------------------
Plots how the semi-empirical photoelectron reduction parameter k (0 <= k <= 1)
varies between SiPMs for each distinct scintillator bar setup:
  1. Veto Planes 0 & 1 (Double-ended, Gel, 8 SiPMs/side)
  2. Veto Plane 2 (Single-ended, Top Readout + Mirror, 8 SiPMs)
  3. Upstream (US) with Optical Gel (Single-ended, 6 large SiPMs)
  4. Upstream (US) without Gel / Air (Single-ended, 6 large SiPMs)
  5. Downstream (DS) Horizontal (Double-ended, Air, Left vs Right SiPM)
  6. Overall Comparative Distribution across all 6 setups

Outputs:
  plots/sipm_reduction_factor_by_bar_type.png
  plots/sipm_reduction_factor_profiles.png
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd

# Check ddfUtils styling
try:
    import ddfUtils.mpl_styling as dms
    HAS_DDFUTILS = True
except ImportError:
    HAS_DDFUTILS = False

CSV_PATH = "config/sipm_photoelectron_reduction_factors_detailed.csv"
OUT_DIR = "plots"
ARTIFACT_DIR = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"


def load_and_classify_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    def get_cat(r):
        sub = int(r["subsystem"])
        pl = int(r["plane"])
        ch = int(r["sipm_number"])
        b = int(r["bar"])
        if sub == 1:
            return "Veto 0&1" if pl < 2 else "Veto 2"
        elif sub == 2:
            st = pl
            side = "left" if ch < 8 else "right"
            if st in (0, 2):
                return "US Gel" if side == "right" else "US Air"
            else:
                return "US Gel" if side == "left" else "US Air"
        elif sub == 3:
            return "DS Horizontal" if b < 60 else "DS Vertical"
        return "Other"

    df["category"] = df.apply(get_cat, axis=1)
    df["rel_sipm"] = df["sipm_number"] % 8
    df["side"] = df.apply(
        lambda r: "Left" if r["sipm_number"] < 8 else "Right" if r["subsystem"] in (1, 2)
        else ("Left" if r["sipm_number"] == 0 else "Right") if r["subsystem"] == 3 and r["bar"] < 60
        else "Top",
        axis=1,
    )
    return df


def plot_sipm_profiles(df: pd.DataFrame, out_path: str) -> None:
    """Generate 6-panel figure showing k variation across SiPM positions for each bar setup."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.1

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)

    # -------------------------------------------------------------------------
    # Panel 1: Veto Planes 0 & 1 (Double-ended, 8 Left + 8 Right)
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    sub_df = df[df["category"] == "Veto 0&1"]
    left_df = sub_df[sub_df["side"] == "Left"]
    right_df = sub_df[sub_df["side"] == "Right"]

    l_means = left_df.groupby("rel_sipm")["k"].mean()
    l_stds = left_df.groupby("rel_sipm")["k"].std()
    r_means = right_df.groupby("rel_sipm")["k"].mean()
    r_stds = right_df.groupby("rel_sipm")["k"].std()

    # Jitter scatter
    rng = np.random.default_rng(42)
    ax.scatter(
        left_df["rel_sipm"] - 0.12 + rng.uniform(-0.04, 0.04, len(left_df)),
        left_df["k"], color="#1f77b4", alpha=0.35, s=20, label="Left SiPMs (Data)",
    )
    ax.scatter(
        right_df["rel_sipm"] + 0.12 + rng.uniform(-0.04, 0.04, len(right_df)),
        right_df["k"], color="#d62728", alpha=0.35, s=20, label="Right SiPMs (Data)",
    )

    ax.errorbar(
        l_means.index - 0.12, l_means.values, yerr=l_stds.values,
        fmt="o-", color="#0d47a1", lw=2.0, ms=6, capsize=3, label=r"Left Mean $\pm 1\sigma$",
    )
    ax.errorbar(
        r_means.index + 0.12, r_means.values, yerr=r_stds.values,
        fmt="s-", color="#b71c1c", lw=2.0, ms=6, capsize=3, label=r"Right Mean $\pm 1\sigma$",
    )

    ax.set_title("Veto Planes 0 & 1 ($42\\times 6\\times 1$ cm, Gel)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Relative SiPM Index along Bar Width (0 to 7)", fontsize=11)
    ax.set_ylabel("Photoelectron Reduction Factor $k$", fontsize=11)
    ax.set_xticks(range(8))
    ax.set_ylim(0.35, 1.05)
    ax.axhline(1.0, color="gray", linestyle=":", lw=1.2)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # -------------------------------------------------------------------------
    # Panel 2: Veto Plane 2 (Post-2024, Top Readout + Mirror)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    v2_df = df[df["category"] == "Veto 2"]
    if len(v2_df) > 0:
        v2_means = v2_df.groupby("rel_sipm")["k"].mean()
        v2_stds = v2_df.groupby("rel_sipm")["k"].std()
        ax.scatter(
            v2_df["rel_sipm"] + rng.uniform(-0.06, 0.06, len(v2_df)),
            v2_df["k"], color="#9c27b0", alpha=0.45, s=28, label="Veto 2 Top SiPMs",
        )
        ax.errorbar(
            v2_means.index, v2_means.values, yerr=v2_stds.values,
            fmt="D-", color="#4a148c", lw=2.2, ms=6, capsize=3.5, label=r"Top Mean $\pm 1\sigma$",
        )
        ax.set_title("Veto Plane 2 ($46\\times 6\\times 1$ cm, Top Readout + Mirror)", fontweight="bold", fontsize=12)
        ax.set_xlabel("Top SiPM Index along Bar Width (0 to 7)", fontsize=11)
        ax.set_ylabel("Photoelectron Reduction Factor $k$", fontsize=11)
        ax.set_xticks(range(8))
        ax.set_ylim(0.35, 1.05)
        ax.axhline(1.0, color="gray", linestyle=":", lw=1.2)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "Veto Plane 2 (2024 Only)", ha="center", va="center")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # -------------------------------------------------------------------------
    # Panel 3: Upstream (US) with Optical Gel
    # -------------------------------------------------------------------------
    ax = axes[0, 2]
    us_gel_df = df[df["category"] == "US Gel"]
    ug_means = us_gel_df.groupby("rel_sipm")["k"].mean()
    ug_stds = us_gel_df.groupby("rel_sipm")["k"].std()

    ax.scatter(
        us_gel_df["rel_sipm"] + rng.uniform(-0.06, 0.06, len(us_gel_df)),
        us_gel_df["k"], color="#2ca02c", alpha=0.35, s=22, label="US Gel Large SiPMs (Data)",
    )
    ax.errorbar(
        ug_means.index, ug_means.values, yerr=ug_stds.values,
        fmt="o-", color="#1b5e20", lw=2.2, ms=6.5, capsize=3.5, label=r"Gel Side Mean $\pm 1\sigma$",
    )
    ax.set_title("US Bars with Optical Gel ($82.5\\times 6\\times 1$ cm)", fontweight="bold", fontsize=12)
    ax.set_xlabel("SiPM Channel on Bar End (2 & 5 small/uncalibrated)", fontsize=11)
    ax.set_ylabel("Photoelectron Reduction Factor $k$", fontsize=11)
    ax.set_xticks([0, 1, 3, 4, 6, 7])
    ax.set_xticklabels(["0", "1", "3", "4", "6", "7"])
    ax.set_ylim(0.05, 1.05)
    ax.axhline(1.0, color="gray", linestyle=":", lw=1.2)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # -------------------------------------------------------------------------
    # Panel 4: Upstream (US) without Gel (Dry Air Interface)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    us_air_df = df[df["category"] == "US Air"]
    ua_means = us_air_df.groupby("rel_sipm")["k"].mean()
    ua_stds = us_air_df.groupby("rel_sipm")["k"].std()

    ax.scatter(
        us_air_df["rel_sipm"] + rng.uniform(-0.06, 0.06, len(us_air_df)),
        us_air_df["k"], color="#ff7f0e", alpha=0.35, s=22, label="US Air Large SiPMs (Data)",
    )
    ax.errorbar(
        ua_means.index, ua_means.values, yerr=ua_stds.values,
        fmt="s-", color="#e65100", lw=2.2, ms=6.5, capsize=3.5, label=r"Air Side Mean $\pm 1\sigma$",
    )
    ax.set_title("US Bars without Gel / Air Interface ($82.5\\times 6\\times 1$ cm)", fontweight="bold", fontsize=12)
    ax.set_xlabel("SiPM Channel on Bar End (2 & 5 small/uncalibrated)", fontsize=11)
    ax.set_ylabel("Photoelectron Reduction Factor $k$", fontsize=11)
    ax.set_xticks([0, 1, 3, 4, 6, 7])
    ax.set_xticklabels(["0", "1", "3", "4", "6", "7"])
    ax.set_ylim(0.20, 1.05)
    ax.axhline(1.0, color="gray", linestyle=":", lw=1.2)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # -------------------------------------------------------------------------
    # Panel 5: Downstream (DS) Horizontal (Left vs Right SiPM)
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    dsh_df = df[df["category"] == "DS Horizontal"]
    dsh_left = dsh_df[dsh_df["sipm_number"] == 0]["k"]
    dsh_right = dsh_df[dsh_df["sipm_number"] == 1]["k"]

    box_data = [dsh_left.values, dsh_right.values]
    bp = ax.boxplot(
        box_data, positions=[1, 2], widths=0.45, patch_artist=True,
        boxprops=dict(facecolor="#e1f5fe", color="#0288d1", lw=1.5),
        medianprops=dict(color="#d32f2f", lw=2.0),
        whiskerprops=dict(color="#0288d1", lw=1.2),
        capprops=dict(color="#0288d1", lw=1.2),
        flierprops=dict(marker="o", color="#0288d1", alpha=0.4, ms=4),
    )
    # Overlay jitter scatter
    ax.scatter(np.full(len(dsh_left), 1) + rng.uniform(-0.1, 0.1, len(dsh_left)), dsh_left, color="#0288d1", alpha=0.35, s=20)
    ax.scatter(np.full(len(dsh_right), 2) + rng.uniform(-0.1, 0.1, len(dsh_right)), dsh_right, color="#e65100", alpha=0.35, s=20)

    ax.set_title("DS Horizontal ($82.5\\times 1\\times 1$ cm, 1 SiPM/side)", fontweight="bold", fontsize=12)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([
        f"Left SiPM (0)\nMean={dsh_left.mean():.3f} $\\pm$ {dsh_left.std():.3f}",
        f"Right SiPM (1)\nMean={dsh_right.mean():.3f} $\\pm$ {dsh_right.std():.3f}",
    ])
    ax.set_ylabel("Photoelectron Reduction Factor $k$", fontsize=11)
    ax.set_ylim(0.0, 1.08)
    ax.axhline(1.0, color="gray", linestyle=":", lw=1.2)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # -------------------------------------------------------------------------
    # Panel 6: Comparative Distributions Across All Scintillator Setups
    # -------------------------------------------------------------------------
    ax = axes[1, 2]
    all_cats = ["Veto 0&1", "Veto 2", "US Gel", "US Air", "DS Horizontal", "DS Vertical"]
    all_colors = ["#1f77b4", "#9c27b0", "#2ca02c", "#ff7f0e", "#00acc1", "#3949ab"]
    active_pairs = [(c, col) for c, col in zip(all_cats, all_colors) if len(df[df["category"] == c]) > 0]
    cats = [c for c, _ in active_pairs]
    cat_colors = [col for _, col in active_pairs]
    plot_data = [df[df["category"] == c]["k"].values for c in cats]

    parts = ax.violinplot(plot_data, positions=range(len(cats)), showmeans=True, showmedians=False, widths=0.7)
    for idx, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(cat_colors[idx])
        pc.set_edgecolor(cat_colors[idx])
        pc.set_alpha(0.65)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(2.0)
    parts["cbars"].set_color("#666666")
    parts["cmins"].set_color("#666666")
    parts["cmaxes"].set_color("#666666")

    # Annotate means
    for idx, c in enumerate(cats):
        mean_val = df[df["category"] == c]["k"].mean()
        ax.text(idx, 0.03, f"{mean_val:.2f}", ha="center", fontsize=9.5, fontweight="bold", color=cat_colors[idx])

    ax.set_title("Comparison Across Bar Configurations", fontweight="bold", fontsize=12)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.set_ylabel("Photoelectron Reduction Factor $k$", fontsize=11)
    ax.set_ylim(0.0, 1.08)
    ax.axhline(1.0, color="gray", linestyle=":", lw=1.2)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle(
        "SiPM Photoelectron Reduction Factor ($k = N_{\\rm pe}^{\\rm data} / N_{\\rm pe}^{\\rm th}$) Across Bar Types & Channels",
        fontsize=15, fontweight="bold",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Success] Saved figure: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_and_classify_data(CSV_PATH)

    out_png = os.path.join(OUT_DIR, "sipm_reduction_factor_by_bar_type.png")
    plot_sipm_profiles(df, out_png)

    if os.path.exists(ARTIFACT_DIR):
        os.system(f"cp -f {out_png} {ARTIFACT_DIR}/sipm_reduction_factor_by_bar_type.png")
        print(f"[Success] Copied to artifact directory: {ARTIFACT_DIR}/sipm_reduction_factor_by_bar_type.png")


if __name__ == "__main__":
    main()
