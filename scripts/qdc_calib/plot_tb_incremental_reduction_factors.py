#!/usr/bin/env python3
"""
scripts/qdc_calib/plot_tb_incremental_reduction_factors.py
----------------------------------------------------------
Visualizes the incremental addition of each physical reduction factor
in 180 GeV pi+ testbeam simulation compared to testbeam Data (Run 100671):

Stages:
  1. Raw: Pure scintillation yield * PDE (10000 photons/MeV * 0.50)
  2. + f_area: Geometric bar-end area coverage (0.36 / 6.0 cm^2)
  3. + f_cone: Acceptance cone / Total Internal Reflection trapping
  4. + S_att * f_Fresnel: Longitudinal attenuation & Fresnel interface
  5. + f_Birks: Generalized particle-dependent Birks quenching
  6. + f_edge: Lateral dead layer (delta_dead=1 mm) & edge loss
  7. + f_disp: Photon arrival time dispersion convolved with pulse shape
  8. + k (Final): Channel-specific SiPM reduction factor from MIP calibration

All incremental stages follow a continuous colormap (turbo / plasma).
Outputs:
  plots/tb_incremental_reduction_factors.png
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

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import scripts.qdc_calib.min_pe as mpe


def main():
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"

    calc = mpe.MinPE(year=2023, is_testbeam=True, include_edge_effect=True, include_dispersion=True)
    th_pe = calc.get_sipm_threshold_pe()
    print(f"Single-SiPM discriminator threshold: {th_pe:.2f} PE")

    # -------------------------------------------------------------------------
    # 1. Read Testbeam Data (Run 100671)
    # -------------------------------------------------------------------------
    data_file = "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100671/sndsw_raw-0000.root"
    print(f"[1/2] Reading Data from {data_file}...")
    f_d = ROOT.TFile.Open(data_file)
    t_d = f_d.Get("rawConv")
    t_d.SetBranchStatus("*", 0)
    t_d.SetBranchStatus("Digi_ScifiHits*", 1)
    t_d.SetBranchStatus("Digi_MuFilterHits*", 1)

    d_us0_mult = []
    d_us0_bars = np.zeros(10, dtype=float)
    n_showers_d = 0

    for i in range(min(35000, t_d.GetEntries())):
        t_d.GetEntry(i)
        if len(t_d.Digi_ScifiHits) < 100:
            continue
        n_showers_d += 1
        fired_b = set()
        for mh in t_d.Digi_MuFilterHits:
            if not mh.isValid():
                continue
            did = mh.GetDetectorID()
            if did // 1000 == 20:  # US Plane 0
                bar = did % 1000
                times = [s.second for s in mh.GetAllTimes() if s.second != -999.0]
                if times and any(abs(t) <= 15.0 for t in times):
                    fired_b.add(bar)
        d_us0_mult.append(len(fired_b))
        for b in fired_b:
            if 0 <= b < 10:
                d_us0_bars[b] += 1

    f_d.Close()
    d_prob_bars = (d_us0_bars / n_showers_d) * 100.0
    mean_d_mult = np.mean(d_us0_mult)
    print(f"      Data: Analyzed {n_showers_d} showers. US0 Mean Fired Bars = {mean_d_mult:.2f}")

    # -------------------------------------------------------------------------
    # 2. Read Geant4 Simulation and Evaluate 8 Stages
    # -------------------------------------------------------------------------
    mc_file = "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/180GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root"
    print(f"[2/2] Reading MC from {mc_file}...")
    f_m = ROOT.TFile.Open(mc_file)
    t_m = f_m.Get("cbmsim")
    t_m.SetBranchStatus("*", 0)
    t_m.SetBranchStatus("MuFilterPoint*", 1)

    stage_names = [
        "1. Raw Scintillation",
        "2. + $f_{\\mathrm{area}}$",
        "3. + $f_{\\mathrm{cone}}$",
        "4. + $S_{\\mathrm{att}} \\cdot f_{\\mathrm{Fresnel}}$",
        "5. + $f_{\\mathrm{Birks}}$ (Hadron)",
        "6. + $f_{\\mathrm{edge}}$ (Dead Skin)",
        "7. + $f_{\\mathrm{disp}}$ (Arrival)",
        "8. + $k$ (MIP-Calib, Final)",
    ]

    mc_us0_mult = [[] for _ in range(8)]
    mc_us0_bars = [np.zeros(10, dtype=float) for _ in range(8)]
    n_mc_events = min(1200, t_m.GetEntries())

    for iev in range(n_mc_events):
        t_m.GetEntry(iev)
        by_bar = calc.group_mcpoints_by_bar(t_m.MuFilterPoint)
        counts = [0] * 8
        fired_sets = [set() for _ in range(8)]

        for did, clusters in by_bar.items():
            if did // 1000 != 20:
                continue
            bar = did % 1000
            for pts in clusters:
                tot_e = sum(p.GetEnergyLoss() * 1000.0 for p in pts)
                if tot_e <= 0:
                    continue

                # Stage 1: Raw
                pe1 = 10000.0 * tot_e * 0.50

                # Stage 2: + f_area
                pe2 = pe1 * (0.36 / 6.0)

                # Stage 3: + f_cone
                f_c = calc.get_fcone(pts[0], 0)
                pe3 = pe2 * f_c

                # Stage 4: + S_att * f_Fresnel
                s_att_fres = sum((p.GetEnergyLoss() * 1000.0) * calc.get_Satt(p, 0) * calc.get_ffresnel(p, 0) for p in pts) / tot_e
                pe4 = pe3 * s_att_fres

                # Stage 5: + Birks
                e_birks = sum((p.GetEnergyLoss() * 1000.0) * calc.get_fbirks(p) * calc.get_Satt(p, 0) * calc.get_ffresnel(p, 0) for p in pts)
                pe5 = 10000.0 * e_birks * (0.36 / 6.0) * f_c * 0.50

                # Stage 6: + Edge
                e_edge = sum((p.GetEnergyLoss() * 1000.0) * calc.get_fbirks(p) * calc.get_Satt(p, 0) * calc.get_ffresnel(p, 0) * (calc.get_fcone_eff(p, 0) / f_c) for p in pts)
                pe6 = 10000.0 * e_edge * (0.36 / 6.0) * f_c * 0.50

                # Stage 7: + Disp
                f_d = calc.get_fdisp(pts, 0)
                pe7 = pe6 * f_d

                # Stage 8: + k (MIP calibration factor)
                k_val = calc.get_reduction_factor(did, 0)
                pe8 = pe7 * k_val

                pes = [pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8]
                for s_idx, p_val in enumerate(pes):
                    if p_val >= th_pe:
                        fired_sets[s_idx].add(bar)
                break

        for s_idx in range(8):
            mc_us0_mult[s_idx].append(len(fired_sets[s_idx]))
            for b in fired_sets[s_idx]:
                if 0 <= b < 10:
                    mc_us0_bars[s_idx][b] += 1

    f_m.Close()

    mc_prob_bars = [(bars / n_mc_events) * 100.0 for bars in mc_us0_bars]
    mc_means = [np.mean(m) for m in mc_us0_mult]
    for s_idx, name in enumerate(stage_names):
        print(f"      {name:30s}: Mean = {mc_means[s_idx]:.2f}")

    # -------------------------------------------------------------------------
    # 3. Create Publication-Quality Plot with Colormap
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])

    # Colormap setup: turbo or plasma spanning stages 1 to 8
    cmap = plt.cm.turbo
    stage_colors = [cmap(val) for val in np.linspace(0.08, 0.95, 8)]

    # -------------------------------------------------------------------------
    # Panel 1: US Plane 0 Transverse Profile (Bar 0 to 9) with Subpanel Ratio
    # -------------------------------------------------------------------------
    gs_left = gs[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax_top = fig.add_subplot(gs_left[0])
    ax_bot = fig.add_subplot(gs_left[1], sharex=ax_top)

    x_bars = np.arange(10)

    # Data as filled black background bars
    ax_top.bar(x_bars, d_prob_bars, width=0.75, color="#212121", alpha=0.35, label=f"Data ($|t|\\leq 15$ ns, Mean={mean_d_mult:.2f})")
    ax_top.step(np.append(x_bars - 0.375, 9.375), np.append(d_prob_bars, d_prob_bars[-1]), where="post", color="#111111", lw=2.2)

    # Plot incremental stages with colormap
    for s_idx in range(8):
        c = stage_colors[s_idx]
        lw = 2.4 if s_idx in (0, 4, 7) else 1.6
        ls = "-" if s_idx == 7 else ("--" if s_idx == 0 else "-.")
        ax_top.plot(x_bars, mc_prob_bars[s_idx], marker="o", markersize=4.5, lw=lw, ls=ls, color=c, label=f"{stage_names[s_idx]} ($\\mu$={mc_means[s_idx]:.2f})")

        # Ratio to Data in bottom subpanel
        ratio = np.where(d_prob_bars > 0.5, mc_prob_bars[s_idx] / d_prob_bars, np.nan)
        ax_bot.plot(x_bars, ratio, marker="o", markersize=3.5, lw=1.5, ls=ls, color=c)

    ax_top.set_ylabel("Firing Probability [%]", fontsize=11, fontweight="bold")
    ax_top.set_title("Transverse Profile in US Plane 0: Incremental Factor Evolution", fontsize=12, fontweight="bold")
    ax_top.grid(True, linestyle=":", alpha=0.5)
    ax_top.set_ylim(0, 115)
    ax_top.legend(fontsize=8.0, loc="upper right", framealpha=0.92, ncol=2)

    ax_bot.axhline(1.0, color="#111111", lw=1.5, linestyle="--")
    ax_bot.set_ylabel("MC / Data", fontsize=10, fontweight="bold")
    ax_bot.set_xlabel("US Plane 0 Bar Index (0=Bottom, 9=Top)", fontsize=11, fontweight="bold")
    ax_bot.set_ylim(0.2, 3.8)
    ax_bot.set_xticks(x_bars)
    ax_bot.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 2: Hit Multiplicity Distribution (Fired Bars in US Plane 0)
    # -------------------------------------------------------------------------
    ax_mult = fig.add_subplot(gs[0, 1])
    bins_mult = np.arange(-0.5, 11.5, 1.0)
    mult_centers = np.arange(11)

    # Data filled histogram
    ax_mult.hist(d_us0_mult, bins=bins_mult, density=True, histtype="stepfilled", color="#424242", alpha=0.35, label=f"Data ($\\mu$={mean_d_mult:.2f})")
    ax_mult.hist(d_us0_mult, bins=bins_mult, density=True, histtype="step", color="#111111", lw=2.2)

    for s_idx in range(8):
        c = stage_colors[s_idx]
        lw = 2.4 if s_idx in (0, 4, 7) else 1.5
        ls = "-" if s_idx == 7 else ("--" if s_idx == 0 else "-.")
        ax_mult.hist(mc_us0_mult[s_idx], bins=bins_mult, density=True, histtype="step", lw=lw, ls=ls, color=c, label=f"{stage_names[s_idx]} ($\\mu$={mc_means[s_idx]:.2f})")

    ax_mult.set_title("US Plane 0 Multiplicity Distribution Evolution", fontsize=12, fontweight="bold")
    ax_mult.set_xlabel("Number of Fired Bars (0 to 10)", fontsize=11, fontweight="bold")
    ax_mult.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax_mult.grid(True, linestyle=":", alpha=0.5)
    ax_mult.set_xlim(-0.5, 10.5)
    ax_mult.legend(fontsize=8.0, loc="upper left", framealpha=0.92)

    # -------------------------------------------------------------------------
    # Panel 3: Waterfall Progression of Mean Fired Bars
    # -------------------------------------------------------------------------
    ax_waterfall = fig.add_subplot(gs[1, 0])
    all_stages = stage_names + ["Testbeam Data"]
    all_means = mc_means + [mean_d_mult]
    colors_waterfall = stage_colors + ["#111111"]

    bars_w = ax_waterfall.barh(np.arange(len(all_stages))[::-1], all_means, color=colors_waterfall, alpha=0.85, edgecolor="black", lw=1.0)
    for b_obj, m_val in zip(bars_w, all_means):
        ax_waterfall.text(m_val + 0.12, b_obj.get_y() + b_obj.get_height() / 2.0, f"{m_val:.2f}", va="center", ha="left", fontsize=9.5, fontweight="bold")

    ax_waterfall.axvline(mean_d_mult, color="#111111", linestyle="--", lw=1.8, label=f"Data Target ({mean_d_mult:.2f})")
    ax_waterfall.set_yticks(np.arange(len(all_stages))[::-1])
    ax_waterfall.set_yticklabels(all_stages, fontsize=9.5)
    ax_waterfall.set_xlabel("Mean Fired Bars in US Plane 0", fontsize=11, fontweight="bold")
    ax_waterfall.set_title("Incremental Suppression of Peripheral Shower Leakage", fontsize=12, fontweight="bold")
    ax_waterfall.set_xlim(0, 10.5)
    ax_waterfall.grid(True, linestyle=":", alpha=0.5, axis="x")
    ax_waterfall.legend(loc="lower right", fontsize=9.5)

    # -------------------------------------------------------------------------
    # Panel 4: Peripheral Wing Firing Probability (Bars 0, 1, 8, 9)
    # -------------------------------------------------------------------------
    ax_wings = fig.add_subplot(gs[1, 1])
    # Average firing probability of peripheral bars (0, 1, 8, 9)
    wing_indices = [0, 1, 8, 9]
    d_wing_prob = np.mean([d_prob_bars[b] for b in wing_indices])
    mc_wing_probs = [np.mean([mc_prob_bars[s][b] for b in wing_indices]) for s in range(8)]

    stages_x = np.arange(1, 9)
    ax_wings.plot(stages_x, mc_wing_probs, marker="s", markersize=8, color="#d32f2f", lw=2.5, label="Simulation Wings (Bars 0, 1, 8, 9)")
    for sx, my in zip(stages_x, mc_wing_probs):
        ax_wings.text(sx, my + 1.8, f"{my:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#b71c1c")

    ax_wings.axhline(d_wing_prob, color="#111111", lw=2.2, linestyle="--", label=f"Data Wings ({d_wing_prob:.1f}%)")
    ax_wings.fill_between([0.5, 8.5], 0, d_wing_prob, color="#424242", alpha=0.15)

    ax_wings.set_xticks(stages_x)
    ax_wings.set_xticklabels([f"S{i}" for i in range(1, 9)], fontsize=10, fontweight="bold")
    ax_wings.set_xlabel("Incremental Reduction Stage", fontsize=11, fontweight="bold")
    ax_wings.set_ylabel("Peripheral Wing Firing Probability [%]", fontsize=11, fontweight="bold")
    ax_wings.set_title("Suppression of Low-Energy Peripheral Splash", fontsize=12, fontweight="bold")
    ax_wings.set_ylim(0, 65)
    ax_wings.set_xlim(0.5, 8.5)
    ax_wings.grid(True, linestyle=":", alpha=0.5)
    ax_wings.legend(fontsize=9.5, loc="upper right")

    # Add explanatory annotations on S5 (Birks) and S7 (Dispersion)
    ax_wings.annotate(
        "Birks Quenching\n(Hadronic Recoils)",
        xy=(5, mc_wing_probs[4]),
        xytext=(4.2, mc_wing_probs[4] + 12.0),
        arrowprops=dict(facecolor="#1976d2", arrowstyle="->", lw=1.5),
        fontsize=8.5,
        fontweight="bold",
        color="#0d47a1",
    )
    ax_wings.annotate(
        "Arrival Dispersion\n& Calibration ($k$)",
        xy=(8, mc_wing_probs[7]),
        xytext=(6.5, mc_wing_probs[7] + 16.0),
        arrowprops=dict(facecolor="#388e3c", arrowstyle="->", lw=1.5),
        fontsize=8.5,
        fontweight="bold",
        color="#1b5e20",
    )

    out_png = os.path.join(plot_dir, "tb_incremental_reduction_factors.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    os.system(f"cp -f {out_png} {art_dir}/")
    print(f"[SUCCESS] Saved incremental factors plot to {out_png} and {art_dir}/")


if __name__ == "__main__":
    main()
