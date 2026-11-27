#!/usr/bin/env python3
"""
scripts/qdc_calib/generate_mufi_digi_plots.py
--------------------------------------------
Generates publication-quality validation figures for MuFilter digitization:
1. Incremental physics reduction factor evolution (4-panel figure).
   - Top-left: Data as black markers with error bars, Stage 8 with vertical error bars,
     and filled histograms with high transparency for each colormap curve.
2. Hit multiplicity distributions per system (US, DS, Veto).
   - Comparing Raw uncalibrated MC, Fully calibrated MC, and Data (black markers + error bars).
3. Hit multiplicity distributions per plane (US0..US4, DS1-H, DS1-V).
   - Comparing Raw uncalibrated MC, Fully calibrated MC, and Data (black markers + error bars).
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


def run_extraction(n_data_entries: int = 35000, n_mc_events: int = 600):
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
    d_us_planes_mult = {p: [] for p in range(5)}
    d_us_tot_mult = []
    d_ds_h_mult = []
    d_ds_v_mult = []
    d_ds_tot_mult = []
    d_veto_tot_mult = []
    n_showers_d = 0

    max_d = min(n_data_entries, t_d.GetEntries())
    for i in range(max_d):
        t_d.GetEntry(i)
        if len(t_d.Digi_ScifiHits) < 100:
            continue
        n_showers_d += 1
        fired_us = set()
        fired_us_planes = {p: set() for p in range(5)}
        fired_ds_h = set()
        fired_ds_v = set()
        fired_veto = set()

        for mh in t_d.Digi_MuFilterHits:
            if not mh.isValid():
                continue
            did = mh.GetDetectorID()
            subsys, plane, bar = calc.decode_detector_id(did)
            times = [s.second for s in mh.GetAllTimes() if s.second != -999.0]
            if times and any(abs(t) <= 15.0 for t in times):
                if subsys == 1:
                    fired_veto.add(did)
                elif subsys == 2:
                    fired_us.add(did)
                    fired_us_planes[plane].add(bar)
                elif subsys == 3:
                    if bar < 60:
                        fired_ds_h.add(bar)
                    else:
                        fired_ds_v.add(bar)

        d_us0_mult.append(len(fired_us_planes[0]))
        for b in fired_us_planes[0]:
            if 0 <= b < 10:
                d_us0_bars[b] += 1
        for p in range(5):
            d_us_planes_mult[p].append(len(fired_us_planes[p]))
        d_us_tot_mult.append(len(fired_us))
        d_ds_h_mult.append(len(fired_ds_h))
        d_ds_v_mult.append(len(fired_ds_v))
        d_ds_tot_mult.append(len(fired_ds_h) + len(fired_ds_v))
        d_veto_tot_mult.append(len(fired_veto))

    f_d.Close()
    print(f"      Data: Analyzed {n_showers_d} showers.")

    # -------------------------------------------------------------------------
    # 2. Read Geant4 Simulation
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

    # Full detector distributions
    raw_us_tot_mult = []
    cal_us_tot_mult = []
    raw_us_planes_mult = {p: [] for p in range(5)}
    cal_us_planes_mult = {p: [] for p in range(5)}
    raw_ds_h_mult = []
    cal_ds_h_mult = []
    raw_ds_v_mult = []
    cal_ds_v_mult = []
    raw_ds_tot_mult = []
    cal_ds_tot_mult = []
    raw_veto_tot_mult = []
    cal_veto_tot_mult = []

    actual_mc = min(n_mc_events, t_m.GetEntries())
    for iev in range(actual_mc):
        t_m.GetEntry(iev)
        by_bar = calc.group_mcpoints_by_bar(t_m.MuFilterPoint)

        fired_sets = [set() for _ in range(8)]
        raw_fired_bars = set()
        cal_fired_bars = set()

        for did, clusters in by_bar.items():
            subsys, plane, bar = calc.decode_detector_id(did)
            for pts in clusters:
                tot_e = sum(p.GetEnergyLoss() * 1000.0 for p in pts)
                if tot_e <= 0:
                    continue

                # Raw PE
                pe_raw = 10000.0 * tot_e * 0.50
                if pe_raw >= th_pe:
                    raw_fired_bars.add(did)

                # Calibrated PE
                pes = calc.get_sipm_pes(pts, det_id=did)
                if any(not calc.is_small_sipm(did, s_idx) and p_val >= th_pe for s_idx, p_val in enumerate(pes)):
                    cal_fired_bars.add(did)

                # US Plane 0 incremental stages
                if did // 1000 == 20:
                    pe1 = pe_raw
                    pe2 = pe1 * (0.36 / 6.0)
                    f_c = calc.get_fcone(pts[0], 0)
                    pe3 = pe2 * f_c
                    s_att_fres = sum((p.GetEnergyLoss() * 1000.0) * calc.get_Satt(p, 0) * calc.get_ffresnel(p, 0) for p in pts) / tot_e
                    pe4 = pe3 * s_att_fres
                    e_birks = sum((p.GetEnergyLoss() * 1000.0) * calc.get_fbirks(p) * calc.get_Satt(p, 0) * calc.get_ffresnel(p, 0) for p in pts)
                    pe5 = 10000.0 * e_birks * (0.36 / 6.0) * f_c * 0.50
                    e_edge = sum((p.GetEnergyLoss() * 1000.0) * calc.get_fbirks(p) * calc.get_Satt(p, 0) * calc.get_ffresnel(p, 0) * (calc.get_fcone_eff(p, 0) / f_c) for p in pts)
                    pe6 = 10000.0 * e_edge * (0.36 / 6.0) * f_c * 0.50
                    f_d = calc.get_fdisp(pts, 0)
                    pe7 = pe6 * f_d
                    k_val = calc.get_reduction_factor(did, 0)
                    pe8 = pe7 * k_val

                    pes_inc = [pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8]
                    for s_idx, p_val in enumerate(pes_inc):
                        if p_val >= th_pe:
                            fired_sets[s_idx].add(bar)

                break

        for s_idx in range(8):
            mc_us0_mult[s_idx].append(len(fired_sets[s_idx]))
            for b in fired_sets[s_idx]:
                if 0 <= b < 10:
                    mc_us0_bars[s_idx][b] += 1

        # Multiplicities per system and plane
        raw_us_fired = [d for d in raw_fired_bars if d // 10000 == 2]
        cal_us_fired = [d for d in cal_fired_bars if d // 10000 == 2]
        raw_us_tot_mult.append(len(raw_us_fired))
        cal_us_tot_mult.append(len(cal_us_fired))

        for p in range(5):
            raw_us_planes_mult[p].append(sum(1 for d in raw_us_fired if (d % 10000)//1000 == p))
            cal_us_planes_mult[p].append(sum(1 for d in cal_us_fired if (d % 10000)//1000 == p))

        raw_ds_h = sum(1 for d in raw_fired_bars if d // 10000 == 3 and (d % 1000) < 60)
        raw_ds_v = sum(1 for d in raw_fired_bars if d // 10000 == 3 and (d % 1000) >= 60)
        cal_ds_h = sum(1 for d in cal_fired_bars if d // 10000 == 3 and (d % 1000) < 60)
        cal_ds_v = sum(1 for d in cal_fired_bars if d // 10000 == 3 and (d % 1000) >= 60)

        raw_ds_h_mult.append(raw_ds_h)
        raw_ds_v_mult.append(raw_ds_v)
        raw_ds_tot_mult.append(raw_ds_h + raw_ds_v)

        cal_ds_h_mult.append(cal_ds_h)
        cal_ds_v_mult.append(cal_ds_v)
        cal_ds_tot_mult.append(cal_ds_h + cal_ds_v)

        raw_veto_tot_mult.append(sum(1 for d in raw_fired_bars if d // 10000 == 1))
        cal_veto_tot_mult.append(sum(1 for d in cal_fired_bars if d // 10000 == 1))

    f_m.Close()
    print(f"      MC: Analyzed {actual_mc} events.")

    return {
        "n_showers_d": n_showers_d,
        "n_mc_events": actual_mc,
        "stage_names": stage_names,
        "d_us0_bars": d_us0_bars,
        "d_us0_mult": d_us0_mult,
        "mc_us0_bars": mc_us0_bars,
        "mc_us0_mult": mc_us0_mult,
        # Multiplicities
        "d_us_tot_mult": d_us_tot_mult,
        "raw_us_tot_mult": raw_us_tot_mult,
        "cal_us_tot_mult": cal_us_tot_mult,
        "d_ds_tot_mult": d_ds_tot_mult,
        "raw_ds_tot_mult": raw_ds_tot_mult,
        "cal_ds_tot_mult": cal_ds_tot_mult,
        "d_veto_tot_mult": d_veto_tot_mult,
        "raw_veto_tot_mult": raw_veto_tot_mult,
        "cal_veto_tot_mult": cal_veto_tot_mult,
        "d_us_planes_mult": d_us_planes_mult,
        "raw_us_planes_mult": raw_us_planes_mult,
        "cal_us_planes_mult": cal_us_planes_mult,
        "d_ds_h_mult": d_ds_h_mult,
        "raw_ds_h_mult": raw_ds_h_mult,
        "cal_ds_h_mult": cal_ds_h_mult,
        "d_ds_v_mult": d_ds_v_mult,
        "raw_ds_v_mult": raw_ds_v_mult,
        "cal_ds_v_mult": cal_ds_v_mult,
    }


def plot_incremental(data_dict, out_path="plots/tb_incremental_reduction_factors.png"):
    n_d = data_dict["n_showers_d"]
    n_mc = data_dict["n_mc_events"]
    stage_names = data_dict["stage_names"]

    d_prob_bars = (data_dict["d_us0_bars"] / n_d) * 100.0
    d_prob_err = np.sqrt(d_prob_bars * (100.0 - d_prob_bars) / n_d)
    mean_d_mult = np.mean(data_dict["d_us0_mult"])

    mc_prob_bars = [(bars / n_mc) * 100.0 for bars in data_dict["mc_us0_bars"]]
    mc_prob_err = [np.sqrt(pb * (100.0 - pb) / n_mc) for pb in mc_prob_bars]
    mc_means = [np.mean(m) for m in data_dict["mc_us0_mult"]]

    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])

    cmap = plt.cm.turbo
    stage_colors = [cmap(val) for val in np.linspace(0.08, 0.95, 8)]

    # -------------------------------------------------------------------------
    # Panel 1: US Plane 0 Transverse Profile (Bar 0 to 9) with Subpanel Ratio
    # -------------------------------------------------------------------------
    gs_left = gs[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax_top = fig.add_subplot(gs_left[0])
    ax_bot = fig.add_subplot(gs_left[1], sharex=ax_top)

    x_bars = np.arange(10)

    # Plot incremental stages with colormap and filled area
    for s_idx in range(8):
        c = stage_colors[s_idx]
        lw = 2.4 if s_idx in (0, 4, 7) else 1.6
        ls = "-" if s_idx == 7 else ("--" if s_idx == 0 else "-.")
        
        # High-transparency filled histogram
        ax_top.fill_between(
            x_bars, 0, mc_prob_bars[s_idx],
            step="mid", color=c, alpha=0.15,
        )
        
        # Incremental curve
        if s_idx == 7:  # Stage 8: include vertical error bars
            ax_top.errorbar(
                x_bars, mc_prob_bars[s_idx], yerr=mc_prob_err[s_idx],
                marker="s", markersize=5.0, lw=lw, ls=ls, color=c,
                ecolor=c, elinewidth=1.6, capsize=3.0,
                label=f"{stage_names[s_idx]} ($\\mu$={mc_means[s_idx]:.2f})",
                zorder=6
            )
        else:
            ax_top.plot(
                x_bars, mc_prob_bars[s_idx],
                marker="o", markersize=4.0, lw=lw, ls=ls, color=c,
                label=f"{stage_names[s_idx]} ($\\mu$={mc_means[s_idx]:.2f})",
                zorder=4
            )

        # Ratio to Data in bottom subpanel
        ratio = np.where(d_prob_bars > 0.5, mc_prob_bars[s_idx] / d_prob_bars, np.nan)
        ax_bot.plot(x_bars, ratio, marker="o", markersize=3.5, lw=1.5, ls=ls, color=c)

    # Data as black markers with error bars
    ax_top.errorbar(
        x_bars, d_prob_bars, yerr=d_prob_err,
        fmt="o", color="#111111", ecolor="#111111", elinewidth=2.0, capsize=3.5,
        markersize=6.5, label=f"Data ($|t|\\leq 15$ ns, Mean={mean_d_mult:.2f})",
        zorder=10
    )

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

    for s_idx in range(8):
        c = stage_colors[s_idx]
        lw = 2.4 if s_idx in (0, 4, 7) else 1.5
        ls = "-" if s_idx == 7 else ("--" if s_idx == 0 else "-.")
        ax_mult.hist(
            data_dict["mc_us0_mult"][s_idx], bins=bins_mult, density=True,
            histtype="stepfilled", color=c, alpha=0.15,
        )
        ax_mult.hist(
            data_dict["mc_us0_mult"][s_idx], bins=bins_mult, density=True,
            histtype="step", lw=lw, ls=ls, color=c,
            label=f"{stage_names[s_idx]} ($\\mu$={mc_means[s_idx]:.2f})",
        )

    # Data as black markers with error bars
    d_counts, _ = np.histogram(data_dict["d_us0_mult"], bins=bins_mult)
    d_prob = d_counts / n_d
    d_err = np.sqrt(d_counts) / n_d
    ax_mult.errorbar(
        mult_centers, d_prob, yerr=d_err,
        fmt="o", color="#111111", ecolor="#111111", elinewidth=2.0, capsize=3.5,
        markersize=6.0, label=f"Data ($\\mu$={mean_d_mult:.2f})", zorder=10
    )

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

    ax_wings.annotate(
        "Birks Quenching\n(Hadronic Recoils)",
        xy=(5, mc_wing_probs[4]),
        xytext=(4.0, mc_wing_probs[4] + 12.0),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9.0, fontweight="bold", color="#0d47a1",
    )
    ax_wings.annotate(
        "Arrival Dispersion\n& Calibration ($k$)",
        xy=(8, mc_wing_probs[7]),
        xytext=(6.7, mc_wing_probs[7] + 12.0),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9.0, fontweight="bold", color="#1b5e20",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved incremental plot to {out_path}")


def plot_multiplicity_by_system(data_dict, out_path="plots/tb_mult_by_system.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    systems = [
        ("Upstream (US)", "d_us_tot_mult", "raw_us_tot_mult", "cal_us_tot_mult", np.arange(-0.5, 51.5, 1.0), (0, 45)),
        ("Downstream (DS)", "d_ds_tot_mult", "raw_ds_tot_mult", "cal_ds_tot_mult", np.arange(-0.5, 41.5, 1.0), (0, 35)),
        ("Veto (Not in H8 Beam)", "d_veto_tot_mult", "raw_veto_tot_mult", "cal_veto_tot_mult", np.arange(-0.5, 5.5, 1.0), (-0.5, 4.5)),
    ]

    for ax, (title, k_d, k_raw, k_cal, bins, xlim) in zip(axes, systems):
        d_vals = data_dict[k_d]
        r_vals = data_dict[k_raw]
        c_vals = data_dict[k_cal]

        # Raw MC
        ax.hist(r_vals, bins=bins, density=True, histtype="stepfilled", color="#1976d2", alpha=0.18)
        ax.hist(r_vals, bins=bins, density=True, histtype="step", color="#1976d2", lw=2.0, ls="--", label=f"Raw MC ($\\mu$={np.mean(r_vals):.2f})")

        # Calibrated MC
        ax.hist(c_vals, bins=bins, density=True, histtype="stepfilled", color="#d32f2f", alpha=0.18)
        ax.hist(c_vals, bins=bins, density=True, histtype="step", color="#d32f2f", lw=2.2, label=f"Calibrated MC ($\\mu$={np.mean(c_vals):.2f})")

        # Data as black markers with error bars
        counts, edges = np.histogram(d_vals, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        norm = len(d_vals)
        prob = counts / norm
        err = np.sqrt(counts) / norm
        ax.errorbar(centers, prob, yerr=err, fmt="o", color="#111111", ecolor="#111111", elinewidth=1.8, capsize=3.0, markersize=5.0, label=f"Data ($\\mu$={np.mean(d_vals):.2f})", zorder=10)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Total Fired Bars", fontsize=11, fontweight="bold")
        ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
        ax.set_xlim(xlim)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=9.0, loc="upper right", framealpha=0.92)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved multiplicity by system plot to {out_path}")


def plot_multiplicity_by_plane(data_dict, out_path="plots/tb_mult_by_plane.png"):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    axes_flat = axes.flatten()

    planes = [
        ("US Plane 0", data_dict["d_us_planes_mult"][0], data_dict["raw_us_planes_mult"][0], data_dict["cal_us_planes_mult"][0], np.arange(-0.5, 11.5, 1.0), (-0.5, 10.5)),
        ("US Plane 1", data_dict["d_us_planes_mult"][1], data_dict["raw_us_planes_mult"][1], data_dict["cal_us_planes_mult"][1], np.arange(-0.5, 11.5, 1.0), (-0.5, 10.5)),
        ("US Plane 2", data_dict["d_us_planes_mult"][2], data_dict["raw_us_planes_mult"][2], data_dict["cal_us_planes_mult"][2], np.arange(-0.5, 11.5, 1.0), (-0.5, 10.5)),
        ("US Plane 3", data_dict["d_us_planes_mult"][3], data_dict["raw_us_planes_mult"][3], data_dict["cal_us_planes_mult"][3], np.arange(-0.5, 11.5, 1.0), (-0.5, 10.5)),
        ("US Plane 4", data_dict["d_us_planes_mult"][4], data_dict["raw_us_planes_mult"][4], data_dict["cal_us_planes_mult"][4], np.arange(-0.5, 11.5, 1.0), (-0.5, 10.5)),
        ("DS Station 1 Horiz.", data_dict["d_ds_h_mult"], data_dict["raw_ds_h_mult"], data_dict["cal_ds_h_mult"], np.arange(-0.5, 21.5, 1.0), (-0.5, 20.5)),
        ("DS Station 1 Vert.", data_dict["d_ds_v_mult"], data_dict["raw_ds_v_mult"], data_dict["cal_ds_v_mult"], np.arange(-0.5, 21.5, 1.0), (-0.5, 20.5)),
    ]

    for idx, (title, d_vals, r_vals, c_vals, bins, xlim) in enumerate(planes):
        ax = axes_flat[idx]

        # Raw MC
        ax.hist(r_vals, bins=bins, density=True, histtype="stepfilled", color="#1976d2", alpha=0.18)
        ax.hist(r_vals, bins=bins, density=True, histtype="step", color="#1976d2", lw=1.8, ls="--", label=f"Raw MC ($\\mu$={np.mean(r_vals):.2f})")

        # Calibrated MC
        ax.hist(c_vals, bins=bins, density=True, histtype="stepfilled", color="#d32f2f", alpha=0.18)
        ax.hist(c_vals, bins=bins, density=True, histtype="step", color="#d32f2f", lw=2.0, label=f"Calibrated MC ($\\mu$={np.mean(c_vals):.2f})")

        # Data as black markers with error bars
        counts, edges = np.histogram(d_vals, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        norm = len(d_vals)
        prob = counts / norm
        err = np.sqrt(counts) / norm
        ax.errorbar(centers, prob, yerr=err, fmt="o", color="#111111", ecolor="#111111", elinewidth=1.8, capsize=3.0, markersize=4.5, label=f"Data ($\\mu$={np.mean(d_vals):.2f})", zorder=10)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Fired Bars", fontsize=10, fontweight="bold")
        ax.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
        ax.set_xlim(xlim)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8.0, loc="upper right", framealpha=0.92)

    # Turn off the last unused subplot (axes_flat[7])
    axes_flat[7].axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved multiplicity by plane plot to {out_path}")


if __name__ == "__main__":
    data_dict = run_extraction(n_data_entries=35000, n_mc_events=600)
    # Save cache for fast notebook loading
    cache_path = "data/tb_mufi_mult_cache.npz"
    os.makedirs("data", exist_ok=True)
    np.savez_compressed(cache_path, **data_dict)
    print(f"Cached extracted data to {cache_path}")

    plot_incremental(data_dict, "plots/tb_incremental_reduction_factors.png")
    plot_multiplicity_by_system(data_dict, "plots/tb_mult_by_system.png")
    plot_multiplicity_by_plane(data_dict, "plots/tb_mult_by_plane.png")

    # Also save to artifact directory
    art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"
    plot_incremental(data_dict, os.path.join(art_dir, "tb_incremental_reduction_factors.png"))
    plot_multiplicity_by_system(data_dict, os.path.join(art_dir, "tb_mult_by_system.png"))
    plot_multiplicity_by_plane(data_dict, os.path.join(art_dir, "tb_mult_by_plane.png"))
