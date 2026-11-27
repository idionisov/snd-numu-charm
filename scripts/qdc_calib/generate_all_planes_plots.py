#!/usr/bin/env python3
"""
scripts/qdc_calib/generate_all_planes_plots.py
----------------------------------------------
Extracts 8-stage incremental reduction factors for:
  - All 5 Upstream (US) Calorimeter Planes (US 0 to 4) & Total US
  - Downstream (DS) Muon Filter Stations & Total DS
Compares against 180 GeV pi+ testbeam data (Run 100671).

Generates:
  1. plots/tb_all_us_planes_incremental_factors.png (Total US 4-panel evolution)
  2. plots/tb_us_planes_5panel_transverse_profiles.png (US Planes 0..4 transverse profiles + ratios)
  3. plots/tb_ds_planes_incremental_factors.png (DS MuFilter 4-panel evolution)
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import load_trident_libraries
load_trident_libraries(_repo_root)

import scripts.qdc_calib.min_pe as mpe

# Parameter sources: August 2023 '3 walls, ALL IN' 180 GeV pi+ testbeam run (Run 100671) from CERN SPS H8 logbook:
# Physics justification for incremental reduction stages:
# Stage 1: Raw Scintillation yield (LIGHT_YIELD = 10000 ph/MeV from EJ-200 datasheet)
# Stage 2: Geometric area coverage fraction f_area (SiPM active area / bar endcap area)
# Stage 3: Optical acceptance cone fraction f_cone (total internal reflection trapping from Snell's law)
# Stage 4: Bulk technical attenuation S_att and Fresnel interface transmission f_fresnel
# Stage 5: Birks quenching f_birks (hadronic recoils and high dE/dx stopping power saturation)
# Stage 6: Cut edge degradation and dead skin layer f_edge (machining micro-roughness and wrapping loss)
# Stage 7: Photon arrival time dispersion attenuation f_disp (waveform spreading across shower lateral width)
# Stage 8: SiPM channel-specific empirical calibration factor k (optical coupling, alignment, ASIC baseline)
STAGE_NAMES = [
    "1. Raw Scintillation",
    "2. + $f_{\\mathrm{area}}$",
    "3. + $f_{\\mathrm{cone}}$",
    "4. + $S_{\\mathrm{att}} \\cdot f_{\\mathrm{Fresnel}}$",
    "5. + $f_{\\mathrm{Birks}}$ (Hadron)",
    "6. + $f_{\\mathrm{edge}}$ (Dead Skin)",
    "7. + $f_{\\mathrm{disp}}$ (Arrival)",
    "8. + $k$ (MIP-Calib, Final)",
]


def extract_data_and_mc(n_data_entries: int = 25000, n_mc_events: int = 600, cache_path: str = "data/tb_mufi_all_planes_cache.npz"):
    calc = mpe.MinPE(year=2023, is_testbeam=True, include_edge_effect=True, include_dispersion=True)
    th_pe = calc.get_sipm_threshold_pe()

    data_file = "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100671/sndsw_raw-0000.root"
    print(f"[1/2] Reading Data from {data_file}...")
    f_d = ROOT.TFile.Open(data_file)
    t_d = f_d.Get("rawConv")
    t_d.SetBranchStatus("*", 0)
    t_d.SetBranchStatus("Digi_ScifiHits*", 1)
    t_d.SetBranchStatus("Digi_MuFilterHits*", 1)

    d_us_bars = [np.zeros(10, dtype=float) for _ in range(5)]
    d_us_mult = [[] for _ in range(5)]
    d_us_tot_mult = []

    d_dsh_bars = np.zeros(60, dtype=float)
    d_dsv_bars = np.zeros(60, dtype=float)
    d_dsh_mult = []
    d_dsv_mult = []
    d_ds_tot_mult = []

    n_showers_d = 0
    max_d = min(n_data_entries, t_d.GetEntries())
    for i in range(max_d):
        t_d.GetEntry(i)
        if len(t_d.Digi_ScifiHits) < 100:
            continue
        n_showers_d += 1

        fired_us_p = [set() for _ in range(5)]
        fired_dsh = set()
        fired_dsv = set()

        for mh in t_d.Digi_MuFilterHits:
            if not mh.isValid():
                continue
            did = mh.GetDetectorID()
            subsys, plane, bar = calc.decode_detector_id(did)
            times = [s.second for s in mh.GetAllTimes() if s.second != -999.0]
            if times and any(abs(t) <= 15.0 for t in times):
                if subsys == 2 and 0 <= plane < 5:
                    fired_us_p[plane].add(bar)
                elif subsys == 3:
                    if bar < 60:
                        fired_dsh.add(bar)
                    else:
                        fired_dsv.add(bar - 60)

        tot_us = 0
        for p in range(5):
            d_us_mult[p].append(len(fired_us_p[p]))
            tot_us += len(fired_us_p[p])
            for b in fired_us_p[p]:
                if 0 <= b < 10:
                    d_us_bars[p][b] += 1
        d_us_tot_mult.append(tot_us)

        d_dsh_mult.append(len(fired_dsh))
        for b in fired_dsh:
            if 0 <= b < 60:
                d_dsh_bars[b] += 1

        d_dsv_mult.append(len(fired_dsv))
        for b in fired_dsv:
            if 0 <= b < 60:
                d_dsv_bars[b] += 1

        d_ds_tot_mult.append(len(fired_dsh) + len(fired_dsv))

    f_d.Close()
    print(f"      Data: Analyzed {n_showers_d} showers. Total US Mean = {np.mean(d_us_tot_mult):.2f}, Total DS Mean = {np.mean(d_ds_tot_mult):.2f}")

    mc_file = "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/180GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root"
    print(f"[2/2] Reading MC from {mc_file}...")
    f_m = ROOT.TFile.Open(mc_file)
    t_m = f_m.Get("cbmsim")
    t_m.SetBranchStatus("*", 0)
    t_m.SetBranchStatus("MuFilterPoint*", 1)

    mc_us_bars = [[np.zeros(10, dtype=float) for _ in range(8)] for _ in range(5)]
    mc_us_mult = [{p: [] for p in range(5)} for _ in range(8)]
    mc_us_tot_mult = [[] for _ in range(8)]

    mc_dsh_bars = [np.zeros(60, dtype=float) for _ in range(8)]
    mc_dsv_bars = [np.zeros(60, dtype=float) for _ in range(8)]
    mc_dsh_mult = [[] for _ in range(8)]
    mc_dsv_mult = [[] for _ in range(8)]
    mc_ds_tot_mult = [[] for _ in range(8)]

    actual_mc = min(n_mc_events, t_m.GetEntries())
    for iev in range(actual_mc):
        t_m.GetEntry(iev)
        by_bar = calc.group_mcpoints_by_bar(t_m.MuFilterPoint)

        fired_bars_by_stage = [set() for _ in range(8)]

        for did, clusters in by_bar.items():
            subsys, plane, bar = calc.decode_detector_id(did)
            if subsys not in (2, 3):
                continue
            is_ds = (subsys == 3)
            area_ratio = (0.36 / 1.0) if is_ds else (0.36 / 6.0)

            for pts in clusters:
                tot_e = sum(p.GetEnergyLoss() * 1000.0 for p in pts)
                if tot_e <= 0:
                    continue

                pe1 = 10000.0 * tot_e * 0.50
                pe2 = pe1 * area_ratio
                f_c = calc.get_fcone(pts[0], 0)
                pe3 = pe2 * f_c
                s_att_fres = sum((p.GetEnergyLoss() * 1000.0) * calc.get_Satt(p, 0) * calc.get_ffresnel(p, 0) for p in pts) / tot_e
                pe4 = pe3 * s_att_fres
                e_birks = sum((p.GetEnergyLoss() * 1000.0) * calc.get_fbirks(p) * calc.get_Satt(p, 0) * calc.get_ffresnel(p, 0) for p in pts)
                pe5 = 10000.0 * e_birks * area_ratio * f_c * 0.50
                e_edge = sum((p.GetEnergyLoss() * 1000.0) * calc.get_fbirks(p) * calc.get_Satt(p, 0) * calc.get_ffresnel(p, 0) * (calc.get_fcone_eff(p, 0) / f_c) for p in pts)
                pe6 = 10000.0 * e_edge * area_ratio * f_c * 0.50
                f_d = calc.get_fdisp(pts, 0)
                pe7 = pe6 * f_d
                k_val = calc.get_reduction_factor(did, 0)
                pe8 = pe7 * k_val

                pes_inc = [pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8]
                for s_idx, p_val in enumerate(pes_inc):
                    if p_val >= th_pe:
                        fired_bars_by_stage[s_idx].add(did)
                break

        for s_idx in range(8):
            fired = fired_bars_by_stage[s_idx]
            us_fired = [d for d in fired if d // 10000 == 2]
            ds_fired = [d for d in fired if d // 10000 == 3]

            mc_us_tot_mult[s_idx].append(len(us_fired))
            mc_ds_tot_mult[s_idx].append(len(ds_fired))

            for p in range(5):
                p_bars = [(d % 1000) for d in us_fired if (d % 10000) // 1000 == p]
                mc_us_mult[s_idx][p].append(len(p_bars))
                for b in p_bars:
                    if 0 <= b < 10:
                        mc_us_bars[p][s_idx][b] += 1

            dsh_bars = [(d % 1000) for d in ds_fired if (d % 1000) < 60]
            dsv_bars = [(d % 1000) - 60 for d in ds_fired if (d % 1000) >= 60]
            mc_dsh_mult[s_idx].append(len(dsh_bars))
            mc_dsv_mult[s_idx].append(len(dsv_bars))
            for b in dsh_bars:
                if 0 <= b < 60:
                    mc_dsh_bars[s_idx][b] += 1
            for b in dsv_bars:
                if 0 <= b < 60:
                    mc_dsv_bars[s_idx][b] += 1

    f_m.Close()
    print(f"      MC: Analyzed {actual_mc} events. Total US Mean S1={np.mean(mc_us_tot_mult[0]):.2f} -> S8={np.mean(mc_us_tot_mult[7]):.2f}")

    out_data = {
        "n_showers_d": n_showers_d,
        "actual_mc": actual_mc,
        "d_us_bars": d_us_bars,
        "d_us_mult": d_us_mult,
        "d_us_tot_mult": d_us_tot_mult,
        "d_dsh_bars": d_dsh_bars,
        "d_dsv_bars": d_dsv_bars,
        "d_dsh_mult": d_dsh_mult,
        "d_dsv_mult": d_dsv_mult,
        "d_ds_tot_mult": d_ds_tot_mult,
        "mc_us_bars": mc_us_bars,
        "mc_us_mult": mc_us_mult,
        "mc_us_tot_mult": mc_us_tot_mult,
        "mc_dsh_bars": mc_dsh_bars,
        "mc_dsv_bars": mc_dsv_bars,
        "mc_dsh_mult": mc_dsh_mult,
        "mc_dsv_mult": mc_dsv_mult,
        "mc_ds_tot_mult": mc_ds_tot_mult,
    }
    np.savez_compressed(cache_path, **out_data)
    print(f"[✓] Saved extracted cache to {cache_path}")
    return out_data


def plot_all_us_planes(data_dict: dict, out_png: str):
    """Generate 4-panel evolution plot for Total US system."""
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])

    cmap = plt.cm.turbo
    stage_colors = [cmap(val) for val in np.linspace(0.08, 0.95, 8)]

    n_d = data_dict["n_showers_d"]
    n_mc = data_dict["actual_mc"]

    gs_left = gs[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax_top = fig.add_subplot(gs_left[0])
    ax_bot = fig.add_subplot(gs_left[1], sharex=ax_top)

    x_planes = np.arange(5)
    d_means = np.array([float(np.mean(data_dict["d_us_mult"][p])) for p in range(5)])
    d_errs = np.array([float(np.std(data_dict["d_us_mult"][p]) / np.sqrt(n_d)) for p in range(5)])

    mc_means_by_stage = []
    mc_errs_by_stage = []
    for s_idx in range(8):
        m_s = np.array([float(np.mean(data_dict["mc_us_mult"][s_idx][p])) for p in range(5)])
        e_s = np.array([float(np.std(data_dict["mc_us_mult"][s_idx][p]) / np.sqrt(n_mc)) for p in range(5)])
        mc_means_by_stage.append(m_s)
        mc_errs_by_stage.append(e_s)

    for s_idx in range(8):
        c = stage_colors[s_idx]
        lw = 2.4 if s_idx in (0, 4, 7) else 1.6
        ls = "-" if s_idx == 7 else ("--" if s_idx == 0 else "-.")
        m_vals = mc_means_by_stage[s_idx]
        tot_m = float(np.mean(data_dict["mc_us_tot_mult"][s_idx]))

        if s_idx == 7:
            ax_top.errorbar(
                x_planes, m_vals, yerr=mc_errs_by_stage[s_idx],
                marker="s", markersize=5.5, lw=lw, ls=ls, color=c,
                ecolor=c, elinewidth=1.6, capsize=3.0,
                label=f"{STAGE_NAMES[s_idx]} (Total={tot_m:.1f})", zorder=6
            )
            ratio_s8 = m_vals / d_means
            r_err = ratio_s8 * np.sqrt((mc_errs_by_stage[s_idx] / m_vals)**2 + (d_errs / d_means)**2)
            ax_bot.errorbar(x_planes, ratio_s8, yerr=r_err, marker="o", markersize=4.0, lw=1.8, ls=ls, color=c, ecolor=c, elinewidth=1.4, capsize=2.5, zorder=6)
        else:
            ax_top.plot(x_planes, m_vals, marker="o", markersize=4.0, lw=lw, ls=ls, color=c, label=f"{STAGE_NAMES[s_idx]} (Total={tot_m:.1f})", zorder=4)
            ratio = m_vals / d_means
            ax_bot.plot(x_planes, ratio, marker="o", markersize=3.5, lw=1.5, ls=ls, color=c)

    d_tot_mean = float(np.mean(data_dict["d_us_tot_mult"]))
    ax_top.errorbar(x_planes, d_means, yerr=d_errs, fmt="o", color="#111111", ecolor="#111111", elinewidth=2.0, capsize=3.5, markersize=6.5, label=f"Data (Total={d_tot_mean:.1f})", zorder=10)
    ax_top.set_ylabel("Mean Fired Bars", fontsize=11, fontweight="bold")
    ax_top.set_title("Longitudinal Shower Profile across US Planes (0 to 4)", fontsize=12, fontweight="bold")
    ax_top.grid(True, linestyle=":", alpha=0.5)
    ax_top.set_ylim(0, 11)
    ax_top.legend(fontsize=8.0, loc="upper right", framealpha=0.92, ncol=2)

    ax_bot.axhline(1.0, color="#111111", lw=1.5, linestyle="--")
    ax_bot.set_ylabel("MC / Data", fontsize=10, fontweight="bold")
    ax_bot.set_xlabel("US Plane Index (0=US1 Front, 4=US5 Rear)", fontsize=11, fontweight="bold")
    ax_bot.set_ylim(0.5, 2.8)
    ax_bot.set_xticks(x_planes)
    ax_bot.set_xticklabels([f"Plane {p}" for p in range(5)], fontsize=10, fontweight="bold")
    ax_bot.grid(True, linestyle=":", alpha=0.5)

    ax_mult = fig.add_subplot(gs[0, 1])
    bins_tot = np.arange(-0.5, 50.5, 2.0)
    centers_tot = 0.5 * (bins_tot[:-1] + bins_tot[1:])

    for s_idx in range(8):
        c = stage_colors[s_idx]
        lw = 2.4 if s_idx in (0, 4, 7) else 1.5
        ls = "-" if s_idx == 7 else ("--" if s_idx == 0 else "-.")
        ax_mult.hist(data_dict["mc_us_tot_mult"][s_idx], bins=bins_tot, density=True, histtype="stepfilled", color=c, alpha=0.06)
        ax_mult.hist(data_dict["mc_us_tot_mult"][s_idx], bins=bins_tot, density=True, histtype="step", lw=lw, ls=ls, color=c, label=f"{STAGE_NAMES[s_idx]} ($\mu$={np.mean(data_dict['mc_us_tot_mult'][s_idx]):.1f})")

    counts_s8, _ = np.histogram(data_dict["mc_us_tot_mult"][7], bins=bins_tot)
    prob_s8 = counts_s8 / (n_mc * 2.0)
    err_s8 = np.sqrt(counts_s8) / (n_mc * 2.0)
    mask_s8 = prob_s8 > 0
    ax_mult.errorbar(centers_tot[mask_s8], prob_s8[mask_s8], yerr=err_s8[mask_s8], fmt="none", ecolor=stage_colors[7], elinewidth=1.6, capsize=2.0, zorder=7)

    d_counts, _ = np.histogram(data_dict["d_us_tot_mult"], bins=bins_tot)
    d_prob = d_counts / (n_d * 2.0)
    d_err = np.sqrt(d_counts) / (n_d * 2.0)
    mask_d = d_prob > 0
    ax_mult.errorbar(centers_tot[mask_d], d_prob[mask_d], yerr=d_err[mask_d], fmt="o", color="#111111", ecolor="#111111", elinewidth=2.0, capsize=3.0, markersize=5.5, label=f"Data ($\mu$={d_tot_mean:.1f})", zorder=10)

    ax_mult.set_xlabel("Total Fired Bars in US Subsystem (0 to 50)", fontsize=11, fontweight="bold")
    ax_mult.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax_mult.set_title("Total US Multiplicity Distribution Evolution", fontsize=12, fontweight="bold")
    ax_mult.grid(True, linestyle=":", alpha=0.5)
    ax_mult.set_xlim(-0.5, 48)
    ax_mult.legend(fontsize=8.0, loc="upper right", framealpha=0.92)

    ax_waterfall = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(9)
    means_us_tot = [float(np.mean(data_dict["mc_us_tot_mult"][s])) for s in range(8)] + [d_tot_mean]
    colors_waterfall = stage_colors + ["#212121"]
    labels_waterfall = STAGE_NAMES + ["Testbeam Data"]

    bars = ax_waterfall.barh(y_pos, means_us_tot, color=colors_waterfall, edgecolor="#111111", lw=1.2, height=0.68, alpha=0.88)
    ax_waterfall.axvline(d_tot_mean, color="black", linestyle="--", lw=2.0, label=f"Data Target ({d_tot_mean:.1f})")

    for b, val in zip(bars, means_us_tot):
        ax_waterfall.text(val + 0.5, b.get_y() + b.get_height() / 2, f"{val:.1f}", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#111111")

    ax_waterfall.set_yticks(y_pos)
    ax_waterfall.set_yticklabels(labels_waterfall, fontsize=9.5)
    ax_waterfall.invert_yaxis()
    ax_waterfall.set_xlabel("Mean Total Fired Bars in US Subsystem", fontsize=11, fontweight="bold")
    ax_waterfall.set_title("Suppression of Total US Hadronic Shower Multiplicity", fontsize=12, fontweight="bold")
    ax_waterfall.set_xlim(0, 45)
    ax_waterfall.grid(True, linestyle=":", alpha=0.5, axis="x")
    ax_waterfall.legend(fontsize=10.0, loc="lower right")

    ax_wings = fig.add_subplot(gs[1, 1])
    stages_x = np.arange(1, 9)

    d_wing_prob = float(np.mean([
        (data_dict["d_us_bars"][p][b] / n_d) * 100.0 for p in range(5) for b in (0, 1, 8, 9)
    ]))
    mc_wing_probs = []
    mc_wing_errs = []
    for s in range(8):
        vals = [(data_dict["mc_us_bars"][p][s][b] / n_mc) * 100.0 for p in range(5) for b in (0, 1, 8, 9)]
        mc_wing_probs.append(float(np.mean(vals)))
        mc_wing_errs.append(float(np.std(vals) / np.sqrt(len(vals))))

    ax_wings.plot(stages_x, mc_wing_probs, marker="s", markersize=6.5, lw=2.4, color="#d32f2f", label="Simulation Wings (Bars 0, 1, 8, 9 across US 0..4)", zorder=5)
    ax_wings.errorbar(stages_x, mc_wing_probs, yerr=mc_wing_errs, fmt="none", ecolor="#d32f2f", elinewidth=1.6, capsize=3.0, zorder=6)
    ax_wings.axhline(d_wing_prob, color="#111111", linestyle="--", lw=2.0, label=f"Data Wings ({d_wing_prob:.1f}%)")
    ax_wings.fill_between([0.5, 8.5], 0, d_wing_prob, color="#424242", alpha=0.15)

    ax_wings.set_xticks(stages_x)
    ax_wings.set_xticklabels([f"S{i}" for i in range(1, 9)], fontsize=10, fontweight="bold")
    ax_wings.set_xlabel("Incremental Reduction Stage", fontsize=11, fontweight="bold")
    ax_wings.set_ylabel("Mean Wing Firing Probability [%]", fontsize=11, fontweight="bold")
    ax_wings.set_title("Suppression of Peripheral Shower Leakage across All US Planes", fontsize=12, fontweight="bold")
    ax_wings.set_ylim(0, 100)
    ax_wings.set_xlim(0.5, 8.5)
    ax_wings.grid(True, linestyle=":", alpha=0.5)
    ax_wings.legend(fontsize=9.5, loc="upper right")

    ax_wings.annotate("Birks Quenching\n(Hadronic Recoils)", xy=(5, mc_wing_probs[4]), xytext=(3.8, mc_wing_probs[4] + 16.0), arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6), fontsize=9.0, fontweight="bold", color="#0d47a1")
    ax_wings.annotate("Arrival Dispersion\n& Calibration ($k$)", xy=(8, mc_wing_probs[7]), xytext=(6.2, mc_wing_probs[7] + 16.0), arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6), fontsize=9.0, fontweight="bold", color="#1b5e20")

    if out_png:
        plt.savefig(out_png, dpi=300)
        print(f"[✓] Saved {out_png}")
    return fig


def plot_5panel_us_transverse_profiles(data_dict: dict, out_png: str):
    """Generate 5-column plot showing transverse profiles and MC/Data ratio for US Planes 0 to 4."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), gridspec_kw={"height_ratios": [3, 1.2]}, constrained_layout=True)

    n_d = data_dict["n_showers_d"]
    n_mc = data_dict["actual_mc"]
    x_bars = np.arange(10)

    for p in range(5):
        ax_main = axes[0, p]
        ax_ratio = axes[1, p]

        d_prob = (data_dict["d_us_bars"][p] / n_d) * 100.0
        d_err = np.sqrt(d_prob * (100.0 - d_prob) / n_d)
        mean_d = float(np.mean(data_dict["d_us_mult"][p]))

        raw_prob = (data_dict["mc_us_bars"][p][0] / n_mc) * 100.0
        mean_raw = float(np.mean(data_dict["mc_us_mult"][0][p]))

        cal_prob = (data_dict["mc_us_bars"][p][7] / n_mc) * 100.0
        cal_err = np.sqrt(cal_prob * (100.0 - cal_prob) / n_mc)
        mean_cal = float(np.mean(data_dict["mc_us_mult"][7][p]))

        ax_main.plot(x_bars, raw_prob, marker="s", markersize=4.0, lw=1.8, ls=":", color="#1976d2", label=f"Raw MC ($\mu$={mean_raw:.2f})")
        ax_main.errorbar(x_bars, cal_prob, yerr=cal_err, marker="o", markersize=4.5, lw=2.2, ls="-", color="#d32f2f", ecolor="#d32f2f", elinewidth=1.6, capsize=2.5, label=f"Calibrated MC ($\mu$={mean_cal:.2f})", zorder=6)
        ax_main.errorbar(x_bars, d_prob, yerr=d_err, fmt="o", color="#111111", ecolor="#111111", elinewidth=1.8, capsize=3.0, markersize=5.0, label=f"Data ($\mu$={mean_d:.2f})", zorder=10)

        ax_main.set_title(f"US Plane {p}", fontsize=12, fontweight="bold")
        if p == 0:
            ax_main.set_ylabel("Firing Probability [%]", fontsize=11, fontweight="bold")
        ax_main.set_ylim(0, 115)
        ax_main.grid(True, linestyle=":", alpha=0.5)
        ax_main.legend(fontsize=8.0, loc="upper right")

        ratio_raw = np.where(d_prob > 1.0, raw_prob / d_prob, np.nan)
        ratio_cal = np.where(d_prob > 1.0, cal_prob / d_prob, np.nan)
        mask_rc = (d_prob > 1.0) & (cal_prob > 0)
        r_err = np.full_like(ratio_cal, np.nan)
        r_err[mask_rc] = ratio_cal[mask_rc] * np.sqrt((cal_err[mask_rc] / cal_prob[mask_rc])**2 + (d_err[mask_rc] / d_prob[mask_rc])**2)

        ax_ratio.axhline(1.0, color="#111111", lw=1.4, linestyle="--")
        ax_ratio.plot(x_bars, ratio_raw, marker="s", markersize=3.5, color="#1976d2", ls=":", lw=1.4)
        ax_ratio.errorbar(x_bars, ratio_cal, yerr=r_err, marker="o", markersize=4.0, color="#d32f2f", ecolor="#d32f2f", elinewidth=1.4, capsize=2.0, ls="-", lw=1.8)

        if p == 0:
            ax_ratio.set_ylabel("MC / Data", fontsize=10, fontweight="bold")
        ax_ratio.set_xlabel("Bar Index (0=Bottom, 9=Top)", fontsize=10, fontweight="bold")
        ax_ratio.set_ylim(0.2, 3.8)
        ax_ratio.set_xticks(x_bars)
        ax_ratio.grid(True, linestyle=":", alpha=0.5)

    if out_png:
        plt.savefig(out_png, dpi=300)
        print(f"[✓] Saved {out_png}")
    return fig


def plot_ds_planes(data_dict: dict, out_png: str):
    """Generate 4-panel evolution plot for Downstream (DS) Muon Filter."""
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])

    cmap = plt.cm.turbo
    stage_colors = [cmap(val) for val in np.linspace(0.08, 0.95, 8)]

    n_d = data_dict["n_showers_d"]
    n_mc = data_dict["actual_mc"]

    gs_left = gs[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax_top = fig.add_subplot(gs_left[0])
    ax_bot = fig.add_subplot(gs_left[1], sharex=ax_top)

    sub_bars = np.arange(15, 46)
    d_dsh_prob = (data_dict["d_dsh_bars"][sub_bars] / n_d) * 100.0
    d_dsh_err = np.sqrt(d_dsh_prob * (100.0 - d_dsh_prob) / n_d)
    mean_d_ds = float(np.mean(data_dict["d_ds_tot_mult"]))

    for s_idx in range(8):
        c = stage_colors[s_idx]
        lw = 2.4 if s_idx in (0, 4, 7) else 1.6
        ls = "-" if s_idx == 7 else ("--" if s_idx == 0 else "-.")
        mc_prob = (data_dict["mc_dsh_bars"][s_idx][sub_bars] / n_mc) * 100.0
        tot_m = float(np.mean(data_dict["mc_ds_tot_mult"][s_idx]))

        if s_idx == 7:
            mc_err = np.sqrt(mc_prob * (100.0 - mc_prob) / n_mc)
            ax_top.errorbar(sub_bars, mc_prob, yerr=mc_err, marker="s", markersize=4.5, lw=lw, ls=ls, color=c, ecolor=c, elinewidth=1.5, capsize=2.5, label=f"{STAGE_NAMES[s_idx]} (Total={tot_m:.1f})", zorder=6)
            ratio_s8 = np.where(d_dsh_prob > 0.5, mc_prob / d_dsh_prob, np.nan)
            r_err = ratio_s8 * np.sqrt((mc_err / np.maximum(mc_prob, 1e-3))**2 + (d_dsh_err / np.maximum(d_dsh_prob, 1e-3))**2)
            ax_bot.errorbar(sub_bars, ratio_s8, yerr=r_err, marker="o", markersize=3.5, lw=1.8, ls=ls, color=c, ecolor=c, elinewidth=1.4, capsize=2.0, zorder=6)
        else:
            ax_top.plot(sub_bars, mc_prob, marker="o", markersize=3.5, lw=lw, ls=ls, color=c, label=f"{STAGE_NAMES[s_idx]} (Total={tot_m:.1f})", zorder=4)
            ratio = np.where(d_dsh_prob > 0.5, mc_prob / d_dsh_prob, np.nan)
            ax_bot.plot(sub_bars, ratio, marker="o", markersize=3.0, lw=1.4, ls=ls, color=c)

    ax_top.errorbar(sub_bars, d_dsh_prob, yerr=d_dsh_err, fmt="o", color="#111111", ecolor="#111111", elinewidth=1.8, capsize=3.0, markersize=5.0, label=f"Data (Total={mean_d_ds:.1f})", zorder=10)
    ax_top.set_ylabel("Firing Probability [%]", fontsize=11, fontweight="bold")
    ax_top.set_title("Transverse Profile in DS Station 0 (Horizontal Bars 15 to 45)", fontsize=12, fontweight="bold")
    ax_top.grid(True, linestyle=":", alpha=0.5)
    ax_top.set_ylim(0, 45)
    ax_top.legend(fontsize=8.0, loc="upper right", framealpha=0.92, ncol=2)

    ax_bot.axhline(1.0, color="#111111", lw=1.5, linestyle="--")
    ax_bot.set_ylabel("MC / Data", fontsize=10, fontweight="bold")
    ax_bot.set_xlabel("DS-H Bar Index (1 cm pitch)", fontsize=11, fontweight="bold")
    ax_bot.set_ylim(0.2, 3.8)
    ax_bot.grid(True, linestyle=":", alpha=0.5)

    ax_mult = fig.add_subplot(gs[0, 1])
    bins_ds = np.arange(-0.5, 25.5, 1.0)
    centers_ds = np.arange(25)

    for s_idx in range(8):
        c = stage_colors[s_idx]
        lw = 2.4 if s_idx in (0, 4, 7) else 1.5
        ls = "-" if s_idx == 7 else ("--" if s_idx == 0 else "-.")
        ax_mult.hist(data_dict["mc_ds_tot_mult"][s_idx], bins=bins_ds, density=True, histtype="stepfilled", color=c, alpha=0.06)
        ax_mult.hist(data_dict["mc_ds_tot_mult"][s_idx], bins=bins_ds, density=True, histtype="step", lw=lw, ls=ls, color=c, label=f"{STAGE_NAMES[s_idx]} ($\mu$={np.mean(data_dict['mc_ds_tot_mult'][s_idx]):.2f})")

    counts_s8, _ = np.histogram(data_dict["mc_ds_tot_mult"][7], bins=bins_ds)
    prob_s8 = counts_s8 / n_mc
    err_s8 = np.sqrt(counts_s8) / n_mc
    ax_mult.errorbar(centers_ds, prob_s8, yerr=err_s8, fmt="none", ecolor=stage_colors[7], elinewidth=1.6, capsize=2.0, zorder=7)

    d_counts, _ = np.histogram(data_dict["d_ds_tot_mult"], bins=bins_ds)
    d_prob = d_counts / n_d
    d_err = np.sqrt(d_counts) / n_d
    ax_mult.errorbar(centers_ds, d_prob, yerr=d_err, fmt="o", color="#111111", ecolor="#111111", elinewidth=2.0, capsize=3.0, markersize=5.5, label=f"Data ($\mu$={mean_d_ds:.2f})", zorder=10)

    ax_mult.set_xlabel("Total Fired Bars in DS Subsystem", fontsize=11, fontweight="bold")
    ax_mult.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax_mult.set_title("DS Multiplicity Distribution Evolution", fontsize=12, fontweight="bold")
    ax_mult.grid(True, linestyle=":", alpha=0.5)
    ax_mult.set_xlim(-0.5, 22)
    ax_mult.legend(fontsize=8.0, loc="upper right", framealpha=0.92)

    ax_waterfall = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(9)
    means_ds = [float(np.mean(data_dict["mc_ds_tot_mult"][s])) for s in range(8)] + [mean_d_ds]
    colors_waterfall = stage_colors + ["#212121"]
    labels_waterfall = STAGE_NAMES + ["Testbeam Data"]

    bars = ax_waterfall.barh(y_pos, means_ds, color=colors_waterfall, edgecolor="#111111", lw=1.2, height=0.68, alpha=0.88)
    ax_waterfall.axvline(mean_d_ds, color="black", linestyle="--", lw=2.0, label=f"Data Target ({mean_d_ds:.2f})")

    for b, val in zip(bars, means_ds):
        ax_waterfall.text(val + 0.15, b.get_y() + b.get_height() / 2, f"{val:.2f}", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#111111")

    ax_waterfall.set_yticks(y_pos)
    ax_waterfall.set_yticklabels(labels_waterfall, fontsize=9.5)
    ax_waterfall.invert_yaxis()
    ax_waterfall.set_xlabel("Mean Total Fired Bars in DS Subsystem", fontsize=11, fontweight="bold")
    ax_waterfall.set_title("Suppression of Downstream Shower Punch-Through", fontsize=12, fontweight="bold")
    ax_waterfall.set_xlim(0, 11)
    ax_waterfall.grid(True, linestyle=":", alpha=0.5, axis="x")
    ax_waterfall.legend(fontsize=10.0, loc="lower right")

    ax_wings = fig.add_subplot(gs[1, 1])
    stages_x = np.arange(1, 9)

    wing_indices = np.concatenate([np.arange(0, 15), np.arange(45, 60)])
    d_wing_prob = float(np.mean((data_dict["d_dsh_bars"][wing_indices] / n_d) * 100.0))

    mc_wing_probs = []
    mc_wing_errs = []
    for s in range(8):
        vals = (data_dict["mc_dsh_bars"][s][wing_indices] / n_mc) * 100.0
        mc_wing_probs.append(float(np.mean(vals)))
        mc_wing_errs.append(float(np.std(vals) / np.sqrt(len(vals))))

    ax_wings.plot(stages_x, mc_wing_probs, marker="s", markersize=6.5, lw=2.4, color="#d32f2f", label="Simulation DS Wings (Bars <15 & >45)", zorder=5)
    ax_wings.errorbar(stages_x, mc_wing_probs, yerr=mc_wing_errs, fmt="none", ecolor="#d32f2f", elinewidth=1.6, capsize=3.0, zorder=6)
    ax_wings.axhline(d_wing_prob, color="#111111", linestyle="--", lw=2.0, label=f"Data DS Wings ({d_wing_prob:.2f}%)")
    ax_wings.fill_between([0.5, 8.5], 0, d_wing_prob, color="#424242", alpha=0.15)

    ax_wings.set_xticks(stages_x)
    ax_wings.set_xticklabels([f"S{i}" for i in range(1, 9)], fontsize=10, fontweight="bold")
    ax_wings.set_xlabel("Incremental Reduction Stage", fontsize=11, fontweight="bold")
    ax_wings.set_ylabel("Mean Wing Firing Probability [%]", fontsize=11, fontweight="bold")
    ax_wings.set_title("Suppression of Low-Energy Peripheral Splash in DS", fontsize=12, fontweight="bold")
    ax_wings.set_ylim(0, 50)
    ax_wings.set_xlim(0.5, 8.5)
    ax_wings.grid(True, linestyle=":", alpha=0.5)
    ax_wings.legend(fontsize=9.5, loc="upper right")

    ax_wings.annotate("Birks Quenching\n(Hadronic Recoils)", xy=(5, mc_wing_probs[4]), xytext=(3.8, mc_wing_probs[4] + 8.0), arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6), fontsize=9.0, fontweight="bold", color="#0d47a1")
    ax_wings.annotate("Arrival Dispersion\n& Calibration ($k$)", xy=(8, mc_wing_probs[7]), xytext=(6.2, mc_wing_probs[7] + 8.0), arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6), fontsize=9.0, fontweight="bold", color="#1b5e20")

    if out_png:
        plt.savefig(out_png, dpi=300)
        print(f"[✓] Saved {out_png}")
    return fig


def main():
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"
    cache_path = "data/tb_mufi_all_planes_cache.npz"

    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        data_dict = np.load(cache_path, allow_pickle=True)
        data_dict = {k: data_dict[k] for k in data_dict.files}
    else:
        data_dict = extract_data_and_mc(n_data_entries=25000, n_mc_events=600, cache_path=cache_path)

    png_us = os.path.join(plot_dir, "tb_all_us_planes_incremental_factors.png")
    plot_all_us_planes(data_dict, png_us)

    png_5p = os.path.join(plot_dir, "tb_us_planes_5panel_transverse_profiles.png")
    plot_5panel_us_transverse_profiles(data_dict, png_5p)

    png_ds = os.path.join(plot_dir, "tb_ds_planes_incremental_factors.png")
    plot_ds_planes(data_dict, png_ds)

    if os.path.exists(art_dir):
        for png in [png_us, png_5p, png_ds]:
            os.system(f"cp {png} {os.path.join(art_dir, os.path.basename(png))}")
    print("[✓] All plots generated and synchronized.")


if __name__ == "__main__":
    if "IPython" not in sys.modules and os.environ.get("DISPLAY") is None:
        matplotlib.use("Agg")
    main()
