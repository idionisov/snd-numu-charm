#!/usr/bin/env python3
"""
scripts/qdc_calib/extract_multienergy_tb.py
-------------------------------------------
Extracts 8-stage incremental reduction factors across all available testbeam pion energies:
  - 100 GeV pi+ (Run 100243)
  - 140 GeV pi+ (Run 100242)
  - 180 GeV pi+ (Run 100671)
  - 240 GeV pi- (Run 100265)
  - 300 GeV pi- (Run 100266)

Generates publication-quality 4-panel plots for each plane and each pion energy:
  - Top-Left: Transverse Profile with MC/Data ratio subpanel
  - Top-Right: Multiplicity Distribution Evolution
  - Bottom-Left: Mean Fired Bars vs Pion Energy (replacing the horizontal bar chart),
                 with the current beam energy highlighted
  - Bottom-Right: Suppression of Peripheral Wing Splash across S1 to S8 vs Data
Displays both the Plane Number and Pion Beam Energy of the used data.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import multiprocessing as mp
from typing import Dict, List, Tuple, Any

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

from snd import load_trident_libraries
load_trident_libraries(_repo_root)

import scripts.qdc_calib.min_pe as mpe

# Parameter sources: August 2023 '3 walls, ALL IN' testbeam runs from CERN SPS H8 testbeam logbook:
# Physics justification for incremental reduction stages:
# Stage 1: Raw Scintillation yield (LIGHT_YIELD = 10000 ph/MeV from EJ-200 datasheet)
# Stage 2: Geometric area coverage fraction f_area (SiPM active area / bar endcap area)
# Stage 3: Optical acceptance cone fraction f_cone (total internal reflection trapping from Snell's law)
# Stage 4: Bulk technical attenuation S_att and Fresnel interface transmission f_fresnel
# Stage 5: Birks quenching f_birks (hadronic recoils and high dE/dx stopping power saturation)
# Stage 6: Cut edge degradation and dead skin layer f_edge (machining micro-roughness and wrapping loss)
# Stage 7: Photon arrival time dispersion attenuation f_disp (waveform spreading across shower lateral width)
# Stage 8: SiPM channel-specific empirical calibration factor k (optical coupling, alignment, ASIC baseline)
ENERGY_CONFIGS = {
    100: {
        "energy": 100, "pid": 211, "name": r"100 GeV $\pi^+$", "label": "100GeV",
        "data_file": "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100677/sndsw_raw-0000.root",
        "mc_file": "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/100GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root",
        "run_num": 100677,
    },
    140: {
        "energy": 140, "pid": 211, "name": r"140 GeV $\pi^+$", "label": "140GeV",
        "data_file": "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100673/sndsw_raw-0000.root",
        "mc_file": "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/140GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root",
        "run_num": 100673,
    },
    180: {
        "energy": 180, "pid": 211, "name": r"180 GeV $\pi^+$", "label": "180GeV",
        "data_file": "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100671/sndsw_raw-0000.root",
        "mc_file": "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/180GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root",
        "run_num": 100671,
    },
    240: {
        "energy": 240, "pid": -211, "name": r"240 GeV $\pi^-$", "label": "240GeV",
        "data_file": "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100648/sndsw_raw-0000.root",
        "mc_file": "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/240GeV_-211/0/sndLHC.PG_-211-TGeant4_MCEB_dig.root",
        "run_num": 100648,
    },
    300: {
        "energy": 300, "pid": -211, "name": r"300 GeV $\pi^-$", "label": "300GeV",
        "data_file": "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100645/sndsw_raw-0000.root",
        "mc_file": "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/300GeV_-211/0/sndLHC.PG_-211-TGeant4_MCEB_dig.root",
        "run_num": 100645,
    },
}

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


def extract_single_energy(cfg: dict, max_data: int = 15000, max_mc: int = 350) -> dict:
    """Extract hits for single energy data and MC."""
    energy = cfg["energy"]
    print(f"[*] Starting extraction for {cfg['name']} (Run {cfg['run_num']})...")
    calc = mpe.MinPE(year=2023, is_testbeam=True, include_edge_effect=True, include_dispersion=True)
    th_pe = calc.get_sipm_threshold_pe()

    fd = ROOT.TFile.Open(cfg["data_file"])
    td = fd.Get("rawConv")
    td.SetBranchStatus("*", 0)
    td.SetBranchStatus("Digi_ScifiHits*", 1)
    td.SetBranchStatus("Digi_MuFilterHits*", 1)

    d_us_bars = [np.zeros(10, dtype=float) for _ in range(5)]
    d_us_mult = [[] for _ in range(5)]
    d_us_tot_mult = []
    d_dsh_bars = np.zeros(60, dtype=float)
    d_ds_tot_mult = []

    n_showers_d = 0
    nentries_d = min(max_data, td.GetEntries())
    for i in range(nentries_d):
        td.GetEntry(i)
        if len(td.Digi_ScifiHits) < 100:
            continue
        n_showers_d += 1
        fired_us_p = [set() for _ in range(5)]
        fired_dsh = set()
        fired_dsv = set()

        for mh in td.Digi_MuFilterHits:
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
        for b in fired_dsh:
            if 0 <= b < 60:
                d_dsh_bars[b] += 1
        d_ds_tot_mult.append(len(fired_dsh) + len(fired_dsv))

    fd.Close()

    fm = ROOT.TFile.Open(cfg["mc_file"])
    tm = fm.Get("cbmsim")
    tm.SetBranchStatus("*", 0)
    tm.SetBranchStatus("MuFilterPoint*", 1)

    mc_us_bars = [[np.zeros(10, dtype=float) for _ in range(8)] for _ in range(5)]
    mc_us_mult = [{p: [] for p in range(5)} for _ in range(8)]
    mc_us_tot_mult = [[] for _ in range(8)]
    mc_dsh_bars = [np.zeros(60, dtype=float) for _ in range(8)]
    mc_ds_tot_mult = [[] for _ in range(8)]

    actual_mc = min(max_mc, tm.GetEntries())
    for iev in range(actual_mc):
        tm.GetEntry(iev)
        by_bar = calc.group_mcpoints_by_bar(tm.MuFilterPoint)
        fired_by_stage = [set() for _ in range(8)]

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
                        fired_by_stage[s_idx].add(did)
                break

        for s_idx in range(8):
            fired = fired_by_stage[s_idx]
            us_f = [d for d in fired if d // 10000 == 2]
            ds_f = [d for d in fired if d // 10000 == 3]
            mc_us_tot_mult[s_idx].append(len(us_f))
            mc_ds_tot_mult[s_idx].append(len(ds_f))

            for p in range(5):
                p_bars = [(d % 1000) for d in us_f if (d % 10000) // 1000 == p]
                mc_us_mult[s_idx][p].append(len(p_bars))
                for b in p_bars:
                    if 0 <= b < 10:
                        mc_us_bars[p][s_idx][b] += 1

            for d in ds_f:
                b = d % 1000
                if b < 60:
                    mc_dsh_bars[s_idx][b] += 1

    fm.Close()
    print(f"[✓] Finished {cfg['name']}: {n_showers_d} Data showers, {actual_mc} MC showers.")
    return {
        "energy": energy,
        "name": cfg["name"],
        "run_num": cfg["run_num"],
        "n_showers_d": n_showers_d,
        "actual_mc": actual_mc,
        "d_us_bars": d_us_bars,
        "d_us_mult": d_us_mult,
        "d_us_tot_mult": d_us_tot_mult,
        "d_dsh_bars": d_dsh_bars,
        "d_ds_tot_mult": d_ds_tot_mult,
        "mc_us_bars": mc_us_bars,
        "mc_us_mult": mc_us_mult,
        "mc_us_tot_mult": mc_us_tot_mult,
        "mc_dsh_bars": mc_dsh_bars,
        "mc_ds_tot_mult": mc_ds_tot_mult,
    }


def _worker_task(args):
    cfg, max_d, max_m = args
    return extract_single_energy(cfg, max_data=max_d, max_mc=max_m)


def extract_all_energies(max_data: int = 15000, max_mc: int = 350, n_workers: int = 5, cache_path: str = "data/tb_multienergy_cache.npz") -> dict:
    if os.path.exists(cache_path):
        print(f"Loading cached multi-energy data from {cache_path}...")
        raw = np.load(cache_path, allow_pickle=True)
        return {int(k): raw[k].item() for k in raw.files}

    print(f"[*] Extracting all 5 pion energies in parallel using {n_workers} workers...")
    args_list = [(ENERGY_CONFIGS[e], max_data, max_mc) for e in sorted(ENERGY_CONFIGS.keys())]

    with mp.Pool(processes=min(n_workers, len(args_list))) as pool:
        results = pool.map(_worker_task, args_list)

    all_res = {r["energy"]: r for r in results}
    save_dict = {str(k): v for k, v in all_res.items()}
    np.savez_compressed(cache_path, **save_dict)
    print(f"[✓] Saved multi-energy cache to {cache_path}")
    return all_res


def plot_single_plane_evolution(
    plane_idx: int,
    current_energy: int,
    all_energy_data: dict,
    out_png: str,
):
    """
    4-panel evolution plot for a single detector plane at a specific beam energy:
      - Top-Left: Transverse Profile with MC/Data Ratio subpanel
      - Top-Right: Multiplicity Distribution Evolution
      - Bottom-Left: Mean Fired Bars vs Pion Beam Energy (100 to 300 GeV) with current energy highlighted
      - Bottom-Right: Suppression of Low-Energy Peripheral Splash vs Stage
    """
    data_dict = all_energy_data[current_energy]
    cfg = ENERGY_CONFIGS[current_energy]
    energy_name = cfg["name"]
    run_num = cfg["run_num"]
    is_ds = (plane_idx == 5)

    plane_title = f"US Plane {plane_idx}" if not is_ds else "Downstream (DS) MuFilter"
    plane_coord = "US Bar Index (0=Bottom, 9=Top)" if not is_ds else "DS-H Bar Index (1 cm pitch)"

    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])

    cmap = plt.cm.turbo
    stage_colors = [cmap(val) for val in np.linspace(0.08, 0.95, 8)]

    n_d = data_dict["n_showers_d"]
    n_mc = data_dict["actual_mc"]

    gs_left = gs[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax_top = fig.add_subplot(gs_left[0])
    ax_bot = fig.add_subplot(gs_left[1], sharex=ax_top)

    if not is_ds:
        x_bars = np.arange(10)
        d_bars_raw = data_dict["d_us_bars"][plane_idx]
        d_mult_vals = data_dict["d_us_mult"][plane_idx]
    else:
        x_bars = np.arange(15, 46)
        d_bars_raw = data_dict["d_dsh_bars"][x_bars]
        d_mult_vals = data_dict["d_ds_tot_mult"]

    d_prob = (d_bars_raw / n_d) * 100.0 if not is_ds else (d_bars_raw / n_d) * 100.0
    d_err = np.sqrt(d_prob * (100.0 - d_prob) / n_d)
    mean_d = float(np.mean(d_mult_vals))

    mc_means_inc = []
    for s_idx in range(8):
        c = stage_colors[s_idx]
        lw = 2.4 if s_idx in (0, 4, 7) else 1.6
        ls = "-" if s_idx == 7 else ("--" if s_idx == 0 else "-.")

        if not is_ds:
            mc_b_raw = data_dict["mc_us_bars"][plane_idx][s_idx]
            mc_mult_s = data_dict["mc_us_mult"][s_idx][plane_idx]
        else:
            mc_b_raw = data_dict["mc_dsh_bars"][s_idx][x_bars]
            mc_mult_s = data_dict["mc_ds_tot_mult"][s_idx]

        mc_prob = (mc_b_raw / n_mc) * 100.0
        m_val = float(np.mean(mc_mult_s))
        mc_means_inc.append(m_val)

        if s_idx == 7:
            mc_err = np.sqrt(mc_prob * (100.0 - mc_prob) / n_mc)
            ax_top.errorbar(
                x_bars, mc_prob, yerr=mc_err, marker="s", markersize=5.0, lw=lw, ls=ls, color=c,
                ecolor=c, elinewidth=1.6, capsize=3.0,
                label=f"{STAGE_NAMES[s_idx]} ($\mu$={m_val:.2f})", zorder=6
            )
            ratio_s8 = np.where(d_prob > 0.5, mc_prob / d_prob, np.nan)
            r_err = ratio_s8 * np.sqrt((mc_err / np.maximum(mc_prob, 1e-3))**2 + (d_err / np.maximum(d_prob, 1e-3))**2)
            ax_bot.errorbar(x_bars, ratio_s8, yerr=r_err, marker="o", markersize=4.0, lw=1.8, ls=ls, color=c, ecolor=c, elinewidth=1.4, capsize=2.5, zorder=6)
        else:
            ax_top.plot(x_bars, mc_prob, marker="o", markersize=4.0, lw=lw, ls=ls, color=c, label=f"{STAGE_NAMES[s_idx]} ($\mu$={m_val:.2f})", zorder=4)
            ratio = np.where(d_prob > 0.5, mc_prob / d_prob, np.nan)
            ax_bot.plot(x_bars, ratio, marker="o", markersize=3.5, lw=1.5, ls=ls, color=c)

    ax_top.errorbar(
        x_bars, d_prob, yerr=d_err, fmt="o", color="#111111", ecolor="#111111", elinewidth=2.0,
        capsize=3.5, markersize=6.5, label=f"Data ({energy_name}, Run {run_num}, Mean={mean_d:.2f})", zorder=10
    )
    ax_top.set_ylabel("Firing Probability [%]", fontsize=11, fontweight="bold")
    ax_top.set_title(f"Transverse Profile: {plane_title} — {energy_name} (Run {run_num})", fontsize=12, fontweight="bold")
    ax_top.grid(True, linestyle=":", alpha=0.5)
    ax_top.set_ylim(0, 115 if not is_ds else 45)
    ax_top.legend(fontsize=8.0, loc="upper right", framealpha=0.92, ncol=2)

    ax_bot.axhline(1.0, color="#111111", lw=1.5, linestyle="--")
    ax_bot.set_ylabel("MC / Data", fontsize=10, fontweight="bold")
    ax_bot.set_xlabel(plane_coord, fontsize=11, fontweight="bold")
    ax_bot.set_ylim(0.2, 3.8)
    if not is_ds:
        ax_bot.set_xticks(x_bars)
    ax_bot.grid(True, linestyle=":", alpha=0.5)

    ax_mult = fig.add_subplot(gs[0, 1])
    bins_mult = np.arange(-0.5, 11.5, 1.0) if not is_ds else np.arange(-0.5, 20.5, 1.0)
    centers_mult = 0.5 * (bins_mult[:-1] + bins_mult[1:])

    for s_idx in range(8):
        c = stage_colors[s_idx]
        lw = 2.4 if s_idx in (0, 4, 7) else 1.5
        ls = "-" if s_idx == 7 else ("--" if s_idx == 0 else "-.")
        m_arr = data_dict["mc_us_mult"][s_idx][plane_idx] if not is_ds else data_dict["mc_ds_tot_mult"][s_idx]
        ax_mult.hist(m_arr, bins=bins_mult, density=True, histtype="stepfilled", color=c, alpha=0.06)
        ax_mult.hist(m_arr, bins=bins_mult, density=True, histtype="step", lw=lw, ls=ls, color=c, label=f"{STAGE_NAMES[s_idx]} ($\mu$={mc_means_inc[s_idx]:.2f})")

    m_s8 = data_dict["mc_us_mult"][7][plane_idx] if not is_ds else data_dict["mc_ds_tot_mult"][7]
    c_s8, _ = np.histogram(m_s8, bins=bins_mult)
    prob_s8 = c_s8 / n_mc
    err_s8 = np.sqrt(c_s8) / n_mc
    ax_mult.errorbar(centers_mult, prob_s8, yerr=err_s8, fmt="none", ecolor=stage_colors[7], elinewidth=1.6, capsize=2.0, zorder=7)

    c_d, _ = np.histogram(d_mult_vals, bins=bins_mult)
    prob_d = c_d / n_d
    err_d = np.sqrt(c_d) / n_d
    ax_mult.errorbar(centers_mult, prob_d, yerr=err_d, fmt="o", color="#111111", ecolor="#111111", elinewidth=2.0, capsize=3.0, markersize=5.5, label=f"Data ({energy_name}, $\mu$={mean_d:.2f})", zorder=10)

    ax_mult.set_xlabel(f"Number of Fired Bars in {plane_title}", fontsize=11, fontweight="bold")
    ax_mult.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax_mult.set_title(f"Multiplicity Distribution Evolution: {plane_title} ({energy_name})", fontsize=12, fontweight="bold")
    ax_mult.grid(True, linestyle=":", alpha=0.5)
    ax_mult.legend(fontsize=8.0, loc="upper right", framealpha=0.92)

    gs_bot_left = gs[1, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax_e_top = fig.add_subplot(gs_bot_left[0])
    ax_e_bot = fig.add_subplot(gs_bot_left[1], sharex=ax_e_top)

    energies = sorted(all_energy_data.keys())
    data_e_means = []
    data_e_errs = []
    mc_raw_e_means = []
    mc_cal_e_means = []
    mc_cal_e_errs = []

    for e in energies:
        ed = all_energy_data[e]
        nd_e = ed["n_showers_d"]
        nm_e = ed["actual_mc"]
        vals_d = ed["d_us_mult"][plane_idx] if not is_ds else ed["d_ds_tot_mult"]
        vals_raw = ed["mc_us_mult"][0][plane_idx] if not is_ds else ed["mc_ds_tot_mult"][0]
        vals_cal = ed["mc_us_mult"][7][plane_idx] if not is_ds else ed["mc_ds_tot_mult"][7]

        data_e_means.append(float(np.mean(vals_d)))
        data_e_errs.append(float(np.std(vals_d) / np.sqrt(nd_e)))
        mc_raw_e_means.append(float(np.mean(vals_raw)))
        mc_cal_e_means.append(float(np.mean(vals_cal)))
        mc_cal_e_errs.append(float(np.std(vals_cal) / np.sqrt(nm_e)))

    data_e_means = np.array(data_e_means)
    data_e_errs = np.array(data_e_errs)
    mc_raw_e_means = np.array(mc_raw_e_means)
    mc_cal_e_means = np.array(mc_cal_e_means)
    mc_cal_e_errs = np.array(mc_cal_e_errs)

    ax_e_top.plot(energies, mc_raw_e_means, marker="s", markersize=6.0, lw=2.0, ls=":", color="#1976d2", label="Raw MC (Stage 1)")
    ax_e_top.errorbar(energies, mc_cal_e_means, yerr=mc_cal_e_errs, marker="o", markersize=6.5, lw=2.4, ls="-", color="#d32f2f", ecolor="#d32f2f", elinewidth=1.8, capsize=3.0, label="Calibrated MC (Stage 8, Final)", zorder=6)
    ax_e_top.errorbar(energies, data_e_means, yerr=data_e_errs, fmt="o", color="#111111", ecolor="#111111", elinewidth=2.2, capsize=3.5, markersize=7.0, label="Recorded Testbeam Data", zorder=10)

    cur_d_idx = energies.index(current_energy)
    ax_e_top.plot([current_energy], [data_e_means[cur_d_idx]], marker="*", markersize=16.0, color="#ffb300", markeredgecolor="#111111", markeredgewidth=1.2, label=f"Current: {energy_name}", zorder=12)
    ax_e_top.axvline(current_energy, color="#ffb300", linestyle="--", lw=1.5, alpha=0.7)

    ax_e_top.set_ylabel(f"Mean Fired Bars", fontsize=11, fontweight="bold")
    ax_e_top.set_title(f"Energy Scaling in {plane_title}: 100 to 300 GeV", fontsize=12, fontweight="bold")
    ax_e_top.grid(True, linestyle=":", alpha=0.5)
    ax_e_top.legend(fontsize=8.5, loc="lower right", framealpha=0.92)

    ratio_raw_e = mc_raw_e_means / data_e_means
    ratio_cal_e = mc_cal_e_means / data_e_means
    r_err_e = ratio_cal_e * np.sqrt((mc_cal_e_errs / mc_cal_e_means)**2 + (data_e_errs / data_e_means)**2)

    ax_e_bot.axhline(1.0, color="#111111", lw=1.5, linestyle="--")
    ax_e_bot.plot(energies, ratio_raw_e, marker="s", markersize=4.0, color="#1976d2", ls=":", lw=1.5)
    ax_e_bot.errorbar(energies, ratio_cal_e, yerr=r_err_e, marker="o", markersize=5.0, color="#d32f2f", ecolor="#d32f2f", elinewidth=1.6, capsize=2.5, ls="-", lw=2.0)
    ax_e_bot.axvline(current_energy, color="#ffb300", linestyle="--", lw=1.5, alpha=0.7)

    ax_e_bot.set_ylabel("MC / Data", fontsize=10, fontweight="bold")
    ax_e_bot.set_xlabel("Pion Beam Energy [GeV]", fontsize=11, fontweight="bold")
    ax_e_bot.set_xticks(energies)
    ax_e_bot.set_xticklabels(["100", "140", "180", "240", "300"], fontsize=10, fontweight="bold")
    ax_e_bot.set_ylim(0.5, 2.5)
    ax_e_bot.grid(True, linestyle=":", alpha=0.5)

    ax_wings = fig.add_subplot(gs[1, 1])
    stages_x = np.arange(1, 9)

    if not is_ds:
        d_wing_prob = float(np.mean([(data_dict["d_us_bars"][plane_idx][b] / n_d) * 100.0 for b in (0, 1, 8, 9)]))
        mc_wing_probs = []
        mc_wing_errs = []
        for s in range(8):
            vals = [(data_dict["mc_us_bars"][plane_idx][s][b] / n_mc) * 100.0 for b in (0, 1, 8, 9)]
            mc_wing_probs.append(float(np.mean(vals)))
            mc_wing_errs.append(float(np.std(vals) / np.sqrt(len(vals))))
        wing_lbl = f"{plane_title} Wings (Bars 0, 1, 8, 9)"
    else:
        wing_indices = np.concatenate([np.arange(0, 15), np.arange(45, 60)])
        d_wing_prob = float(np.mean((data_dict["d_dsh_bars"][wing_indices] / n_d) * 100.0))
        mc_wing_probs = []
        mc_wing_errs = []
        for s in range(8):
            vals = (data_dict["mc_dsh_bars"][s][wing_indices] / n_mc) * 100.0
            mc_wing_probs.append(float(np.mean(vals)))
            mc_wing_errs.append(float(np.std(vals) / np.sqrt(len(vals))))
        wing_lbl = "DS Wings (Bars <15 & >45)"

    ax_wings.plot(stages_x, mc_wing_probs, marker="s", markersize=6.5, lw=2.4, color="#d32f2f", label=f"Simulation {wing_lbl}", zorder=5)
    ax_wings.errorbar(stages_x, mc_wing_probs, yerr=mc_wing_errs, fmt="none", ecolor="#d32f2f", elinewidth=1.6, capsize=3.0, zorder=6)
    ax_wings.axhline(d_wing_prob, color="#111111", linestyle="--", lw=2.0, label=f"Data Baseline ({d_wing_prob:.1f}%)")
    ax_wings.fill_between([0.5, 8.5], 0, d_wing_prob, color="#424242", alpha=0.15)

    ax_wings.set_xticks(stages_x)
    ax_wings.set_xticklabels([f"S{i}" for i in range(1, 9)], fontsize=10, fontweight="bold")
    ax_wings.set_xlabel("Incremental Reduction Stage", fontsize=11, fontweight="bold")
    ax_wings.set_ylabel("Peripheral Wing Firing Probability [%]", fontsize=11, fontweight="bold")
    ax_wings.set_title(f"Peripheral Shower Leakage Suppression: {plane_title} ({energy_name})", fontsize=12, fontweight="bold")
    ax_wings.set_ylim(0, 100 if not is_ds else 50)
    ax_wings.set_xlim(0.5, 8.5)
    ax_wings.grid(True, linestyle=":", alpha=0.5)
    ax_wings.legend(fontsize=9.0, loc="upper right")

    ax_wings.annotate("Birks Quenching\n(Hadronic Recoils)", xy=(5, mc_wing_probs[4]), xytext=(3.8, mc_wing_probs[4] + 16.0), arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6), fontsize=9.0, fontweight="bold", color="#0d47a1")
    ax_wings.annotate("Arrival Dispersion\n& Calibration ($k$)", xy=(8, mc_wing_probs[7]), xytext=(6.2, mc_wing_probs[7] + 16.0), arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6), fontsize=9.0, fontweight="bold", color="#1b5e20")

    plt.suptitle(f"SND@LHC MuFilter Digitization Validation — {plane_title} | Beam Energy: {energy_name} (Run {run_num})", fontsize=14, fontweight="bold", y=1.01)

    if out_png:
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"[✓] Generated: {out_png}")
    plt.show()
    plt.close()


def main():
    plot_dir = "plots/multienergy"
    os.makedirs(plot_dir, exist_ok=True)
    art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"
    cache_path = "data/tb_multienergy_cache.npz"

    all_energy_data = extract_all_energies(max_data=15000, max_mc=350, n_workers=5, cache_path=cache_path)

    generated_plots = []
    for energy in sorted(all_energy_data.keys()):
        for plane_idx in range(6):
            p_name = f"us_plane_{plane_idx}" if plane_idx < 5 else "ds_subsystem"
            fname = f"tb_{p_name}_{energy}gev_evolution.png"
            out_png = os.path.join(plot_dir, fname)
            plot_single_plane_evolution(plane_idx, energy, all_energy_data, out_png)
            generated_plots.append(out_png)

            if os.path.exists(art_dir):
                os.system(f"cp {out_png} {os.path.join(art_dir, fname)}")

    print(f"\n[✓] Successfully generated {len(generated_plots)} multi-energy evolution plots across all planes.")


if __name__ == "__main__":
    main()
