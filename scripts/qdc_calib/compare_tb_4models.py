#!/usr/bin/env python3
"""
scripts/qdc_calib/compare_tb_4models.py
---------------------------------------
Side-by-side comparison of Testbeam H8 Data vs Simulations across 4 threshold models:
  1. H8 Testbeam In-time Data (conditioned on SciFi shower: N_SciFi >= 100, |t| <= 15 ns)
  2. Simulation: Raw (E_th = 0, every deposited hit counts)
  3. Simulation: Theoretical threshold (k = 1.0)
  4. Simulation: Theoretical + channel-specific photoelectron reduction factor (k from CSV)
  5. Simulation: Simple step activation (1.0 MeV for DS, 1.5 MeV for US/Veto)

Apples-to-Apples Event Selection:
  - Confirmed target shower: N_SciFi >= 100 in both Data and Simulation
  - Beam spot alignment: Primary beam entering in central bars (US Plane 0 bars 3, 4, 5)

Evaluates:
  - Collective subsystems: Total US Hits, Total DS Hits
  - All individual planes:
    * US Planes 0, 1, 2, 3, 4
    * DS Stations: DS1-H, DS1-V, DS2-H, DS2-V, DS3-H, DS3-V, DS4-V
  - Multi-energy comparison across 100, 140, 180, 240, 300 GeV

Saves canvases and histograms to:
  output/tb_vs_sim_4models.root
and generates publication-quality PNG figures in plots/.
"""

from __future__ import annotations

import os
import sys
import pickle
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional

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

# Observable histogram specifications: (title, xlabel, nbins, xmin, xmax)
TB_HIST_SPECS = {
    # Subsystems
    "us_total":   ("US Total Hits", "N_{hits}^{US}", 46, 4.5, 50.5),
    "ds_total":   ("DS Total Hits", "N_{hits}^{DS}", 26, -0.5, 25.5),
    # US Planes
    "us_pl0":     ("US Plane 0 Hits", "N_{hits}^{US 0}", 11, -0.5, 10.5),
    "us_pl1":     ("US Plane 1 Hits", "N_{hits}^{US 1}", 11, -0.5, 10.5),
    "us_pl2":     ("US Plane 2 Hits", "N_{hits}^{US 2}", 11, -0.5, 10.5),
    "us_pl3":     ("US Plane 3 Hits", "N_{hits}^{US 3}", 11, -0.5, 10.5),
    "us_pl4":     ("US Plane 4 Hits", "N_{hits}^{US 4}", 11, -0.5, 10.5),
    # DS Stations
    "ds_h_pl0":   ("DS Station 1 H Hits", "N_{hits}^{DS1-H}", 11, -0.5, 10.5),
    "ds_v_pl0":   ("DS Station 1 V Hits", "N_{hits}^{DS1-V}", 11, -0.5, 10.5),
    "ds_h_pl1":   ("DS Station 2 H Hits", "N_{hits}^{DS2-H}", 11, -0.5, 10.5),
    "ds_v_pl1":   ("DS Station 2 V Hits", "N_{hits}^{DS2-V}", 11, -0.5, 10.5),
    "ds_h_pl2":   ("DS Station 3 H Hits", "N_{hits}^{DS3-H}", 11, -0.5, 10.5),
    "ds_v_pl2":   ("DS Station 3 V Hits", "N_{hits}^{DS3-V}", 11, -0.5, 10.5),
    "ds_v_pl3":   ("DS Station 4 V Hits", "N_{hits}^{DS4-V}", 11, -0.5, 10.5),
}

# Precompute physical threshold maps
def precompute_threshold_maps() -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Returns (th_theo_kev, th_calib_kev, th_step_kev) for all MuFilter bars in keV,
    incorporating 2023 H8 testbeam configuration:
    - is_testbeam=True (alternating gel: US0, US2, US4 Left; US1, US3 Right)
    - DS: DS1-H (both sides) and DS1-V (Top only). DS2..4 did not exist in testbeam.
    - Testbeam SiPM reduction factors from muon run calibration.
    """
    calc_theo = mpe.MinPE(year=2023, is_testbeam=True)
    calc_theo._k_factors = {} # all k = 1.0

    calc_calib = mpe.MinPE(year=2023, is_testbeam=True) # k from testbeam CSV

    th_theo: Dict[int, float] = {}
    th_calib: Dict[int, float] = {}
    th_step: Dict[int, float] = {}

    # US (subsystem 2): 5 planes, 10 bars
    for pl in range(5):
        for b in range(10):
            did = 20000 + pl * 1000 + b
            p = {"detector_id": did, "x": -37.7, "y": 0.0, "z": 0.0, "eloss": 1.0}
            th_theo[did] = float(calc_theo.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_calib[did] = float(calc_calib.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_step[did] = 1500.0 # 1.5 MeV in keV

    # DS (subsystem 3): In 2023 H8 testbeam, ONLY Station 1 existed (plane 0)
    for b in range(60):
        did = 30000 + b
        p = {"detector_id": did, "x": -37.7, "y": 0.0, "z": 0.0, "eloss": 1.0}
        th_theo[did] = float(calc_theo.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
        th_calib[did] = float(calc_calib.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
        th_step[did] = 1000.0 # 1.0 MeV in keV
    for b in range(60, 120):
        did = 30000 + b
        p = {"detector_id": did, "x": 0.0, "y": 20.0, "z": 0.0, "eloss": 1.0}
        th_theo[did] = float(calc_theo.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
        th_calib[did] = float(calc_calib.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
        th_step[did] = 1000.0 # 1.0 MeV in keV

    # Stations 1..3 did not exist in testbeam
    for pl in range(1, 4):
        for b in range(120):
            did = 30000 + pl * 1000 + b
            th_theo[did] = np.inf
            th_calib[did] = np.inf
            th_step[did] = np.inf

    return th_theo, th_calib, th_step


def evaluate_tb_energy(
    energy: int,
    data_events: List[Dict[str, Any]],
    mc_events: List[Dict[str, Any]],
    th_theo: Dict[int, float],
    th_calib: Dict[int, float],
    th_step: Dict[int, float],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """
    Evaluates data and the 4 MC models for a given beam energy.
    """
    # 1. Format Data
    d_dict = {k: [] for k in TB_HIST_SPECS}
    for ev in data_events:
        d_dict["us_total"].append(ev["us_total"])
        d_dict["ds_total"].append(ev["ds_total"])
        for p in range(5):
            d_dict[f"us_pl{p}"].append(ev["us_planes"][p])
        d_dict["ds_h_pl0"].append(ev["ds_planes"]["DS1_H"])
        d_dict["ds_v_pl0"].append(ev["ds_planes"]["DS1_V"])
        d_dict["ds_h_pl1"].append(ev["ds_planes"]["DS2_H"])
        d_dict["ds_v_pl1"].append(ev["ds_planes"]["DS2_V"])
        d_dict["ds_h_pl2"].append(ev["ds_planes"]["DS3_H"])
        d_dict["ds_v_pl2"].append(ev["ds_planes"]["DS3_V"])
        d_dict["ds_v_pl3"].append(ev["ds_planes"]["DS4_V"])

    data_arrays = {k: np.array(d_dict[k], dtype=np.int32) for k in TB_HIST_SPECS}

    # 2. Evaluate 4 Simulation Models on MC
    models = ["raw", "theo", "calib", "step"]
    mc_dict: Dict[str, Dict[str, List[int]]] = {m: {k: [] for k in TB_HIST_SPECS} for m in models}

    for ev in mc_events:
        bar_edep = ev["bar_edep"]

        recs = {m: {k: 0 for k in TB_HIST_SPECS} for m in models}

        for did, edep_kev in bar_edep.items():
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000

            # Determine key
            k_pl = ""
            if sub == 2:
                if 0 <= pl < 5:
                    k_pl = f"us_pl{pl}"
            elif sub == 3:
                if pl == 0: k_pl = "ds_h_pl0" if bar < 60 else "ds_v_pl0"
                elif pl == 1: k_pl = "ds_h_pl1" if bar < 60 else "ds_v_pl1"
                elif pl == 2: k_pl = "ds_h_pl2" if bar < 60 else "ds_v_pl2"
                elif pl == 3: k_pl = "ds_v_pl3"

            # Model 1: Raw (E_th = 0)
            if edep_kev > 0.0:
                if sub == 2:
                    recs["raw"]["us_total"] += 1
                elif sub == 3 and pl == 0:
                    recs["raw"]["ds_total"] += 1
                if k_pl: recs["raw"][k_pl] += 1

            # Model 2: Theoretical (k=1.0)
            if edep_kev > th_theo.get(did, np.inf):
                if sub == 2:
                    recs["theo"]["us_total"] += 1
                elif sub == 3 and pl == 0:
                    recs["theo"]["ds_total"] += 1
                if k_pl: recs["theo"][k_pl] += 1

            # Model 3: Theoretical + k (Calibrated)
            if edep_kev > th_calib.get(did, np.inf):
                if sub == 2:
                    recs["calib"]["us_total"] += 1
                elif sub == 3 and pl == 0:
                    recs["calib"]["ds_total"] += 1
                if k_pl: recs["calib"][k_pl] += 1

            # Model 4: Simple Step (1.0 MeV DS, 1.5 MeV US)
            if edep_kev > th_step.get(did, np.inf):
                if sub == 2:
                    recs["step"]["us_total"] += 1
                elif sub == 3 and pl == 0:
                    recs["step"]["ds_total"] += 1
                if k_pl: recs["step"][k_pl] += 1

        for m in models:
            for k in TB_HIST_SPECS:
                mc_dict[m][k].append(recs[m][k])

    mc_arrays = {m: {k: np.array(mc_dict[m][k], dtype=np.int32) for k in TB_HIST_SPECS} for m in models}
    return data_arrays, mc_arrays


def plot_tb_4models(
    data_dict: Dict[str, np.ndarray],
    mc_models: Dict[str, Dict[str, np.ndarray]],
    keys: List[str],
    fig_title: str,
    out_png: str,
    ncols: int = 3,
):
    """Plot Testbeam comparison normalized to unit area."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.0
    nrows = (len(keys) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows * 2, ncols,
        figsize=(5.8 * ncols, 4.0 * nrows),
        gridspec_kw={"height_ratios": [3, 1.2] * nrows},
        constrained_layout=True,
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif nrows == 1:
        axes = np.array([axes[0, :], axes[1, :]])
    elif ncols == 1:
        axes = axes.reshape(nrows * 2, 1)

    model_styles = {
        "raw":   {"label": "Raw (Eth=0)", "color": "#1f77b4", "ls": ":", "lw": 1.6, "marker": "s"},
        "theo":  {"label": "Theoretical (k=1.0)", "color": "#9467bd", "ls": "--", "lw": 1.8, "marker": "D"},
        "calib": {"label": "Theoretical + k(SiPM)", "color": "#d62728", "ls": "-", "lw": 2.0, "marker": "o"},
        "step":  {"label": "Simple Step (1 / 1.5 MeV)", "color": "#2ca02c", "ls": "-.", "lw": 1.8, "marker": "^"},
    }

    for idx, key in enumerate(keys):
        row = (idx // ncols) * 2
        col = idx % ncols
        ax_main = axes[row, col]
        ax_ratio = axes[row + 1, col]

        title, xlabel, nbins, xmin, xmax = TB_HIST_SPECS[key]
        bins = np.linspace(xmin, xmax, nbins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = bins[1] - bins[0]

        # 1. Data (normalized density)
        data_vals = data_dict[key]
        n_data = len(data_vals)
        h_data, _ = np.histogram(data_vals, bins=bins)
        norm_d = n_data * bin_width if n_data > 0 else 1.0
        y_data = h_data / norm_d
        y_data_err = np.sqrt(h_data) / norm_d

        # 2. Simulation Models (normalized density)
        y_sim = {}
        y_sim_err = {}
        for m in ["raw", "theo", "calib", "step"]:
            vals = mc_models[m][key]
            n_m = len(vals)
            h_m, _ = np.histogram(vals, bins=bins)
            norm_m = n_m * bin_width if n_m > 0 else 1.0
            y_sim[m] = h_m / norm_m
            y_sim_err[m] = np.sqrt(h_m) / norm_m

            st = model_styles[m]
            ax_main.step(
                bins, np.append(y_sim[m], y_sim[m][-1]), where="post",
                color=st["color"], lw=st["lw"], ls=st["ls"], label=f"{st['label']} (μ={np.mean(vals):.1f})"
            )
            if m == "calib":
                ax_main.fill_between(bins, np.append(y_sim[m], y_sim[m][-1]), step="post", color="#ff7f7f", alpha=0.18)

        # Data markers
        mask_d = y_data > 0
        ax_main.errorbar(
            bin_centers[mask_d], y_data[mask_d], yerr=y_data_err[mask_d],
            fmt="o", color="black", ms=5.0, capsize=2, lw=1.2, label=f"Data (μ={np.mean(data_vals):.1f})"
        )

        ax_main.set_title(title, fontsize=11, fontweight="bold")
        ax_main.set_ylabel("Probability Density [1/bin]", fontsize=9)
        ax_main.set_xlim(xmin, xmax)
        max_val = max(np.max(y_data), max(np.max(y_sim[m]) for m in y_sim)) if len(y_data) > 0 else 1.0
        ax_main.set_ylim(0, max_val * 1.4 if max_val > 0 else 1.0)
        ax_main.grid(True, linestyle=":", alpha=0.6)
        ax_main.legend(loc="upper right", fontsize=7.2, framealpha=0.9)
        ax_main.xaxis.set_minor_locator(AutoMinorLocator())
        ax_main.yaxis.set_minor_locator(AutoMinorLocator())

        # Ratio panel: Model / Data
        ax_ratio.axhline(1.0, color="gray", linestyle="--", lw=1.0)
        offsets = {"raw": -0.15, "theo": -0.05, "calib": 0.05, "step": 0.15}

        for m in ["raw", "theo", "calib", "step"]:
            st = model_styles[m]
            mask_r = (y_data > 0) & (y_sim[m] > 0)
            if np.any(mask_r):
                r_val = y_sim[m][mask_r] / y_data[mask_r]
                r_err = r_val * np.sqrt((y_data_err[mask_r] / y_data[mask_r])**2 + (y_sim_err[m][mask_r] / y_sim[m][mask_r])**2)
                ax_ratio.errorbar(
                    bin_centers[mask_r] + offsets[m] * (bin_centers[1] - bin_centers[0]),
                    r_val, yerr=r_err,
                    fmt=st["marker"], color=st["color"], mfc="none" if m == "raw" else st["color"],
                    ms=4.0, capsize=2, lw=1.0, label=f"{m} / Data"
                )

        ax_ratio.set_xlim(xmin, xmax)
        ax_ratio.set_ylim(0.0, 3.0)
        ax_ratio.set_xlabel(xlabel, fontsize=10)
        ax_ratio.set_ylabel("Sim / Data", fontsize=9)
        ax_ratio.grid(True, linestyle=":", alpha=0.5)
        ax_ratio.xaxis.set_minor_locator(AutoMinorLocator())

    for idx in range(len(keys), nrows * ncols):
        row = (idx // ncols) * 2
        col = idx % ncols
        axes[row, col].set_visible(False)
        axes[row + 1, col].set_visible(False)

    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved figure: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Testbeam comparison across 4 threshold models")
    parser.add_argument("--cache", default="output/tb_scifi100_cache.pkl")
    parser.add_argument("--output", default="output/tb_vs_sim_4models.root")
    parser.add_argument("--plot-dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading cached Testbeam events from {args.cache}...")
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)

    # Precompute thresholds
    th_theo, th_calib, th_step = precompute_threshold_maps()

    # 1. Primary Benchmark: 180 GeV pi+
    print("\n[Testbeam] Processing 180 GeV pi+ ...")
    d_180, mc_180 = evaluate_tb_energy(180, cache[180]["data"], cache[180]["mc"], th_theo, th_calib, th_step)

    # Generate 180 GeV multi-panel PNGs
    groups = [
        ("Subsystems (US & DS Total, 180 GeV pi+)", ["us_total", "ds_total"], "tb_180gev_subsystems_4models.png", 2),
        ("US Planes (Planes 0 to 4, 180 GeV pi+)", ["us_pl0", "us_pl1", "us_pl2", "us_pl3", "us_pl4"], "tb_180gev_us_planes_4models.png", 3),
        ("DS Stations (180 GeV pi+)", [
            "ds_h_pl0", "ds_v_pl0", "ds_h_pl1", "ds_v_pl1", "ds_h_pl2", "ds_v_pl2", "ds_v_pl3"
        ], "tb_180gev_ds_planes_4models.png", 4),
    ]

    for title, keys, fname, ncols in groups:
        out_png = os.path.join(args.plot_dir, fname)
        plot_tb_4models(d_180, mc_180, keys, title, out_png, ncols=ncols)
        art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"
        if os.path.exists(art_dir):
            os.system(f"cp {out_png} {os.path.join(art_dir, fname)}")

    # 2. Multi-Energy Summary Figure
    fig_sum, axes_sum = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    energies = [100, 140, 180, 240, 300]
    
    # Store summary means
    summary_means = {
        "energies": energies,
        "data_us": [], "raw_us": [], "theo_us": [], "calib_us": [], "step_us": [],
        "data_ds": [], "raw_ds": [], "theo_ds": [], "calib_ds": [], "step_ds": [],
    }

    all_energy_data = {}
    all_energy_mc = {}

    for e in energies:
        d_e, mc_e = evaluate_tb_energy(e, cache[e]["data"], cache[e]["mc"], th_theo, th_calib, th_step)
        all_energy_data[e] = d_e
        all_energy_mc[e] = mc_e

        summary_means["data_us"].append(np.mean(d_e["us_total"]))
        summary_means["raw_us"].append(np.mean(mc_e["raw"]["us_total"]))
        summary_means["theo_us"].append(np.mean(mc_e["theo"]["us_total"]))
        summary_means["calib_us"].append(np.mean(mc_e["calib"]["us_total"]))
        summary_means["step_us"].append(np.mean(mc_e["step"]["us_total"]))

        summary_means["data_ds"].append(np.mean(d_e["ds_total"]))
        summary_means["raw_ds"].append(np.mean(mc_e["raw"]["ds_total"]))
        summary_means["theo_ds"].append(np.mean(mc_e["theo"]["ds_total"]))
        summary_means["calib_ds"].append(np.mean(mc_e["calib"]["ds_total"]))
        summary_means["step_ds"].append(np.mean(mc_e["step"]["ds_total"]))

    # US Hit Multiplicity vs Energy
    ax_us = axes_sum[0, 0]
    ax_us.plot(energies, summary_means["data_us"], "o-", color="black", lw=2.2, label="Testbeam Data")
    ax_us.plot(energies, summary_means["raw_us"], "s:", color="#1f77b4", lw=1.8, label="Sim Raw (Eth=0)")
    ax_us.plot(energies, summary_means["theo_us"], "D--", color="#9467bd", lw=1.8, label="Sim Theo (k=1)")
    ax_us.plot(energies, summary_means["calib_us"], "o-", color="#d62728", lw=2.0, label="Sim Theo+k")
    ax_us.plot(energies, summary_means["step_us"], "^-.", color="#2ca02c", lw=1.8, label="Sim Step (1.5 MeV)")
    ax_us.set_title("Mean Upstream Hits vs Beam Energy", fontsize=11, fontweight="bold")
    ax_us.set_xlabel("Beam Energy [GeV]", fontsize=10)
    ax_us.set_ylabel("Mean US Hits", fontsize=10)
    ax_us.legend(fontsize=8)
    ax_us.grid(True, linestyle=":", alpha=0.6)

    # DS Hit Multiplicity vs Energy
    ax_ds = axes_sum[0, 1]
    ax_ds.plot(energies, summary_means["data_ds"], "o-", color="black", lw=2.2, label="Testbeam Data")
    ax_ds.plot(energies, summary_means["raw_ds"], "s:", color="#1f77b4", lw=1.8, label="Sim Raw (Eth=0)")
    ax_ds.plot(energies, summary_means["theo_ds"], "D--", color="#9467bd", lw=1.8, label="Sim Theo (k=1)")
    ax_ds.plot(energies, summary_means["calib_ds"], "o-", color="#d62728", lw=2.0, label="Sim Theo+k")
    ax_ds.plot(energies, summary_means["step_ds"], "^-.", color="#2ca02c", lw=1.8, label="Sim Step (1.0 MeV)")
    ax_ds.set_title("Mean Downstream Hits vs Beam Energy", fontsize=11, fontweight="bold")
    ax_ds.set_xlabel("Beam Energy [GeV]", fontsize=10)
    ax_ds.set_ylabel("Mean DS Hits", fontsize=10)
    ax_ds.legend(fontsize=8)
    ax_ds.grid(True, linestyle=":", alpha=0.6)

    # Ratio to Data vs Energy (US)
    ax_rus = axes_sum[1, 0]
    ax_rus.axhline(1.0, color="gray", linestyle="--")
    ax_rus.plot(energies, np.array(summary_means["raw_us"]) / np.array(summary_means["data_us"]), "s:", color="#1f77b4", label="Raw / Data")
    ax_rus.plot(energies, np.array(summary_means["theo_us"]) / np.array(summary_means["data_us"]), "D--", color="#9467bd", label="Theo / Data")
    ax_rus.plot(energies, np.array(summary_means["calib_us"]) / np.array(summary_means["data_us"]), "o-", color="#d62728", label="Theo+k / Data")
    ax_rus.plot(energies, np.array(summary_means["step_us"]) / np.array(summary_means["data_us"]), "^-.", color="#2ca02c", label="Step / Data")
    ax_rus.set_title("US Sim/Data Ratio vs Energy", fontsize=11, fontweight="bold")
    ax_rus.set_xlabel("Beam Energy [GeV]", fontsize=10)
    ax_rus.set_ylabel("Sim / Data Ratio", fontsize=10)
    ax_rus.set_ylim(0.5, 3.5)
    ax_rus.legend(fontsize=8)
    ax_rus.grid(True, linestyle=":", alpha=0.6)

    # Ratio to Data vs Energy (DS)
    ax_rds = axes_sum[1, 1]
    ax_rds.axhline(1.0, color="gray", linestyle="--")
    ax_rds.plot(energies, np.array(summary_means["raw_ds"]) / np.array(summary_means["data_ds"]), "s:", color="#1f77b4", label="Raw / Data")
    ax_rds.plot(energies, np.array(summary_means["theo_ds"]) / np.array(summary_means["data_ds"]), "D--", color="#9467bd", label="Theo / Data")
    ax_rds.plot(energies, np.array(summary_means["calib_ds"]) / np.array(summary_means["data_ds"]), "o-", color="#d62728", label="Theo+k / Data")
    ax_rds.plot(energies, np.array(summary_means["step_ds"]) / np.array(summary_means["data_ds"]), "^-.", color="#2ca02c", label="Step / Data")
    ax_rds.set_title("DS Sim/Data Ratio vs Energy", fontsize=11, fontweight="bold")
    ax_rds.set_xlabel("Beam Energy [GeV]", fontsize=10)
    ax_rds.set_ylabel("Sim / Data Ratio", fontsize=10)
    ax_rds.set_ylim(0.5, 1.8)
    ax_rds.legend(fontsize=8)
    ax_rds.grid(True, linestyle=":", alpha=0.6)

    # 180 GeV Longitudinal Profile across US Planes
    ax_long = axes_sum[0, 2]
    planes = np.arange(5)
    d_pl = [np.mean(d_180[f"us_pl{p}"]) for p in planes]
    raw_pl = [np.mean(mc_180["raw"][f"us_pl{p}"]) for p in planes]
    theo_pl = [np.mean(mc_180["theo"][f"us_pl{p}"]) for p in planes]
    calib_pl = [np.mean(mc_180["calib"][f"us_pl{p}"]) for p in planes]
    step_pl = [np.mean(mc_180["step"][f"us_pl{p}"]) for p in planes]

    ax_long.plot(planes, d_pl, "o-", color="black", lw=2.2, label="Data")
    ax_long.plot(planes, raw_pl, "s:", color="#1f77b4", lw=1.8, label="Raw")
    ax_long.plot(planes, theo_pl, "D--", color="#9467bd", lw=1.8, label="Theo")
    ax_long.plot(planes, calib_pl, "o-", color="#d62728", lw=2.0, label="Theo+k")
    ax_long.plot(planes, step_pl, "^-.", color="#2ca02c", lw=1.8, label="Step")
    ax_long.set_title("180 GeV Longitudinal Profile (US Planes)", fontsize=11, fontweight="bold")
    ax_long.set_xlabel("US Plane", fontsize=10)
    ax_long.set_ylabel("Mean Hits per Plane", fontsize=10)
    ax_long.set_xticks(planes)
    ax_long.set_xticklabels([f"Pl {p}" for p in planes])
    ax_long.legend(fontsize=8)
    ax_long.grid(True, linestyle=":", alpha=0.6)

    # 180 GeV DS Stations Profile
    ax_ds_prof = axes_sum[1, 2]
    ds_labels = ["DS1-H", "DS1-V", "DS2-H", "DS2-V", "DS3-H", "DS3-V", "DS4-V"]
    ds_keys = ["ds_h_pl0", "ds_v_pl0", "ds_h_pl1", "ds_v_pl1", "ds_h_pl2", "ds_v_pl2", "ds_v_pl3"]
    idx_ds = np.arange(len(ds_keys))
    d_ds_vals = [np.mean(d_180[k]) for k in ds_keys]
    raw_ds_vals = [np.mean(mc_180["raw"][k]) for k in ds_keys]
    calib_ds_vals = [np.mean(mc_180["calib"][k]) for k in ds_keys]
    step_ds_vals = [np.mean(mc_180["step"][k]) for k in ds_keys]

    ax_ds_prof.plot(idx_ds, d_ds_vals, "o-", color="black", lw=2.2, label="Data")
    ax_ds_prof.plot(idx_ds, raw_ds_vals, "s:", color="#1f77b4", lw=1.8, label="Raw")
    ax_ds_prof.plot(idx_ds, calib_ds_vals, "o-", color="#d62728", lw=2.0, label="Theo+k")
    ax_ds_prof.plot(idx_ds, step_ds_vals, "^-.", color="#2ca02c", lw=1.8, label="Step")
    ax_ds_prof.set_title("180 GeV DS Profile (Stations)", fontsize=11, fontweight="bold")
    ax_ds_prof.set_xlabel("DS Station", fontsize=10)
    ax_ds_prof.set_ylabel("Mean Hits per Station", fontsize=10)
    ax_ds_prof.set_xticks(idx_ds)
    ax_ds_prof.set_xticklabels(ds_labels, rotation=30, fontsize=8)
    ax_ds_prof.legend(fontsize=8)
    ax_ds_prof.grid(True, linestyle=":", alpha=0.6)

    sum_png = os.path.join(args.plot_dir, "tb_all_energies_summary_4models.png")
    fig_sum.savefig(sum_png, dpi=200)
    plt.close(fig_sum)
    print(f"Saved figure: {sum_png}")
    art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"
    if os.path.exists(art_dir):
        os.system(f"cp {sum_png} {os.path.join(art_dir, 'tb_all_energies_summary_4models.png')}")

    # 3. Write ROOT Canvases and Histograms
    print(f"\n[ROOT] Writing Testbeam canvases and histograms to {args.output}...")
    rfile = ROOT.TFile.Open(args.output, "RECREATE")

    tb_canvas_configs = [
        ("c_tb_subsystems", "TB 180 GeV Subsystems (4 Models)", ["us_total", "ds_total"], 2, 1),
        ("c_tb_us_planes", "TB 180 GeV US Planes (4 Models)", ["us_pl0", "us_pl1", "us_pl2", "us_pl3", "us_pl4"], 3, 2),
        ("c_tb_ds_planes", "TB 180 GeV DS Planes (4 Models)", [
            "ds_h_pl0", "ds_v_pl0", "ds_h_pl1", "ds_v_pl1", "ds_h_pl2", "ds_v_pl2", "ds_v_pl3"
        ], 4, 2),
    ]

    for cname, ctitle, keys, nx, ny in tb_canvas_configs:
        c = ROOT.TCanvas(cname, ctitle, 400 * nx, 350 * ny)
        c.Divide(nx, ny)

        for p_idx, k in enumerate(keys):
            c.cd(p_idx + 1)
            title, xlabel, nbins, xmin, xmax = TB_HIST_SPECS[k]

            h_data = ROOT.TH1D(f"h_tb_data_{k}", f"{title} (Data);{xlabel};Normalized Entries", nbins, xmin, xmax)
            h_raw = ROOT.TH1D(f"h_tb_mc_raw_{k}", f"{title} (Raw);{xlabel};Normalized Entries", nbins, xmin, xmax)
            h_theo = ROOT.TH1D(f"h_tb_mc_theo_{k}", f"{title} (Theo);{xlabel};Normalized Entries", nbins, xmin, xmax)
            h_calib = ROOT.TH1D(f"h_tb_mc_calib_{k}", f"{title} (Calib);{xlabel};Normalized Entries", nbins, xmin, xmax)
            h_step = ROOT.TH1D(f"h_tb_mc_step_{k}", f"{title} (Step);{xlabel};Normalized Entries", nbins, xmin, xmax)

            for v in d_180[k]: h_data.Fill(float(v))
            for v in mc_180["raw"][k]: h_raw.Fill(float(v))
            for v in mc_180["theo"][k]: h_theo.Fill(float(v))
            for v in mc_180["calib"][k]: h_calib.Fill(float(v))
            for v in mc_180["step"][k]: h_step.Fill(float(v))

            # Normalize to unit area
            if h_data.Integral() > 0: h_data.Scale(1.0 / h_data.Integral())
            if h_raw.Integral() > 0: h_raw.Scale(1.0 / h_raw.Integral())
            if h_theo.Integral() > 0: h_theo.Scale(1.0 / h_theo.Integral())
            if h_calib.Integral() > 0: h_calib.Scale(1.0 / h_calib.Integral())
            if h_step.Integral() > 0: h_step.Scale(1.0 / h_step.Integral())

            h_data.SetLineColor(ROOT.kBlack)
            h_data.SetMarkerStyle(20)
            h_data.SetMarkerSize(0.8)

            h_raw.SetLineColor(ROOT.kBlue + 1)
            h_raw.SetLineStyle(3)
            h_raw.SetLineWidth(2)

            h_theo.SetLineColor(ROOT.kMagenta + 2)
            h_theo.SetLineStyle(2)
            h_theo.SetLineWidth(2)

            h_calib.SetLineColor(ROOT.kRed + 1)
            h_calib.SetLineWidth(2)

            h_step.SetLineColor(ROOT.kGreen + 2)
            h_step.SetLineStyle(4)
            h_step.SetLineWidth(2)

            max_y = max(h_data.GetMaximum(), h_raw.GetMaximum(), h_theo.GetMaximum(), h_calib.GetMaximum(), h_step.GetMaximum())
            h_data.SetMaximum(max_y * 1.35)

            h_data.Draw("E P")
            h_raw.Draw("HIST SAME")
            h_theo.Draw("HIST SAME")
            h_calib.Draw("HIST SAME")
            h_step.Draw("HIST SAME")

            leg = ROOT.TLegend(0.55, 0.65, 0.88, 0.88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.AddEntry(h_data, "Data 180 GeV", "ep")
            leg.AddEntry(h_raw, "Sim Raw", "l")
            leg.AddEntry(h_theo, "Sim Theo (k=1)", "l")
            leg.AddEntry(h_calib, "Sim Theo+k", "l")
            leg.AddEntry(h_step, "Sim Step (1/1.5 MeV)", "l")
            leg.Draw()

            c.store = getattr(c, "store", [])
            c.store.extend([h_data, h_raw, h_theo, h_calib, h_step, leg])

            h_data.Write()
            h_raw.Write()
            h_theo.Write()
            h_calib.Write()
            h_step.Write()

        c.Write()

    rfile.Close()
    print(f"Successfully saved Testbeam canvases to {args.output}")


if __name__ == "__main__":
    main()
