#!/usr/bin/env python3
"""
scripts/qdc_calib/study_tb_beam_shape_and_leakage.py
---------------------------------------------------
Infers the beam shape from H8 testbeam data and studies shower leakage effects:
1. Reconstructs the 2D incoming beam profile (x, y) from:
   - US Plane 0 bar occupancy (y profile, 10 bars)
   - DS1-V vertical bar occupancy (x profile, 60 bars)
   - US Plane 0 time difference dt = t0 - t1 (x position via propagation speed)
2. Compares real beam shape with MC pencil-beam simulation (delta function at x=-38.17, y=44.35).
3. Investigates shower leakage by applying self-consistent impact position restrictions:
   - Evaluates threshold criteria first (Raw, Theo, Calib, Step on MC; in-time on Data).
   - In pre-shower / initial planes (Nhits <= 2), rejects events with hits on detector edge bars.
   - Allows outer hits once shower cluster has formed (Nhits > 2).
4. Produces publication-quality figures and ROOT output:
   - plots/tb_beam_shape_inferred_from_data.png
   - plots/tb_leakage_study_unrestricted_vs_restricted.png
   - plots/tb_leakage_restricted_planes_comparison.png
   - output/tb_leakage_restricted_comparison.root
"""

from __future__ import annotations

import os
import sys
import pickle
import time
from typing import Dict, List, Tuple, Any, Set

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


# ==============================================================================
# 1. Data Extraction: Beam Shape and Detailed Shower Hits
# ==============================================================================
def extract_testbeam_data(
    data_file: str, max_events: int = 40000, max_showers: int = 1500
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extracts beam shape observables and clean shower events with full bar hit details.
    """
    print(f"\n[Data] Extracting from {data_file} (up to {max_events:,} events)...")
    t0 = time.time()

    f = ROOT.TFile.Open(data_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open data file: {data_file}")
    tree = f.Get("rawConv")

    beam_shape = {
        "single_pl0_bars": [],
        "all_pl0_bars": [],
        "ds1_v_bars": [],
        "pl0_dt_ns": [],
        "pl0_x_cm": [],
    }
    showers = []
    v_eff = 13.9  # cm/ns effective light velocity in scintillator (c_scint = 13.88 +- 0.57 cm/ns)

    n_entries = min(max_events, tree.GetEntries())
    for iev in range(n_entries):
        tree.GetEntry(iev)
        n_mf = len(tree.Digi_MuFilterHits)
        if n_mf == 0:
            continue

        n_scifi = len(tree.Digi_ScifiHits)

        # Extract in-time MuFilter hits
        us_hits_by_plane: Dict[int, Set[int]] = {p: set() for p in range(5)}
        ds_hits_by_plane: Dict[str, Set[int]] = {
            "ds_h_pl0": set(), "ds_v_pl0": set(),
            "ds_h_pl1": set(), "ds_v_pl1": set(),
            "ds_h_pl2": set(), "ds_v_pl2": set(),
            "ds_v_pl3": set(),
        }

        pl0_in_time_bars = []
        for h in tree.Digi_MuFilterHits:
            if not h.isValid():
                continue
            times = [h.GetTime(ch) for ch in range(h.GetnSides()) if h.GetTime(ch) != -999.0]
            if not any(abs(t) <= 15.0 for t in times):
                continue

            did = h.GetDetectorID()
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000

            if sub == 2 and 0 <= pl < 5:
                us_hits_by_plane[pl].add(bar)
                if pl == 0:
                    pl0_in_time_bars.append(bar)
                    if h.GetnSides() == 2:
                        t0_ch = h.GetTime(0)
                        t1_ch = h.GetTime(1)
                        if t0_ch != -999.0 and t1_ch != -999.0 and abs(t0_ch) <= 15 and abs(t1_ch) <= 15:
                            dt = t0_ch - t1_ch
                            beam_shape["pl0_dt_ns"].append(dt)
                            beam_shape["pl0_x_cm"].append(0.5 * v_eff * dt)

            elif sub == 3:
                if pl == 0:
                    if bar < 60: ds_hits_by_plane["ds_h_pl0"].add(bar)
                    else:
                        ds_hits_by_plane["ds_v_pl0"].add(bar - 60)
                        beam_shape["ds1_v_bars"].append(bar - 60)
                elif pl == 1:
                    if bar < 60: ds_hits_by_plane["ds_h_pl1"].add(bar)
                    else: ds_hits_by_plane["ds_v_pl1"].add(bar - 60)
                elif pl == 2:
                    if bar < 60: ds_hits_by_plane["ds_h_pl2"].add(bar)
                    else: ds_hits_by_plane["ds_v_pl2"].add(bar - 60)
                elif pl == 3 and bar >= 60:
                    ds_hits_by_plane["ds_v_pl3"].add(bar - 60)

        # Beam shape single-track recording
        for b in pl0_in_time_bars:
            beam_shape["all_pl0_bars"].append(b)
        if len(pl0_in_time_bars) == 1:
            beam_shape["single_pl0_bars"].append(pl0_in_time_bars[0])

        # Shower event recording (SciFi >= 100)
        if n_scifi >= 100:
            us_tot = sum(len(us_hits_by_plane[p]) for p in range(5))
            ds_tot = sum(len(ds_hits_by_plane[k]) for k in ds_hits_by_plane)
            showers.append({
                "us_total": us_tot,
                "ds_total": ds_tot,
                "us_planes": {p: sorted(us_hits_by_plane[p]) for p in range(5)},
                "ds_planes": {k: sorted(ds_hits_by_plane[k]) for k in ds_hits_by_plane},
            })
            if len(showers) >= max_showers:
                break

    f.Close()
    print(f"[Data] Extracted {len(showers)} shower events, {len(beam_shape['single_pl0_bars'])} single-MIP tracks in {time.time()-t0:.1f}s.")
    return beam_shape, showers


# ==============================================================================
# 2. Simulation Thresholds Precomputation
# ==============================================================================
def precompute_threshold_maps() -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Returns (th_theo_kev, th_calib_kev, th_step_kev) for all MuFilter bars in keV."""
    calc_theo = mpe.MinPE(year=2023)
    calc_theo._k_factors = {}
    calc_calib = mpe.MinPE(year=2023)

    th_theo, th_calib, th_step = {}, {}, {}

    for pl in range(5):
        for b in range(10):
            did = 20000 + pl * 1000 + b
            p = {"detector_id": did, "x": -37.7, "y": 0.0, "z": 0.0, "eloss": 1.0}
            th_theo[did] = float(calc_theo.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_calib[did] = float(calc_calib.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_step[did] = 1500.0

    for pl in range(4):
        if pl < 3:
            for b in range(60):
                did = 30000 + pl * 1000 + b
                p = {"detector_id": did, "x": -37.7, "y": 0.0, "z": 0.0, "eloss": 1.0}
                th_theo[did] = float(calc_theo.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
                th_calib[did] = float(calc_calib.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
                th_step[did] = 1000.0
        for b in range(60, 120):
            did = 30000 + pl * 1000 + b
            p = {"detector_id": did, "x": 0.0, "y": 20.0, "z": 0.0, "eloss": 1.0}
            th_theo[did] = float(calc_theo.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_calib[did] = float(calc_calib.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_step[did] = 1000.0

    return th_theo, th_calib, th_step


# ==============================================================================
# 3. Self-Consistent Restriction Evaluator
# ==============================================================================
def check_edge_restriction(
    us_pl_bars: Dict[int, List[int]],
    max_initial_hits: int = 2,
    edge_mode: str = "edge_09",
) -> bool:
    """
    Returns True if the event passes the self-consistent impact position restriction.
    A plane is initial if 1 <= len(bars) <= max_initial_hits.
    """
    if edge_mode == "unrestricted":
        return True

    edge_bars = {0, 9} if edge_mode == "edge_09" else {0, 1, 8, 9}

    for pl in range(5):
        bars = us_pl_bars[pl]
        n_h = len(bars)
        if 1 <= n_h <= max_initial_hits:
            if any(b in edge_bars for b in bars):
                return False
    return True


def evaluate_dataset_under_restrictions(
    data_showers: List[Dict[str, Any]],
    mc_showers: List[Dict[str, Any]],
    th_maps: Tuple[Dict[int, float], Dict[int, float], Dict[int, float]],
    edge_mode: str = "unrestricted",
    max_initial_hits: int = 2,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """
    Evaluates Data and all 4 simulation models self-consistently under the chosen edge restriction.
    """
    th_theo, th_calib, th_step = th_maps
    models = ["raw", "theo", "calib", "step"]

    # 1. Filter Data
    filtered_data = {
        "us_total": [], "ds_total": [],
        "us_pl0": [], "us_pl1": [], "us_pl2": [], "us_pl3": [], "us_pl4": [],
        "ds_h_pl0": [], "ds_v_pl0": [], "ds_h_pl1": [], "ds_v_pl1": [],
        "ds_h_pl2": [], "ds_v_pl2": [], "ds_v_pl3": [],
    }

    for ev in data_showers:
        if not check_edge_restriction(ev["us_planes"], max_initial_hits=max_initial_hits, edge_mode=edge_mode):
            continue
        filtered_data["us_total"].append(ev["us_total"])
        filtered_data["ds_total"].append(ev["ds_total"])
        for p in range(5):
            filtered_data[f"us_pl{p}"].append(len(ev["us_planes"][p]))
        for k in ["ds_h_pl0", "ds_v_pl0", "ds_h_pl1", "ds_v_pl1", "ds_h_pl2", "ds_v_pl2", "ds_v_pl3"]:
            filtered_data[k].append(len(ev["ds_planes"][k]))

    data_arrays = {k: np.array(v, dtype=np.int32) for k, v in filtered_data.items()}

    # 2. Filter MC across all 4 models self-consistently
    filtered_mc: Dict[str, Dict[str, List[int]]] = {m: {k: [] for k in filtered_data} for m in models}

    for ev in mc_showers:
        bar_edep = ev["bar_edep"]

        for m in models:
            # First determine surviving hits for this model
            us_pl_bars = {p: [] for p in range(5)}
            ds_pl_counts = {
                "ds_h_pl0": 0, "ds_v_pl0": 0,
                "ds_h_pl1": 0, "ds_v_pl1": 0,
                "ds_h_pl2": 0, "ds_v_pl2": 0,
                "ds_v_pl3": 0,
            }

            for did, edep_kev in bar_edep.items():
                # Apply model threshold
                if m == "raw": pass_th = (edep_kev > 0.0)
                elif m == "theo": pass_th = (edep_kev > th_theo.get(did, 300.0))
                elif m == "calib": pass_th = (edep_kev > th_calib.get(did, 300.0))
                elif m == "step": pass_th = (edep_kev > th_step.get(did, 1500.0))

                if not pass_th:
                    continue

                sub = did // 10000
                pl = (did // 1000) % 10
                bar = did % 1000

                if sub == 2 and 0 <= pl < 5:
                    us_pl_bars[pl].append(bar)
                elif sub == 3:
                    if pl == 0:
                        if bar < 60: ds_pl_counts["ds_h_pl0"] += 1
                        else: ds_pl_counts["ds_v_pl0"] += 1
                    elif pl == 1:
                        if bar < 60: ds_pl_counts["ds_h_pl1"] += 1
                        else: ds_pl_counts["ds_v_pl1"] += 1
                    elif pl == 2:
                        if bar < 60: ds_pl_counts["ds_h_pl2"] += 1
                        else: ds_pl_counts["ds_v_pl2"] += 1
                    elif pl == 3 and bar >= 60:
                        ds_pl_counts["ds_v_pl3"] += 1

            # Check self-consistent edge restriction on surviving hits
            if not check_edge_restriction(us_pl_bars, max_initial_hits=max_initial_hits, edge_mode=edge_mode):
                continue

            us_tot = sum(len(us_pl_bars[p]) for p in range(5))
            ds_tot = sum(ds_pl_counts.values())

            filtered_mc[m]["us_total"].append(us_tot)
            filtered_mc[m]["ds_total"].append(ds_tot)
            for p in range(5):
                filtered_mc[m][f"us_pl{p}"].append(len(us_pl_bars[p]))
            for k, cnt in ds_pl_counts.items():
                filtered_mc[m][k].append(cnt)

    mc_arrays = {m: {k: np.array(v, dtype=np.int32) for k, v in filtered_mc[m].items()} for m in models}
    return data_arrays, mc_arrays


# ==============================================================================
# 4. Plotting Beam Shape
# ==============================================================================
def plot_beam_shape(beam_shape: Dict[str, Any], out_png: str):
    """Plots the inferred incoming beam shape from real data."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    # 1. Vertical profile from single-MIP US Plane 0
    ax1 = axes[0]
    single_bars = np.array(beam_shape["single_pl0_bars"])
    n_single = len(single_bars)
    counts = np.bincount(single_bars, minlength=10)[:10]
    fractions = counts / n_single * 100.0 if n_single > 0 else np.zeros(10)
    bars_x = np.arange(10)

    colors = ["#d9534f" if b in (0, 1, 8, 9) else "#0275d8" for b in bars_x]
    bars_plot = ax1.bar(bars_x, fractions, color=colors, edgecolor="black", alpha=0.85, width=0.7)
    ax1.set_title(r"Beam Vertical Profile ($y$): US Plane 0", fontsize=11, fontweight="bold")
    ax1.set_xlabel("US Plane 0 Bar Index (0=Bottom, 9=Top)", fontsize=10)
    ax1.set_ylabel("Incident Pion Fraction [%]", fontsize=10)
    ax1.set_xticks(bars_x)
    ax1.set_ylim(0, max(fractions) * 1.25)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Annotate core and leakage
    core_pct = fractions[4] + fractions[5]
    edge_pct = fractions[0] + fractions[1] + fractions[8] + fractions[9]
    ax1.text(0.05, 0.90, f"Central Core (Bars 4-5): {core_pct:.1f}%\nEdge Bars (0,1,8,9): {edge_pct:.1f}%",
             transform=ax1.transAxes, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # 2. Horizontal profile from DS1-V (60 bars across x)
    ax2 = axes[1]
    ds1_v = np.array(beam_shape["ds1_v_bars"])
    ax2.hist(ds1_v, bins=60, range=(-0.5, 59.5), color="#5cb85c", edgecolor="#2d682d", alpha=0.8)
    ax2.set_title(r"Beam Horizontal Profile ($x$): DS1-V", fontsize=11, fontweight="bold")
    ax2.set_xlabel("DS Station 1 Vertical Bar Index (0..59)", fontsize=10)
    ax2.set_ylabel("In-time Hits", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.axvline(np.median(ds1_v) if len(ds1_v) > 0 else 30, color="darkred", linestyle="--", lw=1.5,
                label=f"Median Bar: {np.median(ds1_v):.1f}")
    ax2.legend(fontsize=9, loc="upper left")

    # 3. Horizontal position from dt in US Plane 0
    ax3 = axes[2]
    xs = np.array(beam_shape["pl0_x_cm"])
    xs_cut = xs[np.abs(xs) <= 45.0]
    ax3.hist(xs_cut, bins=45, range=(-45, 45), color="#f0ad4e", edgecolor="#b27318", alpha=0.85)
    ax3.set_title(r"Beam Horizontal Reconstruction from US $\Delta t$", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Reconstructed $x$ Position [cm] ($v_{eff} = 15$ cm/ns)", fontsize=10)
    ax3.set_ylabel("Pions / 2 cm", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.axvline(0.0, color="gray", linestyle=":")
    if len(xs_cut) > 0:
        ax3.text(0.05, 0.88, f"Mean x: {np.mean(xs_cut):.1f} cm\nRMS: {np.std(xs_cut):.1f} cm\nMC Pencil Beam: (-38.2, 44.4) cm",
                 transform=ax3.transAxes, fontsize=9.2, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    plt.suptitle("180 GeV H8 Pion Beam Shape Inferred from Testbeam Data vs. Pencil-Beam MC", fontsize=13, fontweight="bold")
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"Saved figure: {out_png}")


# ==============================================================================
# 5. Plotting Leakage Restricted Multiplicity
# ==============================================================================
def plot_leakage_comparison(
    unres_data: Dict[str, np.ndarray],
    unres_mc: Dict[str, Dict[str, np.ndarray]],
    res_data: Dict[str, np.ndarray],
    res_mc: Dict[str, Dict[str, np.ndarray]],
    out_png: str,
):
    """
    Side-by-side comparison of US and DS hit multiplicities:
    Unrestricted (leaking data) vs Restricted (central impact only).
    """
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    model_styles = {
        "raw":   {"label": "Sim Raw (Eth=0)", "color": "#1f77b4", "ls": ":", "lw": 1.6},
        "theo":  {"label": "Sim Theo (k=1.0)", "color": "#9467bd", "ls": "--", "lw": 1.8},
        "calib": {"label": "Sim Theo + k(SiPM)", "color": "#d62728", "ls": "-", "lw": 2.0},
        "step":  {"label": "Sim Step (1/1.5 MeV)", "color": "#2ca02c", "ls": "-.", "lw": 1.8},
    }

    # Column 0: US Total Hits
    # Row 0: Unrestricted
    ax_u0 = axes[0, 0]
    ax_u0.set_title("US Total Hits: Unrestricted (Inclusive Showers)", fontsize=11, fontweight="bold")
    bins_u = np.arange(4.5, 51.5, 1.0)
    b_cent_u = 0.5 * (bins_u[:-1] + bins_u[1:])

    h_du, _ = np.histogram(unres_data["us_total"], bins=bins_u)
    p_du = h_du / np.sum(h_du) if np.sum(h_du) > 0 else h_du
    ax_u0.errorbar(b_cent_u[p_du > 0], p_du[p_du > 0], yerr=np.sqrt(h_du[p_du > 0])/np.sum(h_du),
                   fmt="o", color="black", ms=4, capsize=2, label=f"Data (Mean={np.mean(unres_data['us_total']):.1f})")

    for m in ["raw", "theo", "calib", "step"]:
        h_m, _ = np.histogram(unres_mc[m]["us_total"], bins=bins_u)
        p_m = h_m / np.sum(h_m) if np.sum(h_m) > 0 else h_m
        st = model_styles[m]
        lbl = f"{st['label']} (Mean={np.mean(unres_mc[m]['us_total']):.1f})"
        ax_u0.step(bins_u, np.append(p_m, p_m[-1]), where="post", color=st["color"], ls=st["ls"], lw=st["lw"], label=lbl)

    ax_u0.set_xlabel(r"Total US Hits $N_{hits}^{US}$", fontsize=10)
    ax_u0.set_ylabel("Normalized Density [1 / hit]", fontsize=10)
    ax_u0.grid(True, linestyle=":", alpha=0.5)
    ax_u0.legend(fontsize=8, loc="upper right")

    # Row 1: Restricted (No edge hits in pre-shower planes)
    ax_u1 = axes[1, 0]
    ax_u1.set_title("US Total Hits: Impact Restricted (No Edges in Pre-shower)", fontsize=11, fontweight="bold")
    h_dru, _ = np.histogram(res_data["us_total"], bins=bins_u)
    p_dru = h_dru / np.sum(h_dru) if np.sum(h_dru) > 0 else h_dru
    ax_u1.errorbar(b_cent_u[p_dru > 0], p_dru[p_dru > 0], yerr=np.sqrt(h_dru[p_dru > 0])/np.sum(h_dru),
                   fmt="o", color="black", ms=4, capsize=2, label=f"Data Restricted (Mean={np.mean(res_data['us_total']):.1f})")

    for m in ["raw", "theo", "calib", "step"]:
        h_m, _ = np.histogram(res_mc[m]["us_total"], bins=bins_u)
        p_m = h_m / np.sum(h_m) if np.sum(h_m) > 0 else h_m
        st = model_styles[m]
        lbl = f"{st['label']} (Mean={np.mean(res_mc[m]['us_total']):.1f})"
        ax_u1.step(bins_u, np.append(p_m, p_m[-1]), where="post", color=st["color"], ls=st["ls"], lw=st["lw"], label=lbl)

    ax_u1.set_xlabel(r"Total US Hits $N_{hits}^{US}$", fontsize=10)
    ax_u1.set_ylabel("Normalized Density [1 / hit]", fontsize=10)
    ax_u1.grid(True, linestyle=":", alpha=0.5)
    ax_u1.legend(fontsize=8, loc="upper right")

    # Column 1: DS Total Hits
    bins_d = np.arange(-0.5, 26.5, 1.0)
    b_cent_d = 0.5 * (bins_d[:-1] + bins_d[1:])

    # Row 0: Unrestricted
    ax_d0 = axes[0, 1]
    ax_d0.set_title("DS Total Hits: Unrestricted (Inclusive Showers)", fontsize=11, fontweight="bold")
    h_dd, _ = np.histogram(unres_data["ds_total"], bins=bins_d)
    p_dd = h_dd / np.sum(h_dd) if np.sum(h_dd) > 0 else h_dd
    ax_d0.errorbar(b_cent_d[p_dd > 0], p_dd[p_dd > 0], yerr=np.sqrt(h_dd[p_dd > 0])/np.sum(h_dd),
                   fmt="o", color="black", ms=4, capsize=2, label=f"Data (Mean={np.mean(unres_data['ds_total']):.1f})")

    for m in ["raw", "theo", "calib", "step"]:
        h_m, _ = np.histogram(unres_mc[m]["ds_total"], bins=bins_d)
        p_m = h_m / np.sum(h_m) if np.sum(h_m) > 0 else h_m
        st = model_styles[m]
        lbl = f"{st['label']} (Mean={np.mean(unres_mc[m]['ds_total']):.1f})"
        ax_d0.step(bins_d, np.append(p_m, p_m[-1]), where="post", color=st["color"], ls=st["ls"], lw=st["lw"], label=lbl)

    ax_d0.set_xlabel(r"Total DS Hits $N_{hits}^{DS}$", fontsize=10)
    ax_d0.set_ylabel("Normalized Density [1 / hit]", fontsize=10)
    ax_d0.grid(True, linestyle=":", alpha=0.5)
    ax_d0.legend(fontsize=8, loc="upper right")

    # Row 1: Restricted
    ax_d1 = axes[1, 1]
    ax_d1.set_title("DS Total Hits: Impact Restricted (No Edges in Pre-shower)", fontsize=11, fontweight="bold")
    h_drd, _ = np.histogram(res_data["ds_total"], bins=bins_d)
    p_drd = h_drd / np.sum(h_drd) if np.sum(h_drd) > 0 else h_drd
    ax_d1.errorbar(b_cent_d[p_drd > 0], p_drd[p_drd > 0], yerr=np.sqrt(h_drd[p_drd > 0])/np.sum(h_drd),
                   fmt="o", color="black", ms=4, capsize=2, label=f"Data Restricted (Mean={np.mean(res_data['ds_total']):.1f})")

    for m in ["raw", "theo", "calib", "step"]:
        h_m, _ = np.histogram(res_mc[m]["ds_total"], bins=bins_d)
        p_m = h_m / np.sum(h_m) if np.sum(h_m) > 0 else h_m
        st = model_styles[m]
        lbl = f"{st['label']} (Mean={np.mean(res_mc[m]['ds_total']):.1f})"
        ax_d1.step(bins_d, np.append(p_m, p_m[-1]), where="post", color=st["color"], ls=st["ls"], lw=st["lw"], label=lbl)

    ax_d1.set_xlabel(r"Total DS Hits $N_{hits}^{DS}$", fontsize=10)
    ax_d1.set_ylabel("Normalized Density [1 / hit]", fontsize=10)
    ax_d1.grid(True, linestyle=":", alpha=0.5)
    ax_d1.legend(fontsize=8, loc="upper right")

    plt.suptitle("Shower Leakage Impact: Multiplicity Before vs. After Edge Restriction (180 GeV pi+)", fontsize=13, fontweight="bold")
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"Saved figure: {out_png}")


# ==============================================================================
# 6. Main Orchestrator
# ==============================================================================
def main():
    out_root = "output/tb_leakage_restricted_comparison.root"
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)

    data_file = "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100671/sndsw_raw-0000.root"
    mc_cache_file = "output/tb_scifi100_cache.pkl"

    # 1. Extract beam shape and shower events
    beam_shape, data_showers = extract_testbeam_data(data_file, max_events=35000, max_showers=1200)

    # 2. Plot inferred beam shape
    png_beam = os.path.join(plot_dir, "tb_beam_shape_inferred_from_data.png")
    plot_beam_shape(beam_shape, png_beam)

    # 3. Load MC events
    with open(mc_cache_file, "rb") as f:
        mc_cache = pickle.load(f)
    mc_showers = mc_cache[180]["mc"]

    # 4. Precompute thresholds
    th_maps = precompute_threshold_maps()

    # 5. Evaluate Unrestricted
    print("\n[Evaluation] Evaluating Unrestricted Dataset...")
    unres_data, unres_mc = evaluate_dataset_under_restrictions(data_showers, mc_showers, th_maps, edge_mode="unrestricted")

    # 6. Evaluate Restricted (no hits on edge bars 0,1,8,9 in pre-shower planes)
    print("[Evaluation] Evaluating Restricted Dataset (Excluding Edge Bars 0,1,8,9 in Pre-shower Planes)...")
    res_data, res_mc = evaluate_dataset_under_restrictions(
        data_showers, mc_showers, th_maps, edge_mode="edge_wide", max_initial_hits=2
    )

    # 7. Plot comparison
    png_leakage = os.path.join(plot_dir, "tb_leakage_study_unrestricted_vs_restricted.png")
    plot_leakage_comparison(unres_data, unres_mc, res_data, res_mc, png_leakage)

    # 8. Save ROOT file with canvases
    f_out = ROOT.TFile.Open(out_root, "RECREATE")
    c_leakage = ROOT.TCanvas("c_leakage_comparison", "Shower Leakage Comparison", 1200, 800)
    c_leakage.Divide(2, 2)

    keys = ["us_total", "ds_total"]
    hists_store = []
    for idx, k in enumerate(keys):
        nbins = 46 if k == "us_total" else 26
        xmin = 4.5 if k == "us_total" else -0.5
        xmax = 50.5 if k == "us_total" else 25.5

        # Unrestricted
        c_leakage.cd(idx * 2 + 1)
        ROOT.gPad.SetGrid()
        h_d_un = ROOT.TH1D(f"h_data_unres_{k}", f"Data Unrestricted {k};Hits;Density", nbins, xmin, xmax)
        for v in unres_data[k]: h_d_un.Fill(float(v))
        if h_d_un.Integral() > 0: h_d_un.Scale(1.0 / h_d_un.Integral())
        h_d_un.SetMarkerStyle(20)
        h_d_un.SetMarkerSize(0.8)
        h_d_un.SetLineColor(ROOT.kBlack)
        h_d_un.Draw("E P")

        h_mc_un = ROOT.TH1D(f"h_mc_calib_unres_{k}", f"Sim Calib Unrestricted {k};Hits;Density", nbins, xmin, xmax)
        for v in unres_mc["calib"][k]: h_mc_un.Fill(float(v))
        if h_mc_un.Integral() > 0: h_mc_un.Scale(1.0 / h_mc_un.Integral())
        h_mc_un.SetLineColor(ROOT.kRed + 1)
        h_mc_un.SetLineWidth(2)
        h_mc_un.Draw("HIST SAME")
        hists_store.extend([h_d_un, h_mc_un])

        # Restricted
        c_leakage.cd(idx * 2 + 2)
        ROOT.gPad.SetGrid()
        h_d_res = ROOT.TH1D(f"h_data_res_{k}", f"Data Restricted {k};Hits;Density", nbins, xmin, xmax)
        for v in res_data[k]: h_d_res.Fill(float(v))
        if h_d_res.Integral() > 0: h_d_res.Scale(1.0 / h_d_res.Integral())
        h_d_res.SetMarkerStyle(20)
        h_d_res.SetMarkerSize(0.8)
        h_d_res.SetLineColor(ROOT.kBlack)
        h_d_res.Draw("E P")

        h_mc_res = ROOT.TH1D(f"h_mc_calib_res_{k}", f"Sim Calib Restricted {k};Hits;Density", nbins, xmin, xmax)
        for v in res_mc["calib"][k]: h_mc_res.Fill(float(v))
        if h_mc_res.Integral() > 0: h_mc_res.Scale(1.0 / h_mc_res.Integral())
        h_mc_res.SetLineColor(ROOT.kRed + 1)
        h_mc_res.SetLineWidth(2)
        h_mc_res.Draw("HIST SAME")
        hists_store.extend([h_d_res, h_mc_res])

    c_leakage.Write()
    f_out.Close()
    print(f"Saved ROOT canvases and histograms to {out_root}")

    # Copy artifacts to brain
    art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"
    os.system(f"cp -f {png_beam} {png_leakage} {out_root} {art_dir}/")
    print("Copied figures and ROOT file to artifact directory.")


if __name__ == "__main__":
    main()
