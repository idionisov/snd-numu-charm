#!/usr/bin/env python3
"""
scripts/qdc_calib/plot_tb_dispersion_comparison.py
--------------------------------------------------
Compares 180 GeV pi+ Testbeam Data (Run 100671) against Geant4 MC under 4 models:
1. Data: In-time (|t| <= 15 ns) and Inclusive
2. MC Raw: E_dep > 0
3. MC Calibrated: Calibrated k_SiPM, bulk f_cone
4. MC Edge-Corrected: alpha_edge = 0.35, lambda_edge = 1.8 mm, delta_dead = 0.5 mm
5. MC Full Physical: Calibrated k_SiPM + Edge Correction + Temporal Dispersion (f_disp)

Outputs:
- plots/tb_dispersion_us0_profile.png
- plots/tb_dispersion_subsystems.png
- plots/tb_dispersion_us_planes.png
- output/tb_dispersion_comparison.root
"""

from __future__ import annotations

import os
import sys
import math
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
    os.makedirs("output", exist_ok=True)
    art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"

    # 1. Read Testbeam Data (Run 100671)
    data_file = "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100671/sndsw_raw-0000.root"
    print(f"[1/3] Extracting Data from {data_file}...")
    f_d = ROOT.TFile.Open(data_file)
    t_d = f_d.Get("rawConv")
    t_d.SetBranchStatus("*", 0)
    t_d.SetBranchStatus("Digi_ScifiHits*", 1)
    t_d.SetBranchStatus("Digi_MuFilterHits*", 1)

    d_intime = {"us_total": [], "ds_total": [], "us_pl": {p: [] for p in range(5)}}
    d_us0_occ = np.zeros(10)

    n_scan = min(35000, t_d.GetEntries())
    n_showers_data = 0
    for i in range(n_scan):
        t_d.GetEntry(i)
        if len(t_d.Digi_ScifiHits) < 100:
            continue
        n_showers_data += 1

        us_bars = {p: set() for p in range(5)}
        ds_hits = 0

        for h in t_d.Digi_MuFilterHits:
            if not h.isValid():
                continue
            times = [h.GetTime(ch) for ch in range(h.GetnSides()) if h.GetTime(ch) != -999.0]
            if not any(abs(t) <= 15.0 for t in times):
                continue
            did = h.GetDetectorID()
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000
            if sub == 2:
                us_bars[pl].add(bar)
                if pl == 0:
                    d_us0_occ[bar] += 1
            elif sub == 3 and pl == 0:
                ds_hits += 1

        nu = sum(len(us_bars[p]) for p in range(5))
        d_intime["us_total"].append(nu)
        d_intime["ds_total"].append(ds_hits)
        for p in range(5):
            d_intime["us_pl"][p].append(len(us_bars[p]))

    f_d.Close()
    d_us0_occ /= n_showers_data
    print(f"      Data: Extracted {n_showers_data} showers. Mean US0 bars = {np.sum(d_us0_occ):.2f}")

    # 2. Read MC Simulation (180 GeV pi+)
    mc_file = "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/180GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root"
    print(f"[2/3] Extracting and digitizing MC from {mc_file}...")
    f_m = ROOT.TFile.Open(mc_file)
    t_m = f_m.Get("cbmsim")
    t_m.SetBranchStatus("*", 0)
    t_m.SetBranchStatus("Digi_ScifiHits*", 1)
    t_m.SetBranchStatus("MuFilterPoint*", 1)

    # Initialize calculators
    calc_base = mpe.MinPE(year=2023, is_testbeam=True, include_edge_effect=False, include_dispersion=False)
    calc_edge = mpe.MinPE(year=2023, is_testbeam=True, include_edge_effect=True, include_dispersion=False)
    calc_full = mpe.MinPE(year=2023, is_testbeam=True, include_edge_effect=True, include_dispersion=True)

    mc_raw = {"us_total": [], "ds_total": [], "us_pl": {p: [] for p in range(5)}}
    mc_base = {"us_total": [], "ds_total": [], "us_pl": {p: [] for p in range(5)}}
    mc_edge = {"us_total": [], "ds_total": [], "us_pl": {p: [] for p in range(5)}}
    mc_full = {"us_total": [], "ds_total": [], "us_pl": {p: [] for p in range(5)}}

    occ_raw = np.zeros(10)
    occ_base = np.zeros(10)
    occ_edge = np.zeros(10)
    occ_full = np.zeros(10)

    n_showers_mc = 0
    max_mc_events = 1500
    n_entries_mc = min(max_mc_events, t_m.GetEntries())

    for i in range(n_entries_mc):
        t_m.GetEntry(i)
        if len(t_m.Digi_ScifiHits) < 100:
            continue
        n_showers_mc += 1
        if n_showers_mc % 200 == 0:
            print(f"      MC: Processed {n_showers_mc} showers...")

        # Group points by detector ID
        pts_by_bar = {}
        for pt in t_m.MuFilterPoint:
            did = pt.GetDetectorID()
            pts_by_bar.setdefault(did, []).append({
                "detector_id": did,
                "x": float(pt.GetX()),
                "y": float(pt.GetY()),
                "z": float(pt.GetZ()),
                "time": float(pt.GetTime()),
                "eloss": float(pt.GetEnergyLoss()),
                "pdg": int(pt.PdgCode()),
            })

        # Evaluate models
        u_raw, u_base, u_edge, u_full = {p: set() for p in range(5)}, {p: set() for p in range(5)}, {p: set() for p in range(5)}, {p: set() for p in range(5)}
        d_raw, d_base, d_edge, d_full = 0, 0, 0, 0

        for did, pts in pts_by_bar.items():
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000

            # 1. Raw
            total_e = sum(p["eloss"] for p in pts)
            if total_e > 0.0:
                if sub == 2:
                    u_raw[pl].add(bar)
                    if pl == 0: occ_raw[bar] += 1
                elif sub == 3 and pl == 0:
                    d_raw += 1

            # 2. Base
            pes_b = calc_base.get_sipm_pes(pts, det_id=did)
            th_b = np.array([calc_base.get_sipm_threshold_pe() for _ in range(len(pes_b))])
            if any(pes_b >= th_b):
                if sub == 2:
                    u_base[pl].add(bar)
                    if pl == 0: occ_base[bar] += 1
                elif sub == 3 and pl == 0:
                    d_base += 1

            # 3. Edge-Only
            pes_e = calc_edge.get_sipm_pes(pts, det_id=did)
            th_e = np.array([calc_edge.get_sipm_threshold_pe() for _ in range(len(pes_e))])
            if any(pes_e >= th_e):
                if sub == 2:
                    u_edge[pl].add(bar)
                    if pl == 0: occ_edge[bar] += 1
                elif sub == 3 and pl == 0:
                    d_edge += 1

            # 4. Full (Edge + Temporal Dispersion)
            pes_f = calc_full.get_sipm_pes(pts, det_id=did)
            th_f = np.array([calc_full.get_sipm_threshold_pe() for _ in range(len(pes_f))])
            if any(pes_f >= th_f):
                if sub == 2:
                    u_full[pl].add(bar)
                    if pl == 0: occ_full[bar] += 1
                elif sub == 3 and pl == 0:
                    d_full += 1

        mc_raw["us_total"].append(sum(len(u_raw[p]) for p in range(5)))
        mc_raw["ds_total"].append(d_raw)
        mc_base["us_total"].append(sum(len(u_base[p]) for p in range(5)))
        mc_base["ds_total"].append(d_base)
        mc_edge["us_total"].append(sum(len(u_edge[p]) for p in range(5)))
        mc_edge["ds_total"].append(d_edge)
        mc_full["us_total"].append(sum(len(u_full[p]) for p in range(5)))
        mc_full["ds_total"].append(d_full)

        for p in range(5):
            mc_raw["us_pl"][p].append(len(u_raw[p]))
            mc_base["us_pl"][p].append(len(u_base[p]))
            mc_edge["us_pl"][p].append(len(u_edge[p]))
            mc_full["us_pl"][p].append(len(u_full[p]))

    f_m.Close()
    occ_raw /= n_showers_mc
    occ_base /= n_showers_mc
    occ_edge /= n_showers_mc
    occ_full /= n_showers_mc

    print(f"      MC: Evaluated {n_showers_mc} showers.")
    print(f"      Mean US0 Bars: Raw={np.sum(occ_raw):.2f}, Base={np.sum(occ_base):.2f}, Edge={np.sum(occ_edge):.2f}, Full(Disp)={np.sum(occ_full):.2f}")

    # ==============================================================================
    # PLOT 1: US Plane 0 Transverse Bar Profile
    # ==============================================================================
    print("[3/3] Generating comparison plots...")
    fig, (ax1, ax_rat) = plt.subplots(2, 1, figsize=(9.5, 6.8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12}, sharex=True)
    bars = np.arange(10)
    w = 0.18

    ax1.bar(bars - 1.5*w, d_us0_occ * 100.0, w, label=f"Data (|t|<=15 ns, Mean={np.sum(d_us0_occ):.2f})", color="black", alpha=0.85)
    ax1.bar(bars - 0.5*w, occ_full * 100.0, w, label=f"MC Full: Edge + Disp (Mean={np.sum(occ_full):.2f})", color="#28a745", alpha=0.90)
    ax1.bar(bars + 0.5*w, occ_edge * 100.0, w, label=f"MC Edge-Only (Mean={np.sum(occ_edge):.2f})", color="#0275d8", alpha=0.80)
    ax1.bar(bars + 1.5*w, occ_base * 100.0, w, label=f"MC Baseline (Mean={np.sum(occ_base):.2f})", color="#d9534f", alpha=0.70)

    ax1.set_ylabel("Firing Probability [%]", fontsize=11, fontweight="bold")
    ax1.set_title("180 GeV $\\pi^+$ Transverse Profile in US Plane 0: Edge & Dispersion Impact", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9.5, loc="upper right")

    # Ratio panel
    safe_data = np.where(d_us0_occ > 0.005, d_us0_occ, np.nan)
    ax_rat.plot(bars, occ_full / safe_data, "o-", color="#28a745", label="Full / Data", lw=1.8, markersize=5)
    ax_rat.plot(bars, occ_edge / safe_data, "s--", color="#0275d8", label="Edge / Data", lw=1.5, markersize=4)
    ax_rat.plot(bars, occ_base / safe_data, "^:", color="#d9534f", label="Base / Data", lw=1.5, markersize=4)
    ax_rat.axhline(1.0, color="black", linestyle="--", alpha=0.7)
    ax_rat.set_ylabel("MC / Data", fontsize=10, fontweight="bold")
    ax_rat.set_xlabel("US Plane 0 Bar Index (0=Bottom, 9=Top)", fontsize=11, fontweight="bold")
    ax_rat.set_ylim(0.5, 4.0)
    ax_rat.set_xticks(bars)
    ax_rat.grid(True, linestyle=":", alpha=0.5)
    ax_rat.legend(fontsize=8, loc="upper right", ncol=3)

    p1 = os.path.join(plot_dir, "tb_dispersion_us0_profile.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    os.system(f"cp {p1} {os.path.join(art_dir, 'tb_dispersion_us0_profile.png')}")
    plt.close()

    # ==============================================================================
    # PLOT 2: Subsystems Hit Multiplicity (US Total & DS Total)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # US Total
    ax = axes[0]
    bins_us = np.arange(0, 51, 1)
    ax.hist(d_intime["us_total"], bins=bins_us, density=True, histtype="step", color="black", lw=2.2, label=f"Data (Mean={np.mean(d_intime['us_total']):.1f})")
    ax.hist(mc_full["us_total"], bins=bins_us, density=True, histtype="step", color="#28a745", lw=2.0, label=f"MC Full (Mean={np.mean(mc_full['us_total']):.1f})")
    ax.hist(mc_edge["us_total"], bins=bins_us, density=True, histtype="step", color="#0275d8", lw=1.6, linestyle="--", label=f"MC Edge (Mean={np.mean(mc_edge['us_total']):.1f})")
    ax.hist(mc_base["us_total"], bins=bins_us, density=True, histtype="step", color="#d9534f", lw=1.6, linestyle=":", label=f"MC Base (Mean={np.mean(mc_base['us_total']):.1f})")
    ax.set_title("US Total Hit Multiplicity (180 GeV $\\pi^+$)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Total US Fired Bars", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    # DS Station 1
    ax = axes[1]
    bins_ds = np.arange(0, 31, 1)
    ax.hist(d_intime["ds_total"], bins=bins_ds, density=True, histtype="step", color="black", lw=2.2, label=f"Data (Mean={np.mean(d_intime['ds_total']):.1f})")
    ax.hist(mc_full["ds_total"], bins=bins_ds, density=True, histtype="step", color="#28a745", lw=2.0, label=f"MC Full (Mean={np.mean(mc_full['ds_total']):.1f})")
    ax.hist(mc_edge["ds_total"], bins=bins_ds, density=True, histtype="step", color="#0275d8", lw=1.6, linestyle="--", label=f"MC Edge (Mean={np.mean(mc_edge['ds_total']):.1f})")
    ax.hist(mc_base["ds_total"], bins=bins_ds, density=True, histtype="step", color="#d9534f", lw=1.6, linestyle=":", label=f"MC Base (Mean={np.mean(mc_base['ds_total']):.1f})")
    ax.set_title("DS Station 1 Hit Multiplicity (180 GeV $\\pi^+$)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Fired Bars in DS1", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    p2 = os.path.join(plot_dir, "tb_dispersion_subsystems.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    os.system(f"cp {p2} {os.path.join(art_dir, 'tb_dispersion_subsystems.png')}")
    plt.close()

    # ==============================================================================
    # PLOT 3: US Planes 0 to 4 Multiplicities
    # ==============================================================================
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharey=True, constrained_layout=True)
    for p in range(5):
        ax = axes[p]
        bins_p = np.arange(-0.5, 11.5, 1.0)
        ax.hist(d_intime["us_pl"][p], bins=bins_p, density=True, histtype="step", color="black", lw=2.0, label="Data")
        ax.hist(mc_full["us_pl"][p], bins=bins_p, density=True, histtype="step", color="#28a745", lw=1.8, label="MC Full")
        ax.hist(mc_base["us_pl"][p], bins=bins_p, density=True, histtype="step", color="#d9534f", lw=1.5, linestyle=":", label="MC Base")
        ax.set_title(f"US Plane {p}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Fired Bars", fontsize=10)
        if p == 0:
            ax.set_ylabel("Probability Density", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.5)

    p3 = os.path.join(plot_dir, "tb_dispersion_us_planes.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    os.system(f"cp {p3} {os.path.join(art_dir, 'tb_dispersion_us_planes.png')}")
    plt.close()

    print("[SUCCESS] All comparison plots successfully generated and archived.")


if __name__ == "__main__":
    main()
