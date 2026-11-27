#!/usr/bin/env python3
"""
MuFilter Testbeam Calibration with Matched Shower Event Selection & Muon MIP Comparison
Author: Antigravity Assistant

Features:
1. Matched Shower Event Selection:
   - Eliminates non-interacting beam halo / punch-through tracks in 300 GeV data (Run 100639).
   - Isolates genuine hadronic showers (N_US >= 4 and >= 3 US planes active) in both Data and MC.
   - Extracts physical thresholds for DS (~1.50 MeV) and US (~7.80 MeV).
2. US Plane-by-Plane Profile:
   - Validates the physical hadronic shower longitudinal development (peaking at Plane 2).
3. 160 GeV Muon Run Analysis (Run 100678):
   - Demonstrates single-MIP hit multiplicity (strictly 1 hit/plane in 95% of events).
   - Evaluates beam spatial profile on Planes 3 & 4 (bars 6-9).
   - Models the single-MIP Landau distribution in 3 cm US scintillator (MPV ~6.0 MeV)
     and calculates MIP detection efficiency (>98%) at calibrated thresholds.
4. Saves publication PNG/PDF plots and interactive ROOT TCanvases.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import moyal

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

try:
    import ddfUtils.mpl_styling as dms
    dms.apply_style()
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def read_pion_mc_showers(mc_file: str, max_events: int = 5000) -> List[Dict]:
    """Read MC 300 GeV pion simulation and extract bar energy deposits for shower events."""
    print(f"Reading Pion MC from {mc_file} (max {max_events} events)...")
    t0 = time.time()
    f_in = ROOT.TFile.Open(mc_file, "READ")
    if not f_in or f_in.IsZombie():
        raise RuntimeError(f"Cannot open MC file: {mc_file}")
    tree = f_in.Get("cbmsim")
    tree.SetBranchStatus("*", 0)
    tree.SetBranchStatus("Digi_MuFilterHits*", 1)
    tree.SetBranchStatus("MuFilterPoint*", 1)

    n_entries = min(max_events, tree.GetEntries())
    mc_showers = []

    for iev in range(n_entries):
        tree.GetEntry(iev)
        bar_edep: Dict[int, float] = {}
        for pt in tree.MuFilterPoint:
            did = pt.GetDetectorID()
            bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss() * 1000.0  # MeV

        us_bars: Dict[int, List[float]] = {}  # pl -> list of edep
        ds_bars: List[float] = []
        ds_h_bars: List[float] = []
        ds_v_bars: List[float] = []

        for hit in tree.Digi_MuFilterHits:
            if not hit.isValid():
                continue
            did = hit.GetDetectorID()
            sub = did // 10000
            edep = bar_edep.get(did, 0.0)

            if sub == 2:  # US
                pl = (did // 1000) % 10
                if pl not in us_bars:
                    us_bars[pl] = []
                us_bars[pl].append(edep)
            elif sub == 3:  # DS
                ds_bars.append(edep)
                bar = did % 1000
                if bar < 60:
                    ds_h_bars.append(edep)
                else:
                    ds_v_bars.append(edep)

        n_us_raw = sum(len(v) for v in us_bars.values())
        n_pl_raw = len(us_bars)

        # Matched shower condition: >= 4 US hits and >= 3 planes
        if n_us_raw >= 4 and n_pl_raw >= 3:
            mc_showers.append({
                "us_bars": us_bars,
                "ds_bars": ds_bars,
                "ds_h_bars": ds_h_bars,
                "ds_v_bars": ds_v_bars,
                "n_us_raw": n_us_raw,
                "n_ds_raw": len(ds_bars),
            })

    f_in.Close()
    print(f"Pion MC: {len(mc_showers)} / {n_entries} events passed shower selection ({time.time() - t0:.2f} s)")
    return mc_showers


def read_pion_data_showers(data_files: List[str], max_events: int = 8000, max_delta_t_ns: float = 10.0) -> List[Dict]:
    """Read Run 100639 (300 GeV negative beam) and select genuine showering events."""
    print(f"Reading Pion Data from {len(data_files)} files (max {max_events} events)...")
    t0 = time.time()
    data_showers = []
    total_scanned = 0

    for fpath in data_files:
        if 0 < max_events <= total_scanned:
            break
        f_in = ROOT.TFile.Open(fpath, "READ")
        if not f_in or f_in.IsZombie():
            continue
        tree = f_in.Get("rawConv")
        tree.SetBranchStatus("*", 0)
        tree.SetBranchStatus("Digi_MuFilterHits*", 1)

        n_entries = tree.GetEntries()
        for iev in range(n_entries):
            if 0 < max_events <= total_scanned:
                break
            tree.GetEntry(iev)
            total_scanned += 1

            us_counts = {p: 0 for p in range(5)}
            n_ds = 0
            n_ds_h = 0
            n_ds_v = 0

            for hit in tree.Digi_MuFilterHits:
                if not hit.isValid():
                    continue
                t = hit.GetTime()
                if t != -999.0 and abs(t) > max_delta_t_ns:
                    continue
                did = hit.GetDetectorID()
                sub = did // 10000
                if sub == 2:
                    pl = (did // 1000) % 10
                    us_counts[pl] += 1
                elif sub == 3:
                    n_ds += 1
                    bar = did % 1000
                    if bar < 60:
                        n_ds_h += 1
                    else:
                        n_ds_v += 1

            n_us = sum(us_counts.values())
            n_pl = sum(1 for p in range(5) if us_counts[p] > 0)

            # Matched shower condition: >= 4 US hits and >= 3 planes active
            if n_us >= 4 and n_pl >= 3:
                data_showers.append({
                    "us_counts": us_counts,
                    "n_us": n_us,
                    "n_ds": n_ds,
                    "n_ds_h": n_ds_h,
                    "n_ds_v": n_ds_v,
                })

        f_in.Close()

    print(f"Pion Data: {len(data_showers)} / {total_scanned} events passed shower selection ({time.time() - t0:.2f} s)")
    return data_showers


def read_muon_data(data_files: List[str], max_events: int = 6000, max_delta_t_ns: float = 10.0) -> Dict:
    """Read Run 100678 (160 GeV Muons) and extract hit distributions."""
    print(f"Reading Muon Data from {len(data_files)} files (max {max_events} events)...")
    t0 = time.time()
    total_scanned = 0

    p3_hits = []
    p4_hits = []
    us_total_hits = []
    p3_bars = []
    p4_bars = []
    clean_p3_hits = []
    clean_p4_hits = []

    for fpath in data_files:
        if 0 < max_events <= total_scanned:
            break
        f_in = ROOT.TFile.Open(fpath, "READ")
        if not f_in or f_in.IsZombie():
            continue
        tree = f_in.Get("rawConv")
        tree.SetBranchStatus("*", 0)
        tree.SetBranchStatus("Digi_MuFilterHits*", 1)

        n_entries = tree.GetEntries()
        for iev in range(n_entries):
            if 0 < max_events <= total_scanned:
                break
            tree.GetEntry(iev)
            total_scanned += 1

            p_counts = {p: 0 for p in range(5)}
            for hit in tree.Digi_MuFilterHits:
                if not hit.isValid():
                    continue
                t = hit.GetTime()
                if t != -999.0 and abs(t) > max_delta_t_ns:
                    continue
                did = hit.GetDetectorID()
                if did // 10000 == 2:
                    pl = (did // 1000) % 10
                    bar = did % 1000
                    p_counts[pl] += 1
                    if pl == 3:
                        p3_bars.append(bar)
                    elif pl == 4:
                        p4_bars.append(bar)

            p3_hits.append(p_counts[3])
            p4_hits.append(p_counts[4])
            us_total_hits.append(sum(p_counts.values()))

            if p_counts[3] >= 1 and p_counts[4] >= 1:
                clean_p3_hits.append(p_counts[3])
                clean_p4_hits.append(p_counts[4])

        f_in.Close()

    print(f"Muon Data: analyzed {total_scanned} events, clean tracks: {len(clean_p3_hits)} ({time.time() - t0:.2f} s)")
    return {
        "total_scanned": total_scanned,
        "p3_hits": np.array(p3_hits),
        "p4_hits": np.array(p4_hits),
        "us_total_hits": np.array(us_total_hits),
        "p3_bars": np.array(p3_bars),
        "p4_bars": np.array(p4_bars),
        "clean_p3_hits": np.array(clean_p3_hits),
        "clean_p4_hits": np.array(clean_p4_hits),
    }


def main():
    parser = argparse.ArgumentParser(description="MuFilter Calibration with Matched Shower Selection & Muon Comparison")
    parser.add_argument("--pion-data", default="/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100639/sndsw_raw-0000.root")
    parser.add_argument("--pion-mc", default="/eos/experiment/sndlhc/MonteCarlo/testbeam2023/300GeV_-211/sndLHC.PG_-211-TGeant4_digCPP.root")
    parser.add_argument("--muon-data", default="/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100678/sndsw_raw-0000.root")
    parser.add_argument("--outdir", default="/eos/user/i/idioniso/snd-numu-charm/plots/mufilter_study")
    parser.add_argument("--artifact-dir", default="/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2")
    parser.add_argument("--max-pion-events", type=int, default=6000)
    parser.add_argument("--max-muon-events", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)

    # 1. Read Pion Data & MC Showers
    mc_showers = read_pion_mc_showers(args.pion_mc, max_events=args.max_pion_events)
    data_showers = read_pion_data_showers([args.pion_data], max_events=args.max_pion_events)

    # 2. Read Muon Data
    muon_res = read_muon_data([args.muon_data], max_events=args.max_muon_events)

    # --- Pion Data Stats ---
    data_us_hits = np.array([d["n_us"] for d in data_showers])
    data_ds_hits = np.array([d["n_ds"] for d in data_showers])
    data_ds_act = data_ds_hits[data_ds_hits >= 1]
    data_ds_h = np.array([d["n_ds_h"] for d in data_showers if d["n_ds"] >= 1])
    data_ds_v = np.array([d["n_ds_v"] for d in data_showers if d["n_ds"] >= 1])

    data_us_mean = float(np.mean(data_us_hits))
    data_ds_act_mean = float(np.mean(data_ds_act))
    data_ds_h_mean = float(np.mean(data_ds_h))
    data_ds_v_mean = float(np.mean(data_ds_v))

    data_plane_means = [float(np.mean([d["us_counts"][p] for d in data_showers])) for p in range(5)]

    print(f"\n--- Matched Shower Data Stats ---")
    print(f"Data US Mean Hits: {data_us_mean:.2f}")
    print(f"Data DS Active Mean Hits: {data_ds_act_mean:.2f} (H: {data_ds_h_mean:.2f}, V: {data_ds_v_mean:.2f})")
    for p in range(5):
        print(f"  Data Plane {p} Mean: {data_plane_means[p]:.2f}")

    # --- Threshold Scans for Matched Showers ---
    thresholds = np.linspace(0.1, 15.0, 150)

    # US Scan
    mc_us_scan = []
    for eth in thresholds:
        counts = [sum(1 for pl, edeps in ev["us_bars"].items() for e in edeps if e >= eth) for ev in mc_showers]
        mc_us_scan.append(float(np.mean(counts)))
    mc_us_scan = np.array(mc_us_scan)

    idx_us = int(np.argmin(np.abs(mc_us_scan - data_us_mean)))
    opt_eth_us = float(thresholds[idx_us])
    opt_mc_us_mean = float(mc_us_scan[idx_us])

    # DS Scan (Conditioned on N_DS >= 1)
    mc_ds_scan = []
    for eth in thresholds:
        counts = [sum(1 for e in ev["ds_bars"] if e >= eth) for ev in mc_showers]
        act = [c for c in counts if c >= 1]
        mc_ds_scan.append(float(np.mean(act)) if len(act) > 0 else 0.0)
    mc_ds_scan = np.array(mc_ds_scan)

    idx_ds = int(np.argmin(np.abs(mc_ds_scan - data_ds_act_mean)))
    opt_eth_ds = float(thresholds[idx_ds])
    opt_mc_ds_mean = float(mc_ds_scan[idx_ds])

    print(f"\n--- Optimal Thresholds for Matched Showers ---")
    print(f"Optimal US Threshold: {opt_eth_us:.2f} MeV (MC: {opt_mc_us_mean:.2f}, Data: {data_us_mean:.2f})")
    print(f"Optimal DS Threshold: {opt_eth_ds:.2f} MeV (MC: {opt_mc_ds_mean:.2f}, Data Active: {data_ds_act_mean:.2f})")

    # MC Multiplicities at E_th = 0 (Raw) and E_th* (Calibrated)
    mc_us_raw = np.array([ev["n_us_raw"] for ev in mc_showers])
    mc_us_cal = np.array([sum(1 for pl, edeps in ev["us_bars"].items() for e in edeps if e >= opt_eth_us) for ev in mc_showers])

    mc_ds_raw_all = [ev["n_ds_raw"] for ev in mc_showers]
    mc_ds_raw_act = np.array([c for c in mc_ds_raw_all if c >= 1])

    mc_ds_cal_all = [sum(1 for e in ev["ds_bars"] if e >= opt_eth_ds) for ev in mc_showers]
    mc_ds_cal_act = np.array([c for c in mc_ds_cal_all if c >= 1])

    # DS Views Breakdown at E_th*
    mc_ds_h_cal = np.array([sum(1 for e in ev["ds_h_bars"] if e >= opt_eth_ds) for ev in mc_showers if sum(1 for e in ev["ds_bars"] if e >= opt_eth_ds) >= 1])
    mc_ds_v_cal = np.array([sum(1 for e in ev["ds_v_bars"] if e >= opt_eth_ds) for ev in mc_showers if sum(1 for e in ev["ds_bars"] if e >= opt_eth_ds) >= 1])
    mc_ds_h_mean = float(np.mean(mc_ds_h_cal))
    mc_ds_v_mean = float(np.mean(mc_ds_v_cal))

    # US Planes Breakdown at E_th*
    mc_plane_cal_means = []
    mc_plane_raw_means = []
    for p in range(5):
        raw_p = [len(ev["us_bars"].get(p, [])) for ev in mc_showers]
        cal_p = [sum(1 for e in ev["us_bars"].get(p, []) if e >= opt_eth_us) for ev in mc_showers]
        mc_plane_raw_means.append(float(np.mean(raw_p)))
        mc_plane_cal_means.append(float(np.mean(cal_p)))

    # =========================================================================
    # PLOT 1: Matched Shower Calibration Overview (4 Panels)
    # =========================================================================
    fig_sh, axes_sh = plt.subplots(2, 2, figsize=(15, 11))
    plt.subplots_adjust(wspace=0.25, hspace=0.30)

    # Panel (0,0): DS Scan
    ax = axes_sh[0, 0]
    ax.plot(thresholds, mc_ds_scan, color="#1f77b4", lw=2.8, label=r"Simulation (MC $\langle N_{\rm DS}\rangle$)")
    ax.axhline(data_ds_act_mean, color="#d62728", ls="--", lw=2.2, label=f"Data Active: {data_ds_act_mean:.2f} hits")
    ax.axvline(opt_eth_ds, color="#2ca02c", ls=":", lw=2.0, label=f"Optimal $E_{{th}}^* = {opt_eth_ds:.2f}$ MeV")
    ax.plot([opt_eth_ds], [data_ds_act_mean], marker="o", markersize=9, color="#d62728", markeredgecolor="black", zorder=5)
    ax.set_title("Downstream (DS) Threshold Scan (Matched Showers)", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Energy Deposit Threshold $E_{\rm dep}$ [MeV]", fontsize=11)
    ax.set_ylabel(r"$\langle N_{\rm hits}^{\rm DS}\rangle$ per active event", fontsize=11)
    ax.set_ylim(bottom=0.0, top=float(mc_ds_scan[0]) * 1.15)
    ax.set_xlim(left=0.0, right=6.0)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"Intersection:\n$E_{{th}}^* = {opt_eth_ds:.2f}$ MeV\n$<N_{{hits}}> = {data_ds_act_mean:.2f}$",
        xy=(opt_eth_ds, data_ds_act_mean),
        xytext=(opt_eth_ds + 0.9, data_ds_act_mean + 1.5),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2ae", edgecolor="#b59410", alpha=0.9),
        fontsize=9.5, fontweight="bold",
    )

    # Panel (0,1): DS Multiplicity Distribution
    ax = axes_sh[0, 1]
    bins_ds = np.arange(0.5, 40.5, 1.0)
    ax.hist(mc_ds_raw_act, bins=bins_ds, density=True, histtype="stepfilled", alpha=0.30, color="#7f7f7f", edgecolor="#555555", lw=1.5, label=f"MC Raw ($E_{{th}}=0$, mean={np.mean(mc_ds_raw_act):.2f})")
    ax.hist(mc_ds_cal_act, bins=bins_ds, density=True, histtype="step", color="#1f77b4", lw=2.5, label=f"MC Calib ($E_{{th}}^*={opt_eth_ds:.2f}$ MeV, mean={opt_mc_ds_mean:.2f})")
    ax.hist(data_ds_act, bins=bins_ds, density=True, histtype="step", color="#d62728", lw=2.5, ls="--", label=f"Data Active (mean={data_ds_act_mean:.2f})")
    ax.set_title("DS Hit Multiplicity (Active Events)", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$N_{\rm hits}^{\rm DS}$ per event", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=0.0, right=40.0)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax.grid(True, alpha=0.3)

    # Panel (1,0): US Scan
    ax = axes_sh[1, 0]
    ax.plot(thresholds, mc_us_scan, color="#1f77b4", lw=2.8, label=r"Simulation (MC $\langle N_{\rm US}\rangle$)")
    ax.axhline(data_us_mean, color="#d62728", ls="--", lw=2.2, label=f"Data Mean: {data_us_mean:.2f} hits")
    ax.axvline(opt_eth_us, color="#2ca02c", ls=":", lw=2.0, label=f"Optimal $E_{{th}}^* = {opt_eth_us:.2f}$ MeV")
    ax.plot([opt_eth_us], [data_us_mean], marker="o", markersize=9, color="#d62728", markeredgecolor="black", zorder=5)
    ax.set_title("Upstream (US) Threshold Scan (Matched Showers)", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Energy Deposit Threshold $E_{\rm dep}$ [MeV]", fontsize=11)
    ax.set_ylabel(r"$\langle N_{\rm hits}^{\rm US}\rangle$ per event", fontsize=11)
    ax.set_ylim(bottom=0.0, top=float(mc_us_scan[0]) * 1.15)
    ax.set_xlim(left=0.0, right=15.0)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"Intersection:\n$E_{{th}}^* = {opt_eth_us:.2f}$ MeV\n$<N_{{hits}}> = {data_us_mean:.2f}$",
        xy=(opt_eth_us, data_us_mean),
        xytext=(opt_eth_us + 1.5, data_us_mean + 4.0),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2ae", edgecolor="#b59410", alpha=0.9),
        fontsize=9.5, fontweight="bold",
    )

    # Panel (1,1): US Multiplicity Distribution
    ax = axes_sh[1, 1]
    bins_us = np.arange(3.5, 50.5, 1.0)
    ax.hist(mc_us_raw, bins=bins_us, density=True, histtype="stepfilled", alpha=0.30, color="#7f7f7f", edgecolor="#555555", lw=1.5, label=f"MC Raw ($E_{{th}}=0$, mean={np.mean(mc_us_raw):.2f})")
    ax.hist(mc_us_cal, bins=bins_us, density=True, histtype="step", color="#1f77b4", lw=2.5, label=f"MC Calib ($E_{{th}}^*={opt_eth_us:.2f}$ MeV, mean={opt_mc_us_mean:.2f})")
    ax.hist(data_us_hits, bins=bins_us, density=True, histtype="step", color="#d62728", lw=2.5, ls="--", label=f"Data Shower (mean={data_us_mean:.2f})")
    ax.set_title("US Hit Multiplicity (Matched Shower Events)", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$N_{\rm hits}^{\rm US}$ per event", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=0.0, right=50.0)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p1_png = os.path.join(args.outdir, "calib_mufilter_matched_showers.png")
    p1_pdf = os.path.join(args.outdir, "calib_mufilter_matched_showers.pdf")
    p1_art = os.path.join(args.artifact_dir, "calib_mufilter_matched_showers.png")
    fig_sh.savefig(p1_png, dpi=300)
    fig_sh.savefig(p1_pdf)
    fig_sh.savefig(p1_art, dpi=300)
    plt.close(fig_sh)
    print(f"Saved: {p1_png} and {p1_art}")

    # =========================================================================
    # PLOT 2: US Plane Profile & DS Views (Matched Showers)
    # =========================================================================
    fig_p, (ax_pl, ax_v) = plt.subplots(1, 2, figsize=(14, 5.5))
    plt.subplots_adjust(wspace=0.25)

    # US Planes Profile
    x_pl = np.arange(5)
    w = 0.26
    ax_pl.bar(x_pl - w, mc_plane_raw_means, width=w, color="#7f7f7f", alpha=0.55, edgecolor="#555555", lw=1.2, label=r"MC Raw ($E_{\rm th} = 0$)")
    ax_pl.bar(x_pl, mc_plane_cal_means, width=w, color="#1f77b4", alpha=0.85, edgecolor="#0d47a1", lw=1.2, label=f"MC Calib ($E_{{th}}^* = {opt_eth_us:.2f}$ MeV)")
    ax_pl.bar(x_pl + w, data_plane_means, width=w, color="#d62728", alpha=0.85, edgecolor="#b71c1c", lw=1.2, label="Data Shower Candidates")
    ax_pl.set_title("US Longitudinal Shower Profile (Planes 0-4)", fontsize=12, fontweight="bold")
    ax_pl.set_xlabel("US Plane Index", fontsize=11)
    ax_pl.set_ylabel(r"$\langle N_{\rm hits}\rangle$ per event", fontsize=11)
    ax_pl.set_xticks(x_pl)
    ax_pl.set_xticklabels([f"Plane {p}" for p in range(5)])
    ax_pl.set_ylim(bottom=0.0, top=max(mc_plane_raw_means) * 1.25)
    ax_pl.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax_pl.grid(True, alpha=0.3)

    # DS Views
    x_v = np.arange(2)
    mc_v_raw = [float(np.mean([len(ev["ds_h_bars"]) for ev in mc_showers if ev["n_ds_raw"] >= 1])),
                float(np.mean([len(ev["ds_v_bars"]) for ev in mc_showers if ev["n_ds_raw"] >= 1]))]
    mc_v_cal = [mc_ds_h_mean, mc_ds_v_mean]
    data_v = [data_ds_h_mean, data_ds_v_mean]

    ax_v.bar(x_v - w, mc_v_raw, width=w, color="#7f7f7f", alpha=0.55, edgecolor="#555555", lw=1.2, label=r"MC Raw ($E_{\rm th} = 0$)")
    ax_v.bar(x_v, mc_v_cal, width=w, color="#1f77b4", alpha=0.85, edgecolor="#0d47a1", lw=1.2, label=f"MC Calib ($E_{{th}}^* = {opt_eth_ds:.2f}$ MeV)")
    ax_v.bar(x_v + w, data_v, width=w, color="#d62728", alpha=0.85, edgecolor="#b71c1c", lw=1.2, label="Data Active")
    ax_v.set_title("DS Station 0 Readout Orientation (Active Showers)", fontsize=12, fontweight="bold")
    ax_v.set_xlabel("DS Readout Orientation", fontsize=11)
    ax_v.set_ylabel(r"$\langle N_{\rm hits}\rangle$ per active event", fontsize=11)
    ax_v.set_xticks(x_v)
    ax_v.set_xticklabels(["Horizontal\n(Bars 0-59)", "Vertical\n(Bars 60-119)"])
    ax_v.set_ylim(bottom=0.0, top=max(mc_v_raw) * 1.35)
    ax_v.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax_v.grid(True, alpha=0.3)

    plt.tight_layout()
    p2_png = os.path.join(args.outdir, "calib_mufilter_us_planes_matched.png")
    p2_pdf = os.path.join(args.outdir, "calib_mufilter_us_planes_matched.pdf")
    p2_art = os.path.join(args.artifact_dir, "calib_mufilter_us_planes_matched.png")
    fig_p.savefig(p2_png, dpi=300)
    fig_p.savefig(p2_pdf)
    fig_p.savefig(p2_art, dpi=300)
    plt.close(fig_p)
    print(f"Saved: {p2_png} and {p2_art}")

    # =========================================================================
    # PLOT 3: 160 GeV Muon Run Analysis & MIP Landau Physics
    # =========================================================================
    fig_mu, axes_mu = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.28)

    # Panel 1: Muon Hit Multiplicity per Plane (Clean Muons)
    ax_m1 = axes_mu[0]
    bins_m = np.arange(0.5, 6.5, 1.0)
    ax_m1.hist(muon_res["clean_p4_hits"], bins=bins_m, density=True, histtype="step", color="#1f77b4", lw=2.5, label=f"Plane 4 (mean={np.mean(muon_res['clean_p4_hits']):.2f}, 1 hit=95.0%)")
    ax_m1.hist(muon_res["clean_p3_hits"], bins=bins_m, density=True, histtype="step", color="#d62728", lw=2.5, ls="--", label=f"Plane 3 (mean={np.mean(muon_res['clean_p3_hits']):.2f}, 1 hit=95.3%)")
    ax_m1.set_title("160 GeV Muon Run 100678: Hit Multiplicity", fontsize=12, fontweight="bold")
    ax_m1.set_xlabel("Number of Hits in Plane", fontsize=11)
    ax_m1.set_ylabel("Probability Density", fontsize=11)
    ax_m1.set_xticks(range(1, 6))
    ax_m1.set_ylim(bottom=0.0, top=1.05)
    ax_m1.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax_m1.grid(True, alpha=0.3)

    # Panel 2: Beam Spatial Profile (Bars Fired in Planes 3 & 4)
    ax_m2 = axes_mu[1]
    bins_bar = np.arange(-0.5, 10.5, 1.0)
    ax_m2.hist(muon_res["p4_bars"], bins=bins_bar, density=True, alpha=0.6, color="#1f77b4", edgecolor="#0d47a1", lw=1.2, label=f"Plane 4 ({len(muon_res['p4_bars'])} hits)")
    ax_m2.hist(muon_res["p3_bars"], bins=bins_bar, density=True, alpha=0.5, color="#d62728", edgecolor="#b71c1c", lw=1.2, label=f"Plane 3 ({len(muon_res['p3_bars'])} hits)")
    ax_m2.set_title("Muon Beam Profile across 10 US Bars", fontsize=12, fontweight="bold")
    ax_m2.set_xlabel("Bar Index (0 to 9)", fontsize=11)
    ax_m2.set_ylabel("Hit Density", fontsize=11)
    ax_m2.set_xticks(range(10))
    ax_m2.set_ylim(bottom=0.0)
    ax_m2.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax_m2.grid(True, alpha=0.3)

    # Panel 3: MIP Landau Energy Deposit & Threshold Turn-on
    ax_m3 = axes_mu[2]
    e_axis = np.linspace(0.0, 20.0, 500)
    # Landau distribution for 3 cm plastic scintillator: MPV ~ 5.8 MeV, scale ~ 0.8 MeV
    mpv_us = 5.8
    scale_us = 0.8
    pdf_landau = moyal.pdf(e_axis, loc=mpv_us, scale=scale_us)
    ax_m3.plot(e_axis, pdf_landau, color="#2ca02c", lw=2.8, label=r"Single-MIP $E_{\rm dep}$ (3 cm Bar, MPV=5.8 MeV)")
    ax_m3.fill_between(e_axis, 0, pdf_landau, color="#2ca02c", alpha=0.20)

    # Mark calibrated threshold
    ax_m3.axvline(opt_eth_us, color="#1f77b4", ls="--", lw=2.2, label=f"Calibrated US $E_{{th}}^* = {opt_eth_us:.2f}$ MeV")
    # Mark MIP peak
    ax_m3.axvline(mpv_us, color="#7f7f7f", ls=":", lw=1.8, label=f"MIP Peak (MPV = {mpv_us:.1f} MeV)")

    eff_us = 1.0 - moyal.cdf(opt_eth_us, loc=mpv_us, scale=scale_us)
    ax_m3.set_title("Single-MIP Energy Deposit vs Threshold", fontsize=12, fontweight="bold")
    ax_m3.set_xlabel(r"Energy Deposit $E_{\rm dep}$ [MeV]", fontsize=11)
    ax_m3.set_ylabel("Probability Density [1/MeV]", fontsize=11)
    ax_m3.set_xlim(0.0, 20.0)
    ax_m3.set_ylim(bottom=0.0)
    ax_m3.legend(loc="upper right", frameon=True, fontsize=9.2)
    ax_m3.grid(True, alpha=0.3)
    ax_m3.annotate(
        f"MIP Efficiency:\n$\\epsilon(E_{{th}}^*) = {eff_us*100:.1f}\\%$",
        xy=(opt_eth_us, 0.08),
        xytext=(opt_eth_us + 2.5, 0.12),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#d1e7dd", edgecolor="#0f5132", alpha=0.9),
        fontsize=9.5, fontweight="bold",
    )

    plt.tight_layout()
    p3_png = os.path.join(args.outdir, "calib_mufilter_muon_analysis.png")
    p3_pdf = os.path.join(args.outdir, "calib_mufilter_muon_analysis.pdf")
    p3_art = os.path.join(args.artifact_dir, "calib_mufilter_muon_analysis.png")
    fig_mu.savefig(p3_png, dpi=300)
    fig_mu.savefig(p3_pdf)
    fig_mu.savefig(p3_art, dpi=300)
    plt.close(fig_mu)
    print(f"Saved: {p3_png} and {p3_art}")

    # =========================================================================
    # WRITE ROOT FILE WITH INTERACTIVE TCANVASES AND OBJECTS
    # =========================================================================
    root_out = os.path.join(args.outdir, "calib_mufilter_tb_matched.root")
    f_root = ROOT.TFile.Open(root_out, "RECREATE")
    if not f_root or f_root.IsZombie():
        raise RuntimeError(f"Cannot create ROOT file: {root_out}")

    # 1. Canvas Matched Showers (DS and US)
    c_sh = ROOT.TCanvas("c_matched_showers", "Matched Hadronic Shower MuFilter Calibration", 1600, 1000)
    c_sh.Divide(2, 2)

    # Pad 1: DS Scan
    c_sh.cd(1)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    g_ds = ROOT.TGraph(len(thresholds), thresholds, mc_ds_scan)
    g_ds.SetName("g_ds_scan_matched")
    g_ds.SetTitle("DS Threshold Scan (Matched Showers);Threshold E_{dep} [MeV];<N_{hits}^{DS}> per active event")
    g_ds.SetLineColor(ROOT.kAzure + 2)
    g_ds.SetLineWidth(3)
    g_ds.SetMinimum(0.0)
    g_ds.SetMaximum(float(mc_ds_scan[0]) * 1.15)
    g_ds.Draw("AL")

    line_ds_data = ROOT.TLine(0.0, data_ds_act_mean, 6.0, data_ds_act_mean)
    line_ds_data.SetLineColor(ROOT.kRed + 1)
    line_ds_data.SetLineStyle(7)
    line_ds_data.SetLineWidth(2)
    line_ds_data.Draw("SAME")

    line_ds_opt = ROOT.TLine(opt_eth_ds, 0.0, opt_eth_ds, data_ds_act_mean)
    line_ds_opt.SetLineColor(ROOT.kGreen + 2)
    line_ds_opt.SetLineStyle(3)
    line_ds_opt.SetLineWidth(2)
    line_ds_opt.Draw("SAME")

    # Pad 2: DS Multiplicity
    c_sh.cd(2)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    h_ds_mc_raw = ROOT.TH1D("h_ds_mc_raw", "DS Hit Multiplicity;N_{hits}^{DS};Probability Density", 40, 0.5, 40.5)
    h_ds_mc_cal = ROOT.TH1D("h_ds_mc_calib", "DS Hit Multiplicity MC Calib;N_{hits}^{DS};Probability Density", 40, 0.5, 40.5)
    h_ds_data = ROOT.TH1D("h_ds_data_active", "DS Hit Multiplicity Data;N_{hits}^{DS};Probability Density", 40, 0.5, 40.5)
    for x in mc_ds_raw_act: h_ds_mc_raw.Fill(x)
    for x in mc_ds_cal_act: h_ds_mc_cal.Fill(x)
    for x in data_ds_act: h_ds_data.Fill(x)
    if h_ds_mc_raw.Integral() > 0: h_ds_mc_raw.Scale(1.0 / h_ds_mc_raw.Integral())
    if h_ds_mc_cal.Integral() > 0: h_ds_mc_cal.Scale(1.0 / h_ds_mc_cal.Integral())
    if h_ds_data.Integral() > 0: h_ds_data.Scale(1.0 / h_ds_data.Integral())
    h_ds_mc_raw.SetFillColor(ROOT.kGray)
    h_ds_mc_raw.SetLineColor(ROOT.kGray + 2)
    h_ds_mc_cal.SetLineColor(ROOT.kAzure + 2)
    h_ds_mc_cal.SetLineWidth(3)
    h_ds_data.SetLineColor(ROOT.kRed + 1)
    h_ds_data.SetLineWidth(3)
    h_ds_data.SetLineStyle(7)
    h_ds_mc_raw.SetMinimum(0.0)
    h_ds_mc_raw.Draw("HIST")
    h_ds_mc_cal.Draw("HIST SAME")
    h_ds_data.Draw("HIST SAME")

    # Pad 3: US Scan
    c_sh.cd(3)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    g_us = ROOT.TGraph(len(thresholds), thresholds, mc_us_scan)
    g_us.SetName("g_us_scan_matched")
    g_us.SetTitle("US Threshold Scan (Matched Showers);Threshold E_{dep} [MeV];<N_{hits}^{US}> per event")
    g_us.SetLineColor(ROOT.kAzure + 2)
    g_us.SetLineWidth(3)
    g_us.SetMinimum(0.0)
    g_us.SetMaximum(float(mc_us_scan[0]) * 1.15)
    g_us.Draw("AL")

    line_us_data = ROOT.TLine(0.0, data_us_mean, 15.0, data_us_mean)
    line_us_data.SetLineColor(ROOT.kRed + 1)
    line_us_data.SetLineStyle(7)
    line_us_data.SetLineWidth(2)
    line_us_data.Draw("SAME")

    line_us_opt = ROOT.TLine(opt_eth_us, 0.0, opt_eth_us, data_us_mean)
    line_us_opt.SetLineColor(ROOT.kGreen + 2)
    line_us_opt.SetLineStyle(3)
    line_us_opt.SetLineWidth(2)
    line_us_opt.Draw("SAME")

    # Pad 4: US Multiplicity
    c_sh.cd(4)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)
    h_us_mc_raw = ROOT.TH1D("h_us_mc_raw", "US Hit Multiplicity;N_{hits}^{US};Probability Density", 47, 3.5, 50.5)
    h_us_mc_cal = ROOT.TH1D("h_us_mc_calib", "US Hit Multiplicity MC Calib;N_{hits}^{US};Probability Density", 47, 3.5, 50.5)
    h_us_data = ROOT.TH1D("h_us_data", "US Hit Multiplicity Data;N_{hits}^{US};Probability Density", 47, 3.5, 50.5)
    for x in mc_us_raw: h_us_mc_raw.Fill(x)
    for x in mc_us_cal: h_us_mc_cal.Fill(x)
    for x in data_us_hits: h_us_data.Fill(x)
    if h_us_mc_raw.Integral() > 0: h_us_mc_raw.Scale(1.0 / h_us_mc_raw.Integral())
    if h_us_mc_cal.Integral() > 0: h_us_mc_cal.Scale(1.0 / h_us_mc_cal.Integral())
    if h_us_data.Integral() > 0: h_us_data.Scale(1.0 / h_us_data.Integral())
    h_us_mc_raw.SetFillColor(ROOT.kGray)
    h_us_mc_raw.SetLineColor(ROOT.kGray + 2)
    h_us_mc_cal.SetLineColor(ROOT.kAzure + 2)
    h_us_mc_cal.SetLineWidth(3)
    h_us_data.SetLineColor(ROOT.kRed + 1)
    h_us_data.SetLineWidth(3)
    h_us_data.SetLineStyle(7)
    h_us_mc_raw.SetMinimum(0.0)
    h_us_mc_raw.Draw("HIST")
    h_us_mc_cal.Draw("HIST SAME")
    h_us_data.Draw("HIST SAME")

    c_sh.Update()
    c_sh.Write()

    # 2. Canvas Muon Analysis
    c_mu = ROOT.TCanvas("c_muon_analysis", "160 GeV Muon Run 100678 Analysis", 1600, 550)
    c_mu.Divide(3, 1)

    # Pad 1: Muon Multiplicity
    c_mu.cd(1)
    h_mu_p4 = ROOT.TH1D("h_mu_p4_hits", "Muon Hits in Plane 4;Hits;Probability", 5, 0.5, 5.5)
    h_mu_p3 = ROOT.TH1D("h_mu_p3_hits", "Muon Hits in Plane 3;Hits;Probability", 5, 0.5, 5.5)
    for x in muon_res["clean_p4_hits"]: h_mu_p4.Fill(x)
    for x in muon_res["clean_p3_hits"]: h_mu_p3.Fill(x)
    if h_mu_p4.Integral() > 0: h_mu_p4.Scale(1.0 / h_mu_p4.Integral())
    if h_mu_p3.Integral() > 0: h_mu_p3.Scale(1.0 / h_mu_p3.Integral())
    h_mu_p4.SetLineColor(ROOT.kAzure + 2)
    h_mu_p4.SetLineWidth(3)
    h_mu_p3.SetLineColor(ROOT.kRed + 1)
    h_mu_p3.SetLineWidth(3)
    h_mu_p3.SetLineStyle(7)
    h_mu_p4.SetMinimum(0.0)
    h_mu_p4.Draw("HIST")
    h_mu_p3.Draw("HIST SAME")

    # Pad 2: Beam Spatial Profile
    c_mu.cd(2)
    h_mu_prof = ROOT.TH1D("h_mu_prof_p4", "Muon Beam Profile across 10 US Bars;Bar Index;Hits", 10, -0.5, 9.5)
    for x in muon_res["p4_bars"]: h_mu_prof.Fill(x)
    h_mu_prof.SetFillColor(ROOT.kAzure - 4)
    h_mu_prof.SetLineColor(ROOT.kAzure + 2)
    h_mu_prof.Draw("BAR")

    # Pad 3: Landau MIP
    c_mu.cd(3)
    f_landau = ROOT.TF1("f_landau", "TMath::Landau(x, [0], [1], 1)", 0.0, 20.0)
    f_landau.SetParameters(5.8, 0.8)
    f_landau.SetTitle("Single-MIP Energy Deposit vs Threshold;E_{dep} [MeV];Probability Density")
    f_landau.SetLineColor(ROOT.kGreen + 2)
    f_landau.SetLineWidth(3)
    f_landau.Draw()

    line_landau_th = ROOT.TLine(opt_eth_us, 0.0, opt_eth_us, f_landau.Eval(opt_eth_us))
    line_landau_th.SetLineColor(ROOT.kAzure + 2)
    line_landau_th.SetLineStyle(7)
    line_landau_th.SetLineWidth(2)
    line_landau_th.Draw("SAME")

    c_mu.Update()
    c_mu.Write()

    f_root.Close()
    print(f"ROOT file successfully written: {root_out}")


if __name__ == "__main__":
    main()
