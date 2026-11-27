#!/usr/bin/env python3
"""
MuFilter Calibration: 180 GeV pi+ with 3 Walls Setup (Run 100671 vs 180GeV_211 MC)
Selection: N_US >= 8 applied identically in both Data and Simulation.
Author: Antigravity Assistant
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

try:
    import ddfUtils.mpl_styling as dms
    dms.apply_style()
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


import pickle

def read_mc_180gev(mc_path: str, max_events: int = 5000, min_us_hits: int = 8) -> List[Dict]:
    """Read 180 GeV pi+ MC with 3 walls and extract energy deposits."""
    cache_path = f"/tmp/mc_180gev_{max_events}_{min_us_hits}.pkl"
    if os.path.exists(cache_path):
        print(f"Loading cached MC from {cache_path}...", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Reading MC from {mc_path} (max {max_events} events, min_us_hits={min_us_hits})...", flush=True)
    t0 = time.time()
    f_in = ROOT.TFile.Open(mc_path, "READ")
    if not f_in or f_in.IsZombie():
        raise RuntimeError(f"Cannot open MC file: {mc_path}")
    tree = f_in.Get("cbmsim")
    tree.SetBranchStatus("*", 0)
    tree.SetBranchStatus("Digi_MuFilterHits*", 1)
    tree.SetBranchStatus("MuFilterPoint*", 1)

    n_entries = min(max_events, tree.GetEntries())
    events = []

    for iev in range(n_entries):
        tree.GetEntry(iev)
        bar_edep: Dict[int, float] = {}
        for pt in tree.MuFilterPoint:
            did = pt.GetDetectorID()
            bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss() * 1000.0  # MeV

        us_bars: Dict[int, List[float]] = {}
        ds_bars: List[float] = []
        ds_h_bars: List[float] = []
        ds_v_bars: List[float] = []

        for hit in tree.Digi_MuFilterHits:
            if not hit.isValid():
                continue
            did = hit.GetDetectorID()
            sub = did // 10000
            edep = bar_edep.get(did, 0.0)
            if sub == 2:
                pl = (did // 1000) % 10
                if pl not in us_bars:
                    us_bars[pl] = []
                us_bars[pl].append(edep)
            elif sub == 3:
                ds_bars.append(edep)
                bar = did % 1000
                if bar < 60:
                    ds_h_bars.append(edep)
                else:
                    ds_v_bars.append(edep)

        n_us_raw = sum(len(v) for v in us_bars.values())
        # Apply selection: N_US >= min_us_hits
        if n_us_raw >= min_us_hits:
            events.append({
                "us_bars": us_bars,
                "ds_bars": ds_bars,
                "ds_h_bars": ds_h_bars,
                "ds_v_bars": ds_v_bars,
                "n_us_raw": n_us_raw,
                "n_ds_raw": len(ds_bars),
            })

    f_in.Close()
    print(f"MC: {len(events)} / {n_entries} events selected ({len(events)/n_entries*100:.1f}%) in {time.time()-t0:.2f} s", flush=True)
    with open(cache_path, "wb") as f:
        pickle.dump(events, f)
    return events


def read_data_100671(data_files: List[str], max_events: int = 8000, min_us_hits: int = 8, min_scifi_hits: int = 50, max_delta_t_ns: float = 10.0) -> List[Dict]:
    """Read Run 100672 (pi+/180GeV, 3 walls) requiring genuine beam shower in SciFi (N_SciFi >= min_scifi_hits)."""
    cache_path = f"/tmp/data_clean_{os.path.basename(data_files[0])}_{max_events}_{min_us_hits}_{min_scifi_hits}.pkl"
    if os.path.exists(cache_path):
        print(f"Loading cached Data from {cache_path}...", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Reading Data from {len(data_files)} files (max {max_events} events, min_us_hits={min_us_hits}, min_scifi_hits={min_scifi_hits})...", flush=True)
    t0 = time.time()
    events = []
    total_scanned = 0

    for fpath in data_files:
        if 0 < max_events <= len(events):
            break
        f_in = ROOT.TFile.Open(fpath, "READ")
        if not f_in or f_in.IsZombie():
            continue
        tree = f_in.Get("rawConv")
        tree.SetBranchStatus("*", 0)
        tree.SetBranchStatus("Digi_MuFilterHits*", 1)
        tree.SetBranchStatus("Digi_ScifiHits*", 1)

        n_entries = tree.GetEntries()
        for iev in range(n_entries):
            if 0 < max_events <= len(events):
                break
            tree.GetEntry(iev)
            total_scanned += 1

            # Require genuine beam particle shower in SciFi to reject DAQ board saturation bursts and dark count triggers
            if len(tree.Digi_ScifiHits) < min_scifi_hits:
                continue

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
            # Apply identical selection: N_US >= min_us_hits
            if n_us >= min_us_hits:
                events.append({
                    "us_counts": us_counts,
                    "n_us": n_us,
                    "n_ds": n_ds,
                    "n_ds_h": n_ds_h,
                    "n_ds_v": n_ds_v,
                })

        f_in.Close()

    print(f"Data: {len(events)} / {total_scanned} events selected ({len(events)/total_scanned*100:.1f}%) in {time.time()-t0:.2f} s", flush=True)
    with open(cache_path, "wb") as f:
        pickle.dump(events, f)
    return events


def main():
    parser = argparse.ArgumentParser(description="MuFilter Calibration: 180 GeV 3 Walls Setup")
    parser.add_argument("--mc-file", default="/eos/experiment/sndlhc/MonteCarlo/testbeam2023/180GeV_211/sndLHC.PG_211-TGeant4_digCPP.root")
    parser.add_argument("--data-file", default="/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100672/sndsw_raw-0000.root")
    parser.add_argument("--min-us-hits", type=int, default=8)
    parser.add_argument("--min-scifi-hits", type=int, default=50)
    parser.add_argument("--max-mc-events", type=int, default=4000)
    parser.add_argument("--max-data-events", type=int, default=8000)
    parser.add_argument("--outdir", default="/eos/user/i/idioniso/snd-numu-charm/plots/mufilter_study")
    parser.add_argument("--artifact-dir", default="/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)

    # 1. Read MC & Data with N_US >= min_us_hits
    mc_events = read_mc_180gev(args.mc_file, max_events=args.max_mc_events, min_us_hits=args.min_us_hits)
    data_events = read_data_100671([args.data_file], max_events=args.max_data_events, min_us_hits=args.min_us_hits, min_scifi_hits=args.min_scifi_hits)

    if len(mc_events) == 0 or len(data_events) == 0:
        raise RuntimeError("No events passed selection!")

    # 2. Extract Data Multiplicities
    data_us_hits = np.array([d["n_us"] for d in data_events])
    data_ds_hits = np.array([d["n_ds"] for d in data_events])
    data_ds_act = data_ds_hits[data_ds_hits >= 1]
    data_ds_h = np.array([d["n_ds_h"] for d in data_events if d["n_ds"] >= 1])
    data_ds_v = np.array([d["n_ds_v"] for d in data_events if d["n_ds"] >= 1])

    data_us_mean = float(np.mean(data_us_hits))
    data_ds_act_mean = float(np.mean(data_ds_act)) if len(data_ds_act) > 0 else 0.0
    data_ds_h_mean = float(np.mean(data_ds_h)) if len(data_ds_h) > 0 else 0.0
    data_ds_v_mean = float(np.mean(data_ds_v)) if len(data_ds_v) > 0 else 0.0
    data_plane_means = [float(np.mean([d["us_counts"][p] for d in data_events])) for p in range(5)]

    print(f"\n=== Selected Events Stats (N_US >= {args.min_us_hits}) ===")
    print(f"Data US Mean: {data_us_mean:.2f} hits (median: {np.median(data_us_hits):.1f})")
    print(f"Data DS Active Mean: {data_ds_act_mean:.2f} hits (H: {data_ds_h_mean:.2f}, V: {data_ds_v_mean:.2f})")
    for p in range(5):
        print(f"  Plane {p} mean: {data_plane_means[p]:.2f} hits")

    # 3. Threshold Scan on MC (N_US >= min_us_hits applied at each threshold for consistency)
    thresholds = np.linspace(0.1, 15.0, 150)

    # US Threshold Scan (consistently requiring N_US >= min_us_hits post-threshold)
    mc_us_scan = []
    for eth in thresholds:
        counts = [sum(1 for pl, edeps in ev["us_bars"].items() for e in edeps if e >= eth) for ev in mc_events]
        act = [c for c in counts if c >= args.min_us_hits]
        mc_us_scan.append(float(np.mean(act)) if len(act) > 0 else 0.0)
    mc_us_scan = np.array(mc_us_scan)

    idx_us = int(np.argmin(np.abs(mc_us_scan - data_us_mean)))
    opt_eth_us = float(thresholds[idx_us])
    opt_mc_us_mean = float(mc_us_scan[idx_us])

    # DS Threshold Scan (Conditioned on N_DS >= 1)
    mc_ds_scan = []
    for eth in thresholds:
        counts = [sum(1 for e in ev["ds_bars"] if e >= eth) for ev in mc_events]
        act = [c for c in counts if c >= 1]
        mc_ds_scan.append(float(np.mean(act)) if len(act) > 0 else 0.0)
    mc_ds_scan = np.array(mc_ds_scan)

    idx_ds = int(np.argmin(np.abs(mc_ds_scan - data_ds_act_mean)))
    opt_eth_ds = float(thresholds[idx_ds])
    opt_mc_ds_mean = float(mc_ds_scan[idx_ds])

    print(f"\n=== Calibration Results (180 GeV pi+, 3 Walls) ===")
    print(f"Optimal US Threshold: {opt_eth_us:.2f} MeV (MC: {opt_mc_us_mean:.2f}, Data: {data_us_mean:.2f})")
    print(f"Optimal DS Threshold: {opt_eth_ds:.2f} MeV (MC: {opt_mc_ds_mean:.2f}, Data Active: {data_ds_act_mean:.2f})")

    # MC Multiplicities at E_th = 0 and E_th*
    mc_us_raw = np.array([ev["n_us_raw"] for ev in mc_events])
    mc_us_cal = np.array([sum(1 for pl, edeps in ev["us_bars"].items() for e in edeps if e >= opt_eth_us) for ev in mc_events])
    mc_us_cal = mc_us_cal[mc_us_cal >= args.min_us_hits]

    mc_ds_raw_act = np.array([ev["n_ds_raw"] for ev in mc_events if ev["n_ds_raw"] >= 1])
    mc_ds_cal_act = np.array([sum(1 for e in ev["ds_bars"] if e >= opt_eth_ds) for ev in mc_events if sum(1 for e in ev["ds_bars"] if e >= opt_eth_ds) >= 1])

    # Plane means
    mc_events_cal_sel = [ev for ev in mc_events if sum(1 for pl, edeps in ev["us_bars"].items() for e in edeps if e >= opt_eth_us) >= args.min_us_hits]
    mc_plane_cal_means = []
    mc_plane_raw_means = []
    for p in range(5):
        raw_p = [len(ev["us_bars"].get(p, [])) for ev in mc_events]
        cal_p = [sum(1 for e in ev["us_bars"].get(p, []) if e >= opt_eth_us) for ev in mc_events_cal_sel]
        mc_plane_raw_means.append(float(np.mean(raw_p)))
        mc_plane_cal_means.append(float(np.mean(cal_p)))

    # DS Views Breakdown
    mc_ds_h_cal = np.array([sum(1 for e in ev["ds_h_bars"] if e >= opt_eth_ds) for ev in mc_events if sum(1 for e in ev["ds_bars"] if e >= opt_eth_ds) >= 1])
    mc_ds_v_cal = np.array([sum(1 for e in ev["ds_v_bars"] if e >= opt_eth_ds) for ev in mc_events if sum(1 for e in ev["ds_bars"] if e >= opt_eth_ds) >= 1])
    mc_ds_h_mean = float(np.mean(mc_ds_h_cal)) if len(mc_ds_h_cal) > 0 else 0.0
    mc_ds_v_mean = float(np.mean(mc_ds_v_cal)) if len(mc_ds_v_cal) > 0 else 0.0

    # =========================================================================
    # PLOT 1: Comprehensive Calibration Figure (4 Panels)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    plt.subplots_adjust(wspace=0.25, hspace=0.30)

    # Panel (0,0): DS Scan
    ax = axes[0, 0]
    ax.plot(thresholds, mc_ds_scan, color="#1f77b4", lw=2.8, label=r"Simulation (MC $\langle N_{\rm DS}\rangle$)")
    ax.axhline(data_ds_act_mean, color="#d62728", ls="--", lw=2.2, label=f"Data Active: {data_ds_act_mean:.2f} hits")
    ax.axvline(opt_eth_ds, color="#2ca02c", ls=":", lw=2.0, label=f"Optimal $E_{{th}}^* = {opt_eth_ds:.2f}$ MeV")
    ax.plot([opt_eth_ds], [data_ds_act_mean], marker="o", markersize=9, color="#d62728", markeredgecolor="black", zorder=5)
    ax.set_title("Downstream (DS) Threshold Scan (180 GeV, 3 Walls)", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Energy Deposit Threshold $E_{\rm dep}$ [MeV]", fontsize=11)
    ax.set_ylabel(r"$\langle N_{\rm hits}^{\rm DS}\rangle$ per active event", fontsize=11)
    ax.set_ylim(bottom=0.0, top=float(mc_ds_scan[0]) * 1.15)
    ax.set_xlim(left=0.0, right=6.0)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"Intersection:\n$E_{{th}}^* = {opt_eth_ds:.2f}$ MeV\n$<N_{{hits}}> = {data_ds_act_mean:.2f}$",
        xy=(opt_eth_ds, data_ds_act_mean),
        xytext=(opt_eth_ds + 0.9, data_ds_act_mean + 1.2),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2ae", edgecolor="#b59410", alpha=0.9),
        fontsize=9.5, fontweight="bold",
    )

    # Panel (0,1): DS Hit Multiplicity
    ax = axes[0, 1]
    bins_ds = np.arange(0.5, 35.5, 1.0)
    ax.hist(mc_ds_raw_act, bins=bins_ds, density=True, histtype="stepfilled", alpha=0.30, color="#7f7f7f", edgecolor="#555555", lw=1.5, label=f"MC Raw ($E_{{th}}=0$, mean={np.mean(mc_ds_raw_act):.2f})")
    ax.hist(mc_ds_cal_act, bins=bins_ds, density=True, histtype="step", color="#1f77b4", lw=2.5, label=f"MC Calib ($E_{{th}}^*={opt_eth_ds:.2f}$ MeV, mean={opt_mc_ds_mean:.2f})")
    if len(data_ds_act) > 0:
        ax.hist(data_ds_act, bins=bins_ds, density=True, histtype="step", color="#d62728", lw=2.5, ls="--", label=f"Data Active (mean={data_ds_act_mean:.2f})")
    ax.set_title(f"DS Hit Multiplicity (Active Events, $N_{{\\rm US}} \\geq {args.min_us_hits}$)", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$N_{\rm hits}^{\rm DS}$ per event", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=0.0, right=35.0)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax.grid(True, alpha=0.3)

    # Panel (1,0): US Scan
    ax = axes[1, 0]
    ax.plot(thresholds, mc_us_scan, color="#1f77b4", lw=2.8, label=r"Simulation (MC $\langle N_{\rm US}\rangle$)")
    ax.axhline(data_us_mean, color="#d62728", ls="--", lw=2.2, label=f"Data Mean: {data_us_mean:.2f} hits")
    ax.axvline(opt_eth_us, color="#2ca02c", ls=":", lw=2.0, label=f"Optimal $E_{{th}}^* = {opt_eth_us:.2f}$ MeV")
    ax.plot([opt_eth_us], [data_us_mean], marker="o", markersize=9, color="#d62728", markeredgecolor="black", zorder=5)
    ax.set_title(f"Upstream (US) Threshold Scan ($N_{{\\rm US}} \\geq {args.min_us_hits}$)", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Energy Deposit Threshold $E_{\rm dep}$ [MeV]", fontsize=11)
    ax.set_ylabel(r"$\langle N_{\rm hits}^{\rm US}\rangle$ per event", fontsize=11)
    ax.set_ylim(bottom=0.0, top=float(mc_us_scan[0]) * 1.15)
    ax.set_xlim(left=0.0, right=15.0)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"Intersection:\n$E_{{th}}^* = {opt_eth_us:.2f}$ MeV\n$<N_{{hits}}> = {data_us_mean:.2f}$",
        xy=(opt_eth_us, data_us_mean),
        xytext=(opt_eth_us + 1.5, data_us_mean + 3.0),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2ae", edgecolor="#b59410", alpha=0.9),
        fontsize=9.5, fontweight="bold",
    )

    # Panel (1,1): US Hit Multiplicity Distribution
    ax = axes[1, 1]
    bins_us = np.arange(7.5, 45.5, 1.0)
    ax.hist(mc_us_raw, bins=bins_us, density=True, histtype="stepfilled", alpha=0.30, color="#7f7f7f", edgecolor="#555555", lw=1.5, label=f"MC Raw ($E_{{th}}=0$, mean={np.mean(mc_us_raw):.2f})")
    ax.hist(mc_us_cal, bins=bins_us, density=True, histtype="step", color="#1f77b4", lw=2.5, label=f"MC Calib ($E_{{th}}^*={opt_eth_us:.2f}$ MeV, mean={opt_mc_us_mean:.2f})")
    ax.hist(data_us_hits, bins=bins_us, density=True, histtype="step", color="#d62728", lw=2.5, ls="--", label=f"Data Run 100672 (mean={data_us_mean:.2f})")
    ax.set_title(f"US Hit Multiplicity ($N_{{\\rm US}} \\geq {args.min_us_hits}$ Both Samples)", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$N_{\rm hits}^{\rm US}$ per event", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=7.0, right=45.0)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p1_png = os.path.join(args.outdir, "calib_mufilter_180gev_matched.png")
    p1_pdf = os.path.join(args.outdir, "calib_mufilter_180gev_matched.pdf")
    p1_art = os.path.join(args.artifact_dir, "calib_mufilter_180gev_matched.png")
    fig.savefig(p1_png, dpi=300)
    fig.savefig(p1_pdf)
    fig.savefig(p1_art, dpi=300)
    plt.close(fig)
    print(f"Saved: {p1_png} and {p1_art}", flush=True)

    # =========================================================================
    # PLOT 2: US Plane Profile & DS Views
    # =========================================================================
    fig_p, (ax_pl, ax_v) = plt.subplots(1, 2, figsize=(14, 5.5))
    plt.subplots_adjust(wspace=0.25)

    x_pl = np.arange(5)
    w = 0.26
    ax_pl.bar(x_pl - w, mc_plane_raw_means, width=w, color="#7f7f7f", alpha=0.55, edgecolor="#555555", lw=1.2, label=r"MC Raw ($E_{\rm th} = 0$)")
    ax_pl.bar(x_pl, mc_plane_cal_means, width=w, color="#1f77b4", alpha=0.85, edgecolor="#0d47a1", lw=1.2, label=f"MC Calib ($E_{{th}}^* = {opt_eth_us:.2f}$ MeV)")
    ax_pl.bar(x_pl + w, data_plane_means, width=w, color="#d62728", alpha=0.85, edgecolor="#b71c1c", lw=1.2, label="Data Run 100671")
    ax_pl.set_title("US Longitudinal Shower Profile (180 GeV, 3 Walls)", fontsize=12, fontweight="bold")
    ax_pl.set_xlabel("US Plane Index", fontsize=11)
    ax_pl.set_ylabel(r"$\langle N_{\rm hits}\rangle$ per event", fontsize=11)
    ax_pl.set_xticks(x_pl)
    ax_pl.set_xticklabels([f"Plane {p}" for p in range(5)])
    ax_pl.set_ylim(bottom=0.0, top=max(mc_plane_raw_means) * 1.25)
    ax_pl.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax_pl.grid(True, alpha=0.3)

    x_v = np.arange(2)
    mc_v_raw = [float(np.mean([len(ev["ds_h_bars"]) for ev in mc_events if ev["n_ds_raw"] >= 1])),
                float(np.mean([len(ev["ds_v_bars"]) for ev in mc_events if ev["n_ds_raw"] >= 1]))]
    mc_v_cal = [mc_ds_h_mean, mc_ds_v_mean]
    data_v = [data_ds_h_mean, data_ds_v_mean]

    ax_v.bar(x_v - w, mc_v_raw, width=w, color="#7f7f7f", alpha=0.55, edgecolor="#555555", lw=1.2, label=r"MC Raw ($E_{\rm th} = 0$)")
    ax_v.bar(x_v, mc_v_cal, width=w, color="#1f77b4", alpha=0.85, edgecolor="#0d47a1", lw=1.2, label=f"MC Calib ($E_{{th}}^* = {opt_eth_ds:.2f}$ MeV)")
    ax_v.bar(x_v + w, data_v, width=w, color="#d62728", alpha=0.85, edgecolor="#b71c1c", lw=1.2, label="Data Active")
    ax_v.set_title("DS Station 0 Readout Orientation", fontsize=12, fontweight="bold")
    ax_v.set_xlabel("DS Readout Orientation", fontsize=11)
    ax_v.set_ylabel(r"$\langle N_{\rm hits}\rangle$ per active event", fontsize=11)
    ax_v.set_xticks(x_v)
    ax_v.set_xticklabels(["Horizontal\n(Bars 0-59)", "Vertical\n(Bars 60-119)"])
    ax_v.set_ylim(bottom=0.0, top=max(mc_v_raw) * 1.35)
    ax_v.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax_v.grid(True, alpha=0.3)

    plt.tight_layout()
    p2_png = os.path.join(args.outdir, "calib_mufilter_180gev_planes.png")
    p2_pdf = os.path.join(args.outdir, "calib_mufilter_180gev_planes.pdf")
    p2_art = os.path.join(args.artifact_dir, "calib_mufilter_180gev_planes.png")
    fig_p.savefig(p2_png, dpi=300)
    fig_p.savefig(p2_pdf)
    fig_p.savefig(p2_art, dpi=300)
    plt.close(fig_p)
    print(f"Saved: {p2_png} and {p2_art}", flush=True)

    # =========================================================================
    # WRITE ROOT FILE WITH CANVASES
    # =========================================================================
    root_out = os.path.join(args.outdir, "calib_mufilter_180gev_3walls.root")
    f_root = ROOT.TFile.Open(root_out, "RECREATE")

    c_sum = ROOT.TCanvas("c_180gev_calibration", "180 GeV pi+ 3 Walls Calibration", 1600, 1000)
    c_sum.Divide(2, 2)

    # Pad 1: DS Scan
    c_sum.cd(1)
    g_ds = ROOT.TGraph(len(thresholds), thresholds, mc_ds_scan)
    g_ds.SetName("g_ds_scan_180gev")
    g_ds.SetTitle("DS Threshold Scan (180 GeV);Threshold E_{dep} [MeV];<N_{hits}^{DS}> per active event")
    g_ds.SetLineColor(ROOT.kAzure + 2)
    g_ds.SetLineWidth(3)
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
    c_sum.cd(2)
    h_ds_mc_raw = ROOT.TH1D("h_ds_mc_raw", "DS Hit Multiplicity;N_{hits}^{DS};Probability Density", 35, 0.5, 35.5)
    h_ds_mc_cal = ROOT.TH1D("h_ds_mc_calib", "DS Hit Multiplicity MC Calib;N_{hits}^{DS};Probability Density", 35, 0.5, 35.5)
    h_ds_data = ROOT.TH1D("h_ds_data_active", "DS Hit Multiplicity Data;N_{hits}^{DS};Probability Density", 35, 0.5, 35.5)
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
    h_ds_mc_raw.Draw("HIST")
    h_ds_mc_cal.Draw("HIST SAME")
    h_ds_data.Draw("HIST SAME")

    # Pad 3: US Scan
    c_sum.cd(3)
    g_us = ROOT.TGraph(len(thresholds), thresholds, mc_us_scan)
    g_us.SetName("g_us_scan_180gev")
    g_us.SetTitle("US Threshold Scan (180 GeV);Threshold E_{dep} [MeV];<N_{hits}^{US}> per event")
    g_us.SetLineColor(ROOT.kAzure + 2)
    g_us.SetLineWidth(3)
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
    c_sum.cd(4)
    h_us_mc_raw = ROOT.TH1D("h_us_mc_raw", "US Hit Multiplicity;N_{hits}^{US};Probability Density", 38, 7.5, 45.5)
    h_us_mc_cal = ROOT.TH1D("h_us_mc_calib", "US Hit Multiplicity MC Calib;N_{hits}^{US};Probability Density", 38, 7.5, 45.5)
    h_us_data = ROOT.TH1D("h_us_data", "US Hit Multiplicity Data;N_{hits}^{US};Probability Density", 38, 7.5, 45.5)
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
    h_us_mc_raw.Draw("HIST")
    h_us_mc_cal.Draw("HIST SAME")
    h_us_data.Draw("HIST SAME")

    c_sum.Update()
    c_sum.Write()
    f_root.Close()
    print(f"ROOT file written: {root_out}", flush=True)


if __name__ == "__main__":
    main()
