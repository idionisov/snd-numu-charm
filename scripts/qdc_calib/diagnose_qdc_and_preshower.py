#!/usr/bin/env python3
"""
scripts/qdc_calib/diagnose_qdc_and_preshower.py
----------------------------------------------
Executes Diagnostics 3 and 4:
- Diagnostic 3: Energy / QDC signal distributions in Core (Bar 4) vs Wings (Bars 2 & 7)
- Diagnostic 4: Longitudinal shower development and pre-shower depth in Scifi Stations 1-5

Outputs:
- plots/tb_diag3_qdc_spectrum.png
- plots/tb_diag4_preshower_profile.png
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    th = calc.get_sipm_threshold_pe()

    # -------------------------------------------------------------------------
    # 1. Read Testbeam Data (Run 100671)
    # -------------------------------------------------------------------------
    data_file = "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100671/sndsw_raw-0000.root"
    print(f"[1/3] Reading Data from {data_file}...")
    f_d = ROOT.TFile.Open(data_file)
    t_d = f_d.Get("rawConv")
    t_d.SetBranchStatus("*", 0)
    t_d.SetBranchStatus("Digi_ScifiHits*", 1)
    t_d.SetBranchStatus("Digi_MuFilterHits*", 1)

    d_qdc = {2: [], 4: [], 7: []}
    d_scifi_mult = {s: [] for s in range(1, 6)}
    d_start_station = []

    n_scan_d = min(35000, t_d.GetEntries())
    for i in range(n_scan_d):
        t_d.GetEntry(i)
        if len(t_d.Digi_ScifiHits) < 100:
            continue

        # Scifi hits per station
        s_counts = {s: 0 for s in range(1, 6)}
        for sh in t_d.Digi_ScifiHits:
            if sh.isValid():
                st = int(sh.GetStation())
                if 1 <= st <= 5:
                    s_counts[st] += 1

        for s in range(1, 6):
            d_scifi_mult[s].append(s_counts[s])

        # Find first station with shower formed (>= 15 hits)
        start_st = 6
        for s in range(1, 6):
            if s_counts[s] >= 15:
                start_st = s
                break
        d_start_station.append(start_st)

        # MuFilter signals in US0
        for mh in t_d.Digi_MuFilterHits:
            if not mh.isValid():
                continue
            did = mh.GetDetectorID()
            if did // 1000 == 20:  # US Plane 0
                bar = did % 1000
                if bar in (2, 4, 7):
                    # Check in-time signals
                    tot_sig = 0.0
                    for ch in range(16):
                        t_val = mh.GetTime(ch)
                        if t_val != -999.0 and abs(t_val) <= 15.0:
                            s_val = mh.GetSignal(ch)
                            if s_val > 0.0:
                                tot_sig += s_val
                    if tot_sig > 0.0:
                        d_qdc[bar].append(tot_sig)

        if len(d_start_station) >= 1500:
            break

    f_d.Close()
    print(f"      Data: Extracted {len(d_start_station)} showers.")

    # -------------------------------------------------------------------------
    # 2. Read Geant4 MC Simulation (180 GeV pi+)
    # -------------------------------------------------------------------------
    mc_file = "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/180GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root"
    print(f"[2/3] Reading MC from {mc_file}...")
    f_m = ROOT.TFile.Open(mc_file)
    t_m = f_m.Get("cbmsim")
    t_m.SetBranchStatus("*", 0)
    t_m.SetBranchStatus("Digi_ScifiHits*", 1)
    t_m.SetBranchStatus("MuFilterPoint*", 1)

    mc_pes = {2: [], 4: [], 7: []}
    mc_eloss = {2: [], 4: [], 7: []}
    mc_scifi_mult = {s: [] for s in range(1, 6)}
    mc_start_station = []

    n_events_m = min(1500, t_m.GetEntries())
    for i in range(n_events_m):
        t_m.GetEntry(i)
        if len(t_m.Digi_ScifiHits) < 100:
            continue

        # Scifi hits per station
        s_counts = {s: 0 for s in range(1, 6)}
        for sh in t_m.Digi_ScifiHits:
            if sh.isValid():
                st = int(sh.GetStation())
                if 1 <= st <= 5:
                    s_counts[st] += 1

        for s in range(1, 6):
            mc_scifi_mult[s].append(s_counts[s])

        start_st = 6
        for s in range(1, 6):
            if s_counts[s] >= 15:
                start_st = s
                break
        mc_start_station.append(start_st)

        # MuFilterPoints in US0
        pts_by_bar = {}
        for pt in t_m.MuFilterPoint:
            did = pt.GetDetectorID()
            if did // 1000 == 20:
                bar = did % 1000
                if bar in (2, 4, 7):
                    pts_by_bar.setdefault(bar, []).append({
                        "detector_id": did,
                        "x": float(pt.GetX()),
                        "y": float(pt.GetY()),
                        "z": float(pt.GetZ()),
                        "time": float(pt.GetTime()),
                        "eloss": float(pt.GetEnergyLoss()),
                        "pdg": int(pt.PdgCode()),
                    })

        for bar in (2, 4, 7):
            if bar in pts_by_bar:
                pts = pts_by_bar[bar]
                tot_e = sum(p["eloss"] * 1000.0 for p in pts)
                mc_eloss[bar].append(tot_e)

                pes = calc.get_sipm_pes(pts, det_id=20000 + bar)
                fired_pes = pes[pes >= th]
                if len(fired_pes) > 0:
                    mc_pes[bar].append(float(np.sum(pes)))

    f_m.Close()
    print(f"      MC: Extracted {len(mc_start_station)} showers.")

    # -------------------------------------------------------------------------
    # 3. Plots & Analysis
    # -------------------------------------------------------------------------
    print("[3/3] Generating diagnostic figures...")

    # Figure 1: Diagnostic 3 - Signal / Energy Spectra in Core vs Wings
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), constrained_layout=True)
    bars_to_plot = [4, 2, 7]
    titles = ["Bar 4 (Shower Core)", "Bar 2 (Bottom Wing)", "Bar 7 (Top Wing)"]

    for idx, bar in enumerate(bars_to_plot):
        ax = axes[idx]
        # Data QDC (arbitrary ADC units, normalize shape)
        d_vals = np.array(d_qdc[bar])
        m_vals = np.array(mc_pes[bar])

        if len(d_vals) > 0:
            ax.hist(d_vals, bins=40, density=True, histtype="step", color="black", lw=2.0, label=f"Data QDC (N={len(d_vals)})")
        if len(m_vals) > 0:
            # Scale MC PE to visually align MIP peak with Data for shape comparison
            scale_factor = (np.median(d_vals) / max(1.0, np.median(m_vals))) if len(d_vals) > 0 else 1.0
            ax.hist(m_vals * scale_factor, bins=40, density=True, histtype="step", color="#28a745", lw=2.0, linestyle="--", label=f"MC PE Scaled (N={len(m_vals)})")

        ax.set_title(titles[idx], fontsize=12, fontweight="bold")
        ax.set_xlabel("Signal Amplitude (a.u.)", fontsize=11)
        ax.set_ylabel("Normalized Density", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=9, loc="upper right")

    p1 = os.path.join(plot_dir, "tb_diag3_qdc_spectrum.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    os.system(f"cp {p1} {os.path.join(art_dir, 'tb_diag3_qdc_spectrum.png')}")
    plt.close()

    # Figure 2: Diagnostic 4 - Longitudinal Development & Pre-Shower Station
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)

    # Panel A: Mean Scifi Hits vs Station
    ax = axes[0]
    stations = np.arange(1, 6)
    d_means = [np.mean(d_scifi_mult[s]) for s in stations]
    d_errs = [np.std(d_scifi_mult[s]) / np.sqrt(len(d_scifi_mult[s])) for s in stations]
    m_means = [np.mean(mc_scifi_mult[s]) for s in stations]
    m_errs = [np.std(mc_scifi_mult[s]) / np.sqrt(len(mc_scifi_mult[s])) for s in stations]

    ax.errorbar(stations - 0.05, d_means, yerr=d_errs, fmt="o-", color="black", lw=2.0, capsize=4, label="Data (180 GeV $\\pi^+$)")
    ax.errorbar(stations + 0.05, m_means, yerr=m_errs, fmt="s--", color="#28a745", lw=2.0, capsize=4, label="MC Simulation")
    ax.set_title("Longitudinal Shower Profile in Target Scifi", fontsize=12, fontweight="bold")
    ax.set_xlabel("Scifi Station Index (1=Front, 5=Back of Target)", fontsize=11)
    ax.set_ylabel("Mean Scifi Hit Multiplicity", fontsize=11)
    ax.set_xticks(stations)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=10, loc="upper left")

    # Panel B: Starting Station Distribution
    ax = axes[1]
    bins_st = np.arange(0.5, 7.5, 1.0)
    ax.hist(d_start_station, bins=bins_st, density=True, histtype="step", color="black", lw=2.2, label="Data")
    ax.hist(mc_start_station, bins=bins_st, density=True, histtype="step", color="#28a745", lw=2.0, linestyle="--", label="MC Simulation")
    ax.set_title("Shower Initiation Station ($N_{hits} \\geq 15$)", fontsize=12, fontweight="bold")
    ax.set_xlabel("First Showering Station (1..5, 6=Penetrated to US0)", fontsize=11)
    ax.set_ylabel("Fraction of Events", fontsize=11)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(["St 1", "St 2", "St 3", "St 4", "St 5", "US0"])
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=10, loc="upper right")

    p2 = os.path.join(plot_dir, "tb_diag4_preshower_profile.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    os.system(f"cp {p2} {os.path.join(art_dir, 'tb_diag4_preshower_profile.png')}")
    plt.close()

    print("\n=== DIAGNOSTIC 4 QUANTITATIVE RESULTS ===")
    print("Scifi Station Multiplicities:")
    for s in stations:
        print(f"  Station {s}: Data = {d_means[s-1]:.1f} hits | MC = {m_means[s-1]:.1f} hits (Ratio MC/Data = {m_means[s-1]/max(0.1, d_means[s-1]):.2f})")

    d_st_counts = np.bincount(d_start_station, minlength=7)[1:]
    m_st_counts = np.bincount(mc_start_station, minlength=7)[1:]
    print("\nShower Starting Station Fraction (%):")
    for s in range(1, 7):
        lbl = f"Station {s}" if s <= 5 else "Late/US0"
        d_pct = d_st_counts[s-1] / len(d_start_station) * 100.0
        m_pct = m_st_counts[s-1] / len(mc_start_station) * 100.0
        print(f"  {lbl:10s}: Data = {d_pct:5.1f}% | MC = {m_pct:5.1f}%")

    print("\n=== DIAGNOSTIC 3 QUANTITATIVE RESULTS ===")
    for b in (4, 2, 7):
        d_arr = np.array(d_qdc[b])
        m_arr = np.array(mc_pes[b])
        print(f"Bar {b}:")
        print(f"  Data Fired: N={len(d_arr):4d} | Median QDC={np.median(d_arr) if len(d_arr) else 0:.1f} | Mean={np.mean(d_arr) if len(d_arr) else 0:.1f}")
        print(f"  MC Fired:   N={len(m_arr):4d} | Median PE ={np.median(m_arr) if len(m_arr) else 0:.1f} | Mean={np.mean(m_arr) if len(m_arr) else 0:.1f}")

    print("\n[SUCCESS] Diagnostics 3 and 4 completed successfully.")


if __name__ == "__main__":
    main()
