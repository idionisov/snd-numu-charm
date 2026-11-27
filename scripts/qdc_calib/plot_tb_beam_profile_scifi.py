#!/usr/bin/env python3
"""
scripts/qdc_calib/plot_tb_beam_profile_scifi.py
----------------------------------------------
Extracts and visualizes the testbeam pion beam profile:
- In Simulation: MCTrack[0] starting trajectory and Scifi Station 1 hits
- In Data: Scifi Station 1 hits from 180 GeV pi+ run (Run 100671)

Outputs:
- plots/tb_beam_profile_data_vs_mc.png
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def gaus(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def extract_beam_profiles():
    geo_file = "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/180GeV_211/geofile_full.PG_211-TGeant4.root"
    import SndlhcGeo
    geo = SndlhcGeo.GeoInterface(geo_file)
    scifi = geo.modules.get("Scifi")
    vl, vr = ROOT.TVector3(), ROOT.TVector3()

    # Precompute channel position cache for Station 1 to make event loop instantaneous
    pos_cache = {}
    for o in (0, 1):
        for m in range(3):
            for sipm in range(4):
                for ch in range(128):
                    did = 1000000 + o * 100000 + m * 10000 + sipm * 1000 + ch
                    scifi.GetSiPMPosition(did, vl, vr)
                    val = 0.5 * (vl.X() + vr.X()) if o == 1 else 0.5 * (vl.Y() + vr.Y())
                    pos_cache[did] = (o, float(val))

    # 1. MC Simulation
    mc_file = "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/180GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root"
    print(f"Reading MC from {mc_file}...")
    f_m = ROOT.TFile.Open(mc_file)
    t_m = f_m.Get("cbmsim")
    t_m.SetBranchStatus("*", 0)
    t_m.SetBranchStatus("MCTrack*", 1)
    t_m.SetBranchStatus("Digi_ScifiHits*", 1)

    mc_track_x, mc_track_y = [], []
    mc_scifi_x, mc_scifi_y = [], []

    for i in range(min(300, t_m.GetEntries())):
        t_m.GetEntry(i)
        tr = t_m.MCTrack[0]
        mc_track_x.append(tr.GetStartX())
        mc_track_y.append(tr.GetStartY())

        xh, yh = [], []
        for h in t_m.Digi_ScifiHits:
            if not h.isValid():
                continue
            info = pos_cache.get(h.GetDetectorID())
            if info is not None:
                if info[0] == 1:
                    xh.append(info[1])
                else:
                    yh.append(info[1])
        if xh:
            mc_scifi_x.append(np.mean(xh))
        if yh:
            mc_scifi_y.append(np.mean(yh))
    f_m.Close()

    # 2. Testbeam Data
    data_file = "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100671/sndsw_raw-0000.root"
    print(f"Reading Data from {data_file}...")
    f_d = ROOT.TFile.Open(data_file)
    t_d = f_d.Get("rawConv")
    t_d.SetBranchStatus("*", 0)
    t_d.SetBranchStatus("Digi_ScifiHits*", 1)

    data_scifi_x, data_scifi_y = [], []
    for i in range(min(20000, t_d.GetEntries())):
        t_d.GetEntry(i)
        if len(t_d.Digi_ScifiHits) < 20:
            continue
        xh, yh = [], []
        for h in t_d.Digi_ScifiHits:
            if not h.isValid():
                continue
            info = pos_cache.get(h.GetDetectorID())
            if info is not None:
                if info[0] == 1:
                    xh.append(info[1])
                else:
                    yh.append(info[1])
        if xh and yh:
            data_scifi_x.append(np.median(xh))
            data_scifi_y.append(np.median(yh))
    f_d.Close()

    return {
        "mc_track_x": np.array(mc_track_x),
        "mc_track_y": np.array(mc_track_y),
        "mc_scifi_x": np.array(mc_scifi_x),
        "mc_scifi_y": np.array(mc_scifi_y),
        "data_scifi_x": np.array(data_scifi_x),
        "data_scifi_y": np.array(data_scifi_y),
    }


def main():
    data = extract_beam_profiles()
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"

    dx = data["data_scifi_x"]
    dy = data["data_scifi_y"]
    mx = data["mc_scifi_x"]
    my = data["mc_scifi_y"]
    mtx = data["mc_track_x"]
    mty = data["mc_track_y"]

    mean_dx, std_dx = np.mean(dx), np.std(dx)
    mean_dy, std_dy = np.mean(dy), np.std(dy)

    mean_mx, std_mx = np.mean(mx), np.std(mx)
    mean_my, std_my = np.mean(my), np.std(my)

    print("--- Beam Profile Summary ---")
    print(f"Data Scifi St1: X = {mean_dx:.2f} +- {std_dx:.2f} cm, Y = {mean_dy:.2f} +- {std_dy:.2f} cm (N={len(dx)})")
    print(f"MC MCTrack[0]:  X = {np.mean(mtx):.2f} +- {np.std(mtx):.2f} cm, Y = {np.mean(mty):.2f} +- {np.std(mty):.2f} cm")
    print(f"MC Scifi St1:   X = {mean_mx:.2f} +- {std_mx:.2f} cm, Y = {mean_my:.2f} +- {std_my:.2f} cm (N={len(mx)})")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    # -------------------------------------------------------------------------
    # Panel 1: 2D Beam Spot in Scifi Station 1
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    h2 = ax.hist2d(dx, dy, bins=[np.linspace(-46, -32, 50), np.linspace(38, 50, 50)], cmap="Blues", cmin=1)
    cb = fig.colorbar(h2[3], ax=ax)
    cb.set_label("Data Events / Bin", fontsize=10)

    # Plot 1-sigma and 2-sigma ellipses for Data
    ellipse1 = Ellipse((mean_dx, mean_dy), 2 * std_dx, 2 * std_dy, fill=False, edgecolor="#d32f2f", lw=2, linestyle="-", label=f"Data 1$\\sigma$ ({std_dx:.1f} $\\times$ {std_dy:.1f} cm)")
    ellipse2 = Ellipse((mean_dx, mean_dy), 4 * std_dx, 4 * std_dy, fill=False, edgecolor="#d32f2f", lw=1.5, linestyle="--", label="Data 2$\\sigma$ contour")
    ax.add_patch(ellipse1)
    ax.add_patch(ellipse2)

    # Plot MC point
    ax.plot(np.mean(mtx), np.mean(mty), marker="*", color="#ffd600", markersize=16, markeredgecolor="black", markeredgewidth=1.5, label=f"MC Pencil Beam ({np.mean(mtx):.2f}, {np.mean(mty):.2f})")

    ax.set_title("Testbeam Beam Spot at Scifi Station 1: Data vs. MC", fontsize=12, fontweight="bold")
    ax.set_xlabel("Horizontal Position $X$ [cm]", fontsize=11)
    ax.set_ylabel("Vertical Position $Y$ [cm]", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9.5, loc="upper right")

    # -------------------------------------------------------------------------
    # Panel 2: 1D Horizontal Profile (X)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    bins_x = np.linspace(-46, -32, 56)
    counts_x, edges_x, _ = ax.hist(dx, bins=bins_x, density=True, histtype="stepfilled", color="#1976d2", alpha=0.45, label=f"Data (Mean={mean_dx:.2f}, $\\sigma$={std_dx:.2f} cm)")
    ax.hist(dx, bins=bins_x, density=True, histtype="step", color="#0d47a1", lw=2.0)

    # Fit Gaussian to Data X
    centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
    try:
        popt_x, _ = curve_fit(gaus, centers_x, counts_x, p0=[max(counts_x), mean_dx, std_dx])
        x_eval = np.linspace(-46, -32, 200)
        ax.plot(x_eval, gaus(x_eval, *popt_x), color="#d32f2f", lw=2.2, linestyle="-", label=f"Data Fit: $\\mu$={popt_x[1]:.2f}, $\\sigma$={abs(popt_x[2]):.2f} cm")
    except Exception:
        pass

    # MC X delta peak representation
    ax.axvline(np.mean(mtx), color="#e65100", lw=2.5, linestyle="--", label=f"MC Primary Track ($X$ = {np.mean(mtx):.2f} cm)")
    ax.set_title("Horizontal Beam Profile ($X$)", fontsize=12, fontweight="bold")
    ax.set_xlabel("$X$ Position [cm]", fontsize=11)
    ax.set_ylabel("Probability Density [1/cm]", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9.5, loc="upper left")

    # -------------------------------------------------------------------------
    # Panel 3: 1D Vertical Profile (Y)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    bins_y = np.linspace(38, 50, 48)
    counts_y, edges_y, _ = ax.hist(dy, bins=bins_y, density=True, histtype="stepfilled", color="#388e3c", alpha=0.45, label=f"Data (Mean={mean_dy:.2f}, $\\sigma$={std_dy:.2f} cm)")
    ax.hist(dy, bins=bins_y, density=True, histtype="step", color="#1b5e20", lw=2.0)

    # Fit Gaussian to Data Y
    centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])
    try:
        popt_y, _ = curve_fit(gaus, centers_y, counts_y, p0=[max(counts_y), mean_dy, std_dy])
        y_eval = np.linspace(38, 50, 200)
        ax.plot(y_eval, gaus(y_eval, *popt_y), color="#d32f2f", lw=2.2, linestyle="-", label=f"Data Fit: $\\mu$={popt_y[1]:.2f}, $\\sigma$={abs(popt_y[2]):.2f} cm")
    except Exception:
        pass

    # MC Y delta peak representation
    ax.axvline(np.mean(mty), color="#e65100", lw=2.5, linestyle="--", label=f"MC Primary Track ($Y$ = {np.mean(mty):.2f} cm)")
    ax.set_title("Vertical Beam Profile ($Y$)", fontsize=12, fontweight="bold")
    ax.set_xlabel("$Y$ Position [cm]", fontsize=11)
    ax.set_ylabel("Probability Density [1/cm]", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9.5, loc="upper right")

    # -------------------------------------------------------------------------
    # Panel 4: Vertical Beam Position Overlaid with US Plane 0 Bars
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    # US0 bar centers: bar 0 is bottom, bar 9 is top. Width = 6.01 cm. Center of bar 0 ~ 17.0 cm.
    # In testbeam coordinate system:
    # Bar centers: Y_b = 17.004 + 6.01 * b
    bar_edges = [14.0 + 6.01 * b for b in range(11)]
    for b in range(10):
        y_lo = bar_edges[b]
        y_hi = bar_edges[b + 1]
        color = "#e0e0e0" if b % 2 == 0 else "#f5f5f5"
        ax.axvspan(y_lo, y_hi, color=color, alpha=0.7)
        ax.text(0.5 * (y_lo + y_hi), 0.35, f"Bar {b}", ha="center", va="center", fontsize=9, rotation=90, color="#616161")

    # Plot Data Y distribution and MC line
    ax.hist(dy, bins=np.linspace(14, 74, 120), density=True, histtype="stepfilled", color="#7b1fa2", alpha=0.45, label="Data Beam Spot")
    ax.hist(dy, bins=np.linspace(14, 74, 120), density=True, histtype="step", color="#4a148c", lw=2.0)
    ax.axvline(np.mean(mty), color="#e65100", lw=2.5, linestyle="--", label=f"MC Pencil Beam ({np.mean(mty):.2f} cm)")

    ax.set_xlim(25, 62)
    ax.set_title("Beam Spot Incident on Upstream Plane 0 Bar Geometry", fontsize=12, fontweight="bold")
    ax.set_xlabel("Vertical Coordinate $Y$ [cm]", fontsize=11)
    ax.set_ylabel("Probability Density [1/cm]", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9.5, loc="upper right")

    out_png = os.path.join(plot_dir, "tb_beam_profile_data_vs_mc.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    os.system(f"cp -f {out_png} {art_dir}/")
    print(f"[SUCCESS] Saved beam profile plot to {out_png} and {art_dir}/")


if __name__ == "__main__":
    main()
