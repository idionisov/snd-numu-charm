#!/usr/bin/env python3
"""
scripts/diagnostics/plot_veto_backsplash_trajectories.py
-------------------------------------------------------
Diagnostic script analyzing particle trajectories in Monte Carlo simulation
for events where the Veto detector is struck later than SciFi.

Algorithm:
  1. Select events satisfying the timing condition:
     Earliest Veto hit occurs later than the latest SciFi hit:
     t(Veto_min) > t(SciFi_max).
     (Optionally configurable via --condition).
  2. In passing events, extract all Veto MCPoints (detectorID system == 1).
  3. Pair each Veto MCPoint with its corresponding MCTrack via mcpoint.GetTrackID()
     (or read MCPoint momentum if trackID < 0).
  4. Compute trajectory angle observables:
     - Projected angles theta_xz and theta_yz in [-pi/2, pi/2] (0 = along z-axis).
     - Full 3D polar angle theta = arccos(pz / p) in [0, pi] (0 = forward, pi = backward).
     - Azimuthal angle phi = arctan2(py, px) in [-pi, pi].
     - Particle species (PDG code).
  5. Generate:
     - Trajectory distribution plot in [-pi/2, pi/2].
     - Full 3D polar angle distribution [0, pi].
     - 2D Polar projection plot (r = theta, phi) in matplotlib.
     - Particle composition breakdown (PDG IDs).
     - Structured ROOT histograms & canvases saved to file.

Usage:
  # Run on partition 1:
  python3 scripts/diagnostics/plot_veto_backsplash_trajectories.py -p 1

  # Run on partitions 1-5 with 4 workers:
  python3 scripts/diagnostics/plot_veto_backsplash_trajectories.py -p 1-5 -j 4 -o plots/veto_trajectories
"""

from __future__ import annotations

import os
import sys
import math
import time
import argparse
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat("emruo")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import ddfUtils.mpl_styling as dms
    HAS_DDF_STYLING = True
except ImportError:
    HAS_DDF_STYLING = False


def parse_partitions(part_arg: Optional[str]) -> List[int]:
    """Parse partition range or list."""
    if not part_arg or part_arg.strip().lower() == "all":
        return list(range(1, 1001))

    partitions = set()
    for token in part_arg.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start, end = int(parts[0]), int(parts[1])
                for p in range(min(start, end), max(start, end) + 1):
                    partitions.add(p)
        elif token.isdigit():
            partitions.add(int(token))

    return sorted(list(partitions))


def format_partition_path(pattern: str, partition: int | str) -> str:
    """Format partition pattern if placeholder present."""
    p_str = str(partition)
    count = pattern.count("%s")
    if count > 0:
        return pattern % tuple(p_str for _ in range(count))
    return pattern


PDG_NAMES = {
    2112: "n (neutron)",
    -2112: "anti-n",
    2212: "p (proton)",
    -2212: "anti-p",
    22: "#gamma (photon)",
    11: "e^{-}",
    -11: "e^{+}",
    13: "#mu^{-}",
    -13: "#mu^{+}",
    211: "#pi^{+}",
    -211: "#pi^{-}",
    111: "#pi^{0}",
    321: "K^{+}",
    -321: "K^{-}",
    130: "K^{0}_{L}",
    310: "K^{0}_{S}",
}


def get_pdg_name(pdg: int) -> str:
    return PDG_NAMES.get(pdg, f"PDG {pdg}")


def process_single_file(
    filepath: str,
    condition: str = "earliest_veto_later_than_latest_scifi",
    max_events: int = -1,
) -> Dict[str, Any]:
    """Process a single ROOT file and collect Veto MCPoint/MCTrack trajectories."""
    data = {
        "theta_xz": [],
        "theta_yz": [],
        "theta_3d": [],
        "phi": [],
        "p_tot": [],
        "pz": [],
        "pt": [],
        "pdg": [],
        "n_events_total": 0,
        "n_events_veto": 0,
        "n_events_passed": 0,
        "n_veto_points": 0,
    }

    if not os.path.isfile(filepath):
        return data

    f_in = ROOT.TFile.Open(filepath, "READ")
    if not f_in or f_in.IsZombie():
        if f_in:
            f_in.Close()
        return data

    t_cbm = f_in.Get("cbmsim")
    if not t_cbm:
        f_in.Close()
        return data

    t_cbm.SetBranchStatus("*", 0)
    t_cbm.SetBranchStatus("Digi_ScifiHits*", 1)
    t_cbm.SetBranchStatus("Digi_MuFilterHits*", 1)
    t_cbm.SetBranchStatus("MuFilterPoint*", 1)
    t_cbm.SetBranchStatus("MCTrack*", 1)

    n_entries = t_cbm.GetEntries()
    limit = n_entries if max_events <= 0 else min(n_entries, max_events)
    data["n_events_total"] = limit

    for i in range(limit):
        t_cbm.GetEntry(i)

        # Collect Veto and SciFi hit times
        v_times = []
        if t_cbm.Digi_MuFilterHits is not None:
            for j in range(t_cbm.Digi_MuFilterHits.GetEntriesFast()):
                h = t_cbm.Digi_MuFilterHits.At(j)
                if h and h.isValid() and int(h.GetSystem()) == 1:
                    t_val = float(h.GetTime(0)) if h.isVertical() else float(0.5 * (h.GetTime(0) + h.GetTime(1)))
                    v_times.append(t_val)

        sf_times = []
        if t_cbm.Digi_ScifiHits is not None:
            for j in range(t_cbm.Digi_ScifiHits.GetEntriesFast()):
                h = t_cbm.Digi_ScifiHits.At(j)
                if h and h.isValid():
                    sf_times.append(float(h.GetTime()))

        if not v_times:
            continue

        data["n_events_veto"] += 1

        # Evaluate condition
        passed = False
        t_v_min = min(v_times)
        t_v_max = max(v_times)
        t_sf_min = min(sf_times) if sf_times else None
        t_sf_max = max(sf_times) if sf_times else None

        if condition == "earliest_veto_later_than_latest_scifi":
            if t_sf_max is not None and t_v_min > t_sf_max:
                passed = True
        elif condition == "earliest_veto_later_than_earliest_scifi":
            if t_sf_min is not None and t_v_min > t_sf_min:
                passed = True
        elif condition == "latest_veto_later_than_earliest_scifi":
            if t_sf_min is not None and t_v_max > t_sf_min:
                passed = True
        elif condition == "all_veto":
            passed = True

        if not passed:
            continue

        data["n_events_passed"] += 1

        # Extract Veto MCPoints
        if t_cbm.MuFilterPoint is None:
            continue

        n_mufi_points = t_cbm.MuFilterPoint.GetEntriesFast()
        n_tracks = t_cbm.MCTrack.GetEntriesFast() if t_cbm.MCTrack is not None else 0

        for j in range(n_mufi_points):
            p = t_cbm.MuFilterPoint.At(j)
            if not p:
                continue

            det_id = int(p.GetDetectorID())
            if (det_id // 10000) != 1:  # Veto detector system ID is 1
                continue

            tid = int(p.GetTrackID())
            px, py, pz = float(p.GetPx()), float(p.GetPy()), float(p.GetPz())
            pdg = int(p.PdgCode())

            # Attempt to pair with MCTrack
            if 0 <= tid < n_tracks:
                trk = t_cbm.MCTrack.At(tid)
                if trk:
                    px = float(trk.GetPx())
                    py = float(trk.GetPy())
                    pz = float(trk.GetPz())
                    pdg = int(trk.GetPdgCode())

            p_tot = math.sqrt(px * px + py * py + pz * pz)
            if p_tot <= 1e-9:
                continue

            pt = math.sqrt(px * px + py * py)

            # Full [-pi, +pi] trajectory angle relative to z-axis:
            # 0 corresponds to straight ahead (+z)
            # +/- pi corresponds to purely backwards (-z)
            txz = math.atan2(px, pz)
            tyz = math.atan2(py, pz)

            # Full 3D polar angle theta in [0, pi] (0 = forward +z, pi = backward -z)
            theta_3d = math.acos(max(-1.0, min(1.0, pz / p_tot)))

            # Transverse azimuthal angle phi in [-pi, pi] (direction on detector face)
            phi = math.atan2(py, px)

            data["theta_xz"].append(txz)
            data["theta_yz"].append(tyz)
            data["theta_3d"].append(theta_3d)
            data["phi"].append(phi)
            data["p_tot"].append(p_tot)
            data["pz"].append(pz)
            data["pt"].append(pt)
            data["pdg"].append(pdg)
            data["n_veto_points"] += 1

    f_in.Close()
    return data


def create_root_histograms(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create and fill ROOT TH1D and TH2D histograms in units of pi."""
    hists: Dict[str, Any] = {}

    pi = math.pi

    # 1. Full [-pi, +pi] trajectory angles in units of pi [-1.0, 1.0]:
    # 0 = straight ahead (+z), +/- 1 = purely backwards (-z)
    hists["h_theta_xz"] = ROOT.TH1D(
        "h_theta_xz",
        "Veto Trajectory #theta_{xz} / #pi = atan2(p_{x}, p_{z}) / #pi;#theta_{xz} / #pi (0=ahead, #pm1=backwards);Entries / 0.04 #pi",
        50, -1.0, 1.0,
    )
    hists["h_theta_yz"] = ROOT.TH1D(
        "h_theta_yz",
        "Veto Trajectory #theta_{yz} / #pi = atan2(p_{y}, p_{z}) / #pi;#theta_{yz} / #pi (0=ahead, #pm1=backwards);Entries / 0.04 #pi",
        50, -1.0, 1.0,
    )

    for val in data["theta_xz"]:
        hists["h_theta_xz"].Fill(val / pi)
    for val in data["theta_yz"]:
        hists["h_theta_yz"].Fill(val / pi)

    # 2. 3D Polar angle in units of pi [0, 1.0] (0 = forward, 1 = backward)
    hists["h_theta_3d"] = ROOT.TH1D(
        "h_theta_3d",
        "Veto Particle 3D Polar Angle: #theta_{3D} / #pi;#theta_{3D} / #pi (0=ahead, 1=backwards);Entries / 0.02 #pi",
        50, 0.0, 1.0,
    )
    for val in data["theta_3d"]:
        hists["h_theta_3d"].Fill(val / pi)

    # 3. Azimuthal angle phi in units of pi [-1.0, 1.0]
    hists["h_phi"] = ROOT.TH1D(
        "h_phi",
        "Veto Particle Azimuthal Angle: #phi / #pi;#phi / #pi = atan2(p_{y}, p_{x}) / #pi;Entries / 0.04 #pi",
        50, -1.0, 1.0,
    )
    for val in data["phi"]:
        hists["h_phi"].Fill(val / pi)

    # 4. 2D Angle distribution (theta_3d vs phi) in units of pi
    hists["h2_theta_vs_phi"] = ROOT.TH2D(
        "h2_theta_vs_phi",
        "Veto Particle Trajectory (#theta_{3D} vs #phi);#phi / #pi;#theta_{3D} / #pi",
        40, -1.0, 1.0, 40, 0.0, 1.0,
    )
    for th, ph in zip(data["theta_3d"], data["phi"]):
        hists["h2_theta_vs_phi"].Fill(ph / pi, th / pi)

    # 5. Momentum distribution
    hists["h_ptot"] = ROOT.TH1D(
        "h_ptot",
        "Veto Particle Total Momentum;Momentum p [GeV/c];Entries",
        50, 0.0, 5.0,
    )
    for p in data["p_tot"]:
        hists["h_ptot"].Fill(p)

    return hists


def build_root_canvases(hists: Dict[str, Any]) -> Dict[str, ROOT.TCanvas]:
    """Build formatted ROOT TCanvases."""
    canvases: Dict[str, ROOT.TCanvas] = {}

    # Canvas 1: Full [-pi, +pi] Trajectory Angles (0 = straight ahead, +/- pi = backwards)
    c1 = ROOT.TCanvas("c_trajectory_angles_z_centered", "Veto Trajectory Angles Centered on Z Axis", 1200, 600)
    c1.Divide(2, 1)

    c1.cd(1).SetGrid(1, 1)
    hists["h_theta_xz"].SetLineColor(ROOT.kBlue + 1)
    hists["h_theta_xz"].SetLineWidth(2)
    hists["h_theta_xz"].SetFillColorAlpha(ROOT.kBlue - 9, 0.35)
    hists["h_theta_xz"].Draw("HIST")

    c1.cd(2).SetGrid(1, 1)
    hists["h_theta_yz"].SetLineColor(ROOT.kRed + 1)
    hists["h_theta_yz"].SetLineWidth(2)
    hists["h_theta_yz"].SetFillColorAlpha(ROOT.kRed - 9, 0.35)
    hists["h_theta_yz"].Draw("HIST")

    canvases["c_trajectory_angles_z_centered"] = c1

    # Canvas 2: 3D Polar Angle [0, pi] and Phi
    c2 = ROOT.TCanvas("c_polar_angles", "Veto 3D Polar Angle and Phi", 1200, 600)
    c2.Divide(2, 1)

    c2.cd(1).SetGrid(1, 1)
    hists["h_theta_3d"].SetLineColor(ROOT.kGreen + 2)
    hists["h_theta_3d"].SetLineWidth(2)
    hists["h_theta_3d"].SetFillColorAlpha(ROOT.kGreen - 9, 0.35)
    hists["h_theta_3d"].Draw("HIST")

    c2.cd(2).SetGrid(1, 1)
    hists["h_phi"].SetLineColor(ROOT.kMagenta + 2)
    hists["h_phi"].SetLineWidth(2)
    hists["h_phi"].SetFillColorAlpha(ROOT.kMagenta - 9, 0.35)
    hists["h_phi"].Draw("HIST")

    canvases["c_polar_angles"] = c2

    # Canvas 3: 2D Correlation
    c3 = ROOT.TCanvas("c_theta_vs_phi_2d", "Theta vs Phi", 800, 700)
    c3.SetRightMargin(0.14)
    c3.SetGrid(1, 1)
    hists["h2_theta_vs_phi"].Draw("COLZ")
    canvases["c_theta_vs_phi_2d"] = c3

    return canvases


def plot_polar_and_projections_matplotlib(
    data: Dict[str, Any],
    output_dir: str,
) -> None:
    """Generate publication-quality trajectory and polar plots using Matplotlib."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    thetas_xz = np.array(data["theta_xz"])
    thetas_yz = np.array(data["theta_yz"])
    thetas_xz_pi = thetas_xz / np.pi
    thetas_yz_pi = thetas_yz / np.pi
    thetas_3d = np.array(data["theta_3d"])
    phis = np.array(data["phi"])
    pdgs = np.array(data["pdg"])

    # -------------------------------------------------------------------------
    # Plot 1: Full [-1, +1] Trajectory Angles in Units of pi (0 = Ahead, +/-1 = Backwards)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    bins_pi = np.linspace(-1.0, 1.0, 51)
    ax1.hist(thetas_xz_pi, bins=bins_pi, color="royalblue", alpha=0.75, edgecolor="darkblue", lw=1.2)
    ax1.axvline(0, color="forestgreen", ls="--", lw=2, label="Ahead (+z, 0)")
    ax1.axvline(-1.0, color="crimson", ls=":", lw=1.5, label=r"Backwards ($-z, \pm 1\pi$)")
    ax1.axvline(1.0, color="crimson", ls=":", lw=1.5)
    ax1.set_xlabel(r"$\theta_{xz} / \pi = \operatorname{atan2}(p_x, p_z) / \pi$ [$\pi$ units]", fontsize=12)
    ax1.set_ylabel(r"Veto Hits / $0.04\pi$", fontsize=12)
    ax1.set_title(r"Horizontal Trajectory Angle $\theta_{xz}$ (in units of $\pi$)", fontsize=13, fontweight="bold")
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax1.set_xticklabels([r"$-1.0\pi$" + "\n(Backward)", r"$-0.5\pi$" + "\n(Left, -x)", "0\n(Ahead, +z)", r"$+0.5\pi$" + "\n(Right, +x)", r"$+1.0\pi$" + "\n(Backward)"])
    ax1.legend(loc="upper center", frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.hist(thetas_yz_pi, bins=bins_pi, color="darkorange", alpha=0.75, edgecolor="brown", lw=1.2)
    ax2.axvline(0, color="forestgreen", ls="--", lw=2, label="Ahead (+z, 0)")
    ax2.axvline(-1.0, color="crimson", ls=":", lw=1.5, label=r"Backwards ($-z, \pm 1\pi$)")
    ax2.axvline(1.0, color="crimson", ls=":", lw=1.5)
    ax2.set_xlabel(r"$\theta_{yz} / \pi = \operatorname{atan2}(p_y, p_z) / \pi$ [$\pi$ units]", fontsize=12)
    ax2.set_ylabel(r"Veto Hits / $0.04\pi$", fontsize=12)
    ax2.set_title(r"Vertical Trajectory Angle $\theta_{yz}$ (in units of $\pi$)", fontsize=13, fontweight="bold")
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax2.set_xticklabels([r"$-1.0\pi$" + "\n(Backward)", r"$-0.5\pi$" + "\n(Down, -y)", "0\n(Ahead, +z)", r"$+0.5\pi$" + "\n(Up, +y)", r"$+1.0\pi$" + "\n(Backward)"])
    ax2.legend(loc="upper center", frameon=True)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    proj_png = os.path.join(output_dir, "veto_trajectory_z_axis_centered.png")
    proj_pdf = os.path.join(output_dir, "veto_trajectory_z_axis_centered.pdf")
    fig.savefig(proj_png, dpi=300)
    fig.savefig(proj_pdf)
    plt.close(fig)
    print(f"   -> Saved: {proj_png}")

    # -------------------------------------------------------------------------
    # Plot 2: Transverse Azimuth phi = atan2(py, px) Broken Down by Particle Species in units of pi
    # -------------------------------------------------------------------------
    from collections import Counter
    counts = Counter(pdgs)
    top_species = [p for p, _ in counts.most_common(3)]

    if top_species:
        n_plots = len(top_species)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), sharey=True)
        if n_plots == 1:
            axes = [axes]

        colors = ["royalblue", "crimson", "forestgreen", "purple"]
        for idx, (p_id, ax) in enumerate(zip(top_species, axes)):
            mask = (pdgs == p_id)
            phi_sub_pi = phis[mask] / np.pi
            pname = get_pdg_name(p_id)
            ax.hist(phi_sub_pi, bins=np.linspace(-1.0, 1.0, 35), color=colors[idx % len(colors)],
                    alpha=0.75, edgecolor="black", lw=1.1)
            ax.set_title(f"{pname} (N={len(phi_sub_pi)})", fontsize=13, fontweight="bold")
            ax.set_xlabel(r"$\phi / \pi = \operatorname{atan2}(p_y, p_x) / \pi$ [$\pi$ units]", fontsize=11)
            ax.set_xlim(-1.0, 1.0)
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.set_xticklabels([r"$-1.0\pi$ (-x)", r"$-0.5\pi$ (-y)", r"$0$ (+x)", r"$+0.5\pi$ (+y)", r"$+1.0\pi$ (-x)"])
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.set_ylabel(r"Veto Hits / $0.06\pi$", fontsize=11)

        fig.suptitle(r"Transverse Orientation $\phi / \pi = \operatorname{atan2}(p_y, p_x) / \pi$ by Species",
                     fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        phi_breakdown_png = os.path.join(output_dir, "veto_transverse_phi_by_species.png")
        phi_breakdown_pdf = os.path.join(output_dir, "veto_transverse_phi_by_species.pdf")
        fig.savefig(phi_breakdown_png, dpi=300)
        fig.savefig(phi_breakdown_pdf)
        plt.close(fig)
        print(f"   -> Saved: {phi_breakdown_png}")

    # -------------------------------------------------------------------------
    # Plot 3: Polar Plot (radius = theta_3D, angle = phi)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="polar")

    # r: theta in degrees (0 = forward +z, 90 = transverse, 180 = backward -z)
    thetas_deg = np.degrees(thetas_3d)

    if len(thetas_3d) > 20:
        hb = ax.hexbin(phis, thetas_deg, gridsize=35, cmap="plasma", mincnt=1)
        cb = fig.colorbar(hb, ax=ax, orientation="vertical", shrink=0.75, pad=0.1)
        cb.set_label("Veto MCPoints Density", fontsize=12)
    elif len(thetas_3d) > 0:
        ax.scatter(phis, thetas_deg, color="crimson", s=40, alpha=0.8, edgecolors="black", label="Veto Hits")
        ax.legend(loc="upper right")

    ax.set_ylim(0, 180)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_yticklabels([r"$0^\circ$ (+z, Ahead)", r"$45^\circ$", r"$90^\circ$ (Transv)", r"$135^\circ$", r"$180^\circ$ (-z, Backwards)"])
    ax.set_theta_zero_location("E")  # 0 rad along positive X axis
    ax.set_title(
        r"Veto Particle Trajectories in Polar Coordinates $(\theta_{3D}, \phi)$"
        "\nCenter = Ahead (+z), Outer Ring = Backwards (-z)",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    polar_png = os.path.join(output_dir, "veto_trajectory_polar_plot.png")
    polar_pdf = os.path.join(output_dir, "veto_trajectory_polar_plot.pdf")
    fig.savefig(polar_png, dpi=300)
    fig.savefig(polar_pdf)
    plt.close(fig)
    print(f"   -> Saved: {polar_png}")

    # -------------------------------------------------------------------------
    # Plot 4: Particle Species Breakdown (PDG IDs)
    # -------------------------------------------------------------------------
    if data["pdg"]:
        counts = Counter(data["pdg"])
        top_pdgs = counts.most_common(8)

        labels = [get_pdg_name(p) for p, _ in top_pdgs]
        vals = [c for _, c in top_pdgs]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, vals, color="mediumpurple", edgecolor="indigo", alpha=0.85, width=0.55)
        ax.set_ylabel("Veto MCPoints Count", fontsize=12)
        ax.set_title("Species Composition of Particles Causing Late Veto Hits", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + max(vals) * 0.015, f"{int(yval)}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        fig.tight_layout()
        pdg_png = os.path.join(output_dir, "veto_particle_species.png")
        pdg_pdf = os.path.join(output_dir, "veto_particle_species.pdf")
        fig.savefig(pdg_png, dpi=300)
        fig.savefig(pdg_pdf)
        plt.close(fig)
        print(f"   -> Saved: {pdg_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot particle trajectories reaching the Veto detector in MC events."
    )
    parser.add_argument(
        "-i", "--input",
        default="/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_MCEB_dig.root",
        help="Input ROOT file path or pattern containing '%%s' placeholder (default: 2022 MCEB dig).",
    )
    parser.add_argument(
        "-p", "--partitions",
        default="1",
        help="Partitions to process: range ('1-10'), comma-separated ('1,2,5'), or 'all' (default: 1).",
    )
    parser.add_argument(
        "-f", "--partitions-file",
        default=None,
        help="Text file with partition IDs (one per line).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="plots/veto_backsplash_trajectories",
        help="Output directory for generated plots & ROOT file (default: plots/veto_backsplash_trajectories).",
    )
    parser.add_argument(
        "--root-output",
        default="veto_backsplash_trajectories.root",
        help="Output ROOT filename containing histograms & canvases (default: veto_backsplash_trajectories.root).",
    )
    parser.add_argument(
        "--condition",
        default="earliest_veto_later_than_latest_scifi",
        choices=[
            "earliest_veto_later_than_latest_scifi",
            "earliest_veto_later_than_earliest_scifi",
            "latest_veto_later_than_earliest_scifi",
            "all_veto",
        ],
        help="Selection condition: 'earliest_veto_later_than_latest_scifi' (default), 'earliest_veto_later_than_earliest_scifi', 'latest_veto_later_than_earliest_scifi', or 'all_veto'.",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of concurrent processes (default: 4).",
    )
    parser.add_argument(
        "-n", "--max-events",
        type=int,
        default=-1,
        help="Maximum events to analyze per partition (-1 for all, default: -1).",
    )

    args = parser.parse_args()

    # Determine partition list
    partitions = []
    if args.partitions_file and os.path.isfile(args.partitions_file):
        with open(args.partitions_file, "r") as pf:
            for line in pf:
                line = line.strip()
                if line and not line.startswith("#") and line.isdigit():
                    partitions.append(int(line))
        partitions = sorted(list(set(partitions)))
    elif "%s" in args.input:
        partitions = parse_partitions(args.partitions)
    else:
        partitions = [1]

    print("=" * 80)
    print(" VETO BACKSPLASH TRAJECTORY & POLAR DIAGNOSTIC (MC ONLY)")
    print("=" * 80)
    print(f" Input Pattern       : {args.input}")
    print(f" Partitions          : {len(partitions)} partition(s)")
    print(f" Timing Condition    : {args.condition}")
    print(f" Output Directory    : {args.output_dir}")
    print(f" Output ROOT File    : {args.root_output}")
    print(f" Max Events / Part   : {'All' if args.max_events <= 0 else args.max_events}")
    print(f" Parallel Workers    : {args.jobs}")
    print("=" * 80)

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect file paths
    files_to_process = []
    for p in partitions:
        fpath = format_partition_path(args.input, p)
        if os.path.isfile(fpath):
            files_to_process.append(fpath)

    if not files_to_process:
        print("ERROR: No valid input files found.", file=sys.stderr)
        sys.exit(1)

    print(f" Found {len(files_to_process)} input files to analyze.")

    # Process files
    results = []
    if args.jobs > 1 and len(files_to_process) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(process_single_file, fp, args.condition, args.max_events): fp
                for fp in files_to_process
            }
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for fp in files_to_process:
            results.append(process_single_file(fp, args.condition, args.max_events))

    # Merge results
    merged_data: Dict[str, Any] = {
        "theta_xz": [],
        "theta_yz": [],
        "theta_3d": [],
        "phi": [],
        "p_tot": [],
        "pz": [],
        "pt": [],
        "pdg": [],
        "n_events_total": 0,
        "n_events_veto": 0,
        "n_events_passed": 0,
        "n_veto_points": 0,
    }

    for r in results:
        merged_data["n_events_total"] += r["n_events_total"]
        merged_data["n_events_veto"] += r["n_events_veto"]
        merged_data["n_events_passed"] += r["n_events_passed"]
        merged_data["n_veto_points"] += r["n_veto_points"]
        for k in ["theta_xz", "theta_yz", "theta_3d", "phi", "p_tot", "pz", "pt", "pdg"]:
            merged_data[k].extend(r[k])

    print(f"\n Processed {merged_data['n_events_total']} events in {time.time() - t0:.2f}s.")
    print(f" Events with Veto Hits        : {merged_data['n_events_veto']}")
    print(f" Events Passing Timing Filter : {merged_data['n_events_passed']}")
    print(f" Total Veto MCPoints Paired   : {merged_data['n_veto_points']}")

    if merged_data["n_veto_points"] == 0:
        print("WARNING: No Veto MCPoints found matching criteria.", file=sys.stderr)
        sys.exit(0)

    # Particle species breakdown
    from collections import Counter
    counts = Counter(merged_data["pdg"])
    print("\n Dominant Particle Species in Veto:")
    for pdg, c in counts.most_common(6):
        pct = (c / merged_data["n_veto_points"]) * 100.0
        print(f"   * {get_pdg_name(pdg):<18} (PDG {pdg:>6}): {c:>5} points ({pct:>5.1f}%)")

    # Angular summary
    thetas_deg = [math.degrees(th) for th in merged_data["theta_3d"]]
    mean_th = sum(thetas_deg) / len(thetas_deg)
    backwards_count = sum(1 for th in thetas_deg if th > 90.0)
    print(f"\n 3D Polar Angle (#theta) Summary:")
    print(f"   * Mean #theta                 : {mean_th:.1f}°")
    print(f"   * Min / Max #theta            : {min(thetas_deg):.1f}° / {max(thetas_deg):.1f}°")
    print(f"   * Backwards (#theta > 90°)    : {backwards_count}/{len(thetas_deg)} ({backwards_count/len(thetas_deg)*100:.1f}%)")

    # ROOT histograms & canvases
    hists = create_root_histograms(merged_data)
    canvases = build_root_canvases(hists)

    root_filepath = os.path.join(args.output_dir, args.root_output)
    f_out = ROOT.TFile.Open(root_filepath, "RECREATE")
    for h in hists.values():
        h.Write()
    for c in canvases.values():
        c.Write()
    f_out.Close()
    print(f"\n [ROOT] Histograms & Canvases saved to: {root_filepath}")

    for cname, c in canvases.items():
        png_path = os.path.join(args.output_dir, f"{cname}.png")
        pdf_path = os.path.join(args.output_dir, f"{cname}.pdf")
        c.Print(png_path)
        c.Print(pdf_path)

    # Matplotlib polar & projected angle figures
    print("\n [Matplotlib] Exporting projected angle, polar, and species plots:")
    plot_polar_and_projections_matplotlib(merged_data, args.output_dir)

    print("=" * 80)
    print(" Veto trajectory diagnostic completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
