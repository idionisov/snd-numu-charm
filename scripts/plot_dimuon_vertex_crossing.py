#!/usr/bin/env python3
"""
scripts/plot_dimuon_vertex_crossing.py
--------------------------------------
Analyzes the longitudinal vertex resolution of 2D reconstructed Hough line pairs:
  z_{lineVtx} = - (b1 - b2) / (m1 - m2)
compared against the true neutrino interaction vertex z_{vtx} and charm decay vertex z_{charmDecay}.

Generates:
1. 1D distributions of (z_{vtx} - z_{lineVtx}) separately for XZ and YZ projections,
   with central zoom-in [-150, 150] cm and 68% containment intervals.
2. 2D correlation between opening angle Delta theta and vertex error Delta z
   illustrating the impact of Multiple Coulomb Scattering (MCS) at small opening angles.
3. Distribution of reconstructed z_{lineVtx} compared against true z_{vtx} and SND@LHC detector boundaries.

Outputs saved to:
  /eos/user/i/idioniso/snd-numu-charm/plots/dimuon_vertex_crossing/
"""

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import ddfUtils.mpl_styling as dms
    has_dms = True
except ImportError:
    has_dms = False

from concurrent.futures import ProcessPoolExecutor

# Import Hough reconstruction infrastructure
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pgun", "scripts")))
from tune_dimuon_hough import StandaloneHough, run_reco_pairwise_event


# Optimal Rank 1 Hough Parameters from Systematic Tuning
RANK1_PARAMS = {
    "h_format": "linearSlopeIntercept",
    "binning_factor": 2.90,
    "n_xH": 638,
    "n_yH": 354,
    "xH_range_xz": (-1.1, 1.1),
    "yH_range_xz": (-105.0, 17.0),
    "xH_range_yz": (-1.1, 1.1),
    "yH_range_yz": (0.0, 122.0),
    "z_offset": 430.0,
    "det_zlen": 80.0,
    "z_target": 352.0,
    "target_window_xz": (-50.0, -3.0),
    "target_window_yz": (5.0, 60.0),
    "max_angle": 0.200,
    "max_opening_angle": 0.200,
    "min_opening_angle": 0.002,
    "min_unique_planes": 1,
    "tolerance": 0.5,
    "smooth_full": True,
    "sigma": 0.5,
    "truncate": 1.0,
    "n_random": 0,
    "use_pairwise": True,
}


def _extract_single_vtx(item):
    """Worker function to read mctruth vtx coordinates via uproot."""
    import uproot
    eid = item["event_id"]
    part = int(eid.split("_")[0].replace("part", ""))
    entry = int(eid.split("_")[1].replace("ev", ""))
    path = f"/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/{part}/sndLHC.Genie-TGeant4_dig_2MuTrks_truth_charmDimuonDirect.root"
    try:
        with uproot.open(path) as f:
            t = f["mctruth"]
            vz = float(t["vtx_z"].array(entry_start=entry, entry_stop=entry + 1)[0])
            vx = float(t["vtx_x"].array(entry_start=entry, entry_stop=entry + 1)[0])
            vy = float(t["vtx_y"].array(entry_start=entry, entry_stop=entry + 1)[0])
            cdz = float(t["charm_decay_z"].array(entry_start=entry, entry_stop=entry + 1)[0])
            return eid, vx, vy, vz, cdz
    except Exception as e:
        return eid, None, None, None, None


def get_or_create_vertex_cache(events, cache_path="pgun/genie_signal_vertices_215ev.pkl", workers=12):
    """Retrieve or build the truth vertex cache for GENIE signal events."""
    if os.path.exists(cache_path):
        print(f"Loading vertex cache from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Extracting truth vertices for {len(events)} events using {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        raw_results = list(pool.map(_extract_single_vtx, events))

    vtx_dict = {}
    for eid, vx, vy, vz, cdz in raw_results:
        if vz is not None:
            vtx_dict[eid] = {
                "vtx_x": vx,
                "vtx_y": vy,
                "vtx_z": vz,
                "charm_decay_z": cdz,
            }

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(vtx_dict, f)
    print(f"Saved {len(vtx_dict)} truth vertices to {cache_path}.")
    return vtx_dict


def compute_quantiles(arr, quantiles=(0.16, 0.50, 0.84)):
    """Compute quantiles and 68% containment interval."""
    clean = arr[np.isfinite(arr)]
    if len(clean) == 0:
        return np.nan, np.nan, np.nan, np.nan
    q16, q50, q84 = np.quantile(clean, quantiles)
    half_width = 0.5 * (q84 - q16)
    return q50, q16, q84, half_width


def main():
    parser = argparse.ArgumentParser(description="Analyze 2D Line Crossing Vertex Resolution.")
    parser.add_argument("--cache", default="pgun/tuning_cache_genie_341ev.pkl", help="GENIE signal cache")
    parser.add_argument("--vtx-cache", default="pgun/genie_signal_vertices_215ev.pkl", help="Vertex cache")
    parser.add_argument("--out-dir", default="/eos/user/i/idioniso/snd-numu-charm/plots/dimuon_vertex_crossing", help="Output directory")
    parser.add_argument("-w", "--workers", type=int, default=12, help="Number of worker processes")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading GENIE signal cache: {args.cache}...")
    with open(args.cache, "rb") as f:
        events = pickle.load(f)
    print(f"Loaded {len(events)} events.")

    vtx_dict = get_or_create_vertex_cache(events, cache_path=args.vtx_cache, workers=args.workers)

    # Initialize Hough Accumulators
    print("Initializing Hough Accumulators with Rank 1 parameters...")
    h_zx = StandaloneHough(
        RANK1_PARAMS["n_yH"], RANK1_PARAMS["yH_range_xz"],
        RANK1_PARAMS["n_xH"], RANK1_PARAMS["xH_range_xz"],
        RANK1_PARAMS["z_offset"], RANK1_PARAMS["h_format"], RANK1_PARAMS["det_zlen"],
        smooth_full=RANK1_PARAMS["smooth_full"], sigma=RANK1_PARAMS["sigma"], truncate=RANK1_PARAMS["truncate"],
        max_angle=RANK1_PARAMS["max_angle"], target_window=RANK1_PARAMS["target_window_xz"], z_target=RANK1_PARAMS["z_target"]
    )
    h_zy = StandaloneHough(
        RANK1_PARAMS["n_yH"], RANK1_PARAMS["yH_range_yz"],
        RANK1_PARAMS["n_xH"], RANK1_PARAMS["xH_range_yz"],
        RANK1_PARAMS["z_offset"], RANK1_PARAMS["h_format"], RANK1_PARAMS["det_zlen"],
        smooth_full=RANK1_PARAMS["smooth_full"], sigma=RANK1_PARAMS["sigma"], truncate=RANK1_PARAMS["truncate"],
        max_angle=RANK1_PARAMS["max_angle"], target_window=RANK1_PARAMS["target_window_yz"], z_target=RANK1_PARAMS["z_target"]
    )

    results = []
    print("Running pairwise Hough reconstruction over events...")
    for ev in events:
        eid = ev["event_id"]
        if eid not in vtx_dict:
            continue
        vinfo = vtx_dict[eid]
        z_vtx = vinfo["vtx_z"]
        z_charm = vinfo["charm_decay_z"]

        tracks = run_reco_pairwise_event(ev, RANK1_PARAMS, h_zy=h_zy, h_zx=h_zx)
        if len(tracks) < 2:
            continue

        t1, t2 = tracks[0], tracks[1]

        # XZ projection
        dmx = t1["tx"] - t2["tx"]
        dbx = t1["bx"] - t2["bx"]
        if abs(dmx) > 1e-6:
            zx_cross = -dbx / dmx
            dz_x = z_vtx - zx_cross
            dz_x_charm = z_charm - zx_cross
        else:
            zx_cross = np.nan
            dz_x = np.nan
            dz_x_charm = np.nan

        # YZ projection
        dmy = t1["ty"] - t2["ty"]
        dby = t1["by"] - t2["by"]
        if abs(dmy) > 1e-6:
            zy_cross = -dby / dmy
            dz_y = z_vtx - zy_cross
            dz_y_charm = z_charm - zy_cross
        else:
            zy_cross = np.nan
            dz_y = np.nan
            dz_y_charm = np.nan

        results.append({
            "eid": eid,
            "z_vtx": z_vtx,
            "z_charm": z_charm,
            "zx_cross": zx_cross,
            "zy_cross": zy_cross,
            "dz_x": dz_x,
            "dz_y": dz_y,
            "dz_x_charm": dz_x_charm,
            "dz_y_charm": dz_y_charm,
            "d_theta_x": abs(dmx),
            "d_theta_y": abs(dmy),
            "m1x": t1["tx"], "m2x": t2["tx"],
            "m1y": t1["ty"], "m2y": t2["ty"],
        })

    print(f"Successfully reconstructed track pairs for {len(results)} / {len(events)} events ({len(results)/len(events)*100:.1f}%).")

    # Convert to arrays
    dz_x_arr = np.array([r["dz_x"] for r in results])
    dz_y_arr = np.array([r["dz_y"] for r in results])
    zx_arr = np.array([r["zx_cross"] for r in results])
    zy_arr = np.array([r["zy_cross"] for r in results])
    zvtx_arr = np.array([r["z_vtx"] for r in results])
    dth_x_arr = np.array([r["d_theta_x"] for r in results])
    dth_y_arr = np.array([r["d_theta_y"] for r in results])

    # Compute statistics
    med_x, q16_x, q84_x, w68_x = compute_quantiles(dz_x_arr)
    med_y, q16_y, q84_y, w68_y = compute_quantiles(dz_y_arr)

    frac_x_50 = np.mean(np.abs(dz_x_arr[np.isfinite(dz_x_arr)]) < 50.0) * 100.0
    frac_x_100 = np.mean(np.abs(dz_x_arr[np.isfinite(dz_x_arr)]) < 100.0) * 100.0
    frac_y_50 = np.mean(np.abs(dz_y_arr[np.isfinite(dz_y_arr)]) < 50.0) * 100.0
    frac_y_100 = np.mean(np.abs(dz_y_arr[np.isfinite(dz_y_arr)]) < 100.0) * 100.0

    print("=========================================================================")
    print(" VERTEX RESOLUTION SUMMARY: z_{vtx} - z_{lineVtx}")
    print("=========================================================================")
    print(f"XZ Projection: Median = {med_x:+.1f} cm | 68% containment width = {w68_x:.1f} cm ([{q16_x:.1f}, {q84_x:.1f}] cm)")
    print(f"               Within +/-50 cm: {frac_x_50:.1f}% | Within +/-100 cm: {frac_x_100:.1f}%")
    print(f"YZ Projection: Median = {med_y:+.1f} cm | 68% containment width = {w68_y:.1f} cm ([{q16_y:.1f}, {q84_y:.1f}] cm)")
    print(f"               Within +/-50 cm: {frac_y_50:.1f}% | Within +/-100 cm: {frac_y_100:.1f}%")
    print("=========================================================================")

    # -------------------------------------------------------------
    # Plot 1: 1D Distribution of z_{vtx} - z_{lineVtx} (XZ and YZ)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Style
    plt.rcParams["font.size"] = 12

    # XZ
    ax = axes[0]
    bins_zoom = np.linspace(-150, 150, 61)
    ax.hist(dz_x_arr, bins=bins_zoom, color="#1f77b4", alpha=0.7, edgecolor="black", label=r"Events ($N=%d$)" % len(dz_x_arr))
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, label=r"Ideal $\Delta z = 0$")
    ax.axvline(med_x, color="red", linestyle="-", linewidth=2.0, label=f"Median ({med_x:+.1f} cm)")
    ax.axvspan(q16_x, q84_x, color="red", alpha=0.15, label=f"68% Containment ($\\pm${w68_x:.1f} cm)")
    ax.set_title("XZ Projection (Vertical Bars)", fontsize=13, fontweight="bold")
    ax.set_xlabel(r"$z_{\mathrm{vtx}} - z_{\mathrm{lineVtx,\,XZ}}$ [cm]", fontsize=12)
    ax.set_ylabel("Candidate Events / 5 cm", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    stats_text_x = (
        f"Reconstructed: {len(dz_x_arr)} pairs\n"
        f"Median: {med_x:+.1f} cm\n"
        f"68% Width: $\\pm${w68_x:.1f} cm\n"
        f"$|\\Delta z| < 50$ cm: {frac_x_50:.1f}%\n"
        f"$|\\Delta z| < 100$ cm: {frac_x_100:.1f}%"
    )
    ax.text(0.04, 0.95, stats_text_x, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray"))

    # YZ
    ax = axes[1]
    ax.hist(dz_y_arr, bins=bins_zoom, color="#2ca02c", alpha=0.7, edgecolor="black", label=r"Events ($N=%d$)" % len(dz_y_arr))
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, label=r"Ideal $\Delta z = 0$")
    ax.axvline(med_y, color="red", linestyle="-", linewidth=2.0, label=f"Median ({med_y:+.1f} cm)")
    ax.axvspan(q16_y, q84_y, color="red", alpha=0.15, label=f"68% Containment ($\\pm${w68_y:.1f} cm)")
    ax.set_title("YZ Projection (Horizontal Bars)", fontsize=13, fontweight="bold")
    ax.set_xlabel(r"$z_{\mathrm{vtx}} - z_{\mathrm{lineVtx,\,YZ}}$ [cm]", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    stats_text_y = (
        f"Reconstructed: {len(dz_y_arr)} pairs\n"
        f"Median: {med_y:+.1f} cm\n"
        f"68% Width: $\\pm${w68_y:.1f} cm\n"
        f"$|\\Delta z| < 50$ cm: {frac_y_50:.1f}%\n"
        f"$|\\Delta z| < 100$ cm: {frac_y_100:.1f}%"
    )
    ax.text(0.04, 0.95, stats_text_y, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray"))

    fig.suptitle(r"Longitudinal Vertex Resolution $\Delta z = z_{\mathrm{vtx}} - z_{\mathrm{lineVtx}}$ (GENIE Direct Charm)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_p1_png = os.path.join(args.out_dir, "dimuon_vertex_z_difference.png")
    out_p1_pdf = os.path.join(args.out_dir, "dimuon_vertex_z_difference.pdf")
    plt.savefig(out_p1_png, dpi=300)
    plt.savefig(out_p1_pdf)
    plt.close()
    print(f"Saved: {out_p1_png}")

    # -------------------------------------------------------------
    # Plot 2: Error vs Opening Angle (Multiple Scattering Blow-up)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # XZ
    ax = axes[0]
    sc1 = ax.scatter(dth_x_arr * 1e3, dz_x_arr, color="#1f77b4", alpha=0.7, edgecolors="none", s=35)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5)
    ax.set_ylim(-300, 300)
    ax.set_xlim(0, 200)
    ax.set_xlabel(r"XZ Opening Angle $\Delta\theta_x = |m_{1x} - m_{2x}|$ [mrad]", fontsize=12)
    ax.set_ylabel(r"$z_{\mathrm{vtx}} - z_{\mathrm{lineVtx,\,XZ}}$ [cm]", fontsize=12)
    ax.set_title("XZ: Vertex Discrepancy vs Opening Angle", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Add theoretical envelope curves: delta_z = delta_x / Delta theta for delta_x = +/- 1 cm, +/- 2 cm
    th_grid = np.linspace(2, 200, 200) * 1e-3
    ax.plot(th_grid * 1e3, 1.0 / th_grid, color="darkred", linestyle=":", linewidth=1.5, label=r"$\pm 1$ cm detector position uncertainty")
    ax.plot(th_grid * 1e3, -1.0 / th_grid, color="darkred", linestyle=":", linewidth=1.5)
    ax.plot(th_grid * 1e3, 2.5 / th_grid, color="darkorange", linestyle="--", linewidth=1.5, label=r"$\pm 2.5$ cm MCS deflection envelope")
    ax.plot(th_grid * 1e3, -2.5 / th_grid, color="darkorange", linestyle="--", linewidth=1.5)
    ax.legend(loc="upper right", fontsize=10)

    # YZ
    ax = axes[1]
    sc2 = ax.scatter(dth_y_arr * 1e3, dz_y_arr, color="#2ca02c", alpha=0.7, edgecolors="none", s=35)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5)
    ax.set_ylim(-300, 300)
    ax.set_xlim(0, 200)
    ax.set_xlabel(r"YZ Opening Angle $\Delta\theta_y = |m_{1y} - m_{2y}|$ [mrad]", fontsize=12)
    ax.set_ylabel(r"$z_{\mathrm{vtx}} - z_{\mathrm{lineVtx,\,YZ}}$ [cm]", fontsize=12)
    ax.set_title("YZ: Vertex Discrepancy vs Opening Angle", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax.plot(th_grid * 1e3, 1.0 / th_grid, color="darkred", linestyle=":", linewidth=1.5, label=r"$\pm 1$ cm detector position uncertainty")
    ax.plot(th_grid * 1e3, -1.0 / th_grid, color="darkred", linestyle=":", linewidth=1.5)
    ax.plot(th_grid * 1e3, 2.5 / th_grid, color="darkorange", linestyle="--", linewidth=1.5, label=r"$\pm 2.5$ cm MCS deflection envelope")
    ax.plot(th_grid * 1e3, -2.5 / th_grid, color="darkorange", linestyle="--", linewidth=1.5)
    ax.legend(loc="upper right", fontsize=10)

    fig.suptitle("Impact of Small Opening Angles and Multiple Scattering on Longitudinal Vertexing", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_p2_png = os.path.join(args.out_dir, "dimuon_vertex_vs_opening_angle.png")
    out_p2_pdf = os.path.join(args.out_dir, "dimuon_vertex_vs_opening_angle.pdf")
    plt.savefig(out_p2_png, dpi=300)
    plt.savefig(out_p2_pdf)
    plt.close()
    print(f"Saved: {out_p2_png}")

    # -------------------------------------------------------------
    # Plot 3: Absolute Reconstructed z_{lineVtx} vs Detector Layout
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    bins_abs = np.linspace(150, 650, 51)

    # XZ
    ax = axes[0]
    ax.axvspan(290, 355, color="wheat", alpha=0.5, label="Target / SciFi (z: 290-355 cm)")
    ax.axvspan(380, 430, color="lightgray", alpha=0.5, label="US MuFilter (z: 380-430 cm)")
    ax.axvspan(430, 520, color="lightblue", alpha=0.5, label="DS MuFilter (z: 430-520 cm)")
    ax.hist(zvtx_arr, bins=bins_abs, color="black", histtype="step", linewidth=2.0, label="True Vertex $z_{\\mathrm{vtx}}$")
    ax.hist(zx_arr, bins=bins_abs, color="#1f77b4", alpha=0.6, edgecolor="black", label=r"Reco Crossing $z_{\mathrm{lineVtx,\,XZ}}$")
    ax.set_title("XZ Reconstructed Crossing $z$", fontsize=13, fontweight="bold")
    ax.set_xlabel(r"$z$ [cm]", fontsize=12)
    ax.set_ylabel("Candidate Events / 10 cm", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    # YZ
    ax = axes[1]
    ax.axvspan(290, 355, color="wheat", alpha=0.5, label="Target / SciFi (z: 290-355 cm)")
    ax.axvspan(380, 430, color="lightgray", alpha=0.5, label="US MuFilter (z: 380-430 cm)")
    ax.axvspan(430, 520, color="lightblue", alpha=0.5, label="DS MuFilter (z: 430-520 cm)")
    ax.hist(zvtx_arr, bins=bins_abs, color="black", histtype="step", linewidth=2.0, label="True Vertex $z_{\\mathrm{vtx}}$")
    ax.hist(zy_arr, bins=bins_abs, color="#2ca02c", alpha=0.6, edgecolor="black", label=r"Reco Crossing $z_{\mathrm{lineVtx,\,YZ}}$")
    ax.set_title("YZ Reconstructed Crossing $z$", fontsize=13, fontweight="bold")
    ax.set_xlabel(r"$z$ [cm]", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    fig.suptitle(r"Reconstructed 2D Line Crossing $z_{\mathrm{lineVtx}}$ vs Detector Geometry", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_p3_png = os.path.join(args.out_dir, "dimuon_lineVtx_z_distribution.png")
    out_p3_pdf = os.path.join(args.out_dir, "dimuon_lineVtx_z_distribution.pdf")
    plt.savefig(out_p3_png, dpi=300)
    plt.savefig(out_p3_pdf)
    plt.close()
    print(f"Saved: {out_p3_png}")


if __name__ == "__main__":
    main()
