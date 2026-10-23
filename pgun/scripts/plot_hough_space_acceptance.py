#!/usr/bin/env python3
"""
plot_hough_space_acceptance.py
------------------------------
Plots 2D Hough spaces (linearSlopeIntercept) for dimuon events and highlights
the bins corresponding to tracks that DO NOT cross the target plane at z=352 cm:
    Target window: -50 <= x <= -3 cm, 5 <= y <= 60 cm

Generates 4-panel diagnostic visualizations:
  1. ZX Hough Accumulator (slope_zx vs intercept_zx) with target crossing boundary
  2. ZY Hough Accumulator (slope_zy vs intercept_zy) with target crossing boundary
  3. Real-space ZX display (z vs x) showing DS hits, tracks, and target box
  4. Real-space ZY display (z vs y) showing DS hits, tracks, and target box

Generates plots for:
  - 20 Non-signal background events that were reconstructed with 2 Hough tracks
  - 20 Particle Gun signal events
  - 20 Genie Direct Signal events
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import ddfUtils.mpl_styling as dms
    has_dms = True
except ImportError:
    has_dms = False

from pgun.scripts.tune_dimuon_hough import (
    StandaloneHough,
    hit_finder,
    num_planes_hit,
    run_reco_for_event,
)


def compute_hough_accumulator(
    event_data: Dict[str, Any],
    params: Dict[str, Any],
    proj: str = "zx",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandaloneHough]:
    """Compute and return (raw_accumulator, smoothed_accumulator, xH_bins, yH_bins, hough_obj)."""
    pos = event_data["pos"]
    d = event_data["d"]
    vert = event_data["vert"]
    system = event_data["system"]
    z_offset = event_data["z_offset"]
    det_zlen = event_data["det_zlen"]

    if proj == "zx":
        triplet = vert & (system == 3)
        pts = np.column_stack([pos[2, triplet], pos[0, triplet]])
        d_pts = np.column_stack([d[2, triplet], d[0, triplet]])
        yH_range = params["yH_range_xz"]
        xH_range = params["xH_range_xz"]
    else:  # zy
        triplet = (~vert) & (system == 3)
        pts = np.column_stack([pos[2, triplet], pos[1, triplet]])
        d_pts = np.column_stack([d[2, triplet], d[1, triplet]])
        yH_range = params["yH_range_yz"]
        xH_range = params["xH_range_yz"]

    h = StandaloneHough(
        n_yH=params["n_yH"],
        yH_range=yH_range,
        n_xH=params["n_xH"],
        xH_range=xH_range,
        z_offset=z_offset,
        h_format=params["h_format"],
        det_zlen=det_zlen,
        smooth_full=params["smooth_full"],
        sigma=params["sigma"],
        truncate=params["truncate"],
        n_quantile=0.5,
        res=5.0,
    )

    if len(pts) == 0:
        acc_raw = np.zeros((params["n_yH"], params["n_xH"]), dtype=np.float32)
        return acc_raw, acc_raw, h.xH_bins, h.yH_bins, h

    shifted_z = pts[:, 0] - z_offset
    y_vals = pts[:, 1]
    hit_yH = y_vals[:, None] - shifted_z[:, None] * h.xH_bins[None, :]
    valid = (hit_yH >= yH_range[0]) & (hit_yH <= yH_range[1])

    idx_y = np.floor((hit_yH - yH_range[0]) / (yH_range[1] - yH_range[0]) * params["n_yH"]).astype(int)
    idx_y = np.clip(idx_y, 0, params["n_yH"] - 1)
    col_idx = np.broadcast_to(h.xH_i, hit_yH.shape)

    acc_raw = np.zeros((params["n_yH"], params["n_xH"]), dtype=np.float32)
    np.add.at(acc_raw, (idx_y[valid], col_idx[valid]), 1.0)

    if params["smooth_full"]:
        acc_smooth = scipy.ndimage.gaussian_filter(acc_raw, params["sigma"], truncate=params["truncate"])
    else:
        acc_smooth = acc_raw.copy()

    return acc_raw, acc_smooth, h.xH_bins, h.yH_bins, h


def plot_event_hough_acceptance(
    event_data: Dict[str, Any],
    params: Dict[str, Any],
    output_png: str,
    sample_label: str = "Non-Signal Background",
    z_target: float = 352.0,
    x_target_range: Tuple[float, float] = (-50.0, -3.0),
    y_target_range: Tuple[float, float] = (5.0, 60.0),
    angle_cut: float = 0.175,
):
    """Generate 4-panel diagnostic plot for one event."""
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)

    z_offset = event_data["z_offset"]  # typically 430.0 cm
    dz_target = z_target - z_offset     # 352 - 430 = -78.0 cm

    # Run reconstruction to get tracks (returns slope m and b0 at z=0)
    tracks = run_reco_for_event(event_data, params)

    # Accumulators (raw vote count and smoothed)
    acc_zx_raw, acc_zx_smooth, x_bins_zx, y_bins_zx, _ = compute_hough_accumulator(event_data, params, proj="zx")
    acc_zy_raw, acc_zy_smooth, x_bins_zy, y_bins_zy, _ = compute_hough_accumulator(event_data, params, proj="zy")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ["gold", "magenta", "lime", "cyan"]

    # -----------------------------------------------------------------------
    # Panel 1: ZX Hough Space (slope_zx vs intercept_zx at z_offset)
    # -----------------------------------------------------------------------
    ax = axes[0, 0]
    extent_zx = [x_bins_zx[0], x_bins_zx[-1], y_bins_zx[0], y_bins_zx[-1]]
    im1 = ax.imshow(acc_zx_raw, origin="lower", extent=extent_zx, aspect="auto", cmap="viridis", interpolation="nearest")
    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label("Votes in Bin", fontsize=11)

    # Angular acceptance edges and transparent shading (-0.175 rad <= xz_angle <= 0.175 rad)
    ax.axvline(-angle_cut, color="cyan", linestyle=":", linewidth=2.0, alpha=0.9, label=f"Angle Edge ({angle_cut:+.3f} rad)")
    ax.axvline(angle_cut, color="cyan", linestyle=":", linewidth=2.0, alpha=0.9)
    ax.axvspan(-angle_cut, angle_cut, color="cyan", alpha=0.08, label=rf"$|\theta_{{xz}}| \leq {angle_cut}$ rad")

    # Outline boundary of bins whose Hough lines miss the target window at z=352
    m_arr = np.linspace(x_bins_zx[0], x_bins_zx[-1], 200)
    b_min_zx = x_target_range[0] - m_arr * dz_target  # -50 + 78*m
    b_max_zx = x_target_range[1] - m_arr * dz_target  # -3  + 78*m
    ax.plot(m_arr, b_min_zx, color="red", linestyle="--", linewidth=2.0, label=f"Target Bound x={x_target_range[0]:.0f} cm @ z={z_target:.0f}")
    ax.plot(m_arr, b_max_zx, color="red", linestyle="-.", linewidth=2.0, label=f"Target Bound x={x_target_range[1]:.0f} cm @ z={z_target:.0f}")

    # Mark reconstructed tracks at their accumulator peak
    for idx, trk in enumerate(tracks):
        m_val = trk["tx"]
        b_ds = trk["bx"] + m_val * z_offset
        x_at_352 = m_val * z_target + trk["bx"]
        stat = "IN" if (x_target_range[0] <= x_at_352 <= x_target_range[1]) else "MISS"
        lbl = f"Reco Trk {idx+1}: m={m_val:+.3f}, b_ds={b_ds:+.1f} -> x(352)={x_at_352:+.1f} [{stat}]"
        ax.plot(m_val, b_ds, marker="*", markersize=14, color=colors[idx % len(colors)],
                markeredgecolor="black", markeredgewidth=1.2, label=lbl)

    ax.set_xlim(x_bins_zx[0], x_bins_zx[-1])
    ax.set_ylim(y_bins_zx[0], y_bins_zx[-1])
    ax.set_xlabel(r"Slope $m_{zx} = dx/dz$ [rad]", fontsize=11)
    ax.set_ylabel(r"Intercept at DS start $b_{zx}$ [cm]", fontsize=11)
    ax.set_title(r"ZX Hough Space (2D Histogram Accumulator)", fontsize=12, weight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # -----------------------------------------------------------------------
    # Panel 2: ZY Hough Space (slope_zy vs intercept_zy at z_offset)
    # -----------------------------------------------------------------------
    ax = axes[0, 1]
    extent_zy = [x_bins_zy[0], x_bins_zy[-1], y_bins_zy[0], y_bins_zy[-1]]
    im2 = ax.imshow(acc_zy_raw, origin="lower", extent=extent_zy, aspect="auto", cmap="viridis", interpolation="nearest")
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label("Votes in Bin", fontsize=11)

    # Angular acceptance edges and transparent shading (-0.175 rad <= yz_angle <= 0.175 rad)
    ax.axvline(-angle_cut, color="cyan", linestyle=":", linewidth=2.0, alpha=0.9, label=f"Angle Edge ({angle_cut:+.3f} rad)")
    ax.axvline(angle_cut, color="cyan", linestyle=":", linewidth=2.0, alpha=0.9)
    ax.axvspan(-angle_cut, angle_cut, color="cyan", alpha=0.08, label=rf"$|\theta_{{yz}}| \leq {angle_cut}$ rad")

    # Outline boundary of bins whose Hough lines miss the target window at z=352
    b_min_zy = y_target_range[0] - m_arr * dz_target  # 5  + 78*m
    b_max_zy = y_target_range[1] - m_arr * dz_target  # 60 + 78*m
    ax.plot(m_arr, b_min_zy, color="red", linestyle="--", linewidth=2.0, label=f"Target Bound y={y_target_range[0]:.0f} cm @ z={z_target:.0f}")
    ax.plot(m_arr, b_max_zy, color="red", linestyle="-.", linewidth=2.0, label=f"Target Bound y={y_target_range[1]:.0f} cm @ z={z_target:.0f}")

    # Mark reconstructed tracks at their accumulator peak
    for idx, trk in enumerate(tracks):
        m_val = trk["ty"]
        b_ds = trk["by"] + m_val * z_offset
        y_at_352 = m_val * z_target + trk["by"]
        stat = "IN" if (y_target_range[0] <= y_at_352 <= y_target_range[1]) else "MISS"
        lbl = f"Reco Trk {idx+1}: m={m_val:+.3f}, b_ds={b_ds:+.1f} -> y(352)={y_at_352:+.1f} [{stat}]"
        ax.plot(m_val, b_ds, marker="*", markersize=14, color=colors[idx % len(colors)],
                markeredgecolor="black", markeredgewidth=1.2, label=lbl)

    ax.set_xlim(x_bins_zy[0], x_bins_zy[-1])
    ax.set_ylim(y_bins_zy[0], y_bins_zy[-1])
    ax.set_xlabel(r"Slope $m_{zy} = dy/dz$ [rad]", fontsize=11)
    ax.set_ylabel(r"Intercept at DS start $b_{zy}$ [cm]", fontsize=11)
    ax.set_title(r"ZY Hough Space (2D Histogram Accumulator)", fontsize=12, weight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # -----------------------------------------------------------------------
    # Panel 3: Real Space ZX Projection (z vs x)
    # -----------------------------------------------------------------------
    ax = axes[1, 0]
    pos = event_data["pos"]
    d = event_data["d"]
    vert = event_data["vert"]
    system = event_data["system"]

    # Vertical bars (measure x): draw detector hit boxes
    v_mask = vert & (system == 3)
    first_hit = True
    for zi, xi, dzi, dxi in zip(pos[2, v_mask], pos[0, v_mask], d[2, v_mask], d[0, v_mask]):
        rect = patches.Rectangle((zi - dzi/2, xi - dxi/2), dzi, dxi,
                                 linewidth=0.8, edgecolor="navy", facecolor="dodgerblue", alpha=0.6,
                                 label="DS Vertical Hits (x)" if first_hit else "")
        first_hit = False
        ax.add_patch(rect)

    # Plot Target Window at z=352
    ax.plot([z_target, z_target], [x_target_range[0], x_target_range[1]],
            color="forestgreen", linewidth=4.5, label=f"Target Window [{x_target_range[0]:.0f}, {x_target_range[1]:.0f}] cm")

    # Draw reconstructed tracks: x(z) = m_x * z + b_x (where b_x is intercept at z=0)
    z_line = np.linspace(z_target - 15, 565, 300)
    for idx, trk in enumerate(tracks):
        col = colors[idx % len(colors)]
        x_line = trk["tx"] * z_line + trk["bx"]
        ax.plot(z_line, x_line, color=col, linewidth=2.2, linestyle="-", label=f"Reco Trk {idx+1}")

    ax.axvline(z_offset, color="gray", linestyle=":", alpha=0.7, label=f"DS start (z={z_offset:.0f})")
    ax.set_xlim(z_target - 20, 570)
    ax.set_ylim(-110, 25)
    ax.set_xlabel("z [cm]", fontsize=11)
    ax.set_ylabel("x [cm]", fontsize=11)
    ax.set_title("Real-Space ZX Projection (extrapolated to Target)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)

    # -----------------------------------------------------------------------
    # Panel 4: Real Space ZY Projection (z vs y)
    # -----------------------------------------------------------------------
    ax = axes[1, 1]
    # Horizontal bars (measure y): draw detector hit boxes
    h_mask = (~vert) & (system == 3)
    first_hit_y = True
    for zi, yi, dzi, dyi in zip(pos[2, h_mask], pos[1, h_mask], d[2, h_mask], d[1, h_mask]):
        rect = patches.Rectangle((zi - dzi/2, yi - dyi/2), dzi, dyi,
                                 linewidth=0.8, edgecolor="navy", facecolor="dodgerblue", alpha=0.6,
                                 label="DS Horizontal Hits (y)" if first_hit_y else "")
        first_hit_y = False
        ax.add_patch(rect)

    # Plot Target Window at z=352
    ax.plot([z_target, z_target], [y_target_range[0], y_target_range[1]],
            color="forestgreen", linewidth=4.5, label=f"Target Window [{y_target_range[0]:.0f}, {y_target_range[1]:.0f}] cm")

    # Draw reconstructed tracks: y(z) = m_y * z + b_y
    for idx, trk in enumerate(tracks):
        col = colors[idx % len(colors)]
        y_line = trk["ty"] * z_line + trk["by"]
        ax.plot(z_line, y_line, color=col, linewidth=2.2, linestyle="-", label=f"Reco Trk {idx+1}")

    ax.axvline(z_offset, color="gray", linestyle=":", alpha=0.7, label=f"DS start (z={z_offset:.0f})")
    ax.set_xlim(z_target - 20, 570)
    ax.set_ylim(-10, 130)
    ax.set_xlabel("z [cm]", fontsize=11)
    ax.set_ylabel("y [cm]", fontsize=11)
    ax.set_title("Real-Space ZY Projection (extrapolated to Target)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    ev_id = event_data.get("event_id", "unknown")
    plt.suptitle(f"Event Hough Space & Target Acceptance: {sample_label} (Event {ev_id})\nTarget at z={z_target} cm: x in [{x_target_range[0]}, {x_target_range[1]}], y in [{y_target_range[0]}, {y_target_range[1]}]",
                 fontsize=14, weight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    pdf_path = output_png.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Hough spaces with target acceptance boundary")
    parser.add_argument("--bkg-cache", default="pgun/cache_genie_bkg_20ev_2trks.pkl", help="2-track background cache")
    parser.add_argument("--pmu-cache", default="pgun/cache_pmu_bkg_500ev.pkl", help="pmu background cache fallback")
    parser.add_argument("--pgun-cache", default="pgun/tuning_cache_500ev.pkl", help="Particle Gun cache")
    parser.add_argument("--genie-sig-cache", default="pgun/tuning_cache_genie_341ev.pkl", help="Genie Signal cache")
    parser.add_argument("-n", "--n-events", type=int, default=20, help="Number of events per category")
    parser.add_argument("-o", "--out-base", default="/eos/user/i/idioniso/snd-numu-charm/plots/hough_space",
                        help="Base output directory on EOS")
    args = parser.parse_args()

    # User's exact Hough parameters
    params = {
        "h_format": "linearSlopeIntercept",
        "n_yH": 335,
        "n_xH": 605,
        "yH_range_xz": (-105.0, 17.0),
        "yH_range_yz": (0.0, 122.0),
        "xH_range_xz": (-1.1, 1.1),
        "xH_range_yz": (-1.1, 1.1),
        "tolerance": 0.4,
        "smooth_full": True,
        "sigma": 1.0,
        "truncate": 2.0,
        "max_angle": 1.0,
        "min_planes_hit": 3,
        "max_reco_muons": 2,
        "n_random": 0,
    }

    # 1. Background events (must have 2 Hough tracks)
    bkg_events = []
    if os.path.isfile(args.bkg_cache):
        with open(args.bkg_cache, "rb") as f:
            bkg_events = pickle.load(f)
    if len(bkg_events) < args.n_events and os.path.isfile(args.pmu_cache):
        with open(args.pmu_cache, "rb") as f:
            pmu_evs = pickle.load(f)
        for ev in pmu_evs:
            if len(bkg_events) >= args.n_events:
                break
            if len(run_reco_for_event(ev, params)) == 2:
                bkg_events.append(ev)

    print(f"Loaded {len(bkg_events)} background events with 2 tracks.")

    # 2. Particle Gun events
    with open(args.pgun_cache, "rb") as f:
        pgun_all = pickle.load(f)
    pgun_events = pgun_all[:args.n_events]
    print(f"Loaded {len(pgun_events)} Particle Gun events.")

    # 3. Genie Signal events
    with open(args.genie_sig_cache, "rb") as f:
        genie_all = pickle.load(f)
    genie_events = genie_all[:args.n_events]
    print(f"Loaded {len(genie_events)} Genie Signal events.")

    # Generate plots
    categories = [
        ("genie_bkg", bkg_events[:args.n_events], "Non-Signal Background (2 Reconstructed Tracks)"),
        ("pgun_sig",  pgun_events,               "Particle Gun Dimuon Signal"),
        ("genie_sig", genie_events,              "GENIE Direct Charm Dimuon Signal"),
    ]

    for cat_name, ev_list, lbl in categories:
        cat_dir = os.path.join(args.out_base, cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        print(f"\nGenerating {len(ev_list)} plots for {cat_name} in {cat_dir}...")
        for idx, ev in enumerate(ev_list):
            ev_id = ev.get("event_id", f"{idx}")
            out_file = os.path.join(cat_dir, f"hough_space_{cat_name}_ev{idx+1:02d}_{ev_id}.png")
            plot_event_hough_acceptance(ev, params, out_file, sample_label=lbl)
            if (idx + 1) % 5 == 0 or (idx + 1) == len(ev_list):
                print(f"  [{idx+1}/{len(ev_list)}] Saved {os.path.basename(out_file)}")

    print(f"\nAll plots successfully generated and stored in {args.out_base}/")


if __name__ == "__main__":
    main()
