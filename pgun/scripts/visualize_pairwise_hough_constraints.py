#!/usr/bin/env python3
"""
visualize_pairwise_hough_constraints.py
---------------------------------------
Comprehensive visualization of the 8D pairwise Hough reconstruction space for a dimuon event:
  - Panel 1: 2D Accumulator XZ (slope vs intercept) + identified candidate peaks
  - Panel 2: 2D Accumulator YZ (slope vs intercept) + identified candidate peaks
  - Panel 3: (z_cross_XZ vs z_cross_YZ) correlation plane with:
      * Longitudinal crossing window boundaries [zCross_min, zCross_max]
      * Transverse projection consistency band: |z_cross_XZ - z_cross_YZ| <= delta_zCross_max
      * Candidate 4-tuples evaluated for the event
  - Panel 4: Reconstructed Trajectories in detector space & common dimuon vertex
  - Panel 5: Pairwise Compatibility Matrix (Idea 3: Heatmap of delta zCross across X-pairs vs Y-pairs)
  - Panel 6: Downstream Isolation at DS3 plane (Idea 4: Delta x_DS3 vs Delta y_DS3 transverse separation)
  - Panel 7: Longitudinal Convergence Profile (Idea 5: Track separation |x1-x2| and |y1-y2| vs z)
"""

import os
import sys
import pickle
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import ddfUtils.mpl_styling as dms

# Add repo root to import tune_dimuon_hough components
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from pgun.scripts.tune_dimuon_hough import StandaloneHough, num_planes_hit


def compute_acc(hough_obj, hit_collection):
    shifted_z = hit_collection[:, 0] - hough_obj.z_offset
    y_vals = hit_collection[:, 1]
    hit_yH = y_vals[:, None] - shifted_z[:, None] * hough_obj.xH_bins[None, :]
    valid = (hit_yH >= hough_obj.yH_range[0]) & (hit_yH <= hough_obj.yH_range[1])
    idx_y = np.floor((hit_yH - hough_obj.yH_range[0]) / (hough_obj.yH_range[1] - hough_obj.yH_range[0]) * hough_obj.n_yH).astype(int)
    idx_y = np.clip(idx_y, 0, hough_obj.n_yH - 1)
    col_idx = np.broadcast_to(hough_obj.xH_i, hit_yH.shape)
    lin_idx = idx_y[valid] * hough_obj.n_xH + col_idx[valid]
    acc = np.bincount(lin_idx, minlength=hough_obj.n_yH * hough_obj.n_xH).reshape((hough_obj.n_yH, hough_obj.n_xH)).astype(np.float32)
    if hough_obj.fiducial_mask is not None:
        acc[~hough_obj.fiducial_mask] = 0.0
    if hough_obj.smooth_full:
        acc = scipy.ndimage.gaussian_filter(acc, hough_obj.sigma, truncate=hough_obj.truncate)
        if hough_obj.fiducial_mask is not None:
            acc[~hough_obj.fiducial_mask] = 0.0
    return acc


def make_visualization(
    event_idx=0,
    cache_path="/eos/user/i/idioniso/snd-numu-charm/plots/hough_tuning/pgun/tuning_cache_genie_341ev.pkl",
    out_path="/afs/cern.ch/work/i/idioniso/snd-numu-charm/plots/hough_tuning/pairwise_8d_hough_event.png",
):
    with open(cache_path, "rb") as f:
        events = pickle.load(f)

    event_data = events[event_idx]
    pos = event_data["pos"]
    d = event_data["d"]
    vert = event_data["vert"]
    system = event_data["system"]
    det_id = event_data["det_id"]

    # Parameter setup
    tol = 0.5
    max_angle = 0.200
    max_opening_angle = 0.200
    min_opening_angle = 0.002
    min_unique_planes = 1
    delta_zCross_max = 100.0
    zCross_min = 200.0
    zCross_max = 475.0
    delta_zDS3_max = 30.0
    z_ds3 = 543.0

    z_offset = 430.0
    det_zlen = 80.0
    z_plane = 352.0

    h_zy = StandaloneHough(
        n_yH=244,
        yH_range=(0.0, 122.0),
        n_xH=440,
        xH_range=(-1.1, 1.1),
        z_offset=z_offset,
        h_format="linearSlopeIntercept",
        det_zlen=det_zlen,
        smooth_full=True,
        sigma=1.0,
        truncate=3.0,
        n_quantile=0.5,
        res=5.0,
        target_window=(5.0, 60.0),
        max_angle=max_angle,
        z_target=z_plane,
    )

    h_zx = StandaloneHough(
        n_yH=244,
        yH_range=(-105.0, 17.0),
        n_xH=440,
        xH_range=(-1.1, 1.1),
        z_offset=z_offset,
        h_format="linearSlopeIntercept",
        det_zlen=det_zlen,
        smooth_full=True,
        sigma=1.0,
        truncate=3.0,
        n_quantile=0.5,
        res=5.0,
        target_window=(-50.0, -3.0),
        max_angle=max_angle,
        z_target=z_plane,
    )

    triplet_h = (~vert) & (system == 3)
    triplet_v = vert & (system == 3)

    zy_pts = np.column_stack([pos[2, triplet_h], pos[1, triplet_h]])
    zx_pts = np.column_stack([pos[2, triplet_v], pos[0, triplet_v]])

    acc_x = compute_acc(h_zx, zx_pts)
    acc_y = compute_acc(h_zy, zy_pts)

    peaks_x = h_zx.find_peaks(zx_pts, footprint_size=5, min_votes=0.5, max_peaks=15)
    peaks_y = h_zy.find_peaks(zy_pts, footprint_size=5, min_votes=0.5, max_peaks=15)

    # Form 2D valid candidates in X
    z_v = pos[2, vert]
    x_v = pos[0, vert]
    dz_v = d[2, vert]
    dx_v = d[0, vert]
    half_dx_v = dx_v * 0.5
    z1_v = z_v - dz_v * 0.5
    z2_v = z_v + dz_v * 0.5
    det_id_v = det_id[vert]
    sys_v = system[vert]
    ds_mask_v = (sys_v == 3)
    ds_planes_v = (det_id_v % 10000) // 1000

    cands_x = []
    for p in peaks_x:
        if abs(p["slope"]) > max_angle:
            continue
        slope, intercept = p["slope"], p["intercept"]
        dist = np.abs(x_v - (slope * z_v + intercept))
        line1 = slope * z1_v + intercept
        line2 = slope * z2_v + intercept
        clips = ((line1 - x_v) * (line2 - x_v) <= 0)
        hits = np.where((dist <= (half_dx_v + tol)) | clips)[0]
        planes = set(ds_planes_v[hits][ds_mask_v[hits]])
        if len(planes) < 3:
            continue
        p_dids = det_id_v[hits]
        cands_x.append({
            "peak": p,
            "hits": hits,
            "planes": planes,
            "det_ids": set(p_dids),
            "ds_det_ids": set(p_dids[ds_mask_v[hits]]),
        })

    # Form 2D valid candidates in Y
    horiz = ~vert
    z_h = pos[2, horiz]
    y_h = pos[1, horiz]
    dz_h = d[2, horiz]
    dy_h = d[1, horiz]
    half_dy_h = dy_h * 0.5
    z1_h = z_h - dz_h * 0.5
    z2_h = z_h + dz_h * 0.5
    det_id_h = det_id[horiz]
    sys_h = system[horiz]
    ds_mask_h = (sys_h == 3)
    ds_planes_h = (det_id_h % 10000) // 1000

    cands_y = []
    for p in peaks_y:
        if abs(p["slope"]) > max_angle:
            continue
        slope, intercept = p["slope"], p["intercept"]
        dist = np.abs(y_h - (slope * z_h + intercept))
        line1 = slope * z1_h + intercept
        line2 = slope * z2_h + intercept
        clips = ((line1 - y_h) * (line2 - y_h) <= 0)
        hits = np.where((dist <= (half_dy_h + tol)) | clips)[0]
        planes = set(ds_planes_h[hits][ds_mask_h[hits]])
        if len(planes) < 3:
            continue
        p_dids = det_id_h[hits]
        cands_y.append({
            "peak": p,
            "hits": hits,
            "planes": planes,
            "det_ids": set(p_dids),
            "ds_det_ids": set(p_dids[ds_mask_h[hits]]),
        })

    # Precompute valid 2D pairs in X and Y
    pairs_x = []
    for ix in range(len(cands_x)):
        c_x1 = cands_x[ix]
        p_x1 = c_x1["peak"]
        for jx in range(ix + 1, len(cands_x)):
            c_x2 = cands_x[jx]
            p_x2 = c_x2["peak"]
            d_angle_x = abs(p_x1["slope"] - p_x2["slope"])
            if not (min_opening_angle <= d_angle_x <= max_opening_angle):
                continue
            shared_x = c_x1["det_ids"] & c_x2["det_ids"]
            unq1_x = c_x1["ds_det_ids"] - shared_x
            unq2_x = c_x2["ds_det_ids"] - shared_x
            if len({(did % 10000) // 1000 for did in unq1_x}) < min_unique_planes or \
               len({(did % 10000) // 1000 for did in unq2_x}) < min_unique_planes:
                continue
            denom_x = p_x1["slope"] - p_x2["slope"]
            z_cross_xz = -(p_x1["intercept"] - p_x2["intercept"]) / denom_x if abs(denom_x) > 1e-6 else -999.0
            if not (zCross_min <= z_cross_xz <= zCross_max):
                continue
            x1_ds3 = p_x1["slope"] * z_ds3 + p_x1["intercept"]
            x2_ds3 = p_x2["slope"] * z_ds3 + p_x2["intercept"]
            dx_ds3 = abs(x1_ds3 - x2_ds3)
            if dx_ds3 > delta_zDS3_max:
                continue
            score_x = p_x1["votes"] + p_x2["votes"] - 0.5 * len(shared_x)
            pairs_x.append({
                "ix": ix, "jx": jx, "c_x1": c_x1, "c_x2": c_x2,
                "z_cross": z_cross_xz, "d_angle": d_angle_x,
                "dx_ds3": dx_ds3, "x1_ds3": x1_ds3, "x2_ds3": x2_ds3,
                "score_x": score_x, "shared": shared_x,
            })

    pairs_y = []
    for iy in range(len(cands_y)):
        c_y1 = cands_y[iy]
        p_y1 = c_y1["peak"]
        for jy in range(iy + 1, len(cands_y)):
            c_y2 = cands_y[jy]
            p_y2 = c_y2["peak"]
            d_angle_y = abs(p_y1["slope"] - p_y2["slope"])
            if not (min_opening_angle <= d_angle_y <= max_opening_angle):
                continue
            shared_y = c_y1["det_ids"] & c_y2["det_ids"]
            unq1_y = c_y1["ds_det_ids"] - shared_y
            unq2_y = c_y2["ds_det_ids"] - shared_y
            if len({(did % 10000) // 1000 for did in unq1_y}) < min_unique_planes or \
               len({(did % 10000) // 1000 for did in unq2_y}) < min_unique_planes:
                continue
            denom_y = p_y1["slope"] - p_y2["slope"]
            z_cross_yz = -(p_y1["intercept"] - p_y2["intercept"]) / denom_y if abs(denom_y) > 1e-6 else -999.0
            if not (zCross_min <= z_cross_yz <= zCross_max):
                continue
            y1_ds3 = p_y1["slope"] * z_ds3 + p_y1["intercept"]
            y2_ds3 = p_y2["slope"] * z_ds3 + p_y2["intercept"]
            dy_ds3 = abs(y1_ds3 - y2_ds3)
            if dy_ds3 > delta_zDS3_max:
                continue
            score_y = p_y1["votes"] + p_y2["votes"] - 0.5 * len(shared_y)
            pairs_y.append({
                "iy": iy, "jy": jy, "c_y1": c_y1, "c_y2": c_y2,
                "z_cross": z_cross_yz, "d_angle": d_angle_y,
                "dy_ds3": dy_ds3, "y1_ds3": y1_ds3, "y2_ds3": y2_ds3,
                "score_y": score_y, "shared": shared_y,
            })

    # Full 8D quadruplet evaluation and compatibility matrix
    candidates_8d = []
    matrix_delta_z = np.full((len(pairs_y), len(pairs_x)), np.nan)
    matrix_score_8d = np.full((len(pairs_y), len(pairs_x)), np.nan)
    pair_points = []

    for px_idx, px_item in enumerate(pairs_x):
        for py_idx, py_item in enumerate(pairs_y):
            dz = abs(px_item["z_cross"] - py_item["z_cross"])
            matrix_delta_z[py_idx, px_idx] = dz
            pass_dz = (dz <= delta_zCross_max)

            # Pairing permutation & score
            c_x1, c_x2 = px_item["c_x1"], px_item["c_x2"]
            c_y1, c_y2 = py_item["c_y1"], py_item["c_y2"]
            p_x1, p_x2 = c_x1["peak"], c_x2["peak"]
            p_y1, p_y2 = c_y1["peak"], c_y2["peak"]

            overlap_p0 = len(c_x1["planes"] & c_y1["planes"]) + len(c_x2["planes"] & c_y2["planes"])
            overlap_p1 = len(c_x1["planes"] & c_y2["planes"]) + len(c_x2["planes"] & c_y1["planes"])
            vdiff_p0 = abs(p_x1["votes"] - p_y1["votes"]) + abs(p_x2["votes"] - p_y2["votes"])
            vdiff_p1 = abs(p_x1["votes"] - p_y2["votes"]) + abs(p_x2["votes"] - p_y1["votes"])
            perm = 0 if (overlap_p0 > overlap_p1 or (overlap_p0 == overlap_p1 and vdiff_p0 <= vdiff_p1)) else 1
            match_bonus = (overlap_p0 - 0.1 * vdiff_p0) if perm == 0 else (overlap_p1 - 0.1 * vdiff_p1)

            score_8d = px_item["score_x"] + py_item["score_y"] + match_bonus
            matrix_score_8d[py_idx, px_idx] = score_8d

            item_8d = {
                "score_8d": score_8d,
                "dz": dz,
                "zx": px_item["z_cross"],
                "zy": py_item["z_cross"],
                "dx_ds3": px_item["dx_ds3"],
                "dy_ds3": py_item["dy_ds3"],
                "pass": pass_dz,
                "px_idx": px_idx,
                "py_idx": py_idx,
                "c_x1": c_x1, "c_x2": c_x2,
                "c_y1": c_y1, "c_y2": c_y2,
                "perm": perm,
            }
            pair_points.append(item_8d)
            if pass_dz:
                candidates_8d.append(item_8d)

    candidates_8d.sort(key=lambda c: -c["score_8d"])
    best_candidate = candidates_8d[0] if candidates_8d else (pair_points[0] if pair_points else None)

    # -------------------------------------------------------------------------
    # Layout 7 Panels across a structured grid (3 rows x 3 columns)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(22, 17))
    gs = fig.add_gridspec(3, 3, hspace=0.32, wspace=0.28)

    # Panel 1: Accumulator XZ
    ax1 = fig.add_subplot(gs[0, 0])
    im_x = ax1.imshow(
        acc_x, origin="lower",
        extent=[h_zx.xH_range[0], h_zx.xH_range[1], h_zx.yH_range[0], h_zx.yH_range[1]],
        aspect="auto", cmap="viridis"
    )
    plt.colorbar(im_x, ax=ax1, label="Hough Votes", fraction=0.046, pad=0.04)
    for idx, p in enumerate(peaks_x):
        yH_peak = p["intercept"] + p["slope"] * z_offset
        ax1.plot(p["slope"], yH_peak, "r*", markersize=10)
        ax1.text(p["slope"] + 0.008, yH_peak + 1.2, f"X{idx+1}", color="yellow", fontsize=9, fontweight="bold")
    ax1.set_title("1. Hough Accumulator: XZ Projection", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Slope $t_x$ (rad)")
    ax1.set_ylabel(f"Intercept at $z={z_offset}$ cm (cm)")
    ax1.set_xlim(-0.25, 0.25)
    ax1.grid(alpha=0.25)

    # Panel 2: Accumulator YZ
    ax2 = fig.add_subplot(gs[0, 1])
    im_y = ax2.imshow(
        acc_y, origin="lower",
        extent=[h_zy.xH_range[0], h_zy.xH_range[1], h_zy.yH_range[0], h_zy.yH_range[1]],
        aspect="auto", cmap="viridis"
    )
    plt.colorbar(im_y, ax=ax2, label="Hough Votes", fraction=0.046, pad=0.04)
    for idx, p in enumerate(peaks_y):
        yH_peak = p["intercept"] + p["slope"] * z_offset
        ax2.plot(p["slope"], yH_peak, "m*", markersize=10)
        ax2.text(p["slope"] + 0.008, yH_peak + 1.2, f"Y{idx+1}", color="cyan", fontsize=9, fontweight="bold")
    ax2.set_title("2. Hough Accumulator: YZ Projection", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Slope $t_y$ (rad)")
    ax2.set_ylabel(f"Intercept at $z={z_offset}$ cm (cm)")
    ax2.set_xlim(-0.25, 0.25)
    ax2.grid(alpha=0.25)

    # Panel 3: (z_cross_XZ vs z_cross_YZ) Correlation Plane
    ax3 = fig.add_subplot(gs[0, 2])
    z_min_plot, z_max_plot = 150.0, 550.0
    z_line = np.linspace(z_min_plot, z_max_plot, 200)
    diag_pts = np.array([
        [z_min_plot, z_min_plot - delta_zCross_max],
        [z_max_plot, z_max_plot - delta_zCross_max],
        [z_max_plot, z_max_plot + delta_zCross_max],
        [z_min_plot, z_min_plot + delta_zCross_max],
    ])
    poly_diag = Polygon(diag_pts, closed=True, facecolor="lightblue", alpha=0.35, edgecolor="blue", linestyle="--", label=rf"$|\Delta z_{{cross}}| \leq {delta_zCross_max:.0f}$ cm")
    ax3.add_patch(poly_diag)
    box = Rectangle(
        (zCross_min, zCross_min), zCross_max - zCross_min, zCross_max - zCross_min,
        facecolor="none", edgecolor="red", linewidth=2, linestyle="-", label=f"Target window [{zCross_min:.0f}, {zCross_max:.0f}] cm"
    )
    ax3.add_patch(box)
    if pair_points:
        zx_pass = [pt["zx"] for pt in pair_points if pt["pass"]]
        zy_pass = [pt["zy"] for pt in pair_points if pt["pass"]]
        zx_fail = [pt["zx"] for pt in pair_points if not pt["pass"] and (z_min_plot <= pt["zx"] <= z_max_plot) and (z_min_plot <= pt["zy"] <= z_max_plot)]
        zy_fail = [pt["zy"] for pt in pair_points if not pt["pass"] and (z_min_plot <= pt["zx"] <= z_max_plot) and (z_min_plot <= pt["zy"] <= z_max_plot)]
        if zx_fail:
            ax3.scatter(zx_fail, zy_fail, color="gray", alpha=0.5, s=35, marker="x", label=f"Rejected ({len(zx_fail)})")
        if zx_pass:
            ax3.scatter(zx_pass, zy_pass, color="green", edgecolor="black", s=75, marker="o", label=f"Accepted ({len(zx_pass)})")
        if best_candidate and best_candidate["pass"]:
            ax3.scatter(best_candidate["zx"], best_candidate["zy"], color="gold", edgecolor="black", s=180, marker="*", label="Selected Best Pair", zorder=5)

    ax3.plot(z_line, z_line, "k:", alpha=0.5, label="$z_{XZ} = z_{YZ}$")
    ax3.set_title(r"3. Crossing Correlation: $z_{cross}^{XZ}$ vs $z_{cross}^{YZ}$", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Reconstructed $z_{cross}^{XZ}$ (cm)")
    ax3.set_ylabel("Reconstructed $z_{cross}^{YZ}$ (cm)")
    ax3.set_xlim(z_min_plot, z_max_plot)
    ax3.set_ylim(z_min_plot, z_max_plot)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper left", fontsize=8)

    # Panel 4: Reconstructed Trajectories in Space
    ax4 = fig.add_subplot(gs[1, 0])
    z_proj = np.linspace(200.0, 560.0, 200)
    ax4.scatter(pos[2, triplet_v], pos[0, triplet_v], color="blue", alpha=0.4, s=15, label="DS Vertical Hits (X)")
    ax4.scatter(pos[2, triplet_h], pos[1, triplet_h], color="red", alpha=0.4, s=15, label="DS Horizontal Hits (Y)")
    if best_candidate:
        bx1, bx2 = best_candidate["c_x1"]["peak"], best_candidate["c_x2"]["peak"]
        by1, by2 = best_candidate["c_y1"]["peak"], best_candidate["c_y2"]["peak"]
        ax4.plot(z_proj, bx1["slope"] * z_proj + bx1["intercept"], "b-", linewidth=2, label=r"Muon 1 ($X-Z$)")
        ax4.plot(z_proj, bx2["slope"] * z_proj + bx2["intercept"], "b--", linewidth=2, label=r"Muon 2 ($X-Z$)")
        ax4.plot(z_proj, by1["slope"] * z_proj + by1["intercept"], "r-", linewidth=2, label=r"Muon 1 ($Y-Z$)")
        ax4.plot(z_proj, by2["slope"] * z_proj + by2["intercept"], "r--", linewidth=2, label=r"Muon 2 ($Y-Z$)")
        ax4.axvline(best_candidate["zx"], color="blue", linestyle=":", alpha=0.7, label=f"$z_{{XZ}} = {best_candidate['zx']:.1f}$ cm")
        ax4.axvline(best_candidate["zy"], color="red", linestyle=":", alpha=0.7, label=f"$z_{{YZ}} = {best_candidate['zy']:.1f}$ cm")

    ax4.set_title("4. Reconstructed Tracks & Common Dimuon Vertex", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Z (cm)")
    ax4.set_ylabel("Transverse Position X/Y (cm)")
    ax4.set_xlim(200.0, 560.0)
    ax4.set_ylim(-70.0, 70.0)
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="upper left", fontsize=8)

    # -------------------------------------------------------------------------
    # Idea 3: Pairwise Geometric Compatibility Matrix (Panel 5)
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    if matrix_delta_z.size > 0 and not np.all(np.isnan(matrix_delta_z)):
        im5 = ax5.imshow(matrix_delta_z, origin="lower", aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=200.0)
        cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        cbar5.set_label(r"$|\Delta z_{cross}|$ (cm)", fontsize=10)
        ax5.set_xticks(range(len(pairs_x)))
        ax5.set_xticklabels([f"X{p['ix']+1}-X{p['jx']+1}" for p in pairs_x], rotation=45, ha="right", fontsize=8)
        ax5.set_yticks(range(len(pairs_y)))
        ax5.set_yticklabels([f"Y{p['iy']+1}-Y{p['jy']+1}" for p in pairs_y], fontsize=8)
        if best_candidate and best_candidate["pass"]:
            ax5.plot(best_candidate["px_idx"], best_candidate["py_idx"], marker="*", markersize=16, color="gold", markeredgecolor="black", zorder=5)
        ax5.set_title(r"5. Compatibility Matrix: $|\Delta z_{cross}(X_i X_j, Y_k Y_l)|$", fontsize=11, fontweight="bold")
        ax5.set_xlabel(r"XZ Pair Candidate ($X_i - X_j$)")
        ax5.set_ylabel(r"YZ Pair Candidate ($Y_k - Y_l$)")
    else:
        ax5.text(0.5, 0.5, "No multi-track pairs evaluated", ha="center", va="center")
        ax5.set_title("5. Compatibility Matrix", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # Idea 4: DS3 Transverse Isolation Plane (Panel 6)
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    # Cut boundary: Delta x_DS3 <= delta_zDS3_max and Delta y_DS3 <= delta_zDS3_max
    rect_ds3 = Rectangle(
        (0.0, 0.0), delta_zDS3_max, delta_zDS3_max,
        facecolor="lightgreen", alpha=0.35, edgecolor="green", linewidth=2, linestyle="--",
        label=rf"Allowed DS3 Isolation ($\Delta \leq {delta_zDS3_max:.0f}$ cm)"
    )
    ax6.add_patch(rect_ds3)

    if pair_points:
        dx_vals = [pt["dx_ds3"] for pt in pair_points]
        dy_vals = [pt["dy_ds3"] for pt in pair_points]
        ax6.scatter(dx_vals, dy_vals, color="gray", alpha=0.5, s=35, marker="o", label=f"Evaluated Pairs ({len(pair_points)})")
        if best_candidate:
            ax6.scatter(best_candidate["dx_ds3"], best_candidate["dy_ds3"], color="gold", edgecolor="black", s=180, marker="*", label="Selected Best Pair", zorder=5)

    ax6.set_title(r"6. DS3 Isolation Plane ($z_{DS3} = 543$ cm)", fontsize=11, fontweight="bold")
    ax6.set_xlabel(r"$\Delta x_{DS3} = |x_1 - x_2|_{DS3}$ (cm)")
    ax6.set_ylabel(r"$\Delta y_{DS3} = |y_1 - y_2|_{DS3}$ (cm)")
    ax6.set_xlim(0.0, 60.0)
    ax6.set_ylim(0.0, 60.0)
    ax6.grid(True, linestyle="--", alpha=0.4)
    ax6.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------------------
    # Idea 5: Longitudinal Convergence Profile vs Z (Panel 7)
    # -------------------------------------------------------------------------
    ax7 = fig.add_subplot(gs[2, :])
    z_profile = np.linspace(150.0, 600.0, 400)
    if best_candidate:
        bx1, bx2 = best_candidate["c_x1"]["peak"], best_candidate["c_x2"]["peak"]
        by1, by2 = best_candidate["c_y1"]["peak"], best_candidate["c_y2"]["peak"]
        sep_x = np.abs((bx1["slope"] - bx2["slope"]) * z_profile + (bx1["intercept"] - bx2["intercept"]))
        sep_y = np.abs((by1["slope"] - by2["slope"]) * z_profile + (by1["intercept"] - by2["intercept"]))
        sep_3d = np.sqrt(sep_x**2 + sep_y**2)

        ax7.plot(z_profile, sep_x, "b-", linewidth=2.2, label=r"Transverse Separation $\Delta x(z) = |x_1(z) - x_2(z)|$")
        ax7.plot(z_profile, sep_y, "r-", linewidth=2.2, label=r"Transverse Separation $\Delta y(z) = |y_1(z) - y_2(z)|$")
        ax7.plot(z_profile, sep_3d, "k-.", linewidth=2.0, alpha=0.8, label=r"3D Transverse Separation $\Delta r_{3D}(z) = \sqrt{\Delta x^2 + \Delta y^2}$")

        ax7.axvline(best_candidate["zx"], color="blue", linestyle=":", linewidth=2, label=rf"Crossing $z_{{XZ}} = {best_candidate['zx']:.1f}$ cm (Minima $\Delta x = 0$)")
        ax7.axvline(best_candidate["zy"], color="red", linestyle=":", linewidth=2, label=rf"Crossing $z_{{YZ}} = {best_candidate['zy']:.1f}$ cm (Minima $\Delta y = 0$)")

        # Shaded target crossing interval
        ax7.axvspan(zCross_min, zCross_max, color="gray", alpha=0.15, label=f"Allowed Target Crossing Window [{zCross_min:.0f}, {zCross_max:.0f}] cm")
        # Line at DS3 position
        ax7.axvline(z_ds3, color="green", linestyle="--", alpha=0.7, label=f"DS3 Plane ($z={z_ds3:.0f}$ cm)")
        # Horizontal isolation limit at DS3
        ax7.axhline(delta_zDS3_max, color="green", linestyle=":", alpha=0.6, label=rf"DS3 Max Separation Limit ($\Delta \leq {delta_zDS3_max:.0f}$ cm)")

        # Highlight longitudinal discrepancy
        min_z = min(best_candidate["zx"], best_candidate["zy"])
        max_z = max(best_candidate["zx"], best_candidate["zy"])
        ax7.axvspan(min_z, max_z, color="gold", alpha=0.3, label=rf"Longitudinal Vertex Discrepancy $|\Delta z_{{cross}}| = {best_candidate['dz']:.1f}$ cm")

    ax7.set_title(r"7. Longitudinal Track Convergence Profile ($\Delta x$, $\Delta y$, $\Delta r_{3D}$ vs $z$)", fontsize=12, fontweight="bold")
    ax7.set_xlabel("Longitudinal Coordinate $z$ along beam axis (cm)", fontsize=11)
    ax7.set_ylabel("Track Separation (cm)", fontsize=11)
    ax7.set_xlim(150.0, 600.0)
    ax7.set_ylim(0.0, 80.0)
    ax7.grid(True, linestyle="--", alpha=0.4)
    ax7.legend(loc="upper right", ncol=3, fontsize=8.5)

    dms.add_label(mainText="SND@LHC", extraText="Pairwise 8D Hough", suppText=f"Event {event_idx}", ax=ax1)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    pdf_path = out_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Generated comprehensive 8D pairwise Hough constraint visualization: {out_path} and {pdf_path}")


if __name__ == "__main__":
    make_visualization()
