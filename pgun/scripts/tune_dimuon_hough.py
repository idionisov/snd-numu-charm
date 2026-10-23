#!/usr/bin/env python3
"""
tune_dimuon_hough.py
--------------------
Hough parameter tuning and grid-search optimizer for SND@LHC Downstream Dimuon Tracking.

Optimizes tracking configuration based on the user-defined sensitivity figure of merit:
    Score = (1 - FPR) * (TPR)^2

Evaluates 4 cross-dataset benchmark combinations:
  1. Genie Signal (TPR) vs. Genie Background (FPR)
  2. Genie Signal (TPR) vs. pmu2022 Background (FPR)
  3. Particle Gun (TPR) vs. Genie Background (FPR)
  4. Particle Gun (TPR) vs. pmu2022 Background (FPR)

Features:
- Enforces strict in-space fiducial mask: |theta| <= 175 mrad and target window at z=352 cm.
- Grid search over Hough space binning factors:
    - linearSlopeIntercept: [0.75, 1.00, 1.30, 1.60, 1.80, 2.00, 2.25, 2.50, 2.60, 2.70, 2.75, 2.80, 2.90, 3.00, 3.50, 4.00]
    - normal:               [0.50, 0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.25, 1.50]
- Grid search over hit tolerances: [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0] cm.
- Grid search over Gaussian smoothing: (sigma, truncate) in [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)], plus no-smoothing.
- Multiprocessed parallel evaluation across available CPU cores.
"""

from __future__ import annotations

import os
import sys

# Prevent OpenMP / BLAS thread contention when running parallel workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import pickle
import argparse
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import scipy.ndimage


# ---------------------------------------------------------------------------
# 1. Standalone Hough Implementation (Matches SndlhcMuonReco.py)
# ---------------------------------------------------------------------------

def hit_finder(slope: float, intercept: float, box_centers: np.ndarray, box_ds: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Find hits intersecting or within tolerance of a 2D line (x = slope*z + intercept)."""
    if len(box_centers) == 0:
        return np.array([], dtype=int)
    z = box_centers[:, :, 0]
    coord = box_centers[:, :, 1]
    d_z = box_ds[:, :, 0]
    d_coord = box_ds[:, :, 1]

    line_coord = slope * z + intercept
    center_in_box = np.abs(coord - line_coord) <= (d_coord / 2.0 + tol)

    z1 = z - d_z / 2.0
    z2 = z + d_z / 2.0
    line1 = slope * z1 + intercept
    line2 = slope * z2 + intercept
    clips_corner = ((line1 - coord) * (line2 - coord) <= 0)

    hit_mask = np.logical_or(center_in_box, clips_corner).flatten()
    return np.where(hit_mask)[0]


def num_planes_hit(systems: np.ndarray, detector_ids: np.ndarray) -> int:
    """Count unique fired planes in DS."""
    if len(systems) == 0:
        return 0
    planes = set()
    for sys_id, det_id in zip(systems, detector_ids):
        if sys_id == 3:  # Downstream
            plane = (det_id % 10000) // 1000
            planes.add(plane)
    return len(planes)


class StandaloneHough:
    def __init__(
        self,
        n_yH: int,
        yH_range: Tuple[float, float],
        n_xH: int,
        xH_range: Tuple[float, float],
        z_offset: float,
        h_format: str,
        det_zlen: float,
        smooth_full: bool = True,
        sigma: float = 1.0,
        truncate: float = 2.0,
        n_quantile: float = 0.5,
        res: float = 5.0,
        target_window: Optional[Tuple[float, float]] = None,
        max_angle: Optional[float] = None,
        z_target: float = 352.0,
    ):
        self.n_yH = n_yH
        self.n_xH = n_xH
        self.yH_range = yH_range
        self.xH_range = xH_range
        self.z_offset = z_offset
        self.h_format = h_format
        self.det_zlen = det_zlen
        self.smooth_full = smooth_full
        self.sigma = sigma
        self.truncate = truncate
        self.n_quantile = n_quantile
        self.res = res
        self.target_window = target_window
        self.max_angle = max_angle
        self.z_target = z_target

        self.yH_bins = np.linspace(yH_range[0], yH_range[1], n_yH)
        self.xH_bins = np.linspace(xH_range[0], xH_range[1], n_xH)
        self.cos_thetas = np.cos(self.xH_bins)
        self.sin_thetas = np.sin(self.xH_bins)
        self.xH_i = np.arange(n_xH)

        # Precompute 2D boolean mask for dedicated fiducial search space
        self.fiducial_mask = None
        if target_window is not None and max_angle is not None:
            dz_target = z_target - z_offset
            if h_format == 'linearSlopeIntercept':
                M, B = np.meshgrid(self.xH_bins, self.yH_bins)
                pos_at_tgt = B + M * dz_target
                angle_ok = np.abs(M) <= max_angle
                tgt_ok = (pos_at_tgt >= target_window[0]) & (pos_at_tgt <= target_window[1])
                self.fiducial_mask = angle_ok & tgt_ok
            elif h_format == 'normal':
                Thetas, R = np.meshgrid(self.xH_bins, self.yH_bins)
                tan_theta = np.tan(Thetas)
                valid_tan = np.abs(tan_theta) > 1e-5
                slope = np.zeros_like(tan_theta)
                slope[valid_tan] = -1.0 / tan_theta[valid_tan]
                sin_theta = np.sin(Thetas)
                valid_sin = np.abs(sin_theta) > 1e-5
                b_ds = np.zeros_like(R)
                b_ds[valid_sin] = R[valid_sin] / sin_theta[valid_sin]
                pos_at_tgt = b_ds + slope * dz_target
                angle_ok = (np.abs(slope) <= max_angle) & valid_tan
                tgt_ok = (pos_at_tgt >= target_window[0]) & (pos_at_tgt <= target_window[1])
                self.fiducial_mask = angle_ok & tgt_ok & valid_sin

    def fit(self, hit_collection: np.ndarray, hit_d: np.ndarray, n_random: int = 0) -> Tuple[float, float]:
        if len(hit_collection) == 0:
            return (-999.0, -999.0)

        if n_random > 0:
            rnd = (np.random.uniform(size=(n_random, len(hit_collection), 2)) - 0.5) * hit_d[None, :, :]
            pts_to_use = (hit_collection[None, :, :] + rnd).reshape(-1, 2)
        else:
            pts_to_use = hit_collection

        shifted_z = pts_to_use[:, 0] - self.z_offset
        y_vals = pts_to_use[:, 1]

        accumulator = np.zeros((self.n_yH, self.n_xH), dtype=np.float32)

        if self.h_format == 'normal':
            hit_yH = shifted_z[:, None] * self.cos_thetas[None, :] + y_vals[:, None] * self.sin_thetas[None, :]
            valid = (hit_yH >= self.yH_range[0]) & (hit_yH <= self.yH_range[1])
            if not np.any(valid):
                return (-999.0, -999.0)
            idx_y = np.floor((hit_yH - self.yH_range[0]) / (self.yH_range[1] - self.yH_range[0]) * self.n_yH).astype(int)
            idx_y = np.clip(idx_y, 0, self.n_yH - 1)
            col_idx = np.broadcast_to(self.xH_i, hit_yH.shape)
            np.add.at(accumulator, (idx_y[valid], col_idx[valid]), 1.0)
        elif self.h_format == 'linearSlopeIntercept':
            # Vectorized accumulator calculation
            hit_yH = y_vals[:, None] - shifted_z[:, None] * self.xH_bins[None, :]
            valid = (hit_yH >= self.yH_range[0]) & (hit_yH <= self.yH_range[1])
            if not np.any(valid):
                return (-999.0, -999.0)
            idx_y = np.floor((hit_yH - self.yH_range[0]) / (self.yH_range[1] - self.yH_range[0]) * self.n_yH).astype(int)
            idx_y = np.clip(idx_y, 0, self.n_yH - 1)
            col_idx = np.broadcast_to(self.xH_i, hit_yH.shape)
            np.add.at(accumulator, (idx_y[valid], col_idx[valid]), 1.0)
        else:
            raise ValueError(f"Unknown Hough format: {self.h_format}")

        # Restrict accumulator search space strictly to fiducial constraints
        if self.fiducial_mask is not None:
            accumulator[~self.fiducial_mask] = 0.0

        if self.smooth_full:
            accumulator = scipy.ndimage.gaussian_filter(accumulator, self.sigma, truncate=self.truncate)
            if self.fiducial_mask is not None:
                accumulator[~self.fiducial_mask] = 0.0

        max_val = np.amax(accumulator)
        if max_val == 0:
            return (-999.0, -999.0)

        if self.smooth_full:
            i_max = np.unravel_index(accumulator.argmax(), accumulator.shape)
        else:
            maxima = np.argwhere(accumulator == max_val)
            if len(maxima) == 1:
                i_max = maxima[0]
            elif len(maxima) > 1 and abs(min(m[1] for m in maxima) - max(m[1] for m in maxima)) < self.res:
                i_max = maxima[0]
            else:
                slopes = np.array([m[1] for m in maxima])
                q_val = np.quantile(slopes, self.n_quantile)
                within = sum(1 for m in maxima if abs(m[1] - q_val) < self.res)
                if within / len(maxima) > self.n_quantile:
                    i_max = min(maxima, key=lambda m: abs(m[1] - q_val))
                else:
                    return (-999.0, -999.0)

        found_yH = self.yH_bins[int(i_max[0])]
        found_xH = self.xH_bins[int(i_max[1])]

        if self.h_format == 'normal':
            tan_x = np.tan(found_xH)
            if abs(tan_x) < 1e-5:
                return (-999.0, -999.0)
            slope = -1.0 / tan_x
            intercept_shift = found_yH / np.sin(found_xH)
            intercept = (tan_x * intercept_shift + self.z_offset) / tan_x
        elif self.h_format == 'linearSlopeIntercept':
            slope = found_xH
            intercept = found_yH - slope * self.z_offset

        return (float(slope), float(intercept))


def run_reco_for_event(
    event_data: Dict[str, Any],
    params: Dict[str, Any],
    h_zy: Optional[StandaloneHough] = None,
    h_zx: Optional[StandaloneHough] = None,
) -> List[Dict[str, float]]:
    """Reconstruct tracks for an event strictly within fiducial SciFi/Target acceptance."""
    max_reco_muons = 2
    min_planes_hit = 3
    max_angle = params.get("max_angle", 0.175)  # 175 mrad (SciFi angular acceptance)
    tol = params["tolerance"]
    z_target = params.get("z_target", 352.0)
    tgt_xz = params.get("target_window_xz", (-50.0, -3.0))
    tgt_yz = params.get("target_window_yz", (5.0, 60.0))

    pos = np.copy(event_data["pos"])
    d = np.copy(event_data["d"])
    vert = np.copy(event_data["vert"])
    system = np.copy(event_data["system"])
    det_id = np.copy(event_data["det_id"])

    z_offset = event_data["z_offset"]
    det_zlen = event_data["det_zlen"]

    if h_zy is None:
        h_zy = StandaloneHough(
            n_yH=params["n_yH"],
            yH_range=params["yH_range_yz"],
            n_xH=params["n_xH"],
            xH_range=params["xH_range_yz"],
            z_offset=z_offset,
            h_format=params["h_format"],
            det_zlen=det_zlen,
            smooth_full=params["smooth_full"],
            sigma=params["sigma"],
            truncate=params["truncate"],
            n_quantile=0.5,
            res=5.0,
            target_window=tgt_yz,
            max_angle=max_angle,
            z_target=z_target,
        )

    if h_zx is None:
        h_zx = StandaloneHough(
            n_yH=params["n_yH"],
            yH_range=params["yH_range_xz"],
            n_xH=params["n_xH"],
            xH_range=params["xH_range_xz"],
            z_offset=z_offset,
            h_format=params["h_format"],
            det_zlen=det_zlen,
            smooth_full=params["smooth_full"],
            sigma=params["sigma"],
            truncate=params["truncate"],
            n_quantile=0.5,
            res=5.0,
            target_window=tgt_xz,
            max_angle=max_angle,
            z_target=z_target,
        )

    reconstructed_tracks = []

    for _ in range(max_reco_muons):
        if len(system) == 0:
            break

        triplet_h = (~vert) & (system == 3)
        triplet_v = vert & (system == 3)

        n_pl_zy = num_planes_hit(system[triplet_h], det_id[triplet_h])
        n_pl_zx = num_planes_hit(system[triplet_v], det_id[triplet_v])
        if n_pl_zy < min_planes_hit or n_pl_zx < min_planes_hit:
            break

        zy_pts = np.column_stack([pos[2, triplet_h], pos[1, triplet_h]])
        d_zy = np.column_stack([d[2, triplet_h], d[1, triplet_h]])
        zx_pts = np.column_stack([pos[2, triplet_v], pos[0, triplet_v]])
        d_zx = np.column_stack([d[2, triplet_v], d[0, triplet_v]])

        slope_zy, b_zy = h_zy.fit(zy_pts, d_zy, n_random=params.get("n_random", 0))
        slope_zx, b_zx = h_zx.fit(zx_pts, d_zx, n_random=params.get("n_random", 0))

        if slope_zy == -999.0 or slope_zx == -999.0:
            break
        # Fixed 250 mrad angular acceptance cut
        if abs(slope_zy) > max_angle or abs(slope_zx) > max_angle:
            break

        # Associate hits
        box_c_zy = np.column_stack([pos[2, ~vert], pos[1, ~vert]])[:, np.newaxis, :]
        box_d_zy = np.column_stack([d[2, ~vert], d[1, ~vert]])[:, np.newaxis, :]
        hits_zy = hit_finder(slope_zy, b_zy, box_c_zy, box_d_zy, tol=tol)

        box_c_zx = np.column_stack([pos[2, vert], pos[0, vert]])[:, np.newaxis, :]
        box_d_zx = np.column_stack([d[2, vert], d[0, vert]])[:, np.newaxis, :]
        hits_zx = hit_finder(slope_zx, b_zx, box_c_zx, box_d_zx, tol=tol)

        # Confirm planes
        h_planes = num_planes_hit(system[~vert][hits_zy], det_id[~vert][hits_zy])
        v_planes = num_planes_hit(system[vert][hits_zx], det_id[vert][hits_zx])
        if h_planes < min_planes_hit or v_planes < min_planes_hit:
            break

        reconstructed_tracks.append({
            "tx": slope_zx,
            "ty": slope_zy,
            "bx": b_zx,
            "by": b_zy,
            "n_hits_h": len(hits_zy),
            "n_hits_v": len(hits_zx),
        })

        used_det_ids = set(det_id[vert][hits_zx]).union(det_id[~vert][hits_zy])
        mask_keep = ~np.isin(det_id, list(used_det_ids))
        if not np.any(mask_keep):
            break

        pos = pos[:, mask_keep]
        d = d[:, mask_keep]
        vert = vert[mask_keep]
        system = system[mask_keep]
        det_id = det_id[mask_keep]

    return reconstructed_tracks


# ---------------------------------------------------------------------------
# 2. Evaluation Worker Function
# ---------------------------------------------------------------------------

_worker_samples = None


def init_worker(samples: Dict[str, List[Dict[str, Any]]]):
    global _worker_samples
    _worker_samples = samples


def evaluate_config_on_samples(item: Any) -> Dict[str, Any]:
    """Worker function to evaluate a single configuration on all 4 datasets."""
    global _worker_samples
    if isinstance(item, tuple):
        params, samples = item
    else:
        params = item
        samples = _worker_samples

    t0 = time.time()

    tgt_xz = params.get("target_window_xz", (-50.0, -3.0))
    tgt_yz = params.get("target_window_yz", (5.0, 60.0))
    max_angle = params.get("max_angle", 0.175)
    z_target = params.get("z_target", 352.0)
    z_offset = 430.0
    det_zlen = 80.0

    h_zy = StandaloneHough(
        n_yH=params["n_yH"],
        yH_range=params["yH_range_yz"],
        n_xH=params["n_xH"],
        xH_range=params["xH_range_yz"],
        z_offset=z_offset,
        h_format=params["h_format"],
        det_zlen=det_zlen,
        smooth_full=params["smooth_full"],
        sigma=params["sigma"],
        truncate=params["truncate"],
        n_quantile=0.5,
        res=5.0,
        target_window=tgt_yz,
        max_angle=max_angle,
        z_target=z_target,
    )

    h_zx = StandaloneHough(
        n_yH=params["n_yH"],
        yH_range=params["yH_range_xz"],
        n_xH=params["n_xH"],
        xH_range=params["xH_range_xz"],
        z_offset=z_offset,
        h_format=params["h_format"],
        det_zlen=det_zlen,
        smooth_full=params["smooth_full"],
        sigma=params["sigma"],
        truncate=params["truncate"],
        n_quantile=0.5,
        res=5.0,
        target_window=tgt_xz,
        max_angle=max_angle,
        z_target=z_target,
    )

    # 1. Evaluate Genie Signal (TPR)
    genie_sig = samples.get("genie_sig", [])
    n_2trk_gs = sum(1 for ev in genie_sig if len(run_reco_for_event(ev, params, h_zy, h_zx)) == 2)
    tpr_genie = (n_2trk_gs / len(genie_sig)) if len(genie_sig) > 0 else 0.0

    # 2. Evaluate Particle Gun Signal (TPR)
    pgun_sig = samples.get("pgun_sig", [])
    n_2trk_pg = sum(1 for ev in pgun_sig if len(run_reco_for_event(ev, params, h_zy, h_zx)) == 2)
    tpr_pgun = (n_2trk_pg / len(pgun_sig)) if len(pgun_sig) > 0 else 0.0

    # 3. Evaluate Genie Background (FPR)
    genie_bkg = samples.get("genie_bkg", [])
    n_2trk_gb = sum(1 for ev in genie_bkg if len(run_reco_for_event(ev, params, h_zy, h_zx)) == 2)
    fpr_genie = (n_2trk_gb / len(genie_bkg)) if len(genie_bkg) > 0 else 0.0

    # 4. Evaluate pmu2022 Background (FPR)
    pmu_bkg = samples.get("pmu_bkg", [])
    n_2trk_pb = sum(1 for ev in pmu_bkg if len(run_reco_for_event(ev, params, h_zy, h_zx)) == 2)
    fpr_pmu = (n_2trk_pb / len(pmu_bkg)) if len(pmu_bkg) > 0 else 0.0

    # Compute Figure of Merit: Score = (1 - FPR) * (TPR)^2
    score_genie_genie = (1.0 - fpr_genie) * (tpr_genie ** 2)
    score_genie_pmu   = (1.0 - fpr_pmu)   * (tpr_genie ** 2)
    score_pgun_genie  = (1.0 - fpr_genie) * (tpr_pgun ** 2)
    score_pgun_pmu    = (1.0 - fpr_pmu)   * (tpr_pgun ** 2)
    score_mean = 0.25 * (score_genie_genie + score_genie_pmu + score_pgun_genie + score_pgun_pmu)

    elapsed = time.time() - t0

    return {
        "params": params,
        "tpr_genie": tpr_genie,
        "tpr_pgun": tpr_pgun,
        "fpr_genie": fpr_genie,
        "fpr_pmu": fpr_pmu,
        "score_genie_genie": score_genie_genie,
        "score_genie_pmu": score_genie_pmu,
        "score_pgun_genie": score_pgun_genie,
        "score_pgun_pmu": score_pgun_pmu,
        "score_mean": score_mean,
        "time_sec": elapsed,
    }


evaluate_config = evaluate_config_on_samples


# ---------------------------------------------------------------------------
# 3. Parameter Grid Generation
# ---------------------------------------------------------------------------

def generate_parameter_grid(
    h_format: str = "linearSlopeIntercept",
    max_angles: Optional[List[float]] = None,
    sigmas: Optional[List[float]] = None,
    truncates: Optional[List[float]] = None,
    include_no_smooth: bool = True,
    z_plane: float = 352.0,
    target_window_xz: Tuple[float, float] = (-50.0, -3.0),
    target_window_yz: Tuple[float, float] = (5.0, 60.0),
) -> List[Dict[str, Any]]:
    """Build grid of parameter configurations with requested binning factors, max angles, and smoothing options."""
    tolerances = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
    if max_angles is None:
        max_angles = [0.175]
    z_target = z_plane

    # Requested Linear Slope-Intercept binning factors
    factors_linear = [0.75, 1.00, 1.30, 1.60, 1.80, 2.00, 2.25, 2.50, 2.60, 2.70, 2.75, 2.80, 2.90, 3.00, 3.50, 4.00]
    base_ny_linear, base_nx_linear = 122, 220

    # Requested Normal representation binning factors
    factors_normal = [0.50, 0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.25, 1.50]
    base_ny_normal, base_nx_normal = 1000, 2500

    if sigmas is None:
        sigmas = [0.5, 1.0, 1.5, 2.0, 3.0]
    if truncates is None:
        truncates = [1.0, 2.0, 3.0, 4.0]

    smooth_configs = []
    for sig in sigmas:
        for trunc in truncates:
            smooth_configs.append((True, float(sig), float(trunc)))

    grid = []

    # 1. Linear Slope-Intercept Grid
    if h_format in ("linearSlopeIntercept", "all"):
        for max_ang in max_angles:
            for f in factors_linear:
                ny = int(round(base_ny_linear * f))
                nx = int(round(base_nx_linear * f))
                for tol in tolerances:
                    # Gaussian smoothing cases
                    for smth, sig, trunc in smooth_configs:
                        grid.append({
                            "h_format": "linearSlopeIntercept",
                            "binning_factor": f,
                            "tolerance": tol,
                            "max_angle": float(max_ang),
                            "z_target": z_target,
                            "target_window_xz": target_window_xz,
                            "target_window_yz": target_window_yz,
                            "n_yH": ny,
                            "n_xH": nx,
                            "yH_range_xz": (-105.0, 17.0),
                            "yH_range_yz": (0.0, 122.0),
                            "xH_range_xz": (-1.1, 1.1),
                            "xH_range_yz": (-1.1, 1.1),
                            "smooth_full": smth,
                            "sigma": sig,
                            "truncate": trunc,
                            "n_random": 0,
                        })
                    # No-smoothing case
                    if include_no_smooth:
                        grid.append({
                            "h_format": "linearSlopeIntercept",
                            "binning_factor": f,
                            "tolerance": tol,
                            "max_angle": float(max_ang),
                            "z_target": z_target,
                            "target_window_xz": target_window_xz,
                            "target_window_yz": target_window_yz,
                            "n_yH": ny,
                            "n_xH": nx,
                            "yH_range_xz": (-105.0, 17.0),
                            "yH_range_yz": (0.0, 122.0),
                            "xH_range_xz": (-1.1, 1.1),
                            "xH_range_yz": (-1.1, 1.1),
                            "smooth_full": False,
                            "sigma": 0.0,
                            "truncate": 0.0,
                            "n_random": 0,
                        })

    # 2. Normal Representation Grid
    if h_format in ("normal", "all"):
        for max_ang in max_angles:
            for f in factors_normal:
                ny = int(round(base_ny_normal * f))
                nx = int(round(base_nx_normal * f))
                for tol in tolerances:
                    for smth, sig, trunc in smooth_configs:
                        grid.append({
                            "h_format": "normal",
                            "binning_factor": f,
                            "tolerance": tol,
                            "max_angle": float(max_ang),
                            "z_target": z_target,
                            "target_window_xz": target_window_xz,
                            "target_window_yz": target_window_yz,
                            "n_yH": ny,
                            "n_xH": nx,
                            "yH_range_xz": (-80.0, 0.0),
                            "yH_range_yz": (0.0, 80.0),
                            "xH_range_xz": (-1.57, 1.57),
                            "xH_range_yz": (-1.57, 1.57),
                            "smooth_full": smth,
                            "sigma": sig,
                            "truncate": trunc,
                            "n_random": 0,
                        })
                    if include_no_smooth:
                        grid.append({
                            "h_format": "normal",
                            "binning_factor": f,
                            "tolerance": tol,
                            "max_angle": float(max_ang),
                            "z_target": z_target,
                            "target_window_xz": target_window_xz,
                            "target_window_yz": target_window_yz,
                            "n_yH": ny,
                            "n_xH": nx,
                            "yH_range_xz": (-80.0, 0.0),
                            "yH_range_yz": (0.0, 80.0),
                            "xH_range_xz": (-1.57, 1.57),
                            "xH_range_yz": (-1.57, 1.57),
                            "smooth_full": False,
                            "sigma": 0.0,
                            "truncate": 0.0,
                            "n_random": 0,
                        })

    return grid


# ---------------------------------------------------------------------------
# 4. Scoreboard and Reporting
# ---------------------------------------------------------------------------

def print_scoreboard(results: List[Dict[str, Any]], sort_key: str = "score_genie_genie", top_k: int = 25):
    header_titles = {
        "score_genie_genie": "GENIE Signal vs GENIE Bkg",
        "score_genie_pmu":   "GENIE Signal vs pmu2022 Bkg",
        "score_pgun_genie":  "Particle Gun vs GENIE Bkg",
        "score_pgun_pmu":    "Particle Gun vs pmu2022 Bkg",
        "score_mean":        "Composite Mean Score",
    }
    title = header_titles.get(sort_key, sort_key)

    print("\n" + "=" * 160)
    print(f" HOUGH PARAMETER OPTIMIZATION SCOREBOARD (Ranked by: {title})")
    print(f" Target Objective: Score = (1 - FPR) * (TPR)^2   |   Fiducial Acceptance: Target @ z=352 cm")
    print("=" * 163)
    print(f"{'Rank':<4} | {'Format':<14} | {'MaxAng':<6} | {'Factor':<6} | {'Bins':<11} | {'Tol':<4} | {'Smth':<4} | {'Sig':<4} | {'Trn':<4} | "
          f"{'TPR(Gen)':<8} | {'FPR(Gen)':<8} | {'Score(GG)':<9} | {'TPR(Pgun)':<9} | {'FPR(pmu)':<8} | {'Score(GP)':<9} | {'Score(PG)':<9} | {'Score(PP)':<9}")
    print("-" * 163)

    for i, res in enumerate(results[:top_k]):
        p = res["params"]
        fmt_short = "Linear" if p["h_format"] == "linearSlopeIntercept" else "Normal"
        bins_str = f"{p['n_xH']}x{p['n_yH']}"
        smth = "Yes" if p["smooth_full"] else "No"
        sig_str = f"{p['sigma']:.1f}" if p["smooth_full"] else "-"
        trunc_str = f"{p['truncate']:.1f}" if p["smooth_full"] else "-"
        max_ang_str = f"{p.get('max_angle', 0.175):.3f}"

        print(f"{i+1:<4} | {fmt_short:<14} | {max_ang_str:<6} | {p['binning_factor']:<6.2f} | {bins_str:<11} | {p['tolerance']:<4.1f} | {smth:<4} | {sig_str:<4} | {trunc_str:<4} | "
              f"{res['tpr_genie']*100:<7.1f}% | {res['fpr_genie']*100:<7.1f}% | {res['score_genie_genie']:<9.4f} | "
              f"{res['tpr_pgun']*100:<8.1f}% | {res['fpr_pmu']*100:<7.1f}% | {res['score_genie_pmu']:<9.4f} | "
              f"{res['score_pgun_genie']:<9.4f} | {res['score_pgun_pmu']:<9.4f}")
    print("=" * 163 + "\n")


# ---------------------------------------------------------------------------
# 5. Plotting Response Curves & Surfaces
# ---------------------------------------------------------------------------

def plot_tuning_summary(results: List[Dict[str, Any]], output_plot_path: str = "pgun/plots/hough_fom_score_landscape.png"):
    """Plot Score, TPR, and FPR across binning factors and tolerances."""
    os.makedirs(os.path.dirname(os.path.abspath(output_plot_path)), exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import ddfUtils.mpl_styling as dms
        has_dms = True
    except ImportError:
        has_dms = False

    best = results[0]
    bp = best["params"]
    fmt = bp["h_format"]

    # Filter results for best format and best smoothing
    slice_res = [
        r for r in results 
        if r["params"]["h_format"] == fmt 
        and r["params"]["smooth_full"] == bp["smooth_full"] 
        and r["params"]["sigma"] == bp["sigma"]
        and r["params"]["truncate"] == bp["truncate"]
    ]

    factors = sorted(list(set(r["params"]["binning_factor"] for r in slice_res)))
    tols = sorted(list(set(r["params"]["tolerance"] for r in slice_res)))

    mat_score_gg = np.full((len(tols), len(factors)), np.nan)
    mat_tpr_g = np.full((len(tols), len(factors)), np.nan)
    mat_fpr_g = np.full((len(tols), len(factors)), np.nan)
    mat_score_gp = np.full((len(tols), len(factors)), np.nan)

    for r in slice_res:
        i_t = tols.index(r["params"]["tolerance"])
        i_f = factors.index(r["params"]["binning_factor"])
        mat_score_gg[i_t, i_f] = r["score_genie_genie"]
        mat_tpr_g[i_t, i_f] = r["tpr_genie"] * 100.0
        mat_fpr_g[i_t, i_f] = r["fpr_genie"] * 100.0
        mat_score_gp[i_t, i_f] = r["score_genie_pmu"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    panels = [
        (axes[0, 0], mat_score_gg, "Score (Genie Sig vs Genie Bkg)", "Score = (1 - FPR) * TPR^2", "viridis"),
        (axes[0, 1], mat_tpr_g, "Genie Signal TPR [%]", "True Positive Rate [%]", "plasma"),
        (axes[1, 0], mat_fpr_g, "Genie Background FPR [%]", "False Positive Rate [%]", "inferno"),
        (axes[1, 1], mat_score_gp, "Score (Genie Sig vs pmu2022 Bkg)", "Score = (1 - FPR) * TPR^2", "viridis"),
    ]

    for ax, data, title, cb_lbl, cmap in panels:
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(factors)))
        ax.set_xticklabels([f"{x:.2f}" for x in factors], rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(tols)))
        ax.set_yticklabels([f"{y:.1f}" for y in tols])
        ax.set_xlabel("Hough Space Binning Factor", fontsize=11)
        ax.set_ylabel("Tolerance [cm]", fontsize=11)
        ax.set_title(title, fontsize=12, pad=8)

        # Mark optimum
        best_f_idx = factors.index(bp["binning_factor"])
        best_t_idx = tols.index(bp["tolerance"])
        ax.plot(best_f_idx, best_t_idx, marker="*", markersize=16, color="red", markeredgecolor="white", markeredgewidth=1.5, zorder=5)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cb_lbl, fontsize=10)

    if has_dms:
        dms.add_label(mainText="SND@LHC", extraText="Hough Optimization (FOM)", suppText=r"|\theta| \le 175 mrad, Target @ z=352", ax=axes[0, 0])

    plt.suptitle(f"Dimuon Hough Parameter Optimization Landscape ({fmt})\nObjective: Score = (1 - FPR) * TPR^2", fontsize=14, weight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200, bbox_inches="tight")
    pdf_path = output_plot_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved response heatmaps to {output_plot_path} and {pdf_path}")


# ---------------------------------------------------------------------------
# 6. Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-Sample Hough Parameter Optimizer with (1 - FPR) * (TPR)^2 Metric")
    parser.add_argument("--genie-sig-cache", default="pgun/tuning_cache_genie_341ev.pkl", help="Genie Direct Signal cache")
    parser.add_argument("--pgun-sig-cache", default="pgun/tuning_cache_500ev.pkl", help="Particle Gun Signal cache")
    parser.add_argument("--genie-bkg-cache", default="pgun/cache_genie_bkg_500ev.pkl", help="Genie Background cache")
    parser.add_argument("--pmu-bkg-cache", default="pgun/cache_pmu_bkg_500ev.pkl", help="pmu2022 Background cache")
    parser.add_argument("--sort-by", choices=["genie_genie", "genie_pmu", "pgun_genie", "pgun_pmu", "mean"], default="genie_genie",
                        help="Metric pair to rank scoreboard by")
    parser.add_argument("--hf", "--h-format", "--hough-format", dest="hf",
                        choices=["linearSlopeIntercept", "normal", "all"],
                        default="linearSlopeIntercept",
                        help="Hough space format to search over (default: linearSlopeIntercept)")
    parser.add_argument("--max-angles", nargs="+", type=float, default=[0.180, 0.200, 0.220],
                        help="Maximum track angles (in rad) to include in grid search (default: 0.180 0.200 0.220)")
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 3.0],
                        help="Gaussian smoothing sigma values in units of bins (default: 0.5 1.0 1.5 2.0 3.0)")
    parser.add_argument("--truncates", "--truncs", nargs="+", type=float, default=[1.0, 2.0, 3.0, 4.0],
                        help="Gaussian filter truncation factors in units of sigma (default: 1.0 2.0 3.0 4.0)")
    parser.add_argument("--no-smoothing", dest="include_no_smooth", action="store_false", default=True,
                        help="Exclude the unsmoothed (no-smoothing) configuration from grid search")
    parser.add_argument("--xml", default="config/TrackingParams.xml", help="Path to TrackingParams.xml")
    parser.add_argument("-w", "--workers", type=int, default=12, help="Number of parallel worker processes")
    parser.add_argument("--max-events", type=int, default=0, help="Cap number of events per dataset (0 = all cached events)")
    parser.add_argument("--output-pickle", default="pgun/plots/hough_fom_results.pkl", help="Output pickle path for results")
    args = parser.parse_args()

    # Load plane crossing constraints from XML if available
    z_plane = 352.0
    target_window_xz = (-50.0, -3.0)
    target_window_yz = (5.0, 60.0)
    if os.path.isfile(args.xml):
        try:
            tree_xml = ET.parse(args.xml)
            for tc in tree_xml.getroot().findall("tracking_case"):
                if tc.attrib.get("name") == "dimuon_DS":
                    if tc.find("z_plane") is not None:
                        z_plane = float(tc.findtext("z_plane").strip())
                    elif tc.find("z_sf5") is not None:
                        z_plane = float(tc.findtext("z_sf5").strip())
                    if tc.find("x_area_min") is not None and tc.find("x_area_max") is not None:
                        target_window_xz = (float(tc.findtext("x_area_min").strip()), float(tc.findtext("x_area_max").strip()))
                    if tc.find("y_area_min") is not None and tc.find("y_area_max") is not None:
                        target_window_yz = (float(tc.findtext("y_area_min").strip()), float(tc.findtext("y_area_max").strip()))
                    break
        except Exception as e:
            print(f"Warning: Could not parse crossing window from {args.xml}: {e}")

    # Load the 4 benchmark datasets
    samples = {}
    for name, cpath in [
        ("genie_sig", args.genie_sig_cache),
        ("pgun_sig",  args.pgun_sig_cache),
        ("genie_bkg", args.genie_bkg_cache),
        ("pmu_bkg",   args.pmu_bkg_cache),
    ]:
        if not os.path.isfile(cpath):
            print(f"ERROR: Required cache file '{cpath}' not found. Run extract_hough_caches.py first.")
            sys.exit(1)
        with open(cpath, "rb") as f:
            raw_data = pickle.load(f)
            if args.max_events and len(raw_data) > args.max_events:
                samples[name] = raw_data[:args.max_events]
            else:
                samples[name] = raw_data
        print(f"Loaded {name:<10}: {len(samples[name])} events from {cpath}")

    # Build parameter grid
    grid = generate_parameter_grid(
        h_format=args.hf,
        max_angles=args.max_angles,
        sigmas=args.sigmas,
        truncates=args.truncates,
        include_no_smooth=args.include_no_smooth,
        z_plane=z_plane,
        target_window_xz=target_window_xz,
        target_window_yz=target_window_yz,
    )
    n_smooth_combos = len(args.sigmas) * len(args.truncates) + (1 if args.include_no_smooth else 0)
    print(f"\nGenerated parameter grid with {len(grid)} configurations (format: {args.hf}):")
    print(f"  - Max angle cuts:         {args.max_angles} rad ({len(args.max_angles)} values)")
    print(f"  - Smoothing configs:      {len(args.sigmas)} sigmas {args.sigmas} x {len(args.truncates)} truncates {args.truncates} + {'no-smooth' if args.include_no_smooth else 'none'} = {n_smooth_combos} combinations")
    if args.hf in ("linearSlopeIntercept", "all"):
        print(f"  - Linear Slope-Intercept: {len(args.max_angles)} angles x 16 binning factors x 8 tolerances x {n_smooth_combos} smooth configs = {len(args.max_angles) * 16 * 8 * n_smooth_combos} configs")
    if args.hf in ("normal", "all"):
        print(f"  - Normal representation:  {len(args.max_angles)} angles x 11 binning factors x 8 tolerances x {n_smooth_combos} smooth configs = {len(args.max_angles) * 11 * 8 * n_smooth_combos} configs")
    print(f"  - Working Area Window:    Plane z_plane={z_plane} cm, x in {target_window_xz}, y in {target_window_yz}")
    print(f"\nStarting parallel evaluation using {args.workers} worker processes...")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(samples,)) as ex:
        results = list(ex.map(evaluate_config, grid))
    elapsed = time.time() - t0
    print(f"Evaluation complete in {elapsed:.1f} s ({elapsed/len(grid)*1000:.1f} ms / config across all 4 samples).")

    # Sort results
    sort_key_map = {
        "genie_genie": "score_genie_genie",
        "genie_pmu":   "score_genie_pmu",
        "pgun_genie":  "score_pgun_genie",
        "pgun_pmu":    "score_pgun_pmu",
        "mean":        "score_mean",
    }
    target_metric = sort_key_map[args.sort_by]
    results.sort(key=lambda r: -r[target_metric])

    # Print scoreboard
    print_scoreboard(results, sort_key=target_metric, top_k=25)

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output_pickle)), exist_ok=True)
    with open(args.output_pickle, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved full evaluation results to {args.output_pickle}")

    # Plot response landscapes
    plot_tuning_summary(results)

    best = results[0]
    bp = best["params"]
    print("=" * 70)
    print(" >>> OPTIMAL HOUGH CONFIGURATION <<<")
    print(f"  Format:          {bp['h_format']}")
    print(f"  Binning Factor:  {bp['binning_factor']}")
    print(f"  Accumulator Res: {bp['n_xH']} x {bp['n_yH']} bins")
    print(f"  Tolerance:       {bp['tolerance']} cm")
    print(f"  Smooth Full:     {bp['smooth_full']} (sigma={bp['sigma']}, truncate={bp['truncate']})")
    print(f"  Max Angle:       {bp['max_angle']} rad (175 mrad)")
    print(f"  Target Window:   z={bp['z_target']} cm, x in {bp['target_window_xz']}, y in {bp['target_window_yz']}")
    print("-" * 70)
    print(f"  Genie Signal TPR:       {best['tpr_genie']*100:.2f}%")
    print(f"  Genie Background FPR:   {best['fpr_genie']*100:.2f}% (Rejection: {(1-best['fpr_genie'])*100:.2f}%)")
    print(f"  Score (Genie vs Genie): {best['score_genie_genie']:.4f}")
    print(f"  Score (Genie vs pmu):   {best['score_genie_pmu']:.4f}")
    print(f"  Score (Pgun vs Genie):  {best['score_pgun_genie']:.4f}")
    print(f"  Score (Pgun vs pmu):    {best['score_pgun_pmu']:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
