#!/usr/bin/env python3
"""
tune_dimuon_hough.py
--------------------
Fast standalone benchmark and grid-search optimizer for SND@LHC Downstream Dimuon Hough Tracking.

Decouples Hough pattern recognition and hit association from FairRoot and Genfit I/O:
1. Caches hit coordinates and truth kinematics for ~500 particle-gun dimuon events.
2. Varies Hough parameters (tolerance, smooth_full, sigma, truncate, binning, representation)
   while keeping fixed:
     - min_planes_hit = 3
     - max_reco_muons = 2
     - max_angle = 1.0 rad
     - n_quantile = 0.5
     - res = 5
3. Evaluates 2-track efficiency, purity, ghost rate, and angular residuals against MC truth.
4. Outputs the optimal parameter set and a ranked summary table.
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import scipy.ndimage
from typing import Dict, List, Tuple, Any, Optional

import ROOT
import shipunit as u


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
    
    # Check corners
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
        sigma: float = 3.0,
        truncate: float = 4.0,
        n_quantile: float = 0.5,
        res: float = 5.0,
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

        self.yH_bins = np.linspace(yH_range[0], yH_range[1], n_yH)
        self.xH_bins = np.linspace(xH_range[0], xH_range[1], n_xH)
        self.cos_thetas = np.cos(self.xH_bins)
        self.sin_thetas = np.sin(self.xH_bins)
        self.xH_i = np.arange(n_xH)

    def fit(self, hit_collection: np.ndarray, hit_d: np.ndarray, n_random: int = 0) -> Tuple[float, float]:
        if len(hit_collection) == 0:
            return (-999.0, -999.0)

        # Handle random throws if requested
        if n_random > 0:
            points = []
            for hit, d in zip(hit_collection, hit_d):
                points.append(hit)
                for _ in range(n_random):
                    rnd_shift = np.random.uniform(-0.5, 0.5) * d[1]
                    points.append([hit[0], hit[1] + rnd_shift])
            hits_to_use = np.array(points)
        else:
            hits_to_use = hit_collection

        accumulator = np.zeros((self.n_yH, self.n_xH), dtype=np.float32)
        shifted_z = hits_to_use[:, 0] - self.z_offset
        y_vals = hits_to_use[:, 1]

        for s_z, y_pos in zip(shifted_z, y_vals):
            if self.h_format == 'normal':
                hit_yH = s_z * self.cos_thetas + y_pos * self.sin_thetas
            elif self.h_format == 'linearSlopeIntercept':
                hit_yH = y_pos - s_z * self.xH_bins
            elif self.h_format == 'linearIntercepts':
                denom = (self.det_zlen - s_z)
                if abs(denom) < 1e-4:
                    continue
                hit_yH = (self.det_zlen * y_pos - s_z * self.xH_bins) / denom
            else:
                raise ValueError(f"Unknown Hough format: {self.h_format}")

            valid = (hit_yH >= self.yH_range[0]) & (hit_yH <= self.yH_range[1])
            if not np.any(valid):
                continue
            idx_y = np.floor((hit_yH[valid] - self.yH_range[0]) / (self.yH_range[1] - self.yH_range[0]) * self.n_yH).astype(int)
            idx_y = np.clip(idx_y, 0, self.n_yH - 1)
            accumulator[idx_y, self.xH_i[valid]] += 1.0

        if self.smooth_full:
            accumulator = scipy.ndimage.gaussian_filter(accumulator, self.sigma, truncate=self.truncate)
            i_max = np.unravel_index(accumulator.argmax(), accumulator.shape)
        else:
            max_val = np.amax(accumulator)
            if max_val == 0:
                return (-999.0, -999.0)
            maxima = np.argwhere(accumulator == max_val)
            if len(maxima) == 1:
                i_max = maxima[0]
            elif len(maxima) > 1 and self.h_format == 'linearIntercepts':
                return (-999.0, -999.0)
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
        elif self.h_format == 'linearIntercepts':
            slope = (found_xH - found_yH) / self.det_zlen
            intercept = found_yH - slope * self.z_offset

        return (float(slope), float(intercept))


def run_reco_for_event(
    event_data: Dict[str, Any],
    params: Dict[str, Any],
) -> List[Dict[str, float]]:
    """Reconstruct up to max_reco_muons tracks for an event hit collection."""
    max_reco_muons = 2
    min_planes_hit = 3
    max_angle = 1.0
    tol = params["tolerance"]

    # Working copies of hit arrays
    pos = np.copy(event_data["pos"])        # shape (3, N)
    d = np.copy(event_data["d"])            # shape (3, N)
    vert = np.copy(event_data["vert"])      # shape (N,)
    system = np.copy(event_data["system"])  # shape (N,)
    det_id = np.copy(event_data["det_id"])  # shape (N,)

    z_offset = event_data["z_offset"]
    det_zlen = event_data["det_zlen"]

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
        if abs(slope_zy) > max_angle or abs(slope_zx) > max_angle:
            break

        # Associate hits
        box_c_zy = np.column_stack([pos[2, ~vert], pos[1, ~vert]])[:, np.newaxis, :]
        box_d_zy = np.column_stack([d[2, ~vert], d[1, ~vert]])[:, np.newaxis, :]
        hits_zy = hit_finder(slope_zy, b_zy, box_c_zy, box_d_zy, tol=tol)

        box_c_zx = np.column_stack([pos[2, vert], pos[0, vert]])[:, np.newaxis, :]
        box_d_zx = np.column_stack([d[2, vert], d[0, vert]])[:, np.newaxis, :]
        hits_zx = hit_finder(slope_zx, b_zx, box_c_zx, box_d_zx, tol=tol)

        # Check plane confirmation
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
            "assigned_det_ids": set(det_id[vert][hits_zx]).union(det_id[~vert][hits_zy]),
        })

        # Remove used hits
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
# 2. Benchmark Cache Extraction
# ---------------------------------------------------------------------------

def extract_cache(dig_file: str, geo_file: str, max_events: int = 500) -> List[Dict[str, Any]]:
    """Extract hits and truth muons from dig ROOT file into in-memory structure."""
    print(f"Extracting benchmark cache from {dig_file} (max {max_events} events)...")
    import SndlhcGeo
    geo = SndlhcGeo.GeoInterface(geo_file)
    f = ROOT.TFile.Open(dig_file)
    tree = f.Get("cbmsim")

    events = []
    nav = ROOT.gGeoManager.GetCurrentNavigator()

    for i_ev in range(min(max_events, tree.GetEntries())):
        tree.GetEntry(i_ev)

        # 1. Truth muons
        mctracks = tree.MCTrack
        muons = []
        for trk in mctracks:
            if abs(trk.GetPdgCode()) == 13 and trk.GetMotherId() < 0:
                p = ROOT.TVector3(trk.GetPx(), trk.GetPy(), trk.GetPz())
                if p.Pz() > 0:
                    muons.append({
                        "tx": p.Px() / p.Pz(),
                        "ty": p.Py() / p.Pz(),
                        "E": trk.GetEnergy(),
                    })
        if len(muons) < 2:
            continue

        # 2. Hits in DS
        mufi_hits = tree.Digi_MuFilterHits
        pos_list = []
        d_list = []
        vert_list = []
        sys_list = []
        detid_list = []

        a = ROOT.TVector3()
        b = ROOT.TVector3()

        for hit in mufi_hits:
            if not hit.isValid():
                continue
            det_id = hit.GetDetectorID()
            subsys = (det_id // 10000)
            if subsys != 3:  # DS only
                continue

            geo.modules["MuFilter"].GetPosition(det_id, a, b)
            center = (a + b) * 0.5
            is_vert = bool(hit.isVertical())
            
            # Bar dimension
            if is_vert:
                dx = abs(a.X() - b.X()) if abs(a.X() - b.X()) > 0.1 else 6.0
                dy = 6.0
            else:
                dx = 6.0
                dy = abs(a.Y() - b.Y()) if abs(a.Y() - b.Y()) > 0.1 else 6.0
            dz = 6.0

            pos_list.append([center.X(), center.Y(), center.Z()])
            d_list.append([dx, dy, dz])
            vert_list.append(is_vert)
            sys_list.append(subsys)
            detid_list.append(det_id)

        if len(pos_list) < 6:
            continue

        # DS Z-range parameters
        z_first = 430.0  # approximate DS start
        z_last = 520.0   # approximate DS end
        det_zlen = z_last - z_first

        events.append({
            "event_id": i_ev,
            "true_muons": muons,
            "pos": np.array(pos_list).T,  # (3, N)
            "d": np.array(d_list).T,      # (3, N)
            "vert": np.array(vert_list),
            "system": np.array(sys_list),
            "det_id": np.array(detid_list),
            "z_offset": z_first,
            "det_zlen": det_zlen,
        })

    f.Close()
    print(f"Extraction complete: cached {len(events)} valid dimuon events.")
    return events


# ---------------------------------------------------------------------------
# 3. Grid Search & Optimization Bench
# ---------------------------------------------------------------------------

def evaluate_parameters(
    events: List[Dict[str, Any]],
    param_grid: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results = []

    for idx, params in enumerate(param_grid):
        t0 = time.time()
        n_2trk = 0
        n_1trk = 0
        n_0trk = 0
        n_both_matched = 0
        ang_residuals = []

        for ev in events:
            reco_tracks = run_reco_for_event(ev, params)
            n_reco = len(reco_tracks)
            if n_reco == 2:
                n_2trk += 1
            elif n_reco == 1:
                n_1trk += 1
            else:
                n_0trk += 1

            # Match reconstructed tracks to truth muons
            true_muons = ev["true_muons"]
            matched_truth = set()
            for rtrk in reco_tracks:
                best_dtheta = 1e9
                best_mu_idx = -1
                for m_idx, mu in enumerate(true_muons):
                    if m_idx in matched_truth:
                        continue
                    dtx = rtrk["tx"] - mu["tx"]
                    dty = rtrk["ty"] - mu["ty"]
                    dtheta = np.sqrt(dtx**2 + dty**2)
                    if dtheta < best_dtheta:
                        best_dtheta = dtheta
                        best_mu_idx = m_idx
                if best_dtheta < 0.050:  # 50 mrad matching window
                    matched_truth.add(best_mu_idx)
                    ang_residuals.append(best_dtheta)

            if len(matched_truth) == 2:
                n_both_matched += 1

        elapsed = time.time() - t0
        total_ev = len(events)
        eff_2trk = n_both_matched / total_ev if total_ev > 0 else 0.0
        frac_2trk = n_2trk / total_ev if total_ev > 0 else 0.0
        mean_res = float(np.mean(ang_residuals) * 1000.0) if ang_residuals else 999.0

        res_entry = {
            "params": params,
            "eff_both_matched": eff_2trk,
            "frac_2tracks": frac_2trk,
            "n_2trk": n_2trk,
            "n_1trk": n_1trk,
            "n_0trk": n_0trk,
            "mean_res_mrad": mean_res,
            "time_ms_per_ev": (elapsed / total_ev) * 1000.0,
        }
        results.append(res_entry)

    # Sort by 2-track efficiency descending, then mean residual ascending
    results.sort(key=lambda r: (-r["eff_both_matched"], r["mean_res_mrad"]))
    return results


def plot_tuning_surfaces(results: List[Dict[str, Any]], output_plot_path: str = "pgun/plots/hough_tuning_landscape.png", output_root_path: str = "pgun/hough_tuning_2D.root"):
    """
    Generate 2D response surface heatmaps and ROOT TH2D histograms around the optimal parameter set.
    Visualizes 2D profile slices showing the local maximum in efficiency.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_plot_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_root_path)), exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import ddfUtils.mpl_styling as dms
        has_dms = True
    except ImportError:
        has_dms = False

    best = results[0]
    best_p = best["params"]

    # 1. Prepare 2D slices around best format
    fmt = best_p["h_format"]
    
    # Grid 1: Tolerance vs. Sigma (at best Truncate, Format, Res)
    tol_vals = sorted(list(set(r["params"]["tolerance"] for r in results)))
    sig_vals = sorted(list(set(r["params"]["sigma"] for r in results if r["params"]["smooth_full"])))
    trunc_vals = sorted(list(set(r["params"]["truncate"] for r in results if r["params"]["smooth_full"])))

    # Slice A: Tolerance vs Sigma (at best truncate)
    map_tol_sig = np.full((len(sig_vals), len(tol_vals)), np.nan)
    for r in results:
        p = r["params"]
        if p["h_format"] == fmt and p["smooth_full"] and p["truncate"] == best_p["truncate"] and p["n_xH"] == best_p["n_xH"]:
            i_sig = sig_vals.index(p["sigma"])
            i_tol = tol_vals.index(p["tolerance"])
            map_tol_sig[i_sig, i_tol] = r["eff_both_matched"] * 100.0

    # Slice B: Tolerance vs Truncate (at best sigma)
    map_tol_trunc = np.full((len(trunc_vals), len(tol_vals)), np.nan)
    for r in results:
        p = r["params"]
        if p["h_format"] == fmt and p["smooth_full"] and p["sigma"] == best_p["sigma"] and p["n_xH"] == best_p["n_xH"]:
            i_tr = trunc_vals.index(p["truncate"])
            i_tol = tol_vals.index(p["tolerance"])
            map_tol_trunc[i_tr, i_tol] = r["eff_both_matched"] * 100.0

    # Slice C: Sigma vs Truncate (at best tolerance)
    map_sig_trunc = np.full((len(trunc_vals), len(sig_vals)), np.nan)
    for r in results:
        p = r["params"]
        if p["h_format"] == fmt and p["smooth_full"] and p["tolerance"] == best_p["tolerance"] and p["n_xH"] == best_p["n_xH"]:
            i_tr = trunc_vals.index(p["truncate"])
            i_sig = sig_vals.index(p["sigma"])
            map_sig_trunc[i_tr, i_sig] = r["eff_both_matched"] * 100.0

    # Slice D: Tolerance vs Format/Resolution
    res_labels = ["linear (122x220)", "linear (244x440)", "intercepts (610)"]
    map_tol_res = np.full((len(res_labels), len(tol_vals)), np.nan)
    for r in results:
        p = r["params"]
        if p["smooth_full"] and p["sigma"] == best_p["sigma"] and p["truncate"] == best_p["truncate"]:
            if p["h_format"] == "linearSlopeIntercept" and p["n_xH"] == 220:
                i_res = 0
            elif p["h_format"] == "linearSlopeIntercept" and p["n_xH"] == 440:
                i_res = 1
            elif p["h_format"] == "linearIntercepts":
                i_res = 2
            else:
                continue
            i_tol = tol_vals.index(p["tolerance"])
            map_tol_res[i_res, i_tol] = r["eff_both_matched"] * 100.0

    # 2. Matplotlib 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    cmap = "viridis"

    panels = [
        (axes[0, 0], map_tol_sig, tol_vals, sig_vals, "Tolerance [cm]", r"Kernel Width $\sigma$", 
         f"Slice @ Truncate={best_p['truncate']:.0f}, Res={best_p['n_xH']}x{best_p['n_yH']}",
         tol_vals.index(best_p["tolerance"]), sig_vals.index(best_p["sigma"])),
        (axes[0, 1], map_tol_trunc, tol_vals, trunc_vals, "Tolerance [cm]", "Kernel Truncate",
         f"Slice @ $\sigma$={best_p['sigma']:.0f}, Res={best_p['n_xH']}x{best_p['n_yH']}",
         tol_vals.index(best_p["tolerance"]), trunc_vals.index(best_p["truncate"])),
        (axes[1, 0], map_sig_trunc, sig_vals, trunc_vals, r"Kernel Width $\sigma$", "Kernel Truncate",
         f"Slice @ Tol={best_p['tolerance']:.1f} cm, Res={best_p['n_xH']}x{best_p['n_yH']}",
         sig_vals.index(best_p["sigma"]), trunc_vals.index(best_p["truncate"])),
        (axes[1, 1], map_tol_res, tol_vals, [0, 1, 2], "Tolerance [cm]", "Representation & Resolution",
         f"Slice @ $\sigma$={best_p['sigma']:.0f}, Truncate={best_p['truncate']:.0f}",
         tol_vals.index(best_p["tolerance"]), 1),
    ]

    for ax, data, x_ticks, y_ticks, x_lbl, y_lbl, title, best_x_idx, best_y_idx in panels:
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels([f"{x:.1f}" if isinstance(x, float) else str(x) for x in x_ticks])
        ax.set_yticks(range(len(y_ticks)))
        if y_lbl == "Representation & Resolution":
            ax.set_yticklabels(res_labels)
        else:
            ax.set_yticklabels([f"{y:.1f}" if isinstance(y, float) else str(y) for y in y_ticks])

        ax.set_xlabel(x_lbl, fontsize=12)
        ax.set_ylabel(y_lbl, fontsize=12)
        ax.set_title(title, fontsize=12, pad=8)

        # Annotate values and mark maximum
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if not np.isnan(val):
                    color = "black" if val > (np.nanmax(data) * 0.75) else "white"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=10, weight="bold")

        # Mark optimum with star
        ax.plot(best_x_idx, best_y_idx, marker="*", markersize=18, color="red", markeredgecolor="white", markeredgewidth=1.5, zorder=5)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("2-Track Efficiency [%]", fontsize=11)

    if has_dms:
        dms.add_label(mainText="SND@LHC", extraText="Simulation - Particle Gun", suppText=f"Optimum: {best['eff_both_matched']*100:.1f}%", ax=axes[0, 0])

    plt.suptitle("Dimuon Hough Tracking Optimization Landscape (Local Maxima Profiles)", fontsize=15, weight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200, bbox_inches="tight")
    pdf_path = output_plot_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved response surface heatmaps to {output_plot_path} and {pdf_path}")

    # 3. Save ROOT TH2D histograms
    r_file = ROOT.TFile.Open(output_root_path, "RECREATE")
    
    # Histogram 1: Tol vs Sigma
    h2_tol_sig = ROOT.TH2D("h2_eff_tol_sigma", "2-Track Efficiency vs (Tolerance, #sigma);Tolerance [cm];#sigma",
                           len(tol_vals), 0, len(tol_vals), len(sig_vals), 0, len(sig_vals))
    for i, t in enumerate(tol_vals):
        h2_tol_sig.GetXaxis().SetBinLabel(i + 1, f"{t:.1f}")
    for i, s in enumerate(sig_vals):
        h2_tol_sig.GetYaxis().SetBinLabel(i + 1, f"{s:.1f}")
    for i in range(len(sig_vals)):
        for j in range(len(tol_vals)):
            if not np.isnan(map_tol_sig[i, j]):
                h2_tol_sig.SetBinContent(j + 1, i + 1, map_tol_sig[i, j])
    h2_tol_sig.Write()

    # Histogram 2: Tol vs Truncate
    h2_tol_tr = ROOT.TH2D("h2_eff_tol_truncate", "2-Track Efficiency vs (Tolerance, Truncate);Tolerance [cm];Truncate",
                          len(tol_vals), 0, len(tol_vals), len(trunc_vals), 0, len(trunc_vals))
    for i, t in enumerate(tol_vals):
        h2_tol_tr.GetXaxis().SetBinLabel(i + 1, f"{t:.1f}")
    for i, tr in enumerate(trunc_vals):
        h2_tol_tr.GetYaxis().SetBinLabel(i + 1, f"{tr:.1f}")
    for i in range(len(trunc_vals)):
        for j in range(len(tol_vals)):
            if not np.isnan(map_tol_trunc[i, j]):
                h2_tol_tr.SetBinContent(j + 1, i + 1, map_tol_trunc[i, j])
    h2_tol_tr.Write()

    # Histogram 3: Sigma vs Truncate
    h2_sig_tr = ROOT.TH2D("h2_eff_sigma_truncate", "2-Track Efficiency vs (#sigma, Truncate);#sigma;Truncate",
                          len(sig_vals), 0, len(sig_vals), len(trunc_vals), 0, len(trunc_vals))
    for i, s in enumerate(sig_vals):
        h2_sig_tr.GetXaxis().SetBinLabel(i + 1, f"{s:.1f}")
    for i, tr in enumerate(trunc_vals):
        h2_sig_tr.GetYaxis().SetBinLabel(i + 1, f"{tr:.1f}")
    for i in range(len(trunc_vals)):
        for j in range(len(sig_vals)):
            if not np.isnan(map_sig_trunc[i, j]):
                h2_sig_tr.SetBinContent(j + 1, i + 1, map_sig_trunc[i, j])
    h2_sig_tr.Write()

    r_file.Close()
    print(f"Saved ROOT 2D histograms to {output_root_path}")


def print_scoreboard(results: List[Dict[str, Any]], top_k: int = 15):
    print("\n" + "=" * 115)
    print(f"{'Rank':<4} | {'Format':<20} | {'Tol (cm)':<8} | {'Smth':<4} | {'Sigma':<5} | {'Trunc':<5} | {'2Trk Eff':<9} | {'2Trk %':<7} | {'Res (mrad)':<10} | {'Time (ms)':<9}")
    print("-" * 115)
    for i, res in enumerate(results[:top_k]):
        p = res["params"]
        smth = "Yes" if p["smooth_full"] else "No"
        print(f"{i+1:<4} | {p['h_format']:<20} | {p['tolerance']:<8.2f} | {smth:<4} | {p['sigma']:<5.1f} | {p['truncate']:<5.1f} | "
              f"{res['eff_both_matched']*100:<8.1f}% | {res['frac_2tracks']*100:<6.1f}% | {res['mean_res_mrad']:<10.2f} | {res['time_ms_per_ev']:<9.2f}")
    print("=" * 115)


def generate_parameter_grid() -> List[Dict[str, Any]]:
    """Build grid of parameter configurations to evaluate."""
    tolerances = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
    formats = [
        # (name, n_yH, n_xH, yH_range_xz, yH_range_yz, xH_range_xz, xH_range_yz)
        ('linearSlopeIntercept', 122, 220, (-105.0, 17.0), (0.0, 122.0), (-1.1, 1.1), (-1.1, 1.1)),
        ('linearSlopeIntercept', 244, 440, (-105.0, 17.0), (0.0, 122.0), (-1.1, 1.1), (-1.1, 1.1)),  # 2x fine
        ('linearIntercepts', 610, 610, (-105.0, 17.0), (0.0, 122.0), (-105.0, 17.0), (0.0, 122.0)),
    ]
    smooth_configs = [
        (True, 1.0, 2.0),
        (True, 2.0, 3.0),
        (True, 3.0, 4.0),
        (False, 0.0, 0.0),
    ]

    grid = []
    for fmt_name, ny, nx, yrange_xz, yrange_yz, xrange_xz, xrange_yz in formats:
        for tol in tolerances:
            for smth, sig, trunc in smooth_configs:
                # Disallow un-smoothed linearIntercepts (fails if multiple maxima)
                if not smth and fmt_name == 'linearIntercepts':
                    continue
                grid.append({
                    "h_format": fmt_name,
                    "tolerance": tol,
                    "n_yH": ny,
                    "n_xH": nx,
                    "yH_range_xz": yrange_xz,
                    "yH_range_yz": yrange_yz,
                    "xH_range_xz": xrange_xz,
                    "xH_range_yz": xrange_yz,
                    "smooth_full": smth,
                    "sigma": sig,
                    "truncate": trunc,
                    "n_random": 0,
                })
    return grid


# ---------------------------------------------------------------------------
# 4. Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hough Parameter Optimizer for SND@LHC Dimuon Tracking")
    parser.add_argument("-f", "--digFile", default="/eos/user/i/idioniso/snd-numu-charm/data/particle_gun_dimuon/sndLHC.TwoMuons-TGeant4_dig.root")
    parser.add_argument("-g", "--geoFile", default="/eos/user/i/idioniso/snd-numu-charm/data/particle_gun_dimuon/geofile_full.TwoMuons-TGeant4.root")
    parser.add_argument("-c", "--cache", default="pgun/tuning_cache_500ev.pkl", help="Local cache path")
    parser.add_argument("-n", "--nEvents", type=int, default=300, help="Number of events to benchmark")
    parser.add_argument("--force-cache", action="store_true", help="Re-extract cache from ROOT file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)

    if os.path.exists(args.cache) and not args.force_cache:
        print(f"Loading cached benchmark events from {args.cache}...")
        with open(args.cache, "rb") as cf:
            events = pickle.load(cf)
        print(f"Loaded {len(events)} events.")
    else:
        events = extract_cache(args.digFile, args.geoFile, max_events=args.nEvents)
        with open(args.cache, "wb") as cf:
            pickle.dump(events, cf)
        print(f"Saved benchmark cache to {args.cache}.")

    if not events:
        print("ERROR: No valid benchmark events found.")
        sys.exit(1)

    grid = generate_parameter_grid()
    print(f"\nInitiating Hough parameter scan across {len(grid)} configurations on {len(events)} events...")
    results = evaluate_parameters(events, grid)
    print_scoreboard(results, top_k=20)
    plot_tuning_surfaces(results)

    best = results[0]
    bp = best["params"]
    print("\n>>> BEST PARAMETER CONFIGURATION <<<")
    print(f"  Format         : {bp['h_format']}")
    print(f"  Tolerance      : {bp['tolerance']} cm")
    print(f"  Smooth Full    : {bp['smooth_full']}")
    print(f"  Sigma          : {bp['sigma']}")
    print(f"  Truncate       : {bp['truncate']}")
    print(f"  Accumulator Res: {bp['n_xH']} x {bp['n_yH']} bins")
    print(f"  2-Track Eff    : {best['eff_both_matched']*100:.2f}%")
    print(f"  Ang Residual   : {best['mean_res_mrad']:.2f} mrad")
    print(f"  Speed          : {best['time_ms_per_ev']:.2f} ms / event\n")


if __name__ == "__main__":
    main()
