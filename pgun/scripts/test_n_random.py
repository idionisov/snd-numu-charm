#!/usr/bin/env python3
"""
test_n_random.py
----------------
Direct 1D comparison of n_random across two smoothing regimes:
1. Regime A: Gaussian smoothing enabled (sigma = 1, truncate = 2), varying n_random in [0, 1, 2, 3]
2. Regime B: Gaussian smoothing disabled (smooth_full = 0), varying n_random in [0, 1, 2, 3, 5]

Fixed at the current optimal set:
  - Format: linearSlopeIntercept
  - Accumulator: 605 x 335 bins
  - Tolerance: 0.40 cm
"""

import os
import sys
import time
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple, Any

import numpy as np
import scipy.ndimage


def hit_finder(slope: float, intercept: float, box_centers: np.ndarray, box_ds: np.ndarray, tol: float = 0.0) -> np.ndarray:
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
    if len(systems) == 0:
        return 0
    planes = set()
    for sys_id, det_id in zip(systems, detector_ids):
        if sys_id == 3:
            planes.add((det_id % 10000) // 1000)
    return len(planes)


class StandaloneHoughRandom:
    def __init__(
        self,
        n_yH: int,
        yH_range: Tuple[float, float],
        n_xH: int,
        xH_range: Tuple[float, float],
        z_offset: float,
        smooth_full: bool = True,
        sigma: float = 1.0,
        truncate: float = 2.0,
        n_quantile: float = 0.5,
        res: float = 5.0,
    ):
        self.n_yH = n_yH
        self.n_xH = n_xH
        self.yH_range = yH_range
        self.xH_range = xH_range
        self.z_offset = z_offset
        self.smooth_full = smooth_full
        self.sigma = sigma
        self.truncate = truncate
        self.n_quantile = n_quantile
        self.res = res

        self.yH_bins = np.linspace(yH_range[0], yH_range[1], n_yH)
        self.xH_bins = np.linspace(xH_range[0], xH_range[1], n_xH)
        self.xH_i = np.arange(n_xH)

    def fit_random(self, hit_collection: np.ndarray, hit_d: np.ndarray, n_random: int = 0) -> Tuple[float, float]:
        if len(hit_collection) == 0:
            return (-999.0, -999.0)

        # Apply random smearing across bar dimension if n_random > 0
        if n_random > 0:
            rnd = (np.random.uniform(size=(n_random, len(hit_collection), 2)) - 0.5) * hit_d[None, :, :]
            pts_to_use = (hit_collection[None, :, :] + rnd).reshape(-1, 2)
        else:
            pts_to_use = hit_collection

        shifted_z = pts_to_use[:, 0] - self.z_offset
        y_vals = pts_to_use[:, 1]
        n_pts = len(shifted_z)

        # Vectorized accumulator filling for linearSlopeIntercept
        hit_yH = y_vals[:, None] - shifted_z[:, None] * self.xH_bins[None, :]
        valid = (hit_yH >= self.yH_range[0]) & (hit_yH <= self.yH_range[1])
        if not np.any(valid):
            return (-999.0, -999.0)

        idx_y = np.floor((hit_yH - self.yH_range[0]) / (self.yH_range[1] - self.yH_range[0]) * self.n_yH).astype(int)
        idx_y = np.clip(idx_y, 0, self.n_yH - 1)
        col_idx = np.broadcast_to(self.xH_i, hit_yH.shape)

        accumulator = np.zeros((self.n_yH, self.n_xH), dtype=np.float32)
        np.add.at(accumulator, (idx_y[valid], col_idx[valid]), 1.0)

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

        slope = found_xH
        intercept = found_yH - slope * self.z_offset
        return (float(slope), float(intercept))


def run_reco_event(ev: Dict[str, Any], params: Dict[str, Any]) -> List[Dict[str, float]]:
    max_reco_muons = 2
    min_planes_hit = 3
    max_angle = 1.0
    tol = params["tolerance"]
    n_rnd = params["n_random"]

    pos = np.copy(ev["pos"])
    d = np.copy(ev["d"])
    vert = np.copy(ev["vert"])
    system = np.copy(ev["system"])
    det_id = np.copy(ev["det_id"])
    z_offset = ev["z_offset"]

    h_zy = StandaloneHoughRandom(
        n_yH=params["n_yH"], yH_range=params["yH_range_yz"],
        n_xH=params["n_xH"], xH_range=params["xH_range_yz"],
        z_offset=z_offset, smooth_full=params["smooth_full"],
        sigma=params["sigma"], truncate=params["truncate"],
    )

    h_zx = StandaloneHoughRandom(
        n_yH=params["n_yH"], yH_range=params["yH_range_xz"],
        n_xH=params["n_xH"], xH_range=params["xH_range_xz"],
        z_offset=z_offset, smooth_full=params["smooth_full"],
        sigma=params["sigma"], truncate=params["truncate"],
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

        slope_zy, b_zy = h_zy.fit_random(zy_pts, d_zy, n_random=n_rnd)
        slope_zx, b_zx = h_zx.fit_random(zx_pts, d_zx, n_random=n_rnd)

        if slope_zy == -999.0 or slope_zx == -999.0:
            break
        if abs(slope_zy) > max_angle or abs(slope_zx) > max_angle:
            break

        box_c_zy = np.column_stack([pos[2, ~vert], pos[1, ~vert]])[:, np.newaxis, :]
        box_d_zy = np.column_stack([d[2, ~vert], d[1, ~vert]])[:, np.newaxis, :]
        hits_zy = hit_finder(slope_zy, b_zy, box_c_zy, box_d_zy, tol=tol)

        box_c_zx = np.column_stack([pos[2, vert], pos[0, vert]])[:, np.newaxis, :]
        box_d_zx = np.column_stack([d[2, vert], d[0, vert]])[:, np.newaxis, :]
        hits_zx = hit_finder(slope_zx, b_zx, box_c_zx, box_d_zx, tol=tol)

        h_planes = num_planes_hit(system[~vert][hits_zy], det_id[~vert][hits_zy])
        v_planes = num_planes_hit(system[vert][hits_zx], det_id[vert][hits_zx])
        if h_planes < min_planes_hit or v_planes < min_planes_hit:
            break

        reconstructed_tracks.append({
            "tx": slope_zx, "ty": slope_zy,
            "bx": b_zx, "by": b_zy,
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


def eval_config(args: Tuple[List[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
    events, p = args
    # Set seed for reproducible stochastic comparison across configs
    np.random.seed(42)

    t0 = time.time()
    n_2trk = 0
    n_both_matched = 0
    ang_residuals = []

    for ev in events:
        tracks = run_reco_event(ev, p)
        if len(tracks) == 2:
            n_2trk += 1

        true_muons = ev["true_muons"]
        matched_truth = set()
        for rtrk in tracks:
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
            if best_dtheta < 0.050:
                matched_truth.add(best_mu_idx)
                ang_residuals.append(best_dtheta)

        if len(matched_truth) == 2:
            n_both_matched += 1

    elapsed = time.time() - t0
    tot = len(events)
    return {
        "regime": p["regime"],
        "smooth_full": p["smooth_full"],
        "sigma": p["sigma"],
        "n_random": p["n_random"],
        "eff_matched": (n_both_matched / tot) * 100.0,
        "eff_unconstrained": (n_2trk / tot) * 100.0,
        "mean_res": float(np.mean(ang_residuals) * 1000.0) if ang_residuals else 999.0,
        "ms_per_ev": (elapsed / tot) * 1000.0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test n_random across smoothing regimes")
    parser.add_argument("--sample", choices=["pgun", "genie"], default="pgun", help="Sample source: pgun or genie")
    parser.add_argument("--cache", default=None, help="Cache file path")
    args = parser.parse_args()

    if args.sample == "genie":
        cache_path = args.cache or "pgun/tuning_cache_genie_341ev.pkl"
    else:
        cache_path = args.cache or "pgun/tuning_cache_500ev.pkl"

    if not os.path.isfile(cache_path):
        print(f"ERROR: Cache file {cache_path} not found.")
        sys.exit(1)

    with open(cache_path, "rb") as f:
        events = pickle.load(f)

    # Base parameters matching the best set
    base_p = {
        "n_xH": 605,
        "n_yH": 335,
        "tolerance": 0.40,
        "yH_range_xz": (-105.0, 17.0),
        "yH_range_yz": (0.0, 122.0),
        "xH_range_xz": (-1.1, 1.1),
        "xH_range_yz": (-1.1, 1.1),
        "truncate": 2.0,
    }

    n_rnd_vals = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]

    regimes_def = [
        {"regime": "Gaussian sigma=1.0", "smooth_full": True, "sigma": 1.0, "color": "tab:blue", "marker": "o"},
        {"regime": "Gaussian sigma=0.7", "smooth_full": True, "sigma": 0.7, "color": "tab:cyan", "marker": "v"},
        {"regime": "Gaussian sigma=0.5", "smooth_full": True, "sigma": 0.5, "color": "tab:green", "marker": "s"},
        {"regime": "Gaussian sigma=0.3", "smooth_full": True, "sigma": 0.3, "color": "tab:orange", "marker": "D"},
        {"regime": "No Gaussian",        "smooth_full": False, "sigma": 0.0, "color": "tab:red", "marker": "^"},
    ]

    test_configs = []
    for reg in regimes_def:
        for n in n_rnd_vals:
            test_configs.append({
                **base_p,
                "regime": reg["regime"],
                "smooth_full": reg["smooth_full"],
                "sigma": reg["sigma"],
                "n_random": n,
            })

    print(f"\nEvaluating extended n_random grid ({len(test_configs)} configurations) on {len(events)} cached {args.sample.upper()} events...")
    tasks = [(events, cfg) for cfg in test_configs]
    with ProcessPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(eval_config, tasks))

    print("\n" + "=" * 102)
    print(f"{'Regime':<22} | {'Smooth':<7} | {'Sigma':<5} | {'n_rnd':<6} | {'Matched Eff':<12} | {'No-Cut Eff':<12} | {'Res (mrad)':<11} | {'Time (ms)':<9}")
    print("-" * 102)
    for r in results:
        sm_str = "Yes" if r["smooth_full"] else "No"
        sig_str = f"{r['sigma']:.1f}" if r["smooth_full"] else "-"
        print(f"{r['regime']:<22} | {sm_str:<7} | {sig_str:<5} | {r['n_random']:<6} | "
              f"{r['eff_matched']:<10.2f} % | {r['eff_unconstrained']:<10.2f} % | {r['mean_res']:<11.2f} | {r['ms_per_ev']:<9.2f}")
    print("=" * 102 + "\n")

    # Save results dictionary
    res_path = f"pgun/plots/n_random_results_{args.sample}.pkl"
    with open(res_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results cache to {res_path}")

    # Plot response curves
    plot_n_random_curves(results, regimes_def, sample_name=args.sample)


def plot_n_random_curves(results: List[Dict[str, Any]], regimes_def: List[Dict[str, Any]], sample_name: str = "genie", output_dir: str = "pgun/plots"):
    os.makedirs(output_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import ddfUtils.mpl_styling as dms
        has_dms = True
    except ImportError:
        has_dms = False

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    for reg_info in regimes_def:
        reg = reg_info["regime"]
        col = reg_info["color"]
        mkr = reg_info["marker"]
        reg_res = [r for r in results if r["regime"] == reg]
        reg_res.sort(key=lambda x: x["n_random"])
        xs = [r["n_random"] for r in reg_res]

        axes[0, 0].plot(xs, [r["eff_matched"] for r in reg_res], label=reg, color=col, marker=mkr, linewidth=2.0, markersize=5)
        axes[0, 1].plot(xs, [r["eff_unconstrained"] for r in reg_res], label=reg, color=col, marker=mkr, linewidth=2.0, markersize=5)
        axes[1, 0].plot(xs, [r["mean_res"] for r in reg_res], label=reg, color=col, marker=mkr, linewidth=2.0, markersize=5)
        axes[1, 1].plot(xs, [r["ms_per_ev"] for r in reg_res], label=reg, color=col, marker=mkr, linewidth=2.0, markersize=5)

    axes[0, 0].set_ylabel("Truth-Matched Efficiency [%]", fontsize=11)
    axes[0, 0].set_title("Dimuon Truth-Matched Efficiency (#Delta#theta < 50 mrad)", fontsize=12)
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].legend(fontsize=10)

    axes[0, 1].set_ylabel("2-Track Recovery [%] (No Cut)", fontsize=11)
    axes[0, 1].set_title("Unconstrained 2-Track Recovery (Without Angle Cut)", fontsize=12)
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    axes[0, 1].legend(fontsize=10)

    axes[1, 0].set_xlabel("n_random (Throws per measurement)", fontsize=11)
    axes[1, 0].set_ylabel("Mean Angular Residual [mrad]", fontsize=11)
    axes[1, 0].set_title("Reconstruction Angular Resolution", fontsize=12)
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    axes[1, 1].set_xlabel("n_random (Throws per measurement)", fontsize=11)
    axes[1, 1].set_ylabel("Execution Time [ms / event]", fontsize=11)
    axes[1, 1].set_title("Tracking Computation Time", fontsize=12)
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    sample_label = "GENIE Signal (in DS Acceptance)" if sample_name == "genie" else "Particle Gun"
    if has_dms:
        dms.add_label(mainText="SND@LHC", extraText=sample_label, suppText="linearSlopeIntercept", ax=axes[0, 0])

    plt.suptitle(f"Hough Reconstruction Optimization: n_random Response Curves\n({sample_label})", fontsize=14, weight="bold", y=0.99)
    plt.tight_layout()
    out_png = f"{output_dir}/n_random_scan_{sample_name}.png"
    out_pdf = f"{output_dir}/n_random_scan_{sample_name}.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved n_random response curves to {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
