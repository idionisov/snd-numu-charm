#!/usr/bin/env python3
"""
scripts/timing/plot_backward_muon_timing.py
-------------------------------------------
Comprehensive timing analysis and visualization script for backward-moving muons (e.g. B2noB1).
Extracts absolute and relative hit times across all 4 detector subsystems:
  - Veto (System 1)
  - SciFi (System 0 / ScifiHits)
  - Upstream Muon Filter / HCAL (US, System 2)
  - Downstream Muon Filter (DS, System 3)

Plots:
  1. Absolute hit times:
     - All hits per subsystem (Veto, SciFi, US, DS)
     - Earliest (min) hit time per event per subsystem
     - Latest (max) hit time per event per subsystem
     - Multi-subsystem overlay comparisons
  2. Relative hit times (time differences):
     - Earliest SciFi - Earliest Veto:  t(SciFi_min) - t(Veto_min)
     - Latest SciFi - Latest Veto:      t(SciFi_max) - t(Veto_max)
     - Latest US - Earliest Veto:       t(US_max) - t(Veto_min)
     - Latest US - Latest SciFi:        t(US_max) - t(SciFi_max)
     - Latest DS - Latest Veto:         t(DS_max) - t(Veto_max)
     - Latest DS - Earliest SciFi:      t(DS_max) - t(SciFi_min)
     - Latest DS - Earliest US:         t(DS_max) - t(US_min)
     - Inverted SciFi-DS timing cut:    t(SciFi_max) - t(DS_min)
     - Additional complementary relative timing combinations
  3. 2D correlation distributions between subsystem hit times.

Outputs:
  - Formatted ROOT file (.root) with all 1D/2D histograms and summary canvases.
  - Publication-grade PNG and PDF figures styled via ddfUtils.mpl_styling.
  - Tabulated summary of statistical parameters (entries, mean, std dev, median, IQR).

Usage examples:
  # Analyze first 5 prefiltered partitions of Run 6596:
  python3 scripts/timing/plot_backward_muon_timing.py --max-files 5

  # Analyze a specific single ROOT file:
  python3 scripts/timing/plot_backward_muon_timing.py -i /eos/user/i/idioniso/snd-numu-charm/prefiltered/2023/b2nob1Muons/run006596/sndsw_raw-0000.root

  # Run on all partitions with 8 parallel worker processes:
  python3 scripts/timing/plot_backward_muon_timing.py -j 8 -o plots/timing/run6596_b2nob1_full
"""

from __future__ import annotations

import os
import sys
import glob
import time
import argparse
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Force matplotlib headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import ddfUtils.mpl_styling as dms
    HAS_DMS = True
except ImportError:
    HAS_DMS = False

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat("emruo")


DEFAULT_INPUT_DIR = "/eos/user/i/idioniso/snd-numu-charm/prefiltered/2023/b2nob1Muons/run006596"
DEFAULT_INPUT_PATTERN = os.path.join(DEFAULT_INPUT_DIR, "sndsw_raw-*.root")
DEFAULT_OUTPUT_DIR = "plots/timing/run006596_b2nob1"
DEFAULT_GEOFILE = "/eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18_V3_2023.root"
TDC2NS = 6.23768  # 1 clock tick = 1 / 0.160316 GHz ~ 6.23768 ns


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Timing distributions and relative time differences for backward muons."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=DEFAULT_INPUT_PATTERN,
        help=f"Input file, wildcard pattern, or directory (default: {DEFAULT_INPUT_PATTERN})",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for generated plots and ROOT file (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "-n", "--max-events",
        type=int,
        default=0,
        help="Maximum total events to process across all files (default: 0 = all)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum number of files to process (default: 0 = all matched files)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of parallel worker processes for file reading (default: 4)",
    )
    parser.add_argument(
        "--units",
        choices=["ns", "tdc"],
        default="ns",
        help="Time units: 'ns' (nanoseconds, default) or 'tdc' (raw clock cycles)",
    )
    parser.add_argument(
        "--geo", "--geofile",
        dest="geofile",
        type=str,
        default=DEFAULT_GEOFILE,
        help=f"Geofile path for detector geometry calibration (default: {DEFAULT_GEOFILE})",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="Run 6596 B2noB1 Muons",
        help="Label tag displayed on plot headers",
    )
    parser.add_argument(
        "--root-only",
        action="store_true",
        default=True,
        help="Only save ROOT file (.root) with histograms; do not generate PNG/PDF figures (default: True)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=False,
        help="Generate PNG and PDF figures in addition to the ROOT file",
    )
    return parser.parse_args()


def resolve_input_files(input_arg: str, max_files: int = 0) -> List[str]:
    """Resolve single file, comma-separated list, directory, or glob pattern into sorted file paths."""
    resolved: List[str] = []
    if "," in input_arg:
        tokens = [t.strip() for t in input_arg.split(",") if t.strip()]
        for tok in tokens:
            resolved.extend(resolve_input_files(tok))
    elif os.path.isdir(input_arg):
        resolved = sorted(glob.glob(os.path.join(input_arg, "*.root")))
    elif "%s" in input_arg or "%04d" in input_arg:
        base_dir = os.path.dirname(input_arg) or "."
        resolved = sorted(glob.glob(os.path.join(base_dir, "*.root")))
    elif any(char in input_arg for char in ["*", "?", "["]):
        resolved = sorted(glob.glob(input_arg))
    elif os.path.isfile(input_arg):
        resolved = [input_arg]
    else:
        # Fallback glob
        resolved = sorted(glob.glob(input_arg))

    if max_files > 0:
        resolved = resolved[:max_files]
    return resolved


def _process_single_file(args_tuple: Tuple[str, int, str, Optional[str]]) -> Dict[str, Any]:
    """Worker task extracting timing arrays from one ROOT file."""
    filepath, file_event_limit, units, geofile = args_tuple

    res: Dict[str, Any] = {
        "events_processed": 0,
        # Absolute hit times (all hits)
        "all_v": [],
        "all_sf": [],
        "all_us": [],
        "all_ds": [],
        # Event extrema
        "min_v": [],
        "max_v": [],
        "min_sf": [],
        "max_sf": [],
        "min_us": [],
        "max_us": [],
        "min_ds": [],
        "max_ds": [],
        # Requested relative differences
        "dt_sf_min_veto_min": [],
        "dt_sf_max_veto_max": [],
        "dt_us_max_veto_min": [],
        "dt_us_max_sf_max": [],
        "dt_ds_max_veto_max": [],
        "dt_ds_max_sf_min": [],
        "dt_ds_max_us_min": [],
        # Reference times and relative differences
        "t_ref_scifi": [],
        "t_ref_ds": [],
        "dt_ref_scifi_min_veto": [],
        "dt_ref_ds_min_veto": [],
        "dt_ref_ds_ref_scifi": [],
        # Track direction observables from sndRecoTrack in Reco_MuonTracks
        "track_direction": [],
        "track_velocity": [],
        "track_dir_slope": [],
        # Complementary differences
        "dt_sf_max_ds_min": [],
        "dt_ds_min_veto_min": [],
        "dt_us_min_sf_min": [],
        "dt_ds_min_sf_min": [],
    }

    if not os.path.isfile(filepath):
        return res

    f_in = ROOT.TFile.Open(filepath, "READ")
    if not f_in or f_in.IsZombie():
        if f_in:
            f_in.Close()
        return res

    tree = f_in.Get("rawConv") or f_in.Get("cbmsim")
    if not tree:
        f_in.Close()
        return res

    is_data = (tree.GetName() == "rawConv")
    tree.SetBranchStatus("*", 0)
    tree.SetBranchStatus("Digi_ScifiHits*", 1)
    tree.SetBranchStatus("Digi_MuFilterHits*", 1)
    has_tracks = bool(tree.GetBranch("Reco_MuonTracks") or tree.GetBranch("Reco_MuonTracks."))
    if has_tracks:
        tree.SetBranchStatus("Reco_MuonTracks*", 1)

    scale_factor = TDC2NS if (units == "ns" and is_data) else 1.0

    n_entries = tree.GetEntries()
    limit = n_entries if file_event_limit <= 0 else min(n_entries, file_event_limit)
    res["events_processed"] = limit

    for iev in range(limit):
        tree.GetEntry(iev)

        # 1. SciFi hit times
        sf_times: List[float] = []
        sf_horiz_times: List[float] = []
        scifi_hits = tree.Digi_ScifiHits
        if scifi_hits is not None:
            n_sf = scifi_hits.GetEntriesFast()
            for i in range(n_sf):
                h = scifi_hits.At(i)
                if h and h.isValid():
                    raw_t = float(h.GetTime()) * scale_factor
                    sf_times.append(raw_t)
                    if not h.isVertical():
                        sf_horiz_times.append(raw_t)

        # 2. MuFilter hit times (Veto = 1, US = 2, DS = 3)
        v_times: List[float] = []
        us_times: List[float] = []
        ds_times: List[float] = []
        ds_horiz_sipm_times: List[float] = []
        mufi_hits = tree.Digi_MuFilterHits
        if mufi_hits is not None:
            n_mf = mufi_hits.GetEntriesFast()
            for i in range(n_mf):
                h = mufi_hits.At(i)
                if not h or not h.isValid():
                    continue
                sys_id = int(h.GetSystem())
                if is_data:
                    t0 = float(h.GetTime(0))
                    t1 = float(h.GetTime(1))
                    has_t0 = t0 > -900.0
                    has_t1 = t1 > -900.0
                    if not has_t0 and not has_t1:
                        continue
                    if h.isVertical():
                        t_val = t0 if has_t0 else t1
                    else:
                        if has_t0 and has_t1:
                            t_val = 0.5 * (t0 + t1)
                        elif has_t0:
                            t_val = t0
                        else:
                            t_val = t1
                    # For DS horizontal planes (readout on both sides), record individual SiPM times
                    if sys_id == 3 and not h.isVertical():
                        if has_t0:
                            ds_horiz_sipm_times.append(t0 * scale_factor)
                        if has_t1:
                            ds_horiz_sipm_times.append(t1 * scale_factor)
                else:
                    t_val = float(h.GetTime(0)) if h.isVertical() else float(0.5 * (h.GetTime(0) + h.GetTime(1)))
                    if sys_id == 3 and not h.isVertical():
                        ds_horiz_sipm_times.append(float(h.GetTime(0)) * scale_factor)
                        ds_horiz_sipm_times.append(float(h.GetTime(1)) * scale_factor)

                t_scaled = t_val * scale_factor
                if sys_id == 1:
                    v_times.append(t_scaled)
                elif sys_id == 2:
                    us_times.append(t_scaled)
                elif sys_id == 3:
                    ds_times.append(t_scaled)

        # Accumulate all hit times
        res["all_v"].extend(v_times)
        res["all_sf"].extend(sf_times)
        res["all_us"].extend(us_times)
        res["all_ds"].extend(ds_times)

        # Reference event times for horizontal planes
        t_ref_sf = (sum(sf_horiz_times) / len(sf_horiz_times)) if len(sf_horiz_times) > 0 else None
        t_ref_ds = (sum(ds_horiz_sipm_times) / len(ds_horiz_sipm_times)) if len(ds_horiz_sipm_times) > 0 else None

        if t_ref_sf is not None:
            res["t_ref_scifi"].append(t_ref_sf)
        if t_ref_ds is not None:
            res["t_ref_ds"].append(t_ref_ds)

        # Event extrema
        min_v = min(v_times) if v_times else None
        max_v = max(v_times) if v_times else None
        min_sf = min(sf_times) if sf_times else None
        max_sf = max(sf_times) if sf_times else None
        min_us = min(us_times) if us_times else None
        max_us = max(us_times) if us_times else None
        min_ds = min(ds_times) if ds_times else None
        max_ds = max(ds_times) if ds_times else None

        if min_v is not None:
            res["min_v"].append(min_v)
            res["max_v"].append(max_v)
        if min_sf is not None:
            res["min_sf"].append(min_sf)
            res["max_sf"].append(max_sf)
        if min_us is not None:
            res["min_us"].append(min_us)
            res["max_us"].append(max_us)
        if min_ds is not None:
            res["min_ds"].append(min_ds)
            res["max_ds"].append(max_ds)

        # Relative differences
        # 1. Earliest SciFi - Earliest Veto
        if min_sf is not None and min_v is not None:
            res["dt_sf_min_veto_min"].append(min_sf - min_v)

        # 2. Latest SciFi - Latest Veto
        if max_sf is not None and max_v is not None:
            res["dt_sf_max_veto_max"].append(max_sf - max_v)

        # 3. Latest US - Earliest Veto
        if max_us is not None and min_v is not None:
            res["dt_us_max_veto_min"].append(max_us - min_v)

        # 4. Latest US - Latest SciFi
        if max_us is not None and max_sf is not None:
            res["dt_us_max_sf_max"].append(max_us - max_sf)

        # 5. Latest DS - Latest Veto
        if max_ds is not None and max_v is not None:
            res["dt_ds_max_veto_max"].append(max_ds - max_v)

        # 6. Latest DS - Earliest SciFi
        if max_ds is not None and min_sf is not None:
            res["dt_ds_max_sf_min"].append(max_ds - min_sf)

        # 7. Latest DS - Earliest US
        if max_ds is not None and min_us is not None:
            res["dt_ds_max_us_min"].append(max_ds - min_us)

        # Reference time differences
        if t_ref_sf is not None and min_v is not None:
            res["dt_ref_scifi_min_veto"].append(t_ref_sf - min_v)
        if t_ref_ds is not None and min_v is not None:
            res["dt_ref_ds_min_veto"].append(t_ref_ds - min_v)
        if t_ref_ds is not None and t_ref_sf is not None:
            res["dt_ref_ds_ref_scifi"].append(t_ref_ds - t_ref_sf)

        # Track propagation direction observables
        if has_tracks and tree.Reco_MuonTracks is not None and tree.Reco_MuonTracks.GetEntriesFast() > 0:
            for tr_idx in range(tree.Reco_MuonTracks.GetEntriesFast()):
                tr = tree.Reco_MuonTracks.At(tr_idx)
                if not tr:
                    continue
                pts = list(tr.getTrackPoints())
                raw_times_list = list(tr.getRawMeasTimes())
                if len(pts) >= 2 and len(raw_times_list) >= 2:
                    t_vals = [list(x)[0] for x in raw_times_list if len(list(x)) > 0]
                    z_vals = [p.Z() for p in pts[:len(t_vals)]]
                    if len(t_vals) >= 2:
                        # Linear regression: t = t0 + (dt/dz) * z
                        # Positive slope (dt/dz > 0) -> particle moves forward (+z)
                        # Negative slope (dt/dz < 0) -> particle moves backward (-z)
                        dz = z_vals[-1] - z_vals[0]
                        dt = t_vals[-1] - t_vals[0]
                        if abs(dz) > 1e-3:
                            slope = dt / dz
                            direction = 1.0 if slope >= 0 else -1.0
                            vel = dz / dt if abs(dt) > 1e-4 else (999.0 if dz > 0 else -999.0)
                            res["track_direction"].append(direction)
                            res["track_velocity"].append(float(vel))
                            res["track_dir_slope"].append(float(slope))

        # Inverted SciFi-DS timing cut metric: t(SciFi_max) - t(DS_min)
        if max_sf is not None and min_ds is not None:
            res["dt_sf_max_ds_min"].append(max_sf - min_ds)

        # Earliest DS - Earliest Veto
        if min_ds is not None and min_v is not None:
            res["dt_ds_min_veto_min"].append(min_ds - min_v)

        # Earliest US - Earliest SciFi
        if min_us is not None and min_sf is not None:
            res["dt_us_min_sf_min"].append(min_us - min_sf)

        # Earliest DS - Earliest SciFi
        if min_ds is not None and min_sf is not None:
            res["dt_ds_min_sf_min"].append(min_ds - min_sf)

    f_in.Close()
    return res


def compute_statistics(arr: np.ndarray) -> Dict[str, float]:
    """Compute standard summary metrics for distribution."""
    if len(arr) == 0:
        return {"entries": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "iqr": 0.0}
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return {
        "entries": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(q50),
        "iqr": float(q75 - q25),
    }


def plot_single_distribution(
    data: np.ndarray,
    title: str,
    xlabel: str,
    output_base: str,
    bins: int = 100,
    x_range: Optional[Tuple[float, float]] = None,
    color: str = "#1f77b4",
    tag: str = "Run 6596 B2noB1 Muons",
    unit_str: str = "ns",
    is_log: bool = False,
) -> None:
    """Plot a clean 1D distribution with statistics box and save PNG/PDF."""
    if len(data) == 0:
        return

    stats = compute_statistics(data)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Auto range if not provided: 1st and 99th percentiles with margin
    if x_range is None:
        p1, p99 = np.percentile(data, [0.5, 99.5])
        margin = max(abs(p99 - p1) * 0.08, 0.5)
        x_min, x_max = p1 - margin, p99 + margin
    else:
        x_min, x_max = x_range

    counts, bin_edges, _ = ax.hist(
        data,
        bins=bins,
        range=(x_min, x_max),
        color=color,
        alpha=0.75,
        edgecolor=color,
        linewidth=1.2,
        histtype="stepfilled",
    )

    ax.set_title(title, fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    bin_width = (x_max - x_min) / bins
    ax.set_ylabel(f"Events / {bin_width:.2f} {unit_str}", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    if is_log:
        ax.set_yscale("log")
        y_max = max(counts) * 2.0 if len(counts) > 0 and max(counts) > 0 else 10
        ax.set_ylim(0.5, y_max)
    else:
        y_max = max(counts) * 1.25 if len(counts) > 0 else 10
        ax.set_ylim(0, y_max)

    # Statistics Box
    stat_text = (
        f"Entries: {stats['entries']:,}\n"
        f"Mean: {stats['mean']:.2f} {unit_str}\n"
        f"Std Dev: {stats['std']:.2f} {unit_str}\n"
        f"Median: {stats['median']:.2f} {unit_str}"
    )
    ax.text(
        0.96, 0.94, stat_text,
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    if HAS_DMS:
        dms.add_label(ax=ax, mainText="SND@LHC", extraText=tag)

    plt.tight_layout()
    fig.savefig(f"{output_base}.png")
    fig.savefig(f"{output_base}.pdf")
    plt.close(fig)


def plot_subsystems_overlay(
    subsystems_dict: Dict[str, Tuple[np.ndarray, str]],
    title: str,
    xlabel: str,
    output_base: str,
    bins: int = 100,
    x_range: Tuple[float, float] = (0.0, 30.0),
    tag: str = "Run 6596 B2noB1 Muons",
    unit_str: str = "ns",
) -> None:
    """Plot multi-subsystem overlay comparison (Veto vs SciFi vs US vs DS)."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    for label, (arr, col) in subsystems_dict.items():
        if len(arr) == 0:
            continue
        ax.hist(
            arr,
            bins=bins,
            range=x_range,
            label=f"{label} (N={len(arr):,}, #mu={np.mean(arr):.1f})",
            color=col,
            alpha=0.35,
            edgecolor=col,
            linewidth=1.8,
            histtype="stepfilled",
            density=True,
        )

    ax.set_title(title, fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Probability Density [1/ns]", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    if HAS_DMS:
        dms.add_label(ax=ax, mainText="SND@LHC", extraText=tag)

    plt.tight_layout()
    fig.savefig(f"{output_base}.png")
    fig.savefig(f"{output_base}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_arguments()
    start_time = time.time()

    input_files = resolve_input_files(args.input, max_files=args.max_files)
    if not input_files:
        print(f"[ERROR] No ROOT files found matching: {args.input}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "figures")
    if args.plots:
        os.makedirs(plots_dir, exist_ok=True)

    print("=" * 80)
    print(" SND@LHC Backward Muon Timing Analysis")
    print("=" * 80)
    print(f" Matched Files   : {len(input_files)}")
    print(f" First File      : {os.path.basename(input_files[0])}")
    print(f" Last File       : {os.path.basename(input_files[-1])}")
    print(f" Units           : {args.units}")
    print(f" Worker Processes: {args.jobs}")
    print(f" Output Dir      : {args.output_dir}")
    print("=" * 80)

    # Prepare worker tasks
    tasks = []
    events_per_file = 0
    if args.max_events > 0:
        events_per_file = max(1, int(np.ceil(args.max_events / len(input_files))))

    for f_path in input_files:
        tasks.append((f_path, events_per_file, args.units, args.geofile))

    # Aggregated storage
    combined: Dict[str, List[float]] = {
        "all_v": [], "all_sf": [], "all_us": [], "all_ds": [],
        "min_v": [], "max_v": [], "min_sf": [], "max_sf": [],
        "min_us": [], "max_us": [], "min_ds": [], "max_ds": [],
        "dt_sf_min_veto_min": [],
        "dt_sf_max_veto_max": [],
        "dt_us_max_veto_min": [],
        "dt_us_max_sf_max": [],
        "dt_ds_max_veto_max": [],
        "dt_ds_max_sf_min": [],
        "dt_ds_max_us_min": [],
        # Reference times and relative differences
        "t_ref_scifi": [],
        "t_ref_ds": [],
        "dt_ref_scifi_min_veto": [],
        "dt_ref_ds_min_veto": [],
        "dt_ref_ds_ref_scifi": [],
        # Track direction observables
        "track_direction": [],
        "track_velocity": [],
        "track_dir_slope": [],
        # Complementary differences
        "dt_sf_max_ds_min": [],
        "dt_ds_min_veto_min": [],
        "dt_us_min_sf_min": [],
        "dt_ds_min_sf_min": [],
    }
    total_events = 0

    # Execute parallel reading
    completed = 0
    if args.jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_process_single_file, t) for t in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                total_events += res["events_processed"]
                for k in combined:
                    combined[k].extend(res[k])
                completed += 1
                if completed % max(1, len(tasks) // 10) == 0 or completed == len(tasks):
                    print(f" [{completed:4d}/{len(tasks)}] Processed ({total_events:,} events extracted so far)...")
    else:
        for t in tasks:
            res = _process_single_file(t)
            total_events += res["events_processed"]
            for k in combined:
                combined[k].extend(res[k])
            completed += 1
            if completed % max(1, len(tasks) // 10) == 0 or completed == len(tasks):
                print(f" [{completed:4d}/{len(tasks)}] Processed ({total_events:,} events extracted so far)...")

    elapsed = time.time() - start_time
    print(f"\nExtracted timing from {total_events:,} events across {len(input_files)} files in {elapsed:.1f}s.")
    print("=" * 80)

    # Convert to numpy arrays
    np_data: Dict[str, np.ndarray] = {k: np.array(v, dtype=np.float32) for k, v in combined.items()}

    u_str = "ns" if args.units == "ns" else "clock ticks"

    # Tabulate and print summary statistics
    print("\nTIMING DISTRIBUTIONS SUMMARY:")
    print(f"{'Observable':<35} | {'Entries':>10} | {'Mean ['+u_str+']':>12} | {'StdDev ['+u_str+']':>12} | {'Median':>10}")
    print("-" * 90)

    metrics_list = [
        ("all_v", "Veto Hit Times (All)"),
        ("all_sf", "SciFi Hit Times (All)"),
        ("all_us", "US Hit Times (All)"),
        ("all_ds", "DS Hit Times (All)"),
        ("min_v", "Earliest Veto Hit per Event"),
        ("max_v", "Latest Veto Hit per Event"),
        ("min_sf", "Earliest SciFi Hit per Event"),
        ("max_sf", "Latest SciFi Hit per Event"),
        ("min_us", "Earliest US Hit per Event"),
        ("max_us", "Latest US Hit per Event"),
        ("min_ds", "Earliest DS Hit per Event"),
        ("max_ds", "Latest DS Hit per Event"),
        ("dt_sf_min_veto_min", "Earliest SciFi - Earliest Veto"),
        ("dt_sf_max_veto_max", "Latest SciFi - Latest Veto"),
        ("dt_us_max_veto_min", "Latest US - Earliest Veto"),
        ("dt_us_max_sf_max", "Latest US - Latest SciFi"),
        ("dt_ds_max_veto_max", "Latest DS - Latest Veto"),
        ("dt_ds_max_sf_min", "Latest DS - Earliest SciFi"),
        ("dt_ds_max_us_min", "Latest DS - Earliest US"),
        ("t_ref_scifi", "Reference SciFi Time (Horiz Avg)"),
        ("t_ref_ds", "Reference DS Time (Horiz SiPMs Avg)"),
        ("dt_ref_scifi_min_veto", "Ref SciFi - Earliest Veto"),
        ("dt_ref_ds_min_veto", "Ref DS - Earliest Veto"),
        ("dt_ref_ds_ref_scifi", "Ref DS - Ref SciFi"),
        ("track_direction", "Track Direction (+1=down, -1=up)"),
        ("track_velocity", "Track Velocity [cm/ns]"),
        ("track_dir_slope", "TrackDir Fit Slope [ns/cm]"),
        ("dt_sf_max_ds_min", "Latest SciFi - Earliest DS (Cut)"),
        ("dt_ds_min_veto_min", "Earliest DS - Earliest Veto"),
        ("dt_us_min_sf_min", "Earliest US - Earliest SciFi"),
        ("dt_ds_min_sf_min", "Earliest DS - Earliest SciFi"),
    ]

    for key, desc in metrics_list:
        st = compute_statistics(np_data[key])
        print(f"{desc:<35} | {st['entries']:>10,d} | {st['mean']:>12.2f} | {st['std']:>12.2f} | {st['median']:>10.2f}")
    print("-" * 90)

    # -------------------------------------------------------------------------
    # Generate Matplotlib Figures (only if --plots is set)
    # -------------------------------------------------------------------------
    if args.plots:
        print("\nGenerating publication figures (PNG and PDF)...")

        # 1. Absolute Hit Times per Subsystem
        plot_single_distribution(
            np_data["all_v"], "Veto Hit Times Distribution", f"Veto Hit Time [{u_str}]",
            os.path.join(plots_dir, "time_veto_all"), color="#d62728", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["all_sf"], "SciFi Hit Times Distribution", f"SciFi Hit Time [{u_str}]",
            os.path.join(plots_dir, "time_scifi_all"), color="#1f77b4", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["all_us"], "Upstream MuFilter (US) Hit Times Distribution", f"US Hit Time [{u_str}]",
            os.path.join(plots_dir, "time_us_all"), color="#2ca02c", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["all_ds"], "Downstream MuFilter (DS) Hit Times Distribution", f"DS Hit Time [{u_str}]",
            os.path.join(plots_dir, "time_ds_all"), color="#9467bd", tag=args.tag, unit_str=u_str,
        )

        # 2. Earliest and Latest Extrema
        plot_single_distribution(
            np_data["min_v"], "Earliest Veto Hit Time per Event", f"Earliest Veto Time [{u_str}]",
            os.path.join(plots_dir, "time_veto_earliest"), color="#d62728", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["max_v"], "Latest Veto Hit Time per Event", f"Latest Veto Time [{u_str}]",
            os.path.join(plots_dir, "time_veto_latest"), color="#b2182b", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["min_sf"], "Earliest SciFi Hit Time per Event", f"Earliest SciFi Time [{u_str}]",
            os.path.join(plots_dir, "time_scifi_earliest"), color="#1f77b4", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["max_sf"], "Latest SciFi Hit Time per Event", f"Latest SciFi Time [{u_str}]",
            os.path.join(plots_dir, "time_scifi_latest"), color="#2166ac", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["min_us"], "Earliest US Hit Time per Event", f"Earliest US Time [{u_str}]",
            os.path.join(plots_dir, "time_us_earliest"), color="#2ca02c", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["max_us"], "Latest US Hit Time per Event", f"Latest US Time [{u_str}]",
            os.path.join(plots_dir, "time_us_latest"), color="#1b7837", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["min_ds"], "Earliest DS Hit Time per Event", f"Earliest DS Time [{u_str}]",
            os.path.join(plots_dir, "time_ds_earliest"), color="#9467bd", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["max_ds"], "Latest DS Hit Time per Event", f"Latest DS Time [{u_str}]",
            os.path.join(plots_dir, "time_ds_latest"), color="#762a83", tag=args.tag, unit_str=u_str,
        )

        # 3. Subsystem Overlays
        subsys_palette = {
            "Veto": (np_data["all_v"], "#d62728"),
            "SciFi": (np_data["all_sf"], "#1f77b4"),
            "US": (np_data["all_us"], "#2ca02c"),
            "DS": (np_data["all_ds"], "#9467bd"),
        }
        plot_subsystems_overlay(
            subsys_palette, "Normalized Subsystem Hit Times Comparison",
            f"Hit Time [{u_str}]", os.path.join(plots_dir, "overlay_subsystems_all_hits"),
            tag=args.tag, unit_str=u_str,
        )

        subsys_earliest_palette = {
            "Earliest Veto": (np_data["min_v"], "#d62728"),
            "Earliest SciFi": (np_data["min_sf"], "#1f77b4"),
            "Earliest US": (np_data["min_us"], "#2ca02c"),
            "Earliest DS": (np_data["min_ds"], "#9467bd"),
        }
        plot_subsystems_overlay(
            subsys_earliest_palette, "Normalized Subsystem Earliest Hit Times Comparison",
            f"Earliest Hit Time [{u_str}]", os.path.join(plots_dir, "overlay_subsystems_earliest"),
            tag=args.tag, unit_str=u_str,
        )

        # 4. User-Requested Relative Time Differences
        plot_single_distribution(
            np_data["dt_sf_min_veto_min"],
            "Earliest SciFi Hit Time - Earliest Veto Hit Time",
            f"t_{{SciFi}}^{{min}} - t_{{Veto}}^{{min}} [{u_str}]",
            os.path.join(plots_dir, "dt_scifi_min_minus_veto_min"),
            color="#2b83ba", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["dt_sf_max_veto_max"],
            "Latest SciFi Hit Time - Latest Veto Hit Time",
            f"t_{{SciFi}}^{{max}} - t_{{Veto}}^{{max}} [{u_str}]",
            os.path.join(plots_dir, "dt_scifi_max_minus_veto_max"),
            color="#41b6c4", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["dt_us_max_veto_min"],
            "Latest US Hit Time - Earliest Veto Hit Time",
            f"t_{{US}}^{{max}} - t_{{Veto}}^{{min}} [{u_str}]",
            os.path.join(plots_dir, "dt_us_max_minus_veto_min"),
            color="#74add1", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["dt_us_max_sf_max"],
            "Latest US Hit Time - Latest SciFi Hit Time",
            f"t_{{US}}^{{max}} - t_{{SciFi}}^{{max}} [{u_str}]",
            os.path.join(plots_dir, "dt_us_max_minus_scifi_max"),
            color="#fdae61", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["dt_ds_max_veto_max"],
            "Latest DS Hit Time - Latest Veto Hit Time",
            f"t_{{DS}}^{{max}} - t_{{Veto}}^{{max}} [{u_str}]",
            os.path.join(plots_dir, "dt_ds_max_minus_veto_max"),
            color="#8073ac", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["dt_ds_max_sf_min"],
            "Latest DS Hit Time - Earliest SciFi Hit Time",
            f"t_{{DS}}^{{max}} - t_{{SciFi}}^{{min}} [{u_str}]",
            os.path.join(plots_dir, "dt_ds_max_minus_scifi_min"),
            color="#e08214", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["dt_ds_max_us_min"],
            "Latest DS Hit Time - Earliest US Hit Time",
            f"t_{{DS}}^{{max}} - t_{{US}}^{{min}} [{u_str}]",
            os.path.join(plots_dir, "dt_ds_max_minus_us_min"),
            color="#b2182b", tag=args.tag, unit_str=u_str,
        )

        # Reference times and relative differences
        plot_single_distribution(
            np_data["dt_ref_scifi_min_veto"],
            "Reference SciFi Time - Earliest Veto Hit Time",
            f"t_{{SciFi}}^{{ref}} - t_{{Veto}}^{{min}} [{u_str}]",
            os.path.join(plots_dir, "dt_ref_scifi_minus_veto_min"),
            color="#2ca02c", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["dt_ref_ds_min_veto"],
            "Reference DS Time - Earliest Veto Hit Time",
            f"t_{{DS}}^{{ref}} - t_{{Veto}}^{{min}} [{u_str}]",
            os.path.join(plots_dir, "dt_ref_ds_minus_veto_min"),
            color="#762a83", tag=args.tag, unit_str=u_str,
        )
        plot_single_distribution(
            np_data["dt_ref_ds_ref_scifi"],
            "Reference DS Time - Reference SciFi Time",
            f"t_{{DS}}^{{ref}} - t_{{SciFi}}^{{ref}} [{u_str}]",
            os.path.join(plots_dir, "dt_ref_ds_minus_ref_scifi"),
            color="#d95f02", tag=args.tag, unit_str=u_str,
        )

        # 5. Inverted SciFi-DS Timing Cut Observable
        plot_single_distribution(
            np_data["dt_sf_max_ds_min"],
            "Latest SciFi Hit Time - Earliest DS Hit Time (Inverted SciFi-DS Cut)",
            f"t_{{SciFi}}^{{max}} - t_{{DS}}^{{min}} [{u_str}]",
            os.path.join(plots_dir, "dt_scifi_max_minus_ds_min"),
            color="#006837", tag=args.tag, unit_str=u_str,
        )

        # 6. Track Direction Observables
        plot_single_distribution(
            np_data["track_direction"],
            "Reco Muon Track Propagation Direction (+1=downstream, -1=upstream)",
            "Track Direction (+1 / -1)",
            os.path.join(plots_dir, "reco_track_direction"),
            bins=5, x_range=(-2.5, 2.5), color="#386cb0", tag=args.tag, unit_str="",
        )
        plot_single_distribution(
            np_data["track_velocity"],
            "Reco Muon Track Velocity (dL/dT)",
            "Velocity [cm/ns]",
            os.path.join(plots_dir, "reco_track_velocity"),
            color="#f0027f", tag=args.tag, unit_str="cm/ns",
        )
        plot_single_distribution(
            np_data["track_dir_slope"],
            "Reco Muon Track Dir Slope (Time of Flight Fit)",
            "Slope [ns/cm]",
            os.path.join(plots_dir, "reco_track_dir_slope"),
            color="#bf5b17", tag=args.tag, unit_str="ns/cm",
        )

    # -------------------------------------------------------------------------
    # Save ROOT Output File with TH1D and Canvases
    # -------------------------------------------------------------------------
    root_out_path = os.path.join(args.output_dir, "backward_muon_timing.root")
    print(f"\nSaving ROOT file with histograms and canvases: {root_out_path} ...")
    f_root = ROOT.TFile.Open(root_out_path, "RECREATE")

    dir_abs = f_root.mkdir("absolute_times")
    dir_rel = f_root.mkdir("relative_times")
    dir_ref = f_root.mkdir("reference_times")
    dir_trk = f_root.mkdir("track_direction")

    # Helper function for filling TH1D
    def book_and_fill(target_dir, name, title, arr, nbins=120, xmin=None, xmax=None):
        target_dir.cd()
        if len(arr) == 0:
            h = ROOT.TH1D(name, title, nbins, -50.0, 50.0)
            h.Write()
            return h
        if xmin is None or xmax is None:
            p1, p99 = np.percentile(arr, [0.5, 99.5])
            span = max(abs(p99 - p1) * 0.1, 0.5)
            xmin = float(p1 - span)
            xmax = float(p99 + span)
        h = ROOT.TH1D(name, title, nbins, xmin, xmax)
        for val in arr:
            h.Fill(float(val))
        h.Write()
        return h

    # Book ROOT histograms
    book_and_fill(dir_abs, "h_time_veto_all", f"All Veto Hit Times;Time [{u_str}];Hits", np_data["all_v"], 120, 0, 30)
    book_and_fill(dir_abs, "h_time_scifi_all", f"All SciFi Hit Times;Time [{u_str}];Hits", np_data["all_sf"], 120, 0, 30)
    book_and_fill(dir_abs, "h_time_us_all", f"All US Hit Times;Time [{u_str}];Hits", np_data["all_us"], 120, 0, 30)
    book_and_fill(dir_abs, "h_time_ds_all", f"All DS Hit Times;Time [{u_str}];Hits", np_data["all_ds"], 120, 0, 30)

    book_and_fill(dir_abs, "h_time_veto_earliest", f"Earliest Veto Hit Time per Event;Time [{u_str}];Events", np_data["min_v"], 120, 0, 30)
    book_and_fill(dir_abs, "h_time_veto_latest", f"Latest Veto Hit Time per Event;Time [{u_str}];Events", np_data["max_v"], 120, 0, 30)
    book_and_fill(dir_abs, "h_time_scifi_earliest", f"Earliest SciFi Hit Time per Event;Time [{u_str}];Events", np_data["min_sf"], 120, 0, 30)
    book_and_fill(dir_abs, "h_time_scifi_latest", f"Latest SciFi Hit Time per Event;Time [{u_str}];Events", np_data["max_sf"], 120, 0, 30)
    book_and_fill(dir_abs, "h_time_us_earliest", f"Earliest US Hit Time per Event;Time [{u_str}];Events", np_data["min_us"], 120, 0, 30)
    book_and_fill(dir_abs, "h_time_us_latest", f"Latest US Hit Time per Event;Time [{u_str}];Events", np_data["max_us"], 120, 0, 30)
    book_and_fill(dir_abs, "h_time_ds_earliest", f"Earliest DS Hit Time per Event;Time [{u_str}];Events", np_data["min_ds"], 120, 0, 30)
    book_and_fill(dir_abs, "h_time_ds_latest", f"Latest DS Hit Time per Event;Time [{u_str}];Events", np_data["max_ds"], 120, 0, 30)

    book_and_fill(dir_rel, "h_dt_scifi_min_veto_min", f"t(SciFi min) - t(Veto min);#Delta t [{u_str}];Events", np_data["dt_sf_min_veto_min"])
    book_and_fill(dir_rel, "h_dt_scifi_max_veto_max", f"t(SciFi max) - t(Veto max);#Delta t [{u_str}];Events", np_data["dt_sf_max_veto_max"])
    book_and_fill(dir_rel, "h_dt_us_max_veto_min", f"t(US max) - t(Veto min);#Delta t [{u_str}];Events", np_data["dt_us_max_veto_min"])
    book_and_fill(dir_rel, "h_dt_us_max_scifi_max", f"t(US max) - t(SciFi max);#Delta t [{u_str}];Events", np_data["dt_us_max_sf_max"])
    book_and_fill(dir_rel, "h_dt_ds_max_veto_max", f"t(DS max) - t(Veto max);#Delta t [{u_str}];Events", np_data["dt_ds_max_veto_max"])
    book_and_fill(dir_rel, "h_dt_ds_max_scifi_min", f"t(DS max) - t(SciFi min);#Delta t [{u_str}];Events", np_data["dt_ds_max_sf_min"])
    book_and_fill(dir_rel, "h_dt_ds_max_us_min", f"t(DS max) - t(US min);#Delta t [{u_str}];Events", np_data["dt_ds_max_us_min"])
    book_and_fill(dir_rel, "h_dt_scifi_max_ds_min", f"t(SciFi max) - t(DS min) [Inverted Cut];#Delta t [{u_str}];Events", np_data["dt_sf_max_ds_min"])

    # Reference times and relative differences
    book_and_fill(dir_ref, "h_t_ref_scifi", f"Reference SciFi Time (Horizontal Planes Average);Time [{u_str}];Events", np_data["t_ref_scifi"], 120, 0, 30)
    book_and_fill(dir_ref, "h_t_ref_ds", f"Reference DS Time (Horizontal Planes SiPMs Average);Time [{u_str}];Events", np_data["t_ref_ds"], 120, 0, 30)
    book_and_fill(dir_ref, "h_dt_ref_scifi_veto_min", f"t(SciFi ref) - t(Veto min);#Delta t [{u_str}];Events", np_data["dt_ref_scifi_min_veto"])
    book_and_fill(dir_ref, "h_dt_ref_ds_veto_min", f"t(DS ref) - t(Veto min);#Delta t [{u_str}];Events", np_data["dt_ref_ds_min_veto"])
    book_and_fill(dir_ref, "h_dt_ref_ds_ref_scifi", f"t(DS ref) - t(SciFi ref);#Delta t [{u_str}];Events", np_data["dt_ref_ds_ref_scifi"])

    # Track direction histograms
    book_and_fill(dir_trk, "h_track_direction", "Reco Muon Track Direction (+1=downstream, -1=upstream);Direction;Tracks", np_data["track_direction"], 5, -2.5, 2.5)
    book_and_fill(dir_trk, "h_track_velocity", "Reco Muon Track Velocity;v [cm/ns];Tracks", np_data["track_velocity"], 100, -200, 200)
    book_and_fill(dir_trk, "h_track_dir_slope", "Reco Muon Track Dir Slope;Slope [ns/cm];Tracks", np_data["track_dir_slope"], 100, -0.1, 0.1)

    f_root.Close()

    total_time = time.time() - start_time
    print("=" * 80)
    print(f" Analysis completed in {total_time:.1f} seconds!")
    print(f" Figures Directory: {os.path.abspath(plots_dir)}")
    print(f" ROOT File        : {os.path.abspath(root_out_path)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
