#!/usr/bin/env python3
"""
merge_hough_tuning_csv.py
-------------------------
Gathers, merges, sorts, and visualizes distributed HTCondor Hough parameter tuning CSV outputs.

Usage:
  python3 merge_hough_tuning_csv.py \\
    --input-dir /eos/user/i/idioniso/snd-numu-charm/plots/hough_tuning/jobs \\
    --output-csv /eos/user/i/idioniso/snd-numu-charm/plots/hough_tuning/hough_fom_pairwise_results.csv \\
    --sort-by asimov_z \\
    --plot
"""

from __future__ import annotations

import os
import sys
import glob
import csv
import argparse
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from pgun.scripts.tune_dimuon_hough import print_scoreboard, plot_tuning_summary
except ImportError:
    print_scoreboard = None
    plot_tuning_summary = None


def parse_args():
    parser = argparse.ArgumentParser(description="Merge distributed HTCondor Hough parameter tuning results.")
    parser.add_argument(
        "-i", "--input-dir",
        default="/eos/user/i/idioniso/snd-numu-charm/plots/hough_tuning/jobs",
        help="Directory containing per-job CSV outputs (hough_tuning_job_*.csv)",
    )
    parser.add_argument(
        "-o", "--output-csv",
        default="/eos/user/i/idioniso/snd-numu-charm/plots/hough_tuning/hough_fom_pairwise_results.csv",
        help="Destination path for merged CSV file",
    )
    parser.add_argument(
        "--output-pickle",
        default=None,
        help="Optional pickle output destination for reconstructed result dictionary objects",
    )
    parser.add_argument(
        "--sort-by",
        default="asimov_z",
        choices=["asimov_z", "score_mean", "score_genie_genie", "score_genie_pmu", "score_pgun_genie", "score_pgun_pmu", "TPR_genie"],
        help="Metric to rank combined results by (default: asimov_z)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of top configurations to display on scoreboard (default: 25)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate response landscape summary plots from merged results",
    )
    return parser.parse_args()


def load_and_merge_csvs(input_dir: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    pattern = os.path.join(input_dir, "hough_tuning_job_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"ERROR: No CSV files found matching '{pattern}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} job CSV files in {input_dir}.")

    all_rows: List[Dict[str, Any]] = []
    fieldnames: Optional[List[str]] = None
    seen_signatures = set()

    float_keys = {
        "max_angle", "max_opening_angle", "binning_factor", "tolerance", "sigma", "truncate",
        "delta_zCross_max", "zCross_min", "zCross_max", "delta_zDS3_max", "z_ds3",
        "s_genie", "b_genie", "b_pmu", "z_asimov_genie", "z_asimov_pmu", "z_asimov_comb",
        "TPR_genie", "TPR_pgun", "FPR_genie", "FPR_pmu",
        "score_genie_genie", "score_genie_pmu", "score_pgun_genie", "score_pgun_pmu", "score_mean"
    }
    int_keys = {"n_xH", "n_yH"}

    for fpath in csv_files:
        if os.path.getsize(fpath) == 0:
            continue
        try:
            with open(fpath, "r", newline="") as f:
                reader = csv.DictReader(f)
                if fieldnames is None and reader.fieldnames:
                    fieldnames = list(reader.fieldnames)

                for row in reader:
                    # Deduplicate identical configurations if jobs overlapped
                    sig = (
                        row.get("format"),
                        row.get("max_angle"),
                        row.get("max_opening_angle"),
                        row.get("binning_factor"),
                        row.get("tolerance"),
                        row.get("smooth_full"),
                        row.get("sigma"),
                        row.get("truncate"),
                        row.get("delta_zCross_max"),
                        row.get("zCross_min"),
                        row.get("zCross_max"),
                        row.get("delta_zDS3_max"),
                    )
                    if sig in seen_signatures:
                        continue
                    seen_signatures.add(sig)

                    # Cast typed values
                    typed_row: Dict[str, Any] = {}
                    for k, v in row.items():
                        if v is None or v == "":
                            typed_row[k] = None
                        elif k in float_keys:
                            try:
                                typed_row[k] = float(v)
                            except ValueError:
                                typed_row[k] = None
                        elif k in int_keys:
                            try:
                                typed_row[k] = int(float(v))
                            except ValueError:
                                typed_row[k] = None
                        elif k == "smooth_full":
                            typed_row[k] = (v.lower() == "true")
                        else:
                            typed_row[k] = v

                    all_rows.append(typed_row)
        except Exception as e:
            print(f"Warning: Failed reading {fpath}: {e}", file=sys.stderr)

    return all_rows, (fieldnames or [])


def main():
    args = parse_args()
    rows, fieldnames = load_and_merge_csvs(args.input_dir)

    if not rows:
        print("ERROR: No valid evaluation rows loaded from CSV files.", file=sys.stderr)
        sys.exit(2)

    # Sort key mapping
    sort_key_map = {
        "asimov_z": "z_asimov_genie",
        "score_mean": "score_mean",
        "score_genie_genie": "score_genie_genie",
        "score_genie_pmu": "score_genie_pmu",
        "score_pgun_genie": "score_pgun_genie",
        "score_pgun_pmu": "score_pgun_pmu",
        "TPR_genie": "TPR_genie",
    }
    target_metric = sort_key_map.get(args.sort_by, "z_asimov_genie")

    rows.sort(key=lambda r: float(r.get(target_metric) or -999.0), reverse=True)

    # Save consolidated CSV
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSuccessfully wrote {len(rows)} merged configurations to:\n  -> {args.output_csv}")

    # Automatically save high-performance Parquet format alongside CSV
    parquet_path = os.path.splitext(args.output_csv)[0] + ".parquet"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pa_table = pa.Table.from_pylist(rows)
        pq.write_table(pa_table, parquet_path, compression="zstd")
        print(f"Successfully wrote fast Parquet file to:\n  -> {parquet_path}")
    except Exception as e:
        print(f"Notice: Parquet export skipped ({e}).")

    # Build standard dictionary objects for scoreboard and plotting
    result_objects = []
    for r in rows:
        obj = {
            "params": {
                "h_format": r.get("format"),
                "binning_factor": r.get("binning_factor"),
                "tolerance": r.get("tolerance"),
                "max_angle": r.get("max_angle"),
                "max_opening_angle": r.get("max_opening_angle"),
                "smooth_full": r.get("smooth_full"),
                "sigma": r.get("sigma"),
                "truncate": r.get("truncate"),
                "delta_zCross_max": r.get("delta_zCross_max"),
                "zCross_min": r.get("zCross_min"),
                "zCross_max": r.get("zCross_max"),
                "delta_zDS3_max": r.get("delta_zDS3_max"),
                "z_ds3": r.get("z_ds3"),
                "n_xH": r.get("n_xH"),
                "n_yH": r.get("n_yH"),
            },
            "s_genie": r.get("s_genie", 0.0),
            "s_pgun": r.get("s_pgun", 0.0),
            "b_genie": r.get("b_genie", 0.0),
            "b_pmu": r.get("b_pmu", 0.0),
            "z_asimov_genie": r.get("z_asimov_genie", 0.0),
            "z_asimov_pmu": r.get("z_asimov_pmu", 0.0),
            "z_asimov_comb": r.get("z_asimov_comb", 0.0),
            "tpr_genie": r.get("TPR_genie", 0.0),
            "tpr_pgun": r.get("TPR_pgun", 0.0),
            "fpr_genie": r.get("FPR_genie", 0.0),
            "fpr_pmu": r.get("FPR_pmu", 0.0),
            "score_genie_genie": r.get("score_genie_genie", 0.0),
            "score_genie_pmu": r.get("score_genie_pmu", 0.0),
            "score_pgun_genie": r.get("score_pgun_genie", 0.0),
            "score_pgun_pmu": r.get("score_pgun_pmu", 0.0),
            "score_mean": r.get("score_mean", 0.0),
        }
        result_objects.append(obj)

    if args.output_pickle:
        import pickle
        p_dir = os.path.dirname(os.path.abspath(args.output_pickle))
        if p_dir:
            os.makedirs(p_dir, exist_ok=True)
        with open(args.output_pickle, "wb") as pf:
            pickle.dump(result_objects, pf)
        print(f"Saved merged pickle results to:\n  -> {args.output_pickle}")

    # Display Top-25 Scoreboard
    if print_scoreboard is not None:
        print_scoreboard(result_objects, sort_key=target_metric, top_k=args.top_k)

    # Display optimal parameters
    best = result_objects[0]
    bp = best["params"]
    print("=" * 70)
    print(" >>> OPTIMAL HOUGH CONFIGURATION (FROM BATCH GRID SEARCH) <<<")
    print(f"  Format:          {bp['h_format']}")
    print(f"  Binning Factor:  {bp['binning_factor']}")
    print(f"  Accumulator Res: {bp['n_xH']} x {bp['n_yH']} bins")
    print(f"  Tolerance:       {bp['tolerance']} cm")
    print(f"  Smooth Full:     {bp['smooth_full']} (sigma={bp['sigma']}, truncate={bp['truncate']})")
    print(f"  Max Angle:       {bp['max_angle']} rad")
    if bp.get("max_opening_angle") is not None:
        open_str = f"{bp['max_opening_angle']} rad" if float(bp['max_opening_angle']) < 900.0 else "unconstrained"
        print(f"  Max Open Angle:  {open_str}")
    if bp.get("delta_zCross_max") is not None:
        print(f"  delta_zCross_max:{bp['delta_zCross_max']} cm")
    if bp.get("zCross_max") is not None:
        print(f"  zCross Range:    [{bp.get('zCross_min', 200.0)}, {bp['zCross_max']}] cm")
    if bp.get("delta_zDS3_max") is not None:
        print(f"  delta_zDS3_max:  {bp['delta_zDS3_max']} cm (at z={bp.get('z_ds3', 543.0)} cm)")
    print("-" * 70)
    print(f"  Signal s (Genie):       {best.get('s_genie', 0):.1f}")
    print(f"  Background B (Genie):   {best.get('b_genie', 0):.1f}")
    print(f"  Background B (pmu):     {best.get('b_pmu', 0):.1f}")
    print(f"  Asimov Z (Genie vs G):  {best.get('z_asimov_genie', 0.0):.4f}")
    print(f"  Asimov Z (Genie vs C):  {best.get('z_asimov_comb', 0.0):.4f}")
    print(f"  Genie Signal TPR:       {best['tpr_genie']*100:.2f}%")
    print(f"  Genie Background FPR:   {best['fpr_genie']*100:.2f}% (Rejection: {(1-best['fpr_genie'])*100:.2f}%)")
    print(f"  Score (Genie vs Genie): {best['score_genie_genie']:.4f}")
    print(f"  Score Mean (All 4):     {best['score_mean']:.4f}")
    print("=" * 70 + "\n")

    if args.plot and plot_tuning_summary is not None:
        plot_tuning_summary(result_objects)


if __name__ == "__main__":
    main()
