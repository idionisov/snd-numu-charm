#!/usr/bin/env python3
"""
scripts/extract_simulation_truth.py
------------------------------------
Master pipeline script to process SND@LHC Monte Carlo simulations:
1. Computes and attaches hierarchical truth branches for ALL events into:
   <output_truth_pattern> (default: .../data/%s/sndLHC.Genie-TGeant4_digCPP_truth.root)
2. Automatically creates symlinks to all ROOT files from the original input partition directory
   into the destination directory.
3. Simultaneously extracts and stores only signal events (nu_mu CC charm dimuon with prompt decay muon
   in DS acceptance) along with the truth branches into:
   <output_signal_pattern> (default: .../data/%s/sndLHC.Genie-TGeant4_digCPP_truth_numuCC_charm_dimuon.root)
Both outputs are produced concurrently in a single pass per file with zero duplicate processing.

Usage examples:
  # Process partitions 0 through 5:
  python3 scripts/extract_simulation_truth.py -p 0-5

  # Process single partition 0 (quick test with 100 events):
  python3 scripts/extract_simulation_truth.py -p 0 -n 100

  # Process all partitions (0 to 400):
  python3 scripts/extract_simulation_truth.py -p 0-400
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import re
from typing import List, Optional

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ROOT
from snd import (
    load_trident_libraries,
    load_config,
    build_processor,
    extract_captures,
    resolve_input_files,
    process_simulation_file_dual_truth,
)


def parse_partitions(part_arg: Optional[str]) -> Optional[List[int]]:
    """
    Parse partition specification string:
      - '0-400' -> [0, 1, ..., 400]
      - '0,1,2,5' -> [0, 1, 2, 5]
      - '0-5,10,12-14' -> [0,1,2,3,4,5,10,12,13,14]
      - 'all' or None -> None (auto-discover all)
    """
    if not part_arg or part_arg.strip().lower() == "all":
        return None

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


def format_partition_path(pattern: str, partition: str | int) -> str:
    """Replace '%s' placeholders in pattern with partition string."""
    p_str = str(partition)
    count = pattern.count("%s")
    if count > 0:
        return pattern % tuple(p_str for _ in range(count))
    return pattern


def main():
    parser = argparse.ArgumentParser(
        description="Master truth extraction and signal skimming pipeline for SND@LHC MC simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    DEFAULT_INPUT = (
        "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/"
        "sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/%s/sndLHC.Genie-TGeant4_20240126_digCPP.root"
    )
    DEFAULT_OUT_TRUTH = (
        "/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_truth.root"
    )
    DEFAULT_OUT_SIGNAL = (
        "/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_truth_numuCC_charm_dimuon.root"
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Input simulation file pattern with '%%s' partition placeholder",
    )
    parser.add_argument(
        "-p", "--partitions",
        type=str,
        default=None,
        help="Partitions to process: range ('0-400'), comma-separated ('0,1,2'), or 'all' to auto-detect",
    )
    parser.add_argument(
        "-o-truth", "--output-truth",
        type=str,
        default=DEFAULT_OUT_TRUTH,
        help="Output ROOT file pattern for ALL events with truth branches",
    )
    parser.add_argument(
        "-o-signal", "--output-signal",
        type=str,
        default=DEFAULT_OUT_SIGNAL,
        help="Output ROOT file pattern for SIGNAL events (nu_mu CC charm dimuon in DS acceptance)",
    )
    parser.add_argument(
        "-n", "--entries",
        type=int,
        default=-1,
        help="Max events per file to process (-1 for all events)",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        default=False,
        help="Disable automatic symlink creation for input directory ROOT files",
    )
    parser.add_argument(
        "--keep-empty-signal",
        action="store_true",
        default=False,
        help="Keep signal ROOT file even if 0 signal events matched in partition",
    )

    args = parser.parse_args()

    # Preload C++ libraries and dictionaries
    load_trident_libraries()
    cfg = load_config(args.config)
    processor = build_processor(cfg.get("processor", {}))

    # Resolve partition list and input files
    selected_partitions = parse_partitions(args.partitions)
    tasks = []

    if selected_partitions is not None:
        for p in selected_partitions:
            in_file = format_partition_path(args.input, p)
            if os.path.exists(in_file):
                out_truth = format_partition_path(args.output_truth, p)
                out_sig = format_partition_path(args.output_signal, p)
                tasks.append((str(p), in_file, out_truth, out_sig))
            else:
                print(f"[Warning] Partition {p} input file not found: {in_file}")
    else:
        # Auto-discover matching files from input pattern
        matched_files = resolve_input_files(args.input)
        for in_file in matched_files:
            captures = extract_captures(in_file, args.input)
            part_str = captures[0] if captures else "0"
            out_truth = format_partition_path(args.output_truth, part_str)
            out_sig = format_partition_path(args.output_signal, part_str)
            tasks.append((part_str, in_file, out_truth, out_sig))

    if not tasks:
        print(f"[Error] No files found matching input pattern: {args.input}")
        sys.exit(1)

    print("=" * 80)
    print(" SND@LHC: Master Simulation Truth Extraction & Signal Filter")
    print("=" * 80)
    print(f" Input Pattern        : {args.input}")
    print(f" Partitions Matched   : {len(tasks)}")
    print(f" Output All Truth     : {args.output_truth}")
    print(f" Output Signal Truth  : {args.output_signal}")
    print(f" Max Entries/File     : {args.entries if args.entries > 0 else 'All'}")
    print(f" Create Symlinks      : {not args.no_symlinks}")
    print(f" Keep Empty Signal    : {args.keep_empty_signal}")
    print("=" * 80)

    start_time = time.time()
    results = []
    tot_events = 0
    tot_signal = 0

    for idx, (part_str, in_file, out_truth, out_sig) in enumerate(tasks, start=1):
        print(f"\n[{idx}/{len(tasks)}] Processing Partition: {part_str}")
        print(f"  Input : {in_file}")
        print(f"  Truth : {out_truth}")
        print(f"  Signal: {out_sig}")

        t0 = time.time()
        res = process_simulation_file_dual_truth(
            input_file=in_file,
            truth_output_file=out_truth,
            signal_output_file=out_sig,
            processor=processor,
            cfg=cfg,
            max_entries=args.entries,
            skip_empty_signal=not args.keep_empty_signal,
            create_symlinks=not args.no_symlinks,
        )
        elapsed = time.time() - t0

        n_tot = res.get("total", 0)
        n_proc = res.get("processed", 0)
        n_sig = res.get("signal", 0)
        tot_events += n_proc
        tot_signal += n_sig
        results.append(res)

        sig_pct = (100.0 * n_sig / max(n_proc, 1)) if n_proc > 0 else 0.0
        print(f"  Result: {n_proc}/{n_tot} processed -> {n_sig} signal ({sig_pct:.2f}%) in {elapsed:.1f}s")
        if res.get("symlinks_created", 0) > 0:
            print(f"  Symlinks: {res['symlinks_created']} file(s) linked in destination directory")

    total_elapsed = time.time() - start_time
    overall_sig_pct = (100.0 * tot_signal / max(tot_events, 1)) if tot_events > 0 else 0.0

    print("\n" + "=" * 80)
    print(" MASTER SIMULATION TRUTH EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"{'Partition':<10} | {'Processed':>10} | {'Signal Evts':>12} | {'Signal Yield':>14} | {'Status':<10}")
    print("-" * 80)
    for (part_str, _, _, _), res in zip(tasks, results):
        p_cnt = res.get("processed", 0)
        s_cnt = res.get("signal", 0)
        y_pct = (100.0 * s_cnt / max(p_cnt, 1)) if p_cnt > 0 else 0.0
        st = res.get("status", "unknown")
        print(f"{part_str:<10} | {p_cnt:>10} | {s_cnt:>12} | {y_pct:>13.2f}% | {st:<10}")

    print("=" * 80)
    print(f" Total Files Processed : {len(tasks)}")
    print(f" Total Events Read     : {tot_events}")
    print(f" Total Signal Selected : {tot_signal} ({overall_sig_pct:.2f}%)")
    print(f" Total Elapsed Time    : {total_elapsed:.1f} seconds")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
