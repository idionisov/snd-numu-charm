#!/usr/bin/env python3
"""
scripts/extract_charm_event_displays.py
---------------------------------------
Extract 2D interactive event displays for signal (charmDimuonDirect)
and fake signal-like (charmDimuonIndirect) skim files without HTCondor.

Uses unpruned MCTrack truth coordinates via --truth-source.

Usage examples:
    # Single partition (both direct and indirect):
    python3 scripts/extract_charm_event_displays.py -p 17

    # Range of partitions using 8 local worker processes:
    python3 scripts/extract_charm_event_displays.py -p 1-50 -j 8

    # Only direct signal events, forcing overwrite:
    python3 scripts/extract_charm_event_displays.py -p 1-1000 -j 12 --mode direct --force
"""

import argparse
import os
import sys
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_partition_spec(spec_str):
    partitions = []
    for chunk in spec_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            parts = chunk.split("-")
            partitions.extend(range(int(parts[0]), int(parts[1]) + 1))
        else:
            partitions.append(int(chunk))
    return sorted(list(set(partitions)))


def run_display_job(cmd):
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    return res.returncode, dt, res.stdout


def process_partition(task):
    part, mode, data_dir, disp_base_dir, truth_source_pat, force, python_bin, script_path = task

    results = []
    modes_to_run = []
    if mode in ("direct", "both"):
        modes_to_run.append("direct")
    if mode in ("indirect", "both"):
        modes_to_run.append("indirect")

    for m in modes_to_run:
        suffix = "charmDimuonDirect" if m == "direct" else "charmDimuonIndirect"
        in_file = os.path.join(data_dir, str(part), f"sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_{suffix}.root")
        out_dir = os.path.join(disp_base_dir, m)
        out_file = os.path.join(out_dir, f"displays_part{part}.root")

        if not os.path.exists(in_file):
            results.append((part, m, "SKIPPED_NO_INPUT", 0.0, f"Input missing: {in_file}"))
            continue

        if not force and os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
            results.append((part, m, "EXISTS", 0.0, f"Output already exists: {out_file}"))
            continue

        os.makedirs(out_dir, exist_ok=True)
        truth_source = truth_source_pat % str(part) if "%s" in truth_source_pat else truth_source_pat

        cmd = [
            python_bin,
            script_path,
            "-i", in_file,
            "-o", out_file,
            "--mc-truth",
            "--recoMuons",
            "-j", "1",
        ]
        if os.path.exists(truth_source):
            cmd.extend(["--truth-source", truth_source])

        code, dt, out = run_display_job(cmd)
        if code == 0:
            results.append((part, m, "OK", dt, out_file))
        else:
            results.append((part, m, "FAILED", dt, out[-500:] if len(out) > 500 else out))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract 2D interactive event displays for charmDirect and charmIndirect files locally without HTCondor."
    )
    parser.add_argument(
        "-p", "--partitions",
        type=str,
        default="1-1000",
        help="Partitions specification, e.g. '17', '1-50', or '1-1000' (default: '1-1000')",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=8,
        help="Number of concurrent parallel worker processes (default: 8)",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["direct", "indirect", "both"],
        default="both",
        help="Extraction mode: 'direct', 'indirect', or 'both' (default: 'both')",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/data/newP8",
        help="Base path to data partitions directory (default: newP8)",
    )
    parser.add_argument(
        "--disp-dir",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions",
        help="Base directory for partition event displays (subdirs direct/ and indirect/)",
    )
    parser.add_argument(
        "--truth-source",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_dig.root",
        help="Pattern to unpruned simulation truth file (defaults to pre-MCEB simulation)",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Overwrite existing display files",
    )

    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(repo_dir, "scripts", "generate_2DEventDisplays.py")
    python_bin = sys.executable

    partitions = parse_partition_spec(args.partitions)

    print("=" * 78)
    print(" SND@LHC Local Charm Event Display Extractor (Without HTCondor)")
    print("=" * 78)
    print(f" Partitions   : {len(partitions)} ({partitions[0]}..{partitions[-1]})")
    print(f" Mode         : {args.mode}")
    print(f" Workers      : {args.jobs}")
    print(f" Data Dir     : {args.data_dir}")
    print(f" Display Dir  : {args.disp_dir}")
    print(f" Truth Source : {args.truth_source}")
    print(f" Force        : {args.force}")
    print("=" * 78)

    tasks = [
        (p, args.mode, args.data_dir, args.disp_dir, args.truth_source, args.force, python_bin, script_path)
        for p in partitions
    ]

    total_tasks = len(tasks)
    completed_tasks = 0
    start_time = time.time()
    ok_count = 0
    fail_count = 0
    skip_count = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(process_partition, t): t[0] for t in tasks}

        for fut in as_completed(futures):
            completed_tasks += 1
            part_results = fut.result()
            for part, m, status, dt, info in part_results:
                if status == "OK":
                    ok_count += 1
                    print(f" [{completed_tasks}/{total_tasks}] Part {part:>4} | {m:<8} | OK ({dt:.1f}s)")
                elif status == "EXISTS":
                    skip_count += 1
                elif status == "SKIPPED_NO_INPUT":
                    skip_count += 1
                else:
                    fail_count += 1
                    print(f" [{completed_tasks}/{total_tasks}] Part {part:>4} | {m:<8} | FAILED ({dt:.1f}s): {info}")

    total_dt = time.time() - start_time
    print("=" * 78)
    print(f" Finished in {total_dt:.1f}s | Success: {ok_count} | Skipped/Existing: {skip_count} | Failed: {fail_count}")
    print("=" * 78)


if __name__ == "__main__":
    main()
