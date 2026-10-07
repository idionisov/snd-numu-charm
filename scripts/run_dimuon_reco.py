#!/usr/bin/env python3
"""
scripts/run_dimuon_reco.py
--------------------------
High-level parallel orchestrator for SND@LHC Downstream Dimuon Tracking.

Executes:
  python $SNDSW_ROOT/shipLHC/scripts/run_TrackSelections.py \\
    -ht -t dimuon_DS --nTracks 2 \\
    -f "<input_file>" -s <start_event> -n <n_events> \\
    -o "<output_file>" -g "<geofile>"

Features:
  - Supports input and output file patterns with '%s' partition placeholders.
  - Automatically skips partitions whose input ROOT file does not exist.
  - Automatically resolves geometry files (partition-local geofile or run-mapped geo).
  - Parallel execution across partitions using a worker thread pool.
  - Gracefully handles self-termination exit codes (SIGTERM / pyExit) from sndsw.
  - Verifies generated ROOT output files and reports event yields.

Usage examples:
  # Run over all existing signal partitions in parallel (4 workers):
  python3 scripts/run_dimuon_reco.py \\
    -i "/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_truth_numuCC_charm_dimuon.root" \\
    -o "/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_truth_numuCC_charm_dimuon_DiMuTrks.root" \\
    -p 0-400 \\
    -j 4

  # Test single partition 0 (first 100 events):
  python3 scripts/run_dimuon_reco.py \\
    -i "/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_truth_numuCC_charm_dimuon.root" \\
    -o "/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_truth_numuCC_charm_dimuon_DiMuTrks.root" \\
    -p 0 \\
    -n 100
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Dict, Any

# Ensure project root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from snd.io_utils import get_geofile_for_run


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
    """Replace all '%s' placeholders in pattern with the partition string."""
    p_str = str(partition)
    count = pattern.count("%s")
    if count > 0:
        return pattern % tuple(p_str for _ in range(count))
    return pattern


def discover_available_partitions(input_pattern: str, max_check: int = 500) -> List[int]:
    """
    Scan for partitions that currently have an existing input file.
    If parent directory contains numeric folders, scans them; otherwise tests 0..max_check.
    """
    found = []
    # If pattern has /data/%s/ or similar, try listing parent directory
    base_dir = input_pattern.split("%s")[0]
    if os.path.isdir(base_dir):
        try:
            entries = os.listdir(base_dir)
            numeric_dirs = sorted([int(e) for e in entries if e.isdigit()])
            for p in numeric_dirs:
                candidate = format_partition_path(input_pattern, p)
                if os.path.isfile(candidate):
                    found.append(p)
            if found:
                return found
        except Exception:
            pass

    # Fallback brute-force check
    for p in range(max_check + 1):
        candidate = format_partition_path(input_pattern, p)
        if os.path.isfile(candidate):
            found.append(p)
    return found


def resolve_tracking_script() -> str:
    """Find the path to run_TrackSelections.py in SNDSW_ROOT or known user locations."""
    candidates = []
    sndsw_root = os.environ.get("SNDSW_ROOT")
    if sndsw_root:
        candidates.append(os.path.join(sndsw_root, "shipLHC", "scripts", "run_TrackSelections.py"))

    # User local source repos
    candidates.extend([
        "/afs/cern.ch/user/i/idioniso/snd_master/sndsw/shipLHC/scripts/run_TrackSelections.py",
        "/afs/cern.ch/user/i/idioniso/snd_master/sw3.12/slc9_x86-64/sndsw/master-local1/shipLHC/scripts/run_TrackSelections.py",
    ])

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Could not find run_TrackSelections.py. Ensure SNDSW_ROOT is set and sndsw is initialized."
    )


def resolve_geofile_path(partition: int | str, input_file: str, explicit_geofile: Optional[str] = None) -> str:
    """
    Determine the appropriate geofile for reconstruction.
    1. If explicit_geofile is specified, use it (formatted if it contains '%s').
    2. Check partition-local geofile_full.Genie-TGeant4.root.
    3. Use snd.io_utils.get_geofile_for_run.
    """
    if explicit_geofile:
        if "%s" in explicit_geofile:
            return format_partition_path(explicit_geofile, partition)
        return explicit_geofile

    # Partition-local geofile
    in_dir = os.path.dirname(os.path.abspath(input_file))
    local_geo = os.path.join(in_dir, "geofile_full.Genie-TGeant4.root")
    if os.path.isfile(local_geo):
        return local_geo

    # Run / Year database fallback
    return get_geofile_for_run(run_number=partition, input_path=input_file)


def verify_root_file(file_path: str) -> Tuple[bool, int]:
    """
    Verify if a ROOT output file was created, is not a zombie, and retrieve its entry count.
    Returns (is_valid, entry_count).
    """
    if not os.path.isfile(file_path) or os.path.getsize(file_path) < 500:
        return False, 0

    try:
        import ROOT
        ROOT.gROOT.SetBatch(True)
        f = ROOT.TFile.Open(file_path, "READ")
        if not f or f.IsZombie():
            if f:
                f.Close()
            return False, 0

        entries = 0
        tree = f.Get("cbmsim") or f.Get("rawConv")
        if tree:
            entries = int(tree.GetEntries())
        f.Close()
        return True, entries
    except Exception:
        # If ROOT import fails in thread, check file size as basic validity
        return os.path.getsize(file_path) > 1000, 0


def run_single_partition(
    partition: int | str,
    input_file: str,
    output_file: str,
    geo_file: str,
    tracking_script: str,
    start_event: int = 0,
    n_events: int = -1,
    track_type: str = "dimuon_DS",
    n_tracks: int = 2,
    hough_tracking: bool = True,
    par_file: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    save_logs: bool = False,
    lock: Optional[threading.Lock] = None,
) -> Dict[str, Any]:
    """
    Execute run_TrackSelections.py for a single partition.
    """
    result = {
        "partition": partition,
        "input_file": input_file,
        "output_file": output_file,
        "status": "UNKNOWN",
        "entries": 0,
        "elapsed_sec": 0.0,
        "error_message": "",
    }

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable,
        tracking_script,
        "-t", str(track_type),
        "--nTracks", str(n_tracks),
        "-f", input_file,
        "-o", output_file,
        "-g", geo_file,
        "-s", str(start_event),
        "-n", str(n_events),
    ]

    if hough_tracking:
        cmd.append("-ht")

    if par_file:
        cmd.extend(["-par", par_file])

    if extra_args:
        cmd.extend(extra_args)

    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        elapsed = time.time() - start_time
        result["elapsed_sec"] = elapsed

        # Save output log if requested
        if save_logs:
            log_file = output_file + ".log"
            try:
                with open(log_file, "w") as lf:
                    lf.write(f"Command: {' '.join(cmd)}\n\n")
                    lf.write(proc.stdout)
            except Exception as e:
                pass

        # run_TrackSelections.py calls pyExit() with self-SIGTERM (exit code -15 or 143)
        # Verify output ROOT file directly
        is_valid, entries = verify_root_file(output_file)
        result["entries"] = entries

        if is_valid:
            result["status"] = "SUCCESS"
        else:
            result["status"] = "ERROR"
            # Extract last few error lines from stdout
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            tail = "\n  ".join(lines[-10:]) if lines else "Process terminated without output"
            result["error_message"] = f"Exit code {proc.returncode}; Output invalid.\n  {tail}"

    except Exception as err:
        result["elapsed_sec"] = time.time() - start_time
        result["status"] = "EXCEPTION"
        result["error_message"] = str(err)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="High-level parallel Downstream Dimuon Tracking orchestrator for SND@LHC data and MC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input ROOT file path or pattern with '%%s' placeholder",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output ROOT file path or pattern with '%%s' placeholder",
    )
    parser.add_argument(
        "-p", "--partitions",
        type=str,
        default=None,
        help="Partitions to process: '0-400', '0,1,2', single integer, or 'all' (auto-detect existing)",
    )
    parser.add_argument(
        "-j", "--jobs", "--parallel",
        dest="jobs",
        type=int,
        default=4,
        help="Number of concurrent worker processes",
    )
    parser.add_argument(
        "-g", "--geofile",
        type=str,
        default=None,
        help="Path to geofile (auto-detected from partition directory or run database if omitted)",
    )
    parser.add_argument(
        "-s", "--start-event", "--nStart",
        dest="start_event",
        type=int,
        default=0,
        help="First event index to process",
    )
    parser.add_argument(
        "-n", "--n-events", "--nEvents", "--max-events",
        dest="n_events",
        type=int,
        default=-1,
        help="Number of events to process per partition (-1 for all)",
    )
    parser.add_argument(
        "-t", "--track-type",
        type=str,
        default="dimuon_DS",
        help="Reconstruction track type ('dimuon_DS', 'DS', 'ScifiDS', 'Scifi')",
    )
    parser.add_argument(
        "--nTracks",
        type=int,
        default=2,
        help="Required minimum number of reconstructed tracks to save event",
    )
    parser.add_argument(
        "--no-ht",
        dest="hough_tracking",
        action="store_false",
        default=True,
        help="Disable Hough Tracking (-ht flag)",
    )
    parser.add_argument(
        "-par", "--par-file",
        dest="par_file",
        type=str,
        default=None,
        help="TrackingParams.xml parameter file path override",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip partition if valid output ROOT file already exists",
    )
    parser.add_argument(
        "--save-logs",
        action="store_true",
        default=False,
        help="Save stdout/stderr to '<output_file>.log' for each partition",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print partition plan without executing reconstruction",
    )

    args, extra_args = parser.parse_known_args()

    # 1. Resolve tracking script location
    tracking_script = resolve_tracking_script()

    # 2. Determine target partition list
    has_placeholder = ("%s" in args.input)
    if has_placeholder:
        if args.partitions:
            parsed = parse_partitions(args.partitions)
            target_partitions = parsed if parsed is not None else list(range(401))
        else:
            print("No partitions specified (-p); auto-discovering available input files...")
            target_partitions = discover_available_partitions(args.input)
            if not target_partitions:
                print("Could not auto-detect files. Checking default partition range 0-400...")
                target_partitions = list(range(401))
    else:
        target_partitions = [0]

    print("=" * 80)
    print(" SND@LHC DOWNSTREAM DIMUON RECONSTRUCTION")
    print("=" * 80)
    print(f" Input Pattern:       {args.input}")
    print(f" Output Pattern:      {args.output}")
    print(f" Track Script:        {tracking_script}")
    print(f" Track Type:          {args.track_type}")
    print(f" Minimum Tracks:      {args.nTracks}")
    print(f" Hough Tracking:      {args.hough_tracking}")
    print(f" Start / Max Events:  {args.start_event} / {args.n_events if args.n_events >= 0 else 'All'}")
    print(f" Parallel Workers:    {args.jobs}")
    print(f" Partitions Scanned:  {len(target_partitions)}")
    print("=" * 80)

    # 3. Filter partitions by existence of input file
    tasks_to_run = []
    skipped_missing = []
    skipped_existing = []

    for p in target_partitions:
        in_file = format_partition_path(args.input, p)
        out_file = format_partition_path(args.output, p)

        if not os.path.isfile(in_file):
            skipped_missing.append((p, in_file))
            continue

        if args.skip_existing:
            valid, _ = verify_root_file(out_file)
            if valid:
                skipped_existing.append((p, out_file))
                continue

        geo_file = resolve_geofile_path(p, in_file, args.geofile)
        tasks_to_run.append({
            "partition": p,
            "input_file": in_file,
            "output_file": out_file,
            "geo_file": geo_file,
        })

    print(f" Plan: {len(tasks_to_run)} to process | {len(skipped_missing)} skipped (input not found) | {len(skipped_existing)} skipped (existing)")
    if skipped_missing and len(skipped_missing) <= 10:
        for p, f in skipped_missing:
            print(f"   [Skip Missing] Partition {p}: {os.path.basename(f)}")
    elif skipped_missing:
        sample = ", ".join(str(p) for p, _ in skipped_missing[:8])
        print(f"   [Skip Missing] {len(skipped_missing)} partitions without input (e.g. {sample}, ...)")

    if not tasks_to_run:
        print("\nNo matching tasks to process. Exiting.")
        sys.exit(0)

    if args.dry_run:
        print("\n[Dry Run] Sample commands that would be executed:")
        for t in tasks_to_run[:5]:
            cmd_preview = (
                f"python {tracking_script} -ht -t {args.track_type} --nTracks {args.nTracks} "
                f"-f {t['input_file']} -o {t['output_file']} -g {t['geo_file']} "
                f"-s {args.start_event} -n {args.n_events}"
            )
            print(f"  Partition {t['partition']}:\n    {cmd_preview}")
        if len(tasks_to_run) > 5:
            print(f"  ... and {len(tasks_to_run) - 5} more partitions.")
        sys.exit(0)

    # 4. Execute reconstruction tasks in parallel
    print("-" * 80)
    print(f" Launching {len(tasks_to_run)} reconstruction jobs using {args.jobs} worker threads...")
    print("-" * 80)

    print_lock = threading.Lock()
    completed_count = 0
    total_tasks = len(tasks_to_run)
    successful_results = []
    failed_results = []
    start_total_time = time.time()

    def worker_wrapper(task_info: Dict[str, Any]) -> Dict[str, Any]:
        p = task_info["partition"]
        with print_lock:
            print(f"  [Start] Partition {p:<4} -> {os.path.basename(task_info['output_file'])}")

        res = run_single_partition(
            partition=p,
            input_file=task_info["input_file"],
            output_file=task_info["output_file"],
            geo_file=task_info["geo_file"],
            tracking_script=tracking_script,
            start_event=args.start_event,
            n_events=args.n_events,
            track_type=args.track_type,
            n_tracks=args.nTracks,
            hough_tracking=args.hough_tracking,
            par_file=args.par_file,
            extra_args=extra_args,
            save_logs=args.save_logs,
            lock=print_lock,
        )

        nonlocal completed_count
        with print_lock:
            completed_count += 1
            if res["status"] == "SUCCESS":
                print(f"  [Done]  Partition {p:<4} ({completed_count}/{total_tasks}) in {res['elapsed_sec']:.1f}s | {res['entries']} dimuon events")
            else:
                print(f"  [FAIL]  Partition {p:<4} ({completed_count}/{total_tasks}) in {res['elapsed_sec']:.1f}s | Error: {res['error_message']}")

        return res

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {executor.submit(worker_wrapper, task): task for task in tasks_to_run}
        for future in as_completed(futures):
            res = future.result()
            if res["status"] == "SUCCESS":
                successful_results.append(res)
            else:
                failed_results.append(res)

    total_time = time.time() - start_total_time
    total_dimuon_events = sum(r["entries"] for r in successful_results)

    # 5. Summary
    print("\n" + "=" * 80)
    print(" DIMUON RECONSTRUCTION SUMMARY")
    print("=" * 80)
    print(f" Partitions Scanned:        {len(target_partitions)}")
    print(f" Missing Input (Skipped):   {len(skipped_missing)}")
    if args.skip_existing:
        print(f" Already Existing (Skipped):{len(skipped_existing)}")
    print(f" Successfully Processed:    {len(successful_results)}/{total_tasks}")
    if failed_results:
        print(f" Failed Tasks:              {len(failed_results)}/{total_tasks}")
    print(f" Total Dimuon Events Found: {total_dimuon_events}")
    print(f" Total Elapsed Time:        {total_time:.1f} seconds")
    print("=" * 80)

    if failed_results:
        print("\nFailed Partitions:")
        for r in failed_results:
            print(f"  * Partition {r['partition']}: {r['error_message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
