#!/usr/bin/env python3
"""
SND@LHC: Parallel Local Pipeline Runner (Dimuon Tracking -> MCTruth -> Event Displays)

Runs the multi-stage pipeline on multiple partitions concurrently (e.g., 10 parallel workers)
with live progress monitoring, per-partition logging, and resume support.

Usage:
    # Run 10 workers on the 2022 nu14 (1000 partitions) dataset:
    ./scripts/run_pipeline_parallel.py --preset nu14_2022 -j 10

    # Run on specific partitions:
    ./scripts/run_pipeline_parallel.py --preset nu14_2022 -j 10 -p 1-50

    # Run in background with nohup:
    nohup ./scripts/run_pipeline_parallel.py --preset nu14_2022 -j 10 > local_pipeline.log 2>&1 &
"""

import argparse
import os
import sys
import time
import subprocess
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

DEFAULT_PRESETS = {
    "nu14_2022": {
        "input_pattern": "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2mmRangeCut/sndlhc_15000fb-1_2022_down/nu14/volume_volTarget/%s/sndLHC.Genie-TGeant4_dig.root",
        "track_pattern": "/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_dig_2MuTrks.root",
        "truth_pattern": "/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root",
        "disp_pattern": "/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/partitions/displays_part%s.root",
        "default_partitions": list(range(1, 1001)),
    },
    "old_100fb_2022": {
        "input_pattern": "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/%s/sndLHC.Genie-TGeant4_digCPP.root",
        "track_pattern": "/eos/user/i/idioniso/snd-numu-charm/data/old_sndlhc_100fb-1_2022_down_volTarget/%s/sndLHC.Genie-TGeant4_digCPP_2MuTrks.root",
        "truth_pattern": "/eos/user/i/idioniso/snd-numu-charm/data/old_sndlhc_100fb-1_2022_down_volTarget/%s/sndLHC.Genie-TGeant4_digCPP_2MuTrks_truth.root",
        "disp_pattern": "/eos/user/i/idioniso/snd-numu-charm/event_displays/old_sndlhc_100fb-1_2022_down_volTarget/partitions/displays_part%s.root",
        "default_partitions": list(range(0, 401)),
    },
    "100fb": {
        "input_pattern": "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/%s/sndLHC.Genie-TGeant4_digCPP.root",
        "track_pattern": "/eos/user/i/idioniso/snd-numu-charm/data/old_sndlhc_100fb-1_2022_down_volTarget/%s/sndLHC.Genie-TGeant4_digCPP_2MuTrks.root",
        "truth_pattern": "/eos/user/i/idioniso/snd-numu-charm/data/old_sndlhc_100fb-1_2022_down_volTarget/%s/sndLHC.Genie-TGeant4_digCPP_2MuTrks_truth.root",
        "disp_pattern": "/eos/user/i/idioniso/snd-numu-charm/event_displays/old_sndlhc_100fb-1_2022_down_volTarget/partitions/displays_part%s.root",
        "default_partitions": list(range(0, 401)),
    }
}

stop_requested = False

def sigint_handler(signum, frame):
    global stop_requested
    print("\n[!] Interrupt signal received! Stopping new tasks and waiting for running tasks to exit...")
    stop_requested = True

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)


def parse_partition_spec(spec_str):
    """Parses partition strings like '1-100', '1,2,5-10', or 'all'."""
    parts = []
    for item in spec_str.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = map(int, item.split("-"))
            parts.extend(range(start, end + 1))
        else:
            parts.append(int(item))
    return sorted(list(set(parts)))


def is_partition_complete(p, truth_pat, disp_pat):
    """Checks if a partition already has valid truth and display outputs."""
    truth_file = truth_pat % p
    disp_file = disp_pat % p
    if os.path.isfile(truth_file) and os.path.getsize(truth_file) > 1000:
        if os.path.isfile(disp_file) and os.path.getsize(disp_file) > 500:
            return True
    return False


def create_partition_symlinks(p, in_pat, trk_pat):
    """Creates symlinks to all ROOT files from the input directory into the output directory."""
    try:
        in_file = in_pat % p
        out_file = trk_pat % p
        in_dir = os.path.dirname(os.path.abspath(in_file))
        out_dir = os.path.dirname(os.path.abspath(out_file))
        if os.path.isdir(in_dir) and os.path.abspath(in_dir) != os.path.abspath(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            for entry in os.listdir(in_dir):
                if not entry.endswith(".root"):
                    continue
                src_path = os.path.join(in_dir, entry)
                dst_path = os.path.join(out_dir, entry)
                if not os.path.isfile(src_path):
                    continue
                if os.path.islink(dst_path):
                    if os.path.realpath(dst_path) == os.path.realpath(src_path) or os.readlink(dst_path) == src_path:
                        continue
                    os.remove(dst_path)
                elif os.path.exists(dst_path):
                    continue
                try:
                    os.symlink(src_path, dst_path)
                except OSError:
                    pass
    except Exception:
        pass


def run_single_partition(p, script_path, in_pat, trk_pat, tru_pat, dsp_pat, log_dir, create_symlinks=True):
    """Executes run_pipeline_partition.sh for a single partition and logs to a file."""
    if stop_requested:
        return p, "cancelled", 0.0, ""

    if create_symlinks:
        create_partition_symlinks(p, in_pat, trk_pat)

    log_file = os.path.join(log_dir, f"pipeline_part{p}.log")
    cmd = [
        script_path,
        str(p),
        in_pat,
        trk_pat,
        tru_pat,
        dsp_pat,
    ]

    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )

        while proc.poll() is None:
            if stop_requested:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    pass
                return p, "cancelled", time.time() - t0
            time.sleep(1)

        retcode = proc.returncode

    duration = time.time() - t0
    err_tail = ""
    if retcode == 0:
        status = "success"
    else:
        status = f"failed (exit {retcode})"
        try:
            with open(log_file, "r") as lf:
                lines = [line.strip() for line in lf if line.strip()]
                err_tail = " | ".join(lines[-3:]) if lines else "No log output recorded"
        except Exception as e:
            err_tail = str(e)
    return p, status, duration, err_tail


def main():
    parser = argparse.ArgumentParser(
        description="Run SND@LHC Chained Pipeline (Tracking -> Truth -> Displays) in parallel locally."
    )
    parser.add_argument(
        "--preset",
        choices=list(DEFAULT_PRESETS.keys()),
        default="nu14_2022",
        help="Preset dataset configuration (default: nu14_2022)."
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=10,
        help="Number of concurrent worker threads/processes (default: 10)."
    )
    parser.add_argument(
        "-p", "--partitions",
        type=str,
        default=None,
        help="Partitions to process, e.g. '1-100', '1,2,5-10'. Defaults to all partitions in preset."
    )
    parser.add_argument(
        "--partitions-file",
        type=str,
        default=None,
        help="Path to a text file with one partition ID per line."
    )
    parser.add_argument(
        "--input-pattern",
        type=str,
        default=None,
        help="Custom raw digit input pattern."
    )
    parser.add_argument(
        "--track-pattern",
        type=str,
        default=None,
        help="Custom tracked output pattern."
    )
    parser.add_argument(
        "--truth-pattern",
        type=str,
        default=None,
        help="Custom truth output pattern."
    )
    parser.add_argument(
        "--disp-pattern",
        type=str,
        default=None,
        help="Custom display output pattern."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to store per-partition log files."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if partition outputs already exist."
    )
    parser.add_argument(
        "--no-symlinks",
        dest="create_symlinks",
        action="store_false",
        default=True,
        help="Disable automatic symlink creation for input directory ROOT files into output directory."
    )

    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_script = os.path.join(repo_dir, "htcondor", "tracking_and_truth", "run_pipeline_partition.sh")

    if not os.path.isfile(pipeline_script):
        print(f"ERROR: Could not find pipeline script at: {pipeline_script}", file=sys.stderr)
        sys.exit(1)

    cfg = DEFAULT_PRESETS[args.preset]
    in_pat = args.input_pattern or cfg["input_pattern"]
    trk_pat = args.track_pattern or cfg["track_pattern"]
    tru_pat = args.truth_pattern or cfg["truth_pattern"]
    dsp_pat = args.disp_pattern or cfg["disp_pattern"]

    if args.partitions_file:
        with open(args.partitions_file) as f:
            partitions = [int(line.strip()) for line in f if line.strip()]
    elif args.partitions:
        partitions = parse_partition_spec(args.partitions)
    else:
        partitions = cfg["default_partitions"]

    log_dir = args.log_dir or os.path.join(repo_dir, "logs_local_pipeline", args.preset)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 76)
    print("  SND@LHC Parallel Pipeline Orchestrator")
    print("=" * 76)
    print(f"  Preset Dataset    : {args.preset}")
    print(f"  Parallel Workers  : {args.jobs}")
    print(f"  Total Partitions  : {len(partitions)}")
    print(f"  Input Pattern     : {in_pat}")
    print(f"  Tracking Target   : {trk_pat}")
    print(f"  Truth Target      : {tru_pat}")
    print(f"  Displays Target   : {dsp_pat}")
    print(f"  Partition Logs Dir: {log_dir}")
    print("=" * 76)

    # Pre-check already completed partitions
    to_run = []
    already_done = 0
    for p in partitions:
        if not args.force and is_partition_complete(p, tru_pat, dsp_pat):
            already_done += 1
        else:
            to_run.append(p)

    print(f"  Status Summary    : {already_done} already completed | {len(to_run)} to process\n")

    # If symlinks enabled, ensure already completed partitions also have original ROOT files linked
    if args.create_symlinks and already_done > 0:
        for p in partitions:
            if p not in to_run:
                create_partition_symlinks(p, in_pat, trk_pat)

    if not to_run:
        print("  All partitions are already completed! Nothing to run.")
        return

    start_time = time.time()
    completed_count = already_done
    total_count = len(partitions)
    succeeded = 0
    failed = 0

    active_futures = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for p in to_run:
            if stop_requested:
                break
            fut = executor.submit(
                run_single_partition,
                p, pipeline_script, in_pat, trk_pat, tru_pat, dsp_pat, log_dir, args.create_symlinks
            )
            active_futures[fut] = p

        for fut in as_completed(active_futures):
            p = active_futures[fut]
            try:
                part, status, duration, err_tail = fut.result()
            except Exception as e:
                part, status, duration, err_tail = p, f"exception ({e})", 0.0, str(e)

            if status == "success":
                succeeded += 1
                completed_count += 1
                stat_tag = "\033[92mSUCCESS\033[0m"
            elif status == "cancelled":
                stat_tag = "\033[93mCANCELLED\033[0m"
            else:
                failed += 1
                completed_count += 1
                stat_tag = f"\033[91m{status.upper()}\033[0m"

            elapsed = time.time() - start_time
            rate = succeeded / (elapsed / 60.0) if elapsed > 1 else 0.0
            remaining_tasks = len(to_run) - (succeeded + failed)
            eta_str = str(timedelta(seconds=int(remaining_tasks / (rate / 60.0)))) if rate > 0 else "--:--:--"

            pct = (completed_count / total_count) * 100.0
            now_str = datetime.now().strftime("%H:%M:%S")

            print(
                f"[{now_str}] Part {part:>4} -> {stat_tag} ({duration:>5.1f}s) | "
                f"Total: {completed_count}/{total_count} ({pct:>5.1f}%) | "
                f"Rate: {rate:>4.1f}/min | ETA: {eta_str}"
            )
            if err_tail and status != "success" and status != "cancelled":
                print(f"         \033[33mError log tail:\033[0m {err_tail}")

            if stop_requested:
                break

    total_time = time.time() - start_time
    print("\n" + "=" * 76)
    print("  PARALLEL PROCESSING COMPLETED")
    print(f"  Duration : {timedelta(seconds=int(total_time))}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed   : {failed}")
    print(f"  Skipped  : {already_done}")
    print(f"  Logs     : {log_dir}")
    print("=" * 76)

if __name__ == "__main__":
    main()
