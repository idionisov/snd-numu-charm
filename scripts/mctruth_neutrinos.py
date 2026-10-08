#!/usr/bin/env python3
"""
scripts/filter_neutrinos.py
----------------------------
Comprehensive SND@LHC neutrino event categorization and truth analysis pipeline.

Categorizes all simulated neutrino interactions without discarding events:
  1. Neutrino Flavour:
     - nu_mu (14), anti_nu_mu (-14)
     - nu_e (12), anti_nu_e (-12)
     - nu_tau (16), anti_nu_tau (-16)
  2. Interaction Current:
     - Charged Current (CC) vs Neutral Current (NC)
  3. Immediate Interaction Products & Topologies:
     - Assigns a unique canonical channel_id to each unique combination of
       particles produced at the primary neutrino interaction vertex (motherId == 0).
     - Maintains a persistent lookup table in config/interaction_channels.json.
  4. Charmed Hadron Production & Direct Decay Modes:
     - Identifies charm species (D0, D+, Ds, Lambda_c, etc.)
     - Categorizes immediate charm decay daughters (charm_decay_channel_id)
     - Identifies direct decay modes: "to_muon" (prompt dimuon), "to_electron", "hadronic"
  5. Kinematics & Detector Acceptance:
     - DIS variables (Q^2, x, y, W)
     - Primary lepton kinematics (electron, muon, tau)
     - Displaced decay kinematics and impact parameters
     - Downstream (DS) MuFilter acceptance tracking

Produces output ROOT files containing:
  - Cloned event tree (e.g. cbmsim / rawConv) with all detector hits & MCTracks
  - Comprehensive flat truth TTree with all categorization & kinematic branches
  - Auxiliary metadata & symlinks preserved for full simulation fidelity

Usage examples:
  # Process partition 33:
  python3 scripts/filter_neutrinos.py -p 33

  # Process partitions 0 through 10 with 4 workers:
  python3 scripts/filter_neutrinos.py -p 0-10 -j 4

  # Process single file with 500 events:
  python3 scripts/filter_neutrinos.py -i "/path/to/sndLHC.Genie-TGeant4_digCPP.root" -n 500 -o "/tmp/categorized.root"
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import List, Optional, Dict, Any
from collections import Counter

# Add repository root to python path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import ROOT
from snd import (
    load_trident_libraries,
    load_config,
    resolve_input_files,
    determine_output_path,
    symlink_input_root_files,
    build_processor,
    extract_captures,
    ChannelLookupManager,
    pdg_to_name,
    process_categorized_neutrino_file,
)

# Load C++ libraries and disable GUI
load_trident_libraries(_repo_root)
ROOT.gROOT.SetBatch(True)


def parse_partitions(part_arg: Optional[str]) -> Optional[List[int]]:
    """Parse partition specification string (e.g. '0-400', '0,1,2', 'all')."""
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


_worker_processor = None
_worker_cfg = None


def _init_worker(config_path: Optional[str]):
    """Initialize worker process with ROOT batch mode and processor."""
    global _worker_processor, _worker_cfg
    import ROOT
    ROOT.gROOT.SetBatch(True)
    from snd import load_trident_libraries, load_config, build_processor
    load_trident_libraries(_repo_root)
    _worker_cfg = load_config(config_path)
    _worker_processor = build_processor(_worker_cfg.get("processor", {}))


def _run_worker_task(task_args: tuple) -> dict:
    """Worker task execution function for parallel partition processing."""
    part_str, in_file, out_file, max_entries, create_symlinks, config_path = task_args
    global _worker_processor, _worker_cfg
    if _worker_processor is None:
        _init_worker(config_path)

    t0 = time.time()
    try:
        from snd import process_categorized_neutrino_file
        res = process_categorized_neutrino_file(
            input_file=in_file,
            output_file=out_file,
            processor=_worker_processor,
            cfg=_worker_cfg,
            max_entries=max_entries,
            create_symlinks=create_symlinks,
        )
        res["elapsed"] = time.time() - t0
        res["partition"] = part_str
        return res
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "partition": part_str,
            "input_file": in_file,
            "status": f"error: {str(e)}",
            "total": 0,
            "processed": 0,
            "elapsed": time.time() - t0,
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Categorize SND@LHC neutrino interactions by flavor, current, immediate products, and charm decay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Input file path or pattern (supports wildcard '*' and '%%s' placeholder)",
    )
    parser.add_argument(
        "-p", "--partitions",
        type=str,
        default=None,
        help="Partitions to process: range ('0-400'), comma-separated ('0,1,2'), or 'all'",
    )
    parser.add_argument(
        "-n", "--entries",
        type=int,
        default=-1,
        help="Max events per file to process (-1 for all)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory or pattern with '%%s' placeholders",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=os.path.join(_repo_root, "config", "filter_neutrinos_config.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--symlink-input",
        dest="symlink_input",
        action="store_true",
        default=True,
        help="Create symlinks in output partition directory to input ROOT files",
    )
    parser.add_argument(
        "--no-symlink-input",
        dest="symlink_input",
        action="store_false",
        default=True,
        help="Disable creating symlinks in output partition directory",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=False,
        help="Skip partitions where the categorized ROOT file already exists and is non-empty",
    )
    parser.add_argument(
        "--symlink-only",
        action="store_true",
        default=False,
        help="Only create symlinks to input ROOT files without processing events",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # 1. Load Configuration
    config_path = args.config
    if not os.path.exists(config_path):
        alt_path = os.path.join(_repo_root, "config", "filter_numu_charm_config.yaml")
        if os.path.exists(alt_path):
            config_path = alt_path

    cfg = load_config(config_path)

    input_cfg = cfg.get("input", {})
    file_pattern = args.input if args.input is not None else input_cfg.get(
        "file_pattern",
        "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/"
        "sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/%s/sndLHC.Genie-TGeant4_digCPP.root"
    )
    max_files = input_cfg.get("max_files", -1)

    output_cfg = cfg.get("output", {})
    out_pattern = args.output if args.output is not None else output_cfg.get(
        "output_pattern",
        "/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_categorized.root"
    )

    do_symlink = args.symlink_input

    # 2. Resolve Tasks
    selected_partitions = parse_partitions(args.partitions)
    tasks = []

    if selected_partitions is not None:
        for p in selected_partitions:
            in_file = format_partition_path(file_pattern, p)
            if os.path.exists(in_file):
                out_file = format_partition_path(out_pattern, p)
                tasks.append((str(p), in_file, out_file))
            else:
                print(f"[Warning] Partition {p} input file not found: {in_file}")
    else:
        matched_files = resolve_input_files(file_pattern, max_files=max_files)
        for idx, in_file in enumerate(matched_files):
            captures = extract_captures(in_file, file_pattern)
            part_str = captures[0] if captures else str(idx)
            out_file = format_partition_path(out_pattern, part_str)
            tasks.append((part_str, in_file, out_file))

    if not tasks:
        print(f"[Error] No input files found matching pattern: {file_pattern}")
        sys.exit(1)

    if args.skip_existing:
        filtered_tasks = []
        skipped_count = 0
        for p, in_f, out_f in tasks:
            if os.path.exists(out_f) and os.path.getsize(out_f) > 1024:
                skipped_count += 1
            else:
                filtered_tasks.append((p, in_f, out_f))
        print(f"[Info] --skip-existing: skipped {skipped_count} existing partition(s), {len(filtered_tasks)} remaining.")
        tasks = filtered_tasks
        if not tasks:
            print("[Info] All requested partitions already processed. Exiting.")
            return

    total_tasks = len(tasks)

    # Standalone symlink-only mode
    if args.symlink_only:
        print("=" * 80)
        print(" SND@LHC: Symlinking Input ROOT Files to Output Partitions")
        print("=" * 80)
        print(f"Total partitions: {total_tasks}")
        total_linked = 0
        for idx, (part_str, in_file, out_file) in enumerate(tasks):
            in_dir = os.path.dirname(os.path.abspath(in_file))
            out_dir = os.path.dirname(os.path.abspath(out_file))
            created = symlink_input_root_files(
                input_dir=in_dir,
                output_dir=out_dir,
                exclude_filenames={os.path.basename(out_file), os.path.basename(in_file)}
            )
            total_linked += len(created)
            print(f"[{idx + 1}/{total_tasks}] Partition {part_str:<6} | {len(created)} ROOT file(s) symlinked -> {out_dir}")
        print("=" * 80)
        print(f"Done! Created {total_linked} symlinks across {total_tasks} partition(s).")
        return

    # Initialize Channel Manager
    ch_mgr = ChannelLookupManager.get_instance()

    print("=" * 80)
    print(" SND@LHC: Comprehensive Neutrino Event Categorization Pipeline")
    print("=" * 80)
    print(f"Configuration file           : {config_path}")
    print(f"Input file pattern           : {file_pattern}")
    print(f"Output file pattern          : {out_pattern}")
    print(f"Channel lookup table         : {ch_mgr.table_path}")
    print(f"Known primary channels       : {len(ch_mgr.primary_channels)}")
    print(f"Known charm decay channels   : {len(ch_mgr.charm_decay_channels)}")
    print(f"Total partitions to process  : {total_tasks}")
    print(f"Max events per partition     : {args.entries if args.entries > 0 else 'All'}")
    print(f"Parallel worker processes    : {args.jobs}")
    print(f"Create partition symlinks    : {do_symlink}")
    print("=" * 80)

    # 3. Execution
    start_time = time.time()
    results = []

    if args.jobs > 1 and total_tasks > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        worker_args = [
            (part_str, in_f, out_f, args.entries, do_symlink, config_path)
            for (part_str, in_f, out_f) in tasks
        ]
        print(f"\nDispatching {total_tasks} partition tasks to pool of {args.jobs} worker processes...\n")
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            future_to_part = {
                executor.submit(_run_worker_task, w_args): w_args[0]
                for w_args in worker_args
            }
            completed_count = 0
            for future in as_completed(future_to_part):
                completed_count += 1
                res = future.result()
                results.append(res)
                p = res.get("partition", "?")
                n_proc = res.get("processed", 0)
                n_tot = res.get("total", 0)
                el = res.get("elapsed", 0.0)
                st = res.get("status", "unknown")
                print(f"[{completed_count:3d}/{total_tasks:3d}] Partition {p:<6} | {n_proc:5d}/{n_tot:5d} events categorized | {el:6.1f}s | {st}")
    else:
        # Sequential processing
        proc_cfg = cfg.get("processor", {})
        processor = build_processor(proc_cfg)

        for idx, (part_str, in_file, out_file) in enumerate(tasks):
            print(f"\n[{idx + 1}/{total_tasks}] Processing Partition: {part_str}")
            print(f"  Input : {in_file}")
            print(f"  Output: {out_file}")

            t0 = time.time()
            res = process_categorized_neutrino_file(
                input_file=in_file,
                output_file=out_file,
                processor=processor,
                cfg=cfg,
                max_entries=args.entries,
                create_symlinks=do_symlink,
            )
            res["elapsed"] = time.time() - t0
            res["partition"] = part_str
            results.append(res)

            n_proc = res.get("processed", 0)
            n_tot = res.get("total", 0)
            el = res.get("elapsed", 0.0)
            print(f"  Categorized {n_proc}/{n_tot} events in {el:.1f}s.")

    # Save any new channels registered across the run
    ch_mgr.save()

    total_wall_time = time.time() - start_time

    # 4. Print Overall Categorization Summary
    tot_events_all = sum(r.get("processed", 0) for r in results)
    flavor_totals = Counter()
    current_totals = Counter()
    tot_charm = 0
    tot_charm_mu = 0
    tot_charm_e = 0
    tot_charm_had = 0

    for r in results:
        for flv, cnt in r.get("flavor_counts", {}).items():
            flavor_totals[flv] += cnt
        for cur, cnt in r.get("current_counts", {}).items():
            current_totals[cur] += cnt
        tot_charm += r.get("n_charm", 0)
        tot_charm_mu += r.get("n_charm_to_muon", 0)
        tot_charm_e += r.get("n_charm_to_electron", 0)
        tot_charm_had += r.get("n_charm_hadronic", 0)

    print("\n" + "=" * 80)
    print(" OVERALL NEUTRINO CATEGORIZATION SUMMARY")
    print("=" * 80)
    print(f"Total Partitions Processed : {len(results)}")
    print(f"Total Events Categorized   : {tot_events_all}")
    print(f"Total Processing Time      : {total_wall_time:.1f}s")
    print("-" * 80)
    print(" BREAKDOWN BY INTERACTION CURRENT:")
    for cur, count in sorted(current_totals.items()):
        pct = (100.0 * count / max(tot_events_all, 1))
        cur_str = str(cur)
        print(f"  {cur_str:<10} : {count:8d} ({pct:6.2f}%)")
    print("-" * 80)
    print(" BREAKDOWN BY NEUTRINO FLAVOUR & CURRENT:")
    for flv, count in sorted(flavor_totals.items(), key=lambda x: -x[1]):
        pct = (100.0 * count / max(tot_events_all, 1))
        flv_str = str(flv)
        print(f"  {flv_str:<24} : {count:8d} ({pct:6.2f}%)")
    print("-" * 80)
    print(" CHARMED HADRON PRODUCTION & DIRECT DECAY BREAKDOWN:")
    print(f"  Total Charmed Hadron Events : {tot_charm:8d} ({100.0 * tot_charm / max(tot_events_all, 1):6.2f}% of all events)")
    print(f"  Direct Decay to Muon (mu2)  : {tot_charm_mu:8d} ({100.0 * tot_charm_mu / max(tot_charm, 1):6.2f}% of charm)")
    print(f"  Direct Decay to Electron    : {tot_charm_e:8d} ({100.0 * tot_charm_e / max(tot_charm, 1):6.2f}% of charm)")
    print(f"  Direct Hadronic Decay       : {tot_charm_had:8d} ({100.0 * tot_charm_had / max(tot_charm, 1):6.2f}% of charm)")
    print("-" * 80)
    print(f"Unique Primary Channels in Lookup Table : {len(ch_mgr.primary_channels)}")
    print(f"Unique Charm Decays in Lookup Table    : {len(ch_mgr.charm_decay_channels)}")
    print(f"Lookup Table File                       : {ch_mgr.table_path}")
    print("=" * 80)
    print("\nCategorization complete! Output files are ready for custom downstream selection.")


if __name__ == "__main__":
    main()
