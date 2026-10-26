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
    """Parse partition specification string (e.g. '0-400', '0,1,2', 'all', or path to file)."""
    if not part_arg or part_arg.strip().lower() == "all":
        return None

    if os.path.isfile(part_arg):
        partitions = set()
        with open(part_arg, "r") as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    partitions.add(int(line))
        return sorted(list(partitions))

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


def get_mctruth_weight(event: Any = None, info: Any = None) -> float:
    """
    Returns the Monte Carlo truth event weight.
    For neutrino productions, returns 1.0 for all events (leaving raw_weight and mc_weight
    as computed by the C++ truth processor).
    """
    return 1.0


def get_passing_muon_weight(
    event: Any,
    info: Any = None,
    pot_scale: float = 80.0 * 1e12,
    sim_pp: float = 100e6,
) -> dict:
    """
    Returns the weights for passing muon (FLUKA pmu) simulation:
      - raw_weight = event.MCTrack[0].GetWeight() (or 0.0 if MCTrack is empty)
      - mc_weight  = (raw_weight * pot_scale) / sim_pp
      - mctruth_weight = mc_weight
    """
    tracks = getattr(event, "MCTrack", None)
    if tracks is not None and tracks.GetEntries() > 0:
        raw_w = float(tracks[0].GetWeight())
    else:
        raw_w = 0.0

    mc_w = (raw_w * pot_scale) / sim_pp
    return {
        "raw_weight": raw_w,
        "mc_weight": mc_w,
        "mctruth_weight": mc_w,
    }


def resolve_weight_function(
    sample_type: str,
    pot_scale: Optional[float] = None,
    sim_pp: Optional[float] = None,
    cfg: Optional[dict] = None,
):
    """Resolve event weighting function based on sample type."""
    if sample_type in ("passing_muon", "pmu", "pmu2022"):
        eff_pot = pot_scale if pot_scale is not None else float(cfg.get("pmu_pot_scale", 80.0 * 1e12) if cfg else 80.0 * 1e12)
        eff_sim = sim_pp if sim_pp is not None else float(cfg.get("pmu_sim_pp", 100e6) if cfg else 100e6)
        return lambda ev, inf: get_passing_muon_weight(ev, inf, pot_scale=eff_pot, sim_pp=eff_sim)
    return get_mctruth_weight


def _run_worker_task(task_args: tuple) -> dict:
    """Worker task execution function for parallel partition processing."""
    if len(task_args) >= 10:
        part_str, in_file, out_file, max_entries, create_symlinks, config_path, truth_tree_name, store_in_cbmsim, sample_type, truth_source = task_args[:10]
    elif len(task_args) >= 9:
        part_str, in_file, out_file, max_entries, create_symlinks, config_path, truth_tree_name, store_in_cbmsim, sample_type = task_args[:9]
        truth_source = None
    elif len(task_args) == 8:
        part_str, in_file, out_file, max_entries, create_symlinks, config_path, truth_tree_name, store_in_cbmsim = task_args[:8]
        sample_type = None
        truth_source = None
    else:
        part_str, in_file, out_file, max_entries, create_symlinks, config_path = task_args[:6]
        truth_tree_name = "mctruth"
        store_in_cbmsim = False
        sample_type = None
        truth_source = None
    global _worker_processor, _worker_cfg
    if _worker_processor is None:
        _init_worker(config_path)

    eff_sample_type = sample_type if sample_type is not None else _worker_cfg.get("sample_type", "neutrino")
    weight_fn = resolve_weight_function(eff_sample_type, cfg=_worker_cfg)

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
            weight_fn=weight_fn,
            truth_tree_name=truth_tree_name,
            store_in_cbmsim=store_in_cbmsim,
            truth_source=truth_source,
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
        default=os.path.join(_repo_root, "config", "mctruth_neutrinos_config.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--truth-source",
        type=str,
        default=None,
        help="Optional external ROOT file or pattern with '%%s' to extract truth information from (mapped by event number)",
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
        "--truth-tree-name",
        type=str,
        default=None,
        help="Name of dedicated truth TTree (default: from config or 'mctruth')",
    )
    parser.add_argument(
        "--store-in-cbmsim",
        dest="store_in_cbmsim",
        action="store_true",
        default=None,
        help="Also duplicate truth branches inside the cbmsim tree (default: False)",
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
    parser.add_argument(
        "--sample-type",
        type=str,
        default=None,
        choices=["neutrino", "passing_muon", "pmu", "pmu2022"],
        help="Sample simulation type: 'neutrino' (default) or 'passing_muon' / 'pmu'",
    )
    parser.add_argument(
        "--pmu-pot-scale",
        type=float,
        default=None,
        help="Proton / luminosity scale for passing muons (default: 80 * 1e12)",
    )
    parser.add_argument(
        "--pmu-sim-pp",
        type=float,
        default=None,
        help="Number of simulated pp interactions for passing muons (default: 100e6)",
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
    truth_tree_name = args.truth_tree_name if args.truth_tree_name is not None else output_cfg.get("truth_tree_name", "mctruth")
    store_in_cbmsim = args.store_in_cbmsim if args.store_in_cbmsim is not None else output_cfg.get("store_in_cbmsim", False)

    # 2. Resolve Tasks
    selected_partitions = parse_partitions(args.partitions)
    tasks = []

    truth_source_pattern = args.truth_source

    if selected_partitions is not None:
        for p in selected_partitions:
            in_file = format_partition_path(file_pattern, p)
            if os.path.exists(in_file):
                out_file = format_partition_path(out_pattern, p)
                truth_src = format_partition_path(truth_source_pattern, p) if truth_source_pattern else None
                tasks.append((str(p), in_file, out_file, truth_src))
            else:
                print(f"[Warning] Partition {p} input file not found: {in_file}")
    else:
        matched_files = resolve_input_files(file_pattern, max_files=max_files)
        for idx, in_file in enumerate(matched_files):
            captures = extract_captures(in_file, file_pattern)
            part_str = captures[0] if captures else str(idx)
            out_file = format_partition_path(out_pattern, part_str)
            truth_src = format_partition_path(truth_source_pattern, part_str) if truth_source_pattern else None
            tasks.append((part_str, in_file, out_file, truth_src))

    if not tasks:
        print(f"[Error] No input files found matching pattern: {file_pattern}")
        sys.exit(1)

    if args.skip_existing:
        filtered_tasks = []
        skipped_count = 0
        for p, in_f, out_f, truth_src in tasks:
            if os.path.exists(out_f) and os.path.getsize(out_f) > 1024:
                skipped_count += 1
            else:
                filtered_tasks.append((p, in_f, out_f, truth_src))
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
        for idx, (part_str, in_file, out_file, *_) in enumerate(tasks):
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

    sample_type = args.sample_type or cfg.get("sample_type", "neutrino")
    weight_fn = resolve_weight_function(
        sample_type=sample_type,
        pot_scale=args.pmu_pot_scale,
        sim_pp=args.pmu_sim_pp,
        cfg=cfg,
    )

    print("=" * 80)
    print(" SND@LHC: Comprehensive Neutrino Event Categorization Pipeline")
    print("=" * 80)
    print(f"Configuration file           : {config_path}")
    print(f"Sample simulation type       : {sample_type}")
    print(f"Input file pattern           : {file_pattern}")
    if truth_source_pattern:
        print(f"External truth source        : {truth_source_pattern}")
    print(f"Output file pattern          : {out_pattern}")
    print(f"Channel lookup table         : {ch_mgr.table_path}")
    print(f"Known primary channels       : {len(ch_mgr.primary_channels)}")
    print(f"Known charm decay channels   : {len(ch_mgr.charm_decay_channels)}")
    print(f"Truth tree name              : {truth_tree_name}")
    print(f"Store truth in cbmsim        : {store_in_cbmsim}")
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
            (part_str, in_f, out_f, args.entries, do_symlink, config_path, truth_tree_name, store_in_cbmsim, sample_type, truth_src)
            for (part_str, in_f, out_f, truth_src) in tasks
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

        for idx, (part_str, in_file, out_file, truth_src) in enumerate(tasks):
            print(f"\n[{idx + 1}/{total_tasks}] Processing Partition: {part_str}")
            print(f"  Input        : {in_file}")
            if truth_src:
                print(f"  Truth Source : {truth_src}")
            print(f"  Output       : {out_file}")

            t0 = time.time()
            res = process_categorized_neutrino_file(
                input_file=in_file,
                output_file=out_file,
                processor=processor,
                cfg=cfg,
                max_entries=args.entries,
                create_symlinks=do_symlink,
                weight_fn=weight_fn,
                truth_tree_name=truth_tree_name,
                store_in_cbmsim=store_in_cbmsim,
                truth_source=truth_src,
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
    tot_charm_ds_mu = 0
    tot_two_muons_in_ds = 0
    tot_other_ds_dimuon = 0
    tot_signal_like = 0

    for r in results:
        for flv, cnt in r.get("flavor_counts", {}).items():
            flavor_totals[flv] += cnt
        for cur, cnt in r.get("current_counts", {}).items():
            current_totals[cur] += cnt
        tot_charm += r.get("n_charm", 0)
        tot_charm_mu += r.get("n_charm_to_muon", 0)
        tot_charm_e += r.get("n_charm_to_electron", 0)
        tot_charm_had += r.get("n_charm_hadronic", 0)
        tot_charm_ds_mu += r.get("n_charm_hadronic_downstream_mu", 0)
        tot_two_muons_in_ds += r.get("n_two_muons_in_ds", 0)
        tot_other_ds_dimuon += r.get("n_other_ds_dimuon", 0)
        tot_signal_like += r.get("n_signal_like", 0)

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
    print(" CHARMED HADRON PRODUCTION & DECAY BREAKDOWN:")
    print(f"  Total Charmed Hadron Events         : {tot_charm:8d} ({100.0 * tot_charm / max(tot_events_all, 1):6.2f}% of all events)")
    print(f"  Direct Decay to Muon (Prompt mu2)   : {tot_charm_mu:8d} ({100.0 * tot_charm_mu / max(tot_charm, 1):6.2f}% of charm)")
    print(f"  Direct Decay to Electron            : {tot_charm_e:8d} ({100.0 * tot_charm_e / max(tot_charm, 1):6.2f}% of charm)")
    print(f"  Direct Hadronic Decay               : {tot_charm_had:8d} ({100.0 * tot_charm_had / max(tot_charm, 1):6.2f}% of charm)")
    print(f"  Hadronic Decay with Downstream Muon : {tot_charm_ds_mu:8d} ({100.0 * tot_charm_ds_mu / max(tot_charm, 1):6.2f}% of charm)")
    print("-" * 80)
    print(" DOWNSTREAM (DS) DIMUONS & SIGNAL-LIKE CANDIDATES (>=3 Hor & >=3 Ver Points):")
    print(f"  Events with >=2 Muons in DS         : {tot_two_muons_in_ds:8d} ({100.0 * tot_two_muons_in_ds / max(tot_events_all, 1):6.2f}% of all events)")
    print(f"  Direct Charm Dimuon Signal          : {tot_charm_mu:8d} ({100.0 * tot_charm_mu / max(tot_events_all, 1):6.2f}% of all events)")
    print(f"  Indirect Charm Dimuon Fake          : {tot_charm_ds_mu:8d} ({100.0 * tot_charm_ds_mu / max(tot_events_all, 1):6.2f}% of all events)")
    print(f"  Other Signal-Like Dimuon Bkg        : {tot_other_ds_dimuon:8d} ({100.0 * tot_other_ds_dimuon / max(tot_events_all, 1):6.2f}% of all events)")
    print(f"  Total Signal-Like Dimuons           : {tot_signal_like:8d} ({100.0 * tot_signal_like / max(tot_events_all, 1):6.2f}% of all events)")
    print("-" * 80)
    print(f"Unique Primary Channels in Lookup Table : {len(ch_mgr.primary_channels)}")
    print(f"Unique Charm Decays in Lookup Table    : {len(ch_mgr.charm_decay_channels)}")
    print(f"Lookup Table File                       : {ch_mgr.table_path}")
    print("=" * 80)
    print("\nCategorization complete! Output files are ready for custom downstream selection.")


if __name__ == "__main__":
    main()
