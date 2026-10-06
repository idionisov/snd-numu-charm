#!/usr/bin/env python3
"""
scripts/filter_signal_numu_charm.py
-----------------------------------
Hierarchical neutrino interaction skimmer and truth analysis pipeline for SND@LHC.

Configuration is fully specified via YAML (see config/filter_numu_charm_config.yaml):
- Level 1: Neutrino flavor (any, numu, nue, nutau)
  └── Level 2: Interaction current (any, CC, NC)
        └── Level 3: Charmed hadron production (require: true/false, species: any/D0/D+/Ds/Lambda_c)
              └── Level 4: Charm decay channel (decay: any/to_muon, require_opposite_sign: true/false)

Truth TTree Modular Tiers:
- Universal: Neutrino kinematics, interaction flags, DIS variables, vertex coordinates.
- Lepton: Primary outgoing charged lepton (mu1 for nu_mu CC) kinematics.
- Charm: Charmed hadron 4-momentum, decay length, lifetimes, species, and decay vertex.
- Decay Muon: Prompt muon (mu2) from charm decay kinematics, pTrel, and impact parameters.
- Dimuon System: Composite pair invariant mass, opening angles, azimuthal delta-phi, and asymmetry.
Only the branches corresponding to the active hierarchy tiers are stored in the TTree (in "auto" mode).

CLI arguments:
  -i, --input     : Input file path or pattern (supports wildcard '*' and '%s' placeholder)
  -n, --entries   : Max events per file (-1 for all)
  -j, --threads   : Worker threads
  -o, --output    : Output directory or pattern with '%s' placeholders
  -c, --config    : Path to YAML configuration file
  --fiducial      : Require interaction vertex in Target fiducial volume
  --symlink-input : Create symlinks in output partition directories to all input ROOT files
  --symlink-only  : Only create symlinks to input ROOT files without running skimmer
"""

from __future__ import annotations

import os
import sys
import argparse
import ROOT

# Add repo root to python path to import snd package
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import (
    load_trident_libraries,
    load_config,
    resolve_input_files,
    determine_output_path,
    symlink_input_root_files,
    build_processor,
    resolve_hierarchical_selection,
    process_single_file,
    process_simulation_file_dual_truth,
    extract_captures,
)

# Pre-load libraries so ROOT has all custom C++ dictionaries and classes available
load_trident_libraries(_repo_root)

# Disable ROOT GUI windows in batch mode
ROOT.gROOT.SetBatch(True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Skim neutrino events with hierarchical truth TTree generation from SND@LHC MC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Input file path or pattern (supports wildcard '*' and '%%s' placeholder)"
    )
    parser.add_argument(
        "-n", "--entries",
        type=int,
        default=-1,
        help="Max events per file to process (-1 for all events)"
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        default=1,
        help="Number of worker threads"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory or pattern with '%%s' placeholders (e.g. /path/%%s/file_%%s.root)"
    )
    parser.add_argument(
        "--fiducial",
        action="store_true",
        default=False,
        help="Require interaction vertex within the Target fiducial volume"
    )
    parser.add_argument(
        "--require-ds-acceptance",
        dest="require_ds_acceptance",
        action="store_true",
        default=None,
        help="Require charm decay muon to have >= 3 MCPoints in Downstream (DS) MuFilter system"
    )
    parser.add_argument(
        "--no-ds-acceptance",
        dest="require_ds_acceptance",
        action="store_false",
        default=None,
        help="Do not require charm decay muon to be in DS acceptance"
    )
    parser.add_argument(
        "--symlink-input",
        dest="symlink_input",
        action="store_true",
        default=None,
        help="Create symlinks in output partition directory to all ROOT files from input directory"
    )
    parser.add_argument(
        "--no-symlink-input",
        dest="symlink_input",
        action="store_false",
        default=None,
        help="Disable creating symlinks in output partition directory"
    )
    parser.add_argument(
        "--symlink-only",
        action="store_true",
        default=False,
        help="Only create symlinks to input ROOT files in output directories without running event skimmer"
    )
    parser.add_argument(
        "--extract-all-truth",
        action="store_true",
        default=False,
        help="Extract truth branches for ALL events without exception into output-truth, and simultaneously store signal events in output-signal",
    )
    parser.add_argument(
        "--output-truth",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_truth.root",
        help="Output ROOT file pattern for ALL events with truth branches (when using --extract-all-truth)",
    )
    parser.add_argument(
        "--output-signal",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_truth_numuCC_charm_dimuon.root",
        help="Output ROOT file pattern for SIGNAL events (when using --extract-all-truth)",
    )
    parser.add_argument(
        "-p", "--partitions",
        type=str,
        default=None,
        help="Partitions to process: range ('0-400'), comma-separated ('0,1,2'), or 'all'",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=os.path.join(_repo_root, "config", "filter_numu_charm_config.yaml"),
        help="Path to YAML configuration file"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # 1. Load Configuration
    cfg = load_config(args.config)

    input_cfg = cfg.get("input", {})
    file_pattern = args.input if args.input is not None else input_cfg.get(
        "file_pattern",
        input_cfg.get(
            "pattern",
            "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/"
            "sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/%s/sndLHC.Genie-TGeant4_digCPP.root"
        )
    )
    max_files = input_cfg.get("max_files", -1)
    output_cfg = cfg.get("output", {})

    # 2. Resolve Selection Criteria & Active Truth Tiers from YAML Config
    filter_expr, filter_desc, predicate, active_tiers = resolve_hierarchical_selection(
        cfg=cfg,
        cli_fiducial=args.fiducial,
        cli_ds_acceptance=args.require_ds_acceptance,
    )

    # 3. Resolve Input Files
    input_files = resolve_input_files(file_pattern, max_files=max_files)
    total_files = len(input_files)

    do_symlink = args.symlink_input if args.symlink_input is not None else output_cfg.get("symlink_input_root_files", False)
    symlink_only_selected = output_cfg.get("symlink_only_selected", True)

    # Standalone symlink-only mode
    if args.symlink_only:
        print("=" * 80)
        print(" SND@LHC: Symlinking Input ROOT Files to Output Partitions")
        print("=" * 80)
        print(f"Total partitions: {total_files}")
        total_linked = 0
        for idx, in_file in enumerate(input_files):
            out_file = determine_output_path(
                input_path=in_file,
                input_pattern=file_pattern,
                output_arg=args.output,
                output_cfg=output_cfg,
                index=idx,
                total_files=total_files
            )
            in_dir = os.path.dirname(os.path.abspath(in_file))
            out_dir = os.path.dirname(os.path.abspath(out_file))
            created = symlink_input_root_files(
                input_dir=in_dir,
                output_dir=out_dir,
                exclude_filenames={os.path.basename(out_file)}
            )
            total_linked += len(created)
            partition = os.path.basename(in_dir)
            print(f"[{idx + 1}/{total_files}] Partition {partition:<6} | {len(created)} ROOT file(s) symlinked -> {out_dir}")
        print("=" * 80)
        print(f"Done! Verified/created symlinks across {total_files} partition(s).")
        return

    # Dual truth extraction mode (All Events + Signal with symlinks)
    if args.extract_all_truth:
        proc_cfg = cfg.get("processor", {})
        processor = build_processor(proc_cfg)
        print("=" * 80)
        print(" SND@LHC: Dual Truth Extraction (All Events & Signal)")
        print("=" * 80)
        print(f"Input file pattern       : {file_pattern}")
        print(f"Output All-Events Truth  : {args.output_truth}")
        print(f"Output Signal Truth      : {args.output_signal}")
        print(f"Total files to process   : {total_files}")
        print(f"Max events per file      : {args.entries if args.entries > 0 else 'All'}")
        print(f"Create symlinks          : {do_symlink}")
        print("=" * 80)

        for idx, in_file in enumerate(input_files):
            captures = extract_captures(in_file, file_pattern)
            part_str = captures[0] if captures else str(idx)
            count_t = args.output_truth.count("%s")
            out_truth = args.output_truth % tuple(part_str for _ in range(count_t)) if count_t > 0 else args.output_truth
            count_s = args.output_signal.count("%s")
            out_sig = args.output_signal % tuple(part_str for _ in range(count_s)) if count_s > 0 else args.output_signal

            print(f"\n[{idx + 1}/{total_files}] Processing Partition: {part_str}")
            print(f"  Input : {in_file}")
            print(f"  Truth : {out_truth}")
            print(f"  Signal: {out_sig}")

            res = process_simulation_file_dual_truth(
                input_file=in_file,
                truth_output_file=out_truth,
                signal_output_file=out_sig,
                processor=processor,
                cfg=cfg,
                max_entries=args.entries,
                create_symlinks=do_symlink,
            )
            n_proc = res.get("processed", 0)
            n_sig = res.get("signal", 0)
            print(f"  Result: {n_proc} total processed -> {n_sig} signal events saved.")
        return

    # 4. Configure Processor
    proc_cfg = cfg.get("processor", {})
    processor = build_processor(proc_cfg)

    print("=" * 80)
    print(" SND@LHC: Hierarchical Neutrino Skimmer & Truth Tree Generator")
    print("=" * 80)
    print(f"Configuration file           : {args.config}")
    print(f"Input file pattern           : {file_pattern}")
    print(f"Selection hierarchy          : {filter_desc}")
    print(f"Filter expression            : {filter_expr}")
    print(f"Active truth TTree tiers     : {sorted(list(active_tiers))}")
    print(f"Total input files to process : {total_files}")
    print(f"Max events per file          : {args.entries if args.entries > 0 else 'All'}")
    print(f"Skip empty files             : {output_cfg.get('skip_empty_files', True)}")
    print(f"Symlink input ROOT files     : {do_symlink} (only selected: {symlink_only_selected})")
    print("=" * 80)

    # 5. Process each file 1-to-1
    results = []
    tot_events_all = 0
    tot_signal_all = 0
    files_written = 0

    for idx, in_file in enumerate(input_files):
        out_file = determine_output_path(
            input_path=in_file,
            input_pattern=file_pattern,
            output_arg=args.output,
            output_cfg=output_cfg,
            index=idx,
            total_files=total_files
        )
        print(f"\n[{idx + 1}/{total_files}] Processing:\n  Input : {in_file}")

        res = process_single_file(
            input_file=in_file,
            output_file=out_file,
            processor=processor,
            cfg=cfg,
            filter_expr=filter_expr,
            predicate=predicate,
            active_tiers=active_tiers,
            max_entries=args.entries
        )
        results.append(res)

        n_tot = res.get("total", 0)
        n_sig = res.get("signal", 0)
        tot_events_all += n_tot
        tot_signal_all += n_sig

        status = res.get("status", "unknown")
        if status == "skipped_empty":
            print(f"  Summary: 0/{n_tot} matched. Skipped (empty file not created).")
            if do_symlink and not symlink_only_selected:
                in_dir = os.path.dirname(os.path.abspath(in_file))
                out_dir = os.path.dirname(os.path.abspath(out_file))
                created = symlink_input_root_files(
                    input_dir=in_dir,
                    output_dir=out_dir,
                    exclude_filenames={os.path.basename(out_file)}
                )
                if created:
                    print(f"  Symlinked {len(created)} input ROOT file(s) into {out_dir}")
        elif status == "success":
            files_written += 1
            pct = (100.0 * n_sig / max(n_tot, 1)) if n_tot > 0 else 0.0
            print(f"  Summary: {n_sig}/{n_tot} events saved ({pct:.2f}%). Output: {out_file}")
            if do_symlink:
                in_dir = os.path.dirname(os.path.abspath(in_file))
                out_dir = os.path.dirname(os.path.abspath(out_file))
                created = symlink_input_root_files(
                    input_dir=in_dir,
                    output_dir=out_dir,
                    exclude_filenames={os.path.basename(out_file)}
                )
                if created:
                    print(f"  Symlinked {len(created)} input ROOT file(s) into {out_dir}")
        else:
            print(f"  Summary: Status={status}")

    # 6. Final Processing Summary Table
    print("\n" + "=" * 80)
    print(" OVERALL PROCESSING SUMMARY")
    print("=" * 80)
    print(f"{'#':<3} | {'Partition':<12} | {'Total':>8} | {'Selected':>8} | {'Yield (%)':>10} | {'Status':<14}")
    print("-" * 80)
    for idx, (in_file, res) in enumerate(zip(input_files, results)):
        partition = os.path.basename(os.path.dirname(in_file))
        n_tot = res.get("total", 0)
        n_sig = res.get("signal", 0)
        pct = (100.0 * n_sig / max(n_tot, 1)) if n_tot > 0 else 0.0
        status = res.get("status", "unknown")
        print(f"{idx + 1:<3} | {partition:<12} | {n_tot:8d} | {n_sig:8d} | {pct:9.2f}% | {status:<14}")

    print("=" * 80)
    overall_pct = (100.0 * tot_signal_all / max(tot_events_all, 1)) if tot_events_all > 0 else 0.0
    print(f"TOTAL: {tot_signal_all} events selected out of {tot_events_all} total events ({overall_pct:.2f}%).")
    print(f"Saved {files_written} non-empty output file(s) (skipped {total_files - files_written} empty files).")
    print("=" * 80)
    print("\nDone!")


if __name__ == "__main__":
    main()
