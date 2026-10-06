#!/usr/bin/env python3
"""
scripts/prefilter.py
--------------------
Universal event prefiltering pipeline for SND@LHC data and Monte Carlo simulations.
Loads datasets via snd.DataManager, applies RDataFrame-based selection cuts
(including AvgScifiFiducialCut), snapshots filtered events to output ROOT files,
and creates symlinks to auxiliary files from the input directory.

Usage examples:
  # Single partition / run:
  python3 scripts/prefilter.py \
    -i "/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_numu_charm_signal_%s.root" \
    -o "/eos/user/i/idioniso/snd-numu-charm/prefiltered/%s/preselected_signal_run%s.root"

  # On raw / converted data with RDataFrame multi-threading:
  python3 scripts/prefilter.py \
    -i "/eos/experiment/sndlhc/convertedData/physics/2022/run_%s/sndsw_raw-0000.root" \
    -o "prefiltered_data/run_%s/preselected_run%s.root" \
    -j 4
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import re
from typing import List, Optional, Tuple, Dict, Any

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ROOT
from snd import (
    DataManager,
    AvgScifiFiducialCut,
    load_trident_libraries,
    resolve_input_files,
    extract_captures,
    symlink_input_root_files,
)


def format_output_path(
    input_file: str,
    input_pattern: str,
    output_pattern: str,
    index: int = 0,
) -> str:
    """
    Map an input file path to its corresponding output file path,
    transferring the capture matched by '%s' in the input pattern to all '%s'
    placeholders in the output pattern.
    """
    captures = extract_captures(input_file, input_pattern)

    if not captures:
        # Fallback: extract parent directory name or run number from filename
        parent = os.path.basename(os.path.dirname(os.path.abspath(input_file)))
        if parent and parent != ".":
            tag = parent
        else:
            m = re.search(r'(?:run|signal)?[_-]?(\d+)', os.path.basename(input_file))
            tag = m.group(1) if m else str(index)
    else:
        tag = captures[0]

    if "%s" in output_pattern:
        num_placeholders = output_pattern.count("%s")
        return output_pattern % tuple([tag] * num_placeholders)

    # If output_pattern is a directory
    if os.path.isdir(output_pattern) or not output_pattern.endswith(".root"):
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        return os.path.join(output_pattern, tag, f"{base_name}_run{tag}_preselection.root")

    return output_pattern


def copy_auxiliary_metadata(input_path: str, output_path: str) -> None:
    """
    Copy FairRoot metadata keys (BranchList, TimeBasedBranchList, FileHeader)
    from input file to output file if present.
    """
    keys_to_copy = ["BranchList", "TimeBasedBranchList", "FileHeader", "FileHeaderHeader"]
    try:
        fin = ROOT.TFile.Open(input_path, "READ")
        if not fin or fin.IsZombie():
            return

        found_objects = []
        for key_name in keys_to_copy:
            obj = fin.Get(key_name)
            if obj:
                found_objects.append((key_name, obj.Clone()))
        fin.Close()

        if not found_objects:
            return

        fout = ROOT.TFile.Open(output_path, "UPDATE")
        if fout and not fout.IsZombie():
            fout.cd()
            for key_name, obj in found_objects:
                if not fout.Get(key_name):
                    obj.Write(key_name, ROOT.TObject.kSingleKey)
            fout.Write()
            fout.Close()
    except Exception as err:
        print(f"  [Warning] Could not copy auxiliary metadata: {err}")


def prefilter_file(
    input_file: str,
    output_file: str,
    cut: ROOT.snd.AvgScifiFiducialCut,
    tree_name: Optional[str] = None,
    jobs: int = 1,
    max_events: Optional[int] = None,
    create_symlinks: bool = True,
    skip_empty: bool = False,
) -> Dict[str, Any]:
    """
    Process a single ROOT file with RDataFrame and apply AvgScifiFiducialCut.
    """
    result = {
        "input_file": input_file,
        "output_file": output_file,
        "total_entries": 0,
        "passed_entries": 0,
        "yield_pct": 0.0,
        "status": "OK",
        "error": None,
        "symlinks_created": 0,
    }

    try:
        # 1. Setup multi-threading
        # Note: ROOT RDataFrame does not support Range() when ImplicitMT is enabled
        effective_mt = jobs > 1 and (max_events is None or max_events <= 0)
        if effective_mt:
            ROOT.ROOT.EnableImplicitMT(jobs)
        else:
            ROOT.ROOT.DisableImplicitMT()

        # 2. Load dataset via DataManager
        dm = DataManager(input_file, tree_name=tree_name, load_libraries=False)
        resolved_tree = dm.tree_name

        if not dm.has_branch("Digi_ScifiHits"):
            result["status"] = "ERROR"
            result["error"] = f"Branch 'Digi_ScifiHits' not found in tree '{resolved_tree}'"
            return result

        # 3. Build RDataFrame
        df = dm.rdf(range_limit=max_events)
        total_entries = int(df.Count().GetValue())
        result["total_entries"] = total_entries

        # 4. Apply cut and extract passed entry IDs
        filtered_df = df.Filter(cut, ["Digi_ScifiHits"], "AvgScifiFiducialCut")
        passed_entry_list = sorted(list(filtered_df.Take['ULong64_t']('rdfentry_').GetValue()))
        passed_entries = len(passed_entry_list)
        result["passed_entries"] = passed_entries

        if total_entries > 0:
            result["yield_pct"] = (passed_entries / total_entries) * 100.0

        # Check if skipping empty
        if passed_entries == 0 and skip_empty:
            result["status"] = "SKIPPED_EMPTY"
            return result

        # 5. Write filtered events preserving complete tree schema
        out_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(out_dir, exist_ok=True)

        chain = dm.get_chain()
        fout = ROOT.TFile.Open(output_file, "RECREATE")
        out_tree = chain.CloneTree(0)

        for iev in passed_entry_list:
            chain.GetEntry(iev)
            out_tree.Fill()

        fout.Write()
        fout.Close()

        # 6. Copy FairRoot auxiliary metadata objects if present
        copy_auxiliary_metadata(input_file, output_file)

        # 7. Create symlinks to auxiliary files in the input directory
        if create_symlinks:
            in_dir = os.path.dirname(os.path.abspath(input_file))
            links = symlink_input_root_files(
                input_dir=in_dir,
                output_dir=out_dir,
                exclude_filenames={os.path.basename(output_file), os.path.basename(input_file)},
            )
            result["symlinks_created"] = len(links)

    except Exception as err:
        result["status"] = "ERROR"
        result["error"] = str(err)
    finally:
        # Reset implicit MT
        ROOT.ROOT.DisableImplicitMT()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Universal event prefiltering pipeline for SND@LHC data and MC using RDataFrame and DataManager.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input file path or pattern with '%%s' placeholder (e.g. /path/to/files/%%s/sndsw_raw-0000.root)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file pattern with '%%s' placeholder (e.g. /path/to/out/%%s/preselected_run%%s.root)",
    )
    parser.add_argument(
        "--tree-name",
        type=str,
        default=None,
        help="Tree name override ('rawConv' for data, 'cbmsim' for MC). Auto-detected if omitted.",
    )

    # Multi-threading options
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Number of worker threads for RDataFrame processing (default: 1; >1 enables multi-threading)",
    )

    # Cut parameters
    parser.add_argument(
        "--vertical-min",
        type=float,
        default=200.0,
        help="Minimum average vertical SciFi channel number",
    )
    parser.add_argument(
        "--vertical-max",
        type=float,
        default=1200.0,
        help="Maximum average vertical SciFi channel number",
    )
    parser.add_argument(
        "--horizontal-min",
        type=float,
        default=300.0,
        help="Minimum average horizontal SciFi channel number",
    )
    parser.add_argument(
        "--horizontal-max",
        type=float,
        default=1336.0,
        help="Maximum average horizontal SciFi channel number (128*12 - 200 = 1336)",
    )
    parser.add_argument(
        "--reversed",
        action="store_true",
        default=False,
        help="Select events outside the fiducial boundary (reversed cut)",
    )

    # Workflow options
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        default=False,
        help="Disable automatic symlink creation for input directory ROOT files",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        default=False,
        help="Do not save output ROOT files when 0 events pass the cut",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Maximum number of input files to process (-1 for all)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum events to process per file (useful for fast testing)",
    )

    args = parser.parse_args()

    # Load C++ analysis libraries and dictionaries
    load_trident_libraries()

    # Resolve input files
    input_files = resolve_input_files(args.input, max_files=args.max_files)
    if not input_files:
        print(f"[Error] No files matched input pattern: {args.input}")
        sys.exit(1)

    # Instantiate cut
    cut = AvgScifiFiducialCut(
        vertical_min=args.vertical_min,
        vertical_max=args.vertical_max,
        horizontal_min=args.horizontal_min,
        horizontal_max=args.horizontal_max,
        reversed=args.reversed,
    )

    print("=" * 80)
    print(" SND@LHC Universal Event Prefilter")
    print("=" * 80)
    print(f" Input Pattern:        {args.input}")
    print(f" Output Pattern:       {args.output}")
    print(f" Files Matched:        {len(input_files)}")
    thread_info = f"{args.jobs} threads (multi-threaded)" if args.jobs > 1 else "1 (single-threaded)"
    print(f" RDataFrame Workers:   {thread_info}")
    print(f" Cut Name:             AvgScifiFiducialCut {'[REVERSED]' if args.reversed else '[STANDARD]'}")
    print(f"   Vertical Range:     [{args.vertical_min:.1f}, {args.vertical_max:.1f}]")
    print(f"   Horizontal Range:   [{args.horizontal_min:.1f}, {args.horizontal_max:.1f}]")
    print(f" Symlink Auxiliary:    {not args.no_symlinks}")
    print(f" Skip Empty Files:     {args.skip_empty}")
    print("=" * 80)

    start_time = time.time()
    results = []

    for idx, input_file in enumerate(input_files, start=1):
        out_file = format_output_path(input_file, args.input, args.output, index=idx)
        print(f"[{idx}/{len(input_files)}] Processing: {os.path.basename(input_file)}")
        print(f"  Input:  {input_file}")
        print(f"  Output: {out_file}")

        file_start = time.time()
        res = prefilter_file(
            input_file=input_file,
            output_file=out_file,
            cut=cut,
            tree_name=args.tree_name,
            jobs=args.jobs,
            max_events=args.max_events,
            create_symlinks=not args.no_symlinks,
            skip_empty=args.skip_empty,
        )
        elapsed_file = time.time() - file_start

        if res["status"] == "OK":
            print(f"  Result: {res['passed_entries']}/{res['total_entries']} passed ({res['yield_pct']:.2f}%) in {elapsed_file:.1f}s")
            if res["symlinks_created"] > 0:
                print(f"  Symlinks: {res['symlinks_created']} file(s) linked")
        elif res["status"] == "SKIPPED_EMPTY":
            print(f"  Result: 0/{res['total_entries']} passed (skipped creating empty output)")
        else:
            print(f"  Result: ERROR: {res['error']}")

        results.append(res)
        print("-" * 80)

    total_elapsed = time.time() - start_time
    total_in = sum(r["total_entries"] for r in results)
    total_out = sum(r["passed_entries"] for r in results)
    overall_yield = (total_out / total_in * 100.0) if total_in > 0 else 0.0
    successful_files = sum(1 for r in results if r["status"] in ("OK", "SKIPPED_EMPTY"))

    print("=" * 80)
    print(" PREFILTERING SUMMARY")
    print("=" * 80)
    print(f" Files Processed:     {successful_files}/{len(input_files)} successful")
    print(f" Total Input Events:  {total_in}")
    print(f" Total Passed Events: {total_out} ({overall_yield:.2f}%)")
    print(f" Total Elapsed Time:  {total_elapsed:.1f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
