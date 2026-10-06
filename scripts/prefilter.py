#!/usr/bin/env python3
"""
scripts/prefilter.py
--------------------
Universal event prefiltering pipeline for SND@LHC data and Monte Carlo simulations.
Loads datasets via snd.DataManager, applies RDataFrame-based selection cuts
(including AvgScifiFiducialCut), snapshots filtered events to output ROOT files,
and creates symlinks to auxiliary files from the input directory.

Supports an optional --cutflow argument to generate comprehensive cutflow counters
and diagnostic histograms (SciFi hit activity, busiest stations, hit density weights,
HCAL/US QDC, DS QDC, Veto QDC, ratios, and projections) after each incremental cut,
configured through an external YAML configuration file.

Usage examples:
  # Filter events using default fiducial cut:
  python3 scripts/prefilter.py \\
    -i "/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_numu_charm_signal_%s.root" \\
    -o "prefiltered/%s/preselected_signal_run%s.root"

  # Produce complete cutflow and diagnostic histograms:
  python3 scripts/prefilter.py \\
    -i "/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_numu_charm_signal_%s.root" \\
    -o "prefiltered/%s/preselected_signal_run%s.root" \\
    --cutflow "cutflow_histograms.root" \\
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
    load_cutflow_config,
    define_preselection_metrics,
    save_cutflow_root_file,
    merge_stage_histograms,
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
    if captures:
        out_placeholders = output_pattern.count("%s")
        if out_placeholders > 0:
            fill_values = tuple(captures[0] for _ in range(out_placeholders))
            return output_pattern % fill_values
        return output_pattern

    if "%s" in output_pattern:
        tag = str(index)
        out_count = output_pattern.count("%s")
        return output_pattern % tuple(tag for _ in range(out_count))

    if os.path.isdir(output_pattern) or output_pattern.endswith("/"):
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        tag = str(index)
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


def print_cutflow_summary(
    cutflow_stats: List[Dict[str, Any]],
    cutflow_filepath: Optional[str] = None,
) -> None:
    """
    Print an aligned summary table of incremental selection cuts and efficiencies.
    """
    if not cutflow_stats:
        return

    print("\n" + "=" * 80)
    print(" CUTFLOW PROCESSING SUMMARY")
    print("=" * 80)
    print(f"{'Step':<5} | {'Cut ID / Name':<38} | {'Passed':>10} | {'Rel. Eff':>10} | {'Cum. Eff':>10}")
    print("-" * 80)

    raw_count = float(cutflow_stats[0]["count"]) if cutflow_stats and cutflow_stats[0]["count"] > 0 else 1.0
    prev_count = raw_count

    for i, step in enumerate(cutflow_stats):
        c_val = float(step["count"])
        rel_pct = (c_val / prev_count * 100.0) if prev_count > 0 else 0.0
        cum_pct = (c_val / raw_count * 100.0) if raw_count > 0 else 0.0
        name = step.get("name", step["id"])[:38]
        print(f"{i:<5} | {name:<38} | {int(c_val):>10} | {rel_pct:>9.2f}% | {cum_pct:>9.2f}%")
        prev_count = c_val

    print("=" * 80)
    if cutflow_filepath:
        print(f" Cutflow ROOT file: {cutflow_filepath}")
        print("=" * 80 + "\n")


def prefilter_file(
    input_file: str,
    output_file: str,
    cut: ROOT.snd.AvgScifiFiducialCut,
    cutflow_config: Optional[Dict[str, Any]] = None,
    record_cutflow: bool = False,
    tree_name: Optional[str] = None,
    jobs: int = 1,
    max_events: Optional[int] = None,
    create_symlinks: bool = True,
    skip_empty: bool = False,
) -> Dict[str, Any]:
    """
    Process a single ROOT file with RDataFrame, apply sequential cuts,
    book diagnostic histograms for the cutflow, and write filtered events.
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
        "cutflow_stats": [],
        "stage_histograms": {},
    }

    try:
        # 1. Setup multi-threading
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

        # 4. Configure selection chain and metrics
        active_cuts = []
        if cutflow_config and "cuts" in cutflow_config:
            active_cuts = [c for c in cutflow_config["cuts"] if c.get("enabled", True)]

        # If no cuts specified in config, use raw + fiducial cut
        if not active_cuts:
            active_cuts = [
                {"id": "step0_raw", "name": "Raw / All Events", "filter": ""},
                {"id": "step1_scifi_fiducial", "name": "SciFi Average Fiducial Cut", "filter": "pass_scifi_fiducial"},
            ]

        # Define metrics observables if MuFilter is present
        has_mufi = dm.has_branch("Digi_MuFilterHits")
        if has_mufi:
            df_processed = define_preselection_metrics(df)
        else:
            df_processed = df

        # Define fiducial boolean
        df_processed = df_processed.Define("pass_scifi_fiducial", cut, ["Digi_ScifiHits"])

        # 5. Build incremental cut nodes and book histograms
        hist_configs = cutflow_config.get("histograms", []) if cutflow_config else []
        cut_nodes = []
        current_node = df_processed

        for c_info in active_cuts:
            c_id = c_info["id"]
            c_name = c_info.get("name", c_id)
            c_filter = c_info.get("filter", "").strip()

            if c_filter:
                current_node = current_node.Filter(c_filter, c_name)

            c_count = current_node.Count()
            c_hists = {}

            if record_cutflow and hist_configs:
                for h_cfg in hist_configs:
                    h_name = h_cfg["name"]
                    h_key = f"{c_id}__{h_name}"
                    h_title = h_cfg.get("title", f"{h_name};{h_cfg.get('var', '')};Events")
                    if h_cfg.get("type") == "2D":
                        h_ptr = current_node.Histo2D(
                            (h_key, h_title,
                             int(h_cfg["bins_x"]), float(h_cfg["xmin"]), float(h_cfg["xmax"]),
                             int(h_cfg["bins_y"]), float(h_cfg["ymin"]), float(h_cfg["ymax"])),
                            str(h_cfg["var_x"]), str(h_cfg["var_y"])
                        )
                    else:
                        var_name = str(h_cfg["var"])
                        h_ptr = current_node.Histo1D(
                            (h_key, h_title, int(h_cfg["bins"]), float(h_cfg["xmin"]), float(h_cfg["xmax"])),
                            var_name
                        )
                    c_hists[h_name] = h_ptr

            cut_nodes.append({
                "id": c_id,
                "name": c_name,
                "node": current_node,
                "count": c_count,
                "hists": c_hists,
            })

        # 6. Evaluate selected entry IDs from final filter stage
        final_stage = cut_nodes[-1]
        passed_entry_list = sorted(list(final_stage["node"].Take['ULong64_t']('rdfentry_').GetValue()))
        passed_entries = len(passed_entry_list)
        result["passed_entries"] = passed_entries

        if total_entries > 0:
            result["yield_pct"] = (passed_entries / total_entries) * 100.0

        # Collect cutflow counters and histograms
        cutflow_stats = []
        stage_hists = {}
        for stage in cut_nodes:
            cnt = int(stage["count"].GetValue())
            cutflow_stats.append({
                "id": stage["id"],
                "name": stage["name"],
                "count": cnt,
            })
            if record_cutflow:
                stage_hists[stage["id"]] = {}
                for h_name, h_ptr in stage["hists"].items():
                    h_obj = h_ptr.GetValue().Clone()
                    h_obj.SetDirectory(0)
                    stage_hists[stage["id"]][h_name] = h_obj

        result["cutflow_stats"] = cutflow_stats
        result["stage_histograms"] = stage_hists

        # Check if skipping empty
        if passed_entries == 0 and skip_empty:
            result["status"] = "SKIPPED_EMPTY"
            return result

        # 7. Write filtered events preserving complete tree schema
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

        # 8. Copy FairRoot auxiliary metadata objects if present
        copy_auxiliary_metadata(input_file, output_file)

        # 9. Create symlinks to auxiliary files in the input directory
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
        "--cutflow",
        type=str,
        default=None,
        help="Output ROOT file path for cutflow counters and diagnostic histograms (supports '%%s' or a single combined file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "config", "prefilter_cutflow_config.yaml"),
        help="Path to YAML configuration file defining cuts and diagnostic histograms",
    )
    parser.add_argument(
        "--enable-cuts",
        type=str,
        default=None,
        help="Cuts to enable: 'all', 'fiducial', or comma-separated list of cut IDs. Defaults to enabled flags in config.",
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

    # Load cutflow config if available
    cutflow_cfg = {}
    if os.path.exists(args.config):
        cutflow_cfg = load_cutflow_config(args.config)

    # Resolve cuts configuration
    if "cuts" in cutflow_cfg:
        for c in cutflow_cfg["cuts"]:
            # Update fiducial cut parameters if specified via CLI
            if c.get("id") == "step1_scifi_fiducial":
                c["vertical_min"] = args.vertical_min
                c["vertical_max"] = args.vertical_max
                c["horizontal_min"] = args.horizontal_min
                c["horizontal_max"] = args.horizontal_max
                c["reversed"] = args.reversed

            # Handle --enable-cuts selection
            if args.enable_cuts:
                if args.enable_cuts.lower() == "all":
                    c["enabled"] = True
                elif args.enable_cuts.lower() == "fiducial":
                    c["enabled"] = (c.get("id") in ("step0_raw", "step1_scifi_fiducial"))
                else:
                    selected_ids = [s.strip() for s in args.enable_cuts.split(",")]
                    c["enabled"] = (c.get("id") in selected_ids or c.get("id") == "step0_raw")

    # Resolve input files
    input_files = resolve_input_files(args.input, max_files=args.max_files)
    if not input_files:
        print(f"[Error] No files matched input pattern: {args.input}")
        sys.exit(1)

    # Instantiate fiducial cut
    cut = AvgScifiFiducialCut(
        vertical_min=args.vertical_min,
        vertical_max=args.vertical_max,
        horizontal_min=args.horizontal_min,
        horizontal_max=args.horizontal_max,
        reversed=args.reversed,
    )

    active_cut_list = [c["name"] for c in cutflow_cfg.get("cuts", []) if c.get("enabled", True)]

    print("=" * 80)
    print(" SND@LHC Universal Event Prefilter")
    print("=" * 80)
    print(f" Input Pattern:        {args.input}")
    print(f" Output Pattern:       {args.output}")
    print(f" Files Matched:        {len(input_files)}")
    thread_info = f"{args.jobs} threads (multi-threaded)" if args.jobs > 1 else "1 (single-threaded)"
    print(f" RDataFrame Workers:   {thread_info}")
    print(f" Cutflow Output:       {args.cutflow if args.cutflow else 'Disabled (counters only)'}")
    print(f" Config File:          {args.config}")
    print(f" Active Cuts ({len(active_cut_list)}):    {', '.join(active_cut_list)}")
    print(f"   Vertical Range:     [{args.vertical_min:.1f}, {args.vertical_max:.1f}]")
    print(f"   Horizontal Range:   [{args.horizontal_min:.1f}, {args.horizontal_max:.1f}]")
    print(f" Symlink Auxiliary:    {not args.no_symlinks}")
    print(f" Skip Empty Files:     {args.skip_empty}")
    print("=" * 80)

    start_time = time.time()
    results = []

    cumulative_cutflow_stats: List[Dict[str, Any]] = []
    cumulative_stage_histograms: Dict[str, Dict[str, Any]] = {}

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
            cutflow_config=cutflow_cfg,
            record_cutflow=(args.cutflow is not None),
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

        # Handle cutflow recording
        if args.cutflow and res["status"] in ("OK", "SKIPPED_EMPTY"):
            if "%s" in args.cutflow:
                file_cutflow_path = format_output_path(input_file, args.input, args.cutflow, index=idx)
                save_cutflow_root_file(file_cutflow_path, res["cutflow_stats"], res["stage_histograms"])
                print(f"  Cutflow: Saved {file_cutflow_path}")
            else:
                # Accumulate for single combined cutflow file
                if not cumulative_cutflow_stats:
                    cumulative_cutflow_stats = [
                        {"id": s["id"], "name": s["name"], "count": s["count"]}
                        for s in res["cutflow_stats"]
                    ]
                else:
                    for i, s in enumerate(res["cutflow_stats"]):
                        cumulative_cutflow_stats[i]["count"] += s["count"]

                merge_stage_histograms(cumulative_stage_histograms, res["stage_histograms"])

        print("-" * 80)

    total_elapsed = time.time() - start_time
    total_in = sum(r["total_entries"] for r in results)
    total_out = sum(r["passed_entries"] for r in results)
    overall_yield = (total_out / total_in * 100.0) if total_in > 0 else 0.0
    successful_files = sum(1 for r in results if r["status"] in ("OK", "SKIPPED_EMPTY"))

    # Write combined cutflow file if single filepath was requested
    if args.cutflow and "%s" not in args.cutflow and cumulative_cutflow_stats:
        save_cutflow_root_file(args.cutflow, cumulative_cutflow_stats, cumulative_stage_histograms)
        print_cutflow_summary(cumulative_cutflow_stats, args.cutflow)
    elif results and results[0].get("cutflow_stats"):
        print_cutflow_summary(results[0]["cutflow_stats"])

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
