#!/usr/bin/env python3
"""
scripts/generate_signal_event_displays.py
-----------------------------------------
Generates 2D interactive event displays (XZ top view, YZ side view) for SND@LHC events.

Operating Modes:
1. Real Data Mode (Default):
   - Truth overlays and truth kinematics are disabled.
   - Event displays are stored in top-level TDirectory named Run<####>
     (e.g. Run0027, Run100522), extracted from event.EventHeader.GetRunId().
   - Tree auto-detects 'rawConv' or 'cbmsim' (or --tree-name).

2. Monte Carlo Truth Mode (--mc-truth):
   - Truth tracks, interaction vertices, and kinematics summary tables are rendered.
   - Event displays are stored under top-level TDirectory 'MCTruth'
     (e.g. MCTruth/numu/CC/toCharm/charmToMuon/inAcceptance or .../other).

Event Selection:
- By default (no --events), runs on ALL events in the input ROOT files.
- If --events <ev1> <ev2> ... is supplied, only those event numbers or entry indices
  are processed.

Usage examples:
  # Real data default (all events, stored under Run<####>):
  python3 scripts/generate_signal_event_displays.py \
      -i /eos/experiment/sndlhc/convertedData/physics/2022/run_00100522/sndLHC.raw.root \
      -o ./event_displays/data_event_displays.root

  # Specific events only:
  python3 scripts/generate_signal_event_displays.py \
      -i /eos/user/i/idioniso/snd-numu-charm/data/27/sndLHC.Genie-TGeant4_digCPP_truth_numuCC_charm_dimuon.root \
      -o ./event_displays/selected_displays.root \
      --events 0 5 12

  # Monte Carlo truth mode with DS acceptance splitting:
  python3 scripts/generate_signal_event_displays.py \
      -i /eos/user/i/idioniso/snd-numu-charm/data \
      -o ./event_displays/signal_event_displays.root \
      --mc-truth \
      -j 8
"""

import os
import sys
import re
import time
import shutil
import tempfile
import argparse
from typing import List, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to sys.path so 'snd' package can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from snd import (
    load_config,
    resolve_input_files,
    resolve_tdirectory_hierarchy,
    get_or_create_tdirectory,
    get_event_header_number,
    get_event_header_run_id,
    get_geofile_for_run,
    count_mu2_ds_mcpoints,
)

DEFAULT_GEOFILE = "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root"


def copy_tcanvases_recursive(src_dir, dest_dir) -> None:
    """
    Recursively copies all TCanvases and directory structures from src_dir to dest_dir.
    """
    import ROOT

    for key in src_dir.GetListOfKeys():
        cls_name = key.GetClassName()
        if cls_name == "TCanvas":
            obj = key.ReadObj()
            if obj and not (hasattr(obj, "IsZombie") and obj.IsZombie()):
                dest_dir.cd()
                obj.Write(key.GetName(), ROOT.TObject.kOverwrite)
        elif "TDirectory" in cls_name:
            sub_src = key.ReadObj()
            if sub_src:
                sub_name = key.GetName()
                sub_dest = dest_dir.GetDirectory(sub_name)
                if not sub_dest:
                    sub_dest = dest_dir.mkdir(sub_name)
                copy_tcanvases_recursive(sub_src, sub_dest)


def process_single_file_worker(args_tuple):
    """
    Worker function executed in a separate process.
    Initializes Snd2DEventDisplay with its own TGeoManager/ROOT context.
    Writes event displays to a worker-local temporary ROOT file.
    """
    (
        input_path,
        run_num,
        geofile_to_use,
        temp_dir,
        canvas_name_format,
        mc_truth,
        events_filter,
        candidates_only,
        base_tdirectory_parts,
        tree_name,
        color_by_qdc_and_density,
        max_density,
        max_qdc,
        max_events,
        save_images,
        images_dir,
        split_acceptance,
        min_ds_hor_points,
        min_ds_ver_points,
        in_acc_name,
        other_name,
    ) = args_tuple

    import ROOT
    ROOT.gROOT.SetBatch(True)
    from snd import Snd2DEventDisplay, get_event_header_number, get_event_header_run_id, is_dimuon_in_ds_acceptance

    fname = os.path.basename(input_path)

    result = {
        "file": fname,
        "run_num": run_num,
        "total_entries": 0,
        "saved_displays": 0,
        "saved_in_acceptance": 0,
        "saved_other": 0,
        "temp_root": None,
        "canvas_names": [],
        "status": "OK",
        "error": None,
    }

    try:
        fin = ROOT.TFile.Open(input_path, "READ")
        if not fin or fin.IsZombie():
            result["status"] = "ERROR"
            result["error"] = f"Could not open input file {input_path}"
            return result

        # Discover TTree
        tree = None
        if tree_name:
            tree = fin.Get(tree_name)
        else:
            for candidate in ["rawConv", "cbmsim"]:
                tree = fin.Get(candidate)
                if tree:
                    break
            if not tree:
                for k in fin.GetListOfKeys():
                    if k.GetClassName() == "TTree":
                        tree = fin.Get(k.GetName())
                        break

        if not tree:
            result["status"] = "ERROR"
            result["error"] = f"No suitable TTree found in {input_path}"
            fin.Close()
            return result

        total_entries = tree.GetEntries()
        result["total_entries"] = total_entries

        # Determine events to process
        target_events_set: Optional[Set[int]] = set(events_filter) if events_filter is not None else None
        selected_entries = []

        if target_events_set is not None:
            if total_entries == 0:
                fin.Close()
                return result

            # Quick boundary check on first and last event
            tree.GetEntry(0)
            first_ev = get_event_header_number(tree, default_idx=0)
            tree.GetEntry(total_entries - 1)
            last_ev = get_event_header_number(tree, default_idx=total_entries - 1)

            min_ev = min(first_ev, last_ev)
            max_ev = max(first_ev, last_ev)

            has_index_match = any(0 <= t < total_entries for t in target_events_set)
            has_ev_match = any(min_ev <= t <= max_ev for t in target_events_set)

            if not has_index_match and not has_ev_match:
                fin.Close()
                return result

            # Probe candidate entries first (direct estimated index jump)
            candidate_entries = set()
            for target in target_events_set:
                if 0 <= target < total_entries:
                    candidate_entries.add(target)
                if min_ev <= target <= max_ev:
                    est_entry = int(target - first_ev)
                    if 0 <= est_entry < total_entries:
                        candidate_entries.add(est_entry)

            matched_targets = set()
            for cand in sorted(candidate_entries):
                tree.GetEntry(cand)
                ev_num = get_event_header_number(tree, default_idx=cand)
                if cand in target_events_set or ev_num in target_events_set:
                    if candidates_only and mc_truth:
                        if hasattr(tree, "has_candidate") and not bool(tree.has_candidate):
                            continue
                        elif hasattr(tree, "charm_pdg") and tree.charm_pdg == 0:
                            continue
                    selected_entries.append((cand, ev_num))
                    matched_targets.add(cand)
                    matched_targets.add(ev_num)

            remaining_targets = target_events_set - matched_targets
            if remaining_targets:
                for iev in range(total_entries):
                    if max_events is not None and len(selected_entries) >= max_events:
                        break
                    if iev in candidate_entries:
                        continue
                    tree.GetEntry(iev)
                    ev_num = get_event_header_number(tree, default_idx=iev)
                    if iev in remaining_targets or ev_num in remaining_targets:
                        if candidates_only and mc_truth:
                            if hasattr(tree, "has_candidate") and not bool(tree.has_candidate):
                                continue
                            elif hasattr(tree, "charm_pdg") and tree.charm_pdg == 0:
                                continue
                        selected_entries.append((iev, ev_num))
        else:
            for iev in range(total_entries):
                if max_events is not None and len(selected_entries) >= max_events:
                    break
                tree.GetEntry(iev)
                ev_num = get_event_header_number(tree, default_idx=iev)
                if candidates_only and mc_truth:
                    if hasattr(tree, "has_candidate") and not bool(tree.has_candidate):
                        continue
                    elif hasattr(tree, "charm_pdg") and tree.charm_pdg == 0:
                        continue
                selected_entries.append((iev, ev_num))

        if not selected_entries:
            fin.Close()
            return result

        # Create temporary ROOT file to store canvases for this worker
        temp_fd, temp_path = tempfile.mkstemp(suffix=".root", prefix=f"disp_r{run_num}_", dir=temp_dir)
        os.close(temp_fd)
        ftemp = ROOT.TFile.Open(temp_path, "RECREATE")

        # Initialize display engine
        display = Snd2DEventDisplay(
            geo_file=geofile_to_use,
            color_by_qdc_and_density=color_by_qdc_and_density,
            max_density=max_density,
            max_qdc=max_qdc,
        )

        for iev, event_num in selected_entries:
            tree.GetEntry(iev)
            ev_run_id = get_event_header_run_id(tree, default_run=run_num)

            # Determine base TDirectory hierarchy
            if base_tdirectory_parts is not None:
                base_parts = list(base_tdirectory_parts)
            else:
                # Real data mode default: Run<####>
                if isinstance(ev_run_id, int) and 0 <= ev_run_id < 10000:
                    run_dir_name = f"Run{ev_run_id:04d}"
                else:
                    run_dir_name = f"Run{ev_run_id}"
                base_parts = [run_dir_name]

            # Determine acceptance partition (MC mode only)
            if mc_truth and split_acceptance:
                is_in_acceptance = is_dimuon_in_ds_acceptance(
                    tree,
                    min_hor_points=min_ds_hor_points,
                    min_ver_points=min_ds_ver_points,
                )
                category = in_acc_name if is_in_acceptance else other_name
                target_parts = base_parts + [category]
            else:
                is_in_acceptance = False
                category = base_parts[-1]
                target_parts = base_parts

            canvas_name = canvas_name_format.format(run_num=ev_run_id, event_num=event_num)
            if mc_truth and split_acceptance:
                canvas_title = f"SND@LHC Run {ev_run_id} Event {event_num} ({'inAcceptance' if is_in_acceptance else 'other'})"
            else:
                canvas_title = f"SND@LHC Run {ev_run_id} Event {event_num}"

            canvas = display.draw_event(
                tree,
                event_idx=iev,
                canvas_name=canvas_name,
                canvas_title=canvas_title,
                run_number=ev_run_id,
                event_number=event_num,
                show_mc_truth=mc_truth,
            )

            target_tdir = get_or_create_tdirectory(ftemp, target_parts)
            target_tdir.cd()
            canvas.Write(canvas_name)

            result["saved_displays"] += 1
            if mc_truth and split_acceptance:
                if is_in_acceptance:
                    result["saved_in_acceptance"] += 1
                else:
                    result["saved_other"] += 1
            result["canvas_names"].append((canvas_name, category))

            if save_images:
                sub_img_dir = os.path.join(images_dir, *target_parts)
                os.makedirs(sub_img_dir, exist_ok=True)
                img_path = os.path.join(sub_img_dir, f"{canvas_name}.png")
                canvas.Print(img_path)

        ftemp.Write()
        ftemp.Close()
        fin.Close()
        result["temp_root"] = temp_path

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)

    return result


def main():
    default_config_path = os.path.join(PROJECT_ROOT, "config", "filter_numu_charm_config.yaml")

    parser = argparse.ArgumentParser(
        description="Generate 2D interactive event displays for SND@LHC events (real data or MC truth).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=default_config_path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Input dataset directory (containing partition subdirs 0..399), file pattern, or single ROOT file",
    )
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        default=None,
        help="Output single ROOT file containing all event display TCanvases",
    )
    parser.add_argument(
        "-g", "--geofile",
        type=str,
        default=None,
        help="Custom detector geometry ROOT file (default: auto-detected from SNDSW geo_paths.csv by run/year or local simulation geofile)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--mc-truth",
        action="store_true",
        default=False,
        help="Enable Monte Carlo truth overlay (tracks, vertices, kinematics table) and default top TDirectory 'MCTruth'",
    )
    parser.add_argument(
        "--events",
        nargs="+",
        type=int,
        default=None,
        help="Specific event numbers or entry indices to display (default: all events)",
    )
    parser.add_argument(
        "--tree-name",
        type=str,
        default=None,
        help="Explicit name of the TTree to read (auto-detects 'rawConv' or 'cbmsim' if omitted)",
    )
    parser.add_argument(
        "--tdirectory",
        type=str,
        default=None,
        help="Custom base TDirectory hierarchy (overrides default Run<####> or MCTruth/...)",
    )
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        default=False,
        help="Only display candidate events (has_candidate=1 in MC)",
    )
    parser.add_argument(
        "--min-ds-points",
        type=int,
        default=None,
        help="Minimum number of DS MCPoints for muons to be inAcceptance (default: 3)",
    )
    parser.add_argument(
        "--min-ds-hor-points",
        type=int,
        default=None,
        help="Minimum number of horizontal plane DS MCPoints for muons to be inAcceptance (default: 3)",
    )
    parser.add_argument(
        "--min-ds-ver-points",
        type=int,
        default=None,
        help="Minimum number of vertical plane DS MCPoints for muons to be inAcceptance (default: 3)",
    )
    parser.add_argument(
        "--no-acceptance-split",
        action="store_true",
        help="Disable splitting into inAcceptance and other subdirectories",
    )
    parser.add_argument(
        "--in-acceptance-dir",
        type=str,
        default=None,
        help="Subdirectory name for events in acceptance (default: 'inAcceptance')",
    )
    parser.add_argument(
        "--other-dir",
        type=str,
        default=None,
        help="Subdirectory name for events not in acceptance (default: 'other')",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=None,
        help="Also export PNG images along with the ROOT file",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory to save exported PNG images",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of input files to process",
    )
    parser.add_argument(
        "--max-events-per-file",
        type=int,
        default=None,
        help="Maximum events to display per file",
    )
    parser.add_argument(
        "--no-qdc-density",
        action="store_true",
        help="Disable hit density and QDC color coding, using solid colors instead",
    )
    parser.add_argument(
        "--max-density",
        type=int,
        default=None,
        help="Maximum SciFi hit density for color scale [hits/cm]",
    )
    parser.add_argument(
        "--max-qdc",
        type=float,
        default=None,
        help="Maximum MuFilter QDC for color scale [QDC units]",
    )

    args = parser.parse_args()

    # Load configuration
    cfg = {}
    if os.path.exists(args.config):
        cfg = load_config(args.config)
    disp_cfg = cfg.get("event_displays", {})

    # Determine parameter values
    input_path = args.input if args.input is not None else disp_cfg.get("input_dir", "/eos/user/i/idioniso/snd-numu-charm/data")
    output_file = args.output_file if args.output_file is not None else disp_cfg.get("output_file", "event_displays/signal_event_displays.root")
    canvas_name_format = disp_cfg.get("canvas_name_format", "c_run{run_num}_ev{event_num}")

    candidates_only = args.candidates_only

    # Acceptance subdirectories config (only relevant in MC mode)
    acc_cfg = disp_cfg.get("acceptance_subdirs", {})
    if not args.mc_truth or args.no_acceptance_split:
        split_acceptance = False
    else:
        split_acceptance = bool(acc_cfg.get("enabled", True))

    min_ds_points = args.min_ds_points if args.min_ds_points is not None else int(acc_cfg.get("min_ds_points", 3))
    min_ds_hor_points = args.min_ds_hor_points if args.min_ds_hor_points is not None else int(acc_cfg.get("min_ds_hor_points", min_ds_points))
    min_ds_ver_points = args.min_ds_ver_points if args.min_ds_ver_points is not None else int(acc_cfg.get("min_ds_ver_points", min_ds_points))
    in_acc_name = args.in_acceptance_dir or acc_cfg.get("in_acceptance", "inAcceptance")
    other_name = args.other_dir or acc_cfg.get("other", "other")

    save_images = args.save_images if args.save_images is not None else bool(disp_cfg.get("save_images", False))
    images_dir = args.images_dir if args.images_dir is not None else disp_cfg.get("images_dir", "event_displays/images")
    if not os.path.isabs(images_dir):
        images_dir = os.path.abspath(images_dir)

    color_by_qdc_and_density = not args.no_qdc_density if args.no_qdc_density else bool(disp_cfg.get("color_by_qdc_and_density", True))
    max_density = args.max_density if args.max_density is not None else int(disp_cfg.get("max_density", 40))
    max_qdc = args.max_qdc if args.max_qdc is not None else float(disp_cfg.get("max_qdc", 3200.0))

    # Resolve base TDirectory hierarchy
    if args.tdirectory:
        base_tdirectory_parts = [p.strip() for p in args.tdirectory.replace("->", "/").split("/") if p.strip()]
        hierarchy_display_str = "/".join(base_tdirectory_parts)
    elif args.mc_truth:
        base_tdirectory_parts = resolve_tdirectory_hierarchy(cfg, top_name="MCTruth")
        hierarchy_display_str = "/".join(base_tdirectory_parts)
    else:
        # Real data default: dynamically determined per-event as Run<####>
        base_tdirectory_parts = None
        hierarchy_display_str = "Run<####> (per event run number)"

    # Locate input files
    input_files = resolve_input_files(input_path, max_files=args.max_files or -1)

    print("=" * 78)
    print(" SND@LHC 2D Event Display Generator")
    print("=" * 78)
    print(f" Mode:                 {'Monte Carlo Simulation (--mc-truth)' if args.mc_truth else 'Real Data (Default)'}")
    print(f" Config File:          {args.config}")
    print(f" Input Target:         {input_path}")
    print(f" Files Matched:        {len(input_files)}")
    print(f" Single Output ROOT:   {os.path.abspath(output_file)}")
    print(f" Base TDirectory:      {hierarchy_display_str}")
    if args.events:
        print(f" Events Filter:        {args.events}")
    else:
        print(f" Events Filter:        All events")
    if split_acceptance:
        print(f" Acceptance Splitting: Enabled (Min DS points: hor>={min_ds_hor_points}, ver>={min_ds_ver_points})")
        print(f"   In-Acceptance Dir:  {hierarchy_display_str}/{in_acc_name}")
        print(f"   Other Dir:          {hierarchy_display_str}/{other_name}")
    else:
        print(f" Acceptance Splitting: Disabled")
    print(f" Canvas Name Format:   {canvas_name_format}")
    print(f" Filter Candidates:    {'Candidates only (has_candidate=1)' if candidates_only else 'All events'}")
    print(f" QDC / Density Colors: {color_by_qdc_and_density} (Max Dens: {max_density}, Max QDC: {max_qdc})")
    print(f" Parallel Workers:     {args.jobs}")
    print(f" Save PNG Images:      {save_images}" + (f" -> {images_dir}" if save_images else ""))
    print("=" * 78)

    # Prepare temp directory for worker temporary ROOT files
    temp_dir = tempfile.mkdtemp(prefix="snd_event_displays_")

    # Build tasks for workers
    tasks = []
    for fpath in input_files:
        # Determine fallback run number from parent directory (e.g. partition folder '0'..'399' or 'run_006596')
        parent_name = os.path.basename(os.path.dirname(os.path.abspath(fpath)))
        if parent_name.isdigit():
            run_num = parent_name
        else:
            m = re.search(r'(?:run|signal)?[_-]?(\d+)', os.path.basename(fpath))
            if not m:
                m = re.search(r'run_0*(\d+)', fpath)
            run_num = m.group(1) if m else parent_name

        # Resolve detector geometry using SNDSW geo_paths.csv mapping or local simulation geofile
        geofile_to_use = get_geofile_for_run(
            run_number=run_num,
            input_path=fpath,
            default_geofile=args.geofile,
        )

        tasks.append((
            fpath,
            run_num,
            geofile_to_use,
            temp_dir,
            canvas_name_format,
            args.mc_truth,
            args.events,
            candidates_only,
            base_tdirectory_parts,
            args.tree_name,
            color_by_qdc_and_density,
            max_density,
            max_qdc,
            args.max_events_per_file,
            save_images,
            images_dir,
            split_acceptance,
            min_ds_hor_points,
            min_ds_ver_points,
            in_acc_name,
            other_name,
        ))

    total_saved = 0
    total_in_acc = 0
    total_other = 0
    total_processed = 0
    start_time = time.time()
    worker_results = []

    try:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(process_single_file_worker, t): t[0] for t in tasks}

            for fut in as_completed(futures):
                res = fut.result()
                total_processed += 1
                worker_results.append(res)

                if res["status"] == "OK":
                    saved_in_file = res["saved_displays"]
                    if saved_in_file > 0:
                        if split_acceptance:
                            print(f" [{total_processed}/{len(input_files)}] {res['file']} (run {res['run_num']}): "
                                  f"+{saved_in_file} displays ({in_acc_name}: {res['saved_in_acceptance']}, {other_name}: {res['saved_other']})")
                        else:
                            print(f" [{total_processed}/{len(input_files)}] {res['file']} (run {res['run_num']}): "
                                  f"+{saved_in_file} displays")
                    else:
                        print(f" [{total_processed}/{len(input_files)}] {res['file']} (run {res['run_num']}): 0 events selected.")
                else:
                    print(f" [{total_processed}/{len(input_files)}] {res['file']}: ERROR: {res['error']}")

        # Merge worker outputs into single final ROOT file after all workers finish
        import ROOT
        ROOT.gROOT.SetBatch(True)

        out_root_abs = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(out_root_abs), exist_ok=True)
        fout = ROOT.TFile.Open(out_root_abs, "RECREATE")

        for res in worker_results:
            if res["status"] != "OK":
                continue
            temp_root = res.get("temp_root")
            saved_in_file = res.get("saved_displays", 0)
            if saved_in_file > 0 and temp_root and os.path.exists(temp_root):
                ftemp = ROOT.TFile.Open(temp_root, "READ")
                if ftemp and not ftemp.IsZombie():
                    copy_tcanvases_recursive(ftemp, fout)
                    ftemp.Close()

                try:
                    os.remove(temp_root)
                except OSError:
                    pass

                total_saved += saved_in_file
                total_in_acc += res["saved_in_acceptance"]
                total_other += res["saved_other"]

        # Flush and close single output ROOT file
        fout.Write()
        fout.Close()

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed = time.time() - start_time
    print("=" * 78)
    print(f" Completed! Saved {total_saved} event displays in {elapsed:.1f} seconds.")
    print(f" Single Output ROOT File: {out_root_abs}")
    print(f" Hierarchy:               {hierarchy_display_str}")
    if split_acceptance:
        print(f"   -> {hierarchy_display_str}/{in_acc_name}: {total_in_acc} displays")
        print(f"   -> {hierarchy_display_str}/{other_name}:        {total_other} displays")
    if save_images:
        print(f" Exported PNG Images:     {images_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
