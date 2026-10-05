#!/usr/bin/env python3
"""
scripts/generate_signal_event_displays.py
-----------------------------------------
Generates 2D interactive event displays (XZ top view, YZ side view) for signal
events (numu CC charm dimuon events) from input ROOT files.

Saves all generated TCanvas objects into a single output ROOT file within a
hierarchical TDirectory structure (e.g. neutrinoEvents/numu/CC/toCharm/charmToMuon),
allowing interactive inspection with ROOT TBrowser, and optionally exports PNG images.

Canvas naming format:
  c_run{run_num}_ev{event_num}
  where run_num is derived from the parent directory (e.g. 0 to 399) and
  event_num is extracted from tree.EventHeader.GetEventNumber() / GetMCEntryNumber().

Usage examples:
  # Using defaults from config/filter_numu_charm_config.yaml:
  python3 scripts/generate_signal_event_displays.py

  # Processing specific partition or directory:
  python3 scripts/generate_signal_event_displays.py \
      -i /eos/user/i/idioniso/snd-numu-charm/data/0 \
      -o ./event_displays/signal_event_displays.root \
      -j 4

  # Processing all 400 EOS partitions with PNG image export:
  python3 scripts/generate_signal_event_displays.py \
      -i /eos/user/i/idioniso/snd-numu-charm/data \
      -o /eos/user/i/idioniso/snd-numu-charm/signal/signal_event_displays.root \
      -j 8 \
      --save-images
"""

import os
import sys
import re
import time
import shutil
import tempfile
import argparse
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
)

DEFAULT_GEOFILE = "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root"


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
        candidates_only,
        color_by_qdc_and_density,
        max_density,
        max_qdc,
        max_events,
        save_images,
        images_dir,
    ) = args_tuple

    import ROOT
    ROOT.gROOT.SetBatch(True)
    from snd import Snd2DEventDisplay, get_event_header_number

    fname = os.path.basename(input_path)

    result = {
        "file": fname,
        "run_num": run_num,
        "total_entries": 0,
        "saved_displays": 0,
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

        tree = fin.Get("cbmsim")
        if not tree:
            result["status"] = "ERROR"
            result["error"] = f"'cbmsim' tree not found in {input_path}"
            fin.Close()
            return result

        total_entries = tree.GetEntries()
        result["total_entries"] = total_entries

        # Determine events to process
        selected_entries = []
        for iev in range(total_entries):
            if max_events is not None and len(selected_entries) >= max_events:
                break
            tree.GetEntry(iev)
            if candidates_only:
                # Check has_candidate branch
                if hasattr(tree, "has_candidate"):
                    if not bool(tree.has_candidate):
                        continue
                # Fallback to charm_pdg if has_candidate not present
                elif hasattr(tree, "charm_pdg") and tree.charm_pdg == 0:
                    continue
            selected_entries.append(iev)

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

        if save_images:
            os.makedirs(images_dir, exist_ok=True)

        for iev in selected_entries:
            tree.GetEntry(iev)
            event_num = get_event_header_number(tree, default_idx=iev)

            canvas_name = canvas_name_format.format(run_num=run_num, event_num=event_num)
            canvas_title = f"SND@LHC Run {run_num} Event {event_num}"

            canvas = display.draw_event(
                tree,
                event_idx=iev,
                canvas_name=canvas_name,
                canvas_title=canvas_title,
                run_number=run_num,
                event_number=event_num,
            )

            ftemp.cd()
            canvas.Write(canvas_name)
            result["saved_displays"] += 1
            result["canvas_names"].append(canvas_name)

            if save_images:
                img_path = os.path.join(images_dir, f"{canvas_name}.png")
                canvas.Print(img_path)

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
        description="Generate 2D interactive event displays for SND@LHC signal events into a single hierarchical ROOT file.",
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
        default=DEFAULT_GEOFILE,
        help="Fallback path to detector geometry ROOT file (local geofile in input dir is used if present)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--tdirectory",
        type=str,
        default=None,
        help="Custom TDirectory hierarchy (e.g. neutrinoEvents/MuonNeutrinoEvents/... or arrow -> separated)",
    )
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        default=None,
        help="Only display candidate events (has_candidate=1)",
    )
    parser.add_argument(
        "--all-events",
        action="store_true",
        help="Display all events in input files (overrides candidates_only)",
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

    # Determine parameter values (CLI arguments override config defaults)
    input_path = args.input if args.input is not None else disp_cfg.get("input_dir", "/eos/user/i/idioniso/snd-numu-charm/data")
    output_file = args.output_file if args.output_file is not None else disp_cfg.get("output_file", "event_displays/signal_event_displays.root")
    canvas_name_format = disp_cfg.get("canvas_name_format", "c_run{run_num}_ev{event_num}")

    if args.all_events:
        candidates_only = False
    elif args.candidates_only is not None:
        candidates_only = args.candidates_only
    else:
        candidates_only = bool(disp_cfg.get("candidates_only", True))

    save_images = args.save_images if args.save_images is not None else bool(disp_cfg.get("save_images", False))
    images_dir = args.images_dir if args.images_dir is not None else disp_cfg.get("images_dir", "event_displays/images")
    if not os.path.isabs(images_dir):
        images_dir = os.path.abspath(images_dir)

    color_by_qdc_and_density = not args.no_qdc_density if args.no_qdc_density else bool(disp_cfg.get("color_by_qdc_and_density", True))
    max_density = args.max_density if args.max_density is not None else int(disp_cfg.get("max_density", 40))
    max_qdc = args.max_qdc if args.max_qdc is not None else float(disp_cfg.get("max_qdc", 3200.0))

    # Resolve TDirectory hierarchy
    if args.tdirectory:
        hierarchy = [p.strip() for p in args.tdirectory.replace("->", "/").split("/") if p.strip()]
    else:
        hierarchy = resolve_tdirectory_hierarchy(cfg)
    hierarchy_str = "/".join(hierarchy)

    # Locate input files
    input_files = resolve_input_files(input_path, max_files=args.max_files or -1)

    print("=" * 78)
    print(" SND@LHC 2D Event Display Generator -> Single ROOT File")
    print("=" * 78)
    print(f" Config File:          {args.config}")
    print(f" Input Target:         {input_path}")
    print(f" Files Matched:        {len(input_files)}")
    print(f" Single Output ROOT:   {os.path.abspath(output_file)}")
    print(f" TDirectory Hierarchy: {hierarchy_str}")
    print(f" Canvas Name Format:   {canvas_name_format}")
    print(f" Filter Mode:          {'Candidates only (has_candidate=1)' if candidates_only else 'All events'}")
    print(f" QDC / Density Colors: {color_by_qdc_and_density} (Max Dens: {max_density}, Max QDC: {max_qdc})")
    print(f" Parallel Workers:     {args.jobs}")
    print(f" Save PNG Images:      {save_images}" + (f" -> {images_dir}" if save_images else ""))
    print("=" * 78)

    # Prepare temp directory for worker temporary ROOT files
    temp_dir = tempfile.mkdtemp(prefix="snd_event_displays_")

    # Build tasks for workers
    tasks = []
    for fpath in input_files:
        # Determine run number from parent directory (e.g. partition folder '0'..'399')
        parent_name = os.path.basename(os.path.dirname(os.path.abspath(fpath)))
        if parent_name.isdigit():
            run_num = parent_name
        else:
            m = re.search(r'(?:run|signal)?[_-]?(\d+)', os.path.basename(fpath))
            run_num = m.group(1) if m else parent_name

        # Detect partition-local geofile
        local_geofile = os.path.join(os.path.dirname(os.path.abspath(fpath)), "geofile_full.Genie-TGeant4.root")
        geofile_to_use = local_geofile if os.path.exists(local_geofile) else args.geofile

        tasks.append((
            fpath,
            run_num,
            geofile_to_use,
            temp_dir,
            canvas_name_format,
            candidates_only,
            color_by_qdc_and_density,
            max_density,
            max_qdc,
            args.max_events_per_file,
            save_images,
            images_dir,
        ))

    # Initialize single output ROOT file and TDirectory hierarchy
    import ROOT
    ROOT.gROOT.SetBatch(True)

    out_root_abs = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(out_root_abs), exist_ok=True)
    fout = ROOT.TFile.Open(out_root_abs, "RECREATE")
    target_tdir = get_or_create_tdirectory(fout, hierarchy)

    total_saved = 0
    total_processed = 0
    start_time = time.time()

    try:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(process_single_file_worker, t): t[0] for t in tasks}

            for fut in as_completed(futures):
                res = fut.result()
                total_processed += 1

                if res["status"] == "OK":
                    temp_root = res["temp_root"]
                    saved_in_file = res["saved_displays"]
                    if saved_in_file > 0 and temp_root and os.path.exists(temp_root):
                        # Merge canvases from temp ROOT file into single output ROOT file
                        ftemp = ROOT.TFile.Open(temp_root, "READ")
                        if ftemp and not ftemp.IsZombie():
                            for key in ftemp.GetListOfKeys():
                                obj = key.ReadObj()
                                target_tdir.cd()
                                obj.Write(key.GetName())
                            ftemp.Close()
                        try:
                            os.remove(temp_root)
                        except OSError:
                            pass

                        print(f" [{total_processed}/{len(input_files)}] {res['file']} (run {res['run_num']}): "
                              f"+{saved_in_file} displays -> {hierarchy_str}/[{res['canvas_names'][0]}...]")
                        total_saved += saved_in_file
                    else:
                        print(f" [{total_processed}/{len(input_files)}] {res['file']} (run {res['run_num']}): 0 candidates found.")
                else:
                    print(f" [{total_processed}/{len(input_files)}] {res['file']}: ERROR: {res['error']}")

        # Flush and close the single output ROOT file
        fout.Write()
        fout.Close()

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed = time.time() - start_time
    print("=" * 78)
    print(f" Completed! Saved {total_saved} event displays in {elapsed:.1f} seconds.")
    print(f" Single Output ROOT File: {out_root_abs}")
    print(f" TDirectory Hierarchy:    {hierarchy_str}")
    if save_images:
        print(f" Exported PNG Images:     {images_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
