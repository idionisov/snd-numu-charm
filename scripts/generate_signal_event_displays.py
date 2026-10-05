#!/usr/bin/env python3
"""
scripts/generate_signal_event_displays.py
-----------------------------------------
Generates 2D interactive event displays (XZ top view, YZ side view) for signal
events (numu CC charm dimuon events) from input ROOT files.

Saves TCanvas objects into an 'eventdisplay' subdirectory in output ROOT files,
allowing interactive inspection with ROOT TBrowser, and optionally exports PNG images.

Usage:
  python3 scripts/generate_signal_event_displays.py \
      -i /eos/user/i/idioniso/snd-numu-charm/signal \
      -o ./event_displays \
      -j 4 \
      --save-images
"""

import os
import sys
import glob
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to sys.path so 'snd' package can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def process_single_file(args_tuple):
    """
    Worker function executed in a separate process.
    Initializes Snd2DEventDisplay with its own TGeoManager/ROOT context.
    """
    (
        input_path,
        output_dir,
        geofile,
        save_images,
        candidates_only,
        color_by_qdc_and_density,
        max_density,
        max_qdc,
        max_events,
    ) = args_tuple

    import ROOT
    ROOT.gROOT.SetBatch(True)
    from snd import Snd2DEventDisplay


    fname = os.path.basename(input_path)
    base_name = os.path.splitext(fname)[0]

    out_root_path = os.path.join(output_dir, f"{base_name}_displays.root")
    images_dir = os.path.join(output_dir, "images")
    if save_images:
        os.makedirs(images_dir, exist_ok=True)

    result = {
        "file": fname,
        "total_entries": 0,
        "saved_displays": 0,
        "output_file": out_root_path,
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
                if hasattr(tree, "has_candidate") and not bool(tree.has_candidate):
                    continue
                # Also check has_charm or charm_pdg if has_candidate not present
                elif not hasattr(tree, "has_candidate") and hasattr(tree, "charm_pdg") and tree.charm_pdg == 0:
                    continue
            selected_entries.append(iev)

        if not selected_entries:
            fin.Close()
            return result

        # Initialize display engine
        display = Snd2DEventDisplay(
            geo_file=geofile,
            color_by_qdc_and_density=color_by_qdc_and_density,
            max_density=max_density,
            max_qdc=max_qdc,
        )

        fout = ROOT.TFile.Open(out_root_path, "RECREATE")
        subdir = fout.mkdir("eventdisplay")


        for iev in selected_entries:
            canvas_name = f"display_event_{iev}"
            canvas_title = f"SND@LHC Event {iev} ({fname})"
            canvas = display.draw_event(
                tree,
                event_idx=iev,
                canvas_name=canvas_name,
                canvas_title=canvas_title,
            )
            subdir.cd()
            canvas.Write(canvas_name)
            result["saved_displays"] += 1

            if save_images:
                img_path = os.path.join(images_dir, f"{base_name}_ev{iev}.png")
                canvas.Print(img_path)

        fout.Close()
        fin.Close()

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D interactive event displays for SND@LHC signal files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/signal",
        help="Input directory containing skimmed signal ROOT files",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./event_displays",
        help="Output directory for generated ROOT files and images",
    )
    parser.add_argument(
        "-g", "--geofile",
        type=str,
        default="/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root",
        help="Path to detector geometry ROOT file",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        help="Only display events with has_candidate=1 (default is all events in input files)",
    )
    parser.add_argument(
        "--all-events",
        action="store_true",
        default=True,
        help="Generate displays for all events in input files (default)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also export PNG images along with the ROOT files",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process",
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
        help="Disable Viridis hit density and QDC color coding, using solid colors instead",
    )
    parser.add_argument(
        "--max-density",
        type=int,
        default=40,
        help="Maximum SciFi hit density for color scale [hits/cm]",
    )
    parser.add_argument(
        "--max-qdc",
        type=float,
        default=3200.0,
        help="Maximum MuFilter QDC for color scale [QDC units]",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Locate input files
    pattern = os.path.join(args.input_dir, "*.root")
    input_files = sorted(glob.glob(pattern))

    if not input_files:
        print(f"Error: No ROOT files found in {args.input_dir}")
        sys.exit(1)

    if args.max_files is not None:
        input_files = input_files[: args.max_files]

    candidates_only = args.candidates_only
    color_qdc_dens = not args.no_qdc_density

    print("=" * 70)
    print(" SND@LHC 2D Event Display Generator")
    print("=" * 70)
    print(f" Input Directory:       {args.input_dir}")
    print(f" Files to Process:      {len(input_files)}")
    print(f" Output Directory:      {args.output_dir}")
    print(f" Geometry File:         {args.geofile}")
    print(f" Filter Mode:           {'Candidates only (has_candidate=1)' if candidates_only else 'All events'}")
    print(f" QDC / Density Colors:  {color_qdc_dens} (Max Density: {args.max_density}, Max QDC: {args.max_qdc})")
    print(f" Parallel Workers:      {args.jobs}")
    print(f" Save PNG Images:       {args.save_images}")
    print("=" * 70)

    start_time = time.time()
    task_args = [
        (
            fpath,
            args.output_dir,
            args.geofile,
            args.save_images,
            candidates_only,
            color_qdc_dens,
            args.max_density,
            args.max_qdc,
            args.max_events_per_file,
        )
        for fpath in input_files
    ]


    total_saved = 0
    total_processed = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(process_single_file, t_arg): t_arg[0] for t_arg in task_args}

        for fut in as_completed(futures):
            res = fut.result()
            total_processed += 1
            if res["status"] == "OK":
                if res["saved_displays"] > 0:
                    print(f" [{total_processed}/{len(input_files)}] {res['file']}: Saved {res['saved_displays']} displays -> {os.path.basename(res['output_file'])}")
                    total_saved += res["saved_displays"]
                else:
                    print(f" [{total_processed}/{len(input_files)}] {res['file']}: 0 candidates found.")
            else:
                print(f" [{total_processed}/{len(input_files)}] {res['file']}: ERROR: {res['error']}")

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f" Done! Saved {total_saved} event displays in {elapsed:.1f} seconds.")
    print(f" Output ROOT files located in: {os.path.abspath(args.output_dir)}")
    if args.save_images:
        print(f" Output PNG images located in: {os.path.abspath(os.path.join(args.output_dir, 'images'))}")
    print("=" * 70)


if __name__ == "__main__":
    main()
