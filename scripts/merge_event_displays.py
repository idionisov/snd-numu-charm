#!/usr/bin/env python3
"""
scripts/merge_event_displays.py
--------------------------------
Merges event display ROOT files from multiple partitions into one single ROOT file,
preserving the entire TDirectory hierarchy and all TCanvases.

Usage:
  # Using preset:
  python3 scripts/merge_event_displays.py --preset nu14_2022
  python3 scripts/merge_event_displays.py --preset old_100fb_2022

  # Custom paths:
  python3 scripts/merge_event_displays.py \
    -i "/eos/user/i/idioniso/snd-numu-charm/event_displays/partitions/*.root" \
    -o "/eos/user/i/idioniso/snd-numu-charm/event_displays/numu_dimuon_signal_event_displays.root"
"""

import os
import sys
import glob
import re
import time
import argparse

# Add repo root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from snd import copy_tcanvases_recursive

DEFAULT_PRESETS = {
    "nu14_2022": {
        "input": "/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/partitions/*.root",
        "output": "/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_displays.root",
    },
    "old_100fb_2022": {
        "input": "/eos/user/i/idioniso/snd-numu-charm/event_displays/old_sndlhc_100fb-1_2022_down_volTarget/partitions/*.root",
        "output": "/eos/user/i/idioniso/snd-numu-charm/event_displays/old_sndlhc_100fb-1_2022_down_volTarget_displays.root",
    },
    "100fb": {
        "input": "/eos/user/i/idioniso/snd-numu-charm/event_displays/old_sndlhc_100fb-1_2022_down_volTarget/partitions/*.root",
        "output": "/eos/user/i/idioniso/snd-numu-charm/event_displays/old_sndlhc_100fb-1_2022_down_volTarget_displays.root",
    },
}


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Merge per-partition event display ROOT files into a single master ROOT file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=list(DEFAULT_PRESETS.keys()),
        default=None,
        help="Preset dataset configuration (e.g. nu14_2022, old_100fb_2022)",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Input path pattern (wildcard '*') or directory containing partition event display ROOT files",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output single ROOT file to save all merged event displays",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append to existing output ROOT file instead of recreating",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    import ROOT
    ROOT.gROOT.SetBatch(True)

    if args.preset and args.preset in DEFAULT_PRESETS:
        cfg = DEFAULT_PRESETS[args.preset]
        input_pattern = args.input or cfg["input"]
        output_file = args.output or cfg["output"]
    else:
        input_pattern = args.input or "/eos/user/i/idioniso/snd-numu-charm/event_displays/partitions/*.root"
        output_file = args.output or "/eos/user/i/idioniso/snd-numu-charm/event_displays/numu_dimuon_signal_event_displays.root"

    if os.path.isdir(input_pattern):
        input_files = sorted(glob.glob(os.path.join(input_pattern, "*.root")), key=natural_sort_key)
        if not input_files:
            input_files = sorted(glob.glob(os.path.join(input_pattern, "*", "*.root")), key=natural_sort_key)
    elif "*" in input_pattern:
        input_files = sorted(glob.glob(input_pattern), key=natural_sort_key)
    elif "%s" in input_pattern:
        input_files = []
        for p in range(1001):
            cand = input_pattern % str(p)
            if os.path.isfile(cand):
                input_files.append(cand)
    else:
        if os.path.isfile(input_pattern):
            input_files = [input_pattern]
        else:
            input_files = []

    if not input_files:
        print(f"[Warning] No input files found matching: {input_pattern}")
        sys.exit(0)

    output_file = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    mode = "UPDATE" if (args.append and os.path.exists(output_file)) else "RECREATE"
    print("=" * 78)
    print(" SND@LHC Event Display Merger")
    print("=" * 78)
    if args.preset:
        print(f" Preset Dataset    : {args.preset}")
    print(f" Input Files Found : {len(input_files)}")
    print(f" Output File       : {output_file} (mode: {mode})")
    print("=" * 78)

    fout = ROOT.TFile.Open(output_file, mode)
    if not fout or fout.IsZombie():
        print(f"[Error] Failed to open output file: {output_file}")
        sys.exit(1)

    t0 = time.time()
    total_canvases = 0
    merged_files = 0
    empty_skipped = 0

    for idx, fpath in enumerate(input_files, start=1):
        if not os.path.isfile(fpath) or os.path.getsize(fpath) < 1000:
            empty_skipped += 1
            continue

        fin = ROOT.TFile.Open(fpath, "READ")
        if not fin or fin.IsZombie():
            if fin:
                fin.Close()
            continue

        copied = copy_tcanvases_recursive(fin, fout)
        fin.Close()

        if copied > 0:
            total_canvases += copied
            merged_files += 1
            print(f" [{idx:>4}/{len(input_files)}] {os.path.basename(fpath)}: +{copied} display(s)")
        else:
            empty_skipped += 1

    fout.Write()
    fout.Close()

    elapsed = time.time() - t0
    print("=" * 78)
    print(f" Merge Complete in {elapsed:.1f}s!")
    print(f" Merged Files    : {merged_files} (non-empty) / {len(input_files)} total")
    print(f" Skipped Empty   : {empty_skipped}")
    print(f" Total Displays  : {total_canvases}")
    print(f" Output File     : {output_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()
