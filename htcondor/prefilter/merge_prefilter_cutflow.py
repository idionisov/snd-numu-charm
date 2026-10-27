#!/usr/bin/env python3
"""
htcondor/prefilter/merge_prefilter_cutflow.py
---------------------------------------------
Merges per-partition cutflow ROOT files and failed-cuts CSV files into
the master cutflow ROOT file and CSV file.

Usage:
  python3 merge_prefilter_cutflow.py
"""

import os
import glob
import subprocess

CUTFLOW_PARTS = "/eos/user/i/idioniso/snd-numu-charm/cutflows/newP8/partitions/cutflow_essentials_part*.root"
TARGET_CUTFLOW = "/eos/user/i/idioniso/snd-numu-charm/cutflows/cutflow_essentials_newP8.root"

CSV_PARTS = "/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8/partitions/failed_cuts_part*.txt"
TARGET_CSV = "/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8/failed_cuts.txt"


def merge_root_cutflows():
    root_files = sorted(glob.glob(CUTFLOW_PARTS))
    if not root_files:
        print(f"[Cutflow] No partition ROOT files found matching: {CUTFLOW_PARTS}")
        return
    print(f"[Cutflow] Merging {len(root_files)} partition cutflow ROOT files into: {TARGET_CUTFLOW}")
    os.makedirs(os.path.dirname(TARGET_CUTFLOW), exist_ok=True)
    cmd = ["hadd", "-f", TARGET_CUTFLOW] + root_files
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[Cutflow] Successfully merged cutflow ROOT file ({os.path.getsize(TARGET_CUTFLOW)} bytes)!")
    else:
        print(f"[Cutflow] ERROR during hadd:\n{res.stderr}")


def merge_csvs():
    csv_files = sorted(glob.glob(CSV_PARTS))
    if not csv_files:
        print(f"[CSV] No partition CSV files found matching: {CSV_PARTS}")
        return
    print(f"[CSV] Concatenating {len(csv_files)} partition CSV files into: {TARGET_CSV}")
    os.makedirs(os.path.dirname(TARGET_CSV), exist_ok=True)
    
    header_written = False
    with open(TARGET_CSV, "w") as f_out:
        for csv_file in csv_files:
            with open(csv_file, "r") as f_in:
                for line in f_in:
                    if line.startswith("partition,event_number"):
                        if not header_written:
                            f_out.write(line)
                            header_written = True
                    else:
                        f_out.write(line)
    print(f"[CSV] Successfully concatenated failed cuts CSV ({os.path.getsize(TARGET_CSV)} bytes)!")


if __name__ == "__main__":
    merge_root_cutflows()
    merge_csvs()
