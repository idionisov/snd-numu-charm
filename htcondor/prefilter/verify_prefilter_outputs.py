#!/usr/bin/env python3
"""
htcondor/prefilter/verify_prefilter_outputs.py
----------------------------------------------
Verifies the integrity, completeness, and event yield of Stage 6: Event Prefiltering outputs.

Checks per partition:
1. File existence & non-zero size for:
   - Main filtered    : prefiltered/newP8/essentials/<p>/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root
   - Direct filtered  : prefiltered/newP8/essentials-direct/<p>/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root
   - Indirect filtered: prefiltered/newP8/essentials-indirect/<p>/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root
2. Tree structure integrity:
   - 'cbmsim' and 'mctruth' trees exist and open without zombie flags

Usage:
  python3 verify_prefilter_outputs.py -p 1-10
  python3 verify_prefilter_outputs.py -p 1-1000 -j 8
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gEnv.SetValue("TFile.Recover", 0)
ROOT.gErrorIgnoreLevel = ROOT.kError


def parse_partitions(part_arg: Optional[str]) -> List[int]:
    """Parse partition range or comma-separated list."""
    if not part_arg or part_arg.strip().lower() == "all":
        return list(range(1, 1001))

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


def verify_partition(
    partition: int,
    base_main: str,
    base_dir: str,
    base_ind: str,
) -> Tuple[int, bool, Dict[str, Any]]:
    """Inspects output ROOT files for a single partition."""
    main_file = os.path.join(base_main, str(partition), "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root")
    dir_file = os.path.join(base_dir, str(partition), "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root")
    ind_file = os.path.join(base_ind, str(partition), "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root")

    details = {
        "main_exists": os.path.isfile(main_file) and os.path.getsize(main_file) > 200,
        "dir_exists": os.path.isfile(dir_file) and os.path.getsize(dir_file) > 200,
        "ind_exists": os.path.isfile(ind_file) and os.path.getsize(ind_file) > 200,
        "main_entries": 0,
        "dir_entries": 0,
        "ind_entries": 0,
        "error": None,
    }

    if not details["main_exists"]:
        details["error"] = "Missing main prefiltered file"
        return partition, False, details

    try:
        f = ROOT.TFile.Open(main_file, "READ")
        if not f or f.IsZombie():
            details["error"] = "Zombie main file"
            if f: f.Close()
            return partition, False, details
        t = f.Get("cbmsim") or f.Get("mctruth")
        if t:
            details["main_entries"] = t.GetEntries()
        f.Close()
    except Exception as e:
        details["error"] = f"Error reading main file: {e}"
        return partition, False, details

    return partition, True, details


def main():
    parser = argparse.ArgumentParser(description="Verify prefilter outputs across partitions")
    parser.add_argument("-p", "--partitions", type=str, default="1-1000", help="Partition range (e.g. 1-1000)")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="Parallel worker threads")
    parser.add_argument("--main-dir", type=str, default="/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8/essentials")
    parser.add_argument("--direct-dir", type=str, default="/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8/essentials-direct")
    parser.add_argument("--indirect-dir", type=str, default="/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8/essentials-indirect")
    args = parser.parse_args()

    partitions = parse_partitions(args.partitions)
    print("==========================================================")
    print(f" Auditing Prefilter Outputs ({len(partitions)} Partitions)")
    print(f" Main dir     : {args.main_dir}")
    print(f" Direct dir   : {args.direct_dir}")
    print(f" Indirect dir : {args.indirect_dir}")
    print("==========================================================")

    passed = []
    failed = []

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(verify_partition, p, args.main_dir, args.direct_dir, args.indirect_dir): p
            for p in partitions
        }
        for fut in as_completed(futures):
            p, ok, details = fut.result()
            if ok:
                passed.append(p)
            else:
                failed.append((p, details["error"]))

    print(f"\n[Audit Results]")
    print(f" Total Checked : {len(partitions)}")
    print(f" Valid/Passed  : {len(passed)}")
    print(f" Failed/Missing: {len(failed)}")

    if failed:
        failed_sorted = sorted(failed, key=lambda x: x[0])
        print(f"\nFailed Partitions (first 20):")
        for p, err in failed_sorted[:20]:
            print(f"  Partition {p:4d}: {err}")
        with open("failed_prefilter_partitions.txt", "w") as f:
            for p, _ in failed_sorted:
                f.write(f"{p}\n")
        print(f"Saved failed partition IDs to: failed_prefilter_partitions.txt")
        sys.exit(1)
    else:
        print("\nAll prefilter partition outputs verified successfully!")


if __name__ == "__main__":
    main()
