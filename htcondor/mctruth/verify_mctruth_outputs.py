#!/usr/bin/env python3
"""
verify_mctruth_outputs.py
-------------------------
Audits completed Stage 3 MCEB truth-enriched files on EOS:
  - sndLHC.Genie-TGeant4_MCEB.root

Verifies:
  1. cbmsim;1 exists (original MCEB cycle)
  2. cbmsim;2 exists (contains unpruned MCTrack, ScifiPoint, MuFilterPoint)
  3. mctruth;1 tree exists (flat categorized truth observables)

Outputs missing or corrupted partitions into failed_mctruth_partitions.txt.

Usage:
    python3 verify_mctruth_outputs.py [-d /eos/.../data/newP8] [-p 1-1000]
"""

import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor

import ROOT
ROOT.gROOT.SetBatch(True)


def parse_partitions(part_arg: str):
    if not part_arg:
        return None
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


def check_partition(base_dir: str, partition: int, filename: str = "sndLHC.Genie-TGeant4_MCEB.root"):
    p_str = str(partition)
    p_dir = os.path.join(base_dir, p_str)
    target_file = os.path.join(p_dir, filename)

    if not os.path.isdir(p_dir):
        return partition, False, "Directory missing"

    if not os.path.isfile(target_file) or os.path.getsize(target_file) < 100000:
        return partition, False, f"{filename} missing or too small"

    try:
        f = ROOT.TFile.Open(target_file, "READ")
        if not f or f.IsZombie():
            if f:
                f.Close()
            return partition, False, f"{filename} is Zombie"

        k_c1 = f.GetKey("cbmsim", 1)
        k_c2 = f.GetKey("cbmsim", 2)
        k_truth = f.GetKey("mctruth")

        if not k_c1:
            f.Close()
            return partition, False, "cbmsim;1 missing"

        if not k_c2:
            f.Close()
            return partition, False, "cbmsim;2 missing"

        if not k_truth:
            f.Close()
            return partition, False, "mctruth tree missing"

        t_cbm = f.Get("cbmsim")
        t_truth = f.Get("mctruth")

        if not t_cbm or t_cbm.GetEntries() <= 0:
            f.Close()
            return partition, False, "cbmsim empty"

        if not t_truth or t_truth.GetEntries() <= 0:
            f.Close()
            return partition, False, "mctruth empty"

        # Check unpruned MCTrack size on entry 0
        t_cbm.GetEntry(0)
        mctracks = getattr(t_cbm, "MCTrack", None)
        n_trk = mctracks.GetEntries() if mctracks else 0
        f.Close()

        if n_trk <= 0:
            return partition, False, f"entry 0 MCTrack empty (got {n_trk})"

    except Exception as e:
        return partition, False, f"ROOT exception: {e}"

    return partition, True, None


def main():
    parser = argparse.ArgumentParser(description="Audit Stage 3 MC Truth extraction files on EOS.")
    parser.add_argument("-d", "--data-dir", default="/eos/user/i/idioniso/snd-numu-charm/data/newP8", help="Base directory")
    parser.add_argument("-p", "--partitions", default="1-1000", help="Partitions range or comma-separated list")
    parser.add_argument("-f", "--file-name", default="sndLHC.Genie-TGeant4_MCEB.root", help="Target ROOT file name")
    parser.add_argument("-j", "--jobs", type=int, default=16, help="Concurrent scanner threads")
    parser.add_argument("-o", "--output-failed", default="failed_mctruth_partitions.txt", help="Failed partitions list file")
    args = parser.parse_args()

    partitions = parse_partitions(args.partitions)
    if not partitions:
        partitions = list(range(1, 1001))

    print(f"Auditing Stage 3 Truth Extraction outputs for {len(partitions)} partitions in {args.data_dir}...")
    valid = []
    failed = []

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(check_partition, args.data_dir, p, args.file_name) for p in partitions]
        for fut in futures:
            part, is_ok, reason = fut.result()
            if is_ok:
                valid.append(part)
            else:
                failed.append((part, reason))

    valid.sort()
    failed.sort(key=lambda x: x[0])

    print("=" * 70)
    print(f"SUMMARY: {len(valid)} valid, {len(failed)} missing/corrupted out of {len(partitions)}")
    print("=" * 70)

    if failed:
        with open(args.output_failed, "w") as f:
            for part, reason in failed:
                f.write(f"{part}\n")
        print(f"Wrote {len(failed)} failed partitions to {args.output_failed}")
        print("First 10 failures:")
        for part, reason in failed[:10]:
            print(f"  Partition {part:>4}: {reason}")
    else:
        if os.path.exists(args.output_failed):
            os.remove(args.output_failed)
        print("All audited Stage 3 files are healthy with cbmsim;1, cbmsim;2, and mctruth!")


if __name__ == "__main__":
    main()
