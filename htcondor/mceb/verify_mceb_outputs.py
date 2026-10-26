#!/usr/bin/env python3
"""
verify_mceb_outputs.py
----------------------
Audits completed Stage 2 MCEB files on EOS:
  - sndLHC.Genie-TGeant4_MCEB.root

Checks presence, non-zero file sizes, and non-zombie ROOT tree integrity.
Outputs missing or corrupted partitions into failed_mceb_partitions.txt.

Usage:
    python3 verify_mceb_outputs.py [-d /eos/.../data/newP8] [-p 1-1000]
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


def check_partition(base_dir: str, partition: int):
    p_str = str(partition)
    p_dir = os.path.join(base_dir, p_str)
    sim_file = os.path.join(p_dir, "sndLHC.Genie-TGeant4.root")
    mceb_file = os.path.join(p_dir, "sndLHC.Genie-TGeant4_MCEB.root")

    if not os.path.isdir(p_dir):
        return partition, False, "Directory missing"

    if not os.path.isfile(sim_file) or os.path.getsize(sim_file) < 100000:
        return partition, False, "Prerequisite sndLHC.Genie-TGeant4.root missing"

    if not os.path.isfile(mceb_file) or os.path.getsize(mceb_file) < 100000:
        return partition, False, "sndLHC.Genie-TGeant4_MCEB.root missing or too small"

    # Fast ROOT integrity check
    try:
        f = ROOT.TFile.Open(mceb_file, "READ")
        if not f or f.IsZombie():
            if f:
                f.Close()
            return partition, False, "sndLHC.Genie-TGeant4_MCEB.root is Zombie"
        t = f.Get("cbmsim")
        if not t:
            f.Close()
            return partition, False, "cbmsim tree missing in MCEB file"
        entries = t.GetEntries()
        f.Close()
        if entries <= 0:
            return partition, False, "cbmsim tree has 0 entries"
    except Exception as e:
        return partition, False, f"ROOT exception: {e}"

    return partition, True, None


def main():
    parser = argparse.ArgumentParser(description="Audit Stage 2 MCEventBuilder files on EOS.")
    parser.add_argument("-d", "--data-dir", default="/eos/user/i/idioniso/snd-numu-charm/data/newP8", help="Base directory")
    parser.add_argument("-p", "--partitions", default="1-1000", help="Partitions range or comma-separated list")
    parser.add_argument("-j", "--jobs", type=int, default=16, help="Concurrent scanner threads")
    parser.add_argument("-o", "--output-failed", default="failed_mceb_partitions.txt", help="Failed partitions list file")
    args = parser.parse_args()

    partitions = parse_partitions(args.partitions)
    if not partitions:
        partitions = list(range(1, 1001))

    print(f"Auditing Stage 2 MCEB outputs for {len(partitions)} partitions in {args.data_dir}...")
    valid = []
    failed = []

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(check_partition, args.data_dir, p) for p in partitions]
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
        print("All audited MCEB partitions are healthy and verified!")


if __name__ == "__main__":
    main()
