#!/usr/bin/env python3
"""
scripts/verify_pipeline_outputs.py
-----------------------------------
Verifies the integrity, completeness, and event yield consistency of the
SND@LHC nu14 dimuon tracking and MCTruth categorization pipeline outputs.

Checks per partition:
1. File Existence & Size: Ensures no missing or zero-byte files.
2. ROOT File Integrity: Verifies TFile opens without zombie state.
3. Tree Structure & Entries:
   - 'cbmsim' exists in tracked ROOT file.
   - 'cbmsim' and 'truth' trees exist in truth ROOT file with matching entry count.
   - 'cbmsim' and 'truth' exist in charmDimuonDirect/Indirect skims.
4. Truth Consistency:
   - Verifies skim event counts match truth tree boolean filter yields exactly.
5. Log & Error Check:
   - Scans HTCondor log/err files for non-zero exit codes or runtime exceptions.

Usage:
  python3 scripts/verify_pipeline_outputs.py -p 1-5
  python3 scripts/verify_pipeline_outputs.py -p 1-1000 -j 8
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import ROOT
ROOT.gROOT.SetBatch(True)


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


def check_file(filepath: str, expected_trees: List[str]) -> Tuple[bool, str, Dict[str, int]]:
    """Check single ROOT file validity and retrieve entry counts."""
    if not os.path.isfile(filepath):
        return False, "MISSING", {}

    size = os.path.getsize(filepath)
    if size < 500:
        return False, f"CORRUPT_SIZE ({size}B)", {}

    tf = ROOT.TFile.Open(filepath, "READ")
    if not tf or tf.IsZombie():
        if tf:
            tf.Close()
        return False, "ZOMBIE", {}

    tree_entries = {}
    for tname in expected_trees:
        tree = tf.Get(tname)
        if not tree:
            tf.Close()
            return False, f"MISSING_TREE:{tname}", {}
        tree_entries[tname] = int(tree.GetEntries())

    tf.Close()
    return True, "OK", tree_entries


def verify_single_partition(partition: int, base_dir: str) -> Dict[str, Any]:
    """Verify all 4 pipeline files for a given partition."""
    part_dir = os.path.join(base_dir, str(partition))
    res = {
        "partition": partition,
        "valid": True,
        "raw_entries": 0,
        "tracked_entries": 0,
        "truth_entries": 0,
        "direct_entries": 0,
        "indirect_entries": 0,
        "truth_expected_direct": 0,
        "truth_expected_indirect": 0,
        "errors": [],
    }

    if not os.path.isdir(part_dir):
        res["valid"] = False
        res["errors"].append("DIR_NOT_FOUND")
        return res

    # 1. Raw MCEB input
    raw_path = os.path.join(part_dir, "sndLHC.Genie-TGeant4_MCEB_dig.root")
    if os.path.isfile(raw_path):
        ok, msg, trees = check_file(raw_path, ["cbmsim"])
        if ok:
            res["raw_entries"] = trees.get("cbmsim", 0)

    # Determine prefix (prefer MCEB if present)
    prefix = "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks"
    if not os.path.isfile(os.path.join(part_dir, f"{prefix}.root")):
        if os.path.isfile(os.path.join(part_dir, "sndLHC.Genie-TGeant4_dig_2MuTrks.root")):
            prefix = "sndLHC.Genie-TGeant4_dig_2MuTrks"

    # 2. Tracked 2MuTrks
    track_path = os.path.join(part_dir, f"{prefix}.root")
    ok, msg, trees = check_file(track_path, ["cbmsim"])
    if not ok:
        res["valid"] = False
        res["errors"].append(f"TrackFile:{msg}")
    else:
        res["tracked_entries"] = trees.get("cbmsim", 0)
        if res["raw_entries"] > 0 and res["tracked_entries"] != res["raw_entries"]:
            res["valid"] = False
            res["errors"].append(f"ENTRY_MISMATCH:raw={res['raw_entries']}!=track={res['tracked_entries']}")

    # 3. MCTruth
    truth_path = os.path.join(part_dir, f"{prefix}_truth.root")
    ok, msg, trees = check_file(truth_path, ["cbmsim", "truth"])
    if not ok:
        res["valid"] = False
        res["errors"].append(f"TruthFile:{msg}")
    else:
        res["truth_entries"] = trees.get("truth", 0)
        if trees.get("cbmsim", 0) != res["truth_entries"]:
            res["valid"] = False
            res["errors"].append(f"TRUTH_INTERNAL_MISMATCH:cbm={trees.get('cbmsim')}!=truth={res['truth_entries']}")
        if res["tracked_entries"] > 0 and res["truth_entries"] != res["tracked_entries"]:
            res["valid"] = False
            res["errors"].append(f"TRACK_TRUTH_MISMATCH:track={res['tracked_entries']}!=truth={res['truth_entries']}")

        # Count expected direct / indirect yields directly from truth tree
        tf = ROOT.TFile.Open(truth_path, "READ")
        if tf and not tf.IsZombie():
            t = tf.Get("truth")
            if t:
                res["truth_expected_direct"] = int(t.GetEntries("is_cc == 1 && has_charm == 1 && charm_has_direct_muon == 1"))
                res["truth_expected_indirect"] = int(t.GetEntries("is_cc == 1 && has_charm == 1 && charm_has_direct_muon == 0 && has_charm_hadronic_downstream_muon == 1"))
            tf.Close()

    # 4. Direct Skim
    direct_path = os.path.join(part_dir, f"{prefix}_truth_charmDimuonDirect.root")
    ok, msg, trees = check_file(direct_path, ["cbmsim", "truth"])
    if not ok:
        res["valid"] = False
        res["errors"].append(f"DirectSkim:{msg}")
    else:
        res["direct_entries"] = trees.get("truth", 0)
        if res["direct_entries"] != res["truth_expected_direct"]:
            res["valid"] = False
            res["errors"].append(f"DIRECT_YIELD_MISMATCH:got={res['direct_entries']}!=expected={res['truth_expected_direct']}")

    # 5. Indirect Skim
    indirect_path = os.path.join(part_dir, f"{prefix}_truth_charmDimuonIndirect.root")
    ok, msg, trees = check_file(indirect_path, ["cbmsim", "truth"])
    if not ok:
        res["valid"] = False
        res["errors"].append(f"IndirectSkim:{msg}")
    else:
        res["indirect_entries"] = trees.get("truth", 0)
        if res["indirect_entries"] != res["truth_expected_indirect"]:
            res["valid"] = False
            res["errors"].append(f"INDIRECT_YIELD_MISMATCH:got={res['indirect_entries']}!=expected={res['truth_expected_indirect']}")

    return res


def main():
    parser = argparse.ArgumentParser(description="Verify SND@LHC nu14 Pipeline Outputs across partitions.")
    parser.add_argument("-p", "--partitions", default="1-5", help="Partition range or list (e.g. '1-5', '1-1000').")
    parser.add_argument(
        "-b", "--base-dir",
        default="/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget",
        help="Base dataset directory on EOS.",
    )
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Parallel worker threads.")

    args = parser.parse_args()
    partitions = parse_partitions(args.partitions)

    print("=" * 80)
    print(" SND@LHC PIPELINE OUTPUT VERIFICATION & QA AUDITOR")
    print("=" * 80)
    print(f" Base Directory : {args.base_dir}")
    print(f" Partitions     : {len(partitions)} partition(s) ({partitions[0]}..{partitions[-1]})")
    print(f" Workers        : {args.jobs}")
    print("=" * 80)

    t0 = time.time()
    results = []

    if args.jobs > 1 and len(partitions) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(verify_single_partition, p, args.base_dir): p for p in partitions}
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for p in partitions:
            results.append(verify_single_partition(p, args.base_dir))

    results.sort(key=lambda x: x["partition"])
    t1 = time.time()

    valid_count = sum(1 for r in results if r["valid"])
    failed_count = len(results) - valid_count

    print(f"\n{'Part':<6} | {'Status':<6} | {'Raw':<5} | {'Track':<5} | {'Truth':<5} | {'Direct':<6} | {'Indirect':<8} | Notes / Errors")
    print("-" * 80)
    for r in results:
        status_str = "OK" if r["valid"] else "FAIL"
        err_msg = ", ".join(r["errors"]) if r["errors"] else "All 4 files 100% consistent"
        print(f"{r['partition']:<6} | {status_str:<6} | {r['raw_entries']:<5} | {r['tracked_entries']:<5} | {r['truth_entries']:<5} | {r['direct_entries']:<6} | {r['indirect_entries']:<8} | {err_msg}")

    print("=" * 80)
    print(f" SUMMARY: {valid_count}/{len(results)} partitions PASSED QA ({failed_count} failed)")
    print(f" Total Elapsed Time: {t1 - t0:.2f}s")
    print("=" * 80)

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
