#!/usr/bin/env python3
"""
scripts/clean_duplicate_branches.py
-----------------------------------
Cleans duplicate branches (e.g. Reco_MuonTracks) from SND@LHC ROOT files,
keeping only the final (most recent) branch cycle.

Usage:
  python3 scripts/clean_duplicate_branches.py \
    -i "/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root" \
    -p "1-1000" \
    -b "Reco_MuonTracks" \
    -w 12
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import shutil
from typing import List, Dict, Any, Optional
import multiprocessing as mp


def clean_file_branches(
    file_path: str,
    target_branch: str = "Reco_MuonTracks",
    tree_name: str = "cbmsim",
) -> Dict[str, Any]:
    """
    Remove duplicate cycles of target_branch in tree_name, preserving only the last cycle.
    Rewrites atomically via a temporary file.
    """
    import ROOT
    ROOT.gROOT.SetBatch(True)

    result = {
        "file": file_path,
        "status": "SKIPPED",
        "original_branches": 0,
        "cleaned_branches": 0,
        "entries": 0,
        "error": "",
    }

    if not os.path.isfile(file_path):
        result["status"] = "NOT_FOUND"
        return result

    f_in = ROOT.TFile.Open(file_path, "READ")
    if not f_in or f_in.IsZombie():
        result["status"] = "ERROR_OPENING"
        return result

    t_in = f_in.Get(tree_name)
    if not t_in:
        f_in.Close()
        result["status"] = "NO_TREE"
        return result

    all_branches = [b.GetName() for b in t_in.GetListOfBranches()]
    dup_count = all_branches.count(target_branch)
    result["original_branches"] = len(all_branches)
    result["entries"] = t_in.GetEntries()

    if dup_count <= 1:
        f_in.Close()
        result["status"] = "ALREADY_CLEAN"
        result["cleaned_branches"] = len(all_branches)
        return result

    # Temporary file on local /tmp to ensure fast, reliable cloning
    tmp_path = f"/tmp/clean_{os.getpid()}_{os.path.basename(file_path)}"

    try:
        t_in.SetBranchStatus("*", 1)
        matching_branches = [b for b in t_in.GetListOfBranches() if b.GetName() == target_branch]
        for b in matching_branches[:-1]:
            b.SetStatus(0)
        matching_branches[-1].SetStatus(1)

        f_out = ROOT.TFile.Open(tmp_path, "RECREATE")
        if not f_out or f_out.IsZombie():
            f_in.Close()
            result["status"] = "ERROR_CREATING_TMP"
            return result

        out_tree = t_in.CloneTree(-1, "fast")
        out_tree.Write(tree_name)

        # Preserve auxiliary trees/objects (e.g. BranchList, mctruth)
        for k in f_in.GetListOfKeys():
            kname = k.GetName()
            if kname != tree_name:
                obj = f_in.Get(kname)
                if obj:
                    f_out.cd()
                    if isinstance(obj, ROOT.TTree):
                        obj_clone = obj.CloneTree(-1, "fast")
                        obj_clone.Write(kname, ROOT.TObject.kOverwrite)
                    else:
                        obj.Write(kname, ROOT.TObject.kOverwrite | ROOT.TObject.kSingleKey)

        f_out.Close()
        f_in.Close()

        # Atomic replacement
        shutil.copyfile(tmp_path, file_path)
        result["status"] = "SUCCESS"
        result["cleaned_branches"] = len(all_branches) - (dup_count - 1)
    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = str(exc)
        if f_in and not f_in.IsZombie():
            f_in.Close()
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return result


def _clean_worker_task(args):
    file_path, target_branch, tree_name = args
    return clean_file_branches(file_path, target_branch, tree_name)


def parse_partition_spec(spec: str) -> List[int]:
    parts = []
    for item in spec.split(","):
        item = item.strip()
        if "-" in item:
            start, end = map(int, item.split("-", 1))
            parts.extend(range(start, end + 1))
        elif item.isdigit():
            parts.append(int(item))
    return sorted(set(parts))


def main():
    parser = argparse.ArgumentParser(description="Clean duplicate branch cycles from ROOT trees.")
    parser.add_argument("-i", "--input", required=True, help="Input pattern with '%s' partition placeholder")
    parser.add_argument("-p", "--partitions", required=True, help="Partitions to process ('1-1000')")
    parser.add_argument("-b", "--branch", default="Reco_MuonTracks", help="Branch name to deduplicate")
    parser.add_argument("-t", "--tree", default="cbmsim", help="Tree name (default: cbmsim)")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Number of parallel worker processes")

    args = parser.parse_args()

    partitions = parse_partition_spec(args.partitions)
    tasks = []
    for p in partitions:
        c_p = args.input.count("%s")
        p_str = str(p)
        filepath = args.input % tuple(p_str for _ in range(c_p)) if c_p > 0 else args.input
        tasks.append((filepath, args.branch, args.tree))

    print(f"Cleaning branch '{args.branch}' across {len(tasks)} files using {args.workers} workers...")
    t0 = time.time()

    success_count = 0
    clean_count = 0
    skipped_count = 0
    error_count = 0

    with mp.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for i, res in enumerate(pool.imap_unordered(_clean_worker_task, tasks), 1):
            st = res["status"]
            if st == "SUCCESS":
                success_count += 1
                print(f"[{i:4d}/{len(tasks)}] Cleaned: {os.path.basename(res['file'])} ({res['original_branches']} -> {res['cleaned_branches']} branches)")
            elif st == "ALREADY_CLEAN":
                clean_count += 1
            elif st in ("NOT_FOUND", "SKIPPED", "NO_TREE"):
                skipped_count += 1
            else:
                error_count += 1
                print(f"[{i:4d}/{len(tasks)}] Error on {res['file']}: {res['error']}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s | Cleaned: {success_count} | Already clean: {clean_count} | Skipped: {skipped_count} | Errors: {error_count}")


if __name__ == "__main__":
    main()
