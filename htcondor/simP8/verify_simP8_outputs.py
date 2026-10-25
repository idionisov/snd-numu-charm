#!/usr/bin/env python3
"""
verify_simP8_outputs.py
-----------------------
Verifies the integrity, completeness, and event yield consistency of the
SND@LHC nu14 simulation, MCEB, and digitization outputs.

Usage:
  python3 verify_simP8_outputs.py -p 1-5
  python3 verify_simP8_outputs.py -p 1-1000 -j 16
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import ROOT
ROOT.gROOT.SetBatch(True)


def parse_partitions(part_arg: Optional[str]) -> List[int]:
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
                partitions.update(range(min(start, end), max(start, end) + 1))
        elif token.isdigit():
            partitions.add(int(token))
    return sorted(partitions)


def check_partition(part_id: int, base_dir: str) -> Dict[str, Any]:
    part_dir = os.path.join(base_dir, str(part_id))
    result = {
        "partition": part_id,
        "valid": False,
        "missing": [],
        "corrupted": [],
        "n_events": None,
        "error": None,
    }

    if not os.path.isdir(part_dir):
        result["error"] = f"Directory missing: {part_dir}"
        return result

    required_files = [
        "geofile_full.Genie-TGeant4.root",
        "sndLHC.Genie-TGeant4.root",
        "sndLHC.Genie-TGeant4_MCEB.root",
        "sndLHC.Genie-TGeant4_dig.root",
        "sndLHC.Genie-TGeant4_MCEB_dig.root",
    ]

    for fname in required_files:
        fpath = os.path.join(part_dir, fname)
        if not os.path.isfile(fpath) or os.path.getsize(fpath) == 0:
            result["missing"].append(fname)

    if result["missing"]:
        result["error"] = f"Missing/empty files: {', '.join(result['missing'])}"
        return result

    # Check ROOT file integrity
    sim_path = os.path.join(part_dir, "sndLHC.Genie-TGeant4.root")
    mceb_path = os.path.join(part_dir, "sndLHC.Genie-TGeant4_MCEB.root")
    digi_path = os.path.join(part_dir, "sndLHC.Genie-TGeant4_dig.root")
    mceb_digi_path = os.path.join(part_dir, "sndLHC.Genie-TGeant4_MCEB_dig.root")

    f_sim = ROOT.TFile.Open(sim_path)
    if not f_sim or f_sim.IsZombie():
        result["corrupted"].append("sndLHC.Genie-TGeant4.root")
    else:
        tree = f_sim.Get("cbmsim")
        gst = f_sim.Get("gst")
        if not tree:
            result["corrupted"].append("sndLHC.Genie-TGeant4.root (no cbmsim)")
        else:
            result["n_events"] = tree.GetEntries()
            if gst and gst.GetEntries() != tree.GetEntries():
                result["corrupted"].append(f"sim (gst {gst.GetEntries()} != cbmsim {tree.GetEntries()})")
        f_sim.Close()

    for path, name in [
        (mceb_path, "sndLHC.Genie-TGeant4_MCEB.root"),
        (digi_path, "sndLHC.Genie-TGeant4_dig.root"),
        (mceb_digi_path, "sndLHC.Genie-TGeant4_MCEB_dig.root"),
    ]:
        f = ROOT.TFile.Open(path)
        if not f or f.IsZombie():
            result["corrupted"].append(name)
        else:
            t = f.Get("cbmsim")
            if not t or (result["n_events"] is not None and t.GetEntries() != result["n_events"]):
                result["corrupted"].append(f"{name} (entry mismatch)")
            f.Close()

    if result["corrupted"]:
        result["error"] = f"Corrupted/mismatched: {', '.join(result['corrupted'])}"
    else:
        result["valid"] = True

    return result


def main():
    parser = argparse.ArgumentParser(description="Verify simP8 outputs")
    parser.add_argument("-d", "--data-dir", default="/eos/user/i/idioniso/snd-numu-charm/data/newP8", help="Path to newP8 directory")
    parser.add_argument("-p", "--partitions", default="1-1000", help="Partition range (e.g. 1-5, 1-1000)")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="Number of parallel verification workers")
    args = parser.parse_args()

    partitions = parse_partitions(args.partitions)
    print(f"Verifying {len(partitions)} partitions in {args.data_dir} with {args.jobs} workers...")

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(check_partition, p, args.data_dir): p for p in partitions}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            if not res["valid"]:
                print(f" [FAIL] Partition {res['partition']:4d}: {res['error']}")

    valid_count = sum(1 for r in results if r["valid"])
    total = len(partitions)
    pct = (valid_count / total) * 100 if total else 0.0

    print("=" * 60)
    print(f" Verification Summary: {valid_count}/{total} valid ({pct:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
