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

    # Check digi matches sim entries
    f_digi = ROOT.TFile.Open(digi_path)
    if not f_digi or f_digi.IsZombie():
        result["corrupted"].append("sndLHC.Genie-TGeant4_dig.root")
    else:
        t_digi = f_digi.Get("cbmsim")
        if not t_digi or (result["n_events"] is not None and t_digi.GetEntries() != result["n_events"]):
            result["corrupted"].append(f"sndLHC.Genie-TGeant4_dig.root (entries {t_digi.GetEntries() if t_digi else None} != sim {result['n_events']})")
        f_digi.Close()

    # Check MCEB (MCEB splits by 25ns, so entries >= sim)
    n_mceb = None
    f_mceb = ROOT.TFile.Open(mceb_path)
    if not f_mceb or f_mceb.IsZombie():
        result["corrupted"].append("sndLHC.Genie-TGeant4_MCEB.root")
    else:
        t_mceb = f_mceb.Get("cbmsim")
        if not t_mceb or (result["n_events"] is not None and t_mceb.GetEntries() < result["n_events"]):
            result["corrupted"].append(f"sndLHC.Genie-TGeant4_MCEB.root (entries {t_mceb.GetEntries() if t_mceb else None} < sim {result['n_events']})")
        elif t_mceb:
            n_mceb = t_mceb.GetEntries()
        f_mceb.Close()

    # Check MCEB digi matches MCEB entries
    f_mceb_digi = ROOT.TFile.Open(mceb_digi_path)
    if not f_mceb_digi or f_mceb_digi.IsZombie():
        result["corrupted"].append("sndLHC.Genie-TGeant4_MCEB_dig.root")
    else:
        t_mceb_digi = f_mceb_digi.Get("cbmsim")
        if not t_mceb_digi or (n_mceb is not None and t_mceb_digi.GetEntries() != n_mceb):
            result["corrupted"].append(f"sndLHC.Genie-TGeant4_MCEB_dig.root (entries {t_mceb_digi.GetEntries() if t_mceb_digi else None} != mceb {n_mceb})")
        f_mceb_digi.Close()

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
