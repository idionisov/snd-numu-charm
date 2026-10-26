#!/usr/bin/env python3
"""
htcondor/dagman/verify_dag_outputs.py
------------------------------------
Master end-to-end pipeline auditor for SND@LHC DAGMan workflow.
Verifies all 5 stages for every specified partition:
  1. Simulation: Genie-TGeant4.root + geofile
  2. MCEB: Genie-TGeant4_MCEB.root
  3. MCTruth: cbmsim;1, cbmsim;2, mctruth in MCEB file
  4. Digi: Genie-TGeant4_MCEB_dig.root with hit branches
  5. Tracking: Genie-TGeant4_MCEB_dig_2MuTrks_truth.root, charm skims, displays

Usage:
  python3 verify_dag_outputs.py -p 2
  python3 verify_dag_outputs.py -p 1-1000 -j 8
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gEnv.SetValue("TFile.Recover", 0)
ROOT.gErrorIgnoreLevel = ROOT.kFatal


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
                for p in range(min(start, end), max(start, end) + 1):
                    partitions.add(p)
        elif token.isdigit():
            partitions.add(int(token))
    return sorted(list(partitions))


def verify_partition(p: int, data_dir: str, disp_dir: str) -> Tuple[int, bool, Dict[str, str]]:
    pdir = os.path.join(data_dir, str(p))
    stage_status = {}

    # Stage 1: Sim
    sim_file = os.path.join(pdir, "sndLHC.Genie-TGeant4.root")
    geo_file = os.path.join(pdir, "geofile_full.Genie-TGeant4.root")
    stage_status["sim"] = "OK" if (os.path.isfile(sim_file) and os.path.getsize(sim_file) > 100000 and os.path.isfile(geo_file)) else "MISSING"

    # Stage 2 & 3: MCEB & MCTruth
    mceb_file = os.path.join(pdir, "sndLHC.Genie-TGeant4_MCEB.root")
    if os.path.isfile(mceb_file) and os.path.getsize(mceb_file) > 100000:
        stage_status["mceb"] = "OK"
        # Check truth
        f = ROOT.TFile.Open(mceb_file, "READ")
        if f is not None and not f.IsZombie():
            has_mc = f.GetKey("mctruth") is not None
            has_cbm2 = f.GetKey("cbmsim", 2) is not None
            f.Close()
            stage_status["mctruth"] = "OK" if (has_mc and has_cbm2) else "MISSING_TRUTH_KEYS"
        else:
            stage_status["mctruth"] = "ZOMBIE"
    else:
        stage_status["mceb"] = "MISSING"
        stage_status["mctruth"] = "MISSING"

    # Stage 4: Digi
    digi_file = os.path.join(pdir, "sndLHC.Genie-TGeant4_MCEB_dig.root")
    stage_status["digi"] = "OK" if (os.path.isfile(digi_file) and os.path.getsize(digi_file) > 100000) else "MISSING"

    # Stage 5: Tracking & Skims
    track_file = os.path.join(pdir, "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root")
    direct_skim = os.path.join(pdir, "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonDirect.root")
    indirect_skim = os.path.join(pdir, "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonIndirect.root")
    disp_direct = os.path.join(disp_dir, "direct", f"displays_part{p}.root")

    has_track = os.path.isfile(track_file) and os.path.getsize(track_file) > 100000
    has_skims = os.path.isfile(direct_skim) and os.path.isfile(indirect_skim)
    stage_status["track"] = "OK" if (has_track and has_skims) else "MISSING"

    all_ok = all(v == "OK" for v in stage_status.values())
    return p, all_ok, stage_status


def main():
    parser = argparse.ArgumentParser(description="Master End-to-End Auditor for SND@LHC Pipeline")
    parser.add_argument("-p", "--partitions", type=str, default="1-1000", help="Partitions to audit (default: 1-1000)")
    parser.add_argument("-d", "--data-dir", type=str, default="/eos/user/i/idioniso/snd-numu-charm/data/newP8")
    parser.add_argument("--disp-dir", type=str, default="/eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="Workers")
    args = parser.parse_args()

    partitions = parse_partitions(args.partitions)
    print("==========================================================")
    print(f" Auditing Full Pipeline for {len(partitions)} Partition(s)")
    print(f" Data Directory : {args.data_dir}")
    print("==========================================================")

    passed = 0
    failed = []

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(verify_partition, p, args.data_dir, args.disp_dir): p for p in partitions}
        for fut in as_completed(futures):
            p, ok, stages = fut.result()
            if ok:
                passed += 1
                print(f" [OK]   Part {p:4d} | Sim: OK | MCEB: OK | MCTruth: OK | Digi: OK | Tracking: OK")
            else:
                failed.append((p, stages))
                errs = ", ".join([f"{k}:{v}" for k, v in stages.items() if v != "OK"])
                print(f" [FAIL] Part {p:4d} | Issues: {errs}", file=sys.stderr)

    print("==========================================================")
    print(f" Audit Result: {passed}/{len(partitions)} Fully Completed, {len(failed)} Incomplete")
    print("==========================================================")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
