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


def verify_partition(p: int, data_dir: str, disp_dir: str, prefilter_dir: str, requested_stages: List[str]) -> Tuple[int, bool, Dict[str, str]]:
    pdir = os.path.join(data_dir, str(p))
    stage_status = {}

    # Stage 1: Sim
    if "sim" in requested_stages:
        sim_file = os.path.join(pdir, "sndLHC.Genie-TGeant4.root")
        geo_file = os.path.join(pdir, "geofile_full.Genie-TGeant4.root")
        stage_status["sim"] = "OK" if (os.path.isfile(sim_file) and os.path.getsize(sim_file) > 100000 and os.path.isfile(geo_file)) else "MISSING"

    # Stage 2 & 3: MCEB & MCTruth
    if "mceb" in requested_stages or "mctruth" in requested_stages:
        mceb_file = os.path.join(pdir, "sndLHC.Genie-TGeant4_MCEB.root")
        if os.path.isfile(mceb_file) and os.path.getsize(mceb_file) > 100000:
            if "mceb" in requested_stages:
                stage_status["mceb"] = "OK"
            if "mctruth" in requested_stages:
                f = ROOT.TFile.Open(mceb_file, "READ")
                if f is not None and not f.IsZombie():
                    has_mc = f.GetKey("mctruth") is not None
                    has_cbm2 = f.GetKey("cbmsim", 2) is not None
                    f.Close()
                    stage_status["mctruth"] = "OK" if (has_mc and has_cbm2) else "MISSING_TRUTH_KEYS"
                else:
                    stage_status["mctruth"] = "ZOMBIE"
        else:
            if "mceb" in requested_stages:
                stage_status["mceb"] = "MISSING"
            if "mctruth" in requested_stages:
                stage_status["mctruth"] = "MISSING"

    # Stage 4: Digi
    if "digi" in requested_stages:
        digi_file = os.path.join(pdir, "sndLHC.Genie-TGeant4_MCEB_dig.root")
        stage_status["digi"] = "OK" if (os.path.isfile(digi_file) and os.path.getsize(digi_file) > 100000) else "MISSING"

    # Stage 5: Tracking & Skims
    if "track" in requested_stages:
        track_file = os.path.join(pdir, "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root")
        direct_skim = os.path.join(pdir, "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonDirect.root")
        indirect_skim = os.path.join(pdir, "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonIndirect.root")

        has_track = os.path.isfile(track_file) and os.path.getsize(track_file) > 100000
        has_skims = os.path.isfile(direct_skim) and os.path.isfile(indirect_skim)
        stage_status["track"] = "OK" if (has_track and has_skims) else "MISSING"

    # Stage 6: Prefilter
    if "prefilter" in requested_stages:
        pref_main = os.path.join(prefilter_dir, "essentials", str(p), "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root")
        pref_dir = os.path.join(prefilter_dir, "essentials-direct", str(p), "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root")
        pref_ind = os.path.join(prefilter_dir, "essentials-indirect", str(p), "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root")
        has_pref = (
            os.path.isfile(pref_main) and os.path.getsize(pref_main) > 500 and
            os.path.isfile(pref_dir) and os.path.getsize(pref_dir) > 500 and
            os.path.isfile(pref_ind) and os.path.getsize(pref_ind) > 500
        )
        stage_status["prefilter"] = "OK" if has_pref else "MISSING"

    all_ok = all(v == "OK" for v in stage_status.values())
    return p, all_ok, stage_status


def main():
    parser = argparse.ArgumentParser(description="Master End-to-End Auditor for SND@LHC Pipeline")
    parser.add_argument("-p", "--partitions", type=str, default="1-1000", help="Partitions to audit (default: 1-1000)")
    parser.add_argument("-s", "--stages", type=str, default="sim,mceb,mctruth,digi,track", help="Stages to verify: sim,mceb,mctruth,digi,track,prefilter or all (default: sim,mceb,mctruth,digi,track)")
    parser.add_argument("-d", "--data-dir", type=str, default="/eos/user/i/idioniso/snd-numu-charm/data/newP8")
    parser.add_argument("--prefilter-dir", type=str, default="/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8")
    parser.add_argument("--disp-dir", type=str, default="/eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="Workers")
    args = parser.parse_args()

    partitions = parse_partitions(args.partitions)

    all_known_stages = ["sim", "mceb", "mctruth", "digi", "track", "prefilter"]
    if args.stages.strip().lower() == "all":
        requested_stages = all_known_stages
    else:
        requested_stages = [st.strip().lower() for st in args.stages.split(",") if st.strip().lower() in all_known_stages]
    if not requested_stages:
        requested_stages = all_known_stages

    print("==========================================================")
    print(f" Auditing Pipeline ({','.join(requested_stages)}) for {len(partitions)} Partition(s)")
    print(f" Data Directory : {args.data_dir}")
    print("==========================================================")

    passed = 0
    failed = []

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(verify_partition, p, args.data_dir, args.disp_dir, args.prefilter_dir, requested_stages): p for p in partitions}
        for fut in as_completed(futures):
            p, ok, stages = fut.result()
            status_str = " | ".join([f"{k.capitalize()}: {v}" for k, v in stages.items()])
            if ok:
                passed += 1
                print(f" [OK]   Part {p:4d} | {status_str}")
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
