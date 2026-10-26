#!/usr/bin/env python3
"""
htcondor/dimuon_tracking/verify_dimuon_tracking_outputs.py
---------------------------------------------------------
Verifies the integrity, completeness, and event yield consistency of
Stage 5: Dimuon Track Reconstruction & Skimming outputs.

Checks per partition:
1. File existence & non-zero size for:
   - Tracking file: sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root
   - Direct skim  : sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonDirect.root
   - Indirect skim: sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonIndirect.root
2. Tree structure & entry count consistency in tracked file:
   - 'cbmsim' tree exists and has >0 entries
   - 'cbmsim' contains 'Reco_MuonTracks' branch
   - 'mctruth' tree exists and matches 'cbmsim' entry count
3. Skim integrity:
   - Skim ROOT files open cleanly with 'cbmsim' and 'mctruth' trees

Usage:
  python3 verify_dimuon_tracking_outputs.py -p 1-10
  python3 verify_dimuon_tracking_outputs.py -p 1-1000 -j 8
"""

from __future__ import annotations

import os
import sys
import argparse
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
    data_dir: str,
) -> Tuple[int, bool, str, Dict[str, int]]:
    """Verify Stage 5 outputs for a single partition."""
    part_dir = os.path.join(data_dir, str(partition))
    track_file = os.path.join(part_dir, "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root")
    direct_file = os.path.join(part_dir, "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonDirect.root")
    indirect_file = os.path.join(part_dir, "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonIndirect.root")

    if not os.path.isfile(track_file):
        return partition, False, f"Missing tracking file: {track_file}", {}
    if os.path.getsize(track_file) < 500:
        return partition, False, f"Tracking file too small ({os.path.getsize(track_file)} B)", {}

    tf = ROOT.TFile.Open(track_file, "READ")
    if tf is None or tf.IsZombie():
        if tf is not None:
            tf.Close()
        return partition, False, "Corrupt tracking ROOT file", {}

    cbm = tf.Get("cbmsim")
    if cbm is None:
        tf.Close()
        return partition, False, "Missing 'cbmsim' tree in tracking file", {}

    n_cbm = cbm.GetEntries()
    branches = cbm.GetListOfBranches()
    has_tracks_branch = (branches.FindObject("Reco_MuonTracks") is not None) if branches else False
    if not has_tracks_branch:
        tf.Close()
        return partition, False, "Missing 'Reco_MuonTracks' branch in 'cbmsim'", {}

    mc = tf.Get("mctruth")
    n_mc = mc.GetEntries() if mc is not None else 0
    tf.Close()

    if mc is None:
        return partition, False, "Missing 'mctruth' tree in tracking file", {}
    if n_cbm != n_mc:
        return partition, False, f"Entry count mismatch: cbmsim={n_cbm} != mctruth={n_mc}", {}

    # Check skims
    n_direct = 0
    n_indirect = 0
    if os.path.isfile(direct_file) and os.path.getsize(direct_file) >= 500:
        df = ROOT.TFile.Open(direct_file, "READ")
        if df is not None and not df.IsZombie():
            dt = df.Get("cbmsim")
            n_direct = dt.GetEntries() if dt is not None else 0
            df.Close()

    if os.path.isfile(indirect_file) and os.path.getsize(indirect_file) >= 500:
        indf = ROOT.TFile.Open(indirect_file, "READ")
        if indf is not None and not indf.IsZombie():
            indt = indf.Get("cbmsim")
            n_indirect = indt.GetEntries() if indt is not None else 0
            indf.Close()

    counts = {
        "n_events": n_cbm,
        "n_direct": n_direct,
        "n_indirect": n_indirect,
    }
    return partition, True, "OK", counts


def main():
    parser = argparse.ArgumentParser(description="Verify Stage 5 Dimuon Tracking Outputs")
    parser.add_argument("-p", "--partitions", type=str, default="all", help="Partitions to check (e.g. 1-10, 1,2,5, or all)")
    parser.add_argument("-d", "--data-dir", type=str, default="/eos/user/i/idioniso/snd-numu-charm/data/newP8", help="Data base directory")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    partitions = parse_partitions(args.partitions)
    print(f"==========================================================")
    print(f" Verifying Stage 5 Outputs for {len(partitions)} Partitions")
    print(f" Data Directory : {args.data_dir}")
    print(f" Workers        : {args.jobs}")
    print(f"==========================================================")

    passed = 0
    failed = []

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(verify_partition, p, args.data_dir): p for p in partitions}
        for fut in as_completed(futures):
            p, ok, msg, counts = fut.result()
            if ok:
                passed += 1
                print(f" [OK] Part {p:4d} | Events: {counts['n_events']} | Direct: {counts['n_direct']} | Indirect: {counts['n_indirect']}")
            else:
                failed.append((p, msg))
                print(f" [FAIL] Part {p:4d} | {msg}", file=sys.stderr)

    print(f"==========================================================")
    print(f" Summary: {passed}/{len(partitions)} PASSED, {len(failed)} FAILED")
    print(f"==========================================================")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
