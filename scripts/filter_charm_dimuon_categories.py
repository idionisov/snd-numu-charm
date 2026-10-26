#!/usr/bin/env python3
"""
scripts/filter_charm_dimuon_categories.py
-----------------------------------------
Extracts dedicated skimmed ROOT files for:
1. Signal events (charm directly decays to muon):
   -> sndLHC.Genie-TGeant4_dig_2MuTrks_truth_charmDimuonDirect.root
   Criteria: is_cc == 1, has_charm == 1, charm_has_direct_muon == 1

2. Fake signal events (charm decays hadronically and a downstream decay product produces a second muon):
   -> sndLHC.Genie-TGeant4_dig_2MuTrks_truth_charmDimuonIndirect.root
   Criteria: is_cc == 1, has_charm == 1, charm_has_direct_muon == 0, has_charm_hadronic_downstream_muon == 1

Both 'cbmsim' and 'truth' trees are cloned with exact matching events.
Non-matching events are discarded completely.

Usage:
  # Test on partition 1:
  python3 scripts/filter_charm_dimuon_categories.py -p 1

  # Run over partitions 1 to 10 with 5 workers:
  python3 scripts/filter_charm_dimuon_categories.py -p 1-10 -j 5

  # Run across all partitions 1 to 1000 in parallel with 10 workers:
  python3 scripts/filter_charm_dimuon_categories.py -p 1-1000 -j 10 --skip-existing
"""

import os
import sys
import time
import shutil
import argparse
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gEnv.SetValue("TFile.Recover", 0)
ROOT.gEnv.SetValue("XNet.ConnectTimeout", 30)
ROOT.gEnv.SetValue("XNet.RequestTimeout", 120)
ROOT.gEnv.SetValue("XNet.MaxRedirects", 5)
ROOT.gErrorIgnoreLevel = ROOT.kError
try:
    ROOT.gSystem.Load("libShipData.so")
    ROOT.gSystem.Load("libshipLHC.so")
    ROOT.gInterpreter.ProcessLine("#include <unordered_map>")
    ROOT.gInterpreter.ProcessLine("#include <utility>")
    if not ROOT.TClass.GetClass("pair<int,unordered_map<int,float> >"):
        ROOT.gInterpreter.GenerateDictionary("pair<int,unordered_map<int,float> >", "utility;unordered_map")
except Exception:
    pass


def parse_partitions(part_arg: Optional[str]) -> List[int]:
    """Parse partition string e.g. '1-1000', '1,2,3', 'all', or path to file."""
    if not part_arg or part_arg.strip().lower() == "all":
        return list(range(1, 1001))

    if os.path.isfile(part_arg):
        partitions = set()
        with open(part_arg, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and line.isdigit():
                    partitions.add(int(line))
        return sorted(list(partitions))

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


def process_single_partition(
    partition: int,
    base_dir: str,
    skip_existing: bool = False,
    write_empty: bool = True,
    input_pattern: Optional[str] = None,
) -> Dict[str, Any]:
    """Processes a single partition and produces direct and indirect skimmed ROOT files."""
    part_dir = os.path.join(base_dir, str(partition))

    input_file = None
    if input_pattern:
        if "%s" in input_pattern:
            input_file = input_pattern % str(partition)
        else:
            input_file = input_pattern
            part_dir = os.path.dirname(os.path.abspath(input_file))
    else:
        for cand in [
            "sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root",
            "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root",
        ]:
            cand_path = os.path.join(part_dir, cand)
            if os.path.isfile(cand_path):
                input_file = cand_path
                break
        if not input_file:
            input_file = os.path.join(part_dir, "sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root")

    prefix = os.path.basename(input_file)
    if prefix.endswith(".root"):
        prefix = prefix[:-5]

    out_direct = os.path.join(part_dir, f"{prefix}_charmDimuonDirect.root")
    out_indirect = os.path.join(part_dir, f"{prefix}_charmDimuonIndirect.root")

    if not os.path.isfile(input_file):
        return {
            "partition": partition,
            "status": "missing_input",
            "total": 0,
            "direct": 0,
            "indirect": 0,
        }

    if skip_existing and os.path.isfile(out_direct) and os.path.isfile(out_indirect):
        if os.path.getsize(out_direct) > 500 and os.path.getsize(out_indirect) > 500:
            return {
                "partition": partition,
                "status": "skipped_existing",
                "total": 0,
                "direct": 0,
                "indirect": 0,
            }

    t0 = time.time()
    try:
        f_in = ROOT.TFile.Open(input_file, "READ")
    except (OSError, Exception):
        f_in = None
    if not f_in or f_in.IsZombie():
        return {
            "partition": partition,
            "status": "zombie_input",
            "total": 0,
            "direct": 0,
            "indirect": 0,
        }

    t_truth = f_in.Get("mctruth") or f_in.Get("mc_truth") or f_in.Get("truth")
    t_cbm = f_in.Get("cbmsim")
    branch_list = f_in.Get("BranchList")

    if not t_truth:
        f_in.Close()
        return {
            "partition": partition,
            "status": "missing_truth_tree",
            "total": 0,
            "direct": 0,
            "indirect": 0,
        }

    if t_cbm:
        t_cbm.SetCacheSize(30 * 1024 * 1024)
    t_truth.SetCacheSize(10 * 1024 * 1024)

    n_tot = t_truth.GetEntries()
    direct_entries: List[int] = []
    indirect_entries: List[int] = []

    for iev in range(n_tot):
        t_truth.GetEntry(iev)
        is_cc = int(getattr(t_truth, "is_cc", 0))
        has_charm = int(getattr(t_truth, "has_charm", 0))

        if is_cc == 1 and has_charm == 1:
            direct_mu = int(getattr(t_truth, "charm_has_direct_muon", 0))
            if direct_mu == 1:
                direct_entries.append(iev)
            else:
                ds_mu = int(getattr(t_truth, "has_charm_hadronic_downstream_muon", 0))
                if ds_mu == 1:
                    indirect_entries.append(iev)

    tmp_direct = f"/tmp/{prefix}_dir_{os.getpid()}_{partition}.root"
    tmp_indirect = f"/tmp/{prefix}_ind_{os.getpid()}_{partition}.root"

    try:
        # 1. Write Direct Skim
        if len(direct_entries) > 0 or write_empty:
            f_out_dir = ROOT.TFile.Open(tmp_direct, "RECREATE")
            cbm_dir = t_cbm.CloneTree(0) if t_cbm else None
            truth_dir = t_truth.CloneTree(0)

            for iev in direct_entries:
                if t_cbm and cbm_dir:
                    nb = t_cbm.GetEntry(iev)
                    if nb > 0:
                        cbm_dir.Fill()
                nb_tr = t_truth.GetEntry(iev)
                if nb_tr > 0:
                    truth_dir.Fill()

            f_out_dir.cd()
            if cbm_dir:
                cbm_dir.Write()
            truth_dir.Write()
            if branch_list:
                branch_list.Write("BranchList")
            f_out_dir.Close()
            shutil.move(tmp_direct, out_direct)

        # 2. Write Indirect Skim
        if len(indirect_entries) > 0 or write_empty:
            f_out_ind = ROOT.TFile.Open(tmp_indirect, "RECREATE")
            cbm_ind = t_cbm.CloneTree(0) if t_cbm else None
            truth_ind = t_truth.CloneTree(0)

            for iev in indirect_entries:
                if t_cbm and cbm_ind:
                    nb = t_cbm.GetEntry(iev)
                    if nb > 0:
                        cbm_ind.Fill()
                nb_tr = t_truth.GetEntry(iev)
                if nb_tr > 0:
                    truth_ind.Fill()

            f_out_ind.cd()
            if cbm_ind:
                cbm_ind.Write()
            truth_ind.Write()
            if branch_list:
                branch_list.Write("BranchList")
            f_out_ind.Close()
            shutil.move(tmp_indirect, out_indirect)
    finally:
        if os.path.exists(tmp_direct):
            try:
                os.remove(tmp_direct)
            except OSError:
                pass
        if os.path.exists(tmp_indirect):
            try:
                os.remove(tmp_indirect)
            except OSError:
                pass

    f_in.Close()
    elapsed = time.time() - t0

    return {
        "partition": partition,
        "status": "success",
        "total": n_tot,
        "direct": len(direct_entries),
        "indirect": len(indirect_entries),
        "time": elapsed,
    }


def _worker_wrapper(args_tuple: tuple) -> Dict[str, Any]:
    return process_single_partition(*args_tuple)


def main():
    parser = argparse.ArgumentParser(description="Create charmDimuonDirect and charmDimuonIndirect skims.")
    parser.add_argument(
        "-p", "--partitions",
        default="1-1000",
        help="Partition range or list (e.g. '1-1000', '1-10', '1,2,3', 'all'). Default: '1-1000'",
    )
    parser.add_argument(
        "-f", "--args-file", "--partitions-file",
        dest="partitions_file",
        default=None,
        help="Text file with partition IDs (one per line, e.g. args_nu14_volTarget_1000_failed.txt)",
    )
    parser.add_argument(
        "-i", "--input",
        default=None,
        help="Input file pattern or base directory (e.g. '/path/to/%%s/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root')",
    )
    parser.add_argument(
        "-b", "--base-dir",
        default="/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget",
        help="Base directory containing partition folders [1-1000].",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=10,
        help="Number of concurrent processes (default: 10).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip partitions where both skim output files already exist.",
    )
    parser.add_argument(
        "--no-empty",
        action="store_true",
        help="Do not create files for partitions with 0 selected events.",
    )

    args = parser.parse_args()

    base_dir = args.base_dir
    if args.input:
        inp = args.input.strip()
        if "%s" in inp:
            prefix = inp.split("%s")[0].rstrip("/")
            if prefix and prefix != "/":
                base_dir = prefix
        elif os.path.isdir(inp):
            base_dir = inp
        elif os.path.isfile(inp):
            base_dir = os.path.dirname(os.path.dirname(inp))

    if args.partitions_file and os.path.isfile(args.partitions_file):
        partitions = []
        with open(args.partitions_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and line.isdigit():
                    partitions.append(int(line))
        partitions = sorted(list(set(partitions)))
    else:
        partitions = parse_partitions(args.partitions)
    write_empty = not args.no_empty

    print("=" * 80)
    print(" CHARM DIMUON DIRECT & INDIRECT CATEGORY SKIMMER")
    print("=" * 80)
    print(f" Base Directory       : {base_dir}")
    print(f" Partition Selection  : {len(partitions)} partition(s) specified")
    print(f" Parallel Workers     : {args.jobs}")
    print(f" Skip Existing        : {args.skip_existing}")
    print(f" Write Empty Files    : {write_empty}")
    sample_prefix = "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth" if (args.input and "MCEB" in args.input) else "sndLHC.Genie-TGeant4_dig_2MuTrks_truth"
    print(f" Output 1 (Direct)    : {sample_prefix}_charmDimuonDirect.root")
    print(f" Output 2 (Indirect)  : {sample_prefix}_charmDimuonIndirect.root")
    print("=" * 80)

    t_start = time.time()
    tot_events = 0
    tot_direct = 0
    tot_indirect = 0
    processed_count = 0
    missing_count = 0
    skipped_count = 0

    task_args = [
        (p, base_dir, args.skip_existing, write_empty, args.input)
        for p in partitions
    ]

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(_worker_wrapper, t): t[0] for t in task_args}

        for future in as_completed(futures):
            res = future.result()
            p = res["partition"]
            status = res["status"]

            if status == "success":
                processed_count += 1
                tot_events += res["total"]
                tot_direct += res["direct"]
                tot_indirect += res["indirect"]
                print(
                    f" [Part {p:4d}] Done in {res['time']:5.2f}s | "
                    f"Total: {res['total']:3d} | Direct: {res['direct']:2d} | Indirect: {res['indirect']:2d}"
                )
            elif status == "skipped_existing":
                skipped_count += 1
            elif status == "missing_input":
                missing_count += 1
            else:
                print(f" [Part {p:4d}] Status: {status}")

    t_tot = time.time() - t_start

    print("\n" + "=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    print(f" Processed partitions : {processed_count}")
    print(f" Skipped existing     : {skipped_count}")
    print(f" Missing input files  : {missing_count}")
    print(f" Total events scanned : {tot_events}")
    print(f" Direct signal events : {tot_direct} ({100.0 * tot_direct / max(1, tot_events):.2f}%)")
    print(f" Indirect fake events : {tot_indirect} ({100.0 * tot_indirect / max(1, tot_events):.2f}%)")
    print(f" Total execution time : {t_tot:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
