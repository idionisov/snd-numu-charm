#!/usr/bin/env python3
"""
scripts/map_mctruth_to_mceb.py
------------------------------
Maps the flat MCTruth tree from an original simulation file (e.g. standard DIG/truth)
to a Monte Carlo Event Builder (MCEB) file via FairEventHeader event matching.

Produces an output file with identical MCEB structure and branches (cbmsim, BranchList, etc.),
augmented with the complete, uncorrupted MCTruth tree mapped 1-to-1 to each MCEB event.

Usage examples:
  # Single file mode:
  python3 scripts/map_mctruth_to_mceb.py \\
      --input-truth /path/to/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root \\
      --input-mceb /path/to/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root \\
      --output /path/to/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root

  # Batch across partitions:
  python3 scripts/map_mctruth_to_mceb.py \\
      --input-truth "/eos/.../%s/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root" \\
      --input-mceb "/eos/.../%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root" \\
      --output "/eos/.../%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root" \\
      -p 1-100 -j 8
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import ROOT
ROOT.gROOT.SetBatch(True)


def parse_partitions(part_arg: Optional[str]) -> Optional[List[int]]:
    """Parse partition range or comma-separated list."""
    if not part_arg or part_arg.strip().lower() == "all":
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


def format_partition_path(pattern: str, partition: int | str) -> str:
    """Format partition pattern if placeholder present."""
    p_str = str(partition)
    count = pattern.count("%s")
    if count > 0:
        return pattern % tuple(p_str for _ in range(count))
    return pattern


def map_single_file(
    input_truth_path: str,
    input_mceb_path: str,
    output_path: str,
    truth_tree_name: str = "mctruth",
) -> Dict[str, Any]:
    """
    Map mctruth tree from input_truth to input_mceb by EventHeader.GetEventNumber().
    Overwrites output_path if it already exists.
    """
    t0 = time.time()
    res = {
        "input_truth": input_truth_path,
        "input_mceb": input_mceb_path,
        "output": output_path,
        "total_mceb": 0,
        "mapped_truth": 0,
        "status": "ok",
        "error": None,
        "elapsed": 0.0,
    }

    if not os.path.isfile(input_truth_path):
        res["status"] = "error"
        res["error"] = f"Truth file not found: {input_truth_path}"
        return res

    if not os.path.isfile(input_mceb_path):
        res["status"] = "error"
        res["error"] = f"MCEB file not found: {input_mceb_path}"
        return res

    # 1. Open truth file and build EventNumber -> EntryIndex index
    f_truth = ROOT.TFile.Open(input_truth_path, "READ")
    if not f_truth or f_truth.IsZombie():
        if f_truth:
            f_truth.Close()
        res["status"] = "error"
        res["error"] = f"Could not open truth ROOT file: {input_truth_path}"
        return res

    t_truth = f_truth.Get(truth_tree_name)
    if not t_truth:
        for cand in ["mctruth", "truth", "mc_truth"]:
            t_truth = f_truth.Get(cand)
            if t_truth:
                break

    if not t_truth:
        f_truth.Close()
        res["status"] = "error"
        res["error"] = f"Truth tree '{truth_tree_name}' not found in {input_truth_path}"
        return res

    t_cbm_truth = f_truth.Get("cbmsim")
    if not t_cbm_truth:
        f_truth.Close()
        res["status"] = "error"
        res["error"] = f"'cbmsim' tree not found in truth file {input_truth_path}"
        return res

    # Fast event header scan by deactivating unneeded branches
    t_cbm_truth.SetBranchStatus("*", 0)
    t_cbm_truth.SetBranchStatus("EventHeader*", 1)

    event_map: Dict[int, int] = {}
    for i in range(t_cbm_truth.GetEntries()):
        t_cbm_truth.GetEntry(i)
        ev_header = getattr(t_cbm_truth, "EventHeader", None)
        if ev_header:
            event_map[int(ev_header.GetEventNumber())] = i
        else:
            event_map[i] = i

    t_cbm_truth.SetBranchStatus("*", 1)

    # 2. Open MCEB file and read event numbers
    f_mceb = ROOT.TFile.Open(input_mceb_path, "READ")
    if not f_mceb or f_mceb.IsZombie():
        if f_mceb:
            f_mceb.Close()
        f_truth.Close()
        res["status"] = "error"
        res["error"] = f"Could not open MCEB ROOT file: {input_mceb_path}"
        return res

    t_cbm_mceb = f_mceb.Get("cbmsim")
    if not t_cbm_mceb:
        f_mceb.Close()
        f_truth.Close()
        res["status"] = "error"
        res["error"] = f"'cbmsim' tree not found in MCEB file {input_mceb_path}"
        return res

    n_mceb = t_cbm_mceb.GetEntries()
    res["total_mceb"] = n_mceb

    t_cbm_mceb.SetBranchStatus("*", 0)
    t_cbm_mceb.SetBranchStatus("EventHeader*", 1)

    mceb_events: List[Tuple[int, Optional[int]]] = []
    for j in range(n_mceb):
        t_cbm_mceb.GetEntry(j)
        ev_header = getattr(t_cbm_mceb, "EventHeader", None)
        ev_num = int(ev_header.GetEventNumber()) if ev_header else j
        truth_idx = event_map.get(ev_num, None)
        mceb_events.append((ev_num, truth_idx))

    t_cbm_mceb.SetBranchStatus("*", 1)

    # 3. Prepare output file (overwrite existing)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    f_out = ROOT.TFile.Open(output_path, "RECREATE")
    if not f_out or f_out.IsZombie():
        if f_out:
            f_out.Close()
        f_mceb.Close()
        f_truth.Close()
        res["status"] = "error"
        res["error"] = f"Could not create output ROOT file: {output_path}"
        return res

    # Clone MCEB cbmsim tree
    out_cbm = t_cbm_mceb.CloneTree(-1, "fast")
    out_cbm.SetName("cbmsim")

    # Clone truth tree structure
    t_truth.SetBranchStatus("*", 1)
    out_truth = t_truth.CloneTree(0)
    out_truth.SetName(truth_tree_name)
    out_truth.SetTitle(t_truth.GetTitle() or "Categorized Neutrino Truth Observables")

    mapped_count = 0
    for ev_num, truth_idx in mceb_events:
        if truth_idx is not None:
            t_truth.GetEntry(truth_idx)
            out_truth.Fill()
            mapped_count += 1
        else:
            # Fallback for unmapped event (e.g. zero entry)
            out_truth.Fill()

    res["mapped_truth"] = mapped_count

    # Write trees to output
    out_cbm.Write("cbmsim", ROOT.TObject.kOverwrite)
    out_truth.Write(truth_tree_name, ROOT.TObject.kOverwrite)

    # Copy any auxiliary objects from MCEB file (e.g. BranchList)
    branch_list = f_mceb.Get("BranchList")
    if branch_list:
        branch_list.Write("BranchList", ROOT.TObject.kSingleKey)

    for key in f_mceb.GetListOfKeys():
        kname = key.GetName()
        if kname not in ("cbmsim", truth_tree_name, "BranchList"):
            obj = f_mceb.Get(kname)
            if obj:
                obj.Write(kname, ROOT.TObject.kSingleKey)

    f_out.Close()
    f_mceb.Close()
    f_truth.Close()

    res["elapsed"] = time.time() - t0
    return res


def main():
    parser = argparse.ArgumentParser(
        description="Map MCTruth branch/tree from original simulation files to MCEB files via EventHeader matching."
    )
    parser.add_argument(
        "--input-truth",
        required=True,
        help="Path or pattern for original truth file (e.g. sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root)",
    )
    parser.add_argument(
        "--input-mceb",
        required=True,
        help="Path or pattern for MCEB file (e.g. sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path or pattern for output file (replaces file if it already exists)",
    )
    parser.add_argument(
        "--truth-tree",
        default="mctruth",
        help="Name of truth tree (default: 'mctruth')",
    )
    parser.add_argument(
        "-p", "--partitions",
        default=None,
        help="Optional partition range or list for batch processing (e.g. '1-100', '1,2,5')",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of parallel worker processes for batch mode (default: 4)",
    )

    args = parser.parse_args()

    partitions = parse_partitions(args.partitions)

    print("=" * 80)
    print(" SND@LHC MCTRUTH TO MCEB MAPPER & TREE TRANSPLANTER")
    print("=" * 80)
    print(f" Input Truth Pattern : {args.input_truth}")
    print(f" Input MCEB Pattern  : {args.input_mceb}")
    print(f" Output Pattern      : {args.output}")
    print(f" Truth Tree Name     : {args.truth_tree}")
    print(f" Overwrite Mode      : Always Recreate (RECREATE)")
    if partitions:
        print(f" Partitions          : {len(partitions)} ({partitions[0]}..{partitions[-1]})")
        print(f" Workers             : {args.jobs}")
    print("=" * 80)

    t0 = time.time()

    if partitions:
        tasks = []
        for p in partitions:
            in_t = format_partition_path(args.input_truth, p)
            in_m = format_partition_path(args.input_mceb, p)
            out_f = format_partition_path(args.output, p)
            tasks.append((p, in_t, in_m, out_f))

        results = []
        if args.jobs > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = {
                    executor.submit(map_single_file, in_t, in_m, out_f, args.truth_tree): p
                    for p, in_t, in_m, out_f in tasks
                }
                for fut in as_completed(futures):
                    p = futures[fut]
                    res = fut.result()
                    res["partition"] = p
                    results.append(res)
        else:
            for p, in_t, in_m, out_f in tasks:
                res = map_single_file(in_t, in_m, out_f, args.truth_tree)
                res["partition"] = p
                results.append(res)

        results.sort(key=lambda x: x.get("partition", 0))
        n_ok = sum(1 for r in results if r["status"] == "ok")
        n_err = sum(1 for r in results if r["status"] != "ok")

        for r in results:
            p = r.get("partition", "-")
            status_str = "OK" if r["status"] == "ok" else "FAIL"
            err_str = f" ({r['error']})" if r["error"] else ""
            print(f" [Part {p:>4}] {status_str:<4} | MCEB: {r['total_mceb']:>4} | Mapped Truth: {r['mapped_truth']:>4} | {r['elapsed']:>5.2f}s{err_str}")

        print("=" * 80)
        print(f" SUMMARY: {n_ok}/{len(results)} partitions mapped successfully ({n_err} failed)")
        print(f" Total elapsed time: {time.time() - t0:.2f}s")
        print("=" * 80)

        if n_err > 0:
            sys.exit(1)
    else:
        # Single file execution
        res = map_single_file(args.input_truth, args.input_mceb, args.output, args.truth_tree)
        if res["status"] == "ok":
            print(f"\n Mapping completed successfully in {res['elapsed']:.2f}s:")
            print(f"   MCEB Events  : {res['total_mceb']}")
            print(f"   Mapped Truth : {res['mapped_truth']}")
            print(f"   Output File  : {res['output']}")
            print("=" * 80)
        else:
            print(f"\nERROR: Mapping failed: {res['error']}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
