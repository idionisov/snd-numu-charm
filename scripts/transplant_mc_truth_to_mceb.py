#!/usr/bin/env python3
"""
scripts/transplant_mc_truth_to_mceb.py
--------------------------------------
Transplants complete, unpruned MC truth branches (MCTrack, MuFilterPoint, ScifiPoint,
MCEventHeader) from non-MCEB simulation files into MCEB files, matching events 1-to-1
via EventHeader.GetEventNumber().

Replaces pruned MCEB MCTrack and MCPoints with their original, full-depth simulation
truth collections so downstream analysis tools (tracking QA, event displays, RDataFrame,
and get_hists_numu_truth.py) see full charm hadron decays and complete MCPoint hits.

Usage examples:
    # Single partition test:
    python3 scripts/transplant_mc_truth_to_mceb.py -p 17

    # Parallel processing across all 1000 partitions on lxplus / local:
    python3 scripts/transplant_mc_truth_to_mceb.py -p 1-1000 -j 8

    # Custom file paths / patterns:
python3 scripts/transplant_mc_truth_to_mceb.py \\
    --input-mceb "/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root" \\
    --truth-source "/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_dig.root" \\
    --output "/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_unpruned.root" \\
    -p 1-100 -j 10
"""

from __future__ import annotations

import os
import sys
import re
import time
import argparse
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kBreak


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


def format_partition_path(pattern: str, partition: Any) -> str:
    """Format path pattern replacing %s with partition string."""
    if "%s" in pattern:
        return pattern % str(partition)
    elif "{}" in pattern:
        return pattern.format(str(partition))
    return pattern


def transplant_single_file(
    truth_source_path: str,
    input_mceb_path: str,
    output_path: str,
    tree_name: str = "cbmsim",
    branches_to_transplant: Optional[List[str]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Transplants MC branches from truth_source_path to input_mceb_path mapped by EventNumber.
    Writes result atomically to output_path.
    """
    if branches_to_transplant is None:
        branches_to_transplant = ["MCTrack", "MuFilterPoint", "ScifiPoint", "MCEventHeader"]

    t0 = time.time()
    res = {
        "truth_source": truth_source_path,
        "input_mceb": input_mceb_path,
        "output": output_path,
        "total_mceb": 0,
        "transplanted": 0,
        "status": "ok",
        "error": None,
        "elapsed": 0.0,
    }

    if not os.path.isfile(truth_source_path):
        res["status"] = "error"
        res["error"] = f"Truth source file not found: {truth_source_path}"
        return res

    if not os.path.isfile(input_mceb_path):
        res["status"] = "error"
        res["error"] = f"Input MCEB file not found: {input_mceb_path}"
        return res

    if not force and os.path.isfile(output_path) and os.path.getsize(output_path) > 100000:
        res["status"] = "exists"
        res["error"] = f"Output file already exists: {output_path}"
        return res

    # 1. Open truth source file and build EventNumber -> EntryIndex index
    f_truth = ROOT.TFile.Open(truth_source_path, "READ")
    if not f_truth or f_truth.IsZombie():
        if f_truth:
            f_truth.Close()
        res["status"] = "error"
        res["error"] = f"Could not open truth ROOT file: {truth_source_path}"
        return res

    t_truth = f_truth.Get(tree_name) or f_truth.Get("cbmsim") or f_truth.Get("rawConv")
    if not t_truth:
        f_truth.Close()
        res["status"] = "error"
        res["error"] = f"Tree '{tree_name}' not found in truth file {truth_source_path}"
        return res

    # Fast event header scan
    t_truth.SetBranchStatus("*", 0)
    t_truth.SetBranchStatus("EventHeader*", 1)

    event_map: Dict[int, int] = {}
    for i in range(t_truth.GetEntries()):
        t_truth.GetEntry(i)
        ev_header = getattr(t_truth, "EventHeader", None)
        if ev_header and hasattr(ev_header, "GetEventNumber"):
            event_map[int(ev_header.GetEventNumber())] = i
        else:
            event_map[i] = i

    # 2. Open input MCEB file
    f_mceb = ROOT.TFile.Open(input_mceb_path, "READ")
    if not f_mceb or f_mceb.IsZombie():
        f_truth.Close()
        if f_mceb:
            f_mceb.Close()
        res["status"] = "error"
        res["error"] = f"Could not open input MCEB file: {input_mceb_path}"
        return res

    t_mceb = f_mceb.Get(tree_name) or f_mceb.Get("cbmsim")
    if not t_mceb:
        f_truth.Close()
        f_mceb.Close()
        res["status"] = "error"
        res["error"] = f"Tree '{tree_name}' not found in MCEB file {input_mceb_path}"
        return res

    n_mceb = t_mceb.GetEntries()
    res["total_mceb"] = n_mceb

    # Prepare atomic temporary output file in same directory or /tmp
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    temp_fd, temp_out = tempfile.mkstemp(suffix=".root", prefix="transplant_", dir=out_dir)
    os.close(temp_fd)

    f_out = ROOT.TFile.Open(temp_out, "RECREATE")
    if not f_out or f_out.IsZombie():
        f_truth.Close()
        f_mceb.Close()
        if os.path.exists(temp_out):
            os.remove(temp_out)
        res["status"] = "error"
        res["error"] = f"Could not create temporary output file {temp_out}"
        return res

    # Copy all auxiliary metadata objects (FileHeader, BranchList, etc.)
    for key in f_mceb.GetListOfKeys():
        kname = key.GetName()
        kclass = key.GetClassName()
        if kname == t_mceb.GetName() or kclass == "TTree":
            continue
        obj = key.ReadObj()
        if obj:
            f_out.cd()
            obj.Write(kname, ROOT.TObject.kOverwrite)

    # Disable old pruned MC branches in MCEB tree before cloning structure
    t_mceb.SetBranchStatus("*", 1)
    # Deactivate heavy linker maps and old pruned branches
    deactivate_list = ["Digi_ScifiHits2MCPoints", "Digi_MuFilterHits2MCPoints"] + branches_to_transplant
    for bname in deactivate_list:
        t_mceb.SetBranchStatus(bname + "*", 0)

    f_out.cd()
    t_out = t_mceb.CloneTree(0)

    # Setup unpruned branches from truth tree
    t_truth.SetBranchStatus("*", 0)

    # Branch storage pointers
    has_mctrack = "MCTrack" in branches_to_transplant and t_truth.GetBranch("MCTrack")
    has_mufilter = "MuFilterPoint" in branches_to_transplant and t_truth.GetBranch("MuFilterPoint")
    has_scifi = "ScifiPoint" in branches_to_transplant and t_truth.GetBranch("ScifiPoint")
    has_mchdr = "MCEventHeader" in branches_to_transplant and (t_truth.GetBranch("MCEventHeader") or t_truth.GetBranch("MCEventHeader."))

    mc_tracks = ROOT.TClonesArray("ShipMCTrack") if has_mctrack else None
    mufilter_pts = ROOT.TClonesArray("FairMCPoint") if has_mufilter else None
    scifi_pts = ROOT.TClonesArray("FairMCPoint") if has_scifi else None
    mc_header = ROOT.FairMCEventHeader() if has_mchdr else None

    if has_mctrack:
        t_truth.SetBranchStatus("MCTrack*", 1)
        t_truth.SetBranchAddress("MCTrack", mc_tracks)
        t_out.Branch("MCTrack", mc_tracks)

    if has_mufilter:
        t_truth.SetBranchStatus("MuFilterPoint*", 1)
        t_truth.SetBranchAddress("MuFilterPoint", mufilter_pts)
        t_out.Branch("MuFilterPoint", mufilter_pts)

    if has_scifi:
        t_truth.SetBranchStatus("ScifiPoint*", 1)
        t_truth.SetBranchAddress("ScifiPoint", scifi_pts)
        t_out.Branch("ScifiPoint", scifi_pts)

    if has_mchdr:
        t_truth.SetBranchStatus("MCEventHeader*", 1)
        t_truth.SetBranchAddress("MCEventHeader.", mc_header)
        t_out.Branch("MCEventHeader.", mc_header)

    # 3. Main event loop: Fill MCEB events with mapped truth branches
    for i in range(n_mceb):
        t_mceb.GetEntry(i)
        ev_hdr = getattr(t_mceb, "EventHeader", None)
        ev_num = int(ev_hdr.GetEventNumber()) if ev_hdr and hasattr(ev_hdr, "GetEventNumber") else i

        truth_idx = event_map.get(ev_num, None)
        if truth_idx is not None:
            t_truth.GetEntry(truth_idx)
        else:
            if has_mctrack:
                mc_tracks.Clear()
            if has_mufilter:
                mufilter_pts.Clear()
            if has_scifi:
                scifi_pts.Clear()

        t_out.Fill()

    f_out.cd()
    t_out.Write("", ROOT.TObject.kOverwrite)
    f_out.Close()
    f_mceb.Close()
    f_truth.Close()

    # Move temporary file to final destination atomically
    shutil.move(temp_out, output_path)

    res["transplanted"] = n_mceb
    res["elapsed"] = time.time() - t0
    return res


def _worker_wrapper(args_tuple: Tuple[str, str, str, str, Optional[List[str]], bool]) -> Dict[str, Any]:
    truth_src, in_mceb, out_f, tree_name, branches, force = args_tuple
    return transplant_single_file(
        truth_source_path=truth_src,
        input_mceb_path=in_mceb,
        output_path=out_f,
        tree_name=tree_name,
        branches_to_transplant=branches,
        force=force,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Transplant unpruned MC truth branches (MCTrack, MuFilterPoint, ScifiPoint) from non-MCEB into MCEB files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input-mceb",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root",
        help="Path or pattern for input MCEB file(s)",
    )
    parser.add_argument(
        "--truth-source",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_dig.root",
        help="Path or pattern for unpruned non-MCEB truth simulation file(s)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path or pattern for output file(s). If omitted, overwrites input-mceb in place.",
    )
    parser.add_argument(
        "-p", "--partitions",
        type=str,
        default="1-1000",
        help="Partitions to process, e.g. '17', '1-50', or '1-1000'",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=8,
        help="Number of concurrent parallel worker processes",
    )
    parser.add_argument(
        "-t", "--tree",
        type=str,
        default="cbmsim",
        help="TTree name containing detector and MC branches",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force overwrite existing output files",
    )

    args = parser.parse_args()

    partitions = parse_partitions(args.partitions)
    if partitions is None:
        partitions = list(range(1, 1001))

    # Determine tasks
    tasks = []
    for p in partitions:
        truth_src = format_partition_path(args.truth_source, p)
        in_mceb = format_partition_path(args.input_mceb, p)
        out_f = format_partition_path(args.output, p) if args.output else in_mceb
        if os.path.isfile(truth_src) and os.path.isfile(in_mceb):
            tasks.append((truth_src, in_mceb, out_f, args.tree, None, args.force))
        elif not os.path.isfile(in_mceb):
            pass

    print("=" * 80)
    print(" SND@LHC: MC Truth Branch Transplanter (Non-MCEB -> MCEB)")
    print("=" * 80)
    print(f" Input MCEB pattern     : {args.input_mceb}")
    print(f" Truth source pattern   : {args.truth_source}")
    print(f" Output pattern         : {args.output or '[In-Place Overwrite]'}")
    print(f" Target Partitions      : {len(partitions)} ({partitions[0]}..{partitions[-1]})")
    print(f" Valid Matched Tasks    : {len(tasks)}")
    print(f" Concurrent Workers     : {args.jobs}")
    print(f" Force Overwrite        : {args.force}")
    print("=" * 80)

    if not tasks:
        print("ERROR: No valid tasks found to process. Verify input paths.")
        sys.exit(1)

    start_time = time.time()
    n_success = 0
    n_exists = 0
    n_failed = 0
    total_events = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(_worker_wrapper, t): t for t in tasks}

        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            p_val = re.search(r'/(\d+)/', res["input_mceb"])
            part_str = p_val.group(1) if p_val else str(i)

            if res["status"] == "ok":
                n_success += 1
                total_events += res["transplanted"]
                print(f" [{i:>4}/{len(tasks)}] Partition {part_str:>4} | OK | {res['transplanted']} events transplanted in {res['elapsed']:.1f}s")
            elif res["status"] == "exists":
                n_exists += 1
                print(f" [{i:>4}/{len(tasks)}] Partition {part_str:>4} | SKIPPED | Output exists")
            else:
                n_failed += 1
                print(f" [{i:>4}/{len(tasks)}] Partition {part_str:>4} | FAILED  | Error: {res['error']}")

    total_time = time.time() - start_time
    print("=" * 80)
    print(f" All done in {total_time:.1f}s ({total_time/60.0:.1f} min)")
    print(f" Success: {n_success} | Skipped: {n_exists} | Failed: {n_failed} | Events: {total_events}")
    print("=" * 80)


if __name__ == "__main__":
    main()
