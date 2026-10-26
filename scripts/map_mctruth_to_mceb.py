#!/usr/bin/env python3
"""
scripts/map_mctruth_to_mceb.py
------------------------------
Maps and transfers unpruned MC truth from original non-MCEB simulation files
(e.g. sndLHC.Genie-TGeant4_dig.root or sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root)
into Monte Carlo Event Builder (MCEB) files via EventHeader event matching.

Performs two crucial operations in a single atomic pass:
  1. cbmsim truth transplantation:
     Replaces pruned MCTrack, ScifiPoint, and MuFilterPoint branches in MCEB cbmsim
     with the unpruned truth from the non-MCEB file, writing a new cycle (cbmsim;2)
     while preserving cbmsim;1 and all MCEB hits, reconstructed tracks, and metadata.
  2. mctruth tree extraction / mapping:
     If the truth source already contains an mctruth tree, it is mapped 1-to-1 by event number.
     If the truth source only contains raw cbmsim truth, the comprehensive flat mctruth
     tree is extracted on the fly via NeutrinoTruthProcessor.

Can be run:
  - BEFORE tracking: on raw digitized MCEB files (sndLHC.Genie-TGeant4_MCEB_dig.root)
  - AFTER tracking: on reconstructed MCEB files (sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root)
  - IN-PLACE: by setting --output identical to --input-mceb (saves 100% extra disk space)

Usage examples:
  # Single partition in-place:
  python3 scripts/map_mctruth_to_mceb.py \\
      -t /eos/.../17/sndLHC.Genie-TGeant4_dig.root \\
      -i /eos/.../17/sndLHC.Genie-TGeant4_MCEB_dig.root \\
      -o /eos/.../17/sndLHC.Genie-TGeant4_MCEB_dig.root

  # Batch across partitions 1-1000 with 10 workers (auto-skips completed):
  python3 scripts/map_mctruth_to_mceb.py \\
      -t "/eos/.../%s/sndLHC.Genie-TGeant4_dig.root" \\
      -i "/eos/.../%s/sndLHC.Genie-TGeant4_MCEB_dig.root" \\
      -o "/eos/.../%s/sndLHC.Genie-TGeant4_MCEB_dig.root" \\
      -p 1-1000 -j 10
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import tempfile
import argparse
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import ROOT
ROOT.gROOT.SetBatch(True)

# Repository paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_ddf_path = os.environ.get("ddfUtils", "/afs/cern.ch/user/i/idioniso/ddfUtils")
if os.path.exists(_ddf_path) and _ddf_path not in sys.path:
    sys.path.insert(0, _ddf_path)

# Global worker processor caching
_worker_processor = None
_worker_cfg = None


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


def _get_event_number(tree: ROOT.TTree, entry_idx: int) -> int:
    """Robustly extract event number from TTree entry via leaves or header objects."""
    for leaf_name in [
        "EventHeader.fEventNumber",
        "EventHeader..fEventNumber",
        "fEventNumber",
        "MCEventHeader.fEventId",
        "MCEventHeader.fEventNumber",
    ]:
        leaf = tree.GetLeaf(leaf_name)
        if leaf:
            return int(leaf.GetValue())

    for attr in ["EventHeader", "EventHeader.", "MCEventHeader", "MCEventHeader."]:
        hdr = getattr(tree, attr, None)
        if hdr:
            if hasattr(hdr, "GetEventNumber"):
                return int(hdr.GetEventNumber())
            if hasattr(hdr, "GetEventId"):
                return int(hdr.GetEventId())
    return entry_idx


def _get_processor(config_path: Optional[str] = None):
    """Lazily initialize NeutrinoTruthProcessor for dynamic extraction."""
    global _worker_processor, _worker_cfg
    if _worker_processor is None:
        from snd import load_trident_libraries, load_config, build_processor
        load_trident_libraries(_repo_root)
        if config_path is None or not os.path.isfile(config_path):
            config_path = os.path.join(_repo_root, "config", "mctruth_neutrinos_config.yaml")
        _worker_cfg = load_config(config_path) if os.path.isfile(config_path) else {}
        _worker_processor = build_processor(_worker_cfg.get("processor", {}))
    return _worker_processor, _worker_cfg


def map_and_transplant_single_file(
    input_truth_path: str,
    input_mceb_path: str,
    output_path: str,
    truth_tree_name: str = "mctruth",
    config_path: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Transplants unpruned cbmsim truth (MCTrack, ScifiPoint, MuFilterPoint) into cbmsim;2
    and maps/extracts the flat mctruth tree into output_path.
    Stages safely in local /tmp to avoid EOS XRootD update crashes.
    """
    t0 = time.time()
    res = {
        "input_truth": input_truth_path,
        "input_mceb": input_mceb_path,
        "output": output_path,
        "total_mceb": 0,
        "transplanted": 0,
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

    # Fast check: skip if output already exists with valid cycle 2 and truth tree
    if not force and os.path.isfile(output_path) and os.path.getsize(output_path) > 1000:
        try:
            f_chk = ROOT.TFile.Open(output_path, "READ")
            if f_chk and not f_chk.IsZombie():
                has_c2 = bool(f_chk.GetKey("cbmsim", 2))
                has_truth = bool(f_chk.GetKey(truth_tree_name))
                f_chk.Close()
                if has_c2 and has_truth:
                    res["status"] = "exists"
                    res["error"] = f"Already transplanted with cbmsim;2 and {truth_tree_name}: {output_path}"
                    return res
            elif f_chk:
                f_chk.Close()
        except Exception:
            pass

    # 1. Open truth file directly and build EventNumber -> EntryIndex index
    f_truth = ROOT.TFile.Open(input_truth_path, "READ")
    if not f_truth or f_truth.IsZombie():
        if f_truth:
            f_truth.Close()
        res["status"] = "error"
        res["error"] = f"Could not open truth ROOT file: {input_truth_path}"
        return res

    t_cbm_truth = f_truth.Get("cbmsim") or f_truth.Get("rawConv") or f_truth.Get("events")
    if not t_cbm_truth:
        f_truth.Close()
        res["status"] = "error"
        res["error"] = f"'cbmsim' tree not found in truth file {input_truth_path}"
        return res

    # Fast event header scan by deactivating all heavy branches during index build
    t_cbm_truth.SetBranchStatus("*", 0)
    t_cbm_truth.SetBranchStatus("EventHeader*", 1)
    t_cbm_truth.SetBranchStatus("MCEventHeader*", 1)

    event_map: Dict[int, int] = {}
    for i in range(t_cbm_truth.GetEntries()):
        t_cbm_truth.GetEntry(i)
        ev_num = _get_event_number(t_cbm_truth, i)
        event_map[ev_num] = i

    # Check whether truth file already has pre-computed mctruth tree
    t_precomputed_truth = None
    for cand in [truth_tree_name, "mctruth", "truth", "mc_truth"]:
        t_cand = f_truth.Get(cand)
        if t_cand and isinstance(t_cand, ROOT.TTree):
            t_precomputed_truth = t_cand
            break

    # 2. Setup local staging in /tmp to prevent EOS XRootD update crashes
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    temp_fd, temp_out = tempfile.mkstemp(suffix=".root", prefix="map_mctruth_")
    os.close(temp_fd)

    f_up = None

    try:
        # Copy input MCEB file to temp_out to retain full content & cycle 1
        shutil.copyfile(input_mceb_path, temp_out)

        f_up = ROOT.TFile.Open(temp_out, "UPDATE")
        if not f_up or f_up.IsZombie():
            f_truth.Close()
            res["status"] = "error"
            res["error"] = f"Could not open temp file {temp_out} in UPDATE mode"
            return res

        t_old = f_up.Get("cbmsim;1") or f_up.Get("cbmsim")
        if not t_old:
            f_up.Close()
            f_truth.Close()
            res["status"] = "error"
            res["error"] = f"Tree 'cbmsim' not found in MCEB file {input_mceb_path}"
            return res

        n_mceb = t_old.GetEntries()
        res["total_mceb"] = n_mceb

        # Keep everything from input file as is, except the branches to be replaced
        branches_to_transplant = ["MCTrack", "ScifiPoint", "MuFilterPoint"]
        t_old.SetBranchStatus("*", 1)
        for bname in branches_to_transplant:
            t_old.SetBranchStatus(bname + "*", 0)

        f_up.cd()
        t_new = t_old.CloneTree(0)
        # Prevent TTree::ChangeFile rollover and disable intermediate autosave cycles
        t_new.SetMaxTreeSize(1000000000000)
        t_new.SetAutoSave(0)

        # Setup unpruned branches from truth source cbmsim
        t_cbm_truth.SetBranchStatus("*", 0)
        has_mctrack = bool(t_cbm_truth.GetBranch("MCTrack"))
        has_scifi = bool(t_cbm_truth.GetBranch("ScifiPoint"))
        has_mufilter = bool(t_cbm_truth.GetBranch("MuFilterPoint"))

        mc_tracks = ROOT.TClonesArray("ShipMCTrack") if has_mctrack else None
        scifi_pts = ROOT.TClonesArray("FairMCPoint") if has_scifi else None
        mufilter_pts = ROOT.TClonesArray("FairMCPoint") if has_mufilter else None

        if has_mctrack:
            t_cbm_truth.SetBranchStatus("MCTrack*", 1)
            t_cbm_truth.SetBranchAddress("MCTrack", mc_tracks)
            t_new.Branch("MCTrack", mc_tracks)

        if has_scifi:
            t_cbm_truth.SetBranchStatus("ScifiPoint*", 1)
            t_cbm_truth.SetBranchAddress("ScifiPoint", scifi_pts)
            t_new.Branch("ScifiPoint", scifi_pts)

        if has_mufilter:
            t_cbm_truth.SetBranchStatus("MuFilterPoint*", 1)
            t_cbm_truth.SetBranchAddress("MuFilterPoint", mufilter_pts)
            t_new.Branch("MuFilterPoint", mufilter_pts)

        # 3. Setup mctruth tree
        f_up.cd()
        if t_precomputed_truth is not None:
            # Map pre-existing mctruth tree directly
            t_precomputed_truth.SetBranchStatus("*", 1)
            out_truth = t_precomputed_truth.CloneTree(0)
            out_truth.SetName(truth_tree_name)
            dynamic_extraction = False
            processor = None
            truth_buffers = None
            all_tiers = None
        else:
            # Dynamically extract on the fly via NeutrinoTruthProcessor
            from snd import setup_truth_branches
            from snd.filter import fill_truth_buffers
            try:
                from scripts.mctruth_neutrinos import resolve_weight_function
            except ImportError:
                resolve_weight_function = lambda st, cfg=None: (lambda ev, inf: 1.0)
            processor, cfg = _get_processor(config_path)
            sample_type = cfg.get("sample_type", "neutrino") if cfg else "neutrino"
            weight_fn = resolve_weight_function(sample_type, cfg=cfg)
            out_truth = ROOT.TTree(truth_tree_name, "Categorized Neutrino Truth Observables")
            all_tiers = {"universal", "lepton", "charm", "decay_muon", "dimuon"}
            truth_buffers = setup_truth_branches([out_truth], all_tiers)
            dynamic_extraction = True

        out_truth.SetMaxTreeSize(1000000000000)
        out_truth.SetAutoSave(0)

        mapped_count = 0
        last_truth_idx = -1
        cached_info = None
        cached_raw_w = None
        cached_mc_w = None
        cached_ev_weight = 1.0

        # 4. Main Event Loop
        for j in range(n_mceb):
            t_old.GetEntry(j)
            ev_num = _get_event_number(t_old, j)

            truth_idx = event_map.get(ev_num, None)
            if truth_idx is not None:
                if truth_idx != last_truth_idx:
                    t_cbm_truth.GetEntry(truth_idx)
                    last_truth_idx = truth_idx
                    if dynamic_extraction:
                        cached_info = processor.processMuonNeutrino(mc_tracks, mufilter_pts)
                        cached_ev_weight = 1.0
                        cached_raw_w = None
                        cached_mc_w = None
                        if weight_fn is not None:
                            try:
                                res_w = weight_fn(t_cbm_truth, cached_info)
                                if isinstance(res_w, dict):
                                    cached_raw_w = float(res_w["raw_weight"]) if "raw_weight" in res_w else None
                                    cached_mc_w = float(res_w["mc_weight"]) if "mc_weight" in res_w else None
                                    cached_ev_weight = float(res_w.get("mctruth_weight", cached_mc_w if cached_mc_w is not None else 1.0))
                                elif isinstance(res_w, (tuple, list)) and len(res_w) >= 2:
                                    cached_raw_w = float(res_w[0])
                                    cached_mc_w = float(res_w[1])
                                    cached_ev_weight = float(res_w[2]) if len(res_w) >= 3 else (cached_mc_w if cached_mc_w is not None else 1.0)
                                elif res_w is not None:
                                    cached_ev_weight = float(res_w)
                            except Exception:
                                cached_ev_weight = 1.0

                if not dynamic_extraction:
                    t_precomputed_truth.GetEntry(truth_idx)
                    out_truth.Fill()
                else:
                    fill_truth_buffers(
                        truth_buffers,
                        cached_info,
                        j,
                        all_tiers,
                        mctruth_weight=cached_ev_weight,
                        mc_tracks=mc_tracks,
                        mu_filter_points=mufilter_pts,
                        raw_weight=cached_raw_w,
                        mc_weight=cached_mc_w,
                    )
                    out_truth.Fill()
                mapped_count += 1
            else:
                if has_mctrack:
                    mc_tracks.Clear()
                if has_scifi:
                    scifi_pts.Clear()
                if has_mufilter:
                    mufilter_pts.Clear()
                out_truth.Fill()

            t_new.Fill()

        res["transplanted"] = n_mceb
        res["mapped_truth"] = mapped_count

        f_up.cd()
        # Write cbmsim;2 preserving cbmsim;1
        t_new.Write("cbmsim")
        out_truth.Write(truth_tree_name, ROOT.TObject.kOverwrite)

        if f_up and f_up.IsOpen():
            f_up.Close()
        if f_truth and f_truth.IsOpen():
            f_truth.Close()

        # Move finalized file to destination atomically
        shutil.move(temp_out, output_path)

        res["elapsed"] = time.time() - t0
        return res
    finally:
        if f_up and f_up.IsOpen():
            try:
                f_up.Close()
            except Exception:
                pass
        if f_truth and f_truth.IsOpen():
            try:
                f_truth.Close()
            except Exception:
                pass
        if os.path.exists(temp_out):
            try:
                os.remove(temp_out)
            except OSError:
                pass


def _worker_wrapper(args_tuple: Tuple[str, str, str, str, Optional[str], bool]) -> Dict[str, Any]:
    in_t, in_m, out_f, tree_name, cfg_p, force = args_tuple
    return map_and_transplant_single_file(
        input_truth_path=in_t,
        input_mceb_path=in_m,
        output_path=out_f,
        truth_tree_name=tree_name,
        config_path=cfg_p,
        force=force,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Transplant unpruned MC truth into MCEB cbmsim and extract/map mctruth tree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t", "--input-truth", "--truth-source",
        dest="input_truth",
        required=True,
        help="Path or pattern for non-MCEB truth file (e.g. sndLHC.Genie-TGeant4_dig.root)",
    )
    parser.add_argument(
        "-i", "--input-mceb",
        required=True,
        help="Path or pattern for MCEB file (e.g. sndLHC.Genie-TGeant4_MCEB_dig.root or ..._2MuTrks.root)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path or pattern for output file (can match --input-mceb for in-place update)",
    )
    parser.add_argument(
        "--truth-tree",
        default="mctruth",
        help="Name of flat truth tree (default: 'mctruth')",
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to neutrino truth processor configuration YAML",
    )
    parser.add_argument(
        "-p", "--partitions",
        default=None,
        help="Optional partition range or list for batch processing (e.g. '1-100', '1,2,5')",
    )
    parser.add_argument(
        "--partitions-file", "--args-file",
        dest="partitions_file",
        default=None,
        help="Text file containing partition IDs (one per line)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=8,
        help="Number of parallel worker processes for batch mode",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force overwrite existing output files (default: skip already completed files)",
    )

    args = parser.parse_args()

    partitions = []
    if args.partitions_file and os.path.isfile(args.partitions_file):
        with open(args.partitions_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and line.isdigit():
                    partitions.append(int(line))
        partitions = sorted(list(set(partitions)))
    elif args.partitions:
        partitions = parse_partitions(args.partitions)

    print("=" * 80)
    print(" SND@LHC MCEB TRUTH TRANSPLANTER & MCTRUTH EXTRACTOR")
    print("=" * 80)
    print(f" Truth Source Pattern : {args.input_truth}")
    print(f" Input MCEB Pattern   : {args.input_mceb}")
    print(f" Output Target Pattern: {args.output}")
    print(f" Truth Tree Name      : {args.truth_tree}")
    print(f" Force Overwrite      : {args.force}")
    if partitions:
        print(f" Partitions           : {len(partitions)} ({partitions[0]}..{partitions[-1]})")
        print(f" Workers              : {args.jobs}")
    print("=" * 80)

    t0 = time.time()

    if partitions:
        tasks = []
        for p in partitions:
            in_t = format_partition_path(args.input_truth, p)
            in_m = format_partition_path(args.input_mceb, p)
            out_f = format_partition_path(args.output, p)
            tasks.append((in_t, in_m, out_f, args.truth_tree, args.config, args.force))

        results = []
        if args.jobs > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = {
                    executor.submit(_worker_wrapper, t_args): partitions[idx]
                    for idx, t_args in enumerate(tasks)
                }
                for fut in as_completed(futures):
                    p = futures[fut]
                    try:
                        res = fut.result()
                        res["partition"] = p
                    except Exception as e:
                        res = {
                            "partition": p,
                            "status": "error",
                            "error": str(e),
                            "total_mceb": 0,
                            "mapped_truth": 0,
                            "elapsed": 0.0,
                        }
                    results.append(res)
                    p_str = str(p)
                    stat = res["status"].upper()
                    if stat == "OK":
                        print(f" [{len(results):>4}/{len(tasks)}] Partition {p_str:>4} | OK | {res['total_mceb']} events transplanted in {res['elapsed']:.1f}s")
                    elif stat == "EXISTS":
                        print(f" [{len(results):>4}/{len(tasks)}] Partition {p_str:>4} | SKIPPED | Output exists with cbmsim;2 and {args.truth_tree}")
                    else:
                        print(f" [{len(results):>4}/{len(tasks)}] Partition {p_str:>4} | FAIL | Error: {res['error']}")
        else:
            for idx, t_args in enumerate(tasks):
                p = partitions[idx]
                try:
                    res = _worker_wrapper(t_args)
                    res["partition"] = p
                except Exception as e:
                    res = {
                        "partition": p,
                        "status": "error",
                        "error": str(e),
                        "total_mceb": 0,
                        "mapped_truth": 0,
                        "elapsed": 0.0,
                    }
                results.append(res)
                p_str = str(p)
                stat = res["status"].upper()
                if stat == "OK":
                    print(f" [{len(results):>4}/{len(tasks)}] Partition {p_str:>4} | OK | {res['total_mceb']} events transplanted in {res['elapsed']:.1f}s")
                elif stat == "EXISTS":
                    print(f" [{len(results):>4}/{len(tasks)}] Partition {p_str:>4} | SKIPPED | Output exists with cbmsim;2 and {args.truth_tree}")
                else:
                    print(f" [{len(results):>4}/{len(tasks)}] Partition {p_str:>4} | FAIL | Error: {res['error']}")

        results.sort(key=lambda x: x.get("partition", 0))
        n_ok = sum(1 for r in results if r["status"] == "ok")
        n_exists = sum(1 for r in results if r["status"] == "exists")
        n_err = sum(1 for r in results if r["status"] == "error")

        print("=" * 80)
        print(f" SUMMARY: {n_ok} ok, {n_exists} skipped, {n_err} failed out of {len(results)} partitions")
        print(f" Total elapsed time: {time.time() - t0:.2f}s")
        print("=" * 80)

        if n_err > 0:
            sys.exit(1)
    else:
        # Single file execution
        res = map_and_transplant_single_file(
            args.input_truth,
            args.input_mceb,
            args.output,
            truth_tree_name=args.truth_tree,
            config_path=args.config,
            force=args.force,
        )
        if res["status"] in ("ok", "exists"):
            print(f"\n Processing completed ({res['status']}) in {res['elapsed']:.2f}s:")
            print(f"   MCEB Events  : {res['total_mceb']}")
            print(f"   Mapped Truth : {res['mapped_truth']}")
            print(f"   Output File  : {res['output']}")
            print("=" * 80)
        else:
            print(f"\nERROR: Processing failed: {res['error']}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
