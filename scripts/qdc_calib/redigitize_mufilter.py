#!/usr/bin/env python3
"""
scripts/qdc_calib/redigitize_mufilter.py
---------------------------------------
Re-digitizes MuFilter hits in Monte Carlo simulation files (GENIE newP8 partitions and PMU2022)
using the calibrated MinPE physics response (Birks quenching, edge effect, arrival dispersion,
and channel-specific SiPM reduction factor k).

Writes a new cycle `cbmsim;2` into the ROOT file while preserving original simulation branches.
Subsequent calls to `f->Get("cbmsim")` will automatically retrieve the calibrated `cbmsim;2` tree.

Usage:
  # Single test partition:
  python3 scripts/qdc_calib/redigitize_mufilter.py --file /path/to/sndLHC.Genie-TGeant4_dig.root

  # 1000 GENIE partitions in parallel:
  python3 scripts/qdc_calib/redigitize_mufilter.py --preset newP8 -j 16

  # PMU sample:
  python3 scripts/qdc_calib/redigitize_mufilter.py --pmu -j 8
"""

from __future__ import annotations

import os
import sys
import glob
import time
import argparse
from typing import List, Optional
import multiprocessing as mp

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import load_trident_libraries
load_trident_libraries(_repo_root)

import scripts.qdc_calib.min_pe as mpe


def redigitize_single_file(file_path: str, is_testbeam: bool = False, year: int = 2022, max_events: Optional[int] = None) -> bool:
    """
    Reads cbmsim from file_path, filters Digi_MuFilterHits through calibrated MinPE model,
    and overwrites cycle `cbmsim;2` (or writes cbmsim).
    Subsequent calls to `f->Get("cbmsim")` will automatically retrieve the calibrated `cbmsim;2` tree.
    """
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return False

    t0 = time.time()
    calc = mpe.MinPE(year=year, is_testbeam=is_testbeam, include_edge_effect=True, include_dispersion=True)
    th_pe = calc.get_sipm_threshold_pe()

    # Open file in UPDATE mode
    f = ROOT.TFile.Open(file_path, "UPDATE")
    if not f or f.IsZombie():
        print(f"[!] Cannot open {file_path} in UPDATE mode.")
        return False

    t_src = f.Get("cbmsim;2") or f.Get("cbmsim;1") or f.Get("cbmsim")
    if not t_src:
        print(f"[!] No cbmsim tree in {file_path}")
        f.Close()
        return False

    n_entries = t_src.GetEntries()
    if max_events is not None and max_events > 0:
        n_entries = min(n_entries, max_events)

    # Disable original Digi_MuFilterHits from auto-cloning
    has_digi_branch = (t_src.GetBranch("Digi_MuFilterHits") is not None)
    if has_digi_branch:
        t_src.SetBranchStatus("Digi_MuFilterHits", 0)

    # Clone structure
    t_new = t_src.CloneTree(0)

    # Re-enable source branch to read original hits
    orig_hits = ROOT.TClonesArray("MuFilterHit")
    if has_digi_branch:
        t_src.SetBranchStatus("Digi_MuFilterHits", 1)
        t_src.SetBranchAddress("Digi_MuFilterHits", orig_hits)

    # Create new calibrated branch
    cal_hits = ROOT.TClonesArray("MuFilterHit")
    t_new.Branch("Digi_MuFilterHits", cal_hits)

    total_orig = 0
    total_cal = 0

    for i in range(n_entries):
        t_src.GetEntry(i)
        cal_hits.Clear()

        by_bar = calc.group_mcpoints_by_bar(t_src.MuFilterPoint)

        if has_digi_branch:
            n_orig = orig_hits.GetEntriesFast()
            total_orig += n_orig
            hit_idx = 0
            for j in range(n_orig):
                h = orig_hits.At(j)
                did = h.GetDetectorID()
                clusters = by_bar.get(did, [])
                if not clusters:
                    continue
                pts = clusters[0]
                pes = calc.get_sipm_pes(pts, det_id=did)
                if any(not calc.is_small_sipm(did, ch) and pe >= th_pe for ch, pe in enumerate(pes)):
                    new_h = cal_hits.ConstructedAt(hit_idx)
                    ROOT.MuFilterHit(h).Copy(new_h)
                    hit_idx += 1
            total_cal += cal_hits.GetEntriesFast()
        else:
            # Create hits from scratch if file was undigitized
            hit_idx = 0
            for did, clusters in by_bar.items():
                pts = clusters[0]
                pes = calc.get_sipm_pes(pts, det_id=did)
                if any(not calc.is_small_sipm(did, ch) and pe >= th_pe for ch, pe in enumerate(pes)):
                    new_h = cal_hits.ConstructedAt(hit_idx)
                    new_h.SetDetectorID(did)
                    hit_idx += 1
            total_cal += cal_hits.GetEntriesFast()

        t_new.Fill()

    # Overwrite as cycle cbmsim;2 in ROOT file
    t_new.Write("cbmsim", ROOT.TObject.kOverwrite)
    f.Close()

    elapsed = time.time() - t0
    retention = (total_cal / total_orig * 100.0) if total_orig > 0 else 100.0
    print(f"[✓] {os.path.basename(file_path)}: {n_entries} events in {elapsed:.1f}s | Hits: {total_orig} -> {total_cal} ({retention:.1f}% retained)")
    return True


def _worker_wrapper(args_tuple):
    fpath, is_tb, yr, max_ev = args_tuple
    try:
        return redigitize_single_file(fpath, is_testbeam=is_tb, year=yr, max_events=max_ev)
    except Exception as exc:
        print(f"[x] Error processing {fpath}: {exc}")
        return False


def parse_partition_range(part_str: str) -> List[int]:
    """Parse comma/dash partition specs: '1-100', '1,2,3', '1-50,60-100'."""
    parts = []
    for chunk in part_str.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            parts.extend(range(int(start_s), int(end_s) + 1))
        elif chunk.isdigit():
            parts.append(int(chunk))
    return sorted(list(set(parts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-digitize MuFilter hits with calibrated MinPE response.")
    parser.add_argument("--file", type=str, help="Single file to re-digitize")
    parser.add_argument("--preset", type=str, choices=["newP8", "pmu2022"], help="Preset dataset to process")
    parser.add_argument("--pmu", action="store_true", help="Shortcut for --preset pmu2022")
    parser.add_argument("--partitions", type=str, default="1-1000", help="Partition range (e.g. 1-10, 1-1000)")
    parser.add_argument("--max-events", type=int, default=None, help="Max events per file (for testing)")
    parser.add_argument("-j", "--workers", type=int, default=8, help="Number of concurrent worker processes")
    args = parser.parse_args()

    target_files = []
    if args.file:
        target_files.append(args.file)
    elif args.pmu or args.preset == "pmu2022":
        pmu_f = "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP_2MuTrks_truth.root"
        if os.path.exists(pmu_f):
            target_files.append(pmu_f)
        else:
            print(f"[!] PMU target file not found: {pmu_f}")
    elif args.preset == "newP8":
        parts = parse_partition_range(args.partitions)
        for p in parts:
            fpath = f"/eos/user/i/idioniso/snd-numu-charm/data/newP8/{p}/sndLHC.Genie-TGeant4_dig.root"
            if os.path.exists(fpath):
                target_files.append(fpath)
            else:
                pass

    if not target_files:
        print("[!] No target files selected to re-digitize.")
        sys.exit(1)

    print(f"[*] Found {len(target_files)} file(s) to re-digitize using {args.workers} worker process(es).")
    t_start = time.time()

    if len(target_files) == 1 or args.workers <= 1:
        for fpath in target_files:
            redigitize_single_file(fpath, max_events=args.max_events)
    else:
        worker_args = [(fp, False, 2022, args.max_events) for fp in target_files]
        with mp.Pool(processes=min(args.workers, len(target_files))) as pool:
            results = pool.map(_worker_wrapper, worker_args)
        n_ok = sum(1 for r in results if r)
        print(f"[*] Completed {n_ok}/{len(target_files)} files in {time.time() - t_start:.1f}s.")
