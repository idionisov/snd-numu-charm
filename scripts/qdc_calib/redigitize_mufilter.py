#!/usr/bin/env python3
"""
scripts/qdc_calib/redigitize_mufilter.py
---------------------------------------
Re-digitizes MuFilter hits in Monte Carlo simulation files (GENIE newP8 partitions, PMU2022,
and 2023 Testbeam multi-energy runs) using the calibrated MinPE physics response
(Birks quenching, edge effect, arrival dispersion, and channel-specific SiPM reduction factor k).

For personal simulation samples, writes a new cycle `cbmsim;2` into the ROOT file.
For testbeam runs, streams the cloned structure and calibrated hits into:
  /eos/user/i/idioniso/mufi-digi/testbeam2023/run<run_num:06d>/*.root

Usage:
  # Testbeam all runs:
  python3 scripts/qdc_calib/redigitize_mufilter.py --tb --outdir /eos/user/i/idioniso/mufi-digi/testbeam2023 -j 8

  # Testbeam single run:
  python3 scripts/qdc_calib/redigitize_mufilter.py --tb-run 100671 --outdir /eos/user/i/idioniso/mufi-digi/testbeam2023

  # GENIE partitions:
  python3 scripts/qdc_calib/redigitize_mufilter.py --preset newP8 -j 16

  # PMU sample:
  python3 scripts/qdc_calib/redigitize_mufilter.py --pmu -j 8
"""

from __future__ import annotations

import os
import sys
import glob
import time
import shutil
import argparse
from typing import Dict, List, Optional, Tuple
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

TB_MC_BASE = "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut"

TB_RUN_MAP: Dict[int, Dict[str, Any]] = {
    100677: {
        "energy": 100,
        "files": [f"{TB_MC_BASE}/100GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root"],
        "geofile": f"{TB_MC_BASE}/100GeV_211/geofile_full.PG_211-TGeant4.root",
    },
    100673: {
        "energy": 140,
        "files": [f"{TB_MC_BASE}/140GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root"],
        "geofile": f"{TB_MC_BASE}/140GeV_211/geofile_full.PG_211-TGeant4.root",
    },
    100671: {
        "energy": 180,
        "files": [f"{TB_MC_BASE}/180GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root"],
        "geofile": f"{TB_MC_BASE}/180GeV_211/geofile_full.PG_211-TGeant4.root",
    },
    100648: {
        "energy": 240,
        "files": sorted(glob.glob(f"{TB_MC_BASE}/240GeV_-211/*/sndLHC.PG_-211-TGeant4_MCEB_dig.root")),
        "geofile": f"{TB_MC_BASE}/240GeV_-211/0/geofile_full.PG_-211-TGeant4.root",
    },
    100645: {
        "energy": 300,
        "files": sorted(glob.glob(f"{TB_MC_BASE}/300GeV_-211/*/sndLHC.PG_-211-TGeant4_MCEB_dig.root")),
        "geofile": f"{TB_MC_BASE}/300GeV_-211/0/geofile_full.PG_-211-TGeant4.root",
    },
    100242: {
        "energy": 100,
        "files": [f"{TB_MC_BASE}/100GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root"],
        "geofile": f"{TB_MC_BASE}/100GeV_211/geofile_full.PG_211-TGeant4.root",
    },
    100243: {
        "energy": 180,
        "files": [f"{TB_MC_BASE}/180GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root"],
        "geofile": f"{TB_MC_BASE}/180GeV_211/geofile_full.PG_211-TGeant4.root",
    },
    100265: {
        "energy": 240,
        "files": sorted(glob.glob(f"{TB_MC_BASE}/240GeV_-211/*/sndLHC.PG_-211-TGeant4_MCEB_dig.root")),
        "geofile": f"{TB_MC_BASE}/240GeV_-211/0/geofile_full.PG_-211-TGeant4.root",
    },
    100266: {
        "energy": 300,
        "files": sorted(glob.glob(f"{TB_MC_BASE}/300GeV_-211/*/sndLHC.PG_-211-TGeant4_MCEB_dig.root")),
        "geofile": f"{TB_MC_BASE}/300GeV_-211/0/geofile_full.PG_-211-TGeant4.root",
    },
}


def redigitize_single_file(
    file_path: str,
    output_path: Optional[str] = None,
    is_testbeam: bool = False,
    year: int = 2022,
    max_events: Optional[int] = None,
) -> bool:
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return False

    t0 = time.time()
    calc = mpe.MinPE(year=year, is_testbeam=is_testbeam, include_edge_effect=True, include_dispersion=True)
    th_pe = calc.get_sipm_threshold_pe()

    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)

        f_src = ROOT.TFile.Open(file_path, "READ")
        if not f_src or f_src.IsZombie():
            print(f"[!] Cannot open {file_path} in READ mode.")
            return False

        t_src = f_src.Get("cbmsim;2") or f_src.Get("cbmsim;1") or f_src.Get("cbmsim")
        if not t_src:
            print(f"[!] No cbmsim tree in {file_path}")
            f_src.Close()
            return False

        n_entries = t_src.GetEntries()
        if max_events is not None and max_events > 0:
            n_entries = min(n_entries, max_events)

        f_dst = ROOT.TFile.Open(output_path, "RECREATE")
        f_dst.cd()

        for k in f_src.GetListOfKeys():
            name = k.GetName()
            if name != "cbmsim":
                obj = k.ReadObj()
                obj.Write(name, ROOT.TObject.kSingleKey)

        has_digi_branch = (t_src.GetBranch("Digi_MuFilterHits") is not None)
        has_raw_branch = (t_src.GetBranch("Digi_MuFilterHits_Raw") is not None)

        t_new = t_src.CloneTree(0)

        # Retain original hits in Digi_MuFilterHits_Raw
        if has_digi_branch and not has_raw_branch:
            orig_b = t_new.GetBranch("Digi_MuFilterHits")
            if orig_b:
                orig_b.SetName("Digi_MuFilterHits_Raw")

        orig_hits = ROOT.TClonesArray("MuFilterHit")
        if has_digi_branch:
            t_src.SetBranchAddress("Digi_MuFilterHits", orig_hits)
        elif has_raw_branch:
            t_src.SetBranchAddress("Digi_MuFilterHits_Raw", orig_hits)

        cal_hits = ROOT.TClonesArray("MuFilterHit")
        t_new.Branch("Digi_MuFilterHits", cal_hits)

        total_orig = 0
        total_cal = 0
        has_any_orig = has_digi_branch or has_raw_branch

        for i in range(n_entries):
            t_src.GetEntry(i)
            cal_hits.Clear()

            by_bar = calc.group_mcpoints_by_bar(t_src.MuFilterPoint)

            if has_any_orig:
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

        t_new.Write("cbmsim", ROOT.TObject.kOverwrite)
        f_dst.Close()
        f_src.Close()

        elapsed = time.time() - t0
        retention = (total_cal / total_orig * 100.0) if total_orig > 0 else 100.0
        print(f"[✓] -> {output_path}: {n_entries} events in {elapsed:.1f}s | Hits: {total_orig} -> {total_cal} ({retention:.1f}% retained)")
        return True

    else:
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

        has_digi_branch = (t_src.GetBranch("Digi_MuFilterHits") is not None)
        has_raw_branch = (t_src.GetBranch("Digi_MuFilterHits_Raw") is not None)

        t_new = t_src.CloneTree(0)

        # Retain original hits in Digi_MuFilterHits_Raw
        if has_digi_branch and not has_raw_branch:
            orig_b = t_new.GetBranch("Digi_MuFilterHits")
            if orig_b:
                orig_b.SetName("Digi_MuFilterHits_Raw")

        orig_hits = ROOT.TClonesArray("MuFilterHit")
        if has_digi_branch:
            t_src.SetBranchAddress("Digi_MuFilterHits", orig_hits)
        elif has_raw_branch:
            t_src.SetBranchAddress("Digi_MuFilterHits_Raw", orig_hits)

        cal_hits = ROOT.TClonesArray("MuFilterHit")
        t_new.Branch("Digi_MuFilterHits", cal_hits)

        total_orig = 0
        total_cal = 0
        has_any_orig = has_digi_branch or has_raw_branch

        for i in range(n_entries):
            t_src.GetEntry(i)
            cal_hits.Clear()

            by_bar = calc.group_mcpoints_by_bar(t_src.MuFilterPoint)

            if has_any_orig:
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

        t_new.Write("cbmsim", ROOT.TObject.kOverwrite)
        f.Close()

        elapsed = time.time() - t0
        retention = (total_cal / total_orig * 100.0) if total_orig > 0 else 100.0
        print(f"[✓] {os.path.basename(file_path)}: {n_entries} events in {elapsed:.1f}s | Hits: {total_orig} -> {total_cal} ({retention:.1f}% retained)")
        return True


def _worker_wrapper(args_tuple):
    fpath, out_path, is_tb, yr, max_ev = args_tuple
    try:
        return redigitize_single_file(fpath, output_path=out_path, is_testbeam=is_tb, year=yr, max_events=max_ev)
    except Exception as exc:
        print(f"[x] Error processing {fpath}: {exc}")
        return False


def parse_partition_range(part_str: str) -> List[int]:
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
    parser.add_argument("--output", type=str, default=None, help="Output destination file path (optional)")
    parser.add_argument("--preset", type=str, choices=["newP8", "pmu2022"], help="Preset dataset to process")
    parser.add_argument("--pmu", action="store_true", help="Shortcut for --preset pmu2022")
    parser.add_argument("--partitions", type=str, default="1-1000", help="Partition range (e.g. 1-10, 1-1000)")
    parser.add_argument("--tb", action="store_true", help="Process all testbeam runs (August 2023: 100677, 100673, 100671, 100648, 100645)")
    parser.add_argument("--tb-run", type=int, default=None, help="Specific testbeam run number to re-digitize (e.g. 100671)")
    parser.add_argument("--outdir", type=str, default="/eos/user/i/idioniso/mufi-digi/testbeam2023", help="Output directory root for testbeam runs")
    parser.add_argument("--max-events", type=int, default=None, help="Max events per file (for testing)")
    parser.add_argument("-j", "--workers", type=int, default=8, help="Number of concurrent worker processes")
    args = parser.parse_args()

    tasks: List[Tuple[str, Optional[str], bool, int, Optional[int]]] = []

    if args.tb or args.tb_run is not None:
        selected_runs = [args.tb_run] if args.tb_run is not None else [100677, 100673, 100671, 100648, 100645]
        for r in selected_runs:
            if r not in TB_RUN_MAP:
                print(f"[!] Unknown testbeam run: {r}")
                continue
            r_info = TB_RUN_MAP[r]
            run_dir = os.path.join(args.outdir, f"run{r:06d}")
            os.makedirs(run_dir, exist_ok=True)

            geofile = r_info.get("geofile")
            if geofile and os.path.exists(geofile):
                dst_geo = os.path.join(run_dir, os.path.basename(geofile))
                if not os.path.exists(dst_geo):
                    try:
                        shutil.copy2(geofile, dst_geo)
                        print(f"[*] Copied geofile -> {dst_geo}")
                    except Exception as e:
                        print(f"[!] Warning copying geofile: {e}")

            for f_in in r_info["files"]:
                p_dir = os.path.basename(os.path.dirname(f_in))
                b_name = os.path.basename(f_in)
                dst_name = b_name.replace(".root", f"_part{p_dir}.root") if p_dir.isdigit() else b_name
                f_out = os.path.join(run_dir, dst_name)
                tasks.append((f_in, f_out, True, 2023, args.max_events))

    elif args.file:
        tasks.append((args.file, args.output, False, 2022, args.max_events))
    elif args.pmu or args.preset == "pmu2022":
        pmu_f = "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP_2MuTrks_truth.root"
        if os.path.exists(pmu_f):
            tasks.append((pmu_f, None, False, 2022, args.max_events))
        else:
            print(f"[!] PMU target file not found: {pmu_f}")
    elif args.preset == "newP8":
        parts = parse_partition_range(args.partitions)
        for p in parts:
            fpath = f"/eos/user/i/idioniso/snd-numu-charm/data/newP8/{p}/sndLHC.Genie-TGeant4_dig.root"
            if os.path.exists(fpath):
                tasks.append((fpath, None, False, 2022, args.max_events))

    if not tasks:
        print("[!] No target tasks selected to process.")
        sys.exit(1)

    print(f"[*] Preparing {len(tasks)} file task(s) using {args.workers} worker process(es)...")
    t_start = time.time()

    if len(tasks) == 1 or args.workers <= 1:
        n_ok = 0
        for task in tasks:
            if _worker_wrapper(task):
                n_ok += 1
    else:
        with mp.Pool(processes=min(args.workers, len(tasks))) as pool:
            results = pool.map(_worker_wrapper, tasks)
        n_ok = sum(1 for r in results if r)

    print(f"[*] Finished {n_ok}/{len(tasks)} files in {time.time() - t_start:.1f}s.")
