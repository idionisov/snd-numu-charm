#!/usr/bin/env python3
"""
Add 'Edep_MuFilterHit' branch (std::vector<float> in MeV) to cbmsim in digitized ROOT files.
For every event, index i in 'Edep_MuFilterHit' matches index i in 'Digi_MuFilterHits'.
"""

import os
import sys
import glob
import time
import argparse
from multiprocessing import Pool, cpu_count
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True


def process_file(fpath: str, force: bool = False, verbose: bool = False) -> bool:
    """Add Edep_MuFilterHit branch to cbmsim in fpath."""
    if os.path.islink(fpath):
        if verbose:
            print(f"[SKIP SYMLINK] {fpath}")
        return True

    if not os.path.exists(fpath):
        print(f"[ERROR] File not found: {fpath}", file=sys.stderr)
        return False

    t0 = time.time()
    try:
        f = ROOT.TFile.Open(fpath, "UPDATE")
        if not f or f.IsZombie():
            print(f"[ERROR] Cannot open for UPDATE: {fpath}", file=sys.stderr)
            return False

        tree = f.Get("cbmsim")
        if not tree:
            print(f"[ERROR] No cbmsim tree in: {fpath}", file=sys.stderr)
            f.Close()
            return False

        # Check if branch already exists
        has_branch = bool(tree.GetBranch("Edep_MuFilterHit"))
        if has_branch and not force:
            if verbose:
                print(f"[ALREADY EXISTS] Edep_MuFilterHit in {os.path.basename(fpath)}")
            f.Close()
            return True

        # Optimize I/O by only reading needed branches
        tree.SetBranchStatus("*", 0)
        tree.SetBranchStatus("MuFilterPoint*", 1)
        tree.SetBranchStatus("Digi_MuFilterHits*", 1)

        # Create branch
        edeps = ROOT.std.vector("float")()
        branch = tree.Branch("Edep_MuFilterHit", edeps)

        n_entries = tree.GetEntries()
        for iev in range(n_entries):
            tree.GetEntry(iev)
            edeps.clear()

            # Map detectorID -> sum of energy deposits (in MeV)
            eloss_map = {}
            for pt in tree.MuFilterPoint:
                did = pt.GetDetectorID()
                eloss_map[did] = eloss_map.get(did, 0.0) + pt.GetEnergyLoss() * 1000.0  # MeV

            # Match 1-to-1 with Digi_MuFilterHits
            for hit in tree.Digi_MuFilterHits:
                did = hit.GetDetectorID()
                edeps.push_back(eloss_map.get(did, 0.0))

            branch.Fill()

        # Write tree header update
        tree.Write("", ROOT.TObject.kOverwrite)
        f.Close()

        elapsed = time.time() - t0
        print(f"[OK] {os.path.basename(fpath)}: {n_entries} events updated in {elapsed:.2f} s")
        return True

    except Exception as e:
        print(f"[EXCEPTION] Failed processing {fpath}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Add Edep_MuFilterHit branch to cbmsim")
    parser.add_argument("--file", "-f", type=str, default=None, help="Process single file")
    parser.add_argument("--dir", "-d", type=str, default=None, help="Process single directory")
    parser.add_argument("--base-dir", type=str, default="/eos/user/i/idioniso/snd-numu-charm/data/newP8", help="Base directory")
    parser.add_argument("--start-dir", type=int, default=1, help="Start directory index")
    parser.add_argument("--end-dir", type=int, default=10, help="End directory index (inclusive)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing branch")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    files_to_process = []

    if args.file:
        files_to_process = [args.file]
    elif args.dir:
        pattern = os.path.join(args.dir, "*_dig_*.root")
        files_to_process = [f for f in glob.glob(pattern) if not os.path.islink(f)]
    else:
        for idx in range(args.start_dir, args.end_dir + 1):
            d = os.path.join(args.base_dir, str(idx))
            if os.path.isdir(d):
                pattern = os.path.join(d, "*_dig_*.root")
                files_to_process.extend([f for f in glob.glob(pattern) if not os.path.islink(f)])

    print(f"Total target files found to process: {len(files_to_process)}")
    if len(files_to_process) == 0:
        return

    t_start = time.time()
    if args.workers > 1 and len(files_to_process) > 1:
        with Pool(args.workers) as pool:
            results = pool.starmap(process_file, [(f, args.force, args.verbose) for f in files_to_process])
        successes = sum(results)
    else:
        successes = sum(process_file(f, args.force, args.verbose) for f in files_to_process)

    total_time = time.time() - t_start
    print(f"\nDone! Processed {successes}/{len(files_to_process)} files successfully in {total_time:.2f} s")


if __name__ == "__main__":
    main()
