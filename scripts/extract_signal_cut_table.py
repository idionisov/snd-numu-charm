#!/usr/bin/env python3
"""
scripts/extract_signal_cut_table.py
-----------------------------------
Extracts a comprehensive tabular summary (CSV) of all signal events across partitions,
recording for each event whether each cut stage passed (1) or failed (0).

Evaluates the minimalistic cutset:
1. LHC IP1 (1 for MC)
2. Stable Beams (1 for MC)
3. Event Delta t (1 for MC)
4. SciFi-DS Timing (Latest DS hit time > earliest SciFi hit time)
5. No Veto Hits (0 hits in Veto system)
6. DS Dimuon (>=2 hits/plane in 3 planes in one proj, >=1 hit/plane in 3 planes in other)
7. Two DS Tracks (exactly 2 reconstructed DS Hough tracks)

Usage:
  python3 scripts/extract_signal_cut_table.py \
    --signal-input "/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_dig_2MuTrks_truth_charmDimuonDirect.root" \
    --output "/eos/user/i/idioniso/snd-numu-charm/signal/2022_down_nu14_signal_cut_table.csv" \
    -p "1-1000" \
    -w 12
"""

from __future__ import annotations

import os
import sys
import argparse
import time
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_single_partition_signal(
    partition: int,
    signal_input_pattern: str,
    indirect_input_pattern: Optional[str] = None,
) -> List[Dict[str, Any]]:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    from snd import (
        load_trident_libraries,
        SciFiDSTimeCut,
        VetoCut,
        DSDimuonActivityCut,
        DSHoughTrackCut,
    )
    load_trident_libraries()

    cut_timing = SciFiDSTimeCut(0.0, True, False)
    cut_veto = VetoCut(0, 0, False)
    cut_dimuon = DSDimuonActivityCut(3, 2, 3, 1, False, False)
    cut_tracks = DSHoughTrackCut(2, 2, 13, False)

    records: List[Dict[str, Any]] = []

    streams = [("direct", signal_input_pattern)]
    if indirect_input_pattern:
        streams.append(("indirect", indirect_input_pattern))

    p_str = str(partition)
    for stype, pat in streams:
        if not pat:
            continue
        c_p = pat.count("%s")
        filepath = pat % tuple(p_str for _ in range(c_p)) if c_p > 0 else pat

        if not os.path.isfile(filepath):
            continue

        f_in = ROOT.TFile.Open(filepath, "READ")
        if not f_in or f_in.IsZombie():
            continue

        t_cbm = f_in.Get("cbmsim") or f_in.Get("rawConv")
        t_truth = f_in.Get("mctruth")
        n_entries = t_cbm.GetEntries() if t_cbm else 0

        if not t_cbm or n_entries == 0:
            f_in.Close()
            continue

        t_cbm.SetBranchStatus("*", 0)
        t_cbm.SetBranchStatus("Digi_ScifiHits*", 1)
        t_cbm.SetBranchStatus("Digi_MuFilterHits*", 1)
        t_cbm.SetBranchStatus("Reco_MuonTracks*", 1)
        t_cbm.SetBranchStatus("EventHeader*", 1)

        has_truth = bool(t_truth and t_truth.GetEntries() == n_entries)

        for iev in range(n_entries):
            t_cbm.GetEntry(iev)
            h = getattr(t_cbm, "EventHeader.", None) or getattr(t_cbm, "EventHeader", None)
            ev_num = h.GetEventNumber() if h else iev

            # Event weights
            mc_w = 1.0
            raw_w = 1.0
            nu_e = -1.0
            charm_pdg = 0
            if has_truth:
                t_truth.GetEntry(iev)
                mc_w = float(getattr(t_truth, "mc_weight", 1.0))
                raw_w = float(getattr(t_truth, "raw_weight", 1.0))
                nu_e = float(getattr(t_truth, "nu_e", -1.0))
                charm_pdg = int(getattr(t_truth, "charm_pdg", 0))

            # Diagnostic hit counts
            sf_hits = getattr(t_cbm, "Digi_ScifiHits", None)
            mf_hits = getattr(t_cbm, "Digi_MuFilterHits", None)
            n_veto = cut_veto.countVetoHits(mf_hits, sf_hits) if mf_hits else 0
            n_ds_tracks = cut_tracks.countTracks(t_cbm.Reco_MuonTracks) if hasattr(t_cbm, "Reco_MuonTracks") else 0

            # Independent cut evaluations (1 = pass, 0 = fail)
            p_ip1 = 1
            p_sb = 1
            p_dt = 1
            p_timing = int(bool(cut_timing(t_cbm.Digi_ScifiHits, t_cbm.Digi_MuFilterHits)))
            p_veto = int(bool(cut_veto(t_cbm.Digi_MuFilterHits, t_cbm.Digi_ScifiHits)))
            p_dimuon = int(bool(cut_dimuon(t_cbm.Digi_MuFilterHits)))
            p_tracks = int(bool(cut_tracks(t_cbm.Reco_MuonTracks)))

            p_all = int(p_ip1 and p_sb and p_dt and p_timing and p_veto and p_dimuon and p_tracks)

            records.append({
                "partition": partition,
                "event_number": ev_num,
                "entry": iev,
                "signal_type": stype,
                "mc_weight": mc_w,
                "raw_weight": raw_w,
                "nu_e": nu_e,
                "charm_pdg": charm_pdg,
                "pass_lhc_ip1": p_ip1,
                "pass_stable_beams": p_sb,
                "pass_event_deltat": p_dt,
                "pass_scifi_ds_timing": p_timing,
                "pass_no_veto_hits": p_veto,
                "pass_ds_dimuon": p_dimuon,
                "pass_two_ds_tracks": p_tracks,
                "pass_all_cuts": p_all,
                "n_veto_hits": n_veto,
                "n_ds_tracks": n_ds_tracks,
            })

        f_in.Close()

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Extract tabular summary of signal event cut status across partitions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--signal-input",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_dig_2MuTrks_truth_charmDimuonDirect.root",
        help="Input path pattern for direct signal MC ROOT files (supports '%s')",
    )
    parser.add_argument(
        "--indirect-input",
        type=str,
        default=None,
        help="Optional input path pattern for indirect signal MC ROOT files (supports '%s')",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/signal/2022_down_nu14_signal_cut_table.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "-p", "--partitions",
        type=str,
        default="1-1000",
        help="Partition range ('1-1000') or comma-separated list",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=12,
        help="Number of concurrent worker processes",
    )

    args = parser.parse_args()

    from scripts.extract_simulation_truth import parse_partitions
    partitions = parse_partitions(args.partitions)
    if partitions is None:
        partitions = list(range(1, 1001))

    print("=" * 80)
    print(" SIGNAL CUT TABLE EXTRACTION")
    print("=" * 80)
    print(f" Signal Input Pattern:   {args.signal_input}")
    if args.indirect_input:
        print(f" Indirect Input Pattern: {args.indirect_input}")
    print(f" Output CSV:             {args.output}")
    print(f" Partitions to Scan:     {len(partitions)} ({min(partitions)}..{max(partitions)})")
    print(f" Parallel Workers:       {args.workers}")
    print("=" * 80)

    t0 = time.time()
    all_records: List[Dict[str, Any]] = []

    tasks = [(p, args.signal_input, args.indirect_input) for p in partitions]

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_part = {
                executor.submit(process_single_partition_signal, *t): t[0]
                for t in tasks
            }
            completed = 0
            for fut in as_completed(future_to_part):
                completed += 1
                part = future_to_part[fut]
                try:
                    recs = fut.result()
                    all_records.extend(recs)
                except Exception as err:
                    print(f"  [Warning] Partition {part} failed: {err}")
                if completed % 100 == 0 or completed == len(tasks):
                    rate = completed / max(time.time() - t0, 0.1)
                    eta = (len(tasks) - completed) / max(rate, 0.001)
                    print(f"[{completed:4d}/{len(tasks)}] Scanned partitions | Found {len(all_records):4d} signal events (ETA: {int(eta)}s)")
    else:
        for t in tasks:
            recs = process_single_partition_signal(*t)
            all_records.extend(recs)

    all_records.sort(key=lambda r: (r["partition"], r["entry"]))

    # Write output CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fieldnames = [
        "partition",
        "event_number",
        "entry",
        "signal_type",
        "mc_weight",
        "raw_weight",
        "nu_e",
        "charm_pdg",
        "pass_lhc_ip1",
        "pass_stable_beams",
        "pass_event_deltat",
        "pass_scifi_ds_timing",
        "pass_no_veto_hits",
        "pass_ds_dimuon",
        "pass_two_ds_tracks",
        "pass_all_cuts",
        "n_veto_hits",
        "n_ds_tracks",
    ]

    import csv
    with open(args.output, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    elapsed = time.time() - t0
    n_tot = len(all_records)
    n_pass_timing = sum(r["pass_scifi_ds_timing"] for r in all_records)
    n_pass_veto = sum(r["pass_no_veto_hits"] for r in all_records)
    n_pass_dimuon = sum(r["pass_ds_dimuon"] for r in all_records)
    n_pass_tracks = sum(r["pass_two_ds_tracks"] for r in all_records)
    n_pass_all = sum(r["pass_all_cuts"] for r in all_records)

    print("\n" + "=" * 80)
    print(" EXTRACTION SUMMARY")
    print("=" * 80)
    print(f" Total Signal Events:        {n_tot}")
    print(f" Pass SciFi-DS Timing:       {n_pass_timing}/{n_tot} ({(n_pass_timing/n_tot*100) if n_tot else 0:.1f}%)")
    print(f" Pass No Veto Hits:          {n_pass_veto}/{n_tot} ({(n_pass_veto/n_tot*100) if n_tot else 0:.1f}%)")
    print(f" Pass DS Dimuon Activity:    {n_pass_dimuon}/{n_tot} ({(n_pass_dimuon/n_tot*100) if n_tot else 0:.1f}%)")
    print(f" Pass Two DS Tracks:         {n_pass_tracks}/{n_tot} ({(n_pass_tracks/n_tot*100) if n_tot else 0:.1f}%)")
    print(f" Pass All Cuts:              {n_pass_all}/{n_tot} ({(n_pass_all/n_tot*100) if n_tot else 0:.1f}%)")
    print(f" Output Written to:          {args.output}")
    print(f" Elapsed Time:               {elapsed:.1f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
