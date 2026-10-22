#!/usr/bin/env python3
"""
scripts/diagnostics/diagnose_mceb_mctruth.py
---------------------------------------------
Diagnostic tool investigating differences in MCTrack and MCPoint structures
between standard simulation files (sndLHC.Genie-TGeant4_dig.root) and
Monte Carlo Event Builder files (sndLHC.Genie-TGeant4_MCEB_dig.root).

Identifies why downstream MCTruth analysis (e.g. scripts/mctruth_neutrinos.py)
fails to observe charmed hadrons and signal dimuons in MCEB files.

Usage:
  python3 scripts/diagnostics/diagnose_mceb_mctruth.py -p 1
  python3 scripts/diagnostics/diagnose_mceb_mctruth.py -p 1 -n 30
  python3 scripts/diagnostics/diagnose_mceb_mctruth.py -p 1 --event 3524
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import ROOT
ROOT.gROOT.SetBatch(True)


CHARM_HADRONS = {
    411: "D+",
    421: "D0",
    431: "Ds+",
    4122: "Lambda_c+",
    4132: "Xi_c0",
    4232: "Xi_c+",
    4332: "Omega_c0",
}


def is_charmed_hadron(pdg: int) -> bool:
    """Check if PDG code corresponds to an open charm hadron."""
    return abs(pdg) in CHARM_HADRONS


def build_event_index(tree: ROOT.TTree, max_entries: int = -1) -> Dict[int, int]:
    """Map FairEventHeader EventNumber -> entry index in TTree."""
    index = {}
    n_entries = tree.GetEntries()
    limit = n_entries if max_entries <= 0 else min(n_entries, max_entries)

    for i in range(limit):
        tree.GetEntry(i)
        header = getattr(tree, "EventHeader", None)
        if header is not None:
            ev_num = int(header.GetEventNumber())
            index[ev_num] = i
        else:
            index[i] = i
    return index


def inspect_event_mctracks(
    trk_dig: Any,
    trk_mceb: Any,
) -> Dict[str, Any]:
    """Compare MCTrack arrays for a single matched event."""
    n_dig = len(trk_dig) if trk_dig else 0
    n_mceb = len(trk_mceb) if trk_mceb else 0

    # Non-zero tracks
    n_nonzero_dig = sum(1 for t in trk_dig if t.GetPdgCode() != 0) if trk_dig else 0
    n_nonzero_mceb = sum(1 for t in trk_mceb if t.GetPdgCode() != 0) if trk_mceb else 0

    # Charmed hadrons in DIG
    charm_dig = []
    if trk_dig:
        for idx, t in enumerate(trk_dig):
            pdg = t.GetPdgCode()
            if is_charmed_hadron(pdg) and t.GetMotherId() == 0:
                charm_dig.append((idx, pdg, t.GetP(), t.GetStartZ()))

    # Status of those charm tracks in MCEB
    charm_in_mceb = []
    for idx, pdg, p, z in charm_dig:
        if trk_mceb and idx < len(trk_mceb):
            t_m = trk_mceb[idx]
            charm_in_mceb.append((idx, pdg, t_m.GetPdgCode(), t_m.GetMotherId(), t_m.GetP()))
        else:
            charm_in_mceb.append((idx, pdg, None, None, None))

    # Primary muon (outgoing lepton)
    mu1_dig = None
    if trk_dig and len(trk_dig) > 1:
        for idx in range(1, len(trk_dig)):
            t = trk_dig[idx]
            if abs(t.GetPdgCode()) == 13 and t.GetMotherId() == 0:
                mu1_dig = (idx, t.GetPdgCode(), t.GetP())
                break

    mu1_mceb = None
    if trk_mceb and len(trk_mceb) > 1:
        for idx in range(1, len(trk_mceb)):
            t = trk_mceb[idx]
            if abs(t.GetPdgCode()) == 13 and t.GetMotherId() == 0:
                mu1_mceb = (idx, t.GetPdgCode(), t.GetP())
                break

    return {
        "n_dig": n_dig,
        "n_mceb": n_mceb,
        "n_nonzero_dig": n_nonzero_dig,
        "n_nonzero_mceb": n_nonzero_mceb,
        "charm_dig": charm_dig,
        "charm_in_mceb": charm_in_mceb,
        "mu1_dig": mu1_dig,
        "mu1_mceb": mu1_mceb,
    }


def inspect_event_mcpoints(
    t_dig: ROOT.TTree,
    t_mceb: ROOT.TTree,
) -> Dict[str, Any]:
    """Compare ScifiPoint and MuFilterPoint arrays."""
    scifi_dig = getattr(t_dig, "ScifiPoint", None)
    mufi_dig = getattr(t_dig, "MuFilterPoint", None)
    scifi_mceb = getattr(t_mceb, "ScifiPoint", None)
    mufi_mceb = getattr(t_mceb, "MuFilterPoint", None)

    n_scifi_dig = len(scifi_dig) if scifi_dig else 0
    n_mufi_dig = len(mufi_dig) if mufi_dig else 0
    n_scifi_mceb = len(scifi_mceb) if scifi_mceb else 0
    n_mufi_mceb = len(mufi_mceb) if mufi_mceb else 0

    times_scifi_dig = [p.GetTime() for p in scifi_dig] if scifi_dig else []
    times_scifi_mceb = [p.GetTime() for p in scifi_mceb] if scifi_mceb else []
    times_mufi_dig = [p.GetTime() for p in mufi_dig] if mufi_dig else []
    times_mufi_mceb = [p.GetTime() for p in mufi_mceb] if mufi_mceb else []

    return {
        "n_scifi_dig": n_scifi_dig,
        "n_scifi_mceb": n_scifi_mceb,
        "n_mufi_dig": n_mufi_dig,
        "n_mufi_mceb": n_mufi_mceb,
        "t_scifi_dig_range": (min(times_scifi_dig), max(times_scifi_dig)) if times_scifi_dig else (0, 0),
        "t_scifi_mceb_range": (min(times_scifi_mceb), max(times_scifi_mceb)) if times_scifi_mceb else (0, 0),
        "t_mufi_dig_range": (min(times_mufi_dig), max(times_mufi_dig)) if times_mufi_dig else (0, 0),
        "t_mufi_mceb_range": (min(times_mufi_mceb), max(times_mufi_mceb)) if times_mufi_mceb else (0, 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose MCTrack and MCPoint differences between standard DIG and MCEB files."
    )
    parser.add_argument("-p", "--partition", type=int, default=1, help="Partition number (default: 1).")
    parser.add_argument("-n", "--max-events", type=int, default=50, help="Max events to scan (default: 50).")
    parser.add_argument("-e", "--event", type=int, default=None, help="Specific EventNumber to inspect in detail.")
    parser.add_argument(
        "--dig-file",
        default=None,
        help="Path to standard sndLHC.Genie-TGeant4_dig.root",
    )
    parser.add_argument(
        "--mceb-file",
        default=None,
        help="Path to sndLHC.Genie-TGeant4_MCEB_dig.root",
    )

    args = parser.parse_args()

    base_dir = f"/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/{args.partition}"
    dig_path = args.dig_file or os.path.join(base_dir, "sndLHC.Genie-TGeant4_dig.root")
    mceb_path = args.mceb_file or os.path.join(base_dir, "sndLHC.Genie-TGeant4_MCEB_dig.root")

    print("=" * 80)
    print(" SND@LHC MCEB vs DIG TRUTH & DETECTOR SIMULATION DIAGNOSTIC")
    print("=" * 80)
    print(f" Partition   : {args.partition}")
    print(f" DIG File    : {dig_path}")
    print(f" MCEB File   : {mceb_path}")
    print("=" * 80)

    if not os.path.isfile(dig_path):
        print(f"ERROR: DIG file not found: {dig_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(mceb_path):
        print(f"ERROR: MCEB file not found: {mceb_path}", file=sys.stderr)
        sys.exit(1)

    f_dig = ROOT.TFile.Open(dig_path, "READ")
    t_dig = f_dig.Get("cbmsim")
    f_mceb = ROOT.TFile.Open(mceb_path, "READ")
    t_mceb = f_mceb.Get("cbmsim")

    if not t_dig or not t_mceb:
        print("ERROR: Could not retrieve cbmsim tree from one or both files.", file=sys.stderr)
        sys.exit(2)

    n_dig_tot = t_dig.GetEntries()
    n_mceb_tot = t_mceb.GetEntries()
    print(f"\n[1] Event Parity:")
    print(f"    Standard DIG Total Entries : {n_dig_tot}")
    print(f"    MCEB Total Entries         : {n_mceb_tot}")
    print(f"    Delta (Noise filtered)     : {n_dig_tot - n_mceb_tot} event(s)")

    print("\n[2] Indexing Events by EventHeader.GetEventNumber()...")
    dig_index = build_event_index(t_dig, args.max_events)
    mceb_index = build_event_index(t_mceb, args.max_events)

    common_evs = sorted(list(set(dig_index.keys()) & set(mceb_index.keys())))
    missing_in_mceb = sorted(list(set(dig_index.keys()) - set(mceb_index.keys())))
    print(f"    Scanned Events             : {len(dig_index)} DIG, {len(mceb_index)} MCEB")
    print(f"    Matched Event Numbers      : {len(common_evs)}")
    if missing_in_mceb:
        print(f"    Filtered/Dropped by MCEB   : {len(missing_in_mceb)} events (e.g. {missing_in_mceb[:5]})")

    # Aggregate statistics
    tot_charm_dig = 0
    tot_charm_preserved_mceb = 0
    tot_tracks_dig = 0
    tot_tracks_nonzero_mceb = 0
    tot_tracks_zeroed_mceb = 0

    event_charm_reports = []

    for ev_num in common_evs:
        idx_d = dig_index[ev_num]
        idx_m = mceb_index[ev_num]
        t_dig.GetEntry(idx_d)
        t_mceb.GetEntry(idx_m)

        res_trk = inspect_event_mctracks(t_dig.MCTrack, t_mceb.MCTrack)
        tot_tracks_dig += res_trk["n_nonzero_dig"]
        tot_tracks_nonzero_mceb += res_trk["n_nonzero_mceb"]
        tot_tracks_zeroed_mceb += (res_trk["n_mceb"] - res_trk["n_nonzero_mceb"])

        if res_trk["charm_dig"]:
            tot_charm_dig += len(res_trk["charm_dig"])
            n_pres = sum(1 for c in res_trk["charm_in_mceb"] if c[2] != 0 and c[2] is not None)
            tot_charm_preserved_mceb += n_pres
            event_charm_reports.append((ev_num, idx_d, idx_m, res_trk))

    print("\n[3] MCTrack Zeroing & Pruning Analysis across matched events:")
    print(f"    Total Valid MCTracks in DIG    : {tot_tracks_dig}")
    print(f"    Total Non-Zero MCTracks in MCEB: {tot_tracks_nonzero_mceb} ({tot_tracks_nonzero_mceb/tot_tracks_dig*100:.1f}%)" if tot_tracks_dig > 0 else "")
    print(f"    Total Zeroed (PDG=0) in MCEB   : {tot_tracks_zeroed_mceb} dummy tracks")
    print(f"    Charmed Hadrons in DIG         : {tot_charm_dig}")
    print(f"    Charmed Hadrons Preserved MCEB : {tot_charm_preserved_mceb} (LOST: {tot_charm_dig - tot_charm_preserved_mceb})")

    print("\n[4] Detailed Charm Hadron Inspection (First 5 charm events):")
    print(f"    {'EventNum':<9} | {'TrackID':<7} | {'Species':<10} | {'P (GeV)':<8} | {'DIG PDG':<8} | {'MCEB PDG':<8} | {'MCEB MotherID':<13} | Status")
    print("    " + "-" * 75)
    for ev_num, idx_d, idx_m, rep in event_charm_reports[:5]:
        for (trk_id, pdg, p, z), (_, _, m_pdg, m_moth, _) in zip(rep["charm_dig"], rep["charm_in_mceb"]):
            spec_name = CHARM_HADRONS.get(abs(pdg), str(pdg))
            status = "PRESERVED" if m_pdg == pdg else "ZEROED_OUT (PDG=0)"
            print(f"    {ev_num:<9} | {trk_id:<7} | {spec_name:<10} | {p:<8.2f} | {pdg:<8} | {str(m_pdg):<8} | {str(m_moth):<13} | {status}")

    # Inspect specific event or the first charm event in depth
    target_ev = args.event
    if target_ev is None and event_charm_reports:
        target_ev = event_charm_reports[0][0]

    if target_ev is not None and target_ev in dig_index and target_ev in mceb_index:
        idx_d = dig_index[target_ev]
        idx_m = mceb_index[target_ev]
        t_dig.GetEntry(idx_d)
        t_mceb.GetEntry(idx_m)

        pts_rep = inspect_event_mcpoints(t_dig, t_mceb)
        print(f"\n[5] Deep Dive into Event {target_ev}:")
        print(f"    Entry in DIG  : {idx_d}")
        print(f"    Entry in MCEB : {idx_m}")
        print(f"    ScifiPoint Count    : DIG={pts_rep['n_scifi_dig']} vs MCEB={pts_rep['n_scifi_mceb']}")
        print(f"    ScifiPoint Time (ns): DIG=[{pts_rep['t_scifi_dig_range'][0]:.2f}, {pts_rep['t_scifi_dig_range'][1]:.2f}] vs MCEB=[{pts_rep['t_scifi_mceb_range'][0]:.2f}, {pts_rep['t_scifi_mceb_range'][1]:.2f}]")
        print(f"    MuFilterPoint Count : DIG={pts_rep['n_mufi_dig']} vs MCEB={pts_rep['n_mufi_mceb']}")
        print(f"    MuFilter Time (ns)  : DIG=[{pts_rep['t_mufi_dig_range'][0]:.2f}, {pts_rep['t_mufi_dig_range'][1]:.2f}] vs MCEB=[{pts_rep['t_mufi_mceb_range'][0]:.2f}, {pts_rep['t_mufi_mceb_range'][1]:.2f}]")

        print(f"\n    First 10 MCTracks in Event {target_ev}:")
        print(f"    {'ID':<4} | {'DIG PDG':<8} | {'DIG Moth':<8} | {'DIG P':<7} | {'MCEB PDG':<8} | {'MCEB Moth':<9} | {'MCEB P':<7} | Description")
        print("    " + "-" * 75)
        for i in range(min(12, len(t_dig.MCTrack))):
            td = t_dig.MCTrack[i]
            tm = t_mceb.MCTrack[i] if i < len(t_mceb.MCTrack) else None
            m_pdg = tm.GetPdgCode() if tm else "-"
            m_moth = tm.GetMotherId() if tm else "-"
            m_p = f"{tm.GetP():.2f}" if tm else "-"
            desc = ""
            if td.GetMotherId() == -1:
                desc = "Incoming Neutrino"
            elif td.GetMotherId() == 0:
                if abs(td.GetPdgCode()) == 13:
                    desc = "Primary Outgoing Muon (mu_1)"
                elif is_charmed_hadron(td.GetPdgCode()):
                    desc = f"Charmed Hadron ({CHARM_HADRONS.get(abs(td.GetPdgCode()), '')})"
                else:
                    desc = "Primary Hadron/Fragment"
            print(f"    {i:<4} | {td.GetPdgCode():<8} | {td.GetMotherId():<8} | {td.GetP():<7.2f} | {str(m_pdg):<8} | {str(m_moth):<9} | {m_p:<7} | {desc}")

    print("\n" + "=" * 80)
    print(" ROOT CAUSE SUMMARY:")
    print(" 1. In sndsw/sndFairTasks/MCEventBuilder.cxx (lines 325-385):")
    print("    - fOutMCTrackArray is sized via ExpandCreate(mcTrackClones.size()), initializing all tracks with default PDG=0, MotherId=-1.")
    print("    - Only tracks that generated a ScifiPoint or MuFilterPoint in that 25-ns time window are filled via:")
    print("        ShipMCTrack* newTrack = new ((*fOutMCTrackArray)[tid]) ShipMCTrack(*mcTrackClones[tid]);")
    print("    - Charmed hadrons (D0, D+, Ds+, Lambda_c+) decay within c*tau ~ 60-300 um inside the target volume.")
    print("      They NEVER traverse active detector bars/fibres, so they NEVER produce a ScifiPoint or MuFilterPoint.")
    print("    - Consequently, MCEventBuilder leaves their MCTrack entries permanently zeroed out (PDG=0).")
    print(" 2. Impact on mctruth_neutrinos.py:")
    print("    - NeutrinoTruthProcessor::findPrimaryCharmedHadrons requires:")
    print("        trk->GetMotherId() == 0 && isCharmedHadron(trk->GetPdgCode())")
    print("    - Because the charm track PDG is 0, isCharmedHadron(0) returns false for 100% of events.")
    print("    - info.hasCharm is set to false, and charm direct/indirect dimuon categorization evaluates to 0.")
    print("=" * 80)

    f_dig.Close()
    f_mceb.Close()


if __name__ == "__main__":
    main()
