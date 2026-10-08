#!/usr/bin/env python3
"""
scripts/diagnostics/mu2_diagnostic.py
-------------------------------------
Diagnostic tool to inspect the parent particles of secondary muons (mu2) originating
from charmed hadron decay chains.

Uses snd.DataManager to load the dataset, then runs a transparent Python loop over events
and MCTracks with explicit manual checks.

For each event:
  1. Identifies the primary prompt muon (mu1) from the neutrino interaction vertex (motherId == 0).
  2. Searches for secondary muons (mu2) whose mother ancestry traces back to a charmed hadron.
  3. Identifies the IMMEDIATE parent particle of mu2 (parent = mc_tracks[mu2.GetMotherId()]).
  4. Records the PDG code of the parent particle for each charm species and for all combined.

Produces:
  - Histograms of mu2 parent PDG codes for each charm species (D0, D+, Ds, Lambda_c, etc.).
  - Combined histogram for all charm species together.
  - Labeled bins with particle names (e.g. 'pi+ (211)', 'K- (-321)', 'D+ (411)', etc.).
  - All histograms formatted with 'HIST' draw style.
  - Detailed terminal summary table analyzing whether mu2 is prompt (from charm) or decay-in-flight (from pi/K).

Usage:
  python3 scripts/diagnostics/mu2_diagnostic.py
  python3 scripts/diagnostics/mu2_diagnostic.py -i "/eos/experiment/sndlhc/MonteCarlo/.../0/sndLHC.Genie-TGeant4_digCPP.root"
  python3 scripts/diagnostics/mu2_diagnostic.py -i "/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_truth.root" -n 1000
"""

from __future__ import annotations

import os
import sys
import math
import time
import argparse
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter

import ROOT
ROOT.gROOT.SetBatch(True)

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import DataManager


# Common particle names by PDG code for clean histogram labels
PDG_NAMES: Dict[int, str] = {
    11: "e-",
    -11: "e+",
    12: "nu_e",
    -12: "anti_nu_e",
    13: "mu-",
    -13: "mu+",
    14: "nu_mu",
    -14: "anti_nu_mu",
    15: "tau-",
    -15: "tau+",
    16: "nu_tau",
    -16: "anti_nu_tau",
    22: "gamma",
    111: "pi0",
    211: "pi+",
    -211: "pi-",
    130: "K_L0",
    310: "K_S0",
    311: "K0",
    -311: "anti_K0",
    321: "K+",
    -321: "K-",
    221: "eta",
    331: "eta'",
    2112: "n",
    -2112: "anti_n",
    2212: "p",
    -2212: "anti_p",
    3122: "Lambda0",
    -3122: "anti_Lambda0",
    3222: "Sigma+",
    -3222: "anti_Sigma-",
    3212: "Sigma0",
    -3212: "anti_Sigma0",
    3112: "Sigma-",
    -3112: "anti_Sigma+",
    3322: "Xi0",
    -3322: "anti_Xi0",
    3312: "Xi-",
    -3312: "anti_Xi+",
    411: "D+",
    -411: "D-",
    421: "D0",
    -421: "anti_D0",
    431: "Ds+",
    -431: "Ds-",
    413: "D*+",
    -413: "D*-",
    423: "D*0",
    -423: "anti_D*0",
    433: "Ds*+",
    -433: "Ds*-",
    4122: "Lambda_c+",
    -4122: "anti_Lambda_c-",
    4222: "Sigma_c++",
    4212: "Sigma_c+",
    4112: "Sigma_c0",
    4232: "Xi_c+",
    4132: "Xi_c0",
    4332: "Omega_c0",
}

SPECIES_NAMES: Dict[int, str] = {
    421: "D0",
    411: "Dplus",
    431: "Ds",
    4122: "Lambda_c",
    413: "Dstar_plus",
    423: "Dstar0",
    433: "Ds_star",
    4222: "Sigma_c_pp",
    4212: "Sigma_c_p",
    4112: "Sigma_c_0",
    4232: "Xi_c_p",
    4132: "Xi_c_0",
    4332: "Omega_c_0",
}


def is_charm_hadron(pdg: int) -> bool:
    """Check if a PDG code corresponds to a charmed hadron."""
    code = abs(pdg)
    if code < 100 or code >= 10000000:
        return False
    mod10 = code // 10
    q1 = mod10 % 10
    q2 = (mod10 // 10) % 10
    q3 = (mod10 // 100) % 10
    return (q1 == 4 or q2 == 4 or q3 == 4)


def get_particle_label(pdg: int) -> str:
    """Format readable particle label (e.g. 'pi+ (211)')."""
    name = PDG_NAMES.get(pdg, f"PDG_{pdg}")
    return f"{name} ({pdg})"


def get_species_name(abs_pdg: int) -> str:
    """Format readable charm species name."""
    return SPECIES_NAMES.get(abs_pdg, f"Charm_{abs_pdg}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic script to analyze the parent particles of secondary muons (mu2) from charm decay chains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_input = (
        "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/"
        "sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/"
        "sndLHC.Genie-TGeant4_digCPP.root"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=default_input,
        help="Input ROOT file, directory, or pattern to load via DataManager",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="mu2_parent_diagnostic.root",
        help="Output ROOT file to store parent PDG histograms",
    )
    parser.add_argument(
        "-n", "--max-events",
        type=int,
        default=-1,
        help="Maximum number of events to process (-1 for all)",
    )
    parser.add_argument(
        "--max-flight-distance",
        type=float,
        default=100.0,
        help="Maximum distance [cm] between charm vertex and mu2 production vertex",
    )
    parser.add_argument(
        "--min-muon-momentum",
        type=float,
        default=0.0,
        help="Minimum momentum threshold for mu2 [GeV/c]",
    )
    parser.add_argument(
        "--progress-step",
        type=int,
        default=100,
        help="Print progress every N events",
    )

    args = parser.parse_args()

    print("=" * 80)
    print(" SND@LHC: Secondary Muon (mu2) Parent Particle Diagnostic")
    print("=" * 80)
    print(f" Input Source         : {args.input}")
    print(f" Output ROOT File     : {os.path.abspath(args.output)}")
    print(f" Max Events           : {'All' if args.max_events < 0 else args.max_events}")
    print(f" Max Flight Distance  : {args.max_flight_distance} cm")
    print(f" Min Muon Momentum    : {args.min_muon_momentum} GeV/c")
    print("=" * 80)

    # 1. Load dataset with DataManager
    print("\n[1/3] Loading dataset with DataManager...")
    dm = DataManager(source=args.input, tree_name="cbmsim")
    chain = dm.get_chain()
    total_entries = chain.GetEntries()
    print(f"  Loaded chain successfully! Total available events: {total_entries}")

    # Optimize I/O: activate necessary branches
    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("MCTrack*", 1)
    if chain.GetBranch("mu2_track_id"):
        chain.SetBranchStatus("mu2_track_id*", 1)
    if chain.GetBranch("charm_track_id"):
        chain.SetBranchStatus("charm_track_id*", 1)
    if chain.GetBranch("charm_pdg"):
        chain.SetBranchStatus("charm_pdg*", 1)

    n_to_process = total_entries if args.max_events < 0 else min(args.max_events, total_entries)
    print(f"  Will process {n_to_process} events.\n")

    # Tracking counters
    # parent_counts_by_species[abs_charm_pdg][parent_pdg] -> count
    parent_counts_by_species: Dict[int, Dict[int, int]] = {}
    all_parent_counts: Dict[int, int] = {}
    # parent_process_counts[parent_pdg][proc_name] -> count
    parent_process_counts: Dict[int, Dict[str, int]] = {}

    total_charm_events = 0
    total_mu2_found = 0
    direct_charm_parent_count = 0
    secondary_parent_count = 0

    # 2. Main Event Loop
    print("[2/3] Running event loop...")
    t_start = time.time()

    for i_event in range(n_to_process):
        chain.GetEntry(i_event)

        # Check for MCTrack collection
        if not hasattr(chain, "MCTrack"):
            continue

        mc_tracks = chain.MCTrack
        n_tracks = mc_tracks.GetEntries()
        if n_tracks < 2:
            continue

        # Step A: Identify all charm hadrons in event
        charm_indices = []
        for i_track in range(n_tracks):
            trk = mc_tracks[i_track]
            if is_charm_hadron(trk.GetPdgCode()):
                charm_indices.append(i_track)

        if not charm_indices:
            continue

        total_charm_events += 1

        # Step B: Identify primary prompt muon (mu1) from interaction vertex (motherId == 0)
        primary_mu1_id = -1
        max_mu1_p = -1.0
        for i_track in range(n_tracks):
            trk = mc_tracks[i_track]
            if abs(trk.GetPdgCode()) == 13 and trk.GetMotherId() == 0:
                if trk.GetP() > max_mu1_p:
                    max_mu1_p = trk.GetP()
                    primary_mu1_id = i_track

        # Step C: Search for mu2 candidate(s) originating from charm ancestry
        # If pre-extracted branch mu2_track_id exists and is valid, prefer it
        found_mu2_tracks: List[Tuple[int, int, int]] = [] # (mu2_track_id, parent_pdg, ancestor_charm_abs_pdg)

        if hasattr(chain, "mu2_track_id") and getattr(chain, "mu2_track_id", -1) is not None:
            try:
                mu2_id = int(chain.mu2_track_id)
                if 0 <= mu2_id < n_tracks:
                    mu2_trk = mc_tracks[mu2_id]
                    parent_id = mu2_trk.GetMotherId()
                    if 0 <= parent_id < n_tracks:
                        parent_trk = mc_tracks[parent_id]
                        parent_pdg = parent_trk.GetPdgCode()

                        charm_abs = 0
                        if hasattr(chain, "charm_pdg") and getattr(chain, "charm_pdg", 0) is not None:
                            charm_abs = abs(int(chain.charm_pdg))
                        if charm_abs == 0 and hasattr(chain, "charm_track_id"):
                            c_id = int(chain.charm_track_id)
                            if 0 <= c_id < n_tracks:
                                charm_abs = abs(mc_tracks[c_id].GetPdgCode())
                        if charm_abs == 0:
                            charm_abs = abs(mc_tracks[charm_indices[0]].GetPdgCode())

                        found_mu2_tracks.append((mu2_id, parent_pdg, charm_abs))
            except Exception:
                pass

        # If not already found from truth branches, perform full ancestry trace
        if not found_mu2_tracks:
            best_mu2_id = -1
            max_mu2_p = -1.0
            best_parent_pdg = 0
            best_charm_abs_pdg = 0

            for i_track in range(1, n_tracks):
                if i_track == primary_mu1_id:
                    continue

                trk = mc_tracks[i_track]
                if abs(trk.GetPdgCode()) != 13:
                    continue
                if trk.GetP() < args.min_muon_momentum:
                    continue

                # Trace mother ancestry upwards
                curr_mid = trk.GetMotherId()
                matched_charm_id = -1

                while curr_mid >= 0 and curr_mid < n_tracks:
                    parent_trk = mc_tracks[curr_mid]
                    if is_charm_hadron(parent_trk.GetPdgCode()):
                        matched_charm_id = curr_mid
                        break
                    curr_mid = parent_trk.GetMotherId()

                if matched_charm_id != -1:
                    # Check physical flight distance from charm origin
                    c_trk = mc_tracks[matched_charm_id]
                    dx = trk.GetStartX() - c_trk.GetStartX()
                    dy = trk.GetStartY() - c_trk.GetStartY()
                    dz = trk.GetStartZ() - c_trk.GetStartZ()
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)

                    if dist <= args.max_flight_distance:
                        if trk.GetP() > max_mu2_p:
                            max_mu2_p = trk.GetP()
                            best_mu2_id = i_track
                            # Immediate parent
                            immed_parent_id = trk.GetMotherId()
                            if 0 <= immed_parent_id < n_tracks:
                                best_parent_pdg = mc_tracks[immed_parent_id].GetPdgCode()
                            best_charm_abs_pdg = abs(c_trk.GetPdgCode())

            if best_mu2_id != -1:
                found_mu2_tracks.append((best_mu2_id, best_parent_pdg, best_charm_abs_pdg))

        # Record findings
        for mu2_id, parent_pdg, charm_abs in found_mu2_tracks:
            total_mu2_found += 1
            mu2_trk = mc_tracks[mu2_id]
            proc = mu2_trk.GetProcName()

            # Parent tracking
            if charm_abs not in parent_counts_by_species:
                parent_counts_by_species[charm_abs] = {}

            parent_counts_by_species[charm_abs][parent_pdg] = (
                parent_counts_by_species[charm_abs].get(parent_pdg, 0) + 1
            )
            all_parent_counts[parent_pdg] = all_parent_counts.get(parent_pdg, 0) + 1

            # Process tracking
            if parent_pdg not in parent_process_counts:
                parent_process_counts[parent_pdg] = {}
            parent_process_counts[parent_pdg][proc] = (
                parent_process_counts[parent_pdg].get(proc, 0) + 1
            )

            # Check if parent is directly charm
            if is_charm_hadron(parent_pdg):
                direct_charm_parent_count += 1
            else:
                secondary_parent_count += 1

        # Progress reporting
        if (i_event + 1) % args.progress_step == 0 or (i_event + 1) == n_to_process:
            elapsed = time.time() - t_start
            rate = (i_event + 1) / max(elapsed, 0.001)
            print(f"  Processed {i_event + 1:6d} / {n_to_process:6d} events "
                  f"({100.0 * (i_event + 1) / n_to_process:5.1f}%) | "
                  f"Charm events: {total_charm_events:4d} | "
                  f"mu2 found: {total_mu2_found:4d} | "
                  f"Rate: {rate:5.1f} ev/s", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\n  Done! Processed {n_to_process} events in {elapsed_total:.2f}s "
          f"({n_to_process / max(elapsed_total, 0.001):.1f} ev/s).")
    print(f"  Found {total_charm_events} charm events, {total_mu2_found} mu2 tracks.\n")

    # 3. Create ROOT histograms
    print("[3/3] Creating and writing ROOT histograms (with HIST draw style)...")
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fout = ROOT.TFile.Open(args.output, "RECREATE")
    if not fout or fout.IsZombie():
        print(f"[Error] Failed to create output ROOT file: {args.output}")
        sys.exit(1)

    sorted_parent_pdgs = [pdg for pdg, _ in sorted(all_parent_counts.items(), key=lambda x: x[1], reverse=True)]

    def book_parent_histograms(
        tag: str,
        display_title: str,
        counts_map: Dict[int, int],
    ) -> Tuple[ROOT.TH1D, ROOT.TH1D, ROOT.TH1D]:
        """
        Creates three histograms for mu2 parent particle:
          1. h_mu2_parent_labeled_<tag>: Categorical with readable particle names
          2. h_mu2_parent_pdg_<tag>: Numeric signed PDG code (-4000 to +4000)
          3. h_mu2_parent_abs_pdg_<tag>: Numeric absolute PDG code (0 to 4000)
        All histograms have default draw option 'HIST'.
        """
        # 1. Categorical / Labeled
        n_bins = len(sorted_parent_pdgs) if sorted_parent_pdgs else 1
        h_labeled = ROOT.TH1D(
            f"h_mu2_parent_labeled_{tag}",
            f"mu2 Parent Particle Identification ({display_title});Parent Particle;Entries",
            n_bins, 0.5, n_bins + 0.5,
        )
        h_labeled.SetDirectory(ROOT.gDirectory)

        for idx, p_pdg in enumerate(sorted_parent_pdgs, start=1):
            h_labeled.GetXaxis().SetBinLabel(idx, get_particle_label(p_pdg))

        # 2. Signed Numeric PDG
        h_signed = ROOT.TH1D(
            f"h_mu2_parent_pdg_{tag}",
            f"mu2 Parent Signed PDG ({display_title});Parent PDG;Entries",
            8001, -4000.5, 4000.5,
        )
        h_signed.SetDirectory(ROOT.gDirectory)

        # 3. Absolute Numeric PDG
        h_abs = ROOT.TH1D(
            f"h_mu2_parent_abs_pdg_{tag}",
            f"mu2 Parent |PDG| ({display_title});|Parent PDG|;Entries",
            4000, 0.5, 4000.5,
        )
        h_abs.SetDirectory(ROOT.gDirectory)

        # Fill counts
        for p_pdg, count in counts_map.items():
            h_signed.Fill(p_pdg, count)
            h_abs.Fill(abs(p_pdg), count)
            if p_pdg in sorted_parent_pdgs:
                bin_num = sorted_parent_pdgs.index(p_pdg) + 1
                h_labeled.SetBinContent(bin_num, h_labeled.GetBinContent(bin_num) + count)

        # Set HIST draw option on all
        for h in [h_labeled, h_signed, h_abs]:
            h.SetOption("HIST")
            h.SetDrawOption("HIST")

        return h_labeled, h_signed, h_abs

    # --- A. Combined Histogram (All Charm Species) ---
    fout.cd()
    dir_all = fout.mkdir("AllCharm")
    dir_all.cd()
    h_all_labeled, h_all_signed, h_all_abs = book_parent_histograms(
        tag="all_charm",
        display_title="All Charm Species Combined",
        counts_map=all_parent_counts,
    )
    h_all_labeled.Write()
    h_all_signed.Write()
    h_all_abs.Write()

    # Also write copies to top directory of ROOT file
    fout.cd()
    h_all_labeled.Write("h_all_mu2_parent_labeled")
    h_all_signed.Write("h_all_mu2_parent_pdg")
    h_all_abs.Write("h_all_mu2_parent_abs_pdg")

    # --- B. Per-Species Histograms ---
    species_dir = fout.mkdir("Species")
    for abs_c in sorted(parent_counts_by_species.keys()):
        sp_name = get_species_name(abs_c)
        c_counts = parent_counts_by_species[abs_c]

        species_dir.cd()
        sub_dir = species_dir.mkdir(sp_name)
        sub_dir.cd()

        h_sp_labeled, h_sp_signed, h_sp_abs = book_parent_histograms(
            tag=sp_name,
            display_title=f"Charm {sp_name} (|PDG|={abs_c})",
            counts_map=c_counts,
        )
        h_sp_labeled.Write()
        h_sp_signed.Write()
        h_sp_abs.Write()

        # Also write clean top-level histograms
        fout.cd()
        h_sp_labeled.Write(f"h_{sp_name}_mu2_parent_labeled")
        h_sp_signed.Write(f"h_{sp_name}_mu2_parent_pdg")
        h_sp_abs.Write(f"h_{sp_name}_mu2_parent_abs_pdg")

    fout.Close()
    print(f"  Histograms successfully written to: {os.path.abspath(args.output)}")

    # 4. Terminal Summary Table
    print("\n" + "=" * 80)
    print(" SECONDARY MUON (mu2) PARENT PARTICLE SUMMARY")
    print("=" * 80)
    print(f" Total Charm Events Analyzed : {total_charm_events}")
    print(f" Total mu2 Muons Identified  : {total_mu2_found}")
    if total_mu2_found > 0:
        dir_frac = 100.0 * direct_charm_parent_count / total_mu2_found
        sec_frac = 100.0 * secondary_parent_count / total_mu2_found
        print(f"   -> Direct Prompt Charm Parent (D / Lambda_c -> mu X) : {direct_charm_parent_count:4d} ({dir_frac:5.1f}%)")
        print(f"   -> Secondary Decay-in-Flight Parent (pi / K -> mu nu): {secondary_parent_count:4d} ({sec_frac:5.1f}%)")
    print("=" * 80)

    # Combined Table
    tot_all_p = sum(all_parent_counts.values())
    print(f"\n[All Charm Ancestry Combined] (Total mu2: {total_mu2_found})")
    print(f" {'Parent Particle':<18s} {'Parent PDG':<12s} {'Count':<8s} {'Fraction':<12s} {'Origin Type':<26s}")
    print("-" * 80)
    for p_pdg in sorted_parent_pdgs:
        cnt = all_parent_counts[p_pdg]
        frac = 100.0 * cnt / max(tot_all_p, 1)
        name = PDG_NAMES.get(p_pdg, f"PDG_{p_pdg}")
        orig = "Direct Charm Decay" if is_charm_hadron(p_pdg) else "Secondary Decay-in-Flight"
        print(f" {name:<18s} {p_pdg:<12d} {cnt:<8d} {frac:6.2f}%      {orig:<26s}")

    # Per-Species Table
    for abs_c in sorted(parent_counts_by_species.keys()):
        sp_name = get_species_name(abs_c)
        c_dict = parent_counts_by_species[abs_c]
        tot_sp = sum(c_dict.values())
        sorted_sp = sorted(c_dict.items(), key=lambda x: x[1], reverse=True)

        print(f"\n[{sp_name} Ancestry] (|PDG|={abs_c}, mu2 count: {tot_sp})")
        print(f" {'Parent Particle':<18s} {'Parent PDG':<12s} {'Count':<8s} {'Fraction':<12s} {'Origin Type':<26s}")
        print("-" * 80)
        for p_pdg, cnt in sorted_sp:
            frac = 100.0 * cnt / max(tot_sp, 1)
            name = PDG_NAMES.get(p_pdg, f"PDG_{p_pdg}")
            orig = "Direct Charm Decay" if is_charm_hadron(p_pdg) else "Secondary Decay-in-Flight"
            print(f" {name:<18s} {p_pdg:<12d} {cnt:<8d} {frac:6.2f}%      {orig:<26s}")

    print("\n" + "=" * 80)
    print(" Diagnostic completed successfully!")
    print(f" Output ROOT file: {os.path.abspath(args.output)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
