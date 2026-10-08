#!/usr/bin/env python3
"""
scripts/charm_diagnostic.py
---------------------------
Diagnostic tool to inspect Monte Carlo truth charmed hadrons and their direct decay products.

Uses snd.DataManager to load the dataset, then runs a transparent Python loop over events
and MCTracks with explicit manual checks.

Specifically tests whether this Monte Carlo production implements realistic inclusive branching
ratios or is constrained to forced/exclusive decay channels (such as only the most probable channel).

Produces:
  - Exclusive decay channel analysis for each charm species.
  - One histogram per charm species (D0, D+, Ds, Lambda_c, etc.).
  - One combined histogram for all charm species together.
  - Histograms of full decay channel configurations (e.g. "K- pi+ pi0", "K- pi+ pi+").
  - Labeled bins with particle names (e.g. pi+, K-, pi0, mu+, e-, etc.) and numeric PDG histograms.
  - Comprehensive terminal diagnostics testing for forced branching ratios.

Usage:
  python3 scripts/charm_diagnostic.py
  python3 scripts/charm_diagnostic.py -i "/eos/experiment/sndlhc/MonteCarlo/.../0/sndLHC.Genie-TGeant4_digCPP.root"
  python3 scripts/charm_diagnostic.py -n 500 -o charm_decays.root
"""

from __future__ import annotations

import os
import sys
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

# Standard charm species names
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

# PDG World Average inclusive semi-leptonic branching fractions for reference
PDG_SEMILEPTONIC_BR: Dict[int, Tuple[float, float]] = {
    421: (6.8, 6.7),    # D0: BR(e+ X) ~ 6.5%, BR(mu+ X) ~ 6.8% (Total semi-leptonic ~ 13.5%)
    411: (16.1, 15.6),  # D+: BR(e+ X) ~ 16.1%, BR(mu+ X) ~ 15.6%
    431: (6.3, 6.3),    # Ds+: BR(e+ X) ~ 6.3%, BR(mu+ X) ~ 6.3%
    4122: (4.0, 3.5),   # Lambda_c+: BR(e+ X) ~ 4.0%, BR(mu+ X) ~ 3.5%
}


def is_charm_hadron(pdg: int) -> bool:
    """
    Checks if a PDG code corresponds to a charmed hadron (meson or baryon).
    Excludes nuclear codes (|PDG| >= 10000000).
    A hadron contains a charm quark if one of its constituent quark digits is 4.
    """
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
    """Format readable charm species name (e.g. 'D0', 'Dplus', 'Ds', 'Lambda_c')."""
    return SPECIES_NAMES.get(abs_pdg, f"Charm_{abs_pdg}")


def format_channel_formula(pdg_tuple: Tuple[int, ...]) -> str:
    """Format a tuple of daughter PDGs into human-readable decay channel formula."""
    names = [PDG_NAMES.get(p, str(p)) for p in pdg_tuple]
    return " + ".join(sorted(names))


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic script to analyze MCTrack charmed hadrons and test for forced decay channels.",
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
        default="charm_decay_diagnostic.root",
        help="Output ROOT file to store histograms",
    )
    parser.add_argument(
        "-n", "--max-events",
        type=int,
        default=-1,
        help="Maximum number of events to process (-1 for all)",
    )
    parser.add_argument(
        "--decay-only",
        action="store_true",
        default=True,
        help="Only count direct daughters from 'Decay' process (excludes hadronic inelastic scattering in material)",
    )
    parser.add_argument(
        "--include-all-interactions",
        action="store_false",
        dest="decay_only",
        help="Include all secondaries (even hadronic inelastic collisions in detector material)",
    )
    parser.add_argument(
        "--ground-state-only",
        action="store_true",
        default=False,
        help="Only analyze ground-state weakly decaying charm hadrons (D0, D+, Ds, Lambda_c)",
    )
    parser.add_argument(
        "--progress-step",
        type=int,
        default=100,
        help="Print progress every N events",
    )

    args = parser.parse_args()

    print("=" * 80)
    print(" SND@LHC: Charmed Hadron Decay Daughters Diagnostic & Forced Channel Test")
    print("=" * 80)
    print(f" Input Source         : {args.input}")
    print(f" Output ROOT File     : {os.path.abspath(args.output)}")
    print(f" Max Events           : {'All' if args.max_events < 0 else args.max_events}")
    print(f" Decay Process Only   : {args.decay_only} (filters out material inelastic collisions)")
    print(f" Ground State Only    : {args.ground_state_only} (only D0, D+, Ds, Lambda_c)")
    print("=" * 80)

    # 1. Load dataset with DataManager
    print("\n[1/3] Loading dataset with DataManager...")
    dm = DataManager(source=args.input, tree_name="cbmsim")
    chain = dm.get_chain()
    total_entries = chain.GetEntries()
    print(f"  Loaded chain successfully! Total available events: {total_entries}")

    # Optimize I/O: only read the MCTrack branch from disk
    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("MCTrack*", 1)

    n_to_process = total_entries if args.max_events < 0 else min(args.max_events, total_entries)
    print(f"  Will process {n_to_process} events.\n")

    # Tracking counters
    # species_counts[abs_charm_pdg][daughter_pdg] -> count
    species_counts: Dict[int, Dict[int, int]] = {}
    # all_counts[daughter_pdg] -> count
    all_counts: Dict[int, int] = {}
    # decay_channels_by_species[abs_charm_pdg][tuple(sorted_daughters)] -> count
    decay_channels_by_species: Dict[int, Dict[Tuple[int, ...], int]] = {}
    all_decay_channels: Dict[Tuple[int, ...], int] = {}

    # Semi-leptonic decay counts
    semi_muonic_counts: Dict[int, int] = {}
    semi_electronic_counts: Dict[int, int] = {}
    total_decays_by_species: Dict[int, int] = {}

    # species_hadron_counts[abs_charm_pdg] -> number of charm hadrons found
    species_hadron_counts: Dict[int, int] = {}
    total_charm_hadrons = 0

    ground_state_pdgs = {411, 421, 431, 4122}

    # 2. Main Event Loop
    print("[2/3] Running event loop...")
    t_start = time.time()

    for i_event in range(n_to_process):
        chain.GetEntry(i_event)

        # Manual check 1: Event must have MCTrack collection
        if not hasattr(chain, "MCTrack"):
            continue

        mc_tracks = chain.MCTrack
        n_tracks = mc_tracks.GetEntries()
        if n_tracks < 2:
            continue

        # Manual check 2: Find all charm hadrons in this event
        # Store as list of tuples: (track_index, pdg_code, abs_pdg_code)
        charm_hadrons_in_event: List[Tuple[int, int, int]] = []

        for i_track in range(n_tracks):
            track = mc_tracks[i_track]
            pdg = track.GetPdgCode()

            if is_charm_hadron(pdg):
                abs_pdg = abs(pdg)
                if args.ground_state_only and abs_pdg not in ground_state_pdgs:
                    continue

                charm_hadrons_in_event.append((i_track, pdg, abs_pdg))

                # Count charm hadrons
                species_hadron_counts[abs_pdg] = species_hadron_counts.get(abs_pdg, 0) + 1
                total_charm_hadrons += 1

        # If no charm hadron found in this event, proceed to next event
        if not charm_hadrons_in_event:
            continue

        # Manual check 3: For each charm hadron, find its DIRECT daughters
        # A track is a DIRECT daughter of charm hadron `i_charm` if:
        #   candidate.GetMotherId() == i_charm
        # (Daughters of daughters have candidate.GetMotherId() == daughter_index, so they are excluded!)
        for i_charm, charm_pdg, abs_charm_pdg in charm_hadrons_in_event:
            if abs_charm_pdg not in species_counts:
                species_counts[abs_charm_pdg] = {}
            if abs_charm_pdg not in decay_channels_by_species:
                decay_channels_by_species[abs_charm_pdg] = {}

            daughters: List[int] = []

            for j_track in range(n_tracks):
                candidate_daughter = mc_tracks[j_track]

                # Direct daughter check: Mother ID must match charm track index
                if candidate_daughter.GetMotherId() == i_charm:
                    # Optional check: Process must be 'Decay' (to exclude nuclear material collisions)
                    proc_name = candidate_daughter.GetProcName()
                    if args.decay_only and proc_name != "Decay":
                        continue

                    daughter_pdg = candidate_daughter.GetPdgCode()
                    daughters.append(daughter_pdg)

                    # Record daughter count for this species
                    species_counts[abs_charm_pdg][daughter_pdg] = (
                        species_counts[abs_charm_pdg].get(daughter_pdg, 0) + 1
                    )
                    # Record daughter count for all charm combined
                    all_counts[daughter_pdg] = all_counts.get(daughter_pdg, 0) + 1

            if daughters:
                total_decays_by_species[abs_charm_pdg] = total_decays_by_species.get(abs_charm_pdg, 0) + 1

                # Record full exclusive decay channel (sorted tuple of daughter PDGs)
                ch_tuple = tuple(sorted(daughters))
                decay_channels_by_species[abs_charm_pdg][ch_tuple] = (
                    decay_channels_by_species[abs_charm_pdg].get(ch_tuple, 0) + 1
                )
                all_decay_channels[ch_tuple] = all_decay_channels.get(ch_tuple, 0) + 1

                # Check semi-leptonic content
                if any(abs(d) == 13 for d in daughters):
                    semi_muonic_counts[abs_charm_pdg] = semi_muonic_counts.get(abs_charm_pdg, 0) + 1
                if any(abs(d) == 11 for d in daughters):
                    semi_electronic_counts[abs_charm_pdg] = semi_electronic_counts.get(abs_charm_pdg, 0) + 1

        # Progress reporting
        if (i_event + 1) % args.progress_step == 0 or (i_event + 1) == n_to_process:
            elapsed = time.time() - t_start
            rate = (i_event + 1) / max(elapsed, 0.001)
            print(f"  Processed {i_event + 1:6d} / {n_to_process:6d} events "
                  f"({100.0 * (i_event + 1) / n_to_process:5.1f}%) | "
                  f"Charm hadrons: {total_charm_hadrons:4d} | "
                  f"Rate: {rate:5.1f} ev/s", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\n  Done! Processed {n_to_process} events in {elapsed_total:.2f}s "
          f"({n_to_process / max(elapsed_total, 0.001):.1f} ev/s).", flush=True)
    print(f"  Found {total_charm_hadrons} total charm hadrons.\n", flush=True)

    # 3. Create ROOT histograms
    print("[3/3] Creating and writing ROOT histograms...")
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fout = ROOT.TFile.Open(args.output, "RECREATE")
    if not fout or fout.IsZombie():
        print(f"[Error] Failed to create output ROOT file: {args.output}")
        sys.exit(1)

    # Sorted list of unique daughter PDGs by overall frequency
    sorted_d_pdgs = [pdg for pdg, _ in sorted(all_counts.items(), key=lambda x: x[1], reverse=True)]

    def book_histograms_for_category(
        tag: str,
        display_title: str,
        counts_map: Dict[int, int],
        channels_map: Optional[Dict[Tuple[int, ...], int]] = None,
    ) -> Tuple[ROOT.TH1D, ROOT.TH1D, ROOT.TH1D, Optional[ROOT.TH1D]]:
        """
        Creates histograms for a given category:
          1. h_daughters_labeled_<tag>: Categorical with readable particle names
          2. h_daughters_pdg_<tag>: Numeric signed PDG code (-4000 to +4000)
          3. h_daughters_abs_pdg_<tag>: Numeric absolute PDG code (0 to 4000)
          4. h_decay_channels_<tag>: Full decay channel formulas (e.g. 'K- pi+ pi0')
        """
        # 1. Categorical / Labeled
        n_bins = len(sorted_d_pdgs) if sorted_d_pdgs else 1
        h_labeled = ROOT.TH1D(
            f"h_daughters_labeled_{tag}",
            f"Decay Daughters Particle ID ({display_title});Daughter Particle;Entries",
            n_bins, 0.5, n_bins + 0.5,
        )
        h_labeled.SetDirectory(ROOT.gDirectory)

        for idx, d_pdg in enumerate(sorted_d_pdgs, start=1):
            h_labeled.GetXaxis().SetBinLabel(idx, get_particle_label(d_pdg))

        # 2. Signed Numeric PDG
        h_signed = ROOT.TH1D(
            f"h_daughters_pdg_{tag}",
            f"Decay Daughters Signed PDG ({display_title});Daughter PDG;Entries",
            8001, -4000.5, 4000.5,
        )
        h_signed.SetDirectory(ROOT.gDirectory)

        # 3. Absolute Numeric PDG
        h_abs = ROOT.TH1D(
            f"h_daughters_abs_pdg_{tag}",
            f"Decay Daughters |PDG| ({display_title});|Daughter PDG|;Entries",
            4000, 0.5, 4000.5,
        )
        h_abs.SetDirectory(ROOT.gDirectory)

        # Fill counts
        for d_pdg, count in counts_map.items():
            h_signed.Fill(d_pdg, count)
            h_abs.Fill(abs(d_pdg), count)
            if d_pdg in sorted_d_pdgs:
                bin_num = sorted_d_pdgs.index(d_pdg) + 1
                h_labeled.SetBinContent(bin_num, h_labeled.GetBinContent(bin_num) + count)

        # 4. Exclusive Decay Channels Histogram
        h_channels = None
        if channels_map:
            sorted_channels = sorted(channels_map.items(), key=lambda x: x[1], reverse=True)
            n_ch_bins = min(len(sorted_channels), 25)
            h_channels = ROOT.TH1D(
                f"h_decay_channels_{tag}",
                f"Exclusive Decay Channels ({display_title});Decay Channel;Decays",
                n_ch_bins, 0.5, n_ch_bins + 0.5
            )
            h_channels.SetDirectory(ROOT.gDirectory)
            for ch_idx, (ch_tuple, ch_cnt) in enumerate(sorted_channels[:n_ch_bins], start=1):
                ch_label = format_channel_formula(ch_tuple)
                h_channels.GetXaxis().SetBinLabel(ch_idx, ch_label)
                h_channels.SetBinContent(ch_idx, ch_cnt)

        # Set HIST draw option as default
        for h in [h_labeled, h_signed, h_abs]:
            h.SetOption("HIST")
            h.SetDrawOption("HIST")
        if h_channels:
            h_channels.SetOption("HIST")
            h_channels.SetDrawOption("HIST")

        return h_labeled, h_signed, h_abs, h_channels

    # --- A. Combined Histogram (All Charm Species) ---
    fout.cd()
    dir_all = fout.mkdir("AllCharm")
    dir_all.cd()
    h_all_labeled, h_all_signed, h_all_abs, h_all_channels = book_histograms_for_category(
        tag="all_charm",
        display_title="All Charmed Hadrons Combined",
        counts_map=all_counts,
        channels_map=all_decay_channels,
    )
    h_all_labeled.Write()
    h_all_signed.Write()
    h_all_abs.Write()
    if h_all_channels:
        h_all_channels.Write()

    # Also write copies to top directory of ROOT file
    fout.cd()
    h_all_labeled.Write("h_all_charm_daughters_labeled")
    h_all_signed.Write("h_all_charm_daughters_pdg")
    h_all_abs.Write("h_all_charm_daughters_abs_pdg")
    if h_all_channels:
        h_all_channels.Write("h_all_charm_decay_channels")

    # --- B. Per-Species Histograms ---
    species_dir = fout.mkdir("Species")
    for abs_c in sorted(species_counts.keys()):
        sp_name = get_species_name(abs_c)
        c_counts = species_counts[abs_c]
        c_channels = decay_channels_by_species.get(abs_c, {})

        species_dir.cd()
        sub_dir = species_dir.mkdir(sp_name)
        sub_dir.cd()

        h_sp_labeled, h_sp_signed, h_sp_abs, h_sp_channels = book_histograms_for_category(
            tag=sp_name,
            display_title=f"{sp_name} (|PDG|={abs_c})",
            counts_map=c_counts,
            channels_map=c_channels,
        )
        h_sp_labeled.Write()
        h_sp_signed.Write()
        h_sp_abs.Write()
        if h_sp_channels:
            h_sp_channels.Write()

        # Also write clean top-level histograms
        fout.cd()
        h_sp_labeled.Write(f"h_{sp_name}_daughters_labeled")
        h_sp_signed.Write(f"h_{sp_name}_daughters_pdg")
        h_sp_abs.Write(f"h_{sp_name}_daughters_abs_pdg")
        if h_sp_channels:
            h_sp_channels.Write(f"h_{sp_name}_decay_channels")

    fout.Close()
    print(f"  Histograms successfully written to: {os.path.abspath(args.output)}")

    # 4. Terminal Summary Table
    print("\n" + "=" * 80)
    print(" CHARMED HADRON SPECIES SUMMARY")
    print("=" * 80)
    print(f" {'Species':<16s} {'|PDG|':<8s} {'Count':<8s} {'Share':<10s} {'Total Decay Daughters':<24s}")
    print("-" * 80)
    for abs_c in sorted(species_hadron_counts.keys()):
        sp_name = get_species_name(abs_c)
        cnt = species_hadron_counts[abs_c]
        share = 100.0 * cnt / max(total_charm_hadrons, 1)
        tot_d = sum(species_counts.get(abs_c, {}).values())
        print(f" {sp_name:<16s} {abs_c:<8d} {cnt:<8d} {share:6.2f}%    {tot_d:<24d}")
    print("-" * 80)

    # 5. DIAGNOSTIC: Test for forced decay channels & branching ratios
    print("\n" + "=" * 80)
    print(" CHARM DECAY BRANCHING RATIO DIAGNOSTIC & FORCED CHANNEL TEST")
    print("=" * 80)

    any_forced_detected = False

    for abs_c in sorted(decay_channels_by_species.keys()):
        sp_name = get_species_name(abs_c)
        ch_counter = decay_channels_by_species[abs_c]
        tot_decays = total_decays_by_species.get(abs_c, sum(ch_counter.values()))
        if tot_decays == 0:
            continue

        sorted_channels = sorted(ch_counter.items(), key=lambda x: x[1], reverse=True)
        top_channel, top_count = sorted_channels[0]
        top_frac = 100.0 * top_count / tot_decays
        top_formula = format_channel_formula(top_channel)

        n_mu = semi_muonic_counts.get(abs_c, 0)
        frac_mu = 100.0 * n_mu / tot_decays
        n_e = semi_electronic_counts.get(abs_c, 0)
        frac_e = 100.0 * n_e / tot_decays

        print(f"\n[{sp_name}] (|PDG| = {abs_c}, Total Analyzed Decays = {tot_decays}):")
        print(f"  Distinct Decay Channels Observed : {len(ch_counter)}")
        print(f"  Most Frequent Decay Channel      : {top_formula}")
        print(f"  Top Channel Fraction             : {top_count}/{tot_decays} ({top_frac:5.1f}%)")
        print(f"  Semi-muonic Decays (-> mu X)     : {n_mu}/{tot_decays} ({frac_mu:5.1f}%)")
        print(f"  Semi-electronic Decays (-> e X)  : {n_e}/{tot_decays} ({frac_e:5.1f}%)")

        # Compare with PDG expectation
        if abs_c in PDG_SEMILEPTONIC_BR:
            exp_mu, exp_e = PDG_SEMILEPTONIC_BR[abs_c]
            print(f"  PDG World Average Reference      : BR(-> mu X) ~ {exp_mu}%, BR(-> e X) ~ {exp_e}%")

        # Check for forced channel anomaly
        if top_frac >= 80.0:
            any_forced_detected = True
            print(f"  >>> STATUS: [WARNING] FORCED EXCLUSIVE CHANNEL DETECTED! <<<")
            print(f"      {top_frac:4.1f}% of all {sp_name} decays are forced into '{top_formula}'.")
            if n_mu == 0:
                print(f"      [CRITICAL] Semi-muonic decays (-> mu) are COMPLETELY ABSENT (0.00%)!")
        else:
            print(f"  >>> STATUS: [OK] Multiple realistic branching modes observed.")

    print("\n" + "=" * 80)
    print(" OVERALL DIAGNOSTIC VERDICT")
    print("=" * 80)
    if any_forced_detected:
        print(" [CONFIRMED WARNING]: This Monte Carlo production appears to have FORCED")
        print(" single exclusive hadronic decay channels enabled for charm hadrons.")
        print(" Ground-state charm hadrons (D0, D+, Ds, Lambda_c) are decaying almost exclusively")
        print(" into their single most probable hadronic mode (e.g. D0 -> K- pi+ pi0, D+ -> K- pi+ pi+).")
        print(" Consequently, prompt semi-leptonic charm decays (such as charm -> mu) are MISSING")
        print(" in this dataset, explaining why no charm decay muons are generated!")
    else:
        print(" [OK]: No forced decay anomaly detected. Decay channels appear distributed.")
    print("=" * 80)

    # Detailed Decay Products Table
    print("\n" + "=" * 80)
    print(" TOP DIRECT DECAY PRODUCTS PER CHARM SPECIES")
    print("=" * 80)

    # Combined Table
    tot_all_d = sum(all_counts.values())
    print(f"\n[All Charm Combined] (Total charm: {total_charm_hadrons}, Total decay products: {tot_all_d})")
    print(f" {'Particle':<18s} {'PDG':<8s} {'Count':<8s} {'Fraction':<12s} {'Mean Multiplicity':<18s}")
    print("-" * 80)
    for d_pdg in sorted_d_pdgs[:12]:
        cnt = all_counts[d_pdg]
        frac = 100.0 * cnt / max(tot_all_d, 1)
        mult = cnt / max(total_charm_hadrons, 1)
        name = PDG_NAMES.get(d_pdg, f"PDG_{d_pdg}")
        print(f" {name:<18s} {d_pdg:<8d} {cnt:<8d} {frac:6.2f}%      {mult:6.3f} / charm")

    # Individual Species Table
    for abs_c in sorted(species_counts.keys()):
        sp_name = get_species_name(abs_c)
        c_dict = species_counts[abs_c]
        tot_sp_d = sum(c_dict.values())
        n_hadrons = species_hadron_counts.get(abs_c, 0)
        sorted_sp_d = sorted(c_dict.items(), key=lambda x: x[1], reverse=True)

        print(f"\n[{sp_name}] (|PDG|={abs_c}, Charm count: {n_hadrons}, Total decay products: {tot_sp_d})")
        print(f" {'Particle':<18s} {'PDG':<8s} {'Count':<8s} {'Fraction':<12s} {'Mean Multiplicity':<18s}")
        print("-" * 80)
        for d_pdg, cnt in sorted_sp_d[:8]:
            frac = 100.0 * cnt / max(tot_sp_d, 1)
            mult = cnt / max(n_hadrons, 1)
            name = PDG_NAMES.get(d_pdg, f"PDG_{d_pdg}")
            print(f" {name:<18s} {d_pdg:<8d} {cnt:<8d} {frac:6.2f}%      {mult:6.3f} / hadron")

    print("\n" + "=" * 80)
    print(" Diagnostic completed successfully!")
    print(f" Output ROOT file: {os.path.abspath(args.output)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
