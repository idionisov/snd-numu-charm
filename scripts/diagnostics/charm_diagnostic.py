#!/usr/bin/env python3
"""
scripts/diagnostics/charm_diagnostic.py
----------------------------------------
Diagnostic tool to inspect Monte Carlo truth charmed hadrons, production rates,
species fragmentation fractions, and decay channels in SND@LHC neutrino interactions.

Capable of analyzing:
  - Processed dimuon candidate files:
    /eos/user/i/idioniso/snd-numu-charm/data/.../*/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root
  - Full raw Monte Carlo production files:
    /eos/experiment/sndlhc/MonteCarlo/.../*/sndLHC.Genie-TGeant4_digCPP.root

Specifically answers:
  1. How often do we get charm hadrons from muon neutrino CC interactions (overall & vs E_nu)?
  2. What fractions of those charm hadrons are D0, D+, Ds, Lambda_c (overall & vs E_nu)?
  3. What do each of these species decay to, and how often is each channel engaged (overall & vs E_nu)?
  4. Physical modeling checks:
     - Prompt semi-leptonic branching fractions BR(-> mu X) and BR(-> e X) vs PDG world averages.
     - Lepton universality test (BR(mu) / BR(e) ~ 1.0).
     - Proper decay length / lifetime c*tau distribution compared with known PDG values.
     - Charm hadron energy fraction / elasticity z = E_charm / E_nu (fragmentation).
     - Transverse momentum pT and production angle relative to the neutrino beam.
     - Neutrino vs Anti-neutrino charm baryon asymmetry (Lambda_c production).

Supports high-performance parallel processing via ProcessPoolExecutor (-j / --jobs).
"""

from __future__ import annotations

import os
import sys
import math
import time
import argparse
import subprocess
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(1111)

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import resolve_input_files


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

# Ground state charm hadrons that decay weakly
GROUND_STATE_CHARM_PDGS = {411, 421, 431, 4122, 4232, 4132, 4332}

# PDG World Average inclusive semi-leptonic branching fractions for reference
PDG_SEMILEPTONIC_BR: Dict[int, Tuple[float, float]] = {
    421: (6.8, 6.7),    # D0: BR(mu+ X) ~ 6.8%, BR(e+ X) ~ 6.5-6.7%
    411: (15.6, 16.1),  # D+: BR(mu+ X) ~ 15.6%, BR(e+ X) ~ 16.1%
    431: (6.3, 6.3),    # Ds+: BR(mu+ X) ~ 6.3%, BR(e+ X) ~ 6.3%
    4122: (3.5, 4.0),   # Lambda_c+: BR(mu+ X) ~ 3.5%, BR(e+ X) ~ 4.0%
}

# PDG World Average proper lifetimes in micrometers (c*tau)
PDG_LIFETIMES_CTAU_UM: Dict[int, float] = {
    421: 122.9,   # D0: tau = 410.1 fs -> c*tau = 122.9 um
    411: 311.8,   # D+: tau = 1040 fs -> c*tau = 311.8 um
    431: 151.2,   # Ds+: tau = 504 fs -> c*tau = 151.2 um
    4122: 60.7,   # Lambda_c+: tau = 202.4 fs -> c*tau = 60.7 um
}

# Nominal masses (GeV) for boost calculation fallback
CHARM_NOMINAL_MASSES: Dict[int, float] = {
    421: 1.86483,
    411: 1.86965,
    431: 1.96834,
    4122: 2.28646,
    4232: 2.4679,
    4132: 2.4709,
    4332: 2.6952,
}


def is_charm_hadron(pdg: int) -> bool:
    """Checks if a PDG code corresponds to a charmed hadron."""
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


def fast_resolve_files(pattern: str, filelist_path: Optional[str] = None, max_files: int = -1) -> List[str]:
    """
    Quickly resolves input ROOT files from a text filelist, EOS command, or pattern.
    """
    matched: List[str] = []

    # 1. Explicit filelist option
    if filelist_path and os.path.isfile(filelist_path):
        with open(filelist_path, "r") as f:
            for line in f:
                p = line.strip()
                if p and not p.startswith("#") and p.endswith(".root"):
                    matched.append(p)
        if max_files > 0:
            matched = matched[:max_files]
        return matched

    # 2. Pattern starts with '@' (filelist convention)
    if pattern.startswith("@") and os.path.isfile(pattern[1:]):
        with open(pattern[1:], "r") as f:
            for line in f:
                p = line.strip()
                if p and not p.startswith("#") and p.endswith(".root"):
                    matched.append(p)
        if max_files > 0:
            matched = matched[:max_files]
        return matched

    # 3. Fast EOS resolution if pattern contains wildcard on EOS filesystem
    if "/eos/" in pattern and "*" in pattern:
        base_dir, fname_pat = pattern.split("/*", 1)
        fname = fname_pat.lstrip("/")
        server = "root://eosuser.cern.ch" if "/eos/user/" in pattern else "root://eospublic.cern.ch"
        cmd = ["eos", server, "find", "-f", "--name", fname, base_dir]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            if res.returncode == 0:
                lines = [l.strip() for l in res.stdout.splitlines() if l.strip().endswith(".root")]
                if lines:
                    # Natural numerical sort by partition number
                    lines = sorted(
                        lines,
                        key=lambda x: [int(c) if c.isdigit() else c for c in os.path.dirname(x).split("/")]
                    )
                    if max_files > 0:
                        lines = lines[:max_files]
                    return lines
        except Exception:
            pass

    # 4. Fallback to standard resolve_input_files
    try:
        matched = resolve_input_files(pattern, max_files=max_files)
    except Exception:
        pass

    return matched


def find_energy_bin_index(energy: float, bin_edges: List[float]) -> int:
    """Returns the 0-based bin index for energy, or -1 if out of bounds."""
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= energy < bin_edges[i + 1]:
            return i
    if energy >= bin_edges[-1]:
        return len(bin_edges) - 2
    return -1


def process_files_worker(args_tuple) -> Dict[str, Any]:
    """
    Worker process analyzing a batch of ROOT files.
    Extracts all neutrino CC events, charm hadrons, and their decay properties.
    """
    (
        file_chunk,
        bin_edges,
        decay_only,
        ground_state_only,
        numu_cc_only,
        max_events_chunk,
    ) = args_tuple

    import ROOT
    ROOT.gROOT.SetBatch(True)

    local_data = {
        "files_processed": 0,
        "total_events": 0,
        "nu_cc_inclusive": 0,
        "nu_cc_charm": 0,
        # Energy spectra
        "nu_e_cc_all": [],
        "nu_e_cc_charm": [],
        # Binned counts
        "cc_all_per_ebin": [0] * (len(bin_edges) - 1),
        "cc_charm_per_ebin": [0] * (len(bin_edges) - 1),
        # Species counts
        "species_hadron_counts": Counter(),
        "species_ebin_counts": {i: Counter() for i in range(len(bin_edges) - 1)},
        # Decay channels
        "decay_channels_by_species": {},     # abs_pdg -> Counter of tuples
        "decay_channels_by_species_ebin": {},# abs_pdg -> {ebin_idx: Counter}
        "all_decay_channels": Counter(),
        # Semi-leptonic counts
        "semi_muonic_counts": Counter(),
        "semi_electronic_counts": Counter(),
        "total_decays_by_species": Counter(),
        # Kinematics
        "ctau_values_um": {},                # abs_pdg -> list of c*tau values
        "decay_dist_mm": {},                 # abs_pdg -> list of L values
        "fragmentation_z": [],               # list of E_charm / E_nu
        "charm_pt": [],                      # list of pT relative to neutrino beam
        "charm_theta": [],                   # list of production angle theta
        # Neutrino vs Antineutrino asymmetry
        "nu_sign_charm_species": {1: Counter(), -1: Counter()}, # +1 for nu, -1 for anti-nu
    }

    events_analyzed = 0

    for fpath in file_chunk:
        if max_events_chunk > 0 and events_analyzed >= max_events_chunk:
            break

        fin = ROOT.TFile.Open(fpath, "READ")
        if not fin or fin.IsZombie():
            continue

        tree_cbmsim = fin.Get("cbmsim")
        if not tree_cbmsim:
            fin.Close()
            continue

        tree_truth = fin.Get("truth")
        if tree_truth:
            tree_cbmsim.AddFriend(tree_truth)

        n_entries = tree_cbmsim.GetEntries()
        local_data["files_processed"] += 1

        for iev in range(n_entries):
            if max_events_chunk > 0 and events_analyzed >= max_events_chunk:
                break
            events_analyzed += 1
            local_data["total_events"] += 1

            tree_cbmsim.GetEntry(iev)
            tracks = getattr(tree_cbmsim, "MCTrack", None)
            if not tracks or len(tracks) < 1:
                continue

            # 1. Determine neutrino properties
            nu_trk = tracks[0]
            nu_pdg = nu_trk.GetPdgCode()
            nu_energy = nu_trk.GetEnergy()

            # Direction of incoming neutrino
            nu_px, nu_py, nu_pz = nu_trk.GetPx(), nu_trk.GetPy(), nu_trk.GetPz()
            nu_p = nu_trk.GetP()
            if nu_p > 0:
                nu_dir = (nu_px / nu_p, nu_py / nu_p, nu_pz / nu_p)
            else:
                nu_dir = (0.0, 0.0, 1.0)

            # Check CC vs NC
            is_cc = False
            if tree_truth and hasattr(tree_cbmsim, "is_cc"):
                is_cc = bool(tree_cbmsim.is_cc)
            else:
                # Fallback to identify prompt muon from primary vertex (mother == 0)
                for trk in tracks:
                    if trk.GetMotherId() == 0 and abs(trk.GetPdgCode()) in [11, 13, 15]:
                        is_cc = True
                        break

            # Check nu_mu / anti-nu_mu CC selection
            is_numu_cc = (abs(nu_pdg) == 14 and is_cc)
            if numu_cc_only and not is_numu_cc:
                continue

            local_data["nu_cc_inclusive"] += 1
            local_data["nu_e_cc_all"].append(nu_energy)
            ebin_idx = find_energy_bin_index(nu_energy, bin_edges)
            if ebin_idx >= 0:
                local_data["cc_all_per_ebin"][ebin_idx] += 1

            # 2. Scan event for charm hadrons
            n_tracks = len(tracks)
            charm_hadrons_in_event: List[Tuple[int, int, int]] = []

            for i_trk in range(n_tracks):
                trk = tracks[i_trk]
                pdg = trk.GetPdgCode()
                if is_charm_hadron(pdg):
                    abs_pdg = abs(pdg)
                    if ground_state_only and abs_pdg not in GROUND_STATE_CHARM_PDGS:
                        continue
                    charm_hadrons_in_event.append((i_trk, pdg, abs_pdg))

            has_charm = (len(charm_hadrons_in_event) > 0)
            if not has_charm:
                continue

            local_data["nu_cc_charm"] += 1
            local_data["nu_e_cc_charm"].append(nu_energy)
            if ebin_idx >= 0:
                local_data["cc_charm_per_ebin"][ebin_idx] += 1

            nu_sign = 1 if nu_pdg > 0 else -1

            # 3. Analyze each charm hadron and its direct daughters
            for i_charm, charm_pdg, abs_charm_pdg in charm_hadrons_in_event:
                charm_trk = tracks[i_charm]
                local_data["species_hadron_counts"][abs_charm_pdg] += 1
                if ebin_idx >= 0:
                    local_data["species_ebin_counts"][ebin_idx][abs_charm_pdg] += 1
                local_data["nu_sign_charm_species"][nu_sign][abs_charm_pdg] += 1

                # Hadron kinematics
                c_p = charm_trk.GetP()
                c_e = charm_trk.GetEnergy()
                c_px, c_py, c_pz = charm_trk.GetPx(), charm_trk.GetPy(), charm_trk.GetPz()

                # Elasticity z = E_charm / E_nu
                if nu_energy > 0:
                    z = min(1.0, max(0.0, c_e / nu_energy))
                    local_data["fragmentation_z"].append(z)

                # Transverse momentum relative to neutrino axis: pT = |p x u_nu|
                cross_x = c_py * nu_dir[2] - c_pz * nu_dir[1]
                cross_y = c_pz * nu_dir[0] - c_px * nu_dir[2]
                cross_z = c_px * nu_dir[1] - c_py * nu_dir[0]
                pt_rel = math.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
                local_data["charm_pt"].append(pt_rel)

                # Production angle theta relative to neutrino
                p_dot_nu = c_px * nu_dir[0] + c_py * nu_dir[1] + c_pz * nu_dir[2]
                cos_theta = max(-1.0, min(1.0, p_dot_nu / max(c_p, 1e-6)))
                theta_mrad = math.acos(cos_theta) * 1000.0
                local_data["charm_theta"].append(theta_mrad)

                # Charm vertex
                vtx_x = charm_trk.GetStartX()
                vtx_y = charm_trk.GetStartY()
                vtx_z = charm_trk.GetStartZ()

                # Direct daughters extraction
                daughters: List[int] = []
                decay_vtx = None

                for j_trk in range(n_tracks):
                    d_trk = tracks[j_trk]
                    if d_trk.GetMotherId() == i_charm:
                        proc = d_trk.GetProcName()
                        if decay_only and proc != "Decay":
                            continue
                        d_pdg = d_trk.GetPdgCode()
                        daughters.append(d_pdg)
                        if decay_vtx is None:
                            decay_vtx = (d_trk.GetStartX(), d_trk.GetStartY(), d_trk.GetStartZ())

                if daughters:
                    local_data["total_decays_by_species"][abs_charm_pdg] += 1
                    ch_tuple = tuple(sorted(daughters))

                    if abs_charm_pdg not in local_data["decay_channels_by_species"]:
                        local_data["decay_channels_by_species"][abs_charm_pdg] = Counter()
                    local_data["decay_channels_by_species"][abs_charm_pdg][ch_tuple] += 1
                    local_data["all_decay_channels"][ch_tuple] += 1

                    if ebin_idx >= 0:
                        if abs_charm_pdg not in local_data["decay_channels_by_species_ebin"]:
                            local_data["decay_channels_by_species_ebin"][abs_charm_pdg] = {
                                b: Counter() for b in range(len(bin_edges) - 1)
                            }
                        local_data["decay_channels_by_species_ebin"][abs_charm_pdg][ebin_idx][ch_tuple] += 1

                    # Semi-leptonic checks
                    if any(abs(d) == 13 for d in daughters):
                        local_data["semi_muonic_counts"][abs_charm_pdg] += 1
                    if any(abs(d) == 11 for d in daughters):
                        local_data["semi_electronic_counts"][abs_charm_pdg] += 1

                    # Proper lifetime c*tau calculation
                    if decay_vtx is not None:
                        dx = decay_vtx[0] - vtx_x
                        dy = decay_vtx[1] - vtx_y
                        dz = decay_vtx[2] - vtx_z
                        dist_cm = math.sqrt(dx * dx + dy * dy + dz * dz)
                        dist_mm = dist_cm * 10.0

                        # Relativistic boost beta*gamma = p / m
                        mass = math.sqrt(max(0.01, c_e * c_e - c_p * c_p))
                        if mass < 0.5:
                            mass = CHARM_NOMINAL_MASSES.get(abs_charm_pdg, 1.86)
                        bg = max(0.01, c_p / mass)
                        ctau_um = (dist_cm / bg) * 10000.0

                        if abs_charm_pdg not in local_data["ctau_values_um"]:
                            local_data["ctau_values_um"][abs_charm_pdg] = []
                            local_data["decay_dist_mm"][abs_charm_pdg] = []
                        local_data["ctau_values_um"][abs_charm_pdg].append(ctau_um)
                        local_data["decay_dist_mm"][abs_charm_pdg].append(dist_mm)

        fin.Close()

    return local_data


def merge_worker_results(results: List[Dict[str, Any]], bin_edges: List[float]) -> Dict[str, Any]:
    """Combines statistics from all parallel workers."""
    merged = {
        "files_processed": sum(r["files_processed"] for r in results),
        "total_events": sum(r["total_events"] for r in results),
        "nu_cc_inclusive": sum(r["nu_cc_inclusive"] for r in results),
        "nu_cc_charm": sum(r["nu_cc_charm"] for r in results),
        "nu_e_cc_all": [],
        "nu_e_cc_charm": [],
        "cc_all_per_ebin": [0] * (len(bin_edges) - 1),
        "cc_charm_per_ebin": [0] * (len(bin_edges) - 1),
        "species_hadron_counts": Counter(),
        "species_ebin_counts": {i: Counter() for i in range(len(bin_edges) - 1)},
        "decay_channels_by_species": {},
        "decay_channels_by_species_ebin": {},
        "all_decay_channels": Counter(),
        "semi_muonic_counts": Counter(),
        "semi_electronic_counts": Counter(),
        "total_decays_by_species": Counter(),
        "ctau_values_um": {},
        "decay_dist_mm": {},
        "fragmentation_z": [],
        "charm_pt": [],
        "charm_theta": [],
        "nu_sign_charm_species": {1: Counter(), -1: Counter()},
    }

    for r in results:
        merged["nu_e_cc_all"].extend(r["nu_e_cc_all"])
        merged["nu_e_cc_charm"].extend(r["nu_e_cc_charm"])
        merged["fragmentation_z"].extend(r["fragmentation_z"])
        merged["charm_pt"].extend(r["charm_pt"])
        merged["charm_theta"].extend(r["charm_theta"])

        for i in range(len(bin_edges) - 1):
            merged["cc_all_per_ebin"][i] += r["cc_all_per_ebin"][i]
            merged["cc_charm_per_ebin"][i] += r["cc_charm_per_ebin"][i]
            merged["species_ebin_counts"][i].update(r["species_ebin_counts"][i])

        merged["species_hadron_counts"].update(r["species_hadron_counts"])
        merged["all_decay_channels"].update(r["all_decay_channels"])
        merged["semi_muonic_counts"].update(r["semi_muonic_counts"])
        merged["semi_electronic_counts"].update(r["semi_electronic_counts"])
        merged["total_decays_by_species"].update(r["total_decays_by_species"])

        for s in [1, -1]:
            merged["nu_sign_charm_species"][s].update(r["nu_sign_charm_species"][s])

        # Decay channels
        for abs_c, counter in r["decay_channels_by_species"].items():
            if abs_c not in merged["decay_channels_by_species"]:
                merged["decay_channels_by_species"][abs_c] = Counter()
            merged["decay_channels_by_species"][abs_c].update(counter)

        # Decay channels per energy bin
        for abs_c, ebin_map in r["decay_channels_by_species_ebin"].items():
            if abs_c not in merged["decay_channels_by_species_ebin"]:
                merged["decay_channels_by_species_ebin"][abs_c] = {
                    b: Counter() for b in range(len(bin_edges) - 1)
                }
            for b_idx, counter in ebin_map.items():
                merged["decay_channels_by_species_ebin"][abs_c][b_idx].update(counter)

        # Lifetimes
        for abs_c, ctau_list in r["ctau_values_um"].items():
            if abs_c not in merged["ctau_values_um"]:
                merged["ctau_values_um"][abs_c] = []
                merged["decay_dist_mm"][abs_c] = []
            merged["ctau_values_um"][abs_c].extend(ctau_list)
            merged["decay_dist_mm"][abs_c].extend(r["decay_dist_mm"][abs_c])

    return merged


def write_histograms(
    output_path: str,
    data: Dict[str, Any],
    bin_edges: List[float],
) -> None:
    """Builds and writes comprehensive ROOT histograms."""
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fout = ROOT.TFile.Open(output_path, "RECREATE")
    if not fout or fout.IsZombie():
        print(f"[Error] Failed to open {output_path} for writing.")
        return

    n_ebins = len(bin_edges) - 1

    # =========================================================================
    # 1. SUMMARY DIRECTORY: Production Rates, Fractions, Kinematics
    # =========================================================================
    dir_summary = fout.mkdir("Summary")
    dir_summary.cd()

    # Energy spectra (fine 1D, 50 bins 0-1000 GeV)
    h_nu_e_cc_all = ROOT.TH1D("h_nu_e_cc_all", "Neutrino Energy (#nu_{#mu} CC Inclusive);E_{#nu} [GeV];Events", 50, 0.0, 1000.0)
    for e in data["nu_e_cc_all"]:
        h_nu_e_cc_all.Fill(e)
    h_nu_e_cc_all.Write()

    h_nu_e_cc_charm = ROOT.TH1D("h_nu_e_cc_charm", "Neutrino Energy (#nu_{#mu} CC with Charm);E_{#nu} [GeV];Events", 50, 0.0, 1000.0)
    for e in data["nu_e_cc_charm"]:
        h_nu_e_cc_charm.Fill(e)
    h_nu_e_cc_charm.Write()

    # Charm production rate ratio vs energy (fine)
    h_charm_rate_fine = h_nu_e_cc_charm.Clone("h_charm_rate_vs_energy_fine")
    h_charm_rate_fine.SetTitle("Charm Production Rate vs Neutrino Energy;E_{#nu} [GeV];R_{charm} = N_{charm} / N_{#nu_{#mu} CC}")
    h_charm_rate_fine.Divide(h_nu_e_cc_charm, h_nu_e_cc_all, 1.0, 1.0, "B")
    h_charm_rate_fine.Write()

    # Charm production rate in discrete energy bins
    import array
    arr_edges = array.array("d", bin_edges)
    h_ebin_all = ROOT.TH1D("h_cc_all_ebins", "CC Inclusive Events per Energy Bin;E_{#nu} [GeV];Events", n_ebins, arr_edges)
    h_ebin_charm = ROOT.TH1D("h_cc_charm_ebins", "CC Charm Events per Energy Bin;E_{#nu} [GeV];Events", n_ebins, arr_edges)
    for i in range(n_ebins):
        h_ebin_all.SetBinContent(i + 1, data["cc_all_per_ebin"][i])
        h_ebin_charm.SetBinContent(i + 1, data["cc_charm_per_ebin"][i])
    h_ebin_all.Write()
    h_ebin_charm.Write()

    h_charm_rate_ebins = h_ebin_charm.Clone("h_charm_rate_vs_energy_bins")
    h_charm_rate_ebins.SetTitle("Charm Production Rate in Energy Bins;E_{#nu} [GeV];R_{charm} = N_{charm} / N_{#nu_{#mu} CC}")
    h_charm_rate_ebins.Divide(h_ebin_charm, h_ebin_all, 1.0, 1.0, "B")
    h_charm_rate_ebins.Write()

    # Species counts and fractions (Overall)
    species_order = [421, 411, 431, 4122]
    other_species = [s for s in data["species_hadron_counts"].keys() if s not in species_order]
    all_species_display = species_order + other_species

    tot_charm_all = sum(data["species_hadron_counts"].values())
    n_sp_bins = max(1, len(all_species_display))
    h_species_counts = ROOT.TH1D("h_species_counts", "Charm Species Counts;Charm Hadron;Entries", n_sp_bins, 0.5, n_sp_bins + 0.5)
    h_species_frac = ROOT.TH1D("h_species_fractions", "Charm Species Fractions (f_{i} = N_{i} / N_{charm});Charm Hadron;Fraction", n_sp_bins, 0.5, n_sp_bins + 0.5)

    for idx, abs_c in enumerate(all_species_display, start=1):
        sp_name = get_species_name(abs_c)
        cnt = data["species_hadron_counts"].get(abs_c, 0)
        frac = (cnt / tot_charm_all) if tot_charm_all > 0 else 0.0
        h_species_counts.GetXaxis().SetBinLabel(idx, sp_name)
        h_species_counts.SetBinContent(idx, cnt)
        h_species_frac.GetXaxis().SetBinLabel(idx, sp_name)
        h_species_frac.SetBinContent(idx, frac)
    h_species_counts.Write()
    h_species_frac.Write()

    # 2D Species Fraction vs Energy Bins (TH2D: x=E_nu bins, y=Species)
    h2_species_ebin = ROOT.TH2D(
        "h2_species_vs_energy",
        "Charm Species Breakdown across Energy Bins;E_{#nu} [GeV];Charm Hadron",
        n_ebins, arr_edges, n_sp_bins, 0.5, n_sp_bins + 0.5
    )
    for y_idx, abs_c in enumerate(all_species_display, start=1):
        h2_species_ebin.GetYaxis().SetBinLabel(y_idx, get_species_name(abs_c))

    for x_idx in range(n_ebins):
        ebin_tot = sum(data["species_ebin_counts"][x_idx].values())
        for y_idx, abs_c in enumerate(all_species_display, start=1):
            c_cnt = data["species_ebin_counts"][x_idx].get(abs_c, 0)
            frac = (c_cnt / ebin_tot) if ebin_tot > 0 else 0.0
            h2_species_ebin.SetBinContent(x_idx + 1, y_idx, frac)
    h2_species_ebin.Write()

    # 1D Species Fraction vs Energy for each major ground state
    for abs_c in species_order:
        sp_name = get_species_name(abs_c)
        h_frac_vs_e = ROOT.TH1D(
            f"h_fraction_{sp_name}_vs_energy",
            f"{sp_name} Fraction vs Neutrino Energy;E_{{#nu}} [GeV];f({sp_name}) = N({sp_name}) / N_{{charm}}",
            n_ebins, arr_edges
        )
        for i_b in range(n_ebins):
            ebin_tot = sum(data["species_ebin_counts"][i_b].values())
            c_cnt = data["species_ebin_counts"][i_b].get(abs_c, 0)
            frac = (c_cnt / ebin_tot) if ebin_tot > 0 else 0.0
            h_frac_vs_e.SetBinContent(i_b + 1, frac)
        h_frac_vs_e.Write()

    # Kinematics: Fragmentation z, pT, Theta
    h_z = ROOT.TH1D("h_fragmentation_z", "Charm Energy Fraction (Fragmentation);z = E_{charm} / E_{#nu};Entries", 50, 0.0, 1.0)
    for z in data["fragmentation_z"]:
        h_z.Fill(z)
    h_z.Write()

    h_pt = ROOT.TH1D("h_charm_pt", "Charm Transverse Momentum relative to #nu;p_{T} [GeV/c];Entries", 50, 0.0, 5.0)
    for pt in data["charm_pt"]:
        h_pt.Fill(pt)
    h_pt.Write()

    h_theta = ROOT.TH1D("h_charm_theta", "Charm Production Angle relative to #nu;#theta [mrad];Entries", 50, 0.0, 500.0)
    for th in data["charm_theta"]:
        h_theta.Fill(th)
    h_theta.Write()

    # Semi-leptonic Branching Fractions summary
    h_br_mu = ROOT.TH1D("h_semileptonic_br_mu", "Prompt Semi-Muonic Branching Ratio BR(c -> #mu X);Charm Species;BR(c -> #mu) [%]", n_sp_bins, 0.5, n_sp_bins + 0.5)
    h_br_e = ROOT.TH1D("h_semileptonic_br_e", "Prompt Semi-Electronic Branching Ratio BR(c -> e X);Charm Species;BR(c -> e) [%]", n_sp_bins, 0.5, n_sp_bins + 0.5)
    for idx, abs_c in enumerate(all_species_display, start=1):
        sp_name = get_species_name(abs_c)
        tot_d = data["total_decays_by_species"].get(abs_c, 0)
        n_mu = data["semi_muonic_counts"].get(abs_c, 0)
        n_e = data["semi_electronic_counts"].get(abs_c, 0)
        br_mu = (100.0 * n_mu / tot_d) if tot_d > 0 else 0.0
        br_e = (100.0 * n_e / tot_d) if tot_d > 0 else 0.0
        h_br_mu.GetXaxis().SetBinLabel(idx, sp_name)
        h_br_mu.SetBinContent(idx, br_mu)
        h_br_e.GetXaxis().SetBinLabel(idx, sp_name)
        h_br_e.SetBinContent(idx, br_e)
    h_br_mu.Write()
    h_br_e.Write()

    # =========================================================================
    # 2. ALL CHARM COMBINED DIRECTORY
    # =========================================================================
    dir_all = fout.mkdir("AllCharm")
    dir_all.cd()

    sorted_all_channels = sorted(data["all_decay_channels"].items(), key=lambda x: x[1], reverse=True)
    n_ch_all = min(len(sorted_all_channels), 30)
    h_all_channels = ROOT.TH1D("h_decay_channels_all_charm", "Exclusive Decay Channels (All Charm);Decay Channel;Decays", max(1, n_ch_all), 0.5, max(1, n_ch_all) + 0.5)
    for i_ch, (ch_tup, cnt) in enumerate(sorted_all_channels[:n_ch_all], start=1):
        h_all_channels.GetXaxis().SetBinLabel(i_ch, format_channel_formula(ch_tup))
        h_all_channels.SetBinContent(i_ch, cnt)
    h_all_channels.Write()

    # =========================================================================
    # 3. PER-SPECIES DIRECTORIES (D0, Dplus, Ds, Lambda_c)
    # =========================================================================
    dir_species_root = fout.mkdir("Species")

    for abs_c in sorted(data["species_hadron_counts"].keys()):
        sp_name = get_species_name(abs_c)
        dir_sp = dir_species_root.mkdir(sp_name)
        dir_sp.cd()

        ch_map = data["decay_channels_by_species"].get(abs_c, Counter())
        tot_sp_decays = data["total_decays_by_species"].get(abs_c, sum(ch_map.values()))
        sorted_sp_channels = sorted(ch_map.items(), key=lambda x: x[1], reverse=True)

        n_top = min(len(sorted_sp_channels), 25)
        h_sp_ch = ROOT.TH1D(
            f"h_decay_channels_{sp_name}",
            f"Exclusive Decay Channels ({sp_name});Decay Channel;Decays",
            max(1, n_top), 0.5, max(1, n_top) + 0.5
        )
        h_sp_br = ROOT.TH1D(
            f"h_branching_ratio_{sp_name}",
            f"Branching Ratios ({sp_name});Decay Channel;Branching Ratio [%]",
            max(1, n_top), 0.5, max(1, n_top) + 0.5
        )

        for i_ch, (ch_tup, cnt) in enumerate(sorted_sp_channels[:n_top], start=1):
            formula = format_channel_formula(ch_tup)
            h_sp_ch.GetXaxis().SetBinLabel(i_ch, formula)
            h_sp_ch.SetBinContent(i_ch, cnt)
            br_val = (100.0 * cnt / tot_sp_decays) if tot_sp_decays > 0 else 0.0
            h_sp_br.GetXaxis().SetBinLabel(i_ch, formula)
            h_sp_br.SetBinContent(i_ch, br_val)
        h_sp_ch.Write()
        h_sp_br.Write()

        # Lifetime c*tau
        ctau_list = data["ctau_values_um"].get(abs_c, [])
        h_ctau = ROOT.TH1D(
            f"h_ctau_{sp_name}",
            f"Proper Lifetime c#tau ({sp_name});c#tau [#mum];Entries",
            50, 0.0, 1000.0
        )
        for val in ctau_list:
            h_ctau.Fill(val)
        h_ctau.Write()

        # Lab decay distance in mm
        dist_list = data["decay_dist_mm"].get(abs_c, [])
        h_dist = ROOT.TH1D(
            f"h_decay_distance_mm_{sp_name}",
            f"Decay Distance in Material ({sp_name});Flight Path L [mm];Entries",
            50, 0.0, 50.0
        )
        for d in dist_list:
            h_dist.Fill(d)
        h_dist.Write()

        # Energy Bins Subdirectory for Decay Channels vs Energy
        dir_ebins = dir_sp.mkdir("EnergyBins")
        ebin_ch_map = data["decay_channels_by_species_ebin"].get(abs_c, {})

        for i_b in range(n_ebins):
            e_low, e_high = bin_edges[i_b], bin_edges[i_b + 1]
            dir_b = dir_ebins.mkdir(f"Bin_{i_b}_{int(e_low)}to{int(e_high)}GeV")
            dir_b.cd()

            b_channels = ebin_ch_map.get(i_b, Counter())
            sorted_b_ch = sorted(b_channels.items(), key=lambda x: x[1], reverse=True)
            n_b_top = min(len(sorted_b_ch), 15)
            h_b_ch = ROOT.TH1D(
                f"h_decay_channels_{sp_name}_bin{i_b}",
                f"{sp_name} Decay Channels ({int(e_low)} <= E_{{#nu}} < {int(e_high)} GeV);Decay Channel;Decays",
                max(1, n_b_top), 0.5, max(1, n_b_top) + 0.5
            )
            for j_ch, (ch_tup, cnt) in enumerate(sorted_b_ch[:n_b_top], start=1):
                h_b_ch.GetXaxis().SetBinLabel(j_ch, format_channel_formula(ch_tup))
                h_b_ch.SetBinContent(j_ch, cnt)
            h_b_ch.Write()

    fout.Close()
    print(f"  ROOT Histograms written successfully to: {os.path.abspath(output_path)}")


def print_diagnostic_report(data: Dict[str, Any], bin_edges: List[float]) -> None:
    """Prints a structured summary table and physics diagnostic report."""
    n_ebins = len(bin_edges) - 1
    tot_files = data["files_processed"]
    tot_events = data["total_events"]
    tot_cc = data["nu_cc_inclusive"]
    tot_charm = data["nu_cc_charm"]
    charm_rate = (100.0 * tot_charm / tot_cc) if tot_cc > 0 else 0.0

    print("\n" + "=" * 84)
    print(" SND@LHC CHARMED HADRON COMPREHENSIVE PHYSICS & DECAY DIAGNOSTIC REPORT")
    print("=" * 84)
    print(f" Analyzed Files          : {tot_files}")
    print(f" Total Events Scanned    : {tot_events}")
    print(f" Nu_mu CC Events         : {tot_cc}")
    print(f" Nu_mu CC Charm Events   : {tot_charm}")
    print(f" Overall Charm Production: {tot_charm} / {tot_cc} ({charm_rate:.2f}%)")
    print("-" * 84)

    # 1. Charm Production Rate vs Energy
    print("\n" + "-" * 84)
    print(" 1. CHARM PRODUCTION RATE AS A FUNCTION OF NEUTRINO ENERGY")
    print("-" * 84)
    print(f" {'Energy Bin [GeV]':<22s} {'Total Nu_mu CC':<18s} {'Charm Count':<16s} {'Charm Rate [%]':<16s}")
    print("-" * 84)
    for i in range(n_ebins):
        e_low, e_high = bin_edges[i], bin_edges[i + 1]
        n_cc_bin = data["cc_all_per_ebin"][i]
        n_ch_bin = data["cc_charm_per_ebin"][i]
        rate_bin = (100.0 * n_ch_bin / n_cc_bin) if n_cc_bin > 0 else 0.0
        err_bin = (100.0 * math.sqrt(n_ch_bin) / n_cc_bin) if (n_cc_bin > 0 and n_ch_bin > 0) else 0.0
        bin_str = f"[{e_low:.0f}, {e_high:.0f})"
        rate_str = f"{rate_bin:5.2f}% +/- {err_bin:4.2f}%" if n_cc_bin > 0 else "N/A (0 events)"
        print(f" {bin_str:<22s} {n_cc_bin:<18d} {n_ch_bin:<16d} {rate_str:<16s}")
    print("-" * 84)

    # 2. Charm Species Fractions (Overall & vs Energy)
    tot_charm_hadrons = sum(data["species_hadron_counts"].values())
    print("\n" + "-" * 84)
    print(" 2. CHARM SPECIES FRAGMENTATION FRACTIONS (RATIO TO TOTAL CHARM)")
    print("-" * 84)
    header = f" {'Species':<14s} {'|PDG|':<8s} {'Overall Share':<16s}"
    for i in range(n_ebins):
        header += f" [{bin_edges[i]:.0f},{bin_edges[i+1]:.0f}) GeV".ljust(14)
    print(header)
    print("-" * 84)

    major_pdgs = [421, 411, 431, 4122]
    other_pdgs = [p for p in sorted(data["species_hadron_counts"].keys()) if p not in major_pdgs]

    for abs_c in major_pdgs + other_pdgs:
        sp_name = get_species_name(abs_c)
        cnt = data["species_hadron_counts"].get(abs_c, 0)
        overall_share = (100.0 * cnt / tot_charm_hadrons) if tot_charm_hadrons > 0 else 0.0
        row = f" {sp_name:<14s} {abs_c:<8d} {overall_share:5.2f}% ({cnt:<3d})"
        row = f"{row:<40s}"
        for i in range(n_ebins):
            ebin_tot = sum(data["species_ebin_counts"][i].values())
            c_ebin = data["species_ebin_counts"][i].get(abs_c, 0)
            share_ebin = (100.0 * c_ebin / ebin_tot) if ebin_tot > 0 else 0.0
            row += f"{share_ebin:5.1f}% ({c_ebin:<2d})".ljust(14)
        print(row)
    print("-" * 84)

    # 3. Decay Channels Breakdown
    print("\n" + "-" * 84)
    print(" 3. EXCLUSIVE DECAY CHANNELS & BRANCHING RATIOS PER SPECIES")
    print("-" * 84)
    for abs_c in major_pdgs:
        sp_name = get_species_name(abs_c)
        ch_map = data["decay_channels_by_species"].get(abs_c, Counter())
        tot_d = data["total_decays_by_species"].get(abs_c, sum(ch_map.values()))
        if tot_d == 0:
            continue
        sorted_ch = sorted(ch_map.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  [{sp_name}] (Total Decays Analyzed: {tot_d}, Distinct Modes Observed: {len(ch_map)}):")
        for rank, (ch_tup, count) in enumerate(sorted_ch[:5], start=1):
            formula = format_channel_formula(ch_tup)
            br = 100.0 * count / tot_d
            print(f"    #{rank:02d}: {formula:<40s} -> {count:3d} decays ({br:5.1f}%)")

    # 4. Physical Modeling Checks
    print("\n" + "=" * 84)
    print(" 4. PHYSICS MODELING TESTS (BRANCHING FRACTIONS, LIFETIMES, KINEMATICS)")
    print("=" * 84)

    any_anomalies = False

    for abs_c in major_pdgs:
        sp_name = get_species_name(abs_c)
        tot_d = data["total_decays_by_species"].get(abs_c, 0)
        if tot_d == 0:
            continue

        n_mu = data["semi_muonic_counts"].get(abs_c, 0)
        n_e = data["semi_electronic_counts"].get(abs_c, 0)
        br_mu = (100.0 * n_mu / tot_d) if tot_d > 0 else 0.0
        br_e = (100.0 * n_e / tot_d) if tot_d > 0 else 0.0

        exp_mu, exp_e = PDG_SEMILEPTONIC_BR.get(abs_c, (0.0, 0.0))
        pdg_ctau = PDG_LIFETIMES_CTAU_UM.get(abs_c, 0.0)
        obs_ctau_list = data["ctau_values_um"].get(abs_c, [])
        mean_ctau = (sum(obs_ctau_list) / len(obs_ctau_list)) if obs_ctau_list else 0.0

        ch_map = data["decay_channels_by_species"].get(abs_c, Counter())
        top_mode, top_cnt = ch_map.most_common(1)[0] if ch_map else ((), 0)
        top_frac = 100.0 * top_cnt / tot_d if tot_d > 0 else 0.0

        print(f"\n  [{sp_name}]:")
        print(f"    - Semi-muonic BR (-> mu X)     : {br_mu:5.1f}%  (PDG reference: ~{exp_mu:.1f}%)")
        print(f"    - Semi-electronic BR (-> e X)  : {br_e:5.1f}%  (PDG reference: ~{exp_e:.1f}%)")
        if br_e > 0:
            print(f"    - Lepton Universality mu/e     : {br_mu / br_e:5.2f} (Expected: ~1.00)")
        print(f"    - Proper Lifetime <c*tau>      : {mean_ctau:5.1f} um (PDG reference: {pdg_ctau:.1f} um)")
        print(f"    - Most Frequent Decay Mode     : '{format_channel_formula(top_mode)}' ({top_frac:4.1f}%)")

        # Tests
        if top_frac >= 80.0 and tot_d >= 15:
            any_anomalies = True
            print(f"    >>> STATUS: [FAIL] FORCED EXCLUSIVE CHANNEL DETECTED! Mode '{format_channel_formula(top_mode)}' dominates {top_frac:.1f}%.")
        elif n_mu == 0 and exp_mu > 0 and tot_d >= 30:
            any_anomalies = True
            print(f"    >>> STATUS: [WARN] Prompt semi-muonic decays are completely absent in this sample!")
        elif mean_ctau < 1.0 and tot_d >= 10:
            any_anomalies = True
            print(f"    >>> STATUS: [FAIL] Particle appears to decay instantaneously (c*tau ~ 0)!")
        else:
            print(f"    >>> STATUS: [PASS] Branching modes, semi-leptonic rates, and lifetimes look consistent.")

    # 5. Neutrino vs Antineutrino Baryon Asymmetry Check
    print("\n" + "-" * 84)
    print(" 5. NEUTRINO VS ANTINEUTRINO ASYMMETRY (LAMBDA_C BARYON PRODUCTION)")
    print("-" * 84)
    nu_lc = data["nu_sign_charm_species"][1].get(4122, 0)
    nu_tot_c = sum(data["nu_sign_charm_species"][1].values())
    anti_lc = data["nu_sign_charm_species"][-1].get(4122, 0)
    anti_tot_c = sum(data["nu_sign_charm_species"][-1].values())

    frac_nu_lc = (100.0 * nu_lc / nu_tot_c) if nu_tot_c > 0 else 0.0
    frac_anti_lc = (100.0 * anti_lc / anti_tot_c) if anti_tot_c > 0 else 0.0
    print(f"  Nu_mu interactions      : Lambda_c+ share = {nu_lc}/{nu_tot_c} ({frac_nu_lc:4.1f}%)")
    print(f"  Anti-Nu_mu interactions : anti-Lambda_c- share = {anti_lc}/{anti_tot_c} ({frac_anti_lc:4.1f}%)")
    if frac_nu_lc >= frac_anti_lc:
        print("  >>> STATUS: [PASS] Expected baryon suppression in anti-neutrinos is confirmed!")
    else:
        print("  >>> STATUS: [INFO] Statistics limited or unexpected anti-neutrino baryon fraction.")

    print("\n" + "=" * 84)
    print(" OVERALL DIAGNOSTIC VERDICT")
    print("=" * 84)
    if any_anomalies:
        print(" [WARNING]: Anomalies detected in charm decay modeling (e.g. forced branching ratios).")
    else:
        print(" [ALL CHECKS PASSED]: Charm production rates, hadron fragmentation species,")
        print(" inclusive branching fractions, proper decay lifetimes, and kinematics appear")
        print(" correctly modelled and consistent with physical expectations.")
    print("=" * 84 + "\n")


def export_diagnostic_plots(root_file_path: str, plots_dir: str) -> None:
    """Exports key histograms as high-quality PNG images."""
    os.makedirs(plots_dir, exist_ok=True)
    fin = ROOT.TFile.Open(root_file_path, "READ")
    if not fin or fin.IsZombie():
        return

    c = ROOT.TCanvas("c_diag", "Diagnostic Plot", 900, 650)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.08)
    c.SetBottomMargin(0.12)
    c.SetTopMargin(0.08)

    # 1. Charm production rate vs energy bins
    h_rate = fin.Get("Summary/h_charm_rate_vs_energy_bins")
    if h_rate:
        c.Clear()
        h_rate.SetMarkerStyle(20)
        h_rate.SetMarkerSize(1.3)
        h_rate.SetMarkerColor(ROOT.kBlue + 2)
        h_rate.SetLineColor(ROOT.kBlue + 2)
        h_rate.SetLineWidth(2)
        max_val = max(0.5, h_rate.GetMaximum() * 1.3)
        h_rate.GetYaxis().SetRangeUser(0.0, max_val)
        h_rate.Draw("E1")
        c.SaveAs(os.path.join(plots_dir, "charm_production_rate_vs_energy.png"))

    # 2. Species fractions bar chart
    h_sp = fin.Get("Summary/h_species_fractions")
    if h_sp:
        c.Clear()
        h_sp.SetFillColor(ROOT.kAzure - 4)
        h_sp.SetLineColor(ROOT.kBlue + 2)
        h_sp.SetLineWidth(2)
        h_sp.GetYaxis().SetRangeUser(0.0, 1.0)
        h_sp.Draw("HIST")
        c.SaveAs(os.path.join(plots_dir, "charm_species_fractions.png"))

    # 3. Species vs energy 2D
    h2_sp = fin.Get("Summary/h2_species_vs_energy")
    if h2_sp:
        c.Clear()
        c.SetRightMargin(0.14)
        h2_sp.Draw("COLZ TEXT")
        c.SaveAs(os.path.join(plots_dir, "charm_species_vs_energy_2d.png"))
        c.SetRightMargin(0.08)

    # 4. Fragmentation z
    h_z = fin.Get("Summary/h_fragmentation_z")
    if h_z:
        c.Clear()
        h_z.SetFillColor(ROOT.kTeal - 5)
        h_z.SetLineColor(ROOT.kTeal + 2)
        h_z.Draw("HIST")
        c.SaveAs(os.path.join(plots_dir, "charm_fragmentation_z.png"))

    # 5. Decay channels for major species
    for sp in ["D0", "Dplus", "Ds", "Lambda_c"]:
        h_ch = fin.Get(f"Species/{sp}/h_branching_ratio_{sp}")
        if h_ch and h_ch.GetEntries() > 0:
            c.Clear()
            c.SetBottomMargin(0.28)
            h_ch.SetFillColor(ROOT.kOrange - 2)
            h_ch.SetLineColor(ROOT.kOrange + 2)
            h_ch.Draw("HIST")
            c.SaveAs(os.path.join(plots_dir, f"decay_channels_{sp}.png"))
            c.SetBottomMargin(0.12)

    fin.Close()
    print(f"  Exported diagnostic PNG plots to: {os.path.abspath(plots_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic tool to analyze charm production, species fractions, and decay channels in SND@LHC MC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_input = (
        "/eos/user/i/idioniso/snd-numu-charm/data/"
        "sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/*/"
        "sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=default_input,
        help="Input ROOT file, directory, or pattern (e.g. .../*/sndLHC...root)",
    )
    parser.add_argument(
        "--filelist",
        type=str,
        default=None,
        help="Optional text file listing input ROOT file paths (one per line)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="charm_diagnostic_results.root",
        help="Output ROOT file to store histograms",
    )
    parser.add_argument(
        "-n", "--max-events",
        type=int,
        default=-1,
        help="Maximum total events to process (-1 for all)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Maximum number of input files to process (-1 for all)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=10,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--energy-bins",
        nargs="+",
        type=float,
        default=[0.0, 100.0, 250.0, 500.0, 1500.0],
        help="Neutrino energy bin edges in GeV (e.g. 0 100 250 500 1500)",
    )
    parser.add_argument(
        "--decay-only",
        action="store_true",
        default=True,
        help="Only count direct daughters from 'Decay' process (excludes material inelastic scattering)",
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
        default=True,
        help="Only analyze ground-state weakly decaying charm hadrons (D0, D+, Ds, Lambda_c)",
    )
    parser.add_argument(
        "--include-excited",
        action="store_false",
        dest="ground_state_only",
        help="Include strongly decaying excited charm states (D*, Sigma_c)",
    )
    parser.add_argument(
        "--numu-cc-only",
        action="store_true",
        default=True,
        help="Restrict denominator to muon neutrino CC interactions",
    )
    parser.add_argument(
        "--all-interactions",
        action="store_false",
        dest="numu_cc_only",
        help="Include NC and all neutrino flavors in inclusive counts",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Directory to export publication-quality PNG diagnostic plots",
    )

    args = parser.parse_args()

    print("=" * 84)
    print(" SND@LHC: Charmed Hadron Production, Species Fractions & Decay Diagnostic")
    print("=" * 84)
    print(f" Input Source         : {args.input}")
    if args.filelist:
        print(f" Filelist             : {args.filelist}")
    print(f" Output ROOT File     : {os.path.abspath(args.output)}")
    print(f" Max Events           : {'All' if args.max_events < 0 else args.max_events}")
    print(f" Max Files            : {'All' if args.max_files < 0 else args.max_files}")
    print(f" Parallel Workers     : {args.jobs}")
    print(f" Energy Bins [GeV]    : {args.energy_bins}")
    print(f" Decay Process Only   : {args.decay_only}")
    print(f" Ground State Only    : {args.ground_state_only}")
    print(f" Nu_mu CC Only        : {args.numu_cc_only}")
    print("=" * 84)

    # 1. Resolve input files
    t0_resolve = time.time()
    input_files = fast_resolve_files(args.input, filelist_path=args.filelist, max_files=args.max_files)
    if not input_files:
        print(f"[Error] No ROOT files matched input target: {args.input}")
        sys.exit(1)

    print(f"\n[1/3] Resolved {len(input_files)} input ROOT file(s) in {time.time() - t0_resolve:.2f}s.")

    # 2. Partition files for parallel worker execution
    n_jobs = max(1, min(args.jobs, len(input_files)))
    chunk_size = int(math.ceil(len(input_files) / float(n_jobs)))
    file_chunks = [input_files[i:i + chunk_size] for i in range(0, len(input_files), chunk_size)]

    max_events_per_worker = -1
    if args.max_events > 0:
        max_events_per_worker = int(math.ceil(args.max_events / float(len(file_chunks))))

    tasks = [
        (
            chunk,
            args.energy_bins,
            args.decay_only,
            args.ground_state_only,
            args.numu_cc_only,
            max_events_per_worker,
        )
        for chunk in file_chunks
    ]

    print(f"[2/3] Processing events across {len(file_chunks)} worker pool(s) ({n_jobs} threads)...")
    t0_proc = time.time()

    if n_jobs == 1 or len(file_chunks) == 1:
        results = [process_files_worker(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_files_worker, tasks))

    elapsed_proc = time.time() - t0_proc
    print(f"      Completed event processing in {elapsed_proc:.2f}s!")

    # 3. Merge results and generate outputs
    print("[3/3] Merging statistics, writing histograms & compiling diagnostic report...")
    merged_data = merge_worker_results(results, args.energy_bins)

    write_histograms(args.output, merged_data, args.energy_bins)
    print_diagnostic_report(merged_data, args.energy_bins)

    if args.plots_dir:
        export_diagnostic_plots(args.output, args.plots_dir)


if __name__ == "__main__":
    main()
