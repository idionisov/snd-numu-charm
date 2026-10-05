#!/usr/bin/env python3
"""
filter_signal_numu_charm.py
---------------------------
Hierarchical neutrino interaction skimmer and truth analysis pipeline for SND@LHC.

Configuration is fully specified via YAML:
- Level 1: Neutrino flavor (any, numu, nue, nutau)
  └── Level 2: Interaction current (any, CC, NC)
        └── Level 3: Charmed hadron production (require: true/false, species: any/D0/D+/Ds/Lambda_c)
              └── Level 4: Charm decay channel (decay: any/to_muon, require_opposite_sign: true/false)

Truth TTree Modular Tiers:
- Universal: Neutrino kinematics, interaction flags, DIS variables, vertex coordinates.
- Lepton: Primary outgoing charged lepton (mu1 for nu_mu CC) kinematics.
- Charm: Charmed hadron 4-momentum, decay length, lifetimes, species, and decay vertex.
- Decay Muon: Prompt muon (mu2) from charm decay kinematics, pTrel, and impact parameters.
- Dimuon System: Composite pair invariant mass, opening angles, azimuthal delta-phi, and asymmetry.
Only the branches corresponding to the active hierarchy tiers are stored in the TTree (in "auto" mode).

CLI arguments are deliberately kept minimal:
  -n, --entries   : Max events per file (-1 for all)
  -j, --threads   : Worker threads
  -o, --output    : Output directory or file
  -c, --config    : Path to YAML configuration file
  --fiducial      : Require interaction vertex in Target fiducial volume
"""

from __future__ import annotations

import os
import sys
import glob
import array
import argparse
import ROOT

# Add repo root to python path to import snd package
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import DataManager
from snd.data_manager import load_trident_libraries

# Pre-load libraries so ROOT has all custom C++ dictionaries and classes available
load_trident_libraries(_repo_root)

# Disable ROOT GUI windows in batch mode
ROOT.gROOT.SetBatch(True)


def parse_arguments():
    """Parse minimal command line arguments."""
    parser = argparse.ArgumentParser(
        description="Skim neutrino events with hierarchical truth TTree generation from SND@LHC MC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-n", "--entries",
        type=int,
        default=-1,
        help="Max events per file to process (-1 for all events)"
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        default=1,
        help="Number of worker threads"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (or base filename) for filtered files"
    )
    parser.add_argument(
        "--fiducial",
        action="store_true",
        default=False,
        help="Require interaction vertex within the Target fiducial volume"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=os.path.join(_repo_root, "config", "filter_numu_charm_config.yaml"),
        help="Path to YAML configuration file"
    )
    return parser.parse_args()


def load_config(config_path: str = None) -> dict:
    """Load analysis configuration from YAML (or JSON fallback)."""
    if config_path is None:
        config_path = os.path.join(_repo_root, "config", "filter_numu_charm_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        import json
        with open(config_path, "r") as f:
            cfg = json.load(f)

    return cfg


def resolve_input_files(pattern: str, max_files: int = -1) -> list:
    """
    Resolve input file pattern into an ordered list of existing files.
    Optimized for multi-directory wildcards (e.g. .../*/filename) over network mounts.
    """
    print(f"Resolving input files from pattern:\n  {pattern}")
    matched_files = []

    if "/*/" in pattern and pattern.count("*") == 1:
        base_dir, filename = pattern.split("/*/")
        if os.path.isdir(base_dir):
            subdirs = sorted(
                [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
                key=lambda x: int(x) if x.isdigit() else x
            )
            for d in subdirs:
                if max_files > 0 and len(matched_files) >= max_files:
                    break
                fpath = os.path.join(base_dir, d, filename)
                if os.path.exists(fpath):
                    matched_files.append(fpath)

    if not matched_files:
        matched_files = sorted(glob.glob(pattern))

    if not matched_files:
        raise FileNotFoundError(
            f"No files matched input pattern: {pattern}\n"
            "Please check the path and your EOS credentials."
        )

    if max_files > 0:
        matched_files = matched_files[:max_files]

    print(f"Found {len(matched_files)} matching ROOT file(s) to process.")
    return matched_files


def build_processor(proc_cfg: dict):
    """Build and configure the MuonNeutrinoTruthProcessor from config dictionary."""
    config = ROOT.snd.NeutrinoTruthConfig()
    config.targetZMin = float(proc_cfg.get("target_z_min", 260.0))
    config.targetZMax = float(proc_cfg.get("target_z_max", 360.0))
    config.fidXMin = float(proc_cfg.get("fid_x_min", -47.5))
    config.fidXMax = float(proc_cfg.get("fid_x_max", -8.5))
    config.fidYMin = float(proc_cfg.get("fid_y_min", 15.5))
    config.fidYMax = float(proc_cfg.get("fid_y_max", 54.5))
    config.fiducialMargin = float(proc_cfg.get("fiducial_margin", 1.5))
    config.minLeptonMomentum = float(proc_cfg.get("min_lepton_momentum", 0.5))
    config.maxCharmFlightDistance = float(proc_cfg.get("max_charm_flight_distance", 15.0))
    config.weightScale = float(proc_cfg.get("weight_scale", 1.0))
    return ROOT.snd.MuonNeutrinoTruthProcessor(config)


def resolve_hierarchical_selection(cfg: dict, cli_fiducial: bool):
    """
    Parse the hierarchical selection options from the config file:
      flavor -> interaction -> charm (require, species, decay, require_opposite_sign)
    Returns:
      filter_expr     : RDataFrame C++ boolean filter expression
      description     : Human-readable summary
      predicate       : Python callable predicate(info) -> bool for in-loop verification
      active_tiers    : Set of active truth branch tiers: {'universal', 'lepton', 'charm', 'decay_muon', 'dimuon'}
    """
    sel_cfg = cfg.get("selection", {})
    custom_filter = sel_cfg.get("custom_filter", "").strip()

    flavor = str(sel_cfg.get("flavor", "any")).lower().replace("_", "").replace("-", "")
    interaction = str(sel_cfg.get("interaction", "any")).upper().strip()
    charm_cfg = sel_cfg.get("charm", {})
    require_charm = bool(charm_cfg.get("require", False))
    charm_species = str(charm_cfg.get("species", "any")).lower().strip()
    charm_decay = str(charm_cfg.get("decay", "any")).lower().replace("-", "").replace("_", "")
    require_os = bool(charm_cfg.get("require_opposite_sign", True))
    require_fiducial = cli_fiducial or bool(sel_cfg.get("require_fiducial", False))

    clauses = []
    desc_parts = []
    active_tiers = {"universal"}

    # 1. Custom filter override
    if custom_filter:
        active_tiers.update(["lepton", "charm", "decay_muon", "dimuon"])
        return custom_filter, f"Custom filter: {custom_filter}", (lambda info: True), active_tiers

    # 2. Flavor selection (Level 1)
    if flavor in ["numu", "muon"]:
        clauses.append("std::abs(nu_pdg) == 14")
        desc_parts.append("nu_mu (14)")
    elif flavor in ["nue", "electron"]:
        clauses.append("std::abs(nu_pdg) == 12")
        desc_parts.append("nu_e (12)")
    elif flavor in ["nutau", "tau"]:
        clauses.append("std::abs(nu_pdg) == 16")
        desc_parts.append("nu_tau (16)")
    else:
        desc_parts.append("All neutrino flavors")

    # 3. Interaction selection (Level 2)
    if interaction == "CC":
        if flavor in ["numu", "muon"]:
            clauses.append("(is_numu_cc || is_anti_numu_cc)")
        else:
            clauses.append("is_cc")
        desc_parts.append("CC")
        active_tiers.add("lepton")
    elif interaction == "NC":
        clauses.append("is_nc")
        desc_parts.append("NC")
    else:
        desc_parts.append("CC+NC")
        active_tiers.add("lepton")

    # 4. Charm selection (Level 3 & 4)
    if require_charm:
        clauses.append("has_charm")
        active_tiers.add("charm")

        # Species constraint
        if charm_species in ["d0", "421"]:
            clauses.append("abs_charm_pdg == 421")
            desc_parts.append("D0")
        elif charm_species in ["dplus", "d+", "411"]:
            clauses.append("abs_charm_pdg == 411")
            desc_parts.append("D+")
        elif charm_species in ["ds", "dsplus", "431"]:
            clauses.append("abs_charm_pdg == 431")
            desc_parts.append("Ds+")
        elif charm_species in ["lambdac", "4122"]:
            clauses.append("abs_charm_pdg == 4122")
            desc_parts.append("Lambda_c+")
        else:
            desc_parts.append("charmed hadron")

        # Decay channel (Level 4)
        if charm_decay in ["tomuon", "muon", "dimuon", "decaymuon"]:
            active_tiers.add("decay_muon")
            if interaction == "CC" and flavor in ["numu", "muon"]:
                active_tiers.add("dimuon")
                if require_os:
                    clauses.append("has_candidate && is_opposite_sign")
                    desc_parts.append("decay to prompt muon (opposite-sign dimuon candidate)")
                else:
                    clauses.append("has_candidate")
                    desc_parts.append("decay to prompt muon (dimuon candidate)")
            else:
                clauses.append("has_prompt_charm_muon")
                desc_parts.append("decay to prompt muon")
        else:
            desc_parts.append("inclusive decay")

    # 5. Fiducial requirement
    if require_fiducial:
        clauses.append("is_fiducial")
        desc_parts.append("[Target Fiducial]")

    # Construct overall C++ expression
    if clauses:
        filter_expr = " && ".join(f"({c})" for c in clauses)
    else:
        filter_expr = "1"

    description = " -> ".join(desc_parts)

    # Construct strict Python validation predicate
    def predicate(info) -> bool:
        # Flavor check
        if flavor in ["numu", "muon"] and abs(info.nuPdg) != 14:
            return False
        if flavor in ["nue", "electron"] and abs(info.nuPdg) != 12:
            return False
        if flavor in ["nutau", "tau"] and abs(info.nuPdg) != 16:
            return False

        # Current check
        if interaction == "CC":
            if flavor in ["numu", "muon"] and not (info.isNuMuCC or info.isAntiNuMuCC):
                return False
            elif not info.isCC:
                return False
        elif interaction == "NC" and not info.isNC:
            return False

        # Charm check
        if require_charm:
            if not info.hasCharm:
                return False
            if charm_species in ["d0", "421"] and abs(info.charmPdg) != 421:
                return False
            if charm_species in ["dplus", "d+", "411"] and abs(info.charmPdg) != 411:
                return False
            if charm_species in ["ds", "dsplus", "431"] and abs(info.charmPdg) != 431:
                return False
            if charm_species in ["lambdac", "4122"] and abs(info.charmPdg) != 4122:
                return False

            if charm_decay in ["tomuon", "muon", "dimuon", "decaymuon"]:
                if interaction == "CC" and flavor in ["numu", "muon"]:
                    if not info.hasCandidate:
                        return False
                    if require_os and not info.isOppositeSignDimuon:
                        return False
                elif not info.hasPromptCharmMuon:
                    return False

        # Fiducial check
        if require_fiducial and not info.isFiducial:
            return False

        return True

    # Check truth_tree branch mode in config: "auto" vs "all"
    tree_branch_mode = str(cfg.get("truth_tree", {}).get("branch_mode", "auto")).lower().strip()
    if tree_branch_mode == "all":
        active_tiers = {"universal", "lepton", "charm", "decay_muon", "dimuon"}

    return filter_expr, description, predicate, active_tiers


def determine_output_path(input_path: str, output_arg: str, output_cfg: dict, index: int, total_files: int) -> str:
    """Determine the output file path corresponding to an input file."""
    file_suffix = output_cfg.get("file_suffix", "_signal")
    preserve_subdirs = output_cfg.get("preserve_subdirs", True)
    default_dir = output_cfg.get("output_dir", "output_signal_filtered")

    dirname, filename = os.path.split(input_path)
    parent_partition = os.path.basename(dirname)
    base_stem, ext = os.path.splitext(filename)

    if output_arg:
        if output_arg.endswith(".root"):
            if total_files == 1:
                return output_arg
            out_stem, out_ext = os.path.splitext(output_arg)
            partition_tag = parent_partition if (parent_partition and parent_partition != ".") else str(index + 1)
            return f"{out_stem}_{partition_tag}{out_ext}"
        else:
            base_out_dir = output_arg
    else:
        base_out_dir = default_dir

    if preserve_subdirs and parent_partition and parent_partition != ".":
        out_file = os.path.join(base_out_dir, parent_partition, f"{base_stem}{file_suffix}{ext}")
    else:
        out_file = os.path.join(base_out_dir, f"{parent_partition}_{base_stem}{file_suffix}{ext}")

    return out_file


def setup_truth_branches(trees: list[ROOT.TTree], active_tiers: set[str]) -> dict:
    """
    Attach modular truth branches to one or more TTrees based on active_tiers:
      - 'universal'  : Neutrino 4-momentum, DIS kinematics, primary vertex, weights, flags
      - 'lepton'     : Primary outgoing charged lepton (mu1 for nu_mu CC) kinematics
      - 'charm'      : Charmed hadron properties, flight lengths, proper lifetime, decay vertex
      - 'decay_muon' : Prompt decay muon (mu2) from charm kinematics and impact parameters
      - 'dimuon'     : Composite dimuon pair observables (mass, opening angles, delta phi)
    """
    int_vars = []
    double_vars = []

    # 1. Universal Tier
    if "universal" in active_tiers:
        int_vars.extend([
            "entry", "nu_pdg",
            "is_cc", "is_nc", "is_numu_cc", "is_anti_numu_cc",
            "interaction_type", "region_type", "is_fiducial",
            "n_primary_tracks", "n_primary_hadrons"
        ])
        double_vars.extend([
            "mc_weight", "raw_weight",
            "nu_e", "nu_p", "nu_px", "nu_py", "nu_pz", "nu_pt", "nu_eta", "nu_phi", "nu_theta",
            "vtx_x", "vtx_y", "vtx_z", "vtx_t",
            "q2", "bjorken_x", "inelasticity_y", "hadronic_w",
            "hadronic_e_total", "hadronic_pt", "missing_pt"
        ])

    # 2. Primary Lepton Tier
    if "lepton" in active_tiers:
        int_vars.extend([
            "primary_lepton_track_id", "primary_lepton_pdg", "primary_lepton_charge"
        ])
        double_vars.extend([
            "mu1_p", "mu1_pt", "mu1_px", "mu1_py", "mu1_pz", "mu1_e", "mu1_eta", "mu1_phi", "mu1_theta",
            "mu1_slope_xz", "mu1_slope_yz"
        ])

    # 3. Charmed Hadron Tier
    if "charm" in active_tiers:
        int_vars.extend([
            "has_charm", "charm_track_id", "charm_pdg", "abs_charm_pdg", "charm_species_id",
            "charm_quark_content", "n_charm_daughters", "charm_has_kaon", "charm_kaon_pdg",
            "n_charmed_hadrons"
        ])
        double_vars.extend([
            "charm_mass", "charm_p", "charm_pt", "charm_px", "charm_py", "charm_pz", "charm_e",
            "charm_eta", "charm_phi", "charm_theta", "charm_z_frac",
            "decay_length_3d", "decay_length_xy", "decay_length_z", "proper_time_ctau", "proper_lifetime_ps",
            "charm_decay_x", "charm_decay_y", "charm_decay_z", "charm_decay_t"
        ])

    # 4. Decay Muon Tier
    if "decay_muon" in active_tiers:
        int_vars.extend([
            "has_prompt_charm_muon", "mu2_track_id", "mu2_pdg", "mu2_charge",
            "mu2_mother_track_id", "mu2_mother_pdg"
        ])
        double_vars.extend([
            "mu2_p", "mu2_pt", "mu2_px", "mu2_py", "mu2_pz", "mu2_e", "mu2_eta", "mu2_phi", "mu2_theta",
            "mu2_slope_xz", "mu2_slope_yz",
            "mu2_ptrel", "mu2_ip3d", "mu2_ipxy", "mu2_opening_angle_charm"
        ])

    # 5. Composite Dimuon Tier
    if "dimuon" in active_tiers:
        int_vars.extend([
            "is_opposite_sign", "has_candidate", "n_muons_in_event"
        ])
        double_vars.extend([
            "dimuon_mass", "dimuon_pt", "dimuon_p", "dimuon_opening_angle",
            "dimuon_opening_angle_mrad", "dimuon_delta_phi", "dimuon_delta_eta", "dimuon_delta_r",
            "dimuon_energy_asym", "dimuon_p_ratio"
        ])

    buffers = {}
    for var in int_vars:
        buffers[var] = array.array('i', [0])
    for var in double_vars:
        buffers[var] = array.array('d', [0.0])

    for tree in trees:
        if not tree:
            continue
        for var in int_vars:
            tree.Branch(var, buffers[var], f"{var}/I")
        for var in double_vars:
            tree.Branch(var, buffers[var], f"{var}/D")

    return buffers


def fill_truth_buffers(buffers: dict, info: object, entry_idx: int, active_tiers: set[str]):
    """Populate truth buffers according to active tiers."""
    # 1. Universal Tier
    if "universal" in active_tiers:
        buffers["entry"][0] = int(entry_idx)
        buffers["nu_pdg"][0] = int(info.nuPdg)
        buffers["is_cc"][0] = int(info.isCC)
        buffers["is_nc"][0] = int(info.isNC)
        buffers["is_numu_cc"][0] = int(info.isNuMuCC)
        buffers["is_anti_numu_cc"][0] = int(info.isAntiNuMuCC)
        buffers["interaction_type"][0] = int(info.interactionType)
        buffers["region_type"][0] = int(info.regionType)
        buffers["is_fiducial"][0] = int(info.isFiducial)
        buffers["n_primary_tracks"][0] = int(info.nPrimaryTracks)
        buffers["n_primary_hadrons"][0] = int(info.nPrimaryHadrons)

        buffers["mc_weight"][0] = float(info.mcWeight)
        buffers["raw_weight"][0] = float(info.rawWeight)
        buffers["nu_e"][0] = float(info.nuE)
        buffers["nu_p"][0] = float(info.nuP)
        buffers["nu_px"][0] = float(info.nuPx)
        buffers["nu_py"][0] = float(info.nuPy)
        buffers["nu_pz"][0] = float(info.nuPz)
        buffers["nu_pt"][0] = float(info.nuPt)
        buffers["nu_eta"][0] = float(info.nuEta)
        buffers["nu_phi"][0] = float(info.nuPhi)
        buffers["nu_theta"][0] = float(info.nuTheta)

        buffers["vtx_x"][0] = float(info.vtxX)
        buffers["vtx_y"][0] = float(info.vtxY)
        buffers["vtx_z"][0] = float(info.vtxZ)
        buffers["vtx_t"][0] = float(info.vtxT)

        buffers["q2"][0] = float(info.Q2)
        buffers["bjorken_x"][0] = float(info.BjorkenX)
        buffers["inelasticity_y"][0] = float(info.InelasticityY)
        buffers["hadronic_w"][0] = float(info.HadronicW)
        buffers["hadronic_e_total"][0] = float(info.hadronicEnergyTotal)
        buffers["hadronic_pt"][0] = float(info.hadronicRecoilPt)
        buffers["missing_pt"][0] = float(info.missingPt)

    # 2. Lepton Tier
    if "lepton" in active_tiers:
        buffers["primary_lepton_track_id"][0] = int(info.primaryLeptonTrackId)
        buffers["primary_lepton_pdg"][0] = int(info.primaryLeptonPdg)
        buffers["primary_lepton_charge"][0] = int(info.primaryLeptonCharge)

        buffers["mu1_p"][0] = float(info.mu1P)
        buffers["mu1_pt"][0] = float(info.mu1Pt)
        buffers["mu1_px"][0] = float(info.mu1Px)
        buffers["mu1_py"][0] = float(info.mu1Py)
        buffers["mu1_pz"][0] = float(info.mu1Pz)
        buffers["mu1_e"][0] = float(info.mu1E)
        buffers["mu1_eta"][0] = float(info.mu1Eta)
        buffers["mu1_phi"][0] = float(info.mu1Phi)
        buffers["mu1_theta"][0] = float(info.mu1Theta)
        buffers["mu1_slope_xz"][0] = float(info.mu1SlopeXZ)
        buffers["mu1_slope_yz"][0] = float(info.mu1SlopeYZ)

    # 3. Charm Tier
    if "charm" in active_tiers:
        buffers["has_charm"][0] = int(info.hasCharm)
        buffers["charm_track_id"][0] = int(info.charmTrackId)
        buffers["charm_pdg"][0] = int(info.charmPdg)
        buffers["abs_charm_pdg"][0] = abs(int(info.charmPdg))
        buffers["charm_quark_content"][0] = int(info.charmQuarkContent)
        buffers["n_charm_daughters"][0] = int(info.nCharmDecayDaughters)
        buffers["charm_has_kaon"][0] = int(info.charmHasKaonDaughter)
        buffers["charm_kaon_pdg"][0] = int(info.charmKaonPdg)
        buffers["n_charmed_hadrons"][0] = int(info.nCharmedHadronsInEvent)

        pdg = abs(int(info.charmPdg))
        if pdg == 421: sp_id = 1
        elif pdg == 411: sp_id = 2
        elif pdg == 431: sp_id = 3
        elif pdg == 4122: sp_id = 4
        elif pdg > 4000: sp_id = 5
        elif pdg > 0: sp_id = 6
        else: sp_id = 0
        buffers["charm_species_id"][0] = sp_id

        buffers["charm_mass"][0] = float(info.charmMass)
        buffers["charm_p"][0] = float(info.charmP)
        buffers["charm_pt"][0] = float(info.charmPt)
        buffers["charm_px"][0] = float(info.charmPx)
        buffers["charm_py"][0] = float(info.charmPy)
        buffers["charm_pz"][0] = float(info.charmPz)
        buffers["charm_e"][0] = float(info.charmE)
        buffers["charm_eta"][0] = float(info.charmEta)
        buffers["charm_phi"][0] = float(info.charmPhi)
        buffers["charm_theta"][0] = float(info.charmTheta)
        buffers["charm_z_frac"][0] = float(info.charmEnergyFractionZ)

        buffers["decay_length_3d"][0] = float(info.decayLength3D)
        buffers["decay_length_xy"][0] = float(info.decayLengthXY)
        buffers["decay_length_z"][0] = float(info.decayLengthZ)
        buffers["proper_time_ctau"][0] = float(info.properDecayTimeCTau)
        buffers["proper_lifetime_ps"][0] = float(info.properLifetimePs)
        buffers["charm_decay_x"][0] = float(info.charmDecayX)
        buffers["charm_decay_y"][0] = float(info.charmDecayY)
        buffers["charm_decay_z"][0] = float(info.charmDecayZ)
        buffers["charm_decay_t"][0] = float(info.charmDecayT)

    # 4. Decay Muon Tier
    if "decay_muon" in active_tiers:
        buffers["has_prompt_charm_muon"][0] = int(info.hasPromptCharmMuon)
        buffers["mu2_track_id"][0] = int(info.mu2TrackId)
        buffers["mu2_pdg"][0] = int(info.mu2Pdg)
        buffers["mu2_charge"][0] = int(info.mu2Charge)
        buffers["mu2_mother_track_id"][0] = int(info.mu2MotherTrackId)
        buffers["mu2_mother_pdg"][0] = int(info.mu2MotherPdg)

        buffers["mu2_p"][0] = float(info.mu2P)
        buffers["mu2_pt"][0] = float(info.mu2Pt)
        buffers["mu2_px"][0] = float(info.mu2Px)
        buffers["mu2_py"][0] = float(info.mu2Py)
        buffers["mu2_pz"][0] = float(info.mu2Pz)
        buffers["mu2_e"][0] = float(info.mu2E)
        buffers["mu2_eta"][0] = float(info.mu2Eta)
        buffers["mu2_phi"][0] = float(info.mu2Phi)
        buffers["mu2_theta"][0] = float(info.mu2Theta)
        buffers["mu2_slope_xz"][0] = float(info.mu2SlopeXZ)
        buffers["mu2_slope_yz"][0] = float(info.mu2SlopeYZ)
        buffers["mu2_ptrel"][0] = float(info.mu2PtRel)
        buffers["mu2_ip3d"][0] = float(info.mu2IP3D)
        buffers["mu2_ipxy"][0] = float(info.mu2IPXY)
        buffers["mu2_opening_angle_charm"][0] = float(info.mu2OpeningAngleWithCharm)

    # 5. Dimuon Tier
    if "dimuon" in active_tiers:
        buffers["is_opposite_sign"][0] = int(info.isOppositeSignDimuon)
        buffers["has_candidate"][0] = int(info.hasCandidate)
        buffers["n_muons_in_event"][0] = int(info.nMuonsInEvent)

        buffers["dimuon_mass"][0] = float(info.dimuonInvMass)
        buffers["dimuon_pt"][0] = float(info.dimuonPt)
        buffers["dimuon_p"][0] = float(info.dimuonP)
        buffers["dimuon_opening_angle"][0] = float(info.dimuonOpeningAngle)
        buffers["dimuon_opening_angle_mrad"][0] = float(info.dimuonOpeningAngleMrad)
        buffers["dimuon_delta_phi"][0] = float(info.dimuonDeltaPhi)
        buffers["dimuon_delta_eta"][0] = float(info.dimuonDeltaEta)
        buffers["dimuon_delta_r"][0] = float(info.dimuonDeltaR)
        buffers["dimuon_energy_asym"][0] = float(info.dimuonEnergyAsymmetry)
        buffers["dimuon_p_ratio"][0] = float(info.dimuonMomentumRatio)


def process_single_file(
    input_file: str,
    output_file: str,
    processor: ROOT.snd.MuonNeutrinoTruthProcessor,
    cfg: dict,
    filter_expr: str,
    predicate: callable,
    active_tiers: set[str],
    max_entries: int = -1
) -> dict:
    """
    Process a single input file:
    1. Evaluates truth observables and finds matching entry numbers using sequential RDataFrame.
    2. If 0 events match, completely skips creating/saving the ROOT file.
    3. If events match, clones cbmsim and creates a dedicated flat `truth` TTree with active tiers.
    4. Applies in-loop validation guard before saving each entry.
    5. Saves diagnostic histograms and trees into output file.
    """
    tree_name = cfg.get("input", {}).get("tree_name", "cbmsim")
    skip_empty = cfg.get("output", {}).get("skip_empty_files", True)

    # Verify input file and tree existence
    f_test = ROOT.TFile.Open(input_file)
    if not f_test or f_test.IsZombie():
        print(f"  [Warning] Cannot open input file: {input_file}")
        return {"total": 0, "signal": 0, "status": "error_open"}

    actual_tree_name = tree_name
    if not f_test.Get(actual_tree_name):
        for alt_name in ["cbmsim", "rawConv"]:
            if f_test.Get(alt_name):
                actual_tree_name = alt_name
                break
    f_test.Close()

    # Disable implicit multi-threading per file so Take("rdfentry_") is 100% exact and deterministic
    ROOT.DisableImplicitMT()

    # 1. RDataFrame setup for fast truth processing and filtering
    df_raw = ROOT.RDataFrame(actual_tree_name, input_file)
    if max_entries > 0:
        df_raw = df_raw.Range(max_entries)

    # Define truth observables
    df_truth = (
        df_raw.Define("truth", processor, ["MCTrack"])
              .Define("nu_pdg", "truth.nuPdg")
              .Define("is_cc", "truth.isCC")
              .Define("is_nc", "truth.isNC")
              .Define("is_numu_cc", "truth.isNuMuCC")
              .Define("is_anti_numu_cc", "truth.isAntiNuMuCC")
              .Define("is_fiducial", "truth.isFiducial")
              .Define("has_charm", "truth.hasCharm")
              .Define("has_prompt_charm_muon", "truth.hasPromptCharmMuon")
              .Define("is_opposite_sign", "truth.isOppositeSignDimuon")
              .Define("has_candidate", "truth.hasCandidate")
              .Define("mc_weight", "truth.mcWeight")
              .Define("raw_weight", "truth.rawWeight")
              .Define("nu_e", "truth.nuE")
              .Define("vtx_x", "truth.vtxX")
              .Define("vtx_y", "truth.vtxY")
              .Define("vtx_z", "truth.vtxZ")
              .Define("q2", "truth.Q2")
              .Define("bjorken_x", "truth.BjorkenX")
              .Define("inelasticity_y", "truth.InelasticityY")
              .Define("hadronic_w", "truth.HadronicW")
              .Define("mu1_p", "truth.mu1P")
              .Define("mu1_pt", "truth.mu1Pt")
              .Define("charm_pdg", "truth.charmPdg")
              .Define("abs_charm_pdg", "std::abs(truth.charmPdg)")
              .Define("charm_p", "truth.charmP")
              .Define("charm_pt", "truth.charmPt")
              .Define("charm_decay_length", "truth.decayLength3D")
              .Define("decay_length_3d", "truth.decayLength3D")
              .Define("proper_lifetime_ps", "truth.properLifetimePs")
              .Define("charm_species_id", """
                  int pdg = std::abs(truth.charmPdg);
                  if (pdg == 421) return 1;
                  if (pdg == 411) return 2;
                  if (pdg == 431) return 3;
                  if (pdg == 4122) return 4;
                  if (pdg > 4000) return 5;
                  if (pdg > 0) return 6;
                  return 0;
              """)
              .Define("mu2_p", "truth.mu2P")
              .Define("mu2_pt", "truth.mu2Pt")
              .Define("mu2_ip3d", "truth.mu2IP3D")
              .Define("mu2_ptrel", "truth.mu2PtRel")
              .Define("dimuon_mass", "truth.dimuonInvMass")
              .Define("dimuon_pt", "truth.dimuonPt")
              .Define("dimuon_opening_angle_mrad", "truth.dimuonOpeningAngleMrad")
              .Define("dimuon_delta_phi", "truth.dimuonDeltaPhi")
    )

    df_filtered = df_truth.Filter(filter_expr, "Event Selection")

    # Book counters & signal entry list
    c_tot = df_truth.Count()
    c_sig = df_filtered.Count()
    signal_entries_rptr = df_filtered.Take["ULong64_t"]("rdfentry_")

    # Book diagnostic histograms on selected events
    booked_histograms = {}
    for h_cfg in cfg.get("histograms", []):
        h_name = h_cfg["name"]
        h_title = h_cfg.get("title", h_name)
        col = h_cfg["column"]
        w_col = h_cfg.get("weight")

        if "bin_edges" in h_cfg and h_cfg["bin_edges"]:
            edges = array.array("d", h_cfg["bin_edges"])
            model = ROOT.RDF.TH1DModel(h_name, h_title, len(edges) - 1, edges)
        else:
            nbins = int(h_cfg.get("bins", 50))
            xmin = float(h_cfg.get("xmin", 0.0))
            xmax = float(h_cfg.get("xmax", 1.0))
            model = ROOT.RDF.TH1DModel(h_name, h_title, nbins, xmin, xmax)

        if w_col:
            booked_histograms[h_name] = df_filtered.Histo1D(model, col, w_col)
        else:
            booked_histograms[h_name] = df_filtered.Histo1D(model, col)

    # Run RDF graph
    all_actions = [c_tot, c_sig, signal_entries_rptr]
    all_actions.extend(booked_histograms.values())
    ROOT.RDF.RunGraphs(all_actions)

    n_tot = c_tot.GetValue()
    n_sig = c_sig.GetValue()
    signal_entries = list(signal_entries_rptr.GetValue())

    # 2. DO NOT STORE EMPTY ROOT FILES
    if skip_empty and (n_sig == 0 or len(signal_entries) == 0):
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except OSError:
                pass
        return {
            "total": n_tot,
            "signal": 0,
            "status": "skipped_empty",
            "output_file": None
        }

    # 3. Create output file and save selected events
    f_in = ROOT.TFile.Open(input_file)
    t_in = f_in.Get(actual_tree_name)

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    f_out = ROOT.TFile.Open(output_file, "RECREATE")
    if not f_out or f_out.IsZombie():
        f_in.Close()
        return {"total": n_tot, "signal": n_sig, "status": "error_create_out", "output_file": None}

    # Clone empty event tree (retains all original branches: MCTrack, SciFi, MuFilter, etc.)
    out_tree = t_in.CloneTree(0)
    out_tree.SetName(actual_tree_name)

    # Create dedicated flat truth TTree
    truth_tree = ROOT.TTree("truth", "Hierarchical Truth Observables for Selected Events")

    # Attach truth branches to both out_tree (cbmsim) and truth_tree
    truth_buffers = setup_truth_branches([out_tree, truth_tree], active_tiers)

    # Fill selected entries with strict validation guard
    n_filled = 0
    for entry_idx in signal_entries:
        t_in.GetEntry(entry_idx)
        info = processor.processMuonNeutrino(t_in.MCTrack)

        # In-loop validation guard
        if not predicate(info):
            continue

        fill_truth_buffers(truth_buffers, info, entry_idx, active_tiers)
        out_tree.Fill()
        truth_tree.Fill()
        n_filled += 1

    # Check if any entries were actually filled
    if skip_empty and n_filled == 0:
        f_out.Close()
        f_in.Close()
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except OSError:
                pass
        return {
            "total": n_tot,
            "signal": 0,
            "status": "skipped_empty",
            "output_file": None
        }

    f_out.cd()
    out_tree.Write()
    truth_tree.Write()

    # Write diagnostic histograms
    for h_name, h_result in booked_histograms.items():
        h = h_result.GetValue()
        h.Write()

    f_out.Close()
    f_in.Close()

    return {
        "total": n_tot,
        "signal": n_filled,
        "status": "success",
        "output_file": output_file
    }


def main():
    args = parse_arguments()

    # 1. Load Configuration
    cfg = load_config(args.config)

    input_cfg = cfg.get("input", {})
    file_pattern = input_cfg.get(
        "file_pattern",
        "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/"
        "sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/*/sndLHC.Genie-TGeant4_digCPP.root"
    )
    max_files = input_cfg.get("max_files", -1)
    output_cfg = cfg.get("output", {})

    # 2. Resolve Selection Criteria & Active Truth Tiers from YAML Config
    filter_expr, filter_desc, predicate, active_tiers = resolve_hierarchical_selection(
        cfg=cfg,
        cli_fiducial=args.fiducial
    )

    # 3. Resolve Input Files
    input_files = resolve_input_files(file_pattern, max_files=max_files)
    total_files = len(input_files)

    # 4. Configure Processor
    proc_cfg = cfg.get("processor", {})
    processor = build_processor(proc_cfg)

    print("=" * 80)
    print(" SND@LHC: Hierarchical Neutrino Skimmer & Truth Tree Generator")
    print("=" * 80)
    print(f"Configuration file           : {args.config}")
    print(f"Selection hierarchy          : {filter_desc}")
    print(f"Filter expression            : {filter_expr}")
    print(f"Active truth TTree tiers     : {sorted(list(active_tiers))}")
    print(f"Total input files to process : {total_files}")
    print(f"Max events per file          : {args.entries if args.entries > 0 else 'All'}")
    print(f"Skip empty files             : {output_cfg.get('skip_empty_files', True)}")
    print("=" * 80)

    # 5. Process each file 1-to-1
    results = []
    tot_events_all = 0
    tot_signal_all = 0
    files_written = 0

    for idx, in_file in enumerate(input_files):
        out_file = determine_output_path(in_file, args.output, output_cfg, idx, total_files)
        print(f"\n[{idx + 1}/{total_files}] Processing:\n  Input : {in_file}")

        res = process_single_file(
            input_file=in_file,
            output_file=out_file,
            processor=processor,
            cfg=cfg,
            filter_expr=filter_expr,
            predicate=predicate,
            active_tiers=active_tiers,
            max_entries=args.entries
        )
        results.append(res)

        n_tot = res.get("total", 0)
        n_sig = res.get("signal", 0)
        tot_events_all += n_tot
        tot_signal_all += n_sig

        status = res.get("status", "unknown")
        if status == "skipped_empty":
            print(f"  Summary: 0/{n_tot} matched. Skipped (empty file not created).")
        elif status == "success":
            files_written += 1
            pct = (100.0 * n_sig / max(n_tot, 1)) if n_tot > 0 else 0.0
            print(f"  Summary: {n_sig}/{n_tot} events saved ({pct:.2f}%). Output: {out_file}")
        else:
            print(f"  Summary: Status={status}")

    # 6. Final Processing Summary Table
    print("\n" + "=" * 80)
    print(" OVERALL PROCESSING SUMMARY")
    print("=" * 80)
    print(f"{'#':<3} | {'Partition':<12} | {'Total':>8} | {'Selected':>8} | {'Yield (%)':>10} | {'Status':<14}")
    print("-" * 80)
    for idx, (in_file, res) in enumerate(zip(input_files, results)):
        partition = os.path.basename(os.path.dirname(in_file))
        n_tot = res.get("total", 0)
        n_sig = res.get("signal", 0)
        pct = (100.0 * n_sig / max(n_tot, 1)) if n_tot > 0 else 0.0
        status = res.get("status", "unknown")
        print(f"{idx + 1:<3} | {partition:<12} | {n_tot:8d} | {n_sig:8d} | {pct:9.2f}% | {status:<14}")

    print("=" * 80)
    overall_pct = (100.0 * tot_signal_all / max(tot_events_all, 1)) if tot_events_all > 0 else 0.0
    print(f"TOTAL: {tot_signal_all} events selected out of {tot_events_all} total events ({overall_pct:.2f}%).")
    print(f"Saved {files_written} non-empty output file(s) (skipped {total_files - files_written} empty files).")
    print("=" * 80)
    print("\nDone!")


if __name__ == "__main__":
    main()
