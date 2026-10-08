"""
snd.filter
----------
Modular event filtering, selection resolution, truth TTree management,
and single-file RDataFrame processing pipelines for SND@LHC.
"""

from __future__ import annotations

import os
import array
from typing import Set, Dict, Any, Tuple, Callable, Optional, List
import ROOT

from .channels import ChannelLookupManager


def build_processor(
    proc_cfg: dict,
    flavor: str = "numu"
) -> ROOT.snd.NeutrinoTruthProcessor:
    """Build and configure a NeutrinoTruthProcessor for specified flavor (numu, nue, nutau, universal)."""
    config = ROOT.snd.NeutrinoTruthConfig()
    config.targetZMin = float(proc_cfg.get("target_z_min", 260.0))
    config.targetZMax = float(proc_cfg.get("target_z_max", 360.0))
    config.fidXMin = float(proc_cfg.get("fid_x_min", -60.0))
    config.fidXMax = float(proc_cfg.get("fid_x_max", -0.0))
    config.fidYMin = float(proc_cfg.get("fid_y_min", 3.0))
    config.fidYMax = float(proc_cfg.get("fid_y_max", 63.0))
    config.fiducialMargin = float(proc_cfg.get("fiducial_margin", 0.0))
    config.minLeptonMomentum = float(proc_cfg.get("min_lepton_momentum", 0.0))
    config.maxCharmFlightDistance = float(proc_cfg.get("max_charm_flight_distance", 100.0))
    config.weightScale = float(proc_cfg.get("weight_scale", 1.0))
    config.minDSPoints = int(proc_cfg.get("min_ds_points", 3))
    config.requireDirectCharmDecay = bool(proc_cfg.get("require_direct_charm_decay", True))

    flv = str(flavor).lower().replace("_", "").replace("-", "")
    if flv in ["nue", "electron"]:
        return ROOT.snd.ElectronNeutrinoTruthProcessor(config)
    elif flv in ["nutau", "tau"]:
        return ROOT.snd.TauNeutrinoTruthProcessor(config)
    elif flv in ["universal", "any", "inclusive"]:
        return ROOT.snd.NeutrinoTruthProcessor(config)
    else:
        return ROOT.snd.MuonNeutrinoTruthProcessor(config)


def resolve_hierarchical_selection(
    cfg: dict,
    cli_fiducial: bool = False,
    cli_ds_acceptance: Optional[bool] = None,
) -> Tuple[str, str, Callable, Set[str]]:
    """
    Parse the hierarchical selection options from the config file:
      flavor -> interaction -> charm (require, species, decay, require_opposite_sign, require_ds_acceptance)
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

    if cli_ds_acceptance is not None:
        require_ds_acc = cli_ds_acceptance
    else:
        require_ds_acc = bool(charm_cfg.get("require_ds_acceptance", False))
    min_ds_points = int(charm_cfg.get("min_ds_points", 3))

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
        desc_parts.append("inclusive flavor")

    # 3. Interaction current selection (Level 2)
    if interaction == "CC":
        clauses.append("(is_numu_cc || is_anti_numu_cc)")
        desc_parts.append("CC")
        active_tiers.add("lepton")
    elif interaction == "NC":
        clauses.append("is_nc")
        desc_parts.append("NC")
    else:
        desc_parts.append("inclusive current")
        if flavor in ["numu", "muon"]:
            active_tiers.add("lepton")

    # 4. Charmed hadron production (Level 3)
    if require_charm:
        clauses.append("has_charm")
        desc_parts.append("charmed hadron")
        active_tiers.add("charm")

        # Species filter
        species_pdg_map = {
            "d0": 421,
            "dplus": 411,
            "d+": 411,
            "ds": 431,
            "dsplus": 431,
            "lambdac": 4122,
            "lambda_c": 4122,
        }
        if charm_species in species_pdg_map:
            target_pdg = species_pdg_map[charm_species]
            clauses.append(f"std::abs(charm_pdg) == {target_pdg}")
            desc_parts.append(f"{charm_species} (|pdg|={target_pdg})")
        elif charm_species != "any":
            try:
                pdg_val = abs(int(charm_species))
                clauses.append(f"std::abs(charm_pdg) == {pdg_val}")
                desc_parts.append(f"pdg={pdg_val}")
            except ValueError:
                pass

        # 5. Charm decay channel (Level 4)
        if charm_decay in ["muon", "tomuon", "dimuon"]:
            active_tiers.update(["decay_muon", "dimuon"])
            if require_os:
                clauses.append("has_candidate && is_opposite_sign")
                desc_parts.append("decay to prompt muon (opposite-sign dimuon candidate)")
            else:
                clauses.append("has_prompt_charm_muon")
                desc_parts.append("decay to prompt muon")

            if require_ds_acc:
                clauses.append("mu2_in_ds_acceptance")
                desc_parts.append(f"mu2 DS acceptance (>= {min_ds_points} DS points)")
        elif charm_decay != "any":
            desc_parts.append(f"charm decay={charm_decay}")
    else:
        desc_parts.append("inclusive charm")

    # 6. Fiducial volume requirement
    if require_fiducial:
        clauses.append("is_fiducial")
        desc_parts.append("Target fiducial volume")

    # Assemble RDF expression
    filter_expr = " && ".join(f"({c})" for c in clauses) if clauses else "true"
    description = " -> ".join(desc_parts)

    # In-loop Python verification predicate
    def predicate(info) -> bool:
        if flavor in ["numu", "muon"] and abs(info.nuPdg) != 14:
            return False
        elif flavor in ["nue", "electron"] and abs(info.nuPdg) != 12:
            return False
        elif flavor in ["nutau", "tau"] and abs(info.nuPdg) != 16:
            return False

        if interaction == "CC" and not (info.isNuMuCC or info.isAntiNuMuCC):
            return False
        elif interaction == "NC" and not info.isNC:
            return False

        if require_charm:
            if not info.hasCharm:
                return False
            if charm_species in species_pdg_map and abs(info.charmPdg) != species_pdg_map[charm_species]:
                return False

            if charm_decay in ["muon", "tomuon", "dimuon"]:
                if require_os:
                    if not (info.hasCandidate and info.isOppositeSignDimuon):
                        return False
                elif not info.hasPromptCharmMuon:
                    return False

                if require_ds_acc and not info.mu2InDS:
                    return False

        if require_fiducial and not info.isFiducial:
            return False

        return True

    # Check truth_tree branch mode in config: "auto" vs "all"
    tree_branch_mode = str(cfg.get("truth_tree", {}).get("branch_mode", "auto")).lower().strip()
    if tree_branch_mode == "all":
        active_tiers = {"universal", "lepton", "charm", "decay_muon", "dimuon"}

    return filter_expr, description, predicate, active_tiers


def setup_truth_branches(trees: List[ROOT.TTree], active_tiers: Set[str]) -> Dict[str, array.array]:
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
            "entry", "nu_pdg", "nu_flavor", "is_neutrino",
            "is_cc", "is_nc", "is_numu_cc", "is_anti_numu_cc",
            "interaction_type", "region_type", "is_fiducial",
            "channel_id", "n_primary_tracks", "n_primary_particles",
            "n_primary_charged", "n_primary_neutral",
            "n_primary_leptons", "n_primary_mesons", "n_primary_baryons",
            "n_primary_hadrons",
            "charm_decay_channel_id",
            "charm_has_direct_muon", "charm_has_direct_electron",
            "charm_has_direct_pion", "charm_has_direct_kaon",
            "has_downstream_charm_muon", "downstream_muon_track_id",
            "downstream_muon_pdg", "downstream_muon_mother_pdg",
            "has_charm_hadronic_downstream_muon"
        ])
        double_vars.extend([
            "mc_weight", "raw_weight",
            "nu_e", "nu_p", "nu_px", "nu_py", "nu_pz", "nu_pt", "nu_eta", "nu_phi", "nu_theta",
            "vtx_x", "vtx_y", "vtx_z", "vtx_t",
            "q2", "bjorken_x", "inelasticity_y", "hadronic_w",
            "hadronic_e_total", "hadronic_pt", "missing_pt",
            "downstream_muon_p", "downstream_muon_pt"
        ])

    # 2. Primary Lepton Tier
    if "lepton" in active_tiers:
        int_vars.extend([
            "primary_lepton_track_id", "primary_lepton_pdg", "primary_lepton_charge",
            "mu1_n_ds_points", "mu1_in_ds_acceptance"
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
            "mu2_mother_track_id", "mu2_mother_pdg",
            "mu2_n_ds_points", "mu2_in_ds_acceptance"
        ])
        double_vars.extend([
            "mu2_p", "mu2_pt", "mu2_px", "mu2_py", "mu2_pz", "mu2_e", "mu2_eta", "mu2_phi", "mu2_theta",
            "mu2_slope_xz", "mu2_slope_yz",
            "mu2_ptrel", "mu2_ip3d", "mu2_ipxy", "mu2_opening_angle_charm"
        ])

    # 5. Composite Dimuon Tier
    if "dimuon" in active_tiers:
        int_vars.extend([
            "is_opposite_sign", "has_candidate", "n_muons_in_event", "dimuon_in_ds_acceptance"
        ])
        double_vars.extend([
            "dimuon_mass", "dimuon_pt", "dimuon_p", "dimuon_opening_angle",
            "dimuon_opening_angle_mrad", "dimuon_delta_phi", "dimuon_delta_eta", "dimuon_delta_r",
            "dimuon_energy_asym", "dimuon_p_ratio",
            "dimuon_delta_e", "dimuon_abs_delta_e", "dimuon_delta_p", "dimuon_delta_pt", "dimuon_e_ratio"
        ])

    buffers: Dict[str, array.array] = {}
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


def fill_truth_buffers(buffers: dict, info: Any, entry_idx: int, active_tiers: Set[str]):
    """Populate truth buffers according to active tiers."""
    # 1. Universal Tier
    if "universal" in active_tiers:
        buffers["entry"][0] = int(entry_idx)
        buffers["nu_pdg"][0] = int(info.nuPdg)

        # Topology and channel lookup
        channel_id = 0
        decay_id = 0
        if hasattr(info, "primaryPdgsStr") and info.primaryPdgsStr:
            try:
                prim_pdgs = [int(p) for p in str(info.primaryPdgsStr).split(",") if p]
                ch_mgr = ChannelLookupManager.get_instance()
                channel_id, _, _ = ch_mgr.get_or_register_channel(prim_pdgs, int(info.nuPdg))
            except Exception:
                pass

        if getattr(info, "hasCharm", False) and hasattr(info, "charmDaughterPdgsStr") and info.charmDaughterPdgsStr:
            try:
                d_pdgs = [int(p) for p in str(info.charmDaughterPdgsStr).split(",") if p]
                ch_mgr = ChannelLookupManager.get_instance()
                decay_id, _, _, _ = ch_mgr.get_or_register_charm_decay(int(info.charmPdg), d_pdgs)
            except Exception:
                pass

        buffers["nu_flavor"][0] = int(getattr(info, "nuFlavor", abs(info.nuPdg)))
        buffers["is_neutrino"][0] = int(getattr(info, "isNeutrino", 1 if info.nuPdg > 0 else 0))
        buffers["channel_id"][0] = int(channel_id)
        buffers["n_primary_particles"][0] = int(getattr(info, "nPrimaryTracks", 0))
        buffers["n_primary_charged"][0] = int(getattr(info, "nPrimaryCharged", 0))
        buffers["n_primary_neutral"][0] = int(getattr(info, "nPrimaryNeutral", 0))
        buffers["n_primary_leptons"][0] = int(getattr(info, "nPrimaryLeptons", 0))
        buffers["n_primary_mesons"][0] = int(getattr(info, "nPrimaryMesons", 0))
        buffers["n_primary_baryons"][0] = int(getattr(info, "nPrimaryBaryons", 0))
        buffers["charm_decay_channel_id"][0] = int(decay_id)
        buffers["charm_has_direct_muon"][0] = int(getattr(info, "charmHasDirectMuon", False))
        buffers["charm_has_direct_electron"][0] = int(getattr(info, "charmHasDirectElectron", False))
        buffers["charm_has_direct_pion"][0] = int(getattr(info, "charmHasDirectPion", False))
        buffers["charm_has_direct_kaon"][0] = int(getattr(info, "charmHasDirectKaon", False))
        buffers["has_downstream_charm_muon"][0] = int(getattr(info, "hasDownstreamCharmMuon", False))
        buffers["downstream_muon_track_id"][0] = int(getattr(info, "downstreamMuonTrackId", -1))
        buffers["downstream_muon_pdg"][0] = int(getattr(info, "downstreamMuonPdg", 0))
        buffers["downstream_muon_mother_pdg"][0] = int(getattr(info, "downstreamMuonMotherPdg", 0))
        buffers["has_charm_hadronic_downstream_muon"][0] = int(getattr(info, "hasCharmHadronicDownstreamMuon", False))
        buffers["downstream_muon_p"][0] = float(getattr(info, "downstreamMuonP", 0.0))
        buffers["downstream_muon_pt"][0] = float(getattr(info, "downstreamMuonPt", 0.0))

        buffers["is_cc"][0] = int(info.isCC)
        buffers["is_nc"][0] = int(info.isNC)
        buffers["is_numu_cc"][0] = int(getattr(info, "isNuMuCC", False))
        buffers["is_anti_numu_cc"][0] = int(getattr(info, "isAntiNuMuCC", False))
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
        buffers["mu1_n_ds_points"][0] = int(getattr(info, "mu1nDSPoints", 0))
        buffers["mu1_in_ds_acceptance"][0] = int(getattr(info, "mu1InDS", 0))

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
        buffers["mu2_n_ds_points"][0] = int(info.mu2nDSPoints)
        buffers["mu2_in_ds_acceptance"][0] = int(info.mu2InDS)

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
        buffers["dimuon_in_ds_acceptance"][0] = int(getattr(info, "dimuonInDSAcceptance", False))

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
        buffers["dimuon_delta_e"][0] = float(info.mu1E - info.mu2E)
        buffers["dimuon_abs_delta_e"][0] = float(abs(info.mu1E - info.mu2E))
        buffers["dimuon_delta_p"][0] = float(info.mu1P - info.mu2P)
        buffers["dimuon_delta_pt"][0] = float(info.mu1Pt - info.mu2Pt)
        buffers["dimuon_e_ratio"][0] = float(info.mu2E / info.mu1E) if info.mu1E > 1e-6 else 0.0


def process_single_file(
    input_file: str,
    output_file: str,
    processor: ROOT.snd.MuonNeutrinoTruthProcessor,
    cfg: dict,
    filter_expr: str,
    predicate: Callable,
    active_tiers: Set[str],
    max_entries: int = -1
) -> dict:
    """
    Process a single input file:
    1. Evaluates truth observables and finds matching entry numbers using sequential RDataFrame.
    2. If 0 events match and skip_empty_files is true, completely skips creating/saving ROOT file.
    3. If events match, clones cbmsim and creates a dedicated flat `truth` TTree with active tiers.
    4. Applies in-loop validation guard before saving each entry.
    5. Saves diagnostic histograms and trees into output file.
    """
    tree_name = cfg.get("input", {}).get("tree_name", "cbmsim")
    skip_empty = cfg.get("output", {}).get("skip_empty_files", True)

    # Verify input file and tree existence
    has_mufilter = False
    f_test = ROOT.TFile.Open(input_file)
    if not f_test or f_test.IsZombie():
        print(f"  [Warning] Cannot open input file: {input_file}")
        return {"total": 0, "signal": 0, "status": "error_open", "output_file": None}

    actual_tree_name = tree_name
    if not f_test.Get(actual_tree_name):
        for alt_name in ["cbmsim", "rawConv"]:
            if f_test.Get(alt_name):
                actual_tree_name = alt_name
                break
    t_check = f_test.Get(actual_tree_name)
    if t_check and t_check.GetBranch("MuFilterPoint"):
        has_mufilter = True
    f_test.Close()

    # Disable implicit multi-threading per file so Take("rdfentry_") is 100% exact and deterministic
    ROOT.DisableImplicitMT()

    # 1. RDataFrame setup for fast truth processing and filtering
    df_raw = ROOT.RDataFrame(actual_tree_name, input_file)
    if max_entries > 0:
        df_raw = df_raw.Range(max_entries)

    # Define truth observables
    if has_mufilter:
        proc_ds = ROOT.snd.MuonNeutrinoTruthWithDSProcessor(processor)
        df_base = df_raw.Define("truth", proc_ds, ["MCTrack", "MuFilterPoint"])
    else:
        df_base = df_raw.Define("truth", processor, ["MCTrack"])

    df_truth = (
        df_base
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
              .Define("mu1_e", "truth.mu1E")
              .Define("charm_pdg", "truth.charmPdg")
              .Define("abs_charm_pdg", "std::abs(truth.charmPdg)")
              .Define("charm_p", "truth.charmP")
              .Define("charm_pt", "truth.charmPt")
              .Define("charm_e", "truth.charmE")
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
              .Define("mu1_track_id", "truth.mu1TrackId")
              .Define("mu1_n_ds_points", "truth.mu1nDSPoints")
              .Define("mu1_n_ds_hor_points", "truth.mu1nDSHorizontalPoints")
              .Define("mu1_n_ds_ver_points", "truth.mu1nDSVerticalPoints")
              .Define("mu1_in_ds_acceptance", "truth.mu1InDS")
              .Define("charm_track_id", "truth.charmTrackId")
              .Define("mu2_track_id", "truth.mu2TrackId")
              .Define("mu2_p", "truth.mu2P")
              .Define("mu2_pt", "truth.mu2Pt")
              .Define("mu2_e", "truth.mu2E")
              .Define("mu2_ip3d", "truth.mu2IP3D")
              .Define("mu2_ptrel", "truth.mu2PtRel")
              .Define("mu2_n_ds_points", "truth.mu2nDSPoints")
              .Define("mu2_n_ds_hor_points", "truth.mu2nDSHorizontalPoints")
              .Define("mu2_n_ds_ver_points", "truth.mu2nDSVerticalPoints")
              .Define("mu2_in_ds_acceptance", "truth.mu2InDS")
              .Define("dimuon_in_ds_acceptance", "truth.dimuonInDSAcceptance")
              .Define("dimuon_mass", "truth.dimuonInvMass")
              .Define("dimuon_pt", "truth.dimuonPt")
              .Define("dimuon_opening_angle_mrad", "truth.dimuonOpeningAngleMrad")
              .Define("dimuon_delta_phi", "truth.dimuonDeltaPhi")
              .Define("dimuon_e_sum", "mu1_e + mu2_e")
              .Define("dimuon_delta_e", "mu1_e - mu2_e")
              .Define("dimuon_abs_delta_e", "std::abs(mu1_e - mu2_e)")
              .Define("dimuon_delta_p", "truth.mu1P - mu2_p")
              .Define("dimuon_delta_pt", "truth.mu1Pt - mu2_pt")
              .Define("dimuon_e_ratio", "(mu1_e > 1e-6) ? (mu2_e / mu1_e) : 0.0")
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
        mufilter_pts = getattr(t_in, "MuFilterPoint", None)
        info = processor.processMuonNeutrino(t_in.MCTrack, mufilter_pts)

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


def find_primary_muon_track_id(tree: Any) -> int:
    """
    Returns the MCTrack index of the prompt muon from the neutrino interaction vertex.
    """
    if hasattr(tree, "mu1_track_id"):
        try:
            tid = int(tree.mu1_track_id)
            if tid >= 0:
                return tid
        except Exception:
            pass

    if not hasattr(tree, "MCTrack") or tree.MCTrack.GetEntries() == 0:
        return -1

    best_id = -1
    max_p = -1.0
    for i, trk in enumerate(tree.MCTrack):
        if abs(trk.GetPdgCode()) == 13 and trk.GetMotherId() == 0:
            p = trk.GetP()
            if p > max_p:
                max_p = p
                best_id = i
    return best_id


def find_charm_muon_track_id(tree: Any, direct_only: bool = True) -> int:
    """
    Returns the MCTrack index of the secondary muon originating from the charm hadron decay.
    First inspects tree.mu2_track_id if available.
    Otherwise searches MCTrack for the charm descendant muon with MAXIMUM momentum.
    If direct_only is True (default), requires the muon to be an immediate direct daughter of the charmed hadron.
    """
    if hasattr(tree, "mu2_track_id"):
        try:
            tid = int(tree.mu2_track_id)
            if tid >= 0:
                return tid
        except Exception:
            pass

    if not hasattr(tree, "MCTrack") or tree.MCTrack.GetEntries() == 0:
        return -1

    mu1_trk_id = find_primary_muon_track_id(tree)
    charm_trk_id = -1

    # Locate primary charmed hadron
    for i, trk in enumerate(tree.MCTrack):
        mother = trk.GetMotherId()
        abs_pdg = abs(trk.GetPdgCode())
        if abs_pdg in [411, 421, 431, 4122, 4232, 4132, 4332] and mother == 0 and charm_trk_id < 0:
            charm_trk_id = i

    if charm_trk_id < 0:
        return -1

    # Locate muon daughter descending from charm with MAXIMUM momentum
    best_mu2_id = -1
    max_p = -1.0
    n_tracks = tree.MCTrack.GetEntries()
    for i, trk in enumerate(tree.MCTrack):
        if abs(trk.GetPdgCode()) == 13 and i != mu1_trk_id:
            mid = trk.GetMotherId()
            if direct_only:
                if 0 <= mid < n_tracks:
                    parent = tree.MCTrack[mid]
                    parent_pdg = abs(parent.GetPdgCode())
                    is_charm = (
                        parent_pdg in [411, 421, 431, 4122, 4232, 4132, 4332]
                        or ((parent_pdg // 10) % 10 == 4 or (parent_pdg // 100) % 10 == 4)
                    )
                    if is_charm:
                        p = trk.GetP()
                        if p > max_p:
                            max_p = p
                            best_mu2_id = i
            else:
                curr_mid = mid
                while 0 <= curr_mid < n_tracks:
                    if curr_mid == charm_trk_id:
                        p = trk.GetP()
                        if p > max_p:
                            max_p = p
                            best_mu2_id = i
                        break
                    curr = tree.MCTrack[curr_mid]
                    curr_mid = curr.GetMotherId()

    return best_mu2_id


def count_ds_mcpoints(tree: Any, track_id: int) -> Tuple[int, int, int]:
    """
    Count the number of MCPoints in the Downstream (DS) MuFilter system (system == 3)
    for a given track ID, separating horizontal and vertical planes.

    Returns:
        (n_horizontal, n_vertical, n_total)
        where:
          - horizontal planes in DS have bar < 60
          - vertical planes in DS have bar >= 60
    """
    if track_id < 0 or not hasattr(tree, "MuFilterPoint"):
        return 0, 0, 0

    n_hor = 0
    n_ver = 0
    for pt in tree.MuFilterPoint:
        if pt.GetTrackID() == track_id:
            det_id = pt.GetDetectorID()
            if (det_id // 10000) == 3:  # Downstream (DS) MuFilter subsystem
                bar = det_id % 1000
                if bar < 60:
                    n_hor += 1
                else:
                    n_ver += 1

    return n_hor, n_ver, n_hor + n_ver


def is_dimuon_in_ds_acceptance(
    tree: Any,
    min_hor_points: int = 3,
    min_ver_points: int = 3,
    mu1_id: int = -1,
    mu2_id: int = -1,
) -> bool:
    """
    Checks if BOTH muons (prompt mu1 and charm decay mu2) travel through the DS system,
    requiring >= min_hor_points in horizontal planes AND >= min_ver_points in vertical planes
    in the DS subsystem.
    """
    if hasattr(tree, "dimuon_in_ds_acceptance"):
        try:
            return bool(tree.dimuon_in_ds_acceptance)
        except Exception:
            pass

    if mu1_id < 0:
        mu1_id = find_primary_muon_track_id(tree)
    if mu2_id < 0:
        mu2_id = find_charm_muon_track_id(tree)

    if mu1_id < 0 or mu2_id < 0:
        return False

    mu1_hor, mu1_ver, _ = count_ds_mcpoints(tree, mu1_id)
    mu2_hor, mu2_ver, _ = count_ds_mcpoints(tree, mu2_id)

    mu1_pass = (mu1_hor >= min_hor_points and mu1_ver >= min_ver_points)
    mu2_pass = (mu2_hor >= min_hor_points and mu2_ver >= min_ver_points)

    return (mu1_pass and mu2_pass)


def count_mu2_ds_mcpoints(tree: Any, mu2_track_id: int = -1) -> int:
    """Backwards-compatible helper returning total DS MCPoints for charm decay muon."""
    if mu2_track_id < 0:
        mu2_track_id = find_charm_muon_track_id(tree)
    _, _, total = count_ds_mcpoints(tree, mu2_track_id)
    return total


def process_simulation_file_dual_truth(
    input_file: str,
    truth_output_file: str,
    signal_output_file: Optional[str] = None,
    processor: Optional[ROOT.snd.MuonNeutrinoTruthProcessor] = None,
    cfg: Optional[dict] = None,
    signal_filter_expr: Optional[str] = None,
    max_entries: int = -1,
    skip_empty_signal: bool = True,
    create_symlinks: bool = True,
) -> Dict[str, Any]:
    """
    Extract truth observables for ALL events into truth_output_file,
    and simultaneously extract signal events (nu_mu CC charm dimuon in DS acceptance)
    into signal_output_file in a single pass over the input simulation file.
    Also creates symlinks to auxiliary input ROOT files in the output directory.
    """
    from .io_utils import copy_auxiliary_metadata, symlink_input_root_files, load_config

    if cfg is None:
        cfg = load_config()
    if processor is None:
        processor = build_processor(cfg.get("processor", {}))

    if signal_filter_expr is None:
        signal_filter_expr = (
            "(is_numu_cc || is_anti_numu_cc) && has_charm && "
            "has_prompt_charm_muon && is_opposite_sign && mu2_in_ds_acceptance"
        )

    # 1. Inspect input tree
    f_test = ROOT.TFile.Open(input_file, "READ")
    if not f_test or f_test.IsZombie():
        return {"input_file": input_file, "total": 0, "signal": 0, "status": "error_open_in"}

    actual_tree_name = "cbmsim"
    if not f_test.Get(actual_tree_name):
        for alt in ["rawConv", "events"]:
            if f_test.Get(alt):
                actual_tree_name = alt
                break

    t_check = f_test.Get(actual_tree_name)
    if not t_check:
        f_test.Close()
        return {"input_file": input_file, "total": 0, "signal": 0, "status": "no_tree"}

    n_tot = t_check.GetEntries()
    has_mufilter = bool(t_check.GetBranch("MuFilterPoint"))
    f_test.Close()

    ROOT.DisableImplicitMT()

    # 2. RDataFrame truth evaluation & signal entry extraction
    df_raw = ROOT.RDataFrame(actual_tree_name, input_file)
    if max_entries > 0:
        df_raw = df_raw.Range(max_entries)
    n_process = min(n_tot, max_entries) if max_entries > 0 else n_tot

    if has_mufilter:
        proc_ds = ROOT.snd.MuonNeutrinoTruthWithDSProcessor(processor)
        df_base = df_raw.Define("truth", proc_ds, ["MCTrack", "MuFilterPoint"])
    else:
        df_base = df_raw.Define("truth", processor, ["MCTrack"])

    df_truth = (
        df_base
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
        .Define("mu1_in_ds_acceptance", "truth.mu1InDS")
        .Define("mu2_in_ds_acceptance", "truth.mu2InDS")
        .Define("dimuon_in_ds_acceptance", "truth.dimuonInDSAcceptance")
    )

    df_sig = df_truth.Filter(signal_filter_expr, "Signal Selection")
    c_sig = df_sig.Count()
    sig_entries_rptr = df_sig.Take["ULong64_t"]("rdfentry_")
    ROOT.RDF.RunGraphs([c_sig, sig_entries_rptr])

    n_sig = int(c_sig.GetValue())
    signal_entries_set = set(sig_entries_rptr.GetValue())

    # 3. Setup output file for ALL events
    out_dir_1 = os.path.dirname(os.path.abspath(truth_output_file))
    if out_dir_1:
        os.makedirs(out_dir_1, exist_ok=True)

    f_in = ROOT.TFile.Open(input_file, "READ")
    t_in = f_in.Get(actual_tree_name)

    f_out_all = ROOT.TFile.Open(truth_output_file, "RECREATE")
    if not f_out_all or f_out_all.IsZombie():
        f_in.Close()
        return {"input_file": input_file, "total": n_tot, "signal": 0, "status": "error_create_all"}

    out_tree_all = t_in.CloneTree(0)
    out_tree_all.SetName(actual_tree_name)
    truth_tree_all = ROOT.TTree("truth", "Hierarchical Truth Observables for All Events")
    all_tiers = {"universal", "lepton", "charm", "decay_muon", "dimuon"}
    truth_buffers_all = setup_truth_branches([out_tree_all, truth_tree_all], all_tiers)

    # 4. Setup output file for SIGNAL events
    write_signal = bool(signal_output_file and (n_sig > 0 or not skip_empty_signal))
    f_out_sig = None
    out_tree_sig = None
    truth_tree_sig = None
    truth_buffers_sig = None

    if write_signal:
        out_dir_2 = os.path.dirname(os.path.abspath(signal_output_file))
        if out_dir_2:
            os.makedirs(out_dir_2, exist_ok=True)
        f_out_sig = ROOT.TFile.Open(signal_output_file, "RECREATE")
        if f_out_sig and not f_out_sig.IsZombie():
            out_tree_sig = t_in.CloneTree(0)
            out_tree_sig.SetName(actual_tree_name)
            truth_tree_sig = ROOT.TTree("truth", "Hierarchical Truth Observables for NuMu CC Charm Dimuon Signal")
            truth_buffers_sig = setup_truth_branches([out_tree_sig, truth_tree_sig], all_tiers)

    # 5. Single-pass Fill Loop
    n_sig_filled = 0
    for iev in range(n_process):
        t_in.GetEntry(iev)
        mufilter_pts = getattr(t_in, "MuFilterPoint", None)
        info = processor.processMuonNeutrino(t_in.MCTrack, mufilter_pts)

        # Always fill all-events trees
        fill_truth_buffers(truth_buffers_all, info, iev, all_tiers)
        out_tree_all.Fill()
        truth_tree_all.Fill()

        # Fill signal trees if event matched
        if f_out_sig and (iev in signal_entries_set):
            fill_truth_buffers(truth_buffers_sig, info, iev, all_tiers)
            out_tree_sig.Fill()
            truth_tree_sig.Fill()
            n_sig_filled += 1

    # 6. Write and close
    f_out_all.cd()
    out_tree_all.Write()
    truth_tree_all.Write()
    f_out_all.Close()

    if f_out_sig:
        f_out_sig.cd()
        out_tree_sig.Write()
        truth_tree_sig.Write()
        f_out_sig.Close()

    f_in.Close()

    # 7. Copy FairRoot metadata
    copy_auxiliary_metadata(input_file, truth_output_file)
    if write_signal and signal_output_file and os.path.exists(signal_output_file):
        copy_auxiliary_metadata(input_file, signal_output_file)

    # 8. Symlink auxiliary input files into output directory
    symlinks_created = 0
    if create_symlinks:
        in_dir = os.path.dirname(os.path.abspath(input_file))
        exclude = {
            os.path.basename(truth_output_file),
            os.path.basename(input_file),
        }
        if signal_output_file:
            exclude.add(os.path.basename(signal_output_file))
        links = symlink_input_root_files(input_dir=in_dir, output_dir=out_dir_1, exclude_filenames=exclude)
        symlinks_created = len(links)

    return {
        "input_file": input_file,
        "total": n_tot,
        "processed": n_process,
        "signal": n_sig_filled,
        "truth_file": truth_output_file,
        "signal_file": signal_output_file if (write_signal and n_sig_filled > 0) else None,
        "symlinks_created": symlinks_created,
        "status": "success",
    }


def process_categorized_neutrino_file(
    input_file: str,
    output_file: str,
    processor: Optional[ROOT.snd.NeutrinoTruthProcessor] = None,
    cfg: Optional[dict] = None,
    max_entries: int = -1,
    create_symlinks: bool = True,
) -> Dict[str, Any]:
    """
    Process an SND@LHC neutrino simulation file and categorize all events:
    - Neutrino flavor (nu_mu, nu_e, nu_tau and neutrino vs antineutrino)
    - Interaction current (CC vs NC)
    - Immediate interaction products (assigned unique channel_id from lookup table)
    - Charmed hadron production and direct decay channels (charm_decay_channel_id, direct to muon, etc.)
    - Kinematics and DIS variables
    Stores all categorized events into output_file with cloned detector tree (cbmsim)
    and flat truth tree without dropping non-signal events.
    """
    from .io_utils import copy_auxiliary_metadata, symlink_input_root_files, load_config
    from collections import Counter

    if cfg is None:
        cfg = load_config()
    if processor is None:
        processor = build_processor(cfg.get("processor", {}))

    f_test = ROOT.TFile.Open(input_file, "READ")
    if not f_test or f_test.IsZombie():
        return {"input_file": input_file, "total": 0, "processed": 0, "status": "error_open_in"}

    actual_tree_name = "cbmsim"
    if not f_test.Get(actual_tree_name):
        for alt in ["rawConv", "events"]:
            if f_test.Get(alt):
                actual_tree_name = alt
                break

    t_check = f_test.Get(actual_tree_name)
    if not t_check:
        f_test.Close()
        return {"input_file": input_file, "total": 0, "processed": 0, "status": "no_tree"}

    n_tot = t_check.GetEntries()
    f_test.Close()

    n_process = min(n_tot, max_entries) if max_entries > 0 else n_tot

    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    f_in = ROOT.TFile.Open(input_file, "READ")
    t_in = f_in.Get(actual_tree_name)

    f_out = ROOT.TFile.Open(output_file, "RECREATE")
    if not f_out or f_out.IsZombie():
        f_in.Close()
        return {"input_file": input_file, "total": n_tot, "processed": 0, "status": "error_create_out"}

    out_tree = t_in.CloneTree(0)
    out_tree.SetName(actual_tree_name)
    truth_tree = ROOT.TTree("truth", "Categorized Neutrino Truth Observables")

    all_tiers = {"universal", "lepton", "charm", "decay_muon", "dimuon"}
    truth_buffers = setup_truth_branches([out_tree, truth_tree], all_tiers)

    # Statistics tracking
    flavor_counts = Counter()
    current_counts = Counter()
    n_charm = 0
    n_charm_to_muon = 0
    n_charm_to_electron = 0
    n_charm_hadronic = 0
    n_charm_hadronic_downstream_mu = 0
    channel_ids = set()

    for iev in range(n_process):
        t_in.GetEntry(iev)
        mufilter_pts = getattr(t_in, "MuFilterPoint", None)
        if hasattr(processor, "processMuonNeutrino"):
            info = processor.processMuonNeutrino(t_in.MCTrack, mufilter_pts)
        else:
            info = processor.process(t_in.MCTrack)

        fill_truth_buffers(truth_buffers, info, iev, all_tiers)
        out_tree.Fill()
        truth_tree.Fill()

        # Update stats
        flavor_counts[str(info.interactionName)] += 1
        if info.isCC:
            current_counts["CC"] += 1
        elif info.isNC:
            current_counts["NC"] += 1
        if info.hasCharm:
            n_charm += 1
            if getattr(info, "charmHasDirectMuon", False):
                n_charm_to_muon += 1
            elif getattr(info, "charmHasDirectElectron", False):
                n_charm_to_electron += 1
            elif getattr(info, "charmDirectDecayMode", "") == "hadronic":
                n_charm_hadronic += 1
            if getattr(info, "hasCharmHadronicDownstreamMuon", False):
                n_charm_hadronic_downstream_mu += 1
        channel_ids.add(truth_buffers["channel_id"][0])

    f_out.cd()
    out_tree.Write()
    truth_tree.Write()
    f_out.Close()
    f_in.Close()

    # Save lookup table
    try:
        ChannelLookupManager.get_instance().save()
    except Exception as e:
        print(f"[Warning] Failed to save channel lookup table: {e}")

    try:
        copy_auxiliary_metadata(input_file, output_file)
    except Exception as e:
        print(f"[Warning] Failed to copy auxiliary metadata: {e}")

    symlinks_created = 0
    if create_symlinks:
        in_dir = os.path.dirname(os.path.abspath(input_file))
        exclude = {
            os.path.basename(output_file),
            os.path.basename(input_file),
        }
        links = symlink_input_root_files(input_dir=in_dir, output_dir=out_dir, exclude_filenames=exclude)
        symlinks_created = len(links)

    return {
        "input_file": input_file,
        "output_file": output_file,
        "total": n_tot,
        "processed": n_process,
        "flavor_counts": dict(flavor_counts),
        "current_counts": dict(current_counts),
        "n_charm": n_charm,
        "n_charm_to_muon": n_charm_to_muon,
        "n_charm_to_electron": n_charm_to_electron,
        "n_charm_hadronic": n_charm_hadronic,
        "n_charm_hadronic_downstream_mu": n_charm_hadronic_downstream_mu,
        "n_unique_channels": len(channel_ids),
        "symlinks_created": symlinks_created,
        "status": "success",
    }

