#!/usr/bin/env python3
"""
scripts/get_hists_numu_truth.py
--------------------------------
Comprehensive truth analysis pipeline and histogram generator for
charmed hadron production and prompt dimuon signatures in muon neutrino
charged current (nu_mu CC) interactions at SND@LHC.

Truth definitions, fiducial volume thresholds, Downstream (DS) MuFilter
acceptance logic, and kinematic tier calculations are aligned 100%
identically with scripts/filter_signal_numu_charm.py.

Produces extensive 1D histograms, 2D kinematic correlations, and TProfiles,
specifically featuring:
  - Energy distributions of both muons:
      * mu1: prompt primary muon from neutrino interaction vertex
      * mu2: prompt secondary muon from semi-leptonic charm hadron decay
  - Comprehensive comparison between signal events in Downstream (DS)
    acceptance and those outside acceptance
  - Acceptance efficiency TProfiles vs neutrino and muon kinematics
  - Charmed hadron species breakdowns (D0, D+, Ds, Lambda_c)
  - Preselection and cutflow diagnostics

CLI Arguments:
  -i, --input     : Input file path or pattern (supports wildcard '*' and '%s' placeholder)
  -n, --entries   : Max events to process (-1 for full sample)
  -o, --output    : Output ROOT file for summary histograms
  -j, --threads   : Number of worker threads (>1 enables ROOT ImplicitMT)
  -c, --config    : Path to YAML configuration file
  --fiducial      : Require interaction vertex in Target fiducial volume
  --require-direct-charm-decay : Require second muon to be an immediate direct decay daughter of charm
  --no-direct-charm-decay      : Allow second muon to come from charm decay chain via intermediate hadrons
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

from snd import (
    DataManager,
    load_trident_libraries,
    load_config,
    resolve_input_files,
    build_processor,
)

# Pre-load libraries so ROOT has all custom C++ dictionaries and classes available
load_trident_libraries(_repo_root)

# Disable ROOT GUI windows in batch mode
ROOT.gROOT.SetBatch(True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Study charm production and dimuon kinematics in nu_mu CC events with SND@LHC Truth Processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Input ROOT file path or pattern (supports wildcard '*' and '%%s' placeholder)",
    )
    parser.add_argument(
        "-n", "--entries",
        type=int,
        default=-1,
        help="Number of events to process (-1 for all events)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output ROOT file for summary histograms",
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        default=None,
        help="Number of worker threads (>1 enables ROOT ImplicitMT)",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=os.path.join(_repo_root, "config", "filter_numu_charm_config.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--fiducial",
        action="store_true",
        default=False,
        help="Require primary interaction vertex within the Target fiducial volume",
    )
    parser.add_argument(
        "--require-direct-charm-decay",
        dest="require_direct_charm_decay",
        action="store_true",
        default=None,
        help="Require secondary muon to be an immediate direct decay daughter of a charmed hadron (default: True)",
    )
    parser.add_argument(
        "--no-direct-charm-decay",
        dest="require_direct_charm_decay",
        action="store_false",
        default=None,
        help="Allow secondary muon to come from charm decay chain via intermediate pions/kaons",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # 1. Load Configuration
    config_path = args.config
    if not os.path.exists(config_path):
        # Fallback to muon_neutrino_truth_config.yaml if filter config doesn't exist
        alt_path = os.path.join(_repo_root, "config", "muon_neutrino_truth_config.yaml")
        if os.path.exists(alt_path):
            config_path = alt_path

    cfg = load_config(config_path)

    input_cfg = cfg.get("input", {})
    file_pattern = args.input if args.input is not None else input_cfg.get(
        "file_pattern",
        "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/"
        "sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/*/sndLHC.Genie-TGeant4_digCPP.root"
    )
    tree_name = input_cfg.get("tree_name", "cbmsim")
    max_files = input_cfg.get("max_files", -1)

    output_cfg = cfg.get("output", {})
    output_file = args.output if args.output else output_cfg.get("default_file", "muon_neutrino_charm_truth.root")

    perf_cfg = cfg.get("performance", {})
    n_threads = args.threads if args.threads is not None else int(perf_cfg.get("threads", 1))

    print("=" * 80)
    print(" SND@LHC: Nu_Mu CC Charm Truth Analysis & Acceptance Exploration")
    print("=" * 80)
    print(f"Configuration file : {config_path}")
    print(f"Input file pattern : {file_pattern}")
    print(f"Tree name          : {tree_name}")
    print(f"Max entries        : {args.entries if args.entries > 0 else 'All'}")
    print(f"Output file        : {output_file}")
    print(f"Threads            : {n_threads}")
    print(f"Require fiducial   : {args.fiducial}")
    direct_decay_flag = args.require_direct_charm_decay if args.require_direct_charm_decay is not None else cfg.get("processor", {}).get("require_direct_charm_decay", True)
    print(f"Direct charm decay : {direct_decay_flag}")

    # Configure ROOT Implicit Multi-Threading (IMT)
    effective_threads = n_threads
    if n_threads > 1:
        if args.entries > 0:
            print(f"\n[Notice] Multi-threading (-j {n_threads}) cannot be combined with event Range (-n {args.entries}) in ROOT RDataFrame.")
            print("         Running sequentially in single-threaded mode to preserve the -n range limit.")
            effective_threads = 0
        else:
            ROOT.EnableImplicitMT(n_threads)
            print(f"\nEnabled ROOT Implicit Multi-Threading with {n_threads} worker threads.")

    # 2. Resolve Input Files
    input_files = resolve_input_files(file_pattern, max_files=max_files)
    if not input_files:
        raise FileNotFoundError(f"No files matched input pattern: {file_pattern}")

    # Check MuFilterPoint branch in first file to enable DS acceptance processor
    has_mufilter = False
    f_test = ROOT.TFile.Open(input_files[0])
    if f_test and not f_test.IsZombie():
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

    print(f"MuFilterPoint branch available: {has_mufilter} (DS acceptance tracking enabled: {has_mufilter})")

    # 3. Configure MuonNeutrinoTruthProcessor identically to filter_signal_numu_charm.py
    proc_cfg = cfg.get("processor", {})
    if args.require_direct_charm_decay is not None:
        proc_cfg["require_direct_charm_decay"] = args.require_direct_charm_decay
    processor = build_processor(proc_cfg)

    # 4. Initialize DataManager for dataset loading
    dm = DataManager(
        source=input_files,
        tree_name=tree_name,
        num_threads=effective_threads,
        n_files=max_files if max_files > 0 else -1,
        load_libraries=True,
    )
    df = dm.rdf(range_limit=args.entries if args.entries > 0 else None)

    # 5. Define Truth Observables
    if has_mufilter:
        proc_ds = ROOT.snd.MuonNeutrinoTruthWithDSProcessor(processor)
        df_base = df.Define("truth", proc_ds, ["MCTrack", "MuFilterPoint"])
    else:
        df_base = df.Define("truth", processor, ["MCTrack"])

    df_truth = (
        df_base
              # Event and selection flags
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
              .Define("interaction_type", "truth.interactionType")
              .Define("region_type", "truth.regionType")
              .Define("n_primary_tracks", "truth.nPrimaryTracks")
              .Define("n_primary_hadrons", "truth.nPrimaryHadrons")

              # Primary incoming neutrino
              .Define("nu_e", "truth.nuE")
              .Define("nu_p", "truth.nuP")
              .Define("nu_px", "truth.nuPx")
              .Define("nu_py", "truth.nuPy")
              .Define("nu_pz", "truth.nuPz")
              .Define("nu_pt", "truth.nuPt")
              .Define("nu_eta", "truth.nuEta")
              .Define("nu_phi", "truth.nuPhi")
              .Define("nu_theta", "truth.nuTheta")
              .Define("vtx_x", "truth.vtxX")
              .Define("vtx_y", "truth.vtxY")
              .Define("vtx_z", "truth.vtxZ")
              .Define("vtx_t", "truth.vtxT")

              # DIS Kinematics
              .Define("q2", "truth.Q2")
              .Define("bjorken_x", "truth.BjorkenX")
              .Define("inelasticity_y", "truth.InelasticityY")
              .Define("hadronic_w", "truth.HadronicW")
              .Define("hadronic_e_total", "truth.hadronicEnergyTotal")
              .Define("hadronic_pt", "truth.hadronicRecoilPt")
              .Define("missing_pt", "truth.missingPt")

              # Primary Muon (mu_1) from Neutrino Interaction Vertex
              .Define("primary_lepton_track_id", "truth.primaryLeptonTrackId")
              .Define("primary_lepton_pdg", "truth.primaryLeptonPdg")
              .Define("primary_lepton_charge", "truth.primaryLeptonCharge")
              .Define("mu1_track_id", "truth.mu1TrackId")
              .Define("mu1_pdg", "truth.mu1Pdg")
              .Define("mu1_charge", "truth.mu1Charge")
              .Define("mu1_e", "truth.mu1E")
              .Define("mu1_p", "truth.mu1P")
              .Define("mu1_pt", "truth.mu1Pt")
              .Define("mu1_px", "truth.mu1Px")
              .Define("mu1_py", "truth.mu1Py")
              .Define("mu1_pz", "truth.mu1Pz")
              .Define("mu1_eta", "truth.mu1Eta")
              .Define("mu1_phi", "truth.mu1Phi")
              .Define("mu1_theta", "truth.mu1Theta")
              .Define("mu1_slope_xz", "truth.mu1SlopeXZ")
              .Define("mu1_slope_yz", "truth.mu1SlopeYZ")
              .Define("mu1_n_ds_points", "truth.mu1nDSPoints")
              .Define("mu1_n_ds_hor_points", "truth.mu1nDSHorizontalPoints")
              .Define("mu1_n_ds_ver_points", "truth.mu1nDSVerticalPoints")
              .Define("mu1_in_ds_acceptance", "truth.mu1InDS")

              # Charmed Hadron Properties
              .Define("charm_track_id", "truth.charmTrackId")
              .Define("charm_pdg", "truth.charmPdg")
              .Define("abs_charm_pdg", "std::abs(truth.charmPdg)")
              .Define("charm_quark_content", "truth.charmQuarkContent")
              .Define("n_charm_daughters", "truth.nCharmDecayDaughters")
              .Define("charm_has_kaon", "truth.charmHasKaonDaughter")
              .Define("charm_kaon_pdg", "truth.charmKaonPdg")
              .Define("n_charmed_hadrons", "truth.nCharmedHadronsInEvent")
              .Define("charm_mass", "truth.charmMass")
              .Define("charm_e", "truth.charmE")
              .Define("charm_p", "truth.charmP")
              .Define("charm_pt", "truth.charmPt")
              .Define("charm_px", "truth.charmPx")
              .Define("charm_py", "truth.charmPy")
              .Define("charm_pz", "truth.charmPz")
              .Define("charm_eta", "truth.charmEta")
              .Define("charm_phi", "truth.charmPhi")
              .Define("charm_theta", "truth.charmTheta")
              .Define("charm_z_frac", "truth.charmEnergyFractionZ")
              .Define("decay_length_3d", "truth.decayLength3D")
              .Define("charm_decay_length", "truth.decayLength3D")
              .Define("decay_length_xy", "truth.decayLengthXY")
              .Define("decay_length_z", "truth.decayLengthZ")
              .Define("proper_time_ctau", "truth.properDecayTimeCTau")
              .Define("proper_lifetime_ps", "truth.properLifetimePs")
              .Define("charm_decay_x", "truth.charmDecayX")
              .Define("charm_decay_y", "truth.charmDecayY")
              .Define("charm_decay_z", "truth.charmDecayZ")
              .Define("charm_decay_t", "truth.charmDecayT")
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

              # Secondary Muon (mu_2) from Prompt Charm Decay
              .Define("mu2_track_id", "truth.mu2TrackId")
              .Define("mu2_pdg", "truth.mu2Pdg")
              .Define("mu2_charge", "truth.mu2Charge")
              .Define("mu2_mother_track_id", "truth.mu2MotherTrackId")
              .Define("mu2_mother_pdg", "truth.mu2MotherPdg")
              .Define("mu2_e", "truth.mu2E")
              .Define("mu2_p", "truth.mu2P")
              .Define("mu2_pt", "truth.mu2Pt")
              .Define("mu2_px", "truth.mu2Px")
              .Define("mu2_py", "truth.mu2Py")
              .Define("mu2_pz", "truth.mu2Pz")
              .Define("mu2_eta", "truth.mu2Eta")
              .Define("mu2_phi", "truth.mu2Phi")
              .Define("mu2_theta", "truth.mu2Theta")
              .Define("mu2_slope_xz", "truth.mu2SlopeXZ")
              .Define("mu2_slope_yz", "truth.mu2SlopeYZ")
              .Define("mu2_ptrel", "truth.mu2PtRel")
              .Define("mu2_ip3d", "truth.mu2IP3D")
              .Define("mu2_ipxy", "truth.mu2IPXY")
              .Define("mu2_opening_angle_charm", "truth.mu2OpeningAngleWithCharm")
              .Define("mu2_n_ds_points", "truth.mu2nDSPoints")
              .Define("mu2_n_ds_hor_points", "truth.mu2nDSHorizontalPoints")
              .Define("mu2_n_ds_ver_points", "truth.mu2nDSVerticalPoints")
              .Define("mu2_in_ds_acceptance", "truth.mu2InDS")

              # Composite Dimuon System (mu_1 + mu_2)
              .Define("dimuon_in_ds_acceptance", "truth.dimuonInDSAcceptance")
              .Define("n_muons_in_event", "truth.nMuonsInEvent")
              .Define("dimuon_mass", "truth.dimuonInvMass")
              .Define("dimuon_pt", "truth.dimuonPt")
              .Define("dimuon_p", "truth.dimuonP")
              .Define("dimuon_opening_angle", "truth.dimuonOpeningAngle")
              .Define("dimuon_opening_angle_mrad", "truth.dimuonOpeningAngleMrad")
              .Define("dimuon_delta_phi", "truth.dimuonDeltaPhi")
              .Define("dimuon_delta_eta", "truth.dimuonDeltaEta")
              .Define("dimuon_delta_r", "truth.dimuonDeltaR")
              .Define("dimuon_energy_asym", "truth.dimuonEnergyAsymmetry")
              .Define("dimuon_p_ratio", "truth.dimuonMomentumRatio")

              # Derived composite and acceptance indicators for histograms & TProfiles
              .Define("dimuon_e_sum", "mu1_e + mu2_e")
              .Define("dimuon_delta_e", "mu1_e - mu2_e")
              .Define("dimuon_abs_delta_e", "std::abs(mu1_e - mu2_e)")
              .Define("dimuon_delta_p", "mu1_p - mu2_p")
              .Define("dimuon_abs_delta_p", "std::abs(mu1_p - mu2_p)")
              .Define("dimuon_delta_pt", "mu1_pt - mu2_pt")
              .Define("dimuon_e_ratio", "(mu1_e > 1e-6) ? (mu2_e / mu1_e) : 0.0")
              .Define("dimuon_e_frac_nu", "(nu_e > 1e-6) ? (mu1_e + mu2_e) / nu_e : 0.0")
              .Define("mu2_e_frac_charm", "(charm_e > 1e-6) ? mu2_e / charm_e : 0.0")
              .Define("mu1_e_frac_nu", "(nu_e > 1e-6) ? mu1_e / nu_e : 0.0")
              .Define("mu2_e_frac_nu", "(nu_e > 1e-6) ? mu2_e / nu_e : 0.0")
              .Define("dimuon_in_ds_acc_double", "dimuon_in_ds_acceptance ? 1.0 : 0.0")
              .Define("mu1_in_ds_acc_double", "mu1_in_ds_acceptance ? 1.0 : 0.0")
              .Define("mu2_in_ds_acc_double", "mu2_in_ds_acceptance ? 1.0 : 0.0")
    )

    # 6. Define Filtered Subsamples
    df_numu_cc = df_truth.Filter("(std::abs(nu_pdg) == 14) && (is_numu_cc || is_anti_numu_cc)", "All Nu_mu CC")
    df_fiducial = df_numu_cc.Filter("is_fiducial", "Target Fiducial Volume")
    df_charm = df_numu_cc.Filter("has_charm", "Nu_mu CC with Charmed Hadron")

    # Candidate Signal (Nu_mu CC + charm + opposite-sign prompt dimuons)
    if args.fiducial:
        df_candidate = df_numu_cc.Filter(
            "is_fiducial && has_charm && has_candidate && is_opposite_sign",
            "Fiducial OS Dimuon Charm Candidate"
        )
    else:
        df_candidate = df_numu_cc.Filter(
            "has_charm && has_candidate && is_opposite_sign",
            "Opposite-Sign Dimuon Charm Candidate"
        )

    # Acceptance-partitioned signal samples
    df_in_acc = df_candidate.Filter("dimuon_in_ds_acceptance", "Signal In DS Acceptance (Both Muons)")
    df_out_acc = df_candidate.Filter("!dimuon_in_ds_acceptance", "Signal Outside DS Acceptance")
    df_mu2_in_acc = df_candidate.Filter("mu2_in_ds_acceptance", "Signal Mu2 In DS Acceptance")
    df_mu2_out_acc = df_candidate.Filter("!mu2_in_ds_acceptance", "Signal Mu2 Outside DS Acceptance")

    # Specific charm hadron species filters
    df_d0 = df_candidate.Filter("abs_charm_pdg == 421", "D0 / anti-D0")
    df_dplus = df_candidate.Filter("abs_charm_pdg == 411", "D+ / D-")
    df_ds = df_candidate.Filter("abs_charm_pdg == 431", "Ds+ / Ds-")
    df_lambdac = df_candidate.Filter("abs_charm_pdg == 4122", "Lambda_c+ / anti-Lambda_c-")

    # Register filter nodes for booking
    filter_nodes = {
        "all": df_truth,
        "numu_cc": df_numu_cc,
        "fiducial": df_fiducial,
        "charm": df_charm,
        "candidate": df_candidate,
        "signal": df_candidate,
        "in_acc": df_in_acc,
        "in_acceptance": df_in_acc,
        "out_acc": df_out_acc,
        "outside_acceptance": df_out_acc,
        "mu2_in_acc": df_mu2_in_acc,
        "mu2_out_acc": df_mu2_out_acc,
        "d0": df_d0,
        "dplus": df_dplus,
        "ds": df_ds,
        "lambdac": df_lambdac,
    }

    # 7. Book Counters
    c_total = df_truth.Count()
    c_numu_cc = df_numu_cc.Count()
    c_fiducial = df_fiducial.Count()
    c_charm = df_charm.Count()
    c_candidate = df_candidate.Count()
    c_in_acc = df_in_acc.Count()
    c_out_acc = df_out_acc.Count()
    c_mu1_in_acc = df_candidate.Filter("mu1_in_ds_acceptance").Count()
    c_mu2_in_acc = df_mu2_in_acc.Count()
    c_d0 = df_d0.Count()
    c_dplus = df_dplus.Count()
    c_ds = df_ds.Count()
    c_lambdac = df_lambdac.Count()

    booked_counters = [
        c_total, c_numu_cc, c_fiducial, c_charm, c_candidate,
        c_in_acc, c_out_acc, c_mu1_in_acc, c_mu2_in_acc,
        c_d0, c_dplus, c_ds, c_lambdac
    ]

    # 8. Book Comprehensive Suite of Histograms
    booked_histograms_1d = {}
    booked_histograms_2d = {}
    booked_profiles = {}

    # Helper function to book 1D histogram
    def book_1d(name, title, col, node, nbins=50, xmin=0.0, xmax=1.0, weight="mc_weight"):
        model = ROOT.RDF.TH1DModel(name, title, nbins, xmin, xmax)
        if weight:
            booked_histograms_1d[name] = node.Histo1D(model, col, weight)
        else:
            booked_histograms_1d[name] = node.Histo1D(model, col)

    # Helper function to book 2D histogram
    def book_2d(name, title, xcol, ycol, node, nbins_x=50, xmin=0.0, xmax=1.0, nbins_y=50, ymin=0.0, ymax=1.0, weight="mc_weight"):
        model = ROOT.RDF.TH2DModel(name, title, nbins_x, xmin, xmax, nbins_y, ymin, ymax)
        if weight:
            booked_histograms_2d[name] = node.Histo2D(model, xcol, ycol, weight)
        else:
            booked_histograms_2d[name] = node.Histo2D(model, xcol, ycol)

    # Helper function to book TProfile
    def book_prof(name, title, xcol, ycol, node, nbins=40, xmin=0.0, xmax=1.0, ymin=None, ymax=None, weight="mc_weight"):
        if ymin is not None and ymax is not None:
            model = ROOT.RDF.TProfile1DModel(name, title, nbins, xmin, xmax, ymin, ymax)
        else:
            model = ROOT.RDF.TProfile1DModel(name, title, nbins, xmin, xmax)
        if weight:
            booked_profiles[name] = node.Profile1D(model, xcol, ycol, weight)
        else:
            booked_profiles[name] = node.Profile1D(model, xcol, ycol)

    # --------------------------------------------------------------------------
    # 8a. Primary Muon (mu1) and Decay Muon (mu2) Energies & Kinematics
    # Booked for: Inclusive Signal, In-Acceptance, and Outside-Acceptance
    # --------------------------------------------------------------------------
    samples_1d = [
        ("signal", df_candidate, "Signal"),
        ("in_acc", df_in_acc, "In DS Acceptance"),
        ("out_acc", df_out_acc, "Outside DS Acceptance"),
    ]

    for prefix, node, label in samples_1d:
        # Muon Energies (user-requested core focus)
        book_1d(f"h_{prefix}_mu1_e", f"Primary #mu_{{1}} Energy ({label});E_{{#mu1}} [GeV];Events", "mu1_e", node, 100, 0.0, 800.0)
        book_1d(f"h_{prefix}_mu2_e", f"Charm Decay #mu_{{2}} Energy ({label});E_{{#mu2}} [GeV];Events", "mu2_e", node, 100, 0.0, 10.0)
        book_1d(f"h_{prefix}_mu2_e_zoom", f"Charm Decay #mu_{{2}} Energy (Zoom) ({label});E_{{#mu2}} [GeV];Events", "mu2_e", node, 60, 0.0, 3.0)
        book_1d(f"h_{prefix}_mu2_e_wide", f"Charm Decay #mu_{{2}} Energy (Full Tail) ({label});E_{{#mu2}} [GeV];Events", "mu2_e", node, 100, 0.0, 50.0)
        book_1d(f"h_{prefix}_dimuon_e_sum", f"Total Muon Energy E_{{#mu1}} + E_{{#mu2}} ({label});E_{{#mu1}} + E_{{#mu2}} [GeV];Events", "dimuon_e_sum", node, 100, 0.0, 1000.0)
        book_1d(f"h_{prefix}_dimuon_delta_e", f"Muon Energy Difference E_{{#mu1}} - E_{{#mu2}} ({label});E_{{#mu1}} - E_{{#mu2}} [GeV];Events", "dimuon_delta_e", node, 100, -50.0, 700.0)
        book_1d(f"h_{prefix}_dimuon_abs_delta_e", f"Absolute Muon Energy Difference |E_{{#mu1}} - E_{{#mu2}}| ({label});|E_{{#mu1}} - E_{{#mu2}}| [GeV];Events", "dimuon_abs_delta_e", node, 100, 0.0, 700.0)
        book_1d(f"h_{prefix}_dimuon_e_ratio", f"Muon Energy Ratio E_{{#mu2}} / E_{{#mu1}} ({label});E_{{#mu2}} / E_{{#mu1}};Events", "dimuon_e_ratio", node, 50, 0.0, 0.2)
        book_1d(f"h_{prefix}_dimuon_delta_p", f"Muon Momentum Difference p_{{#mu1}} - p_{{#mu2}} ({label});p_{{#mu1}} - p_{{#mu2}} [GeV/c];Events", "dimuon_delta_p", node, 100, -50.0, 700.0)
        book_1d(f"h_{prefix}_dimuon_delta_pt", f"Muon Transverse Momentum Difference p_{{T,#mu1}} - p_{{T,#mu2}} ({label});p_{{T,#mu1}} - p_{{T,#mu2}} [GeV/c];Events", "dimuon_delta_pt", node, 50, -2.0, 5.0)
        book_1d(f"h_{prefix}_dimuon_e_frac_nu", f"Dimuon Energy Fraction of E_{{#nu}} ({label});(E_{{#mu1}} + E_{{#mu2}}) / E_{{#nu}};Events", "dimuon_e_frac_nu", node, 50, 0.0, 1.0)
        book_1d(f"h_{prefix}_mu1_e_frac_nu", f"Primary #mu_{{1}} Energy Fraction of E_{{#nu}} ({label});E_{{#mu1}} / E_{{#nu}};Events", "mu1_e_frac_nu", node, 50, 0.0, 1.0)
        book_1d(f"h_{prefix}_mu2_e_frac_charm", f"Decay #mu_{{2}} Energy Fraction of E_{{charm}} ({label});E_{{#mu2}} / E_{{charm}};Events", "mu2_e_frac_charm", node, 50, 0.0, 1.0)

        # Muon Momenta and Angles
        book_1d(f"h_{prefix}_mu1_p", f"Primary #mu_{{1}} Total Momentum ({label});p_{{#mu1}} [GeV/c];Events", "mu1_p", node, 100, 0.0, 800.0)
        book_1d(f"h_{prefix}_mu2_p", f"Charm Decay #mu_{{2}} Total Momentum ({label});p_{{#mu2}} [GeV/c];Events", "mu2_p", node, 100, 0.0, 10.0)
        book_1d(f"h_{prefix}_mu2_p_zoom", f"Charm Decay #mu_{{2}} Total Momentum (Zoom) ({label});p_{{#mu2}} [GeV/c];Events", "mu2_p", node, 60, 0.0, 3.0)
        book_1d(f"h_{prefix}_mu2_p_wide", f"Charm Decay #mu_{{2}} Total Momentum (Full Tail) ({label});p_{{#mu2}} [GeV/c];Events", "mu2_p", node, 100, 0.0, 50.0)
        book_1d(f"h_{prefix}_mu1_pt", f"Primary #mu_{{1}} Transverse Momentum ({label});p_{{T,#mu1}} [GeV/c];Events", "mu1_pt", node, 50, 0.0, 5.0)
        book_1d(f"h_{prefix}_mu2_pt", f"Charm Decay #mu_{{2}} Transverse Momentum ({label});p_{{T,#mu2}} [GeV/c];Events", "mu2_pt", node, 60, 0.0, 1.5)
        book_1d(f"h_{prefix}_mu2_pt_zoom", f"Charm Decay #mu_{{2}} Transverse Momentum (Zoom) ({label});p_{{T,#mu2}} [GeV/c];Events", "mu2_pt", node, 50, 0.0, 0.5)
        book_1d(f"h_{prefix}_mu1_eta", f"Primary #mu_{{1}} Pseudorapidity ({label});#eta_{{#mu1}};Events", "mu1_eta", node, 50, 4.0, 10.0)
        book_1d(f"h_{prefix}_mu2_eta", f"Charm Decay #mu_{{2}} Pseudorapidity ({label});#eta_{{#mu2}};Events", "mu2_eta", node, 50, 4.0, 10.0)
        book_1d(f"h_{prefix}_mu1_theta", f"Primary #mu_{{1}} Polar Angle ({label});#theta_{{#mu1}} [rad];Events", "mu1_theta", node, 50, 0.0, 0.2)
        book_1d(f"h_{prefix}_mu2_theta", f"Charm Decay #mu_{{2}} Polar Angle ({label});#theta_{{#mu2}} [rad];Events", "mu2_theta", node, 50, 0.0, 0.2)

        # Charm Decay Displaced Vertex & Relative Kinematics
        book_1d(f"h_{prefix}_mu2_ptrel", f"Decay #mu_{{2}} p_{{T}}^{{rel}} to Charm Axis ({label});p_{{T}}^{{rel}} [GeV/c];Events", "mu2_ptrel", node, 50, 0.0, 2.0)
        book_1d(f"h_{prefix}_mu2_ip3d", f"Decay #mu_{{2}} 3D Impact Parameter to Primary Vertex ({label});IP_{{3D}} [cm];Events", "mu2_ip3d", node, 50, 0.0, 0.5)
        book_1d(f"h_{prefix}_mu2_ipxy", f"Decay #mu_{{2}} Transverse Impact Parameter ({label});IP_{{xy}} [cm];Events", "mu2_ipxy", node, 50, 0.0, 0.5)
        book_1d(f"h_{prefix}_mu2_opening_angle_charm", f"Opening Angle Between Charm and #mu_{{2}} ({label});#theta(#mu_{{2}},charm) [rad];Events", "mu2_opening_angle_charm", node, 50, 0.0, 0.2)
        book_1d(f"h_{prefix}_mu1_n_ds_points", f"Primary #mu_{{1}} DS MCPoints ({label});N_{{DS}}(#mu_{{1}});Events", "mu1_n_ds_points", node, 20, -0.5, 19.5)
        book_1d(f"h_{prefix}_mu2_n_ds_points", f"Charm Decay #mu_{{2}} DS MCPoints ({label});N_{{DS}}(#mu_{{2}});Events", "mu2_n_ds_points", node, 20, -0.5, 19.5)

        # Dimuon System Observables
        book_1d(f"h_{prefix}_dimuon_mass", f"Dimuon Invariant Mass ({label});M_{{#mu#mu}} [GeV/c^{{2}}];Events", "dimuon_mass", node, 50, 0.0, 10.0)
        book_1d(f"h_{prefix}_dimuon_pt", f"Dimuon Pair Transverse Momentum ({label});p_{{T,#mu#mu}} [GeV/c];Events", "dimuon_pt", node, 50, 0.0, 8.0)
        book_1d(f"h_{prefix}_dimuon_p", f"Dimuon Pair Total Momentum ({label});p_{{#mu#mu}} [GeV/c];Events", "dimuon_p", node, 50, 0.0, 800.0)
        book_1d(f"h_{prefix}_dimuon_opening_angle_mrad", f"Dimuon 3D Opening Angle ({label});#theta_{{#mu#mu}} [mrad];Events", "dimuon_opening_angle_mrad", node, 50, 0.0, 200.0)
        book_1d(f"h_{prefix}_dimuon_delta_phi", f"Dimuon Azimuthal Separation ({label});#Delta#phi_{{#mu#mu}} [rad];Events", "dimuon_delta_phi", node, 50, 0.0, 3.14159)
        book_1d(f"h_{prefix}_dimuon_delta_eta", f"Dimuon Pseudorapidity Separation ({label});#Delta#eta_{{#mu#mu}};Events", "dimuon_delta_eta", node, 50, 0.0, 3.0)
        book_1d(f"h_{prefix}_dimuon_delta_r", f"Dimuon Separation #Delta R ({label});#Delta R_{{#mu#mu}};Events", "dimuon_delta_r", node, 50, 0.0, 3.5)
        book_1d(f"h_{prefix}_dimuon_energy_asym", f"Dimuon Energy Asymmetry ({label});(E_{{1}} - E_{{2}}) / (E_{{1}} + E_{{2}});Events", "dimuon_energy_asym", node, 50, -1.0, 1.0)
        book_1d(f"h_{prefix}_dimuon_p_ratio", f"Dimuon Momentum Ratio p_{{2}} / p_{{1}} ({label});p_{{#mu2}} / p_{{#mu1}};Events", "dimuon_p_ratio", node, 50, 0.0, 2.0)

        # Incoming Neutrino & DIS Variables
        book_1d(f"h_{prefix}_nu_e", f"Incoming #nu_{{#mu}} Energy ({label});E_{{#nu}} [GeV];Events", "nu_e", node, 50, 0.0, 1000.0)
        book_1d(f"h_{prefix}_q2", f"4-Momentum Transfer Squared Q^{{2}} ({label});Q^{{2}} [GeV^{{2}}];Events", "q2", node, 50, 0.0, 50.0)
        book_1d(f"h_{prefix}_bjorken_x", f"Bjorken x ({label});x;Events", "bjorken_x", node, 50, 0.0, 1.0)
        book_1d(f"h_{prefix}_inelasticity_y", f"Inelasticity y ({label});y_{{inel}};Events", "inelasticity_y", node, 50, 0.0, 1.0)
        book_1d(f"h_{prefix}_hadronic_w", f"Hadronic Invariant Mass W ({label});W [GeV/c^{{2}}];Events", "hadronic_w", node, 50, 0.0, 30.0)
        book_1d(f"h_{prefix}_vtx_z", f"Interaction Vertex Z Position ({label});z_{{vtx}} [cm];Events", "vtx_z", node, 50, 260.0, 360.0)
        book_1d(f"h_{prefix}_vtx_x", f"Interaction Vertex X Position ({label});x_{{vtx}} [cm];Events", "vtx_x", node, 60, -60.0, 0.0)
        book_1d(f"h_{prefix}_vtx_y", f"Interaction Vertex Y Position ({label});y_{{vtx}} [cm];Events", "vtx_y", node, 60, 3.0, 63.0)

        # Charmed Hadron Properties
        book_1d(f"h_{prefix}_charm_species", f"Charmed Hadron Species ({label});Species;Events", "charm_species_id", node, 7, 0.5, 7.5)
        book_1d(f"h_{prefix}_charm_e", f"Charmed Hadron Energy ({label});E_{{charm}} [GeV];Events", "charm_e", node, 50, 0.0, 500.0)
        book_1d(f"h_{prefix}_charm_p", f"Charmed Hadron Momentum ({label});p_{{charm}} [GeV/c];Events", "charm_p", node, 50, 0.0, 500.0)
        book_1d(f"h_{prefix}_charm_pt", f"Charmed Hadron Transverse Momentum ({label});p_{{T,charm}} [GeV/c];Events", "charm_pt", node, 50, 0.0, 5.0)
        book_1d(f"h_{prefix}_charm_eta", f"Charmed Hadron Pseudorapidity ({label});#eta_{{charm}};Events", "charm_eta", node, 50, 4.0, 10.0)
        book_1d(f"h_{prefix}_charm_z_frac", f"Charm Energy Fraction z = E_{{c}}/E_{{had}} ({label});z_{{c}};Events", "charm_z_frac", node, 50, 0.0, 1.0)
        book_1d(f"h_{prefix}_decay_length_3d", f"Charm 3D Flight Length ({label});L_{{3D}} [cm];Events", "decay_length_3d", node, 50, 0.0, 5.0)
        book_1d(f"h_{prefix}_decay_length_xy", f"Charm Transverse Flight Length ({label});L_{{xy}} [cm];Events", "decay_length_xy", node, 50, 0.0, 1.0)
        book_1d(f"h_{prefix}_proper_time_ctau", f"Charm Proper Decay Length ({label});c#tau [cm];Events", "proper_time_ctau", node, 50, 0.0, 0.1)
        book_1d(f"h_{prefix}_proper_lifetime_ps", f"Charm Proper Lifetime ({label});#tau [ps];Events", "proper_lifetime_ps", node, 50, 0.0, 3.0)

    # --------------------------------------------------------------------------
    # 8b. 2D Histograms: Dimuon Energy & Kinematic Correlations
    # --------------------------------------------------------------------------
    # Core Dimuon Energy Correlation (E_mu2 vs E_mu1) across samples
    for prefix, node, label in samples_1d:
        book_2d(f"h2_{prefix}_mu2_e_vs_mu1_e", f"Dimuon Energy Correlation ({label});E_{{#mu1}} [GeV];E_{{#mu2}} [GeV]", "mu1_e", "mu2_e", node, 50, 0.0, 500.0, 50, 0.0, 10.0)
        book_2d(f"h2_{prefix}_mu2_e_vs_mu1_e_zoom", f"Dimuon Energy Correlation (Zoom) ({label});E_{{#mu1}} [GeV];E_{{#mu2}} [GeV]", "mu1_e", "mu2_e", node, 50, 0.0, 500.0, 60, 0.0, 3.0)
        book_2d(f"h2_{prefix}_mu2_e_vs_mu1_e_wide", f"Dimuon Energy Correlation (Wide) ({label});E_{{#mu1}} [GeV];E_{{#mu2}} [GeV]", "mu1_e", "mu2_e", node, 50, 0.0, 500.0, 50, 0.0, 50.0)

        book_2d(f"h2_{prefix}_mu2_p_vs_mu1_p", f"Dimuon Momentum Correlation ({label});p_{{#mu1}} [GeV/c];p_{{#mu2}} [GeV/c]", "mu1_p", "mu2_p", node, 50, 0.0, 500.0, 50, 0.0, 10.0)
        book_2d(f"h2_{prefix}_mu2_p_vs_mu1_p_zoom", f"Dimuon Momentum Correlation (Zoom) ({label});p_{{#mu1}} [GeV/c];p_{{#mu2}} [GeV/c]", "mu1_p", "mu2_p", node, 50, 0.0, 500.0, 60, 0.0, 3.0)
        book_2d(f"h2_{prefix}_mu2_p_vs_mu1_p_wide", f"Dimuon Momentum Correlation (Wide) ({label});p_{{#mu1}} [GeV/c];p_{{#mu2}} [GeV/c]", "mu1_p", "mu2_p", node, 50, 0.0, 500.0, 50, 0.0, 50.0)

        book_2d(f"h2_{prefix}_mu2_pt_vs_mu1_pt", f"Dimuon Transverse Momentum Correlation ({label});p_{{T,#mu1}} [GeV/c];p_{{T,#mu2}} [GeV/c]", "mu1_pt", "mu2_pt", node, 50, 0.0, 4.0, 50, 0.0, 1.5)
        book_2d(f"h2_{prefix}_mu2_pt_vs_mu1_pt_zoom", f"Dimuon Transverse Momentum Correlation (Zoom) ({label});p_{{T,#mu1}} [GeV/c];p_{{T,#mu2}} [GeV/c]", "mu1_pt", "mu2_pt", node, 50, 0.0, 4.0, 50, 0.0, 0.5)
        book_2d(f"h2_{prefix}_mu2_eta_vs_mu1_eta", f"Dimuon Pseudorapidity Correlation ({label});#eta_{{#mu1}};#eta_{{#mu2}}", "mu1_eta", "mu2_eta", node, 50, 4.0, 10.0, 50, 4.0, 10.0)

    # Kinematic correlations for candidate signal
    book_2d("h2_signal_mu1_e_vs_nu_e", "Primary #mu_{1} Energy vs E_{#nu};E_{#nu} [GeV];E_{#mu1} [GeV]", "nu_e", "mu1_e", df_candidate, 50, 0.0, 800.0, 50, 0.0, 600.0)
    book_2d("h2_signal_mu2_e_vs_nu_e", "Decay #mu_{2} Energy vs E_{#nu};E_{#nu} [GeV];E_{#mu2} [GeV]", "nu_e", "mu2_e", df_candidate, 50, 0.0, 800.0, 50, 0.0, 10.0)
    book_2d("h2_signal_mu2_e_vs_nu_e_zoom", "Decay #mu_{2} Energy vs E_{#nu} (Zoom);E_{#nu} [GeV];E_{#mu2} [GeV]", "nu_e", "mu2_e", df_candidate, 50, 0.0, 800.0, 60, 0.0, 3.0)
    book_2d("h2_signal_dimuon_delta_e_vs_nu_e", "Muon Energy Difference vs E_{#nu};E_{#nu} [GeV];E_{#mu1} - E_{#mu2} [GeV]", "nu_e", "dimuon_delta_e", df_candidate, 50, 0.0, 800.0, 50, -50.0, 600.0)
    book_2d("h2_signal_dimuon_delta_e_vs_dimuon_mass", "Muon Energy Difference vs M_{#mu#mu};M_{#mu#mu} [GeV/c^{2}];E_{#mu1} - E_{#mu2} [GeV]", "dimuon_mass", "dimuon_delta_e", df_candidate, 50, 0.0, 8.0, 50, -50.0, 600.0)
    book_2d("h2_signal_dimuon_delta_e_vs_opening_angle", "Muon Energy Difference vs #theta_{#mu#mu};#theta_{#mu#mu} [mrad];E_{#mu1} - E_{#mu2} [GeV]", "dimuon_opening_angle_mrad", "dimuon_delta_e", df_candidate, 50, 0.0, 200.0, 50, -50.0, 600.0)
    book_2d("h2_signal_mu2_e_vs_delta_e", "Decay #mu_{2} Energy vs (E_{#mu1} - E_{#mu2});E_{#mu1} - E_{#mu2} [GeV];E_{#mu2} [GeV]", "dimuon_delta_e", "mu2_e", df_candidate, 50, -50.0, 600.0, 50, 0.0, 10.0)
    book_2d("h2_signal_mu2_e_vs_charm_e", "Decay #mu_{2} Energy vs Charm Energy;E_{charm} [GeV];E_{#mu2} [GeV]", "charm_e", "mu2_e", df_candidate, 50, 0.0, 500.0, 50, 0.0, 10.0)
    book_2d("h2_signal_dimuon_mass_vs_opening_angle", "Dimuon Mass vs Opening Angle;#theta_{#mu#mu} [mrad];M_{#mu#mu} [GeV/c^{2}]", "dimuon_opening_angle_mrad", "dimuon_mass", df_candidate, 50, 0.0, 200.0, 50, 0.0, 8.0)
    book_2d("h2_signal_dimuon_mass_vs_pt", "Dimuon Mass vs Pair p_{T};p_{T,#mu#mu} [GeV/c];M_{#mu#mu} [GeV/c^{2}]", "dimuon_pt", "dimuon_mass", df_candidate, 50, 0.0, 8.0, 50, 0.0, 8.0)
    book_2d("h2_signal_opening_angle_vs_delta_phi", "Opening Angle vs Azimuthal Separation;#Delta#phi_{#mu#mu} [rad];#theta_{#mu#mu} [mrad]", "dimuon_delta_phi", "dimuon_opening_angle_mrad", df_candidate, 50, 0.0, 3.14159, 50, 0.0, 200.0)
    book_2d("h2_signal_mu2_ptrel_vs_mu2_p", "Decay #mu_{2} p_{T}^{rel} vs p_{#mu2};p_{#mu2} [GeV/c];p_{T}^{rel} [GeV/c]", "mu2_p", "mu2_ptrel", df_candidate, 50, 0.0, 10.0, 50, 0.0, 2.0)
    book_2d("h2_signal_mu2_ip3d_vs_decay_length", "Decay #mu_{2} IP_{3D} vs Charm Flight Length;L_{3D} [cm];IP_{3D} [cm]", "decay_length_3d", "mu2_ip3d", df_candidate, 50, 0.0, 3.0, 50, 0.0, 0.3)
    book_2d("h2_signal_decay_length_vs_charm_p", "Charm Flight Length vs Momentum;p_{charm} [GeV/c];L_{3D} [cm]", "charm_p", "decay_length_3d", df_candidate, 50, 0.0, 500.0, 50, 0.0, 5.0)
    book_2d("h2_signal_q2_vs_bjorken_x", "Q^{2} vs Bjorken x;Bjorken x;Q^{2} [GeV^{2}]", "bjorken_x", "q2", df_candidate, 50, 0.0, 1.0, 50, 0.0, 50.0)
    book_2d("h2_signal_vtx_xy", "Interaction Vertex Transverse Profile;x_{vtx} [cm];y_{vtx} [cm]", "vtx_x", "vtx_y", df_candidate, 60, -60.0, 0.0, 60, 3.0, 63.0)
    book_2d("h2_signal_mu1_nds_vs_mu1_e", "Primary #mu_{1} DS MCPoints vs Energy;E_{#mu1} [GeV];N_{DS}(#mu_{1})", "mu1_e", "mu1_n_ds_points", df_candidate, 50, 0.0, 800.0, 20, -0.5, 19.5)
    book_2d("h2_signal_mu2_nds_vs_mu2_e", "Charm Decay #mu_{2} DS MCPoints vs Energy;E_{#mu2} [GeV];N_{DS}(#mu_{2})", "mu2_e", "mu2_n_ds_points", df_candidate, 50, 0.0, 10.0, 20, -0.5, 19.5)
    book_2d("h2_signal_mu2_nds_vs_mu2_eta", "Charm Decay #mu_{2} DS MCPoints vs #eta;#eta_{#mu2};N_{DS}(#mu_{2})", "mu2_eta", "mu2_n_ds_points", df_candidate, 50, 4.0, 10.0, 20, -0.5, 19.5)

    # --------------------------------------------------------------------------
    # 8c. TProfiles: Kinematic Means & Acceptance Probability Profiles
    # --------------------------------------------------------------------------
    # Mean kinematics vs Neutrino Energy
    book_prof("p_signal_mu1_e_vs_nu_e", "Mean Primary #mu_{1} Energy vs E_{#nu};E_{#nu} [GeV];<E_{#mu1}> [GeV]", "nu_e", "mu1_e", df_candidate, 40, 0.0, 800.0)
    book_prof("p_signal_mu2_e_vs_nu_e", "Mean Decay #mu_{2} Energy vs E_{#nu};E_{#nu} [GeV];<E_{#mu2}> [GeV]", "nu_e", "mu2_e", df_candidate, 40, 0.0, 800.0)
    book_prof("p_signal_mu2_e_vs_charm_e", "Mean Decay #mu_{2} Energy vs Charm Energy;E_{charm} [GeV];<E_{#mu2}> [GeV]", "charm_e", "mu2_e", df_candidate, 40, 0.0, 500.0)
    book_prof("p_signal_mu1_p_vs_nu_e", "Mean Primary #mu_{1} Momentum vs E_{#nu};E_{#nu} [GeV];<p_{#mu1}> [GeV/c]", "nu_e", "mu1_p", df_candidate, 40, 0.0, 800.0)
    book_prof("p_signal_mu2_p_vs_nu_e", "Mean Decay #mu_{2} Momentum vs E_{#nu};E_{#nu} [GeV];<p_{#mu2}> [GeV/c]", "nu_e", "mu2_p", df_candidate, 40, 0.0, 800.0)
    book_prof("p_signal_mu1_pt_vs_nu_e", "Mean Primary #mu_{1} p_{T} vs E_{#nu};E_{#nu} [GeV];<p_{T,#mu1}> [GeV/c]", "nu_e", "mu1_pt", df_candidate, 40, 0.0, 800.0)
    book_prof("p_signal_mu2_pt_vs_nu_e", "Mean Decay #mu_{2} p_{T} vs E_{#nu};E_{#nu} [GeV];<p_{T,#mu2}> [GeV/c]", "nu_e", "mu2_pt", df_candidate, 40, 0.0, 800.0)
    book_prof("p_signal_charm_p_vs_nu_e", "Mean Charmed Hadron Momentum vs E_{#nu};E_{#nu} [GeV];<p_{charm}> [GeV/c]", "nu_e", "charm_p", df_candidate, 40, 0.0, 800.0)
    book_prof("p_signal_dimuon_mass_vs_nu_e", "Mean Dimuon Invariant Mass vs E_{#nu};E_{#nu} [GeV];<M_{#mu#mu}> [GeV/c^{2}]", "nu_e", "dimuon_mass", df_candidate, 30, 0.0, 800.0)
    book_prof("p_signal_opening_angle_vs_nu_e", "Mean Dimuon Opening Angle vs E_{#nu};E_{#nu} [GeV];<#theta_{#mu#mu}> [mrad]", "nu_e", "dimuon_opening_angle_mrad", df_candidate, 30, 0.0, 800.0)
    book_prof("p_signal_charm_z_frac_vs_hadronic_w", "Mean Charm Energy Fraction vs Hadronic Mass W;W [GeV/c^{2}];<z_{c}>", "hadronic_w", "charm_z_frac", df_candidate, 30, 2.0, 30.0)
    book_prof("p_signal_decay_length_vs_charm_p", "Mean Charm Flight Length vs Momentum;p_{charm} [GeV/c];<L_{3D}> [cm]", "charm_p", "decay_length_3d", df_candidate, 30, 0.0, 400.0)
    book_prof("p_signal_ip3d_vs_decay_length", "Mean Decay #mu_{2} IP_{3D} vs Flight Length;L_{3D} [cm];<IP_{3D}> [cm]", "decay_length_3d", "mu2_ip3d", df_candidate, 30, 0.0, 2.0)

    # Acceptance Probability Profiles (TProfile on 0/1 indicator = exact acceptance efficiency!)
    book_prof("p_acc_eff_vs_nu_e", "Dimuon DS Acceptance Efficiency vs E_{#nu};E_{#nu} [GeV];Acceptance Efficiency #epsilon_{DS}", "nu_e", "dimuon_in_ds_acc_double", df_candidate, 40, 0.0, 800.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu1_e", "Dimuon DS Acceptance Efficiency vs E_{#mu1};E_{#mu1} [GeV];Acceptance Efficiency #epsilon_{DS}", "mu1_e", "dimuon_in_ds_acc_double", df_candidate, 40, 0.0, 800.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu2_e", "Dimuon DS Acceptance Efficiency vs E_{#mu2};E_{#mu2} [GeV];Acceptance Efficiency #epsilon_{DS}", "mu2_e", "dimuon_in_ds_acc_double", df_candidate, 50, 0.0, 10.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_dimuon_delta_e", "Dimuon DS Acceptance Efficiency vs (E_{#mu1} - E_{#mu2});E_{#mu1} - E_{#mu2} [GeV];Acceptance Efficiency #epsilon_{DS}", "dimuon_delta_e", "dimuon_in_ds_acc_double", df_candidate, 40, -50.0, 600.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_dimuon_abs_delta_e", "Dimuon DS Acceptance Efficiency vs |E_{#mu1} - E_{#mu2}|;|E_{#mu1} - E_{#mu2}| [GeV];Acceptance Efficiency #epsilon_{DS}", "dimuon_abs_delta_e", "dimuon_in_ds_acc_double", df_candidate, 40, 0.0, 600.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_dimuon_energy_asym", "Dimuon DS Acceptance Efficiency vs Energy Asymmetry;A_{E} = (E_{1}-E_{2})/(E_{1}+E_{2});Acceptance Efficiency #epsilon_{DS}", "dimuon_energy_asym", "dimuon_in_ds_acc_double", df_candidate, 40, -1.0, 1.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu1_p", "Dimuon DS Acceptance Efficiency vs p_{#mu1};p_{#mu1} [GeV/c];Acceptance Efficiency #epsilon_{DS}", "mu1_p", "dimuon_in_ds_acc_double", df_candidate, 40, 0.0, 800.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu2_p", "Dimuon DS Acceptance Efficiency vs p_{#mu2};p_{#mu2} [GeV/c];Acceptance Efficiency #epsilon_{DS}", "mu2_p", "dimuon_in_ds_acc_double", df_candidate, 50, 0.0, 10.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu1_pt", "Dimuon DS Acceptance Efficiency vs p_{T,#mu1};p_{T,#mu1} [GeV/c];Acceptance Efficiency #epsilon_{DS}", "mu1_pt", "dimuon_in_ds_acc_double", df_candidate, 40, 0.0, 4.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu2_pt", "Dimuon DS Acceptance Efficiency vs p_{T,#mu2};p_{T,#mu2} [GeV/c];Acceptance Efficiency #epsilon_{DS}", "mu2_pt", "dimuon_in_ds_acc_double", df_candidate, 40, 0.0, 1.5, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu1_eta", "Dimuon DS Acceptance Efficiency vs #eta_{#mu1};#eta_{#mu1};Acceptance Efficiency #epsilon_{DS}", "mu1_eta", "dimuon_in_ds_acc_double", df_candidate, 40, 4.0, 10.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu2_eta", "Dimuon DS Acceptance Efficiency vs #eta_{#mu2};#eta_{#mu2};Acceptance Efficiency #epsilon_{DS}", "mu2_eta", "dimuon_in_ds_acc_double", df_candidate, 40, 4.0, 10.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_opening_angle", "Dimuon DS Acceptance Efficiency vs #theta_{#mu#mu};#theta_{#mu#mu} [mrad];Acceptance Efficiency #epsilon_{DS}", "dimuon_opening_angle_mrad", "dimuon_in_ds_acc_double", df_candidate, 40, 0.0, 200.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_dimuon_mass", "Dimuon DS Acceptance Efficiency vs M_{#mu#mu};M_{#mu#mu} [GeV/c^{2}];Acceptance Efficiency #epsilon_{DS}", "dimuon_mass", "dimuon_in_ds_acc_double", df_candidate, 40, 0.0, 8.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_vtx_z", "Dimuon DS Acceptance Efficiency vs z_{vtx};z_{vtx} [cm];Acceptance Efficiency #epsilon_{DS}", "vtx_z", "dimuon_in_ds_acc_double", df_candidate, 40, 260.0, 360.0, 0.0, 1.0)

    book_prof("p_mu1_acc_eff_vs_mu1_e", "Primary #mu_{1} DS Acceptance vs E_{#mu1};E_{#mu1} [GeV];Acceptance Efficiency #epsilon_{DS}(#mu_{1})", "mu1_e", "mu1_in_ds_acc_double", df_candidate, 40, 0.0, 800.0, 0.0, 1.0)
    book_prof("p_mu1_acc_eff_vs_mu1_eta", "Primary #mu_{1} DS Acceptance vs #eta_{#mu1};#eta_{#mu1};Acceptance Efficiency #epsilon_{DS}(#mu_{1})", "mu1_eta", "mu1_in_ds_acc_double", df_candidate, 40, 4.0, 10.0, 0.0, 1.0)
    book_prof("p_mu2_acc_eff_vs_mu2_e", "Decay #mu_{2} DS Acceptance vs E_{#mu2};E_{#mu2} [GeV];Acceptance Efficiency #epsilon_{DS}(#mu_{2})", "mu2_e", "mu2_in_ds_acc_double", df_candidate, 50, 0.0, 10.0, 0.0, 1.0)
    book_prof("p_mu2_acc_eff_vs_mu2_eta", "Decay #mu_{2} DS Acceptance vs #eta_{#mu2};#eta_{#mu2};Acceptance Efficiency #epsilon_{DS}(#mu_{2})", "mu2_eta", "mu2_in_ds_acc_double", df_candidate, 40, 4.0, 10.0, 0.0, 1.0)
    book_prof("p_mu2_acc_eff_vs_mu2_pt", "Decay #mu_{2} DS Acceptance vs p_{T,#mu2};p_{T,#mu2} [GeV/c];Acceptance Efficiency #epsilon_{DS}(#mu_{2})", "mu2_pt", "mu2_in_ds_acc_double", df_candidate, 40, 0.0, 1.5, 0.0, 1.0)

    # --------------------------------------------------------------------------
    # 8d. Charmed Hadron Species Breakdowns
    # --------------------------------------------------------------------------
    species_samples = [
        ("d0", df_d0, "D^{0}"),
        ("dplus", df_dplus, "D^{+}"),
        ("ds", df_ds, "D_{s}^{+}"),
        ("lambdac", df_lambdac, "#Lambda_{c}^{+}"),
    ]

    for sp_key, sp_node, sp_lbl in species_samples:
        book_1d(f"h_species_{sp_key}_mu1_e", f"Primary #mu_{{1}} Energy ({sp_lbl});E_{{#mu1}} [GeV];Events", "mu1_e", sp_node, 50, 0.0, 600.0)
        book_1d(f"h_species_{sp_key}_mu2_e", f"Decay #mu_{{2}} Energy ({sp_lbl});E_{{#mu2}} [GeV];Events", "mu2_e", sp_node, 50, 0.0, 10.0)
        book_1d(f"h_species_{sp_key}_mu2_e_zoom", f"Decay #mu_{{2}} Energy (Zoom) ({sp_lbl});E_{{#mu2}} [GeV];Events", "mu2_e", sp_node, 60, 0.0, 3.0)
        book_1d(f"h_species_{sp_key}_dimuon_delta_e", f"Muon Energy Difference E_{{#mu1}} - E_{{#mu2}} ({sp_lbl});E_{{#mu1}} - E_{{#mu2}} [GeV];Events", "dimuon_delta_e", sp_node, 50, -50.0, 600.0)
        book_1d(f"h_species_{sp_key}_charm_p", f"{sp_lbl} Momentum;p_{{charm}} [GeV/c];Events", "charm_p", sp_node, 50, 0.0, 500.0)
        book_1d(f"h_species_{sp_key}_decay_length", f"{sp_lbl} 3D Flight Length;L_{{3D}} [cm];Events", "decay_length_3d", sp_node, 50, 0.0, 5.0)
        book_1d(f"h_species_{sp_key}_lifetime", f"{sp_lbl} Proper Lifetime;#tau [ps];Events", "proper_lifetime_ps", sp_node, 50, 0.0, 3.0)
        book_1d(f"h_species_{sp_key}_ip3d", f"Decay #mu_{{2}} IP_{{3D}} ({sp_lbl});IP_{{3D}} [cm];Events", "mu2_ip3d", sp_node, 50, 0.0, 0.3)
        book_1d(f"h_species_{sp_key}_dimuon_mass", f"Dimuon Invariant Mass ({sp_lbl});M_{{#mu#mu}} [GeV/c^{{2}}];Events", "dimuon_mass", sp_node, 50, 0.0, 8.0)

    # --------------------------------------------------------------------------
    # 8e. Preselection and Cutflow Stage Distributions
    # --------------------------------------------------------------------------
    cutflow_stages = [
        ("all", df_truth, "All Events"),
        ("numu_cc", df_numu_cc, "All Nu_mu CC"),
        ("fiducial", df_fiducial, "Fiducial Nu_mu CC"),
        ("charm", df_charm, "Nu_mu CC with Charm"),
        ("candidate", df_candidate, "Dimuon Candidate"),
    ]

    for cf_key, cf_node, cf_lbl in cutflow_stages:
        book_1d(f"h_cutflow_{cf_key}_nu_e", f"Incoming #nu_{{#mu}} Energy ({cf_lbl});E_{{#nu}} [GeV];Events", "nu_e", cf_node, 50, 0.0, 1000.0)
        book_1d(f"h_cutflow_{cf_key}_vtx_z", f"Vertex Z Position ({cf_lbl});z_{{vtx}} [cm];Events", "vtx_z", cf_node, 50, 260.0, 360.0)

    # --------------------------------------------------------------------------
    # 8f. Dynamic Histograms from Config (Custom additions preserved)
    # --------------------------------------------------------------------------
    for h_cfg in cfg.get("histograms", []):
        h_name = h_cfg["name"]
        if h_name in booked_histograms_1d:
            continue
        h_title = h_cfg.get("title", h_name)
        col = h_cfg["column"]
        weight_col = h_cfg.get("weight")
        filt_key = h_cfg.get("filter", "candidate").lower()
        node = filter_nodes.get(filt_key, df_candidate)

        if "bin_edges" in h_cfg and h_cfg["bin_edges"]:
            edges = array.array("d", h_cfg["bin_edges"])
            model = ROOT.RDF.TH1DModel(h_name, h_title, len(edges) - 1, edges)
        else:
            nbins = int(h_cfg.get("bins", 50))
            xmin = float(h_cfg.get("xmin", 0.0))
            xmax = float(h_cfg.get("xmax", 1.0))
            model = ROOT.RDF.TH1DModel(h_name, h_title, nbins, xmin, xmax)

        if weight_col:
            booked_histograms_1d[h_name] = node.Histo1D(model, col, weight_col)
        else:
            booked_histograms_1d[h_name] = node.Histo1D(model, col)

    for h_cfg in cfg.get("histograms_2d", []):
        h_name = h_cfg["name"]
        if h_name in booked_histograms_2d:
            continue
        h_title = h_cfg.get("title", h_name)
        x_col = h_cfg["x_column"]
        y_col = h_cfg["y_column"]
        weight_col = h_cfg.get("weight")
        filt_key = h_cfg.get("filter", "candidate").lower()
        node = filter_nodes.get(filt_key, df_candidate)

        nbins_x = int(h_cfg.get("bins_x", 50))
        xmin = float(h_cfg.get("xmin", 0.0))
        xmax = float(h_cfg.get("xmax", 1.0))
        nbins_y = int(h_cfg.get("bins_y", 50))
        ymin = float(h_cfg.get("ymin", 0.0))
        ymax = float(h_cfg.get("ymax", 1.0))

        if "x_bin_edges" in h_cfg and "y_bin_edges" in h_cfg:
            x_edges = array.array("d", h_cfg["x_bin_edges"])
            y_edges = array.array("d", h_cfg["y_bin_edges"])
            model = ROOT.RDF.TH2DModel(h_name, h_title, len(x_edges) - 1, x_edges, len(y_edges) - 1, y_edges)
        else:
            model = ROOT.RDF.TH2DModel(h_name, h_title, nbins_x, xmin, xmax, nbins_y, ymin, ymax)

        if weight_col:
            booked_histograms_2d[h_name] = node.Histo2D(model, x_col, y_col, weight_col)
        else:
            booked_histograms_2d[h_name] = node.Histo2D(model, x_col, y_col)

    for p_cfg in cfg.get("profiles", []):
        p_name = p_cfg["name"]
        if p_name in booked_profiles:
            continue
        p_title = p_cfg.get("title", p_name)
        x_col = p_cfg["x_column"]
        y_col = p_cfg["y_column"]
        weight_col = p_cfg.get("weight")
        filt_key = p_cfg.get("filter", "candidate").lower()
        node = filter_nodes.get(filt_key, df_candidate)

        nbins = int(p_cfg.get("bins", 40))
        xmin = float(p_cfg.get("xmin", 0.0))
        xmax = float(p_cfg.get("xmax", 1.0))
        ymin = float(p_cfg["ymin"]) if "ymin" in p_cfg else None
        ymax = float(p_cfg["ymax"]) if "ymax" in p_cfg else None

        if "bin_edges" in p_cfg and p_cfg["bin_edges"]:
            edges = array.array("d", p_cfg["bin_edges"])
            model = ROOT.RDF.TProfile1DModel(p_name, p_title, len(edges) - 1, edges)
        elif ymin is not None and ymax is not None:
            model = ROOT.RDF.TProfile1DModel(p_name, p_title, nbins, xmin, xmax, ymin, ymax)
        else:
            model = ROOT.RDF.TProfile1DModel(p_name, p_title, nbins, xmin, xmax)

        if weight_col:
            booked_profiles[p_name] = node.Profile1D(model, x_col, y_col, weight_col)
        else:
            booked_profiles[p_name] = node.Profile1D(model, x_col, y_col)

    # 9. Execute Computation Graph in Single Pass
    all_actions = list(booked_counters)
    all_actions.extend(booked_histograms_1d.values())
    all_actions.extend(booked_histograms_2d.values())
    all_actions.extend(booked_profiles.values())

    n_total_plots = len(booked_histograms_1d) + len(booked_histograms_2d) + len(booked_profiles)
    print(f"\nRunning RDataFrame computation graph over {n_total_plots} booked objects "
          f"({len(booked_histograms_1d)} 1D, {len(booked_histograms_2d)} 2D, {len(booked_profiles)} Profiles)...")
    ROOT.RDF.RunGraphs(all_actions)

    # 10. Print Comprehensive Cutflow, Acceptance, and Kinematic Summary
    n_tot = c_total.GetValue()
    n_numu = c_numu_cc.GetValue()
    n_fid = c_fiducial.GetValue()
    n_chm = c_charm.GetValue()
    n_cand = c_candidate.GetValue()
    n_in_acc = c_in_acc.GetValue()
    n_out_acc = c_out_acc.GetValue()
    n_mu1_acc = c_mu1_in_acc.GetValue()
    n_mu2_acc = c_mu2_in_acc.GetValue()
    n_d0 = c_d0.GetValue()
    n_dplus = c_dplus.GetValue()
    n_ds = c_ds.GetValue()
    n_lambdac = c_lambdac.GetValue()

    print("\n" + "=" * 80)
    print(" EVENT SUMMARY & CUTFLOW")
    print("=" * 80)
    print(f"Total Events Processed       : {n_tot:8d}")
    print(f"Muon Neutrino CC             : {n_numu:8d} ({100.0 * n_numu / max(n_tot, 1):6.2f}%)")
    print(f"Target Fiducial Volume CC    : {n_fid:8d} ({100.0 * n_fid / max(n_numu, 1):6.2f}% of CC)")
    print(f"Nu_mu CC with Charm Hadron   : {n_chm:8d} ({100.0 * n_chm / max(n_numu, 1):6.2f}% of CC)")
    print(f"Opposite-Sign Dimuon Signal  : {n_cand:8d} ({100.0 * n_cand / max(n_chm, 1):6.2f}% of charm)")
    print("-" * 80)
    print(" DOWNSTREAM (DS) MUFILTER ACCEPTANCE BREAKDOWN FOR SIGNAL DIMUONS")
    print("-" * 80)
    print(f"Both Muons in DS Acceptance  : {n_in_acc:8d} ({100.0 * n_in_acc / max(n_cand, 1):6.2f}% of signal)")
    print(f"Outside DS Acceptance        : {n_out_acc:8d} ({100.0 * n_out_acc / max(n_cand, 1):6.2f}% of signal)")
    print(f"Primary Muon (mu1) in DS     : {n_mu1_acc:8d} ({100.0 * n_mu1_acc / max(n_cand, 1):6.2f}% of signal)")
    print(f"Charm Decay Muon (mu2) in DS : {n_mu2_acc:8d} ({100.0 * n_mu2_acc / max(n_cand, 1):6.2f}% of signal)")
    print("-" * 80)
    print(" CHARMED HADRON SPECIES IN SIGNAL SAMPLE")
    print("-" * 80)
    print(f"D0 / anti-D0                 : {n_d0:8d} ({100.0 * n_d0 / max(n_cand, 1):6.2f}%)")
    print(f"D+ / D-                      : {n_dplus:8d} ({100.0 * n_dplus / max(n_cand, 1):6.2f}%)")
    print(f"Ds+ / Ds-                    : {n_ds:8d} ({100.0 * n_ds / max(n_cand, 1):6.2f}%)")
    print(f"Lambda_c+ / anti-Lambda_c-   : {n_lambdac:8d} ({100.0 * n_lambdac / max(n_cand, 1):6.2f}%)")
    print("=" * 80)

    # Print Key Kinematic Means Comparison (Inclusive vs In-Acceptance vs Outside-Acceptance)
    print("\n" + "=" * 80)
    print(" KINEMATIC COMPARISON: INCLUSIVE vs IN-ACCEPTANCE vs OUTSIDE-ACCEPTANCE")
    print("=" * 80)
    print(f"{'Observable':<32} | {'Inclusive':>14} | {'In-Acceptance':>14} | {'Outside-Acc':>14}")
    print("-" * 80)

    key_obs = [
        ("Primary Muon Energy E(mu1) [GeV]", "h_signal_mu1_e", "h_in_acc_mu1_e", "h_out_acc_mu1_e"),
        ("Decay Muon Energy E(mu2) [GeV]", "h_signal_mu2_e", "h_in_acc_mu2_e", "h_out_acc_mu2_e"),
        ("Primary Muon Momentum p(mu1) [GeV/c]", "h_signal_mu1_p", "h_in_acc_mu1_p", "h_out_acc_mu1_p"),
        ("Decay Muon Momentum p(mu2) [GeV/c]", "h_signal_mu2_p", "h_in_acc_mu2_p", "h_out_acc_mu2_p"),
        ("Primary Muon pT [GeV/c]", "h_signal_mu1_pt", "h_in_acc_mu1_pt", "h_out_acc_mu1_pt"),
        ("Decay Muon pT [GeV/c]", "h_signal_mu2_pt", "h_in_acc_mu2_pt", "h_out_acc_mu2_pt"),
        ("Primary Muon Pseudorapidity", "h_signal_mu1_eta", "h_in_acc_mu1_eta", "h_out_acc_mu1_eta"),
        ("Decay Muon Pseudorapidity", "h_signal_mu2_eta", "h_in_acc_mu2_eta", "h_out_acc_mu2_eta"),
        ("Dimuon Invariant Mass [GeV/c^2]", "h_signal_dimuon_mass", "h_in_acc_dimuon_mass", "h_out_acc_dimuon_mass"),
        ("Dimuon Opening Angle [mrad]", "h_signal_dimuon_opening_angle_mrad", "h_in_acc_dimuon_opening_angle_mrad", "h_out_acc_dimuon_opening_angle_mrad"),
        ("Dimuon Energy Sum [GeV]", "h_signal_dimuon_e_sum", "h_in_acc_dimuon_e_sum", "h_out_acc_dimuon_e_sum"),
        ("Dimuon Energy Diff [GeV]", "h_signal_dimuon_delta_e", "h_in_acc_dimuon_delta_e", "h_out_acc_dimuon_delta_e"),
        ("Dimuon Energy Asymmetry", "h_signal_dimuon_energy_asym", "h_in_acc_dimuon_energy_asym", "h_out_acc_dimuon_energy_asym"),
        ("Neutrino Energy E(nu) [GeV]", "h_signal_nu_e", "h_in_acc_nu_e", "h_out_acc_nu_e"),
        ("Charm 3D Flight Length [cm]", "h_signal_decay_length_3d", "h_in_acc_decay_length_3d", "h_out_acc_decay_length_3d"),
        ("Decay Muon IP_3D [cm]", "h_signal_mu2_ip3d", "h_in_acc_mu2_ip3d", "h_out_acc_mu2_ip3d"),
    ]

    for label_str, h_inc_name, h_in_name, h_out_name in key_obs:
        m_inc = booked_histograms_1d[h_inc_name].GetValue().GetMean() if h_inc_name in booked_histograms_1d else 0.0
        m_in = booked_histograms_1d[h_in_name].GetValue().GetMean() if h_in_name in booked_histograms_1d else 0.0
        m_out = booked_histograms_1d[h_out_name].GetValue().GetMean() if h_out_name in booked_histograms_1d else 0.0
        print(f"{label_str:<32} | {m_inc:14.4f} | {m_in:14.4f} | {m_out:14.4f}")
    print("=" * 80)

    # 11. Save Histograms & Profiles into Hierarchical Output ROOT File
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_file = ROOT.TFile.Open(output_file, "RECREATE")
    if out_file and not out_file.IsZombie():
        out_file.cd()

        # 11a. Write top-level key signal overview histograms
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_signal_"):
                h_res.GetValue().Write()
        for name, p_res in booked_profiles.items():
            if name.startswith("p_acc_eff_"):
                p_res.GetValue().Write()

        # 11b. Hierarchical structured subdirectories for TBrowser & rootls
        d_sig = out_file.mkdir("signal_inclusive")
        d_in = out_file.mkdir("in_acceptance")
        d_out = out_file.mkdir("outside_acceptance")
        d_comp = out_file.mkdir("acceptance_comparison")
        d_2d = out_file.mkdir("correlations_2d")
        d_prof = out_file.mkdir("profiles")
        d_spec = out_file.mkdir("species")
        d_cf = out_file.mkdir("cutflow")

        # Flat subdirectories for backward compatibility
        d_all_1d = out_file.mkdir("histograms_1d")
        d_all_2d = out_file.mkdir("histograms_2d")

        # Populate signal_inclusive
        d_sig.cd()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_signal_"):
                h_res.GetValue().Write()

        # Populate in_acceptance
        d_in.cd()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_in_acc_"):
                h_res.GetValue().Write()

        # Populate outside_acceptance
        d_out.cd()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_out_acc_"):
                h_res.GetValue().Write()

        # Populate acceptance_comparison
        d_comp.cd()
        for name, p_res in booked_profiles.items():
            if "acc" in name:
                p_res.GetValue().Write()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_in_acc_") or name.startswith("h_out_acc_"):
                h_res.GetValue().Write()

        # Populate correlations_2d
        d_2d.cd()
        for h_res in booked_histograms_2d.values():
            h_res.GetValue().Write()

        # Populate profiles
        d_prof.cd()
        for p_res in booked_profiles.values():
            p_res.GetValue().Write()

        # Populate species
        d_spec.cd()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_species_"):
                h_res.GetValue().Write()

        # Populate cutflow
        d_cf.cd()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_cutflow_"):
                h_res.GetValue().Write()

        # Populate all 1D & 2D
        d_all_1d.cd()
        for h_res in booked_histograms_1d.values():
            h_res.GetValue().Write()

        d_all_2d.cd()
        for h_res in booked_histograms_2d.values():
            h_res.GetValue().Write()

        out_file.Close()
        print(f"\nSaved {n_total_plots} objects ({len(booked_histograms_1d)} 1D, {len(booked_histograms_2d)} 2D, {len(booked_profiles)} Profiles) to: {output_file}")

    print("\nTruth analysis and histogram generation completed successfully!")


if __name__ == "__main__":
    main()
