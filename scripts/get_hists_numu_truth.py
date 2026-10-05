#!/usr/bin/env python3
"""
example_muon_neutrino_truth.py
--------------------------------
Example script demonstrating how to use the MuonNeutrinoTruthProcessor
with ROOT's RDataFrame in Python to study charm production from muon neutrino
charged current (nu_mu CC) interactions.

Data loading is managed through snd.DataManager.
Analysis cuts, input patterns, and histogram definitions (including charm
species separation) are outsourced to ./config/muon_neutrino_truth_config.yaml.

CLI Arguments:
  -n, --entries : Number of events to process (-1 for full sample)
  -o, --output  : Output ROOT file (defaults to config default_file)
  -j, --threads : Number of worker threads for ROOT ImplicitMT
"""

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

from snd import DataManager, load_trident_libraries

# Pre-load libraries so ROOT has all custom C++ dictionaries and classes available
load_trident_libraries(_repo_root)

# Disable ROOT GUI windows in batch mode
ROOT.gROOT.SetBatch(True)


def parse_arguments():
    """Parse command line arguments: strictly -n, -o, and -j."""
    parser = argparse.ArgumentParser(
        description="Study charm production in muon neutrino CC interactions with SND@LHC Truth Processor"
    )
    parser.add_argument(
        "-n", "--entries",
        type=int,
        default=-1,
        help="Number of events to process (-1 for all events)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output ROOT file for summary histograms (overrides config output)"
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        default=None,
        help="Number of worker threads (default: 1 or from config; >1 enables ROOT ImplicitMT)"
    )
    return parser.parse_args()


def load_config(config_path: str = None) -> dict:
    """Load analysis configuration from YAML (or JSON fallback)."""
    if config_path is None:
        config_path = os.path.join(_repo_root, "config", "muon_neutrino_truth_config.yaml")

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

    # Fast scan for single wildcard directory patterns /path/to/base/*/filename
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

    # Standard glob fallback
    if not matched_files:
        matched_files = sorted(glob.glob(pattern))

    if not matched_files:
        raise FileNotFoundError(
            f"No files matched input pattern: {pattern}\n"
            "Please check the path and your EOS credentials."
        )

    if max_files > 0:
        matched_files = matched_files[:max_files]

    print(f"Found {len(matched_files)} matching ROOT files.")
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


def main():
    args = parse_arguments()

    # 1. Load Configuration
    cfg = load_config()

    input_cfg = cfg.get("input", {})
    file_pattern = input_cfg.get(
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

    print("=" * 70)
    print(" SND@LHC: Nu_Mu CC Charm Production Truth Exploration")
    print("=" * 70)
    print(f"Config input pattern: {file_pattern}")
    print(f"Tree name           : {tree_name}")
    print(f"Max entries         : {args.entries if args.entries > 0 else 'All'}")
    print(f"Output file         : {output_file}")
    print(f"Threads             : {n_threads}")

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

    # 3. Initialize DataManager for dataset loading
    dm = DataManager(
        source=input_files,
        tree_name=tree_name,
        num_threads=effective_threads,
        n_files=max_files if max_files > 0 else -1,
        load_libraries=True,
    )
    df = dm.rdf(range_limit=args.entries if args.entries > 0 else None)

    # 4. Configure MuonNeutrinoTruthProcessor
    proc_cfg = cfg.get("processor", {})
    processor = build_processor(proc_cfg)

    # 5. Define Truth Observables
    df_truth = (
        df.Define("truth", processor, ["MCTrack"])
          # Event and selection flags
          .Define("is_cc", "truth.isCC")
          .Define("is_numu_cc", "truth.isNuMuCC")
          .Define("is_anti_numu_cc", "truth.isAntiNuMuCC")
          .Define("is_fiducial", "truth.isFiducial")
          .Define("has_charm", "truth.hasCharm")
          .Define("has_prompt_charm_muon", "truth.hasPromptCharmMuon")
          .Define("is_opposite_sign", "truth.isOppositeSignDimuon")
          .Define("has_candidate", "truth.hasCandidate")
          .Define("mc_weight", "truth.mcWeight")
          .Define("raw_weight", "truth.rawWeight")
          # Primary incoming neutrino
          .Define("nu_e", "truth.nuE")
          .Define("vtx_x", "truth.vtxX")
          .Define("vtx_y", "truth.vtxY")
          .Define("vtx_z", "truth.vtxZ")
          # DIS Kinematics
          .Define("q2", "truth.Q2")
          .Define("bjorken_x", "truth.BjorkenX")
          .Define("inelasticity_y", "truth.InelasticityY")
          .Define("hadronic_w", "truth.HadronicW")
          # Primary muon (mu_1)
          .Define("mu1_p", "truth.mu1P")
          .Define("mu1_pt", "truth.mu1Pt")
          .Define("mu1_eta", "truth.mu1Eta")
          .Define("mu1_phi", "truth.mu1Phi")
          # Charmed hadron species and kinematics
          .Define("charm_pdg", "truth.charmPdg")
          .Define("abs_charm_pdg", "std::abs(truth.charmPdg)")
          .Define("charm_p", "truth.charmP")
          .Define("charm_pt", "truth.charmPt")
          .Define("charm_e", "truth.charmE")
          .Define("charm_z_frac", "truth.charmEnergyFractionZ")
          .Define("decay_length_3d", "truth.decayLength3D")
          .Define("proper_time_ctau", "truth.properDecayTimeCTau")
          .Define("proper_lifetime_ps", "truth.properLifetimePs")
          # Species categorization ID:
          # 1: D0, 2: D+, 3: D_s+, 4: Lambda_c+, 5: other baryon, 6: other meson
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
          # Secondary decay muon (mu_2)
          .Define("mu2_p", "truth.mu2P")
          .Define("mu2_pt", "truth.mu2Pt")
          .Define("mu2_eta", "truth.mu2Eta")
          .Define("mu2_phi", "truth.mu2Phi")
          .Define("mu2_ip3d", "truth.mu2IP3D")
          .Define("mu2_ipxy", "truth.mu2IPXY")
          .Define("mu2_ptrel", "truth.mu2PtRel")
          # Dimuon pair composite kinematics
          .Define("dimuon_mass", "truth.dimuonInvMass")
          .Define("dimuon_pt", "truth.dimuonPt")
          .Define("dimuon_opening_angle_mrad", "truth.dimuonOpeningAngleMrad")
          .Define("dimuon_delta_phi", "truth.dimuonDeltaPhi")
          .Define("dimuon_energy_asym", "truth.dimuonEnergyAsymmetry")
          .Define("dimuon_p_ratio", "truth.dimuonMomentumRatio")
    )

    # 6. Define Filtered Subsamples
    df_numu_cc   = df_truth.Filter("is_numu_cc || is_anti_numu_cc", "All Nu_mu CC")
    df_fiducial  = df_numu_cc.Filter("is_fiducial", "Target Fiducial Volume")
    df_charm     = df_fiducial.Filter("has_charm", "Fiducial with Charmed Hadron")
    df_candidate = df_charm.Filter("has_candidate && is_opposite_sign", "Opposite-Sign Dimuon Charm Candidate")

    # Specific charm hadron species filters
    df_d0        = df_charm.Filter("abs_charm_pdg == 421", "D0 / anti-D0")
    df_dplus     = df_charm.Filter("abs_charm_pdg == 411", "D+ / D-")
    df_ds        = df_charm.Filter("abs_charm_pdg == 431", "Ds+ / Ds-")
    df_lambdac   = df_charm.Filter("abs_charm_pdg == 4122", "Lambda_c+ / anti-Lambda_c-")

    filter_nodes = {
        "all": df_truth,
        "numu_cc": df_numu_cc,
        "fiducial": df_fiducial,
        "charm": df_charm,
        "d0": df_d0,
        "dplus": df_dplus,
        "ds": df_ds,
        "lambdac": df_lambdac,
        "candidate": df_candidate,
    }

    # 7. Book Counters (Cutflow & Species Breakdown)
    c_total     = df_truth.Count()
    c_numu_cc   = df_numu_cc.Count()
    c_fiducial  = df_fiducial.Count()
    c_charm     = df_charm.Count()
    c_d0        = df_d0.Count()
    c_dplus     = df_dplus.Count()
    c_ds        = df_ds.Count()
    c_lambdac   = df_lambdac.Count()
    c_candidate = df_candidate.Count()

    # 8. Book Histograms & Profiles dynamically from Config
    # 8a. 1D Histograms
    booked_histograms_1d = {}
    for h_cfg in cfg.get("histograms", []):
        h_name = h_cfg["name"]
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

    # 8b. 2D Histograms
    booked_histograms_2d = {}
    for h_cfg in cfg.get("histograms_2d", []):
        h_name = h_cfg["name"]
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

    # 8c. TProfiles
    booked_profiles = {}
    for p_cfg in cfg.get("profiles", []):
        p_name = p_cfg["name"]
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
    all_actions = [
        c_total, c_numu_cc, c_fiducial, c_charm,
        c_d0, c_dplus, c_ds, c_lambdac, c_candidate
    ]
    all_actions.extend(booked_histograms_1d.values())
    all_actions.extend(booked_histograms_2d.values())
    all_actions.extend(booked_profiles.values())

    n_total_plots = len(booked_histograms_1d) + len(booked_histograms_2d) + len(booked_profiles)
    print(f"\nRunning RDataFrame computation graph over {n_total_plots} booked objects ({len(booked_histograms_1d)} 1D, {len(booked_histograms_2d)} 2D, {len(booked_profiles)} Profiles)...")
    ROOT.RDF.RunGraphs(all_actions)

    # 10. Print Event Cutflow & Charm Species Summary
    n_tot     = c_total.GetValue()
    n_numu    = c_numu_cc.GetValue()
    n_fid     = c_fiducial.GetValue()
    n_chm     = c_charm.GetValue()
    n_d0      = c_d0.GetValue()
    n_dplus   = c_dplus.GetValue()
    n_ds      = c_ds.GetValue()
    n_lambdac = c_lambdac.GetValue()
    n_cand    = c_candidate.GetValue()

    print("\n" + "=" * 75)
    print(" EVENT SUMMARY & CUTFLOW")
    print("=" * 75)
    print(f"Total Events Processed    : {n_tot:8d}")
    print(f"Muon Neutrino CC          : {n_numu:8d} ({100.0 * n_numu / max(n_tot, 1):.2f}%)")
    print(f"Target Fiducial CC        : {n_fid:8d} ({100.0 * n_fid / max(n_numu, 1):.2f}%)")
    print(f"Fiducial CC with Charm    : {n_chm:8d} ({100.0 * n_chm / max(n_fid, 1):.2f}%)")
    print("  └─ Charmed Hadron Species:")
    print(f"     D0 / anti-D0         : {n_d0:8d} ({100.0 * n_d0 / max(n_chm, 1):.2f}% of charm)")
    print(f"     D+ / D-              : {n_dplus:8d} ({100.0 * n_dplus / max(n_chm, 1):.2f}% of charm)")
    print(f"     Ds+ / Ds-            : {n_ds:8d} ({100.0 * n_ds / max(n_chm, 1):.2f}% of charm)")
    print(f"     Lambda_c+ / anti-L_c-: {n_lambdac:8d} ({100.0 * n_lambdac / max(n_chm, 1):.2f}% of charm)")
    print(f"Golden Dimuon Candidates  : {n_cand:8d} ({100.0 * n_cand / max(n_chm, 1):.2f}% of charm)")
    print("=" * 75)

    # Print 1D histogram diagnostics
    if booked_histograms_1d:
        print("\n" + "=" * 75)
        print(f"{'1D Histogram':<30} | {'Entries':>8} | {'Integral':>10} | {'Mean':>10} | {'StdDev':>10}")
        print("-" * 75)
        for name, h_result in booked_histograms_1d.items():
            h = h_result.GetValue()
            print(f"{name:<30} | {h.GetEntries():8.0f} | {h.Integral():10.1f} | {h.GetMean():10.4f} | {h.GetStdDev():10.4f}")
        print("=" * 75)

    # Print 2D histogram diagnostics
    if booked_histograms_2d:
        print("\n" + "=" * 75)
        print(f"{'2D Histogram':<32} | {'Entries':>8} | {'Integral':>10} | {'Mean X':>9} | {'Mean Y':>9}")
        print("-" * 75)
        for name, h_result in booked_histograms_2d.items():
            h = h_result.GetValue()
            print(f"{name:<32} | {h.GetEntries():8.0f} | {h.Integral():10.1f} | {h.GetMean(1):9.4f} | {h.GetMean(2):9.4f}")
        print("=" * 75)

    # Print TProfile diagnostics
    if booked_profiles:
        print("\n" + "=" * 75)
        print(f"{'TProfile':<32} | {'Entries':>8} | {'Mean X':>10} | {'Mean Y':>10}")
        print("-" * 75)
        for name, p_result in booked_profiles.items():
            p = p_result.GetValue()
            print(f"{name:<32} | {p.GetEntries():8.0f} | {p.GetMean(1):10.4f} | {p.GetMean(2):10.4f}")
        print("=" * 75)

    # 11. Save Histograms & Profiles to Output File
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_file = ROOT.TFile.Open(output_file, "RECREATE")
    if out_file and not out_file.IsZombie():
        out_file.cd()

        # Top-level directory (all objects accessible directly)
        for h_result in booked_histograms_1d.values():
            h_result.GetValue().Write()
        for h_result in booked_histograms_2d.values():
            h_result.GetValue().Write()
        for p_result in booked_profiles.values():
            p_result.GetValue().Write()

        # Dedicated subdirectories for structured browsing in TBrowser / rootls
        if booked_histograms_1d:
            d1 = out_file.mkdir("histograms_1d")
            d1.cd()
            for h_result in booked_histograms_1d.values():
                h_result.GetValue().Write()

        if booked_histograms_2d:
            d2 = out_file.mkdir("histograms_2d")
            d2.cd()
            for h_result in booked_histograms_2d.values():
                h_result.GetValue().Write()

        if booked_profiles:
            dp = out_file.mkdir("profiles")
            dp.cd()
            for p_result in booked_profiles.values():
                p_result.GetValue().Write()

        out_file.Close()
        print(f"\nSaved {n_total_plots} objects ({len(booked_histograms_1d)} 1D, {len(booked_histograms_2d)} 2D, {len(booked_profiles)} Profiles) to: {output_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
