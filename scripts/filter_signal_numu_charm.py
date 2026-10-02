#!/usr/bin/env python3
"""
filter_signal_numu_charm.py
---------------------------
Filters simulated neutrino interaction files to extract golden signal events:
  nu_mu CC -> charmed hadron -> prompt muon decay (dimuon final state).

Features:
- Processes input files 1-to-1: N input files produce N output files.
- Stores the filtered events containing only signal candidates.
- Preserves 100% of the original cbmsim / rawConv tree structure (MCTracks,
  SciFi hits/clusters, MuFilter hits, etc.).
- Adds dedicated truth branches (nu_e, mu1_p, mu2_p, charm_pdg, dimuon_mass,
  decay_length_3d, etc.) and the full MuonNeutrinoTruthInfo object.
- Stores diagnostic histograms inside each filtered output ROOT file.
- Analysis cuts, input patterns, and histogram definitions are configured via
  ./config/filter_numu_charm_config.yaml.

CLI Arguments:
  -n, --entries : Maximum events per file to evaluate (-1 for full sample)
  -o, --output  : Output directory or base filename (overrides config output)
  -j, --threads : Number of worker threads for RDataFrame per file
  -c, --config  : Path to custom YAML configuration file
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter and skim signal nu_mu CC charm dimuon events from SND@LHC MC"
    )
    parser.add_argument(
        "-n", "--entries",
        type=int,
        default=-1,
        help="Max events per file to process (-1 for all events)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (or base filename) for filtered files"
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        default=None,
        help="Number of worker threads (>1 enables ROOT ImplicitMT)"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: config/filter_numu_charm_config.yaml)"
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


def determine_output_path(input_path: str, output_arg: str, output_cfg: dict, index: int, total_files: int) -> str:
    """
    Determine the 1-to-1 output file path corresponding to an input file.
    Preserves directory partition structure (e.g. output_dir/1/filename.root).
    """
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
            # If a single .root filename is given but multiple input files exist, append partition/index
            out_stem, out_ext = os.path.splitext(output_arg)
            partition_tag = parent_partition if (parent_partition and parent_partition != ".") else str(index + 1)
            return f"{out_stem}_{partition_tag}{out_ext}"
        else:
            base_out_dir = output_arg
    else:
        base_out_dir = default_dir

    if preserve_subdirs and parent_partition and parent_partition != ".":
        target_dir = os.path.join(base_out_dir, parent_partition)
    else:
        target_dir = base_out_dir

    os.makedirs(target_dir, exist_ok=True)
    out_name = f"{base_stem}{file_suffix}{ext}"
    return os.path.join(target_dir, out_name)


def setup_truth_branches(out_tree: ROOT.TTree, truth_info_obj: object):
    """
    Attach the complete MuonNeutrinoTruthInfo struct as well as individual scalar
    branches to the output cloned tree for fast inspection in ROOT / TTree::Draw.
    """
    out_tree.Branch("truth", truth_info_obj)

    branch_buffers = {
        # Int flags
        "is_cc": array.array('i', [0]),
        "is_numu_cc": array.array('i', [0]),
        "is_anti_numu_cc": array.array('i', [0]),
        "is_fiducial": array.array('i', [0]),
        "has_charm": array.array('i', [0]),
        "has_prompt_charm_muon": array.array('i', [0]),
        "is_opposite_sign": array.array('i', [0]),
        "has_candidate": array.array('i', [0]),
        "charm_pdg": array.array('i', [0]),
        "abs_charm_pdg": array.array('i', [0]),
        "charm_species_id": array.array('i', [0]),
        # Double observables
        "mc_weight": array.array('d', [0.0]),
        "raw_weight": array.array('d', [0.0]),
        "nu_e": array.array('d', [0.0]),
        "vtx_x": array.array('d', [0.0]),
        "vtx_y": array.array('d', [0.0]),
        "vtx_z": array.array('d', [0.0]),
        "q2": array.array('d', [0.0]),
        "bjorken_x": array.array('d', [0.0]),
        "inelasticity_y": array.array('d', [0.0]),
        "hadronic_w": array.array('d', [0.0]),
        "mu1_p": array.array('d', [0.0]),
        "mu1_pt": array.array('d', [0.0]),
        "mu1_eta": array.array('d', [0.0]),
        "mu1_phi": array.array('d', [0.0]),
        "charm_p": array.array('d', [0.0]),
        "charm_pt": array.array('d', [0.0]),
        "charm_e": array.array('d', [0.0]),
        "charm_z_frac": array.array('d', [0.0]),
        "decay_length_3d": array.array('d', [0.0]),
        "proper_time_ctau": array.array('d', [0.0]),
        "proper_lifetime_ps": array.array('d', [0.0]),
        "mu2_p": array.array('d', [0.0]),
        "mu2_pt": array.array('d', [0.0]),
        "mu2_eta": array.array('d', [0.0]),
        "mu2_phi": array.array('d', [0.0]),
        "mu2_ip3d": array.array('d', [0.0]),
        "mu2_ipxy": array.array('d', [0.0]),
        "mu2_ptrel": array.array('d', [0.0]),
        "dimuon_mass": array.array('d', [0.0]),
        "dimuon_pt": array.array('d', [0.0]),
        "dimuon_opening_angle_mrad": array.array('d', [0.0]),
        "dimuon_delta_phi": array.array('d', [0.0]),
        "dimuon_energy_asym": array.array('d', [0.0]),
        "dimuon_p_ratio": array.array('d', [0.0]),
    }

    for bname, buf in branch_buffers.items():
        type_str = "I" if buf.typecode == 'i' else "D"
        out_tree.Branch(bname, buf, f"{bname}/{type_str}")

    return branch_buffers


def fill_truth_buffers(buffers: dict, info: object):
    """Update scalar branch buffers from truth info struct."""
    buffers["is_cc"][0] = int(info.isCC)
    buffers["is_numu_cc"][0] = int(info.isNuMuCC)
    buffers["is_anti_numu_cc"][0] = int(info.isAntiNuMuCC)
    buffers["is_fiducial"][0] = int(info.isFiducial)
    buffers["has_charm"][0] = int(info.hasCharm)
    buffers["has_prompt_charm_muon"][0] = int(info.hasPromptCharmMuon)
    buffers["is_opposite_sign"][0] = int(info.isOppositeSignDimuon)
    buffers["has_candidate"][0] = int(info.hasCandidate)
    buffers["charm_pdg"][0] = info.charmPdg
    buffers["abs_charm_pdg"][0] = abs(info.charmPdg)

    # Species ID: 1: D0, 2: D+, 3: Ds+, 4: Lambda_c+, 5: other baryon, 6: other meson
    pdg = abs(info.charmPdg)
    if pdg == 421: sp_id = 1
    elif pdg == 411: sp_id = 2
    elif pdg == 431: sp_id = 3
    elif pdg == 4122: sp_id = 4
    elif pdg > 4000: sp_id = 5
    elif pdg > 0: sp_id = 6
    else: sp_id = 0
    buffers["charm_species_id"][0] = sp_id

    buffers["mc_weight"][0] = info.mcWeight
    buffers["raw_weight"][0] = info.rawWeight
    buffers["nu_e"][0] = info.nuE
    buffers["vtx_x"][0] = info.vtxX
    buffers["vtx_y"][0] = info.vtxY
    buffers["vtx_z"][0] = info.vtxZ
    buffers["q2"][0] = info.Q2
    buffers["bjorken_x"][0] = info.BjorkenX
    buffers["inelasticity_y"][0] = info.InelasticityY
    buffers["hadronic_w"][0] = info.HadronicW
    buffers["mu1_p"][0] = info.mu1P
    buffers["mu1_pt"][0] = info.mu1Pt
    buffers["mu1_eta"][0] = info.mu1Eta
    buffers["mu1_phi"][0] = info.mu1Phi
    buffers["charm_p"][0] = info.charmP
    buffers["charm_pt"][0] = info.charmPt
    buffers["charm_e"][0] = info.charmE
    buffers["charm_z_frac"][0] = info.charmEnergyFractionZ
    buffers["decay_length_3d"][0] = info.decayLength3D
    buffers["proper_time_ctau"][0] = info.properDecayTimeCTau
    buffers["proper_lifetime_ps"][0] = info.properLifetimePs
    buffers["mu2_p"][0] = info.mu2P
    buffers["mu2_pt"][0] = info.mu2Pt
    buffers["mu2_eta"][0] = info.mu2Eta
    buffers["mu2_phi"][0] = info.mu2Phi
    buffers["mu2_ip3d"][0] = info.mu2IP3D
    buffers["mu2_ipxy"][0] = info.mu2IPXY
    buffers["mu2_ptrel"][0] = info.mu2PtRel
    buffers["dimuon_mass"][0] = info.dimuonInvMass
    buffers["dimuon_pt"][0] = info.dimuonPt
    buffers["dimuon_opening_angle_mrad"][0] = info.dimuonOpeningAngleMrad
    buffers["dimuon_delta_phi"][0] = info.dimuonDeltaPhi
    buffers["dimuon_energy_asym"][0] = info.dimuonEnergyAsymmetry
    buffers["dimuon_p_ratio"][0] = info.dimuonMomentumRatio


def process_single_file(
    input_file: str,
    output_file: str,
    processor: ROOT.snd.MuonNeutrinoTruthProcessor,
    cfg: dict,
    max_entries: int = -1,
    effective_threads: int = 1
) -> dict:
    """
    Process a single input file:
    1. Runs RDataFrame to evaluate truth observables, cutflow, and diagnostic histograms.
    2. Identifies matching signal entry numbers.
    3. Clones the original tree structure (cbmsim/rawConv) into the output file.
    4. Populates the signal entries with both original branches and truth branches.
    5. Saves diagnostic histograms into the output file.
    """
    tree_name = cfg.get("input", {}).get("tree_name", "cbmsim")

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

    # 1. RDataFrame setup for fast truth processing and histogramming
    df_raw = ROOT.RDataFrame(actual_tree_name, input_file)
    if max_entries > 0:
        df_raw = df_raw.Range(max_entries)

    # Define truth observables
    df_truth = (
        df_raw.Define("truth", processor, ["MCTrack"])
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
              .Define("nu_e", "truth.nuE")
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

    # Signal filter condition
    filter_cfg = cfg.get("filter", {})
    signal_expr = filter_cfg.get("signal_expression", "has_candidate && is_opposite_sign")
    if filter_cfg.get("require_fiducial", False):
        signal_expr = f"({signal_expr}) && is_fiducial"

    df_signal = df_truth.Filter(signal_expr, "Signal Selection")

    # Book counters & signal entry list
    c_tot = df_truth.Count()
    c_sig = df_signal.Count()
    signal_entries_rptr = df_signal.Take["ULong64_t"]("rdfentry_")

    # Book diagnostic histograms on signal events
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
            booked_histograms[h_name] = df_signal.Histo1D(model, col, w_col)
        else:
            booked_histograms[h_name] = df_signal.Histo1D(model, col)

    # Run RDF graph
    all_actions = [c_tot, c_sig, signal_entries_rptr]
    all_actions.extend(booked_histograms.values())
    ROOT.RDF.RunGraphs(all_actions)

    n_tot = c_tot.GetValue()
    n_sig = c_sig.GetValue()
    signal_entries = list(signal_entries_rptr.GetValue())

    # 2. Clone original tree and store only signal events + truth branches
    f_in = ROOT.TFile.Open(input_file)
    t_in = f_in.Get(actual_tree_name)

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    f_out = ROOT.TFile.Open(output_file, "RECREATE")
    if not f_out or f_out.IsZombie():
        f_in.Close()
        return {"total": n_tot, "signal": n_sig, "status": "error_create_out"}

    # Clone empty tree structure (preserves all original branches: MCTrack, SciFi, MuFilter, etc.)
    out_tree = t_in.CloneTree(0)
    out_tree.SetName(actual_tree_name)

    # Attach truth struct and scalar branches
    truth_info_obj = ROOT.snd.MuonNeutrinoTruthInfo()
    truth_buffers = setup_truth_branches(out_tree, truth_info_obj)

    # Fill signal entries
    for entry_idx in signal_entries:
        t_in.GetEntry(entry_idx)
        # Evaluate truth for this entry
        info = processor.processMuonNeutrino(t_in.MCTrack)
        truth_info_obj = info
        fill_truth_buffers(truth_buffers, info)
        out_tree.Fill()

    f_out.cd()
    out_tree.Write()

    # Write diagnostic histograms
    for h_name, h_result in booked_histograms.items():
        h = h_result.GetValue()
        h.Write()

    f_out.Close()
    f_in.Close()

    return {
        "total": n_tot,
        "signal": n_sig,
        "status": "success",
        "output_file": output_file
    }


def main():
    args = parse_arguments()

    # Load libraries so MuonNeutrinoTruthProcessor and data classes are available
    load_trident_libraries(_repo_root)

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
    perf_cfg = cfg.get("performance", {})
    n_threads = args.threads if args.threads is not None else int(perf_cfg.get("threads", 1))

    # Configure ROOT Multi-Threading
    effective_threads = n_threads
    if n_threads > 1:
        if args.entries > 0:
            print(f"[Notice] Multi-threading (-j {n_threads}) cannot be combined with event Range (-n {args.entries}) in ROOT RDataFrame.")
            print("         Running sequentially in single-threaded mode to preserve the -n range limit.")
            effective_threads = 0
        else:
            ROOT.EnableImplicitMT(n_threads)
            print(f"Enabled ROOT Implicit Multi-Threading with {n_threads} worker threads.")

    # 2. Resolve Input Files
    input_files = resolve_input_files(file_pattern, max_files=max_files)
    total_files = len(input_files)

    # 3. Configure Processor
    proc_cfg = cfg.get("processor", {})
    processor = build_processor(proc_cfg)

    print("=" * 80)
    print(" SND@LHC: Skimming Nu_Mu CC Charm Dimuon Signal Events")
    print("=" * 80)
    print(f"Total input files to process : {total_files}")
    print(f"Max events per file          : {args.entries if args.entries > 0 else 'All'}")
    print(f"Signal filter expression     : {cfg.get('filter', {}).get('signal_expression')}")
    print(f"Worker threads               : {n_threads}")
    print("=" * 80)

    # 4. Process each file 1-to-1
    results = []
    tot_events_all = 0
    tot_signal_all = 0

    for idx, in_file in enumerate(input_files):
        out_file = determine_output_path(in_file, args.output, output_cfg, idx, total_files)
        print(f"\n[{idx + 1}/{total_files}] Processing:\n  Input : {in_file}\n  Output: {out_file}")

        res = process_single_file(
            input_file=in_file,
            output_file=out_file,
            processor=processor,
            cfg=cfg,
            max_entries=args.entries,
            effective_threads=effective_threads
        )
        results.append(res)

        n_tot = res.get("total", 0)
        n_sig = res.get("signal", 0)
        tot_events_all += n_tot
        tot_signal_all += n_sig

        pct = (100.0 * n_sig / max(n_tot, 1)) if n_tot > 0 else 0.0
        print(f"  Summary: {n_sig} signal events extracted out of {n_tot} total ({pct:.2f}%)")

    # 5. Final Processing Summary Table
    print("\n" + "=" * 80)
    print(" OVERALL PROCESSING SUMMARY")
    print("=" * 80)
    print(f"{'#':<3} | {'Partition':<12} | {'Total':>8} | {'Signal':>8} | {'Yield (%)':>10} | {'Status':<10}")
    print("-" * 80)
    for idx, (in_file, res) in enumerate(zip(input_files, results)):
        partition = os.path.basename(os.path.dirname(in_file))
        n_tot = res.get("total", 0)
        n_sig = res.get("signal", 0)
        pct = (100.0 * n_sig / max(n_tot, 1)) if n_tot > 0 else 0.0
        status = res.get("status", "unknown")
        print(f"{idx + 1:<3} | {partition:<12} | {n_tot:8d} | {n_sig:8d} | {pct:9.2f}% | {status:<10}")

    print("=" * 80)
    overall_pct = (100.0 * tot_signal_all / max(tot_events_all, 1)) if tot_events_all > 0 else 0.0
    print(f"TOTAL: {tot_signal_all} signal events skimmed out of {tot_events_all} total events ({overall_pct:.2f}%).")
    print(f"Produced {total_files} output file(s).")
    print("=" * 80)
    print("\nDone!")


if __name__ == "__main__":
    main()
