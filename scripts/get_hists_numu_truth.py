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


PRESETS = {
    "nu14_2022": {
        "input": "/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/[1-1000]/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root",
        "tree": "truth",
        "description": "2022 Downward 15000fb-1 Nu14 volTarget (truth-analyzed 2MuTrks)",
    }
}

DEFAULT_INPUT_PATTERN = "/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/[1-1000]/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Study charm production and dimuon kinematics in nu_mu CC events with SND@LHC Truth Processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(PRESETS.keys()),
        help="Use a predefined input configuration preset (e.g. nu14_2022)",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Input ROOT file path or pattern (supports wildcard '*', range '[1-1000]', and '%%s' placeholder)",
    )
    parser.add_argument(
        "--tree",
        type=str,
        default=None,
        help="TTree name to load (default: auto-detect 'truth' if available, otherwise 'cbmsim')",
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
        default=os.path.join(_repo_root, "config", "muon_neutrino_truth_config.yaml"),
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
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all", "direct", "indirect"],
        help="Focus category: 'all' (inclusive, direct, and indirect), 'direct', or 'indirect'",
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
    if args.preset:
        preset_info = PRESETS[args.preset]
        file_pattern = preset_info["input"]
        tree_name = args.tree or preset_info.get("tree", "truth")
        print(f"Applying preset '{args.preset}': {preset_info.get('description', '')}")
    else:
        file_pattern = args.input if args.input is not None else input_cfg.get(
            "file_pattern",
            DEFAULT_INPUT_PATTERN,
        )
        tree_name = args.tree if args.tree is not None else input_cfg.get("tree_name", "truth")
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

    # Inspect first file to detect available TTrees and branches
    has_truth_tree = False
    has_mufilter = False
    actual_tree_name = tree_name

    f_test = ROOT.TFile.Open(input_files[0])
    if f_test and not f_test.IsZombie():
        if f_test.Get("truth"):
            has_truth_tree = True
        if not args.tree:
            # Auto-detect best tree: prefer 'cbmsim' if it contains MCTrack to reconstruct full 3D kinematics,
            # or pre-analyzed 'truth' tree if present
            if f_test.Get("cbmsim") and f_test.Get("cbmsim").GetBranch("MCTrack"):
                actual_tree_name = "cbmsim"
            elif has_truth_tree:
                actual_tree_name = "truth"
            elif f_test.Get(tree_name):
                actual_tree_name = tree_name
            else:
                for alt_name in ["cbmsim", "rawConv"]:
                    if f_test.Get(alt_name):
                        actual_tree_name = alt_name
                        break
        t_check = f_test.Get(actual_tree_name)
        if t_check and t_check.GetBranch("MuFilterPoint"):
            has_mufilter = True
        f_test.Close()

    print(f"Loaded TTree       : {actual_tree_name}")
    print(f"Pre-analyzed truth : {has_truth_tree}")
    print(f"MuFilter available : {has_mufilter} (DS acceptance tracking enabled: {has_mufilter or actual_tree_name == 'truth' or actual_tree_name == 'cbmsim'})")

    # Declare C++ helper functions for unified 3D kinematics and reconstructed track projections
    ROOT.gInterpreter.Declare("""
    #include "TClonesArray.h"
    #include "TVector3.h"
    #include "ShipMCTrack.h"
    #include "sndRecoTrack.h"
    #include "FairMCPoint.h"
    #include <cmath>
    #include <algorithm>

    namespace RecoTracksExt {
        inline int count_total_tracks(const TClonesArray& tracks) {
            return tracks.GetEntriesFast();
        }

        inline int count_ds_tracks(const TClonesArray& tracks) {
            int count = 0;
            const int nTracks = tracks.GetEntriesFast();
            for (int i = 0; i < nTracks; ++i) {
                auto* trk = const_cast<sndRecoTrack*>(static_cast<const sndRecoTrack*>(tracks.At(i)));
                if (!trk) continue;
                if (trk->getTrackType() == 13) count++;
            }
            return count;
        }

        struct CrossPoint {
            double z_xz;
            double z_yz;
            double x_cross;
            double y_cross;
            double delta_z;
            double abs_delta_z;
            bool valid_xz;
            bool valid_yz;
        };

        inline CrossPoint get_crossing(const TClonesArray& tracks) {
            CrossPoint res = {-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, false, false};
            const int nTracks = tracks.GetEntriesFast();
            sndRecoTrack* trk1 = nullptr;
            sndRecoTrack* trk2 = nullptr;
            for (int i = 0; i < nTracks; ++i) {
                auto* trk = const_cast<sndRecoTrack*>(static_cast<const sndRecoTrack*>(tracks.At(i)));
                if (!trk || trk->getTrackType() != 13) continue;
                if (!trk1) trk1 = trk;
                else if (!trk2) { trk2 = trk; break; }
            }
            if (!trk1 || !trk2) return res;

            double m1x = trk1->getSlopeXZ();
            double m2x = trk2->getSlopeXZ();
            double m1y = trk1->getSlopeYZ();
            double m2y = trk2->getSlopeYZ();

            TVector3 p1 = trk1->getStart();
            TVector3 p2 = trk2->getStart();

            double x01 = p1.X() - m1x * p1.Z();
            double x02 = p2.X() - m2x * p2.Z();
            double y01 = p1.Y() - m1y * p1.Z();
            double y02 = p2.Y() - m2y * p2.Z();

            if (std::abs(m1x - m2x) > 1e-6) {
                res.z_xz = (x02 - x01) / (m1x - m2x);
                res.x_cross = x01 + m1x * res.z_xz;
                res.valid_xz = true;
            }
            if (std::abs(m1y - m2y) > 1e-6) {
                res.z_yz = (y02 - y01) / (m1y - m2y);
                res.y_cross = y01 + m1y * res.z_yz;
                res.valid_yz = true;
            }
            if (res.valid_xz && res.valid_yz) {
                res.delta_z = res.z_xz - res.z_yz;
                res.abs_delta_z = std::abs(res.delta_z);
            }
            return res;
        }

        inline double cross_z_xz(const TClonesArray& tracks) { return get_crossing(tracks).z_xz; }
        inline double cross_z_yz(const TClonesArray& tracks) { return get_crossing(tracks).z_yz; }
        inline double cross_x(const TClonesArray& tracks) { return get_crossing(tracks).x_cross; }
        inline double cross_y(const TClonesArray& tracks) { return get_crossing(tracks).y_cross; }
        inline double delta_z_cross(const TClonesArray& tracks) { return get_crossing(tracks).delta_z; }
        inline double abs_delta_z_cross(const TClonesArray& tracks) { return get_crossing(tracks).abs_delta_z; }
    }

    namespace TruthExt {
        inline double sec_mu_px(int has_dir, double mu2_px, int ds_id, const TClonesArray& tracks) {
            if (has_dir && (std::abs(mu2_px) > 1e-9 || has_dir == 1)) return mu2_px;
            if (ds_id >= 0 && ds_id < tracks.GetEntriesFast()) {
                auto* trk = static_cast<ShipMCTrack*>(tracks.At(ds_id));
                if (trk) return trk->GetPx();
            }
            return 0.0;
        }
        inline double sec_mu_py(int has_dir, double mu2_py, int ds_id, const TClonesArray& tracks) {
            if (has_dir && (std::abs(mu2_py) > 1e-9 || has_dir == 1)) return mu2_py;
            if (ds_id >= 0 && ds_id < tracks.GetEntriesFast()) {
                auto* trk = static_cast<ShipMCTrack*>(tracks.At(ds_id));
                if (trk) return trk->GetPy();
            }
            return 0.0;
        }
        inline double sec_mu_pz(int has_dir, double mu2_pz, int ds_id, const TClonesArray& tracks) {
            if (has_dir && (std::abs(mu2_pz) > 1e-9 || has_dir == 1)) return mu2_pz;
            if (ds_id >= 0 && ds_id < tracks.GetEntriesFast()) {
                auto* trk = static_cast<ShipMCTrack*>(tracks.At(ds_id));
                if (trk) return trk->GetPz();
            }
            return 0.0;
        }
        inline double sec_mu_eta(int has_dir, double mu2_eta, int ds_id, const TClonesArray& tracks) {
            if (has_dir && (std::abs(mu2_eta) > 1e-9 || has_dir == 1)) return mu2_eta;
            if (ds_id >= 0 && ds_id < tracks.GetEntriesFast()) {
                auto* trk = static_cast<ShipMCTrack*>(tracks.At(ds_id));
                if (trk) {
                    double p = trk->GetP();
                    double pz = trk->GetPz();
                    if (p - pz > 1e-9 && p + pz > 1e-9) return 0.5 * std::log((p + pz) / (p - pz));
                }
            }
            return 0.0;
        }
        inline double sec_mu_phi(int has_dir, double mu2_phi, int ds_id, const TClonesArray& tracks) {
            if (has_dir && (std::abs(mu2_phi) > 1e-9 || has_dir == 1)) return mu2_phi;
            if (ds_id >= 0 && ds_id < tracks.GetEntriesFast()) {
                auto* trk = static_cast<ShipMCTrack*>(tracks.At(ds_id));
                if (trk) return std::atan2(trk->GetPy(), trk->GetPx());
            }
            return 0.0;
        }
        inline double sec_mu_theta(int has_dir, double mu2_theta, int ds_id, const TClonesArray& tracks) {
            if (has_dir && (std::abs(mu2_theta) > 1e-9 || has_dir == 1)) return mu2_theta;
            if (ds_id >= 0 && ds_id < tracks.GetEntriesFast()) {
                auto* trk = static_cast<ShipMCTrack*>(tracks.At(ds_id));
                if (trk) {
                    double p = trk->GetP();
                    double pt = trk->GetPt();
                    return (p > 1e-6) ? std::asin(std::min(1.0, pt / p)) : 0.0;
                }
            }
            return 0.0;
        }
        inline double opening_angle_mrad(double px1, double py1, double pz1, double px2, double py2, double pz2) {
            double p1 = std::sqrt(px1*px1 + py1*py1 + pz1*pz1);
            double p2 = std::sqrt(px2*px2 + py2*py2 + pz2*pz2);
            if (p1 < 1e-6 || p2 < 1e-6) return 0.0;
            double dot = px1*px2 + py1*py2 + pz1*pz2;
            double cos_th = std::max(-1.0, std::min(1.0, dot / (p1 * p2)));
            return std::acos(cos_th) * 1000.0;
        }
        inline double projected_angle_diff_mrad(double px1, double py1, double pz1, double px2, double py2, double pz2, bool is_xz) {
            if (std::abs(pz1) < 1e-9 || std::abs(pz2) < 1e-9) return 0.0;
            double ang1 = is_xz ? std::atan2(px1, pz1) : std::atan2(py1, pz1);
            double ang2 = is_xz ? std::atan2(px2, pz2) : std::atan2(py2, pz2);
            return std::abs(ang1 - ang2) * 1000.0;
        }
        inline double inv_mass(double e1, double px1, double py1, double pz1, double e2, double px2, double py2, double pz2) {
            double etot = e1 + e2;
            double ptot2 = (px1+px2)*(px1+px2) + (py1+py2)*(py1+py2) + (pz1+pz2)*(pz1+pz2);
            double m2 = etot*etot - ptot2;
            return (m2 > 0.0) ? std::sqrt(m2) : 0.0;
        }
        inline double pt(double px1, double py1, double px2, double py2) {
            return std::sqrt((px1+px2)*(px1+px2) + (py1+py2)*(py1+py2));
        }
        inline double delta_phi(double phi1, double phi2) {
            double dphi = std::abs(phi1 - phi2);
            const double pi = 3.14159265358979323846;
            while (dphi > pi) dphi = std::abs(dphi - 2.0 * pi);
            return dphi;
        }
        inline double delta_eta(double eta1, double eta2) {
            return std::abs(eta1 - eta2);
        }
        inline double delta_r(double dphi, double deta) {
            return std::sqrt(dphi*dphi + deta*deta);
        }
        inline bool is_trk_in_ds(int trk_id, const TClonesArray& pts, int min_hor = 3, int min_ver = 3) {
            if (trk_id < 0) return false;
            int nHor = 0, nVer = 0;
            const int nPoints = pts.GetEntriesFast();
            for (int i = 0; i < nPoints; ++i) {
                auto* pt = static_cast<const FairMCPoint*>(pts.At(i));
                if (!pt || pt->GetTrackID() != trk_id) continue;
                int detID = pt->GetDetectorID();
                if ((detID / 10000) == 3) {
                    int bar = detID % 1000;
                    if (bar < 60) nHor++;
                    else nVer++;
                }
            }
            return (nHor >= min_hor && nVer >= min_ver);
        }
        inline int count_ds_points(int trk_id, const TClonesArray& pts) {
            if (trk_id < 0) return 0;
            int nTot = 0;
            const int nPoints = pts.GetEntriesFast();
            for (int i = 0; i < nPoints; ++i) {
                auto* pt = static_cast<const FairMCPoint*>(pts.At(i));
                if (!pt || pt->GetTrackID() != trk_id) continue;
                int detID = pt->GetDetectorID();
                if ((detID / 10000) == 3) nTot++;
            }
            return nTot;
        }
        inline int count_ds_hor_points(int trk_id, const TClonesArray& pts) {
            if (trk_id < 0) return 0;
            int nHor = 0;
            const int nPoints = pts.GetEntriesFast();
            for (int i = 0; i < nPoints; ++i) {
                auto* pt = static_cast<const FairMCPoint*>(pts.At(i));
                if (!pt || pt->GetTrackID() != trk_id) continue;
                int detID = pt->GetDetectorID();
                if ((detID / 10000) == 3 && (detID % 1000 < 60)) nHor++;
            }
            return nHor;
        }
        inline int count_ds_ver_points(int trk_id, const TClonesArray& pts) {
            if (trk_id < 0) return 0;
            int nVer = 0;
            const int nPoints = pts.GetEntriesFast();
            for (int i = 0; i < nPoints; ++i) {
                auto* pt = static_cast<const FairMCPoint*>(pts.At(i));
                if (!pt || pt->GetTrackID() != trk_id) continue;
                int detID = pt->GetDetectorID();
                if ((detID / 10000) == 3 && (detID % 1000 >= 60)) nVer++;
            }
            return nVer;
        }
    }
    """)

    # 3. Configure MuonNeutrinoTruthProcessor identically to filter_signal_numu_charm.py
    proc_cfg = cfg.get("processor", {})
    if args.require_direct_charm_decay is not None:
        proc_cfg["require_direct_charm_decay"] = args.require_direct_charm_decay
    processor = build_processor(proc_cfg)

    # 4. Initialize DataManager for dataset loading
    dm = DataManager(
        source=input_files,
        tree_name=actual_tree_name,
        num_threads=effective_threads,
        n_files=max_files if max_files > 0 else -1,
        load_libraries=True,
    )
    df = dm.rdf(range_limit=args.entries if args.entries > 0 else None)

    # 5. Define Truth Observables
    existing_cols = set(df.GetColumnNames())

    def safe_def(curr_df, col_name, expr):
        if col_name in existing_cols:
            return curr_df
        return curr_df.Define(col_name, expr)

    if actual_tree_name == "truth" or ("nu_e" in existing_cols and "is_numu_cc" in existing_cols):
        # Input TTree already contains pre-computed truth observables!
        df_truth = df
        df_truth = safe_def(df_truth, "mu1_track_id", "primary_lepton_track_id")
        df_truth = safe_def(df_truth, "mu1_pdg", "primary_lepton_pdg")
        df_truth = safe_def(df_truth, "mu1_charge", "primary_lepton_charge")
        df_truth = safe_def(df_truth, "abs_charm_pdg", "std::abs(charm_pdg)")
        df_truth = safe_def(df_truth, "dimuon_e_sum", "mu1_e + mu2_e")
        df_truth = safe_def(df_truth, "dimuon_delta_e", "mu1_e - mu2_e")
        df_truth = safe_def(df_truth, "dimuon_abs_delta_e", "std::abs(mu1_e - mu2_e)")
        df_truth = safe_def(df_truth, "dimuon_delta_p", "mu1_p - mu2_p")
        df_truth = safe_def(df_truth, "dimuon_abs_delta_p", "std::abs(mu1_p - mu2_p)")
        df_truth = safe_def(df_truth, "dimuon_delta_pt", "mu1_pt - mu2_pt")
        df_truth = safe_def(df_truth, "dimuon_e_ratio", "(mu1_e > 1e-6) ? (mu2_e / mu1_e) : 0.0")
        df_truth = safe_def(df_truth, "dimuon_e_frac_nu", "(nu_e > 1e-6) ? (mu1_e + mu2_e) / nu_e : 0.0")
        df_truth = safe_def(df_truth, "mu2_e_frac_charm", "(charm_e > 1e-6) ? mu2_e / charm_e : 0.0")
        df_truth = safe_def(df_truth, "mu1_e_frac_nu", "(nu_e > 1e-6) ? mu1_e / nu_e : 0.0")
        df_truth = safe_def(df_truth, "mu2_e_frac_nu", "(nu_e > 1e-6) ? mu2_e / nu_e : 0.0")
        df_truth = safe_def(df_truth, "dimuon_in_ds_acc_double", "dimuon_in_ds_acceptance ? 1.0 : 0.0")
        df_truth = safe_def(df_truth, "mu1_in_ds_acc_double", "mu1_in_ds_acceptance ? 1.0 : 0.0")
        df_truth = safe_def(df_truth, "mu2_in_ds_acc_double", "mu2_in_ds_acceptance ? 1.0 : 0.0")

        # Fallback defaults for category observables in pre-analyzed truth
        df_truth = safe_def(df_truth, "charm_has_direct_muon", "0")
        df_truth = safe_def(df_truth, "has_charm_hadronic_downstream_muon", "0")
        df_truth = safe_def(df_truth, "downstream_muon_p", "0.0")
        df_truth = safe_def(df_truth, "downstream_muon_pt", "0.0")
        df_truth = safe_def(df_truth, "downstream_muon_mother_pdg", "0")
        df_truth = safe_def(df_truth, "downstream_muon_track_id", "-1")
        df_truth = safe_def(df_truth, "downstream_muon_pdg", "0")

        # Unified secondary muon observables across direct and indirect decays
        df_truth = safe_def(df_truth, "sec_mu_track_id", "(charm_has_direct_muon == 1) ? mu2_track_id : downstream_muon_track_id")
        if has_mufilter or "MuFilterPoint" in existing_cols:
            df_truth = df_truth.Redefine("mu1_in_ds_acceptance", "TruthExt::is_trk_in_ds(mu1_track_id, MuFilterPoint)")
            df_truth = df_truth.Redefine("mu2_in_ds_acceptance", "(charm_has_direct_muon == 1) ? mu2_in_ds_acceptance : TruthExt::is_trk_in_ds(sec_mu_track_id, MuFilterPoint)")
            df_truth = df_truth.Redefine("dimuon_in_ds_acceptance", "mu1_in_ds_acceptance && mu2_in_ds_acceptance")
            df_truth = df_truth.Redefine("dimuon_in_ds_acc_double", "dimuon_in_ds_acceptance ? 1.0 : 0.0")
            df_truth = df_truth.Redefine("mu1_in_ds_acc_double", "mu1_in_ds_acceptance ? 1.0 : 0.0")
            df_truth = df_truth.Redefine("mu2_in_ds_acc_double", "mu2_in_ds_acceptance ? 1.0 : 0.0")
            df_truth = safe_def(df_truth, "sec_mu_in_ds_acceptance", "mu2_in_ds_acceptance")
            df_truth = safe_def(df_truth, "sec_mu_n_ds_points", "(charm_has_direct_muon == 1) ? mu2_n_ds_points : TruthExt::count_ds_points(sec_mu_track_id, MuFilterPoint)")
        else:
            df_truth = safe_def(df_truth, "sec_mu_in_ds_acceptance", "mu2_in_ds_acceptance")
            df_truth = safe_def(df_truth, "sec_mu_n_ds_points", "mu2_n_ds_points")

        df_truth = safe_def(df_truth, "is_direct_charm_decay", "charm_has_direct_muon == 1 ? 1 : 0")
        df_truth = safe_def(df_truth, "is_indirect_charm_decay", "(charm_has_direct_muon == 0 && has_charm_hadronic_downstream_muon == 1) ? 1 : 0")
        df_truth = safe_def(df_truth, "sec_mu_p", "(charm_has_direct_muon == 1 && mu2_p > 0.0) ? mu2_p : downstream_muon_p")
        df_truth = safe_def(df_truth, "sec_mu_pt", "(charm_has_direct_muon == 1 && mu2_pt > 0.0) ? mu2_pt : downstream_muon_pt")
        df_truth = safe_def(df_truth, "sec_mu_e", "(charm_has_direct_muon == 1 && mu2_e > 0.0) ? mu2_e : (downstream_muon_p > 0.0 ? std::sqrt(downstream_muon_p * downstream_muon_p + 0.1056583755 * 0.1056583755) : 0.0)")
        df_truth = safe_def(df_truth, "sec_mu_mother_pdg", "(charm_has_direct_muon == 1) ? mu2_mother_pdg : downstream_muon_mother_pdg")
        df_truth = safe_def(df_truth, "sec_mu_abs_mother_pdg", "std::abs((charm_has_direct_muon == 1) ? mu2_mother_pdg : downstream_muon_mother_pdg)")

        # Full 3D vectors for secondary muon from MCTrack (if present) or prompt mu2 fallback
        if "MCTrack" in existing_cols:
            df_truth = safe_def(df_truth, "sec_mu_px", "TruthExt::sec_mu_px(charm_has_direct_muon, mu2_px, downstream_muon_track_id, MCTrack)")
            df_truth = safe_def(df_truth, "sec_mu_py", "TruthExt::sec_mu_py(charm_has_direct_muon, mu2_py, downstream_muon_track_id, MCTrack)")
            df_truth = safe_def(df_truth, "sec_mu_pz", "TruthExt::sec_mu_pz(charm_has_direct_muon, mu2_pz, downstream_muon_track_id, MCTrack)")
            df_truth = safe_def(df_truth, "sec_mu_eta", "TruthExt::sec_mu_eta(charm_has_direct_muon, mu2_eta, downstream_muon_track_id, MCTrack)")
            df_truth = safe_def(df_truth, "sec_mu_phi", "TruthExt::sec_mu_phi(charm_has_direct_muon, mu2_phi, downstream_muon_track_id, MCTrack)")
            df_truth = safe_def(df_truth, "sec_mu_theta", "TruthExt::sec_mu_theta(charm_has_direct_muon, mu2_theta, downstream_muon_track_id, MCTrack)")
        else:
            df_truth = safe_def(df_truth, "sec_mu_px", "charm_has_direct_muon == 1 ? mu2_px : 0.0")
            df_truth = safe_def(df_truth, "sec_mu_py", "charm_has_direct_muon == 1 ? mu2_py : 0.0")
            df_truth = safe_def(df_truth, "sec_mu_pz", "charm_has_direct_muon == 1 ? mu2_pz : 0.0")
            df_truth = safe_def(df_truth, "sec_mu_eta", "charm_has_direct_muon == 1 ? mu2_eta : 0.0")
            df_truth = safe_def(df_truth, "sec_mu_phi", "charm_has_direct_muon == 1 ? mu2_phi : 0.0")
            df_truth = safe_def(df_truth, "sec_mu_theta", "charm_has_direct_muon == 1 ? mu2_theta : 0.0")

        # Unified composite dimuon kinematics calculated using full 3D vectors
        df_truth = safe_def(df_truth, "dimuon_opening_angle_mrad_unified", "TruthExt::opening_angle_mrad(mu1_px, mu1_py, mu1_pz, sec_mu_px, sec_mu_py, sec_mu_pz)")
        df_truth = safe_def(df_truth, "dimuon_mass_unified", "TruthExt::inv_mass(mu1_e, mu1_px, mu1_py, mu1_pz, sec_mu_e, sec_mu_px, sec_mu_py, sec_mu_pz)")
        df_truth = safe_def(df_truth, "dimuon_pt_unified", "TruthExt::pt(mu1_px, mu1_py, sec_mu_px, sec_mu_py)")
        df_truth = safe_def(df_truth, "dimuon_delta_phi_unified", "TruthExt::delta_phi(mu1_phi, sec_mu_phi)")
        df_truth = safe_def(df_truth, "dimuon_delta_eta_unified", "TruthExt::delta_eta(mu1_eta, sec_mu_eta)")
        df_truth = safe_def(df_truth, "dimuon_delta_r_unified", "TruthExt::delta_r(dimuon_delta_phi_unified, dimuon_delta_eta_unified)")

        df_truth = safe_def(df_truth, "dimuon_e_sum_unified", "mu1_e + sec_mu_e")
        df_truth = safe_def(df_truth, "dimuon_delta_e_unified", "mu1_e - sec_mu_e")
        df_truth = safe_def(df_truth, "dimuon_abs_delta_e_unified", "std::abs(mu1_e - sec_mu_e)")
        df_truth = safe_def(df_truth, "dimuon_e_ratio_unified", "(mu1_e > 1e-6) ? (sec_mu_e / mu1_e) : 0.0")
        df_truth = safe_def(df_truth, "dimuon_delta_p_unified", "mu1_p - sec_mu_p")
        df_truth = safe_def(df_truth, "dimuon_delta_pt_unified", "mu1_pt - sec_mu_pt")
        df_truth = safe_def(df_truth, "dimuon_energy_asym_unified", "(mu1_e + sec_mu_e > 1e-6) ? (mu1_e - sec_mu_e) / (mu1_e + sec_mu_e) : 0.0")
        df_truth = safe_def(df_truth, "dimuon_p_ratio_unified", "(mu1_p > 1e-6) ? sec_mu_p / mu1_p : 0.0")
        df_truth = safe_def(df_truth, "dimuon_e_frac_nu_unified", "(nu_e > 1e-6) ? (mu1_e + sec_mu_e) / nu_e : 0.0")
        df_truth = safe_def(df_truth, "sec_mu_e_frac_charm", "(charm_e > 1e-6) ? sec_mu_e / charm_e : 0.0")
        df_truth = safe_def(df_truth, "indirect_mother_category", """
            int abs_m = std::abs(downstream_muon_mother_pdg);
            if (abs_m == 211) return 1;
            if (abs_m == 321 || abs_m == 130 || abs_m == 310) return 2;
            if (abs_m > 0) return 3;
            return 0;
        """)
    else:
        # Raw TTree: define truth via C++ processor
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

                  # Direct vs Indirect Decay Observables
                  .Define("charm_has_direct_muon", "truth.charmHasDirectMuon ? 1 : 0")
                  .Define("has_charm_hadronic_downstream_muon", "truth.hasCharmHadronicDownstreamMuon ? 1 : 0")
                  .Define("downstream_muon_track_id", "truth.downstreamMuonTrackId")
                  .Define("downstream_muon_pdg", "truth.downstreamMuonPdg")
                  .Define("downstream_muon_mother_pdg", "truth.downstreamMuonMotherPdg")
                  .Define("downstream_muon_p", "truth.downstreamMuonP")
                  .Define("downstream_muon_pt", "truth.downstreamMuonPt")
                  .Define("is_direct_charm_decay", "truth.charmHasDirectMuon ? 1 : 0")
                  .Define("is_indirect_charm_decay", "(!truth.charmHasDirectMuon && truth.hasCharmHadronicDownstreamMuon) ? 1 : 0")
                  .Define("sec_mu_track_id", "truth.charmHasDirectMuon ? truth.mu2TrackId : truth.downstreamMuonTrackId")
                  .Define("sec_mu_in_ds_acceptance", "truth.mu2InDS")
                  .Define("sec_mu_n_ds_points", "truth.mu2nDSPoints")
                  .Define("sec_mu_p", "(truth.charmHasDirectMuon && truth.mu2P > 0.0) ? truth.mu2P : truth.downstreamMuonP")
                  .Define("sec_mu_pt", "(truth.charmHasDirectMuon && truth.mu2Pt > 0.0) ? truth.mu2Pt : truth.downstreamMuonPt")
                  .Define("sec_mu_e", "(truth.charmHasDirectMuon && truth.mu2E > 0.0) ? truth.mu2E : (truth.downstreamMuonP > 0.0 ? std::sqrt(truth.downstreamMuonP * truth.downstreamMuonP + 0.1056583755 * 0.1056583755) : 0.0)")
                  .Define("sec_mu_mother_pdg", "truth.charmHasDirectMuon ? truth.mu2MotherPdg : truth.downstreamMuonMotherPdg")
                  .Define("sec_mu_abs_mother_pdg", "std::abs(truth.charmHasDirectMuon ? truth.mu2MotherPdg : truth.downstreamMuonMotherPdg)")
                  .Define("sec_mu_px", "TruthExt::sec_mu_px(truth.charmHasDirectMuon ? 1 : 0, truth.mu2Px, truth.downstreamMuonTrackId, MCTrack)")
                  .Define("sec_mu_py", "TruthExt::sec_mu_py(truth.charmHasDirectMuon ? 1 : 0, truth.mu2Py, truth.downstreamMuonTrackId, MCTrack)")
                  .Define("sec_mu_pz", "TruthExt::sec_mu_pz(truth.charmHasDirectMuon ? 1 : 0, truth.mu2Pz, truth.downstreamMuonTrackId, MCTrack)")
                  .Define("sec_mu_eta", "TruthExt::sec_mu_eta(truth.charmHasDirectMuon ? 1 : 0, truth.mu2Eta, truth.downstreamMuonTrackId, MCTrack)")
                  .Define("sec_mu_phi", "TruthExt::sec_mu_phi(truth.charmHasDirectMuon ? 1 : 0, truth.mu2Phi, truth.downstreamMuonTrackId, MCTrack)")
                  .Define("sec_mu_theta", "TruthExt::sec_mu_theta(truth.charmHasDirectMuon ? 1 : 0, truth.mu2Theta, truth.downstreamMuonTrackId, MCTrack)")
                  .Define("dimuon_opening_angle_mrad_unified", "TruthExt::opening_angle_mrad(truth.mu1Px, truth.mu1Py, truth.mu1Pz, sec_mu_px, sec_mu_py, sec_mu_pz)")
                  .Define("dimuon_mass_unified", "TruthExt::inv_mass(truth.mu1E, truth.mu1Px, truth.mu1Py, truth.mu1Pz, sec_mu_e, sec_mu_px, sec_mu_py, sec_mu_pz)")
                  .Define("dimuon_pt_unified", "TruthExt::pt(truth.mu1Px, truth.mu1Py, sec_mu_px, sec_mu_py)")
                  .Define("dimuon_delta_phi_unified", "TruthExt::delta_phi(truth.mu1Phi, sec_mu_phi)")
                  .Define("dimuon_delta_eta_unified", "TruthExt::delta_eta(truth.mu1Eta, sec_mu_eta)")
                  .Define("dimuon_delta_r_unified", "TruthExt::delta_r(dimuon_delta_phi_unified, dimuon_delta_eta_unified)")
                  .Define("dimuon_e_sum_unified", "truth.mu1E + sec_mu_e")
                  .Define("dimuon_delta_e_unified", "truth.mu1E - sec_mu_e")
                  .Define("dimuon_abs_delta_e_unified", "std::abs(truth.mu1E - sec_mu_e)")
                  .Define("dimuon_e_ratio_unified", "(truth.mu1E > 1e-6) ? (sec_mu_e / truth.mu1E) : 0.0")
                  .Define("dimuon_delta_p_unified", "truth.mu1P - sec_mu_p")
                  .Define("dimuon_delta_pt_unified", "truth.mu1Pt - sec_mu_pt")
                  .Define("dimuon_energy_asym_unified", "(truth.mu1E + sec_mu_e > 1e-6) ? (truth.mu1E - sec_mu_e) / (truth.mu1E + sec_mu_e) : 0.0")
                  .Define("dimuon_p_ratio_unified", "(truth.mu1P > 1e-6) ? sec_mu_p / truth.mu1P : 0.0")
                  .Define("dimuon_e_frac_nu_unified", "(truth.nuE > 1e-6) ? (truth.mu1E + sec_mu_e) / truth.nuE : 0.0")
                  .Define("sec_mu_e_frac_charm", "(truth.charmE > 1e-6) ? sec_mu_e / truth.charmE : 0.0")
                  .Define("indirect_mother_category", """
                      int abs_m = std::abs(truth.downstreamMuonMotherPdg);
                      if (abs_m == 211) return 1;
                      if (abs_m == 321 || abs_m == 130 || abs_m == 310) return 2;
                      if (abs_m > 0) return 3;
                      return 0;
                  """)
        )

    # Define XZ and YZ projected observables and separation at detector planes
    df_truth = safe_def(df_truth, "mu1_slope_xz", "(std::abs(mu1_pz) > 1e-9) ? mu1_px / mu1_pz : 0.0")
    df_truth = safe_def(df_truth, "mu1_slope_yz", "(std::abs(mu1_pz) > 1e-9) ? mu1_py / mu1_pz : 0.0")
    df_truth = safe_def(df_truth, "mu1_theta_xz_mrad", "std::atan2(mu1_px, mu1_pz) * 1000.0")
    df_truth = safe_def(df_truth, "mu1_theta_yz_mrad", "std::atan2(mu1_py, mu1_pz) * 1000.0")

    df_truth = safe_def(df_truth, "sec_mu_slope_xz", "(std::abs(sec_mu_pz) > 1e-9) ? sec_mu_px / sec_mu_pz : 0.0")
    df_truth = safe_def(df_truth, "sec_mu_slope_yz", "(std::abs(sec_mu_pz) > 1e-9) ? sec_mu_py / sec_mu_pz : 0.0")
    df_truth = safe_def(df_truth, "sec_mu_theta_xz_mrad", "std::atan2(sec_mu_px, sec_mu_pz) * 1000.0")
    df_truth = safe_def(df_truth, "sec_mu_theta_yz_mrad", "std::atan2(sec_mu_py, sec_mu_pz) * 1000.0")

    df_truth = safe_def(df_truth, "dimuon_opening_angle_xz_mrad", "std::abs(mu1_theta_xz_mrad - sec_mu_theta_xz_mrad)")
    df_truth = safe_def(df_truth, "dimuon_opening_angle_yz_mrad", "std::abs(mu1_theta_yz_mrad - sec_mu_theta_yz_mrad)")
    df_truth = safe_def(df_truth, "dimuon_delta_slope_xz", "std::abs(mu1_slope_xz - sec_mu_slope_xz)")
    df_truth = safe_def(df_truth, "dimuon_delta_slope_yz", "std::abs(mu1_slope_yz - sec_mu_slope_yz)")

    # Spatial separations at detector planes (SciFi start ~300cm, SciFi exit ~390cm, DS entrance ~470cm)
    df_truth = safe_def(df_truth, "dimuon_sep_x_scifi_entrance", "std::abs(mu1_slope_xz - sec_mu_slope_xz) * std::max(0.0, 300.0 - vtx_z)")
    df_truth = safe_def(df_truth, "dimuon_sep_y_scifi_entrance", "std::abs(mu1_slope_yz - sec_mu_slope_yz) * std::max(0.0, 300.0 - vtx_z)")
    df_truth = safe_def(df_truth, "dimuon_sep_3d_scifi_entrance", "std::sqrt(dimuon_sep_x_scifi_entrance * dimuon_sep_x_scifi_entrance + dimuon_sep_y_scifi_entrance * dimuon_sep_y_scifi_entrance)")

    df_truth = safe_def(df_truth, "dimuon_sep_x_scifi_exit", "std::abs(mu1_slope_xz - sec_mu_slope_xz) * std::max(0.0, 390.0 - vtx_z)")
    df_truth = safe_def(df_truth, "dimuon_sep_y_scifi_exit", "std::abs(mu1_slope_yz - sec_mu_slope_yz) * std::max(0.0, 390.0 - vtx_z)")
    df_truth = safe_def(df_truth, "dimuon_sep_3d_scifi_exit", "std::sqrt(dimuon_sep_x_scifi_exit * dimuon_sep_x_scifi_exit + dimuon_sep_y_scifi_exit * dimuon_sep_y_scifi_exit)")

    df_truth = safe_def(df_truth, "dimuon_sep_x_ds_entrance", "std::abs(mu1_slope_xz - sec_mu_slope_xz) * std::max(0.0, 470.0 - vtx_z)")
    df_truth = safe_def(df_truth, "dimuon_sep_y_ds_entrance", "std::abs(mu1_slope_yz - sec_mu_slope_yz) * std::max(0.0, 470.0 - vtx_z)")
    df_truth = safe_def(df_truth, "dimuon_sep_3d_ds_entrance", "std::sqrt(dimuon_sep_x_ds_entrance * dimuon_sep_x_ds_entrance + dimuon_sep_y_ds_entrance * dimuon_sep_y_ds_entrance)")

    # Projected decay lengths
    df_truth = safe_def(df_truth, "decay_length_xz", "std::sqrt((charm_decay_x - vtx_x)*(charm_decay_x - vtx_x) + (charm_decay_z - vtx_z)*(charm_decay_z - vtx_z))")
    df_truth = safe_def(df_truth, "decay_length_yz", "std::sqrt((charm_decay_y - vtx_y)*(charm_decay_y - vtx_y) + (charm_decay_z - vtx_z)*(charm_decay_z - vtx_z))")

    # Reco_MuonTracks definitions (when available)
    has_reco_tracks_branch = ("Reco_MuonTracks" in existing_cols)
    if has_reco_tracks_branch:
        df_truth = safe_def(df_truth, "n_reco_muon_tracks", "RecoTracksExt::count_total_tracks(Reco_MuonTracks)")
        df_truth = safe_def(df_truth, "n_ds_hough_tracks", "RecoTracksExt::count_ds_tracks(Reco_MuonTracks)")
        df_truth = safe_def(df_truth, "has_two_ds_tracks", "n_ds_hough_tracks >= 2 ? 1 : 0")
        df_truth = safe_def(df_truth, "has_exactly_two_ds_tracks", "n_ds_hough_tracks == 2 ? 1 : 0")
        df_truth = safe_def(df_truth, "reco_ds_cross_z_xz", "RecoTracksExt::cross_z_xz(Reco_MuonTracks)")
        df_truth = safe_def(df_truth, "reco_ds_cross_z_yz", "RecoTracksExt::cross_z_yz(Reco_MuonTracks)")
        df_truth = safe_def(df_truth, "reco_ds_cross_x", "RecoTracksExt::cross_x(Reco_MuonTracks)")
        df_truth = safe_def(df_truth, "reco_ds_cross_y", "RecoTracksExt::cross_y(Reco_MuonTracks)")
        df_truth = safe_def(df_truth, "reco_ds_delta_z_cross", "RecoTracksExt::delta_z_cross(Reco_MuonTracks)")
        df_truth = safe_def(df_truth, "reco_ds_abs_delta_z_cross", "RecoTracksExt::abs_delta_z_cross(Reco_MuonTracks)")
        df_truth = safe_def(df_truth, "reco_ds_res_z_xz", "(reco_ds_cross_z_xz > -9000.0) ? (reco_ds_cross_z_xz - vtx_z) : -9999.0")
        df_truth = safe_def(df_truth, "reco_ds_res_z_yz", "(reco_ds_cross_z_yz > -9000.0) ? (reco_ds_cross_z_yz - vtx_z) : -9999.0")
    else:
        df_truth = safe_def(df_truth, "n_reco_muon_tracks", "0")
        df_truth = safe_def(df_truth, "n_ds_hough_tracks", "0")
        df_truth = safe_def(df_truth, "has_two_ds_tracks", "0")
        df_truth = safe_def(df_truth, "has_exactly_two_ds_tracks", "0")
        df_truth = safe_def(df_truth, "reco_ds_cross_z_xz", "-9999.0")
        df_truth = safe_def(df_truth, "reco_ds_cross_z_yz", "-9999.0")
        df_truth = safe_def(df_truth, "reco_ds_cross_x", "-9999.0")
        df_truth = safe_def(df_truth, "reco_ds_cross_y", "-9999.0")
        df_truth = safe_def(df_truth, "reco_ds_delta_z_cross", "-9999.0")
        df_truth = safe_def(df_truth, "reco_ds_abs_delta_z_cross", "-9999.0")
        df_truth = safe_def(df_truth, "reco_ds_res_z_xz", "-9999.0")
        df_truth = safe_def(df_truth, "reco_ds_res_z_yz", "-9999.0")

    # 6. Define Filtered Subsamples
    # All Interacting Muon Neutrinos in the Detector (CC + NC, nu_mu and anti-nu_mu)
    df_numu_all = df_truth.Filter(
        "(std::abs(nu_pdg) == 14) || is_numu_cc || is_anti_numu_cc",
        "All Interacting Nu_mu in Detector (CC + NC)"
    )
    df_numu_all_fiducial = df_numu_all.Filter("is_fiducial", "Fiducial Interacting Nu_mu (CC + NC)")

    # Breakdown by interaction type (CC vs NC)
    df_numu_cc = df_numu_all.Filter("is_cc || is_numu_cc || is_anti_numu_cc", "All Nu_mu CC")
    df_numu_nc = df_numu_all.Filter("is_nc", "All Nu_mu NC")
    df_numu_cc_fiducial = df_numu_all_fiducial.Filter("is_cc || is_numu_cc || is_anti_numu_cc", "Fiducial Nu_mu CC")
    df_numu_nc_fiducial = df_numu_all_fiducial.Filter("is_nc", "Fiducial Nu_mu NC")

    # Breakdown by neutrino vs anti-neutrino
    df_numu_nu = df_numu_all.Filter("nu_pdg == 14", "Nu_mu (PDG +14)")
    df_numu_antinu = df_numu_all.Filter("nu_pdg == -14", "Anti-Nu_mu (PDG -14)")
    df_numu_nu_fiducial = df_numu_all_fiducial.Filter("nu_pdg == 14", "Fiducial Nu_mu (PDG +14)")
    df_numu_antinu_fiducial = df_numu_all_fiducial.Filter("nu_pdg == -14", "Fiducial Anti-Nu_mu (PDG -14)")

    # Target Fiducial Volume Nu_mu CC
    df_fiducial = df_numu_cc.Filter("is_fiducial", "Target Fiducial Volume Nu_mu CC")
    df_charm = df_numu_cc.Filter("has_charm", "Nu_mu CC with Charmed Hadron")

    # Direct, Indirect, and Inclusive Charm Dimuon Candidate Samples
    fid_prefix = "is_fiducial && " if args.fiducial else ""

    # 1. Direct Charm Dimuon Signal (charm directly decays to muon)
    df_direct = df_numu_cc.Filter(
        f"{fid_prefix}has_charm && (charm_has_direct_muon == 1) && (has_candidate || (is_cc && is_opposite_sign))",
        "Direct Charm Dimuon Signal (c -> mu)"
    )

    # 2. Indirect Charm Dimuon Fake Signal (charm decays hadronically -> intermediate daughter -> second muon)
    df_indirect = df_numu_cc.Filter(
        f"{fid_prefix}has_charm && (charm_has_direct_muon == 0) && (has_charm_hadronic_downstream_muon == 1)",
        "Indirect Charm Dimuon Fake Signal (c -> hadrons -> mu)"
    )

    # 3. Inclusive Dimuon Charm (Direct + Indirect)
    if args.require_direct_charm_decay is True or args.category == "direct":
        df_candidate = df_direct
    elif args.require_direct_charm_decay is False or args.category == "indirect":
        df_candidate = df_indirect
    else:
        df_candidate = df_numu_cc.Filter(
            f"{fid_prefix}has_charm && (((charm_has_direct_muon == 1) && (has_candidate || (is_cc && is_opposite_sign))) || ((charm_has_direct_muon == 0) && (has_charm_hadronic_downstream_muon == 1)))",
            "Inclusive Dimuon Charm (Direct + Indirect)"
        )

    # Acceptance-partitioned signal samples
    df_in_acc = df_candidate.Filter("dimuon_in_ds_acceptance", "Inclusive Signal In DS Acceptance")
    df_out_acc = df_candidate.Filter("!dimuon_in_ds_acceptance", "Inclusive Signal Outside DS Acceptance")
    df_mu2_in_acc = df_candidate.Filter("mu2_in_ds_acceptance", "Signal Mu2 In DS Acceptance")
    df_mu2_out_acc = df_candidate.Filter("!mu2_in_ds_acceptance", "Signal Mu2 Outside DS Acceptance")

    df_direct_in_acc = df_direct.Filter("dimuon_in_ds_acceptance", "Direct Signal In DS Acceptance")
    df_direct_out_acc = df_direct.Filter("!dimuon_in_ds_acceptance", "Direct Signal Outside DS Acceptance")

    df_indirect_in_acc = df_indirect.Filter("dimuon_in_ds_acceptance", "Indirect Fake In DS Acceptance")
    df_indirect_out_acc = df_indirect.Filter("!dimuon_in_ds_acceptance", "Indirect Fake Outside DS Acceptance")

    # Specific charm hadron species filters
    df_d0 = df_candidate.Filter("abs_charm_pdg == 421", "D0 / anti-D0")
    df_dplus = df_candidate.Filter("abs_charm_pdg == 411", "D+ / D-")
    df_ds = df_candidate.Filter("abs_charm_pdg == 431", "Ds+ / Ds-")
    df_lambdac = df_candidate.Filter("abs_charm_pdg == 4122", "Lambda_c+ / anti-Lambda_c-")

    df_direct_d0 = df_direct.Filter("abs_charm_pdg == 421", "Direct D0")
    df_direct_dplus = df_direct.Filter("abs_charm_pdg == 411", "Direct D+")
    df_direct_ds = df_direct.Filter("abs_charm_pdg == 431", "Direct Ds+")
    df_direct_lambdac = df_direct.Filter("abs_charm_pdg == 4122", "Direct Lambda_c+")

    df_indirect_d0 = df_indirect.Filter("abs_charm_pdg == 421", "Indirect D0")
    df_indirect_dplus = df_indirect.Filter("abs_charm_pdg == 411", "Indirect D+")
    df_indirect_ds = df_indirect.Filter("abs_charm_pdg == 431", "Indirect Ds+")
    df_indirect_lambdac = df_indirect.Filter("abs_charm_pdg == 4122", "Indirect Lambda_c+")

    # Intermediate mother particle breakdown for Indirect
    df_indirect_pion = df_indirect.Filter("std::abs(downstream_muon_mother_pdg) == 211", "Indirect via Pion")
    df_indirect_kaon = df_indirect.Filter("std::abs(downstream_muon_mother_pdg) == 321 || std::abs(downstream_muon_mother_pdg) == 130 || std::abs(downstream_muon_mother_pdg) == 310", "Indirect via Kaon")
    df_indirect_other = df_indirect.Filter("std::abs(downstream_muon_mother_pdg) != 211 && std::abs(downstream_muon_mother_pdg) != 321 && std::abs(downstream_muon_mother_pdg) != 130 && std::abs(downstream_muon_mother_pdg) != 310", "Indirect via Other")

    # Reconstructed 2 DS Tracks filters (when Reco_MuonTracks is present)
    df_signal_2ds = df_candidate.Filter("has_two_ds_tracks == 1", "Signal with >=2 DS Tracks")
    df_direct_2ds = df_direct.Filter("has_two_ds_tracks == 1", "Direct with >=2 DS Tracks")
    df_indirect_2ds = df_indirect.Filter("has_two_ds_tracks == 1", "Indirect with >=2 DS Tracks")

    df_signal_2ds_valid_cross = df_signal_2ds.Filter("reco_ds_cross_z_xz > -9000.0 && reco_ds_cross_z_yz > -9000.0", "Signal with Valid 2D Crossings")
    df_direct_2ds_valid_cross = df_direct_2ds.Filter("reco_ds_cross_z_xz > -9000.0 && reco_ds_cross_z_yz > -9000.0", "Direct with Valid 2D Crossings")
    df_indirect_2ds_valid_cross = df_indirect_2ds.Filter("reco_ds_cross_z_xz > -9000.0 && reco_ds_cross_z_yz > -9000.0", "Indirect with Valid 2D Crossings")

    df_signal_2ds_in_acc = df_signal_2ds.Filter("dimuon_in_ds_acceptance", "Signal with >=2 DS Tracks In DS Acc")
    df_direct_2ds_in_acc = df_direct_2ds.Filter("dimuon_in_ds_acceptance", "Direct with >=2 DS Tracks In DS Acc")
    df_indirect_2ds_in_acc = df_indirect_2ds.Filter("dimuon_in_ds_acceptance", "Indirect with >=2 DS Tracks In DS Acc")

    # Register filter nodes for booking
    filter_nodes = {
        "all": df_truth,
        "all_numu": df_numu_all,
        "all_numu_fiducial": df_numu_all_fiducial,
        "numu_cc": df_numu_cc,
        "numu_nc": df_numu_nc,
        "numu_nu": df_numu_nu,
        "numu_antinu": df_numu_antinu,
        "fiducial": df_fiducial,
        "charm": df_charm,
        "candidate": df_candidate,
        "signal": df_candidate,
        "direct": df_direct,
        "indirect": df_indirect,
        "in_acc": df_in_acc,
        "in_acceptance": df_in_acc,
        "out_acc": df_out_acc,
        "outside_acceptance": df_out_acc,
        "direct_in_acc": df_direct_in_acc,
        "direct_out_acc": df_direct_out_acc,
        "indirect_in_acc": df_indirect_in_acc,
        "indirect_out_acc": df_indirect_out_acc,
        "mu2_in_acc": df_mu2_in_acc,
        "mu2_out_acc": df_mu2_out_acc,
        "d0": df_d0,
        "dplus": df_dplus,
        "ds": df_ds,
        "lambdac": df_lambdac,
        "direct_d0": df_direct_d0,
        "direct_dplus": df_direct_dplus,
        "direct_ds": df_direct_ds,
        "direct_lambdac": df_direct_lambdac,
        "indirect_d0": df_indirect_d0,
        "indirect_dplus": df_indirect_dplus,
        "indirect_ds": df_indirect_ds,
        "indirect_lambdac": df_indirect_lambdac,
        "indirect_pion": df_indirect_pion,
        "indirect_kaon": df_indirect_kaon,
        "indirect_other": df_indirect_other,
        "signal_2ds": df_signal_2ds,
        "direct_2ds": df_direct_2ds,
        "indirect_2ds": df_indirect_2ds,
        "signal_2ds_cross": df_signal_2ds_valid_cross,
        "direct_2ds_cross": df_direct_2ds_valid_cross,
        "indirect_2ds_cross": df_indirect_2ds_valid_cross,
    }

    # 7. Book Counters
    c_total = df_truth.Count()
    c_all_numu = df_numu_all.Count()
    c_all_numu_fid = df_numu_all_fiducial.Count()
    c_numu_cc = df_numu_cc.Count()
    c_numu_nc = df_numu_nc.Count()
    c_numu_nu = df_numu_nu.Count()
    c_numu_antinu = df_numu_antinu.Count()
    c_fiducial = df_fiducial.Count()
    c_charm = df_charm.Count()
    c_candidate = df_candidate.Count()
    c_direct = df_direct.Count()
    c_indirect = df_indirect.Count()
    c_in_acc = df_in_acc.Count()
    c_out_acc = df_out_acc.Count()
    c_direct_in_acc = df_direct_in_acc.Count()
    c_direct_out_acc = df_direct_out_acc.Count()
    c_indirect_in_acc = df_indirect_in_acc.Count()
    c_indirect_out_acc = df_indirect_out_acc.Count()
    c_mu1_in_acc = df_candidate.Filter("mu1_in_ds_acceptance").Count()
    c_mu2_in_acc = df_mu2_in_acc.Count()
    c_d0 = df_d0.Count()
    c_dplus = df_dplus.Count()
    c_ds = df_ds.Count()
    c_lambdac = df_lambdac.Count()
    c_direct_d0 = df_direct_d0.Count()
    c_direct_dplus = df_direct_dplus.Count()
    c_direct_ds = df_direct_ds.Count()
    c_direct_lambdac = df_direct_lambdac.Count()
    c_indirect_d0 = df_indirect_d0.Count()
    c_indirect_dplus = df_indirect_dplus.Count()
    c_indirect_ds = df_indirect_ds.Count()
    c_indirect_lambdac = df_indirect_lambdac.Count()
    c_indirect_pion = df_indirect_pion.Count()
    c_indirect_kaon = df_indirect_kaon.Count()
    c_indirect_other = df_indirect_other.Count()

    # Reconstructed 2 DS tracks counters
    c_2ds_signal = df_signal_2ds.Count()
    c_2ds_direct = df_direct_2ds.Count()
    c_2ds_indirect = df_indirect_2ds.Count()
    c_2ds_signal_in_acc = df_signal_2ds_in_acc.Count()
    c_2ds_direct_in_acc = df_direct_2ds_in_acc.Count()
    c_2ds_indirect_in_acc = df_indirect_2ds_in_acc.Count()

    booked_counters = [
        c_total, c_all_numu, c_all_numu_fid, c_numu_cc, c_numu_nc,
        c_numu_nu, c_numu_antinu, c_fiducial, c_charm, c_candidate,
        c_direct, c_indirect,
        c_in_acc, c_out_acc, c_direct_in_acc, c_direct_out_acc,
        c_indirect_in_acc, c_indirect_out_acc,
        c_mu1_in_acc, c_mu2_in_acc,
        c_d0, c_dplus, c_ds, c_lambdac,
        c_direct_d0, c_direct_dplus, c_direct_ds, c_direct_lambdac,
        c_indirect_d0, c_indirect_dplus, c_indirect_ds, c_indirect_lambdac,
        c_indirect_pion, c_indirect_kaon, c_indirect_other,
        c_2ds_signal, c_2ds_direct, c_2ds_indirect,
        c_2ds_signal_in_acc, c_2ds_direct_in_acc, c_2ds_indirect_in_acc
    ]

    # 8. Book Comprehensive Suite of Histograms
    booked_histograms_1d = {}
    booked_histograms_2d = {}
    booked_profiles = {}
    booked_teff_pairs = {}

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

    # Helper function to book TEfficiency pairs (unweighted for Clopper-Pearson frequentist evaluation)
    def book_teff(name, title, col, node_pass, node_tot, nbins=40, xmin=0.0, xmax=1.0):
        m_pass = ROOT.RDF.TH1DModel(f"{name}_pass", title, nbins, xmin, xmax)
        m_tot = ROOT.RDF.TH1DModel(f"{name}_tot", title, nbins, xmin, xmax)
        h_pass = node_pass.Histo1D(m_pass, col)
        h_tot = node_tot.Histo1D(m_tot, col)
        booked_teff_pairs[name] = (title, h_pass, h_tot)

    # --------------------------------------------------------------------------
    # 8a. Energy Distributions of All Muon Neutrinos Interacting in Detector
    # --------------------------------------------------------------------------
    # 1. Total Interacting Muon Neutrinos (CC + NC, inclusive across full detector)
    book_1d("h_all_numu_nu_e", "Energy of All Interacting #nu_{#mu} (CC+NC);E_{#nu} [GeV];Events / 10 GeV", "nu_e", df_numu_all, 100, 0.0, 1000.0)
    book_1d("h_all_numu_nu_e_unweighted", "Energy of All Interacting #nu_{#mu} (Unweighted);E_{#nu} [GeV];Raw Counts / 10 GeV", "nu_e", df_numu_all, 100, 0.0, 1000.0, weight=None)

    # 2. Fiducial Interacting Muon Neutrinos (Target Fiducial Volume, CC + NC)
    book_1d("h_all_numu_fiducial_nu_e", "Energy of Fiducial Interacting #nu_{#mu} (CC+NC);E_{#nu} [GeV];Events / 10 GeV", "nu_e", df_numu_all_fiducial, 100, 0.0, 1000.0)
    book_1d("h_all_numu_fiducial_nu_e_unweighted", "Energy of Fiducial Interacting #nu_{#mu} (Unweighted);E_{#nu} [GeV];Raw Counts / 10 GeV", "nu_e", df_numu_all_fiducial, 100, 0.0, 1000.0, weight=None)

    # 3. Interaction Mode Breakdown: Charged Current (CC) vs Neutral Current (NC)
    book_1d("h_all_numu_cc_nu_e", "Energy of Interacting #nu_{#mu} CC;E_{#nu} [GeV];Events / 10 GeV", "nu_e", df_numu_cc, 100, 0.0, 1000.0)
    book_1d("h_all_numu_nc_nu_e", "Energy of Interacting #nu_{#mu} NC;E_{#nu} [GeV];Events / 10 GeV", "nu_e", df_numu_nc, 100, 0.0, 1000.0)

    # 4. Neutrino (#nu_{#mu}, PDG +14) vs Anti-Neutrino (#bar{#nu}_{#mu}, PDG -14)
    book_1d("h_all_numu_pdg14_nu_e", "Energy of Interacting #nu_{#mu} (PDG +14);E_{#nu} [GeV];Events / 10 GeV", "nu_e", df_numu_nu, 100, 0.0, 1000.0)
    book_1d("h_all_numu_pdg_minus14_nu_e", "Energy of Interacting #bar{#nu}_{#mu} (PDG -14);E_{#nu} [GeV];Events / 10 GeV", "nu_e", df_numu_antinu, 100, 0.0, 1000.0)

    # 5. Interaction Vertex and Kinematic Profiles for All Interacting Nu_mu
    book_2d("h2_all_numu_vtx_xy", "Interaction Vertex Transverse Profile (All Nu_mu);x_{vtx} [cm];y_{vtx} [cm]", "vtx_x", "vtx_y", df_numu_all, 60, -60.0, 0.0, 60, 3.0, 63.0)
    book_2d("h2_all_numu_vtx_z_vs_nu_e", "Interaction Vertex Z vs E_{#nu} (All Nu_mu);E_{#nu} [GeV];z_{vtx} [cm]", "nu_e", "vtx_z", df_numu_all, 50, 0.0, 1000.0, 50, 260.0, 360.0)
    book_2d("h2_all_numu_q2_vs_nu_e", "Q^{2} vs E_{#nu} (All Nu_mu);E_{#nu} [GeV];Q^{2} [GeV^{2}]", "nu_e", "q2", df_numu_all, 50, 0.0, 1000.0, 50, 0.0, 50.0)

    # --------------------------------------------------------------------------
    # 8b. Primary Muon (mu1) and Decay Muon (mu2) Energies & Kinematics
    # Booked for: Inclusive Signal, In-Acceptance, and Outside-Acceptance
    # --------------------------------------------------------------------------
    samples_1d = [
        ("signal", df_candidate, "Inclusive Signal"),
        ("direct", df_direct, "Direct Signal (c #rightarrow #mu)"),
        ("indirect", df_indirect, "Indirect Fake Signal (c #rightarrow h #rightarrow #mu)"),
        ("in_acc", df_in_acc, "Inclusive In DS Acceptance"),
        ("out_acc", df_out_acc, "Inclusive Outside DS Acceptance"),
        ("direct_in_acc", df_direct_in_acc, "Direct In DS Acceptance"),
        ("direct_out_acc", df_direct_out_acc, "Direct Outside DS Acceptance"),
        ("indirect_in_acc", df_indirect_in_acc, "Indirect In DS Acceptance"),
        ("indirect_out_acc", df_indirect_out_acc, "Indirect Outside DS Acceptance"),
    ]

    for prefix, node, label in samples_1d:
        # Muon Energies (user-requested core focus)
        book_1d(f"h_{prefix}_mu1_e", f"Primary #mu_{{1}} Energy ({label});E_{{#mu1}} [GeV];Events", "mu1_e", node, 100, 0.0, 800.0)
        book_1d(f"h_{prefix}_mu2_e", f"Secondary #mu_{{2}} Energy ({label});E_{{#mu2}} [GeV];Events", "sec_mu_e", node, 100, 0.0, 300.0)
        book_1d(f"h_{prefix}_dimuon_e_sum", f"Total Muon Energy E_{{#mu1}} + E_{{#mu2}} ({label});E_{{#mu1}} + E_{{#mu2}} [GeV];Events", "dimuon_e_sum_unified", node, 100, 0.0, 1000.0)
        book_1d(f"h_{prefix}_dimuon_delta_e", f"Muon Energy Difference E_{{#mu1}} - E_{{#mu2}} ({label});E_{{#mu1}} - E_{{#mu2}} [GeV];Events", "dimuon_delta_e_unified", node, 100, -50.0, 700.0)
        book_1d(f"h_{prefix}_dimuon_abs_delta_e", f"Absolute Muon Energy Difference |E_{{#mu1}} - E_{{#mu2}}| ({label});|E_{{#mu1}} - E_{{#mu2}}| [GeV];Events", "dimuon_abs_delta_e_unified", node, 100, 0.0, 700.0)
        book_1d(f"h_{prefix}_dimuon_e_ratio", f"Muon Energy Ratio E_{{#mu2}} / E_{{#mu1}} ({label});E_{{#mu2}} / E_{{#mu1}};Events", "dimuon_e_ratio_unified", node, 50, 0.0, 1.0)
        book_1d(f"h_{prefix}_dimuon_delta_p", f"Muon Momentum Difference p_{{#mu1}} - p_{{#mu2}} ({label});p_{{#mu1}} - p_{{#mu2}} [GeV/c];Events", "dimuon_delta_p_unified", node, 100, -50.0, 700.0)
        book_1d(f"h_{prefix}_dimuon_delta_pt", f"Muon Transverse Momentum Difference p_{{T,#mu1}} - p_{{T,#mu2}} ({label});p_{{T,#mu1}} - p_{{T,#mu2}} [GeV/c];Events", "dimuon_delta_pt_unified", node, 50, -2.0, 5.0)
        book_1d(f"h_{prefix}_dimuon_e_frac_nu", f"Dimuon Energy Fraction of E_{{#nu}} ({label});(E_{{#mu1}} + E_{{#mu2}}) / E_{{#nu}};Events", "dimuon_e_frac_nu_unified", node, 50, 0.0, 1.0)
        book_1d(f"h_{prefix}_mu1_e_frac_nu", f"Primary #mu_{{1}} Energy Fraction of E_{{#nu}} ({label});E_{{#mu1}} / E_{{#nu}};Events", "mu1_e_frac_nu", node, 50, 0.0, 1.0)
        book_1d(f"h_{prefix}_mu2_e_frac_charm", f"Secondary #mu_{{2}} Energy Fraction of E_{{charm}} ({label});E_{{#mu2}} / E_{{charm}};Events", "sec_mu_e_frac_charm", node, 50, 0.0, 1.0)

        # Muon Momenta and Angles
        book_1d(f"h_{prefix}_mu1_p", f"Primary #mu_{{1}} Total Momentum ({label});p_{{#mu1}} [GeV/c];Events", "mu1_p", node, 100, 0.0, 800.0)
        book_1d(f"h_{prefix}_mu2_p", f"Secondary #mu_{{2}} Total Momentum ({label});p_{{#mu2}} [GeV/c];Events", "sec_mu_p", node, 100, 0.0, 300.0)
        book_1d(f"h_{prefix}_mu1_pt", f"Primary #mu_{{1}} Transverse Momentum ({label});p_{{T,#mu1}} [GeV/c];Events", "mu1_pt", node, 50, 0.0, 5.0)
        book_1d(f"h_{prefix}_mu2_pt", f"Secondary #mu_{{2}} Transverse Momentum ({label});p_{{T,#mu2}} [GeV/c];Events", "sec_mu_pt", node, 50, 0.0, 5.0)
        book_1d(f"h_{prefix}_mu1_eta", f"Primary #mu_{{1}} Pseudorapidity ({label});#eta_{{#mu1}};Events", "mu1_eta", node, 50, 4.0, 10.0)
        book_1d(f"h_{prefix}_mu2_eta", f"Secondary #mu_{{2}} Pseudorapidity ({label});#eta_{{#mu2}};Events", "sec_mu_eta", node, 50, 4.0, 10.0)
        book_1d(f"h_{prefix}_mu1_theta", f"Primary #mu_{{1}} Polar Angle ({label});#theta_{{#mu1}} [rad];Events", "mu1_theta", node, 50, 0.0, 0.2)
        book_1d(f"h_{prefix}_mu2_theta", f"Secondary #mu_{{2}} Polar Angle ({label});#theta_{{#mu2}} [rad];Events", "sec_mu_theta", node, 50, 0.0, 0.2)

        # Charm Decay Displaced Vertex & Relative Kinematics
        book_1d(f"h_{prefix}_mu2_ptrel", f"Secondary #mu_{{2}} p_{{T}}^{{rel}} to Charm Axis ({label});p_{{T}}^{{rel}} [GeV/c];Events", "mu2_ptrel", node, 50, 0.0, 2.0)
        book_1d(f"h_{prefix}_mu2_ip3d", f"Secondary #mu_{{2}} 3D Impact Parameter to Primary Vertex ({label});IP_{{3D}} [cm];Events", "mu2_ip3d", node, 50, 0.0, 0.5)
        book_1d(f"h_{prefix}_mu2_ipxy", f"Secondary #mu_{{2}} Transverse Impact Parameter ({label});IP_{{xy}} [cm];Events", "mu2_ipxy", node, 50, 0.0, 0.5)
        book_1d(f"h_{prefix}_mu2_opening_angle_charm", f"Opening Angle Between Charm and #mu_{{2}} ({label});#theta(#mu_{{2}},charm) [rad];Events", "mu2_opening_angle_charm", node, 50, 0.0, 0.2)
        book_1d(f"h_{prefix}_mu1_n_ds_points", f"Primary #mu_{{1}} DS MCPoints ({label});N_{{DS}}(#mu_{{1}});Events", "mu1_n_ds_points", node, 20, -0.5, 19.5)
        book_1d(f"h_{prefix}_mu2_n_ds_points", f"Secondary #mu_{{2}} DS MCPoints ({label});N_{{DS}}(#mu_{{2}});Events", "mu2_n_ds_points", node, 20, -0.5, 19.5)

        # Dimuon System Observables (Unified 3D Kinematics)
        book_1d(f"h_{prefix}_dimuon_mass", f"Dimuon Invariant Mass ({label});M_{{#mu#mu}} [GeV/c^{{2}}];Events", "dimuon_mass_unified", node, 60, 0.0, 15.0)
        book_1d(f"h_{prefix}_dimuon_pt", f"Dimuon Pair Transverse Momentum ({label});p_{{T,#mu#mu}} [GeV/c];Events", "dimuon_pt_unified", node, 50, 0.0, 8.0)
        book_1d(f"h_{prefix}_dimuon_p", f"Dimuon Pair Total Momentum ({label});p_{{#mu#mu}} [GeV/c];Events", "dimuon_p", node, 50, 0.0, 800.0)
        book_1d(f"h_{prefix}_dimuon_opening_angle_mrad", f"Dimuon 3D Opening Angle ({label});#theta_{{#mu#mu}} [mrad];Events", "dimuon_opening_angle_mrad_unified", node, 64, 0.0, 3200.0)
        book_1d(f"h_{prefix}_dimuon_delta_phi", f"Dimuon Azimuthal Separation ({label});#Delta#phi_{{#mu#mu}} [rad];Events", "dimuon_delta_phi_unified", node, 50, 0.0, 3.14159)
        book_1d(f"h_{prefix}_dimuon_delta_eta", f"Dimuon Pseudorapidity Separation ({label});#Delta#eta_{{#mu#mu}};Events", "dimuon_delta_eta_unified", node, 50, 0.0, 6.0)
        book_1d(f"h_{prefix}_dimuon_delta_r", f"Dimuon Separation #Delta R ({label});#Delta R_{{#mu#mu}};Events", "dimuon_delta_r_unified", node, 50, 0.0, 6.0)
        book_1d(f"h_{prefix}_dimuon_energy_asym", f"Dimuon Energy Asymmetry ({label});(E_{{1}} - E_{{2}}) / (E_{{1}} + E_{{2}});Events", "dimuon_energy_asym_unified", node, 50, -1.0, 1.0)
        book_1d(f"h_{prefix}_dimuon_p_ratio", f"Dimuon Momentum Ratio p_{{2}} / p_{{1}} ({label});p_{{#mu2}} / p_{{#mu1}};Events", "dimuon_p_ratio_unified", node, 50, 0.0, 2.0)

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
    # 8b-2. Intermediate Mother Particles for Indirect Charm Decays
    # --------------------------------------------------------------------------
    book_1d("h_indirect_downstream_muon_mother_pdg", "Indirect Muon Mother PDG (c #rightarrow h #rightarrow #mu);PDG ID;Events", "downstream_muon_mother_pdg", df_indirect, 800, -400.0, 400.0)
    book_1d("h_indirect_downstream_muon_mother_abs_pdg", "Indirect Muon Mother |PDG|;Abs PDG ID;Events", "sec_mu_abs_mother_pdg", df_indirect, 400, 0.0, 400.0)
    book_1d("h_indirect_mother_category", "Indirect Decay Intermediate Mother Category;Category (1: #pi^{#pm}, 2: K, 3: Other);Events", "indirect_mother_category", df_indirect, 4, 0.5, 4.5)

    indirect_subsamples = [
        ("pion", df_indirect_pion, "via #pi^{#pm}"),
        ("kaon", df_indirect_kaon, "via Kaon"),
        ("other", df_indirect_other, "via Other Hadron"),
    ]
    for m_key, m_node, m_lbl in indirect_subsamples:
        book_1d(f"h_indirect_{m_key}_sec_mu_e", f"Secondary #mu Energy ({m_lbl});E_{{#mu2}} [GeV];Events", "sec_mu_e", m_node, 100, 0.0, 10.0)
        book_1d(f"h_indirect_{m_key}_sec_mu_p", f"Secondary #mu Momentum ({m_lbl});p_{{#mu2}} [GeV/c];Events", "sec_mu_p", m_node, 100, 0.0, 10.0)
        book_1d(f"h_indirect_{m_key}_sec_mu_pt", f"Secondary #mu p_{{T}} ({m_lbl});p_{{T,#mu2}} [GeV/c];Events", "sec_mu_pt", m_node, 60, 0.0, 1.5)
        book_1d(f"h_indirect_{m_key}_dimuon_delta_e", f"Dimuon #Delta E ({m_lbl});E_{{#mu1}} - E_{{#mu2}} [GeV];Events", "dimuon_delta_e_unified", m_node, 100, -50.0, 700.0)
        book_1d(f"h_indirect_{m_key}_nu_e", f"Incoming #nu Energy ({m_lbl});E_{{#nu}} [GeV];Events", "nu_e", m_node, 50, 0.0, 1000.0)

    # --------------------------------------------------------------------------
    # 8b-3. 2D Histograms: Dimuon Energy & Kinematic Correlations
    # --------------------------------------------------------------------------
    # Core Dimuon Energy Correlation (E_mu2 vs E_mu1) across samples
    for prefix, node, label in samples_1d:
        book_2d(f"h2_{prefix}_mu2_e_vs_mu1_e", f"Dimuon Energy Correlation ({label});E_{{#mu1}} [GeV];E_{{#mu2}} [GeV]", "mu1_e", "sec_mu_e", node, 35, 0.0, 700.0, 35, 0.0, 700.0)
        if prefix in ["signal", "direct", "indirect"]:
            book_2d(f"h2_{prefix}_mu2_e_vs_mu1_e_zoom", f"Dimuon Energy Correlation (Zoom) ({label});E_{{#mu1}} [GeV];E_{{#mu2}} [GeV]", "mu1_e", "sec_mu_e", node, 35, 0.0, 700.0, 50, 0.0, 10.0)

        book_2d(f"h2_{prefix}_mu2_p_vs_mu1_p", f"Dimuon Momentum Correlation ({label});p_{{#mu1}} [GeV/c];p_{{#mu2}} [GeV/c]", "mu1_p", "sec_mu_p", node, 35, 0.0, 700.0, 35, 0.0, 700.0)
        book_2d(f"h2_{prefix}_mu2_pt_vs_mu1_pt", f"Dimuon Transverse Momentum Correlation ({label});p_{{T,#mu1}} [GeV/c];p_{{T,#mu2}} [GeV/c]", "mu1_pt", "sec_mu_pt", node, 50, 0.0, 4.0, 50, 0.0, 1.5)
        book_2d(f"h2_{prefix}_mu2_eta_vs_mu1_eta", f"Dimuon Pseudorapidity Correlation ({label});#eta_{{#mu1}};#eta_{{#mu2}}", "mu1_eta", "sec_mu_eta", node, 50, 4.0, 10.0, 50, 4.0, 10.0)

    # Kinematic correlations for candidate signal, direct signal, and indirect fake signal
    for pref, node, lbl in [("signal", df_candidate, "Signal"), ("direct", df_direct, "Direct Signal"), ("indirect", df_indirect, "Indirect Fake")]:
        book_2d(f"h2_{pref}_mu1_e_vs_nu_e", f"Primary #mu_{{1}} Energy vs E_{{#nu}} ({lbl});E_{{#nu}} [GeV];E_{{#mu1}} [GeV]", "nu_e", "mu1_e", node, 50, 0.0, 800.0, 50, 0.0, 600.0)
        book_2d(f"h2_{pref}_mu2_e_vs_nu_e", f"Secondary #mu_{{2}} Energy vs E_{{#nu}} ({lbl});E_{{#nu}} [GeV];E_{{#mu2}} [GeV]", "nu_e", "sec_mu_e", node, 50, 0.0, 800.0, 50, 0.0, 10.0)
        book_2d(f"h2_{pref}_dimuon_delta_e_vs_nu_e", f"Muon Energy Difference vs E_{{#nu}} ({lbl});E_{{#nu}} [GeV];E_{{#mu1}} - E_{{#mu2}} [GeV]", "nu_e", "dimuon_delta_e_unified", node, 50, 0.0, 800.0, 50, -50.0, 600.0)
        book_2d(f"h2_{pref}_mu2_e_vs_charm_e", f"Secondary #mu_{{2}} Energy vs Charm Energy ({lbl});E_{{charm}} [GeV];E_{{#mu2}} [GeV]", "charm_e", "sec_mu_e", node, 50, 0.0, 500.0, 50, 0.0, 10.0)

    book_2d("h2_signal_dimuon_delta_e_vs_dimuon_mass", "Muon Energy Difference vs M_{#mu#mu};M_{#mu#mu} [GeV/c^{2}];E_{#mu1} - E_{#mu2} [GeV]", "dimuon_mass_unified", "dimuon_delta_e_unified", df_candidate, 60, 0.0, 15.0, 50, -50.0, 600.0)
    book_2d("h2_signal_dimuon_delta_e_vs_opening_angle", "Muon Energy Difference vs #theta_{#mu#mu};#theta_{#mu#mu} [mrad];E_{#mu1} - E_{#mu2} [GeV]", "dimuon_opening_angle_mrad_unified", "dimuon_delta_e_unified", df_candidate, 64, 0.0, 3200.0, 50, -50.0, 600.0)
    book_2d("h2_signal_mu2_e_vs_delta_e", "Secondary #mu_{2} Energy vs (E_{#mu1} - E_{#mu2});E_{#mu1} - E_{#mu2} [GeV];E_{#mu2}} [GeV]", "dimuon_delta_e_unified", "sec_mu_e", df_candidate, 50, -50.0, 600.0, 50, 0.0, 10.0)
    book_2d("h2_signal_dimuon_mass_vs_opening_angle", "Dimuon Mass vs Opening Angle;#theta_{#mu#mu} [mrad];M_{#mu#mu} [GeV/c^{2}]", "dimuon_opening_angle_mrad_unified", "dimuon_mass_unified", df_candidate, 64, 0.0, 3200.0, 60, 0.0, 15.0)
    book_2d("h2_signal_dimuon_mass_vs_pt", "Dimuon Mass vs Pair p_{T};p_{T,#mu#mu} [GeV/c];M_{#mu#mu} [GeV/c^{2}]", "dimuon_pt_unified", "dimuon_mass_unified", df_candidate, 50, 0.0, 8.0, 60, 0.0, 15.0)
    book_2d("h2_signal_opening_angle_vs_delta_phi", "Opening Angle vs Azimuthal Separation;#Delta#phi_{#mu#mu} [rad];#theta_{#mu#mu} [mrad]", "dimuon_delta_phi_unified", "dimuon_opening_angle_mrad_unified", df_candidate, 50, 0.0, 3.14159, 64, 0.0, 3200.0)
    book_2d("h2_signal_mu2_ptrel_vs_mu2_p", "Secondary #mu_{2} p_{T}^{rel} vs p_{#mu2};p_{#mu2} [GeV/c];p_{T}^{rel} [GeV/c]", "sec_mu_p", "mu2_ptrel", df_candidate, 50, 0.0, 10.0, 50, 0.0, 2.0)
    book_2d("h2_signal_mu2_ip3d_vs_decay_length", "Secondary #mu_{2} IP_{3D} vs Charm Flight Length;L_{3D} [cm];IP_{3D} [cm]", "decay_length_3d", "mu2_ip3d", df_candidate, 50, 0.0, 3.0, 50, 0.0, 0.3)
    book_2d("h2_signal_decay_length_vs_charm_p", "Charm Flight Length vs Momentum;p_{charm} [GeV/c];L_{3D} [cm]", "charm_p", "decay_length_3d", df_candidate, 50, 0.0, 500.0, 50, 0.0, 5.0)
    book_2d("h2_signal_q2_vs_bjorken_x", "Q^{2} vs Bjorken x;Bjorken x;Q^{2} [GeV^{2}]", "bjorken_x", "q2", df_candidate, 50, 0.0, 1.0, 50, 0.0, 50.0)
    book_2d("h2_signal_vtx_xy", "Interaction Vertex Transverse Profile;x_{vtx} [cm];y_{vtx} [cm]", "vtx_x", "vtx_y", df_candidate, 60, -60.0, 0.0, 60, 3.0, 63.0)
    book_2d("h2_signal_mu1_nds_vs_mu1_e", "Primary #mu_{1} DS MCPoints vs Energy;E_{#mu1} [GeV];N_{DS}(#mu_{1})", "mu1_e", "mu1_n_ds_points", df_candidate, 50, 0.0, 800.0, 20, -0.5, 19.5)
    book_2d("h2_signal_mu2_nds_vs_mu2_e", "Secondary #mu_{2} DS MCPoints vs Energy;E_{#mu2} [GeV];N_{DS}(#mu_{2})", "sec_mu_e", "mu2_n_ds_points", df_candidate, 50, 0.0, 10.0, 20, -0.5, 19.5)
    book_2d("h2_signal_mu2_nds_vs_mu2_eta", "Secondary #mu_{2} DS MCPoints vs #eta;#eta_{#mu2};N_{DS}(#mu_{2})", "sec_mu_eta", "mu2_n_ds_points", df_candidate, 50, 4.0, 10.0, 20, -0.5, 19.5)

    # --------------------------------------------------------------------------
    # 8b-4. XZ and YZ Projected Observables (Truth 2D Projections)
    # --------------------------------------------------------------------------
    proj_samples = [
        ("signal", df_candidate, "Inclusive"),
        ("direct", df_direct, "Direct Signal"),
        ("indirect", df_indirect, "Indirect Fake"),
    ]

    for pref, node, lbl in proj_samples:
        # Angles and slopes in XZ projection
        book_1d(f"h_{pref}_mu1_slope_xz", f"Primary #mu_{{1}} Slope XZ ({lbl});dx/dz (#mu_{{1}});Events", "mu1_slope_xz", node, 60, -0.3, 0.3)
        book_1d(f"h_{pref}_sec_mu_slope_xz", f"Secondary #mu_{{2}} Slope XZ ({lbl});dx/dz (#mu_{{2}});Events", "sec_mu_slope_xz", node, 60, -0.3, 0.3)
        book_1d(f"h_{pref}_dimuon_delta_slope_xz", f"Dimuon Slope Difference XZ ({lbl});|dx/dz_{{1}} - dx/dz_{{2}}|;Events", "dimuon_delta_slope_xz", node, 60, 0.0, 0.3)
        book_1d(f"h_{pref}_mu1_theta_xz_mrad", f"Primary #mu_{{1}} #theta_{{XZ}} ({lbl});#theta_{{XZ}}(#mu_{{1}}) [mrad];Events", "mu1_theta_xz_mrad", node, 60, -300.0, 300.0)
        book_1d(f"h_{pref}_sec_mu_theta_xz_mrad", f"Secondary #mu_{{2}} #theta_{{XZ}} ({lbl});#theta_{{XZ}}(#mu_{{2}}) [mrad];Events", "sec_mu_theta_xz_mrad", node, 60, -300.0, 300.0)
        book_1d(f"h_{pref}_dimuon_opening_angle_xz_mrad", f"Dimuon Opening Angle in XZ ({lbl});#Delta#theta_{{XZ}} [mrad];Events", "dimuon_opening_angle_xz_mrad", node, 60, 0.0, 300.0)

        # Angles and slopes in YZ projection
        book_1d(f"h_{pref}_mu1_slope_yz", f"Primary #mu_{{1}} Slope YZ ({lbl});dy/dz (#mu_{{1}});Events", "mu1_slope_yz", node, 60, -0.3, 0.3)
        book_1d(f"h_{pref}_sec_mu_slope_yz", f"Secondary #mu_{{2}} Slope YZ ({lbl});dy/dz (#mu_{{2}});Events", "sec_mu_slope_yz", node, 60, -0.3, 0.3)
        book_1d(f"h_{pref}_dimuon_delta_slope_yz", f"Dimuon Slope Difference YZ ({lbl});|dy/dz_{{1}} - dy/dz_{{2}}|;Events", "dimuon_delta_slope_yz", node, 60, 0.0, 0.3)
        book_1d(f"h_{pref}_mu1_theta_yz_mrad", f"Primary #mu_{{1}} #theta_{{YZ}} ({lbl});#theta_{{YZ}}(#mu_{{1}}) [mrad];Events", "mu1_theta_yz_mrad", node, 60, -300.0, 300.0)
        book_1d(f"h_{pref}_sec_mu_theta_yz_mrad", f"Secondary #mu_{{2}} #theta_{{YZ}} ({lbl});#theta_{{YZ}}(#mu_{{2}}) [mrad];Events", "sec_mu_theta_yz_mrad", node, 60, -300.0, 300.0)
        book_1d(f"h_{pref}_dimuon_opening_angle_yz_mrad", f"Dimuon Opening Angle in YZ ({lbl});#Delta#theta_{{YZ}} [mrad];Events", "dimuon_opening_angle_yz_mrad", node, 60, 0.0, 300.0)

        # Projected flight lengths
        book_1d(f"h_{pref}_decay_length_xz", f"Charm Flight Length in XZ ({lbl});L_{{XZ}} [cm];Events", "decay_length_xz", node, 50, 0.0, 5.0)
        book_1d(f"h_{pref}_decay_length_yz", f"Charm Flight Length in YZ ({lbl});L_{{YZ}} [cm];Events", "decay_length_yz", node, 50, 0.0, 5.0)

        # Spatial separations at detector planes
        # SciFi Entrance (z = 300 cm)
        book_1d(f"h_{pref}_sep_x_scifi_entrance", f"Dimuon Separation #Delta x at SciFi Entrance ({lbl});#Delta x [cm];Events", "dimuon_sep_x_scifi_entrance", node, 50, 0.0, 25.0)
        book_1d(f"h_{pref}_sep_y_scifi_entrance", f"Dimuon Separation #Delta y at SciFi Entrance ({lbl});#Delta y [cm];Events", "dimuon_sep_y_scifi_entrance", node, 50, 0.0, 25.0)
        book_1d(f"h_{pref}_sep_3d_scifi_entrance", f"Dimuon 3D Separation at SciFi Entrance ({lbl});#Delta R_{{3D}} [cm];Events", "dimuon_sep_3d_scifi_entrance", node, 50, 0.0, 35.0)

        # SciFi Exit (z = 390 cm)
        book_1d(f"h_{pref}_sep_x_scifi_exit", f"Dimuon Separation #Delta x at SciFi Exit ({lbl});#Delta x [cm];Events", "dimuon_sep_x_scifi_exit", node, 50, 0.0, 40.0)
        book_1d(f"h_{pref}_sep_y_scifi_exit", f"Dimuon Separation #Delta y at SciFi Exit ({lbl});#Delta y [cm];Events", "dimuon_sep_y_scifi_exit", node, 50, 0.0, 40.0)
        book_1d(f"h_{pref}_sep_3d_scifi_exit", f"Dimuon 3D Separation at SciFi Exit ({lbl});#Delta R_{{3D}} [cm];Events", "dimuon_sep_3d_scifi_exit", node, 50, 0.0, 50.0)

        # DS Entrance (z = 470 cm)
        book_1d(f"h_{pref}_sep_x_ds_entrance", f"Dimuon Separation #Delta x at DS Entrance ({lbl});#Delta x [cm];Events", "dimuon_sep_x_ds_entrance", node, 50, 0.0, 60.0)
        book_1d(f"h_{pref}_sep_y_ds_entrance", f"Dimuon Separation #Delta y at DS Entrance ({lbl});#Delta y [cm];Events", "dimuon_sep_y_ds_entrance", node, 50, 0.0, 60.0)
        book_1d(f"h_{pref}_sep_3d_ds_entrance", f"Dimuon 3D Separation at DS Entrance ({lbl});#Delta R_{{3D}} [cm];Events", "dimuon_sep_3d_ds_entrance", node, 50, 0.0, 80.0)

        # 2D Correlations
        book_2d(f"h2_{pref}_sec_mu_theta_xz_vs_mu1_theta_xz", f"#theta_{{XZ}}(#mu_{{2}}) vs #theta_{{XZ}}(#mu_{{1}}) ({lbl});#theta_{{XZ}}(#mu_{{1}}) [mrad];#theta_{{XZ}}(#mu_{{2}}) [mrad]", "mu1_theta_xz_mrad", "sec_mu_theta_xz_mrad", node, 50, -300.0, 300.0, 50, -300.0, 300.0)
        book_2d(f"h2_{pref}_sec_mu_theta_yz_vs_mu1_theta_yz", f"#theta_{{YZ}}(#mu_{{2}}) vs #theta_{{YZ}}(#mu_{{1}}) ({lbl});#theta_{{YZ}}(#mu_{{1}}) [mrad];#theta_{{YZ}}(#mu_{{2}}) [mrad]", "mu1_theta_yz_mrad", "sec_mu_theta_yz_mrad", node, 50, -300.0, 300.0, 50, -300.0, 300.0)
        book_2d(f"h2_{pref}_opening_angle_yz_vs_xz", f"Opening Angle YZ vs XZ ({lbl});#Delta#theta_{{XZ}} [mrad];#Delta#theta_{{YZ}} [mrad]", "dimuon_opening_angle_xz_mrad", "dimuon_opening_angle_yz_mrad", node, 50, 0.0, 300.0, 50, 0.0, 300.0)
        book_2d(f"h2_{pref}_sep_ds_entrance_y_vs_x", f"Dimuon Separation at DS Entrance ({lbl});#Delta x [cm];#Delta y [cm]", "dimuon_sep_x_ds_entrance", "dimuon_sep_y_ds_entrance", node, 50, 0.0, 60.0, 50, 0.0, 60.0)

    # --------------------------------------------------------------------------
    # 8b-5. Reconstructed DS Hough Tracks & Crossing Points (Reco_MuonTracks)
    # --------------------------------------------------------------------------
    reco_samples = [
        ("signal", df_candidate, df_signal_2ds, df_signal_2ds_valid_cross, "Inclusive"),
        ("direct", df_direct, df_direct_2ds, df_direct_2ds_valid_cross, "Direct Signal"),
        ("indirect", df_indirect, df_indirect_2ds, df_indirect_2ds_valid_cross, "Indirect Fake"),
    ]

    for pref, cand_node, node_2ds, node_cross, lbl in reco_samples:
        # Multiplicities
        book_1d(f"h_{pref}_n_reco_muon_tracks", f"Total Reconstructed Tracks ({lbl});N_{{tracks}};Events", "n_reco_muon_tracks", cand_node, 11, -0.5, 10.5)
        book_1d(f"h_{pref}_n_ds_hough_tracks", f"Reconstructed DS Hough Tracks ({lbl});N_{{DS tracks}} (type 13);Events", "n_ds_hough_tracks", cand_node, 11, -0.5, 10.5)
        book_1d(f"h_{pref}_has_two_ds_tracks", f"Has >=2 Reconstructed DS Tracks ({lbl});Has >=2 DS Tracks;Events", "has_two_ds_tracks", cand_node, 2, -0.5, 1.5)

        # For events with >= 2 DS tracks: Crossing Points & Vertex Comparison
        book_1d(f"h_{pref}_2ds_reco_cross_z_xz", f"DS Tracks Crossing z in XZ ({lbl});z_{{cross, XZ}} [cm];Events", "reco_ds_cross_z_xz", node_cross, 70, 150.0, 500.0)
        book_1d(f"h_{pref}_2ds_reco_cross_z_yz", f"DS Tracks Crossing z in YZ ({lbl});z_{{cross, YZ}} [cm];Events", "reco_ds_cross_z_yz", node_cross, 70, 150.0, 500.0)
        book_1d(f"h_{pref}_2ds_reco_cross_x", f"DS Tracks Crossing x ({lbl});x_{{cross}} [cm];Events", "reco_ds_cross_x", node_cross, 60, -60.0, 0.0)
        book_1d(f"h_{pref}_2ds_reco_cross_y", f"DS Tracks Crossing y ({lbl});y_{{cross}} [cm];Events", "reco_ds_cross_y", node_cross, 60, 5.0, 65.0)

        # True neutrino vertex z for the 2 DS track events
        book_1d(f"h_{pref}_2ds_truth_vtx_z", f"True Neutrino Interaction Vertex z for 2 DS Track Events ({lbl});z_{{vtx}}^{{true}} [cm];Events", "vtx_z", node_cross, 50, 260.0, 360.0)

        # Delta z between XZ and YZ crossings
        book_1d(f"h_{pref}_2ds_delta_z_cross", f"Crossing z Difference XZ - YZ ({lbl});#Delta z_{{cross}} = z_{{cross, XZ}} - z_{{cross, YZ}} [cm];Events", "reco_ds_delta_z_cross", node_cross, 60, -150.0, 150.0)
        book_1d(f"h_{pref}_2ds_abs_delta_z_cross", f"Absolute Crossing z Difference |XZ - YZ| ({lbl});|#Delta z_{{cross}}| [cm];Events", "reco_ds_abs_delta_z_cross", node_cross, 60, 0.0, 150.0)

        # Residuals relative to true neutrino vertex z
        book_1d(f"h_{pref}_2ds_res_z_xz", f"Crossing z Residual in XZ ({lbl});z_{{cross, XZ}} - z_{{vtx}}^{{true}} [cm];Events", "reco_ds_res_z_xz", node_cross, 60, -150.0, 150.0)
        book_1d(f"h_{pref}_2ds_res_z_yz", f"Crossing z Residual in YZ ({lbl});z_{{cross, YZ}} - z_{{vtx}}^{{true}} [cm];Events", "reco_ds_res_z_yz", node_cross, 60, -150.0, 150.0)

        # 2D Correlations
        book_2d(f"h2_{pref}_2ds_cross_z_yz_vs_xz", f"DS Crossing z_{{YZ}} vs z_{{XZ}} ({lbl});z_{{cross, XZ}} [cm];z_{{cross, YZ}} [cm]", "reco_ds_cross_z_xz", "reco_ds_cross_z_yz", node_cross, 60, 150.0, 450.0, 60, 150.0, 450.0)
        book_2d(f"h2_{pref}_2ds_cross_z_xz_vs_vtx_z", f"DS Crossing z_{{XZ}} vs True Vertex z ({lbl});z_{{vtx}}^{{true}} [cm];z_{{cross, XZ}} [cm]", "vtx_z", "reco_ds_cross_z_xz", node_cross, 50, 260.0, 360.0, 60, 150.0, 450.0)
        book_2d(f"h2_{pref}_2ds_cross_z_yz_vs_vtx_z", f"DS Crossing z_{{YZ}} vs True Vertex z ({lbl});z_{{vtx}}^{{true}} [cm];z_{{cross, YZ}} [cm]", "vtx_z", "reco_ds_cross_z_yz", node_cross, 50, 260.0, 360.0, 60, 150.0, 450.0)
        book_2d(f"h2_{pref}_2ds_cross_y_vs_x", f"DS Crossing y vs x ({lbl});x_{{cross}} [cm];y_{{cross}} [cm]", "reco_ds_cross_x", "reco_ds_cross_y", node_cross, 50, -60.0, 0.0, 50, 5.0, 65.0)



    # --------------------------------------------------------------------------
    # 8c. TProfiles: Kinematic Means & Acceptance Probability Profiles
    # --------------------------------------------------------------------------
    # Mean kinematics vs Neutrino Energy for candidate, direct, and indirect
    for pref, node, lbl in [("signal", df_candidate, "Inclusive"), ("direct", df_direct, "Direct Signal"), ("indirect", df_indirect, "Indirect Fake")]:
        book_prof(f"p_{pref}_mu1_e_vs_nu_e", f"Mean Primary #mu_{{1}} Energy vs E_{{#nu}} ({lbl});E_{{#nu}} [GeV];<E_{{#mu1}}> [GeV]", "nu_e", "mu1_e", node, 40, 0.0, 800.0)
        book_prof(f"p_{pref}_mu2_e_vs_nu_e", f"Mean Secondary #mu_{{2}} Energy vs E_{{#nu}} ({lbl});E_{{#nu}} [GeV];<E_{{#mu2}}> [GeV]", "nu_e", "sec_mu_e", node, 40, 0.0, 800.0)
        book_prof(f"p_{pref}_mu2_e_vs_charm_e", f"Mean Secondary #mu_{{2}} Energy vs Charm Energy ({lbl});E_{{charm}} [GeV];<E_{{#mu2}}> [GeV]", "charm_e", "sec_mu_e", node, 40, 0.0, 500.0)
        book_prof(f"p_{pref}_mu1_p_vs_nu_e", f"Mean Primary #mu_{{1}} Momentum vs E_{{#nu}} ({lbl});E_{{#nu}} [GeV];<p_{{#mu1}}> [GeV/c]", "nu_e", "mu1_p", node, 40, 0.0, 800.0)
        book_prof(f"p_{pref}_mu2_p_vs_nu_e", f"Mean Secondary #mu_{{2}} Momentum vs E_{{#nu}} ({lbl});E_{{#nu}} [GeV];<p_{{#mu2}}> [GeV/c]", "nu_e", "sec_mu_p", node, 40, 0.0, 800.0)
        book_prof(f"p_{pref}_mu1_pt_vs_nu_e", f"Mean Primary #mu_{{1}} p_{{T}} vs E_{{#nu}} ({lbl});E_{{#nu}} [GeV];<p_{{T,#mu1}}> [GeV/c]", "nu_e", "mu1_pt", node, 40, 0.0, 800.0)
        book_prof(f"p_{pref}_mu2_pt_vs_nu_e", f"Mean Secondary #mu_{{2}} p_{{T}} vs E_{{#nu}} ({lbl});E_{{#nu}} [GeV];<p_{{T,#mu2}}> [GeV/c]", "nu_e", "sec_mu_pt", node, 40, 0.0, 800.0)

    book_prof("p_signal_charm_p_vs_nu_e", "Mean Charmed Hadron Momentum vs E_{#nu};E_{#nu} [GeV];<p_{charm}> [GeV/c]", "nu_e", "charm_p", df_candidate, 40, 0.0, 800.0)
    book_prof("p_signal_dimuon_mass_vs_nu_e", "Mean Dimuon Invariant Mass vs E_{#nu};E_{#nu} [GeV];<M_{#mu#mu}> [GeV/c^{2}]", "nu_e", "dimuon_mass_unified", df_candidate, 30, 0.0, 800.0)
    book_prof("p_signal_opening_angle_vs_nu_e", "Mean Dimuon Opening Angle vs E_{#nu};E_{#nu} [GeV];<#theta_{#mu#mu}> [mrad]", "nu_e", "dimuon_opening_angle_mrad_unified", df_candidate, 30, 0.0, 800.0)
    book_prof("p_signal_charm_z_frac_vs_hadronic_w", "Mean Charm Energy Fraction vs Hadronic Mass W;W [GeV/c^{2}];<z_{c}>", "hadronic_w", "charm_z_frac", df_candidate, 30, 2.0, 30.0)
    book_prof("p_signal_decay_length_vs_charm_p", "Mean Charm Flight Length vs Momentum;p_{charm} [GeV/c];<L_{3D}> [cm]", "charm_p", "decay_length_3d", df_candidate, 30, 0.0, 400.0)
    book_prof("p_signal_ip3d_vs_decay_length", "Mean Secondary #mu_{2} IP_{3D} vs Flight Length;L_{3D} [cm];<IP_{3D}> [cm]", "decay_length_3d", "mu2_ip3d", df_candidate, 30, 0.0, 2.0)

    # Acceptance Probability Profiles (TProfile on 0/1 indicator = exact acceptance efficiency!)
    book_prof("p_acc_eff_vs_nu_e", "Dimuon DS Acceptance Efficiency vs E_{#nu};E_{#nu} [GeV];Acceptance Efficiency #epsilon_{DS}", "nu_e", "dimuon_in_ds_acc_double", df_candidate, 40, 0.0, 800.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu1_e", "Dimuon DS Acceptance Efficiency vs E_{#mu1};E_{#mu1} [GeV];Acceptance Efficiency #epsilon_{DS}", "mu1_e", "dimuon_in_ds_acc_double", df_candidate, 40, 0.0, 800.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu2_e", "Dimuon DS Acceptance Efficiency vs E_{#mu2};E_{#mu2} [GeV];Acceptance Efficiency #epsilon_{DS}", "sec_mu_e", "dimuon_in_ds_acc_double", df_candidate, 50, 0.0, 300.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_dimuon_delta_e", "Dimuon DS Acceptance Efficiency vs (E_{#mu1} - E_{#mu2});E_{#mu1} - E_{#mu2} [GeV];Acceptance Efficiency #epsilon_{DS}", "dimuon_delta_e_unified", "dimuon_in_ds_acc_double", df_candidate, 40, -50.0, 600.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_dimuon_abs_delta_e", "Dimuon DS Acceptance Efficiency vs |E_{#mu1} - E_{#mu2}|;|E_{#mu1} - E_{#mu2}| [GeV];Acceptance Efficiency #epsilon_{DS}", "dimuon_abs_delta_e_unified", "dimuon_in_ds_acc_double", df_candidate, 40, 0.0, 600.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_dimuon_energy_asym", "Dimuon DS Acceptance Efficiency vs Energy Asymmetry;A_{E} = (E_{1}-E_{2})/(E_{1}+E_{2});Acceptance Efficiency #epsilon_{DS}", "dimuon_energy_asym_unified", "dimuon_in_ds_acc_double", df_candidate, 40, -1.0, 1.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu1_p", "Dimuon DS Acceptance Efficiency vs p_{#mu1};p_{#mu1} [GeV/c];Acceptance Efficiency #epsilon_{DS}", "mu1_p", "dimuon_in_ds_acc_double", df_candidate, 40, 0.0, 800.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu2_p", "Dimuon DS Acceptance Efficiency vs p_{#mu2};p_{#mu2} [GeV/c];Acceptance Efficiency #epsilon_{DS}", "sec_mu_p", "dimuon_in_ds_acc_double", df_candidate, 50, 0.0, 300.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu1_pt", "Dimuon DS Acceptance Efficiency vs p_{T,#mu1};p_{T,#mu1} [GeV/c];Acceptance Efficiency #epsilon_{DS}", "mu1_pt", "dimuon_in_ds_acc_double", df_candidate, 40, 0.0, 4.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu2_pt", "Dimuon DS Acceptance Efficiency vs p_{T,#mu2};p_{T,#mu2} [GeV/c];Acceptance Efficiency #epsilon_{DS}", "sec_mu_pt", "dimuon_in_ds_acc_double", df_candidate, 50, 0.0, 5.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu1_eta", "Dimuon DS Acceptance Efficiency vs #eta_{#mu1};#eta_{#mu1};Acceptance Efficiency #epsilon_{DS}", "mu1_eta", "dimuon_in_ds_acc_double", df_candidate, 40, 4.0, 10.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_mu2_eta", "Dimuon DS Acceptance Efficiency vs #eta_{#mu2};#eta_{#mu2};Acceptance Efficiency #epsilon_{DS}", "sec_mu_eta", "dimuon_in_ds_acc_double", df_candidate, 40, 4.0, 10.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_opening_angle", "Dimuon DS Acceptance Efficiency vs #theta_{#mu#mu};#theta_{#mu#mu} [mrad];Acceptance Efficiency #epsilon_{DS}", "dimuon_opening_angle_mrad_unified", "dimuon_in_ds_acc_double", df_candidate, 64, 0.0, 3200.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_dimuon_mass", "Dimuon DS Acceptance Efficiency vs M_{#mu#mu};M_{#mu#mu} [GeV/c^{2}];Acceptance Efficiency #epsilon_{DS}", "dimuon_mass_unified", "dimuon_in_ds_acc_double", df_candidate, 60, 0.0, 15.0, 0.0, 1.0)
    book_prof("p_acc_eff_vs_vtx_z", "Dimuon DS Acceptance Efficiency vs z_{vtx};z_{vtx} [cm];Acceptance Efficiency #epsilon_{DS}", "vtx_z", "dimuon_in_ds_acc_double", df_candidate, 40, 260.0, 360.0, 0.0, 1.0)

    # Acceptance Profiles for Direct Signal and Indirect Fake Signal
    book_prof("p_direct_acc_eff_vs_nu_e", "Direct Signal DS Acceptance Efficiency vs E_{#nu};E_{#nu} [GeV];Acceptance #epsilon_{DS}", "nu_e", "dimuon_in_ds_acc_double", df_direct, 40, 0.0, 800.0, 0.0, 1.0)
    book_prof("p_direct_acc_eff_vs_mu2_e", "Direct Signal DS Acceptance Efficiency vs E_{#mu2};E_{#mu2} [GeV];Acceptance #epsilon_{DS}", "sec_mu_e", "dimuon_in_ds_acc_double", df_direct, 50, 0.0, 300.0, 0.0, 1.0)
    book_prof("p_direct_acc_eff_vs_dimuon_delta_e", "Direct Signal DS Acceptance vs #Delta E;#Delta E [GeV];Acceptance #epsilon_{DS}", "dimuon_delta_e_unified", "dimuon_in_ds_acc_double", df_direct, 40, -50.0, 600.0, 0.0, 1.0)

    book_prof("p_indirect_acc_eff_vs_nu_e", "Indirect Fake DS Acceptance Efficiency vs E_{#nu};E_{#nu} [GeV];Acceptance #epsilon_{DS}", "nu_e", "dimuon_in_ds_acc_double", df_indirect, 40, 0.0, 800.0, 0.0, 1.0)
    book_prof("p_indirect_acc_eff_vs_mu2_e", "Indirect Fake DS Acceptance Efficiency vs E_{#mu2};E_{#mu2} [GeV];Acceptance #epsilon_{DS}", "sec_mu_e", "dimuon_in_ds_acc_double", df_indirect, 50, 0.0, 50.0, 0.0, 1.0)
    book_prof("p_indirect_acc_eff_vs_dimuon_delta_e", "Indirect Fake DS Acceptance vs #Delta E;#Delta E [GeV];Acceptance #epsilon_{DS}", "dimuon_delta_e_unified", "dimuon_in_ds_acc_double", df_indirect, 40, -50.0, 600.0, 0.0, 1.0)

    book_prof("p_mu1_acc_eff_vs_mu1_e", "Primary #mu_{1} DS Acceptance vs E_{#mu1};E_{#mu1} [GeV];Acceptance Efficiency #epsilon_{DS}(#mu_{1})", "mu1_e", "mu1_in_ds_acc_double", df_candidate, 40, 0.0, 800.0, 0.0, 1.0)
    book_prof("p_mu1_acc_eff_vs_mu1_eta", "Primary #mu_{1} DS Acceptance vs #eta_{#mu1};#eta_{#mu1};Acceptance Efficiency #epsilon_{DS}(#mu_{1})", "mu1_eta", "mu1_in_ds_acc_double", df_candidate, 40, 4.0, 10.0, 0.0, 1.0)
    book_prof("p_mu2_acc_eff_vs_mu2_e", "Secondary #mu_{2} DS Acceptance vs E_{#mu2};E_{#mu2} [GeV];Acceptance Efficiency #epsilon_{DS}(#mu_{2})", "sec_mu_e", "mu2_in_ds_acc_double", df_candidate, 50, 0.0, 300.0, 0.0, 1.0)
    book_prof("p_mu2_acc_eff_vs_mu2_eta", "Secondary #mu_{2} DS Acceptance vs #eta_{#mu2};#eta_{#mu2};Acceptance Efficiency #epsilon_{DS}(#mu_{2})", "sec_mu_eta", "mu2_in_ds_acc_double", df_candidate, 40, 4.0, 10.0, 0.0, 1.0)
    book_prof("p_mu2_acc_eff_vs_mu2_pt", "Secondary #mu_{2} DS Acceptance vs p_{T,#mu2};p_{T,#mu2} [GeV/c];Acceptance Efficiency #epsilon_{DS}(#mu_{2})", "sec_mu_pt", "mu2_in_ds_acc_double", df_candidate, 50, 0.0, 5.0, 0.0, 1.0)

    # --------------------------------------------------------------------------
    # 8d. TEfficiency with Clopper-Pearson Intervals
    # --------------------------------------------------------------------------
    # 1. DS Acceptance Efficiencies
    eff_acc_targets = [
        ("signal", df_in_acc, df_candidate, "Inclusive"),
        ("direct", df_direct_in_acc, df_direct, "Direct Signal"),
        ("indirect", df_indirect_in_acc, df_indirect, "Indirect Fake"),
    ]

    for pref, node_p, node_t, lbl in eff_acc_targets:
        book_teff(f"teff_ds_acc_{pref}_vs_nu_e", f"DS Acceptance Efficiency vs E_{{#nu}} ({lbl});E_{{#nu}} [GeV];#epsilon_{{DS}}", "nu_e", node_p, node_t, 40, 0.0, 800.0)
        book_teff(f"teff_ds_acc_{pref}_vs_mu1_e", f"DS Acceptance Efficiency vs E_{{#mu1}} ({lbl});E_{{#mu1}} [GeV];#epsilon_{{DS}}", "mu1_e", node_p, node_t, 40, 0.0, 800.0)
        book_teff(f"teff_ds_acc_{pref}_vs_mu2_e", f"DS Acceptance Efficiency vs E_{{#mu2}} ({lbl});E_{{#mu2}} [GeV];#epsilon_{{DS}}", "sec_mu_e", node_p, node_t, 50, 0.0, 300.0)
        book_teff(f"teff_ds_acc_{pref}_vs_dimuon_delta_e", f"DS Acceptance Efficiency vs #Delta E ({lbl});E_{{#mu1}} - E_{{#mu2}} [GeV];#epsilon_{{DS}}", "dimuon_delta_e_unified", node_p, node_t, 40, -50.0, 600.0)
        book_teff(f"teff_ds_acc_{pref}_vs_dimuon_energy_asym", f"DS Acceptance Efficiency vs Energy Asymmetry ({lbl});A_{{E}};#epsilon_{{DS}}", "dimuon_energy_asym_unified", node_p, node_t, 40, -1.0, 1.0)
        book_teff(f"teff_ds_acc_{pref}_vs_vtx_z", f"DS Acceptance Efficiency vs z_{{vtx}} ({lbl});z_{{vtx}} [cm];#epsilon_{{DS}}", "vtx_z", node_p, node_t, 40, 260.0, 360.0)
        book_teff(f"teff_ds_acc_{pref}_vs_opening_angle_3d", f"DS Acceptance Efficiency vs #theta_{{3D}} ({lbl});#theta_{{#mu#mu}} [mrad];#epsilon_{{DS}}", "dimuon_opening_angle_mrad_unified", node_p, node_t, 50, 0.0, 2000.0)
        book_teff(f"teff_ds_acc_{pref}_vs_opening_angle_xz", f"DS Acceptance Efficiency vs #Delta#theta_{{XZ}} ({lbl});#Delta#theta_{{XZ}} [mrad];#epsilon_{{DS}}", "dimuon_opening_angle_xz_mrad", node_p, node_t, 50, 0.0, 500.0)
        book_teff(f"teff_ds_acc_{pref}_vs_opening_angle_yz", f"DS Acceptance Efficiency vs #Delta#theta_{{YZ}} ({lbl});#Delta#theta_{{YZ}} [mrad];#epsilon_{{DS}}", "dimuon_opening_angle_yz_mrad", node_p, node_t, 50, 0.0, 500.0)
        book_teff(f"teff_ds_acc_{pref}_vs_dimuon_mass", f"DS Acceptance Efficiency vs M_{{#mu#mu}} ({lbl});M_{{#mu#mu}} [GeV/c^{{2}}];#epsilon_{{DS}}", "dimuon_mass_unified", node_p, node_t, 50, 0.0, 15.0)

    # Single muon acceptance
    df_mu1_in_acc_node = df_candidate.Filter("mu1_in_ds_acceptance")
    book_teff("teff_ds_acc_mu1_vs_mu1_e", "Primary #mu_{1} DS Acceptance vs E_{#mu1};E_{#mu1} [GeV];#epsilon_{DS}(#mu_{1})", "mu1_e", df_mu1_in_acc_node, df_candidate, 40, 0.0, 800.0)
    book_teff("teff_ds_acc_mu1_vs_mu1_eta", "Primary #mu_{1} DS Acceptance vs #eta_{#mu1};#eta_{#mu1};#epsilon_{DS}(#mu_{1})", "mu1_eta", df_mu1_in_acc_node, df_candidate, 40, 4.0, 10.0)
    book_teff("teff_ds_acc_mu2_vs_mu2_e", "Secondary #mu_{2} DS Acceptance vs E_{#mu2};E_{#mu2} [GeV];#epsilon_{DS}(#mu_{2})", "sec_mu_e", df_mu2_in_acc, df_candidate, 50, 0.0, 300.0)
    book_teff("teff_ds_acc_mu2_vs_mu2_eta", "Secondary #mu_{2} DS Acceptance vs #eta_{#mu2};#eta_{#mu2};#epsilon_{DS}(#mu_{2})", "sec_mu_eta", df_mu2_in_acc, df_candidate, 40, 4.0, 10.0)

    # 2. Reconstructed Two DS Tracks Overall Efficiency (relative to candidate sample)
    eff_2ds_targets = [
        ("signal", df_signal_2ds, df_candidate, "Inclusive"),
        ("direct", df_direct_2ds, df_direct, "Direct Signal"),
        ("indirect", df_indirect_2ds, df_indirect, "Indirect Fake"),
    ]

    for pref, node_p, node_t, lbl in eff_2ds_targets:
        book_teff(f"teff_2ds_overall_{pref}_vs_nu_e", f"Two DS Tracks Efficiency (Overall) vs E_{{#nu}} ({lbl});E_{{#nu}} [GeV];#epsilon_{{2 DS}}", "nu_e", node_p, node_t, 40, 0.0, 800.0)
        book_teff(f"teff_2ds_overall_{pref}_vs_mu1_e", f"Two DS Tracks Efficiency (Overall) vs E_{{#mu1}} ({lbl});E_{{#mu1}} [GeV];#epsilon_{{2 DS}}", "mu1_e", node_p, node_t, 40, 0.0, 800.0)
        book_teff(f"teff_2ds_overall_{pref}_vs_mu2_e", f"Two DS Tracks Efficiency (Overall) vs E_{{#mu2}} ({lbl});E_{{#mu2}} [GeV];#epsilon_{{2 DS}}", "sec_mu_e", node_p, node_t, 50, 0.0, 300.0)
        book_teff(f"teff_2ds_overall_{pref}_vs_vtx_z", f"Two DS Tracks Efficiency (Overall) vs z_{{vtx}} ({lbl});z_{{vtx}} [cm];#epsilon_{{2 DS}}", "vtx_z", node_p, node_t, 40, 260.0, 360.0)
        book_teff(f"teff_2ds_overall_{pref}_vs_opening_angle_3d", f"Two DS Tracks Efficiency (Overall) vs #theta_{{3D}} ({lbl});#theta_{{#mu#mu}} [mrad];#epsilon_{{2 DS}}", "dimuon_opening_angle_mrad_unified", node_p, node_t, 50, 0.0, 2000.0)
        book_teff(f"teff_2ds_overall_{pref}_vs_opening_angle_xz", f"Two DS Tracks Efficiency (Overall) vs #Delta#theta_{{XZ}} ({lbl});#Delta#theta_{{XZ}} [mrad];#epsilon_{{2 DS}}", "dimuon_opening_angle_xz_mrad", node_p, node_t, 50, 0.0, 500.0)
        book_teff(f"teff_2ds_overall_{pref}_vs_opening_angle_yz", f"Two DS Tracks Efficiency (Overall) vs #Delta#theta_{{YZ}} ({lbl});#Delta#theta_{{YZ}} [mrad];#epsilon_{{2 DS}}", "dimuon_opening_angle_yz_mrad", node_p, node_t, 50, 0.0, 500.0)
        book_teff(f"teff_2ds_overall_{pref}_vs_sep_ds_entrance_x", f"Two DS Tracks Efficiency (Overall) vs #Delta x at DS ({lbl});#Delta x [cm];#epsilon_{{2 DS}}", "dimuon_sep_x_ds_entrance", node_p, node_t, 50, 0.0, 60.0)
        book_teff(f"teff_2ds_overall_{pref}_vs_sep_ds_entrance_y", f"Two DS Tracks Efficiency (Overall) vs #Delta y at DS ({lbl});#Delta y [cm];#epsilon_{{2 DS}}", "dimuon_sep_y_ds_entrance", node_p, node_t, 50, 0.0, 60.0)
        book_teff(f"teff_2ds_overall_{pref}_vs_sep_ds_entrance_3d", f"Two DS Tracks Efficiency (Overall) vs #Delta R_{{3D}} at DS ({lbl});#Delta R_{{3D}} [cm];#epsilon_{{2 DS}}", "dimuon_sep_3d_ds_entrance", node_p, node_t, 50, 0.0, 80.0)

    # 3. Reconstructed Two DS Tracks Within DS Acceptance (reconstruction efficiency given both muons reach DS)
    eff_2ds_in_acc_targets = [
        ("signal", df_signal_2ds_in_acc, df_in_acc, "Inclusive In DS Acc"),
        ("direct", df_direct_2ds_in_acc, df_direct_in_acc, "Direct Signal In DS Acc"),
        ("indirect", df_indirect_2ds_in_acc, df_indirect_in_acc, "Indirect Fake In DS Acc"),
    ]

    for pref, node_p, node_t, lbl in eff_2ds_in_acc_targets:
        book_teff(f"teff_2ds_in_acc_{pref}_vs_nu_e", f"Two DS Tracks Reconstruction Efficiency vs E_{{#nu}} ({lbl});E_{{#nu}} [GeV];#epsilon_{{2 DS | DS Acc}}", "nu_e", node_p, node_t, 40, 0.0, 800.0)
        book_teff(f"teff_2ds_in_acc_{pref}_vs_mu1_e", f"Two DS Tracks Reconstruction Efficiency vs E_{{#mu1}} ({lbl});E_{{#mu1}} [GeV];#epsilon_{{2 DS | DS Acc}}", "mu1_e", node_p, node_t, 40, 0.0, 800.0)
        book_teff(f"teff_2ds_in_acc_{pref}_vs_mu2_e", f"Two DS Tracks Reconstruction Efficiency vs E_{{#mu2}} ({lbl});E_{{#mu2}} [GeV];#epsilon_{{2 DS | DS Acc}}", "sec_mu_e", node_p, node_t, 50, 0.0, 300.0)
        book_teff(f"teff_2ds_in_acc_{pref}_vs_vtx_z", f"Two DS Tracks Reconstruction Efficiency vs z_{{vtx}} ({lbl});z_{{vtx}} [cm];#epsilon_{{2 DS | DS Acc}}", "vtx_z", node_p, node_t, 40, 260.0, 360.0)
        book_teff(f"teff_2ds_in_acc_{pref}_vs_opening_angle_3d", f"Two DS Tracks Reconstruction Efficiency vs #theta_{{3D}} ({lbl});#theta_{{#mu#mu}} [mrad];#epsilon_{{2 DS | DS Acc}}", "dimuon_opening_angle_mrad_unified", node_p, node_t, 50, 0.0, 2000.0)
        book_teff(f"teff_2ds_in_acc_{pref}_vs_opening_angle_xz", f"Two DS Tracks Reconstruction Efficiency vs #Delta#theta_{{XZ}} ({lbl});#Delta#theta_{{XZ}} [mrad];#epsilon_{{2 DS | DS Acc}}", "dimuon_opening_angle_xz_mrad", node_p, node_t, 50, 0.0, 500.0)
        book_teff(f"teff_2ds_in_acc_{pref}_vs_opening_angle_yz", f"Two DS Tracks Reconstruction Efficiency vs #Delta#theta_{{YZ}} ({lbl});#Delta#theta_{{YZ}} [mrad];#epsilon_{{2 DS | DS Acc}}", "dimuon_opening_angle_yz_mrad", node_p, node_t, 50, 0.0, 500.0)
        book_teff(f"teff_2ds_in_acc_{pref}_vs_sep_ds_entrance_x", f"Two DS Tracks Reconstruction Efficiency vs #Delta x ({lbl});#Delta x [cm];#epsilon_{{2 DS | DS Acc}}", "dimuon_sep_x_ds_entrance", node_p, node_t, 50, 0.0, 60.0)
        book_teff(f"teff_2ds_in_acc_{pref}_vs_sep_ds_entrance_y", f"Two DS Tracks Reconstruction Efficiency vs #Delta y ({lbl});#Delta y [cm];#epsilon_{{2 DS | DS Acc}}", "dimuon_sep_y_ds_entrance", node_p, node_t, 50, 0.0, 60.0)
        book_teff(f"teff_2ds_in_acc_{pref}_vs_sep_ds_entrance_3d", f"Two DS Tracks Reconstruction Efficiency vs #Delta R_{{3D}} ({lbl});#Delta R_{{3D}} [cm];#epsilon_{{2 DS | DS Acc}}", "dimuon_sep_3d_ds_entrance", node_p, node_t, 50, 0.0, 80.0)

    # --------------------------------------------------------------------------
    # 8e. Charmed Hadron Species Breakdowns
    # --------------------------------------------------------------------------
    species_samples = [
        ("d0", df_d0, "D^{0}"),
        ("dplus", df_dplus, "D^{+}"),
        ("ds", df_ds, "D_{s}^{+}"),
        ("lambdac", df_lambdac, "#Lambda_{c}^{+}"),
        ("direct_d0", df_direct_d0, "Direct D^{0}"),
        ("direct_dplus", df_direct_dplus, "Direct D^{+}"),
        ("direct_ds", df_direct_ds, "Direct D_{s}^{+}"),
        ("direct_lambdac", df_direct_lambdac, "Direct #Lambda_{c}^{+}"),
        ("indirect_d0", df_indirect_d0, "Indirect D^{0}"),
        ("indirect_dplus", df_indirect_dplus, "Indirect D^{+}"),
        ("indirect_ds", df_indirect_ds, "Indirect D_{s}^{+}"),
        ("indirect_lambdac", df_indirect_lambdac, "Indirect #Lambda_{c}^{+}"),
    ]

    for sp_key, sp_node, sp_lbl in species_samples:
        book_1d(f"h_species_{sp_key}_mu1_e", f"Primary #mu_{{1}} Energy ({sp_lbl});E_{{#mu1}} [GeV];Events", "mu1_e", sp_node, 50, 0.0, 600.0)
        book_1d(f"h_species_{sp_key}_mu2_e", f"Secondary #mu_{{2}} Energy ({sp_lbl});E_{{#mu2}} [GeV];Events", "sec_mu_e", sp_node, 50, 0.0, 300.0)
        book_1d(f"h_species_{sp_key}_dimuon_delta_e", f"Muon Energy Difference E_{{#mu1}} - E_{{#mu2}} ({sp_lbl});E_{{#mu1}} - E_{{#mu2}} [GeV];Events", "dimuon_delta_e_unified", sp_node, 50, -50.0, 600.0)
        book_1d(f"h_species_{sp_key}_charm_p", f"{sp_lbl} Momentum;p_{{charm}} [GeV/c];Events", "charm_p", sp_node, 50, 0.0, 500.0)
        book_1d(f"h_species_{sp_key}_decay_length", f"{sp_lbl} 3D Flight Length;L_{{3D}} [cm];Events", "decay_length_3d", sp_node, 50, 0.0, 5.0)
        book_1d(f"h_species_{sp_key}_lifetime", f"{sp_lbl} Proper Lifetime;#tau [ps];Events", "proper_lifetime_ps", sp_node, 50, 0.0, 3.0)
        book_1d(f"h_species_{sp_key}_ip3d", f"Secondary #mu_{{2}} IP_{{3D}} ({sp_lbl});IP_{{3D}} [cm];Events", "mu2_ip3d", sp_node, 50, 0.0, 0.3)
        book_1d(f"h_species_{sp_key}_dimuon_mass", f"Dimuon Invariant Mass ({sp_lbl});M_{{#mu#mu}} [GeV/c^{{2}}];Events", "dimuon_mass_unified", sp_node, 60, 0.0, 15.0)

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
    for title, h_p, h_t in booked_teff_pairs.values():
        all_actions.append(h_p)
        all_actions.append(h_t)

    n_total_plots = len(booked_histograms_1d) + len(booked_histograms_2d) + len(booked_profiles) + len(booked_teff_pairs)
    print(f"\nRunning RDataFrame computation graph over {n_total_plots} booked objects "
          f"({len(booked_histograms_1d)} 1D, {len(booked_histograms_2d)} 2D, {len(booked_profiles)} Profiles, {len(booked_teff_pairs)} TEfficiencies)...")
    ROOT.RDF.RunGraphs(all_actions)

    # Instantiate TEfficiency objects with Clopper-Pearson confidence intervals
    teff_objects = {}
    for name, (title, h_pass_res, h_tot_res) in booked_teff_pairs.items():
        h_p = h_pass_res.GetValue()
        h_t = h_tot_res.GetValue()
        if ROOT.TEfficiency.CheckConsistency(h_p, h_t):
            eff = ROOT.TEfficiency(h_p, h_t)
            eff.SetName(name)
            eff.SetTitle(title)
            eff.SetStatisticOption(ROOT.TEfficiency.kFCP) # Clopper-Pearson
            teff_objects[name] = eff
        else:
            print(f"Warning: Consistency check failed for TEfficiency {name}")

    # 10. Print Comprehensive Cutflow, Acceptance, and Kinematic Summary
    n_tot = c_total.GetValue()
    n_all_numu = c_all_numu.GetValue()
    n_all_numu_fid = c_all_numu_fid.GetValue()
    n_numu_cc = c_numu_cc.GetValue()
    n_numu_nc = c_numu_nc.GetValue()
    n_numu_nu = c_numu_nu.GetValue()
    n_numu_antinu = c_numu_antinu.GetValue()
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
    n_dir = c_direct.GetValue()
    n_indir = c_indirect.GetValue()
    n_dir_in_acc = c_direct_in_acc.GetValue()
    n_dir_out_acc = c_direct_out_acc.GetValue()
    n_indir_in_acc = c_indirect_in_acc.GetValue()
    n_indir_out_acc = c_indirect_out_acc.GetValue()
    n_indir_pion = c_indirect_pion.GetValue()
    n_indir_kaon = c_indirect_kaon.GetValue()
    n_indir_other = c_indirect_other.GetValue()
    n_dir_d0 = c_direct_d0.GetValue()
    n_dir_dplus = c_direct_dplus.GetValue()
    n_dir_ds = c_direct_ds.GetValue()
    n_dir_lambdac = c_direct_lambdac.GetValue()
    n_indir_d0 = c_indirect_d0.GetValue()
    n_indir_dplus = c_indirect_dplus.GetValue()
    n_indir_ds = c_indirect_ds.GetValue()
    n_indir_lambdac = c_indirect_lambdac.GetValue()
    n_2ds_signal = c_2ds_signal.GetValue()
    n_2ds_direct = c_2ds_direct.GetValue()
    n_2ds_indirect = c_2ds_indirect.GetValue()
    n_2ds_signal_in_acc = c_2ds_signal_in_acc.GetValue()
    n_2ds_direct_in_acc = c_2ds_direct_in_acc.GetValue()
    n_2ds_indirect_in_acc = c_2ds_indirect_in_acc.GetValue()

    print("\n" + "=" * 80)
    print(" ALL INTERACTING MUON NEUTRINOS IN DETECTOR")
    print("=" * 80)
    print(f"Total Events Processed       : {n_tot:8d}")
    print(f"All Interacting Nu_mu (CC+NC): {n_all_numu:8d} ({100.0 * n_all_numu / max(n_tot, 1):6.2f}%)")
    print(f"  - Target Fiducial (CC+NC)  : {n_all_numu_fid:8d} ({100.0 * n_all_numu_fid / max(n_all_numu, 1):6.2f}%)")
    print(f"  - Charged Current (CC)     : {n_numu_cc:8d} ({100.0 * n_numu_cc / max(n_all_numu, 1):6.2f}%)")
    print(f"  - Neutral Current (NC)     : {n_numu_nc:8d} ({100.0 * n_numu_nc / max(n_all_numu, 1):6.2f}%)")
    print(f"  - Nu_mu (PDG +14)          : {n_numu_nu:8d} ({100.0 * n_numu_nu / max(n_all_numu, 1):6.2f}%)")
    print(f"  - Anti-Nu_mu (PDG -14)     : {n_numu_antinu:8d} ({100.0 * n_numu_antinu / max(n_all_numu, 1):6.2f}%)")
    if "h_all_numu_nu_e" in booked_histograms_1d:
        h_all = booked_histograms_1d["h_all_numu_nu_e"].GetValue()
        print(f"Mean Nu_mu Energy (Inclusive): {h_all.GetMean():8.2f} GeV (RMS: {h_all.GetRMS():.2f} GeV)")
    if "h_all_numu_fiducial_nu_e" in booked_histograms_1d:
        print(f"Mean Nu_mu Energy (Fiducial) : {booked_histograms_1d['h_all_numu_fiducial_nu_e'].GetValue().GetMean():8.2f} GeV")
    if "h_all_numu_cc_nu_e" in booked_histograms_1d:
        print(f"Mean Nu_mu Energy (CC)       : {booked_histograms_1d['h_all_numu_cc_nu_e'].GetValue().GetMean():8.2f} GeV")
    if "h_all_numu_nc_nu_e" in booked_histograms_1d:
        print(f"Mean Nu_mu Energy (NC)       : {booked_histograms_1d['h_all_numu_nc_nu_e'].GetValue().GetMean():8.2f} GeV")
    print("-" * 80)
    print(" CHARM & SIGNAL DIMUON CUTFLOW")
    print("-" * 80)
    print(f"Muon Neutrino CC             : {n_numu_cc:8d} ({100.0 * n_numu_cc / max(n_tot, 1):6.2f}%)")
    print(f"Target Fiducial Volume CC    : {n_fid:8d} ({100.0 * n_fid / max(n_numu_cc, 1):6.2f}% of CC)")
    print(f"Nu_mu CC with Charm Hadron   : {n_chm:8d} ({100.0 * n_chm / max(n_numu_cc, 1):6.2f}% of CC)")
    print(f"Inclusive Dimuon Candidates  : {n_cand:8d} ({100.0 * n_cand / max(n_chm, 1):6.2f}% of charm)")
    print(f"  - Direct Charm Signal (c->mu): {n_dir:6d} ({100.0 * n_dir / max(n_cand, 1):6.2f}% of inclusive)")
    print(f"  - Indirect Fake Signal (c->h->mu): {n_indir:4d} ({100.0 * n_indir / max(n_cand, 1):6.2f}% of inclusive)")
    print(f"      * via Charged Pion (pi): {n_indir_pion:6d} ({100.0 * n_indir_pion / max(n_indir, 1):6.2f}% of indirect)")
    print(f"      * via Kaon (K)         : {n_indir_kaon:6d} ({100.0 * n_indir_kaon / max(n_indir, 1):6.2f}% of indirect)")
    print(f"      * via Other Hadron     : {n_indir_other:6d} ({100.0 * n_indir_other / max(n_indir, 1):6.2f}% of indirect)")
    print("-" * 80)
    print(" DOWNSTREAM (DS) MUFILTER ACCEPTANCE BREAKDOWN")
    print("-" * 80)
    print(f"Inclusive In DS Acceptance   : {n_in_acc:8d} ({100.0 * n_in_acc / max(n_cand, 1):6.2f}% of inclusive)")
    print(f"  - Direct In DS Acceptance  : {n_dir_in_acc:8d} ({100.0 * n_dir_in_acc / max(n_dir, 1):6.2f}% of direct)")
    print(f"  - Indirect In DS Acceptance: {n_indir_in_acc:8d} ({100.0 * n_indir_in_acc / max(n_indir, 1):6.2f}% of indirect)")
    print(f"Inclusive Outside DS         : {n_out_acc:8d} ({100.0 * n_out_acc / max(n_cand, 1):6.2f}% of inclusive)")
    print(f"Primary Muon (mu1) in DS     : {n_mu1_acc:8d} ({100.0 * n_mu1_acc / max(n_cand, 1):6.2f}% of inclusive)")
    print(f"Charm Decay Muon (mu2) in DS : {n_mu2_acc:8d} ({100.0 * n_mu2_acc / max(n_cand, 1):6.2f}% of inclusive)")
    print("-" * 80)
    print(" RECONSTRUCTED DS HOUGH TRACKS BREAKDOWN (sndRecoTrack::getTrackType() == 13)")
    print("-" * 80)
    print(f"Inclusive >=2 DS Tracks       : {n_2ds_signal:8d} ({100.0 * n_2ds_signal / max(n_cand, 1):6.2f}% of inclusive candidates)")
    print(f"  - Direct Signal >=2 DS Tracks: {n_2ds_direct:8d} ({100.0 * n_2ds_direct / max(n_dir, 1):6.2f}% of direct signal)")
    print(f"  - Indirect Fake >=2 DS Tracks: {n_2ds_indirect:8d} ({100.0 * n_2ds_indirect / max(n_indir, 1):6.2f}% of indirect fake)")
    print(f"Within DS Acceptance:")
    print(f"Inclusive >=2 DS Tracks In Acc: {n_2ds_signal_in_acc:8d} ({100.0 * n_2ds_signal_in_acc / max(n_in_acc, 1):6.2f}% of in-acc inclusive)")
    print(f"  - Direct Signal In Acc       : {n_2ds_direct_in_acc:8d} ({100.0 * n_2ds_direct_in_acc / max(n_dir_in_acc, 1):6.2f}% of in-acc direct)")
    print(f"  - Indirect Fake In Acc       : {n_2ds_indirect_in_acc:8d} ({100.0 * n_2ds_indirect_in_acc / max(n_indir_in_acc, 1):6.2f}% of in-acc indirect)")
    if "h_signal_2ds_reco_cross_z_xz" in booked_histograms_1d:
        print("-" * 80)
        print(" RECONSTRUCTED DS TRACKS CROSSING VERTEX (FOR EVENTS WITH >= 2 DS TRACKS)")
        print("-" * 80)
        print(f"{'Observable':<36} | {'Inclusive':>13} | {'Direct Signal':>13} | {'Indirect Fake':>13}")
        print("-" * 80)
        reco_cross_obs = [
            ("Crossing z in XZ [cm]", "h_signal_2ds_reco_cross_z_xz", "h_direct_2ds_reco_cross_z_xz", "h_indirect_2ds_reco_cross_z_xz"),
            ("Crossing z in YZ [cm]", "h_signal_2ds_reco_cross_z_yz", "h_direct_2ds_reco_cross_z_yz", "h_indirect_2ds_reco_cross_z_yz"),
            ("Delta z (XZ - YZ) [cm]", "h_signal_2ds_delta_z_cross", "h_direct_2ds_delta_z_cross", "h_indirect_2ds_delta_z_cross"),
            ("|Delta z| (XZ - YZ) [cm]", "h_signal_2ds_abs_delta_z_cross", "h_direct_2ds_abs_delta_z_cross", "h_indirect_2ds_abs_delta_z_cross"),
            ("True Vertex z [cm]", "h_signal_2ds_truth_vtx_z", "h_direct_2ds_truth_vtx_z", "h_indirect_2ds_truth_vtx_z"),
            ("Residual z in XZ (z_xz - z_vtx)", "h_signal_2ds_res_z_xz", "h_direct_2ds_res_z_xz", "h_indirect_2ds_res_z_xz"),
            ("Residual z in YZ (z_yz - z_vtx)", "h_signal_2ds_res_z_yz", "h_direct_2ds_res_z_yz", "h_indirect_2ds_res_z_yz"),
        ]
        for label_str, h_inc_name, h_dir_name, h_indir_name in reco_cross_obs:
            m_inc = booked_histograms_1d[h_inc_name].GetValue().GetMean() if h_inc_name in booked_histograms_1d else 0.0
            m_dir = booked_histograms_1d[h_dir_name].GetValue().GetMean() if h_dir_name in booked_histograms_1d else 0.0
            m_indir = booked_histograms_1d[h_indir_name].GetValue().GetMean() if h_indir_name in booked_histograms_1d else 0.0
            print(f"{label_str:<36} | {m_inc:13.2f} | {m_dir:13.2f} | {m_indir:13.2f}")
    print("-" * 80)
    print(" CHARMED HADRON SPECIES IN SIGNAL SAMPLE")
    print("-" * 80)
    print(f"{'Species':<24} | {'Inclusive':>12} | {'Direct Signal':>14} | {'Indirect Fake':>14}")
    print("-" * 80)
    print(f"{'D0 / anti-D0':<24} | {n_d0:12d} | {n_dir_d0:14d} | {n_indir_d0:14d}")
    print(f"{'D+ / D-':<24} | {n_dplus:12d} | {n_dir_dplus:14d} | {n_indir_dplus:14d}")
    print(f"{'Ds+ / Ds-':<24} | {n_ds:12d} | {n_dir_ds:14d} | {n_indir_ds:14d}")
    print(f"{'Lambda_c+ / Lambdac-':<24} | {n_lambdac:12d} | {n_dir_lambdac:14d} | {n_indir_lambdac:14d}")
    print("=" * 80)

    # Print Key Kinematic Means Comparison (Inclusive vs In-Acceptance vs Outside-Acceptance)
    print("\n" + "=" * 80)
    print(" KINEMATIC COMPARISON: INCLUSIVE vs IN-ACCEPTANCE vs OUTSIDE-ACCEPTANCE")
    print("=" * 80)
    print(f"{'Observable':<32} | {'Inclusive':>14} | {'In-Acceptance':>14} | {'Outside-Acc':>14}")
    print("-" * 80)

    key_obs = [
        ("Primary Muon Energy E(mu1) [GeV]", "h_signal_mu1_e", "h_in_acc_mu1_e", "h_out_acc_mu1_e"),
        ("Secondary Muon Energy E(mu2) [GeV]", "h_signal_mu2_e", "h_in_acc_mu2_e", "h_out_acc_mu2_e"),
        ("Primary Muon Momentum p(mu1) [GeV/c]", "h_signal_mu1_p", "h_in_acc_mu1_p", "h_out_acc_mu1_p"),
        ("Secondary Muon Momentum p(mu2) [GeV/c]", "h_signal_mu2_p", "h_in_acc_mu2_p", "h_out_acc_mu2_p"),
        ("Primary Muon pT [GeV/c]", "h_signal_mu1_pt", "h_in_acc_mu1_pt", "h_out_acc_mu1_pt"),
        ("Secondary Muon pT [GeV/c]", "h_signal_mu2_pt", "h_in_acc_mu2_pt", "h_out_acc_mu2_pt"),
        ("Primary Muon Pseudorapidity", "h_signal_mu1_eta", "h_in_acc_mu1_eta", "h_out_acc_mu1_eta"),
        ("Secondary Muon Pseudorapidity", "h_signal_mu2_eta", "h_in_acc_mu2_eta", "h_out_acc_mu2_eta"),
        ("Dimuon Invariant Mass [GeV/c^2]", "h_signal_dimuon_mass", "h_in_acc_dimuon_mass", "h_out_acc_dimuon_mass"),
        ("Dimuon Opening Angle [mrad]", "h_signal_dimuon_opening_angle_mrad", "h_in_acc_dimuon_opening_angle_mrad", "h_out_acc_dimuon_opening_angle_mrad"),
        ("Dimuon Energy Sum [GeV]", "h_signal_dimuon_e_sum", "h_in_acc_dimuon_e_sum", "h_out_acc_dimuon_e_sum"),
        ("Dimuon Energy Diff [GeV]", "h_signal_dimuon_delta_e", "h_in_acc_dimuon_delta_e", "h_out_acc_dimuon_delta_e"),
        ("Dimuon Energy Asymmetry", "h_signal_dimuon_energy_asym", "h_in_acc_dimuon_energy_asym", "h_out_acc_dimuon_energy_asym"),
        ("Neutrino Energy E(nu) [GeV]", "h_signal_nu_e", "h_in_acc_nu_e", "h_out_acc_nu_e"),
        ("Charm 3D Flight Length [cm]", "h_signal_decay_length_3d", "h_in_acc_decay_length_3d", "h_out_acc_decay_length_3d"),
        ("Secondary Muon IP_3D [cm]", "h_signal_mu2_ip3d", "h_in_acc_mu2_ip3d", "h_out_acc_mu2_ip3d"),
    ]

    for label_str, h_inc_name, h_in_name, h_out_name in key_obs:
        m_inc = booked_histograms_1d[h_inc_name].GetValue().GetMean() if h_inc_name in booked_histograms_1d else 0.0
        m_in = booked_histograms_1d[h_in_name].GetValue().GetMean() if h_in_name in booked_histograms_1d else 0.0
        m_out = booked_histograms_1d[h_out_name].GetValue().GetMean() if h_out_name in booked_histograms_1d else 0.0
        print(f"{label_str:<32} | {m_inc:14.4f} | {m_in:14.4f} | {m_out:14.4f}")
    print("=" * 80)

    # Print Key Kinematic Means Comparison: Direct Signal vs Indirect Fake Signal
    print("\n" + "=" * 80)
    print(" KINEMATIC COMPARISON: DIRECT SIGNAL vs INDIRECT FAKE SIGNAL")
    print("=" * 80)
    print(f"{'Observable':<32} | {'Inclusive':>14} | {'Direct Signal':>14} | {'Indirect Fake':>14}")
    print("-" * 80)

    direct_vs_indirect_obs = [
        ("Primary Muon Energy E(mu1) [GeV]", "h_signal_mu1_e", "h_direct_mu1_e", "h_indirect_mu1_e"),
        ("Secondary Muon Energy E(mu2) [GeV]", "h_signal_mu2_e", "h_direct_mu2_e", "h_indirect_mu2_e"),
        ("Primary Muon Momentum p(mu1) [GeV/c]", "h_signal_mu1_p", "h_direct_mu1_p", "h_indirect_mu1_p"),
        ("Secondary Muon Momentum p(mu2) [GeV/c]", "h_signal_mu2_p", "h_direct_mu2_p", "h_indirect_mu2_p"),
        ("Primary Muon pT [GeV/c]", "h_signal_mu1_pt", "h_direct_mu1_pt", "h_indirect_mu1_pt"),
        ("Secondary Muon pT [GeV/c]", "h_signal_mu2_pt", "h_direct_mu2_pt", "h_indirect_mu2_pt"),
        ("Primary Muon Pseudorapidity", "h_signal_mu1_eta", "h_direct_mu1_eta", "h_indirect_mu1_eta"),
        ("Secondary Muon Pseudorapidity", "h_signal_mu2_eta", "h_direct_mu2_eta", "h_indirect_mu2_eta"),
        ("Dimuon Invariant Mass [GeV/c^2]", "h_signal_dimuon_mass", "h_direct_dimuon_mass", "h_indirect_dimuon_mass"),
        ("Dimuon Opening Angle [mrad]", "h_signal_dimuon_opening_angle_mrad", "h_direct_dimuon_opening_angle_mrad", "h_indirect_dimuon_opening_angle_mrad"),
        ("Dimuon Energy Sum [GeV]", "h_signal_dimuon_e_sum", "h_direct_dimuon_e_sum", "h_indirect_dimuon_e_sum"),
        ("Dimuon Energy Diff [GeV]", "h_signal_dimuon_delta_e", "h_direct_dimuon_delta_e", "h_indirect_dimuon_delta_e"),
        ("Dimuon Energy Asymmetry", "h_signal_dimuon_energy_asym", "h_direct_dimuon_energy_asym", "h_indirect_dimuon_energy_asym"),
        ("Neutrino Energy E(nu) [GeV]", "h_signal_nu_e", "h_direct_nu_e", "h_indirect_nu_e"),
        ("Charm Hadron Energy [GeV]", "h_signal_charm_e", "h_direct_charm_e", "h_indirect_charm_e"),
        ("Charm 3D Flight Length [cm]", "h_signal_decay_length_3d", "h_direct_decay_length_3d", "h_indirect_decay_length_3d"),
    ]

    for label_str, h_inc_name, h_dir_name, h_indir_name in direct_vs_indirect_obs:
        m_inc = booked_histograms_1d[h_inc_name].GetValue().GetMean() if h_inc_name in booked_histograms_1d else 0.0
        m_dir = booked_histograms_1d[h_dir_name].GetValue().GetMean() if h_dir_name in booked_histograms_1d else 0.0
        m_indir = booked_histograms_1d[h_indir_name].GetValue().GetMean() if h_indir_name in booked_histograms_1d else 0.0
        print(f"{label_str:<32} | {m_inc:14.4f} | {m_dir:14.4f} | {m_indir:14.4f}")
    print("=" * 80)

    # 11. Save Histograms, Profiles & TEfficiencies into Hierarchical Output ROOT File
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_file = ROOT.TFile.Open(output_file, "RECREATE")
    if out_file and not out_file.IsZombie():
        out_file.cd()

        # 11a. Write top-level key overview histograms for quick inspection
        top_overview_keys = [
            "h_all_numu_nu_e",
            "h_signal_mu1_e",
            "h_signal_mu2_e",
            "h2_signal_mu2_e_vs_mu1_e",
            "h2_signal_mu2_e_vs_mu1_e_zoom",
            "h_signal_dimuon_opening_angle_mrad",
            "h_signal_dimuon_mass",
            "teff_ds_acc_signal_vs_nu_e",
            "teff_2ds_overall_signal_vs_nu_e",
        ]
        for key in top_overview_keys:
            if key in booked_histograms_1d:
                booked_histograms_1d[key].GetValue().Write()
            elif key in booked_histograms_2d:
                booked_histograms_2d[key].GetValue().Write()
            elif key in booked_profiles:
                booked_profiles[key].GetValue().Write()
            elif key in teff_objects:
                teff_objects[key].Write()

        # 11b. Hierarchical structured subdirectories
        # 1. all_muon_neutrinos
        d_numu = out_file.mkdir("all_muon_neutrinos")
        d_numu.cd()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_all_numu_"):
                h_res.GetValue().Write()
        for name, h2_res in booked_histograms_2d.items():
            if name.startswith("h2_all_numu_"):
                h2_res.GetValue().Write()

        # Helper to separate projected and reco variables from pure 3D folders
        def is_proj_or_reco(n):
            return any(x in n for x in ["_slope_", "_theta_xz", "_theta_yz", "_opening_angle_xz", "_opening_angle_yz", "_sep_", "_reco_", "_2ds_", "has_two_ds"])

        # 2. truth_3d
        d_truth3d = out_file.mkdir("truth_3d")
        d_truth3d_inc = d_truth3d.mkdir("inclusive_signal")
        d_truth3d_dir = d_truth3d.mkdir("direct_signal")
        d_truth3d_indir = d_truth3d.mkdir("indirect_fake_signal")
        d_truth3d_inter = d_truth3d_indir.mkdir("intermediate_daughters")
        d_truth3d_2d = d_truth3d.mkdir("correlations_2d")
        d_truth3d_prof = d_truth3d.mkdir("profiles")

        d_truth3d_inc.cd()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_signal_") and not is_proj_or_reco(name):
                h_res.GetValue().Write()

        d_truth3d_dir.cd()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_direct_") and not ("_in_acc" in name or "_out_acc" in name or is_proj_or_reco(name)):
                h_res.GetValue().Write()

        d_truth3d_indir.cd()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_indirect_") and not ("_in_acc" in name or "_out_acc" in name or is_proj_or_reco(name) or any(x in name for x in ["downstream_muon", "mother", "pion", "kaon", "other"])):
                h_res.GetValue().Write()

        d_truth3d_inter.cd()
        for name, h_res in booked_histograms_1d.items():
            if any(name.startswith(x) for x in ["h_indirect_downstream_muon_", "h_indirect_mother_", "h_indirect_pion_", "h_indirect_kaon_", "h_indirect_other_"]):
                h_res.GetValue().Write()

        d_truth3d_2d.cd()
        for name, h2_res in booked_histograms_2d.items():
            if (name.startswith("h2_signal_") or name.startswith("h2_direct_") or name.startswith("h2_indirect_")) and not is_proj_or_reco(name):
                h2_res.GetValue().Write()

        d_truth3d_prof.cd()
        for name, p_res in booked_profiles.items():
            if (name.startswith("p_signal_") or name.startswith("p_direct_") or name.startswith("p_indirect_")) and not ("acc_eff" in name):
                p_res.GetValue().Write()

        # 3. truth_projections_xz_yz
        d_proj = out_file.mkdir("truth_projections_xz_yz")
        d_proj_xz = d_proj.mkdir("angles_and_slopes_xz")
        d_proj_yz = d_proj.mkdir("angles_and_slopes_yz")
        d_proj_sep = d_proj.mkdir("spatial_separations")
        d_proj_decay = d_proj.mkdir("decay_lengths")
        d_proj_2d = d_proj.mkdir("correlations_2d")

        d_proj_xz.cd()
        for name, h_res in booked_histograms_1d.items():
            if any(x in name for x in ["_slope_xz", "_theta_xz_mrad", "_opening_angle_xz_mrad", "_delta_slope_xz"]):
                h_res.GetValue().Write()

        d_proj_yz.cd()
        for name, h_res in booked_histograms_1d.items():
            if any(x in name for x in ["_slope_yz", "_theta_yz_mrad", "_opening_angle_yz_mrad", "_delta_slope_yz"]):
                h_res.GetValue().Write()

        d_proj_sep.cd()
        for name, h_res in booked_histograms_1d.items():
            if "_sep_" in name:
                h_res.GetValue().Write()

        d_proj_decay.cd()
        for name, h_res in booked_histograms_1d.items():
            if any(x in name for x in ["_decay_length_xz", "_decay_length_yz"]):
                h_res.GetValue().Write()

        d_proj_2d.cd()
        for name, h2_res in booked_histograms_2d.items():
            if any(x in name for x in ["_theta_xz_", "_theta_yz_", "opening_angle_yz_vs_xz", "_sep_ds_entrance_"]):
                h2_res.GetValue().Write()

        # 4. reco_ds_tracks
        d_reco = out_file.mkdir("reco_ds_tracks")
        d_reco_mult = d_reco.mkdir("multiplicities")
        d_reco_cross = d_reco.mkdir("crossing_points")
        d_reco_res = d_reco.mkdir("delta_z_and_residuals")
        d_reco_2d = d_reco.mkdir("correlations_2d")

        d_reco_mult.cd()
        for name, h_res in booked_histograms_1d.items():
            if any(x in name for x in ["_n_reco_muon_tracks", "_n_ds_hough_tracks", "_has_two_ds_tracks"]):
                h_res.GetValue().Write()

        d_reco_cross.cd()
        for name, h_res in booked_histograms_1d.items():
            if any(x in name for x in ["_2ds_reco_cross_", "_2ds_truth_vtx_z"]):
                h_res.GetValue().Write()

        d_reco_res.cd()
        for name, h_res in booked_histograms_1d.items():
            if any(x in name for x in ["_2ds_delta_z_cross", "_2ds_abs_delta_z_cross", "_2ds_res_z_"]):
                h_res.GetValue().Write()

        d_reco_2d.cd()
        for name, h2_res in booked_histograms_2d.items():
            if "_2ds_" in name:
                h2_res.GetValue().Write()

        # 5. efficiencies_clopper_pearson
        d_eff = out_file.mkdir("efficiencies_clopper_pearson")
        d_eff_ds = d_eff.mkdir("ds_acceptance")
        d_eff_2ds_ovr = d_eff.mkdir("two_ds_tracks_overall")
        d_eff_2ds_acc = d_eff.mkdir("two_ds_tracks_in_ds_acceptance")

        d_eff_ds.cd()
        for name, eff in teff_objects.items():
            if name.startswith("teff_ds_acc_"):
                eff.Write()

        d_eff_2ds_ovr.cd()
        for name, eff in teff_objects.items():
            if name.startswith("teff_2ds_overall_"):
                eff.Write()

        d_eff_2ds_acc.cd()
        for name, eff in teff_objects.items():
            if name.startswith("teff_2ds_in_acc_"):
                eff.Write()

        # 6. acceptance_comparison
        d_acc = out_file.mkdir("acceptance_comparison")
        d_in = d_acc.mkdir("in_acceptance")
        d_out = d_acc.mkdir("outside_acceptance")
        d_acc_prof = d_acc.mkdir("acceptance_profiles_legacy")

        d_in.cd()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_in_acc_") or name.startswith("h_direct_in_acc_") or name.startswith("h_indirect_in_acc_"):
                h_res.GetValue().Write()

        d_out.cd()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_out_acc_") or name.startswith("h_direct_out_acc_") or name.startswith("h_indirect_out_acc_"):
                h_res.GetValue().Write()

        d_acc_prof.cd()
        for name, p_res in booked_profiles.items():
            if "acc_eff" in name:
                p_res.GetValue().Write()

        # 7. species
        d_spec = out_file.mkdir("species")
        d_spec.cd()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_species_"):
                h_res.GetValue().Write()

        # 8. cutflow
        d_cf = out_file.mkdir("cutflow")
        d_cf.cd()
        for name, h_res in booked_histograms_1d.items():
            if name.startswith("h_cutflow_"):
                h_res.GetValue().Write()

        out_file.Close()
        print(f"\nSaved {n_total_plots} objects ({len(booked_histograms_1d)} 1D, {len(booked_histograms_2d)} 2D, {len(booked_profiles)} Profiles, {len(teff_objects)} TEfficiencies) to: {output_file}")

    print("\nTruth analysis and histogram generation completed successfully!")


if __name__ == "__main__":
    main()
