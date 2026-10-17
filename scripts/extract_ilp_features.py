#!/usr/bin/env python3
"""
scripts/extract_ilp_features.py
--------------------------------
Integer Linear Programming (ILP) track reconstruction and feature extraction for SND@LHC Downstream (DS) MuFilter.

Solves track finding as a mathematical combinatorial optimization problem:
  min sum_i c_i x_i
  s.t. sum_{i in S(h)} x_i <= 1  (hit exclusivity: each hit belongs to <= 1 track)
       x_i in {0, 1}

Extracts:
  - Optimal 1-track cost J*(1) and 2-track cost J*(2)
  - Delta J = J*(1) - J*(2) (discriminates single tracks vs dimuons)
  - Reconstructed track parameters (slopes, intercepts, chi2, NDF, planes, timing, QDC)
  - 3D vertex and distance of closest approach (DCA_3D, DCA_vtx, opening angle theta_12)
  - Combined dimuon quality metric chi2_2mu and discriminant score D_2mu
  - True Monte Carlo topology (is_signal, has_charm, charm_pdg, nu_e, is_cc, true_mu1_p, true_mu2_p)

Writes all variables directly to a standalone ROOT TTree named "ilp".
Supports multi-processing across partitions.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple, Set
from itertools import combinations, product
from concurrent.futures import ProcessPoolExecutor, as_completed
import array
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ROOT
ROOT.gROOT.SetBatch(True)

DEFAULT_GEOFILE = (
    "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2mmRangeCut/"
    "sndlhc_15000fb-1_2022_down/nu14/volume_volTarget/1/geofile_full.Genie-TGeant4.root"
)


def parse_partitions(part_arg: Optional[str]) -> Optional[List[int]]:
    """Parse partition specification string (e.g. '1-100', '1,2,5')."""
    if not part_arg or part_arg.strip().lower() == "all":
        return None
    partitions = set()
    for token in part_arg.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start, end = int(parts[0]), int(parts[1])
                for p in range(min(start, end), max(start, end) + 1):
                    partitions.add(p)
        elif token.isdigit():
            partitions.add(int(token))
    return sorted(list(partitions))


def extract_truth_info(tree: ROOT.TTree) -> Dict[str, Any]:
    """Extract Monte Carlo truth quantities from MCTrack."""
    info = {
        "is_signal": 0,
        "has_charm": 0,
        "charm_pdg": 0,
        "nu_pdg": 0,
        "nu_e": 0.0,
        "is_cc": 0,
        "true_mu1_p": -1.0,
        "true_mu2_p": -1.0,
        "true_vtx_x": -999.0,
        "true_vtx_y": -999.0,
        "true_vtx_z": -999.0,
    }
    if not hasattr(tree, "MCTrack") or len(tree.MCTrack) == 0:
        return info

    tracks = tree.MCTrack
    primary_nu = tracks[0]
    info["nu_pdg"] = primary_nu.GetPdgCode()
    info["nu_e"] = float(primary_nu.GetEnergy()) if hasattr(primary_nu, "GetEnergy") else float(primary_nu.GetP())
    info["true_vtx_x"] = float(primary_nu.GetStartX())
    info["true_vtx_y"] = float(primary_nu.GetStartY())
    info["true_vtx_z"] = float(primary_nu.GetStartZ())

    charm_track_id = -1
    charm_pdgs = {411, 421, 431, 4122, 4232, 4132, 4332, 441, 443}

    # Find primary muon (from nu_mu CC)
    for idx, trk in enumerate(tracks):
        pdg = trk.GetPdgCode()
        mother = trk.GetMotherId()
        if mother == 0 and abs(pdg) == 13:
            info["is_cc"] = 1
            info["true_mu1_p"] = float(trk.GetP())
            break

    # Find charm hadron
    for idx, trk in enumerate(tracks):
        pdg = abs(trk.GetPdgCode())
        mother = trk.GetMotherId()
        if (pdg in charm_pdgs or ((pdg // 100) % 10 == 4) or ((pdg // 1000) % 10 == 4)):
            info["has_charm"] = 1
            info["charm_pdg"] = trk.GetPdgCode()
            charm_track_id = idx
            break

    # Find prompt decay muon from charm
    if charm_track_id >= 0:
        for idx, trk in enumerate(tracks):
            mother = trk.GetMotherId()
            if mother == charm_track_id and abs(trk.GetPdgCode()) == 13:
                info["true_mu2_p"] = float(trk.GetP())
                if info["is_cc"] == 1:
                    info["is_signal"] = 1
                break

    return info


def extract_ds_hits(
    tree: ROOT.TTree,
    mufilter_mod: Any,
    vec_a: ROOT.TVector3,
    vec_b: ROOT.TVector3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract Downstream (DS) MuFilter hits grouped into vertical (measuring X)
    and horizontal (measuring Y) bars.
    """
    hits_x: List[Dict[str, Any]] = []
    hits_y: List[Dict[str, Any]] = []

    if not hasattr(tree, "Digi_MuFilterHits"):
        return hits_x, hits_y

    for hit in tree.Digi_MuFilterHits:
        if hasattr(hit, "isValid") and not hit.isValid():
            continue
        if hit.GetSystem() != 3:  # Downstream detector system ID is 3
            continue

        det_id = hit.GetDetectorID()
        plane = hit.GetPlane()  # Stations 0..3
        mufilter_mod.GetPosition(det_id, vec_a, vec_b)

        cx = 0.5 * (vec_a.X() + vec_b.X())
        cy = 0.5 * (vec_a.Y() + vec_b.Y())
        cz = 0.5 * (vec_a.Z() + vec_b.Z())

        # Extract timing
        n_sipms = hit.GetnSiPMs() * max(1, hit.GetnSides())
        times = [hit.GetTime(i) for i in range(n_sipms) if hit.GetTime(i) > 0.0]
        hit_t = float(np.mean(times)) if times else 0.0

        # Extract QDC
        qdcs = [hit.GetSignal(i) for i in range(n_sipms) if hit.GetSignal(i) >= 0.0]
        hit_qdc = float(np.sum(qdcs)) if qdcs else 0.0

        hit_dict = {
            "id": det_id,
            "plane": plane,
            "z": cz,
            "t": hit_t,
            "qdc": hit_qdc,
        }

        if hit.isVertical():
            hit_dict["u"] = cx
            hits_x.append(hit_dict)
        else:
            hit_dict["u"] = cy
            hits_y.append(hit_dict)

    return hits_x, hits_y


def generate_track_candidates(
    hits: List[Dict[str, Any]],
    sigma: float = 0.2887,
    lambda_hit: float = 10.0,
    max_slope: float = 0.35,
    max_chi2_red: float = 20.0,
) -> List[Dict[str, Any]]:
    """
    Generate all valid 2D track candidates crossing >= 3 distinct DS stations.
    Calculates cost c_i = chi2_i - lambda_hit * N_hits.
    """
    planes_dict: Dict[int, List[Dict[str, Any]]] = {}
    for h in hits:
        planes_dict.setdefault(h["plane"], []).append(h)

    available_planes = sorted(planes_dict.keys())
    if len(available_planes) < 3:
        return []

    candidates: List[Dict[str, Any]] = []

    # Combos spanning 4 stations, then 3 stations
    for k in [4, 3]:
        for p_comb in combinations(available_planes, k):
            hit_pools = [planes_dict[p] for p in p_comb]
            for combo in product(*hit_pools):
                zs = np.array([h["z"] for h in combo])
                us = np.array([h["u"] for h in combo])

                z_mean, u_mean = float(np.mean(zs)), float(np.mean(us))
                var_z = float(np.sum((zs - z_mean) ** 2))
                if var_z < 1e-5:
                    continue

                slope = float(np.sum((zs - z_mean) * (us - u_mean)) / var_z)
                if abs(slope) > max_slope:
                    continue

                intercept = float(u_mean - slope * z_mean)
                residuals = us - (slope * zs + intercept)
                chi2 = float(np.sum((residuals / sigma) ** 2))
                ndf = len(combo) - 2
                chi2_red = chi2 / max(1, ndf)

                if chi2_red > max_chi2_red:
                    continue

                cost = chi2 - lambda_hit * len(combo)
                candidates.append({
                    "hit_ids": [h["id"] for h in combo],
                    "n_hits": len(combo),
                    "n_planes": len(set(h["plane"] for h in combo)),
                    "slope": slope,
                    "intercept": intercept,
                    "chi2": chi2,
                    "ndf": ndf,
                    "chi2_red": chi2_red,
                    "cost": cost,
                    "t_mean": float(np.mean([h["t"] for h in combo])),
                    "qdc_total": float(np.sum([h["qdc"] for h in combo])),
                })

    return candidates


def solve_ilp_tracks(
    candidates: List[Dict[str, Any]],
    n_tracks_req: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Solve ILP track finding via scipy.optimize.milp with hit exclusivity constraints.
    Returns: (list_of_selected_candidates, optimal_objective_cost)
    """
    if not candidates:
        return [], 0.0

    M = len(candidates)
    c = np.array([cand["cost"] for cand in candidates])

    # Build unique hit ID mapping
    all_hit_ids = list({h_id for cand in candidates for h_id in cand["hit_ids"]})
    hit_to_idx = {h_id: i for i, h_id in enumerate(all_hit_ids)}

    # Hit exclusivity constraint: sum_{track contains hit} x_i <= 1
    A_hit = np.zeros((len(all_hit_ids), M))
    for j, cand in enumerate(candidates):
        for h_id in cand["hit_ids"]:
            A_hit[hit_to_idx[h_id], j] = 1.0

    constraints = [LinearConstraint(A_hit, -np.inf, 1.0)]

    if n_tracks_req is not None:
        if M < n_tracks_req:
            return [], 0.0
        A_sum = np.ones((1, M))
        constraints.append(LinearConstraint(A_sum, n_tracks_req, n_tracks_req))

    integrality = np.ones(M)
    bounds = Bounds(0, 1)

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    if res.success and res.x is not None:
        selected_indices = [i for i, val in enumerate(res.x) if val > 0.5]
        selected_cands = [candidates[i] for i in selected_indices]
        return selected_cands, float(res.fun)

    return [], 0.0


def calculate_3d_line_dca(
    trk1_slopes: Tuple[float, float],
    trk1_intercepts: Tuple[float, float],
    trk2_slopes: Tuple[float, float],
    trk2_intercepts: Tuple[float, float],
) -> Tuple[float, np.ndarray, float]:
    """
    Calculates analytical 3D Distance of Closest Approach (DCA), common midpoint vertex,
    and 3D opening angle between two spatial lines:
      Line 1: r1(z) = (ax1*z + bx1, ay1*z + by1, z)
      Line 2: r2(z) = (ax2*z + bx2, ay2*z + by2, z)
    """
    ax1, ay1 = trk1_slopes
    bx1, by1 = trk1_intercepts
    ax2, ay2 = trk2_slopes
    bx2, by2 = trk2_intercepts

    d1 = np.array([ax1, ay1, 1.0])
    d2 = np.array([ax2, ay2, 1.0])
    p1 = np.array([bx1, by1, 0.0])
    p2 = np.array([bx2, by2, 0.0])

    norm_d1 = np.linalg.norm(d1)
    norm_d2 = np.linalg.norm(d2)
    cos_theta = np.dot(d1, d2) / (norm_d1 * norm_d2)
    opening_angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    w0 = p1 - p2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))

    denom = a * c - b * b
    if denom > 1e-7:
        s1 = (b * e - c * d) / denom
        s2 = (a * e - b * d) / denom
        pc1 = p1 + s1 * d1
        pc2 = p2 + s2 * d2
        dca_3d = float(np.linalg.norm(pc1 - pc2))
        vtx_common = 0.5 * (pc1 + pc2)
    else:
        # Parallel lines
        cross_prod = np.cross(d1, w0)
        dca_3d = float(np.linalg.norm(cross_prod) / np.sqrt(a))
        vtx_common = p1

    return dca_3d, vtx_common, float(opening_angle_rad)


def create_ilp_tree_buffers(tree: ROOT.TTree) -> Dict[str, array.array]:
    """Create all branch buffers and attach them to the 'ilp' TTree."""
    int_branches = [
        "partition", "entry", "event_number",
        "is_signal", "has_charm", "charm_pdg", "nu_pdg", "is_cc",
        "n_ds_hits", "n_ds_hits_x", "n_ds_hits_y",
        "n_tracks_hough",
        "n_candidates_x", "n_candidates_y",
        "n_tracks_ilp_x", "n_tracks_ilp_y", "n_tracks_ilp_3d",
        "trk1_n_hits_x", "trk1_n_hits_y", "trk1_n_planes",
        "trk2_n_hits_x", "trk2_n_hits_y", "trk2_n_planes",
    ]

    float_branches = [
        "nu_e", "true_mu1_p", "true_mu2_p",
        "true_vtx_x", "true_vtx_y", "true_vtx_z",
        "j_opt_1track_x", "j_opt_2track_x", "delta_j_x",
        "j_opt_1track_y", "j_opt_2track_y", "delta_j_y",
        "delta_j_total",
        "trk1_chi2_x", "trk1_chi2_y", "trk1_chi2_total", "trk1_reduced_chi2",
        "trk1_slope_xz", "trk1_intercept_xz",
        "trk1_slope_yz", "trk1_intercept_yz",
        "trk1_t_mean", "trk1_qdc_total",
        "trk2_chi2_x", "trk2_chi2_y", "trk2_chi2_total", "trk2_reduced_chi2",
        "trk2_slope_xz", "trk2_intercept_xz",
        "trk2_slope_yz", "trk2_intercept_yz",
        "trk2_t_mean", "trk2_qdc_total",
        "delta_theta_xz_mrad", "delta_theta_yz_mrad",
        "opening_angle_3d_mrad", "dca_3d",
        "vtx_reco_x", "vtx_reco_y", "vtx_reco_z", "dca_vtx",
        "delta_t_ns", "chi2_2mu", "dimuon_discriminant",
    ]

    buffers: Dict[str, array.array] = {}
    for name in int_branches:
        buf = array.array("i", [0])
        tree.Branch(name, buf, f"{name}/I")
        buffers[name] = buf

    for name in float_branches:
        buf = array.array("f", [-999.0])
        tree.Branch(name, buf, f"{name}/F")
        buffers[name] = buf

    return buffers


def reset_buffers(buffers: Dict[str, array.array]) -> None:
    """Reset branch buffers to default sentinel values."""
    for name, buf in buffers.items():
        if buf.typecode == "i":
            buf[0] = -1 if "n_tracks" in name or "charm_pdg" in name else 0
        else:
            buf[0] = -999.0


def process_partition_file(
    input_path: str,
    partition_id: int,
    geofile_path: str,
    temp_output_path: str,
    max_events: int = -1,
) -> Dict[str, Any]:
    """Worker task: processes a single partition ROOT file and writes to temp_output_path."""
    import SndlhcGeo

    res = {
        "partition": partition_id,
        "input_path": input_path,
        "processed_entries": 0,
        "temp_root": temp_output_path,
        "status": "OK",
        "error": None,
    }

    if not os.path.exists(input_path):
        res["status"] = "SKIPPED_NOT_FOUND"
        return res

    fin = ROOT.TFile.Open(input_path, "READ")
    if not fin or fin.IsZombie():
        res["status"] = "ERROR_OPEN"
        res["error"] = f"Failed to open {input_path}"
        return res

    tree = fin.Get("cbmsim")
    if not tree:
        for alt in ["rawConv", "events"]:
            tree = fin.Get(alt)
            if tree:
                break

    if not tree:
        fin.Close()
        res["status"] = "ERROR_NO_TREE"
        res["error"] = f"No TTree found in {input_path}"
        return res

    total_entries = tree.GetEntries()
    if total_entries == 0:
        fin.Close()
        res["status"] = "EMPTY"
        return res

    # Initialize geometry
    geo = SndlhcGeo.GeoInterface(geofile_path)
    mufilter_mod = geo.modules.get("MuFilter")
    vec_a = ROOT.TVector3()
    vec_b = ROOT.TVector3()

    # Create temporary output ROOT file and TTree
    fout = ROOT.TFile.Open(temp_output_path, "RECREATE")
    tree_out = ROOT.TTree("ilp", "Integer Linear Programming Track Reconstruction Features")
    buffers = create_ilp_tree_buffers(tree_out)

    n_process = min(total_entries, max_events) if max_events > 0 else total_entries

    for iev in range(n_process):
        tree.GetEntry(iev)
        reset_buffers(buffers)

        buffers["partition"][0] = int(partition_id)
        buffers["entry"][0] = int(iev)
        event_num = getattr(tree.EventHeader, "GetEventNumber")() if hasattr(tree, "EventHeader") else iev
        buffers["event_number"][0] = int(event_num)

        # 1. MC Truth extraction
        truth_info = extract_truth_info(tree)
        for k, v in truth_info.items():
            if k in buffers:
                if buffers[k].typecode == "i":
                    buffers[k][0] = int(v)
                else:
                    buffers[k][0] = float(v)

        # 2. Existing Hough track count
        if hasattr(tree, "Reco_MuonTracks"):
            buffers["n_tracks_hough"][0] = len(tree.Reco_MuonTracks)
        else:
            buffers["n_tracks_hough"][0] = -1

        # 3. Extract DS hits
        hits_x, hits_y = extract_ds_hits(tree, mufilter_mod, vec_a, vec_b)
        buffers["n_ds_hits"][0] = len(hits_x) + len(hits_y)
        buffers["n_ds_hits_x"][0] = len(hits_x)
        buffers["n_ds_hits_y"][0] = len(hits_y)

        # 4. Generate candidate tracks in X-Z and Y-Z
        cands_x = generate_track_candidates(hits_x)
        cands_y = generate_track_candidates(hits_y)
        buffers["n_candidates_x"][0] = len(cands_x)
        buffers["n_candidates_y"][0] = len(cands_y)

        # 5. Solve ILP for 1-track and 2-track hypotheses in X-Z
        t1_cands_x, j1_x = solve_ilp_tracks(cands_x, n_tracks_req=1)
        t2_cands_x, j2_x = solve_ilp_tracks(cands_x, n_tracks_req=2)
        buffers["j_opt_1track_x"][0] = float(j1_x)
        buffers["j_opt_2track_x"][0] = float(j2_x)
        delta_j_x = (j1_x - j2_x) if (len(t1_cands_x) > 0 and len(t2_cands_x) > 0) else 0.0
        buffers["delta_j_x"][0] = float(delta_j_x)

        # Unconstrained best tracks in X-Z
        best_cands_x, _ = solve_ilp_tracks(cands_x, n_tracks_req=None)
        buffers["n_tracks_ilp_x"][0] = len(best_cands_x)

        # 6. Solve ILP for 1-track and 2-track hypotheses in Y-Z
        t1_cands_y, j1_y = solve_ilp_tracks(cands_y, n_tracks_req=1)
        t2_cands_y, j2_y = solve_ilp_tracks(cands_y, n_tracks_req=2)
        buffers["j_opt_1track_y"][0] = float(j1_y)
        buffers["j_opt_2track_y"][0] = float(j2_y)
        delta_j_y = (j1_y - j2_y) if (len(t1_cands_y) > 0 and len(t2_cands_y) > 0) else 0.0
        buffers["delta_j_y"][0] = float(delta_j_y)

        # Unconstrained best tracks in Y-Z
        best_cands_y, _ = solve_ilp_tracks(cands_y, n_tracks_req=None)
        buffers["n_tracks_ilp_y"][0] = len(best_cands_y)

        # Total Delta J
        delta_j_total = delta_j_x + delta_j_y
        buffers["delta_j_total"][0] = float(delta_j_total)

        # 7. Form 3D Tracks & Calculate Dimuon Pair Observables
        # Check if we have 2 tracks in both projections
        if len(t2_cands_x) == 2 and len(t2_cands_y) == 2:
            buffers["n_tracks_ilp_3d"][0] = 2
            x1, x2 = t2_cands_x[0], t2_cands_x[1]
            y1, y2 = t2_cands_y[0], t2_cands_y[1]

            # Solve 2x2 pairing ambiguity by minimizing timing difference
            dt_pair_a = abs(x1["t_mean"] - y1["t_mean"]) + abs(x2["t_mean"] - y2["t_mean"])
            dt_pair_b = abs(x1["t_mean"] - y2["t_mean"]) + abs(x2["t_mean"] - y1["t_mean"])

            if dt_pair_a <= dt_pair_b:
                trk1_x, trk1_y = x1, y1
                trk2_x, trk2_y = x2, y2
            else:
                trk1_x, trk1_y = x1, y2
                trk2_x, trk2_y = x2, y1

            # Populate Track 1
            buffers["trk1_chi2_x"][0] = float(trk1_x["chi2"])
            buffers["trk1_chi2_y"][0] = float(trk1_y["chi2"])
            trk1_chi2 = float(trk1_x["chi2"] + trk1_y["chi2"])
            trk1_ndf = trk1_x["ndf"] + trk1_y["ndf"]
            buffers["trk1_chi2_total"][0] = trk1_chi2
            buffers["trk1_reduced_chi2"][0] = trk1_chi2 / max(1, trk1_ndf)
            buffers["trk1_n_hits_x"][0] = int(trk1_x["n_hits"])
            buffers["trk1_n_hits_y"][0] = int(trk1_y["n_hits"])
            buffers["trk1_n_planes"][0] = int(max(trk1_x["n_planes"], trk1_y["n_planes"]))
            buffers["trk1_slope_xz"][0] = float(trk1_x["slope"])
            buffers["trk1_intercept_xz"][0] = float(trk1_x["intercept"])
            buffers["trk1_slope_yz"][0] = float(trk1_y["slope"])
            buffers["trk1_intercept_yz"][0] = float(trk1_y["intercept"])
            t1_mean = 0.5 * (trk1_x["t_mean"] + trk1_y["t_mean"])
            buffers["trk1_t_mean"][0] = float(t1_mean)
            buffers["trk1_qdc_total"][0] = float(trk1_x["qdc_total"] + trk1_y["qdc_total"])

            # Populate Track 2
            buffers["trk2_chi2_x"][0] = float(trk2_x["chi2"])
            buffers["trk2_chi2_y"][0] = float(trk2_y["chi2"])
            trk2_chi2 = float(trk2_x["chi2"] + trk2_y["chi2"])
            trk2_ndf = trk2_x["ndf"] + trk2_y["ndf"]
            buffers["trk2_chi2_total"][0] = trk2_chi2
            buffers["trk2_reduced_chi2"][0] = trk2_chi2 / max(1, trk2_ndf)
            buffers["trk2_n_hits_x"][0] = int(trk2_x["n_hits"])
            buffers["trk2_n_hits_y"][0] = int(trk2_y["n_hits"])
            buffers["trk2_n_planes"][0] = int(max(trk2_x["n_planes"], trk2_y["n_planes"]))
            buffers["trk2_slope_xz"][0] = float(trk2_x["slope"])
            buffers["trk2_intercept_xz"][0] = float(trk2_x["intercept"])
            buffers["trk2_slope_yz"][0] = float(trk2_y["slope"])
            buffers["trk2_intercept_yz"][0] = float(trk2_y["intercept"])
            t2_mean = 0.5 * (trk2_x["t_mean"] + trk2_y["t_mean"])
            buffers["trk2_t_mean"][0] = float(t2_mean)
            buffers["trk2_qdc_total"][0] = float(trk2_x["qdc_total"] + trk2_y["qdc_total"])

            # Kinematic angular differences
            dtheta_xz = abs(trk1_x["slope"] - trk2_x["slope"]) * 1000.0
            dtheta_yz = abs(trk1_y["slope"] - trk2_y["slope"]) * 1000.0
            buffers["delta_theta_xz_mrad"][0] = float(dtheta_xz)
            buffers["delta_theta_yz_mrad"][0] = float(dtheta_yz)

            # 3D Line DCA & Common Vertex
            dca_3d, vtx_common, op_angle = calculate_3d_line_dca(
                (trk1_x["slope"], trk1_y["slope"]),
                (trk1_x["intercept"], trk1_y["intercept"]),
                (trk2_x["slope"], trk2_y["slope"]),
                (trk2_x["intercept"], trk2_y["intercept"]),
            )
            buffers["opening_angle_3d_mrad"][0] = float(op_angle * 1000.0)
            buffers["dca_3d"][0] = float(dca_3d)
            buffers["vtx_reco_x"][0] = float(vtx_common[0])
            buffers["vtx_reco_y"][0] = float(vtx_common[1])
            buffers["vtx_reco_z"][0] = float(vtx_common[2])

            # Distance to true interaction vertex
            true_vtx = np.array([truth_info["true_vtx_x"], truth_info["true_vtx_y"], truth_info["true_vtx_z"]])
            if true_vtx[2] > -900:
                buffers["dca_vtx"][0] = float(np.linalg.norm(vtx_common - true_vtx))

            # Timing delta
            delta_t = abs(t1_mean - t2_mean)
            buffers["delta_t_ns"][0] = float(delta_t)

            # Combined Dimuon Quality chi2_2mu
            ndf_total = trk1_ndf + trk2_ndf + 2
            dca_term = (dca_3d / 1.5) ** 2
            dt_term = (delta_t / 1.0) ** 2
            chi2_2mu = (trk1_chi2 + trk2_chi2 + dca_term + dt_term) / max(1, ndf_total)
            buffers["chi2_2mu"][0] = float(chi2_2mu)

            # Composite Dimuon Discriminant D_2mu
            disc = (delta_j_total / 10.0) - np.log(1.0 + max(0.0, chi2_2mu)) - (dca_3d / 2.0)
            buffers["dimuon_discriminant"][0] = float(disc)

        elif len(t1_cands_x) >= 1 and len(t1_cands_y) >= 1:
            # Single reconstructed track
            buffers["n_tracks_ilp_3d"][0] = 1
            trk1_x = t1_cands_x[0]
            trk1_y = t1_cands_y[0]

            buffers["trk1_chi2_x"][0] = float(trk1_x["chi2"])
            buffers["trk1_chi2_y"][0] = float(trk1_y["chi2"])
            trk1_chi2 = float(trk1_x["chi2"] + trk1_y["chi2"])
            trk1_ndf = trk1_x["ndf"] + trk1_y["ndf"]
            buffers["trk1_chi2_total"][0] = trk1_chi2
            buffers["trk1_reduced_chi2"][0] = trk1_chi2 / max(1, trk1_ndf)
            buffers["trk1_n_hits_x"][0] = int(trk1_x["n_hits"])
            buffers["trk1_n_hits_y"][0] = int(trk1_y["n_hits"])
            buffers["trk1_n_planes"][0] = int(max(trk1_x["n_planes"], trk1_y["n_planes"]))
            buffers["trk1_slope_xz"][0] = float(trk1_x["slope"])
            buffers["trk1_intercept_xz"][0] = float(trk1_x["intercept"])
            buffers["trk1_slope_yz"][0] = float(trk1_y["slope"])
            buffers["trk1_intercept_yz"][0] = float(trk1_y["intercept"])
            buffers["trk1_t_mean"][0] = float(0.5 * (trk1_x["t_mean"] + trk1_y["t_mean"]))
            buffers["trk1_qdc_total"][0] = float(trk1_x["qdc_total"] + trk1_y["qdc_total"])

        else:
            buffers["n_tracks_ilp_3d"][0] = 0

        tree_out.Fill()

    fout.Write()
    fout.Close()
    fin.Close()

    res["processed_entries"] = n_process
    return res


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Integer Linear Programming (ILP) track reconstruction & feature extraction for SND@LHC."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input ROOT path pattern (e.g. '.../%s/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root')",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/plots/ilp_dimuon_features.root",
        help="Output ROOT file containing the 'ilp' TTree (default: /eos/user/i/idioniso/snd-numu-charm/plots/ilp_dimuon_features.root)",
    )
    parser.add_argument(
        "-g", "--geofile",
        type=str,
        default=DEFAULT_GEOFILE,
        help="Detector geometry ROOT file",
    )
    parser.add_argument(
        "-p", "--partitions",
        type=str,
        default=None,
        help="Partition specification string (e.g. '1-100', '1,2,5')",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=12,
        help="Number of parallel worker processes (default: 12)",
    )
    parser.add_argument(
        "-n", "--max-events",
        type=int,
        default=-1,
        help="Max events to process per file (-1 for all)",
    )

    args = parser.parse_args()

    # Determine partition list
    partitions = parse_partitions(args.partitions)
    if partitions is None:
        if "%s" in args.input or "%d" in args.input:
            partitions = list(range(1, 1001))
        else:
            partitions = [0]

    # Verify input files
    tasks = []
    temp_dir = tempfile.mkdtemp(prefix="snd_ilp_features_")

    for p in partitions:
        fpath = args.input % p if ("%s" in args.input or "%d" in args.input) else args.input
        if os.path.exists(fpath):
            temp_out = os.path.join(temp_dir, f"ilp_part_{p}.root")
            tasks.append((fpath, p, args.geofile, temp_out, args.max_events))

    print("=" * 80)
    print(" SND@LHC Integer Linear Programming (ILP) Feature Extractor")
    print("=" * 80)
    print(f" Input Pattern:        {args.input}")
    print(f" Output ROOT File:     {args.output}")
    print(f" Geometry File:        {args.geofile}")
    print(f" Partitions Matched:   {len(tasks)}")
    print(f" Worker Processes:     {args.workers}")
    print(f" Max Events/File:      {args.max_events if args.max_events > 0 else 'All'}")
    print(f" Output TTree Name:    ilp")
    print("=" * 80)

    if not tasks:
        print("[Error] No input files found matching criteria.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 1

    t0 = time.time()
    worker_results = []
    temp_root_files = []

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_partition_file, *t): t[1]
                for t in tasks
            }

            for idx, fut in enumerate(as_completed(futures), 1):
                part_id = futures[fut]
                res = fut.result()
                worker_results.append(res)
                if res["status"] == "OK" and os.path.exists(res["temp_root"]):
                    temp_root_files.append(res["temp_root"])

                elapsed = time.time() - t0
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - idx) / rate if rate > 0 else 0
                print(
                    f"[{idx:4d}/{len(tasks)}] Part {part_id:4d}: "
                    f"Entries: {res['processed_entries']:3d} | Status: {res['status']} "
                    f"({elapsed:.1f}s, ETA: {int(eta//60):02d}:{int(eta%60):02d})"
                )

        # Merge worker ROOT files into final output file
        if temp_root_files:
            print(f"\n[Merge] Merging {len(temp_root_files)} partition files into {args.output}...")
            out_dir = os.path.dirname(os.path.abspath(args.output))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            merger = ROOT.TFileMerger(False)
            merger.OutputFile(args.output, "RECREATE")
            for tf in temp_root_files:
                merger.AddFile(tf)

            if merger.Merge():
                print(f"[Done] Successfully created {args.output}")
            else:
                print(f"[Warning] TFileMerger failed. Falling back to hadd...")
                cmd = f"hadd -f -k {args.output} " + " ".join(temp_root_files)
                os.system(cmd)
        else:
            print("[Warning] No partition outputs produced.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    total_time = time.time() - t0
    total_events = sum(r["processed_entries"] for r in worker_results)
    print("=" * 80)
    print(f" Finished processing {len(tasks)} partitions ({total_events} events) in {total_time:.1f}s")
    print(f" Output saved to: {args.output}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
