#!/usr/bin/env python3
"""
scripts/qdc_calib/validate_ti18_thresholds_methods1234.py
---------------------------------------------------------
Validates MuFilter detection thresholds using TI18 Collision Data (Run 5408)
against PMU Simulations across 4 threshold models:
  1. Data (Run 5408)
  2. MC Raw (E_th = 0)
  3. MC Theoretical (k = 1.0)
  4. MC Calibrated (k_SiPM from config/sipm_photoelectron_reduction_factors.csv)
  5. MC Simple Step (1.0 MeV for DS, 1.5 MeV for US/Veto)

Implements 4 Complementary In-Situ Validation Methods:
  - Method 1: Tag-and-probe detection efficiency vs. impact position along the bar
              (SciFi tracks for Veto; DS tracks for US and DS bars to capture full acceptance).
  - Method 2: Detection efficiency vs. track angle theta (E_dep ~ MIP / cos(theta)).
  - Method 3: Low-end QDC spectrum cutoff / turn-on threshold matching.
  - Method 4: Muon-induced shower lateral profile in DS bars only (hit probability vs. r_perp in 1 cm bins).

Geometry constraints:
  - All bars are 1.0 cm thick along beam (z).
  - DS bars are 1.0 cm wide.
  - US and Veto bars are 6.0 cm wide.

Stores all canvases and histograms in a single ROOT file with structured TDirectories:
  output/ti18_threshold_validation_methods1234.root
"""

from __future__ import annotations

import os
import sys
import math
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import scripts.qdc_calib.min_pe as mpe
import SndlhcGeo
import SndlhcTracking

# Physical Constants & Luminosity scaling for Run 5408 (1.53 pb^-1)
LUMI_SCALE_MC: float = 0.406569
QDC_TO_PE: float = 3.60 / 0.43258  # 8.322 pe / QDC count


def build_geometry_maps(geofile: str):
    """
    Extracts 3D coordinates and bounding boxes for all MuFilter bars from SndlhcGeo.
    Returns:
      bars: Dict[det_id, Dict[str, Any]]
    """
    geo = SndlhcGeo.GeoInterface(geofile)
    mufi = geo.modules["MuFilter"]
    A, B = ROOT.TVector3(), ROOT.TVector3()
    bars = {}

    # 1. Veto (subsystem 1, planes 0..1, bars 0..6)
    for pl in range(2):
        for b in range(7):
            did = 10000 + pl * 1000 + b
            mufi.GetPosition(did, A, B)
            bars[did] = {
                "sub": 1, "plane": pl, "bar": b,
                "x_min": min(A.X(), B.X()), "x_max": max(A.X(), B.X()),
                "y_min": min(A.Y(), B.Y()), "y_max": max(A.Y(), B.Y()),
                "z": 0.5 * (A.Z() + B.Z()),
                "width": 6.0, "is_vert": True,
            }

    # 2. Upstream (subsystem 2, planes 0..4, bars 0..9)
    for pl in range(5):
        for b in range(10):
            did = 20000 + pl * 1000 + b
            mufi.GetPosition(did, A, B)
            bars[did] = {
                "sub": 2, "plane": pl, "bar": b,
                "x_min": min(A.X(), B.X()), "x_max": max(A.X(), B.X()),
                "y_min": min(A.Y(), B.Y()), "y_max": max(A.Y(), B.Y()),
                "z": 0.5 * (A.Z() + B.Z()),
                "width": 6.0, "is_vert": False,
            }

    # 3. Downstream (subsystem 3, planes 0..3)
    for pl in range(4):
        b_start = 60 if pl == 3 else 0
        for b in range(b_start, 120):
            did = 30000 + pl * 1000 + b
            mufi.GetPosition(did, A, B)
            bars[did] = {
                "sub": 3, "plane": pl, "bar": b,
                "x_min": min(A.X(), B.X()), "x_max": max(A.X(), B.X()),
                "y_min": min(A.Y(), B.Y()), "y_max": max(A.Y(), B.Y()),
                "z": 0.5 * (A.Z() + B.Z()),
                "width": 1.0, "is_vert": (b >= 60),
            }

    return bars


def precompute_ti18_thresholds():
    """Precomputes theoretical and calibrated thresholds in keV for all TI18 bars."""
    calc_theo = mpe.MinPE(year=2022, is_testbeam=False)
    calc_theo._k_factors = {}  # all k = 1.0

    calc_calib = mpe.MinPE(year=2022, is_testbeam=False)

    th_theo: Dict[int, float] = {}
    th_calib: Dict[int, float] = {}
    th_step: Dict[int, float] = {}

    # Veto
    for pl in range(2):
        for b in range(7):
            did = 10000 + pl * 1000 + b
            p = {"detector_id": did, "x": -37.7, "y": 0.0, "z": 0.0, "eloss": 1.0}
            th_theo[did] = float(calc_theo.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_calib[did] = float(calc_calib.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_step[did] = 1500.0

    # US
    for pl in range(5):
        for b in range(10):
            did = 20000 + pl * 1000 + b
            p = {"detector_id": did, "x": -37.7, "y": 0.0, "z": 0.0, "eloss": 1.0}
            th_theo[did] = float(calc_theo.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_calib[did] = float(calc_calib.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_step[did] = 1500.0

    # DS
    for pl in range(4):
        b_start = 60 if pl == 3 else 0
        for b in range(b_start, 120):
            did = 30000 + pl * 1000 + b
            is_vert = (b >= 60)
            p = {"detector_id": did, "x": 0.0 if is_vert else -37.7, "y": 20.0 if is_vert else 0.0, "z": 0.0, "eloss": 1.0}
            th_theo[did] = float(calc_theo.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_calib[did] = float(calc_calib.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_step[did] = 1000.0

    return th_theo, th_calib, th_step


def run_validation(
    data_file: str,
    mc_file: str,
    geofile: str,
    out_root: str = "output/ti18_threshold_validation_methods1234.root",
    max_data_events: int = 35000,
    max_mc_events: int = 80000,
    target_muons: int = 1500,
):
    os.makedirs(os.path.dirname(out_root) or ".", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    print("[Init] Loading detector geometry...")
    bars_map = build_geometry_maps(geofile)
    print(f"[Init] Mapped {len(bars_map)} MuFilter bars.")

    print("[Init] Precomputing physical threshold maps...")
    th_theo, th_calib, th_step = precompute_ti18_thresholds()

    # Data structures for results
    # Method 1: Efficiency vs position along bar (20 bins)
    # Coordinate: US (0..83 cm), DS-H (0..82.5 cm), DS-V (0..63.5 cm), Veto (0..42 cm)
    m1_bins = {
        "us": np.linspace(0.0, 83.0, 21),
        "dsh": np.linspace(0.0, 82.5, 21),
        "dsv": np.linspace(0.0, 63.5, 21),
        "veto": np.linspace(0.0, 42.0, 15),
    }

    # Method 2: Efficiency vs track angle theta (0..10 deg, 10 bins)
    m2_bins = np.linspace(0.0, 10.0, 11)

    # Method 3: QDC spectrum (0..40 QDC counts, 40 bins)
    m3_bins = np.linspace(0.0, 40.0, 41)

    # Method 4: DS Shower lateral profile (0..25 cm from track, 25 bins of 1 cm)
    m4_bins = np.linspace(0.0, 25.0, 26)

    models = ["data", "raw", "theo", "calib", "step"]

    # Storage dicts: {sample: {subsystem/var: np.ndarray / counts}}
    res_m1_exp = {m: {k: np.zeros(len(m1_bins[k]) - 1) for k in m1_bins} for m in models}
    res_m1_det = {m: {k: np.zeros(len(m1_bins[k]) - 1) for k in m1_bins} for m in models}

    res_m2_exp = {m: {k: np.zeros(len(m2_bins) - 1) for k in ["us", "ds", "veto"]} for m in models}
    res_m2_det = {m: {k: np.zeros(len(m2_bins) - 1) for k in ["us", "ds", "veto"]} for m in models}

    res_m3_qdc = {m: {k: [] for k in ["us", "ds", "veto"]} for m in models}

    res_m4_tested = {m: np.zeros(len(m4_bins) - 1) for m in models}
    res_m4_fired = {m: np.zeros(len(m4_bins) - 1) for m in models}

    # =========================================================================
    # Process 1: Real Collision Data (Run 5408)
    # =========================================================================
    print(f"\n[Processing Data] Run 5408 from {data_file}...")
    t_data = ROOT.TChain("rawConv")
    t_data.Add(data_file)
    n_entries_d = min(max_data_events, t_data.GetEntries())
    ioman_d = ROOT.FairRootManager.Instance()
    ioman_d.SetInChain(t_data)

    trackTask = SndlhcTracking.Tracking()
    trackTask.SetName("simpleTracking")
    trackTask.SetTrackClassType(False)
    trackTask.Init()

    passed_d = 0
    prev_time = None

    for iev in range(n_entries_d):
        if (iev + 1) % 5000 == 0:
            print(f"  Data entry {iev+1:,}/{n_entries_d:,} (Muons analyzed: {passed_d})...", flush=True)
        if passed_d >= target_muons:
            break

        t_data.GetEvent(iev)
        h = t_data.EventHeader
        if not h.isIP1() or h.GetBeamMode() != 11:
            continue
        cur_t = h.GetEventTime()
        if prev_time is not None and abs(cur_t - prev_time) <= 100:
            prev_time = cur_t
            continue
        prev_time = cur_t

        # Hits in data
        fired_d: Dict[int, float] = {}
        for hit in t_data.Digi_MuFilterHits:
            if not hit.isValid():
                continue
            did = hit.GetDetectorID()
            # Extract signal
            sig = -1.0
            try:
                s_map = hit.SumOfSignals()
                if "Sum" in s_map and float(s_map["Sum"]) > 0:
                    sig = float(s_map["Sum"])
            except Exception:
                pass
            if sig < 0:
                for s in range(hit.GetnSides()):
                    v = float(hit.GetSignal(s))
                    if v > 0:
                        sig = max(sig, v)
            if sig > 0:
                fired_d[did] = sig

        # Reconstruct Tracks
        trackTask.fittedTracks.Delete()
        trackTask.ExecuteTask(option="ScifiDS")

        scifi_trks, ds_trks = [], []
        for trk in trackTask.fittedTracks:
            if trk.getChi2Ndf() <= 5.0 and abs(trk.getSlopeXZ()) <= 0.1 and abs(trk.getSlopeYZ()) <= 0.1:
                if trk.getTrackType() == 1:
                    scifi_trks.append(trk)
                elif trk.getTrackType() == 3:
                    ds_trks.append(trk)

        if not scifi_trks and not ds_trks:
            continue

        passed_d += 1

        # --- Method 1 & 2: Veto (probed by SciFi tracks) ---
        if scifi_trks:
            st = scifi_trks[0]
            theta_deg = math.degrees(math.atan(math.hypot(st.getSlopeXZ(), st.getSlopeYZ())))
            th_bin = int(np.digitize(theta_deg, m2_bins) - 1)

            for pl in range(2):
                for b in range(7):
                    did = 10000 + pl * 1000 + b
                    b_info = bars_map[did]
                    pt = st.getPointAtZ(b_info["z"])
                    x_c = 0.5 * (b_info["x_min"] + b_info["x_max"])
                    if abs(pt.X() - x_c) <= 2.5 and (b_info["y_min"] + 1.0 <= pt.Y() <= b_info["y_max"] - 1.0):
                        y_rel = pt.Y() - b_info["y_min"]
                        p_bin = int(np.digitize(y_rel, m1_bins["veto"]) - 1)
                        if 0 <= p_bin < len(m1_bins["veto"]) - 1:
                            res_m1_exp["data"]["veto"][p_bin] += 1
                            if did in fired_d:
                                res_m1_det["data"]["veto"][p_bin] += 1
                        if 0 <= th_bin < len(m2_bins) - 1:
                            res_m2_exp["data"]["veto"][th_bin] += 1
                            if did in fired_d:
                                res_m2_det["data"]["veto"][th_bin] += 1

        # --- Method 1 & 2: US and DS (probed by DS tracks without SciFi requirement) ---
        if ds_trks:
            dt = ds_trks[0]
            theta_deg = math.degrees(math.atan(math.hypot(dt.getSlopeXZ(), dt.getSlopeYZ())))
            th_bin = int(np.digitize(theta_deg, m2_bins) - 1)

            # US Bars (Planes 0..4, horizontal)
            for pl in range(5):
                for b in range(10):
                    did = 20000 + pl * 1000 + b
                    b_info = bars_map[did]
                    pt = dt.getPointAtZ(b_info["z"])
                    y_c = 0.5 * (b_info["y_min"] + b_info["y_max"])
                    if abs(pt.Y() - y_c) <= 2.5 and (b_info["x_min"] + 1.0 <= pt.X() <= b_info["x_max"] - 1.0):
                        x_rel = pt.X() - b_info["x_min"]
                        p_bin = int(np.digitize(x_rel, m1_bins["us"]) - 1)
                        if 0 <= p_bin < len(m1_bins["us"]) - 1:
                            res_m1_exp["data"]["us"][p_bin] += 1
                            if did in fired_d:
                                res_m1_det["data"]["us"][p_bin] += 1
                        if 0 <= th_bin < len(m2_bins) - 1:
                            res_m2_exp["data"]["us"][th_bin] += 1
                            if did in fired_d:
                                res_m2_det["data"]["us"][th_bin] += 1

            # DS Bars (Planes 0..3)
            for pl in range(4):
                b_start = 60 if pl == 3 else 0
                for b in range(b_start, 120):
                    did = 30000 + pl * 1000 + b
                    b_info = bars_map[did]
                    pt = dt.getPointAtZ(b_info["z"])
                    if b < 60:
                        # Horizontal bar
                        y_c = 0.5 * (b_info["y_min"] + b_info["y_max"])
                        if abs(pt.Y() - y_c) <= 0.4 and (b_info["x_min"] + 1.0 <= pt.X() <= b_info["x_max"] - 1.0):
                            x_rel = pt.X() - b_info["x_min"]
                            p_bin = int(np.digitize(x_rel, m1_bins["dsh"]) - 1)
                            if 0 <= p_bin < len(m1_bins["dsh"]) - 1:
                                res_m1_exp["data"]["dsh"][p_bin] += 1
                                if did in fired_d:
                                    res_m1_det["data"]["dsh"][p_bin] += 1
                            if 0 <= th_bin < len(m2_bins) - 1:
                                res_m2_exp["data"]["ds"][th_bin] += 1
                                if did in fired_d:
                                    res_m2_det["data"]["ds"][th_bin] += 1
                    else:
                        # Vertical bar
                        x_c = 0.5 * (b_info["x_min"] + b_info["x_max"])
                        if abs(pt.X() - x_c) <= 0.4 and (b_info["y_min"] + 1.0 <= pt.Y() <= b_info["y_max"] - 1.0):
                            y_rel = pt.Y() - b_info["y_min"]
                            p_bin = int(np.digitize(y_rel, m1_bins["dsv"]) - 1)
                            if 0 <= p_bin < len(m1_bins["dsv"]) - 1:
                                res_m1_exp["data"]["dsv"][p_bin] += 1
                                if did in fired_d:
                                    res_m1_det["data"]["dsv"][p_bin] += 1

            # --- Method 4: Muon-Induced Shower Lateral Profile in DS ---
            n_ds_hits = sum(1 for did in fired_d if did // 10000 == 3)
            if n_ds_hits >= 5:
                for did, b_info in bars_map.items():
                    if b_info["sub"] != 3:
                        continue
                    pt = dt.getPointAtZ(b_info["z"])
                    if not b_info["is_vert"]:
                        y_c = 0.5 * (b_info["y_min"] + b_info["y_max"])
                        r_perp = abs(pt.Y() - y_c)
                    else:
                        x_c = 0.5 * (b_info["x_min"] + b_info["x_max"])
                        r_perp = abs(pt.X() - x_c)

                    r_bin = int(np.digitize(r_perp, m4_bins) - 1)
                    if 0 <= r_bin < len(m4_bins) - 1:
                        res_m4_tested["data"][r_bin] += 1
                        if did in fired_d:
                            res_m4_fired["data"][r_bin] += 1

        # --- Method 3: QDC Cutoff ---
        for did, q in fired_d.items():
            sub = did // 10000
            if sub == 1:
                res_m3_qdc["data"]["veto"].append(q)
            elif sub == 2:
                res_m3_qdc["data"]["us"].append(q)
            elif sub == 3:
                res_m3_qdc["data"]["ds"].append(q)

    print(f"[Data Done] Analyzed {passed_d:,} muons.")

    # =========================================================================
    # Process 2: PMU Simulation across 4 Models
    # =========================================================================
    print(f"\n[Processing Simulation] PMU from {mc_file}...")
    t_mc = ROOT.TChain("cbmsim")
    t_mc.Add(mc_file)
    n_entries_m = min(max_mc_events, t_mc.GetEntries())
    ioman_m = ROOT.FairRootManager.Instance()
    ioman_m.SetInChain(t_mc)

    trackTask = SndlhcTracking.Tracking()
    trackTask.SetName("simpleTracking")
    trackTask.SetTrackClassType(False)
    trackTask.Init()

    passed_m = 0

    for iev in range(n_entries_m):
        if (iev + 1) % 10000 == 0:
            print(f"  MC entry {iev+1:,}/{n_entries_m:,} (Muons analyzed: {passed_m})...", flush=True)
        if passed_m >= target_muons:
            break

        t_mc.GetEvent(iev)
        weight = float(getattr(t_mc, "mc_weight", 1.0)) * LUMI_SCALE_MC
        if weight <= 0:
            weight = 1.0

        # Accumulate energy deposits per bar from MuFilterPoint
        bar_edep_kev: Dict[int, float] = {}
        for pt in t_mc.MuFilterPoint:
            did = pt.GetDetectorID()
            bar_edep_kev[did] = bar_edep_kev.get(did, 0.0) + pt.GetEnergyLoss() * 1.0e6  # to keV

        # Form hit decisions for the 4 models:
        fired_mc: Dict[str, Dict[int, float]] = {m: {} for m in ["raw", "theo", "calib", "step"]}
        for did, edep in bar_edep_kev.items():
            if edep <= 0.0:
                continue
            # Model 1: Raw
            fired_mc["raw"][did] = edep
            # Model 2: Theo
            if edep > th_theo.get(did, 300.0):
                fired_mc["theo"][did] = edep
            # Model 3: Calib
            if edep > th_calib.get(did, 300.0):
                fired_mc["calib"][did] = edep
            # Model 4: Step
            if edep > th_step.get(did, 1500.0):
                fired_mc["step"][did] = edep

        # Track reconstruction
        trackTask.fittedTracks.Delete()
        trackTask.ExecuteTask(option="ScifiDS")

        scifi_trks, ds_trks = [], []
        for trk in trackTask.fittedTracks:
            if trk.getChi2Ndf() <= 5.0 and abs(trk.getSlopeXZ()) <= 0.1 and abs(trk.getSlopeYZ()) <= 0.1:
                if trk.getTrackType() == 1:
                    scifi_trks.append(trk)
                elif trk.getTrackType() == 3:
                    ds_trks.append(trk)

        if not scifi_trks and not ds_trks:
            continue

        passed_m += 1

        # --- Method 1 & 2: Veto ---
        if scifi_trks:
            st = scifi_trks[0]
            theta_deg = math.degrees(math.atan(math.hypot(st.getSlopeXZ(), st.getSlopeYZ())))
            th_bin = int(np.digitize(theta_deg, m2_bins) - 1)

            for pl in range(2):
                for b in range(7):
                    did = 10000 + pl * 1000 + b
                    b_info = bars_map[did]
                    pt = st.getPointAtZ(b_info["z"])
                    x_c = 0.5 * (b_info["x_min"] + b_info["x_max"])
                    if abs(pt.X() - x_c) <= 2.5 and (b_info["y_min"] + 1.0 <= pt.Y() <= b_info["y_max"] - 1.0):
                        y_rel = pt.Y() - b_info["y_min"]
                        p_bin = int(np.digitize(y_rel, m1_bins["veto"]) - 1)

                        for m in ["raw", "theo", "calib", "step"]:
                            if 0 <= p_bin < len(m1_bins["veto"]) - 1:
                                res_m1_exp[m]["veto"][p_bin] += 1
                                if did in fired_mc[m]:
                                    res_m1_det[m]["veto"][p_bin] += 1
                            if 0 <= th_bin < len(m2_bins) - 1:
                                res_m2_exp[m]["veto"][th_bin] += 1
                                if did in fired_mc[m]:
                                    res_m2_det[m]["veto"][th_bin] += 1

        # --- Method 1 & 2: US and DS ---
        if ds_trks:
            dt = ds_trks[0]
            theta_deg = math.degrees(math.atan(math.hypot(dt.getSlopeXZ(), dt.getSlopeYZ())))
            th_bin = int(np.digitize(theta_deg, m2_bins) - 1)

            # US Bars
            for pl in range(5):
                for b in range(10):
                    did = 20000 + pl * 1000 + b
                    b_info = bars_map[did]
                    pt = dt.getPointAtZ(b_info["z"])
                    y_c = 0.5 * (b_info["y_min"] + b_info["y_max"])
                    if abs(pt.Y() - y_c) <= 2.5 and (b_info["x_min"] + 1.0 <= pt.X() <= b_info["x_max"] - 1.0):
                        x_rel = pt.X() - b_info["x_min"]
                        p_bin = int(np.digitize(x_rel, m1_bins["us"]) - 1)
                        for m in ["raw", "theo", "calib", "step"]:
                            if 0 <= p_bin < len(m1_bins["us"]) - 1:
                                res_m1_exp[m]["us"][p_bin] += 1
                                if did in fired_mc[m]:
                                    res_m1_det[m]["us"][p_bin] += 1
                            if 0 <= th_bin < len(m2_bins) - 1:
                                res_m2_exp[m]["us"][th_bin] += 1
                                if did in fired_mc[m]:
                                    res_m2_det[m]["us"][th_bin] += 1

            # DS Bars
            for pl in range(4):
                b_start = 60 if pl == 3 else 0
                for b in range(b_start, 120):
                    did = 30000 + pl * 1000 + b
                    b_info = bars_map[did]
                    pt = dt.getPointAtZ(b_info["z"])
                    if b < 60:
                        y_c = 0.5 * (b_info["y_min"] + b_info["y_max"])
                        if abs(pt.Y() - y_c) <= 0.4 and (b_info["x_min"] + 1.0 <= pt.X() <= b_info["x_max"] - 1.0):
                            x_rel = pt.X() - b_info["x_min"]
                            p_bin = int(np.digitize(x_rel, m1_bins["dsh"]) - 1)
                            for m in ["raw", "theo", "calib", "step"]:
                                if 0 <= p_bin < len(m1_bins["dsh"]) - 1:
                                    res_m1_exp[m]["dsh"][p_bin] += 1
                                    if did in fired_mc[m]:
                                        res_m1_det[m]["dsh"][p_bin] += 1
                                if 0 <= th_bin < len(m2_bins) - 1:
                                    res_m2_exp[m]["ds"][th_bin] += 1
                                    if did in fired_mc[m]:
                                        res_m2_det[m]["ds"][th_bin] += 1
                    else:
                        x_c = 0.5 * (b_info["x_min"] + b_info["x_max"])
                        if abs(pt.X() - x_c) <= 0.4 and (b_info["y_min"] + 1.0 <= pt.Y() <= b_info["y_max"] - 1.0):
                            y_rel = pt.Y() - b_info["y_min"]
                            p_bin = int(np.digitize(y_rel, m1_bins["dsv"]) - 1)
                            for m in ["raw", "theo", "calib", "step"]:
                                if 0 <= p_bin < len(m1_bins["dsv"]) - 1:
                                    res_m1_exp[m]["dsv"][p_bin] += 1
                                    if did in fired_mc[m]:
                                        res_m1_det[m]["dsv"][p_bin] += 1

            # --- Method 4: Muon-Induced Shower Lateral Profile in DS ---
            for m in ["raw", "theo", "calib", "step"]:
                n_ds_hits = sum(1 for did in fired_mc[m] if did // 10000 == 3)
                if n_ds_hits >= 5:
                    for did, b_info in bars_map.items():
                        if b_info["sub"] != 3:
                            continue
                        pt = dt.getPointAtZ(b_info["z"])
                        if not b_info["is_vert"]:
                            y_c = 0.5 * (b_info["y_min"] + b_info["y_max"])
                            r_perp = abs(pt.Y() - y_c)
                        else:
                            x_c = 0.5 * (b_info["x_min"] + b_info["x_max"])
                            r_perp = abs(pt.X() - x_c)

                        r_bin = int(np.digitize(r_perp, m4_bins) - 1)
                        if 0 <= r_bin < len(m4_bins) - 1:
                            res_m4_tested[m][r_bin] += 1
                            if did in fired_mc[m]:
                                res_m4_fired[m][r_bin] += 1

        # --- Method 3: QDC Cutoff ---
        # Convert deposited energy (MeV) to predicted QDC
        for m in ["raw", "theo", "calib", "step"]:
            for did, edep_kev in fired_mc[m].items():
                sub = did // 10000
                edep_mev = edep_kev / 1000.0
                # Light yield: ~10,000 photons/MeV * 0.175 * 0.75 * 0.06 * 0.50 ~ 39 pe/MeV in US -> QDC = N_pe / 8.322
                if sub == 1:
                    q_pred = (edep_mev * 10000.0 * 0.175 * 0.90 * 0.06 * 0.50) / QDC_TO_PE
                    res_m3_qdc[m]["veto"].append(q_pred)
                elif sub == 2:
                    q_pred = (edep_mev * 10000.0 * 0.175 * 0.75 * 0.06 * 0.50) / QDC_TO_PE
                    res_m3_qdc[m]["us"].append(q_pred)
                elif sub == 3:
                    q_pred = (edep_mev * 10000.0 * 0.113 * 0.85 * 0.36 * 0.50) / QDC_TO_PE
                    res_m3_qdc[m]["ds"].append(q_pred)

    print(f"[MC Done] Analyzed {passed_m:,} muons.")

    # =========================================================================
    # Write to Single ROOT File with Structured TDirectories
    # =========================================================================
    print(f"\n[Writing ROOT] Creating structured ROOT file: {out_root}...")
    f_out = ROOT.TFile.Open(out_root, "RECREATE")

    dir_m1 = f_out.mkdir("Method1_Efficiency_vs_Position")
    dir_m2 = f_out.mkdir("Method2_Efficiency_vs_Angle")
    dir_m3 = f_out.mkdir("Method3_LowEnd_QDC_Cutoff")
    dir_m4 = f_out.mkdir("Method4_DS_Shower_Profile")
    dir_sum = f_out.mkdir("Canvases_Summary")

    # Helper styling
    style_specs = {
        "data":  {"color": ROOT.kBlack, "marker": 20, "ms": 1.1, "lw": 2, "name": "Data (Run 5408)"},
        "raw":   {"color": ROOT.kAzure + 1, "marker": 21, "ms": 1.0, "lw": 2, "name": "Sim Raw (Eth=0)"},
        "theo":  {"color": ROOT.kViolet + 1, "marker": 22, "ms": 1.0, "lw": 2, "name": "Sim Theo (k=1)"},
        "calib": {"color": ROOT.kRed + 1, "marker": 23, "ms": 1.1, "lw": 2, "name": "Sim Theo+k(SiPM)"},
        "step":  {"color": ROOT.kGreen + 2, "marker": 24, "ms": 1.0, "lw": 2, "name": "Sim Step (1/1.5 MeV)"},
    }

    # -------------------------------------------------------------------------
    # 1. Write Method 1 (Efficiency vs Position)
    # -------------------------------------------------------------------------
    dir_m1.cd()
    d1_us = dir_m1.mkdir("US_Planes")
    d1_ds = dir_m1.mkdir("DS_Planes")
    d1_veto = dir_m1.mkdir("Veto_Planes")

    m1_targets = [
        ("us", d1_us, "US Efficiency vs Position along Bar", "Position along US Bar [cm]", m1_bins["us"]),
        ("dsh", d1_ds, "DS-Horizontal Efficiency vs Position", "Position along DS-H Bar [cm]", m1_bins["dsh"]),
        ("dsv", d1_ds, "DS-Vertical Efficiency vs Position", "Position along DS-V Bar [cm]", m1_bins["dsv"]),
        ("veto", d1_veto, "Veto Efficiency vs Position", "Position along Veto Bar [cm]", m1_bins["veto"]),
    ]

    for key, target_dir, title, xlabel, b_edges in m1_targets:
        target_dir.cd()
        nb = len(b_edges) - 1
        c = ROOT.TCanvas(f"c_eff_pos_{key}", title, 900, 650)
        c.SetGridx()
        c.SetGridy()

        leg = ROOT.TLegend(0.60, 0.20, 0.88, 0.45)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)

        first = True
        for m in models:
            h = ROOT.TH1D(f"h_eff_pos_{key}_{m}", f"{title};{xlabel};Efficiency", nb, b_edges)
            exp = res_m1_exp[m][key]
            det = res_m1_det[m][key]
            for i in range(nb):
                val = (det[i] / exp[i]) if exp[i] > 0 else 0.0
                err = (math.sqrt(val * (1.0 - val) / exp[i])) if (exp[i] > 0 and 0.0 <= val <= 1.0) else 0.0
                h.SetBinContent(i + 1, val)
                h.SetBinError(i + 1, err)

            h.SetLineColor(style_specs[m]["color"])
            h.SetMarkerColor(style_specs[m]["color"])
            h.SetMarkerStyle(style_specs[m]["marker"])
            h.SetMarkerSize(style_specs[m]["ms"])
            h.SetLineWidth(style_specs[m]["lw"])
            h.SetMinimum(0.60)
            h.SetMaximum(1.05)
            h.Write()

            draw_opt = "E1" if first else "E1 SAME"
            h.Draw(draw_opt)
            first = False
            leg.AddEntry(h, style_specs[m]["name"], "lp")

        leg.Draw()
        c.Write()

    # -------------------------------------------------------------------------
    # 2. Write Method 2 (Efficiency vs Angle)
    # -------------------------------------------------------------------------
    dir_m2.cd()
    d2_us = dir_m2.mkdir("US_Angle")
    d2_ds = dir_m2.mkdir("DS_Angle")
    d2_veto = dir_m2.mkdir("Veto_Angle")

    m2_targets = [
        ("us", d2_us, "US Efficiency vs Track Angle #theta", "#theta [deg]"),
        ("ds", d2_ds, "DS Efficiency vs Track Angle #theta", "#theta [deg]"),
        ("veto", d2_veto, "Veto Efficiency vs Track Angle #theta", "#theta [deg]"),
    ]

    for key, target_dir, title, xlabel in m2_targets:
        target_dir.cd()
        nb = len(m2_bins) - 1
        c = ROOT.TCanvas(f"c_eff_ang_{key}", title, 900, 650)
        c.SetGridx()
        c.SetGridy()

        leg = ROOT.TLegend(0.60, 0.20, 0.88, 0.45)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)

        first = True
        for m in models:
            h = ROOT.TH1D(f"h_eff_ang_{key}_{m}", f"{title};{xlabel};Efficiency", nb, m2_bins)
            exp = res_m2_exp[m][key]
            det = res_m2_det[m][key]
            for i in range(nb):
                val = (det[i] / exp[i]) if exp[i] > 0 else 0.0
                err = (math.sqrt(val * (1.0 - val) / exp[i])) if (exp[i] > 0 and 0.0 <= val <= 1.0) else 0.0
                h.SetBinContent(i + 1, val)
                h.SetBinError(i + 1, err)

            h.SetLineColor(style_specs[m]["color"])
            h.SetMarkerColor(style_specs[m]["color"])
            h.SetMarkerStyle(style_specs[m]["marker"])
            h.SetMarkerSize(style_specs[m]["ms"])
            h.SetLineWidth(style_specs[m]["lw"])
            h.SetMinimum(0.60)
            h.SetMaximum(1.05)
            h.Write()

            draw_opt = "E1" if first else "E1 SAME"
            h.Draw(draw_opt)
            first = False
            leg.AddEntry(h, style_specs[m]["name"], "lp")

        leg.Draw()
        c.Write()

    # -------------------------------------------------------------------------
    # 3. Write Method 3 (Low-End QDC Cutoff Truncation)
    # -------------------------------------------------------------------------
    dir_m3.cd()
    d3_us = dir_m3.mkdir("US_QDC")
    d3_ds = dir_m3.mkdir("DS_QDC")
    d3_veto = dir_m3.mkdir("Veto_QDC")

    m3_targets = [
        ("us", d3_us, "US Low-End QDC Spectrum", "QDC [counts]"),
        ("ds", d3_ds, "DS Low-End QDC Spectrum", "QDC [counts]"),
        ("veto", d3_veto, "Veto Low-End QDC Spectrum", "QDC [counts]"),
    ]

    for key, target_dir, title, xlabel in m3_targets:
        target_dir.cd()
        nb = len(m3_bins) - 1
        c = ROOT.TCanvas(f"c_qdc_cutoff_{key}", title, 900, 650)
        c.SetGridx()
        c.SetGridy()

        leg = ROOT.TLegend(0.60, 0.65, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)

        first = True
        for m in models:
            h = ROOT.TH1D(f"h_qdc_cutoff_{key}_{m}", f"{title};{xlabel};Normalized Density", nb, m3_bins)
            vals = np.array(res_m3_qdc[m][key])
            if len(vals) > 0:
                for v in vals:
                    if v < 40.0:
                        h.Fill(v)
                if h.Integral() > 0:
                    h.Scale(1.0 / h.Integral())

            h.SetLineColor(style_specs[m]["color"])
            h.SetLineWidth(style_specs[m]["lw"])
            h.Write()

            draw_opt = "HIST" if first else "HIST SAME"
            h.Draw(draw_opt)
            first = False
            leg.AddEntry(h, style_specs[m]["name"], "l")

        leg.Draw()
        c.Write()

    # -------------------------------------------------------------------------
    # 4. Write Method 4 (Muon-Induced Shower Lateral Profile in DS)
    # -------------------------------------------------------------------------
    dir_m4.cd()
    d4_lat = dir_m4.mkdir("DS_Lateral_Profile")
    d4_lat.cd()

    c_m4 = ROOT.TCanvas("c_ds_shower_lateral_profile", "DS Shower Hit Probability vs r_perp", 900, 650)
    c_m4.SetGridx()
    c_m4.SetGridy()
    c_m4.SetLogy()

    leg_m4 = ROOT.TLegend(0.60, 0.65, 0.88, 0.88)
    leg_m4.SetBorderSize(0)
    leg_m4.SetFillStyle(0)

    nb_m4 = len(m4_bins) - 1
    first = True
    for m in models:
        h = ROOT.TH1D(f"h_ds_shower_profile_{m}", "DS Shower Hit Probability vs r_{#perp};r_{#perp} from Track [cm];Hit Probability", nb_m4, m4_bins)
        tst = res_m4_tested[m]
        fid = res_m4_fired[m]
        for i in range(nb_m4):
            prob = (fid[i] / tst[i]) if tst[i] > 0 else 1.0e-5
            err = (math.sqrt(prob * (1.0 - prob) / tst[i])) if (tst[i] > 0 and 0.0 < prob < 1.0) else 0.0
            h.SetBinContent(i + 1, max(prob, 1.0e-5))
            h.SetBinError(i + 1, err)

        h.SetLineColor(style_specs[m]["color"])
        h.SetMarkerColor(style_specs[m]["color"])
        h.SetMarkerStyle(style_specs[m]["marker"])
        h.SetMarkerSize(style_specs[m]["ms"])
        h.SetLineWidth(style_specs[m]["lw"])
        h.SetMinimum(1.0e-4)
        h.SetMaximum(1.2)
        h.Write()

        draw_opt = "E1" if first else "E1 SAME"
        h.Draw(draw_opt)
        first = False
        leg_m4.AddEntry(h, style_specs[m]["name"], "lp")

    leg_m4.Draw()
    c_m4.Write()

    # -------------------------------------------------------------------------
    # 5. Write Canvases Summary
    # -------------------------------------------------------------------------
    dir_sum.cd()
    # 4-panel multi-canvas
    c_sum = ROOT.TCanvas("c_validation_summary_methods1234", "TI18 Threshold Validation: Methods 1, 2, 3, 4", 1600, 1200)
    c_sum.Divide(2, 2)

    c_sum.cd(1)
    dir_m1.Get("US_Planes/c_eff_pos_us").DrawClonePad()
    c_sum.cd(2)
    dir_m2.Get("US_Angle/c_eff_ang_us").DrawClonePad()
    c_sum.cd(3)
    dir_m3.Get("US_QDC/c_qdc_cutoff_us").DrawClonePad()
    c_sum.cd(4)
    dir_m4.Get("DS_Lateral_Profile/c_ds_shower_lateral_profile").DrawClonePad()

    c_sum.Write()
    f_out.Close()

    print(f"\n[Done] Successfully saved all methods to {out_root}")

    # Generate Matplotlib PNG Figures
    generate_png_figures(
        res_m1_exp, res_m1_det, m1_bins,
        res_m2_exp, res_m2_det, m2_bins,
        res_m3_qdc, m3_bins,
        res_m4_tested, res_m4_fired, m4_bins,
        models,
    )


def generate_png_figures(
    m1_exp, m1_det, m1_bins,
    m2_exp, m2_det, m2_bins,
    m3_qdc, m3_bins,
    m4_tested, m4_fired, m4_bins,
    models,
):
    """Generates 4 publication-quality PNG figures for the 4 methods."""
    mpl_styles = {
        "data":  {"label": "Run 5408 Data", "color": "black", "ls": "none", "marker": "o", "lw": 1.5},
        "raw":   {"label": "Sim Raw (Eth=0)", "color": "#1f77b4", "ls": ":", "marker": "s", "lw": 1.6},
        "theo":  {"label": "Sim Theo (k=1)", "color": "#9467bd", "ls": "--", "marker": "D", "lw": 1.8},
        "calib": {"label": "Sim Theo+k(SiPM)", "color": "#d62728", "ls": "-", "marker": "v", "lw": 2.0},
        "step":  {"label": "Sim Step (1/1.5 MeV)", "color": "#2ca02c", "ls": "-.", "marker": "^", "lw": 1.8},
    }

    # Figure 1: Method 1 (Efficiency vs Position)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    subs = [("us", "US Bars (Planes 0..4)", axes[0, 0]),
            ("dsh", "DS Horizontal Bars", axes[0, 1]),
            ("dsv", "DS Vertical Bars", axes[1, 0]),
            ("veto", "Veto Bars", axes[1, 1])]

    for key, title, ax in subs:
        edges = m1_bins[key]
        centers = 0.5 * (edges[:-1] + edges[1:])
        for m in models:
            exp = m1_exp[m][key]
            det = m1_det[m][key]
            eff = np.zeros(len(centers))
            mask = exp > 0
            eff[mask] = det[mask] / exp[mask]
            st = mpl_styles[m]
            if m == "data":
                ax.errorbar(centers[mask], eff[mask], yerr=np.sqrt(eff[mask]*(1-eff[mask])/exp[mask]),
                            fmt="o", color=st["color"], label=st["label"], ms=5.0, capsize=2)
            else:
                ax.plot(centers[mask], eff[mask], color=st["color"], ls=st["ls"], label=st["label"], lw=st["lw"])

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Position along Bar [cm]", fontsize=10)
        ax.set_ylabel("Detection Efficiency", fontsize=10)
        ax.set_ylim(0.70, 1.02)
        ax.grid(True, ls=":", alpha=0.6)
        if ax == axes[0, 0]:
            ax.legend(frameon=False, fontsize=9)

    plt.savefig("plots/ti18_method1_efficiency_vs_position.png", dpi=200)
    plt.close()
    print("Saved plots/ti18_method1_efficiency_vs_position.png")

    # Figure 2: Method 2 (Efficiency vs Angle)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    ang_subs = [("us", "US Detector", axes[0]), ("ds", "DS Detector", axes[1]), ("veto", "Veto Detector", axes[2])]
    centers_ang = 0.5 * (m2_bins[:-1] + m2_bins[1:])

    for key, title, ax in ang_subs:
        for m in models:
            exp = m2_exp[m][key]
            det = m2_det[m][key]
            eff = np.zeros(len(centers_ang))
            mask = exp > 0
            eff[mask] = det[mask] / exp[mask]
            st = mpl_styles[m]
            if m == "data":
                ax.errorbar(centers_ang[mask], eff[mask], yerr=np.sqrt(eff[mask]*(1-eff[mask])/exp[mask]),
                            fmt="o", color=st["color"], label=st["label"], ms=5.0, capsize=2)
            else:
                ax.plot(centers_ang[mask], eff[mask], color=st["color"], ls=st["ls"], label=st["label"], lw=st["lw"])

        ax.set_title(f"{title}: Efficiency vs #theta", fontsize=12, fontweight="bold")
        ax.set_xlabel(r"Track Angle $\theta$ [deg]", fontsize=10)
        ax.set_ylabel("Detection Efficiency", fontsize=10)
        ax.set_ylim(0.80, 1.02)
        ax.grid(True, ls=":", alpha=0.6)
        if ax == axes[0]:
            ax.legend(frameon=False, fontsize=9)

    plt.savefig("plots/ti18_method2_efficiency_vs_angle.png", dpi=200)
    plt.close()
    print("Saved plots/ti18_method2_efficiency_vs_angle.png")

    # Figure 3: Method 3 (Low-End QDC Cutoff Truncation)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    qdc_subs = [("us", "US Low-End QDC Spectrum", axes[0]),
                ("ds", "DS Low-End QDC Spectrum", axes[1]),
                ("veto", "Veto Low-End QDC Spectrum", axes[2])]
    centers_qdc = 0.5 * (m3_bins[:-1] + m3_bins[1:])

    for key, title, ax in qdc_subs:
        for m in models:
            vals = np.array(m3_qdc[m][key])
            if len(vals) == 0:
                continue
            h, _ = np.histogram(vals[vals < 40.0], bins=m3_bins)
            norm = np.sum(h) * (m3_bins[1] - m3_bins[0])
            density = h / norm if norm > 0 else h
            st = mpl_styles[m]
            if m == "data":
                ax.step(centers_qdc, density, where="mid", color=st["color"], lw=2.0, label=st["label"])
            else:
                ax.step(centers_qdc, density, where="mid", color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("QDC [counts]", fontsize=10)
        ax.set_ylabel("Normalized Density", fontsize=10)
        ax.set_xlim(0, 40)
        ax.grid(True, ls=":", alpha=0.6)
        if ax == axes[0]:
            ax.legend(frameon=False, fontsize=9)

    plt.savefig("plots/ti18_method3_lowend_qdc_cutoff.png", dpi=200)
    plt.close()
    print("Saved plots/ti18_method3_lowend_qdc_cutoff.png")

    # Figure 4: Method 4 (Muon-Induced Shower Lateral Profile in DS)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    centers_r = 0.5 * (m4_bins[:-1] + m4_bins[1:])

    for m in models:
        tst = m4_tested[m]
        fid = m4_fired[m]
        prob = np.zeros(len(centers_r))
        mask = tst > 0
        prob[mask] = fid[mask] / tst[mask]
        st = mpl_styles[m]
        if m == "data":
            ax.errorbar(centers_r[mask], prob[mask], yerr=np.sqrt(prob[mask]*(1-prob[mask])/tst[mask]),
                        fmt="o", color=st["color"], label=st["label"], ms=5.0, capsize=2)
        else:
            ax.plot(centers_r[mask], prob[mask], color=st["color"], ls=st["ls"], label=st["label"], lw=st["lw"])

    ax.set_yscale("log")
    ax.set_title("Muon-Induced Shower Lateral Profile in DS Bars", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Transverse Distance $r_{\perp}$ from Track [cm]", fontsize=11)
    ax.set_ylabel(r"Hit Probability $P(\mathrm{hit} \mid r_{\perp})$", fontsize=11)
    ax.set_xlim(0, 25)
    ax.set_ylim(5.0e-4, 1.2)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(frameon=False, fontsize=10)

    plt.savefig("plots/ti18_method4_ds_shower_lateral_profile.png", dpi=200)
    plt.close()
    print("Saved plots/ti18_method4_ds_shower_lateral_profile.png")


def main():
    parser = argparse.ArgumentParser(description="Validate TI18 thresholds across Methods 1, 2, 3, 4")
    parser.add_argument("--data", default="/eos/experiment/sndlhc/convertedData/physics/2022/run_005408/sndsw_raw-0002.root")
    parser.add_argument("--mc", default="/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP_truth.root")
    parser.add_argument("--geofile", default="/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/geofile_full.Ntuple-TGeant4.root")
    parser.add_argument("--output", default="output/ti18_threshold_validation_methods1234.root")
    parser.add_argument("--max-data", type=int, default=30000)
    parser.add_argument("--max-mc", type=int, default=80000)
    parser.add_argument("--target-muons", type=int, default=1200)
    args = parser.parse_args()

    run_validation(
        data_file=args.data,
        mc_file=args.mc,
        geofile=args.geofile,
        out_root=args.output,
        max_data_events=args.max_data,
        max_mc_events=args.max_mc,
        target_muons=args.target_muons,
    )


if __name__ == "__main__":
    main()
