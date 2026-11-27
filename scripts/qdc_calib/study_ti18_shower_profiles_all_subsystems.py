#!/usr/bin/env python3
"""
scripts/qdc_calib/study_ti18_shower_profiles_all_subsystems.py
--------------------------------------------------------------
High-statistics in-situ shower lateral profile analysis for:
  1. DS Horizontal Bars (1 cm pitch, 2-sided readout)
  2. DS Vertical Bars   (1 cm pitch, 1-sided readout)
  3. US Scintillator Bars (6 cm pitch, alternating gel/air)

Methodology:
- Select clean muon-induced shower events in TI18 collision data (Run 5408) vs PMU simulation.
- Using reconstructed DS muon track, compute transverse perpendicular distance r_perp
  from the primary track core to each bar:
    * DS-H: r_perp = |y_bar - y_track(z_bar)| (in 1 cm bins)
    * DS-V: r_perp = |x_bar - x_track(z_bar)| (in 1 cm bins)
    * US:   r_perp = |y_bar - y_track(z_bar)| (in 3 cm bins, 0 to 36 cm)
- Measure hit probability P(hit | r_perp) = N_fired / N_tested with rigorous binomial errors.
- Evaluate 5 threshold models:
    * Data (Run 5408)
    * Sim Raw (Eth = 0)
    * Sim Theo (k = 1.0)
    * Sim Calib (k_SiPM from CSV)
    * Sim Step (1.0 MeV DS, 1.5 MeV US)

Saves all histograms, canvases, and ratio panels to:
  output/ti18_shower_profiles_all_subsystems.root
and generates:
  plots/ti18_shower_profiles_combined.png
  plots/ti18_shower_profile_ds_horizontal.png
  plots/ti18_shower_profile_ds_vertical.png
  plots/ti18_shower_profile_us_bars.png
"""

from __future__ import annotations

import os
import sys
import math
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import scripts.qdc_calib.min_pe as mpe
import SndlhcGeo
import SndlhcTracking

LUMI_SCALE_MC: float = 0.406569


def build_bars_geometry(geofile: str) -> Dict[int, Dict[str, Any]]:
    """Builds geometry coordinates for all Veto, US and DS bars."""
    geo = SndlhcGeo.GeoInterface(geofile)
    mufi = geo.modules["MuFilter"]
    A, B = ROOT.TVector3(), ROOT.TVector3()
    bars = {}

    # Veto (sub 1, planes 0..1, bars 0..6)
    for pl in range(2):
        for b in range(7):
            did = 10000 + pl * 1000 + b
            mufi.GetPosition(did, A, B)
            bars[did] = {
                "sub": 1, "plane": pl, "bar": b, "is_vert": False,
                "x_min": min(A.X(), B.X()), "x_max": max(A.X(), B.X()),
                "y_min": min(A.Y(), B.Y()), "y_max": max(A.Y(), B.Y()),
                "x_c": 0.5 * (A.X() + B.X()), "y_c": 0.5 * (A.Y() + B.Y()),
                "z": 0.5 * (A.Z() + B.Z()), "width": 6.0,
            }

    # US (sub 2, planes 0..4, bars 0..9)
    for pl in range(5):
        for b in range(10):
            did = 20000 + pl * 1000 + b
            mufi.GetPosition(did, A, B)
            bars[did] = {
                "sub": 2, "plane": pl, "bar": b, "is_vert": False,
                "x_min": min(A.X(), B.X()), "x_max": max(A.X(), B.X()),
                "y_min": min(A.Y(), B.Y()), "y_max": max(A.Y(), B.Y()),
                "x_c": 0.5 * (A.X() + B.X()), "y_c": 0.5 * (A.Y() + B.Y()),
                "z": 0.5 * (A.Z() + B.Z()), "width": 6.0,
            }

    # DS (sub 3, planes 0..3)
    for pl in range(4):
        b_start = 60 if pl == 3 else 0
        for b in range(b_start, 120):
            did = 30000 + pl * 1000 + b
            mufi.GetPosition(did, A, B)
            is_vert = (b >= 60)
            bars[did] = {
                "sub": 3, "plane": pl, "bar": b, "is_vert": is_vert,
                "x_min": min(A.X(), B.X()), "x_max": max(A.X(), B.X()),
                "y_min": min(A.Y(), B.Y()), "y_max": max(A.Y(), B.Y()),
                "x_c": 0.5 * (A.X() + B.X()), "y_c": 0.5 * (A.Y() + B.Y()),
                "z": 0.5 * (A.Z() + B.Z()), "width": 1.0,
            }

    return bars


def precompute_thresholds() -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Precomputes thresholds in keV with Scintillator Rise Time (0.9 ns) and Birks' quenching:
      1. Without SiPM-specific calibration factors (k = 1.0)
      2. With SiPM-specific calibration factors (k = k_SiPM from MIP calibrations)
    """
    calc_no_sipm = mpe.MinPE(year=2022, is_testbeam=False, include_rise_time=True, include_birks=True)
    calc_no_sipm._k_factors = {}

    calc_sipm = mpe.MinPE(year=2022, is_testbeam=False, include_rise_time=True, include_birks=True)

    th_no_sipm: Dict[int, float] = {}
    th_sipm: Dict[int, float] = {}

    # Veto
    for pl in range(2):
        for b in range(7):
            did = 10000 + pl * 1000 + b
            p = {"detector_id": did, "x": -28.0, "y": 0.0, "z": 0.0, "eloss": 1.0}
            th_no_sipm[did] = float(calc_no_sipm.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_sipm[did] = float(calc_sipm.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0

    # US
    for pl in range(5):
        for b in range(10):
            did = 20000 + pl * 1000 + b
            p = {"detector_id": did, "x": -37.7, "y": 0.0, "z": 0.0, "eloss": 1.0}
            th_no_sipm[did] = float(calc_no_sipm.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_sipm[did] = float(calc_sipm.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0

    # DS
    for pl in range(4):
        b_start = 60 if pl == 3 else 0
        for b in range(b_start, 120):
            did = 30000 + pl * 1000 + b
            is_vert = (b >= 60)
            p = {"detector_id": did, "x": 0.0 if is_vert else -37.7, "y": 20.0 if is_vert else 0.0, "z": 0.0, "eloss": 1.0}
            th_no_sipm[did] = float(calc_no_sipm.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_sipm[did] = float(calc_sipm.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0

    return th_no_sipm, th_sipm


def process_all_shower_profiles(
    data_file: str,
    mc_file: str,
    geofile: str,
    out_root: str = "output/ti18_shower_profiles_all_subsystems.root",
    max_data_events: int = 100000,
    max_mc_events: int = 150000,
    target_showers: int = 2500,
):
    os.makedirs(os.path.dirname(out_root) or ".", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    print("[Init] Extracting detector geometry...")
    bars_map = build_bars_geometry(geofile)
    print(f"[Init] Mapped {len(bars_map)} US and DS bars.")

    print("[Init] Precomputing physical thresholds...")
    th_no_sipm, th_sipm = precompute_thresholds()

    # Bin specifications
    # DS-H and DS-V: 0 to 25 cm (1.0 cm bins, 25 bins)
    bins_ds = np.linspace(0.0, 25.0, 26)
    nb_ds = len(bins_ds) - 1

    # US: 0 to 36 cm (3.0 cm bins, 12 bins)
    bins_us = np.linspace(0.0, 36.0, 13)
    nb_us = len(bins_us) - 1

    # Veto: 0 to 24 cm (3.0 cm bins, 8 bins)
    bins_veto = np.linspace(0.0, 24.0, 9)
    nb_veto = len(bins_veto) - 1

    targets = ["ds_h", "ds_v", "us", "us_pl4", "veto"]
    models = ["data", "raw", "corr_no_sipm", "corr_sipm"]
    model_labels = {
        "data": "Run 5408 Data",
        "raw": "Sim Raw (Eth=0)",
        "corr_no_sipm": "Sim Corrected (Rise+Birks, k=1.0)",
        "corr_sipm": "Sim Corrected (Rise+Birks+k_SiPM)",
    }

    def get_nb(t: str) -> int:
        if "ds" in t:
            return nb_ds
        elif "us" in t:
            return nb_us
        else:
            return nb_veto

    n_tested = {t: {m: np.zeros(get_nb(t), dtype=np.float64) for m in models} for t in targets}
    n_fired = {t: {m: np.zeros(get_nb(t), dtype=np.float64) for m in models} for t in targets}

    # =========================================================================
    # 1. Real Collision Data (Run 5408)
    # =========================================================================
    print(f"\n[Processing Data] Scanning {max_data_events:,} entries from {data_file}...")
    t_data = ROOT.TChain("rawConv")
    t_data.Add(data_file)
    ioman_d = ROOT.FairRootManager.Instance()
    ioman_d.SetInChain(t_data)

    trackTask = SndlhcTracking.Tracking()
    trackTask.SetName("simpleTracking")
    trackTask.SetTrackClassType(False)
    trackTask.Init()

    n_showers_d = 0
    prev_time = None
    n_entries_d = min(max_data_events, t_data.GetEntries())

    for iev in range(n_entries_d):
        if (iev + 1) % 20000 == 0:
            print(f"  Data {iev+1:,}/{n_entries_d:,} (Showers: {n_showers_d})...", flush=True)
        if n_showers_d >= target_showers:
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

        fired_bars = set()
        for hit in t_data.Digi_MuFilterHits:
            if hit.isValid():
                fired_bars.add(hit.GetDetectorID())

        n_ds = sum(1 for did in fired_bars if did // 10000 == 3)
        n_us = sum(1 for did in fired_bars if did // 10000 == 2)
        if (n_ds + n_us) < 8:
            continue

        trackTask.fittedTracks.Delete()
        trackTask.ExecuteTask(option="ScifiDS")

        ds_trk = None
        scifi_trk = None
        for trk in trackTask.fittedTracks:
            if trk.getChi2Ndf() <= 5.0 and abs(trk.getSlopeXZ()) <= 0.1 and abs(trk.getSlopeYZ()) <= 0.1:
                if trk.getTrackType() == 3 and ds_trk is None:
                    ds_trk = trk
                elif trk.getTrackType() == 1 and scifi_trk is None:
                    scifi_trk = trk

        if ds_trk is None:
            continue

        n_showers_d += 1

        for did, b_info in bars_map.items():
            sub = b_info["sub"]
            trk_eval = scifi_trk if (sub == 1 and scifi_trk is not None) else ds_trk
            pt = trk_eval.getPointAtZ(b_info["z"])

            if sub == 3 and not b_info["is_vert"]:
                # DS Horizontal
                if b_info["x_min"] - 2.0 <= pt.X() <= b_info["x_max"] + 2.0:
                    r = abs(pt.Y() - b_info["y_c"])
                    b_idx = int(np.digitize(r, bins_ds) - 1)
                    if 0 <= b_idx < nb_ds:
                        n_tested["ds_h"]["data"][b_idx] += 1.0
                        if did in fired_bars:
                            n_fired["ds_h"]["data"][b_idx] += 1.0

            elif sub == 3 and b_info["is_vert"]:
                # DS Vertical
                if b_info["y_min"] - 2.0 <= pt.Y() <= b_info["y_max"] + 2.0:
                    r = abs(pt.X() - b_info["x_c"])
                    b_idx = int(np.digitize(r, bins_ds) - 1)
                    if 0 <= b_idx < nb_ds:
                        n_tested["ds_v"]["data"][b_idx] += 1.0
                        if did in fired_bars:
                            n_fired["ds_v"]["data"][b_idx] += 1.0

            elif sub == 2:
                # US Bars (Horizontal)
                if b_info["x_min"] - 2.0 <= pt.X() <= b_info["x_max"] + 2.0:
                    r = abs(pt.Y() - b_info["y_c"])
                    b_idx = int(np.digitize(r, bins_us) - 1)
                    if 0 <= b_idx < nb_us:
                        n_tested["us"]["data"][b_idx] += 1.0
                        if did in fired_bars:
                            n_fired["us"]["data"][b_idx] += 1.0
                        if b_info["plane"] == 4:
                            n_tested["us_pl4"]["data"][b_idx] += 1.0
                            if did in fired_bars:
                                n_fired["us_pl4"]["data"][b_idx] += 1.0

            elif sub == 1:
                # Veto Bars (Horizontal)
                if b_info["x_min"] - 2.0 <= pt.X() <= b_info["x_max"] + 2.0:
                    r = abs(pt.Y() - b_info["y_c"])
                    b_idx = int(np.digitize(r, bins_veto) - 1)
                    if 0 <= b_idx < nb_veto:
                        n_tested["veto"]["data"][b_idx] += 1.0
                        if did in fired_bars:
                            n_fired["veto"]["data"][b_idx] += 1.0

    print(f"[Data Done] Total shower events analyzed: {n_showers_d:,}")

    # =========================================================================
    # 2. PMU Simulation Sample across 4 Models
    # =========================================================================
    print(f"\n[Processing Simulation] Scanning {max_mc_events:,} entries from {mc_file}...")
    t_mc = ROOT.TChain("cbmsim")
    t_mc.Add(mc_file)
    ioman_m = ROOT.FairRootManager.Instance()
    ioman_m.SetInChain(t_mc)

    trackTask = SndlhcTracking.Tracking()
    trackTask.SetName("simpleTracking")
    trackTask.SetTrackClassType(False)
    trackTask.Init()

    n_showers_m = 0
    n_entries_m = min(max_mc_events, t_mc.GetEntries())

    for iev in range(n_entries_m):
        if (iev + 1) % 25000 == 0:
            print(f"  MC {iev+1:,}/{n_entries_m:,} (Showers: {n_showers_m})...", flush=True)
        if n_showers_m >= target_showers:
            break

        t_mc.GetEvent(iev)

        bar_edep_kev: Dict[int, float] = {}
        for pt in t_mc.MuFilterPoint:
            did = pt.GetDetectorID()
            bar_edep_kev[did] = bar_edep_kev.get(did, 0.0) + pt.GetEnergyLoss() * 1.0e6

        if len(bar_edep_kev) < 8:
            continue

        # Form hit sets for the 3 MC models
        fired_mc: Dict[str, set] = {m: set() for m in ["raw", "corr_no_sipm", "corr_sipm"]}
        for did, edep in bar_edep_kev.items():
            if edep <= 0.0:
                continue
            fired_mc["raw"].add(did)
            if edep > th_no_sipm.get(did, 300.0):
                fired_mc["corr_no_sipm"].add(did)
            if edep > th_sipm.get(did, 300.0):
                fired_mc["corr_sipm"].add(did)

        trackTask.fittedTracks.Delete()
        trackTask.ExecuteTask(option="ScifiDS")

        ds_trk = None
        scifi_trk = None
        for trk in trackTask.fittedTracks:
            if trk.getChi2Ndf() <= 5.0 and abs(trk.getSlopeXZ()) <= 0.1 and abs(trk.getSlopeYZ()) <= 0.1:
                if trk.getTrackType() == 3 and ds_trk is None:
                    ds_trk = trk
                elif trk.getTrackType() == 1 and scifi_trk is None:
                    scifi_trk = trk

        if ds_trk is None:
            continue

        n_showers_m += 1

        for did, b_info in bars_map.items():
            sub = b_info["sub"]
            trk_eval = scifi_trk if (sub == 1 and scifi_trk is not None) else ds_trk
            pt = trk_eval.getPointAtZ(b_info["z"])

            if sub == 3 and not b_info["is_vert"]:
                # DS Horizontal
                if b_info["x_min"] - 2.0 <= pt.X() <= b_info["x_max"] + 2.0:
                    r = abs(pt.Y() - b_info["y_c"])
                    b_idx = int(np.digitize(r, bins_ds) - 1)
                    if 0 <= b_idx < nb_ds:
                        for m in ["raw", "corr_no_sipm", "corr_sipm"]:
                            n_tested["ds_h"][m][b_idx] += 1.0
                            if did in fired_mc[m]:
                                n_fired["ds_h"][m][b_idx] += 1.0

            elif sub == 3 and b_info["is_vert"]:
                # DS Vertical
                if b_info["y_min"] - 2.0 <= pt.Y() <= b_info["y_max"] + 2.0:
                    r = abs(pt.X() - b_info["x_c"])
                    b_idx = int(np.digitize(r, bins_ds) - 1)
                    if 0 <= b_idx < nb_ds:
                        for m in ["raw", "corr_no_sipm", "corr_sipm"]:
                            n_tested["ds_v"][m][b_idx] += 1.0
                            if did in fired_mc[m]:
                                n_fired["ds_v"][m][b_idx] += 1.0

            elif sub == 2:
                # US Bars
                if b_info["x_min"] - 2.0 <= pt.X() <= b_info["x_max"] + 2.0:
                    r = abs(pt.Y() - b_info["y_c"])
                    b_idx = int(np.digitize(r, bins_us) - 1)
                    if 0 <= b_idx < nb_us:
                        for m in ["raw", "corr_no_sipm", "corr_sipm"]:
                            n_tested["us"][m][b_idx] += 1.0
                            if did in fired_mc[m]:
                                n_fired["us"][m][b_idx] += 1.0
                            if b_info["plane"] == 4:
                                n_tested["us_pl4"][m][b_idx] += 1.0
                                if did in fired_mc[m]:
                                    n_fired["us_pl4"][m][b_idx] += 1.0

            elif sub == 1:
                # Veto Bars
                if b_info["x_min"] - 2.0 <= pt.X() <= b_info["x_max"] + 2.0:
                    r = abs(pt.Y() - b_info["y_c"])
                    b_idx = int(np.digitize(r, bins_veto) - 1)
                    if 0 <= b_idx < nb_veto:
                        for m in ["raw", "corr_no_sipm", "corr_sipm"]:
                            n_tested["veto"][m][b_idx] += 1.0
                            if did in fired_mc[m]:
                                n_fired["veto"][m][b_idx] += 1.0

    print(f"[MC Done] Total shower events analyzed: {n_showers_m:,}")

    # =========================================================================
    # 3. Save to ROOT File with Structured Directories
    # =========================================================================
    print(f"\n[Writing ROOT] Saving to {out_root}...")
    f_out = ROOT.TFile.Open(out_root, "RECREATE")

    dir_dsh = f_out.mkdir("DS_Horizontal")
    dir_dsv = f_out.mkdir("DS_Vertical")
    dir_us = f_out.mkdir("US_Bars")
    dir_us4 = f_out.mkdir("US_Plane4")
    dir_veto = f_out.mkdir("Veto_Bars")
    dir_sum = f_out.mkdir("Summary_Canvases")

    dir_map = {
        "ds_h": dir_dsh,
        "ds_v": dir_dsv,
        "us": dir_us,
        "us_pl4": dir_us4,
        "veto": dir_veto,
    }
    bin_map = {
        "ds_h": bins_ds,
        "ds_v": bins_ds,
        "us": bins_us,
        "us_pl4": bins_us,
        "veto": bins_veto,
    }

    root_styles = {
        "data":         {"color": ROOT.kBlack, "marker": 20, "ms": 1.2, "lw": 2},
        "raw":          {"color": ROOT.kGray + 1, "marker": 21, "ms": 1.0, "lw": 1},
        "corr_no_sipm": {"color": ROOT.kOrange + 7, "marker": 22, "ms": 1.1, "lw": 2},
        "corr_sipm":    {"color": ROOT.kRed + 1, "marker": 20, "ms": 1.2, "lw": 2},
    }

    probs: Dict[str, Dict[str, np.ndarray]] = {t: {} for t in targets}
    errs: Dict[str, Dict[str, np.ndarray]] = {t: {} for t in targets}

    for t in targets:
        tdir = dir_map[t]
        tdir.cd()
        b_edges = bin_map[t]
        nb_t = len(b_edges) - 1

        c = ROOT.TCanvas(f"c_shower_profile_{t}", f"Shower Lateral Profile: {t.upper()}", 900, 700)
        c.SetLogy()
        c.SetGridx()
        c.SetGridy()

        leg = ROOT.TLegend(0.55, 0.65, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)

        first = True
        for m in models:
            tst = n_tested[t][m]
            fid = n_fired[t][m]
            p = np.zeros(nb_t, dtype=np.float64)
            e = np.zeros(nb_t, dtype=np.float64)
            mask = tst > 0
            p[mask] = fid[mask] / tst[mask]
            e[mask] = np.sqrt(np.clip(p[mask] * (1.0 - p[mask]) / tst[mask], 1e-12, 1.0))
            probs[t][m] = p
            errs[t][m] = e

            h = ROOT.TH1D(f"h_shower_prof_{t}_{m}", f"{model_labels[m]};r_{{#perp}} from Track [cm];Hit Probability", nb_t, b_edges)
            for i in range(nb_t):
                h.SetBinContent(i + 1, max(p[i], 1e-5))
                h.SetBinError(i + 1, e[i])

            st = root_styles[m]
            h.SetLineColor(st["color"])
            h.SetMarkerColor(st["color"])
            h.SetMarkerStyle(st["marker"])
            h.SetMarkerSize(st["ms"])
            h.SetLineWidth(st["lw"])
            h.SetMinimum(2.0e-4)
            h.SetMaximum(1.5)
            h.Write()

            opt = "E1" if first else "E1 SAME"
            h.Draw(opt)
            first = False
            leg.AddEntry(h, model_labels[m], "lp")

        leg.Draw()
        c.Write()

    # Write summary multi-canvas (2x2 grid)
    dir_sum.cd()
    c_sum = ROOT.TCanvas("c_shower_profiles_all_subsystems", "Shower Profiles across DS-H, DS-V, US, Veto", 1400, 1100)
    c_sum.Divide(2, 2)
    c_sum.cd(1)
    dir_dsh.Get("c_shower_profile_ds_h").DrawClonePad()
    c_sum.cd(2)
    dir_dsv.Get("c_shower_profile_ds_v").DrawClonePad()
    c_sum.cd(3)
    dir_us.Get("c_shower_profile_us").DrawClonePad()
    c_sum.cd(4)
    dir_veto.Get("c_shower_profile_veto").DrawClonePad()
    c_sum.Write()

    f_out.Close()
    print(f"Saved ROOT file to: {out_root}")

    # =========================================================================
    # 4. Generate Matplotlib Figures
    # =========================================================================
    generate_all_profile_figures(probs, errs, bin_map, models, model_labels, n_showers_d)


def generate_all_profile_figures(probs, errs, bin_map, models, model_labels, n_data):
    """Generates 4 separate individual panel PNGs and 1 combined 4-panel figure (2x2)."""
    mpl_styles = {
        "data":         {"label": f"Run 5408 Data (N={n_data:,})", "color": "black", "ls": "none", "marker": "o", "ms": 5.5, "lw": 1.5},
        "raw":          {"label": "Sim Raw (Eth=0)", "color": "#7f7f7f", "ls": ":", "marker": "s", "ms": 4.0, "lw": 1.4},
        "corr_no_sipm": {"label": "Sim Corrected (Rise+Birks, k=1.0)", "color": "#ff7f0e", "ls": "--", "marker": "^", "ms": 5.0, "lw": 1.8},
        "corr_sipm":    {"label": "Sim Corrected (Rise+Birks+k_SiPM)", "color": "#d62728", "ls": "-", "marker": "o", "ms": 5.0, "lw": 2.0},
    }

    target_titles = {
        "ds_h":   ("DS Horizontal Bars (1 cm pitch, 2-sided readout)", "plots/ti18_shower_profile_ds_horizontal.png"),
        "ds_v":   ("DS Vertical Bars (1 cm pitch, 1-sided readout)", "plots/ti18_shower_profile_ds_vertical.png"),
        "us":     ("US Bars All Planes (6 cm pitch, alternating gel/air)", "plots/ti18_shower_profile_us_bars.png"),
        "us_pl4": ("US Plane 4 Only (Downstream-most, closest to DS)", "plots/ti18_shower_profile_us_plane4.png"),
        "veto":   ("Veto Bars (6 cm pitch, 2-sided readout, silicone gel)", "plots/ti18_shower_profile_veto_bars.png"),
    }

    # 1. Generate Individual Figures with Ratio Panels
    for t, (title, out_png) in target_titles.items():
        edges = bin_map[t]
        centers = 0.5 * (edges[:-1] + edges[1:])
        nb = len(centers)

        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1, figsize=(8.5, 7.0),
            gridspec_kw={"height_ratios": [3.2, 1.3], "hspace": 0.08},
            constrained_layout=False,
        )

        for m in models:
            st = mpl_styles[m]
            p = probs[t][m]
            e = errs[t][m]
            mask = p > 0

            if m == "data":
                ax_main.errorbar(
                    centers[mask], p[mask], yerr=e[mask],
                    fmt=st["marker"], color=st["color"], ms=st["ms"],
                    lw=st["lw"], capsize=2.5, label=st["label"], zorder=10
                )
            else:
                ax_main.errorbar(
                    centers[mask], p[mask], yerr=e[mask],
                    fmt=st["marker"], color=st["color"], ls=st["ls"],
                    ms=st["ms"], lw=st["lw"], capsize=1.5, label=st["label"], alpha=0.9
                )

        ax_main.set_yscale("log")
        ax_main.set_xlim(0, edges[-1])
        ax_main.set_ylim(2.0e-4, 1.2)
        ax_main.set_ylabel(r"Hit Probability $P(\mathrm{hit} \mid r_{\perp})$", fontsize=11, fontweight="bold")
        ax_main.set_title(title, fontsize=12, fontweight="bold")
        ax_main.grid(True, which="both", ls=":", alpha=0.55)
        ax_main.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.0, loc="upper right")
        ax_main.tick_params(labelbottom=False, labelsize=10)

        # Ratio Panel (MC / Data)
        p_d = probs[t]["data"]
        e_d = errs[t]["data"]

        for m in ["raw", "corr_no_sipm", "corr_sipm"]:
            st = mpl_styles[m]
            p_m = probs[t][m]
            e_m = errs[t][m]

            valid = (p_d > 0) & (p_m > 0)
            ratio = np.full(nb, np.nan)
            ratio_err = np.full(nb, np.nan)
            ratio[valid] = p_m[valid] / p_d[valid]
            rel_err = np.sqrt((e_m[valid] / p_m[valid])**2 + (e_d[valid] / p_d[valid])**2)
            ratio_err[valid] = ratio[valid] * rel_err

            ax_ratio.errorbar(
                centers[valid], ratio[valid], yerr=ratio_err[valid],
                fmt=st["marker"], color=st["color"], ls=st["ls"],
                ms=st["ms"] * 0.9, lw=st["lw"] * 0.9, capsize=1.5
            )

        ax_ratio.axhline(1.0, color="black", ls="--", lw=1.2)
        ax_ratio.set_xlim(0, edges[-1])
        ax_ratio.set_ylim(0.0, 4.0)
        ax_ratio.set_xlabel(r"Transverse Distance from Track Core $r_{\perp}$ [cm]", fontsize=11, fontweight="bold")
        ax_ratio.set_ylabel("MC / Data", fontsize=10, fontweight="bold")
        ax_ratio.grid(True, ls=":", alpha=0.55)
        ax_ratio.tick_params(labelsize=10)
        ax_ratio.yaxis.set_minor_locator(AutoMinorLocator())

        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_png}")

    # 2. Combined 4-Panel Figure (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(17, 13), constrained_layout=True)
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    comb_targets = ["ds_h", "ds_v", "us", "veto"]

    for idx, t in enumerate(comb_targets):
        title, _ = target_titles[t]
        ax = ax_list[idx]
        edges = bin_map[t]
        centers = 0.5 * (edges[:-1] + edges[1:])

        for m in models:
            st = mpl_styles[m]
            p = probs[t][m]
            e = errs[t][m]
            mask = p > 0
            if m == "data":
                ax.errorbar(
                    centers[mask], p[mask], yerr=e[mask],
                    fmt=st["marker"], color=st["color"], ms=st["ms"],
                    lw=st["lw"], capsize=2.5, label=st["label"], zorder=10
                )
            else:
                ax.errorbar(
                    centers[mask], p[mask], yerr=e[mask],
                    fmt=st["marker"], color=st["color"], ls=st["ls"],
                    ms=st["ms"], lw=st["lw"], capsize=1.5, label=st["label"], alpha=0.9
                )

        ax.set_yscale("log")
        ax.set_xlim(0, edges[-1])
        ax.set_ylim(2.0e-4, 1.2)
        ax.set_xlabel(r"$r_{\perp}$ from Track [cm]", fontsize=11, fontweight="bold")
        ax.set_ylabel(r"Hit Probability $P(\mathrm{hit} \mid r_{\perp})$", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", ls=":", alpha=0.55)
        if idx == 0:
            ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.0)

    comb_png = "plots/ti18_shower_profiles_combined.png"
    plt.savefig(comb_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {comb_png}")

    # 3. Direct Comparison: US Plane 4 vs US All Planes (with MCS Track Scattering Uncertainty)
    fig_comp, (ax_c_main, ax_c_ratio) = plt.subplots(
        2, 1, figsize=(9.0, 7.5),
        gridspec_kw={"height_ratios": [3.2, 1.3], "hspace": 0.08},
        constrained_layout=False,
    )
    edges_u = bin_map["us"]
    centers_u = 0.5 * (edges_u[:-1] + edges_u[1:])
    nb_u = len(centers_u)

    # Data All vs Pl4
    p_d_all = probs["us"]["data"]
    e_d_all = errs["us"]["data"]
    p_d_pl4 = probs["us_pl4"]["data"]
    e_d_pl4 = errs["us_pl4"]["data"]

    # MC Corrected (k_SiPM) All vs Pl4
    p_m_all = probs["us"]["corr_sipm"]
    e_m_all = errs["us"]["corr_sipm"]
    p_m_pl4 = probs["us_pl4"]["corr_sipm"]
    e_m_pl4 = errs["us_pl4"]["corr_sipm"]

    m_d_all = p_d_all > 0
    m_d_pl4 = p_d_pl4 > 0
    m_m_all = p_m_all > 0
    m_m_pl4 = p_m_pl4 > 0

    ax_c_main.errorbar(centers_u[m_d_all], p_d_all[m_d_all], yerr=e_d_all[m_d_all],
                       fmt="o", color="black", ms=5.5, lw=1.5, capsize=2.0, label=f"Run 5408 Data (All 5 US Planes)")
    ax_c_main.errorbar(centers_u[m_d_pl4], p_d_pl4[m_d_pl4], yerr=e_d_pl4[m_d_pl4],
                       fmt="s", color="#1f77b4", ms=5.5, lw=1.5, capsize=2.0, label=f"Run 5408 Data (US Plane 4 Only)")

    ax_c_main.errorbar(centers_u[m_m_all], p_m_all[m_m_all], yerr=e_m_all[m_m_all],
                       fmt="^", color="#d62728", ls="-", ms=5.0, lw=1.8, capsize=1.5, label="Sim Corr (Rise+Birks+k_SiPM, All US)")
    ax_c_main.errorbar(centers_u[m_m_pl4], p_m_pl4[m_m_pl4], yerr=e_m_pl4[m_m_pl4],
                       fmt="v", color="#2ca02c", ls="--", ms=5.0, lw=1.8, capsize=1.5, label="Sim Corr (Rise+Birks+k_SiPM, US Plane 4)")

    # Multiple Coulomb Scattering (MCS) Uncertainty Band on Sim
    # Highland formula: sigma_theta ~ (13.6 MeV / p) * sqrt(x / X0)
    # Extrapolating back to US Pl4 (8 cm Fe, delta_z = 22.5 cm) gives sigma_perp ~ 0.3 cm
    # Extrapolating back to US Pl0 (40 cm Fe, delta_z = 113 cm) gives sigma_perp ~ 1.8 cm
    # Propagated to P(hit) via local slope dP/dr
    dp_dr = np.zeros(nb_u)
    for i in range(nb_u):
        if 0 < i < nb_u - 1 and p_m_all[i-1] > 0 and p_m_all[i+1] > 0:
            dp_dr[i] = abs(p_m_all[i+1] - p_m_all[i-1]) / (centers_u[i+1] - centers_u[i-1])
        elif i == 0 and p_m_all[1] > 0:
            dp_dr[i] = abs(p_m_all[1] - p_m_all[0]) / (centers_u[1] - centers_u[0])

    sigma_perp_all = 1.2  # cm average MCS lever arm across 5 planes
    sigma_p_track = dp_dr * sigma_perp_all
    p_band_low = np.clip(p_m_all - sigma_p_track, 1.0e-4, 1.5)
    p_band_high = p_m_all + sigma_p_track
    ax_c_main.fill_between(centers_u, p_band_low, p_band_high, color="#d62728", alpha=0.18, label="Track MCS Uncertainty Band (sigma_r ~ 1.2 cm)")

    ax_c_main.set_yscale("log")
    ax_c_main.set_xlim(0, edges_u[-1])
    ax_c_main.set_ylim(2.0e-4, 1.2)
    ax_c_main.set_ylabel(r"Hit Probability $P(\mathrm{hit} \mid r_{\perp})$", fontsize=11, fontweight="bold")
    ax_c_main.set_title("US Plane 4 (Closest to DS, 1 Fe Plate) vs. All 5 US Planes", fontsize=12, fontweight="bold")
    ax_c_main.grid(True, which="both", ls=":", alpha=0.55)
    ax_c_main.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.8, loc="upper right")
    ax_c_main.tick_params(labelbottom=False, labelsize=10)

    # Ratio Panel: Sim / Data for All vs Pl4
    v_all = (p_d_all > 0) & (p_m_all > 0)
    r_all = np.full(nb_u, np.nan)
    r_all[v_all] = p_m_all[v_all] / p_d_all[v_all]

    v_pl4 = (p_d_pl4 > 0) & (p_m_pl4 > 0)
    r_pl4 = np.full(nb_u, np.nan)
    r_pl4[v_pl4] = p_m_pl4[v_pl4] / p_d_pl4[v_pl4]

    ax_c_ratio.plot(centers_u[v_all], r_all[v_all], "o-", color="#d62728", lw=1.6, ms=4.5, label="MC/Data: All US Planes")
    ax_c_ratio.plot(centers_u[v_pl4], r_pl4[v_pl4], "s--", color="#2ca02c", lw=1.6, ms=5.0, label="MC/Data: US Plane 4 Only")

    ax_c_ratio.axhline(1.0, color="black", ls="--", lw=1.2)
    ax_c_ratio.set_xlim(0, edges_u[-1])
    ax_c_ratio.set_ylim(0.0, 4.0)
    ax_c_ratio.set_xlabel(r"Transverse Distance from Track Core $r_{\perp}$ [cm]", fontsize=11, fontweight="bold")
    ax_c_ratio.set_ylabel("MC / Data", fontsize=10, fontweight="bold")
    ax_c_ratio.grid(True, ls=":", alpha=0.55)
    ax_c_ratio.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5, loc="upper right")
    ax_c_ratio.tick_params(labelsize=10)
    ax_c_ratio.yaxis.set_minor_locator(AutoMinorLocator())

    pl4_comp_png = "plots/ti18_us_plane4_vs_all.png"
    plt.savefig(pl4_comp_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pl4_comp_png}")


def main():
    parser = argparse.ArgumentParser(description="Shower lateral profile across DS-H, DS-V, US")
    parser.add_argument("--data", default="/eos/experiment/sndlhc/convertedData/physics/2022/run_005408/sndsw_raw-0002.root")
    parser.add_argument("--mc", default="/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP_truth.root")
    parser.add_argument("--geofile", default="/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/geofile_full.Ntuple-TGeant4.root")
    parser.add_argument("--output", default="output/ti18_shower_profiles_all_subsystems.root")
    parser.add_argument("--max-data", type=int, default=80000)
    parser.add_argument("--max-mc", type=int, default=120000)
    parser.add_argument("--target-showers", type=int, default=2000)
    args = parser.parse_args()

    process_all_shower_profiles(
        data_file=args.data,
        mc_file=args.mc,
        geofile=args.geofile,
        out_root=args.output,
        max_data_events=args.max_data,
        max_mc_events=args.max_mc,
        target_showers=args.target_showers,
    )


if __name__ == "__main__":
    main()
