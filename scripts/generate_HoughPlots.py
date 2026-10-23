#!/usr/bin/env python3
"""
generate_HoughPlots.py
----------------------
Generates binned 2D Hough space plots and histograms for SND@LHC events, mirroring
the command-line interface, multi-processing architecture, and TDirectory hierarchy
of scripts/generate_2DEventDisplays.py.

Key Features:
  - Parses tracking parameters from an XML file (e.g. config/TrackingParams.xml)
    for a chosen tracking case (default: 'dimuon_DS') and Hough format ('linearSlopeIntercept').
  - Computes the 2D Hough space accumulator with configurable binning, ranges, and Gaussian smoothing.
  - Generates 2D histogram accumulators (TH2D) with vote counts for both XZ and YZ projections.
  - Draws interactive TCanvases featuring:
      * 2D binned Hough vote distribution with Viridis color palette.
      * Transparent shaded region and vertical boundary lines for angular acceptance (|theta| <= 175 mrad).
      * Outlines/boundary lines for lines that miss the predefined target fiducial area at z = 352 cm:
          - XZ target window: [-50.0, -3.0] cm
          - YZ target window: [5.0, 60.0] cm
      * Reconstructed track markers (Reco_MuonTracks) plotted at peak (m, b) when --recoMuons is enabled.
  - Stores TCanvases and TH2Ds into a single output ROOT file following the exact dynamic
    TDirectory structure as scripts/generate_2DEventDisplays.py:
      * MCTruth/<flavor>/<current>/<mu_mult>/<acceptance>/<process>/<category> (in --mc-truth mode)
      * Custom hierarchy specified by --tdirectory
      * Run<####> (in real data mode)
  - Supports multi-core parallel processing (-j / --jobs).

Usage examples:
  # Generate Hough plots for a direct charm signal dataset:
  python3 scripts/generate_HoughPlots.py \\
      -i "/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/[1-10]/sndLHC.Genie-TGeant4_dig_2MuTrks_truth_charmDimuonDirect.root" \\
      -o ./plots/hough_plots_direct_signal.root \\
      --mc-truth \\
      --recoMuons \\
      -j 8

  # Specific events only:
  python3 scripts/generate_HoughPlots.py \\
      -i /eos/user/i/idioniso/snd-numu-charm/data/particle_gun_dimuon/sndLHC.TwoMuons-TGeant4_dig_2MuTrks.root \\
      -o ./plots/hough_plots_pgun.root \\
      --events 0 1 2 3 \\
      --recoMuons
"""

from __future__ import annotations

import os
import sys
import re
import time
import shutil
import tempfile
import argparse
import xml.etree.ElementTree as ET
from typing import List, Optional, Set, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.ndimage

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetPalette(ROOT.kViridis)

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from snd import (
    load_config,
    resolve_input_files,
    resolve_mctruth_directory_hierarchy,
    get_or_create_tdirectory,
    get_event_header_number,
    get_event_header_run_id,
    get_geofile_for_run,
    is_dimuon_in_ds_acceptance,
)

DEFAULT_GEOFILE = "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root"
DEFAULT_XML_CONFIG = os.path.join(PROJECT_ROOT, "config", "TrackingParams.xml")


# ---------------------------------------------------------------------------
# 1. XML Tracking Configuration Loader
# ---------------------------------------------------------------------------

def load_tracking_xml(
    xml_path: str,
    case_name: str = "dimuon_DS",
    h_format_name: str = "linearSlopeIntercept",
) -> Dict[str, Any]:
    """Parse tracking and Hough parameters from TrackingParams.xml."""
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Tracking XML configuration file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    target_case = None
    for tc in root.findall("tracking_case"):
        if tc.attrib.get("name") == case_name:
            target_case = tc
            break

    if target_case is None:
        available_cases = [tc.attrib.get("name") for tc in root.findall("tracking_case")]
        raise ValueError(f"Tracking case '{case_name}' not found in {xml_path}. Available: {available_cases}")

    # General tracking parameters
    params: Dict[str, Any] = {
        "case_name": case_name,
        "hits_to_fit": target_case.findtext("hits_to_fit", "ds").strip().lower(),
        "hits_to_validate": target_case.findtext("hits_to_validate", "ds").strip().lower(),
        "min_planes_hit": int(target_case.findtext("min_planes_hit", "3").strip()),
        "max_reco_muons": int(target_case.findtext("max_reco_muons", "2").strip()),
        "max_angle": float(target_case.findtext("max_angle", "1.0").strip()),
        "tolerance": float(target_case.findtext("tolerance", "0.4").strip()),
        "smooth_full": bool(int(target_case.findtext("smooth_full", "1").strip())),
        "sigma": float(target_case.findtext("sigma", "1.0").strip()),
        "truncate": float(target_case.findtext("truncate", "2.0").strip()),
        "h_format": h_format_name,
        "z_offset": 430.0,
        "angle_cut": 0.175,  # 175 mrad physical acceptance
    }

    # Optional plane crossing requirement (e.g. z_plane and area window in dimuon_DS)
    z_plane_node = target_case.find("z_plane")
    if z_plane_node is None:
        z_plane_node = target_case.find("z_sf5")
    x_min_node = target_case.find("x_area_min")
    x_max_node = target_case.find("x_area_max")
    y_min_node = target_case.find("y_area_min")
    y_max_node = target_case.find("y_area_max")

    has_crossing_plane = (
        z_plane_node is not None
        and x_min_node is not None
        and x_max_node is not None
        and y_min_node is not None
        and y_max_node is not None
    )

    if has_crossing_plane:
        z_plane = float(z_plane_node.text.strip())
        target_window_xz = (float(x_min_node.text.strip()), float(x_max_node.text.strip()))
        target_window_yz = (float(y_min_node.text.strip()), float(y_max_node.text.strip()))
    else:
        z_plane = None
        target_window_xz = None
        target_window_yz = None

    params["has_crossing_plane"] = has_crossing_plane
    params["z_plane"] = z_plane
    params["z_target"] = z_plane
    params["target_window_xz"] = target_window_xz
    params["target_window_yz"] = target_window_yz

    # Format-specific binning parameters
    fmt_elem = None
    for hf in target_case.findall("Hough_space_format"):
        if hf.attrib.get("name") == h_format_name:
            fmt_elem = hf
            break

    if fmt_elem is not None:
        params["n_yH"] = int(fmt_elem.findtext("N_yH_bins", "335").strip())
        params["n_xH"] = int(fmt_elem.findtext("N_xH_bins", "605").strip())
        params["yH_range_xz"] = (
            float(fmt_elem.findtext("yH_min_xz", "-105.0").strip()),
            float(fmt_elem.findtext("yH_max_xz", "17.0").strip()),
        )
        params["yH_range_yz"] = (
            float(fmt_elem.findtext("yH_min_yz", "0.0").strip()),
            float(fmt_elem.findtext("yH_max_yz", "122.0").strip()),
        )
        params["xH_range_xz"] = (
            float(fmt_elem.findtext("xH_min_xz", "-1.1").strip()),
            float(fmt_elem.findtext("xH_max_xz", "1.1").strip()),
        )
        params["xH_range_yz"] = (
            float(fmt_elem.findtext("xH_min_yz", "-1.1").strip()),
            float(fmt_elem.findtext("xH_max_yz", "1.1").strip()),
        )
    else:
        # Fallbacks for linearSlopeIntercept
        params["n_yH"] = 335
        params["n_xH"] = 605
        params["yH_range_xz"] = (-105.0, 17.0)
        params["yH_range_yz"] = (0.0, 122.0)
        params["xH_range_xz"] = (-1.1, 1.1)
        params["xH_range_yz"] = (-1.1, 1.1)

    return params


# ---------------------------------------------------------------------------
# 2. Geometry Caching for MuFilter Hits
# ---------------------------------------------------------------------------

_BAR_GEO_CACHE: Dict[str, Dict[int, Tuple[float, float, float, float, float, float, bool]]] = {}


def load_bar_geometry(geofile_path: str) -> Dict[int, Tuple[float, float, float, float, float, float, bool]]:
    """Cache bar positions and dimensions for all Downstream (DS) MuFilter bars."""
    if geofile_path in _BAR_GEO_CACHE:
        return _BAR_GEO_CACHE[geofile_path]

    import SndlhcGeo
    geo = SndlhcGeo.GeoInterface(geofile_path)
    mf_mod = geo.modules.get("MuFilter")
    if not mf_mod:
        _BAR_GEO_CACHE[geofile_path] = {}
        return _BAR_GEO_CACHE[geofile_path]

    a = ROOT.TVector3()
    b = ROOT.TVector3()
    bar_geo = {}
    # In Downstream (DS, System 3):
    # Planes 0, 1, 2 have 60 horizontal bars (0..59) and 60 vertical bars (60..119)
    # Plane 3 has only 60 vertical bars (60..119) and NO horizontal bars
    for plane in range(4):
        if plane < 3:
            for bar in range(60):
                det_id = 30000 + plane * 1000 + bar
                try:
                    mf_mod.GetPosition(det_id, a, b)
                    center = (a + b) * 0.5
                    dx = 6.0
                    dy = abs(a.Y() - b.Y()) if abs(a.Y() - b.Y()) > 0.1 else 6.0
                    dz = 6.0
                    bar_geo[det_id] = (center.X(), center.Y(), center.Z(), dx, dy, dz, False)
                except Exception:
                    pass

        for bar in range(60, 120):
            det_id = 30000 + plane * 1000 + bar
            try:
                mf_mod.GetPosition(det_id, a, b)
                center = (a + b) * 0.5
                dx = abs(a.X() - b.X()) if abs(a.X() - b.X()) > 0.1 else 6.0
                dy = 6.0
                dz = 6.0
                bar_geo[det_id] = (center.X(), center.Y(), center.Z(), dx, dy, dz, True)
            except Exception:
                pass

    _BAR_GEO_CACHE[geofile_path] = bar_geo
    return bar_geo


# ---------------------------------------------------------------------------
# 3. Vectorized Hough Accumulator Calculation
# ---------------------------------------------------------------------------

def compute_hough_accumulator_arrays(
    pts: np.ndarray,
    n_xH: int,
    n_yH: int,
    xH_range: Tuple[float, float],
    yH_range: Tuple[float, float],
    z_offset: float = 430.0,
    smooth_full: bool = True,
    sigma: float = 1.0,
    truncate: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes (acc_raw, acc_smooth, xH_bins, yH_bins) for hit points in (z, coord).
    Uses linearSlopeIntercept: yH = coord - (z - z_offset) * slope.
    """
    xH_bins = np.linspace(xH_range[0], xH_range[1], n_xH)
    yH_bins = np.linspace(yH_range[0], yH_range[1], n_yH)
    acc_raw = np.zeros((n_yH, n_xH), dtype=np.float32)

    if len(pts) == 0:
        return acc_raw, acc_raw.copy(), xH_bins, yH_bins

    shifted_z = pts[:, 0] - z_offset
    y_vals = pts[:, 1]
    hit_yH = y_vals[:, None] - shifted_z[:, None] * xH_bins[None, :]
    valid = (hit_yH >= yH_range[0]) & (hit_yH <= yH_range[1])

    if np.any(valid):
        idx_y = np.floor((hit_yH - yH_range[0]) / (yH_range[1] - yH_range[0]) * n_yH).astype(int)
        idx_y = np.clip(idx_y, 0, n_yH - 1)
        col_idx = np.broadcast_to(np.arange(n_xH), hit_yH.shape)
        np.add.at(acc_raw, (idx_y[valid], col_idx[valid]), 1.0)

    if smooth_full and sigma > 0:
        acc_smooth = scipy.ndimage.gaussian_filter(acc_raw, sigma, truncate=truncate)
    else:
        acc_smooth = acc_raw.copy()

    return acc_raw, acc_smooth, xH_bins, yH_bins


# ---------------------------------------------------------------------------
# 4. Canvas Creation & Overlay Drawing
# ---------------------------------------------------------------------------

def create_hough_plot_canvas(
    event_idx: int,
    event_number: int,
    run_number: int,
    tree: Any,
    bar_geo: Dict[int, Any],
    params: Dict[str, Any],
    canvas_name: str,
    canvas_title: str,
    show_reco_tracks: bool = True,
    use_smoothed_for_canvas: bool = True,
) -> Tuple[ROOT.TCanvas, ROOT.TH2D, ROOT.TH2D]:
    """
    Builds the interactive TCanvas displaying the binned 2D Hough space for XZ and YZ
    with fiducial and angular overlays, returning (canvas, h_xz, h_yz).
    """
    z_offset = params["z_offset"]
    has_crossing_plane = params.get("has_crossing_plane", False)
    z_target = params.get("z_plane") or params.get("z_sf5") or params.get("z_target")
    dz_target = (z_target - z_offset) if z_target is not None else None
    angle_cut = params["angle_cut"]
    tgt_xz = params.get("target_window_xz")
    tgt_yz = params.get("target_window_yz")

    # Extract DS hits
    pts_xz = []
    pts_yz = []
    mufi_hits = getattr(tree, "Digi_MuFilterHits", None)
    if mufi_hits:
        for hit in mufi_hits:
            if not hit.isValid():
                continue
            det_id = hit.GetDetectorID()
            if (det_id // 10000) != 3:
                continue
            geo_info = bar_geo.get(det_id)
            if not geo_info:
                continue
            cx, cy, cz, dx, dy, dz, is_vert = geo_info
            if is_vert:
                pts_xz.append((cz, cx))
            else:
                pts_yz.append((cz, cy))

    arr_xz = np.array(pts_xz, dtype=np.float32) if pts_xz else np.empty((0, 2), dtype=np.float32)
    arr_yz = np.array(pts_yz, dtype=np.float32) if pts_yz else np.empty((0, 2), dtype=np.float32)

    # Compute accumulators
    acc_xz_raw, acc_xz_smooth, x_bins_xz, y_bins_xz = compute_hough_accumulator_arrays(
        arr_xz,
        n_xH=params["n_xH"],
        n_yH=params["n_yH"],
        xH_range=params["xH_range_xz"],
        yH_range=params["yH_range_xz"],
        z_offset=z_offset,
        smooth_full=params["smooth_full"],
        sigma=params["sigma"],
        truncate=params["truncate"],
    )
    acc_yz_raw, acc_yz_smooth, x_bins_yz, y_bins_yz = compute_hough_accumulator_arrays(
        arr_yz,
        n_xH=params["n_xH"],
        n_yH=params["n_yH"],
        xH_range=params["xH_range_yz"],
        yH_range=params["yH_range_yz"],
        z_offset=z_offset,
        smooth_full=params["smooth_full"],
        sigma=params["sigma"],
        truncate=params["truncate"],
    )

    disp_xz = (acc_xz_smooth if use_smoothed_for_canvas else acc_xz_raw).copy()
    disp_yz = (acc_yz_smooth if use_smoothed_for_canvas else acc_yz_raw).copy()

    # If z_plane option is active, mask all accumulator bins outside the working region to zero
    if has_crossing_plane and z_target is not None and dz_target is not None:
        if tgt_xz is not None:
            M_xz, B_xz = np.meshgrid(x_bins_xz, y_bins_xz)
            pos_at_plane_xz = B_xz + M_xz * dz_target
            mask_xz = (
                (np.abs(M_xz) <= angle_cut)
                & (pos_at_plane_xz >= tgt_xz[0])
                & (pos_at_plane_xz <= tgt_xz[1])
            )
            disp_xz[~mask_xz] = 0.0

        if tgt_yz is not None:
            M_yz, B_yz = np.meshgrid(x_bins_yz, y_bins_yz)
            pos_at_plane_yz = B_yz + M_yz * dz_target
            mask_yz = (
                (np.abs(M_yz) <= angle_cut)
                & (pos_at_plane_yz >= tgt_yz[0])
                & (pos_at_plane_yz <= tgt_yz[1])
            )
            disp_yz[~mask_yz] = 0.0

    # Book TH2D Histograms
    h_name_xz = f"h_hough_xz_{run_number}_{event_number}"
    h_name_yz = f"h_hough_yz_{run_number}_{event_number}"
    h_xz = ROOT.TH2D(
        h_name_xz,
        f"ZX Hough Space - Run {run_number} Event {event_number};Slope m_{{zx}} = dx/dz [rad];Intercept at DS start b_{{zx}} [cm];Votes in Bin",
        params["n_xH"], params["xH_range_xz"][0], params["xH_range_xz"][1],
        params["n_yH"], params["yH_range_xz"][0], params["yH_range_xz"][1],
    )
    h_yz = ROOT.TH2D(
        h_name_yz,
        f"ZY Hough Space - Run {run_number} Event {event_number};Slope m_{{zy}} = dy/dz [rad];Intercept at DS start b_{{zy}} [cm];Votes in Bin",
        params["n_xH"], params["xH_range_yz"][0], params["xH_range_yz"][1],
        params["n_yH"], params["yH_range_yz"][0], params["yH_range_yz"][1],
    )

    for h in [h_xz, h_yz]:
        h.SetDirectory(0)
        h.SetStats(False)

    # Fill TH2D bin contents
    for iy in range(params["n_yH"]):
        for ix in range(params["n_xH"]):
            v_xz = float(disp_xz[iy, ix])
            if v_xz > 0:
                h_xz.SetBinContent(ix + 1, iy + 1, v_xz)
            v_yz = float(disp_yz[iy, ix])
            if v_yz > 0:
                h_yz.SetBinContent(ix + 1, iy + 1, v_yz)

    # Create TCanvas with two sub-pads
    canvas = ROOT.TCanvas(canvas_name, canvas_title, 1600, 750)
    canvas.Divide(2, 1, 0.005, 0.005)

    decorations = []

    # Reconstructed tracks extraction (if available)
    reco_tracks: List[Dict[str, float]] = []
    if show_reco_tracks and hasattr(tree, "Reco_MuonTracks") and tree.Reco_MuonTracks is not None:
        try:
            n_trks = tree.Reco_MuonTracks.GetEntries()
        except Exception:
            n_trks = len(tree.Reco_MuonTracks) if hasattr(tree.Reco_MuonTracks, "__len__") else 0
        for i_trk in range(n_trks):
            trk = tree.Reco_MuonTracks[i_trk]
            if not trk:
                continue
            if hasattr(trk, "getTrackFlag") and not trk.getTrackFlag():
                continue
            if hasattr(trk, "getSlopeXZ") and hasattr(trk, "getSlopeYZ") and hasattr(trk, "getStart"):
                tx = float(trk.getSlopeXZ())
                ty = float(trk.getSlopeYZ())
                st = trk.getStart()
                bx0 = float(st.X() - tx * st.Z())
                by0 = float(st.Y() - ty * st.Z())
                b_ds_x = bx0 + tx * z_offset
                b_ds_y = by0 + ty * z_offset
                reco_tracks.append({"tx": tx, "ty": ty, "b_ds_x": b_ds_x, "b_ds_y": b_ds_y})

    track_colors = [ROOT.kRed + 1, ROOT.kMagenta + 1, ROOT.kGreen + 2, ROOT.kOrange + 7]

    # --- Panel 1: ZX Hough Space ---
    pad1 = canvas.cd(1)
    pad1.SetLeftMargin(0.12)
    pad1.SetRightMargin(0.15)
    pad1.SetBottomMargin(0.12)
    pad1.SetTopMargin(0.08)
    h_xz.Draw("COLZ")

    # Shaded band for physical angular cut (|theta| <= angle_cut)
    box_xz = ROOT.TBox(-angle_cut, params["yH_range_xz"][0], angle_cut, params["yH_range_xz"][1])
    box_xz.SetFillColorAlpha(ROOT.kCyan - 8, 0.18)
    box_xz.SetLineColor(ROOT.kCyan + 2)
    box_xz.SetLineStyle(2)
    box_xz.SetLineWidth(2)
    box_xz.Draw("same")
    decorations.append(box_xz)

    # Angle edge lines
    l_ang_m = ROOT.TLine(-angle_cut, params["yH_range_xz"][0], -angle_cut, params["yH_range_xz"][1])
    l_ang_p = ROOT.TLine(angle_cut, params["yH_range_xz"][0], angle_cut, params["yH_range_xz"][1])
    for l in [l_ang_m, l_ang_p]:
        l.SetLineColor(ROOT.kCyan + 2)
        l.SetLineWidth(2)
        l.SetLineStyle(3)
        l.Draw("same")
        decorations.append(l)

    # Target boundary lines at plane z = z_plane: b = x_tgt - m * dz_target
    l_tgt_xz1 = None
    if has_crossing_plane and z_target is not None and tgt_xz is not None and dz_target is not None:
        m_min = params["xH_range_xz"][0]
        m_max = params["xH_range_xz"][1]
        b_min1 = tgt_xz[0] - m_min * dz_target
        b_min2 = tgt_xz[0] - m_max * dz_target
        b_max1 = tgt_xz[1] - m_min * dz_target
        b_max2 = tgt_xz[1] - m_max * dz_target

        l_tgt_xz1 = ROOT.TLine(m_min, b_min1, m_max, b_min2)
        l_tgt_xz2 = ROOT.TLine(m_min, b_max1, m_max, b_max2)
        for l, ls in [(l_tgt_xz1, 7), (l_tgt_xz2, 9)]:
            l.SetLineColor(ROOT.kRed + 1)
            l.SetLineWidth(2)
            l.SetLineStyle(ls)
            l.Draw("same")
            decorations.append(l)

    # Reconstructed track peaks
    for idx_trk, trk_info in enumerate(reco_tracks):
        mkr = ROOT.TMarker(trk_info["tx"], trk_info["b_ds_x"], 29)
        mkr.SetMarkerColor(track_colors[idx_trk % len(track_colors)])
        mkr.SetMarkerSize(2.2)
        mkr.Draw("same")
        decorations.append(mkr)

    # Legend for Pad 1
    leg1 = ROOT.TLegend(0.14, 0.76, 0.58, 0.90)
    leg1.SetBorderSize(1)
    leg1.SetFillColorAlpha(ROOT.kWhite, 0.85)
    leg1.SetTextFont(42)
    leg1.SetTextSize(0.026)
    leg1.AddEntry(box_xz, f"Angular Acc (|#theta| #leq {angle_cut:.3f} rad)", "f")
    if l_tgt_xz1 is not None and z_target is not None:
        leg1.AddEntry(l_tgt_xz1, f"Plane z={z_target:.0f} Bounds", "l")
    for idx_trk in range(len(reco_tracks)):
        leg1.AddEntry(decorations[-len(reco_tracks) + idx_trk], f"Reco Track {idx_trk+1}", "p")
    leg1.Draw("same")
    decorations.append(leg1)

    # --- Panel 2: ZY Hough Space ---
    pad2 = canvas.cd(2)
    pad2.SetLeftMargin(0.12)
    pad2.SetRightMargin(0.15)
    pad2.SetBottomMargin(0.12)
    pad2.SetTopMargin(0.08)
    h_yz.Draw("COLZ")

    # Shaded band for physical angular cut (|theta| <= angle_cut)
    box_yz = ROOT.TBox(-angle_cut, params["yH_range_yz"][0], angle_cut, params["yH_range_yz"][1])
    box_yz.SetFillColorAlpha(ROOT.kCyan - 8, 0.18)
    box_yz.SetLineColor(ROOT.kCyan + 2)
    box_yz.SetLineStyle(2)
    box_yz.SetLineWidth(2)
    box_yz.Draw("same")
    decorations.append(box_yz)

    l_ang_my = ROOT.TLine(-angle_cut, params["yH_range_yz"][0], -angle_cut, params["yH_range_yz"][1])
    l_ang_py = ROOT.TLine(angle_cut, params["yH_range_yz"][0], angle_cut, params["yH_range_yz"][1])
    for l in [l_ang_my, l_ang_py]:
        l.SetLineColor(ROOT.kCyan + 2)
        l.SetLineWidth(2)
        l.SetLineStyle(3)
        l.Draw("same")
        decorations.append(l)

    # Target boundary lines at plane z = z_plane: b = y_tgt - m * dz_target
    l_tgt_yz1 = None
    if has_crossing_plane and z_target is not None and tgt_yz is not None and dz_target is not None:
        m_min_y = params["xH_range_yz"][0]
        m_max_y = params["xH_range_yz"][1]
        by_min1 = tgt_yz[0] - m_min_y * dz_target
        by_min2 = tgt_yz[0] - m_max_y * dz_target
        by_max1 = tgt_yz[1] - m_min_y * dz_target
        by_max2 = tgt_yz[1] - m_max_y * dz_target

        l_tgt_yz1 = ROOT.TLine(m_min_y, by_min1, m_max_y, by_min2)
        l_tgt_yz2 = ROOT.TLine(m_min_y, by_max1, m_max_y, by_max2)
        for l, ls in [(l_tgt_yz1, 7), (l_tgt_yz2, 9)]:
            l.SetLineColor(ROOT.kRed + 1)
            l.SetLineWidth(2)
            l.SetLineStyle(ls)
            l.Draw("same")
            decorations.append(l)

    # Reconstructed track peaks
    for idx_trk, trk_info in enumerate(reco_tracks):
        mkr = ROOT.TMarker(trk_info["ty"], trk_info["b_ds_y"], 29)
        mkr.SetMarkerColor(track_colors[idx_trk % len(track_colors)])
        mkr.SetMarkerSize(2.2)
        mkr.Draw("same")
        decorations.append(mkr)

    # Legend for Pad 2
    leg2 = ROOT.TLegend(0.14, 0.76, 0.58, 0.90)
    leg2.SetBorderSize(1)
    leg2.SetFillColorAlpha(ROOT.kWhite, 0.85)
    leg2.SetTextFont(42)
    leg2.SetTextSize(0.026)
    leg2.AddEntry(box_yz, f"Angular Acc (|#theta| #leq {angle_cut:.3f} rad)", "f")
    if l_tgt_yz1 is not None and z_target is not None:
        leg2.AddEntry(l_tgt_yz1, f"Plane z={z_target:.0f} Bounds", "l")
    for idx_trk in range(len(reco_tracks)):
        leg2.AddEntry(decorations[-len(reco_tracks) + idx_trk], f"Reco Track {idx_trk+1}", "p")
    leg2.Draw("same")
    decorations.append(leg2)

    # Prevent garbage collection & double-free conflicts with ROOT C++ destructor
    ROOT.gROOT.GetListOfCanvases().Remove(canvas)
    ROOT.SetOwnership(canvas, False)
    for obj in [h_xz, h_yz] + decorations:
        ROOT.SetOwnership(obj, False)

    return canvas, h_xz, h_yz


# ---------------------------------------------------------------------------
# 5. Recursive Object Merger for Temporary Worker Files
# ---------------------------------------------------------------------------

def copy_objects_recursive(src_dir, dest_dir) -> None:
    """Recursively copies all TCanvases and TH2Ds from src_dir to dest_dir."""
    for key in src_dir.GetListOfKeys():
        cls_name = key.GetClassName()
        if cls_name in ("TCanvas", "TH2D", "TH2F", "TH1D", "TH1F"):
            obj = key.ReadObj()
            if obj and not (hasattr(obj, "IsZombie") and obj.IsZombie()):
                dest_dir.cd()
                ROOT.TColor.DefinedColors(1)
                obj.Write(key.GetName(), ROOT.TObject.kOverwrite)
                if cls_name == "TCanvas":
                    ROOT.gROOT.GetListOfCanvases().Remove(obj)
                    ROOT.SetOwnership(obj, False)
        elif "TDirectory" in cls_name:
            sub_src = key.ReadObj()
            if sub_src:
                sub_name = key.GetName()
                sub_dest = dest_dir.GetDirectory(sub_name)
                if not sub_dest:
                    sub_dest = dest_dir.mkdir(sub_name)
                copy_objects_recursive(sub_src, sub_dest)


# ---------------------------------------------------------------------------
# 6. Worker Execution Engine
# ---------------------------------------------------------------------------

def process_single_file_worker(args_tuple):
    """Worker function executed in parallel for one input ROOT file."""
    (
        input_path,
        run_num,
        geofile_to_use,
        temp_dir,
        canvas_name_format,
        mc_truth,
        events_filter,
        candidates_only,
        base_tdirectory_parts,
        tree_name,
        params,
        save_images,
        images_dir,
        split_acceptance,
        min_ds_hor_points,
        min_ds_ver_points,
        in_acc_name,
        other_name,
        reco_muons,
        use_smoothed,
    ) = args_tuple

    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetPalette(ROOT.kViridis)

    fname = os.path.basename(input_path)
    result = {
        "file": fname,
        "run_num": run_num,
        "total_entries": 0,
        "saved_displays": 0,
        "saved_in_acceptance": 0,
        "saved_other": 0,
        "temp_root": None,
        "canvas_names": [],
        "status": "OK",
        "error": None,
    }

    try:
        fin = ROOT.TFile.Open(input_path, "READ")
        if not fin or fin.IsZombie():
            result["status"] = "ERROR"
            result["error"] = f"Could not open input file {input_path}"
            return result

        tree = None
        if tree_name:
            tree = fin.Get(tree_name)
        else:
            for cand in ["cbmsim", "rawConv", "sndLHC", "Events"]:
                t = fin.Get(cand)
                if t and isinstance(t, ROOT.TTree):
                    tree = t
                    break

        if not tree:
            fin.Close()
            result["status"] = "ERROR"
            result["error"] = f"No valid TTree found in {input_path}"
            return result

        truth_tree = fin.Get("mctruth") or fin.Get("mc_truth") or fin.Get("truth")
        if truth_tree and not hasattr(tree, "nu_pdg"):
            tree.AddFriend(truth_tree.GetName())

        total_entries = tree.GetEntries()
        result["total_entries"] = total_entries

        # Filter events
        target_events_set: Optional[Set[int]] = set(events_filter) if events_filter is not None else None
        selected_entries = []

        if target_events_set is not None:
            if total_entries == 0:
                fin.Close()
                return result

            tree.GetEntry(0)
            first_ev = get_event_header_number(tree, default_idx=0)
            tree.GetEntry(total_entries - 1)
            last_ev = get_event_header_number(tree, default_idx=total_entries - 1)
            min_ev = min(first_ev, last_ev)
            max_ev = max(first_ev, last_ev)

            has_index_match = any(0 <= t < total_entries for t in target_events_set)
            has_ev_match = any(min_ev <= t <= max_ev for t in target_events_set)
            if not has_index_match and not has_ev_match:
                fin.Close()
                return result

            for iev in range(total_entries):
                tree.GetEntry(iev)
                ev_num = get_event_header_number(tree, default_idx=iev)
                if iev in target_events_set or ev_num in target_events_set:
                    if candidates_only and mc_truth:
                        if hasattr(tree, "has_candidate") and not bool(tree.has_candidate):
                            continue
                    selected_entries.append((iev, ev_num))
        else:
            for iev in range(total_entries):
                tree.GetEntry(iev)
                ev_num = get_event_header_number(tree, default_idx=iev)
                if candidates_only and mc_truth:
                    if hasattr(tree, "has_candidate") and not bool(tree.has_candidate):
                        continue
                selected_entries.append((iev, ev_num))

        if not selected_entries:
            fin.Close()
            return result

        # Load geometry
        bar_geo = load_bar_geometry(geofile_to_use)

        # Create worker temporary ROOT file
        temp_root_path = os.path.join(temp_dir, f"hough_worker_{run_num}_{int(time.time()*1e6)%1000000}.root")
        ftemp = ROOT.TFile.Open(temp_root_path, "RECREATE")
        result["temp_root"] = temp_root_path

        truth_processor = None
        if mc_truth and not hasattr(tree, "nu_pdg"):
            try:
                from snd import build_processor
                truth_processor = build_processor({})
            except Exception:
                truth_processor = None

        for iev, event_num in selected_entries:
            tree.GetEntry(iev)
            ev_run_id = get_event_header_run_id(tree, default_run=run_num)

            # Determine TDirectory hierarchy matching generate_2DEventDisplays.py
            if base_tdirectory_parts is not None:
                base_parts = list(base_tdirectory_parts)
                if mc_truth and split_acceptance:
                    is_in_acceptance = is_dimuon_in_ds_acceptance(
                        tree,
                        min_hor_points=min_ds_hor_points,
                        min_ver_points=min_ds_ver_points,
                    )
                    category = in_acc_name if is_in_acceptance else other_name
                    target_parts = base_parts + [category]
                else:
                    is_in_acceptance = False
                    target_parts = base_parts
            elif mc_truth:
                target_parts = resolve_mctruth_directory_hierarchy(
                    tree,
                    processor=truth_processor,
                    min_ds_hor_points=min_ds_hor_points,
                    min_ds_ver_points=min_ds_ver_points,
                    top_name="MCTruth",
                    in_acceptance_name=in_acc_name,
                    not_in_acceptance_name=other_name,
                )
                is_in_acceptance = (in_acc_name in target_parts)
                if not split_acceptance and len(target_parts) > 4:
                    del target_parts[4]
            else:
                run_dir_name = f"Run{ev_run_id:04d}" if isinstance(ev_run_id, int) and 0 <= ev_run_id < 10000 else f"Run{ev_run_id}"
                target_parts = [run_dir_name]
                is_in_acceptance = False

            # Partition / Run identifier
            if mc_truth and (ev_run_id <= 1 or ev_run_id == run_num):
                run_label = f"part{run_num}" if run_num != 0 else f"run{ev_run_id}"
            else:
                run_label = f"{ev_run_id}"

            canvas_name = canvas_name_format.format(run_num=run_label, event_num=event_num)
            if mc_truth:
                hierarchy_label = "/".join(target_parts[1:])
                canvas_title = f"SND@LHC Hough Space - {run_label} Event {event_num} ({hierarchy_label})"
            else:
                canvas_title = f"SND@LHC Hough Space - Run {ev_run_id} Event {event_num}"

            # Generate interactive TCanvas and histograms
            canvas, h_xz, h_yz = create_hough_plot_canvas(
                event_idx=iev,
                event_number=event_num,
                run_number=run_label,
                tree=tree,
                bar_geo=bar_geo,
                params=params,
                canvas_name=canvas_name,
                canvas_title=canvas_title,
                show_reco_tracks=reco_muons,
                use_smoothed_for_canvas=use_smoothed,
            )

            target_tdir = get_or_create_tdirectory(ftemp, target_parts)
            target_tdir.cd()
            ROOT.TColor.DefinedColors(1)
            canvas.Write(canvas_name)
            h_xz.Write(h_xz.GetName())
            h_yz.Write(h_yz.GetName())

            ROOT.gROOT.GetListOfCanvases().Remove(canvas)
            ROOT.SetOwnership(canvas, False)

            result["saved_displays"] += 1
            if mc_truth and split_acceptance:
                if is_in_acceptance:
                    result["saved_in_acceptance"] += 1
                else:
                    result["saved_other"] += 1
            result["canvas_names"].append((canvas_name, "/".join(target_parts)))

            if save_images:
                sub_img_dir = os.path.join(images_dir, *target_parts)
                os.makedirs(sub_img_dir, exist_ok=True)
                png_path = os.path.join(sub_img_dir, f"{canvas_name}.png")
                canvas.Print(png_path)

        ftemp.cd()
        ROOT.gROOT.GetListOfColors().Write("ListOfColors", ROOT.TObject.kSingleKey)
        ftemp.Write()
        ftemp.Close()
        fin.Close()
        return result

    except Exception as e:
        import traceback
        result["status"] = "ERROR"
        result["error"] = f"Exception in worker on {input_path}: {e}\n{traceback.format_exc()}"
        return result


# ---------------------------------------------------------------------------
# 7. Main CLI & Parallel Dispatcher
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate binned 2D Hough Space plots and histograms in ROOT format matching scripts/generate_2DEventDisplays.py logic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input-dir", "--input",
        type=str,
        default=None,
        help="Input dataset directory, file pattern, or single ROOT file",
    )
    parser.add_argument(
        "-o", "--output-file", "--output",
        type=str,
        default=None,
        help="Output single ROOT file containing all Hough space TCanvases & TH2Ds",
    )
    parser.add_argument(
        "-g", "--geofile",
        type=str,
        default=None,
        help="Custom detector geometry ROOT file (default: auto-detected)",
    )
    parser.add_argument(
        "--xml", "--tracking-config",
        type=str,
        default=DEFAULT_XML_CONFIG,
        help="Path to TrackingParams.xml tracking configuration",
    )
    parser.add_argument(
        "--case", "--tracking-case",
        type=str,
        default="dimuon_DS",
        help="Tracking case element inside the XML file",
    )
    parser.add_argument(
        "--hf", "--h-format", "--hough-format",
        type=str,
        default="linearSlopeIntercept",
        choices=["linearSlopeIntercept", "normal"],
        help="Hough representation format from the XML file",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--mc-truth",
        action="store_true",
        default=False,
        help="Enable Monte Carlo truth dynamic multi-tiered TDirectory hierarchy: MCTruth/<flavor>/<current>/<mu_mult>/<acceptance>/<process>/<category>",
    )
    parser.add_argument(
        "--recoMuons", "--reco-muons", "--muonReco", "--muon-reco",
        dest="reco_muons",
        action="store_true",
        default=True,
        help="Plot reconstructed muon tracks from Reco_MuonTracks branch as markers on the Hough space",
    )
    parser.add_argument(
        "--no-reco-muons",
        dest="reco_muons",
        action="store_false",
        help="Do not plot reconstructed muon tracks from Reco_MuonTracks",
    )
    parser.add_argument(
        "--smoothed",
        action="store_true",
        default=None,
        help="Display the smoothed Hough accumulator (default: enabled when smooth_full=1 in XML)",
    )
    parser.add_argument(
        "--raw",
        dest="smoothed",
        action="store_false",
        help="Force raw discrete vote counts instead of the smoothed accumulator",
    )
    parser.add_argument(
        "--events",
        nargs="+",
        type=int,
        default=None,
        help="Specific event numbers or entry indices to display (default: all events)",
    )
    parser.add_argument(
        "--tree-name",
        type=str,
        default=None,
        help="Explicit name of the TTree to read (auto-detects 'cbmsim' or 'rawConv' if omitted)",
    )
    parser.add_argument(
        "--tdirectory",
        type=str,
        default=None,
        help="Custom base TDirectory hierarchy (overrides dynamic Run<####> or MCTruth/... hierarchy)",
    )
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        default=False,
        help="Only display candidate events (has_candidate=1 in MC)",
    )
    parser.add_argument(
        "--min-ds-hor-points",
        type=int,
        default=3,
        help="Minimum number of horizontal plane DS MCPoints for muons to be in DSAcceptance",
    )
    parser.add_argument(
        "--min-ds-ver-points",
        type=int,
        default=3,
        help="Minimum number of vertical plane DS MCPoints for muons to be in DSAcceptance",
    )
    parser.add_argument(
        "--no-acceptance-split",
        action="store_true",
        help="Disable splitting into inDSAcceptance and notInDSAcceptance subdirectories",
    )
    parser.add_argument(
        "--in-acceptance-dir",
        type=str,
        default="inDSAcceptance",
        help="Subdirectory name for events in acceptance",
    )
    parser.add_argument(
        "--other-dir",
        type=str,
        default="other",
        help="Subdirectory name for events outside acceptance",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=False,
        help="Also export PNG images of the generated Hough plots",
    )
    parser.add_argument(
        "-u", "--update",
        action="store_true",
        default=False,
        help="Open output ROOT file in UPDATE mode to append into existing TDirectories without wiping existing files/canvases",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="./hough_images",
        help="Output directory for PNG images when --save-images is enabled",
    )

    args = parser.parse_args()

    # 1. Parse XML Tracking Configuration
    hough_params = load_tracking_xml(
        xml_path=args.xml,
        case_name=args.case,
        h_format_name=args.hf,
    )

    # 2. Resolve Input Files
    input_pattern = args.input_dir or "/eos/user/i/idioniso/snd-numu-charm/data"
    input_files = resolve_input_files(input_pattern)
    if not input_files:
        print(f"Error: No ROOT files found matching pattern: {input_pattern}")
        sys.exit(1)

    # 3. Output Configuration
    output_root = args.output_file or "./hough_space_plots.root"
    out_root_abs = os.path.abspath(output_root)
    os.makedirs(os.path.dirname(out_root_abs), exist_ok=True)

    base_tdir_parts = [p.strip() for p in args.tdirectory.split("/") if p.strip()] if args.tdirectory else None
    split_acceptance = not args.no_acceptance_split

    print("=" * 80)
    print(" SND@LHC Binned 2D Hough Space Generator")
    print("=" * 80)
    print(f" Input Files Found:     {len(input_files)}")
    print(f" Output ROOT File:      {out_root_abs}")
    print(f" XML Configuration:     {args.xml} (Case: '{args.case}', Format: '{args.hf}')")
    print(f" Accumulator Bins:      XZ: {hough_params['n_xH']}x{hough_params['n_yH']} bins in [{hough_params['xH_range_xz'][0]}, {hough_params['xH_range_xz'][1]}] rad x [{hough_params['yH_range_xz'][0]}, {hough_params['yH_range_xz'][1]}] cm")
    print(f"                        YZ: {hough_params['n_xH']}x{hough_params['n_yH']} bins in [{hough_params['xH_range_yz'][0]}, {hough_params['xH_range_yz'][1]}] rad x [{hough_params['yH_range_yz'][0]}, {hough_params['yH_range_yz'][1]}] cm")
    use_smoothed = hough_params.get("smooth_full", True) if args.smoothed is None else bool(args.smoothed)
    print(f" Accumulator Display:   {'Smoothed (Gaussian smeared)' if use_smoothed else 'Raw (Unsmoothed discrete votes)'}")
    print(f" Smoothing Parameters:  Full={hough_params['smooth_full']} (sigma={hough_params['sigma']}, truncate={hough_params['truncate']})")
    if hough_params.get("has_crossing_plane"):
        print(f" Working Area Window:   Plane z_plane={hough_params.get('z_plane', hough_params.get('z_sf5'))} cm (x in {hough_params['target_window_xz']}, y in {hough_params['target_window_yz']})")
    else:
        print(f" Working Area Window:   Disabled (Not configured for case '{args.case}')")
    print(f" Angular Acceptance:    Shaded |theta| <= {hough_params['angle_cut']} rad (175 mrad)")
    print(f" Reco Muon Tracks:      {'Enabled' if args.reco_muons else 'Disabled'}")
    print(f" Events Filter:         {args.events if args.events else 'All events'}")
    print(f" Parallel Workers:      {args.jobs}")
    print(f" Save PNG Images:       {args.save_images}" + (f" -> {args.images_dir}" if args.save_images else ""))
    print("=" * 80)

    # Temporary directory for worker files
    temp_dir = tempfile.mkdtemp(prefix="snd_hough_plots_")

    # Build worker arguments
    worker_tasks = []
    for fpath in input_files:
        run_num = 0
        match = re.search(r"run_?(\d+)", fpath, re.IGNORECASE)
        if match:
            run_num = int(match.group(1))
        else:
            match_p = re.search(r"/(\d+)/", fpath)
            if match_p:
                run_num = int(match_p.group(1))

        geofile_to_use = args.geofile
        if not geofile_to_use:
            geofile_to_use = get_geofile_for_run(run_num) or DEFAULT_GEOFILE
            if not os.path.exists(geofile_to_use):
                geofile_to_use = DEFAULT_GEOFILE

        worker_tasks.append((
            fpath,
            run_num,
            geofile_to_use,
            temp_dir,
            "c_hough_{run_num}_{event_num}",
            args.mc_truth,
            args.events,
            args.candidates_only,
            base_tdir_parts,
            args.tree_name,
            hough_params,
            args.save_images,
            args.images_dir,
            split_acceptance,
            args.min_ds_hor_points,
            args.min_ds_ver_points,
            args.in_acceptance_dir,
            args.other_dir,
            args.reco_muons,
            use_smoothed,
        ))

    total_saved = 0
    total_in_acc = 0
    total_other = 0
    total_processed = 0
    start_time = time.time()
    worker_results = []

    print(f"\nDispatching {len(worker_tasks)} file tasks across {args.jobs} parallel workers...\n")

    if args.jobs <= 1:
        for t in worker_tasks:
            res = process_single_file_worker(t)
            worker_results.append(res)
            total_processed += res.get("total_entries", 0)
            if res["status"] == "OK":
                print(f"  [DONE] {res['file']}: {res['saved_displays']} Hough plots generated ({res['total_entries']} entries)")
            else:
                print(f"  [ERROR] {res['file']}: {res['error']}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            future_to_file = {executor.submit(process_single_file_worker, t): t[0] for t in worker_tasks}
            for future in as_completed(future_to_file):
                orig_file = future_to_file[future]
                try:
                    res = future.result()
                    worker_results.append(res)
                    total_processed += res.get("total_entries", 0)
                    if res["status"] == "OK":
                        print(f"  [DONE] {res['file']}: {res['saved_displays']} Hough plots generated ({res['total_entries']} entries)")
                    else:
                        print(f"  [ERROR] {res['file']}: {res['error']}")
                except Exception as e:
                    print(f"  [CRASH] {os.path.basename(orig_file)}: worker exception: {e}")

    # Merge worker temporary ROOT files into final destination
    print(f"\nMerging worker ROOT files into final output: {out_root_abs} ...")
    fout_mode = "UPDATE" if args.update and os.path.exists(out_root_abs) else "RECREATE"
    fout = ROOT.TFile.Open(out_root_abs, fout_mode)

    for res in worker_results:
        if res["status"] != "OK":
            continue
        temp_root = res.get("temp_root")
        saved_in_file = res.get("saved_displays", 0)
        if saved_in_file > 0 and temp_root and os.path.exists(temp_root):
            ftemp = ROOT.TFile.Open(temp_root, "READ")
            if ftemp and not ftemp.IsZombie():
                copy_objects_recursive(ftemp, fout)
                ftemp.Close()

            try:
                os.remove(temp_root)
            except OSError:
                pass

            total_saved += saved_in_file
            total_in_acc += res["saved_in_acceptance"]
            total_other += res["saved_other"]

    fout.Close()

    # Clean up temp directory
    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f" Processing Complete in {elapsed:.1f} s")
    print(f" Total Entries Evaluated: {total_processed}")
    print(f" Total Hough Plots Saved: {total_saved}")
    if args.mc_truth and split_acceptance:
        print(f"   In DS Acceptance:      {total_in_acc}")
        print(f"   Outside DS Acceptance: {total_other}")
    print(f" Output File Written:     {out_root_abs}")
    print("=" * 80)


if __name__ == "__main__":
    main()
