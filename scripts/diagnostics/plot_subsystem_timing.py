#!/usr/bin/env python3
"""
scripts/diagnostics/plot_subsystem_timing.py
--------------------------------------------
Diagnostic script analyzing and visualizing timing observables across SND@LHC detector subsystems:
  - Veto (System 1)
  - SciFi (System 0 / ScifiHits)
  - Upstream Muon Filter / HCAL (US, System 2)
  - Downstream Muon Filter (DS, System 3)

Produces distributions of:
  1. Hit times since the beginning of the 25-ns readout window for all 4 subsystems.
  2. Earliest (min) and latest (max) hit times per event.
  3. Pairwise time differences between subsystems:
     - Earliest SciFi hit minus latest Veto hit: t(SciFi_min) - t(Veto_max)
     - Earliest SciFi hit minus earliest Veto hit: t(SciFi_min) - t(Veto_min)
     - Earliest US hit minus latest Veto hit: t(US_min) - t(Veto_max)
     - Earliest DS hit minus latest Veto hit: t(DS_min) - t(Veto_max)
     - Earliest US hit minus earliest SciFi hit: t(US_min) - t(SciFi_min)
     - Latest DS hit minus earliest SciFi hit: t(DS_max) - t(SciFi_min) [SciFi-DS Timing cut metric]
     - Earliest DS hit minus earliest US hit: t(DS_min) - t(US_min)
  4. 2D timing correlation histograms.

Saves structured ROOT TCanvases, TH1D/TH2D histograms into a ROOT file,
and exports publication-quality PNG and PDF figures.

Usage:
  # Run on MCEB partition 1:
  python3 scripts/diagnostics/plot_subsystem_timing.py -p 1

  # Run on partitions 1-10:
  python3 scripts/diagnostics/plot_subsystem_timing.py -p 1-10 -j 4 -o plots/timing_mceb

  # Run on custom file:
  python3 scripts/diagnostics/plot_subsystem_timing.py -i /path/to/file.root -o plots/timing_custom
"""

from __future__ import annotations

import os
import sys
import glob
import time
import argparse
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat("emruo")


def get_geofile_for_run(run_id: int, geo_csv: str = "config/geo_paths.csv") -> Optional[str]:
    """Look up official geofile path for a given run ID from configuration CSV."""
    if not os.path.isfile(geo_csv):
        return None
    with open(geo_csv, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
                min_r, max_r, path = int(parts[0]), int(parts[1]), parts[2].strip()
                if min_r <= run_id <= max_r:
                    # Resolve clean local POSIX path if accessible
                    local_path = path
                    if "://" in path:
                        # strip scheme://host/
                        after_host = path.split("://", 1)[1].split("/", 1)[1]
                        local_path = "/" + after_host.lstrip("/")
                    if os.path.exists(local_path):
                        return local_path
                    return path
    return None


def parse_partitions(part_arg: Optional[str]) -> List[int]:
    """Parse partition range or list."""
    if not part_arg or part_arg.strip().lower() == "all":
        return list(range(1, 1001))

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


def format_partition_path(pattern: str, partition: int | str) -> str:
    """Format partition pattern if placeholder present."""
    p_str = str(partition)
    count = pattern.count("%s")
    if count > 0:
        return pattern % tuple(p_str for _ in range(count))
    return pattern


def extract_event_timing(
    scifi_hits: Any,
    mufi_hits: Any,
    is_data: bool = False,
    tdc2ns: float = 6.23768,
    scifi_mod: Any = None,
    mufi_mod: Any = None,
) -> Dict[str, Any]:
    """Extract timing metrics for all 4 detector subsystems in an event."""
    sf_times = []
    if scifi_hits is not None:
        for i in range(scifi_hits.GetEntriesFast()):
            h = scifi_hits.At(i)
            if not h or not h.isValid():
                continue
            raw_t = float(h.GetTime())
            if is_data:
                raw_t *= tdc2ns
                if scifi_mod is not None:
                    raw_t = float(scifi_mod.GetCorrectedTime(int(h.GetDetectorID()), raw_t, 0.0))
            sf_times.append(raw_t)

    v_times = []
    us_times = []
    ds_times = []

    if mufi_hits is not None:
        for i in range(mufi_hits.GetEntriesFast()):
            h = mufi_hits.At(i)
            if not h or not h.isValid():
                continue
            sys_id = int(h.GetSystem())
            det_id = int(h.GetDetectorID())
            is_vert = bool(h.isVertical())

            t_val: Optional[float] = None
            if not is_data:
                t_val = float(h.GetTime(0)) if is_vert else float(0.5 * (h.GetTime(0) + h.GetTime(1)))
            else:
                t0_raw = float(h.GetTime(0))
                t1_raw = float(h.GetTime(1))
                has_t0 = t0_raw > -900.0
                has_t1 = t1_raw > -900.0

                if not has_t0 and not has_t1:
                    continue

                if sys_id == 3:  # Downstream (DS)
                    t0_cor = float(mufi_mod.GetCorrectedTime(det_id, 0, t0_raw * tdc2ns, 0.0)) if (has_t0 and mufi_mod) else (t0_raw * tdc2ns if has_t0 else None)
                    t1_cor = float(mufi_mod.GetCorrectedTime(det_id, 1, t1_raw * tdc2ns, 0.0)) if (has_t1 and mufi_mod) else (t1_raw * tdc2ns if has_t1 else None)
                    if is_vert:
                        t_val = t0_cor if t0_cor is not None else t1_cor
                    else:
                        if t0_cor is not None and t1_cor is not None:
                            t_val = 0.5 * (t0_cor + t1_cor)
                        elif t0_cor is not None:
                            t_val = t0_cor
                        else:
                            t_val = t1_cor
                else:  # Veto (1) or Upstream (2)
                    t0_ns = t0_raw * tdc2ns if has_t0 else None
                    t1_ns = t1_raw * tdc2ns if has_t1 else None
                    if is_vert:
                        t_val = t0_ns if t0_ns is not None else t1_ns
                    else:
                        if t0_ns is not None and t1_ns is not None:
                            t_val = 0.5 * (t0_ns + t1_ns)
                        elif t0_ns is not None:
                            t_val = t0_ns
                        else:
                            t_val = t1_ns

            if t_val is not None:
                if sys_id == 1:
                    v_times.append(t_val)
                elif sys_id == 2:
                    us_times.append(t_val)
                elif sys_id == 3:
                    ds_times.append(t_val)

    return {
        "all_sf": sf_times,
        "all_v": v_times,
        "all_us": us_times,
        "all_ds": ds_times,
        "min_sf": min(sf_times) if sf_times else None,
        "max_sf": max(sf_times) if sf_times else None,
        "min_v": min(v_times) if v_times else None,
        "max_v": max(v_times) if v_times else None,
        "min_us": min(us_times) if us_times else None,
        "max_us": max(us_times) if us_times else None,
        "min_ds": min(ds_times) if ds_times else None,
        "max_ds": max(ds_times) if ds_times else None,
    }


def process_single_file(
    filepath: str,
    max_events: int = -1,
    geofile: Optional[str] = None,
    geo_csv: str = "config/geo_paths.csv",
) -> Dict[str, Any]:
    """Process a single ROOT file and aggregate timing arrays."""
    data = {
        "all_v": [],
        "all_sf": [],
        "all_us": [],
        "all_ds": [],
        "min_v": [],
        "min_sf": [],
        "min_us": [],
        "min_ds": [],
        "max_v": [],
        "max_sf": [],
        "max_us": [],
        "max_ds": [],
        "dt_sf_earliest_veto_latest": [],
        "dt_sf_earliest_veto_earliest": [],
        "dt_us_earliest_veto_latest": [],
        "dt_ds_earliest_veto_latest": [],
        "dt_us_earliest_sf_earliest": [],
        "dt_ds_latest_sf_earliest": [],
        "dt_ds_earliest_sf_earliest": [],
        "dt_ds_earliest_us_earliest": [],
        "corr_sf_veto": [],
        "corr_ds_sf": [],
        "corr_us_sf": [],
        "corr_ds_us": [],
        "events_total": 0,
    }

    if not os.path.isfile(filepath):
        return data

    f_in = ROOT.TFile.Open(filepath, "READ")
    if not f_in or f_in.IsZombie():
        if f_in:
            f_in.Close()
        return data

    t_cbm = f_in.Get("rawConv") or f_in.Get("cbmsim")
    if not t_cbm:
        f_in.Close()
        return data

    is_data = (t_cbm.GetName() == "rawConv")
    t_cbm.SetBranchStatus("*", 0)
    t_cbm.SetBranchStatus("Digi_ScifiHits*", 1)
    t_cbm.SetBranchStatus("Digi_MuFilterHits*", 1)

    scifi_mod = None
    mufi_mod = None
    tdc2ns = 6.23768  # default SND TDC tick ~ 6.23768 ns (160.316 MHz)

    if is_data:
        t_cbm.SetBranchStatus("EventHeader*", 1)
        try:
            import shipunit as u
            tdc2ns = float(u.snd_TDC2ns)
        except Exception:
            pass

        t_cbm.GetEntry(0)
        run_id = 0
        if hasattr(t_cbm, "EventHeader") and t_cbm.EventHeader:
            run_id = int(t_cbm.EventHeader.GetRunId())

        geo_path = geofile
        if not geo_path and run_id > 0:
            geo_path = get_geofile_for_run(run_id, geo_csv)

        if geo_path:
            import SndlhcGeo
            geo = SndlhcGeo.GeoInterface(geo_path)
            scifi_mod = geo.modules.get("Scifi")
            mufi_mod = geo.modules.get("MuFilter")
            if scifi_mod and hasattr(t_cbm, "EventHeader"):
                scifi_mod.InitEvent(t_cbm.EventHeader)
            if mufi_mod and hasattr(t_cbm, "EventHeader"):
                mufi_mod.InitEvent(t_cbm.EventHeader)

    n_entries = t_cbm.GetEntries()
    limit = n_entries if max_events <= 0 else min(n_entries, max_events)
    data["events_total"] = limit

    for i in range(limit):
        t_cbm.GetEntry(i)
        ev = extract_event_timing(
            t_cbm.Digi_ScifiHits,
            t_cbm.Digi_MuFilterHits,
            is_data=is_data,
            tdc2ns=tdc2ns,
            scifi_mod=scifi_mod,
            mufi_mod=mufi_mod,
        )

        data["all_v"].extend(ev["all_v"])
        data["all_sf"].extend(ev["all_sf"])
        data["all_us"].extend(ev["all_us"])
        data["all_ds"].extend(ev["all_ds"])

        if ev["min_v"] is not None:
            data["min_v"].append(ev["min_v"])
            data["max_v"].append(ev["max_v"])
        if ev["min_sf"] is not None:
            data["min_sf"].append(ev["min_sf"])
            data["max_sf"].append(ev["max_sf"])
        if ev["min_us"] is not None:
            data["min_us"].append(ev["min_us"])
            data["max_us"].append(ev["max_us"])
        if ev["min_ds"] is not None:
            data["min_ds"].append(ev["min_ds"])
            data["max_ds"].append(ev["max_ds"])

        # Differences involving Veto
        if ev["min_sf"] is not None and ev["max_v"] is not None:
            dt = ev["min_sf"] - ev["max_v"]
            data["dt_sf_earliest_veto_latest"].append(dt)
            data["corr_sf_veto"].append((ev["max_v"], ev["min_sf"]))

        if ev["min_sf"] is not None and ev["min_v"] is not None:
            data["dt_sf_earliest_veto_earliest"].append(ev["min_sf"] - ev["min_v"])

        if ev["min_us"] is not None and ev["max_v"] is not None:
            data["dt_us_earliest_veto_latest"].append(ev["min_us"] - ev["max_v"])

        if ev["min_ds"] is not None and ev["max_v"] is not None:
            data["dt_ds_earliest_veto_latest"].append(ev["min_ds"] - ev["max_v"])

        # Differences involving SciFi
        if ev["min_us"] is not None and ev["min_sf"] is not None:
            data["dt_us_earliest_sf_earliest"].append(ev["min_us"] - ev["min_sf"])
            data["corr_us_sf"].append((ev["min_sf"], ev["min_us"]))

        if ev["max_ds"] is not None and ev["min_sf"] is not None:
            data["dt_ds_latest_sf_earliest"].append(ev["max_ds"] - ev["min_sf"])
            data["corr_ds_sf"].append((ev["min_sf"], ev["max_ds"]))

        if ev["min_ds"] is not None and ev["min_sf"] is not None:
            data["dt_ds_earliest_sf_earliest"].append(ev["min_ds"] - ev["min_sf"])

        # Differences between US and DS
        if ev["min_ds"] is not None and ev["min_us"] is not None:
            data["dt_ds_earliest_us_earliest"].append(ev["min_ds"] - ev["min_us"])
            data["corr_ds_us"].append((ev["min_us"], ev["max_ds"] if ev["max_ds"] is not None else ev["min_ds"]))

    f_in.Close()
    return data


def create_and_fill_histograms(
    aggregated_data: Dict[str, Any],
    window_max: float = 25.0,
) -> Dict[str, Any]:
    """Create and fill TH1D and TH2D timing histograms."""
    hists: Dict[str, Any] = {}

    # Window hit times (0 to window_max ns, default 25ns)
    n_bins_win = int(window_max * 2)  # 0.5 ns bins
    hists["h_time_veto"] = ROOT.TH1D("h_time_veto", "Veto Hit Times;Hit Time [ns];Hits / 0.5 ns", n_bins_win, 0, window_max)
    hists["h_time_scifi"] = ROOT.TH1D("h_time_scifi", "SciFi Hit Times;Hit Time [ns];Hits / 0.5 ns", n_bins_win, 0, window_max)
    hists["h_time_us"] = ROOT.TH1D("h_time_us", "US Hit Times;Hit Time [ns];Hits / 0.5 ns", n_bins_win, 0, window_max)
    hists["h_time_ds"] = ROOT.TH1D("h_time_ds", "DS Hit Times;Hit Time [ns];Hits / 0.5 ns", n_bins_win, 0, window_max)

    for t in aggregated_data["all_v"]:
        hists["h_time_veto"].Fill(t)
    for t in aggregated_data["all_sf"]:
        hists["h_time_scifi"].Fill(t)
    for t in aggregated_data["all_us"]:
        hists["h_time_us"].Fill(t)
    for t in aggregated_data["all_ds"]:
        hists["h_time_ds"].Fill(t)

    # Earliest (min) hit times per event
    hists["h_time_min_veto"] = ROOT.TH1D("h_time_min_veto", "Earliest Veto Hit Time per Event;Earliest Time [ns];Events / 0.5 ns", n_bins_win, 0, window_max)
    hists["h_time_min_scifi"] = ROOT.TH1D("h_time_min_scifi", "Earliest SciFi Hit Time per Event;Earliest Time [ns];Events / 0.5 ns", n_bins_win, 0, window_max)
    hists["h_time_min_us"] = ROOT.TH1D("h_time_min_us", "Earliest US Hit Time per Event;Earliest Time [ns];Events / 0.5 ns", n_bins_win, 0, window_max)
    hists["h_time_min_ds"] = ROOT.TH1D("h_time_min_ds", "Earliest DS Hit Time per Event;Earliest Time [ns];Events / 0.5 ns", n_bins_win, 0, window_max)

    for t in aggregated_data["min_v"]:
        hists["h_time_min_veto"].Fill(t)
    for t in aggregated_data["min_sf"]:
        hists["h_time_min_scifi"].Fill(t)
    for t in aggregated_data["min_us"]:
        hists["h_time_min_us"].Fill(t)
    for t in aggregated_data["min_ds"]:
        hists["h_time_min_ds"].Fill(t)

    # Time Differences
    dt_bins = 120
    dt_min = -30.0
    dt_max = 30.0

    hists["h_dt_scifi_earliest_minus_veto_latest"] = ROOT.TH1D(
        "h_dt_scifi_earliest_minus_veto_latest",
        "t_{SciFi, earliest} - t_{Veto, latest};t_{SciFi, earliest} - t_{Veto, latest} [ns];Events / 0.5 ns",
        dt_bins, dt_min, dt_max
    )
    for val in aggregated_data["dt_sf_earliest_veto_latest"]:
        hists["h_dt_scifi_earliest_minus_veto_latest"].Fill(val)

    hists["h_dt_scifi_earliest_minus_veto_earliest"] = ROOT.TH1D(
        "h_dt_scifi_earliest_minus_veto_earliest",
        "t_{SciFi, earliest} - t_{Veto, earliest};t_{SciFi, earliest} - t_{Veto, earliest} [ns];Events / 0.5 ns",
        dt_bins, dt_min, dt_max
    )
    for val in aggregated_data["dt_sf_earliest_veto_earliest"]:
        hists["h_dt_scifi_earliest_minus_veto_earliest"].Fill(val)

    hists["h_dt_us_earliest_minus_veto_latest"] = ROOT.TH1D(
        "h_dt_us_earliest_minus_veto_latest",
        "t_{US, earliest} - t_{Veto, latest};t_{US, earliest} - t_{Veto, latest} [ns];Events / 0.5 ns",
        dt_bins, dt_min, dt_max
    )
    for val in aggregated_data["dt_us_earliest_veto_latest"]:
        hists["h_dt_us_earliest_minus_veto_latest"].Fill(val)

    hists["h_dt_ds_earliest_minus_veto_latest"] = ROOT.TH1D(
        "h_dt_ds_earliest_minus_veto_latest",
        "t_{DS, earliest} - t_{Veto, latest};t_{DS, earliest} - t_{Veto, latest} [ns];Events / 0.5 ns",
        dt_bins, dt_min, dt_max
    )
    for val in aggregated_data["dt_ds_earliest_veto_latest"]:
        hists["h_dt_ds_earliest_minus_veto_latest"].Fill(val)

    hists["h_dt_us_earliest_minus_scifi_earliest"] = ROOT.TH1D(
        "h_dt_us_earliest_minus_scifi_earliest",
        "t_{US, earliest} - t_{SciFi, earliest};t_{US, earliest} - t_{SciFi, earliest} [ns];Events / 0.5 ns",
        dt_bins, dt_min, dt_max
    )
    for val in aggregated_data["dt_us_earliest_sf_earliest"]:
        hists["h_dt_us_earliest_minus_scifi_earliest"].Fill(val)

    hists["h_dt_ds_latest_minus_scifi_earliest"] = ROOT.TH1D(
        "h_dt_ds_latest_minus_scifi_earliest",
        "t_{DS, latest} - t_{SciFi, earliest} (SciFi-DS Timing Cut);t_{DS, latest} - t_{SciFi, earliest} [ns];Events / 0.5 ns",
        dt_bins, dt_min, dt_max
    )
    for val in aggregated_data["dt_ds_latest_sf_earliest"]:
        hists["h_dt_ds_latest_minus_scifi_earliest"].Fill(val)

    hists["h_dt_ds_earliest_minus_scifi_earliest"] = ROOT.TH1D(
        "h_dt_ds_earliest_minus_scifi_earliest",
        "t_{DS, earliest} - t_{SciFi, earliest};t_{DS, earliest} - t_{SciFi, earliest} [ns];Events / 0.5 ns",
        dt_bins, dt_min, dt_max
    )
    for val in aggregated_data["dt_ds_earliest_sf_earliest"]:
        hists["h_dt_ds_earliest_minus_scifi_earliest"].Fill(val)

    hists["h_dt_ds_earliest_minus_us_earliest"] = ROOT.TH1D(
        "h_dt_ds_earliest_minus_us_earliest",
        "t_{DS, earliest} - t_{US, earliest};t_{DS, earliest} - t_{US, earliest} [ns];Events / 0.5 ns",
        dt_bins, dt_min, dt_max
    )
    for val in aggregated_data["dt_ds_earliest_us_earliest"]:
        hists["h_dt_ds_earliest_minus_us_earliest"].Fill(val)

    # 2D Correlations
    hists["h2_scifi_earliest_vs_veto_latest"] = ROOT.TH2D(
        "h2_scifi_earliest_vs_veto_latest",
        "t_{SciFi, earliest} vs t_{Veto, latest};t_{Veto, latest} [ns];t_{SciFi, earliest} [ns]",
        50, 0, window_max, 50, 0, window_max
    )
    for x, y in aggregated_data["corr_sf_veto"]:
        hists["h2_scifi_earliest_vs_veto_latest"].Fill(x, y)

    hists["h2_ds_latest_vs_scifi_earliest"] = ROOT.TH2D(
        "h2_ds_latest_vs_scifi_earliest",
        "t_{DS, latest} vs t_{SciFi, earliest};t_{SciFi, earliest} [ns];t_{DS, latest} [ns]",
        50, 0, window_max, 50, 0, window_max
    )
    for x, y in aggregated_data["corr_ds_sf"]:
        hists["h2_ds_latest_vs_scifi_earliest"].Fill(x, y)

    hists["h2_us_earliest_vs_scifi_earliest"] = ROOT.TH2D(
        "h2_us_earliest_vs_scifi_earliest",
        "t_{US, earliest} vs t_{SciFi, earliest};t_{SciFi, earliest} [ns];t_{US, earliest} [ns]",
        50, 0, window_max, 50, 0, window_max
    )
    for x, y in aggregated_data["corr_us_sf"]:
        hists["h2_us_earliest_vs_scifi_earliest"].Fill(x, y)

    hists["h2_ds_latest_vs_us_earliest"] = ROOT.TH2D(
        "h2_ds_latest_vs_us_earliest",
        "t_{DS, latest} vs t_{US, earliest};t_{US, earliest} [ns];t_{DS, latest} [ns]",
        50, 0, window_max, 50, 0, window_max
    )
    for x, y in aggregated_data["corr_ds_us"]:
        hists["h2_ds_latest_vs_us_earliest"].Fill(x, y)

    # Stylize 1D histograms
    colors = {
        "veto": ROOT.kRed + 1,
        "scifi": ROOT.kAzure + 1,
        "us": ROOT.kOrange + 7,
        "ds": ROOT.kSpring - 1,
    }

    for key, h in hists.items():
        if isinstance(h, ROOT.TH1D):
            h.SetLineWidth(2)
            if "veto" in key:
                h.SetLineColor(colors["veto"])
            elif "scifi" in key:
                h.SetLineColor(colors["scifi"])
            elif "us" in key:
                h.SetLineColor(colors["us"])
            elif "ds" in key:
                h.SetLineColor(colors["ds"])
            else:
                h.SetLineColor(ROOT.kBlack)

    return hists


def build_canvases(
    hists: Dict[str, Any],
    window_max: float = 25.0,
) -> Dict[str, ROOT.TCanvas]:
    """Assemble structured TCanvases for timing analysis."""
    canvases = {}

    # 1. Four-subsystem overlay within 25ns window
    c_win_overlay = ROOT.TCanvas("c_subsystem_window_overlay", "Subsystem Hit Times in 25ns Window", 900, 700)
    c_win_overlay.SetGrid(1, 1)
    c_win_overlay.SetLeftMargin(0.12)
    c_win_overlay.SetRightMargin(0.08)

    max_val = max(
        hists["h_time_veto"].GetMaximum(),
        hists["h_time_scifi"].GetMaximum(),
        hists["h_time_us"].GetMaximum(),
        hists["h_time_ds"].GetMaximum(),
    ) * 1.25

    hists["h_time_scifi"].SetMaximum(max_val)
    hists["h_time_scifi"].SetTitle("Hit Time Distributions within 25ns Readout Window;Hit Time [ns];Normalized Count")
    hists["h_time_scifi"].DrawNormalized("HIST")
    hists["h_time_veto"].DrawNormalized("HIST SAME")
    hists["h_time_us"].DrawNormalized("HIST SAME")
    hists["h_time_ds"].DrawNormalized("HIST SAME")

    leg = ROOT.TLegend(0.60, 0.68, 0.88, 0.88)
    leg.SetBorderSize(1)
    leg.SetFillColor(0)
    leg.AddEntry(hists["h_time_veto"], "Veto Hits", "l")
    leg.AddEntry(hists["h_time_scifi"], "SciFi Hits", "l")
    leg.AddEntry(hists["h_time_us"], "US (HCAL) Hits", "l")
    leg.AddEntry(hists["h_time_ds"], "DS (Muon Filter) Hits", "l")
    leg.Draw()
    canvases["c_subsystem_window_overlay"] = c_win_overlay

    # 2. 4-Panel Grid of Window Hit Times
    c_win_grid = ROOT.TCanvas("c_subsystem_window_grid", "Subsystem Window Distributions", 1100, 850)
    c_win_grid.Divide(2, 2)
    subsystems = [("h_time_veto", 1), ("h_time_scifi", 2), ("h_time_us", 3), ("h_time_ds", 4)]
    for hname, pad_idx in subsystems:
        pad = c_win_grid.cd(pad_idx)
        pad.SetGrid(1, 1)
        h = hists[hname]
        h.Draw("HIST")
    canvases["c_subsystem_window_grid"] = c_win_grid

    # 3. 4-Panel Grid of Earliest Hit Times
    c_min_grid = ROOT.TCanvas("c_subsystem_earliest_grid", "Earliest Subsystem Hit Time per Event", 1100, 850)
    c_min_grid.Divide(2, 2)
    mins = [("h_time_min_veto", 1), ("h_time_min_scifi", 2), ("h_time_min_us", 3), ("h_time_min_ds", 4)]
    for hname, pad_idx in mins:
        pad = c_min_grid.cd(pad_idx)
        pad.SetGrid(1, 1)
        hists[hname].Draw("HIST")
    canvases["c_subsystem_earliest_grid"] = c_min_grid

    # 4. Key Time Differences (SciFi-Veto, SciFi-DS, US-Veto, SciFi-US)
    c_dts = ROOT.TCanvas("c_subsystem_time_differences", "Subsystem Time Differences", 1200, 900)
    c_dts.Divide(2, 2)

    c_dts.cd(1).SetGrid(1, 1)
    hists["h_dt_scifi_earliest_minus_veto_latest"].Draw("HIST")
    c_dts.cd(2).SetGrid(1, 1)
    hists["h_dt_ds_latest_minus_scifi_earliest"].Draw("HIST")
    c_dts.cd(3).SetGrid(1, 1)
    hists["h_dt_us_earliest_minus_veto_latest"].Draw("HIST")
    c_dts.cd(4).SetGrid(1, 1)
    hists["h_dt_us_earliest_minus_scifi_earliest"].Draw("HIST")
    canvases["c_subsystem_time_differences"] = c_dts

    # 5. Additional Time Differences (Veto-DS, SciFi-DS earliest, US-DS)
    c_dts2 = ROOT.TCanvas("c_subsystem_time_differences_extra", "Additional Subsystem Time Differences", 1200, 900)
    c_dts2.Divide(2, 2)

    c_dts2.cd(1).SetGrid(1, 1)
    hists["h_dt_scifi_earliest_minus_veto_earliest"].Draw("HIST")
    c_dts2.cd(2).SetGrid(1, 1)
    hists["h_dt_ds_earliest_minus_veto_latest"].Draw("HIST")
    c_dts2.cd(3).SetGrid(1, 1)
    hists["h_dt_ds_earliest_minus_scifi_earliest"].Draw("HIST")
    c_dts2.cd(4).SetGrid(1, 1)
    hists["h_dt_ds_earliest_minus_us_earliest"].Draw("HIST")
    canvases["c_subsystem_time_differences_extra"] = c_dts2

    # 6. 2D Timing Correlations
    c_corrs = ROOT.TCanvas("c_subsystem_correlations", "2D Timing Correlations", 1200, 900)
    c_corrs.Divide(2, 2)

    c_corrs.cd(1).SetGrid(1, 1)
    hists["h2_scifi_earliest_vs_veto_latest"].Draw("COLZ")
    c_corrs.cd(2).SetGrid(1, 1)
    hists["h2_ds_latest_vs_scifi_earliest"].Draw("COLZ")
    c_corrs.cd(3).SetGrid(1, 1)
    hists["h2_us_earliest_vs_scifi_earliest"].Draw("COLZ")
    c_corrs.cd(4).SetGrid(1, 1)
    hists["h2_ds_latest_vs_us_earliest"].Draw("COLZ")
    canvases["c_subsystem_correlations"] = c_corrs

    return canvases


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose and plot timing distributions and differences across SND@LHC detector subsystems."
    )
    parser.add_argument(
        "-i", "--input",
        default="/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_MCEB_dig.root",
        help="Input ROOT file path or pattern containing '%%s' placeholder (default: 2022 MCEB dig).",
    )
    parser.add_argument(
        "-r", "--run",
        type=int,
        default=None,
        help="Run number for collision data (e.g. 6596). Automatically locates data path and official geofile.",
    )
    parser.add_argument(
        "--geofile",
        default=None,
        help="Custom geofile path. If omitted for collision data, automatically looked up in config/geo_paths.csv.",
    )
    parser.add_argument(
        "--geo-csv",
        default="config/geo_paths.csv",
        help="CSV file mapping run numbers to geofile paths (default: config/geo_paths.csv).",
    )
    parser.add_argument(
        "-p", "--partitions",
        default="1",
        help="Partitions to process: range ('1-10'), comma-separated ('1,2,5'), or 'all' (default: 1; or 0 for data).",
    )
    parser.add_argument(
        "-f", "--partitions-file",
        default=None,
        help="Text file with partition IDs (one per line).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="plots/timing_diagnostics",
        help="Output directory for generated plots & ROOT file (default: plots/timing_diagnostics).",
    )
    parser.add_argument(
        "--root-output",
        default="subsystem_timing_diagnostics.root",
        help="Output ROOT filename containing histograms & canvases (default: subsystem_timing_diagnostics.root).",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of concurrent processes for multi-partition processing (default: 4).",
    )
    parser.add_argument(
        "-n", "--max-events",
        type=int,
        default=-1,
        help="Maximum events to analyze per partition (-1 for all, default: -1).",
    )
    parser.add_argument(
        "-w", "--window-max",
        type=float,
        default=25.0,
        help="Maximum window time in nanoseconds for window histograms (default: 25.0 ns).",
    )

    args = parser.parse_args()

    # Determine partition list and input files
    files_to_process = []
    if args.run is not None:
        run_matches = glob.glob(f"/eos/experiment/sndlhc/convertedData/physics/*/run_{args.run:06d}")
        if not run_matches:
            print(f"ERROR: Could not locate directory for run {args.run:06d} under /eos/experiment/sndlhc/convertedData/physics/*/", file=sys.stderr)
            sys.exit(1)
        run_dir = run_matches[0]
        all_run_files = sorted(glob.glob(f"{run_dir}/sndsw_raw-*.root"))
        if not all_run_files:
            print(f"ERROR: No sndsw_raw-*.root files found in {run_dir}", file=sys.stderr)
            sys.exit(1)

        if args.output_dir == "plots/timing_diagnostics":
            args.output_dir = f"plots/timing_run_{args.run:06d}"

        if args.partitions != "1" or args.partitions_file:
            parts = parse_partitions(args.partitions) if not args.partitions_file else []
            if args.partitions_file and os.path.isfile(args.partitions_file):
                with open(args.partitions_file, "r") as pf:
                    parts = [int(l.strip()) for l in pf if l.strip().isdigit()]
            for p in parts:
                candidate = f"{run_dir}/sndsw_raw-{p:04d}.root"
                if os.path.isfile(candidate):
                    files_to_process.append(candidate)
        else:
            files_to_process = [all_run_files[0]]
    else:
        partitions = []
        if args.partitions_file and os.path.isfile(args.partitions_file):
            with open(args.partitions_file, "r") as pf:
                for line in pf:
                    line = line.strip()
                    if line and not line.startswith("#") and line.isdigit():
                        partitions.append(int(line))
            partitions = sorted(list(set(partitions)))
        elif "%s" in args.input:
            partitions = parse_partitions(args.partitions)
        else:
            partitions = [1]

        for p in partitions:
            fpath = format_partition_path(args.input, p)
            if os.path.isfile(fpath):
                files_to_process.append(fpath)

    if not files_to_process:
        print("ERROR: No valid input files found.", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print(" SND@LHC DETECTOR SUBSYSTEM TIMING DIAGNOSTIC")
    print("=" * 80)
    if args.run is not None:
        print(f" Collision Run       : {args.run:06d}")
    print(f" Files to Analyze    : {len(files_to_process)} file(s)")
    print(f" Output Directory    : {args.output_dir}")
    print(f" Output ROOT File    : {args.root_output}")
    print(f" Max Events / File   : {'All' if args.max_events <= 0 else args.max_events}")
    print(f" Readout Window Max  : {args.window_max} ns")
    print(f" Parallel Workers    : {args.jobs}")
    print("=" * 80)

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # Process files
    results = []
    if args.jobs > 1 and len(files_to_process) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(process_single_file, fp, args.max_events, args.geofile, args.geo_csv): fp
                for fp in files_to_process
            }
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for fp in files_to_process:
            results.append(process_single_file(fp, args.max_events, args.geofile, args.geo_csv))

    # Merge results
    merged_data: Dict[str, Any] = {
        "all_v": [], "all_sf": [], "all_us": [], "all_ds": [],
        "min_v": [], "min_sf": [], "min_us": [], "min_ds": [],
        "max_v": [], "max_sf": [], "max_us": [], "max_ds": [],
        "dt_sf_earliest_veto_latest": [],
        "dt_sf_earliest_veto_earliest": [],
        "dt_us_earliest_veto_latest": [],
        "dt_ds_earliest_veto_latest": [],
        "dt_us_earliest_sf_earliest": [],
        "dt_ds_latest_sf_earliest": [],
        "dt_ds_earliest_sf_earliest": [],
        "dt_ds_earliest_us_earliest": [],
        "corr_sf_veto": [],
        "corr_ds_sf": [],
        "corr_us_sf": [],
        "corr_ds_us": [],
        "events_total": 0,
    }

    for r in results:
        merged_data["events_total"] += r["events_total"]
        for key in merged_data:
            if key != "events_total" and key in r:
                merged_data[key].extend(r[key])

    print(f"\n Processed {merged_data['events_total']} events across {len(files_to_process)} files in {time.time() - t0:.2f}s.")
    print(f" Total Hits Collected:")
    print(f"   * Veto Hits : {len(merged_data['all_v']):>8}")
    print(f"   * SciFi Hits: {len(merged_data['all_sf']):>8}")
    print(f"   * US Hits   : {len(merged_data['all_us']):>8}")
    print(f"   * DS Hits   : {len(merged_data['all_ds']):>8}")

    print("\n Timing Metric Summary:")
    dts_info = [
        ("t(SciFi_min) - t(Veto_max)", merged_data["dt_sf_earliest_veto_latest"]),
        ("t(SciFi_min) - t(Veto_min)", merged_data["dt_sf_earliest_veto_earliest"]),
        ("t(US_min) - t(Veto_max)", merged_data["dt_us_earliest_veto_latest"]),
        ("t(DS_min) - t(Veto_max)", merged_data["dt_ds_earliest_veto_latest"]),
        ("t(US_min) - t(SciFi_min)", merged_data["dt_us_earliest_sf_earliest"]),
        ("t(DS_max) - t(SciFi_min)", merged_data["dt_ds_latest_sf_earliest"]),
        ("t(DS_min) - t(SciFi_min)", merged_data["dt_ds_earliest_sf_earliest"]),
        ("t(DS_min) - t(US_min)", merged_data["dt_ds_earliest_us_earliest"]),
    ]

    for label, vals in dts_info:
        if vals:
            mean_v = sum(vals) / len(vals)
            print(f"   * {label:<32}: N={len(vals):>6} | Mean={mean_v:>6.2f} ns | Min={min(vals):>6.2f} ns | Max={max(vals):>6.2f} ns")
        else:
            print(f"   * {label:<32}: N=0 (no coincident hits)")

    # Build Histograms & Canvases
    hists = create_and_fill_histograms(merged_data, window_max=args.window_max)
    canvases = build_canvases(hists, window_max=args.window_max)

    # Save to ROOT file
    root_filepath = os.path.join(args.output_dir, args.root_output)
    f_out = ROOT.TFile.Open(root_filepath, "RECREATE")
    for h in hists.values():
        h.Write()
    for c in canvases.values():
        c.Write()
    f_out.Close()
    print(f"\n [ROOT] Histograms & Canvases saved to: {root_filepath}")

    # Export images
    print(" [Export] Saving PNG and PDF figures:")
    for cname, c in canvases.items():
        png_path = os.path.join(args.output_dir, f"{cname}.png")
        pdf_path = os.path.join(args.output_dir, f"{cname}.pdf")
        c.Print(png_path)
        c.Print(pdf_path)
        print(f"   -> {png_path}")

    print("=" * 80)
    print(" Timing diagnostics completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
