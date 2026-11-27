#!/usr/bin/env python3
"""
scripts/qdc_calib/study_ti18_ds_shower_profile.py
------------------------------------------------
High-statistics in-situ validation of DS detection thresholds using
muon-induced showers in TI18 collision data (Run 5408) vs PMU simulations.

Methodology:
1. Select clean through-going halo muons with electromagnetic shower / delta-ray
   production in the Downstream calorimeter (N_DS >= 5 hits).
2. For each reconstructed muon track, compute the transverse perpendicular distance
   r_perp to every DS bar:
     - Horizontal bars: r_perp = |y_bar - y_track(z_bar)|
     - Vertical bars:   r_perp = |x_bar - x_track(z_bar)|
3. Measure the hit probability:
     P(hit | r_perp) = N_fired(r_perp) / N_tested(r_perp)
   in 1.0 cm bins from r_perp = 0 to 25 cm.
4. Compare Data against 5 simulation threshold models:
   - Sim Raw (E_th = 0)
   - Sim Theo (k = 1.0)
   - Sim Calib (k_SiPM from config/sipm_photoelectron_reduction_factors.csv)
   - Sim Step (1.0 MeV)
   - Sim Tuned Step (0.35 MeV)

Calculates rigorous binomial error bars for both Data and MC,
ratios to Data (MC / Data), and writes canvases and histograms to:
  output/ti18_ds_shower_profile_validation.root
and generates:
  plots/ti18_ds_shower_profile_highstat.png
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


def build_ds_geometry(geofile: str) -> Dict[int, Dict[str, Any]]:
    """Builds geometry coordinates for all DS bars (subsystem 3)."""
    geo = SndlhcGeo.GeoInterface(geofile)
    mufi = geo.modules["MuFilter"]
    A, B = ROOT.TVector3(), ROOT.TVector3()
    ds_bars = {}

    for pl in range(4):
        b_start = 60 if pl == 3 else 0
        for b in range(b_start, 120):
            did = 30000 + pl * 1000 + b
            mufi.GetPosition(did, A, B)
            is_vert = (b >= 60)
            ds_bars[did] = {
                "plane": pl,
                "bar": b,
                "is_vert": is_vert,
                "x_min": min(A.X(), B.X()),
                "x_max": max(A.X(), B.X()),
                "y_min": min(A.Y(), B.Y()),
                "y_max": max(A.Y(), B.Y()),
                "x_c": 0.5 * (A.X() + B.X()),
                "y_c": 0.5 * (A.Y() + B.Y()),
                "z": 0.5 * (A.Z() + B.Z()),
            }
    return ds_bars


def precompute_thresholds() -> Tuple[Dict[int, float], Dict[int, float]]:
    """Precomputes theoretical (k=1) and calibrated (k_SiPM) thresholds in keV for DS bars."""
    calc_theo = mpe.MinPE(year=2022, is_testbeam=False)
    calc_theo._k_factors = {}

    calc_calib = mpe.MinPE(year=2022, is_testbeam=False)

    th_theo: Dict[int, float] = {}
    th_calib: Dict[int, float] = {}

    for pl in range(4):
        b_start = 60 if pl == 3 else 0
        for b in range(b_start, 120):
            did = 30000 + pl * 1000 + b
            is_vert = (b >= 60)
            p = {"detector_id": did, "x": 0.0 if is_vert else -37.7, "y": 20.0 if is_vert else 0.0, "z": 0.0, "eloss": 1.0}
            th_theo[did] = float(calc_theo.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0
            th_calib[did] = float(calc_calib.get_lowest_trigger_threshold([p], det_id=did)["min_eloss_mev"]) * 1000.0

    return th_theo, th_calib


def process_shower_profile(
    data_file: str,
    mc_file: str,
    geofile: str,
    out_root: str = "output/ti18_ds_shower_profile_validation.root",
    out_png: str = "plots/ti18_ds_shower_profile_highstat.png",
    max_data_events: int = 100000,
    max_mc_events: int = 150000,
    target_showers: int = 2500,
):
    os.makedirs(os.path.dirname(out_root) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    print("[Init] Extracting DS detector geometry...")
    ds_bars = build_ds_geometry(geofile)
    print(f"[Init] Mapped {len(ds_bars)} Downstream bars.")

    print("[Init] Precomputing physical thresholds...")
    th_theo, th_calib = precompute_thresholds()

    # Bins: 0 to 25 cm in 1.0 cm bins (25 bins)
    r_bins = np.linspace(0.0, 25.0, 26)
    nb = len(r_bins) - 1

    models = ["data", "raw", "theo", "calib", "step10", "step035"]
    model_labels = {
        "data": "Run 5408 Data",
        "raw": "Sim Raw (Eth=0)",
        "theo": "Sim Theo (k=1.0)",
        "calib": "Sim Theo+k(SiPM)",
        "step10": "Sim Step (1.0 MeV)",
        "step035": "Sim Tuned Step (0.35 MeV)",
    }

    n_tested = {m: np.zeros(nb, dtype=np.float64) for m in models}
    n_fired = {m: np.zeros(nb, dtype=np.float64) for m in models}

    # =========================================================================
    # 1. Real Collision Data (Run 5408)
    # =========================================================================
    print(f"\n[Processing Data] Scanning up to {max_data_events:,} entries from {data_file}...")
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
        if (iev + 1) % 15000 == 0:
            print(f"  Data {iev+1:,}/{n_entries_d:,} (Showers analyzed: {n_showers_d})...", flush=True)
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

        fired_ds = set()
        for hit in t_data.Digi_MuFilterHits:
            if not hit.isValid():
                continue
            did = hit.GetDetectorID()
            if did // 10000 == 3:
                fired_ds.add(did)

        # Shower selection: require secondary activity (N_DS >= 5)
        if len(fired_ds) < 5:
            continue

        trackTask.fittedTracks.Delete()
        trackTask.ExecuteTask(option="ScifiDS")

        ds_trk = None
        for trk in trackTask.fittedTracks:
            if trk.getTrackType() == 3 and trk.getChi2Ndf() <= 5.0:
                if abs(trk.getSlopeXZ()) <= 0.1 and abs(trk.getSlopeYZ()) <= 0.1:
                    ds_trk = trk
                    break

        if ds_trk is None:
            continue

        n_showers_d += 1

        for did, b_info in ds_bars.items():
            pt = ds_trk.getPointAtZ(b_info["z"])
            if not b_info["is_vert"]:
                # Horizontal bar: measures y
                if b_info["x_min"] - 2.0 <= pt.X() <= b_info["x_max"] + 2.0:
                    r_perp = abs(pt.Y() - b_info["y_c"])
                    bin_idx = int(np.digitize(r_perp, r_bins) - 1)
                    if 0 <= bin_idx < nb:
                        n_tested["data"][bin_idx] += 1.0
                        if did in fired_ds:
                            n_fired["data"][bin_idx] += 1.0
            else:
                # Vertical bar: measures x
                if b_info["y_min"] - 2.0 <= pt.Y() <= b_info["y_max"] + 2.0:
                    r_perp = abs(pt.X() - b_info["x_c"])
                    bin_idx = int(np.digitize(r_perp, r_bins) - 1)
                    if 0 <= bin_idx < nb:
                        n_tested["data"][bin_idx] += 1.0
                        if did in fired_ds:
                            n_fired["data"][bin_idx] += 1.0

    print(f"[Data Done] Total shower events analyzed: {n_showers_d:,}")

    # =========================================================================
    # 2. PMU Simulation Sample across 5 Models
    # =========================================================================
    print(f"\n[Processing Simulation] Scanning up to {max_mc_events:,} entries from {mc_file}...")
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
        if (iev + 1) % 20000 == 0:
            print(f"  MC {iev+1:,}/{n_entries_m:,} (Showers analyzed: {n_showers_m})...", flush=True)
        if n_showers_m >= target_showers:
            break

        t_mc.GetEvent(iev)

        bar_edep_kev: Dict[int, float] = {}
        for pt in t_mc.MuFilterPoint:
            did = pt.GetDetectorID()
            if did // 10000 == 3:
                bar_edep_kev[did] = bar_edep_kev.get(did, 0.0) + pt.GetEnergyLoss() * 1.0e6

        if len(bar_edep_kev) < 5:
            continue

        # Form hit sets for the 5 MC models
        fired_mc: Dict[str, set] = {m: set() for m in ["raw", "theo", "calib", "step10", "step035"]}
        for did, edep in bar_edep_kev.items():
            if edep <= 0.0:
                continue
            fired_mc["raw"].add(did)
            if edep > th_theo.get(did, 300.0):
                fired_mc["theo"].add(did)
            if edep > th_calib.get(did, 300.0):
                fired_mc["calib"].add(did)
            if edep > 1000.0:  # 1.0 MeV
                fired_mc["step10"].add(did)
            if edep > 350.0:   # 0.35 MeV
                fired_mc["step035"].add(did)

        # Track reconstruction
        trackTask.fittedTracks.Delete()
        trackTask.ExecuteTask(option="ScifiDS")

        ds_trk = None
        for trk in trackTask.fittedTracks:
            if trk.getTrackType() == 3 and trk.getChi2Ndf() <= 5.0:
                if abs(trk.getSlopeXZ()) <= 0.1 and abs(trk.getSlopeYZ()) <= 0.1:
                    ds_trk = trk
                    break

        if ds_trk is None:
            continue

        n_showers_m += 1

        for did, b_info in ds_bars.items():
            pt = ds_trk.getPointAtZ(b_info["z"])
            if not b_info["is_vert"]:
                if b_info["x_min"] - 2.0 <= pt.X() <= b_info["x_max"] + 2.0:
                    r_perp = abs(pt.Y() - b_info["y_c"])
                    bin_idx = int(np.digitize(r_perp, r_bins) - 1)
                    if 0 <= bin_idx < nb:
                        for m in ["raw", "theo", "calib", "step10", "step035"]:
                            n_tested[m][bin_idx] += 1.0
                            if did in fired_mc[m]:
                                n_fired[m][bin_idx] += 1.0
            else:
                if b_info["y_min"] - 2.0 <= pt.Y() <= b_info["y_max"] + 2.0:
                    r_perp = abs(pt.X() - b_info["x_c"])
                    bin_idx = int(np.digitize(r_perp, r_bins) - 1)
                    if 0 <= bin_idx < nb:
                        for m in ["raw", "theo", "calib", "step10", "step035"]:
                            n_tested[m][bin_idx] += 1.0
                            if did in fired_mc[m]:
                                n_fired[m][bin_idx] += 1.0

    print(f"[MC Done] Total shower events analyzed: {n_showers_m:,}")

    # =========================================================================
    # 3. Calculate Hit Probabilities, Binomial Errors, and Ratios
    # =========================================================================
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    probs: Dict[str, np.ndarray] = {}
    errs: Dict[str, np.ndarray] = {}

    for m in models:
        tst = n_tested[m]
        fid = n_fired[m]
        p = np.zeros(nb, dtype=np.float64)
        e = np.zeros(nb, dtype=np.float64)
        mask = tst > 0
        p[mask] = fid[mask] / tst[mask]
        e[mask] = np.sqrt(np.clip(p[mask] * (1.0 - p[mask]) / tst[mask], 1e-12, 1.0))
        # Handle zero-count upper limit (Clopper-Pearson 68% for 0 counts ~ 1/N)
        zero_mask = (tst > 0) & (fid == 0)
        e[zero_mask] = 1.0 / tst[zero_mask]
        probs[m] = p
        errs[m] = e

    # =========================================================================
    # 4. Save to ROOT File
    # =========================================================================
    print(f"\n[Writing ROOT] Saving to {out_root}...")
    f_out = ROOT.TFile.Open(out_root, "RECREATE")
    dir_prof = f_out.mkdir("DS_Shower_Lateral_Profile")
    dir_prof.cd()

    root_styles = {
        "data":    {"color": ROOT.kBlack, "marker": 20, "ms": 1.2, "lw": 2},
        "raw":     {"color": ROOT.kAzure + 1, "marker": 21, "ms": 1.0, "lw": 2},
        "theo":    {"color": ROOT.kViolet + 1, "marker": 22, "ms": 1.0, "lw": 2},
        "calib":   {"color": ROOT.kRed + 1, "marker": 23, "ms": 1.1, "lw": 2},
        "step10":  {"color": ROOT.kGreen + 2, "marker": 24, "ms": 1.0, "lw": 2},
        "step035": {"color": ROOT.kOrange + 7, "marker": 25, "ms": 1.0, "lw": 2},
    }

    c_root = ROOT.TCanvas("c_ds_shower_profile_highstat", "DS Shower Hit Probability vs r_perp (High Stats)", 900, 700)
    c_root.SetLogy()
    c_root.SetGridx()
    c_root.SetGridy()

    leg = ROOT.TLegend(0.55, 0.65, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    first = True
    for m in models:
        h = ROOT.TH1D(f"h_ds_shower_prof_{m}", f"{model_labels[m]};r_{{#perp}} from Track [cm];Hit Probability", nb, r_bins)
        for i in range(nb):
            h.SetBinContent(i + 1, max(probs[m][i], 1e-5))
            h.SetBinError(i + 1, errs[m][i])

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
    c_root.Write()
    f_out.Close()
    print(f"Saved ROOT histograms and canvas to: {out_root}")

    # =========================================================================
    # 5. Publication-Quality Matplotlib Figure with Ratio Panel
    # =========================================================================
    mpl_styles = {
        "data":    {"label": f"Run 5408 Data (N={n_showers_d:,})", "color": "black", "ls": "none", "marker": "o", "ms": 5.5, "lw": 1.5},
        "raw":     {"label": "Sim Raw (Eth=0)", "color": "#1f77b4", "ls": ":", "marker": "s", "ms": 4.5, "lw": 1.6},
        "theo":    {"label": "Sim Theo (k=1.0)", "color": "#9467bd", "ls": "--", "marker": "D", "ms": 4.5, "lw": 1.8},
        "calib":   {"label": "Sim Theo+k(SiPM)", "color": "#d62728", "ls": "-", "marker": "v", "ms": 5.0, "lw": 2.0},
        "step10":  {"label": "Sim Step (1.0 MeV)", "color": "#2ca02c", "ls": "-.", "marker": "^", "ms": 4.5, "lw": 1.8},
        "step035": {"label": "Sim Tuned Step (0.35 MeV)", "color": "#ff7f0e", "ls": "-", "marker": "p", "ms": 4.5, "lw": 1.8},
    }

    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(9, 7.5),
        gridspec_kw={"height_ratios": [3.2, 1.3], "hspace": 0.08},
        constrained_layout=False,
    )

    # Main plot
    for m in models:
        st = mpl_styles[m]
        p = probs[m]
        e = errs[m]
        mask = p > 0

        if m == "data":
            ax_main.errorbar(
                r_centers[mask], p[mask], yerr=e[mask],
                fmt=st["marker"], color=st["color"], ms=st["ms"],
                lw=st["lw"], capsize=2.5, label=st["label"], zorder=10
            )
        else:
            ax_main.errorbar(
                r_centers[mask], p[mask], yerr=e[mask],
                fmt=st["marker"], color=st["color"], ls=st["ls"],
                ms=st["ms"], lw=st["lw"], capsize=1.5, label=st["label"], alpha=0.9
            )

    ax_main.set_yscale("log")
    ax_main.set_xlim(0, 25)
    ax_main.set_ylim(2.0e-4, 1.2)
    ax_main.set_ylabel(r"Hit Probability $P(\mathrm{hit} \mid r_{\perp})$", fontsize=12, fontweight="bold")
    ax_main.set_title("TI18 Muon-Induced Shower Lateral Profile in DS Bars", fontsize=13, fontweight="bold")
    ax_main.grid(True, which="both", ls=":", alpha=0.55)
    ax_main.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5, loc="upper right")
    ax_main.tick_params(labelbottom=False, labelsize=10)

    # Ratio plot (MC / Data)
    p_data = probs["data"]
    e_data = errs["data"]

    for m in ["raw", "theo", "calib", "step10", "step035"]:
        st = mpl_styles[m]
        p_mc = probs[m]
        e_mc = errs[m]

        valid = (p_data > 0) & (p_mc > 0)
        ratio = np.full(nb, np.nan)
        ratio_err = np.full(nb, np.nan)

        ratio[valid] = p_mc[valid] / p_data[valid]
        # Relative error propagation: sqrt((e_mc/p_mc)^2 + (e_d/p_d)^2) * ratio
        rel_err = np.sqrt((e_mc[valid] / p_mc[valid])**2 + (e_data[valid] / p_data[valid])**2)
        ratio_err[valid] = ratio[valid] * rel_err

        ax_ratio.errorbar(
            r_centers[valid], ratio[valid], yerr=ratio_err[valid],
            fmt=st["marker"], color=st["color"], ls=st["ls"],
            ms=st["ms"] * 0.9, lw=st["lw"] * 0.9, capsize=1.5
        )

    ax_ratio.axhline(1.0, color="black", ls="--", lw=1.2)
    ax_ratio.set_xlim(0, 25)
    ax_ratio.set_ylim(0.0, 4.5)
    ax_ratio.set_xlabel(r"Transverse Distance from Track $r_{\perp}$ [cm]", fontsize=11, fontweight="bold")
    ax_ratio.set_ylabel("MC / Data", fontsize=11, fontweight="bold")
    ax_ratio.grid(True, ls=":", alpha=0.55)
    ax_ratio.tick_params(labelsize=10)
    ax_ratio.yaxis.set_minor_locator(AutoMinorLocator())

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved high-statistics PNG figure to: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="High-statistics DS shower profile threshold validation")
    parser.add_argument("--data", default="/eos/experiment/sndlhc/convertedData/physics/2022/run_005408/sndsw_raw-0002.root")
    parser.add_argument("--mc", default="/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP_truth.root")
    parser.add_argument("--geofile", default="/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/geofile_full.Ntuple-TGeant4.root")
    parser.add_argument("--output", default="output/ti18_ds_shower_profile_validation.root")
    parser.add_argument("--png", default="plots/ti18_ds_shower_profile_highstat.png")
    parser.add_argument("--max-data", type=int, default=100000)
    parser.add_argument("--max-mc", type=int, default=150000)
    parser.add_argument("--target-showers", type=int, default=2500)
    args = parser.parse_args()

    process_shower_profile(
        data_file=args.data,
        mc_file=args.mc,
        geofile=args.geofile,
        out_root=args.output,
        out_png=args.png,
        max_data_events=args.max_data,
        max_mc_events=args.max_mc,
        target_showers=args.target_showers,
    )


if __name__ == "__main__":
    main()
