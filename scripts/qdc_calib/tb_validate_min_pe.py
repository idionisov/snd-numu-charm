#!/usr/bin/env python3
"""
Testbeam 2023 H8 MuFilter Validation of the Physical Threshold Model (MinPE with SiPM k-factors).

Compares across 5 beam energies (100, 140, 180, 240, 300 GeV pi+/-):
1. In-time Collision / Testbeam Data
2. Uncorrected Simulation (E_th = 0)
3. Nominal Flat Thresholds (US = 130 keV, DS = 22.1 keV)
4. Calibrated Physical Threshold Model (MinPE with channel-by-channel SiPM reduction factors k)

Computes mean hit counts, Wasserstein distances (W1), and generates ROOT and PNG comparison figures.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import wasserstein_distance

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Add scripts directory for MinPE
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, SCRIPT_DIR)
from min_pe import MinPE

# Testbeam run configurations
CONFIGS = {
    100: {
        'name': '100 GeV #pi^{+}',
        'particle': 'pi+',
        'mc_files': ['/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/100GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root'],
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100677/sndsw_raw-0000.root'],
        'run_num': 100677,
    },
    140: {
        'name': '140 GeV #pi^{+}',
        'particle': 'pi+',
        'mc_files': ['/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/140GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root'],
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100674/sndsw_raw-0000.root'],
        'run_num': 100674,
    },
    180: {
        'name': '180 GeV #pi^{+}',
        'particle': 'pi+',
        'mc_files': ['/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/180GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root'],
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100635/sndsw_raw-0000.root'],
        'run_num': 100635,
    },
    240: {
        'name': '240 GeV #pi^{-}',
        'particle': 'pi-',
        'mc_files': [
            f'/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/240GeV_-211/{p}/sndLHC.PG_-211-TGeant4_MCEB_dig.root'
            for p in range(5) if os.path.exists(f'/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/240GeV_-211/{p}/sndLHC.PG_-211-TGeant4_MCEB_dig.root')
        ],
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100637/sndsw_raw-0000.root'],
        'run_num': 100637,
    },
    300: {
        'name': '300 GeV #pi^{-}',
        'particle': 'pi-',
        'mc_files': [
            f'/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/300GeV_-211/{p}/sndLHC.PG_-211-TGeant4_MCEB_dig.root'
            for p in [0, 1, 2, 4] if os.path.exists(f'/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/300GeV_-211/{p}/sndLHC.PG_-211-TGeant4_MCEB_dig.root')
        ],
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100639/sndsw_raw-0000.root'],
        'run_num': 100639,
    },
}

# Empirical flat thresholds for reference comparison
E_TH_FLAT_US = 0.000130   # 130 keV in GeV
E_TH_FLAT_DS = 0.0000221  # 22.1 keV in GeV


def precompute_physical_thresholds(calc: MinPE) -> Dict[int, float]:
    """Precompute physical energy threshold [GeV] for all MuFilter bars using MinPE with k-factors."""
    bar_th: Dict[int, float] = {}
    # Upstream (subsystem 2): 5 planes, 10 bars per plane
    for pl in range(5):
        for b in range(10):
            det_id = 20000 + pl * 1000 + b
            p = {'detector_id': det_id, 'x': -37.7, 'y': 0.0, 'z': 0.0, 'eloss': 1.0}
            th = calc.get_lowest_trigger_threshold([p], det_id=det_id)
            bar_th[det_id] = float(th['min_eloss_mev']) / 1000.0  # Convert MeV to GeV

    # Downstream (subsystem 3):
    for pl in range(4):
        # Horizontal bars (pl 0, 1, 2): bars 0..59
        if pl < 3:
            for b in range(60):
                det_id = 30000 + pl * 1000 + b
                p = {'detector_id': det_id, 'x': -37.7, 'y': 0.0, 'z': 0.0, 'eloss': 1.0}
                th = calc.get_lowest_trigger_threshold([p], det_id=det_id)
                bar_th[det_id] = float(th['min_eloss_mev']) / 1000.0
        # Vertical bars (pl 0, 1, 2, 3): bars 60..119
        for b in range(60, 120):
            det_id = 30000 + pl * 1000 + b
            p = {'detector_id': det_id, 'x': 0.0, 'y': 20.0, 'z': 0.0, 'eloss': 1.0}
            th = calc.get_lowest_trigger_threshold([p], det_id=det_id)
            bar_th[det_id] = float(th['min_eloss_mev']) / 1000.0

    return bar_th


def extract_data_showers(file_list: List[str], max_events: int = 4000) -> List[Dict[str, Any]]:
    """Extract in-time shower hit distributions from testbeam raw data."""
    chain = ROOT.TChain("rawConv")
    for f in file_list:
        chain.Add(f)
    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("Digi_MuFilterHits*", 1)

    n_entries = min(max_events, chain.GetEntries())
    events = []

    for iev in range(n_entries):
        chain.GetEntry(iev)
        hits = getattr(chain, "Digi_MuFilterHits", None)
        if not hits:
            continue

        us_counts = {p: 0 for p in range(5)}
        ds_counts = {
            'DS1_H': 0, 'DS1_V': 0,
            'DS2_H': 0, 'DS2_V': 0,
            'DS3_H': 0, 'DS3_V': 0,
            'DS4_V': 0,
        }
        us_bars_seen = set()
        ds_bars_seen = set()

        for hit in hits:
            if not hit.isValid():
                continue
            times = [hit.GetTime(ch) for ch in range(hit.GetnSides()) if hit.GetTime(ch) != -999.0]
            if not any(abs(t) <= 15.0 for t in times):
                continue
            did = hit.GetDetectorID()
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000

            if sub == 2:
                if 0 <= pl < 5 and did not in us_bars_seen:
                    us_bars_seen.add(did)
                    us_counts[pl] += 1
            elif sub == 3:
                if did not in ds_bars_seen:
                    ds_bars_seen.add(did)
                    if pl == 0:
                        key = 'DS1_H' if bar < 60 else 'DS1_V'
                    elif pl == 1:
                        key = 'DS2_H' if bar < 60 else 'DS2_V'
                    elif pl == 2:
                        key = 'DS3_H' if bar < 60 else 'DS3_V'
                    elif pl == 3:
                        key = 'DS4_V'
                    else:
                        continue
                    ds_counts[key] += 1

        nu = sum(us_counts.values())
        nu_pl = sum(1 for p in range(5) if us_counts[p] > 0)
        max_pl = max(us_counts.values()) if nu > 0 else 0

        # Clean shower selection: nu >= 6, >= 3 active planes, max/plane >= 2
        if nu >= 6 and nu_pl >= 3 and max_pl >= 2:
            events.append({
                'us_total': nu,
                'us_planes': us_counts,
                'ds_total': sum(ds_counts.values()),
                'ds_planes': ds_counts,
            })

    return events


def extract_mc_showers(
    file_list: List[str],
    physical_th_map: Dict[int, float],
    max_events: int = 4000,
) -> List[Dict[str, Any]]:
    """
    Extract MC events applying:
    1. Uncorrected: E_th = 0
    2. Flat nominal: E_th = 130 keV (US) / 22.1 keV (DS)
    3. Physical MinPE: E_th from channel-dependent physical model with k-factors
    Selection applied on physical model hits to match real data shower qualification.
    """
    chain = ROOT.TChain("cbmsim")
    for f in file_list:
        chain.Add(f)
    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("MuFilterPoint*", 1)

    n_entries = min(max_events, chain.GetEntries())
    events = []

    for iev in range(n_entries):
        chain.GetEntry(iev)
        pts = getattr(chain, "MuFilterPoint", None)
        if not pts:
            continue

        bar_edep: Dict[int, float] = {}
        for pt in pts:
            did = pt.GetDetectorID()
            bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss()

        # Counts for the 3 simulation models
        counts_uncorr_us = {p: 0 for p in range(5)}
        counts_uncorr_ds = {'DS1_H': 0, 'DS1_V': 0, 'DS2_H': 0, 'DS2_V': 0, 'DS3_H': 0, 'DS3_V': 0, 'DS4_V': 0}

        counts_flat_us = {p: 0 for p in range(5)}
        counts_flat_ds = {'DS1_H': 0, 'DS1_V': 0, 'DS2_H': 0, 'DS2_V': 0, 'DS3_H': 0, 'DS3_V': 0, 'DS4_V': 0}

        counts_phys_us = {p: 0 for p in range(5)}
        counts_phys_ds = {'DS1_H': 0, 'DS1_V': 0, 'DS2_H': 0, 'DS2_V': 0, 'DS3_H': 0, 'DS3_V': 0, 'DS4_V': 0}

        for did, edep in bar_edep.items():
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000

            th_phys = physical_th_map.get(did, 0.000300)

            if sub == 2 and 0 <= pl < 5:
                # Uncorrected
                counts_uncorr_us[pl] += 1
                # Flat
                if edep > E_TH_FLAT_US:
                    counts_flat_us[pl] += 1
                # Physical MinPE
                if edep > th_phys:
                    counts_phys_us[pl] += 1

            elif sub == 3:
                if pl == 0:
                    ds_key = 'DS1_H' if bar < 60 else 'DS1_V'
                elif pl == 1:
                    ds_key = 'DS2_H' if bar < 60 else 'DS2_V'
                elif pl == 2:
                    ds_key = 'DS3_H' if bar < 60 else 'DS3_V'
                elif pl == 3:
                    ds_key = 'DS4_V'
                else:
                    continue

                counts_uncorr_ds[ds_key] += 1
                if edep > E_TH_FLAT_DS:
                    counts_flat_ds[ds_key] += 1
                if edep > th_phys:
                    counts_phys_ds[ds_key] += 1

        nu_p = sum(counts_phys_us.values())
        nu_pl_p = sum(1 for p in range(5) if counts_phys_us[p] > 0)
        max_pl_p = max(counts_phys_us.values()) if nu_p > 0 else 0

        # Shower qualification based on physical model hits
        if nu_p >= 6 and nu_pl_p >= 3 and max_pl_p >= 2:
            events.append({
                'phys': {
                    'us_total': nu_p,
                    'us_planes': counts_phys_us,
                    'ds_total': sum(counts_phys_ds.values()),
                    'ds_planes': counts_phys_ds,
                },
                'flat': {
                    'us_total': sum(counts_flat_us.values()),
                    'us_planes': counts_flat_us,
                    'ds_total': sum(counts_flat_ds.values()),
                    'ds_planes': counts_flat_ds,
                },
                'uncorr': {
                    'us_total': sum(counts_uncorr_us.values()),
                    'us_planes': counts_uncorr_us,
                    'ds_total': sum(counts_uncorr_ds.values()),
                    'ds_planes': counts_uncorr_ds,
                },
            })

    return events


def main():
    parser = argparse.ArgumentParser(description="Testbeam Validation of Physical MinPE Model")
    parser.add_argument("--max-events", type=int, default=3000, help="Max events per beam energy")
    parser.add_argument("--out-dir", default="plots", help="Directory for output plots")
    parser.add_argument("--artifact-dir", default="/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)

    print("================================================================================")
    print("SND@LHC Testbeam 2023 Validation: Physical MinPE Threshold Model")
    print("================================================================================")

    # 1. Initialize MinPE and precompute physical thresholds
    calc = MinPE()
    print("Precomputing channel-by-channel physical energy thresholds...")
    phys_th_map = precompute_physical_thresholds(calc)

    us_vals = [phys_th_map[did] * 1e6 for did in phys_th_map if did // 10000 == 2]
    dsh_vals = [phys_th_map[did] * 1e6 for did in phys_th_map if did // 10000 == 3 and not calc.is_vertical(did)]
    dsv_vals = [phys_th_map[did] * 1e6 for did in phys_th_map if did // 10000 == 3 and calc.is_vertical(did)]
    print(f"  US Bars Thresholds  : mean = {np.mean(us_vals):.1f} keV, std = {np.std(us_vals):.1f} keV [min={min(us_vals):.1f}, max={max(us_vals):.1f}]")
    print(f"  DS Horiz Thresholds : mean = {np.mean(dsh_vals):.1f} keV, std = {np.std(dsh_vals):.1f} keV [min={min(dsh_vals):.1f}, max={max(dsh_vals):.1f}]")
    print(f"  DS Vert Thresholds  : mean = {np.mean(dsv_vals):.1f} keV, std = {np.std(dsv_vals):.1f} keV [min={min(dsv_vals):.1f}, max={max(dsv_vals):.1f}]")

    keys_order = [
        ('us_total', 'Total US Hits', 50, 0, 50),
        ('us_pl_0', 'US Plane 0', 12, 0, 12),
        ('us_pl_1', 'US Plane 1', 12, 0, 12),
        ('us_pl_2', 'US Plane 2', 12, 0, 12),
        ('us_pl_3', 'US Plane 3', 12, 0, 12),
        ('us_pl_4', 'US Plane 4', 12, 0, 12),
        ('ds_total', 'Total DS Hits', 40, 0, 40),
        ('ds_pl_DS1_H', 'DS1 H Hits', 10, 0, 10),
        ('ds_pl_DS1_V', 'DS1 V Hits', 10, 0, 10),
    ]

    summary_results = {}
    rfile = ROOT.TFile("output/tb_min_pe_validation.root", "RECREATE")

    for energy in [100, 140, 180, 240, 300]:
        cfg = CONFIGS[energy]
        print(f"\n>>> Processing {cfg['name']} (Energy = {energy} GeV)...")

        t0 = time.time()
        data_ev = extract_data_showers(cfg['data_files'], max_events=args.max_events)
        mc_ev = extract_mc_showers(cfg['mc_files'], phys_th_map, max_events=args.max_events)
        print(f"    Data events: {len(data_ev)} | MC events: {len(mc_ev)} in {time.time()-t0:.2f} s")

        d_arrs = {
            'us_total': np.array([e['us_total'] for e in data_ev]),
            'ds_total': np.array([e['ds_total'] for e in data_ev]),
            **{f'us_pl_{p}': np.array([e['us_planes'][p] for e in data_ev]) for p in range(5)},
            **{f'ds_pl_{k}': np.array([e['ds_planes'][k] for e in data_ev]) for k in ['DS1_H', 'DS1_V', 'DS2_H', 'DS2_V', 'DS3_H', 'DS3_V', 'DS4_V']},
        }

        u_arrs = {
            'us_total': np.array([e['uncorr']['us_total'] for e in mc_ev]),
            'ds_total': np.array([e['uncorr']['ds_total'] for e in mc_ev]),
            **{f'us_pl_{p}': np.array([e['uncorr']['us_planes'][p] for e in mc_ev]) for p in range(5)},
            **{f'ds_pl_{k}': np.array([e['uncorr']['ds_planes'][k] for e in mc_ev]) for k in ['DS1_H', 'DS1_V', 'DS2_H', 'DS2_V', 'DS3_H', 'DS3_V', 'DS4_V']},
        }

        f_arrs = {
            'us_total': np.array([e['flat']['us_total'] for e in mc_ev]),
            'ds_total': np.array([e['flat']['ds_total'] for e in mc_ev]),
            **{f'us_pl_{p}': np.array([e['flat']['us_planes'][p] for e in mc_ev]) for p in range(5)},
            **{f'ds_pl_{k}': np.array([e['flat']['ds_planes'][k] for e in mc_ev]) for k in ['DS1_H', 'DS1_V', 'DS2_H', 'DS2_V', 'DS3_H', 'DS3_V', 'DS4_V']},
        }

        p_arrs = {
            'us_total': np.array([e['phys']['us_total'] for e in mc_ev]),
            'ds_total': np.array([e['phys']['ds_total'] for e in mc_ev]),
            **{f'us_pl_{p}': np.array([e['phys']['us_planes'][p] for e in mc_ev]) for p in range(5)},
            **{f'ds_pl_{k}': np.array([e['phys']['ds_planes'][k] for e in mc_ev]) for k in ['DS1_H', 'DS1_V', 'DS2_H', 'DS2_V', 'DS3_H', 'DS3_V', 'DS4_V']},
        }

        w_uncorr, w_flat, w_phys = {}, {}, {}
        m_data, m_uncorr, m_flat, m_phys = {}, {}, {}, {}

        for k, _, _, _, _ in keys_order:
            d = d_arrs[k]
            u = u_arrs[k]
            f = f_arrs[k]
            p = p_arrs[k]

            m_data[k] = float(np.mean(d)) if len(d) > 0 else 0.0
            m_uncorr[k] = float(np.mean(u)) if len(u) > 0 else 0.0
            m_flat[k] = float(np.mean(f)) if len(f) > 0 else 0.0
            m_phys[k] = float(np.mean(p)) if len(p) > 0 else 0.0

            w_uncorr[k] = float(wasserstein_distance(d, u)) if len(d) > 0 and len(u) > 0 else 0.0
            w_flat[k] = float(wasserstein_distance(d, f)) if len(d) > 0 and len(f) > 0 else 0.0
            w_phys[k] = float(wasserstein_distance(d, p)) if len(d) > 0 and len(p) > 0 else 0.0

        summary_results[energy] = {
            'energy': energy,
            'name': cfg['name'],
            'n_data': len(data_ev),
            'n_mc': len(mc_ev),
            'means': {'data': m_data, 'uncorr': m_uncorr, 'flat': m_flat, 'phys': m_phys},
            'w1': {'uncorr': w_uncorr, 'flat': w_flat, 'phys': w_phys},
        }

        print(f"    US Total Mean: Data = {m_data['us_total']:.2f} | Uncorr = {m_uncorr['us_total']:.2f} | Flat = {m_flat['us_total']:.2f} | MinPE = {m_phys['us_total']:.2f}")
        print(f"    US Total W1  : Uncorr = {w_uncorr['us_total']:.2f} | Flat = {w_flat['us_total']:.2f} | MinPE = {w_phys['us_total']:.2f} (Δ = {w_flat['us_total'] - w_phys['us_total']:+.2f})")
        print(f"    DS Total Mean: Data = {m_data['ds_total']:.2f} | Uncorr = {m_uncorr['ds_total']:.2f} | Flat = {m_flat['ds_total']:.2f} | MinPE = {m_phys['ds_total']:.2f}")
        print(f"    DS Total W1  : Uncorr = {w_uncorr['ds_total']:.2f} | Flat = {w_flat['ds_total']:.2f} | MinPE = {w_phys['ds_total']:.2f} (Δ = {w_flat['ds_total'] - w_phys['ds_total']:+.2f})")

        # -------------------------------------------------------------
        # Create Multi-Panel ROOT Canvas
        # -------------------------------------------------------------
        c_dist = ROOT.TCanvas(f"c_tb_minpe_{energy}gev", f"Hit Distributions: Data vs MinPE ({cfg['name']})", 1600, 1200)
        c_dist.Divide(3, 3)
        store = []

        for idx, (k, title, nbins, xmin, xmax) in enumerate(keys_order):
            c_dist.cd(idx + 1)
            ROOT.gPad.SetGrid(1, 1)
            ROOT.gPad.SetLeftMargin(0.12)
            ROOT.gPad.SetRightMargin(0.05)
            ROOT.gPad.SetBottomMargin(0.12)

            h_data = ROOT.TH1D(f"h_d_{k}_{energy}", f"{title} ({cfg['name']});Number of Hits;Normalized Density", nbins, xmin, xmax)
            h_uncorr = ROOT.TH1D(f"h_u_{k}_{energy}", f"Uncorr MC;Number of Hits;Normalized Density", nbins, xmin, xmax)
            h_flat = ROOT.TH1D(f"h_f_{k}_{energy}", f"Flat MC;Number of Hits;Normalized Density", nbins, xmin, xmax)
            h_phys = ROOT.TH1D(f"h_p_{k}_{energy}", f"MinPE MC;Number of Hits;Normalized Density", nbins, xmin, xmax)

            for v in d_arrs[k]: h_data.Fill(v)
            for v in u_arrs[k]: h_uncorr.Fill(v)
            for v in f_arrs[k]: h_flat.Fill(v)
            for v in p_arrs[k]: h_phys.Fill(v)

            if h_data.Integral() > 0: h_data.Scale(1.0 / h_data.Integral())
            if h_uncorr.Integral() > 0: h_uncorr.Scale(1.0 / h_uncorr.Integral())
            if h_flat.Integral() > 0: h_flat.Scale(1.0 / h_flat.Integral())
            if h_phys.Integral() > 0: h_phys.Scale(1.0 / h_phys.Integral())

            h_data.SetLineColor(ROOT.kBlack)
            h_data.SetLineWidth(3)

            h_uncorr.SetLineColor(ROOT.kBlue + 1)
            h_uncorr.SetLineStyle(2)
            h_uncorr.SetLineWidth(2)

            h_flat.SetLineColor(ROOT.kGray + 2)
            h_flat.SetLineStyle(3)
            h_flat.SetLineWidth(2)

            h_phys.SetLineColor(ROOT.kRed + 1)
            h_phys.SetLineWidth(3)

            max_y = max(h_data.GetMaximum(), h_uncorr.GetMaximum(), h_flat.GetMaximum(), h_phys.GetMaximum(), 0.05) * 1.35
            h_data.SetMaximum(max_y)
            h_data.SetMinimum(0)

            h_data.Draw("HIST")
            h_uncorr.Draw("HIST SAME")
            h_flat.Draw("HIST SAME")
            h_phys.Draw("HIST SAME")

            leg = ROOT.TLegend(0.35, 0.62, 0.93, 0.88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetTextSize(0.034)
            leg.AddEntry(h_data, f"Data (μ={m_data[k]:.1f})", "l")
            leg.AddEntry(h_uncorr, f"Uncorr E_{{th}}=0 (W_{{1}}={w_uncorr[k]:.2f})", "l")
            leg.AddEntry(h_flat, f"Flat Nominal (W_{{1}}={w_flat[k]:.2f})", "l")
            leg.AddEntry(h_phys, f"MinPE Calib (W_{{1}}={w_phys[k]:.2f})", "l")
            leg.Draw()

            store.extend([h_data, h_uncorr, h_flat, h_phys, leg])

        c_dist.Write()
        out_png = f"{args.out_dir}/tb_minpe_{energy}gev_hit_distributions.png"
        c_dist.SaveAs(out_png)
        c_dist.SaveAs(os.path.join(args.artifact_dir, f"tb_minpe_{energy}gev_hit_distributions.png"))
        store.append(c_dist)

    rfile.Close()

    # Save summary JSON
    json_path = "output/tb_min_pe_validation.json"
    with open(json_path, "w") as jf:
        json.dump(summary_results, jf, indent=2)

    # Generate Summary Wasserstein Distance Comparison Figure across energies
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    energies = [100, 140, 180, 240, 300]
    us_w1_uncorr = [summary_results[e]['w1']['uncorr']['us_total'] for e in energies]
    us_w1_flat = [summary_results[e]['w1']['flat']['us_total'] for e in energies]
    us_w1_phys = [summary_results[e]['w1']['phys']['us_total'] for e in energies]

    ds_w1_uncorr = [summary_results[e]['w1']['uncorr']['ds_total'] for e in energies]
    ds_w1_flat = [summary_results[e]['w1']['flat']['ds_total'] for e in energies]
    ds_w1_phys = [summary_results[e]['w1']['phys']['ds_total'] for e in energies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.plot(energies, us_w1_uncorr, 'o--', color='#0275D8', lw=2.0, ms=7, label=f'Uncorrected ($E_{{th}}=0$) [Mean={np.mean(us_w1_uncorr):.2f}]')
    ax1.plot(energies, us_w1_flat, 's:', color='#6c757d', lw=2.0, ms=7, label=f'Flat Nominal ($130$ keV) [Mean={np.mean(us_w1_flat):.2f}]')
    ax1.plot(energies, us_w1_phys, 'D-', color='#D9534F', lw=2.5, ms=8, label=f'MinPE Calibrated ($k$-model) [Mean={np.mean(us_w1_phys):.2f}]')
    ax1.set_title('Upstream (US) Total Hits: Wasserstein Distance $W_1$ to Data', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Beam Energy [GeV]', fontsize=11)
    ax1.set_ylabel('$W_1$ (Earth Mover\'s Distance) [hits]', fontsize=11)
    ax1.set_xticks(energies)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(frameon=True, fontsize=9.5)

    ax2.plot(energies, ds_w1_uncorr, 'o--', color='#0275D8', lw=2.0, ms=7, label=f'Uncorrected ($E_{{th}}=0$) [Mean={np.mean(ds_w1_uncorr):.2f}]')
    ax2.plot(energies, ds_w1_flat, 's:', color='#6c757d', lw=2.0, ms=7, label=f'Flat Nominal ($22.1$ keV) [Mean={np.mean(ds_w1_flat):.2f}]')
    ax2.plot(energies, ds_w1_phys, 'D-', color='#D9534F', lw=2.5, ms=8, label=f'MinPE Calibrated ($k$-model) [Mean={np.mean(ds_w1_phys):.2f}]')
    ax2.set_title('Downstream (DS) Total Hits: Wasserstein Distance $W_1$ to Data', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Beam Energy [GeV]', fontsize=11)
    ax2.set_ylabel('$W_1$ (Earth Mover\'s Distance) [hits]', fontsize=11)
    ax2.set_xticks(energies)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(frameon=True, fontsize=9.5)

    plt.tight_layout()
    summary_png = f"{args.out_dir}/tb_minpe_wasserstein_summary.png"
    plt.savefig(summary_png, dpi=300)
    plt.savefig(os.path.join(args.artifact_dir, "tb_minpe_wasserstein_summary.png"), dpi=300)
    plt.close()

    print("\n================================================================================")
    print("Testbeam Validation Complete!")
    print(f"Summary PNG saved to: {summary_png}")
    print(f"ROOT file saved to  : output/tb_min_pe_validation.root")
    print("================================================================================")


if __name__ == "__main__":
    main()
