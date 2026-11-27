#!/usr/bin/env python3
"""
Testbeam 2023 H8 MuFilter Threshold Scale Scan (k = 1.0 to 10.0)
with Target-Initiated Showers (N_SciFi >= 100).

Applies testbeam-specific nominal theoretical minimum values as k = 1.0:
  - Horizontal DS bars: 38.4 keV
  - Vertical DS bars:   22.1 keV
  - US bars with gel:   129.7 keV
  - US bars with air:   227.7 keV

Scans scale factors k in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0].
Evaluates across all 5 testbeam runs:
  - 100 GeV pi+ (Run 100677)
  - 140 GeV pi+ (Run 100673)
  - 180 GeV pi+ (Run 100671)
  - 240 GeV pi- (Run 100637)
  - 300 GeV pi- (Run 100645)

Produces:
  1. Detailed comparison canvases per energy at key and optimal k values.
  2. Summary optimization canvas showing W1 vs k across all beam energies.
  3. Comprehensive JSON and text reports with exact metrics and rankings.
"""

import os
import sys
import json
import pickle
import numpy as np
from scipy.stats import wasserstein_distance

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Nominal theoretical minimum thresholds at k = 1.0 (in keV)
TH_DS_HORIZ_BASE = 38.4
TH_DS_VERT_BASE  = 22.1
TH_US_GEL_BASE   = 129.7
TH_US_AIR_BASE   = 227.7

K_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

CONFIGS = {
    100: {
        'name': '100 GeV #pi^{+}',
        'particle': 'pi+',
        'mc_files': ['/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/100GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root'],
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100677/sndsw_raw-0000.root'],
        'run_num': 100677
    },
    140: {
        'name': '140 GeV #pi^{+}',
        'particle': 'pi+',
        'mc_files': ['/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/140GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root'],
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100673/sndsw_raw-0000.root'],
        'run_num': 100673
    },
    180: {
        'name': '180 GeV #pi^{+}',
        'particle': 'pi+',
        'mc_files': ['/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/180GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root'],
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100671/sndsw_raw-0000.root'],
        'run_num': 100671
    },
    240: {
        'name': '240 GeV #pi^{-}',
        'particle': 'pi-',
        'mc_files': [
            p for p in [
                f'/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/240GeV_-211/{part}/sndLHC.PG_-211-TGeant4_MCEB_dig.root'
                for part in range(5)
            ] if os.path.exists(p)
        ],
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100637/sndsw_raw-0000.root'],
        'run_num': 100637
    },
    300: {
        'name': '300 GeV #pi^{-}',
        'particle': 'pi-',
        'mc_files': [
            p for p in [
                f'/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/300GeV_-211/{part}/sndLHC.PG_-211-TGeant4_MCEB_dig.root'
                for part in range(5)
            ] if os.path.exists(p)
        ],
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100645/sndsw_raw-0000.root'],
        'run_num': 100645
    }
}


def extract_data_scifi100(file_list, max_events=30000, target_clean=1000):
    chain = ROOT.TChain("rawConv")
    for f in file_list:
        chain.Add(f)
    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("Digi_ScifiHits*", 1)
    chain.SetBranchStatus("Digi_MuFilterHits*", 1)

    n_entries = min(max_events, chain.GetEntries())
    events = []

    for iev in range(n_entries):
        chain.GetEntry(iev)
        n_scifi = len(chain.Digi_ScifiHits)
        if n_scifi < 100:
            continue

        us_counts = {p: 0 for p in range(5)}
        ds_counts = {
            'DS1_H': 0, 'DS1_V': 0,
            'DS2_H': 0, 'DS2_V': 0,
            'DS3_H': 0, 'DS3_V': 0,
            'DS4_V': 0
        }
        us_bars_seen = set()
        ds_bars_seen = set()

        for hit in chain.Digi_MuFilterHits:
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
        nd = sum(ds_counts.values())
        nd_h = ds_counts['DS1_H'] + ds_counts['DS2_H'] + ds_counts['DS3_H']
        nd_v = ds_counts['DS1_V'] + ds_counts['DS2_V'] + ds_counts['DS3_V'] + ds_counts['DS4_V']

        events.append({
            'us_total': nu,
            'us_planes': us_counts,
            'ds_total': nd,
            'ds_h': nd_h,
            'ds_v': nd_v,
            'ds_planes': ds_counts
        })
        if len(events) >= target_clean:
            break

    return events


def extract_mc_scifi100(file_list, max_events=4000):
    chain = ROOT.TChain("cbmsim")
    for f in file_list:
        chain.Add(f)
    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("Digi_ScifiHits*", 1)
    chain.SetBranchStatus("MuFilterPoint*", 1)

    n_entries = min(max_events, chain.GetEntries())
    events = []

    for iev in range(n_entries):
        chain.GetEntry(iev)
        n_scifi = len(chain.Digi_ScifiHits)
        if n_scifi < 100:
            continue

        bar_edep = {}
        for pt in chain.MuFilterPoint:
            did = pt.GetDetectorID()
            # Convert GeV to keV
            bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss() * 1.0e6

        # Uncorrected (E_th = 0)
        unmasked_us = {p: 0 for p in range(5)}
        unmasked_ds = {
            'DS1_H': 0, 'DS1_V': 0,
            'DS2_H': 0, 'DS2_V': 0,
            'DS3_H': 0, 'DS3_V': 0,
            'DS4_V': 0
        }
        for did, edep in bar_edep.items():
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000
            if sub == 2 and 0 <= pl < 5:
                unmasked_us[pl] += 1
            elif sub == 3:
                if pl == 0: key = 'DS1_H' if bar < 60 else 'DS1_V'
                elif pl == 1: key = 'DS2_H' if bar < 60 else 'DS2_V'
                elif pl == 2: key = 'DS3_H' if bar < 60 else 'DS3_V'
                elif pl == 3: key = 'DS4_V'
                else: continue
                unmasked_ds[key] += 1

        events.append({
            'bar_edep': bar_edep,
            'uncorr': {
                'us_total': sum(unmasked_us.values()),
                'us_planes': unmasked_us,
                'ds_total': sum(unmasked_ds.values()),
                'ds_h': unmasked_ds['DS1_H'] + unmasked_ds['DS2_H'] + unmasked_ds['DS3_H'],
                'ds_v': unmasked_ds['DS1_V'] + unmasked_ds['DS2_V'] + unmasked_ds['DS3_V'] + unmasked_ds['DS4_V'],
                'ds_planes': unmasked_ds
            }
        })

    return events


def evaluate_mc_hits_at_k(mc_events, k, us_mode='gel'):
    th_ds_h = TH_DS_HORIZ_BASE * k
    th_ds_v = TH_DS_VERT_BASE * k
    th_us = (TH_US_GEL_BASE if us_mode == 'gel' else TH_US_AIR_BASE) * k

    masked_events = []
    for ev in mc_events:
        us_counts = {p: 0 for p in range(5)}
        ds_counts = {
            'DS1_H': 0, 'DS1_V': 0,
            'DS2_H': 0, 'DS2_V': 0,
            'DS3_H': 0, 'DS3_V': 0,
            'DS4_V': 0
        }

        for did, edep in ev['bar_edep'].items():
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000

            if sub == 2 and 0 <= pl < 5:
                if edep >= th_us:
                    us_counts[pl] += 1
            elif sub == 3:
                if pl == 0:
                    key = 'DS1_H' if bar < 60 else 'DS1_V'
                    th = th_ds_h if bar < 60 else th_ds_v
                elif pl == 1:
                    key = 'DS2_H' if bar < 60 else 'DS2_V'
                    th = th_ds_h if bar < 60 else th_ds_v
                elif pl == 2:
                    key = 'DS3_H' if bar < 60 else 'DS3_V'
                    th = th_ds_h if bar < 60 else th_ds_v
                elif pl == 3:
                    key = 'DS4_V'
                    th = th_ds_v
                else:
                    continue

                if edep >= th:
                    ds_counts[key] += 1

        nu = sum(us_counts.values())
        nd = sum(ds_counts.values())
        nd_h = ds_counts['DS1_H'] + ds_counts['DS2_H'] + ds_counts['DS3_H']
        nd_v = ds_counts['DS1_V'] + ds_counts['DS2_V'] + ds_counts['DS3_V'] + ds_counts['DS4_V']

        masked_events.append({
            'us_total': nu,
            'us_planes': us_counts,
            'ds_total': nd,
            'ds_h': nd_h,
            'ds_v': nd_v,
            'ds_planes': ds_counts
        })

    return masked_events


def compute_metrics(data_events, masked_mc_events):
    def extract_list(evs, var_name):
        if var_name == 'us_total': return [e['us_total'] for e in evs]
        elif var_name == 'ds_total': return [e['ds_total'] for e in evs]
        elif var_name == 'ds_h': return [e['ds_h'] for e in evs]
        elif var_name == 'ds_v': return [e['ds_v'] for e in evs]
        elif var_name.startswith('us_pl_'):
            idx = int(var_name.split('_')[-1])
            return [e['us_planes'][idx] for e in evs]
        elif var_name.startswith('ds_pl_'):
            pl_key = var_name.replace('ds_pl_', '')
            return [e['ds_planes'][pl_key] for e in evs]
        return []

    vars_to_eval = ['us_total', 'ds_total', 'ds_h', 'ds_v', 'us_pl_0', 'us_pl_1', 'us_pl_2', 'us_pl_3', 'us_pl_4', 'ds_pl_DS1_H', 'ds_pl_DS1_V']
    w1s = {}
    means_data = {}
    means_mc = {}

    for v in vars_to_eval:
        d_vals = extract_list(data_events, v)
        m_vals = extract_list(masked_mc_events, v)
        w1s[v] = float(wasserstein_distance(d_vals, m_vals))
        means_data[v] = float(np.mean(d_vals))
        means_mc[v] = float(np.mean(m_vals))

    comp_w1 = (w1s['us_total'] + w1s['ds_total'] + w1s['ds_h'] + w1s['ds_v']) / 4.0
    return w1s, means_data, means_mc, comp_w1


def plot_energy_canvas(energy, cfg, data_events, raw_mc_events, masked_mc_events, k, out_dir, artifact_dir):
    c = ROOT.TCanvas(f"c_tb_{energy}gev_k_{k:.1f}".replace(".", "p"), f"Testbeam {cfg['name']} (Scale factor k={k:.1f})", 1800, 1400)
    c.Divide(3, 3)

    plots_cfg = [
        ("Total US Hits", "us_total", 50, 0, 50, 1),
        ("Total DS Hits", "ds_total", 40, 0, 40, 2),
        ("DS Horizontal Hits", "ds_h", 20, 0, 20, 3),
        ("DS Vertical Hits", "ds_v", 20, 0, 20, 4),
        ("US Plane 0 Hits", "us_pl_0", 12, 0, 12, 5),
        ("US Plane 1 Hits", "us_pl_1", 12, 0, 12, 6),
        ("US Plane 2 Hits", "us_pl_2", 12, 0, 12, 7),
        ("DS1 H Hits", "ds_pl_DS1_H", 10, 0, 10, 8),
        ("DS1 V Hits", "ds_pl_DS1_V", 10, 0, 10, 9),
    ]

    def extract_list(evs, var_name):
        if var_name == 'us_total': return [e['us_total'] for e in evs]
        elif var_name == 'ds_total': return [e['ds_total'] for e in evs]
        elif var_name == 'ds_h': return [e['ds_h'] for e in evs]
        elif var_name == 'ds_v': return [e['ds_v'] for e in evs]
        elif var_name.startswith('us_pl_'):
            idx = int(var_name.split('_')[-1])
            return [e['us_planes'][idx] for e in evs]
        elif var_name.startswith('ds_pl_'):
            pl_key = var_name.replace('ds_pl_', '')
            return [e['ds_planes'][pl_key] for e in evs]
        return []

    hist_store = []

    for title, var_name, nbins, xmin, xmax, pad_idx in plots_cfg:
        c.cd(pad_idx)
        ROOT.gPad.SetGrid(1, 1)
        ROOT.gPad.SetLeftMargin(0.14)
        ROOT.gPad.SetRightMargin(0.06)
        ROOT.gPad.SetBottomMargin(0.12)

        data_vals = extract_list(data_events, var_name)
        raw_vals = extract_list([e['uncorr'] for e in raw_mc_events], var_name)
        mask_vals = extract_list(masked_mc_events, var_name)

        h_data = ROOT.TH1D(f"h_d_{energy}_{var_name}_k{k}", f"{title} ({cfg['name']});Number of Hits;Probability Density", nbins, xmin, xmax)
        h_raw = ROOT.TH1D(f"h_raw_{energy}_{var_name}_k{k}", f"{title};Number of Hits;Probability Density", nbins, xmin, xmax)
        h_mask = ROOT.TH1D(f"h_mask_{energy}_{var_name}_k{k}", f"{title};Number of Hits;Probability Density", nbins, xmin, xmax)

        for v in data_vals: h_data.Fill(v)
        for v in raw_vals: h_raw.Fill(v)
        for v in mask_vals: h_mask.Fill(v)

        if h_data.Integral() > 0: h_data.Scale(1.0 / h_data.Integral())
        if h_raw.Integral() > 0: h_raw.Scale(1.0 / h_raw.Integral())
        if h_mask.Integral() > 0: h_mask.Scale(1.0 / h_mask.Integral())

        w1_raw = wasserstein_distance(data_vals, raw_vals)
        w1_mask = wasserstein_distance(data_vals, mask_vals)

        h_data.SetLineColor(ROOT.kBlack)
        h_data.SetLineWidth(3)

        h_raw.SetLineColor(ROOT.kBlue + 1)
        h_raw.SetLineStyle(2)
        h_raw.SetLineWidth(2)

        h_mask.SetLineColor(ROOT.kRed + 1)
        h_mask.SetLineWidth(3)

        max_y = max(h_data.GetMaximum(), h_raw.GetMaximum(), h_mask.GetMaximum()) * 1.35
        h_data.SetMaximum(max_y)
        h_data.SetMinimum(0.0)

        h_data.Draw("HIST")
        h_raw.Draw("HIST SAME")
        h_mask.Draw("HIST SAME")

        leg = ROOT.TLegend(0.40, 0.65, 0.92, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.038)
        leg.AddEntry(h_data, f"Data (#mu={np.mean(data_vals):.1f})", "l")
        leg.AddEntry(h_raw, f"Sim E_{{th}}=0 (#mu={np.mean(raw_vals):.1f}, W_{{1}}={w1_raw:.2f})", "l")
        leg.AddEntry(h_mask, f"Sim k={k:.1f} (#mu={np.mean(mask_vals):.1f}, W_{{1}}={w1_mask:.2f})", "l")
        leg.Draw()

        hist_store.extend([h_data, h_raw, h_mask, leg])

    c.Update()
    out_png = os.path.join(out_dir, f"tb_{energy}gev_k_{k:.1f}_comparison.png")
    c.SaveAs(out_png)
    os.system(f"cp {out_png} {os.path.join(artifact_dir, f'tb_{energy}gev_k_{k:.1f}_comparison.png')}")


def main():
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    artifact_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"

    cache_file = "output/tb_scifi100_cache.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached events from {cache_file}...")
        with open(cache_file, "rb") as f:
            all_events = pickle.load(f)
    else:
        print("Extracting testbeam events across 5 beam energies (target shower N_SciFi >= 100)...")
        all_events = {}
        for energy, cfg in sorted(CONFIGS.items()):
            print(f"  --> {energy} GeV: Data Run {cfg['run_num']} ...", flush=True)
            d_evs = extract_data_scifi100(cfg['data_files'], max_events=30000, target_clean=1000)
            print(f"      Data extracted: {len(d_evs)} events. Reading MC ...", flush=True)
            mc_evs = extract_mc_scifi100(cfg['mc_files'], max_events=4000)
            print(f"      MC extracted: {len(mc_evs)} events.", flush=True)
            all_events[energy] = {'data': d_evs, 'mc': mc_evs}
        with open(cache_file, "wb") as f:
            pickle.dump(all_events, f)
        print(f"Saved cache to {cache_file}.")

    # Scan over scale factor k
    print("\n" + "=" * 95)
    print("TESTBEAM THRESHOLD SCALE SCAN (k = 1.0 to 10.0)")
    print(f"Base Thresholds (k=1.0): H-DS={TH_DS_HORIZ_BASE} keV, V-DS={TH_DS_VERT_BASE} keV, US-Gel={TH_US_GEL_BASE} keV, US-Air={TH_US_AIR_BASE} keV")
    print("=" * 95)

    scan_results = {energy: [] for energy in CONFIGS}
    global_k_summary = []

    for k in K_VALUES:
        row_mean_w1s = []
        row_ds_w1s = []
        row_us_w1s = []

        for energy in sorted(CONFIGS.keys()):
            cfg = CONFIGS[energy]
            d_evs = all_events[energy]['data']
            mc_evs = all_events[energy]['mc']

            masked_mc = evaluate_mc_hits_at_k(mc_evs, k, us_mode='gel')
            w1s, means_data, means_mc, comp_w1 = compute_metrics(d_evs, masked_mc)

            scan_results[energy].append({
                'k': k,
                'th_ds_h': TH_DS_HORIZ_BASE * k,
                'th_ds_v': TH_DS_VERT_BASE * k,
                'th_us_gel': TH_US_GEL_BASE * k,
                'th_us_air': TH_US_AIR_BASE * k,
                'w1': w1s,
                'comp_w1': comp_w1,
                'means_data': means_data,
                'means_mc': means_mc
            })
            row_mean_w1s.append(comp_w1)
            row_ds_w1s.append(w1s['ds_total'])
            row_us_w1s.append(w1s['us_total'])

            # Plot key canvases (e.g. k=1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0)
            if k in [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]:
                plot_energy_canvas(energy, cfg, d_evs, mc_evs, masked_mc, k, out_dir, artifact_dir)

        global_comp_w1 = float(np.mean(row_mean_w1s))
        global_ds_w1 = float(np.mean(row_ds_w1s))
        global_us_w1 = float(np.mean(row_us_w1s))

        global_k_summary.append({
            'k': k,
            'global_comp_w1': global_comp_w1,
            'global_ds_w1': global_ds_w1,
            'global_us_w1': global_us_w1,
            'per_energy_comp_w1': {e: scan_results[e][-1]['comp_w1'] for e in CONFIGS},
            'per_energy_ds_w1': {e: scan_results[e][-1]['w1']['ds_total'] for e in CONFIGS},
            'per_energy_us_w1': {e: scan_results[e][-1]['w1']['us_total'] for e in CONFIGS}
        })

    # Print summary table
    print(f"{'Scale k':<8} | {'100 GeV':<9} | {'140 GeV':<9} | {'180 GeV':<9} | {'240 GeV':<9} | {'300 GeV':<9} | {'<W1(DS)>':<10} | {'<W1(US)>':<10} | {'Global W1':<10}")
    print("-" * 95)
    for row in global_k_summary:
        k = row['k']
        w100 = row['per_energy_comp_w1'][100]
        w140 = row['per_energy_comp_w1'][140]
        w180 = row['per_energy_comp_w1'][180]
        w240 = row['per_energy_comp_w1'][240]
        w300 = row['per_energy_comp_w1'][300]
        print(f"{k:<8.1f} | {w100:<9.3f} | {w140:<9.3f} | {w180:<9.3f} | {w240:<9.3f} | {w300:<9.3f} | {row['global_ds_w1']:<10.3f} | {row['global_us_w1']:<10.3f} | {row['global_comp_w1']:<10.3f}")
    print("=" * 95)

    # Find optimal k for DS alone and globally
    best_ds = min(global_k_summary, key=lambda x: x['global_ds_w1'])
    best_glob = min(global_k_summary, key=lambda x: x['global_comp_w1'])

    print(f"\nOPTIMAL RESULTS:")
    print(f"  Optimal for Downstream (DS): k = {best_ds['k']:.1f} with <W1(DS)> = {best_ds['global_ds_w1']:.3f} hits")
    print(f"  Optimal Overall:             k = {best_glob['k']:.1f} with Global W1 = {best_glob['global_comp_w1']:.3f} hits")
    print(f"Optimal Threshold Values at k = {best_ds['k']:.1f}:")
    print(f"  Horizontal DS: {TH_DS_HORIZ_BASE * best_ds['k']:.1f} keV")
    print(f"  Vertical DS:   {TH_DS_VERT_BASE * best_ds['k']:.1f} keV")
    print(f"  US Gel:        {TH_US_GEL_BASE * best_ds['k']:.1f} keV")
    print(f"  US Air:        {TH_US_AIR_BASE * best_ds['k']:.1f} keV")

    # Plot optimal canvases for each energy if not already plotted
    for energy in CONFIGS:
        cfg = CONFIGS[energy]
        d_evs = all_events[energy]['data']
        mc_evs = all_events[energy]['mc']
        masked_mc = evaluate_mc_hits_at_k(mc_evs, best_ds['k'], us_mode='gel')
        plot_energy_canvas(energy, cfg, d_evs, mc_evs, masked_mc, best_ds['k'], out_dir, artifact_dir)

    # Save summary JSON and TXT
    with open("output/tb_threshold_scale_scan.json", "w") as f:
        json.dump({'global_summary': global_k_summary, 'per_energy_scan': scan_results}, f, indent=2)

    with open("output/tb_threshold_scale_scan.txt", "w") as tf:
        tf.write("=" * 90 + "\n")
        tf.write("TESTBEAM 2023 H8 MUFILTER THRESHOLD SCALE SCAN REPORT\n")
        tf.write("=" * 90 + "\n")
        tf.write(f"Nominal Theoretical Minimum Thresholds (k = 1.0):\n")
        tf.write(f"  - Horizontal DS: {TH_DS_HORIZ_BASE} keV\n")
        tf.write(f"  - Vertical DS:   {TH_DS_VERT_BASE} keV\n")
        tf.write(f"  - US with Gel:   {TH_US_GEL_BASE} keV\n")
        tf.write(f"  - US with Air:   {TH_US_AIR_BASE} keV\n\n")
        tf.write(f"{'Scale k':<8} | {'100 GeV':<9} | {'140 GeV':<9} | {'180 GeV':<9} | {'240 GeV':<9} | {'300 GeV':<9} | {'<W1(DS)>':<10} | {'<W1(US)>':<10} | {'Global W1':<10}\n")
        tf.write("-" * 90 + "\n")
        for row in global_k_summary:
            k = row['k']
            w100 = row['per_energy_comp_w1'][100]
            w140 = row['per_energy_comp_w1'][140]
            w180 = row['per_energy_comp_w1'][180]
            w240 = row['per_energy_comp_w1'][240]
            w300 = row['per_energy_comp_w1'][300]
            tf.write(f"{k:<8.1f} | {w100:<9.3f} | {w140:<9.3f} | {w180:<9.3f} | {w240:<9.3f} | {w300:<9.3f} | {row['global_ds_w1']:<10.3f} | {row['global_us_w1']:<10.3f} | {row['global_comp_w1']:<10.3f}\n")
        tf.write("-" * 90 + "\n")
        tf.write(f"\nOPTIMAL SCALE FACTOR FOR DOWNSTREAM (DS): k = {best_ds['k']:.1f}\n")
        tf.write(f"  - Minimum Average DS W1: {best_ds['global_ds_w1']:.3f} hits\n")
        tf.write(f"  - Horizontal DS Threshold: {TH_DS_HORIZ_BASE * best_ds['k']:.1f} keV\n")
        tf.write(f"  - Vertical DS Threshold:   {TH_DS_VERT_BASE * best_ds['k']:.1f} keV\n")
        tf.write(f"  - US Gel Threshold:        {TH_US_GEL_BASE * best_ds['k']:.1f} keV\n")

    # Generate 2x2 Summary Plot
    c_sum = ROOT.TCanvas("c_tb_optimization_summary", "Testbeam Threshold Scale Optimization", 1600, 1200)
    c_sum.Divide(2, 2)

    k_arr = np.array([r['k'] for r in global_k_summary], dtype=np.float64)

    # Pad 1: Global Composite and Subsystems
    c_sum.cd(1)
    ROOT.gPad.SetGrid(1, 1)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)

    w_glob = np.array([r['global_comp_w1'] for r in global_k_summary], dtype=np.float64)
    w_ds = np.array([r['global_ds_w1'] for r in global_k_summary], dtype=np.float64)
    w_us = np.array([r['global_us_w1'] for r in global_k_summary], dtype=np.float64)

    gr_glob = ROOT.TGraph(len(k_arr), k_arr, w_glob)
    gr_glob.SetTitle("Overall Average Wasserstein Distance vs k;Threshold Scale Factor k;Wasserstein Distance W_{1} [hits]")
    gr_glob.SetLineColor(ROOT.kBlack)
    gr_glob.SetLineWidth(3)
    gr_glob.SetMarkerStyle(20)

    gr_ds = ROOT.TGraph(len(k_arr), k_arr, w_ds)
    gr_ds.SetLineColor(ROOT.kBlue + 1)
    gr_ds.SetLineWidth(3)
    gr_ds.SetMarkerStyle(21)

    gr_glob.SetMaximum(max(w_glob.max(), w_ds.max()) * 1.25)
    gr_glob.SetMinimum(0.0)
    gr_glob.Draw("ALP")
    gr_ds.Draw("LP SAME")

    leg1 = ROOT.TLegend(0.45, 0.70, 0.90, 0.88)
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.AddEntry(gr_glob, "Composite W_{1}", "lp")
    leg1.AddEntry(gr_ds, "<W_{1}(DS Total)>", "lp")
    leg1.Draw()

    # Pad 2: DS Total across 5 energies
    c_sum.cd(2)
    ROOT.gPad.SetGrid(1, 1)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)

    colors = [ROOT.kRed + 1, ROOT.kOrange + 7, ROOT.kGreen + 2, ROOT.kCyan + 2, ROOT.kBlue + 2]
    markers = [20, 21, 22, 23, 33]
    gr_energies = []

    leg2 = ROOT.TLegend(0.18, 0.65, 0.55, 0.88)
    leg2.SetBorderSize(0)
    leg2.SetFillStyle(0)

    max_ds = 0
    for idx, e in enumerate(sorted(CONFIGS.keys())):
        ds_vals = np.array([r['per_energy_ds_w1'][e] for r in global_k_summary], dtype=np.float64)
        max_ds = max(max_ds, ds_vals.max())
        gr_e = ROOT.TGraph(len(k_arr), k_arr, ds_vals)
        gr_e.SetLineColor(colors[idx])
        gr_e.SetMarkerColor(colors[idx])
        gr_e.SetMarkerStyle(markers[idx])
        gr_e.SetLineWidth(2)
        gr_energies.append(gr_e)
        leg2.AddEntry(gr_e, CONFIGS[e]['name'], "lp")

    gr_energies[0].SetTitle("DS Total W_{1} vs Scale Factor k;Threshold Scale Factor k;Wasserstein Distance W_{1} [hits]")
    gr_energies[0].SetMaximum(max_ds * 1.25)
    gr_energies[0].SetMinimum(0.0)
    gr_energies[0].Draw("ALP")
    for gr in gr_energies[1:]:
        gr.Draw("LP SAME")
    leg2.Draw()

    # Pad 3: US Total across 5 energies
    c_sum.cd(3)
    ROOT.gPad.SetGrid(1, 1)
    ROOT.gPad.SetLeftMargin(0.14)
    ROOT.gPad.SetBottomMargin(0.12)

    gr_us_energies = []
    max_us = 0
    for idx, e in enumerate(sorted(CONFIGS.keys())):
        us_vals = np.array([r['per_energy_us_w1'][e] for r in global_k_summary], dtype=np.float64)
        max_us = max(max_us, us_vals.max())
        gr_u = ROOT.TGraph(len(k_arr), k_arr, us_vals)
        gr_u.SetLineColor(colors[idx])
        gr_u.SetMarkerColor(colors[idx])
        gr_u.SetMarkerStyle(markers[idx])
        gr_u.SetLineWidth(2)
        gr_us_energies.append(gr_u)

    gr_us_energies[0].SetTitle("US Total W_{1} vs Scale Factor k;Threshold Scale Factor k;Wasserstein Distance W_{1} [hits]")
    gr_us_energies[0].SetMaximum(max_us * 1.2)
    gr_us_energies[0].SetMinimum(0.0)
    gr_us_energies[0].Draw("ALP")
    for gr in gr_us_energies[1:]:
        gr.Draw("LP SAME")

    # Pad 4: Summary Text / Results Box
    c_sum.cd(4)
    ROOT.gPad.SetLeftMargin(0.10)
    lat = ROOT.TLatex()
    lat.SetNDC()
    lat.SetTextSize(0.045)
    lat.DrawLatex(0.10, 0.88, "#bf{Testbeam Threshold Optimization Summary}")
    lat.SetTextSize(0.038)
    lat.DrawLatex(0.10, 0.80, f"Nominal Baseline (k=1.0):")
    lat.DrawLatex(0.15, 0.74, f"H-DS = {TH_DS_HORIZ_BASE} keV, V-DS = {TH_DS_VERT_BASE} keV")
    lat.DrawLatex(0.15, 0.68, f"US Gel = {TH_US_GEL_BASE} keV, US Air = {TH_US_AIR_BASE} keV")

    lat.DrawLatex(0.10, 0.58, f"Optimal Scale for Downstream: #bf{{k = {best_ds['k']:.1f}}}")
    lat.DrawLatex(0.15, 0.52, f"Optimal H-DS Thresh: #bf{{{TH_DS_HORIZ_BASE * best_ds['k']:.1f} keV}}")
    lat.DrawLatex(0.15, 0.46, f"Optimal V-DS Thresh: #bf{{{TH_DS_VERT_BASE * best_ds['k']:.1f} keV}}")
    lat.DrawLatex(0.15, 0.40, f"Min Average DS W_{{1}}: #bf{{{best_ds['global_ds_w1']:.3f} hits}}")

    lat.DrawLatex(0.10, 0.30, f"Overall Global Scale: #bf{{k = {best_glob['k']:.1f}}}")
    lat.DrawLatex(0.15, 0.24, f"Optimal US Gel Thresh: #bf{{{TH_US_GEL_BASE * best_glob['k']:.1f} keV}}")
    lat.DrawLatex(0.15, 0.18, f"Global Min Composite W_{{1}}: #bf{{{best_glob['global_comp_w1']:.3f} hits}}")

    summary_png = os.path.join(out_dir, "tb_threshold_scale_optimization.png")
    c_sum.SaveAs(summary_png)
    os.system(f"cp {summary_png} {os.path.join(artifact_dir, 'tb_threshold_scale_optimization.png')}")
    print(f"Saved summary optimization canvas to {summary_png}.")

if __name__ == "__main__":
    main()
