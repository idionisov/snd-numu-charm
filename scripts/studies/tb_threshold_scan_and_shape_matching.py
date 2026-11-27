#!/usr/bin/env python3
"""
Testbeam 2023 H8 MuFilter Threshold Scan & Distribution Shape Matching
Author: Antigravity Assistant

Energies: 100 GeV, 140 GeV, 180 GeV (pi+), 240 GeV, 300 GeV (pi-)
Data: Commissioning testbeam June 2023 H8 (3 target walls in)
Simulations: 2mmRangeCut _MCEB_dig samples

Features:
1. Apples-to-apples matched shower event selection (N_US >= 4, N_US_planes >= 3).
2. Energy threshold scan from 0 to 20 MeV in steps of 0.25 MeV (81 steps).
3. Monotonically decreasing hit expectation curves vs threshold with data crossing points.
4. Distribution shape matching using Wasserstein-1 distance and KS statistic.
5. Multi-panel canvases comparing Data, Uncorrected MC (E_th = 0), and Corrected MC (optimal E_th).
"""

import os
import sys
import time
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Energy configuration with verified beam-centered, 3-walls ALL-IN runs
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
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100674/sndsw_raw-0000.root'],
        'run_num': 100674
    },
    180: {
        'name': '180 GeV #pi^{+}',
        'particle': 'pi+',
        'mc_files': ['/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/180GeV_211/sndLHC.PG_211-TGeant4_MCEB_dig.root'],
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100635/sndsw_raw-0000.root'],
        'run_num': 100635
    },
    240: {
        'name': '240 GeV #pi^{-}',
        'particle': 'pi-',
        'mc_files': [
            f'/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/240GeV_-211/{p}/sndLHC.PG_-211-TGeant4_MCEB_dig.root'
            for p in range(5)
        ],
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100637/sndsw_raw-0000.root'],
        'run_num': 100637
    },
    300: {
        'name': '300 GeV #pi^{-}',
        'particle': 'pi-',
        'mc_files': [
            f'/eos/experiment/sndlhc/MonteCarlo/testbeam2023/2mmRangeCut/300GeV_-211/{p}/sndLHC.PG_-211-TGeant4_MCEB_dig.root'
            for p in range(5)
        ],
        'data_files': ['/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100639/sndsw_raw-0000.root'],
        'run_num': 100639
    }
}

THRESHOLDS = np.arange(0.0, 20.25, 0.25) # 81 threshold steps (MeV)

def extract_data_showers(file_list, max_events=6000):
    """Extract shower hit counts per event from rawConv testbeam data."""
    chain = ROOT.TChain("rawConv")
    for f in file_list:
        chain.Add(f)
    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("Digi_MuFilterHits*", 1)
    
    n_entries = min(max_events, chain.GetEntries())
    events = []
    
    for iev in range(n_entries):
        chain.GetEntry(iev)
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
            # Strictly in-time: must have at least one valid TDC timestamp within |t| <= 15 ns
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
        
        # Clean shower selection: nu >= 6, >= 3 active planes, and max hits in a plane >= 2
        # (rejects single through-going MIP tracks and halo particles)
        if nu >= 6 and nu_pl >= 3 and max_pl >= 2:
            events.append({
                'us_total': nu,
                'us_planes': us_counts,
                'ds_total': sum(ds_counts.values()),
                'ds_planes': ds_counts
            })
            
    return events

def extract_mc_showers(file_list, max_events=2500):
    """Extract MuFilterPoint cumulative bar energy loss per shower event in MC."""
    chain = ROOT.TChain("cbmsim")
    for f in file_list:
        chain.Add(f)
    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("MuFilterPoint*", 1)
    chain.SetBranchStatus("Digi_MuFilterHits*", 1)
    
    n_entries = min(max_events, chain.GetEntries())
    events = []
    
    for iev in range(n_entries):
        chain.GetEntry(iev)
        
        # Check raw Digi multiplicity first to match shower selection
        us_digi_pl = {p: 0 for p in range(5)}
        for h in chain.Digi_MuFilterHits:
            if not h.isValid(): continue
            did = h.GetDetectorID()
            if did // 10000 == 2:
                pl = (did // 1000) % 10
                if 0 <= pl < 5:
                    us_digi_pl[pl] += 1
                
        nu_digi = sum(us_digi_pl.values())
        nu_digi_pl = sum(1 for p in range(5) if us_digi_pl[p] > 0)
        max_digi_pl = max(us_digi_pl.values()) if nu_digi > 0 else 0
        
        if not (nu_digi >= 6 and nu_digi_pl >= 3 and max_digi_pl >= 2):
            continue
            
        # Accumulate cumulative energy per bar
        bar_edep = {}
        for pt in chain.MuFilterPoint:
            did = pt.GetDetectorID()
            bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss() * 1000.0 # MeV
            
        us_bars = {p: [] for p in range(5)}
        ds_bars = {
            'DS1_H': [], 'DS1_V': [],
            'DS2_H': [], 'DS2_V': [],
            'DS3_H': [], 'DS3_V': [],
            'DS4_V': []
        }
        
        for did, edep in bar_edep.items():
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000
            if sub == 2:
                if 0 <= pl < 5:
                    us_bars[pl].append(edep)
            elif sub == 3:
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
                ds_bars[key].append(edep)
                
        events.append({
            'us_bars': us_bars,
            'ds_bars': ds_bars
        })
        
    return events

def run_threshold_scan(mc_events, data_events, thresholds=THRESHOLDS):
    """Scan threshold array and compute mean hits and shape similarity metrics."""
    # Data reference distributions
    d_us_tot = np.array([e['us_total'] for e in data_events])
    d_ds_tot = np.array([e['ds_total'] for e in data_events])
    d_us_pl = {p: np.array([e['us_planes'][p] for e in data_events]) for p in range(5)}
    d_ds_pl = {k: np.array([e['ds_planes'][k] for e in data_events]) for k in [
        'DS1_H', 'DS1_V', 'DS2_H', 'DS2_V', 'DS3_H', 'DS3_V', 'DS4_V'
    ]}
    
    data_means = {
        'us_total': np.mean(d_us_tot),
        'ds_total': np.mean(d_ds_tot),
        **{f'us_pl_{p}': np.mean(d_us_pl[p]) for p in range(5)},
        **{f'ds_pl_{k}': np.mean(d_ds_pl[k]) for k in d_ds_pl}
    }
    
    scan_results = {
        'thresholds': thresholds,
        'data_means': data_means,
        'mc_means': {k: [] for k in data_means},
        'wasserstein': {k: [] for k in data_means},
        'ks_stat': {k: [] for k in data_means},
        'hit_distributions': {} # store distributions at Eth=0 and optimal Eth
    }
    
    n_mc = len(mc_events)
    for eth in thresholds:
        mc_us_tot = np.zeros(n_mc)
        mc_ds_tot = np.zeros(n_mc)
        mc_us_pl = {p: np.zeros(n_mc) for p in range(5)}
        mc_ds_pl = {k: np.zeros(n_mc) for k in d_ds_pl}
        
        for i, ev in enumerate(mc_events):
            # US
            tot_u = 0
            for p in range(5):
                cnt = sum(1 for e in ev['us_bars'][p] if e >= eth)
                mc_us_pl[p][i] = cnt
                tot_u += cnt
            mc_us_tot[i] = tot_u
            
            # DS
            tot_d = 0
            for k in d_ds_pl:
                cnt = sum(1 for e in ev['ds_bars'][k] if e >= eth)
                mc_ds_pl[k][i] = cnt
                tot_d += cnt
            mc_ds_tot[i] = tot_d
            
        # Store means
        scan_results['mc_means']['us_total'].append(np.mean(mc_us_tot))
        scan_results['mc_means']['ds_total'].append(np.mean(mc_ds_tot))
        for p in range(5):
            scan_results['mc_means'][f'us_pl_{p}'].append(np.mean(mc_us_pl[p]))
        for k in d_ds_pl:
            scan_results['mc_means'][f'ds_pl_{k}'].append(np.mean(mc_ds_pl[k]))
            
        # Shape similarity metrics
        # 1. Total US
        scan_results['wasserstein']['us_total'].append(wasserstein_distance(mc_us_tot, d_us_tot))
        scan_results['ks_stat']['us_total'].append(ks_2samp(mc_us_tot, d_us_tot).statistic)
        # 2. Total DS
        scan_results['wasserstein']['ds_total'].append(wasserstein_distance(mc_ds_tot, d_ds_tot))
        scan_results['ks_stat']['ds_total'].append(ks_2samp(mc_ds_tot, d_ds_tot).statistic)
        # 3. US planes
        for p in range(5):
            scan_results['wasserstein'][f'us_pl_{p}'].append(wasserstein_distance(mc_us_pl[p], d_us_pl[p]))
            scan_results['ks_stat'][f'us_pl_{p}'].append(ks_2samp(mc_us_pl[p], d_us_pl[p]).statistic)
        # 4. DS planes
        for k in d_ds_pl:
            scan_results['wasserstein'][f'ds_pl_{k}'].append(wasserstein_distance(mc_ds_pl[k], d_ds_pl[k]))
            scan_results['ks_stat'][f'ds_pl_{k}'].append(ks_2samp(mc_ds_pl[k], d_ds_pl[k]).statistic)
            
    # Find crossing threshold where MC mean crosses Data mean
    crossing_thresholds = {}
    for k in data_means:
        mc_arr = np.array(scan_results['mc_means'][k])
        d_val = data_means[k]
        # find where mc_arr crosses d_val (from above to below)
        cross_th = None
        for j in range(len(thresholds) - 1):
            if (mc_arr[j] >= d_val and mc_arr[j+1] <= d_val) or (mc_arr[j] <= d_val and mc_arr[j+1] >= d_val):
                # linear interpolation
                slope = (mc_arr[j+1] - mc_arr[j]) / (thresholds[j+1] - thresholds[j])
                if abs(slope) > 1e-6:
                    cross_th = thresholds[j] + (d_val - mc_arr[j]) / slope
                else:
                    cross_th = thresholds[j]
                break
        crossing_thresholds[k] = cross_th
    scan_results['crossing_thresholds'] = crossing_thresholds
    
    # Also find threshold that minimizes Wasserstein distance (optimal shape matching)
    optimal_shape_thresholds = {}
    for k in data_means:
        w_arr = np.array(scan_results['wasserstein'][k])
        opt_idx = np.argmin(w_arr)
        optimal_shape_thresholds[k] = thresholds[opt_idx]
    scan_results['optimal_shape_thresholds'] = optimal_shape_thresholds
    
    # Store distributions at Eth=0 (uncorrected) and optimal Eth (corrected)
    opt_us_th = optimal_shape_thresholds['us_total']
    opt_ds_th = optimal_shape_thresholds['ds_total']
    
    uncorr_dist = {'us_total': [], 'ds_total': [], **{f'us_pl_{p}': [] for p in range(5)}, **{f'ds_pl_{k}': [] for k in d_ds_pl}}
    corr_dist = {'us_total': [], 'ds_total': [], **{f'us_pl_{p}': [] for p in range(5)}, **{f'ds_pl_{k}': [] for k in d_ds_pl}}
    
    for ev in mc_events:
        # Uncorrected (Eth = 0)
        tot_u0 = sum(len(ev['us_bars'][p]) for p in range(5))
        uncorr_dist['us_total'].append(tot_u0)
        for p in range(5): uncorr_dist[f'us_pl_{p}'].append(len(ev['us_bars'][p]))
        
        tot_d0 = sum(len(ev['ds_bars'][k]) for k in d_ds_pl)
        uncorr_dist['ds_total'].append(tot_d0)
        for k in d_ds_pl: uncorr_dist[f'ds_pl_{k}'].append(len(ev['ds_bars'][k]))
        
        # Corrected (optimal Eth)
        tot_uc = 0
        for p in range(5):
            c = sum(1 for e in ev['us_bars'][p] if e >= opt_us_th)
            corr_dist[f'us_pl_{p}'].append(c)
            tot_uc += c
        corr_dist['us_total'].append(tot_uc)
        
        tot_dc = 0
        for k in d_ds_pl:
            c = sum(1 for e in ev['ds_bars'][k] if e >= opt_ds_th)
            corr_dist[f'ds_pl_{k}'].append(c)
            tot_dc += c
        corr_dist['ds_total'].append(tot_dc)
        
    scan_results['data_dist'] = {
        'us_total': d_us_tot, 'ds_total': d_ds_tot,
        **{f'us_pl_{p}': d_us_pl[p] for p in range(5)},
        **{f'ds_pl_{k}': d_ds_pl[k] for k in d_ds_pl}
    }
    scan_results['uncorr_dist'] = {k: np.array(v) for k, v in uncorr_dist.items()}
    scan_results['corr_dist'] = {k: np.array(v) for k, v in corr_dist.items()}
    
    return scan_results

def plot_energy_results(energy, res, out_dir="plots"):
    """Create monotonic curves, shape similarity curves, and comparison canvases."""
    cfg = CONFIGS[energy]
    os.makedirs(out_dir, exist_ok=True)
    f_root = ROOT.TFile.Open(f"{out_dir}/tb_{energy}gev_threshold_scan.root", "RECREATE")
    
    # -------------------------------------------------------------
    # 1. Canvas: Monotonically Decreasing Curves & Data Crossing
    # -------------------------------------------------------------
    c_mono = ROOT.TCanvas(f"c_mono_{energy}gev", f"Monotonic Hit Curves vs E_th ({cfg['name']})", 1600, 1000)
    c_mono.Divide(2, 2)
    
    plots_info = [
        ("Total US", 'us_total', 1, 0, 50),
        ("US Planes (Longitudinal Profile)", [f'us_pl_{p}' for p in range(5)], 2, 0, 15),
        ("Total DS", 'ds_total', 3, 0, 30),
        ("DS Planes", [f'ds_pl_{k}' for k in ['DS1_H', 'DS1_V', 'DS2_H', 'DS2_V', 'DS3_H', 'DS3_V', 'DS4_V']], 4, 0, 10)
    ]
    
    th_vals = res['thresholds']
    n_pts = len(th_vals)
    store = []
    
    for title, key_spec, pad_idx, ymin, ymax in plots_info:
        c_mono.cd(pad_idx)
        ROOT.gPad.SetGrid(1, 1)
        
        if isinstance(key_spec, str):
            # Single curve
            k = key_spec
            mc_vals = np.array(res['mc_means'][k])
            d_val = res['data_means'][k]
            cross_th = res['crossing_thresholds'][k]
            
            gr_mc = ROOT.TGraph(n_pts, th_vals.astype(np.float64), mc_vals.astype(np.float64))
            gr_mc.SetTitle(f"{title} ({cfg['name']});Energy Threshold E_{{th}} (MeV);Mean Hit Multiplicity #LTN_{{hits}}#GT")
            gr_mc.SetLineColor(ROOT.kRed + 1)
            gr_mc.SetLineWidth(3)
            gr_mc.SetMinimum(ymin)
            gr_mc.SetMaximum(max(ymax, np.max(mc_vals) * 1.15))
            gr_mc.Draw("AL")
            
            # Data horizontal line
            l_data = ROOT.TLine(0.0, d_val, 20.0, d_val)
            l_data.SetLineColor(ROOT.kBlack)
            l_data.SetLineWidth(2)
            l_data.SetLineStyle(2)
            l_data.Draw("SAME")
            
            # Crossing vertical line
            leg = ROOT.TLegend(0.42, 0.68, 0.88, 0.88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.AddEntry(gr_mc, f"Simulation #LTN#GT (E_{{th}}=0: {mc_vals[0]:.1f})", "l")
            leg.AddEntry(l_data, f"Real Data: {d_val:.2f}", "l")
            if cross_th is not None:
                l_cross = ROOT.TLine(cross_th, ymin, cross_th, d_val)
                l_cross.SetLineColor(ROOT.kBlue + 1)
                l_cross.SetLineStyle(3)
                l_cross.SetLineWidth(2)
                l_cross.Draw("SAME")
                leg.AddEntry(l_cross, f"Crossing: E_{{th}} = {cross_th:.2f} MeV", "l")
            leg.Draw()
            store.extend([gr_mc, l_data, leg])
        else:
            # Multi-plane curves
            colors = [ROOT.kRed+1, ROOT.kBlue+1, ROOT.kGreen+2, ROOT.kOrange+7, ROOT.kMagenta+1, ROOT.kCyan+2, ROOT.kGray+2]
            mg = ROOT.TMultiGraph()
            mg.SetTitle(f"{title} ({cfg['name']});Energy Threshold E_{{th}} (MeV);Mean Hit Multiplicity #LTN_{{hits}}#GT")
            leg = ROOT.TLegend(0.55, 0.50, 0.88, 0.88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            
            for idx, k in enumerate(key_spec):
                col = colors[idx % len(colors)]
                mc_vals = np.array(res['mc_means'][k])
                d_val = res['data_means'][k]
                gr = ROOT.TGraph(n_pts, th_vals.astype(np.float64), mc_vals.astype(np.float64))
                gr.SetLineColor(col)
                gr.SetLineWidth(2)
                mg.Add(gr)
                p_label = k.replace('us_pl_', 'US Pl ').replace('ds_pl_', '')
                cr = res['crossing_thresholds'][k]
                cr_str = f"{cr:.1f} MeV" if cr is not None else ">20"
                leg.AddEntry(gr, f"{p_label} (Data:{d_val:.1f}, E_{{x}}:{cr_str})", "l")
                store.append(gr)
                
            mg.SetMinimum(ymin)
            mg.SetMaximum(ymax)
            mg.Draw("AL")
            leg.Draw()
            store.extend([mg, leg])
            
    c_mono.Write()
    c_mono.SaveAs(f"{out_dir}/tb_{energy}gev_monotonic_threshold_curves.png")
    
    # -------------------------------------------------------------
    # 2. Canvas: Shape Similarity (Wasserstein-1 Distance vs E_th)
    # -------------------------------------------------------------
    c_shape = ROOT.TCanvas(f"c_shape_{energy}gev", f"Distribution Shape Distance W1 vs E_th ({cfg['name']})", 1400, 600)
    c_shape.Divide(2, 1)
    
    for idx, (lbl, key) in enumerate([("Total US Hits", 'us_total'), ("Total DS Hits", 'ds_total')]):
        c_shape.cd(idx + 1)
        ROOT.gPad.SetGrid(1, 1)
        w_vals = np.array(res['wasserstein'][key])
        gr_w = ROOT.TGraph(n_pts, th_vals.astype(np.float64), w_vals.astype(np.float64))
        gr_w.SetTitle(f"Shape Distance W_{{1}}: {lbl} ({cfg['name']});Energy Threshold E_{{th}} (MeV);Wasserstein-1 Distance (Hits)")
        gr_w.SetLineColor(ROOT.kAzure + 2)
        gr_w.SetLineWidth(3)
        gr_w.Draw("AL")
        
        opt_th = res['optimal_shape_thresholds'][key]
        min_w = np.min(w_vals)
        l_opt = ROOT.TLine(opt_th, 0, opt_th, min_w * 1.5)
        l_opt.SetLineColor(ROOT.kRed + 1)
        l_opt.SetLineStyle(2)
        l_opt.SetLineWidth(2)
        l_opt.Draw("SAME")
        
        leg = ROOT.TLegend(0.40, 0.70, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.AddEntry(gr_w, f"W_{{1}} Shape Distance", "l")
        leg.AddEntry(l_opt, f"Optimal E_{{th}} = {opt_th:.2f} MeV (W_{{1}} = {min_w:.2f})", "l")
        leg.Draw()
        store.extend([gr_w, l_opt, leg])
        
    c_shape.Write()
    c_shape.SaveAs(f"{out_dir}/tb_{energy}gev_shape_similarity.png")
    
    # -------------------------------------------------------------
    # 3. Canvas: Multi-Panel Hit Distribution Comparison
    #    (Data vs Uncorrected MC vs Corrected MC)
    # -------------------------------------------------------------
    c_dist = ROOT.TCanvas(f"c_dist_{energy}gev", f"Hit Distributions: Data vs Sim ({cfg['name']})", 1600, 1200)
    c_dist.Divide(3, 3) # 9 panels: US Tot, US 0..4, DS Tot, DS1_H, DS1_V
    
    dist_panels = [
        ("Total US Hits", 'us_total', 1, 50, 0, 50),
        ("US Plane 0", 'us_pl_0', 2, 12, 0, 12),
        ("US Plane 1", 'us_pl_1', 3, 12, 0, 12),
        ("US Plane 2", 'us_pl_2', 4, 12, 0, 12),
        ("US Plane 3", 'us_pl_3', 5, 12, 0, 12),
        ("US Plane 4", 'us_pl_4', 6, 12, 0, 12),
        ("Total DS Hits", 'ds_total', 7, 40, 0, 40),
        ("DS1 H Hits", 'ds_pl_DS1_H', 8, 10, 0, 10),
        ("DS1 V Hits", 'ds_pl_DS1_V', 9, 10, 0, 10),
    ]
    
    for title, k, pad_idx, nbins, xmin, xmax in dist_panels:
        c_dist.cd(pad_idx)
        ROOT.gPad.SetGrid(1, 1)
        
        d_arr = res['data_dist'][k]
        u_arr = res['uncorr_dist'][k]
        c_arr = res['corr_dist'][k]
        
        h_data = ROOT.TH1D(f"h_d_{k}_{energy}", f"{title} ({cfg['name']});Number of Hits;Probability Density", nbins, xmin, xmax)
        h_uncorr = ROOT.TH1D(f"h_u_{k}_{energy}", f"Uncorr MC;Number of Hits;Probability Density", nbins, xmin, xmax)
        h_corr = ROOT.TH1D(f"h_c_{k}_{energy}", f"Corrected MC;Number of Hits;Probability Density", nbins, xmin, xmax)
        
        for v in d_arr: h_data.Fill(v)
        for v in u_arr: h_uncorr.Fill(v)
        for v in c_arr: h_corr.Fill(v)
        
        if h_data.Integral() > 0: h_data.Scale(1.0 / h_data.Integral())
        if h_uncorr.Integral() > 0: h_uncorr.Scale(1.0 / h_uncorr.Integral())
        if h_corr.Integral() > 0: h_corr.Scale(1.0 / h_corr.Integral())
        
        h_data.SetLineColor(ROOT.kBlack)
        h_data.SetLineWidth(3)
        h_uncorr.SetLineColor(ROOT.kBlue + 1)
        h_uncorr.SetLineStyle(2)
        h_uncorr.SetLineWidth(2)
        h_corr.SetLineColor(ROOT.kRed + 1)
        h_corr.SetLineWidth(3)
        
        max_y = max(h_data.GetMaximum(), h_uncorr.GetMaximum(), h_corr.GetMaximum(), 0.05) * 1.35
        h_data.SetMaximum(max_y)
        
        h_data.Draw("HIST")
        h_uncorr.Draw("HIST SAME")
        h_corr.Draw("HIST SAME")
        
        leg = ROOT.TLegend(0.42, 0.65, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.AddEntry(h_data, f"Data (#mu={np.mean(d_arr):.1f})", "l")
        leg.AddEntry(h_uncorr, f"Sim E_{{th}}=0 (#mu={np.mean(u_arr):.1f})", "l")
        leg.AddEntry(h_corr, f"Sim Corrected (#mu={np.mean(c_arr):.1f})", "l")
        leg.Draw()
        
        store.extend([h_data, h_uncorr, h_corr, leg])
        
    c_dist.Write()
    c_dist.SaveAs(f"{out_dir}/tb_{energy}gev_hit_distributions_multipanel.png")
    
    f_root.Close()
    print(f"[{energy} GeV] Saved canvases and ROOT file to {out_dir}/tb_{energy}gev_threshold_scan.root")

def main():
    print("=================================================================")
    print("Testbeam 2023 H8 MuFilter Threshold Scan & Shape Matching")
    print("=================================================================")
    
    summary_table = []
    
    for energy in [100, 140, 180, 240, 300]:
        cfg = CONFIGS[energy]
        print(f"\n>>> Processing {energy} GeV ({cfg['name']})...")
        t0 = time.time()
        
        # 1. Read Data
        data_ev = extract_data_showers(cfg['data_files'], max_events=6000)
        print(f"  Data: {len(data_ev)} clean shower events selected.")
        
        # 2. Read MC
        mc_ev = extract_mc_showers(cfg['mc_files'], max_events=2500)
        print(f"  Simulation: {len(mc_ev)} shower events extracted.")
        
        # 3. Threshold Scan
        res = run_threshold_scan(mc_ev, data_ev)
        
        # 4. Plots
        plot_energy_results(energy, res, out_dir="plots")
        
        us_cr = res['crossing_thresholds']['us_total']
        ds_cr = res['crossing_thresholds']['ds_total']
        us_opt = res['optimal_shape_thresholds']['us_total']
        ds_opt = res['optimal_shape_thresholds']['ds_total']
        
        summary_table.append({
            'energy': energy,
            'name': cfg['name'],
            'data_us': res['data_means']['us_total'],
            'mc_uncorr_us': res['mc_means']['us_total'][0],
            'us_crossing_th': us_cr,
            'us_optimal_th': us_opt,
            'data_ds': res['data_means']['ds_total'],
            'mc_uncorr_ds': res['mc_means']['ds_total'][0],
            'ds_crossing_th': ds_cr,
            'ds_optimal_th': ds_opt,
            'us_w1_uncorr': res['wasserstein']['us_total'][0],
            'us_w1_corr': np.min(res['wasserstein']['us_total']),
            'ds_w1_uncorr': res['wasserstein']['ds_total'][0],
            'ds_w1_corr': np.min(res['wasserstein']['ds_total']),
        })
        us_cr_str = f"{us_cr:.2f} MeV" if us_cr is not None else ">20.00 MeV"
        ds_cr_str = f"{ds_cr:.2f} MeV" if ds_cr is not None else ">20.00 MeV"
        print(f"  Done {energy} GeV in {time.time()-t0:.1f} s. US Cross: {us_cr_str}, DS Cross: {ds_cr_str}.")
        
    print("\n" + "="*80)
    print("FINAL SUMMARY: THRESHOLD SCANS ACROSS ALL TESTBEAM ENERGIES")
    print("="*80)
    print(f"{'Energy':<10} | {'Data US':<8} | {'Sim uncorr':<10} | {'US Cross':<10} | {'US Opt W1':<9} | {'Data DS':<8} | {'Sim uncorr':<10} | {'DS Cross':<10} | {'DS Opt W1':<9}")
    print("-" * 92)
    for row in summary_table:
        u_c = f"{row['us_crossing_th']:.2f}" if row['us_crossing_th'] is not None else ">20.00"
        d_c = f"{row['ds_crossing_th']:.2f}" if row['ds_crossing_th'] is not None else ">20.00"
        print(f"{row['energy']} GeV    | {row['data_us']:<8.2f} | {row['mc_uncorr_us']:<10.2f} | {u_c:<10} | {row['us_optimal_th']:<9.2f} | {row['data_ds']:<8.2f} | {row['mc_uncorr_ds']:<10.2f} | {d_c:<10} | {row['ds_optimal_th']:<9.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
