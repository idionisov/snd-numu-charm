#!/usr/bin/env python3
"""
Testbeam 2023 H8 MuFilter Analysis: Matched Target Showers (N_SciFi >= 100)
with Hit Masking (130 keV US / 22.1 keV DS).

Selection:
- N_SciFi >= 100 (applied to both data and simulation)
- In data: strictly in-time hits (|t| <= 15 ns) and unique bar counting per subsystem/plane
- Hit Masking: E_bar > 130 keV for US bars, > 22.1 keV for DS bars
- Only masked hits used for corrected distributions and analysis
- Comparison: Data (solid black), Corrected MC (solid red), Uncorrected MC (dotted blue)
- Wasserstein-1 distance computed and written to text and JSON reports.
"""

import os
import sys
import json
import numpy as np
from scipy.stats import wasserstein_distance

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Energy configuration
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

E_TH_US = 0.000130   # 130 keV in GeV
E_TH_DS = 0.0000221  # 22.1 keV in GeV

def extract_data_scifi100(file_list, max_events=30000, target_clean=1000):
    """Extract Data events passing N_SciFi >= 100 with in-time hits."""
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
        
        events.append({
            'us_total': nu,
            'us_planes': us_counts,
            'ds_total': sum(ds_counts.values()),
            'ds_planes': ds_counts
        })
        if len(events) >= target_clean:
            break
                
    return events

def extract_mc_scifi100(file_list, max_events=4000):
    """
    Extract MC events applying:
    1. Bar energy loss sum E_bar = sum(E_dep_i)
    2. Hit masking: E_bar > 130 keV for US, > 22.1 keV for DS
    3. Event selection applied: N_SciFi >= 100
    4. Records both masked (corrected) and unmasked (uncorrected) hits for passing events.
    """
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
            
        # 1. Sum energy deposit per bar
        bar_edep = {}
        for pt in chain.MuFilterPoint:
            did = pt.GetDetectorID()
            bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss()
            
        # 2. Hit masking
        masked_us_counts = {p: 0 for p in range(5)}
        unmasked_us_counts = {p: 0 for p in range(5)}
        
        masked_ds_counts = {
            'DS1_H': 0, 'DS1_V': 0,
            'DS2_H': 0, 'DS2_V': 0,
            'DS3_H': 0, 'DS3_V': 0,
            'DS4_V': 0
        }
        unmasked_ds_counts = {
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
                unmasked_us_counts[pl] += 1
                if edep > E_TH_US:
                    masked_us_counts[pl] += 1
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
                unmasked_ds_counts[key] += 1
                if edep > E_TH_DS:
                    masked_ds_counts[key] += 1
                    
        nu_m = sum(masked_us_counts.values())
        
        events.append({
            'corr': {
                'us_total': nu_m,
                'us_planes': masked_us_counts,
                'ds_total': sum(masked_ds_counts.values()),
                'ds_planes': masked_ds_counts
            },
            'uncorr': {
                'us_total': sum(unmasked_us_counts.values()),
                'us_planes': unmasked_us_counts,
                'ds_total': sum(unmasked_ds_counts.values()),
                'ds_planes': unmasked_ds_counts
            }
        })
            
    return events

def main():
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    rfile = ROOT.TFile("output/tb_scifi100_analysis.root", "RECREATE")
    all_summary = {}
    
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

    for energy in [100, 140, 180, 240, 300]:
        cfg = CONFIGS[energy]
        print(f"\n==========================================")
        print(f"Processing SciFi>=100: {cfg['name']} (Energy = {energy} GeV)...")
        print(f"==========================================")
        
        data_ev = extract_data_scifi100(cfg['data_files'], max_events=30000, target_clean=1000)
        mc_ev = extract_mc_scifi100(cfg['mc_files'], max_events=4000)
        
        print(f"  Selected clean Data target-showers: {len(data_ev)}")
        print(f"  Selected clean MC target-showers (masked): {len(mc_ev)}")
        
        d_arrs = {
            'us_total': np.array([e['us_total'] for e in data_ev]),
            'ds_total': np.array([e['ds_total'] for e in data_ev]),
            **{f'us_pl_{p}': np.array([e['us_planes'][p] for e in data_ev]) for p in range(5)},
            **{f'ds_pl_{k}': np.array([e['ds_planes'][k] for e in data_ev]) for k in ['DS1_H', 'DS1_V', 'DS2_H', 'DS2_V', 'DS3_H', 'DS3_V', 'DS4_V']}
        }
        
        c_arrs = {
            'us_total': np.array([e['corr']['us_total'] for e in mc_ev]),
            'ds_total': np.array([e['corr']['ds_total'] for e in mc_ev]),
            **{f'us_pl_{p}': np.array([e['corr']['us_planes'][p] for e in mc_ev]) for p in range(5)},
            **{f'ds_pl_{k}': np.array([e['corr']['ds_planes'][k] for e in mc_ev]) for k in ['DS1_H', 'DS1_V', 'DS2_H', 'DS2_V', 'DS3_H', 'DS3_V', 'DS4_V']}
        }
        
        u_arrs = {
            'us_total': np.array([e['uncorr']['us_total'] for e in mc_ev]),
            'ds_total': np.array([e['uncorr']['ds_total'] for e in mc_ev]),
            **{f'us_pl_{p}': np.array([e['uncorr']['us_planes'][p] for e in mc_ev]) for p in range(5)},
            **{f'ds_pl_{k}': np.array([e['uncorr']['ds_planes'][k] for e in mc_ev]) for k in ['DS1_H', 'DS1_V', 'DS2_H', 'DS2_V', 'DS3_H', 'DS3_V', 'DS4_V']}
        }
        
        w_corr = {}
        w_uncorr = {}
        means_data = {}
        means_corr = {}
        means_uncorr = {}
        
        for k, _, _, _, _ in keys_order:
            d = d_arrs[k]
            c = c_arrs[k]
            u = u_arrs[k]
            
            means_data[k] = float(np.mean(d)) if len(d) > 0 else 0.0
            means_corr[k] = float(np.mean(c)) if len(c) > 0 else 0.0
            means_uncorr[k] = float(np.mean(u)) if len(u) > 0 else 0.0
            
            w_c = float(wasserstein_distance(d, c)) if len(d) > 0 and len(c) > 0 else 0.0
            w_u = float(wasserstein_distance(d, u)) if len(d) > 0 and len(u) > 0 else 0.0
            
            w_corr[k] = w_c
            w_uncorr[k] = w_u
            
        all_summary[energy] = {
            'energy': energy,
            'name': cfg['name'],
            'n_data': len(data_ev),
            'n_mc': len(mc_ev),
            'means_data': means_data,
            'means_corr': means_corr,
            'means_uncorr': means_uncorr,
            'w1_corrected': w_corr,
            'w1_uncorrected': w_uncorr,
        }
        
        print(f"  US Total Mean: Data={means_data['us_total']:.2f}, Corrected MC={means_corr['us_total']:.2f}, Uncorrected MC={means_uncorr['us_total']:.2f}")
        print(f"  US Total Wasserstein W1: Corrected={w_corr['us_total']:.2f}, Uncorrected={w_uncorr['us_total']:.2f}")
        print(f"  DS Total Mean: Data={means_data['ds_total']:.2f}, Corrected MC={means_corr['ds_total']:.2f}, Uncorrected MC={means_uncorr['ds_total']:.2f}")
        print(f"  DS Total Wasserstein W1: Corrected={w_corr['ds_total']:.2f}, Uncorrected={w_uncorr['ds_total']:.2f}")
        
        # -------------------------------------------------------------
        # Create Multi-Panel Hit Distribution Canvas
        # -------------------------------------------------------------
        c_dist = ROOT.TCanvas(f"c_dist_scifi100_{energy}gev", f"SciFi>=100 Hit Distributions ({cfg['name']})", 1600, 1200)
        c_dist.Divide(3, 3)
        store = []
        
        for idx, (k, title, nbins, xmin, xmax) in enumerate(keys_order):
            c_dist.cd(idx + 1)
            ROOT.gPad.SetGrid(1, 1)
            ROOT.gPad.SetLeftMargin(0.12)
            ROOT.gPad.SetRightMargin(0.05)
            ROOT.gPad.SetBottomMargin(0.12)
            
            d_arr = d_arrs[k]
            u_arr = u_arrs[k]
            c_arr = c_arrs[k]
            
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
            h_uncorr.SetLineStyle(2) # Dotted line
            h_uncorr.SetLineWidth(2)
            
            h_corr.SetLineColor(ROOT.kRed + 1)
            h_corr.SetLineWidth(3)
            
            max_y = max(h_data.GetMaximum(), h_uncorr.GetMaximum(), h_corr.GetMaximum(), 0.05) * 1.35
            h_data.SetMaximum(max_y)
            h_data.SetMinimum(0)
            
            h_data.Draw("HIST")
            h_uncorr.Draw("HIST SAME")
            h_corr.Draw("HIST SAME")
            
            leg = ROOT.TLegend(0.38, 0.65, 0.93, 0.88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetTextSize(0.038)
            leg.AddEntry(h_data, f"Data (#mu={means_data[k]:.1f})", "l")
            leg.AddEntry(h_uncorr, f"Sim E_{{th}}=0 (#mu={means_uncorr[k]:.1f}, W_{{1}}={w_uncorr[k]:.2f})", "l")
            leg.AddEntry(h_corr, f"Sim Masked (#mu={means_corr[k]:.1f}, W_{{1}}={w_corr[k]:.2f})", "l")
            leg.Draw()
            
            store.extend([h_data, h_uncorr, h_corr, leg])
            
        c_dist.Write()
        c_dist.SaveAs(f"{out_dir}/tb_{energy}gev_hit_distributions_scifi100.png")
        store.append(c_dist)
        
    rfile.Close()
    
    with open("output/tb_scifi100_wasserstein.json", "w") as jf:
        json.dump(all_summary, jf, indent=2)
        
    with open("output/tb_scifi100_wasserstein.txt", "w") as tf:
        tf.write("========================================================================================\n")
        tf.write("SND@LHC Testbeam 2023 Analysis: Target Showers (N_SciFi >= 100)\n")
        tf.write("Hit Masking Thresholds: US = 130 keV, DS = 22.1 keV\n")
        tf.write("========================================================================================\n\n")
        
        tf.write(f"{'Energy':<10} | {'Subsystem':<14} | {'Data Mean':<10} | {'Uncorr Mean':<12} | {'Corr Mean':<10} | {'W1 (Uncorr)':<12} | {'W1 (Corr)':<10} | {'W1 Imprv':<8}\n")
        tf.write("-" * 92 + "\n")
        for e in [100, 140, 180, 240, 300]:
            s = all_summary[e]
            for k in ['us_total', 'ds_total']:
                lbl = 'Total US' if k == 'us_total' else 'Total DS'
                dm = s['means_data'][k]
                um = s['means_uncorr'][k]
                cm = s['means_corr'][k]
                wu = s['w1_uncorrected'][k]
                wc = s['w1_corrected'][k]
                diff = wu - wc
                tf.write(f"{e} GeV{' ':<4} | {lbl:<14} | {dm:<10.2f} | {um:<12.2f} | {cm:<10.2f} | {wu:<12.2f} | {wc:<10.2f} | {diff:<+8.2f}\n")
            tf.write("-" * 92 + "\n")
            
    print("\nProcessing complete! Output saved to:")
    print("  - output/tb_scifi100_analysis.root")
    print("  - output/tb_scifi100_wasserstein.json")
    print("  - output/tb_scifi100_wasserstein.txt")
    print("  - plots/tb_*gev_hit_distributions_scifi100.png")

if __name__ == '__main__':
    main()
