#!/usr/bin/env python3
"""
Relative Hit & Activated Veto Event Reduction Analysis for NuMu Charm Simulation (newP8)
Calculates and plots:
1. Relative hit reduction curves for Veto, US, DS (subsystems and planes)
2. Relative reduction of events with an activated Veto:
   - Any Veto hit (N_veto >= 1)
   - Veto coincidence (N_veto_pl0 >= 1 AND N_veto_pl1 >= 1)
   - Plane 0 activated (N_veto_pl0 >= 1)
   - Plane 1 activated (N_veto_pl1 >= 1)
3. Evaluates exact hit and event reduction at:
   - 119.4 keV for Horizontal Veto
   - 69.9 keV for Vertical Veto
   - 129.7 keV for US Gel
   - 227.7 keV for US Air
   - 38.4 keV for Horizontal DS
   - 22.1 keV for Vertical DS
"""

import os
import sys
import json
import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Target thresholds (in keV)
TH_DS_HORIZ   = 38.4
TH_DS_VERT    = 22.1
TH_US_GEL     = 129.7
TH_US_AIR     = 227.7
TH_VETO_HORIZ = 119.4
TH_VETO_VERT  = 69.9

def build_p8_chain(base_dir="/eos/user/i/idioniso/snd-numu-charm/data/newP8", max_partitions=40):
    chain = ROOT.TChain("cbmsim")
    n_added = 0
    for p in range(1, max_partitions + 1):
        fpath = os.path.join(base_dir, str(p), "sndLHC.Genie-TGeant4_MCEB_dig.root")
        if os.path.exists(fpath):
            chain.Add(fpath)
            n_added += 1
    print(f"Chained {n_added} partitions, total entries: {chain.GetEntries()}")
    return chain

def extract_p8_events(chain, max_events=4000):
    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("MuFilterPoint*", 1)
    
    n_entries = min(max_events, chain.GetEntries())
    events = []
    
    for iev in range(n_entries):
        chain.GetEntry(iev)
        
        bar_edep = {}
        for pt in chain.MuFilterPoint:
            did = pt.GetDetectorID()
            bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss() * 1.0e6 # keV
            
        ev = {
            # Veto bar edeps
            'veto_all': [],
            'veto_pl_0': [],
            'veto_pl_1': [],
            'veto_pl_2': [],
            'veto_horiz': [],
            'veto_vert': [],
            # US bar edeps
            'us_all': [],
            'us_pl_0': [],
            'us_pl_1': [],
            'us_pl_2': [],
            'us_pl_3': [],
            'us_pl_4': [],
            # DS bar edeps
            'ds_all': [],
            'ds_horiz': [],
            'ds_vert': [],
            'ds_pl_DS1_H': [],
            'ds_pl_DS1_V': [],
            'ds_pl_DS2_H': [],
            'ds_pl_DS2_V': [],
            'ds_pl_DS3_H': [],
            'ds_pl_DS3_V': [],
            'ds_pl_DS4_V': []
        }
        
        for did, edep in bar_edep.items():
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000
            
            if sub == 1: # Veto
                ev['veto_all'].append(edep)
                if pl == 0:
                    ev['veto_pl_0'].append(edep)
                    ev['veto_horiz'].append(edep)
                elif pl == 1:
                    ev['veto_pl_1'].append(edep)
                    ev['veto_horiz'].append(edep)
                elif pl == 2:
                    ev['veto_pl_2'].append(edep)
                    ev['veto_vert'].append(edep)
            elif sub == 2: # US
                if 0 <= pl < 5:
                    ev['us_all'].append(edep)
                    ev[f'us_pl_{pl}'].append(edep)
            elif sub == 3: # DS
                ev['ds_all'].append(edep)
                if pl == 0:
                    if bar < 60:
                        ev['ds_horiz'].append(edep)
                        ev['ds_pl_DS1_H'].append(edep)
                    else:
                        ev['ds_vert'].append(edep)
                        ev['ds_pl_DS1_V'].append(edep)
                elif pl == 1:
                    if bar < 60:
                        ev['ds_horiz'].append(edep)
                        ev['ds_pl_DS2_H'].append(edep)
                    else:
                        ev['ds_vert'].append(edep)
                        ev['ds_pl_DS2_V'].append(edep)
                elif pl == 2:
                    if bar < 60:
                        ev['ds_horiz'].append(edep)
                        ev['ds_pl_DS3_H'].append(edep)
                    else:
                        ev['ds_vert'].append(edep)
                        ev['ds_pl_DS3_V'].append(edep)
                elif pl == 3:
                    ev['ds_vert'].append(edep)
                    ev['ds_pl_DS4_V'].append(edep)
                    
        events.append(ev)
    return events

def analyze_reduction(events, thresholds):
    n_ev = len(events)
    th_arr = np.array(thresholds, dtype=np.float64)
    
    # Baseline hits and activated events
    base_hits = {cat: sum(len(e[cat]) for e in events) for cat in events[0].keys()}
    
    base_ev_veto_any = sum(1 for e in events if len(e['veto_all']) >= 1)
    base_ev_veto_coinc = sum(1 for e in events if len(e['veto_pl_0']) >= 1 and len(e['veto_pl_1']) >= 1)
    base_ev_veto_pl0 = sum(1 for e in events if len(e['veto_pl_0']) >= 1)
    base_ev_veto_pl1 = sum(1 for e in events if len(e['veto_pl_1']) >= 1)
    
    # Reduction curves
    hit_curves = {cat: {'surviving_frac': [], 'pct_removed': []} for cat in events[0].keys()}
    
    ev_curves = {
        'veto_any': {'surviving_frac': [], 'pct_removed': []},
        'veto_coinc': {'surviving_frac': [], 'pct_removed': []},
        'veto_pl_0': {'surviving_frac': [], 'pct_removed': []},
        'veto_pl_1': {'surviving_frac': [], 'pct_removed': []}
    }
    
    for eth in thresholds:
        # Hit curves
        for cat in hit_curves:
            surv = sum(sum(1 for edep in e[cat] if edep > eth) for e in events)
            base = base_hits[cat]
            frac = surv / base if base > 0 else 0.0
            hit_curves[cat]['surviving_frac'].append(frac)
            hit_curves[cat]['pct_removed'].append((1.0 - frac) * 100.0)
            
        # Veto activated event curves
        # 1. Any veto bar > eth
        n_any = sum(1 for e in events if any(edep > eth for edep in e['veto_all']))
        f_any = n_any / base_ev_veto_any if base_ev_veto_any > 0 else 0.0
        ev_curves['veto_any']['surviving_frac'].append(f_any)
        ev_curves['veto_any']['pct_removed'].append((1.0 - f_any) * 100.0)
        
        # 2. Coincidence: at least 1 bar in pl0 > eth AND at least 1 bar in pl1 > eth
        n_coinc = sum(1 for e in events if any(edep > eth for edep in e['veto_pl_0']) and any(edep > eth for edep in e['veto_pl_1']))
        f_coinc = n_coinc / base_ev_veto_coinc if base_ev_veto_coinc > 0 else 0.0
        ev_curves['veto_coinc']['surviving_frac'].append(f_coinc)
        ev_curves['veto_coinc']['pct_removed'].append((1.0 - f_coinc) * 100.0)
        
        # 3. Plane 0
        n_p0 = sum(1 for e in events if any(edep > eth for edep in e['veto_pl_0']))
        f_p0 = n_p0 / base_ev_veto_pl0 if base_ev_veto_pl0 > 0 else 0.0
        ev_curves['veto_pl_0']['surviving_frac'].append(f_p0)
        ev_curves['veto_pl_0']['pct_removed'].append((1.0 - f_p0) * 100.0)
        
        # 4. Plane 1
        n_p1 = sum(1 for e in events if any(edep > eth for edep in e['veto_pl_1']))
        f_p1 = n_p1 / base_ev_veto_pl1 if base_ev_veto_pl1 > 0 else 0.0
        ev_curves['veto_pl_1']['surviving_frac'].append(f_p1)
        ev_curves['veto_pl_1']['pct_removed'].append((1.0 - f_p1) * 100.0)
        
    # Evaluate exact numbers at 119.4 keV
    eth = TH_VETO_HORIZ
    surv_hits_veto = sum(sum(1 for edep in e['veto_all'] if edep > eth) for e in events)
    surv_ev_any = sum(1 for e in events if any(edep > eth for edep in e['veto_all']))
    surv_ev_coinc = sum(1 for e in events if any(edep > eth for edep in e['veto_pl_0']) and any(edep > eth for edep in e['veto_pl_1']))
    surv_ev_pl0 = sum(1 for e in events if any(edep > eth for edep in e['veto_pl_0']))
    surv_ev_pl1 = sum(1 for e in events if any(edep > eth for edep in e['veto_pl_1']))
    
    exact_veto = {
        'threshold_kev': eth,
        'hits': {
            'baseline': base_hits['veto_all'],
            'surviving': surv_hits_veto,
            'pct_removed': (1.0 - surv_hits_veto / base_hits['veto_all']) * 100.0,
            'pct_retained': (surv_hits_veto / base_hits['veto_all']) * 100.0
        },
        'events_any': {
            'baseline': base_ev_veto_any,
            'surviving': surv_ev_any,
            'pct_removed': (1.0 - surv_ev_any / base_ev_veto_any) * 100.0,
            'pct_retained': (surv_ev_any / base_ev_veto_any) * 100.0
        },
        'events_coincidence': {
            'baseline': base_ev_veto_coinc,
            'surviving': surv_ev_coinc,
            'pct_removed': (1.0 - surv_ev_coinc / base_ev_veto_coinc) * 100.0,
            'pct_retained': (surv_ev_coinc / base_ev_veto_coinc) * 100.0
        },
        'events_plane0': {
            'baseline': base_ev_veto_pl0,
            'surviving': surv_ev_pl0,
            'pct_removed': (1.0 - surv_ev_pl0 / base_ev_veto_pl0) * 100.0,
            'pct_retained': (surv_ev_pl0 / base_ev_veto_pl0) * 100.0
        },
        'events_plane1': {
            'baseline': base_ev_veto_pl1,
            'surviving': surv_ev_pl1,
            'pct_removed': (1.0 - surv_ev_pl1 / base_ev_veto_pl1) * 100.0,
            'pct_retained': (surv_ev_pl1 / base_ev_veto_pl1) * 100.0
        }
    }
    
    return hit_curves, ev_curves, exact_veto, base_hits

def plot_veto_combined_canvas(hit_curves, ev_curves, exact_veto, thresholds, out_dir="plots"):
    th_arr = np.array(thresholds, dtype=np.float64)
    n_pts = len(thresholds)
    
    c = ROOT.TCanvas("c_p8_reduction_veto_events_and_hits", "Veto Hit & Activated Event Reduction (newP8)", 1200, 800)
    c.SetGrid(1, 1)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.06)
    c.SetBottomMargin(0.12)
    
    mg = ROOT.TMultiGraph()
    mg.SetTitle("Veto Relative Reduction vs Threshold (newP8 Simulation);Energy Threshold E_{th} [keV];Fraction Retained N(E_{th}) / N_{0}")
    
    # 1. Total Veto Hits
    gr_hits = ROOT.TGraph(n_pts, th_arr, np.array(hit_curves['veto_all']['surviving_frac'], dtype=np.float64))
    gr_hits.SetLineColor(ROOT.kBlack)
    gr_hits.SetLineWidth(3)
    mg.Add(gr_hits)
    
    # 2. Events with ANY Veto hit
    gr_ev_any = ROOT.TGraph(n_pts, th_arr, np.array(ev_curves['veto_any']['surviving_frac'], dtype=np.float64))
    gr_ev_any.SetLineColor(ROOT.kRed + 1)
    gr_ev_any.SetLineWidth(3)
    mg.Add(gr_ev_any)
    
    # 3. Events with V1 & V2 Coincidence
    gr_ev_coinc = ROOT.TGraph(n_pts, th_arr, np.array(ev_curves['veto_coinc']['surviving_frac'], dtype=np.float64))
    gr_ev_coinc.SetLineColor(ROOT.kAzure + 2)
    gr_ev_coinc.SetLineWidth(3)
    mg.Add(gr_ev_coinc)
    
    # 4. Events with Veto Plane 0
    gr_ev_p0 = ROOT.TGraph(n_pts, th_arr, np.array(ev_curves['veto_pl_0']['surviving_frac'], dtype=np.float64))
    gr_ev_p0.SetLineColor(ROOT.kGreen + 2)
    gr_ev_p0.SetLineWidth(2)
    gr_ev_p0.SetLineStyle(2)
    mg.Add(gr_ev_p0)
    
    # 5. Events with Veto Plane 1
    gr_ev_p1 = ROOT.TGraph(n_pts, th_arr, np.array(ev_curves['veto_pl_1']['surviving_frac'], dtype=np.float64))
    gr_ev_p1.SetLineColor(ROOT.kMagenta + 2)
    gr_ev_p1.SetLineWidth(2)
    gr_ev_p1.SetLineStyle(2)
    mg.Add(gr_ev_p1)
    
    mg.Draw("AL")
    mg.SetMinimum(0.0)
    mg.SetMaximum(1.05)
    
    # Vertical line at 119.4 keV
    eth = TH_VETO_HORIZ
    line_v = ROOT.TLine(eth, 0.0, eth, 1.0)
    line_v.SetLineColor(ROOT.kGray + 2)
    line_v.SetLineStyle(2)
    line_v.SetLineWidth(3)
    line_v.Draw("SAME")
    
    # Legend
    leg = ROOT.TLegend(0.48, 0.62, 0.92, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.032)
    leg.AddEntry(gr_hits, f"Veto Total Hits (rem: {exact_veto['hits']['pct_removed']:.1f}%)", "l")
    leg.AddEntry(gr_ev_any, f"Events w/ #geq1 Veto Hit (rem: {exact_veto['events_any']['pct_removed']:.1f}%)", "l")
    leg.AddEntry(gr_ev_coinc, f"Events w/ V1#capV2 Coincidence (rem: {exact_veto['events_coincidence']['pct_removed']:.1f}%)", "l")
    leg.AddEntry(gr_ev_p0, f"Events w/ Veto Plane 0 (rem: {exact_veto['events_plane0']['pct_removed']:.1f}%)", "l")
    leg.AddEntry(gr_ev_p1, f"Events w/ Veto Plane 1 (rem: {exact_veto['events_plane1']['pct_removed']:.1f}%)", "l")
    leg.Draw()
    
    lat = ROOT.TLatex()
    lat.SetNDC()
    lat.SetTextSize(0.033)
    lat.SetTextColor(ROOT.kRed + 2)
    lat.DrawLatex(0.18, 0.40, f"Threshold E_{{th}} = {eth} keV:")
    lat.SetTextColor(ROOT.kBlack)
    lat.DrawLatex(0.18, 0.35, f"#bullet Hits Removed: #bf{{{exact_veto['hits']['pct_removed']:.1f}%}}")
    lat.SetTextColor(ROOT.kRed + 1)
    lat.DrawLatex(0.18, 0.30, f"#bullet Events w/ #geq1 Hit Removed: #bf{{{exact_veto['events_any']['pct_removed']:.1f}%}}")
    lat.SetTextColor(ROOT.kAzure + 2)
    lat.DrawLatex(0.18, 0.25, f"#bullet Events w/ Coincidence Removed: #bf{{{exact_veto['events_coincidence']['pct_removed']:.1f}%}}")
    
    c.SaveAs(f"{out_dir}/p8_reduction_veto_events_and_hits.png")
    return c

def main():
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    chain = build_p8_chain("/eos/user/i/idioniso/snd-numu-charm/data/newP8", max_partitions=40)
    events = extract_p8_events(chain, max_events=4000)
    print(f"Extracted {len(events)} events.")
    
    thresholds = list(np.linspace(0.0, 500.0, 251))
    hit_curves, ev_curves, exact_veto, base_hits = analyze_reduction(events, thresholds)
    
    c_veto_ev = plot_veto_combined_canvas(hit_curves, ev_curves, exact_veto, thresholds, out_dir=out_dir)
    
    # Save ROOT file
    rfile = ROOT.TFile("output/p8_veto_event_reduction.root", "RECREATE")
    c_veto_ev.Write()
    rfile.Close()
    
    # Save JSON report
    with open("output/p8_veto_event_reduction.json", "w") as jf:
        json.dump(exact_veto, jf, indent=2)
        
    # Write readable text report
    with open("output/p8_veto_event_reduction.txt", "w") as tf:
        tf.write("========================================================================================\n")
        tf.write("SND@LHC NuMu Charm Simulation (newP8): Veto Hits & Activated Events Reduction\n")
        tf.write(f"Evaluated at Horizontal Veto Threshold = {TH_VETO_HORIZ} keV\n")
        tf.write("========================================================================================\n\n")
        
        tf.write(f"Total Analyzed Events: {len(events)}\n\n")
        
        tf.write(f"{'Metric':<35} | {'Baseline (Eth=0)':<18} | {'Surviving':<12} | {'% Removed':<10} | {'% Retained':<10}\n")
        tf.write("-" * 92 + "\n")
        tf.write(f"{'Total Veto Hits':<35} | {exact_veto['hits']['baseline']:<18d} | {exact_veto['hits']['surviving']:<12d} | {exact_veto['hits']['pct_removed']:<10.2f}% | {exact_veto['hits']['pct_retained']:<10.2f}%\n")
        tf.write(f"{'Events with >= 1 Veto Hit':<35} | {exact_veto['events_any']['baseline']:<18d} | {exact_veto['events_any']['surviving']:<12d} | {exact_veto['events_any']['pct_removed']:<10.2f}% | {exact_veto['events_any']['pct_retained']:<10.2f}%\n")
        tf.write(f"{'Events with V1 & V2 Coincidence':<35} | {exact_veto['events_coincidence']['baseline']:<18d} | {exact_veto['events_coincidence']['surviving']:<12d} | {exact_veto['events_coincidence']['pct_removed']:<10.2f}% | {exact_veto['events_coincidence']['pct_retained']:<10.2f}%\n")
        tf.write(f"{'Events with Veto Plane 0':<35} | {exact_veto['events_plane0']['baseline']:<18d} | {exact_veto['events_plane0']['surviving']:<12d} | {exact_veto['events_plane0']['pct_removed']:<10.2f}% | {exact_veto['events_plane0']['pct_retained']:<10.2f}%\n")
        tf.write(f"{'Events with Veto Plane 1':<35} | {exact_veto['events_plane1']['baseline']:<18d} | {exact_veto['events_plane1']['surviving']:<12d} | {exact_veto['events_plane1']['pct_removed']:<10.2f}% | {exact_veto['events_plane1']['pct_retained']:<10.2f}%\n")
        tf.write("-" * 92 + "\n")
        
    print("\nProcessing complete! Output saved to:")
    print("  - output/p8_veto_event_reduction.root")
    print("  - output/p8_veto_event_reduction.json")
    print("  - output/p8_veto_event_reduction.txt")
    print("  - plots/p8_reduction_veto_events_and_hits.png")

if __name__ == '__main__':
    main()
