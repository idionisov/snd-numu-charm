#!/usr/bin/env python3
"""
Relative Hit Reduction Analysis for NuMu Charm Production (newP8)
Calculates and plots the monotonically decreasing surviving hit fraction
vs energy threshold for Veto, US, and DS (subsystems and individual planes).

Thresholds evaluated with vertical marker lines:
- 38.4 keV for horizontal DS hits
- 22.1 keV for vertical DS hits
- 129.7 keV for US hits with gel at readout
- 227.7 keV for US hits without gel (air) at readout
- 119.4 keV for horizontal veto hits
- 69.9 keV for vertical veto hits (plane 2, for 2024+ setups)
"""

import os
import sys
import json
import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Target thresholds (in keV)
TH_DS_HORIZ = 38.4
TH_DS_VERT  = 22.1
TH_US_GEL   = 129.7
TH_US_AIR   = 227.7
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
    print(f"Chained {n_added} partitions from {base_dir}, total entries: {chain.GetEntries()}")
    return chain

def extract_all_bar_energies(chain, max_events=4000):
    """
    Extract cumulative energy deposit (in keV) per bar for each event.
    Returns a list of event dictionaries categorized by subsystem and plane.
    """
    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("MuFilterPoint*", 1)
    
    n_entries = min(max_events, chain.GetEntries())
    events = []
    
    # US Gel mapping by plane: 0:Right, 1:Left, 2:Right, 3:Left, 4:Left
    # For a bar deposit, we record the bar edep
    for iev in range(n_entries):
        chain.GetEntry(iev)
        
        bar_edep = {}
        for pt in chain.MuFilterPoint:
            did = pt.GetDetectorID()
            # Convert GeV to keV: 1 GeV = 1e6 keV
            bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss() * 1.0e6
            
        ev_data = {
            'veto_all': [],
            'veto_pl_0': [],
            'veto_pl_1': [],
            'veto_pl_2': [],
            'veto_horiz': [],
            'veto_vert': [],
            'us_all': [],
            'us_pl_0': [],
            'us_pl_1': [],
            'us_pl_2': [],
            'us_pl_3': [],
            'us_pl_4': [],
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
        
        for did, edep_kev in bar_edep.items():
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000
            
            if sub == 1: # Veto
                ev_data['veto_all'].append(edep_kev)
                if pl == 0:
                    ev_data['veto_pl_0'].append(edep_kev)
                    ev_data['veto_horiz'].append(edep_kev)
                elif pl == 1:
                    ev_data['veto_pl_1'].append(edep_kev)
                    ev_data['veto_horiz'].append(edep_kev)
                elif pl == 2:
                    ev_data['veto_pl_2'].append(edep_kev)
                    ev_data['veto_vert'].append(edep_kev)
                    
            elif sub == 2: # Upstream
                if 0 <= pl < 5:
                    ev_data['us_all'].append(edep_kev)
                    ev_data[f'us_pl_{pl}'].append(edep_kev)
                    
            elif sub == 3: # Downstream
                ev_data['ds_all'].append(edep_kev)
                if pl == 0:
                    if bar < 60:
                        ev_data['ds_horiz'].append(edep_kev)
                        ev_data['ds_pl_DS1_H'].append(edep_kev)
                    else:
                        ev_data['ds_vert'].append(edep_kev)
                        ev_data['ds_pl_DS1_V'].append(edep_kev)
                elif pl == 1:
                    if bar < 60:
                        ev_data['ds_horiz'].append(edep_kev)
                        ev_data['ds_pl_DS2_H'].append(edep_kev)
                    else:
                        ev_data['ds_vert'].append(edep_kev)
                        ev_data['ds_pl_DS2_V'].append(edep_kev)
                elif pl == 2:
                    if bar < 60:
                        ev_data['ds_horiz'].append(edep_kev)
                        ev_data['ds_pl_DS3_H'].append(edep_kev)
                    else:
                        ev_data['ds_vert'].append(edep_kev)
                        ev_data['ds_pl_DS3_V'].append(edep_kev)
                elif pl == 3:
                    ev_data['ds_vert'].append(edep_kev)
                    ev_data['ds_pl_DS4_V'].append(edep_kev)
                    
        events.append(ev_data)
        
    return events

def compute_reduction_curves(events, thresholds):
    """
    Computes fraction of surviving hits R(E_th) = N(E_th) / N(0) for all categories.
    """
    categories = list(events[0].keys())
    # Baseline total hits at E_th = 0
    baseline_totals = {cat: sum(len(ev[cat]) for ev in events) for cat in categories}
    
    results = {cat: {'thresholds': thresholds, 'surviving_frac': [], 'mean_hits': [], 'pct_removed': []} for cat in categories}
    
    for eth in thresholds:
        for cat in categories:
            total_surviving = sum(sum(1 for edep in ev[cat] if edep > eth) for ev in events)
            base = baseline_totals[cat]
            frac = total_surviving / base if base > 0 else 0.0
            results[cat]['surviving_frac'].append(frac)
            results[cat]['mean_hits'].append(total_surviving / len(events))
            results[cat]['pct_removed'].append((1.0 - frac) * 100.0)
            
    # Compute exact reduction at marker thresholds
    exact_markers = {}
    test_pts = [TH_DS_HORIZ, TH_DS_VERT, TH_US_GEL, TH_US_AIR, TH_VETO_HORIZ, TH_VETO_VERT]
    for eth in test_pts:
        exact_markers[eth] = {}
        for cat in categories:
            total_surviving = sum(sum(1 for edep in ev[cat] if edep > eth) for ev in events)
            base = baseline_totals[cat]
            frac = total_surviving / base if base > 0 else 0.0
            exact_markers[eth][cat] = {
                'surviving_frac': frac,
                'pct_removed': (1.0 - frac) * 100.0,
                'mean_hits': total_surviving / len(events)
            }
            
    return results, baseline_totals, exact_markers

def plot_canvases(curves, markers, thresholds, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    th_arr = np.array(thresholds, dtype=np.float64)
    n_pts = len(thresholds)
    
    colors = [ROOT.kBlack, ROOT.kRed + 1, ROOT.kAzure + 2, ROOT.kGreen + 2, ROOT.kMagenta + 2, ROOT.kOrange + 7, ROOT.kTeal + 2]
    
    # -------------------------------------------------------------
    # 1. Canvas: Veto Subsystem & Planes
    # -------------------------------------------------------------
    c_veto = ROOT.TCanvas("c_p8_reduction_veto", "Veto Hit Reduction vs Threshold (newP8)", 1000, 750)
    c_veto.SetGrid(1, 1)
    c_veto.SetLeftMargin(0.12)
    c_veto.SetBottomMargin(0.12)
    
    mg_veto = ROOT.TMultiGraph()
    mg_veto.SetTitle("Veto Hit Reduction vs Threshold (newP8);Energy Threshold E_{th} [keV];Fraction of Retained Hits N(E_{th}) / N_{0}")
    
    veto_cats = [('veto_all', 'Veto Total'), ('veto_pl_0', 'Veto Plane 0 (H)'), ('veto_pl_1', 'Veto Plane 1 (H)')]
    if len(curves['veto_pl_2']['surviving_frac']) > 0 and sum(curves['veto_pl_2']['surviving_frac']) > 0:
        veto_cats.append(('veto_pl_2', 'Veto Plane 2 (V)'))
        
    leg_veto = ROOT.TLegend(0.48, 0.68, 0.88, 0.88)
    leg_veto.SetBorderSize(0)
    leg_veto.SetFillStyle(0)
    leg_veto.SetTextSize(0.035)
    
    graphs_veto = []
    for idx, (cat, lbl) in enumerate(veto_cats):
        y_vals = np.array(curves[cat]['surviving_frac'], dtype=np.float64)
        gr = ROOT.TGraph(n_pts, th_arr, y_vals)
        gr.SetLineColor(colors[idx % len(colors)])
        gr.SetLineWidth(3)
        mg_veto.Add(gr)
        leg_veto.AddEntry(gr, lbl, "l")
        graphs_veto.append(gr)
        
    mg_veto.Draw("AL")
    mg_veto.SetMinimum(0.0)
    mg_veto.SetMaximum(1.05)
    
    # Vertical line at 119.4 keV for Horizontal Veto
    rem_vh = markers[TH_VETO_HORIZ]['veto_horiz']['pct_removed']
    line_vh = ROOT.TLine(TH_VETO_HORIZ, 0.0, TH_VETO_HORIZ, 1.0)
    line_vh.SetLineColor(ROOT.kRed + 2)
    line_vh.SetLineStyle(2)
    line_vh.SetLineWidth(3)
    line_vh.Draw("SAME")
    
    lat_vh = ROOT.TLatex()
    lat_vh.SetNDC()
    lat_vh.SetTextSize(0.034)
    lat_vh.SetTextColor(ROOT.kRed + 2)
    lat_vh.DrawLatex(0.38, 0.50, f"H-Veto Thresh = {TH_VETO_HORIZ} keV: #bf{{{rem_vh:.1f}% removed}}")
    
    leg_veto.Draw()
    c_veto.SaveAs(f"{out_dir}/p8_reduction_veto.png")
    
    # -------------------------------------------------------------
    # 2. Canvas: Upstream (US) Subsystem & Planes
    # -------------------------------------------------------------
    c_us = ROOT.TCanvas("c_p8_reduction_us", "US Hit Reduction vs Threshold (newP8)", 1100, 750)
    c_us.SetGrid(1, 1)
    c_us.SetLeftMargin(0.12)
    c_us.SetBottomMargin(0.12)
    
    mg_us = ROOT.TMultiGraph()
    mg_us.SetTitle("Upstream (US) Hit Reduction vs Threshold (newP8);Energy Threshold E_{th} [keV];Fraction of Retained Hits N(E_{th}) / N_{0}")
    
    us_cats = [
        ('us_all', 'US Total'),
        ('us_pl_0', 'US Plane 0'),
        ('us_pl_1', 'US Plane 1'),
        ('us_pl_2', 'US Plane 2'),
        ('us_pl_3', 'US Plane 3'),
        ('us_pl_4', 'US Plane 4'),
    ]
    
    leg_us = ROOT.TLegend(0.62, 0.58, 0.88, 0.88)
    leg_us.SetBorderSize(0)
    leg_us.SetFillStyle(0)
    leg_us.SetTextSize(0.032)
    
    graphs_us = []
    for idx, (cat, lbl) in enumerate(us_cats):
        y_vals = np.array(curves[cat]['surviving_frac'], dtype=np.float64)
        gr = ROOT.TGraph(n_pts, th_arr, y_vals)
        gr.SetLineColor(colors[idx % len(colors)])
        gr.SetLineWidth(3 if cat == 'us_all' else 2)
        if cat != 'us_all':
            gr.SetLineStyle(idx + 1)
        mg_us.Add(gr)
        leg_us.AddEntry(gr, lbl, "l")
        graphs_us.append(gr)
        
    mg_us.Draw("AL")
    mg_us.SetMinimum(0.0)
    mg_us.SetMaximum(1.05)
    
    # Vertical line at 129.7 keV (Gel)
    rem_gel = markers[TH_US_GEL]['us_all']['pct_removed']
    line_gel = ROOT.TLine(TH_US_GEL, 0.0, TH_US_GEL, 1.0)
    line_gel.SetLineColor(ROOT.kGreen + 2)
    line_gel.SetLineStyle(2)
    line_gel.SetLineWidth(3)
    line_gel.Draw("SAME")
    
    # Vertical line at 227.7 keV (Air)
    rem_air = markers[TH_US_AIR]['us_all']['pct_removed']
    line_air = ROOT.TLine(TH_US_AIR, 0.0, TH_US_AIR, 1.0)
    line_air.SetLineColor(ROOT.kBlue + 2)
    line_air.SetLineStyle(2)
    line_air.SetLineWidth(3)
    line_air.Draw("SAME")
    
    lat_us = ROOT.TLatex()
    lat_us.SetNDC()
    lat_us.SetTextSize(0.034)
    lat_us.SetTextColor(ROOT.kGreen + 2)
    lat_us.DrawLatex(0.20, 0.42, f"US Gel = {TH_US_GEL} keV: #bf{{{rem_gel:.1f}% removed}}")
    lat_us.SetTextColor(ROOT.kBlue + 2)
    lat_us.DrawLatex(0.20, 0.36, f"US Air = {TH_US_AIR} keV: #bf{{{rem_air:.1f}% removed}}")
    
    leg_us.Draw()
    c_us.SaveAs(f"{out_dir}/p8_reduction_us.png")
    
    # -------------------------------------------------------------
    # 3. Canvas: Downstream (DS) Horizontal vs Vertical Subsystems
    # -------------------------------------------------------------
    c_ds_sub = ROOT.TCanvas("c_p8_reduction_ds_sub", "DS Subsystem Hit Reduction vs Threshold (newP8)", 1100, 750)
    c_ds_sub.SetGrid(1, 1)
    c_ds_sub.SetLeftMargin(0.12)
    c_ds_sub.SetBottomMargin(0.12)
    
    mg_ds_sub = ROOT.TMultiGraph()
    mg_ds_sub.SetTitle("Downstream (DS) Subsystems Hit Reduction (newP8);Energy Threshold E_{th} [keV];Fraction of Retained Hits N(E_{th}) / N_{0}")
    
    ds_sub_cats = [
        ('ds_all', 'DS Total'),
        ('ds_horiz', 'DS Horizontal Planes'),
        ('ds_vert', 'DS Vertical Planes')
    ]
    
    leg_ds_sub = ROOT.TLegend(0.55, 0.68, 0.88, 0.88)
    leg_ds_sub.SetBorderSize(0)
    leg_ds_sub.SetFillStyle(0)
    leg_ds_sub.SetTextSize(0.035)
    
    graphs_ds_sub = []
    for idx, (cat, lbl) in enumerate(ds_sub_cats):
        y_vals = np.array(curves[cat]['surviving_frac'], dtype=np.float64)
        gr = ROOT.TGraph(n_pts, th_arr, y_vals)
        gr.SetLineColor(colors[idx % len(colors)])
        gr.SetLineWidth(3)
        mg_ds_sub.Add(gr)
        leg_ds_sub.AddEntry(gr, lbl, "l")
        graphs_ds_sub.append(gr)
        
    mg_ds_sub.Draw("AL")
    mg_ds_sub.SetMinimum(0.0)
    mg_ds_sub.SetMaximum(1.05)
    
    # Vertical line at 22.1 keV (Vertical DS)
    rem_dsv = markers[TH_DS_VERT]['ds_vert']['pct_removed']
    line_dsv = ROOT.TLine(TH_DS_VERT, 0.0, TH_DS_VERT, 1.0)
    line_dsv.SetLineColor(ROOT.kAzure + 2)
    line_dsv.SetLineStyle(2)
    line_dsv.SetLineWidth(3)
    line_dsv.Draw("SAME")
    
    # Vertical line at 38.4 keV (Horizontal DS)
    rem_dsh = markers[TH_DS_HORIZ]['ds_horiz']['pct_removed']
    line_dsh = ROOT.TLine(TH_DS_HORIZ, 0.0, TH_DS_HORIZ, 1.0)
    line_dsh.SetLineColor(ROOT.kRed + 1)
    line_dsh.SetLineStyle(2)
    line_dsh.SetLineWidth(3)
    line_dsh.Draw("SAME")
    
    lat_ds = ROOT.TLatex()
    lat_ds.SetNDC()
    lat_ds.SetTextSize(0.034)
    lat_ds.SetTextColor(ROOT.kAzure + 2)
    lat_ds.DrawLatex(0.20, 0.42, f"V-DS Thresh = {TH_DS_VERT} keV: #bf{{{rem_dsv:.1f}% removed}}")
    lat_ds.SetTextColor(ROOT.kRed + 1)
    lat_ds.DrawLatex(0.20, 0.36, f"H-DS Thresh = {TH_DS_HORIZ} keV: #bf{{{rem_dsh:.1f}% removed}}")
    
    leg_ds_sub.Draw()
    c_ds_sub.SaveAs(f"{out_dir}/p8_reduction_ds_subsystems.png")
    
    # -------------------------------------------------------------
    # 4. Canvas: Downstream (DS) Individual Planes
    # -------------------------------------------------------------
    c_ds_pl = ROOT.TCanvas("c_p8_reduction_ds_planes", "DS Planes Hit Reduction vs Threshold (newP8)", 1100, 750)
    c_ds_pl.SetGrid(1, 1)
    c_ds_pl.SetLeftMargin(0.12)
    c_ds_pl.SetBottomMargin(0.12)
    
    mg_ds_pl = ROOT.TMultiGraph()
    mg_ds_pl.SetTitle("Downstream (DS) Individual Planes Hit Reduction (newP8);Energy Threshold E_{th} [keV];Fraction of Retained Hits N(E_{th}) / N_{0}")
    
    ds_pl_cats = [
        ('ds_pl_DS1_H', 'DS1 Horizontal'),
        ('ds_pl_DS1_V', 'DS1 Vertical'),
        ('ds_pl_DS2_H', 'DS2 Horizontal'),
        ('ds_pl_DS2_V', 'DS2 Vertical'),
        ('ds_pl_DS3_H', 'DS3 Horizontal'),
        ('ds_pl_DS3_V', 'DS3 Vertical'),
        ('ds_pl_DS4_V', 'DS4 Vertical')
    ]
    
    leg_ds_pl = ROOT.TLegend(0.60, 0.52, 0.88, 0.88)
    leg_ds_pl.SetBorderSize(0)
    leg_ds_pl.SetFillStyle(0)
    leg_ds_pl.SetTextSize(0.030)
    
    graphs_ds_pl = []
    for idx, (cat, lbl) in enumerate(ds_pl_cats):
        y_vals = np.array(curves[cat]['surviving_frac'], dtype=np.float64)
        gr = ROOT.TGraph(n_pts, th_arr, y_vals)
        gr.SetLineColor(colors[idx % len(colors)])
        gr.SetLineWidth(2)
        mg_ds_pl.Add(gr)
        leg_ds_pl.AddEntry(gr, lbl, "l")
        graphs_ds_pl.append(gr)
        
    mg_ds_pl.Draw("AL")
    mg_ds_pl.SetMinimum(0.0)
    mg_ds_pl.SetMaximum(1.05)
    
    line_dsv.Draw("SAME")
    line_dsh.Draw("SAME")
    lat_ds.DrawLatex(0.20, 0.42, f"V-DS Thresh = {TH_DS_VERT} keV: #bf{{{rem_dsv:.1f}% removed}}")
    lat_ds.DrawLatex(0.20, 0.36, f"H-DS Thresh = {TH_DS_HORIZ} keV: #bf{{{rem_dsh:.1f}% removed}}")
    
    leg_ds_pl.Draw()
    c_ds_pl.SaveAs(f"{out_dir}/p8_reduction_ds_planes.png")
    
    return [c_veto, c_us, c_ds_sub, c_ds_pl]

def main():
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    chain = build_p8_chain("/eos/user/i/idioniso/snd-numu-charm/data/newP8", max_partitions=40)
    events = extract_all_bar_energies(chain, max_events=4000)
    print(f"Extracted bar energies from {len(events)} events.")
    
    # Define threshold grid: 0 to 500 keV in steps of 2 keV (251 points)
    thresholds = list(np.linspace(0.0, 500.0, 251))
    
    curves, baselines, markers = compute_reduction_curves(events, thresholds)
    
    # Save ROOT file
    rfile = ROOT.TFile("output/p8_threshold_reduction.root", "RECREATE")
    canvases = plot_canvases(curves, markers, thresholds, out_dir=out_dir)
    for c in canvases:
        c.Write()
    rfile.Close()
    
    # Save JSON summary
    summary = {
        'baseline_hits': baselines,
        'marker_evaluations': markers,
        'curves': {cat: {'thresholds': curves[cat]['thresholds'], 'surviving_frac': curves[cat]['surviving_frac'], 'pct_removed': curves[cat]['pct_removed']} for cat in curves}
    }
    with open("output/p8_threshold_reduction.json", "w") as jf:
        json.dump(summary, jf, indent=2)
        
    # Write readable text report
    with open("output/p8_threshold_reduction.txt", "w") as tf:
        tf.write("========================================================================================\n")
        tf.write("SND@LHC NuMu Charm Simulation (newP8): Hit Reduction vs Energy Threshold\n")
        tf.write("========================================================================================\n\n")
        
        tf.write("EXACT HIT REDUCTION AT SPECIFIED THRESHOLDS:\n")
        tf.write("-" * 75 + "\n")
        tf.write(f"1. Horizontal DS Hits (Threshold = {TH_DS_HORIZ} keV):\n")
        tf.write(f"   - Hits removed: {markers[TH_DS_HORIZ]['ds_horiz']['pct_removed']:.2f}%\n")
        tf.write(f"   - Fraction retained: {markers[TH_DS_HORIZ]['ds_horiz']['surviving_frac']*100:.2f}%\n\n")
        
        tf.write(f"2. Vertical DS Hits (Threshold = {TH_DS_VERT} keV):\n")
        tf.write(f"   - Hits removed: {markers[TH_DS_VERT]['ds_vert']['pct_removed']:.2f}%\n")
        tf.write(f"   - Fraction retained: {markers[TH_DS_VERT]['ds_vert']['surviving_frac']*100:.2f}%\n\n")
        
        tf.write(f"3. US Hits with Gel at Readout (Threshold = {TH_US_GEL} keV):\n")
        tf.write(f"   - US Total hits removed: {markers[TH_US_GEL]['us_all']['pct_removed']:.2f}%\n")
        tf.write(f"   - Fraction retained: {markers[TH_US_GEL]['us_all']['surviving_frac']*100:.2f}%\n\n")
        
        tf.write(f"4. US Hits without Gel (Air) at Readout (Threshold = {TH_US_AIR} keV):\n")
        tf.write(f"   - US Total hits removed: {markers[TH_US_AIR]['us_all']['pct_removed']:.2f}%\n")
        tf.write(f"   - Fraction retained: {markers[TH_US_AIR]['us_all']['surviving_frac']*100:.2f}%\n\n")
        
        tf.write(f"5. Horizontal Veto Hits (Threshold = {TH_VETO_HORIZ} keV):\n")
        tf.write(f"   - Hits removed: {markers[TH_VETO_HORIZ]['veto_horiz']['pct_removed']:.2f}%\n")
        tf.write(f"   - Fraction retained: {markers[TH_VETO_HORIZ]['veto_horiz']['surviving_frac']*100:.2f}%\n\n")
        
        if baselines['veto_pl_2'] > 0:
            tf.write(f"6. Vertical Veto Hits (Threshold = {TH_VETO_VERT} keV):\n")
            tf.write(f"   - Hits removed: {markers[TH_VETO_VERT]['veto_vert']['pct_removed']:.2f}%\n")
            tf.write(f"   - Fraction retained: {markers[TH_VETO_VERT]['veto_vert']['surviving_frac']*100:.2f}%\n\n")
        else:
            tf.write("6. Vertical Veto Hits: Not present in this geometry (pre-2024 setup).\n\n")
            
        tf.write("-" * 75 + "\n")
        tf.write("PLANE-BY-PLANE REDUCTION SUMMARY AT SPECIFIED THRESHOLDS:\n")
        tf.write(f"{'Category':<16} | {'Threshold [keV]':<16} | {'Baseline Hits':<14} | {'% Removed':<10} | {'% Retained':<10}\n")
        tf.write("-" * 75 + "\n")
        
        plane_specs = [
            ('veto_pl_0', TH_VETO_HORIZ),
            ('veto_pl_1', TH_VETO_HORIZ),
            ('us_pl_0', TH_US_GEL),
            ('us_pl_1', TH_US_GEL),
            ('us_pl_2', TH_US_GEL),
            ('us_pl_3', TH_US_GEL),
            ('us_pl_4', TH_US_GEL),
            ('ds_pl_DS1_H', TH_DS_HORIZ),
            ('ds_pl_DS1_V', TH_DS_VERT),
            ('ds_pl_DS2_H', TH_DS_HORIZ),
            ('ds_pl_DS2_V', TH_DS_VERT),
            ('ds_pl_DS3_H', TH_DS_HORIZ),
            ('ds_pl_DS3_V', TH_DS_VERT),
            ('ds_pl_DS4_V', TH_DS_VERT)
        ]
        for cat, th in plane_specs:
            rem = markers[th][cat]['pct_removed']
            ret = markers[th][cat]['surviving_frac'] * 100.0
            base = baselines[cat]
            tf.write(f"{cat:<16} | {th:<16.1f} | {base:<14d} | {rem:<10.2f}% | {ret:<10.2f}%\n")
        tf.write("-" * 75 + "\n")
        
    print("\nProcessing complete! Output saved to:")
    print("  - output/p8_threshold_reduction.root")
    print("  - output/p8_threshold_reduction.json")
    print("  - output/p8_threshold_reduction.txt")
    print("  - plots/p8_reduction_*.png")

if __name__ == '__main__':
    main()
