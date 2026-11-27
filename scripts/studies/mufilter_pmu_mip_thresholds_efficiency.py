#!/usr/bin/env python3
"""
MIP Thresholds and Efficiencies with MuFilterPoint Thresholding in PMU Simulation
and Comparison to Run 5408 Data.

Features:
  - Uses MuFilterPoint cumulative energy deposition per bar to filter MC hits before applying multiplicity cuts.
  - Scales MC event weights by 0.406569 (Run 5408 luminosity = 0.4066 fb^-1).
  - Plots true energy deposition distributions (MeV) for MC MIP vs Other hits.
  - Plots QDC distributions for Data MIP vs Other hits.
  - Computes and plots in-situ detection efficiency per plane for Data vs MC.
"""

import os
import sys
import numpy as np
import ROOT
import SndlhcGeo
import SndlhcTracking

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kViridis)

# Physical detection thresholds (MeV)
TH_VETO = 1.20 # MeV
TH_US   = 1.50 # MeV (single MIP traversing 3 cm bar)
TH_DSH  = 0.90 # MeV
TH_DSV  = 1.35 # MeV

# Luminosity scaling for Run 5408 (0.406569 fb^-1 / 1.0 fb^-1)
LUMI_SCALE_MC = 0.406569

def build_detector_geometry(geo):
    mufi = geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()
    bars = {}
    
    # Veto (sub 1, planes 0..1, bars 0..6)
    for pl in range(2):
        for b in range(7):
            did = 10000 + pl * 1000 + b
            mufi.GetPosition(did, A, B)
            bars[did] = {
                'sub': 1, 'plane': pl, 'bar': b,
                'x_min': min(A.X(), B.X()), 'x_max': max(A.X(), B.X()),
                'y_min': min(A.Y(), B.Y()), 'y_max': max(A.Y(), B.Y()),
                'z': 0.5 * (A.Z() + B.Z()),
                'width': 6.0,
                'is_vert': True
            }
            
    # US (sub 2, planes 0..4, bars 0..9)
    for pl in range(5):
        for b in range(10):
            did = 20000 + pl * 1000 + b
            mufi.GetPosition(did, A, B)
            bars[did] = {
                'sub': 2, 'plane': pl, 'bar': b,
                'x_min': min(A.X(), B.X()), 'x_max': max(A.X(), B.X()),
                'y_min': min(A.Y(), B.Y()), 'y_max': max(A.Y(), B.Y()),
                'z': 0.5 * (A.Z() + B.Z()),
                'width': 6.0,
                'is_vert': False
            }
            
    # DS (sub 3, planes 0..3)
    for pl in range(4):
        b_start = 60 if pl == 3 else 0
        b_max = 120
        for b in range(b_start, b_max):
            did = 30000 + pl * 1000 + b
            mufi.GetPosition(did, A, B)
            bars[did] = {
                'sub': 3, 'plane': pl, 'bar': b,
                'x_min': min(A.X(), B.X()), 'x_max': max(A.X(), B.X()),
                'y_min': min(A.Y(), B.Y()), 'y_max': max(A.Y(), B.Y()),
                'z': 0.5 * (A.Z() + B.Z()),
                'width': 1.0,
                'is_vert': (b >= 60)
            }
    return bars

def get_hit_signal(hit):
    try:
        s_map = hit.SumOfSignals()
        if "Sum" in s_map:
            val = float(s_map["Sum"])
            if val > 0: return val
    except Exception:
        pass
    try:
        sigs = [v for k, v in hit.GetAllSignals()]
        if sigs: return float(sum(sigs))
    except Exception:
        pass
    return 1.0

def process_sample(file_list, tree_name, geofile, is_mc=False, max_events=20000, target_passed=100):
    print(f"\n==========================================")
    print(f"Processing {'Simulation (PMU)' if is_mc else 'Real Data (Run 5408)'}...")
    print(f"==========================================")
    
    geo = SndlhcGeo.GeoInterface(geofile)
    bars_map = build_detector_geometry(geo)
    
    chain = ROOT.TChain(tree_name)
    for f in file_list:
        chain.Add(f)
        
    if is_mc:
        chain_mc = ROOT.TChain("mctruth")
        for f in file_list:
            chain_mc.Add(f)
        chain.AddFriend(chain_mc)
        
    n_entries = min(max_events, chain.GetEntries()) if not is_mc else max_events
    print(f"Scanning up to {n_entries} entries (target passing: {target_passed})...", flush=True)
    
    ioman = ROOT.FairRootManager.Instance()
    ioman.SetInChain(chain)
    
    trackTask = SndlhcTracking.Tracking()
    trackTask.SetName("simpleTracking")
    trackTask.SetTrackClassType(False) # sndRecoTrack
    trackTask.Init()
    
    results = {
        'veto_mip_val': [], 'veto_other_val': [],
        'us_mip_val': [], 'us_other_val': [],
        'dsh_mip_val': [], 'dsh_other_val': [],
        'dsv_mip_val': [], 'dsv_other_val': [],
        'weights_veto_mip': [], 'weights_veto_other': [],
        'weights_us_mip': [], 'weights_us_other': [],
        'weights_dsh_mip': [], 'weights_dsh_other': [],
        'weights_dsv_mip': [], 'weights_dsv_other': [],
        'eff_expected': {p_name: 0.0 for p_name in [
            'Veto_0', 'Veto_1',
            'US_0', 'US_1', 'US_2', 'US_3', 'US_4',
            'DS1_H', 'DS1_V', 'DS2_H', 'DS2_V', 'DS3_H', 'DS3_V', 'DS4_V'
        ]},
        'eff_detected': {p_name: 0.0 for p_name in [
            'Veto_0', 'Veto_1',
            'US_0', 'US_1', 'US_2', 'US_3', 'US_4',
            'DS1_H', 'DS1_V', 'DS2_H', 'DS2_V', 'DS3_H', 'DS3_V', 'DS4_V'
        ]},
        'events_passed': 0
    }
    
    prev_time = None
    
    for iev in range(n_entries):
        if (iev + 1) % 2000 == 0 or iev == 0:
            print(f"  Event {iev+1}/{n_entries} (Passed: {results['events_passed']}) ...", flush=True)
            
        chain.GetEvent(iev)
        
        # 1. Data-only cuts
        if not is_mc:
            h = chain.EventHeader
            if not h.isIP1(): continue
            if h.GetBeamMode() != 11: continue
            
            t = h.GetEventTime()
            if prev_time is not None:
                dt = abs(t - prev_time)
                if dt <= 100:
                    prev_time = t
                    continue
            prev_time = t
            
        # Weight scaling
        if is_mc:
            raw_w = float(getattr(chain, "mc_weight", 0.0))
            if raw_w <= 0.0:
                continue
            weight = raw_w * LUMI_SCALE_MC
        else:
            weight = 1.0
            
        # 2. Multiplicity Filtering
        nsf = len(chain.Digi_ScifiHits)
        if not (6 <= nsf <= 27):
            continue
            
        if is_mc:
            # Accumulate energy loss per bar from MuFilterPoint
            bar_edep = {}
            for pt in chain.MuFilterPoint:
                did = pt.GetDetectorID()
                bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss() * 1000.0 # MeV
                
            fired_bars = {}
            nv = 0
            nu = 0
            nd = 0
            for did, edep in bar_edep.items():
                sub = did // 10000
                bar = did % 1000
                if sub == 1 and edep >= TH_VETO:
                    nv += 1
                    fired_bars[did] = edep
                elif sub == 2 and edep >= TH_US:
                    nu += 1
                    fired_bars[did] = edep
                elif sub == 3:
                    th = TH_DSH if bar < 60 else TH_DSV
                    if edep >= th:
                        nd += 1
                        fired_bars[did] = edep
        else:
            fired_bars = {}
            nv = 0
            nu = 0
            nd = 0
            for hit in chain.Digi_MuFilterHits:
                if not hit.isValid(): continue
                did = hit.GetDetectorID()
                sub = did // 10000
                if sub == 1: nv += 1
                elif sub == 2: nu += 1
                elif sub == 3: nd += 1
                fired_bars[did] = get_hit_signal(hit)
                
        # Multiplicity window cut
        if not (2 <= nv <= 4 and 4 <= nu <= 8 and 6 <= nd <= 13):
            continue
            
        # 3. Track Reconstruction
        trackTask.fittedTracks.Delete()
        trackTask.ExecuteTask(option='ScifiDS')
        
        scifi_trks = []
        ds_trks = []
        for trk in trackTask.fittedTracks:
            tt = trk.getTrackType()
            chi2ndf = trk.getChi2Ndf()
            sl_x = trk.getSlopeXZ()
            sl_y = trk.getSlopeYZ()
            if chi2ndf <= 5.0 and abs(sl_x) <= 0.1 and abs(sl_y) <= 0.1:
                if tt == 1: scifi_trks.append(trk)
                elif tt == 3: ds_trks.append(trk)
                
        if not scifi_trks or not ds_trks:
            continue
            
        # 4. Track Matching at z = 430 cm (< 3 cm)
        best_match = None
        min_dist = 999.0
        for st in scifi_trks:
            ps = st.getPointAtZ(430.0)
            for dt in ds_trks:
                pd = dt.getPointAtZ(430.0)
                dist = np.hypot(ps.X() - pd.X(), ps.Y() - pd.Y())
                if dist < 3.0 and dist < min_dist:
                    min_dist = dist
                    best_match = (st, dt)
                    
        if best_match is None:
            continue
            
        results['events_passed'] += 1
        st_match, dt_match = best_match
        
        # 5. Bar Traversal and Classification
        crossed_bars = set()
        
        # (A) Veto bars crossed by SciFi track
        for pl in range(2):
            p_name = f"Veto_{pl}"
            for b in range(7):
                did = 10000 + pl * 1000 + b
                b_info = bars_map[did]
                p_trk = st_match.getPointAtZ(b_info['z'])
                x_trk, y_trk = p_trk.X(), p_trk.Y()
                x_c = 0.5 * (b_info['x_min'] + b_info['x_max'])
                
                if abs(x_trk - x_c) <= 3.0 and (b_info['y_min'] - 2.0 <= y_trk <= b_info['y_max'] + 2.0):
                    crossed_bars.add(did)
                    results['eff_expected'][p_name] += weight
                    if did in fired_bars:
                        results['eff_detected'][p_name] += weight
                        
        # (B) US bars crossed by DS track
        for pl in range(5):
            p_name = f"US_{pl}"
            for b in range(10):
                did = 20000 + pl * 1000 + b
                b_info = bars_map[did]
                p_trk = dt_match.getPointAtZ(b_info['z'])
                x_trk, y_trk = p_trk.X(), p_trk.Y()
                y_c = 0.5 * (b_info['y_min'] + b_info['y_max'])
                
                if abs(y_trk - y_c) <= 3.0 and (b_info['x_min'] - 2.0 <= x_trk <= b_info['x_max'] + 2.0):
                    crossed_bars.add(did)
                    results['eff_expected'][p_name] += weight
                    if did in fired_bars:
                        results['eff_detected'][p_name] += weight

        # (C) DS bars within 1 cm of DS track
        for pl in range(4):
            # Horizontal (Planes 0..2)
            if pl < 3:
                p_name_h = f"DS{pl+1}_H"
                for b in range(60):
                    did = 30000 + pl * 1000 + b
                    b_info = bars_map[did]
                    p_trk = dt_match.getPointAtZ(b_info['z'])
                    x_trk, y_trk = p_trk.X(), p_trk.Y()
                    y_c = 0.5 * (b_info['y_min'] + b_info['y_max'])
                    
                    if abs(y_trk - y_c) <= 1.0 and (b_info['x_min'] - 1.0 <= x_trk <= b_info['x_max'] + 1.0):
                        crossed_bars.add(did)
                        results['eff_expected'][p_name_h] += weight
                        if did in fired_bars:
                            results['eff_detected'][p_name_h] += weight
                            
            # Vertical (Planes 0..3)
            p_name_v = f"DS{pl+1}_V"
            for b in range(60, 120):
                did = 30000 + pl * 1000 + b
                b_info = bars_map[did]
                p_trk = dt_match.getPointAtZ(b_info['z'])
                x_trk, y_trk = p_trk.X(), p_trk.Y()
                x_c = 0.5 * (b_info['x_min'] + b_info['x_max'])
                
                if abs(x_trk - x_c) <= 1.0 and (b_info['y_min'] - 1.0 <= y_trk <= b_info['y_max'] + 1.0):
                    crossed_bars.add(did)
                    results['eff_expected'][p_name_v] += weight
                    if did in fired_bars:
                        results['eff_detected'][p_name_v] += weight

        # 6. Record values for MIP Hits vs Other Hits
        for did, val in fired_bars.items():
            sub = did // 10000
            bar = did % 1000
            is_mip = (did in crossed_bars)
            
            if sub == 1:
                if is_mip:
                    results['veto_mip_val'].append(val)
                    results['weights_veto_mip'].append(weight)
                else:
                    results['veto_other_val'].append(val)
                    results['weights_veto_other'].append(weight)
            elif sub == 2:
                if is_mip:
                    results['us_mip_val'].append(val)
                    results['weights_us_mip'].append(weight)
                else:
                    results['us_other_val'].append(val)
                    results['weights_us_other'].append(weight)
            elif sub == 3:
                if bar < 60:
                    if is_mip:
                        results['dsh_mip_val'].append(val)
                        results['weights_dsh_mip'].append(weight)
                    else:
                        results['dsh_other_val'].append(val)
                        results['weights_dsh_other'].append(weight)
                else:
                    if is_mip:
                        results['dsv_mip_val'].append(val)
                        results['weights_dsv_mip'].append(weight)
                    else:
                        results['dsv_other_val'].append(val)
                        results['weights_dsv_other'].append(weight)
                        
        if results['events_passed'] >= target_passed:
            print(f"Target of {target_passed} clean muons reached at event {iev+1}!", flush=True)
            break
                        
    print(f"Done. Processed {iev+1} events, selected {results['events_passed']} clean passing muons.")
    return results

def create_superimposed_plots(data_res, mc_res, out_root_path):
    f_out = ROOT.TFile.Open(out_root_path, "RECREATE")
    
    # 1. Canvas: True Energy Deposition in Simulation (MeV)
    c_mc_edep = ROOT.TCanvas("c_mc_edep", "PMU Simulation: True Energy Loss in Bars (MIP vs Other)", 1600, 1200)
    c_mc_edep.Divide(2, 2)
    
    subsystems_mc = [
        ('Veto', 'veto_mip_val', 'veto_other_val', 'weights_veto_mip', 'weights_veto_other', 50, 0, 10, "Energy Loss (MeV)"),
        ('US', 'us_mip_val', 'us_other_val', 'weights_us_mip', 'weights_us_other', 50, 0, 25, "Energy Loss (MeV)"),
        ('DS_H', 'dsh_mip_val', 'dsh_other_val', 'weights_dsh_mip', 'weights_dsh_other', 50, 0, 10, "Energy Loss (MeV)"),
        ('DS_V', 'dsv_mip_val', 'dsv_other_val', 'weights_dsv_mip', 'weights_dsv_other', 50, 0, 10, "Energy Loss (MeV)")
    ]
    
    hist_store = []
    
    for idx, (sub_name, mip_key, oth_key, w_mip_key, w_oth_key, nbins, xmin, xmax, xlabel) in enumerate(subsystems_mc):
        c_mc_edep.cd(idx + 1)
        ROOT.gPad.SetGrid(1, 1)
        
        h_mip = ROOT.TH1D(f"h_mc_mip_{sub_name}", f"Simulation {sub_name}: Energy Deposition (MIP vs Other);{xlabel};Normalized Entries", nbins, xmin, xmax)
        h_oth = ROOT.TH1D(f"h_mc_oth_{sub_name}", f"Simulation {sub_name}: Other Hits;{xlabel};Normalized Entries", nbins, xmin, xmax)
        
        for val, w in zip(mc_res[mip_key], mc_res[w_mip_key]): h_mip.Fill(val, w)
        for val, w in zip(mc_res[oth_key], mc_res[w_oth_key]): h_oth.Fill(val, w)
        
        if h_mip.Integral() > 0: h_mip.Scale(1.0 / h_mip.Integral())
        if h_oth.Integral() > 0: h_oth.Scale(1.0 / h_oth.Integral())
        
        h_mip.SetLineColor(ROOT.kRed + 1)
        h_mip.SetLineWidth(3)
        h_oth.SetLineColor(ROOT.kBlue + 1)
        h_oth.SetLineWidth(2)
        h_oth.SetLineStyle(2)
        
        max_y = max(h_mip.GetMaximum(), h_oth.GetMaximum(), 0.05) * 1.3
        h_mip.SetMaximum(max_y)
        
        h_mip.Draw("HIST")
        h_oth.Draw("HIST SAME")
        
        leg = ROOT.TLegend(0.45, 0.68, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.AddEntry(h_mip, f"MIP Hits (#mu={h_mip.GetMean():.2f} MeV)", "l")
        leg.AddEntry(h_oth, f"Other Hits (#mu={h_oth.GetMean():.2f} MeV)", "l")
        leg.Draw()
        
        hist_store.extend([h_mip, h_oth, leg])
        
    c_mc_edep.Write()
    
    # 2. Canvas: Data QDC Signals (Run 5408)
    c_data_qdc = ROOT.TCanvas("c_data_qdc", "Data (Run 5408): QDC Distributions (MIP vs Other)", 1600, 1200)
    c_data_qdc.Divide(2, 2)
    
    subsystems_data = [
        ('Veto', 'veto_mip_val', 'veto_other_val', 60, 0, 150, "QDC Units"),
        ('US', 'us_mip_val', 'us_other_val', 60, 0, 200, "QDC Units"),
        ('DS_H', 'dsh_mip_val', 'dsh_other_val', 50, 0, 100, "QDC Units"),
        ('DS_V', 'dsv_mip_val', 'dsv_other_val', 50, 0, 100, "QDC Units")
    ]
    
    for idx, (sub_name, mip_key, oth_key, nbins, xmin, xmax, xlabel) in enumerate(subsystems_data):
        c_data_qdc.cd(idx + 1)
        ROOT.gPad.SetGrid(1, 1)
        
        h_d_mip = ROOT.TH1D(f"h_d_mip_{sub_name}", f"Data {sub_name}: MIP Hits (Run 5408);{xlabel};Normalized Entries", nbins, xmin, xmax)
        h_d_oth = ROOT.TH1D(f"h_d_oth_{sub_name}", f"Data {sub_name}: Other Hits;{xlabel};Normalized Entries", nbins, xmin, xmax)
        
        for val in data_res[mip_key]: h_d_mip.Fill(val)
        for val in data_res[oth_key]: h_d_oth.Fill(val)
        
        if h_d_mip.Integral() > 0: h_d_mip.Scale(1.0 / h_d_mip.Integral())
        if h_d_oth.Integral() > 0: h_d_oth.Scale(1.0 / h_d_oth.Integral())
        
        h_d_mip.SetLineColor(ROOT.kBlack)
        h_d_mip.SetLineWidth(3)
        h_d_oth.SetLineColor(ROOT.kGray + 2)
        h_d_oth.SetLineWidth(2)
        h_d_oth.SetLineStyle(2)
        
        max_y = max(h_d_mip.GetMaximum(), h_d_oth.GetMaximum(), 0.05) * 1.3
        h_d_mip.SetMaximum(max_y)
        
        h_d_mip.Draw("HIST")
        h_d_oth.Draw("HIST SAME")
        
        leg = ROOT.TLegend(0.45, 0.68, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.AddEntry(h_d_mip, f"Data MIP (#mu={h_d_mip.GetMean():.1f})", "l")
        leg.AddEntry(h_d_oth, f"Data Other (#mu={h_d_oth.GetMean():.1f})", "l")
        leg.Draw()
        
        hist_store.extend([h_d_mip, h_d_oth, leg])
        
    c_data_qdc.Write()
    
    # 3. Canvas: Efficiency per Plane (Data vs MC)
    c_eff = ROOT.TCanvas("c_mufilter_plane_efficiency", "MIP Detection Efficiency per Plane (Data vs PMU MC)", 1400, 700)
    c_eff.SetGrid(1, 1)
    
    plane_keys = [
        'Veto_0', 'Veto_1',
        'US_0', 'US_1', 'US_2', 'US_3', 'US_4',
        'DS1_H', 'DS1_V', 'DS2_H', 'DS2_V', 'DS3_H', 'DS3_V', 'DS4_V'
    ]
    
    n_planes = len(plane_keys)
    h_eff_data = ROOT.TH1D("h_eff_data", "MIP Detection Efficiency per Plane;Detector Plane;Efficiency", n_planes, 0.5, n_planes + 0.5)
    h_eff_mc = ROOT.TH1D("h_eff_mc", "MIP Detection Efficiency per Plane (MC);Detector Plane;Efficiency", n_planes, 0.5, n_planes + 0.5)
    
    print("\n==========================================")
    print("MIP Detection Efficiencies Summary:")
    print("==========================================")
    print(f"{'Plane':10s} | {'Data Eff (%)':12s} | {'MC Eff (%)':12s}")
    print("-" * 42)
    
    for i, pk in enumerate(plane_keys, start=1):
        h_eff_data.GetXaxis().SetBinLabel(i, pk)
        h_eff_mc.GetXaxis().SetBinLabel(i, pk)
        
        exp_d = data_res['eff_expected'][pk]
        det_d = data_res['eff_detected'][pk]
        eff_d = (det_d / exp_d) if exp_d > 0 else 0.0
        
        exp_m = mc_res['eff_expected'][pk]
        det_m = mc_res['eff_detected'][pk]
        eff_m = (det_m / exp_m) if exp_m > 0 else 0.0
        
        h_eff_data.SetBinContent(i, eff_d)
        h_eff_mc.SetBinContent(i, eff_m)
        
        print(f"{pk:10s} | {eff_d*100.0:11.2f}% | {eff_m*100.0:11.2f}%")
        
    h_eff_data.SetLineColor(ROOT.kBlack)
    h_eff_data.SetMarkerColor(ROOT.kBlack)
    h_eff_data.SetMarkerStyle(20)
    h_eff_data.SetMarkerSize(1.3)
    h_eff_data.SetLineWidth(2)
    h_eff_data.SetMinimum(0.40)
    h_eff_data.SetMaximum(1.05)
    
    h_eff_mc.SetLineColor(ROOT.kRed + 1)
    h_eff_mc.SetMarkerColor(ROOT.kRed + 1)
    h_eff_mc.SetMarkerStyle(22)
    h_eff_mc.SetMarkerSize(1.4)
    h_eff_mc.SetLineWidth(2)
    
    h_eff_data.Draw("P")
    h_eff_mc.Draw("P SAME")
    
    leg_eff = ROOT.TLegend(0.65, 0.20, 0.88, 0.35)
    leg_eff.SetBorderSize(1)
    leg_eff.AddEntry(h_eff_data, "Data (Run 5408)", "lp")
    leg_eff.AddEntry(h_eff_mc, "PMU Simulation (Thresholded)", "lp")
    leg_eff.Draw()
    
    c_eff.Write()
    f_out.Close()
    print(f"\nSaved ROOT file and canvases to: {out_root_path}")
    
    # Export PNGs as well
    os.makedirs("plots", exist_ok=True)
    c_mc_edep.SaveAs("plots/mufilter_mc_edep_distributions.png")
    c_data_qdc.SaveAs("plots/mufilter_data_qdc_distributions.png")
    c_eff.SaveAs("plots/mufilter_plane_efficiency_ti18.png")

def main():
    geofile = "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/geofile_full.Ntuple-TGeant4.root"
    
    # 1. Real Data (Run 5408, partition 0002)
    data_files = [
        "/eos/experiment/sndlhc/convertedData/physics/2022/run_005408/sndsw_raw-0002.root"
    ]
    data_res = process_sample(data_files, "rawConv", geofile, is_mc=False, max_events=10000, target_passed=250)
    
    # 2. PMU Simulation Sample (scan up to 400000 entries or target 100 passed events)
    mc_files = [
        "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP_truth.root"
    ]
    mc_res = process_sample(mc_files, "cbmsim", geofile, is_mc=True, max_events=400000, target_passed=100)
    
    out_root = "plots/mufilter_pmu_mip_thresholds_efficiency.root"
    create_superimposed_plots(data_res, mc_res, out_root)

if __name__ == "__main__":
    main()
