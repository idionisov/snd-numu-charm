#!/usr/bin/env python3
"""
Prompt Muon (PMU 2022) vs Real TI18 Data (Run 5408) Energy Threshold Scan.

Evaluates minimum energy thresholds:
  - 38.4 keV for Horizontal DS
  - 22.1 keV for Vertical DS
  - 129.7 keV for US with gel
  - 227.7 keV for US without gel (air)
  - 119.4 keV for Horizontal Veto
  - 69.9 keV for Vertical Veto

Scans scale factors k in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0].
Generates multi-panel comparison canvases for each k and computes the Wasserstein-1 distance
to identify the scale factor that best replicates real TI18 data.
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

# Baseline thresholds at k = 1.0 (in keV)
TH_DS_HORIZ_BASE   = 38.4
TH_DS_VERT_BASE    = 22.1
TH_US_GEL_BASE     = 129.7
TH_US_AIR_BASE     = 227.7
TH_VETO_HORIZ_BASE = 119.4
TH_VETO_VERT_BASE  = 69.9

K_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

def extract_data_events(data_path, max_events=12000, target_muons=1000):
    print(f"Extracting real data from {data_path} (target: {target_muons} muons)...", flush=True)
    f = ROOT.TFile.Open(data_path)
    t = f.Get("rawConv")
    t.SetBranchStatus("*", 0)
    t.SetBranchStatus("EventHeader*", 1)
    t.SetBranchStatus("Digi_MuFilterHits*", 1)
    t.SetBranchStatus("Digi_ScifiHits*", 1)

    n_tot = min(max_events, t.GetEntries())
    events = []
    prev_time = None

    for iev in range(n_tot):
        t.GetEntry(iev)
        h = t.EventHeader
        if not h.isIP1(): continue
        if h.GetBeamMode() != 11: continue

        ev_time = h.GetEventTime()
        if prev_time is not None and abs(ev_time - prev_time) <= 100:
            prev_time = ev_time
            continue
        prev_time = ev_time

        mufi = t.Digi_MuFilterHits
        scifi = t.Digi_ScifiHits
        nsf = scifi.GetEntries()
        if not (6 <= nsf <= 30):
            continue

        veto_hits = 0
        us_hits = 0
        ds_hits = 0
        ds_h_hits = 0
        ds_v_hits = 0
        us_pl = [0] * 5
        ds_pl_h = [0] * 3
        ds_pl_v = [0] * 4

        for hit in mufi:
            if not hit.isValid(): continue
            did = hit.GetDetectorID()
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000

            if sub == 1:
                veto_hits += 1
            elif sub == 2:
                us_hits += 1
                if 0 <= pl < 5:
                    us_pl[pl] += 1
            elif sub == 3:
                ds_hits += 1
                if pl < 3:
                    if bar < 60:
                        ds_h_hits += 1
                        ds_pl_h[pl] += 1
                    else:
                        ds_v_hits += 1
                        ds_pl_v[pl] += 1
                elif pl == 3:
                    ds_v_hits += 1
                    ds_pl_v[3] += 1

        # Through-going muon selection
        if 1 <= veto_hits <= 4 and 4 <= us_hits <= 8 and 6 <= ds_hits <= 16:
            events.append({
                'veto': veto_hits,
                'us': us_hits,
                'ds': ds_hits,
                'ds_h': ds_h_hits,
                'ds_v': ds_v_hits,
                'us_pl': us_pl,
                'ds_pl_h': ds_pl_h,
                'ds_pl_v': ds_pl_v,
            })
            if len(events) >= target_muons:
                break

    print(f"Data: extracted {len(events)} clean through-going muon events.", flush=True)
    return events


def extract_mc_events(mc_path, max_events=120000, target_muons=600):
    print(f"Extracting MC events from {mc_path} (target: {target_muons} muons)...", flush=True)
    f = ROOT.TFile.Open(mc_path)
    t = f.Get("cbmsim")
    t.SetBranchStatus("*", 0)
    t.SetBranchStatus("MuFilterPoint*", 1)
    t.SetBranchStatus("Digi_MuFilterHits*", 1)
    t.SetBranchStatus("Digi_ScifiHits*", 1)

    n_tot = min(max_events, t.GetEntries())
    events = []

    for iev in range(n_tot):
        t.GetEntry(iev)
        mufi = t.Digi_MuFilterHits
        scifi = t.Digi_ScifiHits
        if mufi.GetEntries() < 9 or scifi.GetEntries() < 6:
            continue

        nsf = scifi.GetEntries()
        if not (6 <= nsf <= 30):
            continue

        # Check raw hit counts
        raw_v = 0
        raw_u = 0
        raw_d = 0
        raw_dh = 0
        raw_dv = 0
        raw_us_pl = [0] * 5
        raw_ds_pl_h = [0] * 3
        raw_ds_pl_v = [0] * 4

        for hit in mufi:
            if not hit.isValid(): continue
            did = hit.GetDetectorID()
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000

            if sub == 1 and pl < 2:  # Only planes 0 and 1 (2022 setup, exclude 3rd vertical plane)
                raw_v += 1
            elif sub == 2:
                raw_u += 1
                if 0 <= pl < 5:
                    raw_us_pl[pl] += 1
            elif sub == 3:
                raw_d += 1
                if pl < 3:
                    if bar < 60:
                        raw_dh += 1
                        raw_ds_pl_h[pl] += 1
                    else:
                        raw_dv += 1
                        raw_ds_pl_v[pl] += 1
                elif pl == 3:
                    raw_dv += 1
                    raw_ds_pl_v[3] += 1

        if not (1 <= raw_v <= 4 and 4 <= raw_u <= 8 and 6 <= raw_d <= 16):
            continue

        # Accumulate Geant4 energy loss per bar from MuFilterPoint
        bar_edep = {}
        for pt in t.MuFilterPoint:
            did = pt.GetDetectorID()
            # 1 GeV = 1e6 keV
            bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss() * 1.0e6

        events.append({
            'raw_hits': {
                'veto': raw_v,
                'us': raw_u,
                'ds': raw_d,
                'ds_h': raw_dh,
                'ds_v': raw_dv,
                'us_pl': raw_us_pl,
                'ds_pl_h': raw_ds_pl_h,
                'ds_pl_v': raw_ds_pl_v,
            },
            'bar_edep': bar_edep
        })
        if len(events) >= target_muons:
            break

    print(f"MC: extracted {len(events)} clean through-going muon events.", flush=True)
    return events


def evaluate_masked_hits(mc_events, k, us_mode='gel'):
    """
    Evaluates active hits for each MC event under scale factor k.
    us_mode: 'gel' uses TH_US_GEL_BASE * k, 'air' uses TH_US_AIR_BASE * k
    """
    th_ds_h = TH_DS_HORIZ_BASE * k
    th_ds_v = TH_DS_VERT_BASE * k
    th_us = (TH_US_GEL_BASE if us_mode == 'gel' else TH_US_AIR_BASE) * k
    th_veto_h = TH_VETO_HORIZ_BASE * k
    th_veto_v = TH_VETO_VERT_BASE * k

    masked_events = []
    for ev in mc_events:
        v_count = 0
        u_count = 0
        d_count = 0
        dh_count = 0
        dv_count = 0
        u_pl = [0] * 5
        d_pl_h = [0] * 3
        d_pl_v = [0] * 4

        for did, edep in ev['bar_edep'].items():
            sub = did // 10000
            pl = (did // 1000) % 10
            bar = did % 1000

            if sub == 1 and pl < 2:  # 2022 setup: only planes 0 and 1 (horizontal)
                if edep >= th_veto_h:
                    v_count += 1
            elif sub == 2:
                if 0 <= pl < 5 and edep >= th_us:
                    u_count += 1
                    u_pl[pl] += 1
            elif sub == 3:
                if pl < 3:
                    if bar < 60:
                        if edep >= th_ds_h:
                            d_count += 1
                            dh_count += 1
                            d_pl_h[pl] += 1
                    else:
                        if edep >= th_ds_v:
                            d_count += 1
                            dv_count += 1
                            d_pl_v[pl] += 1
                elif pl == 3:
                    if edep >= th_ds_v:
                        d_count += 1
                        dv_count += 1
                        d_pl_v[3] += 1

        masked_events.append({
            'veto': v_count,
            'us': u_count,
            'ds': d_count,
            'ds_h': dh_count,
            'ds_v': dv_count,
            'us_pl': u_pl,
            'ds_pl_h': d_pl_h,
            'ds_pl_v': d_pl_v,
        })
    return masked_events


def plot_comparison_canvas(data_events, raw_mc_events, masked_mc_events, k, out_dir, artifact_dir):
    """
    Plots a 3x3 multi-panel canvas comparing Data, Uncorrected Sim, and Masked Sim at scale k.
    """
    c = ROOT.TCanvas(f"c_pmu_k_{k:.1f}".replace(".", "p"), f"Data vs PMU Simulation (Scale factor k={k:.1f})", 1800, 1400)
    c.Divide(3, 3)

    plots_cfg = [
        ("Total US Hits", "us", 15, -0.5, 14.5, 1),
        ("Total DS Hits", "ds", 25, -0.5, 24.5, 2),
        ("DS Horizontal Hits", "ds_h", 15, -0.5, 14.5, 3),
        ("DS Vertical Hits", "ds_v", 15, -0.5, 14.5, 4),
        ("Total Veto Hits", "veto", 10, -0.5, 9.5, 5),
        ("US Plane 0 Hits", "us_pl_0", 8, -0.5, 7.5, 6),
        ("US Plane 1 Hits", "us_pl_1", 8, -0.5, 7.5, 7),
        ("DS1 H Hits", "ds_pl_DS1_H", 8, -0.5, 7.5, 8),
        ("DS1 V Hits", "ds_pl_DS1_V", 8, -0.5, 7.5, 9),
    ]

    w1_metrics = {}

    def extract_vals(evs, key):
        if key.startswith("us_pl_"):
            idx = int(key.split("_")[-1])
            return [e['us_pl'][idx] for e in evs]
        elif key == "ds_pl_DS1_H":
            return [e['ds_pl_h'][0] for e in evs]
        elif key == "ds_pl_DS1_V":
            return [e['ds_pl_v'][0] for e in evs]
        else:
            return [e[key] for e in evs]

    # Keep objects alive
    hist_store = []
    lines_store = []

    for title, var_name, nbins, xmin, xmax, pad_idx in plots_cfg:
        c.cd(pad_idx)
        ROOT.gPad.SetGrid(1, 1)
        ROOT.gPad.SetLeftMargin(0.14)
        ROOT.gPad.SetRightMargin(0.06)
        ROOT.gPad.SetBottomMargin(0.12)

        data_vals = extract_vals(data_events, var_name)
        raw_vals = extract_vals([e['raw_hits'] for e in raw_mc_events], var_name)
        mask_vals = extract_vals(masked_mc_events, var_name)

        h_data = ROOT.TH1D(f"h_data_{var_name}_k{k}", f"{title} (k={k:.1f});Number of Hits;Normalized Density", nbins, xmin, xmax)
        h_raw = ROOT.TH1D(f"h_raw_{var_name}_k{k}", f"{title};Number of Hits;Normalized Density", nbins, xmin, xmax)
        h_mask = ROOT.TH1D(f"h_mask_{var_name}_k{k}", f"{title};Number of Hits;Normalized Density", nbins, xmin, xmax)

        for v in data_vals: h_data.Fill(v)
        for v in raw_vals: h_raw.Fill(v)
        for v in mask_vals: h_mask.Fill(v)

        if h_data.Integral() > 0: h_data.Scale(1.0 / h_data.Integral())
        if h_raw.Integral() > 0: h_raw.Scale(1.0 / h_raw.Integral())
        if h_mask.Integral() > 0: h_mask.Scale(1.0 / h_mask.Integral())

        w1 = wasserstein_distance(data_vals, mask_vals)
        w1_raw = wasserstein_distance(data_vals, raw_vals)
        w1_metrics[var_name] = w1

        # Styling
        h_data.SetLineColor(ROOT.kBlack)
        h_data.SetLineWidth(3)
        h_data.SetMarkerStyle(20)
        h_data.SetMarkerSize(0.8)

        h_raw.SetLineColor(ROOT.kBlue + 1)
        h_raw.SetLineStyle(2)
        h_raw.SetLineWidth(2)

        h_mask.SetLineColor(ROOT.kRed + 1)
        h_mask.SetLineWidth(3)

        max_y = max(h_data.GetMaximum(), h_raw.GetMaximum(), h_mask.GetMaximum()) * 1.35
        h_data.SetMaximum(max_y)
        h_data.SetMinimum(0.0)

        h_data.Draw("HIST E")
        h_raw.Draw("HIST SAME")
        h_mask.Draw("HIST SAME")

        # Stats info in legend
        leg = ROOT.TLegend(0.40, 0.65, 0.92, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.038)
        leg.AddEntry(h_data, f"Data (mean={np.mean(data_vals):.2f})", "l")
        leg.AddEntry(h_raw, f"Sim Uncorr (mean={np.mean(raw_vals):.2f}, W_{{1}}={w1_raw:.2f})", "l")
        leg.AddEntry(h_mask, f"Sim k={k:.1f} (mean={np.mean(mask_vals):.2f}, W_{{1}}={w1:.2f})", "l")
        leg.Draw()

        hist_store.extend([h_data, h_raw, h_mask, leg])

    c.Update()
    out_png = os.path.join(out_dir, f"pmu_k_{k:.1f}_comparison.png")
    c.SaveAs(out_png)

    # Copy to artifact directory
    art_png = os.path.join(artifact_dir, f"pmu_k_{k:.1f}_comparison.png")
    os.system(f"cp {out_png} {art_png}")

    return w1_metrics


def main():
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    artifact_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"

    cache_file = "output/pmu_data_matched_events.pkl"
    data_path = "/eos/experiment/sndlhc/convertedData/physics/2022/run_005408/sndsw_raw-0002.root"
    mc_path = "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP.root"

    if os.path.exists(cache_file):
        print(f"Loading cached matched events from {cache_file}...")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
            data_events = cache['data']
            mc_events = cache['mc']
    else:
        data_events = extract_data_events(data_path, max_events=15000, target_muons=1000)
        mc_events = extract_mc_events(mc_path, max_events=80000, target_muons=500)
        with open(cache_file, "wb") as f:
            pickle.dump({'data': data_events, 'mc': mc_events}, f)
        print(f"Cached {len(data_events)} data and {len(mc_events)} MC events to {cache_file}.")

    print(f"\nProceeding with {len(data_events)} data muons and {len(mc_events)} MC muons.")
    print("=" * 80)
    print(f"{'Scale k':<8} | {'W1(US)':<10} | {'W1(DS)':<10} | {'W1(DSH)':<10} | {'W1(DSV)':<10} | {'W1(Veto)':<10} | {'Mean W1':<10}")
    print("=" * 80)

    scan_results = []
    
    for k in K_VALUES:
        masked_mc = evaluate_masked_hits(mc_events, k, us_mode='gel')
        w1s = plot_comparison_canvas(data_events, mc_events, masked_mc, k, out_dir, artifact_dir)
        
        mean_w1 = np.mean([w1s['us'], w1s['ds'], w1s['ds_h'], w1s['ds_v'], w1s['veto']])
        print(f"{k:<8.1f} | {w1s['us']:<10.3f} | {w1s['ds']:<10.3f} | {w1s['ds_h']:<10.3f} | {w1s['ds_v']:<10.3f} | {w1s['veto']:<10.3f} | {mean_w1:<10.3f}", flush=True)
        
        scan_results.append({
            'k': k,
            'th_ds_h': TH_DS_HORIZ_BASE * k,
            'th_ds_v': TH_DS_VERT_BASE * k,
            'th_us_gel': TH_US_GEL_BASE * k,
            'th_us_air': TH_US_AIR_BASE * k,
            'th_veto_h': TH_VETO_HORIZ_BASE * k,
            'th_veto_v': TH_VETO_VERT_BASE * k,
            'w1': w1s,
            'mean_w1': mean_w1
        })

    # Find best k
    best_res = min(scan_results, key=lambda x: x['mean_w1'])
    print("=" * 80)
    print(f"OPTIMAL SCALE FACTOR: k = {best_res['k']:.1f}")
    print(f"Minimum Average Wasserstein Distance: W1 = {best_res['mean_w1']:.3f}")
    print(f"Optimal Thresholds:")
    print(f"  Horizontal DS: {best_res['th_ds_h']:.1f} keV")
    print(f"  Vertical DS:   {best_res['th_ds_v']:.1f} keV")
    print(f"  US Gel:        {best_res['th_us_gel']:.1f} keV")
    print(f"  US Air:        {best_res['th_us_air']:.1f} keV")
    print(f"  Horizontal Veto: {best_res['th_veto_h']:.1f} keV")
    print("=" * 80)

    # Save results to JSON
    with open("output/pmu_data_threshold_scan.json", "w") as f:
        json.dump(scan_results, f, indent=2)

    # Summary plot of W1 vs k
    c_summary = ROOT.TCanvas("c_w1_summary", "Wasserstein Distance vs Threshold Scale Factor k", 1200, 800)
    ROOT.gPad.SetGrid(1, 1)
    ROOT.gPad.SetLeftMargin(0.12)
    ROOT.gPad.SetBottomMargin(0.12)

    k_arr = np.array([r['k'] for r in scan_results], dtype=np.float64)
    w_mean = np.array([r['mean_w1'] for r in scan_results], dtype=np.float64)
    w_us = np.array([r['w1']['us'] for r in scan_results], dtype=np.float64)
    w_ds = np.array([r['w1']['ds'] for r in scan_results], dtype=np.float64)
    w_dsh = np.array([r['w1']['ds_h'] for r in scan_results], dtype=np.float64)
    w_dsv = np.array([r['w1']['ds_v'] for r in scan_results], dtype=np.float64)

    gr_mean = ROOT.TGraph(len(k_arr), k_arr, w_mean)
    gr_mean.SetTitle("Wasserstein-1 Distance (Data vs Sim) vs Scale Factor k;Threshold Scale Factor k;Wasserstein Distance W_{1} [hits]")
    gr_mean.SetLineColor(ROOT.kBlack)
    gr_mean.SetLineWidth(4)
    gr_mean.SetMarkerStyle(20)
    gr_mean.SetMarkerSize(1.2)

    gr_us = ROOT.TGraph(len(k_arr), k_arr, w_us)
    gr_us.SetLineColor(ROOT.kGreen + 2)
    gr_us.SetLineWidth(2)
    gr_us.SetLineStyle(2)

    gr_ds = ROOT.TGraph(len(k_arr), k_arr, w_ds)
    gr_ds.SetLineColor(ROOT.kBlue + 1)
    gr_ds.SetLineWidth(2)
    gr_ds.SetLineStyle(2)

    gr_dsh = ROOT.TGraph(len(k_arr), k_arr, w_dsh)
    gr_dsh.SetLineColor(ROOT.kCyan + 2)
    gr_dsh.SetLineWidth(2)
    gr_dsh.SetLineStyle(3)

    gr_dsv = ROOT.TGraph(len(k_arr), k_arr, w_dsv)
    gr_dsv.SetLineColor(ROOT.kMagenta + 2)
    gr_dsv.SetLineWidth(2)
    gr_dsv.SetLineStyle(3)

    gr_mean.SetMaximum(max(w_mean.max(), w_us.max(), w_ds.max()) * 1.25)
    gr_mean.SetMinimum(0.0)
    gr_mean.Draw("ALP")
    gr_us.Draw("LP SAME")
    gr_ds.Draw("LP SAME")
    gr_dsh.Draw("LP SAME")
    gr_dsv.Draw("LP SAME")

    # Mark the minimum
    best_k = best_res['k']
    best_val = best_res['mean_w1']
    m_line = ROOT.TLine(best_k, 0, best_k, best_val * 1.2)
    m_line.SetLineColor(ROOT.kRed)
    m_line.SetLineStyle(2)
    m_line.SetLineWidth(2)
    m_line.Draw("SAME")

    lat = ROOT.TLatex()
    lat.SetNDC()
    lat.SetTextSize(0.038)
    lat.DrawLatex(0.16, 0.84, f"Optimal Scale: #bf{{k = {best_k:.1f}}} (Min #bar{{W}}_{{1}} = {best_val:.2f} hits)")

    leg_sum = ROOT.TLegend(0.60, 0.65, 0.88, 0.88)
    leg_sum.SetBorderSize(0)
    leg_sum.SetFillStyle(0)
    leg_sum.AddEntry(gr_mean, "Overall Mean W_{1}", "lp")
    leg_sum.AddEntry(gr_us, "US Total", "lp")
    leg_sum.AddEntry(gr_ds, "DS Total", "lp")
    leg_sum.AddEntry(gr_dsh, "DS Horiz", "lp")
    leg_sum.AddEntry(gr_dsv, "DS Vert", "lp")
    leg_sum.Draw()

    summary_png = os.path.join(out_dir, "pmu_threshold_scale_optimization.png")
    c_summary.SaveAs(summary_png)
    os.system(f"cp {summary_png} {os.path.join(artifact_dir, 'pmu_threshold_scale_optimization.png')}")
    print(f"Saved optimization summary to {summary_png}.")

if __name__ == "__main__":
    main()
