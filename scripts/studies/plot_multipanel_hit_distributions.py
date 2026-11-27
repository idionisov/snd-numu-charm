#!/usr/bin/env python3
"""
Multi-panel hit distribution comparison across US and DS planes for:
  - 100 GeV pions (Data: Run 100677, MC: 100GeV_211)
  - 180 GeV pions (Data: Run 100672, MC: 180GeV_211)
  - 300 GeV pions (Data: Run 100639, MC: 300GeV_-211)

Styles per user requirement:
  - Data: Solid black line (k-)
  - Uncorrected simulation: Dotted red line (r:)
  - S-curve corrected simulation: Solid red line (r-)
  - Simple threshold simulation: Solid blue line (b-)
"""

import os
import pickle
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import ROOT

# Set reproducible seed
np.random.seed(42)

# Styling parameters
STYLE = {
    'data': {'color': 'black', 'linestyle': '-', 'linewidth': 2.2, 'label': 'Data'},
    'raw_mc': {'color': 'red', 'linestyle': ':', 'linewidth': 2.0, 'label': 'Uncorrected MC'},
    'scurve_mc': {'color': 'red', 'linestyle': '-', 'linewidth': 2.0, 'label': 'S-curve Corrected'},
    'simple_mc': {'color': 'blue', 'linestyle': '-', 'linewidth': 2.0, 'label': 'Simple Threshold (ΣE > Eth)'}
}

# Physical Turn-on / Threshold Parameters
# US: Eth = 5.5 MeV, sigma = 2.0 MeV (core-fringe continuous response)
# DS-H (double-ended): Eth = 0.90 MeV, sigma = 0.35 MeV
# DS-V (single-ended): Eth = 1.35 MeV, sigma = 0.40 MeV
TH_US, SIG_US = 5.5, 2.0
TH_DSH, SIG_DSH = 0.90, 0.35
TH_DSV, SIG_DSV = 1.35, 0.40

# Simple sharp thresholds
TH_US_SIMPLE = 5.0 # MeV
TH_DSH_SIMPLE = 1.0 # MeV
TH_DSV_SIMPLE = 1.35 # MeV

def load_data(fpath):
    with open(fpath, 'rb') as f:
        events = pickle.load(f)
    us_planes = {p: [] for p in range(5)}
    n_us = []
    ds_h = []
    ds_v = []
    n_ds = []
    
    for ev in events:
        # Handles both dict structures (180 GeV cache and 100/300 GeV cache)
        if 'us_counts' in ev:
            u_p = ev['us_counts']
            dh = ev['n_ds_h']
            dv = ev['n_ds_v']
            tot_u = ev['n_us']
            tot_d = ev['n_ds']
        else:
            u_p = ev['us_planes']
            dh = ev['ds_h']
            dv = ev['ds_v']
            tot_u = ev['n_us']
            tot_d = ev['n_ds']
            
        for p in range(5):
            us_planes[p].append(u_p.get(p, 0))
        n_us.append(tot_u)
        ds_h.append(dh)
        ds_v.append(dv)
        n_ds.append(tot_d)
        
    return {
        'us_planes': {p: np.array(us_planes[p]) for p in range(5)},
        'n_us': np.array(n_us),
        'ds_h': np.array(ds_h),
        'ds_v': np.array(ds_v),
        'n_ds': np.array(n_ds)
    }

def process_mc(fpath):
    with open(fpath, 'rb') as f:
        events = pickle.load(f)
        
    raw_us_planes = {p: [] for p in range(5)}
    raw_n_us = []
    raw_ds_h = []
    raw_ds_v = []
    raw_n_ds = []
    
    scurve_us_planes = {p: [] for p in range(5)}
    scurve_n_us = []
    scurve_ds_h = []
    scurve_ds_v = []
    scurve_n_ds = []
    
    simple_us_planes = {p: [] for p in range(5)}
    simple_n_us = []
    simple_ds_h = []
    simple_ds_v = []
    simple_n_ds = []
    
    for ev in events:
        # 1. Uncorrected (Raw) MC
        u_p_raw = {p: len(ev['us_bars'].get(p, [])) for p in range(5)}
        dh_raw = len(ev['ds_h_bars'])
        dv_raw = len(ev['ds_v_bars'])
        
        for p in range(5):
            raw_us_planes[p].append(u_p_raw[p])
        raw_n_us.append(sum(u_p_raw.values()))
        raw_ds_h.append(dh_raw)
        raw_ds_v.append(dv_raw)
        raw_n_ds.append(dh_raw + dv_raw)
        
        # 2. S-curve Corrected MC
        u_p_sc = {}
        for p in range(5):
            h_cnt = 0
            for edep in ev['us_bars'].get(p, []):
                p_eff = 0.5 * (1.0 + erf((edep - TH_US) / (np.sqrt(2) * SIG_US)))
                if np.random.rand() < p_eff:
                    h_cnt += 1
            u_p_sc[p] = h_cnt
            scurve_us_planes[p].append(h_cnt)
            
        dh_sc = sum(1 for edep in ev['ds_h_bars'] if np.random.rand() < 0.5 * (1.0 + erf((edep - TH_DSH) / (np.sqrt(2) * SIG_DSH))))
        dv_sc = sum(1 for edep in ev['ds_v_bars'] if np.random.rand() < 0.5 * (1.0 + erf((edep - TH_DSV) / (np.sqrt(2) * SIG_DSV))))
        scurve_n_us.append(sum(u_p_sc.values()))
        scurve_ds_h.append(dh_sc)
        scurve_ds_v.append(dv_sc)
        scurve_n_ds.append(dh_sc + dv_sc)
        
        # 3. Simple Threshold MC (ΣE > Eth on the bar)
        u_p_sim = {}
        for p in range(5):
            h_sim = sum(1 for edep in ev['us_bars'].get(p, []) if edep >= TH_US_SIMPLE)
            u_p_sim[p] = h_sim
            simple_us_planes[p].append(h_sim)
            
        dh_sim = sum(1 for edep in ev['ds_h_bars'] if edep >= TH_DSH_SIMPLE)
        dv_sim = sum(1 for edep in ev['ds_v_bars'] if edep >= TH_DSV_SIMPLE)
        simple_n_us.append(sum(u_p_sim.values()))
        simple_ds_h.append(dh_sim)
        simple_ds_v.append(dv_sim)
        simple_n_ds.append(dh_sim + dv_sim)
        
    return {
        'raw': {
            'us_planes': {p: np.array(raw_us_planes[p]) for p in range(5)},
            'n_us': np.array(raw_n_us),
            'ds_h': np.array(raw_ds_h),
            'ds_v': np.array(raw_ds_v),
            'n_ds': np.array(raw_n_ds)
        },
        'scurve': {
            'us_planes': {p: np.array(scurve_us_planes[p]) for p in range(5)},
            'n_us': np.array(scurve_n_us),
            'ds_h': np.array(scurve_ds_h),
            'ds_v': np.array(scurve_ds_v),
            'n_ds': np.array(scurve_n_ds)
        },
        'simple': {
            'us_planes': {p: np.array(simple_us_planes[p]) for p in range(5)},
            'n_us': np.array(simple_n_us),
            'ds_h': np.array(simple_ds_h),
            'ds_v': np.array(simple_ds_v),
            'n_ds': np.array(simple_n_ds)
        }
    }

def plot_step_hist(ax, data, bins, color, linestyle, linewidth, label):
    counts, edges = np.histogram(data, bins=bins, density=True)
    # Step plot matching standard ROOT/physics style
    x = np.repeat(edges, 2)
    y = np.pad(np.repeat(counts, 2), 1, mode='constant', constant_values=0)
    ax.plot(edges[:-1] + 0.5 * np.diff(edges), counts, drawstyle='steps-mid',
            color=color, linestyle=linestyle, linewidth=linewidth, label=label)

def plot_multipanel_energy(energy_str, data_dict, mc_dict, out_png):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f"MuFilter Hit Multiplicity per Plane: {energy_str} Pions (Data vs MC)\n"
                 r"Solid Black: Data | Dotted Red: Uncorrected MC | Solid Blue: Simple Threshold | Solid Red: S-Curve Corrected",
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 9 Panels definition:
    # Row 0: US Plane 0, US Plane 1, US Plane 2
    # Row 1: US Plane 3, US Plane 4, Total US
    # Row 2: DS-H, DS-V, Total DS
    panel_configs = [
        (0, 0, 'US Plane 0 Multiplicity', data_dict['us_planes'][0],
         mc_dict['raw']['us_planes'][0], mc_dict['simple']['us_planes'][0], mc_dict['scurve']['us_planes'][0],
         np.arange(-0.5, 12.5, 1.0), 'Hits in US Plane 0'),
        (0, 1, 'US Plane 1 Multiplicity', data_dict['us_planes'][1],
         mc_dict['raw']['us_planes'][1], mc_dict['simple']['us_planes'][1], mc_dict['scurve']['us_planes'][1],
         np.arange(-0.5, 12.5, 1.0), 'Hits in US Plane 1'),
        (0, 2, 'US Plane 2 (Shower Max) Multiplicity', data_dict['us_planes'][2],
         mc_dict['raw']['us_planes'][2], mc_dict['simple']['us_planes'][2], mc_dict['scurve']['us_planes'][2],
         np.arange(-0.5, 12.5, 1.0), 'Hits in US Plane 2'),
        (1, 0, 'US Plane 3 Multiplicity', data_dict['us_planes'][3],
         mc_dict['raw']['us_planes'][3], mc_dict['simple']['us_planes'][3], mc_dict['scurve']['us_planes'][3],
         np.arange(-0.5, 12.5, 1.0), 'Hits in US Plane 3'),
        (1, 1, 'US Plane 4 Multiplicity', data_dict['us_planes'][4],
         mc_dict['raw']['us_planes'][4], mc_dict['simple']['us_planes'][4], mc_dict['scurve']['us_planes'][4],
         np.arange(-0.5, 12.5, 1.0), 'Hits in US Plane 4'),
        (1, 2, 'Total US Multiplicity', data_dict['n_us'],
         mc_dict['raw']['n_us'], mc_dict['simple']['n_us'], mc_dict['scurve']['n_us'],
         np.arange(7.5, 45.5, 2.0), 'Total Hits in US (Planes 0-4)'),
        (2, 0, 'Downstream Horizontal (DS-H)', data_dict['ds_h'],
         mc_dict['raw']['ds_h'], mc_dict['simple']['ds_h'], mc_dict['scurve']['ds_h'],
         np.arange(-0.5, 16.5, 1.0), 'Hits in DS-H'),
        (2, 1, 'Downstream Vertical (DS-V)', data_dict['ds_v'],
         mc_dict['raw']['ds_v'], mc_dict['simple']['ds_v'], mc_dict['scurve']['ds_v'],
         np.arange(-0.5, 16.5, 1.0), 'Hits in DS-V'),
        (2, 2, 'Total DS Multiplicity', data_dict['n_ds'],
         mc_dict['raw']['n_ds'], mc_dict['simple']['n_ds'], mc_dict['scurve']['n_ds'],
         np.arange(-0.5, 28.5, 1.0), 'Total Hits in DS (DS-H + DS-V)')
    ]
    
    for r, c, title, d_arr, raw_arr, sim_arr, sc_arr, bins, xlabel in panel_configs:
        ax = axes[r, c]
        
        # Calculate means
        mu_d = np.mean(d_arr)
        mu_raw = np.mean(raw_arr)
        mu_sim = np.mean(sim_arr)
        mu_sc = np.mean(sc_arr)
        
        # Plot curves
        plot_step_hist(ax, d_arr, bins, color='black', linestyle='-', linewidth=2.4,
                       label=f"Data (μ={mu_d:.2f})")
        plot_step_hist(ax, raw_arr, bins, color='red', linestyle=':', linewidth=2.0,
                       label=f"Uncorrected (μ={mu_raw:.2f})")
        plot_step_hist(ax, sim_arr, bins, color='blue', linestyle='-', linewidth=2.0,
                       label=f"Simple Thresh (μ={mu_sim:.2f})")
        plot_step_hist(ax, sc_arr, bins, color='red', linestyle='-', linewidth=2.2,
                       label=f"S-Curve (μ={mu_sc:.2f})")
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8.5, loc='upper right', frameon=True)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved figure: {out_png}")

def save_root_canvases(all_datasets, root_out_path):
    f_root = ROOT.TFile.Open(root_out_path, "RECREATE")
    
    for energy_label, (d_dict, m_dict) in all_datasets.items():
        c = ROOT.TCanvas(f"c_mufilter_{energy_label}", f"MuFilter Hit Multiplicities - {energy_label}", 1800, 1400)
        c.Divide(3, 3)
        
        panels = [
            ("US_Pl0", "US Plane 0", d_dict['us_planes'][0], m_dict['raw']['us_planes'][0], m_dict['simple']['us_planes'][0], m_dict['scurve']['us_planes'][0], 13, -0.5, 12.5),
            ("US_Pl1", "US Plane 1", d_dict['us_planes'][1], m_dict['raw']['us_planes'][1], m_dict['simple']['us_planes'][1], m_dict['scurve']['us_planes'][1], 13, -0.5, 12.5),
            ("US_Pl2", "US Plane 2", d_dict['us_planes'][2], m_dict['raw']['us_planes'][2], m_dict['simple']['us_planes'][2], m_dict['scurve']['us_planes'][2], 13, -0.5, 12.5),
            ("US_Pl3", "US Plane 3", d_dict['us_planes'][3], m_dict['raw']['us_planes'][3], m_dict['simple']['us_planes'][3], m_dict['scurve']['us_planes'][3], 13, -0.5, 12.5),
            ("US_Pl4", "US Plane 4", d_dict['us_planes'][4], m_dict['raw']['us_planes'][4], m_dict['simple']['us_planes'][4], m_dict['scurve']['us_planes'][4], 13, -0.5, 12.5),
            ("US_Total", "Total US", d_dict['n_us'], m_dict['raw']['n_us'], m_dict['simple']['n_us'], m_dict['scurve']['n_us'], 25, 7.5, 45.5),
            ("DS_H", "DS Horizontal", d_dict['ds_h'], m_dict['raw']['ds_h'], m_dict['simple']['ds_h'], m_dict['scurve']['ds_h'], 17, -0.5, 16.5),
            ("DS_V", "DS Vertical", d_dict['ds_v'], m_dict['raw']['ds_v'], m_dict['simple']['ds_v'], m_dict['scurve']['ds_v'], 17, -0.5, 16.5),
            ("DS_Total", "Total DS", d_dict['n_ds'], m_dict['raw']['n_ds'], m_dict['simple']['n_ds'], m_dict['scurve']['n_ds'], 29, -0.5, 28.5),
        ]
        
        # Keep references alive
        hist_storage = []
        
        for idx, (p_id, p_name, d_arr, raw_arr, sim_arr, sc_arr, nbins, xmin, xmax) in enumerate(panels):
            c.cd(idx + 1)
            ROOT.gPad.SetGrid(1, 1)
            
            h_data = ROOT.TH1D(f"h_data_{energy_label}_{p_id}", f"{p_name} ({energy_label});Hits;Density", nbins, xmin, xmax)
            h_raw = ROOT.TH1D(f"h_raw_{energy_label}_{p_id}", f"{p_name} Raw;Hits;Density", nbins, xmin, xmax)
            h_sim = ROOT.TH1D(f"h_sim_{energy_label}_{p_id}", f"{p_name} Simple;Hits;Density", nbins, xmin, xmax)
            h_sc = ROOT.TH1D(f"h_sc_{energy_label}_{p_id}", f"{p_name} SCurve;Hits;Density", nbins, xmin, xmax)
            
            for v in d_arr: h_data.Fill(v)
            for v in raw_arr: h_raw.Fill(v)
            for v in sim_arr: h_sim.Fill(v)
            for v in sc_arr: h_sc.Fill(v)
            
            # Normalize to density
            for h in [h_data, h_raw, h_sim, h_sc]:
                if h.Integral() > 0:
                    h.Scale(1.0 / (h.Integral() * h.GetBinWidth(1)))
                    
            # Set User requested styles:
            # Data: solid black
            h_data.SetLineColor(ROOT.kBlack)
            h_data.SetLineWidth(2)
            
            # Raw: dotted red
            h_raw.SetLineColor(ROOT.kRed)
            h_raw.SetLineWidth(2)
            h_raw.SetLineStyle(3) # dotted
            
            # Simple threshold: solid blue
            h_sim.SetLineColor(ROOT.kBlue)
            h_sim.SetLineWidth(2)
            h_sim.SetLineStyle(1)
            
            # S-curve: solid red
            h_sc.SetLineColor(ROOT.kRed)
            h_sc.SetLineWidth(2)
            h_sc.SetLineStyle(1)
            
            max_y = max(h_data.GetMaximum(), h_raw.GetMaximum(), h_sim.GetMaximum(), h_sc.GetMaximum()) * 1.3
            h_data.SetMaximum(max_y)
            
            h_data.Draw("HIST")
            h_raw.Draw("HIST SAME")
            h_sim.Draw("HIST SAME")
            h_sc.Draw("HIST SAME")
            
            leg = ROOT.TLegend(0.48, 0.65, 0.88, 0.88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.AddEntry(h_data, f"Data (#mu={h_data.GetMean():.2f})", "l")
            leg.AddEntry(h_raw, f"Uncorrected (#mu={h_raw.GetMean():.2f})", "l")
            leg.AddEntry(h_sim, f"Simple Thresh (#mu={h_sim.GetMean():.2f})", "l")
            leg.AddEntry(h_sc, f"S-Curve (#mu={h_sc.GetMean():.2f})", "l")
            leg.Draw()
            
            hist_storage.extend([h_data, h_raw, h_sim, h_sc, leg])
            
        c.Write()
        
    f_root.Close()
    print(f"Saved interactive ROOT canvases to: {root_out_path}")

def main():
    os.makedirs("plots", exist_ok=True)
    
    datasets = {
        '100GeV': {
            'data': '/tmp/data_pion_100gev_clean.pkl',
            'mc': '/tmp/mc_pion_100gev_clean.pkl',
            'label': '100 GeV'
        },
        '180GeV': {
            'data': '/tmp/data_clean_sndsw_raw-0000.root_30000_8_50.pkl',
            'mc': '/tmp/mc_180gev_4000_8.pkl',
            'label': '180 GeV'
        },
        '300GeV': {
            'data': '/tmp/data_pion_300gev_clean.pkl',
            'mc': '/tmp/mc_pion_300gev_clean.pkl',
            'label': '300 GeV'
        }
    }
    
    processed_all = {}
    
    for en_key, dinfo in datasets.items():
        print(f"\n--- Processing {dinfo['label']} ---")
        d_dict = load_data(dinfo['data'])
        m_dict = process_mc(dinfo['mc'])
        processed_all[en_key] = (d_dict, m_dict)
        
        out_png = f"plots/mufilter_planes_{en_key.lower()}.png"
        plot_multipanel_energy(dinfo['label'], d_dict, m_dict, out_png)
        
    # ROOT File with TCanvases
    root_out = "plots/mufilter_planes_comparison.root"
    save_root_canvases(processed_all, root_out)

if __name__ == '__main__':
    main()
