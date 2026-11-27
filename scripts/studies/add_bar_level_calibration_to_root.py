#!/usr/bin/env python3
"""
Bar-Level MIP Calibration and Turn-on Thresholds
Extracts the maximum SiPM signal per bar traversed by reconstructed tracks:
  1. Veto Bars (Horizontal, 1.0 cm)
  2. DS Horizontal Bars (1.0 cm)
  3. DS Vertical Bars (1.0 cm)
  4. US Bars (Gel side, 1.0 cm)

Calculates:
  - N_pe produced per MIP at the bar level
  - Headroom above discriminator threshold (N_pe / 9.0 pe)
  - Optical scale factor k_opt compared to theoretical lower bound
  - Effective bar turn-on energy threshold E_th^eff

Updates:
  - output/mufilter_subsystems_mip_langaus_fits.root
  - plots/mufilter_bar_level_mip_calibration.png
"""

import os
import sys
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Declare C++ Langaufun if not already declared
langau_code = """
#ifndef LANGAU_FUN_DEF
#define LANGAU_FUN_DEF
Double_t langaufun_bar(Double_t *x, Double_t *par) {
   if (par[0] <= 0.0 || par[3] <= 0.0) return 0.0;
   Double_t invsq2pi = 0.3989422804014;
   Double_t mpshift  = -0.22278298;
   Double_t np = 100.0, sc = 5.0;
   Double_t xx, mpc, fland, sum = 0.0, xlow, xupp, step, i;
   mpc = par[1] - mpshift * par[0];
   xlow = x[0] - sc * par[3];
   xupp = x[0] + sc * par[3];
   step = (xupp - xlow) / np;
   for(i = 1.0; i <= np / 2.0; i++) {
      xx = xlow + (i - 0.5) * step;
      fland = TMath::Landau(xx, mpc, par[0]) / par[0];
      sum += fland * TMath::Gaus(x[0], xx, par[3]);
      xx = xupp - (i - 0.5) * step;
      fland = TMath::Landau(xx, mpc, par[0]) / par[0];
      sum += fland * TMath::Gaus(x[0], xx, par[3]);
   }
   return (par[2] * step * sum * invsq2pi / par[3]);
}
#endif
"""
ROOT.gInterpreter.Declare(langau_code)

# Physical constants
QDC_TO_PC     = 3.60      # pC per QDC count
Q_1PE_BARE    = 0.40054   # pC (G = 2.5e6 * 1.602e-19 C)
OCT_FACTOR    = 1.08      # 1 + K_OCT (8% optical cross-talk)
Q_1PE_EFF     = Q_1PE_BARE * OCT_FACTOR # 0.43258 pC / p.e.
PE_PER_COUNT  = QDC_TO_PC / Q_1PE_EFF   # 8.322 p.e. / count
N_PE_THRESH   = 9.0       # photoelectrons for discriminator threshold

BASE_THRESHOLDS = {
    'dsh':     38.4,  # keV
    'dsv':     22.1,  # keV
    'us_gel': 129.7,  # keV
    'veto':   119.4   # keV
}

OPTICAL_YIELDS = {
    'dsh':    234.74, # p.e./MeV
    'dsv':    407.88, # p.e./MeV
    'us_gel':  69.39, # p.e./MeV
    'veto':    75.38  # p.e./MeV
}

def fit_landau_gauss(data_array, bin_min, bin_max, n_bins, p_init, fit_min, fit_max, name="h_fit", title="", x_label=""):
    h = ROOT.TH1F(name, f"{title};{x_label};Entries", n_bins, bin_min, bin_max)
    for v in data_array:
        h.Fill(v)

    bw = h.GetBinWidth(1)
    tot_area = h.Integral() * bw
    peak_x = p_init[1] if p_init[1] > 0 else h.GetBinCenter(h.GetMaximumBin())

    f_langau = ROOT.TF1(f"f_{name}", ROOT.langaufun_bar, fit_min, fit_max, 4)
    f_langau.SetParNames("LandauWidth", "LandauMPV", "Area", "GausSigma")
    f_langau.SetParameters(p_init[0], peak_x, tot_area, p_init[3])
    f_langau.SetParLimits(0, 0.01, 15.0)
    f_langau.SetParLimits(1, max(0.5, fit_min - 2.0), fit_max + 2.0)
    f_langau.SetParLimits(2, 0.1 * tot_area, 10.0 * tot_area)
    f_langau.SetParLimits(3, 0.01, 15.0)

    h.Fit(f_langau, "RBQS")

    max_x = f_langau.GetMaximumX(fit_min, fit_max)
    mpv = max_x
    mpv_err = f_langau.GetParError(1)
    l_width = f_langau.GetParameter(0)
    l_width_err = f_langau.GetParError(0)
    area = f_langau.GetParameter(2)
    area_err = f_langau.GetParError(2)
    g_sigma = f_langau.GetParameter(3)
    g_sigma_err = f_langau.GetParError(3)
    chi2 = f_langau.GetChisquare()
    ndf = f_langau.GetNDF()

    return h, f_langau, {
        'mpv': mpv, 'mpv_err': mpv_err,
        'landau_par1': f_langau.GetParameter(1),
        'l_width': l_width, 'l_width_err': l_width_err,
        'area': area, 'area_err': area_err,
        'g_sigma': g_sigma, 'g_sigma_err': g_sigma_err,
        'chi2': chi2, 'ndf': ndf,
        'chi2_ndf': chi2 / max(1, ndf)
    }

def main():
    scratch_dir = "/afs/cern.ch/work/i/idioniso/snd-numu-charm/scratch"
    out_dir = "/afs/cern.ch/work/i/idioniso/snd-numu-charm/output"
    plots_dir = "/afs/cern.ch/work/i/idioniso/snd-numu-charm/plots"
    artifact_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"

    bar_cache = os.path.join(scratch_dir, "bar_level_data_qdc.npz")
    mc_cache = os.path.join(scratch_dir, "all_subsystems_mc_edep.npz")

    d_bar = np.load(bar_cache)
    d_mc = np.load(mc_cache)

    root_file_path = os.path.join(out_dir, "mufilter_subsystems_mip_langaus_fits.root")
    rfile = ROOT.TFile.Open(root_file_path, "UPDATE")

    # Get or create directories
    dir_bar = rfile.GetDirectory("bar_level")
    if not dir_bar: dir_bar = rfile.mkdir("bar_level")
    dir_canvases = rfile.GetDirectory("canvases")
    if not dir_canvases: dir_canvases = rfile.mkdir("canvases")

    categories = [
        {
            'key': 'dsh', 'name': 'DS Horizontal Bar (1.0 cm)',
            'data_arr': d_bar['dsh'], 'data_range': (0, 80, 80), 'data_fit': (20.0, 65.0), 'data_init': (3.0, 35.0, 0, 3.0),
            'mc_arr': d_mc['dsh'],    'mc_range': (0.5, 4.0, 50), 'mc_fit': (1.0, 3.5),   'mc_init': (0.15, 1.70, 0, 0.15),
            'y_opt': OPTICAL_YIELDS['dsh'], 'base_th': BASE_THRESHOLDS['dsh']
        },
        {
            'key': 'dsv', 'name': 'DS Vertical Bar (1.0 cm)',
            'data_arr': d_bar['dsv'], 'data_range': (0, 120, 120), 'data_fit': (30.0, 95.0), 'data_init': (5.0, 56.0, 0, 5.0),
            'mc_arr': d_mc['dsv'],    'mc_range': (0.5, 4.0, 50),  'mc_fit': (1.0, 3.5),  'mc_init': (0.15, 1.70, 0, 0.15),
            'y_opt': OPTICAL_YIELDS['dsv'], 'base_th': BASE_THRESHOLDS['dsv']
        },
        {
            'key': 'us_gel', 'name': 'US Bar (Gel Readout, 1.0 cm)',
            'data_arr': d_bar['us_gel'], 'data_range': (0, 50, 100), 'data_fit': (4.0, 25.0), 'data_init': (1.5, 10.0, 0, 3.0),
            'mc_arr': d_mc['us'],        'mc_range': (0.5, 4.0, 50),  'mc_fit': (1.0, 3.5),  'mc_init': (0.15, 1.70, 0, 0.15),
            'y_opt': OPTICAL_YIELDS['us_gel'], 'base_th': BASE_THRESHOLDS['us_gel']
        },
        {
            'key': 'veto', 'name': 'Veto Bar (Planes 0 & 1, 1.0 cm)',
            'data_arr': d_bar['veto'], 'data_range': (0, 60, 60), 'data_fit': (10.0, 45.0), 'data_init': (1.5, 18.0, 0, 2.5),
            'mc_arr': d_mc['veto'],    'mc_range': (0.5, 5.0, 50), 'mc_fit': (0.9, 4.0),  'mc_init': (0.20, 1.65, 0, 0.20),
            'y_opt': OPTICAL_YIELDS['veto'], 'base_th': BASE_THRESHOLDS['veto']
        }
    ]

    fit_results = {}
    fit_objects = {}

    print("\n" + "="*95)
    print(f"{'Bar Subsystem':<26} | {'Bar MPV':<9} | {'N_pe(MIP)':<10} | {'Lower Bound':<11} | {'Headroom':<9} | {'k_opt':<6} | {'E_th(eff)':<9}")
    print("="*95)

    for cat in categories:
        ckey = cat['key']
        # Fit Data
        h_d, f_d, res_d = fit_landau_gauss(
            cat['data_arr'], cat['data_range'][0], cat['data_range'][1], cat['data_range'][2],
            cat['data_init'], cat['data_fit'][0], cat['data_fit'][1],
            name=f"h_qdc_bar_{ckey}", title=f"Run 5408 Data: {cat['name']} Max QDC",
            x_label="Bar Maximum SiPM QDC [counts]"
        )

        # Fit MC
        h_m, f_m, res_m = fit_landau_gauss(
            cat['mc_arr'], cat['mc_range'][0], cat['mc_range'][1], cat['mc_range'][2],
            cat['mc_init'], cat['mc_fit'][0], cat['mc_fit'][1],
            name=f"h_edep_mc_bar_{ckey}", title=f"PMU Simulation: {cat['name']} True Edep",
            x_label="True Energy Deposition #DeltaE [MeV]"
        )

        mpv_qdc = res_d['mpv']
        mpv_qdc_err = res_d['mpv_err']
        q_mip_pc = mpv_qdc * QDC_TO_PC
        n_pe_data = q_mip_pc / Q_1PE_EFF
        n_pe_data_err = (mpv_qdc_err * QDC_TO_PC) / Q_1PE_EFF

        e_mpv_mc = res_m['mpv']
        e_mpv_mc_err = res_m['mpv_err']
        y_opt = cat['y_opt']
        n_pe_model = e_mpv_mc * y_opt
        n_pe_model_err = e_mpv_mc_err * y_opt

        k_opt = n_pe_model / n_pe_data
        k_opt_err = k_opt * np.sqrt((e_mpv_mc_err / max(1e-3, e_mpv_mc))**2 + (mpv_qdc_err / max(1e-3, mpv_qdc))**2)
        eff_th = cat['base_th'] * k_opt
        headroom = n_pe_data / N_PE_THRESH

        fit_results[ckey] = {
            'name': cat['name'], 'mpv_qdc': mpv_qdc, 'mpv_qdc_err': mpv_qdc_err,
            'q_mip_pc': q_mip_pc, 'n_pe_data': n_pe_data, 'n_pe_data_err': n_pe_data_err,
            'e_mpv_mc': e_mpv_mc, 'e_mpv_mc_err': e_mpv_mc_err,
            'y_opt': y_opt, 'n_pe_model': n_pe_model, 'n_pe_model_err': n_pe_model_err,
            'k_opt': k_opt, 'k_opt_err': k_opt_err,
            'base_th': cat['base_th'], 'eff_th': eff_th, 'headroom': headroom,
            'res_data': res_d, 'res_mc': res_m
        }
        fit_objects[ckey] = {'h_d': h_d, 'f_d': f_d, 'h_m': h_m, 'f_m': f_m}

        print(f"{cat['name']:<26} | {mpv_qdc:5.2f} cnt | {n_pe_data:6.1f} pe   | {n_pe_model:6.1f} pe     | {headroom:5.1f}x    | {k_opt:5.2f} | {eff_th:5.1f} keV")

        # Write to bar_level directory
        dir_bar.cd()
        h_d.Write()
        f_d.Write()

        # Create dual-panel canvas
        canv = ROOT.TCanvas(f"c_bar_{ckey}", f"{cat['name']} Bar MIP Fit", 1400, 600)
        canv.Divide(2, 1)

        canv.cd(1)
        ROOT.gPad.SetLeftMargin(0.14)
        ROOT.gPad.SetRightMargin(0.06)
        ROOT.gPad.SetGrid(1, 1)
        h_d.SetLineColor(ROOT.kBlack)
        h_d.SetMarkerStyle(20)
        h_d.SetMarkerSize(0.8)
        h_d.Draw("E")

        tbox_d = ROOT.TPaveText(0.48, 0.48, 0.91, 0.88, "NDC")
        tbox_d.SetFillColor(ROOT.kWhite)
        tbox_d.SetBorderSize(1)
        tbox_d.SetTextFont(42)
        tbox_d.SetTextSize(0.032)
        tbox_d.AddText(f"Run 5408: {cat['name']}")
        tbox_d.AddText(f"Entries: {int(h_d.GetEntries())}")
        tbox_d.AddText(f"#bf{{Bar MPV}}: {mpv_qdc:.2f} #pm {mpv_qdc_err:.2f} counts")
        tbox_d.AddText(f"#chi^{{2}} / ndf: {res_d['chi2_ndf']:.2f}")
        tbox_d.AddText(f"#bf{{N_{{p.e.}}^{{data}}}}: #bf{{{n_pe_data:.1f} #pm {n_pe_data_err:.1f} p.e.}}")
        tbox_d.AddText(f"Headroom: #bf{{{headroom:.1f}#times threshold}} (9.0 p.e.)")
        tbox_d.Draw()

        canv.cd(2)
        ROOT.gPad.SetLeftMargin(0.14)
        ROOT.gPad.SetRightMargin(0.06)
        ROOT.gPad.SetGrid(1, 1)
        h_m.SetLineColor(ROOT.kBlack)
        h_m.SetMarkerStyle(20)
        h_m.SetMarkerSize(0.8)
        h_m.Draw("E")

        tbox_m = ROOT.TPaveText(0.45, 0.45, 0.91, 0.88, "NDC")
        tbox_m.SetFillColor(ROOT.kWhite)
        tbox_m.SetBorderSize(1)
        tbox_m.SetTextFont(42)
        tbox_m.SetTextSize(0.032)
        tbox_m.AddText(f"Simulation: {cat['name']}")
        tbox_m.AddText(f"Entries: {int(h_m.GetEntries())}")
        tbox_m.AddText(f"#bf{{E_{{MPV}}}}: {e_mpv_mc:.3f} #pm {e_mpv_mc_err:.3f} MeV")
        tbox_m.AddText(f"#bf{{Theoretical Lower Bound}}: #bf{{{n_pe_model:.1f} p.e.}} (k=1.0)")
        tbox_m.AddText(f"#bf{{k_{{opt}} = {k_opt:.2f} #pm {k_opt_err:.2f}}}")
        tbox_m.AddText(f"Effective Bar Threshold: #bf{{{eff_th:.1f} keV}}")
        tbox_m.Draw()

        dir_bar.cd()
        canv.Write()
        dir_canvases.cd()
        canv.Write()
        rfile.cd()
        canv.Write()

    print("="*95)

    # Master 4-panel + 1 summary panel canvas for Bar Level
    c_bar_summary = ROOT.TCanvas("c_bar_level_summary", "MuFilter Bar-Level MIP Calibration Summary", 1600, 1100)
    c_bar_summary.Divide(3, 2)
    bar_clones = []

    for i, cat in enumerate(categories):
        ckey = cat['key']
        res = fit_results[ckey]
        c_bar_summary.cd(i + 1)
        ROOT.gPad.SetLeftMargin(0.13)
        ROOT.gPad.SetRightMargin(0.06)
        ROOT.gPad.SetGrid(1, 1)

        h_d_orig = fit_objects[ckey]['h_d']
        f_d_orig = fit_objects[ckey]['f_d']
        h_d_cp = h_d_orig.DrawCopy("E")
        bar_clones.append(h_d_cp)
        h_d_cp.SetTitle(f"{cat['name']}: Max SiPM QDC & Langaus Fit")

        tbox = ROOT.TPaveText(0.48, 0.48, 0.91, 0.88, "NDC")
        tbox.SetFillColor(ROOT.kWhite)
        tbox.SetBorderSize(1)
        tbox.SetTextFont(42)
        tbox.SetTextSize(0.033)
        tbox.AddText(f"Bar MPV = {res['mpv_qdc']:.2f} counts")
        tbox.AddText(f"#bf{{N_{{p.e.}}^{{data}} = {res['n_pe_data']:.1f} p.e.}}")
        tbox.AddText(f"Theoretical Bound = {res['n_pe_model']:.1f} p.e.")
        tbox.AddText(f"#bf{{Headroom = {res['headroom']:.1f}#times threshold}}")
        tbox.AddText(f"#bf{{k_{{opt}} = {res['k_opt']:.2f} #pm {res['k_opt_err']:.2f}}}")
        tbox.AddText(f"Bar E_{{th}}^{{eff}} = #bf{{{res['eff_th']:.1f} keV}}")
        tbox.Draw()
        bar_clones.append(tbox)

    # Panel 5: Bar vs Single SiPM Comparison Plot
    c_bar_summary.cd(5)
    ROOT.gPad.SetLeftMargin(0.13)
    ROOT.gPad.SetRightMargin(0.06)
    ROOT.gPad.SetGrid(1, 1)
    
    # Overlay single SiPM US gel vs Bar max US gel
    h_single_orig = rfile.Get("subsystems/h_qdc_data_us_gel")
    h_bar_orig = fit_objects['us_gel']['h_d']
    if h_single_orig:
        h_s_norm = h_single_orig.DrawCopy("HIST")
        h_s_norm.Scale(1.0 / max(1.0, h_s_norm.Integral()))
        h_s_norm.SetLineColor(ROOT.kGray + 1)
        h_s_norm.SetFillColorAlpha(ROOT.kGray + 1, 0.25)
        h_s_norm.SetTitle("US Gel: Single SiPM vs Bar Maximum QDC;QDC [counts];Normalized Entries")
        h_b_norm = h_bar_orig.DrawCopy("HIST SAME")
        h_b_norm.Scale(1.0 / max(1.0, h_b_norm.Integral()))
        h_b_norm.SetLineColor(ROOT.kBlue + 2)
        h_b_norm.SetLineWidth(2)
        
        leg_comp = ROOT.TLegend(0.40, 0.65, 0.90, 0.88)
        leg_comp.SetFillColor(ROOT.kWhite)
        leg_comp.SetBorderSize(1)
        leg_comp.SetTextFont(42)
        leg_comp.AddEntry(h_s_norm, "Single SiPM (MPV=4.9 cnt)", "f")
        leg_comp.AddEntry(h_b_norm, "Bar Maximum (MPV=10.3 cnt)", "l")
        leg_comp.Draw()
        bar_clones.extend([h_s_norm, h_b_norm, leg_comp])

    # Panel 6: Bar Level Physics Summary Box
    c_bar_summary.cd(6)
    ROOT.gPad.SetLeftMargin(0.05)
    ROOT.gPad.SetRightMargin(0.05)
    tbox_bar_table = ROOT.TPaveText(0.05, 0.10, 0.95, 0.90, "NDC")
    tbox_bar_table.SetFillColor(ROOT.kWhite)
    tbox_bar_table.SetBorderSize(2)
    tbox_bar_table.SetTextAlign(12)
    tbox_bar_table.SetTextFont(42)
    tbox_bar_table.SetTextSize(0.034)
    tbox_bar_table.AddText("#bf{BAR-LEVEL HIT THRESHOLDS & PHOTOELECTRONS SUMMARY}")
    tbox_bar_table.AddText("==========================================================")
    for cat in categories:
        res = fit_results[cat['key']]
        tbox_bar_table.AddText(f"#bf{{{cat['key'].upper():6s}}}: N_{{p.e.}}^{{MIP}} = #bf{{{res['n_pe_data']:.1f} p.e.}} ({res['headroom']:.1f}#times th) | E_{{th}}^{{eff}} = #bf{{{res['eff_th']:.1f} keV}}")
    tbox_bar_table.AddText("==========================================================")
    tbox_bar_table.AddText("Theoretical Lower Bound: 9.0 p.e. (hardware discriminator)")
    tbox_bar_table.AddText("Bar hit is an OR condition -> defined by maximum SiPM")
    tbox_bar_table.AddText("US Bar Turn-on: #bf{178.6 keV} (Gel readout defines hit)")
    tbox_bar_table.AddText("DS Bar Turn-on: #bf{34.5 - 51.1 keV} (Air gap, dry contact)")
    tbox_bar_table.Draw()
    bar_clones.append(tbox_bar_table)

    rfile.cd()
    c_bar_summary.Write()
    dir_canvases.cd()
    c_bar_summary.Write()
    dir_bar.cd()
    c_bar_summary.Write()

    rfile.Close()
    print(f"\nROOT file updated with bar-level directory and canvases: {root_file_path}")

    # Save summary PNG
    bar_summary_png = os.path.join(plots_dir, "mufilter_bar_level_mip_calibration.png")
    c_bar_summary.SaveAs(bar_summary_png)
    os.system(f"cp -f {bar_summary_png} {os.path.join(artifact_dir, 'mufilter_bar_level_mip_calibration.png')}")
    print(f"Summary PNG saved to {bar_summary_png} and copied to artifact directory.")

if __name__ == '__main__':
    main()
