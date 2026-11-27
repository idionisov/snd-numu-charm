import os
import sys
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

langau_code = """
Double_t langaufun(Double_t *x, Double_t *par) {
   Double_t invsq2pi = 0.3989422804014;
   Double_t mpshift  = -0.22278298;
   Double_t np = 100.0;
   Double_t sc =   5.0;
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
"""
ROOT.gInterpreter.Declare(langau_code)

# Physical constants for Downstream Horizontal EJ-200 Scintillator & SiPM
SCINT_Y0      = 10000.0      # Photons / MeV (Eljen EJ-200 PVT)
# In rectangular bar, TIR trapping fraction towards ONE end face is 20.5% (with skew rays)
# (Isotropic cone is 18.35%, rectangular geometry adds ~2.15% trapped skew rays)
F_TRAP_SINGLE = 0.205        # Geometric trapping fraction per bar end
LAMBDA_ATT    = 180.0        # Effective optical attenuation length [cm]
D_CENTER      = 41.25        # Distance from bar center to SiPM [cm]
ETA_ATT       = np.exp(-D_CENTER / LAMBDA_ATT) # ~0.7952 attenuation factor
EPS_GEOM      = 36.0 / 100.0 # 6x6 mm^2 SiPM on 10x10 mm^2 face = 0.360
PDE_SIPM      = 0.400        # Hamamatsu S14160/S13360-6050HS PDE at 425 nm, 3V OV
QDC_TO_PC     = 3.60         # pC per QDC count (TOFPET2 at G_Q = 1.0)
Q_1PE_BARE    = 0.40054      # pC (G = 2.5e6 * 1.602e-19 C)
OCT_FACTOR    = 1.08         # 1 + K_OCT (8% optical cross-talk)
Q_1PE_EFF     = Q_1PE_BARE * OCT_FACTOR # 0.43258 pC / p.e.

# Single-SiPM optical transport yield factor: N_pe / MeV
YIELD_PE_PER_MEV = SCINT_Y0 * F_TRAP_SINGLE * ETA_ATT * EPS_GEOM * PDE_SIPM

# Load caches
cd = np.load('/afs/cern.ch/work/i/idioniso/snd-numu-charm/scratch/data_mip_qdc.npz')
data_qdc_center = cd['center']

cmc = np.load('/afs/cern.ch/work/i/idioniso/snd-numu-charm/scratch/mc_mip_edep.npz')
mc_edep_center = cmc['center']

def fit_landau_gauss(data_array, bin_min, bin_max, n_bins, p_init, fit_min, fit_max, name="h_fit", title=""):
    h = ROOT.TH1F(name, title, n_bins, bin_min, bin_max)
    for v in data_array:
        h.Fill(v)

    f_langau = ROOT.TF1(f"f_{name}", ROOT.langaufun, fit_min, fit_max, 4)
    f_langau.SetParNames("LandauWidth", "LandauMPV", "Area", "GausSigma")
    f_langau.SetParameters(*p_init)
    f_langau.SetParLimits(0, 0.05 * p_init[0], 5.0 * p_init[0])
    f_langau.SetParLimits(1, 0.50 * p_init[1], 2.0 * p_init[1])
    f_langau.SetParLimits(2, 0.10 * p_init[2], 10.0 * p_init[2])
    f_langau.SetParLimits(3, 0.05 * p_init[3], 5.0 * p_init[3])

    h.Fit(f_langau, "RBQS")
    
    mpv = f_langau.GetParameter(1)
    mpv_err = f_langau.GetParError(1)
    l_width = f_langau.GetParameter(0)
    l_width_err = f_langau.GetParError(0)
    g_sigma = f_langau.GetParameter(3)
    g_sigma_err = f_langau.GetParError(3)
    chi2 = f_langau.GetChisquare()
    ndf = f_langau.GetNDF()

    return h, f_langau, {
        'mpv': mpv, 'mpv_err': mpv_err,
        'l_width': l_width, 'l_width_err': l_width_err,
        'g_sigma': g_sigma, 'g_sigma_err': g_sigma_err,
        'chi2': chi2, 'ndf': ndf,
        'chi2_ndf': chi2 / max(1, ndf)
    }

# 1. Fit Data QDC (Central)
qdc_median = np.median(data_qdc_center)
qdc_n = len(data_qdc_center)
p_init_data = [4.0, qdc_median, qdc_n * (60.0 / 60.0), 5.0]
h_data, f_data, res_data = fit_landau_gauss(
    data_qdc_center, 0.0, 70.0, 70, p_init_data, 12.0, 60.0,
    name="h_qdc_data", title="Run 5408 Data: Central DS Hor Single-SiPM QDC"
)

# 2. Fit MC Edep (Central)
edep_median = np.median(mc_edep_center)
edep_n = len(mc_edep_center)
p_init_mc = [0.15, 1.70, edep_n * (5.0 / 50.0), 0.15]
h_mc, f_mc, res_mc = fit_landau_gauss(
    mc_edep_center, 0.5, 4.0, 50, p_init_mc, 1.0, 3.5,
    name="h_edep_mc", title="PMU Simulation: Central DS Hor Bar True Energy Loss"
)

# 3. Conversion and k_opt Computation
mpv_qdc = res_data['mpv']
mpv_qdc_err = res_data['mpv_err']
q_mip_pc = mpv_qdc * QDC_TO_PC # pC
q_mip_pc_err = mpv_qdc_err * QDC_TO_PC

# Measured Photoelectrons
n_pe_data_eff = q_mip_pc / Q_1PE_EFF
n_pe_data_eff_err = q_mip_pc_err / Q_1PE_EFF

n_pe_data_bare = q_mip_pc / Q_1PE_BARE
n_pe_data_bare_err = q_mip_pc_err / Q_1PE_BARE

# Simulation Energy and Model Photoelectrons
e_mpv_mc = res_mc['mpv']
e_mpv_mc_err = res_mc['mpv_err']
n_pe_model = e_mpv_mc * YIELD_PE_PER_MEV
n_pe_model_err = e_mpv_mc_err * YIELD_PE_PER_MEV

# Theoretical expectation for 1.78 MeV MPV
e_mpv_theory = 1.78
n_pe_model_theory = e_mpv_theory * YIELD_PE_PER_MEV

# k_opt ratios
k_opt_eff = n_pe_model / n_pe_data_eff
k_opt_eff_err = k_opt_eff * np.sqrt((e_mpv_mc_err / e_mpv_mc)**2 + (mpv_qdc_err / mpv_qdc)**2)

k_opt_bare = n_pe_model / n_pe_data_bare
k_opt_bare_err = k_opt_bare * np.sqrt((e_mpv_mc_err / e_mpv_mc)**2 + (mpv_qdc_err / mpv_qdc)**2)

k_opt_theory = n_pe_model_theory / n_pe_data_eff

print("\n" + "="*70)
print("FINAL RESULTS: MIP LANDAU PEAK, CHARGE, PHOTOELECTRONS, AND k_opt")
print("="*70)
print(f"Data Single-SiPM QDC MPV:      {mpv_qdc:.2f} +/- {mpv_qdc_err:.2f} counts (chi2/ndf = {res_data['chi2_ndf']:.2f})")
print(f"Data Integrated Charge Q_MIP:  {q_mip_pc:.2f} +/- {q_mip_pc_err:.2f} pC")
print(f"Data Photoelectrons N_pe (eff): {n_pe_data_eff:.1f} +/- {n_pe_data_eff_err:.1f} p.e. (with 8% OCT)")
print(f"Data Photoelectrons N_pe (bare):{n_pe_data_bare:.1f} +/- {n_pe_data_bare_err:.1f} p.e. (bare gain)")
print(f"MC True Energy Loss E_MPV:     {e_mpv_mc:.3f} +/- {e_mpv_mc_err:.3f} MeV (chi2/ndf = {res_mc['chi2_ndf']:.2f})")
print(f"Optical Transport Yield:       {YIELD_PE_PER_MEV:.2f} p.e. / MeV (single SiPM at d = 41.25 cm)")
print(f"Predicted N_pe (fitted E_MPV): {n_pe_model:.1f} +/- {n_pe_model_err:.1f} p.e.")
print(f"Predicted N_pe (nominal 1.78): {n_pe_model_theory:.1f} p.e.")
print(f"----------------------------------------------------------------------")
print(f"OPTIMAL SCALE FACTOR k_opt:   {k_opt_eff:.3f} +/- {k_opt_eff_err:.3f} (Model / Data, with OCT)")
print(f"OPTIMAL SCALE FACTOR (bare):  {k_opt_bare:.3f} +/- {k_opt_bare_err:.3f} (Model / Data, bare gain)")
print(f"Scale with nominal 1.78 MeV:   {k_opt_theory:.3f}")
print("="*70)

# Plotting Canvas
canv = ROOT.TCanvas("c_mip_calib", "MIP Landau Calibration", 1550, 700)
canv.Divide(2, 1)

# Left Pad: Data QDC
canv.cd(1)
ROOT.gPad.SetLeftMargin(0.14)
ROOT.gPad.SetRightMargin(0.06)
ROOT.gPad.SetTopMargin(0.10)
ROOT.gPad.SetBottomMargin(0.12)
ROOT.gPad.SetGrid(1, 1)

h_data.SetTitle("")
h_data.GetXaxis().SetTitle("Single-SiPM Calibrated QDC [counts]")
h_data.GetYaxis().SetTitle("Entries / 1.0 count")
h_data.GetXaxis().SetTitleSize(0.045)
h_data.GetYaxis().SetTitleSize(0.045)
h_data.GetXaxis().SetLabelSize(0.040)
h_data.GetYaxis().SetLabelSize(0.040)
h_data.SetLineColor(ROOT.kBlack)
h_data.SetMarkerStyle(20)
h_data.SetMarkerSize(0.9)
h_data.SetMarkerColor(ROOT.kBlack)
h_data.Draw("E")

f_data.SetLineColor(ROOT.kBlue + 1)
f_data.SetLineWidth(3)
f_data.Draw("SAME")

lat1 = ROOT.TLatex()
lat1.SetNDC()
lat1.SetTextFont(42)
lat1.SetTextSize(0.045)
lat1.DrawLatex(0.14, 0.92, "#bf{SND@LHC} #it{Work in Progress} - Run 5408 TI18 Data")

tbox1 = ROOT.TPaveText(0.46, 0.46, 0.91, 0.88, "NDC")
tbox1.SetFillColor(ROOT.kWhite)
tbox1.SetBorderSize(1)
tbox1.SetTextAlign(12)
tbox1.SetTextFont(42)
tbox1.SetTextSize(0.033)
tbox1.AddText("Through-going MIPs (|x - x_{c}| #leq 5 cm)")
tbox1.AddText("Subsystem: DS Horizontal (1.0 cm PVT bar)")
tbox1.AddText(f"Entries: {int(h_data.GetEntries())}")
tbox1.AddText(f"#bf{{mpv_qdc}}: {mpv_qdc:.2f} #pm {mpv_qdc_err:.2f} counts")
tbox1.AddText(f"#sigma_{{Landau}}: {res_data['l_width']:.2f} #pm {res_data['l_width_err']:.2f}")
tbox1.AddText(f"#sigma_{{Gaus}}: {res_data['g_sigma']:.2f} #pm {res_data['g_sigma_err']:.2f}")
tbox1.AddText(f"#chi^{{2}} / ndf: {res_data['chi2_ndf']:.2f}")
tbox1.AddText(f"#bf{{Q_{{MIP}}}}: {q_mip_pc:.1f} #pm {q_mip_pc_err:.1f} pC (3.6 pC/cnt)")
tbox1.AddText(f"#bf{{N_{{p.e.}}^{{data}}}}: #bf{{{n_pe_data_eff:.1f} #pm {n_pe_data_eff_err:.1f} p.e.}} (with OCT)")
tbox1.Draw()

# Right Pad: MC Edep
canv.cd(2)
ROOT.gPad.SetLeftMargin(0.14)
ROOT.gPad.SetRightMargin(0.06)
ROOT.gPad.SetTopMargin(0.10)
ROOT.gPad.SetBottomMargin(0.12)
ROOT.gPad.SetGrid(1, 1)

h_mc.SetTitle("")
h_mc.GetXaxis().SetTitle("True Energy Deposition #DeltaE [MeV]")
h_mc.GetYaxis().SetTitle("Entries / 0.07 MeV")
h_mc.GetXaxis().SetTitleSize(0.045)
h_mc.GetYaxis().SetTitleSize(0.045)
h_mc.GetXaxis().SetLabelSize(0.040)
h_mc.GetYaxis().SetLabelSize(0.040)
h_mc.SetLineColor(ROOT.kBlack)
h_mc.SetMarkerStyle(20)
h_mc.SetMarkerSize(0.9)
h_mc.SetMarkerColor(ROOT.kBlack)
h_mc.Draw("E")

f_mc.SetLineColor(ROOT.kRed + 1)
f_mc.SetLineWidth(3)
f_mc.Draw("SAME")

lat2 = ROOT.TLatex()
lat2.SetNDC()
lat2.SetTextFont(42)
lat2.SetTextSize(0.045)
lat2.DrawLatex(0.14, 0.92, "#bf{SND@LHC} #it{Simulation} - PMU 2022 Sample")

tbox2 = ROOT.TPaveText(0.44, 0.42, 0.91, 0.88, "NDC")
tbox2.SetFillColor(ROOT.kWhite)
tbox2.SetBorderSize(1)
tbox2.SetTextAlign(12)
tbox2.SetTextFont(42)
tbox2.SetTextSize(0.033)
tbox2.AddText("Through-going MIPs (|x - x_{c}| #leq 5 cm)")
tbox2.AddText(f"Entries: {int(h_mc.GetEntries())}")
tbox2.AddText(f"#bf{{E_{{MPV}}}}: {e_mpv_mc:.3f} #pm {e_mpv_mc_err:.3f} MeV")
tbox2.AddText(f"#sigma_{{Landau}}: {res_mc['l_width']:.3f} #pm {res_mc['l_width_err']:.3f} MeV")
tbox2.AddText(f"#sigma_{{Gaus}}: {res_mc['g_sigma']:.3f} #pm {res_mc['g_sigma_err']:.3f} MeV")
tbox2.AddText(f"#chi^{{2}} / ndf: {res_mc['chi2_ndf']:.2f}")
tbox2.AddText(f"#bf{{N_{{p.e.}}^{{model}}}}: #bf{{{n_pe_model:.1f} #pm {n_pe_model_err:.1f} p.e.}} (d = 41.25 cm)")
tbox2.AddText(f"Optical Yield: {YIELD_PE_PER_MEV:.1f} p.e. / MeV")
tbox2.AddText("----------------------------------------")
tbox2.AddText(f"#bf{{k_{{opt}} = N_{{p.e.}}^{{model}} / N_{{p.e.}}^{{data}} = {k_opt_eff:.2f} #pm {k_opt_eff_err:.2f}}}")
tbox2.Draw()

out_path = "/afs/cern.ch/work/i/idioniso/snd-numu-charm/plots/pmu_mip_landau_pe_calibration.png"
canv.SaveAs(out_path)
artifact_path = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2/pmu_mip_landau_pe_calibration.png"
os.system(f"cp -f {out_path} {artifact_path}")
print(f"Canvas saved and copied to {artifact_path} successfully!")
