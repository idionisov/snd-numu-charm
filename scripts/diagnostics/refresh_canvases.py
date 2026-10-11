#!/usr/bin/env python3
"""
Refreshes the diagnostic canvases and summary objects in charm_diagnostic_nu14_2022.root
and re-exports PNGs to /eos/user/i/idioniso/snd-numu-charm/plots_charm_diagnostic.
Ensures CHORUS and E531 are drawn on sigma_charm/sigma_CC canvases and omitted from dimuon canvases.
"""
import os
import sys
import math
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Add diagnostics directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from charm_diagnostic import create_slow_rescaling_model, clopper_pearson_errors

ROOT_PATH = "/eos/user/i/idioniso/snd-numu-charm/charm_diagnostic_nu14_2022.root"
PLOTS_DIR = "/eos/user/i/idioniso/snd-numu-charm/plots_charm_diagnostic"

def main():
    print(f"Opening {ROOT_PATH} for canvas refresh...")
    fin = ROOT.TFile.Open(ROOT_PATH, "UPDATE")
    if not fin or fin.IsZombie():
        print(f"Error opening {ROOT_PATH}")
        sys.exit(1)

    # 1. Read binned histograms from Summary
    h_all = fin.Get("Summary/h_cc_all_ebins")
    h_charm = fin.Get("Summary/h_cc_charm_ebins")
    h_all_nu = fin.Get("Summary/h_cc_all_nu_ebins")
    h_charm_nu = fin.Get("Summary/h_cc_charm_nu_ebins")
    h_all_an = fin.Get("Summary/h_cc_all_antinu_ebins")
    h_charm_an = fin.Get("Summary/h_cc_charm_antinu_ebins")
    h_2mu = fin.Get("Summary/h_cc_dimuon_ebins")
    h_2mu_nu = fin.Get("Summary/h_cc_dimuon_nu_ebins")
    h_2mu_an = fin.Get("Summary/h_cc_dimuon_antinu_ebins")

    n_nu_ebins = h_all.GetNbinsX()
    nu_bin_edges = [h_all.GetXaxis().GetBinLowEdge(1)] + [h_all.GetXaxis().GetBinUpEdge(b) for b in range(1, n_nu_ebins + 1)]
    e_max_model = max(1000.0, nu_bin_edges[-1])

    # 2. Build Slow Rescaling Models (Inclusive Charm)
    f_sr_w_nu, g_sr_w_nu_inner, g_sr_w_nu_outer = create_slow_rescaling_model(
        name="f_slow_rescaling_w_nu",
        e_min=0.0,
        e_max=e_max_model,
        target="tungsten",
        nu_type="nu",
        line_color=ROOT.kBlue + 1,
        band_color=ROOT.kAzure - 4,
    )
    f_sr_w_antinu, g_sr_w_antinu_inner, g_sr_w_antinu_outer = create_slow_rescaling_model(
        name="f_slow_rescaling_w_antinu",
        e_min=0.0,
        e_max=e_max_model,
        target="tungsten",
        nu_type="antinu",
        line_color=ROOT.kMagenta + 2,
        band_color=ROOT.kMagenta - 9,
    )
    f_sr_w_inc, g_sr_w_inc_inner, g_sr_w_inc_outer = create_slow_rescaling_model(
        name="f_slow_rescaling_w_inclusive",
        e_min=0.0,
        e_max=e_max_model,
        target="tungsten",
        nu_type="inclusive",
        line_color=ROOT.kRed + 1,
        band_color=ROOT.kRed - 7,
    )
    f_sr_iso_inc, _, _ = create_slow_rescaling_model(
        name="f_slow_rescaling_isoscalar_inclusive",
        e_min=0.0,
        e_max=e_max_model,
        target="isoscalar",
        nu_type="inclusive",
        line_color=ROOT.kGray + 2,
    )
    f_sr_iso_inc.SetLineStyle(7)
    f_sr_iso_inc.SetLineWidth(2)

    f_sr_iso_nu, _, _ = create_slow_rescaling_model(
        name="f_slow_rescaling_isoscalar_nu",
        e_min=0.0,
        e_max=e_max_model,
        target="isoscalar",
        nu_type="nu",
        line_color=ROOT.kGray + 2,
    )
    f_sr_iso_nu.SetLineStyle(7)
    f_sr_iso_nu.SetLineWidth(2)

    f_sr_iso_antinu, _, _ = create_slow_rescaling_model(
        name="f_slow_rescaling_isoscalar_antinu",
        e_min=0.0,
        e_max=e_max_model,
        target="isoscalar",
        nu_type="antinu",
        line_color=ROOT.kGray + 2,
    )
    f_sr_iso_antinu.SetLineStyle(7)
    f_sr_iso_antinu.SetLineWidth(2)

    # 3. Load CHORUS and E531
    g_chorus = None
    chorus_csv_path = os.path.join(SCRIPT_DIR, "measurements", "chorus_sigmaCharmDivSigmaCC_vs_Enu.csv")
    if os.path.exists(chorus_csv_path):
        try:
            chorus_pts = []
            with open(chorus_csv_path, "r") as f_ch:
                for line in f_ch:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("E_nu"):
                        continue
                    parts = [float(p.strip()) for p in line.split(",")]
                    chorus_pts.append(parts)
            if chorus_pts:
                g_chorus = ROOT.TGraphErrors(len(chorus_pts))
                g_chorus.SetName("g_chorus_measurement")
                g_chorus.SetTitle("CHORUS (Emulsion, #nu_{#mu})")
                for idx_ch, (enu, enu_err, val, val_err) in enumerate(chorus_pts):
                    g_chorus.SetPoint(idx_ch, enu, val)
                    g_chorus.SetPointError(idx_ch, enu_err, val_err)
                g_chorus.SetMarkerStyle(25)  # Open square
                g_chorus.SetMarkerSize(1.05)
                g_chorus.SetMarkerColor(ROOT.kGray + 2)
                g_chorus.SetLineColor(ROOT.kGray + 2)
                g_chorus.SetLineWidth(1)
        except Exception as e_ch:
            print(f"[Warning] Failed to load CHORUS measurement: {e_ch}")

    g_e531 = None
    e531_csv_path = os.path.join(SCRIPT_DIR, "measurements", "e531_sigmaCharmDivSigmaCC_vs_Enu.csv")
    if os.path.exists(e531_csv_path):
        try:
            e531_pts = []
            with open(e531_csv_path, "r", encoding="utf-8-sig") as f_e:
                for line in f_e:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("bin_index"):
                        continue
                    parts = [float(p.strip()) for p in line.split(",")]
                    e531_pts.append(parts)
            if e531_pts:
                g_e531 = ROOT.TGraphAsymmErrors(len(e531_pts))
                g_e531.SetName("g_e531_measurement")
                g_e531.SetTitle("E531 (Emulsion, #nu_{#mu})")
                for idx_e, p in enumerate(e531_pts):
                    e_mean = p[3]
                    e_err_lo = p[4]
                    e_err_hi = p[5]
                    r_val = p[6] / 100.0
                    r_err_lo = p[7] / 100.0
                    r_err_hi = p[8] / 100.0
                    g_e531.SetPoint(idx_e, e_mean, r_val)
                    g_e531.SetPointError(idx_e, e_err_lo, e_err_hi, r_err_lo, r_err_hi)
                g_e531.SetMarkerStyle(24)  # Open circle
                g_e531.SetMarkerSize(1.05)
                g_e531.SetMarkerColor(ROOT.kGray + 1)
                g_e531.SetLineColor(ROOT.kGray + 1)
                g_e531.SetLineWidth(1)
        except Exception as e_e531:
            print(f"[Warning] Failed to load E531 measurement: {e_e531}")

    # 4. Build MC Graphs
    g_charm_rate_mc = ROOT.TGraphAsymmErrors(n_nu_ebins)
    g_charm_rate_mc.SetName("g_charm_rate_mc")
    g_charm_rate_mc_nu = ROOT.TGraphAsymmErrors(n_nu_ebins)
    g_charm_rate_mc_nu.SetName("g_charm_rate_mc_nu")
    g_charm_rate_mc_antinu = ROOT.TGraphAsymmErrors(n_nu_ebins)
    g_charm_rate_mc_antinu.SetName("g_charm_rate_mc_antinu")

    for i_b in range(n_nu_ebins):
        x_center = 0.5 * (nu_bin_edges[i_b] + nu_bin_edges[i_b + 1])
        x_err = 0.5 * (nu_bin_edges[i_b + 1] - nu_bin_edges[i_b])

        # Inclusive
        n_cc = h_all.GetBinContent(i_b + 1)
        n_c = h_charm.GetBinContent(i_b + 1)
        val_pct, err_lo_pct, err_hi_pct = clopper_pearson_errors(n_c, n_cc, 0.6827)
        g_charm_rate_mc.SetPoint(i_b, x_center, val_pct / 100.0)
        g_charm_rate_mc.SetPointError(i_b, x_err, x_err, err_lo_pct / 100.0, err_hi_pct / 100.0)

        # nu_mu
        n_cc_nu = h_all_nu.GetBinContent(i_b + 1) if h_all_nu else 0
        n_c_nu = h_charm_nu.GetBinContent(i_b + 1) if h_charm_nu else 0
        val_pct_nu, lo_nu, hi_nu = clopper_pearson_errors(n_c_nu, n_cc_nu, 0.6827)
        g_charm_rate_mc_nu.SetPoint(i_b, x_center, val_pct_nu / 100.0)
        g_charm_rate_mc_nu.SetPointError(i_b, x_err, x_err, lo_nu / 100.0, hi_nu / 100.0)

        # anti_nu_mu
        n_cc_an = h_all_an.GetBinContent(i_b + 1) if h_all_an else 0
        n_c_an = h_charm_an.GetBinContent(i_b + 1) if h_charm_an else 0
        val_pct_an, lo_an, hi_an = clopper_pearson_errors(n_c_an, n_cc_an, 0.6827)
        g_charm_rate_mc_antinu.SetPoint(i_b, x_center, val_pct_an / 100.0)
        g_charm_rate_mc_antinu.SetPointError(i_b, x_err, x_err, lo_an / 100.0, hi_an / 100.0)

    g_charm_rate_mc.SetMarkerStyle(20)
    g_charm_rate_mc.SetMarkerSize(1.2)
    g_charm_rate_mc.SetMarkerColor(ROOT.kBlack)
    g_charm_rate_mc.SetLineColor(ROOT.kBlack)
    g_charm_rate_mc.SetLineWidth(2)

    g_charm_rate_mc_nu.SetMarkerStyle(20)
    g_charm_rate_mc_nu.SetMarkerSize(1.2)
    g_charm_rate_mc_nu.SetMarkerColor(ROOT.kBlue + 2)
    g_charm_rate_mc_nu.SetLineColor(ROOT.kBlue + 2)
    g_charm_rate_mc_nu.SetLineWidth(2)

    g_charm_rate_mc_antinu.SetMarkerStyle(20)
    g_charm_rate_mc_antinu.SetMarkerSize(1.2)
    g_charm_rate_mc_antinu.SetMarkerColor(ROOT.kMagenta + 2)
    g_charm_rate_mc_antinu.SetLineColor(ROOT.kMagenta + 2)
    g_charm_rate_mc_antinu.SetLineWidth(2)

    y_max_frame = max(0.28, (h_charm.GetMaximum() / max(1.0, h_all.GetMaximum())) * 1.50)

    # 5. Build Inclusive Canvases WITH CHORUS and E531
    dir_canvases = fin.Get("Canvases")
    if not dir_canvases:
        dir_canvases = fin.mkdir("Canvases")
    dir_canvases.cd()

    # Canvas 1a: Inclusive
    c_rate_inc = ROOT.TCanvas("c_charm_rate_vs_energy", "Charm Production Rate vs Neutrino Energy (Inclusive)", 950, 700)
    c_rate_inc.SetLeftMargin(0.12)
    c_rate_inc.SetRightMargin(0.06)
    c_rate_inc.SetTopMargin(0.08)
    c_rate_inc.SetBottomMargin(0.12)
    c_rate_inc.SetGrid()

    h_frame_inc = c_rate_inc.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_frame)
    h_frame_inc.SetTitle("Charm Production Rate: Inclusive (#nu_{#mu} + #bar{#nu}_{#mu});E_{#nu} [GeV];#sigma_{charm} / #sigma_{CC}")
    h_frame_inc.GetYaxis().SetTitleOffset(1.25)

    g_sr_w_inc_outer.Draw("3 same")
    g_sr_w_inc_inner.Draw("3 same")
    f_sr_iso_inc.Draw("same")
    f_sr_w_inc.Draw("same")
    g_charm_rate_mc.Draw("P E1 same")
    if g_chorus:
        g_chorus.Draw("P E1 same")
    if g_e531:
        g_e531.Draw("P E1 same")

    leg_inc = ROOT.TLegend(0.48, 0.58, 0.92, 0.89)
    leg_inc.SetBorderSize(1)
    leg_inc.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_inc.SetTextSize(0.022)
    leg_inc.AddEntry(g_charm_rate_mc, "MC Inclusive (#nu_{#mu}+#bar{#nu}_{#mu})", "lep")
    if g_chorus:
        leg_inc.AddEntry(g_chorus, "CHORUS (Emulsion, #nu_{#mu})", "lep")
    if g_e531:
        leg_inc.AddEntry(g_e531, "E531 (Emulsion, #nu_{#mu})", "lep")
    leg_inc.AddEntry(f_sr_w_inc, "Slow Rescaling (W, #nu+#bar{#nu})", "l")
    leg_inc.AddEntry(g_sr_w_inc_inner, "Slow Rescaling Base 1#sigma Belt (W)", "f")
    leg_inc.AddEntry(g_sr_w_inc_outer, "+ High-E QCD / Sea Evolution", "f")
    leg_inc.AddEntry(f_sr_iso_inc, "Slow Rescaling (Isoscalar N=Z)", "l")
    leg_inc.Draw("same")
    c_rate_inc.Write("c_charm_rate_vs_energy", ROOT.TObject.kOverwrite)

    # Canvas 1b: nu_only
    c_rate_nu = ROOT.TCanvas("c_charm_rate_vs_energy_nu_only", "Charm Production Rate vs Neutrino Energy (nu_mu)", 950, 700)
    c_rate_nu.SetLeftMargin(0.12)
    c_rate_nu.SetRightMargin(0.06)
    c_rate_nu.SetTopMargin(0.08)
    c_rate_nu.SetBottomMargin(0.12)
    c_rate_nu.SetGrid()

    h_frame_nu = c_rate_nu.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_frame)
    h_frame_nu.SetTitle("Charm Production Rate: #nu_{#mu} CC;E_{#nu} [GeV];#sigma_{charm} / #sigma_{CC}")
    h_frame_nu.GetYaxis().SetTitleOffset(1.25)

    g_sr_w_nu_outer.Draw("3 same")
    g_sr_w_nu_inner.Draw("3 same")
    f_sr_iso_nu.Draw("same")
    f_sr_w_nu.Draw("same")
    g_charm_rate_mc_nu.Draw("P E1 same")
    if g_chorus:
        g_chorus.Draw("P E1 same")
    if g_e531:
        g_e531.Draw("P E1 same")

    leg_nu = ROOT.TLegend(0.48, 0.58, 0.92, 0.89)
    leg_nu.SetBorderSize(1)
    leg_nu.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_nu.SetTextSize(0.022)
    leg_nu.AddEntry(g_charm_rate_mc_nu, "MC #nu_{#mu} CC", "lep")
    if g_chorus:
        leg_nu.AddEntry(g_chorus, "CHORUS (Emulsion, #nu_{#mu})", "lep")
    if g_e531:
        leg_nu.AddEntry(g_e531, "E531 (Emulsion, #nu_{#mu})", "lep")
    leg_nu.AddEntry(f_sr_w_nu, "Slow Rescaling (W, #nu_{#mu})", "l")
    leg_nu.AddEntry(g_sr_w_nu_inner, "Slow Rescaling Base 1#sigma Belt (W)", "f")
    leg_nu.AddEntry(g_sr_w_nu_outer, "+ High-E QCD / Sea Evolution", "f")
    leg_nu.AddEntry(f_sr_iso_nu, "Slow Rescaling (Isoscalar N=Z)", "l")
    leg_nu.Draw("same")
    c_rate_nu.Write("c_charm_rate_vs_energy_nu_only", ROOT.TObject.kOverwrite)

    # Canvas 1c: antinu_only
    c_rate_antinu = ROOT.TCanvas("c_charm_rate_vs_energy_antinu_only", "Charm Production Rate vs Neutrino Energy (anti_nu_mu)", 950, 700)
    c_rate_antinu.SetLeftMargin(0.12)
    c_rate_antinu.SetRightMargin(0.06)
    c_rate_antinu.SetTopMargin(0.08)
    c_rate_antinu.SetBottomMargin(0.12)
    c_rate_antinu.SetGrid()

    h_frame_antinu = c_rate_antinu.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_frame * 1.25)
    h_frame_antinu.SetTitle("Charm Production Rate: #bar{#nu}_{#mu} CC;E_{#nu} [GeV];#sigma_{charm} / #sigma_{CC}")
    h_frame_antinu.GetYaxis().SetTitleOffset(1.25)

    g_sr_w_antinu_outer.Draw("3 same")
    g_sr_w_antinu_inner.Draw("3 same")
    f_sr_iso_antinu.Draw("same")
    f_sr_w_antinu.Draw("same")
    g_charm_rate_mc_antinu.Draw("P E1 same")

    leg_antinu = ROOT.TLegend(0.48, 0.66, 0.92, 0.89)
    leg_antinu.SetBorderSize(1)
    leg_antinu.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_antinu.SetTextSize(0.024)
    leg_antinu.AddEntry(g_charm_rate_mc_antinu, "MC #bar{#nu}_{#mu} CC", "lep")
    leg_antinu.AddEntry(f_sr_w_antinu, "Slow Rescaling (W, #bar{#nu}_{#mu})", "l")
    leg_antinu.AddEntry(g_sr_w_antinu_inner, "Slow Rescaling Base 1#sigma Belt (W)", "f")
    leg_antinu.AddEntry(g_sr_w_antinu_outer, "+ High-E QCD / Sea Evolution", "f")
    leg_antinu.AddEntry(f_sr_iso_antinu, "Slow Rescaling (Isoscalar N=Z)", "l")
    leg_antinu.Draw("same")
    # 5b. Build Dimuon Canvases (WITHOUT CHORUS and E531)
    g_dimuon_rate_mc = ROOT.TGraphAsymmErrors(n_nu_ebins)
    g_dimuon_rate_mc.SetName("g_dimuon_rate_mc")
    g_dimuon_rate_mc_nu = ROOT.TGraphAsymmErrors(n_nu_ebins)
    g_dimuon_rate_mc_nu.SetName("g_dimuon_rate_mc_nu")
    g_dimuon_rate_mc_antinu = ROOT.TGraphAsymmErrors(n_nu_ebins)
    g_dimuon_rate_mc_antinu.SetName("g_dimuon_rate_mc_antinu")

    for i_b in range(n_nu_ebins):
        x_center = 0.5 * (nu_bin_edges[i_b] + nu_bin_edges[i_b + 1])
        x_err = 0.5 * (nu_bin_edges[i_b + 1] - nu_bin_edges[i_b])

        # Inclusive
        n_cc = h_all.GetBinContent(i_b + 1)
        n_2mu = h_2mu.GetBinContent(i_b + 1) if h_2mu else 0
        val_pct, err_lo_pct, err_hi_pct = clopper_pearson_errors(n_2mu, n_cc, 0.6827)
        g_dimuon_rate_mc.SetPoint(i_b, x_center, val_pct / 100.0)
        g_dimuon_rate_mc.SetPointError(i_b, x_err, x_err, err_lo_pct / 100.0, err_hi_pct / 100.0)

        # nu_mu
        n_cc_nu = h_all_nu.GetBinContent(i_b + 1) if h_all_nu else 0
        n_2mu_nu = h_2mu_nu.GetBinContent(i_b + 1) if h_2mu_nu else 0
        val_pct_nu, lo_nu, hi_nu = clopper_pearson_errors(n_2mu_nu, n_cc_nu, 0.6827)
        g_dimuon_rate_mc_nu.SetPoint(i_b, x_center, val_pct_nu / 100.0)
        g_dimuon_rate_mc_nu.SetPointError(i_b, x_err, x_err, lo_nu / 100.0, hi_nu / 100.0)

        # anti_nu_mu
        n_cc_an = h_all_an.GetBinContent(i_b + 1) if h_all_an else 0
        n_2mu_an = h_2mu_an.GetBinContent(i_b + 1) if h_2mu_an else 0
        val_pct_an, lo_an, hi_an = clopper_pearson_errors(n_2mu_an, n_cc_an, 0.6827)
        g_dimuon_rate_mc_antinu.SetPoint(i_b, x_center, val_pct_an / 100.0)
        g_dimuon_rate_mc_antinu.SetPointError(i_b, x_err, x_err, lo_an / 100.0, hi_an / 100.0)

    g_dimuon_rate_mc.SetMarkerStyle(20)
    g_dimuon_rate_mc.SetMarkerSize(1.2)
    g_dimuon_rate_mc.SetMarkerColor(ROOT.kBlack)
    g_dimuon_rate_mc.SetLineColor(ROOT.kBlack)
    g_dimuon_rate_mc.SetLineWidth(2)

    g_dimuon_rate_mc_nu.SetMarkerStyle(20)
    g_dimuon_rate_mc_nu.SetMarkerSize(1.2)
    g_dimuon_rate_mc_nu.SetMarkerColor(ROOT.kBlue + 2)
    g_dimuon_rate_mc_nu.SetLineColor(ROOT.kBlue + 2)
    g_dimuon_rate_mc_nu.SetLineWidth(2)

    g_dimuon_rate_mc_antinu.SetMarkerStyle(20)
    g_dimuon_rate_mc_antinu.SetMarkerSize(1.2)
    g_dimuon_rate_mc_antinu.SetMarkerColor(ROOT.kMagenta + 2)
    g_dimuon_rate_mc_antinu.SetLineColor(ROOT.kMagenta + 2)
    g_dimuon_rate_mc_antinu.SetLineWidth(2)

    # Models with B_mu
    f_sr_2mu_w_inc, g_sr_2mu_w_inc_inner, g_sr_2mu_w_inc_outer = create_slow_rescaling_model(
        name="f_slow_rescaling_dimuon_w_inclusive", e_min=0.0, e_max=e_max_model,
        target="tungsten", nu_type="both", line_color=ROOT.kRed + 1, band_color=ROOT.kRed - 9,
        scale=0.0848, scale_err=0.0035,
    )
    f_sr_2mu_iso_inc, _, _ = create_slow_rescaling_model(
        name="f_slow_rescaling_dimuon_isoscalar_inclusive", e_min=0.0, e_max=e_max_model,
        target="isoscalar", nu_type="both", line_color=ROOT.kGray + 2,
        scale=0.0848, scale_err=0.0035,
    )
    f_sr_2mu_iso_inc.SetLineStyle(7)
    f_sr_2mu_iso_inc.SetLineWidth(2)

    f_sr_2mu_w_nu, g_sr_2mu_w_nu_inner, g_sr_2mu_w_nu_outer = create_slow_rescaling_model(
        name="f_slow_rescaling_dimuon_w_nu", e_min=0.0, e_max=e_max_model,
        target="tungsten", nu_type="nu", line_color=ROOT.kBlue + 1, band_color=ROOT.kAzure - 4,
        scale=0.0848, scale_err=0.0035,
    )
    f_sr_2mu_iso_nu, _, _ = create_slow_rescaling_model(
        name="f_slow_rescaling_dimuon_isoscalar_nu", e_min=0.0, e_max=e_max_model,
        target="isoscalar", nu_type="nu", line_color=ROOT.kGray + 2,
        scale=0.0848, scale_err=0.0035,
    )
    f_sr_2mu_iso_nu.SetLineStyle(7)
    f_sr_2mu_iso_nu.SetLineWidth(2)

    f_sr_2mu_w_antinu, g_sr_2mu_w_antinu_inner, g_sr_2mu_w_antinu_outer = create_slow_rescaling_model(
        name="f_slow_rescaling_dimuon_w_antinu", e_min=0.0, e_max=e_max_model,
        target="tungsten", nu_type="antinu", line_color=ROOT.kMagenta + 1, band_color=ROOT.kMagenta - 9,
        scale=0.0848, scale_err=0.0035,
    )
    f_sr_2mu_iso_antinu, _, _ = create_slow_rescaling_model(
        name="f_slow_rescaling_dimuon_isoscalar_antinu", e_min=0.0, e_max=e_max_model,
        target="isoscalar", nu_type="antinu", line_color=ROOT.kGray + 2,
        scale=0.0848, scale_err=0.0035,
    )
    f_sr_2mu_iso_antinu.SetLineStyle(7)
    f_sr_2mu_iso_antinu.SetLineWidth(2)

    y_max_dimuon = 0.022
    y_max_dimuon_nu = 0.020
    y_max_dimuon_antinu = 0.028

    c_dimuon_inc = ROOT.TCanvas("c_charm_dimuon_rate_vs_energy", "Charm Dimuon Production Rate vs Neutrino Energy (Inclusive)", 950, 700)
    c_dimuon_inc.SetLeftMargin(0.12)
    c_dimuon_inc.SetRightMargin(0.06)
    c_dimuon_inc.SetTopMargin(0.08)
    c_dimuon_inc.SetBottomMargin(0.12)
    c_dimuon_inc.SetGrid()
    h_frame_d_inc = c_dimuon_inc.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_dimuon)
    h_frame_d_inc.SetTitle("Charm Dimuon Production Rate: Inclusive (#nu_{#mu} + #bar{#nu}_{#mu});E_{#nu} [GeV];#sigma_{charm, 2#mu} / #sigma_{CC}")
    h_frame_d_inc.GetYaxis().SetTitleOffset(1.25)
    g_sr_2mu_w_inc_outer.Draw("3 same")
    g_sr_2mu_w_inc_inner.Draw("3 same")
    f_sr_2mu_iso_inc.Draw("same")
    f_sr_2mu_w_inc.Draw("same")
    g_dimuon_rate_mc.Draw("P E1 same")
    leg_d_inc = ROOT.TLegend(0.48, 0.62, 0.92, 0.89)
    leg_d_inc.SetBorderSize(1)
    leg_d_inc.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_d_inc.SetTextSize(0.022)
    leg_d_inc.AddEntry(g_dimuon_rate_mc, "MC Inclusive Dimuon (#nu_{#mu}+#bar{#nu}_{#mu})", "lep")
    leg_d_inc.AddEntry(f_sr_2mu_w_inc, "Slow Rescaling #times #LTB_{#mu}#GT (W, #nu+#bar{#nu})", "l")
    leg_d_inc.AddEntry(g_sr_2mu_w_inc_inner, "Slow Rescaling Base 1#sigma Belt (W)", "f")
    leg_d_inc.AddEntry(g_sr_2mu_w_inc_outer, "+ High-E QCD / Sea Evolution", "f")
    leg_d_inc.AddEntry(f_sr_2mu_iso_inc, "Slow Rescaling #times #LTB_{#mu}#GT (Isoscalar N=Z)", "l")
    leg_d_inc.Draw("same")
    c_dimuon_inc.Write("c_charm_dimuon_rate_vs_energy", ROOT.TObject.kOverwrite)

    c_dimuon_nu = ROOT.TCanvas("c_charm_dimuon_rate_vs_energy_nu_only", "Charm Dimuon Production Rate vs Neutrino Energy (nu_mu)", 950, 700)
    c_dimuon_nu.SetLeftMargin(0.12)
    c_dimuon_nu.SetRightMargin(0.06)
    c_dimuon_nu.SetTopMargin(0.08)
    c_dimuon_nu.SetBottomMargin(0.12)
    c_dimuon_nu.SetGrid()
    h_frame_d_nu = c_dimuon_nu.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_dimuon_nu)
    h_frame_d_nu.SetTitle("Charm Dimuon Production Rate: #nu_{#mu} CC;E_{#nu} [GeV];#sigma_{charm, 2#mu} / #sigma_{CC}")
    h_frame_d_nu.GetYaxis().SetTitleOffset(1.25)
    g_sr_2mu_w_nu_outer.Draw("3 same")
    g_sr_2mu_w_nu_inner.Draw("3 same")
    f_sr_2mu_iso_nu.Draw("same")
    f_sr_2mu_w_nu.Draw("same")
    g_dimuon_rate_mc_nu.Draw("P E1 same")
    leg_d_nu = ROOT.TLegend(0.48, 0.62, 0.92, 0.89)
    leg_d_nu.SetBorderSize(1)
    leg_d_nu.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_d_nu.SetTextSize(0.022)
    leg_d_nu.AddEntry(g_dimuon_rate_mc_nu, "MC #nu_{#mu} Dimuon CC", "lep")
    leg_d_nu.AddEntry(f_sr_2mu_w_nu, "Slow Rescaling #times #LTB_{#mu}#GT (W, #nu_{#mu})", "l")
    leg_d_nu.AddEntry(g_sr_2mu_w_nu_inner, "Slow Rescaling Base 1#sigma Belt (W)", "f")
    leg_d_nu.AddEntry(g_sr_2mu_w_nu_outer, "+ High-E QCD / Sea Evolution", "f")
    leg_d_nu.AddEntry(f_sr_2mu_iso_nu, "Slow Rescaling #times #LTB_{#mu}#GT (Isoscalar N=Z)", "l")
    leg_d_nu.Draw("same")
    c_dimuon_nu.Write("c_charm_dimuon_rate_vs_energy_nu_only", ROOT.TObject.kOverwrite)

    c_dimuon_antinu = ROOT.TCanvas("c_charm_dimuon_rate_vs_energy_antinu_only", "Charm Dimuon Production Rate vs Neutrino Energy (anti_nu_mu)", 950, 700)
    c_dimuon_antinu.SetLeftMargin(0.12)
    c_dimuon_antinu.SetRightMargin(0.06)
    c_dimuon_antinu.SetTopMargin(0.08)
    c_dimuon_antinu.SetBottomMargin(0.12)
    c_dimuon_antinu.SetGrid()
    h_frame_d_an = c_dimuon_antinu.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_dimuon_antinu)
    h_frame_d_an.SetTitle("Charm Dimuon Production Rate: #bar{#nu}_{#mu} CC;E_{#nu} [GeV];#sigma_{charm, 2#mu} / #sigma_{CC}")
    h_frame_d_an.GetYaxis().SetTitleOffset(1.25)
    g_sr_2mu_w_antinu_outer.Draw("3 same")
    g_sr_2mu_w_antinu_inner.Draw("3 same")
    f_sr_2mu_iso_antinu.Draw("same")
    f_sr_2mu_w_antinu.Draw("same")
    g_dimuon_rate_mc_antinu.Draw("P E1 same")
    leg_d_an = ROOT.TLegend(0.48, 0.62, 0.92, 0.89)
    leg_d_an.SetBorderSize(1)
    leg_d_an.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_d_an.SetTextSize(0.022)
    leg_d_an.AddEntry(g_dimuon_rate_mc_antinu, "MC #bar{#nu}_{#mu} Dimuon CC", "lep")
    leg_d_an.AddEntry(f_sr_2mu_w_antinu, "Slow Rescaling #times #LTB_{#mu}#GT (W, #bar{#nu}_{#mu})", "l")
    leg_d_an.AddEntry(g_sr_2mu_w_antinu_inner, "Slow Rescaling Base 1#sigma Belt (W)", "f")
    leg_d_an.AddEntry(g_sr_2mu_w_antinu_outer, "+ High-E QCD / Sea Evolution", "f")
    leg_d_an.AddEntry(f_sr_2mu_iso_antinu, "Slow Rescaling #times #LTB_{#mu}#GT (Isoscalar N=Z)", "l")
    leg_d_an.Draw("same")
    c_dimuon_antinu.Write("c_charm_dimuon_rate_vs_energy_antinu_only", ROOT.TObject.kOverwrite)

    # Save dimuon objects to Summary as well
    dir_summary = fin.Get("Summary")
    if dir_summary:
        dir_summary.cd()
        if g_chorus:
            g_chorus.Write("g_chorus_measurement", ROOT.TObject.kOverwrite)
        if g_e531:
            g_e531.Write("g_e531_measurement", ROOT.TObject.kOverwrite)
        g_dimuon_rate_mc.Write("g_dimuon_rate_mc", ROOT.TObject.kOverwrite)
        g_dimuon_rate_mc_nu.Write("g_dimuon_rate_mc_nu", ROOT.TObject.kOverwrite)
        g_dimuon_rate_mc_antinu.Write("g_dimuon_rate_mc_antinu", ROOT.TObject.kOverwrite)

    # 6. Export PNGs
    os.makedirs(PLOTS_DIR, exist_ok=True)
    c_rate_inc.SaveAs(os.path.join(PLOTS_DIR, "c_charm_rate_vs_energy.png"))
    c_rate_nu.SaveAs(os.path.join(PLOTS_DIR, "c_charm_rate_vs_energy_nu_only.png"))
    c_rate_antinu.SaveAs(os.path.join(PLOTS_DIR, "c_charm_rate_vs_energy_antinu_only.png"))
    c_dimuon_inc.SaveAs(os.path.join(PLOTS_DIR, "c_charm_dimuon_rate_vs_energy.png"))
    c_dimuon_nu.SaveAs(os.path.join(PLOTS_DIR, "c_charm_dimuon_rate_vs_energy_nu_only.png"))
    c_dimuon_antinu.SaveAs(os.path.join(PLOTS_DIR, "c_charm_dimuon_rate_vs_energy_antinu_only.png"))

    fin.Close()
    print("Canvas refresh complete! Updated ROOT file and exported PNGs.")

if __name__ == "__main__":
    main()
