#!/usr/bin/env python3
"""
Finalize Publication Plots and ROOT Canvases from Merged ROOT Histograms
========================================================================
Reads the merged ROOT histograms from parallel processing of the full PMU MC sample,
builds TEfficiencies, draws smoothed S-curves, and saves ROOT TCanvases with TLegends.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Finalize MC Threshold Plots from Merged ROOT File")
    parser.add_argument(
        "--root-file",
        type=str,
        default="plots/mc_threshold_study.root",
        help="Path to merged ROOT file",
    )
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory for plots")
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2",
        help="Artifact directory path",
    )
    return parser.parse_args()


def finalize():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reading merged ROOT file: {args.root_file}")
    f_root = ROOT.TFile.Open(args.root_file, "UPDATE")
    if not f_root or f_root.IsZombie():
        raise FileNotFoundError(f"Could not open {args.root_file}")

    cat_display_names = {
        "all_veto": "All Veto Bars",
        "all_us": "All US Bars",
        "us_012": "US0, US1, US2",
        "us_34": "US3, US4",
        "ds_ver": "All Vertical DS Bars",
        "ds_hor": "All Horizontal DS Bars",
    }

    dir_cats = f_root.Get("categories")
    dir_bars = f_root.Get("bars")

    # Re-evaluate category TEfficiencies and extract points for plotting
    cat_curves = {}
    for cat_name, disp_name in cat_display_names.items():
        h_pass = dir_cats.Get(f"h_pass_{cat_name}")
        h_tot = dir_cats.Get(f"h_tot_{cat_name}")
        if not h_pass or not h_tot:
            continue

        dir_cats.cd()
        if ROOT.TEfficiency.CheckConsistency(h_pass, h_tot):
            teff = ROOT.TEfficiency(h_pass, h_tot)
            teff.SetName(f"teff_{cat_name}")
            teff.SetTitle(f"Hit Efficiency S-Curve: {disp_name};#Delta E [keV];Efficiency")
            teff.Write("", ROOT.TObject.kOverwrite)

        # Extract data points
        n_bins = h_tot.GetNbinsX()
        x_pts, y_pts, y_errs = [], [], []
        for ib in range(1, n_bins + 1):
            tot = h_tot.GetBinContent(ib)
            pas = h_pass.GetBinContent(ib)
            cen = h_tot.GetBinCenter(ib)
            if tot > 0:
                r = pas / tot
                eff = r * 100.0
                tot_err = h_tot.GetBinError(ib)
                pas_err = h_pass.GetBinError(ib)
                var_r = (pas_err**2 * (1.0 - 2.0 * r) + (r**2) * (tot_err**2)) / (tot**2)
                err = np.sqrt(max(0.0, var_r)) * 100.0
                x_pts.append(cen)
                y_pts.append(eff)
                y_errs.append(err)
        cat_curves[cat_name] = (np.array(x_pts), np.array(y_pts), np.array(y_errs))

    # -------------------------------------------------------------
    # Matplotlib Plot 1: Subsystem Category S-Curves
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6.5))
    cat_styles = {
        "all_veto": {"color": "#1f77b4", "ls": "-", "marker": "o", "label": "All Veto Bars"},
        "all_us": {"color": "#2ca02c", "ls": "-", "marker": "s", "label": "All US Bars"},
        "us_012": {"color": "#17becf", "ls": "--", "marker": "^", "label": "US0, US1, US2 (Stations 0-2)"},
        "us_34": {"color": "#98df8a", "ls": "-.", "marker": "v", "label": "US3, US4 (Stations 3-4)"},
        "ds_ver": {"color": "#d62728", "ls": "-", "marker": "D", "label": "All Vertical DS Bars"},
        "ds_hor": {"color": "#ff7f0e", "ls": "-", "marker": "P", "label": "All Horizontal DS Bars"},
    }

    for cat_name, style in cat_styles.items():
        if cat_name not in cat_curves:
            continue
        xs, ys, yerrs = cat_curves[cat_name]
        plt.errorbar(
            xs, ys, yerr=yerrs,
            color=style["color"], linestyle=style["ls"], marker=style["marker"],
            label=style["label"], linewidth=2.0, markersize=5, capsize=2, alpha=0.9,
        )

    plt.title("MuFilter & Veto Hit Formation Efficiency S-Curves by Subsystem Category\n(High Statistics PMU Simulation Traversals Forming Hits vs. Deposited Energy)", fontsize=12, weight="bold")
    plt.xlabel("Sum of Clustered MCPoint Energy Depositions $\\Delta E$ in Bar [keV]", fontsize=11)
    plt.ylabel("Hit Formation Efficiency [%]", fontsize=11)
    plt.axhline(50.0, color="gray", linestyle=":", label="50% Threshold")
    plt.axhline(99.0, color="black", linestyle="--", alpha=0.7, label="99% Full Turn-On")
    plt.ylim(-5, 105)
    plt.xlim(0, 450)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=9.5, loc="lower right")
    plt.tight_layout()

    p_scurve = os.path.join(args.outdir, "mc_subsystem_efficiency_scurves.png")
    plt.savefig(p_scurve, dpi=200)
    plt.close()
    print(f"Saved smoothed plot: {p_scurve}")

    # -------------------------------------------------------------
    # Matplotlib Plot 2: Bar-by-Bar S-Curves
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bar-Level Hit Formation Efficiency Turn-On S-Curves for Each Bar in Detector\n(Sum of Clustered MCPoint Energy Depositions vs. Hit Formation Fraction)", fontsize=13, weight="bold")

    bar_sub_map = {
        1: (axes[0, 0], "Veto Bars (14 bars)", plt.cm.Blues, "all_veto"),
        2: (axes[0, 1], "Upstream Bars (50 bars, US0-US4)", plt.cm.Greens, "all_us"),
        30: (axes[1, 0], "DS Horizontal Bars (240 bars)", plt.cm.Oranges, "ds_hor"),
        31: (axes[1, 1], "DS Vertical Bars (180 bars)", plt.cm.Reds, "ds_ver"),
    }

    if dir_bars:
        for k in dir_bars.GetListOfKeys():
            kname = k.GetName()
            if not kname.startswith("h_tot_bar_"):
                continue
            det_id = int(kname.replace("h_tot_bar_", ""))
            h_tot_b = dir_bars.Get(kname)
            h_pas_b = dir_bars.Get(f"h_pass_bar_{det_id}")
            if not h_tot_b or not h_pas_b or h_tot_b.GetEntries() < 10:
                continue

            subsystem = det_id // 10000
            bar = det_id % 1000
            target_key = subsystem if subsystem in (1, 2) else (30 if bar < 60 else 31)
            ax, _, cmap, _ = bar_sub_map[target_key]

            b_xs, b_ys = [], []
            for ib in range(1, h_tot_b.GetNbinsX() + 1):
                tot = h_tot_b.GetBinContent(ib)
                pas = h_pas_b.GetBinContent(ib)
                if tot > 0:
                    b_xs.append(h_tot_b.GetBinCenter(ib))
                    b_ys.append((pas / tot) * 100.0)

            c_val = 0.35 + 0.60 * ((det_id % 50) / 50.0)
            ax.plot(b_xs, b_ys, "-", color=cmap(c_val), alpha=0.35, linewidth=1.0)

    for target_key, (ax, title, cmap, cat_key) in bar_sub_map.items():
        if cat_key in cat_curves:
            xs, ys, _ = cat_curves[cat_key]
            ax.plot(xs, ys, "k-", linewidth=2.5, label="Subsystem Average")

        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("Sum of Clustered Energy Depositions $\\Delta E$ [keV]", fontsize=10)
        ax.set_ylabel("Hit Formation Efficiency [%]", fontsize=10)
        ax.axhline(50.0, color="gray", linestyle=":", alpha=0.7)
        ax.axhline(99.0, color="black", linestyle="--", alpha=0.5)
        ax.set_ylim(-5, 105)
        ax.set_xlim(0, 450)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9.5, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p_bars = os.path.join(args.outdir, "mc_bar_level_efficiency_curves.png")
    plt.savefig(p_bars, dpi=200)
    plt.close()
    print(f"Saved smoothed bar curves: {p_bars}")

    # -------------------------------------------------------------
    # Matplotlib Plot 3: 2D Threshold vs Position
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Minimum Turn-On Energy Threshold $\\Delta E_{\\rm min}$ vs. Hit Position Along Scintillator Bar\n(Spatial Variation Due to Attenuation & Coupling)", fontsize=13, weight="bold")

    pos_configs = [
        ("us", "US Bar: $\\Delta E_{\\rm min}$ vs Distance to Gel SiPM", axes[0, 0], "Greens", "Distance to Gel Coupling End $d_{\\rm gel}$ [cm]", (0, 85), (100, 250)),
        ("dsh", "DS Horizontal: $\\Delta E_{\\rm min}$ vs Impact Position $x$", axes[0, 1], "Oranges", "Horizontal Impact Position $x$ [cm]", (-42, 42), (30, 80)),
        ("dsv", "DS Vertical: $\\Delta E_{\\rm min}$ vs Distance to Top SiPM", axes[1, 0], "Reds", "Distance from Top Readout End $d_{\\rm top}$ [cm]", (0, 65), (20, 60)),
        ("veto", "Veto Bar: $\\Delta E_{\\rm min}$ vs Impact Position $x$ (SND@LHC Geometry)", axes[1, 1], "Blues", "Horizontal Impact Position $x$ [cm]", (-65, 0), (90, 180)),
    ]

    for sub_k, title, ax, cmap_name, xlabel, xlim, ylim in pos_configs:
        d_sub = f_root.Get(sub_k)
        if not d_sub:
            continue
        h2 = d_sub.Get(f"h2_eth_vs_pos_{sub_k}")
        if not h2:
            continue

        nx = h2.GetNbinsX()
        ny = h2.GetNbinsY()
        x_edges = np.array([h2.GetXaxis().GetBinLowEdge(i) for i in range(1, nx + 2)])
        y_edges = np.array([h2.GetYaxis().GetBinLowEdge(j) for j in range(1, ny + 2)])
        z_vals = np.zeros((ny, nx))
        for ix in range(1, nx + 1):
            for iy in range(1, ny + 1):
                z_vals[iy - 1, ix - 1] = h2.GetBinContent(ix, iy)

        mesh = ax.pcolormesh(x_edges, y_edges, z_vals, cmap=cmap_name, shading="auto")
        cb = plt.colorbar(mesh, ax=ax)
        cb.set_label("Traversals", fontsize=9)

        # Profile
        prof = h2.ProfileX()
        p_x, p_y = [], []
        for ib in range(1, prof.GetNbinsX() + 1):
            if prof.GetBinEntries(ib) >= 5:
                p_x.append(prof.GetBinCenter(ib))
                p_y.append(prof.GetBinContent(ib))
        if p_x:
            ax.plot(p_x, p_y, "k.-", linewidth=2.0, markersize=7, label="Mean Profile")

        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Minimum Turn-On Energy $\\Delta E_{\\rm min}$ [keV]", fontsize=10)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p_pos = os.path.join(args.outdir, "mc_threshold_vs_position.png")
    plt.savefig(p_pos, dpi=200)
    plt.close()
    print(f"Saved threshold vs position plot: {p_pos}")

    # -------------------------------------------------------------
    # Rebuild ROOT TCanvases and Write to canvases/ directory
    # -------------------------------------------------------------
    f_root.cd()
    dir_cvs = f_root.Get("canvases")
    if not dir_cvs:
        dir_cvs = f_root.mkdir("canvases")
    dir_cvs.cd()

    # ROOT Canvas 1: Subsystem S-curves
    c_cat = ROOT.TCanvas("c_mc_subsystem_efficiency_scurves", "Subsystem Hit Efficiency S-Curves", 1000, 700)
    c_cat.SetGrid()
    c_cat.SetLeftMargin(0.12)
    c_cat.SetBottomMargin(0.12)

    mg_cat = ROOT.TMultiGraph("mg_categories", "MuFilter & Veto Hit Efficiency S-Curves;#Delta E in Bar [keV];Hit Qualification Efficiency [%]")
    leg_cat = ROOT.TLegend(0.60, 0.16, 0.88, 0.44)
    leg_cat.SetBorderSize(1)
    leg_cat.SetFillColor(ROOT.kWhite)
    leg_cat.SetTextSize(0.028)

    root_cat_colors = {
        "all_veto": (ROOT.kBlue + 1, 20),
        "all_us": (ROOT.kGreen + 2, 21),
        "us_012": (ROOT.kCyan + 2, 22),
        "us_34": (ROOT.kSpring + 4, 23),
        "ds_ver": (ROOT.kRed + 1, 33),
        "ds_hor": (ROOT.kOrange + 1, 34),
    }

    draw_keep = []
    for cat_name, (color_code, marker_style) in root_cat_colors.items():
        if cat_name not in cat_curves:
            continue
        xs, ys, yerrs = cat_curves[cat_name]
        gr = ROOT.TGraphErrors()
        for idx, (x_val, y_val, err_val) in enumerate(zip(xs, ys, yerrs)):
            gr.SetPoint(idx, float(x_val), float(y_val))
            gr.SetPointError(idx, 0.0, float(err_val))
        gr.SetLineColor(color_code)
        gr.SetLineWidth(2)
        gr.SetMarkerColor(color_code)
        gr.SetMarkerStyle(marker_style)
        gr.SetMarkerSize(1.0)
        mg_cat.Add(gr, "lp")
        leg_cat.AddEntry(gr, cat_display_names[cat_name], "lp")
        draw_keep.append(gr)

    mg_cat.Draw("a")
    mg_cat.GetXaxis().SetLimits(0, 450)
    mg_cat.SetMinimum(-5)
    mg_cat.SetMaximum(105)
    mg_cat.GetXaxis().SetTitle("Sum of Clustered MCPoint Energy Depositions #Delta E in Bar [keV]")
    mg_cat.GetYaxis().SetTitle("Hit Formation Efficiency [%]")

    line50 = ROOT.TLine(0, 50, 450, 50)
    line50.SetLineColor(ROOT.kGray + 2)
    line50.SetLineStyle(3)
    line50.Draw("same")
    leg_cat.AddEntry(line50, "50% Threshold", "l")

    line99 = ROOT.TLine(0, 99, 450, 99)
    line99.SetLineColor(ROOT.kBlack)
    line99.SetLineStyle(2)
    line99.Draw("same")
    leg_cat.AddEntry(line99, "99% Full Turn-On", "l")

    leg_cat.Draw()
    c_cat.Write("", ROOT.TObject.kOverwrite)
    draw_keep.extend([mg_cat, leg_cat, line50, line99])

    # ROOT Canvas 2: Threshold vs Position
    c_pos = ROOT.TCanvas("c_mc_threshold_vs_position", "Turn-On Threshold vs Position", 1200, 900)
    c_pos.Divide(2, 2)
    for pad_i, sub_k in enumerate(["us", "dsh", "dsv", "veto"], start=1):
        c_pos.cd(pad_i)
        ROOT.gPad.SetGrid()
        d_sub = f_root.Get(sub_k)
        if d_sub:
            h2 = d_sub.Get(f"h2_eth_vs_pos_{sub_k}")
            if h2:
                h2.Draw("COLZ")
    c_pos.Write("", ROOT.TObject.kOverwrite)

    # ROOT Canvas 3: Bar-Level Hit Efficiency S-Curves (4 pads)
    c_bars = ROOT.TCanvas("c_mc_bar_level_efficiency_curves", "Bar-by-Bar Efficiency S-Curves", 1300, 900)
    c_bars.Divide(2, 2)
    mg_b1 = ROOT.TMultiGraph("mg_bars_veto", "Veto Bars (14 bars);#Delta E in Bar [keV];Efficiency [%]")
    mg_b2 = ROOT.TMultiGraph("mg_bars_us", "Upstream Bars (50 bars);#Delta E in Bar [keV];Efficiency [%]")
    mg_b3 = ROOT.TMultiGraph("mg_bars_dsh", "DS Horizontal Bars (240 bars);#Delta E in Bar [keV];Efficiency [%]")
    mg_b4 = ROOT.TMultiGraph("mg_bars_dsv", "DS Vertical Bars (180 bars);#Delta E in Bar [keV];Efficiency [%]")

    mg_map = {1: mg_b1, 2: mg_b2, 30: mg_b3, 31: mg_b4}

    if dir_bars:
        for k in dir_bars.GetListOfKeys():
            kname = k.GetName()
            if not kname.startswith("h_tot_bar_"):
                continue
            det_id = int(kname.replace("h_tot_bar_", ""))
            h_tot_b = dir_bars.Get(kname)
            h_pas_b = dir_bars.Get(f"h_pass_bar_{det_id}")
            if not h_tot_b or not h_pas_b or h_tot_b.GetEntries() < 10:
                continue

            subsystem = det_id // 10000
            bar = det_id % 1000
            target_key = subsystem if subsystem in (1, 2) else (30 if bar < 60 else 31)
            target_mg = mg_map[target_key]

            gr_b = ROOT.TGraph()
            pt_idx = 0
            for ib in range(1, h_tot_b.GetNbinsX() + 1):
                tot = h_tot_b.GetBinContent(ib)
                pas = h_pas_b.GetBinContent(ib)
                if tot > 0:
                    gr_b.SetPoint(pt_idx, float(h_tot_b.GetBinCenter(ib)), float((pas / tot) * 100.0))
                    pt_idx += 1
            if pt_idx >= 2:
                gr_b.SetLineColor(ROOT.kGray + 1)
                gr_b.SetLineWidth(1)
                target_mg.Add(gr_b, "l")
                draw_keep.append(gr_b)

    # Add subsystem average curves to each pad
    cat_pad_map = [(1, mg_b1, "all_veto", ROOT.kBlue + 1), (2, mg_b2, "all_us", ROOT.kGreen + 2), (3, mg_b3, "ds_hor", ROOT.kOrange + 1), (4, mg_b4, "ds_ver", ROOT.kRed + 1)]
    for pad_idx, target_mg, cat_k, color_code in cat_pad_map:
        c_bars.cd(pad_idx)
        ROOT.gPad.SetGrid()
        if cat_k in cat_curves:
            xs, ys, _ = cat_curves[cat_k]
            gr_avg = ROOT.TGraph()
            for idx, (x_val, y_val) in enumerate(zip(xs, ys)):
                gr_avg.SetPoint(idx, float(x_val), float(y_val))
            gr_avg.SetLineColor(color_code)
            gr_avg.SetLineWidth(3)
            target_mg.Add(gr_avg, "l")
            draw_keep.append(gr_avg)

        target_mg.Draw("a")
        target_mg.GetXaxis().SetLimits(0, 450)
        target_mg.SetMinimum(-5)
        target_mg.SetMaximum(105)
        target_mg.GetXaxis().SetTitle("#Delta E in Bar [keV]")
        target_mg.GetYaxis().SetTitle("Efficiency [%]")
        draw_keep.append(target_mg)

    c_bars.Write("", ROOT.TObject.kOverwrite)

    f_root.Close()
    print("Finalized ROOT file with updated TCanvases successfully.")

    # Copy updated plots to artifact directory
    if args.artifact_dir:
        import shutil
        for p in [p_scurve, p_bars, p_pos, args.root_file]:
            dest = os.path.join(args.artifact_dir, os.path.basename(p))
            shutil.copyfile(p, dest)
            print(f"Copied to artifact directory: {dest}")


if __name__ == "__main__":
    finalize()
