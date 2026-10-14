#!/usr/bin/env python3
"""
scripts/diagnostics/plot_angular_comparisons.py
------------------------------------------------
Plots angular comparisons between Signal (direct charm decay: numu -> c -> mu)
and Signal-Like events (2 muons traversing DS / indirect / non-charm) across
four sequential analysis stages:

  Stage 1: All dimuon events (MC truth angles)
  Stage 2: Dimuon events in Downstream (DS) acceptance (MC truth angles)
  Stage 3: Dimuon events in DS acceptance with >= 2 reconstructed DS Hough tracks (MC truth angles)
  Stage 4: Dimuon events in DS acceptance with >= 2 reconstructed DS Hough tracks (Reconstructed track angles)

For each stage, generates three 2-panel comparison figures:
  - Figure A: theta_XZ(mu1) and theta_XZ(mu2) [-300, 300] mrad (log-scale)
  - Figure B: theta_YZ(mu1) and theta_YZ(mu2) [-300, 300] mrad (log-scale)
  - Figure C: Delta theta_XZ and Delta theta_YZ [0, 300] mrad (linear-scale)

Outputs are saved as ROOT TCanvases in a ROOT file and exported to high-resolution PNG and PDF.

Usage:
  python3 scripts/diagnostics/plot_angular_comparisons.py -i output_hists.root -o plots/angular_comparisons
  python3 scripts/diagnostics/plot_angular_comparisons.py --help
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetCanvasColor(0)
ROOT.gStyle.SetPadColor(0)
ROOT.gStyle.SetFrameFillColor(0)
ROOT.gStyle.SetFrameBorderMode(0)
ROOT.gStyle.SetPadBorderMode(0)


def setup_style():
    """Configures CERN ROOT styling for publication-quality comparison canvases."""
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetPalette(ROOT.kRainBow)
    ROOT.gStyle.SetCanvasColor(0)
    ROOT.gStyle.SetPadColor(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetPadLeftMargin(0.14)
    ROOT.gStyle.SetPadRightMargin(0.06)
    ROOT.gStyle.SetPadTopMargin(0.08)
    ROOT.gStyle.SetPadBottomMargin(0.14)


def find_histogram(f: ROOT.TFile, hname: str) -> Optional[ROOT.TH1]:
    """Recursively searches for a histogram by name in a ROOT TFile."""
    obj = f.Get(hname)
    if obj and isinstance(obj, ROOT.TH1):
        return obj

    def scan_dir(d):
        for k in d.GetListOfKeys():
            o = d.Get(k.GetName())
            if isinstance(o, ROOT.TDirectory):
                res = scan_dir(o)
                if res:
                    return res
            elif isinstance(o, ROOT.TH1) and k.GetName() == hname:
                return o
        return None

    return scan_dir(f)


def draw_two_panel_comparison(
    h_sig: ROOT.TH1,
    h_bkg: ROOT.TH1,
    h_sig_right: ROOT.TH1,
    h_bkg_right: ROOT.TH1,
    title_left_x: str,
    title_right_x: str,
    is_log_y: bool,
    y_min: float,
    y_max_left: float,
    y_max_right: float,
    stage_label: str,
    sig_label: str = "numu -> c -> mu",
    bkg_label: str = "Signal-Like (2 muons traversing DS)",
    out_prefix: str = "canvas",
    out_dir: str = "plots",
) -> ROOT.TCanvas:
    """
    Renders a 2-pad comparison canvas:
      - Left pad: Primary muon or Delta theta_XZ
      - Right pad: Secondary muon or Delta theta_YZ
    Histogram formatting:
      - Signal: Black solid line (HIST)
      - Background / Signal-Like: Red markers with error bars (E1)
    """
    os.makedirs(out_dir, exist_ok=True)
    cname = f"c_{out_prefix}"
    c = ROOT.TCanvas(cname, cname, 1300, 560)
    c.Divide(2, 1, 0.002, 0.002)

    pads = [c.cd(1), c.cd(2)]
    pairs = [
        (pads[0], h_sig, h_bkg, title_left_x, y_max_left),
        (pads[1], h_sig_right, h_bkg_right, title_right_x, y_max_right),
    ]

    for pad, hs, hb, xtitle, ymax in pairs:
        pad.cd()
        pad.SetLeftMargin(0.15)
        pad.SetRightMargin(0.06)
        pad.SetBottomMargin(0.15)
        pad.SetTopMargin(0.08)
        if is_log_y:
            pad.SetLogy(1)
        else:
            pad.SetLogy(0)

        # Style copies to avoid modifying originals
        hs_draw = hs.Clone(f"{hs.GetName()}_draw_{pad.GetName()}")
        hb_draw = hb.Clone(f"{hb.GetName()}_draw_{pad.GetName()}")

        # Axis properties
        hs_draw.GetXaxis().SetTitle(xtitle)
        hs_draw.GetXaxis().SetTitleSize(0.055)
        hs_draw.GetXaxis().SetTitleOffset(1.1)
        hs_draw.GetXaxis().SetLabelSize(0.05)

        hs_draw.GetYaxis().SetTitle("Events")
        hs_draw.GetYaxis().SetTitleSize(0.055)
        hs_draw.GetYaxis().SetTitleOffset(1.2)
        hs_draw.GetYaxis().SetLabelSize(0.05)

        # Limits
        effective_ymax = max(ymax, hs_draw.GetMaximum() * 1.5, hb_draw.GetMaximum() * 1.5)
        if is_log_y:
            hs_draw.SetMinimum(max(y_min, 0.5))
            hs_draw.SetMaximum(effective_ymax * 3.0)
        else:
            hs_draw.SetMinimum(0.0)
            hs_draw.SetMaximum(effective_ymax * 1.25)

        # Signal style (Black line)
        hs_draw.SetLineColor(ROOT.kBlack)
        hs_draw.SetLineWidth(2)
        hs_draw.SetLineStyle(1)

        # Signal-Like style (Red markers with error bars)
        hb_draw.SetLineColor(ROOT.kRed + 1)
        hb_draw.SetLineWidth(2)
        hb_draw.SetMarkerColor(ROOT.kRed + 1)
        hb_draw.SetMarkerStyle(20)
        hb_draw.SetMarkerSize(0.8)

        # Draw
        hs_draw.Draw("HIST")
        hb_draw.Draw("E1 SAME")

        # Text labels on pads
        lat = ROOT.TLatex()
        lat.SetNDC(True)
        lat.SetTextFont(42)
        lat.SetTextSize(0.048)

        # Red label (Signal-Like / Background)
        lat.SetTextColor(ROOT.kRed + 1)
        lat.SetTextAngle(22)
        lat.DrawLatex(0.20, 0.68, bkg_label)

        # Black label (Signal)
        lat.SetTextColor(ROOT.kBlack)
        lat.SetTextAngle(18)
        lat.DrawLatex(0.35, 0.22, sig_label)

        # Stage watermark
        lat.SetTextAngle(0)
        lat.SetTextSize(0.040)
        lat.SetTextColor(ROOT.kGray + 2)
        lat.DrawLatex(0.18, 0.86, stage_label)

    c.Update()
    c.SaveAs(os.path.join(out_dir, f"{out_prefix}.png"))
    c.SaveAs(os.path.join(out_dir, f"{out_prefix}.pdf"))
    return c


def run_angular_comparisons(
    input_file: str,
    output_dir: str,
    sig_key: str = "direct",
    bkg_key: str = "signal_like",
    sig_label: str = "numu -> c -> mu",
    bkg_label: str = "Signal-Like (2 muons traversing DS)",
):
    """Generates all 4 stages of angular comparison plots from the input ROOT file."""
    setup_style()
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/3] Opening input ROOT file: {input_file}")
    f = ROOT.TFile.Open(input_file, "READ")
    if not f or f.IsZombie():
        print(f"[Error] Could not open {input_file}")
        sys.exit(1)

    out_root_path = os.path.join(output_dir, "angular_comparisons_all_stages.root")
    f_out = ROOT.TFile.Open(out_root_path, "RECREATE")

    stages = [
        # (Stage index, Stage title, Sig pref, Bkg pref, is_reco)
        (
            1,
            "Stage 1: All Dimuon Events",
            sig_key,
            bkg_key,
            False,
            "stage1_all_dimuons",
        ),
        (
            2,
            "Stage 2: Dimuons in DS Acceptance",
            f"{sig_key}_in_acc",
            f"{bkg_key}_in_acc",
            False,
            "stage2_in_ds_acceptance",
        ),
        (
            3,
            "Stage 3: DS Acc + >=2 DS Tracks (MC Truth Angles)",
            f"{sig_key}_2ds_in_acc",
            f"{bkg_key}_2ds_in_acc",
            False,
            "stage3_in_acc_2ds_truth",
        ),
        (
            4,
            "Stage 4: DS Acc + >=2 DS Tracks (Reco Track Angles)",
            f"{sig_key}_2ds_in_acc",
            f"{bkg_key}_2ds_in_acc",
            True,
            "stage4_in_acc_2ds_reco",
        ),
    ]

    print(f"[2/3] Processing 4 comparison stages...")
    canvases: List[ROOT.TCanvas] = []

    for stage_idx, stage_title, spref, bpref, is_reco, tag in stages:
        print(f"\n--- Processing {stage_title} ({tag}) ---")
        st_dir = f_out.mkdir(tag)
        st_dir.cd()

        # Variable prefixes
        var_pfx = "reco_" if is_reco else ""
        angle_lbl = "#theta^{reco}" if is_reco else "#theta"
        delta_lbl = "#Delta#theta^{reco}" if is_reco else "#Delta#theta"

        # 1. Figure A: theta_XZ(mu1) and theta_XZ(mu2)
        h_s_xz1 = find_histogram(f, f"h_{spref}_{var_pfx}mu1_theta_xz_mrad")
        h_b_xz1 = find_histogram(f, f"h_{bpref}_{var_pfx}mu1_theta_xz_mrad")
        h_s_xz2 = find_histogram(f, f"h_{spref}_{var_pfx}sec_mu_theta_xz_mrad")
        h_b_xz2 = find_histogram(f, f"h_{bpref}_{var_pfx}sec_mu_theta_xz_mrad")

        if h_s_xz1 and h_b_xz1 and h_s_xz2 and h_b_xz2:
            cA = draw_two_panel_comparison(
                h_sig=h_s_xz1,
                h_bkg=h_b_xz1,
                h_sig_right=h_s_xz2,
                h_bkg_right=h_b_xz2,
                title_left_x=f"{angle_lbl}_{{XZ}}(#mu_{{1}}) [mrad]",
                title_right_x=f"{angle_lbl}_{{XZ}}(#mu_{{2}}) [mrad]",
                is_log_y=True,
                y_min=1.0,
                y_max_left=1000.0,
                y_max_right=100.0,
                stage_label=stage_title,
                sig_label=sig_label,
                bkg_label=bkg_label,
                out_prefix=f"{tag}_theta_xz",
                out_dir=output_dir,
            )
            st_dir.cd()
            cA.Write(f"c_{tag}_theta_xz")
            canvases.append(cA)
        else:
            print(f"  [Warning] Missing XZ angle histograms for {tag}")

        # 2. Figure B: theta_YZ(mu1) and theta_YZ(mu2)
        h_s_yz1 = find_histogram(f, f"h_{spref}_{var_pfx}mu1_theta_yz_mrad")
        h_b_yz1 = find_histogram(f, f"h_{bpref}_{var_pfx}mu1_theta_yz_mrad")
        h_s_yz2 = find_histogram(f, f"h_{spref}_{var_pfx}sec_mu_theta_yz_mrad")
        h_b_yz2 = find_histogram(f, f"h_{bpref}_{var_pfx}sec_mu_theta_yz_mrad")

        if h_s_yz1 and h_b_yz1 and h_s_yz2 and h_b_yz2:
            cB = draw_two_panel_comparison(
                h_sig=h_s_yz1,
                h_bkg=h_b_yz1,
                h_sig_right=h_s_yz2,
                h_bkg_right=h_b_yz2,
                title_left_x=f"{angle_lbl}_{{YZ}}(#mu_{{1}}) [mrad]",
                title_right_x=f"{angle_lbl}_{{YZ}}(#mu_{{2}}) [mrad]",
                is_log_y=True,
                y_min=1.0,
                y_max_left=1000.0,
                y_max_right=100.0,
                stage_label=stage_title,
                sig_label=sig_label,
                bkg_label=bkg_label,
                out_prefix=f"{tag}_theta_yz",
                out_dir=output_dir,
            )
            st_dir.cd()
            cB.Write(f"c_{tag}_theta_yz")
            canvases.append(cB)
        else:
            print(f"  [Warning] Missing YZ angle histograms for {tag}")

        # 3. Figure C: Delta theta_XZ and Delta theta_YZ
        h_s_dxz = find_histogram(f, f"h_{spref}_{var_pfx}dimuon_opening_angle_xz_mrad")
        h_b_dxz = find_histogram(f, f"h_{bpref}_{var_pfx}dimuon_opening_angle_xz_mrad")
        h_s_dyz = find_histogram(f, f"h_{spref}_{var_pfx}dimuon_opening_angle_yz_mrad")
        h_b_dyz = find_histogram(f, f"h_{bpref}_{var_pfx}dimuon_opening_angle_yz_mrad")

        if h_s_dxz and h_b_dxz and h_s_dyz and h_b_dyz:
            cC = draw_two_panel_comparison(
                h_sig=h_s_dxz,
                h_bkg=h_b_dxz,
                h_sig_right=h_s_dyz,
                h_bkg_right=h_b_dyz,
                title_left_x=f"{delta_lbl}_{{XZ}} [mrad]",
                title_right_x=f"{delta_lbl}_{{YZ}} [mrad]",
                is_log_y=False,
                y_min=0.0,
                y_max_left=30.0,
                y_max_right=30.0,
                stage_label=stage_title,
                sig_label=sig_label,
                bkg_label=bkg_label,
                out_prefix=f"{tag}_delta_theta",
                out_dir=output_dir,
            )
            st_dir.cd()
            cC.Write(f"c_{tag}_delta_theta")
            canvases.append(cC)
        else:
            print(f"  [Warning] Missing opening angle histograms for {tag}")

    f_out.Close()
    f.Close()
    print(f"\n[3/3] Successfully saved {len(canvases)} canvases to:")
    print(f"  - ROOT file : {out_root_path}")
    print(f"  - PNG & PDF : {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Plot 2-panel angular comparisons across 4 analysis stages.")
    parser.add_argument("-i", "--input", default="/eos/user/i/idioniso/snd-numu-charm/output_nu14_all_hists.root",
                        help="Input ROOT file containing truth and reco histograms")
    parser.add_argument("-o", "--output-dir", default="plots/angular_comparisons",
                        help="Output directory for canvases and figures")
    parser.add_argument("--sig-key", default="direct",
                        help="Histogram prefix for signal (default: direct)")
    parser.add_argument("--bkg-key", default="signal_like",
                        help="Histogram prefix for background/signal-like (default: signal_like)")
    parser.add_argument("--sig-label", default="numu -> c -> mu",
                        help="Text annotation for signal")
    parser.add_argument("--bkg-label", default="Signal-Like (2 muons traversing DS)",
                        help="Text annotation for background/signal-like")
    args = parser.parse_args()

    run_angular_comparisons(
        input_file=args.input,
        output_dir=args.output_dir,
        sig_key=args.sig_key,
        bkg_key=args.bkg_key,
        sig_label=args.sig_label,
        bkg_label=args.bkg_label,
    )


if __name__ == "__main__":
    main()
