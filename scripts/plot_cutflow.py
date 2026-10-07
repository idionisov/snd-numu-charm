#!/usr/bin/env python3
"""
scripts/plot_cutflow.py
-----------------------
Visualizes and compares event selection cutflows (All Events vs Signal) on a single canvas.
Can take two existing cutflow ROOT files (one for all events, one for signal),
or draw an existing comparison canvas from a combined cutflow ROOT file.

Usage:
  python3 scripts/plot_cutflow.py \\
    -a "cutflow_inclusive.root" \\
    -s "cutflow_signal.root" \\
    -o "cutflow_comparison.png"
"""

from __future__ import annotations

import os
import sys
import argparse
import ROOT

ROOT.gROOT.SetBatch(True)


def plot_cutflow_comparison(
    all_filepath: str,
    sig_filepath: str,
    output_filepath: str,
    title: str = "SND@LHC Selection Cutflow: All Events vs Signal",
    all_label: str = "All Events",
    sig_label: str = "Signal (#nu_{#mu} Charm #rightarrow #mu_{2} in DS)",
) -> None:
    """
    Load h_cutflow from both files and draw them on a single TCanvas.
    """
    f_all = ROOT.TFile.Open(all_filepath, "READ")
    if not f_all or f_all.IsZombie():
        raise IOError(f"Could not open all-events file: {all_filepath}")
    h_all = f_all.Get("h_cutflow")
    if not h_all:
        raise ValueError(f"'h_cutflow' not found in {all_filepath}")

    f_sig = ROOT.TFile.Open(sig_filepath, "READ")
    if not f_sig or f_sig.IsZombie():
        raise IOError(f"Could not open signal file: {sig_filepath}")
    h_sig = f_sig.Get("h_cutflow")
    if not h_sig:
        h_sig = f_sig.Get("h_cutflow_signal")
    if not h_sig:
        raise ValueError(f"Neither 'h_cutflow' nor 'h_cutflow_signal' found in {sig_filepath}")

    # Build canvas
    c = ROOT.TCanvas("c_cutflow_comparison", title, 1100, 750)
    c.SetGridx(1)
    c.SetGridy(1)
    c.SetLogy(1)
    c.SetBottomMargin(0.20)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.08)

    h_all_draw = h_all.Clone("h_all_draw")
    h_all_draw.SetDirectory(0)
    h_all_draw.SetTitle(f"{title};Cut Stage;Events")
    h_all_draw.SetLineColor(ROOT.kAzure + 2)
    h_all_draw.SetLineWidth(3)
    h_all_draw.SetMarkerColor(ROOT.kAzure + 2)
    h_all_draw.SetMarkerStyle(20)
    h_all_draw.SetMarkerSize(1.2)
    h_all_draw.GetXaxis().LabelsOption("v")
    h_all_draw.GetXaxis().SetLabelSize(0.035)

    h_sig_draw = h_sig.Clone("h_sig_draw")
    h_sig_draw.SetDirectory(0)
    h_sig_draw.SetLineColor(ROOT.kRed + 1)
    h_sig_draw.SetLineWidth(3)
    h_sig_draw.SetMarkerColor(ROOT.kRed + 1)
    h_sig_draw.SetMarkerStyle(21)
    h_sig_draw.SetMarkerSize(1.2)

    raw_all = h_all_draw.GetBinContent(1)
    raw_sig = h_sig_draw.GetBinContent(1)

    max_val = max(h_all_draw.GetMaximum(), h_sig_draw.GetMaximum(), 1.0)
    h_all_draw.SetMinimum(0.1)
    h_all_draw.SetMaximum(max_val * 8.0)

    h_all_draw.Draw("HIST")
    h_all_draw.Draw("E SAME")
    h_sig_draw.Draw("HIST SAME")
    h_sig_draw.Draw("E SAME")

    leg = ROOT.TLegend(0.52, 0.76, 0.89, 0.88)
    leg.SetBorderSize(1)
    leg.SetFillStyle(1001)
    leg.SetFillColor(ROOT.kWhite)
    leg.SetTextSize(0.03)
    leg.AddEntry(h_all_draw, f"{all_label} (Initial: {int(raw_all)})", "lp")
    leg.AddEntry(h_sig_draw, f"{sig_label} (Initial: {int(raw_sig)})", "lp")
    leg.Draw()

    out_dir = os.path.dirname(os.path.abspath(output_filepath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if output_filepath.endswith(".root"):
        f_out = ROOT.TFile.Open(output_filepath, "RECREATE")
        c.Write("c_cutflow_comparison")
        h_all_draw.Write("h_cutflow_all")
        h_sig_draw.Write("h_cutflow_signal")
        f_out.Close()
    else:
        c.SaveAs(output_filepath)

    f_all.Close()
    f_sig.Close()
    print(f"[Done] Cutflow comparison saved to: {output_filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay All Events vs Signal selection cutflow on a single canvas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--combined", default=None, help="Path to combined cutflow ROOT file containing both all-events and signal cutflow")
    parser.add_argument("-a", "--all", default=None, help="Path to all-events cutflow ROOT file")
    parser.add_argument("-s", "--signal", default=None, help="Path to signal cutflow ROOT file")
    parser.add_argument("-o", "--output", default="cutflow_comparison.png", help="Output file path (.root, .png, .pdf)")
    parser.add_argument("--all-label", default="All Events", help="Legend label for all events")
    parser.add_argument("--sig-label", default="Signal (#nu_{#mu} Charm #rightarrow #mu_{2} in DS)", help="Legend label for signal")
    args = parser.parse_args()

    if args.combined:
        all_path = args.combined
        sig_path = args.combined
    elif args.all and args.signal:
        all_path = args.all
        sig_path = args.signal
    else:
        parser.error("Must specify either -c/--combined <file.root> OR both -a/--all and -s/--signal")

    plot_cutflow_comparison(
        all_filepath=all_path,
        sig_filepath=sig_path,
        output_filepath=args.output,
        all_label=args.all_label,
        sig_label=args.sig_label,
    )


if __name__ == "__main__":
    main()
