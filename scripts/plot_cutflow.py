#!/usr/bin/env python3
"""
scripts/plot_cutflow.py
-----------------------
Visualizes and compares event selection cutflows across datasets:
Inclusive Neutrino, Direct Signal, Indirect Fake Signal, and Passing Muons.
Supports unweighted, raw_weight, and mc_weight weighting, with configurable
--nu-scaling and --pmu-scaling factors, and generates normalized copies (bin 1 = 1.0).

Usage:
  python3 scripts/plot_cutflow.py \\
    -a "cutflow_inclusive.root" \\
    -s "cutflow_signal.root" \\
    -p "cutflow_pmu2022.root" \\
    --nu-scaling 1.0 \\
    --pmu-scaling 1.0 \\
    -o "cutflow_comparison.root"
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, Dict, Any, List, Tuple
import ROOT

ROOT.gROOT.SetBatch(True)

from snd.cuts import draw_rotated_labels_on_canvas, make_normalized_histogram


def load_dataset_histograms(
    filepath: str,
    prefix_type: str = "all",
) -> Dict[str, ROOT.TH1D]:
    """
    Extract unweighted, raw_weight, and mc_weight cutflow histograms from a ROOT file.
    """
    if not os.path.exists(filepath):
        raise IOError(f"Cutflow file does not exist: {filepath}")

    f_in = ROOT.TFile.Open(filepath, "READ")
    if not f_in or f_in.IsZombie():
        raise IOError(f"Could not open cutflow file: {filepath}")

    candidate_keys = {
        "mc_weight": [
            f"h_cutflow_{prefix_type}_mc_weight",
            f"h_cutflow_{prefix_type}",
            "h_cutflow_mc_weight",
            "h_cutflow",
        ],
        "raw_weight": [
            f"h_cutflow_{prefix_type}_raw_weight",
            f"h_cutflow_{prefix_type}",
            "h_cutflow_raw_weight",
            "h_cutflow",
        ],
        "unweighted": [
            f"h_cutflow_{prefix_type}_unweighted",
            f"h_cutflow_{prefix_type}",
            "h_cutflow_unweighted",
            "h_cutflow",
        ],
    }

    result = {}
    for weight_type, keys in candidate_keys.items():
        found = None
        for k in keys:
            h = f_in.Get(k)
            if h and isinstance(h, ROOT.TH1):
                found = h.Clone(f"{prefix_type}_{weight_type}")
                found.SetDirectory(0)
                break
        if not found:
            # Fallback to generic h_cutflow
            h_gen = f_in.Get("h_cutflow")
            if h_gen and isinstance(h_gen, ROOT.TH1):
                found = h_gen.Clone(f"{prefix_type}_{weight_type}")
                found.SetDirectory(0)
        if not found:
            f_in.Close()
            raise ValueError(f"Could not find cutflow histogram for {weight_type} in {filepath}")
        result[weight_type] = found

    f_in.Close()
    return result


MINIMALISTIC_CUTS = [
    "All Events",
    "LHC IP1",
    "Stable Beams",
    "Event #Delta t",
    "SciFi-DS Timing",
    "No Veto Hits",
    "DS Dimuon",
    "2 DS Tracks",
]


def canonical_label(lbl: str) -> str:
    """Normalize cut labels across simulation and data cutflow variants."""
    l_low = lbl.strip().lower()
    if "all" in l_low and "event" in l_low:
        return "All Events"
    if "ip1" in l_low:
        return "LHC IP1"
    if "stable" in l_low:
        return "Stable Beams"
    if "delta" in l_low or "#delta" in l_low:
        return "Event #Delta t"
    if "timing" in l_low or "scifi-ds" in l_low:
        return "SciFi-DS Timing"
    if "veto" in l_low:
        return "No Veto Hits"
    if "dimuon" in l_low:
        return "DS Dimuon"
    if "track" in l_low:
        return "2 DS Tracks"
    return lbl.strip()


def build_comparison_canvas(
    dataset_entries: List[Dict[str, Any]],
    weight_key: str,
    canvas_name: str,
    title: str,
    y_title: str = "Events",
    is_norm: bool = False,
    cut_sequence: Optional[List[str]] = None,
) -> Tuple[ROOT.TCanvas, List[ROOT.TH1D]]:
    """
    Create a single TCanvas overlaying histograms for the specified weight_key.
    Uses THStack with 'nostack' so that all histograms strictly share the exact same
    coordinate frame, Y-axis, and X-axis.
    """
    cuts = cut_sequence if cut_sequence is not None else MINIMALISTIC_CUTS
    n_bins = len(cuts)

    c = ROOT.TCanvas(canvas_name, title, 1150, 750)
    c.SetGridx(1)
    c.SetGridy(1)
    c.SetLogy(1)
    c.SetBottomMargin(0.22)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.08)
    c.SetTopMargin(0.08)

    max_val = 1e-6
    min_val = 1e9

    hstack = ROOT.THStack(f"hstack_{canvas_name}", f"{title};;{y_title}")
    drawn_hists = []

    for d in dataset_entries:
        h_orig = d["hists"][weight_key]

        # Extract map of canonical labels to (content, error) from h_orig
        orig_map = {}
        for b in range(1, h_orig.GetNbinsX() + 1):
            lbl = canonical_label(h_orig.GetXaxis().GetBinLabel(b))
            orig_map[lbl] = (h_orig.GetBinContent(b), h_orig.GetBinError(b))

        # Initial event count (for data-only cuts fallback on MC)
        init_cnt, init_err = orig_map.get("All Events", (h_orig.GetBinContent(1), h_orig.GetBinError(1)))

        norm_suffix = "_norm" if is_norm else ""
        h_aligned = ROOT.TH1D(f"h_{canvas_name}_{d['key']}{norm_suffix}", f"{d['label']};;{y_title}", n_bins, 0.5, n_bins + 0.5)
        h_aligned.SetDirectory(0)
        h_aligned.SetLineColor(d["color"])
        h_aligned.SetLineWidth(3)
        h_aligned.SetMarkerColor(d["color"])
        h_aligned.SetMarkerStyle(d["marker"])
        h_aligned.SetMarkerSize(1.3)
        h_aligned.GetXaxis().SetLabelSize(0.0)

        for b, c_name in enumerate(cuts, start=1):
            h_aligned.GetXaxis().SetBinLabel(b, c_name)
            if c_name in orig_map:
                bc, be = orig_map[c_name]
            elif c_name in ("LHC IP1", "Stable Beams", "Event #Delta t"):
                # Data-only cuts evaluate to 100% on MC
                bc, be = init_cnt, init_err
            else:
                bc, be = 0.0, 0.0

            h_aligned.SetBinContent(b, bc)
            h_aligned.SetBinError(b, be)

        # If normalized canvas is requested, scale such that bin 1 = 1.0
        if is_norm:
            first_b = h_aligned.GetBinContent(1)
            if first_b > 0:
                h_aligned.Scale(1.0 / first_b)

        for b in range(1, n_bins + 1):
            bc = h_aligned.GetBinContent(b)
            if bc > 0:
                max_val = max(max_val, bc)
                min_val = min(min_val, bc)

        # Clear individual histogram stored limits so ROOT uses the THStack scale
        h_aligned.SetMinimum(-1111)
        h_aligned.SetMaximum(-1111)

        hstack.Add(h_aligned)
        drawn_hists.append((d, h_aligned))

    c.cd()
    hstack.Draw("nostack HIST")
    hstack.Draw("nostack E SAME")

    # Set unified range on the THStack (pad coordinate system)
    if is_norm:
        hstack.SetMaximum(3.0)
        hstack.SetMinimum(max(min_val * 0.1, 1e-5))
    else:
        hstack.SetMaximum(max_val * 15.0)
        hstack.SetMinimum(max(min_val * 0.1, 0.05))

    c.Update()
    if hstack.GetHistogram():
        hstack.GetHistogram().GetXaxis().SetLabelSize(0.0)
        hstack.GetHistogram().GetYaxis().SetTitle(y_title)
        hstack.GetHistogram().GetYaxis().SetTitleOffset(1.2)
        if is_norm:
            hstack.GetHistogram().GetYaxis().SetRangeUser(max(min_val * 0.1, 1e-5), 3.0)
        else:
            hstack.GetHistogram().GetYaxis().SetRangeUser(max(min_val * 0.1, 0.05), max_val * 15.0)

    leg = ROOT.TLegend(0.38, 0.67, 0.90, 0.89)
    leg.SetBorderSize(1)
    leg.SetFillStyle(1001)
    leg.SetFillColor(ROOT.kWhite)
    leg.SetTextSize(0.024)

    for d, h_draw in drawn_hists:
        init_val = h_draw.GetBinContent(1)
        if is_norm:
            leg.AddEntry(h_draw, d["label"], "lp")
        elif init_val >= 100:
            leg.AddEntry(h_draw, f"{d['label']} (Initial: {init_val:.1f})", "lp")
        else:
            leg.AddEntry(h_draw, f"{d['label']} (Initial: {init_val:.3f})", "lp")
    leg.Draw()

    # Draw 45-degree rotated labels on canvas
    ref_h = drawn_hists[0][1]
    draw_rotated_labels_on_canvas(c, ref_h, angle=45.0, text_size=0.028)

    c.Modified()
    c.Update()

    # Keep references on canvas to prevent Python garbage collection
    c._hstack_ref = hstack
    c._hists_ref = [h for _, h in drawn_hists]
    c._leg_ref = leg
    return c, [h for _, h in drawn_hists]


def plot_cutflow_comparison(
    all_filepath: Optional[str] = None,
    sig_filepath: Optional[str] = None,
    pmu_filepath: Optional[str] = None,
    indirect_filepath: Optional[str] = None,
    combined_filepath: Optional[str] = None,
    output_filepath: str = "cutflow_comparison.root",
    nu_scaling: float = 1.0,
    pmu_scaling: float = 1.0,
    title: str = "SND@LHC Selection Cutflow Comparison",
    all_label: str = "Inclusive Neutrino (2MuTrks)",
    sig_label: str = "Direct Signal (Charm #rightarrow #mu_{2})",
    indirect_label: str = "Indirect Fake Signal (Charm #rightarrow hadrons #rightarrow #mu_{2})",
    pmu_label: str = "Passing Muons (pmu2022, 68 fb^{-1})",
    save_images: bool = False,
) -> None:
    """
    Main orchestration function to load, scale, normalize, and draw comparison cutflows.
    """
    # 1. Resolve filepaths
    if combined_filepath:
        if not all_filepath:
            all_filepath = combined_filepath
        if not sig_filepath:
            sig_filepath = combined_filepath
        if not indirect_filepath and os.path.isfile(combined_filepath):
            f_chk = ROOT.TFile.Open(combined_filepath, "READ")
            if f_chk and not f_chk.IsZombie():
                if f_chk.Get("h_cutflow_indirect") or f_chk.Get("h_cutflow_indirect_mc_weight"):
                    indirect_filepath = combined_filepath
                f_chk.Close()

    dataset_entries: List[Dict[str, Any]] = []

    # Load All / Inclusive Neutrino
    if all_filepath:
        all_hists = load_dataset_histograms(all_filepath, prefix_type="all")
        if nu_scaling != 1.0:
            for h in all_hists.values():
                h.Scale(nu_scaling)
        dataset_entries.append({
            "key": "all",
            "label": all_label,
            "color": ROOT.kAzure + 2,
            "marker": 20,
            "hists": all_hists,
        })

    # Load Direct Signal Neutrino
    if sig_filepath:
        sig_hists = load_dataset_histograms(sig_filepath, prefix_type="signal")
        if nu_scaling != 1.0:
            for h in sig_hists.values():
                h.Scale(nu_scaling)
        dataset_entries.append({
            "key": "signal",
            "label": sig_label,
            "color": ROOT.kRed + 1,
            "marker": 21,
            "hists": sig_hists,
        })

    # Load Indirect Fake Signal Neutrino (optional)
    if indirect_filepath:
        ind_hists = load_dataset_histograms(indirect_filepath, prefix_type="indirect")
        if nu_scaling != 1.0:
            for h in ind_hists.values():
                h.Scale(nu_scaling)
        dataset_entries.append({
            "key": "indirect",
            "label": indirect_label,
            "color": ROOT.kGreen + 2,
            "marker": 22,
            "hists": ind_hists,
        })

    # Load Passing Muons (optional)
    if pmu_filepath:
        pmu_hists = load_dataset_histograms(pmu_filepath, prefix_type="pmu")
        if pmu_scaling != 1.0:
            for h in pmu_hists.values():
                h.Scale(pmu_scaling)
        dataset_entries.append({
            "key": "pmu",
            "label": pmu_label,
            "color": ROOT.kMagenta + 2,
            "marker": 33,
            "hists": pmu_hists,
        })

    if not dataset_entries:
        raise ValueError("No valid datasets specified for cutflow comparison.")

    # 4. Build Canvases
    canvases = {}

    # mc_weight
    c_mc, h_mc = build_comparison_canvas(
        dataset_entries,
        weight_key="mc_weight",
        canvas_name="c_cutflow_mc_weight",
        title=f"{title} (Weighted: mc_weight)",
        y_title="Weighted Events",
        is_norm=False,
    )
    canvases["c_cutflow_mc_weight"] = c_mc
    canvases["c_cutflow_comparison"] = c_mc  # primary alias

    # mc_weight normalized
    c_mc_norm, h_mc_norm = build_comparison_canvas(
        dataset_entries,
        weight_key="mc_weight",
        canvas_name="c_cutflow_mc_weight_norm",
        title=f"{title} (Normalized: mc_weight)",
        y_title="Fraction Passed",
        is_norm=True,
    )
    canvases["c_cutflow_mc_weight_norm"] = c_mc_norm
    canvases["c_cutflow_comparison_norm"] = c_mc_norm

    # raw_weight
    c_raw, h_raw = build_comparison_canvas(
        dataset_entries,
        weight_key="raw_weight",
        canvas_name="c_cutflow_raw_weight",
        title=f"{title} (Raw Weight: raw_weight)",
        y_title="Raw Weighted Events",
        is_norm=False,
    )
    canvases["c_cutflow_raw_weight"] = c_raw

    # raw_weight normalized
    c_raw_norm, h_raw_norm = build_comparison_canvas(
        dataset_entries,
        weight_key="raw_weight",
        canvas_name="c_cutflow_raw_weight_norm",
        title=f"{title} (Normalized: raw_weight)",
        y_title="Fraction Passed",
        is_norm=True,
    )
    canvases["c_cutflow_raw_weight_norm"] = c_raw_norm

    # unweighted
    c_unw, h_unw = build_comparison_canvas(
        dataset_entries,
        weight_key="unweighted",
        canvas_name="c_cutflow_unweighted",
        title=f"{title} (Unweighted Counts)",
        y_title="Events",
        is_norm=False,
    )
    canvases["c_cutflow_unweighted"] = c_unw

    # unweighted normalized
    c_unw_norm, h_unw_norm = build_comparison_canvas(
        dataset_entries,
        weight_key="unweighted",
        canvas_name="c_cutflow_unweighted_norm",
        title=f"{title} (Normalized: Unweighted)",
        y_title="Fraction Passed",
        is_norm=True,
    )
    canvases["c_cutflow_unweighted_norm"] = c_unw_norm

    # 5. Output handling
    out_dir = os.path.dirname(os.path.abspath(output_filepath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if output_filepath.endswith(".root"):
        f_out = ROOT.TFile.Open(output_filepath, "RECREATE")
        for cname, c in canvases.items():
            c.Write(cname)

        # Write individual aligned TH1D objects
        for h in (h_mc + h_mc_norm + h_raw + h_raw_norm + h_unw + h_unw_norm):
            h.Write()

        f_out.Close()
        print(f"[Done] Cutflow comparison canvases and histograms saved to ROOT file: {output_filepath}")

        # Save companion images (.png, .pdf) only if requested
        if save_images:
            base_img = output_filepath[:-5]
            for cname in ["c_cutflow_mc_weight", "c_cutflow_mc_weight_norm", "c_cutflow_raw_weight", "c_cutflow_unweighted"]:
                if cname in canvases:
                    canvases[cname].Print(f"{base_img}_{cname}.png")
                    canvases[cname].Print(f"{base_img}_{cname}.pdf")
            print(f"[Done] Cutflow companion images (.png, .pdf) saved to: {base_img}_*.png / .pdf")
    else:
        # Save images
        base, ext = os.path.splitext(output_filepath)
        canvases["c_cutflow_mc_weight"].SaveAs(output_filepath)
        canvases["c_cutflow_mc_weight_norm"].SaveAs(f"{base}_norm{ext}")
        canvases["c_cutflow_raw_weight"].SaveAs(f"{base}_raw_weight{ext}")
        canvases["c_cutflow_raw_weight_norm"].SaveAs(f"{base}_raw_weight_norm{ext}")
        canvases["c_cutflow_unweighted"].SaveAs(f"{base}_unweighted{ext}")
        canvases["c_cutflow_unweighted_norm"].SaveAs(f"{base}_unweighted_norm{ext}")
        print(f"[Done] Cutflow comparison images saved to: {base}*{ext}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay selection cutflows (Inclusive Neutrino vs Direct Signal vs Indirect vs Passing Muons).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--combined", default=None, help="Path to combined cutflow ROOT file containing both all-events and signal cutflows")
    parser.add_argument("-a", "--all", default=None, help="Path to all-events (inclusive neutrino) cutflow ROOT file")
    parser.add_argument("-s", "--signal", default=None, help="Path to direct signal cutflow ROOT file")
    parser.add_argument("-i", "--indirect", default=None, help="Path to indirect fake signal cutflow ROOT file")
    parser.add_argument("-p", "--pmu", default=None, help="Path to passing muon cutflow ROOT file (defaults to cutflow_pmu2022_essentials_68fb-1.root if present)")
    parser.add_argument("--no-pmu", action="store_true", default=False, help="Explicitly disable passing muon cutflow in comparison plots")
    parser.add_argument("-o", "--output", default="cutflow_comparison.root", help="Output file path (.root, .png, .pdf)")
    parser.add_argument("--nu-scaling", type=float, default=1.0, help="Scaling factor for neutrino cutflow histograms")
    parser.add_argument("--pmu-scaling", type=float, default=1.0, help="Scaling factor for passing muon cutflow histograms")
    parser.add_argument("--title", default="SND@LHC Selection Cutflow Comparison", help="Plot title")
    parser.add_argument("--all-label", default="Inclusive Neutrino (2MuTrks)", help="Legend label for inclusive neutrino")
    parser.add_argument("--sig-label", default="Direct Signal (Charm #rightarrow #mu_{2})", help="Legend label for direct signal")
    parser.add_argument("--indirect-label", default="Indirect Fake Signal (Charm #rightarrow hadrons #rightarrow #mu_{2})", help="Legend label for indirect fake signal")
    parser.add_argument("--pmu-label", default="Passing Muons (pmu2022, 68 fb^{-1})", help="Legend label for passing muons")
    parser.add_argument("--save-images", action="store_true", default=False, help="Save companion .png and .pdf images alongside the ROOT file")

    args = parser.parse_args()

    default_pmu_path = "/eos/user/i/idioniso/snd-numu-charm/cutflows/cutflow_pmu2022_essentials_68fb-1.root"
    if not args.pmu and not args.no_pmu and os.path.isfile(default_pmu_path):
        args.pmu = default_pmu_path
        print(f" ~ [Info] Automatically including passing muon cutflow: {args.pmu}")

    if not args.combined and not args.all and not args.pmu:
        parser.error("Must specify at least one cutflow dataset (-c/--combined, -a/--all, or -p/--pmu)")

    plot_cutflow_comparison(
        all_filepath=args.all,
        sig_filepath=args.signal,
        pmu_filepath=args.pmu,
        indirect_filepath=args.indirect,
        combined_filepath=args.combined,
        output_filepath=args.output,
        nu_scaling=args.nu_scaling,
        pmu_scaling=args.pmu_scaling,
        title=args.title,
        all_label=args.all_label,
        sig_label=args.sig_label,
        indirect_label=args.indirect_label,
        pmu_label=args.pmu_label,
        save_images=args.save_images,
    )


if __name__ == "__main__":
    main()
