#!/usr/bin/env python3
"""
scripts/qdc_calib/finalize_pmu_4models.py
-----------------------------------------
Finalizes the 4-model comparison for PMU vs Run 5408 Data:
1. Re-scales MC histograms to the exact Run 5408 luminosity (1.5307 pb^-1).
2. Populates Data histograms from the full 49,375 front-entering muons in cache_pmu_vs_run5408.npz.
3. Builds full ROOT canvases with styling, legends, and ratio panels.
4. Produces publication-quality multi-panel PNGs for:
   - Subsystems (Veto, US, DS)
   - Veto Planes (Planes 0 & 1)
   - US Planes (Planes 0 to 4)
   - DS Stations (DS-H 0..2, DS-V 0..3)
5. Saves to output/pmu_vs_run5408_4models.root and copies plots to brain.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

HIST_SPECS = {
    # Subsystems
    "veto_total": ("Veto Total Hits", "N_{hits}^{Veto}", 16, -0.5, 15.5),
    "us_total":   ("US Total Hits", "N_{hits}^{US}", 21, 4.5, 25.5),
    "ds_total":   ("DS Total Hits", "N_{hits}^{DS}", 31, -0.5, 30.5),
    # Veto Planes
    "veto_pl0":   ("Veto Plane 0 Hits", "N_{hits}^{Veto 0}", 11, -0.5, 10.5),
    "veto_pl1":   ("Veto Plane 1 Hits", "N_{hits}^{Veto 1}", 11, -0.5, 10.5),
    # US Planes
    "us_pl0":     ("US Plane 0 Hits", "N_{hits}^{US 0}", 10, 0.5, 10.5),
    "us_pl1":     ("US Plane 1 Hits", "N_{hits}^{US 1}", 10, 0.5, 10.5),
    "us_pl2":     ("US Plane 2 Hits", "N_{hits}^{US 2}", 10, 0.5, 10.5),
    "us_pl3":     ("US Plane 3 Hits", "N_{hits}^{US 3}", 10, 0.5, 10.5),
    "us_pl4":     ("US Plane 4 Hits", "N_{hits}^{US 4}", 10, 0.5, 10.5),
    # DS Horizontal Planes
    "ds_h_pl0":   ("DS Station 0 H Hits", "N_{hits}^{DS-H 0}", 11, -0.5, 10.5),
    "ds_h_pl1":   ("DS Station 1 H Hits", "N_{hits}^{DS-H 1}", 11, -0.5, 10.5),
    "ds_h_pl2":   ("DS Station 2 H Hits", "N_{hits}^{DS-H 2}", 11, -0.5, 10.5),
    # DS Vertical Planes
    "ds_v_pl0":   ("DS Station 0 V Hits", "N_{hits}^{DS-V 0}", 11, -0.5, 10.5),
    "ds_v_pl1":   ("DS Station 1 V Hits", "N_{hits}^{DS-V 1}", 11, -0.5, 10.5),
    "ds_v_pl2":   ("DS Station 2 V Hits", "N_{hits}^{DS-V 2}", 11, -0.5, 10.5),
    "ds_v_pl3":   ("DS Station 3 V Hits", "N_{hits}^{DS-V 3}", 11, -0.5, 10.5),
}

def main():
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"

    # 1. Load Data from cache
    cache_path = "plots/cache_pmu_vs_run5408.npz"
    print(f"Loading data from {cache_path}...")
    npz = np.load(cache_path, allow_pickle=True)
    data_dict = {k: npz[f"data_{k}"] for k in HIST_SPECS}
    n_data = len(data_dict["veto_total"])
    print(f"Loaded {n_data:,} collision muons from Run 5408.")

    # 2. Load existing MC histograms
    src_root = "output/pmu_vs_run5408_4models.root"
    f_src = ROOT.TFile.Open(src_root)
    if not f_src or f_src.IsZombie():
        print(f"Error: cannot open {src_root}")
        sys.exit(1)

    # The existing root file accidentally had MC histograms scaled twice by 3.737e7.
    # We rescale MC back down to the exact Run 5408 luminosity (1.5307 pb^-1)
    scale_factor = 7.56869430e-10 / 0.02828743804666667

    mc_hists = {m: {} for m in ["raw", "theo", "calib", "step"]}
    for k in HIST_SPECS:
        for m in ["raw", "theo", "calib", "step"]:
            h_orig = f_src.Get(f"h_mc_{m}_{k}")
            h_new = h_orig.Clone(f"h_mc_{m}_{k}_scaled")
            h_new.SetDirectory(0)
            h_new.Scale(scale_factor)
            mc_hists[m][k] = h_new

    f_src.Close()

    # 3. Create ROOT output file with scaled histograms and canvases
    dst_root = "output/pmu_vs_run5408_4models.root"
    f_dst = ROOT.TFile.Open(dst_root, "RECREATE")

    data_hists = {}
    for k in HIST_SPECS:
        title, xlabel, nbins, xmin, xmax = HIST_SPECS[k]
        h_d = ROOT.TH1D(f"h_data_{k}", f"{title};{xlabel};Events / bin", nbins, xmin, xmax)
        h_d.SetDirectory(f_dst)
        for v in data_dict[k]:
            h_d.Fill(float(v))
        h_d.SetLineColor(ROOT.kBlack)
        h_d.SetMarkerStyle(20)
        h_d.SetMarkerSize(0.8)
        data_hists[k] = h_d
        h_d.Write()

        for m in ["raw", "theo", "calib", "step"]:
            h = mc_hists[m][k]
            h.SetName(f"h_mc_{m}_{k}")
            h.SetDirectory(f_dst)
            h.Write()

    # Create and write canvases
    canvas_configs = [
        ("c_subsystems", "Subsystems Hit Multiplicity (4 Models)", ["veto_total", "us_total", "ds_total"], 3, 1),
        ("c_veto_planes", "Veto Planes Hit Multiplicity (4 Models)", ["veto_pl0", "veto_pl1"], 2, 1),
        ("c_us_planes", "US Planes Hit Multiplicity (4 Models)", ["us_pl0", "us_pl1", "us_pl2", "us_pl3", "us_pl4"], 3, 2),
        ("c_ds_planes", "DS Planes Hit Multiplicity (4 Models)", [
            "ds_h_pl0", "ds_v_pl0", "ds_h_pl1", "ds_v_pl1", "ds_h_pl2", "ds_v_pl2", "ds_v_pl3"
        ], 4, 2),
    ]

    for cname, ctitle, keys, nx, ny in canvas_configs:
        c = ROOT.TCanvas(cname, ctitle, 420 * nx, 360 * ny)
        c.Divide(nx, ny)
        c.store = []

        for p_idx, k in enumerate(keys):
            c.cd(p_idx + 1)
            ROOT.gPad.SetGrid()

            h_d = data_hists[k]
            h_raw = mc_hists["raw"][k]
            h_theo = mc_hists["theo"][k]
            h_calib = mc_hists["calib"][k]
            h_step = mc_hists["step"][k]

            h_raw.SetLineColor(ROOT.kBlue + 1)
            h_raw.SetLineStyle(3)
            h_raw.SetLineWidth(2)

            h_theo.SetLineColor(ROOT.kMagenta + 2)
            h_theo.SetLineStyle(2)
            h_theo.SetLineWidth(2)

            h_calib.SetLineColor(ROOT.kRed + 1)
            h_calib.SetLineWidth(2)

            h_step.SetLineColor(ROOT.kGreen + 2)
            h_step.SetLineStyle(4)
            h_step.SetLineWidth(2)

            max_y = max(h_d.GetMaximum(), h_raw.GetMaximum(), h_theo.GetMaximum(), h_calib.GetMaximum(), h_step.GetMaximum())
            h_d.SetMaximum(max_y * 1.35)

            h_d.Draw("E P")
            h_raw.Draw("HIST SAME")
            h_theo.Draw("HIST SAME")
            h_calib.Draw("HIST SAME")
            h_step.Draw("HIST SAME")

            leg = ROOT.TLegend(0.55, 0.65, 0.88, 0.88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.AddEntry(h_d, f"Run 5408 Data (N={n_data:,})", "ep")
            leg.AddEntry(h_raw, "Sim Raw", "l")
            leg.AddEntry(h_theo, "Sim Theo (k=1)", "l")
            leg.AddEntry(h_calib, "Sim Theo+k", "l")
            leg.AddEntry(h_step, "Sim Step (1/1.5 MeV)", "l")
            leg.Draw()

            c.store.extend([h_d, h_raw, h_theo, h_calib, h_step, leg])

        c.Write()

    # Extract bin contents into numpy arrays before closing f_dst
    mc_arrays = {m: {} for m in ["raw", "theo", "calib", "step"]}
    mc_errors = {m: {} for m in ["raw", "theo", "calib", "step"]}
    for k in HIST_SPECS:
        _, _, nbins, _, _ = HIST_SPECS[k]
        for m in ["raw", "theo", "calib", "step"]:
            h_m = mc_hists[m][k]
            mc_arrays[m][k] = np.array([h_m.GetBinContent(b) for b in range(1, nbins + 1)], dtype=float)
            mc_errors[m][k] = np.array([h_m.GetBinError(b) for b in range(1, nbins + 1)], dtype=float)

    f_dst.Close()
    print(f"Saved ROOT canvases and histograms to {dst_root}")

    # 4. Generate matplotlib PNG figures with ratio panels
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.0

    model_styles = {
        "raw":   {"label": "Raw (Eth=0)", "color": "#1f77b4", "ls": ":", "lw": 1.6, "marker": "s"},
        "theo":  {"label": "Theoretical (k=1.0)", "color": "#9467bd", "ls": "--", "lw": 1.8, "marker": "D"},
        "calib": {"label": "Theoretical + k(SiPM)", "color": "#d62728", "ls": "-", "lw": 2.0, "marker": "o"},
        "step":  {"label": "Simple Step (1 / 1.5 MeV)", "color": "#2ca02c", "ls": "-.", "lw": 1.8, "marker": "^"},
    }

    groups = [
        ("Subsystems (Veto, US, DS)", ["veto_total", "us_total", "ds_total"], "pmu_vs_run5408_subsystems_4models.png", 3),
        ("Veto Planes (Planes 0 & 1)", ["veto_pl0", "veto_pl1"], "pmu_vs_run5408_veto_planes_4models.png", 2),
        ("Upstream Calorimeter Planes (US 0 to 4)", ["us_pl0", "us_pl1", "us_pl2", "us_pl3", "us_pl4"], "pmu_vs_run5408_us_planes_4models.png", 3),
        ("Downstream Muon Filter Planes (DS Stations 0 to 3)", [
            "ds_h_pl0", "ds_v_pl0", "ds_h_pl1", "ds_v_pl1", "ds_h_pl2", "ds_v_pl2", "ds_v_pl3"
        ], "pmu_vs_run5408_ds_planes_4models.png", 4),
    ]

    for group_title, keys, fname, ncols in groups:
        nrows = (len(keys) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows * 2, ncols,
            figsize=(5.8 * ncols, 4.0 * nrows),
            gridspec_kw={"height_ratios": [3, 1.2] * nrows},
            constrained_layout=True,
        )

        if nrows == 1 and ncols == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        elif nrows == 1:
            axes = np.array([axes[0, :], axes[1, :]])
        elif ncols == 1:
            axes = axes.reshape(nrows * 2, 1)

        for idx, key in enumerate(keys):
            row = (idx // ncols) * 2
            col = idx % ncols
            ax_main = axes[row, col]
            ax_ratio = axes[row + 1, col]

            title, xlabel, nbins, xmin, xmax = HIST_SPECS[key]
            bins = np.linspace(xmin, xmax, nbins + 1)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # Data
            h_data, _ = np.histogram(data_dict[key], bins=bins)
            y_data = h_data.astype(float)
            y_data_err = np.sqrt(h_data)

            # MC models from histograms
            y_sim = {}
            y_sim_err = {}
            for m in ["raw", "theo", "calib", "step"]:
                y_sim[m] = mc_arrays[m][key]
                y_sim_err[m] = mc_errors[m][key]

                st = model_styles[m]
                ax_main.step(
                    bins, np.append(y_sim[m], y_sim[m][-1]), where="post",
                    color=st["color"], lw=st["lw"], ls=st["ls"], label=st["label"]
                )
                if m == "calib":
                    ax_main.fill_between(bins, np.append(y_sim[m], y_sim[m][-1]), step="post", color="#ff7f7f", alpha=0.18)

            # Plot Data
            mask_d = y_data > 0
            ax_main.errorbar(
                bin_centers[mask_d], y_data[mask_d], yerr=y_data_err[mask_d],
                fmt="o", color="black", ms=5.0, capsize=2, lw=1.2, label=f"Run 5408 Data (N={n_data:,})"
            )

            ax_main.set_title(title, fontsize=11, fontweight="bold")
            ax_main.set_ylabel(r"Events / bin ($\mathcal{L} = 1.53$ pb$^{-1}$)", fontsize=9)
            ax_main.set_xlim(xmin, xmax)
            max_val = max(np.max(y_data), max(np.max(y_sim[m]) for m in y_sim)) if len(y_data) > 0 else 1.0
            ax_main.set_ylim(0, max_val * 1.4 if max_val > 0 else 1.0)
            ax_main.grid(True, linestyle=":", alpha=0.6)
            ax_main.legend(loc="upper right", fontsize=7.2, framealpha=0.9)
            ax_main.xaxis.set_minor_locator(AutoMinorLocator())
            ax_main.yaxis.set_minor_locator(AutoMinorLocator())

            # Ratio panel
            ax_ratio.axhline(1.0, color="gray", linestyle="--", lw=1.0)
            offsets = {"raw": -0.15, "theo": -0.05, "calib": 0.05, "step": 0.15}

            for m in ["raw", "theo", "calib", "step"]:
                st = model_styles[m]
                mask_r = (y_data > 0) & (y_sim[m] > 0)
                if np.any(mask_r):
                    r_val = y_sim[m][mask_r] / y_data[mask_r]
                    r_err = r_val * np.sqrt((y_data_err[mask_r] / y_data[mask_r])**2 + (y_sim_err[m][mask_r] / y_sim[m][mask_r])**2)
                    ax_ratio.errorbar(
                        bin_centers[mask_r] + offsets[m] * (bin_centers[1] - bin_centers[0]),
                        r_val, yerr=r_err,
                        fmt=st["marker"], color=st["color"], mfc="none" if m == "raw" else st["color"],
                        ms=4.0, capsize=2, lw=1.0, label=f"{m} / Data"
                    )

            ax_ratio.set_xlim(xmin, xmax)
            ax_ratio.set_ylim(0.2, 2.2)
            ax_ratio.set_xlabel(xlabel, fontsize=10)
            ax_ratio.set_ylabel("Sim / Data", fontsize=9)
            ax_ratio.grid(True, linestyle=":", alpha=0.5)
            ax_ratio.xaxis.set_minor_locator(AutoMinorLocator())

        for idx in range(len(keys), nrows * ncols):
            row = (idx // ncols) * 2
            col = idx % ncols
            axes[row, col].set_visible(False)
            axes[row + 1, col].set_visible(False)

        out_png = os.path.join(out_dir, fname)
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"Saved figure: {out_png}")
        if os.path.exists(art_dir):
            os.system(f"cp {out_png} {os.path.join(art_dir, fname)}")

    print("\nAll PMU 4-model figures and ROOT canvases successfully produced.")

if __name__ == "__main__":
    main()
