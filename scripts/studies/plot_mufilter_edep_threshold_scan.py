#!/usr/bin/env python3
"""
Plot number of US hits, DS hits, and Veto hits as a function of Edep threshold (0 to 20 MeV in 0.25 MeV steps).
Uses newP8 charm dimuon Monte Carlo events.
"""

import os
import sys
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
import ROOT

ROOT.gROOT.SetBatch(True)


def load_events(max_dirs: int = 10, cache_path: str = "/tmp/cache_newP8_edep_events.pkl"):
    """Load Veto, US, DS energy deposit arrays across newP8 folders."""
    if os.path.exists(cache_path):
        print(f"Loading cached events from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Reading newP8 simulation events from first {max_dirs} folders...")
    t0 = time.time()
    base_dir = "/eos/user/i/idioniso/snd-numu-charm/data/newP8"

    all_veto = []
    all_us = []
    all_ds = []
    total_events = 0

    for idx in range(1, max_dirs + 1):
        fpath = os.path.join(base_dir, str(idx), "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root")
        if not os.path.exists(fpath):
            continue
        try:
            f = uproot.open(fpath)
            if "cbmsim" not in f:
                continue
            tree = f["cbmsim"]
            dids = tree["Digi_MuFilterHits/Digi_MuFilterHits.fDetectorID"].array()

            if "Edep_MuFilterHit" in tree.keys():
                edeps = tree["Edep_MuFilterHit"].array()
            else:
                pt_did = tree["MuFilterPoint/MuFilterPoint.fDetectorID"].array()
                pt_elo = tree["MuFilterPoint/MuFilterPoint.fELoss"].array() * 1000.0  # MeV
                edeps = []
                for ev_hdid, ev_pdid, ev_pelo in zip(dids, pt_did, pt_elo):
                    m = {}
                    for d, e in zip(ev_pdid, ev_pelo):
                        m[d] = m.get(d, 0.0) + e
                    edeps.append(np.array([m.get(d, 0.0) for d in ev_hdid]))

            for ev_d, ev_e in zip(dids, edeps):
                subs = np.array(ev_d) // 10000
                e = np.array(ev_e)
                all_veto.append(e[subs == 1])
                all_us.append(e[subs == 2])
                all_ds.append(e[subs == 3])
                total_events += 1

        except Exception as exc:
            print(f"Warning: could not process {fpath}: {exc}", file=sys.stderr)

    print(f"Loaded {total_events} events in {time.time()-t0:.2f} s")
    data = {"veto": all_veto, "us": all_us, "ds": all_ds}
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    return data


def main():
    outdir = "/afs/cern.ch/work/i/idioniso/snd-numu-charm/plots/mufilter_study"
    artifact_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    data = load_events(max_dirs=10)
    veto_events = data["veto"]
    us_events = data["us"]
    ds_events = data["ds"]
    n_events = len(us_events)
    print(f"Analyzing {n_events} events...")

    # Threshold grid: 0 to 20 MeV in 0.25 MeV steps
    thresholds = np.arange(0.0, 20.25, 0.25)

    veto_means = []
    us_means = []
    ds_means = []

    veto_stds = []
    us_stds = []
    ds_stds = []

    for eth in thresholds:
        # Count hits >= eth per event
        v_c = np.array([np.sum(ev >= eth) for ev in veto_events])
        u_c = np.array([np.sum(ev >= eth) for ev in us_events])
        d_c = np.array([np.sum(ev >= eth) for ev in ds_events])

        veto_means.append(np.mean(v_c))
        us_means.append(np.mean(u_c))
        ds_means.append(np.mean(d_c))

        veto_stds.append(np.std(v_c) / np.sqrt(n_events))
        us_stds.append(np.std(u_c) / np.sqrt(n_events))
        ds_stds.append(np.std(d_c) / np.sqrt(n_events))

    veto_means = np.array(veto_means)
    us_means = np.array(us_means)
    ds_means = np.array(ds_means)

    # -------------------------------------------------------------------------
    # Multi-Panel Figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    plt.subplots_adjust(wspace=0.25, hspace=0.30)

    # Panel (0,0): Absolute Multiplicity vs Edep Threshold
    ax = axes[0, 0]
    ax.plot(thresholds, ds_means, color="#d62728", lw=2.8, label=f"Downstream (DS): Raw={ds_means[0]:.1f}")
    ax.plot(thresholds, us_means, color="#1f77b4", lw=2.8, label=f"Upstream (US): Raw={us_means[0]:.1f}")
    ax.plot(thresholds, veto_means, color="#2ca02c", lw=2.8, label=f"Veto: Raw={veto_means[0]:.1f}")

    # Mark calibrated operating points
    idx_ds_cal = int(np.argmin(np.abs(thresholds - 1.50)))
    idx_us_cal = int(np.argmin(np.abs(thresholds - 5.00)))
    ax.plot([1.50], [ds_means[idx_ds_cal]], "o", color="#d62728", markersize=9, markeredgecolor="black", zorder=5)
    ax.plot([5.00], [us_means[idx_us_cal]], "o", color="#1f77b4", markersize=9, markeredgecolor="black", zorder=5)

    ax.annotate(f"DS (1.5 MeV):\n{ds_means[idx_ds_cal]:.1f} hits ({ds_means[idx_ds_cal]/ds_means[0]*100:.1f}%)",
                xy=(1.50, ds_means[idx_ds_cal]), xytext=(3.0, ds_means[idx_ds_cal] + 6),
                arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="#d62728", alpha=0.9),
                fontsize=9.0, fontweight="bold")

    ax.annotate(f"US (5.0 MeV):\n{us_means[idx_us_cal]:.1f} hits ({us_means[idx_us_cal]/us_means[0]*100:.1f}%)",
                xy=(5.00, us_means[idx_us_cal]), xytext=(7.5, us_means[idx_us_cal] + 6),
                arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#e3f2fd", edgecolor="#1f77b4", alpha=0.9),
                fontsize=9.0, fontweight="bold")

    ax.set_title("Mean Hit Multiplicity vs $E_{\\rm dep}$ Threshold", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Energy Deposit Threshold $E_{\rm dep}$ [MeV]", fontsize=11)
    ax.set_ylabel(r"$\langle N_{\rm hits}\rangle$ per event", fontsize=11)
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(0.0, max(ds_means) * 1.08)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel (0,1): Normalized Hit Survival Fraction
    ax = axes[0, 1]
    ax.plot(thresholds, ds_means / ds_means[0] * 100.0, color="#d62728", lw=2.8, label="Downstream (DS)")
    ax.plot(thresholds, us_means / us_means[0] * 100.0, color="#1f77b4", lw=2.8, label="Upstream (US)")
    ax.plot(thresholds, veto_means / veto_means[0] * 100.0, color="#2ca02c", lw=2.8, label="Veto")

    ax.axhline(50.0, color="#7f7f7f", ls=":", lw=1.5, label="50% Survival Level")
    ax.axvline(1.50, color="#d62728", ls="--", lw=1.5, alpha=0.7)
    ax.axvline(5.00, color="#1f77b4", ls="--", lw=1.5, alpha=0.7)

    ax.set_title("Hit Survival Fraction vs $E_{\\rm dep}$ Threshold", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Energy Deposit Threshold $E_{\rm dep}$ [MeV]", fontsize=11)
    ax.set_ylabel(r"Remaining Hits Fraction [\%]", fontsize=11)
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(0.0, 105.0)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel (1,0): Log-Scale Multiplicity (Revealing Soft Tail vs Core)
    ax = axes[1, 0]
    ax.plot(thresholds, ds_means, color="#d62728", lw=2.8, label="Downstream (DS)")
    ax.plot(thresholds, us_means, color="#1f77b4", lw=2.8, label="Upstream (US)")
    ax.plot(thresholds, veto_means, color="#2ca02c", lw=2.8, label="Veto")

    ax.set_yscale("log")
    ax.set_title("Hit Multiplicity vs Threshold (Log Scale)", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Energy Deposit Threshold $E_{\rm dep}$ [MeV]", fontsize=11)
    ax.set_ylabel(r"$\langle N_{\rm hits}\rangle$ per event [Log Scale]", fontsize=11)
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(bottom=0.2)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    # Panel (1,1): Summary Table at Benchmark Thresholds
    ax = axes[1, 1]
    ax.axis("off")

    bench_ths = [0.0, 0.5, 1.0, 1.5, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
    table_data = [["Threshold", "Veto Hits (Rel %)", "US Hits (Rel %)", "DS Hits (Rel %)"]]
    for th in bench_ths:
        idx = int(np.argmin(np.abs(thresholds - th)))
        v_val = f"{veto_means[idx]:.2f} ({veto_means[idx]/veto_means[0]*100:.1f}%)"
        u_val = f"{us_means[idx]:.2f} ({us_means[idx]/us_means[0]*100:.1f}%)"
        d_val = f"{ds_means[idx]:.2f} ({ds_means[idx]/ds_means[0]*100:.1f}%)"
        th_label = f"{th:.2f} MeV"
        if th == 1.50:
            th_label += " (DS*)"
        elif th == 5.00:
            th_label += " (US*)"
        table_data.append([th_label, v_val, u_val, d_val])

    table = ax.table(cellText=table_data, colWidths=[0.24, 0.25, 0.25, 0.26], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.7)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif "DS*" in table_data[row][0]:
            cell.set_facecolor("#ffebee")
            cell.get_text().set_weight("bold")
        elif "US*" in table_data[row][0]:
            cell.set_facecolor("#e3f2fd")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f8f9fa" if row % 2 == 0 else "#ffffff")

    ax.set_title("Benchmark Threshold Summary (newP8 Charm Events)", fontsize=12, fontweight="bold", pad=15)

    # Save PNG and PDF
    png_path = os.path.join(outdir, "mufilter_hits_vs_edep_threshold.png")
    pdf_path = os.path.join(outdir, "mufilter_hits_vs_edep_threshold.pdf")
    art_path = os.path.join(artifact_dir, "mufilter_hits_vs_edep_threshold.png")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(art_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path} and {art_path}")

    # -------------------------------------------------------------------------
    # Save ROOT Graphs & Canvas to Master File
    # -------------------------------------------------------------------------
    root_path = os.path.join(outdir, "mufilter_hits_vs_edep.root")
    f_root = ROOT.TFile.Open(root_path, "RECREATE")

    n_pts = len(thresholds)
    g_veto = ROOT.TGraphErrors(n_pts, thresholds.astype(np.float64), veto_means.astype(np.float64),
                               np.zeros(n_pts, dtype=np.float64), np.array(veto_stds, dtype=np.float64))
    g_veto.SetName("g_veto_hits_vs_edep")
    g_veto.SetTitle("Veto Hits vs Edep Threshold;E_{dep} Threshold [MeV];<N_{hits}> per event")
    g_veto.SetLineColor(ROOT.kGreen + 2)
    g_veto.SetLineWidth(2)
    g_veto.Write("g_veto_hits_vs_edep", ROOT.TObject.kOverwrite)

    g_us = ROOT.TGraphErrors(n_pts, thresholds.astype(np.float64), us_means.astype(np.float64),
                             np.zeros(n_pts, dtype=np.float64), np.array(us_stds, dtype=np.float64))
    g_us.SetName("g_us_hits_vs_edep")
    g_us.SetTitle("US Hits vs Edep Threshold;E_{dep} Threshold [MeV];<N_{hits}> per event")
    g_us.SetLineColor(ROOT.kBlue)
    g_us.SetLineWidth(2)
    g_us.Write("g_us_hits_vs_edep", ROOT.TObject.kOverwrite)

    g_ds = ROOT.TGraphErrors(n_pts, thresholds.astype(np.float64), ds_means.astype(np.float64),
                             np.zeros(n_pts, dtype=np.float64), np.array(ds_stds, dtype=np.float64))
    g_ds.SetName("g_ds_hits_vs_edep")
    g_ds.SetTitle("DS Hits vs Edep Threshold;E_{dep} Threshold [MeV];<N_{hits}> per event")
    g_ds.SetLineColor(ROOT.kRed)
    g_ds.SetLineWidth(2)
    g_ds.Write("g_ds_hits_vs_edep", ROOT.TObject.kOverwrite)

    c = ROOT.TCanvas("c_hits_vs_edep_threshold", "MuFilter Hit Multiplicity vs Edep Threshold", 1200, 800)
    c.SetGrid()
    g_ds.Draw("ALP")
    g_us.Draw("LP SAME")
    g_veto.Draw("LP SAME")
    c.BuildLegend(0.65, 0.70, 0.88, 0.88)
    c.Write("c_hits_vs_edep_threshold", ROOT.TObject.kOverwrite)

    f_root.Close()
    print(f"Canvas c_hits_vs_edep_threshold and graphs written to {root_path}")


if __name__ == "__main__":
    main()
