#!/usr/bin/env python3
"""
scripts/studies/lowest_qdc.py

Processes converted raw data from SND@LHC runs in parallel to extract:
  - Lowest possible QDC value for individual channels and hits (Veto, US, DS),
    including negative QDC values (excluding unhit placeholder -999).
  - Both all-inclusive (with negative QDC) and positive-only minimums and percentiles.
  - Saves all QDC distributions (channel & bar, zoom & full range) to a ROOT file.
  - Generates companion publication-ready plots.

Usage:
  python3 scripts/studies/lowest_qdc.py \
    --run-dir /eos/experiment/sndlhc/convertedData/physics/2023/run_006640 \
    --workers 8 \
    --output-root /afs/cern.ch/work/i/idioniso/snd-numu-charm/plots/qdc_distributions_run_006640.root \
    --output-png /afs/cern.ch/work/i/idioniso/snd-numu-charm/plots/qdc_distributions_run_006640.png
"""

import os
import sys
import glob
import time
import json
import argparse
from multiprocessing import Pool
import numpy as np

# ROOT must be imported inside worker or properly guarded
import ROOT

# Pre-compile C++ scanner inside ROOT
CPP_SCANNER = """
#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"
#include "MuFilterHit.h"
#include <vector>
#include <algorithm>

struct FileQDCResult {
    double min_chan_all[4];       // lowest QDC including negative (q > -800)
    double min_chan_pos[4];       // lowest positive QDC (q > 0)
    double min_bar_all[4];        // lowest bar sum QDC including negative
    double min_bar_pos[4];        // lowest bar sum QDC (sum > 0)
    long long n_channels_all[4];
    long long n_channels_pos[4];
    long long n_channels_neg[4];
    long long n_hits_all[4];
    long long n_hits_pos[4];
    
    // Histograms
    // 1. Zoomed channel: 400 bins in [-20, +20), width = 0.1
    long long hist_chan_zoom[4][400];
    // 2. Full channel: 500 bins in [-100, +400), width = 1.0
    long long hist_chan_full[4][500];
    // 3. Full bar sum: 600 bins in [-100, +500), width = 1.0
    long long hist_bar_full[4][600];
    
    bool success;
};

FileQDCResult scan_root_file(const char* fname, long long max_entries=-1, bool allow_masked=true) {
    FileQDCResult res;
    res.success = false;
    for (int s = 0; s < 4; ++s) {
        res.min_chan_all[s] = 1e9;
        res.min_chan_pos[s] = 1e9;
        res.min_bar_all[s] = 1e9;
        res.min_bar_pos[s] = 1e9;
        res.n_channels_all[s] = 0;
        res.n_channels_pos[s] = 0;
        res.n_channels_neg[s] = 0;
        res.n_hits_all[s] = 0;
        res.n_hits_pos[s] = 0;
        for (int b = 0; b < 400; ++b) res.hist_chan_zoom[s][b] = 0;
        for (int b = 0; b < 500; ++b) res.hist_chan_full[s][b] = 0;
        for (int b = 0; b < 600; ++b) res.hist_bar_full[s][b] = 0;
    }

    TFile* f = TFile::Open(fname, "READ");
    if (!f || f->IsZombie()) {
        if (f) delete f;
        return res;
    }
    TTree* t = (TTree*)f->Get("rawConv");
    if (!t) {
        t = (TTree*)f->Get("cbmsim");
    }
    if (!t) {
        f->Close();
        delete f;
        return res;
    }

    t->SetBranchStatus("*", 0);
    t->SetBranchStatus("Digi_MuFilterHits*", 1);
    TClonesArray* hits = nullptr;
    t->SetBranchAddress("Digi_MuFilterHits", &hits);

    long long n = t->GetEntries();
    if (max_entries > 0 && max_entries < n) n = max_entries;

    for (long long i = 0; i < n; ++i) {
        t->GetEntry(i);
        if (!hits) continue;
        int nHits = hits->GetEntries();
        for (int h = 0; h < nHits; ++h) {
            MuFilterHit* hit = (MuFilterHit*)hits->At(h);
            if (!hit || !hit->isValid()) continue;
            int sys = hit->GetSystem();
            if (sys < 1 || sys > 3) continue;

            double sum_qdc = 0;
            bool has_chan = false;
            int nSiPMs = hit->GetnSiPMs();
            int nSides = hit->GetnSides();
            int total_channels = nSiPMs * nSides;
            if (total_channels > 16) total_channels = 16;

            for (int c = 0; c < total_channels; ++c) {
                if (!allow_masked && hit->isMasked(c)) continue;
                double q = hit->GetSignal(c);
                // Exclude empty channel sentinel (-999)
                if (q <= -800.0) continue;

                has_chan = true;
                sum_qdc += q;
                res.n_channels_all[sys]++;
                if (q < res.min_chan_all[sys]) res.min_chan_all[sys] = q;

                if (q > 0.0) {
                    res.n_channels_pos[sys]++;
                    if (q < res.min_chan_pos[sys]) res.min_chan_pos[sys] = q;
                } else {
                    res.n_channels_neg[sys]++;
                }

                // 1. Zoomed: [-20, 20) with width 0.1 -> bin = (q + 20) * 10
                if (q >= -20.0 && q < 20.0) {
                    int bin = (int)((q + 20.0) * 10.0);
                    if (bin >= 0 && bin < 400) res.hist_chan_zoom[sys][bin]++;
                }
                // 2. Full: [-100, 400) with width 1.0 -> bin = q + 100
                if (q >= -100.0 && q < 400.0) {
                    int bin = (int)(q + 100.0);
                    if (bin >= 0 && bin < 500) res.hist_chan_full[sys][bin]++;
                }
            }

            if (has_chan) {
                res.n_hits_all[sys]++;
                if (sum_qdc < res.min_bar_all[sys]) res.min_bar_all[sys] = sum_qdc;

                if (sum_qdc > 0.0) {
                    res.n_hits_pos[sys]++;
                    if (sum_qdc < res.min_bar_pos[sys]) res.min_bar_pos[sys] = sum_qdc;
                }

                // 3. Bar sum full: [-100, 500) with width 1.0
                if (sum_qdc >= -100.0 && sum_qdc < 500.0) {
                    int bin = (int)(sum_qdc + 100.0);
                    if (bin >= 0 && bin < 600) res.hist_bar_full[sys][bin]++;
                }
            }
        }
    }

    f->Close();
    delete f;
    res.success = true;
    return res;
}
"""

def init_worker():
    """Initialize ROOT and load shared libraries inside each worker process."""
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    ROOT.gSystem.Load("libshipLHC")
    ROOT.gInterpreter.Declare(CPP_SCANNER)

def worker_task(args):
    """Worker task to process a single ROOT file."""
    fname, max_events, allow_masked = args
    try:
        raw_res = ROOT.scan_root_file(fname, max_events, allow_masked)
        if not raw_res.success:
            return {"file": fname, "success": False}

        result = {
            "file": fname,
            "success": True,
            "min_chan_all": [float(raw_res.min_chan_all[s]) for s in range(4)],
            "min_chan_pos": [float(raw_res.min_chan_pos[s]) for s in range(4)],
            "min_bar_all": [float(raw_res.min_bar_all[s]) for s in range(4)],
            "min_bar_pos": [float(raw_res.min_bar_pos[s]) for s in range(4)],
            "n_channels_all": [int(raw_res.n_channels_all[s]) for s in range(4)],
            "n_channels_pos": [int(raw_res.n_channels_pos[s]) for s in range(4)],
            "n_channels_neg": [int(raw_res.n_channels_neg[s]) for s in range(4)],
            "n_hits_all": [int(raw_res.n_hits_all[s]) for s in range(4)],
            "n_hits_pos": [int(raw_res.n_hits_pos[s]) for s in range(4)],
            "hist_chan_zoom": [[int(raw_res.hist_chan_zoom[s][b]) for b in range(400)] for s in range(4)],
            "hist_chan_full": [[int(raw_res.hist_chan_full[s][b]) for b in range(500)] for s in range(4)],
            "hist_bar_full": [[int(raw_res.hist_bar_full[s][b]) for b in range(600)] for s in range(4)],
        }
        return result
    except Exception as e:
        return {"file": fname, "success": False, "error": str(e)}

def compute_percentiles(hist_counts, total_count, min_val, bin_width, percentiles=[0.1, 1.0, 5.0, 10.0, 50.0]):
    """Compute approximate percentiles from a binned histogram."""
    if total_count == 0 or np.sum(hist_counts) == 0:
        return {p: 0.0 for p in percentiles}
    cdf = np.cumsum(hist_counts) / np.sum(hist_counts) * 100.0
    results = {}
    for p in percentiles:
        idx = np.searchsorted(cdf, p)
        if idx < len(hist_counts):
            results[p] = round(min_val + (idx + 0.5) * bin_width, 3)
        else:
            results[p] = round(min_val + len(hist_counts) * bin_width, 3)
    return results

def save_to_root_file(output_path, hist_chan_zoom, hist_chan_full, hist_bar_full):
    """Export all accumulated histograms into an organized ROOT file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fout = ROOT.TFile.Open(output_path, "RECREATE")
    sys_tags = {1: "veto", 2: "us", 3: "ds"}
    sys_titles = {1: "Veto", 2: "Upstream (US)", 3: "Downstream (DS)"}

    for s in [1, 2, 3]:
        tag = sys_tags[s]
        title = sys_titles[s]

        # 1. Zoomed Channel Histogram: 400 bins in [-20, 20], width 0.1
        h_zoom = ROOT.TH1D(f"h_qdc_chan_zoom_{tag}", f"{title} Channel QDC [-20 to +20];Channel QDC;Entries / 0.1 counts", 400, -20, 20)
        for b in range(400):
            h_zoom.SetBinContent(b + 1, hist_chan_zoom[s][b])
        h_zoom.Write()

        # 2. Full Channel Histogram: 500 bins in [-100, 400], width 1.0
        h_full = ROOT.TH1D(f"h_qdc_chan_full_{tag}", f"{title} Channel QDC [-100 to +400];Channel QDC;Entries / 1 count", 500, -100, 400)
        for b in range(500):
            h_full.SetBinContent(b + 1, hist_chan_full[s][b])
        h_full.Write()

        # 3. Full Bar Histogram: 600 bins in [-100, 500], width 1.0
        h_bar = ROOT.TH1D(f"h_qdc_bar_full_{tag}", f"{title} Bar Sum QDC [-100 to +500];Bar Sum QDC;Entries / 1 count", 600, -100, 500)
        for b in range(600):
            h_bar.SetBinContent(b + 1, hist_bar_full[s][b])
        h_bar.Write()

    fout.Close()
    print(f"Exported ROOT histograms successfully to: {output_path}")

def plot_distributions(png_path, hist_chan_zoom, hist_chan_full, hist_bar_full):
    """Plot QDC distributions across subsystems using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import ddfUtils.mpl_styling as dms
    except ImportError:
        import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    labels = {1: "Veto", 2: "US", 3: "DS"}

    # 1. Zoomed Channel QDC (-20 to +20)
    ax0 = axes[0]
    bins_zoom = np.linspace(-20, 20, 401)
    centers_zoom = 0.5 * (bins_zoom[:-1] + bins_zoom[1:])
    for s in [1, 2, 3]:
        ax0.plot(centers_zoom, hist_chan_zoom[s], label=labels[s], color=colors[s], lw=1.5)
    ax0.axvline(0, color="gray", ls=":", alpha=0.7)
    ax0.set_xlabel("Channel QDC")
    ax0.set_ylabel("Entries / 0.1 counts")
    ax0.set_title("Channel QDC Zoom [-20 to +20] (incl. negative)")
    ax0.set_yscale("log")
    ax0.grid(True, which="both", ls="--", alpha=0.4)
    ax0.legend()

    # 2. Full Range Channel (-50 to +200)
    ax1 = axes[1]
    bins_full = np.linspace(-100, 400, 501)
    centers_full = 0.5 * (bins_full[:-1] + bins_full[1:])
    # Slice to [-50, 200]
    mask_full = (centers_full >= -50) & (centers_full <= 200)
    for s in [1, 2, 3]:
        ax1.plot(centers_full[mask_full], hist_chan_full[s][mask_full], label=labels[s], color=colors[s], lw=1.5)
    ax1.axvline(0, color="gray", ls=":", alpha=0.7)
    ax1.set_xlabel("Channel QDC")
    ax1.set_ylabel("Entries / 1.0 count")
    ax1.set_title("Channel QDC Distribution [-50 to +200]")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend()

    # 3. Full Range Bar Sum (-50 to +300)
    ax2 = axes[2]
    bins_bar = np.linspace(-100, 500, 601)
    centers_bar = 0.5 * (bins_bar[:-1] + bins_bar[1:])
    mask_bar = (centers_bar >= -50) & (centers_bar <= 300)
    for s in [1, 2, 3]:
        ax2.plot(centers_bar[mask_bar], hist_bar_full[s][mask_bar], label=labels[s], color=colors[s], lw=1.5)
    ax2.axvline(0, color="gray", ls=":", alpha=0.7)
    ax2.set_xlabel("Bar Sum QDC")
    ax2.set_ylabel("Entries / 1.0 count")
    ax2.set_title("Bar Reconstructed Sum QDC [-50 to +300]")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Generated QDC distribution summary plot: {png_path}")

def main():
    parser = argparse.ArgumentParser(description="Find lowest QDC values (including negative) and export distributions to ROOT.")
    parser.add_argument("--run-dir", default="/eos/experiment/sndlhc/convertedData/physics/2023/run_006640",
                        help="Directory containing sndsw_raw-*.root files")
    parser.add_argument("--pattern", default="sndsw_raw-*.root", help="File pattern to match")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument("--max-files", type=int, default=-1, help="Max files to process (-1 for all)")
    parser.add_argument("--max-events-per-file", type=int, default=-1, help="Max events per file (-1 for all)")
    parser.add_argument("--allow-masked", action="store_true", default=True, help="Include masked channels")
    parser.add_argument("--output-json", default="lowest_qdc_run_006640.json", help="Path to save summary JSON")
    parser.add_argument("--output-root", default="/afs/cern.ch/work/i/idioniso/snd-numu-charm/plots/qdc_distributions_run_006640.root",
                        help="Output ROOT file for QDC histograms")
    parser.add_argument("--output-png", default="/afs/cern.ch/work/i/idioniso/snd-numu-charm/plots/qdc_distributions_run_006640.png",
                        help="Output PNG figure for visual inspection")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.run_dir, args.pattern)))
    if not files:
        print(f"Error: No files found matching {os.path.join(args.run_dir, args.pattern)}")
        sys.exit(1)

    if args.max_files > 0:
        files = files[:args.max_files]

    total_files = len(files)
    print(f"=== Starting QDC Scan (Allowing Negative Values) for Run: {args.run_dir} ===")
    print(f"Total files to process: {total_files}")
    print(f"Workers: {args.workers}, Max events per file: {args.max_events_per_file}")

    t_start = time.time()
    task_args = [(f, args.max_events_per_file, args.allow_masked) for f in files]

    # Global accumulators
    global_min_chan_all = [1e9] * 4
    global_min_chan_pos = [1e9] * 4
    global_min_bar_all = [1e9] * 4
    global_min_bar_pos = [1e9] * 4
    global_n_chan_all = [0] * 4
    global_n_chan_pos = [0] * 4
    global_n_chan_neg = [0] * 4
    global_n_hits_all = [0] * 4
    global_n_hits_pos = [0] * 4
    global_hist_chan_zoom = np.zeros((4, 400), dtype=np.int64)
    global_hist_chan_full = np.zeros((4, 500), dtype=np.int64)
    global_hist_bar_full = np.zeros((4, 600), dtype=np.int64)

    successful_files = 0
    with Pool(processes=args.workers, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(worker_task, task_args), start=1):
            if not res["success"]:
                continue
            successful_files += 1
            for s in range(1, 4):
                if res["min_chan_all"][s] < global_min_chan_all[s]:
                    global_min_chan_all[s] = res["min_chan_all"][s]
                if res["min_chan_pos"][s] < global_min_chan_pos[s]:
                    global_min_chan_pos[s] = res["min_chan_pos"][s]
                if res["min_bar_all"][s] < global_min_bar_all[s]:
                    global_min_bar_all[s] = res["min_bar_all"][s]
                if res["min_bar_pos"][s] < global_min_bar_pos[s]:
                    global_min_bar_pos[s] = res["min_bar_pos"][s]

                global_n_chan_all[s] += res["n_channels_all"][s]
                global_n_chan_pos[s] += res["n_channels_pos"][s]
                global_n_chan_neg[s] += res["n_channels_neg"][s]
                global_n_hits_all[s] += res["n_hits_all"][s]
                global_n_hits_pos[s] += res["n_hits_pos"][s]

                global_hist_chan_zoom[s] += np.array(res["hist_chan_zoom"][s], dtype=np.int64)
                global_hist_chan_full[s] += np.array(res["hist_chan_full"][s], dtype=np.int64)
                global_hist_bar_full[s] += np.array(res["hist_bar_full"][s], dtype=np.int64)

            if i % 10 == 0 or i == total_files:
                elapsed = time.time() - t_start
                rate = i / elapsed
                rem = (total_files - i) / rate if rate > 0 else 0
                print(f"[{i}/{total_files}] ({i/total_files*100:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | Rate: {rate:.2f} files/s | ETA: {rem:.1f}s",
                      flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nScan completed in {elapsed_total:.2f} seconds across {successful_files}/{total_files} files.\n")

    sys_names = {1: "Veto", 2: "US (Upstream)", 3: "DS (Downstream)"}
    summary_data = {
        "run_dir": args.run_dir,
        "files_processed": successful_files,
        "elapsed_seconds": round(elapsed_total, 2),
        "subsystems": {}
    }

    print("=" * 105)
    print(f"{'Subsystem':<15} | {'Min Chan (All)':<14} | {'Min Chan (Pos)':<14} | {'Min Bar (All)':<13} | {'Neg Frac (%)':<12} | {'Median QDC':<10} | {'Total Chans':<12}")
    print("-" * 105)

    for s in [1, 2, 3]:
        pcts_chan_zoom = compute_percentiles(global_hist_chan_zoom[s], global_n_chan_all[s], min_val=-20.0, bin_width=0.1)
        pcts_bar_full = compute_percentiles(global_hist_bar_full[s], global_n_hits_all[s], min_val=-100.0, bin_width=1.0)

        neg_frac = (global_n_chan_neg[s] / global_n_chan_all[s] * 100.0) if global_n_chan_all[s] > 0 else 0.0

        summary_data["subsystems"][sys_names[s]] = {
            "min_channel_all": round(global_min_chan_all[s], 4) if global_min_chan_all[s] < 1e8 else None,
            "min_channel_pos": round(global_min_chan_pos[s], 4) if global_min_chan_pos[s] < 1e8 else None,
            "min_bar_all": round(global_min_bar_all[s], 4) if global_min_bar_all[s] < 1e8 else None,
            "min_bar_pos": round(global_min_bar_pos[s], 4) if global_min_bar_pos[s] < 1e8 else None,
            "total_channels_all": int(global_n_chan_all[s]),
            "total_channels_pos": int(global_n_chan_pos[s]),
            "total_channels_neg": int(global_n_chan_neg[s]),
            "negative_fraction_pct": round(neg_frac, 2),
            "total_hits_all": int(global_n_hits_all[s]),
            "percentiles_channel": pcts_chan_zoom,
            "percentiles_bar": pcts_bar_full,
        }

        min_all_str = f"{global_min_chan_all[s]:.4f}" if global_min_chan_all[s] < 1e8 else "N/A"
        min_pos_str = f"{global_min_chan_pos[s]:.4f}" if global_min_chan_pos[s] < 1e8 else "N/A"
        min_bar_str = f"{global_min_bar_all[s]:.4f}" if global_min_bar_all[s] < 1e8 else "N/A"
        median_str = f"{pcts_chan_zoom.get(50.0, 0.0):.2f}"

        print(f"{sys_names[s]:<15} | {min_all_str:<14} | {min_pos_str:<14} | {min_bar_str:<13} | {neg_frac:<12.2f} | {median_str:<10} | {global_n_chan_all[s]:<12}")

    print("=" * 105)

    # Save output json
    with open(args.output_json, "w") as jf:
        json.dump(summary_data, jf, indent=2)
    print(f"Detailed summary JSON written to: {args.output_json}")

    # Export to ROOT file
    save_to_root_file(args.output_root, global_hist_chan_zoom, global_hist_chan_full, global_hist_bar_full)

    # Generate companion plot
    plot_distributions(args.output_png, global_hist_chan_zoom, global_hist_chan_full, global_hist_bar_full)

if __name__ == "__main__":
    main()
