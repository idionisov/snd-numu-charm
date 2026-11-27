#!/usr/bin/env python3
"""
Study MCPoint Turn-On Energy Thresholds and Digitization Efficiency
===================================================================
Investigates the turn-on energy thresholds and efficiency of MuFilter and Veto
plastic scintillator bars using Geant4 MCPoints and TOFPET2 discriminator triggers.

Key Features:
  1. Bar-level and individual SiPM trigger evaluation matching official DigiTaskSND.cxx.
  2. Evaluates the turn-on energy deposit Delta E_min [keV] across all SiPMs and per-bar.
  3. Maps turn-on threshold as a function of hit position along the bar (Veto x in [-65, 0]).
  4. S-curve hit efficiency vs. sum of clustered MCPoint energy deposit:
     - For every bar in the detector (stored in ROOT file and summary plots)
     - For specified subsystem categories:
         * all veto bars
         * all US bars
         * US0, US1, US2
         * US3, US4
         * All vertical DS bars
         * All horizontal DS bars
     - For individual SiPMs vs. full bar (OR condition).
  5. Exports publication-quality PNG figures, summary JSON, and ROOT file containing:
     - TEfficiencies and histograms for all bars and categories
     - Styled ROOT TCanvas objects with TLegends and grid lines for every plot.
"""

import os
import sys
import time
import json
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import ROOT
import min_pe as mpe

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Study MCPoint Turn-On Thresholds")
    parser.add_argument(
        "--mc-file",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP.root",
        help="Path to PMU simulation ROOT file",
    )
    parser.add_argument("--start-event", type=int, default=0, help="Starting event index")
    parser.add_argument("--max-events", type=int, default=30000, help="Max events to process (-1 for all)")
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory for plots")
    parser.add_argument("--root-output", type=str, default=None, help="Custom ROOT output file path")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG generation (for parallel workers)")
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2",
        help="Artifact directory path",
    )
    return parser.parse_args()


def get_categories(subsystem: int, plane: int, bar: int):
    """Assign bar to predefined subsystem categories."""
    cats = []
    if subsystem == 1:
        cats.append("all_veto")
    elif subsystem == 2:
        cats.append("all_us")
        if plane in (0, 1, 2):
            cats.append("us_012")
        elif plane in (3, 4):
            cats.append("us_34")
    elif subsystem == 3:
        if bar < 60:
            cats.append("ds_hor")
        else:
            cats.append("ds_ver")
    return cats


def run_study():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    root_out_path = args.root_output if args.root_output else os.path.join(args.outdir, "mc_threshold_study.root")

    print(f"Opening PMU simulation file: {args.mc_file}")
    f_in = ROOT.TFile.Open(args.mc_file)
    if not f_in or f_in.IsZombie():
        raise FileNotFoundError(f"Could not open {args.mc_file}")

    tree = f_in.Get("cbmsim")
    if not tree:
        raise KeyError("cbmsim tree not found in file")

    n_entries = tree.GetEntries()
    start_ev = max(0, args.start_event)
    if args.max_events < 0:
        end_ev = n_entries
    else:
        end_ev = min(start_ev + args.max_events, n_entries)
    n_to_process = end_ev - start_ev
    print(f"Tree has {n_entries} events. Processing events {start_ev} to {end_ev} ({n_to_process} events)...")

    # Data structures for subsystem position and threshold distributions
    records = {
        "veto": {"eth_kev": [], "pe_trig": [], "pe_gel": [], "pe_air": [], "edep_mev": [], "qualified": [], "dist_cm": [], "pos_cm": [], "weight": []},
        "us": {"eth_kev": [], "pe_trig": [], "pe_gel": [], "pe_air": [], "edep_mev": [], "qualified": [], "station": [], "dist_cm": [], "pos_cm": [], "weight": []},
        "dsh": {"eth_kev": [], "pe_trig": [], "pe_left": [], "pe_right": [], "edep_mev": [], "qualified": [], "dist_cm": [], "pos_cm": [], "weight": []},
        "dsv": {"eth_kev": [], "pe_trig": [], "pe_top": [], "edep_mev": [], "qualified": [], "dist_cm": [], "pos_cm": [], "weight": []},
    }

    # Category tracking
    cat_data = {
        "all_veto": {"edep_kev": [], "qualified": [], "weight": []},
        "all_us": {"edep_kev": [], "qualified": [], "weight": []},
        "us_012": {"edep_kev": [], "qualified": [], "weight": []},
        "us_34": {"edep_kev": [], "qualified": [], "weight": []},
        "ds_ver": {"edep_kev": [], "qualified": [], "weight": []},
        "ds_hor": {"edep_kev": [], "qualified": [], "weight": []},
    }

    # Per-bar tracking: det_id -> {"edep_kev": [], "qualified": [], "weight": [], "subsystem": s, "plane": p, "bar": b}
    bar_data = defaultdict(lambda: {"edep_kev": [], "qualified": [], "weight": [], "subsystem": 0, "plane": 0, "bar": 0})

    # Individual SiPM vs Full Bar tracking
    sipm_data = {
        "us": {"edep_kev": [], "bar_qual": [], "gel_qual": [], "air_qual": [], "weight": []},
        "dsh": {"edep_kev": [], "bar_qual": [], "left_qual": [], "right_qual": [], "weight": []},
    }

    # Optimize ROOT I/O by enabling only needed branches
    tree.SetBranchStatus("*", 0)
    tree.SetBranchStatus("MCTrack*", 1)
    tree.SetBranchStatus("MuFilterPoint*", 1)

    t0 = time.time()
    total_bars_traversed = 0

    for iev in range(start_ev, end_ev):
        tree.GetEntry(iev)
        pts_branch = tree.MuFilterPoint
        if not pts_branch or pts_branch.GetEntries() == 0:
            continue

        tracks = getattr(tree, "MCTrack", None)
        event_weight = float(tracks.At(0).GetWeight()) if (tracks and tracks.GetEntriesFast() > 0) else 1.0

        grouped = mpe.group_mcpoints_by_bar(pts_branch)

        for det_id, clusters in grouped.items():
            for pts in clusters:
                total_bars_traversed += 1
                subsystem, plane, bar = mpe._DEFAULT_CALCULATOR.decode_detector_id(det_id)

                pts_info = [mpe._DEFAULT_CALCULATOR._extract_point_info(p) for p in pts]
                total_edep = sum(p[5] for p in pts_info)
                if total_edep <= 1e-6:
                    continue

                total_edep_kev = total_edep * 1000.0
                x_wt = sum(p[1] * p[5] for p in pts_info) / total_edep
                y_wt = sum(p[2] * p[5] for p in pts_info) / total_edep
                z_wt = sum(p[3] * p[5] for p in pts_info) / total_edep
                mean_pt = {"detID": det_id, "x": x_wt, "y": y_wt, "z": z_wt, "time": 0.0, "eloss": total_edep}

                th_info = mpe.get_lowest_trigger_threshold(pts, det_id=det_id, include_mirror=True)
                eth_kev = th_info["min_eloss_mev"] * 1000.0
                is_qual = th_info["current_hit_qualified"]
                current_pes = th_info["current_pes"]
                max_pe = float(np.max(current_pes)) if len(current_pes) > 0 else 0.0
                sipm_qual = th_info["sipm_hit_qualified"]

                # Record per-bar data
                b_entry = bar_data[det_id]
                b_entry["edep_kev"].append(total_edep_kev)
                b_entry["qualified"].append(is_qual)
                b_entry["weight"].append(event_weight)
                b_entry["subsystem"] = subsystem
                b_entry["plane"] = plane
                b_entry["bar"] = bar

                # Record category data
                for cat in get_categories(subsystem, plane, bar):
                    cat_data[cat]["edep_kev"].append(total_edep_kev)
                    cat_data[cat]["qualified"].append(is_qual)
                    cat_data[cat]["weight"].append(event_weight)

                # Record subsystem spatial & headroom data
                if subsystem == 1:
                    pe_l = current_pes[0] if len(current_pes) > 0 else 0.0
                    pe_r = current_pes[8] if len(current_pes) > 8 else pe_l
                    d_left = mpe._DEFAULT_CALCULATOR.get_distance_to_sipm(mean_pt, 0)
                    records["veto"]["eth_kev"].append(eth_kev)
                    records["veto"]["pe_trig"].append(max_pe)
                    records["veto"]["pe_gel"].append(max(pe_l, pe_r))
                    records["veto"]["pe_air"].append(min(pe_l, pe_r))
                    records["veto"]["edep_mev"].append(total_edep)
                    records["veto"]["qualified"].append(is_qual)
                    records["veto"]["dist_cm"].append(d_left)
                    records["veto"]["pos_cm"].append(x_wt)
                    records["veto"]["weight"].append(event_weight)

                elif subsystem == 2:
                    pe_l = current_pes[0] if len(current_pes) > 0 else 0.0
                    pe_r = current_pes[8] if len(current_pes) > 8 else pe_l
                    has_gel_left = mpe._DEFAULT_CALCULATOR.has_gel(det_id, 0)
                    pe_gel = pe_l if has_gel_left else pe_r
                    pe_air = pe_r if has_gel_left else pe_l
                    gel_ch = 0 if has_gel_left else 8
                    d_gel = mpe._DEFAULT_CALCULATOR.get_distance_to_sipm(mean_pt, gel_ch)

                    records["us"]["eth_kev"].append(eth_kev)
                    records["us"]["pe_trig"].append(max_pe)
                    records["us"]["pe_gel"].append(pe_gel)
                    records["us"]["pe_air"].append(pe_air)
                    records["us"]["edep_mev"].append(total_edep)
                    records["us"]["qualified"].append(is_qual)
                    records["us"]["station"].append(plane)
                    records["us"]["dist_cm"].append(d_gel)
                    records["us"]["pos_cm"].append(x_wt)
                    records["us"]["weight"].append(event_weight)

                    gel_pass = bool(np.any(sipm_qual[:8])) if has_gel_left else bool(np.any(sipm_qual[8:]))
                    air_pass = bool(np.any(sipm_qual[8:])) if has_gel_left else bool(np.any(sipm_qual[:8]))
                    sipm_data["us"]["edep_kev"].append(total_edep_kev)
                    sipm_data["us"]["bar_qual"].append(is_qual)
                    sipm_data["us"]["gel_qual"].append(gel_pass)
                    sipm_data["us"]["air_qual"].append(air_pass)
                    sipm_data["us"]["weight"].append(event_weight)

                elif subsystem == 3:
                    if bar < 60:
                        pe_l = current_pes[0] if len(current_pes) > 0 else 0.0
                        pe_r = current_pes[1] if len(current_pes) > 1 else 0.0
                        d_l = mpe._DEFAULT_CALCULATOR.get_distance_to_sipm(mean_pt, 0)
                        records["dsh"]["eth_kev"].append(eth_kev)
                        records["dsh"]["pe_trig"].append(max_pe)
                        records["dsh"]["pe_left"].append(pe_l)
                        records["dsh"]["pe_right"].append(pe_r)
                        records["dsh"]["edep_mev"].append(total_edep)
                        records["dsh"]["qualified"].append(is_qual)
                        records["dsh"]["dist_cm"].append(d_l)
                        records["dsh"]["pos_cm"].append(x_wt)
                        records["dsh"]["weight"].append(event_weight)

                        l_pass = bool(sipm_qual[0]) if len(sipm_qual) > 0 else False
                        r_pass = bool(sipm_qual[1]) if len(sipm_qual) > 1 else False
                        sipm_data["dsh"]["edep_kev"].append(total_edep_kev)
                        sipm_data["dsh"]["bar_qual"].append(is_qual)
                        sipm_data["dsh"]["left_qual"].append(l_pass)
                        sipm_data["dsh"]["right_qual"].append(r_pass)
                        sipm_data["dsh"]["weight"].append(event_weight)
                    else:
                        pe_top = current_pes[0] if len(current_pes) > 0 else 0.0
                        d_top = mpe._DEFAULT_CALCULATOR.get_distance_to_sipm(mean_pt, 0)
                        records["dsv"]["eth_kev"].append(eth_kev)
                        records["dsv"]["pe_trig"].append(max_pe)
                        records["dsv"]["pe_top"].append(pe_top)
                        records["dsv"]["edep_mev"].append(total_edep)
                        records["dsv"]["qualified"].append(is_qual)
                        records["dsv"]["dist_cm"].append(d_top)
                        records["dsv"]["pos_cm"].append(y_wt)
                        records["dsv"]["weight"].append(event_weight)

        cur_processed = iev - start_ev + 1
        if cur_processed % 5000 == 0 or cur_processed == n_to_process:
            elapsed = time.time() - t0
            rate = cur_processed / max(elapsed, 1e-6)
            print(f"  Processed {cur_processed}/{n_to_process} events ({rate:.1f} ev/s, {total_bars_traversed} bars seen)")

    f_in.Close()

    # Convert records to numpy arrays
    for cat in records:
        for k in records[cat]:
            records[cat][k] = np.array(records[cat][k])
    for cat in cat_data:
        for k in cat_data[cat]:
            cat_data[cat][k] = np.array(cat_data[cat][k])
    for cat in sipm_data:
        for k in sipm_data[cat]:
            sipm_data[cat][k] = np.array(sipm_data[cat][k])
    for det_id in bar_data:
        bar_data[det_id]["edep_kev"] = np.array(bar_data[det_id]["edep_kev"])
        bar_data[det_id]["qualified"] = np.array(bar_data[det_id]["qualified"])
        bar_data[det_id]["weight"] = np.array(bar_data[det_id]["weight"])

    # Print summary statistics table
    print("\n" + "=" * 95)
    print(f"{'Subsystem Category':<24} | {'Traversals':<10} | {'Qualified':<10} | {'Efficiency':<10} | {'Median Eth [keV]':<18}")
    print("=" * 95)

    summary_stats = {}
    cat_display_names = {
        "all_veto": "All Veto Bars",
        "all_us": "All US Bars",
        "us_012": "US0, US1, US2",
        "us_34": "US3, US4",
        "ds_ver": "All Vertical DS Bars",
        "ds_hor": "All Horizontal DS Bars",
    }

    for cat, name in cat_display_names.items():
        n_tot = len(cat_data[cat]["qualified"])
        n_pass = int(np.sum(cat_data[cat]["qualified"])) if n_tot > 0 else 0
        eff = (n_pass / n_tot * 100.0) if n_tot > 0 else 0.0
        if cat == "all_veto":
            med_eth = float(np.median(records["veto"]["eth_kev"])) if len(records["veto"]["eth_kev"]) > 0 else 0.0
        elif cat == "all_us":
            med_eth = float(np.median(records["us"]["eth_kev"])) if len(records["us"]["eth_kev"]) > 0 else 0.0
        elif cat == "us_012":
            m_012 = np.isin(records["us"]["station"], [0, 1, 2])
            med_eth = float(np.median(records["us"]["eth_kev"][m_012])) if np.any(m_012) else 0.0
        elif cat == "us_34":
            m_34 = np.isin(records["us"]["station"], [3, 4])
            med_eth = float(np.median(records["us"]["eth_kev"][m_34])) if np.any(m_34) else 0.0
        elif cat == "ds_ver":
            med_eth = float(np.median(records["dsv"]["eth_kev"])) if len(records["dsv"]["eth_kev"]) > 0 else 0.0
        elif cat == "ds_hor":
            med_eth = float(np.median(records["dsh"]["eth_kev"])) if len(records["dsh"]["eth_kev"]) > 0 else 0.0

        summary_stats[cat] = {
            "name": name,
            "traversals": n_tot,
            "qualified": n_pass,
            "efficiency_pct": eff,
            "median_eth_kev": med_eth,
        }
        print(f"{name:<24} | {n_tot:<10} | {n_pass:<10} | {eff:<9.2f}% | {med_eth:<18.1f}")
    print("=" * 95)

    # -------------------------------------------------------------
    # Open ROOT File to Store Histograms, TEfficiencies, and Canvases
    # -------------------------------------------------------------
    f_root = ROOT.TFile.Open(root_out_path, "RECREATE")

    # 1. Category directories & TEfficiency
    dir_cats = f_root.mkdir("categories")
    dir_cats.cd()
    teff_dict = {}
    h_pass_dict = {}
    h_tot_dict = {}
    for cat in cat_display_names:
        edeps = cat_data[cat]["edep_kev"]
        quals = cat_data[cat]["qualified"]
        wgts = cat_data[cat]["weight"]
        h_pass = ROOT.TH1F(f"h_pass_{cat}", f"Passed Clusters {cat};#Delta E [keV];Weighted Counts", 50, 0, 500)
        h_tot = ROOT.TH1F(f"h_tot_{cat}", f"Total Clusters {cat};#Delta E [keV];Weighted Counts", 50, 0, 500)
        h_pass.Sumw2()
        h_tot.Sumw2()
        for e, q, w in zip(edeps, quals, wgts):
            h_tot.Fill(e, w)
            if q:
                h_pass.Fill(e, w)
        h_pass.Write()
        h_tot.Write()
        h_pass_dict[cat] = h_pass
        h_tot_dict[cat] = h_tot
        if ROOT.TEfficiency.CheckConsistency(h_pass, h_tot):
            teff = ROOT.TEfficiency(h_pass, h_tot)
            teff.SetName(f"teff_{cat}")
            teff.SetTitle(f"Hit Efficiency S-Curve: {cat};#Delta E [keV];Efficiency")
            teff.Write()
            teff_dict[cat] = teff

    # 2. Individual Bar directories & TEfficiencies
    dir_bars = f_root.mkdir("bars")
    dir_bars.cd()
    bar_teff_dict = {}
    for det_id, b_entry in sorted(bar_data.items()):
        edeps = b_entry["edep_kev"]
        quals = b_entry["qualified"]
        wgts = b_entry["weight"]
        if len(edeps) < 5:
            continue
        h_p = ROOT.TH1F(f"h_pass_bar_{det_id}", f"Passed Clusters Bar {det_id};#Delta E [keV];Weighted Counts", 50, 0, 500)
        h_t = ROOT.TH1F(f"h_tot_bar_{det_id}", f"Total Clusters Bar {det_id};#Delta E [keV];Weighted Counts", 50, 0, 500)
        h_p.Sumw2()
        h_t.Sumw2()
        for e, q, w in zip(edeps, quals, wgts):
            h_t.Fill(e, w)
            if q:
                h_p.Fill(e, w)
        h_p.Write()
        h_t.Write()
        if ROOT.TEfficiency.CheckConsistency(h_p, h_t):
            teff_b = ROOT.TEfficiency(h_p, h_t)
            teff_b.SetName(f"teff_bar_{det_id}")
            teff_b.SetTitle(f"Hit Efficiency Bar {det_id};#Delta E [keV];Efficiency")
            teff_b.Write()
            bar_teff_dict[det_id] = teff_b

    # 3. Subsystem 2D histograms (Eth vs Pos) and 1D distributions
    h1_eth_dict = {}
    h1_pe_dict = {}
    h2_pos_dict = {}
    for cat in ["veto", "us", "dsh", "dsv"]:
        dir_sub = f_root.mkdir(cat)
        dir_sub.cd()
        h_eth = ROOT.TH1F(f"h_eth_{cat}", f"Turn-On Threshold {cat};#Delta E_{{min}} [keV];Weighted Counts", 100, 0, 400)
        h_eth.Sumw2()
        for v, w in zip(records[cat]["eth_kev"], records[cat]["weight"]):
            h_eth.Fill(v, w)
        h_eth.Write()
        h1_eth_dict[cat] = h_eth

        h_pe = ROOT.TH1F(f"h_pe_{cat}", f"Triggering SiPM Photoelectrons {cat};N_{{pe}} [p.e.];Weighted Counts", 100, 0, 600)
        h_pe.Sumw2()
        for v, w in zip(records[cat]["pe_trig"], records[cat]["weight"]):
            h_pe.Fill(v, w)
        h_pe.Write()
        h1_pe_dict[cat] = h_pe

        x_val = records[cat]["dist_cm"] if cat in ["us", "dsv"] else records[cat]["pos_cm"]
        h2_pos = ROOT.TH2F(f"h2_eth_vs_pos_{cat}", f"Turn-On Threshold vs Pos {cat};Position/Dist [cm];#Delta E_{{min}} [keV]", 50, -65 if cat == "veto" else -50, 90, 50, 0, 300)
        h2_pos.Sumw2()
        for p_x, p_y, w in zip(x_val, records[cat]["eth_kev"], records[cat]["weight"]):
            h2_pos.Fill(p_x, p_y, w)
        h2_pos.Write()
        h2_pos_dict[cat] = h2_pos

    # -------------------------------------------------------------
    # Build & Store ROOT TCanvas Objects with Legends in ROOT File
    # -------------------------------------------------------------
    dir_cvs = f_root.mkdir("canvases")
    dir_cvs.cd()

    # Canvas 1: Subsystem Category S-Curves
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

    # Keep objects in list to prevent ROOT garbage collection before write
    draw_keep = []

    de_bins = np.linspace(0.0, 500.0, 51)
    bin_centers = 0.5 * (de_bins[:-1] + de_bins[1:])

    for cat_name, (color_code, marker_style) in root_cat_colors.items():
        edep_arr = cat_data[cat_name]["edep_kev"]
        qual_arr = cat_data[cat_name]["qualified"]
        if len(edep_arr) == 0:
            continue

        gr = ROOT.TGraphErrors()
        pt_idx = 0
        for i in range(len(de_bins) - 1):
            mask = (edep_arr >= de_bins[i]) & (edep_arr < de_bins[i + 1])
            n_tot_bin = int(np.sum(mask))
            if n_tot_bin >= 8:
                n_pass_bin = int(np.sum(qual_arr[mask]))
                p_eff = (n_pass_bin / n_tot_bin) * 100.0
                err_eff = np.sqrt(p_eff * (100.0 - p_eff) / n_tot_bin) if n_tot_bin > 1 else 0.0
                gr.SetPoint(pt_idx, float(bin_centers[i]), float(p_eff))
                gr.SetPointError(pt_idx, 0.0, float(err_eff))
                pt_idx += 1

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

    # 50% and 99% dashed lines
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
    c_cat.Write()
    draw_keep.extend([mg_cat, leg_cat, line50, line99])

    # Canvas 2: Threshold vs Position (4-pad canvas)
    c_pos = ROOT.TCanvas("c_mc_threshold_vs_position", "Turn-On Threshold vs Position", 1200, 900)
    c_pos.Divide(2, 2)
    c_pos.cd(1)
    ROOT.gPad.SetGrid()
    h2_pos_dict["us"].SetTitle("US Bar: #Delta E_{min} vs Distance to Gel SiPM;d_{gel} [cm];#Delta E_{min} [keV]")
    h2_pos_dict["us"].Draw("COLZ")

    c_pos.cd(2)
    ROOT.gPad.SetGrid()
    h2_pos_dict["dsh"].SetTitle("DS Horizontal Bar: #Delta E_{min} vs Impact Position x;Impact x [cm];#Delta E_{min} [keV]")
    h2_pos_dict["dsh"].Draw("COLZ")

    c_pos.cd(3)
    ROOT.gPad.SetGrid()
    h2_pos_dict["dsv"].SetTitle("DS Vertical Bar: #Delta E_{min} vs Distance to Top SiPM;d_{top} [cm];#Delta E_{min} [keV]")
    h2_pos_dict["dsv"].Draw("COLZ")

    c_pos.cd(4)
    ROOT.gPad.SetGrid()
    h2_pos_dict["veto"].SetTitle("Veto Bar: #Delta E_{min} vs Impact Position x ([-65, 0] cm);Impact x [cm];#Delta E_{min} [keV]")
    h2_pos_dict["veto"].Draw("COLZ")

    c_pos.Write()

    # Canvas 3: Individual SiPMs vs Full Bar S-Curves (2-pad canvas)
    c_sipm = ROOT.TCanvas("c_mc_sipm_vs_bar_turn_on", "Individual SiPM vs Full Bar Hit Qualification", 1200, 550)
    c_sipm.Divide(2, 1)

    # Pad 1: US Bar
    c_sipm.cd(1)
    ROOT.gPad.SetGrid()
    mg_us = ROOT.TMultiGraph("mg_us_sipm", "US Bar (US0-4): Gel vs Air vs Full Bar;#Delta E [keV];Efficiency [%]")
    leg_us = ROOT.TLegend(0.55, 0.16, 0.88, 0.38)
    leg_us.SetBorderSize(1)
    leg_us.SetFillColor(ROOT.kWhite)

    ed_us = sipm_data["us"]["edep_kev"]
    if len(ed_us) > 0:
        for key_s, label_s, col_s, ls_s in [
            ("bar_qual", "Full Bar (OR Condition: >= 1 SiPM)", ROOT.kBlack, 1),
            ("gel_qual", "Gel-Coupled SiPM Side", ROOT.kGreen + 2, 2),
            ("air_qual", "Air-Gap SiPM Side", ROOT.kBlue + 1, 3),
        ]:
            gr_s = ROOT.TGraph()
            pt_idx = 0
            for i in range(len(de_bins) - 1):
                mask = (ed_us >= de_bins[i]) & (ed_us < de_bins[i + 1])
                if np.sum(mask) >= 8:
                    p_val = float(np.mean(sipm_data["us"][key_s][mask]) * 100.0)
                    gr_s.SetPoint(pt_idx, float(bin_centers[i]), p_val)
                    pt_idx += 1
            gr_s.SetLineColor(col_s)
            gr_s.SetLineWidth(2)
            gr_s.SetLineStyle(ls_s)
            mg_us.Add(gr_s, "l")
            leg_us.AddEntry(gr_s, label_s, "l")
            draw_keep.append(gr_s)

    mg_us.Draw("a")
    mg_us.GetXaxis().SetLimits(0, 450)
    mg_us.SetMinimum(-5)
    mg_us.SetMaximum(105)
    mg_us.GetXaxis().SetTitle("Clustered Energy Deposit #Delta E [keV]")
    mg_us.GetYaxis().SetTitle("Efficiency [%]")
    leg_us.Draw()

    # Pad 2: DS Horizontal Bar
    c_sipm.cd(2)
    ROOT.gPad.SetGrid()
    mg_dsh = ROOT.TMultiGraph("mg_dsh_sipm", "DS Horizontal Bar: Left vs Right vs Full Bar;#Delta E [keV];Efficiency [%]")
    leg_dsh = ROOT.TLegend(0.55, 0.16, 0.88, 0.38)
    leg_dsh.SetBorderSize(1)
    leg_dsh.SetFillColor(ROOT.kWhite)

    ed_ds = sipm_data["dsh"]["edep_kev"]
    if len(ed_ds) > 0:
        for key_s, label_s, col_s, ls_s in [
            ("bar_qual", "Full Bar (OR: Left OR Right)", ROOT.kBlack, 1),
            ("left_qual", "Left SiPM Channel", ROOT.kRed + 1, 2),
            ("right_qual", "Right SiPM Channel", ROOT.kBlue + 1, 3),
        ]:
            gr_s = ROOT.TGraph()
            pt_idx = 0
            for i in range(len(de_bins) - 1):
                mask = (ed_ds >= de_bins[i]) & (ed_ds < de_bins[i + 1])
                if np.sum(mask) >= 8:
                    p_val = float(np.mean(sipm_data["dsh"][key_s][mask]) * 100.0)
                    gr_s.SetPoint(pt_idx, float(bin_centers[i]), p_val)
                    pt_idx += 1
            gr_s.SetLineColor(col_s)
            gr_s.SetLineWidth(2)
            gr_s.SetLineStyle(ls_s)
            mg_dsh.Add(gr_s, "l")
            leg_dsh.AddEntry(gr_s, label_s, "l")
            draw_keep.append(gr_s)

    mg_dsh.Draw("a")
    mg_dsh.GetXaxis().SetLimits(0, 250)
    mg_dsh.SetMinimum(-5)
    mg_dsh.SetMaximum(105)
    mg_dsh.GetXaxis().SetTitle("Clustered Energy Deposit #Delta E [keV]")
    mg_dsh.GetYaxis().SetTitle("Efficiency [%]")
    leg_dsh.Draw()

    c_sipm.Write()
    draw_keep.extend([mg_us, leg_us, mg_dsh, leg_dsh])

    # Canvas 4: Bar-by-Bar S-Curves (4-pad canvas)
    c_bars = ROOT.TCanvas("c_mc_bar_level_efficiency_curves", "Bar-by-Bar Efficiency S-Curves", 1300, 900)
    c_bars.Divide(2, 2)
    c_bars.cd(1); ROOT.gPad.SetGrid()
    mg_b1 = ROOT.TMultiGraph("mg_bars_veto", "Veto Bars (14 bars);#Delta E [keV];Efficiency [%]")
    c_bars.cd(2); ROOT.gPad.SetGrid()
    mg_b2 = ROOT.TMultiGraph("mg_bars_us", "Upstream Bars (50 bars);#Delta E [keV];Efficiency [%]")
    c_bars.cd(3); ROOT.gPad.SetGrid()
    mg_b3 = ROOT.TMultiGraph("mg_bars_dsh", "DS Horizontal Bars (240 bars);#Delta E [keV];Efficiency [%]")
    c_bars.cd(4); ROOT.gPad.SetGrid()
    mg_b4 = ROOT.TMultiGraph("mg_bars_dsv", "DS Vertical Bars (180 bars);#Delta E [keV];Efficiency [%]")

    mg_bar_map = {1: mg_b1, 2: mg_b2, 30: mg_b3, 31: mg_b4}
    for det_id, b_entry in sorted(bar_data.items()):
        edeps = b_entry["edep_kev"]
        quals = b_entry["qualified"]
        if len(edeps) < 15:
            continue
        s = b_entry["subsystem"]
        b = b_entry["bar"]
        k_b = s if s in (1, 2) else (30 if b < 60 else 31)
        target_mg = mg_bar_map[k_b]

        gr_b = ROOT.TGraph()
        pt_idx = 0
        for i in range(len(de_bins) - 1):
            mask = (edeps >= de_bins[i]) & (edeps < de_bins[i + 1])
            if np.sum(mask) >= 5:
                gr_b.SetPoint(pt_idx, float(bin_centers[i]), float(np.mean(quals[mask]) * 100.0))
                pt_idx += 1
        if pt_idx > 0:
            gr_b.SetLineColor(ROOT.kGray + 1)
            gr_b.SetLineWidth(1)
            target_mg.Add(gr_b, "l")
            draw_keep.append(gr_b)

    for pad_idx, target_mg in enumerate([mg_b1, mg_b2, mg_b3, mg_b4], start=1):
        c_bars.cd(pad_idx)
        target_mg.Draw("a")
        target_mg.GetXaxis().SetLimits(0, 450)
        target_mg.SetMinimum(-5)
        target_mg.SetMaximum(105)
        target_mg.GetXaxis().SetTitle("#Delta E in Bar [keV]")
        target_mg.GetYaxis().SetTitle("Efficiency [%]")
        draw_keep.append(target_mg)

    c_bars.Write()

    # Canvas 5: Turn-On Energy Threshold Distributions (1D)
    c_eth = ROOT.TCanvas("c_mc_bar_turn_on_thresholds_distribution", "Turn-On Threshold Distributions", 1100, 800)
    c_eth.Divide(2, 2)
    for pad_idx, (cat_k, cat_lbl) in enumerate([("veto", "Veto Bar"), ("us", "US Bar"), ("dsh", "DS Horizontal"), ("dsv", "DS Vertical")], start=1):
        c_eth.cd(pad_idx)
        ROOT.gPad.SetGrid()
        h1_eth_dict[cat_k].SetTitle(f"{cat_lbl};#Delta E_{{min}} [keV];Counts")
        h1_eth_dict[cat_k].SetLineColor(ROOT.kBlue + 2)
        h1_eth_dict[cat_k].SetFillColor(ROOT.kAzure - 9)
        h1_eth_dict[cat_k].Draw("HIST")
    c_eth.Write()

    # Canvas 6: Photoelectron Headroom (1D)
    c_pe = ROOT.TCanvas("c_mc_sipm_photoelectron_headroom", "Photoelectron Headroom vs 11.3 pe", 1100, 800)
    c_pe.Divide(2, 2)
    for pad_idx, (cat_k, cat_lbl) in enumerate([("veto", "Veto Bar"), ("us", "US Bar"), ("dsh", "DS Horizontal"), ("dsv", "DS Vertical")], start=1):
        c_pe.cd(pad_idx)
        ROOT.gPad.SetGrid()
        ROOT.gPad.SetLogy()
        h1_pe_dict[cat_k].SetTitle(f"{cat_lbl};Photoelectrons N_{{pe}} [p.e.];Counts")
        h1_pe_dict[cat_k].SetLineColor(ROOT.kRed + 2)
        h1_pe_dict[cat_k].SetFillColor(ROOT.kOrange - 9)
        h1_pe_dict[cat_k].Draw("HIST")
        l_th = ROOT.TLine(11.33, 0.1, 11.33, h1_pe_dict[cat_k].GetMaximum() * 1.5)
        l_th.SetLineColor(ROOT.kBlack)
        l_th.SetLineStyle(2)
        l_th.SetLineWidth(2)
        l_th.Draw("same")
        draw_keep.append(l_th)
    c_pe.Write()

    f_root.Close()
    print(f"Saved ROOT output file with all TCanvases: {root_out_path}")

    # -------------------------------------------------------------
    # Generate Matplotlib PNG Figures (if not --no-plots)
    # -------------------------------------------------------------
    if not args.no_plots:
        # Plot 1: Turn-On Threshold vs Hit Position Along Bar (Veto x in [-65, 0])
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle("Minimum Turn-On Energy Threshold $\\Delta E_{\\rm min}$ vs. Hit Position Along Scintillator Bar\n(Spatial Variation Due to Attenuation & Coupling)", fontsize=13, weight="bold")

        pos_configs = [
            ("us", "US Bar: $\\Delta E_{\\rm min}$ vs Distance to Gel SiPM", axes[0, 0], "#2ca02c", "Distance to Gel Coupling End $d_{\\rm gel}$ [cm]", (0, 85), (100, 250)),
            ("dsh", "DS Horizontal: $\\Delta E_{\\rm min}$ vs Impact Position $x$", axes[0, 1], "#ff7f0e", "Horizontal Impact Position $x$ [cm]", (-42, 42), (30, 80)),
            ("dsv", "DS Vertical: $\\Delta E_{\\rm min}$ vs Distance to Top SiPM", axes[1, 0], "#d62728", "Distance from Top Readout End $d_{\\rm top}$ [cm]", (0, 65), (20, 60)),
            ("veto", "Veto Bar: $\\Delta E_{\\rm min}$ vs Impact Position $x$ (SND@LHC Geometry)", axes[1, 1], "#1f77b4", "Horizontal Impact Position $x$ [cm]", (-65, 0), (90, 180)),
        ]

        for cat, title, ax, color, xlabel, xlim, ylim in pos_configs:
            eth = records[cat]["eth_kev"]
            x_val = records[cat]["dist_cm"] if "Distance" in xlabel else records[cat]["pos_cm"]
            if len(eth) == 0:
                continue

            hb = ax.hexbin(x_val, eth, gridsize=35, cmap="Blues" if color == "#1f77b4" else ("Greens" if color == "#2ca02c" else ("Oranges" if color == "#ff7f0e" else "Reds")), mincnt=1)
            cb = plt.colorbar(hb, ax=ax)
            cb.set_label("Traversals", fontsize=9)

            n_prof_bins = 15
            p_bins = np.linspace(xlim[0], xlim[1], n_prof_bins + 1)
            p_cen = 0.5 * (p_bins[:-1] + p_bins[1:])
            prof_eth = []
            for ib in range(n_prof_bins):
                m = (x_val >= p_bins[ib]) & (x_val < p_bins[ib + 1])
                prof_eth.append(np.median(eth[m]) if np.sum(m) >= 5 else np.nan)
            ax.plot(p_cen, prof_eth, "k.-", linewidth=2.0, markersize=7, label="Median Profile")

            ax.set_title(title, fontsize=11, weight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Minimum Turn-On Energy $\\Delta E_{\\rm min}$ [keV]", fontsize=10)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(fontsize=9, loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p1_path = os.path.join(args.outdir, "mc_threshold_vs_position.png")
        plt.savefig(p1_path, dpi=200)
        plt.close()
        print(f"Saved: {p1_path}")

        # Plot 2: Subsystem Category Turn-On S-Curves (Requested Categories)
        plt.figure(figsize=(10, 6.5))
        cat_styles = {
            "all_veto": {"color": "#1f77b4", "ls": "-", "marker": "o", "label": "All Veto Bars"},
            "all_us": {"color": "#2ca02c", "ls": "-", "marker": "s", "label": "All US Bars"},
            "us_012": {"color": "#17becf", "ls": "--", "marker": "^", "label": "US0, US1, US2 (Stations 0-2)"},
            "us_34": {"color": "#98df8a", "ls": "-.", "marker": "v", "label": "US3, US4 (Stations 3-4)"},
            "ds_ver": {"color": "#d62728", "ls": "-", "marker": "D", "label": "All Vertical DS Bars"},
            "ds_hor": {"color": "#ff7f0e", "ls": "-", "marker": "P", "label": "All Horizontal DS Bars"},
        }

        for cat, style in cat_styles.items():
            edep_arr = cat_data[cat]["edep_kev"]
            qual_arr = cat_data[cat]["qualified"]
            if len(edep_arr) == 0:
                continue
            effs = []
            for i in range(len(de_bins) - 1):
                mask = (edep_arr >= de_bins[i]) & (edep_arr < de_bins[i + 1])
                if np.sum(mask) >= 8:
                    effs.append(float(np.mean(qual_arr[mask]) * 100.0))
                else:
                    effs.append(np.nan)
            plt.plot(bin_centers, effs, color=style["color"], linestyle=style["ls"], marker=style["marker"], label=style["label"], linewidth=2.0, markersize=5)

        plt.title("MuFilter & Veto Hit Formation Efficiency S-Curves by Subsystem Category\n(Fraction of Clustered MCPoint Traversals Forming Hits vs. Deposited Energy)", fontsize=12, weight="bold")
        plt.xlabel("Sum of Clustered MCPoint Energy Depositions $\\Delta E$ in Bar [keV]", fontsize=11)
        plt.ylabel("Hit Formation Efficiency [%]", fontsize=11)
        plt.axhline(50.0, color="gray", linestyle=":", label="50% Threshold")
        plt.axhline(99.0, color="black", linestyle="--", alpha=0.7, label="99% Full Turn-On")
        plt.ylim(-5, 105)
        plt.xlim(0, 450)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=9.5, loc="lower right")
        plt.tight_layout()

        p2_path = os.path.join(args.outdir, "mc_subsystem_efficiency_scurves.png")
        plt.savefig(p2_path, dpi=200)
        plt.close()
        print(f"Saved: {p2_path}")

        # Plot 3: Bar-by-Bar Turn-On Efficiency S-Curves
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Bar-Level Hit Formation Efficiency Turn-On S-Curves for Each Bar in Detector\n(Sum of Clustered MCPoint Energy Depositions vs. Hit Formation Fraction)", fontsize=13, weight="bold")

        bar_sub_map = {
            1: (axes[0, 0], "Veto Bars (14 bars)", plt.cm.Blues),
            2: (axes[0, 1], "Upstream Bars (50 bars, US0-US4)", plt.cm.Greens),
            30: (axes[1, 0], "DS Horizontal Bars (240 bars)", plt.cm.Oranges),
            31: (axes[1, 1], "DS Vertical Bars (180 bars)", plt.cm.Reds),
        }

        for det_id, b_entry in sorted(bar_data.items()):
            edeps = b_entry["edep_kev"]
            quals = b_entry["qualified"]
            if len(edeps) < 15:
                continue

            s = b_entry["subsystem"]
            b = b_entry["bar"]
            key = s if s in (1, 2) else (30 if b < 60 else 31)
            ax, title, cmap = bar_sub_map[key]

            effs = []
            for i in range(len(de_bins) - 1):
                mask = (edeps >= de_bins[i]) & (edeps < de_bins[i + 1])
                if np.sum(mask) >= 5:
                    effs.append(float(np.mean(quals[mask]) * 100.0))
                else:
                    effs.append(np.nan)

            c_val = 0.35 + 0.60 * ((det_id % 50) / 50.0)
            ax.plot(bin_centers, effs, "-", color=cmap(c_val), alpha=0.35, linewidth=1.0)

        for key, (ax, title, cmap) in bar_sub_map.items():
            cat_key = "all_veto" if key == 1 else ("all_us" if key == 2 else ("ds_hor" if key == 30 else "ds_ver"))
            edep_arr = cat_data[cat_key]["edep_kev"]
            qual_arr = cat_data[cat_key]["qualified"]
            if len(edep_arr) > 0:
                effs_cat = []
                for i in range(len(de_bins) - 1):
                    mask = (edep_arr >= de_bins[i]) & (edep_arr < de_bins[i + 1])
                    effs_cat.append(float(np.mean(qual_arr[mask]) * 100.0) if np.sum(mask) >= 8 else np.nan)
                ax.plot(bin_centers, effs_cat, "k-", linewidth=2.5, label="Subsystem Average")

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
        p3_path = os.path.join(args.outdir, "mc_bar_level_efficiency_curves.png")
        plt.savefig(p3_path, dpi=200)
        plt.close()
        print(f"Saved: {p3_path}")

        # Plot 4: Individual SiPMs vs. Full Bar
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle("Individual SiPM Channels vs. Full Bar Hit Qualification S-Curves\n(Illustrates Multi-SiPM Accumulation & Bar-Level Hit Formation via At Least One SiPM)", fontsize=13, weight="bold")

        ax1 = axes[0]
        ed_us = sipm_data["us"]["edep_kev"]
        if len(ed_us) > 0:
            eff_bar, eff_gel, eff_air = [], [], []
            for i in range(len(de_bins) - 1):
                mask = (ed_us >= de_bins[i]) & (ed_us < de_bins[i + 1])
                if np.sum(mask) >= 8:
                    eff_bar.append(float(np.mean(sipm_data["us"]["bar_qual"][mask]) * 100.0))
                    eff_gel.append(float(np.mean(sipm_data["us"]["gel_qual"][mask]) * 100.0))
                    eff_air.append(float(np.mean(sipm_data["us"]["air_qual"][mask]) * 100.0))
                else:
                    eff_bar.append(np.nan)
                    eff_gel.append(np.nan)
                    eff_air.append(np.nan)

            ax1.plot(bin_centers, eff_bar, "k-", linewidth=2.5, label="Full Bar (OR Condition: >= 1 SiPM)")
            ax1.plot(bin_centers, eff_gel, "g--", linewidth=2.0, label="Gel-Coupled SiPM Side")
            ax1.plot(bin_centers, eff_air, "b:", linewidth=2.0, label="Air-Gap SiPM Side")

        ax1.set_title("Upstream Bar (US0-4): Gel vs. Air vs. Full Bar", fontsize=11, weight="bold")
        ax1.set_xlabel("Clustered Energy Deposit $\\Delta E$ [keV]", fontsize=10)
        ax1.set_ylabel("Efficiency [%]", fontsize=10)
        ax1.axhline(50.0, color="gray", linestyle=":", alpha=0.7)
        ax1.set_ylim(-5, 105)
        ax1.set_xlim(0, 450)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(fontsize=9.5, loc="lower right")

        ax2 = axes[1]
        ed_ds = sipm_data["dsh"]["edep_kev"]
        if len(ed_ds) > 0:
            eff_bar, eff_l, eff_r = [], [], []
            for i in range(len(de_bins) - 1):
                mask = (ed_ds >= de_bins[i]) & (ed_ds < de_bins[i + 1])
                if np.sum(mask) >= 8:
                    eff_bar.append(float(np.mean(sipm_data["dsh"]["bar_qual"][mask]) * 100.0))
                    eff_l.append(float(np.mean(sipm_data["dsh"]["left_qual"][mask]) * 100.0))
                    eff_r.append(float(np.mean(sipm_data["dsh"]["right_qual"][mask]) * 100.0))
                else:
                    eff_bar.append(np.nan)
                    eff_l.append(np.nan)
                    eff_r.append(np.nan)

            ax2.plot(bin_centers, eff_bar, "k-", linewidth=2.5, label="Full Bar (OR Condition: Left OR Right)")
            ax2.plot(bin_centers, eff_l, "r--", linewidth=1.8, label="Left SiPM Channel")
            ax2.plot(bin_centers, eff_r, "b:", linewidth=1.8, label="Right SiPM Channel")

        ax2.set_title("DS Horizontal Bar: Left SiPM vs. Right SiPM vs. Full Bar", fontsize=11, weight="bold")
        ax2.set_xlabel("Clustered Energy Deposit $\\Delta E$ [keV]", fontsize=10)
        ax2.set_ylabel("Efficiency [%]", fontsize=10)
        ax2.axhline(50.0, color="gray", linestyle=":", alpha=0.7)
        ax2.set_ylim(-5, 105)
        ax2.set_xlim(0, 250)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(fontsize=9.5, loc="lower right")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p4_path = os.path.join(args.outdir, "mc_sipm_vs_bar_turn_on.png")
        plt.savefig(p4_path, dpi=200)
        plt.close()
        print(f"Saved: {p4_path}")

        # Copy generated files to artifact directory
        if args.artifact_dir:
            import shutil
            saved_plots = [p1_path, p2_path, p3_path, p4_path, root_out_path]
            for p in saved_plots:
                dest = os.path.join(args.artifact_dir, os.path.basename(p))
                shutil.copyfile(p, dest)
                print(f"Copied to artifact directory: {dest}")

        # Save summary stats JSON
        stats_json = os.path.join(args.outdir, "mc_threshold_efficiency_summary.json")
        with open(stats_json, "w") as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Saved summary JSON: {stats_json}")


if __name__ == "__main__":
    run_study()
