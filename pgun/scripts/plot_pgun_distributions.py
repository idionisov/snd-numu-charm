#!/usr/bin/env python3
"""
SND@LHC Particle Gun Dimuon Distribution Analysis
Plots distributions of:
  - Muon 1 and Muon 2 energies (individual, sum, ratio, 2D correlation)
  - XZ and YZ angles and 3D opening angle
  - Number of Downstream (DS) MuFilter hits (total, horizontal, vertical, stations)
  - Number of reconstructed DS tracks from Reco_MuonTracks
Saves histograms and formatted canvases into an output ROOT file.
"""

import argparse
import math
import os
import sys
import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)
ROOT.gStyle.SetOptStat(111110)
ROOT.gStyle.SetPalette(ROOT.kViridis)
import SndlhcGeo


def setup_root_style():
    """Applies clean publication styling to ROOT plots."""
    ROOT.gStyle.SetFrameBorderMode(0)
    ROOT.gStyle.SetCanvasBorderMode(0)
    ROOT.gStyle.SetPadBorderMode(0)
    ROOT.gStyle.SetPadColor(ROOT.kWhite)
    ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
    ROOT.gStyle.SetTitleFillColor(ROOT.kWhite)
    ROOT.gStyle.SetTitleBorderSize(0)
    ROOT.gStyle.SetTitleFont(42, "xyz")
    ROOT.gStyle.SetLabelFont(42, "xyz")
    ROOT.gStyle.SetTitleSize(0.045, "xyz")
    ROOT.gStyle.SetLabelSize(0.040, "xyz")
    ROOT.gStyle.SetTitleOffset(1.15, "x")
    ROOT.gStyle.SetTitleOffset(1.25, "y")
    ROOT.gStyle.SetHistLineWidth(2)


def main():
    parser = argparse.ArgumentParser(
        description="Plot kinematic, angular, hit, and track distributions for particle gun dimuon simulation."
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_file",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/data/particle_gun_dimuon/sndLHC.TwoMuons-TGeant4_dig_2MuTrks.root",
        help="Input particle gun ROOT file (default: particle_gun_dimuon/sndLHC.TwoMuons-TGeant4_dig_2MuTrks.root)",
    )
    parser.add_argument(
        "-g", "--geofile",
        dest="geofile",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/data/particle_gun_dimuon/geofile_full.TwoMuons-TGeant4.root",
        help="Geometry ROOT file (default: particle_gun_dimuon/geofile_full.TwoMuons-TGeant4.root)",
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/data/particle_gun_dimuon/pgun_distributions.root",
        help="Output ROOT file for histograms and canvases",
    )
    parser.add_argument(
        "-n", "--max-events",
        dest="max_events",
        type=int,
        default=None,
        help="Maximum number of events to process (default: all)",
    )
    parser.add_argument(
        "--save-png",
        dest="save_png_dir",
        type=str,
        default=None,
        help="Optional directory to save exported PNG images of all canvases",
    )
    args = parser.parse_args()

    setup_root_style()

    if not os.path.exists(args.input_file):
        print(f"[Error] Input file not found: {args.input_file}")
        sys.exit(1)

    print("=" * 78)
    print(" SND@LHC Particle Gun Dimuon Distribution Analysis")
    print("=" * 78)
    print(f" Input File:   {args.input_file}")
    print(f" Output File:  {args.output_file}")
    if args.max_events:
        print(f" Max Events:   {args.max_events}")
    else:
        print(f" Max Events:   All")
    if args.save_png_dir:
        print(f" Save PNGs:    {args.save_png_dir}")
        os.makedirs(args.save_png_dir, exist_ok=True)
    print("=" * 78)

    # Initialize geometry
    if args.geofile and os.path.exists(args.geofile):
        print(f" Initializing geometry from: {args.geofile}")
        geo = SndlhcGeo.GeoInterface(args.geofile)

    fin = ROOT.TFile.Open(args.input_file, "READ")
    if not fin or fin.IsZombie():
        print(f"[Error] Failed to open ROOT file: {args.input_file}")
        sys.exit(1)

    tree = fin.Get("cbmsim")
    if not tree:
        print(f"[Error] TTree 'cbmsim' not found in {args.input_file}")
        fin.Close()
        sys.exit(1)

    total_entries = tree.GetEntries()
    n_to_process = total_entries if args.max_events is None else min(total_entries, args.max_events)
    print(f" Total entries in tree: {total_entries} | Processing: {n_to_process}")

    # Optimize I/O by only reading needed branches
    tree.SetBranchStatus("*", 0)
    for b in ["MCTrack*", "Digi_MuFilterHits*", "Reco_MuonTracks*"]:
        tree.SetBranchStatus(b, 1)

    # =========================================================================
    # Book Histograms
    # =========================================================================
    hists = {}

    # 1. Energies
    hists["h_mu1_energy"] = ROOT.TH1F("h_mu1_energy", "Muon 1 (Track 0) Energy;E(#mu_{1}) [GeV];Events / 10 GeV", 100, 0.0, 1000.0)
    hists["h_mu2_energy"] = ROOT.TH1F("h_mu2_energy", "Muon 2 (Track 1) Energy;E(#mu_{2}) [GeV];Events / 5 GeV", 100, 0.0, 500.0)
    hists["h_leading_mu_energy"] = ROOT.TH1F("h_leading_mu_energy", "Leading Muon Energy;E(#mu_{lead}) [GeV];Events / 10 GeV", 100, 0.0, 1000.0)
    hists["h_subleading_mu_energy"] = ROOT.TH1F("h_subleading_mu_energy", "Subleading Muon Energy;E(#mu_{sublead}) [GeV];Events / 5 GeV", 100, 0.0, 500.0)
    hists["h_dimuon_energy_sum"] = ROOT.TH1F("h_dimuon_energy_sum", "Total Dimuon Energy;E(#mu_{1}) + E(#mu_{2}) [GeV];Events / 10 GeV", 100, 0.0, 1200.0)
    hists["h_dimuon_energy_ratio"] = ROOT.TH1F("h_dimuon_energy_ratio", "Energy Ratio E(#mu_{2}) / E(#mu_{1});E(#mu_{2}) / E(#mu_{1});Events / 0.02", 50, 0.0, 1.0)
    hists["h_dimuon_inv_mass"] = ROOT.TH1F("h_dimuon_inv_mass", "Dimuon Invariant Mass M_{#mu#mu};M_{#mu#mu} [GeV/c^{2}];Events / 0.5 GeV/c^{2}", 60, 0.0, 30.0)
    hists["h2_mu2_vs_mu1_energy"] = ROOT.TH2F("h2_mu2_vs_mu1_energy", "Muon 2 vs Muon 1 Energy;E(#mu_{1}) [GeV];E(#mu_{2}) [GeV]", 50, 0.0, 1000.0, 50, 0.0, 500.0)

    # 2. Angles
    hists["h_mu1_theta_xz"] = ROOT.TH1F("h_mu1_theta_xz", "Muon 1 Angle #theta_{XZ};#theta_{XZ}(#mu_{1}) [mrad];Events / 4 mrad", 100, -200.0, 200.0)
    hists["h_mu1_theta_yz"] = ROOT.TH1F("h_mu1_theta_yz", "Muon 1 Angle #theta_{YZ};#theta_{YZ}(#mu_{1}) [mrad];Events / 4 mrad", 100, -200.0, 200.0)
    hists["h_mu2_theta_xz"] = ROOT.TH1F("h_mu2_theta_xz", "Muon 2 Angle #theta_{XZ};#theta_{XZ}(#mu_{2}) [mrad];Events / 4 mrad", 100, -200.0, 200.0)
    hists["h_mu2_theta_yz"] = ROOT.TH1F("h_mu2_theta_yz", "Muon 2 Angle #theta_{YZ};#theta_{YZ}(#mu_{2}) [mrad];Events / 4 mrad", 100, -200.0, 200.0)
    hists["h_delta_theta_xz"] = ROOT.TH1F("h_delta_theta_xz", "Opening Angle #Delta#theta_{XZ};#Delta#theta_{XZ} = |#theta_{XZ,1} - #theta_{XZ,2}| [mrad];Events / 4 mrad", 75, 0.0, 300.0)
    hists["h_delta_theta_yz"] = ROOT.TH1F("h_delta_theta_yz", "Opening Angle #Delta#theta_{YZ};#Delta#theta_{YZ} = |#theta_{YZ,1} - #theta_{YZ,2}| [mrad];Events / 4 mrad", 75, 0.0, 300.0)
    hists["h_opening_angle_3d"] = ROOT.TH1F("h_opening_angle_3d", "3D Opening Angle #theta_{12};#theta_{12} [mrad];Events / 4 mrad", 75, 0.0, 300.0)
    hists["h2_delta_theta_yz_vs_xz"] = ROOT.TH2F("h2_delta_theta_yz_vs_xz", "#Delta#theta_{YZ} vs #Delta#theta_{XZ};#Delta#theta_{XZ} [mrad];#Delta#theta_{YZ} [mrad]", 50, 0.0, 250.0, 50, 0.0, 250.0)

    # 3. Number of DS Hits
    hists["h_n_ds_hits_total"] = ROOT.TH1F("h_n_ds_hits_total", "Total Downstream (DS) MuFilter Hits;Number of DS Hits;Events", 60, -0.5, 59.5)
    hists["h_n_ds_hits_hor"] = ROOT.TH1F("h_n_ds_hits_hor", "Horizontal DS MuFilter Hits;Number of Horizontal DS Hits;Events", 40, -0.5, 39.5)
    hists["h_n_ds_hits_ver"] = ROOT.TH1F("h_n_ds_hits_ver", "Vertical DS MuFilter Hits;Number of Vertical DS Hits;Events", 40, -0.5, 39.5)
    hists["h_n_ds_stations_fired"] = ROOT.TH1F("h_n_ds_stations_fired", "Fired DS Stations Count;Number of Fired DS Stations;Events", 6, -0.5, 5.5)
    hists["h2_ds_hits_ver_vs_hor"] = ROOT.TH2F("h2_ds_hits_ver_vs_hor", "DS Vertical vs Horizontal Hits;Horizontal DS Hits;Vertical DS Hits", 35, -0.5, 34.5, 35, -0.5, 34.5)

    # 4. Number of Reconstructed DS Tracks
    hists["h_n_reco_ds_tracks"] = ROOT.TH1F("h_n_reco_ds_tracks", "Reconstructed DS Tracks (Reco_MuonTracks);Number of Reco DS Tracks;Events", 8, -0.5, 7.5)
    hists["h2_reco_tracks_vs_ds_hits"] = ROOT.TH2F("h2_reco_tracks_vs_ds_hits", "Reco DS Tracks vs Total DS Hits;Total DS Hits;Number of Reco DS Tracks", 40, -0.5, 39.5, 6, -0.5, 5.5)
    hists["h2_reco_tracks_vs_opening_angle"] = ROOT.TH2F("h2_reco_tracks_vs_opening_angle", "Reco DS Tracks vs 3D Opening Angle;#theta_{12} [mrad];Number of Reco DS Tracks", 40, 0.0, 200.0, 6, -0.5, 5.5)

    # Style colors
    hists["h_mu1_energy"].SetLineColor(ROOT.kBlue + 1)
    hists["h_mu2_energy"].SetLineColor(ROOT.kRed + 1)
    hists["h_leading_mu_energy"].SetLineColor(ROOT.kAzure + 2)
    hists["h_subleading_mu_energy"].SetLineColor(ROOT.kOrange + 7)
    hists["h_dimuon_energy_sum"].SetLineColor(ROOT.kGreen + 2)
    hists["h_dimuon_inv_mass"].SetLineColor(ROOT.kMagenta + 2)
    hists["h_delta_theta_xz"].SetLineColor(ROOT.kBlue + 2)
    hists["h_delta_theta_yz"].SetLineColor(ROOT.kRed + 2)
    hists["h_opening_angle_3d"].SetLineColor(ROOT.kBlack)
    hists["h_n_ds_hits_total"].SetLineColor(ROOT.kBlack)
    hists["h_n_ds_hits_hor"].SetLineColor(ROOT.kBlue + 1)
    hists["h_n_ds_hits_ver"].SetLineColor(ROOT.kRed + 1)
    hists["h_n_reco_ds_tracks"].SetLineColor(ROOT.kAzure + 1)
    hists["h_n_reco_ds_tracks"].SetFillColor(ROOT.kAzure - 9)

    # =========================================================================
    # Event Loop
    # =========================================================================
    for iev in range(n_to_process):
        tree.GetEntry(iev)

        if (iev + 1) % 1000 == 0 or (iev + 1) == n_to_process:
            print(f" Processing event [{iev + 1}/{n_to_process}]...")

        # ---------------------------------------------------------------------
        # 1. Truth Muons from MCTrack
        # ---------------------------------------------------------------------
        pg_muons = []
        if hasattr(tree, "MCTrack"):
            n_mc = tree.MCTrack.GetEntries()
            for i_mc in range(n_mc):
                trk = tree.MCTrack[i_mc]
                if abs(trk.GetPdgCode()) == 13 and trk.GetMotherId() < 0:
                    pg_muons.append(trk)

        if len(pg_muons) >= 2:
            m1, m2 = pg_muons[0], pg_muons[1]
            e1 = float(m1.GetEnergy())
            e2 = float(m2.GetEnergy())
            lead_e = max(e1, e2)
            sublead_e = min(e1, e2)

            hists["h_mu1_energy"].Fill(e1)
            hists["h_mu2_energy"].Fill(e2)
            hists["h_leading_mu_energy"].Fill(lead_e)
            hists["h_subleading_mu_energy"].Fill(sublead_e)
            hists["h_dimuon_energy_sum"].Fill(e1 + e2)
            if max(e1, e2) > 0:
                hists["h_dimuon_energy_ratio"].Fill(min(e1, e2) / max(e1, e2))
            hists["h2_mu2_vs_mu1_energy"].Fill(e1, e2)

            # Invariant Mass
            lv1 = ROOT.TLorentzVector(m1.GetPx(), m1.GetPy(), m1.GetPz(), m1.GetEnergy())
            lv2 = ROOT.TLorentzVector(m2.GetPx(), m2.GetPy(), m2.GetPz(), m2.GetEnergy())
            hists["h_dimuon_inv_mass"].Fill((lv1 + lv2).M())

            # Angles
            pz1 = m1.GetPz()
            pz2 = m2.GetPz()
            if abs(pz1) > 1e-4 and abs(pz2) > 1e-4:
                th_xz1 = math.atan2(m1.GetPx(), pz1) * 1000.0
                th_yz1 = math.atan2(m1.GetPy(), pz1) * 1000.0
                th_xz2 = math.atan2(m2.GetPx(), pz2) * 1000.0
                th_yz2 = math.atan2(m2.GetPy(), pz2) * 1000.0

                d_th_xz = abs(th_xz1 - th_xz2)
                d_th_yz = abs(th_yz1 - th_yz2)

                hists["h_mu1_theta_xz"].Fill(th_xz1)
                hists["h_mu1_theta_yz"].Fill(th_yz1)
                hists["h_mu2_theta_xz"].Fill(th_xz2)
                hists["h_mu2_theta_yz"].Fill(th_yz2)
                hists["h_delta_theta_xz"].Fill(d_th_xz)
                hists["h_delta_theta_yz"].Fill(d_th_yz)
                hists["h2_delta_theta_yz_vs_xz"].Fill(d_th_xz, d_th_yz)

            p1 = ROOT.TVector3(m1.GetPx(), m1.GetPy(), m1.GetPz())
            p2 = ROOT.TVector3(m2.GetPx(), m2.GetPy(), m2.GetPz())
            if p1.Mag() > 0 and p2.Mag() > 0:
                op_angle_3d = float(p1.Angle(p2) * 1000.0)
                hists["h_opening_angle_3d"].Fill(op_angle_3d)
        else:
            op_angle_3d = -1.0

        # ---------------------------------------------------------------------
        # 2. Number of DS Hits from Digi_MuFilterHits
        # ---------------------------------------------------------------------
        n_ds_total = 0
        n_ds_hor = 0
        n_ds_ver = 0
        ds_stations = set()

        if hasattr(tree, "Digi_MuFilterHits"):
            n_hits = tree.Digi_MuFilterHits.GetEntries()
            for i_h in range(n_hits):
                hit = tree.Digi_MuFilterHits[i_h]
                if hasattr(hit, "isValid") and not hit.isValid():
                    continue
                if hit.GetSystem() == 3:  # Downstream system
                    n_ds_total += 1
                    plane = (hit.GetDetectorID() % 10000) // 1000
                    ds_stations.add(plane)
                    if hit.isVertical():
                        n_ds_ver += 1
                    else:
                        n_ds_hor += 1

        hists["h_n_ds_hits_total"].Fill(n_ds_total)
        hists["h_n_ds_hits_hor"].Fill(n_ds_hor)
        hists["h_n_ds_hits_ver"].Fill(n_ds_ver)
        hists["h_n_ds_stations_fired"].Fill(len(ds_stations))
        hists["h2_ds_hits_ver_vs_hor"].Fill(n_ds_hor, n_ds_ver)

        # ---------------------------------------------------------------------
        # 3. Number of Reconstructed DS Tracks from Reco_MuonTracks
        # ---------------------------------------------------------------------
        n_reco_ds_tracks = 0
        if hasattr(tree, "Reco_MuonTracks"):
            n_trks = tree.Reco_MuonTracks.GetEntries()
            for i_trk in range(n_trks):
                trk = tree.Reco_MuonTracks[i_trk]
                # In sndsw, DS Hough tracks have trackType == 13 or 3
                t_type = None
                if hasattr(trk, "getTrackType"):
                    t_type = trk.getTrackType()
                elif hasattr(trk, "GetUniqueID"):
                    t_type = trk.GetUniqueID()

                if t_type in (3, 13):
                    n_reco_ds_tracks += 1

        hists["h_n_reco_ds_tracks"].Fill(n_reco_ds_tracks)
        hists["h2_reco_tracks_vs_ds_hits"].Fill(n_ds_total, n_reco_ds_tracks)
        if op_angle_3d >= 0.0:
            hists["h2_reco_tracks_vs_opening_angle"].Fill(op_angle_3d, n_reco_ds_tracks)

    # =========================================================================
    # Build Formatted Summary Canvases
    # =========================================================================
    canvases = []

    # Canvas 1: Muon Energies
    c_energies = ROOT.TCanvas("c_energies", "Particle Gun Muon Energies", 1200, 800)
    c_energies.Divide(2, 2)
    c_energies.cd(1)
    hists["h_mu1_energy"].Draw("HIST")
    c_energies.cd(2)
    hists["h_mu2_energy"].Draw("HIST")
    c_energies.cd(3)
    hists["h_dimuon_energy_sum"].Draw("HIST")
    c_energies.cd(4)
    hists["h2_mu2_vs_mu1_energy"].Draw("COLZ")
    canvases.append(c_energies)

    # Canvas 2: Angles
    c_angles = ROOT.TCanvas("c_angles", "Particle Gun Angles", 1200, 800)
    c_angles.Divide(2, 2)
    c_angles.cd(1)
    hists["h_delta_theta_xz"].Draw("HIST")
    c_angles.cd(2)
    hists["h_delta_theta_yz"].Draw("HIST")
    c_angles.cd(3)
    hists["h_opening_angle_3d"].Draw("HIST")
    c_angles.cd(4)
    hists["h2_delta_theta_yz_vs_xz"].Draw("COLZ")
    canvases.append(c_angles)

    # Canvas 3: DS Hits
    c_ds_hits = ROOT.TCanvas("c_ds_hits", "Particle Gun DS Hits", 1200, 800)
    c_ds_hits.Divide(2, 2)
    c_ds_hits.cd(1)
    hists["h_n_ds_hits_total"].Draw("HIST")
    c_ds_hits.cd(2)
    hists["h_n_ds_hits_hor"].Draw("HIST")
    c_ds_hits.cd(3)
    hists["h_n_ds_hits_ver"].Draw("HIST")
    c_ds_hits.cd(4)
    hists["h_n_ds_stations_fired"].Draw("HIST")
    canvases.append(c_ds_hits)

    # Canvas 4: Reconstructed Tracks
    c_reco = ROOT.TCanvas("c_reco_tracks", "Particle Gun Reconstructed DS Tracks", 1200, 800)
    c_reco.Divide(2, 2)
    c_reco.cd(1)
    hists["h_n_reco_ds_tracks"].Draw("HIST")
    c_reco.cd(2)
    hists["h2_reco_tracks_vs_ds_hits"].Draw("COLZ")
    c_reco.cd(3)
    hists["h2_reco_tracks_vs_opening_angle"].Draw("COLZ")
    c_reco.cd(4)
    hists["h_dimuon_inv_mass"].Draw("HIST")
    canvases.append(c_reco)

    # Optionally export PNGs
    if args.save_png_dir:
        for c in canvases:
            png_path = os.path.join(args.save_png_dir, f"{c.GetName()}.png")
            c.SaveAs(png_path)
            print(f" Saved PNG: {png_path}")

    # =========================================================================
    # Write to Output ROOT File
    # =========================================================================
    fout = ROOT.TFile.Open(args.output_file, "RECREATE")
    if not fout or fout.IsZombie():
        print(f"[Error] Failed to create output ROOT file: {args.output_file}")
        sys.exit(1)

    dir_kinematics = fout.mkdir("kinematics")
    dir_angles = fout.mkdir("angles")
    dir_ds_hits = fout.mkdir("ds_hits")
    dir_reco_tracks = fout.mkdir("reco_tracks")
    dir_canvases = fout.mkdir("canvases")

    dir_kinematics.cd()
    for name in [
        "h_mu1_energy", "h_mu2_energy", "h_leading_mu_energy", "h_subleading_mu_energy",
        "h_dimuon_energy_sum", "h_dimuon_energy_ratio", "h_dimuon_inv_mass", "h2_mu2_vs_mu1_energy"
    ]:
        hists[name].Write()

    dir_angles.cd()
    for name in [
        "h_mu1_theta_xz", "h_mu1_theta_yz", "h_mu2_theta_xz", "h_mu2_theta_yz",
        "h_delta_theta_xz", "h_delta_theta_yz", "h_opening_angle_3d", "h2_delta_theta_yz_vs_xz"
    ]:
        hists[name].Write()

    dir_ds_hits.cd()
    for name in [
        "h_n_ds_hits_total", "h_n_ds_hits_hor", "h_n_ds_hits_ver",
        "h_n_ds_stations_fired", "h2_ds_hits_ver_vs_hor"
    ]:
        hists[name].Write()

    dir_reco_tracks.cd()
    for name in [
        "h_n_reco_ds_tracks", "h2_reco_tracks_vs_ds_hits", "h2_reco_tracks_vs_opening_angle"
    ]:
        hists[name].Write()

    dir_canvases.cd()
    for c in canvases:
        c.Write()

    fout.Close()
    fin.Close()
    print("=" * 78)
    print(f" Analysis complete! All histograms and canvases saved to:")
    print(f"   {os.path.abspath(args.output_file)}")
    print(" Subdirectories in ROOT file:")
    print("   - kinematics/   (muon energies, invariant mass, correlations)")
    print("   - angles/       (XZ and YZ slopes, opening angles)")
    print("   - ds_hits/      (total, hor, ver DS hits, fired stations)")
    print("   - reco_tracks/  (reco DS track multiplicity and 2D correlations)")
    print("   - canvases/     (publication-ready 4-pad TCanvases)")
    print("=" * 78)


if __name__ == "__main__":
    main()
