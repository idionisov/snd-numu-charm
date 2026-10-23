#!/usr/bin/env python3
"""
generate_hough_event_displays.py
--------------------------------
Generates 2D interactive ROOT Event Displays (TCanvases) using scripts/generate_2DEventDisplays.py
infrastructure for the exact 20 events in each of the three Hough space acceptance directories:
  1. genie_bkg: 20 non-signal background events with 2 reconstructed Hough tracks
  2. pgun_sig:  20 particle gun dimuon signal events
  3. genie_sig: 20 GENIE direct charm dimuon signal events

Outputs:
  - Single combined ROOT file: /eos/user/i/idioniso/snd-numu-charm/plots/hough_space/event_displays.root
    containing subdirectories: genie_bkg, pgun_sig, genie_sig
  - Dedicated ROOT file in each directory:
      /eos/user/i/idioniso/snd-numu-charm/plots/hough_space/genie_bkg/event_displays.root
      /eos/user/i/idioniso/snd-numu-charm/plots/hough_space/pgun_sig/event_displays.root
      /eos/user/i/idioniso/snd-numu-charm/plots/hough_space/genie_sig/event_displays.root
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse
from typing import Dict, List, Tuple, Any

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetPalette(ROOT.kViridis)

from snd import Snd2DEventDisplay


def main():
    parser = argparse.ArgumentParser(description="Generate event displays for Hough space benchmark events")
    parser.add_argument("-o", "--out-base", default="/eos/user/i/idioniso/snd-numu-charm/plots/hough_space",
                        help="Base directory on EOS")
    parser.add_argument("-n", "--n-events", type=int, default=20, help="Number of events per category")
    parser.add_argument("--save-png", action="store_true", default=True, help="Also export PNG images of event displays")
    parser.add_argument("--recoMuons", "--reco-muons", "--muonReco", "--muon-reco",
                        dest="reco_muons", action="store_true", default=True,
                        help="Plot reconstructed muon tracks from Reco_MuonTracks branch")
    args = parser.parse_args()

    os.makedirs(args.out_base, exist_ok=True)
    combined_root_path = os.path.join(args.out_base, "event_displays.root")
    f_combined = ROOT.TFile.Open(combined_root_path, "RECREATE")

    # ---------------------------------------------------------------------------
    # 1. Load Caches
    # ---------------------------------------------------------------------------
    # Background events (20 events)
    with open("pgun/cache_genie_bkg_20ev_2trks.pkl", "rb") as f:
        bkg_cache = pickle.load(f)[:args.n_events]

    # Particle Gun events (20 events)
    with open("pgun/tuning_cache_500ev.pkl", "rb") as f:
        pgun_cache = pickle.load(f)[:args.n_events]

    # Genie Direct Signal events (20 events)
    with open("pgun/tuning_cache_genie_341ev.pkl", "rb") as f:
        genie_sig_cache = pickle.load(f)[:args.n_events]

    categories = [
        ("genie_bkg", bkg_cache),
        ("genie_sig", genie_sig_cache),
        ("pgun_sig",  pgun_cache),
    ]

    genie_geofile = "/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/1/geofile_full.Genie-TGeant4.root"
    pgun_geofile  = "/eos/user/i/idioniso/snd-numu-charm/data/particle_gun_dimuon/geofile_full.TwoMuons-TGeant4.root"

    disp_genie = None
    disp_pgun  = None

    for cat_name, ev_list in categories:
        cat_dir = os.path.join(args.out_base, cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        cat_root_path = os.path.join(cat_dir, "event_displays.root")
        f_cat = ROOT.TFile.Open(cat_root_path, "RECREATE")

        # TDirectory in combined ROOT file
        tdir_combined = f_combined.mkdir(cat_name)

        print(f"\n=======================================================")
        print(f"Generating {len(ev_list)} 2D Event Displays for {cat_name}...")
        print(f"  Target dedicated ROOT file: {cat_root_path}")
        print(f"=======================================================")

        # Select display engine
        if "pgun" in cat_name:
            if disp_pgun is None:
                disp_pgun = Snd2DEventDisplay(geo_file=pgun_geofile)
            display = disp_pgun
        else:
            if disp_genie is None:
                disp_genie = Snd2DEventDisplay(geo_file=genie_geofile)
            display = disp_genie

        for idx, ev in enumerate(ev_list):
            eid_str = str(ev.get("event_id", f"{idx}"))

            # Resolve file path and entry
            if cat_name == "genie_bkg":
                # eid_str is e.g. "1_44"
                part_str, entry_str = eid_str.split("_")
                part_num = int(part_str)
                entry_num = int(entry_str)
                input_file = f"/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/{part_num}/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root"
                canvas_name = f"c_display_{cat_name}_ev{idx+1:02d}_{part_num}_{entry_num}"
                canvas_title = f"SND@LHC {cat_name} Display: ev{idx+1:02d} (Part {part_num}, Entry {entry_num})"
            elif cat_name == "genie_sig":
                # eid_str is e.g. "part1_ev0"
                part_clean = eid_str.split("_")[0].replace("part", "")
                entry_clean = eid_str.split("_")[1].replace("ev", "")
                part_num = int(part_clean)
                entry_num = int(entry_clean)
                input_file = f"/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/{part_num}/sndLHC.Genie-TGeant4_dig_2MuTrks_truth_charmDimuonDirect.root"
                canvas_name = f"c_display_{cat_name}_ev{idx+1:02d}_part{part_num}_ev{entry_num}"
                canvas_title = f"SND@LHC {cat_name} Display: ev{idx+1:02d} (Part {part_num}, Entry {entry_num})"
            else:  # pgun_sig
                part_num = 0
                entry_num = int(eid_str)
                input_file = "/eos/user/i/idioniso/snd-numu-charm/data/particle_gun_dimuon/sndLHC.TwoMuons-TGeant4_dig_2MuTrks.root"
                canvas_name = f"c_display_{cat_name}_ev{idx+1:02d}_entry{entry_num}"
                canvas_title = f"SND@LHC {cat_name} Display: ev{idx+1:02d} (Entry {entry_num})"

            fin = ROOT.TFile.Open(input_file, "READ")
            if not fin or fin.IsZombie():
                print(f"  [ERROR] Could not open {input_file}")
                continue

            tree = fin.Get("cbmsim")
            if not tree:
                print(f"  [ERROR] No cbmsim tree in {input_file}")
                fin.Close()
                continue

            tree.GetEntry(entry_num)

            # Generate 2D interactive event display TCanvas
            canvas = display.draw_event(
                tree,
                event_idx=entry_num,
                canvas_name=canvas_name,
                canvas_title=canvas_title,
                run_number=part_num,
                event_number=entry_num,
                show_mc_truth=True,
                draw_reco_tracks=args.reco_muons,
            )

            # Write to dedicated category ROOT file
            f_cat.cd()
            canvas.Write(canvas_name)

            # Write to combined ROOT file under category TDirectory
            tdir_combined.cd()
            canvas.Write(canvas_name)

            # Optionally save PNG image
            if args.save_png:
                png_path = os.path.join(cat_dir, f"{canvas_name}.png")
                canvas.Print(png_path)

            fin.Close()
            print(f"  [{idx+1}/{len(ev_list)}] Saved {canvas_name}")

        f_cat.Close()
        print(f"Wrote {len(ev_list)} canvases to {cat_root_path}")

    f_combined.Close()
    print(f"\nAll event displays successfully written to combined ROOT file: {combined_root_path}")


if __name__ == "__main__":
    main()
