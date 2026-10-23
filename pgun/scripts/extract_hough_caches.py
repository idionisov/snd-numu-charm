#!/usr/bin/env python3
"""
extract_hough_caches.py
-----------------------
Extracts hit collections for:
1. Genie Direct Signal (both muons in DS acceptance)
2. Genie Background (inclusive non-signal)
3. pmu2022 Background (passing muons)
4. Particle Gun Dimuon Signal
"""

import os
import sys
import pickle
import numpy as np
import ROOT
import shipunit as u
import SndlhcGeo


def get_bar_geometry(geo):
    """Pre-cache positions and dimensions for all DS bars (MuFilter System 3)."""
    a = ROOT.TVector3()
    b = ROOT.TVector3()
    mf_mod = geo.modules["MuFilter"]
    bar_geo = {}
    for plane in range(4):
        if plane < 3:
            for bar in range(60):
                det_id = 30000 + plane * 1000 + bar
                try:
                    mf_mod.GetPosition(det_id, a, b)
                    center = (a + b) * 0.5
                    dx = 6.0
                    dy = abs(a.Y() - b.Y()) if abs(a.Y() - b.Y()) > 0.1 else 6.0
                    dz = 6.0
                    bar_geo[det_id] = (center.X(), center.Y(), center.Z(), dx, dy, dz, False)
                except Exception:
                    pass

        for bar in range(60, 120):
            det_id = 30000 + plane * 1000 + bar
            try:
                mf_mod.GetPosition(det_id, a, b)
                center = (a + b) * 0.5
                dx = abs(a.X() - b.X()) if abs(a.X() - b.X()) > 0.1 else 6.0
                dy = 6.0
                dz = 6.0
                bar_geo[det_id] = (center.X(), center.Y(), center.Z(), dx, dy, dz, True)
            except Exception:
                pass
    return bar_geo


def extract_events_from_tree(tree, bar_geo, is_signal=False, max_events=500, min_ds_hits=6):
    events = []
    z_first = 430.0
    z_last = 520.0
    det_zlen = z_last - z_first

    n_entries = tree.GetEntries()
    for i_ev in range(n_entries):
        if len(events) >= max_events:
            break
        tree.GetEntry(i_ev)

        # Truth check for signal vs background if mctruth tree or branch is present
        muons = []
        if is_signal and hasattr(tree, "MCTrack"):
            for trk in tree.MCTrack:
                if abs(trk.GetPdgCode()) == 13 and trk.GetMotherId() < 0:
                    p = ROOT.TVector3(trk.GetPx(), trk.GetPy(), trk.GetPz())
                    if p.Pz() > 0:
                        muons.append({
                            "tx": p.Px() / p.Pz(),
                            "ty": p.Py() / p.Pz(),
                            "E": trk.GetEnergy(),
                        })

        mufi_hits = getattr(tree, "Digi_MuFilterHits", None)
        if not mufi_hits:
            continue

        pos_list = []
        d_list = []
        vert_list = []
        sys_list = []
        detid_list = []

        for hit in mufi_hits:
            if not hit.isValid():
                continue
            det_id = hit.GetDetectorID()
            subsys = det_id // 10000
            if subsys != 3:
                continue

            geo_info = bar_geo.get(det_id)
            if not geo_info:
                continue

            cx, cy, cz, dx, dy, dz, is_vert = geo_info
            pos_list.append([cx, cy, cz])
            d_list.append([dx, dy, dz])
            vert_list.append(is_vert)
            sys_list.append(subsys)
            detid_list.append(det_id)

        if len(pos_list) < min_ds_hits:
            continue

        events.append({
            "event_id": i_ev,
            "true_muons": muons,
            "pos": np.array(pos_list).T,  # (3, N)
            "d": np.array(d_list).T,      # (3, N)
            "vert": np.array(vert_list),
            "system": np.array(sys_list),
            "det_id": np.array(detid_list),
            "z_offset": z_first,
            "det_zlen": det_zlen,
        })

    return events


def extract_pmu_background(out_cache="pgun/cache_pmu_bkg_500ev.pkl", max_events=500):
    pmu_file = "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP_2MuTrks_truth.root"
    geo_file = "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/geofile_full.Ntuple-TGeant4.root"
    print(f"Extracting pmu background from {pmu_file}...")
    geo = SndlhcGeo.GeoInterface(geo_file)
    bar_geo = get_bar_geometry(geo)

    f = ROOT.TFile.Open(pmu_file)
    tree = f.Get("cbmsim")
    events = extract_events_from_tree(tree, bar_geo, is_signal=False, max_events=max_events)
    f.Close()

    with open(out_cache, "wb") as f_out:
        pickle.dump(events, f_out)
    print(f"Saved {len(events)} pmu background events to {out_cache}.")
    return events


def extract_genie_background(out_cache="pgun/cache_genie_bkg_500ev.pkl", max_events=500):
    geo_file = "/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/1/geofile_full.Genie-TGeant4.root"
    print(f"Extracting Genie background across partitions...")
    geo = SndlhcGeo.GeoInterface(geo_file)
    bar_geo = get_bar_geometry(geo)

    events = []
    z_first = 430.0
    z_last = 520.0
    det_zlen = z_last - z_first

    for part in range(1, 100):
        if len(events) >= max_events:
            break
        file_path = f"/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/{part}/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root"
        if not os.path.exists(file_path):
            continue

        f = ROOT.TFile.Open(file_path)
        if not f or f.IsZombie():
            continue
        tree = f.Get("cbmsim")
        mctruth = f.Get("mctruth")
        if not tree:
            f.Close()
            continue

        n_entries = tree.GetEntries()
        for i_ev in range(n_entries):
            if len(events) >= max_events:
                break
            if mctruth:
                mctruth.GetEntry(i_ev)
                # Skip direct charm signal events
                if getattr(mctruth, "is_direct_charm_decay", 0):
                    continue

            tree.GetEntry(i_ev)
            mufi_hits = getattr(tree, "Digi_MuFilterHits", None)
            if not mufi_hits:
                continue

            pos_list = []
            d_list = []
            vert_list = []
            sys_list = []
            detid_list = []

            for hit in mufi_hits:
                if not hit.isValid():
                    continue
                det_id = hit.GetDetectorID()
                if det_id // 10000 != 3:
                    continue
                geo_info = bar_geo.get(det_id)
                if not geo_info:
                    continue

                cx, cy, cz, dx, dy, dz, is_vert = geo_info
                pos_list.append([cx, cy, cz])
                d_list.append([dx, dy, dz])
                vert_list.append(is_vert)
                sys_list.append(3)
                detid_list.append(det_id)

            if len(pos_list) < 6:
                continue

            events.append({
                "event_id": f"{part}_{i_ev}",
                "true_muons": [],
                "pos": np.array(pos_list).T,
                "d": np.array(d_list).T,
                "vert": np.array(vert_list),
                "system": np.array(sys_list),
                "det_id": np.array(detid_list),
                "z_offset": z_first,
                "det_zlen": det_zlen,
            })
        f.Close()

    with open(out_cache, "wb") as f_out:
        pickle.dump(events, f_out)
    print(f"Saved {len(events)} Genie background events to {out_cache}.")
    return events


if __name__ == "__main__":
    os.makedirs("pgun", exist_ok=True)
    extract_pmu_background()
    extract_genie_background()
