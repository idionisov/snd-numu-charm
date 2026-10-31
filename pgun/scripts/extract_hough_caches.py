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


DEFAULT_GEOFILE = "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root"


def resolve_geofile(file_or_dir: str, user_geofile: str = None) -> str:
    """Resolve geometry file:
    1. If explicitly provided, use it.
    2. Check the file's directory (or parent directory) for matching 'geofile*.root'.
    3. Check corresponding partition 1 in /eos/user/i/idioniso/snd-numu-charm/data/newP8/1/.
    4. Fallback to DEFAULT_GEOFILE.
    """
    if user_geofile and os.path.exists(user_geofile):
        return user_geofile

    target_dir = file_or_dir if os.path.isdir(file_or_dir) else os.path.dirname(os.path.abspath(file_or_dir))

    # Search in target directory and up to 2 parent levels
    check_dir = target_dir
    for _ in range(3):
        if os.path.isdir(check_dir):
            try:
                for fname in os.listdir(check_dir):
                    if fname.startswith("geofile") and fname.endswith(".root"):
                        full_path = os.path.join(check_dir, fname)
                        if os.path.exists(full_path):
                            return full_path
            except Exception:
                pass
        parent = os.path.dirname(check_dir)
        if parent == check_dir:
            break
        check_dir = parent

    # Check newP8 partition 1 default
    newp8_geo = "/eos/user/i/idioniso/snd-numu-charm/data/newP8/1/geofile_full.Genie-TGeant4.root"
    if os.path.exists(newp8_geo):
        return newp8_geo

    return DEFAULT_GEOFILE


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


def extract_pmu_background(out_cache="pgun/cache_pmu_bkg_500ev.pkl", max_events=500, geo_file=None):
    pmu_file = "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP_2MuTrks_truth.root"
    geo_path = resolve_geofile(pmu_file, geo_file)
    print(f"Extracting pmu background from {pmu_file} using {geo_path}...")
    geo = SndlhcGeo.GeoInterface(geo_path)
    bar_geo = get_bar_geometry(geo)

    f = ROOT.TFile.Open(pmu_file)
    tree = f.Get("cbmsim")
    events = extract_events_from_tree(tree, bar_geo, is_signal=False, max_events=max_events)
    f.Close()

    with open(out_cache, "wb") as f_out:
        pickle.dump(events, f_out)
    print(f"Saved {len(events)} pmu background events to {out_cache}.")
    return events


def extract_genie_background(out_cache="pgun/cache_genie_bkg_500ev.pkl", max_events=500, geo_file=None):
    geo_path = resolve_geofile("/eos/user/i/idioniso/snd-numu-charm/data/newP8/1", geo_file)
    print(f"Extracting Genie background across partitions using {geo_path}...")
    geo = SndlhcGeo.GeoInterface(geo_path)
    bar_geo = get_bar_geometry(geo)

    events = []
    z_first = 430.0
    z_last = 520.0
    det_zlen = z_last - z_first

    for part in range(1, 1000):
        if len(events) >= max_events:
            break
        file_path = f"/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8/houghTuning-nonsig/{part}/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root"
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


def extract_genie_signal(
    out_cache="pgun/tuning_cache_genie_341ev.pkl",
    max_events=500,
    partitions_dir="/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8/essentials-direct",
    geo_file=None,
    min_ds_hits=6,
):
    geo_path = resolve_geofile(partitions_dir, geo_file)
    print(f"Extracting GENIE signal from {partitions_dir} using {geo_path} ...")
    geo = SndlhcGeo.GeoInterface(geo_path)
    bar_geo = get_bar_geometry(geo)

    events = []
    z_first = 430.0
    z_last = 520.0
    det_zlen = z_last - z_first

    for part in range(1, 2000):
        if len(events) >= max_events:
            break
        file_path = os.path.join(partitions_dir, str(part),
                                 "sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root")
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
            # Keep only direct charm signal events
            if mctruth:
                mctruth.GetEntry(i_ev)
                if not getattr(mctruth, "is_direct_charm_decay", 0):
                    continue

            tree.GetEntry(i_ev)
            mufi_hits = getattr(tree, "Digi_MuFilterHits", None)
            if not mufi_hits:
                continue

            pos_list, d_list, vert_list, sys_list, detid_list = [], [], [], [], []
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

            if len(pos_list) < min_ds_hits:
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

    os.makedirs(os.path.dirname(os.path.abspath(out_cache)), exist_ok=True)
    with open(out_cache, "wb") as f_out:
        pickle.dump(events, f_out)
    print(f"Saved {len(events)} GENIE signal events to {out_cache}.")
    return events


def extract_pgun_signal(
    out_cache="pgun/tuning_cache_500ev.pkl",
    max_events=500,
    pgun_file="/eos/user/i/idioniso/snd-numu-charm/data/pgun_antiD0_test/sndLHC.PG_-421-TGeant4.root",
    geo_file=None,
    min_ds_hits=6,
):
    """Extract particle-gun anti-D0 dimuon signal events."""
    geo_path = resolve_geofile(pgun_file, geo_file)
    print(f"Extracting particle-gun signal from {pgun_file} using {geo_path} ...")
    geo = SndlhcGeo.GeoInterface(geo_path)
    bar_geo = get_bar_geometry(geo)

    f = ROOT.TFile.Open(pgun_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open {pgun_file}")
    tree = f.Get("cbmsim")
    if not tree:
        f.Close()
        raise RuntimeError(f"No 'cbmsim' tree in {pgun_file}")

    events = extract_events_from_tree(
        tree, bar_geo, is_signal=True, max_events=max_events, min_ds_hits=min_ds_hits
    )
    f.Close()

    os.makedirs(os.path.dirname(os.path.abspath(out_cache)), exist_ok=True)
    with open(out_cache, "wb") as f_out:
        pickle.dump(events, f_out)
    print(f"Saved {len(events)} particle-gun signal events to {out_cache}.")
    return events


if __name__ == "__main__":
    import argparse

    EOS_CACHE = "/eos/user/i/idioniso/snd-numu-charm/cache/pgun"

    parser = argparse.ArgumentParser(description="Extract Hough tuning cache files from ROOT inputs.")
    parser.add_argument("--genie-sig",  action="store_true", help="Extract GENIE direct signal cache")
    parser.add_argument("--pgun-sig",   action="store_true", help="Extract particle-gun signal cache")
    parser.add_argument("--genie-bkg",  action="store_true", help="Extract GENIE background cache")
    parser.add_argument("--pmu-bkg",    action="store_true", help="Extract pmu2022 background cache")
    parser.add_argument("--all",        action="store_true", help="Extract all four caches")
    parser.add_argument("--out-dir",    default=EOS_CACHE,   help=f"Output directory (default: {EOS_CACHE})")
    parser.add_argument("--max-events", type=int, default=500, help="Max events per cache (default: 500)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    do_all = args.all or not any([args.genie_sig, args.pgun_sig, args.genie_bkg, args.pmu_bkg])

    if do_all or args.genie_sig:
        extract_genie_signal(
            out_cache=os.path.join(args.out_dir, "tuning_cache_genie_341ev.pkl"),
            max_events=args.max_events,
        )
    if do_all or args.pgun_sig:
        extract_pgun_signal(
            out_cache=os.path.join(args.out_dir, "tuning_cache_500ev.pkl"),
            max_events=args.max_events,
        )
    if do_all or args.genie_bkg:
        extract_genie_background(
            out_cache=os.path.join(args.out_dir, "cache_genie_bkg_500ev.pkl"),
            max_events=args.max_events,
        )
    if do_all or args.pmu_bkg:
        extract_pmu_background(
            out_cache=os.path.join(args.out_dir, "cache_pmu_bkg_500ev.pkl"),
            max_events=args.max_events,
        )
