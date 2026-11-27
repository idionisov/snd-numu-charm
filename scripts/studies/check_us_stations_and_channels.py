#!/usr/bin/env python3
import os
import ROOT
import numpy as np
import SndlhcGeo
import SndlhcTracking

def main():
    geofile = '/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/geofile_full.Ntuple-TGeant4.root'
    geo = SndlhcGeo.GeoInterface(geofile)
    mufi = geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()

    bars = {}
    for pl in range(5):
        for b in range(10):
            did = 20000 + pl * 1000 + b
            mufi.GetPosition(did, A, B)
            bars[did] = {
                'pl': pl, 'b': b,
                'x_min': min(A.X(), B.X()), 'x_max': max(A.X(), B.X()),
                'y_min': min(A.Y(), B.Y()), 'y_max': max(A.Y(), B.Y()),
                'z': 0.5 * (A.Z() + B.Z())
            }

    data_file = "/eos/experiment/sndlhc/convertedData/physics/2022/run_005408/sndsw_raw-0002.root"
    chain = ROOT.TChain("rawConv")
    chain.Add(data_file)

    ioman = ROOT.FairRootManager.Instance()
    ioman.SetInChain(chain)
    trackTask = SndlhcTracking.Tracking()
    trackTask.SetName('simpleTracking')
    trackTask.SetTrackClassType(False)
    trackTask.Init()

    US_GEL_CONFIG = {
        0: ('no GEL', 'with GEL'),   # Plane 0: L=Air, R=Gel
        1: ('with GEL', 'no GEL'),   # Plane 1: L=Gel, R=Air
        2: ('no GEL', 'with GEL'),   # Plane 2: L=Air, R=Gel
        3: ('with GEL', 'no GEL'),   # Plane 3: L=Gel, R=Air
        4: ('with GEL', 'no GEL')    # Plane 4: L=Gel, R=Air
    }

    # Data records: (plane, side, channel, bar, qdc, x_track, y_track)
    records = []

    n_tot = min(4000, chain.GetEntries())
    prev_time = None

    for iev in range(n_tot):
        chain.GetEvent(iev)
        h = chain.EventHeader
        if not h.isIP1() or h.GetBeamMode() != 11: continue
        t = h.GetEventTime()
        if prev_time is not None and abs(t - prev_time) <= 100:
            prev_time = t
            continue
        prev_time = t

        trackTask.ExecuteTask(option='ScifiDS')
        dtrk = None
        for trk in trackTask.fittedTracks:
            if trk.getTrackType() == 3:
                if trk.getChi2Ndf() < 5.0 and abs(trk.getSlopeXZ()) < 0.1 and abs(trk.getSlopeYZ()) < 0.1:
                    dtrk = trk
                    break
        if dtrk is None: continue

        hits_map = {hit.GetDetectorID(): hit for hit in chain.Digi_MuFilterHits if hit.isValid()}
        for pl in range(5):
            for b in range(10):
                did = 20000 + pl * 1000 + b
                if did not in hits_map: continue
                b_info = bars[did]
                p_trk = dtrk.getPointAtZ(b_info['z'])
                y_c = 0.5 * (b_info['y_min'] + b_info['y_max'])
                if abs(p_trk.Y() - y_c) <= 3.0 and (b_info['x_min'] - 1.0 <= p_trk.X() <= b_info['x_max'] + 1.0):
                    hit = hits_map[did]
                    for ch, q in hit.GetAllSignals():
                        if q <= 0.0 or ch % 8 in (2, 5): continue
                        side = 0 if ch < 8 else 1
                        records.append((pl, side, ch, b, q, p_trk.X(), p_trk.Y()))

    print(f"Extracted {len(records)} track-matched US hit readouts.")
    rec_arr = np.array(records, dtype=[('pl', 'i4'), ('side', 'i4'), ('ch', 'i4'), ('bar', 'i4'),
                                       ('qdc', 'f4'), ('x_trk', 'f4'), ('y_trk', 'f4')])
    np.save('/afs/cern.ch/work/i/idioniso/snd-numu-charm/scratch/us_detailed_records.npy', rec_arr)
    print("Saved to scratch/us_detailed_records.npy")

if __name__ == '__main__':
    main()
