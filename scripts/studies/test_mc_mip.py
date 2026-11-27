import ROOT
import numpy as np
import SndlhcGeo
import SndlhcTracking

ROOT.gROOT.SetBatch(True)

mc_file = "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP.root"
geofile = "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/geofile_full.Ntuple-TGeant4.root"

geo = SndlhcGeo.GeoInterface(geofile)
mufi = geo.modules['MuFilter']

# Map DS horizontal bars (sub 3, planes 0..2, bars 0..59)
ds_bars = {}
A, B = ROOT.TVector3(), ROOT.TVector3()
for pl in range(3):
    for b in range(60):
        did = 30000 + pl * 1000 + b
        mufi.GetPosition(did, A, B)
        ds_bars[did] = {
            'plane': pl, 'bar': b,
            'x_min': min(A.X(), B.X()), 'x_max': max(A.X(), B.X()),
            'x_center': 0.5 * (A.X() + B.X()),
            'y_center': 0.5 * (A.Y() + B.Y()),
            'z': 0.5 * (A.Z() + B.Z()),
        }

chain = ROOT.TChain("cbmsim")
chain.Add(mc_file)
chain_mc = ROOT.TChain("mctruth")
chain_mc.Add(mc_file)
chain.AddFriend(chain_mc)

ioman = ROOT.FairRootManager.Instance()
ioman.SetInChain(chain)

trackTask = SndlhcTracking.Tracking()
trackTask.SetName("simpleTracking")
trackTask.SetTrackClassType(False)
trackTask.Init()

h_eloss_center = ROOT.TH1F("h_eloss_center", "DS Hor True Edep (Central);E_{dep} [MeV];Entries", 100, 0, 10)
h_eloss_all = ROOT.TH1F("h_eloss_all", "DS Hor True Edep (All x);E_{dep} [MeV];Entries", 100, 0, 10)

n_entries = min(2000, chain.GetEntries())
n_passed = 0

for iev in range(n_entries):
    chain.GetEvent(iev)
    raw_w = float(getattr(chain, "mc_weight", 0.0))
    if raw_w <= 0: continue

    # Multiplicity cuts
    nsf = len(chain.Digi_ScifiHits)
    if not (6 <= nsf <= 27): continue

    # Hit counts with physical thresholds
    bar_edep = {}
    for pt in chain.MuFilterPoint:
        did = pt.GetDetectorID()
        bar_edep[did] = bar_edep.get(did, 0.0) + pt.GetEnergyLoss() * 1000.0

    nv, nu, nd = 0, 0, 0
    for did, edep in bar_edep.items():
        sub = did // 10000
        pl = (did // 1000) % 10
        b = did % 1000
        if sub == 1 and pl < 2 and edep >= 1.20: nv += 1
        elif sub == 2 and edep >= 1.50: nu += 1
        elif sub == 3:
            th = 0.90 if b < 60 else 1.35
            if edep >= th: nd += 1

    if not (2 <= nv <= 4 and 4 <= nu <= 8 and 6 <= nd <= 13):
        continue

    # Track reconstruction
    trackTask.fittedTracks.Delete()
    trackTask.ExecuteTask(option='ScifiDS')

    scifi_trks, ds_trks = [], []
    for trk in trackTask.fittedTracks:
        tt = trk.getTrackType()
        chi2ndf = trk.getChi2Ndf()
        sl_x = trk.getSlopeXZ()
        sl_y = trk.getSlopeYZ()
        if chi2ndf <= 5.0 and abs(sl_x) <= 0.1 and abs(sl_y) <= 0.1:
            if tt == 1: scifi_trks.append(trk)
            elif tt == 3: ds_trks.append(trk)

    if not scifi_trks or not ds_trks: continue

    # Track match at z = 430
    best_match = None
    min_dist = 999.0
    for st in scifi_trks:
        ps = st.getPointAtZ(430.0)
        for dt in ds_trks:
            pd = dt.getPointAtZ(430.0)
            dist = np.hypot(ps.X() - pd.X(), ps.Y() - pd.Y())
            if dist < 3.0 and dist < min_dist:
                min_dist = dist
                best_match = (st, dt)

    if best_match is None: continue
    n_passed += 1
    st_match, dt_match = best_match

    # Check DS horizontal bars crossed by DS track
    for did, b_info in ds_bars.items():
        if did not in bar_edep: continue
        p_trk = dt_match.getPointAtZ(b_info['z'])
        x_trk, y_trk = p_trk.X(), p_trk.Y()

        if abs(y_trk - b_info['y_center']) <= 0.6:
            if b_info['x_min'] <= x_trk <= b_info['x_max']:
                edep = bar_edep[did]
                dx = x_trk - b_info['x_center']
                h_eloss_all.Fill(edep, raw_w)
                if abs(dx) <= 5.0:
                    h_eloss_center.Fill(edep, raw_w)

print(f"MC Passed clean single muons: {n_passed} / {n_entries}")
print(f"MC h_eloss_center entries: {h_eloss_center.GetEntries()}, Mean: {h_eloss_center.GetMean():.3f} MeV")
print(f"MC h_eloss_all entries: {h_eloss_all.GetEntries()}, Mean: {h_eloss_all.GetMean():.3f} MeV")
