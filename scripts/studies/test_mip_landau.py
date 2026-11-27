import ROOT
import numpy as np
import SndlhcGeo
import SndlhcTracking

ROOT.gROOT.SetBatch(True)

data_file = "/eos/experiment/sndlhc/convertedData/physics/2022/run_005408/sndsw_raw-0002.root"
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
        x_min = min(A.X(), B.X())
        x_max = max(A.X(), B.X())
        ds_bars[did] = {
            'plane': pl, 'bar': b,
            'x_min': x_min, 'x_max': x_max,
            'x_left': A.X(), 'x_right': B.X(), # A is Left (pos X), B is Right (neg X)
            'x_center': 0.5 * (A.X() + B.X()),
            'y_center': 0.5 * (A.Y() + B.Y()),
            'z': 0.5 * (A.Z() + B.Z()),
            'length': (A - B).Mag()
        }

chain = ROOT.TChain("rawConv")
chain.Add(data_file)

ioman = ROOT.FairRootManager.Instance()
ioman.SetInChain(chain)

trackTask = SndlhcTracking.Tracking()
trackTask.SetName("simpleTracking")
trackTask.SetTrackClassType(False)
trackTask.Init()

h_qdc_center = ROOT.TH1F("h_qdc_center", "DS Hor Single-SiPM QDC (Central: |x - x_{c}| < 5 cm);QDC [counts];Entries", 100, 0, 100)
h_qdc_all = ROOT.TH1F("h_qdc_all", "DS Hor Single-SiPM QDC (All x);QDC [counts];Entries", 100, 0, 100)

n_entries = min(10000, chain.GetEntries())
n_passed = 0
prev_time = None

for iev in range(n_entries):
    chain.GetEvent(iev)
    h = chain.EventHeader
    if not h.isIP1(): continue
    if h.GetBeamMode() != 11: continue
    t = h.GetEventTime()
    if prev_time is not None and abs(t - prev_time) <= 100:
        prev_time = t
        continue
    prev_time = t

    # Multiplicity cuts
    nsf = len(chain.Digi_ScifiHits)
    if not (6 <= nsf <= 27): continue

    # Hit counts
    nv, nu, nd = 0, 0, 0
    hit_dict = {}
    for hit in chain.Digi_MuFilterHits:
        if not hit.isValid(): continue
        did = hit.GetDetectorID()
        sub = did // 10000
        if sub == 1: nv += 1
        elif sub == 2: nu += 1
        elif sub == 3: nd += 1
        hit_dict[did] = hit

    if not (2 <= nv <= 4 and 4 <= nu <= 8 and 6 <= nd <= 13):
        continue

    # Reconstruct track
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
        if did not in hit_dict: continue
        p_trk = dt_match.getPointAtZ(b_info['z'])
        x_trk, y_trk = p_trk.X(), p_trk.Y()

        # Crossing check: vertical within 0.6 cm of bar center
        if abs(y_trk - b_info['y_center']) <= 0.6:
            # Check horizontal range
            if b_info['x_min'] <= x_trk <= b_info['x_max']:
                hit = hit_dict[did]
                sigs = hit.GetAllSignals(True, True)
                # Distance from center
                dx = x_trk - b_info['x_center']
                for side, qdc_val in sigs:
                    h_qdc_all.Fill(qdc_val)
                    if abs(dx) <= 5.0:
                        h_qdc_center.Fill(qdc_val)

print(f"Passed clean single muons: {n_passed} / {n_entries}")
print(f"h_qdc_center entries: {h_qdc_center.GetEntries()}, Mean: {h_qdc_center.GetMean():.2f}")
print(f"h_qdc_all entries: {h_qdc_all.GetEntries()}, Mean: {h_qdc_all.GetMean():.2f}")
