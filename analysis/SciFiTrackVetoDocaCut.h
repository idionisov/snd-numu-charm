#pragma once

#include <vector>
#include <cmath>
#include "TClonesArray.h"
#include "sndRecoTrack.h"
#include "MuFilterHit.h"
#include "MuFilter.h"
#include "TROOT.h"
#include "TVector3.h"

namespace snd {

class SciFiTrackVetoDocaCut {
private:
    float fMaxChi2Ndf{5.0f};
    float fMaxDoca{6.0f}; // cm
    int fTrackType{11};   // 11 for SciFi track, -1 for any track
    bool fRequireValid{false};
    bool fReversed{false};
    mutable MuFilter* fMuFilterDet{nullptr};

public:
    SciFiTrackVetoDocaCut() = default;
    explicit SciFiTrackVetoDocaCut(float max_chi2_ndf, float max_doca = 6.0f, int track_type = 11,
                                   bool require_valid = false, bool reversed = false, MuFilter* muFilterDet = nullptr)
        : fMaxChi2Ndf(max_chi2_ndf), fMaxDoca(max_doca), fTrackType(track_type),
          fRequireValid(require_valid), fReversed(reversed), fMuFilterDet(muFilterDet) {}

    virtual ~SciFiTrackVetoDocaCut() = default;

    void setMuFilterDet(MuFilter* det) { fMuFilterDet = det; }
    MuFilter* getMuFilterDet() const {
        if (fMuFilterDet) return fMuFilterDet;
        return dynamic_cast<MuFilter*>(gROOT->GetListOfGlobals()->FindObject("MuFilter"));
    }

    // Callable operator for ROOT RDataFrame:
    // df.Filter(SciFiTrackVetoDocaCut(...), {"Reco_MuonTracks", "Digi_MuFilterHits"})
    bool operator()(const TClonesArray& tracks, const TClonesArray& mufiHits) const {
        return passCut(&tracks, &mufiHits);
    }

    bool passCut(const TClonesArray* tracks, const TClonesArray* mufiHits) const {
        if (!tracks || !mufiHits) return fReversed ? true : false;
        MuFilter* muFilterDet = getMuFilterDet();
        if (!muFilterDet) return fReversed ? true : false;

        const int nTracks = tracks->GetEntriesFast();
        const int nMufi = mufiHits->GetEntriesFast();

        // 1. Collect activated Veto hit detector IDs
        std::vector<int> vetoDetIDs;
        for (int j = 0; j < nMufi; ++j) {
            auto* hit = const_cast<MuFilterHit*>(static_cast<const MuFilterHit*>(mufiHits->At(j)));
            if (!hit) continue;
            if (fRequireValid && !hit->isValid()) continue;
            if (hit->GetSystem() == 1) { // Veto
                vetoDetIDs.push_back(hit->GetDetectorID());
            }
        }

        if (vetoDetIDs.empty()) {
            return fReversed ? true : false;
        }

        // 2. Loop over tracks
        for (int i = 0; i < nTracks; ++i) {
            auto* trk = const_cast<sndRecoTrack*>(static_cast<const sndRecoTrack*>(tracks->At(i)));
            if (!trk) continue;
            if (fTrackType >= 0 && trk->getTrackType() != fTrackType) continue;

            float chi2ndf = trk->getChi2Ndf();
            if (chi2ndf > fMaxChi2Ndf) continue;

            TVector3 pos = trk->getStart();
            TVector3 mom = trk->getTrackMom();
            if (mom.Mag2() < 1e-6) {
                mom = trk->getStop() - trk->getStart();
            }
            if (mom.Mag2() < 1e-6) continue;

            // Calculate DOCA to activated veto bars
            for (int detID : vetoDetIDs) {
                TVector3 A, B;
                muFilterDet->GetPosition(detID, A, B);
                TVector3 u = B - A;
                TVector3 uCrossv = u.Cross(mom);
                double mag = uCrossv.Mag();
                double doca;
                if (mag > 1e-6) {
                    doca = std::abs((A - pos).Dot(uCrossv)) / mag;
                } else {
                    doca = ((A - pos).Cross(mom)).Mag() / mom.Mag();
                }

                if (doca <= fMaxDoca) {
                    return fReversed ? false : true;
                }
            }
        }

        return fReversed ? true : false;
    }

    float getMaxChi2Ndf() const { return fMaxChi2Ndf; }
    float getMaxDoca() const { return fMaxDoca; }
    int getTrackType() const { return fTrackType; }
    bool getRequireValid() const { return fRequireValid; }
    bool isReversed() const { return fReversed; }

    void setMaxChi2Ndf(float c) { fMaxChi2Ndf = c; }
    void setMaxDoca(float d) { fMaxDoca = d; }
    void setTrackType(int t) { fTrackType = t; }
    void setRequireValid(bool req) { fRequireValid = req; }
    void setReversed(bool r) { fReversed = r; }
};

namespace analysis_cuts {
    using SciFiTrackVetoDocaCut = snd::SciFiTrackVetoDocaCut;
    using sciFiTrackVetoDocaCut = snd::SciFiTrackVetoDocaCut;
}

} // namespace snd
