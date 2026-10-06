#pragma once

#include <vector>
#include <string>
#include "TClonesArray.h"
#include "MuFilterHit.h"

namespace snd {

class DSDimuonActivityCut {
private:
    int fMinPlanesDimuon{3};
    int fMinHitsDimuon{2};
    int fMinPlanesOther{3};
    int fMinHitsOther{1};
    bool fRequireValid{false};
    bool fReversed{false};

public:
    DSDimuonActivityCut() = default;
    DSDimuonActivityCut(int min_planes_dimuon, int min_hits_dimuon,
                        int min_planes_other, int min_hits_other,
                        bool require_valid = false, bool reversed = false)
        : fMinPlanesDimuon(min_planes_dimuon),
          fMinHitsDimuon(min_hits_dimuon),
          fMinPlanesOther(min_planes_other),
          fMinHitsOther(min_hits_other),
          fRequireValid(require_valid),
          fReversed(reversed) {}

    virtual ~DSDimuonActivityCut() = default;

    // Callable operator for ROOT RDataFrame:
    // df.Filter(DSDimuonActivityCut(...), {"Digi_MuFilterHits"})
    bool operator()(const TClonesArray& mufiHits) const {
        return passCut(&mufiHits);
    }

    // Pointer-based interface
    bool passCut(const TClonesArray* mufiHits) const {
        if (!mufiHits) {
            return fReversed ? true : false;
        }

        // Horizontal: 3 planes (0, 1, 2)
        // Vertical: 4 planes (0, 1, 2, 3)
        int hits_h[4] = {0, 0, 0, 0};
        int hits_v[4] = {0, 0, 0, 0};

        const int nHits = mufiHits->GetEntriesFast();
        for (int i = 0; i < nHits; ++i) {
            auto* hit = const_cast<MuFilterHit*>(static_cast<const MuFilterHit*>(mufiHits->At(i)));
            if (!hit) continue;
            if (fRequireValid && !hit->isValid()) continue;

            // System 3 is Downstream MuFilter (DS)
            if (hit->GetSystem() != 3) continue;

            int plane = hit->GetPlane();
            if (plane < 0 || plane > 3) continue;

            if (hit->isVertical()) {
                hits_v[plane]++;
            } else {
                hits_h[plane]++;
            }
        }

        // Count horizontal planes meeting thresholds
        int n_h_dimuon = 0;
        int n_h_other = 0;
        for (int p = 0; p < 3; ++p) {
            if (hits_h[p] >= fMinHitsDimuon) n_h_dimuon++;
            if (hits_h[p] >= fMinHitsOther) n_h_other++;
        }

        // Count vertical planes meeting thresholds
        int n_v_dimuon = 0;
        int n_v_other = 0;
        for (int p = 0; p < 4; ++p) {
            if (hits_v[p] >= fMinHitsDimuon) n_v_dimuon++;
            if (hits_v[p] >= fMinHitsOther) n_v_other++;
        }

        // Condition: (>=2 hits in >=3 planes in H && >=1 hit in >=3 planes in V) OR
        //            (>=2 hits in >=3 planes in V && >=1 hit in >=3 planes in H)
        bool cond_h_dimuon = (n_h_dimuon >= fMinPlanesDimuon && n_v_other >= fMinPlanesOther);
        bool cond_v_dimuon = (n_v_dimuon >= fMinPlanesDimuon && n_h_other >= fMinPlanesOther);

        bool passed = (cond_h_dimuon || cond_v_dimuon);

        return fReversed ? !passed : passed;
    }

    // Getters
    int getMinPlanesDimuon() const { return fMinPlanesDimuon; }
    int getMinHitsDimuon() const { return fMinHitsDimuon; }
    int getMinPlanesOther() const { return fMinPlanesOther; }
    int getMinHitsOther() const { return fMinHitsOther; }
    bool getRequireValid() const { return fRequireValid; }
    bool isReversed() const { return fReversed; }

    // Setters
    void setMinPlanesDimuon(int n) { fMinPlanesDimuon = n; }
    void setMinHitsDimuon(int n) { fMinHitsDimuon = n; }
    void setMinPlanesOther(int n) { fMinPlanesOther = n; }
    void setMinHitsOther(int n) { fMinHitsOther = n; }
    void setRequireValid(bool r) { fRequireValid = r; }
    void setReversed(bool r) { fReversed = r; }
};

namespace analysis_cuts {
    using DSDimuonActivityCut = snd::DSDimuonActivityCut;
    using dsDimuonActivityCut = snd::DSDimuonActivityCut;
}

} // namespace snd
