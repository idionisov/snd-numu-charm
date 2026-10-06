#pragma once

#include "TClonesArray.h"
#include "sndScifiHit.h"
#include "MuFilterHit.h"

namespace snd {

class SciFiDSTimeCut {
private:
    float fMinDeltaT{0.0f}; // latest_ds_time - earliest_scifi_time > fMinDeltaT
    bool fRequireValid{true};
    bool fReversed{false};

public:
    SciFiDSTimeCut() = default;
    explicit SciFiDSTimeCut(float min_delta_t, bool require_valid = true, bool reversed = false)
        : fMinDeltaT(min_delta_t), fRequireValid(require_valid), fReversed(reversed) {}

    virtual ~SciFiDSTimeCut() = default;

    // Callable operator for ROOT RDataFrame:
    // df.Filter(SciFiDSTimeCut(...), {"Digi_ScifiHits", "Digi_MuFilterHits"})
    bool operator()(const TClonesArray& scifiHits, const TClonesArray& mufiHits) const {
        return passCut(&scifiHits, &mufiHits);
    }

    bool passCut(const TClonesArray* scifiHits, const TClonesArray* mufiHits) const {
        if (!scifiHits || !mufiHits) return fReversed ? true : false;

        float min_scifi_t = 1e30f;
        bool has_scifi = false;
        const int n_sf = scifiHits->GetEntriesFast();
        for (int i = 0; i < n_sf; ++i) {
            auto* hit = const_cast<sndScifiHit*>(static_cast<const sndScifiHit*>(scifiHits->At(i)));
            if (hit && (!fRequireValid || hit->isValid())) {
                float t = hit->GetTime();
                if (t < min_scifi_t) {
                    min_scifi_t = t;
                    has_scifi = true;
                }
            }
        }

        float max_ds_t = -1e30f;
        bool has_ds = false;
        const int n_mf = mufiHits->GetEntriesFast();
        for (int i = 0; i < n_mf; ++i) {
            auto* hit = const_cast<MuFilterHit*>(static_cast<const MuFilterHit*>(mufiHits->At(i)));
            if (hit && (!fRequireValid || hit->isValid()) && hit->GetSystem() == 3) {
                float t;
                if (hit->isVertical()) {
                    t = hit->GetTime(0);
                } else {
                    t = 0.5f * (hit->GetTime(0) + hit->GetTime(1));
                }
                if (t > max_ds_t) {
                    max_ds_t = t;
                    has_ds = true;
                }
            }
        }

        if (!has_scifi || !has_ds) {
            return fReversed ? true : false;
        }

        bool passed = (max_ds_t - min_scifi_t > fMinDeltaT);
        return fReversed ? !passed : passed;
    }

    float getMinDeltaT() const { return fMinDeltaT; }
    bool getRequireValid() const { return fRequireValid; }
    bool isReversed() const { return fReversed; }

    void setMinDeltaT(float dt) { fMinDeltaT = dt; }
    void setRequireValid(bool req) { fRequireValid = req; }
    void setReversed(bool r) { fReversed = r; }
};

namespace analysis_cuts {
    using SciFiDSTimeCut = snd::SciFiDSTimeCut;
    using sciFiDSTimeCut = snd::SciFiDSTimeCut;
}

} // namespace snd
