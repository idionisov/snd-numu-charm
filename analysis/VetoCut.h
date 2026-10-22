#pragma once

#include <vector>
#include <string>
#include "TClonesArray.h"
#include "sndScifiHit.h"
#include "MuFilterHit.h"

namespace snd {

class VetoCut {
private:
    int fMinHits{0};
    int fMaxHits{0};
    bool fRequireValid{false};
    float fTimeThreshold{0.3f}; // Latency threshold in ns relative to earliest SciFi hit
    bool fUseTiming{true};

public:
    VetoCut() = default;
    VetoCut(int min_hits, int max_hits, bool require_valid = false, float time_threshold = 0.3f, bool use_timing = true)
        : fMinHits(min_hits), fMaxHits(max_hits), fRequireValid(require_valid),
          fTimeThreshold(time_threshold), fUseTiming(use_timing) {}

    virtual ~VetoCut() = default;

    // Callable operator for ROOT RDataFrame:
    // df.Filter(VetoCut(...), {"Digi_MuFilterHits", "Digi_ScifiHits"})
    bool operator()(const TClonesArray& mufiHits, const TClonesArray& scifiHits) const {
        return passCut(&mufiHits, &scifiHits);
    }

    // Reconstruct time for a MuFilter hit
    float getHitTime(const MuFilterHit* hit) const {
        if (!hit) return 0.0f;
        auto* mutable_hit = const_cast<MuFilterHit*>(hit);
        if (mutable_hit->isVertical()) {
            return mutable_hit->GetTime(0);
        }
        float t0 = mutable_hit->GetTime(0);
        float t1 = mutable_hit->GetTime(1);
        if (t0 > -900.0f && t1 > -900.0f) {
            return 0.5f * (t0 + t1);
        } else if (t0 > -900.0f) {
            return t0;
        } else {
            return t1;
        }
    }

    // Count in-time hits in MuFilter System 1 (Veto).
    // If fUseTiming is true and SciFi hits exist, hits arriving later than
    // min_scifi_t + fTimeThreshold are considered delayed backsplash and ignored.
    int countVetoHits(const TClonesArray* mufiHits, const TClonesArray* scifiHits = nullptr) const {
        if (!mufiHits) return 0;

        float min_scifi_t = 1e30f;
        bool has_scifi = false;
        if (fUseTiming && scifiHits) {
            const int n_sf = scifiHits->GetEntriesFast();
            for (int i = 0; i < n_sf; ++i) {
                auto* hit = const_cast<sndScifiHit*>(static_cast<const sndScifiHit*>(scifiHits->At(i)));
                if (!hit) continue;
                if (fRequireValid && !hit->isValid()) continue;
                float t = hit->GetTime();
                if (t < min_scifi_t) {
                    min_scifi_t = t;
                    has_scifi = true;
                }
            }
        }

        int count = 0;
        const int nHits = mufiHits->GetEntriesFast();
        for (int i = 0; i < nHits; ++i) {
            auto* hit = const_cast<MuFilterHit*>(static_cast<const MuFilterHit*>(mufiHits->At(i)));
            if (!hit) continue;
            if (fRequireValid && !hit->isValid()) continue;
            if (hit->GetSystem() == 1) {
                if (fUseTiming && has_scifi) {
                    float t_veto = getHitTime(hit);
                    if (t_veto > min_scifi_t + fTimeThreshold) {
                        // Late hit (latency > threshold): delayed backsplash, ignore
                        continue;
                    }
                }
                count++;
            }
        }
        return count;
    }

    // Pointer-based interface
    bool passCut(const TClonesArray* mufiHits, const TClonesArray* scifiHits = nullptr) const {
        if (!mufiHits) {
            return (fMinHits <= 0);
        }

        int n_veto = countVetoHits(mufiHits, scifiHits);

        if (n_veto < fMinHits) return false;
        if (fMaxHits >= 0 && n_veto > fMaxHits) return false;

        return true;
    }

    // Getters
    int getMinHits() const { return fMinHits; }
    int getMaxHits() const { return fMaxHits; }
    bool getRequireValid() const { return fRequireValid; }
    float getTimeThreshold() const { return fTimeThreshold; }
    bool getUseTiming() const { return fUseTiming; }

    // Setters
    void setMinHits(int min_h) { fMinHits = min_h; }
    void setMaxHits(int max_h) { fMaxHits = max_h; }
    void setRequireValid(bool req) { fRequireValid = req; }
    void setTimeThreshold(float t) { fTimeThreshold = t; }
    void setUseTiming(bool use) { fUseTiming = use; }
};

namespace analysis_cuts {
    using VetoCut = snd::VetoCut;
    using vetoCut = snd::VetoCut;
}

} // namespace snd
