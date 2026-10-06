#pragma once

#include <vector>
#include <string>
#include "TClonesArray.h"
#include "MuFilterHit.h"

namespace snd {

class VetoCut {
private:
    int fMinHits{0};
    int fMaxHits{0};
    bool fRequireValid{false};

public:
    VetoCut() = default;
    VetoCut(int min_hits, int max_hits, bool require_valid = false)
        : fMinHits(min_hits), fMaxHits(max_hits), fRequireValid(require_valid) {}

    virtual ~VetoCut() = default;

    // Callable operator for ROOT RDataFrame:
    // df.Filter(VetoCut(...), {"Digi_MuFilterHits"})
    bool operator()(const TClonesArray& mufiHits) const {
        return passCut(&mufiHits);
    }

    // Pointer-based interface
    bool passCut(const TClonesArray* mufiHits) const {
        if (!mufiHits) {
            return (fMinHits <= 0);
        }

        int n_veto = countVetoHits(mufiHits);

        if (n_veto < fMinHits) return false;
        if (fMaxHits >= 0 && n_veto > fMaxHits) return false;

        return true;
    }

    // Count hits in MuFilter System 1 (Veto)
    int countVetoHits(const TClonesArray* mufiHits) const {
        if (!mufiHits) return 0;
        int count = 0;
        const int nHits = mufiHits->GetEntriesFast();
        for (int i = 0; i < nHits; ++i) {
            auto* hit = const_cast<MuFilterHit*>(static_cast<const MuFilterHit*>(mufiHits->At(i)));
            if (!hit) continue;
            if (fRequireValid && !hit->isValid()) continue;
            if (hit->GetSystem() == 1) {
                count++;
            }
        }
        return count;
    }

    // Getters
    int getMinHits() const { return fMinHits; }
    int getMaxHits() const { return fMaxHits; }
    bool getRequireValid() const { return fRequireValid; }

    // Setters
    void setMinHits(int min_h) { fMinHits = min_h; }
    void setMaxHits(int max_h) { fMaxHits = max_h; }
    void setRequireValid(bool req) { fRequireValid = req; }
};

namespace analysis_cuts {
    using VetoCut = snd::VetoCut;
    using vetoCut = snd::VetoCut;
}

} // namespace snd
