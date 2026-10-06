#pragma once

#include "TClonesArray.h"
#include "MuFilterHit.h"

namespace snd {

class DSActivityCut {
private:
    int fMinUSPlanes{5};
    bool fRequireDSHits{true};
    bool fRequireValid{true};
    bool fReversed{false};

public:
    DSActivityCut() = default;
    explicit DSActivityCut(int min_us_planes, bool require_ds_hits = true,
                           bool require_valid = true, bool reversed = false)
        : fMinUSPlanes(min_us_planes),
          fRequireDSHits(require_ds_hits),
          fRequireValid(require_valid),
          fReversed(reversed) {}

    virtual ~DSActivityCut() = default;

    // Callable operator for ROOT RDataFrame:
    // df.Filter(DSActivityCut(...), {"Digi_MuFilterHits"})
    bool operator()(const TClonesArray& mufiHits) const {
        return passCut(&mufiHits);
    }

    bool passCut(const TClonesArray* mufiHits) const {
        if (!mufiHits) return fReversed ? true : false;

        bool has_ds = false;
        bool us_planes[5] = {false, false, false, false, false};

        const int n_mf = mufiHits->GetEntriesFast();
        for (int i = 0; i < n_mf; ++i) {
            auto* hit = const_cast<MuFilterHit*>(static_cast<const MuFilterHit*>(mufiHits->At(i)));
            if (!hit) continue;
            if (fRequireValid && !hit->isValid()) continue;

            if (hit->GetSystem() == 2) { // US (Upstream)
                int plane = hit->GetPlane();
                if (plane >= 0 && plane < 5) {
                    us_planes[plane] = true;
                }
            } else if (hit->GetSystem() == 3) { // DS (Downstream)
                has_ds = true;
            }
        }

        int n_us = 0;
        for (int p = 0; p < 5; ++p) {
            if (us_planes[p]) n_us++;
        }

        bool passed = false;
        if (has_ds) {
            passed = (n_us >= fMinUSPlanes);
        } else {
            passed = !fRequireDSHits;
        }

        return fReversed ? !passed : passed;
    }

    int getMinUSPlanes() const { return fMinUSPlanes; }
    bool getRequireDSHits() const { return fRequireDSHits; }
    bool getRequireValid() const { return fRequireValid; }
    bool isReversed() const { return fReversed; }

    void setMinUSPlanes(int n) { fMinUSPlanes = n; }
    void setRequireDSHits(bool r) { fRequireDSHits = r; }
    void setRequireValid(bool req) { fRequireValid = req; }
    void setReversed(bool r) { fReversed = r; }
};

} // namespace snd
