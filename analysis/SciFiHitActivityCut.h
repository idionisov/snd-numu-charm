#pragma once

#include "TClonesArray.h"
#include "sndScifiHit.h"

namespace snd {

class SciFiHitActivityCut {
private:
    int fMinH{3};
    int fMinV{3};
    int fMaxTotal{50};
    bool fRequireValid{true};
    bool fReversed{false};

public:
    SciFiHitActivityCut() = default;
    explicit SciFiHitActivityCut(int min_h, int min_v = 3, int max_total = 50, bool require_valid = true, bool reversed = false)
        : fMinH(min_h), fMinV(min_v), fMaxTotal(max_total), fRequireValid(require_valid), fReversed(reversed) {}

    virtual ~SciFiHitActivityCut() = default;

    // Callable operator for ROOT RDataFrame:
    // df.Filter(SciFiHitActivityCut(...), {"Digi_ScifiHits"})
    bool operator()(const TClonesArray& scifiHits) const {
        return passCut(&scifiHits);
    }

    bool passCut(const TClonesArray* scifiHits) const {
        if (!scifiHits) return fReversed ? true : false;

        int n_h = 0;
        int n_v = 0;
        int n_tot = 0;
        const int nHits = scifiHits->GetEntriesFast();
        for (int i = 0; i < nHits; ++i) {
            auto* hit = const_cast<sndScifiHit*>(static_cast<const sndScifiHit*>(scifiHits->At(i)));
            if (!hit) continue;
            if (fRequireValid && !hit->isValid()) continue;

            n_tot++;
            if (hit->isVertical()) {
                n_v++;
            } else {
                n_h++;
            }
        }

        bool pass = (n_h >= fMinH && n_v >= fMinV && (fMaxTotal < 0 || n_tot < fMaxTotal));
        return fReversed ? !pass : pass;
    }

    int getMinH() const { return fMinH; }
    int getMinV() const { return fMinV; }
    int getMaxTotal() const { return fMaxTotal; }
    bool getRequireValid() const { return fRequireValid; }
    bool isReversed() const { return fReversed; }

    void setMinH(int h) { fMinH = h; }
    void setMinV(int v) { fMinV = v; }
    void setMaxTotal(int tot) { fMaxTotal = tot; }
    void setRequireValid(bool req) { fRequireValid = req; }
    void setReversed(bool r) { fReversed = r; }
};

namespace analysis_cuts {
    using SciFiHitActivityCut = snd::SciFiHitActivityCut;
    using sciFiHitActivityCut = snd::SciFiHitActivityCut;
}

} // namespace snd
