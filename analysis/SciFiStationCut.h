#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include "TClonesArray.h"
#include "sndScifiHit.h"

namespace snd {

class SciFiStationCut {
private:
    float fThreshold{0.0f};
    std::vector<int> fExcludedStations{1};
    bool fReversed{false};

public:
    SciFiStationCut() = default;
    SciFiStationCut(float threshold, const std::vector<int>& excluded_stations, bool reversed = false)
        : fThreshold(threshold), fExcludedStations(excluded_stations), fReversed(reversed) {}

    virtual ~SciFiStationCut() = default;

    // Callable operator for ROOT RDataFrame:
    // df.Filter(SciFiStationCut(...), {"Digi_ScifiHits"})
    bool operator()(const TClonesArray& scifiHits) const {
        return passCut(&scifiHits);
    }

    // Determine interaction station using cumulative hit fraction threshold
    int findStation(const TClonesArray* scifiHits) const {
        if (!scifiHits) return 0;
        std::vector<int> hor_hits(5, 0);
        std::vector<int> ver_hits(5, 0);
        int total_hits = 0;
        const int nHits = scifiHits->GetEntriesFast();
        for (int i = 0; i < nHits; ++i) {
            auto* hit = const_cast<sndScifiHit*>(static_cast<const sndScifiHit*>(scifiHits->At(i)));
            if (hit && hit->isValid()) {
                int st = hit->GetStation();
                if (st >= 1 && st <= 5) {
                    if (hit->isVertical()) {
                        ver_hits[st - 1]++;
                    } else {
                        hor_hits[st - 1]++;
                    }
                    total_hits++;
                }
            }
        }
        if (total_hits == 0) return 0;

        std::vector<float> frac(5, 0.0f);
        for (int s = 0; s < 5; ++s) {
            frac[s] = static_cast<float>(hor_hits[s] + ver_hits[s]) / static_cast<float>(total_hits);
        }
        std::vector<float> frac_sum(5, 0.0f);
        std::partial_sum(frac.begin(), frac.end(), frac_sum.begin());
        auto it = std::find_if(frac_sum.begin(), frac_sum.end(), [this](float f) { return f > fThreshold; });
        if (it == frac_sum.end()) return 5;
        return static_cast<int>(std::distance(frac_sum.begin(), it) + 1);
    }

    // Pointer-based interface
    bool passCut(const TClonesArray* scifiHits) const {
        int station = findStation(scifiHits);
        bool is_excluded = (std::find(fExcludedStations.begin(), fExcludedStations.end(), station) != fExcludedStations.end());
        if (fReversed) {
            return is_excluded;
        } else {
            return !is_excluded;
        }
    }

    // Getters
    float getThreshold() const { return fThreshold; }
    const std::vector<int>& getExcludedStations() const { return fExcludedStations; }
    bool isReversed() const { return fReversed; }

    // Setters
    void setThreshold(float th) { fThreshold = th; }
    void setExcludedStations(const std::vector<int>& stations) { fExcludedStations = stations; }
    void setReversed(bool r) { fReversed = r; }
};

namespace analysis_cuts {
    using SciFiStationCut = snd::SciFiStationCut;
    using sciFiStationCut = snd::SciFiStationCut;
}

} // namespace snd
