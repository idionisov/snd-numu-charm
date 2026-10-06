#pragma once

#include <vector>
#include <string>
#include <utility>
#include "TClonesArray.h"
#include "sndScifiHit.h"

namespace snd {

class AvgScifiFiducialCut {
private:
    double fVerticalMin{200.0};
    double fVerticalMax{1200.0};
    double fHorizontalMin{300.0};
    double fHorizontalMax{128.0 * 12.0 - 200.0}; // 1336.0
    bool fReversed{false};

public:
    AvgScifiFiducialCut() = default;
    AvgScifiFiducialCut(double vertical_min, double vertical_max,
                        double horizontal_min, double horizontal_max,
                        bool reversed = false)
        : fVerticalMin(vertical_min),
          fVerticalMax(vertical_max),
          fHorizontalMin(horizontal_min),
          fHorizontalMax(horizontal_max),
          fReversed(reversed) {}

    virtual ~AvgScifiFiducialCut() = default;

    // Callable operator for ROOT RDataFrame:
    // df.Filter(AvgScifiFiducialCut(...), {"Digi_ScifiHits"})
    bool operator()(const TClonesArray& scifiHits) const {
        return passCut(&scifiHits);
    }

    // Pointer-based interface
    bool passCut(const TClonesArray* scifiHits) const {
        if (!scifiHits) {
            return false;
        }

        double avg_ver = 0.0;
        unsigned int n_ver = 0;
        double avg_hor = 0.0;
        unsigned int n_hor = 0;

        const int nHits = scifiHits->GetEntriesFast();
        for (int i = 0; i < nHits; ++i) {
            auto* hit = const_cast<sndScifiHit*>(static_cast<const sndScifiHit*>(scifiHits->At(i)));
            if (!hit || !hit->isValid()) {
                continue;
            }

            int mat = hit->GetMat();
            int sipm = hit->GetSiPM();
            int channel = hit->GetSiPMChan();

            int x = channel + sipm * 128 + mat * 4 * 128;

            if (hit->isVertical()) {
                avg_ver += x;
                n_ver++;
            } else {
                avg_hor += x;
                n_hor++;
            }
        }

        if (n_ver == 0 || n_hor == 0) {
            return false;
        }

        avg_ver /= n_ver;
        avg_hor /= n_hor;

        if (!fReversed) {
            if (avg_hor < fHorizontalMin || avg_hor > fHorizontalMax) return false;
            if (avg_ver < fVerticalMin || avg_ver > fVerticalMax) return false;
        } else {
            if ((avg_hor > fHorizontalMin && avg_hor < fHorizontalMax) &&
                (avg_ver > fVerticalMin && avg_ver < fVerticalMax)) return false;
        }

        return true;
    }

    // Compute averages for a given hit collection
    std::pair<double, double> computeAverages(const TClonesArray* scifiHits) const {
        if (!scifiHits) {
            return {-1.0, -1.0};
        }

        double avg_ver = 0.0;
        unsigned int n_ver = 0;
        double avg_hor = 0.0;
        unsigned int n_hor = 0;

        const int nHits = scifiHits->GetEntriesFast();
        for (int i = 0; i < nHits; ++i) {
            auto* hit = const_cast<sndScifiHit*>(static_cast<const sndScifiHit*>(scifiHits->At(i)));
            if (!hit || !hit->isValid()) {
                continue;
            }

            int mat = hit->GetMat();
            int sipm = hit->GetSiPM();
            int channel = hit->GetSiPMChan();
            int x = channel + sipm * 128 + mat * 4 * 128;

            if (hit->isVertical()) {
                avg_ver += x;
                n_ver++;
            } else {
                avg_hor += x;
                n_hor++;
            }
        }

        double res_ver = (n_ver > 0) ? (avg_ver / n_ver) : -1.0;
        double res_hor = (n_hor > 0) ? (avg_hor / n_hor) : -1.0;
        return {res_ver, res_hor};
    }

    // Getters
    double getVerticalMin() const { return fVerticalMin; }
    double getVerticalMax() const { return fVerticalMax; }
    double getHorizontalMin() const { return fHorizontalMin; }
    double getHorizontalMax() const { return fHorizontalMax; }
    bool isReversed() const { return fReversed; }

    // Setters
    void setVerticalMin(double v) { fVerticalMin = v; }
    void setVerticalMax(double v) { fVerticalMax = v; }
    void setHorizontalMin(double h) { fHorizontalMin = h; }
    void setHorizontalMax(double h) { fHorizontalMax = h; }
    void setReversed(bool r) { fReversed = r; }
};

namespace analysis_cuts {
    using AvgScifiFiducialCut = snd::AvgScifiFiducialCut;
}

} // namespace snd
