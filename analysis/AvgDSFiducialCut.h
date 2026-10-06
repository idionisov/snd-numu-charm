#pragma once

#include <vector>
#include <string>
#include <utility>
#include "TClonesArray.h"
#include "MuFilterHit.h"

namespace snd {

class AvgDSFiducialCut {
private:
    double fVerticalMin{70.0};
    double fVerticalMax{105.0};
    double fHorizontalMin{10.0};
    double fHorizontalMax{50.0};
    bool fReversed{false};

public:
    AvgDSFiducialCut() = default;
    AvgDSFiducialCut(double vertical_min, double vertical_max,
                     double horizontal_min, double horizontal_max,
                     bool reversed = false)
        : fVerticalMin(vertical_min),
          fVerticalMax(vertical_max),
          fHorizontalMin(horizontal_min),
          fHorizontalMax(horizontal_max),
          fReversed(reversed) {}

    virtual ~AvgDSFiducialCut() = default;

    // Callable operator for ROOT RDataFrame:
    // df.Filter(AvgDSFiducialCut(...), {"Digi_MuFilterHits"})
    bool operator()(const TClonesArray& mufiHits) const {
        return passCut(&mufiHits);
    }

    // Pointer-based interface
    bool passCut(const TClonesArray* mufiHits) const {
        if (!mufiHits) {
            return false;
        }

        double avg_ver = 0.0;
        unsigned int n_ver = 0;
        double avg_hor = 0.0;
        unsigned int n_hor = 0;

        const int nHits = mufiHits->GetEntriesFast();
        for (int i = 0; i < nHits; ++i) {
            auto* hit = const_cast<MuFilterHit*>(static_cast<const MuFilterHit*>(mufiHits->At(i)));
            if (!hit || !hit->isValid()) {
                continue;
            }

            // Downstream MuFilter is System 3
            if (hit->GetSystem() != 3) {
                continue;
            }

            int x = hit->GetDetectorID() % 1000;

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
    std::pair<double, double> computeAverages(const TClonesArray* mufiHits) const {
        if (!mufiHits) {
            return {-1.0, -1.0};
        }

        double avg_ver = 0.0;
        unsigned int n_ver = 0;
        double avg_hor = 0.0;
        unsigned int n_hor = 0;

        const int nHits = mufiHits->GetEntriesFast();
        for (int i = 0; i < nHits; ++i) {
            auto* hit = const_cast<MuFilterHit*>(static_cast<const MuFilterHit*>(mufiHits->At(i)));
            if (!hit || !hit->isValid() || hit->GetSystem() != 3) {
                continue;
            }

            int x = hit->GetDetectorID() % 1000;

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
    using AvgDSFiducialCut = snd::AvgDSFiducialCut;
    using avgDSFiducialCut = snd::AvgDSFiducialCut;
}

} // namespace snd
