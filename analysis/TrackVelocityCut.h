#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include "TClonesArray.h"
#include "sndRecoTrack.h"
#include "TVector3.h"

namespace snd {

class TrackVelocityCut {
private:
    float fMaxVelocity{0.0f};     // vel < max_velocity (e.g. 0.0 for negative velocity / backward)
    float fMinVelocity{-9999.0f};  // lower bound on velocity
    int fTrackType{1};            // 1 = SciFi simple track, 13 = DS track, -1 = any track
    bool fRequireNegative{true};  // require vel < 0
    bool fReversed{false};

public:
    TrackVelocityCut() = default;
    TrackVelocityCut(float max_velocity,
                     int track_type = 1,
                     bool reversed = false,
                     float min_velocity = -9999.0f,
                     bool require_negative = true)
        : fMaxVelocity(max_velocity),
          fMinVelocity(min_velocity),
          fTrackType(track_type),
          fRequireNegative(require_negative),
          fReversed(reversed) {}

    virtual ~TrackVelocityCut() = default;

    // Callable operator for ROOT RDataFrame:
    // df.Filter(TrackVelocityCut(...), {"Reco_MuonTracks"})
    bool operator()(const TClonesArray& tracks) const {
        return passCut(&tracks);
    }

    // Pointer-based interface
    bool passCut(const TClonesArray* tracks) const {
        if (!tracks) return fReversed ? true : false;

        const int nTracks = tracks->GetEntriesFast();
        for (int i = 0; i < nTracks; ++i) {
            auto* trk = const_cast<sndRecoTrack*>(static_cast<const sndRecoTrack*>(tracks->At(i)));
            if (!trk) continue;
            if (fTrackType >= 0 && trk->getTrackType() != fTrackType) continue;

            const auto& pts = trk->getTrackPoints();
            const auto& raw_times = trk->getRawMeasTimes();
            size_t n_meas = std::min(pts.size(), raw_times.size());

            std::vector<float> t_vals;
            std::vector<float> z_vals;
            for (size_t k = 0; k < n_meas; ++k) {
                if (!raw_times[k].empty()) {
                    t_vals.push_back(raw_times[k][0]);
                    z_vals.push_back(pts[k].Z());
                }
            }

            if (t_vals.size() >= 2) {
                float dz = z_vals.back() - z_vals.front();
                float dt = t_vals.back() - t_vals.front();
                if (std::abs(dz) > 1e-3f && std::abs(dt) > 1e-4f) {
                    float vel = dz / dt;
                    bool vel_ok = false;
                    if (fRequireNegative) {
                        vel_ok = (vel < fMaxVelocity && vel >= fMinVelocity);
                    } else {
                        vel_ok = (vel >= fMinVelocity && vel <= fMaxVelocity);
                    }

                    if (vel_ok) {
                        return fReversed ? false : true;
                    }
                }
            }
        }

        return fReversed ? true : false;
    }

    // Helper to compute velocity of first valid track of specified type
    float computeVelocity(const TClonesArray* tracks) const {
        if (!tracks) return -9999.0f;
        const int nTracks = tracks->GetEntriesFast();
        for (int i = 0; i < nTracks; ++i) {
            auto* trk = const_cast<sndRecoTrack*>(static_cast<const sndRecoTrack*>(tracks->At(i)));
            if (!trk) continue;
            if (fTrackType >= 0 && trk->getTrackType() != fTrackType) continue;

            const auto& pts = trk->getTrackPoints();
            const auto& raw_times = trk->getRawMeasTimes();
            size_t n_meas = std::min(pts.size(), raw_times.size());

            std::vector<float> t_vals;
            std::vector<float> z_vals;
            for (size_t k = 0; k < n_meas; ++k) {
                if (!raw_times[k].empty()) {
                    t_vals.push_back(raw_times[k][0]);
                    z_vals.push_back(pts[k].Z());
                }
            }

            if (t_vals.size() >= 2) {
                float dz = z_vals.back() - z_vals.front();
                float dt = t_vals.back() - t_vals.front();
                if (std::abs(dz) > 1e-3f && std::abs(dt) > 1e-4f) {
                    return dz / dt;
                }
            }
        }
        return -9999.0f;
    }

    // Getters
    float getMaxVelocity() const { return fMaxVelocity; }
    float getMinVelocity() const { return fMinVelocity; }
    int getTrackType() const { return fTrackType; }
    bool getRequireNegative() const { return fRequireNegative; }
    bool isReversed() const { return fReversed; }

    // Setters
    void setMaxVelocity(float v) { fMaxVelocity = v; }
    void setMinVelocity(float v) { fMinVelocity = v; }
    void setTrackType(int t) { fTrackType = t; }
    void setRequireNegative(bool req) { fRequireNegative = req; }
    void setReversed(bool r) { fReversed = r; }
};

namespace analysis_cuts {
    using TrackVelocityCut = snd::TrackVelocityCut;
    using trackVelocityCut = snd::TrackVelocityCut;
    using BackwardTrackCut = snd::TrackVelocityCut;
    using backwardTrackCut = snd::TrackVelocityCut;
}

} // namespace snd
