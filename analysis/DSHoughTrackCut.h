#pragma once

#include <vector>
#include <string>
#include "TClonesArray.h"
#include "sndRecoTrack.h"

namespace snd {

class DSHoughTrackCut {
private:
    int fMinTracks{2};
    int fMaxTracks{2};
    int fTrackType{13};
    bool fReversed{false};

public:
    DSHoughTrackCut() = default;
    DSHoughTrackCut(int min_tracks, int max_tracks = 2, int track_type = 13, bool reversed = false)
        : fMinTracks(min_tracks),
          fMaxTracks(max_tracks),
          fTrackType(track_type),
          fReversed(reversed) {}

    virtual ~DSHoughTrackCut() = default;

    // Callable operator for ROOT RDataFrame:
    // df.Filter(DSHoughTrackCut(...), {"Reco_MuonTracks"})
    bool operator()(const TClonesArray& tracks) const {
        return passCut(&tracks);
    }

    // Pointer-based interface
    bool passCut(const TClonesArray* tracks) const {
        if (!tracks) {
            bool passed = (fMinTracks <= 0 && (fMaxTracks < 0 || fMaxTracks >= 0));
            return fReversed ? !passed : passed;
        }

        int count = countTracks(tracks);

        bool passed = true;
        if (fMinTracks >= 0 && count < fMinTracks) passed = false;
        if (fMaxTracks >= 0 && count > fMaxTracks) passed = false;

        return fReversed ? !passed : passed;
    }

    int countTracks(const TClonesArray* tracks) const {
        if (!tracks) return 0;
        int count = 0;
        const int nTracks = tracks->GetEntriesFast();
        for (int i = 0; i < nTracks; ++i) {
            auto* trk = const_cast<sndRecoTrack*>(static_cast<const sndRecoTrack*>(tracks->At(i)));
            if (!trk) continue;
            if (trk->getTrackType() == fTrackType) {
                count++;
            }
        }
        return count;
    }

    // Getters
    int getMinTracks() const { return fMinTracks; }
    int getMaxTracks() const { return fMaxTracks; }
    int getTrackType() const { return fTrackType; }
    bool getReversed() const { return fReversed; }

    // Setters
    void setMinTracks(int min_t) { fMinTracks = min_t; }
    void setMaxTracks(int max_t) { fMaxTracks = max_t; }
    void setTrackType(int t_type) { fTrackType = t_type; }
    void setReversed(bool rev) { fReversed = rev; }
};

class DSHoughTrackCounter {
private:
    int fTrackType{13};

public:
    DSHoughTrackCounter(int track_type = 13) : fTrackType(track_type) {}
    virtual ~DSHoughTrackCounter() = default;

    int operator()(const TClonesArray& tracks) const {
        int count = 0;
        const int nTracks = tracks.GetEntriesFast();
        for (int i = 0; i < nTracks; ++i) {
            auto* trk = const_cast<sndRecoTrack*>(static_cast<const sndRecoTrack*>(tracks.At(i)));
            if (!trk) continue;
            if (trk->getTrackType() == fTrackType) {
                count++;
            }
        }
        return count;
    }
};

inline int count_ds_hough_tracks(const TClonesArray& tracks) {
    DSHoughTrackCounter counter(13);
    return counter(tracks);
}

namespace analysis_cuts {
    using DSHoughTrackCut = snd::DSHoughTrackCut;
    using dsHoughTrackCut = snd::DSHoughTrackCut;
    using DSHoughTrackCounter = snd::DSHoughTrackCounter;
}

} // namespace snd
