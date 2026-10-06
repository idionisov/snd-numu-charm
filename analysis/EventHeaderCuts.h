#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include "SNDLHCEventHeader.h"
#include "SNDLHCEventHeaderConst.h"

namespace snd {

class EventHeaderIP1Cut {
private:
    bool fReversed{false};

public:
    EventHeaderIP1Cut() = default;
    explicit EventHeaderIP1Cut(bool reversed) : fReversed(reversed) {}

    virtual ~EventHeaderIP1Cut() = default;

    // Callable operator for ROOT RDataFrame:
    // df.Filter(EventHeaderIP1Cut(), {"EventHeader"})
    bool operator()(const SNDLHCEventHeader& header) const {
        return passCut(&header);
    }

    bool passCut(const SNDLHCEventHeader* header) const {
        if (!header) return fReversed ? true : false;
        auto* h = const_cast<SNDLHCEventHeader*>(header);
        bool is_ip1 = h->isIP1();
        return fReversed ? !is_ip1 : is_ip1;
    }

    bool isReversed() const { return fReversed; }
    void setReversed(bool r) { fReversed = r; }
};

class StableBeamsCut {
private:
    int fBeamMode{static_cast<int>(LhcBeamMode::StableBeams)}; // 11
    bool fReversed{false};

public:
    StableBeamsCut() = default;
    explicit StableBeamsCut(int beam_mode, bool reversed = false)
        : fBeamMode(beam_mode), fReversed(reversed) {}

    virtual ~StableBeamsCut() = default;

    // Callable operator for ROOT RDataFrame:
    // df.Filter(StableBeamsCut(), {"EventHeader"})
    bool operator()(const SNDLHCEventHeader& header) const {
        return passCut(&header);
    }

    bool passCut(const SNDLHCEventHeader* header) const {
        if (!header) return fReversed ? true : false;
        bool is_stable = (header->GetBeamMode() == fBeamMode);
        return fReversed ? !is_stable : is_stable;
    }

    int getBeamMode() const { return fBeamMode; }
    bool isReversed() const { return fReversed; }

    void setBeamMode(int m) { fBeamMode = m; }
    void setReversed(bool r) { fReversed = r; }
};

class EventDeltatCut {
private:
    int64_t fMinDeltaT{100};
    std::vector<int64_t> fEventTimes;
    bool fReversed{false};
    mutable int64_t fLastTime{-1};

public:
    EventDeltatCut() = default;
    explicit EventDeltatCut(int64_t min_delta_t, bool reversed = false)
        : fMinDeltaT(min_delta_t), fReversed(reversed) {}
    EventDeltatCut(int64_t min_delta_t, const std::vector<int64_t>& times, bool reversed = false)
        : fMinDeltaT(min_delta_t), fEventTimes(times), fReversed(reversed) {}

    virtual ~EventDeltatCut() = default;

    void setEventTimes(const std::vector<int64_t>& times) { fEventTimes = times; }
    const std::vector<int64_t>& getEventTimes() const { return fEventTimes; }

    // RDataFrame callable via entry index:
    // df.Filter(EventDeltatCut(...), {"rdfentry_"})
    bool operator()(ULong64_t entry) const {
        if (entry == 0) return !fReversed;
        if (entry < fEventTimes.size()) {
            int64_t dt = fEventTimes[entry] - fEventTimes[entry - 1];
            bool pass = (std::abs(dt) > fMinDeltaT);
            return fReversed ? !pass : pass;
        }
        return !fReversed;
    }

    int64_t getMinDeltaT() const { return fMinDeltaT; }
    bool isReversed() const { return fReversed; }

    void setMinDeltaT(int64_t dt) { fMinDeltaT = dt; }
    void setReversed(bool r) { fReversed = r; }
};

namespace analysis_cuts {
    using EventHeaderIP1Cut = snd::EventHeaderIP1Cut;
    using eventHeaderIP1Cut = snd::EventHeaderIP1Cut;
    using StableBeamsCut = snd::StableBeamsCut;
    using stableBeamsCut = snd::StableBeamsCut;
    using EventDeltatCut = snd::EventDeltatCut;
    using eventDeltatCut = snd::EventDeltatCut;
}

} // namespace snd
