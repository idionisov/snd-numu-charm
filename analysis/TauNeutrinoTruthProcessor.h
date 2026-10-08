#pragma once

#include "NeutrinoTruthProcessor.h"

namespace snd {

struct TauNeutrinoTruthInfo : public NeutrinoTruthInfo {
    // -------------------------------------------------------------
    // 1. Tau Neutrino Flags
    // -------------------------------------------------------------
    bool isNuTauCC{false};               // nu_tau CC: nu_tau (16) -> tau- (15)
    bool isAntiNuTauCC{false};           // anti-nu_tau CC: anti-nu_tau (-16) -> tau+ (-15)
    bool hasCandidate{false};            // Tau neutrino candidate

    // -------------------------------------------------------------
    // 2. Primary Outgoing Tau (tau_1 from Interaction Vertex)
    // -------------------------------------------------------------
    int tau1TrackId{-1};
    int tau1Pdg{0};
    int tau1Charge{0};                   // -1 for tau-, +1 for tau+
    double tau1E{0.0};
    double tau1P{0.0};
    double tau1Px{0.0};
    double tau1Py{0.0};
    double tau1Pz{0.0};
    double tau1Pt{0.0};
    double tau1Eta{0.0};
    double tau1Phi{0.0};
    double tau1Theta{0.0};
    double tau1SlopeXZ{0.0};
    double tau1SlopeYZ{0.0};

    // Tau decay vertex & topology
    double tauDecayX{0.0};
    double tauDecayY{0.0};
    double tauDecayZ{-9999.0};
    double tauFlightLength3D{0.0};
    double tauFlightLengthXY{0.0};
    double tauProperLifetimePs{0.0};
    std::string tauDecayMode{""};        // "to_muon", "to_electron", "1prong_hadron", "3prong_hadron", "other"
    int nTauDaughters{0};
    int tauLeptonicDaughterPdg{0};       // 13 (muon), 11 (electron) if leptonic
};

class TauNeutrinoTruthProcessor : public NeutrinoTruthProcessor {
public:
    explicit TauNeutrinoTruthProcessor(const NeutrinoTruthConfig& config = NeutrinoTruthConfig())
        : NeutrinoTruthProcessor(config) {}

    virtual ~TauNeutrinoTruthProcessor() override = default;

    int findPrimaryTau(const TClonesArray& mcTracks, int nuPdg) const;

    int findPrimaryLepton(const TClonesArray& mcTracks, int nuPdg) const override {
        return findPrimaryTau(mcTracks, nuPdg);
    }

    TauNeutrinoTruthInfo processTauNeutrino(const TClonesArray* mcTracks) const;

    TauNeutrinoTruthInfo operator()(const TClonesArray& mcTracks) const {
        return processTauNeutrino(&mcTracks);
    }

    NeutrinoTruthInfo process(const TClonesArray* mcTracks) const override {
        return processTauNeutrino(mcTracks);
    }
};

} // namespace snd
