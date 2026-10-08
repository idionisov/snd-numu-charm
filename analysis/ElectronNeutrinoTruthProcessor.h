#pragma once

#include "NeutrinoTruthProcessor.h"

namespace snd {

struct ElectronNeutrinoTruthInfo : public NeutrinoTruthInfo {
    // -------------------------------------------------------------
    // 1. Electron Neutrino Flags
    // -------------------------------------------------------------
    bool isNuECC{false};                 // nu_e CC: nu_e (12) -> e- (11)
    bool isAntiNuECC{false};             // anti-nu_e CC: anti-nu_e (-12) -> e+ (-11)
    bool hasPromptCharmElectron{false};  // Electron produced from prompt charm decay
    bool hasPromptCharmMuon{false};      // Muon produced from prompt charm decay (e-mu final state)
    bool isOppositeSignDielectron{false};// e1 and e2 have opposite signs (e- e+ or e+ e-)
    bool hasCandidate{false};            // Golden signature: nu_e CC + primary electron + charm prompt decay lepton

    // -------------------------------------------------------------
    // 2. Primary Outgoing Electron (e_1 from Interaction Vertex)
    // -------------------------------------------------------------
    int e1TrackId{-1};
    int e1Pdg{0};
    int e1Charge{0};                     // -1 for e-, +1 for e+
    double e1E{0.0};
    double e1P{0.0};
    double e1Px{0.0};
    double e1Py{0.0};
    double e1Pz{0.0};
    double e1Pt{0.0};
    double e1Eta{0.0};
    double e1Phi{0.0};
    double e1Theta{0.0};
    double e1SlopeXZ{0.0};               // dx/dz
    double e1SlopeYZ{0.0};               // dy/dz

    // -------------------------------------------------------------
    // 3. Secondary Lepton from Prompt Charm Decay (e_2 or mu_2)
    // -------------------------------------------------------------
    int lep2TrackId{-1};
    int lep2Pdg{0};
    int lep2Charge{0};
    double lep2E{0.0};
    double lep2P{0.0};
    double lep2Pt{0.0};
    double lep2PtRel{0.0};
    double lep2IP3D{0.0};
    double lep2IPXY{0.0};

    // Composite Dilepton Invariant Mass (if applicable)
    double dileptonInvMass{0.0};
    double dileptonOpeningAngle{0.0};
    double dileptonDeltaPhi{0.0};
};

class ElectronNeutrinoTruthProcessor : public NeutrinoTruthProcessor {
public:
    explicit ElectronNeutrinoTruthProcessor(const NeutrinoTruthConfig& config = NeutrinoTruthConfig())
        : NeutrinoTruthProcessor(config) {}

    virtual ~ElectronNeutrinoTruthProcessor() override = default;

    int findPrimaryElectron(const TClonesArray& mcTracks, int nuPdg) const;

    int findPrimaryLepton(const TClonesArray& mcTracks, int nuPdg) const override {
        return findPrimaryElectron(mcTracks, nuPdg);
    }

    int findCharmDecayElectron(const TClonesArray& mcTracks,
                               const std::vector<int>& charmTrackIds,
                               int primaryElectronTrackId,
                               int& outParentCharmId) const;

    ElectronNeutrinoTruthInfo processElectronNeutrino(const TClonesArray* mcTracks) const;

    ElectronNeutrinoTruthInfo operator()(const TClonesArray& mcTracks) const {
        return processElectronNeutrino(&mcTracks);
    }

    NeutrinoTruthInfo process(const TClonesArray* mcTracks) const override {
        return processElectronNeutrino(mcTracks);
    }
};

} // namespace snd
