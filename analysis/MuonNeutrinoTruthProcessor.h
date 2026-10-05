#pragma once

#include "NeutrinoTruthProcessor.h"

namespace snd {

struct MuonNeutrinoTruthInfo : public NeutrinoTruthInfo {
    // -------------------------------------------------------------
    // 1. Muon Neutrino & Dimuon Charm Candidate Flags
    // -------------------------------------------------------------
    bool isNuMuCC{false};                // nu_mu CC: nu_mu (14) -> mu- (13)
    bool isAntiNuMuCC{false};            // anti-nu_mu CC: anti-nu_mu (-14) -> mu+ (-13)
    bool hasPromptCharmMuon{false};      // Muon produced from prompt semi-leptonic charm decay
    bool isOppositeSignDimuon{false};    // mu_1 and mu_2 have opposite signs (mu- mu+ or mu+ mu-)
    bool hasCandidate{false};            // Golden signature: nu_mu CC + primary muon + prompt charm decay muon

    // -------------------------------------------------------------
    // 2. Primary Outgoing Muon (mu_1 from Interaction Vertex)
    // -------------------------------------------------------------
    int mu1TrackId{-1};
    int mu1Pdg{0};
    int mu1Charge{0};                    // -1 for mu-, +1 for mu+
    double mu1E{0.0};
    double mu1P{0.0};
    double mu1Px{0.0};
    double mu1Py{0.0};
    double mu1Pz{0.0};
    double mu1Pt{0.0};
    double mu1Eta{0.0};
    double mu1Phi{0.0};
    double mu1Theta{0.0};
    double mu1SlopeXZ{0.0};              // dx/dz
    double mu1SlopeYZ{0.0};              // dy/dz

    // -------------------------------------------------------------
    // 3. Secondary Muon from Prompt Charm Decay (mu_2)
    // -------------------------------------------------------------
    int mu2TrackId{-1};
    int mu2Pdg{0};
    int mu2Charge{0};                    // +1 for mu+, -1 for mu-
    int mu2MotherTrackId{-1};
    int mu2MotherPdg{0};
    double mu2E{0.0};
    double mu2P{0.0};
    double mu2Px{0.0};
    double mu2Py{0.0};
    double mu2Pz{0.0};
    double mu2Pt{0.0};
    double mu2Eta{0.0};
    double mu2Phi{0.0};
    double mu2Theta{0.0};
    double mu2SlopeXZ{0.0};              // dx/dz
    double mu2SlopeYZ{0.0};              // dy/dz

    // Displaced track impact parameters and relative kinematics
    double mu2PtRel{0.0};                // Transverse momentum relative to charm hadron direction [GeV/c]
    double mu2IP3D{0.0};                 // 3D distance of closest approach to primary interaction vertex [cm]
    double mu2IPXY{0.0};                 // Transverse 2D impact parameter to primary vertex [cm]
    double mu2OpeningAngleWithCharm{0.0};// Opening angle between charm hadron and decay muon [rad]

    // -------------------------------------------------------------
    // 4. Detailed Charm Decay Vertex & Lifetime
    // -------------------------------------------------------------
    double charmDecayX{0.0};
    double charmDecayY{0.0};
    double charmDecayZ{-9999.0};
    double charmDecayT{0.0};
    double decayLength3D{0.0};           // 3D flight path length L = |r_decay - r_vtx| [cm]
    double decayLengthXY{0.0};           // Transverse flight path length [cm]
    double decayLengthZ{0.0};            // Longitudinal flight path length [cm]
    double properDecayTimeCTau{0.0};     // Proper decay length c*tau = L * m / p [cm]
    double properLifetimePs{0.0};        // Proper lifetime tau [picoseconds]
    int nCharmDecayDaughters{0};         // Total number of daughters from charm decay
    bool charmHasKaonDaughter{false};    // Did charm decay produce an associated kaon (K+/-, K0, K_S)?
    int charmKaonPdg{0};

    // -------------------------------------------------------------
    // 5. Dimuon System Composite Kinematics (mu_1 + mu_2)
    // -------------------------------------------------------------
    double dimuonInvMass{0.0};           // Invariant mass M_{mu mu} [GeV/c^2]
    double dimuonPt{0.0};                // Pair transverse momentum [GeV/c]
    double dimuonP{0.0};                 // Pair total momentum [GeV/c]
    double dimuonOpeningAngle{0.0};      // 3D opening angle between mu_1 and mu_2 [rad]
    double dimuonOpeningAngleMrad{0.0};  // Opening angle [mrad]
    double dimuonDeltaPhi{0.0};          // Azimuthal separation |phi_1 - phi_2| in [0, pi] [rad]
    double dimuonDeltaEta{0.0};          // Pseudorapidity separation |eta_1 - eta_2|
    double dimuonDeltaR{0.0};            // Delta R = sqrt(DeltaEta^2 + DeltaPhi^2)
    double dimuonEnergyAsymmetry{0.0};   // (E_1 - E_2) / (E_1 + E_2)
    double dimuonMomentumRatio{0.0};     // p_2 / p_1
    int nMuonsInEvent{0};                // Total number of muons in event

    // -------------------------------------------------------------
    // 6. MuFilter MCPoints and DS Acceptance
    // -------------------------------------------------------------
    int mu1nDSPoints{0};                 // Total MCPoints in DS (sys == 3) for primary muon (mu_1)
    int mu1nDSHorizontalPoints{0};       // Horizontal plane MCPoints in DS (sys == 3, bar < 60) for mu_1
    int mu1nDSVerticalPoints{0};         // Vertical plane MCPoints in DS (sys == 3, bar >= 60) for mu_1
    bool mu1InDS{false};                 // mu_1 satisfies minDSHorizontalPoints and minDSVerticalPoints
    int mu2nDSPoints{0};                 // Total MCPoints in DS (sys == 3) for charm decay muon (mu_2)
    int mu2nDSHorizontalPoints{0};       // Horizontal plane MCPoints in DS (sys == 3, bar < 60) for mu_2
    int mu2nDSVerticalPoints{0};         // Vertical plane MCPoints in DS (sys == 3, bar >= 60) for mu_2
    bool mu2InDS{false};                 // mu_2 satisfies minDSHorizontalPoints and minDSVerticalPoints
    bool dimuonInDSAcceptance{false};    // BOTH mu_1 and mu_2 satisfy DS acceptance requirements
};

class MuonNeutrinoTruthProcessor : public NeutrinoTruthProcessor {
public:
    explicit MuonNeutrinoTruthProcessor(const NeutrinoTruthConfig& config = NeutrinoTruthConfig())
        : NeutrinoTruthProcessor(config) {}

    virtual ~MuonNeutrinoTruthProcessor() override = default;

    // Specific muon identification method
    int findPrimaryMuon(const TClonesArray& mcTracks, int nuPdg) const;

    // Override generic lepton finder to use muon identification
    int findPrimaryLepton(const TClonesArray& mcTracks, int nuPdg) const override {
        return findPrimaryMuon(mcTracks, nuPdg);
    }

    // Identify muon from prompt semi-leptonic charm decay
    int findCharmDecayMuon(const TClonesArray& mcTracks,
                           const std::vector<int>& charmTrackIds,
                           int primaryMuonTrackId,
                           int& outParentCharmId) const;

    // Primary processor returning specialized MuonNeutrinoTruthInfo
    MuonNeutrinoTruthInfo processMuonNeutrino(const TClonesArray* mcTracks,
                                              const TClonesArray* muFilterPoints = nullptr) const;

    // Callable operator for ROOT RDataFrame:
    // df.Define("truth", muonProcessor, {"MCTrack"})
    MuonNeutrinoTruthInfo operator()(const TClonesArray& mcTracks) const {
        return processMuonNeutrino(&mcTracks, nullptr);
    }

    // Polymorphic base override
    NeutrinoTruthInfo process(const TClonesArray* mcTracks) const override {
        return processMuonNeutrino(mcTracks, nullptr);
    }

private:
    void fillDimuonKinematics(MuonNeutrinoTruthInfo& info,
                              const ShipMCTrack* mu1Trk,
                              const ShipMCTrack* mu2Trk) const;

    void fillMuonCharmKinematics(MuonNeutrinoTruthInfo& info,
                                 const ShipMCTrack* charmTrk,
                                 const ShipMCTrack* mu2Trk,
                                 const TVector3& primaryVtx,
                                 const TClonesArray& mcTracks) const;
};

// Dedicated functor for 2-column RDataFrame evaluation:
// df.Define("truth", muonProcessorWithDS, {"MCTrack", "MuFilterPoint"})
class MuonNeutrinoTruthWithDSProcessor {
public:
    explicit MuonNeutrinoTruthWithDSProcessor(const MuonNeutrinoTruthProcessor& processor = MuonNeutrinoTruthProcessor())
        : m_processor(processor) {}

    MuonNeutrinoTruthInfo operator()(const TClonesArray& mcTracks,
                                     const TClonesArray& muFilterPoints) const {
        return m_processor.processMuonNeutrino(&mcTracks, &muFilterPoints);
    }

private:
    MuonNeutrinoTruthProcessor m_processor;
};

namespace trident {
    using MuonNeutrinoTruthInfo           = snd::MuonNeutrinoTruthInfo;
    using MuonNeutrinoTruthProcessor       = snd::MuonNeutrinoTruthProcessor;
    using MuonNeutrinoTruthWithDSProcessor = snd::MuonNeutrinoTruthWithDSProcessor;
}

} // namespace snd
