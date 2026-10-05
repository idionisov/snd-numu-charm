#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <optional>

#include "TClonesArray.h"
#include "TVector3.h"
#include "TLorentzVector.h"
#include "TDatabasePDG.h"
#include "ShipMCTrack.h"

namespace snd {

enum class InteractionType : int {
    kUnknown      = -1,
    kNC           = 0,
    kNuECC        = 12,
    kAntiNuECC    = -12,
    kNuMuCC       = 14,
    kAntiNuMuCC   = -14,
    kNuTauCC      = 16,
    kAntiNuTauCC  = -16
};

enum class RegionType : int {
    kRegionUnknown    = 0,
    kRegionRock       = 1,   // Upstream rock (Z < targetZMin)
    kRegionTarget     = 2,   // Target / SciFi tracking volume (targetZMin <= Z <= targetZMax)
    kRegionMuonFilter = 3,   // Downstream muon filter / calorimeter (Z > targetZMax)
    kRegionOutside    = 4    // Outside transverse acceptance
};

enum class CharmHadronType : int {
    kNone              = 0,
    kD0                = 421,
    kAntiD0            = -421,
    kDPlus             = 411,
    kDMinus            = -411,
    kDsPlus            = 431,
    kDsMinus           = -431,
    kLambdaCPlus       = 4122,
    kAntiLambdaCMinus  = -4122,
    kOtherCharmMeson   = 1,
    kOtherCharmBaryon  = 2
};

struct NeutrinoTruthConfig {
    // Target & SciFi tracker volume limits [cm]
    double targetZMin{260.0};
    double targetZMax{360.0};
    double fidXMin{-47.5};
    double fidXMax{-8.5};
    double fidYMin{15.5};
    double fidYMax{54.5};
    double fiducialMargin{1.5};

    // Lepton kinematic thresholds
    double minLeptonMomentum{0.0};       // Minimum momentum threshold for primary lepton [GeV/c]
    double maxCharmFlightDistance{20.0}; // Maximum physical flight distance for prompt charm [cm]
    double weightScale{1.0};             // Optional global event weight scale

    // MuFilter DS acceptance thresholds
    int minDSPoints{3};                  // Minimum DS MCPoints required for downstream muon acceptance
};

struct NeutrinoTruthInfo {
    virtual ~NeutrinoTruthInfo() = default;

    // -------------------------------------------------------------
    // 1. Event Identification & Flags
    // -------------------------------------------------------------
    bool isCC{false};                    // Charged Current interaction
    bool isNC{false};                    // Neutral Current interaction
    bool hasCharm{false};                // Charmed hadron produced at interaction vertex
    bool isFiducial{false};              // Primary vertex inside Target fiducial volume
    double mcWeight{1.0};                // Generator / event weight (scaled, defaults to 1.0 if raw weight <= 0)
    double rawWeight{0.0};               // Raw MCTrack->GetWeight() directly from neutrino track

    InteractionType interactionType{InteractionType::kUnknown};
    RegionType regionType{RegionType::kRegionUnknown};
    std::string interactionName{""};

    // -------------------------------------------------------------
    // 2. Incoming Neutrino & Primary Interaction Vertex
    // -------------------------------------------------------------
    int nuTrackId{-1};
    int nuPdg{0};
    double nuE{0.0};                     // Neutrino energy [GeV]
    double nuP{0.0};                     // Neutrino momentum [GeV/c]
    double nuPx{0.0};
    double nuPy{0.0};
    double nuPz{0.0};
    double nuPt{0.0};
    double nuEta{0.0};
    double nuPhi{0.0};
    double nuTheta{0.0};                 // Angle with respect to z-axis [rad]

    // Primary interaction vertex [cm, ns]
    double vtxX{0.0};
    double vtxY{0.0};
    double vtxZ{-9999.0};
    double vtxT{0.0};

    // -------------------------------------------------------------
    // 3. Deep Inelastic Scattering (DIS) & Interaction Kinematics
    // -------------------------------------------------------------
    double Q2{0.0};                      // 4-momentum transfer squared Q^2 = -(q)^2 [GeV^2]
    double BjorkenX{0.0};                // Bjorken scaling variable x = Q^2 / (2 * M_N * nu)
    double InelasticityY{0.0};           // Inelasticity y = (E_nu - E_lepton) / E_nu
    double HadronicW{0.0};               // Invariant mass of the hadronic recoil system [GeV/c^2]

    // -------------------------------------------------------------
    // 4. Primary Outgoing Charged Lepton (e, mu, tau for CC)
    // -------------------------------------------------------------
    int primaryLeptonTrackId{-1};
    int primaryLeptonPdg{0};
    int primaryLeptonCharge{0};
    double primaryLeptonE{0.0};
    double primaryLeptonP{0.0};
    double primaryLeptonPx{0.0};
    double primaryLeptonPy{0.0};
    double primaryLeptonPz{0.0};
    double primaryLeptonPt{0.0};
    double primaryLeptonEta{0.0};
    double primaryLeptonPhi{0.0};
    double primaryLeptonTheta{0.0};
    double primaryLeptonSlopeXZ{0.0};   // dx/dz
    double primaryLeptonSlopeYZ{0.0};   // dy/dz

    // -------------------------------------------------------------
    // 5. Charmed Hadron (Leading charm produced at Primary Vertex)
    // -------------------------------------------------------------
    int charmTrackId{-1};
    int charmPdg{0};
    int charmQuarkContent{0};            // +1 for charm (c), -1 for anti-charm (cbar)
    std::string charmName{""};
    CharmHadronType charmType{CharmHadronType::kNone};
    double charmMass{0.0};               // [GeV/c^2]
    double charmE{0.0};
    double charmP{0.0};
    double charmPx{0.0};
    double charmPy{0.0};
    double charmPz{0.0};
    double charmPt{0.0};
    double charmEta{0.0};
    double charmPhi{0.0};
    double charmTheta{0.0};
    double charmEnergyFractionZ{0.0};    // z_c = E_charm / E_hadronic

    // -------------------------------------------------------------
    // 6. Multiplicities & Hadronic Recoil System
    // -------------------------------------------------------------
    int nPrimaryTracks{0};               // Number of primary tracks from interaction vertex
    int nPrimaryHadrons{0};              // Number of primary hadrons from interaction vertex
    int nCharmedHadronsInEvent{0};       // Total number of charmed hadrons found
    double hadronicEnergyTotal{0.0};     // Sum of energy of all primary hadrons [GeV]
    double hadronicRecoilPt{0.0};        // Transverse momentum of hadronic system [GeV/c]
    double missingPt{0.0};               // Transverse missing momentum relative to neutrino beam [GeV/c]

    const char* getInteractionName() const {
        return interactionName.c_str();
    }

    const char* getRegionName() const {
        switch (regionType) {
            case RegionType::kRegionRock:       return "Rock";
            case RegionType::kRegionTarget:     return "Target";
            case RegionType::kRegionMuonFilter: return "MuonFilter";
            case RegionType::kRegionOutside:    return "Outside";
            default:                            return "Unknown";
        }
    }
};

class NeutrinoTruthProcessor {
public:
    explicit NeutrinoTruthProcessor(const NeutrinoTruthConfig& config = NeutrinoTruthConfig())
        : fConfig(config) {}

    virtual ~NeutrinoTruthProcessor() = default;

    // Core processing method
    virtual NeutrinoTruthInfo process(const TClonesArray* mcTracks) const;

    // Callable operator for ROOT RDataFrame:
    // df.Define("truth", processor, {"MCTrack"})
    NeutrinoTruthInfo operator()(const TClonesArray& mcTracks) const {
        return process(&mcTracks);
    }

    // Lepton identification (customizable / overridable in derived classes)
    virtual int findPrimaryLepton(const TClonesArray& mcTracks, int nuPdg) const;

    // Static physics helper methods
    static bool isCharmedHadron(int pdgCode);
    static bool isOpenCharm(int pdgCode);
    static bool isCharmedMeson(int pdgCode);
    static bool isCharmedBaryon(int pdgCode);
    static int getCharmQuarkContent(int pdgCode);
    static CharmHadronType getCharmHadronType(int pdgCode);
    static std::string getParticleName(int pdgCode);
    static double getNominalMass(int pdgCode);

    // Topological and geometric calculation methods
    static double compute3DImpactParameter(const TVector3& primaryVtx,
                                           const TVector3& decayVtx,
                                           const TVector3& pTrack);

    static double computeTransverseImpactParameter(double vtxX, double vtxY,
                                                   double decayX, double decayY,
                                                   double px, double py);

    static double computePtRel(const TVector3& pTrack, const TVector3& pRefAxis);

    // Geometry & Fiducial checks
    bool isFiducial(double x, double y, double z) const;
    RegionType determineRegion(double z, double x = 0.0, double y = 0.0) const;

    // Configuration getters & setters
    void setConfig(const NeutrinoTruthConfig& cfg) { fConfig = cfg; }
    const NeutrinoTruthConfig& getConfig() const { return fConfig; }

protected:
    NeutrinoTruthConfig fConfig;

    // Search and reconstruction helpers
    std::vector<int> findPrimaryCharmedHadrons(const TClonesArray& mcTracks) const;

    void fillNeutrinoKinematics(NeutrinoTruthInfo& info,
                                const ShipMCTrack* nuTrk,
                                const ShipMCTrack* leptonTrk,
                                const TClonesArray& mcTracks) const;

    void fillPrimaryCharmKinematics(NeutrinoTruthInfo& info,
                                    const ShipMCTrack* charmTrk,
                                    const TClonesArray& mcTracks) const;
};

namespace trident {
    using NeutrinoTruthConfig    = snd::NeutrinoTruthConfig;
    using NeutrinoTruthInfo      = snd::NeutrinoTruthInfo;
    using NeutrinoTruthProcessor = snd::NeutrinoTruthProcessor;
}

} // namespace snd
