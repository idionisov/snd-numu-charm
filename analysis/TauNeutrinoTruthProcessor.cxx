#include "TauNeutrinoTruthProcessor.h"
#include <cmath>
#include <algorithm>

namespace snd {

namespace {
    constexpr double kTauMass      = 1.77686;      // GeV/c^2
    constexpr double kSpeedOfLight = 0.0299792458; // cm / ps

    inline double calculateEta(double p, double pz) {
        if (p - pz > 1e-9 && p + pz > 1e-9) {
            return 0.5 * std::log((p + pz) / (p - pz));
        }
        return (pz > 0.0) ? 999.0 : -999.0;
    }
}

int TauNeutrinoTruthProcessor::findPrimaryTau(const TClonesArray& mcTracks, int nuPdg) const {
    const int nTracks = mcTracks.GetEntriesFast();
    if (nTracks < 2) return -1;

    const int expectedPdg = (nuPdg > 0) ? 15 : -15;

    // Check track 1 directly
    auto* trk1 = static_cast<ShipMCTrack*>(mcTracks.At(1));
    if (trk1 && trk1->GetMotherId() == 0 && trk1->GetPdgCode() == expectedPdg) {
        if (trk1->GetP() >= fConfig.minLeptonMomentum) return 1;
    }

    // Scan all primaries
    int bestId = -1;
    double maxP = -1.0;
    for (int i = 1; i < nTracks; ++i) {
        auto* trk = static_cast<ShipMCTrack*>(mcTracks.At(i));
        if (!trk || trk->GetMotherId() != 0) continue;
        if (trk->GetPdgCode() == expectedPdg) {
            if (trk->GetP() > maxP && trk->GetP() >= fConfig.minLeptonMomentum) {
                maxP = trk->GetP();
                bestId = i;
            }
        }
    }
    return bestId;
}

TauNeutrinoTruthInfo TauNeutrinoTruthProcessor::processTauNeutrino(
    const TClonesArray* mcTracks
) const {
    TauNeutrinoTruthInfo info;
    if (!mcTracks || mcTracks->GetEntriesFast() == 0) return info;

    const int nTracks = mcTracks->GetEntriesFast();
    auto* nuTrk = static_cast<ShipMCTrack*>(mcTracks->At(0));
    if (!nuTrk) return info;

    int nuPdg = nuTrk->GetPdgCode();
    info.nuFlavor = std::abs(nuPdg);
    info.isNeutrino = (nuPdg > 0);

    // Primary outgoing tau
    int tau1Id = findPrimaryTau(*mcTracks, nuPdg);
    const ShipMCTrack* tau1Trk = (tau1Id >= 0 && tau1Id < nTracks) ? static_cast<ShipMCTrack*>(mcTracks->At(tau1Id)) : nullptr;

    if (tau1Trk) {
        info.isCC = true;
        info.isNC = false;
        info.tau1TrackId = tau1Id;
        info.tau1Pdg     = tau1Trk->GetPdgCode();
        info.tau1Charge  = (info.tau1Pdg > 0) ? -1 : +1;
        info.tau1Px      = tau1Trk->GetPx();
        info.tau1Py      = tau1Trk->GetPy();
        info.tau1Pz      = tau1Trk->GetPz();
        info.tau1P       = tau1Trk->GetP();
        info.tau1Pt      = tau1Trk->GetPt();
        info.tau1E       = tau1Trk->GetEnergy();
        info.tau1Phi     = std::atan2(info.tau1Py, info.tau1Px);
        info.tau1Theta   = (info.tau1P > 1e-9) ? std::acos(std::clamp(info.tau1Pz / info.tau1P, -1.0, 1.0)) : 0.0;
        info.tau1Eta     = calculateEta(info.tau1P, info.tau1Pz);

        if (std::abs(info.tau1Pz) > 1e-9) {
            info.tau1SlopeXZ = info.tau1Px / info.tau1Pz;
            info.tau1SlopeYZ = info.tau1Py / info.tau1Pz;
        }

        info.primaryLeptonTrackId = tau1Id;
        info.primaryLeptonPdg = info.tau1Pdg;
        info.primaryLeptonCharge = info.tau1Charge;
        info.primaryLeptonP = info.tau1P;
        info.primaryLeptonE = info.tau1E;

        if (nuPdg == 16) {
            info.isNuTauCC = true;
            info.interactionType = InteractionType::kNuTauCC;
            info.interactionName = "nu_tau_CC";
        } else if (nuPdg == -16) {
            info.isAntiNuTauCC = true;
            info.interactionType = InteractionType::kAntiNuTauCC;
            info.interactionName = "anti_nu_tau_CC";
        }

        // Trace tau decay daughters
        int nDaughters = 0;
        int nCharged = 0;
        bool hasMu = false;
        bool hasE = false;
        double decayX = 0, decayY = 0, decayZ = -9999;

        for (int i = 1; i < nTracks; ++i) {
            auto* dTrk = static_cast<ShipMCTrack*>(mcTracks->At(i));
            if (!dTrk || dTrk->GetMotherId() != tau1Id) continue;
            nDaughters++;
            int dPdg = std::abs(dTrk->GetPdgCode());
            if (dPdg == 13) hasMu = true;
            if (dPdg == 11) hasE = true;
            if (dPdg == 11 || dPdg == 13 || dPdg == 211 || dPdg == 321 || dPdg == 2212) {
                nCharged++;
            }
            decayX = dTrk->GetStartX();
            decayY = dTrk->GetStartY();
            decayZ = dTrk->GetStartZ();
        }

        info.nTauDaughters = nDaughters;
        info.tauDecayX = decayX;
        info.tauDecayY = decayY;
        info.tauDecayZ = decayZ;

        if (decayZ > -9000.0) {
            double dx = decayX - nuTrk->GetStartX();
            double dy = decayY - nuTrk->GetStartY();
            double dz = decayZ - nuTrk->GetStartZ();
            info.tauFlightLength3D = std::sqrt(dx*dx + dy*dy + dz*dz);
            info.tauFlightLengthXY = std::hypot(dx, dy);
            if (info.tau1P > 1e-6) {
                double cTau = info.tauFlightLength3D * (kTauMass / info.tau1P);
                info.tauProperLifetimePs = cTau / kSpeedOfLight;
            }
        }

        if (hasMu) {
            info.tauDecayMode = "to_muon";
            info.tauLeptonicDaughterPdg = 13;
        } else if (hasE) {
            info.tauDecayMode = "to_electron";
            info.tauLeptonicDaughterPdg = 11;
        } else if (nCharged == 1) {
            info.tauDecayMode = "1prong_hadron";
        } else if (nCharged == 3) {
            info.tauDecayMode = "3prong_hadron";
        } else {
            info.tauDecayMode = "other";
        }

        info.hasCandidate = true;
    } else {
        info.isCC = false;
        info.isNC = true;
        info.interactionType = InteractionType::kNC;
        info.interactionName = (nuPdg > 0) ? "nu_tau_NC" : "anti_nu_tau_NC";
    }

    fillNeutrinoKinematics(info, nuTrk, tau1Trk, *mcTracks);

    // Charmed hadrons
    std::vector<int> charmIds = findPrimaryCharmedHadrons(*mcTracks);
    info.nCharmedHadronsInEvent = static_cast<int>(charmIds.size());
    info.hasCharm = !charmIds.empty();

    if (info.hasCharm) {
        info.charmTrackId = charmIds.front();
        const ShipMCTrack* charmTrk = static_cast<ShipMCTrack*>(mcTracks->At(info.charmTrackId));
        fillPrimaryCharmKinematics(info, charmTrk, *mcTracks);
    }

    return info;
}

} // namespace snd
