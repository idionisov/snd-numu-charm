#include "ElectronNeutrinoTruthProcessor.h"
#include <cmath>
#include <algorithm>

namespace snd {

namespace {
    constexpr double kElectronMass = 0.0005109989; // GeV/c^2

    inline double calculateEta(double p, double pz) {
        if (p - pz > 1e-9 && p + pz > 1e-9) {
            return 0.5 * std::log((p + pz) / (p - pz));
        }
        return (pz > 0.0) ? 999.0 : -999.0;
    }
}

int ElectronNeutrinoTruthProcessor::findPrimaryElectron(const TClonesArray& mcTracks, int nuPdg) const {
    const int nTracks = mcTracks.GetEntriesFast();
    if (nTracks < 2) return -1;

    const int expectedPdg = (nuPdg > 0) ? 11 : -11;

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

int ElectronNeutrinoTruthProcessor::findCharmDecayElectron(
    const TClonesArray& mcTracks,
    const std::vector<int>& charmTrackIds,
    int primaryElectronTrackId,
    int& outParentCharmId
) const {
    outParentCharmId = -1;
    const int nTracks = mcTracks.GetEntriesFast();
    if (nTracks < 2) return -1;

    int bestId = -1;
    double maxP = -1.0;

    for (int i = 1; i < nTracks; ++i) {
        if (i == primaryElectronTrackId) continue;
        auto* trk = static_cast<ShipMCTrack*>(mcTracks.At(i));
        if (!trk) continue;

        if (std::abs(trk->GetPdgCode()) != 11) continue; // Must be an electron
        if (trk->GetP() < fConfig.minLeptonMomentum) continue;

        int directMotherId = trk->GetMotherId();
        int matchedCharmId = -1;

        if (fConfig.requireDirectCharmDecay) {
            if (directMotherId >= 0 && directMotherId < nTracks) {
                auto* parentTrk = static_cast<ShipMCTrack*>(mcTracks.At(directMotherId));
                if (parentTrk && isCharmedHadron(parentTrk->GetPdgCode())) {
                    matchedCharmId = directMotherId;
                }
            }
        } else {
            int currentMotherId = directMotherId;
            while (currentMotherId > 0 && currentMotherId < nTracks) {
                auto* parentTrk = static_cast<ShipMCTrack*>(mcTracks.At(currentMotherId));
                if (!parentTrk) break;
                if (isCharmedHadron(parentTrk->GetPdgCode())) {
                    matchedCharmId = currentMotherId;
                    break;
                }
                currentMotherId = parentTrk->GetMotherId();
            }
        }

        if (matchedCharmId != -1) {
            auto* charmTrk = static_cast<ShipMCTrack*>(mcTracks.At(matchedCharmId));
            double flightDist = 0.0;
            if (charmTrk) {
                double dx = trk->GetStartX() - charmTrk->GetStartX();
                double dy = trk->GetStartY() - charmTrk->GetStartY();
                double dz = trk->GetStartZ() - charmTrk->GetStartZ();
                flightDist = std::sqrt(dx*dx + dy*dy + dz*dz);
            }
            if (flightDist <= fConfig.maxCharmFlightDistance) {
                if (trk->GetP() > maxP) {
                    maxP = trk->GetP();
                    bestId = i;
                    outParentCharmId = matchedCharmId;
                }
            }
        }
    }
    return bestId;
}

ElectronNeutrinoTruthInfo ElectronNeutrinoTruthProcessor::processElectronNeutrino(
    const TClonesArray* mcTracks
) const {
    ElectronNeutrinoTruthInfo info;
    if (!mcTracks || mcTracks->GetEntriesFast() == 0) return info;

    const int nTracks = mcTracks->GetEntriesFast();
    auto* nuTrk = static_cast<ShipMCTrack*>(mcTracks->At(0));
    if (!nuTrk) return info;

    int nuPdg = nuTrk->GetPdgCode();
    info.nuFlavor = std::abs(nuPdg);
    info.isNeutrino = (nuPdg > 0);

    // Primary outgoing electron
    int e1Id = findPrimaryElectron(*mcTracks, nuPdg);
    const ShipMCTrack* e1Trk = (e1Id >= 0 && e1Id < nTracks) ? static_cast<ShipMCTrack*>(mcTracks->At(e1Id)) : nullptr;

    if (e1Trk) {
        info.isCC = true;
        info.isNC = false;
        info.e1TrackId = e1Id;
        info.e1Pdg     = e1Trk->GetPdgCode();
        info.e1Charge  = (info.e1Pdg > 0) ? -1 : +1;
        info.e1Px      = e1Trk->GetPx();
        info.e1Py      = e1Trk->GetPy();
        info.e1Pz      = e1Trk->GetPz();
        info.e1P       = e1Trk->GetP();
        info.e1Pt      = e1Trk->GetPt();
        info.e1E       = e1Trk->GetEnergy();
        info.e1Phi     = std::atan2(info.e1Py, info.e1Px);
        info.e1Theta   = (info.e1P > 1e-9) ? std::acos(std::clamp(info.e1Pz / info.e1P, -1.0, 1.0)) : 0.0;
        info.e1Eta     = calculateEta(info.e1P, info.e1Pz);

        if (std::abs(info.e1Pz) > 1e-9) {
            info.e1SlopeXZ = info.e1Px / info.e1Pz;
            info.e1SlopeYZ = info.e1Py / info.e1Pz;
        }

        info.primaryLeptonTrackId = e1Id;
        info.primaryLeptonPdg = info.e1Pdg;
        info.primaryLeptonCharge = info.e1Charge;
        info.primaryLeptonP = info.e1P;
        info.primaryLeptonE = info.e1E;

        if (nuPdg == 12) {
            info.isNuECC = true;
            info.interactionType = InteractionType::kNuECC;
            info.interactionName = "nu_e_CC";
        } else if (nuPdg == -12) {
            info.isAntiNuECC = true;
            info.interactionType = InteractionType::kAntiNuECC;
            info.interactionName = "anti_nu_e_CC";
        }
    } else {
        info.isCC = false;
        info.isNC = true;
        info.interactionType = InteractionType::kNC;
        info.interactionName = (nuPdg > 0) ? "nu_e_NC" : "anti_nu_e_NC";
    }

    fillNeutrinoKinematics(info, nuTrk, e1Trk, *mcTracks);

    // Charmed hadrons
    std::vector<int> charmIds = findPrimaryCharmedHadrons(*mcTracks);
    info.nCharmedHadronsInEvent = static_cast<int>(charmIds.size());
    info.hasCharm = !charmIds.empty();

    if (info.hasCharm) {
        info.charmTrackId = charmIds.front();
        const ShipMCTrack* charmTrk = static_cast<ShipMCTrack*>(mcTracks->At(info.charmTrackId));
        fillPrimaryCharmKinematics(info, charmTrk, *mcTracks);

        int parentCharmId = -1;
        int e2Id = findCharmDecayElectron(*mcTracks, charmIds, e1Id, parentCharmId);
        if (e2Id >= 0 && e2Id < nTracks) {
            auto* e2Trk = static_cast<ShipMCTrack*>(mcTracks->At(e2Id));
            info.hasPromptCharmElectron = true;
            info.lep2TrackId = e2Id;
            info.lep2Pdg = e2Trk->GetPdgCode();
            info.lep2Charge = (info.lep2Pdg > 0) ? -1 : +1;
            info.lep2E = e2Trk->GetEnergy();
            info.lep2P = e2Trk->GetP();
            info.lep2Pt = e2Trk->GetPt();
            if (e1Trk) {
                info.isOppositeSignDielectron = (info.e1Pdg * info.lep2Pdg < 0);
                info.hasCandidate = info.isCC && info.isOppositeSignDielectron;
            }
        }
    }

    return info;
}

} // namespace snd
