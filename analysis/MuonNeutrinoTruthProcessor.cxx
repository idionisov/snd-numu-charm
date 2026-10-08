#include "MuonNeutrinoTruthProcessor.h"
#include "FairMCPoint.h"

#include <cmath>
#include <algorithm>

namespace snd {

namespace {
    constexpr double kMuonMass     = 0.1056583755; // GeV/c^2
    constexpr double kSpeedOfLight = 0.0299792458; // cm / ps

    inline double foldAngleToPi(double dphi) {
        dphi = std::abs(dphi);
        while (dphi > M_PI) dphi = std::abs(2.0 * M_PI - dphi);
        return dphi;
    }

    inline double calculateEta(double p, double pz) {
        if (p - pz > 1e-9 && p + pz > 1e-9) {
            return 0.5 * std::log((p + pz) / (p - pz));
        }
        return (pz > 0.0) ? 999.0 : -999.0;
    }
}

int MuonNeutrinoTruthProcessor::findPrimaryMuon(const TClonesArray& mcTracks, int nuPdg) const {
    const int nTracks = mcTracks.GetEntriesFast();
    if (nTracks < 2) return -1;

    // Expected primary muon PDG:
    // nu_mu (14) -> mu- (13)
    // anti_nu_mu (-14) -> mu+ (-13)
    const int expectedPrimaryMuonPdg = (nuPdg > 0) ? 13 : -13;

    // First check track 1 directly (standard GENIE / sndsw primary outgoing lepton convention)
    auto* trk1 = static_cast<ShipMCTrack*>(mcTracks.At(1));
    if (trk1 && trk1->GetMotherId() == 0 && trk1->GetPdgCode() == expectedPrimaryMuonPdg) {
        if (trk1->GetP() >= fConfig.minLeptonMomentum) {
            return 1;
        }
    }

    // Secondary scan across all primary particles (motherId == 0)
    int bestId = -1;
    double maxP = -1.0;
    for (int i = 1; i < nTracks; ++i) {
        auto* trk = static_cast<ShipMCTrack*>(mcTracks.At(i));
        if (!trk || trk->GetMotherId() != 0) continue;

        if (trk->GetPdgCode() == expectedPrimaryMuonPdg) {
            double p = trk->GetP();
            if (p > maxP && p >= fConfig.minLeptonMomentum) {
                maxP = p;
                bestId = i;
            }
        }
    }

    if (bestId != -1) return bestId;

    // Fallback: any muon from primary vertex
    for (int i = 1; i < nTracks; ++i) {
        auto* trk = static_cast<ShipMCTrack*>(mcTracks.At(i));
        if (!trk || trk->GetMotherId() != 0) continue;

        if (std::abs(trk->GetPdgCode()) == 13) {
            double p = trk->GetP();
            if (p > maxP && p >= fConfig.minLeptonMomentum) {
                maxP = p;
                bestId = i;
            }
        }
    }

    return bestId;
}

int MuonNeutrinoTruthProcessor::findCharmDecayMuon(
    const TClonesArray& mcTracks,
    const std::vector<int>& charmTrackIds,
    int primaryMuonTrackId,
    int& outParentCharmId
) const {
    outParentCharmId = -1;
    const int nTracks = mcTracks.GetEntriesFast();
    if (nTracks < 2) return -1;

    int bestMuonId = -1;
    double maxMuonP = -1.0;

    for (int i = 1; i < nTracks; ++i) {
        if (i == primaryMuonTrackId) continue;

        auto* trk = static_cast<ShipMCTrack*>(mcTracks.At(i));
        if (!trk) continue;

        if (std::abs(trk->GetPdgCode()) != 13) continue; // Must be a muon
        if (trk->GetP() < fConfig.minLeptonMomentum) continue;

        int directMotherId = trk->GetMotherId();
        int matchedCharmId = -1;

        if (fConfig.requireDirectCharmDecay) {
            // Muon must be an immediate direct decay daughter of a charmed hadron
            if (directMotherId >= 0 && directMotherId < nTracks) {
                auto* parentTrk = static_cast<ShipMCTrack*>(mcTracks.At(directMotherId));
                if (parentTrk && isCharmedHadron(parentTrk->GetPdgCode())) {
                    matchedCharmId = directMotherId;
                }
            }
        } else {
            // Loose mode: trace mother ancestry upwards across any intermediate particles
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
            // Check physical flight distance
            auto* charmTrk = static_cast<ShipMCTrack*>(mcTracks.At(matchedCharmId));
            double flightDist = 0.0;
            if (charmTrk) {
                double dx = trk->GetStartX() - charmTrk->GetStartX();
                double dy = trk->GetStartY() - charmTrk->GetStartY();
                double dz = trk->GetStartZ() - charmTrk->GetStartZ();
                flightDist = std::sqrt(dx*dx + dy*dy + dz*dz);
            }

            if (flightDist <= fConfig.maxCharmFlightDistance) {
                if (trk->GetP() > maxMuonP) {
                    maxMuonP = trk->GetP();
                    bestMuonId = i;
                    outParentCharmId = matchedCharmId;
                }
            }
        }
    }

    return bestMuonId;
}

void MuonNeutrinoTruthProcessor::fillDimuonKinematics(
    MuonNeutrinoTruthInfo& info,
    const ShipMCTrack* mu1Trk,
    const ShipMCTrack* mu2Trk
) const {
    if (!mu1Trk || !mu2Trk) return;

    TLorentzVector p4Mu1, p4Mu2;
    p4Mu1.SetXYZM(mu1Trk->GetPx(), mu1Trk->GetPy(), mu1Trk->GetPz(), kMuonMass);
    p4Mu2.SetXYZM(mu2Trk->GetPx(), mu2Trk->GetPy(), mu2Trk->GetPz(), kMuonMass);

    TLorentzVector p4Dimuon = p4Mu1 + p4Mu2;

    info.dimuonInvMass = p4Dimuon.M();
    info.dimuonPt      = p4Dimuon.Pt();
    info.dimuonP       = p4Dimuon.P();

    TVector3 v3Mu1 = p4Mu1.Vect();
    TVector3 v3Mu2 = p4Mu2.Vect();

    if (v3Mu1.Mag() > 1e-9 && v3Mu2.Mag() > 1e-9) {
        info.dimuonOpeningAngle     = v3Mu1.Angle(v3Mu2);
        info.dimuonOpeningAngleMrad = info.dimuonOpeningAngle * 1000.0;
    }

    info.dimuonDeltaPhi = foldAngleToPi(info.mu1Phi - info.mu2Phi);
    info.dimuonDeltaEta = std::abs(info.mu1Eta - info.mu2Eta);
    info.dimuonDeltaR   = std::hypot(info.dimuonDeltaEta, info.dimuonDeltaPhi);

    double sumE = info.mu1E + info.mu2E;
    info.dimuonEnergyAsymmetry = (sumE > 1e-6) ? ((info.mu1E - info.mu2E) / sumE) : 0.0;
    info.dimuonMomentumRatio   = (info.mu1P > 1e-6) ? (info.mu2P / info.mu1P) : 0.0;
}

void MuonNeutrinoTruthProcessor::fillMuonCharmKinematics(
    MuonNeutrinoTruthInfo& info,
    const ShipMCTrack* charmTrk,
    const ShipMCTrack* mu2Trk,
    const TVector3& primaryVtx,
    const TClonesArray& mcTracks
) const {
    if (!charmTrk) return;

    // Charm decay vertex is the production vertex of mu_2
    if (mu2Trk) {
        info.charmDecayX = mu2Trk->GetStartX();
        info.charmDecayY = mu2Trk->GetStartY();
        info.charmDecayZ = mu2Trk->GetStartZ();
        info.charmDecayT = mu2Trk->GetStartT();

        TVector3 decayVtx(info.charmDecayX, info.charmDecayY, info.charmDecayZ);
        TVector3 flightVec = decayVtx - primaryVtx;

        info.decayLength3D = flightVec.Mag();
        info.decayLengthXY = std::hypot(flightVec.X(), flightVec.Y());
        info.decayLengthZ  = flightVec.Z();

        if (info.charmP > 1e-6) {
            info.properDecayTimeCTau = info.decayLength3D * (info.charmMass / info.charmP);
            info.properLifetimePs    = info.properDecayTimeCTau / kSpeedOfLight;
        }

        TVector3 pCharm(info.charmPx, info.charmPy, info.charmPz);
        TVector3 pMu2(mu2Trk->GetPx(), mu2Trk->GetPy(), mu2Trk->GetPz());

        info.mu2PtRel = computePtRel(pMu2, pCharm);
        info.mu2IP3D  = compute3DImpactParameter(primaryVtx, decayVtx, pMu2);
        info.mu2IPXY  = computeTransverseImpactParameter(
            primaryVtx.X(), primaryVtx.Y(),
            info.charmDecayX, info.charmDecayY,
            pMu2.X(), pMu2.Y()
        );

        if (pCharm.Mag() > 1e-9 && pMu2.Mag() > 1e-9) {
            info.mu2OpeningAngleWithCharm = pCharm.Angle(pMu2);
        }

        // Inspect charm decay daughters & search for associated kaons
        const int nTracks = mcTracks.GetEntriesFast();
        int nDaughters = 0;
        for (int i = 1; i < nTracks; ++i) {
            auto* trk = static_cast<ShipMCTrack*>(mcTracks.At(i));
            if (!trk) continue;

            if (trk->GetMotherId() == info.charmTrackId || trk->GetMotherId() == info.mu2MotherTrackId) {
                nDaughters++;
                int pdg = std::abs(trk->GetPdgCode());
                if (pdg == 321 || pdg == 311 || pdg == 310 || pdg == 130) {
                    info.charmHasKaonDaughter = true;
                    info.charmKaonPdg = trk->GetPdgCode();
                }
            }
        }
        info.nCharmDecayDaughters = nDaughters;
    }
}

MuonNeutrinoTruthInfo MuonNeutrinoTruthProcessor::processMuonNeutrino(
    const TClonesArray* mcTracks,
    const TClonesArray* muFilterPoints) const {
    MuonNeutrinoTruthInfo info;
    if (!mcTracks || mcTracks->GetEntriesFast() == 0) {
        return info;
    }

    const int nTracks = mcTracks->GetEntriesFast();

    // 1. Primary neutrino track (Track 0)
    auto* nuTrk = static_cast<ShipMCTrack*>(mcTracks->At(0));
    if (!nuTrk) return info;

    int nuPdg = nuTrk->GetPdgCode();
    TVector3 primaryVtx(nuTrk->GetStartX(), nuTrk->GetStartY(), nuTrk->GetStartZ());

    // Count muons and primary particles
    int nMuons = 0;
    int nPrimaries = 0;
    for (int i = 0; i < nTracks; ++i) {
        auto* trk = static_cast<ShipMCTrack*>(mcTracks->At(i));
        if (!trk) continue;
        if (std::abs(trk->GetPdgCode()) == 13) nMuons++;
        if (trk->GetMotherId() == 0) nPrimaries++;
    }
    info.nMuonsInEvent   = nMuons;
    info.nPrimaryTracks  = nPrimaries;

    // 2. Identify Primary Outgoing Muon (mu_1) using specialized findPrimaryMuon
    int mu1Id = findPrimaryMuon(*mcTracks, nuPdg);
    const ShipMCTrack* mu1Trk = (mu1Id >= 0 && mu1Id < nTracks) ? static_cast<ShipMCTrack*>(mcTracks->At(mu1Id)) : nullptr;

    if (mu1Trk) {
        info.mu1TrackId = mu1Id;
        info.mu1Pdg     = mu1Trk->GetPdgCode();
        info.mu1Charge  = (info.mu1Pdg == 13) ? -1 : +1;
        info.mu1Px      = mu1Trk->GetPx();
        info.mu1Py      = mu1Trk->GetPy();
        info.mu1Pz      = mu1Trk->GetPz();
        info.mu1P       = mu1Trk->GetP();
        info.mu1Pt      = mu1Trk->GetPt();
        info.mu1E       = mu1Trk->GetEnergy();
        info.mu1Phi     = std::atan2(info.mu1Py, info.mu1Px);
        info.mu1Theta   = (info.mu1P > 1e-9) ? std::acos(std::clamp(info.mu1Pz / info.mu1P, -1.0, 1.0)) : 0.0;
        info.mu1Eta     = calculateEta(info.mu1P, info.mu1Pz);

        if (std::abs(info.mu1Pz) > 1e-9) {
            info.mu1SlopeXZ = info.mu1Px / info.mu1Pz;
            info.mu1SlopeYZ = info.mu1Py / info.mu1Pz;
        }

        // Set base lepton fields
        info.primaryLeptonTrackId = info.mu1TrackId;
        info.primaryLeptonPdg     = info.mu1Pdg;
        info.primaryLeptonCharge  = info.mu1Charge;
        info.primaryLeptonE       = info.mu1E;
        info.primaryLeptonP       = info.mu1P;
        info.primaryLeptonPx      = info.mu1Px;
        info.primaryLeptonPy      = info.mu1Py;
        info.primaryLeptonPz      = info.mu1Pz;
        info.primaryLeptonPt      = info.mu1Pt;
        info.primaryLeptonEta     = info.mu1Eta;
        info.primaryLeptonPhi     = info.mu1Phi;
        info.primaryLeptonTheta   = info.mu1Theta;
        info.primaryLeptonSlopeXZ = info.mu1SlopeXZ;
        info.primaryLeptonSlopeYZ = info.mu1SlopeYZ;

        // Charged Current categorization
        info.isCC = true;
        info.isNC = false;
        if (nuPdg == 14 && info.mu1Pdg == 13) {
            info.isNuMuCC = true;
            info.interactionType = InteractionType::kNuMuCC;
            info.interactionName = "nu_mu_CC";
        } else if (nuPdg == -14 && info.mu1Pdg == -13) {
            info.isAntiNuMuCC = true;
            info.interactionType = InteractionType::kAntiNuMuCC;
            info.interactionName = "anti_nu_mu_CC";
        } else {
            info.interactionType = InteractionType::kUnknown;
            info.interactionName = "other_CC";
        }
    } else {
        info.isCC = false;
        info.isNC = true;
        info.interactionType = InteractionType::kNC;
        info.interactionName = (nuPdg > 0) ? "nu_mu_NC" : "anti_nu_mu_NC";
    }

    // 3. Fill Neutrino & DIS Kinematics
    fillNeutrinoKinematics(info, nuTrk, mu1Trk, *mcTracks);

    // 4. Identify Primary Charmed Hadrons
    std::vector<int> charmIds = findPrimaryCharmedHadrons(*mcTracks);
    info.nCharmedHadronsInEvent = static_cast<int>(charmIds.size());
    info.hasCharm = !charmIds.empty();

    // 5. Identify Prompt Semi-Leptonic Decay Muon (mu_2)
    int parentCharmId = -1;
    int mu2Id = findCharmDecayMuon(*mcTracks, charmIds, mu1Id, parentCharmId);
    const ShipMCTrack* mu2Trk = (mu2Id >= 0 && mu2Id < nTracks) ? static_cast<ShipMCTrack*>(mcTracks->At(mu2Id)) : nullptr;

    if (mu2Trk) {
        info.hasPromptCharmMuon = true;
        info.mu2TrackId  = mu2Id;
        info.mu2Pdg      = mu2Trk->GetPdgCode();
        info.mu2Charge   = (info.mu2Pdg == 13) ? -1 : +1;
        info.mu2MotherTrackId = mu2Trk->GetMotherId();

        auto* motherTrk = (info.mu2MotherTrackId >= 0 && info.mu2MotherTrackId < nTracks)
            ? static_cast<ShipMCTrack*>(mcTracks->At(info.mu2MotherTrackId)) : nullptr;
        if (motherTrk) {
            info.mu2MotherPdg = motherTrk->GetPdgCode();
        }

        info.mu2Px    = mu2Trk->GetPx();
        info.mu2Py    = mu2Trk->GetPy();
        info.mu2Pz    = mu2Trk->GetPz();
        info.mu2P     = mu2Trk->GetP();
        info.mu2Pt    = mu2Trk->GetPt();
        info.mu2E     = mu2Trk->GetEnergy();
        info.mu2Phi   = std::atan2(info.mu2Py, info.mu2Px);
        info.mu2Theta = (info.mu2P > 1e-9) ? std::acos(std::clamp(info.mu2Pz / info.mu2P, -1.0, 1.0)) : 0.0;
        info.mu2Eta   = calculateEta(info.mu2P, info.mu2Pz);

        if (std::abs(info.mu2Pz) > 1e-9) {
            info.mu2SlopeXZ = info.mu2Px / info.mu2Pz;
            info.mu2SlopeYZ = info.mu2Py / info.mu2Pz;
        }

        // 6. Fill Charmed Hadron Kinematics
        info.charmTrackId = parentCharmId;
        const ShipMCTrack* charmTrk = (parentCharmId >= 0 && parentCharmId < nTracks)
            ? static_cast<ShipMCTrack*>(mcTracks->At(parentCharmId)) : nullptr;

        fillPrimaryCharmKinematics(info, charmTrk, *mcTracks);
        fillMuonCharmKinematics(info, charmTrk, mu2Trk, primaryVtx, *mcTracks);
    } else if (info.hasCharm && !charmIds.empty()) {
        info.charmTrackId = charmIds.front();
        const ShipMCTrack* charmTrk = static_cast<ShipMCTrack*>(mcTracks->At(info.charmTrackId));
        fillPrimaryCharmKinematics(info, charmTrk, *mcTracks);
        fillMuonCharmKinematics(info, charmTrk, nullptr, primaryVtx, *mcTracks);
    }

    // 7. Dimuon System Observables & Candidate Flag
    if (mu1Trk && mu2Trk) {
        fillDimuonKinematics(info, mu1Trk, mu2Trk);

        // Opposite-sign dimuon check:
        // nu_mu CC: mu1 is mu- (13) and charm mu2 is mu+ (-13) -> 13 * -13 < 0
        // anti_nu_mu CC: mu1 is mu+ (-13) and charm mu2 is mu- (13) -> -13 * 13 < 0
        info.isOppositeSignDimuon = (info.mu1Pdg * info.mu2Pdg < 0);

        if (info.isCC && (info.isNuMuCC || info.isAntiNuMuCC) && info.hasPromptCharmMuon) {
            info.hasCandidate = true;
        }
    }

    // 8. MuFilter MCPoints and DS Acceptance for primary muon (mu_1) and charm decay muon (mu_2)
    if (muFilterPoints) {
        const int nPoints = muFilterPoints->GetEntriesFast();

        // 8a. Primary Muon (mu_1) DS Points
        if (info.mu1TrackId >= 0) {
            int nTot = 0, nHor = 0, nVer = 0;
            for (int i = 0; i < nPoints; ++i) {
                auto* pt = static_cast<const FairMCPoint*>(muFilterPoints->At(i));
                if (!pt || pt->GetTrackID() != info.mu1TrackId) continue;
                int detID = pt->GetDetectorID();
                if ((detID / 10000) == 3) {
                    nTot++;
                    int bar = detID % 1000;
                    if (bar < 60) nHor++;
                    else nVer++;
                }
            }
            info.mu1nDSPoints = nTot;
            info.mu1nDSHorizontalPoints = nHor;
            info.mu1nDSVerticalPoints = nVer;
            info.mu1InDS = (nHor >= fConfig.minDSHorizontalPoints && nVer >= fConfig.minDSVerticalPoints);
        }

        // 8b. Charm Decay Muon (mu_2) DS Points
        if (info.mu2TrackId >= 0) {
            int nTot = 0, nHor = 0, nVer = 0;
            for (int i = 0; i < nPoints; ++i) {
                auto* pt = static_cast<const FairMCPoint*>(muFilterPoints->At(i));
                if (!pt || pt->GetTrackID() != info.mu2TrackId) continue;
                int detID = pt->GetDetectorID();
                if ((detID / 10000) == 3) {
                    nTot++;
                    int bar = detID % 1000;
                    if (bar < 60) nHor++;
                    else nVer++;
                }
            }
            info.mu2nDSPoints = nTot;
            info.mu2nDSHorizontalPoints = nHor;
            info.mu2nDSVerticalPoints = nVer;
            info.mu2InDS = (nHor >= fConfig.minDSHorizontalPoints && nVer >= fConfig.minDSVerticalPoints);
        }

        info.dimuonInDSAcceptance = (info.mu1InDS && info.mu2InDS);
    }

    return info;
}

} // namespace snd
