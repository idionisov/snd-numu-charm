#include "NeutrinoTruthProcessor.h"

#include <iostream>
#include <map>
#include <cmath>

namespace snd {

namespace {
    constexpr double kNucleonMass  = 0.938272088;  // GeV/c^2 (average proton/neutron)
    constexpr double kSpeedOfLight = 0.0299792458; // cm / ps

    inline double calculateEta(double p, double pz) {
        if (p - pz > 1e-9 && p + pz > 1e-9) {
            return 0.5 * std::log((p + pz) / (p - pz));
        }
        return (pz > 0.0) ? 999.0 : -999.0;
    }
}

bool NeutrinoTruthProcessor::isCharmedHadron(int pdgCode) {
    int code = std::abs(pdgCode);
    if (code < 100) return false;

    int mod10 = code / 10;
    int q1 = mod10 % 10;
    int q2 = (mod10 / 10) % 10;
    int q3 = (mod10 / 100) % 10;
    int q4 = (mod10 / 1000) % 10;

    return (q1 == 4 || q2 == 4 || q3 == 4 || q4 == 4);
}

bool NeutrinoTruthProcessor::isOpenCharm(int pdgCode) {
    if (!isCharmedHadron(pdgCode)) return false;

    int code = std::abs(pdgCode);
    int mod10 = code / 10;
    int q1 = mod10 % 10;
    int q2 = (mod10 / 10) % 10;
    int q3 = (mod10 / 100) % 10;

    // Charmonium (ccbar mesons: J/psi, eta_c, etc.) has q1==4 && q2==4 with q3==0
    if (q3 == 0 && q1 == 4 && q2 == 4) {
        return false;
    }
    return true;
}

bool NeutrinoTruthProcessor::isCharmedMeson(int pdgCode) {
    if (!isCharmedHadron(pdgCode)) return false;
    int code = std::abs(pdgCode);
    int mod10 = code / 10;
    int q3 = (mod10 / 100) % 10;
    return (q3 == 0);
}

bool NeutrinoTruthProcessor::isCharmedBaryon(int pdgCode) {
    if (!isCharmedHadron(pdgCode)) return false;
    int code = std::abs(pdgCode);
    int mod10 = code / 10;
    int q3 = (mod10 / 100) % 10;
    return (q3 != 0);
}

int NeutrinoTruthProcessor::getCharmQuarkContent(int pdgCode) {
    if (!isOpenCharm(pdgCode)) return 0;
    return (pdgCode > 0) ? +1 : -1;
}

CharmHadronType NeutrinoTruthProcessor::getCharmHadronType(int pdgCode) {
    int code = std::abs(pdgCode);
    switch (code) {
        case 421:  return (pdgCode > 0) ? CharmHadronType::kD0 : CharmHadronType::kAntiD0;
        case 411:  return (pdgCode > 0) ? CharmHadronType::kDPlus : CharmHadronType::kDMinus;
        case 431:  return (pdgCode > 0) ? CharmHadronType::kDsPlus : CharmHadronType::kDsMinus;
        case 4122: return (pdgCode > 0) ? CharmHadronType::kLambdaCPlus : CharmHadronType::kAntiLambdaCMinus;
        default:   break;
    }

    if (isCharmedMeson(pdgCode)) return CharmHadronType::kOtherCharmMeson;
    if (isCharmedBaryon(pdgCode)) return CharmHadronType::kOtherCharmBaryon;
    return CharmHadronType::kNone;
}

std::string NeutrinoTruthProcessor::getParticleName(int pdgCode) {
    switch (pdgCode) {
        case 14:     return "nu_mu";
        case -14:    return "anti_nu_mu";
        case 12:     return "nu_e";
        case -12:    return "anti_nu_e";
        case 16:     return "nu_tau";
        case -16:    return "anti_nu_tau";
        case 11:     return "e-";
        case -11:    return "e+";
        case 13:     return "mu-";
        case -13:    return "mu+";
        case 15:     return "tau-";
        case -15:    return "tau+";
        case 421:    return "D0";
        case -421:   return "anti-D0";
        case 411:    return "D+";
        case -411:   return "D-";
        case 431:    return "Ds+";
        case -431:   return "Ds-";
        case 413:    return "D*+";
        case -413:   return "D*-";
        case 423:    return "D*0";
        case -423:   return "anti-D*0";
        case 433:    return "Ds*+";
        case -433:   return "Ds*-";
        case 4122:   return "Lambda_c+";
        case -4122:  return "anti-Lambda_c-";
        case 4222:   return "Sigma_c++";
        case 4212:   return "Sigma_c+";
        case 4112:   return "Sigma_c0";
        case 4232:   return "Xi_c+";
        case 4132:   return "Xi_c0";
        case 4332:   return "Omega_c0";
        case 443:    return "J/psi";
        default:     break;
    }

    TDatabasePDG* pdgDb = TDatabasePDG::Instance();
    if (pdgDb) {
        TParticlePDG* p = pdgDb->GetParticle(pdgCode);
        if (p) return p->GetName();
    }
    return std::to_string(pdgCode);
}

double NeutrinoTruthProcessor::getNominalMass(int pdgCode) {
    int code = std::abs(pdgCode);
    switch (code) {
        case 11:   return 0.0005109989;
        case 13:   return 0.1056583755;
        case 15:   return 1.77686;
        case 421:  return 1.86484; // D0
        case 411:  return 1.86966; // D+/-
        case 431:  return 1.96835; // Ds+/-
        case 413:  return 2.01026; // D*+/-
        case 423:  return 2.00685; // D*0
        case 433:  return 2.1122;  // Ds*+/-
        case 4122: return 2.28646; // Lambda_c+
        case 4222: return 2.45397; // Sigma_c++
        case 4212: return 2.4529;  // Sigma_c+
        case 4112: return 2.45375; // Sigma_c0
        case 4232: return 2.46771; // Xi_c+
        case 4132: return 2.47044; // Xi_c0
        case 4332: return 2.6952;  // Omega_c0
        case 443:  return 3.0969;  // J/psi
        default:   break;
    }

    TDatabasePDG* pdgDb = TDatabasePDG::Instance();
    if (pdgDb) {
        TParticlePDG* p = pdgDb->GetParticle(pdgCode);
        if (p) return p->Mass();
    }
    return 1.865;
}

double NeutrinoTruthProcessor::compute3DImpactParameter(
    const TVector3& primaryVtx,
    const TVector3& decayVtx,
    const TVector3& pTrack
) {
    TVector3 deltaR = decayVtx - primaryVtx;
    double p = pTrack.Mag();
    if (p <= 1e-9) return deltaR.Mag();
    return deltaR.Cross(pTrack).Mag() / p;
}

double NeutrinoTruthProcessor::computeTransverseImpactParameter(
    double vtxX, double vtxY,
    double decayX, double decayY,
    double px, double py
) {
    double dx = decayX - vtxX;
    double dy = decayY - vtxY;
    double pt = std::hypot(px, py);
    if (pt <= 1e-9) return std::hypot(dx, dy);
    return std::abs(dx * py - dy * px) / pt;
}

double NeutrinoTruthProcessor::computePtRel(
    const TVector3& pTrack,
    const TVector3& pRefAxis
) {
    double pRef = pRefAxis.Mag();
    if (pRef <= 1e-9) return 0.0;
    return pTrack.Cross(pRefAxis).Mag() / pRef;
}

bool NeutrinoTruthProcessor::isFiducial(double x, double y, double z) const {
    if (z < fConfig.targetZMin || z > fConfig.targetZMax) return false;

    const double xMin = fConfig.fidXMin + fConfig.fiducialMargin;
    const double xMax = fConfig.fidXMax - fConfig.fiducialMargin;
    const double yMin = fConfig.fidYMin + fConfig.fiducialMargin;
    const double yMax = fConfig.fidYMax - fConfig.fiducialMargin;

    return (x >= xMin && x <= xMax && y >= yMin && y <= yMax);
}

RegionType NeutrinoTruthProcessor::determineRegion(double z, double x, double y) const {
    if (z < fConfig.targetZMin) return RegionType::kRegionRock;
    if (z > fConfig.targetZMax) return RegionType::kRegionMuonFilter;

    if (x >= fConfig.fidXMin && x <= fConfig.fidXMax &&
        y >= fConfig.fidYMin && y <= fConfig.fidYMax) {
        return RegionType::kRegionTarget;
    }
    return RegionType::kRegionOutside;
}

int NeutrinoTruthProcessor::findPrimaryLepton(const TClonesArray& mcTracks, int nuPdg) const {
    const int nTracks = mcTracks.GetEntriesFast();
    if (nTracks < 2) return -1;

    // Expected charged lepton PDG for CC:
    // nu_e (12) -> e- (11), anti_nu_e (-12) -> e+ (-11)
    // nu_mu (14) -> mu- (13), anti_nu_mu (-14) -> mu+ (-13)
    // nu_tau (16) -> tau- (15), anti_nu_tau (-16) -> tau+ (-15)
    int expectedLeptonPdg = 0;
    if (std::abs(nuPdg) == 12) expectedLeptonPdg = (nuPdg > 0) ? 11 : -11;
    else if (std::abs(nuPdg) == 14) expectedLeptonPdg = (nuPdg > 0) ? 13 : -13;
    else if (std::abs(nuPdg) == 16) expectedLeptonPdg = (nuPdg > 0) ? 15 : -15;

    // First check track 1 directly (standard GENIE / sndsw primary outgoing lepton convention)
    auto* trk1 = static_cast<ShipMCTrack*>(mcTracks.At(1));
    if (trk1 && trk1->GetMotherId() == 0) {
        if (expectedLeptonPdg != 0 && trk1->GetPdgCode() == expectedLeptonPdg) {
            if (trk1->GetP() >= fConfig.minLeptonMomentum) return 1;
        }
    }

    // Secondary scan across primary particles (motherId == 0)
    int bestId = -1;
    double maxP = -1.0;
    for (int i = 1; i < nTracks; ++i) {
        auto* trk = static_cast<ShipMCTrack*>(mcTracks.At(i));
        if (!trk || trk->GetMotherId() != 0) continue;

        if (expectedLeptonPdg != 0 && trk->GetPdgCode() == expectedLeptonPdg) {
            if (trk->GetP() > maxP && trk->GetP() >= fConfig.minLeptonMomentum) {
                maxP = trk->GetP();
                bestId = i;
            }
        }
    }

    return bestId;
}

std::vector<int> NeutrinoTruthProcessor::findPrimaryCharmedHadrons(const TClonesArray& mcTracks) const {
    std::vector<int> charmTrackIds;
    const int nTracks = mcTracks.GetEntriesFast();

    for (int i = 1; i < nTracks; ++i) {
        auto* trk = static_cast<ShipMCTrack*>(mcTracks.At(i));
        if (!trk) continue;

        if (trk->GetMotherId() == 0 && isCharmedHadron(trk->GetPdgCode())) {
            charmTrackIds.push_back(i);
        }
    }

    return charmTrackIds;
}

void NeutrinoTruthProcessor::fillNeutrinoKinematics(
    NeutrinoTruthInfo& info,
    const ShipMCTrack* nuTrk,
    const ShipMCTrack* leptonTrk,
    const TClonesArray& mcTracks
) const {
    if (!nuTrk) return;

    info.nuTrackId = 0;
    info.nuPdg = nuTrk->GetPdgCode();
    info.nuPx = nuTrk->GetPx();
    info.nuPy = nuTrk->GetPy();
    info.nuPz = nuTrk->GetPz();
    info.nuP  = nuTrk->GetP();
    info.nuE  = nuTrk->GetEnergy();
    info.nuPt = nuTrk->GetPt();
    info.nuPhi = std::atan2(info.nuPy, info.nuPx);
    info.nuTheta = (info.nuP > 1e-9) ? std::acos(std::clamp(info.nuPz / info.nuP, -1.0, 1.0)) : 0.0;
    info.nuEta = calculateEta(info.nuP, info.nuPz);

    info.vtxX = nuTrk->GetStartX();
    info.vtxY = nuTrk->GetStartY();
    info.vtxZ = nuTrk->GetStartZ();
    info.vtxT = nuTrk->GetStartT();

    info.isFiducial = isFiducial(info.vtxX, info.vtxY, info.vtxZ);
    double rawW = nuTrk->GetWeight();
    info.rawWeight = rawW;
    info.mcWeight  = (rawW > 0.0 ? rawW : 1.0) * fConfig.weightScale;

    // Deep Inelastic Scattering (DIS) variables
    if (leptonTrk && info.nuE > 1e-6) {
        double eLep = leptonTrk->GetEnergy();
        double pxLep = leptonTrk->GetPx();
        double pyLep = leptonTrk->GetPy();
        double pzLep = leptonTrk->GetPz();

        TLorentzVector p4Nu(info.nuPx, info.nuPy, info.nuPz, info.nuE);
        TLorentzVector p4Lep(pxLep, pyLep, pzLep, eLep);
        TLorentzVector q = p4Nu - p4Lep;

        info.Q2 = -q.M2(); // Space-like Q^2 = -(q^2)
        if (info.Q2 < 0.0) info.Q2 = 0.0;

        double nuTransfer = info.nuE - eLep;
        info.InelasticityY = nuTransfer / info.nuE;

        double denomX = 2.0 * kNucleonMass * nuTransfer;
        info.BjorkenX = (denomX > 1e-6) ? (info.Q2 / denomX) : 0.0;

        double w2 = kNucleonMass * kNucleonMass + 2.0 * kNucleonMass * nuTransfer - info.Q2;
        info.HadronicW = (w2 > 0.0) ? std::sqrt(w2) : 0.0;
    }

    // Hadronic recoil system computation
    const int nTracks = mcTracks.GetEntriesFast();
    double hadPx = 0.0;
    double hadPy = 0.0;
    double hadE = 0.0;
    int nPrimHadrons = 0;

    for (int i = 1; i < nTracks; ++i) {
        auto* trk = static_cast<ShipMCTrack*>(mcTracks.At(i));
        if (!trk || trk->GetMotherId() != 0) continue;

        int pdg = std::abs(trk->GetPdgCode());
        // Exclude outgoing primary leptons
        if (pdg >= 11 && pdg <= 16) continue;

        hadPx += trk->GetPx();
        hadPy += trk->GetPy();
        hadE  += trk->GetEnergy();
        nPrimHadrons++;
    }

    info.nPrimaryHadrons = nPrimHadrons;
    info.hadronicEnergyTotal = hadE;
    info.hadronicRecoilPt = std::hypot(hadPx, hadPy);

    if (leptonTrk) {
        double totPx = leptonTrk->GetPx() + hadPx;
        double totPy = leptonTrk->GetPy() + hadPy;
        info.missingPt = std::hypot(info.nuPx - totPx, info.nuPy - totPy);
    }
}

void NeutrinoTruthProcessor::fillPrimaryCharmKinematics(
    NeutrinoTruthInfo& info,
    const ShipMCTrack* charmTrk,
    const TClonesArray& /*mcTracks*/
) const {
    if (!charmTrk) return;

    info.charmPdg = charmTrk->GetPdgCode();
    info.charmQuarkContent = getCharmQuarkContent(info.charmPdg);
    info.charmName = getParticleName(info.charmPdg);
    info.charmType = getCharmHadronType(info.charmPdg);
    info.charmMass = getNominalMass(info.charmPdg);

    info.charmPx = charmTrk->GetPx();
    info.charmPy = charmTrk->GetPy();
    info.charmPz = charmTrk->GetPz();
    info.charmP  = charmTrk->GetP();
    info.charmPt = charmTrk->GetPt();
    info.charmE  = charmTrk->GetEnergy();
    info.charmPhi = std::atan2(info.charmPy, info.charmPx);
    info.charmTheta = (info.charmP > 1e-9) ? std::acos(std::clamp(info.charmPz / info.charmP, -1.0, 1.0)) : 0.0;
    info.charmEta = calculateEta(info.charmP, info.charmPz);

    if (info.hadronicEnergyTotal > 1e-6) {
        info.charmEnergyFractionZ = info.charmE / info.hadronicEnergyTotal;
    } else if (info.nuE > 1e-6) {
        info.charmEnergyFractionZ = info.charmE / info.nuE;
    }
}

NeutrinoTruthInfo NeutrinoTruthProcessor::process(const TClonesArray* mcTracks) const {
    NeutrinoTruthInfo info;
    if (!mcTracks || mcTracks->GetEntriesFast() == 0) {
        return info;
    }

    const int nTracks = mcTracks->GetEntriesFast();

    // 1. Primary neutrino track (Track 0)
    auto* nuTrk = static_cast<ShipMCTrack*>(mcTracks->At(0));
    if (!nuTrk) return info;

    int nuPdg = nuTrk->GetPdgCode();

    int nPrimaries = 0;
    for (int i = 0; i < nTracks; ++i) {
        auto* trk = static_cast<ShipMCTrack*>(mcTracks->At(i));
        if (!trk) continue;
        if (trk->GetMotherId() == 0) nPrimaries++;
    }
    info.nPrimaryTracks = nPrimaries;

    // 2. Identify Primary Outgoing Lepton
    int lepId = findPrimaryLepton(*mcTracks, nuPdg);
    const ShipMCTrack* lepTrk = (lepId >= 0 && lepId < nTracks) ? static_cast<ShipMCTrack*>(mcTracks->At(lepId)) : nullptr;

    if (lepTrk) {
        info.isCC = true;
        info.isNC = false;
        info.primaryLeptonTrackId = lepId;
        info.primaryLeptonPdg     = lepTrk->GetPdgCode();
        info.primaryLeptonCharge  = (info.primaryLeptonPdg > 0) ? -1 : +1; // e-(11), mu-(13), tau-(15)
        info.primaryLeptonPx      = lepTrk->GetPx();
        info.primaryLeptonPy      = lepTrk->GetPy();
        info.primaryLeptonPz      = lepTrk->GetPz();
        info.primaryLeptonP       = lepTrk->GetP();
        info.primaryLeptonPt      = lepTrk->GetPt();
        info.primaryLeptonE       = lepTrk->GetEnergy();
        info.primaryLeptonPhi     = std::atan2(info.primaryLeptonPy, info.primaryLeptonPx);
        info.primaryLeptonTheta   = (info.primaryLeptonP > 1e-9) ? std::acos(std::clamp(info.primaryLeptonPz / info.primaryLeptonP, -1.0, 1.0)) : 0.0;
        info.primaryLeptonEta     = calculateEta(info.primaryLeptonP, info.primaryLeptonPz);

        if (std::abs(info.primaryLeptonPz) > 1e-9) {
            info.primaryLeptonSlopeXZ = info.primaryLeptonPx / info.primaryLeptonPz;
            info.primaryLeptonSlopeYZ = info.primaryLeptonPy / info.primaryLeptonPz;
        }

        if (nuPdg == 14) {
            info.interactionType = InteractionType::kNuMuCC;
            info.interactionName = "nu_mu_CC";
        } else if (nuPdg == -14) {
            info.interactionType = InteractionType::kAntiNuMuCC;
            info.interactionName = "anti_nu_mu_CC";
        } else if (nuPdg == 12) {
            info.interactionType = InteractionType::kNuECC;
            info.interactionName = "nu_e_CC";
        } else if (nuPdg == -12) {
            info.interactionType = InteractionType::kAntiNuECC;
            info.interactionName = "anti_nu_e_CC";
        } else if (nuPdg == 16) {
            info.interactionType = InteractionType::kNuTauCC;
            info.interactionName = "nu_tau_CC";
        } else if (nuPdg == -16) {
            info.interactionType = InteractionType::kAntiNuTauCC;
            info.interactionName = "anti_nu_tau_CC";
        }
    } else {
        info.isCC = false;
        info.isNC = true;
        info.interactionType = InteractionType::kNC;
        info.interactionName = (nuPdg > 0) ? "nu_NC" : "anti_nu_NC";
    }

    // 3. Fill Neutrino & DIS Kinematics
    fillNeutrinoKinematics(info, nuTrk, lepTrk, *mcTracks);

    // 4. Identify Primary Charmed Hadrons
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
