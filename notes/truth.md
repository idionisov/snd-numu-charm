# Signal Definition & Monte Carlo Truth Processing

This document explains how a signal event for **$\nu_\mu$-induced charm production with dimuon decay** is defined and identified in the codebase, detailing the roles of `ShipMCTrack`, `ScifiPoint`, `MuFilterPoint`, PDG codes, Geant4 process IDs, and target material/geometry.

---

## 1. Physics Signal Topology

A golden signal event corresponds to a muon neutrino (or anti-neutrino) deep-inelastic charged-current (CC) interaction in the tungsten target producing an open charmed hadron ($D^0, D^+, D_s^+, \Lambda_c^+, \dots$), where the charmed hadron subsequently decays semi-muonically:

$$\nu_\mu + \mathcal{N}(\text{W}) \longrightarrow \mu_1^- + C + X, \quad C \longrightarrow \mu_2^+ + \nu_\mu + Y$$
$$\bar{\nu}_\mu + \mathcal{N}(\text{W}) \longrightarrow \mu_1^+ + \bar{C} + X, \quad \bar{C} \longrightarrow \mu_2^- + \bar{\nu}_\mu + Y$$

Key physical signatures:
1. **Primary CC Muon ($\mu_1$)**: Outgoing charged lepton at the primary neutrino interaction vertex.
2. **Open Charmed Hadron ($C$)**: Produced directly at the primary interaction vertex.
3. **Secondary Muon ($\mu_2$)**: Produced at a displaced decay vertex ($c\tau \sim 60\text{--}300\ \mu\text{m}$) from prompt semi-leptonic charm decay.
4. **Opposite-Sign Dimuon (OS $\mu\mu$)**: $\mu_1^- \mu_2^+$ (for $\nu_\mu$) or $\mu_1^+ \mu_2^-$ (for $\bar{\nu}_\mu$).

---

## 2. How Signal Events are Defined in the Current Code

The signal identification logic is implemented in [`MuonNeutrinoTruthProcessor`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/analysis/MuonNeutrinoTruthProcessor.h) (inheriting from [`NeutrinoTruthProcessor`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/analysis/NeutrinoTruthProcessor.h)).

The processor operates on the `MCTrack` branch (`TClonesArray` of `ShipMCTrack`) from the simulation tree `cbmsim`.

### Step-by-Step Selection Logic

| Step | Observable | Requirement in Code | Method in Code |
| :--- | :--- | :--- | :--- |
| **1. Neutrino** | Incoming $\nu_\mu$ / $\bar{\nu}_\mu$ | `MCTrack[0]` has `abs(nuPdg) == 14` | [`MuonNeutrinoTruthProcessor::processMuonNeutrino`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/analysis/MuonNeutrinoTruthProcessor.cxx#L245) |
| **2. Primary Muon ($\mu_1$)** | Primary outgoing CC lepton | `motherId == 0`, `pdg == 13` ($\mu^-$ for $\nu_\mu$) or `-13` ($\mu^+$ for $\bar{\nu}_\mu$), $p \ge p_{\min}$ | [`MuonNeutrinoTruthProcessor::findPrimaryMuon`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/analysis/MuonNeutrinoTruthProcessor.cxx#L26) |
| **3. Charmed Hadron ($C$)** | Open charm at primary vertex | `motherId == 0`, quark content contains charm quark ($q=4$), excluding charmonium ($c\bar{c}$) | [`NeutrinoTruthProcessor::findPrimaryCharmedHadrons`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/analysis/NeutrinoTruthProcessor.cxx#L256) |
| **4. Charm Decay Muon ($\mu_2$)** | Prompt decay $\mu$ | `abs(pdg) == 13`, parent ancestry traces back to charmed hadron ($C$), flight distance $\le 20\text{ cm}$ | [`MuonNeutrinoTruthProcessor::findCharmDecayMuon`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/analysis/MuonNeutrinoTruthProcessor.cxx#L78) |
| **5. Dimuon System** | Opposite-sign charge | `mu1Pdg * mu2Pdg < 0` (`isOppositeSignDimuon == true`) | [`MuonNeutrinoTruthProcessor::processMuonNeutrino`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/analysis/MuonNeutrinoTruthProcessor.cxx#L386) |
| **6. Golden Candidate Flag** | Complete signal | `hasCandidate = isCC && (isNuMuCC || isAntiNuMuCC) && hasPromptCharmMuon` | [`MuonNeutrinoTruthProcessor::processMuonNeutrino`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/analysis/MuonNeutrinoTruthProcessor.cxx#L389) |

---

## 3. Specific Questions Answered

### Q1: Do we identify the signal by Process ID or by PDG code?
**We identify it primarily by PDG codes and particle genealogy (`motherId`), NOT by Geant4 process ID.**
- **Why not Geant4 Process ID?**
  In GENIE + Geant4 Monte Carlo simulations:
  - The primary neutrino-nucleus interaction (the hard scattering, nuclear remnant breakup, and hadronization) is simulated by **GENIE**, not Geant4. All primary particles emerge directly from GENIE with `motherId == 0`.
  - Geant4 is only invoked downstream for particle transport, secondary nuclear interactions, and decays in flight.
  - Process IDs (`trk->GetProcID()`, e.g. `kPDecay`) vary across Geant4 physics lists and generator interfaces. In contrast, standard PDG codes and parent-daughter tree links (`motherId`) are deterministic and generator-independent.
- **How PDG codes identify Charm:**
  [`isCharmedHadron(pdgCode)`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/analysis/NeutrinoTruthProcessor.cxx#L21) decodes the quark content according to the PDG numbering scheme:
  - Mesons ($q_1 q_2$): $D^0 (421)$, $D^+ (411)$, $D_s^+ (431)$, $D^{*+} (413)$, etc.
  - Baryons ($q_1 q_2 q_3$): $\Lambda_c^+ (4122)$, $\Sigma_c (4222, 4212, 4112)$, $\Xi_c (4232, 4132)$, $\Omega_c^0 (4332)$, etc.
  - Hidden charm ($c\bar{c}$ charmonium like $J/\psi\ [443]$) is explicitly excluded via [`isOpenCharm()`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/analysis/NeutrinoTruthProcessor.cxx#L34).

---

### Q2: How is the Target / Tungsten interaction identified?
In the current code:
- **Current implementation:**
  Target selection is defined purely by **geometric fiducial coordinates** in [`isFiducial(x, y, z)`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/analysis/NeutrinoTruthProcessor.cxx#L195):
  - $260.0\text{ cm} \le Z_{\text{vtx}} \le 360.0\text{ cm}$ (the 5 Target walls / SciFi region)
  - $-46.0\text{ cm} \le X_{\text{vtx}} \le -10.0\text{ cm}$
  - $+17.0\text{ cm} \le Y_{\text{vtx}} \le +53.0\text{ cm}$
  - Events originating in the upstream rock ($Z < 260\text{ cm}$) or downstream Muon Filter ($Z > 360\text{ cm}$) are classified as `kRegionRock` and `kRegionMuonFilter`.
- **Target Material Nuance (Tungsten vs Emulsion/SciFi):**
  - In the simulated dataset `...volTarget...`, vertices are generated within the volume `volTarget`.
  - The actual SND@LHC target bricks consist of **tungsten plates** (`volPassive`, material `tungstensifon`), interleaved with nuclear emulsion films, encased in walls.
  - If one requires the vertex to be strictly *inside tungsten* (as opposed to in the emulsion layers or acrylic supports), the TGeoManager can be queried at truth level:
    ```cpp
    TGeoNode* node = gGeoManager->FindNode(vtxX, vtxY, vtxZ);
    bool inTungsten = (node && std::string(node->GetVolume()->GetName()) == "volPassive");
    // or checking the material name:
    // node->GetVolume()->GetMaterial()->GetName() == "tungstensifon"
    ```

---

### Q3: What is the role of `ScifiPoint` and `MuFilterPoint`?
- **Do we use `ScifiPoint` or `MuFilterPoint` to define the truth signal?**
  **No.** `ScifiPoint` and `MuFilterPoint` are **detector simulation hits** (Geant4 energy depositions in active fibers and scintillator bars), not truth genealogy objects.
- **Where are MCPoints used?**
  - **Detector Acceptance & Tracking Efficiency:** You can inspect `ScifiPoint` and `MuFilterPoint` to determine whether $\mu_1$ and $\mu_2$ actually traversed the active detector (e.g., checking if `trk1` and `trk2` produced hits in $\ge 3$ SciFi stations or penetrated into the Downstream Muon Filter).
  - **Digitization cross-checks:** Associating digitized hits back to truth tracks via `Digi_ScifiHits2MCPoints` and `Digi_MuFilterHits2MCPoints`.
  - `ShipMCTrack` already caches hit counts via `trk->GetNPoints(kScifi)` and `trk->GetNPoints(kMuFilter)`.

---

## 4. Summary of Signal Flags in `truth` Info Struct

When running with RDataFrame:
```python
df = df.Define("truth", "snd::MuonNeutrinoTruthProcessor()", ["MCTrack"])
```
The resulting `MuonNeutrinoTruthInfo` provides:
- `truth.hasCandidate`: `true` if $\nu_\mu$ CC + primary muon + prompt charm decay muon.
- `truth.isOppositeSignDimuon`: `true` if $\mu_1$ and $\mu_2$ have opposite electric charges.
- `truth.isFiducial`: `true` if interaction vertex is within Target fiducial coordinates.
- `truth.charmPdg` & `truth.charmType`: Specific charmed hadron produced ($D^0, D^+, D_s^+, \Lambda_c^+$).
- `truth.dimuonInvMass`, `truth.dimuonOpeningAngle`: Kinematic observables of the two muons.
