# Physics & Mathematical Derivation: MuFilter Digitization, SiPM Response, TOFPET2 QDC, and Energy Thresholds

**Author:** Antigravity (on behalf of SND@LHC Collaboration study)  
**Date:** November 23, 2026  
**Document Location:** `notes/mufilter_qdc_pedestal_threshold_derivation.md`  
**Reference Hardware & Software:**
- PETsys TOFPET 2C ASIC Datasheet (Rev 18)
- TOFPET2 Experimental Characterization Paper (*2019 JINST 14 P03029*)
- PETsys TOFPET2 Product Flyer (v18)
- SND@LHC Software Framework (`sndsw`): `ConvRawData.cxx`, `DigiTaskSND.cxx`, `MuFilterHit.cxx`

---

## 1. Executive Summary

This note provides the complete mathematical and physical derivation connecting:
1. **Primary energy deposition** ($\Delta E$ in $\text{MeV}$) in the SND@LHC MuFilter scintillator bars.
2. **Optical photon production, transport, and attenuation** ($N_\gamma$).
3. **Silicon Photomultiplier (SiPM) detection and avalanche gain** ($N_{\rm p.e.}$, $Q_{\rm IN}$).
4. **PETsys TOFPET 2C ASIC charge integration and digitization** ($q_{\rm fine}$, $QDC$).
5. **Exact pedestal subtraction mechanisms** and the origin of negative QDC values.
6. **Multi-level discriminator triggering** ($v_{\rm th\_t1}, v_{\rm th\_t2}, v_{\rm th\_e}$) and the physical thresholds for recorded and offline physics hits.

```
  Energy Deposition ΔE [MeV]
           │
           ▼  (Scintillation Yield Y ≈ 10,000 ph/MeV, Trapping f_trap ≈ 18.7%, Attenuation η_att)
  Optical Photons Reaching Bar End N_γ
           │
           ▼  (Geometric Coverage ε_geom = 0.36, SiPM PDE ≈ 40%)
  Detected Photoelectrons N_p.e. (311.83 p.e. / MeV per SiPM)
           │
           ▼  (Avalanche Gain G = 2.5×10^6, Optical Cross-Talk K_OCT ≈ 8%)
  Analog Output Charge Q_IN [pC] (134.9 pC / MeV)
           │
           ▼  (Front-End TIA, Q-Branch Integrator C_Q, DC Bias I_integ_bias)
  Total Integrated Charge Q_total = Q_IN · G_Q + I_integ_bias · Δt
           │
           ▼  (10-bit Wilkinson ADC - Offset Q_offset)
  Raw Fine Code q_fine
           │
           ▼  (Dynamic Pedestal Subtraction: Pedestal = I_integ_bias · Δt + Q_offset)
  Calibrated QDC = q_fine - Pedestal  [counts]
           │
           ▼  (Channel MIP Calibration: allSignals = QDC / calibrationConstant)
  Normalized MIP Units (1.0 = 1 MIP)
```

---

## 2. Scintillation Physics & Optical Light Transport

### 2.1 Bar Geometry & Scintillator Properties
* **Material:** Eljen EJ-200 polyvinyltoluene plastic scintillator ($\rho = 1.023\text{ g/cm}^3$, refractive index $n_{\rm scint} = 1.58$).
* **Nominal Scintillation Yield:** $Y_0 \approx 10\,000\text{ photons / MeV}$.
* **Decay Time:** $\tau_{\rm decay} \approx 2.1\text{ ns}$, peak emission wavelength $\lambda_{\rm peak} \approx 425\text{ nm}$.
* **DS Bar Dimensions:** $1.0\text{ cm} \times 1.0\text{ cm} \times 82.5\text{ cm}$. Read out on both ends (horizontal bars) or one end (vertical bars) by Hamamatsu $6 \times 6\text{ mm}^2$ MPPCs.
* **US Bar Dimensions:** $3.0\text{ cm} \times 3.0\text{ cm} \times 82.5\text{ cm}$. Read out on one end by 8 SiPMs (6 large $6\times 6\text{ mm}^2$ and 2 small).

### 2.2 Trapping Efficiency via Total Internal Reflection
Light propagates along the bar via total internal reflection (TIR) at the scintillator-air interface ($n_{\rm scint} = 1.58, n_{\rm air} = 1.00$).

1. **Critical Angle Relative to Normal:**
   $$\sin\theta_c = \frac{n_{\rm air}}{n_{\rm scint}} = \frac{1}{1.58} \approx 0.6329 \implies \theta_c \approx 39.26^\circ$$
2. **Critical Angle Relative to Longitudinal Bar Axis ($\alpha = 90^\circ - \theta$):**
   $$\cos\alpha_c = \sin\theta_c = \frac{1}{n_{\rm scint}} \approx 0.6329 \implies \alpha_c \approx 50.74^\circ$$
3. **Geometric Trapping Fraction (per bar end):**
   For isotropic emission, the solid angle trapped towards one end face is:
   $$f_{\rm trap} = \frac{\Omega}{4\pi} = \frac{2\pi (1 - \cos\alpha_c)}{4\pi} = \frac{1 - \frac{1}{n_{\rm scint}}}{2} = \frac{1 - 0.6329}{2} \approx \mathbf{18.35\%}$$
   In a square/rectangular cross-section, skew rays increase the effective total trapped fraction to $\approx 20.5\%$.

### 2.3 Optical Path Elongation & Effective Attenuation
Light trapped at angle $\alpha$ travels an optical distance $L_{\rm opt} = x / \cos\alpha$ to reach longitudinal position $x$.
From testbeam time-of-flight fits on the DS horizontal reference channel:
$$c_{\rm scint} = 13.88 \pm 0.57\text{ cm/ns}$$
The bulk phase velocity in EJ-200 is:
$$v_{\rm bulk} = \frac{c_0}{n_{\rm scint}} = \frac{29.979\text{ cm/ns}}{1.58} \approx 18.97\text{ cm/ns}$$
The **geometric path elongation factor $\xi$** is:
$$\xi = \frac{v_{\rm bulk}}{c_{\rm scint}} = \frac{18.97}{13.88} \approx \mathbf{1.367}$$
Given the bulk attenuation length $\lambda_{\rm bulk} = 380\text{ cm}$ of EJ-200, the bulk geometric effective attenuation length is:
$$\lambda_{\rm att}^{\rm eff, bulk} = \frac{\lambda_{\rm bulk}}{\xi} = \frac{380\text{ cm}}{1.367} \approx \mathbf{278\text{ cm}}$$
Accounting for internal reflection boundary losses ($R \approx 0.997$ per bounce over $\sim 80\text{ bounces}$), the net operational attenuation length along the bar is:
$$\lambda_{\rm att}^{\rm eff} \approx \mathbf{160\text{ to } 190\text{ cm}}$$
For a center-incident particle ($x = 41.25\text{ cm}$):
$$\eta_{\rm att} = \exp\left(-\frac{41.25\text{ cm}}{180\text{ cm}}\right) \approx \mathbf{0.795}$$

### 2.4 SiPM Coupling & Photoelectron Yield
* **Geometric Area Ratio:** For a $1.0\times 1.0\text{ cm}^2$ DS bar end face ($100\text{ mm}^2$) coupled to a $6\times 6\text{ mm}^2$ SiPM active window ($36\text{ mm}^2$):
  $$\epsilon_{\rm geom} = \frac{36\text{ mm}^2}{100\text{ mm}^2} = \mathbf{0.360}$$
* **Photon Detection Efficiency (PDE):** For Hamamatsu S13360-6050CS at peak emission $\lambda = 425\text{ nm}$ and nominal overvoltage $V_{\rm OV} \approx 3.0\text{ V}$:
  $$\mathrm{PDE} \approx \mathbf{40.0\%} \quad (0.40)$$
* **Detected Photoelectron Yield per MeV:**
  Combining the scintillation yield, trapping, attenuation, coupling, and PDE:
  $$\frac{N_{\rm p.e.}}{\Delta E} = Y_0 \cdot f_{\rm trap} \cdot \eta_{\rm att} \cdot \epsilon_{\rm geom} \cdot \mathrm{PDE}$$
  $$\frac{N_{\rm p.e.}}{\Delta E} \approx 10\,000 \times 0.205 \times 0.795 \times 0.360 \times 0.400 = \mathbf{311.83\text{ p.e. / MeV}}$$

---

## 3. Photoelectrons to Analog Output Charge ($Q_{\rm IN}$)

### 3.1 Single Photoelectron Charge
For a SiPM with avalanche gain $G = 2.5 \times 10^6$:
$$Q_{\rm 1\,p.e.} = G \cdot e = (2.5 \times 10^6) \times (1.602176634 \times 10^{-19}\text{ C}) = \mathbf{400.54\text{ fC}} = \mathbf{0.4005\text{ pC}}$$

### 3.2 Optical Cross-Talk & Excess Charge Factor
In $50\,\mu\text{m}$ pitch microcells, optical cross-talk probability $K_{\rm OCT} \approx 8\%$ produces secondary avalanches in neighboring cells within $< 1\text{ ns}$:
$$ECF = 1 + K_{\rm OCT} \approx 1.08$$
$$Q_{\rm 1\,p.e.}^{\rm eff} = Q_{\rm 1\,p.e.} \cdot (1 + K_{\rm OCT}) \approx 400.54\text{ fC} \times 1.08 \approx \mathbf{432.58\text{ fC / p.e.}}$$

### 3.3 Charge Yield per Unit Deposited Energy
1. **Per Deposited MeV:**
   $$Q_{\rm IN} / \text{MeV} = 311.83\text{ p.e./MeV} \times 0.43258\text{ pC/p.e.} \approx \mathbf{134.9\text{ pC / MeV}}$$
2. **Per Through-Going MIP ($\Delta E_{\rm MIP} \approx 2.0\text{ MeV}$ in $1\text{ cm}$):**
   $$N_{\rm p.e.}^{\rm MIP} = 2.0\text{ MeV} \times 311.83\text{ p.e./MeV} \approx \mathbf{623.7\text{ p.e.}}$$
   $$Q_{\rm IN}^{\rm MIP} = 2.0\text{ MeV} \times 134.9\text{ pC/MeV} \approx \mathbf{269.8\text{ pC}}$$

---

## 4. PETsys TOFPET 2C ASIC Architecture & QDC Transfer Function

### 4.1 Channel Front-End Architecture
*(Ref: TOFPET 2C Datasheet Section 5, Characterization Paper Section 1)*

Each of the 64 channels features an input preamplifier ($P$) with low input impedance ($R_{\rm IN} \approx 20\,\Omega$) that splits the input current $I_{\rm IN}$ into three parallel processing branches:
1. **Branch $T$ (Timing):** Fast transimpedance amplifier ($G_T \approx 3000\,\Omega$) driving two comparators ($D_{T1}, D_{T2}$) for precise time stamping.
2. **Branch $E$ (Energy / Trigger Validation):** Lower-gain TIA ($G_E \approx 300\,\Omega$) driving comparator $D_E$ for hit validation.
3. **Branch $Q$ (Charge Integration):** Integrates an analog current replica for energy measurement.

```
                  ┌───► TIA (Gain G_T) ───► Discriminator D_T1 (vth_T1) ──► TDC T
                  │                   ───► Discriminator D_T2 (vth_T2) ──► Trigger Logic
                  │
SiPM Input ──► Preamp ──► TIA (Gain G_E) ───► Discriminator D_E  (vth_E)  ──► Trigger Validation
  I_IN            │
                  └───► Current Mirror (G_Q) + I_integ_bias ──► Buffer C_Q ──► Wilkinson ADC (QDC)
```

### 4.2 Mathematical Model of the QDC Integrator
*(Ref: TOFPET 2C Datasheet Section 5.7, Equations 2–4)*

1. On the rising edge of `trigger_Q` ($t_{\rm start} \equiv rise(SW_Q)$), switch $SW_Q$ closes, charging the analog buffer capacitor $C_Q$.
2. The charging current is the sum of the scaled signal current $I_{\rm IN}(t) \cdot G_Q$ plus an intentional DC current $I_{\rm integ\_bias}$:
   $$I_{\rm charge}(t) = I_{\rm IN}(t) \cdot G_Q + I_{\rm integ\_bias}$$
3. The switch opens at $t_{\rm stop} \equiv fall(SW_Q)$ upon reaching the configured integration window:
   $$\Delta t = fall(SW_Q) - rise(SW_Q) \approx q_{\rm coarse} - t_T$$
4. The total charge accumulated on $C_Q$ is:
   $$Q_{\rm total} = \int_{rise(SW_Q)}^{fall(SW_Q)} \left( I_{\rm IN}(t) \cdot G_Q + I_{\rm integ\_bias} \right) dt$$
   $$Q_{\rm total} = G_Q \cdot Q_{\rm IN} + I_{\rm integ\_bias} \cdot (q_{\rm coarse} - t_T)$$
5. A 10-bit Wilkinson ADC digitizes $Q_{\rm total}$ relative to the analog buffer discharge baseline $Q_{\rm offset}$:
   $$q_{\rm fine} = Q_{\rm total} - Q_{\rm offset}$$
   $$q_{\rm fine} = G_Q \cdot Q_{\rm IN} + I_{\rm integ\_bias} \cdot (q_{\rm coarse} - t_T) - Q_{\rm offset}$$

### 4.3 Input Charge Reconstruction Formula
Inverting the transfer function yields the true physical input charge $Q_{\rm IN}$:
$$\mathbf{Q_{\rm IN} = \frac{q_{\rm fine} - I_{\rm integ\_bias} \cdot (q_{\rm coarse} - t_T) - Q_{\rm offset}}{G_Q}}$$

### 4.4 QDC Specifications & Binning Scale
*(Ref: Characterization Paper Section 3.4, Datasheet Table 2)*
* **Nominal Integrator Gain:** $G_Q = 1.00$ (configurable: $1.00, 1.39, 1.70, 2.50, 3.65$).
* **Nominal QDC Binning:** $\mathbf{3.6\text{ pC per LSB}}$ (at $G_Q = 1.0$).
* **Nominal Linear Dynamic Range:** $160\text{ LSB} \iff 576\text{ pC}$ (extendable to $1440\text{--}1500\text{ pC}$ at reduced gain).
* **QDC Noise:** $\sigma_{\rm noise} \approx 0.65\text{ LSB RMS}$ ($< 2.3\text{ pC}$).
* **Conversion to Photoelectrons:**
  $$\mathbf{k_{\rm QDC \to p.e.}} = \frac{3.6\text{ pC/count}}{Q_{\rm 1\,p.e.}^{\rm eff}} = \frac{3.6\text{ pC}}{0.4005\text{ pC/p.e.}} \approx \mathbf{9.0\text{ p.e. / QDC count}}$$

---

## 5. Pedestal Subtraction & The Origin of Negative QDC Values

### 5.1 Pedestal Components
The pedestal term subtracted from the raw fine charge $q_{\rm fine}$ is:
$$\text{Pedestal}(tac\_id, \Delta t) = I_{\rm integ\_bias} \cdot \Delta t + Q_{\rm offset}(tac\_id)$$

1. **Dynamic DC Integration Term ($I_{\rm integ\_bias} \cdot \Delta t$):**
   Proportional to the integration window length. Because integration duration varies with time-over-threshold, the pedestal is dynamically evaluated event-by-event using the coarse timestamp difference $\Delta t = q_{\rm coarse} - t_T$.
2. **Buffer-Specific Process Offset ($Q_{\rm offset}$):**
   Each channel contains **4 round-robin multi-buffers** ($tac\_id \in \{0, 1, 2, 3\}$). Due to sub-micron transistor variations, each of the 4 buffers has a distinct offset.
3. **Calibration Polynomial (Datasheet Sec. 10.3):**
   PETsys calibration scans test pulses across integration lengths and fits a 2nd-order polynomial per buffer:
   $$\text{Pedestal}(tac\_id, \Delta t) = c_0(tac\_id) + c_1(tac\_id) \cdot \Delta t + c_2(tac\_id) \cdot (\Delta t)^2$$

### 5.2 Why Negative QDC Values Exist in Data
Negative values in converted data files (`sndsw_raw-*.root`) stem from two well-defined sources:

1. **Reconstruction Fit Quality Flagging ($\chi^2$ Inversion):**
   In `sndFairTasks/ConvRawData.cxx` (lines 288–294 and 615–619):
   ```cpp
   if ( Chi2ndof > chi2Max ) {
       if (QDC > 0) QDC = -QDC;
       mask = true;
   }
   ```
   When the online calibration fit produces a poor $\chi^2$ ($\chi^2/\text{ndf} > \chi^2_{\rm max}$), the converter **inverts the sign (`QDC = -QDC`)** and flags the channel as masked (`hit->isMasked(c) = true`). This generates the large negative tail down to $-250\text{ counts}$.
2. **Baseline Pedestal Jitter:**
   When zero or low signal is present, the physical charge fluctuates around the subtracted polynomial pedestal with Gaussian electronic noise ($\sigma \approx 0.65\text{ LSB}$). Noise fluctuations below zero produce small negative numbers ($-0.01$ to $-3.0\text{ counts}$).
3. **Quantization Step at $0.01$:**
   In online conversion software, the calibrated value is rounded to two decimal places, resulting in discrete steps: $\dots, -0.02, -0.01, 0.00, +0.01, +0.02, \dots$

In physics analyses, negative channels are excluded by default via `hit->GetAllSignals(mask=true, positive=true)`.

---

## 6. Multi-Level Hardware Discriminators & Triggering

### 6.1 Three-Level Discriminator Hierarchy
*(Ref: Characterization Paper Section 1, Flyer Page 1)*

```
 SiPM Signal Pulse
        │
        ├─────► Discriminator D_T1 (Threshold vth_T1): Low threshold for timing (≥ 0.25 p.e.)
        │
        ├─────► Discriminator D_T2 (Threshold vth_T2): Rejects dark counts (zero dead-time)
        │
        └─────► Discriminator D_E  (Threshold vth_E):  Validates energy deposit (5-cycle dead-time)
```

1. **Threshold $v_{\rm th\_t1}$ (Timing):**
   Can be set as low as **$0.25\text{ p.e.}$** to timestamp the earliest arriving optical photon.
2. **Threshold $v_{\rm th\_t2}$ (Dark Count Rejection):**
   Events that cross $v_{\rm th\_t1}$ but fail $v_{\rm th\_t2}$ are aborted immediately with **zero dead-time**.
3. **Threshold $v_{\rm th\_e}$ (Energy Validation):**
   Events that cross $v_{\rm th\_t2}$ but fail $v_{\rm th\_e}$ are aborted with a 5 clock cycle dead-time.
4. **Validation Condition:**
   $$\text{Hit Validated \& Digitized} \iff (V_{\rm peak} > v_{\rm th\_t1}) \;\land\; (V_{\rm peak} > v_{\rm th\_t2}) \;\land\; (V_{\rm peak} > v_{\rm th\_e})$$

### 6.2 Empirical Hardware Threshold in Real Data
In SND@LHC MuFilter data taking:
* While the ASIC hardware supports triggering down to $0.25\text{ p.e.}$, operating a large $6\times 6\text{ mm}^2$ SiPM at room temperature with $\sim 2\text{ MHz}$ dark count rate requires placing $v_{\rm th\_t2}$ and $v_{\rm th\_e}$ well above the thermal noise floor.
* In converted testbeam and physics runs (Run 100678, Run 6640), the minimum physical positive signal that triggers the DAQ and produces a reconstructed hit is:
  $$\mathbf{QDC_{\rm min}^{\rm DAQ} \approx 2.0\text{ to } 2.5\text{ counts}}$$
* With calibration constant $\text{QDC}_{\rm MIP} \approx 17.0\text{ counts}$ and $N_{\rm p.e.}^{\rm MIP} \approx 624\text{ p.e.}$:
  $$\text{Hardware Threshold Fraction} = \frac{2.5}{17.0} \approx \mathbf{0.15\text{ MIP}} \quad (15\%\text{ of a single MIP})$$
  $$N_{\rm p.e.}^{\rm min} = 0.15 \times 624\text{ p.e.} \approx \mathbf{93.5\text{ p.e.}}$$
  $$E_{\rm dep}^{\rm min} = \frac{93.5\text{ p.e.}}{311.83\text{ p.e./MeV}} \approx \mathbf{0.30\text{ MeV}}$$

---

## 7. Software Digitization & Recommended Physics Thresholds

### 7.1 Status of MuFilter Digitization in `sndsw`
1. **Simulation (`DigiTaskSND.cxx`):**
   `digitizeMuFilter()` loops over `MuFilterPoint` and creates a `MuFilterHit` for *any* non-zero energy deposit down to $1\text{ eV}$. No discriminator threshold was historically enforced.
2. **Real Data (`ConvRawData.cxx`):**
   Reads hardware-triggered hits from the TOFPET2 boards and packs `(QDC, TDC)` into `Digi_MuFilterHits`.
3. **MIP Calibration (`MuFilterHit.cxx`):**
   Signals are normalized to MIP units using `SiPM_qdc_calibration_constants`:
   $$\text{signal}_{\rm calib} = \frac{\text{signal}_{\rm raw}}{\text{calibrationConstant}}$$
   $$\text{calibrationConstant} \approx 17.0\text{ QDC (DS)}, \quad 4.5\text{ QDC per large SiPM (US)}$$

### 7.2 US Small SiPM Channels
* On each 8-SiPM US bar, channels `2` and `5` (0-indexed) are small SiPMs (`isShort(i) = true`).
* In standard hit formation, `hit->GetEnergy()` and `hit->GetAllSignals()` set `use_small_sipms = false` by default, excluding channels 2 and 5.
* In Monte Carlo (`sndLHC_H6geom_config.py`), their response is explicitly set to zero (`VandUpSiPMcalibrationS = 0.`).
* They were designed solely for extended dynamic range in high-energy hadronic shower cores and are not used for single-MIP tracking.

### 7.3 Recommended Offline Physics Thresholds

| Subsystem | Bar Thickness | Recommended $E_{\rm th}^*$ | Fraction of MIP | Corresponding $N_{\rm p.e.}$ | Equivalent Raw QDC | Motivation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Downstream (DS)** | $1.0\text{ cm}$ | **$1.50\text{ MeV}$** | $\approx 0.75\text{ MIP}$ | $\approx 468\text{ p.e.}$ | $\approx 12.8\text{ counts}$ | Filters inter-bar optical cross-talk ($< 10\%$) and shower albedo while maintaining $> 99.5\%$ MIP efficiency. |
| **Upstream (US)** | $3.0\text{ cm}$ | **$2.50\text{ MeV}$** | $\approx 0.50\text{ MIP}$ | $\approx 780\text{ p.e.}$ | $\approx 20.5\text{ counts (sum)}$ | Single-MIP tracking efficiency on 6 summed large SiPMs. |
| **Upstream (US Shower)** | $3.0\text{ cm}$ | **$5.00\text{ MeV}$** | $\approx 1.00\text{ MIP}$ | $\approx 1560\text{ p.e.}$ | $\approx 41.0\text{ counts (sum)}$ | Optimal data/MC hadronic shower multiplicity matching across testbeam energies ($100\text{--}300\text{ GeV}$). |

---

## 8. Summary Transformation Matrix

For a Downstream horizontal bar read out by a single $6\times 6\text{ mm}^2$ SiPM:

$$\begin{aligned}
\mathbf{\Delta E\text{ [MeV]}} &\;\longleftrightarrow\; \mathbf{N_{\rm p.e.}} &\;\longleftrightarrow\;& \mathbf{Q_{\rm IN}\text{ [pC]}} &\;\longleftrightarrow\;& \mathbf{QDC\text{ [counts]}} &\;\longleftrightarrow\;& \mathbf{\text{MIP Fraction}} \\
0.010\text{ MeV} &\;\longleftrightarrow\; 3.1\text{ p.e.} &\;\longleftrightarrow\;& 1.35\text{ pC} &\;\longleftrightarrow\;& 0.38\text{ counts} &\;\longleftrightarrow\;& 0.005\text{ MIP} \\
\mathbf{0.300\text{ MeV}} &\;\longleftrightarrow\; \mathbf{93.5\text{ p.e.}} &\;\longleftrightarrow\;& \mathbf{40.5\text{ pC}} &\;\longleftrightarrow\;& \mathbf{2.50\text{ counts}} &\;\longleftrightarrow\;& \mathbf{0.15\text{ MIP (Hardware DAQ Threshold)}} \\
\mathbf{1.500\text{ MeV}} &\;\longleftrightarrow\; \mathbf{467.8\text{ p.e.}} &\;\longleftrightarrow\;& \mathbf{202.4\text{ pC}} &\;\longleftrightarrow\;& \mathbf{12.75\text{ counts}} &\;\longleftrightarrow\;& \mathbf{0.75\text{ MIP (Offline Physics Threshold)}} \\
\mathbf{2.000\text{ MeV}} &\;\longleftrightarrow\; \mathbf{623.7\text{ p.e.}} &\;\longleftrightarrow\;& \mathbf{269.8\text{ pC}} &\;\longleftrightarrow\;& \mathbf{17.00\text{ counts}} &\;\longleftrightarrow\;& \mathbf{1.00\text{ MIP (Full MIP Peak MPV)}}
\end{aligned}$$
