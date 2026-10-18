# Integer Linear Programming (ILP) Dimuon Track Reconstruction & Discriminator Variables

This document provides a technical reference for the Integer Linear Programming (ILP) track reconstruction, the multi-track hypothesis testing framework, and the definition of all reconstructed dimuon discriminator observables in SND@LHC.

All variables described herein are calculated solely from digitized detector measurements (`Digi_MuFilterHits`, SciFi/MuFilter geometry, and channel timing). **They do not rely on Monte Carlo truth**, making them directly applicable to both real data and simulation.

---

## 1. Motivation & Challenge in Downstream (DS) Tracking

In SND@LHC, the Downstream Muon Filter (DS) consists of 4 tracking stations equipped with horizontal and vertical scintillator bars:
- DS Station 1–3: Horizontal (60 bars, $y$-coordinate) and Vertical (60 bars, $x$-coordinate) scintillator planes.
- DS Station 4: Vertical scintillator plane (60 bars, $x$-coordinate).

When dimuon events occur (e.g. $\nu_\mu \text{ CC}$ with charm production and semileptonic decay, or particle gun dimuon benchmarks), two charged muons simultaneously traverse the DS system. Reconstructing both tracks presents three major challenges:
1. **High Hit Multiplicity & Noise:** Accompanying hadronic activity or overlapping shower remnants produce extraneous hits.
2. **Hit Contention & Sharing:** Conventional sequential track followers (e.g. standard Kalman or Hough transforms) frequently cross-match hits between the two muon tracks or falsely greedily absorb hits from one track into another.
3. **2D-to-3D Associative Ambiguity:** Hits are recorded in separate 2D projections ($X-Z$ vertical bars and $Y-Z$ horizontal bars). A method is required to solve the $2 \times 2$ pairing between projections without Monte Carlo truth.

The ILP method addresses these challenges globally by formulating track finding as a **constrained binary optimization problem with hit exclusivity**.

---

## 2. Mathematical Formulation of the ILP Method

The track reconstruction is performed independently in the two orthogonal projections ($X-Z$ and $Y-Z$).

```
 ┌────────────────────────────────────────────────────────┐
 │            1. Input Hits (Digi_MuFilterHits)           │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │        2. Track Candidate Generation & Local Fit       │
 │    - Spans >= 3 distinct stations (3-hit or 4-hit)     │
 │    - OLS fit: u = m * z + b                            │
 │    - Cut: |m| <= 0.35,  chi2_red <= 20.0               │
 │    - Cost: c_i = chi2_i - lambda_hit * N_hits          │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │             3. Global ILP Optimization                 │
 │    min sum(c_i * x_i)  s.t.  sum_{i: h in trk} x_i <= 1 │
 │    - Hypothesis 1-track: sum(x_i) = 1  --> J_1         │
 │    - Hypothesis 2-track: sum(x_i) = 2  --> J_2         │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │           4. Multi-Track Multiplicity Test             │
 │               Delta J = J_1 - J_2                      │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │           5. 2D-to-3D Timing-Based Track Pairing       │
 │         min |t_mean(X) - t_mean(Y)| combinations       │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │         6. Dimuon Pair Metrics & Discriminator         │
 │      - DCA_3D and Common Vertex (vtx_reco)             │
 │      - Delta t_ns and Opening Angle theta_12           │
 │      - Normalized Quality: chi2_2mu                    │
 │      - Composite Discriminant Score: D_2mu             │
 └────────────────────────────────────────────────────────┘
```

### 2.1 Track Candidate Generation

For each projection, hits are grouped by detector station plane $p \in \{0, 1, 2, 3\}$.
1. Candidate trajectories are formed by taking combinations of hits spanning $\ge 3$ distinct stations (all combinations of 4 stations, followed by all combinations of 3 stations).
2. For each combination $\mathcal{H}_i = \{(z_k, u_k)\}$, an ordinary least squares (OLS) linear fit is performed:
   $$u = m \cdot z + b$$
   where $u$ is the transverse coordinate ($x$ or $y$) and $z$ is the longitudinal beam-axis coordinate.
3. Physical angular acceptance requirement:
   $$|m| \le 0.35 \quad (\theta \le 350\text{ mrad})$$
4. Hit measurement resolution:
   $$\sigma = \frac{1}{\sqrt{12}} \approx 0.2887\text{ cm}$$
5. Track fit goodness-of-fit:
   $$\chi^2_i = \sum_{k \in \mathcal{H}_i} \left( \frac{u_k - (m z_k + b)}{\sigma} \right)^2, \quad \text{NDF}_i = |\mathcal{H}_i| - 2$$
   $$\chi^2_{\text{red}, i} = \frac{\chi^2_i}{\max(1, \text{NDF}_i)}$$
   Candidates with $\chi^2_{\text{red}, i} > 20.0$ are rejected as unphysical.

### 2.2 Candidate Cost Function

Each valid candidate $i$ is assigned a scalar cost:
$$c_i = \chi^2_i - \lambda_{\text{hit}} \cdot N_{\text{hits}, i}$$
where:
- $\lambda_{\text{hit}} = 10.0$ is the hit reward parameter.
- $N_{\text{hits}, i} = |\mathcal{H}_i|$ is the number of hits in the candidate (3 or 4).

**Physical significance:** A 4-hit track with a good fit ($\chi^2 \approx 2$) receives a large negative cost ($c_i \approx 2 - 40 = -38$). A 3-hit track with $\chi^2 \approx 3$ receives $c_i \approx 3 - 30 = -27$. Minimizing total cost inherently prefers longer tracks with minimal fit residuals.

### 2.3 Optimization Formulation (Mixed-Integer Linear Program)

Let $M$ be the total number of candidate tracks in the projection. We define binary decision variables:
$$x_i \in \{0, 1\} \quad \forall i \in \{1, \dots, M\}$$
where $x_i = 1$ indicates that candidate $i$ is selected as a reconstructed track, and $x_i = 0$ otherwise.

The objective function to minimize is:
$$\min_{\mathbf{x}} J(\mathbf{x}) = \sum_{i=1}^M c_i x_i$$

Subject to the **Hit Exclusivity Constraints**:
$$\sum_{i : h \in \mathcal{H}_i} x_i \le 1 \quad \forall \text{ hit } h$$
This constraint guarantees that no detector hit is assigned to more than one track.

To evaluate competing track hypotheses, the ILP is solved under two constraints:
1. **1-Track Hypothesis:**
   $$\sum_{i=1}^M x_i = 1 \implies J_1^* = \min J(\mathbf{x})$$
2. **2-Track Hypothesis:**
   $$\sum_{i=1}^M x_i = 2 \implies J_2^* = \min J(\mathbf{x})$$

The optimization is solved exactly using the HiGHS Branch-and-Cut solver (`scipy.optimize.milp`).

---

## 3. Definition of Discriminator Observables

### 3.1 Track Multiplicity Discriminator ($\Delta J$)

The difference in optimal objective cost between the 1-track and 2-track hypotheses quantifies the statistical significance of a second track:
$$\Delta J_X = J_{1, X}^* - J_{2, X}^*$$
$$\Delta J_Y = J_{1, Y}^* - J_{2, Y}^*$$
$$\Delta J_{\text{total}} = \Delta J_X + \Delta J_Y$$

- **Genuine Dimuon Events:** The second track contributes another set of consistent hits, lowering the objective by approximately $-(4 \times \lambda_{\text{hit}} - \chi^2) \approx -35$. Therefore, $J_2^* \ll J_1^*$, leading to:
  $$\Delta J_{\text{total}} \gg 0 \quad (\text{typically } +40 \text{ to } +80)$$
- **Single-Muon Events (or 1 Muon + Noise):** Forcing a second track forces the solver to pick isolated noise hits or highly deviant segments with high $\chi^2$, resulting in $J_2^* \approx J_1^*$ or even infeasible/worse cost. Thus:
  $$\Delta J_{\text{total}} \approx 0 \text{ or } \le 0$$

### 3.2 2D-to-3D Matching & Timing Association

In 2-track events, the solver produces two 2D tracks in $X-Z$ ($x_1, x_2$) and two in $Y-Z$ ($y_1, y_2$). To eliminate the $2 \times 2$ pairing ambiguity, the average hit time $\bar{t}$ is computed for each candidate:
$$\bar{t} = \frac{1}{N_{\text{hits}}} \sum_{k \in \mathcal{H}} t_k$$

The pairing that minimizes the projection timing difference is selected:
$$\Delta t_{\text{pair}, A} = |\bar{t}(x_1) - \bar{t}(y_1)| + |\bar{t}(x_2) - \bar{t}(y_2)|$$
$$\Delta t_{\text{pair}, B} = |\bar{t}(x_1) - \bar{t}(y_2)| + |\bar{t}(x_2) - \bar{t}(y_1)|$$

The configuration with $\min(\Delta t_{\text{pair}, A}, \Delta t_{\text{pair}, B})$ unambiguously assigns the 3D tracks: Track 1 and Track 2.

### 3.3 Kinematic & Vertex Observables

From the matched 3D linear tracks:
$$\mathbf{r}_1(z) = \begin{pmatrix} m_{x1} z + b_{x1} \\ m_{y1} z + b_{y1} \\ z \end{pmatrix}, \quad \mathbf{r}_2(z) = \begin{pmatrix} m_{x2} z + b_{x2} \\ m_{y2} z + b_{y2} \\ z \end{pmatrix}$$

#### 3D Opening Angle ($\theta_{12}$)
The directional unit vectors $\mathbf{u}_1$ and $\mathbf{u}_2$ are:
$$\mathbf{u}_1 = \frac{(m_{x1}, m_{y1}, 1)}{\sqrt{m_{x1}^2 + m_{y1}^2 + 1}}, \quad \mathbf{u}_2 = \frac{(m_{x2}, m_{y2}, 1)}{\sqrt{m_{x2}^2 + m_{y2}^2 + 1}}$$
$$\theta_{12} = \arccos(\mathbf{u}_1 \cdot \mathbf{u}_2) \quad [\text{mrad}]$$

#### 3D Distance of Closest Approach (DCA)
The closest approach between the two infinite skew lines in 3D space:
$$\text{DCA}_{3D} = \frac{|(\mathbf{p}_1 - \mathbf{p}_2) \cdot (\mathbf{u}_1 \times \mathbf{u}_2)|}{|\mathbf{u}_1 \times \mathbf{u}_2|}$$
where $\mathbf{p}_1 = (b_{x1}, b_{y1}, 0)$ and $\mathbf{p}_2 = (b_{x2}, b_{y2}, 0)$.

#### Reconstructed Common Vertex ($\mathbf{v}_{\text{reco}}$)
The point midway between the two points of closest approach on the respective tracks:
$$\mathbf{v}_{\text{reco}} = \frac{1}{2} \left[ \mathbf{r}_1(z_1^*) + \mathbf{r}_2(z_2^*) \right]$$
where $z_1^*, z_2^*$ are the longitudinal coordinates of closest approach.

#### Fiducial Volume Flag (`vtx_in_target`)
Checks whether the extrapolated upstream common vertex lies within the physical target volume:
$$z \in [290.0, 450.0]\text{ cm}, \quad x \in [-48.0, -4.0]\text{ cm}, \quad y \in [15.0, 55.0]\text{ cm}$$

#### Timing Separation ($\Delta t$)
The absolute difference in reconstructed mean time between the two 3D tracks:
$$\Delta t = |\bar{t}_{\text{trk1}} - \bar{t}_{\text{trk2}}| \quad [\text{ns}]$$

### 3.4 Combined Dimuon Fit Quality ($\chi^2_{2\mu}$)

Individual track $\chi^2$ values, vertex compatibility, and timing consistency are combined into a single normalized $\chi^2$ observable:

$$\chi^2_{2\mu} = \frac{\chi^2_{\text{trk1}} + \chi^2_{\text{trk2}} + \left(\frac{\text{DCA}_{3D}}{\sigma_{\text{DCA}}}\right)^2 + \left(\frac{\Delta t}{\sigma_{\Delta t}}\right)^2}{\text{NDF}_{\text{total}}}$$

where:
- $\chi^2_{\text{trk1}} = \chi^2_{x1} + \chi^2_{y1}$
- $\chi^2_{\text{trk2}} = \chi^2_{x2} + \chi^2_{y2}$
- $\text{NDF}_{\text{total}} = \text{NDF}_{\text{trk1}} + \text{NDF}_{\text{trk2}} + 2$
- $\sigma_{\text{DCA}} = 1.5\text{ cm}$ (nominal primary vertex spatial resolution scale)
- $\sigma_{\Delta t} = 1.0\text{ ns}$ (nominal scintillator channel timing resolution scale)

For a genuine vertex-originating dimuon pair with high-quality tracking, $\chi^2_{2\mu} \approx 1.0 - 2.5$. If one track is an unrelated background or shower fragment, the DCA term or $\chi^2$ drives $\chi^2_{2\mu} \gg 10$.

### 3.5 Composite Dimuon Discriminant Score ($\mathcal{D}_{2\mu}$)

To provide a single decision variable that cleanly separates genuine dimuons from background (single muon + hadronic splash or random noise), a composite score is defined:

$$\mathcal{D}_{2\mu} = \frac{\Delta J_{\text{total}}}{10.0} - \ln\left(1 + \max(0, \chi^2_{2\mu})\right) - \frac{\text{DCA}_{3D}}{2.0}$$

**Rationale for the terms:**
1. **$+\frac{\Delta J_{\text{total}}}{10.0}$ (Multiplicity Reward):** Strongly rewards events where the detector measurements demand two distinct tracks rather than one.
2. **$-\ln(1 + \chi^2_{2\mu})$ (Fit & Consistency Penalty):** Gently penalizes moderate $\chi^2$ increases while suppressing severe track misfits or timing mismatches.
3. **$-\frac{\text{DCA}_{3D}}{2.0}$ (Common Vertex Penalty):** Strongly penalizes pairs of tracks that do not originate from a common vertex in the target (e.g. punch-through hadrons or independent beam halo muons).

---

## 4. TTree `ilp` Branch Summary Reference

When running `scripts/extract_ilp_features.py`, the following branches are written to the `ilp` TTree:

| Branch Name | Type | Description |
|:---|:---:|:---|
| `n_tracks_ilp_x` | `I` | Number of unconstrained ILP tracks in $X-Z$ |
| `n_tracks_ilp_y` | `I` | Number of unconstrained ILP tracks in $Y-Z$ |
| `n_tracks_ilp_3d` | `I` | Number of matched 3D tracks (0, 1, or 2) |
| `j_opt_1track_x` | `F` | Optimal ILP objective cost for 1-track hypothesis in $X-Z$ |
| `j_opt_2track_x` | `F` | Optimal ILP objective cost for 2-track hypothesis in $X-Z$ |
| `delta_j_x` | `F` | Multiplicity cost gain $\Delta J_X = J_{1, X}^* - J_{2, X}^*$ |
| `j_opt_1track_y` | `F` | Optimal ILP objective cost for 1-track hypothesis in $Y-Z$ |
| `j_opt_2track_y` | `F` | Optimal ILP objective cost for 2-track hypothesis in $Y-Z$ |
| `delta_j_y` | `F` | Multiplicity cost gain $\Delta J_Y = J_{1, Y}^* - J_{2, Y}^*$ |
| `delta_j_total` | `F` | Combined cost gain $\Delta J_{\text{total}} = \Delta J_X + \Delta J_Y$ |
| `trk1_chi2_total` | `F` | Summed $\chi^2 = \chi^2_X + \chi^2_Y$ for Track 1 |
| `trk1_reduced_chi2` | `F` | Reduced $\chi^2 / \text{NDF}$ for Track 1 |
| `trk1_slope_xz` | `F` | Slope $\mathrm{d}x/\mathrm{d}z$ for Track 1 |
| `trk1_intercept_xz` | `F` | Intercept $x(z=0)$ [cm] for Track 1 |
| `trk1_slope_yz` | `F` | Slope $\mathrm{d}y/\mathrm{d}z$ for Track 1 |
| `trk1_intercept_yz` | `F` | Intercept $y(z=0)$ [cm] for Track 1 |
| `trk1_t_mean` | `F` | Mean reconstructed hit time [ns] for Track 1 |
| `trk1_qdc_total` | `F` | Total integrated QDC for Track 1 |
| `trk2_chi2_total` | `F` | Summed $\chi^2 = \chi^2_X + \chi^2_Y$ for Track 2 |
| `trk2_reduced_chi2` | `F` | Reduced $\chi^2 / \text{NDF}$ for Track 2 |
| `trk2_slope_xz` | `F` | Slope $\mathrm{d}x/\mathrm{d}z$ for Track 2 |
| `trk2_intercept_xz` | `F` | Intercept $x(z=0)$ [cm] for Track 2 |
| `trk2_slope_yz` | `F` | Slope $\mathrm{d}y/\mathrm{d}z$ for Track 2 |
| `trk2_intercept_yz` | `F` | Intercept $y(z=0)$ [cm] for Track 2 |
| `trk2_t_mean` | `F` | Mean reconstructed hit time [ns] for Track 2 |
| `trk2_qdc_total` | `F` | Total integrated QDC for Track 2 |
| `delta_theta_xz_mrad`| `F`| Slope difference in $X-Z$ projection [mrad] |
| `delta_theta_yz_mrad`| `F`| Slope difference in $Y-Z$ projection [mrad] |
| `opening_angle_3d_mrad`| `F`| 3D opening angle $\theta_{12}$ [mrad] |
| `dca_3d` | `F` | 3D Distance of Closest Approach between tracks [cm] |
| `vtx_reco_x`, `y`, `z` | `F` | 3D coordinates of reconstructed midpoint vertex [cm] |
| `vtx_in_target` | `I` | Flag: 1 if common vertex is inside target fiducial box, 0 otherwise |
| `delta_t_ns` | `F` | Absolute timing separation $|\bar{t}_1 - \bar{t}_2|$ [ns] |
| `chi2_2mu` | `F` | Combined dimuon quality metric including DCA and $\Delta t$ |
| `dimuon_discriminant`| `F` | Composite decision score $\mathcal{D}_{2\mu}$ |
