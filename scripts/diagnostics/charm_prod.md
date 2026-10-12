# Theoretical and Diagnostic Formulation: Charm Production Rate $\sigma_{\text{charm}} / \sigma_{\text{CC}}$ in SND@LHC

## 1. Introduction and Physics Motivation

In neutrino-induced charged-current (CC) deep inelastic scattering (DIS), charmed hadron production proceeds via the exchange of a $W^\pm$ boson with the target nucleon quarks:
- **Neutrino CC ($\nu_\mu + N \to \mu^- + c + X$)**:
  - $W^+ + d \to c$ (Cabibbo-suppressed, proportional to $|V_{cd}|^2 \approx 0.0506$)
  - $W^+ + s \to c$ (Cabibbo-favored, proportional to $|V_{cs}|^2 \approx 0.9476$)
- **Antineutrino CC ($\bar{\nu}_\mu + N \to \mu^+ + \bar{c} + X$)**:
  - $W^- + \bar{d} \to \bar{c}$ (Cabibbo-suppressed, proportional to $|V_{cd}|^2$)
  - $W^- + \bar{s} \to \bar{c}$ (Cabibbo-favored, proportional to $|V_{cs}|^2$)

Unlike light-quark production, the final-state charm quark possesses a substantial mass ($m_c \approx 1.27 - 1.50\text{ GeV}$), introducing a kinematic threshold and a suppression of the available phase space at low energies. 

The ratio:
$$R_{\text{charm}}(E_\nu) = \frac{\sigma(\nu_\mu N \to \mu^- c X)}{\sigma_{\text{CC}}(\nu_\mu N \to \mu^- X)}$$
serves as a primary probe of the strange quark sea content in the nucleon, provides a direct measurement of the CKM matrix element $|V_{cd}|$, and acts as a standard benchmark for the neutrino event generator (GENIE) within the SND@LHC target.

---

## 2. The Slow Rescaling Model (Georgi–Politzer / Barnett)

### 2.1 Kinematic Derivation
In the parton model with massless final-state quarks, the fraction of nucleon momentum carried by the struck quark is the standard Bjorken scaling variable:
$$x = \frac{Q^2}{2 M_N \nu} = \frac{Q^2}{2 M_N E_\nu y}$$
where $Q^2 = -q^2$ is the four-momentum transfer squared, $\nu = E_\nu - E_\mu$, and $y = \nu / E_\nu$ is the inelasticity.

When producing a heavy charm quark of mass $m_c$, the mass-shell condition for the outgoing quark requires $(p_{\text{quark}} + q)^2 = m_c^2$. Neglecting the light initial quark mass and nucleon target mass corrections, this yields:
$$(\xi p_N + q)^2 = m_c^2 \implies \xi^2 M_N^2 + 2 \xi (p_N \cdot q) - Q^2 = m_c^2$$
Dropping the target mass term $\xi^2 M_N^2 \ll Q^2$, we arrive at the **slow rescaling variable** $\xi$:
$$\xi = x \left(1 + \frac{m_c^2}{Q^2}\right) = x + \frac{m_c^2}{2 M_N E_\nu y}$$

Physical phase space requires $\xi \le 1$. Consequently, the threshold neutrino energy to produce a charm quark on a nucleon at rest is:
$$E_{\nu}^{\text{thresh}} = \frac{(M_N + m_c)^2 - M_N^2}{2 M_N} + \Delta E_{\text{had}} \approx 3.0\text{ to }4.0\text{ GeV}$$

### 2.2 Functional Form
In the asymptotic regime ($E_\nu \gg E_{\nu}^{\text{thresh}}$), the cross-section ratio approaches a plateau $R_\infty$. Near threshold, kinematic mass suppression slows down the rise toward this plateau ("slow rescaling"). The ratio is modeled phenomenologically as:
$$R_{\text{charm}}(E_\nu) = \begin{cases} 0, & E_\nu < E_{\nu}^{\text{thresh}} \\ R_\infty \cdot \left(1 - \frac{E_{\nu}^{\text{thresh}}}{E_\nu}\right)^\alpha \cdot \left(\frac{E_\nu}{E_\nu + E_{\text{scale}}}\right), & E_\nu \ge E_{\nu}^{\text{thresh}} \end{cases}$$
where:
- $E_{\nu}^{\text{thresh}} = \frac{(M_N + m_c)^2 - M_N^2}{2 M_N} + 1.50\text{ GeV} \approx 4.0\text{ GeV}$
- $\alpha \approx 1.15$ governs the threshold turn-on curvature
- $E_{\text{scale}} \approx 28.0\text{ GeV}$ sets the transitional scale between the threshold turn-on and the asymptotic plateau
- $R_\infty$ is determined by parton distribution functions and target composition.

---

## 3. Adapting to Nuclear Targets: Tungsten ($^{184}_{74}\text{W}$) vs. Isoscalar ($N=Z$)

The SND@LHC emulsion target consists of tungsten bricks ($A = 183.84$, $Z = 74$). A natural tungsten nucleus has a significant neutron excess:
$$f_p = \frac{Z}{A} = \frac{74}{183.84} \approx 0.4025, \quad f_n = 1 - f_p = \frac{110}{183.84} \approx 0.5975$$
In contrast, an ideal isoscalar target has $f_p = f_n = 0.50$.

### 3.1 Target Nucleon Parton Densities
By isospin symmetry:
- Proton: $u_p(x) = u_{\text{val}}(x) + \bar{q}(x)$, $d_p(x) = d_{\text{val}}(x) + \bar{q}(x)$
- Neutron: $u_n(x) = d_p(x)$, $d_n(x) = u_p(x)$

At representative DIS scales ($Q^2 \sim 10 - 20\text{ GeV}^2$), the momentum integrals $\int x q(x) dx$ are:
- Valence: $U_{\text{val}} \approx 0.285$, $D_{\text{val}} \approx 0.130$
- Light-quark sea: $\bar{u} \approx \bar{d} \approx \bar{q} \approx 0.035$
- Strange sea: $s = \bar{s} = \kappa_s \cdot \bar{q}$

Here $\kappa_s \equiv \frac{2s}{\bar{u} + \bar{d}}$ is the strange sea suppression factor. Early fixed-target experiments (CDHS, CCFR, NuTeV) measured $\kappa_s \approx 0.38 \pm 0.05$. Modern global PDF fits (NNPDF, CT18, ATLAS $W/Z$) indicate a less suppressed strange sea, $\kappa_s \approx 0.55 \pm 0.08$. In our baseline model, we adopt $\kappa_s = 0.55$, giving:
$$s \approx 0.55 \times 0.035 = 0.01925$$

The average target nucleon densities are:
$$\begin{aligned}
d_{\text{target}} &= f_p D_{\text{val}} + f_n U_{\text{val}} + \bar{q} \\
u_{\text{target}} &= f_p U_{\text{val}} + f_n D_{\text{val}} + \bar{q}
\end{aligned}$$

Numerical values:
| Target | $f_p$ | $f_n$ | $d_{\text{target}}$ | $u_{\text{target}}$ |
| :--- | :--- | :--- | :--- | :--- |
| **Isoscalar ($N=Z$)** | $0.5000$ | $0.5000$ | $0.2425$ | $0.2425$ |
| **Tungsten ($^{184}_{74}\text{W}$)** | $0.4025$ | $0.5975$ | $0.2576$ | $0.2274$ |

Because tungsten is neutron-rich ($f_n \approx 60\%$), $d_{\text{target}}$ is **enhanced** relative to an isoscalar target by $\sim 6.2\%$, while $u_{\text{target}}$ is suppressed.

---

## 4. Predicting $R_\infty$: Neutrino, Antineutrino, and Inclusive

### 4.1 Neutrino Beam ($\nu_\mu$)
The $W^+$ boson interacts with $d$ and $s$ quarks with flat inelasticity $y$ ($\int_0^1 dy = 1$), and with $\bar{u}$ antiquarks with $(1-y)^2$ suppression ($\int_0^1 (1-y)^2 dy = 1/3$):
$$\sigma(\nu_\mu N \to \mu^- c X) \propto |V_{cd}|^2 d_{\text{target}} + |V_{cs}|^2 s$$
$$\sigma_{\text{CC}}(\nu_\mu N \to \mu^- X) \propto d_{\text{target}} + s + \frac{1}{3} \bar{q}$$

$$R_\infty(\nu_\mu) = \frac{|V_{cd}|^2 d_{\text{target}} + |V_{cs}|^2 s}{d_{\text{target}} + s + \frac{1}{3} \bar{q}}$$

Evaluating with $|V_{cd}| = 0.22503$ and $|V_{cs}| = 0.97345$:
- **Isoscalar target**:
  $$R_\infty^{\text{iso}}(\nu_\mu) = \frac{(0.0506 \times 0.2425) + (0.9476 \times 0.01925)}{0.2425 + 0.01925 + 0.01167} = \frac{0.03051}{0.27342} \approx \mathbf{0.1116}$$
- **Tungsten target**:
  $$R_\infty^{\text{W}}(\nu_\mu) = \frac{(0.0506 \times 0.2576) + (0.9476 \times 0.01925)}{0.2576 + 0.01925 + 0.01167} = \frac{0.03127}{0.28852} \approx \mathbf{0.1084}$$

> [!NOTE]
> Why does Tungsten have a slightly lower $R_\infty(\nu_\mu)$ than Isoscalar?
> Even though $d_{\text{target}}$ is larger in Tungsten, $d$-quark charm production is Cabibbo-suppressed ($|V_{cd}|^2 \approx 0.05$). Thus increasing $d_{\text{target}}$ increases the total CC denominator significantly faster than it increases the charm numerator, resulting in a slightly smaller ratio: $10.84\%$ vs. $11.16\%$.

### 4.2 Antineutrino Beam ($\bar{\nu}_\mu$)
The $W^-$ boson interacts with $\bar{d}$ and $\bar{s}$ sea antiquarks with flat $y$, and with $u_{\text{target}}$ valence quarks with $(1-y)^2$ suppression:
$$\sigma(\bar{\nu}_\mu N \to \mu^+ \bar{c} X) \propto |V_{cd}|^2 \bar{q} + |V_{cs}|^2 s$$
$$\sigma_{\text{CC}}(\bar{\nu}_\mu N \to \mu^+ X) \propto \frac{1}{3} u_{\text{target}} + \bar{q} + s$$

$$R_\infty(\bar{\nu}_\mu) = \frac{|V_{cd}|^2 \bar{q} + |V_{cs}|^2 s}{\frac{1}{3} u_{\text{target}} + \bar{q} + s}$$

Evaluating for Tungsten:
$$R_\infty^{\text{W}}(\bar{\nu}_\mu) = \frac{(0.0506 \times 0.035) + (0.9476 \times 0.01925)}{\frac{1}{3}(0.2274) + 0.035 + 0.01925} = \frac{0.02001}{0.13005} \approx \mathbf{0.1539}$$

> [!IMPORTANT]
> The antineutrino charm ratio is **substantially higher** ($\sim 15.4\%$ vs $\sim 10.8\%$) because the total CC denominator is suppressed: the dominant valence interaction $W^- + u \to d$ carries a $(1-y)^2$ factor that integrates to $1/3$, whereas charm production from the strange sea $W^- + \bar{s} \to \bar{c}$ has a flat $y$ distribution.

### 4.3 Inclusive ($\nu_\mu + \bar{\nu}_\mu$)
In the forward LHC neutrino beam at SND@LHC, the CC event sample is an admixture of neutrinos and antineutrinos:
$$\Phi_\nu \approx 65\%, \quad \Phi_{\bar{\nu}} \approx 35\%$$
Weighting the respective numerators and denominators:
$$R_\infty^{\text{inc}} = \frac{0.65 \cdot \sigma_{\text{charm}}(\nu_\mu) + 0.35 \cdot \sigma_{\text{charm}}(\bar{\nu}_\mu)}{0.65 \cdot \sigma_{\text{CC}}(\nu_\mu) + 0.35 \cdot \sigma_{\text{CC}}(\bar{\nu}_\mu)}$$
$$R_\infty^{\text{W, inc}} = \frac{0.65(0.03127) + 0.35(0.02001)}{0.65(0.28852) + 0.35(0.13005)} = \frac{0.02733}{0.23306} \approx \mathbf{0.1173}$$

---

## 5. Why Do Neutrinos and Antineutrinos Deviate Differently from Isoscalar Predictions?

When transitioning from a symmetric isoscalar target ($N = Z$) to a neutron-rich nucleus like Tungsten ($N \approx 1.5\,Z$), the charm production ratio shifts in **opposite directions** for neutrinos and antineutrinos:

$$\text{Neutrino } (\nu_\mu): \quad R_\infty^{\text{iso}} \approx 11.16\% \;\longrightarrow\; R_\infty^{\text{W}} \approx 10.84\% \quad (\mathbf{-2.9\%\text{ relative shift}})$$
$$\text{Antineutrino } (\bar{\nu}_\mu): \quad R_\infty^{\text{iso}} \approx 14.82\% \;\longrightarrow\; R_\infty^{\text{W}} \approx 15.39\% \quad (\mathbf{+3.9\%\text{ relative shift}})$$

This opposing behavior is entirely governed by the interplay between **target isospin composition** and **weak interaction chiral selection rules**:

### 5.1 Flavor Densities in Neutron-Rich Tungsten
Natural tungsten has $f_n = 59.75\%$ and $f_p = 40.25\%$. By isospin symmetry, a neutron contains valence $d$-quarks where a proton contains $u$-quarks:
- The target $d$-density increases: $d_{\text{target}}^{\text{W}} = 0.2576$ (up $+6.2\%$ from $0.2425$ in an isoscalar target).
- The target $u$-density decreases: $u_{\text{target}}^{\text{W}} = 0.2274$ (down $-6.2\%$ from $0.2425$ in an isoscalar target).
- The strange sea ($s = \bar{s}$) and light antiquark sea ($\bar{u} \approx \bar{d}$) are isoscalar and unaffected by nuclear neutron excess.

### 5.2 Why $R_{\text{charm}}(\nu_\mu)$ Shifts Downward in Tungsten
1. **Charm Production**: Proceeding via $W^+ + d \to c$, it is Cabibbo-suppressed ($|V_{cd}|^2 \approx 0.0506$). The dominant charm channel is $W^+ + s \to c$ ($|V_{cs}|^2 \approx 0.9476$).
2. **Total CC Interaction**: Dominated by $W^+ + d \to u$, which is Cabibbo-favored ($|V_{ud}|^2 \approx 0.949$).
3. **The Ratio**:
   $$R(\nu_\mu) \sim \frac{|V_{cd}|^2 d_{\text{target}} + |V_{cs}|^2 s}{d_{\text{target}} + s + \frac{1}{3}\bar{q}}$$
   Adding neutrons increases $d_{\text{target}}$. Because $d_{\text{target}}$ enters the denominator with full unit weight ($1.0$) but enters the numerator multiplied only by $|V_{cd}|^2 \approx 0.05$, **the total CC cross section grows faster than charm production**. Consequently, $R(\nu_\mu)$ decreases slightly in tungsten.

### 5.3 Why $R_{\text{charm}}(\bar{\nu}_\mu)$ Shifts Upward in Tungsten
1. **Charm Production**: Proceeding via $W^- + \bar{d} \to \bar{c}$ and $W^- + \bar{s} \to \bar{c}$, it originates strictly from the **sea antiquarks** ($\bar{d}, \bar{s}$). Since the sea is isoscalar, the charm production numerator does not change with neutron excess!
2. **Total CC Interaction**: Dominated by $W^- + u \to d$ off valence $u$-quarks, suppressed by the $(1-y)^2$ helicity factor ($\int (1-y)^2 dy = 1/3$).
3. **The Ratio**:
   $$R(\bar{\nu}_\mu) \sim \frac{|V_{cd}|^2 \bar{q} + |V_{cs}|^2 s}{\frac{1}{3}u_{\text{target}} + \bar{q} + s}$$
   Because tungsten has fewer protons ($f_p \approx 40.2\%$), the valence $u$-quark density $u_{\text{target}}$ is **depleted**. Since $u_{\text{target}}$ only appears in the denominator, the denominator shrinks while the numerator remains constant. Consequently, $R(\bar{\nu}_\mu)$ increases in tungsten.

### 5.4 Dedicated Isoscalar Baseline per Canvas
To prevent confusing cross-comparisons (such as comparing antineutrino tungsten data against a neutrino isoscalar line), each canvas plots its **exact matching isoscalar baseline**:
- `c_charm_rate_vs_energy`: Compares Tungsten Inclusive ($R_\infty \approx 11.73\%$) against Isoscalar Inclusive ($R_\infty \approx 12.05\%$).
- `c_charm_rate_vs_energy_nu_only`: Compares Tungsten $\nu_\mu$ ($R_\infty \approx 10.84\%$) against Isoscalar $\nu_\mu$ ($R_\infty \approx 11.16\%$).
- `c_charm_rate_vs_energy_antinu_only`: Compares Tungsten $\bar{\nu}_\mu$ ($R_\infty \approx 15.39\%$) against Isoscalar $\bar{\nu}_\mu$ ($R_\infty \approx 14.82\%$).

---

## 6. Why Does the Isoscalar Dashed Line Exceed the Historical $\sim 0.088$ Value?

In earlier literature (e.g. CDHS 1982, CCFR 1995), the charm fraction was quoted as peaking around $8 - 9\%$. 
The reason the modern isoscalar curve plateaus at $\sim 0.111$ (and reaches $\sim 0.108$ at $1\text{ TeV}$) stems from:

1. **Strange Sea Normalization ($\kappa_s$)**:
   - Historical analyses assumed $\kappa_s \approx 0.35 - 0.38$. If $\kappa_s = 0.35$:
     $$R_\infty^{\text{iso}}(\kappa_s = 0.35) \approx \mathbf{0.0897}$$
   - When modern global PDF fits ($\kappa_s \approx 0.55$) are used, the Cabibbo-favored strange channel contribution $|V_{cs}|^2 s$ increases by $\sim 50\%$, elevating the plateau to $11.1\%$.
2. **Kinematic Reach ($E_\nu$)**:
   - At fixed-target energies ($E_\nu < 150\text{ GeV}$), slow rescaling kinematics damp the ratio to below $9.5\%$.
   - At SND@LHC energies ($E_\nu \sim 300 - 1000\text{ GeV}$), the kinematic suppression factor $(1 - E_{\text{thresh}}/E_\nu)^\alpha \cdot \frac{E_\nu}{E_\nu + 28}$ approaches $0.96 - 0.98$, revealing the true asymptotic value.

---

## 6. Two-Tier Uncertainty Estimation

To provide a rigorous theoretical comparison against the MC simulation, the script [`charm_diagnostic.py`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/scripts/diagnostics/charm_diagnostic.py) implements a **two-tier uncertainty belt**:

```
+---------------------------------------------------------------------------------------+
|  Tier 2: Outer Uncertainty Belt (alpha = 0.20, light nuance, dashed boundary)          |
|  - High-energy QCD scale evolution (DGLAP Q^2 scaling violations)                     |
|  - Strange sea evolution at high energies (deviations between LO model and full DIS)  |
|                                                                                       |
|      +-------------------------------------------------------------------------+      |
|      |  Tier 1: Inner Uncertainty Belt (alpha = 0.45, deeper solid fill)       |      |
|      |  - Base charm threshold mass uncertainty: sigma(m_c) = +/- 4.5 MeV      |      |
|      |  - CKM matrix elements: sigma(|V_cd|), sigma(|V_cs|)                     |      |
|      |  - Leading-order PDF normalization: ~ 8% base uncertainty               |      |
|      +-------------------------------------------------------------------------+      |
+---------------------------------------------------------------------------------------+
```

### 6.1 Tier 1: Inner Belt (Threshold Kinematics & LO PDF Base)
The inner band represents fundamental parametric uncertainties in the leading-order slow rescaling framework:
1. **Charm Quark Mass ($m_c$)**:
   - Evaluated at $\bar{m}_c(\bar{m}_c) = 1.2729 \pm 0.0045\text{ GeV}$.
   - The threshold shift is $\frac{\partial E_{\text{thresh}}}{\partial m_c} = \frac{M_N + m_c}{M_N} \approx 2.35$.
   - Kinematic sensitivity:
     $$\sigma_{\text{kin}}(E_\nu) = \left| \frac{\partial R(E_\nu)}{\partial E_{\text{thresh}}} \right| \cdot \sigma(E_{\text{thresh}})$$
2. **Base PDF and CKM Normalization**:
   - A conservative base uncertainty of $\pm 8\%$ is assigned to $R_\infty$, encompassing $|V_{cd}|$, $|V_{cs}|$, and local $\kappa_s$ PDF normalization uncertainties at reference scales:
     $$\sigma_{R,\text{base}}(E_\nu) = \text{kin\_factor}(E_\nu) \times (0.08 \cdot R_\infty)$$
3. **Combined Inner Error**:
   $$\sigma_{\text{inner}}(E_\nu) = \sqrt{\sigma_{R,\text{base}}^2(E_\nu) + \sigma_{\text{kin}}^2(E_\nu)}$$
   - Formatted in ROOT with `SetFillColorAlpha(band_color, 0.45)` and `SetFillStyle(1001)`.

### 6.2 Tier 2: Outer Belt (Asymmetric High-Energy QCD & Strange Sea Evolution)
As neutrino energy reaches several hundreds of GeV up to $1.5\text{ TeV}$, leading-order slow rescaling is subject to additional QCD effects that **asymmetrically enhance** the charm cross-section ratio rather than suppressing it:
1. **Boson-Gluon Fusion ($W^+ g \to c\bar{s}$)**: At $\mathcal{O}(\alpha_s)$, gluon-initiated charm production opens up. Because the gluon density $g(x, Q^2)$ grows steeply at small $x$, this acts as a strictly positive additive contribution that is absent from LO $2 \to 2$ scattering.
2. **DGLAP $Q^2$ Strange Sea Enhancement**: Under DGLAP evolution, $Q^2$ grows with $E_\nu$ ($Q^2 \sim 2 M_N E_\nu x y$). Gluons split into $s\bar{s}$ pairs ($g \to s\bar{s}$), enhancing the effective strange sea fraction at DIS scales. Because charm production from strange quarks is Cabibbo-favored ($|V_{cs}|^2 \approx 0.95$), this disproportionately enhances the charm numerator over the total CC denominator.
3. **Physical Asymmetry**:
   - **Downward bound ($-\Delta R$)**: Constrained to the symmetric Tier 1 baseline (charm mass and base PDF/CKM uncertainty $\sim 8\%$). The ratio cannot drop significantly below LO at high energies because sea quarks and gluons grow while valence quarks decrease.
   - **Upward bound ($+\Delta R$)**: Expands logarithmically with energy from $E_0 = 40\text{ GeV}$ up to $1\text{ TeV}$ to cover the positive shift from boson-gluon fusion and strange sea growth:
     $$\delta_{\text{high-E}}^{\text{up}}(E_\nu) = 0.28 \cdot \frac{\ln\left[\max(1, E_\nu / 40\text{ GeV})\right]}{\ln(1000 / 40)}$$
     $$\sigma_{\text{outer}}^{\text{up}}(E_\nu) = \sqrt{\left[\text{kin\_factor}(E_\nu) \cdot R_\infty \cdot \sqrt{(0.08)^2 + (\delta_{\text{high-E}}^{\text{up}})^2}\right]^2 + \sigma_{\text{kin}}^2(E_\nu)}$$
   - Formatted in ROOT using `TGraphAsymmErrors` with `SetFillColorAlpha(band_color, 0.20)`, `SetFillStyle(1001)`, and a dashed boundary `SetLineStyle(3)`.

---

## 7. ROOT Implementation Summary

The full implementation is encapsulated in `create_slow_rescaling_model()` in [`charm_diagnostic.py`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/scripts/diagnostics/charm_diagnostic.py):

```python
f_sr, g_band_inner, g_band_outer = create_slow_rescaling_model(
    name="f_slow_rescaling_w_inclusive",
    e_min=0.0,
    e_max=1500.0,
    target="tungsten",     # Accurately applies f_p=0.4025, f_n=0.5975
    nu_type="inclusive",   # Weighted average: 65% nu_mu, 35% anti-nu_mu
    line_color=ROOT.kRed + 1,
    band_color=ROOT.kRed - 7,
)
```

Three dedicated canvases are generated:
1. `c_charm_rate_vs_energy`: Inclusive $\nu_\mu + \bar{\nu}_\mu$ CC rate on Tungsten, compared against MC inclusive, with the Isoscalar dashed reference line and CHORUS measurement.
2. `c_charm_rate_vs_energy_nu_only`: Pure $\nu_\mu$ CC rate on Tungsten, compared against MC $\nu_\mu$ CC, with the Isoscalar dashed reference line and CHORUS measurement.
3. `c_charm_rate_vs_energy_antinu_only`: Pure $\bar{\nu}_\mu$ CC rate on Tungsten, compared against MC $\bar{\nu}_\mu$ CC, with the matching $\bar{\nu}_\mu$ Isoscalar dashed reference line.

---

## 8. External Experimental Comparisons: CHORUS and E531 Measurements

To anchor the slow rescaling threshold turn-on against real experimental data without overwhelming the primary SND@LHC MC simulation and theoretical bands, the diagnostic plots overlay published measurements from **CHORUS** and **Fermilab E531** with prominent black open markers:

1. **CHORUS Collaboration** (*New J. Phys.* 13 (2011) 093002, arXiv:1107.0613):
   - **Beam**: CERN SPS West Area Neutrino Facility (WANF), wide-band $\nu_\mu$ beam ($\langle E_\nu \rangle \approx 27\text{ GeV}$, covering $10 - 200\text{ GeV}$).
   - **Target**: Nuclear emulsion ($\langle Z/A \rangle \approx 0.467$).
   - **Styling**: Black open squares ($\Box$, marker 25, `ROOT.kBlack`, size 1.20), linewidth 2.
2. **Fermilab E531 Collaboration** (*Phys. Lett. B* 206 (1988) 375):
   - **Beam**: Fermilab wide-band neutrino beam ($E_\nu \sim 5 - 150\text{ GeV}$).
   - **Target**: Nuclear emulsion hybrid spectrometer.
   - **Styling**: Black open circles ($\bigcirc$, marker 24, `ROOT.kBlack`, size 1.20), linewidth 2.

### 8.1 Nuclear Target Comparison: Emulsion vs Tungsten
| Target | $\langle Z/A \rangle$ ($f_p$) | $f_n$ | Predicted $R_\infty(\nu_\mu)$ |
| :--- | :--- | :--- | :--- |
| **Isoscalar ($N=Z$)** | $0.5000$ | $0.5000$ | $11.16\%$ |
| **Emulsion (CHORUS / E531)** | $0.4670$ | $0.5330$ | $11.05\%$ |
| **SND@LHC Tungsten** | $0.4025$ | $0.5975$ | $10.84\%$ |

Because nuclear emulsion sits between an isoscalar target and Tungsten, the difference in the predicted slow rescaling curve between Emulsion and Tungsten is minuscule ($\Delta R_\infty \approx 0.2\%$). The CHORUS and E531 data points cleanly delineate the low-energy threshold region ($E_\nu < 150\text{ GeV}$), showing consistent agreement with the slow rescaling turn-on kinematics ($E_{\text{scale}} \approx 28\text{ GeV}$, $\alpha \approx 1.15$), while SND@LHC probes the higher-energy DIS evolution ($E_\nu > 150\text{ GeV}$ up to $1.5\text{ TeV}$).

---

## 9. Dimuon Production Ratio $\sigma_{\text{charm, }2\mu} / \sigma_{\text{CC}}$ in SND@LHC

### 9.1 Physics Motivation and Dimuon Signature
In neutrino-induced charged-current deep inelastic scattering, opposite-sign dimuon events ($\mu^\mp \mu^\pm$) provide an experimentally clean and historically crucial signature of charm production:
$$\nu_\mu + N \longrightarrow \mu^- + c + X \longrightarrow \mu^- + \mu^+ + \nu_\mu + X'$$
$$\bar{\nu}_\mu + N \longrightarrow \mu^+ + \bar{c} + X \longrightarrow \mu^+ + \mu^- + \bar{\nu}_\mu + X'$$

The leading primary muon originates directly from the weak leptonic vertex, while the secondary, opposite-sign muon arises from the prompt semi-muonic decay of the produced charmed hadron ($c \to \mu^+ \nu_\mu s$).

The dimuon cross-section ratio is defined as:
$$R_{2\mu}(E_\nu) \equiv \frac{\sigma(\nu_\mu N \to \mu^- \mu^+ X)}{\sigma_{\text{CC}}(\nu_\mu N \to \mu^- X)} = \frac{\sigma_{\text{charm, }2\mu}(E_\nu)}{\sigma_{\text{CC}}(E_\nu)}$$

### 9.2 Relation to Inclusive Charm Production
Under the factorization hypothesis in neutrino DIS, charm hadronization and subsequent weak decay factorize from the hard partonic scattering:
$$R_{2\mu}(E_\nu) = \langle B_\mu \rangle \cdot R_{\text{charm}}(E_\nu)$$
where $\langle B_\mu \rangle$ is the **effective inclusive semi-muonic branching fraction** averaged over the charm hadron species production fractions in neutrino DIS:
$$\langle B_\mu \rangle = \sum_{i} f_i \cdot \mathcal{B}(H_{c, i} \to \mu^+ X)$$
where $i \in \{D^0, D^+, D_s^+, \Lambda_c^+\}$, $f_i$ are the fragmentation fractions, and $\mathcal{B}(H_{c, i} \to \mu^+ X)$ are the inclusive semi-muonic branching ratios.

### 9.3 Derivation of Effective Branching Ratio $\langle B_\mu \rangle$
Using global neutrino DIS measurements (CHORUS, NuTeV, NOMAD) for the fragmentation fractions and PDG 2024 values for the semi-muonic branching fractions:
- **Species Fragmentation Fractions ($f_i$)**:
  - $f(D^0) \approx 0.58 \pm 0.03$
  - $f(D^+) \approx 0.23 \pm 0.02$
  - $f(D_s^+) \approx 0.10 \pm 0.02$
  - $f(\Lambda_c^+) \approx 0.09 \pm 0.02$
- **Inclusive Semi-Muonic Branching Ratios ($\mathcal{B}_i$)**:
  - $\mathcal{B}(D^0 \to \mu^+ X) = (6.8 \pm 0.6)\%$
  - $\mathcal{B}(D^+ \to \mu^+ X) = (15.6 \pm 0.8)\%$
  - $\mathcal{B}(D_s^+ \to \mu^+ X) = (6.3 \pm 0.7)\%$
  - $\mathcal{B}(\Lambda_c^+ \to \mu^+ X) = (3.5 \pm 0.6)\%$

Evaluating the weighted sum:
$$\begin{aligned}
\langle B_\mu \rangle &= (0.58 \times 0.068) + (0.23 \times 0.156) + (0.10 \times 0.063) + (0.09 \times 0.035) \\
&= 0.03944 + 0.03588 + 0.00630 + 0.00315 \\
&= \mathbf{0.0848 \pm 0.0050} \quad (\sim 8.48\%)
\end{aligned}$$

The relative uncertainty on $\langle B_\mu \rangle$ is:
$$\frac{\sigma(\langle B_\mu \rangle)}{\langle B_\mu \rangle} \approx \frac{0.0050}{0.0848} \approx 5.9\%$$

### 9.4 Asymptotic Dimuon Ratios $R_{2\mu, \infty}$
Multiplying the asymptotic inclusive charm ratios $R_\infty$ by $\langle B_\mu \rangle = 0.0848$:

| Category | Target | Inclusive $R_\infty$ | Effective $\langle B_\mu \rangle$ | Dimuon Asymptotic $R_{2\mu, \infty}$ |
| :--- | :--- | :--- | :--- | :--- |
| **Neutrino ($\nu_\mu$)** | Tungsten ($^{184}_{74}\text{W}$) | $10.84\%$ | $8.48\%$ | $\mathbf{0.919\%}$ ($\approx 0.92\%$) |
| **Neutrino ($\nu_\mu$)** | Isoscalar ($N=Z$) | $11.16\%$ | $8.48\%$ | $\mathbf{0.946\%}$ ($\approx 0.95\%$) |
| **Antineutrino ($\bar{\nu}_\mu$)** | Tungsten ($^{184}_{74}\text{W}$) | $15.39\%$ | $8.48\%$ | $\mathbf{1.305\%}$ ($\approx 1.31\%$) |
| **Antineutrino ($\bar{\nu}_\mu$)** | Isoscalar ($N=Z$) | $14.82\%$ | $8.48\%$ | $\mathbf{1.257\%}$ ($\approx 1.26\%$) |
| **Inclusive ($\nu_\mu + \bar{\nu}_\mu$)** | Tungsten ($^{184}_{74}\text{W}$) | $11.73\%$ | $8.48\%$ | $\mathbf{0.995\%}$ ($\approx 0.99\%$) |
| **Inclusive ($\nu_\mu + \bar{\nu}_\mu$)** | Isoscalar ($N=Z$) | $12.05\%$ | $8.48\%$ | $\mathbf{1.022\%}$ ($\approx 1.02\%$) |

### 9.5 Error Propagation
When predicting the dimuon ratio with the slow rescaling model, the parametric uncertainty on $\langle B_\mu \rangle$ ($5.9\%$) is folded in quadrature with the base PDF/CKM normalization uncertainty ($8.0\%$):
$$\sigma_{\text{base}, 2\mu} = \sqrt{(0.08)^2 + (0.059)^2} \cdot R_{2\mu, \infty} \approx 9.9\% \cdot R_{2\mu, \infty}$$

The two-tier uncertainty belts are generated via:
1. **Inner Belt**: Base kinematics ($m_c$, $E_{\text{thresh}}$) + combined base normalization ($\pm 9.9\%$).
2. **Outer Belt**: High-energy asymmetric expansion accounting for boson-gluon fusion ($W^+ g \to c\bar{s}$) and strange sea growth ($g \to s\bar{s}$) under DGLAP evolution.

### 9.6 Diagnostic Canvases & Historical Dimuon Measurements
Three companion canvases are produced in [`charm_diagnostic.py`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/scripts/diagnostics/charm_diagnostic.py):
1. `c_charm_dimuon_rate_vs_energy`: Inclusive ($\nu_\mu + \bar{\nu}_\mu$) dimuon production rate on Tungsten, compared against MC inclusive simulation with Clopper-Pearson error bars, the two-tier uncertainty belt, the dashed Isoscalar reference line, and historical experimental measurements.
2. `c_charm_dimuon_rate_vs_energy_nu_only`: Pure $\nu_\mu$ CC dimuon rate on Tungsten, compared against MC $\nu_\mu$ simulation, theoretical bands, and historical measurements.
3. `c_charm_dimuon_rate_vs_energy_antinu_only`: Pure $\bar{\nu}_\mu$ CC dimuon rate on Tungsten.

To anchor the dimuon rate against historical neutrino experiments without cluttering the SND@LHC MC points, published dimuon cross section ratios $\sigma(2\mu)/\sigma(\text{CC})$ vs. $E_\nu$ from **NOMAD**, **CHORUS**, **CCFR**, and **E53A+E53B** (compiled in [`neutrino_charm_dimuon_ratio_vs_enu.csv`](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/scripts/diagnostics/measurements/neutrino_charm_dimuon_ratio_vs_enu.csv)) are overlaid with prominent black open markers:
- **NOMAD (2013)**: Open diamonds (`ROOT.kBlack`, marker 27, size 1.25, linewidth 2, 19 points)
- **CHORUS (2011)**: Open squares (`ROOT.kBlack`, marker 25, size 1.10, linewidth 2, 7 points)
- **CCFR (1995)**: Open circles (`ROOT.kBlack`, marker 24, size 1.10, linewidth 2, 8 points)
- **E53A+E53B (1983)**: Open triangles (`ROOT.kBlack`, marker 26, size 1.10, linewidth 2, 4 points)

Simulated MC error graphs are styled to match the color of their respective slow rescaling model curves:
- **Inclusive ($\nu_\mu + \bar{\nu}_\mu$)**: Red (`ROOT.kRed + 1`, marker 20)
- **Neutrino ($\nu_\mu$)**: Blue (`ROOT.kBlue + 2`, marker 20)
- **Antineutrino ($\bar{\nu}_\mu$)**: Purple/Magenta (`ROOT.kMagenta + 2`, marker 20)

All individual graphs (`g_dimuon_nomad`, `g_dimuon_chorus`, `g_dimuon_ccfr`, `g_dimuon_e53a_e53b`) and a combined graph (`g_dimuon_all_measurements`) are saved in the `Summary/` directory in the output ROOT file.


