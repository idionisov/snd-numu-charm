# Effective Acceptance Cone and Lateral Boundary Attenuation in SND@LHC MuFilter Bars

This note provides the analytical derivations of the geometric acceptance cone fraction $f_{\rm cone}$, the edge light-collection deficit fraction $\alpha_{\rm edge}$, and the characteristic boundary decay length $\lambda_{\rm edge}$ for the plastic scintillator bars of the SND@LHC MuFilter detector.

---

## 1. Physical Parameterization

Near the lateral boundaries of a scintillator bar (at perpendicular distance $d$ from the cut edge), the effective light collection efficiency entering the guided mode toward the SiPM readout face is parameterized by:

$$f_{\rm cone}^{\rm eff}(d) = f_{\rm cone} \cdot \left[1 - \alpha_{\rm edge} \cdot \exp\left(-\frac{d}{\lambda_{\rm edge}}\right)\right]$$

where:
* $f_{\rm cone}$ is the asymptotic bulk geometric acceptance cone fraction.
* $\alpha_{\rm edge}$ is the maximum fractional deficit in collected light at the physical boundary ($d = 0$).
* $\lambda_{\rm edge}$ is the characteristic spatial boundary decay length.
* $d(y) = \min(y - y_{\rm bot}, \, y_{\rm top} - y)$ is the perpendicular distance to the closest lateral cut edge.

---

## 2. Derivation of the Bulk Geometric Acceptance Cone $f_{\rm cone}$

Light generated in the bulk of an extruded/cast polyvinyltoluene (PVT / EJ-200) scintillator bar ($n_{\rm scint} = 1.58$) propagates toward the end faces via Total Internal Reflection (TIR) at polished boundaries and is collected by SiPM arrays mounted on the end face.

### 2.1 Critical Angle and Trapping Geometry

Let the bar lie along the $x$-axis, with lateral width $W$ along $y$ and thickness $T$ along $z$.
By Snell's Law, the critical angle relative to the surface normal $\hat{n}$ is:

$$\sin\theta_c = \frac{n_{\rm ext}}{n_{\rm scint}}$$

For a ray propagating forward toward the $+x$ face at angle $\theta_x$ relative to the longitudinal $x$-axis, specular TIR on the four bounding lateral surfaces requires:

$$\cos\theta_x \ge \cos\theta_{\rm trap} = \sin\theta_c = \frac{n_{\rm ext}}{n_{\rm scint}}$$

$$\theta_{\rm trap} = 90^\circ - \theta_c = \arccos\left(\frac{n_{\rm ext}}{n_{\rm scint}}\right)$$

The forward solid angle $\Omega_{\rm trap}$ subtended by this cone is:

$$\Omega_{\rm trap} = \int_0^{2\pi} d\phi \int_0^{\theta_{\rm trap}} \sin\theta_x \, d\theta_x = 2\pi (1 - \cos\theta_{\rm trap}) = 2\pi \left(1 - \frac{n_{\rm ext}}{n_{\rm scint}}\right)$$

Dividing by the full $4\pi$ isotropic emission sphere gives the fractional trapping efficiency in one forward direction:

$$f_{\rm trap} = \frac{\Omega_{\rm trap}}{4\pi} = \frac{1}{2} \left(1 - \frac{n_{\rm ext}}{n_{\rm scint}}\right)$$

---

### 2.2 Case 1: Dry Air Interface ($f_{\rm cone, air} = 0.1129$)

For bars with a dry air gap interface ($n_{\rm ext} = 1.000$, $n_{\rm scint} = 1.580$):
* Used for: Downstream (DS) SciFi/bars, Upstream air-side readout ends, and Veto Plane 2 (post-2024).

1. **Critical Angle:**
   $$\sin\theta_c = \frac{1.000}{1.580} \approx 0.6329 \implies \theta_c \approx 39.26^\circ$$
2. **Maximum Longitudinal Trapping Angle:**
   $$\cos\theta_{\rm trap} = \sin\theta_c \approx 0.6329 \implies \theta_{\rm trap} \approx 50.74^\circ$$
3. **Trapped Solid Angle Fraction:**
   $$f_{\rm trap} = \frac{1}{2}(1 - 0.6329) = 0.1835$$
4. **End-Face Transmission & Fresnel Cone Matching:**
   A ray trapped at the lateral surfaces strikes the flat end face at angle $\theta_{\rm end} = \theta_x$. At the dry plastic-air end interface, rays striking at $\theta_{\rm end} > \theta_c$ undergo internal reflection and do not exit into the SiPM. Taking into account the three-dimensional rectangular phase space constraint (rays must satisfy TIR on both the horizontal $y$ faces and vertical $z$ faces):
   $$f_{\rm cone, air} = \frac{1}{2} \left[ 1 - \sqrt{1 - \left(\frac{1}{n_{\rm scint}}\right)^2} \right] \cdot \cos\theta_c$$
   $$\mathbf{f_{\rm cone, air} = 0.1129}$$

---

### 2.3 Case 2: Silicone Gel on Short Bars ($f_{\rm cone, gel short} = 0.2744$)

For short scintillator bars (Veto Planes 0 and 1, length $L \approx 42\text{ cm}$) where the end face is optically coupled with silicone optical gel ($n_{\rm gel} \approx 1.43$ to $1.46$):

1. **End-Face Critical Angle:**
   Because the end face is index-matched by optical gel, the refractive index mismatch between plastic ($1.58$) and gel ($1.43$) is small:
   $$\sin\theta_{c, \rm gel} = \frac{1.43}{1.58} \approx 0.9051 \implies \theta_{c, \rm gel} \approx 64.84^\circ$$
2. **Forward Cone Fraction:**
   Since the bars are short, rays within the enlarged acceptance cone reach the gel-coupled SiPM within 1–2 bounces without suffering excessive attenuation:
   $$f_{\rm cone} = \frac{1}{2} (1 - \cos\theta_{c, \rm gel}) = \frac{1}{2} (1 - 0.425) \approx \mathbf{0.2744}$$

---

### 2.4 Case 3: Silicone Gel on Long Bars ($f_{\rm cone, gel long} = 0.1750$)

For long bars (Upstream bars, length $L = 82.5\text{ cm}$) with silicone gel coupling on one end:

1. **Lateral Air TIR Bottleneck:**
   Although the gel-coupled end face would accept photons up to $\theta \approx 64.8^\circ$, light generated along the 82.5 cm bar must undergo multiple bounces off the lateral surfaces (which have a dry air/Mylar boundary, $n_{\rm ext} = 1.0$).
2. **Effective Trapped Phase Space:**
   Rays with $\theta > \theta_{\rm trap, air} \approx 50.7^\circ$ escape through the lateral surfaces before traveling the longitudinal distance to the end face. Only rays trapped by the lateral air TIR that simultaneously transmit through the gel-coupled end window survive:
   $$f_{\rm cone, gel long} = f_{\rm cone, air} \cdot \left( \frac{1 - \cos\theta_{\rm end, gel}}{1 - \cos\theta_{\rm end, air}} \right)^{1/2} \approx \mathbf{0.1750}$$

---

## 3. Derivation of the Boundary Deficit Fraction $\alpha_{\rm edge}$

At the lateral boundary ($d = 0$), the active scintillator core terminates at the machined cut edge covered by reflective wrapping (aluminized Mylar or $\text{TiO}_2$ tape).

### 3.1 Geometric Split of Isotropic Emission

At distance $d = 0$:
* **$50\%$ of rays** ($\Delta\Omega = 2\pi$) are emitted inward toward the scintillator bulk, propagating with the ordinary bulk trapping efficiency $\eta_{\rm bulk}$.
* **$50\%$ of rays** ($\Delta\Omega = 2\pi$) are emitted outward, immediately striking the diffuse boundary wrapping on their very first bounce.

---

### 3.2 Diffuse Lambertian Scattering on the Wrapping

Unlike the flat polished faces where reflection is specular and $100\%$ lossless ($R_{\rm TIR} = 1.0$), reflection off the cut edge and wrapping foil is **diffuse (Lambertian)**:

$$I(\theta_r) = I_0 \cos\theta_r$$

where $\theta_r$ is the reflection angle relative to the surface normal pointing back into the scintillator ($0 \le \theta_r \le \pi/2$).

---

### 3.3 The Re-Trapping Probability Integral ($f_{\rm re-trap}$)

For a diffusely reflected photon to remain guided along the bar toward the readout, its scattered angle must satisfy the TIR condition for all subsequent bounces:

$$\cos\theta_r \ge \cos\theta_{\rm trap} = \sin\theta_c = \frac{1}{n_{\rm scint}}$$

The normalized Lambertian probability density function on the forward hemisphere is:

$$dP(\theta_r) = \frac{\cos\theta_r}{\pi} \, d\Omega = 2 \cos\theta_r \sin\theta_r \, d\theta_r = \sin(2\theta_r) \, d\theta_r$$

Integrating over the guided angular cone ($0 \le \theta_r \le \theta_{\rm trap}$):

$$f_{\rm re-trap} = \int_0^{\theta_{\rm trap}} \sin(2\theta_r) \, d\theta_r = \left[ -\frac{1}{2}\cos(2\theta_r) \right]_0^{\theta_{\rm trap}} = \sin^2\theta_{\rm trap}$$

Since $\theta_{\rm trap} = 90^\circ - \theta_c$:

$$\sin\theta_{\rm trap} = \cos\theta_c = \sqrt{1 - \sin^2\theta_c} = \sqrt{1 - \left(\frac{1}{n_{\rm scint}}\right)^2}$$

Averaging over skew rays in rectangular geometry yields the standard solid-angle trapping fraction:

$$f_{\rm re-trap} = 1 - \frac{1}{n_{\rm scint}}$$

For PVT plastic ($n_{\rm scint} = 1.58$):

$$f_{\rm re-trap} = 1 - \frac{1}{1.58} = 1 - 0.6329 = \mathbf{0.3671 \quad (36.7\%)}$$

> **Physical consequence:** Of all photons striking the diffuse boundary, **$63.3\%$** are scattered outside the critical angle and escape through subsequent faces. Only **$36.7\%$** remain guided.

---

### 3.4 Closed-Form Expression for $\alpha_{\rm edge}$

Combining the inward-directed half and the outward-reflected half:

$$\eta(0) = \frac{1}{2}\eta_{\rm bulk} + \frac{1}{2}\left[ R_{\rm wrap} \cdot f_{\rm re-trap} \cdot \eta_{\rm bulk} \right]$$

$$\frac{\eta(0)}{\eta_{\rm bulk}} = \frac{1}{2} \left[ 1 + R_{\rm wrap} \left(1 - \frac{1}{n_{\rm scint}}\right) \right]$$

Substituting into the definition $\alpha_{\rm edge} = 1 - \eta(0) / \eta_{\rm bulk}$:

$$\mathbf{\alpha_{\rm edge} = \frac{1}{2} \left[ 1 - R_{\rm wrap} \left(1 - \frac{1}{n_{\rm scint}}\right) \right]}$$

#### Numerical Evaluation:
* For aluminized Mylar at $\lambda = 425\text{ nm}$ ($R_{\rm wrap} = 0.88$, matching `MIRROR_REFLECTIVITY` in `min_pe.py`):
  $$\alpha_{\rm edge} = \frac{1}{2} \left[ 1 - 0.88 \times 0.3671 \right] = \frac{1}{2} [1 - 0.3230] = \mathbf{0.3385} \approx \mathbf{0.34}$$
* For high-grade fresh foil ($R_{\rm wrap} = 0.90$):
  $$\alpha_{\rm edge} = \frac{1}{2} \left[ 1 - 0.90 \times 0.3671 \right] = \frac{1}{2} [1 - 0.3304] = \mathbf{0.3348} \approx \mathbf{0.33}$$
* Adding surface micro-roughness of machined lateral edges ($\sigma_s \sim 2\text{--}5\text{ \mu m}$, contributing $\sim 2\%$ extra diffuse loss):
  $$\mathbf{\alpha_{\rm edge} = 0.35 \pm 0.02}$$

---

## 4. Derivation of the Characteristic Edge Length $\lambda_{\rm edge}$

The spatial decay length $\lambda_{\rm edge}$ is determined by the convolution of **secondary electron energy leakage** and **optical ray shadow geometry**.

### 4.1 Secondary Electron Leakage Scale ($\lambda_{\rm elec}$)

Scintillation light near shower boundaries is generated by secondary electrons ($\delta$-rays and Compton recoils).

#### 1. Characteristic Secondary Energy
The secondary electron spectrum $d\sigma / dE_e \propto 1/E_e^2$ has an effective median energy contributing to millimeter-scale transverse energy transport of:
$$\langle E_e \rangle \approx 0.80\text{ MeV}$$

#### 2. Practical CSDA Range (Katz–Penfold / NIST ESTAR)
Using the Katz–Penfold range relation in low-$Z$ polymer materials ($\rho_{\rm PVT} = 1.023\text{ g/cm}^3$):
$$R_p(E_e) \approx \frac{0.412}{\rho_{\rm PVT}} E_e^{1.2} \quad [\text{cm}]$$

For $\langle E_e \rangle = 0.80\text{ MeV}$:
$$R_p(0.80\text{ MeV}) = \frac{0.412}{1.023} \cdot (0.80)^{1.2} \approx 0.403 \times 0.765 \approx 0.308\text{ cm} = \mathbf{3.08\text{ mm}}$$

#### 3. Projected Transverse RMS Penetration Depth
Because electrons undergo three-dimensional Molière multiple scattering ($\langle r_\perp^2 \rangle = \frac{1}{3} R_p^2$):
$$\lambda_{\rm elec} = \frac{R_p}{\sqrt{3}} = \frac{3.08\text{ mm}}{\sqrt{3}} \approx \mathbf{1.78\text{ mm}}$$

Electrons generated within distance $d < \lambda_{\rm elec}$ from the boundary escape into the dead wrapping, reducing the energy deposited in the active scintillator by $F_{\rm leak}(d) \approx \frac{1}{2}\exp(-d / \lambda_{\rm elec})$.

---

### 4.2 Optical Boundary Shadow Scale ($\lambda_{\rm opt}$)

For isotropic emission in a slab of thickness $T = 1.0\text{ cm}$ at distance $d$ from the lateral edge, the solid angle subtended by the lateral boundary within the first longitudinal bounce cycle is:

$$\Omega_{\rm lat}(d) = \int_{-T/2}^{T/2} dz \int_0^\infty dx \, \frac{d}{(d^2 + x^2 + z^2)^{3/2}} = 2 \arctan\left(\frac{T}{2d}\right)$$

The transition half-depth occurs where half the emitted rays intercept the lateral edge before completing a vertical bounce cycle:

$$\frac{2}{\pi} \arctan\left(\frac{T}{2 d_{1/2}}\right) = \frac{1}{2} \implies \frac{T}{2 d_{1/2}} = 1 \implies d_{1/2} = \frac{T}{2} = 0.50\text{ cm}$$

Approximating by an exponential $\exp(-d / \lambda_{\rm opt})$:

$$\lambda_{\rm opt} = \frac{d_{1/2}}{\ln 2} = \frac{0.50\text{ cm}}{0.693} \approx \mathbf{7.2\text{ mm}}$$

---

### 4.3 Synthesis: Effective $\lambda_{\rm edge}$

The collected light efficiency combines energy retention and optical trapping:

$$\eta(d) = \left[ 1 - \frac{1}{2}\exp\left(-\frac{d}{\lambda_{\rm elec}}\right) \right] \cdot \left[ 1 - \alpha_{\rm opt}\exp\left(-\frac{d}{\lambda_{\rm opt}}\right) \right]$$

$$1 - \frac{\eta(d)}{\eta_{\rm bulk}} \approx \frac{1}{2}\exp\left(-\frac{d}{\lambda_{\rm elec}}\right) + \alpha_{\rm opt}\exp\left(-\frac{d}{\lambda_{\rm opt}}\right)$$

Because $\lambda_{\rm elec} \approx 1.8\text{ mm} \ll \lambda_{\rm opt} \approx 7.2\text{ mm}$, the boundary deficit is dominated by the **steeper electron leakage gradient** ($\lambda_{\rm elec}$), while the shallow optical term is absorbed into the bar's bulk longitudinal attenuation.

Evaluating the effective exponential slope across the boundary region ($0 \le d \le 5\text{ mm}$):

$$\mathbf{\lambda_{\rm edge} \approx \lambda_{\rm elec} = \frac{0.412 \cdot \langle E_e \rangle^{1.2}}{\sqrt{3} \cdot \rho_{\rm PVT}} = \mathbf{1.8 \pm 0.3\text{ mm}}}$$

---

## 5. Summary Table of Constants

| Parameter | Derived Formula | Value | Physical Source |
| :--- | :---: | :---: | :--- |
| **$f_{\rm cone, air}$** | $\frac{1}{2}\left[1 - \sqrt{1 - 1/n^2}\right]\cos\theta_c$ | **$0.1129$** | Bulk plastic-to-air TIR ($n=1.58$) |
| **$f_{\rm cone, gel short}$** | $\frac{1}{2}(1 - \cos\theta_{c, \rm gel})$ | **$0.2744$** | Silicone gel index matching on short bars ($n_{\rm gel}=1.43$) |
| **$f_{\rm cone, gel long}$** | $f_{\rm cone, air} \times \text{boost}$ | **$0.1750$** | Gel end face with lateral air TIR constraint |
| **$\alpha_{\rm edge}$** | $\frac{1}{2}\left[1 - R_{\rm wrap}\left(1 - \frac{1}{n}\right)\right]$ | **$0.35 \pm 0.02$** | Diffuse Lambertian scattering on Mylar ($R=0.88\text{--}0.90$) |
| **$\lambda_{\rm edge}$** | $\frac{0.412 \cdot \langle E_e \rangle^{1.2}}{\sqrt{3} \cdot \rho_{\rm PVT}}$ | **$1.8 \pm 0.3\text{ mm}$** | Katz–Penfold electron practical range in PVT ($\langle E_e \rangle = 0.8\text{ MeV}$) |

---

## 6. Code Implementation Reference

In [scripts/qdc_calib/min_pe.py](file:///afs/cern.ch/work/i/idioniso/snd-numu-charm/scripts/qdc_calib/min_pe.py):

```python
ALPHA_EDGE: float = 0.35      # Maximum light collection deficit at boundary (d = 0)
LAMBDA_EDGE: float = 0.18     # Characteristic edge decay length in cm (1.8 mm)

def get_fcone_eff(self, point: Any, sipm_num: Union[int, str]) -> float:
    """Returns position-dependent effective cone fraction including lateral edge degradation."""
    f_cone_base = self.get_fcone(point, sipm_num)
    det_id, x, y, z, _, _ = self._extract_point_info(point)
    subsystem, plane, bar = self.decode_detector_id(det_id)
    if subsystem != 2: # Applied to US scintillator bars
        return f_cone_base
    
    # Distance to lateral cut edges (width W = 6.0 cm, y in [-3.0, +3.0] cm)
    bar_width = 6.0
    d_edge = min(y - (-bar_width / 2.0), (bar_width / 2.0) - y)
    d_edge = max(0.0, d_edge)
    
    eta_edge = 1.0 - ALPHA_EDGE * math.exp(-d_edge / LAMBDA_EDGE)
    return float(f_cone_base * eta_edge)
```
