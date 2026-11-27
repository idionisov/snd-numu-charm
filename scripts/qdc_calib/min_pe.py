"""
MuFilter Minimum Photoelectron & Optical Transport Calculation
-------------------------------------------------------------
Functions and encapsulation class to evaluate optical transmission factors,
digitized photoelectron yields, TOFPET2C front-end ASIC discriminator triggers,
and effective bar turn-on thresholds for SND@LHC MuFilter and Veto bars:

- get_fcone: Fraction of light entering the acceptance cone (gel vs dry air-gap)
- get_Satt: Distance-dependent light attenuation with optional mirror reflection
- get_ffresnel: Fresnel transmission coefficient at SiPM optical interface
- get_farea: Geometric area coverage fraction (SiPM area / Bar end-cap area)
- get_sipm_pe: Photoelectrons from a single MCPoint in a single SiPM
- group_mcpoints_by_bar: Digitization clustering matching official DigiTaskSND
- get_sipm_pes: Returns an array of photoelectrons for each SiPM on a bar
- tofpet2_lsb_mv: TOFPET2C discriminator DAC step size (mV / DAC count)
- eval_sipm_triggers: Evaluates T1/T2 discriminator crossings & hit qualification
- get_lowest_trigger_threshold: Determines the minimum energy and p.e. to trigger a bar
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Scintillator and optical constants
RISE_TIME_NS: float = 0.9            # EJ-200 scintillation rise time [ns]
DECAY_TIME_NS: float = 2.1           # EJ-200 scintillation decay time [ns]
PROP_SPEED_CM_NS: float = 13.9       # Effective light propagation speed along bar [cm/ns] (c_scint = 13.88 +- 0.57 cm/ns)

# Effective peak current reduction factor for collective scintillation pulse
# (convolution of 0.9 ns rise + 2.1 ns decay emission with TIA front-end shaping)
# relative to an instantaneous impulse:
PEAK_PULSE_FACTOR: float = 0.4530    # 1.0 / 2.207

# Analytical Edge Degradation and Skin Parameters (from notes/light_att/f_cone_eff.md)
ALPHA_EDGE: float = 0.35      # Maximum light collection deficit at lateral cut edge
LAMBDA_EDGE: float = 0.18     # Characteristic spatial boundary decay length in cm (1.8 mm)
DELTA_DEAD: float = 0.05      # Dead wrapping / skin thickness in cm (0.5 mm)

# Temporal Arrival Dispersion Parameters
TAU_PULSE_NS: float = 2.8     # Intrinsic pulse duration scale [ns] (0.9ns rise + 2.1ns decay + TIA response)

F_CONE_AIR: float = 0.1129          # TIR escape cone for dry air-gap coupling
F_CONE_GEL_SHORT: float = 0.2744    # Direct critical angle cone for short bars (Veto 0&1, 42 cm)
F_CONE_GEL_LONG: float = 0.1750     # Trapped guided light limited by lateral air TIR (US bars, 82.5 cm)
F_CONE_GEL: float = 0.1750          # Default gel factor for standard long bars

F_FRESNEL_GEL: float = 0.9958
F_FRESNEL_AIR: float = 0.9115

ATTENUATION_COEFF: float = 0.00426  # cm^-1
MIRROR_REFLECTIVITY: float = 0.88

SIPM_AREA: float = 0.36       # cm^2 (6 mm x 6 mm)
LIGHT_YIELD: float = 10000.0  # photons / MeV in EJ-200 plastic scintillator
SIPM_PDE: float = 0.50        # SiPM photon detection efficiency (50%)
KB_BIRKS: float = 0.0126      # cm / MeV (Birks' quenching parameter for EJ-200 PVT scintillator)

# Default DAQ frontend parameters
DEFAULT_G_T_OHMS: float = 3000.0       # Front-end TIA gain g_t [Ohms] (3.0 mV / uA)
DEFAULT_I_1PE_UA: float = 5.0          # 6x6 mm^2 SiPM peak single-p.e. current [uA]
DEFAULT_THR_T1: int = 20               # T1 discriminator threshold [DAC counts]
DEFAULT_THR_T2: int = 20               # T2 discriminator threshold [DAC counts]
DEFAULT_LSB_THR_T1: int = 59           # T1 LSB range register (lsb_thr_t1 in board.toml)
DEFAULT_LSB_THR_T2: int = 50           # T2 LSB range register (lsb_thr_t2 in board.toml)

# Bar nominal dimensions [cm]: (Length, Width, Thickness)
# End-cap area = Width * Thickness
BAR_DIMS: Dict[str, Tuple[float, float, float]] = {
    "veto_hor": (42.0, 6.0, 1.0),       # Planes 0 & 1: 42 cm long, 6 cm wide, 1 cm thick (End area = 6.0 cm^2)
    "veto_ver": (46.0, 6.0, 1.0),       # Plane 2 (3rd plane, post-2024): 46 cm long, 6 cm wide, 1 cm thick (End area = 6.0 cm^2)
    "us_hor": (82.525, 6.0, 1.0),       # End area = 6.0 cm^2
    "ds_hor": (82.525, 1.0, 1.0),       # End area = 1.0 cm^2
    "ds_ver": (63.525, 1.0, 1.0),       # End area = 1.0 cm^2
}


def tofpet2_lsb_mv(lsb_reg: int, branch: str = "t1") -> float:
    """
    TOFPET2C discriminator DAC step size (mV / DAC count).
    Lower register value -> coarser mV/DAC step.
    """
    if branch in ("t1", "t2"):
        regs = np.array([48, 50, 55, 57, 60, 62], dtype=float)
        mvs = np.array([9.2, 8.5, 6.61, 5.81, 3.23, 1.13], dtype=float)
    elif branch == "e":
        regs = np.array([40, 50, 53, 55, 57, 60], dtype=float)
        mvs = np.array([12.5, 9.5, 8.39, 7.58, 6.29, 3.23], dtype=float)
    else:
        raise ValueError(f"Unknown branch '{branch}'. Expected 't1', 't2', or 'e'.")
    return float(np.interp(lsb_reg, regs, mvs))


def _pulse_kernel(
    dt: np.ndarray,
    decay_ns: float = DECAY_TIME_NS,
    rise_ns: float = 0.0,
) -> np.ndarray:
    """Normalized causal pulse waveform peaking at 1.0 for dt >= 0."""
    out = np.zeros_like(dt, dtype=np.float64)
    mask = dt >= 0.0
    if not np.any(mask):
        return out
    t = dt[mask]
    if rise_ns <= 0.0:
        out[mask] = np.exp(-t / decay_ns)
    else:
        if abs(decay_ns - rise_ns) < 1e-6:
            rise_ns = decay_ns * 0.999
        raw = np.exp(-t / decay_ns) - np.exp(-t / rise_ns)
        t_peak = math.log(decay_ns / rise_ns) / (1.0 / rise_ns - 1.0 / decay_ns)
        peak_val = max(math.exp(-t_peak / decay_ns) - math.exp(-t_peak / rise_ns), 1e-12)
        out[mask] = np.clip(raw / peak_val, 0.0, None)
    return out


class MinPE:
    """
    MuFilter Optical Transmission and Minimum Photoelectron Model.

    Encapsulates geometry resolution, silicone gel coupling mapping,
    optical attenuation, and MCPoint-to-hit digitization clustering.
    """

    def __init__(
        self,
        geo_interface: Optional[Any] = None,
        year: int = 2024,
        is_testbeam: bool = False,
        include_rise_time: bool = True,
        include_birks: bool = True,
        include_edge_effect: bool = True,
        include_dispersion: bool = True,
        global_reduction_factor: float = 1.0,
    ) -> None:
        """
        Initialize with optional SndlhcGeo interface or ROOT MuFilter detector.
        If None, uses nominal geometry coordinate transformations.
        `year` parameter sets configuration: Veto plane 2 (third veto plane) is only active for year >= 2024.
        `is_testbeam`: If True, uses CERN SPS H8 2023 testbeam geometry, gel mapping, and reduction factors.
        `include_rise_time`: If True, accounts for EJ-200 scintillation rise time (0.9 ns) pulse attenuation (PEAK_PULSE_FACTOR = 0.4530).
        `include_birks`: If True, applies Birks' quenching law f_Birks = 1 / (1 + k_B * dE/dx).
        `include_edge_effect`: If True, applies analytical lateral boundary attenuation and dead wrapping skin.
        `include_dispersion`: If True, accounts for photon arrival time dispersion along the bar degrading peak current.
        `global_reduction_factor`: Optional global scaling factor (e.g. for radiative shower calibration).
        """
        self._geo = geo_interface
        self._mufi = None
        self.year = int(year)
        self.is_testbeam = bool(is_testbeam)
        self.include_rise_time = bool(include_rise_time)
        self.include_birks = bool(include_birks)
        self.include_edge_effect = bool(include_edge_effect)
        self.include_dispersion = bool(include_dispersion)
        self.global_reduction_factor = float(global_reduction_factor)
        self._k_factors: Dict[Tuple[int, int], float] = {}
        if self._geo is not None:
            if hasattr(self._geo, "modules") and "MuFilter" in self._geo.modules:
                self._mufi = self._geo.modules["MuFilter"]
            elif hasattr(self._geo, "GetPosition"):
                self._mufi = self._geo

        # Automatically load calibrated channel-by-channel photoelectron reduction parameters
        tb_csv = "config/sipm_photoelectron_reduction_factors_testbeam.csv"
        nom_csv = "config/sipm_photoelectron_reduction_factors.csv"
        csv_to_load = tb_csv if self.is_testbeam else nom_csv
        self.load_reduction_factors(csv_to_load)

    def is_small_sipm(self, det_id: int, sipm_num: Union[int, str]) -> bool:
        """
        Check if SiPM is a small 1x1 mm^2 SiPM on Upstream bars.
        Channels with (sipm_num % 8) in {2, 5} on subsystem 2 are small SiPMs.
        These are uncalibrated and must never be used for triggering or hit qualification.
        """
        subsystem, _, _ = self.decode_detector_id(det_id)
        if subsystem != 2:
            return False
        return (int(sipm_num) % 8) in (2, 5)

    def load_reduction_factors(self, csv_path: str = "config/sipm_photoelectron_reduction_factors.csv") -> None:
        """Load channel-by-channel semi-empirical photoelectron reduction parameter k (0 <= k <= 1)."""
        if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            alt_path = os.path.join(base, csv_path)
            if os.path.exists(alt_path):
                csv_path = alt_path

        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 3 and parts[0] != "detector_id":
                        try:
                            did, sipm_num, k = int(parts[0]), int(parts[1]), float(parts[2])
                            self._k_factors[(did, sipm_num)] = k
                        except ValueError:
                            continue

    def get_reduction_factor(self, det_id: int, sipm_num: Union[int, str]) -> float:
        """Return reduction parameter k for (det_id, sipm_num), scaled by global_reduction_factor. Small SiPMs return 0.0."""
        if self.is_small_sipm(det_id, sipm_num):
            return 0.0
        glob = getattr(self, "global_reduction_factor", 1.0)
        try:
            s_idx = int(sipm_num)
            return float(self._k_factors.get((det_id, s_idx), 1.0) * glob)
        except ValueError:
            return float(glob)

    def is_valid_det_id(self, det_id: int) -> bool:
        """
        Check if detector ID is valid for current year configuration.
        Veto plane 2 (3rd plane, det_id 12000-12006) is only present for year >= 2024.
        """
        subsystem, plane, _ = self.decode_detector_id(det_id)
        if subsystem == 1 and plane >= 2 and self.year < 2024:
            return False
        return True

    def _extract_point_info(self, point: Any) -> Tuple[int, float, float, float, float, float]:
        """
        Extract detID, (x, y, z) [cm], time [ns], and energy loss [MeV] from MuFilterPoint.
        Converts GeV to MeV if energy loss is in Geant4 units (eloss < 0.1 GeV).
        """
        if hasattr(point, "GetDetectorID"):
            det_id = int(point.GetDetectorID())
            x = float(point.GetX())
            y = float(point.GetY())
            z = float(point.GetZ())
            time_ns = float(point.GetTime()) if hasattr(point, "GetTime") else 0.0
            eloss_raw = float(point.GetEnergyLoss()) if hasattr(point, "GetEnergyLoss") else 0.0
            eloss_mev = (eloss_raw * 1000.0) if (0.0 < eloss_raw < 0.1) else eloss_raw
            return det_id, x, y, z, time_ns, eloss_mev
        elif isinstance(point, dict):
            det_id = int(point.get("detID", point.get("detector_id", 0)))
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            z = float(point.get("z", 0.0))
            time_ns = float(point.get("time", point.get("t", 0.0)))
            eloss_raw = float(point.get("eloss", point.get("eLoss", point.get("energy_loss", 1.0))))
            eloss_mev = (eloss_raw * 1000.0) if (0.0 < eloss_raw < 0.1) else eloss_raw
            return det_id, x, y, z, time_ns, eloss_mev
        elif isinstance(point, (tuple, list)):
            det_id = int(point[0])
            x = float(point[1])
            y = float(point[2])
            z = float(point[3])
            time_ns = float(point[4]) if len(point) > 4 else 0.0
            eloss_raw = float(point[5]) if len(point) > 5 else 1.0
            eloss_mev = (eloss_raw * 1000.0) if (0.0 < eloss_raw < 0.1) else eloss_raw
            return det_id, x, y, z, time_ns, eloss_mev
        raise ValueError(f"Cannot extract MuFilterPoint info from: {type(point)}")

    def decode_detector_id(self, det_id: int) -> Tuple[int, int, int]:
        """
        Decode detector ID into (subsystem, plane, bar).
        Subsystems: 1=Veto, 2=Upstream (US), 3=Downstream (DS).
        """
        subsystem = det_id // 10000
        plane = (det_id // 1000) % 10
        bar = det_id % 1000
        return subsystem, plane, bar

    def is_vertical(self, det_id: int) -> bool:
        """Return True if the bar is oriented vertically."""
        subsystem, plane, bar = self.decode_detector_id(det_id)
        if subsystem == 3 and bar >= 60:
            return True
        if subsystem == 1 and plane >= 2:
            return True
        return False

    def n_sides(self, det_id: int) -> int:
        """Return number of readout sides (1 for vertical single-ended, 2 for double-ended)."""
        if self.is_vertical(det_id):
            return 1
        return 2

    def get_num_sipms(self, det_id: int) -> int:
        """
        Return the total number of SiPMs instrumenting this bar:
        - Veto Horizontal (planes 0, 1): 8 Left + 8 Right = 16 SiPMs
        - Veto Vertical (plane 2): 8 Top = 8 SiPMs
        - Upstream (US) Horizontal (stations 0..4): 8 Left + 8 Right = 16 SiPMs
        - Downstream (DS) Horizontal: 1 Left + 1 Right = 2 SiPMs
        - Downstream (DS) Vertical: 1 Top = 1 SiPM
        """
        subsystem, plane, bar = self.decode_detector_id(det_id)
        if subsystem == 1:
            return 8 if plane >= 2 else 16
        elif subsystem == 2:
            return 16
        elif subsystem == 3:
            return 1 if bar >= 60 else 2
        return 16

    def decode_sipm_side(self, det_id: int, sipm_num: Union[int, str]) -> str:
        """Determine readout side ('left', 'right', or 'top') from det_id and sipm_num."""
        if isinstance(sipm_num, str):
            s = sipm_num.strip().lower()
            if s in ("l", "left"):
                return "left"
            if s in ("r", "right"):
                return "right"
            if s in ("t", "top"):
                return "top"
            try:
                sipm_num = int(sipm_num)
            except ValueError:
                raise ValueError(f"Unrecognized sipm_num side string: {sipm_num}")

        subsystem, _, _ = self.decode_detector_id(det_id)

        if self.is_vertical(det_id):
            return "top"

        if subsystem == 1 and sipm_num >= 16:
            if sipm_num < 224:
                rel_plane = sipm_num % 112
                return "left" if rel_plane < 56 else "right"

        if subsystem == 2 and sipm_num >= 224:
            rel_us = (sipm_num - 224) % 160
            return "left" if rel_us < 80 else "right"

        if subsystem == 2 and sipm_num >= 16 and sipm_num < 800:
            rel_st = sipm_num % 160
            return "left" if rel_st < 80 else "right"

        if subsystem == 3:
            return "left" if sipm_num == 0 else "right"

        return "left" if sipm_num < 8 else "right"

    def has_gel(self, det_id: int, sipm_num: Union[int, str]) -> bool:
        """
        Determine whether the specified SiPM channel has optical silicone gel coupling:
        - Veto: Planes 0 & 1 have Wacker silicone gel (True); Plane 2 (post-2024, 3rd plane)
          is wrapped in aluminized Mylar with dry air interface (False).
        - Upstream (US):
            * Station 0: Right (True), Left (False)
            * Station 1: Left (True), Right (False)
            * Station 2: Right (True), Left (False)
            * Station 3: Left (True), Right (False)
            * Station 4: Left (True), Right (False)
        - Downstream (DS): All bars have dry air-gap coupling (always False).
        """
        subsystem, plane, _ = self.decode_detector_id(det_id)

        if subsystem == 1:
            return False if plane >= 2 else True

        if subsystem == 3:
            return False

        if subsystem == 2:
            side = self.decode_sipm_side(det_id, sipm_num)
            station = plane
            if getattr(self, "is_testbeam", False):
                # 2023 H8 Testbeam geometry (HCAL_calibration.pdf Section 2.1):
                # Station 0 (US1): Left (Gel), Right (Air)
                # Station 1 (US2): Left (Air), Right (Gel)
                # Station 2 (US3): Left (Gel), Right (Air)
                # Station 3 (US4): Left (Air), Right (Gel)
                # Station 4 (US5): Left (Gel), Right (Air)
                if station in (0, 2, 4):
                    return side == "left"
                else:
                    return side == "right"
            else:
                # TI18 SND@LHC installed orientation
                if station == 0:
                    return side == "right"
                elif station == 1:
                    return side == "left"
                elif station == 2:
                    return side == "right"
                elif station == 3:
                    return side == "left"
                elif station == 4:
                    return side == "left"
                else:
                    return side == "left"

        return False

    def get_bar_dimensions(self, det_id: int) -> Tuple[float, float, float]:
        """Return nominal (length, width, thickness) in cm for the bar containing det_id."""
        subsystem, plane, bar = self.decode_detector_id(det_id)
        if subsystem == 1:
            return BAR_DIMS["veto_ver"] if plane >= 2 else BAR_DIMS["veto_hor"]
        elif subsystem == 2:
            return BAR_DIMS["us_hor"]
        elif subsystem == 3:
            return BAR_DIMS["ds_ver"] if bar >= 60 else BAR_DIMS["ds_hor"]
        return (82.525, 6.0, 1.0)

    def get_bar_length(self, det_id: int) -> float:
        """Return bar length L [cm]."""
        return self.get_bar_dimensions(det_id)[0]

    def get_bar_end_area(self, det_id: int) -> float:
        """Return end-cap cross-sectional area [cm^2] (Width * Thickness)."""
        dims = self.get_bar_dimensions(det_id)
        return dims[1] * dims[2]

    def get_distance_to_sipm(self, point: Any, sipm_num: Union[int, str]) -> float:
        """
        Calculate distance d [cm] from the MCPoint to the readout SiPM.
        If SndlhcGeo interface is active, uses exact detector node master positions;
        otherwise evaluates along the bar axis using nominal length and geometry.
        """
        det_id, x, y, z, _, _ = self._extract_point_info(point)
        side = self.decode_sipm_side(det_id, sipm_num)
        length = self.get_bar_length(det_id)

        if self._mufi is not None:
            try:
                import ROOT
                v_l = ROOT.TVector3()
                v_r = ROOT.TVector3()
                self._mufi.GetPosition(det_id, v_l, v_r)
                impact = ROOT.TVector3(x, y, z)
                if side in ("left", "top"):
                    return float((v_l - impact).Mag())
                else:
                    return float((v_r - impact).Mag())
            except Exception:
                pass

        is_vert = self.is_vertical(det_id)
        if is_vert:
            subsystem, plane, _ = self.decode_detector_id(det_id)
            if subsystem == 1 and plane >= 2:
                # Veto Plane 2: vertical along Y, read out from the top edge (46 cm long)
                y_top = 46.0
                d = max(0.0, min(length, y_top - y))
                return float(d)
            else:
                y_top = 70.98
                d = max(0.0, min(length, y_top - y))
                return float(d)
        else:
            subsystem, _, _ = self.decode_detector_id(det_id)
            x_left = 3.53 if subsystem in (2, 3) else 4.5
            x_right = x_left - length
            if side == "left":
                d = max(0.0, min(length, x_left - x))
            else:
                d = max(0.0, min(length, x - x_right))
            return float(d)

    def get_fcone(self, point: Any, sipm_num: Union[int, str]) -> float:
        """
        Acceptance cone fraction towards readout face:
        - Dry air interface (DS, US air-side, Veto plane 2): 0.1129
        - Gel on short bars (Veto Planes 0 & 1): 0.2744 (direct critical angle cone)
        - Gel on long bars (US gel-side): 0.1750 (trapped by lateral air TIR)
        """
        det_id, _, _, _, _, _ = self._extract_point_info(point)
        if not self.has_gel(det_id, sipm_num):
            return F_CONE_AIR
        subsystem, _, _ = self.decode_detector_id(det_id)
        if subsystem == 1:
            return F_CONE_GEL_SHORT
        return F_CONE_GEL_LONG

    def get_fcone_eff(self, point: Any, sipm_num: Union[int, str]) -> float:
        """
        Position-dependent effective acceptance cone fraction including boundary edge deficit.
        f_cone_eff(d) = f_cone * [1 - alpha_edge * exp(-(d - delta_dead) / lambda_edge)]
        for distance d to closest lateral cut edge >= delta_dead (0 otherwise).
        """
        f_cone = self.get_fcone(point, sipm_num)
        if not getattr(self, "include_edge_effect", True):
            return f_cone

        det_id, x, y, z, _, _ = self._extract_point_info(point)
        subsystem, plane, bar = self.decode_detector_id(det_id)
        if subsystem != 2:  # Only applies to US wide scintillator bars (W = 6.0 cm)
            return f_cone

        # Lateral distance to cut edge (width W = 6.0 cm)
        if self._mufi is not None:
            try:
                import ROOT
                vl, vr = ROOT.TVector3(), ROOT.TVector3()
                self._mufi.GetPosition(det_id, vl, vr)
                yc = 0.5 * (vl.Y() + vr.Y())
            except Exception:
                yc = 17.004 + 6.01 * bar if getattr(self, "is_testbeam", False) else 12.04 + 6.01 * bar
        else:
            yc = 17.004 + 6.01 * bar if getattr(self, "is_testbeam", False) else 12.04 + 6.01 * bar

        d_edge = 3.0 - abs(y - yc)
        if d_edge < DELTA_DEAD:
            return 0.0  # Lost in dead wrapping skin

        eta_edge = 1.0 - ALPHA_EDGE * math.exp(-(d_edge - DELTA_DEAD) / LAMBDA_EDGE)
        return float(f_cone * max(0.0, eta_edge))

    def get_fdisp(
        self,
        points: Any,
        sipm_num: Union[int, str] = 0,
        det_id: Optional[int] = None,
    ) -> float:
        """
        Temporal dispersion degradation factor for multi-point cluster:
        f_disp = 1 / sqrt(1 + (sigma_t / tau_pulse)^2)
        where sigma_t is the energy-weighted photon arrival time spread at the SiPM.
        """
        if not getattr(self, "include_dispersion", True):
            return 1.0

        pts_list = [points] if hasattr(points, "GetDetectorID") or isinstance(points, dict) else list(points)
        if not pts_list:
            return 1.0

        arr_times = []
        weights = []
        for p in pts_list:
            _, _, _, _, t_pt, eloss = self._extract_point_info(p)
            if eloss <= 0.0:
                continue
            d = self.get_distance_to_sipm(p, sipm_num)
            arr_times.append(t_pt + d / PROP_SPEED_CM_NS)
            f_b = self.get_fbirks(p, eloss_mev=eloss) if getattr(self, "include_birks", True) else 1.0
            w_light = eloss * f_b * math.exp(-ATTENUATION_COEFF * d) * self.get_fcone_eff(p, sipm_num)
            weights.append(max(0.0, w_light))

        if len(weights) <= 1:
            return 1.0

        w = np.array(weights, dtype=np.float64)
        t = np.array(arr_times, dtype=np.float64)
        w_sum = np.sum(w)
        if w_sum <= 0.0:
            return 1.0

        t_mean = np.sum(w * t) / w_sum
        var_t = np.sum(w * (t - t_mean) ** 2) / w_sum
        sigma_t = math.sqrt(max(0.0, var_t))

        return float(1.0 / math.sqrt(1.0 + (sigma_t / TAU_PULSE_NS) ** 2))

    def get_Satt(
        self,
        point: Any,
        sipm_num: Union[int, str],
        include_mirror_in_trigger: bool = False,
    ) -> float:
        """
        Effective light attenuation factor: exp(-0.00426 * d).
        If include_mirror_in_trigger=True and the bar is vertical with readout on top only,
        adds reflected component: + 0.88 * exp(-0.00426 * (2L - d)).
        """
        det_id, _, _, _, _, _ = self._extract_point_info(point)
        d = self.get_distance_to_sipm(point, sipm_num)
        s_direct = math.exp(-ATTENUATION_COEFF * d)

        if include_mirror_in_trigger and self.is_vertical(det_id) and self.n_sides(det_id) == 1:
            length = self.get_bar_length(det_id)
            d_reflected = 2.0 * length - d
            s_reflected = MIRROR_REFLECTIVITY * math.exp(-ATTENUATION_COEFF * d_reflected)
            return float(s_direct + s_reflected)

        return float(s_direct)

    def get_ffresnel(self, point: Any, sipm_num: Union[int, str]) -> float:
        """Return 0.9958 if silicone gel coupled, 0.9115 if air-gap."""
        det_id, _, _, _, _, _ = self._extract_point_info(point)
        return F_FRESNEL_GEL if self.has_gel(det_id, sipm_num) else F_FRESNEL_AIR

    def get_fbirks(self, point: Any, eloss_mev: Optional[float] = None) -> float:
        """
        Birks' quenching suppression factor:
            f_Birks = 1.0 / (1.0 + k_B * (dE/dx))
        where k_B = 0.0126 cm/MeV for EJ-200 PVT plastic scintillator.
        Generalized for:
        - Relativistic MIPs (muons, pions, kaons): dedx ~ 2.0 MeV/cm
        - Electrons / Photons: CSDA range in PVT
        - Protons / Neutron recoils: Bragg peak / stopping power in PVT (~100-500 MeV/cm)
        - Heavy nuclear recoils (Carbon-12, Alphas): extreme dedx > 1000 MeV/cm
        """
        if not getattr(self, "include_birks", True):
            return 1.0
        de = eloss_mev
        if de is None:
            _, _, _, _, _, de = self._extract_point_info(point)
        if de <= 0.0:
            return 1.0

        pdg = None
        if hasattr(point, "PdgCode"):
            pdg = point.PdgCode()
        elif isinstance(point, dict) and "pdg" in point:
            pdg = point["pdg"]

        abs_pdg = abs(int(pdg)) if pdg is not None else 0

        if abs_pdg in (13, 211, 321):  # Muons, charged pions, kaons (MIP regime)
            dedx = 2.0
        elif abs_pdg in (11, 22):  # Electrons, gammas
            r_est = 0.412 * (de ** 1.2) if de < 1.5 else 1.0
            dx = max(0.005, min(1.0, r_est))
            dedx = de / dx
        elif abs_pdg in (2212, 2112):  # Protons, neutron recoils
            r_p = 0.0022 * (de ** 1.75) if de < 15.0 else 0.25
            dx = max(0.0005, min(1.0, r_p))
            dedx = de / dx
        elif abs_pdg >= 1000000000:  # Nuclear recoils (Carbon-12, Alphas, etc.)
            dx = max(0.0002, 0.0005 * (de ** 0.5))
            dedx = de / dx
        else:
            r_est = 0.412 * (de ** 1.2) if de < 1.5 else 1.0
            dx = max(0.005, min(1.0, r_est))
            dedx = de / dx

        return float(1.0 / (1.0 + KB_BIRKS * dedx))

    def get_farea(self, point: Any, sipm_num: Union[int, str]) -> float:
        """Fraction of bar-end area covered by sensitive SiPM area (0.36 cm^2 for large SiPM, 0.0 for small SiPM)."""
        det_id, _, _, _, _, _ = self._extract_point_info(point)
        if self.is_small_sipm(det_id, sipm_num):
            return 0.0
        end_area = self.get_bar_end_area(det_id)
        return float(SIPM_AREA / end_area)

    def get_sipm_pe(
        self,
        point: Any,
        sipm_num: Union[int, str],
        include_mirror_in_trigger: bool = False,
        eloss_mev: Optional[float] = None,
    ) -> float:
        """
        Calculate number of photoelectrons produced by a single MCPoint in a single SiPM:
        N_pe = 10000 * ΔE [MeV] * f_birks * f_cone * S_att * f_fresnel * f_area * PDE (0.50).
        Returns 0.0 for small SiPMs.
        """
        det_id, _, _, _, _, pt_eloss = self._extract_point_info(point)
        if self.is_small_sipm(det_id, sipm_num):
            return 0.0
        de = eloss_mev if eloss_mev is not None else pt_eloss

        f_birks = self.get_fbirks(point, eloss_mev=de)
        f_cone = self.get_fcone_eff(point, sipm_num)
        s_att = self.get_Satt(point, sipm_num, include_mirror_in_trigger=include_mirror_in_trigger)
        f_fresnel = self.get_ffresnel(point, sipm_num)
        f_area = self.get_farea(point, sipm_num)

        n_pe = LIGHT_YIELD * de * f_birks * f_cone * s_att * f_fresnel * f_area * SIPM_PDE
        if hasattr(self, "_k_factors") and self._k_factors:
            k_val = self.get_reduction_factor(det_id, sipm_num)
            n_pe *= k_val
        return float(n_pe)

    def group_mcpoints_by_bar(
        self,
        mcpoints: Any,
        time_window: Optional[float] = None,
    ) -> Dict[int, List[List[Any]]]:
        """Group MCPoints by bar (detID) matching DigiTaskSND::digitizeMuFilter."""
        bar_points: Dict[int, List[Any]] = {}

        n_pts = len(mcpoints) if hasattr(mcpoints, "__len__") else (
            mcpoints.GetEntries() if hasattr(mcpoints, "GetEntries") else 0
        )
        iterator = mcpoints if not hasattr(mcpoints, "GetEntries") else [mcpoints.At(i) for i in range(n_pts)]

        for p in iterator:
            if not p:
                continue
            det_id, _, _, _, _, _ = self._extract_point_info(p)
            if det_id not in bar_points:
                bar_points[det_id] = []
            bar_points[det_id].append(p)

        clustered: Dict[int, List[List[Any]]] = {}
        for det_id, pts in bar_points.items():
            if time_window is None:
                clustered[det_id] = [pts]
            else:
                pts_sorted = sorted(pts, key=lambda p: self._extract_point_info(p)[4])
                clusters = []
                curr_cluster = [pts_sorted[0]]
                for p in pts_sorted[1:]:
                    prev_t = self._extract_point_info(curr_cluster[-1])[4]
                    curr_t = self._extract_point_info(p)[4]
                    if (curr_t - prev_t) <= time_window:
                        curr_cluster.append(p)
                    else:
                        clusters.append(curr_cluster)
                        curr_cluster = [p]
                if curr_cluster:
                    clusters.append(curr_cluster)
                clustered[det_id] = clusters

        return clustered

    def get_sipm_pes(
        self,
        points: Any,
        include_mirror_in_trigger: bool = False,
        det_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return an array of photoelectrons for each SiPM on this bar,
        summed over all clustered MCPoints.
        """
        if hasattr(points, "GetDetectorID") or isinstance(points, dict):
            pts_list = [points]
        elif hasattr(points, "__iter__"):
            pts_list = list(points)
        else:
            pts_list = [points]

        if not pts_list:
            if det_id is not None:
                n_sipms = self.get_num_sipms(det_id)
                return np.zeros(n_sipms, dtype=np.float64)
            return np.zeros(0, dtype=np.float64)

        if det_id is None:
            det_id, _, _, _, _, _ = self._extract_point_info(pts_list[0])

        n_sipms = self.get_num_sipms(det_id)
        pe_array = np.zeros(n_sipms, dtype=np.float64)

        # Identify representative channel for each readout side
        sides: Dict[str, int] = {}
        for s_idx in range(n_sipms):
            side = self.decode_sipm_side(det_id, s_idx)
            if side not in sides:
                sides[side] = s_idx

        base_yields: Dict[str, float] = {}
        f_disps: Dict[str, float] = {}
        for side, rep_idx in sides.items():
            rep_k = self.get_reduction_factor(det_id, rep_idx)
            inv_k = (1.0 / rep_k) if rep_k > 0.0 else 1.0
            tot = sum(
                self.get_sipm_pe(
                    p,
                    sipm_num=rep_idx,
                    include_mirror_in_trigger=include_mirror_in_trigger,
                )
                * inv_k
                for p in pts_list
            )
            base_yields[side] = tot
            if getattr(self, "include_dispersion", True):
                f_disps[side] = self.get_fdisp(pts_list, sipm_num=rep_idx, det_id=det_id)
            else:
                f_disps[side] = 1.0

        for s_idx in range(n_sipms):
            if self.is_small_sipm(det_id, s_idx):
                continue
            side = self.decode_sipm_side(det_id, s_idx)
            k_val = self.get_reduction_factor(det_id, s_idx)
            pe_array[s_idx] = base_yields[side] * k_val * f_disps[side]

        return pe_array

    def eval_sipm_triggers(
        self,
        points: Any,
        sipm_configs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        det_id: Optional[int] = None,
        include_mirror: bool = True,
        decay_ns: float = DECAY_TIME_NS,
        rise_ns: float = 0.0,
        prop_speed_cm_ns: float = PROP_SPEED_CM_NS,
        dt_step_ns: float = 0.1,
        window_before_ns: float = 2.0,
        window_after_ns: float = 25.0,
        sample_photons: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate T1 and T2 discriminator crossings for every large SiPM channel on a bar
        using MCPoint positions, propagation time, optional bottom-mirror reflection,
        and the scintillation decay time (2.1 ns).

        `sipm_configs` can be None, a single dict applied to all channels on the bar,
        or a list of per-channel dicts (length == n_sipms) with keys:
          - thr_t1 (int, default 20)          <- thresholds.csv (`thr_t1`)
          - thr_t2 (int, default 20)          <- thresholds.csv (`thr_t2`)
          - lsb_thr_t1 (int, default 59)      <- board.toml (`lsb_thr_t1`)
          - lsb_thr_t2 (int, default 50)      <- board.toml (`lsb_thr_t2`)
          - g_t (float, default 3000.0)       <- channels.csv (`g_t` [Ohms])
          - i_1pe_ua (float, default 5.0)     <- 6x6 mm^2 large SiPM peak current [uA]
          - trig_t (str, default "t1_and_t2") <- channels.csv (`trig_t`)
          - trig_q (str, default "t2")        <- channels.csv (`trig_q`)
        """
        if hasattr(points, "GetDetectorID") or isinstance(points, dict):
            pts_list = [points]
        elif hasattr(points, "__iter__"):
            pts_list = list(points)
        else:
            pts_list = [points]

        if det_id is None and pts_list:
            det_id, _, _, _, _, _ = self._extract_point_info(pts_list[0])
        if det_id is None:
            raise ValueError("det_id must be provided when points is empty.")

        n_sipms = self.get_num_sipms(det_id)

        if sipm_configs is None:
            cfg_list = [{} for _ in range(n_sipms)]
        elif isinstance(sipm_configs, dict):
            cfg_list = [sipm_configs for _ in range(n_sipms)]
        else:
            cfg_list = list(sipm_configs)
            if len(cfg_list) != n_sipms:
                raise ValueError(
                    f"Expected {n_sipms} sipm_configs for det_id={det_id}, got {len(cfg_list)}"
                )

        t1_fired = np.zeros(n_sipms, dtype=bool)
        t2_fired = np.zeros(n_sipms, dtype=bool)
        hit_qualified = np.zeros(n_sipms, dtype=bool)
        peak_current_ua = np.zeros(n_sipms, dtype=np.float64)

        if not pts_list:
            return {
                "t1_fired": t1_fired,
                "t2_fired": t2_fired,
                "hit_qualified": hit_qualified,
                "peak_current_ua": peak_current_ua,
            }

        use_mirror = bool(include_mirror and self.is_vertical(det_id) and self.n_sides(det_id) == 1)
        bar_len = self.get_bar_length(det_id) if use_mirror else 0.0

        arr_times = []
        for p in pts_list:
            _, _, _, _, t_pt, _ = self._extract_point_info(p)
            d0 = self.get_distance_to_sipm(p, 0)
            arr_times.append(t_pt + d0 / prop_speed_cm_ns)
            if use_mirror:
                arr_times.append(t_pt + (2.0 * bar_len - d0) / prop_speed_cm_ns)

        t_grid = np.arange(
            min(arr_times) - window_before_ns,
            max(arr_times) + window_after_ns + dt_step_ns,
            dt_step_ns,
            dtype=np.float64,
        )

        if rng is None:
            rng = np.random.default_rng()

        # Compute waveform once per unique readout side ('left', 'right', or 'top')
        # All SiPMs on the same side share identical optical distance, coupling, and waveform
        side_waves: Dict[str, np.ndarray] = {}
        unique_sides = set(self.decode_sipm_side(det_id, ch) for ch in range(n_sipms))

        for side in unique_sides:
            # Pick reference channel for this side (0 for left/top, 8 for right in 16-ch bars, 1 for DS hor)
            ref_ch = 0 if side in ("left", "top") else (8 if n_sipms == 16 else 1)
            i_wave = np.zeros_like(t_grid)
            cfg_ref = cfg_list[ref_ch]
            i_1pe = float(cfg_ref.get("i_1pe_ua", DEFAULT_I_1PE_UA))
            if getattr(self, "include_rise_time", True):
                i_eff_1pe = i_1pe * PEAK_PULSE_FACTOR
            else:
                i_eff_1pe = i_1pe

            for p in pts_list:
                _, _, _, _, t_pt, pt_eloss = self._extract_point_info(p)
                d = self.get_distance_to_sipm(p, ref_ch)
                f_birks = self.get_fbirks(p, eloss_mev=pt_eloss) if getattr(self, "include_birks", True) else 1.0
                base_pe = (
                    LIGHT_YIELD
                    * pt_eloss
                    * f_birks
                    * self.get_fcone_eff(p, ref_ch)
                    * self.get_ffresnel(p, ref_ch)
                    * self.get_farea(p, ref_ch)
                    * SIPM_PDE
                )

                pe_dir = base_pe * math.exp(-ATTENUATION_COEFF * d)
                t_dir = t_pt + d / prop_speed_cm_ns

                if sample_photons:
                    n_ph = int(rng.poisson(pe_dir))
                    if n_ph > 0:
                        em_dt = rng.exponential(decay_ns, size=n_ph)
                        for dt_e in em_dt:
                            i_wave += i_eff_1pe * _pulse_kernel(
                                t_grid - (t_dir + dt_e), decay_ns=decay_ns, rise_ns=rise_ns
                            )
                else:
                    i_wave += i_eff_1pe * pe_dir * _pulse_kernel(
                        t_grid - t_dir, decay_ns=decay_ns, rise_ns=rise_ns
                    )

                if use_mirror:
                    pe_refl = base_pe * MIRROR_REFLECTIVITY * math.exp(
                        -ATTENUATION_COEFF * (2.0 * bar_len - d)
                    )
                    t_refl = t_pt + (2.0 * bar_len - d) / prop_speed_cm_ns
                    if sample_photons:
                        n_ph_r = int(rng.poisson(pe_refl))
                        if n_ph_r > 0:
                            em_dt_r = rng.exponential(decay_ns, size=n_ph_r)
                            for dt_e in em_dt_r:
                                i_wave += i_eff_1pe * _pulse_kernel(
                                    t_grid - (t_refl + dt_e), decay_ns=decay_ns, rise_ns=rise_ns
                                )
                    else:
                        i_wave += i_eff_1pe * pe_refl * _pulse_kernel(
                            t_grid - t_refl, decay_ns=decay_ns, rise_ns=rise_ns
                        )

            side_waves[side] = i_wave

        for ch in range(n_sipms):
            if self.is_small_sipm(det_id, ch):
                # Small SiPMs (channels 2 & 5 per side in US) are uncalibrated and never used
                t1_fired[ch] = False
                t2_fired[ch] = False
                hit_qualified[ch] = False
                peak_current_ua[ch] = 0.0
                continue

            cfg = cfg_list[ch]
            thr_t1 = int(cfg.get("thr_t1", DEFAULT_THR_T1))
            thr_t2 = int(cfg.get("thr_t2", DEFAULT_THR_T2))
            lsb_t1 = int(cfg.get("lsb_thr_t1", DEFAULT_LSB_THR_T1))
            lsb_t2 = int(cfg.get("lsb_thr_t2", DEFAULT_LSB_THR_T2))
            g_t = float(cfg.get("g_t", DEFAULT_G_T_OHMS))
            trig_t = str(cfg.get("trig_t", "t1_and_t2"))
            trig_q = str(cfg.get("trig_q", "t2"))

            v_th_t1_mv = thr_t1 * tofpet2_lsb_mv(lsb_t1, "t1")
            v_th_t2_mv = thr_t2 * tofpet2_lsb_mv(lsb_t2, "t2")
            i_th_t1_ua = v_th_t1_mv / (g_t * 1e-3)
            i_th_t2_ua = v_th_t2_mv / (g_t * 1e-3)

            ch_side = self.decode_sipm_side(det_id, ch)
            i_wave = side_waves[ch_side]
            k_val = self.get_reduction_factor(det_id, ch)
            i_peak = float(np.max(i_wave) * k_val) if len(i_wave) > 0 else 0.0
            peak_current_ua[ch] = i_peak

            fired_1 = i_peak >= i_th_t1_ua
            fired_2 = i_peak >= i_th_t2_ua
            t1_fired[ch] = fired_1
            t2_fired[ch] = fired_2

            if trig_t == "t1_and_t2":
                t_ok = fired_1 and fired_2
            elif trig_t == "t1":
                t_ok = fired_1
            elif trig_t == "t2":
                t_ok = fired_2
            else:
                t_ok = fired_1 and fired_2

            q_ok = fired_2 if trig_q == "t2" else (fired_1 if trig_q == "t1" else True)
            hit_qualified[ch] = bool(t_ok and q_ok)

        return {
            "t1_fired": t1_fired,
            "t2_fired": t2_fired,
            "hit_qualified": hit_qualified,
            "peak_current_ua": peak_current_ua,
        }

    def get_sipm_threshold_pe(
        self,
        sipm_config: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate the equivalent single-SiPM instantaneous threshold in photoelectrons
        from the DAQ configuration:
        N_pe^th = max(i_th_t1, i_th_t2) / i_1pe_ua
        """
        cfg = sipm_config or {}
        def_thr = 30 if getattr(self, "is_testbeam", False) else DEFAULT_THR_T1
        thr_t1 = int(cfg.get("thr_t1", def_thr))
        thr_t2 = int(cfg.get("thr_t2", def_thr))
        lsb_t1 = int(cfg.get("lsb_thr_t1", DEFAULT_LSB_THR_T1))
        lsb_t2 = int(cfg.get("lsb_thr_t2", DEFAULT_LSB_THR_T2))
        g_t = float(cfg.get("g_t", DEFAULT_G_T_OHMS))
        i_1pe = float(cfg.get("i_1pe_ua", DEFAULT_I_1PE_UA))
        trig_t = str(cfg.get("trig_t", "t1_and_t2"))
        trig_q = str(cfg.get("trig_q", "t2"))

        v_t1 = thr_t1 * tofpet2_lsb_mv(lsb_t1, "t1")
        v_t2 = thr_t2 * tofpet2_lsb_mv(lsb_t2, "t2")
        i_t1 = v_t1 / (g_t * 1e-3)
        i_t2 = v_t2 / (g_t * 1e-3)

        if trig_t == "t1_and_t2" or trig_q == "t2":
            i_req = max(i_t1, i_t2)
        elif trig_t == "t1":
            i_req = i_t1
        else:
            i_req = i_t2

        # Effective peak current per single photoelectron accounting for scintillation rise time (0.9 ns)
        # and decay time (2.1 ns) convolved with front-end TIA pulse response
        if getattr(self, "include_rise_time", True):
            i_eff_1pe = i_1pe * PEAK_PULSE_FACTOR
        else:
            i_eff_1pe = i_1pe

        return float(i_req / i_eff_1pe)

    def get_lowest_trigger_threshold(
        self,
        points: Any,
        sipm_configs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        det_id: Optional[int] = None,
        include_mirror: bool = True,
    ) -> Dict[str, Any]:
        """
        Determine the lowest photoelectron and energy deposit threshold required
        for the given MCPoint spatial configuration to trigger the bar (hit_qualified=True
        on at least one SiPM).

        Returns:
            Dict containing:
              - 'min_pe_threshold': Minimum photoelectrons required on the triggering SiPM
              - 'min_eloss_mev': Minimum total energy deposit [MeV] across points to trigger
              - 'triggering_channel': Channel index of the most sensitive SiPM
              - 'triggering_side': Readout side ('left', 'right', or 'top')
              - 'sipm_thresholds_pe': Array of p.e. thresholds for each SiPM
              - 'current_pes': Actual photoelectrons per SiPM from the input points
              - 'current_hit_qualified': Whether input points qualify a hit as-is
        """
        # Baseline trigger evaluation with input points
        eval_res = self.eval_sipm_triggers(
            points,
            sipm_configs=sipm_configs,
            det_id=det_id,
            include_mirror=include_mirror,
        )

        pts_list = [points] if hasattr(points, "GetDetectorID") or isinstance(points, dict) else list(points)
        if det_id is None and pts_list:
            det_id, _, _, _, _, _ = self._extract_point_info(pts_list[0])
        if det_id is None:
            raise ValueError("det_id required")

        n_sipms = self.get_num_sipms(det_id)
        current_pes = self.get_sipm_pes(
            pts_list, include_mirror_in_trigger=include_mirror, det_id=det_id
        )

        cfg_list = (
            [sipm_configs for _ in range(n_sipms)] if isinstance(sipm_configs, dict)
            else (list(sipm_configs) if sipm_configs is not None else [{} for _ in range(n_sipms)])
        )

        sipm_th_pe = np.array([self.get_sipm_threshold_pe(cfg) for cfg in cfg_list], dtype=np.float64)

        sub, pl, bar = self.decode_detector_id(det_id)
        if getattr(self, "is_testbeam", False):
            # Testbeam only had DS Station 1 (plane 0)
            if sub == 3 and pl > 0:
                return {
                    "min_pe_threshold": np.inf,
                    "min_eloss_mev": np.inf,
                    "triggering_channel": -1,
                    "triggering_side": "none",
                    "sipm_thresholds_pe": np.full(n_sipms, np.inf),
                    "current_pes": np.zeros(n_sipms),
                    "current_hit_qualified": False,
                }

        # Scale factor needed for each SiPM: scale_factor = sipm_th_pe / current_pes
        # Smallest scale factor identifies the most sensitive SiPM that fires first
        # Small SiPMs (channels 2 & 5 in US) are masked out and never used
        valid_mask = (current_pes > 0.0) & ~np.array([self.is_small_sipm(det_id, ch) for ch in range(n_sipms)])
        if getattr(self, "is_testbeam", False) and sub == 3 and pl == 0 and bar >= 60:
            # DS1-V in testbeam only had Top readout (ch 0); ch 1 was uninstrumented
            if len(valid_mask) > 1:
                valid_mask[1] = False

        scale_factors = np.full(n_sipms, np.inf)
        scale_factors[valid_mask] = sipm_th_pe[valid_mask] / current_pes[valid_mask]

        best_ch = int(np.argmin(scale_factors))
        min_scale = float(scale_factors[best_ch])

        total_input_eloss = sum(self._extract_point_info(p)[5] for p in pts_list)
        min_eloss = total_input_eloss * min_scale
        min_pe = float(sipm_th_pe[best_ch])
        best_side = self.decode_sipm_side(det_id, best_ch)

        return {
            "min_pe_threshold": min_pe,
            "min_eloss_mev": min_eloss,
            "triggering_channel": best_ch,
            "triggering_side": best_side,
            "sipm_thresholds_pe": sipm_th_pe,
            "current_pes": current_pes,
            "current_hit_qualified": bool(np.any(eval_res["hit_qualified"])),
            "sipm_hit_qualified": eval_res["hit_qualified"],
            "peak_current_ua": eval_res["peak_current_ua"],
        }


# Default module-level instance
_DEFAULT_CALCULATOR = MinPE()


def is_small_sipm(det_id: int, sipm_num: Union[int, str]) -> bool:
    return _DEFAULT_CALCULATOR.is_small_sipm(det_id, sipm_num)


def get_fcone(point: Any, sipm_num: Union[int, str]) -> float:
    return _DEFAULT_CALCULATOR.get_fcone(point, sipm_num)


def get_Satt(
    point: Any,
    sipm_num: Union[int, str],
    include_mirror_in_trigger: bool = False,
) -> float:
    return _DEFAULT_CALCULATOR.get_Satt(
        point, sipm_num, include_mirror_in_trigger=include_mirror_in_trigger
    )


def get_ffresnel(point: Any, sipm_num: Union[int, str]) -> float:
    return _DEFAULT_CALCULATOR.get_ffresnel(point, sipm_num)


def get_farea(point: Any, sipm_num: Union[int, str]) -> float:
    return _DEFAULT_CALCULATOR.get_farea(point, sipm_num)


def get_sipm_pe(
    point: Any,
    sipm_num: Union[int, str],
    include_mirror_in_trigger: bool = False,
    eloss_mev: Optional[float] = None,
) -> float:
    return _DEFAULT_CALCULATOR.get_sipm_pe(
        point, sipm_num, include_mirror_in_trigger=include_mirror_in_trigger, eloss_mev=eloss_mev
    )


def group_mcpoints_by_bar(
    mcpoints: Any,
    time_window: Optional[float] = None,
) -> Dict[int, List[List[Any]]]:
    return _DEFAULT_CALCULATOR.group_mcpoints_by_bar(mcpoints, time_window=time_window)


def get_sipm_pes(
    points: Any,
    include_mirror_in_trigger: bool = False,
    det_id: Optional[int] = None,
) -> np.ndarray:
    return _DEFAULT_CALCULATOR.get_sipm_pes(
        points, include_mirror_in_trigger=include_mirror_in_trigger, det_id=det_id
    )


get_sipm_pe_array = get_sipm_pes


def eval_sipm_triggers(
    points: Any,
    sipm_configs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    det_id: Optional[int] = None,
    include_mirror: bool = True,
    decay_ns: float = DECAY_TIME_NS,
    rise_ns: float = 0.0,
    prop_speed_cm_ns: float = PROP_SPEED_CM_NS,
    dt_step_ns: float = 0.1,
    window_before_ns: float = 2.0,
    window_after_ns: float = 25.0,
    sample_photons: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Module-level wrapper for MinPE.eval_sipm_triggers."""
    return _DEFAULT_CALCULATOR.eval_sipm_triggers(
        points,
        sipm_configs=sipm_configs,
        det_id=det_id,
        include_mirror=include_mirror,
        decay_ns=decay_ns,
        rise_ns=rise_ns,
        prop_speed_cm_ns=prop_speed_cm_ns,
        dt_step_ns=dt_step_ns,
        window_before_ns=window_before_ns,
        window_after_ns=window_after_ns,
        sample_photons=sample_photons,
        rng=rng,
    )


def get_lowest_trigger_threshold(
    points: Any,
    sipm_configs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    det_id: Optional[int] = None,
    include_mirror: bool = True,
) -> Dict[str, Any]:
    """Module-level wrapper for MinPE.get_lowest_trigger_threshold."""
    return _DEFAULT_CALCULATOR.get_lowest_trigger_threshold(
        points,
        sipm_configs=sipm_configs,
        det_id=det_id,
        include_mirror=include_mirror,
    )


if __name__ == "__main__":
    print("Running min_pe.py self-test with TOFPET2 trigger evaluation...")

    # 1. TOFPET2 DAC step size verification
    print("DAC LSB sizes:")
    print(f"  T1 (lsb=59): {tofpet2_lsb_mv(59, 't1'):.2f} mV/DAC count")
    print(f"  T2 (lsb=50): {tofpet2_lsb_mv(50, 't2'):.2f} mV/DAC count")
    print(f"  E  (lsb=50): {tofpet2_lsb_mv(50, 'e'):.2f} mV/DAC count")

    # 2. Discriminator threshold evaluation for US1 Bar 21000 with a 1.7 MeV MIP
    pt_us1 = {"detID": 21000, "x": -30.0, "y": 10.0, "z": 400.0, "time": 10.0, "eloss": 1.70}
    trig_res = eval_sipm_triggers(pt_us1)
    print(f"\nUS1 Bar 21000 Trigger Result (1.70 MeV MIP):")
    print(f"  T1 Fired (16 SiPMs)     : {trig_res['t1_fired']}")
    print(f"  T2 Fired (16 SiPMs)     : {trig_res['t2_fired']}")
    print(f"  Hit Qualified (16 SiPMs): {trig_res['hit_qualified']}")
    print(f"  Peak Current Left (Gel) : {trig_res['peak_current_ua'][0]:.2f} uA")
    print(f"  Peak Current Right (Air): {trig_res['peak_current_ua'][8]:.2f} uA")

    # 3. Lowest turn-on threshold determination
    thresh_info = get_lowest_trigger_threshold(pt_us1)
    print(f"\nLowest Turn-On Threshold for US1 Bar 21000:")
    print(f"  Threshold on Triggering SiPM : {thresh_info['min_pe_threshold']:.2f} p.e.")
    print(f"  Triggering Channel / Side    : SiPM {thresh_info['triggering_channel']} ({thresh_info['triggering_side']})")
    print(f"  Minimum Turn-On Energy       : {thresh_info['min_eloss_mev']:.3f} MeV ({thresh_info['min_eloss_mev']*1e3:.1f} keV)")

    print("\nSelf-test finished successfully.")
