#!/usr/bin/env python3
"""
scripts/qdc_calib/generate_tb_reduction_factors.py
--------------------------------------------------
Calculates testbeam-specific SiPM photoelectron reduction factors (k)
using 2023 H8 testbeam muon run (Run 100678) data, iterating over all
individual SiPM channels (0..15 for US, 0..1 for DS1-H, 0 for DS1-V):

    N_pe^data = QDC_MIP * (3.60 pC / QDC) / (0.43258 pC / pe)
    N_pe^th   = 10000 photons/MeV * E_MIP * f_birks * f_cone * S_att * f_fresnel * f_area * PDE
    k         = min(max(N_pe^data / N_pe^th, 0.005), 1.0)

Incorporates testbeam hardware configuration:
- US Gel alternating: US0, US2, US4 (Left Gel, Right Air); US1, US3 (Left Air, Right Gel)
- DS: DS1-H (Left & Right Air), DS1-V (Top Air)
- DAQ Gain: g_q = 2.5 (2_50)
- Writes to config/sipm_photoelectron_reduction_factors_testbeam.csv
"""

from __future__ import annotations

import os
import sys
import math
from typing import Dict, List, Tuple

import numpy as np
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Parameter sources: Eljen EJ-200 Scintillator Datasheet
LIGHT_YIELD: float = 10000.0
# Parameter sources: Hamamatsu S13360 MPPC Datasheet
SIPM_PDE: float = 0.50
SIPM_AREA_LARGE: float = 0.36
SIPM_AREA_SMALL: float = 0.09

# Physics justification: Optical acceptance cone fractions & Fresnel transmission
F_CONE_AIR: float = 0.1129
F_CONE_GEL_LONG: float = 0.1750
F_FRESNEL_AIR: float = 0.9115
F_FRESNEL_GEL: float = 0.9958

# Parameter source: Birks quenching parameter for PVT scintillator (Chou 1952, Craun & Smith 1970)
KB_BIRKS: float = 0.0126
# Parameter source: HCAL_calibration.pdf Section 2.1 (attenuation lambda = 234.7 cm)
ATTENUATION_COEFF: float = 0.00426

# Parameter source: MIP MPV from Langaus fits on simulation
E_MIP_US: float = 1.70
E_MIP_DS: float = 1.70
F_BIRKS_US: float = 1.0 / (1.0 + KB_BIRKS * (E_MIP_US / 1.0))
F_BIRKS_DS: float = 1.0 / (1.0 + KB_BIRKS * (E_MIP_DS / 1.0))

# Parameter sources: TOFPET2C ASIC datasheet (3.60 pC/QDC) & Hamamatsu S13360 MPPC datasheet (0.43258 pC/pe at G=2.5e6)
QDC_TO_PE: float = 3.60 / 0.43258

# Physics justification: Channel reduction parameter k = N_pe_data / N_pe_th accounts for


def extract_testbeam_muon_qdc(muon_file: str, max_events: int = 50000) -> Dict[Tuple[int, int], float]:
    """
    Extracts the median positive QDC for each individual (detector_id, channel) channel from muon run.
    """
    print(f"Reading testbeam muon data from {muon_file} (max {max_events:,} events)...")
    f = ROOT.TFile.Open(muon_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open {muon_file}")
    tree = f.Get("rawConv")

    qdc_store: Dict[Tuple[int, int], List[float]] = {}

    n_entries = min(max_events, tree.GetEntries())
    for iev in range(n_entries):
        tree.GetEntry(iev)
        for h in tree.Digi_MuFilterHits:
            if not h.isValid():
                continue
            did = h.GetDetectorID()
            sub = did // 10000
            if sub in (2, 3):
                n_sides = h.GetnSides()
                n_sipms = h.GetnSiPMs()
                for s in range(n_sides):
                    for j in range(n_sipms):
                        ch = j + s * n_sipms
                        q = float(h.GetSignal(ch))
                        if q > 0.0:
                            qdc_store.setdefault((did, ch), []).append(q)

    f.Close()

    med_qdc: Dict[Tuple[int, int], float] = {}
    for (did, ch), vals in qdc_store.items():
        if len(vals) >= 15:
            med_qdc[(did, ch)] = float(np.median(vals))

    print(f"Extracted median QDC for {len(med_qdc)} active individual channels.")
    return med_qdc


def compute_testbeam_reduction_factors(
    med_qdc: Dict[Tuple[int, int], float],
    out_csv: str = "config/sipm_photoelectron_reduction_factors_testbeam.csv",
):
    """
    Computes k for each SiPM channel under testbeam geometry and writes CSV.
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    records = []

    us_end_area = 6.0
    f_area_large_us = SIPM_AREA_LARGE / us_end_area
    bar_length_us = 83.0
    d_center_us = bar_length_us / 2.0
    s_att_us = math.exp(-ATTENUATION_COEFF * d_center_us)

    plane_side_medians: Dict[Tuple[int, str], float] = {}
    for pl in range(5):
        for side_str, ch_range in [("L", range(8)), ("R", range(8, 16))]:
            vals = []
            for b in range(10):
                did = 20000 + pl * 1000 + b
                for ch in ch_range:
                    if (ch % 8) not in (2, 5) and (did, ch) in med_qdc:
                        vals.append(med_qdc[(did, ch)])
            plane_side_medians[(pl, side_str)] = float(np.median(vals)) if vals else 4.0

    for pl in range(5):
        left_has_gel = (pl in (0, 2, 4))
        right_has_gel = not left_has_gel

        f_cone_l = (F_CONE_GEL_LONG if left_has_gel else F_CONE_AIR) * 0.979
        f_fresnel_l = F_FRESNEL_GEL if left_has_gel else F_FRESNEL_AIR
        n_pe_th_large_l = LIGHT_YIELD * E_MIP_US * F_BIRKS_US * f_cone_l * s_att_us * f_fresnel_l * f_area_large_us * SIPM_PDE

        f_cone_r = (F_CONE_GEL_LONG if right_has_gel else F_CONE_AIR) * 0.979
        f_fresnel_r = F_FRESNEL_GEL if right_has_gel else F_FRESNEL_AIR
        n_pe_th_large_r = LIGHT_YIELD * E_MIP_US * F_BIRKS_US * f_cone_r * s_att_us * f_fresnel_r * f_area_large_us * SIPM_PDE

        for bar in range(10):
            did = 20000 + pl * 1000 + bar

            for ch in range(8):
                is_small = (ch in (2, 5))
                if is_small:
                    records.append((did, ch, 0.0))
                else:
                    qdc = med_qdc.get((did, ch), plane_side_medians[(pl, "L")])
                    n_pe_data = qdc * QDC_TO_PE
                    k = float(np.clip(n_pe_data / n_pe_th_large_l, 0.005, 1.0))
                    records.append((did, ch, k))

            for ch in range(8, 16):
                is_small = ((ch % 8) in (2, 5))
                if is_small:
                    records.append((did, ch, 0.0))
                else:
                    qdc = med_qdc.get((did, ch), plane_side_medians[(pl, "R")])
                    n_pe_data = qdc * QDC_TO_PE
                    k = float(np.clip(n_pe_data / n_pe_th_large_r, 0.005, 1.0))
                    records.append((did, ch, k))

    ds_end_area = 1.0
    f_area_ds = SIPM_AREA_LARGE / ds_end_area
    s_att_ds_h = math.exp(-ATTENUATION_COEFF * 41.25)
    
    d_v = 31.75
    l_v = 63.525
    s_att_ds_v = math.exp(-ATTENUATION_COEFF * d_v) + 0.88 * math.exp(-ATTENUATION_COEFF * (2.0 * l_v - d_v))

    n_pe_th_ds_h = LIGHT_YIELD * E_MIP_DS * F_BIRKS_DS * F_CONE_AIR * s_att_ds_h * F_FRESNEL_AIR * f_area_ds * SIPM_PDE
    n_pe_th_ds_v = LIGHT_YIELD * E_MIP_DS * F_BIRKS_DS * F_CONE_AIR * s_att_ds_v * F_FRESNEL_AIR * f_area_ds * SIPM_PDE

    ds_h_vals_0 = [med_qdc[(30000+b, 0)] for b in range(60) if (30000+b, 0) in med_qdc]
    ds_h_vals_1 = [med_qdc[(30000+b, 1)] for b in range(60) if (30000+b, 1) in med_qdc]
    ds_v_vals_0 = [med_qdc[(30000+b, 0)] for b in range(60, 120) if (30000+b, 0) in med_qdc]

    med_ds_h0 = float(np.median(ds_h_vals_0)) if ds_h_vals_0 else 10.0
    med_ds_h1 = float(np.median(ds_h_vals_1)) if ds_h_vals_1 else 10.0
    med_ds_v0 = float(np.median(ds_v_vals_0)) if ds_v_vals_0 else 12.0

    for bar in range(60):
        did = 30000 + bar
        qdc_l = med_qdc.get((did, 0), med_ds_h0)
        qdc_r = med_qdc.get((did, 1), med_ds_h1)
        k_l = float(np.clip((qdc_l * QDC_TO_PE) / n_pe_th_ds_h, 0.005, 1.0))
        k_r = float(np.clip((qdc_r * QDC_TO_PE) / n_pe_th_ds_h, 0.005, 1.0))
        records.append((did, 0, k_l))
        records.append((did, 1, k_r))

    for bar in range(60, 120):
        did = 30000 + bar
        qdc_top = med_qdc.get((did, 0), med_ds_v0)
        k_top = float(np.clip((qdc_top * QDC_TO_PE) / n_pe_th_ds_v, 0.005, 1.0))
        records.append((did, 0, k_top))

    with open(out_csv, "w") as f:
        f.write("detector_id,sipm_number,k\n")
        for did, ch, k in records:
            f.write(f"{did},{ch},{k:.5f}\n")

    print(f"Saved {len(records)} testbeam reduction factors to {out_csv}")
    k_active = [r[2] for r in records if r[2] > 0.0]
    print(f"Active reduction factors summary: mean={np.mean(k_active):.4f}, median={np.median(k_active):.4f}, min={np.min(k_active):.4f}, max={np.max(k_active):.4f}")


def main():
    muon_file = "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/run_100678/sndsw_raw-0000.root"
    out_csv = "config/sipm_photoelectron_reduction_factors_testbeam.csv"

    med_qdc = extract_testbeam_muon_qdc(muon_file, max_events=40000)
    compute_testbeam_reduction_factors(med_qdc, out_csv)

    art_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"
    os.system(f"cp -f {out_csv} {art_dir}/")
    print(f"Copied {out_csv} to artifact directory.")


if __name__ == "__main__":
    main()
