#!/usr/bin/env python3
"""
Generate SiPM Photoelectron Reduction Parameters (k)
----------------------------------------------------
Calculates the semi-empirical photoelectron reduction parameter k (0 <= k <= 1)
by comparing the theoretical photoelectron yield predicted for a MIP in each SiPM
with the experimental photoelectrons recovered from official MIP-QDC calibrations:

    N_pe^data = QDC_MIP * (3.60 pC / QDC) / (0.43258 pC / pe)
    N_pe^th   = 10000 photons/MeV * E_MIP * f_cone * S_att * f_fresnel * f_area * PDE
    k         = min(max(N_pe^data / N_pe^th, 0.0), 1.0)

Generates a three-column CSV output:
    detector_id,sipm_number,k

Usage:
    python3 scripts/qdc_calib/generate_sipm_photoelectron_reduction_factors.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# Add local path for min_pe
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from min_pe import MinPE, LIGHT_YIELD, SIPM_PDE, KB_BIRKS

# Constants for QDC -> Photoelectrons
# TOFPET2C QDC charge conversion: 3.60 pC / QDC count
# 1 p.e. charge at SiPM gain G = 2.5e6 with 8% optical cross-talk:
# q_1pe = e * 2.5e6 * 1.08 = 0.43258 pC
QDC_TO_PE: float = 3.60 / 0.43258  # ~8.3222 pe / QDC

# Reference MIP energy loss MPV in 1.0 cm EJ-200 plastic scintillator
# MPV from Langaus fits on simulation: 1.65 MeV for Veto, 1.70 MeV for US & DS
E_MIP_DICT: Dict[int, float] = {
    1: 1.65,  # Veto (1.0 cm)
    2: 1.70,  # Upstream (1.0 cm)
    3: 1.70,  # Downstream (1.0 cm)
}


def calculate_sipm_reduction_factors(
    calib_json_path: str,
    v2_json_path: str = "",
    year: int = 2022,
) -> Tuple[List[Dict[str, float]], Dict[str, List[float]]]:
    """
    Computes theoretical N_pe, data N_pe, and reduction parameter k for each SiPM.
    """
    calc = MinPE(year=2024)  # Initialize with full geometry support

    with open(calib_json_path, "r") as f:
        calib_data = json.load(f)

    # Optional 2024 Veto Plane 2 constants
    v2_data = {}
    if v2_json_path and os.path.exists(v2_json_path):
        with open(v2_json_path, "r") as f:
            v2_raw = json.load(f)
            v2_data = {k: v for k, v in v2_raw.items() if len(k) > 2 and int(k) // 100000 == 12}

    all_keys = list(calib_data.keys())
    if v2_data:
        for k, v in v2_data.items():
            if float(v) > 0:
                calib_data[k] = v
                if k not in all_keys:
                    all_keys.append(k)

    records: List[Dict[str, float]] = []
    stats_by_category: Dict[str, List[float]] = {
        "veto_hor": [],
        "veto_ver": [],
        "us_gel": [],
        "us_air": [],
        "ds_hor": [],
        "ds_ver": [],
    }

    # Group calibrated channels by bar for inferring uncalibrated channel values
    bar_k_map: Dict[Tuple[int, str], List[float]] = {}

    for k_str in all_keys:
        if len(k_str) <= 2:
            continue
        kid = int(k_str)
        det_id = kid // 100
        sipm_num = kid % 100
        qdc_val = float(calib_data[k_str])
        if qdc_val <= 0:
            continue

        subsystem, plane, bar = calc.decode_detector_id(det_id)
        if subsystem == 1 and plane >= 2 and year < 2024 and not v2_json_path:
            continue

        # Small SiPMs (channels 2 & 5 in US) are uncalibrated and never used
        if calc.is_small_sipm(det_id, sipm_num):
            continue

        de = E_MIP_DICT.get(subsystem, 1.70)
        length = calc.get_bar_length(det_id)
        d_center = length / 2.0

        pt_dummy = {"detID": det_id, "x": 0.0, "y": 0.0, "z": 0.0}
        f_cone = calc.get_fcone(pt_dummy, sipm_num)
        # Average boundary edge attenuation factor over bar width W = 6.0 cm
        eta_edge = 0.979 if subsystem == 2 else 1.0
        f_cone_eff = f_cone * eta_edge

        f_fresnel = calc.get_ffresnel(pt_dummy, sipm_num)
        f_area = calc.get_farea(pt_dummy, sipm_num)

        # Birks' quenching factor for MIP (1.0 cm PVT scintillator)
        # dE/dx = de [MeV] / 1.0 [cm] = de [MeV/cm]
        f_birks = 1.0 / (1.0 + KB_BIRKS * (de / 1.0))

        # Attenuation to bar center
        s_att = np.exp(-0.00426 * d_center)
        if calc.is_vertical(det_id) and calc.n_sides(det_id) == 1:
            s_att += 0.88 * np.exp(-0.00426 * (2.0 * length - d_center))

        n_pe_th = LIGHT_YIELD * de * f_birks * f_cone_eff * s_att * f_fresnel * f_area * SIPM_PDE
        n_pe_data = qdc_val * QDC_TO_PE

        k = n_pe_data / n_pe_th if n_pe_th > 0 else 0.0
        k_clamped = min(max(k, 0.0), 1.0)

        side = calc.decode_sipm_side(det_id, sipm_num)
        bar_key = (det_id, side)
        if bar_key not in bar_k_map:
            bar_k_map[bar_key] = []
        bar_k_map[bar_key].append(k_clamped)

        rec = {
            "detector_id": int(det_id),
            "sipm_number": int(sipm_num),
            "k": float(k_clamped),
            "qdc": float(qdc_val),
            "n_pe_data": float(n_pe_data),
            "n_pe_th": float(n_pe_th),
            "subsystem": int(subsystem),
            "plane": int(plane),
            "bar": int(bar),
        }
        records.append(rec)

        # Categorize for statistical reporting
        if subsystem == 1:
            if plane >= 2:
                stats_by_category["veto_ver"].append(k_clamped)
            else:
                stats_by_category["veto_hor"].append(k_clamped)
        elif subsystem == 2:
            if calc.has_gel(det_id, sipm_num):
                stats_by_category["us_gel"].append(k_clamped)
            else:
                stats_by_category["us_air"].append(k_clamped)
        elif subsystem == 3:
            if bar < 60:
                stats_by_category["ds_hor"].append(k_clamped)
            else:
                stats_by_category["ds_ver"].append(k_clamped)

    # Sort records cleanly by detector_id and sipm_number
    records.sort(key=lambda r: (r["detector_id"], r["sipm_number"]))
    return records, stats_by_category


def main():
    parser = argparse.ArgumentParser(
        description="Generate SiPM Photoelectron Reduction Parameters (k) from MIP Calibration"
    )
    parser.add_argument(
        "--calib-json",
        default="/eos/experiment/sndlhc/calibration/MuFilter/SiPMqdcCalibration/SiPMqdcCalibrationConstants_r4958-5408.json",
        help="Path to official SiPM QDC calibration JSON (Run 4958-5408)",
    )
    parser.add_argument(
        "--v2-json",
        default="/eos/experiment/sndlhc/calibration/MuFilter/SiPMqdcCalibration/SiPMqdcCalibrationConstants_r13267-13526.json",
        help="Optional path to 2024 JSON containing Veto Plane 2 constants",
    )
    parser.add_argument(
        "--output-csv",
        default="config/sipm_photoelectron_reduction_factors.csv",
        help="Output CSV file path (3 columns: detector_id, sipm_number, k)",
    )
    parser.add_argument(
        "--detailed-csv",
        default="config/sipm_photoelectron_reduction_factors_detailed.csv",
        help="Output detailed CSV file path with intermediate physics quantities",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="Detector year configuration (2022 or 2024)",
    )
    args = parser.parse_args()

    v2_path = args.v2_json if args.year >= 2024 else ""
    records, stats = calculate_sipm_reduction_factors(
        args.calib_json, v2_json_path=v2_path, year=args.year
    )

    # Ensure parent output directories exist
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1. Write the 3-column CSV: detector_id,sipm_number,k
    with open(args.output_csv, "w") as f:
        f.write("detector_id,sipm_number,k\n")
        for r in records:
            f.write(f"{r['detector_id']},{r['sipm_number']},{r['k']:.6f}\n")

    print(f"\n[Success] Saved 3-column reduction parameters to: {args.output_csv}")
    print(f"  Total SiPM channels written: {len(records):,}")

    # 2. Write detailed diagnostic CSV if requested
    if args.detailed_csv:
        det_dir = os.path.dirname(os.path.abspath(args.detailed_csv))
        if det_dir:
            os.makedirs(det_dir, exist_ok=True)
        with open(args.detailed_csv, "w") as f:
            f.write("detector_id,sipm_number,subsystem,plane,bar,qdc_mip,n_pe_data,n_pe_th,k\n")
            for r in records:
                f.write(
                    f"{r['detector_id']},{r['sipm_number']},{r['subsystem']},{r['plane']},{r['bar']},"
                    f"{r['qdc']:.3f},{r['n_pe_data']:.2f},{r['n_pe_th']:.2f},{r['k']:.6f}\n"
                )
        print(f"[Success] Saved detailed breakdown to: {args.detailed_csv}")

    # Copy to brain artifact directory if active
    artifact_dir = "/home/idioniso/.gemini/antigravity-cli/brain/89f2e48e-4381-4e67-a934-8bfea0891fb2"
    if os.path.exists(artifact_dir):
        os.system(f"cp -f {args.output_csv} {artifact_dir}/sipm_photoelectron_reduction_factors.csv")
        if args.detailed_csv:
            os.system(f"cp -f {args.detailed_csv} {artifact_dir}/sipm_photoelectron_reduction_factors_detailed.csv")

    # Print Category Summary Table
    print("\n" + "=" * 80)
    print("PHOTOELECTRON REDUCTION PARAMETER (k) SUMMARY BY SCINTILLATOR SETUP")
    print("=" * 80)
    cat_names = {
        "veto_hor": "Veto Planes 0 & 1 (Double-ended, Gel, 8 SiPMs/side)",
        "veto_ver": "Veto Plane 2 (Single-ended, Top Readout + Mirror, 8 SiPMs)",
        "us_gel":   "Upstream (US) with Optical Gel (Single-ended, 6 large SiPMs)",
        "us_air":   "Upstream (US) without Gel / Air (Single-ended, 6 large SiPMs)",
        "ds_hor":   "Downstream (DS) Horizontal (Double-ended, Air, 1 SiPM/side)",
        "ds_ver":   "Downstream (DS) Vertical (Single-ended, Top Readout + Mirror, 1 SiPM)",
    }

    for cat_key, desc in cat_names.items():
        vals = stats.get(cat_key, [])
        if vals:
            mean_k = np.mean(vals)
            std_k = np.std(vals)
            min_k = np.min(vals)
            max_k = np.max(vals)
            print(f"  * {desc}:")
            print(f"      Count: {len(vals):4d} | Mean k = {mean_k:.4f} +/- {std_k:.4f} | Range: [{min_k:.4f}, {max_k:.4f}]")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
