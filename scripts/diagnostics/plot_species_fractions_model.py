#!/usr/bin/env python3
"""
scripts/diagnostics/plot_species_fractions_model.py
--------------------------------------------------
Standalone script implementing the phenomenological charm hadron fragmentation model
and comparing it directly against simulated Monte Carlo truth points from SND@LHC.

Features:
  - Models the transition from threshold (quasi-elastic/resonance Lambda_c enhancement)
    to asymptotic DIS (parton shower fragmentation plateau).
  - Monte Carlo sampling of transition scale tau, asymptotic weights, and threshold weights.
  - Computes 68% CL (1-sigma) bands for f(D0), f(D+), f(Ds+), f(Lambda_c+), and <B_mu>(E_nu).
  - Superimposes SND@LHC MC points extracted from charm_diagnostic ROOT files.
  - Generates high-resolution publication-ready Matplotlib figures (PNG/PDF).
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt


SPECIES_META = {
    "D0": {
        "latex": r"$D^0$",
        "color": "#d62728",      # Crimson / Red
        "marker": "o",
        "pdg": 421,
    },
    "D+": {
        "latex": r"$D^+$",
        "color": "#1f77b4",      # Steel Blue
        "marker": "s",
        "pdg": 411,
    },
    "Ds+": {
        "latex": r"$D_s^+$",
        "color": "#2ca02c",      # Forest Green
        "marker": "^",
        "pdg": 431,
    },
    "Lambda_c": {
        "latex": r"$\Lambda_c^+$",
        "color": "#9467bd",      # Purple / Magenta
        "marker": "v",
        "pdg": 4122,
    },
}


def run_species_fraction_model(
    e_min: float = 4.5,
    e_max: float = 1000.0,
    n_points: int = 300,
    n_samples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs Monte Carlo sampling of the phenomenological species fraction model."""
    np.random.seed(seed)
    e_nu = np.logspace(np.log10(max(4.5, e_min)), np.log10(e_max), n_points)

    br_params = {
        "D0": (0.067, 0.003),
        "D+": (0.160, 0.004),
        "Lambda_c": (0.035, 0.005),
        "Ds+": (0.068, 0.006),
    }

    species = ["D0", "D+", "Lambda_c", "Ds+"]
    f_samples = {sp: np.zeros((n_samples, len(e_nu))) for sp in species}
    b_mu_samples = np.zeros((n_samples, len(e_nu)))

    for s in range(n_samples):
        tau_s = max(8.0, float(np.random.normal(18.0, 4.0)))

        w_inf = {
            "D0": max(0.40, float(np.random.normal(0.58, 0.04))),
            "D+": max(0.12, float(np.random.normal(0.23, 0.03))),
            "Lambda_c": max(0.03, float(np.random.normal(0.08, 0.02))),
            "Ds+": max(0.04, float(np.random.normal(0.09, 0.02))),
        }
        sum_inf = sum(w_inf.values())
        f_inf_s = {k: v / sum_inf for k, v in w_inf.items()}

        w_thresh = {
            "D0": max(0.25, float(np.random.normal(0.45, 0.06))),
            "D+": max(0.08, float(np.random.normal(0.17, 0.04))),
            "Lambda_c": max(0.12, float(np.random.normal(0.30, 0.08))),
            "Ds+": max(0.01, float(np.random.normal(0.04, 0.02))),
        }
        sum_thresh = sum(w_thresh.values())
        f_thresh_s = {k: v / sum_thresh for k, v in w_thresh.items()}

        decay = np.exp(-(e_nu - 4.5) / tau_s)

        raw = {sp: f_inf_s[sp] + (f_thresh_s[sp] - f_inf_s[sp]) * decay for sp in species}
        tot_raw = sum(raw.values())

        br_s = {sp: max(0.01, float(np.random.normal(br_params[sp][0], br_params[sp][1]))) for sp in species}

        b_avg = np.zeros(len(e_nu))
        for sp in species:
            f_val = raw[sp] / tot_raw
            f_samples[sp][s, :] = f_val
            b_avg += f_val * br_s[sp]
        b_mu_samples[s, :] = b_avg

    f_med = {sp: np.median(f_samples[sp], axis=0) for sp in species}
    f_low = {sp: np.percentile(f_samples[sp], 16, axis=0) for sp in species}
    f_high = {sp: np.percentile(f_samples[sp], 84, axis=0) for sp in species}

    b_med = np.median(b_mu_samples, axis=0) * 100.0
    b_low = np.percentile(b_mu_samples, 16, axis=0) * 100.0
    b_high = np.percentile(b_mu_samples, 84, axis=0) * 100.0

    return {
        "e_nu": e_nu,
        "f_med": f_med,
        "f_low": f_low,
        "f_high": f_high,
        "b_med": b_med,
        "b_low": b_low,
        "b_high": b_high,
    }


def load_mc_graphs_from_root(root_file_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Loads MC species fraction data points from a charm_diagnostic ROOT file."""
    mc_data: Dict[str, Dict[str, np.ndarray]] = {}
    if not os.path.exists(root_file_path):
        return mc_data

    try:
        import ROOT
        ROOT.gROOT.SetBatch(True)
        tfile = ROOT.TFile.Open(root_file_path)
        if not tfile or tfile.IsZombie():
            return mc_data

        name_map = {
            "D0": "D0",
            "D+": "Dplus",
            "Ds+": "Ds",
            "Lambda_c": "Lambda_c",
        }

        for sp_key, r_name in name_map.items():
            g = tfile.Get(f"Summary/g_fraction_{r_name}_vs_nu_energy")
            if not g:
                g = tfile.Get(f"g_fraction_{r_name}_vs_nu_energy")
            if not g:
                continue

            n_pts = g.GetN()
            x = np.array([g.GetPointX(i) for i in range(n_pts)])
            y = np.array([g.GetPointY(i) for i in range(n_pts)])
            x_err_lo = np.array([g.GetErrorXlow(i) for i in range(n_pts)])
            x_err_hi = np.array([g.GetErrorXhigh(i) for i in range(n_pts)])
            y_err_lo = np.array([g.GetErrorYlow(i) for i in range(n_pts)])
            y_err_hi = np.array([g.GetErrorYhigh(i) for i in range(n_pts)])

            mc_data[sp_key] = {
                "x": x,
                "y": y,
                "x_err": (x_err_lo, x_err_hi),
                "y_err": (y_err_lo, y_err_hi),
            }

        tfile.Close()
    except Exception as e:
        print(f"[Warning] Could not extract ROOT data: {e}", file=sys.stderr)

    return mc_data


def plot_comparison(
    model: Dict[str, Any],
    mc_data: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    title_suffix: str = "",
) -> None:
    """Plots species fractions and effective B_mu in a 2-panel figure."""
    fig, (ax_sp, ax_bmu) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2.5, 1.2]}, sharex=True
    )

    e_nu = model["e_nu"]
    species_order = ["D0", "D+", "Ds+", "Lambda_c"]

    # --- Top Panel: Species Fractions ---
    for sp in species_order:
        meta = SPECIES_META[sp]
        col = meta["color"]
        lbl = meta["latex"]

        # Shaded 68% CL model band
        ax_sp.fill_between(
            e_nu,
            model["f_low"][sp],
            model["f_high"][sp],
            color=col,
            alpha=0.20,
            label=f"{lbl} Model (68% CL)",
        )
        # Median model curve
        ax_sp.plot(
            e_nu,
            model["f_med"][sp],
            color=col,
            linewidth=2.0,
            linestyle="-",
        )

        # MC data points
        if sp in mc_data:
            d = mc_data[sp]
            ax_sp.errorbar(
                d["x"],
                d["y"],
                xerr=d["x_err"],
                yerr=d["y_err"],
                fmt=meta["marker"],
                color=col,
                ecolor=col,
                capsize=2.5,
                elinewidth=1.2,
                markersize=6,
                label=f"{lbl} (MC)",
            )

    ax_sp.set_ylabel(r"Fragmentation Fraction $f_i = N_i / N_{\mathrm{charm}}$", fontsize=12)
    ax_sp.set_ylim(0.0, 1.02)
    ax_sp.set_title(
        r"Charm Hadron Species Fragmentation Fractions vs $E_\nu$" + title_suffix,
        fontsize=13,
        pad=10,
    )
    ax_sp.grid(True, linestyle="--", alpha=0.5)
    ax_sp.legend(loc="upper right", ncol=2, framealpha=0.90, fontsize=10)

    # --- Bottom Panel: Effective Semi-Muonic Branching Ratio ---
    ax_bmu.fill_between(
        e_nu,
        model["b_low"],
        model["b_high"],
        color="#d95f02",
        alpha=0.25,
        label=r"Model $\langle B_\mu \rangle(E_\nu)$ (68% CL)",
    )
    ax_bmu.plot(
        e_nu,
        model["b_med"],
        color="#d95f02",
        linewidth=2.0,
        label=r"Model Median $\langle B_\mu \rangle$",
    )
    ax_bmu.axhline(
        8.48,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=r"DIS Asymptotic Baseline ($8.48 \pm 0.50\%$, CHORUS/NOMAD)",
    )

    ax_bmu.set_xlabel(r"Neutrino Energy $E_\nu$ [GeV]", fontsize=12)
    ax_bmu.set_ylabel(r"$\langle B_\mu \rangle$ [%]", fontsize=12)
    ax_bmu.set_ylim(5.0, 11.0)
    ax_bmu.grid(True, linestyle="--", alpha=0.5)
    ax_bmu.legend(loc="lower right", framealpha=0.90, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"[Info] Saved comparison plot to: {os.path.abspath(output_path)}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Charm hadron species fragmentation fraction model vs MC truth"
    )
    parser.add_argument(
        "-r", "--root-file",
        type=str,
        default="/eos/user/i/idioniso/snd-numu-charm/charm_diagnostic_nu14_2022.root",
        help="Path to charm_diagnostic ROOT file containing MC truth points",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="species_fractions_model_comparison.png",
        help="Output plot image filename (.png or .pdf)",
    )
    parser.add_argument(
        "--e-max",
        type=float,
        default=1000.0,
        help="Maximum neutrino energy in GeV (default: 1000.0)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of Monte Carlo sampling iterations (default: 2000)",
    )
    args = parser.parse_args()

    print("=" * 78)
    print(" SND@LHC: Charm Hadron Species Fractions Model vs MC Truth Comparison")
    print("=" * 78)
    print(f" Energy Range : [4.5, {args.e_max}] GeV")
    print(f" MC Samples   : {args.samples}")
    print(f" ROOT Source  : {args.root_file}")
    print(f" Output Plot  : {args.output}")
    print("=" * 78)

    model = run_species_fraction_model(e_min=4.5, e_max=args.e_max, n_samples=args.samples)
    mc_data = load_mc_graphs_from_root(args.root_file) if args.root_file else {}

    if mc_data:
        print(f" Loaded MC graphs for {len(mc_data)} species from ROOT file.")
    else:
        print(" No MC graphs loaded; plotting model predictions only.")

    plot_comparison(model, mc_data, args.output)


if __name__ == "__main__":
    main()
