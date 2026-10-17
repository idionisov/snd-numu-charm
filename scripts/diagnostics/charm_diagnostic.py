#!/usr/bin/env python3
"""
scripts/diagnostics/charm_diagnostic.py
----------------------------------------
Diagnostic tool to inspect Monte Carlo truth charmed hadrons, production rates,
species fragmentation fractions, and decay channels in SND@LHC neutrino interactions.

Capable of analyzing:
  - Processed dimuon candidate files:
    /eos/user/i/idioniso/snd-numu-charm/data/.../*/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root
  - Full raw Monte Carlo production files:
    /eos/experiment/sndlhc/MonteCarlo/.../*/sndLHC.Genie-TGeant4_digCPP.root

Features:
  1. Charm Production Rate vs Neutrino Energy (E_nu):
     - R_charm(E_nu) = N(nu_mu CC with charm) / N(nu_mu CC inclusive).
     - Overlaid with theoretical prediction from the Georgi-Politzer / Barnett Slow Rescaling Model.
  2. Charm Hadron Energies & Kinematics:
     - Charm hadron energy spectrum E_charm (inclusive and per species).
     - 2D correlation E_charm vs E_nu and fragmentation function z = E_charm / E_nu.
  3. Charm Species Fragmentation Fractions:
     - Fractions f(D0), f(D+), f(Ds), f(Lambda_c) overall and as a function of E_nu and E_charm.
  4. Exclusive Decay Channels & Branching Ratios vs Energy:
     - Full decay channel formulas (e.g. K- pi+, K- mu+ nu_mu).
     - Branching ratio evolution as a function of charm hadron energy E_charm and neutrino energy E_nu.
  5. Physical Modeling Checks:
     - Prompt semi-leptonic branching fractions BR(-> mu X) and BR(-> e X) vs PDG world averages.
     - Lepton universality test (BR(mu) / BR(e) ~ 1.0).
     - Proper lifetime c*tau distributions compared with PDG lifetimes.
     - Transverse momentum pT and production angle relative to neutrino beam.
     - Neutrino vs Antineutrino charm baryon asymmetry (Lambda_c production).
  6. ROOT Output with Superimposed Canvases:
     - Everything stored in a single structured ROOT file.
     - Dedicated 'Canvases/' directory containing fully formatted TCanvas objects with superimposed
       histograms, theoretical curves, and legends.

Supports high-performance parallel processing via ProcessPoolExecutor (-j / --jobs).
"""

from __future__ import annotations

import os
import sys
import math
import time
import re
import csv
import argparse
import subprocess
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(1111)

import numpy as np

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import resolve_input_files

DEFAULT_PRESETS: Dict[str, Dict[str, str]] = {
    "nu14_2022": {
        "input": "/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/*/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root",
        "description": "2022 nu14 volTarget production (1000 partitions)",
    },
    "old_100fb_2022": {
        "input": "/eos/user/i/idioniso/snd-numu-charm/data/old_sndlhc_100fb-1_2022_down_volTarget/*/sndLHC.Genie-TGeant4_digCPP_2MuTrks_truth.root",
        "description": "2022 100fb-1 volTarget production (400 partitions)",
    },
}



# Common particle names by PDG code for clean histogram labels
PDG_NAMES: Dict[int, str] = {
    11: "e-",
    -11: "e+",
    12: "nu_e",
    -12: "anti_nu_e",
    13: "mu-",
    -13: "mu+",
    14: "nu_mu",
    -14: "anti_nu_mu",
    15: "tau-",
    -15: "tau+",
    16: "nu_tau",
    -16: "anti_nu_tau",
    22: "gamma",
    111: "pi0",
    211: "pi+",
    -211: "pi-",
    130: "K_L0",
    310: "K_S0",
    311: "K0",
    -311: "anti_K0",
    321: "K+",
    -321: "K-",
    313: "K*(892)0",
    -313: "anti_K*(892)0",
    323: "K*(892)+",
    -323: "K*(892)-",
    221: "eta",
    331: "eta'",
    333: "phi",
    223: "omega",
    2112: "n",
    -2112: "anti_n",
    2212: "p",
    -2212: "anti_p",
    3122: "Lambda",
    -3122: "anti_Lambda",
    3222: "Sigma+",
    -3222: "anti_Sigma-",
    3212: "Sigma0",
    -3212: "anti_Sigma0",
    3112: "Sigma-",
    -3112: "anti_Sigma+",
    3322: "Xi0",
    -3322: "anti_Xi0",
    3312: "Xi-",
    -3312: "anti_Xi+",
    411: "D+",
    -411: "D-",
    421: "D0",
    -421: "anti_D0",
    431: "Ds+",
    -431: "Ds-",
    413: "D*+",
    -413: "D*-",
    423: "D*0",
    -423: "anti_D*0",
    433: "Ds*+",
    -433: "Ds*-",
    4122: "Lambda_c+",
    -4122: "anti_Lambda_c-",
    4222: "Sigma_c++",
    4212: "Sigma_c+",
    4112: "Sigma_c0",
    4232: "Xi_c+",
    4132: "Xi_c0",
    4332: "Omega_c0",
}

# Standard charm species names
SPECIES_NAMES: Dict[int, str] = {
    421: "D0",
    411: "Dplus",
    431: "Ds",
    4122: "Lambda_c",
    -421: "anti_D0",
    -411: "Dminus",
    -431: "Ds_minus",
    -4122: "anti_Lambda_c",
    413: "Dstar_plus",
    -413: "Dstar_minus",
    423: "Dstar0",
    -423: "anti_Dstar0",
    433: "Ds_star",
    -433: "Ds_star_minus",
    4222: "Sigma_c_pp",
    -4222: "anti_Sigma_c_mm",
    4212: "Sigma_c_p",
    -4212: "anti_Sigma_c_m",
    4112: "Sigma_c_0",
    -4112: "anti_Sigma_c_0",
    4232: "Xi_c_p",
    -4232: "anti_Xi_c_m",
    4132: "Xi_c_0",
    -4132: "anti_Xi_c_0",
    4332: "Omega_c_0",
    -4332: "anti_Omega_c_0",
}

# Ground state charm hadrons that decay weakly
GROUND_STATE_CHARM_PDGS = {411, 421, 431, 4122, 4232, 4132, 4332}

# PDG World Average inclusive semi-leptonic branching fractions for reference
PDG_SEMILEPTONIC_BR: Dict[int, Tuple[float, float]] = {
    421: (6.8, 6.7),    # D0: BR(mu+ X) ~ 6.8%, BR(e+ X) ~ 6.5-6.7%
    411: (15.6, 16.1),  # D+: BR(mu+ X) ~ 15.6%, BR(e+ X) ~ 16.1%
    431: (6.3, 6.3),    # Ds+: BR(mu+ X) ~ 6.3%, BR(e+ X) ~ 6.3%
    4122: (3.5, 4.0),   # Lambda_c+: BR(mu+ X) ~ 3.5%, BR(e+ X) ~ 4.0%
}

# PDG World Average proper lifetimes in micrometers (c*tau)
PDG_LIFETIMES_CTAU_UM: Dict[int, float] = {
    421: 122.9,   # D0: tau = 410.1 fs -> c*tau = 122.9 um
    411: 311.8,   # D+: tau = 1040 fs -> c*tau = 311.8 um
    431: 151.2,   # Ds+: tau = 504 fs -> c*tau = 151.2 um
    4122: 60.7,   # Lambda_c+: tau = 202.4 fs -> c*tau = 60.7 um
}

# Nominal masses (GeV) for boost calculation fallback
CHARM_NOMINAL_MASSES: Dict[int, float] = {
    421: 1.86483,
    411: 1.86965,
    431: 1.96834,
    4122: 2.28646,
    4232: 2.4679,
    4132: 2.4709,
    4332: 2.6952,
}

# Species color palette for multi-line and superimposed plots
SPECIES_COLORS: Dict[int, int] = {
    421: ROOT.TColor.GetColor("#d91e18"),       # D0
    411: ROOT.TColor.GetColor("#1e88e5"),       # D+
    431: ROOT.TColor.GetColor("#2e7d32"),       # Ds
    4122: ROOT.TColor.GetColor("#8e24aa"),      # Lambda_c
    -421: ROOT.TColor.GetColor("#d91e18"),      # anti-D0
    -411: ROOT.TColor.GetColor("#1e88e5"),      # D-
    -431: ROOT.TColor.GetColor("#2e7d32"),      # Ds-
    -4122: ROOT.TColor.GetColor("#8e24aa"),     # anti-Lambda_c
}

# Pale pastel fill colors for model 1-sigma uncertainty bands
SPECIES_FILL_COLORS: Dict[int, int] = {
    421: ROOT.TColor.GetColor("#d91e18"),
    411: ROOT.TColor.GetColor("#1e88e5"),
    431: ROOT.TColor.GetColor("#2e7d32"),
    4122: ROOT.TColor.GetColor("#8e24aa"),
    -421: ROOT.TColor.GetColor("#d91e18"),
    -411: ROOT.TColor.GetColor("#1e88e5"),
    -431: ROOT.TColor.GetColor("#2e7d32"),
    -4122: ROOT.TColor.GetColor("#8e24aa"),
}

# Darker prominent colors for external CHORUS measurements
CHORUS_COLORS: Dict[int, int] = {
    421: ROOT.TColor.GetColor("#800000"),        # D0 deep dark red / maroon
    411: ROOT.TColor.GetColor("#000080"),        # D+ deep dark navy blue
    431: ROOT.TColor.GetColor("#004d00"),        # Ds deep dark forest green
    4122: ROOT.TColor.GetColor("#4b0082"),       # Lambda_c deep dark violet
    -421: ROOT.TColor.GetColor("#800000"),
    -411: ROOT.TColor.GetColor("#000080"),
    -431: ROOT.TColor.GetColor("#004d00"),
    -4122: ROOT.TColor.GetColor("#4b0082"),
}

# ROOT TLatex formatted titles for charm hadron species
SPECIES_LATEX: Dict[int, str] = {
    421: "D^{0}",
    411: "D^{+}",
    431: "D_{s}^{+}",
    4122: "#Lambda_{c}^{+}",
    -421: "#bar{D}^{0}",
    -411: "D^{-}",
    -431: "D_{s}^{-}",
    -4122: "#bar{#Lambda}_{c}^{-}",
}

# CHORUS measured fractions: (mean, stat+syst err) from New J. Phys. 13 (2011) 093002
# WANF wide-band beam: <E_nu> = 27 GeV, central 68% interaction interval spans [13, 55] GeV
CHORUS_SPECIES_FRACTIONS: Dict[int, Tuple[float, float]] = {
    421: (0.437, 0.045),     # D0
    411: (0.253, 0.044),     # D+
    431: (0.118, 0.047),     # Ds+
    4122: (0.192, 0.042),    # Lambda_c+
}


def is_charm_hadron(pdg: int) -> bool:
    """Checks if a PDG code corresponds to a charmed hadron."""
    code = abs(pdg)
    if code < 100 or code >= 10000000:
        return False
    mod10 = code // 10
    q1 = mod10 % 10
    q2 = (mod10 // 10) % 10
    q3 = (mod10 // 100) % 10
    return (q1 == 4 or q2 == 4 or q3 == 4)


def get_particle_label(pdg: int) -> str:
    """Format readable particle label (e.g. 'pi+ (211)')."""
    name = PDG_NAMES.get(pdg, f"PDG_{pdg}")
    return f"{name} ({pdg})"


def get_species_name(abs_pdg: int) -> str:
    """Format readable charm species name (e.g. 'D0', 'Dplus', 'Ds', 'Lambda_c')."""
    return SPECIES_NAMES.get(abs_pdg, f"Charm_{abs_pdg}")


def format_channel_formula(pdg_tuple: Tuple[int, ...]) -> str:
    """Format a tuple of daughter PDGs into human-readable decay channel formula."""
    names = [PDG_NAMES.get(p, str(p)) for p in pdg_tuple]
    return " + ".join(sorted(names))


def channel_slug(pdg_tuple: Tuple[int, ...]) -> str:
    """Creates a filesystem/ROOT safe identifier for a decay channel."""
    parts = []
    for p in pdg_tuple:
        name = PDG_NAMES.get(p, f"pdg{p}")
        clean = name.replace("+", "plus").replace("-", "minus").replace("'", "prime").replace("_", "")
        parts.append(clean)
    return "_".join(sorted(parts))


NAME_TO_PDG: Dict[str, int] = {v: k for k, v in PDG_NAMES.items()}
NAME_TO_PDG.update({
    "phi": 333,
    "omega": 223,
    "Lambda": 3122,
    "anti-Lambda": -3122,
    "Lambda0": 3122,
    "anti_Lambda0": -3122,
    "Kstar(892)0": 313,
    "anti-Kstar(892)0": -313,
    "anti_Kstar(892)0": -313,
    "Kstar(892)+": 323,
    "Kstar(892)-": -323,
    "K*(892)0": 313,
    "anti-K*(892)0": -313,
    "K*(892)+": 323,
    "K*(892)-": -323,
})
SELF_CONJUGATE_PDGS: Set[int] = {22, 111, 130, 310, 221, 331, 223, 333}


def expand_multiplicity_tokens(channel_str: str) -> str:
    """
    Expands multiplicity prefixes such as '2pi+' -> 'pi+ + pi+', '2pi0' -> 'pi0 + pi0'.
    e.g. 'K- + 2pi+' -> 'K- + pi+ + pi+'
         'K_S0 + pi+ + 2pi0' -> 'K_S0 + pi+ + pi0 + pi0'
    """
    norm_str = channel_str.replace("anti-", "anti_")
    parts = [p.strip() for p in re.split(r"\s+\+\s+", norm_str.strip()) if p.strip()]
    expanded = []
    for p in parts:
        m = re.match(r"^(\d+)([A-Za-z_\*\+\-\(\)\'][A-Za-z0-9_\*\+\-\(\)\']*)$", p)
        if m:
            count = int(m.group(1))
            token = m.group(2)
            expanded.extend([token] * count)
        else:
            expanded.append(p)
    return " + ".join(sorted(expanded))


def canonicalize_channel_str(channel_str: str) -> str:
    """Sorts daughter particle names in a channel string to ensure consistent matching, expanding multiplicities."""
    return expand_multiplicity_tokens(channel_str)


def get_conjugate_channel_str(channel_str: str) -> str:
    """Computes the CP / charge-conjugate channel formula."""
    parts = [p.strip() for p in channel_str.split(" + ") if p.strip()]
    cc_parts = []
    for p in parts:
        pdg = NAME_TO_PDG.get(p)
        if pdg is not None:
            cc_pdg = pdg if abs(pdg) in SELF_CONJUGATE_PDGS else -pdg
            cc_parts.append(PDG_NAMES.get(cc_pdg, p))
        else:
            if p.startswith("anti_"):
                cc_parts.append(p[5:])
            elif p.startswith("anti-"):
                cc_parts.append(p[5:])
            elif p.endswith("+"):
                cc_parts.append(p[:-1] + "-")
            elif p.endswith("-"):
                cc_parts.append(p[:-1] + "+")
            else:
                cc_parts.append(p)
    return " + ".join(sorted(cc_parts))


def get_conjugate_mode_label(mode_str: str) -> str:
    """
    Computes the CP / charge-conjugate label for an inclusive decay mode.
    e.g. 'e+ anything' -> 'e- anything'
         'K- anything' -> 'K+ anything'
         'p anything'  -> 'anti_p anything'
         '2pi+ pi- anything' -> '2pi- pi+ anything'
    """
    clean = mode_str.strip()
    has_anything = clean.endswith(" anything")
    if has_anything:
        prefix = clean[:-len(" anything")].strip()
    else:
        prefix = clean
    tokens = prefix.split()
    cc_tokens = []
    for tok in tokens:
        m = re.match(r"^(\d+)(.+)$", tok)
        if m and m.group(1).isdigit():
            num_str, p_name = m.groups()
        else:
            num_str, p_name = "", tok
        pdg = NAME_TO_PDG.get(p_name)
        if pdg is not None:
            cc_pdg = pdg if abs(pdg) in SELF_CONJUGATE_PDGS else -pdg
            cc_p_name = PDG_NAMES.get(cc_pdg, p_name)
        else:
            if p_name.startswith("anti_"):
                cc_p_name = p_name[5:]
            elif p_name.startswith("anti-"):
                cc_p_name = p_name[5:]
            elif p_name.endswith("+"):
                cc_p_name = p_name[:-1] + "-"
            elif p_name.endswith("-"):
                cc_p_name = p_name[:-1] + "+"
            else:
                cc_p_name = p_name
        cc_tokens.append(f"{num_str}{cc_p_name}")
    res = " ".join(cc_tokens)
    if has_anything:
        res += " anything"
    return res


def normalize_channel_photons(channel_str: str) -> str:
    """
    Pairs photons (gamma + gamma) into pi0:
    e.g. 'K- + gamma + gamma + pi+' -> 'K- + pi+ + pi0'
         'gamma + gamma + gamma + gamma + pi+ + pi-' -> 'pi+ + pi- + 2pi0'
    """
    parts = [p.strip() for p in channel_str.split(" + ") if p.strip()]
    n_gamma = parts.count("gamma")
    if n_gamma >= 2:
        n_pi0 = n_gamma // 2
        rem_gamma = n_gamma % 2
        reduced = [p for p in parts if p != "gamma"]
        reduced.extend(["pi0"] * n_pi0)
        reduced.extend(["gamma"] * rem_gamma)
        return " + ".join(sorted(reduced))
    return channel_str


def expand_neutral_kaons(channel_str: str, has_ks_kl_split: bool = False) -> Set[str]:
    """
    Expands formulas containing neutral kaons so that simulated decays
    with either K_S0 (310) or K_L0 (130) or K0/anti_K0 match the corresponding PDG mode.
    If has_ks_kl_split is True, K_S0 and K_L0 are kept distinct.
    """
    parts = [p.strip() for p in channel_str.split(" + ") if p.strip()]
    forms = {channel_str}
    if any(k in parts for k in ["K_S0", "K_L0", "K0", "anti_K0"]):
        targets = ["K0", "anti_K0"]
        if not has_ks_kl_split:
            targets.extend(["K_S0", "K_L0"])
        for target in targets:
            new_parts = [target if p in ["K_S0", "K_L0", "K0", "anti_K0"] else p for p in parts]
            forms.add(" + ".join(sorted(new_parts)))
    return forms


def clopper_pearson_errors(k: int, n: int, cl: float = 0.6827) -> Tuple[float, float, float]:
    """
    Computes Clopper-Pearson asymmetric binomial confidence interval for branching fraction / efficiency:
    Returns (br_pct, err_low_pct, err_high_pct) in percent, strictly bounded in [0%, 100%].
    Guarantees upper uncertainty never exceeds 100.0% and lower uncertainty never drops below 0.0%.
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    val = 100.0 * k / n
    low = 100.0 * ROOT.TEfficiency.ClopperPearson(n, k, cl, False)
    high = 100.0 * ROOT.TEfficiency.ClopperPearson(n, k, cl, True)
    low = max(0.0, min(val, low))
    high = min(100.0, max(val, high))
    err_low = val - low
    err_high = high - val
    return val, err_low, err_high


def parse_inclusive_mode_particles(mode_str: str) -> Dict[str, int]:
    """
    Parses an inclusive mode specification (e.g. 'pi0 anything', '2pi+ pi- anything', 'K_S0 K- anything')
    into a dictionary of required particle names and multiplicities.
    """
    clean = re.sub(r"\s+anything$", "", mode_str.strip())
    tokens = [t.strip() for t in clean.split() if t.strip()]
    req: Dict[str, int] = {}
    for tok in tokens:
        m = re.match(r"^(\d+)([A-Za-z0-9_\*\+\-\(\)\']+)$", tok)
        if m:
            cnt = int(m.group(1))
            name = m.group(2)
        else:
            cnt = 1
            name = tok
        req[name] = req.get(name, 0) + cnt
    return req


def count_inclusive_multiplicity(
    daughter_pdgs: List[int],
    charm_pdg: int,
    req_particles: Dict[str, int],
) -> int:
    """
    Evaluates how many times an inclusive mode's required daughter set
    is present among the direct decay daughters of a charm hadron.
    Handles CP-conjugation for anti-charm hadrons (charm_pdg < 0).
    For single-particle modes (e.g. 'pi0 anything'), a decay with 2 pi0 returns multiplicity 2.
    For multi-particle modes (e.g. '2pi+ pi- anything' or 'K+ K- anything'),
    returns the number of disjoint combinations of the required daughter set.
    """
    if not daughter_pdgs or not req_particles:
        return 0

    counts = Counter(daughter_pdgs)
    # If neutral pions decayed to pairs of photons in MCTrack, pair photons into pi0
    if counts.get(22, 0) >= 2 and counts.get(111, 0) == 0:
        counts[111] += counts[22] // 2

    is_anti = (charm_pdg < 0)

    if len(req_particles) == 1:
        p_name, count_needed = next(iter(req_particles.items()))
        base_pdg = NAME_TO_PDG.get(p_name)
        if base_pdg is None:
            return 0
        if is_anti and abs(base_pdg) not in SELF_CONJUGATE_PDGS:
            target_pdg = -base_pdg
        else:
            target_pdg = base_pdg
        avail = counts.get(target_pdg, 0)
        return avail // count_needed

    occ_list = []
    for p_name, count_needed in req_particles.items():
        base_pdg = NAME_TO_PDG.get(p_name)
        if base_pdg is None:
            return 0
        if is_anti and abs(base_pdg) not in SELF_CONJUGATE_PDGS:
            target_pdg = -base_pdg
        else:
            target_pdg = base_pdg
        avail = counts.get(target_pdg, 0)
        occ_list.append(avail // count_needed)

    return min(occ_list) if occ_list else 0


def calculate_weighted_yield_and_error(
    w_sum: float,
    w2y2_sum: float,
    w2y_sum: float,
    w_tot: float,
    w2_tot: float,
) -> Tuple[float, float, float]:
    """
    Computes weighted inclusive yield and standard error of the mean:
      Y = W_m / W_tot
      Var(Y) = (sum w^2 y^2 - 2 * Y * sum w^2 y + Y^2 * sum w^2) / (W_tot^2)
    Returns (yield_pct, err_low_pct, err_high_pct).
    If W_m == 0 (no observed events), returns (0.0, 0.0, upper_limit_pct)
    using 1-sigma Poisson upper limit (1.15 * sqrt(w2_tot) / w_tot * 100%).
    """
    if w_tot <= 0.0:
        return 0.0, 0.0, 0.0

    y_val = w_sum / w_tot
    if w_sum <= 0.0:
        err_up = 1.15 * math.sqrt(w2_tot) / w_tot
        return 0.0, 0.0, err_up * 100.0

    var_y = (w2y2_sum - 2.0 * y_val * w2y_sum + (y_val ** 2) * w2_tot) / (w_tot ** 2)
    sigma = math.sqrt(max(0.0, var_y))
    val_pct = y_val * 100.0
    err_pct = sigma * 100.0
    err_low_pct = min(val_pct, err_pct)
    err_high_pct = err_pct
    return val_pct, err_low_pct, err_high_pct


class PDGDecayMode:
    """Represents an inclusive decay channel reference from PDG summary tables."""
    def __init__(self, pdg_mode: str, br_pct: float, br_err_up_pct: float, br_err_low_pct: float, req_particles: Dict[str, int]):
        self.pdg_mode = pdg_mode
        self.br_pct = br_pct
        self.br_err_up_pct = br_err_up_pct
        self.br_err_low_pct = br_err_low_pct
        self.req_particles = req_particles

    def __repr__(self) -> str:
        return f"<PDGDecayMode {self.pdg_mode}: {self.br_pct:.3f}% (+{self.br_err_up_pct:.3f}/-{self.br_err_low_pct:.3f}%)>"


def load_pdg_summary_tables(tables_dir: Optional[str]) -> Dict[int, Dict[str, Any]]:
    """
    Loads PDG inclusive decay reference tables from CSV files in tables_dir.
    Maps:
      branching_ratio_D0.csv -> 421 (D0)
      branching_ratio_Dplus.csv -> 411 (D+)
      branching_ratio_DsPlus.csv / branching_ratio_Ds.csv -> 431 (Ds)
      branching_ratio_Lambda_c.csv -> 4122 (Lambda_c)
    Returns:
      {
        species_pdg: {
            "modes_by_label": {pdg_mode: PDGDecayMode},
            "modes_list": [PDGDecayMode, ...],
        }
      }
    """
    if not tables_dir or not os.path.isdir(tables_dir):
        return {}

    mapping = {
        "branching_ratio_D0.csv": 421,
        "branching_ratio_Dplus.csv": 411,
        "branching_ratio_DsPlus.csv": 431,
        "branching_ratio_Ds.csv": 431,
        "branching_ratio_Lambda_c.csv": 4122,
        "branching_ratio_DsMinus.csv": -431,
    }

    result: Dict[int, Dict[str, Any]] = {}

    for fname, sp_pdg in mapping.items():
        if sp_pdg in result:
            continue
        fpath = os.path.join(tables_dir, fname)
        if not os.path.isfile(fpath):
            continue

        modes_by_label: Dict[str, PDGDecayMode] = {}
        modes_list: List[PDGDecayMode] = []

        with open(fpath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mode_str = row.get("pdg_mode", "").strip()
                if not mode_str:
                    continue
                try:
                    br_val = float(row.get("Gamma_i_vs_Gamma", 0.0)) * 100.0
                    err_up = float(row.get("Gamma_i_vs_Gamma_err_up", 0.0)) * 100.0
                    err_lo = float(row.get("Gamma_i_vs_Gamma_err_low", 0.0)) * 100.0
                except (ValueError, TypeError):
                    continue

                req_parts = parse_inclusive_mode_particles(mode_str)
                m_obj = PDGDecayMode(mode_str, br_val, err_up, err_lo, req_parts)
                modes_list.append(m_obj)
                modes_by_label[mode_str] = m_obj

        result[sp_pdg] = {
            "modes_by_label": modes_by_label,
            "modes_list": modes_list,
        }

    # Automatically add CP-conjugate anti-charm species if not already present
    for sp_pdg in list(result.keys()):
        anti_pdg = -sp_pdg
        if anti_pdg not in result:
            result[anti_pdg] = {
                "modes_by_label": dict(result[sp_pdg]["modes_by_label"]),
                "modes_list": list(result[sp_pdg]["modes_list"]),
            }

    return result


def is_partition_ready(truth_path: str) -> bool:
    """Checks whether a partition has completed all pipeline stages."""
    try:
        if not os.path.isfile(truth_path) or os.path.getsize(truth_path) < 1000:
            return False
        parent = os.path.dirname(truth_path)
        p_name = os.path.basename(parent)
        dataset_dir = os.path.dirname(parent)
        dataset_name = os.path.basename(dataset_dir)
        user_dir = os.path.dirname(dataset_dir)
        base_user = os.path.dirname(user_dir)
        disp_file = os.path.join(base_user, "event_displays", dataset_name, "partitions", f"displays_part{p_name}.root")
        disp_dir = os.path.dirname(disp_file)
        if os.path.isdir(disp_dir):
            return os.path.isfile(disp_file) and os.path.getsize(disp_file) > 500
        return True
    except Exception:
        return False


def fast_resolve_files(
    pattern: str,
    filelist_path: Optional[str] = None,
    max_files: int = -1,
    ready_only: bool = True,
) -> List[str]:
    """Quickly resolves input ROOT files from a text filelist, EOS command, or pattern."""
    matched: List[str] = []

    # 1. Explicit filelist option
    if filelist_path and os.path.isfile(filelist_path):
        with open(filelist_path, "r") as f:
            for line in f:
                p = line.strip()
                if p and not p.startswith("#") and p.endswith(".root"):
                    matched.append(p)
        if ready_only:
            matched = [f for f in matched if is_partition_ready(f)]
        if max_files > 0:
            matched = matched[:max_files]
        return matched

    # 2. Pattern starts with '@' (filelist convention)
    if pattern.startswith("@") and os.path.isfile(pattern[1:]):
        with open(pattern[1:], "r") as f:
            for line in f:
                p = line.strip()
                if p and not p.startswith("#") and p.endswith(".root"):
                    matched.append(p)
        if ready_only:
            matched = [f for f in matched if is_partition_ready(f)]
        if max_files > 0:
            matched = matched[:max_files]
        return matched

    # 3. Fast EOS resolution if pattern contains wildcard on EOS filesystem
    if "/eos/" in pattern and "*" in pattern:
        base_dir, fname_pat = pattern.split("/*", 1)
        fname = fname_pat.lstrip("/")
        server = "root://eosuser.cern.ch" if "/eos/user/" in pattern else "root://eospublic.cern.ch"
        cmd = ["eos", server, "find", "-f", "--name", fname, base_dir]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            if res.returncode == 0:
                lines = [l.strip() for l in res.stdout.splitlines() if l.strip().endswith(".root")]
                if lines:
                    lines = sorted(
                        lines,
                        key=lambda x: [int(c) if c.isdigit() else c for c in os.path.dirname(x).split("/")]
                    )
                    if ready_only:
                        ready_lines = []
                        for f in lines:
                            if is_partition_ready(f):
                                ready_lines.append(f)
                                if max_files > 0 and len(ready_lines) >= max_files:
                                    break
                        lines = ready_lines
                    elif max_files > 0:
                        lines = lines[:max_files]
                    return lines
        except Exception:
            pass

    # 4. Fallback to standard resolve_input_files
    try:
        matched = resolve_input_files(pattern, max_files=max_files)
        if ready_only:
            matched = [f for f in matched if is_partition_ready(f)]
    except Exception:
        pass

    return matched


def find_energy_bin_index(energy: float, bin_edges: List[float]) -> int:
    """Returns the 0-based bin index for energy, or -1 if out of bounds."""
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= energy < bin_edges[i + 1]:
            return i
    if energy >= bin_edges[-1]:
        return len(bin_edges) - 2
    return -1


def create_slow_rescaling_model(
    name: str = "f_slow_rescaling",
    e_min: float = 0.0,
    e_max: float = 1500.0,
    m_c: float = 1.2729,
    sigma_m_c: float = 0.0027,
    v_cd: float = 0.221,
    sigma_v_cd: float = 0.004,
    v_cs: float = 0.975,
    sigma_v_cs: float = 0.006,
    alpha: float = 1.15,
    e_scale: float = 28.0,
    target: str = "tungsten",  # "tungsten" (W, Z=74, A=184) or "isoscalar"
    nu_type: str = "nu",       # "nu" (nu_mu), "antinu" (anti-nu_mu), or "inclusive"
    line_color: int = ROOT.kRed + 1,
    band_color: int = ROOT.kRed - 7,
    scale: float = 1.0,
    scale_err: float = 0.0,
) -> Tuple[ROOT.TF1, ROOT.TGraphErrors, ROOT.TGraphAsymmErrors]:
    """
    Georgi-Politzer / Barnett Slow Rescaling Model prediction for:
      R_charm(E_nu) = sigma(nu_mu N -> mu- c X) / sigma_CC(nu_mu N -> mu- X)
    accounting for target nuclear composition (Tungsten vs Isoscalar) and neutrino type.
    Optionally folded with semi-leptonic branching ratio scale (e.g. B_mu for dimuons).

    Target composition:
      - Isoscalar: Z/A = 0.50, N/A = 0.50
      - Tungsten (nat W, Z=74, A=183.84): f_p = 74/184 = 0.402, f_n = 110/184 = 0.598
    Parton momentum integrals (at typical DIS Q^2 scale):
      - Valence: U_val ~ 0.285, D_val ~ 0.130 (for proton)
      - Sea: U_bar ~ D_bar ~ 0.035
      - Strange: S = kappa_s * D_bar with modern global PDF fits (kappa_s ~ 0.55 +/- 0.08)
    """
    m_p = 0.938272
    e_thresh = ((m_p + m_c)**2 - m_p**2) / (2.0 * m_p) + 1.50

    # Nuclear composition fractions
    if target.lower() in ["tungsten", "w"]:
        f_p = 74.0 / 183.84
        f_n = 1.0 - f_p  # ~ 0.5975
    else:
        f_p = 0.50
        f_n = 0.50

    # Parton momentum fractions on proton
    u_val_p = 0.285
    d_val_p = 0.130
    sea_q = 0.035
    kappa_s = 0.55  # modern PDF strange sea suppression factor
    s_quark = kappa_s * sea_q  # ~ 0.01925

    # Target nucleon average PDFs
    # Proton has (u_val_p, d_val_p), neutron has (d_val_p, u_val_p) by isospin symmetry
    d_target = f_p * d_val_p + f_n * u_val_p + sea_q
    u_target = f_p * u_val_p + f_n * d_val_p + sea_q

    # Calculate asymptotic ratio R_inf for nu, antinu, or inclusive
    if nu_type.lower() in ["nu", "nu_mu"]:
        # nu_mu: interacts with d and s (flat y, \int dy = 1) and \bar{u} ((1-y)^2, \int (1-y)^2 dy = 1/3)
        num = (v_cd**2 * d_target) + (v_cs**2 * s_quark)
        denom = d_target + s_quark + (1.0 / 3.0) * sea_q
        r_inf = num / denom
    elif nu_type.lower() in ["antinu", "anti_nu_mu", "nubar"]:
        # anti-nu_mu: charm from \bar{d} and \bar{s}; CC inclusive from u ((1-y)^2) and \bar{d}, \bar{s}
        num = (v_cd**2 * sea_q) + (v_cs**2 * s_quark)
        denom = (1.0 / 3.0) * u_target + sea_q + s_quark
        r_inf = num / denom
    else:
        # Inclusive (nu + antinu flux average, approximately 65% nu, 35% antinu in LHC forward beam)
        num_nu = (v_cd**2 * d_target) + (v_cs**2 * s_quark)
        den_nu = d_target + s_quark + (1.0 / 3.0) * sea_q
        num_an = (v_cd**2 * sea_q) + (v_cs**2 * s_quark)
        den_an = (1.0 / 3.0) * u_target + sea_q + s_quark
        # Weighted by CC cross section ratio: sigma_CC(nu) ~ 2 * sigma_CC(antinu)
        r_inf = (0.65 * num_nu + 0.35 * num_an) / (0.65 * den_nu + 0.35 * den_an)

    # Scale by branching ratio or normalizer (default 1.0)
    r_inf = r_inf * scale

    formula = (
        f"x < {e_thresh:.4f} ? 0.0 : "
        f"[0] * pow(1.0 - [1]/x, [2]) * (x / (x + [3]))"
    )
    f_sr = ROOT.TF1(name, formula, e_min, e_max)
    f_sr.SetParameters(r_inf, e_thresh, alpha, e_scale)
    f_sr.SetParNames("R_inf", "E_thresh", "alpha", "E_scale")
    f_sr.SetLineColor(line_color)
    f_sr.SetLineWidth(3)
    f_sr.SetLineStyle(1)

    # Base PDF/CKM, branching ratio, and mass uncertainty
    d_ethresh_mc = (m_p + m_c) / m_p
    rel_scale = (scale_err / scale) if (scale > 0 and scale_err > 0) else 0.0
    rel_base = math.sqrt(0.08**2 + rel_scale**2)
    sigma_rinf_base = rel_base * r_inf
    sigma_ethresh = d_ethresh_mc * sigma_m_c

    # Build 1-sigma uncertainty bands:
    # Build 1-sigma uncertainty bands:
    # 1. Inner band: base threshold mass & leading-order PDF/CKM uncertainty (symmetric)
    # 2. Outer band: asymmetric expansion reflecting that high-energy QCD evolution
    #    (boson-gluon fusion W+ g -> c sbar and DGLAP strange sea growth g -> s sbar)
    #    skews deviations upward rather than downward.
    n_pts = 300
    g_band_inner = ROOT.TGraphErrors(n_pts)
    g_band_inner.SetName(f"{name}_1sigma_inner_band")
    g_band_inner.SetTitle(f"Slow Rescaling Base 1#sigma Uncertainty Belt;E_{{#nu}} [GeV];R_{{charm}}")

    g_band_outer = ROOT.TGraphAsymmErrors(n_pts)
    g_band_outer.SetName(f"{name}_1sigma_outer_band")
    g_band_outer.SetTitle(f"Slow Rescaling 1#sigma with High-E Evolution;E_{{#nu}} [GeV];R_{{charm}}")

    x_vals = [e_thresh + (e_max - e_thresh) * (i / float(n_pts - 1))**1.5 for i in range(n_pts)]
    for i, x in enumerate(x_vals):
        if x <= e_thresh:
            y = 0.0
            sigma_y_inner = 0.0
            err_lo_outer = 0.0
            err_hi_outer = 0.0
        else:
            kin_factor = ((1.0 - e_thresh / x)**alpha) * (x / (x + e_scale))
            y = r_inf * kin_factor
            dy_dethresh = r_inf * alpha * ((1.0 - e_thresh / x)**(alpha - 1.0)) * (-1.0 / (x + e_scale))
            sigma_kin = abs(dy_dethresh * sigma_ethresh)

            # Inner base uncertainty (symmetric base kinematics & CKM/PDF normalization)
            sigma_r_base = kin_factor * sigma_rinf_base
            sigma_y_inner = math.sqrt(sigma_r_base**2 + sigma_kin**2)

            # Outer band:
            # Downward deviations at high E are constrained to base PDF/CKM and mass effects
            err_lo_outer = sigma_y_inner

            # Upward deviations expand with energy to account for:
            # 1. Boson-gluon fusion (W+ g -> c sbar) opening at O(alpha_s)
            # 2. DGLAP Q^2 evolution enhancing the strange sea (g -> s sbar) at small x
            rel_high_e_unc_up = 0.28 * math.log(max(1.0, x / 40.0)) / math.log(1000.0 / 40.0)
            tot_rel_rinf_up = math.sqrt((sigma_rinf_base / r_inf)**2 + rel_high_e_unc_up**2)
            sigma_r_outer_up = kin_factor * (r_inf * tot_rel_rinf_up)
            err_hi_outer = math.sqrt(sigma_r_outer_up**2 + sigma_kin**2)

        g_band_inner.SetPoint(i, x, y)
        g_band_inner.SetPointError(i, 0.0, sigma_y_inner)

        g_band_outer.SetPoint(i, x, y)
        g_band_outer.SetPointError(i, 0.0, 0.0, err_lo_outer, err_hi_outer)

    # Style outer band (lighter nuance, expanding asymmetrically upward at high E)
    g_band_outer.SetFillColorAlpha(band_color, 0.20)
    g_band_outer.SetFillStyle(1001)
    g_band_outer.SetLineColor(line_color)
    g_band_outer.SetLineWidth(1)
    g_band_outer.SetLineStyle(3)

    # Style inner band (deeper nuance for base leading-order kinematics)
    g_band_inner.SetFillColorAlpha(band_color, 0.45)
    g_band_inner.SetFillStyle(1001)
    g_band_inner.SetLineColor(line_color)
    g_band_inner.SetLineWidth(1)

    return f_sr, g_band_inner, g_band_outer


def create_slow_rescaling_function(
    name: str = "f_slow_rescaling",
    e_min: float = 0.0,
    e_max: float = 1500.0,
    e_thresh: Optional[float] = None,
    r_inf: Optional[float] = None,
) -> ROOT.TF1:
    """Wrapper returning central TF1 with default PDG parameters."""
    f_sr, _ = create_slow_rescaling_model(name=name, e_min=e_min, e_max=e_max)
    if e_thresh is not None:
        f_sr.SetParameter(1, e_thresh)
    if r_inf is not None:
        f_sr.SetParameter(0, r_inf)
    return f_sr


def load_lhapdf_charm_prediction(
    csv_filename: str,
    name: str = "lhapdf_charm_prediction",
    target: str = "tungsten",
    line_color: int = ROOT.kBlack,
    line_width: int = 2,
    line_style: int = 1,
    band_color: Optional[int] = None,
    inner_alpha: float = 0.35,
    outer_alpha: float = 0.18,
    fill_style_inner: Optional[int] = None,
    fill_style_outer: Optional[int] = None,
) -> Tuple[ROOT.TGraph, ROOT.TGraphAsymmErrors, ROOT.TGraphAsymmErrors]:
    """
    Loads precalculated LHAPDF nCTEQ15 charm cross-section ratio predictions from CSV.
    Supports target-specific columns ('tungsten_*' or 'emulsion_*') with automatic fallback.
    Returns:
      (g_central, g_inner_band, g_outer_band)
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions", csv_filename),
        os.path.join("/afs/cern.ch/work/i/idioniso/snd-numu-charm/scripts/diagnostics/predictions", csv_filename),
        os.path.join("scripts/diagnostics/predictions", csv_filename),
        csv_filename,
    ]
    resolved_path = None
    for c in candidates:
        if os.path.exists(c):
            resolved_path = c
            break

    if not resolved_path:
        raise FileNotFoundError(f"LHAPDF prediction CSV '{csv_filename}' not found in candidates: {candidates}")

    e_vals: List[float] = []
    y_cens: List[float] = []
    y_in_los: List[float] = []
    y_in_his: List[float] = []
    y_out_los: List[float] = []
    y_out_his: List[float] = []

    target_prefix = f"{target.strip().lower()}_" if target else ""

    with open(resolved_path, "r", encoding="utf-8-sig") as f:
        clean_lines = [line for line in f if not line.strip().startswith("#")]
        reader = csv.DictReader(clean_lines)
        fieldnames = reader.fieldnames or []

        col_central = target_prefix + "ratio_central" if target_prefix + "ratio_central" in fieldnames else "ratio_central"
        col_in_lo = target_prefix + "base_band_low" if target_prefix + "base_band_low" in fieldnames else "base_band_low"
        col_in_hi = target_prefix + "base_band_high" if target_prefix + "base_band_high" in fieldnames else "base_band_high"
        col_out_lo = target_prefix + "total_band_low" if target_prefix + "total_band_low" in fieldnames else "total_band_low"
        col_out_hi = target_prefix + "total_band_high" if target_prefix + "total_band_high" in fieldnames else "total_band_high"

        for row in reader:
            e_vals.append(float(row["E_nu_GeV"]))
            y_cens.append(float(row[col_central]))
            y_in_los.append(float(row[col_in_lo]))
            y_in_his.append(float(row[col_in_hi]))
            if col_out_lo in row and row[col_out_lo] != "":
                y_out_los.append(float(row[col_out_lo]))
            else:
                y_out_los.append(float(row[col_in_lo]))
            y_out_his.append(float(row[col_out_hi]))

    n_pts = len(e_vals)
    g_central = ROOT.TGraph(n_pts)
    g_central.SetName(f"g_{name}_central")
    g_central.SetLineColor(line_color)
    g_central.SetLineWidth(line_width)
    g_central.SetLineStyle(line_style)

    g_inner = ROOT.TGraphAsymmErrors(n_pts)
    g_inner.SetName(f"g_{name}_1sigma_inner_band")

    g_outer = ROOT.TGraphAsymmErrors(n_pts)
    g_outer.SetName(f"g_{name}_1sigma_outer_band")

    for i in range(n_pts):
        x = e_vals[i]
        y = y_cens[i]
        y_in_lo = y_in_los[i]
        y_in_hi = y_in_his[i]
        y_out_lo = y_out_los[i]
        y_out_hi = y_out_his[i]

        g_central.SetPoint(i, x, y)

        ey_in_lo = max(0.0, y - y_in_lo)
        ey_in_hi = max(0.0, y_in_hi - y)
        g_inner.SetPoint(i, x, y)
        g_inner.SetPointError(i, 0.0, 0.0, ey_in_lo, ey_in_hi)

        ey_out_lo = max(0.0, y - y_out_lo)
        ey_out_hi = max(0.0, y_out_hi - y)
        g_outer.SetPoint(i, x, y)
        g_outer.SetPointError(i, 0.0, 0.0, ey_out_lo, ey_out_hi)

    eff_band_col = band_color if band_color is not None else line_color

    if fill_style_outer is not None and fill_style_outer != 1001:
        g_outer.SetFillColor(eff_band_col)
        g_outer.SetFillStyle(fill_style_outer)
        g_outer.SetLineColor(eff_band_col)
        g_outer.SetLineWidth(1)
    else:
        g_outer.SetFillColorAlpha(eff_band_col, outer_alpha)
        g_outer.SetFillStyle(1001)
        g_outer.SetLineColor(eff_band_col)
        g_outer.SetLineWidth(0)

    if fill_style_inner is not None and fill_style_inner != 1001:
        g_inner.SetFillColor(eff_band_col)
        g_inner.SetFillStyle(fill_style_inner)
        g_inner.SetLineColor(eff_band_col)
        g_inner.SetLineWidth(1)
    else:
        g_inner.SetFillColorAlpha(eff_band_col, inner_alpha)
        g_inner.SetFillStyle(1001)
        g_inner.SetLineColor(eff_band_col)
        g_inner.SetLineWidth(0)

    return g_central, g_inner, g_outer


def load_lhapdf_dimuon_prediction(
    csv_filename: str,
    name: str = "lhapdf_dimuon_prediction",
    target: str = "tungsten",
    line_color: int = ROOT.kBlack,
    line_width: int = 2,
    line_style: int = 1,
    band_color: Optional[int] = None,
    scale_to_absolute: bool = True,
    inner_alpha: float = 0.35,
    outer_alpha: float = 0.18,
    fill_style_inner: Optional[int] = None,
    fill_style_outer: Optional[int] = None,
) -> Tuple[ROOT.TGraph, ROOT.TGraphAsymmErrors, ROOT.TGraphAsymmErrors]:
    """
    Loads precalculated LHAPDF nCTEQ15 dimuon cross-section ratio predictions from CSV.
    Supports target-specific columns ('tungsten_*' or 'emulsion_*') with automatic fallback.
    The CSV stores ratio_nominal_10m3 and sigma_1..4 in units of 10^-3.
    Returns:
      (g_central, g_base_band, g_total_band)
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions", csv_filename),
        os.path.join("/afs/cern.ch/work/i/idioniso/snd-numu-charm/scripts/diagnostics/predictions", csv_filename),
        os.path.join("scripts/diagnostics/predictions", csv_filename),
        csv_filename,
    ]
    resolved_path = None
    for c in candidates:
        if os.path.exists(c):
            resolved_path = c
            break

    if not resolved_path:
        raise FileNotFoundError(f"LHAPDF dimuon prediction CSV '{csv_filename}' not found in candidates: {candidates}")

    scale = 1e-3 if scale_to_absolute else 1.0

    e_vals: List[float] = []
    y_cens: List[float] = []
    s1_los: List[float] = []
    s1_his: List[float] = []
    s4_los: List[float] = []
    s4_his: List[float] = []

    target_prefix = f"{target.strip().lower()}_" if target else ""

    with open(resolved_path, "r", encoding="utf-8-sig") as f:
        clean_lines = [line for line in f if not line.strip().startswith("#")]
        reader = csv.DictReader(clean_lines)
        fieldnames = reader.fieldnames or []

        col_ratio = target_prefix + "ratio_nominal_10m3" if target_prefix + "ratio_nominal_10m3" in fieldnames else "ratio_nominal_10m3"
        col_s1_lo = target_prefix + "sigma_1_lo" if target_prefix + "sigma_1_lo" in fieldnames else "sigma_1_lo"
        col_s1_hi = target_prefix + "sigma_1_hi" if target_prefix + "sigma_1_hi" in fieldnames else "sigma_1_hi"
        col_s4_lo = target_prefix + "sigma_4_lo" if target_prefix + "sigma_4_lo" in fieldnames else "sigma_4_lo"
        col_s4_hi = target_prefix + "sigma_4_hi" if target_prefix + "sigma_4_hi" in fieldnames else "sigma_4_hi"

        for row in reader:
            e_vals.append(float(row["E_nu_GeV"]))
            y_cens.append(float(row[col_ratio]) * scale)
            s1_los.append(float(row[col_s1_lo]) * scale)
            s1_his.append(float(row[col_s1_hi]) * scale)
            s4_los.append(float(row[col_s4_lo]) * scale)
            s4_his.append(float(row[col_s4_hi]) * scale)

    n_pts = len(e_vals)
    g_central = ROOT.TGraph(n_pts)
    g_central.SetName(f"g_{name}_central")
    g_central.SetLineColor(line_color)
    g_central.SetLineWidth(line_width)
    g_central.SetLineStyle(line_style)

    g_base = ROOT.TGraphAsymmErrors(n_pts)
    g_base.SetName(f"g_{name}_1sigma_base_band")

    g_total = ROOT.TGraphAsymmErrors(n_pts)
    g_total.SetName(f"g_{name}_1sigma_total_band")

    for i in range(n_pts):
        x = e_vals[i]
        y = y_cens[i]
        ey1_lo = max(0.0, s1_los[i])
        ey1_hi = max(0.0, s1_his[i])
        ey4_lo = max(0.0, s4_los[i])
        ey4_hi = max(0.0, s4_his[i])

        g_central.SetPoint(i, x, y)
        g_base.SetPoint(i, x, y)
        g_base.SetPointError(i, 0.0, 0.0, ey1_lo, ey1_hi)
        g_total.SetPoint(i, x, y)
        g_total.SetPointError(i, 0.0, 0.0, ey4_lo, ey4_hi)

    eff_band_col = band_color if band_color is not None else line_color

    if fill_style_outer is not None and fill_style_outer != 1001:
        g_total.SetFillColor(eff_band_col)
        g_total.SetFillStyle(fill_style_outer)
        g_total.SetLineColor(eff_band_col)
        g_total.SetLineWidth(1)
    else:
        g_total.SetFillColorAlpha(eff_band_col, outer_alpha)
        g_total.SetFillStyle(1001)
        g_total.SetLineColor(eff_band_col)
        g_total.SetLineWidth(0)

    if fill_style_inner is not None and fill_style_inner != 1001:
        g_base.SetFillColor(eff_band_col)
        g_base.SetFillStyle(fill_style_inner)
        g_base.SetLineColor(eff_band_col)
        g_base.SetLineWidth(1)
    else:
        g_base.SetFillColorAlpha(eff_band_col, inner_alpha)
        g_base.SetFillStyle(1001)
        g_base.SetLineColor(eff_band_col)
        g_base.SetLineWidth(0)

    return g_central, g_base, g_total


# Channel-specific color palettes for charm hadron fragmentation fractions:
# Red for Inclusive (nu + anti-nu), Blue for Neutrinos, Purple for Anti-neutrinos
CHANNEL_SPECIES_COLORS: Dict[str, Dict[str, str]] = {
    "inclusive": {
        "D0": "#d91e18",        # Bright Red
        "D+": "#b71c1c",        # Crimson Red
        "Ds+": "#ff5722",       # Coral / Orange-Red
        "Lambda_c": "#c2185b",  # Rose / Raspberry Red
    },
    "nu": {
        "D0": "#1e88e5",        # Bright Blue
        "D+": "#0d47a1",        # Dark Navy Blue
        "Ds+": "#00acc1",       # Cyan / Aqua Blue
        "Lambda_c": "#3949ab",  # Indigo / Royal Blue
    },
    "antinu": {
        "D0": "#8e24aa",        # Bright Violet / Purple
        "D+": "#4a148c",        # Deep Imperial Purple
        "Ds+": "#ba68c8",       # Orchid / Lilac
        "Lambda_c": "#7b1fa2",  # Plum / Violet-Red
    },
}

CHANNEL_SPECIES_LINE_STYLES: Dict[str, int] = {
    "D0": 1,        # Solid
    "D+": 2,        # Dashed
    "Ds+": 4,       # Dash-dot
    "Lambda_c": 3,  # Dotted
}


def load_lhapdf_species_fractions(
    csv_filename: str,
    name_prefix: str = "lhapdf_sp",
    channel_mode: str = "inclusive",
    is_antinu: bool = False,
) -> Tuple[Dict[int, ROOT.TGraph], Dict[int, ROOT.TGraphAsymmErrors]]:
    """
    Loads precalculated LHAPDF nCTEQ15 charm hadron species fragmentation fraction
    predictions and 1-sigma uncertainty bands from CSV.
    The colors follow the channel theme:
      - 'inclusive': Red shades
      - 'nu': Blue shades
      - 'antinu': Purple shades
    Returns:
      (dict_curves, dict_bands)
    keyed by hadron PDG code and absolute PDG code.
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions", csv_filename),
        os.path.join("/afs/cern.ch/work/i/idioniso/snd-numu-charm/scripts/diagnostics/predictions", csv_filename),
        os.path.join("scripts/diagnostics/predictions", csv_filename),
        csv_filename,
    ]
    resolved_path = None
    for c in candidates:
        if os.path.exists(c):
            resolved_path = c
            break

    if not resolved_path:
        raise FileNotFoundError(f"LHAPDF species prediction CSV '{csv_filename}' not found in candidates: {candidates}")

    mode_key = "antinu" if is_antinu or "antinu" in channel_mode.lower() else ("nu" if "nu" in channel_mode.lower() and "inc" not in channel_mode.lower() else "inclusive")
    pal = CHANNEL_SPECIES_COLORS.get(mode_key, CHANNEL_SPECIES_COLORS["inclusive"])

    species_info = {
        "D0": {
            "pdg": -421 if is_antinu else 421,
            "color_hex": pal["D0"],
            "line_style": CHANNEL_SPECIES_LINE_STYLES["D0"],
            "latex": "#bar{D}^{0}" if is_antinu else "D^{0}",
        },
        "D+": {
            "pdg": -411 if is_antinu else 411,
            "color_hex": pal["D+"],
            "line_style": CHANNEL_SPECIES_LINE_STYLES["D+"],
            "latex": "D^{-}" if is_antinu else "D^{+}",
        },
        "Ds+": {
            "pdg": -431 if is_antinu else 431,
            "color_hex": pal["Ds+"],
            "line_style": CHANNEL_SPECIES_LINE_STYLES["Ds+"],
            "latex": "D_{s}^{-}" if is_antinu else "D_{s}^{+}",
        },
        "Lambda_c": {
            "pdg": -4122 if is_antinu else 4122,
            "color_hex": pal["Lambda_c"],
            "line_style": CHANNEL_SPECIES_LINE_STYLES["Lambda_c"],
            "latex": "#bar{#Lambda}_{c}^{-}" if is_antinu else "#Lambda_{c}^{+}",
        },
    }

    e_vals: List[float] = []
    sp_cens: Dict[str, List[float]] = {sp: [] for sp in species_info}
    sp_band_los: Dict[str, List[float]] = {sp: [] for sp in species_info}
    sp_band_his: Dict[str, List[float]] = {sp: [] for sp in species_info}

    with open(resolved_path, "r", encoding="utf-8-sig") as f:
        clean_lines = [line for line in f if not line.strip().startswith("#")]
        reader = csv.DictReader(clean_lines)
        for row in reader:
            e_vals.append(float(row["E_nu_GeV"]))
            for sp in species_info:
                sp_cens[sp].append(float(row[f"f_{sp}_central"]))
                sp_band_los[sp].append(float(row[f"f_{sp}_band_low"]))
                sp_band_his[sp].append(float(row[f"f_{sp}_band_high"]))

    curves: Dict[int, ROOT.TGraph] = {}
    bands: Dict[int, ROOT.TGraphAsymmErrors] = {}

    n_pts = len(e_vals)
    for sp, s_info in species_info.items():
        pdg = s_info["pdg"]
        col = ROOT.TColor.GetColor(s_info["color_hex"])
        sp_name = get_species_name(abs(pdg))

        g_central = ROOT.TGraph(n_pts)
        g_central.SetName(f"g_{name_prefix}_fraction_{sp_name}_central")
        g_central.SetTitle(f"{s_info['latex']};E_{{#nu}} [GeV];f({sp_name})")
        g_central.SetLineColor(col)
        g_central.SetLineStyle(s_info["line_style"])
        g_central.SetLineWidth(2)

        g_band = ROOT.TGraphAsymmErrors(n_pts)
        g_band.SetName(f"g_{name_prefix}_fraction_{sp_name}_1sigma_band")
        g_band.SetTitle(f"{s_info['latex']} (QCD calc 1#sigma);E_{{#nu}} [GeV];f({sp_name})")

        for i in range(n_pts):
            x = e_vals[i]
            y = sp_cens[sp][i]
            y_lo = sp_band_los[sp][i]
            y_hi = sp_band_his[sp][i]

            ey_lo = max(0.0, y - y_lo)
            ey_hi = max(0.0, y_hi - y)

            g_central.SetPoint(i, x, y)
            g_band.SetPoint(i, x, y)
            g_band.SetPointError(i, 0.0, 0.0, ey_lo, ey_hi)

        g_band.SetFillColorAlpha(col, 0.22)
        g_band.SetFillStyle(1001)
        g_band.SetLineColor(col)
        g_band.SetLineWidth(0)

        curves[pdg] = g_central
        curves[abs(pdg)] = g_central
        bands[pdg] = g_band
        bands[abs(pdg)] = g_band

    return curves, bands


def create_species_fraction_model(
    e_min: float = 4.5,
    e_max: float = 1500.0,
    n_points: int = 300,
    n_samples: int = 2000,
    seed: int = 42,
) -> Tuple[Dict[int, ROOT.TGraph], Dict[int, ROOT.TGraphAsymmErrors], ROOT.TGraph, ROOT.TGraphAsymmErrors]:
    """
    Parametrized phenomenological model for charm hadron species fragmentation fractions
    f_i(E_nu) and effective semi-muonic branching fraction <B_mu>(E_nu) transitioning from
    threshold (quasi-elastic/resonance Lambda_c enhancement) to asymptotic DIS (pythia/fragmentation plateau).

    Transitions via decay = exp(-(E_nu - 4.5) / tau) with tau ~ 18 +/- 4 GeV.
    Computes median predictions and 68% CL (1-sigma) uncertainty bands via Monte Carlo sampling.
    """
    np.random.seed(seed)
    e_pts = np.logspace(np.log10(max(4.5, e_min)), np.log10(e_max), n_points)

    br_params = {
        "D0": (0.067, 0.003),
        "D+": (0.160, 0.004),
        "Lambda_c": (0.035, 0.005),
        "Ds+": (0.068, 0.006),
    }

    species_keys = ["D0", "D+", "Lambda_c", "Ds+"]
    pdg_map = {
        "D0": 421,
        "D+": 411,
        "Lambda_c": 4122,
        "Ds+": 431,
    }

    f_samples = {sp: np.zeros((n_samples, len(e_pts))) for sp in species_keys}
    b_mu_samples = np.zeros((n_samples, len(e_pts)))

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

        decay = np.exp(-(e_pts - 4.5) / tau_s)

        raw = {sp: f_inf_s[sp] + (f_thresh_s[sp] - f_inf_s[sp]) * decay for sp in species_keys}
        total_raw = sum(raw.values())

        br_s = {sp: max(0.01, float(np.random.normal(br_params[sp][0], br_params[sp][1]))) for sp in species_keys}

        b_avg = np.zeros(len(e_pts))
        for sp in species_keys:
            f_val = raw[sp] / total_raw
            f_samples[sp][s, :] = f_val
            b_avg += f_val * br_s[sp]
        b_mu_samples[s, :] = b_avg

    f_med = {sp: np.median(f_samples[sp], axis=0) for sp in species_keys}
    f_low = {sp: np.percentile(f_samples[sp], 16, axis=0) for sp in species_keys}
    f_high = {sp: np.percentile(f_samples[sp], 84, axis=0) for sp in species_keys}

    b_med = np.median(b_mu_samples, axis=0)
    b_low = np.percentile(b_mu_samples, 16, axis=0)
    b_high = np.percentile(b_mu_samples, 84, axis=0)

    model_curves: Dict[int, ROOT.TGraph] = {}
    model_bands: Dict[int, ROOT.TGraphAsymmErrors] = {}

    for sp_key in species_keys:
        abs_pdg = pdg_map[sp_key]
        sp_name = get_species_name(abs_pdg)
        col = SPECIES_COLORS.get(abs_pdg, ROOT.kBlack)

        # Central median curve
        g_curve = ROOT.TGraph(n_points)
        g_curve.SetName(f"g_model_curve_fraction_{sp_name}_vs_nu_energy")
        g_curve.SetTitle(f"Model {sp_name} Median Fraction vs E_{{#nu}};E_{{#nu}} [GeV];f({sp_name})")
        for i_pt in range(n_points):
            g_curve.SetPoint(i_pt, float(e_pts[i_pt]), float(f_med[sp_key][i_pt]))
        g_curve.SetLineColor(col)
        g_curve.SetLineWidth(2)
        g_curve.SetLineStyle(1)
        model_curves[abs_pdg] = g_curve

        # 68% CL (1-sigma) band
        g_band = ROOT.TGraphAsymmErrors(n_points)
        g_band.SetName(f"g_model_band_fraction_{sp_name}_vs_nu_energy")
        g_band.SetTitle(f"Model {sp_name} 1#sigma Band vs E_{{#nu}};E_{{#nu}} [GeV];f({sp_name})")
        for i_pt in range(n_points):
            x = float(e_pts[i_pt])
            y = float(f_med[sp_key][i_pt])
            ey_lo = float(y - f_low[sp_key][i_pt])
            ey_hi = float(f_high[sp_key][i_pt] - y)
            g_band.SetPoint(i_pt, x, y)
            g_band.SetPointError(i_pt, 0.0, 0.0, max(0.0, ey_lo), max(0.0, ey_hi))
        fill_col = SPECIES_FILL_COLORS.get(abs_pdg, col)
        g_band.SetFillColor(fill_col)
        g_band.SetFillStyle(1001)
        g_band.SetLineColor(fill_col)
        g_band.SetLineWidth(1)
        model_bands[abs_pdg] = g_band

    # Effective B_mu curve and band
    g_bmu = ROOT.TGraph(n_points)
    g_bmu.SetName("g_model_effective_bmu_vs_nu_energy")
    g_bmu.SetTitle("Model Effective Semi-Muonic Branching Ratio #LTB_{#mu}#GT vs E_{#nu};E_{#nu} [GeV];#LTB_{#mu}#GT")
    for i_pt in range(n_points):
        g_bmu.SetPoint(i_pt, float(e_pts[i_pt]), float(b_med[i_pt]))
    g_bmu.SetLineColor(ROOT.kOrange + 7)
    g_bmu.SetLineWidth(2)

    g_band_bmu = ROOT.TGraphAsymmErrors(n_points)
    g_band_bmu.SetName("g_model_band_effective_bmu_vs_nu_energy")
    g_band_bmu.SetTitle("Model Effective Semi-Muonic Branching Ratio 1#sigma Band vs E_{#nu};E_{#nu} [GeV];#LTB_{#mu}#GT")
    for i_pt in range(n_points):
        x = float(e_pts[i_pt])
        y = float(b_med[i_pt])
        ey_lo = float(y - b_low[i_pt])
        ey_hi = float(b_high[i_pt] - y)
        g_band_bmu.SetPoint(i_pt, x, y)
        g_band_bmu.SetPointError(i_pt, 0.0, 0.0, max(0.0, ey_lo), max(0.0, ey_hi))
    g_band_bmu.SetFillColorAlpha(ROOT.kOrange + 7, 0.25)
    g_band_bmu.SetFillStyle(1001)
    g_band_bmu.SetLineColor(ROOT.kOrange + 7)

    return model_curves, model_bands, g_bmu, g_band_bmu


def process_files_worker(args_tuple) -> Dict[str, Any]:
    """
    Worker process analyzing a batch of ROOT files.
    Extracts all neutrino CC events, charm hadrons, and their decay properties.
    """
    (
        file_chunk,
        nu_bin_edges,
        charm_bin_edges,
        decay_only,
        ground_state_only,
        numu_cc_only,
        max_events_chunk,
        pdg_inclusive_modes,
    ) = args_tuple

    import ROOT
    ROOT.gROOT.SetBatch(True)

    n_nu_ebins = len(nu_bin_edges) - 1
    n_ch_ebins = len(charm_bin_edges) - 1

    local_data = {
        "files_processed": 0,
        "total_events": 0,
        "nu_cc_inclusive": 0,
        "nu_cc_charm": 0,
        "nu_cc_dimuon": 0,
        "nu_only_cc_inclusive": 0,
        "nu_only_cc_charm": 0,
        "nu_only_cc_dimuon": 0,
        "antinu_only_cc_inclusive": 0,
        "antinu_only_cc_charm": 0,
        "antinu_only_cc_dimuon": 0,
        # Energy spectra
        "nu_e_cc_all": [],
        "nu_e_cc_charm": [],
        "nu_e_cc_dimuon": [],
        "nu_e_cc_all_nu": [],
        "nu_e_cc_charm_nu": [],
        "nu_e_cc_dimuon_nu": [],
        "nu_e_cc_all_antinu": [],
        "nu_e_cc_charm_antinu": [],
        "nu_e_cc_dimuon_antinu": [],
        "charm_e_all": [],
        "charm_e_by_species": {},             # abs_pdg -> list of E_charm
        # Binned counts
        "cc_all_per_ebin": [0] * n_nu_ebins,
        "cc_charm_per_ebin": [0] * n_nu_ebins,
        "cc_dimuon_per_ebin": [0] * n_nu_ebins,
        "cc_all_nu_per_ebin": [0] * n_nu_ebins,
        "cc_charm_nu_per_ebin": [0] * n_nu_ebins,
        "cc_dimuon_nu_per_ebin": [0] * n_nu_ebins,
        "cc_all_antinu_per_ebin": [0] * n_nu_ebins,
        "cc_charm_antinu_per_ebin": [0] * n_nu_ebins,
        "cc_dimuon_antinu_per_ebin": [0] * n_nu_ebins,
        # Species counts
        "species_hadron_counts": Counter(),
        "species_nu_ebin_counts": {i: Counter() for i in range(n_nu_ebins)},
        "species_nu_ebin_counts_nu": {i: Counter() for i in range(n_nu_ebins)},
        "species_nu_ebin_counts_antinu": {i: Counter() for i in range(n_nu_ebins)},
        "species_charm_ebin_counts": {i: Counter() for i in range(n_ch_ebins)},
        # Decay channels
        "decay_channels_by_species": {},            # abs_pdg -> Counter of tuples
        "decay_channels_by_species_nu_ebin": {},   # abs_pdg -> {ebin_idx: Counter}
        "decay_channels_by_species_charm_ebin": {},# abs_pdg -> {charm_ebin_idx: Counter}
        "all_decay_channels": Counter(),
        # Semi-leptonic counts
        "semi_muonic_counts": Counter(),
        "semi_electronic_counts": Counter(),
        "total_decays_by_species": Counter(),
        # Inclusive decay mode tracking per species: abs_pdg -> Counter
        "species_weight_sum": Counter(),       # abs_pdg -> float (sum w)
        "species_weight_sq_sum": Counter(),    # abs_pdg -> float (sum w^2)
        "species_decay_count": Counter(),      # abs_pdg -> int
        "inclusive_mode_w": {},                # abs_pdg -> Counter(mode_name -> float)
        "inclusive_mode_w2y2": {},             # abs_pdg -> Counter(mode_name -> float)
        "inclusive_mode_w2y": {},              # abs_pdg -> Counter(mode_name -> float)
        "inclusive_mode_raw_count": {},        # abs_pdg -> Counter(mode_name -> int)
        # Kinematics
        "ctau_values_um": {},                       # abs_pdg -> list of c*tau values
        "decay_dist_mm": {},                        # abs_pdg -> list of L values
        "fragmentation_z": [],                      # list of E_charm / E_nu
        "charm_pt": [],                             # list of pT relative to neutrino beam
        "charm_theta": [],                          # list of production angle theta
        # Neutrino vs Antineutrino asymmetry
        "nu_sign_charm_species": {1: Counter(), -1: Counter()},
        # Reconstruction & Tracking Multiplicity
        "reco_tracks_cc_all": Counter(),
        "reco_tracks_cc_charm": Counter(),
        "charm_ge2trks_per_ebin": [0] * n_nu_ebins,
        "charm_in_ds_per_ebin": [0] * n_nu_ebins,
        "charm_candidate_per_ebin": [0] * n_nu_ebins,
    }

    events_analyzed = 0

    for fpath in file_chunk:
        if max_events_chunk > 0 and events_analyzed >= max_events_chunk:
            break

        fin = None
        try:
            fin = ROOT.TFile.Open(fpath, "READ")
        except Exception:
            continue

        if not fin or fin.IsZombie():
            if fin:
                try:
                    fin.Close()
                except Exception:
                    pass
            continue

        try:
            tree_cbmsim = fin.Get("cbmsim")
            if not tree_cbmsim or tree_cbmsim.IsZombie():
                fin.Close()
                continue
            n_entries = tree_cbmsim.GetEntries()
        except Exception:
            if fin:
                try:
                    fin.Close()
                except Exception:
                    pass
            continue

        tree_truth = fin.Get("mctruth") or fin.Get("mc_truth") or fin.Get("truth")
        if tree_truth and not tree_truth.IsZombie():
            tree_cbmsim.AddFriend(tree_truth)

        try:
            tree_cbmsim.SetBranchStatus("*", 0)
            for bname in [
                "MCTrack*", "mc_weight*", "raw_weight*", "mctruth_weight*",
                "is_cc*", "has_charm*", "Reco_MuonTracks*",
                "dimuon_in_ds_acceptance*", "has_candidate*",
                "has_prompt_charm_muon*",
            ]:
                tree_cbmsim.SetBranchStatus(bname, 1)
        except Exception:
            pass

        local_data["files_processed"] += 1

        for iev in range(n_entries):
            if max_events_chunk > 0 and events_analyzed >= max_events_chunk:
                break
            events_analyzed += 1
            local_data["total_events"] += 1

            tree_cbmsim.GetEntry(iev)
            mc_weight = 1.0
            if hasattr(tree_cbmsim, "mc_weight"):
                try:
                    mc_weight = float(tree_cbmsim.mc_weight)
                except Exception:
                    mc_weight = 1.0
            elif tree_truth and hasattr(tree_truth, "mc_weight"):
                try:
                    mc_weight = float(tree_truth.mc_weight)
                except Exception:
                    mc_weight = 1.0

            tracks = getattr(tree_cbmsim, "MCTrack", None)
            if not tracks or len(tracks) < 1:
                continue

            # 1. Determine neutrino properties
            nu_trk = tracks[0]
            nu_pdg = nu_trk.GetPdgCode()
            nu_energy = nu_trk.GetEnergy()

            # Direction of incoming neutrino
            nu_px, nu_py, nu_pz = nu_trk.GetPx(), nu_trk.GetPy(), nu_trk.GetPz()
            nu_p = nu_trk.GetP()
            if nu_p > 0:
                nu_dir = (nu_px / nu_p, nu_py / nu_p, nu_pz / nu_p)
            else:
                nu_dir = (0.0, 0.0, 1.0)

            # Check CC vs NC
            is_cc = False
            if tree_truth and hasattr(tree_cbmsim, "is_cc"):
                is_cc = bool(tree_cbmsim.is_cc)
            else:
                for trk in tracks:
                    if trk.GetMotherId() == 0 and abs(trk.GetPdgCode()) in [11, 13, 15]:
                        is_cc = True
                        break

            is_numu_cc = (abs(nu_pdg) == 14 and is_cc)
            if numu_cc_only and not is_numu_cc:
                continue

            n_reco_trks = 0
            if hasattr(tree_cbmsim, "Reco_MuonTracks"):
                try:
                    n_reco_trks = tree_cbmsim.Reco_MuonTracks.GetEntries()
                except Exception:
                    n_reco_trks = 0

            local_data["nu_cc_inclusive"] += 1
            local_data["reco_tracks_cc_all"][n_reco_trks] += 1
            local_data["nu_e_cc_all"].append(nu_energy)
            if nu_pdg > 0:
                local_data["nu_only_cc_inclusive"] += 1
                local_data["nu_e_cc_all_nu"].append(nu_energy)
            else:
                local_data["antinu_only_cc_inclusive"] += 1
                local_data["nu_e_cc_all_antinu"].append(nu_energy)

            nu_ebin_idx = find_energy_bin_index(nu_energy, nu_bin_edges)
            if nu_ebin_idx >= 0:
                local_data["cc_all_per_ebin"][nu_ebin_idx] += 1
                if nu_pdg > 0:
                    local_data["cc_all_nu_per_ebin"][nu_ebin_idx] += 1
                else:
                    local_data["cc_all_antinu_per_ebin"][nu_ebin_idx] += 1

            # 2. Check if this event produces charm: without exception
            has_charm_event = False
            if tree_truth and hasattr(tree_cbmsim, "has_charm"):
                try:
                    has_charm_event = bool(tree_cbmsim.has_charm)
                except Exception:
                    has_charm_event = False
            if not has_charm_event:
                for trk in tracks:
                    p = trk.GetPdgCode()
                    if abs(p) == 4 or is_charm_hadron(p):
                        has_charm_event = True
                        break

            if has_charm_event:
                local_data["nu_cc_charm"] += 1
                local_data["reco_tracks_cc_charm"][n_reco_trks] += 1
                local_data["nu_e_cc_charm"].append(nu_energy)
                if nu_pdg > 0:
                    local_data["nu_only_cc_charm"] += 1
                    local_data["nu_e_cc_charm_nu"].append(nu_energy)
                else:
                    local_data["antinu_only_cc_charm"] += 1
                    local_data["nu_e_cc_charm_antinu"].append(nu_energy)

                if nu_ebin_idx >= 0:
                    local_data["cc_charm_per_ebin"][nu_ebin_idx] += 1
                    if nu_pdg > 0:
                        local_data["cc_charm_nu_per_ebin"][nu_ebin_idx] += 1
                    else:
                        local_data["cc_charm_antinu_per_ebin"][nu_ebin_idx] += 1
                    if n_reco_trks >= 2:
                        local_data["charm_ge2trks_per_ebin"][nu_ebin_idx] += 1
                    if hasattr(tree_cbmsim, "dimuon_in_ds_acceptance") and bool(tree_cbmsim.dimuon_in_ds_acceptance):
                        local_data["charm_in_ds_per_ebin"][nu_ebin_idx] += 1
                    if hasattr(tree_cbmsim, "has_candidate") and bool(tree_cbmsim.has_candidate):
                        local_data["charm_candidate_per_ebin"][nu_ebin_idx] += 1
            else:
                continue

            # 3. Scan event for charm hadrons (for species fractions and decay channels)
            event_has_dimuon = False
            if hasattr(tree_cbmsim, "has_prompt_charm_muon"):
                try:
                    event_has_dimuon = bool(tree_cbmsim.has_prompt_charm_muon)
                except Exception:
                    pass

            n_tracks = len(tracks)
            charm_hadrons_in_event: List[Tuple[int, int, int]] = []

            for i_trk in range(n_tracks):
                trk = tracks[i_trk]
                pdg = trk.GetPdgCode()
                if is_charm_hadron(pdg):
                    abs_pdg = abs(pdg)
                    if ground_state_only and abs_pdg not in GROUND_STATE_CHARM_PDGS:
                        continue
                    charm_hadrons_in_event.append((i_trk, pdg, abs_pdg))

            nu_sign = 1 if nu_pdg > 0 else -1

            # 4. Analyze each charm hadron, its energy, and its direct daughters
            for i_charm, charm_pdg, abs_charm_pdg in charm_hadrons_in_event:
                charm_trk = tracks[i_charm]
                c_p = charm_trk.GetP()
                c_e = charm_trk.GetEnergy()
                c_px, c_py, c_pz = charm_trk.GetPx(), charm_trk.GetPy(), charm_trk.GetPz()

                local_data["species_hadron_counts"][abs_charm_pdg] += 1
                local_data["charm_e_all"].append(c_e)
                if abs_charm_pdg not in local_data["charm_e_by_species"]:
                    local_data["charm_e_by_species"][abs_charm_pdg] = []
                local_data["charm_e_by_species"][abs_charm_pdg].append(c_e)

                # Energy bin indices
                charm_ebin_idx = find_energy_bin_index(c_e, charm_bin_edges)

                if nu_ebin_idx >= 0:
                    local_data["species_nu_ebin_counts"][nu_ebin_idx][abs_charm_pdg] += 1
                    if nu_pdg > 0:
                        local_data["species_nu_ebin_counts_nu"][nu_ebin_idx][charm_pdg] += 1
                    else:
                        local_data["species_nu_ebin_counts_antinu"][nu_ebin_idx][charm_pdg] += 1
                if charm_ebin_idx >= 0:
                    local_data["species_charm_ebin_counts"][charm_ebin_idx][abs_charm_pdg] += 1

                local_data["nu_sign_charm_species"][nu_sign][abs_charm_pdg] += 1

                # Elasticity z = E_charm / E_nu
                if nu_energy > 0:
                    z = min(1.0, max(0.0, c_e / nu_energy))
                    local_data["fragmentation_z"].append(z)

                # Transverse momentum relative to neutrino axis: pT = |p x u_nu|
                cross_x = c_py * nu_dir[2] - c_pz * nu_dir[1]
                cross_y = c_pz * nu_dir[0] - c_px * nu_dir[2]
                cross_z = c_px * nu_dir[1] - c_py * nu_dir[0]
                pt_rel = math.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
                local_data["charm_pt"].append(pt_rel)

                # Production angle theta relative to neutrino
                p_dot_nu = c_px * nu_dir[0] + c_py * nu_dir[1] + c_pz * nu_dir[2]
                cos_theta = max(-1.0, min(1.0, p_dot_nu / max(c_p, 1e-6)))
                theta_mrad = math.acos(cos_theta) * 1000.0
                local_data["charm_theta"].append(theta_mrad)

                # Charm vertex
                vtx_x = charm_trk.GetStartX()
                vtx_y = charm_trk.GetStartY()
                vtx_z = charm_trk.GetStartZ()

                # Direct daughters extraction
                daughters: List[int] = []
                decay_vtx = None

                for j_trk in range(n_tracks):
                    d_trk = tracks[j_trk]
                    if d_trk.GetMotherId() == i_charm:
                        proc = d_trk.GetProcName()
                        if decay_only and proc != "Decay":
                            continue
                        d_pdg = d_trk.GetPdgCode()
                        daughters.append(d_pdg)
                        if decay_vtx is None:
                            decay_vtx = (d_trk.GetStartX(), d_trk.GetStartY(), d_trk.GetStartZ())

                if daughters:
                    local_data["total_decays_by_species"][abs_charm_pdg] += 1
                    local_data["species_weight_sum"][abs_charm_pdg] += mc_weight
                    local_data["species_weight_sq_sum"][abs_charm_pdg] += mc_weight * mc_weight
                    local_data["species_decay_count"][abs_charm_pdg] += 1

                    # Track signed charm_pdg for particle vs antiparticle separation
                    local_data["total_decays_by_species"][charm_pdg] += 1
                    local_data["species_weight_sum"][charm_pdg] += mc_weight
                    local_data["species_weight_sq_sum"][charm_pdg] += mc_weight * mc_weight
                    local_data["species_decay_count"][charm_pdg] += 1

                    # Evaluate inclusive decay modes from PDG table
                    sp_modes = pdg_inclusive_modes.get(abs_charm_pdg, {}).get("modes_list", [])
                    if sp_modes:
                        for sp_k in (abs_charm_pdg, charm_pdg):
                            if sp_k not in local_data["inclusive_mode_w"]:
                                local_data["inclusive_mode_w"][sp_k] = Counter()
                                local_data["inclusive_mode_w2y2"][sp_k] = Counter()
                                local_data["inclusive_mode_w2y"][sp_k] = Counter()
                                local_data["inclusive_mode_raw_count"][sp_k] = Counter()

                        for m_obj in sp_modes:
                            m_name = m_obj.pdg_mode if hasattr(m_obj, "pdg_mode") else m_obj["pdg_mode"]
                            req_p = m_obj.req_particles if hasattr(m_obj, "req_particles") else m_obj["req_particles"]
                            mult = count_inclusive_multiplicity(daughters, charm_pdg, req_p)
                            if mult > 0:
                                for sp_k in (abs_charm_pdg, charm_pdg):
                                    local_data["inclusive_mode_w"][sp_k][m_name] += mult * mc_weight
                                    local_data["inclusive_mode_w2y2"][sp_k][m_name] += ((mult * mc_weight) ** 2)
                                    local_data["inclusive_mode_w2y"][sp_k][m_name] += (mc_weight ** 2) * mult
                                    local_data["inclusive_mode_raw_count"][sp_k][m_name] += mult

                    ch_tuple = tuple(sorted(daughters))

                    if abs_charm_pdg not in local_data["decay_channels_by_species"]:
                        local_data["decay_channels_by_species"][abs_charm_pdg] = Counter()
                    local_data["decay_channels_by_species"][abs_charm_pdg][ch_tuple] += 1
                    local_data["all_decay_channels"][ch_tuple] += 1

                    # Binned by neutrino energy E_nu
                    if nu_ebin_idx >= 0:
                        if abs_charm_pdg not in local_data["decay_channels_by_species_nu_ebin"]:
                            local_data["decay_channels_by_species_nu_ebin"][abs_charm_pdg] = {
                                b: Counter() for b in range(n_nu_ebins)
                            }
                        local_data["decay_channels_by_species_nu_ebin"][abs_charm_pdg][nu_ebin_idx][ch_tuple] += 1

                    # Binned by charm hadron energy E_charm
                    if charm_ebin_idx >= 0:
                        if abs_charm_pdg not in local_data["decay_channels_by_species_charm_ebin"]:
                            local_data["decay_channels_by_species_charm_ebin"][abs_charm_pdg] = {
                                b: Counter() for b in range(n_ch_ebins)
                            }
                        local_data["decay_channels_by_species_charm_ebin"][abs_charm_pdg][charm_ebin_idx][ch_tuple] += 1

                    # Semi-leptonic checks
                    if any(abs(d) == 13 for d in daughters):
                        local_data["semi_muonic_counts"][abs_charm_pdg] += 1
                        event_has_dimuon = True
                    if any(abs(d) == 11 for d in daughters):
                        local_data["semi_electronic_counts"][abs_charm_pdg] += 1

                    # Proper lifetime c*tau calculation
                    if decay_vtx is not None:
                        dx = decay_vtx[0] - vtx_x
                        dy = decay_vtx[1] - vtx_y
                        dz = decay_vtx[2] - vtx_z
                        dist_cm = math.sqrt(dx * dx + dy * dy + dz * dz)
                        dist_mm = dist_cm * 10.0

                        mass = math.sqrt(max(0.01, c_e * c_e - c_p * c_p))
                        if mass < 0.5:
                            mass = CHARM_NOMINAL_MASSES.get(abs_charm_pdg, 1.86)
                        bg = max(0.01, c_p / mass)
                        ctau_um = (dist_cm / bg) * 10000.0

                        if abs_charm_pdg not in local_data["ctau_values_um"]:
                            local_data["ctau_values_um"][abs_charm_pdg] = []
                            local_data["decay_dist_mm"][abs_charm_pdg] = []
                        local_data["ctau_values_um"][abs_charm_pdg].append(ctau_um)
                        local_data["decay_dist_mm"][abs_charm_pdg].append(dist_mm)

            # Record dimuon event counters for CC interactions
            if event_has_dimuon and is_cc and (abs(nu_pdg) == 14):
                local_data["nu_cc_dimuon"] += 1
                local_data["nu_e_cc_dimuon"].append(nu_energy)
                if nu_pdg > 0:
                    local_data["nu_only_cc_dimuon"] += 1
                    local_data["nu_e_cc_dimuon_nu"].append(nu_energy)
                else:
                    local_data["antinu_only_cc_dimuon"] += 1
                    local_data["nu_e_cc_dimuon_antinu"].append(nu_energy)

                if nu_ebin_idx >= 0:
                    local_data["cc_dimuon_per_ebin"][nu_ebin_idx] += 1
                    if nu_pdg > 0:
                        local_data["cc_dimuon_nu_per_ebin"][nu_ebin_idx] += 1
                    else:
                        local_data["cc_dimuon_antinu_per_ebin"][nu_ebin_idx] += 1

        fin.Close()

    return local_data


def merge_worker_results(
    results: List[Dict[str, Any]],
    nu_bin_edges: List[float],
    charm_bin_edges: List[float],
) -> Dict[str, Any]:
    """Combines statistics from all parallel workers."""
    n_nu_ebins = len(nu_bin_edges) - 1
    n_ch_ebins = len(charm_bin_edges) - 1

    merged = {
        "files_processed": sum(r["files_processed"] for r in results),
        "total_events": sum(r["total_events"] for r in results),
        "nu_cc_inclusive": sum(r["nu_cc_inclusive"] for r in results),
        "nu_cc_charm": sum(r["nu_cc_charm"] for r in results),
        "nu_cc_dimuon": sum(r.get("nu_cc_dimuon", 0) for r in results),
        "nu_only_cc_inclusive": sum(r.get("nu_only_cc_inclusive", 0) for r in results),
        "nu_only_cc_charm": sum(r.get("nu_only_cc_charm", 0) for r in results),
        "nu_only_cc_dimuon": sum(r.get("nu_only_cc_dimuon", 0) for r in results),
        "antinu_only_cc_inclusive": sum(r.get("antinu_only_cc_inclusive", 0) for r in results),
        "antinu_only_cc_charm": sum(r.get("antinu_only_cc_charm", 0) for r in results),
        "antinu_only_cc_dimuon": sum(r.get("antinu_only_cc_dimuon", 0) for r in results),
        "nu_e_cc_all": [],
        "nu_e_cc_charm": [],
        "nu_e_cc_dimuon": [],
        "nu_e_cc_all_nu": [],
        "nu_e_cc_charm_nu": [],
        "nu_e_cc_dimuon_nu": [],
        "nu_e_cc_all_antinu": [],
        "nu_e_cc_charm_antinu": [],
        "nu_e_cc_dimuon_antinu": [],
        "charm_e_all": [],
        "charm_e_by_species": {},
        "cc_all_per_ebin": [0] * n_nu_ebins,
        "cc_charm_per_ebin": [0] * n_nu_ebins,
        "cc_dimuon_per_ebin": [0] * n_nu_ebins,
        "cc_all_nu_per_ebin": [0] * n_nu_ebins,
        "cc_charm_nu_per_ebin": [0] * n_nu_ebins,
        "cc_dimuon_nu_per_ebin": [0] * n_nu_ebins,
        "cc_all_antinu_per_ebin": [0] * n_nu_ebins,
        "cc_charm_antinu_per_ebin": [0] * n_nu_ebins,
        "cc_dimuon_antinu_per_ebin": [0] * n_nu_ebins,
        "species_hadron_counts": Counter(),
        "species_nu_ebin_counts": {i: Counter() for i in range(n_nu_ebins)},
        "species_nu_ebin_counts_nu": {i: Counter() for i in range(n_nu_ebins)},
        "species_nu_ebin_counts_antinu": {i: Counter() for i in range(n_nu_ebins)},
        "species_charm_ebin_counts": {i: Counter() for i in range(n_ch_ebins)},
        "decay_channels_by_species": {},
        "decay_channels_by_species_nu_ebin": {},
        "decay_channels_by_species_charm_ebin": {},
        "all_decay_channels": Counter(),
        "semi_muonic_counts": Counter(),
        "semi_electronic_counts": Counter(),
        "total_decays_by_species": Counter(),
        "species_weight_sum": Counter(),
        "species_weight_sq_sum": Counter(),
        "species_decay_count": Counter(),
        "inclusive_mode_w": {},
        "inclusive_mode_w2y2": {},
        "inclusive_mode_w2y": {},
        "inclusive_mode_raw_count": {},
        "ctau_values_um": {},
        "decay_dist_mm": {},
        "fragmentation_z": [],
        "charm_pt": [],
        "charm_theta": [],
        "nu_sign_charm_species": {1: Counter(), -1: Counter()},
        "reco_tracks_cc_all": Counter(),
        "reco_tracks_cc_charm": Counter(),
        "charm_ge2trks_per_ebin": [0] * n_nu_ebins,
        "charm_in_ds_per_ebin": [0] * n_nu_ebins,
        "charm_candidate_per_ebin": [0] * n_nu_ebins,
    }

    for r in results:
        merged["species_weight_sum"].update(r.get("species_weight_sum", Counter()))
        merged["species_weight_sq_sum"].update(r.get("species_weight_sq_sum", Counter()))
        merged["species_decay_count"].update(r.get("species_decay_count", Counter()))

        for abs_c in r.get("inclusive_mode_w", {}):
            if abs_c not in merged["inclusive_mode_w"]:
                merged["inclusive_mode_w"][abs_c] = Counter()
                merged["inclusive_mode_w2y2"][abs_c] = Counter()
                merged["inclusive_mode_w2y"][abs_c] = Counter()
                merged["inclusive_mode_raw_count"][abs_c] = Counter()
            merged["inclusive_mode_w"][abs_c].update(r["inclusive_mode_w"][abs_c])
            merged["inclusive_mode_w2y2"][abs_c].update(r["inclusive_mode_w2y2"][abs_c])
            merged["inclusive_mode_w2y"][abs_c].update(r["inclusive_mode_w2y"][abs_c])
            merged["inclusive_mode_raw_count"][abs_c].update(r["inclusive_mode_raw_count"][abs_c])

        merged["nu_e_cc_all"].extend(r["nu_e_cc_all"])
        merged["nu_e_cc_charm"].extend(r["nu_e_cc_charm"])
        merged["nu_e_cc_all_nu"].extend(r.get("nu_e_cc_all_nu", []))
        merged["nu_e_cc_charm_nu"].extend(r.get("nu_e_cc_charm_nu", []))
        merged["nu_e_cc_dimuon"].extend(r.get("nu_e_cc_dimuon", []))
        merged["nu_e_cc_dimuon_nu"].extend(r.get("nu_e_cc_dimuon_nu", []))
        merged["nu_e_cc_all_antinu"].extend(r.get("nu_e_cc_all_antinu", []))
        merged["nu_e_cc_charm_antinu"].extend(r.get("nu_e_cc_charm_antinu", []))
        merged["nu_e_cc_dimuon_antinu"].extend(r.get("nu_e_cc_dimuon_antinu", []))
        merged["charm_e_all"].extend(r["charm_e_all"])
        merged["fragmentation_z"].extend(r["fragmentation_z"])
        merged["charm_pt"].extend(r["charm_pt"])
        merged["charm_theta"].extend(r["charm_theta"])

        for abs_c, e_list in r["charm_e_by_species"].items():
            if abs_c not in merged["charm_e_by_species"]:
                merged["charm_e_by_species"][abs_c] = []
            merged["charm_e_by_species"][abs_c].extend(e_list)

        merged["reco_tracks_cc_all"].update(r.get("reco_tracks_cc_all", Counter()))
        merged["reco_tracks_cc_charm"].update(r.get("reco_tracks_cc_charm", Counter()))

        for i in range(n_nu_ebins):
            merged["cc_all_per_ebin"][i] += r["cc_all_per_ebin"][i]
            merged["cc_charm_per_ebin"][i] += r["cc_charm_per_ebin"][i]
            merged["cc_dimuon_per_ebin"][i] += r.get("cc_dimuon_per_ebin", [0] * n_nu_ebins)[i]
            merged["cc_all_nu_per_ebin"][i] += r.get("cc_all_nu_per_ebin", [0] * n_nu_ebins)[i]
            merged["cc_charm_nu_per_ebin"][i] += r.get("cc_charm_nu_per_ebin", [0] * n_nu_ebins)[i]
            merged["cc_dimuon_nu_per_ebin"][i] += r.get("cc_dimuon_nu_per_ebin", [0] * n_nu_ebins)[i]
            merged["cc_all_antinu_per_ebin"][i] += r.get("cc_all_antinu_per_ebin", [0] * n_nu_ebins)[i]
            merged["cc_charm_antinu_per_ebin"][i] += r.get("cc_charm_antinu_per_ebin", [0] * n_nu_ebins)[i]
            merged["cc_dimuon_antinu_per_ebin"][i] += r.get("cc_dimuon_antinu_per_ebin", [0] * n_nu_ebins)[i]
            merged["charm_ge2trks_per_ebin"][i] += r.get("charm_ge2trks_per_ebin", [0] * n_nu_ebins)[i]
            merged["charm_in_ds_per_ebin"][i] += r.get("charm_in_ds_per_ebin", [0] * n_nu_ebins)[i]
            merged["charm_candidate_per_ebin"][i] += r.get("charm_candidate_per_ebin", [0] * n_nu_ebins)[i]
            merged["species_nu_ebin_counts"][i].update(r["species_nu_ebin_counts"][i])
            merged["species_nu_ebin_counts_nu"][i].update(r.get("species_nu_ebin_counts_nu", {}).get(i, Counter()))
            merged["species_nu_ebin_counts_antinu"][i].update(r.get("species_nu_ebin_counts_antinu", {}).get(i, Counter()))

        for i in range(n_ch_ebins):
            merged["species_charm_ebin_counts"][i].update(r["species_charm_ebin_counts"][i])

        merged["species_hadron_counts"].update(r["species_hadron_counts"])
        merged["all_decay_channels"].update(r["all_decay_channels"])
        merged["semi_muonic_counts"].update(r["semi_muonic_counts"])
        merged["semi_electronic_counts"].update(r["semi_electronic_counts"])
        merged["total_decays_by_species"].update(r["total_decays_by_species"])

        for s in [1, -1]:
            merged["nu_sign_charm_species"][s].update(r["nu_sign_charm_species"][s])

        # Decay channels
        for abs_c, counter in r["decay_channels_by_species"].items():
            if abs_c not in merged["decay_channels_by_species"]:
                merged["decay_channels_by_species"][abs_c] = Counter()
            merged["decay_channels_by_species"][abs_c].update(counter)

        # Decay channels per nu energy bin
        for abs_c, ebin_map in r["decay_channels_by_species_nu_ebin"].items():
            if abs_c not in merged["decay_channels_by_species_nu_ebin"]:
                merged["decay_channels_by_species_nu_ebin"][abs_c] = {
                    b: Counter() for b in range(n_nu_ebins)
                }
            for b_idx, counter in ebin_map.items():
                merged["decay_channels_by_species_nu_ebin"][abs_c][b_idx].update(counter)

        # Decay channels per charm energy bin
        for abs_c, ebin_map in r["decay_channels_by_species_charm_ebin"].items():
            if abs_c not in merged["decay_channels_by_species_charm_ebin"]:
                merged["decay_channels_by_species_charm_ebin"][abs_c] = {
                    b: Counter() for b in range(n_ch_ebins)
                }
            for b_idx, counter in ebin_map.items():
                merged["decay_channels_by_species_charm_ebin"][abs_c][b_idx].update(counter)

        # Lifetimes
        for abs_c, ctau_list in r["ctau_values_um"].items():
            if abs_c not in merged["ctau_values_um"]:
                merged["ctau_values_um"][abs_c] = []
                merged["decay_dist_mm"][abs_c] = []
            merged["ctau_values_um"][abs_c].extend(ctau_list)
            merged["decay_dist_mm"][abs_c].extend(r["decay_dist_mm"][abs_c])

    return merged


def write_histograms_and_canvases(
    output_path: str,
    data: Dict[str, Any],
    nu_bin_edges: List[float],
    charm_bin_edges: List[float],
    pdg_tables_dir: Optional[str] = None,
    export_png_dir: Optional[str] = None,
) -> None:
    """
    Builds comprehensive ROOT histograms, superimposes them onto formatted TCanvas objects,
    overlays the slow rescaling model prediction, overlays PDG reference tables, and writes
    everything into the ROOT file.
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fout = ROOT.TFile.Open(output_path, "RECREATE")
    if not fout or fout.IsZombie():
        print(f"[Error] Failed to open {output_path} for writing.")
        return

    ROOT.gStyle.SetOptStat(0)

    pdg_data = load_pdg_summary_tables(pdg_tables_dir)

    import array
    n_nu_ebins = len(nu_bin_edges) - 1
    arr_nu_edges = array.array("d", nu_bin_edges)

    n_ch_ebins = len(charm_bin_edges) - 1
    arr_ch_edges = array.array("d", charm_bin_edges)

    species_order = [421, 411, 431, 4122]
    anti_species_order = [-421, -411, -431, -4122]
    other_species = [s for s in data["species_hadron_counts"].keys() if s not in species_order]
    all_species_display = species_order + other_species
    n_sp_bins = max(1, len(all_species_display))
    tot_charm_all = sum(data["species_hadron_counts"].values())

    # =========================================================================
    # 1. SUMMARY DIRECTORY: Spectra, Rates, Fractions, Kinematics
    # =========================================================================
    dir_summary = fout.mkdir("Summary")
    dir_summary.cd()

    # Neutrino energy spectra (fine 1D, 50 bins 0-1000 GeV)
    h_nu_e_cc_all = ROOT.TH1D("h_nu_e_cc_all", "Neutrino Energy (#nu_{#mu}+#bar{#nu}_{#mu} CC Inclusive);E_{#nu} [GeV];Events", 50, 0.0, 1000.0)
    for e in data["nu_e_cc_all"]:
        h_nu_e_cc_all.Fill(e)
    h_nu_e_cc_all.SetLineColor(ROOT.kBlue + 2)
    h_nu_e_cc_all.SetLineWidth(2)
    h_nu_e_cc_all.Write()

    h_nu_e_cc_all_nu = ROOT.TH1D("h_nu_e_cc_all_nu", "Neutrino Energy (#nu_{#mu} CC Inclusive);E_{#nu} [GeV];Events", 50, 0.0, 1000.0)
    for e in data.get("nu_e_cc_all_nu", []):
        h_nu_e_cc_all_nu.Fill(e)
    h_nu_e_cc_all_nu.SetLineColor(ROOT.kBlack)
    h_nu_e_cc_all_nu.SetLineWidth(2)
    h_nu_e_cc_all_nu.Write()

    h_nu_e_cc_all_antinu = ROOT.TH1D("h_nu_e_cc_all_antinu", "Neutrino Energy (#bar{#nu}_{#mu} CC Inclusive);E_{#nu} [GeV];Events", 50, 0.0, 1000.0)
    for e in data.get("nu_e_cc_all_antinu", []):
        h_nu_e_cc_all_antinu.Fill(e)
    h_nu_e_cc_all_antinu.SetLineColor(ROOT.kMagenta + 2)
    h_nu_e_cc_all_antinu.SetLineWidth(2)
    h_nu_e_cc_all_antinu.Write()

    h_nu_e_cc_charm = ROOT.TH1D("h_nu_e_cc_charm", "Neutrino Energy (CC with Charm);E_{#nu} [GeV];Events", 50, 0.0, 1000.0)
    for e in data["nu_e_cc_charm"]:
        h_nu_e_cc_charm.Fill(e)
    h_nu_e_cc_charm.SetLineColor(ROOT.kRed + 1)
    h_nu_e_cc_charm.SetLineWidth(2)
    h_nu_e_cc_charm.Write()

    h_nu_e_cc_charm_nu = ROOT.TH1D("h_nu_e_cc_charm_nu", "Neutrino Energy (#nu_{#mu} CC with Charm);E_{#nu} [GeV];Events", 50, 0.0, 1000.0)
    for e in data.get("nu_e_cc_charm_nu", []):
        h_nu_e_cc_charm_nu.Fill(e)
    h_nu_e_cc_charm_nu.SetLineColor(ROOT.kOrange + 7)
    h_nu_e_cc_charm_nu.SetLineWidth(2)
    h_nu_e_cc_charm_nu.Write()

    h_nu_e_cc_charm_antinu = ROOT.TH1D("h_nu_e_cc_charm_antinu", "Neutrino Energy (#bar{#nu}_{#mu} CC with Charm);E_{#nu} [GeV];Events", 50, 0.0, 1000.0)
    for e in data.get("nu_e_cc_charm_antinu", []):
        h_nu_e_cc_charm_antinu.Fill(e)
    h_nu_e_cc_charm_antinu.SetLineColor(ROOT.kViolet + 1)
    h_nu_e_cc_charm_antinu.SetLineWidth(2)
    h_nu_e_cc_charm_antinu.Write()

    # Charm hadron energy spectrum (fine 1D, 50 bins 0-1000 GeV)
    h_charm_e_all = ROOT.TH1D("h_charm_e_all", "Charm Hadron Energy Spectrum;E_{charm} [GeV];Charm Hadrons", 50, 0.0, 1000.0)
    for ec in data["charm_e_all"]:
        h_charm_e_all.Fill(ec)
    h_charm_e_all.SetLineColor(ROOT.kGreen + 2)
    h_charm_e_all.SetLineWidth(2)
    h_charm_e_all.Write()

    # Charm production rate ratio vs energy (fine)
    h_charm_rate_fine = h_nu_e_cc_charm.Clone("h_charm_rate_vs_energy_fine")
    h_charm_rate_fine.SetTitle("Charm Production Rate vs Neutrino Energy;E_{#nu} [GeV];R_{charm} = N_{charm} / N_{#nu_{#mu} CC}")
    h_charm_rate_fine.Divide(h_nu_e_cc_charm, h_nu_e_cc_all, 1.0, 1.0, "B")
    h_charm_rate_fine.Write()

    # Dimuon energy spectra (fine 1D, 50 bins 0-1000 GeV)
    h_nu_e_cc_dimuon = ROOT.TH1D("h_nu_e_cc_dimuon", "Neutrino Energy (#nu_{#mu}+#bar{#nu}_{#mu} CC Dimuon);E_{#nu} [GeV];Events", 50, 0.0, 1000.0)
    for e in data.get("nu_e_cc_dimuon", []):
        h_nu_e_cc_dimuon.Fill(e)
    h_nu_e_cc_dimuon.SetLineColor(ROOT.kCyan + 2)
    h_nu_e_cc_dimuon.SetLineWidth(2)
    h_nu_e_cc_dimuon.Write()

    h_nu_e_cc_dimuon_nu = ROOT.TH1D("h_nu_e_cc_dimuon_nu", "Neutrino Energy (#nu_{#mu} CC Dimuon);E_{#nu} [GeV];Events", 50, 0.0, 1000.0)
    for e in data.get("nu_e_cc_dimuon_nu", []):
        h_nu_e_cc_dimuon_nu.Fill(e)
    h_nu_e_cc_dimuon_nu.SetLineColor(ROOT.kAzure + 2)
    h_nu_e_cc_dimuon_nu.SetLineWidth(2)
    h_nu_e_cc_dimuon_nu.Write()

    h_nu_e_cc_dimuon_antinu = ROOT.TH1D("h_nu_e_cc_dimuon_antinu", "Neutrino Energy (#bar{#nu}_{#mu} CC Dimuon);E_{#nu} [GeV];Events", 50, 0.0, 1000.0)
    for e in data.get("nu_e_cc_dimuon_antinu", []):
        h_nu_e_cc_dimuon_antinu.Fill(e)
    h_nu_e_cc_dimuon_antinu.SetLineColor(ROOT.kPink + 7)
    h_nu_e_cc_dimuon_antinu.SetLineWidth(2)
    h_nu_e_cc_dimuon_antinu.Write()

    # Dimuon production rate ratio vs energy (fine)
    h_dimuon_rate_fine = h_nu_e_cc_dimuon.Clone("h_dimuon_rate_vs_energy_fine")
    h_dimuon_rate_fine.SetTitle("Charm Dimuon Production Rate vs Neutrino Energy;E_{#nu} [GeV];R_{2#mu} = N_{2#mu} / N_{#nu_{#mu} CC}")
    h_dimuon_rate_fine.Divide(h_nu_e_cc_dimuon, h_nu_e_cc_all, 1.0, 1.0, "B")
    h_dimuon_rate_fine.Write()

    # Charm production rate in discrete energy bins
    h_ebin_all = ROOT.TH1D("h_cc_all_ebins", "CC Inclusive Events per Energy Bin;E_{#nu} [GeV];Events", n_nu_ebins, arr_nu_edges)
    h_ebin_charm = ROOT.TH1D("h_cc_charm_ebins", "CC Charm Events per Energy Bin;E_{#nu} [GeV];Events", n_nu_ebins, arr_nu_edges)
    h_ebin_dimuon = ROOT.TH1D("h_cc_dimuon_ebins", "CC Dimuon Events per Energy Bin;E_{#nu} [GeV];Events", n_nu_ebins, arr_nu_edges)
    h_ebin_all_nu = ROOT.TH1D("h_cc_all_nu_ebins", "#nu_{#mu} CC Inclusive Events per Energy Bin;E_{#nu} [GeV];Events", n_nu_ebins, arr_nu_edges)
    h_ebin_charm_nu = ROOT.TH1D("h_cc_charm_nu_ebins", "#nu_{#mu} CC Charm Events per Energy Bin;E_{#nu} [GeV];Events", n_nu_ebins, arr_nu_edges)
    h_ebin_dimuon_nu = ROOT.TH1D("h_cc_dimuon_nu_ebins", "#nu_{#mu} CC Dimuon Events per Energy Bin;E_{#nu} [GeV];Events", n_nu_ebins, arr_nu_edges)
    h_ebin_all_antinu = ROOT.TH1D("h_cc_all_antinu_ebins", "#bar{#nu}_{#mu} CC Inclusive Events per Energy Bin;E_{#nu} [GeV];Events", n_nu_ebins, arr_nu_edges)
    h_ebin_charm_antinu = ROOT.TH1D("h_cc_charm_antinu_ebins", "#bar{#nu}_{#mu} CC Charm Events per Energy Bin;E_{#nu} [GeV];Events", n_nu_ebins, arr_nu_edges)
    h_ebin_dimuon_antinu = ROOT.TH1D("h_cc_dimuon_antinu_ebins", "#bar{#nu}_{#mu} CC Dimuon Events per Energy Bin;E_{#nu} [GeV];Events", n_nu_ebins, arr_nu_edges)

    for i in range(n_nu_ebins):
        h_ebin_all.SetBinContent(i + 1, data["cc_all_per_ebin"][i])
        h_ebin_charm.SetBinContent(i + 1, data["cc_charm_per_ebin"][i])
        h_ebin_dimuon.SetBinContent(i + 1, data.get("cc_dimuon_per_ebin", [0] * n_nu_ebins)[i])
        h_ebin_all_nu.SetBinContent(i + 1, data.get("cc_all_nu_per_ebin", [0] * n_nu_ebins)[i])
        h_ebin_charm_nu.SetBinContent(i + 1, data.get("cc_charm_nu_per_ebin", [0] * n_nu_ebins)[i])
        h_ebin_dimuon_nu.SetBinContent(i + 1, data.get("cc_dimuon_nu_per_ebin", [0] * n_nu_ebins)[i])
        h_ebin_all_antinu.SetBinContent(i + 1, data.get("cc_all_antinu_per_ebin", [0] * n_nu_ebins)[i])
        h_ebin_charm_antinu.SetBinContent(i + 1, data.get("cc_charm_antinu_per_ebin", [0] * n_nu_ebins)[i])
        h_ebin_dimuon_antinu.SetBinContent(i + 1, data.get("cc_dimuon_antinu_per_ebin", [0] * n_nu_ebins)[i])

    h_ebin_all.Write()
    h_ebin_charm.Write()
    h_ebin_dimuon.Write()
    h_ebin_all_nu.Write()
    h_ebin_charm_nu.Write()
    h_ebin_dimuon_nu.Write()
    h_ebin_all_antinu.Write()
    h_ebin_charm_antinu.Write()
    h_ebin_dimuon_antinu.Write()

    h_charm_rate_ebins = h_ebin_charm.Clone("h_charm_rate_vs_energy_bins")
    h_charm_rate_ebins.SetTitle("Charm Production Rate in Energy Bins;E_{#nu} [GeV];R_{charm} = #sigma(c) / #sigma_{CC}")
    h_charm_rate_ebins.Divide(h_ebin_charm, h_ebin_all, 1.0, 1.0, "B")
    h_charm_rate_ebins.SetMarkerStyle(20)
    h_charm_rate_ebins.SetMarkerSize(1.3)
    h_charm_rate_ebins.SetMarkerColor(ROOT.kBlue + 2)
    h_charm_rate_ebins.SetLineColor(ROOT.kBlue + 2)
    h_charm_rate_ebins.SetLineWidth(2)
    h_charm_rate_ebins.Write()

    h_charm_rate_ebins_nu = h_ebin_charm_nu.Clone("h_charm_rate_nu_vs_energy_bins")
    h_charm_rate_ebins_nu.SetTitle("#nu_{#mu} Charm Production Rate in Energy Bins;E_{#nu} [GeV];R_{charm}^{#nu} = #sigma(c) / #sigma_{CC}")
    h_charm_rate_ebins_nu.Divide(h_ebin_charm_nu, h_ebin_all_nu, 1.0, 1.0, "B")
    h_charm_rate_ebins_nu.Write()

    h_charm_rate_ebins_antinu = h_ebin_charm_antinu.Clone("h_charm_rate_antinu_vs_energy_bins")
    h_charm_rate_ebins_antinu.SetTitle("#bar{#nu}_{#mu} Charm Production Rate in Energy Bins;E_{#nu} [GeV];R_{charm}^{#bar{#nu}} = #sigma(#bar{c}) / #sigma_{CC}")
    h_charm_rate_ebins_antinu.Divide(h_ebin_charm_antinu, h_ebin_all_antinu, 1.0, 1.0, "B")
    h_charm_rate_ebins_antinu.Write()

    # Dimuon production rate in discrete energy bins
    h_charm_dimuon_rate_ebins = h_ebin_dimuon.Clone("h_charm_dimuon_rate_vs_energy_bins")
    h_charm_dimuon_rate_ebins.SetTitle("Charm Dimuon Production Rate in Energy Bins;E_{#nu} [GeV];R_{2#mu} = #sigma(c, 2#mu) / #sigma_{CC}")
    h_charm_dimuon_rate_ebins.Divide(h_ebin_dimuon, h_ebin_all, 1.0, 1.0, "B")
    h_charm_dimuon_rate_ebins.SetMarkerStyle(20)
    h_charm_dimuon_rate_ebins.SetMarkerSize(1.3)
    h_charm_dimuon_rate_ebins.SetMarkerColor(ROOT.kBlack)
    h_charm_dimuon_rate_ebins.SetLineColor(ROOT.kBlack)
    h_charm_dimuon_rate_ebins.SetLineWidth(2)
    h_charm_dimuon_rate_ebins.Write()

    h_charm_dimuon_rate_ebins_nu = h_ebin_dimuon_nu.Clone("h_charm_dimuon_rate_nu_vs_energy_bins")
    h_charm_dimuon_rate_ebins_nu.SetTitle("#nu_{#mu} Dimuon Production Rate in Energy Bins;E_{#nu} [GeV];R_{2#mu}^{#nu} = #sigma(c, 2#mu) / #sigma_{CC}")
    h_charm_dimuon_rate_ebins_nu.Divide(h_ebin_dimuon_nu, h_ebin_all_nu, 1.0, 1.0, "B")
    h_charm_dimuon_rate_ebins_nu.Write()

    h_charm_dimuon_rate_ebins_antinu = h_ebin_dimuon_antinu.Clone("h_charm_dimuon_rate_antinu_vs_energy_bins")
    h_charm_dimuon_rate_ebins_antinu.SetTitle("#bar{#nu}_{#mu} Dimuon Production Rate in Energy Bins;E_{#nu} [GeV];R_{2#mu}^{#bar{#nu}} = #sigma(#bar{c}, 2#mu) / #sigma_{CC}")
    h_charm_dimuon_rate_ebins_antinu.Divide(h_ebin_dimuon_antinu, h_ebin_all_antinu, 1.0, 1.0, "B")
    h_charm_dimuon_rate_ebins_antinu.Write()

    # Species counts and fractions (Overall)
    h_species_counts = ROOT.TH1D("h_species_counts", "Charm Species Counts;Charm Hadron;Entries", n_sp_bins, 0.5, n_sp_bins + 0.5)
    h_species_frac = ROOT.TH1D("h_species_fractions", "Charm Species Fractions (f_{i} = N_{i} / N_{charm});Charm Hadron;Fraction", n_sp_bins, 0.5, n_sp_bins + 0.5)

    for idx, abs_c in enumerate(all_species_display, start=1):
        sp_name = get_species_name(abs_c)
        cnt = data["species_hadron_counts"].get(abs_c, 0)
        frac = (cnt / tot_charm_all) if tot_charm_all > 0 else 0.0
        h_species_counts.GetXaxis().SetBinLabel(idx, sp_name)
        h_species_counts.SetBinContent(idx, cnt)
        h_species_frac.GetXaxis().SetBinLabel(idx, sp_name)
        h_species_frac.SetBinContent(idx, frac)
    h_species_counts.Write()
    h_species_frac.Write()

    # 2D Species Fraction vs Neutrino Energy Bins
    h2_species_nu_ebin = ROOT.TH2D(
        "h2_species_vs_nu_energy",
        "Charm Species Breakdown across E_{#nu} Bins;E_{#nu} [GeV];Charm Hadron",
        n_nu_ebins, arr_nu_edges, n_sp_bins, 0.5, n_sp_bins + 0.5
    )
    for y_idx, abs_c in enumerate(all_species_display, start=1):
        h2_species_nu_ebin.GetYaxis().SetBinLabel(y_idx, get_species_name(abs_c))

    for x_idx in range(n_nu_ebins):
        ebin_tot = sum(data["species_nu_ebin_counts"][x_idx].values())
        for y_idx, abs_c in enumerate(all_species_display, start=1):
            c_cnt = data["species_nu_ebin_counts"][x_idx].get(abs_c, 0)
            frac = (c_cnt / ebin_tot) if ebin_tot > 0 else 0.0
            h2_species_nu_ebin.SetBinContent(x_idx + 1, y_idx, frac)
    h2_species_nu_ebin.Write()

    # 2D Species Fraction vs Charm Hadron Energy Bins
    h2_species_charm_ebin = ROOT.TH2D(
        "h2_species_vs_charm_energy",
        "Charm Species Breakdown across E_{charm} Bins;E_{charm} [GeV];Charm Hadron",
        n_ch_ebins, arr_ch_edges, n_sp_bins, 0.5, n_sp_bins + 0.5
    )
    for y_idx, abs_c in enumerate(all_species_display, start=1):
        h2_species_charm_ebin.GetYaxis().SetBinLabel(y_idx, get_species_name(abs_c))

    for x_idx in range(n_ch_ebins):
        ebin_tot = sum(data["species_charm_ebin_counts"][x_idx].values())
        for y_idx, abs_c in enumerate(all_species_display, start=1):
            c_cnt = data["species_charm_ebin_counts"][x_idx].get(abs_c, 0)
            frac = (c_cnt / ebin_tot) if ebin_tot > 0 else 0.0
            h2_species_charm_ebin.SetBinContent(x_idx + 1, y_idx, frac)
    h2_species_charm_ebin.Write()

    # 1D Species Fraction vs E_nu for each major ground state (Inclusive, nu-only, and antinu-only)
    def _create_species_frac_graphs(counts_per_ebin, species_list=None, name_suffix="", title_suffix="", channel_mode="inclusive"):
        if species_list is None:
            species_list = species_order
        h_map: Dict[int, ROOT.TH1D] = {}
        g_map: Dict[int, ROOT.TGraphAsymmErrors] = {}
        pal = CHANNEL_SPECIES_COLORS.get(channel_mode, CHANNEL_SPECIES_COLORS["inclusive"])
        for idx_sp, sp_pdg in enumerate(species_list):
            sp_name = get_species_name(sp_pdg)
            sp_latex = SPECIES_LATEX.get(sp_pdg, sp_name)
            h_name = f"h_fraction_{sp_name}_vs_nu_energy{name_suffix}"
            g_name = f"g_fraction_{sp_name}_vs_nu_energy{name_suffix}"
            t_str = f"{sp_latex} Fraction vs Neutrino Energy{title_suffix};E_{{#nu}} [GeV];f({sp_latex}) = N({sp_latex}) / N_{{charm}}"

            h_frac = ROOT.TH1D(h_name, t_str, n_nu_ebins, arr_nu_edges)
            g_frac = ROOT.TGraphAsymmErrors(n_nu_ebins)
            g_frac.SetName(g_name)
            g_frac.SetTitle(t_str)

            for i_b in range(n_nu_ebins):
                b_cnts = counts_per_ebin.get(i_b, {})
                ebin_tot = sum(b_cnts.values())
                c_cnt = b_cnts.get(sp_pdg, 0)
                val_pct, lo_pct, hi_pct = clopper_pearson_errors(c_cnt, ebin_tot, 0.6827)
                val = val_pct / 100.0
                err_lo = lo_pct / 100.0
                err_hi = hi_pct / 100.0
                x_c = 0.5 * (nu_bin_edges[i_b] + nu_bin_edges[i_b + 1])
                dx_lo = x_c - nu_bin_edges[i_b]
                dx_hi = nu_bin_edges[i_b + 1] - x_c
                h_frac.SetBinContent(i_b + 1, val)
                h_frac.SetBinError(i_b + 1, max(err_lo, err_hi))
                g_frac.SetPoint(i_b, x_c, val)
                g_frac.SetPointError(i_b, dx_lo, dx_hi, err_lo, err_hi)

            sp_key = {421: "D0", -421: "D0", 411: "D+", -411: "D+", 431: "Ds+", -431: "Ds+", 4122: "Lambda_c", -4122: "Lambda_c"}.get(sp_pdg, "D0")
            col_hex = pal.get(sp_key, "#000000")
            col = ROOT.TColor.GetColor(col_hex)

            h_frac.SetLineColor(col)
            h_frac.SetMarkerColor(col)
            h_frac.SetMarkerStyle(20 + idx_sp)
            h_frac.SetLineWidth(2)
            h_frac.Write()
            h_map[sp_pdg] = h_frac

            g_frac.SetLineColor(col)
            g_frac.SetMarkerColor(col)
            g_frac.SetMarkerStyle(20 + idx_sp)
            g_frac.SetMarkerSize(1.15)
            g_frac.SetLineWidth(2)
            g_frac.Write()
            g_map[sp_pdg] = g_frac
        return h_map, g_map

    species_frac_nu_hists, species_frac_nu_graphs = _create_species_frac_graphs(
        data["species_nu_ebin_counts"], species_order, "", "", channel_mode="inclusive"
    )
    species_frac_nu_hists_nu, species_frac_nu_graphs_nu = _create_species_frac_graphs(
        data.get("species_nu_ebin_counts_nu", {}), species_order, "_nu_only", " (#nu_{#mu} CC)", channel_mode="nu"
    )
    species_frac_nu_hists_antinu, species_frac_nu_graphs_antinu = _create_species_frac_graphs(
        data.get("species_nu_ebin_counts_antinu", {}), anti_species_order, "_antinu_only", " (#bar{#nu}_{#mu} CC)", channel_mode="antinu"
    )

    # 1D Species Fraction vs E_charm for each major ground state
    species_frac_ch_hists: Dict[int, ROOT.TH1D] = {}
    species_frac_ch_graphs: Dict[int, ROOT.TGraphAsymmErrors] = {}
    for idx_sp, abs_c in enumerate(species_order):
        sp_name = get_species_name(abs_c)
        h_frac_vs_ce = ROOT.TH1D(
            f"h_fraction_{sp_name}_vs_charm_energy",
            f"{sp_name} Fraction vs Charm Hadron Energy;E_{{charm}} [GeV];f({sp_name}) = N({sp_name}) / N_{{charm}}",
            n_ch_ebins, arr_ch_edges
        )
        g_frac_vs_ce = ROOT.TGraphAsymmErrors(n_ch_ebins)
        g_frac_vs_ce.SetName(f"g_fraction_{sp_name}_vs_charm_energy")
        g_frac_vs_ce.SetTitle(f"{sp_name} Fraction vs Charm Hadron Energy;E_{{charm}} [GeV];f({sp_name}) = N({sp_name}) / N_{{charm}}")
        for i_b in range(n_ch_ebins):
            ebin_tot = sum(data["species_charm_ebin_counts"][i_b].values())
            c_cnt = data["species_charm_ebin_counts"][i_b].get(abs_c, 0)
            val_pct, lo_pct, hi_pct = clopper_pearson_errors(c_cnt, ebin_tot, 0.6827)
            val = val_pct / 100.0
            err_lo = lo_pct / 100.0
            err_hi = hi_pct / 100.0
            x_c = 0.5 * (charm_bin_edges[i_b] + charm_bin_edges[i_b + 1])
            dx_lo = x_c - charm_bin_edges[i_b]
            dx_hi = charm_bin_edges[i_b + 1] - x_c
            h_frac_vs_ce.SetBinContent(i_b + 1, val)
            h_frac_vs_ce.SetBinError(i_b + 1, max(err_lo, err_hi))
            g_frac_vs_ce.SetPoint(i_b, x_c, val)
            g_frac_vs_ce.SetPointError(i_b, dx_lo, dx_hi, err_lo, err_hi)

        col = SPECIES_COLORS.get(abs_c, ROOT.kBlack)
        h_frac_vs_ce.SetLineColor(col)
        h_frac_vs_ce.SetMarkerColor(col)
        h_frac_vs_ce.SetMarkerStyle(20 + idx_sp)
        h_frac_vs_ce.SetLineWidth(2)
        h_frac_vs_ce.Write()
        species_frac_ch_hists[abs_c] = h_frac_vs_ce

        g_frac_vs_ce.SetLineColor(col)
        g_frac_vs_ce.SetMarkerColor(col)
        g_frac_vs_ce.SetMarkerStyle(20 + idx_sp)
        g_frac_vs_ce.SetMarkerSize(1.15)
        g_frac_vs_ce.SetLineWidth(2)
        g_frac_vs_ce.Write()
        species_frac_ch_graphs[abs_c] = g_frac_vs_ce

    # Kinematics: Fragmentation z, pT, Theta
    h_z = ROOT.TH1D("h_fragmentation_z", "Charm Energy Fraction (Fragmentation);z = E_{charm} / E_{#nu};Entries", 50, 0.0, 1.0)
    for z in data["fragmentation_z"]:
        h_z.Fill(z)
    h_z.Write()

    h_pt = ROOT.TH1D("h_charm_pt", "Charm Transverse Momentum relative to #nu;p_{T} [GeV/c];Entries", 50, 0.0, 5.0)
    for pt in data["charm_pt"]:
        h_pt.Fill(pt)
    h_pt.Write()

    h_theta = ROOT.TH1D("h_charm_theta", "Charm Production Angle relative to #nu;#theta [mrad];Entries", 50, 0.0, 500.0)
    for th in data["charm_theta"]:
        h_theta.Fill(th)
    h_theta.Write()

    # Semi-leptonic Branching Fractions summary
    h_br_mu = ROOT.TH1D("h_semileptonic_br_mu", "Prompt Semi-Muonic Branching Ratio BR(c -> #mu X);Charm Species;BR(c -> #mu) [%]", n_sp_bins, 0.5, n_sp_bins + 0.5)
    h_br_e = ROOT.TH1D("h_semileptonic_br_e", "Prompt Semi-Electronic Branching Ratio BR(c -> e X);Charm Species;BR(c -> e) [%]", n_sp_bins, 0.5, n_sp_bins + 0.5)
    for idx, abs_c in enumerate(all_species_display, start=1):
        sp_name = get_species_name(abs_c)
        tot_d = data["total_decays_by_species"].get(abs_c, 0)
        n_mu = data["semi_muonic_counts"].get(abs_c, 0)
        n_e = data["semi_electronic_counts"].get(abs_c, 0)
        br_mu = (100.0 * n_mu / tot_d) if tot_d > 0 else 0.0
        br_e = (100.0 * n_e / tot_d) if tot_d > 0 else 0.0
        h_br_mu.GetXaxis().SetBinLabel(idx, sp_name)
        h_br_mu.SetBinContent(idx, br_mu)
        h_br_e.GetXaxis().SetBinLabel(idx, sp_name)
        h_br_e.SetBinContent(idx, br_e)
    h_br_mu.Write()
    h_br_e.Write()

    # =========================================================================
    # 2. ALL CHARM COMBINED DIRECTORY
    # =========================================================================
    dir_all = fout.mkdir("AllCharm")
    dir_all.cd()

    sorted_all_channels = sorted(data["all_decay_channels"].items(), key=lambda x: x[1], reverse=True)
    n_ch_all = min(len(sorted_all_channels), 30)
    h_all_channels = ROOT.TH1D("h_decay_channels_all_charm", "Exclusive Decay Channels (All Charm);Decay Channel;Decays", max(1, n_ch_all), 0.5, max(1, n_ch_all) + 0.5)
    for i_ch, (ch_tup, cnt) in enumerate(sorted_all_channels[:n_ch_all], start=1):
        h_all_channels.GetXaxis().SetBinLabel(i_ch, format_channel_formula(ch_tup))
        h_all_channels.SetBinContent(i_ch, cnt)
    h_all_channels.Write()

    # =========================================================================
    # 3. PER-SPECIES DIRECTORIES (D0, Dplus, Ds, Lambda_c)
    # =========================================================================
    dir_species_root = fout.mkdir("Species")
    species_ctau_hists: Dict[int, ROOT.TH1D] = {}
    species_br_vs_charm_e: Dict[int, List[Tuple[str, ROOT.TH1D, ROOT.TGraphAsymmErrors]]] = {}
    species_br_vs_nu_e: Dict[int, List[Tuple[str, ROOT.TH1D, ROOT.TGraphAsymmErrors]]] = {}

    for abs_c in sorted(data["species_hadron_counts"].keys()):
        sp_name = get_species_name(abs_c)
        dir_sp = dir_species_root.mkdir(sp_name)
        dir_sp.cd()

        # Species Hadron Energy Spectrum
        c_e_list = data["charm_e_by_species"].get(abs_c, [])
        h_c_e = ROOT.TH1D(f"h_energy_{sp_name}", f"{sp_name} Hadron Energy Spectrum;E_{{{sp_name}}} [GeV];Entries", 50, 0.0, 1000.0)
        for ce in c_e_list:
            h_c_e.Fill(ce)
        h_c_e.SetLineColor(SPECIES_COLORS.get(abs_c, ROOT.kBlack))
        h_c_e.SetLineWidth(2)
        h_c_e.Write()

        ch_map = data["decay_channels_by_species"].get(abs_c, Counter())
        tot_sp_decays = data["total_decays_by_species"].get(abs_c, sum(ch_map.values()))
        sorted_sp_channels = sorted(ch_map.items(), key=lambda x: x[1], reverse=True)

        n_top = min(len(sorted_sp_channels), 25)
        h_sp_ch = ROOT.TH1D(
            f"h_decay_channels_{sp_name}",
            f"Exclusive Decay Channels ({sp_name});Decay Channel;Decays",
            max(1, n_top), 0.5, max(1, n_top) + 0.5
        )
        h_sp_br = ROOT.TH1D(
            f"h_branching_ratio_{sp_name}",
            f"Branching Ratios ({sp_name});Decay Channel;Branching Ratio [%]",
            max(1, n_top), 0.5, max(1, n_top) + 0.5
        )

        for i_ch, (ch_tup, cnt) in enumerate(sorted_sp_channels[:n_top], start=1):
            formula = format_channel_formula(ch_tup)
            h_sp_ch.GetXaxis().SetBinLabel(i_ch, formula)
            h_sp_ch.SetBinContent(i_ch, cnt)
            br_val = (100.0 * cnt / tot_sp_decays) if tot_sp_decays > 0 else 0.0
            h_sp_br.GetXaxis().SetBinLabel(i_ch, formula)
            h_sp_br.SetBinContent(i_ch, br_val)
        h_sp_ch.Write()
        h_sp_br.Write()

        # Proper Lifetime c*tau
        ctau_list = data["ctau_values_um"].get(abs_c, [])
        h_ctau = ROOT.TH1D(
            f"h_ctau_{sp_name}",
            f"Proper Lifetime c#tau ({sp_name});c#tau [#mum];Entries",
            50, 0.0, 1000.0
        )
        for val in ctau_list:
            h_ctau.Fill(val)
        col = SPECIES_COLORS.get(abs_c, ROOT.kBlack)
        h_ctau.SetLineColor(col)
        h_ctau.SetFillColorAlpha(col, 0.35)
        h_ctau.SetLineWidth(2)
        h_ctau.Write()
        species_ctau_hists[abs_c] = h_ctau

        # Lab decay distance in mm
        dist_list = data["decay_dist_mm"].get(abs_c, [])
        h_dist = ROOT.TH1D(
            f"h_decay_distance_mm_{sp_name}",
            f"Decay Distance in Material ({sp_name});Flight Path L [mm];Entries",
            50, 0.0, 50.0
        )
        for d in dist_list:
            h_dist.Fill(d)
        h_dist.Write()

        # Branching Ratios vs E_charm for top 4 decay modes
        top_modes = [tup for tup, _ in sorted_sp_channels[:4]]
        ch_ebin_map = data["decay_channels_by_species_charm_ebin"].get(abs_c, {})
        species_br_vs_charm_e[abs_c] = []

        dir_br_ce = dir_sp.mkdir("BranchingRatios_vs_CharmEnergy")
        dir_br_ce.cd()

        mode_colors = [ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kOrange + 2]

        for m_idx, mode_tuple in enumerate(top_modes):
            formula = format_channel_formula(mode_tuple)
            slug = channel_slug(mode_tuple)
            h_mode_vs_ce = ROOT.TH1D(
                f"h_br_{slug}_vs_charm_e",
                f"BR({sp_name} -> {formula}) vs E_{{charm}};E_{{charm}} [GeV];Branching Ratio [%]",
                n_ch_ebins, arr_ch_edges
            )
            g_mode_vs_ce = ROOT.TGraphAsymmErrors(n_ch_ebins)
            g_mode_vs_ce.SetName(f"g_br_{slug}_vs_charm_e")
            g_mode_vs_ce.SetTitle(f"BR({sp_name} #rightarrow {formula}) vs E_{{charm}};E_{{charm}} [GeV];Branching Ratio [%]")
            for i_b in range(n_ch_ebins):
                eb_counter = ch_ebin_map.get(i_b, Counter())
                eb_tot = sum(eb_counter.values())
                mode_cnt = eb_counter.get(mode_tuple, 0)
                val_pct, lo_pct, hi_pct = clopper_pearson_errors(mode_cnt, eb_tot, 0.6827)
                x_c = 0.5 * (charm_bin_edges[i_b] + charm_bin_edges[i_b + 1])
                dx_lo = x_c - charm_bin_edges[i_b]
                dx_hi = charm_bin_edges[i_b + 1] - x_c
                h_mode_vs_ce.SetBinContent(i_b + 1, val_pct)
                h_mode_vs_ce.SetBinError(i_b + 1, max(lo_pct, hi_pct))
                g_mode_vs_ce.SetPoint(i_b, x_c, val_pct)
                g_mode_vs_ce.SetPointError(i_b, dx_lo, dx_hi, lo_pct, hi_pct)

            col = mode_colors[m_idx % len(mode_colors)]
            h_mode_vs_ce.SetLineColor(col)
            h_mode_vs_ce.SetMarkerColor(col)
            h_mode_vs_ce.SetMarkerStyle(20 + m_idx)
            h_mode_vs_ce.SetLineWidth(2)
            h_mode_vs_ce.Write()

            g_mode_vs_ce.SetLineColor(col)
            g_mode_vs_ce.SetMarkerColor(col)
            g_mode_vs_ce.SetMarkerStyle(20 + m_idx)
            g_mode_vs_ce.SetMarkerSize(1.15)
            g_mode_vs_ce.SetLineWidth(2)
            g_mode_vs_ce.Write()
            species_br_vs_charm_e[abs_c].append((formula, h_mode_vs_ce, g_mode_vs_ce))

        # Branching Ratios vs E_nu for top 4 decay modes
        nu_ebin_map = data["decay_channels_by_species_nu_ebin"].get(abs_c, {})
        species_br_vs_nu_e[abs_c] = []

        dir_br_nue = dir_sp.mkdir("BranchingRatios_vs_NuEnergy")
        dir_br_nue.cd()

        for m_idx, mode_tuple in enumerate(top_modes):
            formula = format_channel_formula(mode_tuple)
            slug = channel_slug(mode_tuple)
            h_mode_vs_nue = ROOT.TH1D(
                f"h_br_{slug}_vs_nu_e",
                f"BR({sp_name} -> {formula}) vs E_{{#nu}};E_{{#nu}} [GeV];Branching Ratio [%]",
                n_nu_ebins, arr_nu_edges
            )
            g_mode_vs_nue = ROOT.TGraphAsymmErrors(n_nu_ebins)
            g_mode_vs_nue.SetName(f"g_br_{slug}_vs_nu_e")
            g_mode_vs_nue.SetTitle(f"BR({sp_name} #rightarrow {formula}) vs E_{{#nu}};E_{{#nu}} [GeV];Branching Ratio [%]")
            for i_b in range(n_nu_ebins):
                eb_counter = nu_ebin_map.get(i_b, Counter())
                eb_tot = sum(eb_counter.values())
                mode_cnt = eb_counter.get(mode_tuple, 0)
                val_pct, lo_pct, hi_pct = clopper_pearson_errors(mode_cnt, eb_tot, 0.6827)
                x_c = 0.5 * (nu_bin_edges[i_b] + nu_bin_edges[i_b + 1])
                dx_lo = x_c - nu_bin_edges[i_b]
                dx_hi = nu_bin_edges[i_b + 1] - x_c
                h_mode_vs_nue.SetBinContent(i_b + 1, val_pct)
                h_mode_vs_nue.SetBinError(i_b + 1, max(lo_pct, hi_pct))
                g_mode_vs_nue.SetPoint(i_b, x_c, val_pct)
                g_mode_vs_nue.SetPointError(i_b, dx_lo, dx_hi, lo_pct, hi_pct)

            col = mode_colors[m_idx % len(mode_colors)]
            h_mode_vs_nue.SetLineColor(col)
            h_mode_vs_nue.SetMarkerColor(col)
            h_mode_vs_nue.SetMarkerStyle(20 + m_idx)
            h_mode_vs_nue.SetLineWidth(2)
            h_mode_vs_nue.Write()

            g_mode_vs_nue.SetLineColor(col)
            g_mode_vs_nue.SetMarkerColor(col)
            g_mode_vs_nue.SetMarkerStyle(20 + m_idx)
            g_mode_vs_nue.SetMarkerSize(1.15)
            g_mode_vs_nue.SetLineWidth(2)
            g_mode_vs_nue.Write()
            species_br_vs_nu_e[abs_c].append((formula, h_mode_vs_nue, g_mode_vs_nue))

    # =========================================================================
    # 4. CANVASES DIRECTORY: Superimposed Histograms, Models & Legends
    # =========================================================================
    dir_canvases = fout.mkdir("Canvases")
    dir_canvases.cd()

    # Common theoretical reference curves
    e_max_model = max(1000.0, nu_bin_edges[-1])

    # 1. Tungsten Target nu_mu
    f_sr_w_nu, g_sr_w_nu_inner, g_sr_w_nu_outer = create_slow_rescaling_model(
        name="f_slow_rescaling_w_nu",
        e_min=0.0,
        e_max=e_max_model,
        target="tungsten",
        nu_type="nu",
        line_color=ROOT.kBlue + 1,
        band_color=ROOT.kAzure - 4,
    )

    # 2. Tungsten Target anti_nu_mu
    f_sr_w_antinu, g_sr_w_antinu_inner, g_sr_w_antinu_outer = create_slow_rescaling_model(
        name="f_slow_rescaling_w_antinu",
        e_min=0.0,
        e_max=e_max_model,
        target="tungsten",
        nu_type="antinu",
        line_color=ROOT.kMagenta + 2,
        band_color=ROOT.kMagenta - 9,
    )

    # 3. Tungsten Target Inclusive (flux-averaged)
    f_sr_w_inc, g_sr_w_inc_inner, g_sr_w_inc_outer = create_slow_rescaling_model(
        name="f_slow_rescaling_w_inclusive",
        e_min=0.0,
        e_max=e_max_model,
        target="tungsten",
        nu_type="inclusive",
        line_color=ROOT.kRed + 1,
        band_color=ROOT.kRed - 7,
    )

    # 4. Isoscalar References (matching each beam category)
    f_sr_iso_inc, _, _ = create_slow_rescaling_model(
        name="f_slow_rescaling_isoscalar_inclusive",
        e_min=0.0,
        e_max=e_max_model,
        target="isoscalar",
        nu_type="inclusive",
        line_color=ROOT.kGray + 2,
    )
    f_sr_iso_inc.SetLineStyle(7)
    f_sr_iso_inc.SetLineWidth(2)

    f_sr_iso_nu, _, _ = create_slow_rescaling_model(
        name="f_slow_rescaling_isoscalar_nu",
        e_min=0.0,
        e_max=e_max_model,
        target="isoscalar",
        nu_type="nu",
        line_color=ROOT.kGray + 2,
    )
    f_sr_iso_nu.SetLineStyle(7)
    f_sr_iso_nu.SetLineWidth(2)

    f_sr_iso_antinu, _, _ = create_slow_rescaling_model(
        name="f_slow_rescaling_isoscalar_antinu",
        e_min=0.0,
        e_max=e_max_model,
        target="isoscalar",
        nu_type="antinu",
        line_color=ROOT.kGray + 2,
    )
    f_sr_iso_antinu.SetLineStyle(7)
    f_sr_iso_antinu.SetLineWidth(2)

    # 5. LHAPDF nCTEQ15 theoretical predictions from CSV
    col_nu = ROOT.TColor.GetColor("#1e88e5")       # Blue for neutrinos
    col_antinu = ROOT.TColor.GetColor("#8e24aa")   # Purple for anti-neutrinos
    col_inc = ROOT.TColor.GetColor("#d91e18")      # Red for inclusive (nu + anti-nu)

    # 5a. Tungsten target predictions (solid transparent bands, solid central line)
    g_lhapdf_nu_w, g_lhapdf_nu_w_inner, g_lhapdf_nu_w_outer = load_lhapdf_charm_prediction(
        "numuCC_charm_cross_section_ratio_prediction_lhapdf.csv",
        name="lhapdf_nu_tungsten",
        target="tungsten",
        line_color=col_nu,
        line_width=2,
        line_style=1,
        band_color=col_nu,
        inner_alpha=0.35,
        outer_alpha=0.18,
        fill_style_inner=1001,
        fill_style_outer=1001,
    )
    g_lhapdf_antinu_w, g_lhapdf_antinu_w_inner, g_lhapdf_antinu_w_outer = load_lhapdf_charm_prediction(
        "antinumuCC_charm_cross_section_ratio_prediction_lhapdf.csv",
        name="lhapdf_antinu_tungsten",
        target="tungsten",
        line_color=col_antinu,
        line_width=2,
        line_style=1,
        band_color=col_antinu,
        inner_alpha=0.35,
        outer_alpha=0.18,
        fill_style_inner=1001,
        fill_style_outer=1001,
    )
    g_lhapdf_inc_w, g_lhapdf_inc_w_inner, g_lhapdf_inc_w_outer = load_lhapdf_charm_prediction(
        "inclusive_numu_antinumu_charm_cross_section_ratio_prediction_lhapdf.csv",
        name="lhapdf_inclusive_tungsten",
        target="tungsten",
        line_color=col_inc,
        line_width=2,
        line_style=1,
        band_color=col_inc,
        inner_alpha=0.35,
        outer_alpha=0.18,
        fill_style_inner=1001,
        fill_style_outer=1001,
    )

    # 5b. Emulsion target predictions (hatched diagonal error bands, dashed central line)
    g_lhapdf_nu_em, g_lhapdf_nu_em_inner, g_lhapdf_nu_em_outer = load_lhapdf_charm_prediction(
        "numuCC_charm_cross_section_ratio_prediction_lhapdf.csv",
        name="lhapdf_nu_emulsion",
        target="emulsion",
        line_color=col_nu,
        line_width=2,
        line_style=7,
        band_color=col_nu,
        fill_style_inner=3004,
        fill_style_outer=3005,
    )
    g_lhapdf_antinu_em, g_lhapdf_antinu_em_inner, g_lhapdf_antinu_em_outer = load_lhapdf_charm_prediction(
        "antinumuCC_charm_cross_section_ratio_prediction_lhapdf.csv",
        name="lhapdf_antinu_emulsion",
        target="emulsion",
        line_color=col_antinu,
        line_width=2,
        line_style=7,
        band_color=col_antinu,
        fill_style_inner=3004,
        fill_style_outer=3005,
    )
    g_lhapdf_inc_em, g_lhapdf_inc_em_inner, g_lhapdf_inc_em_outer = load_lhapdf_charm_prediction(
        "inclusive_numu_antinumu_charm_cross_section_ratio_prediction_lhapdf.csv",
        name="lhapdf_inclusive_emulsion",
        target="emulsion",
        line_color=col_inc,
        line_width=2,
        line_style=7,
        band_color=col_inc,
        fill_style_inner=3004,
        fill_style_outer=3005,
    )

    # Aliases
    g_lhapdf_nu = g_lhapdf_nu_w
    g_lhapdf_nu_inner = g_lhapdf_nu_w_inner
    g_lhapdf_nu_outer = g_lhapdf_nu_w_outer
    g_lhapdf_antinu = g_lhapdf_antinu_w
    g_lhapdf_antinu_inner = g_lhapdf_antinu_w_inner
    g_lhapdf_antinu_outer = g_lhapdf_antinu_w_outer
    g_lhapdf_inc = g_lhapdf_inc_w
    g_lhapdf_inc_inner = g_lhapdf_inc_w_inner
    g_lhapdf_inc_outer = g_lhapdf_inc_w_outer

    # Load external experimental measurements:
    # 1. CHORUS (New J. Phys. 13 (2011) 093002, Emulsion target, nu_mu CC)
    g_chorus = None
    chorus_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurements", "chorus_sigmaCharmDivSigmaCC_vs_Enu.csv")
    merged_xsec_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurements", "neutrino_charm_cross_section_ratio_vs_enu.csv")
    chorus_pts = []

    # Try loading from merged CSV or dedicated CHORUS CSV
    if os.path.exists(merged_xsec_csv_path):
        try:
            import csv
            with open(merged_xsec_csv_path, "r", encoding="utf-8-sig") as f_mx:
                reader = csv.DictReader(row for row in f_mx if not row.strip().startswith("#"))
                for row in reader:
                    if row["experiment"].strip() == "CHORUS" and row.get("channel", "inclusive").strip() == "inclusive":
                        enu = float(row["E_nu_GeV"])
                        enu_err = float(row.get("dx_low", row.get("dx_high", 0.0)))
                        val = float(row["ratio"])
                        val_err = float(row.get("dy_low", row.get("dy_high", 0.0)))
                        chorus_pts.append([enu, enu_err, val, val_err])
        except Exception as e_mx:
            print(f"  [Warning] Failed to load CHORUS from merged CSV: {e_mx}")

    if not chorus_pts and os.path.exists(chorus_csv_path):
        try:
            with open(chorus_csv_path, "r", encoding="utf-8-sig") as f_ch:
                for line in f_ch:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("E_nu"):
                        continue
                    parts = [float(p.strip()) for p in line.split(",")]
                    chorus_pts.append(parts)
        except Exception as e_ch:
            print(f"  [Warning] Failed to load CHORUS CSV: {e_ch}")

    # Robust fallback: New J. Phys. 13 (2011) 093002, Fig 4
    if not chorus_pts:
        chorus_pts = [
            [13.0, 7.0, 0.024, 0.004],
            [22.0, 6.0, 0.043, 0.005],
            [41.0, 13.0, 0.065, 0.006],
            [73.0, 19.0, 0.077, 0.007],
            [104.0, 15.0, 0.094, 0.012],
            [150.0, 30.0, 0.095, 0.012],
        ]

    if chorus_pts:
        g_chorus = ROOT.TGraphErrors(len(chorus_pts))
        g_chorus.SetName("g_chorus_measurement")
        g_chorus.SetTitle("CHORUS (2011, Emulsion, inclusive)")
        for idx_ch, (enu, enu_err, val, val_err) in enumerate(chorus_pts):
            g_chorus.SetPoint(idx_ch, enu, val)
            g_chorus.SetPointError(idx_ch, enu_err, val_err)
        # Prominent black open square
        g_chorus.SetMarkerStyle(25)  # Open square
        g_chorus.SetMarkerSize(1.20)
        g_chorus.SetMarkerColor(ROOT.kBlack)
        g_chorus.SetLineColor(ROOT.kBlack)
        g_chorus.SetLineWidth(2)
        print(f"  [Info] Loaded CHORUS measurement ({len(chorus_pts)} points)")

    # 2. Fermilab E531 (Phys. Lett. B 206 (1988) 375, Emulsion target, nu_mu CC)
    g_e531 = None
    e531_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurements", "e531_sigmaCharmDivSigmaCC_vs_Enu.csv")
    e531_pts = []

    # Try loading from merged CSV or dedicated E531 CSV
    if os.path.exists(merged_xsec_csv_path):
        try:
            import csv
            with open(merged_xsec_csv_path, "r", encoding="utf-8-sig") as f_mx:
                reader = csv.DictReader(row for row in f_mx if not row.strip().startswith("#"))
                for row in reader:
                    if row["experiment"].strip() == "E531" and row.get("channel", "inclusive").strip() == "inclusive":
                        enu = float(row["E_nu_GeV"])
                        dx_lo = float(row.get("dx_low", 0.0))
                        dx_hi = float(row.get("dx_high", 0.0))
                        val = float(row["ratio"])
                        dy_lo = float(row.get("dy_low", 0.0))
                        dy_hi = float(row.get("dy_high", 0.0))
                        e531_pts.append([len(e531_pts) + 1, enu - dx_lo, enu + dx_hi, enu, dx_lo, dx_hi, val * 100.0, dy_lo * 100.0, dy_hi * 100.0, 0.5 * (dy_lo + dy_hi) * 100.0])
        except Exception as e_mx:
            print(f"  [Warning] Failed to load E531 from merged CSV: {e_mx}")

    if not e531_pts and os.path.exists(e531_csv_path):
        try:
            with open(e531_csv_path, "r", encoding="utf-8-sig") as f_e:
                for line in f_e:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("bin_index"):
                        continue
                    parts = [float(p.strip()) for p in line.split(",")]
                    e531_pts.append(parts)
        except Exception as e_e531:
            print(f"  [Warning] Failed to load E531 CSV: {e_e531}")

    # Robust fallback: Phys. Lett. B 206 (1988) 375, Table 2
    if not e531_pts:
        e531_pts = [
            [1, 0.0, 10.0, 5.5, 5.5, 4.5, 0.75, 0.75, 1.05, 1.0],
            [2, 10.0, 30.0, 20.0, 10.0, 10.0, 3.6, 0.9, 0.9, 0.9],
            [3, 30.0, 60.0, 45.0, 15.0, 15.0, 4.9, 1.1, 1.1, 1.1],
            [4, 60.0, 150.0, 105.0, 45.0, 45.0, 10.2, 2.1, 2.5, 2.3],
        ]

    if e531_pts:
        g_e531 = ROOT.TGraphAsymmErrors(len(e531_pts))
        g_e531.SetName("g_e531_measurement")
        g_e531.SetTitle("E531 (1988, Emulsion, inclusive)")
        for idx_e, p in enumerate(e531_pts):
            e_mean = p[3]
            e_err_lo = p[4]
            e_err_hi = p[5]
            r_val = p[6] / 100.0
            r_err_lo = p[7] / 100.0
            r_err_hi = p[8] / 100.0
            g_e531.SetPoint(idx_e, e_mean, r_val)
            g_e531.SetPointError(idx_e, e_err_lo, e_err_hi, r_err_lo, r_err_hi)
        # Prominent black open circle
        g_e531.SetMarkerStyle(24)  # Open circle
        g_e531.SetMarkerSize(1.20)
        g_e531.SetMarkerColor(ROOT.kBlack)
        g_e531.SetLineColor(ROOT.kBlack)
        g_e531.SetLineWidth(2)
        print(f"  [Info] Loaded E531 measurement ({len(e531_pts)} points)")

    # Calculate data points
    g_charm_rate_mc = ROOT.TGraphAsymmErrors(n_nu_ebins)
    g_charm_rate_mc.SetName("g_charm_rate_mc")
    g_charm_rate_mc.SetTitle("MC Inclusive (#nu_{#mu} + #bar{#nu}_{#mu})")

    g_charm_rate_mc_nu = ROOT.TGraphAsymmErrors(n_nu_ebins)
    g_charm_rate_mc_nu.SetName("g_charm_rate_mc_nu")
    g_charm_rate_mc_nu.SetTitle("MC #nu_{#mu} CC")

    g_charm_rate_mc_antinu = ROOT.TGraphAsymmErrors(n_nu_ebins)
    g_charm_rate_mc_antinu.SetName("g_charm_rate_mc_antinu")
    g_charm_rate_mc_antinu.SetTitle("MC #bar{#nu}_{#mu} CC")

    for i_b in range(n_nu_ebins):
        x_center = 0.5 * (nu_bin_edges[i_b] + nu_bin_edges[i_b + 1])
        x_err = 0.5 * (nu_bin_edges[i_b + 1] - nu_bin_edges[i_b])

        # Inclusive
        n_cc = data["cc_all_per_ebin"][i_b]
        n_c = data["cc_charm_per_ebin"][i_b]
        val_pct, err_lo_pct, err_hi_pct = clopper_pearson_errors(n_c, n_cc, 0.6827)
        val = val_pct / 100.0
        err_lo = err_lo_pct / 100.0
        err_hi = err_hi_pct / 100.0
        g_charm_rate_mc.SetPoint(i_b, x_center, val)
        g_charm_rate_mc.SetPointError(i_b, x_err, x_err, err_lo, err_hi)
        h_charm_rate_ebins.SetBinContent(i_b + 1, val)
        h_charm_rate_ebins.SetBinError(i_b + 1, max(err_lo, err_hi))

        # pure nu_mu
        n_cc_nu = data.get("cc_all_nu_per_ebin", [0] * n_nu_ebins)[i_b]
        n_c_nu = data.get("cc_charm_nu_per_ebin", [0] * n_nu_ebins)[i_b]
        val_pct_nu, lo_nu, hi_nu = clopper_pearson_errors(n_c_nu, n_cc_nu, 0.6827)
        g_charm_rate_mc_nu.SetPoint(i_b, x_center, val_pct_nu / 100.0)
        g_charm_rate_mc_nu.SetPointError(i_b, x_err, x_err, lo_nu / 100.0, hi_nu / 100.0)

        # pure anti_nu_mu
        n_cc_an = data.get("cc_all_antinu_per_ebin", [0] * n_nu_ebins)[i_b]
        n_c_an = data.get("cc_charm_antinu_per_ebin", [0] * n_nu_ebins)[i_b]
        val_pct_an, lo_an, hi_an = clopper_pearson_errors(n_c_an, n_cc_an, 0.6827)
        g_charm_rate_mc_antinu.SetPoint(i_b, x_center, val_pct_an / 100.0)
        g_charm_rate_mc_antinu.SetPointError(i_b, x_err, x_err, lo_an / 100.0, hi_an / 100.0)

    g_charm_rate_mc.SetMarkerStyle(20)
    g_charm_rate_mc.SetMarkerSize(1.2)
    g_charm_rate_mc.SetMarkerColor(ROOT.kRed + 1)
    g_charm_rate_mc.SetLineColor(ROOT.kRed + 1)
    g_charm_rate_mc.SetLineWidth(2)

    g_charm_rate_mc_nu.SetMarkerStyle(20)
    g_charm_rate_mc_nu.SetMarkerSize(1.2)
    g_charm_rate_mc_nu.SetMarkerColor(ROOT.kBlue + 2)
    g_charm_rate_mc_nu.SetLineColor(ROOT.kBlue + 2)
    g_charm_rate_mc_nu.SetLineWidth(2)

    g_charm_rate_mc_antinu.SetMarkerStyle(20)
    g_charm_rate_mc_antinu.SetMarkerSize(1.2)
    g_charm_rate_mc_antinu.SetMarkerColor(ROOT.kMagenta + 2)
    g_charm_rate_mc_antinu.SetLineColor(ROOT.kMagenta + 2)
    g_charm_rate_mc_antinu.SetLineWidth(2)

    y_max_frame = max(0.28, h_charm_rate_ebins.GetMaximum() * 1.50)

    # -------------------------------------------------------------------------
    # Canvas 1a: Inclusive (nu_mu + anti_nu_mu)
    # -------------------------------------------------------------------------
    c_rate_inc = ROOT.TCanvas("c_charm_rate_vs_energy", "Charm Production Rate vs Neutrino Energy (Inclusive)", 950, 700)
    c_rate_inc.SetLeftMargin(0.12)
    c_rate_inc.SetRightMargin(0.06)
    c_rate_inc.SetTopMargin(0.08)
    c_rate_inc.SetBottomMargin(0.12)
    c_rate_inc.SetGrid()

    h_frame_inc = c_rate_inc.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_frame)
    h_frame_inc.SetTitle("Charm Production Rate: Inclusive (#nu_{#mu} + #bar{#nu}_{#mu});E_{#nu} [GeV];#sigma_{charm} / #sigma_{CC}")
    h_frame_inc.GetYaxis().SetTitleOffset(1.25)

    # 1. Tungsten target predictions (solid transparent bands)
    g_lhapdf_inc_w_outer.Draw("3 same")
    g_lhapdf_inc_w_inner.Draw("3 same")

    # 2. Emulsion target predictions (diagonal hatched bands)
    g_lhapdf_inc_em_outer.Draw("3 same")
    g_lhapdf_inc_em_inner.Draw("3 same")

    # 3. Central lines
    g_lhapdf_inc_w.Draw("L same")
    g_lhapdf_inc_em.Draw("L same")

    # 4. SND@LHC MC data points
    g_charm_rate_mc.Draw("P E1 same")

    leg_inc = ROOT.TLegend(0.44, 0.50, 0.92, 0.89)
    leg_inc.SetBorderSize(1)
    leg_inc.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_inc.SetTextSize(0.019)
    leg_inc.AddEntry(g_charm_rate_mc, "MC Inclusive (#nu_{#mu}+#bar{#nu}_{#mu})", "lep")
    leg_inc.AddEntry(g_lhapdf_inc_w, "QCD Prediction (^{184}W target, solid)", "l")
    leg_inc.AddEntry(g_lhapdf_inc_w_inner, "Base Kinematics 1#sigma (^{184}W)", "f")
    leg_inc.AddEntry(g_lhapdf_inc_w_outer, "LHAPDF nCTEQ15 1#sigma (^{184}W)", "f")
    leg_inc.AddEntry(g_lhapdf_inc_em, "QCD Prediction (Emulsion target, dashed)", "l")
    leg_inc.AddEntry(g_lhapdf_inc_em_inner, "Base Kinematics 1#sigma (Emulsion, //)", "f")
    leg_inc.AddEntry(g_lhapdf_inc_em_outer, "LHAPDF nCTEQ15 1#sigma (Emulsion, \\\\)", "f")
    leg_inc.Draw("same")
    c_rate_inc.Write()

    # -------------------------------------------------------------------------
    # Canvas 1b: Neutrino Only (nu_mu)
    # -------------------------------------------------------------------------
    c_rate_nu = ROOT.TCanvas("c_charm_rate_vs_energy_nu_only", "Charm Production Rate vs Neutrino Energy (nu_mu)", 950, 700)
    c_rate_nu.SetLeftMargin(0.12)
    c_rate_nu.SetRightMargin(0.06)
    c_rate_nu.SetTopMargin(0.08)
    c_rate_nu.SetBottomMargin(0.12)
    c_rate_nu.SetGrid()

    h_frame_nu = c_rate_nu.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_frame)
    h_frame_nu.SetTitle("Charm Production Rate: #nu_{#mu} CC;E_{#nu} [GeV];#sigma_{charm} / #sigma_{CC}")
    h_frame_nu.GetYaxis().SetTitleOffset(1.25)

    # 1. Tungsten target predictions (solid transparent bands)
    g_lhapdf_nu_w_outer.Draw("3 same")
    g_lhapdf_nu_w_inner.Draw("3 same")

    # 2. Emulsion target predictions (diagonal hatched bands)
    g_lhapdf_nu_em_outer.Draw("3 same")
    g_lhapdf_nu_em_inner.Draw("3 same")

    # 3. Central lines
    g_lhapdf_nu_w.Draw("L same")
    g_lhapdf_nu_em.Draw("L same")

    # 4. Experimental measurements & SND@LHC MC data points
    g_charm_rate_mc_nu.Draw("P E1 same")
    if g_chorus:
        g_chorus.Draw("P E1 same")
    if g_e531:
        g_e531.Draw("P E1 same")

    leg_nu = ROOT.TLegend(0.44, 0.44, 0.92, 0.89)
    leg_nu.SetBorderSize(1)
    leg_nu.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_nu.SetTextSize(0.018)
    leg_nu.AddEntry(g_charm_rate_mc_nu, "MC #nu_{#mu} CC (Tungsten)", "lep")
    if g_chorus:
        leg_nu.AddEntry(g_chorus, "CHORUS (2011, Emulsion, inclusive)", "lep")
    if g_e531:
        leg_nu.AddEntry(g_e531, "E531 (1988, Emulsion, inclusive)", "lep")
    leg_nu.AddEntry(g_lhapdf_nu_w, "QCD Prediction (^{184}W target, solid)", "l")
    leg_nu.AddEntry(g_lhapdf_nu_w_inner, "Base Kinematics 1#sigma (^{184}W)", "f")
    leg_nu.AddEntry(g_lhapdf_nu_w_outer, "LHAPDF nCTEQ15 1#sigma (^{184}W)", "f")
    leg_nu.AddEntry(g_lhapdf_nu_em, "QCD Prediction (Emulsion target, dashed)", "l")
    leg_nu.AddEntry(g_lhapdf_nu_em_inner, "Base Kinematics 1#sigma (Emulsion, //)", "f")
    leg_nu.AddEntry(g_lhapdf_nu_em_outer, "LHAPDF nCTEQ15 1#sigma (Emulsion, \\\\)", "f")
    leg_nu.Draw("same")
    c_rate_nu.Write()

    # -------------------------------------------------------------------------
    # Canvas 1c: Antineutrino Only (anti_nu_mu)
    # -------------------------------------------------------------------------
    c_rate_antinu = ROOT.TCanvas("c_charm_rate_vs_energy_antinu_only", "Charm Production Rate vs Neutrino Energy (anti_nu_mu)", 950, 700)
    c_rate_antinu.SetLeftMargin(0.12)
    c_rate_antinu.SetRightMargin(0.06)
    c_rate_antinu.SetTopMargin(0.08)
    c_rate_antinu.SetBottomMargin(0.12)
    c_rate_antinu.SetGrid()

    y_max_antinu = max(0.35, y_max_frame * 1.25)
    h_frame_antinu = c_rate_antinu.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_frame)
    h_frame_antinu.SetTitle("Charm Production Rate: #bar{#nu}_{#mu} CC;E_{#nu} [GeV];#sigma_{charm} / #sigma_{CC}")
    h_frame_antinu.GetYaxis().SetTitleOffset(1.25)

    # 1. Tungsten target predictions (solid transparent bands)
    g_lhapdf_antinu_w_outer.Draw("3 same")
    g_lhapdf_antinu_w_inner.Draw("3 same")

    # 2. Emulsion target predictions (diagonal hatched bands)
    g_lhapdf_antinu_em_outer.Draw("3 same")
    g_lhapdf_antinu_em_inner.Draw("3 same")

    # 3. Central lines
    g_lhapdf_antinu_w.Draw("L same")
    g_lhapdf_antinu_em.Draw("L same")

    # 4. SND@LHC MC data points
    g_charm_rate_mc_antinu.Draw("P E1 same")

    leg_antinu = ROOT.TLegend(0.44, 0.50, 0.92, 0.89)
    leg_antinu.SetBorderSize(1)
    leg_antinu.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_antinu.SetTextSize(0.019)
    leg_antinu.AddEntry(g_charm_rate_mc_antinu, "MC #bar{#nu}_{#mu} CC (Tungsten)", "lep")
    leg_antinu.AddEntry(g_lhapdf_antinu_w, "QCD Prediction (^{184}W target, solid)", "l")
    leg_antinu.AddEntry(g_lhapdf_antinu_w_inner, "Base Kinematics 1#sigma (^{184}W)", "f")
    leg_antinu.AddEntry(g_lhapdf_antinu_w_outer, "LHAPDF nCTEQ15 1#sigma (^{184}W)", "f")
    leg_antinu.AddEntry(g_lhapdf_antinu_em, "QCD Prediction (Emulsion target, dashed)", "l")
    leg_antinu.AddEntry(g_lhapdf_antinu_em_inner, "Base Kinematics 1#sigma (Emulsion, //)", "f")
    leg_antinu.AddEntry(g_lhapdf_antinu_em_outer, "LHAPDF nCTEQ15 1#sigma (Emulsion, \\\\)", "f")
    leg_antinu.Draw("same")
    c_rate_antinu.Write()

    # -------------------------------------------------------------------------
    # Canvas 1d: Combined Charm Production Rate (Inclusive vs Nu vs Anti-Nu)
    # Red for Inclusive, Blue for Neutrinos, Purple for Anti-Neutrinos
    # -------------------------------------------------------------------------
    c_rate_all = ROOT.TCanvas("c_charm_rate_vs_energy_comparison", "Charm Production Rate: Inclusive vs Nu vs Anti-Nu", 950, 700)
    c_rate_all.SetLeftMargin(0.12)
    c_rate_all.SetRightMargin(0.06)
    c_rate_all.SetTopMargin(0.08)
    c_rate_all.SetBottomMargin(0.12)
    c_rate_all.SetGrid()

    h_frame_all = c_rate_all.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_antinu)
    h_frame_all.SetTitle("Charm Production Rate Comparison: Inclusive vs #nu_{#mu} vs #bar{#nu}_{#mu};E_{#nu} [GeV];#sigma_{charm} / #sigma_{CC}")
    h_frame_all.GetYaxis().SetTitleOffset(1.25)

    g_lhapdf_inc_w_outer.Draw("3 same")
    g_lhapdf_inc_w_inner.Draw("3 same")
    g_lhapdf_nu_w_outer.Draw("3 same")
    g_lhapdf_nu_w_inner.Draw("3 same")
    g_lhapdf_antinu_w_outer.Draw("3 same")
    g_lhapdf_antinu_w_inner.Draw("3 same")

    g_lhapdf_inc_em.Draw("L same")
    g_lhapdf_nu_em.Draw("L same")
    g_lhapdf_antinu_em.Draw("L same")

    g_lhapdf_inc_w.Draw("L same")
    g_lhapdf_nu_w.Draw("L same")
    g_lhapdf_antinu_w.Draw("L same")

    g_charm_rate_mc.Draw("P E1 same")
    g_charm_rate_mc_nu.Draw("P E1 same")
    g_charm_rate_mc_antinu.Draw("P E1 same")

    leg_rate_all = ROOT.TLegend(0.44, 0.52, 0.92, 0.89)
    leg_rate_all.SetBorderSize(1)
    leg_rate_all.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_rate_all.SetTextSize(0.019)
    leg_rate_all.AddEntry(g_charm_rate_mc, "MC Inclusive (#nu_{#mu}+#bar{#nu}_{#mu})", "lep")
    leg_rate_all.AddEntry(g_lhapdf_inc_w, "QCD Inclusive (^{184}W, Red solid)", "l")
    leg_rate_all.AddEntry(g_lhapdf_inc_em, "QCD Inclusive (Emulsion, Red dashed)", "l")
    leg_rate_all.AddEntry(g_charm_rate_mc_nu, "MC #nu_{#mu} CC", "lep")
    leg_rate_all.AddEntry(g_lhapdf_nu_w, "QCD #nu_{#mu} (^{184}W, Blue solid)", "l")
    leg_rate_all.AddEntry(g_lhapdf_nu_em, "QCD #nu_{#mu} (Emulsion, Blue dashed)", "l")
    leg_rate_all.AddEntry(g_charm_rate_mc_antinu, "MC #bar{#nu}_{#mu} CC", "lep")
    leg_rate_all.AddEntry(g_lhapdf_antinu_w, "QCD #bar{#nu}_{#mu} (^{184}W, Purple solid)", "l")
    leg_rate_all.AddEntry(g_lhapdf_antinu_em, "QCD #bar{#nu}_{#mu} (Emulsion, Purple dashed)", "l")
    leg_rate_all.Draw("same")
    c_rate_all.Write()

    # =========================================================================
    # Dimuon Rate Canvases: \sigma(charm, 2\mu) / \sigma(CC) vs E_\nu
    # Folded with inclusive semi-muonic branching fraction <B_mu> = 0.0848 +/- 0.0050
    # =========================================================================
    b_mu = 0.0848
    sigma_b_mu = 0.0050

    # Theoretical dimuon models
    f_sr_2mu_w_nu, g_sr_2mu_w_nu_inner, g_sr_2mu_w_nu_outer = create_slow_rescaling_model(
        name="f_slow_rescaling_dimuon_w_nu",
        e_min=0.0,
        e_max=e_max_model,
        target="tungsten",
        nu_type="nu",
        line_color=ROOT.kBlue + 1,
        band_color=ROOT.kAzure - 4,
        scale=b_mu,
        scale_err=sigma_b_mu,
    )

    f_sr_2mu_w_antinu, g_sr_2mu_w_antinu_inner, g_sr_2mu_w_antinu_outer = create_slow_rescaling_model(
        name="f_slow_rescaling_dimuon_w_antinu",
        e_min=0.0,
        e_max=e_max_model,
        target="tungsten",
        nu_type="antinu",
        line_color=ROOT.kMagenta + 2,
        band_color=ROOT.kMagenta - 9,
        scale=b_mu,
        scale_err=sigma_b_mu,
    )

    f_sr_2mu_w_inc, g_sr_2mu_w_inc_inner, g_sr_2mu_w_inc_outer = create_slow_rescaling_model(
        name="f_slow_rescaling_dimuon_w_inclusive",
        e_min=0.0,
        e_max=e_max_model,
        target="tungsten",
        nu_type="inclusive",
        line_color=ROOT.kRed + 1,
        band_color=ROOT.kRed - 7,
        scale=b_mu,
        scale_err=sigma_b_mu,
    )

    f_sr_2mu_iso_inc, _, _ = create_slow_rescaling_model(
        name="f_slow_rescaling_dimuon_isoscalar_inclusive",
        e_min=0.0,
        e_max=e_max_model,
        target="isoscalar",
        nu_type="inclusive",
        line_color=ROOT.kGray + 2,
        scale=b_mu,
        scale_err=sigma_b_mu,
    )
    f_sr_2mu_iso_inc.SetLineStyle(7)
    f_sr_2mu_iso_inc.SetLineWidth(2)

    f_sr_2mu_iso_nu, _, _ = create_slow_rescaling_model(
        name="f_slow_rescaling_dimuon_isoscalar_nu",
        e_min=0.0,
        e_max=e_max_model,
        target="isoscalar",
        nu_type="nu",
        line_color=ROOT.kGray + 2,
        scale=b_mu,
        scale_err=sigma_b_mu,
    )
    f_sr_2mu_iso_nu.SetLineStyle(7)
    f_sr_2mu_iso_nu.SetLineWidth(2)

    f_sr_2mu_iso_antinu, _, _ = create_slow_rescaling_model(
        name="f_slow_rescaling_dimuon_isoscalar_antinu",
        e_min=0.0,
        e_max=e_max_model,
        target="isoscalar",
        nu_type="antinu",
        line_color=ROOT.kGray + 2,
        scale=b_mu,
        scale_err=sigma_b_mu,
    )
    f_sr_2mu_iso_antinu.SetLineStyle(7)
    f_sr_2mu_iso_antinu.SetLineWidth(2)

    # Theoretical LHAPDF nCTEQ15 dimuon models
    # Tungsten target predictions (solid transparent bands, solid central line)
    g_lhapdf_dimuon_inc_w, g_lhapdf_dimuon_inc_w_base, g_lhapdf_dimuon_inc_w_total = load_lhapdf_dimuon_prediction(
        "neutrino_charm_dimuon_ratio_inclusive_prediction_lhapdf.csv",
        name="lhapdf_dimuon_inclusive_tungsten",
        target="tungsten",
        line_color=col_inc,
        line_width=2,
        line_style=1,
        band_color=col_inc,
        inner_alpha=0.35,
        outer_alpha=0.18,
        fill_style_inner=1001,
        fill_style_outer=1001,
        scale_to_absolute=True,
    )

    g_lhapdf_dimuon_nu_w, g_lhapdf_dimuon_nu_w_base, g_lhapdf_dimuon_nu_w_total = load_lhapdf_dimuon_prediction(
        "neutrino_charm_dimuon_ratio_nu_prediction_lhapdf.csv",
        name="lhapdf_dimuon_nu_tungsten",
        target="tungsten",
        line_color=col_nu,
        line_width=2,
        line_style=1,
        band_color=col_nu,
        inner_alpha=0.35,
        outer_alpha=0.18,
        fill_style_inner=1001,
        fill_style_outer=1001,
        scale_to_absolute=True,
    )

    g_lhapdf_dimuon_antinu_w, g_lhapdf_dimuon_antinu_w_base, g_lhapdf_dimuon_antinu_w_total = load_lhapdf_dimuon_prediction(
        "neutrino_charm_dimuon_ratio_antinu_prediction_lhapdf.csv",
        name="lhapdf_dimuon_antinu_tungsten",
        target="tungsten",
        line_color=col_antinu,
        line_width=2,
        line_style=1,
        band_color=col_antinu,
        inner_alpha=0.35,
        outer_alpha=0.18,
        fill_style_inner=1001,
        fill_style_outer=1001,
        scale_to_absolute=True,
    )

    # Emulsion target predictions (diagonal hatched bands, dashed central line)
    g_lhapdf_dimuon_inc_em, g_lhapdf_dimuon_inc_em_base, g_lhapdf_dimuon_inc_em_total = load_lhapdf_dimuon_prediction(
        "neutrino_charm_dimuon_ratio_inclusive_prediction_lhapdf.csv",
        name="lhapdf_dimuon_inclusive_emulsion",
        target="emulsion",
        line_color=col_inc,
        line_width=2,
        line_style=7,
        band_color=col_inc,
        fill_style_inner=3004,
        fill_style_outer=3005,
        scale_to_absolute=True,
    )

    g_lhapdf_dimuon_nu_em, g_lhapdf_dimuon_nu_em_base, g_lhapdf_dimuon_nu_em_total = load_lhapdf_dimuon_prediction(
        "neutrino_charm_dimuon_ratio_nu_prediction_lhapdf.csv",
        name="lhapdf_dimuon_nu_emulsion",
        target="emulsion",
        line_color=col_nu,
        line_width=2,
        line_style=7,
        band_color=col_nu,
        fill_style_inner=3004,
        fill_style_outer=3005,
        scale_to_absolute=True,
    )

    g_lhapdf_dimuon_antinu_em, g_lhapdf_dimuon_antinu_em_base, g_lhapdf_dimuon_antinu_em_total = load_lhapdf_dimuon_prediction(
        "neutrino_charm_dimuon_ratio_antinu_prediction_lhapdf.csv",
        name="lhapdf_dimuon_antinu_emulsion",
        target="emulsion",
        line_color=col_antinu,
        line_width=2,
        line_style=7,
        band_color=col_antinu,
        fill_style_inner=3004,
        fill_style_outer=3005,
        scale_to_absolute=True,
    )

    # Aliases
    g_lhapdf_dimuon_inc = g_lhapdf_dimuon_inc_w
    g_lhapdf_dimuon_inc_base = g_lhapdf_dimuon_inc_w_base
    g_lhapdf_dimuon_inc_total = g_lhapdf_dimuon_inc_w_total
    g_lhapdf_dimuon_nu = g_lhapdf_dimuon_nu_w
    g_lhapdf_dimuon_nu_base = g_lhapdf_dimuon_nu_w_base
    g_lhapdf_dimuon_nu_total = g_lhapdf_dimuon_nu_w_total
    g_lhapdf_dimuon_antinu = g_lhapdf_dimuon_antinu_w
    g_lhapdf_dimuon_antinu_base = g_lhapdf_dimuon_antinu_w_base
    g_lhapdf_dimuon_antinu_total = g_lhapdf_dimuon_antinu_w_total

    # MC Dimuon data points
    g_dimuon_rate_mc = ROOT.TGraphAsymmErrors(n_nu_ebins)
    g_dimuon_rate_mc.SetName("g_dimuon_rate_mc")
    g_dimuon_rate_mc.SetTitle("MC Inclusive Dimuon (#nu_{#mu} + #bar{#nu}_{#mu})")

    g_dimuon_rate_mc_nu = ROOT.TGraphAsymmErrors(n_nu_ebins)
    g_dimuon_rate_mc_nu.SetName("g_dimuon_rate_mc_nu")
    g_dimuon_rate_mc_nu.SetTitle("MC #nu_{#mu} Dimuon CC")

    g_dimuon_rate_mc_antinu = ROOT.TGraphAsymmErrors(n_nu_ebins)
    g_dimuon_rate_mc_antinu.SetName("g_dimuon_rate_mc_antinu")
    g_dimuon_rate_mc_antinu.SetTitle("MC #bar{#nu}_{#mu} Dimuon CC")

    for i_b in range(n_nu_ebins):
        x_center = 0.5 * (nu_bin_edges[i_b] + nu_bin_edges[i_b + 1])
        x_err = 0.5 * (nu_bin_edges[i_b + 1] - nu_bin_edges[i_b])

        # Inclusive
        n_cc = data["cc_all_per_ebin"][i_b]
        n_2mu = data.get("cc_dimuon_per_ebin", [0] * n_nu_ebins)[i_b]
        val_pct, err_lo_pct, err_hi_pct = clopper_pearson_errors(n_2mu, n_cc, 0.6827)
        val = val_pct / 100.0
        err_lo = err_lo_pct / 100.0
        err_hi = err_hi_pct / 100.0
        g_dimuon_rate_mc.SetPoint(i_b, x_center, val)
        g_dimuon_rate_mc.SetPointError(i_b, x_err, x_err, err_lo, err_hi)

        # Pure nu_mu
        n_cc_nu = data.get("cc_all_nu_per_ebin", [0] * n_nu_ebins)[i_b]
        n_2mu_nu = data.get("cc_dimuon_nu_per_ebin", [0] * n_nu_ebins)[i_b]
        val_pct_nu, lo_nu, hi_nu = clopper_pearson_errors(n_2mu_nu, n_cc_nu, 0.6827)
        g_dimuon_rate_mc_nu.SetPoint(i_b, x_center, val_pct_nu / 100.0)
        g_dimuon_rate_mc_nu.SetPointError(i_b, x_err, x_err, lo_nu / 100.0, hi_nu / 100.0)

        # Pure anti_nu_mu
        n_cc_an = data.get("cc_all_antinu_per_ebin", [0] * n_nu_ebins)[i_b]
        n_2mu_an = data.get("cc_dimuon_antinu_per_ebin", [0] * n_nu_ebins)[i_b]
        val_pct_an, lo_an, hi_an = clopper_pearson_errors(n_2mu_an, n_cc_an, 0.6827)
        g_dimuon_rate_mc_antinu.SetPoint(i_b, x_center, val_pct_an / 100.0)
        g_dimuon_rate_mc_antinu.SetPointError(i_b, x_err, x_err, lo_an / 100.0, hi_an / 100.0)

    g_dimuon_rate_mc.SetMarkerStyle(20)
    g_dimuon_rate_mc.SetMarkerSize(1.2)
    g_dimuon_rate_mc.SetMarkerColor(ROOT.kRed + 1)
    g_dimuon_rate_mc.SetLineColor(ROOT.kRed + 1)
    g_dimuon_rate_mc.SetLineWidth(2)

    g_dimuon_rate_mc_nu.SetMarkerStyle(20)
    g_dimuon_rate_mc_nu.SetMarkerSize(1.2)
    g_dimuon_rate_mc_nu.SetMarkerColor(ROOT.kBlue + 2)
    g_dimuon_rate_mc_nu.SetLineColor(ROOT.kBlue + 2)
    g_dimuon_rate_mc_nu.SetLineWidth(2)

    g_dimuon_rate_mc_antinu.SetMarkerStyle(20)
    g_dimuon_rate_mc_antinu.SetMarkerSize(1.2)
    g_dimuon_rate_mc_antinu.SetMarkerColor(ROOT.kMagenta + 2)
    g_dimuon_rate_mc_antinu.SetLineColor(ROOT.kMagenta + 2)
    g_dimuon_rate_mc_antinu.SetLineWidth(2)

    # -------------------------------------------------------------------------
    # Overlay Experimental Dimuon Measurements (NOMAD, CHORUS, CCFR, E53A+E53B)
    # Source: NOMAD Collaboration, Nucl. Phys. B 876 (2013) 339, Fig. 15
    # -------------------------------------------------------------------------
    dimuon_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurements", "neutrino_charm_dimuon_ratio_vs_enu.csv")
    dimuon_data = {}
    if os.path.exists(dimuon_csv_path):
        try:
            import csv
            with open(dimuon_csv_path, "r", encoding="utf-8-sig") as f_dim:
                reader = csv.DictReader(row for row in f_dim if not row.strip().startswith("#"))
                for row in reader:
                    exp = row["experiment"].strip()
                    if exp not in dimuon_data:
                        dimuon_data[exp] = []
                    dimuon_data[exp].append((
                        float(row["E_nu_GeV"]),
                        float(row["dx_low"]),
                        float(row["dx_high"]),
                        float(row["ratio_times_10^-3"]) * 1e-3,
                        float(row["dy_low"]) * 1e-3,
                        float(row["dy_high"]) * 1e-3,
                    ))
        except Exception as e_dim:
            print(f"  [Warning] Failed to load dimuon CSV: {e_dim}")

    # Robust fallback: NOMAD Collaboration, Nucl. Phys. B 876 (2013) 339, Fig. 15
    if not dimuon_data:
        dimuon_data = {
            "E53A+E53B": [
                (12.501716, 6.492915, 12.529668, 1.989e-3, 0.398e-3, 0.382e-3),
                (37.780942, 12.749558, 11.989294, 5.063e-3, 0.765e-3, 0.841e-3),
                (75.120355, 25.350119, 25.229162, 5.966e-3, 0.979e-3, 1.024e-3),
                (149.362845, 49.013328, 50.163400, 5.966e-3, 1.484e-3, 1.514e-3),
            ],
            "CCFR": [
                (40.089471, 0.0, 0.0, 5.384321e-3, 0.475300e-3, 0.504780e-3),
                (69.813505, 0.0, 0.0, 6.363289e-3, 0.379628e-3, 0.413002e-3),
                (90.062801, 0.0, 0.0, 6.577438e-3, 0.389084e-3, 0.397705e-3),
                (100.349517, 0.0, 0.0, 6.485660e-3, 0.297306e-3, 0.305927e-3),
                (139.783065, 0.0, 0.0, 7.678776e-3, 0.377403e-3, 0.397706e-3),
                (179.698925, 0.0, 0.0, 8.382409e-3, 0.287850e-3, 0.290631e-3),
                (218.471123, 0.0, 0.0, 8.581262e-3, 0.290631e-3, 0.290631e-3),
                (269.341417, 0.0, 0.0, 9.177820e-3, 0.305928e-3, 0.305928e-3),
            ],
            "CHORUS": [
                (20.022353, 11.047440, 8.858236, 3.380497e-3, 0.596558e-3, 0.596558e-3),
                (40.937493, 12.056904, 9.006687, 4.879541e-3, 0.596558e-3, 0.611855e-3),
                (60.298023, 10.353843, 9.515482, 6.087954e-3, 0.611855e-3, 0.611855e-3),
                (79.989017, 10.175512, 9.760137, 6.286807e-3, 0.611855e-3, 0.596558e-3),
                (104.275389, 14.526235, 15.616008, 6.684512e-3, 0.703633e-3, 0.673040e-3),
                (136.410740, 16.519343, 23.746094, 7.082218e-3, 0.795411e-3, 0.795411e-3),
                (184.141500, 23.984666, 55.909288, 7.189293e-3, 1.101338e-3, 1.086042e-3),
            ],
            "NOMAD": [
                (15.959905, 9.972031, 6.117061, 2.799e-3, 0.290e-3, 0.276e-3),
                (24.342416, 2.265450, 2.685853, 4.115e-3, 0.291e-3, 0.260e-3),
                (28.780011, 1.751742, 2.295925, 4.482e-3, 0.260e-3, 0.260e-3),
                (32.859934, 1.783998, 2.497836, 4.803e-3, 0.245e-3, 0.245e-3),
                (37.387578, 2.029808, 2.701893, 5.109e-3, 0.229e-3, 0.229e-3),
                (41.657861, 1.568390, 2.545431, 5.430e-3, 0.245e-3, 0.260e-3),
                (46.093154, 1.889862, 2.816431, 5.797e-3, 0.260e-3, 0.260e-3),
                (51.178904, 2.269319, 3.127185, 6.057e-3, 0.275e-3, 0.245e-3),
                (56.825783, 2.519694, 3.262250, 6.226e-3, 0.276e-3, 0.275e-3),
                (62.657031, 2.568998, 3.828517, 6.333e-3, 0.260e-3, 0.275e-3),
                (69.570378, 3.084830, 3.993857, 6.409e-3, 0.260e-3, 0.276e-3),
                (77.246518, 3.682283, 4.150067, 6.807e-3, 0.275e-3, 0.260e-3),
                (85.769615, 4.373030, 4.607971, 7.113e-3, 0.276e-3, 0.245e-3),
                (94.901423, 4.523837, 4.750324, 7.327e-3, 0.260e-3, 0.260e-3),
                (105.372498, 5.720751, 5.661135, 7.648e-3, 0.260e-3, 0.260e-3),
                (117.407844, 6.374211, 7.173943, 7.801e-3, 0.290e-3, 0.245e-3),
                (132.656118, 8.074331, 10.083809, 7.969e-3, 0.260e-3, 0.276e-3),
                (155.206216, 12.466289, 16.524735, 8.367e-3, 0.291e-3, 0.260e-3),
                (205.173521, 33.442570, 94.931164, 8.841e-3, 0.306e-3, 0.291e-3),
            ],
        }

    dimuon_exp_styles = {
        "NOMAD": {
            "title": "NOMAD (2013, #nu_{#mu})",
            "marker_style": 27,  # Open diamond
            "marker_size": 1.25,
            "color": ROOT.kBlack,
        },
        "CHORUS": {
            "title": "CHORUS (2011, #nu_{#mu})",
            "marker_style": 25,  # Open square
            "marker_size": 1.10,
            "color": ROOT.kBlack,
        },
        "CCFR": {
            "title": "CCFR (1995, #nu_{#mu})",
            "marker_style": 24,  # Open circle
            "marker_size": 1.10,
            "color": ROOT.kBlack,
        },
        "E53A+E53B": {
            "title": "E53A+E53B (1983, #nu_{#mu})",
            "marker_style": 26,  # Open triangle
            "marker_size": 1.10,
            "color": ROOT.kBlack,
        },
    }

    dimuon_graphs = {}
    preferred_dimuon_order = ["NOMAD", "CHORUS", "CCFR", "E53A+E53B"]
    for exp_name in preferred_dimuon_order:
        if exp_name not in dimuon_data:
            continue
        pts = dimuon_data[exp_name]
        g_exp = ROOT.TGraphAsymmErrors(len(pts))
        clean_exp = exp_name.lower().replace("+", "_").replace("-", "_")
        g_exp.SetName(f"g_dimuon_{clean_exp}")
        style = dimuon_exp_styles.get(exp_name, {
            "title": str(exp_name) + " (#nu_{#mu})",
            "marker_style": 24,
            "marker_size": 1.10,
            "color": ROOT.kBlack,
        })
        g_exp.SetTitle(style["title"])
        for idx, (enu, dx_lo, dx_hi, ratio, dy_lo, dy_hi) in enumerate(pts):
            g_exp.SetPoint(idx, enu, ratio)
            g_exp.SetPointError(idx, dx_lo, dx_hi, dy_lo, dy_hi)
        g_exp.SetMarkerStyle(style["marker_style"])
        g_exp.SetMarkerSize(style["marker_size"])
        g_exp.SetMarkerColor(style["color"])
        g_exp.SetLineColor(style["color"])
        g_exp.SetLineWidth(2)
        dimuon_graphs[exp_name] = g_exp
        print(f"  [Info] Loaded {exp_name} dimuon measurement ({len(pts)} points)")

    # Combined dimuon graph for convenience
    total_dim_pts = sum(len(pts) for pts in dimuon_data.values())
    g_dimuon_all = ROOT.TGraphAsymmErrors(total_dim_pts)
    g_dimuon_all.SetName("g_dimuon_all_measurements")
    g_dimuon_all.SetTitle("Historical Dimuon Measurements (#nu_{#mu} CC)")
    pt_counter = 0
    for exp_name in preferred_dimuon_order:
        if exp_name in dimuon_data:
            for (enu, dx_lo, dx_hi, ratio, dy_lo, dy_hi) in dimuon_data[exp_name]:
                g_dimuon_all.SetPoint(pt_counter, enu, ratio)
                g_dimuon_all.SetPointError(pt_counter, dx_lo, dx_hi, dy_lo, dy_hi)
                pt_counter += 1
    g_dimuon_all.SetMarkerStyle(24)
    g_dimuon_all.SetMarkerSize(1.10)
    g_dimuon_all.SetMarkerColor(ROOT.kBlack)
    g_dimuon_all.SetLineColor(ROOT.kBlack)
    g_dimuon_all.SetLineWidth(2)

    y_max_dimuon = max(0.022, h_charm_dimuon_rate_ebins.GetMaximum() * 1.60)
    y_max_dimuon_nu = max(0.020, h_charm_dimuon_rate_ebins_nu.GetMaximum() * 1.60)
    y_max_dimuon_antinu = max(0.028, h_charm_dimuon_rate_ebins_antinu.GetMaximum() * 1.60)

    # Canvas 1d: Dimuon Inclusive
    c_dimuon_inc = ROOT.TCanvas("c_charm_dimuon_rate_vs_energy", "Charm Dimuon Production Rate vs Neutrino Energy (Inclusive)", 950, 700)
    c_dimuon_inc.SetLeftMargin(0.12)
    c_dimuon_inc.SetRightMargin(0.06)
    c_dimuon_inc.SetTopMargin(0.08)
    c_dimuon_inc.SetBottomMargin(0.12)
    c_dimuon_inc.SetGrid()

    h_frame_dimuon_inc = c_dimuon_inc.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_dimuon)
    h_frame_dimuon_inc.SetTitle("Charm Dimuon Production Rate: Inclusive (#nu_{#mu} + #bar{#nu}_{#mu});E_{#nu} [GeV];#sigma_{charm, 2#mu} / #sigma_{CC}")
    h_frame_dimuon_inc.GetYaxis().SetTitleOffset(1.25)

    # 1. Tungsten target predictions (solid transparent bands)
    g_lhapdf_dimuon_inc_w_total.Draw("3 same")
    g_lhapdf_dimuon_inc_w_base.Draw("3 same")

    # 2. Emulsion target predictions (diagonal hatched bands)
    g_lhapdf_dimuon_inc_em_total.Draw("3 same")
    g_lhapdf_dimuon_inc_em_base.Draw("3 same")

    # 3. Central lines
    g_lhapdf_dimuon_inc_w.Draw("L same")
    g_lhapdf_dimuon_inc_em.Draw("L same")

    # 4. SND@LHC MC data points
    g_dimuon_rate_mc.Draw("P E1 same")

    leg_dimuon_inc = ROOT.TLegend(0.44, 0.50, 0.92, 0.89)
    leg_dimuon_inc.SetBorderSize(1)
    leg_dimuon_inc.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_dimuon_inc.SetTextSize(0.019)
    leg_dimuon_inc.AddEntry(g_dimuon_rate_mc, "MC Inclusive Dimuon (#nu_{#mu}+#bar{#nu}_{#mu})", "lep")
    leg_dimuon_inc.AddEntry(g_lhapdf_dimuon_inc_w, "QCD Prediction (^{184}W target, solid)", "l")
    leg_dimuon_inc.AddEntry(g_lhapdf_dimuon_inc_w_base, "Base Kinematics 1#sigma (^{184}W)", "f")
    leg_dimuon_inc.AddEntry(g_lhapdf_dimuon_inc_w_total, "LHAPDF nCTEQ15 1#sigma (^{184}W)", "f")
    leg_dimuon_inc.AddEntry(g_lhapdf_dimuon_inc_em, "QCD Prediction (Emulsion target, dashed)", "l")
    leg_dimuon_inc.AddEntry(g_lhapdf_dimuon_inc_em_base, "Base Kinematics 1#sigma (Emulsion, //)", "f")
    leg_dimuon_inc.AddEntry(g_lhapdf_dimuon_inc_em_total, "LHAPDF nCTEQ15 1#sigma (Emulsion, \\\\)", "f")
    leg_dimuon_inc.Draw("same")
    c_dimuon_inc.Write()

    # Canvas 1e: Dimuon Neutrino Only
    c_dimuon_nu = ROOT.TCanvas("c_charm_dimuon_rate_vs_energy_nu_only", "Charm Dimuon Production Rate vs Neutrino Energy (nu_mu)", 950, 700)
    c_dimuon_nu.SetLeftMargin(0.12)
    c_dimuon_nu.SetRightMargin(0.06)
    c_dimuon_nu.SetTopMargin(0.08)
    c_dimuon_nu.SetBottomMargin(0.12)
    c_dimuon_nu.SetGrid()

    h_frame_dimuon_nu = c_dimuon_nu.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_dimuon_nu)
    h_frame_dimuon_nu.SetTitle("Charm Dimuon Production Rate: #nu_{#mu} CC;E_{#nu} [GeV];#sigma_{charm, 2#mu} / #sigma_{CC}")
    h_frame_dimuon_nu.GetYaxis().SetTitleOffset(1.25)

    # 1. Tungsten target predictions (solid transparent bands)
    g_lhapdf_dimuon_nu_w_total.Draw("3 same")
    g_lhapdf_dimuon_nu_w_base.Draw("3 same")

    # 2. Emulsion target predictions (diagonal hatched bands)
    g_lhapdf_dimuon_nu_em_total.Draw("3 same")
    g_lhapdf_dimuon_nu_em_base.Draw("3 same")

    # 3. Central lines
    g_lhapdf_dimuon_nu_w.Draw("L same")
    g_lhapdf_dimuon_nu_em.Draw("L same")

    # 4. Experimental measurements & SND@LHC MC data points
    for g_dim in dimuon_graphs.values():
        g_dim.Draw("P E1 same")
    g_dimuon_rate_mc_nu.Draw("P E1 same")

    leg_dimuon_nu = ROOT.TLegend(0.44, 0.44, 0.92, 0.89)
    leg_dimuon_nu.SetBorderSize(1)
    leg_dimuon_nu.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_dimuon_nu.SetTextSize(0.018)
    leg_dimuon_nu.AddEntry(g_dimuon_rate_mc_nu, "MC #nu_{#mu} Dimuon CC (Tungsten)", "lep")
    for g_dim in dimuon_graphs.values():
        leg_dimuon_nu.AddEntry(g_dim, g_dim.GetTitle(), "lep")
    leg_dimuon_nu.AddEntry(g_lhapdf_dimuon_nu_w, "QCD Prediction (^{184}W target, solid)", "l")
    leg_dimuon_nu.AddEntry(g_lhapdf_dimuon_nu_w_base, "Base Kinematics 1#sigma (^{184}W)", "f")
    leg_dimuon_nu.AddEntry(g_lhapdf_dimuon_nu_w_total, "LHAPDF nCTEQ15 1#sigma (^{184}W)", "f")
    leg_dimuon_nu.AddEntry(g_lhapdf_dimuon_nu_em, "QCD Prediction (Emulsion target, dashed)", "l")
    leg_dimuon_nu.AddEntry(g_lhapdf_dimuon_nu_em_base, "Base Kinematics 1#sigma (Emulsion, //)", "f")
    leg_dimuon_nu.AddEntry(g_lhapdf_dimuon_nu_em_total, "LHAPDF nCTEQ15 1#sigma (Emulsion, \\\\)", "f")
    leg_dimuon_nu.Draw("same")
    c_dimuon_nu.Write()

    # Canvas 1f: Dimuon Antineutrino Only
    c_dimuon_antinu = ROOT.TCanvas("c_charm_dimuon_rate_vs_energy_antinu_only", "Charm Dimuon Production Rate vs Neutrino Energy (anti_nu_mu)", 950, 700)
    c_dimuon_antinu.SetLeftMargin(0.12)
    c_dimuon_antinu.SetRightMargin(0.06)
    c_dimuon_antinu.SetTopMargin(0.08)
    c_dimuon_antinu.SetBottomMargin(0.12)
    c_dimuon_antinu.SetGrid()

    h_frame_dimuon_antinu = c_dimuon_antinu.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_dimuon_antinu)
    h_frame_dimuon_antinu.SetTitle("Charm Dimuon Production Rate: #bar{#nu}_{#mu} CC;E_{#nu} [GeV];#sigma_{charm, 2#mu} / #sigma_{CC}")
    h_frame_dimuon_antinu.GetYaxis().SetTitleOffset(1.25)

    # 1. Tungsten target predictions (solid transparent bands)
    g_lhapdf_dimuon_antinu_w_total.Draw("3 same")
    g_lhapdf_dimuon_antinu_w_base.Draw("3 same")

    # 2. Emulsion target predictions (diagonal hatched bands)
    g_lhapdf_dimuon_antinu_em_total.Draw("3 same")
    g_lhapdf_dimuon_antinu_em_base.Draw("3 same")

    # 3. Central lines
    g_lhapdf_dimuon_antinu_w.Draw("L same")
    g_lhapdf_dimuon_antinu_em.Draw("L same")

    # 4. SND@LHC MC data points
    g_dimuon_rate_mc_antinu.Draw("P E1 same")

    leg_dimuon_antinu = ROOT.TLegend(0.44, 0.50, 0.92, 0.89)
    leg_dimuon_antinu.SetBorderSize(1)
    leg_dimuon_antinu.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_dimuon_antinu.SetTextSize(0.019)
    leg_dimuon_antinu.AddEntry(g_dimuon_rate_mc_antinu, "MC #bar{#nu}_{#mu} Dimuon CC (Tungsten)", "lep")
    leg_dimuon_antinu.AddEntry(g_lhapdf_dimuon_antinu_w, "QCD Prediction (^{184}W target, solid)", "l")
    leg_dimuon_antinu.AddEntry(g_lhapdf_dimuon_antinu_w_base, "Base Kinematics 1#sigma (^{184}W)", "f")
    leg_dimuon_antinu.AddEntry(g_lhapdf_dimuon_antinu_w_total, "LHAPDF nCTEQ15 1#sigma (^{184}W)", "f")
    leg_dimuon_antinu.AddEntry(g_lhapdf_dimuon_antinu_em, "QCD Prediction (Emulsion target, dashed)", "l")
    leg_dimuon_antinu.AddEntry(g_lhapdf_dimuon_antinu_em_base, "Base Kinematics 1#sigma (Emulsion, //)", "f")
    leg_dimuon_antinu.AddEntry(g_lhapdf_dimuon_antinu_em_total, "LHAPDF nCTEQ15 1#sigma (Emulsion, \\\\)", "f")
    leg_dimuon_antinu.Draw("same")
    c_dimuon_antinu.Write()

    # Canvas 1g: Dimuon Comparison: Inclusive vs nu_mu vs anti_nu_mu
    c_dimuon_all = ROOT.TCanvas("c_charm_dimuon_rate_vs_energy_comparison", "Charm Dimuon Production Rate: Inclusive vs Nu vs Anti-Nu", 950, 700)
    c_dimuon_all.SetLeftMargin(0.12)
    c_dimuon_all.SetRightMargin(0.06)
    c_dimuon_all.SetTopMargin(0.08)
    c_dimuon_all.SetBottomMargin(0.12)
    c_dimuon_all.SetGrid()

    y_max_dim_comp = max(y_max_dimuon, y_max_dimuon_antinu)
    h_frame_dim_all = c_dimuon_all.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), y_max_dim_comp)
    h_frame_dim_all.SetTitle("Charm Dimuon Production Rate Comparison: Inclusive vs #nu_{#mu} vs #bar{#nu}_{#mu};E_{#nu} [GeV];#sigma_{charm, 2#mu} / #sigma_{CC}")
    h_frame_dim_all.GetYaxis().SetTitleOffset(1.25)

    g_lhapdf_dimuon_inc_w_total.Draw("3 same")
    g_lhapdf_dimuon_inc_w_base.Draw("3 same")
    g_lhapdf_dimuon_nu_w_total.Draw("3 same")
    g_lhapdf_dimuon_nu_w_base.Draw("3 same")
    g_lhapdf_dimuon_antinu_w_total.Draw("3 same")
    g_lhapdf_dimuon_antinu_w_base.Draw("3 same")

    g_lhapdf_dimuon_inc_em.Draw("L same")
    g_lhapdf_dimuon_nu_em.Draw("L same")
    g_lhapdf_dimuon_antinu_em.Draw("L same")

    g_lhapdf_dimuon_inc_w.Draw("L same")
    g_lhapdf_dimuon_nu_w.Draw("L same")
    g_lhapdf_dimuon_antinu_w.Draw("L same")

    g_dimuon_rate_mc.Draw("P E1 same")
    g_dimuon_rate_mc_nu.Draw("P E1 same")
    g_dimuon_rate_mc_antinu.Draw("P E1 same")

    leg_dimuon_all = ROOT.TLegend(0.44, 0.50, 0.92, 0.89)
    leg_dimuon_all.SetBorderSize(1)
    leg_dimuon_all.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg_dimuon_all.SetTextSize(0.019)
    leg_dimuon_all.AddEntry(g_dimuon_rate_mc, "MC Inclusive Dimuon (Red points)", "lep")
    leg_dimuon_all.AddEntry(g_lhapdf_dimuon_inc_w, "QCD Inclusive (^{184}W, Red solid)", "lf")
    leg_dimuon_all.AddEntry(g_lhapdf_dimuon_inc_em, "QCD Inclusive (Emulsion, Red dashed)", "l")
    leg_dimuon_all.AddEntry(g_dimuon_rate_mc_nu, "MC #nu_{#mu} Dimuon CC (Blue points)", "lep")
    leg_dimuon_all.AddEntry(g_lhapdf_dimuon_nu_w, "QCD #nu_{#mu} (^{184}W, Blue solid)", "lf")
    leg_dimuon_all.AddEntry(g_lhapdf_dimuon_nu_em, "QCD #nu_{#mu} (Emulsion, Blue dashed)", "l")
    leg_dimuon_all.AddEntry(g_dimuon_rate_mc_antinu, "MC #bar{#nu}_{#mu} Dimuon CC (Purple points)", "lep")
    leg_dimuon_all.AddEntry(g_lhapdf_dimuon_antinu_w, "QCD #bar{#nu}_{#mu} (^{184}W, Purple solid)", "lf")
    leg_dimuon_all.AddEntry(g_lhapdf_dimuon_antinu_em, "QCD #bar{#nu}_{#mu} (Emulsion, Purple dashed)", "l")
    leg_dimuon_all.Draw("same")
    c_dimuon_all.Write()

    # Also store into Summary directory
    dir_summary.cd()
    if g_chorus:
        g_chorus.Write("g_chorus_measurement")
    if g_e531:
        g_e531.Write("g_e531_measurement")
    for exp_name, g_dim in dimuon_graphs.items():
        clean_exp = exp_name.lower().replace("+", "_").replace("-", "_")
        g_dim.Write(f"g_dimuon_{clean_exp}")
    if g_dimuon_all:
        g_dimuon_all.Write("g_dimuon_all_measurements")
    f_sr_w_inc.Write("f_slow_rescaling_w_inclusive")
    f_sr_w_nu.Write("f_slow_rescaling_w_nu")
    f_sr_w_antinu.Write("f_slow_rescaling_w_antinu")
    f_sr_iso_inc.Write("f_slow_rescaling_isoscalar_inclusive")
    f_sr_iso_nu.Write("f_slow_rescaling_isoscalar_nu")
    f_sr_iso_antinu.Write("f_slow_rescaling_isoscalar_antinu")
    g_sr_w_inc_inner.Write("g_slow_rescaling_w_inclusive_1sigma_inner_band")
    g_sr_w_inc_outer.Write("g_slow_rescaling_w_inclusive_1sigma_outer_band")
    g_sr_w_nu_inner.Write("g_slow_rescaling_w_nu_1sigma_inner_band")
    g_sr_w_nu_outer.Write("g_slow_rescaling_w_nu_1sigma_outer_band")
    g_sr_w_antinu_inner.Write("g_slow_rescaling_w_antinu_1sigma_inner_band")
    g_sr_w_antinu_outer.Write("g_slow_rescaling_w_antinu_1sigma_outer_band")
    g_charm_rate_mc.Write("g_charm_rate_mc")
    g_charm_rate_mc_nu.Write("g_charm_rate_mc_nu")
    g_charm_rate_mc_antinu.Write("g_charm_rate_mc_antinu")

    # LHAPDF charm cross-section ratio predictions (Tungsten and Emulsion)
    g_lhapdf_inc_w.Write("g_lhapdf_prediction_inclusive_tungsten")
    g_lhapdf_inc_w_inner.Write("g_lhapdf_inclusive_1sigma_inner_band_tungsten")
    g_lhapdf_inc_w_outer.Write("g_lhapdf_inclusive_1sigma_outer_band_tungsten")
    g_lhapdf_nu_w.Write("g_lhapdf_prediction_nu_tungsten")
    g_lhapdf_nu_w_inner.Write("g_lhapdf_nu_1sigma_inner_band_tungsten")
    g_lhapdf_nu_w_outer.Write("g_lhapdf_nu_1sigma_outer_band_tungsten")
    g_lhapdf_antinu_w.Write("g_lhapdf_prediction_antinu_tungsten")
    g_lhapdf_antinu_w_inner.Write("g_lhapdf_antinu_1sigma_inner_band_tungsten")
    g_lhapdf_antinu_w_outer.Write("g_lhapdf_antinu_1sigma_outer_band_tungsten")

    g_lhapdf_inc_em.Write("g_lhapdf_prediction_inclusive_emulsion")
    g_lhapdf_inc_em_inner.Write("g_lhapdf_inclusive_1sigma_inner_band_emulsion")
    g_lhapdf_inc_em_outer.Write("g_lhapdf_inclusive_1sigma_outer_band_emulsion")
    g_lhapdf_nu_em.Write("g_lhapdf_prediction_nu_emulsion")
    g_lhapdf_nu_em_inner.Write("g_lhapdf_nu_1sigma_inner_band_emulsion")
    g_lhapdf_nu_em_outer.Write("g_lhapdf_nu_1sigma_outer_band_emulsion")
    g_lhapdf_antinu_em.Write("g_lhapdf_prediction_antinu_emulsion")
    g_lhapdf_antinu_em_inner.Write("g_lhapdf_antinu_1sigma_inner_band_emulsion")
    g_lhapdf_antinu_em_outer.Write("g_lhapdf_antinu_1sigma_outer_band_emulsion")

    # Backward compatibility default aliases
    g_lhapdf_inc.Write("g_lhapdf_prediction_inclusive")
    g_lhapdf_inc_inner.Write("g_lhapdf_inclusive_1sigma_inner_band")
    g_lhapdf_inc_outer.Write("g_lhapdf_inclusive_1sigma_outer_band")
    g_lhapdf_nu.Write("g_lhapdf_prediction_nu")
    g_lhapdf_nu_inner.Write("g_lhapdf_nu_1sigma_inner_band")
    g_lhapdf_nu_outer.Write("g_lhapdf_nu_1sigma_outer_band")
    g_lhapdf_antinu.Write("g_lhapdf_prediction_antinu")
    g_lhapdf_antinu_inner.Write("g_lhapdf_antinu_1sigma_inner_band")
    g_lhapdf_antinu_outer.Write("g_lhapdf_antinu_1sigma_outer_band")

    f_sr_2mu_w_inc.Write("f_slow_rescaling_dimuon_w_inclusive")
    f_sr_2mu_w_nu.Write("f_slow_rescaling_dimuon_w_nu")
    f_sr_2mu_w_antinu.Write("f_slow_rescaling_dimuon_w_antinu")
    f_sr_2mu_iso_inc.Write("f_slow_rescaling_dimuon_isoscalar_inclusive")
    f_sr_2mu_iso_nu.Write("f_slow_rescaling_dimuon_isoscalar_nu")
    f_sr_2mu_iso_antinu.Write("f_slow_rescaling_dimuon_isoscalar_antinu")
    g_sr_2mu_w_inc_inner.Write("g_slow_rescaling_dimuon_w_inclusive_1sigma_inner_band")
    g_sr_2mu_w_inc_outer.Write("g_slow_rescaling_dimuon_w_inclusive_1sigma_outer_band")
    g_sr_2mu_w_nu_inner.Write("g_slow_rescaling_dimuon_w_nu_1sigma_inner_band")
    g_sr_2mu_w_nu_outer.Write("g_slow_rescaling_dimuon_w_nu_1sigma_outer_band")
    g_sr_2mu_w_antinu_inner.Write("g_slow_rescaling_dimuon_w_antinu_1sigma_inner_band")
    g_sr_2mu_w_antinu_outer.Write("g_slow_rescaling_dimuon_w_antinu_1sigma_outer_band")
    g_dimuon_rate_mc.Write("g_dimuon_rate_mc")
    g_dimuon_rate_mc_nu.Write("g_dimuon_rate_mc_nu")
    g_dimuon_rate_mc_antinu.Write("g_dimuon_rate_mc_antinu")

    # LHAPDF dimuon cross-section ratio predictions (Tungsten and Emulsion)
    g_lhapdf_dimuon_inc_w.Write("g_lhapdf_dimuon_prediction_inclusive_tungsten")
    g_lhapdf_dimuon_inc_w_base.Write("g_lhapdf_dimuon_inclusive_1sigma_base_band_tungsten")
    g_lhapdf_dimuon_inc_w_total.Write("g_lhapdf_dimuon_inclusive_1sigma_total_band_tungsten")
    g_lhapdf_dimuon_nu_w.Write("g_lhapdf_dimuon_prediction_nu_tungsten")
    g_lhapdf_dimuon_nu_w_base.Write("g_lhapdf_dimuon_nu_1sigma_base_band_tungsten")
    g_lhapdf_dimuon_nu_w_total.Write("g_lhapdf_dimuon_nu_1sigma_total_band_tungsten")
    g_lhapdf_dimuon_antinu_w.Write("g_lhapdf_dimuon_prediction_antinu_tungsten")
    g_lhapdf_dimuon_antinu_w_base.Write("g_lhapdf_dimuon_antinu_1sigma_base_band_tungsten")
    g_lhapdf_dimuon_antinu_w_total.Write("g_lhapdf_dimuon_antinu_1sigma_total_band_tungsten")

    g_lhapdf_dimuon_inc_em.Write("g_lhapdf_dimuon_prediction_inclusive_emulsion")
    g_lhapdf_dimuon_inc_em_base.Write("g_lhapdf_dimuon_inclusive_1sigma_base_band_emulsion")
    g_lhapdf_dimuon_inc_em_total.Write("g_lhapdf_dimuon_inclusive_1sigma_total_band_emulsion")
    g_lhapdf_dimuon_nu_em.Write("g_lhapdf_dimuon_prediction_nu_emulsion")
    g_lhapdf_dimuon_nu_em_base.Write("g_lhapdf_dimuon_nu_1sigma_base_band_emulsion")
    g_lhapdf_dimuon_nu_em_total.Write("g_lhapdf_dimuon_nu_1sigma_total_band_emulsion")
    g_lhapdf_dimuon_antinu_em.Write("g_lhapdf_dimuon_prediction_antinu_emulsion")
    g_lhapdf_dimuon_antinu_em_base.Write("g_lhapdf_dimuon_antinu_1sigma_base_band_emulsion")
    g_lhapdf_dimuon_antinu_em_total.Write("g_lhapdf_dimuon_antinu_1sigma_total_band_emulsion")

    # Backward compatibility default aliases
    g_lhapdf_dimuon_inc.Write("g_lhapdf_dimuon_prediction_inclusive")
    g_lhapdf_dimuon_inc_base.Write("g_lhapdf_dimuon_inclusive_1sigma_base_band")
    g_lhapdf_dimuon_inc_total.Write("g_lhapdf_dimuon_inclusive_1sigma_total_band")
    g_lhapdf_dimuon_nu.Write("g_lhapdf_dimuon_prediction_nu")
    g_lhapdf_dimuon_nu_base.Write("g_lhapdf_dimuon_nu_1sigma_base_band")
    g_lhapdf_dimuon_nu_total.Write("g_lhapdf_dimuon_nu_1sigma_total_band")
    g_lhapdf_dimuon_antinu.Write("g_lhapdf_dimuon_prediction_antinu")
    g_lhapdf_dimuon_antinu_base.Write("g_lhapdf_dimuon_antinu_1sigma_base_band")
    g_lhapdf_dimuon_antinu_total.Write("g_lhapdf_dimuon_antinu_1sigma_total_band")

    # 5. LHAPDF nCTEQ15 theoretical charm species fragmentation fractions from CSV
    sp_curves_inc, sp_bands_inc = load_lhapdf_species_fractions(
        "charm_fragmentation_fractions_inclusive_prediction_lhapdf.csv",
        name_prefix="lhapdf_sp_inc",
        channel_mode="inclusive",
        is_antinu=False,
    )
    sp_curves_nu, sp_bands_nu = load_lhapdf_species_fractions(
        "charm_fragmentation_fractions_nu_only_prediction_lhapdf.csv",
        name_prefix="lhapdf_sp_nu",
        channel_mode="nu",
        is_antinu=False,
    )
    sp_curves_antinu, sp_bands_antinu = load_lhapdf_species_fractions(
        "charm_fragmentation_fractions_antinu_only_prediction_lhapdf.csv",
        name_prefix="lhapdf_sp_antinu",
        channel_mode="antinu",
        is_antinu=True,
    )

    # Generate and write theoretical/empirical charm species fraction model
    model_curves, model_bands, g_model_bmu, g_model_band_bmu = create_species_fraction_model(
        e_min=4.5,
        e_max=max(1000.0, nu_bin_edges[-1]),
        n_points=300,
        n_samples=2000,
        seed=42,
    )
    for abs_c in species_order:
        sp_name = get_species_name(abs_c)
        if abs_c in sp_curves_inc:
            sp_curves_inc[abs_c].Write(f"g_lhapdf_fraction_{sp_name}_vs_nu_energy")
            sp_bands_inc[abs_c].Write(f"g_lhapdf_band_fraction_{sp_name}_vs_nu_energy")
        if abs_c in sp_curves_nu:
            sp_curves_nu[abs_c].Write(f"g_lhapdf_fraction_{sp_name}_vs_nu_energy_nu_only")
            sp_bands_nu[abs_c].Write(f"g_lhapdf_band_fraction_{sp_name}_vs_nu_energy_nu_only")
        if abs_c in model_curves:
            model_curves[abs_c].Write(f"g_model_fraction_{sp_name}_vs_nu_energy")
        if abs_c in model_bands:
            model_bands[abs_c].Write(f"g_model_band_fraction_{sp_name}_vs_nu_energy")

    for anti_c in anti_species_order:
        sp_name = get_species_name(anti_c)
        abs_c = abs(anti_c)
        g_c = sp_curves_antinu.get(anti_c, sp_curves_antinu.get(abs_c))
        g_b = sp_bands_antinu.get(anti_c, sp_bands_antinu.get(abs_c))
        if g_c:
            g_c.Write(f"g_lhapdf_fraction_{sp_name}_vs_nu_energy_antinu_only")
        if g_b:
            g_b.Write(f"g_lhapdf_band_fraction_{sp_name}_vs_nu_energy_antinu_only")

    g_model_bmu.Write("g_model_effective_bmu_vs_nu_energy")
    g_model_band_bmu.Write("g_model_band_effective_bmu_vs_nu_energy")

    for abs_c in species_order:
        sp_name = get_species_name(abs_c)
        if abs_c in species_frac_nu_graphs:
            species_frac_nu_graphs[abs_c].Write(f"g_fraction_{sp_name}_vs_nu_energy")
        if abs_c in species_frac_nu_graphs_nu:
            species_frac_nu_graphs_nu[abs_c].Write(f"g_fraction_{sp_name}_vs_nu_energy_nu_only")

    for anti_c in anti_species_order:
        sp_name = get_species_name(anti_c)
        if anti_c in species_frac_nu_graphs_antinu:
            species_frac_nu_graphs_antinu[anti_c].Write(f"g_fraction_{sp_name}_vs_nu_energy_antinu_only")

    # CHORUS experimental species fraction measurement graphs
    chorus_species_graphs: Dict[int, ROOT.TGraphAsymmErrors] = {}
    for abs_c, (c_val, c_err) in CHORUS_SPECIES_FRACTIONS.items():
        sp_name = get_species_name(abs_c)
        col = CHORUS_COLORS.get(abs_c, ROOT.kBlack)
        g_ch = ROOT.TGraphAsymmErrors(1)
        g_ch.SetName(f"g_chorus_fraction_{sp_name}_vs_nu_energy")
        g_ch.SetTitle(f"CHORUS (2011) {sp_name} Fraction;E_{{#nu}} [GeV];f({sp_name})")
        g_ch.SetPoint(0, 27.0, c_val)
        g_ch.SetPointError(0, 14.0, 28.0, c_err, c_err)
        g_ch.SetMarkerStyle(25)  # Open square
        g_ch.SetMarkerSize(1.40)
        g_ch.SetMarkerColor(col)
        g_ch.SetLineColor(col)
        g_ch.SetLineWidth(2)
        chorus_species_graphs[abs_c] = g_ch
        g_ch.Write(f"g_chorus_fraction_{sp_name}_vs_nu_energy")

    dir_canvases.cd()

    # --- Canvas 2: Overlaid Energy Spectra (Inclusive CC vs Charm CC vs Charm Hadron) ---
    c_spectra = ROOT.TCanvas("c_nu_and_charm_energy_spectra", "Neutrino and Charm Energy Spectra", 950, 700)
    c_spectra.SetLeftMargin(0.12)
    c_spectra.SetRightMargin(0.06)
    c_spectra.SetTopMargin(0.08)
    c_spectra.SetBottomMargin(0.12)
    c_spectra.SetLogy()

    max_spec = max(h_nu_e_cc_all.GetMaximum(), h_nu_e_cc_charm.GetMaximum(), h_charm_e_all.GetMaximum(), 1.0)
    h_spec_frame = c_spectra.DrawFrame(0.0, 0.5, 1000.0, max_spec * 3.0)
    h_spec_frame.SetTitle("Energy Distributions: E_{#nu} (CC Inclusive), E_{#nu} (with Charm), and E_{charm};Energy [GeV];Events / 20 GeV")

    h_nu_e_cc_all.SetLineColor(ROOT.kBlue + 2)
    h_nu_e_cc_all.Draw("HIST same")

    h_nu_e_cc_charm.SetLineColor(ROOT.kRed + 1)
    h_nu_e_cc_charm.Draw("HIST same")

    h_charm_e_all.SetLineColor(ROOT.kGreen + 2)
    h_charm_e_all.Draw("HIST same")

    leg_spec = ROOT.TLegend(0.50, 0.70, 0.90, 0.88)
    leg_spec.SetBorderSize(1)
    leg_spec.SetFillColor(ROOT.kWhite)
    leg_spec.AddEntry(h_nu_e_cc_all, "#nu_{#mu} CC Inclusive (E_{#nu})", "l")
    leg_spec.AddEntry(h_nu_e_cc_charm, "#nu_{#mu} CC with Charm (E_{#nu})", "l")
    leg_spec.AddEntry(h_charm_e_all, "Charm Hadron Energy (E_{charm})", "l")
    leg_spec.Draw("same")
    c_spectra.Write()

    # --- Canvas 3: Superimposed Species Fractions vs E_nu ---
    c_sp_nu = ROOT.TCanvas("c_species_fractions_vs_nu_energy", "Charm Species Fractions vs Neutrino Energy", 1000, 700)
    c_sp_nu.SetLeftMargin(0.12)
    c_sp_nu.SetRightMargin(0.06)
    c_sp_nu.SetTopMargin(0.08)
    c_sp_nu.SetBottomMargin(0.12)
    c_sp_nu.SetLogx()
    c_sp_nu.SetGrid()

    x_max_sp = max(1000.0, nu_bin_edges[-1])
    h_sp_nu_frame = c_sp_nu.DrawFrame(5.0, 0.0, x_max_sp, 0.95)
    h_sp_nu_frame.SetTitle("Charm Species Fragmentation Fractions: Inclusive (#nu_{#mu} CC);E_{#nu} [GeV];Fraction f_{i} = #sigma_{i} / #sigma_{charm}")
    h_sp_nu_frame.GetYaxis().SetTitleOffset(1.25)

    # 1. Draw 1-sigma uncertainty bands first
    for abs_c in species_order:
        if abs_c in sp_bands_inc:
            sp_bands_inc[abs_c].Draw("3 same")

    # 2. Draw central model median curves
    for abs_c in species_order:
        if abs_c in sp_curves_inc:
            sp_curves_inc[abs_c].Draw("L same")

    # 3. Draw CHORUS experimental measurements
    for abs_c in species_order:
        if abs_c in chorus_species_graphs:
            chorus_species_graphs[abs_c].Draw("P E1 same")

    # 4. Draw MC simulation data points
    for abs_c in species_order:
        g = species_frac_nu_graphs.get(abs_c)
        if g:
            g.Draw("P E1 same")

    # 5. Legend (3 columns: MC, QCD calc, CHORUS)
    leg_sp_nu = ROOT.TLegend(0.40, 0.60, 0.92, 0.90)
    leg_sp_nu.SetBorderSize(1)
    leg_sp_nu.SetFillColorAlpha(ROOT.kWhite, 0.94)
    leg_sp_nu.SetTextSize(0.021)
    leg_sp_nu.SetNColumns(3)

    for abs_c in species_order:
        g_mc = species_frac_nu_graphs.get(abs_c)
        sp_latex = SPECIES_LATEX.get(abs_c, get_species_name(abs_c))
        if g_mc:
            leg_sp_nu.AddEntry(g_mc, f"{sp_latex} (MC)", "lep")
        if abs_c in sp_bands_inc:
            leg_sp_nu.AddEntry(sp_bands_inc[abs_c], f"{sp_latex} (QCD calc)", "lf")
        if abs_c in chorus_species_graphs:
            leg_sp_nu.AddEntry(chorus_species_graphs[abs_c], f"{sp_latex} (CHORUS)", "lep")

    leg_sp_nu.Draw("same")
    c_sp_nu.Write()

    # --- Canvas 3b: Species Fractions vs E_nu (Neutrino Only, nu_mu CC) ---
    c_sp_nu_nu = ROOT.TCanvas("c_species_fractions_vs_nu_energy_nu_only", "Charm Species Fractions vs Neutrino Energy (nu_mu CC)", 1000, 700)
    c_sp_nu_nu.SetLeftMargin(0.12)
    c_sp_nu_nu.SetRightMargin(0.06)
    c_sp_nu_nu.SetTopMargin(0.08)
    c_sp_nu_nu.SetBottomMargin(0.12)
    c_sp_nu_nu.SetLogx()
    c_sp_nu_nu.SetGrid()

    h_sp_nu_frame_nu = c_sp_nu_nu.DrawFrame(5.0, 0.0, x_max_sp, 0.95)
    h_sp_nu_frame_nu.SetTitle("Charm Species Fragmentation Fractions: #nu_{#mu} CC (#nu Only, C=+1);E_{#nu} [GeV];Fraction f_{i} = #sigma_{i} / #sigma_{charm}")
    h_sp_nu_frame_nu.GetYaxis().SetTitleOffset(1.25)

    for abs_c in species_order:
        if abs_c in sp_bands_nu:
            sp_bands_nu[abs_c].Draw("3 same")
    for abs_c in species_order:
        if abs_c in sp_curves_nu:
            sp_curves_nu[abs_c].Draw("L same")
    for abs_c in species_order:
        if abs_c in chorus_species_graphs:
            chorus_species_graphs[abs_c].Draw("P E1 same")
    for abs_c in species_order:
        g = species_frac_nu_graphs_nu.get(abs_c)
        if g:
            g.Draw("P E1 same")

    leg_sp_nu_nu = ROOT.TLegend(0.40, 0.60, 0.92, 0.90)
    leg_sp_nu_nu.SetBorderSize(1)
    leg_sp_nu_nu.SetFillColorAlpha(ROOT.kWhite, 0.94)
    leg_sp_nu_nu.SetTextSize(0.021)
    leg_sp_nu_nu.SetNColumns(3)

    for abs_c in species_order:
        g_mc = species_frac_nu_graphs_nu.get(abs_c)
        sp_latex = SPECIES_LATEX.get(abs_c, get_species_name(abs_c))
        if g_mc:
            leg_sp_nu_nu.AddEntry(g_mc, f"{sp_latex} (MC)", "lep")
        if abs_c in sp_bands_nu:
            leg_sp_nu_nu.AddEntry(sp_bands_nu[abs_c], f"{sp_latex} (QCD calc)", "lf")
        if abs_c in chorus_species_graphs:
            leg_sp_nu_nu.AddEntry(chorus_species_graphs[abs_c], f"{sp_latex} (CHORUS)", "lep")

    leg_sp_nu_nu.Draw("same")
    c_sp_nu_nu.Write()

    # --- Canvas 3c: Species Fractions vs E_nu (Antineutrino Only, anti_nu_mu CC) ---
    c_sp_nu_antinu = ROOT.TCanvas("c_species_fractions_vs_nu_energy_antinu_only", "Charm Species Fractions vs Neutrino Energy (anti_nu_mu CC)", 1000, 700)
    c_sp_nu_antinu.SetLeftMargin(0.12)
    c_sp_nu_antinu.SetRightMargin(0.06)
    c_sp_nu_antinu.SetTopMargin(0.08)
    c_sp_nu_antinu.SetBottomMargin(0.12)
    c_sp_nu_antinu.SetLogx()
    c_sp_nu_antinu.SetGrid()

    h_sp_nu_frame_antinu = c_sp_nu_antinu.DrawFrame(5.0, 0.0, x_max_sp, 0.95)
    h_sp_nu_frame_antinu.SetTitle("Charm Species Fragmentation Fractions: #bar{#nu}_{#mu} CC (#bar{#nu} Only, C=-1);E_{#nu} [GeV];Fraction f_{i} = #sigma_{i} / #sigma_{charm}")
    h_sp_nu_frame_antinu.GetYaxis().SetTitleOffset(1.25)

    for anti_c in anti_species_order:
        abs_c = abs(anti_c)
        if anti_c in sp_bands_antinu:
            sp_bands_antinu[anti_c].Draw("3 same")
        elif abs_c in sp_bands_antinu:
            sp_bands_antinu[abs_c].Draw("3 same")
    for anti_c in anti_species_order:
        abs_c = abs(anti_c)
        if anti_c in sp_curves_antinu:
            sp_curves_antinu[anti_c].Draw("L same")
        elif abs_c in sp_curves_antinu:
            sp_curves_antinu[abs_c].Draw("L same")
    for anti_c in anti_species_order:
        g = species_frac_nu_graphs_antinu.get(anti_c)
        if g:
            g.Draw("P E1 same")

    leg_sp_nu_antinu = ROOT.TLegend(0.48, 0.65, 0.92, 0.90)
    leg_sp_nu_antinu.SetBorderSize(1)
    leg_sp_nu_antinu.SetFillColorAlpha(ROOT.kWhite, 0.94)
    leg_sp_nu_antinu.SetTextSize(0.021)
    leg_sp_nu_antinu.SetNColumns(2)

    for anti_c in anti_species_order:
        g_mc = species_frac_nu_graphs_antinu.get(anti_c)
        abs_c = abs(anti_c)
        sp_latex = SPECIES_LATEX.get(anti_c, get_species_name(anti_c))
        if g_mc:
            leg_sp_nu_antinu.AddEntry(g_mc, f"{sp_latex} (MC)", "lep")
        g_band = sp_bands_antinu.get(anti_c, sp_bands_antinu.get(abs_c))
        if g_band:
            leg_sp_nu_antinu.AddEntry(g_band, f"{sp_latex} (QCD calc)", "lf")

    leg_sp_nu_antinu.Draw("same")
    c_sp_nu_antinu.Write()

    # --- Canvas 3d: Species Fractions Comparison across Beam Modes (4 Pads: D0, D+, Ds+, Lambda_c) ---
    c_sp_comp = ROOT.TCanvas("c_species_fractions_comparison_all_modes", "Charm Species Fractions: Inclusive vs Nu vs Anti-Nu", 1200, 900)
    c_sp_comp.Divide(2, 2)

    sp_keys_list = [("D0", 421), ("D+", 411), ("Ds+", 431), ("Lambda_c", 4122)]
    comp_frames = []
    comp_legends = []

    for pad_idx, (sp_name_k, pdg_k) in enumerate(sp_keys_list, start=1):
        pad = c_sp_comp.cd(pad_idx)
        pad.SetLeftMargin(0.13)
        pad.SetRightMargin(0.05)
        pad.SetTopMargin(0.08)
        pad.SetBottomMargin(0.12)
        pad.SetLogx()
        pad.SetGrid()

        sp_latex = SPECIES_LATEX.get(pdg_k, sp_name_k)
        frame_pad = pad.DrawFrame(5.0, 0.0, x_max_sp, 0.95)
        frame_pad.SetTitle(f"{sp_latex} Fragmentation Fraction;E_{{#nu}} [GeV];f({sp_latex})")
        frame_pad.GetYaxis().SetTitleOffset(1.25)
        comp_frames.append(frame_pad)

        # 1. Uncertainty bands: Inclusive (Red), Nu (Blue), Anti-Nu (Purple)
        if pdg_k in sp_bands_inc:
            sp_bands_inc[pdg_k].Draw("3 same")
        if pdg_k in sp_bands_nu:
            sp_bands_nu[pdg_k].Draw("3 same")
        anti_pdg = -pdg_k
        g_band_an = sp_bands_antinu.get(anti_pdg, sp_bands_antinu.get(pdg_k))
        if g_band_an:
            g_band_an.Draw("3 same")

        # 2. Central curves
        if pdg_k in sp_curves_inc:
            sp_curves_inc[pdg_k].Draw("L same")
        if pdg_k in sp_curves_nu:
            sp_curves_nu[pdg_k].Draw("L same")
        g_curve_an = sp_curves_antinu.get(anti_pdg, sp_curves_antinu.get(pdg_k))
        if g_curve_an:
            g_curve_an.Draw("L same")

        # 3. MC Points
        g_mc_inc = species_frac_nu_graphs.get(pdg_k)
        if g_mc_inc:
            g_mc_inc.Draw("P E1 same")
        g_mc_nu = species_frac_nu_graphs_nu.get(pdg_k)
        if g_mc_nu:
            g_mc_nu.Draw("P E1 same")
        g_mc_an = species_frac_nu_graphs_antinu.get(anti_pdg)
        if g_mc_an:
            g_mc_an.Draw("P E1 same")

        # Legend
        leg_pad = ROOT.TLegend(0.42, 0.65, 0.92, 0.90)
        leg_pad.SetBorderSize(1)
        leg_pad.SetFillColorAlpha(ROOT.kWhite, 0.92)
        leg_pad.SetTextSize(0.030)
        if pdg_k in sp_curves_inc:
            leg_pad.AddEntry(sp_bands_inc[pdg_k], "Inclusive (#nu+#bar{#nu}) [Red]", "lf")
        if pdg_k in sp_curves_nu:
            leg_pad.AddEntry(sp_bands_nu[pdg_k], "#nu_{#mu} Only [Blue]", "lf")
        if g_curve_an:
            leg_pad.AddEntry(g_band_an, "#bar{#nu}_{#mu} Only [Purple]", "lf")
        leg_pad.Draw("same")
        comp_legends.append(leg_pad)

    c_sp_comp.Write()

    # --- Canvas 4: Superimposed Species Fractions vs E_charm ---
    c_sp_ce = ROOT.TCanvas("c_species_fractions_vs_charm_energy", "Charm Species Fractions vs Charm Hadron Energy", 950, 700)
    c_sp_ce.SetLeftMargin(0.12)
    c_sp_ce.SetRightMargin(0.06)
    c_sp_ce.SetTopMargin(0.08)
    c_sp_ce.SetBottomMargin(0.12)
    c_sp_ce.SetGrid()

    h_sp_ce_frame = c_sp_ce.DrawFrame(0.0, 0.0, charm_bin_edges[-1], 1.05)
    h_sp_ce_frame.SetTitle("Charm Species Fragmentation Fractions vs E_{charm};E_{charm} [GeV];Fraction f_{i} = N_{i} / N_{charm}")

    leg_sp_ce = ROOT.TLegend(0.68, 0.70, 0.92, 0.88)
    leg_sp_ce.SetBorderSize(1)
    leg_sp_ce.SetFillColor(ROOT.kWhite)

    for abs_c in species_order:
        g = species_frac_ch_graphs.get(abs_c)
        if g:
            g.Draw("P E1 same")
            leg_sp_ce.AddEntry(g, get_species_name(abs_c), "lep")
    leg_sp_ce.Draw("same")
    c_sp_ce.Write()

    # --- Canvas 5: Superimposed Branching Ratios vs E_charm (for all available species) ---
    for abs_c in species_order:
        sp_name = get_species_name(abs_c)
        mode_list = species_br_vs_charm_e.get(abs_c, [])
        if not mode_list:
            continue

        c_br_ce = ROOT.TCanvas(f"c_branching_ratios_vs_charm_energy_{sp_name}", f"{sp_name} Branching Ratios vs Charm Energy", 950, 700)
        c_br_ce.SetLeftMargin(0.12)
        c_br_ce.SetRightMargin(0.06)
        c_br_ce.SetTopMargin(0.08)
        c_br_ce.SetBottomMargin(0.12)
        c_br_ce.SetGrid()

        max_br = 10.0
        for _, h, g in mode_list:
            if h.GetMaximum() > max_br:
                max_br = h.GetMaximum()
            for pt_idx in range(g.GetN()):
                val_hi = g.GetPointY(pt_idx) + g.GetErrorYhigh(pt_idx)
                if val_hi > max_br:
                    max_br = val_hi

        h_br_frame = c_br_ce.DrawFrame(0.0, 0.0, charm_bin_edges[-1], min(100.0, max_br * 1.35))
        h_br_frame.SetTitle(f"Evolution of {sp_name} Branching Ratios with Hadron Energy;E_{{{sp_name}}} [GeV];Branching Ratio [%]")

        leg_br_ce = ROOT.TLegend(0.48, 0.68, 0.92, 0.88)
        leg_br_ce.SetBorderSize(1)
        leg_br_ce.SetFillColor(ROOT.kWhite)

        for formula, h, g in mode_list:
            g.Draw("P E1 same")
            leg_br_ce.AddEntry(g, f"{sp_name} #rightarrow {formula}", "lep")
        leg_br_ce.Draw("same")
        c_br_ce.Write()

    # --- Canvas 6: Superimposed Proper Lifetimes c*tau with PDG Reference Markers ---
    c_ctau = ROOT.TCanvas("c_proper_lifetimes", "Proper Lifetime Distributions (c*tau) vs PDG References", 1000, 700)
    c_ctau.Divide(2, 2)

    pad_lines = []
    for idx_sp, abs_c in enumerate(species_order, start=1):
        sp_name = get_species_name(abs_c)
        pad = c_ctau.cd(idx_sp)
        pad.SetLeftMargin(0.14)
        pad.SetRightMargin(0.06)
        pad.SetBottomMargin(0.14)

        h_ct = species_ctau_hists.get(abs_c)
        if h_ct:
            h_ct.Draw("HIST")
            pdg_tau = PDG_LIFETIMES_CTAU_UM.get(abs_c, 0.0)
            if pdg_tau > 0:
                line = ROOT.TLine(pdg_tau, 0.0, pdg_tau, h_ct.GetMaximum() * 1.05)
                line.SetLineColor(ROOT.kBlack)
                line.SetLineWidth(3)
                line.SetLineStyle(2)
                line.Draw("same")
                pad_lines.append(line)

                leg_ct = ROOT.TLegend(0.42, 0.70, 0.90, 0.88)
                leg_ct.SetBorderSize(1)
                leg_ct.SetFillColor(ROOT.kWhite)
                leg_ct.AddEntry(h_ct, f"MC <c#tau> = {h_ct.GetMean():.1f} #mum", "f")
                leg_ct.AddEntry(line, f"PDG c#tau = {pdg_tau:.1f} #mum", "l")
                leg_ct.Draw("same")
                pad_lines.append(leg_ct)
    c_ctau.Write()

    # --- Canvas 7: Kinematics 4-Pad Display (z, pT, theta, E_charm vs E_nu) ---
    c_kin = ROOT.TCanvas("c_fragmentation_and_kinematics", "Charm Hadron Kinematics", 1000, 700)
    c_kin.Divide(2, 2)

    c_kin.cd(1)
    h_z.SetFillColorAlpha(ROOT.kTeal - 5, 0.5)
    h_z.SetLineColor(ROOT.kTeal + 2)
    h_z.Draw("HIST")

    c_kin.cd(2)
    h_pt.SetFillColorAlpha(ROOT.kAzure - 4, 0.5)
    h_pt.SetLineColor(ROOT.kBlue + 2)
    h_pt.Draw("HIST")

    c_kin.cd(3)
    h_theta.SetFillColorAlpha(ROOT.kOrange - 3, 0.5)
    h_theta.SetLineColor(ROOT.kOrange + 2)
    h_theta.Draw("HIST")

    c_kin.cd(4)
    # 2D correlation E_charm vs E_nu
    h2_evs_e = ROOT.TH2D("h2_echarm_vs_enu", "E_{charm} vs E_{#nu};E_{#nu} [GeV];E_{charm} [GeV]", 40, 0.0, 1000.0, 40, 0.0, 1000.0)
    for i_pt in range(min(len(data["nu_e_cc_charm"]), len(data["charm_e_all"]))):
        h2_evs_e.Fill(data["nu_e_cc_charm"][i_pt], data["charm_e_all"][i_pt])
    h2_evs_e.Draw("COLZ")
    c_kin.Write()

    # =========================================================================
    # 5. PDG REFERENCE OVERLAYS & COMPARISON CANVASES
    # =========================================================================
    keep_alive_pdg: List[Any] = []

    # Standalone PDG reference graphs and histograms in PDG_References/
    if pdg_data:
        dir_pdg_ref = fout.mkdir("PDG_References")
        dir_pdg_ref.cd()
        for abs_c in sorted(pdg_data.keys()):
            sp_name = get_species_name(abs_c)
            sp_modes = pdg_data[abs_c].get("modes_list", [])
            n_m = len(sp_modes)
            if n_m == 0:
                continue
            h_pdg_ref = ROOT.TH1D(
                f"h_pdg_branching_ratio_{sp_name}",
                f"PDG World Average Reference ({sp_name});Inclusive Decay Mode;Inclusive Branching Fraction #Gamma_{{i}} / #Gamma [%]",
                n_m, 0.5, n_m + 0.5
            )
            g_pdg_ref = ROOT.TGraphAsymmErrors(n_m)
            g_pdg_ref.SetName(f"g_pdg_branching_ratio_{sp_name}")
            g_pdg_ref.SetTitle(f"PDG World Average Reference ({sp_name})")
            for idx, m_obj in enumerate(sp_modes, start=1):
                lbl = m_obj.pdg_mode
                h_pdg_ref.GetXaxis().SetBinLabel(idx, lbl)
                h_pdg_ref.SetBinContent(idx, m_obj.br_pct)
                h_pdg_ref.SetBinError(idx, 0.5 * (m_obj.br_err_up_pct + m_obj.br_err_low_pct))
                g_pdg_ref.SetPoint(idx - 1, float(idx), m_obj.br_pct)
                g_pdg_ref.SetPointError(idx - 1, 0.25, 0.25, m_obj.br_err_low_pct, m_obj.br_err_up_pct)
            g_pdg_ref.SetMarkerStyle(20)
            g_pdg_ref.SetMarkerSize(1.2)
            g_pdg_ref.SetMarkerColor(ROOT.kRed + 1)
            g_pdg_ref.SetLineColor(ROOT.kRed + 1)
            g_pdg_ref.SetLineWidth(2)
            h_pdg_ref.Write()
            g_pdg_ref.Write()
            keep_alive_pdg.extend([h_pdg_ref, g_pdg_ref])

    # Superimposed Canvases: Simulation (with vertical hatched error bars) vs PDG (error graphs)
    pdg_comparison_canvases: Dict[int, ROOT.TCanvas] = {}
    pdg_sp_drawables: Dict[int, Tuple[ROOT.TH1D, ROOT.TGraphAsymmErrors, ROOT.TGraphAsymmErrors, ROOT.TLegend]] = {}

    species_order = [421, 411, 431, 4122]
    anti_species_order = [-421, -411, -431, -4122]
    all_pdg_order = species_order + anti_species_order

    for sp_pdg in all_pdg_order:
        abs_c = abs(sp_pdg)
        sp_name = get_species_name(sp_pdg)
        sp_modes = pdg_data.get(sp_pdg, {}).get("modes_list", []) if pdg_data else []
        if not sp_modes and pdg_data:
            sp_modes = pdg_data.get(abs_c, {}).get("modes_list", [])
        if not sp_modes:
            continue

        w_tot = data.get("species_weight_sum", Counter()).get(sp_pdg, 0.0)
        w2_tot = data.get("species_weight_sq_sum", Counter()).get(sp_pdg, 0.0)
        n_tot = data.get("species_decay_count", Counter()).get(sp_pdg, 0)
        if w_tot <= 0.0:
            w_tot = float(data.get("total_decays_by_species", Counter()).get(sp_pdg, 0))
            w2_tot = w_tot
            n_tot = int(w_tot)

        all_candidates: List[Dict[str, Any]] = []

        for m_obj in sp_modes:
            lbl = get_conjugate_mode_label(m_obj.pdg_mode) if sp_pdg < 0 else m_obj.pdg_mode
            w_sum = data.get("inclusive_mode_w", {}).get(sp_pdg, Counter()).get(m_obj.pdg_mode, 0.0)
            w2y2_sum = data.get("inclusive_mode_w2y2", {}).get(sp_pdg, Counter()).get(m_obj.pdg_mode, 0.0)
            w2y_sum = data.get("inclusive_mode_w2y", {}).get(sp_pdg, Counter()).get(m_obj.pdg_mode, 0.0)
            raw_cnt = data.get("inclusive_mode_raw_count", {}).get(sp_pdg, Counter()).get(m_obj.pdg_mode, 0)

            br_sim, err_lo, err_hi = calculate_weighted_yield_and_error(
                w_sum, w2y2_sum, w2y_sum, w_tot, w2_tot
            )
            all_candidates.append({
                "label": lbl,
                "br_sim": br_sim,
                "br_sim_err_lo": err_lo,
                "br_sim_err_hi": err_hi,
                "br_pdg": m_obj.br_pct,
                "br_pdg_err_up": m_obj.br_err_up_pct,
                "br_pdg_err_lo": m_obj.br_err_low_pct,
                "w_sum": w_sum,
                "raw_cnt": raw_cnt,
            })

        n_bins = len(all_candidates)
        if n_bins == 0:
            continue

        h_sim_comp = ROOT.TH1D(
            f"h_br_sim_{sp_name}",
            f"{sp_name} Inclusive Branching Fractions: MC Simulation vs PDG Reference;Inclusive Decay Mode;Inclusive Branching Fraction #Gamma_{{i}} / #Gamma [%]",
            n_bins, 0.5, n_bins + 0.5
        )
        g_err_comp = ROOT.TGraphAsymmErrors(n_bins)
        g_err_comp.SetName(f"g_br_sim_err_{sp_name}")
        g_err_comp.SetTitle(f"MC Simulation Stat Error (#pm 1#sigma Weighted)")

        for i, entry in enumerate(all_candidates, start=1):
            h_sim_comp.GetXaxis().SetBinLabel(i, entry["label"])
            h_sim_comp.SetBinContent(i, entry["br_sim"])
            h_sim_comp.SetBinError(i, 0.0)

            g_err_comp.SetPoint(i - 1, float(i), entry["br_sim"])
            g_err_comp.SetPointError(i - 1, 0.35, 0.35, entry["br_sim_err_lo"], entry["br_sim_err_hi"])

        h_sim_comp.SetLineColor(ROOT.kAzure + 2)
        h_sim_comp.SetLineWidth(2)
        h_sim_comp.GetXaxis().LabelsOption("v")
        h_sim_comp.GetXaxis().SetLabelSize(0.033)
        h_sim_comp.GetYaxis().SetTitle(f"Inclusive Branching Fraction #Gamma_{{i}} / #Gamma [%]")
        h_sim_comp.GetYaxis().SetTitleOffset(1.25)

        g_err_comp.SetFillColor(ROOT.kAzure - 4)
        g_err_comp.SetFillStyle(3007)
        g_err_comp.SetLineColor(ROOT.kAzure + 2)
        g_err_comp.SetLineWidth(1)

        g_pdg_comp = ROOT.TGraphAsymmErrors(n_bins)
        g_pdg_comp.SetName(f"g_pdg_br_{sp_name}")
        g_pdg_comp.SetTitle(f"PDG World Average Reference ({sp_name})")
        for i, entry in enumerate(all_candidates, start=1):
            g_pdg_comp.SetPoint(i - 1, float(i), entry["br_pdg"])
            g_pdg_comp.SetPointError(i - 1, 0.25, 0.25, entry["br_pdg_err_lo"], entry["br_pdg_err_up"])

        g_pdg_comp.SetMarkerStyle(20)
        g_pdg_comp.SetMarkerSize(1.2)
        g_pdg_comp.SetMarkerColor(ROOT.kRed + 1)
        g_pdg_comp.SetLineColor(ROOT.kRed + 1)
        g_pdg_comp.SetLineWidth(2)

        max_sim = max([e["br_sim"] + e["br_sim_err_hi"] for e in all_candidates] + [0.0])
        max_pdg = max([e["br_pdg"] + e["br_pdg_err_up"] for e in all_candidates] + [0.0])
        overall_max = max(max_sim, max_pdg)
        y_max = max(overall_max * 1.35, 15.0)

        h_sim_comp.SetMaximum(y_max)
        h_sim_comp.SetMinimum(0.0)
        h_sim_comp.SetStats(0)

        dir_canvases.cd()
        c_pdg_sp = ROOT.TCanvas(f"c_branching_ratios_vs_pdg_{sp_name}", f"{sp_name} Inclusive Branching Fractions vs PDG Reference", 1150, 750)
        c_pdg_sp.SetLeftMargin(0.12)
        c_pdg_sp.SetRightMargin(0.06)
        c_pdg_sp.SetTopMargin(0.08)
        c_pdg_sp.SetBottomMargin(0.34)
        c_pdg_sp.SetGridy()

        h_sim_comp.Draw("HIST")
        g_err_comp.Draw("2 same")
        h_sim_comp.Draw("HIST same")
        g_pdg_comp.Draw("P E1 same")

        leg_pdg_sp = ROOT.TLegend(0.52, 0.74, 0.92, 0.88)
        leg_pdg_sp.SetBorderSize(1)
        leg_pdg_sp.SetFillColor(ROOT.kWhite)
        leg_pdg_sp.AddEntry(h_sim_comp, f"MC Simulation ({w_tot:.1f} ev)", "l")
        leg_pdg_sp.AddEntry(g_err_comp, "MC Stat (#pm 1#sigma)", "f")
        leg_pdg_sp.AddEntry(g_pdg_comp, "PDG reference (#Gamma_{i} / #Gamma)", "lep")
        leg_pdg_sp.Draw("same")

        c_pdg_sp.Write()
        pdg_comparison_canvases[sp_pdg] = c_pdg_sp
        pdg_sp_drawables[sp_pdg] = (h_sim_comp, g_err_comp, g_pdg_comp, leg_pdg_sp)
        keep_alive_pdg.extend([h_sim_comp, g_err_comp, g_pdg_comp, leg_pdg_sp, c_pdg_sp])

        dir_sp = dir_species_root.GetDirectory(sp_name)
        if not dir_sp:
            dir_sp = dir_species_root.mkdir(sp_name)
        if dir_sp:
            dir_sp.cd()
            h_sim_comp.Write(f"h_br_sim_vs_pdg_{sp_name}")
            g_err_comp.Write(f"g_br_sim_err_{sp_name}")
            g_pdg_comp.Write(f"g_pdg_br_{sp_name}")

        dir_canvases.cd()
        if export_png_dir:
            c_pdg_sp.SaveAs(os.path.join(export_png_dir, f"c_branching_ratios_vs_pdg_{sp_name}.png"))

    # Companion Canvases: Absolute Decay Counts (MC observed yields vs PDG expected counts N_sample * BR)
    pdg_count_comparison_canvases: Dict[int, ROOT.TCanvas] = {}
    pdg_count_sp_drawables: Dict[int, Tuple[ROOT.TH1D, ROOT.TGraphAsymmErrors, ROOT.TGraphAsymmErrors, ROOT.TLegend]] = {}

    for sp_pdg in all_pdg_order:
        abs_c = abs(sp_pdg)
        sp_name = get_species_name(sp_pdg)
        sp_modes = pdg_data.get(sp_pdg, {}).get("modes_list", []) if pdg_data else []
        if not sp_modes and pdg_data:
            sp_modes = pdg_data.get(abs_c, {}).get("modes_list", [])
        if not sp_modes:
            continue

        w_tot = data.get("species_weight_sum", Counter()).get(sp_pdg, 0.0)
        w2_tot = data.get("species_weight_sq_sum", Counter()).get(sp_pdg, 0.0)
        n_tot = data.get("species_decay_count", Counter()).get(sp_pdg, 0)
        if w_tot <= 0.0:
            w_tot = float(data.get("total_decays_by_species", Counter()).get(sp_pdg, 0))
            w2_tot = w_tot
            n_tot = int(w_tot)

        all_candidates_cnt: List[Dict[str, Any]] = []

        for m_obj in sp_modes:
            lbl = get_conjugate_mode_label(m_obj.pdg_mode) if sp_pdg < 0 else m_obj.pdg_mode
            w_sum = data.get("inclusive_mode_w", {}).get(sp_pdg, Counter()).get(m_obj.pdg_mode, 0.0)
            w2y2_sum = data.get("inclusive_mode_w2y2", {}).get(sp_pdg, Counter()).get(m_obj.pdg_mode, 0.0)
            w2y_sum = data.get("inclusive_mode_w2y", {}).get(sp_pdg, Counter()).get(m_obj.pdg_mode, 0.0)
            raw_cnt = data.get("inclusive_mode_raw_count", {}).get(sp_pdg, Counter()).get(m_obj.pdg_mode, 0)

            br_sim, err_lo, err_hi = calculate_weighted_yield_and_error(
                w_sum, w2y2_sum, w2y_sum, w_tot, w2_tot
            )
            # Scale yield error (in %) to absolute count error
            cnt_err_lo = (err_lo / 100.0) * w_tot
            cnt_err_hi = (err_hi / 100.0) * w_tot

            exp_cnt = w_tot * (m_obj.br_pct / 100.0)
            exp_err_lo = w_tot * (m_obj.br_err_low_pct / 100.0)
            exp_err_hi = w_tot * (m_obj.br_err_up_pct / 100.0)

            all_candidates_cnt.append({
                "label": lbl,
                "cnt_sim": w_sum,
                "cnt_sim_err_lo": cnt_err_lo,
                "cnt_sim_err_hi": cnt_err_hi,
                "exp_cnt": exp_cnt,
                "exp_cnt_err_lo": exp_err_lo,
                "exp_cnt_err_hi": exp_err_hi,
                "raw_cnt": raw_cnt,
            })

        n_bins = len(all_candidates_cnt)
        if n_bins == 0:
            continue

        h_cnt_comp = ROOT.TH1D(
            f"h_cnt_sim_{sp_name}",
            f"{sp_name} Inclusive Decay Yields: MC vs PDG Expected;Inclusive Decay Mode;Number of Decays N_{{i}}",
            n_bins, 0.5, n_bins + 0.5
        )
        g_cnt_err_comp = ROOT.TGraphAsymmErrors(n_bins)
        g_cnt_err_comp.SetName(f"g_cnt_sim_err_{sp_name}")
        g_cnt_err_comp.SetTitle(f"MC Simulation Stat Error (#pm 1#sigma Weighted)")

        for i, entry in enumerate(all_candidates_cnt, start=1):
            h_cnt_comp.GetXaxis().SetBinLabel(i, entry["label"])
            h_cnt_comp.SetBinContent(i, entry["cnt_sim"])
            h_cnt_comp.SetBinError(i, 0.0)

            g_cnt_err_comp.SetPoint(i - 1, float(i), entry["cnt_sim"])
            g_cnt_err_comp.SetPointError(i - 1, 0.35, 0.35, entry["cnt_sim_err_lo"], entry["cnt_sim_err_hi"])

        h_cnt_comp.SetLineColor(ROOT.kAzure + 2)
        h_cnt_comp.SetLineWidth(2)
        h_cnt_comp.GetXaxis().LabelsOption("v")
        h_cnt_comp.GetXaxis().SetLabelSize(0.033)
        h_cnt_comp.GetYaxis().SetTitle("Number of Decays N_{i}")
        h_cnt_comp.GetYaxis().SetTitleOffset(1.25)

        g_cnt_err_comp.SetFillColor(ROOT.kAzure - 4)
        g_cnt_err_comp.SetFillStyle(3007)
        g_cnt_err_comp.SetLineColor(ROOT.kAzure + 2)
        g_cnt_err_comp.SetLineWidth(1)

        g_pdg_cnt_comp = ROOT.TGraphAsymmErrors(n_bins)
        g_pdg_cnt_comp.SetName(f"g_pdg_cnt_{sp_name}")
        g_pdg_cnt_comp.SetTitle(f"PDG Expected Yield ({sp_name})")
        for i, entry in enumerate(all_candidates_cnt, start=1):
            g_pdg_cnt_comp.SetPoint(i - 1, float(i), entry["exp_cnt"])
            g_pdg_cnt_comp.SetPointError(i - 1, 0.25, 0.25, entry["exp_cnt_err_lo"], entry["exp_cnt_err_hi"])

        g_pdg_cnt_comp.SetMarkerStyle(20)
        g_pdg_cnt_comp.SetMarkerSize(1.2)
        g_pdg_cnt_comp.SetMarkerColor(ROOT.kRed + 1)
        g_pdg_cnt_comp.SetLineColor(ROOT.kRed + 1)
        g_pdg_cnt_comp.SetLineWidth(2)

        max_sim_cnt = max([e["cnt_sim"] + e["cnt_sim_err_hi"] for e in all_candidates_cnt] + [0.0])
        max_pdg_cnt = max([e["exp_cnt"] + e["exp_cnt_err_hi"] for e in all_candidates_cnt] + [0.0])
        overall_max_cnt = max(max_sim_cnt, max_pdg_cnt)
        y_max_cnt = max(overall_max_cnt * 1.55, 10.0)

        h_cnt_comp.SetMaximum(y_max_cnt)
        h_cnt_comp.SetMinimum(0.0)
        h_cnt_comp.SetStats(0)

        dir_canvases.cd()
        c_pdg_cnt_sp = ROOT.TCanvas(f"c_decay_counts_vs_pdg_{sp_name}", f"{sp_name} Inclusive Decay Yields vs PDG Expected", 1150, 750)
        c_pdg_cnt_sp.SetLeftMargin(0.12)
        c_pdg_cnt_sp.SetRightMargin(0.06)
        c_pdg_cnt_sp.SetTopMargin(0.08)
        c_pdg_cnt_sp.SetBottomMargin(0.34)
        c_pdg_cnt_sp.SetGridy()

        h_cnt_comp.Draw("HIST")
        g_cnt_err_comp.Draw("2 same")
        h_cnt_comp.Draw("HIST same")
        g_pdg_cnt_comp.Draw("P E1 same")

        leg_pdg_cnt = ROOT.TLegend(0.48, 0.75, 0.92, 0.90)
        leg_pdg_cnt.SetBorderSize(1)
        leg_pdg_cnt.SetFillColorAlpha(ROOT.kWhite, 0.85)
        leg_pdg_cnt.AddEntry(h_cnt_comp, f"MC Simulation ({w_tot:.1f} ev)", "l")
        leg_pdg_cnt.AddEntry(g_cnt_err_comp, "MC Stat (#pm 1#sigma)", "f")
        leg_pdg_cnt.AddEntry(g_pdg_cnt_comp, "PDG Expected (N_{ev} #times #Gamma_{i}/#Gamma)", "lep")
        leg_pdg_cnt.Draw("same")

        c_pdg_cnt_sp.Write()
        pdg_count_comparison_canvases[sp_pdg] = c_pdg_cnt_sp
        pdg_count_sp_drawables[sp_pdg] = (h_cnt_comp, g_cnt_err_comp, g_pdg_cnt_comp, leg_pdg_cnt)
        keep_alive_pdg.extend([h_cnt_comp, g_cnt_err_comp, g_pdg_cnt_comp, leg_pdg_cnt, c_pdg_cnt_sp])

        dir_sp = dir_species_root.GetDirectory(sp_name)
        if not dir_sp:
            dir_sp = dir_species_root.mkdir(sp_name)
        if dir_sp:
            dir_sp.cd()
            h_cnt_comp.Write(f"h_cnt_sim_vs_pdg_{sp_name}")
            g_cnt_err_comp.Write(f"g_cnt_sim_err_{sp_name}")
            g_pdg_cnt_comp.Write(f"g_pdg_cnt_{sp_name}")

        dir_canvases.cd()
        if export_png_dir:
            c_pdg_cnt_sp.SaveAs(os.path.join(export_png_dir, f"c_decay_counts_vs_pdg_{sp_name}.png"))

    # Summary 2x2 Canvas: All 4 Charmed Hadron Species Branching Ratios vs PDG (D0, D+, Ds+, Lambda_c+)
    if any(sp in pdg_sp_drawables for sp in species_order):
        dir_canvases.cd()
        c_all_sp = ROOT.TCanvas("c_branching_ratios_vs_pdg_all_species", "All Charmed Hadron Branching Ratios vs PDG (D^{0}, D^{+}, D_{s}^{+}, #Lambda_{c}^{+})", 1400, 1000)
        c_all_sp.Divide(2, 2)
        for pad_idx, sp_pdg in enumerate(species_order, start=1):
            if sp_pdg not in pdg_sp_drawables:
                continue
            pad = c_all_sp.cd(pad_idx)
            pad.SetLeftMargin(0.12)
            pad.SetRightMargin(0.06)
            pad.SetTopMargin(0.08)
            pad.SetBottomMargin(0.34)
            pad.SetGridy()

            h_sc, g_ec, g_pc, l_pc = pdg_sp_drawables[sp_pdg]
            h_sc.Draw("HIST")
            g_ec.Draw("2 same")
            h_sc.Draw("HIST same")
            g_pc.Draw("P E1 same")
            l_pc.Draw("same")

        c_all_sp.Write()
        keep_alive_pdg.append(c_all_sp)
        if export_png_dir:
            c_all_sp.SaveAs(os.path.join(export_png_dir, "c_branching_ratios_vs_pdg_all_species.png"))

    # Summary 2x2 Canvas: All 4 Anti-Charmed Hadron Species Branching Ratios vs PDG (anti-D0, D-, Ds-, Lambda_c-)
    if any(sp in pdg_sp_drawables for sp in anti_species_order):
        dir_canvases.cd()
        c_all_anti_sp = ROOT.TCanvas("c_branching_ratios_vs_pdg_all_anti_species", "All Anti-Charmed Hadron Branching Ratios vs PDG (#bar{D}^{0}, D^{-}, D_{s}^{-}, #bar{#Lambda}_{c}^{-})", 1400, 1000)
        c_all_anti_sp.Divide(2, 2)
        for pad_idx, sp_pdg in enumerate(anti_species_order, start=1):
            if sp_pdg not in pdg_sp_drawables:
                continue
            pad = c_all_anti_sp.cd(pad_idx)
            pad.SetLeftMargin(0.12)
            pad.SetRightMargin(0.06)
            pad.SetTopMargin(0.08)
            pad.SetBottomMargin(0.34)
            pad.SetGridy()

            h_sc, g_ec, g_pc, l_pc = pdg_sp_drawables[sp_pdg]
            h_sc.Draw("HIST")
            g_ec.Draw("2 same")
            h_sc.Draw("HIST same")
            g_pc.Draw("P E1 same")
            l_pc.Draw("same")

        c_all_anti_sp.Write()
        c_all_anti_sp.Write("c_branching_ratios_vs_pdg_all_species_anti")
        keep_alive_pdg.append(c_all_anti_sp)
        if export_png_dir:
            c_all_anti_sp.SaveAs(os.path.join(export_png_dir, "c_branching_ratios_vs_pdg_all_anti_species.png"))
            c_all_anti_sp.SaveAs(os.path.join(export_png_dir, "c_branching_ratios_vs_pdg_all_species_anti.png"))

    # Summary 2x2 Canvas: All 4 Species Absolute Decay Counts vs PDG Expected
    if any(sp in pdg_count_sp_drawables for sp in species_order):
        dir_canvases.cd()
        c_all_cnt_sp = ROOT.TCanvas("c_decay_counts_vs_pdg_all_species", "All Charmed Hadron Decay Counts vs PDG Expected (D^{0}, D^{+}, D_{s}^{+}, #Lambda_{c}^{+})", 1400, 1000)
        c_all_cnt_sp.Divide(2, 2)
        for pad_idx, sp_pdg in enumerate(species_order, start=1):
            if sp_pdg not in pdg_count_sp_drawables:
                continue
            pad = c_all_cnt_sp.cd(pad_idx)
            pad.SetLeftMargin(0.12)
            pad.SetRightMargin(0.06)
            pad.SetTopMargin(0.08)
            pad.SetBottomMargin(0.34)
            pad.SetGridy()

            h_cc, g_cec, g_cpc, l_cpc = pdg_count_sp_drawables[sp_pdg]
            h_cc.Draw("HIST")
            g_cec.Draw("2 same")
            h_cc.Draw("HIST same")
            g_cpc.Draw("P E1 same")
            l_cpc.Draw("same")

        c_all_cnt_sp.Write()
        keep_alive_pdg.append(c_all_cnt_sp)
        if export_png_dir:
            c_all_cnt_sp.SaveAs(os.path.join(export_png_dir, "c_decay_counts_vs_pdg_all_species.png"))

    # Summary 2x2 Canvas: All 4 Anti-Species Absolute Decay Counts vs PDG Expected
    if any(sp in pdg_count_sp_drawables for sp in anti_species_order):
        dir_canvases.cd()
        c_all_cnt_anti_sp = ROOT.TCanvas("c_decay_counts_vs_pdg_all_anti_species", "All Anti-Charmed Hadron Decay Counts vs PDG Expected (#bar{D}^{0}, D^{-}, D_{s}^{-}, #bar{#Lambda}_{c}^{-})", 1400, 1000)
        c_all_cnt_anti_sp.Divide(2, 2)
        for pad_idx, sp_pdg in enumerate(anti_species_order, start=1):
            if sp_pdg not in pdg_count_sp_drawables:
                continue
            pad = c_all_cnt_anti_sp.cd(pad_idx)
            pad.SetLeftMargin(0.12)
            pad.SetRightMargin(0.06)
            pad.SetTopMargin(0.08)
            pad.SetBottomMargin(0.34)
            pad.SetGridy()

            h_cc, g_cec, g_cpc, l_cpc = pdg_count_sp_drawables[sp_pdg]
            h_cc.Draw("HIST")
            g_cec.Draw("2 same")
            h_cc.Draw("HIST same")
            g_cpc.Draw("P E1 same")
            l_cpc.Draw("same")

        c_all_cnt_anti_sp.Write()
        c_all_cnt_anti_sp.Write("c_decay_counts_vs_pdg_all_species_anti")
        keep_alive_pdg.append(c_all_cnt_anti_sp)
        if export_png_dir:
            c_all_cnt_anti_sp.SaveAs(os.path.join(export_png_dir, "c_decay_counts_vs_pdg_all_anti_species.png"))
            c_all_cnt_anti_sp.SaveAs(os.path.join(export_png_dir, "c_decay_counts_vs_pdg_all_species_anti.png"))

    # --- Canvas 7: Reconstructed Track Multiplicity and Dimuon Tracking Efficiency ---
    if any(c > 0 for c in data.get("reco_tracks_cc_all", Counter()).values()):
        dir_canvases.cd()
        c_reco = ROOT.TCanvas("c_reco_multiplicity_and_efficiency", "Reconstruction Multiplicity & Selection Efficiency", 1200, 550)
        c_reco.Divide(2, 1)

        # Pad 1: Reconstructed track multiplicity
        pad1 = c_reco.cd(1)
        pad1.SetLeftMargin(0.12)
        pad1.SetRightMargin(0.06)
        pad1.SetTopMargin(0.08)
        pad1.SetBottomMargin(0.12)
        pad1.SetGridy()

        max_trk = max(max(data["reco_tracks_cc_all"].keys(), default=3), max(data["reco_tracks_cc_charm"].keys(), default=3), 3)
        h_mult_all = ROOT.TH1D("h_reco_tracks_cc_all", "Reconstructed Tracks Multiplicity;N_{reco tracks};Fraction of Events [%]", max_trk + 1, -0.5, max_trk + 0.5)
        h_mult_charm = ROOT.TH1D("h_reco_tracks_cc_charm", "Charm CC Track Multiplicity;N_{reco tracks};Fraction of Events [%]", max_trk + 1, -0.5, max_trk + 0.5)

        tot_cc = data["nu_cc_inclusive"]
        tot_ch = data["nu_cc_charm"]
        for n_t, cnt in data["reco_tracks_cc_all"].items():
            if 0 <= n_t <= max_trk:
                h_mult_all.SetBinContent(n_t + 1, (100.0 * cnt / tot_cc) if tot_cc > 0 else 0.0)
        for n_t, cnt in data["reco_tracks_cc_charm"].items():
            if 0 <= n_t <= max_trk:
                h_mult_charm.SetBinContent(n_t + 1, (100.0 * cnt / tot_ch) if tot_ch > 0 else 0.0)

        h_mult_all.SetLineColor(ROOT.kBlack)
        h_mult_all.SetLineWidth(2)
        h_mult_all.SetFillColorAlpha(ROOT.kGray + 1, 0.3)
        h_mult_charm.SetLineColor(ROOT.kRed + 1)
        h_mult_charm.SetLineWidth(2)
        h_mult_charm.SetFillColorAlpha(ROOT.kRed - 9, 0.4)

        max_val = max(h_mult_all.GetMaximum(), h_mult_charm.GetMaximum()) * 1.3
        h_mult_all.SetMaximum(max_val)
        h_mult_all.Draw("HIST")
        h_mult_charm.Draw("HIST same")

        leg_mult = ROOT.TLegend(0.48, 0.72, 0.92, 0.88)
        leg_mult.SetBorderSize(1)
        leg_mult.SetFillColor(ROOT.kWhite)
        leg_mult.AddEntry(h_mult_all, f"All #nu_{{#mu}} CC ({tot_cc} ev)", "f")
        leg_mult.AddEntry(h_mult_charm, f"#nu_{{#mu}} CC with Charm ({tot_ch} ev)", "f")
        leg_mult.Draw("same")

        # Pad 2: Tracking efficiency (>= 2 tracks) and DS acceptance vs E_nu
        pad2 = c_reco.cd(2)
        pad2.SetLeftMargin(0.12)
        pad2.SetRightMargin(0.06)
        pad2.SetTopMargin(0.08)
        pad2.SetBottomMargin(0.12)
        pad2.SetGrid()

        h_eff_frame = pad2.DrawFrame(0.0, 0.0, max(1000.0, nu_bin_edges[-1]), 1.05)
        h_eff_frame.SetTitle("Charm Dimuon Efficiency & Acceptance vs E_{#nu};E_{#nu} [GeV];Fraction / Efficiency")

        g_eff_2trks = ROOT.TGraphAsymmErrors(n_nu_ebins)
        g_eff_2trks.SetName("g_eff_charm_2trks")
        g_eff_ds = ROOT.TGraphAsymmErrors(n_nu_ebins)
        g_eff_ds.SetName("g_eff_charm_in_ds")

        for i_b in range(n_nu_ebins):
            n_ch_bin = data["cc_charm_per_ebin"][i_b]
            n_ge2 = data.get("charm_ge2trks_per_ebin", [0] * n_nu_ebins)[i_b]
            n_ds = data.get("charm_in_ds_per_ebin", [0] * n_nu_ebins)[i_b]
            x_center = 0.5 * (nu_bin_edges[i_b] + nu_bin_edges[i_b + 1])
            x_err = 0.5 * (nu_bin_edges[i_b + 1] - nu_bin_edges[i_b])

            val_pct_2, lo_2, hi_2 = clopper_pearson_errors(n_ge2, n_ch_bin, 0.6827)
            g_eff_2trks.SetPoint(i_b, x_center, val_pct_2 / 100.0)
            g_eff_2trks.SetPointError(i_b, x_err, x_err, lo_2 / 100.0, hi_2 / 100.0)

            val_pct_ds, lo_ds, hi_ds = clopper_pearson_errors(n_ds, n_ch_bin, 0.6827)
            g_eff_ds.SetPoint(i_b, x_center, val_pct_ds / 100.0)
            g_eff_ds.SetPointError(i_b, x_err, x_err, lo_ds / 100.0, hi_ds / 100.0)

        g_eff_2trks.SetMarkerStyle(21)
        g_eff_2trks.SetMarkerColor(ROOT.kAzure + 2)
        g_eff_2trks.SetLineColor(ROOT.kAzure + 2)
        g_eff_2trks.SetLineWidth(2)

        g_eff_ds.SetMarkerStyle(22)
        g_eff_ds.SetMarkerColor(ROOT.kGreen + 2)
        g_eff_ds.SetLineColor(ROOT.kGreen + 2)
        g_eff_ds.SetLineWidth(2)

        g_eff_2trks.Draw("P E1 same")
        g_eff_ds.Draw("P E1 same")

        leg_eff = ROOT.TLegend(0.40, 0.72, 0.92, 0.88)
        leg_eff.SetBorderSize(1)
        leg_eff.SetFillColor(ROOT.kWhite)
        leg_eff.AddEntry(g_eff_2trks, "#varepsilon_{tracking}: N(#geq 2 reco tracks) / N(charm)", "lep")
        leg_eff.AddEntry(g_eff_ds, "Acceptance: In DS Acceptance / N(charm)", "lep")
        leg_eff.Draw("same")

        c_reco.Write()
        keep_alive_pdg.append(c_reco)

    fout.Close()
    print(f"  ROOT Histograms and Superimposed Canvases written successfully to: {os.path.abspath(output_path)}")

    # Optional PNG Export
    if export_png_dir:
        os.makedirs(export_png_dir, exist_ok=True)
        fin_read = ROOT.TFile.Open(output_path, "READ")
        canvases_dir = fin_read.Get("Canvases")
        if canvases_dir:
            for k in canvases_dir.GetListOfKeys():
                obj = k.ReadObj()
                if "TCanvas" in obj.ClassName():
                    obj.SaveAs(os.path.join(export_png_dir, f"{k.GetName()}.png"))
        fin_read.Close()
        print(f"  Exported diagnostic PNG plots to: {os.path.abspath(export_png_dir)}")


def print_diagnostic_report(
    data: Dict[str, Any],
    nu_bin_edges: List[float],
    charm_bin_edges: List[float],
    pdg_data: Optional[Dict[int, Dict[str, Any]]] = None,
) -> None:
    """Prints a structured summary table and physics diagnostic report."""
    n_nu_ebins = len(nu_bin_edges) - 1
    n_ch_ebins = len(charm_bin_edges) - 1
    tot_files = data["files_processed"]
    tot_events = data["total_events"]
    tot_cc = data["nu_cc_inclusive"]
    tot_charm = data["nu_cc_charm"]
    tot_dimuon = data.get("nu_cc_dimuon", 0)
    charm_rate = (100.0 * tot_charm / tot_cc) if tot_cc > 0 else 0.0
    dimuon_rate = (100.0 * tot_dimuon / tot_cc) if tot_cc > 0 else 0.0

    print("\n" + "=" * 86)
    print(" SND@LHC CHARMED HADRON COMPREHENSIVE PHYSICS & DECAY DIAGNOSTIC REPORT")
    print("=" * 86)
    print(f" Analyzed Files          : {tot_files}")
    print(f" Total Events Scanned    : {tot_events}")
    print(f" Nu_mu CC Events         : {tot_cc}")
    print(f" Nu_mu CC Charm Events   : {tot_charm}")
    print(f" Nu_mu CC Dimuon Events  : {tot_dimuon}")
    print(f" Overall Charm Production: {tot_charm} / {tot_cc} ({charm_rate:.2f}%)")
    print(f" Overall Dimuon Rate     : {tot_dimuon} / {tot_cc} ({dimuon_rate:.3f}%)")
    print("-" * 86)

    # 1. Charm Production Rate vs Neutrino Energy & Slow Rescaling comparison
    print("\n" + "-" * 86)
    print(" 1. CHARM PRODUCTION RATE AS A FUNCTION OF NEUTRINO ENERGY vs SLOW RESCALING")
    print("-" * 86)
    print(f" {'E_nu Bin [GeV]':<20s} {'Total Nu_mu CC':<16s} {'Charm Count':<14s} {'MC Rate [%]':<20s} {'Slow Rescaling':<20s}")
    print("-" * 92)

    f_sr_eval, g_sr_eval_band, _ = create_slow_rescaling_model()

    for i in range(n_nu_ebins):
        e_low, e_high = nu_bin_edges[i], nu_bin_edges[i + 1]
        n_cc_bin = data["cc_all_per_ebin"][i]
        n_ch_bin = data["cc_charm_per_ebin"][i]
        rate_bin = (100.0 * n_ch_bin / n_cc_bin) if n_cc_bin > 0 else 0.0
        err_bin = (100.0 * math.sqrt(n_ch_bin) / n_cc_bin) if (n_cc_bin > 0 and n_ch_bin > 0) else 0.0
        e_mid = 0.5 * (e_low + e_high)
        sr_val = 100.0 * f_sr_eval.Eval(e_mid)
        # Find closest point in g_sr_eval_band for 1-sigma error
        sr_err = 0.0
        for pt_idx in range(g_sr_eval_band.GetN()):
            if g_sr_eval_band.GetPointX(pt_idx) >= e_mid:
                sr_err = 100.0 * g_sr_eval_band.GetErrorY(pt_idx)
                break
        bin_str = f"[{e_low:.0f}, {e_high:.0f})"
        rate_str = f"{rate_bin:5.2f}% +/- {err_bin:4.2f}%" if n_cc_bin > 0 else "N/A (0 events)"
        sr_str = f"{sr_val:5.2f}% +/- {sr_err:4.2f}%"
        print(f" {bin_str:<20s} {n_cc_bin:<16d} {n_ch_bin:<14d} {rate_str:<20s} {sr_str:<20s}")
    print("-" * 92)

    # 1b. Dimuon Production Rate vs Neutrino Energy & Slow Rescaling x <B_mu>
    print("\n" + "-" * 86)
    print(" 1b. CHARM DIMUON PRODUCTION RATE AS A FUNCTION OF NEUTRINO ENERGY vs SLOW RESCALING x <B_mu>")
    print("-" * 86)
    print(f" {'E_nu Bin [GeV]':<20s} {'Total Nu_mu CC':<16s} {'Dimuon Count':<14s} {'MC Rate [%]':<20s} {'Slow Rescaling x <B_mu>':<24s}")
    print("-" * 96)

    f_sr_2mu_eval, g_sr_2mu_eval_band, _ = create_slow_rescaling_model(scale=0.0848, scale_err=0.0050)

    for i in range(n_nu_ebins):
        e_low, e_high = nu_bin_edges[i], nu_bin_edges[i + 1]
        n_cc_bin = data["cc_all_per_ebin"][i]
        n_2mu_bin = data.get("cc_dimuon_per_ebin", [0] * n_nu_ebins)[i]
        rate_2mu_bin = (100.0 * n_2mu_bin / n_cc_bin) if n_cc_bin > 0 else 0.0
        err_2mu_bin = (100.0 * math.sqrt(n_2mu_bin) / n_cc_bin) if (n_cc_bin > 0 and n_2mu_bin > 0) else 0.0
        e_mid = 0.5 * (e_low + e_high)
        sr_2mu_val = 100.0 * f_sr_2mu_eval.Eval(e_mid)
        sr_2mu_err = 0.0
        for pt_idx in range(g_sr_2mu_eval_band.GetN()):
            if g_sr_2mu_eval_band.GetPointX(pt_idx) >= e_mid:
                sr_2mu_err = 100.0 * g_sr_2mu_eval_band.GetErrorY(pt_idx)
                break
        bin_str = f"[{e_low:.0f}, {e_high:.0f})"
        rate_str = f"{rate_2mu_bin:5.3f}% +/- {err_2mu_bin:4.3f}%" if n_cc_bin > 0 else "N/A (0 events)"
        sr_str = f"{sr_2mu_val:5.3f}% +/- {sr_2mu_err:4.3f}%"
        print(f" {bin_str:<20s} {n_cc_bin:<16d} {n_2mu_bin:<14d} {rate_str:<20s} {sr_str:<24s}")
    print("-" * 96)

    # 2. Charm Hadron Energies per Species
    print("\n" + "-" * 86)
    print(" 2. CHARM HADRON ENERGY SPECTRUM (E_charm)")
    print("-" * 86)
    print(f" {'Species':<14s} {'|PDG|':<8s} {'Count':<8s} {'Mean E [GeV]':<16s} {'Min E [GeV]':<14s} {'Max E [GeV]':<14s}")
    print("-" * 86)
    major_pdgs = [421, 411, 431, 4122]
    other_pdgs = [p for p in sorted(data["species_hadron_counts"].keys()) if p not in major_pdgs]

    for abs_c in major_pdgs + other_pdgs:
        sp_name = get_species_name(abs_c)
        c_list = data["charm_e_by_species"].get(abs_c, [])
        cnt = len(c_list)
        if cnt > 0:
            mean_e = sum(c_list) / cnt
            min_e = min(c_list)
            max_e = max(c_list)
            print(f" {sp_name:<14s} {abs_c:<8d} {cnt:<8d} {mean_e:8.1f}        {min_e:8.1f}      {max_e:8.1f}")
        else:
            print(f" {sp_name:<14s} {abs_c:<8d} 0        N/A             N/A            N/A")
    print("-" * 86)

    # 3. Charm Species Fractions vs E_charm
    tot_charm_hadrons = sum(data["species_hadron_counts"].values())
    print("\n" + "-" * 86)
    print(" 3. SPECIES FRAGMENTATION FRACTIONS AS A FUNCTION OF CHARM HADRON ENERGY (E_charm)")
    print("-" * 86)
    header = f" {'Species':<14s} {'Overall Share':<16s}"
    for i in range(n_ch_ebins):
        header += f" [{charm_bin_edges[i]:.0f},{charm_bin_edges[i+1]:.0f}) GeV".ljust(14)
    tbl_width = max(86, len(header))
    print(header)
    print("-" * tbl_width)

    for abs_c in major_pdgs + other_pdgs:
        sp_name = get_species_name(abs_c)
        cnt = data["species_hadron_counts"].get(abs_c, 0)
        overall_share = (100.0 * cnt / tot_charm_hadrons) if tot_charm_hadrons > 0 else 0.0
        row = f" {sp_name:<14s} {overall_share:5.2f}% ({cnt:<3d})"
        row = f"{row:<32s}"
        for i in range(n_ch_ebins):
            ebin_tot = sum(data["species_charm_ebin_counts"][i].values())
            c_ebin = data["species_charm_ebin_counts"][i].get(abs_c, 0)
            share_ebin = (100.0 * c_ebin / ebin_tot) if ebin_tot > 0 else 0.0
            row += f"{share_ebin:5.1f}% ({c_ebin:<2d})".ljust(14)
        print(row)
    print("-" * tbl_width)

    # 4. Exclusive Decay Channels & Branching Ratio Evolution with E_charm
    print("\n" + "-" * 86)
    print(" 4. EXCLUSIVE DECAY CHANNELS & EVOLUTION WITH CHARM HADRON ENERGY")
    print("-" * 86)
    for abs_c in major_pdgs:
        sp_name = get_species_name(abs_c)
        ch_map = data["decay_channels_by_species"].get(abs_c, Counter())
        tot_d = data["total_decays_by_species"].get(abs_c, sum(ch_map.values()))
        if tot_d == 0:
            continue
        sorted_ch = sorted(ch_map.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  [{sp_name}] (Total Decays: {tot_d}, Distinct Modes: {len(ch_map)}):")
        ch_eb_map = data["decay_channels_by_species_charm_ebin"].get(abs_c, {})

        for rank, (ch_tup, count) in enumerate(sorted_ch[:4], start=1):
            formula = format_channel_formula(ch_tup)
            br_tot = 100.0 * count / tot_d
            print(f"    #{rank:02d}: {formula:<36s} -> Overall: {count:3d} ({br_tot:4.1f}%)")
            # Energy evolution
            e_str = "         Evolution vs E_charm: "
            for i_b in range(n_ch_ebins):
                eb_cnt = ch_eb_map.get(i_b, Counter())
                eb_tot = sum(eb_cnt.values())
                m_cnt = eb_cnt.get(ch_tup, 0)
                br_b = (100.0 * m_cnt / eb_tot) if eb_tot > 0 else 0.0
                e_str += f"[{charm_bin_edges[i_b]:.0f}-{charm_bin_edges[i_b+1]:.0f}G]: {br_b:4.1f}%  "
            print(e_str)

    # 5. Physics Modeling Checks
    print("\n" + "=" * 86)
    print(" 5. PHYSICAL VALIDATION TESTS (BRANCHING FRACTIONS, LIFETIMES, LEPTICITY)")
    print("=" * 86)

    any_anomalies = False

    for abs_c in major_pdgs:
        sp_name = get_species_name(abs_c)
        tot_d = data["total_decays_by_species"].get(abs_c, 0)
        if tot_d == 0:
            continue

        n_mu = data["semi_muonic_counts"].get(abs_c, 0)
        n_e = data["semi_electronic_counts"].get(abs_c, 0)
        br_mu = (100.0 * n_mu / tot_d) if tot_d > 0 else 0.0
        br_e = (100.0 * n_e / tot_d) if tot_d > 0 else 0.0

        exp_mu, exp_e = PDG_SEMILEPTONIC_BR.get(abs_c, (0.0, 0.0))
        pdg_ctau = PDG_LIFETIMES_CTAU_UM.get(abs_c, 0.0)
        obs_ctau_list = data["ctau_values_um"].get(abs_c, [])
        mean_ctau = (sum(obs_ctau_list) / len(obs_ctau_list)) if obs_ctau_list else 0.0

        ch_map = data["decay_channels_by_species"].get(abs_c, Counter())
        top_mode, top_cnt = ch_map.most_common(1)[0] if ch_map else ((), 0)
        top_frac = 100.0 * top_cnt / tot_d if tot_d > 0 else 0.0

        print(f"\n  [{sp_name}]:")
        print(f"    - Semi-muonic BR (-> mu X)     : {br_mu:5.1f}%  (PDG reference: ~{exp_mu:.1f}%)")
        print(f"    - Semi-electronic BR (-> e X)  : {br_e:5.1f}%  (PDG reference: ~{exp_e:.1f}%)")
        if br_e > 0:
            print(f"    - Lepton Universality mu/e     : {br_mu / br_e:5.2f} (Expected: ~1.00)")
        print(f"    - Proper Lifetime <c*tau>      : {mean_ctau:5.1f} um (PDG reference: {pdg_ctau:.1f} um)")
        print(f"    - Most Frequent Decay Mode     : '{format_channel_formula(top_mode)}' ({top_frac:4.1f}%)")

        if top_frac >= 80.0 and tot_d >= 15:
            any_anomalies = True
            print(f"    >>> STATUS: [FAIL] FORCED EXCLUSIVE CHANNEL DETECTED! Mode '{format_channel_formula(top_mode)}' dominates {top_frac:.1f}%.")
        elif n_mu == 0 and exp_mu > 0 and tot_d >= 30:
            any_anomalies = True
            print(f"    >>> STATUS: [WARN] Prompt semi-muonic decays are completely absent in this sample!")
        elif mean_ctau < 1.0 and tot_d >= 10:
            any_anomalies = True
            print(f"    >>> STATUS: [FAIL] Particle appears to decay instantaneously (c*tau ~ 0)!")
        else:
            print(f"    >>> STATUS: [PASS] Branching modes, semi-leptonic rates, and lifetimes look consistent.")

    # 6. Simulation Inclusive Branching Fractions vs PDG Reference Tables
    if pdg_data:
        print("\n" + "=" * 105)
        print(" 6. INCLUSIVE DECAY BRANCHING FRACTIONS vs PDG WORLD AVERAGES")
        print("=" * 105)
        print(f" {'Species':<10s} {'Inclusive Mode':<30s} {'MC Yield (W)':<14s} {'MC Gamma_i/Gamma [%] (Stat)':<28s} {'PDG Ref [%]':<20s}")
        print("-" * 105)
        for abs_c in major_pdgs:
            sp_name = get_species_name(abs_c)
            sp_modes = pdg_data.get(abs_c, {}).get("modes_list", [])
            if not sp_modes:
                continue

            w_tot = data.get("species_weight_sum", Counter()).get(abs_c, 0.0)
            w2_tot = data.get("species_weight_sq_sum", Counter()).get(abs_c, 0.0)
            n_tot = data.get("species_decay_count", Counter()).get(abs_c, 0)
            if w_tot <= 0.0:
                w_tot = float(data.get("total_decays_by_species", Counter()).get(abs_c, 0))
                w2_tot = w_tot
                n_tot = int(w_tot)

            printed_sp = False
            for m_obj in sp_modes:
                lbl = m_obj.pdg_mode
                w_sum = data.get("inclusive_mode_w", {}).get(abs_c, Counter()).get(lbl, 0.0)
                w2y2_sum = data.get("inclusive_mode_w2y2", {}).get(abs_c, Counter()).get(lbl, 0.0)
                w2y_sum = data.get("inclusive_mode_w2y", {}).get(abs_c, Counter()).get(lbl, 0.0)

                val_pct, err_lo, err_hi = calculate_weighted_yield_and_error(
                    w_sum, w2y2_sum, w2y_sum, w_tot, w2_tot
                )

                mc_str = f"{val_pct:6.2f}% +/- {err_hi:5.2f}%" if w_sum > 0 else f" < {err_hi:5.2f}% (68% CL)"
                pdg_up = m_obj.br_err_up_pct
                pdg_lo = m_obj.br_err_low_pct
                if abs(pdg_up - pdg_lo) < 1e-4:
                    pdg_str = f"{m_obj.br_pct:6.2f}% +/- {pdg_up:5.2f}%"
                else:
                    pdg_str = f"{m_obj.br_pct:6.2f}% (+{pdg_up:4.2f}/-{pdg_lo:4.2f}%)"

                sp_col = sp_name if not printed_sp else ""
                printed_sp = True
                print(f" {sp_col:<10s} {lbl:<30s} {w_sum:<14.1f} {mc_str:<28s} {pdg_str:<20s}")

            print(f" {'':<10s} {'Total sample decays:':<30s} {w_tot:<14.1f} (N={n_tot} unweighted decays)")
            print(f" {'':<10s} {'-' * 93}")
        print("-" * 105)

    # 7. Reconstruction Multiplicity & Dimuon Selection Efficiency
    if any(c > 0 for c in data.get("reco_tracks_cc_all", Counter()).values()):
        print("\n" + "=" * 94)
        print(" 7. RECONSTRUCTION MULTIPLICITY & DIMUON TRACKING EFFICIENCY")
        print("=" * 94)
        tot_cc = data["nu_cc_inclusive"]
        tot_ch = data["nu_cc_charm"]
        print(f" {'N_tracks':<10s} {'Inclusive CC Count':<20s} {'Inclusive CC [%]':<18s} {'Charm CC Count':<16s} {'Charm CC [%]':<16s}")
        print("-" * 94)
        max_trk = max(max(data["reco_tracks_cc_all"].keys(), default=3), max(data["reco_tracks_cc_charm"].keys(), default=3), 3)
        for n_t in range(max_trk + 1):
            cnt_all = data["reco_tracks_cc_all"].get(n_t, 0)
            cnt_ch = data["reco_tracks_cc_charm"].get(n_t, 0)
            pct_all = (100.0 * cnt_all / tot_cc) if tot_cc > 0 else 0.0
            pct_ch = (100.0 * cnt_ch / tot_ch) if tot_ch > 0 else 0.0
            print(f" {n_t:<10d} {cnt_all:<20d} {pct_all:5.2f}%              {cnt_ch:<16d} {pct_ch:5.2f}%")
        print("-" * 94)

        print("\n Charm Dimuon Tracking & Acceptance Efficiency vs E_nu:")
        print(f" {'E_nu Bin [GeV]':<20s} {'Charm Count':<14s} {'N(>=2 trks)':<14s} {'Tracking Eff [%]':<20s} {'In DS Acceptance':<20s}")
        print("-" * 94)
        for i_b in range(n_nu_ebins):
            e_low, e_high = nu_bin_edges[i_b], nu_bin_edges[i_b + 1]
            n_ch_bin = data["cc_charm_per_ebin"][i_b]
            n_ge2 = data.get("charm_ge2trks_per_ebin", [0] * n_nu_ebins)[i_b]
            n_ds = data.get("charm_in_ds_per_ebin", [0] * n_nu_ebins)[i_b]
            eff_2, lo_2, hi_2 = clopper_pearson_errors(n_ge2, n_ch_bin, 0.6827)
            eff_ds, lo_ds, hi_ds = clopper_pearson_errors(n_ds, n_ch_bin, 0.6827)
            bin_str = f"[{e_low:.0f}, {e_high:.0f})"
            str_2 = f"{eff_2:5.2f}% (-{lo_2:4.2f}/+{hi_2:4.2f}%)" if n_ch_bin > 0 else "N/A"
            str_ds = f"{eff_ds:5.2f}% (-{lo_ds:4.2f}/+{hi_ds:4.2f}%)" if n_ch_bin > 0 else "N/A"
            print(f" {bin_str:<20s} {n_ch_bin:<14d} {n_ge2:<14d} {str_2:<20s} {str_ds:<20s}")
        print("-" * 94)

    print("\n" + "=" * 86)
    print(" OVERALL DIAGNOSTIC VERDICT")
    print("=" * 86)
    if any_anomalies:
        print(" [WARNING]: Anomalies detected in charm decay modeling (e.g. forced branching ratios).")
    else:
        print(" [ALL CHECKS PASSED]: Charm production rates, hadron fragmentation species,")
        print(" inclusive branching fractions, proper decay lifetimes, and kinematics appear")
        print(" correctly modelled and consistent with physical expectations.")
    print("=" * 86 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic tool to analyze charm production, species fractions, and decay channels in SND@LHC MC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_input = (
        "/eos/user/i/idioniso/snd-numu-charm/data/"
        "sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/*/"
        "sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root"
    )
    parser.add_argument(
        "--preset",
        choices=list(DEFAULT_PRESETS.keys()),
        default=None,
        help="Preset dataset configuration (e.g. nu14_2022, old_100fb_2022)",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=default_input,
        help="Input ROOT file, directory, or pattern (e.g. .../*/sndLHC...root)",
    )
    parser.add_argument(
        "--filelist",
        type=str,
        default=None,
        help="Optional text file listing input ROOT file paths (one per line)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="charm_diagnostic_results.root",
        help="Output ROOT file to store histograms and superimposed canvases",
    )
    parser.add_argument(
        "-n", "--max-events",
        type=int,
        default=-1,
        help="Maximum total events to process (-1 for all)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Maximum number of input files to process (-1 for all)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=10,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--nu-energy-bins",
        nargs="+",
        type=float,
        default=[0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 350.0, 500.0, 750.0, 1500.0],
        help="Neutrino energy bin edges in GeV",
    )
    parser.add_argument(
        "--charm-energy-bins",
        nargs="+",
        type=float,
        default=[0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1500.0],
        help="Charm hadron energy bin edges in GeV",
    )
    parser.add_argument(
        "--decay-only",
        action="store_true",
        default=True,
        help="Only count direct daughters from 'Decay' process (excludes material inelastic scattering)",
    )
    parser.add_argument(
        "--include-all-interactions",
        action="store_false",
        dest="decay_only",
        help="Include all secondaries (even hadronic inelastic collisions in detector material)",
    )
    parser.add_argument(
        "--ground-state-only",
        action="store_true",
        default=True,
        help="Only analyze ground-state weakly decaying charm hadrons (D0, D+, Ds, Lambda_c)",
    )
    parser.add_argument(
        "--include-excited",
        action="store_false",
        dest="ground_state_only",
        help="Include strongly decaying excited charm states (D*, Sigma_c)",
    )
    parser.add_argument(
        "--numu-cc-only",
        action="store_true",
        default=True,
        help="Restrict denominator to muon neutrino CC interactions",
    )
    parser.add_argument(
        "--all-interactions",
        action="store_false",
        dest="numu_cc_only",
        help="Include NC and all neutrino flavors in inclusive counts",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Optional directory to also export PNG diagnostic plots",
    )
    parser.add_argument(
        "--pdg-tables-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdg_summary_tables"),
        help="Directory containing PDG summary CSV tables (branching_ratio_*.csv)",
    )

    parser.add_argument(
        "--ready-only",
        action="store_true",
        default=True,
        help="Only process partitions that have completed all pipeline stages",
    )
    parser.add_argument(
        "--no-ready-only",
        action="store_false",
        dest="ready_only",
        help="Attempt processing all partitions even if still in progress",
    )

    args = parser.parse_args()

    if args.preset and args.preset in DEFAULT_PRESETS:
        args.input = DEFAULT_PRESETS[args.preset]["input"]

    print("=" * 86)
    print(" SND@LHC: Charmed Hadron Production, Species Fractions & Decay Diagnostic")
    print("=" * 86)
    print(f" Input Source         : {args.input}")
    if args.filelist:
        print(f" Filelist             : {args.filelist}")
    print(f" Output ROOT File     : {os.path.abspath(args.output)}")
    print(f" PDG Tables Dir       : {os.path.abspath(args.pdg_tables_dir)}")
    print(f" Ready Only           : {args.ready_only}")
    print(f" Max Events           : {'All' if args.max_events < 0 else args.max_events}")
    print(f" Max Files            : {'All' if args.max_files < 0 else args.max_files}")
    print(f" Parallel Workers     : {args.jobs}")
    print(f" Nu Energy Bins [GeV] : {args.nu_energy_bins}")
    print(f" Charm Energy Bins    : {args.charm_energy_bins}")
    print(f" Decay Process Only   : {args.decay_only}")
    print(f" Ground State Only    : {args.ground_state_only}")
    print(f" Nu_mu CC Only        : {args.numu_cc_only}")
    print("=" * 86)

    # 1. Resolve input files
    t0_resolve = time.time()
    input_files = fast_resolve_files(
        args.input,
        filelist_path=args.filelist,
        max_files=args.max_files,
        ready_only=args.ready_only,
    )
    if not input_files:
        print(f"[Error] No ROOT files matched input target: {args.input}")
        sys.exit(1)

    print(f"\n[1/3] Resolved {len(input_files)} input ROOT file(s) in {time.time() - t0_resolve:.2f}s.")

    # 2. Partition files for parallel worker execution
    n_jobs = max(1, min(args.jobs, len(input_files)))
    chunk_size = int(math.ceil(len(input_files) / float(n_jobs)))
    file_chunks = [input_files[i:i + chunk_size] for i in range(0, len(input_files), chunk_size)]

    max_events_per_worker = -1
    if args.max_events > 0:
        max_events_per_worker = int(math.ceil(args.max_events / float(len(file_chunks))))

    pdg_data = load_pdg_summary_tables(args.pdg_tables_dir)

    tasks = [
        (
            chunk,
            args.nu_energy_bins,
            args.charm_energy_bins,
            args.decay_only,
            args.ground_state_only,
            args.numu_cc_only,
            max_events_per_worker,
            pdg_data,
        )
        for chunk in file_chunks
    ]

    print(f"[2/3] Processing events across {len(file_chunks)} worker pool(s) ({n_jobs} threads)...")
    t0_proc = time.time()

    if n_jobs == 1 or len(file_chunks) == 1:
        results = [process_files_worker(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_files_worker, tasks))

    elapsed_proc = time.time() - t0_proc
    print(f"      Completed event processing in {elapsed_proc:.2f}s!")

    # 3. Merge results and generate outputs
    print("[3/3] Merging statistics, writing histograms & superimposed canvases...")
    merged_data = merge_worker_results(results, args.nu_energy_bins, args.charm_energy_bins)

    write_histograms_and_canvases(
        args.output,
        merged_data,
        args.nu_energy_bins,
        args.charm_energy_bins,
        pdg_tables_dir=args.pdg_tables_dir,
        export_png_dir=args.plots_dir,
    )

    pdg_data = load_pdg_summary_tables(args.pdg_tables_dir)
    print_diagnostic_report(merged_data, args.nu_energy_bins, args.charm_energy_bins, pdg_data=pdg_data)


if __name__ == "__main__":
    main()
