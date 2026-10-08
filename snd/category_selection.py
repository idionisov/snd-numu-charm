"""
snd.category_selection
----------------------
Event selection and filtering based on neutrino interaction category and truth observables.
Selects events matching specific physical signatures (e.g. nu_mu CC with prompt charm decay muon)
from categorized neutrino datasets produced by scripts/filter_neutrinos.py.
"""

from __future__ import annotations

import os
import time
from typing import List, Dict, Any, Optional, Callable

import ROOT

from .channels import is_charmed_hadron
from .io_utils import copy_auxiliary_metadata, symlink_input_root_files


def derive_category_output_path(
    input_path: str,
    suffix: str = "_truth_numuCC_charmToMu.root",
    output_dir: Optional[str] = None
) -> str:
    """
    Derive the output ROOT filename for a category selection task.
    Replaces known input endings like '_truth.root', '_categorized.root', or '.root'
    with the target category suffix.
    """
    dirname, filename = os.path.split(input_path)
    target_dir = output_dir if output_dir else dirname

    for ending in ["_truth.root", "_categorized.root", ".root"]:
        if filename.endswith(ending):
            stem = filename[:-len(ending)]
            out_filename = f"{stem}{suffix}"
            return os.path.join(target_dir, out_filename)

    out_filename = f"{filename}{suffix}"
    return os.path.join(target_dir, out_filename)


def build_category_predicate(cat_cfg: Dict[str, Any]) -> Callable[[Any], bool]:
    """
    Construct a fast callable predicate function to test if a TTree entry satisfies
    the requested neutrino interaction category criteria.
    """
    flavor = str(cat_cfg.get("flavor", "any")).lower().strip()
    interaction = str(cat_cfg.get("interaction", "any")).upper().strip()
    require_charm = bool(cat_cfg.get("require_charm", False))
    require_direct_charm_muon = bool(cat_cfg.get("require_direct_charm_muon", False))
    require_charm_parent = bool(cat_cfg.get("require_charm_parent", False))
    require_direct_charm_electron = bool(cat_cfg.get("require_direct_charm_electron", False))
    require_hadronic_charm_decay = bool(cat_cfg.get("require_hadronic_charm_decay", False))
    require_downstream_charm_muon = bool(cat_cfg.get("require_downstream_charm_muon", False))
    require_not_direct_charm_muon = bool(cat_cfg.get("require_not_direct_charm_muon", False))
    require_opposite_sign = bool(cat_cfg.get("require_opposite_sign", False))
    require_ds_acceptance = bool(cat_cfg.get("require_ds_acceptance", False))
    require_fiducial = bool(cat_cfg.get("require_fiducial", False))

    def predicate(truth_entry: Any) -> bool:
        # 1. Neutrino flavor check
        nu_pdg = getattr(truth_entry, "nu_pdg", 0)
        abs_nu_pdg = abs(int(nu_pdg))
        if flavor in ["numu", "muon"] and abs_nu_pdg != 14:
            return False
        elif flavor in ["nue", "electron"] and abs_nu_pdg != 12:
            return False
        elif flavor in ["nutau", "tau"] and abs_nu_pdg != 16:
            return False

        # 2. Interaction current (CC vs NC)
        is_cc = int(getattr(truth_entry, "is_cc", 0))
        is_nc = int(getattr(truth_entry, "is_nc", 0))
        if interaction == "CC" and is_cc != 1:
            return False
        elif interaction == "NC" and is_nc != 1:
            return False

        # 3. Charmed hadron production
        has_charm = int(getattr(truth_entry, "has_charm", 0))
        if require_charm and has_charm != 1:
            return False

        # 4. Direct charm decay to muon
        if require_direct_charm_muon:
            has_direct_mu = int(getattr(truth_entry, "charm_has_direct_muon", 0))
            has_prompt_mu = int(getattr(truth_entry, "has_prompt_charm_muon", 0))
            if has_direct_mu != 1 and has_prompt_mu != 1:
                return False

        # 5. Parent of muon must be the charm hadron
        if require_charm_parent:
            mu2_mother_trk = int(getattr(truth_entry, "mu2_mother_track_id", -1))
            charm_trk = int(getattr(truth_entry, "charm_track_id", -2))
            mu2_mother_pdg = abs(int(getattr(truth_entry, "mu2_mother_pdg", 0)))

            matched_track = (mu2_mother_trk >= 0 and mu2_mother_trk == charm_trk)
            matched_pdg = is_charmed_hadron(mu2_mother_pdg)
            if not (matched_track or matched_pdg):
                return False

        # 6. Direct charm decay to electron
        if require_direct_charm_electron:
            has_direct_e = int(getattr(truth_entry, "charm_has_direct_electron", 0))
            if has_direct_e != 1:
                return False

        # 7. Hadronic charm decay requirement
        if require_hadronic_charm_decay:
            has_direct_mu = int(getattr(truth_entry, "charm_has_direct_muon", 0))
            has_direct_e = int(getattr(truth_entry, "charm_has_direct_electron", 0))
            if has_direct_mu == 1 or has_direct_e == 1:
                return False

        if require_not_direct_charm_muon:
            has_direct_mu = int(getattr(truth_entry, "charm_has_direct_muon", 0))
            if has_direct_mu == 1:
                return False

        # 8. Downstream muon from charm decay products
        if require_downstream_charm_muon:
            has_ds_mu = int(getattr(truth_entry, "has_downstream_charm_muon", 0))
            has_ds_full = int(getattr(truth_entry, "has_charm_hadronic_downstream_muon", 0))
            if has_ds_mu != 1 and has_ds_full != 1:
                return False

        # 9. Optional downstream requirements
        if require_opposite_sign:
            if int(getattr(truth_entry, "is_opposite_sign", 0)) != 1:
                return False

        if require_ds_acceptance:
            if int(getattr(truth_entry, "mu2_in_ds_acceptance", 0)) != 1:
                return False

        if require_fiducial:
            if int(getattr(truth_entry, "is_fiducial", 0)) != 1:
                return False

        return True

    return predicate


def select_categorized_events_single_file(
    input_file: str,
    output_file: str,
    cat_cfg: Dict[str, Any],
    truth_tree_name: str = "truth",
    event_tree_name: str = "cbmsim",
    max_entries: int = -1,
    write_empty_files: bool = True,
    copy_metadata: bool = True,
    create_symlinks: bool = True,
) -> Dict[str, Any]:
    """
    Process a single categorized ROOT file, filter events matching category criteria,
    and write selected events (both cbmsim and truth trees) to output_file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input ROOT file not found: {input_file}")

    f_in = ROOT.TFile.Open(input_file, "READ")
    if not f_in or f_in.IsZombie():
        raise IOError(f"Could not open input ROOT file: {input_file}")

    t_truth = f_in.Get(truth_tree_name)
    if not t_truth:
        f_in.Close()
        raise KeyError(f"Tree '{truth_tree_name}' not found in {input_file}")

    t_cbm = f_in.Get(event_tree_name)

    n_tot = t_truth.GetEntries()
    n_process = n_tot if max_entries < 0 else min(n_tot, max_entries)

    predicate = build_category_predicate(cat_cfg)

    # 1. Identify matching entry indices
    matching_entries: List[int] = []
    for iev in range(n_process):
        t_truth.GetEntry(iev)
        if predicate(t_truth):
            matching_entries.append(iev)

    n_matched = len(matching_entries)

    # 2. Output handling
    if n_matched == 0 and not write_empty_files:
        f_in.Close()
        return {
            "input_file": input_file,
            "output_file": output_file,
            "total": n_tot,
            "processed": n_process,
            "selected": 0,
            "file_written": False,
            "symlinks_created": 0,
            "status": "success",
        }

    out_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(out_dir, exist_ok=True)

    f_out = ROOT.TFile.Open(output_file, "RECREATE")
    if not f_out or f_out.IsZombie():
        f_in.Close()
        raise IOError(f"Could not create output ROOT file: {output_file}")

    out_cbm = t_cbm.CloneTree(0) if t_cbm else None
    out_truth = t_truth.CloneTree(0)

    # 3. Fill selected events
    for iev in matching_entries:
        if t_cbm and out_cbm:
            t_cbm.GetEntry(iev)
            out_cbm.Fill()
        t_truth.GetEntry(iev)
        out_truth.Fill()

    f_out.cd()
    if out_cbm:
        out_cbm.Write()
    out_truth.Write()
    f_out.Close()
    f_in.Close()

    # 4. Copy auxiliary metadata & symlink
    if copy_metadata:
        try:
            copy_auxiliary_metadata(input_file, output_file)
        except Exception as e:
            print(f"[Warning] Failed to copy auxiliary metadata: {e}")

    symlinks_created = 0
    if create_symlinks:
        try:
            in_dir = os.path.dirname(os.path.abspath(input_file))
            exclude = {
                os.path.basename(output_file),
                os.path.basename(input_file),
            }
            links = symlink_input_root_files(input_dir=in_dir, output_dir=out_dir, exclude_filenames=exclude)
            symlinks_created = len(links)
        except Exception as e:
            print(f"[Warning] Failed to create symlinks: {e}")

    return {
        "input_file": input_file,
        "output_file": output_file,
        "total": n_tot,
        "processed": n_process,
        "selected": n_matched,
        "file_written": True,
        "symlinks_created": symlinks_created,
        "status": "success",
    }
