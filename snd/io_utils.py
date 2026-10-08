"""
snd.io_utils
------------
I/O utilities for file pattern expansion, capture extraction,
output path formatting, configuration loading, and symlink management.
"""

from __future__ import annotations

import os
import re
import glob
from typing import Optional, List, Set, Dict, Any


def get_repo_root() -> str:
    """Return the absolute path to the repository root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: Optional[str] = None) -> dict:
    """Load analysis configuration from YAML (or JSON fallback)."""
    if config_path is None:
        config_path = os.path.join(get_repo_root(), "config", "filter_numu_charm_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        import json
        with open(config_path, "r") as f:
            cfg = json.load(f)

    return cfg


def resolve_input_files(pattern: str, max_files: int = -1) -> List[str]:
    """
    Resolve input file pattern into an ordered list of existing files.
    Optimized for multi-directory wildcards (e.g. .../*/filename or .../%s/filename) over network mounts.
    """
    print(f"Resolving input files from pattern:\n  {pattern}")
    matched_files: List[str] = []

    if os.path.isfile(pattern):
        matched_files = [pattern]
    elif os.path.isdir(pattern):
        subdirs = sorted(
            [d for d in os.listdir(pattern) if os.path.isdir(os.path.join(pattern, d))],
            key=lambda x: int(x) if x.isdigit() else x
        )
        if subdirs:
            for d in subdirs:
                if max_files > 0 and len(matched_files) >= max_files:
                    break
                d_path = os.path.join(pattern, d)
                candidates = []
                for f in os.listdir(d_path):
                    if not f.endswith(".root"):
                        continue
                    if f.startswith("geofile") or f.startswith("ship.params") or f.endswith(".gst.root") or f.endswith(".ghep.root"):
                        continue
                    if "signal" in f:
                        candidates.insert(0, os.path.join(d_path, f))
                    else:
                        candidates.append(os.path.join(d_path, f))
                if candidates:
                    matched_files.append(candidates[0])
        else:
            all_valid = []
            signal_files = []
            for f in sorted(os.listdir(pattern)):
                if not f.endswith(".root"):
                    continue
                if f.startswith("geofile") or f.startswith("ship.params") or f.endswith(".gst.root") or f.endswith(".ghep.root"):
                    continue
                fpath = os.path.join(pattern, f)
                all_valid.append(fpath)
                if "signal" in f:
                    signal_files.append(fpath)
            if signal_files:
                matched_files.extend(signal_files)
            else:
                matched_files.extend(all_valid)
    else:
        path_parts = pattern.split("/")
        dir_s_indices = [i for i in range(len(path_parts) - 1) if "%s" in path_parts[i]]
        if dir_s_indices and "*" not in pattern:
            idx = dir_s_indices[0]
            base_dir = "/".join(path_parts[:idx])
            subdir_fmt = path_parts[idx]
            rest = "/".join(path_parts[idx + 1:])
            parts = subdir_fmt.split("%s")
            regex_subdir = re.compile("^" + "(.*?)".join(map(re.escape, parts)) + "$")
            if os.path.isdir(base_dir):
                all_subdirs = sorted(
                    [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
                    key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)]
                )
                for d in all_subdirs:
                    m = regex_subdir.match(d)
                    if not m:
                        continue
                    if max_files > 0 and len(matched_files) >= max_files:
                        break
                    captured_val = m.group(1)
                    fpath = os.path.join(base_dir, d, rest.replace("%s", captured_val))
                    if os.path.exists(fpath):
                        matched_files.append(fpath)
        elif "/*/" in pattern and pattern.count("*") == 1 and "%s" not in pattern:
            base_dir, filename = pattern.split("/*/", 1)
            if os.path.isdir(base_dir):
                subdirs = sorted(
                    [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
                    key=lambda x: int(x) if x.isdigit() else x
                )
                for d in subdirs:
                    if max_files > 0 and len(matched_files) >= max_files:
                        break
                    fpath = os.path.join(base_dir, d, filename)
                    if os.path.exists(fpath):
                        matched_files.append(fpath)

        if not matched_files:
            glob_pattern = pattern.replace("%s", "*")
            matched_files = sorted(
                glob.glob(glob_pattern),
                key=lambda p: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', p)]
            )

    if not matched_files:
        raise FileNotFoundError(
            f"No files matched input pattern: {pattern}\n"
            "Please check the path and your EOS credentials."
        )

    if max_files > 0:
        matched_files = matched_files[:max_files]

    print(f"Found {len(matched_files)} matching ROOT file(s) to process.")
    return matched_files


def extract_captures(input_path: str, input_pattern: str) -> List[str]:
    """Extract captured substrings from input_path matching %s (or *) placeholders in input_pattern."""
    if not input_pattern:
        return []

    # If input_pattern has %s
    if "%s" in input_pattern:
        parts = input_pattern.split("%s")
        regex_str = "^" + "(.*?)".join(map(re.escape, parts)) + "$"
        match = re.match(regex_str, input_path)
        if match:
            return list(match.groups())

    # If input_pattern has *
    if "*" in input_pattern:
        parts = input_pattern.split("*")
        regex_str = "^" + "(.*?)".join(map(re.escape, parts)) + "$"
        match = re.match(regex_str, input_path)
        if match:
            return list(match.groups())

    return []


def determine_output_path(
    input_path: str,
    input_pattern: str,
    output_arg: Optional[str],
    output_cfg: dict,
    index: int,
    total_files: int
) -> str:
    """Determine the output file path corresponding to an input file."""
    captures = extract_captures(input_path, input_pattern)

    dirname, filename = os.path.split(input_path)
    parent_partition = os.path.basename(dirname)
    base_stem, ext = os.path.splitext(filename)

    tag = captures[0] if captures else (parent_partition if parent_partition and parent_partition != "." else str(index + 1))

    # Priority 1: output specified via CLI (-o)
    if output_arg is not None:
        target_template = output_arg
    else:
        # Priority 2: output_pattern specified in YAML config
        # Priority 3: output_dir in YAML config
        target_template = output_cfg.get("output_pattern") or output_cfg.get("output_dir", "output_signal_filtered")

    # If the target template has %s placeholders
    if "%s" in target_template:
        if len(captures) == target_template.count("%s"):
            return target_template % tuple(captures)
        elif len(captures) == 1:
            return target_template.replace("%s", captures[0])
        elif captures:
            return target_template.replace("%s", tag)
        else:
            return target_template.replace("%s", tag)

    # If target_template ends with .root (explicit filename without %s)
    if target_template.endswith(".root"):
        if total_files == 1:
            return target_template
        out_stem, out_ext = os.path.splitext(target_template)
        return f"{out_stem}_{tag}{out_ext}"

    # Target is a directory (from -o or config output_dir)
    base_out_dir = target_template
    preserve_subdirs = output_cfg.get("preserve_subdirs", True)
    file_suffix = output_cfg.get("file_suffix", "_signal")

    # If output_cfg specified an output_pattern template, apply its relative structure under base_out_dir
    cfg_pattern = output_cfg.get("output_pattern")
    if cfg_pattern and "%s" in cfg_pattern:
        rel_pattern = os.path.basename(cfg_pattern)
        parent_rel = os.path.basename(os.path.dirname(cfg_pattern))
        if "%s" in parent_rel:
            rel_path_template = os.path.join(parent_rel, rel_pattern)
        else:
            rel_path_template = rel_pattern
        rel_path = rel_path_template.replace("%s", tag)
        return os.path.join(base_out_dir, rel_path)

    if preserve_subdirs and tag:
        out_file = os.path.join(base_out_dir, tag, f"{base_stem}{file_suffix}_{tag}{ext}")
    else:
        out_file = os.path.join(base_out_dir, f"{tag}_{base_stem}{file_suffix}{ext}")

    return out_file


def symlink_input_root_files(
    input_dir: str,
    output_dir: str,
    exclude_filenames: Optional[Set[str]] = None
) -> List[str]:
    """
    Create symlinks in output_dir to all .root files present in input_dir,
    excluding any file matching exclude_filenames (such as the filtered output file itself).
    """
    if not os.path.isdir(input_dir):
        return []

    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        return []

    os.makedirs(output_dir, exist_ok=True)
    exclude = set(exclude_filenames or [])
    created_links: List[str] = []

    try:
        entries = sorted(os.listdir(input_dir))
    except OSError as err:
        print(f"  [Warning] Could not list input directory {input_dir}: {err}")
        return []

    for entry in entries:
        if not entry.endswith(".root"):
            continue
        if entry in exclude:
            continue

        src_path = os.path.join(input_dir, entry)
        if not os.path.isfile(src_path):
            continue

        dst_path = os.path.join(output_dir, entry)

        # Check if destination already exists or is a symlink
        if os.path.islink(dst_path):
            try:
                # If pointing to the same source, keep it
                if os.path.realpath(dst_path) == os.path.realpath(src_path) or os.readlink(dst_path) == src_path:
                    created_links.append(dst_path)
                    continue
                os.remove(dst_path)
            except OSError:
                pass
        elif os.path.exists(dst_path):
            # Regular file already exists, don't overwrite
            continue

        try:
            os.symlink(src_path, dst_path)
            created_links.append(dst_path)
        except OSError as err:
            print(f"  [Warning] Could not symlink {entry} -> {dst_path}: {err}")

    return created_links


def resolve_tdirectory_hierarchy(cfg: dict, top_name: str = "MCTruth") -> List[str]:
    """
    Determine the TDirectory hierarchy for storing event display canvases.
    If explicitly defined in config under event_displays.tdirectory_path, use it.
    Otherwise, derive it dynamically from selection settings:
      MCTruth -> numu -> CC -> toCharm -> charmToMuon
    """
    disp_cfg = cfg.get("event_displays", {})
    if "tdirectory_path" in disp_cfg and disp_cfg["tdirectory_path"]:
        parts = [p.strip() for p in disp_cfg["tdirectory_path"].replace("->", "/").split("/") if p.strip()]
        if parts:
            if parts[0] in ["neutrinoEvents", "MCTruth"]:
                parts[0] = top_name
            return parts

    sel_cfg = cfg.get("selection", {})
    flavor = str(sel_cfg.get("flavor", "numu")).lower().replace("_", "").replace("-", "")
    interaction = str(sel_cfg.get("interaction", "CC")).upper().strip()
    charm_cfg = sel_cfg.get("charm", {})
    require_charm = bool(charm_cfg.get("require", True))
    charm_species = str(charm_cfg.get("species", "any")).strip()
    charm_decay = str(charm_cfg.get("decay", "to_muon")).lower().replace("-", "").replace("_", "")

    hierarchy = [top_name]

    # Flavor tier
    if flavor in ["numu", "muon"]:
        flavor_dir = "numu"
    elif flavor in ["nue", "electron"]:
        flavor_dir = "nue"
    elif flavor in ["nutau", "tau"]:
        flavor_dir = "nutau"
    else:
        flavor_dir = "allFlavors"
    hierarchy.append(flavor_dir)

    # Interaction tier
    if interaction in ["CC", "NC"]:
        hierarchy.append(interaction)
    else:
        hierarchy.append("inclusive")

    # Charm tier
    if require_charm:
        if charm_species.lower() != "any":
            charm_dir = f"to{charm_species.capitalize()}"
        else:
            charm_dir = "toCharm"
        hierarchy.append(charm_dir)

        # Decay tier
        if charm_decay in ["muon", "tomuon", "dimuon"]:
            decay_dir = "charmToMuon"
            hierarchy.append(decay_dir)
        elif charm_decay != "any":
            decay_dir = f"charm{charm_decay.capitalize()}"
            hierarchy.append(decay_dir)

    return hierarchy


CHARM_SPECIES_NAMES: Dict[int, str] = {
    421: "D0",
    -421: "anti_D0",
    411: "DPlus",
    -411: "DMinus",
    431: "DsPlus",
    -431: "DsMinus",
    4122: "LambdaCPlus",
    -4122: "anti_LambdaCMinus",
    4222: "SigmaCPlusPlus",
    -4222: "anti_SigmaCMinusMinus",
    4212: "SigmaCPlus",
    -4212: "anti_SigmaCMinus",
    4112: "SigmaC0",
    -4112: "anti_SigmaC0",
    4232: "XiCPlus",
    -4232: "anti_XiCMinus",
    4132: "XiC0",
    -4132: "anti_XiC0",
    4332: "OmegaC0",
    -4332: "anti_OmegaC0",
    413: "DStarPlus",
    -413: "DStarMinus",
    423: "DStar0",
    -423: "anti_DStar0",
    433: "DsStarPlus",
    -433: "DsStarMinus",
}


def resolve_mctruth_directory_hierarchy(
    tree: Any,
    processor: Optional[Any] = None,
    min_ds_hor_points: int = 3,
    min_ds_ver_points: int = 3,
    top_name: str = "MCTruth",
    in_acceptance_name: str = "inDSAcceptance",
    not_in_acceptance_name: str = "notInDSAcceptance",
) -> List[str]:
    """
    Dynamically determine the multi-tiered TDirectory hierarchy for an event based on MC truth:
      MCTruth / <flavor> / <current> / <neutrino_process_id_or_label> / <subsequent_process_category>_<N>mu / <acceptance_dir>

    Examples:
      - MCTruth/numu/CC/ch15_mu-_D0_p/D0_directToMu_2mu/inDSAcceptance
      - MCTruth/numu/CC/ch27_mu-_D+_n/DPlus_hadronic_downstreamMu_2mu/inDSAcceptance
      - MCTruth/numu/CC/ch1_mu-_p/noCharm_1mu/inDSAcceptance
      - MCTruth/numu/NC/ch42_pi+_pi-_p/noCharm_0mu/notInDSAcceptance
    """
    from .channels import ChannelLookupManager, pdg_to_name
    from .filter import (
        count_ds_mcpoints,
        is_dimuon_in_ds_acceptance,
        find_primary_muon_track_id,
        find_charm_muon_track_id,
    )

    # 1. Inspect tree for truth information, or evaluate via processor if needed
    nu_pdg = int(getattr(tree, "nu_pdg", 0))
    is_cc = bool(getattr(tree, "is_cc", 0))
    is_nc = bool(getattr(tree, "is_nc", 0))
    channel_id = int(getattr(tree, "channel_id", 0))
    has_charm = bool(getattr(tree, "has_charm", 0))
    charm_pdg = int(getattr(tree, "charm_pdg", 0))
    charm_has_direct_muon = bool(getattr(tree, "charm_has_direct_muon", 0))
    charm_has_direct_electron = bool(getattr(tree, "charm_has_direct_electron", 0))
    has_downstream_charm_muon = bool(
        getattr(tree, "has_downstream_charm_muon", 0) or getattr(tree, "has_charm_hadronic_downstream_muon", 0)
    )
    n_muons_in_event = int(getattr(tree, "n_muons_in_event", 0))
    primary_lepton_track_id = int(getattr(tree, "primary_lepton_track_id", -1))
    mu2_track_id = int(getattr(tree, "mu2_track_id", -1))
    downstream_muon_track_id = int(getattr(tree, "downstream_muon_track_id", -1))

    # Evaluate on the fly if no truth branches on tree
    if nu_pdg == 0 and hasattr(tree, "MCTrack") and processor is not None:
        mufilter_pts = getattr(tree, "MuFilterPoint", None)
        if hasattr(processor, "processMuonNeutrino"):
            info = processor.processMuonNeutrino(tree.MCTrack, mufilter_pts)
        else:
            info = processor.process(tree.MCTrack)
        nu_pdg = int(getattr(info, "nuPdg", 0))
        is_cc = bool(getattr(info, "isCC", False))
        is_nc = bool(getattr(info, "isNC", False))
        has_charm = bool(getattr(info, "hasCharm", False))
        charm_pdg = int(getattr(info, "charmPdg", 0))
        charm_has_direct_muon = bool(getattr(info, "charmHasDirectMuon", False))
        charm_has_direct_electron = bool(getattr(info, "charmHasDirectElectron", False))
        has_downstream_charm_muon = bool(
            getattr(info, "hasDownstreamCharmMuon", False) or getattr(info, "hasCharmHadronicDownstreamMuon", False)
        )
        n_muons_in_event = int(getattr(info, "nMuonsInEvent", 0))
        primary_lepton_track_id = int(getattr(info, "primaryLeptonTrackId", -1))
        mu2_track_id = int(getattr(info, "mu2TrackId", -1))
        downstream_muon_track_id = int(getattr(info, "downstreamMuonTrackId", -1))

        if hasattr(info, "primaryPdgsStr") and info.primaryPdgsStr:
            try:
                prim_pdgs = [int(p) for p in str(info.primaryPdgsStr).split(",") if p]
                ch_mgr = ChannelLookupManager.get_instance()
                channel_id, _, _ = ch_mgr.get_or_register_channel(prim_pdgs, nu_pdg)
            except Exception:
                pass

    # Fallback to discover nu_pdg from MCTrack[0] if still 0
    if nu_pdg == 0 and hasattr(tree, "MCTrack") and len(tree.MCTrack) > 0:
        nu_pdg = tree.MCTrack[0].GetPdgCode()
        for trk in tree.MCTrack:
            if trk.GetMotherId() == 0 and abs(trk.GetPdgCode()) in [11, 13, 15]:
                is_cc = True
                break
        if not is_cc:
            is_nc = True

    # 2. Flavor Tier
    abs_nu = abs(nu_pdg)
    if abs_nu == 14:
        flavor_dir = "numu"
    elif abs_nu == 12:
        flavor_dir = "nue"
    elif abs_nu == 16:
        flavor_dir = "nutau"
    else:
        flavor_dir = f"nu_{abs_nu}" if abs_nu > 0 else "unknown"

    # 3. Current Tier
    if is_cc:
        current_dir = "CC"
    elif is_nc:
        current_dir = "NC"
    else:
        current_dir = "inclusive"

    # 4. Neutrino Interaction Process Label
    ch_mgr = ChannelLookupManager.get_instance()
    if channel_id > 0 and channel_id in ch_mgr.primary_by_id:
        ch_info = ch_mgr.primary_by_id[channel_id]
        formula = ch_info.get("formula", "")
        slug = formula.replace(" + ", "_").replace(" ", "_")
        process_label = f"ch{channel_id}_{slug}"
    elif channel_id > 0:
        process_label = f"ch{channel_id}"
    else:
        process_label = "ch0_unknown"

    # 5. Subsequent Chain Category & Muon Multiplicity
    extra_muon = 0
    if has_charm and charm_pdg != 0:
        species = CHARM_SPECIES_NAMES.get(charm_pdg)
        if not species:
            species = CHARM_SPECIES_NAMES.get(abs(charm_pdg))
        if not species:
            clean_name = pdg_to_name(charm_pdg).replace("+", "Plus").replace("-", "Minus").replace("*", "Star").replace("_", "")
            species = clean_name or f"PDG{charm_pdg}"

        if charm_has_direct_muon:
            mode_label = "directToMu"
            extra_muon = 1
        elif has_downstream_charm_muon:
            mode_label = "hadronic_downstreamMu"
            extra_muon = 1
        elif charm_has_direct_electron:
            mode_label = "directToE"
            extra_muon = 0
        else:
            mode_label = "hadronic"
            extra_muon = 0
        cat_base = f"{species}_{mode_label}"
    else:
        if has_downstream_charm_muon or (n_muons_in_event > (1 if (is_cc and flavor_dir == "numu") else 0)):
            cat_base = "noCharm_downstreamMu"
            extra_muon = 1
        else:
            cat_base = "noCharm"
            extra_muon = 0

    if is_cc and flavor_dir == "numu":
        n_mu = 1 + extra_muon
    elif is_nc:
        n_mu = 0 + extra_muon
    else:
        n_mu = 0 + extra_muon

    subsequent_cat = f"{cat_base}_{n_mu}mu"

    # 6. DS Acceptance Tier
    is_in_ds = False
    if n_mu == 2:
        if hasattr(tree, "dimuon_in_ds_acceptance") and tree.dimuon_in_ds_acceptance != 0:
            is_in_ds = bool(tree.dimuon_in_ds_acceptance)
        else:
            mu1_id = primary_lepton_track_id if primary_lepton_track_id >= 0 else find_primary_muon_track_id(tree)
            mu2_id = mu2_track_id if mu2_track_id >= 0 else (
                downstream_muon_track_id if downstream_muon_track_id >= 0 else find_charm_muon_track_id(tree)
            )
            is_in_ds = is_dimuon_in_ds_acceptance(
                tree,
                min_hor_points=min_ds_hor_points,
                min_ver_points=min_ds_ver_points,
                mu1_id=mu1_id,
                mu2_id=mu2_id,
            )
    elif n_mu == 1:
        if is_cc and flavor_dir == "numu":
            if hasattr(tree, "mu1_in_ds_acceptance") and tree.mu1_in_ds_acceptance != 0:
                is_in_ds = bool(tree.mu1_in_ds_acceptance)
            else:
                mu1_id = primary_lepton_track_id if primary_lepton_track_id >= 0 else find_primary_muon_track_id(tree)
                h, v, _ = count_ds_mcpoints(tree, mu1_id)
                is_in_ds = (h >= min_ds_hor_points and v >= min_ds_ver_points)
        else:
            sec_id = mu2_track_id if mu2_track_id >= 0 else (
                downstream_muon_track_id if downstream_muon_track_id >= 0 else find_charm_muon_track_id(tree)
            )
            h, v, _ = count_ds_mcpoints(tree, sec_id)
            is_in_ds = (h >= min_ds_hor_points and v >= min_ds_ver_points)
    elif n_mu >= 3:
        is_in_ds = is_dimuon_in_ds_acceptance(
            tree, min_hor_points=min_ds_hor_points, min_ver_points=min_ds_ver_points
        )
    else:
        is_in_ds = False

    acc_dir = in_acceptance_name if is_in_ds else not_in_acceptance_name

    return [top_name, flavor_dir, current_dir, process_label, subsequent_cat, acc_dir]


def get_or_create_tdirectory(tfile: Any, path_parts: List[str]) -> Any:
    """Recursively create or navigate to nested TDirectories in a TFile."""
    current = tfile
    for part in path_parts:
        next_dir = current.GetDirectory(part)
        if not next_dir:
            next_dir = current.mkdir(part)
        current = next_dir
    return current


def get_event_header_number(tree: Any, default_idx: int = 0) -> int:
    """Extract event number from tree.EventHeader (GetEventNumber or GetMCEntryNumber)."""
    if hasattr(tree, "EventHeader"):
        h = tree.EventHeader
        if hasattr(h, "GetEventNumber"):
            try:
                return int(h.GetEventNumber())
            except Exception:
                pass
        if hasattr(h, "GetMCEntryNumber"):
            try:
                return int(h.GetMCEntryNumber())
            except Exception:
                pass
    return default_idx


def get_event_header_run_id(tree: Any, default_run: Optional[Any] = None) -> int:
    """Extract run ID from tree.EventHeader.GetRunId() if available, otherwise default_run or 0."""
    if hasattr(tree, "EventHeader"):
        h = tree.EventHeader
        if hasattr(h, "GetRunId"):
            try:
                rid = int(h.GetRunId())
                if rid > 0 or default_run is None:
                    return rid
            except Exception:
                pass
    if default_run is not None:
        try:
            return int(default_run)
        except (ValueError, TypeError):
            pass
    return 0


def copy_auxiliary_metadata(input_path: str, output_path: str) -> None:
    """
    Copy FairRoot metadata keys (BranchList, TimeBasedBranchList, FileHeader, FileHeaderHeader)
    from input ROOT file to output ROOT file if present.
    """
    import ROOT
    keys_to_copy = ["BranchList", "TimeBasedBranchList", "FileHeader", "FileHeaderHeader"]
    try:
        fin = ROOT.TFile.Open(input_path, "READ")
        if not fin or fin.IsZombie():
            return

        found_objects = []
        for key_name in keys_to_copy:
            obj = fin.Get(key_name)
            if obj:
                found_objects.append((key_name, obj.Clone()))
        fin.Close()

        if not found_objects:
            return

        fout = ROOT.TFile.Open(output_path, "UPDATE")
        if fout and not fout.IsZombie():
            fout.cd()
            for key_name, obj in found_objects:
                if not fout.Get(key_name):
                    obj.Write(key_name, ROOT.TObject.kSingleKey)
            fout.Write()
            fout.Close()
    except Exception as err:
        print(f"  [Warning] Could not copy auxiliary metadata: {err}")


def load_geo_paths(csv_path: Optional[str] = None) -> List[Tuple[int, int, str]]:
    """
    Loads run-range to geofile path mappings from geo_paths.csv.
    Checks explicit csv_path, SNDSW_ROOT/analysis/tools/geo_paths.csv, config/geo_paths.csv, or fallback.
    """
    candidates = []
    if csv_path:
        candidates.append(csv_path)

    sndsw_root = os.environ.get("SNDSW_ROOT")
    if sndsw_root:
        candidates.append(os.path.join(sndsw_root, "analysis/tools/geo_paths.csv"))

    candidates.append(os.path.join(get_repo_root(), "config/geo_paths.csv"))

    records: List[Tuple[int, int, str]] = []
    for candidate in candidates:
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or line.startswith("min_run"):
                            continue
                        parts = line.split(",")
                        if len(parts) >= 3:
                            min_run = int(parts[0].strip())
                            max_run = int(parts[1].strip())
                            raw_path = parts[2].strip()
                            records.append((min_run, max_run, raw_path))
                if records:
                    return records
            except Exception as e:
                print(f"[Warning] Error parsing {candidate}: {e}")

    # Built-in fallback
    return [
        (4361, 5422, "root://eospublic.cern.ch//eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root"),
        (5482, 7356, "root://eospublic.cern.ch//eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18_V3_2023.root"),
        (7357, 10422, "root://eospublic.cern.ch//eos/experiment/sndlhc/convertedData/physics/2024/geofile_sndlhc_TI18_V12_2024.root"),
        (10919, 12792, "root://eospublic.cern.ch//eos/experiment/sndlhc/convertedData/physics/2025/geofile_sndlhc_TI18_V8_2025.root"),
        (100238, 100679, "root://eospublic.cern.ch//eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/geofile_sndlhc_H8_2023_3walls.root"),
        (100841, 100953, "root://eospublic.cern.ch//eos/experiment/sndlhc/convertedData/commissioning/testbeam_24/geofile_sndlhc_H4_2024_W_2walls.root"),
        (100954, 100985, "root://eospublic.cern.ch//eos/experiment/sndlhc/convertedData/commissioning/testbeam_24/geofile_sndlhc_H4_2024_Fe_1wall.root"),
    ]


def get_geofile_for_run(
    run_number: Optional[Any] = None,
    input_path: Optional[str] = None,
    default_geofile: Optional[str] = None,
    geo_paths_csv: Optional[str] = None,
) -> str:
    """
    Resolves the appropriate geofile for a given run number or input file path.
    1. Checks partition-local geofile (geofile_full.Genie-TGeant4.root) if input_path provided.
    2. Matches run_number against SNDSW geo_paths.csv mapping.
    3. Checks year hint in input_path (/2022/, /2023/, /2024/, /2025/).
    4. Falls back to default_geofile if specified, or standard 2022 V4 geofile.
    Prefers local /eos/ filesystem paths when available over root:// URLs.
    """
    # 1. Local partition geofile (simulation)
    if input_path:
        local_geo = os.path.join(os.path.dirname(os.path.abspath(input_path)), "geofile_full.Genie-TGeant4.root")
        if os.path.exists(local_geo):
            return local_geo

    geo_table = load_geo_paths(geo_paths_csv)

    # 2. Match by run number
    int_run = None
    if run_number is not None:
        try:
            int_run = int(run_number)
        except (ValueError, TypeError):
            pass

    if int_run is not None and int_run > 0:
        for min_r, max_r, p in geo_table:
            if min_r <= int_run <= max_r:
                if p.startswith("root://eospublic.cern.ch//eos/"):
                    local_p = p.replace("root://eospublic.cern.ch//eos/", "/eos/")
                    if os.path.exists(local_p):
                        return local_p
                return p

    # 3. Match by year in path if run number did not match
    if input_path:
        for year in ["2022", "2023", "2024", "2025"]:
            if f"/{year}/" in input_path:
                for min_r, max_r, p in geo_table:
                    if f"/{year}/" in p:
                        if p.startswith("root://eospublic.cern.ch//eos/"):
                            local_p = p.replace("root://eospublic.cern.ch//eos/", "/eos/")
                            if os.path.exists(local_p):
                                return local_p
                        return p

    # 4. Fallback
    if default_geofile:
        return default_geofile

    fallback = "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root"
    return fallback


def copy_tcanvases_recursive(src_dir, dest_dir) -> int:
    """
    Recursively copies all TCanvases and directory structures from src_dir to dest_dir.
    Returns the total number of TCanvases copied.
    """
    import ROOT
    count = 0
    for key in src_dir.GetListOfKeys():
        cls_name = key.GetClassName()
        if cls_name == "TCanvas":
            obj = key.ReadObj()
            if obj and not (hasattr(obj, "IsZombie") and obj.IsZombie()):
                dest_dir.cd()
                obj.Write(key.GetName(), ROOT.TObject.kOverwrite)
                count += 1
        elif "TDirectory" in cls_name:
            sub_src = key.ReadObj()
            if sub_src:
                sub_name = key.GetName()
                sub_dest = dest_dir.GetDirectory(sub_name)
                if not sub_dest:
                    sub_dest = dest_dir.mkdir(sub_name)
                count += copy_tcanvases_recursive(sub_src, sub_dest)
    return count


