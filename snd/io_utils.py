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
    else:
        sep = None
        if "/%s/" in pattern and pattern.count("%s") == 1 and "*" not in pattern:
            sep = "/%s/"
        elif "/*/" in pattern and pattern.count("*") == 1 and "%s" not in pattern:
            sep = "/*/"

        if sep:
            base_dir, filename = pattern.split(sep, 1)
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
