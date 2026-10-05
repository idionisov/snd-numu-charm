"""
snd
---
SND@LHC Analysis Framework & Tools.
"""

from .data_manager import DataManager, load_trident_libraries, load_neutrino_libraries
from .event_display import Snd2DEventDisplay
from .io_utils import (
    get_repo_root,
    load_config,
    resolve_input_files,
    extract_captures,
    determine_output_path,
    symlink_input_root_files,
)
from .filter import (
    build_processor,
    resolve_hierarchical_selection,
    setup_truth_branches,
    fill_truth_buffers,
    process_single_file,
)

__all__ = [
    "DataManager",
    "load_trident_libraries",
    "load_neutrino_libraries",
    "Snd2DEventDisplay",
    "get_repo_root",
    "load_config",
    "resolve_input_files",
    "extract_captures",
    "determine_output_path",
    "symlink_input_root_files",
    "build_processor",
    "resolve_hierarchical_selection",
    "setup_truth_branches",
    "fill_truth_buffers",
    "process_single_file",
]
