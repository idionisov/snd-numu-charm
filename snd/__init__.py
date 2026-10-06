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
    resolve_tdirectory_hierarchy,
    get_or_create_tdirectory,
    get_event_header_number,
)
from .filter import (
    build_processor,
    resolve_hierarchical_selection,
    setup_truth_branches,
    fill_truth_buffers,
    process_single_file,
    find_primary_muon_track_id,
    find_charm_muon_track_id,
    count_ds_mcpoints,
    count_mu2_ds_mcpoints,
    is_dimuon_in_ds_acceptance,
)
from .cuts import (
    AvgScifiFiducialCut,
    get_avg_scifi_fiducial_cut,
    load_cutflow_config,
    get_preselection_metric_columns,
    define_preselection_metrics,
    save_cutflow_root_file,
    merge_stage_histograms,
)

__all__ = [
    "DataManager",
    "AvgScifiFiducialCut",
    "get_avg_scifi_fiducial_cut",
    "load_cutflow_config",
    "get_preselection_metric_columns",
    "define_preselection_metrics",
    "save_cutflow_root_file",
    "merge_stage_histograms",
    "load_trident_libraries",
    "load_neutrino_libraries",
    "Snd2DEventDisplay",
    "get_repo_root",
    "load_config",
    "resolve_input_files",
    "extract_captures",
    "determine_output_path",
    "symlink_input_root_files",
    "resolve_tdirectory_hierarchy",
    "get_or_create_tdirectory",
    "get_event_header_number",
    "build_processor",
    "resolve_hierarchical_selection",
    "setup_truth_branches",
    "fill_truth_buffers",
    "process_single_file",
    "find_primary_muon_track_id",
    "find_charm_muon_track_id",
    "count_ds_mcpoints",
    "count_mu2_ds_mcpoints",
    "is_dimuon_in_ds_acceptance",
]
