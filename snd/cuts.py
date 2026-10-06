"""
snd.cuts
--------
Physics and detector analysis cuts for SND@LHC datasets.
Includes RDataFrame-compatible functors and cut classes.
"""

from __future__ import annotations
import ROOT
from .data_manager import load_trident_libraries


def get_avg_scifi_fiducial_cut(
    vertical_min: float = 200.0,
    vertical_max: float = 1200.0,
    horizontal_min: float = 300.0,
    horizontal_max: float = 1336.0,
    reversed: bool = False,
) -> ROOT.snd.AvgScifiFiducialCut:
    """
    Instantiate and return the C++ snd::AvgScifiFiducialCut object for RDataFrame.

    Parameters:
        vertical_min: Minimum average vertical SciFi channel number (default: 200.0).
        vertical_max: Maximum average vertical SciFi channel number (default: 1200.0).
        horizontal_min: Minimum average horizontal SciFi channel number (default: 300.0).
        horizontal_max: Maximum average horizontal SciFi channel number (default: 1336.0, i.e. 128*12 - 200).
        reversed: If True, select events outside the fiducial boundary (default: False).

    Returns:
        ROOT.snd.AvgScifiFiducialCut instance.
    """
    load_trident_libraries()
    return ROOT.snd.AvgScifiFiducialCut(
        float(vertical_min),
        float(vertical_max),
        float(horizontal_min),
        float(horizontal_max),
        bool(reversed),
    )


class AvgScifiFiducialCut:
    """
    Python wrapper for snd::AvgScifiFiducialCut.
    Usable directly as a callable in Python or passed to RDataFrame.Filter().
    """

    def __new__(
        cls,
        vertical_min: float = 200.0,
        vertical_max: float = 1200.0,
        horizontal_min: float = 300.0,
        horizontal_max: float = 1336.0,
        reversed: bool = False,
    ):
        return get_avg_scifi_fiducial_cut(
            vertical_min=vertical_min,
            vertical_max=vertical_max,
            horizontal_min=horizontal_min,
            horizontal_max=horizontal_max,
            reversed=reversed,
        )


def load_cutflow_config(config_path: Optional[str] = None) -> dict:
    """
    Load cutflow and diagnostic histogram configuration from a YAML file.
    """
    import os
    if config_path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(repo_root, "config", "prefilter_cutflow_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cutflow configuration file not found: {config_path}")

    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        import json
        with open(config_path, "r") as f:
            cfg = json.load(f)

    return cfg or {}


def get_preselection_metric_columns() -> List[str]:
    """
    Return all observable column names exported from PreselectionMetrics.
    """
    return [
        # SciFi Global & Station Hits
        "scifi_nhits", "scifi_sum_qdc", "scifi_max_qdc", "scifi_mean_qdc",
        "scifi_planes_hit", "scifi_stations_hit", "scifi_max_nhits_plane", "scifi_max_qdc_plane",
        "scifi_sum_hit_density", "scifi_max_hit_density",
        "scifi_sum_qdc_density", "scifi_max_qdc_density",
        "scifi_nhits_st1", "scifi_nhits_st2", "scifi_nhits_st3", "scifi_nhits_st4", "scifi_nhits_st5",
        "scifi_busiest_station_nhits", "scifi_busiest_station_id", "scifi_busiest_station_qdc",
        "scifi_avg_channel_v", "scifi_avg_channel_h",
        "scifi_nhits_st1_h", "scifi_nhits_st1_v", "scifi_nhits_st2_h", "scifi_nhits_st2_v",
        "scifi_nhits_st3_h", "scifi_nhits_st3_v", "scifi_nhits_st4_h", "scifi_nhits_st4_v",
        "scifi_nhits_st5_h", "scifi_nhits_st5_v",
        "scifi_qdc_st1", "scifi_qdc_st2", "scifi_qdc_st3", "scifi_qdc_st4", "scifi_qdc_st5",
        "scifi_qdc_st1_h", "scifi_qdc_st1_v", "scifi_qdc_st2_h", "scifi_qdc_st2_v",
        "scifi_qdc_st3_h", "scifi_qdc_st3_v", "scifi_qdc_st4_h", "scifi_qdc_st4_v",
        "scifi_qdc_st5_h", "scifi_qdc_st5_v",
        "scifi_qdc_ratio_down_up", "scifi_nhits_ratio_down_up", "scifi_qdc_ratio_st5_st1",
        "scifi_qdc_asym_xy", "scifi_nhits_asym_xy",
        # Veto System
        "veto_nhits", "veto_sum_qdc", "veto_planes_hit", "veto_max_qdc",
        # HCAL (US)
        "us_nhits", "us_sum_qdc", "us_planes_hit", "us_max_qdc",
        "us_busiest_station_nhits", "us_busiest_station_qdc",
        "us_nhits_st1", "us_nhits_st2", "us_nhits_st3", "us_nhits_st4", "us_nhits_st5",
        "us_qdc_st1", "us_qdc_st2", "us_qdc_st3", "us_qdc_st4", "us_qdc_st5",
        # DS
        "ds_nhits", "ds_sum_qdc", "ds_max_qdc", "ds_planes_hit", "ds_stations_hit",
        "ds_deepest_station", "ds_deepest_plane", "ds_max_nhits_plane", "ds_max_qdc_plane",
        "ds_busiest_station_nhits", "ds_busiest_station_qdc",
        "ds_nhits_st1_h", "ds_nhits_st1_v", "ds_nhits_st2_h", "ds_nhits_st2_v",
        "ds_nhits_st3_h", "ds_nhits_st3_v", "ds_nhits_st4_v",
        "ds_qdc_st1_h", "ds_qdc_st1_v", "ds_qdc_st2_h", "ds_qdc_st2_v",
        "ds_qdc_st3_h", "ds_qdc_st3_v", "ds_qdc_st4_v",
        "ds_qdc_ratio_back_front", "ds_nhits_ratio_back_front", "ds_qdc_ratio_ds4_ds1",
        # Global
        "total_nhits", "total_sum_qdc",
        "ratio_ds_to_scifi_qdc", "ratio_ds_to_scifi_nhits", "ratio_us_to_scifi_qdc",
        "ratio_mufi_to_scifi_qdc"
    ]


def define_preselection_metrics(
    df: Any,
    processor: Optional[Any] = None,
    scifi_branch: str = "Digi_ScifiHits",
    mufi_branch: str = "Digi_MuFilterHits",
) -> Any:
    """
    Define PreselectionMetrics and unpack all individual observables on an RDataFrame.
    """
    load_trident_libraries()
    if processor is None:
        processor = ROOT.snd.trident.PreselectionProcessor()

    df_metrics = df.Define("metrics", processor, [scifi_branch, mufi_branch])
    for col in get_preselection_metric_columns():
        df_metrics = df_metrics.Define(col, f"metrics.{col}")
    return df_metrics


def save_cutflow_root_file(
    output_filepath: str,
    cutflow_stats: List[Dict[str, Any]],
    stage_histograms: Dict[str, Dict[str, Any]],
) -> None:
    """
    Save cutflow summary counters and stage-by-stage diagnostic histograms into a ROOT file.
    Organizes histograms into subdirectories named after each cut stage.
    """
    import os
    out_dir = os.path.dirname(os.path.abspath(output_filepath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fout = ROOT.TFile.Open(output_filepath, "RECREATE")
    if not fout or fout.IsZombie():
        raise IOError(f"Could not open cutflow output ROOT file: {output_filepath}")

    n_cuts = len(cutflow_stats)
    h_cutflow = ROOT.TH1D("h_cutflow", "Event Cutflow Summary;Cut Stage;Events Passed", n_cuts, 0.5, n_cuts + 0.5)
    h_cum_eff = ROOT.TH1D("h_cutflow_cumulative_efficiency", "Cumulative Selection Efficiency;Cut Stage;Cumulative Efficiency", n_cuts, 0.5, n_cuts + 0.5)
    h_rel_eff = ROOT.TH1D("h_cutflow_relative_efficiency", "Relative Selection Efficiency;Cut Stage;Relative Efficiency", n_cuts, 0.5, n_cuts + 0.5)

    raw_count = float(cutflow_stats[0]["count"]) if cutflow_stats and cutflow_stats[0]["count"] > 0 else 1.0
    prev_count = raw_count

    for i, step in enumerate(cutflow_stats, start=1):
        c_val = float(step["count"])
        label = step.get("name", step["id"])
        h_cutflow.GetXaxis().SetBinLabel(i, label)
        h_cutflow.SetBinContent(i, c_val)

        cum_eff = (c_val / raw_count) if raw_count > 0 else 0.0
        h_cum_eff.GetXaxis().SetBinLabel(i, label)
        h_cum_eff.SetBinContent(i, cum_eff)

        rel_eff = (c_val / prev_count) if prev_count > 0 else 0.0
        h_rel_eff.GetXaxis().SetBinLabel(i, label)
        h_rel_eff.SetBinContent(i, rel_eff)

        prev_count = c_val

    fout.cd()
    h_cutflow.Write()
    h_cum_eff.Write()
    h_rel_eff.Write()

    # Write subdirectories per cut stage
    for step in cutflow_stats:
        c_id = step["id"]
        c_name = step.get("name", c_id)
        stage_dir = fout.mkdir(c_id, f"Cut Stage: {c_name}")
        if stage_dir:
            stage_dir.cd()
            hists = stage_histograms.get(c_id, {})
            for h_name, h_obj in hists.items():
                if h_obj:
                    h_obj.Write(h_name)

    fout.Close()


def merge_stage_histograms(
    base: Dict[str, Dict[str, Any]],
    incoming: Dict[str, Dict[str, Any]],
) -> None:
    """
    Merge incoming stage histograms into base dictionary using TH1::Add.
    """
    for stage_id, hists in incoming.items():
        if stage_id not in base:
            base[stage_id] = {}
        for h_name, h_obj in hists.items():
            if h_obj is None:
                continue
            if h_name not in base[stage_id]:
                cloned = h_obj.Clone()
                cloned.SetDirectory(0)
                base[stage_id][h_name] = cloned
            else:
                base[stage_id][h_name].Add(h_obj)

