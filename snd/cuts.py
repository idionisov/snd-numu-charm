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


def get_avg_ds_fiducial_cut(
    vertical_min: float = 70.0,
    vertical_max: float = 105.0,
    horizontal_min: float = 10.0,
    horizontal_max: float = 50.0,
    reversed: bool = False,
) -> ROOT.snd.AvgDSFiducialCut:
    """
    Instantiate and return the C++ snd::AvgDSFiducialCut object for RDataFrame.

    Parameters:
        vertical_min: Minimum average vertical DS bar number (default: 70.0).
        vertical_max: Maximum average vertical DS bar number (default: 105.0).
        horizontal_min: Minimum average horizontal DS bar number (default: 10.0).
        horizontal_max: Maximum average horizontal DS bar number (default: 50.0).
        reversed: If True, select events outside the fiducial boundary (default: False).

    Returns:
        ROOT.snd.AvgDSFiducialCut instance.
    """
    load_trident_libraries()
    return ROOT.snd.AvgDSFiducialCut(
        float(vertical_min),
        float(vertical_max),
        float(horizontal_min),
        float(horizontal_max),
        bool(reversed),
    )


class AvgDSFiducialCut:
    """
    Python wrapper for snd::AvgDSFiducialCut.
    Usable directly as a callable in Python or passed to RDataFrame.Filter().
    """

    def __new__(
        cls,
        vertical_min: float = 70.0,
        vertical_max: float = 105.0,
        horizontal_min: float = 10.0,
        horizontal_max: float = 50.0,
        reversed: bool = False,
    ):
        return get_avg_ds_fiducial_cut(
            vertical_min=vertical_min,
            vertical_max=vertical_max,
            horizontal_min=horizontal_min,
            horizontal_max=horizontal_max,
            reversed=reversed,
        )


def get_veto_cut(
    min_hits: int = 0,
    max_hits: int = 0,
    require_valid: bool = False,
) -> ROOT.snd.VetoCut:
    """
    Instantiate and return the C++ snd::VetoCut object for RDataFrame.

    Parameters:
        min_hits: Minimum number of hits in MuFilter System 1 (default: 0).
        max_hits: Maximum number of hits in MuFilter System 1 (default: 0; <0 for unbounded).
        require_valid: If True, require hit->isValid() (default: False).

    Returns:
        ROOT.snd.VetoCut instance.
    """
    load_trident_libraries()
    return ROOT.snd.VetoCut(
        int(min_hits),
        int(max_hits),
        bool(require_valid),
    )


class VetoCut:
    """
    Python wrapper for snd::VetoCut.
    Usable directly as a callable in Python or passed to RDataFrame.Filter().
    """

    def __new__(
        cls,
        min_hits: int = 0,
        max_hits: int = 0,
        require_valid: bool = False,
    ):
        return get_veto_cut(
            min_hits=min_hits,
            max_hits=max_hits,
            require_valid=require_valid,
        )


def get_scifi_station_cut(
    threshold: float = 0.0,
    excluded_stations: Sequence[int] = (1,),
    reversed: bool = False,
) -> ROOT.snd.SciFiStationCut:
    """
    Instantiate and return the C++ snd::SciFiStationCut object for RDataFrame.

    Parameters:
        threshold: Fractional hit threshold for finding starting station (default: 0.0).
        excluded_stations: List or tuple of SciFi station numbers to exclude (default: (1,)).
        reversed: If True, require event to start in excluded stations (default: False).

    Returns:
        ROOT.snd.SciFiStationCut instance.
    """
    load_trident_libraries()
    vec = ROOT.std.vector('int')()
    for s in excluded_stations:
        vec.push_back(int(s))
    return ROOT.snd.SciFiStationCut(
        float(threshold),
        vec,
        bool(reversed),
    )


class SciFiStationCut:
    """
    Python wrapper for snd::SciFiStationCut.
    Usable directly as a callable in Python or passed to RDataFrame.Filter().
    """

    def __new__(
        cls,
        threshold: float = 0.0,
        excluded_stations: Sequence[int] = (1,),
        reversed: bool = False,
    ):
        return get_scifi_station_cut(
            threshold=threshold,
            excluded_stations=excluded_stations,
            reversed=reversed,
        )


def get_ds_dimuon_activity_cut(
    min_planes_dimuon: int = 3,
    min_hits_dimuon: int = 2,
    min_planes_other: int = 3,
    min_hits_other: int = 1,
    require_valid: bool = False,
    reversed: bool = False,
) -> ROOT.snd.DSDimuonActivityCut:
    """
    Instantiate and return the C++ snd::DSDimuonActivityCut object for RDataFrame.

    Parameters:
        min_planes_dimuon: Number of planes requiring >= min_hits_dimuon hits (default: 3).
        min_hits_dimuon: Minimum hits per plane in dimuon projection (default: 2).
        min_planes_other: Number of planes requiring >= min_hits_other hits in the other projection (default: 3).
        min_hits_other: Minimum hits per plane in the other projection (default: 1).
        require_valid: If True, require hit->isValid() (default: False).
        reversed: If True, select events failing the activity requirement (default: False).

    Returns:
        ROOT.snd.DSDimuonActivityCut instance.
    """
    load_trident_libraries()
    return ROOT.snd.DSDimuonActivityCut(
        int(min_planes_dimuon),
        int(min_hits_dimuon),
        int(min_planes_other),
        int(min_hits_other),
        bool(require_valid),
        bool(reversed),
    )


class DSDimuonActivityCut:
    """
    Python wrapper for snd::DSDimuonActivityCut.
    Usable directly as a callable in Python or passed to RDataFrame.Filter().
    """

    def __new__(
        cls,
        min_planes_dimuon: int = 3,
        min_hits_dimuon: int = 2,
        min_planes_other: int = 3,
        min_hits_other: int = 1,
        require_valid: bool = False,
        reversed: bool = False,
    ):
        return get_ds_dimuon_activity_cut(
            min_planes_dimuon=min_planes_dimuon,
            min_hits_dimuon=min_hits_dimuon,
            min_planes_other=min_planes_other,
            min_hits_other=min_hits_other,
            require_valid=require_valid,
            reversed=reversed,
        )


def get_scifi_ds_time_cut(
    min_delta_t: float = 0.0,
    require_valid: bool = True,
    reversed: bool = False,
) -> ROOT.snd.SciFiDSTimeCut:
    """
    Instantiate and return the C++ snd::SciFiDSTimeCut object for RDataFrame.
    Requires latest DS hit time > earliest SciFi hit time + min_delta_t.
    """
    load_trident_libraries()
    return ROOT.snd.SciFiDSTimeCut(
        float(min_delta_t),
        bool(require_valid),
        bool(reversed),
    )


class SciFiDSTimeCut:
    """Python wrapper for snd::SciFiDSTimeCut."""
    def __new__(
        cls,
        min_delta_t: float = 0.0,
        require_valid: bool = True,
        reversed: bool = False,
    ):
        return get_scifi_ds_time_cut(
            min_delta_t=min_delta_t,
            require_valid=require_valid,
            reversed=reversed,
        )


def get_ds_activity_cut(
    min_us_planes: int = 5,
    require_ds_hits: bool = True,
    require_valid: bool = True,
    reversed: bool = False,
) -> ROOT.snd.DSActivityCut:
    """
    Instantiate and return the C++ snd::DSActivityCut object for RDataFrame.
    Requires that if DS hits are present, all US planes are hit.
    """
    load_trident_libraries()
    return ROOT.snd.DSActivityCut(
        int(min_us_planes),
        bool(require_ds_hits),
        bool(require_valid),
        bool(reversed),
    )


class DSActivityCut:
    """Python wrapper for snd::DSActivityCut."""
    def __new__(
        cls,
        min_us_planes: int = 5,
        require_ds_hits: bool = True,
        require_valid: bool = True,
        reversed: bool = False,
    ):
        return get_ds_activity_cut(
            min_us_planes=min_us_planes,
            require_ds_hits=require_ds_hits,
            require_valid=require_valid,
            reversed=reversed,
        )


def get_event_header_ip1_cut(
    reversed: bool = False,
) -> ROOT.snd.EventHeaderIP1Cut:
    """
    Instantiate and return the C++ snd::EventHeaderIP1Cut object for RDataFrame.
    Requires EventHeader.isIP1() == True.
    """
    load_trident_libraries()
    return ROOT.snd.EventHeaderIP1Cut(bool(reversed))


class EventHeaderIP1Cut:
    """Python wrapper for snd::EventHeaderIP1Cut."""
    def __new__(cls, reversed: bool = False):
        return get_event_header_ip1_cut(reversed=reversed)


def get_stable_beams_cut(
    beam_mode: int = 11,
    reversed: bool = False,
) -> ROOT.snd.StableBeamsCut:
    """
    Instantiate and return the C++ snd::StableBeamsCut object for RDataFrame.
    Requires EventHeader.GetBeamMode() == 11 (StableBeams).
    """
    load_trident_libraries()
    return ROOT.snd.StableBeamsCut(int(beam_mode), bool(reversed))


class StableBeamsCut:
    """Python wrapper for snd::StableBeamsCut."""
    def __new__(cls, beam_mode: int = 11, reversed: bool = False):
        return get_stable_beams_cut(beam_mode=beam_mode, reversed=reversed)


def get_event_deltat_cut(
    min_delta_t: int = 100,
    event_times: Optional[Sequence[int]] = None,
    reversed: bool = False,
) -> ROOT.snd.EventDeltatCut:
    """
    Instantiate and return the C++ snd::EventDeltatCut object for RDataFrame.
    Requires event time difference from previous event > min_delta_t clock cycles.
    """
    load_trident_libraries()
    if event_times is not None:
        vec = ROOT.std.vector('long long')()
        for t in event_times:
            vec.push_back(int(t))
        return ROOT.snd.EventDeltatCut(int(min_delta_t), vec, bool(reversed))
    return ROOT.snd.EventDeltatCut(int(min_delta_t), bool(reversed))


class EventDeltatCut:
    """Python wrapper for snd::EventDeltatCut."""
    def __new__(
        cls,
        min_delta_t: int = 100,
        event_times: Optional[Sequence[int]] = None,
        reversed: bool = False,
    ):
        return get_event_deltat_cut(
            min_delta_t=min_delta_t,
            event_times=event_times,
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
        "scifi_first_station",
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
        "ds_avg_bar_v", "ds_avg_bar_h",
        "ds_nhits_st1_h", "ds_nhits_st1_v", "ds_nhits_st2_h", "ds_nhits_st2_v",
        "ds_nhits_st3_h", "ds_nhits_st3_v", "ds_nhits_st4_v",
        "ds_qdc_st1_h", "ds_qdc_st1_v", "ds_qdc_st2_h", "ds_qdc_st2_v",
        "ds_qdc_st3_h", "ds_qdc_st3_v", "ds_qdc_st4_v",
        "ds_qdc_ratio_back_front", "ds_nhits_ratio_back_front", "ds_qdc_ratio_ds4_ds1",
        "ds_nplanes_ge1_h", "ds_nplanes_ge1_v", "ds_nplanes_ge2_h", "ds_nplanes_ge2_v",
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
    signal_cutflow_stats: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Save cutflow summary counters and stage-by-stage diagnostic histograms into a ROOT file.
    Organizes histograms into subdirectories named after each cut stage.
    If signal_cutflow_stats is provided, stores h_cutflow_signal and an overlaid TCanvas.
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

    # If signal cutflow is present, book signal histograms and create overlay TCanvas
    if signal_cutflow_stats:
        h_sig = ROOT.TH1D("h_cutflow_signal", "Signal Cutflow Summary;Cut Stage;Signal Events Passed", n_cuts, 0.5, n_cuts + 0.5)
        h_sig_eff = ROOT.TH1D("h_cutflow_signal_efficiency", "Signal Selection Efficiency;Cut Stage;Signal Efficiency", n_cuts, 0.5, n_cuts + 0.5)
        sig_raw = float(signal_cutflow_stats[0]["count"]) if signal_cutflow_stats and signal_cutflow_stats[0]["count"] > 0 else 1.0

        for i, step in enumerate(signal_cutflow_stats, start=1):
            s_val = float(step["count"])
            label = step.get("name", step["id"])
            h_sig.GetXaxis().SetBinLabel(i, label)
            h_sig.SetBinContent(i, s_val)
            s_eff = (s_val / sig_raw) if sig_raw > 0 else 0.0
            h_sig_eff.GetXaxis().SetBinLabel(i, label)
            h_sig_eff.SetBinContent(i, s_eff)

        fout.cd()
        h_sig.Write()
        h_sig_eff.Write()

        # Build combined TCanvas showing both on a single canvas
        c_comp = ROOT.TCanvas("c_cutflow_comparison", "Event Cutflow Comparison (All Events vs Signal)", 1100, 750)
        c_comp.SetGridx(1)
        c_comp.SetGridy(1)
        c_comp.SetLogy(1)
        c_comp.SetBottomMargin(0.20)
        c_comp.SetLeftMargin(0.12)
        c_comp.SetRightMargin(0.08)

        h_all_draw = h_cutflow.Clone("h_cutflow_all_draw")
        h_all_draw.SetDirectory(0)
        h_all_draw.SetTitle("Event Selection Cutflow: All Events vs Signal;Cut Stage;Events")
        h_all_draw.SetLineColor(ROOT.kAzure + 2)
        h_all_draw.SetLineWidth(3)
        h_all_draw.SetMarkerColor(ROOT.kAzure + 2)
        h_all_draw.SetMarkerStyle(20)
        h_all_draw.SetMarkerSize(1.2)
        h_all_draw.GetXaxis().LabelsOption("v")
        h_all_draw.GetXaxis().SetLabelSize(0.035)

        h_sig_draw = h_sig.Clone("h_cutflow_sig_draw")
        h_sig_draw.SetDirectory(0)
        h_sig_draw.SetLineColor(ROOT.kRed + 1)
        h_sig_draw.SetLineWidth(3)
        h_sig_draw.SetMarkerColor(ROOT.kRed + 1)
        h_sig_draw.SetMarkerStyle(21)
        h_sig_draw.SetMarkerSize(1.2)

        max_val = max(h_all_draw.GetMaximum(), h_sig_draw.GetMaximum(), 1.0)
        h_all_draw.SetMinimum(0.1)
        h_all_draw.SetMaximum(max_val * 8.0)

        h_all_draw.Draw("HIST")
        h_all_draw.Draw("E SAME")
        h_sig_draw.Draw("HIST SAME")
        h_sig_draw.Draw("E SAME")

        leg = ROOT.TLegend(0.52, 0.76, 0.89, 0.88)
        leg.SetBorderSize(1)
        leg.SetFillStyle(1001)
        leg.SetFillColor(ROOT.kWhite)
        leg.SetTextSize(0.03)
        leg.AddEntry(h_all_draw, f"All Events (Initial: {int(raw_count)})", "lp")
        leg.AddEntry(h_sig_draw, f"Signal in DS Acceptance (Initial: {int(sig_raw)})", "lp")
        leg.Draw()

        c_comp.Write()

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

