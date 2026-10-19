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


def get_ds_hough_track_cut(
    min_tracks: int = 2,
    max_tracks: int = 2,
    track_type: int = 13,
    reversed: bool = False,
) -> ROOT.snd.DSHoughTrackCut:
    """
    Instantiate and return the C++ snd::DSHoughTrackCut object for RDataFrame.
    Requires events to have between min_tracks and max_tracks reconstructed tracks of track_type in Reco_MuonTracks.
    """
    load_trident_libraries()
    return ROOT.snd.DSHoughTrackCut(
        int(min_tracks),
        int(max_tracks),
        int(track_type),
        bool(reversed),
    )


class DSHoughTrackCut:
    """Python wrapper for snd::DSHoughTrackCut."""
    def __new__(
        cls,
        min_tracks: int = 2,
        max_tracks: int = 2,
        track_type: int = 13,
        reversed: bool = False,
    ):
        return get_ds_hough_track_cut(
            min_tracks=min_tracks,
            max_tracks=max_tracks,
            track_type=track_type,
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


SHORT_CUT_LABEL_MAP = {
    "step0_raw": "All Events",
    "raw": "All Events",
    "raw / all events": "All Events",
    "all events (no cuts)": "All Events",
    "all events": "All Events",

    "step1_ip1": "LHC IP1",
    "ip1": "LHC IP1",
    "lhc ip1": "LHC IP1",
    "lhc ip1 collision bunch crossing": "LHC IP1",
    "lhc collisions (ip1)": "LHC IP1",

    "step2_stable_beams": "Stable Beams",
    "stable_beams": "Stable Beams",
    "stable beams": "Stable Beams",
    "lhc stable beams mode": "Stable Beams",
    "lhc stable beams": "Stable Beams",

    "step3_event_deltat": "Event #Delta t",
    "event_deltat": "Event #Delta t",
    "inter-event delta time (>100 clock cycles)": "Event #Delta t",
    "inter-event delta time": "Event #Delta t",
    "event time window (delta t > 0)": "Event #Delta t",
    "event #delta t": "Event #Delta t",

    "step4_scifi_fiducial": "SciFi Fiducial",
    "scifi_fiducial": "SciFi Fiducial",
    "scifi average channel fiducial": "SciFi Fiducial",
    "scifi fiducial volume": "SciFi Fiducial",
    "scifi fiducial": "SciFi Fiducial",

    "step5_ds_fiducial": "DS Fiducial",
    "ds_fiducial": "DS Fiducial",
    "downstream mufilter average bar fiducial": "DS Fiducial",
    "downstream (ds) fiducial volume": "DS Fiducial",
    "ds fiducial volume": "DS Fiducial",

    "step6_veto_hits": "Veto Cut",
    "veto_hits": "Veto Cut",
    "no hits in veto": "Veto Cut",
    "veto station hits (veto <= 2)": "Veto Cut",
    "veto hits <= 2": "Veto Cut",
    "veto cut": "Veto Cut",

    "step7_scifi_station": "SciFi Station #neq 1",
    "scifi_station": "SciFi Station #neq 1",
    "scifi station cut (exclude station 1)": "SciFi Station #neq 1",
    "scifi interaction station != 1": "SciFi Station #neq 1",
    "scifi station != 1": "SciFi Station #neq 1",
    "scifi station #neq 1": "SciFi Station #neq 1",

    "step8_ds_dimuon": "DS Dimuon",
    "ds_dimuon": "DS Dimuon",
    "downstream mufilter dimuon activity cut": "DS Dimuon",
    "downstream dimuon candidates (ds >= 2 tracks)": "DS Dimuon",
    "ds dimuon candidate": "DS Dimuon",

    "step9_ds_activity": "DS+US Activity",
    "ds_activity": "DS+US Activity",
    "upstream (hcal) activity on downstream hits": "DS+US Activity",
    "downstream and upstream activity": "DS+US Activity",
    "ds+us activity": "DS+US Activity",

    "step10_scifi_ds_timing": "SciFi-DS Timing",
    "scifi_ds_timing": "SciFi-DS Timing",
    "scifi to ds hit timing sequence": "SciFi-DS Timing",
    "scifi-downstream timing coincidence": "SciFi-DS Timing",
    "scifi-ds timing": "SciFi-DS Timing",

    "step11_two_ds_tracks": "2 DS Tracks",
    "two_ds_tracks": "2 DS Tracks",
    "ds_hough_tracks": "2 DS Tracks",
    "two reconstructed ds tracks": "2 DS Tracks",
    "two ds tracks": "2 DS Tracks",
    "2 ds tracks": "2 DS Tracks",
}


def shorten_cut_label(label: str) -> str:
    """Return a concise label for cutflow histogram axes and comparison plots."""
    if not label:
        return ""
    clean = label.strip()
    key = clean.lower()
    if key in SHORT_CUT_LABEL_MAP:
        return SHORT_CUT_LABEL_MAP[key]
    for k, v in SHORT_CUT_LABEL_MAP.items():
        if k in key:
            return v
    return clean


def draw_rotated_labels_on_canvas(
    pad_or_canvas: Any,
    hist: Any,
    angle: float = 45.0,
    text_size: float = 0.028,
    y_offset: float = 0.018,
) -> List[Any]:
    """
    Renders rotated TLatex labels at bin centers along the X axis.
    Labels are attached to pad primitives to ensure persistence in saved ROOT canvases.
    """
    pad_or_canvas.Update()
    x1 = pad_or_canvas.GetUxmin()
    x2 = pad_or_canvas.GetUxmax()
    left = pad_or_canvas.GetLeftMargin()
    right = 1.0 - pad_or_canvas.GetRightMargin()
    bottom = pad_or_canvas.GetBottomMargin()
    width_ndc = right - left
    x_range = x2 - x1 if x2 != x1 else 1.0

    latex_objs = []
    nbins = hist.GetNbinsX()
    for b in range(1, nbins + 1):
        lbl = hist.GetXaxis().GetBinLabel(b)
        if not lbl:
            continue
        x_user = hist.GetXaxis().GetBinCenter(b)
        x_ndc = left + ((x_user - x1) / x_range) * width_ndc
        y_ndc = bottom - y_offset

        lat = ROOT.TLatex()
        lat.SetNDC(True)
        lat.SetTextAngle(angle)
        lat.SetTextAlign(33)  # Top-Right alignment
        lat.SetTextSize(text_size)
        lat.SetTextFont(42)
        drawn = lat.DrawLatex(x_ndc, y_ndc, lbl)
        latex_objs.append(drawn)

    pad_or_canvas.Modified()
    pad_or_canvas.Update()
    return latex_objs


def make_normalized_histogram(h_orig: ROOT.TH1D, norm_name: str, norm_title: Optional[str] = None) -> ROOT.TH1D:
    """Return a clone of h_orig normalized such that the first bin content is scaled to 1.0."""
    h_norm = h_orig.Clone(norm_name)
    h_norm.SetDirectory(0)
    if norm_title:
        h_norm.SetTitle(norm_title)
    val1 = h_orig.GetBinContent(1)
    if val1 > 0:
        h_norm.Scale(1.0 / val1)
    return h_norm


def save_cutflow_root_file(
    output_filepath: str,
    cutflow_stats: List[Dict[str, Any]],
    stage_histograms: Dict[str, Dict[str, Any]],
    signal_cutflow_stats: Optional[List[Dict[str, Any]]] = None,
    scale_factor: float = 1.0,
    indirect_cutflow_stats: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Save cutflow summary counters and stage-by-stage diagnostic histograms into a ROOT file.
    Organizes histograms into subdirectories named after each cut stage.
    Supports both unweighted, raw_weight, and mc_weight cutflows, plus normalized copies (first bin = 1.0).
    Uses shortened cut stage labels and renders canvases with 45-degree angled labels.
    """
    import os
    ROOT.gROOT.SetBatch(True)
    out_dir = os.path.dirname(os.path.abspath(output_filepath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fout = ROOT.TFile.Open(output_filepath, "RECREATE")
    if not fout or fout.IsZombie():
        raise IOError(f"Could not open cutflow output ROOT file: {output_filepath}")

    n_cuts = len(cutflow_stats)
    h_cutflow = ROOT.TH1D("h_cutflow", "Event Cutflow Summary;;Events Passed", n_cuts, 0.5, n_cuts + 0.5)
    h_cutflow_unweighted = ROOT.TH1D("h_cutflow_unweighted", "Event Cutflow (Unweighted);;Events", n_cuts, 0.5, n_cuts + 0.5)
    h_cutflow_raw_weight = ROOT.TH1D("h_cutflow_raw_weight", "Event Cutflow (raw_weight);;Weighted Events", n_cuts, 0.5, n_cuts + 0.5)
    h_cutflow_mc_weight = ROOT.TH1D("h_cutflow_mc_weight", "Event Cutflow (mc_weight);;Weighted Events", n_cuts, 0.5, n_cuts + 0.5)
    h_cum_eff = ROOT.TH1D("h_cutflow_cumulative_efficiency", "Cumulative Selection Efficiency;;Cumulative Efficiency", n_cuts, 0.5, n_cuts + 0.5)
    h_rel_eff = ROOT.TH1D("h_cutflow_relative_efficiency", "Relative Selection Efficiency;;Relative Efficiency", n_cuts, 0.5, n_cuts + 0.5)

    has_weights = any("raw_weight" in s or "mc_weight" in s for s in cutflow_stats)
    raw_count = float(cutflow_stats[0]["count"]) if cutflow_stats and cutflow_stats[0]["count"] > 0 else 1.0
    first_mc_weight = float(cutflow_stats[0].get("mc_weight", raw_count)) * scale_factor if cutflow_stats else 1.0
    ref_initial = first_mc_weight if has_weights and first_mc_weight > 0 else raw_count
    prev_val = ref_initial

    for i, step in enumerate(cutflow_stats, start=1):
        cnt_val = float(step["count"])
        raw_w_val = float(step.get("raw_weight", cnt_val)) * scale_factor
        mc_w_val = float(step.get("mc_weight", cnt_val)) * scale_factor
        c_val = mc_w_val if has_weights else cnt_val

        raw_label = step.get("name", step["id"])
        label = shorten_cut_label(raw_label)

        h_cutflow.GetXaxis().SetBinLabel(i, label)
        h_cutflow.SetBinContent(i, c_val)

        h_cutflow_unweighted.GetXaxis().SetBinLabel(i, label)
        h_cutflow_unweighted.SetBinContent(i, cnt_val)

        h_cutflow_raw_weight.GetXaxis().SetBinLabel(i, label)
        h_cutflow_raw_weight.SetBinContent(i, raw_w_val)

        h_cutflow_mc_weight.GetXaxis().SetBinLabel(i, label)
        h_cutflow_mc_weight.SetBinContent(i, mc_w_val)

        cum_eff = (c_val / ref_initial) if ref_initial > 0 else 0.0
        h_cum_eff.GetXaxis().SetBinLabel(i, label)
        h_cum_eff.SetBinContent(i, cum_eff)

        rel_eff = (c_val / prev_val) if prev_val > 0 else 0.0
        h_rel_eff.GetXaxis().SetBinLabel(i, label)
        h_rel_eff.SetBinContent(i, rel_eff)

        prev_val = c_val

    # Normalized histograms (first bin scaled to height 1.0)
    h_cutflow_norm = make_normalized_histogram(h_cutflow, "h_cutflow_norm", "Event Cutflow (Normalized);;Fraction Passed")
    h_cutflow_unweighted_norm = make_normalized_histogram(h_cutflow_unweighted, "h_cutflow_unweighted_norm", "Event Cutflow Unweighted (Normalized);;Fraction Passed")
    h_cutflow_raw_weight_norm = make_normalized_histogram(h_cutflow_raw_weight, "h_cutflow_raw_weight_norm", "Event Cutflow raw_weight (Normalized);;Fraction Passed")
    h_cutflow_mc_weight_norm = make_normalized_histogram(h_cutflow_mc_weight, "h_cutflow_mc_weight_norm", "Event Cutflow mc_weight (Normalized);;Fraction Passed")

    fout.cd()
    h_cutflow.Write()
    h_cutflow_unweighted.Write()
    h_cutflow_raw_weight.Write()
    h_cutflow_mc_weight.Write()
    h_cum_eff.Write()
    h_rel_eff.Write()
    h_cutflow_norm.Write()
    h_cutflow_unweighted_norm.Write()
    h_cutflow_raw_weight_norm.Write()
    h_cutflow_mc_weight_norm.Write()

    # Create primary cutflow TCanvas with 45-degree rotated labels
    c_cutflow = ROOT.TCanvas("c_cutflow", "Event Cutflow Summary", 1100, 750)
    c_cutflow.SetGridx(1)
    c_cutflow.SetGridy(1)
    c_cutflow.SetLogy(1)
    c_cutflow.SetBottomMargin(0.22)
    c_cutflow.SetLeftMargin(0.12)
    c_cutflow.SetRightMargin(0.08)

    h_draw_single = h_cutflow.Clone("h_cutflow_single_draw")
    h_draw_single.SetDirectory(0)
    h_draw_single.SetStats(0)
    h_draw_single.SetTitle("Event Selection Cutflow;;Events Passed")
    h_draw_single.GetXaxis().SetTitle("")
    h_draw_single.GetXaxis().SetLabelSize(0.0)
    h_draw_single.SetLineColor(ROOT.kAzure + 2)
    h_draw_single.SetLineWidth(3)
    h_draw_single.SetMarkerColor(ROOT.kAzure + 2)
    h_draw_single.SetMarkerStyle(20)
    h_draw_single.SetMarkerSize(1.2)
    h_draw_single.SetMinimum(0.5)
    h_draw_single.SetMaximum(max(h_draw_single.GetMaximum() * 5.0, 10.0))

    h_draw_single.Draw("HIST")
    h_draw_single.Draw("E SAME")

    leg_single = ROOT.TLegend(0.55, 0.78, 0.89, 0.88)
    leg_single.SetBorderSize(1)
    leg_single.SetFillStyle(1001)
    leg_single.SetFillColor(ROOT.kWhite)
    leg_single.SetTextSize(0.03)
    leg_single.AddEntry(h_draw_single, f"Cutflow (Initial: {ref_initial:.2f})", "lp")
    leg_single.Draw()

    draw_rotated_labels_on_canvas(c_cutflow, h_draw_single, angle=45.0, text_size=0.028)
    fout.cd()
    c_cutflow.Write()

    # If signal cutflow is present, book signal histograms and create overlay comparison TCanvas
    if signal_cutflow_stats:
        h_sig = ROOT.TH1D("h_cutflow_signal", "Signal Cutflow Summary;;Signal Events Passed", n_cuts, 0.5, n_cuts + 0.5)
        h_sig_unw = ROOT.TH1D("h_cutflow_signal_unweighted", "Signal Cutflow (Unweighted);;Signal Events", n_cuts, 0.5, n_cuts + 0.5)
        h_sig_raw = ROOT.TH1D("h_cutflow_signal_raw_weight", "Signal Cutflow (raw_weight);;Weighted Signal Events", n_cuts, 0.5, n_cuts + 0.5)
        h_sig_mc = ROOT.TH1D("h_cutflow_signal_mc_weight", "Signal Cutflow (mc_weight);;Weighted Signal Events", n_cuts, 0.5, n_cuts + 0.5)
        h_sig_eff = ROOT.TH1D("h_cutflow_signal_efficiency", "Signal Selection Efficiency;;Signal Efficiency", n_cuts, 0.5, n_cuts + 0.5)

        has_sig_w = any("raw_weight" in s or "mc_weight" in s for s in signal_cutflow_stats)
        sig_raw_cnt = float(signal_cutflow_stats[0]["count"]) if signal_cutflow_stats and signal_cutflow_stats[0]["count"] > 0 else 1.0
        sig_init_mc = float(signal_cutflow_stats[0].get("mc_weight", sig_raw_cnt)) * scale_factor if signal_cutflow_stats else 1.0
        sig_ref = sig_init_mc if has_sig_w and sig_init_mc > 0 else sig_raw_cnt

        for i, step in enumerate(signal_cutflow_stats, start=1):
            s_cnt = float(step["count"])
            s_raw_w = float(step.get("raw_weight", s_cnt)) * scale_factor
            s_mc_w = float(step.get("mc_weight", s_cnt)) * scale_factor
            s_val = s_mc_w if has_sig_w else s_cnt

            raw_label = step.get("name", step["id"])
            label = shorten_cut_label(raw_label)

            h_sig.GetXaxis().SetBinLabel(i, label)
            h_sig.SetBinContent(i, s_val)

            h_sig_unw.GetXaxis().SetBinLabel(i, label)
            h_sig_unw.SetBinContent(i, s_cnt)

            h_sig_raw.GetXaxis().SetBinLabel(i, label)
            h_sig_raw.SetBinContent(i, s_raw_w)

            h_sig_mc.GetXaxis().SetBinLabel(i, label)
            h_sig_mc.SetBinContent(i, s_mc_w)

            s_eff = (s_val / sig_ref) if sig_ref > 0 else 0.0
            h_sig_eff.GetXaxis().SetBinLabel(i, label)
            h_sig_eff.SetBinContent(i, s_eff)

        h_sig_norm = make_normalized_histogram(h_sig, "h_cutflow_signal_norm", "Signal Cutflow (Normalized);;Fraction Passed")
        h_sig_unw_norm = make_normalized_histogram(h_sig_unw, "h_cutflow_signal_unweighted_norm")
        h_sig_raw_norm = make_normalized_histogram(h_sig_raw, "h_cutflow_signal_raw_weight_norm")
        h_sig_mc_norm = make_normalized_histogram(h_sig_mc, "h_cutflow_signal_mc_weight_norm")

        fout.cd()
        h_sig.Write()
        h_sig_unw.Write()
        h_sig_raw.Write()
        h_sig_mc.Write()
        h_sig_eff.Write()
        h_sig_norm.Write()
        h_sig_unw_norm.Write()
        h_sig_raw_norm.Write()
        h_sig_mc_norm.Write()

        h_ind_draw = None
        ind_ref = 1.0
        if indirect_cutflow_stats:
            h_ind = ROOT.TH1D("h_cutflow_indirect", "Indirect Signal Cutflow Summary;;Indirect Signal Events Passed", n_cuts, 0.5, n_cuts + 0.5)
            h_ind_unw = ROOT.TH1D("h_cutflow_indirect_unweighted", "Indirect Signal Cutflow (Unweighted);;Indirect Events", n_cuts, 0.5, n_cuts + 0.5)
            h_ind_raw = ROOT.TH1D("h_cutflow_indirect_raw_weight", "Indirect Signal Cutflow (raw_weight);;Weighted Indirect Events", n_cuts, 0.5, n_cuts + 0.5)
            h_ind_mc = ROOT.TH1D("h_cutflow_indirect_mc_weight", "Indirect Signal Cutflow (mc_weight);;Weighted Indirect Events", n_cuts, 0.5, n_cuts + 0.5)
            h_ind_eff = ROOT.TH1D("h_cutflow_indirect_efficiency", "Indirect Selection Efficiency;;Indirect Efficiency", n_cuts, 0.5, n_cuts + 0.5)

            has_ind_w = any("raw_weight" in s or "mc_weight" in s for s in indirect_cutflow_stats)
            ind_raw_cnt = float(indirect_cutflow_stats[0]["count"]) if indirect_cutflow_stats and indirect_cutflow_stats[0]["count"] > 0 else 1.0
            ind_init_mc = float(indirect_cutflow_stats[0].get("mc_weight", ind_raw_cnt)) * scale_factor if indirect_cutflow_stats else 1.0
            ind_ref = ind_init_mc if has_ind_w and ind_init_mc > 0 else ind_raw_cnt

            for i, step in enumerate(indirect_cutflow_stats, start=1):
                i_cnt = float(step["count"])
                i_raw_w = float(step.get("raw_weight", i_cnt)) * scale_factor
                i_mc_w = float(step.get("mc_weight", i_cnt)) * scale_factor
                i_val = i_mc_w if has_ind_w else i_cnt

                raw_label = step.get("name", step["id"])
                label = shorten_cut_label(raw_label)

                h_ind.GetXaxis().SetBinLabel(i, label)
                h_ind.SetBinContent(i, i_val)

                h_ind_unw.GetXaxis().SetBinLabel(i, label)
                h_ind_unw.SetBinContent(i, i_cnt)

                h_ind_raw.GetXaxis().SetBinLabel(i, label)
                h_ind_raw.SetBinContent(i, i_raw_w)

                h_ind_mc.GetXaxis().SetBinLabel(i, label)
                h_ind_mc.SetBinContent(i, i_mc_w)

                i_eff = (i_val / ind_ref) if ind_ref > 0 else 0.0
                h_ind_eff.GetXaxis().SetBinLabel(i, label)
                h_ind_eff.SetBinContent(i, i_eff)

            h_ind_norm = make_normalized_histogram(h_ind, "h_cutflow_indirect_norm", "Indirect Signal Cutflow (Normalized);;Fraction Passed")
            h_ind_unw_norm = make_normalized_histogram(h_ind_unw, "h_cutflow_indirect_unweighted_norm")
            h_ind_raw_norm = make_normalized_histogram(h_ind_raw, "h_cutflow_indirect_raw_weight_norm")
            h_ind_mc_norm = make_normalized_histogram(h_ind_mc, "h_cutflow_indirect_mc_weight_norm")

            fout.cd()
            h_ind.Write()
            h_ind_unw.Write()
            h_ind_raw.Write()
            h_ind_mc.Write()
            h_ind_eff.Write()
            h_ind_norm.Write()
            h_ind_unw_norm.Write()
            h_ind_raw_norm.Write()
            h_ind_mc_norm.Write()

            h_ind_draw = h_ind.Clone("h_cutflow_ind_draw")
            h_ind_draw.SetDirectory(0)
            h_ind_draw.SetStats(0)
            h_ind_draw.GetXaxis().SetTitle("")
            h_ind_draw.GetXaxis().SetLabelSize(0.0)
            h_ind_draw.SetLineColor(ROOT.kGreen + 2)
            h_ind_draw.SetLineWidth(3)
            h_ind_draw.SetMarkerColor(ROOT.kGreen + 2)
            h_ind_draw.SetMarkerStyle(22)
            h_ind_draw.SetMarkerSize(1.2)

        # Build combined TCanvas showing all on a single canvas
        c_comp = ROOT.TCanvas("c_cutflow_comparison", "Event Cutflow Comparison (All Events vs Signal)", 1100, 750)
        c_comp.SetGridx(1)
        c_comp.SetGridy(1)
        c_comp.SetLogy(1)
        c_comp.SetBottomMargin(0.22)
        c_comp.SetLeftMargin(0.12)
        c_comp.SetRightMargin(0.08)

        h_all_draw = h_cutflow.Clone("h_cutflow_all_draw")
        h_all_draw.SetDirectory(0)
        h_all_draw.SetStats(0)
        h_all_draw.SetTitle("Event Selection Cutflow: All Events vs Signal;;Events")
        h_all_draw.GetXaxis().SetTitle("")
        h_all_draw.GetXaxis().SetLabelSize(0.0)
        h_all_draw.SetLineColor(ROOT.kAzure + 2)
        h_all_draw.SetLineWidth(3)
        h_all_draw.SetMarkerColor(ROOT.kAzure + 2)
        h_all_draw.SetMarkerStyle(20)
        h_all_draw.SetMarkerSize(1.2)

        h_sig_draw = h_sig.Clone("h_cutflow_sig_draw")
        h_sig_draw.SetDirectory(0)
        h_sig_draw.SetStats(0)
        h_sig_draw.GetXaxis().SetTitle("")
        h_sig_draw.GetXaxis().SetLabelSize(0.0)
        h_sig_draw.SetLineColor(ROOT.kRed + 1)
        h_sig_draw.SetLineWidth(3)
        h_sig_draw.SetMarkerColor(ROOT.kRed + 1)
        h_sig_draw.SetMarkerStyle(21)
        h_sig_draw.SetMarkerSize(1.2)

        max_candidates = [h_all_draw.GetMaximum(), h_sig_draw.GetMaximum(), 1.0]
        if h_ind_draw:
            max_candidates.append(h_ind_draw.GetMaximum())
        max_val = max(max_candidates)
        h_all_draw.SetMinimum(0.1)
        h_all_draw.SetMaximum(max_val * 8.0)

        h_all_draw.Draw("HIST")
        h_all_draw.Draw("E SAME")
        h_sig_draw.Draw("HIST SAME")
        h_sig_draw.Draw("E SAME")
        if h_ind_draw:
            h_ind_draw.Draw("HIST SAME")
            h_ind_draw.Draw("E SAME")

        leg = ROOT.TLegend(0.48, 0.73, 0.89, 0.89)
        leg.SetBorderSize(1)
        leg.SetFillStyle(1001)
        leg.SetFillColor(ROOT.kWhite)
        leg.SetTextSize(0.026)
        leg.AddEntry(h_all_draw, f"All Events (Initial: {ref_initial:.2f})", "lp")
        leg.AddEntry(h_sig_draw, f"Direct Signal (Initial: {sig_ref:.2f})", "lp")
        if h_ind_draw:
            leg.AddEntry(h_ind_draw, f"Indirect Fake Signal (Initial: {ind_ref:.2f})", "lp")
        leg.Draw()

        draw_rotated_labels_on_canvas(c_comp, h_all_draw, angle=45.0, text_size=0.028)
        fout.cd()
        c_comp.Write()

        # Save companion images
        try:
            if output_filepath.endswith(".root"):
                base_img = output_filepath[:-5]
                c_comp.Print(f"{base_img}_comparison.png")
                c_comp.Print(f"{base_img}_comparison.pdf")
        except Exception:
            pass
    else:
        # Save companion images for single cutflow
        try:
            if output_filepath.endswith(".root"):
                base_img = output_filepath[:-5]
                c_cutflow.Print(f"{base_img}_cutflow.png")
                c_cutflow.Print(f"{base_img}_cutflow.pdf")
        except Exception:
            pass

    # Write subdirectories per cut stage
    for step in cutflow_stats:
        c_id = step["id"]
        c_name = shorten_cut_label(step.get("name", c_id))
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

