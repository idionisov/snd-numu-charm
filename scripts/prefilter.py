#!/usr/bin/env python3
"""
scripts/prefilter.py
--------------------
Universal event prefiltering pipeline for SND@LHC data and Monte Carlo simulations.
Loads datasets via snd.DataManager, applies RDataFrame-based selection cuts
(including AvgScifiFiducialCut), snapshots filtered events to output ROOT files,
and creates symlinks to auxiliary files from the input directory.

Supports an optional --cutflow argument to generate comprehensive cutflow counters
and diagnostic histograms (SciFi hit activity, busiest stations, hit density weights,
HCAL/US QDC, DS QDC, Veto QDC, ratios, and projections) after each incremental cut,
configured through an external YAML configuration file.

Usage examples:
  # Filter events using default fiducial cut:
  python3 scripts/prefilter.py \\
    -i "/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_numu_charm_signal_%s.root" \\
    -o "prefiltered/%s/preselected_signal_run%s.root"

  # Produce complete cutflow and diagnostic histograms:
  python3 scripts/prefilter.py \\
    -i "/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_numu_charm_signal_%s.root" \\
    -o "prefiltered/%s/preselected_signal_run%s.root" \\
    --cutflow "cutflow_histograms.root" \\
    -j 4
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import re
from datetime import timedelta
from typing import List, Optional, Tuple, Dict, Any

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ROOT
from snd import (
    DataManager,
    AvgScifiFiducialCut,
    AvgDSFiducialCut,
    VetoCut,
    SciFiStationCut,
    DSDimuonActivityCut,
    SciFiDSTimeCut,
    DSActivityCut,
    DSHoughTrackCut,
    EventHeaderIP1Cut,
    StableBeamsCut,
    EventDeltatCut,
    load_trident_libraries,
    resolve_input_files,
    extract_captures,
    symlink_input_root_files,
    load_cutflow_config,
    define_preselection_metrics,
    save_cutflow_root_file,
    merge_stage_histograms,
    shorten_cut_label,
)


def format_output_path(
    input_file: str,
    input_pattern: str,
    output_pattern: str,
    index: int = 0,
) -> str:
    """
    Map an input file path to its corresponding output file path,
    transferring the capture matched by '%s' in the input pattern to all '%s'
    placeholders in the output pattern.
    """
    captures = extract_captures(input_file, input_pattern)
    if captures:
        out_placeholders = output_pattern.count("%s")
        if out_placeholders > 0:
            fill_values = tuple(captures[0] for _ in range(out_placeholders))
            return output_pattern % fill_values
        return output_pattern

    if "%s" in output_pattern:
        tag = str(index)
        out_count = output_pattern.count("%s")
        return output_pattern % tuple(tag for _ in range(out_count))

    if os.path.isdir(output_pattern) or output_pattern.endswith("/"):
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        tag = str(index)
        return os.path.join(output_pattern, tag, f"{base_name}_run{tag}_preselection.root")

    return output_pattern


def copy_auxiliary_metadata(input_path_or_file: Any, output_path: str) -> None:
    """
    Copy FairRoot metadata keys (BranchList, TimeBasedBranchList, FileHeader)
    from input file to output file if present.
    """
    keys_to_copy = ["BranchList", "TimeBasedBranchList", "FileHeader", "FileHeaderHeader"]
    try:
        should_close = False
        if isinstance(input_path_or_file, str):
            fin = ROOT.TFile.Open(input_path_or_file, "READ")
            should_close = True
        else:
            fin = input_path_or_file

        if not fin or fin.IsZombie():
            return

        found_objects = []
        for key_name in keys_to_copy:
            obj = fin.Get(key_name)
            if obj:
                found_objects.append((key_name, obj.Clone()))
        if should_close:
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


def print_cutflow_summary(
    cutflow_stats: List[Dict[str, Any]],
    signal_cutflow_stats: Optional[List[Dict[str, Any]]] = None,
    cutflow_filepath: Optional[str] = None,
) -> None:
    """
    Print an aligned summary table of incremental selection cuts and efficiencies.
    If signal_cutflow_stats is provided, display both side-by-side.
    """
    if not cutflow_stats:
        return

    has_weights = any("raw_weight" in s or "mc_weight" in s for s in cutflow_stats)

    if signal_cutflow_stats:
        width = 112 if has_weights else 96
        print("\n" + "=" * width)
        print(" CUTFLOW PROCESSING SUMMARY (ALL EVENTS vs SIGNAL)")
        print("=" * width)
        if has_weights:
            print(f"{'Step':<4} | {'Cut ID / Name':<28} | {'All Evts':>9} | {'All MC Wgt':>12} | {'All Rel%':>8} | {'Signal':>7} | {'Sig MC Wgt':>11} | {'Sig Rel%':>8}")
        else:
            print(f"{'Step':<4} | {'Cut ID / Name':<34} | {'All Evts':>9} | {'All Rel%':>8} | {'Signal':>7} | {'Sig Rel%':>8} | {'Sig Cum%':>8}")
        print("-" * width)
        raw_all = float(cutflow_stats[0]["count"]) if cutflow_stats and cutflow_stats[0]["count"] > 0 else 1.0
        prev_all = float(cutflow_stats[0].get("mc_weight", raw_all)) if has_weights else raw_all
        raw_sig = float(signal_cutflow_stats[0]["count"]) if signal_cutflow_stats and signal_cutflow_stats[0]["count"] > 0 else 1.0
        prev_sig = float(signal_cutflow_stats[0].get("mc_weight", raw_sig)) if has_weights else raw_sig

        for i, step in enumerate(cutflow_stats):
            c_cnt = float(step["count"])
            c_mc = float(step.get("mc_weight", c_cnt))
            eval_all = c_mc if has_weights else c_cnt
            rel_all = (eval_all / prev_all * 100.0) if prev_all > 0 else 0.0
            prev_all = eval_all

            s_step = signal_cutflow_stats[i] if i < len(signal_cutflow_stats) else {}
            s_cnt = float(s_step.get("count", 0.0))
            s_mc = float(s_step.get("mc_weight", s_cnt))
            eval_sig = s_mc if has_weights else s_cnt
            rel_sig = (eval_sig / prev_sig * 100.0) if prev_sig > 0 else 0.0
            prev_sig = eval_sig

            if has_weights:
                name = shorten_cut_label(step.get("name", step["id"]))[:28]
                print(f"{i:<4} | {name:<28} | {int(c_cnt):>9} | {c_mc:>12.2f} | {rel_all:>7.1f}% | {int(s_cnt):>7} | {s_mc:>11.2f} | {rel_sig:>7.1f}%")
            else:
                cum_sig = (s_cnt / raw_sig * 100.0) if raw_sig > 0 else 0.0
                name = shorten_cut_label(step.get("name", step["id"]))[:34]
                print(f"{i:<4} | {name:<34} | {int(c_cnt):>9} | {rel_all:>7.1f}% | {int(s_cnt):>7} | {rel_sig:>7.1f}% | {cum_sig:>7.1f}%")
    else:
        width = 98 if has_weights else 80
        print("\n" + "=" * width)
        print(" CUTFLOW PROCESSING SUMMARY")
        print("=" * width)
        if has_weights:
            print(f"{'Step':<4} | {'Cut ID / Name':<32} | {'Unweighted':>10} | {'Raw Weight':>12} | {'MC Weight':>14} | {'Rel Eff%':>8} | {'Cum Eff%':>8}")
            print("-" * width)
            raw_w_init = float(cutflow_stats[0].get("mc_weight", cutflow_stats[0]["count"])) if cutflow_stats and cutflow_stats[0].get("mc_weight", cutflow_stats[0]["count"]) > 0 else 1.0
            prev_w = raw_w_init
            for i, step in enumerate(cutflow_stats):
                cnt = float(step["count"])
                raw_w = float(step.get("raw_weight", cnt))
                mc_w = float(step.get("mc_weight", cnt))
                rel_pct = (mc_w / prev_w * 100.0) if prev_w > 0 else 0.0
                cum_pct = (mc_w / raw_w_init * 100.0) if raw_w_init > 0 else 0.0
                name = shorten_cut_label(step.get("name", step["id"]))[:32]
                print(f"{i:<4} | {name:<32} | {int(cnt):>10} | {raw_w:>12.3e} | {mc_w:>14.2f} | {rel_pct:>7.1f}% | {cum_pct:>7.1f}%")
                prev_w = mc_w
        else:
            print(f"{'Step':<5} | {'Cut ID / Name':<38} | {'Passed':>10} | {'Rel. Eff':>10} | {'Cum. Eff':>10}")
            print("-" * width)
            raw_count = float(cutflow_stats[0]["count"]) if cutflow_stats and cutflow_stats[0]["count"] > 0 else 1.0
            prev_count = raw_count
            for i, step in enumerate(cutflow_stats):
                c_val = float(step["count"])
                rel_pct = (c_val / prev_count * 100.0) if prev_count > 0 else 0.0
                cum_pct = (c_val / raw_count * 100.0) if raw_count > 0 else 0.0
                name = shorten_cut_label(step.get("name", step["id"]))[:38]
                print(f"{i:<5} | {name:<38} | {int(c_val):>10} | {rel_pct:>9.2f}% | {cum_pct:>9.2f}%")
                prev_count = c_val

    print("=" * width)
    if cutflow_filepath:
        print(f" Cutflow ROOT file: {cutflow_filepath}")
        print("=" * width + "\n")


def prefilter_file(
    input_file: str,
    output_file: Optional[str] = None,
    cut_scifi: Optional[ROOT.snd.AvgScifiFiducialCut] = None,
    cut_ds: Optional[ROOT.snd.AvgDSFiducialCut] = None,
    cut_veto: Optional[ROOT.snd.VetoCut] = None,
    cut_scifi_station: Optional[ROOT.snd.SciFiStationCut] = None,
    cut_ds_dimuon: Optional[ROOT.snd.DSDimuonActivityCut] = None,
    cut_ds_activity: Optional[ROOT.snd.DSActivityCut] = None,
    cut_scifi_ds_timing: Optional[ROOT.snd.SciFiDSTimeCut] = None,
    cut_two_ds_tracks: Optional[ROOT.snd.DSHoughTrackCut] = None,
    cut_ip1: Optional[ROOT.snd.EventHeaderIP1Cut] = None,
    cut_stable_beams: Optional[ROOT.snd.StableBeamsCut] = None,
    cut_event_deltat: Optional[ROOT.snd.EventDeltatCut] = None,
    cutflow_config: Optional[Dict[str, Any]] = None,
    record_cutflow: bool = False,
    tree_name: Optional[str] = None,
    jobs: int = 1,
    max_events: Optional[int] = None,
    create_symlinks: bool = True,
    skip_empty: bool = False,
) -> Dict[str, Any]:
    """
    Process a single ROOT file with RDataFrame, apply sequential cuts,
    book diagnostic histograms for the cutflow, and write filtered events.
    """
    result = {
        "input_file": input_file,
        "output_file": output_file,
        "total_entries": 0,
        "passed_entries": 0,
        "yield_pct": 0.0,
        "status": "OK",
        "error": None,
        "symlinks_created": 0,
        "cutflow_stats": [],
        "stage_histograms": {},
    }

    try:
        # 1. Setup multi-threading
        effective_mt = jobs > 1 and (max_events is None or max_events <= 0)
        if effective_mt:
            ROOT.ROOT.EnableImplicitMT(jobs)
        else:
            ROOT.ROOT.DisableImplicitMT()

        # 2. Load dataset via DataManager
        dm = DataManager(input_file, tree_name=tree_name, load_libraries=False)
        resolved_tree = dm.tree_name

        if not dm.has_branch("Digi_ScifiHits"):
            result["status"] = "ERROR"
            result["error"] = f"Branch 'Digi_ScifiHits' not found in tree '{resolved_tree}'"
            return result

        # Check if mctruth tree exists in input_file and add as friend
        has_mctruth = False
        try:
            _ftest = ROOT.TFile.Open(input_file, "READ")
            if _ftest and not _ftest.IsZombie():
                has_mctruth = bool(_ftest.Get("mctruth"))
                _ftest.Close()
        except Exception:
            pass

        if has_mctruth:
            chain = dm.get_chain()
            chain.AddFriend("mctruth", input_file)

        # Detect data vs MC
        has_header = dm.has_branch("EventHeader")
        has_mc = dm.has_branch("MCTrack")
        is_data = has_header and not has_mc

        # 3. Build RDataFrame
        df = dm.rdf(range_limit=max_events)
        total_entries = int(df.Count().GetValue())
        result["total_entries"] = total_entries

        # Determine weight columns
        has_weights = has_mctruth or dm.has_branch("mc_weight") or dm.has_branch("raw_weight")
        if not df.HasColumn("mc_weight"):
            if df.HasColumn("mctruth.mc_weight"):
                df = df.Define("mc_weight", "mctruth.mc_weight")
            else:
                df = df.Define("mc_weight", "1.0")

        if not df.HasColumn("raw_weight"):
            if df.HasColumn("mctruth.raw_weight"):
                df = df.Define("raw_weight", "mctruth.raw_weight")
            else:
                df = df.Define("raw_weight", "1.0")

        # 4. Configure selection chain and metrics
        active_cuts = []
        if cutflow_config and "cuts" in cutflow_config:
            active_cuts = [c for c in cutflow_config["cuts"] if c.get("enabled", True)]

        # If no cuts specified in config, use default cut chain
        if not active_cuts:
            active_cuts = [
                {"id": "step0_raw", "name": "All Events", "filter": ""},
                {"id": "step1_ip1", "name": "LHC IP1", "filter": "pass_ip1_cut", "data_only": True},
                {"id": "step2_stable_beams", "name": "Stable Beams", "filter": "pass_stable_beams_cut", "data_only": True},
                {"id": "step3_event_deltat", "name": "Event #Delta t", "filter": "pass_event_deltat_cut", "data_only": True},
                {"id": "step4_scifi_ds_timing", "name": "SciFi-DS Timing", "filter": "pass_scifi_ds_timing_cut"},
                {"id": "step5_veto_hits", "name": "No Veto Hits", "filter": "pass_veto_cut"},
                {"id": "step6_ds_dimuon", "name": "DS Dimuon", "filter": "pass_ds_dimuon_cut"},
                {"id": "step7_two_ds_tracks", "name": "Two DS Tracks", "filter": "pass_two_ds_tracks_cut"},
            ]

        # On Monte Carlo, data-only cuts (IP1, Stable Beams, Delta-t) are preserved in the
        # cutflow sequence and evaluate to 'true' (see below). They do not remove any simulation
        # events, ensuring that data and simulation cutflows match 1-to-1 on the x-axis.

        # Define metrics observables if MuFilter is present and cutflow recording is requested
        has_mufi = dm.has_branch("Digi_MuFilterHits")
        if has_mufi and record_cutflow:
            df_processed = define_preselection_metrics(df)
        else:
            df_processed = df

        # Define Data-only cut booleans
        cuts_list = cutflow_config.get("cuts", []) if cutflow_config else []
        if is_data:
            if cut_ip1 is not None:
                df_processed = df_processed.Define("pass_ip1_cut", cut_ip1, ["EventHeader"])
            else:
                df_processed = df_processed.Define("pass_ip1_cut", "true")

            if cut_stable_beams is not None:
                df_processed = df_processed.Define("pass_stable_beams_cut", cut_stable_beams, ["EventHeader"])
            else:
                df_processed = df_processed.Define("pass_stable_beams_cut", "true")

            if cut_event_deltat is None:
                dt_cfg = next((c for c in cuts_list if c.get("filter") == "pass_event_deltat_cut" or "deltat" in c.get("id", "")), {})
                chain = dm.get_chain()
                n_extract = total_entries
                event_times = []
                for iev in range(n_extract):
                    chain.GetEntry(iev)
                    event_times.append(chain.EventHeader.GetEventTime())
                cut_event_deltat = EventDeltatCut(
                    min_delta_t=dt_cfg.get("min_delta_t", 100),
                    event_times=event_times,
                    reversed=dt_cfg.get("reversed", False),
                )
            df_processed = df_processed.Define("pass_event_deltat_cut", cut_event_deltat, ["rdfentry_"])
        else:
            df_processed = df_processed.Define("pass_ip1_cut", "true")
            df_processed = df_processed.Define("pass_stable_beams_cut", "true")
            df_processed = df_processed.Define("pass_event_deltat_cut", "true")

        # Define SciFi fiducial boolean
        if cut_scifi is not None:
            df_processed = df_processed.Define("pass_scifi_fiducial", cut_scifi, ["Digi_ScifiHits"])
        else:
            df_processed = df_processed.Define("pass_scifi_fiducial", "true")

        # Define DS fiducial boolean
        if has_mufi and cut_ds is not None:
            df_processed = df_processed.Define("pass_ds_fiducial", cut_ds, ["Digi_MuFilterHits"])
        else:
            df_processed = df_processed.Define("pass_ds_fiducial", "true")

        # Define Veto cut boolean
        if has_mufi and cut_veto is not None:
            if has_scifi:
                df_processed = df_processed.Define("pass_veto_cut", cut_veto, ["Digi_MuFilterHits", "Digi_ScifiHits"])
            else:
                df_processed = df_processed.Define("pass_veto_cut", cut_veto, ["Digi_MuFilterHits"])
        else:
            df_processed = df_processed.Define("pass_veto_cut", "true")

        # Define SciFi station cut boolean
        if cut_scifi_station is not None:
            df_processed = df_processed.Define("pass_scifi_station_cut", cut_scifi_station, ["Digi_ScifiHits"])
        else:
            df_processed = df_processed.Define("pass_scifi_station_cut", "true")

        # Define DS dimuon activity cut boolean
        if has_mufi and cut_ds_dimuon is not None:
            df_processed = df_processed.Define("pass_ds_dimuon_cut", cut_ds_dimuon, ["Digi_MuFilterHits"])
        else:
            df_processed = df_processed.Define("pass_ds_dimuon_cut", "true")

        # Define DS activity (all US planes hit if DS hit) boolean
        if has_mufi and cut_ds_activity is not None:
            df_processed = df_processed.Define("pass_ds_activity_cut", cut_ds_activity, ["Digi_MuFilterHits"])
        else:
            df_processed = df_processed.Define("pass_ds_activity_cut", "true")

        # Define SciFi to DS timing boolean
        if has_mufi and cut_scifi_ds_timing is not None:
            df_processed = df_processed.Define("pass_scifi_ds_timing_cut", cut_scifi_ds_timing, ["Digi_ScifiHits", "Digi_MuFilterHits"])
        else:
            df_processed = df_processed.Define("pass_scifi_ds_timing_cut", "true")

        # Define Reconstructed DS Hough tracks observable and cut boolean
        has_reco_tracks = dm.has_branch("Reco_MuonTracks")
        if has_reco_tracks:
            df_processed = df_processed.Define("n_ds_hough_tracks", ROOT.snd.DSHoughTrackCounter(), ["Reco_MuonTracks"])
            if cut_two_ds_tracks is not None:
                df_processed = df_processed.Define("pass_two_ds_tracks_cut", cut_two_ds_tracks, ["Reco_MuonTracks"])
            else:
                df_processed = df_processed.Define("pass_two_ds_tracks_cut", "true")
        else:
            df_processed = df_processed.Define("n_ds_hough_tracks", "0")
            df_processed = df_processed.Define("pass_two_ds_tracks_cut", "true")

        # 5. Build incremental cut nodes and book histograms
        hist_configs = cutflow_config.get("histograms", []) if cutflow_config else []
        cut_nodes = []
        current_node = df_processed

        for c_info in active_cuts:
            c_id = c_info["id"]
            c_name = c_info.get("name", c_id)
            c_filter = c_info.get("filter", "").strip()

            if c_filter:
                current_node = current_node.Filter(c_filter, c_name)

            c_count = current_node.Count()
            c_raw = current_node.Sum("raw_weight")
            c_mc = current_node.Sum("mc_weight")
            c_hists = {}

            if record_cutflow and hist_configs:
                for h_cfg in hist_configs:
                    h_name = h_cfg["name"]
                    h_key = f"{c_id}__{h_name}"
                    h_title = h_cfg.get("title", f"{h_name};{h_cfg.get('var', '')};Events")
                    if h_cfg.get("type") == "2D":
                        h_ptr = current_node.Histo2D(
                            (h_key, h_title,
                             int(h_cfg["bins_x"]), float(h_cfg["xmin"]), float(h_cfg["xmax"]),
                             int(h_cfg["bins_y"]), float(h_cfg["ymin"]), float(h_cfg["ymax"])),
                            str(h_cfg["var_x"]), str(h_cfg["var_y"])
                        )
                    else:
                        var_name = str(h_cfg["var"])
                        h_ptr = current_node.Histo1D(
                            (h_key, h_title, int(h_cfg["bins"]), float(h_cfg["xmin"]), float(h_cfg["xmax"])),
                            var_name
                        )
                    c_hists[h_name] = h_ptr

            cut_nodes.append({
                "id": c_id,
                "name": c_name,
                "node": current_node,
                "count": c_count,
                "raw_weight": c_raw,
                "mc_weight": c_mc,
                "hists": c_hists,
            })

        # 6. Evaluate selected entry IDs from final filter stage
        final_stage = cut_nodes[-1]
        passed_entry_list = sorted(list(final_stage["node"].Take['ULong64_t']('rdfentry_').GetValue()))
        passed_entries = len(passed_entry_list)
        result["passed_entries"] = passed_entries

        if total_entries > 0:
            result["yield_pct"] = (passed_entries / total_entries) * 100.0

        # Collect cutflow counters and histograms
        cutflow_stats = []
        stage_hists = {}
        for stage in cut_nodes:
            cnt = int(stage["count"].GetValue())
            s_dict = {
                "id": stage["id"],
                "name": stage["name"],
                "count": cnt,
            }
            if has_weights:
                s_dict["raw_weight"] = float(stage["raw_weight"].GetValue())
                s_dict["mc_weight"] = float(stage["mc_weight"].GetValue())
            cutflow_stats.append(s_dict)

            if record_cutflow:
                stage_hists[stage["id"]] = {}
                for h_name, h_ptr in stage["hists"].items():
                    h_obj = h_ptr.GetValue().Clone()
                    h_obj.SetDirectory(0)
                    stage_hists[stage["id"]][h_name] = h_obj

        result["cutflow_stats"] = cutflow_stats
        result["stage_histograms"] = stage_hists

        # Check if skipping empty
        if passed_entries == 0 and skip_empty:
            result["status"] = "SKIPPED_EMPTY"
            return result

        # 7. Write filtered events preserving complete tree schema
        if output_file:
            out_dir = os.path.dirname(os.path.abspath(output_file))
            os.makedirs(out_dir, exist_ok=True)

            chain = dm.get_chain()
            fout = ROOT.TFile.Open(output_file, "RECREATE")
            out_tree = chain.CloneTree(0)

            for iev in passed_entry_list:
                chain.GetEntry(iev)
                out_tree.Fill()

            # If input file has mctruth tree, clone and fill it alongside cbmsim
            if has_mctruth:
                fin_truth = ROOT.TFile.Open(input_file, "READ")
                tin_truth = fin_truth.Get("mctruth") if fin_truth else None
                if tin_truth:
                    fout.cd()
                    out_truth = tin_truth.CloneTree(0)
                    for iev in passed_entry_list:
                        tin_truth.GetEntry(iev)
                        out_truth.Fill()
                if fin_truth:
                    fin_truth.Close()

            fout.Write()
            fout.Close()

            # 8. Copy FairRoot auxiliary metadata objects if present
            copy_auxiliary_metadata(input_file, output_file)

            # 9. Create symlinks to auxiliary files in the input directory
            if create_symlinks:
                in_dir = os.path.dirname(os.path.abspath(input_file))
                links = symlink_input_root_files(
                    input_dir=in_dir,
                    output_dir=out_dir,
                    exclude_filenames={os.path.basename(output_file), os.path.basename(input_file)},
                )
                result["symlinks_created"] = len(links)

    except Exception as err:
        result["status"] = "ERROR"
        result["error"] = str(err)
    finally:
        ROOT.ROOT.DisableImplicitMT()

    return result


def _filter_partition_stream(
    input_path: Optional[str],
    output_path: Optional[str],
    cut_timing,
    cut_veto,
    cut_dimuon,
    cut_tracks,
    skip_empty: bool = False,
    record_failures: bool = False,
    partition_id: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import ROOT
    res = {
        "status": "OK",
        "total": 0,
        "passed": 0,
        "cut_counts": [0] * 8,
        "raw_weights": [0.0] * 8,
        "mc_weights": [0.0] * 8,
    }
    failed_records: List[Dict[str, Any]] = []

    if not input_path or not os.path.isfile(input_path):
        res["status"] = "MISSING_INPUT"
        return res, failed_records

    f_in = ROOT.TFile.Open(input_path, "READ")
    if not f_in or f_in.IsZombie():
        res["status"] = "ERROR_OPENING"
        return res, failed_records

    t_cbm = f_in.Get("cbmsim") or f_in.Get("rawConv")
    t_truth = f_in.Get("mctruth")
    n_entries = t_cbm.GetEntries() if t_cbm else 0
    res["total"] = n_entries

    passed_entries: List[int] = []

    if t_cbm and n_entries > 0:
        has_truth = bool(t_truth and t_truth.GetEntries() == n_entries)

        t_cbm.SetBranchStatus("*", 0)
        t_cbm.SetBranchStatus("Digi_ScifiHits*", 1)
        t_cbm.SetBranchStatus("Digi_MuFilterHits*", 1)
        trk_branches = [b for b in t_cbm.GetListOfBranches() if b.GetName() == "Reco_MuonTracks"]
        for b in trk_branches[:-1]:
            b.SetStatus(0)
        if trk_branches:
            trk_branches[-1].SetStatus(1)
        if record_failures:
            t_cbm.SetBranchStatus("EventHeader*", 1)

        leaves = [l for l in t_cbm.GetListOfLeaves() if l.GetName() == "Reco_MuonTracks"]
        last_trk_leaf = leaves[-1] if leaves else None

        for iev in range(n_entries):
            rw = 1.0
            mw = 1.0
            if has_truth:
                t_truth.GetEntry(iev)
                rw = float(getattr(t_truth, "raw_weight", 1.0))
                mw = float(getattr(t_truth, "mc_weight", 1.0))

            # Step 0: All Events
            res["cut_counts"][0] += 1
            res["raw_weights"][0] += rw
            res["mc_weights"][0] += mw

            # Step 1: LHC IP1 (100% pass on MC)
            res["cut_counts"][1] += 1
            res["raw_weights"][1] += rw
            res["mc_weights"][1] += mw

            # Step 2: Stable Beams (100% pass on MC)
            res["cut_counts"][2] += 1
            res["raw_weights"][2] += rw
            res["mc_weights"][2] += mw

            # Step 3: Event Delta T (100% pass on MC)
            res["cut_counts"][3] += 1
            res["raw_weights"][3] += rw
            res["mc_weights"][3] += mw

            t_cbm.GetEntry(iev)
            p_t = bool(cut_timing(t_cbm.Digi_ScifiHits, t_cbm.Digi_MuFilterHits))
            p_v = bool(cut_veto(t_cbm.Digi_MuFilterHits, t_cbm.Digi_ScifiHits))
            p_d = bool(cut_dimuon(t_cbm.Digi_MuFilterHits))
            trks = last_trk_leaf.GetObject() if last_trk_leaf else getattr(t_cbm, "Reco_MuonTracks", None)
            p_tr = bool(cut_tracks(trks)) if trks is not None else False

            if p_t:
                res["cut_counts"][4] += 1
                res["raw_weights"][4] += rw
                res["mc_weights"][4] += mw
                if p_v:
                    res["cut_counts"][5] += 1
                    res["raw_weights"][5] += rw
                    res["mc_weights"][5] += mw
                    if p_d:
                        res["cut_counts"][6] += 1
                        res["raw_weights"][6] += rw
                        res["mc_weights"][6] += mw
                        if p_tr:
                            res["cut_counts"][7] += 1
                            res["raw_weights"][7] += rw
                            res["mc_weights"][7] += mw
                            passed_entries.append(iev)

            if record_failures and not (p_t and p_v and p_d and p_tr):
                h = getattr(t_cbm, "EventHeader.", None) or getattr(t_cbm, "EventHeader", None)
                ev_num = h.GetEventNumber() if h else iev
                if not p_t:
                    failed_stage = "SciFi_DS_Timing"
                elif not p_v:
                    failed_stage = "No_Veto_Hits"
                elif not p_d:
                    failed_stage = "DS_Dimuon_Activity"
                else:
                    failed_stage = "Two_DS_Tracks"

                all_f = []
                if not p_t: all_f.append("SciFi_DS_Timing")
                if not p_v: all_f.append("No_Veto_Hits")
                if not p_d: all_f.append("DS_Dimuon_Activity")
                if not p_tr: all_f.append("Two_DS_Tracks")

                failed_records.append({
                    "partition": partition_id,
                    "event_number": ev_num,
                    "failed_cut_stage": failed_stage,
                    "entry": iev,
                    "failed_cuts": ";".join(all_f),
                })

    res["passed"] = len(passed_entries)

    if output_path:
        if len(passed_entries) == 0 and skip_empty:
            res["status"] = "SKIPPED_EMPTY"
        else:
            if os.path.islink(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            f_out = ROOT.TFile.Open(output_path, "RECREATE")
            if f_out and not f_out.IsZombie():
                f_out.cd()
                if t_cbm:
                    t_cbm.SetBranchStatus("*", 1)
                    out_cbm = t_cbm.CloneTree(0)
                    for iev in passed_entries:
                        t_cbm.GetEntry(iev)
                        out_cbm.Fill()
                    out_cbm.Write()
                if t_truth:
                    t_truth.SetBranchStatus("*", 1)
                    out_truth = t_truth.CloneTree(0)
                    for iev in passed_entries:
                        t_truth.GetEntry(iev)
                        out_truth.Fill()
                    out_truth.Write()
                for k in ["BranchList", "TimeBasedBranchList", "FileHeader", "FileHeaderHeader"]:
                    obj = f_in.Get(k)
                    if obj:
                        obj.Write(k, ROOT.TObject.kSingleKey)
                f_out.Close()

    if t_cbm:
        t_cbm.ResetBranchAddresses()
    if t_truth:
        t_truth.ResetBranchAddresses()
    f_in.Close()
    return res, failed_records


def process_single_partition_pipeline(
    partition: int,
    main_input: str,
    main_output: Optional[str],
    sig_input: Optional[str],
    sig_output: Optional[str],
    ind_input: Optional[str],
    ind_output: Optional[str],
    config_path: str,
    create_symlinks: bool = True,
    skip_empty: bool = False,
    skip_existing: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Process a single partition for main, direct signal, and indirect signal datasets:
    1. Evaluates sequential selection cuts on all 3 streams and clones passed events (both cbmsim and mctruth).
    2. Records failure metadata for failed signal events.
    3. Symlinks auxiliary input files to output directory.
    """
    t0 = time.time()

    if skip_existing:
        main_done = bool(not main_output or (os.path.isfile(main_output) and not os.path.islink(main_output) and os.path.getsize(main_output) > 200))
        sig_done = bool(not sig_output or not sig_input or not os.path.isfile(sig_input) or (os.path.isfile(sig_output) and not os.path.islink(sig_output) and os.path.getsize(sig_output) > 200))
        ind_done = bool(not ind_output or not ind_input or not os.path.isfile(ind_input) or (os.path.isfile(ind_output) and not os.path.islink(ind_output) and os.path.getsize(ind_output) > 200))
        if main_done and sig_done and ind_done:
            return {
                "partition": partition,
                "status": "SKIPPED_EXISTING",
                "time": 0.0,
                "main": {"partition": partition, "status": "SKIPPED_EXISTING", "total": 0, "passed": 0, "cut_counts": [0]*8, "raw_weights": [0.0]*8, "mc_weights": [0.0]*8},
                "sig": {"partition": partition, "status": "SKIPPED_EXISTING", "total": 0, "passed": 0, "cut_counts": [0]*8, "raw_weights": [0.0]*8, "mc_weights": [0.0]*8},
                "ind": {"partition": partition, "status": "SKIPPED_EXISTING", "total": 0, "passed": 0, "cut_counts": [0]*8, "raw_weights": [0.0]*8, "mc_weights": [0.0]*8},
            }, []

    import ROOT
    ROOT.gROOT.SetBatch(True)
    from snd import (
        load_trident_libraries,
        load_cutflow_config,
        VetoCut,
        SciFiDSTimeCut,
        DSDimuonActivityCut,
        DSHoughTrackCut,
        symlink_input_root_files,
    )
    load_trident_libraries()

    cutflow_cfg = load_cutflow_config(config_path) if os.path.exists(config_path) else {}
    cuts_list = cutflow_cfg.get("cuts", [])

    timing_cfg = next((c for c in cuts_list if c.get("filter") == "pass_scifi_ds_timing_cut" or "timing" in c.get("id", "")), {})
    cut_timing = SciFiDSTimeCut(
        min_delta_t=timing_cfg.get("min_delta_t", 0.0),
        require_valid=timing_cfg.get("require_valid", True),
        reversed=timing_cfg.get("reversed", False),
    )

    veto_cfg = next((c for c in cuts_list if c.get("filter") == "pass_veto_cut" or "veto" in c.get("id", "")), {})
    cut_veto = VetoCut(
        min_hits=veto_cfg.get("min_hits", 0),
        max_hits=veto_cfg.get("max_hits", 0),
        require_valid=veto_cfg.get("require_valid", False),
        time_threshold=veto_cfg.get("time_threshold", 0.3),
        use_timing=veto_cfg.get("use_timing", True),
    )

    ds_dimuon_cfg = next((c for c in cuts_list if c.get("filter") == "pass_ds_dimuon_cut" or "ds_dimuon" in c.get("id", "")), {})
    cut_dimuon = DSDimuonActivityCut(
        min_planes_dimuon=ds_dimuon_cfg.get("min_planes_dimuon", 3),
        min_hits_dimuon=ds_dimuon_cfg.get("min_hits_dimuon", 2),
        min_planes_other=ds_dimuon_cfg.get("min_planes_other", 3),
        min_hits_other=ds_dimuon_cfg.get("min_hits_other", 1),
        require_valid=ds_dimuon_cfg.get("require_valid", False),
        reversed=ds_dimuon_cfg.get("reversed", False),
    )

    two_ds_cfg = next((c for c in cuts_list if c.get("filter") == "pass_two_ds_tracks_cut" or "two_ds_tracks" in c.get("id", "") or "hough" in c.get("id", "")), {})
    cut_tracks = DSHoughTrackCut(
        min_tracks=two_ds_cfg.get("min_tracks", 2),
        max_tracks=two_ds_cfg.get("max_tracks", 2),
        track_type=two_ds_cfg.get("track_type", 13),
        reversed=two_ds_cfg.get("reversed", False),
    )

    # 1. Process Main File
    main_res, _ = _filter_partition_stream(
        input_path=main_input,
        output_path=main_output,
        cut_timing=cut_timing,
        cut_veto=cut_veto,
        cut_dimuon=cut_dimuon,
        cut_tracks=cut_tracks,
        skip_empty=skip_empty,
        record_failures=False,
        partition_id=partition,
    )
    main_res["partition"] = partition

    # 2. Process Direct Signal File
    sig_res, sig_failed_records = _filter_partition_stream(
        input_path=sig_input,
        output_path=sig_output,
        cut_timing=cut_timing,
        cut_veto=cut_veto,
        cut_dimuon=cut_dimuon,
        cut_tracks=cut_tracks,
        skip_empty=skip_empty,
        record_failures=True,
        partition_id=partition,
    )
    sig_res["partition"] = partition

    # 3. Process Indirect Signal File
    ind_res, _ = _filter_partition_stream(
        input_path=ind_input,
        output_path=ind_output,
        cut_timing=cut_timing,
        cut_veto=cut_veto,
        cut_dimuon=cut_dimuon,
        cut_tracks=cut_tracks,
        skip_empty=skip_empty,
        record_failures=False,
        partition_id=partition,
    )
    ind_res["partition"] = partition

    # 4. Create Symlinks for auxiliary files
    if create_symlinks and main_output and os.path.isfile(main_input):
        ex_files = {
            os.path.basename(p)
            for p in [main_output, main_input, sig_output, sig_input, ind_output, ind_input]
            if p
        }
        symlink_input_root_files(
            input_dir=os.path.dirname(os.path.abspath(main_input)),
            output_dir=os.path.dirname(os.path.abspath(main_output)),
            exclude_filenames=ex_files,
        )

    elapsed = time.time() - t0
    return {
        "partition": partition,
        "status": "OK",
        "time": elapsed,
        "main": main_res,
        "sig": sig_res,
        "ind": ind_res,
    }, sig_failed_records


def _pipeline_task_runner(args_tuple):
    return process_single_partition_pipeline(*args_tuple)


def load_cutflow_stats_from_root(filepath: str) -> List[Dict[str, Any]]:
    """
    Extract cutflow summary stats from an existing cutflow ROOT file.
    """
    if not os.path.exists(filepath):
        return []
    f = ROOT.TFile.Open(filepath, "READ")
    if not f or f.IsZombie():
        return []
    h = f.Get("h_cutflow")
    h_unw = f.Get("h_cutflow_unweighted")
    h_raw = f.Get("h_cutflow_raw_weight")
    h_mc = f.Get("h_cutflow_mc_weight")
    stats = []
    if h:
        for i in range(1, h.GetNbinsX() + 1):
            lbl = h.GetXaxis().GetBinLabel(i)
            cnt = h_unw.GetBinContent(i) if h_unw else h.GetBinContent(i)
            s_dict = {"id": lbl, "name": lbl, "count": cnt}
            if h_raw:
                s_dict["raw_weight"] = h_raw.GetBinContent(i)
            if h_mc:
                s_dict["mc_weight"] = h_mc.GetBinContent(i)
            stats.append(s_dict)
    f.Close()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Universal event prefiltering pipeline for SND@LHC data and MC using RDataFrame and DataManager.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input file path or pattern with '%%s' placeholder (e.g. /path/to/files/%%s/sndsw_raw-0000.root)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file pattern with '%%s' placeholder (optional if only generating cutflow)",
    )
    parser.add_argument(
        "-p", "--partitions",
        type=str,
        default=None,
        help="Partitions to process: range ('0-400'), comma-separated ('0,1,2'), or 'all'",
    )
    parser.add_argument(
        "--cutflow",
        type=str,
        default=None,
        help="Output ROOT file path for cutflow counters and diagnostic histograms (supports '%%s' or a single combined file)",
    )
    parser.add_argument(
        "--signal-input",
        type=str,
        default=None,
        help="Input file path or pattern for dedicated signal MC ROOT files to evaluate signal cutflow and overlay on canvas",
    )
    parser.add_argument(
        "--signal-cutflow",
        type=str,
        default=None,
        help="Existing signal cutflow ROOT file to overlay onto the cutflow canvas",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "config", "prefilter_cutflow_config.yaml"),
        help="Path to YAML configuration file defining cuts and diagnostic histograms",
    )
    parser.add_argument(
        "--enable-cuts",
        type=str,
        default=None,
        help="Cuts to enable: 'all', 'fiducial', or comma-separated list of cut IDs. Defaults to enabled flags in config.",
    )
    parser.add_argument(
        "--tree-name",
        type=str,
        default=None,
        help="Tree name override ('rawConv' for data, 'cbmsim' for MC). Auto-detected if omitted.",
    )

    # Multi-threading options
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Number of worker threads for RDataFrame processing (default: 1; >1 enables multi-threading)",
    )

    # Workflow options
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        default=False,
        help="Disable automatic symlink creation for input directory ROOT files",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        default=False,
        help="Do not save output ROOT files when 0 events pass the cut",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Maximum number of input files to process (-1 for all)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum events to process per file (useful for fast testing)",
    )
    parser.add_argument(
        "--signal-output",
        type=str,
        default=None,
        help="Output file pattern for filtered signal MC ROOT files (supports '%%s')",
    )
    parser.add_argument(
        "--indirect-input",
        type=str,
        default=None,
        help="Input file path or pattern for indirect charm dimuon MC ROOT files (supports '%%s')",
    )
    parser.add_argument(
        "--indirect-output",
        type=str,
        default=None,
        help="Output file pattern for filtered indirect charm dimuon ROOT files (supports '%%s')",
    )
    parser.add_argument(
        "--failed-cuts-csv",
        type=str,
        default=None,
        help="Output CSV file path to record partitions, event numbers, and failed cut stages for signal events",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=1,
        help="Number of concurrent worker processes for parallel partition processing (default: 1)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip partitions where output filtered files already exist",
    )
    parser.add_argument(
        "--sample-type",
        type=str,
        default="auto",
        choices=["auto", "neutrino", "passing_muon"],
        help="Type of sample being filtered (controls whether --nu-scaling or --pmu-scaling is applied to cutflow).",
    )
    parser.add_argument(
        "--nu-scaling",
        type=float,
        default=1.0,
        help="Scaling factor for neutrino cutflow histograms.",
    )
    parser.add_argument(
        "--pmu-scaling",
        type=float,
        default=1.0,
        help="Scaling factor for passing muon cutflow histograms.",
    )

    args = parser.parse_args()

    # Determine sample scaling factor
    if args.sample_type == "passing_muon":
        sample_scale = args.pmu_scaling
    elif args.sample_type == "neutrino":
        sample_scale = args.nu_scaling
    else:
        is_pmu = any("pmu" in str(p).lower() for p in (args.input or "", args.output or ""))
        sample_scale = args.pmu_scaling if is_pmu else args.nu_scaling

    # Load C++ analysis libraries and dictionaries
    load_trident_libraries()

    # Load cutflow config if available
    cutflow_cfg = {}
    if os.path.exists(args.config):
        cutflow_cfg = load_cutflow_config(args.config)

    # Resolve cuts configuration
    if "cuts" in cutflow_cfg and args.enable_cuts:
        if args.enable_cuts.lower() == "all":
            for c in cutflow_cfg["cuts"]:
                c["enabled"] = True
        elif args.enable_cuts.lower() == "fiducial":
            for c in cutflow_cfg["cuts"]:
                c["enabled"] = ("fiducial" in c.get("id", "") or c.get("id") == "step0_raw")
        else:
            selected_ids = [s.strip() for s in args.enable_cuts.split(",")]
            for c in cutflow_cfg["cuts"]:
                c["enabled"] = (c.get("id") in selected_ids or c.get("id") == "step0_raw")

    if not args.output and not args.cutflow:
        print("[Error] Must specify either -o/--output to write preselected files or --cutflow to produce cutflow histograms.")
        sys.exit(1)

    # Resolve input files
    if args.partitions:
        from scripts.extract_simulation_truth import parse_partitions
        selected_partitions = parse_partitions(args.partitions)
        input_files = []
        if selected_partitions is not None:
            for p in selected_partitions:
                p_str = str(p)
                c_p = args.input.count("%s")
                p_file = args.input % tuple(p_str for _ in range(c_p)) if c_p > 0 else args.input
                if os.path.exists(p_file):
                    input_files.append(p_file)
                else:
                    print(f"[Warning] Partition {p} input file not found: {p_file}")
        else:
            input_files = resolve_input_files(args.input, max_files=args.max_files)
    else:
        input_files = resolve_input_files(args.input, max_files=args.max_files)

    if not input_files:
        print(f"[Error] No files matched input pattern: {args.input}")
        sys.exit(1)

    # Instantiate cut objects from YAML configuration
    cuts_list = cutflow_cfg.get("cuts", [])

    ip1_cfg = next((c for c in cuts_list if c.get("filter") == "pass_ip1_cut" or "ip1" in c.get("id", "")), {})
    cut_ip1 = EventHeaderIP1Cut(reversed=ip1_cfg.get("reversed", False))

    sb_cfg = next((c for c in cuts_list if c.get("filter") == "pass_stable_beams_cut" or "stable_beams" in c.get("id", "")), {})
    cut_stable_beams = StableBeamsCut(beam_mode=sb_cfg.get("beam_mode", 11), reversed=sb_cfg.get("reversed", False))

    scifi_cfg = next((c for c in cuts_list if c.get("filter") == "pass_scifi_fiducial" or "scifi_fiducial" in c.get("id", "")), {})
    cut_scifi = AvgScifiFiducialCut(
        vertical_min=scifi_cfg.get("vertical_min", 200.0),
        vertical_max=scifi_cfg.get("vertical_max", 1200.0),
        horizontal_min=scifi_cfg.get("horizontal_min", 300.0),
        horizontal_max=scifi_cfg.get("horizontal_max", 1336.0),
        reversed=scifi_cfg.get("reversed", False),
    )

    ds_cfg = next((c for c in cuts_list if c.get("filter") == "pass_ds_fiducial" or "ds_fiducial" in c.get("id", "")), {})
    cut_ds = AvgDSFiducialCut(
        vertical_min=ds_cfg.get("vertical_min", 70.0),
        vertical_max=ds_cfg.get("vertical_max", 105.0),
        horizontal_min=ds_cfg.get("horizontal_min", 10.0),
        horizontal_max=ds_cfg.get("horizontal_max", 50.0),
        reversed=ds_cfg.get("reversed", False),
    )

    veto_cfg = next((c for c in cuts_list if c.get("filter") == "pass_veto_cut" or "veto" in c.get("id", "")), {})
    cut_veto = VetoCut(
        min_hits=veto_cfg.get("min_hits", 0),
        max_hits=veto_cfg.get("max_hits", 0),
        require_valid=veto_cfg.get("require_valid", False),
        time_threshold=veto_cfg.get("time_threshold", 0.3),
        use_timing=veto_cfg.get("use_timing", True),
    )

    station_cfg = next((c for c in cuts_list if c.get("filter") == "pass_scifi_station_cut" or "scifi_station" in c.get("id", "")), {})
    cut_scifi_station = SciFiStationCut(
        threshold=station_cfg.get("threshold", 0.0),
        excluded_stations=station_cfg.get("excluded_stations", [1]),
        reversed=station_cfg.get("reversed", False),
    )

    ds_dimuon_cfg = next((c for c in cuts_list if c.get("filter") == "pass_ds_dimuon_cut" or "ds_dimuon" in c.get("id", "")), {})
    cut_ds_dimuon = DSDimuonActivityCut(
        min_planes_dimuon=ds_dimuon_cfg.get("min_planes_dimuon", 3),
        min_hits_dimuon=ds_dimuon_cfg.get("min_hits_dimuon", 2),
        min_planes_other=ds_dimuon_cfg.get("min_planes_other", 3),
        min_hits_other=ds_dimuon_cfg.get("min_hits_other", 1),
        require_valid=ds_dimuon_cfg.get("require_valid", False),
        reversed=ds_dimuon_cfg.get("reversed", False),
    )

    ds_act_cfg = next((c for c in cuts_list if c.get("filter") == "pass_ds_activity_cut" or "ds_activity" in c.get("id", "")), {})
    cut_ds_activity = DSActivityCut(
        min_us_planes=ds_act_cfg.get("min_us_planes", 5),
        require_ds_hits=ds_act_cfg.get("require_ds_hits", True),
        require_valid=ds_act_cfg.get("require_valid", True),
        reversed=ds_act_cfg.get("reversed", False),
    )

    timing_cfg = next((c for c in cuts_list if c.get("filter") == "pass_scifi_ds_timing_cut" or "timing" in c.get("id", "")), {})
    cut_scifi_ds_timing = SciFiDSTimeCut(
        min_delta_t=timing_cfg.get("min_delta_t", 0.0),
        require_valid=timing_cfg.get("require_valid", True),
        reversed=timing_cfg.get("reversed", False),
    )

    two_ds_cfg = next((c for c in cuts_list if c.get("filter") == "pass_two_ds_tracks_cut" or "two_ds_tracks" in c.get("id", "") or "hough" in c.get("id", "")), {})
    cut_two_ds_tracks = DSHoughTrackCut(
        min_tracks=two_ds_cfg.get("min_tracks", 2),
        max_tracks=two_ds_cfg.get("max_tracks", 2),
        track_type=two_ds_cfg.get("track_type", 13),
        reversed=two_ds_cfg.get("reversed", False),
    )

    active_cut_list = [c["name"] for c in cutflow_cfg.get("cuts", []) if c.get("enabled", True)]

    print("=" * 80)
    print(" SND@LHC Universal Event Prefilter")
    print("=" * 80)
    print(f" Input Pattern:        {args.input}")
    print(f" Output Pattern:       {args.output}")
    print(f" Files Matched:        {len(input_files)}")
    thread_info = f"{args.jobs} threads (multi-threaded)" if args.jobs > 1 else "1 (single-threaded)"
    print(f" RDataFrame Workers:   {thread_info}")
    print(f" Cutflow Output:       {args.cutflow if args.cutflow else 'Disabled (counters only)'}")
    print(f" Config File:          {args.config}")
    print(f" Active Cuts ({len(active_cut_list)}):")
    for c in cutflow_cfg.get("cuts", []):
        if c.get("enabled", True):
            print(f"   * {c['id']}: {c.get('name', c['id'])}")
    print(f" Worker Processes:     {args.workers}")
    if args.signal_input:
        print(f" Signal Input Pattern:   {args.signal_input}")
    if args.indirect_input:
        print(f" Indirect Input Pattern: {args.indirect_input}")
    if args.failed_cuts_csv:
        print(f" Failed Cuts CSV:        {args.failed_cuts_csv}")
    print(f" Symlink Auxiliary:      {not args.no_symlinks}")
    print(f" Skip Empty Files:       {args.skip_empty}")
    print("=" * 80)

    if args.partitions:
        from scripts.extract_simulation_truth import parse_partitions
        selected_partitions = parse_partitions(args.partitions)
        if selected_partitions is None:
            selected_partitions = list(range(1, 1001))

        # Check if CSV output requested
        if args.failed_cuts_csv:
            csv_path = args.failed_cuts_csv
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            if not os.path.isfile(csv_path) or not args.skip_existing:
                with open(csv_path, "w") as f_csv:
                    f_csv.write("partition,event_number,failed_cut_stage,entry,failed_cuts\n")

        # Build task list
        tasks = []
        for p in selected_partitions:
            p_str = str(p)
            c_p_in = args.input.count("%s")
            m_in = args.input % tuple(p_str for _ in range(c_p_in)) if c_p_in > 0 else args.input
            m_out = None
            if args.output:
                c_p_out = args.output.count("%s")
                m_out = args.output % tuple(p_str for _ in range(c_p_out)) if c_p_out > 0 else args.output

            s_in = None
            if args.signal_input:
                c_p_sig = args.signal_input.count("%s")
                s_in = args.signal_input % tuple(p_str for _ in range(c_p_sig)) if c_p_sig > 0 else args.signal_input

            s_out = None
            if args.signal_output:
                c_p_sout = args.signal_output.count("%s")
                s_out = args.signal_output % tuple(p_str for _ in range(c_p_sout)) if c_p_sout > 0 else args.signal_output
            elif m_out and s_in:
                s_out = os.path.join(os.path.dirname(m_out), os.path.basename(s_in))

            ind_in = None
            if args.indirect_input:
                c_p_ind = args.indirect_input.count("%s")
                ind_in = args.indirect_input % tuple(p_str for _ in range(c_p_ind)) if c_p_ind > 0 else args.indirect_input

            ind_out = None
            if args.indirect_output:
                c_p_iout = args.indirect_output.count("%s")
                ind_out = args.indirect_output % tuple(p_str for _ in range(c_p_iout)) if c_p_iout > 0 else args.indirect_output
            elif m_out and ind_in:
                ind_out = os.path.join(os.path.dirname(m_out), os.path.basename(ind_in))

            tasks.append((p, m_in, m_out, s_in, s_out, ind_in, ind_out, args.config, not args.no_symlinks, args.skip_empty, args.skip_existing))

        print(f"\n[Parallel] Dispatching {len(tasks)} partition tasks across {args.workers} worker processes...\n")

        tot_main_in = 0
        tot_main_out = 0
        tot_sig_in = 0
        tot_sig_out = 0
        tot_ind_in = 0
        tot_ind_out = 0
        tot_sig_failed = 0

        tot_main_cutflow = [0] * 8
        tot_main_raw = [0.0] * 8
        tot_main_mc = [0.0] * 8

        tot_sig_cutflow = [0] * 8
        tot_sig_raw = [0.0] * 8
        tot_sig_mc = [0.0] * 8

        tot_ind_cutflow = [0] * 8
        tot_ind_raw = [0.0] * 8
        tot_ind_mc = [0.0] * 8

        completed_count = 0
        p_start_time = time.time()

        def _accumulate_result(p_res, failed_records):
            nonlocal tot_main_in, tot_main_out, tot_sig_in, tot_sig_out, tot_ind_in, tot_ind_out, tot_sig_failed
            m_res = p_res.get("main", {})
            s_res = p_res.get("sig", {})
            i_res = p_res.get("ind", {})

            tot_main_in += m_res.get("total", 0)
            tot_main_out += m_res.get("passed", 0)
            tot_sig_in += s_res.get("total", 0)
            tot_sig_out += s_res.get("passed", 0)
            tot_ind_in += i_res.get("total", 0)
            tot_ind_out += i_res.get("passed", 0)
            tot_sig_failed += len(failed_records)

            for idx in range(8):
                if "cut_counts" in m_res: tot_main_cutflow[idx] += m_res["cut_counts"][idx]
                if "raw_weights" in m_res: tot_main_raw[idx] += m_res["raw_weights"][idx]
                if "mc_weights" in m_res: tot_main_mc[idx] += m_res["mc_weights"][idx]

                if "cut_counts" in s_res: tot_sig_cutflow[idx] += s_res["cut_counts"][idx]
                if "raw_weights" in s_res: tot_sig_raw[idx] += s_res["raw_weights"][idx]
                if "mc_weights" in s_res: tot_sig_mc[idx] += s_res["mc_weights"][idx]

                if "cut_counts" in i_res: tot_ind_cutflow[idx] += i_res["cut_counts"][idx]
                if "raw_weights" in i_res: tot_ind_raw[idx] += i_res["raw_weights"][idx]
                if "mc_weights" in i_res: tot_ind_mc[idx] += i_res["mc_weights"][idx]

            if failed_records and args.failed_cuts_csv:
                with open(args.failed_cuts_csv, "a") as f_csv:
                    for r in failed_records:
                        f_csv.write(f"{r['partition']},{r['event_number']},{r['failed_cut_stage']},{r['entry']},{r['failed_cuts']}\n")
                    f_csv.flush()

        if args.workers > 1 or len(tasks) > 1:
            import multiprocessing as mp
            with mp.Pool(processes=max(args.workers, 1), maxtasksperchild=1) as pool:
                for p_res, failed_records in pool.imap_unordered(_pipeline_task_runner, tasks):
                    completed_count += 1
                    _accumulate_result(p_res, failed_records)

                    part = p_res["partition"]
                    m_tot = p_res["main"]["total"]
                    m_pass = p_res["main"]["passed"]
                    s_tot = p_res["sig"]["total"]
                    s_pass = p_res["sig"]["passed"]
                    i_tot = p_res["ind"]["total"]
                    i_pass = p_res["ind"]["passed"]

                    elapsed_so_far = time.time() - p_start_time
                    rate = completed_count / max(elapsed_so_far, 0.1)
                    eta_sec = (len(tasks) - completed_count) / max(rate, 0.001)
                    eta_str = str(timedelta(seconds=int(eta_sec)))

                    print(f"[{completed_count:4d}/{len(tasks)}] Part {part:4d}: Main {m_pass:2d}/{m_tot:2d} | Sig {s_pass:1d}/{s_tot:1d} | Ind {i_pass:1d}/{i_tot:1d} ({p_res['time']:.1f}s, ETA: {eta_str})")
        else:
            for t in tasks:
                completed_count += 1
                p_res, failed_records = process_single_partition_pipeline(*t)
                _accumulate_result(p_res, failed_records)

                part = p_res["partition"]
                m_tot = p_res["main"]["total"]
                m_pass = p_res["main"]["passed"]
                s_tot = p_res["sig"]["total"]
                s_pass = p_res["sig"]["passed"]
                i_tot = p_res["ind"]["total"]
                i_pass = p_res["ind"]["passed"]

                elapsed_so_far = time.time() - p_start_time
                rate = completed_count / max(elapsed_so_far, 0.1)
                eta_sec = (len(tasks) - completed_count) / max(rate, 0.001)
                eta_str = str(timedelta(seconds=int(eta_sec)))

                print(f"[{completed_count:4d}/{len(tasks)}] Part {part:4d}: Main {m_pass:2d}/{m_tot:2d} | Sig {s_pass:1d}/{s_tot:1d} | Ind {i_pass:1d}/{i_tot:1d} ({p_res['time']:.1f}s, ETA: {eta_str})")

        total_elapsed = time.time() - p_start_time
        overall_main_yield = (tot_main_out / tot_main_in * 100.0) if tot_main_in > 0 else 0.0
        overall_sig_yield = (tot_sig_out / tot_sig_in * 100.0) if tot_sig_in > 0 else 0.0
        overall_ind_yield = (tot_ind_out / tot_ind_in * 100.0) if tot_ind_in > 0 else 0.0

        cut_defs = [
            {"id": "step0_raw", "name": "All Events"},
            {"id": "step1_ip1", "name": "LHC IP1"},
            {"id": "step2_stable_beams", "name": "Stable Beams"},
            {"id": "step3_event_deltat", "name": "Event #Delta t"},
            {"id": "step4_scifi_ds_timing", "name": "SciFi-DS Timing"},
            {"id": "step5_veto_hits", "name": "No Veto Hits"},
            {"id": "step6_ds_dimuon", "name": "DS Dimuon"},
            {"id": "step7_two_ds_tracks", "name": "Two DS Tracks"},
        ]
        all_stats = [
            {
                "id": c["id"],
                "name": c["name"],
                "count": tot_main_cutflow[i],
                "raw_weight": tot_main_raw[i],
                "mc_weight": tot_main_mc[i],
            }
            for i, c in enumerate(cut_defs)
        ]
        sig_stats = [
            {
                "id": c["id"],
                "name": c["name"],
                "count": tot_sig_cutflow[i],
                "raw_weight": tot_sig_raw[i],
                "mc_weight": tot_sig_mc[i],
            }
            for i, c in enumerate(cut_defs)
        ]
        ind_stats = [
            {
                "id": c["id"],
                "name": c["name"],
                "count": tot_ind_cutflow[i],
                "raw_weight": tot_ind_raw[i],
                "mc_weight": tot_ind_mc[i],
            }
            for i, c in enumerate(cut_defs)
        ]

        print_cutflow_summary(all_stats, sig_stats, args.cutflow)
        if args.cutflow:
            save_cutflow_root_file(
                output_filepath=args.cutflow,
                cutflow_stats=all_stats,
                stage_histograms={},
                signal_cutflow_stats=sig_stats,
                scale_factor=sample_scale,
                indirect_cutflow_stats=ind_stats,
            )
            print(f" Cutflow ROOT File:    {args.cutflow}")

        print("=" * 80)
        print(" PREFILTERING SUMMARY")
        print("=" * 80)
        print(f" Partitions Processed: {completed_count}/{len(tasks)}")
        print(f" Main Events:          {tot_main_out}/{tot_main_in} passed ({overall_main_yield:.2f}%)")
        print(f" Signal Events:        {tot_sig_out}/{tot_sig_in} passed ({overall_sig_yield:.2f}%)")
        print(f" Indirect Events:      {tot_ind_out}/{tot_ind_in} passed ({overall_ind_yield:.2f}%)")
        print(f" Signal Failed Events: {tot_sig_failed}")
        if args.failed_cuts_csv:
            print(f" Failed Cuts CSV:      {args.failed_cuts_csv}")
        if args.cutflow:
            print(f" Combined Cutflow:     {args.cutflow}")
        print(f" Total Elapsed Time:   {total_elapsed:.1f} seconds")
        print("=" * 80)
        return

    start_time = time.time()
    results = []

    cumulative_cutflow_stats: List[Dict[str, Any]] = []
    cumulative_sig_stats: List[Dict[str, Any]] = []
    cumulative_stage_histograms: Dict[str, Dict[str, Any]] = {}

    # Pre-load static signal cutflow if specified
    static_sig_stats = []
    if args.signal_cutflow:
        static_sig_stats = load_cutflow_stats_from_root(args.signal_cutflow)
        cumulative_sig_stats = list(static_sig_stats)

    # If a consolidated static signal sample is given without placeholders, evaluate it once upfront
    if args.signal_input and "%s" not in args.signal_input and not static_sig_stats:
        if os.path.exists(args.signal_input):
            print(f"Evaluating consolidated signal sample: {os.path.basename(args.signal_input)} ...")
            sig_res = prefilter_file(
                input_file=args.signal_input,
                output_file=None,
                cut_scifi=cut_scifi,
                cut_ds=cut_ds,
                cut_veto=cut_veto,
                cut_scifi_station=cut_scifi_station,
                cut_ds_dimuon=cut_ds_dimuon,
                cut_ds_activity=cut_ds_activity,
                cut_scifi_ds_timing=cut_scifi_ds_timing,
                cut_two_ds_tracks=cut_two_ds_tracks,
                cut_ip1=cut_ip1,
                cut_stable_beams=cut_stable_beams,
                cutflow_config=cutflow_cfg,
                record_cutflow=False,
                tree_name=args.tree_name,
                jobs=1,
                max_events=args.max_events,
                create_symlinks=False,
                skip_empty=False,
            )
            if sig_res["status"] == "OK":
                static_sig_stats = sig_res.get("cutflow_stats", [])
                cumulative_sig_stats = list(static_sig_stats)
                print(f"Signal sample: {sig_res['passed_entries']}/{sig_res['total_entries']} signal events passed\n")
        else:
            print(f"[Warning] Signal file not found: {args.signal_input}")

    for idx, input_file in enumerate(input_files, start=1):
        out_file = format_output_path(input_file, args.input, args.output, index=idx) if args.output else None
        print(f"[{idx}/{len(input_files)}] Processing: {os.path.basename(input_file)}")
        print(f"  Input:  {input_file}")
        if out_file:
            print(f"  Output: {out_file}")

        file_start = time.time()
        res = prefilter_file(
            input_file=input_file,
            output_file=out_file,
            cut_scifi=cut_scifi,
            cut_ds=cut_ds,
            cut_veto=cut_veto,
            cut_scifi_station=cut_scifi_station,
            cut_ds_dimuon=cut_ds_dimuon,
            cut_ds_activity=cut_ds_activity,
            cut_scifi_ds_timing=cut_scifi_ds_timing,
            cut_two_ds_tracks=cut_two_ds_tracks,
            cut_ip1=cut_ip1,
            cut_stable_beams=cut_stable_beams,
            cutflow_config=cutflow_cfg,
            record_cutflow=(args.cutflow is not None),
            tree_name=args.tree_name,
            jobs=args.jobs,
            max_events=args.max_events,
            create_symlinks=(not args.no_symlinks and bool(out_file)),
            skip_empty=args.skip_empty,
        )
        elapsed_file = time.time() - file_start

        if res["status"] == "OK":
            print(f"  Result: {res['passed_entries']}/{res['total_entries']} passed ({res['yield_pct']:.2f}%) in {elapsed_file:.1f}s")
            if res.get("symlinks_created", 0) > 0:
                print(f"  Symlinks: {res['symlinks_created']} file(s) linked")
        elif res["status"] == "SKIPPED_EMPTY":
            print(f"  Result: 0/{res['total_entries']} passed (skipped creating empty output)")
        else:
            print(f"  Result: ERROR: {res['error']}")

        results.append(res)

        # Process signal sample if requested and partitioned
        file_sig_stats = list(static_sig_stats) if static_sig_stats else []
        if args.signal_input and "%s" in args.signal_input:
            sig_file = format_output_path(input_file, args.input, args.signal_input, index=idx)
            if os.path.exists(sig_file):
                print(f"  Signal: Evaluating {os.path.basename(sig_file)} ...")
                sig_res = prefilter_file(
                    input_file=sig_file,
                    output_file=None,
                    cut_scifi=cut_scifi,
                    cut_ds=cut_ds,
                    cut_veto=cut_veto,
                    cut_scifi_station=cut_scifi_station,
                    cut_ds_dimuon=cut_ds_dimuon,
                    cut_ds_activity=cut_ds_activity,
                    cut_scifi_ds_timing=cut_scifi_ds_timing,
                    cut_two_ds_tracks=cut_two_ds_tracks,
                    cut_ip1=cut_ip1,
                    cut_stable_beams=cut_stable_beams,
                    cutflow_config=cutflow_cfg,
                    record_cutflow=False,
                    tree_name=args.tree_name,
                    jobs=1,
                    max_events=args.max_events,
                    create_symlinks=False,
                    skip_empty=False,
                )
                if sig_res["status"] == "OK":
                    file_sig_stats = sig_res.get("cutflow_stats", [])
                    print(f"  Signal: {sig_res['passed_entries']}/{sig_res['total_entries']} signal events passed")
            else:
                pass
                print(f"  [Warning] Signal file not found: {sig_file}")

        # Handle cutflow recording
        if args.cutflow and res["status"] in ("OK", "SKIPPED_EMPTY"):
            if "%s" in args.cutflow:
                file_cutflow_path = format_output_path(input_file, args.input, args.cutflow, index=idx)
                save_cutflow_root_file(
                    file_cutflow_path,
                    res["cutflow_stats"],
                    res["stage_histograms"],
                    signal_cutflow_stats=file_sig_stats,
                    scale_factor=sample_scale,
                )
                print(f"  Cutflow: Saved {file_cutflow_path}")
            else:
                # Accumulate for single combined cutflow file
                if not cumulative_cutflow_stats:
                    cumulative_cutflow_stats = [
                        {
                            "id": s["id"],
                            "name": s["name"],
                            "count": s["count"],
                            "raw_weight": s.get("raw_weight", s["count"]),
                            "mc_weight": s.get("mc_weight", s["count"]),
                        }
                        for s in res["cutflow_stats"]
                    ]
                else:
                    for i, s in enumerate(res["cutflow_stats"]):
                        cumulative_cutflow_stats[i]["count"] += s["count"]
                        if "raw_weight" in s:
                            cumulative_cutflow_stats[i]["raw_weight"] = cumulative_cutflow_stats[i].get("raw_weight", 0.0) + s["raw_weight"]
                        if "mc_weight" in s:
                            cumulative_cutflow_stats[i]["mc_weight"] = cumulative_cutflow_stats[i].get("mc_weight", 0.0) + s["mc_weight"]

                if file_sig_stats:
                    if not cumulative_sig_stats:
                        cumulative_sig_stats = [
                            {
                                "id": s["id"],
                                "name": s["name"],
                                "count": s["count"],
                                "raw_weight": s.get("raw_weight", s["count"]),
                                "mc_weight": s.get("mc_weight", s["count"]),
                            }
                            for s in file_sig_stats
                        ]
                    else:
                        for i, s in enumerate(file_sig_stats):
                            if i < len(cumulative_sig_stats):
                                cumulative_sig_stats[i]["count"] += s["count"]
                                if "raw_weight" in s:
                                    cumulative_sig_stats[i]["raw_weight"] = cumulative_sig_stats[i].get("raw_weight", 0.0) + s["raw_weight"]
                                if "mc_weight" in s:
                                    cumulative_sig_stats[i]["mc_weight"] = cumulative_sig_stats[i].get("mc_weight", 0.0) + s["mc_weight"]

                merge_stage_histograms(cumulative_stage_histograms, res["stage_histograms"])

        print("-" * 80)

    total_elapsed = time.time() - start_time
    total_in = sum(r["total_entries"] for r in results)
    total_out = sum(r["passed_entries"] for r in results)
    overall_yield = (total_out / total_in * 100.0) if total_in > 0 else 0.0
    successful_files = sum(1 for r in results if r["status"] in ("OK", "SKIPPED_EMPTY"))

    # Write combined cutflow file if single filepath was requested
    sig_summary = cumulative_sig_stats if cumulative_sig_stats else (static_sig_stats if static_sig_stats else None)
    if args.cutflow and "%s" not in args.cutflow and cumulative_cutflow_stats:
        save_cutflow_root_file(
            args.cutflow,
            cumulative_cutflow_stats,
            cumulative_stage_histograms,
            signal_cutflow_stats=sig_summary,
            scale_factor=sample_scale,
        )
        print_cutflow_summary(cumulative_cutflow_stats, sig_summary, args.cutflow)
    elif results and results[0].get("cutflow_stats"):
        print_cutflow_summary(results[0]["cutflow_stats"], sig_summary)

    print("=" * 80)
    print(" PREFILTERING SUMMARY")
    print("=" * 80)
    print(f" Files Processed:     {successful_files}/{len(input_files)} successful")
    print(f" Total Input Events:  {total_in}")
    print(f" Total Passed Events: {total_out} ({overall_yield:.2f}%)")
    print(f" Total Elapsed Time:  {total_elapsed:.1f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
