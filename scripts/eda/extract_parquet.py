#!/usr/bin/env python3
"""
scripts/eda/extract_parquet.py
------------------------------
Extracts hit occupancy metrics, track isolation observables, and reconstructed
track line parameters from ROOT trees into a consolidated Apache Parquet dataset
for exploratory data analysis (EDA) and machine learning.

Output Columns:
  - Event metadata: event_id, category, is_signal, mc_weight
  - Hit metrics (SciFi & MuFilter): all 21 occupancy/isolation variables
  - Track line parameters (Track 0 & Track 1): start/stop (x,y,z), slopes, angles, chi2/ndf
  - Relative track topology: opening angle
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
import pandas as pd
import ROOT

# Add project root to sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import DataManager, load_trident_libraries

load_trident_libraries(_repo_root)


def declare_cpp_extract_helpers():
    """Declares C++ helpers for hit metrics and track parameters extraction."""
    ROOT.gInterpreter.Declare("""
#ifndef EXTRACT_PARQUET_HELPERS_H
#define EXTRACT_PARQUET_HELPERS_H

#include <TClonesArray.h>
#include <TVector3.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "Scifi.h"
#include "MuFilter.h"
#include "sndScifiHit.h"
#include "MuFilterHit.h"
#include "sndRecoTrack.h"

namespace EdaExtract {
    struct ChannelInfo {
        float z{0.0f};
        float pos{0.0f};
        bool is_vert{false};
        bool valid{false};
    };

    inline std::vector<ChannelInfo>& getScifiChannelVec() {
        static std::vector<ChannelInfo> vec;
        return vec;
    }

    inline void init_scifi_geometry() {
        Scifi* scifi = dynamic_cast<Scifi*>(gROOT->GetListOfGlobals()->FindObject("Scifi"));
        if (!scifi) return;
        auto& vec = getScifiChannelVec();
        if (vec.size() < 6000000) {
            vec.assign(6000000, ChannelInfo{});
        }
        TVector3 left, right;
        for (int s = 1; s <= 5; ++s) {
            for (int o = 0; o <= 1; ++o) {
                for (int m = 0; m <= 2; ++m) {
                    for (int sipm = 0; sipm <= 3; ++sipm) {
                        for (int ch = 0; ch < 128; ++ch) {
                            int detID = s * 1000000 + o * 100000 + m * 10000 + sipm * 1000 + ch;
                            scifi->GetSiPMPosition(detID, left, right);
                            ChannelInfo info;
                            info.z = 0.5f * (left.Z() + right.Z());
                            info.is_vert = (o == 1);
                            info.pos = info.is_vert ? 0.5f * (left.X() + right.X()) : 0.5f * (left.Y() + right.Y());
                            info.valid = true;
                            if (detID < (int)vec.size()) {
                                vec[detID] = info;
                            }
                        }
                    }
                }
            }
        }
    }

    struct HitMetrics {
        int n_scifi_hits;
        int n_ds_hits;
        int max_scifi_plane_hits;
        double us_total_qdc;
        int max_ds_plane_hits;
        int n_ds_h_hits;
        int n_ds_v_hits;
        double ds_min_max_ratio_all;
        double ds_min_max_ratio_h;
        double ds_min_max_ratio_v;
        double scifi_min_max_ratio_all;
        double scifi_min_max_ratio_h;
        double scifi_min_max_ratio_v;
        double us_qdc_min_max_ratio;
        double us_qdc_plane4_over_plane0;
        int n_scifi_v_hits_pm2cm;
        int n_scifi_h_hits_pm2cm;
        int n_scifi_all_hits_pm2cm;
        double scifi_trk_iso_all;
        double scifi_trk_iso_h;
        double scifi_trk_iso_v;
    };

    inline double safe_ratio(double min_val, double max_val) {
        if (max_val <= 0.0) return 0.0;
        double r = min_val / max_val;
        return (r >= 1.0) ? 0.999999 : ((r < 0.0) ? 0.0 : r);
    }

    inline HitMetrics compute_hit_metrics(const TClonesArray& scifiHits, const TClonesArray& mufiHits, const TClonesArray& recoTracks) {
        HitMetrics m{0, 0, 0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0};

        int n_trks = recoTracks.GetEntries();
        const auto& ch_vec = getScifiChannelVec();
        int n_sf_h = 0;
        int n_sf_v = 0;
        int n_sf_iso_all = 0;
        int n_sf_iso_h = 0;
        int n_sf_iso_v = 0;

        int scifi_plane_hits[10] = {0};
        int scifi_h_hits[5] = {0};
        int scifi_v_hits[5] = {0};
        int n_sf = scifiHits.GetEntries();
        for (int i = 0; i < n_sf; ++i) {
            auto* h = static_cast<sndScifiHit*>(scifiHits.At(i));
            if (!h || !h->isValid()) continue;
            m.n_scifi_hits++;
            int st = h->GetStation();
            bool is_vert = h->isVertical();
            int o = is_vert ? 0 : 1;
            if (st >= 1 && st <= 5) {
                int p_idx = (st - 1) * 2 + o;
                scifi_plane_hits[p_idx]++;
                if (is_vert) {
                    scifi_v_hits[st - 1]++;
                } else {
                    scifi_h_hits[st - 1]++;
                }
            }
            if (is_vert) {
                n_sf_v++;
            } else {
                n_sf_h++;
            }

            int detID = h->GetDetectorID();
            bool in_window = false;
            if (n_trks > 0 && detID < (int)ch_vec.size() && ch_vec[detID].valid) {
                const auto& ch = ch_vec[detID];
                for (int t = 0; t < n_trks; ++t) {
                    auto* trk = static_cast<sndRecoTrack*>(recoTracks.At(t));
                    if (!trk) continue;
                    TVector3 trkPos = trk->getPointAtZ(ch.z);
                    float dist = is_vert ? std::abs(trkPos.X() - ch.pos) : std::abs(trkPos.Y() - ch.pos);
                    if (dist <= 2.0f) {
                        in_window = true;
                        break;
                    }
                }
            }
            if (in_window) {
                n_sf_iso_all++;
                if (is_vert) {
                    n_sf_iso_v++;
                } else {
                    n_sf_iso_h++;
                }
            }
        }
        for (int p = 0; p < 10; ++p) {
            if (scifi_plane_hits[p] > m.max_scifi_plane_hits) {
                m.max_scifi_plane_hits = scifi_plane_hits[p];
            }
        }

        m.n_scifi_v_hits_pm2cm = n_sf_iso_v;
        m.n_scifi_h_hits_pm2cm = n_sf_iso_h;
        m.n_scifi_all_hits_pm2cm = n_sf_iso_all;
        m.scifi_trk_iso_all = (m.n_scifi_hits > 0) ? (static_cast<double>(n_sf_iso_all) / m.n_scifi_hits) : 0.0;
        m.scifi_trk_iso_h = (n_sf_h > 0) ? (static_cast<double>(n_sf_iso_h) / n_sf_h) : 0.0;
        m.scifi_trk_iso_v = (n_sf_v > 0) ? (static_cast<double>(n_sf_iso_v) / n_sf_v) : 0.0;

        int sf_all_min = scifi_plane_hits[0];
        int sf_all_max = scifi_plane_hits[0];
        for (int p = 1; p < 10; ++p) {
            if (scifi_plane_hits[p] < sf_all_min) sf_all_min = scifi_plane_hits[p];
            if (scifi_plane_hits[p] > sf_all_max) sf_all_max = scifi_plane_hits[p];
        }
        m.scifi_min_max_ratio_all = safe_ratio(sf_all_min, sf_all_max);

        int sf_h_min = scifi_h_hits[0];
        int sf_h_max = scifi_h_hits[0];
        for (int p = 1; p < 5; ++p) {
            if (scifi_h_hits[p] < sf_h_min) sf_h_min = scifi_h_hits[p];
            if (scifi_h_hits[p] > sf_h_max) sf_h_max = scifi_h_hits[p];
        }
        m.scifi_min_max_ratio_h = safe_ratio(sf_h_min, sf_h_max);

        int sf_v_min = scifi_v_hits[0];
        int sf_v_max = scifi_v_hits[0];
        for (int p = 1; p < 5; ++p) {
            if (scifi_v_hits[p] < sf_v_min) sf_v_min = scifi_v_hits[p];
            if (scifi_v_hits[p] > sf_v_max) sf_v_max = scifi_v_hits[p];
        }
        m.scifi_min_max_ratio_v = safe_ratio(sf_v_min, sf_v_max);

        double us_plane_qdc[5] = {0.0};
        int ds_plane_hits[7] = {0};
        int ds_h_hits[3] = {0};
        int ds_v_hits[4] = {0};

        int n_mf = mufiHits.GetEntries();
        for (int i = 0; i < n_mf; ++i) {
            auto* h = static_cast<MuFilterHit*>(mufiHits.At(i));
            if (!h || !h->isValid()) continue;
            int sys = h->GetSystem();
            int det_id = h->GetDetectorID();
            int plane_idx = (det_id % 10000) / 1000;
            bool is_vert = h->isVertical();
            int o_idx = is_vert ? 1 : 0;

            if (sys == 2) {
                auto sigs = h->GetAllSignals();
                for (const auto& kv : sigs) {
                    m.us_total_qdc += kv.second;
                    if (plane_idx >= 0 && plane_idx < 5) {
                        us_plane_qdc[plane_idx] += kv.second;
                    }
                }
            } else if (sys == 3) {
                m.n_ds_hits++;
                if (is_vert) {
                    m.n_ds_v_hits++;
                    if (plane_idx >= 0 && plane_idx < 4) {
                        ds_v_hits[plane_idx]++;
                    }
                } else {
                    m.n_ds_h_hits++;
                    if (plane_idx >= 0 && plane_idx < 3) {
                        ds_h_hits[plane_idx]++;
                    }
                }
                if (plane_idx >= 0 && plane_idx < 4) {
                    int global_ds_plane = (plane_idx < 3) ? (plane_idx * 2 + o_idx) : 6;
                    if (global_ds_plane >= 0 && global_ds_plane < 7) {
                        ds_plane_hits[global_ds_plane]++;
                    }
                }
            }
        }
        for (int p = 0; p < 7; ++p) {
            if (ds_plane_hits[p] > m.max_ds_plane_hits) {
                m.max_ds_plane_hits = ds_plane_hits[p];
            }
        }

        int ds_all_min = ds_plane_hits[0];
        int ds_all_max = ds_plane_hits[0];
        for (int p = 1; p < 7; ++p) {
            if (ds_plane_hits[p] < ds_all_min) ds_all_min = ds_plane_hits[p];
            if (ds_plane_hits[p] > ds_all_max) ds_all_max = ds_plane_hits[p];
        }
        m.ds_min_max_ratio_all = safe_ratio(ds_all_min, ds_all_max);

        int ds_h_min = ds_h_hits[0];
        int ds_h_max = ds_h_hits[0];
        for (int p = 1; p < 3; ++p) {
            if (ds_h_hits[p] < ds_h_min) ds_h_min = ds_h_hits[p];
            if (ds_h_hits[p] > ds_h_max) ds_h_max = ds_h_hits[p];
        }
        m.ds_min_max_ratio_h = safe_ratio(ds_h_min, ds_h_max);

        int ds_v_min = ds_v_hits[0];
        int ds_v_max = ds_v_hits[0];
        for (int p = 1; p < 4; ++p) {
            if (ds_v_hits[p] < ds_v_min) ds_v_min = ds_v_hits[p];
            if (ds_v_hits[p] > ds_v_max) ds_v_max = ds_v_hits[p];
        }
        m.ds_min_max_ratio_v = safe_ratio(ds_v_min, ds_v_max);

        double qdc_min = us_plane_qdc[0];
        double qdc_max = us_plane_qdc[0];
        for (int p = 1; p < 5; ++p) {
            if (us_plane_qdc[p] < qdc_min) qdc_min = us_plane_qdc[p];
            if (us_plane_qdc[p] > qdc_max) qdc_max = us_plane_qdc[p];
        }
        m.us_qdc_min_max_ratio = safe_ratio(qdc_min, qdc_max);
        m.us_qdc_plane4_over_plane0 = (us_plane_qdc[0] > 0.0) ? (us_plane_qdc[4] / us_plane_qdc[0]) : 0.0;

        return m;
    }

    inline HitMetrics compute_hit_metrics(const TClonesArray& scifiHits, const TClonesArray& mufiHits) {
        TClonesArray emptyArr("sndRecoTrack", 0);
        return compute_hit_metrics(scifiHits, mufiHits, emptyArr);
    }

    struct TrackMetrics {
        int n_reco_tracks;
        float trk0_start_x{0.0f}, trk0_start_y{0.0f}, trk0_start_z{0.0f};
        float trk0_stop_x{0.0f}, trk0_stop_y{0.0f}, trk0_stop_z{0.0f};
        float trk0_slope_xz{0.0f}, trk0_slope_yz{0.0f};
        float trk0_angle_xz{0.0f}, trk0_angle_yz{0.0f};
        float trk0_chi2{0.0f};
        int trk0_ndf{0};
        float trk0_chi2_ndf{0.0f};

        float trk1_start_x{0.0f}, trk1_start_y{0.0f}, trk1_start_z{0.0f};
        float trk1_stop_x{0.0f}, trk1_stop_y{0.0f}, trk1_stop_z{0.0f};
        float trk1_slope_xz{0.0f}, trk1_slope_yz{0.0f};
        float trk1_angle_xz{0.0f}, trk1_angle_yz{0.0f};
        float trk1_chi2{0.0f};
        int trk1_ndf{0};
        float trk1_chi2_ndf{0.0f};

        float tracks_opening_angle{0.0f};
    };

    inline TrackMetrics compute_track_metrics(const TClonesArray& recoTracks) {
        TrackMetrics tm{};
        int n = recoTracks.GetEntries();
        tm.n_reco_tracks = n;
        if (n == 0) return tm;

        auto* t0 = static_cast<sndRecoTrack*>(recoTracks.At(0));
        if (t0) {
            TVector3 p0 = t0->getStart();
            TVector3 s0 = t0->getStop();
            TVector3 m0 = t0->getTrackMom();
            tm.trk0_start_x = p0.X();
            tm.trk0_start_y = p0.Y();
            tm.trk0_start_z = p0.Z();
            tm.trk0_stop_x = s0.X();
            tm.trk0_stop_y = s0.Y();
            tm.trk0_stop_z = s0.Z();
            tm.trk0_slope_xz = (std::abs(m0.Z()) > 1e-6) ? (m0.X() / m0.Z()) : 0.0f;
            tm.trk0_slope_yz = (std::abs(m0.Z()) > 1e-6) ? (m0.Y() / m0.Z()) : 0.0f;
            tm.trk0_angle_xz = t0->getAngleXZ();
            tm.trk0_angle_yz = t0->getAngleYZ();
            tm.trk0_chi2 = t0->getChi2();
            tm.trk0_ndf = t0->getNdf();
            tm.trk0_chi2_ndf = t0->getChi2Ndf();
        }

        if (n >= 2) {
            auto* t1 = static_cast<sndRecoTrack*>(recoTracks.At(1));
            if (t1) {
                TVector3 p1 = t1->getStart();
                TVector3 s1 = t1->getStop();
                TVector3 m1 = t1->getTrackMom();
                tm.trk1_start_x = p1.X();
                tm.trk1_start_y = p1.Y();
                tm.trk1_start_z = p1.Z();
                tm.trk1_stop_x = s1.X();
                tm.trk1_stop_y = s1.Y();
                tm.trk1_stop_z = s1.Z();
                tm.trk1_slope_xz = (std::abs(m1.Z()) > 1e-6) ? (m1.X() / m1.Z()) : 0.0f;
                tm.trk1_slope_yz = (std::abs(m1.Z()) > 1e-6) ? (m1.Y() / m1.Z()) : 0.0f;
                tm.trk1_angle_xz = t1->getAngleXZ();
                tm.trk1_angle_yz = t1->getAngleYZ();
                tm.trk1_chi2 = t1->getChi2();
                tm.trk1_ndf = t1->getNdf();
                tm.trk1_chi2_ndf = t1->getChi2Ndf();

                if (t0) {
                    TVector3 m0 = t0->getTrackMom();
                    tm.tracks_opening_angle = m0.Angle(m1);
                }
            }
        }
        return tm;
    }
}

#endif // EXTRACT_PARQUET_HELPERS_H
""")


def define_features(df):
    """Enriches RDataFrame with all 21 hit metrics and track parameters."""
    cols = [str(c) for c in df.GetColumnNames()]
    has_reco = "Reco_MuonTracks" in cols

    if has_reco:
        df = df.Define("hit_metrics", "EdaExtract::compute_hit_metrics(Digi_ScifiHits, Digi_MuFilterHits, Reco_MuonTracks)")
        df = df.Define("trk_metrics", "EdaExtract::compute_track_metrics(Reco_MuonTracks)")
    else:
        df = df.Define("hit_metrics", "EdaExtract::compute_hit_metrics(Digi_ScifiHits, Digi_MuFilterHits)")
        df = df.Define("trk_metrics", "EdaExtract::compute_track_metrics(TClonesArray(\"sndRecoTrack\", 0))")

    return (
        df
          # Hit & Occupancy Metrics
          .Define("n_scifi_hits", "hit_metrics.n_scifi_hits")
          .Define("n_ds_hits", "hit_metrics.n_ds_hits")
          .Define("max_scifi_plane_hits", "hit_metrics.max_scifi_plane_hits")
          .Define("us_total_qdc", "hit_metrics.us_total_qdc")
          .Define("max_ds_plane_hits", "hit_metrics.max_ds_plane_hits")
          .Define("n_ds_h_hits", "hit_metrics.n_ds_h_hits")
          .Define("n_ds_v_hits", "hit_metrics.n_ds_v_hits")
          .Define("ds_min_max_ratio_all", "hit_metrics.ds_min_max_ratio_all")
          .Define("ds_min_max_ratio_h", "hit_metrics.ds_min_max_ratio_h")
          .Define("ds_min_max_ratio_v", "hit_metrics.ds_min_max_ratio_v")
          .Define("scifi_min_max_ratio_all", "hit_metrics.scifi_min_max_ratio_all")
          .Define("scifi_min_max_ratio_h", "hit_metrics.scifi_min_max_ratio_h")
          .Define("scifi_min_max_ratio_v", "hit_metrics.scifi_min_max_ratio_v")
          .Define("us_qdc_min_max_ratio", "hit_metrics.us_qdc_min_max_ratio")
          .Define("us_qdc_plane4_over_plane0", "hit_metrics.us_qdc_plane4_over_plane0")
          .Define("n_scifi_v_hits_pm2cm", "hit_metrics.n_scifi_v_hits_pm2cm")
          .Define("n_scifi_h_hits_pm2cm", "hit_metrics.n_scifi_h_hits_pm2cm")
          .Define("n_scifi_all_hits_pm2cm", "hit_metrics.n_scifi_all_hits_pm2cm")
          .Define("scifi_trk_iso_all", "hit_metrics.scifi_trk_iso_all")
          .Define("scifi_trk_iso_h", "hit_metrics.scifi_trk_iso_h")
          .Define("scifi_trk_iso_v", "hit_metrics.scifi_trk_iso_v")
          # Track 0 Line Parameters
          .Define("n_reco_tracks", "trk_metrics.n_reco_tracks")
          .Define("trk0_start_x", "trk_metrics.trk0_start_x")
          .Define("trk0_start_y", "trk_metrics.trk0_start_y")
          .Define("trk0_start_z", "trk_metrics.trk0_start_z")
          .Define("trk0_stop_x", "trk_metrics.trk0_stop_x")
          .Define("trk0_stop_y", "trk_metrics.trk0_stop_y")
          .Define("trk0_stop_z", "trk_metrics.trk0_stop_z")
          .Define("trk0_slope_xz", "trk_metrics.trk0_slope_xz")
          .Define("trk0_slope_yz", "trk_metrics.trk0_slope_yz")
          .Define("trk0_angle_xz", "trk_metrics.trk0_angle_xz")
          .Define("trk0_angle_yz", "trk_metrics.trk0_angle_yz")
          .Define("trk0_chi2", "trk_metrics.trk0_chi2")
          .Define("trk0_ndf", "trk_metrics.trk0_ndf")
          .Define("trk0_chi2_ndf", "trk_metrics.trk0_chi2_ndf")
          # Track 1 Line Parameters
          .Define("trk1_start_x", "trk_metrics.trk1_start_x")
          .Define("trk1_start_y", "trk_metrics.trk1_start_y")
          .Define("trk1_start_z", "trk_metrics.trk1_start_z")
          .Define("trk1_stop_x", "trk_metrics.trk1_stop_x")
          .Define("trk1_stop_y", "trk_metrics.trk1_stop_y")
          .Define("trk1_stop_z", "trk_metrics.trk1_stop_z")
          .Define("trk1_slope_xz", "trk_metrics.trk1_slope_xz")
          .Define("trk1_slope_yz", "trk_metrics.trk1_slope_yz")
          .Define("trk1_angle_xz", "trk_metrics.trk1_angle_xz")
          .Define("trk1_angle_yz", "trk_metrics.trk1_angle_yz")
          .Define("trk1_chi2", "trk_metrics.trk1_chi2")
          .Define("trk1_ndf", "trk_metrics.trk1_ndf")
          .Define("trk1_chi2_ndf", "trk_metrics.trk1_chi2_ndf")
          # Multi-Track Topology
          .Define("tracks_opening_angle", "trk_metrics.tracks_opening_angle")
    )


FEATURE_COLUMNS = [
    # Hit Metrics (21)
    "n_scifi_hits", "n_ds_hits", "max_scifi_plane_hits", "us_total_qdc",
    "max_ds_plane_hits", "n_ds_h_hits", "n_ds_v_hits",
    "ds_min_max_ratio_all", "ds_min_max_ratio_h", "ds_min_max_ratio_v",
    "scifi_min_max_ratio_all", "scifi_min_max_ratio_h", "scifi_min_max_ratio_v",
    "us_qdc_min_max_ratio", "us_qdc_plane4_over_plane0",
    "n_scifi_v_hits_pm2cm", "n_scifi_h_hits_pm2cm", "n_scifi_all_hits_pm2cm",
    "scifi_trk_iso_all", "scifi_trk_iso_h", "scifi_trk_iso_v",
    # Track Metrics (23)
    "n_reco_tracks",
    "trk0_start_x", "trk0_start_y", "trk0_start_z",
    "trk0_stop_x", "trk0_stop_y", "trk0_stop_z",
    "trk0_slope_xz", "trk0_slope_yz",
    "trk0_angle_xz", "trk0_angle_yz",
    "trk0_chi2", "trk0_ndf", "trk0_chi2_ndf",
    "trk1_start_x", "trk1_start_y", "trk1_start_z",
    "trk1_stop_x", "trk1_stop_y", "trk1_stop_z",
    "trk1_slope_xz", "trk1_slope_yz",
    "trk1_angle_xz", "trk1_angle_yz",
    "trk1_chi2", "trk1_ndf", "trk1_chi2_ndf",
    "tracks_opening_angle",
]


def resolve_file_list(pattern: str) -> list[str]:
    """Resolve file pattern or wildcard path."""
    if "*" in pattern or "?" in pattern or "[" in pattern:
        files = sorted(glob.glob(pattern))
    elif os.path.isfile(pattern):
        files = [pattern]
    elif os.path.isdir(pattern):
        files = sorted(glob.glob(os.path.join(pattern, "*.root")))
    else:
        files = sorted(glob.glob(pattern))
    return [f for f in files if os.path.exists(f)]


def extract_dataframe_to_pandas(df_node, category: str, is_signal: int, weight_col: str | None = None) -> pd.DataFrame:
    """Extracts RDataFrame columns into a Pandas DataFrame."""
    cols_to_pull = list(FEATURE_COLUMNS)
    if weight_col and weight_col in df_node.GetColumnNames():
        cols_to_pull.append(weight_col)

    np_dict = df_node.AsNumpy(columns=cols_to_pull)
    pdf = pd.DataFrame(np_dict)
    pdf["category"] = category
    pdf["is_signal"] = int(is_signal)

    if weight_col and weight_col in pdf.columns:
        pdf["mc_weight"] = pdf[weight_col].astype(float)
        if weight_col != "mc_weight":
            pdf.drop(columns=[weight_col], inplace=True)
    else:
        pdf["mc_weight"] = 1.0

    return pdf


def main():
    parser = argparse.ArgumentParser(
        description="Extract hit metrics, track isolation, and track line parameters to Parquet."
    )
    parser.add_argument("-i", "--input-genie", required=True,
                        help="Input GENIE files or pattern (e.g. '/path/to/*/sndLHC.Genie-TGeant4*.root')")
    parser.add_argument("--input-pmu", default=None,
                        help="Optional input pmu2022 muon simulation files or pattern")
    parser.add_argument("--input-data", default=None,
                        help="Optional input collision data files or pattern")
    parser.add_argument("-o", "--output", default=os.path.join(_repo_root, "data/eda_features.parquet"),
                        help="Output Parquet file path (default: data/eda_features.parquet)")
    parser.add_argument("-g", "--geo", default=None,
                        help="Path to geometry file (auto-detected if omitted)")
    parser.add_argument("-j", "--threads", type=int, default=4,
                        help="Number of worker threads for RDataFrame")
    parser.add_argument("-n", "--max-events", type=int, default=None,
                        help="Maximum events to process per sample")
    args = parser.parse_args()

    if args.max_events:
        args.threads = 1
    if args.threads > 1:
        ROOT.EnableImplicitMT(args.threads)
        print(f"Enabled ROOT ImplicitMT with {args.threads} threads.")

    declare_cpp_extract_helpers()

    # Initialize geometry
    geo_path = args.geo
    if not geo_path:
        for c in [
            "/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/geofile_full.Ntuple-TGeant4.root",
            "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root",
        ]:
            if os.path.exists(c):
                geo_path = c
                break

    if geo_path and os.path.exists(geo_path):
        import SndlhcGeo
        print(f"Initializing geometry from: {geo_path}")
        _geo_interface = SndlhcGeo.GeoInterface(geo_path)
        ROOT.EdaExtract.init_scifi_geometry()
        print("Initialized SciFi channel geometry lookup table.")
    else:
        print("Warning: Geometry file not found. SciFi track isolation window calculations will default to 0.")

    dfs_to_concat = []

    # 1. Process GENIE
    genie_files = resolve_file_list(args.input_genie)
    if not genie_files:
        raise FileNotFoundError(f"No files matched GENIE pattern: {args.input_genie}")
    print(f"\nProcessing GENIE: found {len(genie_files)} files.")

    ch_genie = ROOT.TChain("cbmsim")
    ch_mc = ROOT.TChain("mctruth")
    for f in genie_files:
        ch_genie.Add(f)
        ch_mc.Add(f)
    ch_genie.AddFriend(ch_mc)

    dm_genie = DataManager(source=ch_genie)
    df_genie = dm_genie.get_rdataframe()
    if args.max_events:
        df_genie = df_genie.Range(args.max_events)
    df_genie = define_features(df_genie)

    has_truth = "mctruth.is_direct_charm_decay" in df_genie.GetColumnNames()
    weight_col = "mctruth.mc_weight" if "mctruth.mc_weight" in df_genie.GetColumnNames() else None

    if has_truth:
        df_sig = df_genie.Filter("mctruth.is_direct_charm_decay == 1", "GENIE Signal")
        df_ind = df_genie.Filter("mctruth.is_direct_charm_decay == 0 && mctruth.is_indirect_charm_decay == 1", "GENIE Indirect Fake")
        df_bkg = df_genie.Filter("mctruth.is_direct_charm_decay == 0 && mctruth.is_indirect_charm_decay == 0", "GENIE Background")

        print("Extracting GENIE Direct Signal...")
        pdf_sig = extract_dataframe_to_pandas(df_sig, "signal", is_signal=1, weight_col=weight_col)
        print(f"  -> {len(pdf_sig)} signal events")
        dfs_to_concat.append(pdf_sig)

        print("Extracting GENIE Indirect Fake...")
        pdf_ind = extract_dataframe_to_pandas(df_ind, "indirect_fake", is_signal=0, weight_col=weight_col)
        print(f"  -> {len(pdf_ind)} indirect fake events")
        dfs_to_concat.append(pdf_ind)

        print("Extracting GENIE Other Background...")
        pdf_bkg = extract_dataframe_to_pandas(df_bkg, "genie_bkg", is_signal=0, weight_col=weight_col)
        print(f"  -> {len(pdf_bkg)} background events")
        dfs_to_concat.append(pdf_bkg)
    else:
        print("Extracting GENIE Inclusive...")
        pdf_inc = extract_dataframe_to_pandas(df_genie, "genie_inclusive", is_signal=0, weight_col=weight_col)
        dfs_to_concat.append(pdf_inc)

    # 2. Process PMU (Optional)
    if args.input_pmu:
        pmu_files = resolve_file_list(args.input_pmu)
        if pmu_files:
            print(f"\nProcessing PMU2022: found {len(pmu_files)} files.")
            ch_pmu = ROOT.TChain("cbmsim")
            ch_pmu_mc = ROOT.TChain("mctruth")
            for f in pmu_files:
                ch_pmu.Add(f)
                ch_pmu_mc.Add(f)
            ch_pmu.AddFriend(ch_pmu_mc)

            dm_pmu = DataManager(source=ch_pmu)
            df_pmu = dm_pmu.get_rdataframe()
            if args.max_events:
                df_pmu = df_pmu.Range(args.max_events)
            df_pmu = define_features(df_pmu)

            pmu_weight_col = "mctruth.mc_weight" if "mctruth.mc_weight" in df_pmu.GetColumnNames() else None
            pdf_pmu = extract_dataframe_to_pandas(df_pmu, "pmu2022", is_signal=0, weight_col=pmu_weight_col)
            print(f"  -> {len(pdf_pmu)} pmu2022 events")
            dfs_to_concat.append(pdf_pmu)

    # 3. Process Collision Data (Optional)
    if args.input_data:
        data_files = resolve_file_list(args.input_data)
        if data_files:
            print(f"\nProcessing Collision Data: found {len(data_files)} files.")
            first_f = ROOT.TFile.Open(data_files[0])
            tree_name = "rawConv" if (first_f and first_f.Get("rawConv")) else "cbmsim"
            if first_f:
                first_f.Close()

            ch_data = ROOT.TChain(tree_name)
            for f in data_files:
                ch_data.Add(f)

            dm_data = DataManager(source=ch_data)
            df_data = dm_data.get_rdataframe()
            if args.max_events:
                df_data = df_data.Range(args.max_events)
            df_data = define_features(df_data)

            pdf_data = extract_dataframe_to_pandas(df_data, "data", is_signal=0, weight_col=None)
            print(f"  -> {len(pdf_data)} collision data events")
            dfs_to_concat.append(pdf_data)

    if not dfs_to_concat:
        print("No events extracted. Exiting.")
        return

    # Combine into single DataFrame and export to Parquet
    full_df = pd.concat(dfs_to_concat, ignore_index=True)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"\nSaving {len(full_df)} events with {len(full_df.columns)} columns to: {args.output}")
    full_df.to_parquet(args.output, index=False)
    print("Parquet extraction complete!")


if __name__ == "__main__":
    main()
