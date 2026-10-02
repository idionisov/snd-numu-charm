#include "PreselectionProcessor.h"
#include "sndSciFiTools.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <cstring>

namespace snd::trident {

bool PreselectionProcessor::validateHitFast(sndScifiHit* aHit, int ref_station, bool ref_orientation) {
    if (!aHit || !aHit->isValid()) return false;
    if (aHit->GetStation() != ref_station) return false;
    if (aHit->isVertical() != ref_orientation) return false;
    return true;
}

double PreselectionProcessor::qdcDensityFast(int reference_SiPM, const TClonesArray* scifi_hits, int radius) {
    if (!scifi_hits) return 0.0;
    double qdc_density = 0.0;
    bool orientation = (int(reference_SiPM / 100000) % 10 == 1);
    int ref_station = reference_SiPM / 1000000;
    int referenceChannel = snd::analysis_tools::calculateSiPMNumber(reference_SiPM);

    int entries = scifi_hits->GetEntries();
    for (int i = 0; i < entries; ++i) {
        auto* hit = static_cast<sndScifiHit*>(scifi_hits->At(i));
        if (!hit || !validateHitFast(hit, ref_station, orientation)) continue;
        int hitChannel = snd::analysis_tools::calculateSiPMNumber(hit->GetChannelID());
        if (radius == -1) {
            qdc_density += hit->GetSignal();
        } else {
            if (hitChannel > referenceChannel + radius) break;
            if (std::abs(referenceChannel - hitChannel) <= radius) {
                qdc_density += hit->GetSignal();
            }
        }
    }
    return qdc_density;
}

PreselectionMetrics PreselectionProcessor::process(
    const TClonesArray* scifiHits,
    const TClonesArray* mufiHits
) const
{
    PreselectionMetrics m;

    // A. Process SciFi Hits
    if (scifiHits) {
        int n_sf = scifiHits->GetEntries();
        int n_sf_valid = 0;
        std::set<int> sf_planes, sf_stations;
        double qdc_x = 0.0, qdc_y = 0.0;
        int hits_x = 0, hits_y = 0;

        int nhits_st[5] = {0};
        double qdc_st[5] = {0.0};
        int nhits_plane_h[5] = {0};
        int nhits_plane_v[5] = {0};
        double qdc_plane_h[5] = {0.0};
        double qdc_plane_v[5] = {0.0};

        for (int i = 0; i < n_sf; ++i) {
            auto* h = static_cast<sndScifiHit*>(scifiHits->At(i));
            if (!h || !h->isValid()) continue;

            n_sf_valid++;
            double qdc = h->GetSignal();
            m.scifi_sum_qdc += qdc;
            if (qdc > m.scifi_max_qdc) m.scifi_max_qdc = qdc;

            int st = h->GetStation(); // 1..5
            int is_vert = h->isVertical() ? 1 : 0; // 1 = XZ, 0 = YZ

            if (st >= 1 && st <= 5) {
                sf_stations.insert(st);
                sf_planes.insert((st - 1) * 2 + is_vert);
                nhits_st[st - 1] += 1;
                qdc_st[st - 1] += qdc;
                if (is_vert == 1) {
                    nhits_plane_v[st - 1] += 1;
                    qdc_plane_v[st - 1] += qdc;
                    qdc_x += qdc; hits_x += 1;
                } else {
                    nhits_plane_h[st - 1] += 1;
                    qdc_plane_h[st - 1] += qdc;
                    qdc_y += qdc; hits_y += 1;
                }
            }

            double hit_w = static_cast<double>(snd::analysis_tools::densityScifi(
                h->GetChannelID(), *const_cast<TClonesArray*>(scifiHits), fDensityRadius, 1000000.0, false));
            m.scifi_sum_hit_density += hit_w;
            if (hit_w > m.scifi_max_hit_density) m.scifi_max_hit_density = hit_w;

            double qdc_w = qdcDensityFast(h->GetChannelID(), scifiHits, static_cast<int>(fDensityRadius));
            m.scifi_sum_qdc_density += qdc_w;
            if (qdc_w > m.scifi_max_qdc_density) m.scifi_max_qdc_density = qdc_w;
        }

        m.scifi_nhits = n_sf_valid;
        if (m.scifi_nhits > 0) m.scifi_mean_qdc = m.scifi_sum_qdc / m.scifi_nhits;
        m.scifi_planes_hit = static_cast<int>(sf_planes.size());
        m.scifi_stations_hit = static_cast<int>(sf_stations.size());

        // Assign individual station and plane metrics
        m.scifi_nhits_st1 = nhits_st[0]; m.scifi_nhits_st2 = nhits_st[1];
        m.scifi_nhits_st3 = nhits_st[2]; m.scifi_nhits_st4 = nhits_st[3];
        m.scifi_nhits_st5 = nhits_st[4];

        m.scifi_qdc_st1 = qdc_st[0]; m.scifi_qdc_st2 = qdc_st[1];
        m.scifi_qdc_st3 = qdc_st[2]; m.scifi_qdc_st4 = qdc_st[3];
        m.scifi_qdc_st5 = qdc_st[4];

        m.scifi_nhits_st1_h = nhits_plane_h[0]; m.scifi_nhits_st1_v = nhits_plane_v[0];
        m.scifi_nhits_st2_h = nhits_plane_h[1]; m.scifi_nhits_st2_v = nhits_plane_v[1];
        m.scifi_nhits_st3_h = nhits_plane_h[2]; m.scifi_nhits_st3_v = nhits_plane_v[2];
        m.scifi_nhits_st4_h = nhits_plane_h[3]; m.scifi_nhits_st4_v = nhits_plane_v[3];
        m.scifi_nhits_st5_h = nhits_plane_h[4]; m.scifi_nhits_st5_v = nhits_plane_v[4];

        m.scifi_qdc_st1_h = qdc_plane_h[0]; m.scifi_qdc_st1_v = qdc_plane_v[0];
        m.scifi_qdc_st2_h = qdc_plane_h[1]; m.scifi_qdc_st2_v = qdc_plane_v[1];
        m.scifi_qdc_st3_h = qdc_plane_h[2]; m.scifi_qdc_st3_v = qdc_plane_v[2];
        m.scifi_qdc_st4_h = qdc_plane_h[3]; m.scifi_qdc_st4_v = qdc_plane_v[3];
        m.scifi_qdc_st5_h = qdc_plane_h[4]; m.scifi_qdc_st5_v = qdc_plane_v[4];

        for (int s = 0; s < 5; ++s) {
            if (nhits_plane_h[s] > m.scifi_max_nhits_plane) m.scifi_max_nhits_plane = nhits_plane_h[s];
            if (nhits_plane_v[s] > m.scifi_max_nhits_plane) m.scifi_max_nhits_plane = nhits_plane_v[s];
            if (qdc_plane_h[s] > m.scifi_max_qdc_plane) m.scifi_max_qdc_plane = qdc_plane_h[s];
            if (qdc_plane_v[s] > m.scifi_max_qdc_plane) m.scifi_max_qdc_plane = qdc_plane_v[s];
        }

        double up_qdc = qdc_st[0] + qdc_st[1];
        double down_qdc = qdc_st[3] + qdc_st[4];
        m.scifi_qdc_ratio_down_up = down_qdc / (up_qdc + 1e-4);

        double up_hits = nhits_st[0] + nhits_st[1];
        double down_hits = nhits_st[3] + nhits_st[4];
        m.scifi_nhits_ratio_down_up = down_hits / (up_hits + 1e-4);

        m.scifi_qdc_ratio_st5_st1 = qdc_st[4] / (qdc_st[0] + 1e-4);

        if (qdc_x + qdc_y > 0) m.scifi_qdc_asym_xy = (qdc_x - qdc_y) / (qdc_x + qdc_y);
        if (hits_x + hits_y > 0) m.scifi_nhits_asym_xy = double(hits_x - hits_y) / double(hits_x + hits_y);
    }

    // B. Process MuFilter Hits (Veto, US, DS)
    if (mufiHits) {
        int n_mf = mufiHits->GetEntries();
        std::set<int> veto_planes, us_planes, ds_planes, ds_stations;

        int us_nhits_st[5] = {0};
        double us_qdc_st[5] = {0.0};
        int ds_nhits_plane_h[4] = {0};
        int ds_nhits_plane_v[4] = {0};
        double ds_qdc_plane_h[4] = {0.0};
        double ds_qdc_plane_v[4] = {0.0};

        for (int i = 0; i < n_mf; ++i) {
            auto* h = static_cast<MuFilterHit*>(mufiHits->At(i));
            if (!h || !h->isValid()) continue;

            int sys = h->GetSystem(); // 1=Veto, 2=US, 3=DS
            int det_id = h->GetDetectorID();
            int plane_idx = (det_id % 10000) / 1000;
            double qdc = h->GetSignal();
            bool is_vert = h->isVertical();
            int o_idx = is_vert ? 1 : 0;

            if (sys == 1) { // Veto
                m.veto_nhits += 1;
                m.veto_sum_qdc += qdc;
                if (qdc > m.veto_max_qdc) m.veto_max_qdc = qdc;
                veto_planes.insert(plane_idx);
            }
            else if (sys == 2) { // US (5 stations)
                m.us_nhits += 1;
                m.us_sum_qdc += qdc;
                if (qdc > m.us_max_qdc) m.us_max_qdc = qdc;
                if (plane_idx >= 0 && plane_idx < 5) {
                    us_nhits_st[plane_idx] += 1;
                    us_qdc_st[plane_idx] += qdc;
                    us_planes.insert(plane_idx);
                }
            }
            else if (sys == 3) { // DS (4 stations, 7 planes)
                m.ds_nhits += 1;
                m.ds_sum_qdc += qdc;
                if (qdc > m.ds_max_qdc) m.ds_max_qdc = qdc;
                if (plane_idx >= 0 && plane_idx < 4) {
                    ds_stations.insert(plane_idx + 1);
                    if ((plane_idx + 1) > m.ds_deepest_station) m.ds_deepest_station = plane_idx + 1;

                    int global_ds_plane = (plane_idx < 3) ? (plane_idx * 2 + o_idx) : 6;
                    ds_planes.insert(global_ds_plane);
                    if ((global_ds_plane + 1) > m.ds_deepest_plane) m.ds_deepest_plane = global_ds_plane + 1;

                    if (is_vert) {
                        ds_nhits_plane_v[plane_idx] += 1;
                        ds_qdc_plane_v[plane_idx] += qdc;
                    } else {
                        ds_nhits_plane_h[plane_idx] += 1;
                        ds_qdc_plane_h[plane_idx] += qdc;
                    }
                }
            }
        }

        m.veto_planes_hit = static_cast<int>(veto_planes.size());
        m.us_planes_hit = static_cast<int>(us_planes.size());
        m.ds_planes_hit = static_cast<int>(ds_planes.size());
        m.ds_stations_hit = static_cast<int>(ds_stations.size());

        // Assign US station metrics
        m.us_nhits_st1 = us_nhits_st[0]; m.us_nhits_st2 = us_nhits_st[1];
        m.us_nhits_st3 = us_nhits_st[2]; m.us_nhits_st4 = us_nhits_st[3];
        m.us_nhits_st5 = us_nhits_st[4];

        m.us_qdc_st1 = us_qdc_st[0]; m.us_qdc_st2 = us_qdc_st[1];
        m.us_qdc_st3 = us_qdc_st[2]; m.us_qdc_st4 = us_qdc_st[3];
        m.us_qdc_st5 = us_qdc_st[4];

        // Assign DS individual plane metrics
        m.ds_nhits_st1_h = ds_nhits_plane_h[0]; m.ds_nhits_st1_v = ds_nhits_plane_v[0];
        m.ds_nhits_st2_h = ds_nhits_plane_h[1]; m.ds_nhits_st2_v = ds_nhits_plane_v[1];
        m.ds_nhits_st3_h = ds_nhits_plane_h[2]; m.ds_nhits_st3_v = ds_nhits_plane_v[2];
        m.ds_nhits_st4_v = ds_nhits_plane_v[3];

        m.ds_qdc_st1_h = ds_qdc_plane_h[0]; m.ds_qdc_st1_v = ds_qdc_plane_v[0];
        m.ds_qdc_st2_h = ds_qdc_plane_h[1]; m.ds_qdc_st2_v = ds_qdc_plane_v[1];
        m.ds_qdc_st3_h = ds_qdc_plane_h[2]; m.ds_qdc_st3_v = ds_qdc_plane_v[2];
        m.ds_qdc_st4_v = ds_qdc_plane_v[3];

        for (int s = 0; s < 4; ++s) {
            if (ds_nhits_plane_h[s] > m.ds_max_nhits_plane) m.ds_max_nhits_plane = ds_nhits_plane_h[s];
            if (ds_nhits_plane_v[s] > m.ds_max_nhits_plane) m.ds_max_nhits_plane = ds_nhits_plane_v[s];
            if (ds_qdc_plane_h[s] > m.ds_max_qdc_plane) m.ds_max_qdc_plane = ds_qdc_plane_h[s];
            if (ds_qdc_plane_v[s] > m.ds_max_qdc_plane) m.ds_max_qdc_plane = ds_qdc_plane_v[s];
        }

        double front_ds_qdc = ds_qdc_plane_h[0] + ds_qdc_plane_v[0] + ds_qdc_plane_h[1] + ds_qdc_plane_v[1];
        double back_ds_qdc  = ds_qdc_plane_h[2] + ds_qdc_plane_v[2] + ds_qdc_plane_v[3];
        m.ds_qdc_ratio_back_front = back_ds_qdc / (front_ds_qdc + 1e-4);

        double front_ds_hits = ds_nhits_plane_h[0] + ds_nhits_plane_v[0] + ds_nhits_plane_h[1] + ds_nhits_plane_v[1];
        double back_ds_hits  = ds_nhits_plane_h[2] + ds_nhits_plane_v[2] + ds_nhits_plane_v[3];
        m.ds_nhits_ratio_back_front = back_ds_hits / (front_ds_hits + 1e-4);

        double ds1_qdc = ds_qdc_plane_h[0] + ds_qdc_plane_v[0];
        double ds4_qdc = ds_qdc_plane_v[3];
        m.ds_qdc_ratio_ds4_ds1 = ds4_qdc / (ds1_qdc + 1e-4);
    }

    // C. Cross-System Global Metrics
    m.total_nhits = m.scifi_nhits + m.us_nhits + m.ds_nhits + m.veto_nhits;
    m.total_sum_qdc = m.scifi_sum_qdc + m.us_sum_qdc + m.ds_sum_qdc + m.veto_sum_qdc;

    m.ratio_ds_to_scifi_qdc = m.ds_sum_qdc / (m.scifi_sum_qdc + 1e-4);
    m.ratio_ds_to_scifi_nhits = double(m.ds_nhits) / double(m.scifi_nhits + 1e-4);
    m.ratio_us_to_scifi_qdc = m.us_sum_qdc / (m.scifi_sum_qdc + 1e-4);
    m.ratio_mufi_to_scifi_qdc = (m.us_sum_qdc + m.ds_sum_qdc) / (m.scifi_sum_qdc + 1e-4);

    return m;
}

} // namespace snd::trident
