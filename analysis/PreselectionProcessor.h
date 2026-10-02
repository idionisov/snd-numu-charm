#pragma once

#include <vector>
#include <string>
#include "TClonesArray.h"
#include "sndScifiHit.h"
#include "MuFilterHit.h"

namespace snd::trident {

struct PreselectionMetrics {
    // 1. SciFi Global & Integrated Activity
    int scifi_nhits{0};
    double scifi_sum_qdc{0.0};
    double scifi_max_qdc{0.0};
    double scifi_mean_qdc{0.0};
    int scifi_planes_hit{0};
    int scifi_stations_hit{0};
    int scifi_max_nhits_plane{0};
    double scifi_max_qdc_plane{0.0};

    // 2. SciFi Density Weights
    double scifi_max_hit_density{0.0};
    double scifi_sum_hit_density{0.0};
    double scifi_max_qdc_density{0.0};
    double scifi_sum_qdc_density{0.0};

    // 3. SciFi Stations (1 to 5)
    int scifi_nhits_st1{0};
    int scifi_nhits_st2{0};
    int scifi_nhits_st3{0};
    int scifi_nhits_st4{0};
    int scifi_nhits_st5{0};

    double scifi_qdc_st1{0.0};
    double scifi_qdc_st2{0.0};
    double scifi_qdc_st3{0.0};
    double scifi_qdc_st4{0.0};
    double scifi_qdc_st5{0.0};

    // 4. SciFi Individual 10 Tracking Planes (Horizontal YZ and Vertical XZ)
    int scifi_nhits_st1_h{0};
    int scifi_nhits_st1_v{0};
    int scifi_nhits_st2_h{0};
    int scifi_nhits_st2_v{0};
    int scifi_nhits_st3_h{0};
    int scifi_nhits_st3_v{0};
    int scifi_nhits_st4_h{0};
    int scifi_nhits_st4_v{0};
    int scifi_nhits_st5_h{0};
    int scifi_nhits_st5_v{0};

    double scifi_qdc_st1_h{0.0};
    double scifi_qdc_st1_v{0.0};
    double scifi_qdc_st2_h{0.0};
    double scifi_qdc_st2_v{0.0};
    double scifi_qdc_st3_h{0.0};
    double scifi_qdc_st3_v{0.0};
    double scifi_qdc_st4_h{0.0};
    double scifi_qdc_st4_v{0.0};
    double scifi_qdc_st5_h{0.0};
    double scifi_qdc_st5_v{0.0};

    // 5. SciFi Ratios & Asymmetries
    double scifi_qdc_ratio_down_up{0.0};
    double scifi_nhits_ratio_down_up{0.0};
    double scifi_qdc_ratio_st5_st1{0.0};
    double scifi_qdc_asym_xy{0.0};
    double scifi_nhits_asym_xy{0.0};

    // 6. MuFilter Veto System (System 1)
    int veto_nhits{0};
    double veto_sum_qdc{0.0};
    int veto_planes_hit{0};
    double veto_max_qdc{0.0};

    // 7. Upstream MuFilter (US, System 2 - 5 Stations)
    int us_nhits{0};
    double us_sum_qdc{0.0};
    int us_planes_hit{0};
    double us_max_qdc{0.0};

    int us_nhits_st1{0};
    int us_nhits_st2{0};
    int us_nhits_st3{0};
    int us_nhits_st4{0};
    int us_nhits_st5{0};

    double us_qdc_st1{0.0};
    double us_qdc_st2{0.0};
    double us_qdc_st3{0.0};
    double us_qdc_st4{0.0};
    double us_qdc_st5{0.0};

    // 8. Downstream MuFilter (DS, System 3 - 4 Stations, 7 Planes)
    int ds_nhits{0};
    double ds_sum_qdc{0.0};
    double ds_max_qdc{0.0};
    int ds_planes_hit{0};
    int ds_stations_hit{0};
    int ds_deepest_station{0};
    int ds_deepest_plane{0};
    int ds_max_nhits_plane{0};
    double ds_max_qdc_plane{0.0};

    // 9. Downstream Individual 7 Tracking Planes
    int ds_nhits_st1_h{0};
    int ds_nhits_st1_v{0};
    int ds_nhits_st2_h{0};
    int ds_nhits_st2_v{0};
    int ds_nhits_st3_h{0};
    int ds_nhits_st3_v{0};
    int ds_nhits_st4_v{0};

    double ds_qdc_st1_h{0.0};
    double ds_qdc_st1_v{0.0};
    double ds_qdc_st2_h{0.0};
    double ds_qdc_st2_v{0.0};
    double ds_qdc_st3_h{0.0};
    double ds_qdc_st3_v{0.0};
    double ds_qdc_st4_v{0.0};

    // 10. Downstream Longitudinal Ratios & Penetration
    double ds_qdc_ratio_back_front{0.0};
    double ds_nhits_ratio_back_front{0.0};
    double ds_qdc_ratio_ds4_ds1{0.0};

    // 11. Cross-System Global Metrics
    int total_nhits{0};
    double total_sum_qdc{0.0};
    double ratio_ds_to_scifi_qdc{0.0};
    double ratio_ds_to_scifi_nhits{0.0};
    double ratio_us_to_scifi_qdc{0.0};
    double ratio_mufi_to_scifi_qdc{0.0};

    // Helper accessors for Profile1D (returning vector observables for RDataFrame)
    std::vector<double> get_scifi_qdc_vs_station_x() const { return {1.0, 2.0, 3.0, 4.0, 5.0}; }
    std::vector<double> get_scifi_qdc_vs_station_y() const {
        return {scifi_qdc_st1, scifi_qdc_st2, scifi_qdc_st3, scifi_qdc_st4, scifi_qdc_st5};
    }

    std::vector<double> get_scifi_nhits_vs_station_x() const { return {1.0, 2.0, 3.0, 4.0, 5.0}; }
    std::vector<double> get_scifi_nhits_vs_station_y() const {
        return {double(scifi_nhits_st1), double(scifi_nhits_st2), double(scifi_nhits_st3), double(scifi_nhits_st4), double(scifi_nhits_st5)};
    }

    std::vector<double> get_ds_qdc_vs_station_x() const { return {1.0, 2.0, 3.0, 4.0}; }
    std::vector<double> get_ds_qdc_vs_station_y() const {
        return {ds_qdc_st1_h + ds_qdc_st1_v, ds_qdc_st2_h + ds_qdc_st2_v, ds_qdc_st3_h + ds_qdc_st3_v, ds_qdc_st4_v};
    }

    std::vector<double> get_ds_nhits_vs_station_x() const { return {1.0, 2.0, 3.0, 4.0}; }
    std::vector<double> get_ds_nhits_vs_station_y() const {
        return {double(ds_nhits_st1_h + ds_nhits_st1_v), double(ds_nhits_st2_h + ds_nhits_st2_v), double(ds_nhits_st3_h + ds_nhits_st3_v), double(ds_nhits_st4_v)};
    }

    std::vector<double> get_detector_longitudinal_qdc_x() const {
        return {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                21.0, 22.0, 23.0, 24.0};
    }
    std::vector<double> get_detector_longitudinal_qdc_y() const {
        return {
            veto_sum_qdc * 0.5, veto_sum_qdc * 0.5,
            scifi_qdc_st1_h, scifi_qdc_st1_v,
            scifi_qdc_st2_h, scifi_qdc_st2_v,
            scifi_qdc_st3_h, scifi_qdc_st3_v,
            scifi_qdc_st4_h, scifi_qdc_st4_v,
            scifi_qdc_st5_h, scifi_qdc_st5_v,
            us_qdc_st1, us_qdc_st2, us_qdc_st3, us_qdc_st4, us_qdc_st5,
            ds_qdc_st1_h, ds_qdc_st1_v,
            ds_qdc_st2_h, ds_qdc_st2_v,
            ds_qdc_st3_h, ds_qdc_st3_v,
            ds_qdc_st4_v
        };
    }
};

class PreselectionProcessor {
public:
    explicit PreselectionProcessor(double densityRadius = 40.0)
        : fDensityRadius(densityRadius) {}

    PreselectionMetrics process(const TClonesArray* scifiHits, const TClonesArray* mufiHits) const;

    // Callable interface for ROOT RDataFrame:
    // df.Define("metrics", processor, ["Digi_ScifiHits", "Digi_MuFilterHits"])
    PreselectionMetrics operator()(const TClonesArray& scifiHits, const TClonesArray& mufiHits) const {
        return process(&scifiHits, &mufiHits);
    }

    void setDensityRadius(double r) { fDensityRadius = r; }
    double getDensityRadius() const { return fDensityRadius; }

private:
    double fDensityRadius{40.0};

    static bool validateHitFast(sndScifiHit* aHit, int ref_station, bool ref_orientation);
    static double qdcDensityFast(int reference_SiPM, const TClonesArray* scifi_hits, int radius);
};

} // namespace snd::trident
