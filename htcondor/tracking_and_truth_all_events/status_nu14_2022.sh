#!/bin/bash

# ==============================================================================
# Helper script to monitor 2022 nu14 (1000 Partitions) pipeline jobs on EOS
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget"
DISP_DIR="/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/partitions"
MASTER_DISP="/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_displays.root"
TOTAL_COUNT=$(wc -l < args_nu14_volTarget_1000.txt 2>/dev/null || echo 1000)

echo "=========================================================="
echo " SND@LHC nu14 2022 Pipeline Status (All Events) ($(date '+%Y-%m-%d %H:%M:%S'))"
echo " Dataset: sndlhc_15000fb-1_2022_down_nu14_volume_volTarget"
echo "=========================================================="

if [ -d "$DATA_DIR" ]; then
    TRACK_DONE=$(find "$DATA_DIR" -maxdepth 2 -name "*_2MuTrks.root" -type f -size +500c 2>/dev/null | wc -l)
    TRUTH_DONE=$(find "$DATA_DIR" -maxdepth 2 -name "*_2MuTrks_truth.root" -type f -size +500c 2>/dev/null | wc -l)
    
    TRACK_PCT=$(awk "BEGIN {printf \"%.1f\", ($TRACK_DONE / $TOTAL_COUNT) * 100}")
    TRUTH_PCT=$(awk "BEGIN {printf \"%.1f\", ($TRUTH_DONE / $TOTAL_COUNT) * 100}")

    echo " Step 1 (Tracked All Events) : ${TRACK_DONE} / ${TOTAL_COUNT} (${TRACK_PCT}%)"
    echo " Step 2 (Truth Extraction)   : ${TRUTH_DONE} / ${TOTAL_COUNT} (${TRUTH_PCT}%)"
else
    echo " Output directory not yet created: ${DATA_DIR}"
fi

if [ -d "$DISP_DIR" ]; then
    DISP_DONE=$(find "$DISP_DIR" -maxdepth 1 -name "displays_part*.root" -type f -size +500c 2>/dev/null | wc -l)
    DISP_PCT=$(awk "BEGIN {printf \"%.1f\", ($DISP_DONE / $TOTAL_COUNT) * 100}")
    echo " Step 3 (Event Displays)     : ${DISP_DONE} / ${TOTAL_COUNT} (${DISP_PCT}%)"
else
    echo " Step 3 (Event Displays)     : 0 / ${TOTAL_COUNT} (0.0%)"
fi

if [ -f "$MASTER_DISP" ]; then
    SIZE=$(ls -lh "$MASTER_DISP" | awk '{print $5}')
    echo " Master Displays File        : ${MASTER_DISP} (${SIZE})"
else
    echo " Master Displays File        : Not yet merged (run: ./htcondor/tracking_and_truth_all_events/merge_displays_nu14_2022.sh)"
fi

echo "----------------------------------------------------------"

if command -v condor_q >/dev/null 2>&1; then
    echo " HTCondor Queue Summary:"
    condor_q
else
    echo " 'condor_q' not found on this host. Run from lxplus to view queue state."
fi
echo "=========================================================="
