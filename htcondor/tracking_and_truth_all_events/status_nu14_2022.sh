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
    TRACK_DONE=$(find "$DATA_DIR" -maxdepth 2 -name "*dig_2MuTrks.root" -type f -size +500c 2>/dev/null | wc -l)
    TRUTH_DONE=$(find "$DATA_DIR" -maxdepth 2 -name "*dig_2MuTrks_truth.root" -type f -size +500c 2>/dev/null | wc -l)
    DIRECT_DONE=$(find "$DATA_DIR" -maxdepth 2 -name "*dig_2MuTrks_truth_charmDimuonDirect.root" -type f -size +500c 2>/dev/null | wc -l)
    INDIRECT_DONE=$(find "$DATA_DIR" -maxdepth 2 -name "*dig_2MuTrks_truth_charmDimuonIndirect.root" -type f -size +500c 2>/dev/null | wc -l)

    TRACK_PCT=$(awk "BEGIN {printf \"%.1f\", ($TRACK_DONE / $TOTAL_COUNT) * 100}")
    TRUTH_PCT=$(awk "BEGIN {printf \"%.1f\", ($TRUTH_DONE / $TOTAL_COUNT) * 100}")
    DIRECT_PCT=$(awk "BEGIN {printf \"%.1f\", ($DIRECT_DONE / $TOTAL_COUNT) * 100}")
    INDIRECT_PCT=$(awk "BEGIN {printf \"%.1f\", ($INDIRECT_DONE / $TOTAL_COUNT) * 100}")

    echo " Step 1 (Tracked All Events)  : ${TRACK_DONE} / ${TOTAL_COUNT} (${TRACK_PCT}%)"
    echo " Step 2 (Truth Extraction)    : ${TRUTH_DONE} / ${TOTAL_COUNT} (${TRUTH_PCT}%)"
    echo " Step 3a (Direct Skim)        : ${DIRECT_DONE} / ${TOTAL_COUNT} (${DIRECT_PCT}%)"
    echo " Step 3b (Indirect Skim)      : ${INDIRECT_DONE} / ${TOTAL_COUNT} (${INDIRECT_PCT}%)"
else
    echo " Output directory not yet created: ${DATA_DIR}"
fi

DISP_DIR="/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/partitions"
MASTER_DIRECT="/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_displays_charmDimuonDirect.root"
MASTER_INDIRECT="/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_displays_charmDimuonIndirect.root"

if [ -d "$DISP_DIR" ]; then
    DISP_DIR_DONE=$(find "$DISP_DIR/direct" -maxdepth 1 -name "displays_part*.root" -type f -size +500c 2>/dev/null | wc -l)
    DISP_INDIR_DONE=$(find "$DISP_DIR/indirect" -maxdepth 1 -name "displays_part*.root" -type f -size +500c 2>/dev/null | wc -l)
    echo " Step 4a (Direct Displays)    : ${DISP_DIR_DONE} partitions with signal displays"
    echo " Step 4b (Indirect Displays)  : ${DISP_INDIR_DONE} partitions with indirect displays"
fi

if [ -f "$MASTER_DIRECT" ]; then
    SIZE=$(ls -lh "$MASTER_DIRECT" | awk '{print $5}')
    echo " Master Direct Displays       : ${MASTER_DIRECT} (${SIZE})"
fi
if [ -f "$MASTER_INDIRECT" ]; then
    SIZE=$(ls -lh "$MASTER_INDIRECT" | awk '{print $5}')
    echo " Master Indirect Displays     : ${MASTER_INDIRECT} (${SIZE})"
fi

echo "----------------------------------------------------------"

if command -v condor_q >/dev/null 2>&1; then
    echo " HTCondor Queue Summary:"
    condor_q
else
    echo " 'condor_q' not found on this host. Run from lxplus to view queue state."
fi
echo "=========================================================="
