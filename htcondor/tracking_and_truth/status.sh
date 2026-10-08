#!/bin/bash

# ==============================================================================
# Helper script to monitor chained tracking + truth jobs and output files on EOS
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="/eos/user/i/idioniso/snd-numu-charm/data"
TOTAL_COUNT=$(wc -l < args_partitions.txt 2>/dev/null || echo 400)

echo "=========================================================="
echo " SND@LHC Chained Tracking+Truth Status ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "=========================================================="

# 1. Output files on EOS
if [ -d "$DATA_DIR" ]; then
    TRACK_DONE=$(find "$DATA_DIR" -maxdepth 2 -name "*_2MuTrks.root" -type f -size +500c 2>/dev/null | wc -l)
    TRUTH_DONE=$(find "$DATA_DIR" -maxdepth 2 -name "*_2MuTrks_truth.root" -type f -size +500c 2>/dev/null | wc -l)
    
    TRACK_PCT=$(awk "BEGIN {printf \"%.1f\", ($TRACK_DONE / $TOTAL_COUNT) * 100}")
    TRUTH_PCT=$(awk "BEGIN {printf \"%.1f\", ($TRUTH_DONE / $TOTAL_COUNT) * 100}")

    echo " Step 1 (Tracked 2MuTrks)   : ${TRACK_DONE} / ${TOTAL_COUNT} (${TRACK_PCT}%)"
    echo " Step 2 (Final Truth Files) : ${TRUTH_DONE} / ${TOTAL_COUNT} (${TRUTH_PCT}%)"
else
    echo " EOS data directory not found or unreachable: ${DATA_DIR}"
fi

echo "----------------------------------------------------------"

# 2. HTCondor queue status (if on lxplus)
if command -v condor_q >/dev/null 2>&1; then
    echo " HTCondor Queue Summary:"
    condor_q
else
    echo " 'condor_q' not found on this host. Run from lxplus to view queue state."
fi
echo "=========================================================="
