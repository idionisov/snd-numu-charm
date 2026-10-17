#!/bin/bash

# ==============================================================================
# Helper script to monitor ILP track reconstruction jobs on HTCondor
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget"
ARGS_FILE="${SCRIPT_DIR}/args_nu14_volTarget_1000.txt"
TOTAL_COUNT=$(wc -l < "${ARGS_FILE}" 2>/dev/null || echo 1000)

echo "=========================================================="
echo " SND@LHC ILP Track Reconstruction Status ($(date '+%Y-%m-%d %H:%M:%S'))"
echo " Dataset: sndlhc_15000fb-1_2022_down/nu14/volume_volTarget"
echo "=========================================================="

COMPLETED=0
if [ -d "out" ]; then
    COMPLETED=$(grep -l "successfully processed" out/*.out 2>/dev/null | wc -l || echo 0)
fi

ERRORS=0
if [ -d "err" ]; then
    ERRORS=$(find err/ -name "*.err" -type f -size +0c 2>/dev/null | wc -l || echo 0)
fi

PCT=$(awk "BEGIN {printf \"%.1f\", ($COMPLETED / $TOTAL_COUNT) * 100}")

echo " Partitions Processed : ${COMPLETED} / ${TOTAL_COUNT} (${PCT}%)"
echo " Error Log Count      : ${ERRORS}"
echo " Log Directory        : ${SCRIPT_DIR}/log"
echo " Output Directory     : ${SCRIPT_DIR}/out"

echo "----------------------------------------------------------"

if command -v condor_q >/dev/null 2>&1; then
    echo " HTCondor Queue Summary:"
    condor_q
else
    echo " 'condor_q' not found on this host. Run from lxplus to view queue state."
fi
echo "=========================================================="
