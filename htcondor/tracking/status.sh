#!/bin/bash

# ==============================================================================
# Helper script to monitor HTCondor dimuon tracking jobs and output files on EOS
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="/eos/user/i/idioniso/snd-numu-charm/data"

echo "=========================================================="
echo " SND@LHC Dimuon Tracking Status ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "=========================================================="

# 1. Output files on EOS
if [ -d "$DATA_DIR" ]; then
    DONE_COUNT=$(find "$DATA_DIR" -maxdepth 2 -name "*_dimuonReco.root" -o -name "*_muonReco.root" 2>/dev/null | wc -l)
    TOTAL_COUNT=$(wc -l < args_partitions.txt 2>/dev/null || echo 400)
    PERCENT=$(awk "BEGIN {printf \"%.1f\", ($DONE_COUNT / $TOTAL_COUNT) * 100}")
    echo " Completed tracking files on EOS : ${DONE_COUNT} / ${TOTAL_COUNT} (${PERCENT}%)"
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
