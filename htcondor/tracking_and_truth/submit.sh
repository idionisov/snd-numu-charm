#!/bin/bash
set -e

# ==============================================================================
# Helper submission script for HTCondor chained tracking + truth jobs
# Run on lxplus: ./htcondor/tracking_and_truth/submit.sh [extra_args...]
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! command -v condor_submit >/dev/null 2>&1; then
    echo "ERROR: 'condor_submit' not found in PATH." >&2
    echo "Please run this script from an lxplus node (e.g., ssh lxplus.cern.ch)." >&2
    exit 1
fi

mkdir -p out err log

TOTAL_PARTITIONS=$(wc -l < args_partitions.txt)
echo "=========================================================="
echo " Submitting SND@LHC Chained Tracking+Truth Jobs"
echo " Working directory : ${SCRIPT_DIR}"
echo " Total partitions  : ${TOTAL_PARTITIONS}"
echo " Flavour           : longlunch (max 2 hours per job)"
echo " Extra arguments   : $*"
echo "=========================================================="

condor_submit pipeline.sub "$@"

echo "Jobs submitted! You can monitor them using: ./htcondor/tracking_and_truth/status.sh"
