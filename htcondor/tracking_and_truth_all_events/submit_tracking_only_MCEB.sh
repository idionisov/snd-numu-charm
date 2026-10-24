#!/bin/bash
set -e

# ==============================================================================
# HTCondor Submission for 2022 nu14 MCEB Tracking Only (1000 partitions)
#
# Runs:
#   Input:  /eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_MCEB_dig.root
#   Output: /eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root
#
# Reconstructs tracks using tuned Hough parameters and drops any old tracks.
# MCTruth mapping can subsequently be executed with:
#   python3 scripts/map_mctruth_to_mceb.py -p 1-1000 -j 8
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! command -v condor_submit >/dev/null 2>&1; then
    echo "ERROR: 'condor_submit' not found in PATH." >&2
    echo "Please run this script from an lxplus node (e.g., ssh lxplus.cern.ch)." >&2
    exit 1
fi

mkdir -p out err log

ARGS_FILE="${SCRIPT_DIR}/args_nu14_volTarget_1000.txt"
DATA_DIR="/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget"
INPUT_PATTERN="${DATA_DIR}/%s/sndLHC.Genie-TGeant4_MCEB_dig.root"
TRACK_PATTERN="${DATA_DIR}/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root"
TRUTH_PATTERN="${DATA_DIR}/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root"
DISP_PATTERN="none"

TOTAL_PARTITIONS=$(wc -l < "${ARGS_FILE}")
echo "=========================================================="
echo " Submitting 2022 nu14 MCEB (Tracking Only) to HTCondor"
echo " Dataset        : sndlhc_15000fb-1_2022_down/nu14/volume_volTarget (MCEB)"
echo " Total Jobs     : ${TOTAL_PARTITIONS} partitions (1..1000)"
echo " Tracking Mode  : All Events (nTracks=0, --all-events)"
echo " Force Mode     : Enabled (FORCE=1)"
echo " Pipeline Mode  : Tracking Only (TRACKING_ONLY=1)"
echo " Input Pattern  : ${INPUT_PATTERN}"
echo " Track Target   : ${TRACK_PATTERN}"
echo " Extra Args     : $*"
echo "=========================================================="

condor_submit pipeline.sub \
    args_file="${ARGS_FILE}" \
    input_pattern="${INPUT_PATTERN}" \
    track_pattern="${TRACK_PATTERN}" \
    truth_pattern="${TRUTH_PATTERN}" \
    disp_pattern="${DISP_PATTERN}" \
    env_vars="FORCE=1 ALL_EVENTS=1 TRACKING_ONLY=1" \
    "$@"

echo ""
echo "Jobs submitted! You can monitor them using:"
echo "  condor_q"
