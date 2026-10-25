#!/bin/bash
set -e

# ==============================================================================
# HTCondor Submission for 2022 nu14 MCEB Production (1000 partitions, All Events)
#
# Pipeline per partition:
#   Input: /eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_MCEB_dig.root
#   Track: /eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root
#   Truth: /eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root
#   Disps: /eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_MCEB/partitions/displays_part%s.root
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
DISP_PATTERN="/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_MCEB/partitions/displays_part%s.root"
TRUTH_SOURCE_PATTERN="${DATA_DIR}/%s/sndLHC.Genie-TGeant4_dig.root"

mkdir -p "${DATA_DIR}"
mkdir -p "/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_MCEB/partitions"

TOTAL_PARTITIONS=$(wc -l < "${ARGS_FILE}")
echo "=========================================================="
echo " Submitting 2022 nu14 MCEB (All Events, 1000 Partitions) to HTCondor"
echo " Dataset        : sndlhc_15000fb-1_2022_down_nu14_volume_volTarget (MCEB)"
echo " Total Jobs     : ${TOTAL_PARTITIONS} partitions (1..1000)"
echo " Tracking Mode  : All Events (nTracks=0, --all-events)"
echo " Force Mode     : Enabled (FORCE=1)"
echo " Flavour        : longlunch (max 2 hours per job)"
echo " Displays Flags : --mc-truth --recoMuons"
echo " Input Pattern  : ${INPUT_PATTERN}"
echo " Track Target   : ${TRACK_PATTERN}"
echo " Truth Source   : ${TRUTH_SOURCE_PATTERN}"
echo " Truth Target   : ${TRUTH_PATTERN}"
echo " Displays Target: ${DISP_PATTERN}"
echo " Extra Args     : $*"
echo "=========================================================="

condor_submit pipeline_mceb.sub \
    args_file="${ARGS_FILE}" \
    input_pattern="${INPUT_PATTERN}" \
    track_pattern="${TRACK_PATTERN}" \
    truth_pattern="${TRUTH_PATTERN}" \
    disp_pattern="${DISP_PATTERN}" \
    truth_source="${TRUTH_SOURCE_PATTERN}" \
    flavour="longlunch" \
    env_vars="FORCE=1 ALL_EVENTS=1" \
    "$@"

echo ""
echo "Jobs submitted! You can monitor them using:"
echo "  ./htcondor/tracking_and_truth_all_events/status_nu14_2022_MCEB.sh"
