#!/bin/bash
set -e

# ==============================================================================
# HTCondor Submission for newP8 MCEB Production (1000 partitions, All Events)
#
# Pipeline per partition:
#   Input: /eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig.root
#   Track: /eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root
#   Truth: /eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root
#   Disps: /eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions/displays_part%s.root
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
if [ -n "$1" ] && [ -f "$1" ]; then
    ARGS_FILE="$1"
    shift
fi

FLAVOUR="${FLAVOUR:-longlunch}"
DATA_DIR="/eos/user/i/idioniso/snd-numu-charm/data/newP8"
INPUT_PATTERN="${DATA_DIR}/%s/sndLHC.Genie-TGeant4_MCEB_dig.root"
TRACK_PATTERN="${DATA_DIR}/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root"
TRUTH_PATTERN="${DATA_DIR}/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root"
DISP_PATTERN="/eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions/displays_part%s.root"
TRUTH_SOURCE_PATTERN="${DATA_DIR}/%s/sndLHC.Genie-TGeant4_dig.root"

mkdir -p "${DATA_DIR}"
mkdir -p "/eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions"

TOTAL_PARTITIONS=$(wc -l < "${ARGS_FILE}")
echo "=========================================================="
echo " Submitting 2022 nu14 MCEB (All Events, 1000 Partitions) to HTCondor"
echo " Dataset        : newP8 (MCEB)"
echo " Total Jobs     : ${TOTAL_PARTITIONS} partitions (1..1000)"
echo " Tracking Mode  : All Events (nTracks=0, --all-events)"
echo " Force Mode     : Enabled (FORCE=1)"
echo " Flavour        : ${FLAVOUR}"
echo " Displays Flags : --mc-truth --recoMuons"
echo " Input Pattern  : ${INPUT_PATTERN}"
echo " Track Target   : ${TRACK_PATTERN}"
echo " Truth Source   : ${TRUTH_SOURCE_PATTERN}"
echo " Truth Target   : ${TRUTH_PATTERN}"
echo " Displays Target: ${DISP_PATTERN}"
echo " Extra Args     : $*"
echo "=========================================================="

condor_submit pipeline_newP8_mceb.sub \
    args_file="${ARGS_FILE}" \
    input_pattern="${INPUT_PATTERN}" \
    track_pattern="${TRACK_PATTERN}" \
    truth_pattern="${TRUTH_PATTERN}" \
    disp_pattern="${DISP_PATTERN}" \
    truth_source="${TRUTH_SOURCE_PATTERN}" \
    flavour="${FLAVOUR}" \
    env_vars="FORCE=1 ALL_EVENTS=1 SND_CHANNELS_READONLY=1 SKIP_TRACK=1" \
    "$@"

echo ""
echo "Jobs submitted! You can monitor them using condor_q."
