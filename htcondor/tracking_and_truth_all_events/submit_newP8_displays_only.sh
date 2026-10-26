#!/bin/bash
set -e

# ==============================================================================
# HTCondor Submission for newP8 MCEB Displays Only (1000 partitions)
# Regenerates 2D event displays using unpruned MCTrack truth coordinates.
#
# Direct Skim   : /eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions/direct/displays_part%s.root
# Indirect Skim : /eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions/indirect/displays_part%s.root
# Truth Source  : /eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_dig.root
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

FLAVOUR="${FLAVOUR:-microcentury}"
DATA_DIR="/eos/user/i/idioniso/snd-numu-charm/data/newP8"
INPUT_PATTERN="${DATA_DIR}/%s/sndLHC.Genie-TGeant4_MCEB_dig.root"
TRACK_PATTERN="${DATA_DIR}/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root"
TRUTH_PATTERN="${DATA_DIR}/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root"
DISP_PATTERN="/eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions/displays_part%s.root"
TRUTH_SOURCE_PATTERN="${DATA_DIR}/%s/sndLHC.Genie-TGeant4_dig.root"

mkdir -p "/eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions/direct"
mkdir -p "/eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions/indirect"

TOTAL_PARTITIONS=$(wc -l < "${ARGS_FILE}")
echo "=========================================================="
echo " Submitting newP8 MCEB Displays Only to HTCondor"
echo " Dataset        : newP8 (MCEB)"
echo " Total Jobs     : ${TOTAL_PARTITIONS} partitions"
echo " Mode           : Displays Only (DISPLAYS_ONLY=1 FORCE_DISPLAYS=1)"
echo " Flavour        : ${FLAVOUR}"
echo " Truth Source   : ${TRUTH_SOURCE_PATTERN}"
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
    env_vars="DISPLAYS_ONLY=1 FORCE_DISPLAYS=1 ALL_EVENTS=1 SND_CHANNELS_READONLY=1" \
    "$@"

echo ""
echo "Jobs submitted! You can monitor them using: condor_q"
