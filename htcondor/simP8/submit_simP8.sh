#!/bin/bash
set -e

# ==============================================================================
# HTCondor Submission for 2022 nu14 Simulation + MCEB + Digitization (Pythia8)
#
# Pipeline per partition:
#   Input : /eos/experiment/sndlhc/.../nu14/volume_volTarget/%s/*.gst.root
#   Output: /eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/
#     - geofile_full.Genie-TGeant4.root
#     - sndLHC.Genie-TGeant4.root
#     - sndLHC.Genie-TGeant4_MCEB.root
#     - sndLHC.Genie-TGeant4_dig.root
#     - sndLHC.Genie-TGeant4_MCEB_dig.root
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
TMP_ARGS=""

if [ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+-[0-9]+$ ]]; then
    START=$(echo "$1" | cut -d'-' -f1)
    END=$(echo "$1" | cut -d'-' -f2)
    shift
    TMP_ARGS=$(mktemp /tmp/simP8_args_XXXXXX.txt)
    seq "${START}" "${END}" > "${TMP_ARGS}"
    ARGS_FILE="${TMP_ARGS}"
elif [ $# -ge 2 ] && [[ "$1" =~ ^[0-9]+$ ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
    START="$1"
    END="$2"
    shift 2
    TMP_ARGS=$(mktemp /tmp/simP8_args_XXXXXX.txt)
    seq "${START}" "${END}" > "${TMP_ARGS}"
    ARGS_FILE="${TMP_ARGS}"
elif [ "${1:-}" == "-p" ] && [ -n "${2:-}" ]; then
    PART_ARG="$2"
    shift 2
    TMP_ARGS=$(mktemp /tmp/simP8_args_XXXXXX.txt)
    if [[ "${PART_ARG}" =~ ^[0-9]+-[0-9]+$ ]]; then
        START=$(echo "${PART_ARG}" | cut -d'-' -f1)
        END=$(echo "${PART_ARG}" | cut -d'-' -f2)
        seq "${START}" "${END}" > "${TMP_ARGS}"
    else
        echo "${PART_ARG}" | tr ',' '\n' > "${TMP_ARGS}"
    fi
    ARGS_FILE="${TMP_ARGS}"
fi

trap '[ -n "${TMP_ARGS}" ] && rm -f "${TMP_ARGS}"' EXIT
SRC_PATTERN="/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2mmRangeCut/sndlhc_15000fb-1_2022_down/nu14/volume_volTarget/%s"
DEST_PATTERN="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s"

TOTAL_PARTITIONS=$(wc -l < "${ARGS_FILE}")
echo "=========================================================="
echo " Submitting 2022 nu14 Simulation (Pythia8, 1000 Partitions) to HTCondor"
echo " Total Jobs     : ${TOTAL_PARTITIONS} partitions (1..1000)"
echo " Pipeline       : Geant4 Sim (Genie 4) -> MCEB -> Digi (raw + MCEB)"
echo " Decayer Macro  : gconfig/DecayConfigPy8.C (local branch)"
echo " Flavour        : workday (max 8 hours per job)"
echo " Source Pattern : ${SRC_PATTERN}"
echo " Dest Pattern   : ${DEST_PATTERN}"
echo " Extra Args     : $*"
echo "=========================================================="

condor_submit sim_pipeline.sub \
    args_file="${ARGS_FILE}" \
    src_pattern="${SRC_PATTERN}" \
    dest_pattern="${DEST_PATTERN}" \
    flavour="workday" \
    "$@"

echo ""
echo "Jobs submitted! You can monitor them using:"
echo "  ./htcondor/simP8/status_simP8.sh"
