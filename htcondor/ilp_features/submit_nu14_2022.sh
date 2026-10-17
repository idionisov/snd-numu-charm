#!/bin/bash
set -e

# ==============================================================================
# HTCondor Submission for ILP Track Reconstruction (1000 Partitions, workday)
#
# Dataset:
#   Input/Target: /eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root
#   Injected TTree: "ilp" (purely data-compatible observables)
#   Flavour: workday (8 hours)
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
INPUT_PATTERN="${DATA_DIR}/%s/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root"

TOTAL_PARTITIONS=$(wc -l < "${ARGS_FILE}")
FLAVOUR="workday"

echo "=========================================================="
echo " Submitting ILP Track Reconstruction (1000 Jobs, workday)"
echo " Dataset        : sndlhc_15000fb-1_2022_down/nu14/volume_volTarget"
echo " Total Jobs     : ${TOTAL_PARTITIONS} partitions (1..1000)"
echo " Injected Tree  : ilp (Data-level, Zero Truth)"
echo " Flavour        : ${FLAVOUR} (up to 8 hours per job)"
echo " Input Pattern  : ${INPUT_PATTERN}"
echo " Update Target  : [In-Place File Update]"
echo " Extra Args     : $*"
echo "=========================================================="

condor_submit ilp.sub \
    args_file="${ARGS_FILE}" \
    input_pattern="${INPUT_PATTERN}" \
    flavour="${FLAVOUR}" \
    "$@"

echo ""
echo "Jobs submitted! You can monitor them using:"
echo "  ./htcondor/ilp_features/status_nu14_2022.sh"
