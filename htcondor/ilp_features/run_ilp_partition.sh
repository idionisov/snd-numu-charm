#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Worker Script for ILP Feature Extraction & Tree Injection
#
# Processes a single partition:
#   1) Injects data-compatible "ilp" TTree into:
#      .../<partition>/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root
#   2) Preserves all original trees (cbmsim, BranchList, etc.) 1:1
#   3) Auto-detects partition-local detector geometry in the partition directory
#
# Usage:
#   run_ilp_partition.sh <partition_id> [input_pattern]
# ==============================================================================

PARTITION="$1"

if [ -z "$PARTITION" ]; then
    echo "ERROR: Missing required partition ID argument." >&2
    echo "Usage: $0 <partition_id> [input_pattern]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/afs/cern.ch/work/i/idioniso/snd-numu-charm"
if [ ! -d "$REPO_DIR" ]; then
    REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

# 1. Source environment
if [ -f "${SCRIPT_DIR}/sndswEnv.sh" ]; then
    echo " ~ [1/4] Sourcing local environment: ${SCRIPT_DIR}/sndswEnv.sh"
    source "${SCRIPT_DIR}/sndswEnv.sh"
elif [ -f "${REPO_DIR}/htcondor/ilp_features/sndswEnv.sh" ]; then
    echo " ~ [1/4] Sourcing repo environment: ${REPO_DIR}/htcondor/ilp_features/sndswEnv.sh"
    source "${REPO_DIR}/htcondor/ilp_features/sndswEnv.sh"
elif [ -f "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh" ]; then
    echo " ~ [1/4] Sourcing CVMFS stack fallback: /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
    source "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
fi

if [ -d "${SCRIPT_DIR}/ddfUtils" ]; then
    export ddfUtils="${SCRIPT_DIR}/ddfUtils"
else
    export ddfUtils="${DDFUTILS:-/afs/cern.ch/user/i/idioniso/ddfUtils}"
fi

echo " ~ [2/4] Environment configured"
echo "         Repository root : ${REPO_DIR}"
echo "         ddfUtils path   : ${ddfUtils}"
export PYTHONPATH="${SCRIPT_DIR}:${REPO_DIR}:${SNDSW_SRC}/python:${SNDSW_SRC}/shipLHC/scripts:${SNDSW_ROOT}/python:${ddfUtils}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO_DIR}/build/lib:${SNDSW_ROOT}/lib:${LD_LIBRARY_PATH:-}"

# 2. Resolve input file path
DEFAULT_INPUT="/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root"
INPUT_PATTERN="${2:-$DEFAULT_INPUT}"

if [[ "$INPUT_PATTERN" == *"%s"* ]] || [[ "$INPUT_PATTERN" == *"%d"* ]]; then
    # shellcheck disable=SC2059
    INPUT_FILE=$(printf "$INPUT_PATTERN" "$PARTITION")
else
    INPUT_FILE="$INPUT_PATTERN"
fi

echo " ~ [3/4] Validating input file for partition ${PARTITION}"
echo "         Input File: ${INPUT_FILE}"

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file does not exist: ${INPUT_FILE}" >&2
    exit 2
fi

# 3. Execute ILP Track Reconstruction & Tree Injection
echo " ~ [4/4] Executing extract_ilp_features.py (in-place update)"
T0=$(date +%s)

python3 -X faulthandler "${REPO_DIR}/scripts/extract_ilp_features.py" \
    -i "${INPUT_FILE}" \
    -w 1

T1=$(date +%s)
ELAPSED=$((T1 - T0))
echo " ~ Completed ILP feature extraction for partition ${PARTITION} in ${ELAPSED}s"

# Quick verification of ilp tree presence
if command -v rootls >/dev/null 2>&1; then
    echo " ~ Verification keys for ${INPUT_FILE}:"
    rootls "${INPUT_FILE}"
fi

echo "=== Partition ${PARTITION} successfully processed ==="
exit 0
