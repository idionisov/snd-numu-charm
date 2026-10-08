#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Partition Worker Script for Dimuon Tracking
#
# Usage:
#   run_tracking_partition.sh <partition_id> [input_pattern] [output_pattern] [extra_flags...]
# Examples:
#   ./run_tracking_partition.sh 0
#   ./run_tracking_partition.sh 42 "" "" --n-events 100
# ==============================================================================

PARTITION="$1"

if [ -z "$PARTITION" ]; then
    echo "ERROR: Missing required partition ID argument." >&2
    echo "Usage: $0 <partition_id> [input_pattern] [output_pattern] [extra_flags...]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/afs/cern.ch/work/i/idioniso/snd-numu-charm"
if [ ! -d "$REPO_DIR" ]; then
    REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

# 1. Setup sndsw environment
if [ -f "${SCRIPT_DIR}/sndswEnv.sh" ]; then
    echo " ~ [1/4] Sourcing local static environment: ${SCRIPT_DIR}/sndswEnv.sh"
    source "${SCRIPT_DIR}/sndswEnv.sh"
elif [ -f "${REPO_DIR}/htcondor/tracking/sndswEnv.sh" ]; then
    echo " ~ [1/4] Sourcing tracking static environment: ${REPO_DIR}/htcondor/tracking/sndswEnv.sh"
    source "${REPO_DIR}/htcondor/tracking/sndswEnv.sh"
elif [ -f "${REPO_DIR}/htcondor/mctruth_neutrinos/sndswEnv.sh" ]; then
    echo " ~ [1/4] Sourcing mctruth static environment: ${REPO_DIR}/htcondor/mctruth_neutrinos/sndswEnv.sh"
    source "${REPO_DIR}/htcondor/mctruth_neutrinos/sndswEnv.sh"
elif [ -f "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh" ]; then
    echo " ~ [1/4] Sourcing CVMFS stack fallback: /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
    source "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
fi

# 2. Configure repository environment paths
echo " ~ [2/4] Repository root: ${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${REPO_DIR}/build/lib:${REPO_DIR}:${LD_LIBRARY_PATH}"

# 3. Default input and output patterns
# Fallback hierarchy: if symlink in /eos/user/... exists, use it; otherwise experiment MonteCarlo directory
DEFAULT_INPUT="/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/%s/sndLHC.Genie-TGeant4_digCPP.root"
DEFAULT_OUTPUT="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_dimuonReco.root"

INPUT_PATTERN="${2:-$DEFAULT_INPUT}"
OUTPUT_PATTERN="${3:-$DEFAULT_OUTPUT}"

shift 3 2>/dev/null || shift $#
EXTRA_FLAGS="$*"
if [ -z "$EXTRA_FLAGS" ]; then
    EXTRA_FLAGS="--skip-existing"
fi

# Ensure output partition directory exists
OUTPUT_FILE=$(printf "$OUTPUT_PATTERN" "$PARTITION")
OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="$(command -v python3)"
SCRIPT_PATH="${REPO_DIR}/scripts/run_dimuon_reco.py"

echo "======================================================================"
echo "SND@LHC HTCondor Dimuon Tracking Job"
echo "  Date           : $(date)"
echo "  Host           : $(hostname)"
echo "  Partition      : ${PARTITION}"
echo "  Script         : ${SCRIPT_PATH}"
echo "  Input Pattern  : ${INPUT_PATTERN}"
echo "  Output Pattern : ${OUTPUT_PATTERN}"
echo "  Output Target  : ${OUTPUT_FILE}"
echo "  Extra Flags    : ${EXTRA_FLAGS}"
echo "  Python         : ${PYTHON_BIN}"
echo "======================================================================"

# 4. Execute dimuon tracking for this partition
echo " ~ [3/4] Executing run_dimuon_reco.py..."
cd "${REPO_DIR}"

set +e
"${PYTHON_BIN}" "${SCRIPT_PATH}" \
    -i "${INPUT_PATTERN}" \
    -o "${OUTPUT_PATTERN}" \
    -p "${PARTITION}" \
    -j 1 \
    -t "dimuon_DS" \
    --nTracks 2 \
    ${EXTRA_FLAGS}
EXIT_CODE=$?
set -e

# run_TrackSelections.py registers an atexit handler that sends SIGTERM (143) to avoid teardown freezes in ROOT/FairRoot
if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 143 ]; then
    echo " ~ [4/4] Partition ${PARTITION} dimuon tracking completed successfully at $(date)."
    exit 0
else
    echo "ERROR: Partition ${PARTITION} dimuon tracking failed with exit code ${EXIT_CODE} at $(date)." >&2
    exit $EXIT_CODE
fi
