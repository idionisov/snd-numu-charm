#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Chained Worker Script: Tracking -> MCTruth Extraction
#
# Pipeline per partition:
#   1) Dimuon Tracking:    in.root  -> *_2MuTrks.root
#   2) MCTruth Extraction: *_2MuTrks.root -> *_2MuTrks_truth.root
#
# Usage:
#   run_pipeline_partition.sh <partition_id> [input_pattern] [track_pattern] [truth_pattern]
# ==============================================================================

PARTITION="$1"

if [ -z "$PARTITION" ]; then
    echo "ERROR: Missing required partition ID argument." >&2
    echo "Usage: $0 <partition_id> [input_pattern] [track_pattern] [truth_pattern]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/afs/cern.ch/work/i/idioniso/snd-numu-charm"
if [ ! -d "$REPO_DIR" ]; then
    REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

# 1. Setup sndsw environment
if [ -f "${SCRIPT_DIR}/sndswEnv.sh" ]; then
    echo " ~ [1/5] Sourcing local static environment: ${SCRIPT_DIR}/sndswEnv.sh"
    source "${SCRIPT_DIR}/sndswEnv.sh"
elif [ -f "${REPO_DIR}/htcondor/tracking_and_truth/sndswEnv.sh" ]; then
    echo " ~ [1/5] Sourcing tracking_and_truth environment: ${REPO_DIR}/htcondor/tracking_and_truth/sndswEnv.sh"
    source "${REPO_DIR}/htcondor/tracking_and_truth/sndswEnv.sh"
elif [ -f "${REPO_DIR}/htcondor/tracking/sndswEnv.sh" ]; then
    echo " ~ [1/5] Sourcing tracking static environment: ${REPO_DIR}/htcondor/tracking/sndswEnv.sh"
    source "${REPO_DIR}/htcondor/tracking/sndswEnv.sh"
elif [ -f "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh" ]; then
    echo " ~ [1/5] Sourcing CVMFS stack fallback: /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
    source "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
fi

# 2. Configure repository environment paths
echo " ~ [2/5] Repository root: ${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${REPO_DIR}/build/lib:${REPO_DIR}:${LD_LIBRARY_PATH}"

# 3. Path patterns
DEFAULT_INPUT="/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/%s/sndLHC.Genie-TGeant4_digCPP.root"
DEFAULT_TRACK="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_2MuTrks.root"
DEFAULT_TRUTH="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_2MuTrks_truth.root"

INPUT_PATTERN="${2:-$DEFAULT_INPUT}"
TRACK_PATTERN="${3:-$DEFAULT_TRACK}"
TRUTH_PATTERN="${4:-$DEFAULT_TRUTH}"

TRACK_FILE=$(printf "$TRACK_PATTERN" "$PARTITION")
TRUTH_FILE=$(printf "$TRUTH_PATTERN" "$PARTITION")

mkdir -p "$(dirname "$TRACK_FILE")"
mkdir -p "$(dirname "$TRUTH_FILE")"

PYTHON_BIN="$(command -v python3)"
TRACK_SCRIPT="${REPO_DIR}/scripts/run_dimuon_reco.py"
TRUTH_SCRIPT="${REPO_DIR}/scripts/mctruth_neutrinos.py"
CONFIG_FILE="${REPO_DIR}/config/mctruth_neutrinos_config.yaml"

echo "======================================================================"
echo "SND@LHC Chained Pipeline: Dimuon Tracking -> MCTruth Extraction"
echo "  Date         : $(date)"
echo "  Host         : $(hostname)"
echo "  Partition    : ${PARTITION}"
echo "  Raw Input    : ${INPUT_PATTERN}"
echo "  Track Target : ${TRACK_FILE}"
echo "  Truth Target : ${TRUTH_FILE}"
echo "  Python       : ${PYTHON_BIN}"
echo "======================================================================"

# 4. Step 1: Run Dimuon Tracking
echo " ~ [3/5] Step 1/2: Running dimuon tracking for partition ${PARTITION}..."
cd "${REPO_DIR}"

set +e
"${PYTHON_BIN}" "${TRACK_SCRIPT}" \
    -i "${INPUT_PATTERN}" \
    -o "${TRACK_PATTERN}" \
    -p "${PARTITION}" \
    -j 1 \
    -t "dimuon_DS" \
    --nTracks 2 \
    --skip-existing
TRACK_EXIT=$?
set -e

# 0 or 143 (SIGTERM atexit handler in FairRoot) is considered normal success
if [ $TRACK_EXIT -ne 0 ] && [ $TRACK_EXIT -ne 143 ]; then
    echo "ERROR: Dimuon tracking failed for partition ${PARTITION} with exit code ${TRACK_EXIT}." >&2
    exit $TRACK_EXIT
fi

if [ ! -f "${TRACK_FILE}" ]; then
    echo "ERROR: Expected tracking output file does not exist: ${TRACK_FILE}" >&2
    exit 3
fi
echo " ~ Tracking output verified: ${TRACK_FILE}"

# 5. Step 2: Run MCTruth Extraction on tracked dimuon events
echo " ~ [4/5] Step 2/2: Running MCTruth extraction on tracked dimuon events..."
"${PYTHON_BIN}" "${TRUTH_SCRIPT}" \
    -i "${TRACK_PATTERN}" \
    -o "${TRUTH_PATTERN}" \
    -p "${PARTITION}" \
    -j 1 \
    -c "${CONFIG_FILE}" \
    --skip-existing \
    --no-symlink-input
TRUTH_EXIT=$?

if [ $TRUTH_EXIT -ne 0 ]; then
    echo "ERROR: MCTruth extraction failed for partition ${PARTITION} with exit code ${TRUTH_EXIT}." >&2
    exit $TRUTH_EXIT
fi

echo "======================================================================"
echo " ~ [5/5] Pipeline completed successfully for partition ${PARTITION} at $(date)!"
echo "   1) Tracked ROOT file : ${TRACK_FILE}"
echo "   2) Truth ROOT file   : ${TRUTH_FILE}"
echo "======================================================================"
exit 0
