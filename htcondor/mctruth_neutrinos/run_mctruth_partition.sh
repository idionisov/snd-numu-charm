#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Partition Worker Script for MCTruth Neutrino Pipeline
#
# Usage:
#   run_mctruth_partition.sh <partition_id> [extra_flags...]
# Examples:
#   ./run_mctruth_partition.sh 0
#   ./run_mctruth_partition.sh 42 --entries 100
# ==============================================================================

PARTITION="$1"

if [ -z "$PARTITION" ]; then
    echo "ERROR: Missing required partition ID argument." >&2
    echo "Usage: $0 <partition_id> [extra_flags...]" >&2
    exit 1
fi

shift || true
EXTRA_FLAGS="$*"
if [ -z "$EXTRA_FLAGS" ]; then
    EXTRA_FLAGS="--skip-existing"
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
elif [ -f "${REPO_DIR}/htcondor/mctruth_neutrinos/sndswEnv.sh" ]; then
    echo " ~ [1/4] Sourcing repo static environment: ${REPO_DIR}/htcondor/mctruth_neutrinos/sndswEnv.sh"
    source "${REPO_DIR}/htcondor/mctruth_neutrinos/sndswEnv.sh"
elif [ -f "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh" ]; then
    echo " ~ [1/4] Sourcing CVMFS stack fallback: /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
    source "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
fi

# 2. Configure repository environment paths
echo " ~ [2/4] Repository root: ${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${REPO_DIR}/build/lib:${REPO_DIR}:${LD_LIBRARY_PATH}"

# 3. Output directory preparation on EOS
OUTPUT_DIR="/eos/user/i/idioniso/snd-numu-charm/data/${PARTITION}"
mkdir -p "${OUTPUT_DIR}"

PYTHON_BIN="$(command -v python3)"
SCRIPT_PATH="${REPO_DIR}/scripts/mctruth_neutrinos.py"
CONFIG_PATH="${REPO_DIR}/config/mctruth_neutrinos_config.yaml"

echo "======================================================================"
echo "SND@LHC MCTruth Neutrino Job"
echo "  Date       : $(date)"
echo "  Host       : $(hostname)"
echo "  Partition  : ${PARTITION}"
echo "  Script     : ${SCRIPT_PATH}"
echo "  Config     : ${CONFIG_PATH}"
echo "  Output Dir : ${OUTPUT_DIR}"
echo "  Flags      : ${EXTRA_FLAGS}"
echo "  Python     : ${PYTHON_BIN}"
echo "======================================================================"

# 4. Execute mctruth processing for this partition
echo " ~ [3/4] Executing mctruth_neutrinos.py..."
cd "${REPO_DIR}"

"${PYTHON_BIN}" "${SCRIPT_PATH}" \
    -p "${PARTITION}" \
    -j 1 \
    -c "${CONFIG_PATH}" \
    ${EXTRA_FLAGS}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo " ~ [4/4] Partition ${PARTITION} processed successfully at $(date)."
else
    echo "ERROR: Partition ${PARTITION} failed with exit code ${EXIT_CODE} at $(date)." >&2
fi

exit $EXIT_CODE
