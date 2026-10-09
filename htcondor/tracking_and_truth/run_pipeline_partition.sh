#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Chained Worker Script: Tracking -> MCTruth -> Event Displays
#
# Pipeline per partition:
#   1) Dimuon Tracking:    in.root  -> *_2MuTrks.root
#   2) MCTruth Extraction: *_2MuTrks.root -> *_2MuTrks_truth.root
#   3) Event Displays:     *_2MuTrks_truth.root -> displays_part<p>.root
#
# Usage:
#   run_pipeline_partition.sh <partition_id> [input_pattern] [track_pattern] [truth_pattern] [disp_pattern]
# ==============================================================================

PARTITION="$1"

if [ -z "$PARTITION" ]; then
    echo "ERROR: Missing required partition ID argument." >&2
    echo "Usage: $0 <partition_id> [input_pattern] [track_pattern] [truth_pattern] [disp_pattern]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/afs/cern.ch/work/i/idioniso/snd-numu-charm"
if [ ! -d "$REPO_DIR" ]; then
    REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

# 1. Setup sndsw environment (only if not already loaded in current shell)
if [ -z "${SNDSW_ROOT}" ]; then
    if [ -f "${SCRIPT_DIR}/sndswEnv.sh" ]; then
        echo " ~ [1/6] Sourcing local static environment: ${SCRIPT_DIR}/sndswEnv.sh"
        source "${SCRIPT_DIR}/sndswEnv.sh"
    elif [ -f "${REPO_DIR}/htcondor/tracking_and_truth/sndswEnv.sh" ]; then
        echo " ~ [1/6] Sourcing tracking_and_truth environment: ${REPO_DIR}/htcondor/tracking_and_truth/sndswEnv.sh"
        source "${REPO_DIR}/htcondor/tracking_and_truth/sndswEnv.sh"
    elif [ -f "${REPO_DIR}/htcondor/tracking/sndswEnv.sh" ]; then
        echo " ~ [1/6] Sourcing tracking static environment: ${REPO_DIR}/htcondor/tracking/sndswEnv.sh"
        source "${REPO_DIR}/htcondor/tracking/sndswEnv.sh"
    elif [ -f "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh" ]; then
        echo " ~ [1/6] Sourcing CVMFS stack fallback: /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
        source "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
    fi
else
    echo " ~ [1/6] Using existing environment (SNDSW_ROOT=${SNDSW_ROOT})"
fi

# 2. Configure repository environment paths
echo " ~ [2/6] Repository root: ${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${REPO_DIR}/build/lib:${REPO_DIR}:${LD_LIBRARY_PATH}"

# 3. Path patterns
DEFAULT_INPUT="/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/%s/sndLHC.Genie-TGeant4_digCPP.root"
DEFAULT_TRACK="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_2MuTrks.root"
DEFAULT_TRUTH="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_2MuTrks_truth.root"
DEFAULT_DISP="/eos/user/i/idioniso/snd-numu-charm/event_displays/partitions/displays_part%s.root"

INPUT_PATTERN="${2:-$DEFAULT_INPUT}"
TRACK_PATTERN="${3:-$DEFAULT_TRACK}"
TRUTH_PATTERN="${4:-$DEFAULT_TRUTH}"
DISP_PATTERN="${5:-$DEFAULT_DISP}"
STORE_EVENTS_ARG="${6:-}"

# Determine tracking event storage option
if [ "$STORE_EVENTS_ARG" = "all" ] || [ "$STORE_EVENTS_ARG" = "--all-events" ] || [ "${ALL_EVENTS:-0}" = "1" ] || [ "${STORE_EVENTS:-}" = "all" ]; then
    STORE_EVENTS_FLAG="--all-events"
else
    STORE_EVENTS_FLAG="--two-tracks-only"
fi

# Determine skip flag based on FORCE
SKIP_TRACK_FLAG="--skip-existing"
SKIP_TRUTH_FLAG="--skip-existing"
if [ "${FORCE:-0}" = "1" ]; then
    SKIP_TRACK_FLAG=""
    SKIP_TRUTH_FLAG=""
fi

INPUT_FILE=$(printf "$INPUT_PATTERN" "$PARTITION")
TRACK_FILE=$(printf "$TRACK_PATTERN" "$PARTITION")
TRUTH_FILE=$(printf "$TRUTH_PATTERN" "$PARTITION")
DISP_FILE=$(printf "$DISP_PATTERN" "$PARTITION")

OUTPUT_DIR="$(dirname "$TRACK_FILE")"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "$TRUTH_FILE")"
mkdir -p "$(dirname "$DISP_FILE")"

# Symlink original ROOT files from input partition directory to output directory
INPUT_DIR="$(dirname "$INPUT_FILE")"
if [ -d "$INPUT_DIR" ] && [ "$INPUT_DIR" != "$OUTPUT_DIR" ]; then
    for rf in "$INPUT_DIR"/*.root; do
        if [ -f "$rf" ]; then
            rbname=$(basename "$rf")
            if [ ! -e "${OUTPUT_DIR}/${rbname}" ]; then
                ln -sf "$rf" "${OUTPUT_DIR}/${rbname}" 2>/dev/null || true
            fi
        fi
    done
fi

PYTHON_BIN="$(command -v python3)"
TRACK_SCRIPT="${REPO_DIR}/scripts/run_dimuon_reco.py"
TRUTH_SCRIPT="${REPO_DIR}/scripts/mctruth_neutrinos.py"
DISP_SCRIPT="${REPO_DIR}/scripts/generate_2DEventDisplays.py"
CONFIG_FILE="${REPO_DIR}/config/mctruth_neutrinos_config.yaml"

echo "======================================================================"
echo "SND@LHC Chained Pipeline: Dimuon Tracking -> MCTruth -> Event Displays"
echo "  Date         : $(date)"
echo "  Host         : $(hostname)"
echo "  Partition    : ${PARTITION}"
echo "  Raw Input    : ${INPUT_PATTERN}"
echo "  Track Target : ${TRACK_FILE}"
echo "  Truth Target : ${TRUTH_FILE}"
echo "  Disp Target  : ${DISP_FILE}"
echo "  Storage Mode : ${STORE_EVENTS_FLAG}"
echo "  Force Mode   : ${FORCE:-0}"
echo "  Python       : ${PYTHON_BIN}"
echo "======================================================================"

cd "${REPO_DIR}"

if [ "${DISPLAYS_ONLY:-0}" != "1" ]; then
    # 4. Step 1: Run Dimuon Tracking
    echo " ~ [3/6] Step 1/3: Running dimuon tracking for partition ${PARTITION} (${STORE_EVENTS_FLAG})..."
    set +e
    "${PYTHON_BIN}" "${TRACK_SCRIPT}" \
        -i "${INPUT_PATTERN}" \
        -o "${TRACK_PATTERN}" \
        -p "${PARTITION}" \
        -j 1 \
        -t "dimuon_DS" \
        ${STORE_EVENTS_FLAG} \
        ${SKIP_TRACK_FLAG}
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
    echo " ~ [4/6] Step 2/3: Running MCTruth extraction on tracked dimuon events..."
    "${PYTHON_BIN}" "${TRUTH_SCRIPT}" \
        -i "${TRACK_PATTERN}" \
        -o "${TRUTH_PATTERN}" \
        -p "${PARTITION}" \
        -j 1 \
        -c "${CONFIG_FILE}" \
        ${SKIP_TRUTH_FLAG} \
        --no-symlink-input
    TRUTH_EXIT=$?

    if [ $TRUTH_EXIT -ne 0 ]; then
        echo "ERROR: MCTruth extraction failed for partition ${PARTITION} with exit code ${TRUTH_EXIT}." >&2
        exit $TRUTH_EXIT
    fi
else
    echo " ~ [Displays-Only Mode] Skipping Step 1 (Tracking) and Step 2 (Truth Extraction)."
fi

if [ ! -f "${TRUTH_FILE}" ]; then
    echo "ERROR: Expected truth output file does not exist: ${TRUTH_FILE}" >&2
    exit 4
fi
echo " ~ Truth input verified: ${TRUTH_FILE}"

# 6. Step 3: Run 2D Event Display Generation
echo " ~ [5/6] Step 3/3: Generating 2D Event Displays for partition ${PARTITION}..."
if [ "${FORCE:-0}" != "1" ] && [ "${FORCE_DISPLAYS:-0}" != "1" ] && [ -f "${DISP_FILE}" ] && [ $(stat -c%s "${DISP_FILE}" 2>/dev/null || echo 0) -gt 500 ]; then
    echo " ~ Event display file already exists and is non-empty: ${DISP_FILE} (skipping)"
else
    set +e
    "${PYTHON_BIN}" "${DISP_SCRIPT}" \
        -i "${TRUTH_FILE}" \
        -o "${DISP_FILE}" \
        --mc-truth \
        --muonReco \
        -j 1
    DISP_EXIT=$?
    set -e
    if [ $DISP_EXIT -ne 0 ]; then
        echo "WARNING: Event display generation returned exit code ${DISP_EXIT} for partition ${PARTITION}." >&2
    fi
fi

echo "======================================================================"
echo " ~ [6/6] Pipeline completed successfully for partition ${PARTITION} at $(date)!"
echo "   1) Tracked ROOT file : ${TRACK_FILE}"
echo "   2) Truth ROOT file   : ${TRUTH_FILE}"
echo "   3) Display ROOT file : ${DISP_FILE}"
echo "======================================================================"
exit 0
