#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Worker Script for Event Prefiltering (Single Partition)
#
# Usage:
#   run_prefilter_partition.sh <partition_id> [input_pattern] [output_pattern] [cutset]
# ==============================================================================

PARTITION="$1"

if [ -z "$PARTITION" ]; then
    echo "ERROR: Missing required partition ID argument." >&2
    echo "Usage: $0 <partition_id> [input_pattern] [output_pattern] [cutset]" >&2
    exit 1
fi

DEFAULT_INPUT="/eos/experiment/sndlhc/convertedData/physics/2023/run_006596/sndsw_raw-%s.root"
DEFAULT_OUTPUT="/eos/user/i/idioniso/snd-numu-charm/prefiltered/2023/b2nob1Muons/run006596/sndsw_raw-%s.root"
DEFAULT_CUTSET="b2nob1Muons"

sanitize_pattern() {
    local val="$1"
    local fallback="$2"
    if [ -z "$val" ] || [[ "$val" == *'$('* ]]; then
        echo "$fallback"
    else
        echo "$val"
    fi
}

INPUT_PATTERN=$(sanitize_pattern "$2" "$DEFAULT_INPUT")
OUTPUT_PATTERN=$(sanitize_pattern "$3" "$DEFAULT_OUTPUT")
CUTSET="${4:-$DEFAULT_CUTSET}"
if [[ "$CUTSET" == *'$('* ]]; then
    CUTSET="$DEFAULT_CUTSET"
fi

# Resolve partition string (support both padded 0000 and unpadded 0)
PART_STR="$PARTITION"
INPUT_FILE=$(printf "$INPUT_PATTERN" "$PART_STR")
if [ ! -f "$INPUT_FILE" ] && [[ "$PARTITION" =~ ^[0-9]+$ ]]; then
    PADDED=$(printf "%04d" "$((10#$PARTITION))")
    CAND_FILE=$(printf "$INPUT_PATTERN" "$PADDED")
    if [ -f "$CAND_FILE" ]; then
        PART_STR="$PADDED"
        INPUT_FILE="$CAND_FILE"
    fi
fi
OUTPUT_FILE=$(printf "$OUTPUT_PATTERN" "$PART_STR")

SIG_INPUT_FILE=""
if [ -n "${SIG_INPUT_PATTERN:-}" ] && [[ "${SIG_INPUT_PATTERN}" != *'$('* ]]; then
    CAND_SIG=$(printf "$SIG_INPUT_PATTERN" "$PART_STR")
    if [ -f "$CAND_SIG" ]; then
        SIG_INPUT_FILE="$CAND_SIG"
    fi
fi

IND_INPUT_FILE=""
if [ -n "${IND_INPUT_PATTERN:-}" ] && [[ "${IND_INPUT_PATTERN}" != *'$('* ]]; then
    CAND_IND=$(printf "$IND_INPUT_PATTERN" "$PART_STR")
    if [ -f "$CAND_IND" ]; then
        IND_INPUT_FILE="$CAND_IND"
    fi
fi

SIG_OUTPUT_FILE=""
if [ -n "$SIG_INPUT_FILE" ] && [ -n "${SIG_OUTPUT_PATTERN:-}" ] && [[ "${SIG_OUTPUT_PATTERN}" != *'$('* ]]; then
    SIG_OUTPUT_FILE=$(printf "$SIG_OUTPUT_PATTERN" "$PART_STR")
fi

IND_OUTPUT_FILE=""
if [ -n "$IND_INPUT_FILE" ] && [ -n "${IND_OUTPUT_PATTERN:-}" ] && [[ "${IND_OUTPUT_PATTERN}" != *'$('* ]]; then
    IND_OUTPUT_FILE=$(printf "$IND_OUTPUT_PATTERN" "$PART_STR")
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"
[ -n "$SIG_OUTPUT_FILE" ] && mkdir -p "$(dirname "$SIG_OUTPUT_FILE")"
[ -n "$IND_OUTPUT_FILE" ] && mkdir -p "$(dirname "$IND_OUTPUT_FILE")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/afs/cern.ch/work/i/idioniso/snd-numu-charm"
if [ ! -d "$REPO_DIR" ]; then
    REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

# 1. Source environment
if [ -f "${SCRIPT_DIR}/sndswEnv.sh" ]; then
    echo " ~ [1/3] Sourcing local environment: ${SCRIPT_DIR}/sndswEnv.sh"
    source "${SCRIPT_DIR}/sndswEnv.sh"
elif [ -f "${REPO_DIR}/htcondor/simP8/sndswEnv.sh" ]; then
    echo " ~ [1/3] Sourcing repository environment: ${REPO_DIR}/htcondor/simP8/sndswEnv.sh"
    source "${REPO_DIR}/htcondor/simP8/sndswEnv.sh"
elif [ -f "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh" ]; then
    echo " ~ [1/3] Sourcing CVMFS stack fallback: /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
    source "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
fi

if [ -d "${SCRIPT_DIR}/ddfUtils" ]; then
    export ddfUtils="${SCRIPT_DIR}/ddfUtils"
else
    export ddfUtils="${DDFUTILS:-/afs/cern.ch/user/i/idioniso/ddfUtils}"
fi

export PYTHONPATH="${SCRIPT_DIR}:${REPO_DIR}:${SNDSW_SRC}/python:${SNDSW_SRC}/shipLHC/scripts:${SNDSW_ROOT}/python:${ddfUtils}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${REPO_DIR}/build/lib:${SNDSW_ROOT}/lib:${LD_LIBRARY_PATH}"

PYTHON_BIN="$(command -v python3)"
PREFILTER_SCRIPT="${REPO_DIR}/scripts/prefilter.py"

echo "======================================================================"
echo "SND@LHC HTCondor Prefiltering (${CUTSET})"
echo "  Date          : $(date)"
echo "  Host          : $(hostname)"
echo "  Partition     : ${PART_STR}"
echo "  Input File    : ${INPUT_FILE}"
echo "  Output File   : ${OUTPUT_FILE}"
if [ -n "$SIG_INPUT_FILE" ]; then
    echo "  Signal Input  : ${SIG_INPUT_FILE}"
    echo "  Signal Output : ${SIG_OUTPUT_FILE}"
fi
if [ -n "$IND_INPUT_FILE" ]; then
    echo "  Indirect Input: ${IND_INPUT_FILE}"
    echo "  Indirect Out  : ${IND_OUTPUT_FILE}"
fi
echo "  Cutset        : ${CUTSET}"
echo "  Force Mode    : ${FORCE:-0}"
echo "======================================================================"

# 2. Check skip-existing
if [ "${FORCE:-0}" != "1" ] && [ -f "${OUTPUT_FILE}" ] && [ -s "${OUTPUT_FILE}" ]; then
    OUT_SIZE=$(stat -c%s "${OUTPUT_FILE}" 2>/dev/null || echo 0)
    if [ "${OUT_SIZE}" -gt 500 ]; then
        echo " ~ [INFO] Target ${OUTPUT_FILE} already exists (${OUT_SIZE} B). Skipping (set FORCE=1 to rerun)."
        exit 0
    fi
fi

# 3. Check input files
if [ ! -f "${INPUT_FILE}" ] || [ ! -s "${INPUT_FILE}" ]; then
    echo "ERROR: Input file not found or empty: ${INPUT_FILE}" >&2
    exit 2
fi

cd "${REPO_DIR}"

CMD_ARGS=(
    -i "${INPUT_FILE}"
    -o "${OUTPUT_FILE}"
    --cutset "${CUTSET}"
    --no-symlinks
    -j 1
    -w 1
)

if [ -n "${SIG_INPUT_FILE}" ]; then
    CMD_ARGS+=(--signal-input "${SIG_INPUT_FILE}")
    if [ -n "${SIG_OUTPUT_FILE}" ]; then
        CMD_ARGS+=(--signal-output "${SIG_OUTPUT_FILE}")
    fi
fi

if [ -n "${IND_INPUT_FILE}" ]; then
    CMD_ARGS+=(--indirect-input "${IND_INPUT_FILE}")
    if [ -n "${IND_OUTPUT_FILE}" ]; then
        CMD_ARGS+=(--indirect-output "${IND_OUTPUT_FILE}")
    fi
fi

if [ -n "${CUTFLOW_DIR:-}" ]; then
    mkdir -p "${CUTFLOW_DIR}"
    CMD_ARGS+=(--cutflow "${CUTFLOW_DIR}/cutflow_part${PART_STR}.root")
fi

if [ -n "${CSV_DIR:-}" ]; then
    mkdir -p "${CSV_DIR}"
    CMD_ARGS+=(--failed-cuts-csv "${CSV_DIR}/failed_cuts_part${PART_STR}.txt")
fi

if [ "${FORCE:-0}" != "1" ]; then
    CMD_ARGS+=(--skip-existing)
fi

echo " ~ [2/3] Executing prefilter.py for partition ${PART_STR}..."
"${PYTHON_BIN}" "${PREFILTER_SCRIPT}" "${CMD_ARGS[@]}"

PREFILTER_EXIT=$?
if [ $PREFILTER_EXIT -ne 0 ]; then
    echo "ERROR: Prefilter failed for partition ${PART_STR} with exit code ${PREFILTER_EXIT}." >&2
    exit $PREFILTER_EXIT
fi

# 4. Verify output file
if [ ! -f "${OUTPUT_FILE}" ] || [ ! -s "${OUTPUT_FILE}" ]; then
    echo "ERROR: Expected output file does not exist or is empty: ${OUTPUT_FILE}" >&2
    exit 3
fi

MAIN_SIZE=$(stat -c%s "${OUTPUT_FILE}" 2>/dev/null || echo 0)
echo "======================================================================"
echo " ~ [3/3] Prefilter completed successfully for partition ${PART_STR} at $(date)!"
echo "   Output file: ${OUTPUT_FILE} (${MAIN_SIZE} bytes)"
echo "======================================================================"
exit 0
