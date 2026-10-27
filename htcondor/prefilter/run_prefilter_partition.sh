#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Worker Script for Stage 6: Event Prefiltering (Essentials)
#
# Processes per partition:
#   Inputs : sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root
#            sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonDirect.root
#            sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonIndirect.root
#   Outputs: prefiltered/newP8/essentials/<p>/...
#            prefiltered/newP8/essentials-direct/<p>/...
#            prefiltered/newP8/essentials-indirect/<p>/...
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

DEFAULT_INPUT="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root"
DEFAULT_SIG_IN="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonDirect.root"
DEFAULT_IND_IN="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonIndirect.root"

DEFAULT_OUTPUT="/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8/essentials/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root"
DEFAULT_SIG_OUT="/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8/essentials-direct/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root"
DEFAULT_IND_OUT="/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8/essentials-indirect/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root"

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
CUTSET="${4:-essentials}"
if [[ "$CUTSET" == *'$('* ]]; then
    CUTSET="essentials"
fi

SIG_INPUT_PATTERN=$(sanitize_pattern "${SIG_INPUT_PATTERN}" "$DEFAULT_SIG_IN")
IND_INPUT_PATTERN=$(sanitize_pattern "${IND_INPUT_PATTERN}" "$DEFAULT_IND_IN")

SIG_OUTPUT_PATTERN=$(sanitize_pattern "${SIG_OUTPUT_PATTERN}" "$DEFAULT_SIG_OUT")
IND_OUTPUT_PATTERN=$(sanitize_pattern "${IND_OUTPUT_PATTERN}" "$DEFAULT_IND_OUT")

NU_SCALING="${NU_SCALING:-0.004533333333333334}"
PMU_SCALING="${PMU_SCALING:-68.6}"

INPUT_FILE=$(printf "$INPUT_PATTERN" "$PARTITION")
SIG_INPUT_FILE=$(printf "$SIG_INPUT_PATTERN" "$PARTITION")
IND_INPUT_FILE=$(printf "$IND_INPUT_PATTERN" "$PARTITION")

OUTPUT_FILE=$(printf "$OUTPUT_PATTERN" "$PARTITION")
SIG_OUTPUT_FILE=$(printf "$SIG_OUTPUT_PATTERN" "$PARTITION")
IND_OUTPUT_FILE=$(printf "$IND_OUTPUT_PATTERN" "$PARTITION")

mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p "$(dirname "$SIG_OUTPUT_FILE")"
mkdir -p "$(dirname "$IND_OUTPUT_FILE")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/afs/cern.ch/work/i/idioniso/snd-numu-charm"
if [ ! -d "$REPO_DIR" ]; then
    REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

# 1. Source environment
if [ -f "${SCRIPT_DIR}/sndswEnv.sh" ]; then
    echo " ~ [1/4] Sourcing local environment: ${SCRIPT_DIR}/sndswEnv.sh"
    source "${SCRIPT_DIR}/sndswEnv.sh"
elif [ -f "${REPO_DIR}/htcondor/simP8/sndswEnv.sh" ]; then
    echo " ~ [1/4] Sourcing repository environment: ${REPO_DIR}/htcondor/simP8/sndswEnv.sh"
    source "${REPO_DIR}/htcondor/simP8/sndswEnv.sh"
elif [ -f "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh" ]; then
    echo " ~ [1/4] Sourcing CVMFS stack fallback: /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
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
echo "SND@LHC Stage 6: Event Prefiltering (${CUTSET})"
echo "  Date          : $(date)"
echo "  Host          : $(hostname)"
echo "  Partition     : ${PARTITION}"
echo "  Main Input    : ${INPUT_FILE}"
echo "  Signal Input  : ${SIG_INPUT_FILE}"
echo "  Indirect Input: ${IND_INPUT_FILE}"
echo "  Main Output   : ${OUTPUT_FILE}"
echo "  Signal Output : ${SIG_OUTPUT_FILE}"
echo "  Indirect Out  : ${IND_OUTPUT_FILE}"
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

# 4. Optional per-partition cutflow and failed-cuts files
EXTRA_ARGS=()
if [ -n "${CUTFLOW_DIR:-}" ]; then
    mkdir -p "${CUTFLOW_DIR}"
    EXTRA_ARGS+=(--cutflow "${CUTFLOW_DIR}/cutflow_part${PARTITION}.root")
fi
if [ -n "${CSV_DIR:-}" ]; then
    mkdir -p "${CSV_DIR}"
    EXTRA_ARGS+=(--failed-cuts-csv "${CSV_DIR}/failed_cuts_part${PARTITION}.txt")
fi
if [ "${FORCE:-0}" != "1" ]; then
    EXTRA_ARGS+=(--skip-existing)
fi

echo " ~ [2/3] Executing prefilter.py for partition ${PARTITION}..."
"${PYTHON_BIN}" "${PREFILTER_SCRIPT}" \
    -i "${INPUT_PATTERN}" \
    --signal-input "${SIG_INPUT_PATTERN}" \
    --indirect-input "${IND_INPUT_PATTERN}" \
    -o "${OUTPUT_PATTERN}" \
    --signal-output "${SIG_OUTPUT_PATTERN}" \
    --indirect-output "${IND_OUTPUT_PATTERN}" \
    -p "${PARTITION}" \
    --cutset "${CUTSET}" \
    --nu-scaling "${NU_SCALING}" \
    --pmu-scaling "${PMU_SCALING}" \
    -w 1 -j 1 \
    "${EXTRA_ARGS[@]}"

PREFILTER_EXIT=$?
if [ $PREFILTER_EXIT -ne 0 ]; then
    echo "ERROR: Prefilter failed for partition ${PARTITION} with exit code ${PREFILTER_EXIT}." >&2
    exit $PREFILTER_EXIT
fi

# 5. Verify outputs
if [ ! -f "${OUTPUT_FILE}" ] || [ ! -s "${OUTPUT_FILE}" ]; then
    echo "ERROR: Expected output file does not exist or is empty: ${OUTPUT_FILE}" >&2
    exit 3
fi

MAIN_SIZE=$(stat -c%s "${OUTPUT_FILE}" 2>/dev/null || echo 0)
echo "======================================================================"
echo " ~ [3/3] Prefilter completed successfully for partition ${PARTITION} at $(date)!"
echo "   Output file: ${OUTPUT_FILE} (${MAIN_SIZE} bytes)"
echo "======================================================================"
exit 0
