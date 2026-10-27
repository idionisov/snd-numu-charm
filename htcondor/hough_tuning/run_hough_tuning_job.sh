#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Worker Script for Hough Tracking Tuning Optimization
#
# Arguments:
#   $1: JOB_ID     (slice index, 0 to N_JOBS - 1)
#   $2: N_JOBS     (total job slices in grid)
#   $3: OUT_DIR    (target directory for per-job CSV outputs)
#   $4+: EXTRA_ARGS (optional flags passed to tune_dimuon_hough.py)
# ==============================================================================

JOB_ID="$1"
N_JOBS="${2:-50}"
OUT_DIR="${3:-/eos/user/i/idioniso/snd-numu-charm/plots/hough_tuning/jobs}"
shift 3 || true
EXTRA_ARGS="$*"

if [ -z "${JOB_ID}" ]; then
    echo "ERROR: Missing required JOB_ID argument." >&2
    echo "Usage: $0 <job_id> [n_jobs] [out_dir] [extra_args...]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source SNDSW & Python runtime environment
if [ -f "${SCRIPT_DIR}/sndswEnv.sh" ]; then
    source "${SCRIPT_DIR}/sndswEnv.sh" > /dev/null 2>&1 || true
fi

# Limit low-level math library threading
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/pgun:${PYTHONPATH}"

echo "================================================================================"
echo " SND@LHC Hough Tuning Batch Worker - Slice ${JOB_ID} / ${N_JOBS}"
echo "================================================================================"
echo " Host:        $(hostname)"
echo " Date:        $(date)"
echo " Project:     ${PROJECT_ROOT}"
echo " Output Dir:  ${OUT_DIR}"
echo " Job Slice:   ${JOB_ID} of ${N_JOBS}"
echo " Extra Args:  ${EXTRA_ARGS:-<default pairwise>}"
echo "================================================================================"

mkdir -p "${OUT_DIR}"

LOCAL_TMP=$(mktemp -d /tmp/hough_tune_job_${JOB_ID}_XXXXXX)
trap 'rm -rf "${LOCAL_TMP}"' EXIT

LOCAL_CSV="${LOCAL_TMP}/hough_tuning_job_${JOB_ID}.csv"
DEST_CSV="${OUT_DIR}/hough_tuning_job_${JOB_ID}.csv"

# Default to pairwise linearSlopeIntercept if no extra args specified
if [ -z "${EXTRA_ARGS}" ]; then
    EXTRA_ARGS="--pairwise --hf linearSlopeIntercept --sort-by asimov_z"
fi

START_TIME=$(date +%s)

python3 -X faulthandler "${PROJECT_ROOT}/pgun/scripts/tune_dimuon_hough.py" \
    --job-id "${JOB_ID}" \
    --n-jobs "${N_JOBS}" \
    --output-csv "${LOCAL_CSV}" \
    --xml "${PROJECT_ROOT}/config/TrackingParams.xml" \
    --workers 1 \
    ${EXTRA_ARGS}

ELAPSED=$(( $(date +%s) - START_TIME ))

if [ ! -f "${LOCAL_CSV}" ] || [ ! -s "${LOCAL_CSV}" ]; then
    echo "ERROR: Output CSV '${LOCAL_CSV}' was not generated or is empty." >&2
    exit 2
fi

N_ROWS=$(wc -l < "${LOCAL_CSV}")
echo "Generated ${N_ROWS} lines in ${ELAPSED} s."

# Safely copy to final destination
cp -f "${LOCAL_CSV}" "${DEST_CSV}"

if [ -f "${DEST_CSV}" ] && [ -s "${DEST_CSV}" ]; then
    echo "SUCCESS: Output saved to ${DEST_CSV}"
    exit 0
else
    echo "ERROR: Failed to stage output file to ${DEST_CSV}" >&2
    exit 3
fi
