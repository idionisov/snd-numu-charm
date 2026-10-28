#!/bin/bash
set -eo pipefail

# ==============================================================================
# Helper Script to Submit Hough Parameter Tuning Grid Search to HTCondor
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

N_JOBS=50
FLAVOUR="microcentury"
OUT_DIR="/eos/user/i/idioniso/snd-numu-charm/plots/hough_tuning/jobs"
EXTRA_ARGS="--pairwise --hf linearSlopeIntercept --sort-by asimov_z"
DRY_RUN=0
CLEAN_OUT=0
RETRY_FAILED=0
CUSTOM_ARGS_FILE=""
SLICES_ARG=""

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  -j, --jobs <N>        Total number of grid slices (default: 50)
  -f, --flavour <FLAV>  HTCondor flavour (espresso, microcentury, longlunch, workday; default: microcentury)
  -o, --out-dir <DIR>   Output directory for per-job CSV files (default: /eos/.../plots/hough_tuning/jobs)
  -e, --extra <ARGS>    Extra arguments to pass to tune_dimuon_hough.py
      --retry-failed    Automatically find and submit only missing/failed job slices
  -s, --slices <LIST>   Comma- or space-separated list of slice IDs to run (e.g. 19,36,39)
  -a, --args-file <FILE> Custom arguments file containing job_id values (one per line)
      --clean           Remove existing hough_tuning_job_*.csv files in OUT_DIR before submitting
      --dry-run         Prepare arguments file and print condor_submit command without executing
  -h, --help            Show this help message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)
            N_JOBS="$2"
            shift 2
            ;;
        -f|--flavour)
            FLAVOUR="$2"
            shift 2
            ;;
        -o|--out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        -e|--extra)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        --retry-failed)
            RETRY_FAILED=1
            shift
            ;;
        -s|--slices)
            SLICES_ARG="$2"
            shift 2
            ;;
        -a|--args-file)
            CUSTOM_ARGS_FILE="$2"
            shift 2
            ;;
        --clean)
            CLEAN_OUT=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

# Ensure log and output directories exist
mkdir -p out err log "${OUT_DIR}"

if [ "${CLEAN_OUT}" -eq 1 ] && [ -d "${OUT_DIR}" ]; then
    echo "Cleaning existing job CSV files from ${OUT_DIR}..."
    rm -f "${OUT_DIR}"/hough_tuning_job_*.csv
fi

# Prepare args file
ARGS_FILE="${SCRIPT_DIR}/args_tuning_${N_JOBS}.txt"

if [ -n "${CUSTOM_ARGS_FILE}" ]; then
    ARGS_FILE="${CUSTOM_ARGS_FILE}"
elif [ -n "${SLICES_ARG}" ]; then
    ARGS_FILE="${SCRIPT_DIR}/args_tuning_custom.txt"
    echo "${SLICES_ARG}" | tr ',' '\n' | tr ' ' '\n' | grep -v '^$' > "${ARGS_FILE}"
elif [ "${RETRY_FAILED}" -eq 1 ]; then
    echo "Scanning ${OUT_DIR} for missing job slices (out of ${N_JOBS})..."
    MISSING_SLICES=$(python3 -c "
import os, glob, re
base = '${OUT_DIR}'
done = set()
if os.path.isdir(base):
    for f in glob.glob(os.path.join(base, 'hough_tuning_job_*.csv')):
        if os.path.getsize(f) > 0:
            m = re.search(r'hough_tuning_job_(\d+)\.csv', f)
            if m: done.add(int(m.group(1)))
missing = [i for i in range(${N_JOBS}) if i not in done]
print('\n'.join(str(i) for i in missing))
")
    if [ -z "${MISSING_SLICES}" ]; then
        echo "No missing slices found! All ${N_JOBS} jobs completed successfully."
        exit 0
    fi
    ARGS_FILE="${SCRIPT_DIR}/args_tuning_retry_${N_JOBS}.txt"
    echo "${MISSING_SLICES}" > "${ARGS_FILE}"
    N_MISSING=$(wc -l < "${ARGS_FILE}")
    echo "Found ${N_MISSING} missing slice(s): $(tr '\n' ' ' < "${ARGS_FILE}")"
else
    seq 0 $((N_JOBS - 1)) > "${ARGS_FILE}"
fi

echo "================================================================================"
echo " Submitting Hough Tuning Optimization to HTCondor"
echo "================================================================================"
echo " Number of Slices:  ${N_JOBS}"
echo " Job Flavour:       ${FLAVOUR}"
echo " Args File:         ${ARGS_FILE}"
echo " Jobs Output Dir:   ${OUT_DIR}"
echo " Extra Arguments:   ${EXTRA_ARGS}"
echo "================================================================================"

SUBMIT_CMD=(
    condor_submit
    -a "flavour = ${FLAVOUR}"
    -a "args_file = ${ARGS_FILE}"
    -a "n_jobs = ${N_JOBS}"
    -a "out_dir = ${OUT_DIR}"
    -a "extra_args = ${EXTRA_ARGS}"
    "${SCRIPT_DIR}/hough_tuning.sub"
)

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[DRY-RUN] Command that would be executed:"
    printf "  %q" "${SUBMIT_CMD[@]}"
    echo ""
    exit 0
fi

if ! command -v condor_submit >/dev/null 2>&1; then
    echo "ERROR: 'condor_submit' not found in PATH." >&2
    echo "Please run this script from an lxplus node (e.g., ssh lxplus.cern.ch)." >&2
    exit 1
fi

"${SUBMIT_CMD[@]}"

echo ""
echo "Submission complete. Monitor status with:"
echo "  ${SCRIPT_DIR}/status_hough_tuning.sh -j ${N_JOBS} -o ${OUT_DIR}"
echo "Merge results once complete with:"
echo "  python3 ${SCRIPT_DIR}/merge_hough_tuning_csv.py -i ${OUT_DIR}"
