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

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  -j, --jobs <N>        Number of parallel HTCondor job slices (default: 50)
  -f, --flavour <FLAV>  HTCondor flavour (espresso, microcentury, longlunch; default: microcentury)
  -o, --out-dir <DIR>   Output directory for per-job CSV files (default: /eos/.../plots/hough_tuning/jobs)
  -e, --extra <ARGS>    Extra arguments to pass to tune_dimuon_hough.py
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

# Generate args file
ARGS_FILE="${SCRIPT_DIR}/args_tuning_${N_JOBS}.txt"
seq 0 $((N_JOBS - 1)) > "${ARGS_FILE}"

echo "================================================================================"
echo " Submitting Hough Tuning Optimization to HTCondor"
echo "================================================================================"
echo " Number of Slices:  ${N_JOBS}"
echo " Job Flavour:       ${FLAVOUR}"
echo " Args File:         ${ARGS_FILE}"
echo " Jobs Output Dir:   ${OUT_DIR}"
echo " Extra Arguments:   ${EXTRA_ARGS}"
echo "================================================================================"

if ! command -v condor_submit >/dev/null 2>&1; then
    echo "ERROR: 'condor_submit' not found in PATH." >&2
    echo "Please run this script from an lxplus node (e.g., ssh lxplus.cern.ch)." >&2
    exit 1
fi

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

"${SUBMIT_CMD[@]}"

echo ""
echo "Submission complete. Monitor status with:"
echo "  ${SCRIPT_DIR}/status_hough_tuning.sh -j ${N_JOBS} -o ${OUT_DIR}"
echo "Merge results once complete with:"
echo "  python3 ${SCRIPT_DIR}/merge_hough_tuning_csv.py -i ${OUT_DIR}"
