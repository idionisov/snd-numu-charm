#!/bin/bash
set -eo pipefail

# ==============================================================================
# Monitor Status of Hough Parameter Tuning HTCondor Jobs
# ==============================================================================

N_JOBS=50
OUT_DIR="/eos/user/i/idioniso/snd-numu-charm/plots/hough_tuning/jobs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)
            N_JOBS="$2"
            shift 2
            ;;
        -o|--out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

COMPLETED=0
if [ -d "${OUT_DIR}" ]; then
    COMPLETED=$(find "${OUT_DIR}" -maxdepth 1 -name "hough_tuning_job_*.csv" -size +0c 2>/dev/null | wc -l || true)
fi

RUNNING=0
IDLE=0
HELD=0
if command -v condor_q >/dev/null 2>&1; then
    RUNNING=$(condor_q -format "%d\n" JobStatus -constraint 'JobStatus == 2 && regexp("run_hough_tuning_job", Cmd)' 2>/dev/null | wc -l || true)
    IDLE=$(condor_q -format "%d\n" JobStatus -constraint 'JobStatus == 1 && regexp("run_hough_tuning_job", Cmd)' 2>/dev/null | wc -l || true)
    HELD=$(condor_q -format "%d\n" JobStatus -constraint 'JobStatus == 5 && regexp("run_hough_tuning_job", Cmd)' 2>/dev/null | wc -l || true)
fi

RECENT_LOGS=$(find "${SCRIPT_DIR}/log" -maxdepth 1 -name "tuning_*.log" -mmin -120 2>/dev/null | wc -l || true)
RECENT_ERRS=$(find "${SCRIPT_DIR}/err" -maxdepth 1 -name "tuning_job*.err" -size +0c -mmin -120 2>/dev/null | wc -l || true)

PERCENT=0
if [ "${N_JOBS}" -gt 0 ]; then
    PERCENT=$(( COMPLETED * 100 / N_JOBS ))
fi

echo "================================================================================"
echo " Hough Tuning Grid Search: HTCondor Batch Status"
echo "================================================================================"
echo " Total Slices:     ${N_JOBS}"
echo " Completed CSVs:   ${COMPLETED} / ${N_JOBS} (${PERCENT}%)"
echo " Active Running:   ${RUNNING}"
echo " Idle (Queued):    ${IDLE}"
echo " Held:             ${HELD}"
echo " Recent Logs:      ${RECENT_LOGS} (in log/)"
echo " Error Files:      ${RECENT_ERRS} non-empty (in err/)"
echo " Jobs Output Dir:  ${OUT_DIR}"
echo "================================================================================"

if [ "${RUNNING}" -gt 0 ] || [ "${IDLE}" -gt 0 ] || [ "${HELD}" -gt 0 ]; then
    echo "Active HTCondor Jobs:"
    condor_q -constraint 'regexp("run_hough_tuning_job", Cmd)'
    echo "================================================================================"
fi

if [ "${COMPLETED}" -ge "${N_JOBS}" ] && [ "${N_JOBS}" -gt 0 ]; then
    echo "All job slices complete! You can now merge the results with:"
    echo "  python3 htcondor/hough_tuning/merge_hough_tuning_csv.py -i ${OUT_DIR}"
fi
