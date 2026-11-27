#!/usr/bin/env bash
# ==============================================================================
# Run Full PMU Monte Carlo Simulation Threshold & Efficiency Analysis in Parallel
# ==============================================================================
# Splits the 1,108,804 events in the PMU MC sample across N parallel workers,
# merges the resulting chunk histograms with hadd, and generates smoothed plots.
#
# Usage:
#   bash scripts/qdc_calib/run_full_pmu_parallel.sh [N_WORKERS] [MAX_EVENTS]
# Examples:
#   bash scripts/qdc_calib/run_full_pmu_parallel.sh 12 1108804   # Full dataset
#   bash scripts/qdc_calib/run_full_pmu_parallel.sh 12 200000    # Quick high-stats run
# ==============================================================================

set -euo pipefail

N_WORKERS="${1:-12}"
MAX_EVENTS="${2:-1108804}"
MC_FILE="/eos/user/i/idioniso/snd-numu-charm/data/pmu2022/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_MCEB_digCPP.root"
OUTDIR="plots"
CHUNKS_DIR="${OUTDIR}/chunks_pmu"
FINAL_ROOT="${OUTDIR}/mc_threshold_study.root"

mkdir -p "$CHUNKS_DIR"
rm -f "${CHUNKS_DIR}"/chunk_*.root "${CHUNKS_DIR}"/worker_*.log

echo "================================================================================"
echo "Starting Parallel PMU MC Threshold Study"
echo "  Total Events : $MAX_EVENTS"
echo "  Workers      : $N_WORKERS"
echo "  Input MC     : $MC_FILE"
echo "  Chunks Dir   : $CHUNKS_DIR"
echo "  Output ROOT  : $FINAL_ROOT"
echo "================================================================================"

CHUNK_SIZE=$(( (MAX_EVENTS + N_WORKERS - 1) / N_WORKERS ))
PIDS=()

for (( i=0; i<N_WORKERS; i++ )); do
    START_EV=$(( i * CHUNK_SIZE ))
    if [ "$START_EV" -ge "$MAX_EVENTS" ]; then
        break
    fi
    CUR_CHUNK=$CHUNK_SIZE
    REMAINING=$(( MAX_EVENTS - START_EV ))
    if [ "$CUR_CHUNK" -gt "$REMAINING" ]; then
        CUR_CHUNK=$REMAINING
    fi

    CHUNK_ROOT="${CHUNKS_DIR}/chunk_${i}.root"
    CHUNK_LOG="${CHUNKS_DIR}/worker_${i}.log"

    echo "  [Worker $i] Events $START_EV to $(( START_EV + CUR_CHUNK )) ($CUR_CHUNK events) -> $CHUNK_ROOT"

    python3 -X faulthandler scripts/qdc_calib/study_mc_thresholds.py \
        --mc-file "$MC_FILE" \
        --start-event "$START_EV" \
        --max-events "$CUR_CHUNK" \
        --no-plots \
        --root-output "$CHUNK_ROOT" > "$CHUNK_LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $N_WORKERS workers spawned in background. Waiting for completion..."
FAIL_COUNT=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    fi
done

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "ERROR: $FAIL_COUNT worker(s) failed. Check logs in $CHUNKS_DIR/"
    exit 1
fi

echo ""
echo "All workers finished successfully. Merging chunk histograms with hadd..."
hadd -f "$FINAL_ROOT" "${CHUNKS_DIR}"/chunk_*.root

echo ""
echo "Finalizing publication figures and ROOT canvases..."
python3 -X faulthandler scripts/qdc_calib/finalize_threshold_plots.py \
    --root-file "$FINAL_ROOT" \
    --outdir "$OUTDIR"

echo ""
echo "================================================================================"
echo "Parallel PMU Run Complete!"
echo "Merged ROOT file : $FINAL_ROOT"
echo "Plots saved to   : $OUTDIR/"
echo "================================================================================"
