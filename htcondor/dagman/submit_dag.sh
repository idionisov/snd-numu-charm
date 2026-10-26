#!/bin/bash
set -e

# ==============================================================================
# HTCondor DAGMan Submitter for SND@LHC Full Pipeline
#
# Generates (if needed) and submits a per-partition DAGMan workflow.
#
# Usage:
#   ./submit_dag.sh -p 1-10
#   ./submit_dag.sh -p 1-1000
#   ./submit_dag.sh --dag pipeline.dag
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p out err log

DAG_FILE="pipeline.dag"
PARTITIONS="1-1000"
STAGES="all"
FORCE_FLAG=""
SKIP_DISPLAYS_FLAG=""
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--partitions)
      PARTITIONS="$2"
      shift 2
      ;;
    -d|--dag)
      DAG_FILE="$2"
      shift 2
      ;;
    -s|--stages)
      STAGES="$2"
      shift 2
      ;;
    --force)
      FORCE_FLAG="--force"
      shift
      ;;
    --skip-displays)
      SKIP_DISPLAYS_FLAG="--skip-displays"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    *)
      if [[ "$1" == *.dag ]]; then
          DAG_FILE="$1"
          shift
      else
          PARTITIONS="$1"
          shift
      fi
      ;;
  esac
done

# Generate DAG file if not already specified as an existing custom file
echo "=========================================================="
echo " Preparing HTCondor DAGMan Pipeline Submission"
echo " Partitions   : ${PARTITIONS}"
echo " Stages       : ${STAGES}"
echo " DAG File     : ${DAG_FILE}"
echo " Dry Run      : ${DRY_RUN}"
echo "=========================================================="

python3 generate_dag.py \
    -p "${PARTITIONS}" \
    -s "${STAGES}" \
    -o "${DAG_FILE}" \
    ${FORCE_FLAG} \
    ${SKIP_DISPLAYS_FLAG}

if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "[Dry-Run Mode] Command that would be executed on lxplus:"
    echo "condor_submit_dag -config dagman.config ${DAG_FILE}"
    exit 0
fi

if ! command -v condor_submit_dag >/dev/null 2>&1; then
    echo ""
    echo "ERROR: 'condor_submit_dag' not found in PATH." >&2
    echo "Please execute this submission command from an lxplus node (e.g. lxplus.cern.ch):" >&2
    echo "  cd ${SCRIPT_DIR}" >&2
    echo "  condor_submit_dag -config dagman.config ${DAG_FILE}" >&2
    exit 1
fi

condor_submit_dag -config dagman.config "${DAG_FILE}"

echo ""
echo "DAGMan workflow submitted successfully!"
echo "Monitor with:"
echo "  ./status_dag.sh"
echo "  condor_q"
