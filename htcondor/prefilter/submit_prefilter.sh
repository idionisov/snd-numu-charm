#!/bin/bash
set -e

# ==============================================================================
# HTCondor Submission for Event Prefiltering
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p out err log

ARGS_FILE="${SCRIPT_DIR}/args_run6596_287.txt"
TMP_ARGS=""
FLAVOUR="microcentury"
FORCE="0"
DRY_RUN="0"
CUTSET="b2nob1Muons"
INPUT_PATTERN="/eos/experiment/sndlhc/convertedData/physics/2023/run_006596/sndsw_raw-%s.root"
OUTPUT_PATTERN="/eos/user/i/idioniso/snd-numu-charm/prefiltered/2023/b2nob1Muons/run006596/sndsw_raw-%s.root"
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--args-file)
      ARGS_FILE="$2"
      shift 2
      ;;
    -f|--flavour)
      FLAVOUR="$2"
      shift 2
      ;;
    -i|--input-pattern)
      INPUT_PATTERN="$2"
      shift 2
      ;;
    -o|--output-pattern)
      OUTPUT_PATTERN="$2"
      shift 2
      ;;
    --force)
      FORCE="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --cutset)
      CUTSET="$2"
      shift 2
      ;;
    -p|--partitions)
      PART_ARG="$2"
      shift 2
      TMP_ARGS=$(mktemp /tmp/prefilter_args_XXXXXX.txt)
      if [[ "${PART_ARG}" =~ ^[0-9]+-[0-9]+$ ]]; then
          START=$(echo "${PART_ARG}" | cut -d'-' -f1)
          END=$(echo "${PART_ARG}" | cut -d'-' -f2)
          seq -f "%04g" "${START}" "${END}" > "${TMP_ARGS}"
      else
          for p_item in $(echo "${PART_ARG}" | tr ',' ' '); do
              if [[ "$p_item" =~ ^[0-9]+$ ]] && [ ${#p_item} -lt 4 ]; then
                  printf "%04d\n" "$((10#$p_item))" >> "${TMP_ARGS}"
              else
                  echo "$p_item" >> "${TMP_ARGS}"
              fi
          done
      fi
      ARGS_FILE="${TMP_ARGS}"
      ;;
    --run)
      RUN_NUM="$2"
      shift 2
      RUN_ID=$(printf "%06d" "$((10#$RUN_NUM))")
      RUN_DIR_ID=$(printf "%06d" "$((10#$RUN_NUM))")
      INPUT_PATTERN="/eos/experiment/sndlhc/convertedData/physics/2023/run_${RUN_ID}/sndsw_raw-%s.root"
      OUTPUT_PATTERN="/eos/user/i/idioniso/snd-numu-charm/prefiltered/2023/${CUTSET}/run${RUN_DIR_ID}/sndsw_raw-%s.root"
      ;;
    *)
      if [ -f "$1" ]; then
          ARGS_FILE="$1"
          shift
      else
          POSITIONAL+=("$1")
          shift
      fi
      ;;
  esac
done

if [ ${#POSITIONAL[@]} -eq 1 ] && [[ "${POSITIONAL[0]}" =~ ^[0-9]+-[0-9]+$ ]]; then
    START=$(echo "${POSITIONAL[0]}" | cut -d'-' -f1)
    END=$(echo "${POSITIONAL[0]}" | cut -d'-' -f2)
    TMP_ARGS=$(mktemp /tmp/prefilter_args_XXXXXX.txt)
    seq -f "%04g" "${START}" "${END}" > "${TMP_ARGS}"
    ARGS_FILE="${TMP_ARGS}"
elif [ ${#POSITIONAL[@]} -ge 2 ] && [[ "${POSITIONAL[0]}" =~ ^[0-9]+$ ]] && [[ "${POSITIONAL[1]}" =~ ^[0-9]+$ ]]; then
    START="${POSITIONAL[0]}"
    END="${POSITIONAL[1]}"
    TMP_ARGS=$(mktemp /tmp/prefilter_args_XXXXXX.txt)
    seq -f "%04g" "${START}" "${END}" > "${TMP_ARGS}"
    ARGS_FILE="${TMP_ARGS}"
fi

trap '[ -n "${TMP_ARGS}" ] && rm -f "${TMP_ARGS}"' EXIT

if [ ! -f "${ARGS_FILE}" ]; then
    echo "ERROR: Args file not found: ${ARGS_FILE}" >&2
    exit 1
fi

TOTAL_PARTITIONS=$(wc -l < "${ARGS_FILE}")
echo "=========================================================="
echo " Submitting Event Prefiltering (${TOTAL_PARTITIONS} Partitions) to HTCondor"
echo " Total Jobs     : ${TOTAL_PARTITIONS} partitions"
echo " Flavour        : ${FLAVOUR}"
echo " Cutset         : ${CUTSET}"
echo " Force Mode     : ${FORCE}"
echo " Args File      : ${ARGS_FILE}"
echo " Input Pattern  : ${INPUT_PATTERN}"
echo " Output Pattern : ${OUTPUT_PATTERN}"
echo "=========================================================="

if [ "${DRY_RUN}" = "1" ]; then
    echo "[Dry-Run Mode] Command that would be executed:"
    echo "condor_submit prefilter.sub args_file=\"${ARGS_FILE}\" flavour=\"${FLAVOUR}\" force=\"${FORCE}\" cutset=\"${CUTSET}\" input_pattern=\"${INPUT_PATTERN}\" output_pattern=\"${OUTPUT_PATTERN}\""
    exit 0
fi

if ! command -v condor_submit >/dev/null 2>&1; then
    echo "ERROR: 'condor_submit' not found in PATH." >&2
    echo "Please run this script from an lxplus node (e.g., ssh lxplus.cern.ch)." >&2
    exit 1
fi

condor_submit prefilter.sub \
    args_file="${ARGS_FILE}" \
    flavour="${FLAVOUR}" \
    force="${FORCE}" \
    cutset="${CUTSET}" \
    input_pattern="${INPUT_PATTERN}" \
    output_pattern="${OUTPUT_PATTERN}"

echo ""
echo "Jobs submitted! You can monitor them using:"
echo "  ./htcondor/prefilter/status_prefilter.sh"
