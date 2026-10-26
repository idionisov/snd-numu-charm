#!/bin/bash
set -e

# ==============================================================================
# HTCondor Submission for Stage 5: Dimuon Track Reconstruction & Skimming
#
# Process per partition:
#   Inputs : sndLHC.Genie-TGeant4_MCEB_dig.root
#   Outputs: sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root
#            sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonDirect.root
#            sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonIndirect.root
#            displays_part<p>.root (direct & indirect event displays)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p out err log

ARGS_FILE="${SCRIPT_DIR}/args_nu14_volTarget_1000.txt"
TMP_ARGS=""
FLAVOUR="longlunch"
FORCE="0"
DRY_RUN="0"
STORE_EVENTS="all"
SKIP_DISPLAYS="0"
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
    --force)
      FORCE="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --two-tracks-only)
      STORE_EVENTS="two_tracks"
      shift
      ;;
    --skip-displays)
      SKIP_DISPLAYS="1"
      shift
      ;;
    -p|--partitions)
      PART_ARG="$2"
      shift 2
      TMP_ARGS=$(mktemp /tmp/tracking_args_XXXXXX.txt)
      if [[ "${PART_ARG}" =~ ^[0-9]+-[0-9]+$ ]]; then
          START=$(echo "${PART_ARG}" | cut -d'-' -f1)
          END=$(echo "${PART_ARG}" | cut -d'-' -f2)
          seq "${START}" "${END}" > "${TMP_ARGS}"
      else
          echo "${PART_ARG}" | tr ',' '\n' > "${TMP_ARGS}"
      fi
      ARGS_FILE="${TMP_ARGS}"
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
    TMP_ARGS=$(mktemp /tmp/tracking_args_XXXXXX.txt)
    seq "${START}" "${END}" > "${TMP_ARGS}"
    ARGS_FILE="${TMP_ARGS}"
elif [ ${#POSITIONAL[@]} -ge 2 ] && [[ "${POSITIONAL[0]}" =~ ^[0-9]+$ ]] && [[ "${POSITIONAL[1]}" =~ ^[0-9]+$ ]]; then
    START="${POSITIONAL[0]}"
    END="${POSITIONAL[1]}"
    TMP_ARGS=$(mktemp /tmp/tracking_args_XXXXXX.txt)
    seq "${START}" "${END}" > "${TMP_ARGS}"
    ARGS_FILE="${TMP_ARGS}"
fi

trap '[ -n "${TMP_ARGS}" ] && rm -f "${TMP_ARGS}"' EXIT

TOTAL_PARTITIONS=$(wc -l < "${ARGS_FILE}")
echo "=========================================================="
echo " Submitting Stage 5: Dimuon Tracking (${TOTAL_PARTITIONS} Partitions) to HTCondor"
echo " Total Jobs     : ${TOTAL_PARTITIONS} partitions"
echo " Flavour        : ${FLAVOUR}"
echo " Force Mode     : ${FORCE}"
echo " Store Events   : ${STORE_EVENTS}"
echo " Skip Displays  : ${SKIP_DISPLAYS}"
echo " Args File      : ${ARGS_FILE}"
echo "=========================================================="

if [ "${DRY_RUN}" = "1" ]; then
    echo "[Dry-Run Mode] Command that would be executed:"
    echo "condor_submit dimuon_tracking.sub args_file=\"${ARGS_FILE}\" flavour=\"${FLAVOUR}\" force=\"${FORCE}\" store_events=\"${STORE_EVENTS}\" skip_displays=\"${SKIP_DISPLAYS}\""
    exit 0
fi

if ! command -v condor_submit >/dev/null 2>&1; then
    echo "ERROR: 'condor_submit' not found in PATH." >&2
    echo "Please run this script from an lxplus node (e.g., ssh lxplus.cern.ch)." >&2
    exit 1
fi

condor_submit dimuon_tracking.sub \
    args_file="${ARGS_FILE}" \
    flavour="${FLAVOUR}" \
    force="${FORCE}" \
    store_events="${STORE_EVENTS}" \
    skip_displays="${SKIP_DISPLAYS}"

echo ""
echo "Jobs submitted! You can monitor them using:"
echo "  ./htcondor/dimuon_tracking/status_dimuon_tracking.sh"
