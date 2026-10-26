#!/bin/bash
set -e

# ==============================================================================
# HTCondor Submission for Stage 2: Monte Carlo Event Builder (MCEB)
#
# Process per partition:
#   Input : /eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4.root
#   Output: /eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB.root
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p out err log

ARGS_FILE="${SCRIPT_DIR}/args_nu14_volTarget_1000.txt"
TMP_ARGS=""
NEVENTS="0"
FLAVOUR="microcentury"
FORCE="0"
DRY_RUN="0"
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--args-file)
      ARGS_FILE="$2"
      shift 2
      ;;
    -n|--nevents)
      NEVENTS="$2"
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
    -p|--partitions)
      PART_ARG="$2"
      shift 2
      TMP_ARGS=$(mktemp /tmp/mceb_args_XXXXXX.txt)
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
    TMP_ARGS=$(mktemp /tmp/mceb_args_XXXXXX.txt)
    seq "${START}" "${END}" > "${TMP_ARGS}"
    ARGS_FILE="${TMP_ARGS}"
elif [ ${#POSITIONAL[@]} -ge 2 ] && [[ "${POSITIONAL[0]}" =~ ^[0-9]+$ ]] && [[ "${POSITIONAL[1]}" =~ ^[0-9]+$ ]]; then
    START="${POSITIONAL[0]}"
    END="${POSITIONAL[1]}"
    TMP_ARGS=$(mktemp /tmp/mceb_args_XXXXXX.txt)
    seq "${START}" "${END}" > "${TMP_ARGS}"
    ARGS_FILE="${TMP_ARGS}"
fi

# Auto-adjust flavour for short tests
if [ "${NEVENTS}" -gt 0 ] && [ "${NEVENTS}" -le 10 ] && [ "${FLAVOUR}" == "longlunch" ]; then
    FLAVOUR="microcentury"
fi

trap '[ -n "${TMP_ARGS}" ] && rm -f "${TMP_ARGS}"' EXIT
DATA_PATTERN="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s"

TOTAL_PARTITIONS=$(wc -l < "${ARGS_FILE}")
echo "=========================================================="
echo " Submitting Stage 2: MCEventBuilder (${TOTAL_PARTITIONS} Partitions) to HTCondor"
echo " Total Jobs     : ${TOTAL_PARTITIONS} partitions"
echo " Events/job     : ${NEVENTS} (0 = all)"
echo " Flavour        : ${FLAVOUR}"
echo " Force Mode     : ${FORCE}"
echo " Data Pattern   : ${DATA_PATTERN}"
echo " Args File      : ${ARGS_FILE}"
echo "=========================================================="

if [ "${DRY_RUN}" = "1" ]; then
    echo "[Dry-Run Mode] Command that would be executed:"
    echo "condor_submit mceb.sub args_file=\"${ARGS_FILE}\" flavour=\"${FLAVOUR}\" nevents=\"${NEVENTS}\" force=\"${FORCE}\""
    exit 0
fi

if ! command -v condor_submit >/dev/null 2>&1; then
    echo "ERROR: 'condor_submit' not found in PATH." >&2
    echo "Please run this script from an lxplus node (e.g., ssh lxplus.cern.ch)." >&2
    exit 1
fi

condor_submit mceb.sub \
    args_file="${ARGS_FILE}" \
    flavour="${FLAVOUR}" \
    nevents="${NEVENTS}" \
    force="${FORCE}"

echo ""
echo "Jobs submitted! You can monitor them using:"
echo "  ./htcondor/mceb/status_mceb.sh"
