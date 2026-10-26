#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Worker Script for Stage 5: Dimuon Track Reconstruction & Skimming
#
# Pipeline per partition:
#   1) Tracking (All Events, nTracks=0): in.root -> *_2MuTrks_truth.root
#   2) Charm Dimuon Skims:               *_2MuTrks_truth.root -> *_charmDimuon{Direct,Indirect}.root
#   3) 2D Event Displays:                *_charmDimuon*.root -> displays_part<p>.root
#      (--mc-truth and --recoMuons enabled)
#
# Usage:
#   run_dimuon_tracking_partition.sh <partition_id> [input_pattern] [output_pattern] [disp_pattern] [store_events]
# ==============================================================================

PARTITION="$1"

if [ -z "$PARTITION" ]; then
    echo "ERROR: Missing required partition ID argument." >&2
    echo "Usage: $0 <partition_id> [input_pattern] [output_pattern] [disp_pattern] [store_events]" >&2
    exit 1
fi

DEFAULT_INPUT="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig.root"
DEFAULT_OUTPUT="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root"
DEFAULT_DISP="/eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions/displays_part%s.root"

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
DISP_PATTERN=$(sanitize_pattern "$4" "$DEFAULT_DISP")
STORE_EVENTS_ARG="${5:-all}"

if [ "$STORE_EVENTS_ARG" = "two_tracks" ] || [ "$STORE_EVENTS_ARG" = "--two-tracks-only" ]; then
    STORE_EVENTS_FLAG="--two-tracks-only"
else
    STORE_EVENTS_FLAG="--all-events"
fi

SKIP_FLAG="--skip-existing"
if [ "${FORCE:-0}" = "1" ]; then
    SKIP_FLAG=""
fi

INPUT_FILE=$(printf "$INPUT_PATTERN" "$PARTITION")
OUTPUT_FILE=$(printf "$OUTPUT_PATTERN" "$PARTITION")
DISP_FILE=$(printf "$DISP_PATTERN" "$PARTITION")

OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "$DISP_FILE")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/afs/cern.ch/work/i/idioniso/snd-numu-charm"
if [ ! -d "$REPO_DIR" ]; then
    REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

# 1. Source environment
if [ -f "${SCRIPT_DIR}/sndswEnv.sh" ]; then
    echo " ~ [1/5] Sourcing local environment: ${SCRIPT_DIR}/sndswEnv.sh"
    source "${SCRIPT_DIR}/sndswEnv.sh"
elif [ -f "${REPO_DIR}/htcondor/simP8/sndswEnv.sh" ]; then
    echo " ~ [1/5] Sourcing repository environment: ${REPO_DIR}/htcondor/simP8/sndswEnv.sh"
    source "${REPO_DIR}/htcondor/simP8/sndswEnv.sh"
elif [ -f "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh" ]; then
    echo " ~ [1/5] Sourcing CVMFS stack fallback: /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
    source "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
fi

if [ -d "${SCRIPT_DIR}/ddfUtils" ]; then
    export ddfUtils="${SCRIPT_DIR}/ddfUtils"
else
    export ddfUtils="${DDFUTILS:-/afs/cern.ch/user/i/idioniso/ddfUtils}"
fi

export PYTHONPATH="${SCRIPT_DIR}:${REPO_DIR}:${SNDSW_SRC}/python:${SNDSW_SRC}/shipLHC/scripts:${SNDSW_ROOT}/python:${ddfUtils}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${REPO_DIR}/build/lib:${SNDSW_ROOT}/lib:${LD_LIBRARY_PATH}"
export SND_CHANNELS_READONLY="${SND_CHANNELS_READONLY:-1}"

PYTHON_BIN="$(command -v python3)"
TRACK_SCRIPT="${REPO_DIR}/scripts/run_dimuon_reco.py"
SKIM_SCRIPT="${REPO_DIR}/scripts/filter_charm_dimuon_categories.py"
DISP_SCRIPT="${REPO_DIR}/scripts/generate_2DEventDisplays.py"
PAR_FILE="${PAR_FILE:-${REPO_DIR}/config/TrackingParams.xml}"

echo "======================================================================"
echo "SND@LHC Stage 5: Dimuon Track Reconstruction & Skimming"
echo "  Date         : $(date)"
echo "  Host         : $(hostname)"
echo "  Partition    : ${PARTITION}"
echo "  Input File   : ${INPUT_FILE}"
echo "  Output Target: ${OUTPUT_FILE}"
echo "  Disp Target  : ${DISP_FILE}"
echo "  Par File     : ${PAR_FILE}"
echo "  Storage Mode : ${STORE_EVENTS_FLAG}"
echo "  Force Mode   : ${FORCE:-0}"
echo "======================================================================"

cd "${REPO_DIR}"

# 2. Check input prerequisite
if [ ! -f "${INPUT_FILE}" ] || [ ! -s "${INPUT_FILE}" ]; then
    echo "ERROR: Input digitized file not found or empty: ${INPUT_FILE}" >&2
    exit 2
fi

# 3. Step 1: Run Tracking (All Events, nTracks=0)
echo " ~ [2/5] Step 1/3: Running dimuon track reconstruction (${STORE_EVENTS_FLAG})..."
set +e
"${PYTHON_BIN}" "${TRACK_SCRIPT}" \
    -i "${INPUT_PATTERN}" \
    -o "${OUTPUT_PATTERN}" \
    -p "${PARTITION}" \
    -j 1 \
    -t "dimuon_DS" \
    -par "${PAR_FILE}" \
    ${STORE_EVENTS_FLAG} \
    ${SKIP_FLAG}
TRACK_EXIT=$?
set -e

# 0 or 143 (SIGTERM atexit handler in FairRoot) is considered normal success
if [ $TRACK_EXIT -ne 0 ] && [ $TRACK_EXIT -ne 143 ]; then
    echo "ERROR: Tracking failed for partition ${PARTITION} with exit code ${TRACK_EXIT}." >&2
    exit $TRACK_EXIT
fi

if [ ! -f "${OUTPUT_FILE}" ] || [ ! -s "${OUTPUT_FILE}" ]; then
    echo "ERROR: Expected tracking output file does not exist or is empty: ${OUTPUT_FILE}" >&2
    exit 3
fi
echo " ~ Tracking output verified: ${OUTPUT_FILE}"

# Also maintain link for *_2MuTrks.root if distinct
LEGACY_TRACK_FILE="${OUTPUT_DIR}/$(basename "${OUTPUT_FILE}" _truth.root).root"
if [ "${LEGACY_TRACK_FILE}" != "${OUTPUT_FILE}" ] && [ ! -e "${LEGACY_TRACK_FILE}" ]; then
    ln -sf "${OUTPUT_FILE}" "${LEGACY_TRACK_FILE}" 2>/dev/null || true
fi

# 4. Step 2: Run Charm Dimuon Categories Skims (Direct and Indirect)
echo " ~ [3/5] Step 2/3: Creating charm dimuon category skims (direct & indirect)..."
DATA_BASE_DIR="$(dirname "$OUTPUT_DIR")"
"${PYTHON_BIN}" "${SKIM_SCRIPT}" \
    -i "${OUTPUT_PATTERN}" \
    -b "${DATA_BASE_DIR}" \
    -p "${PARTITION}" \
    -j 1 \
    ${SKIP_FLAG}
SKIM_EXIT=$?

if [ $SKIM_EXIT -ne 0 ]; then
    echo "ERROR: Charm dimuon skimming failed for partition ${PARTITION} with exit code ${SKIM_EXIT}." >&2
    exit $SKIM_EXIT
fi

OUT_BASE=$(basename "${OUTPUT_FILE}" .root)
DIRECT_FILE="${OUTPUT_DIR}/${OUT_BASE}_charmDimuonDirect.root"
INDIRECT_FILE="${OUTPUT_DIR}/${OUT_BASE}_charmDimuonIndirect.root"

if [ ! -f "${DIRECT_FILE}" ] || [ ! -f "${INDIRECT_FILE}" ]; then
    echo "ERROR: Expected charm skim file(s) missing in ${OUTPUT_DIR}." >&2
    exit 4
fi
echo " ~ Skim outputs verified:"
echo "     Direct   : ${DIRECT_FILE}"
echo "     Indirect : ${INDIRECT_FILE}"

# 5. Step 3: Run 2D Event Display Generation for Skimmed Signal & Signal-like Events
if [ "${SKIP_DISPLAYS:-0}" = "1" ] || [ -z "${DISP_PATTERN}" ] || [ "${DISP_PATTERN}" = "none" ]; then
    echo " ~ [4/5] Skipping Event Displays generation (SKIP_DISPLAYS=${SKIP_DISPLAYS:-0})."
else
    DISP_BASE_DIR="$(dirname "$DISP_FILE")"
    DISP_DIRECT_DIR="${DISP_BASE_DIR}/direct"
    DISP_INDIRECT_DIR="${DISP_BASE_DIR}/indirect"
    mkdir -p "${DISP_DIRECT_DIR}" "${DISP_INDIRECT_DIR}"

    DISP_DIRECT_FILE="${DISP_DIRECT_DIR}/displays_part${PARTITION}.root"
    DISP_INDIRECT_FILE="${DISP_INDIRECT_DIR}/displays_part${PARTITION}.root"

    # Direct Signal Displays
    N_DIRECT=$("${PYTHON_BIN}" -c "import ROOT; f=ROOT.TFile.Open('${DIRECT_FILE}'); t=f.Get('cbmsim') if f and not f.IsZombie() else None; print(t.GetEntries() if t else 0)")
    if [ "$N_DIRECT" -gt 0 ]; then
        if [ "${FORCE:-0}" != "1" ] && [ -f "${DISP_DIRECT_FILE}" ] && [ $(stat -c%s "${DISP_DIRECT_FILE}" 2>/dev/null || echo 0) -gt 500 ]; then
            echo " ~ [4/5] Direct display file already exists: ${DISP_DIRECT_FILE} (skipping)"
        else
            echo " ~ [4/5] Generating 2D displays for ${N_DIRECT} direct charm dimuon signal event(s)..."
            set +e
            "${PYTHON_BIN}" "${DISP_SCRIPT}" \
                -i "${DIRECT_FILE}" \
                -o "${DISP_DIRECT_FILE}" \
                --mc-truth \
                --recoMuons \
                -j 1
            DISP_EXIT=$?
            set -e
            if [ $DISP_EXIT -ne 0 ]; then
                echo "WARNING: Direct display generation returned exit code ${DISP_EXIT} for partition ${PARTITION}." >&2
            fi
        fi
    else
        echo " ~ [4/5] Direct signal events: 0 (no displays needed)."
    fi

    # Indirect Signal-like Displays
    N_INDIRECT=$("${PYTHON_BIN}" -c "import ROOT; f=ROOT.TFile.Open('${INDIRECT_FILE}'); t=f.Get('cbmsim') if f and not f.IsZombie() else None; print(t.GetEntries() if t else 0)")
    if [ "$N_INDIRECT" -gt 0 ]; then
        if [ "${FORCE:-0}" != "1" ] && [ -f "${DISP_INDIRECT_FILE}" ] && [ $(stat -c%s "${DISP_INDIRECT_FILE}" 2>/dev/null || echo 0) -gt 500 ]; then
            echo " ~ [5/5] Indirect display file already exists: ${DISP_INDIRECT_FILE} (skipping)"
        else
            echo " ~ [5/5] Generating 2D displays for ${N_INDIRECT} indirect charm dimuon event(s)..."
            set +e
            "${PYTHON_BIN}" "${DISP_SCRIPT}" \
                -i "${INDIRECT_FILE}" \
                -o "${DISP_INDIRECT_FILE}" \
                --mc-truth \
                --recoMuons \
                -j 1
            DISP_EXIT=$?
            set -e
            if [ $DISP_EXIT -ne 0 ]; then
                echo "WARNING: Indirect display generation returned exit code ${DISP_EXIT} for partition ${PARTITION}." >&2
            fi
        fi
    else
        echo " ~ [5/5] Indirect signal-like events: 0 (no displays needed)."
    fi
fi

echo "======================================================================"
echo " ~ [SUCCESS] Partition ${PARTITION} tracking & skimming completed at $(date)!"
echo "   Output file: ${OUTPUT_FILE}"
echo "======================================================================"
