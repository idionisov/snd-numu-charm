#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Worker Script for Stage 3: MC Truth Extraction & Transplantation
#
# Process per partition:
#   Input Truth: sndLHC.Genie-TGeant4.root (non-MCEB unpruned truth)
#   Input MCEB : sndLHC.Genie-TGeant4_MCEB.root (MCEB events)
#   Output     : sndLHC.Genie-TGeant4_MCEB.root (or custom output file)
#                Contains cbmsim;1 (MCEB), cbmsim;2 (unpruned truth), and mctruth
#
# Usage:
#   run_mctruth_partition.sh <partition_id> [truth_pattern] [input_mceb_pattern] [output_pattern]
# ==============================================================================

PARTITION="$1"

if [ -z "$PARTITION" ]; then
    echo "ERROR: Missing required partition ID argument." >&2
    echo "Usage: $0 <partition_id> [truth_pattern] [input_mceb_pattern] [output_pattern]" >&2
    exit 1
fi

DEFAULT_TRUTH_PATTERN="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4.root"
DEFAULT_MCEB_PATTERN="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s/sndLHC.Genie-TGeant4_MCEB.root"

TRUTH_PATTERN="${2:-$DEFAULT_TRUTH_PATTERN}"
MCEB_PATTERN="${3:-$DEFAULT_MCEB_PATTERN}"
# Default output is in-place into MCEB file (or distinct file if 4th arg is given)
OUTPUT_PATTERN="${4:-$MCEB_PATTERN}"

# Clean up unexpanded HTCondor macro strings
if [[ "${TRUTH_PATTERN}" == \$* ]] || [ -z "${TRUTH_PATTERN}" ]; then
    TRUTH_PATTERN="$DEFAULT_TRUTH_PATTERN"
fi
if [[ "${MCEB_PATTERN}" == \$* ]] || [ -z "${MCEB_PATTERN}" ]; then
    MCEB_PATTERN="$DEFAULT_MCEB_PATTERN"
fi
if [[ "${OUTPUT_PATTERN}" == \$* ]] || [ -z "${OUTPUT_PATTERN}" ]; then
    OUTPUT_PATTERN="$MCEB_PATTERN"
fi

TRUTH_FILE=$(printf "${TRUTH_PATTERN}" "${PARTITION}")
MCEB_FILE=$(printf "${MCEB_PATTERN}" "${PARTITION}")
OUTPUT_FILE=$(printf "${OUTPUT_PATTERN}" "${PARTITION}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/afs/cern.ch/work/i/idioniso/snd-numu-charm"
if [ ! -d "$REPO_DIR" ]; then
    REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

# 1. Source environment
if [ -f "${SCRIPT_DIR}/sndswEnv.sh" ]; then
    echo " ~ [1/4] Sourcing local environment: ${SCRIPT_DIR}/sndswEnv.sh"
    source "${SCRIPT_DIR}/sndswEnv.sh"
elif [ -f "${REPO_DIR}/htcondor/simP8/sndswEnv.sh" ]; then
    echo " ~ [1/4] Sourcing repository environment: ${REPO_DIR}/htcondor/simP8/sndswEnv.sh"
    source "${REPO_DIR}/htcondor/simP8/sndswEnv.sh"
elif [ -f "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh" ]; then
    echo " ~ [1/4] Sourcing CVMFS stack fallback: /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
    source "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
fi

if [ -d "${SCRIPT_DIR}/ddfUtils" ]; then
    export ddfUtils="${SCRIPT_DIR}/ddfUtils"
else
    export ddfUtils="${DDFUTILS:-/afs/cern.ch/user/i/idioniso/ddfUtils}"
fi

export PYTHONPATH="${REPO_DIR}:${SNDSW_SRC}/python:${SNDSW_SRC}/shipLHC/scripts:${SNDSW_ROOT}/python:${ddfUtils}:${PYTHONPATH}"

echo " ~ [2/4] Environment configured"
echo "         Date             : $(date)"
echo "         Host             : $(hostname)"
echo "         Partition        : ${PARTITION}"
echo "         Truth source     : ${TRUTH_FILE}"
echo "         Input MCEB       : ${MCEB_FILE}"
echo "         Output target    : ${OUTPUT_FILE}"
echo "         Force mode       : ${FORCE:-0}"

# 2. Check prerequisites
if [ ! -f "${TRUTH_FILE}" ] || [ ! -s "${TRUTH_FILE}" ]; then
    # Fallback to digitized truth file if raw simulation is not found
    ALT_TRUTH="${TRUTH_FILE%.root}_dig.root"
    if [ -f "${ALT_TRUTH}" ] && [ -s "${ALT_TRUTH}" ]; then
        echo " ~ [Note] Using alternate truth source: ${ALT_TRUTH}"
        TRUTH_FILE="${ALT_TRUTH}"
    else
        echo "ERROR: Input truth file not found: ${TRUTH_FILE}" >&2
        exit 2
    fi
fi

if [ ! -f "${MCEB_FILE}" ] || [ ! -s "${MCEB_FILE}" ]; then
    echo "ERROR: Input MCEB file not found: ${MCEB_FILE}" >&2
    exit 3
fi

# 3. Fast skip check if output already has cbmsim;2 and mctruth
if [ "${FORCE:-0}" != "1" ] && [ -f "${OUTPUT_FILE}" ] && [ -s "${OUTPUT_FILE}" ]; then
    ALREADY_DONE=$(python3 -c "
import ROOT
ROOT.gROOT.SetBatch(True)
f = ROOT.TFile.Open('${OUTPUT_FILE}', 'READ')
if f and not f.IsZombie():
    has_c2 = bool(f.GetKey('cbmsim', 2))
    has_truth = bool(f.GetKey('mctruth'))
    f.Close()
    print('1' if (has_c2 and has_truth) else '0')
else:
    if f: f.Close()
    print('0')
" 2>/dev/null || echo '0')

    if [ "${ALREADY_DONE}" = "1" ]; then
        echo " ~ [INFO] Output ${OUTPUT_FILE} already contains cbmsim;2 and mctruth. Skipping (set FORCE=1 to rerun)."
        exit 0
    fi
fi

# 4. Run MC Truth Mapping and Transplantation
CONFIG_FILE="${REPO_DIR}/config/mctruth_neutrinos_config.yaml"
MAP_SCRIPT="${REPO_DIR}/scripts/map_mctruth_to_mceb.py"

FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then
    FORCE_FLAG="-f"
fi

echo " ~ [3/4] Running map_mctruth_to_mceb.py..."
python3 -X faulthandler "${MAP_SCRIPT}" \
    -t "${TRUTH_FILE}" \
    -i "${MCEB_FILE}" \
    -o "${OUTPUT_FILE}" \
    -c "${CONFIG_FILE}" \
    ${FORCE_FLAG}
MAP_EXIT=$?

if [ $MAP_EXIT -ne 0 ]; then
    echo "ERROR: map_mctruth_to_mceb.py failed with exit code ${MAP_EXIT}!" >&2
    exit $MAP_EXIT
fi

# 5. Verify final output integrity
echo " ~ [4/4] Verifying output file structure and cycles..."
VERIFY_RESULT=$(python3 -c "
import ROOT
ROOT.gROOT.SetBatch(True)
f = ROOT.TFile.Open('${OUTPUT_FILE}', 'READ')
if not f or f.IsZombie():
    if f: f.Close()
    print('ERROR: Zombie file')
    exit(1)

k_c1 = f.GetKey('cbmsim', 1)
k_c2 = f.GetKey('cbmsim', 2)
k_truth = f.GetKey('mctruth')

if not k_c1:
    f.Close()
    print('ERROR: cbmsim;1 missing')
    exit(2)

if not k_c2:
    f.Close()
    print('ERROR: cbmsim;2 missing')
    exit(3)

if not k_truth:
    f.Close()
    print('ERROR: mctruth tree missing')
    exit(4)

t_cbm = f.Get('cbmsim')
t_truth = f.Get('mctruth')
n_cbm = t_cbm.GetEntries() if t_cbm else 0
n_truth = t_truth.GetEntries() if t_truth else 0

t_cbm.GetEntry(0)
n_mctrack = t_cbm.MCTrack.GetEntries() if hasattr(t_cbm, 'MCTrack') and t_cbm.MCTrack else 0

f.Close()
print(f'OK: cbmsim entries={n_cbm}, mctruth entries={n_truth}, entry0 MCTrack={n_mctrack}')
" 2>&1)
VERIFY_EXIT=$?

if [ $VERIFY_EXIT -ne 0 ]; then
    echo "ERROR: Verification failed: ${VERIFY_RESULT}" >&2
    exit 5
fi

OUT_SIZE=$(stat -c%s "${OUTPUT_FILE}" 2>/dev/null || echo 0)
echo " ~ [SUCCESS] Partition ${PARTITION} truth extraction & transplantation completed at $(date)!"
echo "             Target file: ${OUTPUT_FILE} (${OUT_SIZE} bytes)"
echo "             Status     : ${VERIFY_RESULT}"
