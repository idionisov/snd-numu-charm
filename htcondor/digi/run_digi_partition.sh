#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Worker Script for Stage 4: Digitization
#
# Process per partition:
#   Inputs : sndLHC.Genie-TGeant4_MCEB.root + geofile_full.Genie-TGeant4.root
#            (and sndLHC.Genie-TGeant4.root if raw dig is not yet generated)
#   Outputs: sndLHC.Genie-TGeant4_MCEB_dig.root
#            (and sndLHC.Genie-TGeant4_dig.root if missing)
#
# Usage:
#   run_digi_partition.sh <partition_id> [data_dir_pattern]
# ==============================================================================

PARTITION="$1"

if [ -z "$PARTITION" ]; then
    echo "ERROR: Missing required partition ID argument." >&2
    echo "Usage: $0 <partition_id> [data_dir_pattern]" >&2
    exit 1
fi

DATA_DIR_PATTERN="${2:-/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s}"

# Clean up unexpanded HTCondor macro strings
if [[ "${DATA_DIR_PATTERN}" == \$* ]] || [ -z "${DATA_DIR_PATTERN}" ]; then
    DATA_DIR_PATTERN="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s"
fi

PART_DATA_DIR=$(printf "${DATA_DIR_PATTERN}" "${PARTITION}")

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

echo " ~ [2/4] Environment configured (VMCWORKDIR=${VMCWORKDIR})"
echo "         Date             : $(date)"
echo "         Host             : $(hostname)"
echo "         Partition        : ${PARTITION}"
echo "         Partition data   : ${PART_DATA_DIR}"
echo "         Force mode       : ${FORCE:-0}"

# 2. Fast check: skip if MCEB digitized file already exists and is valid
DEST_MCEB_DIG="${PART_DATA_DIR}/sndLHC.Genie-TGeant4_MCEB_dig.root"
DEST_RAW_DIG="${PART_DATA_DIR}/sndLHC.Genie-TGeant4_dig.root"

if [ "${FORCE:-0}" != "1" ] && [ -f "${DEST_MCEB_DIG}" ] && [ -s "${DEST_MCEB_DIG}" ]; then
    MCEB_DIG_SIZE=$(stat -c%s "${DEST_MCEB_DIG}" 2>/dev/null || echo 0)
    if [ "${MCEB_DIG_SIZE}" -gt 100000 ]; then
        echo " ~ [INFO] Target ${DEST_MCEB_DIG} already exists (${MCEB_DIG_SIZE} B). Skipping (set FORCE=1 to rerun)."
        exit 0
    fi
fi

# 3. Locate input files
GEO_FILE="${PART_DATA_DIR}/geofile_full.Genie-TGeant4.root"
MCEB_FILE="${PART_DATA_DIR}/sndLHC.Genie-TGeant4_MCEB.root"
RAW_SIM_FILE="${PART_DATA_DIR}/sndLHC.Genie-TGeant4.root"

if [ ! -f "${GEO_FILE}" ] || [ ! -s "${GEO_FILE}" ]; then
    echo "ERROR: Prerequisite geometry file not found or empty: ${GEO_FILE}" >&2
    exit 2
fi

if [ ! -f "${MCEB_FILE}" ] || [ ! -s "${MCEB_FILE}" ]; then
    echo "ERROR: Prerequisite MCEB file not found or empty: ${MCEB_FILE}" >&2
    exit 3
fi

# 4. Setup local scratch workspace
if [ -n "${_CONDOR_SCRATCH_DIR:-}" ] && [ -d "${_CONDOR_SCRATCH_DIR}" ]; then
    SCRATCH_DIR="${_CONDOR_SCRATCH_DIR}/digi_${PARTITION}"
else
    SCRATCH_DIR="/tmp/p8_digi_${PARTITION}_$$"
fi

mkdir -p "${SCRATCH_DIR}"
cleanup() {
    local ec=$?
    if [ ${ec} -ne 0 ]; then
        echo " ~ [FAILURE] Digitization pipeline aborted with exit code ${ec}." >&2
        echo "             Scratch workspace retained at: ${SCRATCH_DIR}" >&2
    else
        rm -rf "${SCRATCH_DIR}"
    fi
}
trap cleanup EXIT

# 5. Stage inputs to local scratch
echo " ~ [3/4] Staging files to local scratch..."
cp -f "${GEO_FILE}" "${SCRATCH_DIR}/"
cp -f "${MCEB_FILE}" "${SCRATCH_DIR}/"

LOCAL_GEO="${SCRATCH_DIR}/$(basename "${GEO_FILE}")"
LOCAL_MCEB="${SCRATCH_DIR}/$(basename "${MCEB_FILE}")"
cd "${SCRATCH_DIR}"

# 6. Run Digitization on MCEB file
echo " ~ [4/4] Running Digitization on MCEB file..."
python3 -X faulthandler "${VMCWORKDIR}/shipLHC/run_digiSND.py" \
    -f "${LOCAL_MCEB}" \
    -g "${LOCAL_GEO}" || true

LOCAL_MCEB_DIG="${SCRATCH_DIR}/sndLHC.Genie-TGeant4_MCEB_dig.root"

if [ ! -s "${LOCAL_MCEB_DIG}" ]; then
    echo "ERROR: Digitization of MCEB failed or output is missing/empty!" >&2
    exit 4
fi

# Also digitize raw simulation file if its digitized version is missing on EOS
LOCAL_RAW_DIG=""
if [ ! -s "${DEST_RAW_DIG}" ] && [ -f "${RAW_SIM_FILE}" ] && [ -s "${RAW_SIM_FILE}" ]; then
    echo " ~ [4/4] Digitized raw simulation missing; running Digitization on raw sim..."
    cp -f "${RAW_SIM_FILE}" "${SCRATCH_DIR}/"
    LOCAL_RAW_SIM="${SCRATCH_DIR}/$(basename "${RAW_SIM_FILE}")"
    python3 -X faulthandler "${VMCWORKDIR}/shipLHC/run_digiSND.py" \
        -f "${LOCAL_RAW_SIM}" \
        -g "${LOCAL_GEO}" || true
    LOCAL_RAW_DIG="${SCRATCH_DIR}/sndLHC.Genie-TGeant4_dig.root"
fi

# 7. Verify digitized ROOT tree integrity
VERIFY_OUTPUT=$(python3 -c "
import ROOT
ROOT.gROOT.SetBatch(True)
f = ROOT.TFile.Open('${LOCAL_MCEB_DIG}', 'READ')
if not f or f.IsZombie():
    if f: f.Close()
    print('ERROR: Zombie file')
    exit(1)

t = f.Get('cbmsim')
if not t:
    f.Close()
    print('ERROR: cbmsim missing')
    exit(2)

has_scifi = bool(t.GetBranch('Digi_ScifiHits'))
has_mufi = bool(t.GetBranch('Digi_MuFilterHits'))
entries = t.GetEntries()
f.Close()

if not (has_scifi and has_mufi):
    print(f'ERROR: Missing hit branches (scifi={has_scifi}, mufi={has_mufi})')
    exit(3)

print(f'OK: entries={entries}, scifiHits=True, mufiHits=True')
" 2>&1)
VERIFY_EC=$?

if [ $VERIFY_EC -ne 0 ]; then
    echo "ERROR: Digitized file verification failed: ${VERIFY_OUTPUT}" >&2
    exit 5
fi

# 8. Export finalized outputs to EOS
echo " ~ Exporting digitized file(s) to ${PART_DATA_DIR}..."
mkdir -p "${PART_DATA_DIR}"
cp -f "${LOCAL_MCEB_DIG}" "${DEST_MCEB_DIG}"

if [ -n "${LOCAL_RAW_DIG}" ] && [ -s "${LOCAL_RAW_DIG}" ]; then
    cp -f "${LOCAL_RAW_DIG}" "${DEST_RAW_DIG}"
fi

if [ ! -s "${DEST_MCEB_DIG}" ]; then
    echo "ERROR: Destination file missing or empty after copy: ${DEST_MCEB_DIG}" >&2
    exit 6
fi

FINAL_SIZE=$(stat -c%s "${DEST_MCEB_DIG}" 2>/dev/null || echo 0)
echo " ~ [SUCCESS] Partition ${PARTITION} digitization completed at $(date)!"
echo "             MCEB dig file: ${DEST_MCEB_DIG} (${FINAL_SIZE} bytes)"
echo "             Status       : ${VERIFY_OUTPUT}"
