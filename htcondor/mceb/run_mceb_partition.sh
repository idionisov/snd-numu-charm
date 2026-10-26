#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Worker Script for Stage 2: Monte Carlo Event Builder (MCEB)
#
# Process per partition:
#   Input : sndLHC.Genie-TGeant4.root + geofile_full.Genie-TGeant4.root
#   Output: sndLHC.Genie-TGeant4_MCEB.root
#
# Usage:
#   run_mceb_partition.sh <partition_id> [data_dir_pattern]
# ==============================================================================

PARTITION="$1"

if [ -z "$PARTITION" ]; then
    echo "ERROR: Missing required partition ID argument." >&2
    echo "Usage: $0 <partition_id> [data_dir_pattern]" >&2
    exit 1
fi

DATA_DIR_PATTERN="${2:-/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s}"

# Clean up unexpanded HTCondor macro strings if any
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

# 2. Check existing MCEB output if FORCE is not set
DEST_MCEB_FILE="${PART_DATA_DIR}/sndLHC.Genie-TGeant4_MCEB.root"

if [ "${FORCE:-0}" != "1" ] && [ -f "${DEST_MCEB_FILE}" ] && [ -s "${DEST_MCEB_FILE}" ]; then
    # Verify non-trivial size (> 100 KB)
    CUR_SIZE=$(stat -c%s "${DEST_MCEB_FILE}" 2>/dev/null || echo 0)
    if [ "${CUR_SIZE}" -gt 100000 ]; then
        echo " ~ [INFO] Target ${DEST_MCEB_FILE} already exists (${CUR_SIZE} B). Skipping (set FORCE=1 to rerun)."
        exit 0
    fi
fi

# 3. Locate input simulation and geometry files
SRC_SIM_FILE="${PART_DATA_DIR}/sndLHC.Genie-TGeant4.root"
SRC_GEO_FILE="${PART_DATA_DIR}/geofile_full.Genie-TGeant4.root"

if [ ! -f "${SRC_SIM_FILE}" ] || [ ! -s "${SRC_SIM_FILE}" ]; then
    echo "ERROR: Input simulation file not found or empty: ${SRC_SIM_FILE}" >&2
    exit 2
fi

if [ ! -f "${SRC_GEO_FILE}" ] || [ ! -s "${SRC_GEO_FILE}" ]; then
    echo "ERROR: Input geometry file not found or empty: ${SRC_GEO_FILE}" >&2
    exit 3
fi

# 4. Define local scratch workspace (use HTCondor scratch if present, else local /tmp)
if [ -n "${_CONDOR_SCRATCH_DIR:-}" ] && [ -d "${_CONDOR_SCRATCH_DIR}" ]; then
    SCRATCH_DIR="${_CONDOR_SCRATCH_DIR}/mceb_${PARTITION}"
else
    SCRATCH_DIR="/tmp/p8_mceb_${PARTITION}_$$"
fi

mkdir -p "${SCRATCH_DIR}"
cleanup() {
    local ec=$?
    if [ ${ec} -ne 0 ]; then
        echo " ~ [FAILURE] MCEB pipeline aborted with exit code ${ec}." >&2
        echo "             Scratch workspace retained at: ${SCRATCH_DIR}" >&2
    else
        rm -rf "${SCRATCH_DIR}"
    fi
}
trap cleanup EXIT

# 5. Stage inputs to local scratch for fast I/O
echo " ~ [3/4] Staging simulation files to local scratch..."
cp -f "${SRC_SIM_FILE}" "${SCRATCH_DIR}/"
cp -f "${SRC_GEO_FILE}" "${SCRATCH_DIR}/"

LOCAL_SIM="${SCRATCH_DIR}/$(basename "${SRC_SIM_FILE}")"
LOCAL_GEO="${SCRATCH_DIR}/$(basename "${SRC_GEO_FILE}")"
LOCAL_MCEB="${SCRATCH_DIR}/sndLHC.Genie-TGeant4_MCEB.root"

cd "${SCRATCH_DIR}"

# 6. Run MCEventBuilder
NEVENTS="${NEVENTS:-0}"
EXTRA_MCEB_ARGS=()
if [ -n "${NEVENTS}" ] && [ "${NEVENTS}" -gt 0 ]; then
    EXTRA_MCEB_ARGS+=("-n" "${NEVENTS}")
fi

echo " ~ [4/4] Running MCEventBuilder..."
python3 -X faulthandler "${VMCWORKDIR}/shipLHC/run_MCEventBuilder.py" \
    -f "${LOCAL_SIM}" \
    -g "${LOCAL_GEO}" \
    -o "${LOCAL_MCEB}" \
    "${EXTRA_MCEB_ARGS[@]}" || true

if [ ! -s "${LOCAL_MCEB}" ]; then
    echo "ERROR: MCEventBuilder execution failed or output is missing/empty!" >&2
    exit 4
fi

MCEB_SIZE=$(stat -c%s "${LOCAL_MCEB}" 2>/dev/null || echo 0)
if [ "${MCEB_SIZE}" -lt 100000 ]; then
    echo "ERROR: MCEB output appears truncated (${MCEB_SIZE} bytes)!" >&2
    exit 5
fi

# Quick ROOT tree entry verification
N_ENTRIES=$(python3 -c "
import ROOT
ROOT.gROOT.SetBatch(True)
f = ROOT.TFile.Open('${LOCAL_MCEB}', 'READ')
t = f.Get('cbmsim') if f and not f.IsZombie() else None
print(t.GetEntries() if t else -1)
if f: f.Close()
" 2>/dev/null || echo -1)

if [ "${N_ENTRIES}" -le 0 ]; then
    echo "ERROR: MCEB output tree 'cbmsim' is missing or has 0 entries (got: ${N_ENTRIES})!" >&2
    exit 6
fi

# 7. Export outputs to destination partition directory on EOS
echo " ~ Exporting MCEB output to ${DEST_MCEB_FILE}..."
mkdir -p "${PART_DATA_DIR}"
cp -f "${LOCAL_MCEB}" "${DEST_MCEB_FILE}"

if [ ! -s "${DEST_MCEB_FILE}" ]; then
    echo "ERROR: Destination file missing or empty after copy to ${DEST_MCEB_FILE}!" >&2
    exit 7
fi

echo " ~ [SUCCESS] Partition ${PARTITION} MCEB completed successfully at $(date)!"
echo "             MCEB file   : ${DEST_MCEB_FILE} (${MCEB_SIZE} bytes, ${N_ENTRIES} entries)"
