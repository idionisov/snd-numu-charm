#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Worker Script for Stage 1: Geant4 Neutrino Simulation
# Using updated Pythia8 Decayer (DecayConfigPy8.C)
#
# Process per partition:
#   Input : gst.root (+ ghep.root if present) from GENIE volume_volTarget
#   Output: sndLHC.Genie-TGeant4.root + geofile_full.Genie-TGeant4.root
#
# Usage:
#   run_simulate_partition.sh <partition_id> [src_dir_pattern] [dest_dir_pattern]
# ==============================================================================

PARTITION="$1"

if [ -z "$PARTITION" ]; then
    echo "ERROR: Missing required partition ID argument." >&2
    echo "Usage: $0 <partition_id> [src_dir_pattern] [dest_dir_pattern]" >&2
    exit 1
fi

SRC_DIR_PATTERN="${2:-/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2mmRangeCut/sndlhc_15000fb-1_2022_down/nu14/volume_volTarget/%s}"
DEST_DIR_PATTERN="${3:-/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s}"

# Clean up unexpanded HTCondor macro strings if any
if [[ "${SRC_DIR_PATTERN}" == \$* ]] || [ -z "${SRC_DIR_PATTERN}" ]; then
    SRC_DIR_PATTERN="/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2mmRangeCut/sndlhc_15000fb-1_2022_down/nu14/volume_volTarget/%s"
fi
if [[ "${DEST_DIR_PATTERN}" == \$* ]] || [ -z "${DEST_DIR_PATTERN}" ]; then
    DEST_DIR_PATTERN="/eos/user/i/idioniso/snd-numu-charm/data/newP8/%s"
fi

# Resolve paths for this partition
SRC_PART_DIR=$(printf "${SRC_DIR_PATTERN}" "${PARTITION}")
DEST_PART_DIR=$(printf "${DEST_DIR_PATTERN}" "${PARTITION}")

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
echo "         Source partition : ${SRC_PART_DIR}"
echo "         Target partition : ${DEST_PART_DIR}"
echo "         Force mode       : ${FORCE:-0}"

# 2. Check existing simulation outputs if FORCE is not set
DEST_SIM_FILE="${DEST_PART_DIR}/sndLHC.Genie-TGeant4.root"
DEST_GEO_FILE="${DEST_PART_DIR}/geofile_full.Genie-TGeant4.root"

if [ "${FORCE:-0}" != "1" ] && [ -f "${DEST_SIM_FILE}" ] && [ -s "${DEST_SIM_FILE}" ] && [ -f "${DEST_GEO_FILE}" ] && [ -s "${DEST_GEO_FILE}" ]; then
    echo " ~ [INFO] Targets ${DEST_SIM_FILE} and ${DEST_GEO_FILE} already exist and are non-empty. Skipping (set FORCE=1 to rerun)."
    exit 0
fi

# 3. Define local scratch workspace (use HTCondor scratch if present, else local /tmp)
if [ -n "${_CONDOR_SCRATCH_DIR:-}" ] && [ -d "${_CONDOR_SCRATCH_DIR}" ]; then
    SCRATCH_DIR="${_CONDOR_SCRATCH_DIR}/simulate_${PARTITION}"
else
    SCRATCH_DIR="/tmp/p8_simulate_${PARTITION}_$$"
fi

mkdir -p "${SCRATCH_DIR}"
cleanup() {
    local ec=$?
    if [ ${ec} -ne 0 ]; then
        echo " ~ [FAILURE] Simulation pipeline aborted with exit code ${ec}." >&2
        echo "             Scratch workspace retained at: ${SCRATCH_DIR}" >&2
    else
        rm -rf "${SCRATCH_DIR}"
    fi
}
trap cleanup EXIT

# 4. Find and copy input GENIE files
echo " ~ [3/4] Staging input GENIE files from EOS..."
GST_FILE=$(ls "${SRC_PART_DIR}"/*gst.root 2>/dev/null | head -n 1 || true)
GHEP_FILE=$(ls "${SRC_PART_DIR}"/*ghep.root 2>/dev/null | head -n 1 || true)

if [ -z "${GST_FILE}" ] || [ ! -f "${GST_FILE}" ]; then
    echo "ERROR: Input gst.root file not found in ${SRC_PART_DIR}!" >&2
    exit 2
fi

cp -f "${GST_FILE}" "${SCRATCH_DIR}/"
[ -n "${GHEP_FILE}" ] && [ -f "${GHEP_FILE}" ] && cp -f "${GHEP_FILE}" "${SCRATCH_DIR}/"

LOCAL_GST="${SCRATCH_DIR}/$(basename "${GST_FILE}")"
cd "${SCRATCH_DIR}"

# 5. Run Geant4 Simulation
NEVENTS="${NEVENTS:-1000}"
echo " ~ [4/4] Running Geant4 Simulation (Genie 4, events: ${NEVENTS})..."
python3 -X faulthandler "${VMCWORKDIR}/shipLHC/run_simSND.py" \
    --Genie 4 \
    -f "${LOCAL_GST}" \
    -y 2022 \
    -n "${NEVENTS}" \
    -o "${SCRATCH_DIR}" || true

LOCAL_SIM_FILE="${SCRATCH_DIR}/sndLHC.Genie-TGeant4.root"
LOCAL_GEO_FILE="${SCRATCH_DIR}/geofile_full.Genie-TGeant4.root"

if [ ! -s "${LOCAL_SIM_FILE}" ] || [ ! -s "${LOCAL_GEO_FILE}" ]; then
    echo "ERROR: Simulation failed or output files are missing/empty!" >&2
    exit 3
fi

# Verify minimum reasonable file size (> 100 KB)
SIM_SIZE=$(stat -c%s "${LOCAL_SIM_FILE}" 2>/dev/null || echo 0)
GEO_SIZE=$(stat -c%s "${LOCAL_GEO_FILE}" 2>/dev/null || echo 0)
if [ "${SIM_SIZE}" -lt 100000 ] || [ "${GEO_SIZE}" -lt 50000 ]; then
    echo "ERROR: Simulated output files appear truncated (sim: ${SIM_SIZE} B, geo: ${GEO_SIZE} B)!" >&2
    exit 4
fi

# 6. Export outputs to destination partition directory on EOS
echo " ~ Exporting simulated outputs to ${DEST_PART_DIR}..."
mkdir -p "${DEST_PART_DIR}"

# Copy simulation and geometry files
cp -f "${LOCAL_SIM_FILE}" "${DEST_PART_DIR}/"
cp -f "${LOCAL_GEO_FILE}" "${DEST_PART_DIR}/"

# Also copy local GENIE files if not already present in destination
for f_genie in "${SCRATCH_DIR}"/*gst.root "${SCRATCH_DIR}"/*ghep.root; do
    if [ -f "$f_genie" ]; then
        gbname=$(basename "$f_genie")
        if [ ! -f "${DEST_PART_DIR}/${gbname}" ]; then
            cp -f "$f_genie" "${DEST_PART_DIR}/" 2>/dev/null || true
        fi
    fi
done

# Verify final transfer
if [ ! -s "${DEST_SIM_FILE}" ] || [ ! -s "${DEST_GEO_FILE}" ]; then
    echo "ERROR: Destination files missing or empty after copy to ${DEST_PART_DIR}!" >&2
    exit 5
fi

echo " ~ [SUCCESS] Partition ${PARTITION} simulation completed successfully at $(date)!"
echo "             Simulation file: ${DEST_SIM_FILE} (${SIM_SIZE} bytes)"
echo "             Geometry file  : ${DEST_GEO_FILE} (${GEO_SIZE} bytes)"
