#!/bin/bash
set -eo pipefail

# ==============================================================================
# SND@LHC: HTCondor Worker Script for Simulation + MCEB + Digitization
# Using updated Pythia8 Decayer (DecayConfigPy8.C)
#
# Pipeline per partition:
#   1) Geant4 Simulation: gst.root -> sndLHC.Genie-TGeant4.root + geofile_full
#   2) MCEventBuilder:    sndLHC.Genie-TGeant4.root -> sndLHC.Genie-TGeant4_MCEB.root
#   3) Digitization:      sndLHC.Genie-TGeant4.root -> sndLHC.Genie-TGeant4_dig.root
#   4) MCEB Digitization: sndLHC.Genie-TGeant4_MCEB.root -> sndLHC.Genie-TGeant4_MCEB_dig.root
#
# Usage:
#   run_sim_partition.sh <partition_id> [src_dir_pattern] [dest_dir_pattern]
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

# Source environment
if [ -f "${SCRIPT_DIR}/sndswEnv.sh" ]; then
    echo " ~ [1/5] Sourcing local environment: ${SCRIPT_DIR}/sndswEnv.sh"
    source "${SCRIPT_DIR}/sndswEnv.sh"
elif [ -f "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh" ]; then
    echo " ~ [1/5] Sourcing CVMFS stack: /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
    source "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh"
fi

echo " ~ [2/5] Environment configured (VMCWORKDIR=${VMCWORKDIR})"
echo "         Source partition : ${SRC_PART_DIR}"
echo "         Target partition : ${DEST_PART_DIR}"

# Check existing completed files if FORCE is not set
FINAL_MCEB_DIG="${DEST_PART_DIR}/sndLHC.Genie-TGeant4_MCEB_dig.root"
if [ "${FORCE:-0}" != "1" ] && [ -f "${FINAL_MCEB_DIG}" ] && [ -s "${FINAL_MCEB_DIG}" ]; then
    echo " ~ [INFO] Target ${FINAL_MCEB_DIG} already exists and is non-empty. Skipping (set FORCE=1 to rerun)."
    exit 0
fi

# Define local scratch workspace (use HTCondor scratch if present, else local /tmp)
if [ -n "${_CONDOR_SCRATCH_DIR:-}" ] && [ -d "${_CONDOR_SCRATCH_DIR}" ]; then
    SCRATCH_DIR="${_CONDOR_SCRATCH_DIR}/sim_${PARTITION}"
else
    SCRATCH_DIR="/tmp/p8_sim_${PARTITION}_$$"
fi

mkdir -p "${SCRATCH_DIR}"
# Clean up trap: on failure, retain scratch directory so logs/outputs are not lost
cleanup() {
    local ec=$?
    if [ ${ec} -ne 0 ]; then
        echo " ~ [FAILURE] Pipeline aborted with exit code ${ec}." >&2
        echo "             Scratch workspace retained at: ${SCRATCH_DIR}" >&2
    else
        rm -rf "${SCRATCH_DIR}"
    fi
}
trap cleanup EXIT

# Find and copy input GENIE files
echo " ~ [3/5] Staging input files from EOS..."
GST_FILE=$(ls "${SRC_PART_DIR}"/*gst.root 2>/dev/null | head -n 1 || true)
GHEP_FILE=$(ls "${SRC_PART_DIR}"/*ghep.root 2>/dev/null | head -n 1 || true)

if [ -z "${GST_FILE}" ] || [ ! -f "${GST_FILE}" ]; then
    echo "ERROR: Input gst file not found in ${SRC_PART_DIR}!" >&2
    exit 2
fi

cp -f "${GST_FILE}" "${SCRATCH_DIR}/"
[ -n "${GHEP_FILE}" ] && [ -f "${GHEP_FILE}" ] && cp -f "${GHEP_FILE}" "${SCRATCH_DIR}/"

LOCAL_GST="${SCRATCH_DIR}/$(basename "${GST_FILE}")"
cd "${SCRATCH_DIR}"

# Step 1: Geant4 simulation with updated DecayConfigPy8.C
NEVENTS="${NEVENTS:-1000}"
echo " ~ [4/5] [Step 1/4] Running Geant4 Simulation (Genie 4, events: ${NEVENTS})..."
python3 -X faulthandler "${VMCWORKDIR}/shipLHC/run_simSND.py" \
    --Genie 4 \
    -f "${LOCAL_GST}" \
    -y 2022 \
    -n "${NEVENTS}" \
    -o "${SCRATCH_DIR}" || true

SIM_FILE="${SCRATCH_DIR}/sndLHC.Genie-TGeant4.root"
GEO_FILE="${SCRATCH_DIR}/geofile_full.Genie-TGeant4.root"

if [ ! -s "${SIM_FILE}" ] || [ ! -s "${GEO_FILE}" ]; then
    echo "ERROR: Simulation failed or output files are empty!" >&2
    exit 3
fi

# Step 2: MCEventBuilder
MCEB_FILE="${SCRATCH_DIR}/sndLHC.Genie-TGeant4_MCEB.root"
SAVE_FIRST25_ONLY="${SAVE_FIRST25_ONLY:-1}"
EXTRA_MCEB_ARGS=()
if [ "${SAVE_FIRST25_ONLY}" = "1" ] || [ "${SAVE_FIRST25_ONLY}" = "true" ]; then
    EXTRA_MCEB_ARGS+=("--saveFirst25nsOnly")
fi
echo " ~ [4/5] [Step 2/4] Running MCEventBuilder (saveFirst25nsOnly: ${SAVE_FIRST25_ONLY})..."
python3 -X faulthandler "${VMCWORKDIR}/shipLHC/run_MCEventBuilder.py" \
    -f "${SIM_FILE}" \
    -g "${GEO_FILE}" \
    -o "${MCEB_FILE}" \
    "${EXTRA_MCEB_ARGS[@]}" || true

if [ ! -s "${MCEB_FILE}" ]; then
    echo "ERROR: MCEventBuilder failed!" >&2
    exit 4
fi

# Step 3: Digitization (Raw Simulation)
echo " ~ [4/5] [Step 3/4] Running Digitization on raw sim..."
python3 -X faulthandler "${VMCWORKDIR}/shipLHC/run_digiSND.py" \
    -f "${SIM_FILE}" \
    -g "${GEO_FILE}" || true

if [ ! -s "${SCRATCH_DIR}/sndLHC.Genie-TGeant4_dig.root" ]; then
    echo "ERROR: Digitization of raw sim failed!" >&2
    exit 5
fi

# Step 4: Digitization (MCEB)
echo " ~ [4/5] [Step 4/4] Running Digitization on MCEB..."
python3 -X faulthandler "${VMCWORKDIR}/shipLHC/run_digiSND.py" \
    -f "${MCEB_FILE}" \
    -g "${GEO_FILE}" || true

if [ ! -s "${SCRATCH_DIR}/sndLHC.Genie-TGeant4_MCEB_dig.root" ]; then
    echo "ERROR: Digitization of MCEB failed!" >&2
    exit 6
fi

# Verify all expected outputs exist in scratch
REQUIRED_OUTPUTS=(
    "geofile_full.Genie-TGeant4.root"
    "sndLHC.Genie-TGeant4.root"
    "sndLHC.Genie-TGeant4_dig.root"
    "sndLHC.Genie-TGeant4_MCEB.root"
    "sndLHC.Genie-TGeant4_MCEB_dig.root"
)

for out_name in "${REQUIRED_OUTPUTS[@]}"; do
    if [ ! -s "${SCRATCH_DIR}/${out_name}" ]; then
        echo "ERROR: Expected output ${out_name} is missing or empty!" >&2
        exit 7
    fi
done

# Step 5: Transfer to destination partition directory
echo " ~ [5/5] Exporting outputs to ${DEST_PART_DIR}..."
mkdir -p "${DEST_PART_DIR}"
cp -f "${SCRATCH_DIR}"/*.root "${DEST_PART_DIR}/"

# Verify all files exist in destination
for out_name in "${REQUIRED_OUTPUTS[@]}"; do
    if [ ! -s "${DEST_PART_DIR}/${out_name}" ]; then
        echo "ERROR: Destination file ${DEST_PART_DIR}/${out_name} is missing or empty after copy!" >&2
        exit 8
    fi
done

echo " ~ [SUCCESS] Partition ${PARTITION} fully simulated and digitized at $(date)!"
