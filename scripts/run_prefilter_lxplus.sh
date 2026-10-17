#!/bin/bash
# ==============================================================================
# Run Dimuon Prefilter Pipeline on lxplus (Partitions 1-1000)
# ==============================================================================
set -euo pipefail

# 1. Ensure working directory
REPO_DIR="/afs/cern.ch/work/i/idioniso/snd-numu-charm"
cd "${REPO_DIR}"

# 2. Number of parallel worker processes (adjust to available cores, e.g. 8-16)
WORKERS=${1:-12}
PARTITIONS=${2:-"1-1000"}

echo "Running prefilter pipeline on lxplus..."
echo "Workers:    ${WORKERS}"
echo "Partitions: ${PARTITIONS}"

python3 -X faulthandler scripts/prefilter.py \
  -i "/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root" \
  -o "/eos/user/i/idioniso/snd-numu-charm/prefiltered/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root" \
  --signal-input "/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_dig_2MuTrks_truth_charmDimuonDirect.root" \
  --signal-output "/eos/user/i/idioniso/snd-numu-charm/prefiltered/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/%s/sndLHC.Genie-TGeant4_dig_2MuTrks_truth_charmDimuonDirect.root" \
  --failed-cuts-csv "/eos/user/i/idioniso/snd-numu-charm/signal/2022_down_nu14_failed_cuts.csv" \
  --config "config/dimuon_prefilter_config.yaml" \
  -p "${PARTITIONS}" \
  -w "${WORKERS}" \
  --skip-existing
