#!/bin/bash
# ==============================================================================
# Run Dimuon Prefilter Pipeline on MCEB Files (lxplus / local)
# Cuts: LHC IP1, Stable Beams, Event DeltaT, SciFi-DS Timing, No Veto,
#       DS Dimuon, 2 DS Tracks
# ==============================================================================
set -euo pipefail

REPO_DIR="/afs/cern.ch/work/i/idioniso/snd-numu-charm"
cd "${REPO_DIR}"

WORKERS=${1:-12}
PARTITIONS=${2:-"1-1000"}

DATA_BASE="/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget"
OUT_BASE="/eos/user/i/idioniso/snd-numu-charm/prefiltered/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_MCEB"

mkdir -p "${OUT_BASE}"

echo "=================================================================="
echo " Running MCEB Minimalistic Cutflow Prefilter Pipeline"
echo " Workers:    ${WORKERS}"
echo " Partitions: ${PARTITIONS}"
echo " Config:     config/dimuon_prefilter_config.yaml"
echo "=================================================================="

python3 -X faulthandler scripts/prefilter.py \
  -i "${DATA_BASE}/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root" \
  -o "${OUT_BASE}/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root" \
  --signal-input "${DATA_BASE}/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonDirect.root" \
  --signal-output "${OUT_BASE}/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonDirect.root" \
  --indirect-input "${DATA_BASE}/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonIndirect.root" \
  --indirect-output "${OUT_BASE}/%s/sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonIndirect.root" \
  --cutflow "${OUT_BASE}/cutflow_combined_MCEB.root" \
  --failed-cuts-csv "/eos/user/i/idioniso/snd-numu-charm/signal/2022_down_nu14_MCEB_failed_cuts.csv" \
  --config "config/dimuon_prefilter_config.yaml" \
  -p "${PARTITIONS}" \
  -w "${WORKERS}"
