#!/bin/bash

# ==============================================================================
# Helper script to monitor HTCondor DAGMan Workflow & EOS Pipeline Progress
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${1:-/eos/user/i/idioniso/snd-numu-charm/data/newP8}"
TOTAL_COUNT=1000

echo "=========================================================="
echo " SND@LHC Full Pipeline Status ($(date '+%Y-%m-%d %H:%M:%S'))"
echo " Destination: ${DATA_DIR}"
echo "=========================================================="

if [ -d "$DATA_DIR" ]; then
    read -r SIM_DONE MCEB_DONE TRUTH_DONE DIGI_DONE TRACK_DONE < <(
        python3 -c "
import os
from concurrent.futures import ThreadPoolExecutor

base = '${DATA_DIR}'
try:
    existing_parts = [p for p in os.listdir(base) if p.isdigit()]
except Exception:
    existing_parts = []

def scan(p):
    pdir = os.path.join(base, p)
    try:
        entries = set(os.listdir(pdir))
        return (
            'sndLHC.Genie-TGeant4.root' in entries and os.path.getsize(os.path.join(pdir, 'sndLHC.Genie-TGeant4.root')) > 100000,
            'sndLHC.Genie-TGeant4_MCEB.root' in entries and os.path.getsize(os.path.join(pdir, 'sndLHC.Genie-TGeant4_MCEB.root')) > 100000,
            'sndLHC.Genie-TGeant4_MCEB.root' in entries and os.path.getsize(os.path.join(pdir, 'sndLHC.Genie-TGeant4_MCEB.root')) > 1000000,
            'sndLHC.Genie-TGeant4_MCEB_dig.root' in entries and os.path.getsize(os.path.join(pdir, 'sndLHC.Genie-TGeant4_MCEB_dig.root')) > 100000,
            ('sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root' in entries and os.path.getsize(os.path.join(pdir, 'sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root')) > 100000)
            or ('sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root' in entries and os.path.getsize(os.path.join(pdir, 'sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root')) > 100000),
        )
    except Exception:
        return (False, False, False, False, False)

with ThreadPoolExecutor(max_workers=32) as ex:
    res = list(ex.map(scan, existing_parts))

print(f'{sum(r[0] for r in res)} {sum(r[1] for r in res)} {sum(r[2] for r in res)} {sum(r[3] for r in res)} {sum(r[4] for r in res)}')
"
    )

    SIM_PCT=$(awk "BEGIN {printf \"%.1f\", ($SIM_DONE / $TOTAL_COUNT) * 100}")
    MCEB_PCT=$(awk "BEGIN {printf \"%.1f\", ($MCEB_DONE / $TOTAL_COUNT) * 100}")
    TRUTH_PCT=$(awk "BEGIN {printf \"%.1f\", ($TRUTH_DONE / $TOTAL_COUNT) * 100}")
    DIGI_PCT=$(awk "BEGIN {printf \"%.1f\", ($DIGI_DONE / $TOTAL_COUNT) * 100}")
    TRACK_PCT=$(awk "BEGIN {printf \"%.1f\", ($TRACK_DONE / $TOTAL_COUNT) * 100}")

    echo "Pipeline Stages Completed on EOS:"
    echo "  [1/5] Simulation (Genie-TGeant4.root)          : ${SIM_DONE}/${TOTAL_COUNT} (${SIM_PCT}%)"
    echo "  [2/5] Event Builder (MCEB.root)                : ${MCEB_DONE}/${TOTAL_COUNT} (${MCEB_PCT}%)"
    echo "  [3/5] MCTruth Transplanted (MCEB.root > 1MB)   : ${TRUTH_DONE}/${TOTAL_COUNT} (${TRUTH_PCT}%)"
    echo "  [4/5] Digitization (MCEB_dig.root)             : ${DIGI_DONE}/${TOTAL_COUNT} (${DIGI_PCT}%)"
    echo "  [5/5] Dimuon Tracking (2MuTrks_truth.root)     : ${TRACK_DONE}/${TOTAL_COUNT} (${TRACK_PCT}%)"
else
    echo "Directory ${DATA_DIR} not found!"
fi

echo "----------------------------------------------------------"
echo "DAGMan Log Summary:"
LATEST_DAG_OUT=$(ls -t *.dag.dagman.out 2>/dev/null | head -n 1)
if [ -n "$LATEST_DAG_OUT" ] && [ -f "$LATEST_DAG_OUT" ]; then
    echo "  Latest log file: ${LATEST_DAG_OUT}"
    tail -n 8 "${LATEST_DAG_OUT}" | sed 's/^/    /'
else
    echo "  (No active or previous .dagman.out found in ${SCRIPT_DIR})"
fi

echo "----------------------------------------------------------"
echo "HTCondor Queue Status:"
if command -v condor_q >/dev/null 2>&1; then
    condor_q -dag
else
    echo "  (condor_q command not available on this machine)"
fi
echo "=========================================================="
