#!/bin/bash

# ==============================================================================
# Helper script to monitor 2022 nu14 Pythia8 Simulation (1000 Partitions) on EOS
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${1:-/eos/user/i/idioniso/snd-numu-charm/data/newP8}"
TOTAL_COUNT=$(wc -l < args_nu14_volTarget_1000.txt 2>/dev/null || echo 1000)

echo "=========================================================="
echo " SND@LHC nu14 2022 Pythia8 Simulation Pipeline Status ($(date '+%Y-%m-%d %H:%M:%S'))"
echo " Destination: ${DATA_DIR}"
echo "=========================================================="

if [ -d "$DATA_DIR" ]; then
    read -r SIM_DONE MCEB_DONE DIGI_DONE MCEB_DIGI_DONE < <(
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
            'sndLHC.Genie-TGeant4.root' in entries,
            'sndLHC.Genie-TGeant4_MCEB.root' in entries,
            'sndLHC.Genie-TGeant4_dig.root' in entries,
            'sndLHC.Genie-TGeant4_MCEB_dig.root' in entries,
        )
    except Exception:
        return (False, False, False, False)

with ThreadPoolExecutor(max_workers=32) as ex:
    res = list(ex.map(scan, existing_parts))

print(f'{sum(r[0] for r in res)} {sum(r[1] for r in res)} {sum(r[2] for r in res)} {sum(r[3] for r in res)}')
"
    )

    SIM_PCT=$(awk "BEGIN {printf \"%.1f\", ($SIM_DONE / $TOTAL_COUNT) * 100}")
    MCEB_PCT=$(awk "BEGIN {printf \"%.1f\", ($MCEB_DONE / $TOTAL_COUNT) * 100}")
    DIGI_PCT=$(awk "BEGIN {printf \"%.1f\", ($DIGI_DONE / $TOTAL_COUNT) * 100}")
    MCEB_DIGI_PCT=$(awk "BEGIN {printf \"%.1f\", ($MCEB_DIGI_DONE / $TOTAL_COUNT) * 100}")

    echo " Pipeline Progress:"
    echo "   1. Geant4 Sim       : ${SIM_DONE}/${TOTAL_COUNT} (${SIM_PCT}%)"
    echo "   2. MCEventBuilder   : ${MCEB_DONE}/${TOTAL_COUNT} (${MCEB_PCT}%)"
    echo "   3. Digitization     : ${DIGI_DONE}/${TOTAL_COUNT} (${DIGI_PCT}%)"
    echo "   4. MCEB Dig (Final) : ${MCEB_DIGI_DONE}/${TOTAL_COUNT} (${MCEB_DIGI_PCT}%)"
else
    echo " Destination directory not found: ${DATA_DIR}"
fi

echo "=========================================================="
if command -v condor_q >/dev/null 2>&1; then
    echo " HTCondor Queue Status:"
    condor_q -nobatch 2>&1 | tail -n 10
else
    echo " HTCondor queue summary not available locally (run on lxplus)."
fi
echo "=========================================================="
