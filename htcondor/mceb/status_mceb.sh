#!/bin/bash

# ==============================================================================
# Helper script to monitor Stage 2: MCEventBuilder (1000 Partitions) on EOS
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${1:-/eos/user/i/idioniso/snd-numu-charm/data/newP8}"
TOTAL_COUNT=$(wc -l < args_nu14_volTarget_1000.txt 2>/dev/null || echo 1000)

echo "=========================================================="
echo " SND@LHC Stage 2: MCEventBuilder Status ($(date '+%Y-%m-%d %H:%M:%S'))"
echo " Destination: ${DATA_DIR}"
echo "=========================================================="

if [ -d "$DATA_DIR" ]; then
    read -r SIM_DONE MCEB_DONE < <(
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
        )
    except Exception:
        return (False, False)

with ThreadPoolExecutor(max_workers=32) as ex:
    res = list(ex.map(scan, existing_parts))

print(f'{sum(r[0] for r in res)} {sum(r[1] for r in res)}')
"
    )

    SIM_PCT=$(awk "BEGIN {printf \"%.1f\", ($SIM_DONE / $TOTAL_COUNT) * 100}")
    MCEB_PCT=$(awk "BEGIN {printf \"%.1f\", ($MCEB_DONE / $TOTAL_COUNT) * 100}")

    echo "Partitions Status on EOS:"
    echo "  Inputs Available (sndLHC.Genie-TGeant4.root) : ${SIM_DONE}/${TOTAL_COUNT} (${SIM_PCT}%)"
    echo "  MCEB Completed   (sndLHC.Genie-TGeant4_MCEB.root) : ${MCEB_DONE}/${TOTAL_COUNT} (${MCEB_PCT}%)"
else
    echo "Directory ${DATA_DIR} not found!"
fi

echo "----------------------------------------------------------"
echo "HTCondor Queue Status:"
if command -v condor_q >/dev/null 2>&1; then
    condor_q
else
    echo "  (condor_q command not available on this machine)"
fi
echo "=========================================================="
