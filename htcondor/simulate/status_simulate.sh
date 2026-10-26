#!/bin/bash

# ==============================================================================
# Helper script to monitor Stage 1: Geant4 Simulation (1000 Partitions) on EOS
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${1:-/eos/user/i/idioniso/snd-numu-charm/data/newP8}"
TOTAL_COUNT=$(wc -l < args_nu14_volTarget_1000.txt 2>/dev/null || echo 1000)

echo "=========================================================="
echo " SND@LHC Stage 1: Simulation Status ($(date '+%Y-%m-%d %H:%M:%S'))"
echo " Destination: ${DATA_DIR}"
echo "=========================================================="

if [ -d "$DATA_DIR" ]; then
    read -r SIM_DONE GEO_DONE < <(
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
            'geofile_full.Genie-TGeant4.root' in entries and os.path.getsize(os.path.join(pdir, 'geofile_full.Genie-TGeant4.root')) > 50000,
        )
    except Exception:
        return (False, False)

with ThreadPoolExecutor(max_workers=32) as ex:
    res = list(ex.map(scan, existing_parts))

print(f'{sum(r[0] for r in res)} {sum(r[1] for r in res)}')
"
    )

    SIM_PCT=$(awk "BEGIN {printf \"%.1f\", ($SIM_DONE / $TOTAL_COUNT) * 100}")
    GEO_PCT=$(awk "BEGIN {printf \"%.1f\", ($GEO_DONE / $TOTAL_COUNT) * 100}")

    echo "Completed Partitions on EOS:"
    echo "  sndLHC.Genie-TGeant4.root       : ${SIM_DONE}/${TOTAL_COUNT} (${SIM_PCT}%)"
    echo "  geofile_full.Genie-TGeant4.root : ${GEO_DONE}/${TOTAL_COUNT} (${GEO_PCT}%)"
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
