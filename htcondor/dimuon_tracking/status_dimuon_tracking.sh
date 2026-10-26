#!/bin/bash

# ==============================================================================
# Helper script to monitor Stage 5: Dimuon Track Reconstruction & Skimming on EOS
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${1:-/eos/user/i/idioniso/snd-numu-charm/data/newP8}"
DISP_DIR="${2:-/eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions}"
TOTAL_COUNT=$(wc -l < args_nu14_volTarget_1000.txt 2>/dev/null || echo 1000)

echo "=========================================================="
echo " SND@LHC Stage 5: Dimuon Tracking Status ($(date '+%Y-%m-%d %H:%M:%S'))"
echo " Data Destination   : ${DATA_DIR}"
echo " Display Destination: ${DISP_DIR}"
echo "=========================================================="

if [ -d "$DATA_DIR" ]; then
    read -r DIG_AVAIL TRACK_DONE DIRECT_DONE INDIRECT_DONE < <(
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
        has_dig = 'sndLHC.Genie-TGeant4_MCEB_dig.root' in entries and os.path.getsize(os.path.join(pdir, 'sndLHC.Genie-TGeant4_MCEB_dig.root')) > 100000
        has_track = (
            ('sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root' in entries and os.path.getsize(os.path.join(pdir, 'sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root')) > 100000)
            or ('sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root' in entries and os.path.getsize(os.path.join(pdir, 'sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root')) > 100000)
        )
        has_direct = any(f.endswith('_charmDimuonDirect.root') and os.path.getsize(os.path.join(pdir, f)) > 1000 for f in entries)
        has_indirect = any(f.endswith('_charmDimuonIndirect.root') and os.path.getsize(os.path.join(pdir, f)) > 1000 for f in entries)
        return (has_dig, has_track, has_direct, has_indirect)
    except Exception:
        return (False, False, False, False)

with ThreadPoolExecutor(max_workers=32) as ex:
    res = list(ex.map(scan, existing_parts))

print(f'{sum(r[0] for r in res)} {sum(r[1] for r in res)} {sum(r[2] for r in res)} {sum(r[3] for r in res)}')
"
    )

    DIG_PCT=$(awk "BEGIN {printf \"%.1f\", ($DIG_AVAIL / $TOTAL_COUNT) * 100}")
    TRACK_PCT=$(awk "BEGIN {printf \"%.1f\", ($TRACK_DONE / $TOTAL_COUNT) * 100}")
    DIRECT_PCT=$(awk "BEGIN {printf \"%.1f\", ($DIRECT_DONE / $TOTAL_COUNT) * 100}")
    INDIRECT_PCT=$(awk "BEGIN {printf \"%.1f\", ($INDIRECT_DONE / $TOTAL_COUNT) * 100}")

    echo "Partitions Status on EOS:"
    echo "  Digitized Inputs Available (MCEB_dig.root)      : ${DIG_AVAIL}/${TOTAL_COUNT} (${DIG_PCT}%)"
    echo "  Dimuon Tracking Completed (2MuTrks_truth.root)  : ${TRACK_DONE}/${TOTAL_COUNT} (${TRACK_PCT}%)"
    echo "  Direct Charm Skims Created (Direct.root)        : ${DIRECT_DONE}/${TOTAL_COUNT} (${DIRECT_PCT}%)"
    echo "  Indirect Charm Skims Created (Indirect.root)    : ${INDIRECT_DONE}/${TOTAL_COUNT} (${INDIRECT_PCT}%)"
else
    echo "Directory ${DATA_DIR} not found!"
fi

if [ -d "$DISP_DIR" ]; then
    DIRECT_DISP_COUNT=$(find "${DISP_DIR}/direct" -maxdepth 1 -name "*.root" 2>/dev/null | wc -l)
    INDIRECT_DISP_COUNT=$(find "${DISP_DIR}/indirect" -maxdepth 1 -name "*.root" 2>/dev/null | wc -l)
    echo "Event Displays Created:"
    echo "  Direct Signal Displays      : ${DIRECT_DISP_COUNT} partitions"
    echo "  Indirect Displays           : ${INDIRECT_DISP_COUNT} partitions"
fi

echo "----------------------------------------------------------"
echo "HTCondor Queue Status:"
if command -v condor_q >/dev/null 2>&1; then
    condor_q
else
    echo "  (condor_q command not available on this machine)"
fi
echo "=========================================================="
