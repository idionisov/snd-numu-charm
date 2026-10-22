#!/bin/bash

# ==============================================================================
# Helper script to monitor 2022 nu14 MCEB (1000 Partitions) pipeline jobs on EOS
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget"
DISP_DIR="/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_MCEB/partitions"
MASTER_DIRECT="/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_MCEB_displays_charmDimuonDirect.root"
MASTER_INDIRECT="/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_MCEB_displays_charmDimuonIndirect.root"
TOTAL_COUNT=$(wc -l < args_nu14_volTarget_1000.txt 2>/dev/null || echo 1000)

echo "=========================================================="
echo " SND@LHC nu14 2022 MCEB Pipeline Status (All Events) ($(date '+%Y-%m-%d %H:%M:%S'))"
echo " Dataset: sndlhc_15000fb-1_2022_down_nu14_volume_volTarget (MCEB)"
echo "=========================================================="

if [ -d "$DATA_DIR" ]; then
    read -r TRACK_DONE TRUTH_DONE DIRECT_DONE INDIRECT_DONE < <(
        python3 -c "
import os
from concurrent.futures import ThreadPoolExecutor

base = '${DATA_DIR}'
def scan(p):
    pdir = os.path.join(base, str(p))
    try:
        entries = set(os.listdir(pdir))
        return (
            'sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks.root' in entries,
            'sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root' in entries,
            'sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonDirect.root' in entries,
            'sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonIndirect.root' in entries,
        )
    except Exception:
        return (False, False, False, False)

with ThreadPoolExecutor(max_workers=32) as ex:
    res = list(ex.map(scan, range(1, ${TOTAL_COUNT} + 1)))

print(f'{sum(r[0] for r in res)} {sum(r[1] for r in res)} {sum(r[2] for r in res)} {sum(r[3] for r in res)}')
"
    )

    TRACK_PCT=$(awk "BEGIN {printf \"%.1f\", ($TRACK_DONE / $TOTAL_COUNT) * 100}")
    TRUTH_PCT=$(awk "BEGIN {printf \"%.1f\", ($TRUTH_DONE / $TOTAL_COUNT) * 100}")
    DIRECT_PCT=$(awk "BEGIN {printf \"%.1f\", ($DIRECT_DONE / $TOTAL_COUNT) * 100}")
    INDIRECT_PCT=$(awk "BEGIN {printf \"%.1f\", ($INDIRECT_DONE / $TOTAL_COUNT) * 100}")

    echo " Step 1 (Tracked All Events)  : ${TRACK_DONE} / ${TOTAL_COUNT} (${TRACK_PCT}%)"
    echo " Step 2 (Truth Extraction)    : ${TRUTH_DONE} / ${TOTAL_COUNT} (${TRUTH_PCT}%)"
    echo " Step 3a (Direct Skim)        : ${DIRECT_DONE} / ${TOTAL_COUNT} (${DIRECT_PCT}%)"
    echo " Step 3b (Indirect Skim)      : ${INDIRECT_DONE} / ${TOTAL_COUNT} (${INDIRECT_PCT}%)"
else
    echo " Output directory not yet created: ${DATA_DIR}"
fi

if [ -d "$DISP_DIR" ]; then
    DISP_DIR_DONE=$(find "$DISP_DIR/direct" -maxdepth 1 -name "displays_part*.root" -type f -size +500c 2>/dev/null | wc -l)
    DISP_INDIR_DONE=$(find "$DISP_DIR/indirect" -maxdepth 1 -name "displays_part*.root" -type f -size +500c 2>/dev/null | wc -l)
    echo " Step 4a (Direct Displays)    : ${DISP_DIR_DONE} partitions with signal displays"
    echo " Step 4b (Indirect Displays)  : ${DISP_INDIR_DONE} partitions with indirect displays"
fi

if [ -f "$MASTER_DIRECT" ]; then
    SIZE=$(ls -lh "$MASTER_DIRECT" | awk '{print $5}')
    echo " Master Direct Displays       : ${MASTER_DIRECT} (${SIZE})"
fi
if [ -f "$MASTER_INDIRECT" ]; then
    SIZE=$(ls -lh "$MASTER_INDIRECT" | awk '{print $5}')
    echo " Master Indirect Displays     : ${MASTER_INDIRECT} (${SIZE})"
fi

echo "----------------------------------------------------------"

if command -v condor_q >/dev/null 2>&1; then
    echo " HTCondor Queue Summary:"
    condor_q
else
    echo " 'condor_q' not found on this host. Run from lxplus to view queue state."
fi
echo "=========================================================="
