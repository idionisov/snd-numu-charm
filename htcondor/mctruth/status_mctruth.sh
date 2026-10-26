#!/bin/bash

# ==============================================================================
# Helper script to monitor Stage 3: MC Truth Extraction (1000 Partitions) on EOS
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${1:-/eos/user/i/idioniso/snd-numu-charm/data/newP8}"
TOTAL_COUNT=$(wc -l < args_nu14_volTarget_1000.txt 2>/dev/null || echo 1000)

echo "=========================================================="
echo " SND@LHC Stage 3: MC Truth Extraction Status ($(date '+%Y-%m-%d %H:%M:%S'))"
echo " Destination: ${DATA_DIR}"
echo "=========================================================="

if [ -d "$DATA_DIR" ]; then
    read -r TRUTH_DONE MCEB_AVAIL < <(
        python3 -c "
import os
import ROOT
ROOT.gROOT.SetBatch(True)
from concurrent.futures import ThreadPoolExecutor

base = '${DATA_DIR}'
try:
    existing_parts = [p for p in os.listdir(base) if p.isdigit()]
except Exception:
    existing_parts = []

def scan(p):
    pdir = os.path.join(base, p)
    mceb_f = os.path.join(pdir, 'sndLHC.Genie-TGeant4_MCEB.root')
    has_mceb = os.path.isfile(mceb_f) and os.path.getsize(mceb_f) > 100000
    has_truth_enrich = False

    if has_mceb:
        try:
            f = ROOT.TFile.Open(mceb_f, 'READ')
            if f and not f.IsZombie():
                c2 = bool(f.GetKey('cbmsim', 2))
                tr = bool(f.GetKey('mctruth'))
                has_truth_enrich = (c2 and tr)
            if f:
                f.Close()
        except Exception:
            pass

    return (has_truth_enrich, has_mceb)

with ThreadPoolExecutor(max_workers=32) as ex:
    res = list(ex.map(scan, existing_parts))

print(f'{sum(r[0] for r in res)} {sum(r[1] for r in res)}')
"
    )

    TRUTH_PCT=$(awk "BEGIN {printf \"%.1f\", ($TRUTH_DONE / $TOTAL_COUNT) * 100}")
    MCEB_PCT=$(awk "BEGIN {printf \"%.1f\", ($MCEB_AVAIL / $TOTAL_COUNT) * 100}")

    echo "Partitions Status on EOS:"
    echo "  MCEB Inputs Available (sndLHC.Genie-TGeant4_MCEB.root)     : ${MCEB_AVAIL}/${TOTAL_COUNT} (${MCEB_PCT}%)"
    echo "  Truth Enriched (contains cbmsim;1, cbmsim;2, mctruth)    : ${TRUTH_DONE}/${TOTAL_COUNT} (${TRUTH_PCT}%)"
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
