#!/bin/bash
# ==============================================================================
# Monitor HTCondor jobs for Stage 6: Event Prefiltering
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== HTCondor Queue (Prefilter Jobs) ==="
if command -v condor_q >/dev/null 2>&1; then
    condor_q -constraint 'JobBatchName == "prefilter.sub" || Cmd == "'"${SCRIPT_DIR}"'/run_prefilter_partition.sh"'
else
    echo "condor_q not found (run on lxplus submit node)."
fi

echo ""
echo "=== Output Directories Summary ==="
OUT_RUN6596="/eos/user/i/idioniso/snd-numu-charm/prefiltered/2023/b2nob1Muons/run006596"
OUT_MAIN="/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8/essentials"
OUT_DIR="/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8/essentials-direct"
OUT_IND="/eos/user/i/idioniso/snd-numu-charm/prefiltered/newP8/essentials-indirect"

count_files() {
    local dir="$1"
    if [ -d "$dir" ]; then
        find "$dir" -maxdepth 2 -name "*.root" 2>/dev/null | wc -l
    else
        echo 0
    fi
}

if [ -d "$OUT_RUN6596" ]; then
    echo "Run 6596 (b2nob1Muons) : $(count_files "$OUT_RUN6596") / 287 files"
fi
if [ -d "$OUT_MAIN" ]; then
    echo "Main MC filtered files : $(count_files "$OUT_MAIN") files"
    echo "Direct MC filtered     : $(count_files "$OUT_DIR") files"
    echo "Indirect MC filtered   : $(count_files "$OUT_IND") files"
fi
