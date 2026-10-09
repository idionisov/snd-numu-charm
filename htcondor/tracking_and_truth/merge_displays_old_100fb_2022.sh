#!/bin/bash
set -e

# ==============================================================================
# Merge old 100fb-1 (401 Partitions) event displays into a single master file
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_PATTERN="/eos/user/i/idioniso/snd-numu-charm/event_displays/old_sndlhc_100fb-1_2022_down_volTarget/partitions/*.root"
OUTPUT_FILE="/eos/user/i/idioniso/snd-numu-charm/event_displays/old_sndlhc_100fb-1_2022_down_volTarget_displays.root"

echo "=========================================================="
echo " Merging Old 100fb-1 Displays into Single Master File"
echo " Source : ${SOURCE_PATTERN}"
echo " Output : ${OUTPUT_FILE}"
echo "=========================================================="

python3 "${REPO_DIR}/scripts/merge_event_displays.py" \
    --preset "old_100fb_2022" \
    -i "${SOURCE_PATTERN}" \
    -o "${OUTPUT_FILE}"

echo "=========================================================="
echo " Master event display file ready:"
echo "   ${OUTPUT_FILE}"
echo "=========================================================="
