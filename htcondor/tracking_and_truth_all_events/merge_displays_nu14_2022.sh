#!/bin/bash
set -e

# ==============================================================================
# Merge 2022 nu14 (1000 Partitions) event displays into a single master file
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_PATTERN="/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/partitions/*.root"
OUTPUT_FILE="/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_displays.root"

echo "=========================================================="
echo " Merging 2022 nu14 Displays into Single Master File"
echo " Source : ${SOURCE_PATTERN}"
echo " Output : ${OUTPUT_FILE}"
echo "=========================================================="

python3 "${REPO_DIR}/scripts/merge_event_displays.py" \
    -i "${SOURCE_PATTERN}" \
    -o "${OUTPUT_FILE}"

echo "=========================================================="
echo " Master event display file ready:"
echo "   ${OUTPUT_FILE}"
echo "=========================================================="
