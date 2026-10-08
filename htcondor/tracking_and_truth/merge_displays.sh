#!/bin/bash
set -e

# ==============================================================================
# Helper script to merge all partition event display ROOT files into one single ROOT file
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_PATTERN="${1:-/eos/user/i/idioniso/snd-numu-charm/event_displays/partitions/*.root}"
OUTPUT_FILE="${2:-/eos/user/i/idioniso/snd-numu-charm/event_displays/numu_dimuon_signal_event_displays.root}"

echo "=========================================================="
echo " Merging Partition Event Displays into Single Master File"
echo " Source Pattern : ${SOURCE_PATTERN}"
echo " Output Target  : ${OUTPUT_FILE}"
echo "=========================================================="

python3 "${REPO_DIR}/scripts/merge_event_displays.py" \
    -i "${SOURCE_PATTERN}" \
    -o "${OUTPUT_FILE}"

echo "=========================================================="
echo " Master event display file ready:"
echo "   ${OUTPUT_FILE}"
echo "=========================================================="
