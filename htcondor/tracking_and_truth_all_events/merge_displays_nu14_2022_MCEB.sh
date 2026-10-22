#!/bin/bash
set -e

# ==============================================================================
# Merge 2022 nu14 MCEB (1000 Partitions) event displays into a single master file
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DISP_BASE="/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_MCEB"

# 1. Merge Direct Signal Displays
DIRECT_PATTERN="${DISP_BASE}/partitions/direct/*.root"
DIRECT_MASTER="${DISP_BASE}_displays_charmDimuonDirect.root"

echo "=========================================================="
echo " Merging 2022 nu14 MCEB Direct Charm Dimuon Displays"
echo " Source : ${DIRECT_PATTERN}"
echo " Output : ${DIRECT_MASTER}"
echo "=========================================================="

if compgen -G "${DIRECT_PATTERN}" > /dev/null; then
    python3 "${REPO_DIR}/scripts/merge_event_displays.py" \
        -i "${DIRECT_PATTERN}" \
        -o "${DIRECT_MASTER}"
    echo " Master direct signal displays: ${DIRECT_MASTER}"
else
    echo " No direct display partition files found to merge."
fi

# 2. Merge Indirect Signal-like Displays
INDIRECT_PATTERN="${DISP_BASE}/partitions/indirect/*.root"
INDIRECT_MASTER="${DISP_BASE}_displays_charmDimuonIndirect.root"

echo "=========================================================="
echo " Merging 2022 nu14 MCEB Indirect Charm Dimuon Displays"
echo " Source : ${INDIRECT_PATTERN}"
echo " Output : ${INDIRECT_MASTER}"
echo "=========================================================="

if compgen -G "${INDIRECT_PATTERN}" > /dev/null; then
    python3 "${REPO_DIR}/scripts/merge_event_displays.py" \
        -i "${INDIRECT_PATTERN}" \
        -o "${INDIRECT_MASTER}"
    echo " Master indirect displays: ${INDIRECT_MASTER}"
else
    echo " No indirect display partition files found to merge."
fi
echo "=========================================================="
