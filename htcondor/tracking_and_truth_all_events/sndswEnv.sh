#!/bin/bash
# ==============================================================================
# SND@LHC HTCondor & Worker Node Environment Setup
# Direct profile loading bypassing flawed CVMFS setUp.sh lsb_release detection
# ==============================================================================

export WORK_DIR="/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/sw"
export ALIBUILD_ARCH_PREFIX="slc9_x86-64"

# Source official aliBuild profiles for ROOT, FairRoot, FairLogger, Geant4, etc.
if [ -f "${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/sndsw/latest/etc/profile.d/init.sh" ]; then
    source "${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/sndsw/latest/etc/profile.d/init.sh"
fi

if [ -f "${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/Python-modules/latest/etc/profile.d/init.sh" ]; then
    source "${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/Python-modules/latest/etc/profile.d/init.sh"
fi

# Overlay user custom sndsw build and source (for dimuon_DS, isValid checks)
export USER_SNDSW_BUILD="/afs/cern.ch/user/i/idioniso/snd_master/sw3.12/slc9_x86-64/sndsw/master-local1"
export USER_SNDSW_SRC="/afs/cern.ch/user/i/idioniso/snd_master/sndsw"
export ddfUtils="${DDFUTILS:-/afs/cern.ch/user/i/idioniso/ddfUtils}"

# Ensure user custom libraries and scripts are first in path
export SNDSW_ROOT="${USER_SNDSW_BUILD}"
export LD_LIBRARY_PATH="${USER_SNDSW_BUILD}/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${USER_SNDSW_SRC}/python:${USER_SNDSW_SRC}/shipLHC/scripts:${USER_SNDSW_BUILD}/python:${ddfUtils}:${PYTHONPATH}"
