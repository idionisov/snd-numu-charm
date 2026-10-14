#!/bin/bash
# ==============================================================================
# SND@LHC HTCondor & Worker Node Environment Setup
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

# Essential SNDSW variables required by shipLHC and tracking scripts
export EOSSHIP="root://eospublic.cern.ch/"
export USER_SNDSW_BUILD="/afs/cern.ch/user/i/idioniso/snd_master/sw3.12/slc9_x86-64/sndsw/master-local1"
export USER_SNDSW_SRC="/afs/cern.ch/user/i/idioniso/snd_master/sndsw"
export ddfUtils="${DDFUTILS:-/afs/cern.ch/user/i/idioniso/ddfUtils}"

export SNDSW_ROOT="${USER_SNDSW_BUILD}"
export FAIRSHIP="${USER_SNDSW_BUILD}"
export VMCWORKDIR="${USER_SNDSW_BUILD}"
export FAIRLIBDIR="${USER_SNDSW_BUILD}/lib"
export FAIRROOTPATH="${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/FairRoot/latest"
export GEANT4_VMC_ROOT="${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/GEANT4_VMC/v6-1-p2-snd-local1"
export GENIE_ROOT="${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/GENIE/v3.2.0-local1"
export LHAPDF_ROOT="${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/lhapdf/lhapdf-6.5.3-snd-local1"

# ROOT include paths for C++ JIT compilation (KalmanFitterInfo.h, FairRoot, etc.)
export ROOT_INCLUDE_PATH="${USER_SNDSW_BUILD}/include:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/sndsw/master-local1/include:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/FairRoot/v19.0.0-local1/include:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/FairMQ/v1.9.1-local1/include/fairmq:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/asio/v1.19.1-local1/include:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/FairLogger/v1.9.0-local1/include:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/fmt/7.1.0-local1/include:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/GEANT3/v4-1-local1/include/TGeant3:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/GEANT4_VMC/v6-1-p2-snd-local1/include/g4root:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/GEANT4_VMC/v6-1-p2-snd-local1/include/geant4vmc:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/VMC/v2-0-local1/include/vmc:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/GENIE/v3.2.0-local1/genie/inc:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/GENIE/v3.2.0-local1/genie/src:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/GEANT4/v11.1.0-snd-local1/include:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/GEANT4/v11.1.0-snd-local1/include/Geant4:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/pythia/sndsw-pythia8309-local1/include:${WORK_DIR}/${ALIBUILD_ARCH_PREFIX}/pythia/sndsw-pythia8309-local1/include/Pythia8:${ROOT_INCLUDE_PATH}"

# Library and Python paths
export LD_LIBRARY_PATH="${USER_SNDSW_BUILD}/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${USER_SNDSW_SRC}/python:${USER_SNDSW_SRC}/shipLHC/scripts:${USER_SNDSW_BUILD}/python:${ddfUtils}:${PYTHONPATH}"
