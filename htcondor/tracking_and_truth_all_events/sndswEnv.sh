#!/bin/bash
# ==============================================================================
# SND@LHC HTCondor & Worker Node Complete Environment Setup
# Self-contained environment with full library and python paths for clean shells
# ==============================================================================

CVMFS_SW="/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/sw/slc9_x86-64"
USER_SNDSW_BUILD="/afs/cern.ch/user/i/idioniso/snd_master/sw3.12/slc9_x86-64/sndsw/master-local1"
USER_SNDSW_SRC="/afs/cern.ch/user/i/idioniso/snd_master/sndsw"

# 1. Base SNDSW & FairRoot Core Paths
export EOSSHIP="root://eospublic.cern.ch/"
export USER_SNDSW_BUILD="${USER_SNDSW_BUILD}"
export USER_SNDSW_SRC="${USER_SNDSW_SRC}"
export SNDSW_ROOT="${USER_SNDSW_BUILD}"
export FAIRSHIP="${USER_SNDSW_BUILD}"
export FAIRSHIP_ROOT="${USER_SNDSW_BUILD}"
export FAIRLIBDIR="${USER_SNDSW_BUILD}/lib"
export VMCWORKDIR="${USER_SNDSW_BUILD}"
export GEOMPATH="${USER_SNDSW_BUILD}/geometry"
export CONFIG_DIR="${USER_SNDSW_BUILD}/gconfig"
export GALGCONF="${USER_SNDSW_BUILD}/shipgen/genie_config"

export FAIRROOTPATH="${CVMFS_SW}/FairRoot/latest"
export ROOTSYS="${CVMFS_SW}/ROOT/v6-36-04-alice1-local1"
export ROOT_ROOT="${CVMFS_SW}/ROOT/v6-36-04-alice1-local1"
export GEANT4_VMC_ROOT="${CVMFS_SW}/GEANT4_VMC/v6-1-p2-snd-local1"
export GENIE_ROOT="${CVMFS_SW}/GENIE/v3.2.0-local1"
export LHAPDF_ROOT="${CVMFS_SW}/lhapdf/lhapdf-6.5.3-snd-local1"
export PYTHIA8="${CVMFS_SW}/pythia/sndsw-pythia8309-local1"
export PYTHIA8DATA="${CVMFS_SW}/pythia/sndsw-pythia8309-local1/share/Pythia8/xmldoc"
export PYTHIA_ROOT="${CVMFS_SW}/pythia/sndsw-pythia8309-local1"

# 2. ddfUtils setup
if [ -d "$(dirname "${BASH_SOURCE[0]}")/ddfUtils" ]; then
    export ddfUtils="$(cd "$(dirname "${BASH_SOURCE[0]}")/ddfUtils" && pwd)"
elif [ -n "${DDFUTILS}" ] && [ -d "${DDFUTILS}" ]; then
    export ddfUtils="${DDFUTILS}"
else
    export ddfUtils="/afs/cern.ch/user/i/idioniso/ddfUtils"
fi

# 3. Complete LD_LIBRARY_PATH (Crucial for FairRoot, FairLogger, FairMQ, Geant4, ROOT runtime resolution)
export LD_LIBRARY_PATH="${USER_SNDSW_BUILD}/lib:\
${CVMFS_SW}/sndsw/master-local1/lib:\
${CVMFS_SW}/POWHEG/r4110-forward-snd-local1/lib:\
${CVMFS_SW}/looptools/v2.16-alice1-local1/lib:\
${CVMFS_SW}/fastjet/v3.2.1_1.024-alice1-local1/lib:\
${CVMFS_SW}/cgal/4.6.3-local1/lib:\
${CVMFS_SW}/MPFR/v3.1.3-local1/lib:\
${CVMFS_SW}/GMP/v6.0.0-local1/lib:\
${CVMFS_SW}/FEDRA/rev1530-local1/lib:\
${CVMFS_SW}/GENIE/v3.2.0-local1/lib:\
${CVMFS_SW}/GENIE/v3.2.0-local1/genie/lib:\
${CVMFS_SW}/log4cpp/1b9f8f7c031d6947c7468d54bc1da4b2f414558d-local1/lib:\
${CVMFS_SW}/apfel/3.0.6-local1/lib:\
${CVMFS_SW}/FairRoot/v19.0.0-local1/lib:\
${CVMFS_SW}/FairMQ/v1.9.1-local1/lib:\
${CVMFS_SW}/flatbuffers/v1.11.0-local1/lib:\
${CVMFS_SW}/asiofi/v0.5.1-local1/lib:\
${CVMFS_SW}/asio/v1.19.1-local1/lib:\
${CVMFS_SW}/ofi/v1.14.0-local1/lib:\
${CVMFS_SW}/ZeroMQ/v4.3.5-local1/lib:\
${CVMFS_SW}/FairCMakeModules/v1.0.0-local1/lib:\
${CVMFS_SW}/FairLogger/v1.9.0-local1/lib:\
${CVMFS_SW}/fmt/7.1.0-local1/lib:\
${CVMFS_SW}/simulation/v1.0-local1/lib:\
${CVMFS_SW}/GEANT3/v4-1-local1/lib:\
${CVMFS_SW}/GEANT3/v4-1-local1/lib64:\
${CVMFS_SW}/GEANT4_VMC/v6-1-p2-snd-local1/lib:\
${CVMFS_SW}/vgm/v5-0-snd-local1/lib:\
${CVMFS_SW}/GEANT4/v11.1.0-snd-local1/lib:\
${CVMFS_SW}/XercesC/v3.2.2-local1/lib:\
${CVMFS_SW}/VMC/v2-0-local1/lib:\
${CVMFS_SW}/generators/v1.0-local1/lib:\
${CVMFS_SW}/EvtGen/R01-06-00-ship-local1/lib:\
${CVMFS_SW}/PHOTOSPP/v3.61-local1/lib:\
${CVMFS_SW}/Tauolapp/v1.1.8-local1/lib:\
${CVMFS_SW}/HepMC3/3.3.0-local1/lib:\
${CVMFS_SW}/ROOT/v6-36-04-alice1-local1/lib:\
${CVMFS_SW}/XRootD/v5.7.3-local1/lib:\
${CVMFS_SW}/UUID/v2.27.1-local1/lib:\
${CVMFS_SW}/FFTW3/v3.3.9-local1/lib:\
${CVMFS_SW}/alibuild-recipe-tools/0.2.2-local1/lib:\
${CVMFS_SW}/protobuf/v3.12.3-local1/lib:\
${CVMFS_SW}/GSL/v2.8-local1/lib:\
${CVMFS_SW}/pythia6/v6.4.28-snd-local1/lib:\
${CVMFS_SW}/pythia/sndsw-pythia8309-local1/lib:\
${CVMFS_SW}/HepMC/v2.06.09-local1/lib:\
${CVMFS_SW}/CMake/v3.23.1-local1/lib:\
${CVMFS_SW}/lhapdf/lhapdf-6.5.3-snd-local1/lib:\
${CVMFS_SW}/Python-modules/1.0-local1/lib:\
${CVMFS_SW}/defaults-release/v1-local1/lib\
${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}"

# 4. ROOT Include Path for C++ Cling JIT compilation
export ROOT_INCLUDE_PATH="${USER_SNDSW_BUILD}/include:\
${CVMFS_SW}/sndsw/master-local1/include:\
${CVMFS_SW}/FairRoot/v19.0.0-local1/include:\
${CVMFS_SW}/FairMQ/v1.9.1-local1/include/fairmq:\
${CVMFS_SW}/asio/v1.19.1-local1/include:\
${CVMFS_SW}/FairLogger/v1.9.0-local1/include:\
${CVMFS_SW}/fmt/7.1.0-local1/include:\
${CVMFS_SW}/GEANT3/v4-1-local1/include/TGeant3:\
${CVMFS_SW}/GEANT4_VMC/v6-1-p2-snd-local1/include/g4root:\
${CVMFS_SW}/GEANT4_VMC/v6-1-p2-snd-local1/include/geant4vmc:\
${CVMFS_SW}/VMC/v2-0-local1/include/vmc:\
${CVMFS_SW}/GENIE/v3.2.0-local1/genie/inc:\
${CVMFS_SW}/GENIE/v3.2.0-local1/genie/src:\
${CVMFS_SW}/GEANT4/v11.1.0-snd-local1/include:\
${CVMFS_SW}/GEANT4/v11.1.0-snd-local1/include/Geant4:\
${CVMFS_SW}/pythia/sndsw-pythia8309-local1/include:\
${CVMFS_SW}/pythia/sndsw-pythia8309-local1/include/Pythia8:\
${CVMFS_SW}/GEANT4_VMC/v6-1-p2-snd-local1/include\
${ROOT_INCLUDE_PATH+:$ROOT_INCLUDE_PATH}"

# 5. Full Executable PATH
export PATH="${USER_SNDSW_BUILD}/bin:\
${CVMFS_SW}/sndsw/master-local1/bin:\
${CVMFS_SW}/ROOT/v6-36-04-alice1-local1/bin:\
${CVMFS_SW}/GENIE/v3.2.0-local1/bin:\
${CVMFS_SW}/FairRoot/v19.0.0-local1/bin:\
${CVMFS_SW}/Python-modules/1.0-local1/bin:\
${PATH}"

# 6. PYTHONPATH
# Prioritize user custom python scripts, followed by ddfUtils, CVMFS ROOT, and Python packages
export PYTHONPATH="${USER_SNDSW_SRC}/python:\
${USER_SNDSW_SRC}/shipLHC/scripts:\
${USER_SNDSW_BUILD}/python:\
${ddfUtils}:\
${CVMFS_SW}/ROOT/v6-36-04-alice1-local1/lib:\
${CVMFS_SW}/XRootD/v5.7.3-local1/lib/python/site-packages:\
${HOME}/.local/lib/python3.9/site-packages:\
${CVMFS_SW}/Python-modules/latest/lib/python3.9/site-packages\
${PYTHONPATH+:$PYTHONPATH}"
