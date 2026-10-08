include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

# Header files for dictionary generation
set(ANALYSIS_HEADERS
  NeutrinoTruthProcessor.h
  MuonNeutrinoTruthProcessor.h
  ElectronNeutrinoTruthProcessor.h
  TauNeutrinoTruthProcessor.h
  PreselectionProcessor.h
  DataManager.h
  ProgressPrinter.h
  AvgScifiFiducialCut.h
  AvgDSFiducialCut.h
  VetoCut.h
  SciFiStationCut.h
  DSDimuonActivityCut.h
  SciFiDSTimeCut.h
  DSActivityCut.h
  EventHeaderCuts.h
)

set(ANALYSIS_SRCS
  NeutrinoTruthProcessor.cxx
  MuonNeutrinoTruthProcessor.cxx
  ElectronNeutrinoTruthProcessor.cxx
  TauNeutrinoTruthProcessor.cxx
  PreselectionProcessor.cxx
  DataManager.cxx
  ProgressPrinter.cxx
)

# ROOT generate dictionary
root_generate_dictionary(G__neutrino_analysis
  ${ANALYSIS_HEADERS}
  LINKDEF AnalysisLinkDef.h
)

# Add library for analysis
add_library(neutrino_analysis SHARED
  ${ANALYSIS_SRCS}
  G__neutrino_analysis.cxx
)

# Find sndsw dependencies
find_library(SHIP_DATA_LIB ShipData PATHS "${SNDSW_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(SHIP_LHC_LIB shipLHC PATHS "${SNDSW_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(ANALYSIS_TOOLS_LIB snd_analysis_tools PATHS "${SNDSW_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

target_link_libraries(neutrino_analysis
  PUBLIC
    ROOT::Core
    ROOT::Tree
    ROOT::ROOTDataFrame
    ROOT::RIO
    ROOT::Hist
    ROOT::Geom
    ${SHIP_DATA_LIB}
    ${SHIP_LHC_LIB}
    ${ANALYSIS_TOOLS_LIB}
)
