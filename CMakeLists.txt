cmake_minimum_required(VERSION 3.10)
project(sndNeutrinoAnalysis CXX)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set global output directories for libraries and executables
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/cuts ${CMAKE_CURRENT_SOURCE_DIR})

# Ensure ROOT prefix directory is in CMAKE_PREFIX_PATH for internal dependency searches (like Vdt)
if(DEFINED ENV{ROOT_ROOT})
  list(APPEND CMAKE_PREFIX_PATH $ENV{ROOT_ROOT})
endif()
if(DEFINED ENV{ROOTSYS})
  list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
endif()

# Find ROOT
find_package(ROOT REQUIRED COMPONENTS Core Tree Hist RIO)
include(${ROOT_USE_FILE})

# Check that SNDSW_ROOT is set
if(NOT DEFINED ENV{SNDSW_ROOT})
  message(FATAL_ERROR "SNDSW_ROOT environment variable is not set!")
endif()
set(SNDSW_ROOT $ENV{SNDSW_ROOT})

# Setup base directory of the software stack
get_filename_component(SNDSW_PARENT ${SNDSW_ROOT} DIRECTORY)
get_filename_component(SND_BASE ${SNDSW_PARENT} DIRECTORY)

# Setup include directories
include_directories(
  ${CMAKE_SOURCE_DIR}/cuts
  ${CMAKE_SOURCE_DIR}/analysis
  ${SNDSW_ROOT}/include
  ${ROOT_INCLUDE_DIRS}
)

# Auto-detect FairRoot include directory
if(DEFINED ENV{FAIRROOT_ROOT})
  set(FAIRROOT_PATH $ENV{FAIRROOT_ROOT})
elseif(DEFINED ENV{FAIRROOTPATH})
  set(FAIRROOT_PATH $ENV{FAIRROOTPATH})
else()
  file(GLOB FAIRROOT_DIRS "${SND_BASE}/FairRoot/*")
  if(FAIRROOT_DIRS)
    list(GET FAIRROOT_DIRS 0 FAIRROOT_PATH)
  endif()
endif()
if(FAIRROOT_PATH)
  include_directories(${FAIRROOT_PATH}/include)
  message(STATUS "Found FairRoot include: ${FAIRROOT_PATH}/include")
else()
  message(WARNING "FairRoot include directory not found!")
endif()

# Auto-detect VMC include directory
if(DEFINED ENV{VMC_ROOT})
  set(VMC_PATH $ENV{VMC_ROOT})
else()
  file(GLOB VMC_DIRS "${SND_BASE}/VMC/*")
  if(VMC_DIRS)
    list(GET VMC_DIRS 0 VMC_PATH)
  endif()
endif()
if(VMC_PATH)
  include_directories(${VMC_PATH}/include/vmc)
  message(STATUS "Found VMC include: ${VMC_PATH}/include/vmc")
else()
  message(WARNING "VMC include directory not found!")
endif()

# Auto-detect fmt include directory
if(DEFINED ENV{FMT_ROOT})
  set(FMT_PATH $ENV{FMT_ROOT})
else()
  file(GLOB FMT_DIRS "${SND_BASE}/fmt/*")
  if(FMT_DIRS)
    list(GET FMT_DIRS 0 FMT_PATH)
  endif()
endif()
if(FMT_PATH)
  include_directories(${FMT_PATH}/include)
  message(STATUS "Found fmt include: ${FMT_PATH}/include")
else()
  message(WARNING "fmt include directory not found!")
endif()

# Auto-detect FairLogger include directory
if(DEFINED ENV{FAIRLOGGER_ROOT})
  set(FAIRLOGGER_PATH $ENV{FAIRLOGGER_ROOT})
else()
  file(GLOB FAIRLOGGER_DIRS "${SND_BASE}/FairLogger/*")
  if(FAIRLOGGER_DIRS)
    list(GET FAIRLOGGER_DIRS 0 FAIRLOGGER_PATH)
  endif()
endif()
if(FAIRLOGGER_PATH)
  include_directories(${FAIRLOGGER_PATH}/include)
  message(STATUS "Found FairLogger include: ${FAIRLOGGER_PATH}/include")
else()
  message(WARNING "FairLogger include directory not found!")
endif()

# Link directories
link_directories(
  ${SNDSW_ROOT}/lib
  ${ROOT_LIBRARY_DIRS}
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cuts/CMakeLists.txt)
  add_subdirectory(cuts)
endif()

add_subdirectory(analysis)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tridentSample.cxx)
  # Find sndsw dependencies
  find_library(ANALYSIS_TOOLS_LIB snd_analysis_tools PATHS "${SNDSW_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  find_library(SHIP_DATA_LIB ShipData PATHS "${SNDSW_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  find_library(SHIP_LHC_LIB shipLHC PATHS "${SNDSW_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  find_library(VETO_LIB veto PATHS "${SNDSW_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

  # Add executable
  add_executable(tridentSample tridentSample.cxx)
  target_link_libraries(tridentSample
    PRIVATE
    trident_cuts
    ${SHIP_LHC_LIB}
    ${VETO_LIB}
    ${ANALYSIS_TOOLS_LIB}
    ${SHIP_DATA_LIB}
    ${ROOT_LIBRARIES}
  )

  # Set executable properties
  set_target_properties(tridentSample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
endif()
