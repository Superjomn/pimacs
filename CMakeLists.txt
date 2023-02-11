cmake_minimum_required(VERSION 3.6)
include(ExternalProject)
#cmake_policy("CMP0116)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

project(pyimacs)
include(CTest)
if(NOT WIN32)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pyimacs)

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17 -fvisibility=hidden -fvisibility-inlines-hidden")
if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 11.6)
endif()

option(PYIMACS_BUILD_PYTHON_MODULE "Build Python binding" ON)

##########
# LLVM
##########
if (NOT MLIR_DIR)
  if(NOT LLVM_LIBRARY_DIR)
    if(WIN32)
      find_package(LLVM 13 REQUIRED COMPONENTS nvptx amdgpu)

      include_directories(${LLVM_INCLUDE_DIRS})
      separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
      add_definitions(${LLVM_DEFINITIONS_LIST})

      llvm_map_components_to_libnames(LLVM_LIBRARIES support core
      )
    else()
      find_package(LLVM 14 REQUIRED)
    endif()
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    if(APPLE)
      set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14")
    endif()
  # sometimes we don't want to use llvm-config, since it may have been downloaded for some specific linux distros
  else()
    set(LLVM_LDFLAGS "-L${LLVM_LIBRARY_DIR}")
    set(LLVM_LIBRARIES
      libLLVMMIRParser.a
      libLLVMGlobalISel.a
      libLLVMSelectionDAG.a
      libLLVMipo.a
      libLLVMInstrumentation.a
      libLLVMVectorize.a
      libLLVMLinker.a
      libLLVMIRReader.a
      libLLVMAsmParser.a
      libLLVMFrontendOpenMP.a
      libLLVMAsmPrinter.a
      libLLVMDebugInfoDWARF.a
      libLLVMCodeGen.a
      libLLVMTarget.a
      libLLVMScalarOpts.a
      libLLVMInstCombine.a
      libLLVMAggressiveInstCombine.a
      libLLVMTransformUtils.a
      libLLVMBitWriter.a
      libLLVMAnalysis.a
      libLLVMProfileData.a
      libLLVMObject.a
      libLLVMTextAPI.a
      libLLVMBitReader.a
      libLLVMAMDGPUAsmParser.a
      libLLVMMCParser.a
      libLLVMAMDGPUDesc.a
      libLLVMAMDGPUUtils.a
      libLLVMMC.a
      libLLVMDebugInfoCodeView.a
      libLLVMDebugInfoMSF.a
      libLLVMCore.a
      libLLVMRemarks.a
      libLLVMBitstreamReader.a
      libLLVMBinaryFormat.a
      libLLVMAMDGPUInfo.a
      libLLVMSupport.a
      libLLVMDemangle.a
      libLLVMPasses.a
      libLLVMAnalysis.a
      libLLVMTransformUtils.a
      libLLVMScalarOpts.a
      libLLVMTransformUtils.a
      libLLVMipo.a
      libLLVMObjCARCOpts.a
      libLLVMCoroutines.a
      libLLVMAnalysis.a
    )
  endif()
  set (MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()

#if("${PYIMACS_BUILD_PYTHON_MODULE}")
    message(STATUS "Adding Python module")
    set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
    include_directories("." ${PYTHON_SRC_PATH})
    find_package(Python3 REQUIRED COMPONENTS Development)
    include_directories(${Python3_INCLUDE_DIRS})
    link_directories(${Python3_LIBRARY_DIRS})
    link_libraries(${Python3_LIBRARIES})
    add_link_options(${Python3_LINK_OPTIONS})
    set(PYTHON_SRC ${PYTHON_SRC_PATH}/pyimacs.cc)
#endif()

# MLIR
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen) # required by AddMLIR
include(AddLLVM)
include(AddMLIR)

# include external libs
set(THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/thirds)
include(cmake/core.cmake)
#include(cmake/external/gflags.cmake)
#include(cmake/external/glog.cmake)
include(cmake/external/pybind11.cmake)

# Disable warnings that show up in external code (gtest;pybind11)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR})

add_subdirectory(pyimacs)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_library(pyimacs SHARED ${PYTHON_SRC})
target_link_libraries(pyimacs
        LispIR
        #ElispTranslation
        pybind
        ${dialect_libs}
        ${conversion_libs}
        MLIRPass
        MLIRSupport
        MLIRIR
        )
add_dependencies(pyimacs LispTableGen)
target_link_options(pyimacs PRIVATE ${LLVM_LDFLAGS})

if(NOT WIN32)
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
    # Check if the platform is MacOS
    if(APPLE)
        set(PYTHON_LDFLAGS "-undefined dynamic_lookup -flto")
    endif()
    target_link_libraries(pyimacs ${PYTHON_LDFLAGS})
endif()
