#========================================================================
# Author: Richard Brown
# Copyright 2016, 2017 University College London
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#=========================================================================

PROJECT(SIRFReg)

# NiftyReg
SET(NiftyReg_Source_DIR  CACHE PATH "NiftyReg source directory" )
SET(NiftyReg_Binary_DIR  CACHE PATH "NiftyReg binary directory" )
SET(NiftyReg_Install_DIR CACHE PATH "NiftyReg install directory")

# To be able to build with niftyreg, you need to append the cmake_cxx_flags to suppress errors from converting float to double
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")

IF ((NOT IS_DIRECTORY ${NiftyReg_Source_DIR}) AND (NOT IS_DIRECTORY ${NiftyReg_Binary_DIR}) AND (NOT IS_DIRECTORY ${NiftyReg_Install_DIR}))
	MESSAGE(FATAL_ERROR "Enter the source, binary and install directories of NiftyReg.")
ENDIF()

INCLUDE_DIRECTORIES(${NiftyReg_Source_DIR}/reg-io/nrrd/NrrdIO) # ugly, but required for nrrdIO.h
INCLUDE_DIRECTORIES(${NiftyReg_Binary_DIR})                    # ugly, but required for NrrdConfigure.h
INCLUDE_DIRECTORIES(${NiftyReg_Install_DIR}/include)

OPTION(BUILD_TESTING "Build testing" ON)
IF(BUILD_TESTING)
	ENABLE_TESTING()
	SET(TEST_IN_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/test/inFiles)
	SET(TEST_OUT_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/test/outFiles)
	SET(TEST_DATA_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/test/data)
	SET(TEST_PARAM_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/test/paramFiles)
	SET(TEST_RESULT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/results)
ENDIF()

# Compile the C++
ADD_SUBDIRECTORY(cReg)
INCLUDE_DIRECTORIES(cReg)

# Python or MATLAB
OPTION(BUILD_PYTHON "Build Python SWIG module" OFF)
OPTION(BUILD_MATLAB "Build MATLAB SWIG module" OFF)

# If either, find SWIG
IF(BUILD_PYTHON OR BUILD_MATLAB)
	FIND_PACKAGE(SWIG REQUIRED)
	INCLUDE(${SWIG_USE_FILE})
	SET(CMAKE_SWIG_FLAGS "")
ENDIF()

# Python
IF(BUILD_PYTHON)
    ADD_SUBDIRECTORY(pReg)
ENDIF()

# MATLAB
IF(BUILD_MATLAB)
    ADD_SUBDIRECTORY(mReg)
ENDIF()