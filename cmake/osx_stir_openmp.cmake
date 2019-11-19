
#========================================================================
# Author: Richard Brown
# Copyright 2019 University College London
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

# Find STIR quietly. 
# If we find STIR, check if it was built with OpenMP
# If it was, give the same help that is given for by the STIR CMakeLists.txt

FIND_PACKAGE(STIR QUIET)
if (NOT STIR_FOUND OR NOT STIR_BUILT_WITH_OpenMP)
	return()
endif()

if(NOT DEFINED OPENMP_INCLUDES OR NOT DEFINED OPENMP_LIBRARIES)
	set(OPENMP_INCLUDES "OPENMP_INCLUDES-NOTFOUND" CACHE PATH "Need to set OpenMP include dir on OSX")
	set(OPENMP_LIBRARIES "OPENMP_LIBRARIES-NOTFOUND" CACHE PATH "Need to set OpenMP lib dir on OSX")
endif()
if(NOT EXISTS ${OPENMP_INCLUDES} OR NOT EXISTS ${OPENMP_LIBRARIES})
	message(FATAL_ERROR "Need to set OPENMP_INCLUDES and OPENMP_LIBRARIES on OSX.")
endif()
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
	set(OpenMP_C "${CMAKE_C_COMPILER}")
	set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
	set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
	set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
	set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
	set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
	set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
	set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
	set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
endif()

find_package(OpenMP QUIET)  
add_definitions(${OpenMP_CXX_FLAGS})

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
set (CMAKE_STATIC_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS}")   
include_directories("${OPENMP_INCLUDES}")
link_directories("${OPENMP_LIBRARIES}")