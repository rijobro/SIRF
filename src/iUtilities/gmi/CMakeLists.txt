#========================================================================
# Author: Evgueni Ovtchinnikov
# Copyright 2017 STFC
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

set(CMAKE_POSITION_INDEPENDENT_CODE True)

if(BUILD_MATLAB)

  include_directories(${Matlab_INCLUDE_DIRS})

  include_directories(${PROJECT_SOURCE_DIR}/src/common/include)
  add_executable(gmi_iutilities gmi.cpp)
  target_link_libraries(gmi_iutilities mig)
  INSTALL(TARGETS gmi_iutilities DESTINATION bin)

endif(BUILD_MATLAB)
