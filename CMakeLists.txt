cmake_minimum_required(VERSION 3.0)
project("RVO Test")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_BUILD_TYPE RELEASE)
#set(CMAKE_BUILD_TYPE DEBUG)
set(TARGET_TEST rvo_test)
add_executable(${TARGET_TEST}
  test.cpp
)

target_link_libraries(${TARGET_TEST} PRIVATE
  glog
  stdc++fs
)
