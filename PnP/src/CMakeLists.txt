cmake_minimum_required(VERSION 2.8)
project(PnP)

set(CMAKE_BUILD_TYPE "g++")
set(CMAKE_BUILD_TYPE "Realease")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#eigen
include_directories("/usr/include/eigen3")

add_executable(pnp main.cpp feature_solver.cpp epnp.cpp main.cpp)
target_link_libraries(pnp ${OpenCV_LIBS})