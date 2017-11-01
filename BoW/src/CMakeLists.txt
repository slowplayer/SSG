cmake_minimum_required(VERSION 2.8)
project(BoW)

set(CMAKE_BUILD_TYPE "g++")
set(CMAKE_BUILD_TYPE "Realease")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

#opencv
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#DBoW3
set(DBoW3_INCLUDE_DIRS "/usr/local/include")
set(DBoW3_LIBS "/usr/local/lib/libDBoW3.so")

add_executable(bow_training bow_training.cpp)
target_link_libraries(bow_training ${OpenCV_LIBS} ${DBoW3_LIBS})

add_executable(bow_testing bow_testing.cpp)
target_link_libraries(bow_testing ${OpenCV_LIBS} ${DBoW3_LIBS})

add_executable(bow_demo bow_demo.cpp)
target_link_libraries(bow_demo ${OpenCV_LIBS} ${DBoW3_LIBS})