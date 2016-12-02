cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(tsdflib)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB hdrs *.cuh *.h *.hpp)
file(GLOB srcs *.cu *.cpp)


CUDA_ADD_LIBRARY(tsdflib STATIC
	${hdrs}
	${srcs}
)

target_link_libraries(tsdflib
	${CUDA_LIBRARIES}
)

