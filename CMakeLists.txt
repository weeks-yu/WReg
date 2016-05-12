cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(WReg)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_CONFIGURATION_TYPES Debug Release)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

add_subdirectory(src)
add_subdirectory(aislib)
add_subdirectory(cudalib)
add_subdirectory(robustoptimizer)