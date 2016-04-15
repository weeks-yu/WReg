cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(WReg)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_CONFIGURATION_TYPES Debug Release)

add_subdirectory(src)
add_subdirectory(aislib)
add_subdirectory(cudalib)