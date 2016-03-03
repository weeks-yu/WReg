# =========================================================================
#  libSRBA CMake configuration file
#
#             ** File generated automatically, do not modify **
#
#  Usage from an external project: 
#   In your CMakeLists.txt, add these lines:
#
#    FIND_PACKAGE( SRBA REQUIRED )
#    INCLUDE_DIRECTORIES(${SRBA_INCLUDE_DIRS})
#    FIND_PACKAGE( MRPT REQUIRED ${SRBA_REQUIRED_MRPT_MODULES} )  # You may add additional MRPT modules at the end of the list for your app
#
#   This file will define the following variables:
#    - SRBA_INCLUDE_DIRS: The list of include directories
#    - SRBA_VERSION: The SRBA version (e.g. "1.0.0").
#    - SRBA_VERSION_{MAJOR,MINOR,PATCH}: 3 variables for the version parts
#    - SRBA_REQUIRED_MRPT_MODULES: The minimum list of required MRPT modules
#
#   Remember to link against MRPT libraries in your program with:
#
#     TARGET_LINK_LIBRARIES(YOUR_TARGET ${MRPT_LIBS})
#
# =========================================================================

# SRBA version numbers:
SET(SRBA_VERSION 1.3.9)
SET(SRBA_VERSION_MAJOR 1)
SET(SRBA_VERSION_MINOR 3)
SET(SRBA_VERSION_PATCH 9)

# MRPT dependencies:
SET(SRBA_REQUIRED_MRPT_MODULES base opengl graphs tfest)

# Extract the directory where *this* file has been installed (determined at cmake run-time)
get_filename_component(THIS_SRBA_CONFIG_PATH "${CMAKE_CURRENT_LIST_FILE}" PATH)

SET(SRBA_INCLUDE_DIRS "G:/srba/include")  # SRBA include directories

# Compiler flags:
if(MSVC)
	# For MSVC to avoid the C1128 error about too large object files:
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif(MSVC)
