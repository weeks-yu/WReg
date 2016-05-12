cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})

include_directories(${EIGEN_INCLUDE_DIRS})

set(ROBUSTOPTIMIZER_HEADERS
	vertigo/edge_se3Switchable.h
	vertigo/edge_switchPrior.h
	vertigo/vertex_switchLinear.h)
	
set(ROBUSTOPTIMIZER_SOURCES
	vertigo/edge_se3Switchable.cpp
	vertigo/edge_switchPrior.cpp
	vertigo/vertex_switchLinear.cpp
	vertigo/types_g2o_robust.cpp
)

add_library(robustoptimizer ${ROBUSTOPTIMIZER_HEADERS} ${ROBUSTOPTIMIZER_SOURCES})

target_link_libraries(robustoptimizer
	${G2O_LIBRARIES}
)