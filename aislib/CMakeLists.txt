cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

include_directories(./)

add_subdirectory(csparse)
add_subdirectory(stuff)
add_subdirectory(math)
add_subdirectory(graph)
add_subdirectory(graph_optimizer)
add_subdirectory(graph_optimizer_hogman)

add_library(aislib STATIC
	${CSPARSE_SRC}
	${STUFF_SRC}
	${MATH_SRC}
	${GRAPH_SRC}
	${GO_SRC}
	${GOH_SRC})