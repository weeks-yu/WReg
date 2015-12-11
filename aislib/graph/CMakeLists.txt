cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(GRAPH_SRC
	graph/dijkstra.h
	graph/graph.h
	graph/loadEdges3d.h
	graph/posegraph.h
	graph/posegraph2d.h
	graph/posegraph3d.h
	graph/posegraph3d_gradient.h
	graph/posegraph.hpp
	graph/dijkstra.cpp
	graph/graph.cpp
	graph/loadEdges3d.cpp
	graph/posegraph2d.cpp
	graph/posegraph3d.cpp
	PARENT_SCOPE)