cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

include_directories(../../include/algorithm)
link_directories(${PROJECT_BINARY_DIR}/lib)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#Find OpenCv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(ALGORITHM_HEADER
	../../include/algorithm/Config.h
	../../include/algorithm/Feature.h
	../../include/algorithm/Frame.h
	../../include/algorithm/GraphManager.h
	../../include/algorithm/PointCloud.h
	../../include/algorithm/QuadTree.h
	../../include/algorithm/Transformation.h)
	
set(ALGORITHM_SRC
	Feature.cpp
	GraphManager.cpp
	Transformation.cpp)
	
add_library(algorithm STATIC ${ALGORITHM_HEADER} ${ALGORITHM_SRC})
target_link_libraries(algorithm ${OpenCV_LIBS} ${PCL_LIBRARIES})