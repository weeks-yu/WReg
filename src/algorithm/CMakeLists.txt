cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

include_directories(../../include/algorithm)
include_directories(../../cudalib)
include_directories(../../robustoptimizer)
link_directories(${PROJECT_BINARY_DIR}/lib)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#Find OpenCv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#Find G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})

#Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_ARCH_BIN "20 30 32 35 37 50 52" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              

include(${PROJECT_SOURCE_DIR}/cudalib/CudaComputeTargetFlags.cmake)                  
APPEND_TARGET_ARCH_FLAGS()

<<<<<<< HEAD
file(GLOB ALGORITHM_HEADER ../../include/algorithm/*.h ../../include/algorithm/*.hpp)
file(GLOB ALGORITHM_SRC *.cpp)
=======
set(ALGORITHM_HEADER
	../../include/algorithm/Config.h
	../../include/algorithm/Feature.h
	../../include/algorithm/Frame.h
	../../include/algorithm/ActiveWindow.h
	../../include/algorithm/RobustManager.h
	../../include/algorithm/PointCloud.h
#	../../include/algorithm/QuadTree.h
	../../include/algorithm/Transformation.h
	../../include/algorithm/SlamEngine.h)
	
set(ALGORITHM_SRC
	Config.cpp
	Feature.cpp
	ActiveWindow.cpp
	RobustManager.cpp
	PointCloud.cpp
	Transformation.cpp
	SlamEngine.cpp)
>>>>>>> origin/simplified
	
add_library(algorithm STATIC ${ALGORITHM_HEADER} ${ALGORITHM_SRC})
target_link_libraries(algorithm
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	${CUDA_LIBRARIES}
	${G2O_LIBRARIES}
	icpcuda
	robustoptimizer
	)