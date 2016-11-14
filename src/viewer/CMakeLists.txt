cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

include_directories(../../include/algorithm)
include_directories(../../include/viewer)
include_directories(../../cudalib)
include_directories(../../robustoptimizer)
link_directories(${PROJECT_BINARY_DIR}/lib)

#Find VTK
find_package(VTK 7.0 REQUIRED)

#Find PCL
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

#Find KinectSDK2
find_package(KinectSDK2 REQUIRED)
include_directories(${KinectSDK2_INCLUDE_DIRS})

#Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_ARCH_BIN "20 30 32 35 37 50 52" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              

include(${PROJECT_SOURCE_DIR}/cudalib/CudaComputeTargetFlags.cmake)                  
APPEND_TARGET_ARCH_FLAGS()

#Find Qt Module
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Widgets REQUIRED)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_definitions(${Qt5Widgets_DEFINITIONS})

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
# As moc files are generated in the binary dir, tell CMake
# to always look for includes there:
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(VIEWER_HEADERS
	../../include/viewer/MainWindow.h
	../../include/viewer/PclViewer.h
	../../include/viewer/RegistrationViewer.h
	../../include/viewer/Parser.h
	../../include/viewer/SlamThread.h)
	
set(VIEWER_SOURCES
	main.cpp
	MainWindow.cpp
	PclViewer.cpp
	RegistrationViewer.cpp
	Parser.cpp
	SlamThread.cpp)
	
set(VIEWER_FORMS
	ui/MainWindow.ui
	ui/PclViewer.ui
	ui/RegistrationViewer.ui
	ui/DockRegistration.ui)

#uic
qt5_wrap_ui(VIEWER_UIS ${VIEWER_FORMS})
#rcc
qt5_add_resources(VIEWER_RCCS ui/main.qrc)

add_executable(WReg ${VIEWER_HEADERS} ${VIEWER_SOURCES} ${VIEWER_UIS} ${VIEWER_RCCS})

target_link_libraries(WReg
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	${VTK_LIBRARIES}
	${CUDA_LIBRARIES}
	${G2O_LIBRARIES}
	${KinectSDK2_LIBRARIES}
	vtkGUISupportQt
	cudalib
	algorithm)
qt5_use_modules(WReg Core Gui Widgets)