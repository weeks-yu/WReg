cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

include_directories(../../include/algorithm)
include_directories(../../include/viewer)
include_directories(../../aislib)
link_directories(${PROJECT_BINARY_DIR}/lib)

#Find VTK
find_package(VTK 6.2 REQUIRED)

#Find PCL
find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#Find OpenCv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

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
	../../include/viewer/BenchmarkViewer.h
	../../include/viewer/Parser.h
	../../include/viewer/SlamThread.h)
	
set(VIEWER_SOURCES
	main.cpp
	MainWindow.cpp
	PclViewer.cpp
	BenchmarkViewer.cpp
	Parser.cpp
	SlamThread.cpp)
	
set(VIEWER_FORMS
	ui/MainWindow.ui
	ui/PclViewer.ui
	ui/BenchmarkViewer.ui
	ui/DockBenchmark.ui)

#uic
qt5_wrap_ui(VIEWER_UIS ${VIEWER_FORMS})
#rcc
qt5_add_resources(VIEWER_RCCS ui/main.qrc)

add_executable(WReg ${VIEWER_HEADERS} ${VIEWER_SOURCES} ${VIEWER_UIS} ${VIEWER_RCCS})

target_link_libraries(WReg ${OpenCV_LIBS} ${PCL_LIBRARIES} ${VTK_LIBRARIES} vtkGUISupportQt aislib algorithm)
qt5_use_modules(WReg Core Gui Widgets)
