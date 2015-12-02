cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(WReg)

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
set(CMAKE_CONFIGURATION_TYPES Debug Release)

set(WReg_SOURCES main.cpp MainWindow.cpp PclViewer.cpp BenchmarkViewer.cpp)
set(WReg_HEADERS MainWindow.h PclViewer.h BenchmarkViewer.h)
set(WReg_FORMS MainWindow.ui)

#uic
qt5_wrap_ui(WReg_UIS ${WReg_FORMS})
#rcc
qt5_add_resources(WReg_RCCS main.qrc)

add_executable(WReg ${WReg_HEADERS} ${WReg_SOURCES} ${WReg_UIS} ${WReg_RCCS})

target_link_libraries(WReg ${OpenCV_LIBS} ${PCL_LIBRARIES} ${VTK_LIBRARIES} vtkGUISupportQt)
qt5_use_modules(WReg Core Gui Widgets)
