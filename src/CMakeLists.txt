cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(WReg)

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Widgets REQUIRED)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
# As moc files are generated in the binary dir, tell CMake
# to always look for includes there:
set(CMAKE_INCLUDE_CURRENT_DIR ON)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_definitions(${Qt5Widgets_DEFINITIONS})

qt5_wrap_ui(WReg_UIS mainwindow.ui)

add_executable(WReg main.cpp mainwindow.cpp ${WReg_UIS})

target_link_libraries(WReg ${PCL_LIBRARIES})
qt5_use_modules(WReg Core Gui Widgets)
