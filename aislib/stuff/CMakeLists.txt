cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(STUFF_SRC
	stuff/array_allocator.h
	stuff/color_macros.h
	stuff/dirent.h
	stuff/filesys_tools.h
	stuff/macros.h
	stuff/os_specific.h
	stuff/runtime_error.h
	stuff/string_tools.h
	stuff/time.h
	stuff/os_specific.c
	stuff/filesys_tools.cpp
	stuff/runtime_error.cpp
	stuff/string_tools.cpp
	PARENT_SCOPE)