cmake_minimum_required(VERSION 3.29)
project(gv)

set(CMAKE_CXX_STANDARD 20)

add_library(gv STATIC
        src/Vector3.cpp
        src/Vector2.cpp
        include/Vector3.hpp
        include/Vector2.hpp)

set_target_properties(gv PROPERTIES LINKER_LANGUAGE CXX)