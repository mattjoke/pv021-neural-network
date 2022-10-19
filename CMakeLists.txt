cmake_minimum_required(VERSION 3.22.1)
project(pv021_neural_network)

set(CMAKE_CXX_STANDARD 17)

include_directories(src/headers)

add_executable(pv021_neural_network
        src/main.cpp
        src/utils.cpp
        src/activation.cpp src/headers/activation.h
        src/headers/layer.h src/layer.cpp)