cmake_minimum_required(VERSION 3.22.1)
project(pv021_neural_network)

set(CMAKE_CXX_STANDARD 17)

include_directories(src/headers)

add_executable(pv021_neural_network
        src/main.cpp
        src/matrix.cpp
        src/activation.cpp
        src/headers/layer.h
        src/layer.cpp
        src/headers/initialisation.h
        src/initialisation.cpp
        src/image_holder.cpp
        src/headers/image_holder.h
        src/headers/neural_network.h
        src/headers/utils.h src/initialisation.cpp src/headers/initialisation.h)