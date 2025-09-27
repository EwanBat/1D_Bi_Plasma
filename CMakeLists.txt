cmake_minimum_required(VERSION 3.10)
project(Plasma1D LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define source and include directories
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# Include directories
include_directories(${INCLUDE_DIR})

# Gather all source files
file(GLOB SOURCES 
    ${SRC_DIR}/*.cpp
)

# Gather all header files
file(GLOB HEADERS
    ${INCLUDE_DIR}/*.hpp
)

# Create executable
add_executable(bi_plasma ${SOURCES} ${HEADERS})

# Optionally, set the output directory
set_target_properties(bi_plasma PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)