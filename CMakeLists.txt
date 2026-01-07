cmake_minimum_required(VERSION 3.15)
project(dracox LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Get nanobind via CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
  RESULT_VARIABLE NB_RESULT)

if(NOT NB_RESULT EQUAL 0)
  message(FATAL_ERROR "nanobind not found. Install it with: pip install nanobind")
endif()

list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Add Draco as a subdirectory
# Configure Draco for both encoding and decoding
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(DRACO_POINT_CLOUD_COMPRESSION ON CACHE BOOL "" FORCE)
set(DRACO_MESH_COMPRESSION ON CACHE BOOL "" FORCE)
set(DRACO_STANDARD_EDGEBREAKER ON CACHE BOOL "" FORCE)
set(DRACO_PREDICTIVE_EDGEBREAKER ON CACHE BOOL "" FORCE)
set(DRACO_BACKWARDS_COMPATIBILITY ON CACHE BOOL "" FORCE)

# Enable both encoder and decoder
set(DRACO_ENCODER ON CACHE BOOL "" FORCE)
set(DRACO_DECODER ON CACHE BOOL "" FORCE)

# Disable unnecessary Draco features to minimize build
set(DRACO_TESTS OFF CACHE BOOL "" FORCE)
set(DRACO_WASM OFF CACHE BOOL "" FORCE)
set(DRACO_UNITY_PLUGIN OFF CACHE BOOL "" FORCE)
set(DRACO_MAYA_PLUGIN OFF CACHE BOOL "" FORCE)
set(DRACO_ANIMATION_ENCODING OFF CACHE BOOL "" FORCE)
set(DRACO_TRANSCODER_SUPPORTED OFF CACHE BOOL "" FORCE)
set(DRACO_JS_GLUE OFF CACHE BOOL "" FORCE)

add_subdirectory(draco)

# Create the nanobind extension module
nanobind_add_module(
  dracox_ext
  STABLE_ABI
  NB_STATIC
  src/dracox_ext.cpp
)

# Link against the Draco decoder library
# On MSVC, the library is called 'draco', on other systems 'draco_static'
if(MSVC)
  target_link_libraries(dracox_ext PRIVATE draco)
  add_dependencies(dracox_ext draco)
else()
  target_link_libraries(dracox_ext PRIVATE draco_static)
  add_dependencies(dracox_ext draco_static)
endif()

# Include Draco headers (both source and generated)
target_include_directories(dracox_ext PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/draco/src
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Install the extension module
install(TARGETS dracox_ext LIBRARY DESTINATION dracox)
