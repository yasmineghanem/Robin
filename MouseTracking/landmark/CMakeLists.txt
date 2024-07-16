cmake_minimum_required(VERSION 3.0)
project(LandMark)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set paths to dependencies
set(OpenCV_DIR "C:/TempDesktop/opencv")
set(ONNXRUNTIME_DIR "C:/TempDesktop/onnxruntime-win-x64-1.18.1")
set(CMAKE_PREFIX_PATH "C:/TempDesktop/onnxruntime-win-x64-1.18.1")

# Find OpenCV
find_package(OpenCV 4.9.0 REQUIRED PATHS "C:/TempDesktop/opencv")
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Dlib
set(Dlib_DIR "C:/TempDesktop/dlib-19.24")
include_directories(${Dlib_DIR})
add_subdirectory(${Dlib_DIR} dlib_build)

# Include directories for ONNX Runtime
# include_directories(${ONNXRUNTIME_DIR}/include)
set(ONNX_DIR "C:/TempDesktop/onnxruntime-win-x64-1.18.1")
include_directories(${ONNX_DIR})
add_subdirectory(${ONNX_DIR} ONNX_BUILD)

# Add the executable with additional source files
add_executable(LandMark main_onex.cpp)

# Link libraries
target_link_libraries(LandMark
    ${OpenCV_LIBS}
    dlib::dlib
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"  # Link directly to ONNX Runtime library
)

# Post-build step to copy the executable
add_custom_command(TARGET LandMark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:LandMark>
    ${CMAKE_SOURCE_DIR}/$<TARGET_FILE_NAME:LandMark>
)

# Additional logging for troubleshooting
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "OpenCV Include Dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Dlib Include Dirs: ${Dlib_DIR}")
message(STATUS "ONNX Runtime Include Dirs: ${ONNXRUNTIME_DIR}/include")
