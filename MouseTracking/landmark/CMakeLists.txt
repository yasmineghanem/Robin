cmake_minimum_required(VERSION 3.0)
project(LandMark)


# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Set the OpenCV_DIR variable
set(OpenCV_DIR "C:/TempDesktop/opencv")
set(CMAKE_PREFIX_PATH "C:/TempDesktop/libtorch")

find_package(Torch REQUIRED PATHS "C:/TempDesktop/libtorch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# Find the OpenCV package
find_package(OpenCV 4.9.0 REQUIRED PATHS "C:/TempDesktop/opencv")

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Set the Dlib_DIR variable
set(Dlib_DIR "C:/TempDesktop/dlib-19.24")

# Add Dlib
include_directories(${Dlib_DIR})
add_subdirectory(${Dlib_DIR} dlib_build)

# Include the directory containing FaceDetector.h
# include_directories(${CMAKE_SOURCE_DIR}/viola-jons)

# Add the executable with additional source files
add_executable(LandMark
    main_torch.cpp 
)


# Link OpenCV and Dlib libraries
target_link_libraries(LandMark ${OpenCV_LIBS} dlib::dlib ${TORCH_LIBRARIES})

# Post-build step to copy the executable
add_custom_command(TARGET LandMark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:LandMark>
    ${CMAKE_SOURCE_DIR}/$<TARGET_FILE_NAME:LandMark>
)
