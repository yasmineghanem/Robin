cmake_minimum_required(VERSION 3.0)
project(MouseControlling)

# Set the OpenCV_DIR variable
set(OpenCV_DIR "C:/TempDesktop/opencv")

# Find the OpenCV package
find_package(OpenCV 4.9.0 REQUIRED PATHS "C:/TempDesktop/opencv")

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Set the Dlib_DIR variable
set(Dlib_DIR "C:/TempDesktop/dlib-19.24")

# Add Dlib
include_directories(${Dlib_DIR})
add_subdirectory(${Dlib_DIR} dlib_build)

# Add the executable
add_executable(MouseControlling main.cpp landmark.cpp mousecontroller.cpp)

# Link OpenCV and Dlib libraries
target_link_libraries(MouseControlling ${OpenCV_LIBS} dlib::dlib)

# Post-build step to copy the executable
add_custom_command(TARGET MouseControlling POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:MouseControlling>
    ${CMAKE_SOURCE_DIR}/$<TARGET_FILE_NAME:MouseControlling>
)
