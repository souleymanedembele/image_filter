cmake_minimum_required(VERSION 3.28)
project(Image_Filtering)

set(CMAKE_CXX_STANDARD 17)

# Include FetchContent module
include(FetchContent)

# Fetch Google Test
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# Prevent Google Test from overriding our compiler/linker options
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Fetch OpenCV
FetchContent_Declare(
        opencv
        URL https://github.com/opencv/opencv/archive/4.9.0.zip
)
# After calling FetchContent_MakeAvailable for googletest, do the same for OpenCV
FetchContent_GetProperties(opencv)
if(NOT opencv_POPULATED)
    FetchContent_Populate(opencv)
    # Add OpenCV as a subdirectory so its targets are available
    add_subdirectory(${opencv_SOURCE_DIR} ${opencv_BINARY_DIR})
endif()

# Create your main executable with just main.cpp
add_executable(Image_Filtering main.cpp
        src/utils/utils.h)

# Define a test executable
enable_testing()
add_executable(Image_Filtering_tests test_main.cpp)

# Include directories for Google Test and OpenCV
include_directories(${googletest_SOURCE_DIR}/include ${googletest_BINARY_DIR} ${opencv_SOURCE_DIR}/include)
#include_directories(${googletest_SOURCE_DIR}/include ${googletest_BINARY_DIR})

# Link Google Test to the test executable
target_link_libraries(Image_Filtering_tests gtest_main)

# Link OpenCV to the main executable (gtest and gtest_main not needed here)
target_link_libraries(Image_Filtering ${OpenCV_LIBS})

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( Image_Filtering ${OpenCV_LIBS} )

# Add test cases
include(GoogleTest)
gtest_discover_tests(Image_Filtering_tests)

include_directories(${CMAKE_SOURCE_DIR}/images)