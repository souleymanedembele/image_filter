cmake_minimum_required(VERSION 3.27)
project(image_filter)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/includes ${CMAKE_SOURCE_DIR}/images)

add_executable(image_filter includes/utils/ImageReader.h src/utils/ImageReader.cpp
        src/main.cpp)

target_link_libraries(image_filter ${OpenCV_LIBS})
