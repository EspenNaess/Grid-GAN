cmake_minimum_required(VERSION 3.5)

project(MasterPreprocessing)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Boost_INCLUDE_DIRS} )

link_directories(${CMAKE_SOURCE_DIR}/lib)

# Local source files here
SET(TARGET_SRC
    grid_framework_ops.cpp
    basic_img_ops.cpp
    )

add_executable(MasterPreprocessing preprocessing.cpp)
target_link_libraries( MasterPreprocessing ${OpenCV_LIBS} )
target_link_libraries( MasterPreprocessing Boost::filesystem)

target_compile_features(MasterPreprocessing PUBLIC cxx_std_11)
