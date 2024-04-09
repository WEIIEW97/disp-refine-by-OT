cmake_minimum_required(VERSION 3.5.1)
project(ot)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB sources ${CMAKE_CURRENT_LIST_DIR}/ot/*.cpp)
file(GLOB headers ${CMAKE_CURRENT_LIST_DIR}/ot/*.h)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

add_executable(${PROJECT_NAME} 
        main.cpp
        ${sources}
)

target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${HDF5_LIBRARIES} 
        ${HDF5_CXX_LIBRARIES}
)