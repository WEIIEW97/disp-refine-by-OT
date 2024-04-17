cmake_minimum_required(VERSION 3.5.1)
project(ot)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE sources ${CMAKE_CURRENT_LIST_DIR}/ot/*.cpp)
file(GLOB_RECURSE headers ${CMAKE_CURRENT_LIST_DIR}/ot/*.h)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
if(USE_HDF5)
        find_package(HDF5 REQUIRED COMPONENTS CXX)
endif()
find_package(ZLIB REQUIRED)

add_executable(${PROJECT_NAME} 
        main.cpp
        ${sources}
)

if(USE_HDF5)
        target_include_directories(
                ${PROJECT_NAME}
                PRIVATE
                ${OpenCV_INCLUDE_DIRS}
                ${EIGEN3_INCLUDE_DIR}
                ${HDF5_INCLUDE_DIRS}
                # ${ZLIB_INCLUDE_DIRS}
        )
endif()

target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${ZLIB_INCLUDE_DIRS}
)

if(USE_HDF5)
        target_link_libraries(
                ${PROJECT_NAME}
                ${OpenCV_LIBS}
                ${HDF5_LIBRARIES} 
                ${HDF5_CXX_LIBRARIES}
                # ${ZLIB_LIBRARIES}
        )
endif()
target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${ZLIB_LIBRARIES}
)