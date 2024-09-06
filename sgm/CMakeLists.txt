cmake_minimum_required(VERSION 3.21)
project(sgmcuda)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12/bin/nvcc)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "52;61;72;75;86")
endif()

enable_language(CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(LibSGM_DIR "/home/william/Codes/libSGM/build/install")
set(LibSGM_INCLUDE_DIRS ${LibSGM_DIR}/include)
set(LibSGM_LIBRARY ${LibSGM_DIR}/lib)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(
    ${PROJECT_NAME}
    # main.cpp
    main_debug.cpp
    sample_common.cpp
    sample_common.h
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${LibSGM_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CUDA_INCLURE_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    ${OpenCV_LIBS}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${CMAKE_CUDA_RUNTIME_LIBRARY}
    ${LibSGM_LIBRARY}/libsgm.so
)



