# 最低版本要求
cmake_minimum_required(VERSION 3.20)
# 项目信息
project(global_memory LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_80 --ptxas-options=-v")
# set(CMAKE_CUDA_FLAGS_DEBUG="-G -g -O0")
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else()
message(FATAL_ERROR "Cannot find CUDA")
endif()

# 添加可执行文件
add_executable(global_memory global_memory.cu common.cuh)
target_include_directories(global_memory PRIVATE ${CUDA_INCLUDE_DIRS})
# 链接库
target_link_libraries(global_memory PRIVATE ${CUDA_LIBRARIES})