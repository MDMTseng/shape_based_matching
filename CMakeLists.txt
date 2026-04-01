cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 14)
project(shape_based_matching)

# debug or release
SET(CMAKE_BUILD_TYPE "Release")

# Platform-specific SIMD flags
IF(MSVC)
    # MSVC: use /arch:AVX2 for AVX2 support + define __AVX2__ for our code
    SET(PLATFORM_COMPILE_FLAGS "/arch:AVX2")
    add_definitions(-D__AVX2__)
    SET(COMMON_COMPILE_FLAGS "/W3 /O2")
    SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS}")
    SET(CMAKE_CXX_FLAGS_RELEASE "/O2")
ELSEIF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    SET(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
    SET(COMMON_COMPILE_FLAGS "-Wall -Wno-sign-compare")
    SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
    SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
ELSE()
    SET(PLATFORM_COMPILE_FLAGS "-march=native -mavx2")
    SET(COMMON_COMPILE_FLAGS "-fopenmp -Wall -Wno-sign-compare")
    SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
    SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
ENDIF()

# opencv -- manual linking for prebuilt Windows package
set(OpenCV_INCLUDE_DIRS "C:/opencv/opencv/build/include")
set(OpenCV_LIBS "C:/opencv/opencv/build/x64/vc16/lib/opencv_world4100.lib")
include_directories(${OpenCV_INCLUDE_DIRS})

# include MIPP headers
include_directories (${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/")

# test exe
add_executable(${PROJECT_NAME}_test line2Dup.cpp test.cpp)
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS})

# AVX2 benchmark
add_executable(bench_avx2 line2Dup.cpp bench_avx2.cpp)
target_link_libraries(bench_avx2 ${OpenCV_LIBS})

# Robustness test
add_executable(test_robustness line2Dup.cpp test_robustness.cpp)
target_link_libraries(test_robustness ${OpenCV_LIBS})
