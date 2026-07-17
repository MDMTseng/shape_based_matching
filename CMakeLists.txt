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
    SET(COMMON_COMPILE_FLAGS "/W3 /O2 /openmp")
    SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS}")
    SET(CMAKE_CXX_FLAGS_RELEASE "/O2")
ELSEIF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64|arm64")
    # aarch64 (e.g. Raspberry Pi 5, Apple Silicon Linux): NEON is baseline, no
    # -mfpu needed; -mavx2 would be an invalid option. line2Dup's __aarch64__
    # NEON paths light up under -march=native.
    SET(PLATFORM_COMPILE_FLAGS "-march=native")
    SET(COMMON_COMPILE_FLAGS "-fopenmp -Wall -Wno-sign-compare")
    SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
    SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
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

# OpenCV: prefer a real find_package (Linux/macOS, or Windows with OpenCV_DIR
# set); fall back to the prebuilt Windows package path for the original setup.
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    set(OpenCV_INCLUDE_DIRS "C:/opencv/opencv/build/include")
    set(OpenCV_LIBS "C:/opencv/opencv/build/x64/vc16/lib/opencv_world4100.lib")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# include MIPP headers
include_directories (${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/")

# test exe
add_executable(${PROJECT_NAME}_test line2Dup.cpp test.cpp)
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS})

# AVX2 benchmark
add_executable(bench_avx2 line2Dup.cpp tests/bench_avx2.cpp)
target_link_libraries(bench_avx2 ${OpenCV_LIBS})

# Robustness test
add_executable(test_robustness line2Dup.cpp tests/test_robustness.cpp)
target_link_libraries(test_robustness ${OpenCV_LIBS})

# Preprocessing stage profiler
add_executable(bench_preprocess tests/bench_preprocess.cpp)
target_link_libraries(bench_preprocess ${OpenCV_LIBS})

# Per-stage profiler (clean vs noisy)
add_executable(bench_profile line2Dup.cpp tests/bench_profile.cpp)
target_link_libraries(bench_profile ${OpenCV_LIBS})

# Rotation-sweep match-throughput bench. Self-labels its SIMD backend
# (AVX2 / NEON / scalar) and prints a per-stage profile. Cross-platform;
# used to compare x86 AVX2 vs ARM NEON. See tests/bench_rotation.md.
add_executable(bench_rotation line2Dup.cpp tests/bench_rotation.cpp)
target_link_libraries(bench_rotation ${OpenCV_LIBS})

# Realistic multi-object bench: 5 objects x 360deg = 1800 templates, matched
# against 1.2/5/20 MP noisy scenes, with and without ROI refine (high-level
# ShapeMatcher API). Self-labels SIMD backend. See tests/bench_rotation.md.
add_executable(bench_multiobj line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/bench_multiobj.cpp)
target_link_libraries(bench_multiobj ${OpenCV_LIBS})

# Visual matching test with image output
add_executable(test_visual line2Dup.cpp icp_refine.cpp tests/test_visual.cpp)
target_link_libraries(test_visual ${OpenCV_LIBS})

# ICP accuracy sweep
add_executable(test_icp_accuracy line2Dup.cpp icp_refine.cpp tests/test_icp_accuracy.cpp)
target_link_libraries(test_icp_accuracy ${OpenCV_LIBS})

# Bias table
add_executable(test_bias_table line2Dup.cpp tests/test_bias_table.cpp)
target_link_libraries(test_bias_table ${OpenCV_LIBS})
add_executable(test_rotate_bias line2Dup.cpp tests/test_rotate_bias.cpp)
target_link_libraries(test_rotate_bias ${OpenCV_LIBS})
add_executable(test_bias_final line2Dup.cpp tests/test_bias_final.cpp)
target_link_libraries(test_bias_final ${OpenCV_LIBS})
add_executable(test_multires line2Dup.cpp icp_refine.cpp tests/test_multires.cpp)
target_link_libraries(test_multires ${OpenCV_LIBS})
add_executable(test_icp_bias line2Dup.cpp icp_refine.cpp tests/test_icp_bias.cpp)
target_link_libraries(test_icp_bias ${OpenCV_LIBS})

# New high-level API
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_executable(test_api line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_api.cpp)
target_link_libraries(test_api ${OpenCV_LIBS})

add_executable(test_simple line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_simple.cpp)
target_link_libraries(test_simple ${OpenCV_LIBS})

add_executable(test_regression line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_regression.cpp)
target_link_libraries(test_regression ${OpenCV_LIBS})

add_executable(test_variant line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_variant.cpp)
target_link_libraries(test_variant ${OpenCV_LIBS})

add_executable(test_20mp_noise line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_20mp_noise.cpp)
target_link_libraries(test_20mp_noise ${OpenCV_LIBS})

add_executable(test_selection_compare line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_selection_compare.cpp)
target_link_libraries(test_selection_compare ${OpenCV_LIBS})

add_executable(test_sweep_rotation line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_sweep_rotation.cpp)
target_link_libraries(test_sweep_rotation ${OpenCV_LIBS})

add_executable(test_sweep_noise line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_sweep_noise.cpp)
target_link_libraries(test_sweep_noise ${OpenCV_LIBS})

add_executable(test_sweep_scale line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_sweep_scale.cpp)
target_link_libraries(test_sweep_scale ${OpenCV_LIBS})

add_executable(test_diagnostics line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_diagnostics.cpp)
target_link_libraries(test_diagnostics ${OpenCV_LIBS})

add_executable(test_convergence line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_convergence.cpp)
target_link_libraries(test_convergence ${OpenCV_LIBS})

add_executable(test_constraints line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_constraints.cpp)
target_link_libraries(test_constraints ${OpenCV_LIBS})

add_executable(test_draw_roi line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_draw_roi.cpp)
target_link_libraries(test_draw_roi ${OpenCV_LIBS})

add_executable(test_lock_compare line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_lock_compare.cpp)
target_link_libraries(test_lock_compare ${OpenCV_LIBS})

add_executable(test_select_v2 line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_select_v2.cpp)
target_link_libraries(test_select_v2 ${OpenCV_LIBS})

add_executable(bench_full_profile line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/bench_full_profile.cpp)
target_link_libraries(bench_full_profile ${OpenCV_LIBS})

add_executable(bench_param_sweep line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/bench_param_sweep.cpp)
target_link_libraries(bench_param_sweep ${OpenCV_LIBS})

add_executable(bench_match_scale line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/bench_match_scale.cpp)
target_link_libraries(bench_match_scale ${OpenCV_LIBS})

# Flip-template bug experiment (see flip_template_problem.md). Hermetic; no data.
add_executable(test_flip_refine line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_flip_refine.cpp)
target_link_libraries(test_flip_refine ${OpenCV_LIBS})

# Flip-as-separate-model detection accuracy (see flip_template_problem.md). Hermetic.
add_executable(test_flip_separate line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_flip_separate.cpp)
target_link_libraries(test_flip_separate ${OpenCV_LIBS})

# Visualize the flag/F templates (features + ROI + centre) for the score-collapse probe.
add_executable(test_flag_viz line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_flag_viz.cpp)
target_link_libraries(test_flag_viz ${OpenCV_LIBS})

# Angle-sweep self-match probe: anti-aliasing vs per-angle template generation.
add_executable(test_flag_sweep line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_flag_sweep.cpp)
target_link_libraries(test_flag_sweep ${OpenCV_LIBS})

# Multi-template matching: several distinct shapes + a flip pair in one matcher.
add_executable(test_multi_template line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_multi_template.cpp)
target_link_libraries(test_multi_template ${OpenCV_LIBS})

# Refine-accuracy decomposition: per-shape, per-RefineMode origin/angle error.
add_executable(test_refine_accuracy line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_refine_accuracy.cpp)
target_link_libraries(test_refine_accuracy ${OpenCV_LIBS})

# ROI error decomposition: angle-quantization vs subpixel-match floor vs anti-aliasing.
add_executable(test_roi_error_decomp line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_roi_error_decomp.cpp)
target_link_libraries(test_roi_error_decomp ${OpenCV_LIBS})

# L-shape ROI outlier chase: 1-deg sweep, flag spikes.
add_executable(test_L_outlier line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_L_outlier.cpp)
target_link_libraries(test_L_outlier ${OpenCV_LIBS})

# ROI self-distinctiveness point filter (roi_distinct_pct) under noise.
add_executable(test_roi_distinct line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_roi_distinct.cpp)
target_link_libraries(test_roi_distinct ${OpenCV_LIBS})

# ROI per-point score-gate outlier rejection (roi_reject_low_score) under coarse init + noise.
add_executable(test_roi_reject line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_roi_reject.cpp)
target_link_libraries(test_roi_reject ${OpenCV_LIBS})

# ROI per-result residual confidence: detect off matches (occlusion/clutter).
add_executable(test_roi_confidence line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_roi_confidence.cpp)
target_link_libraries(test_roi_confidence ${OpenCV_LIBS})

# ROI auto-selection point overlap quantification.
add_executable(test_roi_overlap line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_roi_overlap.cpp)
target_link_libraries(test_roi_overlap ${OpenCV_LIBS})

# ROI min-spacing sweep: overlap vs conditioning knee.
add_executable(test_roi_spacing line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_roi_spacing.cpp)
target_link_libraries(test_roi_spacing ${OpenCV_LIBS})

# Edge-uniform (FPS over edges) vs corner-priority ROI point selection.
add_executable(test_edge_uniform line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_edge_uniform.cpp)
target_link_libraries(test_edge_uniform ${OpenCV_LIBS})

# 1D vs 2D ROI refine speed.
add_executable(test_roi_speed_1d line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_roi_speed_1d.cpp)
target_link_libraries(test_roi_speed_1d ${OpenCV_LIBS})

# Library edge-only ROI point selection (original 2D refine).
add_executable(test_edge_only line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_edge_only.cpp)
target_link_libraries(test_edge_only ${OpenCV_LIBS})

# Does ROI iteration need re-matching, or just re-solving?
add_executable(test_roi_iters line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_roi_iters.cpp)
target_link_libraries(test_roi_iters ${OpenCV_LIBS})

# ROI refine cost vs scene size (5MP).
add_executable(test_roi_largescene line2Dup.cpp icp_refine.cpp roi_refine.cpp shape_matcher.cpp tests/test_roi_largescene.cpp)
target_link_libraries(test_roi_largescene ${OpenCV_LIBS})
