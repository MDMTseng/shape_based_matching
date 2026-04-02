#pragma once
/// @file icp_refine.h
/// @brief Edge-based ICP pose refinement for shape matching.
/// Refines coarse (x, y, angle) from template matching to sub-pixel/sub-degree.
///
/// Three refinement levels:
///   1. SO2 ICP: refine (x, y, theta) — rotation + translation
///   2. SO2 + subpixel: + subpixel edge localization via Hessian
///   3. Sim2 ICP: refine (x, y, theta, scale) — adds scale factor

#include <opencv2/core.hpp>
#include <vector>
#include <cmath>

namespace icp_refine {

struct Pose2D {
    float x, y;     ///< Translation
    float angle;    ///< Rotation in degrees
    float scale;    ///< Scale factor (1.0 = no scaling)
    float fitness;  ///< Fraction of inlier points (0-1)
    float rmse;     ///< Root mean square error of inliers

    Pose2D() : x(0), y(0), angle(0), scale(1.0f), fitness(0), rmse(0) {}
    Pose2D(float x_, float y_, float a_, float s_ = 1.0f)
        : x(x_), y(y_), angle(a_), scale(s_), fitness(0), rmse(0) {}
};

struct ICPConfig {
    int max_iterations = 30;
    float max_dist = 10.0f;        ///< Max correspondence distance (pixels)
    float convergence_rmse = 1e-3f;
    float convergence_fitness = 1e-3f;
    bool use_subpixel = false;     ///< Subpixel edge refinement via Hessian
    bool use_scale = false;        ///< Sim2 (with scale) vs SO2 (no scale)
};

/// Build edge scene from Sobel derivatives.
/// Computes Canny edges + normal directions for point-to-plane ICP.
struct EdgeScene {
    cv::Mat edge_map;       ///< CV_8U Canny edge mask
    cv::Mat normal_x;       ///< CV_32F normal x at each edge pixel
    cv::Mat normal_y;       ///< CV_32F normal y at each edge pixel
    cv::Mat closest_x;      ///< CV_32F closest edge x (distance field)
    cv::Mat closest_y;      ///< CV_32F closest edge y (distance field)
    int width, height;

    /// Initialize from Sobel derivatives
    void build(const cv::Mat& sobel_dx, const cv::Mat& sobel_dy,
               float canny_low = 30, float canny_high = 60, float max_dist = 10);
};

/// Extract model edge points from a template at a given pose.
/// @param templ_edges  Edge pixel positions relative to template center.
/// @param pose         Initial pose (x, y, angle, scale).
/// @return Transformed model points in scene coordinates.
std::vector<cv::Point2f> transformModelPoints(
    const std::vector<cv::Point2f>& templ_edges, const Pose2D& pose);

/// Run ICP refinement.
/// @param model_points  Template edge points in scene coords (initial pose).
/// @param scene         Pre-built edge scene.
/// @param initial_pose  Coarse pose from template matching.
/// @param config        ICP parameters.
/// @return Refined pose with fitness and RMSE.
Pose2D refine(const std::vector<cv::Point2f>& templ_edges,
              const EdgeScene& scene,
              const Pose2D& initial_pose,
              const ICPConfig& config = ICPConfig());

} // namespace icp_refine
