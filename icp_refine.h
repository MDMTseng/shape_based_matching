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
    float point_to_point_weight = 0.1f; ///< Blend point-to-point with point-to-plane.
                                        ///< 0 = pure point-to-plane (slides on edges),
                                        ///< 1 = equal weight. 0.1 is a good default.
    float normal_angle_thresh = 45.0f;  ///< Max angle (degrees) between model and scene
                                        ///< edge normals to accept a correspondence.
    bool use_cornerness = false;   ///< Use per-feature cornerness to blend p2p/p2plane.
                                        ///< Edge features (cornerness~0) → point-to-plane only.
                                        ///< Corner features (cornerness~1) → add point-to-point.
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

/// Model edge: position + normal direction
struct EdgePoint {
    cv::Point2f pos;       ///< Position relative to template center
    cv::Point2f normal;    ///< Edge normal direction (unit vector)
    float cornerness = 0;  ///< 0 = pure edge (1D), 1 = corner (2D constraint)
};

/// Extract model edge points with normals from a template image.
std::vector<EdgePoint> extractModelEdges(const cv::Mat& templ_gray);

/// Transform model points to scene coordinates at a given pose.
std::vector<cv::Point2f> transformModelPoints(
    const std::vector<cv::Point2f>& templ_edges, const Pose2D& pose);

/// Refine using model edges with normals (better correspondence filtering).
Pose2D refineWithNormals(const std::vector<EdgePoint>& model_edges,
                         const cv::Mat& scene_dx, const cv::Mat& scene_dy,
                         const Pose2D& initial_pose,
                         int templ_size,
                         int roi_margin = 20,
                         const ICPConfig& config = ICPConfig());

/// Run ICP refinement using pre-built full scene.
Pose2D refine(const std::vector<cv::Point2f>& templ_edges,
              const EdgeScene& scene,
              const Pose2D& initial_pose,
              const ICPConfig& config = ICPConfig());

/// Run ICP refinement using local ROI only.
/// Crops a patch around the match position from scene Sobel derivatives,
/// builds a local EdgeScene, and refines within that patch.
/// Much faster than full-scene build (~0.1ms per object vs ~40ms for full scene).
/// @param templ_edges   Template edge points relative to center.
/// @param scene_dx      Full scene Sobel dx (CV_16S).
/// @param scene_dy      Full scene Sobel dy (CV_16S).
/// @param initial_pose  Coarse pose (x, y, angle) from template matching.
/// @param roi_margin    Extra margin around template bounding box (pixels).
/// @param config        ICP parameters.
Pose2D refineLocal(const std::vector<cv::Point2f>& templ_edges,
                   const cv::Mat& scene_dx, const cv::Mat& scene_dy,
                   const Pose2D& initial_pose,
                   int templ_size,
                   int roi_margin = 20,
                   const ICPConfig& config = ICPConfig());

/// Inverse ICP: build EDT on template, match scene edges against it.
/// Scene edges are inverse-transformed to template space for correspondence.
/// Template EDT is clean (no other objects), potentially better at corners.
Pose2D refineInverse(const cv::Mat& templ_gray,
                     const cv::Mat& scene_gray,
                     const Pose2D& initial_pose,
                     int roi_margin = 20,
                     const ICPConfig& config = ICPConfig());

/// Inverse ICP with pre-built template EdgeScene (avoids rebuilding EDT per call).
/// @param templ_scene  Pre-built EdgeScene from the template (build once at addModel time).
/// @param templ_width  Template width (for center offset).
/// @param templ_height Template height (for center offset).
/// @param scene_gray   Scene grayscale image.
/// @param initial_pose Coarse pose from template matching.
/// @param roi_margin   Extra margin around template bounding box.
/// @param config       ICP parameters.
Pose2D refineInverse(const EdgeScene& templ_scene,
                     int templ_width, int templ_height,
                     const cv::Mat& scene_gray,
                     const Pose2D& initial_pose,
                     int roi_margin = 20,
                     const ICPConfig& config = ICPConfig());

/// Inverse ICP with pre-built template EdgeScene and scene Sobel derivatives.
/// Avoids both template EDT rebuild and scene Gaussian+Sobel recomputation.
Pose2D refineInverse(const EdgeScene& templ_scene,
                     int templ_width, int templ_height,
                     const cv::Mat& scene_dx, const cv::Mat& scene_dy,
                     const Pose2D& initial_pose,
                     int templ_diag,
                     int roi_margin = 20,
                     const ICPConfig& config = ICPConfig());

/// Build a template EdgeScene from a grayscale template image.
/// Call once per template at addModel time, then pass to refineInverse.
EdgeScene buildTemplateScene(const cv::Mat& templ_gray, float max_dist = 20.0f);

} // namespace icp_refine
