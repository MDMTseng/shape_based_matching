#pragma once
/// @file roi_refine.h
/// @brief ROI-based pose refinement: template match + PCA constraint + rigid solve.
///
/// Instead of iterative ICP with closest-edge lookup, this approach:
/// 1. Selects ~10-20 critical points (corners + well-spaced edges)
/// 2. Matches each via small ROI template match (subpixel)
/// 3. PCA on each ROI determines 1D (edge) or 2D (corner) constraint
/// 4. Single rigid body solve (not iterative)

#include <opencv2/core.hpp>
#include <vector>

namespace roi_refine {

/// A sample point on the template for ROI-based refinement.
struct SamplePoint {
    cv::Point2f pos;          ///< Position relative to template center
    int roi_half = 15;        ///< Half-size of the ROI patch
    float lock_major = 1.0f;  ///< Match locking strength in primary direction
    float lock_minor = 0.0f;  ///< Match locking strength in secondary direction
    cv::Point2f lock_normal{0,0};  ///< Precomputed constraint normal (from response surface)
    cv::Point2f lock_tangent{0,0}; ///< Precomputed constraint tangent
    bool lock_is_corner = false;   ///< Precomputed corner classification
};

/// Result of ROI matching + PCA for one sample point.
struct Constraint {
    cv::Point2f src;          ///< Source position (template, absolute coords)
    cv::Point2f dst;          ///< Matched position in scene (subpixel)
    cv::Point2f normal;       ///< Constraint direction (PCA eigenvector)
    float weight;             ///< Constraint weight (from eigenvalue)
};

/// Configuration for ROI refinement.
struct ROIConfig {
    int max_points = 20;          ///< Max sample points to use
    int max_iters = 5;            ///< Number of match-solve iterations
    int roi_half = 15;            ///< Half-size of ROI patch (pixels)
    int search_half = 20;         ///< Half-size of search region around expected position
    float corner_eigen_ratio = 1.5f;  ///< PCA eigenvalue ratio threshold:
                                      ///< < ratio → corner (add both eigenvectors)
                                      ///< > ratio → edge (only normal eigenvector)
    bool verbose = false;             ///< Print debug info to stderr
};

/// Select critical sample points from refine points.
/// Picks corners first, then well-spaced edge points.
std::vector<SamplePoint> selectCriticalPoints(
    const std::vector<cv::Point2f>& edge_positions,
    const std::vector<float>& cornerness,
    int max_points, int templ_width, int templ_height);

/// Refine pose using ROI template matching + PCA constraints.
/// @param templ_img      Template image (for ROI extraction).
/// @param scene_img      Scene image (for ROI matching).
/// @param sample_points  Critical points to match.
/// @param initial_pose   Coarse pose (center_x, center_y, angle_deg).
/// @param config         ROI refinement parameters.
/// @return Refined pose (center_x, center_y, angle_deg).
cv::Vec3f refineROI(const cv::Mat& templ_img,
                    const cv::Mat& scene_img,
                    const std::vector<SamplePoint>& sample_points,
                    const cv::Vec3f& initial_pose,
                    const ROIConfig& config = ROIConfig());

} // namespace roi_refine
