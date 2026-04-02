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

/// Constraint quality analysis for a set of sample points.
struct ConstraintQuality {
    float condition_number;     ///< Ratio of max/min singular value of the constraint matrix.
                                ///< < 10: excellent, 10-50: acceptable, > 50: poor, inf: degenerate.
    float angle_coverage;       ///< Range of PCA normal directions in degrees (0-180).
                                ///< > 60: good, 30-60: marginal, < 30: poor (near-parallel edges).
    int num_edge;               ///< Number of edge constraints (1D).
    int num_corner;             ///< Number of corner constraints (2D).
    int num_directions;         ///< Number of distinct edge directions (binned to 15 deg).
    bool is_valid;              ///< True if constraints can determine (x, y, theta).
    std::string diagnosis;      ///< Human-readable diagnosis.
};

/// Validate whether a set of sample points provides sufficient
/// geometric constraint for pose refinement.
/// @param sample_points  Points to validate.
/// @param templ_img      Template image (for PCA computation).
/// @param config         ROI config (for roi_half, corner threshold).
/// @return Quality analysis with diagnosis.
ConstraintQuality validateConstraints(
    const std::vector<SamplePoint>& sample_points,
    const cv::Mat& templ_img,
    const ROIConfig& config = ROIConfig());

} // namespace roi_refine
