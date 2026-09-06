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
    float score_floor = 0.0f;      ///< Worst self-correlation under the coarse-error
                                   ///< angle envelope (set at setup). A runtime match
                                   ///< scoring far below this matched the wrong place.
    /// Gradient PCA of the UNROTATED template patch at this point, computed once
    /// at addModel (templatePCA) instead of once per candidate per re-match. It is
    /// a function of the template alone. pca_h is the patch half-size it was
    /// computed with; refineROI uses the cache only when its own half matches.
    bool pca_valid = false;
    int pca_h = 0;
    float pca_eig[2] = {0, 0};
    cv::Point2f pca_vec[2] = {cv::Point2f(1, 0), cv::Point2f(0, 1)};
};

/// The patch half-size refineROI will use at (tx,ty): roi_half shrunk at the
/// template border. Exposed so callers can precompute per-point data with it.
int roiHalfAt(int tx, int ty, int cols, int rows, int roi_half);
/// Gradient PCA of templ_img's (2h x 2h) patch centred at (tx,ty), as refineROI does it.
void templatePCA(const cv::Mat& templ_img, int tx, int ty, int h, float eigvals[2], cv::Point2f eigvecs[2]);

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
    bool weight_by_lock = false;      ///< Weight each constraint by the sample point's
                                      ///< self-match distinctiveness (lock_major/minor)
                                      ///< instead of a flat 1.0 — down-weights ambiguous
                                      ///< points without removing them.
    bool iterative_rematch = false;   ///< Re-match correspondences EVERY iteration at
                                      ///< the current (improved) pose, ICP-style,
                                      ///< instead of fixing them after iter 0. Lets the
                                      ///< match re-centre and correct along-edge / large
                                      ///< init error progressively (esp. for 1D edges).
    bool edge_collapse = false;       ///< For edge points: run the fast 2D matchTemplate,
                                      ///< then COLLAPSE (project) the 2D response along the
                                      ///< edge tangent into a 1D profile and take the
                                      ///< subpixel peak across the normal. Speed ~= 2D,
                                      ///< but denoised (tangent sum) + no along-edge peak
                                      ///< ambiguity. Best of 2D speed and 1D constraint.
    bool edge_1d_match = false;       ///< For edge (non-corner) points, match a 1D
                                      ///< intensity profile along the normal instead of
                                      ///< a 2D template match. Faster + avoids the
                                      ///< ill-defined along-edge peak (edge constrains
                                      ///< only across its normal).
    bool reject_low_score = false;    ///< Drop a matched point whose peak correlation
                                      ///< is below score_floor * reject_pct (it matched
                                      ///< the wrong place — gross outlier from large
                                      ///< coarse-init error). Calibrated per point.
    float reject_pct = 0.8f;          ///< Acceptance fraction of score_floor.
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
/// @param out_residual  If non-null, receives the mean |point-to-line| fit residual
///   (px) of the matched points at the final pose. Low = the points agree on a
///   consistent pose (trustworthy); high = they disagree (occlusion / gross
///   mismatch / completely-off match) — use it as a per-result confidence signal.
cv::Vec3f refineROI(const cv::Mat& templ_img,
                    const cv::Mat& scene_img,
                    const std::vector<SamplePoint>& sample_points,
                    const cv::Vec3f& initial_pose,
                    const ROIConfig& config = ROIConfig(),
                    float* out_residual = nullptr);

} // namespace roi_refine
