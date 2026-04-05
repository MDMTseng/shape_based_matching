#pragma once
/// @file shape_matcher.h
/// @brief High-level shape-based matching API.
///
/// Workflow:
///   OFFLINE: extractFeatures() -> FeatureSet -> save()
///   ONLINE:  load() -> addModel() -> match()

#include <opencv2/core.hpp>
#include "../icp_refine.h"
#include <string>
#include <vector>
#include <memory>

namespace sbm {

// ============================================================
// FeatureSet: serializable template features
// ============================================================

/// Extracted edge features from a template image.
/// Includes user-defined origin and angle offset for result mapping.
struct FeatureSet {
    /// Internal feature representation
    struct Feature {
        int x, y;           ///< Position relative to template image origin
        int label;          ///< 8-bin orientation label (0-7)
        float theta;        ///< Orientation in degrees
        float cornerness;   ///< 0 = pure edge, 1 = corner (from structure tensor)
    };

    struct PyramidLevel {
        std::vector<Feature> features;
        int tl_x, tl_y;    ///< Bounding box top-left
        int width, height;  ///< Bounding box size
        int level;          ///< Pyramid level index
    };

    std::vector<PyramidLevel> levels;   ///< Features per pyramid level
    int templ_width;                     ///< Original template image width
    int templ_height;                    ///< Original template image height

    /// Refinement point: edge or corner, for ICP.
    /// Positions relative to template center.
    struct RefinePt {
        float px, py;       ///< Position relative to template center
        float nx, ny;       ///< Normal direction (unit vector, for point-to-plane)
        float cornerness;   ///< 0 = pure edge, 1 = strong corner
        enum Type : uint8_t { EDGE = 0, CORNER = 1 } type;
    };

    /// All refinement points: dense Canny edges + Harris corners.
    std::vector<RefinePt> refine_points;

    /// Template image (stored for ROI-based refinement).
    cv::Mat templ_image;

    // --- User-defined reference frame ---

    /// Origin point in template image coords.
    /// Match results report position of this point in the scene.
    /// Default: template center (templ_width/2, templ_height/2).
    cv::Point2f origin;

    /// Cached optimized sample points (computed once, reused for matching).
    /// Populated by selectOptimizedPoints() or precomputeOptimizedPoints().
    /// Thread safety: pre-computed in addModel() before the parallel match region,
    /// then read-only during match(). No synchronization needed.
    mutable std::vector<cv::Point2f> cached_opt_points;
    mutable int cached_opt_max_points = 0;  ///< max_points arg used to compute cache

    /// Cached per-point lock values (parallel to cached_opt_points).
    /// Computed once at addModel() from matchTemplate response curvature.
    struct LockInfo {
        float major = 1.0f; float minor = 0.0f;
        cv::Point2f normal{1,0};   // eigenvector of weaker response (constraint direction)
        cv::Point2f tangent{0,1};  // eigenvector of stronger response
        bool is_corner = false;    // true if minor/major > threshold (2D lock)
    };
    mutable std::vector<LockInfo> cached_lock_info;

    /// Cached template EdgeScene for inverse ICP (built once at addModel time).
    mutable icp_refine::EdgeScene cached_templ_scene;
    mutable bool templ_scene_valid = false;

    /// Angle offset in degrees. Added to the raw matched angle.
    /// Example: if template was captured at 45° but you want 0° to mean
    /// "pointing right", set angle_offset = -45.
    float angle_offset = 0;

    // --- Methods ---

    /// Set the user-defined origin (reference point in template coords).
    void setOrigin(float x, float y) { origin = cv::Point2f(x, y); }

    /// Set angle offset (degrees). Applied to output angle.
    void setAngleOffset(float deg) { angle_offset = deg; }

    /// Save to file.
    bool save(const std::string& path) const;

    /// Load from file.
    static FeatureSet load(const std::string& path);

    /// Number of features at finest level.
    int numFeatures() const {
        return levels.empty() ? 0 : (int)levels[0].features.size();
    }

    /// Refinement quality score for a set of refine points.
    /// Combines geometric constraint quality + edge response strength.
    /// @return Score 0-100:
    ///   90-100: Excellent — well-constrained, strong edges, robust to ±20px/±20deg
    ///   70-89:  Good — reliable for most conditions
    ///   50-69:  Marginal — may fail under large perturbation or blur
    ///   0-49:   Poor — near-degenerate geometry, unreliable results
    /// Per-feature sensitivity analysis.
    /// Perturbs each sample point's match by ±1px in X and Y,
    /// measures how much the solved pose changes.
    /// Returns: per-point sensitivity (deg/px and px/px).
    struct FeatureSensitivity {
        cv::Point2f pos;          ///< Feature position (relative to center)
        float d_ang;              ///< Total angle sensitivity: |ang_from_dx| + |ang_from_dy|
                                  ///< How much the angle breaks from 1px error (deg/px)
        float d_pos;              ///< Total position sensitivity: hypot(pos_from_dx, pos_from_dy)
                                  ///< How much the position breaks from 1px error (px/px)
        float leverage;           ///< Distance from rotation center (angular torque arm)
    };

    struct SensitivityReport {
        std::vector<FeatureSensitivity> features;
        float worst_angle_sens;   ///< Worst single-feature angle sensitivity
        float worst_pos_sens;     ///< Worst single-feature position sensitivity
        float mean_angle_sens;
        float mean_pos_sens;
        int num_fragile;          ///< Features with sensitivity > 1.0 deg/px
        std::string diagnosis;
    };

    /// Select optimized sample points for ROI refinement.
    /// Uses sensitivity-balanced iterative optimization:
    /// greedy initial selection, then swap least/best to equalize sensitivity.
    /// @param max_points  Target number of points.
    /// @return Positions relative to template center.
    std::vector<cv::Point2f> selectOptimizedPoints(int max_points = 8) const;

    /// V3: Match-confidence augmented D-optimal selection.
    /// Same two-phase structure as V1, but weighted by empirical matchTemplate
    /// reliability (peak sharpness + noise stability). Avoids selecting
    /// geometrically optimal but texturally ambiguous points.
    /// @param max_points    Target number of points.
    /// @param noise_sigma   Noise level for stability testing (default 30).
    std::vector<cv::Point2f> selectOptimizedPointsV3(int max_points = 8, float noise_sigma = 30.0f) const;

    /// Multi-start hat matrix leverage swap selection (V2).
    /// Runs num_restarts random restarts of Fedorov exchange, picks the set
    /// with lowest worst_ang sensitivity.
    /// @param max_points    Target number of points.
    /// @param num_restarts  Number of random restarts (default 20).
    std::vector<cv::Point2f> selectOptimizedPointsV2(int max_points = 8, int num_restarts = 20) const;

    /// Run sensitivity analysis on auto-selected sample points.
    /// Simulates the ROI rigid solve with perturbed correspondences.
    /// @param skip_index  If >= 0, exclude this feature index from the solve.
    SensitivityReport analyzeSensitivity(int skip_index = -1) const;

    /// Geometric constraint quality analysis.
    /// Analyzes the Fisher information matrix J^TJ to measure how well
    /// a point set constrains the 3 DOF (θ, tx, ty).
    struct ConstraintAnalysis {
        // Information matrix eigenvalues (sorted: λ1 ≥ λ2 ≥ λ3)
        float info_eigenvalues[3];     ///< Constraint strength along principal directions
        float condition_number;         ///< λ_max/λ_min — isotropy (1=perfect, ∞=degenerate)

        // Per-DOF constraint strength (diagonal of (J^TJ)^{-1})
        float sigma_theta;             ///< Angle uncertainty (deg) per 1px matching noise
        float sigma_tx;                ///< X translation uncertainty (px) per 1px noise
        float sigma_ty;                ///< Y translation uncertainty (px) per 1px noise

        // Covariance ellipse (2D position uncertainty)
        float ellipse_major;           ///< Major axis of position error ellipse (px)
        float ellipse_minor;           ///< Minor axis of position error ellipse (px)
        float ellipse_angle;           ///< Orientation of major axis (deg)

        // Geometric properties
        float normal_spread;           ///< Angular spread of normal directions (deg, 0=all parallel, 180=ideal)
        float spatial_spread;          ///< RMS distance of points from centroid (px)
        float mean_leverage;           ///< Mean distance from rotation center (px)
        int   num_corners;             ///< Number of corner points (2D constraint)
        int   num_edges;               ///< Number of edge points (1D constraint)
        int   num_points;              ///< Total points

        // Per-point info
        struct PointInfo {
            cv::Point2f pos;           ///< Position relative to center
            cv::Point2f normal;        ///< Primary constraint normal direction
            float leverage;            ///< Distance from rotation center
            float info_contribution;   ///< det(I_with) / det(I_without) — marginal information gain
            bool is_corner;            ///< True if 2D lock (both Hessian eigenvalues significant)
            float lock_major;          ///< Locking strength in primary direction (Hessian eigenvalue)
            float lock_minor;          ///< Locking strength in secondary direction
            float lock_ratio;          ///< lock_major / lock_minor (1 = isotropic 2D lock, ∞ = 1D edge)
        };
        std::vector<PointInfo> points;

        std::string summary;           ///< Human-readable summary
    };

    /// Analyze geometric constraint quality of a point set.
    /// If custom_points is empty, uses the auto-selected optimized points.
    ConstraintAnalysis analyzeConstraints(const std::vector<cv::Point2f>& custom_points = {}) const;
};

/// Extract features from a template image.
/// @param templ_gray  Grayscale template image (CV_8U).
/// @param mask        Object mask (255 = object, 0 = background). Optional.
/// @param num_features  Max features to extract (0 = auto).
/// @param pyramid_T   Decimation factors per pyramid level (e.g., {4, 8}).
FeatureSet extractFeatures(const cv::Mat& templ_gray,
                           const cv::Mat& mask = cv::Mat(),
                           int num_features = 128,
                           const std::vector<int>& pyramid_T = {4, 8});

// ============================================================
// ModelConfig: how to generate rotation/scale/flip variants
// ============================================================

struct AngleRange {
    float start = 0;       ///< Start angle (degrees)
    float end = 360;       ///< End angle (degrees, exclusive)
    float step = 2;        ///< Step size (degrees)
};

struct ScaleRange {
    float min = 1.0f;      ///< Minimum scale
    float max = 1.0f;      ///< Maximum scale (1.0 = no scaling)
    float step = 0.05f;    ///< Scale step
};

struct ModelConfig {
    AngleRange angle;
    ScaleRange scale;
    bool flip = false;     ///< Also match horizontally flipped version
};

// ============================================================
// MatchConfig: matching parameters
// ============================================================

enum class RefineMode {
    None,               ///< Raw coarse result only
    BiasCorrection,     ///< Apply calibrated angle bias correction (free)
    ICP_Sparse,         ///< ICP using sparse matching features (~0.1ms/obj, ~2deg)
    ICP,                ///< ICP using dense Canny edges (~0.3ms/obj, <0.5deg)
    ROI                 ///< ROI template match + PCA constraint (~0.05ms/obj, <0.5deg)
                        ///< Selects ~10-20 critical points (corners + spaced edges),
                        ///< matches each via small ROI template match with subpixel,
                        ///< PCA determines 1D/2D constraint, single rigid solve.
};

struct MatchConfig {
    float min_score = 50.0f;       ///< Minimum similarity (0-100)
    int max_results = 0;           ///< Max results (0 = unlimited)
    float nms_radius = -1;         ///< Spatial NMS radius (-1 = auto from template size)
    float nms_angle = 360.0f;      ///< NMS angle tolerance (deg). Matches at same position
                                    ///< but different angles (>nms_angle) are kept.
                                    ///< Set to 30 for angle-aware NMS (noisy scenes).
    float match_scale = 1.0f;      ///< Scene downscale for faster matching (e.g., 0.7)
                                    ///< Template features are scaled at match time.
                                    ///< Use with ROI refine for full-res accuracy.
    RefineMode refine = RefineMode::ICP;

    // ICP config (used when refine == ICP)
    int icp_iterations = 30;
    float icp_max_dist = 10.0f;

    /// Gaussian blur kernel size before gradient computation.
    /// Larger = more noise-robust but blurs fine edges.
    /// Default 7 handles noise up to ~30 sigma.
    /// Use 11-15 for heavy noise (sigma 40+).
    int blur_kernel_size = 7;

    // Edge gradient thresholds for feature extraction
    float weak_threshold = 50.0f;    ///< Min gradient magnitude to be an edge candidate
    float strong_threshold = 80.0f;  ///< Gradient magnitude for strong edge preference

    bool skip_voting = false;        ///< Skip 3x3 neighborhood voting (saves ~7ms at 20MP).
                                     ///< Safe to enable with higher edge thresholds (50/80).
};

// ============================================================
// MatchResult: what you get back
// ============================================================

struct MatchResult {
    std::string model_name;    ///< Which template matched
    float x, y;                ///< Position of user-defined origin in scene
    float angle;               ///< Orientation (degrees, with user angle_offset applied)
    float scale;               ///< Matched scale factor
    bool flipped;              ///< Whether this is a flipped match
    float score;               ///< Confidence (0-100)
};

// ============================================================
// ShapeMatcher: main class
// ============================================================

class ShapeMatcher {
public:
    /// Constructor with matching configuration.
    explicit ShapeMatcher(const MatchConfig& config = MatchConfig());
    ~ShapeMatcher();

    /// Register a model for matching.
    /// Generates rotation/scale/flip variants from the feature set.
    /// @param name       Unique model name (returned in results).
    /// @param features   Pre-extracted features (from extractFeatures or load).
    /// @param config     Rotation/scale/flip configuration.
    /// @return Number of variants generated, or -1 on failure.
    int addModel(const std::string& name,
                 const FeatureSet& features,
                 const ModelConfig& config = ModelConfig());

    /// Match all registered models against a scene image.
    /// @param scene  Grayscale scene image (CV_8U).
    /// @return Matches sorted by score descending, NMS applied.
    std::vector<MatchResult> match(const cv::Mat& scene) const;

    /// Get number of registered models.
    int numModels() const;

    /// Get total number of template variants across all models.
    int numTemplates() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sbm
