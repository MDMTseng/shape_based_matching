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

// How many ROI refine points the matcher uses when the def names none.
//
// PUBLIC because the studio preview must ask for the SAME number. It used to
// ask for 16 while matching asked for 8, and selectOptimizedPoints caches by
// max_points with "the first call wins" (a cached 16 satisfies a later request
// for 8) -- so merely OPENING the studio changed the point set the localizer
// then ran with, and what the operator saw was never what the machine used.
constexpr int kDefaultOptPointsPublic = 8;

// The ROI search/template window half-size, mirrored from the file-static in
// shape_matcher.cpp. PUBLIC for the same class of reason as the count above: a
// debug dump that crops tiles with a different half-size than the matcher uses
// is a picture of something the matcher never looked at.
constexpr int kDefaultROIHalfPublic = 15;


// ============================================================
// FeatureSet: serializable template features
// ============================================================

/// Extracted edge features from a template image.
/// Includes user-defined origin and angle offset for result mapping.
#define SBM_HAS_USER_OPT_POINTS 1

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

    /// User-supplied ROI refine points (template px, relative to template
    /// center). When user_opt_points_set, selectOptimizedPoints() returns
    /// EXACTLY these -- an empty list means coarse-only (ROI refine is
    /// skipped). Absent (set=false) keeps the automatic selection.
    /// Reimplementation of an API lost with an unpushed commit (b987d179).
    std::vector<cv::Point2f> user_opt_points;
    bool user_opt_points_set = false;

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
        float score_floor = 0.0f;  // worst self-correlation under coarse-error angle envelope
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
    /// @param min_spacing Minimum pairwise distance (px) between selected points, to
    ///   stop their ROI windows from overlapping heavily. 0 = off (legacy), <0 = auto
    ///   (= ROI half-size), >0 = explicit.
    /// @param edge_only  Skip the corner phase and select EDGE points only (D-optimal).
    ///   Avoids corner-clustering degeneracy; fixes worst-case on shapes with clustered
    ///   corners while keeping the original 2D refine. Same per-point matching as before.
    /// @return Positions relative to template center.
    std::vector<cv::Point2f> selectOptimizedPoints(int max_points = 8, float min_spacing = 0.0f,
                                                   bool edge_only = false) const;

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
/// @param weak_thresh   Min gradient magnitude for an edge candidate.
/// @param strong_thresh Gradient magnitude for a strong-edge preference.
FeatureSet extractFeatures(const cv::Mat& templ_gray,
                           const cv::Mat& mask = cv::Mat(),
                           int num_features = 128,
                           const std::vector<int>& pyramid_T = {4, 8},
                           float weak_thresh = 30.0f,
                           float strong_thresh = 60.0f);

/// Keep only the ROTATION-STABLE features of an already-extracted set.
///
/// The matcher generates rotated templates by ANALYTICALLY rotating the base
/// features (position + orientation), so a feature on a corner/junction — whose
/// local orientation does NOT rotate rigidly — mismatches the real rotated scene
/// at off-grid angles. This measures, per feature, whether its analytically-
/// rotated orientation still matches the true orientation on the rotated template
/// across +-angle_range at angle_step (using the matcher's own quantizer; the
/// rotation-sign convention is auto-detected), and keeps the most stable
/// keep_frac. Curation-safe: it only SUBSETS the given features (never adds), so
/// a user-brushed selection is preserved, just trimmed to its rotation-robust
/// core. Enables COARSER ModelConfig.angle steps (fewer templates, less memory,
/// faster coarse match) at similar off-grid-angle robustness.
///
/// EFFECTIVENESS depends on the input: biggest gain on a QUALITY-FILTERED base
/// (fewer, stronger features — e.g. extract from a downscaled template, or with
/// a lower num_features), where the rotation signal is clean. On a dense native
/// extraction (many marginal features) the gain is modest. Measured (aarch64,
/// off-grid angles, sigma=10): from a 0.7-downscale-extracted base, keep_frac=0.5
/// at a 9-deg model step scores 94.5/92.3 (avg/worst) vs the full base's 89.8/84.0
/// at a 1-deg step (360 templates) — ~9x fewer templates at better robustness.
/// Recipe: extractFeatures(downscaled / low num_features) -> selectRotationStable
/// -> addModel with a coarse ModelConfig.angle step.
///
/// DISCRIMINABILITY WARNING: dropping features (keep_frac < 1) raises the score
/// at the correct location (less dilution) but LOWERS discriminability — in
/// cluttered / high-resolution multi-object scenes an aggressive keep_frac (0.5)
/// produces FALSE POSITIVES (e.g. 19 detections for 5 objects at 5 MP). The
/// template-count / speed win comes from the COARSE ANGLE STEP + quality base,
/// not from dropping features. So in real multi-object use keep keep_frac high
/// (>=0.75, or 1.0 to only re-rank without dropping); reserve small keep_frac
/// for isolated single-object matching where precision is not at risk.
///
/// @param fs           Features from extractFeatures (or a curated subset).
/// @param templ_gray   The template image the features were extracted from.
/// @param angle_range  Half-range of the rotation probe in degrees (e.g. 9).
/// @param angle_step   Probe step in degrees (e.g. 3).
/// @param keep_frac    Fraction of features to keep (0..1], per pyramid level.
/// @return A copy of `fs` with each level trimmed to its rotation-stable core.
FeatureSet selectRotationStable(const FeatureSet& fs,
                                const cv::Mat& templ_gray,
                                float angle_range = 9.0f,
                                float angle_step = 3.0f,
                                float keep_frac = 0.8f);

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
    ICP_Sparse,         ///< ICP using sparse matching features (~0.1ms/obj, ~2deg)
    ICP,                ///< ICP using dense Canny edges, integer EDT (~0.3ms/obj, <0.05deg, ~0.66px)
    ICP_Subpixel,       ///< ICP with subpixel EDT via facet model (~0.3ms/obj, ~0.1deg, ~0.11px)
    ROI                 ///< ROI template match + PCA constraint (~0.05ms/obj, <0.5deg)
                        ///< Selects ~10-20 critical points (corners + spaced edges),
                        ///< matches each via small ROI template match with subpixel,
                        ///< PCA determines 1D/2D constraint, single rigid solve.
};

struct MatchConfig {
    float min_score = 65.0f;       ///< Minimum similarity (0-100)
    int max_results = 0;           ///< Max results (0 = unlimited)
    float nms_radius = -1;         ///< Spatial NMS radius (-1 = auto from template size)
    float nms_radius_scale = 0.75f; ///< Auto NMS radius = min(w,h) * this (0.5=tight, 0.75=default, 1.0=aggressive)
    float nms_angle = 360.0f;      ///< NMS angle tolerance (deg). Matches at same position
                                    ///< but different angles (>nms_angle) are kept.
                                    ///< Set to 30 for angle-aware NMS (noisy scenes).
    float match_scale = 1.0f;      ///< Scene downscale for faster matching (e.g., 0.7)
                                    ///< Template features are scaled at match time.
                                    ///< Use with ROI refine for full-res accuracy.
    std::vector<int> T_levels = {4, 8};  ///< Pyramid decimation strides
    RefineMode refine = RefineMode::ICP;

    // ICP config (used when refine == ICP)
    int icp_iterations = 30;
    float icp_max_dist = 10.0f;

    /// ROI sample-point self-distinctiveness filter (0 = disabled).
    /// At addModel() each ROI point self-matches inside the template; its peak
    /// sharpness (response-Hessian major curvature) measures how well-localized the
    /// match is. Points whose distinctiveness < roi_distinct_pct * (max over points)
    /// are dropped from the ROI set (kept >= 4). Edge points (sharp across the edge,
    /// flat along it) are retained; only genuinely flat/ambiguous points are removed.
    /// Typical: 0.2-0.4. Setup-time only, zero per-match cost.
    float roi_distinct_pct = 0.0f;

    /// Weight ROI constraints by per-point self-match distinctiveness (lock_major)
    /// instead of a flat 1.0. Down-weights ambiguous points without removing them
    /// (keeps point-count redundancy). Softer alternative to roi_distinct_pct.
    bool roi_weight_by_distinct = false;

    /// Reject ROI points that matched the wrong place (gross outliers). At addModel
    /// each point's "score floor" is calibrated by self-correlating it under the
    /// coarse-localization angle error envelope (roi_reject_angle_tol); at match a
    /// point scoring below score_floor * roi_reject_pct is dropped. Targets the
    /// "large initial angle -> completely wrong match" case.
    bool roi_reject_low_score = false;
    float roi_reject_angle_tol = 3.0f;   ///< Coarse angle-error envelope (deg) for the floor.
    float roi_reject_pct = 0.8f;         ///< Accept if match score >= floor * this.

    /// Minimum pairwise spacing (px) between auto-selected ROI points, to limit ROI
    /// window overlap. <0 = auto (ROI half-size). 0 = no spacing constraint (legacy).
    /// NOTE: large spacing can exclude clustered discriminative points and hurt
    /// conditioning on some shapes — tune per part.
    float roi_min_spacing = 0.0f;

    /// Select EDGE points only for ROI refine (skip corners), D-optimal. Keeps the
    /// original 2D matchTemplate refine unchanged — only the point set differs.
    /// Avoids corner-clustering degeneracy and fixes worst-case on such shapes.
    /// Pairs well with roi_min_spacing (~12px). Default off (legacy corner+edge mix).
    bool roi_edge_only_points = false;

    /// For edge (non-corner) ROI points, match a 1D intensity profile along the edge
    /// normal instead of a full 2D template match. Faster and avoids the ill-defined
    /// along-edge peak (an edge only constrains across its normal). Corners still use
    /// 2D. Pairs well with edge-heavy point selection.
    bool roi_edge_1d_match = false;

    /// For edge ROI points: run the fast 2D matchTemplate, then collapse the response
    /// along the edge tangent into a 1D profile and take the across-normal subpixel
    /// peak. ~2D speed, but tangent-averaged (denoised) and 1D-constrained.
    bool roi_edge_collapse = false;

    /// Max ROI solve iterations (Gauss-Newton on point-to-line residuals; the dst
    /// correspondences are matched once and re-solved). 0 = library default (3).
    int roi_max_iters = 0;

    /// Re-match ROI correspondences every iteration at the improved pose (ICP-style)
    /// instead of fixing them after the first match. Corrects translation / along-edge
    /// / large initial-error cases a fixed-correspondence solve cannot (esp. helps 1D
    /// edge matching). Costs extra matches per refine.
    bool roi_iterative_rematch = false;

    /// Half-range (full-resolution px) of the 1-D search each ROI point runs along
    /// its edge normal. 0 = library default (15). Together with the point's lever
    /// arm this IS the coarse pose error the refine can absorb: a coarser
    /// angle_step needs a wider search -- and a wider search can lock onto a
    /// neighbouring edge, so this is a per-recipe choice to be verified, not a
    /// default to raise.
    int roi_search_half = 0;

    /// Coarse-to-fine pre-pass: 0 = off; in (0,1) refine first on the scene and
    /// template scaled by this factor (search range unchanged, so the capture
    /// grows by 1/f in full-res px), then the normal full-resolution pass from
    /// there. Widens capture like roi_search_half but keeps the full-res search
    /// narrow. Same caveat: a symmetric part can have its mirror pose confirmed
    /// by the coarse pass. Verify per recipe.
    float roi_prescale = 0.0f;

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
    int group = -1;            ///< Location group. Results sharing a group are ONE object seen at
                               ///< different angles (alternates kept past spatial NMS, best score
                               ///< first). A caller that measures should take the first member that
                               ///< passes its orientation test and ignore the rest; a caller that
                               ///< counts should count groups. Results with different groups are
                               ///< different objects (or poses the caller's nms_angle asked to keep).
    float refine_residual = -1.0f; ///< ROI refine fit quality: mean |point-to-line|
                                   ///< residual (px) of the matched sample points at
                                   ///< the final pose. Low (~<1px) = trustworthy; high
                                   ///< = points disagree (occlusion / wrong / off match).
                                   ///< -1 = not computed (refine != ROI). Use to detect
                                   ///< completely-off matches at the result level.
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

    /// Register a model directly from a template image (grayscale CV_8U).
    /// Behaves like the FeatureSet overload, but when the matcher's
    /// match_scale < 1 the down-scaled matcher's features are RE-EXTRACTED from
    /// a template resized by match_scale, instead of merely coordinate-scaling
    /// the full-res features. This keeps match score under downscale: features
    /// are selected and oriented at the resolution the (resized) scene is
    /// matched at. Full-res features are still extracted for the full-res
    /// detector and ROI/ICP refine.
    /// @param templ_gray  Template image (CV_8U grayscale).
    /// @param mask        Optional mask (CV_8U); empty = whole image.
    /// @param config      Rotation/scale/flip configuration.
    /// @param num_features Max features to extract per scale.
    /// @param scaled_blur_ksize  Optional odd Gaussian kernel (>=3) applied to
    ///        the template before the match_scale re-extraction only. Stabilises
    ///        orientation quantization for THIN/sparse shapes under aggressive
    ///        downscale (e.g. a wireframe star: +~3 score at s=0.5). Slightly
    ///        hurts solid/chunky shapes, so it is off (0) by default and the
    ///        full-res features (for refine) are never blurred.
    /// @return Number of variants generated, or -1 on failure.
    int addModel(const std::string& name,
                 const cv::Mat& templ_gray,
                 const cv::Mat& mask = cv::Mat(),
                 const ModelConfig& config = ModelConfig(),
                 int num_features = 128,
                 int scaled_blur_ksize = 0);

    /// Match all registered models against a scene image.
    /// @param scene  Grayscale scene image (CV_8U).
    /// @return Matches sorted by score descending, NMS applied.
    std::vector<MatchResult> match(const cv::Mat& scene) const;

    /// Get number of registered models.
    int numModels() const;

    /// Get total number of template variants across all models.
    int numTemplates() const;

private:
    /// Shared registration path. `rescaled` (when non-null) supplies features
    /// re-extracted at match_scale for the down-scaled matcher; otherwise the
    /// full-res feature coordinates are scaled.
    int addModelInternal(const std::string& name,
                         const FeatureSet& features,
                         const ModelConfig& config,
                         const FeatureSet* rescaled);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sbm
