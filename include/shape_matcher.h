#pragma once
/// @file shape_matcher.h
/// @brief High-level shape-based matching API.
///
/// Workflow:
///   OFFLINE: extractFeatures() -> FeatureSet -> save()
///   ONLINE:  load() -> addModel() -> match()

#include <opencv2/core.hpp>
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

    /// Dense edge points for ICP refinement (from Canny, not from sparse features).
    /// Positions relative to template center, normals from actual gradient direction.
    struct EdgePoint {
        float px, py;   ///< Position relative to template center
        float nx, ny;   ///< Normal direction (unit vector)
    };
    std::vector<EdgePoint> icp_edges;

    // --- User-defined reference frame ---

    /// Origin point in template image coords.
    /// Match results report position of this point in the scene.
    /// Default: template center (templ_width/2, templ_height/2).
    cv::Point2f origin;

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
    ICP                 ///< ICP using dense Canny edges (~0.3ms/obj, <0.5deg)
};

struct MatchConfig {
    float min_score = 50.0f;       ///< Minimum similarity (0-100)
    int max_results = 0;           ///< Max results (0 = unlimited)
    float nms_radius = -1;         ///< Spatial NMS radius (-1 = auto from template size)
    float match_scale = 1.0f;      ///< Scene downscale for faster matching (e.g., 0.5)
    RefineMode refine = RefineMode::ICP;

    // ICP config (used when refine == ICP)
    int icp_iterations = 30;
    float icp_max_dist = 10.0f;
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
