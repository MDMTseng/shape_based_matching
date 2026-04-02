/// @file shape_matcher.cpp
/// @brief High-level shape-based matching API implementation.

#include "shape_matcher.h"
#include "line2Dup.h"
#include "icp_refine.h"

#include <opencv2/imgproc.hpp>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace sbm {

// ============================================================
// FeatureSet serialization
// ============================================================

bool FeatureSet::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    // Header
    const char magic[] = "SBM1";
    f.write(magic, 4);
    int32_t nl = (int32_t)levels.size();
    f.write((char*)&nl, 4);
    f.write((char*)&templ_width, 4);
    f.write((char*)&templ_height, 4);
    f.write((char*)&origin.x, 4);
    f.write((char*)&origin.y, 4);
    f.write((char*)&angle_offset, 4);

    // ICP edges
    int32_t ne = (int32_t)icp_edges.size();
    f.write((char*)&ne, 4);
    for (auto& e : icp_edges) {
        f.write((char*)&e.px, 4);
        f.write((char*)&e.py, 4);
        f.write((char*)&e.nx, 4);
        f.write((char*)&e.ny, 4);
    }

    // Per level
    for (auto& lv : levels) {
        int32_t nf = (int32_t)lv.features.size();
        f.write((char*)&nf, 4);
        f.write((char*)&lv.tl_x, 4);
        f.write((char*)&lv.tl_y, 4);
        f.write((char*)&lv.width, 4);
        f.write((char*)&lv.height, 4);
        f.write((char*)&lv.level, 4);
        for (auto& ft : lv.features) {
            f.write((char*)&ft.x, 4);
            f.write((char*)&ft.y, 4);
            f.write((char*)&ft.label, 4);
            f.write((char*)&ft.theta, 4);
        }
    }
    return f.good();
}

FeatureSet FeatureSet::load(const std::string& path) {
    FeatureSet fs;
    std::ifstream f(path, std::ios::binary);
    if (!f) return fs;

    char magic[4];
    f.read(magic, 4);
    if (magic[0] != 'S' || magic[1] != 'B' || magic[2] != 'M' || magic[3] != '1')
        return fs;

    int32_t nl;
    f.read((char*)&nl, 4);
    f.read((char*)&fs.templ_width, 4);
    f.read((char*)&fs.templ_height, 4);
    f.read((char*)&fs.origin.x, 4);
    f.read((char*)&fs.origin.y, 4);
    f.read((char*)&fs.angle_offset, 4);

    // ICP edges
    int32_t ne;
    f.read((char*)&ne, 4);
    fs.icp_edges.resize(ne);
    for (auto& e : fs.icp_edges) {
        f.read((char*)&e.px, 4);
        f.read((char*)&e.py, 4);
        f.read((char*)&e.nx, 4);
        f.read((char*)&e.ny, 4);
    }

    fs.levels.resize(nl);
    for (auto& lv : fs.levels) {
        int32_t nf;
        f.read((char*)&nf, 4);
        f.read((char*)&lv.tl_x, 4);
        f.read((char*)&lv.tl_y, 4);
        f.read((char*)&lv.width, 4);
        f.read((char*)&lv.height, 4);
        f.read((char*)&lv.level, 4);
        lv.features.resize(nf);
        for (auto& ft : lv.features) {
            f.read((char*)&ft.x, 4);
            f.read((char*)&ft.y, 4);
            f.read((char*)&ft.label, 4);
            f.read((char*)&ft.theta, 4);
        }
    }
    return fs;
}

// ============================================================
// Feature extraction
// ============================================================

FeatureSet extractFeatures(const cv::Mat& templ_gray,
                           const cv::Mat& mask,
                           int num_features,
                           const std::vector<int>& pyramid_T) {
    FeatureSet fs;
    fs.templ_width = templ_gray.cols;
    fs.templ_height = templ_gray.rows;
    fs.origin = cv::Point2f(templ_gray.cols / 2.0f, templ_gray.rows / 2.0f);

    cv::Mat use_mask = mask;
    if (use_mask.empty())
        use_mask = cv::Mat(templ_gray.size(), CV_8U, cv::Scalar(255));

    // Use meiqua detector to extract features
    line2Dup::Detector det(num_features, pyramid_T, 30, 60);
    int id = det.addTemplate(templ_gray, "_extract_", use_mask);
    if (id < 0) return fs;

    auto& tp = det.getTemplates("_extract_", id);
    fs.levels.resize(tp.size());
    for (size_t i = 0; i < tp.size(); ++i) {
        auto& src = tp[i];
        auto& dst = fs.levels[i];
        dst.tl_x = src.tl_x;
        dst.tl_y = src.tl_y;
        dst.width = src.width;
        dst.height = src.height;
        dst.level = src.pyramid_level;
        dst.features.resize(src.features.size());
        for (size_t j = 0; j < src.features.size(); ++j) {
            dst.features[j].x = src.features[j].x;
            dst.features[j].y = src.features[j].y;
            dst.features[j].label = src.features[j].label;
            dst.features[j].theta = src.features[j].theta;
        }
    }

    // Extract dense Canny edges with accurate normals for ICP
    {
        cv::Mat smooth, dx, dy, edges;
        cv::GaussianBlur(templ_gray, smooth, cv::Size(5, 5), 0);
        cv::Sobel(smooth, dx, CV_16S, 1, 0, 3);
        cv::Sobel(smooth, dy, CV_16S, 0, 1, 3);
        cv::Canny(dx, dy, edges, 30, 60);

        float cx = templ_gray.cols / 2.0f, cy = templ_gray.rows / 2.0f;
        for (int r = 0; r < templ_gray.rows; ++r) {
            const short* dxr = dx.ptr<short>(r);
            const short* dyr = dy.ptr<short>(r);
            for (int c = 0; c < templ_gray.cols; ++c) {
                if (edges.at<uchar>(r, c) == 0) continue;
                float gx = (float)dxr[c], gy = (float)dyr[c];
                float mag = std::sqrt(gx*gx + gy*gy);
                if (mag < 1e-6f) continue;
                FeatureSet::EdgePoint ep;
                ep.px = c - cx;
                ep.py = r - cy;
                ep.nx = gx / mag;
                ep.ny = gy / mag;
                fs.icp_edges.push_back(ep);
            }
        }
    }

    return fs;
}

// ============================================================
// ShapeMatcher::Impl
// ============================================================

struct ModelInfo {
    std::string name;
    FeatureSet features;
    ModelConfig config;
    std::string class_id;           // meiqua class_id (internal)
    std::string class_id_flip;      // flipped variant class_id
    int num_variants = 0;           // total templates for this model
    float angle_bias = 0;           // calibrated bias
};

struct ShapeMatcher::Impl {
    MatchConfig match_config;
    line2Dup::Detector detector;
    std::vector<ModelInfo> models;
    int total_templates = 0;

    Impl(const MatchConfig& cfg)
        : match_config(cfg),
          detector(128, {4, 8}, 30, 60) {}

    // Convert FeatureSet to meiqua TemplatePyramid
    static void featureSetToTemplates(const FeatureSet& fs,
                                       std::vector<line2Dup::Template>& tp) {
        tp.resize(fs.levels.size());
        for (size_t i = 0; i < fs.levels.size(); ++i) {
            auto& src = fs.levels[i];
            tp[i].tl_x = src.tl_x;
            tp[i].tl_y = src.tl_y;
            tp[i].width = src.width;
            tp[i].height = src.height;
            tp[i].pyramid_level = src.level;
            tp[i].angle = 0;
            tp[i].features.resize(src.features.size());
            for (size_t j = 0; j < src.features.size(); ++j) {
                tp[i].features[j].x = src.features[j].x;
                tp[i].features[j].y = src.features[j].y;
                tp[i].features[j].label = src.features[j].label;
                tp[i].features[j].theta = src.features[j].theta;
            }
        }
    }

    // Flip features horizontally around template center
    static FeatureSet flipFeatures(const FeatureSet& fs) {
        FeatureSet flipped = fs;
        for (auto& lv : flipped.levels) {
            int center_y = (lv.level == 0) ? fs.templ_height / 2
                           : fs.templ_height / (2 << lv.level);
            for (auto& f : lv.features) {
                int true_y = lv.tl_y + f.y;
                f.y = (2 * center_y - true_y) - lv.tl_y;
                f.theta = -f.theta;
                if (f.theta < 0) f.theta += 360;
                f.label = (int)(f.theta * 16.0f / 360.0f + 0.5f) & 7;
            }
        }
        flipped.origin.y = fs.templ_height - fs.origin.y;
        return flipped;
    }

    // Add templates for one model at one scale
    int addModelAtScale(const std::string& class_id,
                        const FeatureSet& fs,
                        const AngleRange& angle,
                        float scale) {
        // Convert to meiqua templates
        std::vector<line2Dup::Template> base_tp;
        featureSetToTemplates(fs, base_tp);

        // Scale features if needed
        if (std::abs(scale - 1.0f) > 0.001f) {
            for (auto& t : base_tp) {
                t.tl_x = (int)(t.tl_x * scale + 0.5f);
                t.tl_y = (int)(t.tl_y * scale + 0.5f);
                t.width = (int)(t.width * scale + 0.5f);
                t.height = (int)(t.height * scale + 0.5f);
                for (auto& f : t.features) {
                    f.x = (int)(f.x * scale + 0.5f);
                    f.y = (int)(f.y * scale + 0.5f);
                }
            }
        }

        // Add base (0-degree) template
        auto& tps = detector.getClassTemplates(class_id);
        tps.push_back(base_tp);
        int count = 1;

        // Rotate features for remaining angles
        cv::Point2f center(fs.templ_width * scale / 2.0f,
                           fs.templ_height * scale / 2.0f);
        int pyramid_levels = (int)base_tp.size();

        for (float a = angle.start + angle.step; a < angle.end; a += angle.step) {
            // Rotate feature coordinates + orientation
            std::vector<line2Dup::Template> rot_tp(pyramid_levels);
            float angRad = -a * (float)CV_PI / 180.0f;

            for (int l = 0; l < pyramid_levels; ++l) {
                cv::Point2f lvl_center = center;
                for (int i = 0; i < l; ++i) lvl_center *= 0.5f;

                for (auto& f : base_tp[l].features) {
                    cv::Point2f p((float)(f.x + base_tp[l].tl_x),
                                  (float)(f.y + base_tp[l].tl_y));
                    cv::Point2f pr = p - lvl_center;
                    cv::Point2f rot(
                        std::cos(angRad)*pr.x - std::sin(angRad)*pr.y + lvl_center.x,
                        std::sin(angRad)*pr.x + std::cos(angRad)*pr.y + lvl_center.y);

                    line2Dup::Feature fn;
                    fn.x = (int)(rot.x + 0.5f);
                    fn.y = (int)(rot.y + 0.5f);
                    fn.theta = f.theta - a;
                    while (fn.theta >= 360) fn.theta -= 360;
                    while (fn.theta < 0) fn.theta += 360;
                    fn.label = (int)(fn.theta * 16.0f / 360.0f + 0.5f) & 7;
                    rot_tp[l].features.push_back(fn);
                }
                rot_tp[l].pyramid_level = l;
                rot_tp[l].angle = a;
            }

            // Crop bounding box (inline version of cropTemplates)
            for (auto& t : rot_tp) {
                int min_x = INT_MAX, min_y = INT_MAX, max_x = 0, max_y = 0;
                for (auto& f : t.features) {
                    min_x = std::min(min_x, f.x); max_x = std::max(max_x, f.x);
                    min_y = std::min(min_y, f.y); max_y = std::max(max_y, f.y);
                }
                if (t.features.empty()) continue;
                t.tl_x = min_x; t.tl_y = min_y;
                t.width = max_x - min_x; t.height = max_y - min_y;
                for (auto& f : t.features) { f.x -= t.tl_x; f.y -= t.tl_y; }
            }
            tps.push_back(rot_tp);
            count++;
        }
        return count;
    }
};

// ============================================================
// ShapeMatcher public methods
// ============================================================

ShapeMatcher::ShapeMatcher(const MatchConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

ShapeMatcher::~ShapeMatcher() = default;

int ShapeMatcher::addModel(const std::string& name,
                           const FeatureSet& features,
                           const ModelConfig& config) {
    ModelInfo info;
    info.name = name;
    info.features = features;
    info.config = config;
    info.class_id = "sbm_" + name;
    info.class_id_flip = "sbm_" + name + "_flip";

    int count = 0;

    // For each scale
    float s = config.scale.min;
    do {
        count += impl_->addModelAtScale(info.class_id, features, config.angle, s);

        if (config.flip) {
            auto flipped = Impl::flipFeatures(features);
            count += impl_->addModelAtScale(info.class_id_flip, flipped, config.angle, s);
        }

        s += config.scale.step;
    } while (s <= config.scale.max + 0.001f && config.scale.max > config.scale.min);

    info.num_variants = count;
    impl_->models.push_back(info);
    impl_->total_templates += count;
    return count;
}

std::vector<MatchResult> ShapeMatcher::match(const cv::Mat& scene) const {
    auto& cfg = impl_->match_config;

    // Optional scene downscale for faster matching
    cv::Mat match_scene = scene;
    float inv_scale = 1.0f;
    if (cfg.match_scale < 1.0f && cfg.match_scale > 0.1f) {
        inv_scale = 1.0f / cfg.match_scale;
        cv::resize(scene, match_scene,
                   cv::Size((int)(scene.cols * cfg.match_scale),
                            (int)(scene.rows * cfg.match_scale)));
    }

    // Pad to 16
    int pw = (match_scene.cols + 15) & ~15;
    int ph = (match_scene.rows + 15) & ~15;
    cv::Mat padded;
    if (pw != match_scene.cols || ph != match_scene.rows)
        cv::copyMakeBorder(match_scene, padded, 0, ph - match_scene.rows,
                           0, pw - match_scene.cols, cv::BORDER_CONSTANT, cv::Scalar(0));
    else
        padded = match_scene;

    // Collect all class_ids to match
    std::vector<std::string> class_ids;
    for (auto& m : impl_->models) {
        class_ids.push_back(m.class_id);
        if (m.config.flip)
            class_ids.push_back(m.class_id_flip);
    }

    // Run meiqua matching
    auto raw_matches = impl_->detector.match(padded, cfg.min_score, class_ids);

    // NMS
    float nms_r = cfg.nms_radius;
    std::vector<line2Dup::Match> nms_matches;
    for (auto& m : raw_matches) {
        bool suppressed = false;
        for (auto& k : nms_matches) {
            float dx = (float)((int)(m.x * inv_scale) - (int)(k.x * inv_scale));
            float dy = (float)((int)(m.y * inv_scale) - (int)(k.y * inv_scale));
            if (dx*dx + dy*dy < nms_r * nms_r) { suppressed = true; break; }
        }
        if (!suppressed) {
            nms_matches.push_back(m);
            if (cfg.max_results > 0 && (int)nms_matches.size() >= cfg.max_results)
                break;
        }
    }

    // Convert to MatchResult with user-defined transforms
    std::vector<MatchResult> results;

    // Prepare ICP if needed
    std::vector<icp_refine::EdgePoint> model_edges_cache;

    for (auto& m : nms_matches) {
        // Find which model this belongs to
        ModelInfo* mi = nullptr;
        bool is_flip = false;
        for (auto& model : impl_->models) {
            if (m.class_id == model.class_id) { mi = &model; break; }
            if (m.class_id == model.class_id_flip) { mi = &model; is_flip = true; break; }
        }
        if (!mi) continue;

        auto& fs = mi->features;
        auto& tmpl = impl_->detector.getTemplates(m.class_id, m.template_id);
        float angle_step = mi->config.angle.step;

        // Compute raw angle from template_id
        int templates_per_scale = (int)((mi->config.angle.end - mi->config.angle.start) / angle_step);
        float raw_angle = mi->config.angle.start + (m.template_id % templates_per_scale) * angle_step;

        // Compute scale from template_id
        int scale_idx = m.template_id / templates_per_scale;
        float matched_scale = mi->config.scale.min + scale_idx * mi->config.scale.step;
        if (matched_scale < mi->config.scale.min) matched_scale = mi->config.scale.min;

        // Compute object center (user origin, rotated + scaled)
        float scaled_tw = fs.templ_width * matched_scale;
        float scaled_th = fs.templ_height * matched_scale;
        float scene_x = m.x * inv_scale + (scaled_tw / 2.0f - tmpl[0].tl_x * inv_scale);
        float scene_y = m.y * inv_scale + (scaled_th / 2.0f - tmpl[0].tl_y * inv_scale);

        // Transform user origin from template center to scene coords
        float ox = fs.origin.x - fs.templ_width / 2.0f;
        float oy = fs.origin.y - fs.templ_height / 2.0f;
        if (is_flip) oy = -oy;
        float rad = -raw_angle * (float)CV_PI / 180.0f;
        float rot_ox = (std::cos(rad) * ox - std::sin(rad) * oy) * matched_scale;
        float rot_oy = (std::sin(rad) * ox + std::cos(rad) * oy) * matched_scale;
        float user_x = scene_x + rot_ox;
        float user_y = scene_y + rot_oy;

        float user_angle = raw_angle + fs.angle_offset;
        if (is_flip) user_angle = -user_angle + 2 * fs.angle_offset;
        while (user_angle < 0) user_angle += 360;
        while (user_angle >= 360) user_angle -= 360;

        // ICP refinement at full resolution using dense Canny edges
        if (cfg.refine == RefineMode::ICP && !scene.empty() && !fs.icp_edges.empty()) {
            // Convert stored dense edges to icp_refine format
            std::vector<icp_refine::EdgePoint> edges(fs.icp_edges.size());
            for (size_t ei = 0; ei < fs.icp_edges.size(); ++ei) {
                edges[ei].pos = cv::Point2f(fs.icp_edges[ei].px, fs.icp_edges[ei].py);
                edges[ei].normal = cv::Point2f(fs.icp_edges[ei].nx, fs.icp_edges[ei].ny);
            }

            cv::Mat roi_smooth, roi_dx, roi_dy;
            int margin = (int)(fs.templ_width * matched_scale / 2) + 30;
            int rx = std::max(0, (int)(scene_x - margin));
            int ry = std::max(0, (int)(scene_y - margin));
            int rw = std::min(scene.cols - rx, 2 * margin);
            int rh = std::min(scene.rows - ry, 2 * margin);
            if (rw > 10 && rh > 10) {
                cv::Mat roi = scene(cv::Rect(rx, ry, rw, rh));
                cv::GaussianBlur(roi, roi_smooth, cv::Size(7, 7), 0);
                cv::Sobel(roi_smooth, roi_dx, CV_16S, 1, 0, 3);
                cv::Sobel(roi_smooth, roi_dy, CV_16S, 0, 1, 3);

                icp_refine::ICPConfig icp_cfg;
                icp_cfg.max_iterations = cfg.icp_iterations;
                icp_cfg.max_dist = cfg.icp_max_dist;

                icp_refine::Pose2D init(scene_x - rx, scene_y - ry, raw_angle);
                auto refined = icp_refine::refineWithNormals(
                    edges, roi_dx, roi_dy, init,
                    (int)(fs.templ_width * matched_scale), 20, icp_cfg);

                // Update with refined pose
                scene_x = refined.x + rx;
                scene_y = refined.y + ry;
                raw_angle = refined.angle;

                // Recompute user coordinates from refined pose
                rad = -raw_angle * (float)CV_PI / 180.0f;
                rot_ox = (std::cos(rad) * ox - std::sin(rad) * oy) * matched_scale;
                rot_oy = (std::sin(rad) * ox + std::cos(rad) * oy) * matched_scale;
                user_x = scene_x + rot_ox;
                user_y = scene_y + rot_oy;
                user_angle = raw_angle + fs.angle_offset;
                if (is_flip) user_angle = -user_angle + 2 * fs.angle_offset;
                while (user_angle < 0) user_angle += 360;
                while (user_angle >= 360) user_angle -= 360;
            }
        }

        MatchResult r;
        r.model_name = mi->name;
        r.x = user_x;
        r.y = user_y;
        r.angle = user_angle;
        r.scale = matched_scale;
        r.flipped = is_flip;
        r.score = m.similarity;
        results.push_back(r);
    }

    return results;
}

int ShapeMatcher::numModels() const { return (int)impl_->models.size(); }
int ShapeMatcher::numTemplates() const { return impl_->total_templates; }

} // namespace sbm
