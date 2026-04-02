/// @file shape_matcher.cpp
/// @brief High-level shape-based matching API implementation.

#include "shape_matcher.h"
#include "line2Dup.h"
#include "icp_refine.h"
#include "roi_refine.h"

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

    // Template image (for ROI refinement)
    int32_t img_rows = templ_image.rows, img_cols = templ_image.cols;
    f.write((char*)&img_rows, 4);
    f.write((char*)&img_cols, 4);
    if (img_rows > 0 && img_cols > 0) {
        for (int r = 0; r < img_rows; ++r)
            f.write((char*)templ_image.ptr(r), img_cols);
    }

    // Refine points (edges + corners)
    int32_t ne = (int32_t)refine_points.size();
    f.write((char*)&ne, 4);
    for (auto& rp : refine_points) {
        f.write((char*)&rp.px, 4);
        f.write((char*)&rp.py, 4);
        f.write((char*)&rp.nx, 4);
        f.write((char*)&rp.ny, 4);
        f.write((char*)&rp.cornerness, 4);
        uint8_t t = rp.type;
        f.write((char*)&t, 1);
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
            f.write((char*)&ft.cornerness, 4);
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

    // Template image
    int32_t img_rows, img_cols;
    f.read((char*)&img_rows, 4);
    f.read((char*)&img_cols, 4);
    if (img_rows > 0 && img_cols > 0) {
        fs.templ_image = cv::Mat(img_rows, img_cols, CV_8U);
        for (int r = 0; r < img_rows; ++r)
            f.read((char*)fs.templ_image.ptr(r), img_cols);
    }

    // Refine points (edges + corners)
    int32_t ne;
    f.read((char*)&ne, 4);
    fs.refine_points.resize(ne);
    for (auto& rp : fs.refine_points) {
        f.read((char*)&rp.px, 4);
        f.read((char*)&rp.py, 4);
        f.read((char*)&rp.nx, 4);
        f.read((char*)&rp.ny, 4);
        f.read((char*)&rp.cornerness, 4);
        uint8_t t; f.read((char*)&t, 1);
        rp.type = (FeatureSet::RefinePt::Type)t;
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
            f.read((char*)&ft.cornerness, 4);
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
    fs.templ_image = templ_gray.clone();

    cv::Mat use_mask = mask;
    if (use_mask.empty())
        use_mask = cv::Mat(templ_gray.size(), CV_8U, cv::Scalar(255));

    // Use meiqua detector to extract features
    line2Dup::Detector det(num_features, pyramid_T, 30, 60);
    int id = det.addTemplate(templ_gray, "_extract_", use_mask);
    if (id < 0) return fs;

    // Compute structure tensor for cornerness classification
    cv::Mat smooth_st, dx_st, dy_st;
    cv::GaussianBlur(templ_gray, smooth_st, cv::Size(5, 5), 0);
    cv::Sobel(smooth_st, dx_st, CV_32F, 1, 0, 3);
    cv::Sobel(smooth_st, dy_st, CV_32F, 0, 1, 3);

    // Structure tensor components (smoothed with a window)
    cv::Mat Ixx = dx_st.mul(dx_st);
    cv::Mat Iyy = dy_st.mul(dy_st);
    cv::Mat Ixy = dx_st.mul(dy_st);
    cv::GaussianBlur(Ixx, Ixx, cv::Size(3, 3), 0);
    cv::GaussianBlur(Iyy, Iyy, cv::Size(3, 3), 0);
    cv::GaussianBlur(Ixy, Ixy, cv::Size(3, 3), 0);

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

            // Compute cornerness from structure tensor eigenvalues.
            int fx = src.features[j].x + src.tl_x;
            int fy = src.features[j].y + src.tl_y;
            if (fx >= 0 && fx < templ_gray.cols && fy >= 0 && fy < templ_gray.rows) {
                float a = Ixx.at<float>(fy, fx);
                float b = Ixy.at<float>(fy, fx);
                float c = Iyy.at<float>(fy, fx);
                // Eigenvalues of [[a,b],[b,c]]:
                // λ = (a+c)/2 ± sqrt(((a-c)/2)² + b²)
                float trace = a + c;
                float disc = std::sqrt(std::max(0.0f, (a-c)*(a-c)/4.0f + b*b));
                float lam_max = trace/2.0f + disc;
                float lam_min = trace/2.0f - disc;
                // cornerness = λ_min / λ_max  (0=edge, 1=corner)
                dst.features[j].cornerness = (lam_max > 1e-6f) ?
                    std::min(1.0f, std::max(0.0f, lam_min / lam_max)) : 0;
            } else {
                dst.features[j].cornerness = 0;
            }
        }
    }

    // Extract refinement points: dense Canny edges + Harris corners
    {
        cv::Mat smooth, dx16, dy16, canny_edges;
        cv::GaussianBlur(templ_gray, smooth, cv::Size(5, 5), 0);
        cv::Sobel(smooth, dx16, CV_16S, 1, 0, 3);
        cv::Sobel(smooth, dy16, CV_16S, 0, 1, 3);
        cv::Canny(dx16, dy16, canny_edges, 30, 60);

        float cx = templ_gray.cols / 2.0f, cy = templ_gray.rows / 2.0f;

        // Dense Canny edge points
        for (int r = 0; r < templ_gray.rows; ++r) {
            const short* dxr = dx16.ptr<short>(r);
            const short* dyr = dy16.ptr<short>(r);
            for (int c = 0; c < templ_gray.cols; ++c) {
                if (canny_edges.at<uchar>(r, c) == 0) continue;
                float gx = (float)dxr[c], gy = (float)dyr[c];
                float mag = std::sqrt(gx*gx + gy*gy);
                if (mag < 1e-6f) continue;
                FeatureSet::RefinePt rp;
                rp.px = c - cx;
                rp.py = r - cy;
                rp.nx = gx / mag;
                rp.ny = gy / mag;
                rp.cornerness = 0;
                rp.type = FeatureSet::RefinePt::EDGE;
                fs.refine_points.push_back(rp);
            }
        }

        // Harris corners
        cv::Mat harris_resp;
        cv::cornerHarris(smooth, harris_resp, 3, 3, 0.04);

        // Threshold + NMS to get corner positions
        double max_resp;
        cv::minMaxLoc(harris_resp, nullptr, &max_resp);
        float corner_thresh = (float)(max_resp * 0.1);  // top 10% response

        // Simple 5x5 NMS on Harris response
        for (int r = 3; r < templ_gray.rows - 3; ++r) {
            for (int c = 3; c < templ_gray.cols - 3; ++c) {
                float val = harris_resp.at<float>(r, c);
                if (val < corner_thresh) continue;

                // Check if local maximum in 5x5
                bool is_max = true;
                for (int dr = -2; dr <= 2 && is_max; ++dr)
                    for (int dc = -2; dc <= 2 && is_max; ++dc)
                        if ((dr||dc) && harris_resp.at<float>(r+dr, c+dc) >= val)
                            is_max = false;
                if (!is_max) continue;

                // Get gradient at corner for normal direction
                float gx = (float)dx16.at<short>(r, c);
                float gy = (float)dy16.at<short>(r, c);
                float mag = std::sqrt(gx*gx + gy*gy);

                FeatureSet::RefinePt rp;
                rp.px = c - cx;
                rp.py = r - cy;
                rp.nx = (mag > 1e-6f) ? gx/mag : 0;
                rp.ny = (mag > 1e-6f) ? gy/mag : 0;
                rp.cornerness = std::min(1.0f, val / (float)max_resp);
                rp.type = FeatureSet::RefinePt::CORNER;
                fs.refine_points.push_back(rp);
            }
        }
    }

    return fs;
}

// ============================================================
// FeatureSet quality evaluation
// ============================================================

FeatureSet::QualityReport FeatureSet::evaluateQuality() const {
    QualityReport r;
    r.balance = 0;
    r.strength = 0;
    r.score = 0;
    r.best_cross = 0;
    r.best_cross_sin = 0;
    r.num_edge = 0;
    r.num_corner = 0;

    if (refine_points.empty() || templ_image.empty()) {
        r.diagnosis = "No refine points or template image";
        return r;
    }

    // Select ~15 well-spaced points (same as what ROI refine would use)
    std::vector<cv::Point2f> positions;
    std::vector<float> cornerness;
    for (auto& rp : refine_points) {
        positions.push_back(cv::Point2f(rp.px, rp.py));
        cornerness.push_back(rp.cornerness);
    }

    float tcx = templ_width / 2.0f, tcy = templ_height / 2.0f;
    float min_dist = std::max(templ_width, templ_height) / 16.0f * 1.5f;
    float min_dist_sq = min_dist * min_dist;

    // Greedy select well-spaced points (corners first)
    struct Cand { int idx; float corn; };
    std::vector<Cand> cands(positions.size());
    for (size_t i = 0; i < positions.size(); i++)
        cands[i] = {(int)i, cornerness[i]};
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b){ return a.corn > b.corn; });

    std::vector<int> selected;
    for (auto& c : cands) {
        if ((int)selected.size() >= 15) break;
        auto& p = positions[c.idx];
        bool too_close = false;
        for (int si : selected) {
            float dx = p.x - positions[si].x, dy = p.y - positions[si].y;
            if (dx*dx + dy*dy < min_dist_sq) { too_close = true; break; }
        }
        if (too_close) continue;
        if (std::abs(p.x) > templ_width/2.0f - 5 || std::abs(p.y) > templ_height/2.0f - 5)
            continue;
        selected.push_back(c.idx);
    }

    if (selected.size() < 3) {
        r.diagnosis = "Too few usable points (" + std::to_string(selected.size()) + ")";
        return r;
    }

    int roi_half = 15;

    // Collect gradient vectors at each selected point for cross product analysis
    struct GradInfo { cv::Point2f dir; float mag; };
    std::vector<GradInfo> grads;
    for (int si : selected) {
        auto& rp = refine_points[si];
        int tx=(int)(rp.px+tcx+0.5f), ty=(int)(rp.py+tcy+0.5f);
        int h=roi_half;
        if(tx-h<0||tx+h>=templ_image.cols||ty-h<0||ty+h>=templ_image.rows)
            h=std::min({tx,ty,templ_image.cols-1-tx,templ_image.rows-1-ty});
        if(h<5) continue;
        cv::Mat roi2=templ_image(cv::Rect(tx-h,ty-h,2*h,2*h));
        cv::Mat dx2,dy2,mag2;
        cv::Sobel(roi2,dx2,CV_32F,1,0,3);
        cv::Sobel(roi2,dy2,CV_32F,0,1,3);
        cv::magnitude(dx2,dy2,mag2);
        double max_mag; cv::Point max_loc;
        cv::minMaxLoc(mag2, nullptr, &max_mag, nullptr, &max_loc);
        float gx=dx2.at<float>(max_loc.y,max_loc.x);
        float gy=dy2.at<float>(max_loc.y,max_loc.x);
        grads.push_back({cv::Point2f(gx,gy), (float)max_mag});
    }

    // Find the pair with the BEST cross product
    // |n1 × n2| = |n1.x*n2.y - n1.y*n2.x| = |n1|*|n2|*sin(angle)
    // Captures both angular diversity AND edge strength in one number
    for (size_t i = 0; i < grads.size(); ++i) {
        for (size_t j = i+1; j < grads.size(); ++j) {
            float cross = std::abs(grads[i].dir.x * grads[j].dir.y -
                                   grads[i].dir.y * grads[j].dir.x);
            if (cross > r.best_cross) {
                r.best_cross = cross;
                float mag_prod = grads[i].mag * grads[j].mag;
                r.best_cross_sin = (mag_prod > 1e-6f) ? cross / mag_prod : 0;
            }
        }
    }

    // Find the WEAKEST cross product across orthogonal direction pairs.
    // Split gradients into two groups by direction, find worst inter-group cross.
    // Simpler: compute cross products for ALL pairs, find the median.
    // If median is high → well-distributed. If low → most pairs are parallel.
    std::vector<float> all_cross_sins;
    for (size_t i = 0; i < grads.size(); ++i) {
        for (size_t j = i+1; j < grads.size(); ++j) {
            float cross = std::abs(grads[i].dir.x * grads[j].dir.y -
                                   grads[i].dir.y * grads[j].dir.x);
            float mag_prod = grads[i].mag * grads[j].mag;
            float sin_val = (mag_prod > 1e-6f) ? cross / mag_prod : 0;
            all_cross_sins.push_back(sin_val);
        }
    }
    if (!all_cross_sins.empty()) {
        std::sort(all_cross_sins.begin(), all_cross_sins.end());
    }

    // === BALANCE (0-100) ===
    // sin(angle) of the best pair tells us the max angular diversity.
    // But we also need to check that the weaker direction has enough support.
    // Use: best_sin × (fraction of pairs above sin > 0.3) to penalize when
    // only a few pairs are perpendicular (most are parallel).
    int pairs_above_30 = 0;
    for (float s : all_cross_sins)
        if (s > 0.3f) pairs_above_30++;
    float frac_diverse = all_cross_sins.empty() ? 0 :
        (float)pairs_above_30 / all_cross_sins.size();
    // balance = sin × sqrt(fraction_diverse) — penalize when few pairs are diverse
    r.balance = (int)(100.0f * r.best_cross_sin * std::sqrt(frac_diverse));
    r.balance = std::max(0, std::min(100, r.balance));

    // === STRENGTH (0-100) ===
    // Best cross product magnitude: |n1|×|n2|×sin(angle).
    r.strength = (int)(100.0f * std::min(1.0f, r.best_cross / 400000.0f));

    // === COMBINED ===
    r.score = r.balance * r.strength / 100;

    // Diagnosis
    std::string bal_str = (r.balance >= 70) ? "balanced" :
                          (r.balance >= 40) ? "angled" : "near-parallel";
    std::string str_str = (r.strength >= 70) ? "strong" :
                          (r.strength >= 40) ? "moderate" : "weak";
    r.diagnosis = bal_str + " / " + str_str;

    return r;
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
            float angRad = a * (float)CV_PI / 180.0f;

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
                    fn.theta = f.theta + a;
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

        // ICP refinement at full resolution
        bool do_icp = (cfg.refine == RefineMode::ICP || cfg.refine == RefineMode::ICP_Sparse)
                      && !scene.empty();
        if (do_icp) {
            std::vector<icp_refine::EdgePoint> edges;
            bool use_cornerness = false;

            if (cfg.refine == RefineMode::ICP && !fs.refine_points.empty()) {
                // All refine points (edges + corners) with proper normals + cornerness
                for (auto& rp : fs.refine_points) {
                    icp_refine::EdgePoint ep;
                    ep.pos = cv::Point2f(rp.px, rp.py);
                    ep.normal = cv::Point2f(rp.nx, rp.ny);
                    ep.cornerness = rp.cornerness;
                    edges.push_back(ep);
                }
                use_cornerness = true;  // corners get point-to-point
            } else {
                // Sparse matching features with cornerness-aware ICP weight.
                // Corner features (cornerness > 0.3) use higher point_to_point_weight
                // for 2D constraint. Edge features use point-to-plane only.
                auto& lvl0 = fs.levels[0];
                for (auto& f : lvl0.features) {
                    icp_refine::EdgePoint ep;
                    ep.pos = cv::Point2f((float)(f.x + lvl0.tl_x) - fs.templ_width / 2.0f,
                                         (float)(f.y + lvl0.tl_y) - fs.templ_height / 2.0f);
                    float tr = f.theta * (float)CV_PI / 180.0f;
                    ep.normal = cv::Point2f(std::cos(tr), std::sin(tr));
                    ep.cornerness = f.cornerness;
                    edges.push_back(ep);
                }
                use_cornerness = true;
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
                icp_cfg.use_cornerness = use_cornerness;

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

        // ROI-based refinement
        if (cfg.refine == RefineMode::ROI && !fs.templ_image.empty() && !scene.empty()) {
            // Select critical points from refine_points
            std::vector<cv::Point2f> positions;
            std::vector<float> corner_scores;
            for (auto& rp : fs.refine_points) {
                positions.push_back(cv::Point2f(rp.px, rp.py));
                corner_scores.push_back(rp.cornerness);
            }

            auto sample_pts = roi_refine::selectCriticalPoints(
                positions, corner_scores, 15, fs.templ_width, fs.templ_height);

            if (!sample_pts.empty()) {
                roi_refine::ROIConfig roi_cfg;
                roi_cfg.roi_half = 15;
                roi_cfg.search_half = 15;

                cv::Vec3f init_pose(scene_x, scene_y, raw_angle);
                auto refined_pose = roi_refine::refineROI(
                    fs.templ_image, scene, sample_pts, init_pose, roi_cfg);

                scene_x = refined_pose[0];
                scene_y = refined_pose[1];
                raw_angle = refined_pose[2];

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
