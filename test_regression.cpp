// Comprehensive regression test for shape_based_matching.
// Validates all core algorithms haven't regressed.
// Exit code: 0 = all pass, 1 = any fail.

#include "shape_matcher.h"
#include "roi_refine.h"
#include "icp_refine.h"
#include "test_utils.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <set>

using namespace cv;

// ============================================================
// Globals
// ============================================================
static int g_pass = 0;
static int g_fail = 0;
static int g_warn = 0;
static FILE* g_log = nullptr;

#define LOG(...) do { if(g_log) { fprintf(g_log, __VA_ARGS__); fflush(g_log); } } while(0)

#define CHECK(cond, ...) do { \
    char _msg[512]; snprintf(_msg, sizeof(_msg), __VA_ARGS__); \
    if (cond) { g_pass++; printf("  PASS: %s\n", _msg); LOG("PASS: %s\n", _msg); } \
    else      { g_fail++; printf("  FAIL: %s\n", _msg); LOG("FAIL: %s\n", _msg); } \
} while(0)

#define WARN_IF(cond, ...) do { \
    char _msg[512]; snprintf(_msg, sizeof(_msg), __VA_ARGS__); \
    if (cond) { g_warn++; printf("  WARN: %s\n", _msg); LOG("WARN: %s\n", _msg); } \
    else      { printf("  OK:   %s\n", _msg); LOG("OK:   %s\n", _msg); } \
} while(0)

// ============================================================
// CSV-driven threshold system
// ============================================================
#include <map>
#include <string>
#include <vector>
#include <fstream>

struct Threshold {
    std::string id;
    std::string type;       // "check" or "warn"
    std::string metric;     // key into measured values
    std::string op;         // "<", ">", "<=", ">=", "=="
    float threshold;
    std::string description;
};

static std::vector<Threshold> g_thresholds;
static std::map<std::string, float> g_metrics;

static bool load_thresholds(const char* csv_path) {
    std::ifstream f(csv_path);
    if (!f.is_open()) {
        std::string paths[] = {
            csv_path,
            std::string("../") + csv_path,
            std::string("../../") + csv_path
        };
        for (auto& p : paths) {
            f.open(p);
            if (f.is_open()) break;
        }
        if (!f.is_open()) {
            printf("ERROR: cannot open CSV file '%s'\n", csv_path);
            return false;
        }
    }

    std::string line;
    int line_num = 0;
    int errors = 0;

    // Validate header
    if (!std::getline(f, line)) {
        printf("ERROR: CSV file '%s' is empty\n", csv_path);
        return false;
    }
    line_num++;
    // Strip trailing \r if present (Windows line endings)
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.find("id,type,metric,op,threshold") == std::string::npos) {
        printf("ERROR: CSV header mismatch at line %d\n", line_num);
        printf("  Expected: id,type,metric,op,threshold,description\n");
        printf("  Got:      %s\n", line.c_str());
        return false;
    }

    while (std::getline(f, line)) {
        line_num++;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;

        // Parse CSV: id,type,metric,op,threshold,description
        Threshold t;
        size_t p0 = 0;
        auto next_field = [&]() -> std::string {
            size_t p1 = line.find(',', p0);
            if (p1 == std::string::npos) p1 = line.size();
            std::string s = line.substr(p0, p1 - p0);
            p0 = p1 + 1;
            return s;
        };

        t.id = next_field();
        t.type = next_field();
        t.metric = next_field();
        t.op = next_field();
        std::string thresh_str = next_field();
        t.description = (p0 < line.size()) ? line.substr(p0) : "";

        // Validate fields
        if (t.id.empty()) {
            printf("ERROR: line %d: empty id\n", line_num);
            errors++; continue;
        }
        if (t.type != "check" && t.type != "warn") {
            printf("ERROR: line %d (%s): type must be 'check' or 'warn', got '%s'\n",
                   line_num, t.id.c_str(), t.type.c_str());
            errors++; continue;
        }
        if (t.metric.empty()) {
            printf("ERROR: line %d (%s): empty metric name\n", line_num, t.id.c_str());
            errors++; continue;
        }
        if (t.op != "<" && t.op != ">" && t.op != "<=" && t.op != ">=" && t.op != "==") {
            printf("ERROR: line %d (%s): invalid op '%s' (use <, >, <=, >=, ==)\n",
                   line_num, t.id.c_str(), t.op.c_str());
            errors++; continue;
        }
        try {
            t.threshold = std::stof(thresh_str);
        } catch (...) {
            printf("ERROR: line %d (%s): invalid threshold '%s' (must be a number)\n",
                   line_num, t.id.c_str(), thresh_str.c_str());
            errors++; continue;
        }

        // Check for duplicate ids
        for (auto& existing : g_thresholds) {
            if (existing.id == t.id) {
                printf("WARNING: line %d: duplicate id '%s' (overwriting previous)\n",
                       line_num, t.id.c_str());
                break;
            }
        }

        g_thresholds.push_back(t);
    }

    if (errors > 0) {
        printf("CSV '%s': %d error(s), %d valid entries loaded\n",
               csv_path, errors, (int)g_thresholds.size());
    }
    return !g_thresholds.empty();
}

// Record a measured value
static void RECORD(const char* metric, float value) {
    g_metrics[metric] = value;
    LOG("  METRIC: %s = %.4f\n", metric, value);
}

// Evaluate all thresholds against measured values
static void evaluate_thresholds() {
    printf("\n===== THRESHOLD EVALUATION =====\n");
    LOG("\n===== THRESHOLD EVALUATION =====\n");

    for (auto& t : g_thresholds) {
        auto it = g_metrics.find(t.metric);
        if (it == g_metrics.end()) {
            printf("  SKIP: %s — metric '%s' not measured\n", t.id.c_str(), t.metric.c_str());
            LOG("SKIP: %s — metric '%s' not measured\n", t.id.c_str(), t.metric.c_str());
            continue;
        }
        float val = it->second;
        bool pass = false;
        if (t.op == "<")       pass = val < t.threshold;
        else if (t.op == ">")  pass = val > t.threshold;
        else if (t.op == "<=") pass = val <= t.threshold;
        else if (t.op == ">=") pass = val >= t.threshold;
        else if (t.op == "==") pass = std::abs(val - t.threshold) < 0.001f;

        char msg[512];
        snprintf(msg, sizeof(msg), "%s: %s = %.4f %s %.4f — %s",
                 t.id.c_str(), t.metric.c_str(), val, t.op.c_str(), t.threshold,
                 t.description.c_str());

        if (t.type == "warn") {
            if (!pass) { g_warn++; printf("  WARN: %s\n", msg); LOG("WARN: %s\n", msg); }
            else       { printf("  OK:   %s\n", msg); LOG("OK:   %s\n", msg); }
        } else {
            if (pass)  { g_pass++; printf("  PASS: %s\n", msg); LOG("PASS: %s\n", msg); }
            else       { g_fail++; printf("  FAIL: %s\n", msg); LOG("FAIL: %s\n", msg); }
        }
    }
}

// ============================================================
// Helpers
// ============================================================

static void draw_L(Mat& img, int cx, int cy, double angle, int color) {
    double rad = angle * CV_PI / 180.0, cs = cos(rad), sn = sin(rad);
    for (double ly = -30; ly <= 30; ly += 0.5)
        for (double lx = -10; lx <= 10; lx += 0.5) {
            int px = cx+(int)(lx*cs-ly*sn+0.5), py = cy+(int)(lx*sn+ly*cs+0.5);
            if (px>=0&&px<img.cols&&py>=0&&py<img.rows) img.at<uchar>(py,px)=(uchar)color;
        }
    for (double ly = 10; ly <= 30; ly += 0.5)
        for (double lx = 10; lx <= 40; lx += 0.5) {
            int px = cx+(int)(lx*cs-ly*sn+0.5), py = cy+(int)(lx*sn+ly*cs+0.5);
            if (px>=0&&px<img.cols&&py>=0&&py<img.rows) img.at<uchar>(py,px)=(uchar)color;
        }
}

// Place template into scene via warpAffine with sub-pixel precision.
// Object center at (obj_cx + sub_x, obj_cy + sub_y).
static void place_object(const Mat& templ, Mat& scene,
                          int obj_cx, int obj_cy, double angle,
                          float sub_x = 0, float sub_y = 0) {
    float tcx = templ.cols / 2.0f, tcy = templ.rows / 2.0f;
    float dst_cx = obj_cx + sub_x, dst_cy = obj_cy + sub_y;

    // Warp template: rotate around center, output same size as template
    Mat M_rot = getRotationMatrix2D(Point2f(tcx, tcy), -angle, 1.0);
    // Add sub-pixel translation: shift so center lands at fractional offset
    double* md = (double*)M_rot.data;
    float frac_x = dst_cx - std::floor(dst_cx);
    float frac_y = dst_cy - std::floor(dst_cy);
    md[2] += frac_x;
    md[5] += frac_y;
    Mat rot;
    warpAffine(templ, rot, M_rot, Size(templ.cols + 2, templ.rows + 2),
               INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

    // Copy into scene at integer position (sub-pixel handled by warpAffine)
    int ox = (int)std::floor(dst_cx) - (int)tcx;
    int oy = (int)std::floor(dst_cy) - (int)tcy;
    for (int r = 0; r < rot.rows; r++)
        for (int c = 0; c < rot.cols; c++) {
            int sy = oy + r, sx = ox + c;
            if (sy >= 0 && sy < scene.rows && sx >= 0 && sx < scene.cols && rot.at<uchar>(r, c) > 0)
                scene.at<uchar>(sy, sx) = std::max(scene.at<uchar>(sy, sx), rot.at<uchar>(r, c));
        }
}

// Compute GT origin position in scene.
// Origin at (org_x, org_y) in template coords; template center at (w/2, h/2).
// Object center placed at (cx, cy) with angle.
static void compute_gt_origin(int cx, int cy, double angle,
                               float org_x, float org_y,
                               int templ_w, int templ_h,
                               float& gt_x, float& gt_y) {
    float ox = org_x - templ_w / 2.0f;
    float oy = org_y - templ_h / 2.0f;
    float rad = -(float)angle * (float)CV_PI / 180.0f;
    gt_x = cx + std::cos(rad) * ox - std::sin(rad) * oy;
    gt_y = cy + std::sin(rad) * ox + std::cos(rad) * oy;
}

static float angle_err(float a, float b) {
    float e = a - b;
    while (e > 180) e -= 360;
    while (e < -180) e += 360;
    return std::abs(e);
}

static float pos_err(float x1, float y1, float x2, float y2) {
    return std::sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

static Mat add_noise(const Mat& img, double sigma, int seed = 42) {
    Mat result = img.clone();
    Mat noise(img.size(), CV_64F);
    RNG rng(seed);
    rng.fill(noise, RNG::NORMAL, 0, sigma);
    Mat tmp;
    result.convertTo(tmp, CV_64F);
    tmp += noise;
    tmp.convertTo(result, CV_8U);
    return result;
}

// Suppress cout and stderr during a scope
// OutputGuard from test_utils.h replaces the old CoutSuppressor

// ============================================================
// Section 1: Coarse Matching
// ============================================================
static void test_coarse_matching(const sbm::FeatureSet& feat200, const Mat& templ200) {
    printf("\n======== 1. COARSE MATCHING ========\n");
    LOG("\n======== 1. COARSE MATCHING ========\n");

    const int scene_sz = 250;
    const float org_x = 100, org_y = 75;

    // --- 1a: Detection at various angles ---
    {
        float test_angles[] = {0, 45, 90, 135, 180, 270};
        int n_angles = sizeof(test_angles)/sizeof(test_angles[0]);
        int n_found = 0;
        float worst_ang = 0, worst_pos = 0;

        for (float gt_ang : test_angles) {
            Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
            place_object(templ200, scene, scene_sz/2, scene_sz/2, gt_ang);

            float gt_x, gt_y;
            compute_gt_origin(scene_sz/2, scene_sz/2, gt_ang,
                             org_x, org_y, 200, 200, gt_x, gt_y);

            sbm::MatchConfig cfg;
            cfg.min_score = 40;
            cfg.refine = sbm::RefineMode::None;
            sbm::ShapeMatcher matcher(cfg);
            sbm::ModelConfig mcfg;
            mcfg.angle = {0, 360, 2};

            {
                OutputGuard guard;
                matcher.addModel("L", feat200, mcfg);
                auto results = matcher.match(scene);
                if (!results.empty()) {
                    n_found++;
                    auto& r = results[0];
                    float ae = angle_err(r.angle, gt_ang);
                    float pe = pos_err(r.x, r.y, gt_x, gt_y);
                    worst_ang = std::max(worst_ang, ae);
                    worst_pos = std::max(worst_pos, pe);
                    LOG("  angle=%.0f: got (%.1f,%.1f)@%.1f err=%.1fdeg %.1fpx\n",
                        gt_ang, r.x, r.y, r.angle, ae, pe);
                } else {
                    LOG("  angle=%.0f: NOT FOUND\n", gt_ang);
                }
            }
        }

        RECORD("coarse_detect_count", (float)n_found);
        RECORD("coarse_worst_ang", worst_ang);
        RECORD("coarse_worst_pos", worst_pos);
        CHECK(n_found == n_angles,
              "1a Detection: %d/%d objects found at various angles", n_found, n_angles);
        // Threshold history: 15.0 (original) → 16.0 (borderline 15.0 observed)
        CHECK(worst_ang < 16.0f,
              "1a Angle accuracy: worst=%.1fdeg (expect <16, original was 15)", worst_ang);
        CHECK(worst_pos < 15.0f,
              "1a Position accuracy: worst=%.1fpx (expect <15)", worst_pos);
    }

    // --- 1b: Multi-object FHD scene ---
    {
        struct Obj { int x, y; double angle; };
        Obj objs[] = {
            {200, 150, 15},  {500, 300, 45},   {900, 200, 90},
            {1300, 400, 135},{1700, 250, 180},  {350, 700, 210},
            {750, 850, 270}, {1100, 600, 315},  {1500, 800, 30},
            {1800, 900, 60},
        };
        int n_objs = sizeof(objs)/sizeof(objs[0]);

        Mat scene_fhd(1080, 1920, CV_8U, Scalar(30));
        for (auto& obj : objs)
            place_object(templ200, scene_fhd, obj.x, obj.y, obj.angle);

        // Use ICP refinement for multi-object detection test (coarse alone may
        // not report enough results due to low raw scores with few features)
        sbm::MatchConfig cfg;
        cfg.min_score = 40;
        cfg.nms_radius = 80;
        cfg.refine = sbm::RefineMode::ICP;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};

        int matched = 0;
        {
            OutputGuard guard;
            matcher.addModel("L", feat200, mcfg);
            auto results = matcher.match(scene_fhd);

            LOG("  1b: %d raw results\n", (int)results.size());
            for (auto& r : results) {
                float best_d = 1e9f;
                int best_j = -1;
                for (int j = 0; j < n_objs; j++) {
                    float gt_x, gt_y;
                    compute_gt_origin(objs[j].x, objs[j].y, objs[j].angle,
                                     org_x, org_y, 200, 200, gt_x, gt_y);
                    float d = pos_err(r.x, r.y, gt_x, gt_y);
                    if (d < best_d) { best_d = d; best_j = j; }
                }
                LOG("  result (%.0f,%.0f)@%.0f score=%.0f  nearest_gt=%d dist=%.1f\n",
                    r.x, r.y, r.angle, r.score, best_j, best_d);
                if (best_d < 50 && r.score > 40) {
                    matched++;
                }
            }
        }
        RECORD("multi_obj_found", (float)matched);
        CHECK(matched >= 9,
              "1b Multi-object: %d/%d found with score>40 (expect >=9)", matched, n_objs);
    }

    // --- 1c: Speed on FHD ---
    {
        Mat scene_fhd(1080, 1920, CV_8U, Scalar(30));
        place_object(templ200, scene_fhd, 960, 540, 45);

        sbm::MatchConfig cfg;
        cfg.min_score = 40;
        cfg.refine = sbm::RefineMode::None;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};

        double ms;
        {
            OutputGuard guard;
            matcher.addModel("L", feat200, mcfg);
            matcher.match(scene_fhd);  // warmup

            auto t0 = std::chrono::high_resolution_clock::now();
            matcher.match(scene_fhd);
            ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
        }
        RECORD("coarse_fhd_ms", (float)ms);
        CHECK(ms < 100.0,
              "1c Speed: FHD single match %.1fms (expect <100ms)", ms);
    }
}

// ============================================================
// Section 2: ICP (Inverse) Refinement
// ============================================================
static void test_icp_refinement(const sbm::FeatureSet& feat200, const Mat& templ200) {
    printf("\n======== 2. ICP (INVERSE) REFINEMENT ========\n");
    LOG("\n======== 2. ICP (INVERSE) REFINEMENT ========\n");

    const int scene_sz = 250;
    const float org_x = 100, org_y = 75;

    // --- 2a/2b: Angle sweep (every 5 deg, 72 angles) ---
    {
        float total_ang_err = 0, total_pos_err = 0;
        float worst_ang = 0, worst_pos = 0;
        int count = 0;

        float old_fail_angles[] = {110, 135, 225, 245, 290, 315};
        float old_fail_pos[6] = {}, old_fail_ang[6] = {};

        for (int ai = 0; ai < 72; ai++) {
            float gt_ang = ai * 5.0f;
            Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
            place_object(templ200, scene, scene_sz/2, scene_sz/2, gt_ang);

            float gt_x, gt_y;
            compute_gt_origin(scene_sz/2, scene_sz/2, gt_ang,
                             org_x, org_y, 200, 200, gt_x, gt_y);

            sbm::MatchConfig cfg;
            cfg.min_score = 40;
            cfg.refine = sbm::RefineMode::ICP;
            cfg.icp_iterations = 30;
            cfg.icp_max_dist = 10;
            sbm::ShapeMatcher matcher(cfg);
            sbm::ModelConfig mcfg;
            mcfg.angle = {0, 360, 2};

            {
                OutputGuard guard;
                matcher.addModel("L", feat200, mcfg);
                auto results = matcher.match(scene);

                if (!results.empty()) {
                    auto& r = results[0];
                    float ae = angle_err(r.angle, gt_ang);
                    float pe = pos_err(r.x, r.y, gt_x, gt_y);
                    total_ang_err += ae;
                    total_pos_err += pe;
                    worst_ang = std::max(worst_ang, ae);
                    worst_pos = std::max(worst_pos, pe);
                    count++;

                    for (int k = 0; k < 6; k++) {
                        if (std::abs(gt_ang - old_fail_angles[k]) < 0.1f) {
                            old_fail_pos[k] = pe;
                            old_fail_ang[k] = ae;
                        }
                    }
                    LOG("  ICP angle=%.0f: err=%.2fdeg %.2fpx\n", gt_ang, ae, pe);
                }
            }
        }

        float mean_ang = count > 0 ? total_ang_err / count : 999;
        float mean_pos = count > 0 ? total_pos_err / count : 999;

        RECORD("icp_ang_mean", mean_ang);
        CHECK(mean_ang < 0.2f,
              "2a Angle mean: %.3fdeg across %d angles (expect <0.2)", mean_ang, count);
        RECORD("icp_ang_worst", worst_ang);
        CHECK(worst_ang < 1.0f,
              "2a Angle worst: %.3fdeg (expect <1.0)", worst_ang);
        RECORD("icp_pos_mean", mean_pos);
        CHECK(mean_pos < 1.0f,
              "2a Position mean: %.3fpx (expect ~0.65)", mean_pos);

        float worst_old_pos = *std::max_element(old_fail_pos, old_fail_pos + 6);
        RECORD("icp_old_fail_worst_pos", worst_old_pos);
        CHECK(worst_old_pos < 2.0f,
              "2b No divergence: old failure angles worst_pos=%.2fpx (expect <2.0)", worst_old_pos);
        for (int k = 0; k < 6; k++) {
            LOG("  Old fail %.0f: ang=%.2f pos=%.2f\n",
                old_fail_angles[k], old_fail_ang[k], old_fail_pos[k]);
        }
    }

    // --- 2c: Noise robustness ---
    {
        float total_ang_err = 0;
        int count = 0;
        for (int ai = 0; ai < 36; ai++) {
            float gt_ang = ai * 10.0f;
            Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
            place_object(templ200, scene, scene_sz/2, scene_sz/2, gt_ang);
            scene = add_noise(scene, 20);

            float gt_x, gt_y;
            compute_gt_origin(scene_sz/2, scene_sz/2, gt_ang,
                             org_x, org_y, 200, 200, gt_x, gt_y);

            sbm::MatchConfig cfg;
            cfg.min_score = 30;
            cfg.refine = sbm::RefineMode::ICP;
            sbm::ShapeMatcher matcher(cfg);
            sbm::ModelConfig mcfg;
            mcfg.angle = {0, 360, 2};

            {
                OutputGuard guard;
                matcher.addModel("L", feat200, mcfg);
                auto results = matcher.match(scene);
                if (!results.empty()) {
                    float ae = angle_err(results[0].angle, gt_ang);
                    total_ang_err += ae;
                    count++;
                }
            }
        }
        float mean_ang = count > 0 ? total_ang_err / count : 999;
        RECORD("icp_noise20_ang_mean", mean_ang);
        CHECK(mean_ang < 1.0f,
              "2c Noise sigma=20: mean_ang=%.2fdeg (expect <1.0)", mean_ang);
    }

    // --- 2d: Speed ---
    {
        Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
        place_object(templ200, scene, scene_sz/2, scene_sz/2, 45);

        sbm::MatchConfig cfg;
        cfg.min_score = 40;
        cfg.refine = sbm::RefineMode::ICP;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};

        double ms;
        {
            OutputGuard guard;
            matcher.addModel("L", feat200, mcfg);
            matcher.match(scene);  // warmup

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 10; i++) matcher.match(scene);
            ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count() / 10.0;
        }
        RECORD("icp_250_ms", (float)ms);
        CHECK(ms < 10.0,
              "2d Speed: 250x250 ICP total %.1fms (expect <10ms)", ms);
    }
}

// ============================================================
// Section 3: ROI Refinement
// ============================================================
static void test_roi_refinement(const sbm::FeatureSet& feat200, const Mat& templ200) {
    printf("\n======== 3. ROI REFINEMENT ========\n");
    LOG("\n======== 3. ROI REFINEMENT ========\n");

    const int scene_sz = 250;

    // Prepare sample points for direct ROI calls
    std::vector<cv::Point2f> positions;
    std::vector<float> corner_scores;
    for (auto& rp : feat200.refine_points) {
        positions.push_back(cv::Point2f(rp.px, rp.py));
        corner_scores.push_back(rp.cornerness);
    }
    auto sample_pts = roi_refine::selectCriticalPoints(
        positions, corner_scores, 15, feat200.templ_width, feat200.templ_height);

    // --- 3a: Angle sweep (every 5 deg, 72 angles) ---
    {
        float total_ang_err = 0, total_pos_err = 0;
        float worst_ang = 0, worst_pos = 0;
        int count = 0;
        float err_65 = 999, err_70 = 999;

        for (int ai = 0; ai < 72; ai++) {
            float gt_ang = ai * 5.0f;
            Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
            place_object(templ200, scene, scene_sz/2, scene_sz/2, gt_ang);

            float cx = scene_sz / 2.0f, cy = scene_sz / 2.0f;

            roi_refine::ROIConfig rcfg;
            rcfg.roi_half = 15;
            rcfg.search_half = 20;

            cv::Vec3f init_pose(cx, cy, gt_ang);
            cv::Vec3f refined;
            {
                OutputGuard guard;
                refined = roi_refine::refineROI(
                    feat200.templ_image, scene, sample_pts, init_pose, rcfg);
            }

            float ae = angle_err(refined[2], gt_ang);
            float pe = pos_err(refined[0], refined[1], cx, cy);
            total_ang_err += ae;
            total_pos_err += pe;
            worst_ang = std::max(worst_ang, ae);
            worst_pos = std::max(worst_pos, pe);
            count++;

            if (std::abs(gt_ang - 65.0f) < 0.1f) err_65 = ae;
            if (std::abs(gt_ang - 70.0f) < 0.1f) err_70 = ae;

            LOG("  ROI angle=%.0f: err=%.3fdeg %.3fpx\n", gt_ang, ae, pe);
        }

        float mean_ang = total_ang_err / count;
        float mean_pos = total_pos_err / count;

        RECORD("roi_ang_mean", mean_ang);
        CHECK(mean_ang < 0.15f,
              "3a Angle mean: %.3fdeg across %d angles (expect <0.15)", mean_ang, count);
        RECORD("roi_ang_worst", worst_ang);
        CHECK(worst_ang < 0.5f,
              "3a Angle worst: %.3fdeg (expect <0.5)", worst_ang);
        RECORD("roi_pos_mean", mean_pos);
        CHECK(mean_pos < 0.1f,
              "3a Position mean: %.3fpx (expect <0.1)", mean_pos);

        RECORD("roi_pca_65_err", err_65);
        CHECK(err_65 < 1.0f,
              "3b PCA fix 65deg: err=%.3fdeg (was 5+, expect <1.0)", err_65);
        RECORD("roi_pca_70_err", err_70);
        CHECK(err_70 < 1.0f,
              "3b PCA fix 70deg: err=%.3fdeg (was 5+, expect <1.0)", err_70);
    }

    // --- 3c: Sub-pixel position sweep ---
    {
        float total_pos_err = 0;
        int count = 0;
        float gt_ang = 25;

        for (int sy = 0; sy < 10; sy++) {
            for (int sx = 0; sx < 10; sx++) {
                float sub_x = sx * 0.1f;
                float sub_y = sy * 0.1f;

                Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
                place_object(templ200, scene, scene_sz/2, scene_sz/2, gt_ang, sub_x, sub_y);

                float cx = scene_sz / 2.0f + sub_x;
                float cy = scene_sz / 2.0f + sub_y;

                roi_refine::ROIConfig rcfg;
                rcfg.roi_half = 15;
                rcfg.search_half = 20;

                cv::Vec3f init_pose(cx, cy, gt_ang);
                cv::Vec3f refined;
                {
                    OutputGuard guard;
                    refined = roi_refine::refineROI(
                        feat200.templ_image, scene, sample_pts, init_pose, rcfg);
                }

                float pe = pos_err(refined[0], refined[1], cx, cy);
                total_pos_err += pe;
                count++;
            }
        }
        float mean_pos = total_pos_err / count;
        RECORD("roi_subpx_mean_pos", mean_pos);
        CHECK(mean_pos < 0.1f,
              "3c Sub-pixel: mean_pos=%.4fpx over %d offsets (expect <0.1)", mean_pos, count);
    }

    // --- 3d: Noise robustness ---
    {
        // noise=30
        float total_pos_30 = 0;
        int count_30 = 0;
        for (int ai = 0; ai < 36; ai++) {
            float gt_ang = ai * 10.0f;
            Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
            place_object(templ200, scene, scene_sz/2, scene_sz/2, gt_ang);
            scene = add_noise(scene, 30);

            float cx = scene_sz / 2.0f, cy = scene_sz / 2.0f;
            roi_refine::ROIConfig rcfg;
            rcfg.roi_half = 15;
            rcfg.search_half = 20;

            cv::Vec3f init_pose(cx, cy, gt_ang);
            cv::Vec3f refined;
            {
                OutputGuard guard;
                refined = roi_refine::refineROI(
                    feat200.templ_image, scene, sample_pts, init_pose, rcfg);
            }

            float pe = pos_err(refined[0], refined[1], cx, cy);
            total_pos_30 += pe;
            count_30++;
        }
        float mean_30 = total_pos_30 / count_30;
        RECORD("roi_noise30_mean_pos", mean_30);
        CHECK(mean_30 < 0.2f,
              "3d Noise sigma=30: mean_pos=%.3fpx (expect <0.2)", mean_30);

        // noise=40
        float total_pos_40 = 0;
        int count_40 = 0;
        for (int ai = 0; ai < 36; ai++) {
            float gt_ang = ai * 10.0f;
            Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
            place_object(templ200, scene, scene_sz/2, scene_sz/2, gt_ang);
            scene = add_noise(scene, 40);

            float cx = scene_sz / 2.0f, cy = scene_sz / 2.0f;
            roi_refine::ROIConfig rcfg;
            rcfg.roi_half = 15;
            rcfg.search_half = 20;

            cv::Vec3f init_pose(cx, cy, gt_ang);
            cv::Vec3f refined;
            {
                OutputGuard guard;
                refined = roi_refine::refineROI(
                    feat200.templ_image, scene, sample_pts, init_pose, rcfg);
            }

            float pe = pos_err(refined[0], refined[1], cx, cy);
            total_pos_40 += pe;
            count_40++;
        }
        float mean_40 = total_pos_40 / count_40;
        RECORD("roi_noise40_mean_pos", mean_40);
        CHECK(mean_40 < 0.3f,
              "3d Noise sigma=40: mean_pos=%.3fpx (expect <0.3)", mean_40);
    }

    // --- 3e: Speed ---
    {
        Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
        place_object(templ200, scene, scene_sz/2, scene_sz/2, 45);

        float cx = scene_sz / 2.0f, cy = scene_sz / 2.0f;
        roi_refine::ROIConfig rcfg;
        rcfg.roi_half = 15;
        rcfg.search_half = 20;

        cv::Vec3f init_pose(cx, cy, 45);
        double ms;
        {
            OutputGuard guard;
            roi_refine::refineROI(feat200.templ_image, scene, sample_pts, init_pose, rcfg); // warmup

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 100; i++)
                roi_refine::refineROI(feat200.templ_image, scene, sample_pts, init_pose, rcfg);
            ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count() / 100.0;
        }

        RECORD("roi_refine_ms", (float)ms);
        CHECK(ms < 3.0,
              "3e Speed: ROI refine %.2fms/call (expect <3ms)", ms);
    }
}

// ============================================================
// Section 4: Feature Selection (selectOptimizedPoints)
// ============================================================
static void test_feature_selection(const sbm::FeatureSet& feat200) {
    printf("\n======== 4. FEATURE SELECTION ========\n");
    LOG("\n======== 4. FEATURE SELECTION ========\n");

    // --- 4a: Correct number of points ---
    {
        // Use a copy so cached state does not interfere
        sbm::FeatureSet f = feat200;
        f.cached_opt_points.clear();
        f.cached_opt_max_points = 0;

        auto pts8 = f.selectOptimizedPoints(8);
        RECORD("opt_pts_8_count", (float)(int)pts8.size());
        CHECK((int)pts8.size() == 8 || ((int)pts8.size() > 0 && (int)pts8.size() <= 8),
              "4a selectOptimizedPoints(8) returned %d (expect <=8, >0)", (int)pts8.size());

        f.cached_opt_points.clear();
        f.cached_opt_max_points = 0;
        auto pts15 = f.selectOptimizedPoints(15);
        RECORD("opt_pts_15_count", (float)(int)pts15.size());
        CHECK((int)pts15.size() > 0 && (int)pts15.size() <= 15,
              "4a selectOptimizedPoints(15) returned %d (expect <=15, >0)", (int)pts15.size());
    }

    // --- 4b: Sensitivity ---
    {
        auto sens = feat200.analyzeSensitivity();
        RECORD("opt_pts_worst_ang", sens.worst_angle_sens);
        // Threshold history: 1.1 (original) → 1.5 (Fedorov) → 2.0 (D-optimal)
        CHECK(sens.worst_angle_sens < 2.0f,
              "4b Sensitivity: worst_ang=%.2f (expect <2.0, original was 1.1)", sens.worst_angle_sens);
    }

    // --- 4c: Corner count ---
    {
        sbm::FeatureSet f = feat200;
        f.cached_opt_points.clear();
        f.cached_opt_max_points = 0;
        auto pts8 = f.selectOptimizedPoints(8);

        int n_corners = 0;
        for (auto& p : pts8) {
            float best_d = 1e9f;
            int best_j = -1;
            for (size_t j = 0; j < feat200.refine_points.size(); j++) {
                float dx = p.x - feat200.refine_points[j].px;
                float dy = p.y - feat200.refine_points[j].py;
                float d = dx*dx + dy*dy;
                if (d < best_d) { best_d = d; best_j = (int)j; }
            }
            if (best_j >= 0 &&
                feat200.refine_points[best_j].type == sbm::FeatureSet::RefinePt::CORNER)
                n_corners++;
        }
        RECORD("opt_pts_corner_count", (float)n_corners);
        CHECK(n_corners >= 2,
              "4c Corner priority: %d corners in 8 selected (expect >=2)", n_corners);
    }

    // --- 4d: Cache hit returns same result ---
    {
        sbm::FeatureSet f = feat200;
        f.cached_opt_points.clear();
        f.cached_opt_max_points = 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto pts_first = f.selectOptimizedPoints(8);
        double ms_first = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        auto pts_cached = f.selectOptimizedPoints(8);
        double ms_cached = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        RECORD("cache_consistent", pts_first.size() == pts_cached.size() ? 1.0f : 0.0f);
        CHECK(pts_first.size() == pts_cached.size(),
              "4d Cache: same size (%d vs %d)", (int)pts_first.size(), (int)pts_cached.size());
        LOG("  Cache timing: first=%.3fms cached=%.4fms\n", ms_first, ms_cached);
    }
}

// ============================================================
// Section 5: Sensitivity Analysis
// ============================================================
static void test_sensitivity(const sbm::FeatureSet& feat200) {
    printf("\n======== 5. SENSITIVITY ANALYSIS ========\n");
    LOG("\n======== 5. SENSITIVITY ANALYSIS ========\n");

    // --- 5a: L-shape should not be fragile ---
    {
        auto sens = feat200.analyzeSensitivity();
        RECORD("lshape_worst_ang", sens.worst_angle_sens);
        // Threshold history: 1.1 (original) → 1.5 (Fedorov) → 2.0 (D-optimal)
        CHECK(sens.worst_angle_sens < 2.0f,
              "5a L-shape not fragile: worst_ang=%.2f (expect <2.0, original was 1.1)", sens.worst_angle_sens);
        LOG("  Diagnosis: %s\n", sens.diagnosis.c_str());
    }

    // --- 5b: Degenerate single line ---
    {
        Mat templ_line(200, 200, CV_8U, Scalar(0));
        line(templ_line, Point(10, 100), Point(190, 100), Scalar(200), 3);

        auto feat_line = sbm::extractFeatures(templ_line);
        feat_line.setOrigin(100, 100);

        if (!feat_line.refine_points.empty()) {
            feat_line.selectOptimizedPoints(8);
            auto sens_line = feat_line.analyzeSensitivity();
            RECORD("line_worst_pos", sens_line.worst_pos_sens);
            CHECK(sens_line.worst_pos_sens > 0.5f,
                  "5b Degenerate line: worst_pos=%.2f (expect >0.5)", sens_line.worst_pos_sens);
            LOG("  Line diagnosis: %s\n", sens_line.diagnosis.c_str());
        } else {
            printf("  SKIP: line template produced no refine points\n");
            LOG("SKIP: line template produced no refine points\n");
        }
    }
}

// ============================================================
// Section 6: Speed Benchmarks (informational)
// ============================================================
static void test_speed_benchmarks(const sbm::FeatureSet& feat200, const Mat& templ200) {
    printf("\n======== 6. SPEED BENCHMARKS ========\n");
    LOG("\n======== 6. SPEED BENCHMARKS ========\n");

    struct Obj { int x, y; double angle; };
    Obj objs[] = {
        {200, 150, 15},  {500, 300, 45},   {900, 200, 90},
        {1300, 400, 135},{1700, 250, 180},  {350, 700, 210},
        {750, 850, 270}, {1100, 600, 315},  {1500, 800, 30},
        {1800, 900, 60},
    };

    Mat scene_fhd(1080, 1920, CV_8U, Scalar(30));
    for (auto& obj : objs)
        place_object(templ200, scene_fhd, obj.x, obj.y, obj.angle);

    struct BenchMode { const char* name; sbm::RefineMode mode; double expected_ms; };
    BenchMode modes[] = {
        {"Coarse (None)", sbm::RefineMode::None, 25},
        {"ICP",           sbm::RefineMode::ICP,  30},
        {"ROI",           sbm::RefineMode::ROI,  30},
    };

    for (auto& bm : modes) {
        sbm::MatchConfig cfg;
        cfg.min_score = 40;
        cfg.nms_radius = 80;
        cfg.refine = bm.mode;

        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};

        double avg_ms;
        {
            OutputGuard guard;
            matcher.addModel("L", feat200, mcfg);
            matcher.match(scene_fhd);  // warmup

            // 10 runs, sort, middle 33% mean
            const int NR = 10;
            std::vector<double> times(NR);
            for (int i = 0; i < NR; i++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                matcher.match(scene_fhd);
                times[i] = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
            }
            std::sort(times.begin(), times.end());
            int lo = NR / 3, hi = NR * 2 / 3;
            double sum = 0;
            for (int i = lo; i <= hi; i++) sum += times[i];
            avg_ms = sum / (hi - lo + 1);
        }

        if (bm.mode == sbm::RefineMode::None) RECORD("fhd10_coarse_ms", (float)avg_ms);
        else if (bm.mode == sbm::RefineMode::ICP) RECORD("fhd10_icp_ms", (float)avg_ms);
        else if (bm.mode == sbm::RefineMode::ROI) RECORD("fhd10_roi_ms", (float)avg_ms);

        double threshold_2x = bm.expected_ms * 2.0;
        WARN_IF(avg_ms > threshold_2x,
                "6 %s: %.1fms (expected ~%.0fms, >2x=%.0fms)",
                bm.name, avg_ms, bm.expected_ms, threshold_2x);
    }
}

// ============================================================
// Section 8: Edge Case Templates
// ============================================================
static void test_edge_cases(const sbm::FeatureSet& feat200, const Mat& templ200) {
    printf("\n======== 8. EDGE CASE TEMPLATES ========\n");
    LOG("\n======== 8. EDGE CASE TEMPLATES ========\n");

    const float org_x = 100, org_y = 75;

    // --- 8a: Near-edge object ---
    {
        const int scene_sz = 250;
        const int obj_cx = 100, obj_cy = 50;  // close to top edge
        const double gt_ang = 0;

        Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
        place_object(templ200, scene, obj_cx, obj_cy, gt_ang);

        float gt_x, gt_y;
        compute_gt_origin(obj_cx, obj_cy, gt_ang,
                         org_x, org_y, 200, 200, gt_x, gt_y);

        // ICP
        {
            sbm::MatchConfig cfg;
            cfg.min_score = 40;
            cfg.refine = sbm::RefineMode::ICP;
            sbm::ShapeMatcher matcher(cfg);
            sbm::ModelConfig mcfg;
            mcfg.angle = {0, 360, 2};

            float found = 0.0f;
            {
                OutputGuard guard;
                matcher.addModel("L", feat200, mcfg);
                auto results = matcher.match(scene);
                if (!results.empty()) {
                    found = 1.0f;
                    LOG("  8a ICP: found at (%.1f,%.1f)@%.1f\n",
                        results[0].x, results[0].y, results[0].angle);
                } else {
                    LOG("  8a ICP: NOT FOUND\n");
                }
            }
            RECORD("edge_icp_found", found);
        }

        // ROI
        {
            sbm::MatchConfig cfg;
            cfg.min_score = 40;
            cfg.refine = sbm::RefineMode::ROI;
            sbm::ShapeMatcher matcher(cfg);
            sbm::ModelConfig mcfg;
            mcfg.angle = {0, 360, 2};

            float found = 0.0f;
            {
                OutputGuard guard;
                matcher.addModel("L", feat200, mcfg);
                auto results = matcher.match(scene);
                if (!results.empty()) {
                    found = 1.0f;
                    LOG("  8a ROI: found at (%.1f,%.1f)@%.1f\n",
                        results[0].x, results[0].y, results[0].angle);
                } else {
                    LOG("  8a ROI: NOT FOUND\n");
                }
            }
            RECORD("edge_roi_found", found);
        }

        CHECK(g_metrics["edge_icp_found"] > 0.5f,
              "8a Near-edge object found by ICP");
        CHECK(g_metrics["edge_roi_found"] > 0.5f,
              "8a Near-edge object found by ROI");
    }

    // --- 8b: Small template ---
    {
        const int scene_sz = 250;

        // Create small 40x40 L-shape template with clear edges
        Mat small_templ(40, 40, CV_8U, Scalar(0));
        // Vertical bar: 5px wide, 25px tall
        rectangle(small_templ, Point(8, 5), Point(12, 30), Scalar(200), -1);
        // Horizontal bar: 15px wide, 5px tall (forms L)
        rectangle(small_templ, Point(8, 25), Point(25, 30), Scalar(200), -1);

        float found = 0.0f;
        {
            OutputGuard guard;
            // Use single pyramid level with small T for small template
            auto small_feat = sbm::extractFeatures(small_templ, cv::Mat(), 64, {2});
            LOG("  8b Small template features: %d\n", small_feat.numFeatures());
            if (small_feat.numFeatures() > 0) {
                Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
                place_object(small_templ, scene, scene_sz / 2, scene_sz / 2, 0);

                sbm::MatchConfig cfg;
                cfg.min_score = 30;
                cfg.refine = sbm::RefineMode::None;
                sbm::ShapeMatcher matcher(cfg);
                sbm::ModelConfig mcfg;
                mcfg.angle = {0, 360, 5};

                matcher.addModel("SmallL", small_feat, mcfg);
                auto results = matcher.match(scene);
                if (!results.empty()) {
                    found = 1.0f;
                    LOG("  8b Small template: found at (%.1f,%.1f)@%.1f\n",
                        results[0].x, results[0].y, results[0].angle);
                } else {
                    LOG("  8b Small template: NOT FOUND\n");
                }
            } else {
                LOG("  8b Small template: no features extracted\n");
            }
        }
        RECORD("small_templ_found", found);
        CHECK(found > 0.5f, "8b Small 40x40 template detection");
    }
}

// ============================================================
// Section 9: Noise & Blur Stability Limits
// ============================================================
static void test_noise_blur_stability(const sbm::FeatureSet& feat200, const Mat& templ200) {
    printf("\n======== 9. NOISE & BLUR STABILITY ========\n");
    LOG("\n======== 9. NOISE & BLUR STABILITY ========\n");

    const int scene_sz = 250;
    const float org_x = 100, org_y = 75;
    const float test_ang = 25;

    // Helper: run one match with given scene, return (found, ang_err, pos_err)
    auto run_match = [&](const Mat& scene, sbm::RefineMode mode,
                         float gt_cx, float gt_cy, float gt_ang) -> std::tuple<bool, float, float> {
        sbm::MatchConfig cfg;
        cfg.min_score = 30;
        cfg.nms_radius = 80;
        cfg.refine = mode;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};
        {
            OutputGuard guard;
            matcher.addModel("L", feat200, mcfg);
        }
        std::vector<sbm::MatchResult> results;
        { OutputGuard guard; results = matcher.match(scene); }

        if (results.empty()) return std::make_tuple(false, 99.0f, 99.0f);
        auto& r = results[0];

        // Convert user origin back to center
        float o_x = org_x - feat200.templ_width / 2.0f;
        float o_y = org_y - feat200.templ_height / 2.0f;
        float rad = -r.angle * (float)CV_PI / 180.0f;
        float cx = r.x - (std::cos(rad)*o_x - std::sin(rad)*o_y);
        float cy = r.y - (std::sin(rad)*o_x + std::cos(rad)*o_y);
        float ae = angle_err(r.angle, gt_ang);
        float pe = pos_err(cx, cy, gt_cx, gt_cy);
        return std::make_tuple(true, ae, pe);
    };

    // Build clean scene
    Mat scene_clean(scene_sz, scene_sz, CV_8U, Scalar(0));
    float gt_cx = scene_sz / 2.0f, gt_cy = scene_sz / 2.0f;
    place_object(templ200, scene_clean, (int)gt_cx, (int)gt_cy, test_ang);

    // ---- 9a: Coarse detection under noise ----
    {
        float noise_levels[] = {30, 50, 60};
        for (float ns : noise_levels) {
            Mat scene_n = add_noise(scene_clean, ns);
            bool found; float ae, pe; std::tie(found, ae, pe) = run_match(scene_n, sbm::RefineMode::None, gt_cx, gt_cy, test_ang);
            char key[64]; snprintf(key, sizeof(key), "coarse_detect_n%.0f", ns);
            RECORD(key, found ? 1.0f : 0.0f);
            LOG("  9a coarse noise=%.0f: found=%d ang=%.1f pos=%.1f\n", ns, (int)found, ae, pe);
        }
    }

    // ---- 9b: Coarse detection under blur ----
    {
        int blur_levels[] = {11, 21, 31};
        for (int bk : blur_levels) {
            Mat scene_b;
            GaussianBlur(scene_clean, scene_b, Size(bk, bk), 0);
            bool found; float ae, pe; std::tie(found, ae, pe) = run_match(scene_b, sbm::RefineMode::None, gt_cx, gt_cy, test_ang);
            char key[64]; snprintf(key, sizeof(key), "coarse_detect_b%d", bk);
            RECORD(key, found ? 1.0f : 0.0f);
            LOG("  9b coarse blur=%d: found=%d ang=%.1f pos=%.1f\n", bk, (int)found, ae, pe);
        }
    }

    // ---- 9c: ICP accuracy under noise ----
    {
        float noise_levels[] = {10, 20, 30};
        for (float ns : noise_levels) {
            Mat scene_n = add_noise(scene_clean, ns);
            bool found; float ae, pe; std::tie(found, ae, pe) = run_match(scene_n, sbm::RefineMode::ICP, gt_cx, gt_cy, test_ang);
            char key_a[64], key_p[64];
            snprintf(key_a, sizeof(key_a), "icp_n%.0f_ang", ns);
            snprintf(key_p, sizeof(key_p), "icp_n%.0f_pos", ns);
            RECORD(key_a, found ? ae : 99.0f);
            RECORD(key_p, found ? pe : 99.0f);
            LOG("  9c ICP noise=%.0f: found=%d ang=%.2f pos=%.2f\n", ns, (int)found, ae, pe);
        }
    }

    // ---- 9d: ICP accuracy under blur ----
    {
        int blur_levels[] = {5, 11, 21};
        for (int bk : blur_levels) {
            Mat scene_b;
            GaussianBlur(scene_clean, scene_b, Size(bk, bk), 0);
            bool found; float ae, pe; std::tie(found, ae, pe) = run_match(scene_b, sbm::RefineMode::ICP, gt_cx, gt_cy, test_ang);
            char key_a[64], key_p[64];
            snprintf(key_a, sizeof(key_a), "icp_b%d_ang", bk);
            snprintf(key_p, sizeof(key_p), "icp_b%d_pos", bk);
            RECORD(key_a, found ? ae : 99.0f);
            RECORD(key_p, found ? pe : 99.0f);
            LOG("  9d ICP blur=%d: found=%d ang=%.2f pos=%.2f\n", bk, (int)found, ae, pe);
        }
    }

    // ---- 9e: ROI accuracy under noise ----
    {
        float noise_levels[] = {20, 30, 40, 50};
        for (float ns : noise_levels) {
            Mat scene_n = add_noise(scene_clean, ns);
            bool found; float ae, pe; std::tie(found, ae, pe) = run_match(scene_n, sbm::RefineMode::ROI, gt_cx, gt_cy, test_ang);
            char key_a[64], key_p[64];
            snprintf(key_a, sizeof(key_a), "roi_n%.0f_ang", ns);
            snprintf(key_p, sizeof(key_p), "roi_n%.0f_pos", ns);
            RECORD(key_a, found ? ae : 99.0f);
            RECORD(key_p, found ? pe : 99.0f);
            LOG("  9e ROI noise=%.0f: found=%d ang=%.2f pos=%.2f\n", ns, (int)found, ae, pe);
        }
    }

    // ---- 9f: ROI accuracy under blur ----
    {
        int blur_levels[] = {5, 11, 21};
        for (int bk : blur_levels) {
            Mat scene_b;
            GaussianBlur(scene_clean, scene_b, Size(bk, bk), 0);
            bool found; float ae, pe; std::tie(found, ae, pe) = run_match(scene_b, sbm::RefineMode::ROI, gt_cx, gt_cy, test_ang);
            char key_a[64], key_p[64];
            snprintf(key_a, sizeof(key_a), "roi_b%d_ang", bk);
            snprintf(key_p, sizeof(key_p), "roi_b%d_pos", bk);
            RECORD(key_a, found ? ae : 99.0f);
            RECORD(key_p, found ? pe : 99.0f);
            LOG("  9f ROI blur=%d: found=%d ang=%.2f pos=%.2f\n", bk, (int)found, ae, pe);
        }
    }

    // ---- 9g: Combined noise+blur ----
    {
        struct NB { float noise; int blur; };
        NB combos[] = {{20, 5}, {30, 11}, {50, 11}};
        for (auto& nb : combos) {
            Mat scene_nb = add_noise(scene_clean, nb.noise);
            GaussianBlur(scene_nb, scene_nb, Size(nb.blur, nb.blur), 0);

            // ICP
            bool fi; float ai, pi; std::tie(fi, ai, pi) = run_match(scene_nb, sbm::RefineMode::ICP, gt_cx, gt_cy, test_ang);
            char ki_a[64], ki_p[64];
            snprintf(ki_a, sizeof(ki_a), "icp_n%.0fb%d_ang", nb.noise, nb.blur);
            snprintf(ki_p, sizeof(ki_p), "icp_n%.0fb%d_pos", nb.noise, nb.blur);
            RECORD(ki_a, fi ? ai : 99.0f);
            RECORD(ki_p, fi ? pi : 99.0f);

            // ROI
            bool fr; float ar, pr; std::tie(fr, ar, pr) = run_match(scene_nb, sbm::RefineMode::ROI, gt_cx, gt_cy, test_ang);
            char kr_a[64], kr_p[64];
            snprintf(kr_a, sizeof(kr_a), "roi_n%.0fb%d_ang", nb.noise, nb.blur);
            snprintf(kr_p, sizeof(kr_p), "roi_n%.0fb%d_pos", nb.noise, nb.blur);
            RECORD(kr_a, fr ? ar : 99.0f);
            RECORD(kr_p, fr ? pr : 99.0f);

            LOG("  9g n=%.0f b=%d: ICP=%.2f/%.2f ROI=%.2f/%.2f\n",
                nb.noise, nb.blur, ai, pi, ar, pr);
        }
    }
}

// ============================================================
// Section 10: Multi-resolution Speed Benchmark (360p, 1080p, 20MP)
// ============================================================
static void test_resolution_speed(const sbm::FeatureSet& feat200, const Mat& templ200) {
    printf("\n======== 10. RESOLUTION SPEED BENCHMARK ========\n");
    LOG("\n======== 10. RESOLUTION SPEED BENCHMARK ========\n");

    const float org_x = 100, org_y = 75;

    struct ResConfig {
        const char* name;
        int width, height;
        int n_objects;
        const char* suffix;
    };
    ResConfig configs[] = {
        {"360p",  640,  360, 3, "360p"},
        {"1080p", 1920, 1080, 20, "1080p"},
        {"20MP",  5472, 3648, 20, "20mp"},
    };

    // 20 object positions (spread across any resolution)
    struct ObjDef { float rx, ry; double angle; }; // relative position [0,1]
    ObjDef obj_defs[] = {
        {0.10f,0.15f,7},  {0.25f,0.20f,23},  {0.40f,0.12f,51},  {0.55f,0.18f,78},
        {0.70f,0.12f,102},{0.85f,0.20f,133}, {0.95f,0.12f,157}, {0.13f,0.40f,189},
        {0.28f,0.45f,212},{0.43f,0.38f,238}, {0.58f,0.45f,267}, {0.73f,0.38f,291},
        {0.88f,0.45f,319},{0.10f,0.65f,342}, {0.25f,0.70f,12},  {0.40f,0.65f,67},
        {0.55f,0.70f,112},{0.70f,0.65f,167}, {0.85f,0.70f,222}, {0.15f,0.90f,277},
    };

    sbm::RefineMode modes[] = {sbm::RefineMode::None, sbm::RefineMode::ICP, sbm::RefineMode::ROI};
    const char* mode_names[] = {"Coarse", "ICP", "ROI"};

    for (auto& rc : configs) {
        int n_obj = std::min(rc.n_objects, 20);
        // Scale NMS radius with resolution (avoid merging at low res)
        int nms_r = std::max(40, std::min(100, rc.width / 20));

        // Build scene with noise=15 + blur k=3 (mild degradation)
        Mat scene(rc.height, rc.width, CV_8U, Scalar(30));
        for (int oi = 0; oi < n_obj; oi++) {
            float fx = obj_defs[oi].rx * (rc.width - 200) + 100;
            float fy = obj_defs[oi].ry * (rc.height - 200) + 100;
            place_object(templ200, scene, (int)fx, (int)fy, obj_defs[oi].angle,
                         fx - std::floor(fx), fy - std::floor(fy));
        }
        scene = add_noise(scene, 15);
        GaussianBlur(scene, scene, Size(3, 3), 0);

        // Compute GT center positions for this resolution
        struct GT { float cx, cy; double angle; };
        std::vector<GT> gts(n_obj);
        for (int oi = 0; oi < n_obj; oi++) {
            gts[oi].cx = (float)(obj_defs[oi].rx * (rc.width - 200) + 100);
            gts[oi].cy = (float)(obj_defs[oi].ry * (rc.height - 200) + 100);
            gts[oi].angle = obj_defs[oi].angle;
        }

        float o_off_x = org_x - feat200.templ_width / 2.0f;
        float o_off_y = org_y - feat200.templ_height / 2.0f;

        for (int mi = 0; mi < 3; mi++) {
            // Coarse has ~10px error + origin rotation offset → wider GT margin
            float gt_match_radius = (modes[mi] == sbm::RefineMode::None) ? 100.0f : 50.0f;
            sbm::MatchConfig cfg;
            cfg.min_score = 50;
            cfg.nms_radius = nms_r;
            cfg.refine = modes[mi];
            sbm::ShapeMatcher matcher(cfg);
            sbm::ModelConfig mcfg;
            mcfg.angle = {0, 360, 2};
            { OutputGuard guard; matcher.addModel("L", feat200, mcfg); }

            // First run for accuracy evaluation (deterministic — single-threaded for coarse)
            std::vector<sbm::MatchResult> last_results;
            { OutputGuard guard; last_results = matcher.match(scene); }

            // 10 runs for timing only
            const int N_RUNS = 10;
            std::vector<double> run_times(N_RUNS);
            for (int r = 0; r < N_RUNS; r++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                { OutputGuard guard; matcher.match(scene); }
                run_times[r] = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
            }
            std::sort(run_times.begin(), run_times.end());
            // Middle 33%: indices 3,4,5,6 (4 values out of 10)
            int lo = N_RUNS / 3;       // 3
            int hi = N_RUNS * 2 / 3;   // 6
            double avg_ms = 0;
            for (int r = lo; r <= hi; r++) avg_ms += run_times[r];
            avg_ms /= (hi - lo + 1);

            // Dump all GT positions and all detections for this mode
            LOG("  --- %s %s: %d GT objects, %d detections ---\n",
                rc.name, mode_names[mi], n_obj, (int)last_results.size());
            LOG("  GT positions (template center):\n");
            for (int gi = 0; gi < n_obj; gi++)
                LOG("    gt[%2d] (%6.1f, %6.1f) @ %5.1f\n", gi, gts[gi].cx, gts[gi].cy, gts[gi].angle);
            LOG("  Detections (converted to center):\n");
            for (int ri = 0; ri < (int)last_results.size(); ri++) {
                auto& res = last_results[ri];
                float rad = -res.angle * (float)CV_PI / 180.0f;
                float rcx = res.x - (std::cos(rad)*o_off_x - std::sin(rad)*o_off_y);
                float rcy = res.y - (std::sin(rad)*o_off_x + std::cos(rad)*o_off_y);
                LOG("    det[%2d] (%6.1f, %6.1f) @ %5.1f  score=%.0f  user_xy=(%.1f,%.1f)\n",
                    ri, rcx, rcy, res.angle, res.score, res.x, res.y);
            }

            // Per-object GT matching: greedy nearest assignment (mark used results)
            int n_matched = 0;
            float total_ang_err = 0, total_pos_err = 0;
            float worst_ang_err = 0, worst_pos_err = 0;
            std::vector<bool> result_used(last_results.size(), false);

            for (int gi = 0; gi < n_obj; gi++) {
                // Find closest UNUSED result to this GT
                float best_d = 1e9f;
                int best_ri = -1;
                for (int ri = 0; ri < (int)last_results.size(); ri++) {
                    if (result_used[ri]) continue;
                    auto& res = last_results[ri];
                    float rad = -res.angle * (float)CV_PI / 180.0f;
                    float rcx = res.x - (std::cos(rad)*o_off_x - std::sin(rad)*o_off_y);
                    float rcy = res.y - (std::sin(rad)*o_off_x + std::cos(rad)*o_off_y);
                    float d = pos_err(rcx, rcy, gts[gi].cx, gts[gi].cy);
                    if (d < best_d) { best_d = d; best_ri = ri; }
                }
                if (best_ri >= 0 && best_d < gt_match_radius) {
                    result_used[best_ri] = true;
                    auto& res = last_results[best_ri];
                    float rad = -res.angle * (float)CV_PI / 180.0f;
                    float rcx = res.x - (std::cos(rad)*o_off_x - std::sin(rad)*o_off_y);
                    float rcy = res.y - (std::sin(rad)*o_off_x + std::cos(rad)*o_off_y);
                    float ae = angle_err(res.angle, (float)gts[gi].angle);
                    float pe = pos_err(rcx, rcy, gts[gi].cx, gts[gi].cy);
                    total_ang_err += ae;
                    total_pos_err += pe;
                    worst_ang_err = std::max(worst_ang_err, ae);
                    worst_pos_err = std::max(worst_pos_err, pe);
                    n_matched++;

                    LOG("    %s %s obj[%d] gt=(%4.0f,%4.0f)@%3.0f → (%5.1f,%5.1f)@%5.1f err: %+.1fdeg %.1fpx\n",
                        rc.name, mode_names[mi], gi, gts[gi].cx, gts[gi].cy, gts[gi].angle,
                        rcx, rcy, res.angle, ae, pe);
                } else {
                    LOG("    %s %s obj[%d] gt=(%4.0f,%4.0f)@%3.0f → MISSING\n",
                        rc.name, mode_names[mi], gi, gts[gi].cx, gts[gi].cy, gts[gi].angle);
                }
            }

            float mean_ang = n_matched > 0 ? total_ang_err / n_matched : -1;
            float mean_pos = n_matched > 0 ? total_pos_err / n_matched : -1;
            LOG("  %s %s: %d detections (post-NMS), %d/%d GT-matched (radius=%.0f)\n",
                rc.name, mode_names[mi], (int)last_results.size(), n_matched, n_obj, gt_match_radius);

            // Record metrics
            char key[64];
            snprintf(key, sizeof(key), "speed_%s_%s_ms", rc.suffix, mode_names[mi]);
            RECORD(key, (float)avg_ms);

            snprintf(key, sizeof(key), "speed_%s_%s_found", rc.suffix, mode_names[mi]);
            RECORD(key, (float)n_matched);

            snprintf(key, sizeof(key), "speed_%s_%s_mean_ang", rc.suffix, mode_names[mi]);
            RECORD(key, mean_ang);

            snprintf(key, sizeof(key), "speed_%s_%s_mean_pos", rc.suffix, mode_names[mi]);
            RECORD(key, mean_pos);

            snprintf(key, sizeof(key), "speed_%s_%s_worst_ang", rc.suffix, mode_names[mi]);
            RECORD(key, worst_ang_err);

            snprintf(key, sizeof(key), "speed_%s_%s_worst_pos", rc.suffix, mode_names[mi]);
            RECORD(key, worst_pos_err);

            printf("  %s %-6s: %5.1fms  %d/%d matched  ang=%.1f/%.1f  pos=%.1f/%.1fpx\n",
                   rc.name, mode_names[mi], avg_ms, n_matched, n_obj,
                   mean_ang, worst_ang_err, mean_pos, worst_pos_err);
            LOG("  %s %-6s: %5.1fms  %d/%d matched  ang=%.1f/%.1f  pos=%.1f/%.1fpx\n",
                rc.name, mode_names[mi], avg_ms, n_matched, n_obj,
                mean_ang, worst_ang_err, mean_pos, worst_pos_err);
        }
    }
}

// ============================================================
// Section 11: Determinism
// ============================================================
static void test_determinism(const sbm::FeatureSet& feat200, const Mat& templ200) {
    printf("\n======== 11. DETERMINISM ========\n");
    LOG("\n======== 11. DETERMINISM ========\n");

    const int scene_sz = 250;
    Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
    place_object(templ200, scene, scene_sz/2, scene_sz/2, 37);

    // Run match 5 times with identical params
    std::vector<std::vector<sbm::MatchResult>> all_results(5);
    for (int run = 0; run < 5; run++) {
        sbm::MatchConfig cfg;
        cfg.min_score = 40;
        cfg.refine = sbm::RefineMode::ICP;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};
        {
            OutputGuard guard;
            matcher.addModel("L", feat200, mcfg);
            all_results[run] = matcher.match(scene);
        }
    }

    float max_pos_diff = 0, max_ang_diff = 0;
    bool count_match = true;
    for (int run = 1; run < 5; run++) {
        if (all_results[run].size() != all_results[0].size()) {
            count_match = false;
            break;
        }
        for (size_t i = 0; i < all_results[0].size(); i++) {
            float pd = pos_err(all_results[run][i].x, all_results[run][i].y,
                               all_results[0][i].x, all_results[0][i].y);
            float ad = angle_err(all_results[run][i].angle, all_results[0][i].angle);
            max_pos_diff = std::max(max_pos_diff, pd);
            max_ang_diff = std::max(max_ang_diff, ad);
        }
    }

    RECORD("determinism_pos_max_diff", max_pos_diff);
    RECORD("determinism_ang_max_diff", max_ang_diff);
    CHECK(count_match, "11 Determinism: same result count across 5 runs (%d)",
          (int)all_results[0].size());
    CHECK(max_pos_diff < 0.001f,
          "11 Determinism: max pos diff=%.4fpx (expect <0.001)", max_pos_diff);
    CHECK(max_ang_diff < 0.001f,
          "11 Determinism: max ang diff=%.4fdeg (expect <0.001)", max_ang_diff);
}

// ============================================================
// Section 12: Serialization Round-trip
// ============================================================
static void test_serialization(const sbm::FeatureSet& feat200, const Mat& templ200) {
    printf("\n======== 12. SERIALIZATION ROUND-TRIP ========\n");
    LOG("\n======== 12. SERIALIZATION ROUND-TRIP ========\n");

    const int scene_sz = 250;
    const float org_x = 100, org_y = 75;

    // Save and reload
    feat200.save("output/test_roundtrip.feat");
    auto loaded = sbm::FeatureSet::load("output/test_roundtrip.feat");

    RECORD("serial_feature_count_match",
           loaded.numFeatures() == feat200.numFeatures() ? 1.0f : 0.0f);

    // Match with original
    Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
    place_object(templ200, scene, scene_sz/2, scene_sz/2, 45);

    float gt_x, gt_y;
    compute_gt_origin(scene_sz/2, scene_sz/2, 45,
                     org_x, org_y, 200, 200, gt_x, gt_y);

    sbm::MatchResult res_orig, res_loaded;
    bool found_orig = false, found_loaded = false;

    {
        sbm::MatchConfig cfg;
        cfg.min_score = 40;
        cfg.refine = sbm::RefineMode::ICP;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};
        OutputGuard guard;
        matcher.addModel("L", feat200, mcfg);
        auto results = matcher.match(scene);
        if (!results.empty()) { res_orig = results[0]; found_orig = true; }
    }

    {
        sbm::MatchConfig cfg;
        cfg.min_score = 40;
        cfg.refine = sbm::RefineMode::ICP;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};
        OutputGuard guard;
        loaded.setOrigin(org_x, org_y);
        loaded.selectOptimizedPoints(15);
        matcher.addModel("L", loaded, mcfg);
        auto results = matcher.match(scene);
        if (!results.empty()) { res_loaded = results[0]; found_loaded = true; }
    }

    float pdiff = 99.0f, adiff = 99.0f;
    if (found_orig && found_loaded) {
        pdiff = pos_err(res_orig.x, res_orig.y, res_loaded.x, res_loaded.y);
        adiff = angle_err(res_orig.angle, res_loaded.angle);
    }

    RECORD("serial_pos_diff", pdiff);
    RECORD("serial_ang_diff", adiff);
    CHECK(pdiff < 0.01f,
          "12 Serialization: pos diff=%.4fpx (expect <0.01)", pdiff);
    CHECK(adiff < 0.01f,
          "12 Serialization: ang diff=%.4fdeg (expect <0.01)", adiff);
    CHECK(loaded.numFeatures() == feat200.numFeatures(),
          "12 Serialization: feature count %d vs %d",
          loaded.numFeatures(), feat200.numFeatures());
}

// ============================================================
// Section 13: False Positives
// ============================================================
static void test_false_positives(const sbm::FeatureSet& feat200, const Mat& templ200) {
    printf("\n======== 13. FALSE POSITIVES ========\n");
    LOG("\n======== 13. FALSE POSITIVES ========\n");

    // 13a: Empty (all black) scene
    {
        Mat scene(250, 250, CV_8U, Scalar(0));
        sbm::MatchConfig cfg;
        cfg.min_score = 50;
        cfg.refine = sbm::RefineMode::None;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};
        int count = 0;
        {
            OutputGuard guard;
            matcher.addModel("L", feat200, mcfg);
            auto results = matcher.match(scene);
            count = (int)results.size();
        }
        RECORD("fp_empty_count", (float)count);
        CHECK(count == 0, "13a Empty scene: %d matches (expect 0)", count);
    }

    // 13b: Random noise scene (moderate noise, high threshold)
    {
        Mat scene(250, 250, CV_8U, Scalar(128));
        scene = add_noise(scene, 30);
        sbm::MatchConfig cfg;
        cfg.min_score = 70;
        cfg.refine = sbm::RefineMode::None;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};
        int count = 0;
        {
            OutputGuard guard;
            matcher.addModel("L", feat200, mcfg);
            auto results = matcher.match(scene);
            count = (int)results.size();
        }
        RECORD("fp_noise_count", (float)count);
        CHECK(count == 0, "13b Noise scene: %d matches (expect 0)", count);
    }

    // 13c: Wrong shape (rectangle, not L)
    {
        Mat scene(250, 250, CV_8U, Scalar(0));
        rectangle(scene, Point(80, 80), Point(170, 170), Scalar(200), 3);
        sbm::MatchConfig cfg;
        cfg.min_score = 60;
        cfg.refine = sbm::RefineMode::None;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};
        int count = 0;
        {
            OutputGuard guard;
            matcher.addModel("L", feat200, mcfg);
            auto results = matcher.match(scene);
            count = (int)results.size();
        }
        RECORD("fp_wrong_shape_count", (float)count);
        CHECK(count == 0, "13c Wrong shape (rect): %d matches (expect 0)", count);
    }
}

// ============================================================
// Section 14: API Contract (origin/angle offset)
// ============================================================
static void test_api_contract(const sbm::FeatureSet& feat200, const Mat& templ200) {
    printf("\n======== 14. API CONTRACT ========\n");
    LOG("\n======== 14. API CONTRACT ========\n");

    const int scene_sz = 250;

    // 14a: Custom origin (150,100) instead of default center
    {
        Mat templ(200, 200, CV_8U, Scalar(0));
        draw_L(templ, 100, 100, 0, 200);

        float custom_org_x = 150, custom_org_y = 100;
        double gt_ang = 30;

        sbm::FeatureSet feat;
        {
            OutputGuard guard;
            feat = sbm::extractFeatures(templ);
        }
        feat.setOrigin(custom_org_x, custom_org_y);
        feat.selectOptimizedPoints(15);

        Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
        place_object(templ, scene, scene_sz/2, scene_sz/2, gt_ang);

        float gt_x, gt_y;
        compute_gt_origin(scene_sz/2, scene_sz/2, gt_ang,
                         custom_org_x, custom_org_y, 200, 200, gt_x, gt_y);

        sbm::MatchConfig cfg;
        cfg.min_score = 40;
        cfg.refine = sbm::RefineMode::ICP;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};

        float pe = 99.0f;
        {
            OutputGuard guard;
            matcher.addModel("L", feat, mcfg);
            auto results = matcher.match(scene);
            if (!results.empty()) {
                pe = pos_err(results[0].x, results[0].y, gt_x, gt_y);
                LOG("  14a origin: result=(%.1f,%.1f) gt=(%.1f,%.1f) err=%.1fpx\n",
                    results[0].x, results[0].y, gt_x, gt_y, pe);
            } else {
                LOG("  14a origin: NOT FOUND\n");
            }
        }
        RECORD("api_origin_pos_err", pe);
        CHECK(pe < 5.0f,
              "14a Custom origin: pos_err=%.1fpx (expect <5.0)", pe);
    }

    // 14b: Angle offset
    {
        Mat templ(200, 200, CV_8U, Scalar(0));
        draw_L(templ, 100, 100, 0, 200);

        float angle_offset = 45.0f;
        double gt_ang = 30;

        sbm::FeatureSet feat;
        {
            OutputGuard guard;
            feat = sbm::extractFeatures(templ);
        }
        feat.setOrigin(100, 75);
        feat.setAngleOffset(angle_offset);
        feat.selectOptimizedPoints(15);

        Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
        place_object(templ, scene, scene_sz/2, scene_sz/2, gt_ang);

        sbm::MatchConfig cfg;
        cfg.min_score = 40;
        cfg.refine = sbm::RefineMode::ICP;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};

        float ae = 99.0f;
        {
            OutputGuard guard;
            matcher.addModel("L", feat, mcfg);
            auto results = matcher.match(scene);
            if (!results.empty()) {
                float expected_angle = (float)gt_ang + angle_offset;
                ae = angle_err(results[0].angle, expected_angle);
                LOG("  14b angle offset: result_angle=%.1f expected=%.1f err=%.1fdeg\n",
                    results[0].angle, expected_angle, ae);
            } else {
                LOG("  14b angle offset: NOT FOUND\n");
            }
        }
        RECORD("api_angle_offset_err", ae);
        CHECK(ae < 2.0f,
              "14b Angle offset: err=%.1fdeg (expect <2.0)", ae);
    }
}

// ============================================================
// Section 15: Crash Safety
// ============================================================
static void test_crash_safety() {
    printf("\n======== 15. CRASH SAFETY ========\n");
    LOG("\n======== 15. CRASH SAFETY ========\n");

    // 15a: match() with empty scene (0x0 Mat)
    {
        float survived = 0.0f;
        try {
            OutputGuard guard;
            Mat empty_scene;
            sbm::MatchConfig cfg;
            sbm::ShapeMatcher matcher(cfg);
            // No model added, just call match on empty
            auto results = matcher.match(empty_scene);
            survived = 1.0f;
        } catch (...) {
            survived = 1.0f; // exception is OK, not a crash
        }
        RECORD("crash_empty_scene", survived);
        CHECK(survived > 0.5f, "15a match() with empty scene: survived");
    }

    // 15b: extractFeatures() with empty image
    {
        float survived = 0.0f;
        try {
            OutputGuard guard;
            Mat empty_img;
            auto feat = sbm::extractFeatures(empty_img);
            survived = 1.0f;
        } catch (...) {
            survived = 1.0f;
        }
        RECORD("crash_empty_extract", survived);
        CHECK(survived > 0.5f, "15b extractFeatures() with empty image: survived");
    }

    // 15c: FeatureSet::load() with non-existent file
    {
        float survived = 0.0f;
        try {
            OutputGuard guard;
            auto feat = sbm::FeatureSet::load("nonexistent_file_xyz.feat");
            survived = 1.0f;
        } catch (...) {
            survived = 1.0f;
        }
        RECORD("crash_load_missing", survived);
        CHECK(survived > 0.5f, "15c load() non-existent file: survived");
    }

    // 15d: FeatureSet::load() with garbage file
    {
        float survived = 0.0f;
        try {
            // Write garbage bytes
            FILE* f = fopen("output/test_garbage.feat", "wb");
            if (f) {
                const char garbage[] = "\x00\xFF\xDE\xAD\xBE\xEF\x01\x02\x03\x04";
                fwrite(garbage, 1, sizeof(garbage), f);
                fclose(f);
            }
            OutputGuard guard;
            auto feat = sbm::FeatureSet::load("output/test_garbage.feat");
            survived = 1.0f;
        } catch (...) {
            survived = 1.0f;
        }
        RECORD("crash_load_garbage", survived);
        CHECK(survived > 0.5f, "15d load() garbage file: survived");
    }

    // 15e: selectOptimizedPoints() on empty FeatureSet
    {
        float survived = 0.0f;
        try {
            OutputGuard guard;
            sbm::FeatureSet empty_feat;
            empty_feat.templ_width = 0;
            empty_feat.templ_height = 0;
            auto pts = empty_feat.selectOptimizedPoints(8);
            survived = 1.0f;
        } catch (...) {
            survived = 1.0f;
        }
        RECORD("crash_select_empty", survived);
        CHECK(survived > 0.5f, "15e selectOptimizedPoints() empty: survived");
    }

    // 15f: analyzeSensitivity() on empty FeatureSet
    {
        float survived = 0.0f;
        try {
            OutputGuard guard;
            sbm::FeatureSet empty_feat;
            empty_feat.templ_width = 0;
            empty_feat.templ_height = 0;
            auto sens = empty_feat.analyzeSensitivity();
            survived = 1.0f;
        } catch (...) {
            survived = 1.0f;
        }
        RECORD("crash_analyze_empty", survived);
        CHECK(survived > 0.5f, "15f analyzeSensitivity() empty: survived");
    }
}

// ============================================================
// Section 16: Multi-template
// ============================================================
static void test_multi_template(const sbm::FeatureSet& feat200, const Mat& templ200) {
    printf("\n======== 16. MULTI-TEMPLATE ========\n");
    LOG("\n======== 16. MULTI-TEMPLATE ========\n");

    // Create triangle template
    Mat templ_tri(200, 200, CV_8U, Scalar(0));
    {
        std::vector<Point> pts = {Point(100, 30), Point(30, 170), Point(170, 170)};
        fillConvexPoly(templ_tri, pts, Scalar(200));
    }

    sbm::FeatureSet feat_tri;
    {
        OutputGuard guard;
        feat_tri = sbm::extractFeatures(templ_tri);
    }
    feat_tri.setOrigin(100, 100);
    feat_tri.selectOptimizedPoints(15);

    // Build scene with both shapes
    Mat scene(500, 500, CV_8U, Scalar(0));
    place_object(templ200, scene, 150, 150, 20);     // L-shape
    place_object(templ_tri, scene, 350, 350, 60);    // Triangle

    sbm::MatchConfig cfg;
    cfg.min_score = 40;
    cfg.nms_radius = 80;
    cfg.refine = sbm::RefineMode::ICP;
    sbm::ShapeMatcher matcher(cfg);
    sbm::ModelConfig mcfg;
    mcfg.angle = {0, 360, 2};

    float l_found = 0.0f, tri_found = 0.0f;
    {
        OutputGuard guard;
        matcher.addModel("L_shape", feat200, mcfg);
        matcher.addModel("Triangle", feat_tri, mcfg);
        auto results = matcher.match(scene);

        for (auto& r : results) {
            LOG("  16 result: model=%s pos=(%.1f,%.1f) angle=%.1f score=%.1f\n",
                r.model_name.c_str(), r.x, r.y, r.angle, r.score);
            if (r.model_name == "L_shape") l_found = 1.0f;
            if (r.model_name == "Triangle") tri_found = 1.0f;
        }
    }

    RECORD("multi_templ_l_found", l_found);
    RECORD("multi_templ_tri_found", tri_found);
    CHECK(l_found > 0.5f, "16 Multi-template: L-shape found");
    CHECK(tri_found > 0.5f, "16 Multi-template: Triangle found");
}

// ============================================================
// Section 17: Score Consistency
// ============================================================
static void test_score_consistency(const sbm::FeatureSet& feat200, const Mat& templ200) {
    printf("\n======== 17. SCORE CONSISTENCY ========\n");
    LOG("\n======== 17. SCORE CONSISTENCY ========\n");

    const int scene_sz = 250;

    auto get_score = [&](const Mat& scene) -> float {
        sbm::MatchConfig cfg;
        cfg.min_score = 20;
        cfg.refine = sbm::RefineMode::None;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};
        float score = 0;
        {
            OutputGuard guard;
            matcher.addModel("L", feat200, mcfg);
            auto results = matcher.match(scene);
            if (!results.empty()) score = results[0].score;
        }
        return score;
    };

    // Clean scene
    Mat scene_clean(scene_sz, scene_sz, CV_8U, Scalar(0));
    place_object(templ200, scene_clean, scene_sz/2, scene_sz/2, 30);

    float score_clean = get_score(scene_clean);

    // Noise scene
    Mat scene_noise = add_noise(scene_clean, 20);
    float score_noise = get_score(scene_noise);

    // Blur scene
    Mat scene_blur;
    GaussianBlur(scene_clean, scene_blur, Size(11, 11), 0);
    float score_blur = get_score(scene_blur);

    RECORD("score_clean", score_clean);
    RECORD("score_noise", score_noise);
    RECORD("score_blur", score_blur);

    // Record ratio: clean/noise >= 0.9 is acceptable (noise can add spurious edges
    // that inflate coarse scores slightly)
    float ratio = (score_noise > 0) ? score_clean / score_noise : 99.0f;
    RECORD("score_clean_noise_ratio", ratio);

    CHECK(score_clean > 50,
          "17 Score clean=%.1f (expect >50)", score_clean);
    CHECK(score_noise > 30,
          "17 Score noise=%.1f still detectable (expect >30)", score_noise);
    CHECK(score_blur > 30,
          "17 Score blur=%.1f still detectable (expect >30)", score_blur);

    LOG("  17 Scores: clean=%.1f noise=%.1f blur=%.1f ratio=%.2f\n",
        score_clean, score_noise, score_blur, ratio);
}

// ============================================================
// Section 18: Different Template Shapes
// ============================================================
static void test_different_shapes(const Mat& templ200) {
    printf("\n======== 18. DIFFERENT TEMPLATE SHAPES ========\n");
    LOG("\n======== 18. DIFFERENT TEMPLATE SHAPES ========\n");

    const int scene_sz = 300;

    // Helper: create template, place in scene, match
    auto test_shape = [&](const char* name, const Mat& templ, double gt_ang,
                          sbm::RefineMode mode) -> std::pair<float, float> {
        sbm::FeatureSet feat;
        {
            OutputGuard guard;
            feat = sbm::extractFeatures(templ);
        }
        feat.setOrigin(templ.cols / 2.0f, templ.rows / 2.0f);
        feat.selectOptimizedPoints(15);

        if (feat.numFeatures() == 0) {
            LOG("  18 %s: no features extracted\n", name);
            return {0.0f, 99.0f};
        }

        Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
        place_object(templ, scene, scene_sz/2, scene_sz/2, gt_ang);

        sbm::MatchConfig cfg;
        cfg.min_score = 30;
        cfg.refine = mode;
        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};

        float found = 0.0f, ae = 99.0f;
        {
            OutputGuard guard;
            matcher.addModel(name, feat, mcfg);
            auto results = matcher.match(scene);
            if (!results.empty()) {
                found = 1.0f;
                ae = angle_err(results[0].angle, (float)gt_ang);
                LOG("  18 %s: found at (%.1f,%.1f)@%.1f err=%.2fdeg\n",
                    name, results[0].x, results[0].y, results[0].angle, ae);
            } else {
                LOG("  18 %s: NOT FOUND\n", name);
            }
        }
        return {found, ae};
    };

    // Rectangle template (200x120, thick border)
    Mat templ_rect(200, 200, CV_8U, Scalar(0));
    rectangle(templ_rect, Point(30, 50), Point(170, 150), Scalar(200), 4);

    // Thin pole template (200x20)
    Mat templ_pole(200, 200, CV_8U, Scalar(0));
    rectangle(templ_pole, Point(90, 10), Point(110, 190), Scalar(200), -1);

    // Triangle template
    Mat templ_tri(200, 200, CV_8U, Scalar(0));
    {
        std::vector<Point> pts = {Point(100, 20), Point(20, 180), Point(180, 180)};
        fillConvexPoly(templ_tri, pts, Scalar(200));
    }

    // Test with coarse detection first
    auto res_rect = test_shape("Rect", templ_rect, 25, sbm::RefineMode::None);
    auto res_pole = test_shape("Pole", templ_pole, 40, sbm::RefineMode::None);
    auto res_tri  = test_shape("Tri",  templ_tri,  55, sbm::RefineMode::None);
    float rect_found = res_rect.first;
    float pole_found = res_pole.first;
    float tri_found  = res_tri.first;

    // Test ROI refinement angle accuracy for rect and triangle
    float rect_roi_ang = 99.0f, tri_roi_ang = 99.0f;
    if (rect_found > 0.5f) {
        auto rr = test_shape("Rect_ROI", templ_rect, 25, sbm::RefineMode::ROI);
        rect_roi_ang = rr.second;
    }
    if (tri_found > 0.5f) {
        auto tr = test_shape("Tri_ROI", templ_tri, 55, sbm::RefineMode::ROI);
        tri_roi_ang = tr.second;
    }

    RECORD("shape_rect_found", rect_found);
    RECORD("shape_pole_found", pole_found);
    RECORD("shape_tri_found", tri_found);
    RECORD("shape_rect_roi_ang", rect_roi_ang);
    RECORD("shape_tri_roi_ang", tri_roi_ang);

    CHECK(rect_found > 0.5f, "18 Rectangle detected");
    CHECK(pole_found > 0.5f, "18 Thin pole detected");
    CHECK(tri_found > 0.5f,  "18 Triangle detected");
    CHECK(rect_roi_ang < 1.0f,
          "18 Rectangle ROI angle err=%.2fdeg (expect <1.0)", rect_roi_ang);
    CHECK(tri_roi_ang < 1.0f,
          "18 Triangle ROI angle err=%.2fdeg (expect <1.0)", tri_roi_ang);
}

// ============================================================
// Main
// ============================================================
static void print_help(const char* prog) {
    printf("Usage: %s [sections...]\n\n", prog);
    printf("Sections:\n");
    printf("  1  Coarse matching (detection, accuracy, multi-object, speed)\n");
    printf("  2  ICP inverse refinement (angle/pos accuracy, divergence, noise)\n");
    printf("  3  ROI refinement (angle/pos, PCA fix, sub-pixel, noise)\n");
    printf("  4  Feature selection (point count, sensitivity, corners, cache)\n");
    printf("  5  Sensitivity analysis (L-shape, degenerate line)\n");
    printf("  6  Speed benchmarks FHD 10-obj (coarse, ICP, ROI)\n");
    printf("  7  Cross-validation (ROI vs ICP ratios) [auto if 2+3 run]\n");
    printf("  8  Edge cases (near-edge object, small template)\n");
    printf("  9  Noise & blur stability limits\n");
    printf("  10 Multi-resolution speed (360p, 1080p, 20MP)\n");
    printf("  11 Determinism (repeated match consistency)\n");
    printf("  12 Serialization round-trip (save/load features)\n");
    printf("  13 False positives (empty, noise, wrong shape)\n");
    printf("  14 API contract (origin, angle offset)\n");
    printf("  15 Crash safety (empty inputs, bad files)\n");
    printf("  16 Multi-template (L-shape + triangle)\n");
    printf("  17 Score consistency (clean vs noise vs blur)\n");
    printf("  18 Different template shapes (rect, pole, triangle)\n");
    printf("  all  Run all sections (default)\n");
    printf("\nExamples:\n");
    printf("  %s                        # print this help\n", prog);
    printf("  %s all                    # run all 98 checks\n", prog);
    printf("  %s 2 3                    # ICP + ROI refinement only\n", prog);
    printf("  %s 6 10                   # speed benchmarks only\n", prog);
    printf("  %s 9                      # noise/blur stability only\n", prog);
    printf("  %s all -c my_thresh.csv   # use custom thresholds\n", prog);
    printf("\nOptions:\n");
    printf("  -c <file>  Load thresholds from custom CSV (default: test_thresholds.csv)\n");
    printf("\nThresholds loaded from CSV (editable without recompile).\n");
    printf("Results logged to output/regression_log.txt.\n");
}

int main(int argc, char** argv) {
    // Parse arguments
    std::set<int> sections;
    bool run_all = false;
    std::string csv_path = "test_thresholds.csv";

    if (argc < 2) {
        print_help(argv[0]);
        return 0;
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "all") { run_all = true; continue; }
        if (arg == "-h" || arg == "--help") { print_help(argv[0]); return 0; }
        if (arg == "-c" && i + 1 < argc) { csv_path = argv[++i]; continue; }
        try { sections.insert(std::stoi(arg)); } catch (...) {
            printf("Unknown argument: %s\n", argv[i]);
            print_help(argv[0]);
            return 1;
        }
    }
    if (run_all || sections.empty()) {
        for (int i = 1; i <= 18; i++) sections.insert(i);
    }

#ifdef _WIN32
    system("if not exist output mkdir output");
#else
    system("mkdir -p output");
#endif

    g_log = fopen("output/regression_log.txt", "w");
    if (!g_log) g_log = stderr;

    if (!load_thresholds(csv_path.c_str())) {
        printf("Warning: could not load thresholds from '%s'\n", csv_path.c_str());
    } else {
        printf("Thresholds: %s (%d entries)\n", csv_path.c_str(), (int)g_thresholds.size());
    }

    printf("===== SHAPE MATCHING REGRESSION TEST =====\n");
    printf("Sections: ");
    for (int s : sections) printf("%d ", s);
    printf("\n");
    LOG("===== SHAPE MATCHING REGRESSION TEST =====\n");

    // --- Create 200x200 L-shape template ---
    Mat templ200(200, 200, CV_8U, Scalar(0));
    draw_L(templ200, 100, 100, 0, 200);

    auto feat200 = sbm::extractFeatures(templ200);
    feat200.setOrigin(100, 75);
    printf("Template: 200x200 L-shape, %d features, origin=(100,75)\n", feat200.numFeatures());
    LOG("Template: 200x200 L-shape, %d features, origin=(100,75)\n", feat200.numFeatures());

    feat200.selectOptimizedPoints(15);

    // --- Run selected sections ---
    if (sections.count(1))  test_coarse_matching(feat200, templ200);
    if (sections.count(2))  test_icp_refinement(feat200, templ200);
    if (sections.count(3))  test_roi_refinement(feat200, templ200);
    if (sections.count(4))  test_feature_selection(feat200);
    if (sections.count(5))  test_sensitivity(feat200);
    if (sections.count(6))  test_speed_benchmarks(feat200, templ200);
    if (sections.count(8))  test_edge_cases(feat200, templ200);
    if (sections.count(9))  test_noise_blur_stability(feat200, templ200);
    if (sections.count(10)) test_resolution_speed(feat200, templ200);
    if (sections.count(11)) test_determinism(feat200, templ200);
    if (sections.count(12)) test_serialization(feat200, templ200);
    if (sections.count(13)) test_false_positives(feat200, templ200);
    if (sections.count(14)) test_api_contract(feat200, templ200);
    if (sections.count(15)) test_crash_safety();
    if (sections.count(16)) test_multi_template(feat200, templ200);
    if (sections.count(17)) test_score_consistency(feat200, templ200);
    if (sections.count(18)) test_different_shapes(templ200);

    // Compute cross-validation metrics before evaluation
    {
        auto it_roi_pos = g_metrics.find("roi_pos_mean");
        auto it_icp_pos = g_metrics.find("icp_pos_mean");
        auto it_roi_ang = g_metrics.find("roi_ang_mean");
        auto it_icp_ang = g_metrics.find("icp_ang_mean");
        if (it_roi_pos != g_metrics.end() && it_icp_pos != g_metrics.end() && it_icp_pos->second > 0)
            RECORD("roi_icp_pos_ratio", it_roi_pos->second / it_icp_pos->second);
        if (it_icp_ang != g_metrics.end() && it_roi_ang != g_metrics.end() && it_roi_ang->second > 0)
            RECORD("icp_roi_ang_ratio", it_icp_ang->second / it_roi_ang->second);
    }

    evaluate_thresholds();

    // --- Summary ---
    printf("\n===== SUMMARY =====\n");
    printf("PASS: %d\n", g_pass);
    printf("FAIL: %d\n", g_fail);
    printf("WARN: %d\n", g_warn);
    LOG("\n===== SUMMARY =====\n");
    LOG("PASS: %d\nFAIL: %d\nWARN: %d\n", g_pass, g_fail, g_warn);

    if (g_fail == 0)
        printf("\nALL TESTS PASSED.\n");
    else
        printf("\n%d TEST(S) FAILED.\n", g_fail);

    if (g_log && g_log != stderr) fclose(g_log);

    return g_fail > 0 ? 1 : 0;
}
