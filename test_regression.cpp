// Comprehensive regression test for shape_based_matching.
// Validates all core algorithms haven't regressed.
// Exit code: 0 = all pass, 1 = any fail.

#include "shape_matcher.h"
#include "roi_refine.h"
#include "icp_refine.h"
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
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif

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
        // Try alternate paths
        std::string paths[] = {
            csv_path,
            std::string("../") + csv_path,
            std::string("../../") + csv_path
        };
        for (auto& p : paths) {
            f.open(p);
            if (f.is_open()) break;
        }
        if (!f.is_open()) return false;
    }
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
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
        try { t.threshold = std::stof(next_field()); } catch(...) { continue; }
        t.description = (p0 < line.size()) ? line.substr(p0) : "";
        g_thresholds.push_back(t);
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

// Place template into scene via warpAffine, object center at (obj_cx, obj_cy).
static void place_object(const Mat& templ, Mat& scene,
                          int obj_cx, int obj_cy, double angle,
                          float sub_x = 0, float sub_y = 0) {
    int tcx = templ.cols / 2, tcy = templ.rows / 2;
    Mat M = getRotationMatrix2D(Point2f((float)tcx, (float)tcy), -angle, 1.0);
    double* md = (double*)M.data;
    md[2] += sub_x;
    md[5] += sub_y;
    Mat rot;
    warpAffine(templ, rot, M, templ.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    int ox = obj_cx - tcx, oy = obj_cy - tcy;
    for (int r = 0; r < rot.rows; r++)
        for (int c = 0; c < rot.cols; c++) {
            int sy = oy + r, sx = ox + c;
            if (sy >= 0 && sy < scene.rows && sx >= 0 && sx < scene.cols && rot.at<uchar>(r, c) > 0)
                scene.at<uchar>(sy, sx) = rot.at<uchar>(r, c);
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
struct CoutSuppressor {
    std::streambuf* orig_cout;
    std::ostringstream sink;
    FILE* orig_stderr_copy;
    int orig_stderr_fd;
    int devnull_fd;

    CoutSuppressor() : orig_cout(std::cout.rdbuf()), orig_stderr_copy(nullptr),
                        orig_stderr_fd(-1), devnull_fd(-1) {
        std::cout.rdbuf(sink.rdbuf());
        // Also suppress stderr (ROI debug uses fprintf(stderr,...))
        fflush(stderr);
#ifdef _WIN32
        orig_stderr_fd = _dup(_fileno(stderr));
        devnull_fd = -1;
        FILE* nul = nullptr;
        freopen_s(&nul, "NUL", "w", stderr);
#else
        orig_stderr_fd = dup(fileno(stderr));
        devnull_fd = open("/dev/null", O_WRONLY);
        if (devnull_fd >= 0) dup2(devnull_fd, fileno(stderr));
#endif
    }
    ~CoutSuppressor() {
        std::cout.rdbuf(orig_cout);
        fflush(stderr);
#ifdef _WIN32
        if (orig_stderr_fd >= 0) {
            _dup2(orig_stderr_fd, _fileno(stderr));
            _close(orig_stderr_fd);
        }
#else
        if (orig_stderr_fd >= 0) {
            dup2(orig_stderr_fd, fileno(stderr));
            close(orig_stderr_fd);
        }
        if (devnull_fd >= 0) close(devnull_fd);
#endif
    }
};

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
                CoutSuppressor s;
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
        CHECK(worst_ang < 15.0f,
              "1a Angle accuracy: worst=%.1fdeg (expect <15)", worst_ang);
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
            CoutSuppressor s;
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
            CoutSuppressor s;
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
                CoutSuppressor s;
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
                CoutSuppressor s;
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
            CoutSuppressor s;
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
                CoutSuppressor s;
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
                    CoutSuppressor s;
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
                CoutSuppressor s;
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
                CoutSuppressor s;
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
            CoutSuppressor s;
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
        CHECK(sens.worst_angle_sens < 1.1f,
              "4b Sensitivity: worst_ang=%.2f (expect <1.1 for L-shape)", sens.worst_angle_sens);
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
        CHECK(sens.worst_angle_sens < 1.1f,
              "5a L-shape not fragile: worst_ang=%.2f (expect <1.1)", sens.worst_angle_sens);
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
            CoutSuppressor s;
            matcher.addModel("L", feat200, mcfg);
            matcher.match(scene_fhd);  // warmup

            double total_ms = 0;
            for (int i = 0; i < 5; i++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                matcher.match(scene_fhd);
                total_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
            }
            avg_ms = total_ms / 5.0;
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
                CoutSuppressor s;
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
                CoutSuppressor s;
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
            CoutSuppressor s;
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
// Main
// ============================================================
int main() {
#ifdef _WIN32
    system("if not exist output mkdir output");
#else
    system("mkdir -p output");
#endif

    g_log = fopen("output/regression_log.txt", "w");
    if (!g_log) g_log = stderr;

    load_thresholds("test_thresholds.csv");

    printf("===== SHAPE MATCHING REGRESSION TEST =====\n");
    LOG("===== SHAPE MATCHING REGRESSION TEST =====\n");

    // --- Create 200x200 L-shape template ---
    Mat templ200(200, 200, CV_8U, Scalar(0));
    draw_L(templ200, 100, 100, 0, 200);

    auto feat200 = sbm::extractFeatures(templ200);
    feat200.setOrigin(100, 75);  // user origin offset (0, -25) from center
    printf("Template: 200x200 L-shape, %d features, origin=(100,75)\n", feat200.numFeatures());
    LOG("Template: 200x200 L-shape, %d features, origin=(100,75)\n", feat200.numFeatures());

    // Pre-compute optimized points (also seeds the cache)
    feat200.selectOptimizedPoints(15);

    // --- Run all sections ---
    test_coarse_matching(feat200, templ200);
    test_icp_refinement(feat200, templ200);
    test_roi_refinement(feat200, templ200);
    test_feature_selection(feat200);
    test_sensitivity(feat200);
    test_speed_benchmarks(feat200, templ200);
    test_edge_cases(feat200, templ200);

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
