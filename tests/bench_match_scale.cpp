/// @file bench_match_scale.cpp
/// @brief Per-stage speed profile: s=0.25 T={2,4} vs s=0.5 T={4,8}

#include "shape_matcher.h"
#include "line2Dup.h"
#include "sbm_log.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace cv;
using namespace sbm;
using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static float angle_err(float a, float b) {
    float e = std::fmod(std::abs(a - b), 360.0f);
    return std::min(e, 360.0f - e);
}

struct ObjGT { float x, y; float angle; };

static void place_object(const Mat& templ, Mat& scene, float cx, float cy, double angle) {
    Mat M = getRotationMatrix2D(Point2f(templ.cols/2.0f, templ.rows/2.0f), -angle, 1.0);
    M.at<double>(0,2) += cx - templ.cols/2.0;
    M.at<double>(1,2) += cy - templ.rows/2.0;
    Mat warped;
    warpAffine(templ, warped, M, scene.size(), INTER_LINEAR, BORDER_TRANSPARENT);
    for (int r = 0; r < warped.rows; r++)
        for (int c = 0; c < warped.cols; c++)
            if (warped.at<uchar>(r,c) > 0)
                scene.at<uchar>(r,c) = warped.at<uchar>(r,c);
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    sbm::setLogLevel(sbm::LogLevel::Warning);

    const char* img_path = "C:/Users/TRS001/Documents/workspace/claudePrj/cludefiles/web-terminal-hub/uploads/img_1775409786258_8c045799.png";
    if (argc > 1) img_path = argv[1];

    Mat templ_raw = imread(img_path, IMREAD_GRAYSCALE);
    if (templ_raw.empty()) { printf("ERROR: cannot load %s\n", img_path); return 1; }

    Mat binary; threshold(templ_raw, binary, 30, 255, THRESH_BINARY);
    std::vector<Point> pts; findNonZero(binary, pts);
    Rect bbox = boundingRect(pts);
    int pad = 5;
    bbox.x = std::max(0, bbox.x-pad); bbox.y = std::max(0, bbox.y-pad);
    bbox.width = std::min(templ_raw.cols-bbox.x, bbox.width+2*pad);
    bbox.height = std::min(templ_raw.rows-bbox.y, bbox.height+2*pad);
    Mat templ;
    float sc200 = 200.0f / std::max(bbox.width, bbox.height);
    resize(templ_raw(bbox), templ, Size(), sc200, sc200);

    printf("================================================================\n");
    printf("  Per-stage profile: s=0.25 T={2,4} vs s=0.5 T={4,8}\n");
    printf("  Template: %dx%d, 20MP scene, 24 objects\n", templ.cols, templ.rows);
    printf("================================================================\n\n");

    const int W = 5472, H = 3648;
    float test_angles[] = {5,23,47,78,112,145,178,210,243,275,308,340,
                           15,38,62,95,128,160,195,228,258,290,325,355};
    int N_OBJ = 24;

    std::vector<ObjGT> objects;
    { int cols=6, rows=4; float sx=W/(float)(cols+1), sy=H/(float)(rows+1);
      for (int i=0; i<N_OBJ; i++) objects.push_back({sx*(i%cols+1), sy*(i/cols+1), test_angles[i]}); }

    Mat scene_full(H, W, CV_8U, Scalar(0));
    for (auto& obj : objects) place_object(templ, scene_full, obj.x, obj.y, obj.angle);

    struct Config {
        const char* name;
        float scale;
        std::vector<int> T;
    };
    Config configs[] = {
        {"s=1.0  T={4,8}",  1.0f, {4, 8}},
        {"s=1.0  T={8,16}", 1.0f, {8, 16}},
        {"s=0.5  T={4,8}",  0.5f, {4, 8}},
        {"s=0.3  T={4,8}",  0.3f, {4, 8}},
    };

    for (auto& c : configs) {
        printf("  === %s ===\n", c.name);

        // Resize scene
        auto t0 = Clock::now();
        Mat match_scene;
        resize(scene_full, match_scene, Size((int)(W*c.scale), (int)(H*c.scale)));
        double t_resize = ms_since(t0);
        printf("    Scene: %dx%d\n", match_scene.cols, match_scene.rows);
        printf("    resize:              %7.2f ms\n", t_resize);

        // Use LineMOD profiling
        line2Dup::enableProfiling(true);
        sbm::setLogLevel(sbm::LogLevel::Debug);
        sbm::setLogFile(stdout);

        auto feat = extractFeatures(templ, Mat(), 128, c.T);
        feat.setOrigin(templ.cols/2.0f, templ.rows/2.0f);
        ModelConfig mcfg;
        mcfg.angle = {0, 360, 1};
        MatchConfig cfg;
        cfg.min_score = 50;
        cfg.refine = RefineMode::ROI;
        cfg.skip_voting = true;
        cfg.match_scale = c.scale;
        cfg.T_levels = c.T;

        ShapeMatcher matcher(cfg);
        matcher.addModel("part", feat, mcfg);

        // Warmup
        line2Dup::resetProfiling();
        matcher.match(scene_full);

        // Profiled run
        line2Dup::resetProfiling();
        auto t_total_start = Clock::now();
        auto results = matcher.match(scene_full);
        double t_total = ms_since(t_total_start);
        printf("    --- LineMOD internal ---\n");
        line2Dup::printProfiling();
        printf("    --- End-to-end ---\n");
        printf("    Total match():       %7.2f ms\n", t_total);
        printf("    Found: %d/%d\n", (int)results.size(), N_OBJ);

        // Quick accuracy
        int n_matched = 0;
        for (auto& gt : objects) {
            for (auto& r : results) {
                float dx = r.x - gt.x, dy = r.y - gt.y;
                if (std::sqrt(dx*dx+dy*dy) < 40) { n_matched++; break; }
            }
        }
        printf("    GT matched: %d/%d\n\n", n_matched, N_OBJ);

        line2Dup::enableProfiling(false);
        sbm::setLogFile(nullptr);
        sbm::setLogLevel(sbm::LogLevel::Warning);
    }

    printf("================================================================\n");
    return 0;
}
