// Test the new sbm::ShapeMatcher API.

#include "shape_matcher.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdio>

using namespace cv;

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

static void draw_T(Mat& img, int cx, int cy, double angle, int color) {
    double rad = angle * CV_PI / 180.0, cs = cos(rad), sn = sin(rad);
    for (double ly = -25; ly <= 25; ly += 0.5)
        for (double lx = -5; lx <= 5; lx += 0.5) {
            int px = cx+(int)(lx*cs-ly*sn+0.5), py = cy+(int)(lx*sn+ly*cs+0.5);
            if (px>=0&&px<img.cols&&py>=0&&py<img.rows) img.at<uchar>(py,px)=(uchar)color;
        }
    for (double ly = -25; ly <= -15; ly += 0.5)
        for (double lx = -25; lx <= 25; lx += 0.5) {
            int px = cx+(int)(lx*cs-ly*sn+0.5), py = cy+(int)(lx*sn+ly*cs+0.5);
            if (px>=0&&px<img.cols&&py>=0&&py<img.rows) img.at<uchar>(py,px)=(uchar)color;
        }
}

int main() {
    printf("=== ShapeMatcher API Test ===\n\n");
    using Clock = std::chrono::high_resolution_clock;

    const int TW = 80;
    std::string out_dir = "C:/Users/TRS001/Documents/workspace/templmatch/test_imgs/";

    // === OFFLINE: Extract + save features ===
    printf("--- OFFLINE: Feature extraction ---\n");

    // Template A: L-shape
    Mat templ_L(TW, TW, CV_8U, Scalar(0));
    draw_L(templ_L, TW/2, TW/2, 0, 200);
    auto feat_L = sbm::extractFeatures(templ_L);
    feat_L.setOrigin(TW/2, TW/2);
    feat_L.setAngleOffset(0);
    feat_L.save(out_dir + "feat_L.sbm");
    printf("  L-shape: %d features, saved\n", feat_L.numFeatures());

    // Template B: T-shape with custom origin at the top of the T
    Mat templ_T(TW, TW, CV_8U, Scalar(0));
    draw_T(templ_T, TW/2, TW/2, 0, 200);
    auto feat_T = sbm::extractFeatures(templ_T);
    feat_T.setOrigin(TW/2, TW/2 - 20);  // top of the T
    feat_T.setAngleOffset(90);            // user says 0 deg = pointing up
    feat_T.save(out_dir + "feat_T.sbm");
    printf("  T-shape: %d features, origin at top, angle_offset=90\n", feat_T.numFeatures());

    // === ONLINE: Load + match ===
    printf("\n--- ONLINE: Load + register + match ---\n");

    // Load from files
    auto loaded_L = sbm::FeatureSet::load(out_dir + "feat_L.sbm");
    auto loaded_T = sbm::FeatureSet::load(out_dir + "feat_T.sbm");
    printf("  Loaded L: %d features, T: %d features\n",
           loaded_L.numFeatures(), loaded_T.numFeatures());

    // Create matcher
    sbm::MatchConfig match_cfg;
    match_cfg.min_score = 50;
    match_cfg.nms_radius = 60;
    match_cfg.refine = sbm::RefineMode::ICP;

    sbm::ShapeMatcher matcher(match_cfg);

    auto t0 = Clock::now();
    sbm::ModelConfig mcfg_L;
    mcfg_L.angle = {0, 360, 2};
    int n_L = matcher.addModel("L-shape", loaded_L, mcfg_L);

    sbm::ModelConfig mcfg_T;
    mcfg_T.angle = {0, 360, 2};
    int n_T = matcher.addModel("T-shape", loaded_T, mcfg_T);
    double train_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    printf("  Registered: L=%d variants, T=%d variants (%.1fms)\n", n_L, n_T, train_ms);
    printf("  Total templates: %d\n", matcher.numTemplates());

    // Create scene with mixed objects
    const int W = 1920, H = 1080;
    Mat scene(H, W, CV_8U, Scalar(50));
    struct Obj { const char* type; int x, y; float angle; };
    Obj objects[] = {
        {"L", W/6,   H/4,    30},
        {"L", W/2,   H/4,   150},
        {"T", 5*W/6, H/4,    45},
        {"L", W/4,   H/2,   270},
        {"T", W/2,   H/2,     0},
        {"T", 3*W/4, H/2,   200},
        {"L", W/6,   3*H/4,  90},
        {"T", W/2,   3*H/4, 310},
        {"L", 5*W/6, 3*H/4, 180},
    };
    for (auto& obj : objects) {
        if (obj.type[0] == 'L') draw_L(scene, obj.x, obj.y, obj.angle, 200);
        else draw_T(scene, obj.x, obj.y, obj.angle, 200);
    }

    // Match
    t0 = Clock::now();
    auto results = matcher.match(scene);
    double match_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    printf("\n--- Results (%.1fms) ---\n", match_ms);
    printf("  %-10s  %6s %6s  %7s  %5s  %5s  %s\n",
           "Model", "X", "Y", "Angle", "Scale", "Score", "Flip");
    for (auto& r : results) {
        printf("  %-10s  %6.1f %6.1f  %7.1f  %5.2f  %5.1f  %s\n",
               r.model_name.c_str(), r.x, r.y, r.angle, r.scale, r.score,
               r.flipped ? "Y" : "");
    }

    // Draw results
    Mat vis;
    cvtColor(scene, vis, COLOR_GRAY2BGR);
    for (auto& r : results) {
        Scalar color = (r.model_name == "L-shape") ? Scalar(0,255,0) : Scalar(0,255,255);
        circle(vis, Point((int)r.x, (int)r.y), 4, color, -1);
        double rad = r.angle * CV_PI / 180.0;
        arrowedLine(vis, Point((int)r.x, (int)r.y),
                    Point((int)(r.x + 35*cos(rad)), (int)(r.y + 35*sin(rad))),
                    color, 2, LINE_AA, 0, 0.3);
        char buf[64];
        snprintf(buf, sizeof(buf), "%s %.0f", r.model_name.c_str(), r.angle);
        putText(vis, buf, Point((int)r.x+5, (int)r.y-10),
                FONT_HERSHEY_SIMPLEX, 0.35, color, 1);
    }
    putText(vis, ("Matches: " + std::to_string(results.size()) +
                  "  Time: " + std::to_string((int)match_ms) + "ms").c_str(),
            Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,255,255), 2);
    imwrite(out_dir + "api_test_result.jpg", vis);
    printf("\n  Saved: %sapi_test_result.jpg\n", out_dir.c_str());

    return 0;
}
