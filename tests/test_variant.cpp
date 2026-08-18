// Demo: multi-variant template matching
// Same shape registered under different conditions for better detection

#include "shape_matcher.h"
#include "test_utils.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>   // getRotationMatrix2D moved here in OpenCV 5
#endif
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <chrono>

using namespace cv;

static void draw_L(Mat& img, int cx, int cy, double angle, int color, int thickness = 0) {
    double rad = angle * CV_PI / 180.0, cs = cos(rad), sn = sin(rad);
    // Vertical arm
    for (double ly = -30; ly <= 30; ly += 0.5)
        for (double lx = -10 - thickness; lx <= 10 + thickness; lx += 0.5) {
            int px = cx+(int)(lx*cs-ly*sn+0.5), py = cy+(int)(lx*sn+ly*cs+0.5);
            if (px>=0&&px<img.cols&&py>=0&&py<img.rows) img.at<uchar>(py,px)=(uchar)color;
        }
    // Horizontal arm
    for (double ly = 10 - thickness; ly <= 30 + thickness; ly += 0.5)
        for (double lx = 10; lx <= 40; lx += 0.5) {
            int px = cx+(int)(lx*cs-ly*sn+0.5), py = cy+(int)(lx*sn+ly*cs+0.5);
            if (px>=0&&px<img.cols&&py>=0&&py<img.rows) img.at<uchar>(py,px)=(uchar)color;
        }
}

// OutputGuard from test_utils.h replaces the old CoutSup

int main() {
    system("if not exist output mkdir output");

    // === Create 3 template variants of the same L-shape ===
    // Variant 1: Normal contrast (bright on dark)
    Mat templ_normal(200, 200, CV_8U, Scalar(0));
    draw_L(templ_normal, 100, 100, 0, 200);

    // Variant 2: Low contrast (dim on dark)
    Mat templ_dim(200, 200, CV_8U, Scalar(0));
    draw_L(templ_dim, 100, 100, 0, 50);

    // Variant 3: Thick edges (manufacturing variation)
    Mat templ_thick(200, 200, CV_8U, Scalar(0));
    draw_L(templ_thick, 100, 100, 0, 200, 5);

    // Extract features for each variant
    auto feat_normal = sbm::extractFeatures(templ_normal);
    feat_normal.setOrigin(100, 75);
    auto feat_dim = sbm::extractFeatures(templ_dim);
    feat_dim.setOrigin(100, 75);
    auto feat_thick = sbm::extractFeatures(templ_thick);
    feat_thick.setOrigin(100, 75);

    printf("Features: normal=%d, dim=%d, thick=%d\n",
           feat_normal.numFeatures(), feat_dim.numFeatures(), feat_thick.numFeatures());

    // === Create scene with 6 objects under different conditions ===
    Mat scene(800, 1200, CV_8U, Scalar(30));

    struct Obj { int x, y; double angle; int brightness; int thick; const char* label; };
    Obj objects[] = {
        {200, 200, 25,  200, 0, "normal"},
        {600, 200, 90,   50, 0, "very dim"},
        {1000,200, 150, 200, 5, "very thick"},
        {200, 550, 210, 100, 0, "medium"},
        {600, 550, 300,  35, 0, "ultra dim"},
        {1000,550, 45,  180, 5, "thick+rot"},
    };
    int n_obj = sizeof(objects) / sizeof(objects[0]);

    for (auto& obj : objects) {
        Mat templ_obj(200, 200, CV_8U, Scalar(0));
        draw_L(templ_obj, 100, 100, 0, obj.brightness, obj.thick);
        Mat M = getRotationMatrix2D(Point2f(100, 100), -obj.angle, 1.0);
        Mat rot;
        warpAffine(templ_obj, rot, M, templ_obj.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
        int ox = obj.x - 100, oy = obj.y - 100;
        for (int r = 0; r < rot.rows; r++)
            for (int c = 0; c < rot.cols; c++) {
                int sy = oy + r, sx = ox + c;
                if (sy >= 0 && sy < scene.rows && sx >= 0 && sx < scene.cols && rot.at<uchar>(r, c) > 0)
                    scene.at<uchar>(sy, sx) = rot.at<uchar>(r, c);
            }
    }

    // Add mild noise
    {
        Mat noise(scene.size(), CV_64F);
        RNG rng(42);
        rng.fill(noise, RNG::NORMAL, 0, 15);
        Mat tmp; scene.convertTo(tmp, CV_64F);
        tmp += noise; tmp.convertTo(scene, CV_8U);
    }

    // === Test 1: Single variant (normal only) ===
    printf("\n=== Single variant (normal only) ===\n");
    std::vector<sbm::MatchResult> results_single;
    double ms_single;
    {
        sbm::MatchConfig cfg;
        cfg.min_score = 30;
        cfg.nms_radius = 80;
        cfg.refine = sbm::RefineMode::ROI;

        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};

        OutputGuard guard;
        matcher.addModel("L", feat_normal, mcfg);
        printf(""); // force cout flush before timing
        auto t0 = std::chrono::high_resolution_clock::now();
        results_single = matcher.match(scene);
        ms_single = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
    printf("  Found: %d objects in %.1fms\n", (int)results_single.size(), ms_single);
    for (auto& r : results_single)
        printf("    (%5.1f, %5.1f) @ %5.1f  score=%.0f\n", r.x, r.y, r.angle, r.score);

    // === Test 2: Multi-variant (normal + dim + thick) ===
    printf("\n=== Multi-variant (normal + dim + thick) ===\n");
    std::vector<sbm::MatchResult> results_multi;
    double ms_multi;
    {
        sbm::MatchConfig cfg;
        cfg.min_score = 30;
        cfg.nms_radius = 80;
        cfg.refine = sbm::RefineMode::ROI;

        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};

        OutputGuard guard;
        // Register all 3 variants under the SAME model name
        matcher.addModel("L", feat_normal, mcfg);
        matcher.addModel("L", feat_dim, mcfg);
        matcher.addModel("L", feat_thick, mcfg);

        auto t0 = std::chrono::high_resolution_clock::now();
        results_multi = matcher.match(scene);
        ms_multi = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
    printf("  Found: %d objects in %.1fms\n", (int)results_multi.size(), ms_multi);
    for (auto& r : results_multi)
        printf("    (%5.1f, %5.1f) @ %5.1f  score=%.0f\n", r.x, r.y, r.angle, r.score);

    // === Draw results ===
    auto draw_results = [&](Mat& vis, const std::vector<sbm::MatchResult>& results,
                            Scalar color, const char* label, int y_off) {
        for (auto& r : results) {
            float rad = r.angle * (float)CV_PI / 180.0f;
            float ax = std::cos(rad) * 50, ay = std::sin(rad) * 50;
            cv::arrowedLine(vis, Point((int)r.x, (int)r.y),
                            Point((int)(r.x + ax), (int)(r.y + ay)),
                            color, 2, LINE_AA, 0, 0.3);
            cv::circle(vis, Point((int)r.x, (int)r.y), 5, color, -1, LINE_AA);
            char sc[32];
            snprintf(sc, sizeof(sc), "%.0f", r.score);
            cv::putText(vis, sc, Point((int)r.x + 10, (int)r.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        }
        cv::putText(vis, label, Point(10, y_off),
                    FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
    };

    // Side-by-side comparison
    Mat vis_single, vis_multi;
    cvtColor(scene, vis_single, COLOR_GRAY2BGR);
    vis_single.copyTo(vis_multi);

    // Draw GT labels
    for (auto& obj : objects) {
        cv::putText(vis_single, obj.label, Point(obj.x - 40, obj.y + 80),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
        cv::putText(vis_multi, obj.label, Point(obj.x - 40, obj.y + 80),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
    }

    draw_results(vis_single, results_single, Scalar(0, 255, 0),
                 "Single variant (normal only)", 30);
    char single_info[64];
    snprintf(single_info, sizeof(single_info), "Found: %d/%d  %.1fms",
             (int)results_single.size(), n_obj, ms_single);
    cv::putText(vis_single, single_info, Point(10, 60),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);

    draw_results(vis_multi, results_multi, Scalar(0, 255, 255),
                 "Multi-variant (normal + dim + thick)", 30);
    char multi_info[64];
    snprintf(multi_info, sizeof(multi_info), "Found: %d/%d  %.1fms",
             (int)results_multi.size(), n_obj, ms_multi);
    cv::putText(vis_multi, multi_info, Point(10, 60),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);

    // Combine side by side
    Mat combined(scene.rows, scene.cols * 2 + 10, CV_8UC3, Scalar(50, 50, 50));
    vis_single.copyTo(combined(Rect(0, 0, scene.cols, scene.rows)));
    vis_multi.copyTo(combined(Rect(scene.cols + 10, 0, scene.cols, scene.rows)));

    imwrite("output/variant_comparison.png", combined);
    printf("\n-> saved output/variant_comparison.png\n");

    printf("\nSummary:\n");
    printf("  Single variant: %d/%d found\n", (int)results_single.size(), n_obj);
    printf("  Multi-variant:  %d/%d found\n", (int)results_multi.size(), n_obj);
    printf("  Improvement: +%d objects detected\n",
           (int)results_multi.size() - (int)results_single.size());

    return 0;
}
