/// @file test_lock_compare.cpp
/// @brief Compare baseline (w=1.0) vs lock-weighted across shapes and noise levels.
/// Runs each shape at multiple noise levels, reports angle and position error.

#include "shape_matcher.h"
#include "test_utils.h"
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>   // getRotationMatrix2D moved here in OpenCV 5
#endif
#include <cstdio>
#include <cmath>

using namespace cv;
using namespace sbm;

static void draw_L(Mat& img, int cx, int cy, int color) {
    for (int ly = -30; ly <= 30; ly++)
        for (int lx = -10; lx <= 10; lx++) {
            int px = cx+lx, py = cy+ly;
            if (px>=0 && px<img.cols && py>=0 && py<img.rows) img.at<uchar>(py,px) = (uchar)color;
        }
    for (int ly = 10; ly <= 30; ly++)
        for (int lx = 10; lx <= 40; lx++) {
            int px = cx+lx, py = cy+ly;
            if (px>=0 && px<img.cols && py>=0 && py<img.rows) img.at<uchar>(py,px) = (uchar)color;
        }
}

static void add_noise(Mat& img, double sigma, int seed = 42) {
    Mat noise(img.size(), CV_32F);
    RNG rng(seed);
    rng.fill(noise, RNG::NORMAL, 0, sigma);
    Mat f; img.convertTo(f, CV_32F); f += noise; f.convertTo(img, CV_8U);
}

static void place_object(const Mat& templ, Mat& scene, int cx, int cy, double angle) {
    Mat M = getRotationMatrix2D(Point2f(templ.cols/2.0f, templ.rows/2.0f), -angle, 1.0);
    M.at<double>(0,2) += cx - templ.cols/2.0;
    M.at<double>(1,2) += cy - templ.rows/2.0;
    Mat mask = Mat::ones(templ.size(), CV_8U) * 255;
    Mat warped_templ, warped_mask;
    warpAffine(templ, warped_templ, M, scene.size());
    warpAffine(mask, warped_mask, M, scene.size());
    warped_templ.copyTo(scene, warped_mask);
}

static float angle_err(float a, float b) {
    float d = std::fmod(std::abs(a - b), 360.0f);
    return std::min(d, 360.0f - d);
}

struct TestResult {
    float ang_err;
    float pos_err;
    bool found;
};

static TestResult test_shape_noise(const Mat& templ, float gt_angle, float noise_sigma,
                                    bool use_lock_weights) {
    TestResult r = {99, 99, false};
    int scene_sz = 400;
    float gt_x = scene_sz/2.0f, gt_y = scene_sz/2.0f;

    FeatureSet feat;
    { OutputGuard guard; feat = extractFeatures(templ); }
    feat.setOrigin(templ.cols/2.0f, templ.rows/2.0f);
    feat.selectOptimizedPoints(8);

    Mat scene(scene_sz, scene_sz, CV_8U, Scalar(30));
    place_object(templ, scene, (int)gt_x, (int)gt_y, gt_angle);
    if (noise_sigma > 0) add_noise(scene, noise_sigma);

    MatchConfig cfg;
    cfg.min_score = 15;
    cfg.refine = RefineMode::ROI;
    cfg.blur_kernel_size = 11;  // more aggressive blur for noise robustness
    // Lock weights are controlled by the lock_major/lock_minor in SamplePoint,
    // which are set from cached_lock_info in addModel. The solver uses them
    // if they're != 1.0. To test baseline, we'd need to not set them.
    // For now, lock weights are always active when cached_lock_info is populated.

    ShapeMatcher matcher(cfg);
    ModelConfig mcfg;
    mcfg.angle = {0, 360, 2};

    {
        OutputGuard guard;
        matcher.addModel("test", feat, mcfg);

        // If baseline mode: override lock info to all 1.0
        // (This is a hack - we'd need internal access. Instead we test both
        //  by toggling the solver weight code. For this test, lock weights
        //  are always active since the solver code uses sp.lock_major.)

        auto results = matcher.match(scene);
        if (!results.empty()) {
            r.found = true;
            r.ang_err = angle_err(results[0].angle, gt_angle);
            float dx = results[0].x - gt_x, dy = results[0].y - gt_y;
            r.pos_err = std::sqrt(dx*dx + dy*dy);
        }
    }
    return r;
}

int main() {
    printf("================================================================\n");
    printf("  Lock-Weight Extended Shape Comparison\n");
    printf("  (Current solver: lock_major weighted)\n");
    printf("================================================================\n\n");

    int TW = 200;
    Mat mask = Mat::ones(TW, TW, CV_8U) * 255;

    // Define shapes
    struct Shape { const char* name; Mat templ; };
    std::vector<Shape> shapes;

    // L-shape
    { Mat t(TW, TW, CV_8U, Scalar(0)); draw_L(t, TW/2, TW/2, 200); shapes.push_back({"L-shape", t}); }

    // Rectangle
    { Mat t(TW, TW, CV_8U, Scalar(0)); rectangle(t, Point(30,50), Point(170,150), Scalar(200), -1); shapes.push_back({"Rectangle", t}); }

    // Triangle
    { Mat t(TW, TW, CV_8U, Scalar(0));
      std::vector<Point> pts = {Point(100,20), Point(20,180), Point(180,180)};
      fillConvexPoly(t, pts, Scalar(200)); shapes.push_back({"Triangle", t}); }

    // Pentagon
    { Mat t(TW, TW, CV_8U, Scalar(0));
      std::vector<Point> pts;
      for (int i = 0; i < 5; i++) {
          float a = (float)(i * 72 - 90) * CV_PI / 180.0f;
          pts.push_back(Point(100 + (int)(70*cos(a)), 100 + (int)(70*sin(a))));
      }
      fillConvexPoly(t, pts, Scalar(200)); shapes.push_back({"Pentagon", t}); }

    // Cross
    { Mat t(TW, TW, CV_8U, Scalar(0));
      rectangle(t, Point(70,30), Point(130,170), Scalar(200), -1);
      rectangle(t, Point(30,70), Point(170,130), Scalar(200), -1);
      shapes.push_back({"Cross", t}); }

    // Arrow (pointing right)
    { Mat t(TW, TW, CV_8U, Scalar(0));
      rectangle(t, Point(30,80), Point(120,120), Scalar(200), -1);  // shaft
      std::vector<Point> pts = {Point(120,50), Point(180,100), Point(120,150)};
      fillConvexPoly(t, pts, Scalar(200));  // head
      shapes.push_back({"Arrow", t}); }

    // T-shape
    { Mat t(TW, TW, CV_8U, Scalar(0));
      rectangle(t, Point(80,30), Point(120,170), Scalar(200), -1);  // vertical
      rectangle(t, Point(30,30), Point(170,70), Scalar(200), -1);   // horizontal top
      shapes.push_back({"T-shape", t}); }

    float noise_levels[] = {0, 10, 20, 30, 40};
    float gt_angle = 37.0f;  // non-trivial angle

    // Header
    printf("  %-12s", "Shape");
    for (float n : noise_levels) printf("  n=%-3.0f ang  n=%-3.0f pos", n, n);
    printf("\n");
    printf("  %-12s", "");
    for (size_t i = 0; i < sizeof(noise_levels)/sizeof(float); i++) printf("  %-9s %-9s", "(deg)", "(px)");
    printf("\n");
    printf("  ");
    for (int i = 0; i < 100; i++) printf("-");
    printf("\n");

    for (auto& shape : shapes) {
        printf("  %-12s", shape.name);
        for (float n : noise_levels) {
            auto r = test_shape_noise(shape.templ, gt_angle, n, true);
            if (r.found)
                printf("  %7.3f   %7.3f ", r.ang_err, r.pos_err);
            else
                printf("  %7s   %7s ", "MISS", "MISS");
        }
        printf("\n");
    }

    printf("\n================================================================\n");
    return 0;
}
