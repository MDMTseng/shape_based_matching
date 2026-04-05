/// @file bench_param_sweep.cpp
/// @brief Sweep min_score and blur_kernel_size to find optimal noise rejection.

#include "shape_matcher.h"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace sbm;
using Clock = std::chrono::high_resolution_clock;

static void draw_L(Mat& img, int cx, int cy, double angle, int color, double scale = 2.0) {
    double rad = angle * CV_PI / 180.0;
    double cs = cos(rad), sn = sin(rad);
    for (double ly = -15*scale; ly <= 15*scale; ly += 0.5)
        for (double lx = -5*scale; lx <= 5*scale; lx += 0.5) {
            int px = cx + (int)(lx*cs - ly*sn + 0.5);
            int py = cy + (int)(lx*sn + ly*cs + 0.5);
            if (px >= 0 && px < img.cols && py >= 0 && py < img.rows)
                img.at<uchar>(py, px) = (uchar)color;
        }
    for (double ly = 5*scale; ly <= 15*scale; ly += 0.5)
        for (double lx = 5*scale; lx <= 20*scale; lx += 0.5) {
            int px = cx + (int)(lx*cs - ly*sn + 0.5);
            int py = cy + (int)(lx*sn + ly*cs + 0.5);
            if (px >= 0 && px < img.cols && py >= 0 && py < img.rows)
                img.at<uchar>(py, px) = (uchar)color;
        }
}

struct ObjGT { int x, y; float angle; };

static Mat make_scene(const std::vector<ObjGT>& objects, double noise_sigma) {
    Mat scene(1080, 1920, CV_8U, Scalar(50));
    for (auto& obj : objects)
        draw_L(scene, obj.x, obj.y, obj.angle, 200);
    if (noise_sigma > 0) {
        Mat noise(scene.size(), CV_32F);
        RNG rng(42);
        rng.fill(noise, RNG::NORMAL, 0, noise_sigma);
        Mat f; scene.convertTo(f, CV_32F); f += noise; f.convertTo(scene, CV_8U);
    }
    return scene;
}

static float angle_err(float a, float b) {
    float d = std::fmod(std::abs(a - b), 360.0f);
    return std::min(d, 360.0f - d);
}

int main() {
    printf("================================================================\n");
    printf("  Parameter Sweep: min_score × blur_kernel × noise\n");
    printf("  FHD 1920x1080, 10 L-shape objects\n");
    printf("================================================================\n\n");

    const int TW = 200;
    Mat templ(TW, TW, CV_8U, Scalar(0));
    draw_L(templ, TW/2, TW/2, 0, 200);
    auto feat = extractFeatures(templ);
    feat.setOrigin(TW/2.0f, TW/2.0f);

    std::vector<ObjGT> objects = {
        {200, 200, 15}, {600, 200, 45}, {1000, 200, 80},
        {1400, 200, 120}, {1700, 200, 170},
        {300, 700, 200}, {700, 700, 250}, {1100, 700, 290},
        {1500, 700, 330}, {1800, 700, 5}
    };

    ModelConfig mcfg;
    mcfg.angle = {0, 360, 2};

    float noise_levels[] = {0, 15, 30};
    int min_scores[] = {30, 40, 50, 60, 70};
    int blur_kernels[] = {5, 7, 11, 15};

    printf("  %-6s %-6s %-6s | %6s %6s %6s %8s\n",
           "noise", "score", "blur", "found", "true", "false", "time_ms");
    printf("  ");
    for (int i = 0; i < 60; i++) printf("-");
    printf("\n");

    for (float noise : noise_levels) {
        Mat scene = make_scene(objects, noise);

        for (int blur : blur_kernels) {
            for (int score : min_scores) {
                MatchConfig cfg;
                cfg.min_score = (float)score;
                cfg.blur_kernel_size = blur;
                cfg.refine = RefineMode::None;
                ShapeMatcher matcher(cfg);
                matcher.addModel("L", feat, mcfg);

                // Warmup
                matcher.match(scene);

                // Timed run
                auto t0 = Clock::now();
                auto results = matcher.match(scene);
                double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

                // Count true positives (within 30px of a GT object)
                int true_pos = 0;
                for (auto& gt : objects) {
                    for (auto& r : results) {
                        float dx = r.x - gt.x, dy = r.y - gt.y;
                        if (dx*dx + dy*dy < 30*30 && angle_err(r.angle, gt.angle) < 10) {
                            true_pos++;
                            break;
                        }
                    }
                }
                int false_pos = (int)results.size() - true_pos;

                printf("  %-6.0f %-6d %-6d | %6d %6d %6d %8.1f\n",
                       noise, score, blur,
                       (int)results.size(), true_pos, false_pos, ms);
            }
        }
        printf("\n");
    }

    // Now test with ROI refine at the best settings
    printf("=== ROI refine speed at selected settings ===\n\n");
    struct TestCase { float noise; int score; int blur; };
    TestCase cases[] = {
        {0,  30, 7},   // current defaults
        {15, 30, 7},   // bench_full_profile case
        {15, 50, 7},   // higher score
        {15, 50, 11},  // higher score + more blur
        {30, 50, 11},  // noisy
        {30, 60, 11},  // noisy + aggressive
    };

    printf("  %-6s %-6s %-6s | %6s %6s %8s\n",
           "noise", "score", "blur", "found", "true", "time_ms");
    printf("  ");
    for (int i = 0; i < 52; i++) printf("-");
    printf("\n");

    for (auto& tc : cases) {
        Mat scene = make_scene(objects, tc.noise);

        MatchConfig cfg;
        cfg.min_score = (float)tc.score;
        cfg.blur_kernel_size = tc.blur;
        cfg.refine = RefineMode::ROI;
        ShapeMatcher matcher(cfg);
        matcher.addModel("L", feat, mcfg);

        // Warmup
        matcher.match(scene);

        // Median of 5 runs
        double times[5];
        int found = 0, true_pos = 0;
        for (int i = 0; i < 5; i++) {
            auto t0 = Clock::now();
            auto results = matcher.match(scene);
            times[i] = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            found = (int)results.size();
            if (i == 0) {
                true_pos = 0;
                for (auto& gt : objects) {
                    for (auto& r : results) {
                        float dx = r.x - gt.x, dy = r.y - gt.y;
                        if (dx*dx + dy*dy < 30*30 && angle_err(r.angle, gt.angle) < 10) {
                            true_pos++;
                            break;
                        }
                    }
                }
            }
        }
        std::sort(times, times + 5);

        printf("  %-6.0f %-6d %-6d | %6d %6d %8.1f\n",
               tc.noise, tc.score, tc.blur, found, true_pos, times[2]);
    }

    printf("\n================================================================\n");
    return 0;
}
