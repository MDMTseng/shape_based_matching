// Robustness test: noise, blur, brightness, contrast, occlusion.
// Tests where meiqua's matching breaks down under degradation.

#include "line2Dup.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cmath>
#include <random>

using namespace cv;
using namespace std;

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

static void add_gaussian_noise(Mat& img, double sigma, int seed = 42) {
    Mat noise(img.size(), CV_64F);
    RNG rng(seed);
    rng.fill(noise, RNG::NORMAL, 0, sigma);
    Mat result;
    img.convertTo(result, CV_64F);
    result += noise;
    result.convertTo(img, CV_8U);
}

static void adjust_brightness(Mat& img, int delta) {
    img.convertTo(img, -1, 1.0, delta);
}

static void adjust_contrast(Mat& img, double factor) {
    // contrast around mean
    double mean = cv::mean(img)[0];
    img.convertTo(img, -1, factor, mean * (1 - factor));
}

static void add_occlusion(Mat& img, int cx, int cy, int radius) {
    circle(img, Point(cx, cy), radius, Scalar(50), -1);
}

static Mat make_clean_scene(int w, int h, int bg = 50) {
    Mat scene(h, w, CV_8U, Scalar(bg));
    // 3 L-shapes at known positions and angles
    draw_L(scene, w/4, h/3, 0, 200);
    draw_L(scene, w/2, h/2, 90, 200);
    draw_L(scene, 3*w/4, 2*h/3, 180, 200);
    return scene;
}

static Mat pad16(const Mat& img) {
    int pw = (img.cols + 15) & ~15;
    int ph = (img.rows + 15) & ~15;
    if (pw != img.cols || ph != img.rows) {
        Mat padded;
        copyMakeBorder(img, padded, 0, ph - img.rows, 0, pw - img.cols,
                       BORDER_CONSTANT, Scalar(0));
        return padded;
    }
    return img;
}

struct TestResult {
    int matches_found;
    float best_score;
    float worst_score;
    float avg_score;
};

static TestResult run_test(line2Dup::Detector& det, const Mat& scene, float threshold) {
    Mat padded = pad16(scene);
    auto matches = det.match(padded, threshold);

    TestResult r;
    r.matches_found = (int)matches.size();
    r.best_score = 0;
    r.worst_score = 100;
    r.avg_score = 0;
    for (auto& m : matches) {
        r.best_score = std::max(r.best_score, m.similarity);
        r.worst_score = std::min(r.worst_score, m.similarity);
        r.avg_score += m.similarity;
    }
    if (r.matches_found > 0) r.avg_score /= r.matches_found;
    else r.worst_score = 0;
    return r;
}

int main() {
    printf("================================================================\n");
    printf("  meiqua shape_based_matching - ROBUSTNESS TEST\n");
    printf("================================================================\n\n");

    const int W = 640, H = 480;
    const int TW = 80;
    const float threshold = 50.0f;

    // Build template
    Mat templ(TW, TW, CV_8U, Scalar(0));
    draw_L(templ, TW/2, TW/2, 0, 200);
    Mat mask = Mat::ones(TW, TW, CV_8U) * 255;

    // Single angle detector (simpler to analyze)
    line2Dup::Detector det_single(128, {4, 8}, 30, 60);
    det_single.addTemplate(templ, "L", mask);

    // Multi-angle detector (2 deg steps = 180 templates)
    line2Dup::Detector det_rot(128, {4, 8}, 30, 60);
    for (int angle = 0; angle < 360; angle += 2) {
        Mat rot_templ, rot_mask;
        Mat M = getRotationMatrix2D(Point2f(TW/2.0f, TW/2.0f), -angle, 1.0);
        warpAffine(templ, rot_templ, M, Size(TW, TW));
        warpAffine(mask, rot_mask, M, Size(TW, TW));
        det_rot.addTemplate(rot_templ, "L", rot_mask);
    }

    printf("Templates: single=%d, rotated=%d\n\n",
           det_single.numTemplates("L"), det_rot.numTemplates("L"));

    // --- Baseline ---
    {
        Mat scene = make_clean_scene(W, H);
        auto r = run_test(det_single, scene, threshold);
        printf("=== BASELINE (clean, single angle) ===\n");
        printf("  Found: %d/3  scores: best=%.1f avg=%.1f worst=%.1f\n\n",
               r.matches_found, r.best_score, r.avg_score, r.worst_score);
    }

    // --- Gaussian Noise ---
    printf("=== GAUSSIAN NOISE (single angle, expect 3 matches) ===\n");
    printf("  %-12s  %6s  %8s  %8s  %8s\n", "Sigma", "Found", "Best", "Avg", "Worst");
    printf("  %-12s  %6s  %8s  %8s  %8s\n", "-----", "-----", "----", "---", "-----");
    for (double sigma : {5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0}) {
        Mat scene = make_clean_scene(W, H);
        add_gaussian_noise(scene, sigma);
        auto r = run_test(det_single, scene, threshold);
        printf("  sigma=%-5.0f   %4d    %6.1f    %6.1f    %6.1f%s\n",
               sigma, r.matches_found, r.best_score, r.avg_score, r.worst_score,
               r.matches_found < 3 ? "  ** MISS" : (r.matches_found > 3 ? "  ** FALSE" : ""));
    }

    // --- Gaussian Noise with rotation ---
    printf("\n=== GAUSSIAN NOISE (rotated templates, expect 3 matches) ===\n");
    printf("  %-12s  %6s  %8s  %8s  %8s\n", "Sigma", "Found", "Best", "Avg", "Worst");
    printf("  %-12s  %6s  %8s  %8s  %8s\n", "-----", "-----", "----", "---", "-----");
    for (double sigma : {5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0}) {
        Mat scene = make_clean_scene(W, H);
        add_gaussian_noise(scene, sigma);
        auto r = run_test(det_rot, scene, threshold);
        printf("  sigma=%-5.0f   %4d    %6.1f    %6.1f    %6.1f%s\n",
               sigma, r.matches_found, r.best_score, r.avg_score, r.worst_score,
               r.matches_found < 3 ? "  ** MISS" : (r.matches_found > 3 ? "  ** FALSE" : ""));
    }

    // --- Gaussian Blur ---
    printf("\n=== GAUSSIAN BLUR (single angle, expect 3 matches) ===\n");
    printf("  %-12s  %6s  %8s  %8s  %8s\n", "Kernel", "Found", "Best", "Avg", "Worst");
    printf("  %-12s  %6s  %8s  %8s  %8s\n", "------", "-----", "----", "---", "-----");
    for (int ksize : {3, 5, 7, 9, 11, 15, 21}) {
        Mat scene = make_clean_scene(W, H);
        GaussianBlur(scene, scene, Size(ksize, ksize), 0);
        auto r = run_test(det_single, scene, threshold);
        printf("  ksize=%-5d   %4d    %6.1f    %6.1f    %6.1f%s\n",
               ksize, r.matches_found, r.best_score, r.avg_score, r.worst_score,
               r.matches_found < 3 ? "  ** MISS" : (r.matches_found > 3 ? "  ** FALSE" : ""));
    }

    // --- Brightness shift ---
    printf("\n=== BRIGHTNESS SHIFT (single angle, expect 3 matches) ===\n");
    printf("  %-12s  %6s  %8s  %8s  %8s\n", "Delta", "Found", "Best", "Avg", "Worst");
    printf("  %-12s  %6s  %8s  %8s  %8s\n", "-----", "-----", "----", "---", "-----");
    for (int delta : {-80, -60, -40, -20, 0, 20, 40, 60}) {
        Mat scene = make_clean_scene(W, H);
        adjust_brightness(scene, delta);
        auto r = run_test(det_single, scene, threshold);
        printf("  delta=%-5d   %4d    %6.1f    %6.1f    %6.1f%s\n",
               delta, r.matches_found, r.best_score, r.avg_score, r.worst_score,
               r.matches_found < 3 ? "  ** MISS" : (r.matches_found > 3 ? "  ** FALSE" : ""));
    }

    // --- Contrast reduction ---
    printf("\n=== CONTRAST REDUCTION (single angle, expect 3 matches) ===\n");
    printf("  %-12s  %6s  %8s  %8s  %8s\n", "Factor", "Found", "Best", "Avg", "Worst");
    printf("  %-12s  %6s  %8s  %8s  %8s\n", "------", "-----", "----", "---", "-----");
    for (double factor : {1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}) {
        Mat scene = make_clean_scene(W, H);
        adjust_contrast(scene, factor);
        auto r = run_test(det_single, scene, threshold);
        printf("  factor=%-5.1f  %4d    %6.1f    %6.1f    %6.1f%s\n",
               factor, r.matches_found, r.best_score, r.avg_score, r.worst_score,
               r.matches_found < 3 ? "  ** MISS" : (r.matches_found > 3 ? "  ** FALSE" : ""));
    }

    // --- Occlusion ---
    printf("\n=== OCCLUSION (single angle, one object partially occluded) ===\n");
    printf("  %-12s  %6s  %8s  %8s  %8s\n", "Radius", "Found", "Best", "Avg", "Worst");
    printf("  %-12s  %6s  %8s  %8s  %8s\n", "------", "-----", "----", "---", "-----");
    for (int radius : {5, 10, 15, 20, 25, 30, 35, 40}) {
        Mat scene = make_clean_scene(W, H);
        // Occlude near first L-shape at (W/4, H/3)
        add_occlusion(scene, W/4, H/3, radius);
        auto r = run_test(det_single, scene, threshold);
        printf("  radius=%-4d   %4d    %6.1f    %6.1f    %6.1f%s\n",
               radius, r.matches_found, r.best_score, r.avg_score, r.worst_score,
               r.matches_found < 2 ? "  ** MISS" : "");
    }

    // --- Combined: noise + blur ---
    printf("\n=== COMBINED: NOISE + BLUR (single angle, expect 3) ===\n");
    printf("  %-20s  %6s  %8s  %8s  %8s\n", "Params", "Found", "Best", "Avg", "Worst");
    printf("  %-20s  %6s  %8s  %8s  %8s\n", "------", "-----", "----", "---", "-----");
    {
    struct NB { double sigma; int ksize; };
    NB nb_cases[] = {{10,3},{10,5},{20,3},{20,5},{30,5},{40,5},{50,7}};
    for (auto& nb : nb_cases) {
        Mat scene = make_clean_scene(W, H);
        add_gaussian_noise(scene, nb.sigma);
        GaussianBlur(scene, scene, Size(nb.ksize, nb.ksize), 0);
        auto r = run_test(det_single, scene, threshold);
        char label[32];
        snprintf(label, sizeof(label), "s=%.0f k=%d", nb.sigma, nb.ksize);
        printf("  %-20s  %4d    %6.1f    %6.1f    %6.1f%s\n",
               label, r.matches_found, r.best_score, r.avg_score, r.worst_score,
               r.matches_found < 3 ? "  ** MISS" : (r.matches_found > 3 ? "  ** FALSE" : ""));
    }
    }

    // --- False positive test: textured background ---
    printf("\n=== FALSE POSITIVE: RANDOM TEXTURE BACKGROUND ===\n");
    {
        // Scene with random texture but no L-shapes
        RNG rng(123);
        Mat scene(H, W, CV_8U);
        rng.fill(scene, RNG::UNIFORM, 0, 256);
        GaussianBlur(scene, scene, Size(5, 5), 0);  // structured noise
        auto r = run_test(det_single, scene, threshold);
        printf("  Random texture (no objects): found %d false matches\n", r.matches_found);
        if (r.matches_found > 0)
            printf("    best=%.1f avg=%.1f worst=%.1f\n",
                   r.best_score, r.avg_score, r.worst_score);
    }
    {
        // Scene with edges but no L-shapes (grid pattern)
        Mat scene(H, W, CV_8U, Scalar(50));
        for (int y = 0; y < H; y += 20)
            line(scene, Point(0, y), Point(W, y), Scalar(200), 2);
        for (int x = 0; x < W; x += 20)
            line(scene, Point(x, 0), Point(x, H), Scalar(200), 2);
        auto r = run_test(det_single, scene, threshold);
        printf("  Grid pattern (no objects): found %d false matches\n", r.matches_found);
        if (r.matches_found > 0)
            printf("    best=%.1f avg=%.1f worst=%.1f\n",
                   r.best_score, r.avg_score, r.worst_score);
    }

    printf("\n================================================================\n");
    printf("  DONE\n");
    printf("================================================================\n");

    return 0;
}
