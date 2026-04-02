// Per-stage profiling under clean and noisy conditions.

#include "line2Dup.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>

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

static void add_noise(Mat& img, double sigma) {
    Mat noise(img.size(), CV_64F);
    RNG rng(42);
    rng.fill(noise, RNG::NORMAL, 0, sigma);
    Mat result;
    img.convertTo(result, CV_64F);
    result += noise;
    result.convertTo(img, CV_8U);
}

static Mat make_scene(int w, int h, double noise_sigma) {
    Mat scene(h, w, CV_8U, Scalar(50));
    draw_L(scene, w/4, h/3, 30, 200);
    draw_L(scene, w/2, h/2, 90, 200);
    draw_L(scene, 3*w/4, 2*h/3, 200, 200);
    if (noise_sigma > 0)
        add_noise(scene, noise_sigma);
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

int main() {
    printf("================================================================\n");
    printf("  Per-stage profiling: clean vs noisy\n");
    printf("================================================================\n\n");

    const int TW = 80;
    Mat templ(TW, TW, CV_8U, Scalar(0));
    draw_L(templ, TW/2, TW/2, 0, 200);
    Mat mask_t = Mat::ones(TW, TW, CV_8U) * 255;

    line2Dup::Detector det(128, {4, 8}, 30, 60);
    for (int angle = 0; angle < 360; angle += 5) {
        Mat rot_templ, rot_mask;
        Mat M = getRotationMatrix2D(Point2f(TW/2.0f, TW/2.0f), -angle, 1.0);
        warpAffine(templ, rot_templ, M, Size(TW, TW));
        warpAffine(mask_t, rot_mask, M, Size(TW, TW));
        det.addTemplate(rot_templ, "L", rot_mask);
    }
    printf("Templates: %d\n\n", det.numTemplates("L"));

    struct TestCase { int w, h; double sigma; const char* name; };
    TestCase cases[] = {
        {640, 480, 0,   "VGA clean"},
        {640, 480, 30,  "VGA sigma=30"},
        {640, 480, 50,  "VGA sigma=50"},
        {640, 480, 80,  "VGA sigma=80"},
        {1920, 1080, 0,  "FHD clean"},
        {1920, 1080, 30, "FHD sigma=30"},
        {1920, 1080, 50, "FHD sigma=50"},
    };

    line2Dup::enableProfiling(true);

    for (auto& tc : cases) {
        printf("--- %s (%dx%d) ---\n", tc.name, tc.w, tc.h);

        Mat scene = make_scene(tc.w, tc.h, tc.sigma);
        Mat padded = pad16(scene);

        // Warmup (without profiling)
        line2Dup::enableProfiling(false);
        auto matches = det.match(padded, 50);
        line2Dup::enableProfiling(true);

        // Profile run
        line2Dup::resetProfiling();
        matches = det.match(padded, 50);

        printf("  Matches: %d\n", (int)matches.size());
        line2Dup::printProfiling();
        printf("\n");
    }

    printf("================================================================\n");
    return 0;
}
