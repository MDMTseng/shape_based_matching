// Benchmark: full match() pipeline with AVX2 optimizations.

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

int main() {
    printf("============================================================\n");
    printf("  meiqua shape_based_matching - AVX2 benchmark\n");
    printf("============================================================\n\n");

#ifdef __AVX2__
    printf("AVX2: ENABLED (fused vpshufb LUT in computeResponseMaps)\n");
#else
    printf("AVX2: DISABLED (original MIPP)\n");
#endif
    printf("MIPP SIMD width: %d bytes\n\n", (int)mipp::N<uint8_t>());

    // Template
    const int TW = 80;
    Mat templ(TW, TW, CV_8U, Scalar(0));
    draw_L(templ, TW/2, TW/2, 0, 200);
    Mat mask = Mat::ones(TW, TW, CV_8U) * 255;

    // Create detector and add rotated templates
    line2Dup::Detector detector(128, {4, 8}, 30, 60);

    printf("Training rotation variants...\n");
    auto t0 = chrono::high_resolution_clock::now();
    for (int angle = 0; angle < 360; angle += 5) {
        Mat rot_templ, rot_mask;
        Mat M = getRotationMatrix2D(Point2f(TW/2.0f, TW/2.0f), -angle, 1.0);
        warpAffine(templ, rot_templ, M, Size(TW, TW));
        warpAffine(mask, rot_mask, M, Size(TW, TW));
        detector.addTemplate(rot_templ, "L", rot_mask);
    }
    double train_ms = chrono::duration<double, milli>(
        chrono::high_resolution_clock::now() - t0).count();
    int ntmpl = detector.numTemplates("L");
    printf("  %d templates in %.1f ms\n\n", ntmpl, train_ms);

    // Test
    struct TestSize { int w, h; const char* name; };
    TestSize sizes[] = {
        {640, 480, "VGA"},
        {1920, 1080, "FHD"},
        {6000, 5000, "30MP"},
    };

    const int RUNS = 5;

    for (auto& sz : sizes) {
        printf("--- %s (%dx%d) ---\n", sz.name, sz.w, sz.h);

        Mat scene(sz.h, sz.w, CV_8U, Scalar(50));
        draw_L(scene, sz.w/4, sz.h/3, 30, 200);
        draw_L(scene, sz.w/2, sz.h/2, 90, 200);
        draw_L(scene, 3*sz.w/4, 2*sz.h/3, 200, 200);

        // Pad to 16
        int pw = (sz.w + 15) & ~15;
        int ph = (sz.h + 15) & ~15;
        Mat padded;
        if (pw != sz.w || ph != sz.h)
            copyMakeBorder(scene, padded, 0, ph-sz.h, 0, pw-sz.w, BORDER_CONSTANT, Scalar(0));
        else
            padded = scene;

        // Warm up
        auto matches = detector.match(padded, 50);

        // Time
        double total_ms = 0;
        for (int r = 0; r < RUNS; ++r) {
            auto ta = chrono::high_resolution_clock::now();
            matches = detector.match(padded, 50);
            total_ms += chrono::duration<double, milli>(
                chrono::high_resolution_clock::now() - ta).count();
        }
        double avg = total_ms / RUNS;

        printf("  Time: %.1f ms  (%.3f ms/template)\n", avg, avg/ntmpl);
        printf("  Matches: %d\n\n", (int)matches.size());
    }

    return 0;
}
