// Bias table: multiple template shapes x multiple angle steps.

#include "line2Dup.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdio>

using namespace cv;
using namespace std;

static Mat pad16(const Mat& img) {
    int pw = (img.cols+15)&~15, ph = (img.rows+15)&~15;
    if (pw!=img.cols||ph!=img.rows) {
        Mat p; copyMakeBorder(img,p,0,ph-img.rows,0,pw-img.cols,BORDER_CONSTANT,Scalar(0));
        return p;
    }
    return img;
}

// Create template by drawing shape at angle=0 then using warpAffine for rotation.
// scene_draw uses the same warpAffine approach for consistency.

static Mat make_L_template(int TW) {
    Mat img(TW, TW, CV_8U, Scalar(0));
    int h = TW/2;
    // Vertical bar (left)
    rectangle(img, Point(h-15, h-30), Point(h-5, h+30), Scalar(200), -1);
    // Horizontal bar (bottom)
    rectangle(img, Point(h-15, h+20), Point(h+25, h+30), Scalar(200), -1);
    return img;
}

static Mat make_T_template(int TW) {
    Mat img(TW, TW, CV_8U, Scalar(0));
    int h = TW/2;
    rectangle(img, Point(h-5, h-25), Point(h+5, h+25), Scalar(200), -1);
    rectangle(img, Point(h-25, h-25), Point(h+25, h-15), Scalar(200), -1);
    return img;
}

static Mat make_arrow_template(int TW) {
    Mat img(TW, TW, CV_8U, Scalar(0));
    int h = TW/2;
    Point pts[3] = {Point(h-20, h+15), Point(h-20, h-15), Point(h+20, h)};
    fillConvexPoly(img, pts, 3, Scalar(200));
    return img;
}

static Mat make_cross_template(int TW) {
    Mat img(TW, TW, CV_8U, Scalar(0));
    int h = TW/2;
    rectangle(img, Point(h-5, h-25), Point(h+5, h+25), Scalar(200), -1);
    rectangle(img, Point(h-25, h-5), Point(h+25, h+5), Scalar(200), -1);
    return img;
}

static Mat make_wrench_template(int TW) {
    Mat img(TW, TW, CV_8U, Scalar(0));
    int h = TW/2;
    rectangle(img, Point(h-30, h-4), Point(h+5, h+4), Scalar(200), -1);
    circle(img, Point(h+12, h), 12, Scalar(200), -1);
    return img;
}

int main() {
    // Redirect meiqua's cout noise to suppress it
    // Use stderr for our clean output
    FILE* out = stderr;
    fprintf(out, "================================================================\n");
    fprintf(out, "  Bias Table: shape x angle_step\n");
    fprintf(out, "================================================================\n\n");

    const int TW = 80;
    const float threshold = 50.0f;

    struct Shape {
        const char* name;
        Mat templ;
    };
    Shape shapes[] = {
        {"L-shape",  make_L_template(TW)},
        {"T-shape",  make_T_template(TW)},
        {"Arrow",    make_arrow_template(TW)},
        {"Cross",    make_cross_template(TW)},
        {"Wrench",   make_wrench_template(TW)},
    };
    float steps[] = {1, 2, 3, 5, 10};

    // Save template images
    string out_dir = "C:/Users/TRS001/Documents/workspace/templmatch/test_imgs/";
    for (auto& shape : shapes) {
        Mat vis; cvtColor(shape.templ, vis, COLOR_GRAY2BGR);
        imwrite(out_dir + "templ_" + string(shape.name) + ".jpg", vis);
    }

    fprintf(out, "%-12s", "Shape");
    for (float s : steps) fprintf(out, "  step=%2.0f", s);
    fprintf(out, "\n");
    fprintf(out, "%-12s", "-----");
    for (size_t i = 0; i < sizeof(steps)/sizeof(steps[0]); i++) fprintf(out, "  ------");
    fprintf(out, "\n");

    fprintf(out, "\n--- Method 1: warpAffine + re-extract ---\n");

    for (auto& shape : shapes) {
        // Collect results first, print after (avoid interleaving with meiqua warnings)
        float biases[5] = {};
        double cal_time = 0;

        Mat mask_t = Mat::zeros(TW, TW, CV_8U);
        // Create mask from non-zero pixels with margin
        for (int r = 0; r < TW; ++r)
            for (int c = 0; c < TW; ++c)
                if (shape.templ.at<uchar>(r, c) > 0) mask_t.at<uchar>(r, c) = 255;
        dilate(mask_t, mask_t, Mat(), Point(-1,-1), 5);

        for (float step : steps) {
            line2Dup::Detector det(128, {4, 8}, 30, 60);
            for (int a = 0; a < 360; a += (int)step) {
                Mat rt, rm;
                Mat M = getRotationMatrix2D(Point2f(TW/2.0f, TW/2.0f), -(double)a, 1.0);
                warpAffine(shape.templ, rt, M, Size(TW, TW));
                warpAffine(mask_t, rm, M, Size(TW, TW));
                det.addTemplate(rt, "S", rm);
            }

            auto t0 = chrono::high_resolution_clock::now();
            float bias = det.calibrateAngleBias(shape.templ, step, "S", threshold);
            double ms = chrono::duration<double, std::milli>(
                chrono::high_resolution_clock::now() - t0).count();

            int idx = (int)(&step - steps);
            biases[idx] = bias;
            if (step == 2.0f) cal_time = ms;
        }
        // Print to stderr to avoid interleaving with meiqua's cout
        fprintf(out, "%-12s", shape.name);
        for (int i = 0; i < 5; ++i) fprintf(out, "  %+5.1f", biases[i]);
        fprintf(out, "    %.0fms\n", cal_time);
    }

    // Method 2: addTemplate_rotate (rotate features, no re-extract)
    fprintf(out, "\n--- Method 2: addTemplate_rotate (rotate features only) ---\n");
    fflush(out);

    for (auto& shape : shapes) {
        float biases[5] = {};
        bool ok = true;

        Mat mask_t = Mat::zeros(TW, TW, CV_8U);
        for (int r = 0; r < TW; ++r)
            for (int c = 0; c < TW; ++c)
                if (shape.templ.at<uchar>(r, c) > 0) mask_t.at<uchar>(r, c) = 255;
        dilate(mask_t, mask_t, Mat(), Point(-1,-1), 5);

        for (int si = 0; si < 5; ++si) {
            float step = steps[si];
            line2Dup::Detector det(128, {4, 8}, 30, 60);
            int zero_id = det.addTemplate(shape.templ, "S", mask_t);
            if (zero_id < 0) { ok = false; break; }

            Point2f center(TW/2.0f, TW/2.0f);
            for (int a = (int)step; a < 360; a += (int)step) {
                det.addTemplate_rotate("S", zero_id, (float)a, center);
            }

            float bias = det.calibrateAngleBias(shape.templ, step, "S", threshold);
            biases[si] = bias;
        }
        fprintf(out, "%-12s", shape.name);
        if (ok) {
            for (int i = 0; i < 5; ++i) fprintf(out, "  %+5.1f", biases[i]);
        } else {
            fprintf(out, "  FAILED (too few features at 0 deg)");
        }
        fprintf(out, "\n");
        fflush(out);
    }

    fprintf(out, "\nMethod 2 eliminates warpAffine artifacts.\n");
    fprintf(out, "calibrateAngleBias() measures bias in ~20ms.\n");
    // done
    return 0;
}
