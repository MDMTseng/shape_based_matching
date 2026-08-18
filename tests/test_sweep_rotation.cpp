// Full 360-degree rotation sweep at 1-degree steps.
// Evaluates coarse (None), ICP, and ROI matching accuracy.
// Output: output/sweep_rotation.csv

#include "shape_matcher.h"
#include "roi_refine.h"
#include "icp_refine.h"
#include "test_utils.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>   // getRotationMatrix2D moved here in OpenCV 5
#endif
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace cv;

// ============================================================
// Helpers (same as test_regression.cpp)
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

static void place_object(const Mat& templ, Mat& scene,
                          int obj_cx, int obj_cy, double angle,
                          float sub_x = 0, float sub_y = 0) {
    float tcx = templ.cols / 2.0f, tcy = templ.rows / 2.0f;
    float dst_cx = obj_cx + sub_x, dst_cy = obj_cy + sub_y;

    Mat M_rot = getRotationMatrix2D(Point2f(tcx, tcy), -angle, 1.0);
    double* md = (double*)M_rot.data;
    float frac_x = dst_cx - std::floor(dst_cx);
    float frac_y = dst_cy - std::floor(dst_cy);
    md[2] += frac_x;
    md[5] += frac_y;
    Mat rot;
    warpAffine(templ, rot, M_rot, Size(templ.cols + 2, templ.rows + 2),
               INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

    int ox = (int)std::floor(dst_cx) - (int)tcx;
    int oy = (int)std::floor(dst_cy) - (int)tcy;
    for (int r = 0; r < rot.rows; r++)
        for (int c = 0; c < rot.cols; c++) {
            int sy = oy + r, sx = ox + c;
            if (sy >= 0 && sy < scene.rows && sx >= 0 && sx < scene.cols && rot.at<uchar>(r, c) > 0)
                scene.at<uchar>(sy, sx) = std::max(scene.at<uchar>(sy, sx), rot.at<uchar>(r, c));
        }
}

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

// ============================================================
// Main
// ============================================================

int main() {
#ifdef _WIN32
    system("if not exist output mkdir output");
#else
    system("mkdir -p output");
#endif

    printf("===== ROTATION SWEEP TEST (360 x 1 deg) =====\n\n");

    // --- Create 200x200 L-shape template ---
    Mat templ200(200, 200, CV_8U, Scalar(0));
    draw_L(templ200, 100, 100, 0, 200);

    sbm::FeatureSet feat200;
    {
        OutputGuard guard;
        feat200 = sbm::extractFeatures(templ200);
    }
    feat200.setOrigin(100, 75);
    feat200.selectOptimizedPoints(15);
    printf("Template: 200x200 L-shape, %d features, origin=(100,75)\n", feat200.numFeatures());

    const int scene_sz = 400;
    const int cx = scene_sz / 2, cy = scene_sz / 2;
    const float org_x = 100, org_y = 75;

    // Refine modes to test
    struct Mode {
        const char* name;
        sbm::RefineMode refine;
    };
    Mode modes[] = {
        {"coarse", sbm::RefineMode::None},
        {"icp",    sbm::RefineMode::ICP},
        {"roi",    sbm::RefineMode::ROI},
    };
    const int n_modes = 3;

    // Per-angle results: [angle][mode] -> {ang_err, pos_err}
    struct Result { float ang; float pos; };
    std::vector<std::vector<Result>> results(360, std::vector<Result>(n_modes));

    // Pre-build matchers (one per mode, reuse across angles)
    // Build models once, then match per angle
    printf("Running 360 angles x 3 methods...\n");

    for (int mi = 0; mi < n_modes; mi++) {
        sbm::MatchConfig cfg;
        cfg.min_score = 30;
        cfg.refine = modes[mi].refine;
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 1};

        sbm::ShapeMatcher matcher(cfg);
        {
            OutputGuard guard;
            matcher.addModel("L", feat200, mcfg);
        }

        for (int deg = 0; deg < 360; deg++) {
            float gt_ang = (float)deg;
            Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
            place_object(templ200, scene, cx, cy, gt_ang);

            float gt_x, gt_y;
            compute_gt_origin(cx, cy, gt_ang, org_x, org_y, 200, 200, gt_x, gt_y);

            float ae = 999.0f, pe = 999.0f;
            {
                OutputGuard guard;
                auto res = matcher.match(scene);
                if (!res.empty()) {
                    ae = angle_err(res[0].angle, gt_ang);
                    pe = pos_err(res[0].x, res[0].y, gt_x, gt_y);
                }
            }
            results[deg][mi] = {ae, pe};
        }
        printf("  %s: done\n", modes[mi].name);
    }

    // --- Write CSV ---
    FILE* csv = fopen("output/sweep_rotation.csv", "w");
    if (!csv) {
        printf("ERROR: cannot create output/sweep_rotation.csv\n");
        return 1;
    }
    fprintf(csv, "angle,coarse_ang,coarse_pos,icp_ang,icp_pos,roi_ang,roi_pos\n");
    for (int deg = 0; deg < 360; deg++) {
        fprintf(csv, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                deg,
                results[deg][0].ang, results[deg][0].pos,
                results[deg][1].ang, results[deg][1].pos,
                results[deg][2].ang, results[deg][2].pos);
    }
    fclose(csv);
    printf("\nCSV written: output/sweep_rotation.csv\n");

    // --- Summary ---
    printf("\n===== SUMMARY =====\n");
    printf("%-8s  %10s %10s %10s %10s\n", "Method", "MeanAng", "WorstAng", "MeanPos", "WorstPos");
    printf("%-8s  %10s %10s %10s %10s\n", "------", "-------", "--------", "-------", "--------");

    for (int mi = 0; mi < n_modes; mi++) {
        float sum_a = 0, sum_p = 0, worst_a = 0, worst_p = 0;
        int count = 0;
        for (int deg = 0; deg < 360; deg++) {
            float ae = results[deg][mi].ang;
            float pe = results[deg][mi].pos;
            if (ae < 900) { // valid detection
                sum_a += ae;
                sum_p += pe;
                worst_a = std::max(worst_a, ae);
                worst_p = std::max(worst_p, pe);
                count++;
            }
        }
        float mean_a = count > 0 ? sum_a / count : -1;
        float mean_p = count > 0 ? sum_p / count : -1;
        printf("%-8s  %9.3f%s %9.3f%s %9.3fpx %9.3fpx  (%d/360 detected)\n",
               modes[mi].name,
               mean_a, "\xC2\xB0", worst_a, "\xC2\xB0",
               mean_p, worst_p, count);
    }

    printf("\nDone.\n");
    return 0;
}
