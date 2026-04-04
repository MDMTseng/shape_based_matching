// Noise level sweep with multiple objects.
// Evaluates coarse (None), ICP, and ROI across noise levels.
// Output: output/sweep_noise.csv

#include "shape_matcher.h"
#include "roi_refine.h"
#include "icp_refine.h"
#include "test_utils.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
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

static Mat add_noise(const Mat& img, double sigma, int seed = 42) {
    Mat result = img.clone();
    Mat noise(img.size(), CV_64F);
    RNG rng(seed);
    rng.fill(noise, RNG::NORMAL, 0, sigma);
    Mat tmp;
    result.convertTo(tmp, CV_64F);
    tmp += noise;
    tmp.convertTo(result, CV_8U);
    return result;
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

    printf("===== NOISE SWEEP TEST (10 objects, FHD) =====\n\n");

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

    // --- 10 objects at known positions/angles ---
    struct Obj { int x, y; double angle; };
    Obj objs[] = {
        {200, 150, 15},  {500, 300, 45},   {900, 200, 90},
        {1300, 400, 135},{1700, 250, 180},  {350, 700, 210},
        {750, 850, 270}, {1100, 600, 315},  {1500, 800, 30},
        {1800, 900, 60},
    };
    const int n_objs = 10;
    const float org_x = 100, org_y = 75;
    const float match_radius = 50.0f;

    // Build clean scene
    Mat scene_clean(1080, 1920, CV_8U, Scalar(30));
    for (auto& obj : objs)
        place_object(templ200, scene_clean, obj.x, obj.y, obj.angle);

    // Noise levels
    int sigmas[] = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
    const int n_sigmas = sizeof(sigmas) / sizeof(sigmas[0]);

    // Refine modes
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

    // Build matchers (one per mode, reuse across noise levels)
    std::vector<sbm::ShapeMatcher*> matchers(n_modes);
    for (int mi = 0; mi < n_modes; mi++) {
        sbm::MatchConfig cfg;
        cfg.min_score = 30;
        cfg.nms_radius = 80;
        cfg.refine = modes[mi].refine;
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};

        matchers[mi] = new sbm::ShapeMatcher(cfg);
        {
            OutputGuard guard;
            matchers[mi]->addModel("L", feat200, mcfg);
        }
    }

    // Results storage
    struct SweepResult {
        int sigma;
        int mode_idx;
        int found;
        float ang_mean, ang_worst;
        float pos_mean, pos_worst;
        double ms;
    };
    std::vector<SweepResult> all_results;

    printf("Running %d noise levels x %d methods...\n\n", n_sigmas, n_modes);

    for (int si = 0; si < n_sigmas; si++) {
        int sigma = sigmas[si];
        Mat scene = (sigma == 0) ? scene_clean.clone() : add_noise(scene_clean, sigma, 42);

        for (int mi = 0; mi < n_modes; mi++) {
            SweepResult sr;
            sr.sigma = sigma;
            sr.mode_idx = mi;
            sr.found = 0;
            sr.ang_mean = 0; sr.ang_worst = 0;
            sr.pos_mean = 0; sr.pos_worst = 0;
            sr.ms = 0;

            std::vector<sbm::MatchResult> res;
            {
                OutputGuard guard;
                // Warmup on first sigma
                if (si == 0) matchers[mi]->match(scene);

                auto t0 = std::chrono::high_resolution_clock::now();
                res = matchers[mi]->match(scene);
                sr.ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
            }

            // GT matching: use vector<int> (NOT vector<bool>!) for thread safety
            std::vector<int> result_used(res.size(), 0);
            float sum_ae = 0, sum_pe = 0;

            for (int j = 0; j < n_objs; j++) {
                float gt_x, gt_y;
                compute_gt_origin(objs[j].x, objs[j].y, objs[j].angle,
                                 org_x, org_y, 200, 200, gt_x, gt_y);

                float best_d = 1e9f;
                int best_ri = -1;
                for (int ri = 0; ri < (int)res.size(); ri++) {
                    if (result_used[ri]) continue;
                    float d = pos_err(res[ri].x, res[ri].y, gt_x, gt_y);
                    if (d < best_d) { best_d = d; best_ri = ri; }
                }

                if (best_ri >= 0 && best_d < match_radius) {
                    result_used[best_ri] = 1;
                    sr.found++;
                    float ae = angle_err(res[best_ri].angle, (float)objs[j].angle);
                    float pe = best_d;
                    sum_ae += ae;
                    sum_pe += pe;
                    sr.ang_worst = std::max(sr.ang_worst, ae);
                    sr.pos_worst = std::max(sr.pos_worst, pe);
                }
            }

            if (sr.found > 0) {
                sr.ang_mean = sum_ae / sr.found;
                sr.pos_mean = sum_pe / sr.found;
            }

            all_results.push_back(sr);
        }
    }

    // Clean up matchers
    for (int mi = 0; mi < n_modes; mi++) delete matchers[mi];

    // --- Write CSV ---
    FILE* csv = fopen("output/sweep_noise.csv", "w");
    if (!csv) {
        printf("ERROR: cannot create output/sweep_noise.csv\n");
        return 1;
    }
    fprintf(csv, "sigma,method,found,ang_mean,ang_worst,pos_mean,pos_worst,ms\n");
    for (auto& sr : all_results) {
        fprintf(csv, "%d,%s,%d,%.4f,%.4f,%.4f,%.4f,%.2f\n",
                sr.sigma, modes[sr.mode_idx].name, sr.found,
                sr.ang_mean, sr.ang_worst,
                sr.pos_mean, sr.pos_worst, sr.ms);
    }
    fclose(csv);
    printf("CSV written: output/sweep_noise.csv\n\n");

    // --- Summary table ---
    printf("===== SUMMARY =====\n");
    printf("%-6s  %-7s  %5s  %8s %8s  %8s %8s  %8s\n",
           "Sigma", "Method", "Found", "AngMean", "AngWrst", "PosMean", "PosWrst", "ms");
    printf("%-6s  %-7s  %5s  %8s %8s  %8s %8s  %8s\n",
           "-----", "------", "-----", "-------", "-------", "-------", "-------", "------");

    for (auto& sr : all_results) {
        printf("%-6d  %-7s  %3d/%d  %7.3f%s %7.3f%s  %7.3fpx %7.3fpx  %7.1f\n",
               sr.sigma, modes[sr.mode_idx].name,
               sr.found, n_objs,
               sr.ang_mean, "\xC2\xB0", sr.ang_worst, "\xC2\xB0",
               sr.pos_mean, sr.pos_worst, sr.ms);
    }

    printf("\nDone.\n");
    return 0;
}
