// 20MP 20-object benchmark under various noise levels
// Compares Coarse, ICP, and ROI accuracy and speed

#include "shape_matcher.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <sstream>
#include <algorithm>

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

static float angle_err(float a, float b) {
    float e = a - b; while (e > 180) e -= 360; while (e < -180) e += 360;
    return std::abs(e);
}

struct CoutSup {
    std::streambuf* ob; std::ostringstream sink;
    CoutSup() : ob(std::cout.rdbuf()) { std::cout.rdbuf(sink.rdbuf()); }
    ~CoutSup() { std::cout.rdbuf(ob); }
};

int main() {
    // Template
    Mat templ(200, 200, CV_8U, Scalar(0));
    draw_L(templ, 100, 100, 0, 200);
    auto feat = sbm::extractFeatures(templ);
    feat.setOrigin(100, 75);

    // 20 objects spread across 20MP (5472x3648)
    const int W = 5472, H = 3648;
    struct ObjDef { float rx, ry; double angle; };
    ObjDef defs[] = {
        {0.08f,0.10f,7},  {0.22f,0.15f,23},  {0.36f,0.08f,51},  {0.50f,0.13f,78},
        {0.64f,0.08f,102},{0.78f,0.15f,133}, {0.92f,0.08f,157}, {0.10f,0.35f,189},
        {0.24f,0.40f,212},{0.38f,0.33f,238}, {0.52f,0.40f,267}, {0.66f,0.33f,291},
        {0.80f,0.40f,319},{0.08f,0.60f,342}, {0.22f,0.65f,12},  {0.36f,0.60f,67},
        {0.50f,0.65f,112},{0.64f,0.60f,167}, {0.78f,0.65f,222}, {0.12f,0.85f,277},
    };
    int N_OBJ = 20;

    // Compute GT positions
    struct GT { float cx, cy; double angle; };
    GT gts[20];
    for (int i = 0; i < N_OBJ; i++) {
        gts[i].cx = defs[i].rx * (W - 200) + 100;
        gts[i].cy = defs[i].ry * (H - 200) + 100;
        gts[i].angle = defs[i].angle;
    }

    // Build clean scene
    Mat scene_clean(H, W, CV_8U, Scalar(30));
    for (int i = 0; i < N_OBJ; i++) {
        Mat M = getRotationMatrix2D(Point2f(100, 100), -gts[i].angle, 1.0);
        // Sub-pixel placement via warpAffine with fractional shift
        float frac_x = gts[i].cx - std::floor(gts[i].cx);
        float frac_y = gts[i].cy - std::floor(gts[i].cy);
        double* md = (double*)M.data;
        md[2] += frac_x;
        md[5] += frac_y;
        Mat rot; warpAffine(templ, rot, M, Size(templ.cols+2, templ.rows+2),
                            INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
        int ox = (int)std::floor(gts[i].cx) - 100, oy = (int)std::floor(gts[i].cy) - 100;
        for (int r = 0; r < rot.rows; r++)
            for (int c = 0; c < rot.cols; c++) {
                int sy = oy+r, sx = ox+c;
                if (sy>=0 && sy<H && sx>=0 && sx<W && rot.at<uchar>(r,c) > 0)
                    scene_clean.at<uchar>(sy,sx) = std::max(scene_clean.at<uchar>(sy,sx), rot.at<uchar>(r,c));
            }
    }

    float org_ox = 100 - 100.0f, org_oy = 75 - 100.0f;  // origin offset from center

    // Noise levels to test
    float noise_levels[] = {0, 5, 10, 15, 20, 30, 40, 50};
    int n_noise = sizeof(noise_levels) / sizeof(noise_levels[0]);

    sbm::RefineMode modes[] = {sbm::RefineMode::None, sbm::RefineMode::ICP, sbm::RefineMode::ROI};
    const char* mode_names[] = {"Coarse", "ICP", "ROI"};

    // Header
    printf("20MP (%dx%d) — 20 objects — L-shape 200x200\n\n", W, H);
    printf("%-8s", "Noise");
    for (int mi = 0; mi < 3; mi++)
        printf("  %-7s found  time    ang_m  ang_w  pos_m  pos_w", mode_names[mi]);
    printf("\n");
    for (int i = 0; i < 8 + 3*50; i++) printf("-");
    printf("\n");

    FILE* rf = fopen("output/20mp_noise_sweep.txt", "w");
    fprintf(rf, "20MP (%dx%d) — 20 objects — L-shape 200x200\n\n", W, H);
    fprintf(rf, "%-8s", "Noise");
    for (int mi = 0; mi < 3; mi++)
        fprintf(rf, "  %-7s found  time    ang_m  ang_w  pos_m  pos_w", mode_names[mi]);
    fprintf(rf, "\n");

    for (int ni = 0; ni < n_noise; ni++) {
        float ns = noise_levels[ni];

        // Add noise to clean scene
        Mat scene;
        if (ns > 0) {
            Mat noise_mat(scene_clean.size(), CV_64F);
            RNG rng(42);
            rng.fill(noise_mat, RNG::NORMAL, 0, ns);
            Mat tmp; scene_clean.convertTo(tmp, CV_64F);
            tmp += noise_mat; tmp.convertTo(scene, CV_8U);
        } else {
            scene = scene_clean.clone();
        }

        printf("n=%-6.0f", ns);
        fprintf(rf, "n=%-6.0f", ns);

        for (int mi = 0; mi < 3; mi++) {
            sbm::MatchConfig cfg;
            cfg.min_score = 30;
            cfg.nms_radius = 100;
            cfg.refine = modes[mi];

            sbm::ShapeMatcher matcher(cfg);
            sbm::ModelConfig mcfg;
            mcfg.angle = {0, 360, 2};
            { CoutSup s; matcher.addModel("L", feat, mcfg); }

            // Warm up
            { CoutSup s; matcher.match(scene); }

            // 5 runs, sort, middle 33% mean
            const int NR = 5;
            std::vector<double> times(NR);
            std::vector<sbm::MatchResult> last_results;
            for (int r = 0; r < NR; r++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                { CoutSup s; last_results = matcher.match(scene); }
                times[r] = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
            }
            std::sort(times.begin(), times.end());
            double avg_ms = (times[1] + times[2] + times[3]) / 3.0;

            // Match results to GT
            int n_matched = 0;
            float total_ae = 0, total_pe = 0, worst_ae = 0, worst_pe = 0;

            for (int gi = 0; gi < N_OBJ; gi++) {
                float best_d = 1e9f; int best_ri = -1;
                for (int ri = 0; ri < (int)last_results.size(); ri++) {
                    auto& res = last_results[ri];
                    float rad = -res.angle * (float)CV_PI / 180.0f;
                    float rcx = res.x - (std::cos(rad)*org_ox - std::sin(rad)*org_oy);
                    float rcy = res.y - (std::sin(rad)*org_ox + std::cos(rad)*org_oy);
                    float dx = rcx - gts[gi].cx, dy = rcy - gts[gi].cy;
                    float d = std::sqrt(dx*dx + dy*dy);
                    if (d < best_d) { best_d = d; best_ri = ri; }
                }
                if (best_ri >= 0 && best_d < 50) {
                    auto& res = last_results[best_ri];
                    float rad = -res.angle * (float)CV_PI / 180.0f;
                    float rcx = res.x - (std::cos(rad)*org_ox - std::sin(rad)*org_oy);
                    float rcy = res.y - (std::sin(rad)*org_ox + std::cos(rad)*org_oy);
                    // Log first noise level per-object for debugging
                    if (ni == 0 && mi == 2)  // ROI, clean
                        fprintf(rf, "  obj[%d] gt=(%.2f,%.2f)@%.0f -> (%.2f,%.2f)@%.1f err=%.2fpx\n",
                                gi, gts[gi].cx, gts[gi].cy, gts[gi].angle, rcx, rcy, res.angle, best_d);
                    float ae = angle_err(res.angle, (float)gts[gi].angle);
                    float dx = rcx - gts[gi].cx, dy = rcy - gts[gi].cy;
                    float pe = std::sqrt(dx*dx + dy*dy);
                    total_ae += ae; total_pe += pe;
                    worst_ae = std::max(worst_ae, ae);
                    worst_pe = std::max(worst_pe, pe);
                    n_matched++;
                }
            }

            float ma = n_matched > 0 ? total_ae / n_matched : -1;
            float mp = n_matched > 0 ? total_pe / n_matched : -1;

            printf("  %2d/%d %6.0fms %5.2f %5.1f %5.2f %5.1f",
                   n_matched, N_OBJ, avg_ms, ma, worst_ae, mp, worst_pe);
            fprintf(rf, "  %2d/%d %6.0fms %5.2f %5.1f %5.2f %5.1f",
                    n_matched, N_OBJ, avg_ms, ma, worst_ae, mp, worst_pe);
        }
        printf("\n");
        fprintf(rf, "\n");
    }

    if (rf) fclose(rf);
    printf("\n-> saved output/20mp_noise_sweep.txt\n");
    return 0;
}
