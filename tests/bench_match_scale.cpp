/// @file bench_match_scale.cpp
/// @brief match_scale: re-extract vs scale-in-place comparison.

#include "shape_matcher.h"
#include "line2Dup.h"
#include "sbm_log.h"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace cv;
using namespace sbm;
using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
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
    float e = std::fmod(std::abs(a - b), 360.0f);
    return std::min(e, 360.0f - e);
}

struct ObjGT { int x, y; float angle; };

struct RunResult {
    double time_ms;
    int found;
    float score, ang_m, ang_w, pos_m, pos_w;
    int missed;
};

static RunResult run_config(const Mat& scene, const std::vector<ObjGT>& objects,
                            FeatureSet& feat, ModelConfig& mcfg, MatchConfig& cfg) {
    ShapeMatcher matcher(cfg);
    matcher.addModel("L", feat, mcfg);
    matcher.match(scene); // warmup

    const int RUNS = 3;
    double times[3];
    std::vector<MatchResult> last_results;
    for (int i = 0; i < RUNS; i++) {
        auto t = Clock::now();
        last_results = matcher.match(scene);
        times[i] = ms_since(t);
    }
    std::sort(times, times + RUNS);

    RunResult r = {};
    r.time_ms = times[RUNS/2];
    r.found = 0; r.missed = 0;
    for (auto& res : last_results) r.score += res.score;
    if (!last_results.empty()) r.score /= last_results.size();

    float total_ae = 0, total_pe = 0;
    for (auto& gt : objects) {
        float best_d = 1e9f; int best_ri = -1;
        for (int ri = 0; ri < (int)last_results.size(); ri++) {
            float dx = last_results[ri].x - gt.x, dy = last_results[ri].y - gt.y;
            float d = std::sqrt(dx*dx + dy*dy);
            if (d < best_d) { best_d = d; best_ri = ri; }
        }
        if (best_ri >= 0 && best_d < 30) {
            float ae = angle_err(last_results[best_ri].angle, gt.angle);
            total_ae += ae; total_pe += best_d;
            r.ang_w = std::max(r.ang_w, ae); r.pos_w = std::max(r.pos_w, best_d);
            r.found++;
        } else r.missed++;
    }
    r.ang_m = r.found > 0 ? total_ae / r.found : -1;
    r.pos_m = r.found > 0 ? total_pe / r.found : -1;
    return r;
}

int main() {
    sbm::setLogLevel(sbm::LogLevel::Warning);

    const int TW = 200;
    Mat templ(TW, TW, CV_8U, Scalar(0));
    for (int ly = -30; ly <= 30; ly++)
        for (int lx = -10; lx <= 10; lx++) {
            int px = TW/2+lx, py = TW/2+ly;
            if (px>=0 && px<TW && py>=0 && py<TW) templ.at<uchar>(py,px) = 200;
        }
    for (int ly = 10; ly <= 30; ly++)
        for (int lx = 10; lx <= 40; lx++) {
            int px = TW/2+lx, py = TW/2+ly;
            if (px>=0 && px<TW && py>=0 && py<TW) templ.at<uchar>(py,px) = 200;
        }

    auto feat = extractFeatures(templ);
    feat.setOrigin(TW/2.0f, TW/2.0f);
    ModelConfig mcfg;
    mcfg.angle = {0, 360, 2};

    const int W = 5472, H = 3648;
    float base_angles[] = {5,15,35,55,80,110,140,170,200,230,260,290,320,350,
                           25,50,75,100,125,155,185,215,245,275,305,335,10,30,
                           65,95,120,145,165,190,210,235,255,280,310,340};

    std::vector<ObjGT> objects;
    {
        int cols = 8, rows = 5;
        float sx = W / (float)(cols + 1), sy = H / (float)(rows + 1);
        for (int i = 0; i < 40; i++) {
            int c = i % cols, r = i / cols;
            objects.push_back({(int)(sx*(c+1)), (int)(sy*(r+1)), base_angles[i]});
        }
    }

    Mat scene_clean(H, W, CV_8U, Scalar(50));
    for (auto& obj : objects)
        place_object(templ, scene_clean, obj.x, obj.y, obj.angle);

    Mat scene_noisy = scene_clean.clone();
    { Mat noise(scene_noisy.size(), CV_32F); RNG rng(42);
      rng.fill(noise, RNG::NORMAL, 0, 30);
      Mat f; scene_noisy.convertTo(f, CV_32F); f += noise; f.convertTo(scene_noisy, CV_8U); }

    printf("==========================================================================\n");
    printf("  match_scale: re-extract vs scale-in-place\n");
    printf("  20MP %dx%d, 40 objects, 200x200 L-shape\n", W, H);
    printf("==========================================================================\n\n");

    float scales[] = {1.0f, 0.7f, 0.5f};
    struct Scene { const char* name; const Mat* img; int blur_k; };
    Scene scenes[] = { {"clean", &scene_clean, 7}, {"noise=30", &scene_noisy, 11} };

    for (auto& sc : scenes) {
        for (RefineMode refine : {RefineMode::None, RefineMode::ROI}) {
            const char* ref_name = refine == RefineMode::None ? "Coarse" : "ROI";
            printf("  === %s, %s ===\n", sc.name, ref_name);
            printf("  %-7s  %-12s  %8s  %5s  %5s  %6s %6s %6s %6s\n",
                   "Scale", "Method", "Time", "Found", "Score", "AngM", "AngW", "PosM", "PosW");
            printf("  ");
            for (int i = 0; i < 82; i++) printf("-");
            printf("\n");

            for (float scale : scales) {
                // --- re-extract ---
                {
                    MatchConfig cfg;
                    cfg.min_score = 50;
                    cfg.refine = refine;
                    cfg.blur_kernel_size = sc.blur_k;
                    cfg.skip_voting = true;
                    cfg.match_scale = scale;
                    cfg.match_scale_reextract = true;
                    auto r = run_config(*sc.img, objects, feat, mcfg, cfg);
                    printf("  %-7.1f  %-12s  %7.1fms  %2d/40  %5.1f  %5.1f  %5.1f  %5.2f  %5.1f\n",
                           scale, scale < 0.99f ? "re-extract" : "full-res",
                           r.time_ms, r.found, r.score, r.ang_m, r.ang_w, r.pos_m, r.pos_w);
                }
                // --- scale-in-place (only for reduced scales) ---
                if (scale < 0.99f) {
                    MatchConfig cfg;
                    cfg.min_score = 50;
                    cfg.refine = refine;
                    cfg.blur_kernel_size = sc.blur_k;
                    cfg.skip_voting = true;
                    cfg.match_scale = scale;
                    cfg.match_scale_reextract = false;
                    auto r = run_config(*sc.img, objects, feat, mcfg, cfg);
                    printf("  %-7.1f  %-12s  %7.1fms  %2d/40  %5.1f  %5.1f  %5.1f  %5.2f  %5.1f\n",
                           scale, "scale-only",
                           r.time_ms, r.found, r.score, r.ang_m, r.ang_w, r.pos_m, r.pos_w);
                }
            }
            printf("\n");
        }
    }

    printf("==========================================================================\n");
    return 0;
}
