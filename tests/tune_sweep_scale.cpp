// tune_sweep_scale.cpp — SCALE-robustness sweep (the scale analog of tune_sweep).
//
// tune_sweep only perturbs off-grid ANGLE as ground truth. This one perturbs
// BOTH off-grid SCALE and off-grid angle, so it measures how robust a config is
// when the object appears at a SIZE that falls between the matcher's scale
// variants — the missing measurement axis for "robustness under scaling".
//
// Ground truth = "alter template": resize the template to a known (off-grid)
// scale, rotate to a known (off-grid) angle, place at a known center, add
// Gaussian noise. The matcher covers the scale range via ModelConfig.scale
// variants at a swept step; angle coverage is fixed so the SCALE axis is
// isolated.
//
// Objective = robust detection: rank (detection rate, then WORST-CASE score
// across cases, then speed). min_score is a free post-filter (proven in
// tune_sweep), so we run the matcher once per (num_features x scale_step) and
// evaluate every threshold by filtering the recorded correct-match scores.
//
// Scale variants are ANALYTIC (coordinate-scaled from one base extraction —
// addModel(FeatureSet)), the same mechanism rotation variants use. So the loss
// at a coarse scale_step is POSITIONAL (an off-grid size = features land a few %
// off), distinct from rotation's ANGULAR quantization loss. This harness makes
// that cliff visible and is the baseline for a selectScaleStable A/B.
//
//   cmake --build build --target tune_sweep_scale
//   ./tune_sweep_scale            # built-in thin star (index 4)
//   ./tune_sweep_scale 3          # built-in shape 0..4
//   ./tune_sweep_scale my.png     # your own grayscale template

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

#if defined(__AVX2__)
#  define SBM_SIMD "AVX2"
#elif defined(__aarch64__)
#  define SBM_SIMD "NEON"
#else
#  define SBM_SIMD "scalar"
#endif

// ---- built-in shapes (0..4); 4 = thin star (scale/tuning matters most) -------
static cv::Mat builtin(int which) {
    cv::Mat m(160, 160, CV_8U, cv::Scalar(40));
    const int c = 80;
    switch (which) {
        case 0: cv::rectangle(m, {30,30},{130,130},230,3);
                cv::line(m,{30,30},{130,130},180,2); cv::line(m,{130,30},{30,130},180,2);
                cv::rectangle(m,{55,55},{105,105},120,2); break;
        case 1: cv::circle(m,{c,c},48,220,3); cv::circle(m,{c,c},26,160,2);
                cv::line(m,{c,25},{c,135},120,2); cv::line(m,{25,c},{135,c},120,2); break;
        case 2: { std::vector<cv::Point> t{{c,25},{30,130},{130,130}};
                  cv::polylines(m,t,true,225,3); } break;
        case 3: cv::rectangle(m,{35,35},{70,130},220,cv::FILLED);
                cv::rectangle(m,{35,95},{130,130},220,cv::FILLED); break;
        default: { std::vector<cv::Point> st;   // thin wireframe star
                  for (int i=0;i<10;i++){double a=CV_PI/2+i*CV_PI/5;double r=(i&1)?24:54;
                      st.push_back({(int)(c+r*cos(a)),(int)(c-r*sin(a))});}
                  cv::polylines(m,st,true,225,3); }
    }
    return m;
}

struct TestCase { cv::Mat scene; float cx, cy, scale, angle; };

// Alter the template into a scene: resize by `scl`, rotate by `angleDeg`, place
// centered at (px,py), add noise. Center is scale-invariant (rotation+resize
// about the template center), so ground-truth position is exactly (px,py).
static TestCase make_case(const cv::Mat& tmpl, float scl, float angleDeg,
                          double sigma, int W, int H) {
    cv::Mat scaled;
    cv::resize(tmpl, scaled, cv::Size(), scl, scl,
               scl < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR);
    cv::Mat rot;
    cv::Point2f ctr(scaled.cols/2.f, scaled.rows/2.f);
    cv::Mat R = cv::getRotationMatrix2D(ctr, angleDeg, 1.0);
    cv::warpAffine(scaled, rot, R, scaled.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(40));
    cv::Mat scene(H, W, CV_8U, cv::Scalar(40));
    int px = (W - rot.cols) / 2, py = (H - rot.rows) / 2;
    rot.copyTo(scene(cv::Rect(px, py, rot.cols, rot.rows)));
    cv::Mat noise(H, W, CV_8U); cv::randn(noise, 0, sigma);
    cv::add(scene, noise, scene);
    return {scene, px + rot.cols/2.f, py + rot.rows/2.f, scl, angleDeg};
}

struct Combo { int num_features; float scale_step; float min_score; };
struct Result { Combo c; float detect_rate, worst, mean; double ms; int templates; };

int main(int argc, char** argv) {
    sbm::setLogLevel(sbm::LogLevel::Warning);

    cv::Mat tmpl;
    std::string arg = argc > 1 ? argv[1] : "";
    if (!arg.empty() && arg.find_first_not_of("0123456789") != std::string::npos)
        tmpl = cv::imread(arg, cv::IMREAD_GRAYSCALE);
    if (tmpl.empty()) tmpl = builtin(arg.empty() ? 4 : std::atoi(arg.c_str()));

    const int W = 1280, H = 960;
    const double sigma = 12.0;
    const float pos_tol = 12.0f;   // px

    // Scale coverage the matcher will tile, and the OFF-GRID scales we test at.
    // Test scales are deliberately between typical variant grid points so no
    // scale_step lands exactly on them (fair across the sweep).
    const float SMIN = 0.6f, SMAX = 1.4f;
    const float test_scales[] = {0.63f, 0.71f, 0.89f, 1.07f, 1.23f, 1.37f};
    const float test_angles[] = {17, 88, 163, 251, 320};   // off-grid angles

    // Pre-generate the shared test set (scale x angle outer product).
    std::vector<TestCase> tests;
    for (float s : test_scales)
        for (float a : test_angles)
            tests.push_back(make_case(tmpl, s, a, sigma, W, H));
    const int n = (int)tests.size();

    std::printf("tune_sweep_scale | %s | template %dx%d | %d cases "
                "(%zu scales x %zu angles) @ %dx%d, sigma=%.0f\n",
                SBM_SIMD, tmpl.cols, tmpl.rows, n,
                sizeof(test_scales)/sizeof(float), sizeof(test_angles)/sizeof(float),
                W, H, sigma);
    std::printf("scale coverage [%.2f, %.2f], angle {0,360,4} fixed; "
                "objective: robust detection (detect -> worst -> speed)\n\n", SMIN, SMAX);

    // Sweep grids. scale_step is the primary lever (variant granularity = the
    // scale analog of angle step). angle step fixed at 4 to isolate scale.
    const int   NF[]  = {63, 128};
    const float SS[]  = {0.40f, 0.20f, 0.10f, 0.05f};
    const float MIN[] = {30, 35, 40, 45, 50, 55, 60, 65};
    const float MIN_LOW = MIN[0];

    using Clk = std::chrono::high_resolution_clock;
    std::vector<Result> results;
    int matcher_runs = 0;

    // Per-scale detection breakdown for the finest config (where does it fail?).
    std::vector<float> best_by_scale;  // filled during the finest run

    for (int nf : NF)
    for (float ss : SS) {
        sbm::MatchConfig cfg;
        cfg.min_score = MIN_LOW; cfg.refine = sbm::RefineMode::None;
        cfg.skip_voting = true; cfg.match_scale = 1.0f;
        auto matcher = std::make_unique<sbm::ShapeMatcher>(cfg);

        sbm::FeatureSet fs = sbm::extractFeatures(tmpl, cv::Mat(), nf);
        sbm::ModelConfig mc;
        mc.angle = {0, 360, 4};
        mc.scale = {SMIN, SMAX, ss};
        int templates = matcher->addModel("m", fs, mc);
        ++matcher_runs;

        std::vector<float> raw(n, -1.f);
        double tot = 0;
        for (int i = 0; i < n; ++i) {
            auto t0 = Clk::now();
            auto rs = matcher->match(tests[i].scene);
            tot += std::chrono::duration<double,std::milli>(Clk::now()-t0).count();
            for (auto& r : rs)
                if (std::abs(r.x-tests[i].cx)<pos_tol && std::abs(r.y-tests[i].cy)<pos_tol)
                    raw[i] = std::max(raw[i], r.score);
        }
        double mspf = tot / n;

        // Capture the per-scale detect for the finest+richest config.
        if (nf == NF[(sizeof(NF)/sizeof(int))-1] && ss == SS[(sizeof(SS)/sizeof(float))-1])
            best_by_scale = raw;

        for (float mins : MIN) {
            int det = 0; float worst = 1e9f, sum = 0;
            for (float s : raw) if (s >= mins) { det++; worst = std::min(worst, s); sum += s; }
            results.push_back({{nf,ss,mins}, (float)det/n, det?worst:0.f,
                               det?sum/det:0.f, mspf, templates});
        }
    }
    std::printf("(%d matcher runs -> %zu evaluated combos)\n\n", matcher_runs, results.size());

    std::sort(results.begin(), results.end(), [](const Result&a, const Result&b){
        if (a.detect_rate != b.detect_rate) return a.detect_rate > b.detect_rate;
        if (a.worst != b.worst)             return a.worst > b.worst;
        return a.ms < b.ms;
    });

    std::printf("%-5s %-7s %-4s | %-8s %-7s %-7s %-9s %-5s\n",
                "nfeat","sstep","minS","detect","worst","mean","ms/frame","#tmpl");
    std::printf("--------------------------------------------------------------------\n");
    for (auto& r : results)
        std::printf("%-5d %-7.2f %-4.0f | %6.0f%%  %6.1f  %6.1f  %8.2f  %-5d\n",
                    r.c.num_features, r.c.scale_step, r.c.min_score,
                    r.detect_rate*100, r.worst, r.mean, r.ms, r.templates);

    // Per-scale cliff view (finest config, at min_score=50).
    std::printf("\nper-scale detection @ finest config (nf=%d, sstep=%.2f), min_score=50:\n",
                NF[(sizeof(NF)/sizeof(int))-1], SS[(sizeof(SS)/sizeof(float))-1]);
    std::printf("  %-7s %-8s %-s\n", "scale", "det/ang", "scores");
    const int na = (int)(sizeof(test_angles)/sizeof(float));
    for (size_t si = 0; si < sizeof(test_scales)/sizeof(float); ++si) {
        int det = 0; std::string sc;
        for (int a = 0; a < na; ++a) {
            float s = best_by_scale.empty() ? -1.f : best_by_scale[si*na + a];
            char b[16]; std::snprintf(b, sizeof b, "%5.1f ", s);
            sc += b;
            if (s >= 50.f) ++det;
        }
        std::printf("  %-7.2f %d/%-6d %s\n", test_scales[si], det, na, sc.c_str());
    }

    auto& b = results.front();
    std::printf("\nBEST (robust): num_features=%d scale_step=%.2f min_score=%.0f\n"
                "               -> detect %.0f%%, worst-case score %.1f, <=%.2f ms/frame, "
                "%d templates\n",
                b.c.num_features, b.c.scale_step, b.c.min_score,
                b.detect_rate*100, b.worst, b.ms, b.templates);
    return 0;
}
