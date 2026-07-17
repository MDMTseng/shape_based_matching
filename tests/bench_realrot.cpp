// bench_realrot.cpp — ANALYTIC rotation vs REAL per-angle extraction.
//
// The library builds rotated templates ANALYTICALLY: it extracts features once
// from the upright template, then for every angle rotates the feature
// coordinates (cos/sin), adds the angle to each feature's orientation, and
// re-quantizes (theta*16/360 & 7). Fast to build, but the local gradient
// orientation does NOT rotate perfectly rigidly — at off-grid angles the
// analytic label can drift from what a real detector would see on the rotated
// image, costing match score. selectRotationStable() exists to trim to the
// features that survive that drift.
//
// The ALTERNATIVE this bench tests: rotate the template IMAGE at each angle and
// run real extractFeatures() on it — the true features/orientations at that
// pose, no analytic approximation. Slower to build (one extraction per angle),
// same match cost (same template count). The question:
//
//   Does real per-angle extraction score high enough at a COARSE angle step
//   that you can use far fewer templates than analytic's fine step?
//
// Method: one feature-rich shape. Both approaches build a template bank at
// angle step ∈ {1,5,15,30}. Score both against the SAME off-grid test poses
// (angles NOT on either grid) + Gaussian noise. Report worst/mean coarse score
// and detection rate per (approach, step). refine=None — pure coarse fidelity.
//
//   cmake --build build --target bench_realrot
//   ./bench_realrot            # built-in feature-rich shape
//   ./bench_realrot my.png     # your own grayscale template

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#if defined(__AVX2__)
#  define SBM_SIMD "AVX2"
#elif defined(__aarch64__)
#  define SBM_SIMD "NEON"
#else
#  define SBM_SIMD "scalar"
#endif

using Clk = std::chrono::high_resolution_clock;

// Feature-rich shape (corners + crossings survive rotation extraction well).
static cv::Mat builtin_shape() {
    cv::Mat m(160, 160, CV_8U, cv::Scalar(40));
    cv::rectangle(m, {30,30}, {130,130}, 230, 3);
    cv::line(m, {30,30}, {130,130}, 180, 2);
    cv::line(m, {130,30}, {30,130}, 180, 2);
    cv::rectangle(m, {55,55}, {105,105}, 120, 2);
    cv::circle(m, {80,80}, 18, 200, 2);
    return m;
}

struct TestCase { cv::Mat scene; float cx, cy; };

// Rotate template to angleDeg, place at (px,py) on a WxH canvas, add noise.
static TestCase make_case(const cv::Mat& tmpl, float angleDeg, double sigma,
                          int W, int H, int px, int py) {
    cv::Mat rot;
    cv::Point2f ctr(tmpl.cols/2.f, tmpl.rows/2.f);
    cv::Mat R = cv::getRotationMatrix2D(ctr, angleDeg, 1.0);
    cv::warpAffine(tmpl, rot, R, tmpl.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(40));
    cv::Mat scene(H, W, CV_8U, cv::Scalar(40));
    rot.copyTo(scene(cv::Rect(px, py, tmpl.cols, tmpl.rows)));
    cv::Mat noise(H, W, CV_8U); cv::randn(noise, 0, sigma);
    cv::add(scene, noise, scene);
    return {scene, px + tmpl.cols/2.f, py + tmpl.rows/2.f};
}

struct Eval { float detect, worst, mean; double build_ms, match_ms; int templates; };

// Score a built matcher against the shared test set.
static Eval evaluate(sbm::ShapeMatcher& m, const std::vector<TestCase>& tests,
                     float pos_tol, double build_ms, int templates) {
    std::vector<float> raw(tests.size(), -1.f);
    double tot = 0;
    for (size_t i = 0; i < tests.size(); ++i) {
        auto t0 = Clk::now();
        auto rs = m.match(tests[i].scene);
        tot += std::chrono::duration<double,std::milli>(Clk::now()-t0).count();
        for (auto& r : rs)
            if (std::abs(r.x-tests[i].cx) < pos_tol && std::abs(r.y-tests[i].cy) < pos_tol)
                raw[i] = std::max(raw[i], r.score);
    }
    int det = 0; float worst = 1e9f, sum = 0;
    for (float s : raw) if (s > 0) { det++; worst = std::min(worst, s); sum += s; }
    return {(float)det/tests.size(), det?worst:0.f, det?sum/det:0.f,
            build_ms, tot/tests.size(), templates};
}

int main(int argc, char** argv) {
    sbm::setLogLevel(sbm::LogLevel::Warning);

    cv::Mat tmpl;
    if (argc > 1) tmpl = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (tmpl.empty()) tmpl = builtin_shape();

    const int W = 1280, H = 960, px = 560, py = 400;
    const double sigma = 10.0;
    const float pos_tol = 12.0f;
    const int   num_features = 128;

    // Off-grid test poses: 24 angles at 15.37° spacing so none land on the
    // 1/5/15/30° build grids (worst case for each approach's interpolation).
    std::vector<TestCase> tests;
    for (int i = 0; i < 24; ++i)
        tests.push_back(make_case(tmpl, i * 15.37f + 2.6f, sigma, W, H, px, py));

    std::printf("bench_realrot | %s | template %dx%d | %zu off-grid test poses "
                "@ %dx%d, sigma=%.0f, nf=%d\n",
                SBM_SIMD, tmpl.cols, tmpl.rows, tests.size(), W, H, sigma, num_features);
    std::printf("ANALYTIC = library's rotate-features-in-place; "
                "REAL = rotate image + extractFeatures per angle\n\n");

    sbm::MatchConfig cfg;
    cfg.min_score = 30; cfg.refine = sbm::RefineMode::None; cfg.skip_voting = true;

    const float STEPS[] = {1, 5, 15, 30};

    std::printf("%-9s %-5s %-6s | %-8s %-7s %-7s | %-9s %-8s\n",
                "approach","step","#tmpl","detect","worst","mean","build_ms","match_ms");
    std::printf("--------------------------------------------------------------------------\n");

    for (float step : STEPS) {
        // ---- ANALYTIC: one extraction, library rotates features to each angle.
        {
            auto t0 = Clk::now();
            sbm::FeatureSet fs = sbm::extractFeatures(tmpl, cv::Mat(), num_features,
                                                      cfg.T_levels, cfg.weak_threshold,
                                                      cfg.strong_threshold);
            auto m = std::make_unique<sbm::ShapeMatcher>(cfg);
            sbm::ModelConfig mc; mc.angle = {0, 360, step};
            int nt = m->addModel("m", fs, mc);
            double build = std::chrono::duration<double,std::milli>(Clk::now()-t0).count();
            Eval e = evaluate(*m, tests, pos_tol, build, nt);
            std::printf("%-9s %-5.0f %-6d | %6.0f%%  %6.1f  %6.1f | %8.1f %8.2f\n",
                        "ANALYTIC", step, e.templates, e.detect*100, e.worst, e.mean,
                        e.build_ms, e.match_ms);
        }
        // ---- REAL: rotate the image at each angle, extract real features.
        {
            auto t0 = Clk::now();
            auto m = std::make_unique<sbm::ShapeMatcher>(cfg);
            cv::Point2f ctr(tmpl.cols/2.f, tmpl.rows/2.f);
            int nt = 0, idx = 0;
            for (float a = 0; a < 360; a += step, ++idx) {
                cv::Mat rot;
                cv::Mat R = cv::getRotationMatrix2D(ctr, -a, 1.0);  // model pose +a
                cv::warpAffine(tmpl, rot, R, tmpl.size(), cv::INTER_LINEAR,
                               cv::BORDER_CONSTANT, cv::Scalar(40));
                sbm::FeatureSet fs = sbm::extractFeatures(rot, cv::Mat(), num_features,
                                                          cfg.T_levels, cfg.weak_threshold,
                                                          cfg.strong_threshold);
                if (fs.levels.empty()) continue;
                sbm::ModelConfig mc; mc.angle = {0, 360, 360};  // single template
                nt += m->addModel("r" + std::to_string(idx), fs, mc);
            }
            double build = std::chrono::duration<double,std::milli>(Clk::now()-t0).count();
            Eval e = evaluate(*m, tests, pos_tol, build, nt);
            std::printf("%-9s %-5.0f %-6d | %6.0f%%  %6.1f  %6.1f | %8.1f %8.2f\n",
                        "REAL", step, e.templates, e.detect*100, e.worst, e.mean,
                        e.build_ms, e.match_ms);
        }
        std::printf("--------------------------------------------------------------------------\n");
    }

    std::printf("\nRead: at a COARSE step (15/30), if REAL holds worst/mean score while\n"
                "ANALYTIC collapses, real per-angle extraction buys a coarser grid\n"
                "(fewer templates) at equal robustness. If they track, analytic's\n"
                "cheap build wins. match_ms is ~equal at equal #tmpl (same coarse cost).\n");
    return 0;
}
