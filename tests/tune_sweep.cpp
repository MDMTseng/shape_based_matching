// tune_sweep.cpp — auto-sweep extraction + matching parameters and report the
// most robust config for a template.
//
// Ground truth = "alter template": the tool generates test scenes by rotating
// the template to known (off-grid) angles and adding Gaussian noise, so it
// knows the correct pose without any labelling. Every parameter combo is scored
// against the SAME pre-generated test set (fair comparison).
//
// Objective = robust detection: rank combos by (detection rate, then the
// WORST-CASE score across test cases, then speed). The winner is the config
// least likely to drop a marginal instance.
//
// Sweeps: num_features x match_scale x scaled_blur_ksize x min_score.
// (match_scale<1 uses the addModel image overload = re-extract at scale; blur
// only applies there.) Extend the grids below as needed.
//
//   cmake --build build --target tune_sweep
//   ./tune_sweep            # built-in thin star (the hard case)
//   ./tune_sweep 3          # built-in shape index 0..4
//   ./tune_sweep my.png     # your own grayscale template
//
// This is the standalone validation of an eventual sbm::autoTune() API.

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

// ---- built-in shapes (0..4); 4 = thin star = the shape tuning matters most for
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

struct TestCase { cv::Mat scene; float cx, cy; };

// Alter the template into a scene: rotate to angleDeg, place at center, add noise.
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

struct Combo { int num_features; float match_scale; int blur; float min_score; };
struct Result { Combo c; float detect_rate, worst, mean; double ms; };

int main(int argc, char** argv) {
    sbm::setLogLevel(sbm::LogLevel::Warning);

    cv::Mat tmpl;
    std::string arg = argc > 1 ? argv[1] : "";
    if (!arg.empty() && arg.find_first_not_of("0123456789") != std::string::npos)
        tmpl = cv::imread(arg, cv::IMREAD_GRAYSCALE);
    if (tmpl.empty()) tmpl = builtin(arg.empty() ? 4 : std::atoi(arg.c_str()));

    const int W = 1280, H = 960, px = 560, py = 400;
    const double sigma = 12.0;
    const float pos_tol = 10.0f;   // px

    // Pre-generate the shared test set: off-grid angles + fixed noise.
    const float test_angles[] = {5, 41, 77, 124, 168, 205, 249, 296, 331, 358};
    std::vector<TestCase> tests;
    for (float a : test_angles) tests.push_back(make_case(tmpl, a, sigma, W, H, px, py));

    // Sweep grids (edit to taste). min_score is a FREE post-filter — proven
    // equivalent to re-running (refine=None): one low-threshold run per
    // extraction config yields every threshold by filtering the recorded
    // correct-match scores. So we run the matcher only per (nf, scale, blur),
    // and evaluate the whole min_score range for free at fine resolution.
    const int   NF[]  = {63, 128};
    const float MS[]  = {1.0f, 0.7f, 0.5f};
    const int   BL[]  = {0, 3};
    const float MIN[] = {30, 35, 40, 45, 50, 55, 60, 65};
    const float MIN_LOW = MIN[0];
    const int   n = (int)tests.size();

    std::printf("tune_sweep | %s | template %dx%d | %d test cases @ %dx%d, sigma=%.0f\n",
                SBM_SIMD, tmpl.cols, tmpl.rows, n, W, H, sigma);
    std::printf("objective: robust detection (detect-rate, then worst-case score, then speed)\n");
    std::printf("min_score collapsed: swept %d thresholds from 1 run each (free)\n\n",
                (int)(sizeof(MIN)/sizeof(MIN[0])));

    sbm::ModelConfig mc; mc.angle = {0, 360, 5};   // model coverage for the sweep
    using Clk = std::chrono::high_resolution_clock;

    std::vector<Result> results;
    int matcher_runs = 0;
    for (int nf : NF)
    for (float ms : MS)
    for (int bl : BL) {
        if (ms >= 0.999f && bl != 0) continue;     // blur only affects the scaled path

        // ONE run at the lowest threshold; record each case's best correct score.
        sbm::MatchConfig cfg;
        cfg.min_score = MIN_LOW; cfg.refine = sbm::RefineMode::None;
        cfg.skip_voting = true; cfg.match_scale = ms;
        auto matcher = std::make_unique<sbm::ShapeMatcher>(cfg);
        matcher->addModel("m", tmpl, cv::Mat(), mc, nf, bl);
        ++matcher_runs;

        std::vector<float> raw(n, -1.f);           // best correct-position score / case
        double tot = 0;
        for (int i = 0; i < n; ++i) {
            auto t0 = Clk::now();
            auto rs = matcher->match(tests[i].scene);
            tot += std::chrono::duration<double,std::milli>(Clk::now()-t0).count();
            for (auto& r : rs)
                if (std::abs(r.x-tests[i].cx)<pos_tol && std::abs(r.y-tests[i].cy)<pos_tol)
                    raw[i] = std::max(raw[i], r.score);
        }
        double mspf = tot / n;   // measured at MIN_LOW = conservative (higher min_score is faster)

        // Derive every min_score threshold by filtering — no extra matching.
        for (float mins : MIN) {
            int det = 0; float worst = 1e9f, sum = 0;
            for (float s : raw) if (s >= mins) { det++; worst = std::min(worst, s); sum += s; }
            results.push_back({{nf,ms,bl,mins},
                               (float)det/n, det?worst:0.f, det?sum/det:0.f, mspf});
        }
    }
    std::printf("(%d matcher runs -> %zu evaluated combos)\n\n", matcher_runs, results.size());

    // Rank: robust detection.
    std::sort(results.begin(), results.end(), [](const Result&a, const Result&b){
        if (a.detect_rate != b.detect_rate) return a.detect_rate > b.detect_rate;
        if (a.worst != b.worst)             return a.worst > b.worst;
        return a.ms < b.ms;
    });

    std::printf("%-5s %-6s %-5s %-4s | %-8s %-7s %-7s %-8s\n",
                "nfeat","mscale","blur","minS","detect","worst","mean","ms/frame");
    std::printf("---------------------------------------------------------------\n");
    for (auto& r : results)
        std::printf("%-5d %-6.2f %-5d %-4.0f | %6.0f%%  %6.1f  %6.1f  %7.2f\n",
                    r.c.num_features, r.c.match_scale, r.c.blur, r.c.min_score,
                    r.detect_rate*100, r.worst, r.mean, r.ms);

    std::printf("(ms/frame measured at min_score=%.0f; higher min_score prunes more at the "
                "coarse level, so it is only ever faster.)\n", MIN_LOW);

    auto& b = results.front();
    std::printf("\nBEST (robust): num_features=%d match_scale=%.2f scaled_blur=%d min_score=%.0f\n"
                "               -> detect %.0f%%, worst-case score %.1f, <=%.2f ms/frame\n",
                b.c.num_features, b.c.match_scale, b.c.blur, b.c.min_score,
                b.detect_rate*100, b.worst, b.ms);
    return 0;
}
