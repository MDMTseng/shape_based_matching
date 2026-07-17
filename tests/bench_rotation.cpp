// bench_rotation.cpp — cross-platform match-throughput bench for line2Dup.
//
// Measures the coarse-match hot path (blur -> sobel/quantize -> fused
// spread+LUT+linearize -> coarse similarity) across a scenario matrix, with
// the built-in per-stage profiler. Designed to compare SIMD backends:
//   x86  -> AVX2   (build with the default -march=native -mavx2)
//   ARM  -> NEON   (aarch64; the __aarch64__ NEON paths in line2Dup.cpp)
//   else -> scalar
// The synthetic object and matcher config are fixed, so match scores are
// bit-identical across backends — any score difference is a correctness bug,
// not a perf trade-off.
//
// Build (added as the `bench_rotation` target in CMakeLists.txt):
//   cmake --build . --target bench_rotation
// Run:
//   ./bench_rotation                 # full matrix
//   ./bench_rotation 1280 960 1.0    # single: WxH, rotation step (deg)
//
// Reference numbers (aarch64, Raspberry Pi 5 @2.4GHz, performance governor,
// gcc 14, 4 threads) are in tests/bench_rotation.md. x86 team: drop your
// output next to it so we can compare AVX2 vs NEON on the same scenarios.

#include "line2Dup.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdio>

#if defined(__AVX2__)
#  define SBM_SIMD "AVX2"
#elif defined(__aarch64__)
#  define SBM_SIMD "NEON (aarch64)"
#elif defined(__ARM_NEON)
#  define SBM_SIMD "NEON (armv7)"
#else
#  define SBM_SIMD "scalar"
#endif

// A high-gradient synthetic object so line2Dup finds plenty of features.
static cv::Mat make_object(int w, int h, int sx, int sy, int s) {
    cv::Mat m(h, w, CV_8U, cv::Scalar(40));
    cv::rectangle(m, {sx, sy}, {sx + s, sy + s}, cv::Scalar(230), 3);
    cv::circle(m, {sx + s / 2, sy + s / 2}, s / 3, cv::Scalar(200), 2);
    cv::line(m, {sx, sy}, {sx + s, sy + s}, cv::Scalar(180), 2);
    cv::line(m, {sx + s, sy}, {sx, sy + s}, cv::Scalar(180), 2);
    cv::rectangle(m, {sx + s / 4, sy + s / 4}, {sx + 3 * s / 4, sy + 3 * s / 4},
                  cv::Scalar(120), 2);
    return m;
}

static void run_case(int W, int H, float step) {
    line2Dup::Detector det(63, {4, 8});
    cv::Mat tmpl = make_object(160, 160, 25, 25, 100);
    cv::Mat mask(160, 160, CV_8U, cv::Scalar(255));

    int ntempl;
    if (step <= 0.f) {            // single upright template
        det.addTemplate(tmpl, "obj", mask);
        ntempl = 1;
    } else {                      // full 360-degree rotation sweep
        ntempl = det.addRotatedTemplates(tmpl, mask, "obj", 0, 360, step);
    }

    cv::Mat scene = make_object(W, H, W / 2 - 50, H / 2 - 50, 100);
    std::vector<std::string> ids{"obj"};

    for (int i = 0; i < 3; ++i) det.match(scene, 50.0f, ids);   // warmup

    // Correctness: score must match across backends (see bench_rotation.md).
    auto ms = det.match(scene, 50.0f, ids);
    double best = ms.empty() ? 0.0 : ms[0].similarity;

    using Clk = std::chrono::high_resolution_clock;
    const int N = 20;
    double tot = 0, mn = 1e9, mx = 0;
    for (int i = 0; i < N; ++i) {
        auto t0 = Clk::now();
        auto r = det.match(scene, 50.0f, ids);
        double t = std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
        tot += t; mn = std::min(mn, t); mx = std::max(mx, t);
    }
    double avg = tot / N;
    std::printf("%5dx%-4d  templ=%-4d  match avg %7.2f ms  (min %6.2f / max %6.2f)"
                "  %6.1f fps   nmatch=%zu best=%.4f\n",
                W, H, ntempl, avg, mn, mx, 1000.0 / avg, ms.size(), best);
}

static void run_profile(int W, int H, float step) {
    line2Dup::Detector det(63, {4, 8});
    cv::Mat tmpl = make_object(160, 160, 25, 25, 100);
    cv::Mat mask(160, 160, CV_8U, cv::Scalar(255));
    int ntempl = (step <= 0.f)
        ? (det.addTemplate(tmpl, "obj", mask), 1)
        : det.addRotatedTemplates(tmpl, mask, "obj", 0, 360, step);
    cv::Mat scene = make_object(W, H, W / 2 - 50, H / 2 - 50, 100);
    std::vector<std::string> ids{"obj"};
    for (int i = 0; i < 3; ++i) det.match(scene, 50.0f, ids);

    sbm::setLogLevel(sbm::LogLevel::Debug);
    sbm::setLogFile(stderr);                 // Debug/Info only reach console via the file sink
    line2Dup::enableProfiling(true);
    line2Dup::resetProfiling();
    const int N = 20;
    for (int i = 0; i < N; ++i) det.match(scene, 50.0f, ids);
    std::printf("--- per-stage totals over %d runs @ %dx%d, %d templates (per-frame = /%d) ---\n",
                N, W, H, ntempl, N);
    line2Dup::printProfiling();
}

int main(int argc, char** argv) {
    std::printf("line2Dup bench_rotation | SIMD backend: %s\n", SBM_SIMD);

    if (argc >= 3) {   // explicit single case: W H [step]
        int W = std::atoi(argv[1]), H = std::atoi(argv[2]);
        float step = argc > 3 ? (float)std::atof(argv[3]) : 1.0f;
        run_case(W, H, step);
        run_profile(W, H, step);
        return 0;
    }

    std::printf("== throughput matrix ==\n");
    for (int res = 0; res < 2; ++res) {
        int W = res ? 1280 : 640, H = res ? 960 : 480;
        run_case(W, H, 0.f);     // 1 upright template
        run_case(W, H, 5.f);     // 72 templates  (5-deg sweep)
        run_case(W, H, 1.f);     // 360 templates (1-deg sweep)
    }
    std::printf("\n== per-stage profile: 360 templates @ 1280x960 ==\n");
    run_profile(1280, 960, 1.f);
    return 0;
}
