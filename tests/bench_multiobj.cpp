// bench_multiobj.cpp — realistic multi-object match-throughput bench.
//
// Scenario (fixed so results are comparable across SIMD backends):
//   * 5 distinct object shapes, each swept 0..360 deg at 1-deg step
//     => 5 x 360 = 1800 template variants in one ShapeMatcher.
//   * Scenes at 1.2 / 5 / 20 MP, each with additive Gaussian noise.
//   * Matched twice: WITHOUT refine (RefineMode::None, raw coarse result)
//     and WITH ROI refine (RefineMode::ROI).
//
// Backends (self-labelled in the output):
//   x86 -> AVX2 (default -march=native -mavx2), ARM -> NEON (aarch64), else scalar.
// The synthetic objects/scene/config are deterministic, so match COUNT and
// scores are bit-identical across backends — only timing differs.
//
//   cmake --build . --target bench_multiobj
//   ./bench_multiobj            # full 3-resolution x {none,ROI} matrix
//   ./bench_multiobj 1          # quick: 1.2 MP only
//
// Reference numbers (aarch64, Raspberry Pi 5) live in tests/bench_rotation.md.

#include "shape_matcher.h"
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

// ---- 5 distinct high-gradient objects, each on a 160x160 tile ----
static cv::Mat obj_tile(int which) {
    cv::Mat m(160, 160, CV_8U, cv::Scalar(40));
    const int c = 80, s = 100, x0 = 30, y0 = 30, x1 = 130, y1 = 130;
    switch (which) {
        case 0: // square + diagonals + inner square
            cv::rectangle(m, {x0, y0}, {x1, y1}, 230, 3);
            cv::line(m, {x0, y0}, {x1, y1}, 180, 2);
            cv::line(m, {x1, y0}, {x0, y1}, 180, 2);
            cv::rectangle(m, {55, 55}, {105, 105}, 120, 2);
            break;
        case 1: // concentric circles + cross
            cv::circle(m, {c, c}, 48, 220, 3);
            cv::circle(m, {c, c}, 26, 160, 2);
            cv::line(m, {c, 25}, {c, 135}, 120, 2);
            cv::line(m, {25, c}, {135, c}, 120, 2);
            break;
        case 2: { // triangle + inscribed
            std::vector<cv::Point> t{{c, 25}, {30, 130}, {130, 130}};
            cv::polylines(m, t, true, 225, 3);
            std::vector<cv::Point> t2{{c, 60}, {58, 118}, {102, 118}};
            cv::polylines(m, t2, true, 150, 2);
            break;
        }
        case 3: // L-shape + notch
            cv::rectangle(m, {35, 35}, {70, 130}, 220, cv::FILLED);
            cv::rectangle(m, {35, 95}, {130, 130}, 220, cv::FILLED);
            cv::rectangle(m, {90, 40}, {120, 70}, 40, cv::FILLED);
            break;
        case 4: { // filled 5-point star (solid so its coarse-level score is strong)
            std::vector<cv::Point> star;
            for (int i = 0; i < 10; ++i) {
                double a = CV_PI / 2 + i * CV_PI / 5;
                double r = (i & 1) ? 24 : 54;
                star.push_back({(int)(c + r * cos(a)), (int)(c - r * sin(a))});
            }
            cv::fillPoly(m, std::vector<std::vector<cv::Point>>{star}, 215);
            cv::circle(m, {c, c}, 14, 40, cv::FILLED);   // inner cutout for distinctness
            break;
        }
    }
    return m;
}

// Scene: place the 5 objects on a grid, then add Gaussian noise (sigma).
static cv::Mat make_scene(int W, int H, double sigma, std::vector<cv::Point>& gt) {
    cv::Mat scene(H, W, CV_8U, cv::Scalar(40));
    gt.clear();
    int cols = 3, rows = 2;
    for (int i = 0; i < 5; ++i) {
        int cx = (int)((0.5 + (i % cols)) * W / cols);
        int cy = (int)((0.5 + (i / cols)) * H / rows);
        cv::Mat tile = obj_tile(i);
        int x = std::min(std::max(cx - 80, 0), W - 160);
        int y = std::min(std::max(cy - 80, 0), H - 160);
        tile.copyTo(scene(cv::Rect(x, y, 160, 160)));
        gt.push_back({x + 80, y + 80});
    }
    cv::Mat noise(H, W, CV_8U);
    cv::randn(noise, 0, sigma);
    cv::add(scene, noise, scene);
    return scene;
}

static std::unique_ptr<sbm::ShapeMatcher> build_matcher(sbm::RefineMode refine,
                                                        float match_scale) {
    sbm::MatchConfig cfg;
    // NOTE: min_score is also the coarse-pyramid prune threshold — a candidate
    // whose score at the coarse T=8 level is below it is dropped before fine
    // matching. All 5 objects here are "coarse-strong" (solid/thick outlines)
    // so they clear 50 at every scale; thin sparse shapes would need a lower
    // min_score (which also slows matching by refining more candidates).
    cfg.min_score = 50.0f;
    cfg.refine = refine;
    cfg.match_scale = match_scale; // <1 downscales the scene for a faster coarse
                                   // match; ROI refine recovers full-res accuracy.
    cfg.blur_kernel_size = 7;      // handles ~30 sigma noise
    cfg.skip_voting = true;        // higher thresholds -> safe, saves time at 20MP
    auto matcher = std::make_unique<sbm::ShapeMatcher>(cfg);
    sbm::ModelConfig mc;
    mc.angle = {0, 360, 1};        // 1-deg sweep -> 360 variants / object
    mc.flip = false;
    const char* names[5] = {"square", "rings", "tri", "ell", "star"};
    for (int i = 0; i < 5; ++i)
        matcher->addModel(names[i], sbm::extractFeatures(obj_tile(i)), mc);
    return matcher;
}

static void run(sbm::ShapeMatcher& matcher, const char* label,
                int W, int H, double sigma) {
    double mp = (double)W * H / 1e6;
    std::vector<cv::Point> gt;
    cv::Mat scene = make_scene(W, H, sigma, gt);

    // Fewer timed iterations as the scene grows (20MP x 1800 templ is heavy).
    int N = (mp > 12) ? 5 : (mp > 3) ? 8 : 20;

    matcher.match(scene);                       // warmup + JIT of caches
    auto rs = matcher.match(scene);

    using Clk = std::chrono::high_resolution_clock;
    double tot = 0, mn = 1e9, mx = 0;
    for (int i = 0; i < N; ++i) {
        auto t0 = Clk::now();
        auto r = matcher.match(scene);
        double t = std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
        tot += t; mn = std::min(mn, t); mx = std::max(mx, t);
    }
    double avg = tot / N;
    std::printf("  %5.1f MP (%4dx%-4d)  %-14s  match avg %8.2f ms"
                "  (min %7.2f/max %7.2f)  %6.2f fps   nmatch=%zu/5\n",
                mp, W, H, label, avg, mn, mx, 1000.0 / avg, rs.size());
}

int main(int argc, char** argv) {
    sbm::setLogLevel(sbm::LogLevel::Warning);   // quiet the per-add feature logs
    std::printf("bench_multiobj | SIMD backend: %s | 5 objects x 360deg (1-deg) = 1800 templates\n",
                SBM_SIMD);
    std::printf("scene: 5 objects on a grid + Gaussian noise (sigma=10)\n");

    const double sigma = 10.0;
    struct { int w, h; } res[] = {{1280, 960}, {2592, 1944}, {5184, 3888}};  // 1.2 / 5 / 20 MP
    int only = (argc > 1) ? std::atoi(argv[1]) : 0;   // 1=1.2MP, 2=+5MP, else all
    int nres = only == 1 ? 1 : only == 2 ? 2 : 3;

    // Variants: baseline coarse-only at full res, then ROI refine at full res
    // and at 0.7 / 0.5 scene downscale. Downscale shrinks the coarse-match cost
    // (~scale^2); ROI refine runs at full res to recover accuracy.
    struct Variant { const char* label; sbm::RefineMode refine; float scale; };
    const Variant variants[] = {
        {"none   s=1.00", sbm::RefineMode::None, 1.00f},
        {"ROI    s=1.00", sbm::RefineMode::ROI,  1.00f},
        {"ROI    s=0.70", sbm::RefineMode::ROI,  0.70f},
        {"ROI    s=0.50", sbm::RefineMode::ROI,  0.50f},
    };
    const int nv = (int)(sizeof(variants) / sizeof(variants[0]));

    // Templates are scene-independent — build each matcher once, reuse across
    // resolutions (registering 1800 rotated variants is the expensive setup).
    std::printf("building %d matchers (1800 templates each)...\n", nv);
    std::vector<std::unique_ptr<sbm::ShapeMatcher>> matchers;
    for (auto& v : variants)
        matchers.push_back(build_matcher(v.refine, v.scale));

    for (int i = 0; i < nres; ++i) {
        for (int v = 0; v < nv; ++v)
            run(*matchers[v], variants[v].label, res[i].w, res[i].h, sigma);
    }
    return 0;
}
