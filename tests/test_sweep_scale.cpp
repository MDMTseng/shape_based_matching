// Sweep tests for blur, template size, object count, and resolution.
// Phase 2 tests 2d, 2e, 2f, 2g from TEST_IMPROVEMENT_PLAN.md
// Outputs:
//   output/sweep_blur.csv
//   output/sweep_template_size.csv
//   output/sweep_object_count.csv
//   output/sweep_resolution.csv

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
// Helpers
// ============================================================

static void draw_L(Mat& img, int cx, int cy, double angle, int color, double scale = 2.0) {
    double rad = angle * CV_PI / 180.0, cs = cos(rad), sn = sin(rad);
    for (double ly = -15*scale; ly <= 15*scale; ly += 0.5)
        for (double lx = -5*scale; lx <= 5*scale; lx += 0.5) {
            int px = cx+(int)(lx*cs-ly*sn+0.5), py = cy+(int)(lx*sn+ly*cs+0.5);
            if (px>=0&&px<img.cols&&py>=0&&py<img.rows) img.at<uchar>(py,px)=(uchar)color;
        }
    for (double ly = 5*scale; ly <= 15*scale; ly += 0.5)
        for (double lx = 5*scale; lx <= 20*scale; lx += 0.5) {
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
// Section 1: Blur Sweep (2d)
// ============================================================

static void run_blur_sweep() {
    printf("===== BLUR SWEEP TEST (10 objects, FHD) =====\n\n");

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

    // Blur kernel sizes
    int kernels[] = {1, 3, 5, 7, 9, 11, 15, 21, 31};
    const int n_kernels = sizeof(kernels) / sizeof(kernels[0]);

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

    // Build matchers (one per mode)
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
        int kernel;
        int mode_idx;
        int found;
        float ang_mean, pos_mean;
        double ms;
    };
    std::vector<SweepResult> all_results;

    printf("Running %d blur levels x %d methods...\n\n", n_kernels, n_modes);

    for (int ki = 0; ki < n_kernels; ki++) {
        int k = kernels[ki];

        // Apply external pre-blur (not blur_kernel_size)
        Mat scene;
        if (k <= 1) {
            scene = scene_clean.clone();
        } else {
            GaussianBlur(scene_clean, scene, Size(k, k), 0);
        }

        for (int mi = 0; mi < n_modes; mi++) {
            SweepResult sr;
            sr.kernel = k;
            sr.mode_idx = mi;
            sr.found = 0;
            sr.ang_mean = 0;
            sr.pos_mean = 0;
            sr.ms = 0;

            std::vector<sbm::MatchResult> res;
            {
                OutputGuard guard;
                // Warmup on first kernel
                if (ki == 0) matchers[mi]->match(scene);

                auto t0 = std::chrono::high_resolution_clock::now();
                res = matchers[mi]->match(scene);
                sr.ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
            }

            // GT matching
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
                    sum_ae += angle_err(res[best_ri].angle, (float)objs[j].angle);
                    sum_pe += best_d;
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
    FILE* csv = fopen("output/sweep_blur.csv", "w");
    if (!csv) {
        printf("ERROR: cannot create output/sweep_blur.csv\n");
        return;
    }
    fprintf(csv, "kernel,method,found,ang_mean,pos_mean,ms\n");
    for (auto& sr : all_results) {
        fprintf(csv, "%d,%s,%d,%.4f,%.4f,%.2f\n",
                sr.kernel, modes[sr.mode_idx].name, sr.found,
                sr.ang_mean, sr.pos_mean, sr.ms);
    }
    fclose(csv);
    printf("CSV written: output/sweep_blur.csv\n\n");

    // --- Summary table ---
    printf("%-6s  %-7s  %5s  %8s  %8s  %8s\n",
           "Kernel", "Method", "Found", "AngMean", "PosMean", "ms");
    printf("%-6s  %-7s  %5s  %8s  %8s  %8s\n",
           "------", "------", "-----", "-------", "-------", "------");

    for (auto& sr : all_results) {
        printf("%-6d  %-7s  %3d/%d  %7.3f%s  %7.3fpx  %7.1f\n",
               sr.kernel, modes[sr.mode_idx].name,
               sr.found, n_objs,
               sr.ang_mean, "\xC2\xB0",
               sr.pos_mean, sr.ms);
    }
    printf("\n");
}

// ============================================================
// Section 2: Template Size Sweep (2e)
// ============================================================

static void run_template_size_sweep() {
    printf("===== TEMPLATE SIZE SWEEP =====\n\n");

    int sizes[] = {50, 100, 150, 200, 300, 400};
    const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    const double gt_angle = 25.0;
    const int scene_sz = 600;
    const int cx = scene_sz / 2, cy = scene_sz / 2;

    struct SizeResult {
        int size;
        int features;
        float ang_err;
        float pos_err_val;
        double select_ms;
        double match_ms;
    };
    std::vector<SizeResult> results;

    for (int si = 0; si < n_sizes; si++) {
        int sz = sizes[si];
        // Scale factor relative to 200x200 base (draw_L default scale=2.0 fills ~200px)
        double draw_scale = sz / 100.0;

        // Create template
        Mat templ(sz, sz, CV_8U, Scalar(0));
        draw_L(templ, sz/2, sz/2, 0, 200, draw_scale);

        // Extract features (use single pyramid level for small templates)
        sbm::FeatureSet feat;
        {
            OutputGuard guard;
            std::vector<int> pyr_T = (sz <= 100) ? std::vector<int>{4} : std::vector<int>{4, 8};
            feat = sbm::extractFeatures(templ, cv::Mat(), 128, pyr_T);
        }

        // Scale origin proportionally: for 200x200 it's (100,75), so (sz/2, sz*75/200)
        float org_x = sz / 2.0f;
        float org_y = sz * 75.0f / 200.0f;
        feat.setOrigin(org_x, org_y);

        int n_feat = feat.numFeatures();
        if (n_feat == 0) {
            printf("  size=%3d: 0 features — skipped\n", sz);
            continue;
        }

        // Time the point selection
        auto t_sel0 = std::chrono::high_resolution_clock::now();
        {
            OutputGuard guard;
            feat.selectOptimizedPoints(15);
        }
        double sel_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_sel0).count();

        // Create scene with 1 object at 25 degrees
        Mat scene(scene_sz, scene_sz, CV_8U, Scalar(30));
        place_object(templ, scene, cx, cy, gt_angle);

        // Build matcher with ROI refinement
        sbm::MatchConfig cfg;
        cfg.min_score = 30;
        cfg.refine = sbm::RefineMode::ROI;
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};

        sbm::ShapeMatcher matcher(cfg);
        {
            OutputGuard guard;
            matcher.addModel("L", feat, mcfg);
        }

        // Warmup + timed match
        float ae = 999.0f, pe = 999.0f;
        double m_ms = 0;
        {
            OutputGuard guard;
            matcher.match(scene); // warmup

            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = matcher.match(scene);
            m_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();

            if (!res.empty()) {
                float gt_x, gt_y;
                compute_gt_origin(cx, cy, gt_angle, org_x, org_y, sz, sz, gt_x, gt_y);
                ae = angle_err(res[0].angle, (float)gt_angle);
                pe = pos_err(res[0].x, res[0].y, gt_x, gt_y);
            }
        }

        SizeResult sr;
        sr.size = sz;
        sr.features = n_feat;
        sr.ang_err = ae;
        sr.pos_err_val = pe;
        sr.select_ms = sel_ms;
        sr.match_ms = m_ms;
        results.push_back(sr);

        printf("  size=%3d: %3d features, ang=%.3f, pos=%.3f, sel=%.2fms, match=%.1fms\n",
               sz, n_feat, ae, pe, sel_ms, m_ms);
    }

    // --- Write CSV ---
    FILE* csv = fopen("output/sweep_template_size.csv", "w");
    if (!csv) {
        printf("ERROR: cannot create output/sweep_template_size.csv\n");
        return;
    }
    fprintf(csv, "size,features,ang_err,pos_err,select_ms,match_ms\n");
    for (auto& sr : results) {
        fprintf(csv, "%d,%d,%.4f,%.4f,%.2f,%.2f\n",
                sr.size, sr.features, sr.ang_err, sr.pos_err_val,
                sr.select_ms, sr.match_ms);
    }
    fclose(csv);
    printf("\nCSV written: output/sweep_template_size.csv\n\n");

    // --- Summary table ---
    printf("%-6s  %8s  %8s  %8s  %10s  %10s\n",
           "Size", "Features", "AngErr", "PosErr", "SelectMs", "MatchMs");
    printf("%-6s  %8s  %8s  %8s  %10s  %10s\n",
           "------", "--------", "-------", "-------", "---------", "---------");
    for (auto& sr : results) {
        printf("%-6d  %8d  %7.3f%s  %7.3fpx  %9.2fms  %9.1fms\n",
               sr.size, sr.features,
               sr.ang_err, "\xC2\xB0",
               sr.pos_err_val,
               sr.select_ms, sr.match_ms);
    }
    printf("\n");
}

// ============================================================
// Section 3: Object Count Sweep (2f)
// ============================================================

static void run_object_count_sweep() {
    printf("===== OBJECT COUNT SWEEP (200x200 L, FHD) =====\n\n");

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

    const float org_x = 100, org_y = 75;
    const float match_radius = 50.0f;

    int counts[] = {1, 2, 5, 10, 20};
    const int n_counts = sizeof(counts) / sizeof(counts[0]);

    // Precompute object positions for max count (20), spread evenly
    // Use a grid-like layout with varied angles
    struct Obj { int x, y; double angle; };
    Obj all_objs[20];
    {
        // 4 columns x 5 rows grid in FHD (1920x1080)
        int cols = 4, rows = 5;
        double angles[] = {15, 45, 90, 135, 180, 210, 240, 270, 300, 330,
                           25, 55, 100, 145, 190, 220, 250, 280, 310, 340};
        for (int i = 0; i < 20; i++) {
            int col = i % cols;
            int row = i / cols;
            all_objs[i].x = 200 + col * 400;
            all_objs[i].y = 150 + row * 200;
            all_objs[i].angle = angles[i];
        }
    }

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

    // Build matchers (one per mode)
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

    struct SweepResult {
        int count;
        int mode_idx;
        int found;
        int fp_count;
        double ms;
    };
    std::vector<SweepResult> all_results;

    printf("Running %d object counts x %d methods...\n\n", n_counts, n_modes);

    for (int ci = 0; ci < n_counts; ci++) {
        int n_obj = counts[ci];

        // Build scene with n_obj objects
        Mat scene(1080, 1920, CV_8U, Scalar(30));
        for (int j = 0; j < n_obj; j++)
            place_object(templ200, scene, all_objs[j].x, all_objs[j].y, all_objs[j].angle);

        for (int mi = 0; mi < n_modes; mi++) {
            SweepResult sr;
            sr.count = n_obj;
            sr.mode_idx = mi;
            sr.found = 0;
            sr.fp_count = 0;
            sr.ms = 0;

            std::vector<sbm::MatchResult> res;
            {
                OutputGuard guard;
                // Warmup on first count
                if (ci == 0) matchers[mi]->match(scene);

                auto t0 = std::chrono::high_resolution_clock::now();
                res = matchers[mi]->match(scene);
                sr.ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
            }

            // GT matching
            std::vector<int> result_used(res.size(), 0);

            for (int j = 0; j < n_obj; j++) {
                float gt_x, gt_y;
                compute_gt_origin(all_objs[j].x, all_objs[j].y, all_objs[j].angle,
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
                }
            }

            // Count false positives: unmatched results
            for (int ri = 0; ri < (int)res.size(); ri++) {
                if (!result_used[ri]) sr.fp_count++;
            }

            all_results.push_back(sr);
        }
    }

    // Clean up matchers
    for (int mi = 0; mi < n_modes; mi++) delete matchers[mi];

    // --- Write CSV ---
    FILE* csv = fopen("output/sweep_object_count.csv", "w");
    if (!csv) {
        printf("ERROR: cannot create output/sweep_object_count.csv\n");
        return;
    }
    fprintf(csv, "count,method,found,fp_count,ms\n");
    for (auto& sr : all_results) {
        fprintf(csv, "%d,%s,%d,%d,%.2f\n",
                sr.count, modes[sr.mode_idx].name, sr.found,
                sr.fp_count, sr.ms);
    }
    fclose(csv);
    printf("CSV written: output/sweep_object_count.csv\n\n");

    // --- Summary table ---
    printf("%-6s  %-7s  %5s  %8s  %8s\n",
           "Count", "Method", "Found", "FP", "ms");
    printf("%-6s  %-7s  %5s  %8s  %8s\n",
           "------", "------", "-----", "--------", "------");

    for (auto& sr : all_results) {
        printf("%-6d  %-7s  %3d/%d  %8d  %7.1f\n",
               sr.count, modes[sr.mode_idx].name,
               sr.found, sr.count,
               sr.fp_count, sr.ms);
    }
    printf("\n");
}

// ============================================================
// Section 4: Resolution Sweep (2g)
// ============================================================

static void run_resolution_sweep() {
    printf("===== RESOLUTION SWEEP (10 L-shapes, coarse/ICP/ROI) =====\n\n");

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

    const float org_x = 100, org_y = 75;
    const float match_radius = 50.0f;
    const int n_objs = 10;

    // Resolutions to test
    struct Resolution {
        const char* name;
        int width, height;
    };
    Resolution resolutions[] = {
        {"VGA",  640,  480},
        {"HD",   1280, 720},
        {"FHD",  1920, 1080},
        {"4K",   3840, 2160},
        {"20MP", 5472, 3648},
    };
    const int n_res = sizeof(resolutions) / sizeof(resolutions[0]);

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

    // Build matchers (one per mode)
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

    struct SweepResult {
        const char* res_name;
        int width, height;
        int mode_idx;
        int found;
        double ms;
        float detection_rate;
    };
    std::vector<SweepResult> all_results;

    printf("Running %d resolutions x %d methods...\n\n", n_res, n_modes);

    for (int ri = 0; ri < n_res; ri++) {
        int w = resolutions[ri].width;
        int h = resolutions[ri].height;

        // Place 10 objects spread across the scene, keeping margin from edges
        struct Obj { int x, y; double angle; };
        Obj objs[10];
        {
            double angles[] = {15, 45, 90, 135, 180, 210, 270, 315, 30, 60};
            int margin = 120; // keep objects away from borders
            int usable_w = w - 2 * margin;
            int usable_h = h - 2 * margin;

            // 2 rows x 5 cols grid layout
            int cols = 5, rows = 2;
            for (int i = 0; i < n_objs; i++) {
                int col = i % cols;
                int row = i / cols;
                objs[i].x = margin + (usable_w * (2 * col + 1)) / (2 * cols);
                objs[i].y = margin + (usable_h * (2 * row + 1)) / (2 * rows);
                objs[i].angle = angles[i];
            }
        }

        // Build scene
        Mat scene(h, w, CV_8U, Scalar(30));
        for (int j = 0; j < n_objs; j++)
            place_object(templ200, scene, objs[j].x, objs[j].y, objs[j].angle);

        printf("  %s (%dx%d): ", resolutions[ri].name, w, h);

        for (int mi = 0; mi < n_modes; mi++) {
            SweepResult sr;
            sr.res_name = resolutions[ri].name;
            sr.width = w;
            sr.height = h;
            sr.mode_idx = mi;
            sr.found = 0;
            sr.ms = 0;

            std::vector<sbm::MatchResult> res;
            {
                OutputGuard guard;
                // Warmup on first resolution
                if (ri == 0) matchers[mi]->match(scene);

                auto t0 = std::chrono::high_resolution_clock::now();
                res = matchers[mi]->match(scene);
                sr.ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
            }

            // GT matching
            std::vector<int> result_used(res.size(), 0);

            for (int j = 0; j < n_objs; j++) {
                float gt_x, gt_y;
                compute_gt_origin(objs[j].x, objs[j].y, objs[j].angle,
                                 org_x, org_y, 200, 200, gt_x, gt_y);

                float best_d = 1e9f;
                int best_ri = -1;
                for (int k = 0; k < (int)res.size(); k++) {
                    if (result_used[k]) continue;
                    float d = pos_err(res[k].x, res[k].y, gt_x, gt_y);
                    if (d < best_d) { best_d = d; best_ri = k; }
                }

                if (best_ri >= 0 && best_d < match_radius) {
                    result_used[best_ri] = 1;
                    sr.found++;
                }
            }

            sr.detection_rate = (float)sr.found / (float)n_objs;
            all_results.push_back(sr);
        }

        // Print inline progress
        auto& last3 = all_results;
        int base = (int)last3.size() - n_modes;
        printf("coarse=%d/%d(%.0fms) icp=%d/%d(%.0fms) roi=%d/%d(%.0fms)\n",
               last3[base].found, n_objs, last3[base].ms,
               last3[base+1].found, n_objs, last3[base+1].ms,
               last3[base+2].found, n_objs, last3[base+2].ms);
    }

    // Clean up matchers
    for (int mi = 0; mi < n_modes; mi++) delete matchers[mi];

    // --- Write CSV ---
    FILE* csv = fopen("output/sweep_resolution.csv", "w");
    if (!csv) {
        printf("ERROR: cannot create output/sweep_resolution.csv\n");
        return;
    }
    fprintf(csv, "resolution,width,height,method,found,detection_rate,speed_ms\n");
    for (auto& sr : all_results) {
        fprintf(csv, "%s,%d,%d,%s,%d,%.4f,%.2f\n",
                sr.res_name, sr.width, sr.height,
                modes[sr.mode_idx].name, sr.found,
                sr.detection_rate, sr.ms);
    }
    fclose(csv);
    printf("\nCSV written: output/sweep_resolution.csv\n\n");

    // --- Summary table ---
    printf("%-6s  %11s  %-7s  %5s  %9s  %8s\n",
           "Res", "Dimensions", "Method", "Found", "DetRate", "ms");
    printf("%-6s  %11s  %-7s  %5s  %9s  %8s\n",
           "------", "-----------", "------", "-----", "---------", "------");

    for (auto& sr : all_results) {
        char dim[32];
        snprintf(dim, sizeof(dim), "%dx%d", sr.width, sr.height);
        printf("%-6s  %11s  %-7s  %3d/%d  %8.1f%%  %7.1f\n",
               sr.res_name, dim, modes[sr.mode_idx].name,
               sr.found, n_objs,
               sr.detection_rate * 100.0f, sr.ms);
    }
    printf("\n");
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

    run_blur_sweep();
    printf("\n");
    run_template_size_sweep();
    printf("\n");
    run_object_count_sweep();
    printf("\n");
    run_resolution_sweep();

    printf("===== ALL SWEEPS COMPLETE =====\n");
    return 0;
}
