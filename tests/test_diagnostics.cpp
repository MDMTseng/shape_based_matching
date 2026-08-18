// Phase 2 diagnostic tests 3a (Feature Selection Quality) and 3b (Coarse Matching Score Analysis).
// Produces CSV output for offline analysis, not CI pass/fail.
//
// Output:
//   output/diag_features.csv       — per-feature selection diagnostics
//   output/diag_coarse_scores.csv  — coarse matching score analysis under noise

#include "shape_matcher.h"
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
#include <string>

using namespace cv;

// ============================================================
// Helpers (shared with other tests)
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
// 3a: Feature Selection Quality Diagnostics
// ============================================================

static void run_feature_diagnostics(FILE* csv) {
    printf("===== 3a: FEATURE SELECTION QUALITY DIAGNOSTICS =====\n\n");

    // CSV header
    fprintf(csv, "template,feat_idx,px,py,leverage,cornerness,is_corner,d_ang,d_pos\n");

    struct TemplateInfo {
        std::string name;
        Mat image;
    };

    // Create 4 templates
    std::vector<TemplateInfo> templates;

    // 1) L-shape 200x200
    {
        Mat img(200, 200, CV_8U, Scalar(0));
        draw_L(img, 100, 100, 0, 200);
        templates.push_back({"L-shape", img});
    }

    // 2) Rectangle 200x100
    {
        Mat img(100, 200, CV_8U, Scalar(0));
        cv::rectangle(img, Point(30, 15), Point(170, 85), Scalar(200), -1);
        templates.push_back({"rectangle", img});
    }

    // 3) Thin pole 200x30
    {
        Mat img(200, 30, CV_8U, Scalar(0));
        cv::rectangle(img, Point(5, 10), Point(25, 190), Scalar(200), -1);
        templates.push_back({"thin_pole", img});
    }

    // 4) Circle (filled circle r=80 in 200x200)
    {
        Mat img(200, 200, CV_8U, Scalar(0));
        cv::circle(img, Point(100, 100), 80, Scalar(200), -1);
        templates.push_back({"circle", img});
    }

    printf("%-12s %6s %6s %6s %10s %10s\n",
           "Template", "Total", "Corner", "Edge", "Coverage%", "WorstSens");
    printf("%-12s %6s %6s %6s %10s %10s\n",
           "--------", "-----", "------", "----", "---------", "---------");

    for (auto& ti : templates) {
        sbm::FeatureSet feat;
        {
            OutputGuard guard;
            feat = sbm::extractFeatures(ti.image);
        }

        // Select optimized points
        std::vector<cv::Point2f> opt_pts;
        {
            OutputGuard guard;
            opt_pts = feat.selectOptimizedPoints(8);
        }

        // Run sensitivity analysis
        sbm::FeatureSet::SensitivityReport sens;
        {
            OutputGuard guard;
            sens = feat.analyzeSensitivity();
        }

        // For each selected point, find the matching refine_point to get cornerness
        int corners_selected = 0;
        int edges_selected = 0;
        float min_px = 1e9f, max_px = -1e9f, min_py = 1e9f, max_py = -1e9f;

        float tcx = feat.templ_width / 2.0f;
        float tcy = feat.templ_height / 2.0f;

        int n_sens = (int)sens.features.size();
        int n_opt = (int)opt_pts.size();
        int n_write = std::min(n_sens, n_opt);

        for (int fi = 0; fi < n_write; fi++) {
            float px = opt_pts[fi].x;
            float py = opt_pts[fi].y;

            // Find closest refine_point
            int best_rp = -1;
            float best_d = 1e9f;
            for (size_t ri = 0; ri < feat.refine_points.size(); ri++) {
                float dx = px - feat.refine_points[ri].px;
                float dy = py - feat.refine_points[ri].py;
                float d = dx*dx + dy*dy;
                if (d < best_d) { best_d = d; best_rp = (int)ri; }
            }

            float cornerness = 0;
            bool is_corner = false;
            if (best_rp >= 0) {
                cornerness = feat.refine_points[best_rp].cornerness;
                is_corner = (feat.refine_points[best_rp].type ==
                             sbm::FeatureSet::RefinePt::CORNER);
            }

            float leverage = sens.features[fi].leverage;
            float d_ang = sens.features[fi].d_ang;
            float d_pos = sens.features[fi].d_pos;

            if (is_corner) corners_selected++;
            else edges_selected++;

            // Track bounding box (in template coords)
            float abs_x = px + tcx;
            float abs_y = py + tcy;
            min_px = std::min(min_px, abs_x);
            max_px = std::max(max_px, abs_x);
            min_py = std::min(min_py, abs_y);
            max_py = std::max(max_py, abs_y);

            fprintf(csv, "%s,%d,%.2f,%.2f,%.4f,%.4f,%d,%.4f,%.4f\n",
                    ti.name.c_str(), fi, px, py, leverage, cornerness,
                    is_corner ? 1 : 0, d_ang, d_pos);
        }

        // Spatial coverage: bounding box of selected features as % of template area
        float bbox_w = (n_write > 1) ? (max_px - min_px) : 0;
        float bbox_h = (n_write > 1) ? (max_py - min_py) : 0;
        float coverage_pct = 0;
        if (feat.templ_width > 0 && feat.templ_height > 0)
            coverage_pct = (bbox_w * bbox_h) / (feat.templ_width * feat.templ_height) * 100.0f;

        float worst_sens = sens.worst_angle_sens;

        printf("%-12s %6d %6d %6d %9.1f%% %10.3f\n",
               ti.name.c_str(),
               n_write, corners_selected, edges_selected,
               coverage_pct, worst_sens);
    }

    printf("\nCSV written: output/diag_features.csv\n");
}

// ============================================================
// 3b: Coarse Matching Score Analysis
// ============================================================

static void run_coarse_score_analysis(FILE* csv) {
    printf("\n===== 3b: COARSE MATCHING SCORE ANALYSIS =====\n\n");

    fprintf(csv, "noise,result_idx,x,y,angle,score,dist_to_gt,is_correct\n");

    // Create 200x200 L-shape template
    Mat templ200(200, 200, CV_8U, Scalar(0));
    draw_L(templ200, 100, 100, 0, 200);

    sbm::FeatureSet feat;
    {
        OutputGuard guard;
        feat = sbm::extractFeatures(templ200);
    }
    feat.setOrigin(100, 75);

    const int scene_sz = 400;
    const int obj_cx = 200, obj_cy = 200;
    const float gt_angle = 25.0f;
    const float org_x = 100, org_y = 75;

    float gt_x, gt_y;
    compute_gt_origin(obj_cx, obj_cy, gt_angle, org_x, org_y, 200, 200, gt_x, gt_y);

    int noise_levels[] = {0, 20, 40};

    printf("%-6s %12s %12s %12s %8s\n",
           "Noise", "CorrectScr", "BestFPScr", "Margin", "Results");
    printf("%-6s %12s %12s %12s %8s\n",
           "-----", "----------", "---------", "------", "-------");

    for (int noise : noise_levels) {
        // Create scene: place 1 object at 25 degrees
        Mat scene(scene_sz, scene_sz, CV_8U, Scalar(0));
        place_object(templ200, scene, obj_cx, obj_cy, gt_angle);

        // Add noise if requested (fixed seed for determinism)
        if (noise > 0) {
            Mat noise_mat(scene.size(), CV_16S);
            cv::theRNG() = cv::RNG(42);
            randn(noise_mat, 0, noise);
            Mat scene16;
            scene.convertTo(scene16, CV_16S);
            scene16 += noise_mat;
            scene16.convertTo(scene, CV_8U);
        }

        // Run coarse match with very low threshold to catch everything
        sbm::MatchConfig cfg;
        cfg.min_score = 10;
        cfg.refine = sbm::RefineMode::None;
        cfg.nms_radius = 0;  // No NMS to get all candidates
        cfg.max_results = 0; // Unlimited

        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 1};

        sbm::ShapeMatcher matcher(cfg);
        {
            OutputGuard guard;
            matcher.addModel("L", feat, mcfg);
        }

        std::vector<sbm::MatchResult> results;
        {
            OutputGuard guard;
            results = matcher.match(scene);
        }

        // Classify results: correct match = closest to GT within 50px
        float correct_score = 0;
        float best_fp_score = 0;
        int correct_idx = -1;

        // Find the correct match (best score among those within 50px of GT)
        for (int i = 0; i < (int)results.size(); i++) {
            float dist = pos_err(results[i].x, results[i].y, gt_x, gt_y);
            if (dist < 50.0f) {
                if (correct_idx < 0 || results[i].score > correct_score) {
                    correct_idx = i;
                    correct_score = results[i].score;
                }
            }
        }

        // Find best false positive score
        for (int i = 0; i < (int)results.size(); i++) {
            if (i == correct_idx) continue;
            float dist = pos_err(results[i].x, results[i].y, gt_x, gt_y);
            if (dist >= 50.0f || correct_idx < 0) {
                best_fp_score = std::max(best_fp_score, results[i].score);
            }
        }

        float margin = correct_score - best_fp_score;

        // Write per-result CSV
        for (int i = 0; i < (int)results.size(); i++) {
            float dist = pos_err(results[i].x, results[i].y, gt_x, gt_y);
            bool is_correct = (i == correct_idx);
            fprintf(csv, "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d\n",
                    noise, i,
                    results[i].x, results[i].y, results[i].angle,
                    results[i].score, dist,
                    is_correct ? 1 : 0);
        }

        printf("%-6d %12.2f %12.2f %12.2f %8d\n",
               noise, correct_score, best_fp_score, margin, (int)results.size());
    }

    printf("\nCSV written: output/diag_coarse_scores.csv\n");
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

    printf("===== DIAGNOSTIC TESTS (3a + 3b) =====\n\n");

    // --- 3a: Feature Selection Quality ---
    FILE* feat_csv = fopen("output/diag_features.csv", "w");
    if (!feat_csv) {
        printf("ERROR: cannot create output/diag_features.csv\n");
        return 1;
    }
    run_feature_diagnostics(feat_csv);
    fclose(feat_csv);

    // --- 3b: Coarse Matching Score Analysis ---
    FILE* coarse_csv = fopen("output/diag_coarse_scores.csv", "w");
    if (!coarse_csv) {
        printf("ERROR: cannot create output/diag_coarse_scores.csv\n");
        return 1;
    }
    run_coarse_score_analysis(coarse_csv);
    fclose(coarse_csv);

    printf("\n===== ALL DIAGNOSTICS COMPLETE =====\n");
    return 0;
}
