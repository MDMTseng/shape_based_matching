// Phase 2 diagnostic test 3c: Refinement Convergence Curves.
// Records per-iteration angle_error, pos_error, residual for both ICP and ROI.
//
// Output:
//   output/diag_convergence_icp.csv  — ICP per-iteration convergence data
//   output/diag_convergence_roi.csv  — ROI per-iteration convergence data
//
// Highlights: convergence plateau, oscillation, divergence (if any).

#include "shape_matcher.h"
#include "icp_refine.h"
#include "roi_refine.h"
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
#include <string>

using namespace cv;

// ============================================================
// Helpers
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
// ICP Convergence: run with max_iterations=1..N, record per-step
// ============================================================

struct IterRecord {
    int iteration;
    float angle_gt;
    float angle_error;
    float pos_error;
    float rmse;
    float fitness;
};

static std::vector<IterRecord> run_icp_convergence(
    const icp_refine::EdgeScene& templ_scene,
    int templ_w, int templ_h,
    const Mat& scene_gray,
    const icp_refine::Pose2D& initial_pose,
    float gt_angle, float gt_x, float gt_y,
    float org_x, float org_y,
    int max_iters)
{
    std::vector<IterRecord> records;

    for (int n = 1; n <= max_iters; ++n) {
        icp_refine::ICPConfig icfg;
        icfg.max_iterations = n;
        icfg.max_dist = 10.0f;
        icfg.convergence_rmse = 0;    // disable early convergence to force N iters
        icfg.convergence_fitness = 0;

        icp_refine::Pose2D result;
        {
            OutputGuard guard;
            result = icp_refine::refineInverse(
                templ_scene, templ_w, templ_h,
                scene_gray, initial_pose, 20, icfg);
        }

        // Convert center pose to user-origin pose for error computation
        float rr = -result.angle * (float)CV_PI / 180.0f;
        float ux = org_x - templ_w / 2.0f;
        float uy = org_y - templ_h / 2.0f;
        float res_x = result.x + std::cos(rr) * ux - std::sin(rr) * uy;
        float res_y = result.y + std::sin(rr) * ux + std::cos(rr) * uy;

        IterRecord rec;
        rec.iteration = n;
        rec.angle_gt = gt_angle;
        rec.angle_error = angle_err(result.angle, gt_angle);
        rec.pos_error = pos_err(res_x, res_y, gt_x, gt_y);
        rec.rmse = result.rmse;
        rec.fitness = result.fitness;
        records.push_back(rec);
    }
    return records;
}

// ============================================================
// ROI Convergence: run with max_iters=1..N, record per-step
// ============================================================

static std::vector<IterRecord> run_roi_convergence(
    const Mat& templ_img,
    const Mat& scene_gray,
    const std::vector<roi_refine::SamplePoint>& sample_pts,
    const Vec3f& initial_pose,
    float gt_angle, float gt_x, float gt_y,
    float org_x, float org_y,
    int templ_w, int templ_h,
    int max_iters)
{
    std::vector<IterRecord> records;

    for (int n = 1; n <= max_iters; ++n) {
        roi_refine::ROIConfig rcfg;
        rcfg.max_iters = n;
        rcfg.max_points = 20;
        rcfg.roi_half = 15;
        rcfg.search_half = 20;

        Vec3f result;
        {
            OutputGuard guard;
            result = roi_refine::refineROI(templ_img, scene_gray, sample_pts,
                                           initial_pose, rcfg);
        }

        // Convert center pose to user-origin pose
        float rr = -result[2] * (float)CV_PI / 180.0f;
        float ux = org_x - templ_w / 2.0f;
        float uy = org_y - templ_h / 2.0f;
        float res_x = result[0] + std::cos(rr) * ux - std::sin(rr) * uy;
        float res_y = result[1] + std::sin(rr) * ux + std::cos(rr) * uy;

        IterRecord rec;
        rec.iteration = n;
        rec.angle_gt = gt_angle;
        rec.angle_error = angle_err(result[2], gt_angle);
        rec.pos_error = pos_err(res_x, res_y, gt_x, gt_y);
        rec.rmse = 0;       // ROI does not expose RMSE
        rec.fitness = 0;    // ROI does not expose fitness
        records.push_back(rec);
    }
    return records;
}

// ============================================================
// Classify convergence behavior
// ============================================================

enum ConvergeBehavior { CONVERGED, PLATEAU, OSCILLATION, DIVERGED };

static const char* behavior_str(ConvergeBehavior b) {
    switch (b) {
        case CONVERGED:   return "CONVERGED";
        case PLATEAU:     return "PLATEAU";
        case OSCILLATION: return "OSCILLATION";
        case DIVERGED:    return "DIVERGED";
    }
    return "UNKNOWN";
}

static ConvergeBehavior classify_convergence(const std::vector<IterRecord>& recs,
                                              float conv_thresh = 0.1f) {
    if (recs.empty()) return DIVERGED;

    // Check if final error is below threshold
    float final_ae = recs.back().angle_error;
    float first_ae = recs.front().angle_error;

    // Detect divergence: final error > 2x initial error
    if (final_ae > first_ae * 2.0f && final_ae > 1.0f)
        return DIVERGED;

    // Detect oscillation: count sign changes in angle_error delta
    int sign_changes = 0;
    for (size_t i = 2; i < recs.size(); ++i) {
        float d1 = recs[i-1].angle_error - recs[i-2].angle_error;
        float d2 = recs[i].angle_error - recs[i-1].angle_error;
        if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0))
            sign_changes++;
    }
    if (sign_changes >= 3 && final_ae > conv_thresh)
        return OSCILLATION;

    // Check convergence
    if (final_ae <= conv_thresh)
        return CONVERGED;

    // Plateau: error stopped decreasing but did not reach threshold
    if (recs.size() >= 3) {
        float late_improvement = std::abs(recs.back().angle_error -
                                          recs[recs.size()-3].angle_error);
        if (late_improvement < 0.01f && final_ae > conv_thresh)
            return PLATEAU;
    }

    return PLATEAU;
}

static int iterations_to_threshold(const std::vector<IterRecord>& recs, float thresh) {
    for (auto& r : recs)
        if (r.angle_error < thresh) return r.iteration;
    return -1;  // never reached
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

    printf("===== 3c: REFINEMENT CONVERGENCE CURVES =====\n\n");

    // --- Create 200x200 L-shape template ---
    Mat templ200(200, 200, CV_8U, Scalar(0));
    draw_L(templ200, 100, 100, 0, 200);

    // Extract features
    sbm::FeatureSet feat;
    {
        OutputGuard guard;
        feat = sbm::extractFeatures(templ200);
    }
    feat.setOrigin(100, 75);

    const float org_x = 100, org_y = 75;

    // Build ICP template scene (once)
    icp_refine::EdgeScene templ_scene;
    {
        OutputGuard guard;
        templ_scene = icp_refine::buildTemplateScene(templ200, 20.0f);
    }

    // Build ROI sample points (once)
    std::vector<cv::Point2f> positions;
    std::vector<float> corner_scores;
    for (auto& rp : feat.refine_points) {
        positions.push_back(cv::Point2f(rp.px, rp.py));
        corner_scores.push_back(rp.cornerness);
    }
    auto sample_pts = roi_refine::selectCriticalPoints(
        positions, corner_scores, 20, feat.templ_width, feat.templ_height);

    printf("  Template: 200x200 L-shape, %d refine points, %d ROI sample points\n",
           (int)feat.refine_points.size(), (int)sample_pts.size());

    // Train matcher for coarse detection
    sbm::MatchConfig mcfg;
    mcfg.min_score = 30;
    mcfg.refine = sbm::RefineMode::None;
    mcfg.nms_radius = 50;
    mcfg.max_results = 1;

    sbm::ModelConfig modcfg;
    modcfg.angle = {0, 360, 1};

    sbm::ShapeMatcher matcher(mcfg);
    {
        OutputGuard guard;
        matcher.addModel("L", feat, modcfg);
    }

    // Scene parameters
    const int scene_w = 640, scene_h = 480;
    const int obj_cx = 320, obj_cy = 240;
    const int max_iters = 10;

    // Test angles
    float test_angles[] = {0, 15, 30, 45, 67, 90, 135, 180, 270};
    int num_angles = sizeof(test_angles) / sizeof(test_angles[0]);

    // Open CSV files
    FILE* icp_csv = fopen("output/diag_convergence_icp.csv", "w");
    FILE* roi_csv = fopen("output/diag_convergence_roi.csv", "w");
    if (!icp_csv || !roi_csv) {
        printf("ERROR: cannot create output CSV files\n");
        return 1;
    }

    fprintf(icp_csv, "iteration,angle_gt,method,angle_error,pos_error,rmse,fitness\n");
    fprintf(roi_csv, "iteration,angle_gt,method,angle_error,pos_error,rmse,fitness\n");

    // Summary storage
    struct Summary {
        float angle_gt;
        float coarse_ae, coarse_pe;
        int icp_conv_iter;   // iterations to < 0.1 deg, -1 if never
        int roi_conv_iter;
        float icp_final_ae, icp_final_pe;
        float roi_final_ae, roi_final_pe;
        ConvergeBehavior icp_behavior;
        ConvergeBehavior roi_behavior;
    };
    std::vector<Summary> summaries;

    // --- Run convergence tests ---
    for (int ai = 0; ai < num_angles; ++ai) {
        float gt_angle = test_angles[ai];

        // Create scene
        Mat scene(scene_h, scene_w, CV_8U, Scalar(0));
        place_object(templ200, scene, obj_cx, obj_cy, gt_angle);

        // Ground truth origin position
        float gt_x, gt_y;
        compute_gt_origin(obj_cx, obj_cy, gt_angle, org_x, org_y,
                          feat.templ_width, feat.templ_height, gt_x, gt_y);

        // Coarse match
        std::vector<sbm::MatchResult> results;
        {
            OutputGuard guard;
            results = matcher.match(scene);
        }

        if (results.empty()) {
            printf("  [%6.1f deg] NO COARSE MATCH -- skipping\n", gt_angle);
            continue;
        }

        auto& cr = results[0];
        float coarse_ae = angle_err(cr.angle, gt_angle);
        float coarse_pe = pos_err(cr.x, cr.y, gt_x, gt_y);

        // Compute center pose from coarse result for refinement
        // Coarse result reports user-origin position; convert to center pose
        float cr_rad = -cr.angle * (float)CV_PI / 180.0f;
        float ux = org_x - feat.templ_width / 2.0f;
        float uy = org_y - feat.templ_height / 2.0f;
        // result.x = center.x + cos(r)*ux - sin(r)*uy  =>  center.x = result.x - cos(r)*ux + sin(r)*uy
        float center_x = cr.x - std::cos(cr_rad) * ux + std::sin(cr_rad) * uy;
        float center_y = cr.y - std::sin(cr_rad) * ux - std::cos(cr_rad) * uy;

        // --- ICP convergence ---
        icp_refine::Pose2D init_icp(center_x, center_y, cr.angle);
        auto icp_recs = run_icp_convergence(
            templ_scene, feat.templ_width, feat.templ_height,
            scene, init_icp, gt_angle, gt_x, gt_y, org_x, org_y, max_iters);

        for (auto& r : icp_recs) {
            fprintf(icp_csv, "%d,%.1f,ICP,%.4f,%.4f,%.6f,%.4f\n",
                    r.iteration, r.angle_gt, r.angle_error, r.pos_error,
                    r.rmse, r.fitness);
        }

        // --- ROI convergence ---
        Vec3f init_roi(center_x, center_y, cr.angle);
        auto roi_recs = run_roi_convergence(
            templ200, scene, sample_pts, init_roi,
            gt_angle, gt_x, gt_y, org_x, org_y,
            feat.templ_width, feat.templ_height, max_iters);

        for (auto& r : roi_recs) {
            fprintf(roi_csv, "%d,%.1f,ROI,%.4f,%.4f,%.6f,%.4f\n",
                    r.iteration, r.angle_gt, r.angle_error, r.pos_error,
                    r.rmse, r.fitness);
        }

        // Summary
        Summary s;
        s.angle_gt = gt_angle;
        s.coarse_ae = coarse_ae;
        s.coarse_pe = coarse_pe;
        s.icp_conv_iter = iterations_to_threshold(icp_recs, 0.1f);
        s.roi_conv_iter = iterations_to_threshold(roi_recs, 0.1f);
        s.icp_final_ae = icp_recs.empty() ? -1 : icp_recs.back().angle_error;
        s.icp_final_pe = icp_recs.empty() ? -1 : icp_recs.back().pos_error;
        s.roi_final_ae = roi_recs.empty() ? -1 : roi_recs.back().angle_error;
        s.roi_final_pe = roi_recs.empty() ? -1 : roi_recs.back().pos_error;
        s.icp_behavior = classify_convergence(icp_recs, 0.1f);
        s.roi_behavior = classify_convergence(roi_recs, 0.1f);
        summaries.push_back(s);

        printf("  [%6.1f deg] coarse: ae=%.2f pe=%.2f | ICP: ae=%.3f pe=%.3f (%s, iter=%d) | ROI: ae=%.3f pe=%.3f (%s, iter=%d)\n",
               gt_angle, coarse_ae, coarse_pe,
               s.icp_final_ae, s.icp_final_pe,
               behavior_str(s.icp_behavior), s.icp_conv_iter,
               s.roi_final_ae, s.roi_final_pe,
               behavior_str(s.roi_behavior), s.roi_conv_iter);
    }

    fclose(icp_csv);
    fclose(roi_csv);

    // --- Print summary table ---
    printf("\n===== CONVERGENCE SUMMARY =====\n\n");
    printf("%-8s  %-10s %-10s | %-12s %-10s %-10s %-12s | %-12s %-10s %-10s %-12s\n",
           "GT_Ang", "Coarse_AE", "Coarse_PE",
           "ICP_Behav", "ICP_AE", "ICP_PE", "ICP_Conv@",
           "ROI_Behav", "ROI_AE", "ROI_PE", "ROI_Conv@");
    printf("%-8s  %-10s %-10s | %-12s %-10s %-10s %-12s | %-12s %-10s %-10s %-12s\n",
           "------", "---------", "---------",
           "----------", "--------", "--------", "----------",
           "----------", "--------", "--------", "----------");

    int icp_converged = 0, roi_converged = 0;
    int icp_plateau = 0, roi_plateau = 0;
    int icp_oscillate = 0, roi_oscillate = 0;
    int icp_diverged = 0, roi_diverged = 0;

    for (auto& s : summaries) {
        char icp_conv_str[16], roi_conv_str[16];
        if (s.icp_conv_iter > 0) snprintf(icp_conv_str, sizeof(icp_conv_str), "iter %d", s.icp_conv_iter);
        else snprintf(icp_conv_str, sizeof(icp_conv_str), "never");
        if (s.roi_conv_iter > 0) snprintf(roi_conv_str, sizeof(roi_conv_str), "iter %d", s.roi_conv_iter);
        else snprintf(roi_conv_str, sizeof(roi_conv_str), "never");

        printf("%-8.1f  %-10.3f %-10.3f | %-12s %-10.4f %-10.4f %-12s | %-12s %-10.4f %-10.4f %-12s\n",
               s.angle_gt, s.coarse_ae, s.coarse_pe,
               behavior_str(s.icp_behavior), s.icp_final_ae, s.icp_final_pe, icp_conv_str,
               behavior_str(s.roi_behavior), s.roi_final_ae, s.roi_final_pe, roi_conv_str);

        switch (s.icp_behavior) {
            case CONVERGED: icp_converged++; break;
            case PLATEAU: icp_plateau++; break;
            case OSCILLATION: icp_oscillate++; break;
            case DIVERGED: icp_diverged++; break;
        }
        switch (s.roi_behavior) {
            case CONVERGED: roi_converged++; break;
            case PLATEAU: roi_plateau++; break;
            case OSCILLATION: roi_oscillate++; break;
            case DIVERGED: roi_diverged++; break;
        }
    }

    printf("\n  ICP: %d converged, %d plateau, %d oscillation, %d diverged (of %d)\n",
           icp_converged, icp_plateau, icp_oscillate, icp_diverged, (int)summaries.size());
    printf("  ROI: %d converged, %d plateau, %d oscillation, %d diverged (of %d)\n",
           roi_converged, roi_plateau, roi_oscillate, roi_diverged, (int)summaries.size());

    // Highlight any problematic cases
    bool any_issues = false;
    for (auto& s : summaries) {
        if (s.icp_behavior == DIVERGED) {
            if (!any_issues) { printf("\n  *** HIGHLIGHTED CASES ***\n"); any_issues = true; }
            printf("  [!] ICP DIVERGED at %.1f deg: ae=%.3f pe=%.3f\n",
                   s.angle_gt, s.icp_final_ae, s.icp_final_pe);
        }
        if (s.roi_behavior == DIVERGED) {
            if (!any_issues) { printf("\n  *** HIGHLIGHTED CASES ***\n"); any_issues = true; }
            printf("  [!] ROI DIVERGED at %.1f deg: ae=%.3f pe=%.3f\n",
                   s.angle_gt, s.roi_final_ae, s.roi_final_pe);
        }
        if (s.icp_behavior == OSCILLATION) {
            if (!any_issues) { printf("\n  *** HIGHLIGHTED CASES ***\n"); any_issues = true; }
            printf("  [~] ICP OSCILLATION at %.1f deg: ae=%.3f pe=%.3f\n",
                   s.angle_gt, s.icp_final_ae, s.icp_final_pe);
        }
        if (s.roi_behavior == OSCILLATION) {
            if (!any_issues) { printf("\n  *** HIGHLIGHTED CASES ***\n"); any_issues = true; }
            printf("  [~] ROI OSCILLATION at %.1f deg: ae=%.3f pe=%.3f\n",
                   s.angle_gt, s.roi_final_ae, s.roi_final_pe);
        }
        if (s.icp_behavior == PLATEAU && s.icp_final_ae > 1.0f) {
            if (!any_issues) { printf("\n  *** HIGHLIGHTED CASES ***\n"); any_issues = true; }
            printf("  [P] ICP PLATEAU at %.1f deg: ae=%.3f (>1 deg residual)\n",
                   s.angle_gt, s.icp_final_ae);
        }
        if (s.roi_behavior == PLATEAU && s.roi_final_ae > 1.0f) {
            if (!any_issues) { printf("\n  *** HIGHLIGHTED CASES ***\n"); any_issues = true; }
            printf("  [P] ROI PLATEAU at %.1f deg: ae=%.3f (>1 deg residual)\n",
                   s.angle_gt, s.roi_final_ae);
        }
    }
    if (!any_issues)
        printf("\n  No divergence, oscillation, or high-residual plateau cases detected.\n");

    printf("\nCSV written: output/diag_convergence_icp.csv\n");
    printf("CSV written: output/diag_convergence_roi.csv\n");
    printf("\n===== CONVERGENCE DIAGNOSTICS COMPLETE =====\n");
    return 0;
}
