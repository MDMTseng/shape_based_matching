// ICP accuracy test: single object at many angles, measure orientation error.

#include "line2Dup.h"
#include "icp_refine.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>

using namespace cv;
using namespace std;

static void add_noise(Mat& img, double sigma) {
    Mat noise(img.size(), CV_64F);
    RNG rng(42);
    rng.fill(noise, RNG::NORMAL, 0, sigma);
    Mat result;
    img.convertTo(result, CV_64F);
    result += noise;
    result.convertTo(img, CV_8U);
}

static void draw_L(Mat& img, int cx, int cy, double angle, int color, double scale = 2.0) {
    double rad = angle * CV_PI / 180.0;
    double cs = cos(rad), sn = sin(rad);
    for (double ly = -15*scale; ly <= 15*scale; ly += 0.5)
        for (double lx = -5*scale; lx <= 5*scale; lx += 0.5) {
            int px = cx + (int)(lx*cs - ly*sn + 0.5);
            int py = cy + (int)(lx*sn + ly*cs + 0.5);
            if (px >= 0 && px < img.cols && py >= 0 && py < img.rows)
                img.at<uchar>(py, px) = (uchar)color;
        }
    for (double ly = 5*scale; ly <= 15*scale; ly += 0.5)
        for (double lx = 5*scale; lx <= 20*scale; lx += 0.5) {
            int px = cx + (int)(lx*cs - ly*sn + 0.5);
            int py = cy + (int)(lx*sn + ly*cs + 0.5);
            if (px >= 0 && px < img.cols && py >= 0 && py < img.rows)
                img.at<uchar>(py, px) = (uchar)color;
        }
}

static Mat pad16(const Mat& img) {
    int pw = (img.cols + 15) & ~15;
    int ph = (img.rows + 15) & ~15;
    if (pw != img.cols || ph != img.rows) {
        Mat padded;
        copyMakeBorder(img, padded, 0, ph - img.rows, 0, pw - img.cols,
                       BORDER_CONSTANT, Scalar(0));
        return padded;
    }
    return img;
}

int main() {
    printf("================================================================\n");
    printf("  ICP Orientation Accuracy Test\n");
    printf("================================================================\n\n");

    const int W = 640, H = 480;
    const int TW = 80;
    const float threshold = 50.0f;
    const float angle_step = 2.0f;

    // Build template
    Mat templ(TW, TW, CV_8U, Scalar(0));
    draw_L(templ, TW/2, TW/2, 0, 200);
    Mat mask_t = Mat::ones(TW, TW, CV_8U) * 255;

    // Detector with 2-degree step
    line2Dup::Detector det(128, {4, 8}, 30, 60);
    for (int angle = 0; angle < 360; angle += (int)angle_step) {
        Mat rot_templ, rot_mask;
        Mat M = getRotationMatrix2D(Point2f(TW/2.0f, TW/2.0f), -angle, 1.0);
        warpAffine(templ, rot_templ, M, Size(TW, TW));
        warpAffine(mask_t, rot_mask, M, Size(TW, TW));
        det.addTemplate(rot_templ, "L", rot_mask);
    }
    int ntmpl = det.numTemplates("L");
    printf("Templates: %d (step=%.0f deg)\n", ntmpl, angle_step);

    // Template edge points with normals for ICP
    auto model_edges = icp_refine::extractModelEdges(templ);
    printf("Template edge points: %d\n\n", (int)model_edges.size());

    // Test: single object at center, sweep angle 0-360 in 1-degree steps
    int cx = W/2, cy = H/2;
    // Test conditions: clean, noise, blur, combined
    struct Condition {
        const char* name;
        double noise_sigma;
        int blur_ksize;
    };
    Condition conditions[] = {
        {"clean",      0,  0},
        {"noise_s10", 10,  0},
        {"noise_s30", 30,  0},
        {"noise_s50", 50,  0},
        {"blur_k5",    0,  5},
        {"blur_k11",   0, 11},
        {"blur_k21",   0, 21},
        {"noise+blur", 20, 5},
    };

    for (auto& cond : conditions) {
    printf("\n=== %s ===\n", cond.name);

    int total = 0, good_coarse = 0, good_icp = 0;
    float max_err_coarse = 0, max_err_icp = 0;
    float sum_err_coarse = 0, sum_err_icp = 0;
    int good_pos_coarse = 0, good_pos_icp = 0;
    float max_pos_coarse = 0, max_pos_icp = 0;
    float sum_pos_coarse = 0, sum_pos_icp = 0;
    float sum_signed_angle = 0, sum_signed_dx = 0, sum_signed_dy = 0;
    int miss_count = 0;

    for (int gt_angle = 0; gt_angle < 360; gt_angle += 3) {
        Mat scene(H, W, CV_8U, Scalar(50));
        draw_L(scene, cx, cy, gt_angle, 200);
        if (cond.noise_sigma > 0) add_noise(scene, cond.noise_sigma);
        if (cond.blur_ksize > 0)
            GaussianBlur(scene, scene, Size(cond.blur_ksize, cond.blur_ksize), 0);
        Mat padded = pad16(scene);

        // Match
        auto matches = det.match(padded, threshold);
        if (matches.empty()) {
            miss_count++;
            continue;
        }

        // Best match
        auto& m = matches[0];
        float coarse_angle = m.template_id * angle_step;

        // ICP refine
        Mat scene_smooth, scene_dx, scene_dy;
        GaussianBlur(scene, scene_smooth, Size(7, 7), 0);
        Sobel(scene_smooth, scene_dx, CV_16S, 1, 0, 3);
        Sobel(scene_smooth, scene_dy, CV_16S, 0, 1, 3);

        auto& tmpl_info = det.getTemplates(m.class_id, m.template_id);
        float mcx = m.x + TW/2.0f - tmpl_info[0].tl_x;
        float mcy = m.y + TW/2.0f - tmpl_info[0].tl_y;

        icp_refine::ICPConfig cfg;
        cfg.max_iterations = 30;
        cfg.max_dist = 10.0f;
        cfg.point_to_point_weight = 0.1f;

        // Single-start ICP with normal compatibility filtering.
        // The normal check prevents wrong convergence, so multi-start
        // is no longer needed for correctness.
        icp_refine::Pose2D init_pose(mcx, mcy, coarse_angle);
        auto refined = icp_refine::refineWithNormals(
            model_edges, scene_dx, scene_dy, init_pose, TW, 20, cfg);

        // Compute angular errors (handle wraparound)
        float coarse_err = coarse_angle - gt_angle;
        if (coarse_err > 180) coarse_err -= 360;
        if (coarse_err < -180) coarse_err += 360;

        float icp_err = refined.angle - gt_angle;
        if (icp_err > 180) icp_err -= 360;
        if (icp_err < -180) icp_err += 360;

        // Translation errors
        float coarse_dx = mcx - cx, coarse_dy = mcy - cy;
        float coarse_dist = std::sqrt(coarse_dx*coarse_dx + coarse_dy*coarse_dy);
        float icp_dx = refined.x - cx, icp_dy = refined.y - cy;
        float icp_dist = std::sqrt(icp_dx*icp_dx + icp_dy*icp_dy);

        // Only print details for outliers
        if (std::abs(icp_err) > 2.0f || icp_dist > 2.0f) {
            printf("  %-5d  coarse: ang=%+5.1f d=%.1f  |  icp: ang=%+5.1f d=%.1f  *\n",
                   gt_angle, coarse_err, coarse_dist, icp_err, icp_dist);
        }

        total++;
        sum_err_coarse += std::abs(coarse_err);
        sum_err_icp += std::abs(icp_err);
        max_err_coarse = std::max(max_err_coarse, std::abs(coarse_err));
        max_err_icp = std::max(max_err_icp, std::abs(icp_err));
        if (std::abs(coarse_err) <= 2.0f) good_coarse++;
        if (std::abs(icp_err) <= 2.0f) good_icp++;

        sum_pos_coarse += coarse_dist;
        sum_pos_icp += icp_dist;
        max_pos_coarse = std::max(max_pos_coarse, coarse_dist);
        max_pos_icp = std::max(max_pos_icp, icp_dist);
        if (coarse_dist <= 2.0f) good_pos_coarse++;
        if (icp_dist <= 2.0f) good_pos_icp++;

        sum_signed_angle += coarse_err;
        sum_signed_dx += coarse_dx;
        sum_signed_dy += coarse_dy;
    }

    if (total == 0) { printf("  ALL MISSED\n"); continue; }
    printf("  Misses: %d/%d\n", miss_count, total + miss_count);
    printf("  Angle:  coarse mean=%.1f max=%.1f <=2deg:%d/%d(%.0f%%)  |  "
           "icp mean=%.1f max=%.1f <=2deg:%d/%d(%.0f%%)\n",
           sum_err_coarse/total, max_err_coarse, good_coarse, total,
           100.0f*good_coarse/total,
           sum_err_icp/total, max_err_icp, good_icp, total,
           100.0f*good_icp/total);
    printf("  Pos:    coarse mean=%.1f max=%.1f <=2px:%d/%d(%.0f%%)  |  "
           "icp mean=%.1f max=%.1f <=2px:%d/%d(%.0f%%)\n",
           sum_pos_coarse/total, max_pos_coarse, good_pos_coarse, total,
           100.0f*good_pos_coarse/total,
           sum_pos_icp/total, max_pos_icp, good_pos_icp, total,
           100.0f*good_pos_icp/total);

    } // end conditions loop

    return 0;
}
