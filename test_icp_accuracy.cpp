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

    // Template edge points for ICP
    vector<Point2f> templ_edge_pts;
    {
        Mat ts, tdx, tdy, te;
        GaussianBlur(templ, ts, Size(5,5), 0);
        Sobel(ts, tdx, CV_16S, 1, 0, 3);
        Sobel(ts, tdy, CV_16S, 0, 1, 3);
        Canny(tdx, tdy, te, 30, 60);
        for (int r = 0; r < TW; ++r)
            for (int c = 0; c < TW; ++c)
                if (te.at<uchar>(r, c) > 0)
                    templ_edge_pts.push_back(Point2f((float)(c - TW/2), (float)(r - TW/2)));
        printf("Template edge points: %d\n\n", (int)templ_edge_pts.size());
    }

    // Test: single object at center, sweep angle 0-360 in 1-degree steps
    int cx = W/2, cy = H/2;
    printf("%-8s  %-8s  %-10s  %-10s  %-8s  %-8s\n",
           "GT", "Coarse", "CoarseErr", "Refined", "ICPErr", "Status");
    printf("%-8s  %-8s  %-10s  %-10s  %-8s  %-8s\n",
           "---", "------", "---------", "-------", "------", "------");

    int total = 0, good_coarse = 0, good_icp = 0;
    float max_err_coarse = 0, max_err_icp = 0;
    float sum_err_coarse = 0, sum_err_icp = 0;

    for (int gt_angle = 0; gt_angle < 360; gt_angle += 3) {
        // Draw single object at known angle
        Mat scene(H, W, CV_8U, Scalar(50));
        draw_L(scene, cx, cy, gt_angle, 200);
        Mat padded = pad16(scene);

        // Match
        auto matches = det.match(padded, threshold);
        if (matches.empty()) {
            printf("%-8d  MISS\n", gt_angle);
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
        float mcx = m.x + tmpl_info[0].width / 2.0f;
        float mcy = m.y + tmpl_info[0].height / 2.0f;

        icp_refine::ICPConfig cfg;
        cfg.max_iterations = 30;
        cfg.max_dist = 10.0f;

        icp_refine::Pose2D init_pose(mcx, mcy, coarse_angle);
        auto refined = icp_refine::refineLocal(
            templ_edge_pts, scene_dx, scene_dy, init_pose, TW, 20, cfg);

        // Compute angular errors (handle wraparound)
        float coarse_err = coarse_angle - gt_angle;
        if (coarse_err > 180) coarse_err -= 360;
        if (coarse_err < -180) coarse_err += 360;

        float icp_err = refined.angle - gt_angle;
        if (icp_err > 180) icp_err -= 360;
        if (icp_err < -180) icp_err += 360;

        const char* status = "";
        if (std::abs(icp_err) > 5.0f) status = "** BAD";
        else if (std::abs(icp_err) > 2.0f) status = "* WARN";

        printf("%-8d  %-8.1f  %-+10.1f  %-10.1f  %-+8.1f  %s\n",
               gt_angle, coarse_angle, coarse_err, refined.angle, icp_err, status);

        total++;
        sum_err_coarse += std::abs(coarse_err);
        sum_err_icp += std::abs(icp_err);
        max_err_coarse = std::max(max_err_coarse, std::abs(coarse_err));
        max_err_icp = std::max(max_err_icp, std::abs(icp_err));
        if (std::abs(coarse_err) <= 2.0f) good_coarse++;
        if (std::abs(icp_err) <= 2.0f) good_icp++;
    }

    printf("\n--- Summary (%d angles) ---\n", total);
    printf("  Coarse: mean=%.1f  max=%.1f  <=2deg: %d/%d (%.0f%%)\n",
           sum_err_coarse / total, max_err_coarse, good_coarse, total,
           100.0f * good_coarse / total);
    printf("  ICP:    mean=%.1f  max=%.1f  <=2deg: %d/%d (%.0f%%)\n",
           sum_err_icp / total, max_err_icp, good_icp, total,
           100.0f * good_icp / total);

    return 0;
}
