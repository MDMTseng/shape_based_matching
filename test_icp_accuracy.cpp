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

    // Template edge points with normals for ICP
    auto model_edges = icp_refine::extractModelEdges(templ);
    printf("Template edge points: %d\n\n", (int)model_edges.size());

    // Test: single object at center, sweep angle 0-360 in 1-degree steps
    int cx = W/2, cy = H/2;
    printf("GT     coarse: angle_err  pos_err(x,y) dist  |  icp: angle_err  pos_err(x,y) dist\n");
    printf("-----  ----------------------------------------  ----------------------------------------\n");

    int total = 0, good_coarse = 0, good_icp = 0;
    float max_err_coarse = 0, max_err_icp = 0;
    float sum_err_coarse = 0, sum_err_icp = 0;
    int good_pos_coarse = 0, good_pos_icp = 0;
    float max_pos_coarse = 0, max_pos_icp = 0;
    float sum_pos_coarse = 0, sum_pos_icp = 0;
    float sum_signed_angle = 0, sum_signed_dx = 0, sum_signed_dy = 0;

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

        const char* status = "";
        if (std::abs(icp_err) > 5.0f) status = "** BAD";
        else if (std::abs(icp_err) > 2.0f) status = "* WARN";

        printf("%-5d  coarse: ang=%+5.1f pos=(%+5.1f,%+5.1f) d=%.1f  |  "
               "icp: ang=%+5.1f pos=(%+5.1f,%+5.1f) d=%.1f  %s\n",
               gt_angle,
               coarse_err, coarse_dx, coarse_dy, coarse_dist,
               icp_err, icp_dx, icp_dy, icp_dist, status);

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

    printf("\n--- Orientation Summary (%d angles) ---\n", total);
    printf("  Coarse: mean=%.1f deg  max=%.1f deg  <=2deg: %d/%d (%.0f%%)\n",
           sum_err_coarse / total, max_err_coarse, good_coarse, total,
           100.0f * good_coarse / total);
    printf("  ICP:    mean=%.1f deg  max=%.1f deg  <=2deg: %d/%d (%.0f%%)\n",
           sum_err_icp / total, max_err_icp, good_icp, total,
           100.0f * good_icp / total);
    printf("\n--- Translation Summary (%d angles) ---\n", total);
    printf("  Coarse: mean=%.1f px  max=%.1f px  <=2px: %d/%d (%.0f%%)\n",
           sum_pos_coarse / total, max_pos_coarse, good_pos_coarse, total,
           100.0f * good_pos_coarse / total);
    printf("  ICP:    mean=%.1f px  max=%.1f px  <=2px: %d/%d (%.0f%%)\n",
           sum_pos_icp / total, max_pos_icp, good_pos_icp, total,
           100.0f * good_pos_icp / total);

    printf("\n--- Coarse Bias Analysis ---\n");
    printf("  Angle: mean_signed=%+.2f deg\n", sum_signed_angle / total);
    printf("  Pos X: mean_signed=%+.2f px\n", sum_signed_dx / total);
    printf("  Pos Y: mean_signed=%+.2f px\n", sum_signed_dy / total);

    // Check if angle bias is consistent: bucket by angle quadrant
    printf("\n--- Angle Bias by Quadrant ---\n");
    float qsum[4] = {}, qcnt[4] = {};
    // Re-run to collect per-quadrant stats
    for (int gt_angle = 0; gt_angle < 360; gt_angle += 3) {
        Mat scene2(H, W, CV_8U, Scalar(50));
        draw_L(scene2, cx, cy, gt_angle, 200);
        Mat padded2 = pad16(scene2);
        auto m2 = det.match(padded2, threshold);
        if (m2.empty()) continue;
        float ce = m2[0].template_id * angle_step - gt_angle;
        if (ce > 180) ce -= 360; if (ce < -180) ce += 360;
        int q = gt_angle / 90;
        qsum[q] += ce; qcnt[q]++;
    }
    for (int q = 0; q < 4; ++q)
        printf("  Q%d (%3d-%3d): mean_angle_err=%+.1f deg\n",
               q, q*90, (q+1)*90-1, qcnt[q] > 0 ? qsum[q]/qcnt[q] : 0);

    // Output images for outlier cases (icp_dist > 5px or icp_err > 2deg)
    printf("\n--- Generating outlier images ---\n");
    for (int gt_angle = 0; gt_angle < 360; gt_angle += 3) {
        Mat scene(H, W, CV_8U, Scalar(50));
        draw_L(scene, cx, cy, gt_angle, 200);
        Mat padded = pad16(scene);

        auto matches2 = det.match(padded, threshold);
        if (matches2.empty()) continue;
        auto& m2 = matches2[0];
        float coarse_angle2 = m2.template_id * angle_step;

        Mat scene_smooth2, scene_dx2, scene_dy2;
        GaussianBlur(scene, scene_smooth2, Size(7, 7), 0);
        Sobel(scene_smooth2, scene_dx2, CV_16S, 1, 0, 3);
        Sobel(scene_smooth2, scene_dy2, CV_16S, 0, 1, 3);

        auto& ti2 = det.getTemplates(m2.class_id, m2.template_id);
        float mcx2 = m2.x + TW/2.0f - ti2[0].tl_x;
        float mcy2 = m2.y + TW/2.0f - ti2[0].tl_y;

        icp_refine::ICPConfig cfg2;
        cfg2.max_iterations = 30;
        cfg2.max_dist = 10.0f;

        icp_refine::Pose2D best2;
        best2.fitness = -1;
        for (float ao : {0.0f, angle_step, angle_step*2, angle_step*3}) {
            icp_refine::Pose2D init2(mcx2, mcy2, coarse_angle2 + ao);
            auto r2 = icp_refine::refineWithNormals(
                model_edges, scene_dx2, scene_dy2, init2, TW, 20, cfg2);
            if (r2.fitness > best2.fitness) best2 = r2;
        }
        auto& ref2 = best2;

        float a_err = ref2.angle - gt_angle;
        if (a_err > 180) a_err -= 360;
        if (a_err < -180) a_err += 360;
        float pdx = ref2.x - cx, pdy = ref2.y - cy;
        float pdist = std::sqrt(pdx*pdx + pdy*pdy);

        if (std::abs(a_err) <= 2.0f && pdist <= 2.0f) continue;

        // Draw visualization
        Mat vis;
        // Crop 200x200 around object
        int crop = 100;
        int rx = std::max(0, cx - crop), ry = std::max(0, cy - crop);
        int rw = std::min(W - rx, 2*crop), rh = std::min(H - ry, 2*crop);
        Mat cropped = scene(Rect(rx, ry, rw, rh));
        cvtColor(cropped, vis, cv::COLOR_GRAY2BGR);

        // GT center (green cross)
        int gx = cx - rx, gy = cy - ry;
        line(vis, Point(gx-8,gy), Point(gx+8,gy), Scalar(0,255,0), 2);
        line(vis, Point(gx,gy-8), Point(gx,gy+8), Scalar(0,255,0), 2);

        // GT arrow (green)
        double gt_rad = gt_angle * CV_PI / 180.0;
        arrowedLine(vis, Point(gx,gy),
                    Point(gx+(int)(40*cos(gt_rad)), gy+(int)(40*sin(gt_rad))),
                    Scalar(0,255,0), 2, cv::LINE_AA, 0, 0.3);

        // Coarse center + arrow (yellow)
        int ccx = (int)(mcx2 - rx), ccy = (int)(mcy2 - ry);
        double c_rad = coarse_angle2 * CV_PI / 180.0;
        arrowedLine(vis, Point(ccx,ccy),
                    Point(ccx+(int)(35*cos(c_rad)), ccy+(int)(35*sin(c_rad))),
                    Scalar(0,255,255), 2, cv::LINE_AA, 0, 0.3);

        // ICP center + arrow (red)
        int icx = (int)(ref2.x - rx), icy = (int)(ref2.y - ry);
        double i_rad = ref2.angle * CV_PI / 180.0;
        arrowedLine(vis, Point(icx,icy),
                    Point(icx+(int)(35*cos(i_rad)), icy+(int)(35*sin(i_rad))),
                    Scalar(0,0,255), 2, cv::LINE_AA, 0, 0.3);

        // Text overlay
        char buf[128];
        snprintf(buf, sizeof(buf), "GT=%d  coarse=%.0f(%.1f)  icp=%.1f(%.1f)  pos=%.1fpx",
                 gt_angle, coarse_angle2, coarse_angle2-gt_angle,
                 ref2.angle, a_err, pdist);
        putText(vis, buf, Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0,0,0), 2);
        putText(vis, buf, Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.35, Scalar(255,255,255), 1);

        // Legend
        putText(vis, "green=GT  yellow=coarse  red=ICP", Point(5, rh-5),
                FONT_HERSHEY_SIMPLEX, 0.3, Scalar(200,200,200), 1);

        char fname[64];
        snprintf(fname, sizeof(fname), "outlier_%03d.jpg", gt_angle);
        string path = "C:/Users/TRS001/Documents/workspace/templmatch/test_imgs/" + string(fname);
        imwrite(path, vis);
        printf("  Saved %s (angle_err=%.1f pos_err=%.1f)\n", fname, a_err, pdist);
    }

    return 0;
}
