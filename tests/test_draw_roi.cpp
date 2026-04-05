/// @file test_draw_roi.cpp
/// @brief Draw template with ROI boxes and lock info, save as image.

#include "shape_matcher.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cstdio>

using namespace cv;
using namespace sbm;

int main() {
    // Same triangle as test_regression section 18
    int TW = 200;
    Mat templ_tri(TW, TW, CV_8U, Scalar(0));
    {
        std::vector<Point> pts = {Point(100, 20), Point(20, 180), Point(180, 180)};
        fillConvexPoly(templ_tri, pts, Scalar(200));
    }
    Mat mask = Mat::ones(TW, TW, CV_8U) * 255;

    auto fs = extractFeatures(templ_tri, mask, 128);
    auto opt_points = fs.selectOptimizedPoints(8);
    auto ca = fs.analyzeConstraints();

    printf("Triangle template: %dx%d, %d refine points, %d selected\n",
           TW, TW, (int)fs.refine_points.size(), (int)opt_points.size());
    printf("Constraint analysis: %s\n", ca.summary.c_str());

    // Draw on color image
    Mat vis;
    cvtColor(templ_tri, vis, COLOR_GRAY2BGR);

    float tcx = TW / 2.0f, tcy = TW / 2.0f;
    int roi_half = 15;

    // Draw all refine points as small dots
    for (auto& rp : fs.refine_points) {
        int px = (int)(rp.px + tcx + 0.5f), py = (int)(rp.py + tcy + 0.5f);
        circle(vis, Point(px, py), 1, Scalar(80, 80, 80), -1);
    }

    // Draw selected ROI boxes
    for (int i = 0; i < (int)opt_points.size(); i++) {
        auto& pt = opt_points[i];
        int px = (int)(pt.x + tcx + 0.5f), py = (int)(pt.y + tcy + 0.5f);

        // ROI box
        Scalar color = (i < (int)ca.points.size() && ca.points[i].is_corner)
            ? Scalar(0, 255, 0)   // green = 2D lock
            : Scalar(0, 128, 255); // orange = 1D edge

        rectangle(vis, Rect(px - roi_half, py - roi_half, 2*roi_half+1, 2*roi_half+1),
                  color, 2);

        // Draw normal direction
        if (i < (int)ca.points.size()) {
            auto& pi = ca.points[i];
            int nx = (int)(pi.normal.x * 12), ny = (int)(pi.normal.y * 12);
            arrowedLine(vis, Point(px, py), Point(px + nx, py + ny), color, 2);

            // Label with lock values
            char buf[64];
            snprintf(buf, sizeof(buf), "#%d L:%.2f/%.2f", i,
                     pi.lock_major, pi.lock_minor);
            putText(vis, buf, Point(px - roi_half, py - roi_half - 5),
                    FONT_HERSHEY_SIMPLEX, 0.35, color, 1);
        }

        // Point marker
        circle(vis, Point(px, py), 3, Scalar(0, 0, 255), -1);
    }

    // Add summary text
    char summary[256];
    snprintf(summary, sizeof(summary),
             "sigma: theta=%.3f deg  tx=%.3f px  ty=%.3f px  |  %d pts (%dC/%dE)",
             ca.sigma_theta, ca.sigma_tx, ca.sigma_ty,
             ca.num_points, ca.num_corners, ca.num_edges);
    putText(vis, summary, Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);

    // Scale up for visibility
    Mat vis_big;
    resize(vis, vis_big, Size(), 3, 3, INTER_NEAREST);

    imwrite("output/triangle_roi_boxes.png", vis_big);
    printf("Saved: output/triangle_roi_boxes.png\n");

    // Also do L-shape for comparison
    Mat templ_L(TW, TW, CV_8U, Scalar(0));
    // L-shape from test_regression
    for (double ly = -30; ly <= 30; ly += 0.5)
        for (double lx = -10; lx <= 10; lx += 0.5) {
            int px = TW/2 + (int)(lx + 0.5), py = TW/2 + (int)(ly + 0.5);
            if (px >= 0 && px < TW && py >= 0 && py < TW) templ_L.at<uchar>(py, px) = 200;
        }
    for (double ly = 10; ly <= 30; ly += 0.5)
        for (double lx = 10; lx <= 40; lx += 0.5) {
            int px = TW/2 + (int)(lx + 0.5), py = TW/2 + (int)(ly + 0.5);
            if (px >= 0 && px < TW && py >= 0 && py < TW) templ_L.at<uchar>(py, px) = 200;
        }

    auto fs_L = extractFeatures(templ_L, mask, 128);
    auto opt_L = fs_L.selectOptimizedPoints(8);
    auto ca_L = fs_L.analyzeConstraints();

    Mat vis_L;
    cvtColor(templ_L, vis_L, COLOR_GRAY2BGR);
    for (auto& rp : fs_L.refine_points) {
        int px = (int)(rp.px + tcx + 0.5f), py = (int)(rp.py + tcy + 0.5f);
        circle(vis_L, Point(px, py), 1, Scalar(80, 80, 80), -1);
    }
    for (int i = 0; i < (int)opt_L.size(); i++) {
        auto& pt = opt_L[i];
        int px = (int)(pt.x + tcx + 0.5f), py = (int)(pt.y + tcy + 0.5f);
        Scalar color = (i < (int)ca_L.points.size() && ca_L.points[i].is_corner)
            ? Scalar(0, 255, 0) : Scalar(0, 128, 255);
        rectangle(vis_L, Rect(px - roi_half, py - roi_half, 2*roi_half+1, 2*roi_half+1), color, 2);
        if (i < (int)ca_L.points.size()) {
            auto& pi = ca_L.points[i];
            int nx = (int)(pi.normal.x * 12), ny = (int)(pi.normal.y * 12);
            arrowedLine(vis_L, Point(px, py), Point(px + nx, py + ny), color, 2);
            char buf[64];
            snprintf(buf, sizeof(buf), "#%d L:%.2f/%.2f", i, pi.lock_major, pi.lock_minor);
            putText(vis_L, buf, Point(px - roi_half, py - roi_half - 5),
                    FONT_HERSHEY_SIMPLEX, 0.35, color, 1);
        }
        circle(vis_L, Point(px, py), 3, Scalar(0, 0, 255), -1);
    }
    snprintf(summary, sizeof(summary),
             "sigma: theta=%.3f deg  tx=%.3f px  ty=%.3f px  |  %d pts (%dC/%dE)",
             ca_L.sigma_theta, ca_L.sigma_tx, ca_L.sigma_ty,
             ca_L.num_points, ca_L.num_corners, ca_L.num_edges);
    putText(vis_L, summary, Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);

    Mat vis_L_big;
    resize(vis_L, vis_L_big, Size(), 3, 3, INTER_NEAREST);
    imwrite("output/lshape_roi_boxes.png", vis_L_big);
    printf("Saved: output/lshape_roi_boxes.png\n");

    printf("\nL-shape: %s\n", ca_L.summary.c_str());
    printf("Triangle: %s\n", ca.summary.c_str());

    return 0;
}
