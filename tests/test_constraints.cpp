/// @file test_constraints.cpp
/// @brief Geometric constraint quality analysis for different template shapes.

#include "shape_matcher.h"
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cmath>

using namespace cv;
using namespace sbm;

// Draw L-shape
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

// Draw rectangle
static void draw_rect(Mat& img, int cx, int cy, int w, int h, int color) {
    rectangle(img, Rect(cx-w/2, cy-h/2, w, h), Scalar(color), -1);
}

// Draw triangle
static void draw_triangle(Mat& img, int cx, int cy, int size, int color) {
    std::vector<Point> pts = {
        Point(cx, cy - size),
        Point(cx - size, cy + size/2),
        Point(cx + size/2, cy + size/2)
    };
    fillConvexPoly(img, pts, Scalar(color));
}

// Draw thin line
static void draw_line(Mat& img, int cx, int cy, int length, int color) {
    line(img, Point(cx - length/2, cy), Point(cx + length/2, cy), Scalar(color), 3);
}

// Draw circle
static void draw_circle(Mat& img, int cx, int cy, int radius, int color) {
    circle(img, Point(cx, cy), radius, Scalar(color), -1);
}

static void printAnalysis(const char* name, const FeatureSet::ConstraintAnalysis& ca) {
    printf("\n===== %s =====\n", name);
    printf("  %s\n", ca.summary.c_str());
    printf("\n  Information eigenvalues: [%.2f, %.2f, %.2f]\n",
           ca.info_eigenvalues[0], ca.info_eigenvalues[1], ca.info_eigenvalues[2]);
    printf("  Condition number:  %.1f  %s\n", ca.condition_number,
           ca.condition_number < 10 ? "(good)" :
           ca.condition_number < 100 ? "(moderate)" : "(POOR - near degenerate)");
    printf("\n  Per-DOF uncertainty (per 1px matching noise):\n");
    printf("    sigma_theta = %.4f deg\n", ca.sigma_theta);
    printf("    sigma_tx    = %.4f px\n", ca.sigma_tx);
    printf("    sigma_ty    = %.4f px\n", ca.sigma_ty);
    printf("\n  Position error ellipse:\n");
    printf("    %.4f x %.4f px @ %.0f deg\n",
           ca.ellipse_major, ca.ellipse_minor, ca.ellipse_angle);
    float aspect = ca.ellipse_minor > 1e-6f ? ca.ellipse_major / ca.ellipse_minor : 999;
    printf("    Aspect ratio: %.1f  %s\n", aspect,
           aspect < 2 ? "(isotropic)" :
           aspect < 5 ? "(slightly elongated)" : "(ELONGATED - weak direction)");
    printf("\n  Normal spread: %.0f deg  %s\n", ca.normal_spread,
           ca.normal_spread > 60 ? "(diverse - good)" :
           ca.normal_spread > 30 ? "(moderate)" : "(NARROW - poor angular diversity)");
    printf("  Spatial spread: %.0f px | Mean leverage: %.0f px\n",
           ca.spatial_spread, ca.mean_leverage);

    printf("\n  Per-point details:\n");
    printf("    %-4s  %-12s  %-10s  %-8s  %-8s  %-20s  %s\n",
           "#", "Position", "Normal", "Leverage", "InfoGain", "Lock (maj/min/ratio)", "Type");
    for (int i = 0; i < (int)ca.points.size(); i++) {
        auto& p = ca.points[i];
        char lock_buf[32];
        snprintf(lock_buf, sizeof(lock_buf), "%.3f/%.3f/%.1f", p.lock_major, p.lock_minor, p.lock_ratio);
        printf("    %-4d  (%5.0f,%5.0f)  (%5.2f,%5.2f)  %6.0f    %6.2f    %-20s  %s\n",
               i, p.pos.x, p.pos.y, p.normal.x, p.normal.y,
               p.leverage, p.info_contribution, lock_buf,
               p.is_corner ? "2D-LOCK" : "1D-edge");
    }
}

int main() {
    printf("================================================================\n");
    printf("  Geometric Constraint Quality Analysis\n");
    printf("================================================================\n");

    int TW = 200;

    struct TestShape {
        const char* name;
        std::function<void(Mat&)> draw;
    };

    TestShape shapes[] = {
        {"L-shape 200x200", [&](Mat& img) { draw_L(img, TW/2, TW/2, 0, 200); }},
        {"Rectangle 150x80", [&](Mat& img) { draw_rect(img, TW/2, TW/2, 150, 80, 200); }},
        {"Triangle", [&](Mat& img) { draw_triangle(img, TW/2, TW/2, 60, 200); }},
        {"Thin line (horizontal)", [&](Mat& img) { draw_line(img, TW/2, TW/2, 150, 200); }},
        {"Circle r=50", [&](Mat& img) { draw_circle(img, TW/2, TW/2, 50, 200); }},
    };

    for (auto& shape : shapes) {
        Mat templ(TW, TW, CV_8U, Scalar(50));
        shape.draw(templ);
        Mat mask = Mat::ones(TW, TW, CV_8U) * 255;

        auto fs = extractFeatures(templ, mask, 128);
        if (fs.refine_points.empty()) {
            printf("\n===== %s =====\n  No features extracted\n", shape.name);
            continue;
        }

        auto ca = fs.analyzeConstraints();
        printAnalysis(shape.name, ca);
    }

    // ================================================================
    // Key test: Triangle with center-edge vs corner-adjacent points
    // Same shape, same point count, different placement → different quality
    // ================================================================
    printf("\n\n================================================================\n");
    printf("  PLACEMENT COMPARISON: Triangle edge-center vs corner-adjacent\n");
    printf("================================================================\n");
    {
        // Larger triangle for clearer geometry
        int S = 350;
        Mat templ(S, S, CV_8U, Scalar(50));

        // Large triangle so edge midpoints are far from corners (pure 1D edges)
        Point2f v0(0, -110);    // top
        Point2f v1(-110, 80);   // bottom-left
        Point2f v2(110, 80);    // bottom-right

        std::vector<Point> tri_pts = {
            Point(S/2 + (int)v0.x, S/2 + (int)v0.y),
            Point(S/2 + (int)v1.x, S/2 + (int)v1.y),
            Point(S/2 + (int)v2.x, S/2 + (int)v2.y)
        };
        fillConvexPoly(templ, tri_pts, Scalar(200));
        Mat mask = Mat::ones(S, S, CV_8U) * 255;

        auto fs = extractFeatures(templ, mask, 256);
        printf("\n  Template: %dx%d triangle, %d refine points\n",
               S, S, (int)fs.refine_points.size());

        // Helper: find closest refine_point to a target position
        auto findClosest = [&](float tx, float ty) -> Point2f {
            int best = -1; float best_d = 1e9f;
            for (size_t i = 0; i < fs.refine_points.size(); i++) {
                float dx = tx - fs.refine_points[i].px;
                float dy = ty - fs.refine_points[i].py;
                float d = dx*dx + dy*dy;
                if (d < best_d) { best_d = d; best = (int)i; }
            }
            if (best >= 0) return Point2f(fs.refine_points[best].px, fs.refine_points[best].py);
            return Point2f(tx, ty);
        };

        // Set A: 3 points near edge CENTERS (midpoints of each edge)
        // These have low leverage (close to centroid)
        Point2f mid01 = Point2f((v0.x+v1.x)/2, (v0.y+v1.y)/2);  // mid top-left edge
        Point2f mid12 = Point2f((v1.x+v2.x)/2, (v1.y+v2.y)/2);  // mid bottom edge
        Point2f mid20 = Point2f((v2.x+v0.x)/2, (v2.y+v0.y)/2);  // mid top-right edge

        std::vector<Point2f> center_pts = {
            findClosest(mid01.x, mid01.y),
            findClosest(mid12.x, mid12.y),
            findClosest(mid20.x, mid20.y),
        };

        // Set B: 3 points near CORNERS, one per edge (different edges!)
        // Each point is 15% from a vertex along one specific edge
        // v0→v1 edge, near v0
        Point2f near_v0_on_01(v0.x + 0.15f*(v1.x - v0.x), v0.y + 0.15f*(v1.y - v0.y));
        // v1→v2 edge, near v1
        Point2f near_v1_on_12(v1.x + 0.15f*(v2.x - v1.x), v1.y + 0.15f*(v2.y - v1.y));
        // v2→v0 edge, near v2
        Point2f near_v2_on_20(v2.x + 0.15f*(v0.x - v2.x), v2.y + 0.15f*(v0.y - v2.y));

        std::vector<Point2f> corner_pts = {
            findClosest(near_v0_on_01.x, near_v0_on_01.y),
            findClosest(near_v1_on_12.x, near_v1_on_12.y),
            findClosest(near_v2_on_20.x, near_v2_on_20.y),
        };

        printf("\n  --- Set A: 3 points near edge CENTERS ---\n");
        for (auto& p : center_pts)
            printf("    (%.0f, %.0f) leverage=%.0f\n", p.x, p.y, std::sqrt(p.x*p.x + p.y*p.y));
        auto ca_center = fs.analyzeConstraints(center_pts);
        printAnalysis("Edge-center points (low leverage)", ca_center);

        printf("\n  --- Set B: 3 points near CORNERS ---\n");
        for (auto& p : corner_pts)
            printf("    (%.0f, %.0f) leverage=%.0f\n", p.x, p.y, std::sqrt(p.x*p.x + p.y*p.y));
        auto ca_corner = fs.analyzeConstraints(corner_pts);
        printAnalysis("Corner-adjacent points (high leverage)", ca_corner);

        // ============================================================
        // Test 2: 3 edges near corner vs 3 actual corners
        // Both have high leverage (near vertices), but:
        //   - Edge points: 1D constraint each (normal only)
        //   - Corner points: 2D constraint each (normal + tangent)
        // ============================================================
        printf("\n\n  ====== TEST 2: 3 edge points (near corner) vs 3 corner points ======\n");
        {
            // 3 edge points: one per edge, at the MIDPOINT of each edge (pure 1D)
            // Far from any corner so the ROI only sees a single straight edge
            Point2f e0((v0.x+v1.x)*0.5f, (v0.y+v1.y)*0.5f);  // mid of v0-v1
            Point2f e1((v1.x+v2.x)*0.5f, (v1.y+v2.y)*0.5f);  // mid of v1-v2
            Point2f e2((v2.x+v0.x)*0.5f, (v2.y+v0.y)*0.5f);  // mid of v2-v0

            std::vector<Point2f> edge_pts = {
                findClosest(e0.x, e0.y),
                findClosest(e1.x, e1.y),
                findClosest(e2.x, e2.y),
            };

            // 3 corner points: AT the exact vertices (not snapped to refine_points)
            // The ROI at a vertex contains two edges meeting → PCA should detect corner
            std::vector<Point2f> corner_pts2 = {
                Point2f(v0.x, v0.y),
                Point2f(v1.x, v1.y),
                Point2f(v2.x, v2.y),
            };

            printf("\n  --- Set C: 3 edge points (near corners, 1D constraint each) ---\n");
            for (auto& p : edge_pts)
                printf("    (%.0f, %.0f) leverage=%.0f\n", p.x, p.y, std::sqrt(p.x*p.x + p.y*p.y));
            auto ca_edge = fs.analyzeConstraints(edge_pts);
            printAnalysis("3 edges near corner (1D each)", ca_edge);

            printf("\n  --- Set D: 3 corner points (at vertices, 2D constraint each) ---\n");
            for (auto& p : corner_pts2) {
                printf("    (%.0f, %.0f) leverage=%.0f", p.x, p.y, std::sqrt(p.x*p.x + p.y*p.y));
                // Debug: show PCA eigenvalues at this position
                int itx = (int)(p.x + S/2.0f + 0.5f), ity = (int)(p.y + S/2.0f + 0.5f);
                int ih = 15;
                if (itx-ih>=0 && ity-ih>=0 && itx+ih<S && ity+ih<S) {
                    Mat dbg_roi = templ(Rect(itx-ih, ity-ih, 2*ih+1, 2*ih+1));
                    Mat ddx, ddy, dmag;
                    Sobel(dbg_roi, ddx, CV_32F, 1, 0, 3);
                    Sobel(dbg_roi, ddy, CV_32F, 0, 1, 3);
                    magnitude(ddx, ddy, dmag);
                    float dmax = *std::max_element(dmag.begin<float>(), dmag.end<float>());
                    std::vector<Point2f> dpts;
                    for (int dr=0; dr<dbg_roi.rows; dr++)
                        for (int dc=0; dc<dbg_roi.cols; dc++)
                            if (dmag.at<float>(dr,dc) > 0.3f*dmax) dpts.push_back(Point2f((float)dc,(float)dr));
                    if (dpts.size() >= 3) {
                        Point2f dm(0,0);
                        for (auto& dp : dpts) dm += dp;
                        dm *= (1.0f/dpts.size());
                        float dcxx=0,dcyy=0,dcxy=0;
                        for (auto& dp : dpts) { float ddxx=dp.x-dm.x,ddyy=dp.y-dm.y; dcxx+=ddxx*ddxx; dcyy+=ddyy*ddyy; dcxy+=ddxx*ddyy; }
                        float dn=(float)dpts.size(); dcxx/=dn; dcyy/=dn; dcxy/=dn;
                        float dtr=dcxx+dcyy, ddisc=sqrt(std::max(0.f,(dcxx-dcyy)*(dcxx-dcyy)/4+dcxy*dcxy));
                        float dl1=dtr/2+ddisc, dl2=dtr/2-ddisc;
                        printf("  PCA: lam1=%.1f lam2=%.1f ratio=%.2f (%s) edge_pts=%d",
                               dl1, dl2, dl1/std::max(0.01f,dl2), dl1/std::max(0.01f,dl2)<1.5f?"CORNER":"edge", (int)dpts.size());
                    }
                }
                printf("\n");
            }
            auto ca_corner2 = fs.analyzeConstraints(corner_pts2);
            printAnalysis("3 corners at vertices (2D each)", ca_corner2);

            printf("\n  ====== COMPARISON: Edge (1D) vs Corner (2D) ======\n");
            printf("  %-25s  %10s  %10s\n", "Metric", "3 Edges", "3 Corners");
            printf("  %-25s  %10.3f  %10.3f\n", "sigma_theta (deg)", ca_edge.sigma_theta, ca_corner2.sigma_theta);
            printf("  %-25s  %10.3f  %10.3f\n", "sigma_tx (px)", ca_edge.sigma_tx, ca_corner2.sigma_tx);
            printf("  %-25s  %10.3f  %10.3f\n", "sigma_ty (px)", ca_edge.sigma_ty, ca_corner2.sigma_ty);
            printf("  %-25s  %10.3f  %10.3f\n", "ellipse major (px)", ca_edge.ellipse_major, ca_corner2.ellipse_major);
            printf("  %-25s  %10.3f  %10.3f\n", "ellipse minor (px)", ca_edge.ellipse_minor, ca_corner2.ellipse_minor);
            printf("  %-25s  %10.1f  %10.1f\n", "condition number", ca_edge.condition_number, ca_corner2.condition_number);
            printf("  %-25s  %10.0f  %10.0f\n", "normal spread (deg)", ca_edge.normal_spread, ca_corner2.normal_spread);
            printf("  %-25s  %10.0f  %10.0f\n", "mean leverage (px)", ca_edge.mean_leverage, ca_corner2.mean_leverage);
            printf("  %-25s  %10d  %10d\n", "corners detected", ca_edge.num_corners, ca_corner2.num_corners);

            // Set E: 3 edge points near corners (25% from vertex along each edge)
            Point2f ne0(v0.x + 0.25f*(v1.x-v0.x), v0.y + 0.25f*(v1.y-v0.y));
            Point2f ne1(v1.x + 0.25f*(v2.x-v1.x), v1.y + 0.25f*(v2.y-v1.y));
            Point2f ne2(v2.x + 0.25f*(v0.x-v2.x), v2.y + 0.25f*(v0.y-v2.y));

            std::vector<Point2f> near_corner_pts = {
                findClosest(ne0.x, ne0.y),
                findClosest(ne1.x, ne1.y),
                findClosest(ne2.x, ne2.y),
            };

            printf("\n  --- Set E: 3 edge points NEAR corners (25%% from vertex) ---\n");
            for (auto& p : near_corner_pts)
                printf("    (%.0f, %.0f) leverage=%.0f\n", p.x, p.y, std::sqrt(p.x*p.x + p.y*p.y));
            auto ca_near = fs.analyzeConstraints(near_corner_pts);
            printAnalysis("3 edges near corner (25% from vertex)", ca_near);

            printf("\n  ====== COMPARISON: Edge-mid vs Edge-near-corner vs Vertex ======\n");
            printf("  %-25s  %10s  %10s  %10s\n", "Metric", "Mid-edge", "Near-corner", "Vertex");
            printf("  %-25s  %10.3f  %10.3f  %10.3f\n", "sigma_theta (deg)", ca_edge.sigma_theta, ca_near.sigma_theta, ca_corner2.sigma_theta);
            printf("  %-25s  %10.3f  %10.3f  %10.3f\n", "sigma_tx (px)", ca_edge.sigma_tx, ca_near.sigma_tx, ca_corner2.sigma_tx);
            printf("  %-25s  %10.3f  %10.3f  %10.3f\n", "sigma_ty (px)", ca_edge.sigma_ty, ca_near.sigma_ty, ca_corner2.sigma_ty);
            printf("  %-25s  %10.3f  %10.3f  %10.3f\n", "ellipse major (px)", ca_edge.ellipse_major, ca_near.ellipse_major, ca_corner2.ellipse_major);
            printf("  %-25s  %10.3f  %10.3f  %10.3f\n", "ellipse minor (px)", ca_edge.ellipse_minor, ca_near.ellipse_minor, ca_corner2.ellipse_minor);
            printf("  %-25s  %10.0f  %10.0f  %10.0f\n", "normal spread (deg)", ca_edge.normal_spread, ca_near.normal_spread, ca_corner2.normal_spread);
            printf("  %-25s  %10.0f  %10.0f  %10.0f\n", "mean leverage (px)", ca_edge.mean_leverage, ca_near.mean_leverage, ca_corner2.mean_leverage);
            printf("  %-25s  %10d  %10d  %10d\n", "corners (2D lock)", ca_edge.num_corners, ca_near.num_corners, ca_corner2.num_corners);
            int c_edge = ca_edge.num_corners*2 + ca_edge.num_edges;
            int c_near = ca_near.num_corners*2 + ca_near.num_edges;
            int c_vert = ca_corner2.num_corners*2 + ca_corner2.num_edges;
            printf("  %-25s  %10d  %10d  %10d\n", "total constraints", c_edge, c_near, c_vert);

        // Direct comparison (original test 1)
        printf("\n  ====== COMPARISON: Test 1 (center vs corner) ======\n");
        printf("  %-25s  %10s  %10s\n", "Metric", "Center", "Corner");
        printf("  %-25s  %10.3f  %10.3f\n", "sigma_theta (deg)", ca_center.sigma_theta, ca_corner.sigma_theta);
        printf("  %-25s  %10.3f  %10.3f\n", "sigma_tx (px)", ca_center.sigma_tx, ca_corner.sigma_tx);
        printf("  %-25s  %10.3f  %10.3f\n", "sigma_ty (px)", ca_center.sigma_ty, ca_corner.sigma_ty);
        printf("  %-25s  %10.3f  %10.3f\n", "ellipse major (px)", ca_center.ellipse_major, ca_corner.ellipse_major);
        printf("  %-25s  %10.3f  %10.3f\n", "ellipse minor (px)", ca_center.ellipse_minor, ca_corner.ellipse_minor);
        printf("  %-25s  %10.1f  %10.1f\n", "condition number", ca_center.condition_number, ca_corner.condition_number);
        printf("  %-25s  %10.0f  %10.0f\n", "normal spread (deg)", ca_center.normal_spread, ca_corner.normal_spread);
        printf("  %-25s  %10.0f  %10.0f\n", "mean leverage (px)", ca_center.mean_leverage, ca_corner.mean_leverage);

        bool corner_better_theta = ca_corner.sigma_theta < ca_center.sigma_theta;
        printf("\n  Verdict: corner-adjacent points are %s for angle constraint\n",
               corner_better_theta ? "BETTER" : "WORSE");
    } // end test 1 comparison
    } // end placement comparison block

    printf("\n================================================================\n");
    return 0;
}
