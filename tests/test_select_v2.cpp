// Compare V1 (two-phase) vs V2 (multi-start hat matrix leverage swap)
// for feature selection quality: sensitivity, accuracy, speed.

#include "shape_matcher.h"
#include "roi_refine.h"
#include "test_utils.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>   // getRotationMatrix2D moved here in OpenCV 5
#endif
#include <chrono>
#include <cstdio>
#include <cmath>

using namespace cv;

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

static void draw_rect(Mat& img, int cx, int cy, double angle, int color) {
    double rad = angle * CV_PI / 180.0, cs = cos(rad), sn = sin(rad);
    for (double ly = -25; ly <= 25; ly += 0.5)
        for (double lx = -40; lx <= 40; lx += 0.5) {
            if (std::abs(ly) < 20 && std::abs(lx) < 35) continue;
            int px = cx+(int)(lx*cs-ly*sn+0.5), py = cy+(int)(lx*sn+ly*cs+0.5);
            if (px>=0&&px<img.cols&&py>=0&&py<img.rows) img.at<uchar>(py,px)=(uchar)color;
        }
}

static void draw_tri(Mat& img, int cx, int cy, double angle, int color) {
    double rad = angle * CV_PI / 180.0, cs = cos(rad), sn = sin(rad);
    for (double t = 0; t < 1.0; t += 0.002) {
        double pts[][2] = {{-30,25},{30,25},{0,-30}};
        for (int e = 0; e < 3; e++) {
            double x = pts[e][0]*(1-t)+pts[(e+1)%3][0]*t;
            double y = pts[e][1]*(1-t)+pts[(e+1)%3][1]*t;
            for (double w = -3; w <= 3; w += 0.5) {
                double nx = -(pts[(e+1)%3][1]-pts[e][1]), ny = pts[(e+1)%3][0]-pts[e][0];
                double nm = sqrt(nx*nx+ny*ny); if(nm<1) continue; nx/=nm; ny/=nm;
                int px = cx+(int)((x+w*nx)*cs-(y+w*ny)*sn+0.5);
                int py = cy+(int)((x+w*nx)*sn+(y+w*ny)*cs+0.5);
                if (px>=0&&px<img.cols&&py>=0&&py<img.rows) img.at<uchar>(py,px)=(uchar)color;
            }
        }
    }
}

static void place_object(const Mat& templ, Mat& scene, int cx, int cy, double angle,
                          float sub_x = 0, float sub_y = 0) {
    float tcx = templ.cols / 2.0f, tcy = templ.rows / 2.0f;
    float dst_cx = cx + sub_x, dst_cy = cy + sub_y;
    Mat M = getRotationMatrix2D(Point2f(tcx, tcy), -angle, 1.0);
    double* md = (double*)M.data;
    md[2] += dst_cx - std::floor(dst_cx);
    md[5] += dst_cy - std::floor(dst_cy);
    Mat rot;
    warpAffine(templ, rot, M, Size(templ.cols+2, templ.rows+2), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    int ox = (int)std::floor(dst_cx) - (int)tcx;
    int oy = (int)std::floor(dst_cy) - (int)tcy;
    for (int r = 0; r < rot.rows; r++)
        for (int c = 0; c < rot.cols; c++) {
            int sy = oy+r, sx = ox+c;
            if (sy>=0 && sy<scene.rows && sx>=0 && sx<scene.cols && rot.at<uchar>(r,c)>0)
                scene.at<uchar>(sy,sx) = std::max(scene.at<uchar>(sy,sx), rot.at<uchar>(r,c));
        }
}

static float angle_err(float a, float b) {
    float e = a - b; while(e>180) e-=360; while(e<-180) e+=360; return std::abs(e);
}

struct ShapeInfo {
    const char* name;
    void (*draw)(Mat&, int, int, double, int);
};

int main() {
    ShapeInfo shapes[] = {
        {"L-shape", draw_L},
        {"rectangle", draw_rect},
        {"triangle", draw_tri},
    };
    int n_shapes = 3;
    int pt_counts[] = {6, 8, 10, 12};
    int n_pts = 4;
    int restart_counts[] = {1, 5, 10, 20, 50};
    int n_restarts = 5;

    printf("===== V1 vs V2 Feature Selection Comparison =====\n\n");

    // Test angles for accuracy evaluation
    double test_angles[] = {0, 15, 30, 45, 67, 90, 120, 135, 170, 210, 250, 300, 345};
    int n_angles = 13;

    for (int si = 0; si < n_shapes; si++) {
        auto& shape = shapes[si];
        printf("--- %s ---\n", shape.name);

        // Create template
        Mat templ(200, 200, CV_8U, Scalar(0));
        shape.draw(templ, 100, 100, 0, 200);

        sbm::FeatureSet feat;
        {
            OutputGuard guard;
            feat = sbm::extractFeatures(templ);
        }
        feat.setOrigin(100, 75);

        printf("  Candidates: %d\n\n", feat.numFeatures());

        // Header
        printf("  %-6s  %-8s  %-4s  %10s  %10s  %10s  %10s  %8s\n",
               "Method", "N_pts", "K", "worst_ang", "mean_ang", "acc_ang", "acc_pos", "time_ms");
        printf("  %-6s  %-8s  %-4s  %10s  %10s  %10s  %10s  %8s\n",
               "------", "--------", "----", "----------", "----------", "----------", "----------", "--------");

        for (int pi = 0; pi < n_pts; pi++) {
            int npts = pt_counts[pi];

            // --- V1 ---
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                // Clear cache to force recomputation
                feat.cached_opt_points.clear();
                feat.cached_opt_max_points = 0;
                auto pts_v1 = feat.selectOptimizedPoints(npts);
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();

                // Sensitivity
                auto sr = feat.analyzeSensitivity();

                // Accuracy: match across test angles
                float sum_ae = 0, sum_pe = 0;
                int n_matched = 0;
                {
                    OutputGuard guard;
                    sbm::MatchConfig cfg;
                    cfg.min_score = 30;
                    cfg.refine = sbm::RefineMode::ROI;
                    sbm::ModelConfig mcfg;
                    mcfg.angle = {0, 360, 2};
                    sbm::ShapeMatcher matcher(cfg);
                    matcher.addModel("T", feat, mcfg);

                    for (int ai = 0; ai < n_angles; ai++) {
                        double gt_angle = test_angles[ai];
                        Mat scene(400, 400, CV_8U, Scalar(30));
                        place_object(templ, scene, 200, 200, gt_angle);
                        auto res = matcher.match(scene);
                        if (!res.empty()) {
                            sum_ae += angle_err(res[0].angle, (float)gt_angle);
                            sum_pe += std::sqrt((res[0].x-200)*(res[0].x-200)+(res[0].y-175)*(res[0].y-175));
                            n_matched++;
                        }
                    }
                }
                float acc_ae = n_matched > 0 ? sum_ae / n_matched : 999;
                float acc_pe = n_matched > 0 ? sum_pe / n_matched : 999;

                printf("  %-6s  %-8d  %-4s  %9.4f°  %9.4f°  %9.4f°  %9.4fpx  %7.2f\n",
                       "V1", npts, "-", sr.worst_angle_sens, sr.mean_angle_sens, acc_ae, acc_pe, ms);
            }

            // --- V2 with different restart counts ---
            for (int ri = 0; ri < n_restarts; ri++) {
                int K = restart_counts[ri];

                auto t0 = std::chrono::high_resolution_clock::now();
                auto pts_v2 = feat.selectOptimizedPointsV2(npts, K);
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();

                // Compute sensitivity for V2 points
                // Temporarily override cached points
                auto saved_pts = feat.cached_opt_points;
                auto saved_max = feat.cached_opt_max_points;
                feat.cached_opt_points = pts_v2;
                feat.cached_opt_max_points = npts;
                auto sr = feat.analyzeSensitivity();
                feat.cached_opt_points = saved_pts;
                feat.cached_opt_max_points = saved_max;

                // Accuracy with V2 points
                float sum_ae = 0, sum_pe = 0;
                int n_matched = 0;
                {
                    OutputGuard guard;
                    // Build matcher with V2 points
                    sbm::FeatureSet feat_v2 = feat;
                    feat_v2.cached_opt_points = pts_v2;
                    feat_v2.cached_opt_max_points = npts;

                    sbm::MatchConfig cfg;
                    cfg.min_score = 30;
                    cfg.refine = sbm::RefineMode::ROI;
                    sbm::ModelConfig mcfg;
                    mcfg.angle = {0, 360, 2};
                    sbm::ShapeMatcher matcher(cfg);
                    matcher.addModel("T", feat_v2, mcfg);

                    for (int ai = 0; ai < n_angles; ai++) {
                        double gt_angle = test_angles[ai];
                        Mat scene(400, 400, CV_8U, Scalar(30));
                        place_object(templ, scene, 200, 200, gt_angle);
                        auto res = matcher.match(scene);
                        if (!res.empty()) {
                            sum_ae += angle_err(res[0].angle, (float)gt_angle);
                            sum_pe += std::sqrt((res[0].x-200)*(res[0].x-200)+(res[0].y-175)*(res[0].y-175));
                            n_matched++;
                        }
                    }
                }
                float acc_ae = n_matched > 0 ? sum_ae / n_matched : 999;
                float acc_pe = n_matched > 0 ? sum_pe / n_matched : 999;

                char kstr[16]; snprintf(kstr, sizeof(kstr), "K=%d", K);
                printf("  %-6s  %-8d  %-4s  %9.4f°  %9.4f°  %9.4f°  %9.4fpx  %7.2f\n",
                       "V2", npts, kstr, sr.worst_angle_sens, sr.mean_angle_sens, acc_ae, acc_pe, ms);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("===== COMPARISON COMPLETE =====\n");
    return 0;
}
