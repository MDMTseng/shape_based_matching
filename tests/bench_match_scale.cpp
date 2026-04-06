/// @file bench_match_scale.cpp
/// @brief Total end-to-end profile: s=1.0 T={4,8} with ROI refine.

#include "shape_matcher.h"
#include "line2Dup.h"
#include "sbm_log.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace cv;
using namespace sbm;
using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static float angle_err(float a, float b) {
    float e = std::fmod(std::abs(a - b), 360.0f);
    return std::min(e, 360.0f - e);
}

struct ObjGT { float x, y; float angle; };

static void place_object(const Mat& templ, Mat& scene, float cx, float cy, double angle) {
    Mat M = getRotationMatrix2D(Point2f(templ.cols/2.0f, templ.rows/2.0f), -angle, 1.0);
    M.at<double>(0,2) += cx - templ.cols/2.0;
    M.at<double>(1,2) += cy - templ.rows/2.0;
    Mat warped;
    warpAffine(templ, warped, M, scene.size(), INTER_LINEAR, BORDER_TRANSPARENT);
    for (int r = 0; r < warped.rows; r++)
        for (int c = 0; c < warped.cols; c++)
            if (warped.at<uchar>(r,c) > 0)
                scene.at<uchar>(r,c) = warped.at<uchar>(r,c);
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    sbm::setLogLevel(sbm::LogLevel::Warning);

    const char* img_path = "C:/Users/TRS001/Documents/workspace/claudePrj/cludefiles/web-terminal-hub/uploads/img_1775409786258_8c045799.png";
    if (argc > 1) img_path = argv[1];

    Mat templ_raw = imread(img_path, IMREAD_GRAYSCALE);
    if (templ_raw.empty()) { printf("ERROR: cannot load %s\n", img_path); return 1; }

    Mat binary; threshold(templ_raw, binary, 30, 255, THRESH_BINARY);
    std::vector<Point> pts; findNonZero(binary, pts);
    Rect bbox = boundingRect(pts);
    int pad = 5;
    bbox.x = std::max(0, bbox.x-pad); bbox.y = std::max(0, bbox.y-pad);
    bbox.width = std::min(templ_raw.cols-bbox.x, bbox.width+2*pad);
    bbox.height = std::min(templ_raw.rows-bbox.y, bbox.height+2*pad);
    Mat templ;
    float sc200 = 200.0f / std::max(bbox.width, bbox.height);
    resize(templ_raw(bbox), templ, Size(), sc200, sc200);

    auto feat = extractFeatures(templ);
    feat.setOrigin(templ.cols/2.0f, templ.rows/2.0f);
    // No template blur — sharp template gives best matchTemplate precision
    ModelConfig mcfg;
    mcfg.angle = {0, 360, 1};

    const int W = 1920, H = 1080;
    float test_angles[] = {5,23,47,78,112,145,178,210,243,275,308,340,
                           15,38,62,95,128,160,195,228,258,290,325,355};
    int N_OBJ = 24;

    std::vector<ObjGT> objects;
    { int cols=6, rows=4; float sx=W/(float)(cols+1), sy=H/(float)(rows+1);
      for (int i=0; i<N_OBJ; i++) objects.push_back({sx*(i%cols+1), sy*(i/cols+1), test_angles[i]}); }

    Mat scene(H, W, CV_8U, Scalar(0));
    for (auto& obj : objects) place_object(templ, scene, obj.x, obj.y, obj.angle);
    // No noise

    printf("================================================================\n");
    printf("  Total profile: 20MP, 24 objects, %dx%d template\n", templ.cols, templ.rows);
    printf("  %d features, angle step=1 deg\n",
           feat.levels.empty() ? 0 : (int)feat.levels[0].features.size());
    printf("================================================================\n\n");

    struct Config {
        const char* name;
        float scale;
        float min_score;
    };
    Config configs[] = {
        {"s=1.0 score=65", 1.0f, 65},
        {"s=0.5 score=65", 0.5f, 65},
        {"s=0.3 score=50", 0.3f, 50},
    };

    for (auto& c : configs) {
        printf("  === %s, T={4,8}, ROI ===\n", c.name);

        MatchConfig cfg;
        cfg.min_score = c.min_score;
        cfg.refine = RefineMode::ROI;
        cfg.skip_voting = true;
        cfg.match_scale = c.scale;

        ShapeMatcher matcher(cfg);
        matcher.addModel("part", feat, mcfg);

        // Warmup
        matcher.match(scene);

        // Profiled run with LineMOD internals
        line2Dup::enableProfiling(true);
        sbm::setLogLevel(sbm::LogLevel::Debug);
        sbm::setLogFile(stdout);
        line2Dup::resetProfiling();

        auto t0 = Clock::now();
        auto results = matcher.match(scene);
        double t_total = ms_since(t0);

        printf("    --- LineMOD internal ---\n");
        line2Dup::printProfiling();
        line2Dup::enableProfiling(false);
        sbm::setLogFile(nullptr);
        sbm::setLogLevel(sbm::LogLevel::Warning);

        // Accuracy
        int n_tp = 0;
        float tae = 0, tpe = 0, wae = 0, wpe = 0;
        for (auto& gt : objects) {
            float best_d = 1e9f; int best_ri = -1;
            for (int ri = 0; ri < (int)results.size(); ri++) {
                float dx = results[ri].x - gt.x, dy = results[ri].y - gt.y;
                float d = std::sqrt(dx*dx + dy*dy);
                if (d < best_d) { best_d = d; best_ri = ri; }
            }
            if (best_ri >= 0 && best_d < 40) {
                float ae = angle_err(results[best_ri].angle, gt.angle);
                tae += ae; tpe += best_d;
                wae = std::max(wae, ae); wpe = std::max(wpe, best_d);
                n_tp++;
            }
        }

        int n_fp = (int)results.size() - n_tp;
        printf("    --- Summary ---\n");
        printf("    Total match():       %7.1f ms\n", t_total);
        printf("    Results: %d (%d TP, %d FP)\n", (int)results.size(), n_tp, n_fp);
        printf("    Angle:  mean=%.2f  worst=%.1f\n", n_tp > 0 ? tae/n_tp : -1, wae);
        printf("    Pos:    mean=%.2f  worst=%.1f\n", n_tp > 0 ? tpe/n_tp : -1, wpe);
        printf("    NMS radius: %.0f px\n",
               std::min((float)templ.cols, (float)templ.rows) / 2.0f);

        // Position bias analysis for full-res
        if (n_tp > 0 && c.scale >= 0.99f) {
            float bias_x = 0, bias_y = 0, rms_x = 0, rms_y = 0;
            int nb = 0;
            printf("    Per-object error (x, y):\n");
            for (int gi = 0; gi < N_OBJ; gi++) {
                float best_d = 1e9f; int best_ri = -1;
                for (int ri = 0; ri < (int)results.size(); ri++) {
                    float dx = results[ri].x - objects[gi].x;
                    float dy = results[ri].y - objects[gi].y;
                    float d = std::sqrt(dx*dx+dy*dy);
                    if (d < best_d) { best_d = d; best_ri = ri; }
                }
                if (best_ri >= 0 && best_d < 40) {
                    float ex = results[best_ri].x - objects[gi].x;
                    float ey = results[best_ri].y - objects[gi].y;
                    bias_x += ex; bias_y += ey;
                    rms_x += ex*ex; rms_y += ey*ey;
                    nb++;
                    float ae = angle_err(results[best_ri].angle, objects[gi].angle);
                    printf("      #%-2d ang=%3.0f  pos=(%+.2f,%+.2f) d=%.2f  ang_err=%+.2f\n",
                           gi, objects[gi].angle, ex, ey, best_d, ae);
                }
            }
            if (nb > 0) {
                bias_x /= nb; bias_y /= nb;
                printf("    Bias: (%+.3f, %+.3f) px  magnitude=%.3f\n",
                       bias_x, bias_y, std::sqrt(bias_x*bias_x + bias_y*bias_y));
                printf("    RMS:  (%.3f, %.3f) px\n", std::sqrt(rms_x/nb), std::sqrt(rms_y/nb));
            }
        }

        // FP analysis: distance to nearest GT and nearest TP result
        if (n_fp > 0) {
            // Mark which results are TPs
            std::vector<bool> is_tp(results.size(), false);
            for (int gi = 0; gi < N_OBJ; gi++) {
                float best_d = 1e9f; int best_ri = -1;
                for (int ri = 0; ri < (int)results.size(); ri++) {
                    float dx = results[ri].x - objects[gi].x, dy = results[ri].y - objects[gi].y;
                    float d = std::sqrt(dx*dx+dy*dy);
                    if (d < best_d && d < 40) { best_d = d; best_ri = ri; }
                }
                if (best_ri >= 0) is_tp[best_ri] = true;
            }

            printf("    FP distances (to nearest TP result):\n");
            printf("    %-5s  %-7s  %s\n", "Score", "Dist", "Would NMS catch?");
            float fp_min_dist = 1e9f, fp_max_dist = 0;
            for (int ri = 0; ri < (int)results.size(); ri++) {
                if (is_tp[ri]) continue;
                float min_d = 1e9f;
                for (int rj = 0; rj < (int)results.size(); rj++) {
                    if (!is_tp[rj]) continue;
                    float dx = results[ri].x - results[rj].x;
                    float dy = results[ri].y - results[rj].y;
                    float d = std::sqrt(dx*dx+dy*dy);
                    if (d < min_d) min_d = d;
                }
                fp_min_dist = std::min(fp_min_dist, min_d);
                fp_max_dist = std::max(fp_max_dist, min_d);
                float nms_r = std::min((float)templ.cols, (float)templ.rows) / 2.0f;
                printf("    %5.1f  %5.0fpx  %s\n", results[ri].score, min_d,
                       min_d < nms_r ? "yes (inside radius)" : "no (outside)");
            }
            printf("    FP dist range: %.0f - %.0fpx\n", fp_min_dist, fp_max_dist);
        }
        printf("\n");
    }

    // Draw error chart: 360-degree sweep at 1° steps, s=1.0
    {
        printf("  === 360-degree error sweep (s=1.0, clean) ===\n");
        MatchConfig cfg;
        cfg.min_score = 65;
        cfg.refine = RefineMode::ROI;
        cfg.skip_voting = true;
        cfg.match_scale = 1.0f;
        ShapeMatcher matcher(cfg);
        matcher.addModel("part", feat, mcfg);

        int tw = templ.cols, th = templ.rows;
        int sw = tw * 4, sh = th * 4;
        float cx = sw / 2.0f, cy = sh / 2.0f;

        int N_ANG = 360;
        std::vector<float> ang_err_arr(N_ANG, 0), pos_err_arr(N_ANG, 0);
        std::vector<float> err_x_arr(N_ANG, 0), err_y_arr(N_ANG, 0);

        for (int ai = 0; ai < N_ANG; ai++) {
            float angle = (float)ai;
            Mat test_scene(sh, sw, CV_8U, Scalar(0));
            Mat M = getRotationMatrix2D(Point2f(tw/2.0f, th/2.0f), -angle, 1.0);
            M.at<double>(0,2) += cx - tw/2.0;
            M.at<double>(1,2) += cy - th/2.0;
            Mat warped;
            warpAffine(templ, warped, M, test_scene.size(), INTER_LINEAR, BORDER_TRANSPARENT);
            for (int r = 0; r < warped.rows; r++)
                for (int c = 0; c < warped.cols; c++)
                    if (warped.at<uchar>(r,c) > 0)
                        test_scene.at<uchar>(r,c) = warped.at<uchar>(r,c);

            auto results = matcher.match(test_scene);
            if (!results.empty()) {
                err_x_arr[ai] = results[0].x - cx;
                err_y_arr[ai] = results[0].y - cy;
                pos_err_arr[ai] = std::sqrt(err_x_arr[ai]*err_x_arr[ai] + err_y_arr[ai]*err_y_arr[ai]);
                ang_err_arr[ai] = angle_err(results[0].angle, angle);
            } else {
                pos_err_arr[ai] = -1;
                ang_err_arr[ai] = -1;
            }
        }

        // Draw chart: 720x400, angle on x, error on y
        int cw = 720, ch = 400;
        Mat chart(ch, cw, CV_8UC3, Scalar(30, 30, 30));

        // Grid
        for (int g = 0; g <= 360; g += 45) {
            int x = g * cw / 360;
            line(chart, Point(x, 0), Point(x, ch), Scalar(60, 60, 60));
            char buf[16]; snprintf(buf, sizeof(buf), "%d", g);
            putText(chart, buf, Point(x + 2, ch - 5), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(120,120,120));
        }

        // Position error (cyan, scale: 1px = 200 chart pixels)
        float pos_scale = 200.0f;  // chart pixels per px error
        for (int g = 0; g < 5; g++) {
            int y = ch/2 - (int)(g * 0.2f * pos_scale);
            if (y > 0 && y < ch) {
                line(chart, Point(0, y), Point(cw, y), Scalar(50, 50, 50));
                char buf[16]; snprintf(buf, sizeof(buf), "%.1f", g * 0.2f);
                putText(chart, buf, Point(2, y - 3), FONT_HERSHEY_SIMPLEX, 0.25, Scalar(100,100,100));
            }
        }

        // Draw pos_err
        for (int ai = 1; ai < N_ANG; ai++) {
            if (pos_err_arr[ai-1] < 0 || pos_err_arr[ai] < 0) continue;
            int x0 = (ai-1) * cw / 360, x1 = ai * cw / 360;
            int y0 = ch/2 - (int)(pos_err_arr[ai-1] * pos_scale);
            int y1 = ch/2 - (int)(pos_err_arr[ai] * pos_scale);
            line(chart, Point(x0, y0), Point(x1, y1), Scalar(255, 255, 0), 1, LINE_AA);
        }

        // Draw err_x (red) and err_y (green)
        for (int ai = 1; ai < N_ANG; ai++) {
            if (pos_err_arr[ai-1] < 0 || pos_err_arr[ai] < 0) continue;
            int x0 = (ai-1) * cw / 360, x1 = ai * cw / 360;
            // err_x
            int yx0 = ch/2 - (int)(err_x_arr[ai-1] * pos_scale);
            int yx1 = ch/2 - (int)(err_x_arr[ai] * pos_scale);
            line(chart, Point(x0, yx0), Point(x1, yx1), Scalar(0, 0, 200), 1, LINE_AA);
            // err_y
            int yy0 = ch/2 - (int)(err_y_arr[ai-1] * pos_scale);
            int yy1 = ch/2 - (int)(err_y_arr[ai] * pos_scale);
            line(chart, Point(x0, yy0), Point(x1, yy1), Scalar(0, 200, 0), 1, LINE_AA);
        }

        // Draw angle error (yellow, separate scale: 1 deg = 400 chart pixels)
        float ang_scale = 400.0f;
        for (int ai = 1; ai < N_ANG; ai++) {
            if (ang_err_arr[ai-1] < 0 || ang_err_arr[ai] < 0) continue;
            int x0 = (ai-1) * cw / 360, x1 = ai * cw / 360;
            int y0 = ch - 5 - (int)(ang_err_arr[ai-1] * ang_scale);
            int y1 = ch - 5 - (int)(ang_err_arr[ai] * ang_scale);
            line(chart, Point(x0, y0), Point(x1, y1), Scalar(0, 200, 255), 1, LINE_AA);
        }

        // Zero line
        line(chart, Point(0, ch/2), Point(cw, ch/2), Scalar(100, 100, 100));

        // Legend
        putText(chart, "cyan=pos_dist  red=err_x  green=err_y  orange=ang_err",
                Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.35, Scalar(200, 200, 200));
        putText(chart, "pos scale: 0.2px/grid  ang scale: bottom=0 top=1deg",
                Point(5, 30), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(150, 150, 150));

        imwrite("output/error_chart_360.png", chart);
        printf("    Saved output/error_chart_360.png\n");

        // Print stats
        float sum_pos = 0, sum_ang = 0, max_pos = 0, max_ang = 0;
        int n_valid = 0;
        for (int ai = 0; ai < N_ANG; ai++) {
            if (pos_err_arr[ai] < 0) continue;
            sum_pos += pos_err_arr[ai]; sum_ang += ang_err_arr[ai];
            max_pos = std::max(max_pos, pos_err_arr[ai]);
            max_ang = std::max(max_ang, ang_err_arr[ai]);
            n_valid++;
        }
        printf("    360-sweep: %d/360 found\n", n_valid);
        printf("    Pos: mean=%.3f  worst=%.3f\n", sum_pos/n_valid, max_pos);
        printf("    Ang: mean=%.3f  worst=%.3f\n", sum_ang/n_valid, max_ang);
    }

    printf("\n================================================================\n");
    return 0;
}
