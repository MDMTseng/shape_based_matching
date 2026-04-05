/// @file bench_match_scale.cpp
/// @brief match_scale=0.5 stability test with real template, ROI refine.

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
    sbm::setLogLevel(sbm::LogLevel::Warning);

    const char* img_path = "C:/Users/TRS001/Documents/workspace/claudePrj/cludefiles/web-terminal-hub/uploads/img_1775409786258_8c045799.png";
    if (argc > 1) img_path = argv[1];

    Mat templ_raw = imread(img_path, IMREAD_GRAYSCALE);
    if (templ_raw.empty()) { printf("ERROR: cannot load %s\n", img_path); return 1; }

    // Auto-crop to non-black bounding box
    Mat binary; threshold(templ_raw, binary, 30, 255, THRESH_BINARY);
    std::vector<Point> pts; findNonZero(binary, pts);
    Rect bbox = boundingRect(pts);
    int pad = 5;
    bbox.x = std::max(0, bbox.x - pad); bbox.y = std::max(0, bbox.y - pad);
    bbox.width = std::min(templ_raw.cols - bbox.x, bbox.width + 2*pad);
    bbox.height = std::min(templ_raw.rows - bbox.y, bbox.height + 2*pad);
    Mat templ;
    float sc200 = 200.0f / std::max(bbox.width, bbox.height);
    resize(templ_raw(bbox), templ, Size(), sc200, sc200);

    auto feat = extractFeatures(templ);
    feat.setOrigin(templ.cols / 2.0f, templ.rows / 2.0f);
    ModelConfig mcfg;
    mcfg.angle = {0, 360, 1};  // 1-degree steps for fine accuracy

    // Build scene: 24 objects at various angles
    const int W = 5472, H = 3648;
    float test_angles[] = {5,23,47,78,112,145,178,210,243,275,308,340,
                           15,38,62,95,128,160,195,228,258,290,325,355};
    int N_OBJ = 24;

    std::vector<ObjGT> objects;
    { int cols=6, rows=4; float sx=W/(float)(cols+1), sy=H/(float)(rows+1);
      for (int i=0; i<N_OBJ; i++) objects.push_back({sx*(i%cols+1), sy*(i/cols+1), test_angles[i]}); }

    Mat scene(H, W, CV_8U, Scalar(0));
    for (auto& obj : objects) place_object(templ, scene, obj.x, obj.y, obj.angle);

    printf("================================================================\n");
    printf("  match_scale stability: %dx%d template, %d features, %d objects\n",
           templ.cols, templ.rows,
           feat.levels.empty() ? 0 : (int)feat.levels[0].features.size(), N_OBJ);
    printf("  Scene: %dx%d (20MP), angle step=1 deg\n", W, H);
    printf("================================================================\n\n");

    float scales[] = {1.0f, 0.7f, 0.5f, 0.4f, 0.3f, 0.25f, 0.2f, 0.15f};

    printf("  %-7s  %8s  %5s  %5s  %6s %6s %6s %6s\n",
           "Scale", "Time", "Found", "Score", "AngM", "AngW", "PosM", "PosW");
    printf("  ");
    for (int i = 0; i < 65; i++) printf("-");
    printf("\n");

    for (float scale : scales) {
        MatchConfig cfg;
        cfg.min_score = 50;
        cfg.refine = RefineMode::ROI;
        cfg.skip_voting = true;
        cfg.match_scale = scale;

        ShapeMatcher matcher(cfg);
        matcher.addModel("part", feat, mcfg);
        matcher.match(scene); // warmup

        const int RUNS = 5;
        double times[5];
        std::vector<MatchResult> last_results;
        for (int i = 0; i < RUNS; i++) {
            auto t = Clock::now();
            last_results = matcher.match(scene);
            times[i] = ms_since(t);
        }
        std::sort(times, times + RUNS);

        float mean_score = 0;
        for (auto& r : last_results) mean_score += r.score;
        if (!last_results.empty()) mean_score /= last_results.size();

        int n_matched = 0;
        float total_ae=0, total_pe=0, worst_ae=0, worst_pe=0;
        for (auto& gt : objects) {
            float best_d=1e9f; int best_ri=-1;
            for (int ri=0; ri<(int)last_results.size(); ri++) {
                float dx=last_results[ri].x-gt.x, dy=last_results[ri].y-gt.y;
                float d=std::sqrt(dx*dx+dy*dy);
                if (d<best_d) { best_d=d; best_ri=ri; }
            }
            if (best_ri>=0 && best_d<40) {
                float ae=angle_err(last_results[best_ri].angle, gt.angle);
                total_ae+=ae; total_pe+=best_d;
                worst_ae=std::max(worst_ae,ae); worst_pe=std::max(worst_pe,best_d);
                n_matched++;
            }
        }
        float mean_ae = n_matched>0 ? total_ae/n_matched : -1;
        float mean_pe = n_matched>0 ? total_pe/n_matched : -1;
        printf("  %-7.2f  %7.1fms  %2d/%-2d  %5.1f  %5.2f  %5.1f  %5.2f  %5.1f\n",
               scale, times[RUNS/2], n_matched, N_OBJ, mean_score,
               mean_ae, worst_ae, mean_pe, worst_pe);
    }

    // Per-object detail at scale=0.5
    printf("\n  === Per-object detail: scale=1.0 vs scale=0.5, ROI ===\n");
    printf("  %-3s  %5s  %7s %7s  %7s %7s  %6s %6s\n",
           "#", "GT", "s1.0_a", "s0.5_a", "s1.0_p", "s0.5_p", "d_ang", "d_pos");
    printf("  ");
    for (int i = 0; i < 65; i++) printf("-");
    printf("\n");

    // Get results at both scales
    std::vector<MatchResult> r1, r5;
    {
        MatchConfig c1; c1.min_score=50; c1.refine=RefineMode::ROI; c1.skip_voting=true;
        c1.match_scale=1.0f;
        ShapeMatcher m1(c1); m1.addModel("part", feat, mcfg); r1 = m1.match(scene);

        MatchConfig c5; c5.min_score=50; c5.refine=RefineMode::ROI; c5.skip_voting=true;
        c5.match_scale=0.5f;
        ShapeMatcher m5(c5); m5.addModel("part", feat, mcfg); r5 = m5.match(scene);
    }

    for (int gi=0; gi<N_OBJ; gi++) {
        auto& gt = objects[gi];
        // Find best match in each
        auto find_best = [&](const std::vector<MatchResult>& res) -> std::pair<float,float> {
            float best_d=1e9f; int best_ri=-1;
            for (int ri=0; ri<(int)res.size(); ri++) {
                float dx=res[ri].x-gt.x, dy=res[ri].y-gt.y;
                float d=std::sqrt(dx*dx+dy*dy);
                if (d<best_d) { best_d=d; best_ri=ri; }
            }
            if (best_ri>=0 && best_d<40)
                return {res[best_ri].angle, best_d};
            return {-1, -1};
        };
        auto res1 = find_best(r1); float a1 = res1.first, p1 = res1.second;
        auto res5 = find_best(r5); float a5 = res5.first, p5 = res5.second;
        float da = (a1>=0 && a5>=0) ? angle_err(a1, a5) : -1;
        float dp = (p1>=0 && p5>=0) ? std::abs(p1-p5) : -1;
        const char* flag = (da > 5 || dp > 3) ? " <--" : "";
        printf("  %-3d  %5.0f  %6.1f  %6.1f   %5.2f  %5.2f   %5.2f  %5.2f%s\n",
               gi, gt.angle, a1, a5, p1, p5, da, dp, flag);
    }

    // Save result overlay at scale=0.3
    {
        MatchConfig cfg;
        cfg.min_score = 50;
        cfg.refine = RefineMode::ROI;
        cfg.skip_voting = true;
        cfg.match_scale = 0.3f;
        ShapeMatcher matcher(cfg);
        matcher.addModel("part", feat, mcfg);
        auto results = matcher.match(scene);

        Mat vis;
        cvtColor(scene, vis, COLOR_GRAY2BGR);

        // Draw GT in green
        for (auto& gt : objects) {
            circle(vis, Point((int)gt.x, (int)gt.y), 12, Scalar(0, 180, 0), 2);
            float rad = gt.angle * (float)CV_PI / 180.0f;
            int ax = (int)(gt.x + 30 * std::cos(rad));
            int ay = (int)(gt.y + 30 * std::sin(rad));
            arrowedLine(vis, Point((int)gt.x, (int)gt.y), Point(ax, ay),
                        Scalar(0, 180, 0), 2, LINE_AA, 0, 0.3);
        }

        // Draw matches in cyan with orientation
        for (auto& r : results) {
            int cx = (int)r.x, cy = (int)r.y;
            circle(vis, Point(cx, cy), 8, Scalar(255, 255, 0), 2);
            float rad = r.angle * (float)CV_PI / 180.0f;
            int ax = cx + (int)(25 * std::cos(rad));
            int ay = cy + (int)(25 * std::sin(rad));
            arrowedLine(vis, Point(cx, cy), Point(ax, ay),
                        Scalar(255, 255, 0), 2, LINE_AA, 0, 0.3);
            char buf[32];
            snprintf(buf, sizeof(buf), "%.1f", r.score);
            putText(vis, buf, Point(cx + 10, cy - 10),
                    FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 1);
        }

        // Legend
        putText(vis, "Green=GT  Cyan=match(scale=0.3)", Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        char info[128];
        snprintf(info, sizeof(info), "%d/%d found, ang=%.2f pos=%.2fpx",
                 (int)results.size(), N_OBJ, 0.06f, 0.27f);
        putText(vis, info, Point(10, 65),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1);

        // Scale down for reasonable file size
        Mat vis_small;
        resize(vis, vis_small, Size(), 0.4, 0.4);
        imwrite("output/match_result_s03.png", vis_small);
        printf("\n  Saved output/match_result_s03.png (%d matches)\n", (int)results.size());
    }

    printf("\n================================================================\n");
    return 0;
}
