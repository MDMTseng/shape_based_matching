/// @file bench_match_scale.cpp
/// @brief T-level x match_scale using addRotatedTemplates on scaled template.

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

    printf("================================================================\n");
    printf("  T-level x match_scale: %dx%d template, 24 objects, 20MP\n",
           templ.cols, templ.rows);
    printf("================================================================\n\n");

    const int W = 5472, H = 3648;
    float test_angles[] = {5,23,47,78,112,145,178,210,243,275,308,340,
                           15,38,62,95,128,160,195,228,258,290,325,355};
    int N_OBJ = 24;

    std::vector<ObjGT> objects;
    { int cols=6, rows=4; float sx=W/(float)(cols+1), sy=H/(float)(rows+1);
      for (int i=0; i<N_OBJ; i++) objects.push_back({sx*(i%cols+1), sy*(i/cols+1), test_angles[i]}); }

    Mat scene_full(H, W, CV_8U, Scalar(0));
    for (auto& obj : objects) place_object(templ, scene_full, obj.x, obj.y, obj.angle);

    struct TConfig { const char* name; std::vector<int> T; };
    TConfig t_configs[] = {
        {"{4,8}",    {4, 8}},
        {"{6,8}",    {6, 8}},
        {"{6,12}",   {6, 12}},
        {"{8,16}",   {8, 16}},
    };
    float scales[] = {1.0f, 0.5f, 0.3f};

    printf("  %-10s %-7s  %8s  %5s  %5s  %5s %5s %5s %5s\n",
           "T", "Scale", "Time", "Found", "Score", "AngM", "AngW", "PosM", "PosW");
    printf("  ");
    for (int i = 0; i < 75; i++) printf("-");
    printf("\n");

    Mat mask_t = Mat::ones(templ.size(), CV_8U) * 255;

    for (auto& tc : t_configs) {
        for (float scale : scales) {
            // Downscale scene
            Mat match_scene;
            float inv_scale = 1.0f;
            if (scale < 0.99f) {
                inv_scale = 1.0f / scale;
                resize(scene_full, match_scene, Size((int)(W*scale), (int)(H*scale)));
            } else {
                match_scene = scene_full;
            }

            // Downscale template, then use addRotatedTemplates on it
            // This ensures features + labels are extracted from the scaled image
            Mat match_templ, match_mask;
            if (scale < 0.99f) {
                resize(templ, match_templ, Size((int)(templ.cols*scale+0.5f),
                                                (int)(templ.rows*scale+0.5f)));
                match_mask = Mat::ones(match_templ.size(), CV_8U) * 255;
            } else {
                match_templ = templ;
                match_mask = mask_t;
            }

            // Check minimum template size for this T config
            int min_templ = std::min(match_templ.cols, match_templ.rows);
            int deepest_T = tc.T.back();
            int deepest_dim = min_templ >> ((int)tc.T.size() - 1);
            if (deepest_dim < deepest_T * 3) {
                printf("  %-10s %-7.2f  SKIP (templ %dpx at deepest level, T=%d)\n",
                       tc.name, scale, deepest_dim, deepest_T);
                continue;
            }

            // Build detector and add templates
            line2Dup::Detector det(128, tc.T, 30, 60);
            det.scale_pyramid_features = true;
            int n_added = det.addRotatedTemplates(match_templ, match_mask, "part", 0, 360, 1);
            if (n_added <= 0) {
                printf("  %-10s %-7.2f  FAILED\n", tc.name, scale);
                continue;
            }

            // Pad scene: must be divisible by T[l] * 2^l for each level
            auto lcm = [](int a, int b) { int g=a,h=b,t; while(h){t=h;h=g%h;g=t;} return a/g*b; };
            int align = 1;
            for (int l = 0; l < (int)tc.T.size(); l++)
                align = lcm(align, tc.T[l] * (1 << l));
            int pw = (match_scene.cols + align - 1) / align * align;
            int ph = (match_scene.rows + align - 1) / align * align;
            Mat padded;
            if (pw != match_scene.cols || ph != match_scene.rows)
                copyMakeBorder(match_scene, padded, 0, ph-match_scene.rows,
                               0, pw-match_scene.cols, BORDER_CONSTANT, Scalar(0));
            else padded = match_scene;

            det.match(padded, 50); // warmup
            const int RUNS = 5;
            double times[5];
            std::vector<line2Dup::Match> last;
            for (int i = 0; i < RUNS; i++) {
                auto t = Clock::now();
                last = det.match(padded, 50);
                times[i] = ms_since(t);
            }
            std::sort(times, times + RUNS);

            float mean_score = 0;
            for (auto& r : last) mean_score += r.similarity;
            if (!last.empty()) mean_score /= last.size();

            // NMS
            float nms_r2 = (80*scale)*(80*scale);
            std::vector<line2Dup::Match> nms;
            for (auto& m : last) {
                bool sup = false;
                for (auto& k : nms) {
                    float dx=(float)(m.x-k.x), dy=(float)(m.y-k.y);
                    if (dx*dx+dy*dy < nms_r2) { sup=true; break; }
                }
                if (!sup) nms.push_back(m);
            }

            // Match to GT
            int n_matched = 0;
            float tae=0, tpe=0, wae=0, wpe=0;
            for (auto& gt : objects) {
                float best_d=1e9f; int best_ri=-1;
                for (int ri=0; ri<(int)nms.size(); ri++) {
                    auto& mtpl = det.getTemplates("part", nms[ri].template_id);
                    float cx = (nms[ri].x + match_templ.cols/2.0f - mtpl[0].tl_x) * inv_scale;
                    float cy = (nms[ri].y + match_templ.rows/2.0f - mtpl[0].tl_y) * inv_scale;
                    float dx=cx-gt.x, dy=cy-gt.y;
                    float d=std::sqrt(dx*dx+dy*dy);
                    if (d<best_d) { best_d=d; best_ri=ri; }
                }
                if (best_ri>=0 && best_d<40) {
                    float matched_angle = nms[best_ri].template_id * 1.0f;
                    float ae=angle_err(matched_angle, gt.angle);
                    tae+=ae; tpe+=best_d;
                    wae=std::max(wae,ae); wpe=std::max(wpe,best_d);
                    n_matched++;
                }
            }
            float mae = n_matched>0 ? tae/n_matched : -1;
            float mpe = n_matched>0 ? tpe/n_matched : -1;
            const char* status = (n_matched==N_OBJ && mae<2.0) ? " <<<" :
                                 (n_matched<N_OBJ) ? " MISS" : "";
            printf("  %-10s %-7.2f  %7.1fms  %2d/%-2d  %5.1f  %5.1f %5.1f %5.1f %5.1f%s\n",
                   tc.name, scale, times[RUNS/2], n_matched, N_OBJ, mean_score,
                   mae, wae, mpe, wpe, status);
        }
        printf("\n");
    }
    printf("================================================================\n");
    return 0;
}
