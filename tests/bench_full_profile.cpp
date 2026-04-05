/// @file bench_full_profile.cpp
/// @brief Complete matching pipeline speed profile: FHD, 10 objects, ROI refine.
/// Breaks down: preprocess | coarse match | NMS | ROI refine (warp+match+PCA+solve).

#include "shape_matcher.h"
#include "line2Dup.h"
#include "roi_refine.h"
#include "sbm_log.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace sbm;
using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static void draw_L(Mat& img, int cx, int cy, int color) {
    for (int ly = -30; ly <= 30; ly++)
        for (int lx = -10; lx <= 10; lx++) {
            int px = cx+lx, py = cy+ly;
            if (px>=0 && px<img.cols && py>=0 && py<img.rows) img.at<uchar>(py,px) = (uchar)color;
        }
    for (int ly = 10; ly <= 30; ly++)
        for (int lx = 10; lx <= 40; lx++) {
            int px = cx+lx, py = cy+ly;
            if (px>=0 && px<img.cols && py>=0 && py<img.rows) img.at<uchar>(py,px) = (uchar)color;
        }
}

static void place_object(const Mat& templ, Mat& scene, int cx, int cy, double angle) {
    Mat M = getRotationMatrix2D(Point2f(templ.cols/2.0f, templ.rows/2.0f), -angle, 1.0);
    M.at<double>(0,2) += cx - templ.cols/2.0;
    M.at<double>(1,2) += cy - templ.rows/2.0;
    Mat mask = Mat::ones(templ.size(), CV_8U) * 255;
    Mat warped_templ, warped_mask;
    warpAffine(templ, warped_templ, M, scene.size());
    warpAffine(mask, warped_mask, M, scene.size());
    warped_templ.copyTo(scene, warped_mask);
}

struct ObjGT { int x, y; float angle; };

int main() {
    printf("================================================================\n");
    printf("  Complete Matching Pipeline Profile\n");
    printf("  20MP 5472x3648, 40 objects, L-shape 200x200\n");
    printf("================================================================\n\n");

    // --- Template ---
    const int TW = 200;
    Mat templ(TW, TW, CV_8U, Scalar(0));
    draw_L(templ, TW/2, TW/2, 200);

    // --- Scene with 10 objects ---
    std::vector<ObjGT> objects;
    // 40 objects spread across 20MP (5472x3648)
    {
        float angles[] = {5,15,25,35,50,65,80,95,110,125,140,155,170,185,200,215,230,245,260,275,
                          290,305,320,335,350,10,30,55,75,100,120,145,165,190,210,235,255,280,310,340};
        int cols = 8, rows = 5;
        for (int i = 0; i < 40; i++) {
            int c = i % cols, r = i / cols;
            int x = 300 + c * 650, y = 300 + r * 650;
            objects.push_back({x, y, angles[i]});
        }
    }
    Mat scene(3648, 5472, CV_8U, Scalar(50));
    for (auto& obj : objects)
        place_object(templ, scene, obj.x, obj.y, obj.angle);
    // Add noise sigma=30
    { Mat noise(scene.size(), CV_32F); RNG rng(42);
      rng.fill(noise, RNG::NORMAL, 0, 30);
      Mat f; scene.convertTo(f, CV_32F); f += noise; f.convertTo(scene, CV_8U); }

    printf("Template: %dx%d, Scene: %dx%d, %d objects, noise=0\n\n",
           TW, TW, scene.cols, scene.rows, (int)objects.size());

    // --- Features ---
    auto feat = extractFeatures(templ);
    feat.setOrigin(TW/2.0f, TW/2.0f);

    // ==========================================
    // 1. addModel profile
    // ==========================================
    printf("=== addModel ===\n");
    auto t0 = Clock::now();
    MatchConfig cfg;
    cfg.min_score = 50;
    cfg.refine = RefineMode::ROI;
    cfg.blur_kernel_size = 11;
    cfg.nms_angle = 360;  // position-only NMS to isolate edge threshold effect
    cfg.weak_threshold = 50;
    cfg.strong_threshold = 80;
    ShapeMatcher matcher(cfg);
    ModelConfig mcfg;
    mcfg.angle = {0, 360, 2};
    matcher.addModel("L", feat, mcfg);
    printf("  addModel:              %7.1f ms (180 templates + selectOpt + lock precomp)\n\n", ms_since(t0));

    // Warmup
    matcher.match(scene);

    // ==========================================
    // 2. Full match() profile (coarse only, no ROI)
    // ==========================================
    printf("=== Coarse-only match ===\n");
    {
        MatchConfig cfg_c;
        cfg_c.min_score = 50;
        cfg_c.refine = RefineMode::None;
        ShapeMatcher m_c(cfg_c);
        m_c.addModel("L", feat, mcfg);
        m_c.match(scene); // warmup

        const int N = 10;
        double times[10];
        int found = 0;
        for (int i = 0; i < N; i++) {
            auto t = Clock::now();
            auto r = m_c.match(scene);
            times[i] = ms_since(t);
            found = (int)r.size();
        }
        std::sort(times, times + N);
        printf("  Found: %d matches\n", found);
        printf("  Median: %7.2f ms,  Min: %7.2f ms\n\n", times[N/2], times[0]);

        // Draw coarse results on scene
        auto results = m_c.match(scene);
        Mat vis;
        cvtColor(scene, vis, COLOR_GRAY2BGR);

        // Draw GT in green
        for (auto& gt : objects) {
            circle(vis, Point(gt.x, gt.y), 8, Scalar(0, 255, 0), 2);
            char buf[32]; snprintf(buf, sizeof(buf), "%.0f", gt.angle);
            putText(vis, buf, Point(gt.x+10, gt.y-10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        }

        // Draw matches colored by score
        for (auto& r : results) {
            // Color: red=low score, yellow=mid, blue=high
            float t = (r.score - 30.0f) / 70.0f;
            t = std::max(0.0f, std::min(1.0f, t));
            Scalar color(255*t, 0, 255*(1-t));  // blue=high, red=low
            circle(vis, Point((int)r.x, (int)r.y), 3, color, -1);
        }

        // Add legend
        putText(vis, "Green circle = GT", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        char leg[128];
        snprintf(leg, sizeof(leg), "%d matches (min_score=30), red=low blue=high", (int)results.size());
        putText(vis, leg, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

        imwrite("output/coarse_positions.png", vis);
        printf("  Saved: output/coarse_positions.png\n\n");
    }

    // ==========================================
    // 3. Full match() with ROI refine
    // ==========================================
    printf("=== match() with ROI refine ===\n");
    {
        const int N = 10;
        double times[10];
        int found = 0;
        for (int i = 0; i < N; i++) {
            auto t = Clock::now();
            auto r = matcher.match(scene);
            times[i] = ms_since(t);
            found = (int)r.size();
        }
        std::sort(times, times + N);
        printf("  Found: %d matches\n", found);
        printf("  Median: %7.2f ms,  Min: %7.2f ms\n", times[N/2], times[0]);
        printf("  ROI overhead: ~%.2f ms (= ROI_total - coarse_only)\n\n",
               times[N/2]);  // will compute delta below

        // Draw ROI-refined results
        auto roi_results = matcher.match(scene);
        Mat vis_roi;
        cvtColor(scene, vis_roi, COLOR_GRAY2BGR);

        // Draw GT in green
        for (auto& gt : objects) {
            circle(vis_roi, Point(gt.x, gt.y), 8, Scalar(0, 255, 0), 2);
            char buf[32]; snprintf(buf, sizeof(buf), "%.0f", gt.angle);
            putText(vis_roi, buf, Point(gt.x+10, gt.y-10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        }

        // Draw ROI-refined positions with orientation axes
        for (auto& r : roi_results) {
            int cx = (int)r.x, cy = (int)r.y;
            float rad = r.angle * (float)CV_PI / 180.0f;
            float cs = std::cos(rad), sn = std::sin(rad);
            int axis_len = 40;

            // X-axis (red)
            Point2f xend(cx + axis_len * cs, cy + axis_len * sn);
            arrowedLine(vis_roi, Point(cx, cy), Point((int)xend.x, (int)xend.y),
                        Scalar(0, 0, 255), 2, LINE_AA, 0, 0.2);

            // Y-axis (green) — perpendicular to X
            Point2f yend(cx - axis_len * sn, cy + axis_len * cs);
            arrowedLine(vis_roi, Point(cx, cy), Point((int)yend.x, (int)yend.y),
                        Scalar(0, 255, 0), 2, LINE_AA, 0, 0.2);

            // Center dot
            circle(vis_roi, Point(cx, cy), 3, Scalar(255, 255, 0), -1);

            // Label
            char buf[64]; snprintf(buf, sizeof(buf), "%.1f", r.angle);
            putText(vis_roi, buf, Point(cx+10, cy-10), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 1);
        }

        // Also get coarse-only results — with smaller NMS to see competing matches
        MatchConfig cfg_cmp;
        cfg_cmp.min_score = 40;  // lower to see what else is near
        cfg_cmp.nms_radius = 10; // small NMS to keep competing matches
        cfg_cmp.refine = RefineMode::None;
        cfg_cmp.blur_kernel_size = 11;
        cfg_cmp.weak_threshold = 50;
        cfg_cmp.strong_threshold = 80;
        ShapeMatcher m_cmp(cfg_cmp);
        m_cmp.addModel("L", feat, mcfg);
        auto coarse_res = m_cmp.match(scene);

        // Print per-object error
        printf("\n  Per-object accuracy (GT vs Coarse vs ROI refined):\n");
        printf("    %-4s  %-12s %-12s %-12s  %-8s %-8s %-8s  %-8s %-8s\n",
               "#", "GT pos", "Coarse pos", "ROI pos", "GT ang", "Crs ang", "ROI ang", "pos_err", "ang_err");
        for (auto& gt : objects) {
            float best_dist = 1e6;
            int best_idx = -1;
            for (int ri = 0; ri < (int)roi_results.size(); ri++) {
                float dx = roi_results[ri].x - gt.x, dy = roi_results[ri].y - gt.y;
                float d = std::sqrt(dx*dx + dy*dy);
                if (d < best_dist) { best_dist = d; best_idx = ri; }
            }
            if (best_idx >= 0) {
                auto& r = roi_results[best_idx];
                float ae = std::fmod(std::abs(r.angle - gt.angle), 360.0f);
                ae = std::min(ae, 360.0f - ae);
                // Find matching coarse result
                float crs_x = 0, crs_y = 0, crs_ang = 0;
                for (auto& cr : coarse_res) {
                    float dx = cr.x - gt.x, dy = cr.y - gt.y;
                    if (dx*dx + dy*dy < 50*50) { crs_x = cr.x; crs_y = cr.y; crs_ang = cr.angle; break; }
                }
                const char* flag = (best_dist > 5 || ae > 5) ? " <-- BAD" : "";
                printf("    %-4d  (%4d,%4d)  (%6.1f,%6.1f) (%6.1f,%6.1f)  %6.1f  %6.1f  %6.1f   %6.1f   %6.1f%s\n",
                       (int)(&gt - &objects[0]), gt.x, gt.y,
                       crs_x, crs_y, r.x, r.y,
                       gt.angle, crs_ang, r.angle, best_dist, ae, flag);
            }
        }

        // Debug: show all coarse matches near the failing object (#10)
        {
            ObjGT& gt10 = objects[10];
            printf("\n  All coarse matches near GT #10 (%.0f,%.0f ang=%.0f):\n", (float)gt10.x, (float)gt10.y, gt10.angle);
            printf("    %-6s %-12s %-8s %-8s\n", "idx", "pos", "angle", "score");
            for (int i = 0; i < (int)coarse_res.size(); i++) {
                auto& cr = coarse_res[i];
                float dx = cr.x - gt10.x, dy = cr.y - gt10.y;
                if (dx*dx + dy*dy < 80*80) {
                    printf("    %-6d (%6.1f,%6.1f) %6.1f  %6.1f\n",
                           i, cr.x, cr.y, cr.angle, cr.score);
                }
            }
        }

        putText(vis_roi, "Green = GT, Cyan = ROI refined", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
        char leg2[128];
        snprintf(leg2, sizeof(leg2), "%d matches (min_score=50, ROI refine)", (int)roi_results.size());
        putText(vis_roi, leg2, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

        imwrite("output/roi_positions.png", vis_roi);
        printf("  Saved: output/roi_positions.png\n");
    }

    // ==========================================
    // 4. LineMOD internal stage breakdown
    // ==========================================
    printf("=== LineMOD internal stages (single call, coarse-only) ===\n");
    sbm::setLogLevel(sbm::LogLevel::Debug);
    sbm::setLogFile(stdout);
    line2Dup::enableProfiling(true);
    line2Dup::resetProfiling();
    {
        MatchConfig cfg_c;
        cfg_c.min_score = 50;
        cfg_c.refine = RefineMode::None;
        ShapeMatcher m_c(cfg_c);
        m_c.addModel("L", feat, mcfg);
        m_c.match(scene); // warmup
        line2Dup::resetProfiling();
        m_c.match(scene);
    }
    line2Dup::printProfiling();
    line2Dup::enableProfiling(false);
    sbm::setLogFile(nullptr);
    sbm::setLogLevel(sbm::LogLevel::Warning);
    printf("\n");

    // ==========================================
    // 5. ROI refine isolated profile
    // ==========================================
    printf("=== ROI Refine isolated ===\n");

    // Get coarse results
    MatchConfig cfg_c;
    cfg_c.min_score = 50;
    cfg_c.refine = RefineMode::None;
    ShapeMatcher m_coarse(cfg_c);
    m_coarse.addModel("L", feat, mcfg);
    auto coarse = m_coarse.match(scene);
    printf("  Coarse results: %d\n", (int)coarse.size());

    // Build sample points
    auto opt_pts = feat.selectOptimizedPoints(8);
    std::vector<roi_refine::SamplePoint> spts;
    for (int pi = 0; pi < (int)opt_pts.size(); pi++) {
        roi_refine::SamplePoint sp;
        sp.pos = opt_pts[pi];
        if (pi < (int)feat.cached_lock_info.size()) {
            sp.lock_major = feat.cached_lock_info[pi].major;
            sp.lock_minor = feat.cached_lock_info[pi].minor;
        }
        spts.push_back(sp);
    }
    printf("  Sample points: %d, roi_half=15, search_half=15, max_iters=3\n", (int)spts.size());

    roi_refine::ROIConfig roi_cfg;
    roi_cfg.roi_half = 15;
    roi_cfg.search_half = 15;
    roi_cfg.max_iters = 3;

    // Use only the first 10 matches (true objects)
    int n_obj = std::min(10, (int)coarse.size());

    // Warmup
    for (int i = 0; i < n_obj; i++) {
        cv::Vec3f init(coarse[i].x, coarse[i].y, coarse[i].angle);
        roi_refine::refineROI(feat.templ_image, scene, spts, init, roi_cfg);
    }

    // Profile total refineROI
    const int RUNS = 50;
    double total_roi = 0;
    std::vector<double> per_obj(n_obj, 0);

    for (int run = 0; run < RUNS; run++) {
        for (int i = 0; i < n_obj; i++) {
            cv::Vec3f init(coarse[i].x, coarse[i].y, coarse[i].angle);
            auto t = Clock::now();
            roi_refine::refineROI(feat.templ_image, scene, spts, init, roi_cfg);
            double dt = ms_since(t);
            per_obj[i] += dt;
            total_roi += dt;
        }
    }

    printf("\n  Per-object refineROI (avg of %d runs):\n", RUNS);
    for (int i = 0; i < n_obj; i++)
        printf("    obj %d (angle=%3.0f): %.3f ms\n", i, coarse[i].angle, per_obj[i]/RUNS);
    printf("    --------\n");
    printf("    avg per object: %.3f ms\n", total_roi / RUNS / n_obj);
    printf("    total %d obj:   %.3f ms (sequential)\n", n_obj, total_roi / RUNS);

    // ==========================================
    // 6. ROI sub-stage breakdown
    // ==========================================
    printf("\n=== ROI sub-stage breakdown (single object, %d runs) ===\n", RUNS);
    if (n_obj > 0) {
        float tcx = feat.templ_width / 2.0f, tcy = feat.templ_height / 2.0f;
        double t_warp = 0, t_match = 0, t_pca = 0, t_solve = 0;

        for (int run = 0; run < RUNS; run++) {
            float angle_deg = coarse[0].angle;
            float angle_rad = angle_deg * (float)CV_PI / 180.0f;
            float cs = std::cos(angle_rad), sn = std::sin(angle_rad);

            // Warp
            auto tw = Clock::now();
            for (auto& sp : spts) {
                int tx = (int)(sp.pos.x + tcx + 0.5f), ty = (int)(sp.pos.y + tcy + 0.5f);
                int h = sp.roi_half;
                if (tx-h<0||ty-h<0||tx+h>=feat.templ_image.cols||ty+h>=feat.templ_image.rows) continue;
                Mat roi_unrot = feat.templ_image(Rect(tx-h,ty-h,2*h,2*h));
                Mat roi, M = getRotationMatrix2D(Point2f((float)h,(float)h), -angle_deg, 1.0);
                warpAffine(roi_unrot, roi, M, roi_unrot.size(), INTER_LINEAR, BORDER_REPLICATE);
            }
            t_warp += ms_since(tw);

            // matchTemplate
            auto tm = Clock::now();
            for (auto& sp : spts) {
                int tx = (int)(sp.pos.x + tcx + 0.5f), ty = (int)(sp.pos.y + tcy + 0.5f);
                int h = sp.roi_half;
                if (tx-h<0||ty-h<0||tx+h>=feat.templ_image.cols||ty+h>=feat.templ_image.rows) continue;
                Mat roi_unrot = feat.templ_image(Rect(tx-h,ty-h,2*h,2*h));
                Mat roi, M_r = getRotationMatrix2D(Point2f((float)h,(float)h), -angle_deg, 1.0);
                warpAffine(roi_unrot, roi, M_r, roi_unrot.size(), INTER_LINEAR, BORDER_REPLICATE);

                float ex = cs*sp.pos.x - sn*sp.pos.y + coarse[0].x;
                float ey = sn*sp.pos.x + cs*sp.pos.y + coarse[0].y;
                int iex=(int)(ex+0.5f), iey=(int)(ey+0.5f);
                int sh = roi_cfg.search_half;
                int x0=std::max(0,iex-sh-h), y0=std::max(0,iey-sh-h);
                int x1=std::min(scene.cols,iex+sh+h), y1=std::min(scene.rows,iey+sh+h);
                if (x1-x0<roi.cols||y1-y0<roi.rows) continue;
                Mat search = scene(Rect(x0,y0,x1-x0,y1-y0));
                Mat result;
                matchTemplate(search, roi, result, TM_CCORR_NORMED);
                double mv; Point ml; minMaxLoc(result,nullptr,&mv,nullptr,&ml);
            }
            t_match += ms_since(tm);

            // PCA (Sobel + magnitude + scatter)
            auto tp = Clock::now();
            for (auto& sp : spts) {
                int tx = (int)(sp.pos.x + tcx + 0.5f), ty = (int)(sp.pos.y + tcy + 0.5f);
                int h = sp.roi_half;
                if (tx-h<0||ty-h<0||tx+h>=feat.templ_image.cols||ty+h>=feat.templ_image.rows) continue;
                Mat roi_unrot = feat.templ_image(Rect(tx-h,ty-h,2*h,2*h));
                Mat dx, dy, mag;
                Sobel(roi_unrot, dx, CV_32F, 1, 0, 3);
                Sobel(roi_unrot, dy, CV_32F, 0, 1, 3);
                magnitude(dx, dy, mag);
                // Scatter computation
                float thresh = 0.3f * *std::max_element(mag.begin<float>(), mag.end<float>());
                float cxx=0,cyy=0,cxy=0; int cnt=0;
                for (int r=0;r<roi_unrot.rows;r++)
                    for (int c=0;c<roi_unrot.cols;c++)
                        if (mag.at<float>(r,c)>thresh) { cxx+=c*c; cyy+=r*r; cxy+=c*r; cnt++; }
            }
            t_pca += ms_since(tp);

            // Solve (3x3)
            auto ts = Clock::now();
            for (int iter = 0; iter < roi_cfg.max_iters; iter++) {
                float ATA[3][3]={}, ATb[3]={};
                for (int i=0;i<(int)spts.size();i++) {
                    float sx=spts[i].pos.x, sy=spts[i].pos.y;
                    float j0=-sy, j1=1.0f, j2=0.0f, e=0.01f;
                    ATA[0][0]+=j0*j0; ATA[0][1]+=j0*j1; ATA[1][1]+=j1*j1; ATA[2][2]+=1;
                    ATb[0]-=j0*e; ATb[1]-=j1*e;
                }
                ATA[1][0]=ATA[0][1]; ATA[2][0]=ATA[0][2]; ATA[2][1]=ATA[1][2];
                for(int i=0;i<3;i++) ATA[i][i]+=0.001f;
                Mat A(3,3,CV_32F,ATA), b(3,1,CV_32F,ATb), x;
                cv::solve(A,b,x);
            }
            t_solve += ms_since(ts);
        }

        int n_pts = (int)spts.size();
        int patch_sz = 2*roi_cfg.roi_half;
        int search_sz = 2*(roi_cfg.search_half + roi_cfg.roi_half);
        printf("  Patch: %dx%d, Search: %dx%d, %d points, %d iters\n\n",
               patch_sz, patch_sz, search_sz, search_sz, n_pts, roi_cfg.max_iters);

        double t_total = t_warp + t_match + t_pca + t_solve;
        printf("  %-32s %7.3f ms  (%5.1f%%)\n", "warpAffine",
               t_warp/RUNS, 100.0*t_warp/t_total);
        printf("  %-32s %7.3f ms  (%5.1f%%)\n", "matchTemplate (incl re-warp)",
               t_match/RUNS, 100.0*t_match/t_total);
        printf("  %-32s %7.3f ms  (%5.1f%%)\n", "PCA (Sobel+mag+scatter)",
               t_pca/RUNS, 100.0*t_pca/t_total);
        printf("  %-32s %7.3f ms  (%5.1f%%)\n", "Solve 3x3 (x3 iters)",
               t_solve/RUNS, 100.0*t_solve/t_total);
        printf("  %-32s %7.3f ms\n", "TOTAL sub-stages (1 object)",
               t_total/RUNS);
    }

    printf("\n================================================================\n");
    return 0;
}
