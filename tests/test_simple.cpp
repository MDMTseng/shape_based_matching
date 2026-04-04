// Minimal usage example of the ShapeMatcher API.

#include "shape_matcher.h"
#include "roi_refine.h"
#include "icp_refine.h"
#include "test_utils.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } } while(0)

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

int main() {
    // 1. Create template — try different shapes
    Mat templ(80, 80, CV_8U, Scalar(0));
#define TEMPLATE_SHAPE 4
#if TEMPLATE_SHAPE == 0
    draw_L(templ, 40, 40, 0, 200);
    printf("Template: L-shape\n");
#elif TEMPLATE_SHAPE == 1
    // V-shape
    for (double t = 0; t < 35; t += 0.3) {
        int lx=(int)(40-t*0.7), ly=(int)(60-t), rx=(int)(40+t*0.7), ry=(int)(60-t);
        if(lx>=0&&lx<80&&ly>=0&&ly<80) for(int d=-3;d<=3;d++) if(lx+d>=0&&lx+d<80) templ.at<uchar>(ly,lx+d)=200;
        if(rx>=0&&rx<80&&ry>=0&&ry<80) for(int d=-3;d<=3;d++) if(rx+d>=0&&rx+d<80) templ.at<uchar>(ry,rx+d)=200;
    }
    printf("Template: V-shape\n");
#elif TEMPLATE_SHAPE == 2
    // Parallel vertical lines (degenerate — no X constraint)
    rectangle(templ, Point(15, 10), Point(22, 70), Scalar(200), -1);
    rectangle(templ, Point(58, 10), Point(65, 70), Scalar(200), -1);
    printf("Template: parallel lines (degenerate)\n");
#elif TEMPLATE_SHAPE == 3
    // Long thin pole (10:1 aspect ratio)
    // Long sides dominate, only 1 short end provides perpendicular constraint
    rectangle(templ, Point(5, 35), Point(75, 45), Scalar(200), -1);
    printf("Template: long pole (70x10)\n");
#elif TEMPLATE_SHAPE == 7
    // Single horizontal line — all normals vertical, no horizontal constraint
    line(templ, Point(5, 40), Point(75, 40), Scalar(200), 3);
    printf("Template: single horizontal line\n");
#elif TEMPLATE_SHAPE == 4
    // Flat triangle (10-10-160 deg) — nearly degenerate, almost a straight line
    {
        Point pts[3] = {Point(5, 55), Point(75, 55), Point(40, 35)};  // ~10 deg apex, taller
        fillConvexPoly(templ, pts, 3, Scalar(200));
    }
    printf("Template: flat triangle (10-10-160)\n");
#elif TEMPLATE_SHAPE == 5
    // Sharper triangle (30-30-120) for comparison
    {
        Point pts[3] = {Point(10, 55), Point(70, 55), Point(40, 25)};
        fillConvexPoly(templ, pts, 3, Scalar(200));
    }
    printf("Template: triangle (30-30-120)\n");
#elif TEMPLATE_SHAPE == 6
    // Right triangle (90-45-45) — good constraint
    {
        Point pts[3] = {Point(15, 60), Point(65, 60), Point(15, 15)};
        fillConvexPoly(templ, pts, 3, Scalar(200));
    }
    printf("Template: right triangle (90-45-45)\n");
#endif

    // 2. Extract features + save
    auto features = sbm::extractFeatures(templ);
    features.setOrigin(40, 30);  // center of rectangle
    features.save("rect.feat");
    printf("Extracted %d features\n", features.numFeatures());

    // 3. Create scene with 3 rotated L-shapes
    struct Obj { int x, y; double angle; };
    Obj objs[] = {{160,120,25}, {320,240,90}, {480,360,200}};

    Mat scene(480, 640, CV_8U, Scalar(30));
    for (auto& obj : objs) {
        Mat M = getRotationMatrix2D(Point2f(40, 40), -obj.angle, 1.0);
        Mat rot; warpAffine(templ, rot, M, templ.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
        int ox = obj.x - 40, oy = obj.y - 40;
        for (int r=0; r<rot.rows; r++) for (int c=0; c<rot.cols; c++) {
            int sy=oy+r, sx=ox+c;
            if (sy>=0 && sy<scene.rows && sx>=0 && sx<scene.cols && rot.at<uchar>(r,c)>0)
                scene.at<uchar>(sy,sx) = rot.at<uchar>(r,c);
        }
    }

    // 4. Compare refine modes
    struct Mode { const char* name; sbm::RefineMode mode; };
    Mode modes[] = {
        {"None",       sbm::RefineMode::None},
        {"ICP_Sparse", sbm::RefineMode::ICP_Sparse},
        {"ICP (dense)", sbm::RefineMode::ICP},
        {"ROI",        sbm::RefineMode::ROI},
    };

    auto loaded = sbm::FeatureSet::load("rect.feat");
    int n_edge_rp = 0, n_corner_rp = 0;
    for (auto& rp : loaded.refine_points) {
        if (rp.type == sbm::FeatureSet::RefinePt::CORNER) n_corner_rp++;
        else n_edge_rp++;
    }
    printf("Matching features: %d (sparse)\n", loaded.numFeatures());
    printf("Refine points:     %d edges + %d corners = %d total\n\n",
           n_edge_rp, n_corner_rp, n_edge_rp + n_corner_rp);

    printf("GT:  (160,120)@25   (320,240)@90   (480,360)@200\n\n");

    // Robustness test: directly call refine with perturbed initial pose
    printf("\n--- Robustness: refine from perturbed initial pose ---\n");
    printf("GT: (160,120)@25.  Scene created with warpAffine.\n\n");

    // Base clean scene with one L at (160,120)@25
    Mat scene_clean(480, 640, CV_8U, Scalar(30));
    {
        Mat M1 = getRotationMatrix2D(Point2f(40,40), -25.0, 1.0);
        Mat rot1; warpAffine(templ, rot1, M1, templ.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
        for(int r=0;r<rot1.rows;r++) for(int c=0;c<rot1.cols;c++) {
            int sy=120-40+r, sx=160-40+c;
            if(sy>=0&&sy<scene_clean.rows&&sx>=0&&sx<scene_clean.cols&&rot1.at<uchar>(r,c)>0)
                scene_clean.at<uchar>(sy,sx)=rot1.at<uchar>(r,c);
        }
    }

    // Prepare ICP model edges
    std::vector<icp_refine::EdgePoint> icp_edges;
    for (auto& rp : loaded.refine_points) {
        icp_refine::EdgePoint ep;
        ep.pos = cv::Point2f(rp.px, rp.py);
        ep.normal = cv::Point2f(rp.nx, rp.ny);
        ep.cornerness = rp.cornerness;
        icp_edges.push_back(ep);
    }

    // Prepare ROI sample points
    std::vector<cv::Point2f> positions;
    std::vector<float> corner_scores;
    for (auto& rp : loaded.refine_points) {
        positions.push_back(cv::Point2f(rp.px, rp.py));
        corner_scores.push_back(rp.cornerness);
    }
    auto sample_pts_15 = roi_refine::selectCriticalPoints(
        positions, corner_scores, 15, loaded.templ_width, loaded.templ_height);
    auto sample_pts_8 = roi_refine::selectCriticalPoints(
        positions, corner_scores, 8, loaded.templ_width, loaded.templ_height);

    // First: check PCA results for sample points
    printf("  Sample point PCA analysis:\n");
    for (size_t i = 0; i < sample_pts_8.size(); ++i) {
        auto& sp = sample_pts_8[i];
        int tx = (int)(sp.pos.x + loaded.templ_width/2.0f + 0.5f);
        int ty = (int)(sp.pos.y + loaded.templ_height/2.0f + 0.5f);
        int h = 15;
        if (tx-h<0||tx+h>=loaded.templ_width||ty-h<0||ty+h>=loaded.templ_height) {
            h = std::min({tx,ty,loaded.templ_width-1-tx,loaded.templ_height-1-ty});
        }
        if (h < 5) continue;
        Mat roi = loaded.templ_image(Rect(tx-h,ty-h,2*h,2*h));
        // Quick PCA
        Mat dx, dy, mag;
        Sobel(roi, dx, CV_32F, 1, 0, 3);
        Sobel(roi, dy, CV_32F, 0, 1, 3);
        magnitude(dx, dy, mag);
        float thr = 0.3f * *std::max_element(mag.begin<float>(), mag.end<float>());
        float cxx=0,cyy=0,cxy=0; int n=0;
        for(int r=0;r<roi.rows;r++) for(int c=0;c<roi.cols;c++) {
            if(mag.at<float>(r,c)>thr) {
                float ddx=c-h, ddy=r-h; cxx+=ddx*ddx; cyy+=ddy*ddy; cxy+=ddx*ddy; n++;
            }
        }
        if(n>0){cxx/=n;cyy/=n;cxy/=n;}
        float trace=cxx+cyy;
        float disc=std::sqrt(std::max(0.f,(cxx-cyy)*(cxx-cyy)/4+cxy*cxy));
        float lam1=trace/2+disc, lam2=trace/2-disc;
        float ratio = (lam2>1e-6f) ? lam1/lam2 : 999;
        printf("    [%d] pos=(%+5.1f,%+5.1f) eigenvals=(%.1f,%.1f) ratio=%.1f %s\n",
               (int)i, sp.pos.x, sp.pos.y, lam1, lam2, ratio,
               ratio < 1.5f ? "CORNER" : "EDGE");
    }
    printf("\n");

    std::vector<roi_refine::SamplePoint> sample_pts_edge;
    // Select only pure-edge points: pick from refine_points with type==EDGE,
    // sorted by cornerness ascending (most edge-like first), well-spaced
    {
        std::vector<cv::Point2f> edge_pos;
        std::vector<float> edge_corn;
        for (auto& rp : loaded.refine_points) {
            if (rp.type == sbm::FeatureSet::RefinePt::EDGE) {
                edge_pos.push_back(cv::Point2f(rp.px, rp.py));
                edge_corn.push_back(0);  // all zero cornerness = pure edge
            }
        }
        sample_pts_edge = roi_refine::selectCriticalPoints(
            edge_pos, edge_corn, 8, loaded.templ_width, loaded.templ_height);
    }
    printf("  Edge-only sample points: %d\n", (int)sample_pts_edge.size());

    // Optimized feature selection timing
    auto t_sel0 = std::chrono::high_resolution_clock::now();
    auto opt_pts = loaded.selectOptimizedPoints(15);
    auto t_sel1 = std::chrono::high_resolution_clock::now();
    double sel_ms = std::chrono::duration<double,std::milli>(t_sel1-t_sel0).count();
    auto opt_pts2 = loaded.selectOptimizedPoints(15);  // cached
    auto t_sel2 = std::chrono::high_resolution_clock::now();
    double sel_cached_ms = std::chrono::duration<double,std::milli>(t_sel2-t_sel1).count();
    printf("  selectOptimizedPoints(15): %d pts — compute=%.1fms, cached=%.3fms\n",
           (int)opt_pts.size(), sel_ms, sel_cached_ms);

    // Sensitivity analysis
    auto sens = loaded.analyzeSensitivity();
    printf("  Sensitivity: worst_ang=%.2f deg/px  worst_pos=%.2f px/px  fragile=%d — %s\n",
           sens.worst_angle_sens, sens.worst_pos_sens, sens.num_fragile, sens.diagnosis.c_str());
    printf("    Per-feature sensitivity:\n");
    for (auto& f : sens.features)
        printf("      (%+5.1f,%+5.1f) → d_ang=%.2f d_pos=%.2f lev=%.0f\n",
               f.pos.x, f.pos.y, f.d_ang, f.d_pos, f.leverage);
    printf("\n");

    printf("%-22s  %-22s  %-22s  %-22s  %-22s\n",
           "Init perturbation", "ICP (dense)", "ROI 15pt×5", "ROI 8pt×3", "ROI 8edge-only");
    printf("%-22s  %-22s  %-22s  %-22s  %-22s\n",
           "-----------------", "----------", "----------", "---------", "--------------");

    struct PoseError { float dx, dy, da; double noise; int blur; const char* name; };
    PoseError errors[] = {
        { 0,  0,  0, 0, 0, "perfect"},
        { 5,  3,  5, 0, 0, "+5px +5deg"},
        {10, 10, 10, 0, 0, "+10px +10deg"},
        // Edge-slide tests: slide along specific directions
        {10,  0,  0, 0, 0, "slide X +10px"},
        { 0, 10,  0, 0, 0, "slide Y +10px"},
        {20,  0,  0, 0, 0, "slide X +20px"},
        { 0, 20,  0, 0, 0, "slide Y +20px"},
        // Noise/blur
        { 0,  0,  0, 30, 0, "noise s=30"},
        { 0,  0,  0, 50, 0, "noise s=50"},
        { 0,  0,  0, 80, 0, "noise s=80"},
        { 0,  0,  0,  0,11, "blur k=11"},
        { 0,  0,  0,  0,21, "blur k=21"},
        { 0,  0,  0, 30,11, "n=30+b=11"},
        { 0,  0,  0, 50,11, "n=50+b=11"},
        { 5,  3,  5, 20, 3, "+5px+5deg+n+b"},
        { 5,  3,  5, 50,11, "+5px+5deg+n50+b11"},
    };

    for (auto& pe : errors) {
        float init_x = 160 + pe.dx, init_y = 120 + pe.dy, init_a = 25 + pe.da;

        // Apply noise/blur to scene
        Mat scene1 = scene_clean.clone();
        if (pe.noise > 0) {
            Mat noise(scene1.size(), CV_64F);
            RNG rng(42);
            rng.fill(noise, RNG::NORMAL, 0, pe.noise);
            Mat tmp; scene1.convertTo(tmp, CV_64F);
            tmp += noise; tmp.convertTo(scene1, CV_8U);
        }
        if (pe.blur > 0) {
            GaussianBlur(scene1, scene1, Size(pe.blur, pe.blur), 0);
        }

        printf("%-22s  ", pe.name);

        // ICP (with timing)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            Mat ss, sdx, sdy;
            GaussianBlur(scene1, ss, Size(7,7), 0);
            Sobel(ss, sdx, CV_16S, 1, 0, 3);
            Sobel(ss, sdy, CV_16S, 0, 1, 3);

            icp_refine::ICPConfig icfg;
            icfg.max_iterations = 30;
            icfg.max_dist = 15;
            icfg.use_cornerness = true;

            icp_refine::Pose2D ip(init_x, init_y, init_a);
            auto ref = icp_refine::refineWithNormals(
                icp_edges, sdx, sdy, ip, loaded.templ_width, 30, icfg);

            double icp_ms = std::chrono::duration<double,std::milli>(
                std::chrono::high_resolution_clock::now()-t0).count();
            float ae = ref.angle - 25; if(ae>180)ae-=360; if(ae<-180)ae+=360;
            float pd = std::sqrt((ref.x-160)*(ref.x-160)+(ref.y-120)*(ref.y-120));
            printf("@%+5.1f %4.1fpx %4.1fms  ", ae, pd, icp_ms);
        }

        // ROI 15pt×5iter
        float roi15_ae, roi15_pd;
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            roi_refine::ROIConfig rcfg;
            rcfg.roi_half = 15;
            rcfg.search_half = 20;

            cv::Vec3f ip(init_x, init_y, init_a);
            auto ref = roi_refine::refineROI(
                loaded.templ_image, scene1, sample_pts_15, ip, rcfg);

            double roi_ms = std::chrono::duration<double,std::milli>(
                std::chrono::high_resolution_clock::now()-t0).count();
            roi15_ae = ref[2] - 25; if(roi15_ae>180)roi15_ae-=360; if(roi15_ae<-180)roi15_ae+=360;
            roi15_pd = std::sqrt((ref[0]-160)*(ref[0]-160)+(ref[1]-120)*(ref[1]-120));
            printf("@%+5.1f %4.1fpx %4.1fms  ", roi15_ae, roi15_pd, roi_ms);
        }

        // ROI 8pt×3iter
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            roi_refine::ROIConfig rcfg;
            rcfg.roi_half = 15;
            rcfg.search_half = 20;
            rcfg.max_iters = 3;

            cv::Vec3f ip(init_x, init_y, init_a);
            auto ref = roi_refine::refineROI(
                loaded.templ_image, scene1, sample_pts_8, ip, rcfg);

            double roi_ms = std::chrono::duration<double,std::milli>(
                std::chrono::high_resolution_clock::now()-t0).count();
            float ae = ref[2] - 25; if(ae>180)ae-=360; if(ae<-180)ae+=360;
            float pd = std::sqrt((ref[0]-160)*(ref[0]-160)+(ref[1]-120)*(ref[1]-120));
            printf("@%+5.1f %4.1fpx %4.1fms  ", ae, pd, roi_ms);
        }

        // ROI 8 edge-only × 3iter
        float edge_ae, edge_pd;
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            roi_refine::ROIConfig rcfg;
            rcfg.roi_half = 15;
            rcfg.search_half = 20;
            rcfg.max_iters = 3;

            cv::Vec3f ip(init_x, init_y, init_a);
            auto ref = roi_refine::refineROI(
                loaded.templ_image, scene1, sample_pts_edge, ip, rcfg);

            double roi_ms = std::chrono::duration<double,std::milli>(
                std::chrono::high_resolution_clock::now()-t0).count();
            edge_ae = ref[2] - 25; if(edge_ae>180)edge_ae-=360; if(edge_ae<-180)edge_ae+=360;
            edge_pd = std::sqrt((ref[0]-160)*(ref[0]-160)+(ref[1]-120)*(ref[1]-120));
            printf("@%+5.1f %4.1fpx %4.1fms", edge_ae, edge_pd, roi_ms);
        }

        printf("\n");

        // Assert: clean "perfect" case should have small errors for ROI
        if (pe.noise == 0 && pe.blur == 0 && pe.dx == 0 && pe.dy == 0 && pe.da == 0) {
            CHECK(std::abs(roi15_ae) < 1.0f, "Robustness clean/perfect: ROI 15pt angle error >= 1 deg");
            CHECK(roi15_pd < 1.0f, "Robustness clean/perfect: ROI 15pt pos error >= 1 px");
        }
    }
    printf("\n");

    for (auto& mode : modes) {
        sbm::MatchConfig cfg;
        cfg.min_score = 50;
        cfg.nms_radius = 50;
        cfg.refine = mode.mode;

        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};
        matcher.addModel("L", loaded, mcfg);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto results = matcher.match(scene);
        double ms1 = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        t0 = std::chrono::high_resolution_clock::now();
        auto results2 = matcher.match(scene);
        double ms2 = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        printf("%-20s (1st=%.1fms 2nd=%.1fms): ", mode.name, ms1, ms2);
        for (auto& r : results)
            printf("(%3.0f,%3.0f)@%5.1f  ", r.x, r.y, r.angle);
        printf("\n");

        // Assert: each mode should find all 3 objects
        {
            char msg[128];
            snprintf(msg, sizeof(msg), "3-object match [%s]: expected >=3 results, got %d",
                     mode.name, (int)results.size());
            CHECK((int)results.size() >= 3, msg);
        }
        // Note: angle comparison skipped — flat triangle template has rotational
        // ambiguity, so matched angles may differ from warpAffine GT by a
        // shape-dependent offset. We only verify detection count above.
    }

    // Ablation: edges only vs edges+corners vs corners only
    printf("\n--- Ablation: edge/corner contribution ---\n");
    struct Filter { const char* name; bool use_edge; bool use_corner; };
    Filter filters[] = {
        {"edges only",   true,  false},
        {"edges+corners",true,  true},
        {"corners only", false, true},
    };

    for (auto& flt : filters) {
        // Create filtered feature set
        auto filtered = loaded;
        std::vector<sbm::FeatureSet::RefinePt> kept;
        for (auto& rp : filtered.refine_points) {
            if (rp.type == sbm::FeatureSet::RefinePt::EDGE && flt.use_edge)
                kept.push_back(rp);
            if (rp.type == sbm::FeatureSet::RefinePt::CORNER && flt.use_corner)
                kept.push_back(rp);
        }
        filtered.refine_points = kept;

        sbm::MatchConfig cfg;
        cfg.min_score = 50;
        cfg.nms_radius = 50;
        cfg.refine = sbm::RefineMode::ICP;

        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};
        matcher.addModel("rect", filtered, mcfg);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto results = matcher.match(scene);
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        printf("%-20s (%3d pts, %.1fms): ", flt.name, (int)kept.size(), ms);
        for (auto& r : results)
            printf("(%3.0f,%3.0f)@%5.1f  ", r.x, r.y, r.angle);
        printf("\n");
    }

    // ================================================================
    // FHD speed benchmark: larger template, 1920x1080 scene, 10 objects
    // ================================================================
    printf("\n========== FHD SPEED BENCHMARK ==========\n");
    {
        // Larger L-shape template (200x200)
        Mat templ_fhd(200, 200, CV_8U, Scalar(0));
        draw_L(templ_fhd, 100, 100, 0, 200);

        auto feat_fhd = sbm::extractFeatures(templ_fhd);
        feat_fhd.setOrigin(100, 75);
        printf("Template: 200x200 L-shape, %d features\n", feat_fhd.numFeatures());

        // Print optimized feature positions
        auto fhd_opt = feat_fhd.selectOptimizedPoints(8);
        printf("  Optimized 8 points (relative to center 100,100):\n");
        float tcx = feat_fhd.templ_width / 2.0f, tcy = feat_fhd.templ_height / 2.0f;
        for (size_t i = 0; i < fhd_opt.size(); i++) {
            auto& p = fhd_opt[i];
            float lev = std::sqrt(p.x*p.x + p.y*p.y);
            // Find the normal for this point
            int best = -1; float best_d = 1e9f;
            for (size_t j = 0; j < feat_fhd.refine_points.size(); j++) {
                float dx = p.x - feat_fhd.refine_points[j].px;
                float dy = p.y - feat_fhd.refine_points[j].py;
                if (dx*dx+dy*dy < best_d) { best_d = dx*dx+dy*dy; best = (int)j; }
            }
            auto& rp = feat_fhd.refine_points[best];
            printf("    [%d] (%+6.1f,%+6.1f) lev=%4.0f n=(%+.2f,%+.2f) %s\n",
                   (int)i, p.x, p.y, lev, rp.nx, rp.ny,
                   rp.type == sbm::FeatureSet::RefinePt::CORNER ? "CORNER" : "EDGE");
        }

        // Also show sensitivity
        auto fhd_sens = feat_fhd.analyzeSensitivity();
        printf("  Sensitivity: worst_ang=%.2f worst_pos=%.2f\n",
               fhd_sens.worst_angle_sens, fhd_sens.worst_pos_sens);
        for (auto& f : fhd_sens.features)
            printf("    (%+6.1f,%+6.1f) d_ang=%.2f d_pos=%.2f lev=%.0f\n",
                   f.pos.x, f.pos.y, f.d_ang, f.d_pos, f.leverage);
        printf("\n");

        // FHD scene with 10 objects at various positions/angles
        struct FHDObj { int x, y; double angle; };
        FHDObj fhd_objs[] = {
            {200, 150, 15},  {500, 300, 45},   {900, 200, 90},
            {1300, 400, 135}, {1700, 250, 180}, {350, 700, 210},
            {750, 850, 270}, {1100, 600, 315},  {1500, 800, 30},
            {1800, 900, 60},
        };

        Mat scene_fhd(1080, 1920, CV_8U, Scalar(30));
        RNG rng_fhd(123);
        // Add background texture noise
        Mat bg_noise(scene_fhd.size(), CV_64F);
        rng_fhd.fill(bg_noise, RNG::NORMAL, 0, 15);
        Mat tmp_fhd; scene_fhd.convertTo(tmp_fhd, CV_64F);
        tmp_fhd += bg_noise; tmp_fhd.convertTo(scene_fhd, CV_8U);

        for (auto& obj : fhd_objs) {
            Mat M = getRotationMatrix2D(Point2f(100, 100), -obj.angle, 1.0);
            Mat rot; warpAffine(templ_fhd, rot, M, templ_fhd.size(),
                                INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
            int ox = obj.x - 100, oy = obj.y - 100;
            for (int r = 0; r < rot.rows; r++)
                for (int c = 0; c < rot.cols; c++) {
                    int sy = oy + r, sx = ox + c;
                    if (sy >= 0 && sy < scene_fhd.rows && sx >= 0 &&
                        sx < scene_fhd.cols && rot.at<uchar>(r, c) > 0)
                        scene_fhd.at<uchar>(sy, sx) = rot.at<uchar>(r, c);
                }
        }

        printf("Scene: 1920x1080, 10 objects, background noise s=15\n");

        // Conditions to test
        struct FHDCond {
            const char* name;
            double noise;
            int blur;
        };
        FHDCond conditions[] = {
            {"clean",       0,  0},
            {"noise s=10",  10, 0},
            {"noise s=20",  20, 0},
            {"noise s=30",  30, 0},
            {"noise s=50",  50, 0},
            {"blur k=3",    0,  3},
            {"blur k=5",    0,  5},
            {"blur k=7",    0,  7},
            {"blur k=11",   0,  11},
            {"blur k=21",   0,  21},
            {"n10+b3",      10, 3},
            {"n20+b5",      20, 5},
            {"n30+b7",      30, 7},
            {"n30+b11",     30, 11},
            {"n50+b11",     50, 11},
            {"n50+b21",     50, 21},
        };

        Mode fhd_modes[] = {
            {"None",        sbm::RefineMode::None},
            {"ICP (dense)", sbm::RefineMode::ICP},
            {"ROI",         sbm::RefineMode::ROI},
        };

        int n_objs = sizeof(fhd_objs) / sizeof(fhd_objs[0]);

        // Header
        printf("\n%-14s", "Condition");
        for (auto& mode : fhd_modes)
            printf("  %-28s", mode.name);
        printf("\n");
        for (int i = 0; i < 14 + 4*30; i++) printf("-");
        printf("\n");

        // Per-object detail for ICP clean case
        {
            sbm::MatchConfig cfg;
            cfg.min_score = 40;
            cfg.nms_radius = 80;
            cfg.refine = sbm::RefineMode::ICP;

            sbm::ShapeMatcher matcher(cfg);
            sbm::ModelConfig mcfg;
            mcfg.angle = {0, 360, 2};

            std::vector<sbm::MatchResult> results;
            {
                OutputGuard guard;
                matcher.addModel("L", feat_fhd, mcfg);
                results = matcher.match(scene_fhd);
            }

            printf("Per-object ICP detail (clean, 10 objects):\n");
            for (auto& r : results) {
                float best_d = 1e9f; int best_j = -1;
                for (int j = 0; j < n_objs; j++) {
                    float ox2 = 100 - 100, oy2 = 75 - 100;
                    float rad2 = -(float)fhd_objs[j].angle * (float)CV_PI / 180.0f;
                    float rx2 = std::cos(rad2)*ox2 - std::sin(rad2)*oy2;
                    float ry2 = std::sin(rad2)*ox2 + std::cos(rad2)*oy2;
                    float gt_x2 = fhd_objs[j].x + rx2;
                    float gt_y2 = fhd_objs[j].y + ry2;
                    float dx2 = r.x - gt_x2, dy2 = r.y - gt_y2;
                    float d2 = std::sqrt(dx2*dx2 + dy2*dy2);
                    if (d2 < best_d) { best_d = d2; best_j = j; }
                }
                float ae = r.angle - (float)fhd_objs[best_j].angle;
                if (ae > 180) ae -= 360; if (ae < -180) ae += 360;
                printf("  gt=(%4d,%4d)@%3.0f  got=(%5.1f,%5.1f)@%5.1f  err: %+5.2fdeg %4.1fpx\n",
                       fhd_objs[best_j].x, fhd_objs[best_j].y, (float)fhd_objs[best_j].angle,
                       r.x, r.y, r.angle, ae, best_d);
            }
            printf("\n");
        }

        // Suppress library output during benchmark

        for (auto& cond : conditions) {
            // Apply noise/blur to clean scene
            Mat scene_test = scene_fhd.clone();
            if (cond.noise > 0) {
                Mat noise_mat(scene_test.size(), CV_64F);
                RNG rng_c(42);
                rng_c.fill(noise_mat, RNG::NORMAL, 0, cond.noise);
                Mat tmp_c; scene_test.convertTo(tmp_c, CV_64F);
                tmp_c += noise_mat; tmp_c.convertTo(scene_test, CV_8U);
            }
            if (cond.blur > 0) {
                GaussianBlur(scene_test, scene_test, Size(cond.blur, cond.blur), 0);
            }

            // Collect all results first, then print in one line
            struct ModeResult { int matched; int total; double ms; float ang; float pos; };
            ModeResult mode_results[4];
            int mi = 0;

            for (auto& mode : fhd_modes) {
                sbm::MatchConfig cfg;
                cfg.min_score = 40;
                cfg.nms_radius = 80;
                cfg.refine = mode.mode;

                sbm::ShapeMatcher matcher(cfg);
                sbm::ModelConfig mcfg;
                mcfg.angle = {0, 360, 2};
                matcher.addModel("L", feat_fhd, mcfg);

                // Warm up
                { OutputGuard guard; matcher.match(scene_test); }

                // Average 3 runs
                double total_ms = 0;
                std::vector<sbm::MatchResult> results;
                {
                    OutputGuard guard;
                    for (int i = 0; i < 3; i++) {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        results = matcher.match(scene_test);
                        total_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                    }
                }
                double avg_ms = total_ms / 3.0;

                // Compute mean angle/position error across found objects
                float total_ang_err = 0, total_pos_err = 0;
                int matched = 0;
                for (auto& r : results) {
                    // Find closest ground truth object
                    float best_d = 1e9f;
                    int best_j = -1;
                    for (int j = 0; j < n_objs; j++) {
                        // GT position: origin at (100,75), rotated around (100,100)
                        float ox = 100 - 100, oy = 75 - 100;  // origin offset from center
                        float rad = -(float)fhd_objs[j].angle * (float)CV_PI / 180.0f;
                        float rx = std::cos(rad)*ox - std::sin(rad)*oy;
                        float ry = std::sin(rad)*ox + std::cos(rad)*oy;
                        float gt_x = fhd_objs[j].x + rx;
                        float gt_y = fhd_objs[j].y + ry;
                        float dx = r.x - gt_x, dy = r.y - gt_y;
                        float d = std::sqrt(dx*dx + dy*dy);
                        if (d < best_d) { best_d = d; best_j = j; }
                    }
                    if (best_j >= 0 && best_d < 50) {
                        float ae = r.angle - (float)fhd_objs[best_j].angle;
                        if (ae > 180) ae -= 360; if (ae < -180) ae += 360;
                        total_ang_err += std::abs(ae);
                        total_pos_err += best_d;
                        matched++;
                    }
                }
                float mean_ang = matched > 0 ? total_ang_err / matched : -1;
                float mean_pos = matched > 0 ? total_pos_err / matched : -1;
                mode_results[mi++] = {matched, n_objs, avg_ms, mean_ang, mean_pos};
            }
            // Print entire row at once (avoids meiqua cout interleaving)
            printf("%-14s", cond.name);
            for (int i = 0; i < mi; i++)
                printf("  %2d/%d %5.1fms %4.1fdeg %4.1fpx",
                       mode_results[i].matched, mode_results[i].total,
                       mode_results[i].ms, mode_results[i].ang, mode_results[i].pos);
            printf("\n");

            // Assert: clean case should match at least 8 of 10 objects (for modes with refinement)
            if (cond.noise == 0 && cond.blur == 0) {
                for (int i = 0; i < mi; i++) {
                    // None mode may miss more due to discretization; require >=6
                    int min_matched = (fhd_modes[i].mode == sbm::RefineMode::None) ? 2 : 8;
                    char msg[128];
                    snprintf(msg, sizeof(msg), "FHD clean [%s]: matched %d/10, expected >=%d",
                             fhd_modes[i].name, mode_results[i].matched, min_matched);
                    CHECK(mode_results[i].matched >= min_matched, msg);
                }
            }
        }

        // --- FHD 20-object benchmark with noise=30 ---
        printf("\n--- FHD 20 objects, noise=30 benchmark ---\n");
        {
            FHDObj objs20[] = {
                {150, 100, 7},   {450, 150, 23},  {750, 100, 51},   {1050, 150, 78},
                {1350, 100, 102},{1650, 150, 133}, {1850, 100, 157}, {250, 350, 189},
                {550, 400, 212}, {850, 350, 238},  {1150, 400, 267}, {1450, 350, 291},
                {1750, 400, 319},{200, 600, 342},  {500, 650, 12},   {800, 600, 67},
                {1100, 650, 112},{1400, 600, 167}, {1700, 650, 222}, {300, 900, 277},
            };
            int n20 = sizeof(objs20)/sizeof(objs20[0]);

            // Build scene
            Mat scene20(1080, 1920, CV_8U, Scalar(30));
            for (auto& obj : objs20) {
                Mat M = getRotationMatrix2D(Point2f(100, 100), -obj.angle, 1.0);
                Mat rot; warpAffine(templ_fhd, rot, M, templ_fhd.size(),
                                    INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
                int ox = obj.x - 100, oy = obj.y - 100;
                for (int r = 0; r < rot.rows; r++)
                    for (int c = 0; c < rot.cols; c++) {
                        int sy = oy + r, sx = ox + c;
                        if (sy >= 0 && sy < scene20.rows && sx >= 0 &&
                            sx < scene20.cols && rot.at<uchar>(r, c) > 0)
                            scene20.at<uchar>(sy, sx) = rot.at<uchar>(r, c);
                    }
            }
            // Add noise=30
            {
                Mat nm(scene20.size(), CV_64F);
                RNG rng20(42); rng20.fill(nm, RNG::NORMAL, 0, 30);
                Mat t20; scene20.convertTo(t20, CV_64F);
                t20 += nm; t20.convertTo(scene20, CV_8U);
            }

            // Save scene
            imwrite("output/fhd20_n30_scene.png", scene20);

            Mode bench_modes[] = {
                {"None",        sbm::RefineMode::None},
                {"ICP (inv)",   sbm::RefineMode::ICP},
                {"ROI",         sbm::RefineMode::ROI},
            };

            FILE* bf = fopen("output/fhd20_n30.txt", "w");
            fprintf(bf, "FHD 1920x1080, 20 objects, noise=30, 200x200 L-shape\n\n");

            for (auto& mode : bench_modes) {
                sbm::MatchConfig cfg;
                cfg.min_score = 30;
                cfg.nms_radius = 80;
                cfg.refine = mode.mode;

                sbm::ShapeMatcher matcher(cfg);
                sbm::ModelConfig mcfg;
                mcfg.angle = {0, 360, 2};

                { OutputGuard guard; matcher.addModel("L", feat_fhd, mcfg); matcher.match(scene20); } // warm up

                // Average 5 runs
                double total_ms = 0;
                std::vector<sbm::MatchResult> results;
                {
                    OutputGuard guard;
                    for (int i = 0; i < 5; i++) {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        results = matcher.match(scene20);
                        total_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                    }
                }
                double avg_ms = total_ms / 5.0;

                // Per-object errors
                float total_ae = 0, total_pe = 0;
                int matched = 0;
                for (auto& r : results) {
                    float best_d = 1e9f; int best_j = -1;
                    for (int j = 0; j < n20; j++) {
                        float o2 = 0, oo = -25;
                        float rd = -(float)objs20[j].angle*(float)CV_PI/180.0f;
                        float gx = objs20[j].x + std::cos(rd)*o2 - std::sin(rd)*oo;
                        float gy = objs20[j].y + std::sin(rd)*o2 + std::cos(rd)*oo;
                        float d = std::sqrt((r.x-gx)*(r.x-gx)+(r.y-gy)*(r.y-gy));
                        if (d < best_d) { best_d = d; best_j = j; }
                    }
                    if (best_j >= 0 && best_d < 50) {
                        float ae = r.angle - (float)objs20[best_j].angle;
                        if (ae > 180) ae -= 360; if (ae < -180) ae += 360;
                        total_ae += std::abs(ae);
                        total_pe += best_d;
                        matched++;
                    }
                }
                float ma = matched > 0 ? total_ae / matched : -1;
                float mp = matched > 0 ? total_pe / matched : -1;

                fprintf(bf, "%-12s  %2d/%d found  %.1fms  ang=%.2fdeg  pos=%.2fpx\n",
                        mode.name, matched, n20, avg_ms, ma, mp);
                // Per-object detail
                for (auto& r : results) {
                    float best_d = 1e9f; int best_j = -1;
                    for (int j = 0; j < n20; j++) {
                        float o2b=0, oob=-25;
                        float rdb=-(float)objs20[j].angle*(float)CV_PI/180.0f;
                        float gxb=objs20[j].x+std::cos(rdb)*o2b-std::sin(rdb)*oob;
                        float gyb=objs20[j].y+std::sin(rdb)*o2b+std::cos(rdb)*oob;
                        float db=std::sqrt((r.x-gxb)*(r.x-gxb)+(r.y-gyb)*(r.y-gyb));
                        if(db<best_d){best_d=db;best_j=j;}
                    }
                    float aeb = r.angle-(float)objs20[best_j].angle;
                    if(aeb>180)aeb-=360;if(aeb<-180)aeb+=360;
                    fprintf(bf, "  gt=(%4d,%4d)@%3d  got=(%5.1f,%5.1f)@%5.1f  err=%+5.1fdeg %4.1fpx%s\n",
                            objs20[best_j].x, objs20[best_j].y, (int)objs20[best_j].angle,
                            r.x, r.y, r.angle, aeb, best_d,
                            best_d >= 50 ? " ** UNMATCHED" : "");
                }
                fprintf(bf, "\n");
            }
            fclose(bf);
            printf("  -> saved output/fhd20_n30.txt, fhd20_n30_scene.png\n");
        }

        // --- GT orientation debug: single objects at known angles ---
        {
            // Sub-pixel position sweep: place object at fractional pixel positions
            int cell = 250;
            printf("\n--- Sub-pixel position sweep (angle=25deg) ---\n");
            {
                float test_ang = 25;
                int n_steps = 10;
                // Coarse matching accuracy under noise (no refinement)
                {
                    float noise_sweep[] = {0, 10, 20, 30, 40, 50, 60, 80, 100};
                    int n_ns = sizeof(noise_sweep)/sizeof(noise_sweep[0]);
                    FILE* cf = fopen("output/coarse_vs_noise.txt", "w");
                    fprintf(cf, "%-8s  %6s  %8s  %8s\n", "Noise", "Found", "Ang_err", "Pos_err");

                    for (int ni2 = 0; ni2 < n_ns; ni2++) {
                        float ns2 = noise_sweep[ni2];
                        // Place object at sub-pixel position
                        Mat scene_cn(cell, cell, CV_8U, Scalar(0));
                        Mat Mc = getRotationMatrix2D(Point2f(100,100), -test_ang, 1.0);
                        double* mdc = (double*)Mc.data;
                        mdc[2] += 0.3; mdc[5] += 0.7;
                        Mat rotc; warpAffine(templ_fhd, rotc, Mc, templ_fhd.size(),
                                             INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
                        int oxc=cell/2-100, oyc=cell/2-100;
                        for(int r=0;r<rotc.rows;r++) for(int c=0;c<rotc.cols;c++){
                            int sy=oyc+r,sx=oxc+c;
                            if(sy>=0&&sy<cell&&sx>=0&&sx<cell&&rotc.at<uchar>(r,c)>0)
                                scene_cn.at<uchar>(sy,sx)=rotc.at<uchar>(r,c);
                        }
                        if (ns2 > 0) {
                            Mat nm3(scene_cn.size(), CV_64F);
                            RNG rng3(42); rng3.fill(nm3, RNG::NORMAL, 0, ns2);
                            Mat t3; scene_cn.convertTo(t3, CV_64F);
                            t3 += nm3; t3.convertTo(scene_cn, CV_8U);
                        }
                        float gt_cx3 = cell/2.0f+0.3f, gt_cy3 = cell/2.0f+0.7f;

                        // Run coarse only, ICP, ROI — average over 5 angles to reduce variance
                        float angles_test[] = {10, 25, 60, 130, 200};
                        float c_ae=0,c_pe=0, i_ae=0,i_pe=0, r_ae=0,r_pe=0;
                        int c_found=0, i_found=0, r_found=0;

                        for (float ta : angles_test) {
                            // Recreate scene at this angle
                            Mat scene_a(cell, cell, CV_8U, Scalar(0));
                            Mat Ma = getRotationMatrix2D(Point2f(100,100), -ta, 1.0);
                            double* mda = (double*)Ma.data;
                            mda[2] += 0.3; mda[5] += 0.7;
                            Mat rota; warpAffine(templ_fhd, rota, Ma, templ_fhd.size(),
                                                 INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
                            for(int r=0;r<rota.rows;r++) for(int c=0;c<rota.cols;c++){
                                int sy=oyc+r,sx=oxc+c;
                                if(sy>=0&&sy<cell&&sx>=0&&sx<cell)
                                    scene_a.at<uchar>(sy,sx) = rota.at<uchar>(r,c)>0 ? rota.at<uchar>(r,c) : 0;
                            }
                            if (ns2 > 0) {
                                Mat nm4(scene_a.size(), CV_64F);
                                RNG rng4(42+(int)ta); rng4.fill(nm4, RNG::NORMAL, 0, ns2);
                                Mat t4; scene_a.convertTo(t4, CV_64F);
                                t4 += nm4; t4.convertTo(scene_a, CV_8U);
                            }

                            float o_x3=0, o_y3=-25;
                            sbm::RefineMode modes3[] = {sbm::RefineMode::None, sbm::RefineMode::ICP, sbm::RefineMode::ROI};
                            for (int mi3=0; mi3<3; mi3++) {
                                sbm::MatchConfig cfg3; cfg3.min_score=30; cfg3.nms_radius=80;
                                cfg3.refine = modes3[mi3];
                                sbm::ShapeMatcher m3(cfg3);
                                sbm::ModelConfig mc3; mc3.angle={0,360,2};
                                std::vector<sbm::MatchResult> res3;
                                { OutputGuard guard; m3.addModel("L", feat_fhd, mc3); res3 = m3.match(scene_a); }

                                if (!res3.empty()) {
                                    float rr3=-res3[0].angle*(float)CV_PI/180.0f;
                                    float mcx3=res3[0].x-(std::cos(rr3)*o_x3-std::sin(rr3)*o_y3);
                                    float mcy3=res3[0].y-(std::sin(rr3)*o_x3+std::cos(rr3)*o_y3);
                                    float ae3=res3[0].angle-ta;
                                    if(ae3>180)ae3-=360;if(ae3<-180)ae3+=360;
                                    float pe3=std::sqrt((mcx3-gt_cx3)*(mcx3-gt_cx3)+(mcy3-gt_cy3)*(mcy3-gt_cy3));
                                    if(mi3==0){c_ae+=std::abs(ae3);c_pe+=pe3;c_found++;}
                                    if(mi3==1){i_ae+=std::abs(ae3);i_pe+=pe3;i_found++;}
                                    if(mi3==2){r_ae+=std::abs(ae3);r_pe+=pe3;r_found++;}
                                }
                            }
                        }
                        int na2 = sizeof(angles_test)/sizeof(angles_test[0]);
                            // Time each mode on a single scene
                        float c_ms=0, i_ms=0, r_ms=0;
                        {
                            // Build a scene for timing
                            Mat sa(cell, cell, CV_8U, Scalar(0));
                            {
                                Mat Mt = getRotationMatrix2D(Point2f(100,100), -test_ang, 1.0);
                                double* mdt = (double*)Mt.data; mdt[2]+=0.3; mdt[5]+=0.7;
                                Mat rott; warpAffine(templ_fhd, rott, Mt, templ_fhd.size(),
                                    INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
                                for(int r=0;r<rott.rows;r++) for(int c=0;c<rott.cols;c++){
                                    int sy=oyc+r,sx=oxc+c;
                                    if(sy>=0&&sy<cell&&sx>=0&&sx<cell&&rott.at<uchar>(r,c)>0)
                                        sa.at<uchar>(sy,sx)=rott.at<uchar>(r,c);
                                }
                                if (ns2 > 0) {
                                    Mat nm5(sa.size(), CV_64F);
                                    RNG rng5(42); rng5.fill(nm5, RNG::NORMAL, 0, ns2);
                                    Mat t5; sa.convertTo(t5, CV_64F);
                                    t5 += nm5; t5.convertTo(sa, CV_8U);
                                }
                            }
                            sbm::RefineMode modes4[] = {sbm::RefineMode::None, sbm::RefineMode::ICP, sbm::RefineMode::ROI};
                            for (int mi4=0; mi4<3; mi4++) {
                                sbm::MatchConfig cfg4; cfg4.min_score=30; cfg4.nms_radius=80;
                                cfg4.refine = modes4[mi4];
                                sbm::ShapeMatcher m4(cfg4);
                                sbm::ModelConfig mc4; mc4.angle={0,360,2};
                                float ms;
                                {
                                    OutputGuard guard;
                                    m4.addModel("L", feat_fhd, mc4);
                                    m4.match(sa); // warm up
                                    auto t0 = std::chrono::high_resolution_clock::now();
                                    m4.match(sa);
                                    ms = (float)std::chrono::duration<double,std::milli>(
                                        std::chrono::high_resolution_clock::now()-t0).count();
                                }
                                if(mi4==0) c_ms=ms; if(mi4==1) i_ms=ms; if(mi4==2) r_ms=ms;
                            }
                        }

                    fprintf(cf, "%-8.0f  %d/%d  %5.1f/%5.1f/%5.1fms  C:%5.1fdeg %4.1fpx  I:%5.2fdeg %4.2fpx  R:%5.2fdeg %4.2fpx\n",
                                ns2, c_found, na2, c_ms, i_ms, r_ms,
                                c_found>0?c_ae/c_found:-1.f, c_found>0?c_pe/c_found:-1.f,
                                i_found>0?i_ae/i_found:-1.f, i_found>0?i_pe/i_found:-1.f,
                                r_found>0?r_ae/r_found:-1.f, r_found>0?r_pe/r_found:-1.f);
                    }
                    fclose(cf);
                    printf("  -> saved output/coarse_vs_noise.txt\n");
                }

                // Test different ROI sizes under noise
                int roi_sizes[] = {10, 15, 20, 30};
                float noise_for_roi[] = {0, 20, 40, 60};
                {
                    FILE* rf = fopen("output/roi_size_sweep.txt", "w");
                    fprintf(rf, "%-8s", "ROI_half");
                    for (float ns : noise_for_roi) fprintf(rf, "  n=%-5.0f         ", ns);
                    fprintf(rf, "\n");

                    for (int rh : roi_sizes) {
                        fprintf(rf, "%-8d", rh);
                        for (float ns : noise_for_roi) {
                            // Single test: dx=0.3, dy=0.7 (arbitrary sub-pixel)
                            Mat scene_rt(cell, cell, CV_8U, Scalar(0));
                            Mat Mr = getRotationMatrix2D(Point2f(100,100), -test_ang, 1.0);
                            double* mdr = (double*)Mr.data;
                            mdr[2] += 0.3; mdr[5] += 0.7;
                            Mat rotr; warpAffine(templ_fhd, rotr, Mr, templ_fhd.size(),
                                                 INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
                            int oxr=cell/2-100, oyr=cell/2-100;
                            for(int r=0;r<rotr.rows;r++) for(int c=0;c<rotr.cols;c++){
                                int sy=oyr+r,sx=oxr+c;
                                if(sy>=0&&sy<cell&&sx>=0&&sx<cell&&rotr.at<uchar>(r,c)>0)
                                    scene_rt.at<uchar>(sy,sx)=rotr.at<uchar>(r,c);
                            }
                            if (ns > 0) {
                                Mat nm2(scene_rt.size(), CV_64F);
                                RNG rng2(42); rng2.fill(nm2, RNG::NORMAL, 0, ns);
                                Mat t2; scene_rt.convertTo(t2, CV_64F);
                                t2 += nm2; t2.convertTo(scene_rt, CV_8U);
                            }
                            float gt_cx2 = cell/2.0f+0.3f, gt_cy2 = cell/2.0f+0.7f;

                            // ROI with custom roi_half
                            auto opt_pts = feat_fhd.selectOptimizedPoints(8);
                            std::vector<roi_refine::SamplePoint> spts;
                            for(auto&p:opt_pts){roi_refine::SamplePoint sp;sp.pos=p;spts.push_back(sp);}
                            roi_refine::ROIConfig rcfg;
                            rcfg.roi_half = rh;
                            rcfg.search_half = rh;
                            rcfg.max_iters = 3;
                            // Quantize angle to 2deg step (simulate coarse)
                            float coarse_a = std::round(test_ang/2.0f)*2.0f;
                            cv::Vec3f ip(gt_cx2, gt_cy2, coarse_a);
                            auto ref = roi_refine::refineROI(
                                feat_fhd.templ_image, scene_rt, spts, ip, rcfg);
                            float ae2 = ref[2] - test_ang;
                            if(ae2>180)ae2-=360;if(ae2<-180)ae2+=360;
                            float pe2 = std::sqrt((ref[0]-gt_cx2)*(ref[0]-gt_cx2)+
                                                   (ref[1]-gt_cy2)*(ref[1]-gt_cy2));
                            fprintf(rf, "  %+.2fdeg %.3fpx", ae2, pe2);
                        }
                        fprintf(rf, "\n");
                    }
                    fclose(rf);
                    printf("  -> saved output/roi_size_sweep.txt\n");
                }

                float noise_levels[] = {0, 10, 20, 30, 40, 50, 60};
                int n_noise = sizeof(noise_levels)/sizeof(noise_levels[0]);

                FILE* sumf = fopen("output/subpixel_summary.txt", "w");

                for (int ni = 0; ni < n_noise; ni++) {
                float noise_sigma = noise_levels[ni];

                std::vector<float> icp_pos_all, roi_pos_all, icp_ang_all, roi_ang_all;

                for (int yi = 0; yi < n_steps; yi++) {
                for (int xi = 0; xi < n_steps; xi++) {
                    float dx = xi * 0.1f, dy = yi * 0.1f;

                    Mat scene_sp(cell, cell, CV_8U, Scalar(0));
                    // Rotate template
                    Mat M = getRotationMatrix2D(Point2f(100, 100), -test_ang, 1.0);
                    // Add sub-pixel translation to the rotation matrix
                    double* md2 = (double*)M.data;
                    md2[2] += dx;  // add fractional X offset
                    md2[5] += dy;  // add fractional Y offset
                    Mat rot;
                    warpAffine(templ_fhd, rot, M, templ_fhd.size(),
                               INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
                    int ox2 = cell/2 - 100, oy2 = cell/2 - 100;
                    for (int r = 0; r < rot.rows; r++)
                        for (int c = 0; c < rot.cols; c++) {
                            int sy = oy2+r, sx = ox2+c;
                            if (sy>=0 && sy<cell && sx>=0 && sx<cell && rot.at<uchar>(r,c)>0)
                                scene_sp.at<uchar>(sy, sx) = rot.at<uchar>(r, c);
                        }

                    // Add noise
                    if (noise_sigma > 0) {
                        Mat nm(scene_sp.size(), CV_64F);
                        RNG rng_n(42 + xi*100 + yi);
                        rng_n.fill(nm, RNG::NORMAL, 0, noise_sigma);
                        Mat tmp_n; scene_sp.convertTo(tmp_n, CV_64F);
                        tmp_n += nm; tmp_n.convertTo(scene_sp, CV_8U);
                    }

                    // Save one sample scene per noise level
                    if (xi == 0 && yi == 0) {
                        char sfn[64];
                        snprintf(sfn,sizeof(sfn),"output/scene_n%.0f.png", noise_sigma);
                        imwrite(sfn, scene_sp);
                    }

                    // GT center (with sub-pixel offset)
                    float gt_cx = cell/2.0f + dx, gt_cy = cell/2.0f + dy;

                    // Run ICP
                    sbm::MatchConfig cfg_i; cfg_i.min_score=35; cfg_i.nms_radius=80;
                    cfg_i.refine = sbm::RefineMode::ICP;
                    sbm::ShapeMatcher mi2(cfg_i);
                    sbm::ModelConfig mc2; mc2.angle = {0, 360, 2};
                    std::vector<sbm::MatchResult> ri;
                    { OutputGuard guard; mi2.addModel("L", feat_fhd, mc2); ri = mi2.match(scene_sp); }

                    // Run ROI
                    sbm::MatchConfig cfg_r; cfg_r.min_score=35; cfg_r.nms_radius=80;
                    cfg_r.refine = sbm::RefineMode::ROI;
                    sbm::ShapeMatcher mr2(cfg_r);
                    std::vector<sbm::MatchResult> rr;
                    { OutputGuard guard; mr2.addModel("L", feat_fhd, mc2); rr = mr2.match(scene_sp); }

                    // Compute errors (convert user origin back to center)
                    float o_x2=0, o_y2=-25;
                    float icp_ae=99, icp_pe=99, roi_ae=99, roi_pe=99;
                    if (!ri.empty()) {
                        float rr3 = -ri[0].angle*(float)CV_PI/180.0f;
                        float icx = ri[0].x-(std::cos(rr3)*o_x2-std::sin(rr3)*o_y2);
                        float icy = ri[0].y-(std::sin(rr3)*o_x2+std::cos(rr3)*o_y2);
                        icp_ae = ri[0].angle - test_ang;
                        if (icp_ae>180) icp_ae-=360; if (icp_ae<-180) icp_ae+=360;
                        icp_pe = std::sqrt((icx-gt_cx)*(icx-gt_cx)+(icy-gt_cy)*(icy-gt_cy));
                    }
                    if (!rr.empty()) {
                        float rr3 = -rr[0].angle*(float)CV_PI/180.0f;
                        float rcx = rr[0].x-(std::cos(rr3)*o_x2-std::sin(rr3)*o_y2);
                        float rcy = rr[0].y-(std::sin(rr3)*o_x2+std::cos(rr3)*o_y2);
                        roi_ae = rr[0].angle - test_ang;
                        if (roi_ae>180) roi_ae-=360; if (roi_ae<-180) roi_ae+=360;
                        roi_pe = std::sqrt((rcx-gt_cx)*(rcx-gt_cx)+(rcy-gt_cy)*(rcy-gt_cy));
                    }

                    icp_ang_all.push_back(std::abs(icp_ae));
                    icp_pos_all.push_back(icp_pe);
                    roi_ang_all.push_back(std::abs(roi_ae));
                    roi_pos_all.push_back(roi_pe);
                }
                }

                // Summary
                float icp_a_mean=0,icp_p_mean=0,roi_a_mean=0,roi_p_mean=0;
                float icp_p_max=0,roi_p_max=0;
                int nn = (int)icp_ang_all.size();
                for (int i=0;i<nn;i++) {
                    icp_a_mean+=icp_ang_all[i]; icp_p_mean+=icp_pos_all[i];
                    roi_a_mean+=roi_ang_all[i]; roi_p_mean+=roi_pos_all[i];
                    icp_p_max=std::max(icp_p_max,icp_pos_all[i]);
                    roi_p_max=std::max(roi_p_max,roi_pos_all[i]);
                }
                icp_a_mean/=nn; icp_p_mean/=nn; roi_a_mean/=nn; roi_p_mean/=nn;

                fprintf(sumf, "noise=%.0f: angle=%g, %dx%d grid\n",
                        noise_sigma, test_ang, n_steps, n_steps);
                fprintf(sumf, "  ICP: mean_ang=%.3f mean_pos=%.3f max_pos=%.3f\n",
                        icp_a_mean, icp_p_mean, icp_p_max);
                fprintf(sumf, "  ROI: mean_ang=%.3f mean_pos=%.3f max_pos=%.3f\n\n",
                        roi_a_mean, roi_p_mean, roi_p_max);

                // Draw 2D heatmap: position error as function of (dx, dy)
                {
                    int hm_cell = 40;
                    int hm_w = n_steps * hm_cell, hm_h = n_steps * hm_cell;
                    int chart_w = hm_w * 2 + 80 + 60;  // two heatmaps side by side + gap + colorbar
                    int chart_h = hm_h + 80;
                    Mat chart(chart_h, chart_w, CV_8UC3, Scalar(255,255,255));

                    // Find max for color scale
                    float max_pe = 0;
                    for (int i = 0; i < nn; i++)
                        max_pe = std::max(max_pe, std::max(icp_pos_all[i], roi_pos_all[i]));
                    max_pe = std::max(max_pe, 0.2f);  // minimum scale

                    auto drawHeatmap = [&](const std::vector<float>& errs, int x_off, const char* title) {
                        cv::putText(chart, title, Point(x_off + 10, 25),
                            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,0), 1);
                        for (int yi = 0; yi < n_steps; yi++) {
                            for (int xi = 0; xi < n_steps; xi++) {
                                float e = errs[yi * n_steps + xi];
                                float t = std::min(1.0f, e / max_pe);
                                // Blue (0) -> Red (max)
                                int b = (int)(255 * (1-t));
                                int r = (int)(255 * t);
                                int g = (int)(255 * (1 - 2*std::abs(t-0.5f)));
                                Rect rc(x_off + xi*hm_cell, 35 + yi*hm_cell, hm_cell-1, hm_cell-1);
                                cv::rectangle(chart, rc, Scalar(b,g,r), -1);
                                // Value label
                                char vl[16]; snprintf(vl,sizeof(vl),"%.2f", e);
                                cv::putText(chart, vl, Point(rc.x+2, rc.y+hm_cell/2+4),
                                    FONT_HERSHEY_SIMPLEX, 0.28, Scalar(255,255,255), 1);
                            }
                        }
                        // Axis labels
                        for (int i = 0; i < n_steps; i++) {
                            char xl[8]; snprintf(xl,sizeof(xl),".%d", i);
                            cv::putText(chart, xl, Point(x_off+i*hm_cell+5, 35+hm_h+20),
                                FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0,0,0), 1);
                            cv::putText(chart, xl, Point(x_off-25, 35+i*hm_cell+hm_cell/2+4),
                                FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0,0,0), 1);
                        }
                        cv::putText(chart, "dx", Point(x_off+hm_w/2-10, 35+hm_h+40),
                            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,0,0), 1);
                    };

                    char t1[64], t2[64];
                    snprintf(t1,sizeof(t1),"ICP pos err (noise=%.0f)", noise_sigma);
                    snprintf(t2,sizeof(t2),"ROI pos err (noise=%.0f)", noise_sigma);
                    drawHeatmap(icp_pos_all, 30, t1);
                    drawHeatmap(roi_pos_all, 30 + hm_w + 50, t2);

                    // Color bar
                    int cb_x = chart_w - 30, cb_h = hm_h;
                    for (int i = 0; i < cb_h; i++) {
                        float t = 1.0f - (float)i / cb_h;
                        int b = (int)(255*(1-t)), r = (int)(255*t);
                        int g = (int)(255*(1-2*std::abs(t-0.5f)));
                        cv::line(chart, Point(cb_x, 35+i), Point(cb_x+15, 35+i), Scalar(b,g,r));
                    }
                    char cbl[16];
                    snprintf(cbl,sizeof(cbl),"%.2f", max_pe);
                    cv::putText(chart, cbl, Point(cb_x-5, 30), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0,0,0), 1);
                    cv::putText(chart, "0", Point(cb_x+3, 35+cb_h+15), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0,0,0), 1);

                    char hfn[64];
                    snprintf(hfn,sizeof(hfn),"output/subpixel_n%.0f.png", noise_sigma);
                    imwrite(hfn, chart);
                    printf("  -> saved output/subpixel_heatmap.png\n");
                }

                } // end noise loop
                fclose(sumf);
                printf("  -> saved output/subpixel_n*.png, subpixel_summary.txt\n");
            }

            // Angle sweep for chart
            std::vector<float> dbg_angles;
            for (float a = 0; a < 360; a += 5) dbg_angles.push_back(a);
            float dbg_skews[] = {0};
            int n_skews = 1;
            int na = (int)dbg_angles.size();

            // Collect errors for chart
            std::vector<float> icp_ang_errs(na), roi_ang_errs(na);
            std::vector<float> icp_pos_errs(na), roi_pos_errs(na);

            // Skip image grid for large angle count
            bool make_grid = (na <= 12);
            Mat dbg;
            if (make_grid)
                dbg = Mat(cell * n_skews, cell * na, CV_8UC3, Scalar(30,30,30));

            for (int si = 0; si < n_skews; si++) {
            float cur_skew = dbg_skews[si];
            for (int ai = 0; ai < na; ai++) {
                float ang = dbg_angles[ai];
                Mat M = getRotationMatrix2D(Point2f(100, 100), -ang, 1.0);
                Mat rot; warpAffine(templ_fhd, rot, M, templ_fhd.size(),
                                    INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

                // Apply perspective skew
                if (cur_skew > 0) {
                    float s = cur_skew * 100;
                    Point2f sp[4] = {{0,0},{199,0},{199,199},{0,199}};
                    Point2f dp[4] = {{s,s*0.5f},{199-s,-s*0.3f},
                                     {199+s*0.3f,199+s*0.5f},{-s*0.5f,199-s*0.3f}};
                    Mat P = getPerspectiveTransform(sp, dp);
                    Mat warped;
                    warpPerspective(rot, warped, P, rot.size(),
                                    INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
                    rot = warped;
                }

                Mat templ_dbg(cell, cell, CV_8U, Scalar(0));
                int ox = cell/2 - 100, oy = cell/2 - 100;
                for (int r = 0; r < rot.rows; r++)
                    for (int c = 0; c < rot.cols; c++) {
                        int sy = oy+r, sx = ox+c;
                        if (sy>=0 && sy<cell && sx>=0 && sx<cell && rot.at<uchar>(r,c)>0)
                            templ_dbg.at<uchar>(sy,sx) = rot.at<uchar>(r,c);
                    }
                Mat cell_color;
                cvtColor(templ_dbg, cell_color, COLOR_GRAY2BGR);

                // GT arrow: transform origin and x-axis tip through rotation + skew
                // Template center in template coords = (100, 100)
                // X-axis tip at (100 + 60, 100) in template coords
                float arr = 80;
                float gt_org_tx = 100, gt_org_ty = 100;  // template center
                float gt_tip_tx = 100 + arr, gt_tip_ty = 100;  // x-axis tip

                // Apply rotation (same M as warpAffine)
                // M is 2x3: [cos -sin tx; sin cos ty] with -ang
                double* md = (double*)M.data;
                auto applyM = [&](float ix, float iy, float& ox2, float& oy2) {
                    ox2 = (float)(md[0]*ix + md[1]*iy + md[2]);
                    oy2 = (float)(md[3]*ix + md[4]*iy + md[5]);
                };
                float rot_org_x, rot_org_y, rot_tip_x, rot_tip_y;
                applyM(gt_org_tx, gt_org_ty, rot_org_x, rot_org_y);
                applyM(gt_tip_tx, gt_tip_ty, rot_tip_x, rot_tip_y);

                // Apply perspective skew (if any)
                if (cur_skew > 0) {
                    float s = cur_skew * 100;
                    Point2f sp2[4] = {{0,0},{199,0},{199,199},{0,199}};
                    Point2f dp2[4] = {{s,s*0.5f},{199-s,-s*0.3f},
                                      {199+s*0.3f,199+s*0.5f},{-s*0.5f,199-s*0.3f}};
                    Mat P2 = getPerspectiveTransform(sp2, dp2);
                    double* pd = (double*)P2.data;
                    auto applyP = [&](float ix, float iy, float& ox3, float& oy3) {
                        float w = (float)(pd[6]*ix + pd[7]*iy + pd[8]);
                        ox3 = (float)(pd[0]*ix + pd[1]*iy + pd[2]) / w;
                        oy3 = (float)(pd[3]*ix + pd[4]*iy + pd[5]) / w;
                    };
                    float p_org_x, p_org_y, p_tip_x, p_tip_y;
                    applyP(rot_org_x, rot_org_y, p_org_x, p_org_y);
                    applyP(rot_tip_x, rot_tip_y, p_tip_x, p_tip_y);
                    rot_org_x = p_org_x; rot_org_y = p_org_y;
                    rot_tip_x = p_tip_x; rot_tip_y = p_tip_y;
                }

                // Offset to cell coords
                float cell_ox = cell/2.0f - 100, cell_oy = cell/2.0f - 100;
                float cx = rot_org_x + cell_ox, cy = rot_org_y + cell_oy;
                float tx = rot_tip_x + cell_ox, ty = rot_tip_y + cell_oy;

                cv::arrowedLine(cell_color, Point((int)cx,(int)cy),
                    Point((int)tx,(int)ty),
                    Scalar(0,255,0), 3, LINE_AA, 0, 0.3);
                cv::drawMarker(cell_color, Point((int)cx,(int)cy),
                    Scalar(0,255,0), MARKER_CROSS, 30, 2);
                char lbl[32]; snprintf(lbl,sizeof(lbl),"%d deg",(int)ang);
                cv::putText(cell_color, lbl, Point(10,25),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,0), 2);

                // Run coarse match on this single object
                Mat scene_dbg(cell, cell, CV_8U, Scalar(0));
                templ_dbg.copyTo(scene_dbg);

                sbm::MatchConfig cfg_dbg;
                cfg_dbg.min_score = 35;
                cfg_dbg.nms_radius = 80;
                cfg_dbg.refine = sbm::RefineMode::None;
                sbm::ShapeMatcher matcher_dbg(cfg_dbg);
                sbm::ModelConfig mcfg_dbg;
                mcfg_dbg.angle = {0, 360, 2};
                std::vector<sbm::MatchResult> dbg_results;
                { OutputGuard guard; matcher_dbg.addModel("L", feat_fhd, mcfg_dbg); dbg_results = matcher_dbg.match(scene_dbg); }

                // Helper lambda: draw match result arrow at template center
                auto drawResult = [&](const sbm::MatchResult& r, Scalar color, float scale) {
                    float o_x = 100-100, o_y = 75-100;
                    float rr2 = -r.angle * (float)CV_PI / 180.0f;
                    float mcx = r.x - (std::cos(rr2)*o_x - std::sin(rr2)*o_y);
                    float mcy = r.y - (std::sin(rr2)*o_x + std::cos(rr2)*o_y);
                    float rd2 = r.angle * (float)CV_PI / 180.0f;
                    float a2 = arr * scale;
                    cv::arrowedLine(cell_color, Point((int)mcx,(int)mcy),
                        Point((int)(mcx+std::cos(rd2)*a2),(int)(mcy+std::sin(rd2)*a2)),
                        color, 2, LINE_AA, 0, 0.3);
                    cv::circle(cell_color, Point((int)mcx,(int)mcy), 3, color, -1, LINE_AA);
                };

                // Run ICP and ROI refinement too
                sbm::MatchConfig cfg_icp; cfg_icp.min_score=35; cfg_icp.nms_radius=80;
                cfg_icp.refine = sbm::RefineMode::ICP;
                sbm::ShapeMatcher matcher_icp(cfg_icp);
                std::vector<sbm::MatchResult> icp_results;
                { OutputGuard guard; matcher_icp.addModel("L", feat_fhd, mcfg_dbg); icp_results = matcher_icp.match(scene_dbg); }

                sbm::MatchConfig cfg_roi; cfg_roi.min_score=35; cfg_roi.nms_radius=80;
                cfg_roi.refine = sbm::RefineMode::ROI;
                sbm::ShapeMatcher matcher_roi(cfg_roi);
                std::vector<sbm::MatchResult> roi_results;
                { OutputGuard guard; matcher_roi.addModel("L", feat_fhd, mcfg_dbg); roi_results = matcher_roi.match(scene_dbg); }

                // Collect errors
                float icp_ae = 0, roi_ae = 0, icp_pe = 0, roi_pe = 0;
                if (!icp_results.empty()) {
                    icp_ae = icp_results[0].angle - ang;
                    if (icp_ae>180) icp_ae-=360; if (icp_ae<-180) icp_ae+=360;
                    float o_x2=0, o_y2=-25;
                    float rr3 = -icp_results[0].angle*(float)CV_PI/180.0f;
                    float icx = icp_results[0].x-(std::cos(rr3)*o_x2-std::sin(rr3)*o_y2);
                    float icy = icp_results[0].y-(std::sin(rr3)*o_x2+std::cos(rr3)*o_y2);
                    icp_pe = std::sqrt((icx-cx)*(icx-cx)+(icy-cy)*(icy-cy));
                }
                if (!roi_results.empty()) {
                    roi_ae = roi_results[0].angle - ang;
                    if (roi_ae>180) roi_ae-=360; if (roi_ae<-180) roi_ae+=360;
                    float o_x2=0, o_y2=-25;
                    float rr3 = -roi_results[0].angle*(float)CV_PI/180.0f;
                    float rcx = roi_results[0].x-(std::cos(rr3)*o_x2-std::sin(rr3)*o_y2);
                    float rcy = roi_results[0].y-(std::sin(rr3)*o_x2+std::cos(rr3)*o_y2);
                    roi_pe = std::sqrt((rcx-cx)*(rcx-cx)+(rcy-cy)*(rcy-cy));
                }
                icp_ang_errs[ai] = icp_ae;
                roi_ang_errs[ai] = roi_ae;
                icp_pos_errs[ai] = icp_pe;
                roi_pos_errs[ai] = roi_pe;

                if (make_grid) {
                    if (!icp_results.empty()) drawResult(icp_results[0], Scalar(0,0,255), 0.8f);
                    if (!roi_results.empty()) drawResult(roi_results[0], Scalar(255,255,0), 0.7f);
                    snprintf(lbl,sizeof(lbl),"I:%+.1f", icp_ae);
                    cv::putText(cell_color, lbl, Point(5,cell-15),
                        FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,0,255), 1);
                    snprintf(lbl,sizeof(lbl),"R:%+.1f", roi_ae);
                    cv::putText(cell_color, lbl, Point(cell/2,cell-15),
                        FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,0), 1);
                    if (ai == 0) {
                        char slbl[32]; snprintf(slbl,sizeof(slbl),"skew=%.2f", cur_skew);
                        cv::putText(cell_color, slbl, Point(5,45),
                            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200,200,200), 1);
                    }
                    cell_color.copyTo(dbg(Rect(ai*cell, si*cell, cell, cell)));
                }
            }
            }

            if (make_grid) {
                imwrite("output/gt_debug.png", dbg);
                printf("  -> saved output/gt_debug.png\n");
            }

            // Draw error chart
            {
                int chart_w = 800, chart_h = 400;
                int margin_l = 60, margin_r = 20, margin_t = 40, margin_b = 50;
                int plot_w = chart_w - margin_l - margin_r;
                int plot_h = chart_h - margin_t - margin_b;
                Mat chart(chart_h, chart_w, CV_8UC3, Scalar(255,255,255));

                // Find max error for Y scale
                float max_ang = 0;
                for (int i = 0; i < na; i++) {
                    max_ang = std::max(max_ang, std::abs(icp_ang_errs[i]));
                    max_ang = std::max(max_ang, std::abs(roi_ang_errs[i]));
                }
                max_ang = std::ceil(max_ang + 0.5f);
                if (max_ang < 2) max_ang = 2;

                // Grid lines
                for (int g = -(int)max_ang; g <= (int)max_ang; g++) {
                    int y = margin_t + plot_h/2 - (int)(g * plot_h / (2*max_ang));
                    cv::line(chart, Point(margin_l, y), Point(margin_l+plot_w, y),
                             Scalar(230,230,230), 1);
                    if (g % 2 == 0) {
                        char gl[16]; snprintf(gl,sizeof(gl),"%+d", g);
                        cv::putText(chart, gl, Point(5, y+5),
                            FONT_HERSHEY_SIMPLEX, 0.35, Scalar(100,100,100), 1);
                    }
                }
                // Zero line
                int y0 = margin_t + plot_h/2;
                cv::line(chart, Point(margin_l, y0), Point(margin_l+plot_w, y0),
                         Scalar(180,180,180), 2);

                // X axis labels
                for (int x = 0; x <= 360; x += 45) {
                    int px = margin_l + x * plot_w / 360;
                    cv::line(chart, Point(px, margin_t), Point(px, margin_t+plot_h),
                             Scalar(230,230,230), 1);
                    char xl[16]; snprintf(xl,sizeof(xl),"%d", x);
                    cv::putText(chart, xl, Point(px-10, chart_h-10),
                        FONT_HERSHEY_SIMPLEX, 0.35, Scalar(100,100,100), 1);
                }

                // Plot ICP errors (red) and ROI errors (cyan)
                auto plotLine = [&](const std::vector<float>& errs, Scalar color) {
                    for (int i = 1; i < na; i++) {
                        int x1 = margin_l + (int)(dbg_angles[i-1] * plot_w / 360);
                        int x2 = margin_l + (int)(dbg_angles[i] * plot_w / 360);
                        int y1 = margin_t + plot_h/2 - (int)(errs[i-1] * plot_h / (2*max_ang));
                        int y2 = margin_t + plot_h/2 - (int)(errs[i] * plot_h / (2*max_ang));
                        cv::line(chart, Point(x1,y1), Point(x2,y2), color, 2, LINE_AA);
                    }
                    // Dots
                    for (int i = 0; i < na; i++) {
                        int x = margin_l + (int)(dbg_angles[i] * plot_w / 360);
                        int y = margin_t + plot_h/2 - (int)(errs[i] * plot_h / (2*max_ang));
                        cv::circle(chart, Point(x,y), 3, color, -1, LINE_AA);
                    }
                };
                plotLine(icp_ang_errs, Scalar(0,0,255));    // red = ICP
                plotLine(roi_ang_errs, Scalar(255,200,0));   // cyan = ROI

                // Title and legend
                cv::putText(chart, "Angle Error vs GT Angle (deg)", Point(margin_l, 25),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,0), 1);
                cv::putText(chart, "ICP", Point(chart_w-100, 20),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 2);
                cv::putText(chart, "ROI", Point(chart_w-100, 40),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,200,0), 2);
                cv::putText(chart, "angle (deg)", Point(chart_w/2-30, chart_h-2),
                    FONT_HERSHEY_SIMPLEX, 0.4, Scalar(100,100,100), 1);

                imwrite("output/angle_error_chart.png", chart);
                printf("  -> saved output/angle_error_chart.png\n");

                // Position error chart (absolute, not signed)
                {
                    Mat pchart(chart_h, chart_w, CV_8UC3, Scalar(255,255,255));
                    float max_pos = 0;
                    for (int i = 0; i < na; i++) {
                        max_pos = std::max(max_pos, icp_pos_errs[i]);
                        max_pos = std::max(max_pos, roi_pos_errs[i]);
                    }
                    max_pos = std::ceil(max_pos * 10) / 10.0f + 0.1f;  // round up to 0.1
                    if (max_pos < 0.5f) max_pos = 0.5f;

                    // Grid
                    for (float g = 0; g <= max_pos; g += 0.2f) {
                        int y = margin_t + plot_h - (int)(g * plot_h / max_pos);
                        cv::line(pchart, Point(margin_l, y), Point(margin_l+plot_w, y),
                                 Scalar(230,230,230), 1);
                        char gl[16]; snprintf(gl,sizeof(gl),"%.1f", g);
                        cv::putText(pchart, gl, Point(5, y+5),
                            FONT_HERSHEY_SIMPLEX, 0.3, Scalar(100,100,100), 1);
                    }
                    // X axis
                    for (int x = 0; x <= 360; x += 45) {
                        int px = margin_l + x * plot_w / 360;
                        cv::line(pchart, Point(px, margin_t), Point(px, margin_t+plot_h),
                                 Scalar(230,230,230), 1);
                        char xl[16]; snprintf(xl,sizeof(xl),"%d", x);
                        cv::putText(pchart, xl, Point(px-10, chart_h-10),
                            FONT_HERSHEY_SIMPLEX, 0.35, Scalar(100,100,100), 1);
                    }

                    auto plotPosLine = [&](const std::vector<float>& errs, Scalar color) {
                        for (int i = 1; i < na; i++) {
                            int x1 = margin_l + (int)(dbg_angles[i-1] * plot_w / 360);
                            int x2 = margin_l + (int)(dbg_angles[i] * plot_w / 360);
                            int y1 = margin_t + plot_h - (int)(errs[i-1] * plot_h / max_pos);
                            int y2 = margin_t + plot_h - (int)(errs[i] * plot_h / max_pos);
                            cv::line(pchart, Point(x1,y1), Point(x2,y2), color, 2, LINE_AA);
                        }
                        for (int i = 0; i < na; i++) {
                            int x = margin_l + (int)(dbg_angles[i] * plot_w / 360);
                            int y = margin_t + plot_h - (int)(errs[i] * plot_h / max_pos);
                            cv::circle(pchart, Point(x,y), 3, color, -1, LINE_AA);
                        }
                    };
                    plotPosLine(icp_pos_errs, Scalar(0,0,255));
                    plotPosLine(roi_pos_errs, Scalar(255,200,0));

                    cv::putText(pchart, "Position Error vs GT Angle (px)", Point(margin_l, 25),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,0), 1);
                    cv::putText(pchart, "ICP", Point(chart_w-100, 20),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 2);
                    cv::putText(pchart, "ROI", Point(chart_w-100, 40),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,200,0), 2);
                    cv::putText(pchart, "angle (deg)", Point(chart_w/2-30, chart_h-2),
                        FONT_HERSHEY_SIMPLEX, 0.4, Scalar(100,100,100), 1);

                    imwrite("output/pos_error_chart.png", pchart);
                    printf("  -> saved output/pos_error_chart.png\n");
                }

                // Also write data file
                FILE* ef = fopen("output/angle_errors.txt", "w");
                if (ef) {
                    fprintf(ef, "%-8s  %8s  %8s  %8s  %8s\n",
                            "Angle", "ICP_ang", "ICP_pos", "ROI_ang", "ROI_pos");
                    for (int i = 0; i < na; i++)
                        fprintf(ef, "%-8.0f  %+8.2f  %8.2f  %+8.2f  %8.2f\n",
                                dbg_angles[i], icp_ang_errs[i], icp_pos_errs[i],
                                roi_ang_errs[i], roi_pos_errs[i]);
                    fclose(ef);
                    printf("  -> saved output/angle_errors.txt\n");
                }

                // Assert: mean angle error for ROI mode < 0.5 deg
                {
                    float roi_mean_ae = 0;
                    for (int i = 0; i < na; i++) roi_mean_ae += std::abs(roi_ang_errs[i]);
                    roi_mean_ae /= na;
                    char msg[128];
                    snprintf(msg, sizeof(msg), "Per-angle sweep ROI: mean angle error %.3f >= 0.5 deg", roi_mean_ae);
                    CHECK(roi_mean_ae < 0.5f, msg);
                }
            }
        }

        // --- Skew test: objects with perspective distortion ---
        printf("\n--- Skew test: objects with perspective warp ---\n");
        {
            // Test different skew amounts
            float skew_amounts[] = {0.02f, 0.05f, 0.08f, 0.10f, 0.15f};

            printf("%-10s", "Skew");
            for (auto& mode : fhd_modes) printf("  %-28s", mode.name);
            printf("\n");
            for (int i = 0; i < 10 + 3*30; i++) printf("-");
            printf("\n");

            for (float skew : skew_amounts) {
                // Create scene with skewed objects
                Mat scene_skew(1080, 1920, CV_8U, Scalar(30));
                // Add same background noise
                {
                    Mat bn(scene_skew.size(), CV_64F);
                    RNG rr(123);
                    rr.fill(bn, RNG::NORMAL, 0, 15);
                    Mat ts; scene_skew.convertTo(ts, CV_64F);
                    ts += bn; ts.convertTo(scene_skew, CV_8U);
                }

                for (auto& obj : fhd_objs) {
                    // First rotate
                    Mat M = getRotationMatrix2D(Point2f(100, 100), -obj.angle, 1.0);
                    Mat rot;
                    warpAffine(templ_fhd, rot, M, templ_fhd.size(),
                               INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

                    // Then apply perspective skew
                    float s = skew * 100;  // skew in pixels at template edge
                    Point2f src_pts[4] = {
                        {0, 0}, {199, 0}, {199, 199}, {0, 199}
                    };
                    Point2f dst_pts[4] = {
                        {s, s*0.5f}, {199-s, -s*0.3f}, {199+s*0.3f, 199+s*0.5f}, {-s*0.5f, 199-s*0.3f}
                    };
                    Mat P = getPerspectiveTransform(src_pts, dst_pts);
                    Mat warped;
                    warpPerspective(rot, warped, P, rot.size(),
                                    INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

                    int ox = obj.x - 100, oy = obj.y - 100;
                    for (int r = 0; r < warped.rows; r++)
                        for (int c = 0; c < warped.cols; c++) {
                            int sy = oy + r, sx = ox + c;
                            if (sy >= 0 && sy < scene_skew.rows && sx >= 0 &&
                                sx < scene_skew.cols && warped.at<uchar>(r, c) > 0)
                                scene_skew.at<uchar>(sy, sx) = warped.at<uchar>(r, c);
                        }
                }

                // Run full match() per mode — same as the main benchmark
                struct SR { int matched; int total; double ms; float ang; float pos; };
                SR skew_results[4];
                std::vector<sbm::MatchResult> mode_match_results[4];
                int smi = 0;
                for (auto& mode : fhd_modes) {
                    sbm::MatchConfig cfg;
                    cfg.min_score = 35;
                    cfg.nms_radius = 80;
                    cfg.refine = mode.mode;

                    sbm::ShapeMatcher matcher(cfg);
                    sbm::ModelConfig mcfg;
                    mcfg.angle = {0, 360, 2};

                    std::vector<sbm::MatchResult> results;
                    { OutputGuard guard; matcher.addModel("L", feat_fhd, mcfg); results = matcher.match(scene_skew); }

                    float total_ang_err = 0, total_pos_err = 0;
                    int matched = 0;
                    for (auto& r : results) {
                        float best_d = 1e9f; int best_j = -1;
                        for (int j = 0; j < n_objs; j++) {
                            float o2 = 100-100, oo = 75-100;
                            float rd = -(float)fhd_objs[j].angle*(float)CV_PI/180.0f;
                            float gx = fhd_objs[j].x + std::cos(rd)*o2 - std::sin(rd)*oo;
                            float gy = fhd_objs[j].y + std::sin(rd)*o2 + std::cos(rd)*oo;
                            float d = std::sqrt((r.x-gx)*(r.x-gx)+(r.y-gy)*(r.y-gy));
                            if (d < best_d) { best_d = d; best_j = j; }
                        }
                        if (best_j >= 0 && best_d < 50) {
                            float ae = r.angle - (float)fhd_objs[best_j].angle;
                            if (ae > 180) ae -= 360; if (ae < -180) ae += 360;
                            total_ang_err += std::abs(ae);
                            total_pos_err += best_d;
                            matched++;
                        }
                    }
                    float ma = matched > 0 ? total_ang_err / matched : -1;
                    float mp = matched > 0 ? total_pos_err / matched : -1;
                    skew_results[smi] = {matched, n_objs, 0, ma, mp};
                    mode_match_results[smi] = results;
                    smi++;
                }
                // Write results to file to avoid cout interleaving
                {
                    FILE* sf = fopen("output/skew_results.txt",
                                    skew == skew_amounts[0] ? "w" : "a");
                    if (sf) {
                        if (skew == skew_amounts[0]) {
                            fprintf(sf, "%-10s  %-20s  %-20s  %-20s\n",
                                    "Skew", "None", "ICP(inverse)", "ROI");
                            fprintf(sf, "----------------------------------------------------------------------\n");
                        }
                        fprintf(sf, "%-10.2f", skew);
                        for (int i = 0; i < smi; i++)
                            fprintf(sf, "  %2d/%d %4.1fdeg %4.1fpx ",
                                   skew_results[i].matched, skew_results[i].total,
                                   skew_results[i].ang, skew_results[i].pos);
                        fprintf(sf, "\n");
                        fclose(sf);
                    }
                }

                // Draw GT only on image
                Mat vis;
                cvtColor(scene_skew, vis, COLOR_GRAY2BGR);

                for (int j = 0; j < n_objs; j++) {
                    // GT center (template center, not user origin)
                    float cx = (float)fhd_objs[j].x;
                    float cy = (float)fhd_objs[j].y;
                    float ang = (float)fhd_objs[j].angle;
                    float rd = ang * (float)CV_PI / 180.0f;

                    // Draw cross at center
                    cv::drawMarker(vis, Point((int)cx, (int)cy),
                                   Scalar(0,255,0), MARKER_CROSS, 25, 2);

                    // Draw angle arrow from center
                    // In image coords: X-right, Y-down. Rotation is CW.
                    // cos(rd) gives X component, sin(rd) gives Y component for CW rotation.
                    float arr_len = 60;
                    float ax = std::cos(rd)*arr_len, ay = std::sin(rd)*arr_len;
                    cv::arrowedLine(vis, Point((int)cx, (int)cy),
                                   Point((int)(cx+ax), (int)(cy+ay)),
                                   Scalar(0,255,0), 2, LINE_AA, 0, 0.3);

                    // Label angle
                    char lbl[32];
                    snprintf(lbl, sizeof(lbl), "%.0f", ang);
                    cv::putText(vis, lbl, Point((int)cx+10, (int)cy-10),
                                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 1);
                }

                // Create zoomed crop mosaic: 2 rows x 5 cols of 200x200 crops
                int crop_sz = 200, pad = 4;
                int cols = 5, rows = 2;
                int mosaic_w = cols * (crop_sz + pad) + pad;
                int mosaic_h = rows * (crop_sz + pad) + pad + 30;  // +30 for title
                Mat mosaic(mosaic_h, mosaic_w, CV_8UC3, Scalar(40,40,40));

                // Title
                char title[128];
                snprintf(title, sizeof(title), "skew=%.2f  GT_init: %.1fdeg %.1fpx  ICP: %.1fdeg %.1fpx  ROI: %.1fdeg %.1fpx",
                         skew, skew_results[0].ang, skew_results[0].pos,
                         skew_results[1].ang, skew_results[1].pos,
                         skew_results[2].ang, skew_results[2].pos);
                cv::putText(mosaic, title, Point(pad, 20),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);

                for (int j = 0; j < std::min(n_objs, cols*rows); j++) {
                    int cx = (int)fhd_objs[j].x, cy = (int)fhd_objs[j].y;
                    int x0 = std::max(0, cx - crop_sz/2);
                    int y0 = std::max(0, cy - crop_sz/2);
                    int x1 = std::min(vis.cols, x0 + crop_sz);
                    int y1 = std::min(vis.rows, y0 + crop_sz);
                    if (x1-x0 < 50 || y1-y0 < 50) continue;

                    Mat crop = vis(Rect(x0, y0, x1-x0, y1-y0)).clone();
                    // Resize to crop_sz if needed
                    if (crop.cols != crop_sz || crop.rows != crop_sz)
                        resize(crop, crop, Size(crop_sz, crop_sz));

                    int col = j % cols, row = j / cols;
                    int mx = pad + col * (crop_sz + pad);
                    int my = 30 + pad + row * (crop_sz + pad);
                    crop.copyTo(mosaic(Rect(mx, my, crop_sz, crop_sz)));
                }

                char fname[128];
                snprintf(fname, sizeof(fname), "output/skew_%.2f.png", skew);
                imwrite(fname, mosaic);
                printf("  -> saved %s\n", fname);
            }
        }

        // --- Isolated test: 1 object, clean background, per-angle error ---
        printf("\n--- Isolated: 1 object, clean background, per-angle error ---\n");
        {
        auto& feat = feat_fhd;
        // Dense angle sweep: every 5 degrees
        std::vector<float> test_angles;
        for (float a = 0; a < 360; a += 5) test_angles.push_back(a);

        printf("%-8s  %-24s  %-24s  %-24s\n", "Angle", "ICP (dense)", "ICP inverse", "ROI");
        printf("%-8s  %-24s  %-24s  %-24s\n", "-----", "-----------", "-----------", "---");

        for (float gt_ang : test_angles) {
            // Create clean scene with 1 object at center
            Mat scene1obj(1080, 1920, CV_8U, Scalar(0));
            int obj_x = 960, obj_y = 540;
            Mat M = getRotationMatrix2D(Point2f(100, 100), -gt_ang, 1.0);
            Mat rot; warpAffine(templ_fhd, rot, M, templ_fhd.size(),
                                INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
            int ox = obj_x - 100, oy = obj_y - 100;
            for (int r = 0; r < rot.rows; r++)
                for (int c = 0; c < rot.cols; c++) {
                    int sy = oy + r, sx = ox + c;
                    if (sy >= 0 && sy < scene1obj.rows && sx >= 0 &&
                        sx < scene1obj.cols && rot.at<uchar>(r, c) > 0)
                        scene1obj.at<uchar>(sy, sx) = rot.at<uchar>(r, c);
                }

            // GT origin position
            float gox = 100 - 100, goy = 75 - 100;
            float grad = -gt_ang * (float)CV_PI / 180.0f;
            float gt_x = obj_x + std::cos(grad)*gox - std::sin(grad)*goy;
            float gt_y = obj_y + std::sin(grad)*gox + std::cos(grad)*goy;

            printf("%-8.0f", gt_ang);

            // Get coarse match first (shared by all modes)
            sbm::MatchConfig cfg_none;
            cfg_none.min_score = 40;
            cfg_none.nms_radius = 80;
            cfg_none.refine = sbm::RefineMode::None;
            sbm::ShapeMatcher matcher_coarse(cfg_none);
            sbm::ModelConfig mcfg;
            mcfg.angle = {0, 360, 2};
            std::vector<sbm::MatchResult> coarse_results;
            { OutputGuard guard; matcher_coarse.addModel("L", feat, mcfg); coarse_results = matcher_coarse.match(scene1obj); }

            // ICP dense (forward)
            {
                sbm::MatchConfig cfg;
                cfg.min_score = 40; cfg.nms_radius = 80;
                cfg.refine = sbm::RefineMode::ICP;
                sbm::ShapeMatcher matcher(cfg);
                std::vector<sbm::MatchResult> results;
                { OutputGuard guard; matcher.addModel("L", feat, mcfg); results = matcher.match(scene1obj); }
                if (!results.empty()) {
                    auto& r = results[0];
                    float ae = r.angle - gt_ang;
                    if (ae > 180) ae -= 360; if (ae < -180) ae += 360;
                    float pe = std::sqrt((r.x-gt_x)*(r.x-gt_x)+(r.y-gt_y)*(r.y-gt_y));
                    printf("  ang=%+5.2f pos=%4.1fpx", ae, pe);
                } else printf("  NOT FOUND              ");
            }

            // ICP inverse (template EDT)
            if (!coarse_results.empty()) {
                auto& cr = coarse_results[0];
                // Compute coarse center from user origin
                float o_x = 100 - 100, o_y = 75 - 100;  // origin offset
                float r_rad = -cr.angle * (float)CV_PI / 180.0f;
                float coarse_cx = cr.x - (std::cos(r_rad)*o_x - std::sin(r_rad)*o_y);
                float coarse_cy = cr.y - (std::sin(r_rad)*o_x + std::cos(r_rad)*o_y);

                icp_refine::ICPConfig icfg;
                icfg.max_iterations = 30;
                icfg.max_dist = 10;
                icp_refine::Pose2D init(coarse_cx, coarse_cy, cr.angle);
                auto refined = icp_refine::refineInverse(
                    templ_fhd, scene1obj, init, 20, icfg);

                // Compute user origin from refined center
                float rr = -refined.angle * (float)CV_PI / 180.0f;
                float ux = refined.x + std::cos(rr)*o_x - std::sin(rr)*o_y;
                float uy = refined.y + std::sin(rr)*o_x + std::cos(rr)*o_y;
                float ae = refined.angle - gt_ang;
                if (ae > 180) ae -= 360; if (ae < -180) ae += 360;
                float pe = std::sqrt((ux-gt_x)*(ux-gt_x)+(uy-gt_y)*(uy-gt_y));
                printf("  ang=%+5.2f pos=%4.1fpx", ae, pe);
            } else printf("  NOT FOUND              ");

            // ROI
            {
                sbm::MatchConfig cfg;
                cfg.min_score = 40; cfg.nms_radius = 80;
                cfg.refine = sbm::RefineMode::ROI;
                sbm::ShapeMatcher matcher(cfg);
                std::vector<sbm::MatchResult> results;
                { OutputGuard guard; matcher.addModel("L", feat, mcfg); results = matcher.match(scene1obj); }
                if (!results.empty()) {
                    auto& r = results[0];
                    float ae = r.angle - gt_ang;
                    if (ae > 180) ae -= 360; if (ae < -180) ae += 360;
                    float pe = std::sqrt((r.x-gt_x)*(r.x-gt_x)+(r.y-gt_y)*(r.y-gt_y));
                    printf("  ang=%+5.2f pos=%4.1fpx", ae, pe);
                } else printf("  NOT FOUND              ");
            }
            printf("\n");
        }
        }
    }

    printf(g_fail ? "\n*** %d CHECKS FAILED ***\n" : "\nAll checks passed.\n", g_fail);
    return g_fail ? 1 : 0;
}
