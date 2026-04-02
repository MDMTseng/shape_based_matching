// Minimal usage example of the ShapeMatcher API.

#include "shape_matcher.h"
#include "roi_refine.h"
#include "icp_refine.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>

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
            float ae = ref[2] - 25; if(ae>180)ae-=360; if(ae<-180)ae+=360;
            float pd = std::sqrt((ref[0]-160)*(ref[0]-160)+(ref[1]-120)*(ref[1]-120));
            printf("@%+5.1f %4.1fpx %4.1fms  ", ae, pd, roi_ms);
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
            float ae = ref[2] - 25; if(ae>180)ae-=360; if(ae<-180)ae+=360;
            float pd = std::sqrt((ref[0]-160)*(ref[0]-160)+(ref[1]-120)*(ref[1]-120));
            printf("@%+5.1f %4.1fpx %4.1fms", ae, pd, roi_ms);
        }

        printf("\n");
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

        // Suppress meiqua cout during benchmark
        std::streambuf* orig_cout = std::cout.rdbuf();
        std::ostringstream null_stream;

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

                // Warm up (mute cout)
                std::cout.rdbuf(null_stream.rdbuf());
                matcher.match(scene_test);
                std::cout.rdbuf(orig_cout);

                // Average 3 runs (mute cout)
                double total_ms = 0;
                std::vector<sbm::MatchResult> results;
                std::cout.rdbuf(null_stream.rdbuf());
                for (int i = 0; i < 3; i++) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    results = matcher.match(scene_test);
                    total_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                }
                std::cout.rdbuf(orig_cout);
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
        }

        // --- Isolated test: 1 object, clean background, per-angle error ---
        printf("\n--- Isolated: 1 object, clean background, per-angle error ---\n");
        {
        auto& feat = feat_fhd;
        float test_angles[] = {0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 270, 315};

        printf("%-8s  %-24s  %-24s\n", "Angle", "ICP (dense)", "ROI");
        printf("%-8s  %-24s  %-24s\n", "-----", "-----------", "---");

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

            Mode iso_modes[] = {
                {"ICP", sbm::RefineMode::ICP},
                {"ROI", sbm::RefineMode::ROI},
            };

            for (auto& mode : iso_modes) {
                sbm::MatchConfig cfg;
                cfg.min_score = 40;
                cfg.nms_radius = 80;
                cfg.refine = mode.mode;

                sbm::ShapeMatcher matcher(cfg);
                sbm::ModelConfig mcfg;
                mcfg.angle = {0, 360, 2};
                matcher.addModel("L", feat, mcfg);

                std::cout.rdbuf(null_stream.rdbuf());
                auto results = matcher.match(scene1obj);
                std::cout.rdbuf(orig_cout);

                if (!results.empty()) {
                    auto& r = results[0];
                    float ae = r.angle - gt_ang;
                    if (ae > 180) ae -= 360; if (ae < -180) ae += 360;
                    float dx = r.x - gt_x, dy = r.y - gt_y;
                    float pe = std::sqrt(dx*dx + dy*dy);
                    printf("  ang=%+5.2f pos=%4.1fpx", ae, pe);
                } else {
                    printf("  NOT FOUND              ");
                }
            }
            printf("\n");
        }
        }
    }

    return 0;
}
