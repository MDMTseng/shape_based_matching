// Minimal usage example of the ShapeMatcher API.

#include "shape_matcher.h"
#include "roi_refine.h"
#include "icp_refine.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>

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
#define TEMPLATE_SHAPE 3  // 0=L, 1=V, 2=parallel lines, 3=long pole
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

    // Quality evaluation (user-facing API)
    auto quality = loaded.evaluateQuality();
    printf("\n  Quality Score: %d/100 — %s\n", quality.score, quality.diagnosis.c_str());
    printf("    cond=%.0f  coverage=%.0f deg  dirs=%d  strength=%.0f  %de+%dc\n\n",
           quality.condition_number, quality.angle_coverage_deg,
           quality.num_directions, quality.mean_edge_strength,
           quality.num_edge, quality.num_corner);

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
        { 0,  0,  0,  0,11, "blur k=11"},
        { 5,  3,  5, 20, 3, "+5px+5deg+n+b"},
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
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        printf("%-20s (%.1fms): ", mode.name, ms);
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

    return 0;
}
