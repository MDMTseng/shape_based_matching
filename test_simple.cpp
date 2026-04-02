// Minimal usage example of the ShapeMatcher API.

#include "shape_matcher.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>

using namespace cv;

int main() {
    // 1. Create a template (L-shape, asymmetric — no 180° ambiguity)
    Mat templ(80, 80, CV_8U, Scalar(0));
    rectangle(templ, Point(25, 10), Point(35, 70), Scalar(200), -1);
    rectangle(templ, Point(25, 50), Point(65, 70), Scalar(200), -1);

    // 2. Extract features + save
    auto features = sbm::extractFeatures(templ);
    features.setOrigin(40, 30);  // center of rectangle
    features.save("rect.feat");
    printf("Extracted %d features\n", features.numFeatures());

    // 3. Create scene with 3 rotated rectangles
    struct Obj { int x, y; double angle; };
    Obj objs[] = {{160,120,25}, {320,240,90}, {480,360,200}};

    Mat scene(480, 640, CV_8U, Scalar(30));
    for (auto& obj : objs) {
        Mat M = getRotationMatrix2D(Point2f(40,30), -obj.angle, 1.0);
        Mat rot; warpAffine(templ, rot, M, templ.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
        int ox = obj.x - 40, oy = obj.y - 30;
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

    for (auto& mode : modes) {
        sbm::MatchConfig cfg;
        cfg.min_score = 50;
        cfg.nms_radius = 50;
        cfg.refine = mode.mode;

        sbm::ShapeMatcher matcher(cfg);
        sbm::ModelConfig mcfg;
        mcfg.angle = {0, 360, 2};
        matcher.addModel("rect", loaded, mcfg);

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
