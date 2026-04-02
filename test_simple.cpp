// Minimal usage example of the ShapeMatcher API.

#include "shape_matcher.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>

using namespace cv;

int main() {
    // 1. Create a template (white rectangle on black)
    Mat templ(60, 80, CV_8U, Scalar(0));
    rectangle(templ, Point(10, 10), Point(70, 50), Scalar(200), -1);

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
    };

    auto loaded = sbm::FeatureSet::load("rect.feat");
    int n_corner = 0, n_edge = 0;
    float max_c = 0, sum_c = 0;
    for (auto& f : loaded.levels[0].features) {
        if (f.cornerness > 0.1f) n_corner++; else n_edge++;
        max_c = std::max(max_c, f.cornerness);
        sum_c += f.cornerness;
    }
    printf("ICP edges: %d (dense), %d (sparse: %d edges + %d corners, max_c=%.3f avg_c=%.3f)\n\n",
           (int)loaded.icp_edges.size(), loaded.numFeatures(), n_edge, n_corner,
           max_c, sum_c / loaded.numFeatures());

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

        printf("%-12s (%.1fms): ", mode.name, ms);
        for (auto& r : results)
            printf("(%3.0f,%3.0f)@%5.1f  ", r.x, r.y, r.angle);
        printf("\n");
    }

    return 0;
}
