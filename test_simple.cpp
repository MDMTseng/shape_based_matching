// Minimal usage example of the ShapeMatcher API.

#include "shape_matcher.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
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

    // 3. Create matcher + load features
    sbm::MatchConfig cfg;
    cfg.min_score = 50;
    cfg.nms_radius = 50;
    cfg.refine = sbm::RefineMode::ICP;

    sbm::ShapeMatcher matcher(cfg);

    sbm::ModelConfig mcfg;
    mcfg.angle = {0, 360, 2};

    matcher.addModel("rect", sbm::FeatureSet::load("rect.feat"), mcfg);
    printf("Templates: %d\n", matcher.numTemplates());

    // 4. Create scene with 3 rotated rectangles
    Mat scene(480, 640, CV_8U, Scalar(30));
    for (auto& obj : std::vector<std::pair<Point,double>>{{Point(160,120),25},{Point(320,240),90},{Point(480,360),200}}) {
        Mat M = getRotationMatrix2D(Point2f(40,30), -obj.second, 1.0);
        Mat rot; warpAffine(templ, rot, M, templ.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
        int ox = obj.first.x - 40, oy = obj.first.y - 30;
        for (int r=0; r<rot.rows; r++) for (int c=0; c<rot.cols; c++) {
            int sy=oy+r, sx=ox+c;
            if (sy>=0 && sy<scene.rows && sx>=0 && sx<scene.cols && rot.at<uchar>(r,c)>0)
                scene.at<uchar>(sy,sx) = rot.at<uchar>(r,c);
        }
    }

    // 5. Match
    auto results = matcher.match(scene);
    printf("\nFound %d matches:\n", (int)results.size());
    for (auto& r : results)
        printf("  %s at (%.0f, %.0f) angle=%.1f score=%.0f\n",
               r.model_name.c_str(), r.x, r.y, r.angle, r.score);

    return 0;
}
