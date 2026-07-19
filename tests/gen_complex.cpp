// gen_complex.cpp — generate a cluttered test pattern: N random polygons +
// ellipses overlaid within a region, each rotated, random brightness. Useful as
// a realistic complex template/scene for the feature-set experiments.
//
//   gen_complex [out.png] [seed] [size] [count]

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const char* out = argc > 1 ? argv[1] : "complex.png";
    int seed  = argc > 2 ? std::atoi(argv[2]) : 12345;
    int W     = argc > 3 ? std::atoi(argv[3]) : 480;
    int count = argc > 4 ? std::atoi(argv[4]) : 30;

    cv::RNG rng(seed);
    cv::Mat img(W, W, CV_8U, cv::Scalar(40));   // mid-gray background (matches test shapes)

    const int m = W / 8;                        // region margin
    const int lo = m, hi = W - m;

    for (int i = 0; i < count; ++i) {
        int cx = rng.uniform(lo, hi), cy = rng.uniform(lo, hi);
        int bright = rng.uniform(60, 236);      // random brightness
        double ang = rng.uniform(0.0, 360.0);   // rotation

        if (rng.uniform(0, 2) == 0) {
            // rotated ellipse
            int a = rng.uniform(W / 20, W / 6), b = rng.uniform(W / 20, W / 6);
            cv::ellipse(img, {cx, cy}, {a, b}, ang, 0, 360, bright, cv::FILLED, cv::LINE_AA);
        } else {
            // rotated polygon (3..7 vertices, jittered radius/angle)
            int n = rng.uniform(3, 8);
            double R = rng.uniform(W / 18.0, W / 6.0);
            std::vector<cv::Point> pts;
            for (int k = 0; k < n; ++k) {
                double t = ang * CV_PI / 180.0 + 2 * CV_PI * k / n + rng.uniform(-0.25, 0.25);
                double rr = R * (1.0 + rng.uniform(-0.28, 0.28));
                pts.push_back({(int)(cx + rr * std::cos(t)), (int)(cy + rr * std::sin(t))});
            }
            std::vector<std::vector<cv::Point>> poly{pts};
            cv::fillPoly(img, poly, bright, cv::LINE_AA);
        }
    }

    if (!cv::imwrite(out, img)) { std::fprintf(stderr, "write failed: %s\n", out); return 1; }
    std::printf("wrote %s (%dx%d, %d shapes, seed %d)\n", out, W, W, count, seed);
    return 0;
}
