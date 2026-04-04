#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
using namespace cv;
using namespace std::chrono;

int main() {
    printf("%-20s %10s %10s %10s\n", "Stage", "VGA", "FHD", "4K");
    printf("%-20s %10s %10s %10s\n", "-----", "---", "---", "--");

    struct Sz { int w, h; const char* n; };
    Sz sizes[] = {{640,480,"VGA"},{1920,1080,"FHD"},{3840,2160,"4K"}};
    const int RUNS = 20;

    double blur_ms[3], sobel_ms[3], phase_ms[3], quant_ms[3];

    for (int si = 0; si < 3; ++si) {
        Mat img(sizes[si].h, sizes[si].w, CV_8U, Scalar(128));
        RNG rng(42);
        rng.fill(img, RNG::UNIFORM, 0, 256);

        Mat smoothed, sobel_dx, sobel_dy, mag, ang;

        // Warmup
        GaussianBlur(img, smoothed, Size(7,7), 0, 0, BORDER_REPLICATE);
        Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3);
        Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3);

        // GaussianBlur 7x7
        auto t0 = high_resolution_clock::now();
        for (int r = 0; r < RUNS; ++r)
            GaussianBlur(img, smoothed, Size(7,7), 0, 0, BORDER_REPLICATE);
        blur_ms[si] = duration<double,std::milli>(high_resolution_clock::now()-t0).count()/RUNS;

        // Sobel dx + dy
        t0 = high_resolution_clock::now();
        for (int r = 0; r < RUNS; ++r) {
            Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
            Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
        }
        sobel_ms[si] = duration<double,std::milli>(high_resolution_clock::now()-t0).count()/RUNS;

        // magnitude + phase
        t0 = high_resolution_clock::now();
        for (int r = 0; r < RUNS; ++r) {
            mag = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
            phase(sobel_dx, sobel_dy, ang, true);
        }
        phase_ms[si] = duration<double,std::milli>(high_resolution_clock::now()-t0).count()/RUNS;

        // quantize (angle->8bin + 3x3 voting)
        // Simplified: just the convertTo + histogram part
        t0 = high_resolution_clock::now();
        for (int r = 0; r < RUNS; ++r) {
            Mat quant;
            ang.convertTo(quant, CV_8U, 16.0/360.0);
        }
        quant_ms[si] = duration<double,std::milli>(high_resolution_clock::now()-t0).count()/RUNS;
    }

    printf("%-20s", "GaussianBlur 7x7");
    for (int i = 0; i < 3; ++i) printf(" %8.1fms", blur_ms[i]);
    printf("\n");

    printf("%-20s", "Sobel dx+dy");
    for (int i = 0; i < 3; ++i) printf(" %8.1fms", sobel_ms[i]);
    printf("\n");

    printf("%-20s", "mag+phase");
    for (int i = 0; i < 3; ++i) printf(" %8.1fms", phase_ms[i]);
    printf("\n");

    printf("%-20s", "quantize");
    for (int i = 0; i < 3; ++i) printf(" %8.1fms", quant_ms[i]);
    printf("\n");

    printf("%-20s", "TOTAL preprocess");
    for (int i = 0; i < 3; ++i)
        printf(" %8.1fms", blur_ms[i]+sobel_ms[i]+phase_ms[i]+quant_ms[i]);
    printf("\n");

    return 0;
}
