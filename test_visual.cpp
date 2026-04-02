// Visual matching test: outputs annotated images showing detections.
// Draws match rectangles + scores on scenes under various conditions.

#include "line2Dup.h"
#include "icp_refine.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdio>
#include <string>

using namespace cv;
using namespace std;

static void draw_L(Mat& img, int cx, int cy, double angle, int color, double scale = 2.0) {
    double rad = angle * CV_PI / 180.0;
    double cs = cos(rad), sn = sin(rad);
    for (double ly = -15*scale; ly <= 15*scale; ly += 0.5)
        for (double lx = -5*scale; lx <= 5*scale; lx += 0.5) {
            int px = cx + (int)(lx*cs - ly*sn + 0.5);
            int py = cy + (int)(lx*sn + ly*cs + 0.5);
            if (px >= 0 && px < img.cols && py >= 0 && py < img.rows)
                img.at<uchar>(py, px) = (uchar)color;
        }
    for (double ly = 5*scale; ly <= 15*scale; ly += 0.5)
        for (double lx = 5*scale; lx <= 20*scale; lx += 0.5) {
            int px = cx + (int)(lx*cs - ly*sn + 0.5);
            int py = cy + (int)(lx*sn + ly*cs + 0.5);
            if (px >= 0 && px < img.cols && py >= 0 && py < img.rows)
                img.at<uchar>(py, px) = (uchar)color;
        }
}

static void add_noise(Mat& img, double sigma) {
    Mat noise(img.size(), CV_64F);
    RNG rng(42);
    rng.fill(noise, RNG::NORMAL, 0, sigma);
    Mat result;
    img.convertTo(result, CV_64F);
    result += noise;
    result.convertTo(img, CV_8U);
}

static Mat pad16(const Mat& img) {
    int pw = (img.cols + 15) & ~15;
    int ph = (img.rows + 15) & ~15;
    if (pw != img.cols || ph != img.rows) {
        Mat padded;
        copyMakeBorder(img, padded, 0, ph - img.rows, 0, pw - img.cols,
                       BORDER_CONSTANT, Scalar(0));
        return padded;
    }
    return img;
}

// Spatial NMS: keep best match per location (greedy, sorted by score descending)
static vector<line2Dup::Match> spatial_nms(const vector<line2Dup::Match>& matches, float radius) {
    float r2 = radius * radius;
    vector<line2Dup::Match> kept;
    for (auto& m : matches) {
        bool suppressed = false;
        for (auto& k : kept) {
            float dx = (float)(m.x - k.x), dy = (float)(m.y - k.y);
            if (dx*dx + dy*dy < r2) { suppressed = true; break; }
        }
        if (!suppressed) kept.push_back(m);
    }
    return kept;
}

// Draw matches on a color image
// angle_step: degrees per template_id (e.g., 2 for 2-degree steps)
static Mat draw_matches(const Mat& scene_gray, const vector<line2Dup::Match>& raw_matches,
                        const line2Dup::Detector& det, int TW, double match_ms = -1,
                        double angle_step = 2.0) {
    Mat vis;
    cvtColor(scene_gray, vis, cv::COLOR_GRAY2BGR);

    // Spatial NMS to deduplicate overlapping detections
    auto matches = spatial_nms(raw_matches, (float)TW * 0.8f);

    // (ICP refinement done externally before calling this function)

    int draw_n = std::min((int)matches.size(), 50);
    double arrow_len = TW * 0.45;

    for (int i = 0; i < draw_n; ++i) {
        auto& m = matches[i];
        Scalar color;
        if (m.similarity >= 80) color = Scalar(0, 255, 0);       // green
        else if (m.similarity >= 60) color = Scalar(0, 255, 255); // yellow
        else color = Scalar(0, 0, 255);                           // red

        // Compute object center from template bounding box
        // m.x, m.y is at the template's (tl_x, tl_y) in scene coords
        auto& tmpl = det.getTemplates(m.class_id, m.template_id);
        int cx = m.x + tmpl[0].width / 2;
        int cy = m.y + tmpl[0].height / 2;
        Point center(cx, cy);

        // Orientation arrow: use refined_angle if available, else template_id * step
        double angle_deg = (m.refined_angle >= 0) ? m.refined_angle
                                                   : m.template_id * angle_step;
        double angle_rad = angle_deg * CV_PI / 180.0;
        Point arrow_tip(
            cx + (int)(arrow_len * cos(angle_rad)),
            cy + (int)(arrow_len * sin(angle_rad)));

        // Draw circle at center
        circle(vis, center, 3, color, -1);

        // Draw orientation arrow
        arrowedLine(vis, center, arrow_tip, color, 2, cv::LINE_AA, 0, 0.25);

        // Draw score + refined angle text
        char buf[48];
        snprintf(buf, sizeof(buf), "%.0f @%.1f", m.similarity, angle_deg);
        putText(vis, buf, Point(cx + 5, cy - 8),
                FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0,0,0), 2);
        putText(vis, buf, Point(cx + 5, cy - 8),
                FONT_HERSHEY_SIMPLEX, 0.35, color, 1);
    }

    // Show raw/NMS counts and time
    char buf[128];
    if (match_ms >= 0)
        snprintf(buf, sizeof(buf), "Raw: %d  NMS: %d  Time: %.1fms",
                 (int)raw_matches.size(), (int)matches.size(), match_ms);
    else
        snprintf(buf, sizeof(buf), "Raw: %d  NMS: %d",
                 (int)raw_matches.size(), (int)matches.size());
    putText(vis, buf, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 3);
    putText(vis, buf, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

    return vis;
}

// Draw pre-NMS'd matches directly (no additional NMS)
static Mat draw_matches_direct(const Mat& scene_gray, const vector<line2Dup::Match>& matches,
                               const line2Dup::Detector& det, int TW, double match_ms = -1,
                               double angle_step = 2.0) {
    Mat vis;
    cvtColor(scene_gray, vis, cv::COLOR_GRAY2BGR);

    int draw_n = std::min((int)matches.size(), 50);
    double arrow_len = TW * 0.45;

    for (int i = 0; i < draw_n; ++i) {
        auto& m = matches[i];
        Scalar color;
        if (m.similarity >= 80) color = Scalar(0, 255, 0);
        else if (m.similarity >= 60) color = Scalar(0, 255, 255);
        else color = Scalar(0, 0, 255);

        auto& tmpl = det.getTemplates(m.class_id, m.template_id);
        int cx = m.x + tmpl[0].width / 2;
        int cy = m.y + tmpl[0].height / 2;
        Point center(cx, cy);

        double angle_deg = (m.refined_angle >= 0) ? m.refined_angle
                                                   : m.template_id * angle_step;
        double angle_rad = angle_deg * CV_PI / 180.0;
        Point arrow_tip(cx + (int)(arrow_len * cos(angle_rad)),
                        cy + (int)(arrow_len * sin(angle_rad)));

        circle(vis, center, 3, color, -1);
        arrowedLine(vis, center, arrow_tip, color, 2, cv::LINE_AA, 0, 0.25);

        char buf[48];
        snprintf(buf, sizeof(buf), "%.0f @%.1f", m.similarity, angle_deg);
        putText(vis, buf, Point(cx + 5, cy - 8), FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0,0,0), 2);
        putText(vis, buf, Point(cx + 5, cy - 8), FONT_HERSHEY_SIMPLEX, 0.35, color, 1);
    }

    char buf[128];
    snprintf(buf, sizeof(buf), "NMS: %d  Time: %.1fms", (int)matches.size(), match_ms);
    putText(vis, buf, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 3);
    putText(vis, buf, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

    return vis;
}

int main() {
    const string out_dir = "C:/Users/TRS001/Documents/workspace/templmatch/test_imgs/";
    const int W = 1920, H = 1080;
    const int TW = 80;
    const float threshold = 50.0f;

    // Build template
    Mat templ(TW, TW, CV_8U, Scalar(0));
    draw_L(templ, TW/2, TW/2, 0, 200);
    Mat mask_t = Mat::ones(TW, TW, CV_8U) * 255;

    // Multi-angle detector (2 deg steps = 180 templates)
    line2Dup::Detector det(128, {4, 8}, 30, 60);
    for (int angle = 0; angle < 360; angle += 2) {
        Mat rot_templ, rot_mask;
        Mat M = getRotationMatrix2D(Point2f(TW/2.0f, TW/2.0f), -angle, 1.0);
        warpAffine(templ, rot_templ, M, Size(TW, TW));
        warpAffine(mask_t, rot_mask, M, Size(TW, TW));
        det.addTemplate(rot_templ, "L", rot_mask);
    }
    printf("Templates: %d\n", det.numTemplates("L"));

    // Extract template edge points (relative to center) for ICP
    std::vector<cv::Point2f> templ_edge_pts;
    {
        Mat templ_smooth, templ_dx, templ_dy, templ_edge;
        GaussianBlur(templ, templ_smooth, Size(5, 5), 0);
        Sobel(templ_smooth, templ_dx, CV_16S, 1, 0, 3);
        Sobel(templ_smooth, templ_dy, CV_16S, 0, 1, 3);
        Canny(templ_dx, templ_dy, templ_edge, 30, 60);
        for (int r = 0; r < TW; ++r)
            for (int c = 0; c < TW; ++c)
                if (templ_edge.at<uchar>(r, c) > 0)
                    templ_edge_pts.push_back(Point2f((float)(c - TW/2), (float)(r - TW/2)));
        printf("Template edge points: %d\n", (int)templ_edge_pts.size());
    }

    // Save template image
    {
        Mat templ_vis;
        cvtColor(templ, templ_vis, cv::COLOR_GRAY2BGR);
        imwrite(out_dir + "template.jpg", templ_vis);
    }

    // Test scenes
    struct TestCase {
        string name;
        double noise_sigma;
        int blur_ksize;
        int brightness_delta;
        double contrast;
        int occlusion_radius;
    };

    TestCase cases[] = {
        {"clean",           0,  0,   0, 1.0,  0},
        {"noise_s10",      10,  0,   0, 1.0,  0},
        {"noise_s30",      30,  0,   0, 1.0,  0},
        {"noise_s50",      50,  0,   0, 1.0,  0},
        {"noise_s80",      80,  0,   0, 1.0,  0},
        {"blur_k5",         0,  5,   0, 1.0,  0},
        {"blur_k11",        0, 11,   0, 1.0,  0},
        {"blur_k21",        0, 21,   0, 1.0,  0},
        {"bright_m60",      0,  0, -60, 1.0,  0},
        {"bright_p60",      0,  0,  60, 1.0,  0},
        {"contrast_0.3",    0,  0,   0, 0.3,  0},
        {"contrast_0.5",    0,  0,   0, 0.5,  0},
        {"occlusion_r15",   0,  0,   0, 1.0, 15},
        {"occlusion_r30",   0,  0,   0, 1.0, 30},
        {"noise_blur",     30,  5,   0, 1.0,  0},
        {"noise_contrast", 20,  0,   0, 0.5,  0},
    };

    // 20 objects at various positions and angles
    struct ObjDef { double rx, ry, angle; };
    ObjDef objects[] = {
        {0.10, 0.12,   0}, {0.30, 0.10,  25}, {0.50, 0.08,  50}, {0.70, 0.14,  75}, {0.88, 0.11, 100},
        {0.08, 0.35, 130}, {0.28, 0.32, 155}, {0.48, 0.38, 180}, {0.68, 0.34, 210}, {0.90, 0.36, 240},
        {0.12, 0.58, 270}, {0.32, 0.55, 300}, {0.52, 0.60, 330}, {0.72, 0.57,  15}, {0.88, 0.62,  45},
        {0.10, 0.82,  90}, {0.30, 0.85, 120}, {0.50, 0.80, 170}, {0.72, 0.83, 225}, {0.90, 0.87, 315},
    };
    const int NUM_OBJECTS = 20;

    for (auto& tc : cases) {
        // Create scene with 20 L-shapes at various positions and angles
        Mat scene(H, W, CV_8U, Scalar(50));
        for (int i = 0; i < NUM_OBJECTS; ++i)
            draw_L(scene, (int)(objects[i].rx * W), (int)(objects[i].ry * H),
                   objects[i].angle, 200);

        // Apply degradations
        if (tc.noise_sigma > 0)
            add_noise(scene, tc.noise_sigma);
        if (tc.blur_ksize > 0)
            GaussianBlur(scene, scene, Size(tc.blur_ksize, tc.blur_ksize), 0);
        if (tc.brightness_delta != 0)
            scene.convertTo(scene, -1, 1.0, tc.brightness_delta);
        if (tc.contrast != 1.0) {
            double mean_val = cv::mean(scene)[0];
            scene.convertTo(scene, -1, tc.contrast, mean_val * (1 - tc.contrast));
        }
        if (tc.occlusion_radius > 0) {
            // Occlude 3 objects
            circle(scene, Point((int)(0.10*W), (int)(0.12*H)), tc.occlusion_radius, Scalar(50), -1);
            circle(scene, Point((int)(0.52*W), (int)(0.60*H)), tc.occlusion_radius, Scalar(50), -1);
            circle(scene, Point((int)(0.90*W), (int)(0.87*H)), tc.occlusion_radius, Scalar(50), -1);
        }

        // Match
        Mat padded = pad16(scene);
        auto t0 = chrono::high_resolution_clock::now();
        auto matches = det.match(padded, threshold);
        double ms = chrono::duration<double, std::milli>(
            chrono::high_resolution_clock::now() - t0).count();

        // ICP local refinement on NMS'd matches
        // Reuse scene Sobel (already computed during match preprocessing)
        auto nms_matches = spatial_nms(matches, (float)TW * 0.8f);
        double icp_ms = 0;
        {
            auto icp_t0 = chrono::high_resolution_clock::now();

            // Compute scene Sobel once (in production, cache from match())
            Mat scene_smooth, scene_dx, scene_dy;
            GaussianBlur(scene, scene_smooth, Size(7, 7), 0);
            Sobel(scene_smooth, scene_dx, CV_16S, 1, 0, 3);
            Sobel(scene_smooth, scene_dy, CV_16S, 0, 1, 3);

            auto sobel_done = chrono::high_resolution_clock::now();
            double sobel_ms = chrono::duration<double, std::milli>(
                sobel_done - icp_t0).count();

            icp_refine::ICPConfig icp_cfg;
            icp_cfg.max_iterations = 30;
            icp_cfg.max_dist = 10.0f;

            int refine_n = std::min((int)nms_matches.size(), 50);
            for (int i = 0; i < refine_n; ++i) {
                auto& m = nms_matches[i];
                auto& tmpl_info = det.getTemplates(m.class_id, m.template_id);
                float cx = m.x + tmpl_info[0].width / 2.0f;
                float cy = m.y + tmpl_info[0].height / 2.0f;
                float coarse_angle = m.template_id * 2.0f;

                icp_refine::Pose2D init_pose(cx, cy, coarse_angle);
                auto refined = icp_refine::refineLocal(
                    templ_edge_pts, scene_dx, scene_dy,
                    init_pose, TW, 20, icp_cfg);
                m.refined_angle = refined.angle;
                m.x = (int)(refined.x - tmpl_info[0].width / 2.0f + 0.5f);
                m.y = (int)(refined.y - tmpl_info[0].height / 2.0f + 0.5f);
            }

            icp_ms = chrono::duration<double, std::milli>(
                chrono::high_resolution_clock::now() - icp_t0).count();
            double icp_only_ms = chrono::duration<double, std::milli>(
                chrono::high_resolution_clock::now() - sobel_done).count();
            // Note: sobel_ms could be eliminated by caching from match()
            printf("  (sobel=%.1fms icp=%.1fms)", sobel_ms, icp_only_ms);
        }

        printf("%-20s: %5d matches  match=%.1fms  icp=%.1fms  total=%.1fms",
               tc.name.c_str(), (int)matches.size(), ms, icp_ms, ms + icp_ms);
        if (!nms_matches.empty())
            printf("  best=%.0f", nms_matches[0].similarity);
        printf("\n");

        // Draw refined matches (pass pre-NMS'd + refined matches)
        Mat vis = draw_matches_direct(scene, nms_matches, det, TW, ms);
        imwrite(out_dir + "result_" + tc.name + ".jpg", vis);
    }

    // False positive test: random texture
    {
        RNG rng(123);
        Mat scene(H, W, CV_8U);
        rng.fill(scene, RNG::UNIFORM, 0, 256);
        GaussianBlur(scene, scene, Size(5, 5), 0);
        Mat padded = pad16(scene);
        auto tp0 = chrono::high_resolution_clock::now();
        auto matches = det.match(padded, threshold);
        double tms = chrono::duration<double, std::milli>(
            chrono::high_resolution_clock::now() - tp0).count();
        printf("%-20s: %5d matches  %6.1fms\n", "random_texture", (int)matches.size(), tms);
        Mat vis = draw_matches(scene, matches, det, TW, tms);
        imwrite(out_dir + "result_random_texture.jpg", vis);
    }

    // False positive test: grid pattern
    {
        Mat scene(H, W, CV_8U, Scalar(50));
        for (int y = 0; y < H; y += 20)
            line(scene, Point(0, y), Point(W, y), Scalar(200), 2);
        for (int x = 0; x < W; x += 20)
            line(scene, Point(x, 0), Point(x, H), Scalar(200), 2);
        Mat padded = pad16(scene);
        auto tp0 = chrono::high_resolution_clock::now();
        auto matches = det.match(padded, threshold);
        double tms = chrono::duration<double, std::milli>(
            chrono::high_resolution_clock::now() - tp0).count();
        printf("%-20s: %5d matches  %6.1fms\n", "grid_pattern", (int)matches.size(), tms);
        Mat vis = draw_matches(scene, matches, det, TW, tms);
        imwrite(out_dir + "result_grid_pattern.jpg", vis);
    }

    printf("\nImages saved to %s\n", out_dir.c_str());
    return 0;
}
