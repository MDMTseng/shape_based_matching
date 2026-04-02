// Multi-resolution matching: coarse match at reduced scale, ICP at full scale.

#include "line2Dup.h"
#include "icp_refine.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdio>

using namespace cv;
using namespace std;
using Clock = chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
    return chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

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

static Mat pad16(const Mat& img) {
    int pw = (img.cols + 15) & ~15, ph = (img.rows + 15) & ~15;
    if (pw != img.cols || ph != img.rows) {
        Mat p; copyMakeBorder(img, p, 0, ph-img.rows, 0, pw-img.cols, BORDER_CONSTANT, Scalar(0));
        return p;
    }
    return img;
}

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

int main() {
    fprintf(stderr, "================================================================\n");
    fprintf(stderr, "  Multi-Resolution Matching: coarse at reduced scale + ICP\n");
    fprintf(stderr, "================================================================\n\n");

    const int TW = 80;
    const float angle_step = 2.0f;
    const float threshold = 50.0f;

    // Full-res template + edges for ICP
    Mat templ(TW, TW, CV_8U, Scalar(0));
    draw_L(templ, TW/2, TW/2, 0, 200);
    Mat mask_t = Mat::ones(TW, TW, CV_8U) * 255;
    auto model_edges = icp_refine::extractModelEdges(templ);

    // Test configs
    struct Config {
        const char* name;
        float scale;  // matching scale (1.0 = full, 0.3 = 30%)
        int W, H;     // scene size
    };
    Config configs[] = {
        {"FHD full",   1.0f, 1920, 1080},
        {"FHD 50%",    0.5f, 1920, 1080},
        {"FHD 30%",    0.3f, 1920, 1080},
        {"30MP full",  1.0f, 6000, 5000},
        {"30MP 50%",   0.5f, 6000, 5000},
        {"30MP 30%",   0.3f, 6000, 5000},
    };

    for (auto& cfg : configs) {
        int W = cfg.W, H = cfg.H;
        float scale = cfg.scale;
        int sW = (int)(W * scale), sH = (int)(H * scale);
        int sTW = std::max(16, (int)(TW * scale));

        // Train at full resolution, then scale features for matching
        // addRotatedTemplates extracts features once, rotates coordinates.
        // We then scale all feature coordinates by the match scale factor.
        line2Dup::Detector det(128, {4, 8}, 30, 60);

        // Train at FULL resolution, then scale feature coordinates.
        // This avoids "too few features" at small template sizes.
        // addRotatedTemplates extracts features once at 0 deg, rotates coords.
        det.addRotatedTemplates(templ, mask_t, "L", 0, 360, angle_step);

        // Scale all template features by the match scale factor
        if (scale < 1.0f) {
            auto& tps = det.getClassTemplates("L");
            for (auto& tp : tps) {
                for (auto& t : tp) {
                    t.tl_x = (int)(t.tl_x * scale + 0.5f);
                    t.tl_y = (int)(t.tl_y * scale + 0.5f);
                    t.width = (int)(t.width * scale + 0.5f);
                    t.height = (int)(t.height * scale + 0.5f);
                    for (auto& f : t.features) {
                        f.x = (int)(f.x * scale + 0.5f);
                        f.y = (int)(f.y * scale + 0.5f);
                    }
                }
            }
        }

        // Create scene with 3 objects
        Mat scene(H, W, CV_8U, Scalar(50));
        draw_L(scene, W/4, H/3, 30, 200);
        draw_L(scene, W/2, H/2, 120, 200);
        draw_L(scene, 3*W/4, 2*H/3, 250, 200);

        // === Method A: Full-res matching only ===
        double time_full = 0;
        int matches_full = 0;
        if (scale == 1.0f) {
            auto t0 = Clock::now();
            auto m = det.match(pad16(scene), threshold);
            time_full = ms_since(t0);
            matches_full = (int)m.size();
        }

        // === Method B: Scaled matching + full-res ICP ===
        auto t_total = Clock::now();

        // Step 1: Resize scene
        auto t0 = Clock::now();
        Mat small_scene;
        if (scale < 1.0f)
            resize(scene, small_scene, Size(sW, sH));
        else
            small_scene = scene;
        double t_resize = ms_since(t0);

        // Step 2: Match at reduced scale
        t0 = Clock::now();
        auto matches = det.match(pad16(small_scene), threshold);
        double t_match = ms_since(t0);

        // Step 3: NMS + scale positions back
        auto nms = spatial_nms(matches, (float)sTW * 0.8f);
        for (auto& m : nms) {
            m.x = (int)(m.x / scale);
            m.y = (int)(m.y / scale);
        }

        // Step 4: ICP at full resolution using LOCAL ROI Sobel
        // No full-scene Sobel — compute Sobel only in patches around each match
        t0 = Clock::now();
        icp_refine::ICPConfig icp_cfg;
        icp_cfg.max_iterations = 30;
        icp_cfg.max_dist = 15.0f;

        int refine_n = std::min((int)nms.size(), 50);
        for (int i = 0; i < refine_n; ++i) {
            auto& m = nms[i];
            auto& ti = det.getTemplates(m.class_id, m.template_id);
            float icx = m.x + (TW/2.0f - ti[0].tl_x / scale);
            float icy = m.y + (TW/2.0f - ti[0].tl_y / scale);
            float coarse_angle = m.template_id * angle_step;

            // Local ROI Sobel: crop patch, blur+sobel only that patch
            int margin = TW/2 + 30;
            int rx = std::max(0, (int)(icx - margin));
            int ry = std::max(0, (int)(icy - margin));
            int rw = std::min(W - rx, 2 * margin);
            int rh = std::min(H - ry, 2 * margin);
            if (rw <= 10 || rh <= 10) continue;

            Mat roi = scene(Rect(rx, ry, rw, rh));
            Mat roi_smooth, roi_dx, roi_dy;
            GaussianBlur(roi, roi_smooth, Size(7, 7), 0);
            Sobel(roi_smooth, roi_dx, CV_16S, 1, 0, 3);
            Sobel(roi_smooth, roi_dy, CV_16S, 0, 1, 3);

            // ICP in local coords, then shift back
            icp_refine::Pose2D init_pose(icx - rx, icy - ry, coarse_angle);
            auto refined = icp_refine::refineWithNormals(
                model_edges, roi_dx, roi_dy, init_pose, TW, 20, icp_cfg);
            m.refined_angle = refined.angle;
            m.x = (int)(refined.x + rx + 0.5f);
            m.y = (int)(refined.y + ry + 0.5f);
        }
        double t_icp = ms_since(t0);
        double t_total_ms = ms_since(t_total);

        fprintf(stderr, "%-12s: resize=%.1f match=%.1f icp=%.1f total=%.1fms  (raw=%d nms=%d)",
                cfg.name, t_resize, t_match, t_icp, t_total_ms,
                (int)matches.size(), (int)nms.size());
        if (scale == 1.0f)
            fprintf(stderr, "  [full-only: %.1fms]", time_full);
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "\n");
    return 0;
}
