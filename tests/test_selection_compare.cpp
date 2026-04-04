/// @file test_selection_compare.cpp
/// @brief Compare old vs new feature selection strategies for ROI refinement.
/// Old: corners by R + edges by det(J^T J), fixed d_min.
/// New: corners by R + Shi-Tomasi + edges by precision-weighted det, grid bucketing.

#include "shape_matcher.h"
#include "roi_refine.h"
#include "icp_refine.h"
#include "test_utils.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace cv;

// OutputGuard from test_utils.h replaces the old CoutSup

// ============================================================
// Constants matching shape_matcher.cpp
// ============================================================
static constexpr float kCornerThreshold    = 0.3f;
static constexpr float kSolverRegularization = 0.001f;
static constexpr int   kDefaultOptPoints   = 8;

// ============================================================
// Helper: build constraint from refine point (mirrors shape_matcher.cpp)
// ============================================================
struct LocalConstraint {
    cv::Point2f src, normal;
    float cornerness;
    float grad_mag;
};

static bool buildLocalConstraint(const sbm::FeatureSet& fs, int rp_idx, LocalConstraint& out) {
    auto& rp = fs.refine_points[rp_idx];
    float tcx = fs.templ_width / 2.0f, tcy = fs.templ_height / 2.0f;
    int tx = (int)(rp.px + tcx + 0.5f), ty = (int)(rp.py + tcy + 0.5f);
    int h = 15;
    if (tx-h<0 || tx+h>=fs.templ_image.cols || ty-h<0 || ty+h>=fs.templ_image.rows)
        h = std::min({tx, ty, fs.templ_image.cols-1-tx, fs.templ_image.rows-1-ty});
    if (h < 5) return false;
    cv::Mat roi = fs.templ_image(cv::Rect(tx-h, ty-h, 2*h, 2*h));
    cv::Mat dx, dy, mag;
    cv::Sobel(roi, dx, CV_32F, 1, 0, 3);
    cv::Sobel(roi, dy, CV_32F, 0, 1, 3);
    cv::magnitude(dx, dy, mag);
    double max_mag; cv::Point max_loc;
    cv::minMaxLoc(mag, nullptr, &max_mag, nullptr, &max_loc);
    float gx = dx.at<float>(max_loc.y, max_loc.x);
    float gy = dy.at<float>(max_loc.y, max_loc.x);
    float gm = std::sqrt(gx*gx + gy*gy);
    if (gm < 1e-6f) return false;
    out.src = cv::Point2f(rp.px, rp.py);
    out.normal = cv::Point2f(gx/gm, gy/gm);
    out.cornerness = rp.cornerness;

    // Also compute average gradient magnitude for precision weighting
    float mag_sum = 0;
    int n_pix = 0;
    for (int r = 0; r < roi.rows; r++) {
        const float* mr = mag.ptr<float>(r);
        for (int c = 0; c < roi.cols; c++) { mag_sum += mr[c]; n_pix++; }
    }
    out.grad_mag = n_pix > 0 ? mag_sum / n_pix : 0;
    return true;
}

// ============================================================
// Candidate struct shared by both methods
// ============================================================
struct Cand {
    int rp_idx;
    float px, py, R;
    float j0, j1, j2;       // Jacobian row for rigid solve
    float cornerness;
    float shi_tomasi;
    float grad_mag;
    bool is_corner;
};

// Collect all candidates from a FeatureSet
static std::vector<Cand> collectCandidates(const sbm::FeatureSet& fs, bool use_shi_tomasi) {
    std::vector<Cand> all;
    float tcx = fs.templ_width / 2.0f, tcy = fs.templ_height / 2.0f;

    cv::Mat templ_blur;
    if (use_shi_tomasi)
        cv::GaussianBlur(fs.templ_image, templ_blur, cv::Size(3, 3), 1.0);
    else
        templ_blur = fs.templ_image;

    for (size_t i = 0; i < fs.refine_points.size(); i++) {
        auto& rp = fs.refine_points[i];
        if (std::abs(rp.px) > fs.templ_width/2.0f - 5 || std::abs(rp.py) > fs.templ_height/2.0f - 5)
            continue;

        LocalConstraint lc;
        if (!buildLocalConstraint(fs, (int)i, lc)) continue;
        float nx = lc.normal.x, ny = lc.normal.y;

        // Structure tensor
        int tx = (int)(rp.px + tcx + 0.5f), ty = (int)(rp.py + tcy + 0.5f);
        int h = 8;
        if (tx-h<0||tx+h>=templ_blur.cols||ty-h<0||ty+h>=templ_blur.rows)
            h = std::min({tx, ty, templ_blur.cols-1-tx, templ_blur.rows-1-ty});
        if (h < 3) continue;

        cv::Mat roi = templ_blur(cv::Rect(tx-h, ty-h, 2*h, 2*h));
        cv::Mat dx, dy;
        cv::Sobel(roi, dx, CV_32F, 1, 0, 3);
        cv::Sobel(roi, dy, CV_32F, 0, 1, 3);

        float m00=0, m01=0, m11=0, mag_sum=0;
        int n_pix = 0;
        for (int r = 0; r < roi.rows; r++) {
            const float* dxr = dx.ptr<float>(r);
            const float* dyr = dy.ptr<float>(r);
            for (int c = 0; c < roi.cols; c++) {
                m00 += dxr[c]*dxr[c]; m01 += dxr[c]*dyr[c]; m11 += dyr[c]*dyr[c];
                mag_sum += std::sqrt(dxr[c]*dxr[c] + dyr[c]*dyr[c]);
                n_pix++;
            }
        }
        float trace = m00 + m11;
        float disc = std::sqrt(std::max(0.0f, (m00-m11)*(m00-m11)/4.0f + m01*m01));
        float lam2 = trace/2.0f - disc;

        Cand cd;
        cd.rp_idx = (int)i;
        cd.px = rp.px; cd.py = rp.py;
        cd.R = std::sqrt(rp.px*rp.px + rp.py*rp.py);
        cd.j0 = -rp.py * nx + rp.px * ny;
        cd.j1 = nx; cd.j2 = ny;
        cd.cornerness = rp.cornerness;
        cd.shi_tomasi = std::max(0.0f, lam2);
        cd.grad_mag = n_pix > 0 ? mag_sum / n_pix : 0;

        if (use_shi_tomasi)
            cd.is_corner = (rp.cornerness > kCornerThreshold && cd.shi_tomasi > 5.0f);
        else
            cd.is_corner = (rp.cornerness > kCornerThreshold);

        all.push_back(cd);
    }
    return all;
}

// ============================================================
// OLD selection: corners by R + edges by det(J^T J), fixed d_min
// ============================================================
static std::vector<cv::Point2f> selectOld(const sbm::FeatureSet& fs, int max_points) {
    auto all = collectCandidates(fs, false);

    std::vector<Cand> corners, edges;
    float max_R = 0;
    for (auto& c : all) {
        max_R = std::max(max_R, c.R);
        if (c.is_corner) corners.push_back(c);
        else edges.push_back(c);
    }

    // Fixed d_min as in old code
    float d_min = std::max(fs.templ_width, fs.templ_height) / 16.0f * 1.5f;

    std::vector<int> selected;
    std::vector<cv::Point2f> sel_pts;

    auto tooClose = [&](float px, float py) -> bool {
        for (auto& p : sel_pts) {
            float dx = px - p.x, dy = py - p.y;
            if (std::sqrt(dx*dx + dy*dy) < d_min) return true;
        }
        return false;
    };

    // Sort corners by cornerness*10 + R/max_R (old scoring)
    if (max_R < 1e-6f) max_R = 1.0f;
    std::sort(corners.begin(), corners.end(), [&](const Cand& a, const Cand& b) {
        float sa = a.cornerness * 10.0f + a.R / max_R;
        float sb = b.cornerness * 10.0f + b.R / max_R;
        return sa > sb;
    });

    for (auto& c : corners) {
        if ((int)selected.size() >= max_points) break;
        if (tooClose(c.px, c.py)) continue;
        selected.push_back(c.rp_idx);
        sel_pts.push_back(cv::Point2f(c.px, c.py));
    }

    // Phase 2: edges via unweighted det(J^T J) greedy, no grid
    float Iw[3][3] = {};
    for (int idx : selected) {
        LocalConstraint lc;
        if (!buildLocalConstraint(fs, idx, lc)) continue;
        float nx = lc.normal.x, ny = lc.normal.y;
        float px = fs.refine_points[idx].px, py = fs.refine_points[idx].py;
        float j0 = -py*nx + px*ny, j1 = nx, j2 = ny;
        Iw[0][0]+=j0*j0; Iw[0][1]+=j0*j1; Iw[0][2]+=j0*j2;
        Iw[1][1]+=j1*j1; Iw[1][2]+=j1*j2;
        Iw[2][2]+=j2*j2;
    }
    Iw[1][0]=Iw[0][1]; Iw[2][0]=Iw[0][2]; Iw[2][1]=Iw[1][2];
    for (int i=0;i<3;i++) Iw[i][i] += kSolverRegularization;

    auto det3 = [](float A[3][3]) -> float {
        return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
             - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
             + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    };

    while ((int)selected.size() < max_points) {
        float best_det = -1e30f;
        int best_ei = -1;

        for (int ei = 0; ei < (int)edges.size(); ei++) {
            auto& e = edges[ei];
            bool used = false;
            for (int si : selected) if (si == e.rp_idx) { used = true; break; }
            if (used) continue;
            if (tooClose(e.px, e.py)) continue;

            // Unweighted trial
            float trial[3][3];
            for (int r=0;r<3;r++) for (int c=0;c<3;c++) trial[r][c] = Iw[r][c];
            trial[0][0]+=e.j0*e.j0; trial[0][1]+=e.j0*e.j1; trial[0][2]+=e.j0*e.j2;
            trial[1][1]+=e.j1*e.j1; trial[1][2]+=e.j1*e.j2;
            trial[2][2]+=e.j2*e.j2;
            trial[1][0]=trial[0][1]; trial[2][0]=trial[0][2]; trial[2][1]=trial[1][2];

            float d = det3(trial);
            if (d > best_det) { best_det = d; best_ei = ei; }
        }

        if (best_ei < 0) break;
        auto& e = edges[best_ei];
        selected.push_back(e.rp_idx);
        sel_pts.push_back(cv::Point2f(e.px, e.py));

        Iw[0][0]+=e.j0*e.j0; Iw[0][1]+=e.j0*e.j1; Iw[0][2]+=e.j0*e.j2;
        Iw[1][1]+=e.j1*e.j1; Iw[1][2]+=e.j1*e.j2;
        Iw[2][2]+=e.j2*e.j2;
        Iw[1][0]=Iw[0][1]; Iw[2][0]=Iw[0][2]; Iw[2][1]=Iw[1][2];
    }

    return sel_pts;
}

// ============================================================
// NEW selection: corners by R + Shi-Tomasi + edges by precision-weighted det, grid bucketing
// (mirrors the current selectOptimizedPoints in shape_matcher.cpp)
// ============================================================
static std::vector<cv::Point2f> selectNew(const sbm::FeatureSet& fs, int max_points) {
    auto all = collectCandidates(fs, true);

    float tcx = fs.templ_width / 2.0f, tcy = fs.templ_height / 2.0f;

    // Grid bucketing
    static const int kGridK = 5;
    static const int kMaxPerCell = 2;
    float cell_w = fs.templ_width / (float)kGridK;
    float cell_h = fs.templ_height / (float)kGridK;
    int grid_count[kGridK][kGridK] = {};

    auto gridCell = [&](float px, float py, int& gx, int& gy) {
        gx = std::max(0, std::min(kGridK-1, (int)((px + tcx) / cell_w)));
        gy = std::max(0, std::min(kGridK-1, (int)((py + tcy) / cell_h)));
    };
    auto gridFull = [&](float px, float py) -> bool {
        int gx, gy; gridCell(px, py, gx, gy);
        return grid_count[gy][gx] >= kMaxPerCell;
    };
    auto gridAdd = [&](float px, float py) {
        int gx, gy; gridCell(px, py, gx, gy);
        grid_count[gy][gx]++;
    };

    std::vector<Cand> corners, edges;
    for (auto& c : all) {
        if (c.is_corner) corners.push_back(c);
        else edges.push_back(c);
    }

    std::vector<int> selected;

    // Phase 1: corners sorted by R (leverage)
    std::sort(corners.begin(), corners.end(),
              [](const Cand& a, const Cand& b) { return a.R > b.R; });

    for (auto& c : corners) {
        if ((int)selected.size() >= max_points) break;
        if (gridFull(c.px, c.py)) continue;
        selected.push_back(c.rp_idx);
        gridAdd(c.px, c.py);
    }

    // Phase 2: edges via precision-weighted D-optimal
    float Iw[3][3] = {};
    for (int idx : selected) {
        LocalConstraint lc;
        if (!buildLocalConstraint(fs, idx, lc)) continue;
        float nx = lc.normal.x, ny = lc.normal.y;
        float px = fs.refine_points[idx].px, py = fs.refine_points[idx].py;
        float j0 = -py*nx + px*ny, j1 = nx, j2 = ny;
        Iw[0][0]+=j0*j0; Iw[0][1]+=j0*j1; Iw[0][2]+=j0*j2;
        Iw[1][1]+=j1*j1; Iw[1][2]+=j1*j2;
        Iw[2][2]+=j2*j2;
    }
    Iw[1][0]=Iw[0][1]; Iw[2][0]=Iw[0][2]; Iw[2][1]=Iw[1][2];
    for (int i=0;i<3;i++) Iw[i][i] += kSolverRegularization;

    auto det3 = [](float A[3][3]) -> float {
        return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
             - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
             + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    };

    while ((int)selected.size() < max_points) {
        float best_det = -1e30f;
        int best_ei = -1;

        for (int ei = 0; ei < (int)edges.size(); ei++) {
            auto& e = edges[ei];
            bool used = false;
            for (int si : selected) if (si == e.rp_idx) { used = true; break; }
            if (used) continue;
            if (gridFull(e.px, e.py)) continue;

            float g = std::max(1.0f, e.grad_mag);
            float trial[3][3];
            for (int r=0;r<3;r++) for (int c=0;c<3;c++) trial[r][c] = Iw[r][c];
            trial[0][0]+=g*e.j0*e.j0; trial[0][1]+=g*e.j0*e.j1; trial[0][2]+=g*e.j0*e.j2;
            trial[1][1]+=g*e.j1*e.j1; trial[1][2]+=g*e.j1*e.j2;
            trial[2][2]+=g*e.j2*e.j2;
            trial[1][0]=trial[0][1]; trial[2][0]=trial[0][2]; trial[2][1]=trial[1][2];

            float d = det3(trial);
            if (d > best_det) { best_det = d; best_ei = ei; }
        }

        if (best_ei < 0) break;
        auto& e = edges[best_ei];
        selected.push_back(e.rp_idx);
        gridAdd(e.px, e.py);

        float g = std::max(1.0f, e.grad_mag);
        Iw[0][0]+=g*e.j0*e.j0; Iw[0][1]+=g*e.j0*e.j1; Iw[0][2]+=g*e.j0*e.j2;
        Iw[1][1]+=g*e.j1*e.j1; Iw[1][2]+=g*e.j1*e.j2;
        Iw[2][2]+=g*e.j2*e.j2;
        Iw[1][0]=Iw[0][1]; Iw[2][0]=Iw[0][2]; Iw[2][1]=Iw[1][2];
    }

    std::vector<cv::Point2f> result;
    for (int idx : selected)
        result.push_back(cv::Point2f(fs.refine_points[idx].px, fs.refine_points[idx].py));
    return result;
}

// ============================================================
// Draw L-shape
// ============================================================
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

// ============================================================
// Create template shapes
// ============================================================
struct ShapeInfo {
    const char* name;
    Mat templ;
    Point2f origin;
};

static std::vector<ShapeInfo> createShapes() {
    std::vector<ShapeInfo> shapes;

    // 1. Large L-shape 300x300 (more features, more room to differentiate)
    {
        Mat t(300, 300, CV_8U, Scalar(0));
        draw_L(t, 150, 150, 0, 200);
        shapes.push_back({"L-300", t, Point2f(150, 150)});
    }

    // 2. T-shape 300x300 (asymmetric, 3 corners, 2 edge directions)
    {
        Mat t(300, 300, CV_8U, Scalar(0));
        // Vertical stem
        rectangle(t, Point(130, 80), Point(170, 250), Scalar(200), -1);
        // Horizontal top bar
        rectangle(t, Point(50, 80), Point(250, 120), Scalar(200), -1);
        shapes.push_back({"T-shape", t, Point2f(150, 150)});
    }

    // 3. Wrench shape 400x200 (highly asymmetric — texture at head, plain shaft)
    {
        Mat t(200, 400, CV_8U, Scalar(0));
        // Shaft
        rectangle(t, Point(60, 85), Point(300, 115), Scalar(200), -1);
        // Head (hexagonal opening approximated by circles)
        circle(t, Point(350, 100), 50, Scalar(200), -1);
        circle(t, Point(350, 100), 25, Scalar(0), -1);  // hole
        // Handle end
        circle(t, Point(40, 100), 20, Scalar(200), -1);
        shapes.push_back({"Wrench", t, Point2f(200, 100)});
    }

    // 4. Arrow 300x300 (asymmetric, pointed, 1 strong corner + edges)
    {
        Mat t(300, 300, CV_8U, Scalar(0));
        // Arrow body
        Point body[4] = {Point(50,130), Point(200,130), Point(200,100), Point(50,100)};
        fillConvexPoly(t, body, 4, Scalar(200));
        // Arrow head
        Point head[3] = {Point(200,80), Point(280,115), Point(200,150)};
        fillConvexPoly(t, head, 3, Scalar(200));
        shapes.push_back({"Arrow", t, Point2f(150, 115)});
    }

    // 5. Cross/Plus 300x300 (4-fold symmetry but distinct features at each arm end)
    {
        Mat t(300, 300, CV_8U, Scalar(0));
        rectangle(t, Point(120, 40), Point(180, 260), Scalar(200), -1);  // vertical
        rectangle(t, Point(40, 120), Point(260, 180), Scalar(200), -1);  // horizontal
        shapes.push_back({"Cross", t, Point2f(150, 150)});
    }

    // 6. F-shape 300x300 (like L but with extra arm — fully asymmetric)
    {
        Mat t(300, 300, CV_8U, Scalar(0));
        // Vertical bar
        rectangle(t, Point(60, 40), Point(100, 260), Scalar(200), -1);
        // Top horizontal bar
        rectangle(t, Point(100, 40), Point(240, 80), Scalar(200), -1);
        // Middle horizontal bar (shorter)
        rectangle(t, Point(100, 130), Point(200, 165), Scalar(200), -1);
        shapes.push_back({"F-shape", t, Point2f(150, 150)});
    }

    // 7. Narrow triangle 300x300 (10-170 degrees, nearly degenerate)
    {
        Mat t(300, 300, CV_8U, Scalar(0));
        Point pts[3] = {Point(30, 250), Point(270, 250), Point(150, 50)};
        fillConvexPoly(t, pts, 3, Scalar(200));
        shapes.push_back({"NarrowTri", t, Point2f(150, 150)});
    }

    return shapes;
}

// ============================================================
// Noise / blur conditions
// ============================================================
struct Condition {
    const char* name;
    int noise_sigma;
    int blur_k;
};

static Mat applyCondition(const Mat& scene, const Condition& cond, RNG& rng) {
    Mat result = scene.clone();
    if (cond.noise_sigma > 0) {
        Mat noise(result.size(), CV_64F);
        rng.fill(noise, RNG::NORMAL, 0, cond.noise_sigma);
        Mat tmp; result.convertTo(tmp, CV_64F);
        tmp += noise; tmp.convertTo(result, CV_8U);
    }
    if (cond.blur_k > 1) {
        GaussianBlur(result, result, Size(cond.blur_k, cond.blur_k), 0);
    }
    return result;
}

// ============================================================
// Angle wrapping helper
// ============================================================
static float angleDiff(float a, float b) {
    float d = std::fmod(a - b, 360.0f);
    if (d > 180.0f) d -= 360.0f;
    if (d < -180.0f) d += 360.0f;
    return std::abs(d);
}

// ============================================================
// Results struct
// ============================================================
struct MethodResult {
    float mean_angle_err_icp;
    float mean_pos_err_icp;
    float mean_angle_err_roi;
    float mean_pos_err_roi;
    float worst_angle_err_icp;
    float worst_angle_err_roi;
    float selection_time_us;
    int n_detected;
    int n_total;
};

// ============================================================
// Run one test case: given a shape + condition + selection method
// sweep 36 angles and measure accuracy
// ============================================================
static MethodResult runTest(
    const ShapeInfo& shape,
    const Condition& cond,
    const std::vector<cv::Point2f>& opt_points,
    const char* method_name)
{
    MethodResult mr = {};
    mr.n_total = 36;

    float sum_ae_icp = 0, sum_pe_icp = 0;
    float sum_ae_roi = 0, sum_pe_roi = 0;
    float worst_ae_icp = 0, worst_ae_roi = 0;
    int detected_icp = 0, detected_roi = 0;

    // Extract features once
    auto features = sbm::extractFeatures(shape.templ);
    features.setOrigin(shape.origin.x, shape.origin.y);

    for (int ai = 0; ai < 36; ai++) {
        float gt_angle = ai * 10.0f;

        // Create scene with one object at center
        int scene_sz = std::max(shape.templ.cols, shape.templ.rows) * 3;
        scene_sz = std::max(scene_sz, 500);
        Mat scene(scene_sz, scene_sz, CV_8U, Scalar(30));
        float gt_x = scene_sz / 2.0f, gt_y = scene_sz / 2.0f;

        // Place rotated template
        Mat M = getRotationMatrix2D(Point2f((float)shape.templ.cols/2, (float)shape.templ.rows/2),
                                     -gt_angle, 1.0);
        Mat rot;
        warpAffine(shape.templ, rot, M, shape.templ.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
        int ox = (int)gt_x - shape.templ.cols/2;
        int oy = (int)gt_y - shape.templ.rows/2;
        for (int r = 0; r < rot.rows; r++)
            for (int c = 0; c < rot.cols; c++) {
                int sy = oy + r, sx = ox + c;
                if (sy>=0 && sy<scene.rows && sx>=0 && sx<scene.cols && rot.at<uchar>(r,c) > 0)
                    scene.at<uchar>(sy, sx) = rot.at<uchar>(r, c);
            }

        // Apply noise/blur
        RNG rng(42 + ai);
        Mat test_scene = applyCondition(scene, cond, rng);

        // --- ICP refinement ---
        {
            auto feat_icp = features;
            sbm::MatchConfig cfg;
            cfg.min_score = 30;
            cfg.refine = sbm::RefineMode::ICP;
            cfg.icp_iterations = 30;

            sbm::ShapeMatcher matcher(cfg);
            sbm::ModelConfig mcfg;
            mcfg.angle = {0, 360, 2};

            {
                OutputGuard guard;
                matcher.addModel("test", feat_icp, mcfg);
            }

            std::vector<sbm::MatchResult> results;
            { OutputGuard guard; results = matcher.match(test_scene); }

            if (!results.empty()) {
                auto& best = results[0];
                float ae = angleDiff(best.angle, gt_angle);
                float pe = std::sqrt((best.x-gt_x)*(best.x-gt_x) + (best.y-gt_y)*(best.y-gt_y));
                sum_ae_icp += ae;
                sum_pe_icp += pe;
                worst_ae_icp = std::max(worst_ae_icp, ae);
                detected_icp++;
            }
        }

        // --- ROI refinement with given opt_points ---
        {
            auto feat_roi = features;
            // Override cached_opt_points
            feat_roi.cached_opt_points = opt_points;
            feat_roi.cached_opt_max_points = kDefaultOptPoints;

            sbm::MatchConfig cfg;
            cfg.min_score = 30;
            cfg.refine = sbm::RefineMode::ROI;

            sbm::ShapeMatcher matcher(cfg);
            sbm::ModelConfig mcfg;
            mcfg.angle = {0, 360, 2};

            {
                OutputGuard guard;
                matcher.addModel("test", feat_roi, mcfg);
            }

            std::vector<sbm::MatchResult> results;
            { OutputGuard guard; results = matcher.match(test_scene); }

            if (!results.empty()) {
                auto& best = results[0];
                float ae = angleDiff(best.angle, gt_angle);
                float pe = std::sqrt((best.x-gt_x)*(best.x-gt_x) + (best.y-gt_y)*(best.y-gt_y));
                sum_ae_roi += ae;
                sum_pe_roi += pe;
                worst_ae_roi = std::max(worst_ae_roi, ae);
                detected_roi++;
            }
        }
    }

    mr.n_detected = detected_roi;  // ROI detection count (should == ICP usually)
    mr.mean_angle_err_icp = detected_icp > 0 ? sum_ae_icp / detected_icp : 999.0f;
    mr.mean_pos_err_icp   = detected_icp > 0 ? sum_pe_icp / detected_icp : 999.0f;
    mr.mean_angle_err_roi = detected_roi > 0 ? sum_ae_roi / detected_roi : 999.0f;
    mr.mean_pos_err_roi   = detected_roi > 0 ? sum_pe_roi / detected_roi : 999.0f;
    mr.worst_angle_err_icp = worst_ae_icp;
    mr.worst_angle_err_roi = worst_ae_roi;
    return mr;
}

// ============================================================
// Main
// ============================================================
int main() {
    // Ensure output directory exists
    system("if not exist output mkdir output");

    auto shapes = createShapes();
    Condition conditions[] = {
        {"Clean",       0,  0},
        {"Noise s=20", 20,  0},
        {"Noise s=40", 40,  0},
        {"Blur k=11",   0, 11},
    };
    int n_cond = sizeof(conditions) / sizeof(conditions[0]);

    // Open output file
    FILE* fout = fopen("output/selection_compare.txt", "w");
    if (!fout) { fprintf(stderr, "Cannot open output file\n"); return 1; }

    // Print header
    const char* hdr =
        "Shape           | Condition   | Method | ROI MeanAng | ROI MeanPos | ROI WorstAng | ICP MeanAng | ICP MeanPos | ICP WorstAng | SelectTime | Det\n"
        "----------------|-------------|--------|-------------|-------------|--------------|-------------|-------------|--------------|------------|----\n";
    printf("%s", hdr);
    fprintf(fout, "%s", hdr);

    for (auto& shape : shapes) {
        printf("\n=== Shape: %s ===\n", shape.name);

        // Extract features + measure selection times
        auto features = sbm::extractFeatures(shape.templ);
        features.setOrigin(shape.origin.x, shape.origin.y);

        // Time OLD selection
        auto t0 = std::chrono::high_resolution_clock::now();
        auto pts_old = selectOld(features, kDefaultOptPoints);
        auto t1 = std::chrono::high_resolution_clock::now();
        float old_time_us = (float)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        // Time NEW selection
        t0 = std::chrono::high_resolution_clock::now();
        auto pts_new = selectNew(features, kDefaultOptPoints);
        t1 = std::chrono::high_resolution_clock::now();
        float new_time_us = (float)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        printf("  Old: %d pts (%.0f us), New: %d pts (%.0f us)\n",
               (int)pts_old.size(), old_time_us, (int)pts_new.size(), new_time_us);

        for (int ci = 0; ci < n_cond; ci++) {
            auto& cond = conditions[ci];
            printf("  Condition: %s ... ", cond.name);
            fflush(stdout);

            // Run old method
            auto mr_old = runTest(shape, cond, pts_old, "OLD");
            mr_old.selection_time_us = old_time_us;

            // Run new method
            auto mr_new = runTest(shape, cond, pts_new, "NEW");
            mr_new.selection_time_us = new_time_us;

            // Format and print
            auto printRow = [&](const char* method, const MethodResult& mr) {
                char line[256];
                snprintf(line, sizeof(line),
                    "%-15s | %-11s | %-6s | %11.3f | %11.3f | %12.3f | %11.3f | %11.3f | %12.3f | %8.0f us | %d/%d\n",
                    shape.name, cond.name, method,
                    mr.mean_angle_err_roi, mr.mean_pos_err_roi, mr.worst_angle_err_roi,
                    mr.mean_angle_err_icp, mr.mean_pos_err_icp, mr.worst_angle_err_icp,
                    mr.selection_time_us, mr.n_detected, mr.n_total);
                printf("%s", line);
                fprintf(fout, "%s", line);
            };

            printRow("OLD", mr_old);
            printRow("NEW", mr_new);
        }
    }

    // Summary comparison
    fprintf(fout, "\n\nNotes:\n");
    fprintf(fout, "- OLD: corners by cornerness*10 + R/maxR, edges by unweighted det(J^T J), fixed d_min\n");
    fprintf(fout, "- NEW: corners by R + Shi-Tomasi filter, edges by gradient-weighted det(J^T J), grid bucketing\n");
    fprintf(fout, "- Angles swept: 0, 10, 20, ..., 350 (36 total)\n");
    fprintf(fout, "- ICP columns are independent of selection method (same for OLD/NEW, serves as reference)\n");
    fprintf(fout, "- ROI columns show the effect of different feature selection on ROI refinement\n");

    fclose(fout);
    printf("\nResults saved to output/selection_compare.txt\n");
    return 0;
}
