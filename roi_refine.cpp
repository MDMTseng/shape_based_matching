/// @file roi_refine.cpp
/// @brief ROI-based pose refinement implementation.

#include "roi_refine.h"
#include "sbm_log.h"
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>   // getRotationMatrix2D moved here in OpenCV 5
#endif
#include <algorithm>
#include <cmath>

namespace roi_refine {

// Named constants (extracted from inline magic numbers)
static constexpr float kPCAGradientThreshFactor = 0.3f;   // PCA gradient threshold multiplier
static constexpr float kAngleRematchDeg         = 2.0f;   // angle change threshold for re-matching
static constexpr float kAngleRewarpDeg          = 5.0f;   // angle change threshold for re-warping cached ROIs
static constexpr int   kMinROIHalf              = 5;       // minimum ROI half-size
static constexpr float kOutlierMultiplier       = 2.0f;   // outlier rejection: distance > N × median
static constexpr float kMaxThetaUpdate          = 0.2f;   // theta clamp (radians)
static constexpr float kMaxTransUpdate          = 10.0f;  // translation clamp (pixels)
static constexpr float kSolverRegularization    = 0.001f; // regularization for ATA diagonal
static constexpr float kEpsilon                 = 1e-6f;  // denominator checks

// -----------------------------------------------------------------------
// Select critical points: corners first, then well-spaced edges
// -----------------------------------------------------------------------
std::vector<SamplePoint> selectCriticalPoints(
    const std::vector<cv::Point2f>& positions,
    const std::vector<float>& cornerness,
    int max_points, int templ_width, int templ_height) {

    struct Candidate {
        cv::Point2f pos;
        float cornerness;
        int idx;
    };

    std::vector<Candidate> candidates(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        candidates[i] = {positions[i], cornerness[i], (int)i};
    }

    // Sort by cornerness descending (corners first)
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.cornerness > b.cornerness;
              });

    // Greedy selection: pick points that are well-spaced
    float min_dist = std::max(templ_width, templ_height) / (float)(max_points + 1) * 1.5f;
    float min_dist_sq = min_dist * min_dist;

    std::vector<SamplePoint> selected;
    for (auto& c : candidates) {
        if ((int)selected.size() >= max_points) break;

        // Check distance from already selected points
        bool too_close = false;
        for (auto& s : selected) {
            float dx = c.pos.x - s.pos.x, dy = c.pos.y - s.pos.y;
            if (dx*dx + dy*dy < min_dist_sq) { too_close = true; break; }
        }
        if (too_close) continue;

        // Skip points too close to template border (ROI would be clipped)
        float half_w = templ_width / 2.0f, half_h = templ_height / 2.0f;
        float margin = 5;  // small margin, just enough for ROI extraction
        if (std::abs(c.pos.x) > half_w - margin || std::abs(c.pos.y) > half_h - margin)
            continue;

        SamplePoint sp;
        sp.pos = c.pos;
        selected.push_back(sp);
    }

    return selected;
}

// -----------------------------------------------------------------------
// Bilinear intensity sample (CV_8U)
// -----------------------------------------------------------------------
static inline float bilinearSample(const cv::Mat& img, float x, float y) {
    if (x < 0 || y < 0 || x >= img.cols - 1 || y >= img.rows - 1) return 0.0f;
    int x0 = (int)x, y0 = (int)y; float fx = x - x0, fy = y - y0;
    const uchar* r0 = img.ptr<uchar>(y0);
    const uchar* r1 = img.ptr<uchar>(y0 + 1);
    float a = r0[x0]*(1-fx) + r0[x0+1]*fx;
    float b = r1[x0]*(1-fx) + r1[x0+1]*fx;
    return a*(1-fy) + b*fy;
}

// -----------------------------------------------------------------------
// 1D edge match: sample the template intensity profile ACROSS the edge (along the
// normal) and slide it over the scene profile along the same (rotated) normal.
// Returns the matched scene point (= expected + delta * normal_scene). Only the
// across-edge displacement is measured — the along-edge slide is left unconstrained,
// which is exactly what an edge should contribute (1D constraint).
// -----------------------------------------------------------------------
static cv::Point2f match1D_alongNormal(const cv::Mat& templ_img, const cv::Mat& scene_img,
                                       cv::Point2f templ_pt, cv::Point2f n_templ,
                                       cv::Point2f sc_center, cv::Point2f n_scene,
                                       int half_len, int search_half, float* out_score,
                                       cv::Point2f t_templ = cv::Point2f(0,0),
                                       cv::Point2f t_scene = cv::Point2f(0,0),
                                       int tangent_half = 0) {
    if (out_score) *out_score = -1.0f;
    int M = half_len;            // template profile half-length
    int S = half_len + search_half;
    int W = tangent_half;        // average each profile sample over +-W along the edge
    float inv = 1.0f / (2*W + 1);
    std::vector<float> pt(2*M+1), ps(2*S+1);
    // Each profile sample is averaged over a band of width (2W+1) along the tangent
    // (edge) direction -> denoises without blurring the across-edge step.
    for (int k = -M; k <= M; k++) {
        float bx = templ_pt.x + k*n_templ.x, by = templ_pt.y + k*n_templ.y;
        float s = 0;
        for (int j = -W; j <= W; j++) s += bilinearSample(templ_img, bx + j*t_templ.x, by + j*t_templ.y);
        pt[k+M] = s * inv;
    }
    for (int k = -S; k <= S; k++) {
        float bx = sc_center.x + k*n_scene.x, by = sc_center.y + k*n_scene.y;
        float s = 0;
        for (int j = -W; j <= W; j++) s += bilinearSample(scene_img, bx + j*t_scene.x, by + j*t_scene.y);
        ps[k+S] = s * inv;
    }

    std::vector<float> resp(2*search_half+1, -2.0f);
    float best = -2.0f; int bestd = 0;
    for (int d = -search_half; d <= search_half; d++) {
        double dot = 0, e2 = 0, s2 = 0;
        for (int k = -M; k <= M; k++) { float a = pt[k+M], b = ps[k+d+S]; dot += a*b; e2 += a*a; s2 += b*b; }
        float ncc = (e2 > 1e-6 && s2 > 1e-6) ? (float)(dot / std::sqrt(e2*s2)) : -2.0f;
        resp[d+search_half] = ncc;
        if (ncc > best) { best = ncc; bestd = d; }
    }
    // Parabolic subpixel on the 1D response
    float dd = (float)bestd; int bi = bestd + search_half;
    if (bi > 0 && bi < (int)resp.size() - 1) {
        float a = resp[bi-1], b = resp[bi], c = resp[bi+1], den = a - 2*b + c;
        if (std::abs(den) > kEpsilon) dd += std::max(-1.0f, std::min(1.0f, 0.5f*(a-c)/den));
    }
    if (out_score) *out_score = best;
    return cv::Point2f(sc_center.x + dd*n_scene.x, sc_center.y + dd*n_scene.y);
}

// -----------------------------------------------------------------------
// Edge match by NARROW-SEARCH matchTemplate: build an (n x n) template strip and an
// (n x m) scene strip in the edge frame (n = tangent width, m = n + 2*search along
// the normal). matchTemplate of (n x m) vs (n x n) returns a 1 x (2*search+1) result
// directly — a 1D profile along the normal, with the n-wide template averaging over
// the tangent for free (SIMD). Subpixel peak of that 1D result = across-edge shift.
// -----------------------------------------------------------------------
// Extract a strip aligned to (n=along cols, t=along rows) centred at `c`, of output
// size (2*Lu+1) cols x (2*W+1) rows, via warpAffine (SIMD). Output(u,v) samples
// src at c + (u-Lu)*n + (v-W)*t.
static void extractStrip(const cv::Mat& src, cv::Point2f c, cv::Point2f n, cv::Point2f t,
                         int Lu, int W, cv::Mat& out) {
    double M[6] = { n.x, t.x, c.x - Lu*n.x - W*t.x,
                    n.y, t.y, c.y - Lu*n.y - W*t.y };
    cv::Mat Mm(2, 3, CV_64F, M);
    cv::warpAffine(src, out, Mm, cv::Size(2*Lu+1, 2*W+1),
                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
}

static cv::Point2f matchEdge_strip(const cv::Mat& templ_img, const cv::Mat& scene_img,
                                   cv::Point2f templ_pt, cv::Point2f n_templ, cv::Point2f t_templ,
                                   cv::Point2f sc_center, cv::Point2f n_scene, cv::Point2f t_scene,
                                   int half, int search_half, float* out_score) {
    if (out_score) *out_score = -1.0f;
    int W = std::min(half, 8);          // tangent half-width (averaging) -> n = 2W+1
    int Lt = std::min(half, 6);         // template normal half-extent
    int Ls = Lt + search_half;          // scene normal half-extent -> m = 2Ls+1
    cv::Mat tstrip, sstrip;
    extractStrip(templ_img, templ_pt, n_templ, t_templ, Lt, W, tstrip);   // (2W+1) x (2Lt+1)
    extractStrip(scene_img, sc_center, n_scene, t_scene, Ls, W, sstrip);  // (2W+1) x (2Ls+1)
    cv::Mat result;
    cv::matchTemplate(sstrip, tstrip, result, cv::TM_CCORR_NORMED);  // 1 x (2*search+1)
    const float* R = result.ptr<float>(0);
    int n = result.cols, bi = 0; float best = -2.0f;
    for (int i = 0; i < n; i++) if (R[i] > best) { best = R[i]; bi = i; }
    float dd = (float)bi;
    if (bi > 0 && bi < n-1) {
        float a=R[bi-1], b=R[bi], c=R[bi+1], den=a-2*b+c;
        if (std::abs(den) > kEpsilon) dd += std::max(-1.0f, std::min(1.0f, 0.5f*(a-c)/den));
    }
    float delta = dd - search_half;     // result index 0 == u-offset -search_half
    if (out_score) *out_score = best;
    return cv::Point2f(sc_center.x + delta*n_scene.x, sc_center.y + delta*n_scene.y);
}

// -----------------------------------------------------------------------
// Subpixel ROI template match
// -----------------------------------------------------------------------
static cv::Point2f matchROI_subpixel(const cv::Mat& templ_roi,
                                      const cv::Mat& scene_img,
                                      cv::Point2f expected,
                                      int search_half,
                                      float* out_score = nullptr) {
    if (out_score) *out_score = -1.0f;  // <0 => could not match
    int ex = (int)(expected.x + 0.5f), ey = (int)(expected.y + 0.5f);
    int half = templ_roi.rows / 2;

    int x0 = std::max(0, ex - search_half - half);
    int y0 = std::max(0, ey - search_half - half);
    int x1 = std::min(scene_img.cols, ex + search_half + half);
    int y1 = std::min(scene_img.rows, ey + search_half + half);

    if (x1 - x0 < templ_roi.cols || y1 - y0 < templ_roi.rows)
        return expected;

    // NOTE: cloning this strided search window to contiguous before matchTemplate was
    // tried for large (5MP) scenes and gave NO measurable speedup (per-call refine
    // stays ~0.81ms; the 60x60 window is only ~60 cache lines / microseconds to read).
    // At 5MP the COARSE matching dominates total time, not the refine.
    cv::Mat search_roi = scene_img(cv::Rect(x0, y0, x1 - x0, y1 - y0));

    // EXPERIMENT: match on gradient magnitude instead of raw intensity.
    // SBM_ROI_SOBEL=1 plain Sobel(3); =2 Gaussian(3)+Sobel (DoG, tames the noise a
    // differentiator otherwise amplifies). Both template patch and scene window go
    // through the same operator so TM_CCORR_NORMED still compares like with like.
    static const int kSobel = getenv("SBM_ROI_SOBEL") ? atoi(getenv("SBM_ROI_SOBEL")) : 0;
    cv::Mat t_use = templ_roi, s_use = search_roi;
    cv::Mat t_g, s_g;
    // SBM_ROI_BLUR=<odd k>: low-pass both patch and window before the raw-intensity NCC.
    // Unlike Sobel this is pure denoise (averages zero-mean noise), so it should HELP
    // under noise at the cost of a slightly softer subpixel peak on clean signal.
    static const int kBlur = getenv("SBM_ROI_BLUR") ? (atoi(getenv("SBM_ROI_BLUR")) | 1) : 0;
    cv::Mat t_b, s_b;
    if (kBlur >= 3) {
        cv::GaussianBlur(templ_roi, t_b, cv::Size(kBlur,kBlur), 0);
        cv::GaussianBlur(search_roi, s_b, cv::Size(kBlur,kBlur), 0);
        t_use = t_b; s_use = s_b;   // function-scope buffers; outlive this block
    }
    if (kSobel) {
        auto grad = [](const cv::Mat& in, cv::Mat& out, int mode){
            cv::Mat f = in; if (mode == 2) cv::GaussianBlur(in, f, cv::Size(3,3), 0);
            cv::Mat gx, gy; cv::Sobel(f, gx, CV_32F, 1, 0, 3); cv::Sobel(f, gy, CV_32F, 0, 1, 3);
            cv::magnitude(gx, gy, out);
        };
        grad(t_use, t_g, kSobel); grad(s_use, s_g, kSobel);
        t_use = t_g; s_use = s_g;
    }

    cv::Mat result;
    cv::matchTemplate(s_use, t_use, result, cv::TM_CCORR_NORMED);

    double max_val;
    cv::Point max_loc;
    cv::minMaxLoc(result, nullptr, &max_val, nullptr, &max_loc);
    if (out_score) *out_score = (float)max_val;   // peak correlation = match confidence

    // Subpixel via parabolic interpolation.
    // NOTE: a 2D quadratic facet fit and TM_CCOEFF_NORMED were both tried (2026-06-15)
    // and did NOT lower the ~0.12-0.15px floor (and destabilized some shapes) — the
    // floor is inherent to the warp-and-correlate per-point match, not the peak
    // interpolation. For sub-0.1px use RefineMode::ICP_Subpixel.
    float sx = (float)max_loc.x, sy = (float)max_loc.y;
    int mx = max_loc.x, my = max_loc.y;
    int rh = result.rows, rw = result.cols;

    if (mx > 0 && mx < rw - 1) {
        float a = result.at<float>(my, mx - 1);
        float b = result.at<float>(my, mx);
        float c = result.at<float>(my, mx + 1);
        float denom = a - 2*b + c;
        if (std::abs(denom) > kEpsilon)
            sx += std::max(-1.0f, std::min(1.0f, 0.5f * (a - c) / denom));
    }
    if (my > 0 && my < rh - 1) {
        float a = result.at<float>(my - 1, mx);
        float b = result.at<float>(my, mx);
        float c = result.at<float>(my + 1, mx);
        float denom = a - 2*b + c;
        if (std::abs(denom) > kEpsilon)
            sy += std::max(-1.0f, std::min(1.0f, 0.5f * (a - c) / denom));
    }

    return cv::Point2f(x0 + sx + half, y0 + sy + half);
}

// -----------------------------------------------------------------------
// PCA on ROI gradient → eigenvalues + eigenvectors
// -----------------------------------------------------------------------
static void roiPCA(const cv::Mat& roi, float eigvals[2], cv::Point2f eigvecs[2]) {
    cv::Mat dx, dy;
    cv::Sobel(roi, dx, CV_32F, 1, 0, 3);
    cv::Sobel(roi, dy, CV_32F, 0, 1, 3);

    cv::Mat mag;
    cv::magnitude(dx, dy, mag);
    float thresh = kPCAGradientThreshFactor * *std::max_element(mag.begin<float>(), mag.end<float>());

    // Collect edge pixel positions
    std::vector<cv::Point2f> pts;
    for (int r = 0; r < roi.rows; ++r)
        for (int c = 0; c < roi.cols; ++c)
            if (mag.at<float>(r, c) > thresh)
                pts.push_back(cv::Point2f((float)c, (float)r));

    if (pts.size() < 3) {
        eigvals[0] = eigvals[1] = 0;
        eigvecs[0] = cv::Point2f(1, 0);
        eigvecs[1] = cv::Point2f(0, 1);
        return;
    }

    // Compute covariance
    cv::Point2f mean(0, 0);
    for (auto& p : pts) mean += p;
    mean *= (1.0f / pts.size());

    float cxx = 0, cyy = 0, cxy = 0;
    for (auto& p : pts) {
        float dx = p.x - mean.x, dy = p.y - mean.y;
        cxx += dx*dx; cyy += dy*dy; cxy += dx*dy;
    }
    float n = (float)pts.size();
    cxx /= n; cyy /= n; cxy /= n;

    // Eigenvalues of [[cxx, cxy], [cxy, cyy]]
    float trace = cxx + cyy;
    float disc = std::sqrt(std::max(0.0f, (cxx-cyy)*(cxx-cyy)/4.0f + cxy*cxy));
    eigvals[0] = trace/2.0f + disc;  // larger
    eigvals[1] = trace/2.0f - disc;  // smaller

    // Eigenvectors
    if (std::abs(cxy) > kEpsilon) {
        eigvecs[0] = cv::Point2f(eigvals[0] - cyy, cxy);
        eigvecs[1] = cv::Point2f(eigvals[1] - cyy, cxy);
    } else {
        eigvecs[0] = (cxx >= cyy) ? cv::Point2f(1, 0) : cv::Point2f(0, 1);
        eigvecs[1] = (cxx >= cyy) ? cv::Point2f(0, 1) : cv::Point2f(1, 0);
    }
    // Normalize
    for (int i = 0; i < 2; ++i) {
        float len = std::sqrt(eigvecs[i].x*eigvecs[i].x + eigvecs[i].y*eigvecs[i].y);
        if (len > kEpsilon) eigvecs[i] *= (1.0f / len);
    }
}

// -----------------------------------------------------------------------
// Rigid body solve from constraints: minimize Σ (R*src + t - dst) · n)²
// -----------------------------------------------------------------------
static cv::Vec3f solveRigid(const std::vector<Constraint>& constraints,
                            cv::Point2f center) {
    // Build normal equations: A^T A x = A^T b
    // x = [theta, tx, ty]
    // For each constraint: ((R*src + t - dst) · n) should be 0
    // Linearized: J = [(-src_y'*nx + src_x'*ny), nx, ny] where src' = src - center

    float ATA[3][3] = {}, ATb[3] = {};

    for (auto& c : constraints) {
        float sx = c.src.x - center.x, sy = c.src.y - center.y;
        float dx = c.src.x - c.dst.x, dy = c.src.y - c.dst.y;
        float nx = c.normal.x, ny = c.normal.y;

        float j0 = -sy * nx + sx * ny;
        float j1 = nx;
        float j2 = ny;
        float e = dx * nx + dy * ny;
        float w = c.weight;

        ATA[0][0] += w*j0*j0; ATA[0][1] += w*j0*j1; ATA[0][2] += w*j0*j2;
        ATA[1][1] += w*j1*j1; ATA[1][2] += w*j1*j2;
        ATA[2][2] += w*j2*j2;
        ATb[0] -= w*j0*e;
        ATb[1] -= w*j1*e;
        ATb[2] -= w*j2*e;
    }

    // Symmetrize + regularize
    ATA[1][0] = ATA[0][1]; ATA[2][0] = ATA[0][2]; ATA[2][1] = ATA[1][2];
    for (int i = 0; i < 3; ++i) ATA[i][i] += kSolverRegularization;

    // Solve 3x3 via Cramer's rule (small matrix)
    cv::Mat A(3, 3, CV_32F, ATA);
    cv::Mat b(3, 1, CV_32F, ATb);
    cv::Mat x;
    cv::solve(A, b, x);

    return cv::Vec3f(x.at<float>(0), x.at<float>(1), x.at<float>(2));
}

// -----------------------------------------------------------------------
// Clamp ROI half-size to stay within image bounds
// -----------------------------------------------------------------------
static int safeROIHalf(int tx, int ty, int cols, int rows, int roi_half) {
    if (tx-roi_half<0||tx+roi_half>=cols||ty-roi_half<0||ty+roi_half>=rows)
        return std::min({tx, ty, cols-1-tx, rows-1-ty});
    return roi_half;
}

// -----------------------------------------------------------------------
// Main ROI refinement
// -----------------------------------------------------------------------
int roiHalfAt(int tx, int ty, int cols, int rows, int roi_half) {
    return safeROIHalf(tx, ty, cols, rows, roi_half);
}
void templatePCA(const cv::Mat& templ_img, int tx, int ty, int h, float eigvals[2], cv::Point2f eigvecs[2]) {
    cv::Mat roi_unrot = templ_img(cv::Rect(tx-h, ty-h, 2*h, 2*h));
    roiPCA(roi_unrot, eigvals, eigvecs);
}

cv::Vec3f refineROI(const cv::Mat& templ_img,
                    const cv::Mat& scene_img,
                    const std::vector<SamplePoint>& sample_points,
                    const cv::Vec3f& initial_pose,
                    const ROIConfig& config,
                    float* out_residual) {

    cv::Vec3f pose = initial_pose;

    // Iterate: match on iter 0, reuse correspondences for iter 1+
    // Matched dst positions are fixed after iter 0 — only the rigid
    // solve is re-run with updated src positions from the new pose.
    struct MatchedPoint {
        cv::Point2f dst;           // scene match (fixed after iter 0)
        cv::Point2f normal;        // PCA normal (fixed after iter 0, pre-rotation)
        cv::Point2f tangent;       // PCA tangent (for corners)
        bool is_corner;
        int sample_idx;
        float score;               // peak correlation at match (confidence)
    };
    std::vector<MatchedPoint> matched_points;
    float last_match_angle = -999;

    // Cache rotated ROI patches — warp once, reuse across re-matches
    struct CachedROI {
        cv::Mat patch;    // rotated ROI patch
        int sample_idx;
    };
    std::vector<CachedROI> cached_rois;
    float cached_angle = -999;

    for (int iteration = 0; iteration < config.max_iters; ++iteration) {

    float cx = pose[0], cy = pose[1];
    float angle_deg = pose[2];
    float angle_rad = angle_deg * (float)CV_PI / 180.0f;
    float cs = std::cos(angle_rad), sn = std::sin(angle_rad);

    float tcx = templ_img.cols / 2.0f, tcy = templ_img.rows / 2.0f;

    std::vector<Constraint> constraints;

    // Re-match if first iteration OR angle changed > threshold since last match.
    // iterative_rematch forces a re-match (and re-warp) every iteration (ICP-style),
    // so the match re-centres at the improved pose and corrects translation/along-edge
    // init error that a fixed-correspondence solve cannot.
    bool do_match = config.iterative_rematch ||
                    (std::abs(angle_deg - last_match_angle) > kAngleRematchDeg);

    if (do_match) {
        last_match_angle = angle_deg;
        matched_points.clear();

        // Warp ROI patches only if angle changed significantly from cache
        bool need_warp = config.iterative_rematch ||
                         (std::abs(angle_deg - cached_angle) > kAngleRewarpDeg);
        if (need_warp) {
            cached_angle = angle_deg;
            cached_rois.clear();
        }

        if (cached_rois.empty()) {
            // Warp each ROI patch once
            for (size_t si = 0; si < sample_points.size(); ++si) {
                auto& sp = sample_points[si];
                int tx = (int)(sp.pos.x + tcx + 0.5f);
                int ty = (int)(sp.pos.y + tcy + 0.5f);
                int h = safeROIHalf(tx, ty, templ_img.cols, templ_img.rows, config.roi_half);
                if (h < kMinROIHalf) continue;
                // NOTE: a contiguous-clone "cache-friendly" block was tried and reverted
                // — a 2h x 2h (~30x30) sub-image is only ~30 cache lines and already
                // fits entirely in L1, so cloning adds a copy with no warp speedup. The
                // strided view is fine here; it would only matter for large blocks.
                cv::Mat roi_unrot = templ_img(cv::Rect(tx-h, ty-h, 2*h, 2*h));
                cv::Mat roi;
                if (std::abs(angle_deg) > 0.5f) {
                    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f((float)h,(float)h), -angle_deg, 1.0);
                    cv::warpAffine(roi_unrot, roi, M, roi_unrot.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
                } else {
                    roi = roi_unrot.clone();
                }
                CachedROI cr;
                cr.patch = roi;
                cr.sample_idx = (int)si;
                cached_rois.push_back(cr);
            }
        }

        for (auto& cr : cached_rois) {
            auto& sp = sample_points[cr.sample_idx];
            cv::Mat& roi = cr.patch;

            float ex = cs * sp.pos.x - sn * sp.pos.y + cx;
            float ey = sn * sp.pos.x + cs * sp.pos.y + cy;

            // PCA on UNROTATED template patch (eigenvectors in template space).
            // Computed first so we know the edge normal + corner-ness before matching.
            int tx = (int)(sp.pos.x + tcx + 0.5f);
            int ty = (int)(sp.pos.y + tcy + 0.5f);
            int h = safeROIHalf(tx, ty, templ_img.cols, templ_img.rows, config.roi_half);
            float eigvals[2];
            cv::Point2f eigvecs[2];
            static const bool kNoPcaCache = getenv("SBM_NO_PCA_CACHE") != nullptr;
            static const bool kPcaCheck   = getenv("SBM_PCA_CHECK") != nullptr;
            if (h >= kMinROIHalf && sp.pca_valid && sp.pca_h == h && !kNoPcaCache) {
                // Precomputed at addModel from the same patch (templatePCA).
                eigvals[0] = sp.pca_eig[0]; eigvals[1] = sp.pca_eig[1];
                eigvecs[0] = sp.pca_vec[0]; eigvecs[1] = sp.pca_vec[1];
                if (kPcaCheck) {
                    float e2[2]; cv::Point2f v2[2];
                    cv::Mat roi_unrot = templ_img(cv::Rect(tx-h, ty-h, 2*h, 2*h));
                    roiPCA(roi_unrot, e2, v2);
                    if (e2[0] != eigvals[0] || e2[1] != eigvals[1] || v2[0] != eigvecs[0] || v2[1] != eigvecs[1])
                        fprintf(stderr, "[SBM_PCA_CHECK] pt %d at (%d,%d) h %d: cached eig %.6g %.6g vec (%.6f,%.6f) vs fresh eig %.6g %.6g vec (%.6f,%.6f) templ %dx%d\n",
                                cr.sample_idx, tx, ty, h, eigvals[0], eigvals[1], eigvecs[1].x, eigvecs[1].y, e2[0], e2[1], v2[1].x, v2[1].y, templ_img.cols, templ_img.rows);
                }
            } else if (h >= kMinROIHalf) {
                cv::Mat roi_unrot = templ_img(cv::Rect(tx-h, ty-h, 2*h, 2*h));
                roiPCA(roi_unrot, eigvals, eigvecs);
            } else {
                eigvals[0] = eigvals[1] = 0;
                eigvecs[0] = cv::Point2f(1,0); eigvecs[1] = cv::Point2f(0,1);
            }
            bool is_corner = (eigvals[0] > kEpsilon && eigvals[1] > kEpsilon &&
                              eigvals[0] / eigvals[1] < config.corner_eigen_ratio);
            cv::Point2f n_templ = eigvecs[1];   // across-edge normal (template frame)

            float match_score = -1.0f;
            cv::Point2f matched;
            if (config.edge_collapse && !is_corner && h >= kMinROIHalf) {
                // Narrow-search matchTemplate -> direct 1D across-edge profile.
                cv::Point2f t_templ = eigvecs[0];
                cv::Point2f n_scene(cs*n_templ.x - sn*n_templ.y, sn*n_templ.x + cs*n_templ.y);
                cv::Point2f t_scene(cs*t_templ.x - sn*t_templ.y, sn*t_templ.x + cs*t_templ.y);
                matched = matchEdge_strip(templ_img, scene_img,
                                          cv::Point2f((float)tx, (float)ty), n_templ, t_templ,
                                          cv::Point2f(ex, ey), n_scene, t_scene,
                                          h, config.search_half, &match_score);
            } else if (config.edge_1d_match && !is_corner && h >= kMinROIHalf) {
                // 1D profile match along the (rotated) normal — only the across-edge
                // displacement is measured. Each sample is averaged over a band along
                // the edge tangent (window width) to denoise.
                cv::Point2f n_scene(cs*n_templ.x - sn*n_templ.y,
                                    sn*n_templ.x + cs*n_templ.y);
                cv::Point2f t_templ = eigvecs[0];                 // along-edge (template)
                cv::Point2f t_scene(cs*t_templ.x - sn*t_templ.y,  // along-edge (scene)
                                    sn*t_templ.x + cs*t_templ.y);
                // Short template profile (<= ~6px) so it doesn't span a thin bar and
                // pick up the opposite edge (1D-profile ambiguity on thin features).
                int prof_h = std::min(h, 6);
                int tang_h = std::min(h, 10);                     // tangent averaging band
                matched = match1D_alongNormal(templ_img, scene_img,
                                              cv::Point2f((float)tx, (float)ty), n_templ,
                                              cv::Point2f(ex, ey), n_scene,
                                              prof_h, config.search_half, &match_score,
                                              t_templ, t_scene, tang_h);
            } else {
                matched = matchROI_subpixel(roi, scene_img, cv::Point2f(ex, ey),
                                            config.search_half, &match_score);
            }

            MatchedPoint mp;
            mp.dst = matched;
            mp.normal = n_templ;      // store unrotated — rotate per iteration
            mp.tangent = eigvecs[0];
            mp.is_corner = is_corner;
            mp.sample_idx = cr.sample_idx;
            mp.score = match_score;
            matched_points.push_back(mp);
        }
    }

    // Build constraints from matched points (reused across iterations)
    for (auto& mp : matched_points) {
        auto& sp = sample_points[mp.sample_idx];

        // Score gate: a match scoring far below this point's coarse-error envelope
        // floor landed on the wrong place (gross outlier) — drop it entirely.
        if (config.reject_low_score && sp.score_floor > 0.0f &&
            mp.score >= 0.0f && mp.score < sp.score_floor * config.reject_pct)
            continue;

        float ex = cs * sp.pos.x - sn * sp.pos.y + cx;
        float ey = sn * sp.pos.x + cs * sp.pos.y + cy;

        // Rotate normal by current pose angle
        cv::Point2f normal(cs*mp.normal.x - sn*mp.normal.y,
                           sn*mp.normal.x + cs*mp.normal.y);

        // Weight by self-match distinctiveness (lock_major along the constraint
        // normal) when enabled — ambiguous points contribute less, without being
        // removed (preserves point-count redundancy). Floor keeps every point active.
        float w_normal = config.weight_by_lock ? std::max(0.05f, sp.lock_major) : 1.0f;

        Constraint c1;
        c1.src = cv::Point2f(ex, ey);
        c1.dst = mp.dst;
        c1.normal = normal;
        c1.weight = w_normal;
        constraints.push_back(c1);

        if (mp.is_corner) {
            cv::Point2f tangent(cs*mp.tangent.x - sn*mp.tangent.y,
                                sn*mp.tangent.x + cs*mp.tangent.y);
            float w_tangent = config.weight_by_lock ? std::max(0.05f, sp.lock_minor) : 1.0f;
            Constraint c2;
            c2.src = cv::Point2f(ex, ey);
            c2.dst = mp.dst;
            c2.normal = tangent;
            c2.weight = w_tangent;
            constraints.push_back(c2);
        }
    }

    // Debug: print per-point matching accuracy
    if (config.verbose) {
        sbm::sbm_log(sbm::LogLevel::Debug, "roi", "[ROI] %d constraints from %d samples (angle=%.1f)",
                (int)constraints.size(), (int)sample_points.size(), angle_deg);
        for (size_t i = 0; i < constraints.size(); ++i) {
            auto& c = constraints[i];
            float dx = c.dst.x - c.src.x, dy = c.dst.y - c.src.y;
            float dist = std::sqrt(dx*dx + dy*dy);
            sbm::sbm_log(sbm::LogLevel::Debug, "roi", "  [%2d] src=(%.1f,%.1f) dst=(%.1f,%.1f) d=%.2f n=(%.2f,%.2f) w=%.1f",
                    (int)i, c.src.x, c.src.y, c.dst.x, c.dst.y, dist,
                    c.normal.x, c.normal.y, c.weight);
        }
    }

    // Reject outliers: remove constraints with distance > 2× median
    if (!constraints.empty()) {
        std::vector<float> dists;
        dists.reserve(constraints.size());
        for (auto& c : constraints) {
            float dx = c.dst.x - c.src.x, dy = c.dst.y - c.src.y;
            dists.push_back(std::sqrt(dx*dx + dy*dy));
        }
        std::sort(dists.begin(), dists.end());
        float median = dists[dists.size() / 2];
        float thresh = std::max(kOutlierMultiplier, median * kOutlierMultiplier);

        std::vector<Constraint> filtered;
        for (size_t i = 0; i < constraints.size(); ++i) {
            if (dists[i] <= thresh) filtered.push_back(constraints[i]);
        }
        if (config.verbose)
            sbm::sbm_log(sbm::LogLevel::Debug, "roi", "  outlier rejection: %d -> %d (thresh=%.1f)",
                    (int)constraints.size(), (int)filtered.size(), thresh);
        constraints = std::move(filtered);
    }

    if (constraints.empty())
        return initial_pose;

    // Solve rigid transform
    cv::Vec3f update = solveRigid(constraints, cv::Point2f(cx, cy));

    float d_theta = update[0];
    float d_tx = update[1];
    float d_ty = update[2];

    // Clamp
    d_theta = std::max(-kMaxThetaUpdate, std::min(kMaxThetaUpdate, d_theta));
    d_tx = std::max(-kMaxTransUpdate, std::min(kMaxTransUpdate, d_tx));
    d_ty = std::max(-kMaxTransUpdate, std::min(kMaxTransUpdate, d_ty));

    float refined_angle = std::fmod(angle_deg + d_theta * 180.0f / (float)CV_PI, 360.0f);
    if (refined_angle < 0) refined_angle += 360.0f;

    pose = cv::Vec3f(cx + d_tx, cy + d_ty, refined_angle);
    if (config.verbose)
        sbm::sbm_log(sbm::LogLevel::Debug, "roi", "  iter %d: angle=%.1f pos=(%.1f,%.1f) d_theta=%.3f d_t=(%.2f,%.2f)",
                iteration, pose[2], pose[0], pose[1], d_theta*180/(float)CV_PI, d_tx, d_ty);

    } // end iteration loop

    // Per-result confidence: mean |point-to-line| residual of the matched points at
    // the final pose. Consistent points -> ~0; disagreeing points (occlusion / gross
    // mismatch / completely-off init) -> large.
    if (out_residual) {
        float fa = pose[2] * (float)CV_PI / 180.0f;
        float fcs = std::cos(fa), fsn = std::sin(fa), fcx = pose[0], fcy = pose[1];
        float sum = 0; int cnt = 0;
        for (auto& mp : matched_points) {
            auto& sp = sample_points[mp.sample_idx];
            if (config.reject_low_score && sp.score_floor > 0.0f &&
                mp.score >= 0.0f && mp.score < sp.score_floor * config.reject_pct)
                continue;
            float ex = fcs * sp.pos.x - fsn * sp.pos.y + fcx;
            float ey = fsn * sp.pos.x + fcs * sp.pos.y + fcy;
            cv::Point2f n(fcs * mp.normal.x - fsn * mp.normal.y,
                          fsn * mp.normal.x + fcs * mp.normal.y);
            float e = (ex - mp.dst.x) * n.x + (ey - mp.dst.y) * n.y;
            sum += std::abs(e); cnt++;
        }
        *out_residual = (cnt > 0) ? sum / cnt : -1.0f;
    }

    return pose;
}


} // namespace roi_refine
