/// @file roi_refine.cpp
/// @brief ROI-based pose refinement implementation.

#include "roi_refine.h"
#include <opencv2/imgproc.hpp>
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
// Subpixel ROI template match
// -----------------------------------------------------------------------
static cv::Point2f matchROI_subpixel(const cv::Mat& templ_roi,
                                      const cv::Mat& scene_img,
                                      cv::Point2f expected,
                                      int search_half) {
    int ex = (int)(expected.x + 0.5f), ey = (int)(expected.y + 0.5f);
    int half = templ_roi.rows / 2;

    int x0 = std::max(0, ex - search_half - half);
    int y0 = std::max(0, ey - search_half - half);
    int x1 = std::min(scene_img.cols, ex + search_half + half);
    int y1 = std::min(scene_img.rows, ey + search_half + half);

    if (x1 - x0 < templ_roi.cols || y1 - y0 < templ_roi.rows)
        return expected;

    cv::Mat search_roi = scene_img(cv::Rect(x0, y0, x1 - x0, y1 - y0));

    cv::Mat result;
    cv::matchTemplate(search_roi, templ_roi, result, cv::TM_CCORR_NORMED);

    double max_val;
    cv::Point max_loc;
    cv::minMaxLoc(result, nullptr, &max_val, nullptr, &max_loc);

    // Subpixel via parabolic interpolation
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
cv::Vec3f refineROI(const cv::Mat& templ_img,
                    const cv::Mat& scene_img,
                    const std::vector<SamplePoint>& sample_points,
                    const cv::Vec3f& initial_pose,
                    const ROIConfig& config) {

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

    // Re-match if first iteration OR if angle changed > 2° since last match
    bool do_match = (std::abs(angle_deg - last_match_angle) > kAngleRematchDeg);

    if (do_match) {
        last_match_angle = angle_deg;
        matched_points.clear();

        // Warp ROI patches only if angle changed significantly from cache
        bool need_warp = (std::abs(angle_deg - cached_angle) > kAngleRewarpDeg);
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

            cv::Point2f matched = matchROI_subpixel(roi, scene_img,
                                                     cv::Point2f(ex, ey),
                                                     config.search_half);

            // PCA on UNROTATED template patch (eigenvectors in template space)
            int tx = (int)(sp.pos.x + tcx + 0.5f);
            int ty = (int)(sp.pos.y + tcy + 0.5f);
            int h = safeROIHalf(tx, ty, templ_img.cols, templ_img.rows, config.roi_half);
            float eigvals[2];
            cv::Point2f eigvecs[2];
            if (h >= kMinROIHalf) {
                cv::Mat roi_unrot = templ_img(cv::Rect(tx-h, ty-h, 2*h, 2*h));
                roiPCA(roi_unrot, eigvals, eigvecs);
            } else {
                eigvals[0] = eigvals[1] = 0;
                eigvecs[0] = cv::Point2f(1,0); eigvecs[1] = cv::Point2f(0,1);
            }

            MatchedPoint mp;
            mp.dst = matched;
            mp.normal = eigvecs[1];   // store unrotated — rotate per iteration
            mp.tangent = eigvecs[0];
            mp.is_corner = (eigvals[0] > kEpsilon && eigvals[1] > kEpsilon &&
                            eigvals[0] / eigvals[1] < config.corner_eigen_ratio);
            mp.sample_idx = cr.sample_idx;
            matched_points.push_back(mp);
        }
    }

    // Build constraints from matched points (reused across iterations)
    for (auto& mp : matched_points) {
        auto& sp = sample_points[mp.sample_idx];
        float ex = cs * sp.pos.x - sn * sp.pos.y + cx;
        float ey = sn * sp.pos.x + cs * sp.pos.y + cy;

        // Rotate normal by current pose angle
        cv::Point2f normal(cs*mp.normal.x - sn*mp.normal.y,
                           sn*mp.normal.x + cs*mp.normal.y);

        Constraint c1;
        c1.src = cv::Point2f(ex, ey);
        c1.dst = mp.dst;
        c1.normal = normal;
        c1.weight = 1.0f;
        constraints.push_back(c1);

        if (mp.is_corner) {
            cv::Point2f tangent(cs*mp.tangent.x - sn*mp.tangent.y,
                                sn*mp.tangent.x + cs*mp.tangent.y);
            Constraint c2;
            c2.src = cv::Point2f(ex, ey);
            c2.dst = mp.dst;
            c2.normal = tangent;
            c2.weight = 1.0f;
            constraints.push_back(c2);
        }
    }

    // Debug: print per-point matching accuracy
    if (config.verbose) {
        fprintf(stderr, "[ROI] %d constraints from %d samples (angle=%.1f)\n",
                (int)constraints.size(), (int)sample_points.size(), angle_deg);
        for (size_t i = 0; i < constraints.size(); ++i) {
            auto& c = constraints[i];
            float dx = c.dst.x - c.src.x, dy = c.dst.y - c.src.y;
            float dist = std::sqrt(dx*dx + dy*dy);
            fprintf(stderr, "  [%2d] src=(%.1f,%.1f) dst=(%.1f,%.1f) d=%.2f n=(%.2f,%.2f) w=%.1f\n",
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
            fprintf(stderr, "  outlier rejection: %d -> %d (thresh=%.1f)\n",
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
        fprintf(stderr, "  iter %d: angle=%.1f pos=(%.1f,%.1f) d_theta=%.3f d_t=(%.2f,%.2f)\n",
                iteration, pose[2], pose[0], pose[1], d_theta*180/(float)CV_PI, d_tx, d_ty);

    } // end iteration loop

    return pose;
}


} // namespace roi_refine
