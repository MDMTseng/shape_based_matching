/// @file shape_matcher.cpp
/// @brief High-level shape-based matching API implementation.

#include "shape_matcher.h"
#include "sbm_log.h"
#include "line2Dup.h"
#include "icp_refine.h"
#include "roi_refine.h"

#include <opencv2/imgproc.hpp>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <cmath>

// --- Named constants (extracted from magic numbers) ---
static constexpr float kCornerScoreMultiplier   = 10.0f;   // Cornerness weighting in feature selection score
static constexpr float kOptMinDistFactor        = 0.5f;    // Multiplier on min_dist for optimization swap spacing
static constexpr int   kMaxOptIterations        = 20;      // Max iterations for sensitivity optimization loop
static constexpr float kCornerThreshold         = 0.3f;    // Cornerness above this = corner (don't swap out)
static constexpr float kSensitivityBalanceRatio = 2.0f;    // Stop optimizing when worst/least sensitivity < this
static constexpr float kMinImprovementFactor    = 0.99f;   // Swap must beat current worst by this factor
static constexpr float kSolverRegularization    = 0.001f;  // Tikhonov regularization for 3x3 rigid solver
static constexpr int   kDefaultROIHalf          = 15;      // Default half-size for ROI search/template window
static constexpr int   kDefaultROIMaxIters      = 3;       // Default max iterations for ROI refinement
static constexpr int   kDefaultOptPoints        = 8;       // Default number of optimized sample points

namespace sbm {

// ============================================================
// FeatureSet serialization
// ============================================================

bool FeatureSet::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    // Header
    const char magic[] = "SBM1";
    f.write(magic, 4);
    int32_t nl = (int32_t)levels.size();
    f.write((char*)&nl, 4);
    f.write((char*)&templ_width, 4);
    f.write((char*)&templ_height, 4);
    f.write((char*)&origin.x, 4);
    f.write((char*)&origin.y, 4);
    f.write((char*)&angle_offset, 4);

    // Template image (for ROI refinement)
    int32_t img_rows = templ_image.rows, img_cols = templ_image.cols;
    f.write((char*)&img_rows, 4);
    f.write((char*)&img_cols, 4);
    if (img_rows > 0 && img_cols > 0) {
        for (int r = 0; r < img_rows; ++r)
            f.write((char*)templ_image.ptr(r), img_cols);
    }

    // Refine points (edges + corners)
    int32_t ne = (int32_t)refine_points.size();
    f.write((char*)&ne, 4);
    for (auto& rp : refine_points) {
        f.write((char*)&rp.px, 4);
        f.write((char*)&rp.py, 4);
        f.write((char*)&rp.nx, 4);
        f.write((char*)&rp.ny, 4);
        f.write((char*)&rp.cornerness, 4);
        uint8_t t = rp.type;
        f.write((char*)&t, 1);
    }

    // Per level
    for (auto& lv : levels) {
        int32_t nf = (int32_t)lv.features.size();
        f.write((char*)&nf, 4);
        f.write((char*)&lv.tl_x, 4);
        f.write((char*)&lv.tl_y, 4);
        f.write((char*)&lv.width, 4);
        f.write((char*)&lv.height, 4);
        f.write((char*)&lv.level, 4);
        for (auto& ft : lv.features) {
            f.write((char*)&ft.x, 4);
            f.write((char*)&ft.y, 4);
            f.write((char*)&ft.label, 4);
            f.write((char*)&ft.theta, 4);
            f.write((char*)&ft.cornerness, 4);
        }
    }
    return f.good();
}

FeatureSet FeatureSet::load(const std::string& path) {
    FeatureSet fs;
    std::ifstream f(path, std::ios::binary);
    if (!f) return fs;

    char magic[4];
    f.read(magic, 4);
    if (magic[0] != 'S' || magic[1] != 'B' || magic[2] != 'M' || magic[3] != '1')
        return fs;

    int32_t nl;
    f.read((char*)&nl, 4);
    f.read((char*)&fs.templ_width, 4);
    f.read((char*)&fs.templ_height, 4);
    f.read((char*)&fs.origin.x, 4);
    f.read((char*)&fs.origin.y, 4);
    f.read((char*)&fs.angle_offset, 4);

    // Template image
    int32_t img_rows, img_cols;
    f.read((char*)&img_rows, 4);
    f.read((char*)&img_cols, 4);
    if (img_rows > 0 && img_cols > 0) {
        fs.templ_image = cv::Mat(img_rows, img_cols, CV_8U);
        for (int r = 0; r < img_rows; ++r)
            f.read((char*)fs.templ_image.ptr(r), img_cols);
    }

    // Refine points (edges + corners)
    int32_t ne;
    f.read((char*)&ne, 4);
    fs.refine_points.resize(ne);
    for (auto& rp : fs.refine_points) {
        f.read((char*)&rp.px, 4);
        f.read((char*)&rp.py, 4);
        f.read((char*)&rp.nx, 4);
        f.read((char*)&rp.ny, 4);
        f.read((char*)&rp.cornerness, 4);
        uint8_t t; f.read((char*)&t, 1);
        rp.type = (FeatureSet::RefinePt::Type)t;
    }

    fs.levels.resize(nl);
    for (auto& lv : fs.levels) {
        int32_t nf;
        f.read((char*)&nf, 4);
        f.read((char*)&lv.tl_x, 4);
        f.read((char*)&lv.tl_y, 4);
        f.read((char*)&lv.width, 4);
        f.read((char*)&lv.height, 4);
        f.read((char*)&lv.level, 4);
        lv.features.resize(nf);
        for (auto& ft : lv.features) {
            f.read((char*)&ft.x, 4);
            f.read((char*)&ft.y, 4);
            f.read((char*)&ft.label, 4);
            f.read((char*)&ft.theta, 4);
            f.read((char*)&ft.cornerness, 4);
        }
    }
    if (!f.good()) {
        sbm::sbm_log(sbm::LogLevel::Error, "io", "FeatureSet::load('%s') read error", path.c_str());
        return FeatureSet{};
    }
    return fs;
}

// ============================================================
// Feature extraction
// ============================================================

FeatureSet extractFeatures(const cv::Mat& templ_gray,
                           const cv::Mat& mask,
                           int num_features,
                           const std::vector<int>& pyramid_T) {
    FeatureSet fs;
    fs.templ_width = templ_gray.cols;
    fs.templ_height = templ_gray.rows;
    fs.origin = cv::Point2f(templ_gray.cols / 2.0f, templ_gray.rows / 2.0f);
    fs.templ_image = templ_gray.clone();

    cv::Mat use_mask = mask;
    if (use_mask.empty())
        use_mask = cv::Mat(templ_gray.size(), CV_8U, cv::Scalar(255));

    // Use meiqua detector to extract features
    line2Dup::Detector det(num_features, pyramid_T, 30, 60);
    int id = det.addTemplate(templ_gray, "_extract_", use_mask);
    if (id < 0) return fs;

    // Compute structure tensor for cornerness classification
    cv::Mat smooth_st, dx_st, dy_st;
    cv::GaussianBlur(templ_gray, smooth_st, cv::Size(5, 5), 0);
    cv::Sobel(smooth_st, dx_st, CV_32F, 1, 0, 3);
    cv::Sobel(smooth_st, dy_st, CV_32F, 0, 1, 3);

    // Structure tensor components (smoothed with a window)
    cv::Mat Ixx = dx_st.mul(dx_st);
    cv::Mat Iyy = dy_st.mul(dy_st);
    cv::Mat Ixy = dx_st.mul(dy_st);
    cv::GaussianBlur(Ixx, Ixx, cv::Size(3, 3), 0);
    cv::GaussianBlur(Iyy, Iyy, cv::Size(3, 3), 0);
    cv::GaussianBlur(Ixy, Ixy, cv::Size(3, 3), 0);

    auto& tp = det.getTemplates("_extract_", id);
    fs.levels.resize(tp.size());
    for (size_t i = 0; i < tp.size(); ++i) {
        auto& src = tp[i];
        auto& dst = fs.levels[i];
        dst.tl_x = src.tl_x;
        dst.tl_y = src.tl_y;
        dst.width = src.width;
        dst.height = src.height;
        dst.level = src.pyramid_level;
        dst.features.resize(src.features.size());
        for (size_t j = 0; j < src.features.size(); ++j) {
            dst.features[j].x = src.features[j].x;
            dst.features[j].y = src.features[j].y;
            dst.features[j].label = src.features[j].label;
            dst.features[j].theta = src.features[j].theta;

            // Compute cornerness from structure tensor eigenvalues.
            int fx = src.features[j].x + src.tl_x;
            int fy = src.features[j].y + src.tl_y;
            if (fx >= 0 && fx < templ_gray.cols && fy >= 0 && fy < templ_gray.rows) {
                float a = Ixx.at<float>(fy, fx);
                float b = Ixy.at<float>(fy, fx);
                float c = Iyy.at<float>(fy, fx);
                // Eigenvalues of [[a,b],[b,c]]:
                // λ = (a+c)/2 ± sqrt(((a-c)/2)² + b²)
                float trace = a + c;
                float disc = std::sqrt(std::max(0.0f, (a-c)*(a-c)/4.0f + b*b));
                float lam_max = trace/2.0f + disc;
                float lam_min = trace/2.0f - disc;
                // cornerness = λ_min / λ_max  (0=edge, 1=corner)
                dst.features[j].cornerness = (lam_max > 1e-6f) ?
                    std::min(1.0f, std::max(0.0f, lam_min / lam_max)) : 0;
            } else {
                dst.features[j].cornerness = 0;
            }
        }
    }

    // Extract refinement points: dense Canny edges + Harris corners
    {
        cv::Mat smooth, dx16, dy16, canny_edges;
        cv::GaussianBlur(templ_gray, smooth, cv::Size(5, 5), 0);
        cv::Sobel(smooth, dx16, CV_16S, 1, 0, 3);
        cv::Sobel(smooth, dy16, CV_16S, 0, 1, 3);
        cv::Canny(dx16, dy16, canny_edges, 30, 60);

        float cx = templ_gray.cols / 2.0f, cy = templ_gray.rows / 2.0f;

        // Dense Canny edge points
        for (int r = 0; r < templ_gray.rows; ++r) {
            const short* dxr = dx16.ptr<short>(r);
            const short* dyr = dy16.ptr<short>(r);
            for (int c = 0; c < templ_gray.cols; ++c) {
                if (canny_edges.at<uchar>(r, c) == 0) continue;
                float gx = (float)dxr[c], gy = (float)dyr[c];
                float mag = std::sqrt(gx*gx + gy*gy);
                if (mag < 1e-6f) continue;
                FeatureSet::RefinePt rp;
                rp.px = c - cx;
                rp.py = r - cy;
                rp.nx = gx / mag;
                rp.ny = gy / mag;
                rp.cornerness = 0;
                rp.type = FeatureSet::RefinePt::EDGE;
                fs.refine_points.push_back(rp);
            }
        }

        // Harris corners
        cv::Mat harris_resp;
        cv::cornerHarris(smooth, harris_resp, 3, 3, 0.04);

        // Threshold + NMS to get corner positions
        double max_resp;
        cv::minMaxLoc(harris_resp, nullptr, &max_resp);
        float corner_thresh = (float)(max_resp * 0.1);  // top 10% response

        // Simple 5x5 NMS on Harris response
        for (int r = 3; r < templ_gray.rows - 3; ++r) {
            for (int c = 3; c < templ_gray.cols - 3; ++c) {
                float val = harris_resp.at<float>(r, c);
                if (val < corner_thresh) continue;

                // Check if local maximum in 5x5
                bool is_max = true;
                for (int dr = -2; dr <= 2 && is_max; ++dr)
                    for (int dc = -2; dc <= 2 && is_max; ++dc)
                        if ((dr||dc) && harris_resp.at<float>(r+dr, c+dc) >= val)
                            is_max = false;
                if (!is_max) continue;

                // Get gradient at corner for normal direction
                float gx = (float)dx16.at<short>(r, c);
                float gy = (float)dy16.at<short>(r, c);
                float mag = std::sqrt(gx*gx + gy*gy);

                FeatureSet::RefinePt rp;
                rp.px = c - cx;
                rp.py = r - cy;
                rp.nx = (mag > 1e-6f) ? gx/mag : 0;
                rp.ny = (mag > 1e-6f) ? gy/mag : 0;
                rp.cornerness = std::min(1.0f, val / (float)max_resp);
                rp.type = FeatureSet::RefinePt::CORNER;
                fs.refine_points.push_back(rp);
            }
        }
    }

    return fs;
}


// ============================================================
// Sensitivity analysis
// ============================================================

// Internal constraint for sensitivity analysis
struct SensConstraint {
    cv::Point2f src, dst, normal;
};

// Solve rigid transform from constraints: returns (theta_deg, tx, ty)
static cv::Vec3f solveSens(const std::vector<SensConstraint>& cs) {
    float ATA[3][3]={}, ATb[3]={};
    for (auto& c : cs) {
        float dx=c.src.x-c.dst.x, dy=c.src.y-c.dst.y;
        float nx=c.normal.x, ny=c.normal.y;
        float e = dx*nx + dy*ny;
        float j0=-c.src.y*nx+c.src.x*ny, j1=nx, j2=ny;
        ATA[0][0]+=j0*j0; ATA[0][1]+=j0*j1; ATA[0][2]+=j0*j2;
        ATA[1][1]+=j1*j1; ATA[1][2]+=j1*j2;
        ATA[2][2]+=j2*j2;
        ATb[0]-=j0*e; ATb[1]-=j1*e; ATb[2]-=j2*e;
    }
    ATA[1][0]=ATA[0][1]; ATA[2][0]=ATA[0][2]; ATA[2][1]=ATA[1][2];
    for(int i=0;i<3;i++) ATA[i][i]+=kSolverRegularization;
    cv::Mat A(3,3,CV_32F,ATA), b(3,1,CV_32F,ATb), x;
    cv::solve(A,b,x);
    return cv::Vec3f(x.at<float>(0)*180/(float)CV_PI, x.at<float>(1), x.at<float>(2));
}

// Build a constraint from a refine point (extract normal via Sobel)
static bool buildConstraint(const FeatureSet& fs, int rp_idx, SensConstraint& out) {
    auto& rp = fs.refine_points[rp_idx];
    float tcx = fs.templ_width / 2.0f, tcy = fs.templ_height / 2.0f;
    int tx=(int)(rp.px+tcx+0.5f), ty=(int)(rp.py+tcy+0.5f);
    int h=15;
    if(tx-h<0||tx+h>=fs.templ_image.cols||ty-h<0||ty+h>=fs.templ_image.rows)
        h=std::min({tx,ty,fs.templ_image.cols-1-tx,fs.templ_image.rows-1-ty});
    if(h<5) return false;
    cv::Mat roi2=fs.templ_image(cv::Rect(tx-h,ty-h,2*h,2*h));
    cv::Mat dx2,dy2,mag2;
    cv::Sobel(roi2,dx2,CV_32F,1,0,3);
    cv::Sobel(roi2,dy2,CV_32F,0,1,3);
    cv::magnitude(dx2,dy2,mag2);
    double max_mag; cv::Point max_loc;
    cv::minMaxLoc(mag2, nullptr, &max_mag, nullptr, &max_loc);
    float gx=dx2.at<float>(max_loc.y,max_loc.x);
    float gy=dy2.at<float>(max_loc.y,max_loc.x);
    float gm=std::sqrt(gx*gx+gy*gy);
    if (gm < 1e-6f) return false;
    out.src = cv::Point2f(rp.px, rp.py);
    out.dst = out.src;
    out.normal = cv::Point2f(gx/gm, gy/gm);
    return true;
}

// Build constraint set from points (for sensitivity evaluation)
static std::vector<SensConstraint> buildConstraintSet(const FeatureSet& fs,
                                                       const std::vector<cv::Point2f>& pts) {
    std::vector<SensConstraint> cs;
    for (auto& pt : pts) {
        int best = -1; float best_d = 1e9f;
        for (size_t i = 0; i < fs.refine_points.size(); i++) {
            float dx = pt.x - fs.refine_points[i].px, dy = pt.y - fs.refine_points[i].py;
            float d = dx*dx + dy*dy;
            if (d < best_d) { best_d = d; best = (int)i; }
        }
        if (best >= 0) {
            SensConstraint c;
            if (buildConstraint(fs, best, c))
                cs.push_back(c);
        }
    }
    return cs;
}

// Compute max sensitivity (max of d_ang and d_pos across all features)
static float computeMaxSens(const std::vector<SensConstraint>& baseline) {
    cv::Vec3f base_pose = solveSens(baseline);
    float worst = 0;
    for (size_t fi = 0; fi < baseline.size(); ++fi) {
        auto perturbed = baseline;
        perturbed[fi].dst.x += 1.0f;
        cv::Vec3f pdx = solveSens(perturbed);
        float ang_dx = std::abs(pdx[0] - base_pose[0]);
        float pos_dx = std::sqrt((pdx[1]-base_pose[1])*(pdx[1]-base_pose[1]) +
                                  (pdx[2]-base_pose[2])*(pdx[2]-base_pose[2]));
        perturbed = baseline;
        perturbed[fi].dst.y += 1.0f;
        cv::Vec3f pdy = solveSens(perturbed);
        float ang_dy = std::abs(pdy[0] - base_pose[0]);
        float pos_dy = std::sqrt((pdy[1]-base_pose[1])*(pdy[1]-base_pose[1]) +
                                  (pdy[2]-base_pose[2])*(pdy[2]-base_pose[2]));
        float d_ang = ang_dx + ang_dy;
        float d_pos = std::sqrt(pos_dx*pos_dx + pos_dy*pos_dy);
        worst = std::max(worst, std::max(d_ang, d_pos));
    }
    return worst;
}

// ================================================================
// Match confidence scoring for V3 selection
// ================================================================

/// Measure how distinctively a ROI patch can be matched.
/// Runs matchTemplate of the patch against a padded region of the template,
/// returns (peak - mean) / stddev of the response surface.
/// High value = sharp, unambiguous peak = reliable matching.
static float measurePeakSharpness(const cv::Mat& roi_patch, const cv::Mat& templ_image,
                                   int src_x, int src_y, int search_extra = 10) {
    int half = roi_patch.rows / 2;
    int x0 = std::max(0, src_x - half - search_extra);
    int y0 = std::max(0, src_y - half - search_extra);
    int x1 = std::min(templ_image.cols, src_x + half + search_extra);
    int y1 = std::min(templ_image.rows, src_y + half + search_extra);

    if (x1 - x0 < roi_patch.cols || y1 - y0 < roi_patch.rows)
        return 0.0f;

    cv::Mat search = templ_image(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    cv::Mat result;
    cv::matchTemplate(search, roi_patch, result, cv::TM_CCORR_NORMED);

    if (result.empty()) return 0.0f;

    cv::Scalar mean_s, std_s;
    cv::meanStdDev(result, mean_s, std_s);

    double max_val;
    cv::minMaxLoc(result, nullptr, &max_val);

    float stddev = (float)std_s[0];
    if (stddev < 1e-6f) return 0.0f;

    return (float)(max_val - mean_s[0]) / stddev;
}

/// Measure how stable a ROI patch match is under Gaussian noise.
/// Adds noise to the search region, re-matches, returns stddev of matched positions.
/// Low value = robust to noise. High value = match wanders under noise.
static float measureNoiseStability(const cv::Mat& roi_patch, const cv::Mat& templ_image,
                                    int src_x, int src_y, float noise_sigma = 30.0f,
                                    int n_trials = 3, int search_extra = 10) {
    int half = roi_patch.rows / 2;
    int x0 = std::max(0, src_x - half - search_extra);
    int y0 = std::max(0, src_y - half - search_extra);
    int x1 = std::min(templ_image.cols, src_x + half + search_extra);
    int y1 = std::min(templ_image.rows, src_y + half + search_extra);

    if (x1 - x0 < roi_patch.cols || y1 - y0 < roi_patch.rows)
        return 10.0f;  // bad — too close to border

    cv::Mat search_clean = templ_image(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();

    std::vector<float> dx_vals, dy_vals;
    cv::RNG rng(42);

    for (int t = 0; t < n_trials; t++) {
        cv::Mat noisy = search_clean.clone();
        cv::Mat noise(noisy.size(), CV_32F);
        rng.fill(noise, cv::RNG::NORMAL, 0, noise_sigma);
        cv::Mat noisy_f;
        noisy.convertTo(noisy_f, CV_32F);
        noisy_f += noise;
        noisy_f.convertTo(noisy, CV_8U);

        cv::Mat result;
        cv::matchTemplate(noisy, roi_patch, result, cv::TM_CCORR_NORMED);

        cv::Point max_loc;
        cv::minMaxLoc(result, nullptr, nullptr, nullptr, &max_loc);

        // Expected peak is at (search_extra, search_extra)
        dx_vals.push_back((float)max_loc.x - search_extra);
        dy_vals.push_back((float)max_loc.y - search_extra);
    }

    // Compute stddev of match offsets
    float mean_dx = 0, mean_dy = 0;
    for (int i = 0; i < n_trials; i++) { mean_dx += dx_vals[i]; mean_dy += dy_vals[i]; }
    mean_dx /= n_trials; mean_dy /= n_trials;

    float var = 0;
    for (int i = 0; i < n_trials; i++) {
        float ddx = dx_vals[i] - mean_dx, ddy = dy_vals[i] - mean_dy;
        var += ddx*ddx + ddy*ddy;
    }
    return std::sqrt(var / n_trials);
}

std::vector<cv::Point2f> FeatureSet::selectOptimizedPoints(int max_points) const {
    // Return cached result if available and computed with same (or larger) max_points
    if (!cached_opt_points.empty() && cached_opt_max_points >= max_points)
        return cached_opt_points;

    std::vector<cv::Point2f> result;
    if (refine_points.empty() || templ_image.empty())
        return result;

    float tcx = templ_width / 2.0f, tcy = templ_height / 2.0f;

    // ================================================================
    // Grid-based spatial distribution (replaces fixed d_min)
    // Divides template into KxK cells. Each cell allows max_per_cell features.
    // Handles asymmetric templates better than fixed radius.
    // ================================================================
    static const int kGridK = 5;
    static const int kMaxPerCell = 2;
    float cell_w = templ_width / (float)kGridK;
    float cell_h = templ_height / (float)kGridK;

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

    // ================================================================
    // Collect candidates with precomputed properties
    // Fix #3: Use Shi-Tomasi score min(lambda1, lambda2) instead of ratio.
    //         Apply slight blur before structure tensor for stability.
    // Fix #1: Compute gradient magnitude for edge precision weighting.
    // ================================================================
    struct Cand {
        int rp_idx;
        float px, py, R;
        float j0, j1, j2;      // Jacobian row
        float shi_tomasi;       // min(lambda1, lambda2) — corner strength
        float grad_mag;         // gradient magnitude — edge precision proxy
        bool is_corner;
    };
    std::vector<Cand> corners, edges;

    // Slight blur for stable structure tensor (Fix #3)
    cv::Mat templ_blur;
    cv::GaussianBlur(templ_image, templ_blur, cv::Size(3, 3), 1.0);

    for (size_t i = 0; i < refine_points.size(); i++) {
        auto& rp = refine_points[i];
        if (std::abs(rp.px) > templ_width/2.0f - 5 || std::abs(rp.py) > templ_height/2.0f - 5)
            continue;

        // Get normal from Sobel
        SensConstraint sc;
        if (!buildConstraint(*this, (int)i, sc)) continue;
        float nx = sc.normal.x, ny = sc.normal.y;

        // Compute structure tensor on blurred template for stable corner detection
        int tx = (int)(rp.px + tcx + 0.5f), ty = (int)(rp.py + tcy + 0.5f);
        int h = 8;
        if (tx-h<0||tx+h>=templ_blur.cols||ty-h<0||ty+h>=templ_blur.rows)
            h = std::min({tx, ty, templ_blur.cols-1-tx, templ_blur.rows-1-ty});
        if (h < 3) continue;

        cv::Mat roi = templ_blur(cv::Rect(tx-h, ty-h, 2*h, 2*h));
        cv::Mat dx, dy;
        cv::Sobel(roi, dx, CV_32F, 1, 0, 3);
        cv::Sobel(roi, dy, CV_32F, 0, 1, 3);

        // Structure tensor eigenvalues
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

        // Eigenvalues of [[m00,m01],[m01,m11]]
        float trace = m00 + m11;
        float disc = std::sqrt(std::max(0.0f, (m00-m11)*(m00-m11)/4.0f + m01*m01));
        float lam1 = trace/2.0f + disc;  // larger
        float lam2 = trace/2.0f - disc;  // smaller

        Cand cd;
        cd.rp_idx = (int)i;
        cd.px = rp.px; cd.py = rp.py;
        cd.R = std::sqrt(rp.px*rp.px + rp.py*rp.py);
        cd.j0 = -rp.py * nx + rp.px * ny;
        cd.j1 = nx; cd.j2 = ny;
        cd.shi_tomasi = std::max(0.0f, lam2);  // min eigenvalue = Shi-Tomasi score
        cd.grad_mag = n_pix > 0 ? mag_sum / n_pix : 0;
        // Corner classification: use original cornerness from feature extraction
        // (Harris-based, already calibrated) combined with Shi-Tomasi min eigenvalue
        // from the blurred structure tensor as a stability filter.
        cd.is_corner = (rp.cornerness > kCornerThreshold && cd.shi_tomasi > 5.0f);

        if (cd.is_corner) corners.push_back(cd);
        else edges.push_back(cd);
    }

    std::vector<int> selected;

    // ================================================================
    // Phase 1: Corners by Shi-Tomasi score × leverage
    // Sort by R (leverage) since all corners have strong 2D constraint.
    // Grid bucketing ensures spatial distribution.
    // ================================================================
    std::sort(corners.begin(), corners.end(),
              [](const Cand& a, const Cand& b) { return a.R > b.R; });

    for (auto& c : corners) {
        if ((int)selected.size() >= max_points) break;
        if (gridFull(c.px, c.py)) continue;
        selected.push_back(c.rp_idx);
        gridAdd(c.px, c.py);
    }

    // ================================================================
    // Phase 2: Edges via precision-weighted D-optimal greedy selection.
    // det(I_current + g_i * J_i^T J_i) where g_i = gradient magnitude.
    // High-contrast edges get larger information contribution.
    // Grid bucketing replaces fixed d_min.
    // ================================================================

    // Build current weighted J^T J from selected corners
    float Iw[3][3] = {};
    for (int idx : selected) {
        SensConstraint sc;
        if (!buildConstraint(*this, idx, sc)) continue;
        float nx2 = sc.normal.x, ny2 = sc.normal.y;
        float j0 = -sc.src.y * nx2 + sc.src.x * ny2, j1 = nx2, j2 = ny2;
        Iw[0][0]+=j0*j0; Iw[0][1]+=j0*j1; Iw[0][2]+=j0*j2;
        Iw[1][1]+=j1*j1; Iw[1][2]+=j1*j2;
        Iw[2][2]+=j2*j2;
    }
    Iw[1][0]=Iw[0][1]; Iw[2][0]=Iw[0][2]; Iw[2][1]=Iw[1][2];
    for (int i=0;i<3;i++) Iw[i][i] += kSolverRegularization;

    // 3x3 determinant helper
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

            // Precision-weighted trial: I_trial = I_current + g_i * J^T J
            float g = std::max(1.0f, e.grad_mag);  // clamp to avoid zero weight
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

        // Update I_w
        float g = std::max(1.0f, e.grad_mag);
        Iw[0][0]+=g*e.j0*e.j0; Iw[0][1]+=g*e.j0*e.j1; Iw[0][2]+=g*e.j0*e.j2;
        Iw[1][1]+=g*e.j1*e.j1; Iw[1][2]+=g*e.j1*e.j2;
        Iw[2][2]+=g*e.j2*e.j2;
        Iw[1][0]=Iw[0][1]; Iw[2][0]=Iw[0][2]; Iw[2][1]=Iw[1][2];
    }

    if ((int)selected.size() < 3) {
        for (int idx : selected)
            result.push_back(cv::Point2f(refine_points[idx].px, refine_points[idx].py));
        return result;
    }

    for (int idx : selected)
        result.push_back(cv::Point2f(refine_points[idx].px, refine_points[idx].py));
    cached_opt_points = result;
    cached_opt_max_points = max_points;
    return result;
}

// ================================================================
// V3: Sensitivity-verified D-optimal selection.
// Uses V1's proven two-phase selection, then verifies via sensitivity
// analysis. If the worst-case point exceeds threshold, swaps it with
// the best alternative from the candidate pool. This directly addresses
// the "coin toss" problem: bad selections are caught and fixed.
// ================================================================
std::vector<cv::Point2f> FeatureSet::selectOptimizedPointsV3(int max_points, float noise_sigma) const {
    (void)noise_sigma;  // reserved for future noise-aware scoring

    // Start with V1's selection (proven good on average)
    // Temporarily disable cache so we get a fresh selection
    auto saved_cache = cached_opt_points;
    auto saved_max = cached_opt_max_points;
    cached_opt_points.clear();
    cached_opt_max_points = 0;

    std::vector<cv::Point2f> result = selectOptimizedPoints(max_points);

    // Restore cache state (V1 will have cached its result)
    // We'll override with our improved version if we find one

    if ((int)result.size() < 3)
        return result;

    // Build candidate pool: all valid refine_points not in the current selection
    float tcx = templ_width / 2.0f, tcy = templ_height / 2.0f;
    struct SwapCand {
        int rp_idx;
        float px, py;
    };
    std::vector<SwapCand> pool;
    for (size_t i = 0; i < refine_points.size(); i++) {
        auto& rp = refine_points[i];
        if (std::abs(rp.px) > templ_width/2.0f - 5 || std::abs(rp.py) > templ_height/2.0f - 5)
            continue;
        SensConstraint sc;
        if (!buildConstraint(*this, (int)i, sc)) continue;

        // Check not already selected
        bool in_sel = false;
        for (auto& pt : result) {
            if (std::abs(pt.x - rp.px) < 0.5f && std::abs(pt.y - rp.py) < 0.5f)
                { in_sel = true; break; }
        }
        if (!in_sel)
            pool.push_back({(int)i, rp.px, rp.py});
    }

    // Sensitivity-guided swap refinement:
    // Evaluate current set, find the most sensitive point, try replacing
    // it with each candidate, keep the swap that minimizes worst sensitivity.
    static const int kMaxSwapIters = 10;
    static const float kSensThreshold = 0.5f;  // target: worst_ang < 0.5 deg/px

    for (int iter = 0; iter < kMaxSwapIters; iter++) {
        // Compute sensitivity of current selection
        float worst_sens = computeMaxSens(buildConstraintSet(*this, result));

        if (worst_sens < kSensThreshold)
            break;  // good enough

        // Find which point has highest individual sensitivity
        // by trying skip_index on each
        int worst_idx = -1;
        float worst_individual = -1;

        for (int fi = 0; fi < (int)result.size(); fi++) {
            // Compute sensitivity with this point removed
            std::vector<cv::Point2f> reduced = result;
            reduced.erase(reduced.begin() + fi);
            if ((int)reduced.size() < 3) continue;

            float sens_without = computeMaxSens(buildConstraintSet(*this, reduced));
            // The point that causes the biggest drop when removed is the most problematic
            float contribution = worst_sens - sens_without;
            if (contribution > worst_individual) {
                worst_individual = contribution;
                worst_idx = fi;
            }
        }

        if (worst_idx < 0) break;

        // Try replacing worst_idx with each candidate from pool
        float best_swap_sens = worst_sens;
        int best_swap_pool = -1;

        for (int ci = 0; ci < (int)pool.size(); ci++) {
            auto trial = result;
            trial[worst_idx] = cv::Point2f(pool[ci].px, pool[ci].py);
            float trial_sens = computeMaxSens(buildConstraintSet(*this, trial));
            if (trial_sens < best_swap_sens * kMinImprovementFactor) {
                best_swap_sens = trial_sens;
                best_swap_pool = ci;
            }
        }

        if (best_swap_pool < 0)
            break;  // no improvement found

        // Perform the swap
        cv::Point2f old_pt = result[worst_idx];
        result[worst_idx] = cv::Point2f(pool[best_swap_pool].px, pool[best_swap_pool].py);

        // Move swapped-out point to pool, remove swapped-in point from pool
        pool.push_back({-1, old_pt.x, old_pt.y});
        pool.erase(pool.begin() + best_swap_pool);
    }

    // Cache the result
    cached_opt_points = result;
    cached_opt_max_points = max_points;
    return result;
}

// ================================================================
// Multi-start hat matrix leverage swap selection
// Runs K random restarts of Fedorov exchange, picks the set with
// lowest worst_ang sensitivity. Each restart is fast because the
// hat matrix diagonal h_ii = J_i (J^T J)^{-1} J_i^T identifies
// which features to swap in O(N) instead of brute-force.
// ================================================================
std::vector<cv::Point2f> FeatureSet::selectOptimizedPointsV2(int max_points, int num_restarts) const {
    if (refine_points.empty() || templ_image.empty())
        return {};

    float tcx = templ_width / 2.0f, tcy = templ_height / 2.0f;

    // --- Collect ALL valid candidates with Jacobian rows ---
    struct Cand {
        int rp_idx;
        float px, py;
        float j[3];        // Jacobian row: [-y*nx+x*ny, nx, ny]
        float grad_mag;    // gradient magnitude
        bool is_corner;
    };
    std::vector<Cand> all_cands;

    cv::Mat templ_blur;
    cv::GaussianBlur(templ_image, templ_blur, cv::Size(3, 3), 1.0);

    for (size_t i = 0; i < refine_points.size(); i++) {
        auto& rp = refine_points[i];
        if (std::abs(rp.px) > templ_width/2.0f - 5 || std::abs(rp.py) > templ_height/2.0f - 5)
            continue;

        SensConstraint sc;
        if (!buildConstraint(*this, (int)i, sc)) continue;
        float nx = sc.normal.x, ny = sc.normal.y;

        int tx = (int)(rp.px + tcx + 0.5f), ty = (int)(rp.py + tcy + 0.5f);
        int h = 8;
        if (tx-h<0||tx+h>=templ_blur.cols||ty-h<0||ty+h>=templ_blur.rows)
            h = std::min({tx, ty, templ_blur.cols-1-tx, templ_blur.rows-1-ty});
        if (h < 3) continue;

        cv::Mat roi = templ_blur(cv::Rect(tx-h, ty-h, 2*h, 2*h));
        cv::Mat dx, dy;
        cv::Sobel(roi, dx, CV_32F, 1, 0, 3);
        cv::Sobel(roi, dy, CV_32F, 0, 1, 3);

        float mag_sum = 0; int n_pix = 0;
        float m00=0, m01=0, m11=0;
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
        cd.j[0] = -rp.py * nx + rp.px * ny;
        cd.j[1] = nx; cd.j[2] = ny;
        cd.grad_mag = n_pix > 0 ? mag_sum / n_pix : 0;
        cd.is_corner = (rp.cornerness > kCornerThreshold && lam2 > 5.0f);
        all_cands.push_back(cd);
    }

    int M = (int)all_cands.size();
    int N = std::min(max_points, M);
    if (N < 3) return {};

    // --- Helper: compute 3x3 inverse of J^T J ---
    auto computeJTJinv = [&](const std::vector<int>& sel, float inv[3][3]) -> bool {
        float A[3][3] = {};
        for (int ci : sel) {
            auto& c = all_cands[ci];
            A[0][0]+=c.j[0]*c.j[0]; A[0][1]+=c.j[0]*c.j[1]; A[0][2]+=c.j[0]*c.j[2];
            A[1][1]+=c.j[1]*c.j[1]; A[1][2]+=c.j[1]*c.j[2];
            A[2][2]+=c.j[2]*c.j[2];
        }
        A[1][0]=A[0][1]; A[2][0]=A[0][2]; A[2][1]=A[1][2];
        for (int i=0;i<3;i++) A[i][i] += kSolverRegularization;

        // 3x3 inverse via cofactors
        float det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                  - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
                  + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
        if (std::abs(det) < 1e-12f) return false;
        float id = 1.0f / det;
        inv[0][0]=(A[1][1]*A[2][2]-A[1][2]*A[2][1])*id;
        inv[0][1]=(A[0][2]*A[2][1]-A[0][1]*A[2][2])*id;
        inv[0][2]=(A[0][1]*A[1][2]-A[0][2]*A[1][1])*id;
        inv[1][0]=inv[0][1];
        inv[1][1]=(A[0][0]*A[2][2]-A[0][2]*A[2][0])*id;
        inv[1][2]=(A[0][2]*A[1][0]-A[0][0]*A[1][2])*id;
        inv[2][0]=inv[0][2];
        inv[2][1]=inv[1][2];
        inv[2][2]=(A[0][0]*A[1][1]-A[0][1]*A[1][0])*id;
        return true;
    };

    // --- Helper: hat matrix diagonal h_ii = J_i * (J^T J)^{-1} * J_i^T ---
    auto hatDiag = [&](const Cand& c, float inv[3][3]) -> float {
        // h = J * inv * J^T where J is 1x3 row vector
        float v[3];
        for (int k=0;k<3;k++)
            v[k] = inv[k][0]*c.j[0] + inv[k][1]*c.j[1] + inv[k][2]*c.j[2];
        return c.j[0]*v[0] + c.j[1]*v[1] + c.j[2]*v[2];
    };

    // --- Helper: compute worst_ang sensitivity for a selected set ---
    auto computeWorstAng = [&](const std::vector<int>& sel) -> float {
        std::vector<SensConstraint> cs;
        for (int ci : sel) {
            auto& c = all_cands[ci];
            SensConstraint sc;
            if (buildConstraint(*this, c.rp_idx, sc))
                cs.push_back(sc);
        }
        if ((int)cs.size() < 3) return 1e9f;
        return computeMaxSens(cs);
    };

    // --- Multi-start Fedorov exchange ---
    std::vector<int> best_sel;
    float best_worst_ang = 1e9f;

    // Seed strategies:
    //   restart 0: corners-first greedy (same as current Phase 1)
    //   restart 1: edges-first by grad_mag
    //   restart 2: spatially spread (farthest-point sampling)
    //   restart 3+: random subsets

    // Precompute corner/edge indices
    std::vector<int> corner_idx, edge_idx;
    for (int i = 0; i < M; i++) {
        if (all_cands[i].is_corner) corner_idx.push_back(i);
        else edge_idx.push_back(i);
    }

    for (int restart = 0; restart < num_restarts; restart++) {
        std::vector<int> sel;
        std::vector<bool> in_set(M, false);

        if (restart == 0) {
            // Strategy: corners first sorted by distance from center (leverage)
            auto ci = corner_idx;
            std::sort(ci.begin(), ci.end(), [&](int a, int b) {
                float ra = all_cands[a].px*all_cands[a].px + all_cands[a].py*all_cands[a].py;
                float rb = all_cands[b].px*all_cands[b].px + all_cands[b].py*all_cands[b].py;
                return ra > rb;
            });
            for (int i : ci) { if ((int)sel.size() >= N) break; sel.push_back(i); in_set[i]=true; }
            // Fill remaining with edges by grad_mag
            auto ei = edge_idx;
            std::sort(ei.begin(), ei.end(), [&](int a, int b) {
                return all_cands[a].grad_mag > all_cands[b].grad_mag;
            });
            for (int i : ei) { if ((int)sel.size() >= N) break; if (!in_set[i]) { sel.push_back(i); in_set[i]=true; } }
        } else if (restart == 1) {
            // Strategy: edges first sorted by grad_mag, then corners
            auto ei = edge_idx;
            std::sort(ei.begin(), ei.end(), [&](int a, int b) {
                return all_cands[a].grad_mag > all_cands[b].grad_mag;
            });
            for (int i : ei) { if ((int)sel.size() >= N) break; sel.push_back(i); in_set[i]=true; }
            auto ci = corner_idx;
            std::sort(ci.begin(), ci.end(), [&](int a, int b) {
                float ra = all_cands[a].px*all_cands[a].px + all_cands[a].py*all_cands[a].py;
                float rb = all_cands[b].px*all_cands[b].px + all_cands[b].py*all_cands[b].py;
                return ra > rb;
            });
            for (int i : ci) { if ((int)sel.size() >= N) break; if (!in_set[i]) { sel.push_back(i); in_set[i]=true; } }
        } else if (restart == 2) {
            // Strategy: farthest-point sampling for spatial spread
            // Start from the candidate closest to center
            int first = 0;
            float best_d = 1e9f;
            for (int i = 0; i < M; i++) {
                float d = all_cands[i].px*all_cands[i].px + all_cands[i].py*all_cands[i].py;
                if (d < best_d) { best_d = d; first = i; }
            }
            sel.push_back(first); in_set[first] = true;
            while ((int)sel.size() < N) {
                float max_min_d = -1; int best_ci = -1;
                for (int i = 0; i < M; i++) {
                    if (in_set[i]) continue;
                    float min_d = 1e9f;
                    for (int si : sel) {
                        float dx = all_cands[i].px - all_cands[si].px;
                        float dy = all_cands[i].py - all_cands[si].py;
                        min_d = std::min(min_d, dx*dx+dy*dy);
                    }
                    if (min_d > max_min_d) { max_min_d = min_d; best_ci = i; }
                }
                if (best_ci < 0) break;
                sel.push_back(best_ci); in_set[best_ci] = true;
            }
        } else {
            // Strategy: deterministic pseudo-random (different permutation per restart)
            // Use a simple LCG seeded by restart index for reproducibility
            unsigned seed = 7919u * (unsigned)restart + 31u;
            auto lcg = [&]() -> unsigned { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0x7FFF; };
            // Shuffle indices and take first N
            std::vector<int> perm(M);
            for (int i = 0; i < M; i++) perm[i] = i;
            for (int i = M-1; i > 0; i--) {
                int j = lcg() % (i+1);
                std::swap(perm[i], perm[j]);
            }
            for (int i = 0; i < N && i < M; i++) {
                sel.push_back(perm[i]);
                in_set[perm[i]] = true;
            }
        }

        if ((int)sel.size() < N) {
            // Fill any remaining from unused candidates
            for (int i = 0; i < M && (int)sel.size() < N; i++) {
                if (!in_set[i]) { sel.push_back(i); in_set[i] = true; }
            }
        }

        // --- Fedorov exchange using hat matrix ---
        for (int iter = 0; iter < 50; iter++) {
            float inv[3][3];
            if (!computeJTJinv(sel, inv)) break;

            // Find selected point with highest hat diagonal (most sensitive)
            int worst_si = -1; float worst_h = -1;
            for (int k = 0; k < (int)sel.size(); k++) {
                float h = hatDiag(all_cands[sel[k]], inv);
                if (h > worst_h) { worst_h = h; worst_si = k; }
            }

            // Find unselected candidate with highest hat diagonal (most informative to add)
            int best_ui = -1; float best_h = -1;
            for (int i = 0; i < M; i++) {
                if (in_set[i]) continue;
                float h = hatDiag(all_cands[i], inv);
                if (h > best_h) { best_h = h; best_ui = i; }
            }

            // Swap if it improves: new candidate must have higher leverage than worst selected
            if (worst_si < 0 || best_ui < 0 || best_h <= worst_h * 1.01f)
                break;  // converged

            in_set[sel[worst_si]] = false;
            in_set[best_ui] = true;
            sel[worst_si] = best_ui;
        }

        // Evaluate this restart's sensitivity
        float wa = computeWorstAng(sel);
        if (wa < best_worst_ang) {
            best_worst_ang = wa;
            best_sel = sel;
        }
    }

    // Convert to result
    std::vector<cv::Point2f> result;
    for (int ci : best_sel)
        result.push_back(cv::Point2f(all_cands[ci].px, all_cands[ci].py));
    return result;
}

FeatureSet::SensitivityReport FeatureSet::analyzeSensitivity(int skip_index) const {
    SensitivityReport sr;
    sr.worst_angle_sens = 0;
    sr.worst_pos_sens = 0;
    sr.mean_angle_sens = 0;
    sr.mean_pos_sens = 0;
    sr.num_fragile = 0;

    // Use optimized point selection
    auto opt_points = selectOptimizedPoints(kDefaultOptPoints);
    if ((int)opt_points.size() < 3) {
        sr.diagnosis = "Too few points";
        return sr;
    }

    // Build constraints from optimized points
    // Find matching refine_point for each optimized point
    auto buildSet = [&](const std::vector<cv::Point2f>& pts) -> std::vector<SensConstraint> {
        std::vector<SensConstraint> cs;
        for (auto& pt : pts) {
            // Find closest refine_point
            int best = -1; float best_d = 1e9f;
            for (size_t i = 0; i < refine_points.size(); i++) {
                float dx = pt.x - refine_points[i].px, dy = pt.y - refine_points[i].py;
                float d = dx*dx + dy*dy;
                if (d < best_d) { best_d = d; best = (int)i; }
            }
            if (best >= 0) {
                SensConstraint c;
                if (buildConstraint(*this, best, c))
                    cs.push_back(c);
            }
        }
        return cs;
    };

    auto points = opt_points;
    if (skip_index >= 0 && skip_index < (int)points.size())
        points.erase(points.begin() + skip_index);

    auto baseline = buildSet(points);
    if ((int)baseline.size() < 3) {
        sr.diagnosis = "Too few valid points";
        return sr;
    }

    cv::Vec3f base_pose = solveSens(baseline);

    // Perturb each feature's dst by +1px in X and Y, measure pose change
    for (size_t fi = 0; fi < baseline.size(); ++fi) {
        FeatureSensitivity fs;
        fs.pos = baseline[fi].src;

        auto perturbed = baseline;
        perturbed[fi].dst.x += 1.0f;
        cv::Vec3f pose_dx = solveSens(perturbed);
        float ang_from_dx = std::abs(pose_dx[0] - base_pose[0]);
        float px_from_dx = std::sqrt((pose_dx[1]-base_pose[1])*(pose_dx[1]-base_pose[1]) +
                                      (pose_dx[2]-base_pose[2])*(pose_dx[2]-base_pose[2]));

        perturbed = baseline;
        perturbed[fi].dst.y += 1.0f;
        cv::Vec3f pose_dy = solveSens(perturbed);
        float ang_from_dy = std::abs(pose_dy[0] - base_pose[0]);
        float px_from_dy = std::sqrt((pose_dy[1]-base_pose[1])*(pose_dy[1]-base_pose[1]) +
                                      (pose_dy[2]-base_pose[2])*(pose_dy[2]-base_pose[2]));

        fs.d_ang = ang_from_dx + ang_from_dy;
        fs.d_pos = std::sqrt(px_from_dx*px_from_dx + px_from_dy*px_from_dy);
        fs.leverage = std::sqrt(fs.pos.x * fs.pos.x + fs.pos.y * fs.pos.y);

        sr.features.push_back(fs);
        sr.worst_angle_sens = std::max(sr.worst_angle_sens, fs.d_ang);
        sr.worst_pos_sens = std::max(sr.worst_pos_sens, fs.d_pos);
        sr.mean_angle_sens += fs.d_ang;
        sr.mean_pos_sens += fs.d_pos;
        if (fs.d_ang > 1.0f) sr.num_fragile++;
    }

    sr.mean_angle_sens /= sr.features.size();
    sr.mean_pos_sens /= sr.features.size();

    if (sr.worst_angle_sens < 0.5f)
        sr.diagnosis = "Robust — all features stable";
    else if (sr.worst_angle_sens < 1.0f)
        sr.diagnosis = "Good — minor sensitivity in " + std::to_string(sr.num_fragile) + " features";
    else
        sr.diagnosis = "Fragile — " + std::to_string(sr.num_fragile) +
                       " features cause >" + std::to_string((int)sr.worst_angle_sens) + " deg/px";

    return sr;
}

// ============================================================
// Geometric constraint quality analysis
// ============================================================
FeatureSet::ConstraintAnalysis FeatureSet::analyzeConstraints(const std::vector<cv::Point2f>& custom_points) const {
    ConstraintAnalysis ca = {};

    auto opt_points = custom_points.empty() ? selectOptimizedPoints(kDefaultOptPoints) : custom_points;
    if ((int)opt_points.size() < 3) {
        ca.summary = "Too few points for analysis";
        return ca;
    }
    ca.num_points = (int)opt_points.size();

    // Build constraints and collect per-point info.
    // Uses PCA on each point's ROI patch (same as refineROI) to determine
    // edge (1 constraint: normal) vs corner (2 constraints: normal + tangent).
    float tcx = templ_width / 2.0f, tcy = templ_height / 2.0f;
    // Locking ratio threshold: if Hessian eigenvalue ratio < this, the patch
    // locks in 2D (corner/dot). If above, only 1D locking (edge).
    // Lock ratio threshold: ratio < this → 2D lock. Ratio of ~1 = perfect isotropic lock
    // (dot), ~3-7 = wide-angle corner, ~50+ = pure edge. Threshold of 10 catches
    // triangle vertices (ratio ~6-7) which still provide meaningful 2D constraint.
    static const float kLockRatioThreshold = 10.0f;

    struct JRow { float j[3]; cv::Point2f pos, normal; float leverage; bool is_corner; int rp_idx; int point_idx; };
    std::vector<JRow> rows;

    // Track which rows belong to which point (for per-point info later)
    struct PointEntry { cv::Point2f pos; float leverage; bool is_corner; int rp_idx; int first_row; int num_rows; float lock_major, lock_minor, lock_ratio; };
    std::vector<PointEntry> point_entries;

    for (int pi = 0; pi < (int)opt_points.size(); pi++) {
        auto& pt = opt_points[pi];

        // Use the requested position directly for PCA (not snapped to refine_point).
        // This matches what refineROI does: extract ROI at the exact sample position.
        float px = pt.x, py = pt.y;
        float leverage = std::sqrt(px*px + py*py);
        int tx = (int)(px + tcx + 0.5f), ty = (int)(py + tcy + 0.5f);
        int best = -1;  // for cornerness lookup only
        {
            float best_d = 1e9f;
            for (size_t i = 0; i < refine_points.size(); i++) {
                float dx2 = pt.x - refine_points[i].px, dy2 = pt.y - refine_points[i].py;
                float d = dx2*dx2 + dy2*dy2;
                if (d < best_d) { best_d = d; best = (int)i; }
            }
        }

        // Extract ROI patch and measure 2D locking via matchTemplate Hessian.
        // The Hessian eigenvalues of the correlation surface at the peak tell us
        // how well this patch locks in each direction:
        //   Both large → 2D lock (dot, corner) → 2 constraints
        //   One large, one small → 1D lock (edge) → 1 constraint
        int h = kDefaultROIHalf;
        if (tx-h<0||tx+h>=templ_image.cols||ty-h<0||ty+h>=templ_image.rows)
            h = std::min({tx, ty, templ_image.cols-1-tx, templ_image.rows-1-ty});
        if (h < 5) continue;

        cv::Mat roi_patch = templ_image(cv::Rect(tx-h, ty-h, 2*h+1, 2*h+1));

        // Run matchTemplate of this patch against a padded region of the template.
        // Use full ROI half as search range so the response surface is large enough
        // to properly measure locking curvature.
        int search_extra = h;
        int sx0 = std::max(0, tx - h - search_extra);
        int sy0 = std::max(0, ty - h - search_extra);
        int sx1 = std::min(templ_image.cols, tx + h + 1 + search_extra);
        int sy1 = std::min(templ_image.rows, ty + h + 1 + search_extra);

        if (sx1 - sx0 < roi_patch.cols || sy1 - sy0 < roi_patch.rows) continue;

        cv::Mat search_region = templ_image(cv::Rect(sx0, sy0, sx1 - sx0, sy1 - sy0));
        cv::Mat response;
        cv::matchTemplate(search_region, roi_patch, response, cv::TM_CCORR_NORMED);

        if (response.rows < 3 || response.cols < 3) continue;

        // Find peak
        double peak_val;
        cv::Point max_loc;
        cv::minMaxLoc(response, nullptr, &peak_val, nullptr, &max_loc);
        int mx = max_loc.x, my = max_loc.y;

        // Measure 2D locking by analyzing the correlation response surface.
        // Instead of local Hessian (only ±1px), measure the structure tensor of
        // the response surface — how the drop-off behaves over the full window.
        // Compute Σ(dR/dx)² , Σ(dR/dx)(dR/dy), Σ(dR/dy)² over the response surface.
        // Eigenvalues = how sharply correlation drops in each principal direction.
        // This is the autocorrelation / Harris approach applied to the match surface.
        float sxx = 0, sxy = 0, syy = 0;
        float lock_lam1 = 0, lock_lam2 = 0;
        cv::Point2f evec0(1, 0), evec1(0, 1);

        {
            // Compute gradients of the response surface
            cv::Mat resp_dx, resp_dy;
            cv::Sobel(response, resp_dx, CV_32F, 1, 0, 3);
            cv::Sobel(response, resp_dy, CV_32F, 0, 1, 3);

            // Weight by proximity to peak (Gaussian window centered on peak)
            // so distant response doesn't dominate
            float sigma_w = std::max(3.0f, (float)search_extra * 0.5f);
            float inv_2s2 = 1.0f / (2.0f * sigma_w * sigma_w);

            for (int ry = 0; ry < response.rows; ry++) {
                const float* gx = resp_dx.ptr<float>(ry);
                const float* gy = resp_dy.ptr<float>(ry);
                float dy2 = (float)(ry - my);
                for (int rx = 0; rx < response.cols; rx++) {
                    float dx2 = (float)(rx - mx);
                    float w = std::exp(-(dx2*dx2 + dy2*dy2) * inv_2s2);
                    sxx += w * gx[rx] * gx[rx];
                    sxy += w * gx[rx] * gy[rx];
                    syy += w * gy[rx] * gy[rx];
                }
            }

            // Eigenvalues of structure tensor [[sxx, sxy], [sxy, syy]]
            float tr = sxx + syy;
            float disc = std::sqrt(std::max(0.0f, (sxx-syy)*(sxx-syy)/4.0f + sxy*sxy));
            lock_lam1 = tr/2.0f + disc;  // larger (strongest drop-off direction)
            lock_lam2 = tr/2.0f - disc;  // smaller (weakest drop-off direction)

            // Eigenvectors (directions of steepest / shallowest drop-off)
            // Use axis-aligned fallback when off-diagonal is small relative to diagonal,
            // avoiding degenerate near-zero eigenvectors from (lam - syy, sxy) ≈ (0, 0).
            float diag_diff = std::abs(sxx - syy);
            if (std::abs(sxy) > 0.01f * (diag_diff + 1e-6f)) {
                evec0 = cv::Point2f(lock_lam1 - syy, sxy);
                evec1 = cv::Point2f(lock_lam2 - syy, sxy);
            } else {
                evec0 = (sxx >= syy) ? cv::Point2f(1, 0) : cv::Point2f(0, 1);
                evec1 = (sxx >= syy) ? cv::Point2f(0, 1) : cv::Point2f(1, 0);
            }
            float len0 = std::sqrt(evec0.x*evec0.x + evec0.y*evec0.y);
            float len1 = std::sqrt(evec1.x*evec1.x + evec1.y*evec1.y);
            if (len0 > 1e-6f) evec0 *= (1.0f / len0);
            else evec0 = cv::Point2f(1, 0);
            if (len1 > 1e-6f) evec1 *= (1.0f / len1);
            else evec1 = cv::Point2f(0, 1);
        }

        lock_lam1 = std::max(0.0f, lock_lam1);
        lock_lam2 = std::max(0.0f, lock_lam2);

        float lock_ratio = lock_lam1 / std::max(1e-8f, lock_lam2);
        bool is_2d_lock = (lock_lam2 > 1e-6f && lock_ratio < kLockRatioThreshold);

        PointEntry pe;
        pe.pos = cv::Point2f(px, py);
        pe.leverage = leverage;
        pe.is_corner = is_2d_lock;
        pe.rp_idx = best;
        pe.first_row = (int)rows.size();
        pe.lock_major = lock_lam1;
        pe.lock_minor = lock_lam2;
        pe.lock_ratio = lock_ratio;

        // Primary constraint: direction of strongest curvature (evec0)
        // For an edge, this is the edge normal. For a dot/corner, both directions lock.
        JRow r1;
        r1.pos = cv::Point2f(px, py);
        r1.normal = evec0;
        r1.leverage = leverage;
        r1.is_corner = is_2d_lock;
        r1.rp_idx = best;
        r1.point_idx = pi;
        r1.j[0] = -py * evec0.x + px * evec0.y;
        r1.j[1] = evec0.x;
        r1.j[2] = evec0.y;
        rows.push_back(r1);

        if (is_2d_lock) {
            // Secondary constraint: direction of weaker (but significant) curvature
            JRow r2;
            r2.pos = cv::Point2f(px, py);
            r2.normal = evec1;
            r2.leverage = leverage;
            r2.is_corner = true;
            r2.rp_idx = best;
            r2.point_idx = pi;
            r2.j[0] = -py * evec1.x + px * evec1.y;
            r2.j[1] = evec1.x;
            r2.j[2] = evec1.y;
            rows.push_back(r2);
        }

        pe.num_rows = (int)rows.size() - pe.first_row;
        point_entries.push_back(pe);
    }

    int N = (int)rows.size();
    if (N < 3) {
        ca.summary = "Too few valid constraints";
        return ca;
    }

    // Build J^T J (Fisher information matrix)
    float JTJ[3][3] = {};
    for (auto& r : rows) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                JTJ[i][j] += r.j[i] * r.j[j];
    }

    // Eigenvalues of 3x3 symmetric matrix via cv::eigen
    cv::Mat JTJ_mat(3, 3, CV_32F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            JTJ_mat.at<float>(i, j) = JTJ[i][j];

    cv::Mat eigvals_mat, eigvecs_mat;
    cv::eigen(JTJ_mat, eigvals_mat, eigvecs_mat);

    ca.info_eigenvalues[0] = eigvals_mat.at<float>(0);
    ca.info_eigenvalues[1] = eigvals_mat.at<float>(1);
    ca.info_eigenvalues[2] = eigvals_mat.at<float>(2);

    float lam_max = ca.info_eigenvalues[0];
    float lam_min = std::max(1e-10f, ca.info_eigenvalues[2]);
    ca.condition_number = lam_max / lam_min;

    // Invert J^T J to get covariance: Cov = (J^TJ)^{-1}
    // Diagonal elements give per-DOF variance under unit noise
    cv::Mat JTJ_inv;
    cv::invert(JTJ_mat, JTJ_inv, cv::DECOMP_SVD);

    // sigma_theta: sqrt of (0,0) element, convert radians to degrees
    ca.sigma_theta = std::sqrt(std::max(0.0f, JTJ_inv.at<float>(0, 0))) * 180.0f / (float)CV_PI;
    ca.sigma_tx = std::sqrt(std::max(0.0f, JTJ_inv.at<float>(1, 1)));
    ca.sigma_ty = std::sqrt(std::max(0.0f, JTJ_inv.at<float>(2, 2)));

    // Position error ellipse from the (tx,ty) 2x2 sub-block of Cov
    float cov_xx = JTJ_inv.at<float>(1, 1);
    float cov_xy = JTJ_inv.at<float>(1, 2);
    float cov_yy = JTJ_inv.at<float>(2, 2);

    float trace2 = cov_xx + cov_yy;
    float disc2 = std::sqrt(std::max(0.0f, (cov_xx-cov_yy)*(cov_xx-cov_yy)/4.0f + cov_xy*cov_xy));
    ca.ellipse_major = std::sqrt(std::max(0.0f, trace2/2.0f + disc2));
    ca.ellipse_minor = std::sqrt(std::max(0.0f, trace2/2.0f - disc2));
    ca.ellipse_angle = 0.5f * std::atan2(2*cov_xy, cov_xx - cov_yy) * 180.0f / (float)CV_PI;

    // Normal spread: angular range of normal directions
    // Measure as the angular spread (max angle between any two normals)
    float max_spread = 0;
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            float dot = rows[i].normal.x * rows[j].normal.x + rows[i].normal.y * rows[j].normal.y;
            dot = std::max(-1.0f, std::min(1.0f, dot));
            float angle = std::acos(std::abs(dot)) * 180.0f / (float)CV_PI;
            max_spread = std::max(max_spread, angle);
        }
    }
    ca.normal_spread = max_spread;

    // Spatial spread: RMS distance from centroid (per unique point, not per row)
    int NP = (int)point_entries.size();
    float cx = 0, cy = 0;
    for (auto& pe : point_entries) { cx += pe.pos.x; cy += pe.pos.y; }
    cx /= NP; cy /= NP;
    float rms = 0;
    for (auto& pe : point_entries) {
        float ddx = pe.pos.x - cx, ddy = pe.pos.y - cy;
        rms += ddx*ddx + ddy*ddy;
    }
    ca.spatial_spread = std::sqrt(rms / NP);

    // Mean leverage + corner/edge count (per unique point)
    float lev_sum = 0;
    ca.num_corners = 0; ca.num_edges = 0;
    for (auto& pe : point_entries) {
        lev_sum += pe.leverage;
        if (pe.is_corner) ca.num_corners++; else ca.num_edges++;
    }
    ca.mean_leverage = lev_sum / NP;

    // Per-point information contribution: det(I) / det(I_without_point)
    // For corners, removing a point removes BOTH its normal and tangent rows
    float det_full = (float)cv::determinant(JTJ_mat);
    for (auto& pe : point_entries) {
        ConstraintAnalysis::PointInfo pi;
        pi.pos = pe.pos;
        pi.normal = rows[pe.first_row].normal;  // primary normal
        pi.leverage = pe.leverage;
        pi.is_corner = pe.is_corner;

        // Build J^TJ without ALL rows from this point
        float JTJ_reduced[3][3] = {};
        for (int ri = 0; ri < (int)rows.size(); ri++) {
            if (ri >= pe.first_row && ri < pe.first_row + pe.num_rows) continue;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    JTJ_reduced[i][j] += rows[ri].j[i] * rows[ri].j[j];
        }
        cv::Mat red_mat(3, 3, CV_32F);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                red_mat.at<float>(i, j) = JTJ_reduced[i][j];
        float det_reduced = (float)cv::determinant(red_mat);
        pi.info_contribution = (det_reduced > 1e-10f) ? det_full / det_reduced : 1e6f;
        pi.lock_major = pe.lock_major;
        pi.lock_minor = pe.lock_minor;
        pi.lock_ratio = pe.lock_ratio;

        ca.points.push_back(pi);
    }

    // Summary
    char buf[512];
    snprintf(buf, sizeof(buf),
        "%d pts (%d corners, %d edges, %d constraints) | "
        "cond=%.1f | "
        "sigma: theta=%.3f deg, tx=%.3f px, ty=%.3f px | "
        "ellipse: %.3f x %.3f px @ %.0f deg | "
        "normal_spread=%.0f deg, spatial_spread=%.0f px, mean_leverage=%.0f px",
        ca.num_points, ca.num_corners, ca.num_edges, N,
        ca.condition_number,
        ca.sigma_theta, ca.sigma_tx, ca.sigma_ty,
        ca.ellipse_major, ca.ellipse_minor, ca.ellipse_angle,
        ca.normal_spread, ca.spatial_spread, ca.mean_leverage);
    ca.summary = buf;

    return ca;
}

// ============================================================
// ShapeMatcher::Impl
// ============================================================

struct ModelInfo {
    std::string name;
    FeatureSet features;
    ModelConfig config;
    std::string class_id;           // meiqua class_id (internal)
    std::string class_id_flip;      // flipped variant class_id
    int num_variants = 0;           // total templates for this model
    float angle_bias = 0;           // calibrated bias
};

struct ShapeMatcher::Impl {
    MatchConfig match_config;
    line2Dup::Detector detector;
    std::vector<ModelInfo> models;
    int total_templates = 0;

    Impl(const MatchConfig& cfg)
        : match_config(cfg),
          detector(128, cfg.pyramid_T,
                   cfg.weak_threshold, cfg.strong_threshold) {}

    // Convert FeatureSet to meiqua TemplatePyramid
    static void featureSetToTemplates(const FeatureSet& fs,
                                       std::vector<line2Dup::Template>& tp) {
        tp.resize(fs.levels.size());
        for (size_t i = 0; i < fs.levels.size(); ++i) {
            auto& src = fs.levels[i];
            tp[i].tl_x = src.tl_x;
            tp[i].tl_y = src.tl_y;
            tp[i].width = src.width;
            tp[i].height = src.height;
            tp[i].pyramid_level = src.level;
            tp[i].angle = 0;
            tp[i].features.resize(src.features.size());
            for (size_t j = 0; j < src.features.size(); ++j) {
                tp[i].features[j].x = src.features[j].x;
                tp[i].features[j].y = src.features[j].y;
                tp[i].features[j].label = src.features[j].label;
                tp[i].features[j].theta = src.features[j].theta;
            }
        }
    }

    // Flip features horizontally around template center
    static FeatureSet flipFeatures(const FeatureSet& fs) {
        FeatureSet flipped = fs;
        for (auto& lv : flipped.levels) {
            int center_y = (lv.level == 0) ? fs.templ_height / 2
                           : fs.templ_height / (2 << lv.level);
            for (auto& f : lv.features) {
                int true_y = lv.tl_y + f.y;
                f.y = (2 * center_y - true_y) - lv.tl_y;
                f.theta = -f.theta;
                if (f.theta < 0) f.theta += 360;
                f.label = (int)(f.theta * 16.0f / 360.0f + 0.5f) & 7;
            }
        }
        flipped.origin.y = fs.templ_height - fs.origin.y;
        return flipped;
    }

    // Add templates for one model at one scale
    int addModelAtScale(const std::string& class_id,
                        const FeatureSet& fs,
                        const AngleRange& angle,
                        float scale) {
        // Convert to meiqua templates
        std::vector<line2Dup::Template> base_tp;
        featureSetToTemplates(fs, base_tp);

        // Scale features if needed
        if (std::abs(scale - 1.0f) > 0.001f) {
            for (auto& t : base_tp) {
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

        // Add base (0-degree) template
        auto& tps = detector.getClassTemplates(class_id);
        tps.push_back(base_tp);
        int count = 1;

        // Rotate features for remaining angles
        cv::Point2f center(fs.templ_width * scale / 2.0f,
                           fs.templ_height * scale / 2.0f);
        int pyramid_levels = (int)base_tp.size();

        for (float a = angle.start + angle.step; a < angle.end; a += angle.step) {
            // Rotate feature coordinates + orientation
            std::vector<line2Dup::Template> rot_tp(pyramid_levels);
            float angRad = a * (float)CV_PI / 180.0f;

            for (int l = 0; l < pyramid_levels; ++l) {
                cv::Point2f lvl_center = center;
                for (int i = 0; i < l; ++i) lvl_center *= 0.5f;

                for (auto& f : base_tp[l].features) {
                    cv::Point2f p((float)(f.x + base_tp[l].tl_x),
                                  (float)(f.y + base_tp[l].tl_y));
                    cv::Point2f pr = p - lvl_center;
                    cv::Point2f rot(
                        std::cos(angRad)*pr.x - std::sin(angRad)*pr.y + lvl_center.x,
                        std::sin(angRad)*pr.x + std::cos(angRad)*pr.y + lvl_center.y);

                    line2Dup::Feature fn;
                    fn.x = (int)(rot.x + 0.5f);
                    fn.y = (int)(rot.y + 0.5f);
                    fn.theta = f.theta + a;
                    while (fn.theta >= 360) fn.theta -= 360;
                    while (fn.theta < 0) fn.theta += 360;
                    fn.label = (int)(fn.theta * 16.0f / 360.0f + 0.5f) & 7;
                    rot_tp[l].features.push_back(fn);
                }
                rot_tp[l].pyramid_level = l;
                rot_tp[l].angle = a;
            }

            // Crop bounding box (inline version of cropTemplates)
            for (auto& t : rot_tp) {
                int min_x = INT_MAX, min_y = INT_MAX, max_x = 0, max_y = 0;
                for (auto& f : t.features) {
                    min_x = std::min(min_x, f.x); max_x = std::max(max_x, f.x);
                    min_y = std::min(min_y, f.y); max_y = std::max(max_y, f.y);
                }
                if (t.features.empty()) continue;
                t.tl_x = min_x; t.tl_y = min_y;
                t.width = max_x - min_x; t.height = max_y - min_y;
                for (auto& f : t.features) { f.x -= t.tl_x; f.y -= t.tl_y; }
            }
            tps.push_back(rot_tp);
            count++;
        }
        return count;
    }
};

// ============================================================
// ShapeMatcher public methods
// ============================================================

ShapeMatcher::ShapeMatcher(const MatchConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

ShapeMatcher::~ShapeMatcher() = default;

int ShapeMatcher::addModel(const std::string& name,
                           const FeatureSet& features,
                           const ModelConfig& config) {
    ModelInfo info;
    info.name = name;
    info.features = features;
    // Pre-compute optimized sample points and cache them. These are read-only
    // during match() (parallel region), so no synchronization is needed.
    info.features.selectOptimizedPoints(kDefaultOptPoints);
    // Pre-compute lock info for each sample point (matchTemplate response curvature)
    if (!info.features.templ_image.empty() && !info.features.cached_opt_points.empty()) {
        auto& pts = info.features.cached_opt_points;
        auto& img = info.features.templ_image;
        float tcx_l = info.features.templ_width / 2.0f;
        float tcy_l = info.features.templ_height / 2.0f;
        info.features.cached_lock_info.resize(pts.size());
        for (int pi = 0; pi < (int)pts.size(); pi++) {
            int tx_l = (int)(pts[pi].x + tcx_l + 0.5f);
            int ty_l = (int)(pts[pi].y + tcy_l + 0.5f);
            int h_l = kDefaultROIHalf;
            if (tx_l-h_l<0||ty_l-h_l<0||tx_l+h_l>=img.cols||ty_l+h_l>=img.rows) continue;
            cv::Mat patch = img(cv::Rect(tx_l-h_l, ty_l-h_l, 2*h_l+1, 2*h_l+1));
            int se = h_l;
            int sx0=std::max(0,tx_l-h_l-se), sy0=std::max(0,ty_l-h_l-se);
            int sx1=std::min(img.cols,tx_l+h_l+1+se), sy1=std::min(img.rows,ty_l+h_l+1+se);
            if (sx1-sx0<patch.cols||sy1-sy0<patch.rows) continue;
            cv::Mat search=img(cv::Rect(sx0,sy0,sx1-sx0,sy1-sy0));
            cv::Mat resp; cv::matchTemplate(search,patch,resp,cv::TM_CCORR_NORMED);
            cv::Point ml; cv::minMaxLoc(resp,nullptr,nullptr,nullptr,&ml);
            cv::Mat rdx,rdy; cv::Sobel(resp,rdx,CV_32F,1,0,3); cv::Sobel(resp,rdy,CV_32F,0,1,3);
            float sxx=0,sxy=0,syy=0,sw=std::max(3.f,se*0.5f),inv2s=1.f/(2*sw*sw);
            for(int ry=0;ry<resp.rows;ry++){
                const float*gx=rdx.ptr<float>(ry),*gy=rdy.ptr<float>(ry);
                float dy2=(float)(ry-ml.y);
                for(int rx=0;rx<resp.cols;rx++){
                    float dx2=(float)(rx-ml.x),w=std::exp(-(dx2*dx2+dy2*dy2)*inv2s);
                    sxx+=w*gx[rx]*gx[rx]; sxy+=w*gx[rx]*gy[rx]; syy+=w*gy[rx]*gy[rx];
                }
            }
            float tr=sxx+syy,disc=std::sqrt(std::max(0.f,(sxx-syy)*(sxx-syy)/4+sxy*sxy));
            info.features.cached_lock_info[pi].major = std::max(0.f,tr/2+disc);
            info.features.cached_lock_info[pi].minor = std::max(0.f,tr/2-disc);
            // Compute eigenvectors of structure tensor
            // Eigenvector for larger eigenvalue (major = tangent/slide direction)
            float diag_diff = sxx - syy;
            cv::Point2f ev_major, ev_minor;
            if (std::abs(sxy) > 0.01f * (std::abs(diag_diff) + 1e-6f)) {
                float lam1 = tr/2 + disc;
                ev_major = cv::Point2f(sxy, lam1 - sxx);
                float len = std::sqrt(ev_major.x*ev_major.x + ev_major.y*ev_major.y);
                if (len > 1e-6f) ev_major *= (1.0f / len);
                else ev_major = cv::Point2f(1, 0);
                ev_minor = cv::Point2f(-ev_major.y, ev_major.x);
            } else {
                // Near axis-aligned: sxy ≈ 0
                if (sxx >= syy) {
                    ev_major = cv::Point2f(1, 0);
                    ev_minor = cv::Point2f(0, 1);
                } else {
                    ev_major = cv::Point2f(0, 1);
                    ev_minor = cv::Point2f(1, 0);
                }
            }
            info.features.cached_lock_info[pi].normal = ev_major;   // strongest response gradient = constraint normal (across edge)
            info.features.cached_lock_info[pi].tangent = ev_minor;  // weakest response gradient = along edge (slide)
        }
        // Normalize: max lock_major = 1.0, all others relative
        float max_lock = 0;
        for (auto& li : info.features.cached_lock_info)
            max_lock = std::max(max_lock, li.major);
        if (max_lock > 1e-6f) {
            for (auto& li : info.features.cached_lock_info) {
                li.major /= max_lock;
                li.minor /= max_lock;
                // Floor to prevent zero weight (every point contributes something)
                li.major = std::max(0.1f, li.major);
                li.minor = std::max(0.01f, li.minor);
                // Corner = response surface locks in both directions
                // Use ratio: PCA used eigenval ratio < 2.0 for corners
                li.is_corner = (li.minor > 0.01f && li.major / li.minor < 12.0f);
            }
        }
    }
    // Pre-build template EdgeScene for inverse ICP refinement
    if (!info.features.templ_image.empty()) {
        info.features.cached_templ_scene =
            icp_refine::buildTemplateScene(info.features.templ_image, 20.0f);
        info.features.templ_scene_valid = true;
    }
    info.config = config;
    info.class_id = "sbm_" + name;
    info.class_id_flip = "sbm_" + name + "_flip";

    int count = 0;

    // For each scale
    float s = config.scale.min;
    do {
        count += impl_->addModelAtScale(info.class_id, features, config.angle, s);

        if (config.flip) {
            auto flipped = Impl::flipFeatures(features);
            count += impl_->addModelAtScale(info.class_id_flip, flipped, config.angle, s);
        }

        s += config.scale.step;
    } while (s <= config.scale.max + 0.001f && config.scale.max > config.scale.min);

    info.num_variants = count;
    impl_->models.push_back(info);
    impl_->total_templates += count;
    return count;
}

std::vector<MatchResult> ShapeMatcher::match(const cv::Mat& scene) const {
    auto& cfg = impl_->match_config;

    // Optional scene downscale for faster matching
    cv::Mat match_scene = scene;
    float inv_scale = 1.0f;
    if (cfg.match_scale < 1.0f && cfg.match_scale > 0.1f) {
        inv_scale = 1.0f / cfg.match_scale;
        cv::resize(scene, match_scene,
                   cv::Size((int)(scene.cols * cfg.match_scale),
                            (int)(scene.rows * cfg.match_scale)));
    }

    // Pad to 16
    int pw = (match_scene.cols + 15) & ~15;
    int ph = (match_scene.rows + 15) & ~15;
    cv::Mat padded;
    if (pw != match_scene.cols || ph != match_scene.rows)
        cv::copyMakeBorder(match_scene, padded, 0, ph - match_scene.rows,
                           0, pw - match_scene.cols, cv::BORDER_CONSTANT, cv::Scalar(0));
    else
        padded = match_scene;

    // Collect all class_ids to match
    std::vector<std::string> class_ids;
    for (auto& m : impl_->models) {
        class_ids.push_back(m.class_id);
        if (m.config.flip)
            class_ids.push_back(m.class_id_flip);
    }

    // Apply config to modality
    impl_->detector.getModalities()->blur_kernel_size = cfg.blur_kernel_size;
    impl_->detector.getModalities()->skip_voting = cfg.skip_voting;

    // Run meiqua matching
    auto raw_matches = impl_->detector.match(padded, cfg.min_score, class_ids);

    // NMS — auto radius from template size if not set
    float nms_r = cfg.nms_radius;
    if (nms_r < 0 && !impl_->models.empty()) {
        auto& fs = impl_->models[0].features;
        nms_r = std::min(fs.templ_width, fs.templ_height) / 2.0f;
    }
    if (nms_r < 1) nms_r = 1;
    // Compute angle step for angle-aware NMS
    float nms_angle_step = 2.0f;
    if (!impl_->models.empty())
        nms_angle_step = impl_->models[0].config.angle.step;
    int templates_per_scale = 1;
    if (!impl_->models.empty()) {
        auto& ac = impl_->models[0].config.angle;
        templates_per_scale = std::max(1, (int)((ac.end - ac.start) / ac.step));
    }

    std::vector<line2Dup::Match> nms_matches;
    for (auto& m : raw_matches) {
        bool suppressed = false;
        for (auto& k : nms_matches) {
            float dx = (float)((int)(m.x * inv_scale) - (int)(k.x * inv_scale));
            float dy = (float)((int)(m.y * inv_scale) - (int)(k.y * inv_scale));
            if (dx*dx + dy*dy < nms_r * nms_r) {
                // Also check angle similarity — only suppress if angles are close
                int aid_m = m.template_id % templates_per_scale;
                int aid_k = k.template_id % templates_per_scale;
                int adiff = std::abs(aid_m - aid_k);
                adiff = std::min(adiff, templates_per_scale - adiff);
                float angle_diff = adiff * nms_angle_step;
                if (angle_diff < cfg.nms_angle) { suppressed = true; break; }
            }
        }
        if (!suppressed) {
            nms_matches.push_back(m);
            if (cfg.max_results > 0 && (int)nms_matches.size() >= cfg.max_results)
                break;
        }
    }

    // Convert to MatchResult with user-defined transforms
    std::vector<MatchResult> results;

    // Prepare ICP if needed
    std::vector<icp_refine::EdgePoint> model_edges_cache;

    int n_matches = (int)nms_matches.size();
    results.resize(n_matches);
        // NOTE: Do NOT use vector<bool> here — it bit-packs and causes data races
    // with OpenMP parallel writes to adjacent indices sharing the same byte.
    std::vector<int> valid(n_matches, 0);



    #pragma omp parallel for schedule(dynamic) if(n_matches >= 4)
    for (int mi_idx = 0; mi_idx < n_matches; ++mi_idx) {
        auto& m = nms_matches[mi_idx];
        // Find which model this belongs to
        ModelInfo* mi = nullptr;
        bool is_flip = false;
        for (auto& model : impl_->models) {
            if (m.class_id == model.class_id) { mi = &model; break; }
            if (m.class_id == model.class_id_flip) { mi = &model; is_flip = true; break; }
        }
        if (!mi) continue;

        auto& fs = mi->features;
        auto& tmpl = impl_->detector.getTemplates(m.class_id, m.template_id);
        float angle_step = mi->config.angle.step;

        // Compute raw angle from template_id
        int templates_per_scale = (int)((mi->config.angle.end - mi->config.angle.start) / angle_step);
        float raw_angle = mi->config.angle.start + (m.template_id % templates_per_scale) * angle_step;

        // Compute scale from template_id
        int scale_idx = m.template_id / templates_per_scale;
        float matched_scale = mi->config.scale.min + scale_idx * mi->config.scale.step;
        if (matched_scale < mi->config.scale.min) matched_scale = mi->config.scale.min;

        // Compute object center (user origin, rotated + scaled)
        float scaled_tw = fs.templ_width * matched_scale;
        float scaled_th = fs.templ_height * matched_scale;
        float scene_x = m.x * inv_scale + (scaled_tw / 2.0f - tmpl[0].tl_x * inv_scale);
        float scene_y = m.y * inv_scale + (scaled_th / 2.0f - tmpl[0].tl_y * inv_scale);

        // Transform user origin from template center to scene coords
        float ox = fs.origin.x - fs.templ_width / 2.0f;
        float oy = fs.origin.y - fs.templ_height / 2.0f;
        if (is_flip) oy = -oy;
        float rad = -raw_angle * (float)CV_PI / 180.0f;
        float rot_ox = (std::cos(rad) * ox - std::sin(rad) * oy) * matched_scale;
        float rot_oy = (std::sin(rad) * ox + std::cos(rad) * oy) * matched_scale;
        float user_x = scene_x + rot_ox;
        float user_y = scene_y + rot_oy;

        float user_angle = raw_angle + fs.angle_offset;
        if (is_flip) user_angle = -user_angle + 2 * fs.angle_offset;
        user_angle = std::fmod(user_angle, 360.0f);
        if (user_angle < 0) user_angle += 360.0f;

        // ICP refinement at full resolution (inverse ICP)
        bool do_icp = (cfg.refine == RefineMode::ICP || cfg.refine == RefineMode::ICP_Sparse)
                      && !scene.empty() && fs.templ_scene_valid;
        if (do_icp) {
            icp_refine::ICPConfig icp_cfg;
            icp_cfg.max_iterations = cfg.icp_iterations;
            icp_cfg.max_dist = cfg.icp_max_dist;

            icp_refine::Pose2D init(scene_x, scene_y, raw_angle);
            auto refined = icp_refine::refineInverse(
                fs.cached_templ_scene,
                fs.templ_width, fs.templ_height,
                scene, init, 20, icp_cfg);

            // Update with refined pose
            scene_x = refined.x;
            scene_y = refined.y;
            raw_angle = refined.angle;

            // Recompute user coordinates from refined pose
            rad = -raw_angle * (float)CV_PI / 180.0f;
            rot_ox = (std::cos(rad) * ox - std::sin(rad) * oy) * matched_scale;
            rot_oy = (std::sin(rad) * ox + std::cos(rad) * oy) * matched_scale;
            user_x = scene_x + rot_ox;
            user_y = scene_y + rot_oy;
            user_angle = raw_angle + fs.angle_offset;
            if (is_flip) user_angle = -user_angle + 2 * fs.angle_offset;
            user_angle = std::fmod(user_angle, 360.0f);
            if (user_angle < 0) user_angle += 360.0f;
        }

        // ROI-based refinement
        if (cfg.refine == RefineMode::ROI && !fs.templ_image.empty() && !scene.empty()) {
            // Use sensitivity-optimized point selection
            // cached_opt_points was pre-computed in addModel(); read-only here (thread-safe)
            auto opt_points = fs.selectOptimizedPoints(kDefaultOptPoints);
            std::vector<roi_refine::SamplePoint> sample_pts;
            for (int pi = 0; pi < (int)opt_points.size(); pi++) {
                roi_refine::SamplePoint sp;
                sp.pos = opt_points[pi];
                // Use precomputed lock info from addModel()
                if (pi < (int)fs.cached_lock_info.size()) {
                    sp.lock_major = fs.cached_lock_info[pi].major;
                    sp.lock_minor = fs.cached_lock_info[pi].minor;
                    sp.lock_normal = fs.cached_lock_info[pi].normal;
                    sp.lock_tangent = fs.cached_lock_info[pi].tangent;
                    sp.lock_is_corner = fs.cached_lock_info[pi].is_corner;
                }
                sample_pts.push_back(sp);
            }

            if (!sample_pts.empty()) {
                roi_refine::ROIConfig roi_cfg;
                roi_cfg.roi_half = kDefaultROIHalf;
                roi_cfg.search_half = kDefaultROIHalf;
                roi_cfg.max_iters = kDefaultROIMaxIters;

                cv::Vec3f init_pose(scene_x, scene_y, raw_angle);
                auto refined_pose = roi_refine::refineROI(
                    fs.templ_image, scene, sample_pts, init_pose, roi_cfg);

                scene_x = refined_pose[0];
                scene_y = refined_pose[1];
                raw_angle = refined_pose[2];

                rad = -raw_angle * (float)CV_PI / 180.0f;
                rot_ox = (std::cos(rad) * ox - std::sin(rad) * oy) * matched_scale;
                rot_oy = (std::sin(rad) * ox + std::cos(rad) * oy) * matched_scale;
                user_x = scene_x + rot_ox;
                user_y = scene_y + rot_oy;
                user_angle = raw_angle + fs.angle_offset;
                if (is_flip) user_angle = -user_angle + 2 * fs.angle_offset;
                user_angle = std::fmod(user_angle, 360.0f);
                if (user_angle < 0) user_angle += 360.0f;
            }
        }

        MatchResult r;
        r.model_name = mi->name;
        r.x = user_x;
        r.y = user_y;
        r.angle = user_angle;
        r.scale = matched_scale;
        r.flipped = is_flip;
        r.score = m.similarity;
        results[mi_idx] = r;
        valid[mi_idx] = 1;
    }

    // Remove invalid entries
    std::vector<MatchResult> final_results;
    for (int i = 0; i < n_matches; i++)
        if (valid[i]) final_results.push_back(results[i]);
    return final_results;
}

int ShapeMatcher::numModels() const { return (int)impl_->models.size(); }
int ShapeMatcher::numTemplates() const { return impl_->total_templates; }

} // namespace sbm
