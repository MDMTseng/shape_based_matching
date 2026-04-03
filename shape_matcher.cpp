/// @file shape_matcher.cpp
/// @brief High-level shape-based matching API implementation.

#include "shape_matcher.h"
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
    for(int i=0;i<3;i++) ATA[i][i]+=0.001f;
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

std::vector<cv::Point2f> FeatureSet::selectOptimizedPoints(int max_points) const {
    // Return cached result if available and computed with same (or larger) max_points
    if (!cached_opt_points.empty() && cached_opt_max_points >= max_points)
        return cached_opt_points;

    std::vector<cv::Point2f> result;
    if (refine_points.empty() || templ_image.empty())
        return result;

    // Collect all valid candidate points with margin check
    float max_lev = std::sqrt((float)(templ_width*templ_width + templ_height*templ_height)) / 2.0f;
    struct CandPt { int rp_idx; float corn; float score; };
    std::vector<CandPt> all_cands;
    for (size_t i = 0; i < refine_points.size(); i++) {
        auto& rp = refine_points[i];
        if (std::abs(rp.px) > templ_width/2.0f - 5 || std::abs(rp.py) > templ_height/2.0f - 5)
            continue;
        float lev = std::sqrt(rp.px*rp.px + rp.py*rp.py);
        // Score: corners high priority, then distance from center
        // cornerness [0,1] boosted to dominate, leverage normalized to [0,1]
        float score = rp.cornerness * 10.0f + lev / (max_lev + 1e-6f);
        all_cands.push_back({(int)i, rp.cornerness, score});
    }

    // Initial greedy selection: corners + far-from-center first
    float min_dist = std::max(templ_width, templ_height) / 16.0f * 1.5f;
    float min_dist_sq = min_dist * min_dist;
    std::sort(all_cands.begin(), all_cands.end(),
              [](const CandPt& a, const CandPt& b){ return a.score > b.score; });

    std::vector<int> selected;  // indices into refine_points
    for (auto& c : all_cands) {
        if ((int)selected.size() >= max_points) break;
        auto& rp = refine_points[c.rp_idx];
        cv::Point2f p(rp.px, rp.py);
        bool too_close = false;
        for (int si : selected) {
            float dx = p.x - refine_points[si].px, dy = p.y - refine_points[si].py;
            if (dx*dx + dy*dy < min_dist_sq) { too_close = true; break; }
        }
        if (too_close) continue;
        selected.push_back(c.rp_idx);
    }

    if ((int)selected.size() < 3) {
        for (int idx : selected)
            result.push_back(cv::Point2f(refine_points[idx].px, refine_points[idx].py));
        return result;
    }

    // Build constraints for selected points
    auto buildSet = [&](const std::vector<int>& sel) -> std::vector<SensConstraint> {
        std::vector<SensConstraint> cs;
        for (int idx : sel) {
            SensConstraint c;
            if (buildConstraint(*this, idx, c))
                cs.push_back(c);
        }
        return cs;
    };

    auto inSelected = [&](int idx) {
        for (int s : selected) if (s == idx) return true;
        return false;
    };

    // Iterative optimization: swap least sensitive with best candidate
    float opt_min_dist_sq = (min_dist * 0.5f) * (min_dist * 0.5f);

    for (int opt_iter = 0; opt_iter < 20; ++opt_iter) {
        auto baseline = buildSet(selected);
        if ((int)baseline.size() < 3) break;

        cv::Vec3f base_pose = solveSens(baseline);
        std::vector<float> sens(baseline.size());
        int least_idx = 0;
        float least_sens = 1e9f;
        float worst_sens = 0;
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
            sens[fi] = d_ang + d_pos;
            // Don't swap out corners — they provide 2D constraint
            bool is_corner = refine_points[selected[fi]].cornerness > 0.3f;
            if (sens[fi] < least_sens && !is_corner) { least_sens = sens[fi]; least_idx = (int)fi; }
            worst_sens = std::max(worst_sens, sens[fi]);
        }

        if (least_sens > 1e-6f && worst_sens / least_sens < 2.0f) break;

        int remove_rp_idx = selected[least_idx];
        float best_worst = worst_sens;
        int best_cand = -1;

        for (auto& c : all_cands) {
            if (inSelected(c.rp_idx)) continue;
            auto& rp = refine_points[c.rp_idx];
            cv::Point2f p(rp.px, rp.py);
            bool too_close = false;
            for (int si : selected) {
                if (si == remove_rp_idx) continue;
                float dx = p.x - refine_points[si].px, dy = p.y - refine_points[si].py;
                if (dx*dx + dy*dy < opt_min_dist_sq) { too_close = true; break; }
            }
            if (too_close) continue;

            auto trial = selected;
            trial[least_idx] = c.rp_idx;
            auto trial_cs = buildSet(trial);
            if ((int)trial_cs.size() < 3) continue;

            float trial_worst = computeMaxSens(trial_cs);
            if (trial_worst < best_worst) {
                best_worst = trial_worst;
                best_cand = c.rp_idx;
            }
        }

        if (best_cand < 0 || best_worst >= worst_sens * 0.99f) break;
        selected[least_idx] = best_cand;
    }

    for (int idx : selected)
        result.push_back(cv::Point2f(refine_points[idx].px, refine_points[idx].py));
    cached_opt_points = result;
    cached_opt_max_points = max_points;
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
    auto opt_points = selectOptimizedPoints(15);
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
          detector(128, {4, 8}, 30, 60) {}

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
    info.features.selectOptimizedPoints(15);  // precompute + cache
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

    // Run meiqua matching
    auto raw_matches = impl_->detector.match(padded, cfg.min_score, class_ids);

    // NMS
    float nms_r = cfg.nms_radius;
    std::vector<line2Dup::Match> nms_matches;
    for (auto& m : raw_matches) {
        bool suppressed = false;
        for (auto& k : nms_matches) {
            float dx = (float)((int)(m.x * inv_scale) - (int)(k.x * inv_scale));
            float dy = (float)((int)(m.y * inv_scale) - (int)(k.y * inv_scale));
            if (dx*dx + dy*dy < nms_r * nms_r) { suppressed = true; break; }
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
    std::vector<bool> valid(n_matches, false);

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
            auto opt_points = fs.selectOptimizedPoints(15);
            std::vector<roi_refine::SamplePoint> sample_pts;
            for (auto& p : opt_points) {
                roi_refine::SamplePoint sp;
                sp.pos = p;
                sample_pts.push_back(sp);
            }

            if (!sample_pts.empty()) {
                roi_refine::ROIConfig roi_cfg;
                roi_cfg.roi_half = 15;
                roi_cfg.search_half = 15;
                roi_cfg.max_iters = 3;

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
        valid[mi_idx] = true;
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
