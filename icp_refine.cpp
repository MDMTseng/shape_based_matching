/// @file icp_refine.cpp
/// @brief Edge-based ICP pose refinement implementation.

#include "icp_refine.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstring>

namespace icp_refine {

// Simple 3x3 symmetric positive-definite solver (Cholesky-like)
// Solves A*x = b where A is 3x3 SPD. Returns x.
static bool solve3x3(const float A[3][3], const float b[3], float x[3]) {
    // LDL^T decomposition for 3x3
    float L[3][3] = {};
    float D[3] = {};

    // Row 0
    D[0] = A[0][0];
    if (std::abs(D[0]) < 1e-10f) return false;
    L[0][0] = 1;

    // Row 1
    L[1][0] = A[1][0] / D[0];
    D[1] = A[1][1] - L[1][0] * L[1][0] * D[0];
    if (std::abs(D[1]) < 1e-10f) return false;
    L[1][1] = 1;

    // Row 2
    L[2][0] = A[2][0] / D[0];
    L[2][1] = (A[2][1] - L[2][0] * L[1][0] * D[0]) / D[1];
    D[2] = A[2][2] - L[2][0] * L[2][0] * D[0] - L[2][1] * L[2][1] * D[1];
    if (std::abs(D[2]) < 1e-10f) return false;
    L[2][2] = 1;

    // Forward substitution: L*y = b
    float y[3];
    y[0] = b[0];
    y[1] = b[1] - L[1][0] * y[0];
    y[2] = b[2] - L[2][0] * y[0] - L[2][1] * y[1];

    // Diagonal: D*z = y
    float z[3];
    z[0] = y[0] / D[0];
    z[1] = y[1] / D[1];
    z[2] = y[2] / D[2];

    // Back substitution: L^T * x = z
    x[2] = z[2];
    x[1] = z[1] - L[2][1] * x[2];
    x[0] = z[0] - L[1][0] * x[1] - L[2][0] * x[2];

    return true;
}

// Simple 4x4 solver for Sim2 (with scale)
static bool solve4x4(const float A[4][4], const float b[4], float x[4]) {
    // Gaussian elimination with partial pivoting
    float M[4][5];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) M[i][j] = A[i][j];
        M[i][4] = b[i];
    }

    for (int col = 0; col < 4; ++col) {
        // Pivot
        int best = col;
        for (int row = col + 1; row < 4; ++row)
            if (std::abs(M[row][col]) > std::abs(M[best][col])) best = row;
        if (best != col)
            for (int j = 0; j < 5; ++j) std::swap(M[col][j], M[best][j]);
        if (std::abs(M[col][col]) < 1e-10f) return false;

        // Eliminate
        for (int row = col + 1; row < 4; ++row) {
            float f = M[row][col] / M[col][col];
            for (int j = col; j < 5; ++j) M[row][j] -= f * M[col][j];
        }
    }

    // Back substitution
    for (int i = 3; i >= 0; --i) {
        x[i] = M[i][4];
        for (int j = i + 1; j < 4; ++j) x[i] -= M[i][j] * x[j];
        x[i] /= M[i][i];
    }
    return true;
}

// -----------------------------------------------------------------------
// EdgeScene
// -----------------------------------------------------------------------
void EdgeScene::build(const cv::Mat& sobel_dx, const cv::Mat& sobel_dy,
                      float canny_low, float canny_high, float max_dist) {
    width = sobel_dx.cols;
    height = sobel_dx.rows;

    cv::Mat dx16, dy16;
    if (sobel_dx.type() != CV_16S) sobel_dx.convertTo(dx16, CV_16S);
    else dx16 = sobel_dx;
    if (sobel_dy.type() != CV_16S) sobel_dy.convertTo(dy16, CV_16S);
    else dy16 = sobel_dy;

    cv::Canny(dx16, dy16, edge_map, canny_low, canny_high);

    // Compute normals at edge pixels from gradient direction
    normal_x = cv::Mat::zeros(height, width, CV_32F);
    normal_y = cv::Mat::zeros(height, width, CV_32F);

    for (int r = 0; r < height; ++r) {
        const short* dxr = dx16.ptr<short>(r);
        const short* dyr = dy16.ptr<short>(r);
        const uchar* er = edge_map.ptr<uchar>(r);
        float* nxr = normal_x.ptr<float>(r);
        float* nyr = normal_y.ptr<float>(r);
        for (int c = 0; c < width; ++c) {
            if (er[c] > 0) {
                float gx = (float)dxr[c], gy = (float)dyr[c];
                float mag = std::sqrt(gx * gx + gy * gy);
                if (mag > 1e-6f) {
                    // Normal is perpendicular to edge tangent = gradient direction
                    nxr[c] = gx / mag;
                    nyr[c] = gy / mag;
                }
            }
        }
    }

    // Build closest-edge lookup via OpenCV distance transform with labels.
    // distanceTransformWithLabels assigns each pixel the label of its nearest
    // edge pixel. We then map labels back to coordinates. O(W*H) total.
    closest_x = cv::Mat(height, width, CV_32F, cv::Scalar(-1));
    closest_y = cv::Mat(height, width, CV_32F, cv::Scalar(-1));

    cv::Mat inv_edge;
    cv::bitwise_not(edge_map, inv_edge);

    cv::Mat dist_map, labels;
    cv::distanceTransform(inv_edge, dist_map, labels,
                          cv::DIST_L2, 3, cv::DIST_LABEL_PIXEL);

    // Build label → coordinate map. Labels are 1-based, assigned to each
    // connected zero-pixel (edge pixel) in raster order.
    // Find max label to size the lookup.
    int max_label = 0;
    for (int r = 0; r < height; ++r) {
        const int* lr = labels.ptr<int>(r);
        const uchar* er = edge_map.ptr<uchar>(r);
        for (int c = 0; c < width; ++c) {
            if (er[c] > 0 && lr[c] > max_label)
                max_label = lr[c];
        }
    }

    // Map: label → (x, y) of the edge pixel with that label
    std::vector<cv::Point> label_coords(max_label + 1, cv::Point(-1, -1));
    for (int r = 0; r < height; ++r) {
        const int* lr = labels.ptr<int>(r);
        const uchar* er = edge_map.ptr<uchar>(r);
        for (int c = 0; c < width; ++c) {
            if (er[c] > 0 && lr[c] > 0 && lr[c] <= max_label) {
                label_coords[lr[c]] = cv::Point(c, r);
            }
        }
    }

    // Fill closest_x/y using labels and distance threshold
    for (int r = 0; r < height; ++r) {
        float* cxr = closest_x.ptr<float>(r);
        float* cyr = closest_y.ptr<float>(r);
        const float* dr = dist_map.ptr<float>(r);
        const int* lr = labels.ptr<int>(r);
        for (int c = 0; c < width; ++c) {
            if (dr[c] > max_dist) continue;
            int lbl = lr[c];
            if (lbl > 0 && lbl <= max_label) {
                cv::Point p = label_coords[lbl];
                if (p.x >= 0) {
                    cxr[c] = (float)p.x;
                    cyr[c] = (float)p.y;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------
// Transform model points
// -----------------------------------------------------------------------
std::vector<cv::Point2f> transformModelPoints(
    const std::vector<cv::Point2f>& templ_edges, const Pose2D& pose) {
    float rad = pose.angle * (float)CV_PI / 180.0f;
    float cs = std::cos(rad) * pose.scale;
    float sn = std::sin(rad) * pose.scale;

    std::vector<cv::Point2f> result(templ_edges.size());
    for (size_t i = 0; i < templ_edges.size(); ++i) {
        float px = templ_edges[i].x, py = templ_edges[i].y;
        result[i].x = cs * px - sn * py + pose.x;
        result[i].y = sn * px + cs * py + pose.y;
    }
    return result;
}

// -----------------------------------------------------------------------
// ICP refinement
// -----------------------------------------------------------------------
Pose2D refine(const std::vector<cv::Point2f>& templ_edges,
              const EdgeScene& scene,
              const Pose2D& initial_pose,
              const ICPConfig& config) {
    if (templ_edges.empty()) return initial_pose;

    int N = (int)templ_edges.size();
    Pose2D pose = initial_pose;
    float prev_fitness = 0, prev_rmse = 1e10f;

    for (int iter = 0; iter < config.max_iterations; ++iter) {
        // Transform model to current pose
        auto pts = transformModelPoints(templ_edges, pose);

        // Build correspondences: for each model point, find closest scene edge
        float ATA[3][3] = {}, ATb[3] = {};  // SO2: [theta, tx, ty]
        float ATA4[4][4] = {}, ATb4[4] = {};  // Sim2: [theta, tx, ty, ds]
        float total_error = 0;
        int inlier_count = 0;

        for (int i = 0; i < N; ++i) {
            float mx = pts[i].x, my = pts[i].y;
            int ix = (int)(mx + 0.5f), iy = (int)(my + 0.5f);
            if (ix < 0 || ix >= scene.width || iy < 0 || iy >= scene.height) continue;

            float cx = scene.closest_x.at<float>(iy, ix);
            float cy = scene.closest_y.at<float>(iy, ix);
            if (cx < 0) continue;  // no edge nearby

            float dx = mx - cx, dy = my - cy;
            float dist2 = dx * dx + dy * dy;
            if (dist2 > config.max_dist * config.max_dist) continue;

            // Normal at closest edge point
            int ecx = (int)(cx + 0.5f), ecy = (int)(cy + 0.5f);
            ecx = std::max(0, std::min(scene.width - 1, ecx));
            ecy = std::max(0, std::min(scene.height - 1, ecy));
            float nx = scene.normal_x.at<float>(ecy, ecx);
            float ny = scene.normal_y.at<float>(ecy, ecx);
            if (nx == 0 && ny == 0) continue;

            // Point-to-plane error: e = (model - closest) . normal
            float e = dx * nx + dy * ny;
            total_error += e * e;
            ++inlier_count;

            // Jacobian for SO2: d(e)/d(theta, tx, ty)
            // model = R(theta) * templ + t
            // d(model)/d(theta) = [-sin(theta)*px - cos(theta)*py,
            //                       cos(theta)*px - sin(theta)*py]
            // = [-my_local, mx_local] where mx_local, my_local are in current frame
            // Simplified: J = [(-my*nx + mx*ny), nx, ny]
            float j0 = -my * nx + mx * ny;  // d/d(theta) -- small angle approx
            float j1 = nx;                   // d/d(tx)
            float j2 = ny;                   // d/d(ty)

            // Accumulate J^T * J and J^T * e
            ATA[0][0] += j0 * j0; ATA[0][1] += j0 * j1; ATA[0][2] += j0 * j2;
            ATA[1][1] += j1 * j1; ATA[1][2] += j1 * j2;
            ATA[2][2] += j2 * j2;
            ATb[0] -= j0 * e;
            ATb[1] -= j1 * e;
            ATb[2] -= j2 * e;

            if (config.use_scale) {
                // Sim2: add d/d(scale)
                // d(model)/d(s) at current pose = R * templ = [mx - tx, my - ty] / s
                float j3 = ((mx - pose.x) * nx + (my - pose.y) * ny) / pose.scale;
                ATA4[0][0] += j0*j0; ATA4[0][1] += j0*j1; ATA4[0][2] += j0*j2; ATA4[0][3] += j0*j3;
                ATA4[1][1] += j1*j1; ATA4[1][2] += j1*j2; ATA4[1][3] += j1*j3;
                ATA4[2][2] += j2*j2; ATA4[2][3] += j2*j3;
                ATA4[3][3] += j3*j3;
                ATb4[0] -= j0*e; ATb4[1] -= j1*e; ATb4[2] -= j2*e; ATb4[3] -= j3*e;
            }
        }

        if (inlier_count == 0) break;

        pose.fitness = (float)inlier_count / N;
        pose.rmse = std::sqrt(total_error / inlier_count);

        // Check convergence
        if (iter > 0 &&
            std::abs(pose.fitness - prev_fitness) < config.convergence_fitness &&
            std::abs(pose.rmse - prev_rmse) < config.convergence_rmse) {
            break;
        }
        prev_fitness = pose.fitness;
        prev_rmse = pose.rmse;

        // Solve for update
        float update[4] = {};
        if (config.use_scale) {
            // Symmetrize
            ATA4[1][0] = ATA4[0][1]; ATA4[2][0] = ATA4[0][2]; ATA4[2][1] = ATA4[1][2];
            ATA4[3][0] = ATA4[0][3]; ATA4[3][1] = ATA4[1][3]; ATA4[3][2] = ATA4[2][3];
            // Regularize
            for (int i = 0; i < 4; ++i) ATA4[i][i] += 0.01f;
            if (!solve4x4(ATA4, ATb4, update)) break;
        } else {
            // Symmetrize
            ATA[1][0] = ATA[0][1]; ATA[2][0] = ATA[0][2]; ATA[2][1] = ATA[1][2];
            // Regularize to prevent large updates
            for (int i = 0; i < 3; ++i) ATA[i][i] += 0.01f;
            if (!solve3x3(ATA, ATb, update)) break;
        }

        // Apply update: theta, tx, ty [, scale]
        float d_theta = update[0];
        float d_tx = update[1];
        float d_ty = update[2];

        // Clamp update to avoid divergence
        d_theta = std::max(-0.1f, std::min(0.1f, d_theta));  // ~5.7 degrees max
        d_tx = std::max(-5.0f, std::min(5.0f, d_tx));
        d_ty = std::max(-5.0f, std::min(5.0f, d_ty));

        pose.angle += d_theta * 180.0f / (float)CV_PI;
        pose.x += d_tx;
        pose.y += d_ty;

        if (config.use_scale) {
            float d_scale = update[3];
            d_scale = std::max(-0.05f, std::min(0.05f, d_scale));
            pose.scale += d_scale;
            pose.scale = std::max(0.5f, std::min(2.0f, pose.scale));
        }
    }

    // Normalize angle to [0, 360)
    while (pose.angle < 0) pose.angle += 360.0f;
    while (pose.angle >= 360.0f) pose.angle -= 360.0f;

    return pose;
}

// -----------------------------------------------------------------------
// Local ROI refinement — builds EdgeScene only on a small patch
// -----------------------------------------------------------------------
Pose2D refineLocal(const std::vector<cv::Point2f>& templ_edges,
                   const cv::Mat& scene_dx, const cv::Mat& scene_dy,
                   const Pose2D& initial_pose,
                   int templ_size,
                   int roi_margin,
                   const ICPConfig& config) {
    int sw = scene_dx.cols, sh = scene_dx.rows;
    int half = templ_size / 2 + roi_margin;

    // ROI centered on match position
    int rx = (int)(initial_pose.x + 0.5f) - half;
    int ry = (int)(initial_pose.y + 0.5f) - half;
    int rw = 2 * half;
    int rh = 2 * half;

    // Clamp to image bounds
    if (rx < 0) rx = 0;
    if (ry < 0) ry = 0;
    if (rx + rw > sw) rw = sw - rx;
    if (ry + rh > sh) rh = sh - ry;
    if (rw <= 10 || rh <= 10) return initial_pose;

    cv::Rect roi(rx, ry, rw, rh);
    cv::Mat local_dx = scene_dx(roi);
    cv::Mat local_dy = scene_dy(roi);

    // Build local edge scene
    EdgeScene local_scene;
    local_scene.build(local_dx, local_dy,
                      config.max_dist * 3, config.max_dist * 6,  // Canny thresholds
                      config.max_dist);

    // Shift initial pose to local coordinates
    Pose2D local_pose = initial_pose;
    local_pose.x -= rx;
    local_pose.y -= ry;

    // Run ICP in local coordinates
    Pose2D result = refine(templ_edges, local_scene, local_pose, config);

    // Shift back to global coordinates
    result.x += rx;
    result.y += ry;

    return result;
}

} // namespace icp_refine
