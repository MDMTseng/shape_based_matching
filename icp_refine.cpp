/// @file icp_refine.cpp
/// @brief Edge-based ICP pose refinement implementation.

#include "icp_refine.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstring>

#ifdef __AVX2__
#include <immintrin.h>

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}
#endif

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

            // Point-to-plane error: e_plane = (model - closest) . normal
            float e_plane = dx * nx + dy * ny;
            total_error += e_plane * e_plane;
            ++inlier_count;

            // Jacobian for point-to-plane: J_plane = [(-my*nx + mx*ny), nx, ny]
            float jp0 = -my * nx + mx * ny;  // d/d(theta)
            float jp1 = nx;                   // d/d(tx)
            float jp2 = ny;                   // d/d(ty)

            // Accumulate point-to-plane: J^T * J and J^T * e
            ATA[0][0] += jp0*jp0; ATA[0][1] += jp0*jp1; ATA[0][2] += jp0*jp2;
            ATA[1][1] += jp1*jp1; ATA[1][2] += jp1*jp2;
            ATA[2][2] += jp2*jp2;
            ATb[0] -= jp0 * e_plane;
            ATb[1] -= jp1 * e_plane;
            ATb[2] -= jp2 * e_plane;

            // Point-to-point regularization: prevents sliding along edges.
            // Adds weighted (dx, dy) error with Jacobian for x and y separately.
            // J_x = [-my, 1, 0],  e_x = dx
            // J_y = [ mx, 0, 1],  e_y = dy
            float w = config.point_to_point_weight;
            if (w > 0) {
                float jx0 = -my, jx1 = 1.0f, jx2 = 0.0f;
                float jy0 =  mx, jy1 = 0.0f, jy2 = 1.0f;

                ATA[0][0] += w * (jx0*jx0 + jy0*jy0);
                ATA[0][1] += w * (jx0*jx1 + jy0*jy1);
                ATA[0][2] += w * (jx0*jx2 + jy0*jy2);
                ATA[1][1] += w * (jx1*jx1 + jy1*jy1);
                ATA[1][2] += w * (jx1*jx2 + jy1*jy2);
                ATA[2][2] += w * (jx2*jx2 + jy2*jy2);
                ATb[0] -= w * (jx0*dx + jy0*dy);
                ATb[1] -= w * (jx1*dx + jy1*dy);
                ATb[2] -= w * (jx2*dx + jy2*dy);
            }

            if (config.use_scale) {
                float j3 = ((mx - pose.x) * nx + (my - pose.y) * ny) / pose.scale;
                ATA4[0][0] += jp0*jp0; ATA4[0][1] += jp0*jp1; ATA4[0][2] += jp0*jp2; ATA4[0][3] += jp0*j3;
                ATA4[1][1] += jp1*jp1; ATA4[1][2] += jp1*jp2; ATA4[1][3] += jp1*j3;
                ATA4[2][2] += jp2*jp2; ATA4[2][3] += jp2*j3;
                ATA4[3][3] += j3*j3;
                ATb4[0] -= jp0*e_plane; ATb4[1] -= jp1*e_plane; ATb4[2] -= jp2*e_plane; ATb4[3] -= j3*e_plane;
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
    pose.angle = std::fmod(pose.angle, 360.0f);
    if (pose.angle < 0) pose.angle += 360.0f;

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

// -----------------------------------------------------------------------
// Extract model edges with normals
// -----------------------------------------------------------------------
std::vector<EdgePoint> extractModelEdges(const cv::Mat& templ_gray) {
    int TW = templ_gray.cols;
    cv::Mat smooth, dx, dy, edges;
    cv::GaussianBlur(templ_gray, smooth, cv::Size(5, 5), 0);
    cv::Sobel(smooth, dx, CV_16S, 1, 0, 3);
    cv::Sobel(smooth, dy, CV_16S, 0, 1, 3);
    cv::Canny(dx, dy, edges, 30, 60);

    std::vector<EdgePoint> pts;
    for (int r = 0; r < TW; ++r) {
        const short* dxr = dx.ptr<short>(r);
        const short* dyr = dy.ptr<short>(r);
        for (int c = 0; c < TW; ++c) {
            if (edges.at<uchar>(r, c) > 0) {
                EdgePoint ep;
                ep.pos = cv::Point2f((float)(c - TW / 2), (float)(r - TW / 2));
                float gx = (float)dxr[c], gy = (float)dyr[c];
                float mag = std::sqrt(gx * gx + gy * gy);
                if (mag > 1e-6f) {
                    ep.normal = cv::Point2f(gx / mag, gy / mag);
                } else {
                    ep.normal = cv::Point2f(0, 0);
                }
                pts.push_back(ep);
            }
        }
    }
    return pts;
}

// -----------------------------------------------------------------------
// ICP with normal compatibility filtering
// -----------------------------------------------------------------------
Pose2D refineWithNormals(const std::vector<EdgePoint>& model_edges,
                         const cv::Mat& scene_dx, const cv::Mat& scene_dy,
                         const Pose2D& initial_pose,
                         int templ_size,
                         int roi_margin,
                         const ICPConfig& config) {
    int sw = scene_dx.cols, sh = scene_dx.rows;
    int half = templ_size / 2 + roi_margin;

    int rx = (int)(initial_pose.x + 0.5f) - half;
    int ry = (int)(initial_pose.y + 0.5f) - half;
    int rw = 2 * half, rh = 2 * half;
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
                      config.max_dist * 3, config.max_dist * 6,
                      config.max_dist);

    Pose2D pose = initial_pose;
    pose.x -= rx;
    pose.y -= ry;

    float cos_thresh = std::cos(config.normal_angle_thresh * (float)CV_PI / 180.0f);
    int N = (int)model_edges.size();
    float prev_fitness = 0, prev_rmse = 1e10f;

    // SoA layout for SIMD
    std::vector<float> soa_px(N), soa_py(N), soa_nx(N), soa_ny(N), soa_corn(N);
    for (int i = 0; i < N; i++) {
        soa_px[i] = model_edges[i].pos.x;
        soa_py[i] = model_edges[i].pos.y;
        soa_nx[i] = model_edges[i].normal.x;
        soa_ny[i] = model_edges[i].normal.y;
        soa_corn[i] = model_edges[i].cornerness;
    }

    // Precompute pointers and stride for gather
    float* cx_base = (float*)local_scene.closest_x.data;
    float* cy_base = (float*)local_scene.closest_y.data;
    float* snx_base = (float*)local_scene.normal_x.data;
    float* sny_base = (float*)local_scene.normal_y.data;
    int stride = (int)(local_scene.closest_x.step1());  // floats per row
    float max_dist2 = config.max_dist * config.max_dist;
    float p2p_w = config.point_to_point_weight;
    bool use_corn = config.use_cornerness;

    for (int iter = 0; iter < config.max_iterations; ++iter) {
        float rad = pose.angle * (float)CV_PI / 180.0f;
        float cs = std::cos(rad) * pose.scale;
        float sn = std::sin(rad) * pose.scale;

        // Direct Jacobian accumulation (no correspondence collection)
        float ATA[3][3] = {}, ATb[3] = {};
        float total_error = 0;
        int inlier_count = 0;

        for (int i = 0; i < N; ++i) {
            float px = soa_px[i], py = soa_py[i];
            float mx = cs * px - sn * py + pose.x;
            float my = sn * px + cs * py + pose.y;

            int ix = (int)(mx + 0.5f), iy = (int)(my + 0.5f);
            if (ix < 0 || ix >= local_scene.width || iy < 0 || iy >= local_scene.height) continue;

            float closest_cx = cx_base[iy * stride + ix];
            float closest_cy = cy_base[iy * stride + ix];
            if (closest_cx < 0) continue;

            float ddx = mx - closest_cx, ddy = my - closest_cy;
            if (ddx * ddx + ddy * ddy > max_dist2) continue;

            int ecx = std::max(0, std::min(local_scene.width - 1, (int)(closest_cx + 0.5f)));
            int ecy = std::max(0, std::min(local_scene.height - 1, (int)(closest_cy + 0.5f)));
            float snx_v = snx_base[ecy * stride + ecx];
            float sny_v = sny_base[ecy * stride + ecx];
            if (snx_v == 0 && sny_v == 0) continue;

            float mnx = soa_nx[i], mny = soa_ny[i];
            float rot_mnx = cs * mnx - sn * mny;
            float rot_mny = sn * mnx + cs * mny;
            float nmag = std::sqrt(rot_mnx*rot_mnx + rot_mny*rot_mny);
            if (nmag > 1e-6f) { rot_mnx /= nmag; rot_mny /= nmag; }

            float ndot = std::abs(rot_mnx * snx_v + rot_mny * sny_v);
            if (ndot < cos_thresh) continue;

            float e_plane = ddx * snx_v + ddy * sny_v;
            total_error += e_plane * e_plane;
            ++inlier_count;

            float jp0 = -my * snx_v + mx * sny_v;
            float jp1 = snx_v;
            float jp2 = sny_v;

            ATA[0][0] += jp0*jp0; ATA[0][1] += jp0*jp1; ATA[0][2] += jp0*jp2;
            ATA[1][1] += jp1*jp1; ATA[1][2] += jp1*jp2;
            ATA[2][2] += jp2*jp2;
            ATb[0] -= jp0 * e_plane;
            ATb[1] -= jp1 * e_plane;
            ATb[2] -= jp2 * e_plane;

            float w = p2p_w;
            if (use_corn) {
                w = soa_corn[i] * 1.0f + (1.0f - soa_corn[i]) * 0.01f;
            }
            if (w > 0) {
                ATA[0][0] += w * (my*my + mx*mx);
                ATA[0][1] += w * (-my);
                ATA[0][2] += w * (mx);
                ATA[1][1] += w;
                ATA[2][2] += w;
                ATb[0] -= w * (-my*ddx + mx*ddy);
                ATb[1] -= w * ddx;
                ATb[2] -= w * ddy;
            }
        }

        if (inlier_count == 0) break;
        pose.fitness = (float)inlier_count / N;
        pose.rmse = std::sqrt(total_error / inlier_count);

        if (iter > 0 &&
            std::abs(pose.fitness - prev_fitness) < config.convergence_fitness &&
            std::abs(pose.rmse - prev_rmse) < config.convergence_rmse)
            break;
        prev_fitness = pose.fitness;
        prev_rmse = pose.rmse;

        // Solve
        ATA[1][0] = ATA[0][1]; ATA[2][0] = ATA[0][2]; ATA[2][1] = ATA[1][2];
        for (int i = 0; i < 3; ++i) ATA[i][i] += 0.01f;
        float update[3] = {};
        if (!solve3x3(ATA, ATb, update)) break;

        float d_theta = std::max(-0.1f, std::min(0.1f, update[0]));
        float d_tx = std::max(-5.0f, std::min(5.0f, update[1]));
        float d_ty = std::max(-5.0f, std::min(5.0f, update[2]));
        pose.angle += d_theta * 180.0f / (float)CV_PI;
        pose.x += d_tx;
        pose.y += d_ty;
    }

    pose.angle = std::fmod(pose.angle, 360.0f);
    if (pose.angle < 0) pose.angle += 360.0f;
    pose.x += rx;
    pose.y += ry;
    return pose;
}

// -----------------------------------------------------------------------
// Build template EdgeScene (call once per template)
// -----------------------------------------------------------------------
EdgeScene buildTemplateScene(const cv::Mat& templ_gray, float max_dist) {
    cv::Mat t_smooth, t_dx, t_dy;
    cv::GaussianBlur(templ_gray, t_smooth, cv::Size(5, 5), 0);
    cv::Sobel(t_smooth, t_dx, CV_16S, 1, 0, 3);
    cv::Sobel(t_smooth, t_dy, CV_16S, 0, 1, 3);

    EdgeScene scene;
    scene.build(t_dx, t_dy, 30, 60, max_dist);
    return scene;
}

// -----------------------------------------------------------------------
// Core inverse ICP iteration loop (shared by all refineInverse overloads)
// -----------------------------------------------------------------------
static Pose2D refineInverseCore(
    const EdgeScene& templ_scene, int TW, int TH,
    const float* se_x, const float* se_y,
    const float* se_nx, const float* se_ny,
    int NS,
    const Pose2D& initial_pose,
    const ICPConfig& config)
{
    float tcx = TW / 2.0f, tcy = TH / 2.0f;
    float cos_thresh = std::cos(config.normal_angle_thresh * (float)CV_PI / 180.0f);
    float max_dist2 = config.max_dist * config.max_dist * 4;  // wider search in template space
    float* tcx_base = (float*)templ_scene.closest_x.data;
    float* tcy_base = (float*)templ_scene.closest_y.data;
    float* tnx_base = (float*)templ_scene.normal_x.data;
    float* tny_base = (float*)templ_scene.normal_y.data;
    int tstride = (int)templ_scene.closest_x.step1();

    Pose2D pose = initial_pose;
    float prev_fitness = 0, prev_rmse = 1e10f;

    for (int iter = 0; iter < config.max_iterations; ++iter) {
        float rad = pose.angle * (float)CV_PI / 180.0f;
        float cs = std::cos(rad), sn = std::sin(rad);
        float inv_cs = cs, inv_sn = -sn;  // R^T

        float ATA[3][3] = {}, ATb[3] = {};
        float total_error = 0;
        int inlier_count = 0;

        for (int i = 0; i < NS; i++) {
            float sx = se_x[i], sy = se_y[i];

            // Inverse-transform scene point to template space
            float dx_s = sx - pose.x, dy_s = sy - pose.y;
            float tx = inv_cs * dx_s - inv_sn * dy_s + tcx;
            float ty = inv_sn * dx_s + inv_cs * dy_s + tcy;

            int ix = (int)(tx + 0.5f), iy = (int)(ty + 0.5f);
            if (ix < 0 || ix >= TW || iy < 0 || iy >= TH) continue;

            // Find closest template edge
            float ctx = tcx_base[iy * tstride + ix];
            float cty = tcy_base[iy * tstride + ix];
            if (ctx < 0) continue;

            float ddx = tx - ctx, ddy = ty - cty;
            if (ddx*ddx + ddy*ddy > max_dist2) continue;

            // Template normal at closest edge
            int ecx = std::max(0, std::min(TW - 1, (int)(ctx + 0.5f)));
            int ecy = std::max(0, std::min(TH - 1, (int)(cty + 0.5f)));
            float tnx = tnx_base[ecy * tstride + ecx];
            float tny = tny_base[ecy * tstride + ecx];
            if (tnx == 0 && tny == 0) continue;

            // Normal compatibility: rotate scene normal to template space
            float rot_snx = inv_cs * se_nx[i] - inv_sn * se_ny[i];
            float rot_sny = inv_sn * se_nx[i] + inv_cs * se_ny[i];
            float ndot = std::abs(rot_snx * tnx + rot_sny * tny);
            if (ndot < cos_thresh) continue;

            // Error in template space: point-to-plane
            float e_plane = ddx * tnx + ddy * tny;
            total_error += e_plane * e_plane;
            ++inlier_count;

            // Jacobian
            float dtx_da = -sn * dx_s + cs * dy_s;
            float dty_da = -cs * dx_s - sn * dy_s;

            float jp0 = dtx_da * tnx + dty_da * tny;
            float jp1 = (-cs * tnx + sn * tny);
            float jp2 = (-sn * tnx - cs * tny);

            ATA[0][0] += jp0*jp0; ATA[0][1] += jp0*jp1; ATA[0][2] += jp0*jp2;
            ATA[1][1] += jp1*jp1; ATA[1][2] += jp1*jp2;
            ATA[2][2] += jp2*jp2;
            ATb[0] -= jp0 * e_plane;
            ATb[1] -= jp1 * e_plane;
            ATb[2] -= jp2 * e_plane;

            // Point-to-point regularization
            float w = config.point_to_point_weight;
            if (w > 0) {
                float jx0 = dtx_da, jx1 = -cs, jx2 = -sn;
                float jy0 = dty_da, jy1 = sn,  jy2 = -cs;
                ATA[0][0] += w*(jx0*jx0+jy0*jy0);
                ATA[0][1] += w*(jx0*jx1+jy0*jy1);
                ATA[0][2] += w*(jx0*jx2+jy0*jy2);
                ATA[1][1] += w*(jx1*jx1+jy1*jy1);
                ATA[1][2] += w*(jx1*jx2+jy1*jy2);
                ATA[2][2] += w*(jx2*jx2+jy2*jy2);
                ATb[0] -= w*(jx0*ddx+jy0*ddy);
                ATb[1] -= w*(jx1*ddx+jy1*ddy);
                ATb[2] -= w*(jx2*ddx+jy2*ddy);
            }
        }

        if (inlier_count == 0) break;
        pose.fitness = (float)inlier_count / NS;
        pose.rmse = std::sqrt(total_error / inlier_count);

        if (iter > 0 &&
            std::abs(pose.fitness - prev_fitness) < config.convergence_fitness &&
            std::abs(pose.rmse - prev_rmse) < config.convergence_rmse)
            break;
        prev_fitness = pose.fitness;
        prev_rmse = pose.rmse;

        ATA[1][0]=ATA[0][1]; ATA[2][0]=ATA[0][2]; ATA[2][1]=ATA[1][2];
        for (int i = 0; i < 3; i++) ATA[i][i] += 0.01f;
        float update[3] = {};
        if (!solve3x3(ATA, ATb, update)) break;

        float d_theta = std::max(-0.1f, std::min(0.1f, update[0]));
        float d_tx = std::max(-5.0f, std::min(5.0f, update[1]));
        float d_ty = std::max(-5.0f, std::min(5.0f, update[2]));
        pose.angle += d_theta * 180.0f / (float)CV_PI;
        pose.x += d_tx;
        pose.y += d_ty;
    }

    pose.angle = std::fmod(pose.angle, 360.0f);
    if (pose.angle < 0) pose.angle += 360.0f;
    return pose;
}

// Helper: extract scene edges from Sobel dx/dy in a ROI, return SoA arrays
static int extractSceneEdgesSoA(
    const cv::Mat& s_dx, const cv::Mat& s_dy,
    float offset_x, float offset_y,
    std::vector<float>& out_x, std::vector<float>& out_y,
    std::vector<float>& out_nx, std::vector<float>& out_ny)
{
    cv::Mat s_edges;
    cv::Canny(s_dx, s_dy, s_edges, 50, 100);

    for (int r = 0; r < s_edges.rows; r++) {
        const uchar* er = s_edges.ptr<uchar>(r);
        const short* dxr = s_dx.ptr<short>(r);
        const short* dyr = s_dy.ptr<short>(r);
        for (int c = 0; c < s_edges.cols; c++) {
            if (er[c] == 0) continue;
            float gx = (float)dxr[c], gy = (float)dyr[c];
            float mag = std::sqrt(gx*gx + gy*gy);
            if (mag < 1e-6f) continue;
            out_x.push_back((float)c + offset_x);
            out_y.push_back((float)r + offset_y);
            out_nx.push_back(gx / mag);
            out_ny.push_back(gy / mag);
        }
    }
    return (int)out_x.size();
}

// -----------------------------------------------------------------------
// Inverse ICP: build EDT on template, match scene edges against it
// (original interface — builds template scene each call)
// -----------------------------------------------------------------------
Pose2D refineInverse(const cv::Mat& templ_gray,
                     const cv::Mat& scene_gray,
                     const Pose2D& initial_pose,
                     int roi_margin,
                     const ICPConfig& config) {
    EdgeScene templ_scene = buildTemplateScene(templ_gray, config.max_dist * 2);
    return refineInverse(templ_scene, templ_gray.cols, templ_gray.rows,
                         scene_gray, initial_pose, roi_margin, config);
}

// -----------------------------------------------------------------------
// Inverse ICP with pre-built template EdgeScene + scene gray image
// -----------------------------------------------------------------------
Pose2D refineInverse(const EdgeScene& templ_scene,
                     int templ_width, int templ_height,
                     const cv::Mat& scene_gray,
                     const Pose2D& initial_pose,
                     int roi_margin,
                     const ICPConfig& config) {
    int TW = templ_width, TH = templ_height;

    // Extract scene edges in local ROI
    int margin = TW / 2 + roi_margin;
    int rx = std::max(0, (int)(initial_pose.x + 0.5f) - margin);
    int ry = std::max(0, (int)(initial_pose.y + 0.5f) - margin);
    int rw = std::min(scene_gray.cols - rx, 2 * margin);
    int rh = std::min(scene_gray.rows - ry, 2 * margin);
    if (rw <= 10 || rh <= 10) return initial_pose;

    cv::Mat roi = scene_gray(cv::Rect(rx, ry, rw, rh));
    cv::Mat s_smooth, s_dx, s_dy;
    cv::GaussianBlur(roi, s_smooth, cv::Size(5, 5), 0);
    cv::Sobel(s_smooth, s_dx, CV_16S, 1, 0, 3);
    cv::Sobel(s_smooth, s_dy, CV_16S, 0, 1, 3);

    std::vector<float> se_x, se_y, se_nx, se_ny;
    int NS = extractSceneEdgesSoA(s_dx, s_dy, (float)rx, (float)ry,
                                  se_x, se_y, se_nx, se_ny);
    if (NS == 0) return initial_pose;

    return refineInverseCore(templ_scene, TW, TH,
                             se_x.data(), se_y.data(),
                             se_nx.data(), se_ny.data(), NS,
                             initial_pose, config);
}

// -----------------------------------------------------------------------
// Inverse ICP with pre-built template EdgeScene + scene Sobel derivatives
// -----------------------------------------------------------------------
Pose2D refineInverse(const EdgeScene& templ_scene,
                     int templ_width, int templ_height,
                     const cv::Mat& scene_dx, const cv::Mat& scene_dy,
                     const Pose2D& initial_pose,
                     int templ_diag,
                     int roi_margin,
                     const ICPConfig& config) {
    int TW = templ_width, TH = templ_height;

    // ROI from scene Sobel derivatives
    int margin = templ_diag / 2 + roi_margin;
    int rx = std::max(0, (int)(initial_pose.x + 0.5f) - margin);
    int ry = std::max(0, (int)(initial_pose.y + 0.5f) - margin);
    int rw = std::min(scene_dx.cols - rx, 2 * margin);
    int rh = std::min(scene_dx.rows - ry, 2 * margin);
    if (rw <= 10 || rh <= 10) return initial_pose;

    cv::Rect roi(rx, ry, rw, rh);
    cv::Mat local_dx = scene_dx(roi);
    cv::Mat local_dy = scene_dy(roi);

    std::vector<float> se_x, se_y, se_nx, se_ny;
    int NS = extractSceneEdgesSoA(local_dx, local_dy, (float)rx, (float)ry,
                                  se_x, se_y, se_nx, se_ny);
    if (NS == 0) return initial_pose;

    return refineInverseCore(templ_scene, TW, TH,
                             se_x.data(), se_y.data(),
                             se_nx.data(), se_ny.data(), NS,
                             initial_pose, config);
}

} // namespace icp_refine
