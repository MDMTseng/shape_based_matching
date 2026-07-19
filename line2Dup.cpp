#include "line2Dup.h"
#include "sbm_log.h"
#include <iostream>

// Gradient kernel for fused quantize path:
//   SBM_GRADIENT_KERNEL_CENTRAL_DIFF (0): p[c+1]-p[c-1], 4 loads, needs Gaussian pre-blur
//   SBM_GRADIENT_KERNEL_SOBEL3X3     (1): Sobel 3x3, 8 loads, has built-in [1,2,1] smoothing
#define SBM_GRADIENT_KERNEL_CENTRAL_DIFF 0
#define SBM_GRADIENT_KERNEL_SOBEL3X3     1
#ifndef SBM_GRADIENT_KERNEL
#define SBM_GRADIENT_KERNEL SBM_GRADIENT_KERNEL_CENTRAL_DIFF
#endif

#ifdef __AVX2__
#include <immintrin.h>
#elif defined(_MSC_VER) && defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

using namespace std;
using namespace cv;

#include <chrono>
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        sbm::sbm_log(sbm::LogLevel::Info, "%s\nelasped time:%.7fs\n", message.c_str(), t);
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

// Per-stage profiling accumulator (thread-safe for single match() call)
struct StageProfile {
    double blur_ms = 0;
    double sobel_ms = 0;
    double quantize_ms = 0;
    double voting_ms = 0;
    double fused_spread_lut_ms = 0;
    double coarse_match_ms = 0;
    double refine_ms = 0;
    double sort_nms_ms = 0;
    bool enabled = false;

    void print() const {
        if (!enabled) return;
        sbm::sbm_log(sbm::LogLevel::Debug, "profile", "  %-28s %7.1fms", "GaussianBlur 7x7", blur_ms);
        sbm::sbm_log(sbm::LogLevel::Debug, "profile", "  %-28s %7.1fms", "Sobel dx+dy (int16)", sobel_ms);
        sbm::sbm_log(sbm::LogLevel::Debug, "profile", "  %-28s %7.1fms", "Quantize (comparison)", quantize_ms);
        sbm::sbm_log(sbm::LogLevel::Debug, "profile", "  %-28s %7.1fms", "3x3 voting", voting_ms);
        sbm::sbm_log(sbm::LogLevel::Debug, "profile", "  %-28s %7.1fms", "Fused spread+LUT+linearize", fused_spread_lut_ms);
        sbm::sbm_log(sbm::LogLevel::Debug, "profile", "  %-28s %7.1fms", "Coarse similarity", coarse_match_ms);
        sbm::sbm_log(sbm::LogLevel::Debug, "profile", "  %-28s %7.1fms", "Pyramid refinement", refine_ms);
        sbm::sbm_log(sbm::LogLevel::Debug, "profile", "  %-28s %7.1fms", "Sort + NMS", sort_nms_ms);
        double total = blur_ms + sobel_ms + quantize_ms + voting_ms +
                       fused_spread_lut_ms + coarse_match_ms + refine_ms + sort_nms_ms;
        sbm::sbm_log(sbm::LogLevel::Debug, "profile", "  %-28s %7.1fms", "TOTAL", total);
    }
    void reset() {
        blur_ms = sobel_ms = quantize_ms = voting_ms = 0;
        fused_spread_lut_ms = coarse_match_ms = refine_ms = sort_nms_ms = 0;
    }
};
// Note: not thread-safe for concurrent matching calls. Profile data may be
// inaccurate under OpenMP.
static StageProfile g_profile;

namespace line2Dup
{
// Minimum 3x3 neighborhood votes for a quantized orientation to be accepted.
static const int NEIGHBOR_THRESHOLD = 5;
/**
 * \brief Get the label [0,8) of the single bit set in quantized.
 */
static inline int getLabel(int quantized)
{
    switch (quantized)
    {
    case 1:
        return 0;
    case 2:
        return 1;
    case 4:
        return 2;
    case 8:
        return 3;
    case 16:
        return 4;
    case 32:
        return 5;
    case 64:
        return 6;
    case 128:
        return 7;
    default:
        CV_Error(Error::StsBadArg, "Invalid value of quantized parameter");
        return -1; //avoid warning
    }
}

void Feature::read(const FileNode &fn)
{
    FileNodeIterator fni = fn.begin();
    fni >> x >> y >> label;
}

void Feature::write(FileStorage &fs) const
{
    fs << "[:" << x << y << label << "]";
}

void Template::read(const FileNode &fn)
{
    width = fn["width"];
    height = fn["height"];
    tl_x = fn["tl_x"];
    tl_y = fn["tl_y"];
    pyramid_level = fn["pyramid_level"];

    FileNode features_fn = fn["features"];
    features.resize(features_fn.size());
    FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
    for (int i = 0; it != it_end; ++it, ++i)
    {
        features[i].read(*it);
    }
}

void Template::write(FileStorage &fs) const
{
    fs << "width" << width;
    fs << "height" << height;
    fs << "tl_x" << tl_x;
    fs << "tl_y" << tl_y;
    fs << "pyramid_level" << pyramid_level;

    fs << "features"
       << "[";
    for (int i = 0; i < (int)features.size(); ++i)
    {
        features[i].write(fs);
    }
    fs << "]"; // features
}

static Rect cropTemplates(std::vector<Template> &templates)
{
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();

    // First pass: find min/max feature x,y over all pyramid levels and modalities
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            int x = templ.features[j].x << templ.pyramid_level;
            int y = templ.features[j].y << templ.pyramid_level;
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }

    /// @todo Why require even min_x, min_y?
    if (min_x % 2 == 1)
        --min_x;
    if (min_y % 2 == 1)
        --min_y;

    // Second pass: set width/height and shift all feature positions
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];
        templ.width = (max_x - min_x) >> templ.pyramid_level;
        templ.height = (max_y - min_y) >> templ.pyramid_level;
        templ.tl_x = min_x >> templ.pyramid_level;
        templ.tl_y = min_y  >> templ.pyramid_level;

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            templ.features[j].x -= templ.tl_x;
            templ.features[j].y -= templ.tl_y;
        }
    }

    return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

bool ColorGradientPyramid::selectScatteredFeatures(const std::vector<Candidate> &candidates,
                                                   std::vector<Feature> &features,
                                                   size_t num_features, float distance)
{
    features.clear();
    float distance_sq = distance * distance;
    int i = 0;

    bool first_select = true;

    while(true)
    {
        Candidate c = candidates[i];

        // Add if sufficient distance away from any previously chosen feature
        bool keep = true;
        for (int j = 0; (j < (int)features.size()) && keep; ++j)
        {
            Feature f = features[j];
            keep = (c.f.x - f.x) * (c.f.x - f.x) + (c.f.y - f.y) * (c.f.y - f.y) >= distance_sq;
        }
        if (keep)
            features.push_back(c.f);

        if (++i == (int)candidates.size()){
            bool num_ok = features.size() >= num_features;

            if(first_select){
                if(num_ok){
                    features.clear(); // we don't want too many first time
                    i = 0;
                    distance += 1.0f;
                    distance_sq = distance * distance;
                    continue;
                }else{
                    first_select = false;
                }
            }

            // Start back at beginning, and relax required distance
            i = 0;
            distance -= 1.0f;
            distance_sq = distance * distance;
             if (num_ok || distance < 3){
                 break;
             }
        }
    }
    return true;
}

/****************************************************************************************\
*                                                         Color gradient ColorGradient                                                                        *
\****************************************************************************************/

void hysteresisGradient(Mat &magnitude, Mat &quantized_angle,
                        Mat &angle, float threshold)
{
    // Quantize 360 degree range of orientations into 16 buckets
    // Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
    // for stability of horizontal and vertical features.
    Mat_<unsigned char> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

    // Zero out top and bottom rows
    /// @todo is this necessary, or even correct?
    memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
    memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
    // Zero out first and last columns
    for (int r = 0; r < quantized_unfiltered.rows; ++r)
    {
        quantized_unfiltered(r, 0) = 0;
        quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
    }

    // Mask 16 buckets into 8 quantized orientations
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
        for (int c = 1; c < angle.cols - 1; ++c)
        {
            quant_r[c] &= 7;
        }
    }

    // Filter the raw quantized image. Only accept pixels where the magnitude is above some
    // threshold, and there is local agreement on the quantization.
    quantized_angle = Mat::zeros(angle.size(), CV_8U);
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        float *mag_r = magnitude.ptr<float>(r);

        for (int c = 1; c < angle.cols - 1; ++c)
        {
            if (mag_r[c] > threshold)
            {
                // Compute histogram of quantized bins in 3x3 patch around pixel
                int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                uchar *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                // Find bin with the most votes from the patch
                int max_votes = 0;
                int index = -1;
                for (int i = 0; i < 8; ++i)
                {
                    if (max_votes < histogram[i])
                    {
                        index = i;
                        max_votes = histogram[i];
                    }
                }

                // Only accept the quantization if majority of pixels in the patch agree
                if (max_votes >= NEIGHBOR_THRESHOLD)
                    quantized_angle.at<uchar>(r, c) = uchar(1 << index);
            }
        }
    }
}

static void quantizedOrientations(const Mat &src, Mat &magnitude,
                                  Mat &angle, Mat& angle_ori, float threshold,
                                  int blur_kernel_size = 7,
                                  bool match_only = false,
                                  bool skip_voting = false)
{
    using PClock = std::chrono::high_resolution_clock;
    auto pnow = []() { return PClock::now(); };
    auto pms = [](PClock::time_point t0) {
        return std::chrono::duration<double, std::milli>(PClock::now() - t0).count();
    };

    Mat smoothed;
    int ks = blur_kernel_size | 1;  // ensure odd
    auto pt0 = pnow();
    if (ks <= 1)
        smoothed = src;
    else
        GaussianBlur(src, smoothed, Size(ks, ks), 0, 0, BORDER_REPLICATE);
    if (g_profile.enabled) g_profile.blur_ms += pms(pt0);

    if(src.channels() == 1){

        // Fixed-point tan boundaries for 8-bin orientation quantization.
        static const int TAN_B[4] = {1989, 6682, 14966, 50273};
        float threshold_sq = threshold * threshold;
        int threshold_sq_i = (int)threshold_sq;

        if (skip_voting && match_only) {
            // Fully fused path: Sobel + Quantize + Bitmask in one pass.
            // Reads smoothed image once, writes angle bitmask once.
            // Eliminates dx, dy, quantized_unfiltered, mag_mask intermediates.
            // Memory traffic: read ~60MB (3 rows × width per pixel) → write 20MB = ~80MB
            // vs original: ~340MB across separate Sobel+Quantize+Vote passes.
            pt0 = pnow();
            angle = Mat::zeros(src.size(), CV_8U);

            #pragma omp parallel for schedule(static)
            for (int r = 1; r < src.rows - 1; ++r) {
                const uchar *row_prev = smoothed.ptr<uchar>(r-1);
                const uchar *row_curr = smoothed.ptr<uchar>(r);
                const uchar *row_next = smoothed.ptr<uchar>(r+1);
                uchar *angle_r = angle.ptr<uchar>(r);

                int c = 1;
#ifdef __AVX2__
                // Constants for int32 quantization (used for survivor batches)
                const __m256i zero32 = _mm256_setzero_si256();
                const __m256i four32 = _mm256_set1_epi32(4);
                const __m256i seven32 = _mm256_set1_epi32(7);
                const __m256i ten_k = _mm256_set1_epi32(10000);
                const __m256i tan0v = _mm256_set1_epi32(TAN_B[0]);
                const __m256i tan1v = _mm256_set1_epi32(TAN_B[1]);
                const __m256i tan2v = _mm256_set1_epi32(TAN_B[2]);
                const __m256i tan3v = _mm256_set1_epi32(TAN_B[3]);
                const __m256i one32 = _mm256_set1_epi32(1);
                // L1 threshold: |dx|+|dy| > T is slightly more permissive than dx²+dy²>T²
                // Threshold scaling: central diff is ~4× smaller than Sobel
#if SBM_GRADIENT_KERNEL == SBM_GRADIENT_KERNEL_CENTRAL_DIFF
                int edge_thresh = std::max(1, (int)threshold / 4);
#else
                int edge_thresh = (int)threshold;
#endif
                const __m256i thresh_l1_v = _mm256_set1_epi16((short)edge_thresh);
                const __m256i thresh_sq_v = _mm256_set1_epi32(edge_thresh * edge_thresh);

                for (; c <= src.cols - 1 - 16; c += 16) {
#if SBM_GRADIENT_KERNEL == SBM_GRADIENT_KERNEL_CENTRAL_DIFF
                    // Central difference: 4 loads, needs pre-blur (Gaussian)
                    __m256i curr_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(row_curr + c - 1)));
                    __m256i curr_r = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(row_curr + c + 1)));
                    __m256i prev_c = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(row_prev + c)));
                    __m256i next_c = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(row_next + c)));
                    __m256i gx16 = _mm256_sub_epi16(curr_r, curr_l);
                    __m256i gy16 = _mm256_sub_epi16(next_c, prev_c);
#else
                    // Sobel 3x3: 8 loads (3 per row + shifts), built-in [1,2,1] smoothing
                    __m128i prev_raw = _mm_loadu_si128((const __m128i*)(row_prev + c - 1));
                    __m128i curr_raw = _mm_loadu_si128((const __m128i*)(row_curr + c - 1));
                    __m128i next_raw = _mm_loadu_si128((const __m128i*)(row_next + c - 1));
                    // Widen to int16 for 16-wide processing (only need first 16 pixels from 18-byte load)
                    __m256i prev_l = _mm256_cvtepu8_epi16(prev_raw);
                    __m256i prev_c = _mm256_cvtepu8_epi16(_mm_srli_si128(prev_raw, 1));
                    __m256i prev_r = _mm256_cvtepu8_epi16(_mm_srli_si128(prev_raw, 2));
                    __m256i curr_l = _mm256_cvtepu8_epi16(curr_raw);
                    __m256i curr_r = _mm256_cvtepu8_epi16(_mm_srli_si128(curr_raw, 2));
                    __m256i next_l = _mm256_cvtepu8_epi16(next_raw);
                    __m256i next_c = _mm256_cvtepu8_epi16(_mm_srli_si128(next_raw, 1));
                    __m256i next_r = _mm256_cvtepu8_epi16(_mm_srli_si128(next_raw, 2));
                    __m256i gx16 = _mm256_add_epi16(
                        _mm256_add_epi16(_mm256_sub_epi16(prev_r, prev_l), _mm256_sub_epi16(next_r, next_l)),
                        _mm256_slli_epi16(_mm256_sub_epi16(curr_r, curr_l), 1));
                    __m256i gy16 = _mm256_sub_epi16(
                        _mm256_add_epi16(_mm256_add_epi16(next_l, next_r), _mm256_slli_epi16(next_c, 1)),
                        _mm256_add_epi16(_mm256_add_epi16(prev_l, prev_r), _mm256_slli_epi16(prev_c, 1)));
#endif

                    // L1 pre-filter in int16 (fast rejection of most pixels)
                    __m256i l1_mag = _mm256_adds_epu16(
                        _mm256_abs_epi16(gx16), _mm256_abs_epi16(gy16));
                    __m256i above16 = _mm256_cmpgt_epi16(l1_mag, thresh_l1_v);
                    int above_bits = _mm256_movemask_epi8(above16);
                    if (above_bits == 0) continue;

                    // Process lower 8 and upper 8 pixels separately (quantize needs int32)
                    for (int half = 0; half < 2; ++half) {
                        int h_bits = (half == 0) ? (above_bits & 0xFFFF) : (above_bits >> 16);
                        if (!h_bits) continue;
                        __m256i gx = _mm256_cvtepi16_epi32(half == 0
                            ? _mm256_castsi256_si128(gx16)
                            : _mm256_extracti128_si256(gx16, 1));
                        __m256i gy = _mm256_cvtepi16_epi32(half == 0
                            ? _mm256_castsi256_si128(gy16)
                            : _mm256_extracti128_si256(gy16, 1));

                        // L2 threshold on survivors
                        __m256i mag_sq = _mm256_add_epi32(
                            _mm256_mullo_epi32(gx, gx), _mm256_mullo_epi32(gy, gy));
                        __m256i above = _mm256_cmpgt_epi32(mag_sq, thresh_sq_v);
                        int above32 = _mm256_movemask_ps(_mm256_castsi256_ps(above));
                        if (above32 == 0) continue;

                        // Quantize to 8-bin orientation
                        __m256i gy_neg = _mm256_cmpgt_epi32(zero32, gy);
                        __m256i ugx = _mm256_blendv_epi8(gx, _mm256_sub_epi32(zero32, gx), gy_neg);
                        __m256i ugy = _mm256_blendv_epi8(gy, _mm256_sub_epi32(zero32, gy), gy_neg);
                        ugx = _mm256_blendv_epi8(ugx, _mm256_sub_epi32(zero32, ugx),
                            _mm256_and_si256(_mm256_cmpeq_epi32(ugy, zero32),
                                             _mm256_cmpgt_epi32(zero32, ugx)));
                        __m256i abs_ugx = _mm256_abs_epi32(ugx);
                        __m256i test_y = _mm256_mullo_epi32(ugy, ten_k);
                        __m256i cnt = _mm256_and_si256(one32,
                            _mm256_cmpgt_epi32(_mm256_mullo_epi32(abs_ugx, tan0v), test_y));
                        cnt = _mm256_add_epi32(cnt, _mm256_and_si256(one32,
                            _mm256_cmpgt_epi32(_mm256_mullo_epi32(abs_ugx, tan1v), test_y)));
                        cnt = _mm256_add_epi32(cnt, _mm256_and_si256(one32,
                            _mm256_cmpgt_epi32(_mm256_mullo_epi32(abs_ugx, tan2v), test_y)));
                        cnt = _mm256_add_epi32(cnt, _mm256_and_si256(one32,
                            _mm256_cmpgt_epi32(_mm256_mullo_epi32(abs_ugx, tan3v), test_y)));
                        __m256i bin = _mm256_blendv_epi8(
                            _mm256_sub_epi32(four32, cnt),
                            _mm256_and_si256(_mm256_add_epi32(four32, cnt), seven32),
                            _mm256_cmpgt_epi32(zero32, ugx));

                        // Store bitmask
                        alignas(32) int bin_arr[8];
                        _mm256_store_si256((__m256i*)bin_arr, bin);
                        int offset = half * 8;
                        for (int i = 0; i < 8; ++i)
                            if (above32 & (1 << i))
                                angle_r[c + offset + i] = (uchar)(1 << bin_arr[i]);
                    }
                }
#endif
                // Scalar tail
                for (; c < src.cols - 1; ++c) {
#if SBM_GRADIENT_KERNEL == SBM_GRADIENT_KERNEL_CENTRAL_DIFF
                    // Central difference
                    int gx = row_curr[c+1] - row_curr[c-1];
                    int gy = row_next[c] - row_prev[c];
                    int et = std::max(1, (int)threshold / 4);
#else
                    // Sobel 3x3
                    int gx = (row_prev[c+1] - row_prev[c-1]) + 2*(row_curr[c+1] - row_curr[c-1])
                           + (row_next[c+1] - row_next[c-1]);
                    int gy = (row_next[c-1] + 2*row_next[c] + row_next[c+1])
                           - (row_prev[c-1] + 2*row_prev[c] + row_prev[c+1]);
                    int et = (int)threshold;
#endif
                    if (gx*gx + gy*gy <= et * et) continue;
                    int ugx = gx, ugy = gy;
                    if (ugy < 0) { ugx = -ugx; ugy = -ugy; }
                    if (ugy == 0 && ugx < 0) ugx = -ugx;
                    int bin;
                    if (ugx >= 0) {
                        int ty = ugy * 10000;
                        if      (ty < ugx * TAN_B[0]) bin = 0;
                        else if (ty < ugx * TAN_B[1]) bin = 1;
                        else if (ty < ugx * TAN_B[2]) bin = 2;
                        else if (ty < ugx * TAN_B[3]) bin = 3;
                        else bin = 4;
                    } else {
                        int agx = -ugx, ty = ugy * 10000;
                        if      (ty < agx * TAN_B[0]) bin = 0;
                        else if (ty < agx * TAN_B[1]) bin = 7;
                        else if (ty < agx * TAN_B[2]) bin = 6;
                        else if (ty < agx * TAN_B[3]) bin = 5;
                        else bin = 4;
                    }
                    angle_r[c] = (uchar)(1 << bin);
                }
            }
            if (g_profile.enabled) {
                g_profile.sobel_ms += pms(pt0);
                // quantize+voting time is included in sobel_ms for fused path
            }
        } else {
        // Non-fused path: separate Sobel → Quantize → Vote
        Mat sobel_dx_16s, sobel_dy_16s;
        pt0 = pnow();
        #pragma omp parallel sections
        {
            #pragma omp section
            Sobel(smoothed, sobel_dx_16s, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
            #pragma omp section
            Sobel(smoothed, sobel_dy_16s, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
        }
        if (g_profile.enabled) g_profile.sobel_ms += pms(pt0);

        if (!match_only) {
            magnitude.create(src.size(), CV_32F);
            magnitude.setTo(0);
            // Compute directed gradient angle (0-360) from Sobel for feature theta
            angle_ori.create(src.size(), CV_32F);
            Mat dx_f, dy_f;
            sobel_dx_16s.convertTo(dx_f, CV_32F);
            sobel_dy_16s.convertTo(dy_f, CV_32F);
            cv::phase(dx_f, dy_f, angle_ori, true);  // true = degrees
        }

        if (skip_voting) {
            // Fused quantize+bitmask (but separate from Sobel)
            pt0 = pnow();
            angle = Mat::zeros(src.size(), CV_8U);

            #pragma omp parallel for schedule(static)
            for (int r = 1; r < src.rows - 1; ++r) {
                const short *dx = sobel_dx_16s.ptr<short>(r);
                const short *dy = sobel_dy_16s.ptr<short>(r);
                float *mag_r = match_only ? nullptr : magnitude.ptr<float>(r);
                uchar *angle_r = angle.ptr<uchar>(r);
                for (int c = 1; c < src.cols - 1; ++c) {
                    int gx = dx[c], gy = dy[c];
                    int mag_sq_i = (int)((int64_t)gx*gx + (int64_t)gy*gy);
                    if (mag_r) mag_r[c] = (float)mag_sq_i;
                    if (mag_sq_i <= threshold_sq_i) continue;
                    int ugx = gx, ugy = gy;
                    if (ugy < 0) { ugx = -ugx; ugy = -ugy; }
                    if (ugy == 0 && ugx < 0) ugx = -ugx;
                    int bin;
                    if (ugx >= 0) {
                        int ty = ugy * 10000;
                        if      (ty < ugx * TAN_B[0]) bin = 0;
                        else if (ty < ugx * TAN_B[1]) bin = 1;
                        else if (ty < ugx * TAN_B[2]) bin = 2;
                        else if (ty < ugx * TAN_B[3]) bin = 3;
                        else bin = 4;
                    } else {
                        int agx = -ugx, ty = ugy * 10000;
                        if      (ty < agx * TAN_B[0]) bin = 0;
                        else if (ty < agx * TAN_B[1]) bin = 7;
                        else if (ty < agx * TAN_B[2]) bin = 6;
                        else if (ty < agx * TAN_B[3]) bin = 5;
                        else bin = 4;
                    }
                    angle_r[c] = (uchar)(1 << bin);
                }
            }
            if (g_profile.enabled) g_profile.voting_ms += pms(pt0);
        } else {
        // Voting path: need quantize → intermediate buffers → vote → bitmask
        Mat quantized_unfiltered = Mat::zeros(src.size(), CV_8U);
        Mat mag_mask = Mat::zeros(src.size(), CV_8U);

        #pragma omp parallel for schedule(static)
        for (int r = 1; r < src.rows - 1; ++r) {
            const short *dx = sobel_dx_16s.ptr<short>(r);
            const short *dy = sobel_dy_16s.ptr<short>(r);
            float *mag_r = match_only ? nullptr : magnitude.ptr<float>(r);
            uchar *qr = quantized_unfiltered.ptr<uchar>(r);
            uchar *mask_r = mag_mask.ptr<uchar>(r);
            for (int c = 1; c < src.cols - 1; ++c) {
                int gx = dx[c], gy = dy[c];
                int mag_sq_i = (int)((int64_t)gx*gx + (int64_t)gy*gy);
                if (mag_r) mag_r[c] = (float)mag_sq_i;
                if (mag_sq_i <= threshold_sq_i) continue;
                mask_r[c] = 0xFF;
                int ugx = gx, ugy = gy;
                if (ugy < 0) { ugx = -ugx; ugy = -ugy; }
                if (ugy == 0 && ugx < 0) ugx = -ugx;
                int bin;
                if (ugx >= 0) {
                    int ty = ugy * 10000;
                    if      (ty < ugx * TAN_B[0]) bin = 0;
                    else if (ty < ugx * TAN_B[1]) bin = 1;
                    else if (ty < ugx * TAN_B[2]) bin = 2;
                    else if (ty < ugx * TAN_B[3]) bin = 3;
                    else bin = 4;
                } else {
                    int agx = -ugx, ty = ugy * 10000;
                    if      (ty < agx * TAN_B[0]) bin = 0;
                    else if (ty < agx * TAN_B[1]) bin = 7;
                    else if (ty < agx * TAN_B[2]) bin = 6;
                    else if (ty < agx * TAN_B[3]) bin = 5;
                    else bin = 4;
                }
                qr[c] = (uchar)bin;
            }
        }
        if (g_profile.enabled) g_profile.quantize_ms += pms(pt0);

        pt0 = pnow();
        angle = Mat::zeros(src.size(), CV_8U);
        // 3x3 neighborhood voting: keep bin only if >= NEIGHBOR_THRESHOLD neighbors agree.
#ifdef __AVX2__
        alignas(16) static const uchar bin_to_bit[16] = {
            1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0
        };
        const __m256i bit_lut = _mm256_broadcastsi128_si256(
            _mm_load_si128((const __m128i*)bin_to_bit));
        const __m256i one = _mm256_set1_epi8(1);
        const __m256i thresh_vote = _mm256_set1_epi8((char)(NEIGHBOR_THRESHOLD - 1));

        #pragma omp parallel for schedule(static)
        for (int r = 1; r < src.rows - 1; ++r) {
            const uchar *q_prev = quantized_unfiltered.ptr<uchar>(r-1);
            const uchar *q_curr = quantized_unfiltered.ptr<uchar>(r);
            const uchar *q_next = quantized_unfiltered.ptr<uchar>(r+1);
            const uchar *mask_r = mag_mask.ptr<uchar>(r);
            uchar *angle_r = angle.ptr<uchar>(r);

            int c = 1;
            for (; c <= src.cols - 1 - 32; c += 32) {
                __m256i center = _mm256_loadu_si256((const __m256i*)(q_curr + c));
                __m256i votes = one;
                votes = _mm256_add_epi8(votes, _mm256_and_si256(one,
                    _mm256_cmpeq_epi8(center, _mm256_loadu_si256((const __m256i*)(q_prev + c - 1)))));
                votes = _mm256_add_epi8(votes, _mm256_and_si256(one,
                    _mm256_cmpeq_epi8(center, _mm256_loadu_si256((const __m256i*)(q_prev + c)))));
                votes = _mm256_add_epi8(votes, _mm256_and_si256(one,
                    _mm256_cmpeq_epi8(center, _mm256_loadu_si256((const __m256i*)(q_prev + c + 1)))));
                votes = _mm256_add_epi8(votes, _mm256_and_si256(one,
                    _mm256_cmpeq_epi8(center, _mm256_loadu_si256((const __m256i*)(q_curr + c - 1)))));
                votes = _mm256_add_epi8(votes, _mm256_and_si256(one,
                    _mm256_cmpeq_epi8(center, _mm256_loadu_si256((const __m256i*)(q_curr + c + 1)))));
                votes = _mm256_add_epi8(votes, _mm256_and_si256(one,
                    _mm256_cmpeq_epi8(center, _mm256_loadu_si256((const __m256i*)(q_next + c - 1)))));
                votes = _mm256_add_epi8(votes, _mm256_and_si256(one,
                    _mm256_cmpeq_epi8(center, _mm256_loadu_si256((const __m256i*)(q_next + c)))));
                votes = _mm256_add_epi8(votes, _mm256_and_si256(one,
                    _mm256_cmpeq_epi8(center, _mm256_loadu_si256((const __m256i*)(q_next + c + 1)))));
                __m256i pass = _mm256_cmpgt_epi8(votes, thresh_vote);
                __m256i mag_ok = _mm256_loadu_si256((const __m256i*)(mask_r + c));
                __m256i bitmask = _mm256_shuffle_epi8(bit_lut, center);
                _mm256_storeu_si256((__m256i*)(angle_r + c),
                    _mm256_and_si256(bitmask, _mm256_and_si256(pass, mag_ok)));
            }
            for (; c < src.cols - 1; ++c) {
                if (mask_r[c]) {
                    uchar center_bin = q_curr[c];
                    int votes = 1;
                    votes += (q_prev[c-1] == center_bin); votes += (q_prev[c] == center_bin);
                    votes += (q_prev[c+1] == center_bin); votes += (q_curr[c-1] == center_bin);
                    votes += (q_curr[c+1] == center_bin); votes += (q_next[c-1] == center_bin);
                    votes += (q_next[c] == center_bin);   votes += (q_next[c+1] == center_bin);
                    if (votes >= NEIGHBOR_THRESHOLD)
                        angle_r[c] = (uchar)(1 << center_bin);
                }
            }
        }
#else
        #pragma omp parallel for schedule(static)
        for (int r = 1; r < src.rows - 1; ++r) {
            const uchar *q_prev = quantized_unfiltered.ptr<uchar>(r-1);
            const uchar *q_curr = quantized_unfiltered.ptr<uchar>(r);
            const uchar *q_next = quantized_unfiltered.ptr<uchar>(r+1);
            const uchar *mask_r = mag_mask.ptr<uchar>(r);
            uchar *angle_r = angle.ptr<uchar>(r);
            for (int c = 1; c < src.cols - 1; ++c) {
                if (mask_r[c]) {
                    uchar center_bin = q_curr[c];
                    int votes = 1;
                    votes += (q_prev[c-1] == center_bin); votes += (q_prev[c] == center_bin);
                    votes += (q_prev[c+1] == center_bin); votes += (q_curr[c-1] == center_bin);
                    votes += (q_curr[c+1] == center_bin); votes += (q_next[c-1] == center_bin);
                    votes += (q_next[c] == center_bin);   votes += (q_next[c+1] == center_bin);
                    if (votes >= NEIGHBOR_THRESHOLD)
                        angle_r[c] = (uchar)(1 << center_bin);
                }
            }
        }
#endif
        } // end voting

        if (g_profile.enabled) g_profile.voting_ms += pms(pt0);
        } // end non-fused path

    }else{

        magnitude.create(src.size(), CV_32F);

        // Allocate temporary buffers
        Size size = src.size();
        Mat sobel_3dx;              // per-channel horizontal derivative
        Mat sobel_3dy;              // per-channel vertical derivative
        Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
        Mat sobel_dy(size, CV_32F); // maximum vertical derivative
        Mat sobel_ag;               // final gradient orientation (unquantized)

        Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
        Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);

        short *ptrx = (short *)sobel_3dx.data;
        short *ptry = (short *)sobel_3dy.data;
        float *ptr0x = (float *)sobel_dx.data;
        float *ptr0y = (float *)sobel_dy.data;
        float *ptrmg = (float *)magnitude.data;

        const int length1 = static_cast<const int>(sobel_3dx.step1());
        const int length2 = static_cast<const int>(sobel_3dy.step1());
        const int length3 = static_cast<const int>(sobel_dx.step1());
        const int length4 = static_cast<const int>(sobel_dy.step1());
        const int length5 = static_cast<const int>(magnitude.step1());
        const int length0 = sobel_3dy.cols * 3;

        for (int r = 0; r < sobel_3dy.rows; ++r)
        {
            int ind = 0;

            for (int i = 0; i < length0; i += 3)
            {
                // Use the gradient orientation of the channel whose magnitude is largest
                int mag1 = ptrx[i + 0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];
                int mag2 = ptrx[i + 1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
                int mag3 = ptrx[i + 2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];

                if (mag1 >= mag2 && mag1 >= mag3)
                {
                    ptr0x[ind] = ptrx[i];
                    ptr0y[ind] = ptry[i];
                    ptrmg[ind] = (float)mag1;
                }
                else if (mag2 >= mag1 && mag2 >= mag3)
                {
                    ptr0x[ind] = ptrx[i + 1];
                    ptr0y[ind] = ptry[i + 1];
                    ptrmg[ind] = (float)mag2;
                }
                else
                {
                    ptr0x[ind] = ptrx[i + 2];
                    ptr0y[ind] = ptry[i + 2];
                    ptrmg[ind] = (float)mag3;
                }
                ++ind;
            }
            ptrx += length1;
            ptry += length2;
            ptr0x += length3;
            ptr0y += length4;
            ptrmg += length5;
        }

        // Calculate the final gradient orientations
        phase(sobel_dx, sobel_dy, sobel_ag, true);
        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
        angle_ori = sobel_ag;
    }


}

ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
                                           float _weak_threshold, size_t _num_features,
                                           float _strong_threshold,
                                           bool _match_only)
    : src(_src),
      mask(_mask),
      pyramid_level(0),
      weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold),
      blur_kernel_size(7),
      match_only(_match_only)
{
    update();
}

void ColorGradientPyramid::update()
{
    quantizedOrientations(src, magnitude, angle, angle_ori, weak_threshold, blur_kernel_size, match_only, skip_voting);
}

void ColorGradientPyramid::pyrDown()
{
    // Some parameters need to be adjusted
    num_features /= 2; /// @todo Why not 4?
    ++pyramid_level;

    // Downsample the current inputs
    Size size(src.cols / 2, src.rows / 2);
    Mat next_src;
    cv::pyrDown(src, next_src, size);
    src = next_src;

    if (!mask.empty())
    {
        Mat next_mask;
        resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
        mask = next_mask;
    }

    update();
}

void ColorGradientPyramid::quantize(Mat &dst) const
{
    if (mask.empty()) {
        // No mask: direct reference, no copy (saves 40MB at 20MP)
        dst = angle;
    } else {
        dst = Mat::zeros(angle.size(), CV_8U);
        angle.copyTo(dst, mask);
    }
}

bool ColorGradientPyramid::extractTemplate(Template &templ) const
{
    // Want features on the border to distinguish from background
    Mat local_mask;
    if (!mask.empty())
    {
        erode(mask, local_mask, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
//        subtract(mask, local_mask, local_mask);
    }

    std::vector<Candidate> candidates;
    bool no_mask = local_mask.empty();
    float threshold_sq = strong_threshold * strong_threshold;

    int nms_kernel_size = 5;
    int half_nms = nms_kernel_size / 2;
    cv::Mat magnitude_valid = cv::Mat(magnitude.size(), CV_8UC1, cv::Scalar(255));

    // Precompute row pointers for magnitude to avoid .at<> bounds checks
    int mag_step = static_cast<int>(magnitude.step1());
    int valid_step = static_cast<int>(magnitude_valid.step1());
    const float *mag_data = magnitude.ptr<float>(0);
    uchar *valid_data = magnitude_valid.ptr<uchar>(0);

    for (int r = half_nms; r < magnitude.rows - half_nms; ++r)
    {
        const uchar *mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);
        const float *mag_r = mag_data + r * mag_step;
        uchar *valid_r = valid_data + r * valid_step;
        const uchar *angle_r = angle.ptr<uchar>(r);
        const float *angle_ori_r = angle_ori.ptr<float>(r);

        for (int c = half_nms; c < magnitude.cols - half_nms; ++c)
        {
            if (no_mask || mask_r[c])
            {
                float score = 0;
                if (valid_r[c] > 0) {
                    score = mag_r[c];
                    bool is_max = true;
                    for (int ro = -half_nms; ro <= half_nms && is_max; ++ro) {
                        const float *mag_nr = mag_data + (r + ro) * mag_step;
                        for (int co = -half_nms; co <= half_nms; ++co) {
                            if (ro == 0 && co == 0) continue;
                            if (score < mag_nr[c + co]) {
                                score = 0;
                                is_max = false;
                                break;
                            }
                        }
                    }

                    if (is_max) {
                        for (int ro = -half_nms; ro <= half_nms; ++ro) {
                            uchar *valid_nr = valid_data + (r + ro) * valid_step;
                            for (int co = -half_nms; co <= half_nms; ++co) {
                                if (ro == 0 && co == 0) continue;
                                valid_nr[c + co] = 0;
                            }
                        }
                    }
                }

                if (score > threshold_sq && angle_r[c] > 0)
                {
                    candidates.push_back(Candidate(c, r, getLabel(angle_r[c]), score));
                    float theta = angle_ori_r[c];
                    if (theta == 0.0f && score > 0) {
                        int label = candidates.back().f.label;
                        int bin = 0;
                        while (bin < 8 && !(label & (1 << bin))) ++bin;
                        theta = bin * 22.5f;
                    }
                    candidates.back().f.theta = theta;
                }
            }
        }
    }
    // We require a certain number of features
    if (candidates.size() < num_features){
        if(candidates.size() <= 4) {
            sbm::sbm_log(sbm::LogLevel::Warning, "feature", "too few features, abort");
            return false;
        }
        sbm::sbm_log(sbm::LogLevel::Warning, "feature", "have no enough features, exhaustive mode");
    }

    // Multi-scale orientation consensus: check each candidate's label at
    // additional blur levels (simulating downscaled scene). Boost stable features
    // so selectScatteredFeatures prefers them. This costs ~2 Sobel+phase calls
    // on the template (tiny) but gives much better stability under match_scale.
    if (!match_only && candidates.size() > 4) {
        // Blur levels: sigma ~ 1/scale. For match_scale=0.5 → sigma≈1, for 0.3 → sigma≈1.7
        float extra_sigmas[] = {1.0f, 2.0f};
        int n_extra = 2;

        // Precompute gradient bins at each blur level
        struct BlurLevel {
            cv::Mat dx, dy;
        };
        std::vector<BlurLevel> blur_levels(n_extra);
        for (int si = 0; si < n_extra; si++) {
            cv::Mat blurred;
            int ks = (int)(extra_sigmas[si] * 6) | 1;
            ks = std::max(ks, 3);
            cv::GaussianBlur(src, blurred, cv::Size(ks, ks), extra_sigmas[si]);
            cv::Sobel(blurred, blur_levels[si].dx, CV_16S, 1, 0, 3);
            cv::Sobel(blurred, blur_levels[si].dy, CV_16S, 0, 1, 3);
        }

        int n_stable_debug = 0, n_unstable_debug = 0, n_partial_debug = 0;
        for (auto& cand : candidates) {
            int cx = cand.f.x, cy = cand.f.y;
            if (cx < 1 || cy < 1 || cx >= src.cols-1 || cy >= src.rows-1) continue;

            int base_bin = 0;
            { int lbl = cand.f.label; while (base_bin < 8 && !(lbl & (1 << base_bin))) base_bin++; }

            int n_agree = 0;
            for (int si = 0; si < n_extra; si++) {
                short gx = blur_levels[si].dx.at<short>(cy, cx);
                short gy = blur_levels[si].dy.at<short>(cy, cx);
                // Undirected: force gy >= 0
                if (gy < 0) { gx = -gx; gy = -gy; }
                if (gy == 0 && gx < 0) gx = -gx;
                // Quick 8-bin quantize using atan2 approximation
                float theta = std::atan2((float)gy, (float)gx) * 180.0f / (float)CV_PI;
                if (theta < 0) theta += 180.0f;
                int bin = (int)(theta / 22.5f + 0.5f) & 7;
                if (bin == base_bin) n_agree++;
            }

            // Boost score for stable features: stable features get 1.5x score,
            // partially stable get 1.0x (unchanged), unstable get 0.7x.
            // This biases selectScatteredFeatures toward stable features
            // while still allowing unstable ones if no stable alternative exists.
            if (n_agree == n_extra) {
                cand.score *= 1.5f;  // fully stable across all blur levels
                n_stable_debug++;
            } else if (n_agree == 0) {
                cand.score *= 0.7f;  // unstable — orientation flips at all blur levels
                n_unstable_debug++;
            } else {
                n_partial_debug++;
            }
        }

        sbm::sbm_log(sbm::LogLevel::Info, "feature",
                     "multi-scale consensus: %d stable, %d partial, %d unstable (of %d)",
                     n_stable_debug, n_partial_debug, n_unstable_debug, (int)candidates.size());
    }

    // NOTE: Stable sort to agree with old code, which used std::list::sort()
    std::stable_sort(candidates.begin(), candidates.end());

    // Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
    float distance = static_cast<float>(candidates.size() / num_features + 1);

    // selectScatteredFeatures always return true
    if (!selectScatteredFeatures(candidates, templ.features, num_features, distance))
    {
        return false;
    }

    // Size determined externally, needs to match templates for other modalities
    templ.width = -1;
    templ.height = -1;
    templ.pyramid_level = pyramid_level;

    return true;
}

ColorGradient::ColorGradient()
    : weak_threshold(30.0f),
      num_features(63),
      strong_threshold(60.0f)
{
}

ColorGradient::ColorGradient(float _weak_threshold, size_t _num_features, float _strong_threshold)
    : weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
}

static const char CG_NAME[] = "ColorGradient";

std::string ColorGradient::name() const
{
    return CG_NAME;
}

void ColorGradient::read(const FileNode &fn)
{
    String type = fn["type"];
    CV_Assert(type == CG_NAME);

    weak_threshold = fn["weak_threshold"];
    num_features = int(fn["num_features"]);
    strong_threshold = fn["strong_threshold"];
}

void ColorGradient::write(FileStorage &fs) const
{
    fs << "type" << CG_NAME;
    fs << "weak_threshold" << weak_threshold;
    fs << "num_features" << int(num_features);
    fs << "strong_threshold" << strong_threshold;
}
/****************************************************************************************\
*                                                                 Response maps                                                                                    *
\****************************************************************************************/

static void orUnaligned8u(const uchar *src, const int src_stride,
                          uchar *dst, const int dst_stride,
                          const int width, const int height)
{
    for (int r = 0; r < height; ++r)
    {
        int c = 0;

#ifdef __AVX2__
        // AVX2: 32 bytes per iteration, unaligned loads (no penalty on Intel Haswell+)
        for (; c <= width - 32; c += 32) {
            __m256i s = _mm256_loadu_si256((const __m256i*)(src + c));
            __m256i d = _mm256_loadu_si256((const __m256i*)(dst + c));
            _mm256_storeu_si256((__m256i*)(dst + c), _mm256_or_si256(d, s));
        }
#else
        for (; c <= width - mipp::N<uint8_t>(); c += mipp::N<uint8_t>()) {
            mipp::Reg<uint8_t> src_v((uint8_t*)src + c);
            mipp::Reg<uint8_t> dst_v((uint8_t*)dst + c);
            mipp::orb(src_v, dst_v).store((uint8_t*)dst + c);
        }
#endif

        for (; c < width; c++)
            dst[c] |= src[c];

        src += src_stride;
        dst += dst_stride;
    }
}

static void spread(const Mat &src, Mat &dst, int T)
{
    // OPTIMIZATION: separable spread. Original does T*T full-image OR passes.
    // Separable does 2 passes: horizontal OR (T wide), then vertical OR (T tall).
    // For T=4: 16 passes -> 2 passes = 8x less work.
    // Valid because OR is associative and commutative.

    int half = T / 2;
    int rows = src.rows;
    int cols = src.cols;
    int step = static_cast<int>(src.step1());

    // Pass 1: horizontal OR
    Mat h_spread = Mat::zeros(src.size(), CV_8U);
    for (int r = 0; r < rows; ++r) {
        const uchar *src_row = src.ptr(r);
        uchar *dst_row = h_spread.ptr(r);

        int c = 0;
#ifdef __AVX2__
        // Process interior where all offsets are in-bounds
        for (; c <= cols - 32 - half; c += 32) {
            __m256i acc = _mm256_loadu_si256((const __m256i*)(src_row + c));
            for (int d = 1; d <= half; ++d) {
                if (c - d >= 0)
                    acc = _mm256_or_si256(acc,
                        _mm256_loadu_si256((const __m256i*)(src_row + c - d)));
                if (c + d < cols - 31)
                    acc = _mm256_or_si256(acc,
                        _mm256_loadu_si256((const __m256i*)(src_row + c + d)));
            }
            _mm256_storeu_si256((__m256i*)(dst_row + c), acc);
        }
#endif
        // Scalar for remaining / all if no AVX2
        for (; c < cols; ++c) {
            uchar val = src_row[c];
            for (int d = 1; d <= half; ++d) {
                if (c - d >= 0) val |= src_row[c - d];
                if (c + d < cols) val |= src_row[c + d];
            }
            dst_row[c] = val;
        }
    }

    // Pass 2: vertical OR
    dst = Mat::zeros(src.size(), CV_8U);
    int h_step = static_cast<int>(h_spread.step1());
    int d_step = static_cast<int>(dst.step1());

    for (int r = 0; r < rows; ++r) {
        uchar *dst_row = dst.ptr(r);

        int c = 0;
#ifdef __AVX2__
        for (; c <= cols - 32; c += 32) {
            __m256i acc = _mm256_loadu_si256(
                (const __m256i*)(h_spread.ptr(r) + c));
            for (int d = 1; d <= half; ++d) {
                if (r - d >= 0)
                    acc = _mm256_or_si256(acc,
                        _mm256_loadu_si256((const __m256i*)(h_spread.ptr(r - d) + c)));
                if (r + d < rows)
                    acc = _mm256_or_si256(acc,
                        _mm256_loadu_si256((const __m256i*)(h_spread.ptr(r + d) + c)));
            }
            _mm256_storeu_si256((__m256i*)(dst_row + c), acc);
        }
#endif
        for (; c < cols; ++c) {
            uchar val = h_spread.ptr(r)[c];
            for (int d = 1; d <= half; ++d) {
                if (r - d >= 0) val |= h_spread.ptr(r - d)[c];
                if (r + d < rows) val |= h_spread.ptr(r + d)[c];
            }
            dst_row[c] = val;
        }
    }
}

static const unsigned char LUT3 = 3;
// 1,2-->0 3-->LUT3
CV_DECL_ALIGNED(16)
static const unsigned char SIMILARITY_LUT[256] = {0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4};

static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
{
    CV_Assert((src.rows * src.cols) % 16 == 0);

    // Allocate response maps
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i)
        response_maps[i].create(src.size(), CV_8U);

    const int total = src.rows * src.cols;
    const uchar *src_data = src.ptr<uchar>();

#ifdef __AVX2__
    // =========================================================================
    // AVX2 FAST PATH: 1 orientation per pass, fused nibble-split + vpshufb.
    // 8 passes, each a tight loop with 2 LUT registers + sequential store.
    // The spread image stays in L2 cache across all 8 passes.
    // =========================================================================
    {
        const __m256i nibble_mask = _mm256_set1_epi8(0x0F);

        for (int ori = 0; ori < 8; ++ori) {
            uchar *map_data = response_maps[ori].ptr<uchar>();
            const uchar *lut_ptr = SIMILARITY_LUT + 32 * ori;

            __m256i lut_lo = _mm256_broadcastsi128_si256(
                _mm_loadu_si128((const __m128i*)lut_ptr));
            __m256i lut_hi = _mm256_broadcastsi128_si256(
                _mm_loadu_si128((const __m128i*)(lut_ptr + 16)));

            int i = 0;
            for (; i <= total - 32; i += 32) {
                __m256i spread = _mm256_loadu_si256((const __m256i*)(src_data + i));
                __m256i lo_nib = _mm256_and_si256(spread, nibble_mask);
                __m256i hi_nib = _mm256_and_si256(
                    _mm256_srli_epi16(spread, 4), nibble_mask);
                __m256i result = _mm256_max_epu8(
                    _mm256_shuffle_epi8(lut_lo, lo_nib),
                    _mm256_shuffle_epi8(lut_hi, hi_nib));
                _mm256_storeu_si256((__m256i*)(map_data + i), result);
            }
            for (; i < total; ++i) {
                uchar sb = src_data[i];
                map_data[i] = std::max(lut_ptr[sb & 0x0F], lut_ptr[(sb >> 4) + 16]);
            }
        }
    }
#elif defined(has_shuff_int8_t) && defined(has_max_int8_t)
    // SSE path (original, correct -- mipp::shuff works on 128-bit)
    {
        // Split nibbles into separate buffers (needed for SSE path)
        Mat lsb4(src.size(), CV_8U);
        Mat msb4(src.size(), CV_8U);
        for (int r = 0; r < src.rows; ++r) {
            const uchar *src_r = src.ptr(r);
            uchar *lsb4_r = lsb4.ptr(r);
            uchar *msb4_r = msb4.ptr(r);
            for (int c = 0; c < src.cols; ++c) {
                lsb4_r[c] = src_r[c] & 15;
                msb4_r[c] = (src_r[c] & 240) >> 4;
            }
        }

        uchar *lsb4_data = lsb4.ptr<uchar>();
        uchar *msb4_data = msb4.ptr<uchar>();

        for (int ori = 0; ori < 8; ++ori) {
            uchar *map_data = response_maps[ori].ptr<uchar>();
            const uchar *lut_low = SIMILARITY_LUT + 32 * ori;
            mipp::Reg<uint8_t> lut_low_v((uint8_t*)lut_low);
            mipp::Reg<uint8_t> lut_high_v((uint8_t*)lut_low + 16);

            for (int i = 0; i < total; i += mipp::N<uint8_t>()) {
                mipp::Reg<uint8_t> low_mask((uint8_t*)lsb4_data + i);
                mipp::Reg<uint8_t> high_mask((uint8_t*)msb4_data + i);
                mipp::Reg<uint8_t> low_res = mipp::shuff(lut_low_v, low_mask);
                mipp::Reg<uint8_t> high_res = mipp::shuff(lut_high_v, high_mask);
                mipp::Reg<uint8_t> result = mipp::max(low_res, high_res);
                result.store((uint8_t*)map_data + i);
            }
        }
    }
#else
    // Scalar fallback
    {
        for (int ori = 0; ori < 8; ++ori) {
            uchar *map_data = response_maps[ori].ptr<uchar>();
            const uchar *lut_low = SIMILARITY_LUT + 32 * ori;
            for (int i = 0; i < total; ++i) {
                uchar sb = src_data[i];
                map_data[i] = std::max(
                    lut_low[sb & 0x0F],
                    lut_low[(sb >> 4) + 16]);
            }
        }
    }
#endif
}

static void linearize(const Mat &response_map, Mat &linearized, int T)
{
    CV_Assert(response_map.rows % T == 0);
    CV_Assert(response_map.cols % T == 0);

    // linearized has T^2 rows, where each row is a linear memory
    int mem_width = response_map.cols / T;
    int mem_height = response_map.rows / T;
    linearized.create(T * T, mem_width * mem_height, CV_8U);

    // Outer two for loops iterate over top-left T^2 starting pixels
    int index = 0;
    for (int r_start = 0; r_start < T; ++r_start)
    {
        for (int c_start = 0; c_start < T; ++c_start)
        {
            uchar *memory = linearized.ptr(index);
            ++index;

            // Inner two loops copy every T-th pixel into the linear memory
            for (int r = r_start; r < response_map.rows; r += T)
            {
                const uchar *response_data = response_map.ptr(r);
                for (int c = c_start; c < response_map.cols; c += T)
                    *memory++ = response_data[c];
            }
        }
    }
}
/****************************************************************************************\
*                                                             Linearized similarities                                                                    *
\****************************************************************************************/

static const unsigned char *accessLinearMemory(const std::vector<Mat> &linear_memories,
                                               const Feature &f, int T, int W)
{
    // Retrieve the TxT grid of linear memories associated with the feature label
    const Mat &memory_grid = linear_memories[f.label];
    CV_DbgAssert(memory_grid.rows == T * T);
    CV_DbgAssert(f.x >= 0);
    CV_DbgAssert(f.y >= 0);
    // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
    int grid_x = f.x % T;
    int grid_y = f.y % T;
    int grid_index = grid_y * T + grid_x;
    CV_DbgAssert(grid_index >= 0);
    CV_DbgAssert(grid_index < memory_grid.rows);
    const unsigned char *memory = memory_grid.ptr(grid_index);
    // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
    // input image width decimated by T.
    int lm_x = f.x / T;
    int lm_y = f.y / T;
    int lm_index = lm_y * W + lm_x;
    CV_DbgAssert(lm_index >= 0);
    CV_DbgAssert(lm_index < memory_grid.cols);
    return memory + lm_index;
}

static void similarity(const std::vector<Mat> &linear_memories, const Template &templ,
                       Mat &dst, Size size, int T)
{
    // we only have one modality, so 8192*2, due to mipp, back to 8192
    CV_Assert(templ.features.size() < 8192);

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    int template_positions = span_y * W + span_x + 1;

    dst = Mat::zeros(H, W, CV_16U);
    short *dst_ptr = dst.ptr<short>();

    // Collect valid feature LM pointers upfront
    std::vector<const uchar*> lm_ptrs;
    lm_ptrs.reserve(templ.features.size());
    for (int i = 0; i < (int)templ.features.size(); ++i) {
        Feature f = templ.features[i];
        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        lm_ptrs.push_back(accessLinearMemory(linear_memories, f, T, W));
    }
    int num_valid = (int)lm_ptrs.size();

    // OPTIMIZATION: Chunked uint8 accumulation.
    // Max features per uint8 accumulation batch. 63 × 4 (max score per feature) = 252 < 255.
    // Accumulate in uint8 at FULL SIMD width (32 bytes on AVX2 = 2x throughput).
    // Widen to int16 only between batches.
    const int BATCH = 63;

    // Pre-allocate acc8 outside the batch loop — avoid repeated alloc/free per batch
#if defined(__AVX2__) || defined(__aarch64__)
    std::vector<uint8_t> acc8(template_positions);
#endif

    for (int batch_start = 0; batch_start < num_valid; batch_start += BATCH) {
        int batch_end = std::min(batch_start + BATCH, num_valid);

#ifdef __AVX2__
        // AVX2 path: accumulate batch in uint8 temp buffer, then widen
        std::memset(acc8.data(), 0, template_positions);

        for (int fi = batch_start; fi < batch_end; ++fi) {
            const uchar *lm_ptr = lm_ptrs[fi];
            int j = 0;
            for (; j <= template_positions - 32; j += 32) {
                __m256i s = _mm256_loadu_si256((const __m256i*)(lm_ptr + j));
                __m256i d = _mm256_loadu_si256((const __m256i*)(acc8.data() + j));
                _mm256_storeu_si256((__m256i*)(acc8.data() + j), _mm256_add_epi8(d, s));
            }
            for (; j < template_positions; ++j)
                acc8[j] += lm_ptr[j];
        }

        // Widen uint8 -> int16 and add to dst
        {
            int j = 0;
            for (; j <= template_positions - 16; j += 16) {
                __m128i s8 = _mm_loadu_si128((const __m128i*)(acc8.data() + j));
                __m256i s16 = _mm256_cvtepu8_epi16(s8);
                __m256i d16 = _mm256_loadu_si256((const __m256i*)(dst_ptr + j));
                _mm256_storeu_si256((__m256i*)(dst_ptr + j), _mm256_add_epi16(d16, s16));
            }
            for (; j < template_positions; ++j)
                dst_ptr[j] += (short)acc8[j];
        }
#elif defined(__aarch64__)
        // NEON mirror of the AVX2 uint8 chunked-accumulation: accumulate the
        // batch (<=63 features, max 252 < 255) in uint8 at 16-wide, then widen
        // uint8 -> int16 once per batch. Bit-identical to the MIPP #else, but
        // 16 lanes of uint8 vs 8 lanes of int16 + a per-feature widen.
        std::memset(acc8.data(), 0, template_positions);
        for (int fi = batch_start; fi < batch_end; ++fi) {
            const uchar *lm_ptr = lm_ptrs[fi];
            int j = 0;
            for (; j <= template_positions - 16; j += 16)
                vst1q_u8(acc8.data() + j,
                         vaddq_u8(vld1q_u8(acc8.data() + j), vld1q_u8(lm_ptr + j)));
            for (; j < template_positions; ++j)
                acc8[j] += lm_ptr[j];
        }
        {
            int j = 0;
            for (; j <= template_positions - 16; j += 16) {
                uint8x16_t a = vld1q_u8(acc8.data() + j);
                int16x8_t d0 = vaddq_s16(vld1q_s16(dst_ptr + j),
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a))));
                int16x8_t d1 = vaddq_s16(vld1q_s16(dst_ptr + j + 8),
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a))));
                vst1q_s16(dst_ptr + j, d0);
                vst1q_s16(dst_ptr + j + 8, d1);
            }
            for (; j < template_positions; ++j)
                dst_ptr[j] += (short)acc8[j];
        }
#else
        // Non-AVX2: original MIPP path (int16 widening per feature)
        mipp::Reg<uint8_t> zero_v(uint8_t(0));
        for (int fi = batch_start; fi < batch_end; ++fi) {
            const uchar *lm_ptr = lm_ptrs[fi];
            int j = 0;
            for (; j <= template_positions - mipp::N<int16_t>() * 2; j += mipp::N<int16_t>()) {
                mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + j);
                mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);
                mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + j);
                (src16_v + dst_v).store((int16_t*)dst_ptr + j);
            }
            for (; j < template_positions; ++j)
                dst_ptr[j] += short(lm_ptr[j]);
        }
#endif
    }
}

static void similarityLocal(const std::vector<Mat> &linear_memories, const Template &templ,
                            Mat &dst, Size size, int T, Point center)
{
    CV_Assert(templ.features.size() < 8192);

    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_16U);

    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;
    mipp::Reg<uint8_t> zero_v = uint8_t(0);

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);
        {
            short *dst_ptr = dst.ptr<short>();

            if(mipp::N<uint8_t>() > 32){ //512 bits SIMD
                for (int row = 0; row < 16; row += mipp::N<int16_t>()/16){
                    mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + row*16);

                    // load lm_ptr, 16 bytes once, for half
                    uint8_t local_v[mipp::N<uint8_t>()] = {0};
                    for(int slice=0; slice<mipp::N<uint8_t>()/16/2; slice++){
                        std::copy_n(lm_ptr, 16, &local_v[16*slice]);
                        lm_ptr += W;
                    }
                    mipp::Reg<uint8_t> src8_v(local_v);
                    // uchar to short, once for N bytes
                    mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

                    mipp::Reg<int16_t> res_v = src16_v + dst_v;
                    res_v.store((int16_t*)dst_ptr);

                    dst_ptr += mipp::N<int16_t>();
                }
            }else{ // 256 128 or no SIMD
                for (int row = 0; row < 16; ++row){
                    for(int col=0; col<16; col+=mipp::N<int16_t>()){
                        mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + col);

                        // uchar to short, once for N bytes
                        mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

                        mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + col);
                        mipp::Reg<int16_t> res_v = src16_v + dst_v;
                        res_v.store((int16_t*)dst_ptr + col);
                    }
                    dst_ptr += 16;
                    lm_ptr += W;
                }
            }
        }
    }
}

static void similarity_64(const std::vector<Mat> &linear_memories, const Template &templ,
                          Mat &dst, Size size, int T)
{
    // 63 features or less is a special case because the max similarity per-feature is 4.
    // 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
    // about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
    // general function would use _mm_add_epi16.
    CV_Assert(templ.features.size() < 64);
    /// @todo Handle more than 255/MAX_RESPONSE features!!

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    // Compute number of contiguous (in memory) pixels to check when sliding feature over
    // image. This allows template to wrap around left/right border incorrectly, so any
    // wrapped template matches must be filtered out!
    int template_positions = span_y * W + span_x + 1; // why add 1?
    //int template_positions = (span_y - 1) * W + span_x; // More correct?

    /// @todo In old code, dst is buffer of size m_U. Could make it something like
    /// (span_x)x(span_y) instead?
    dst = Mat::zeros(H, W, CV_8U);
    uchar *dst_ptr = dst.ptr<uchar>();

    // Compute the similarity measure for this template by accumulating the contribution of
    // each feature
    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        // Add the linear memory at the appropriate offset computed from the location of
        // the feature in the template
        Feature f = templ.features[i];
        // Discard feature if out of bounds
        /// @todo Shouldn't actually see x or y < 0 here?
        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        int j = 0;

        for(; j <= template_positions -mipp::N<uint8_t>(); j+=mipp::N<uint8_t>()){
            mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + j);
            mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + j);

            mipp::Reg<uint8_t> res_v = src_v + dst_v;
            res_v.store((uint8_t*)dst_ptr + j);
        }

        for(; j<template_positions; j++)
            dst_ptr[j] += lm_ptr[j];
    }
}

static void similarityLocal_64(const std::vector<Mat> &linear_memories, const Template &templ,
                               Mat &dst, Size size, int T, Point center)
{
    // Similar to whole-image similarity() above. This version takes a position 'center'
    // and computes the energy in the 16x16 patch centered on it.
    CV_Assert(templ.features.size() < 64);

    // Compute the similarity map in a 16x16 patch around center
    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_8U);

    // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
    // center to get the top-left corner of the 16x16 patch.
    // NOTE: We make the offsets multiples of T to agree with results of the original code.
    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        {
            uchar *dst_ptr = dst.ptr<uchar>();

            if(mipp::N<uint8_t>() > 16){ // 256 or 512 bits SIMD
                for (int row = 0; row < 16; row += mipp::N<uint8_t>()/16){
                    mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr);

                    // load lm_ptr, 16 bytes once
                    uint8_t local_v[mipp::N<uint8_t>()];
                    for(int slice=0; slice<mipp::N<uint8_t>()/16; slice++){
                        std::copy_n(lm_ptr, 16, &local_v[16*slice]);
                        lm_ptr += W;
                    }
                    mipp::Reg<uint8_t> src_v(local_v);

                    mipp::Reg<uint8_t> res_v = src_v + dst_v;
                    res_v.store((uint8_t*)dst_ptr);

                    dst_ptr += mipp::N<uint8_t>();
                }
            }else{ // 128 or no SIMD
                for (int row = 0; row < 16; ++row){
                    for(int col=0; col<16; col+=mipp::N<uint8_t>()){
                        mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + col);
                        mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + col);
                        mipp::Reg<uint8_t> res_v = src_v + dst_v;
                        res_v.store((uint8_t*)dst_ptr + col);
                    }
                    dst_ptr += 16;
                    lm_ptr += W;
                }
            }
        }
    }
}

/****************************************************************************************\
*                                                             High-level Detector API                                                                    *
\****************************************************************************************/

Detector::Detector()
{
    this->modality = makePtr<ColorGradient>();
    pyramid_levels = 2;
    T_at_level.push_back(4);
    T_at_level.push_back(8);
}

Detector::Detector(std::vector<int> T)
{
    this->modality = makePtr<ColorGradient>();
    pyramid_levels = T.size();
    T_at_level = T;
}

Detector::Detector(int num_features, std::vector<int> T, float weak_thresh, float strong_threash)
{
    this->modality = makePtr<ColorGradient>(weak_thresh, num_features, strong_threash);
    pyramid_levels = T.size();
    T_at_level = T;
}

std::vector<Match> Detector::match(Mat source, float threshold,
                                   const std::vector<std::string> &class_ids, const Mat mask) const
{
    Timer timer;
    std::vector<Match> matches;

    // Initialize each ColorGradient with our sources
    std::vector<Ptr<ColorGradientPyramid>> quantizers;
    CV_Assert(mask.empty() || mask.size() == source.size());
    quantizers.push_back(modality->process(source, mask, /*match_only=*/true));

    // pyramid level -> ColorGradient -> quantization
    LinearMemoryPyramid lm_pyramid(pyramid_levels,
                                   std::vector<LinearMemories>(1, LinearMemories(8)));

    // For each pyramid level, precompute linear memories for each ColorGradient
    std::vector<Size> sizes;
    for (int l = 0; l < pyramid_levels; ++l)
    {
        int T = T_at_level[l];
        std::vector<LinearMemories> &lm_level = lm_pyramid[l];

        if (l > 0)
        {
            for (int i = 0; i < (int)quantizers.size(); ++i)
                quantizers[i]->pyrDown();
        }

        Mat quantized;
        for (int i = 0; i < (int)quantizers.size(); ++i)
        {
            quantizers[i]->quantize(quantized);

            // Profile: fused spread+LUT+linearize
            {
                using PClock = std::chrono::high_resolution_clock;
                auto fused_t0 = PClock::now();

            // FULLY FUSED: spread + computeResponseMaps + linearize in one pass.
#if 0 // WIP experiments removed — fused angle+spread and cell-level spread were both slower
            // FUSED ANGLE+SPREAD: compute angle bitmask rows into ring buffer,
            // then spread+LUT+decimate from ring buffer. Eliminates 20MB angle image.
            // Sequential per-row (spread depends on adjacent angle rows) but
            // column processing within each row uses AVX2.
            if (skip_voting && match_only) {
                LinearMemories &memories = lm_level[i];
                CV_Assert(quantized.rows % T == 0);
                CV_Assert(quantized.cols % T == 0);
                int src_cols = quantized.cols;
                int src_rows = quantized.rows;
                int mem_w = src_cols / T;
                int mem_h = src_rows / T;
                int half = T / 2;

                for (int ori = 0; ori < 8; ++ori)
                    memories[ori].create(T * T, mem_w * mem_h, CV_8U);

                static const uchar discount_table[9] = {0, 4, 4, 4, 3, 2, 1, 0, 0};
                static const int pop4[16] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
                static const int TAN_B_local[4] = {1989, 6682, 14966, 50273};

                // Ring buffer for angle bitmask rows (T+1 rows keeps enough for spread)
                int ring_size = T + 1;
                std::vector<std::vector<uchar>> angle_ring(ring_size, std::vector<uchar>(src_cols, 0));

                // Pre-compute first (half) rows of angle bitmask
                auto compute_angle_row = [&](int r, uchar* out) {
                    if (r < 1 || r >= src_rows - 1) {
                        std::memset(out, 0, src_cols);
                        return;
                    }
                    std::memset(out, 0, src_cols);
                    const uchar *row_prev = smoothed.ptr<uchar>(r-1);
                    const uchar *row_curr = smoothed.ptr<uchar>(r);
                    const uchar *row_next = smoothed.ptr<uchar>(r+1);
#if SBM_GRADIENT_KERNEL == SBM_GRADIENT_KERNEL_CENTRAL_DIFF
                    int et = std::max(1, (int)threshold / 4);
#else
                    int et = (int)threshold;
#endif
                    int et_sq = et * et;
                    for (int c = 1; c < src_cols - 1; ++c) {
#if SBM_GRADIENT_KERNEL == SBM_GRADIENT_KERNEL_CENTRAL_DIFF
                        int gx = row_curr[c+1] - row_curr[c-1];
                        int gy = row_next[c] - row_prev[c];
#else
                        int gx = (row_prev[c+1]-row_prev[c-1]) + 2*(row_curr[c+1]-row_curr[c-1]) + (row_next[c+1]-row_next[c-1]);
                        int gy = (row_next[c-1]+2*row_next[c]+row_next[c+1]) - (row_prev[c-1]+2*row_prev[c]+row_prev[c+1]);
#endif
                        if (gx*gx + gy*gy <= et_sq) continue;
                        int ugx = gx, ugy = gy;
                        if (ugy < 0) { ugx = -ugx; ugy = -ugy; }
                        if (ugy == 0 && ugx < 0) ugx = -ugx;
                        int bin;
                        if (ugx >= 0) {
                            int ty = ugy * 10000;
                            if      (ty < ugx * TAN_B_local[0]) bin = 0;
                            else if (ty < ugx * TAN_B_local[1]) bin = 1;
                            else if (ty < ugx * TAN_B_local[2]) bin = 2;
                            else if (ty < ugx * TAN_B_local[3]) bin = 3;
                            else bin = 4;
                        } else {
                            int agx = -ugx, ty = ugy * 10000;
                            if      (ty < agx * TAN_B_local[0]) bin = 0;
                            else if (ty < agx * TAN_B_local[1]) bin = 7;
                            else if (ty < agx * TAN_B_local[2]) bin = 6;
                            else if (ty < agx * TAN_B_local[3]) bin = 5;
                            else bin = 4;
                        }
                        out[c] = (uchar)(1 << bin);
                    }
                };

                // Pre-fill ring buffer
                for (int r = 0; r < ring_size && r < src_rows; ++r)
                    compute_angle_row(r, angle_ring[r % ring_size].data());

                // Row-level spread buffers
                std::vector<uchar> v_or_buf(src_cols), hv_spread_buf(src_cols);
                std::vector<uchar> discount_buf(src_cols), response_buf(src_cols);

                // Process each row: spread from ring buffer → LUT → linear memories
                for (int r = 0; r < src_rows; ++r) {
                    // Advance ring buffer: compute next needed angle row
                    int next_r = r + half + 1;
                    if (next_r < src_rows)
                        compute_angle_row(next_r, angle_ring[next_r % ring_size].data());

                    // Vertical OR from ring buffer
                    int y0 = std::max(0, r - half);
                    int y1 = std::min(src_rows - 1, r + half);
                    std::memcpy(v_or_buf.data(), angle_ring[y0 % ring_size].data(), src_cols);
                    for (int yy = y0 + 1; yy <= y1; ++yy) {
                        const uchar *row = angle_ring[yy % ring_size].data();
                        for (int x = 0; x < src_cols; ++x)
                            v_or_buf[x] |= row[x];
                    }

                    // Horizontal OR
                    {
                        const uchar *src_p = v_or_buf.data();
                        uchar *dst = hv_spread_buf.data();
                        for (int x = 0; x < src_cols; ++x) {
                            uchar acc = 0;
                            int x0 = std::max(0, x - half);
                            int x1 = std::min(src_cols - 1, x + half);
                            for (int xx = x0; xx <= x1; ++xx)
                                acc |= src_p[xx];
                            dst[x] = acc;
                        }
                    }

                    // Discount
                    for (int x = 0; x < src_cols; ++x) {
                        uchar sb = hv_spread_buf[x];
                        int nbits = pop4[sb & 0x0F] + pop4[(sb >> 4) & 0x0F];
                        discount_buf[x] = (nbits < 9) ? discount_table[nbits] : 0;
                    }

                    // LUT + decimate
                    int grid_row = r % T;
                    int dec_r = r / T;
                    for (int ori = 0; ori < 8; ++ori) {
                        const uchar *lut_ptr = SIMILARITY_LUT + 32 * ori;
                        for (int c_start = 0; c_start < T; ++c_start) {
                            int grid_index = grid_row * T + c_start;
                            uchar *mem_ptr = memories[ori].ptr(grid_index) + dec_r * mem_w;
                            for (int c = c_start; c < src_cols; c += T) {
                                uchar sb = hv_spread_buf[c];
                                uchar raw = std::max(lut_ptr[sb & 0x0F], lut_ptr[(sb >> 4) + 16]);
                                *mem_ptr++ = (uchar)(raw * discount_buf[c] / 4);
                            }
                        }
                    }
                }

                if (g_profile.enabled)
                    g_profile.fused_spread_lut_ms += std::chrono::duration<double, std::milli>(
                        PClock::now() - fused_t0).count();
            } else
            // CELL-LEVEL: for each T×T cell, load (T+2*half)×(T+2*half) block,
            // compute spread for each sub-position, apply LUT, write to linear memories.
            {
                LinearMemories &memories = lm_level[i];
                CV_Assert(quantized.rows % T == 0);
                CV_Assert(quantized.cols % T == 0);
                int src_cols = quantized.cols;
                int src_rows = quantized.rows;
                int mem_w = src_cols / T;
                int mem_h = src_rows / T;
                int half = T / 2;

                for (int ori = 0; ori < 8; ++ori)
                    memories[ori].create(T * T, mem_w * mem_h, CV_8U);

                static const uchar discount_table[9] = {0, 4, 4, 4, 3, 2, 1, 0, 0};
                static const int pop4[16] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};

                // Process cells in parallel
                int n_cells_y = mem_h, n_cells_x = mem_w;
                #pragma omp parallel for schedule(dynamic, 4) collapse(2)
                for (int cy = 0; cy < n_cells_y; ++cy) {
                    for (int cx = 0; cx < n_cells_x; ++cx) {
                        // Cell origin in source image
                        int r0 = cy * T, c0 = cx * T;

                        // For each sub-position within the cell
                        for (int gr = 0; gr < T; ++gr) {
                            for (int gc = 0; gc < T; ++gc) {
                                int pr = r0 + gr;  // pixel row
                                int pc = c0 + gc;  // pixel col

                                // Compute spread: OR of angle bytes in [pr-half, pr+half] × [pc-half, pc+half]
                                int y0 = std::max(0, pr - half);
                                int y1 = std::min(src_rows - 1, pr + half);
                                int x0 = std::max(0, pc - half);
                                int x1 = std::min(src_cols - 1, pc + half);

                                uchar spread = 0;
                                for (int yy = y0; yy <= y1; ++yy) {
                                    const uchar *row = quantized.ptr<uchar>(yy);
                                    for (int xx = x0; xx <= x1; ++xx)
                                        spread |= row[xx];
                                }

                                // Popcount discount
                                int nbits = pop4[spread & 0x0F] + pop4[(spread >> 4) & 0x0F];
                                uchar disc = (nbits < 9) ? discount_table[nbits] : 0;

                                // Apply LUT for each orientation and write to linear memory
                                int grid_index = gr * T + gc;
                                int mem_offset = cy * mem_w + cx;
                                for (int ori = 0; ori < 8; ++ori) {
                                    const uchar *lut_ptr = SIMILARITY_LUT + 32 * ori;
                                    uchar raw = std::max(
                                        lut_ptr[spread & 0x0F],
                                        lut_ptr[(spread >> 4) + 16]);
                                    memories[ori].ptr(grid_index)[mem_offset] = (uchar)(raw * disc / 4);
                                }
                            }
                        }
                    }
                }
            }
#endif // disabled WIP experiments
            // FULLY FUSED: spread + computeResponseMaps + linearize in one pass.
            {
                LinearMemories &memories = lm_level[i];
                CV_Assert(quantized.rows % T == 0);
                CV_Assert(quantized.cols % T == 0);
                int src_cols = quantized.cols;
                int src_rows = quantized.rows;
                int mem_w = src_cols / T;
                int mem_h = src_rows / T;
                int half = T / 2;

                for (int ori = 0; ori < 8; ++ori)
                    memories[ori].create(T * T, mem_w * mem_h, CV_8U);

                // Popcount discount: score reduction for ambiguous spread bytes.
                // Index = popcount of spread byte.
                // Spreading naturally sets 2-3 bits on clean edges, so we only
                // penalize when many bits are set (ambiguous/noisy regions).
                // popcount 0->0, 1-3->4(full), 4->3, 5->2, 6->1, 7-8->0
                static const uchar discount_table[9] = {0, 4, 4, 4, 3, 2, 1, 0, 0};
                static const int pop4[16] = {
                    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};

                // Parallelized row loop: each thread gets private row buffers.
                // No write conflicts because each (grid_row, dec_r) pair is unique per row r.
                #pragma omp parallel for schedule(dynamic, 16)
                for (int r = 0; r < src_rows; ++r) {
                // Per-thread row buffers (allocated on first use via thread-local)
                thread_local std::vector<uchar> v_or_buf, hv_spread_buf, discount_buf;
                thread_local std::vector<uchar> response_buf;
                if ((int)v_or_buf.size() != src_cols) {
                    v_or_buf.resize(src_cols, 0);
                    hv_spread_buf.resize(src_cols, 0);
                    discount_buf.resize(src_cols, 0);
                    response_buf.resize(src_cols, 0);
                }
                    // --- Step 1: Vertical OR of T rows centered at r ---
                    int y0 = std::max(0, r - half);
                    int y1 = std::min(src_rows - 1, r + half);

                    // Start with first row
                    {
                        const uchar *first = quantized.ptr(y0);
                        int x = 0;
#ifdef __AVX2__
                        for (; x <= src_cols - 32; x += 32) {
                            _mm256_storeu_si256((__m256i*)(v_or_buf.data() + x),
                                _mm256_loadu_si256((const __m256i*)(first + x)));
                        }
#elif defined(__aarch64__)
                        for (; x <= src_cols - 16; x += 16)
                            vst1q_u8(v_or_buf.data() + x, vld1q_u8(first + x));
#endif
                        for (; x < src_cols; ++x) v_or_buf[x] = first[x];
                    }
                    // OR remaining rows
                    for (int yy = y0 + 1; yy <= y1; ++yy) {
                        const uchar *row = quantized.ptr(yy);
                        int x = 0;
#ifdef __AVX2__
                        for (; x <= src_cols - 32; x += 32) {
                            __m256i acc = _mm256_loadu_si256((const __m256i*)(v_or_buf.data() + x));
                            __m256i v = _mm256_loadu_si256((const __m256i*)(row + x));
                            _mm256_storeu_si256((__m256i*)(v_or_buf.data() + x),
                                _mm256_or_si256(acc, v));
                        }
#elif defined(__aarch64__)
                        for (; x <= src_cols - 16; x += 16)
                            vst1q_u8(v_or_buf.data() + x,
                                vorrq_u8(vld1q_u8(v_or_buf.data() + x), vld1q_u8(row + x)));
#endif
                        for (; x < src_cols; ++x) v_or_buf[x] |= row[x];
                    }

                    // --- Step 2: Horizontal OR spread ---
                    {
                        const uchar *src = v_or_buf.data();
                        uchar *dst = hv_spread_buf.data();

                        int x = 0;
#ifdef __AVX2__
                        // AVX2 interior where all offsets are in bounds
                        for (; x <= src_cols - 32 - half; x += 32) {
                            __m256i acc = _mm256_loadu_si256((const __m256i*)(src + x));
                            for (int d = 1; d <= half; ++d) {
                                if (x - d >= 0)
                                    acc = _mm256_or_si256(acc,
                                        _mm256_loadu_si256((const __m256i*)(src + x - d)));
                                if (x + d <= src_cols - 32)
                                    acc = _mm256_or_si256(acc,
                                        _mm256_loadu_si256((const __m256i*)(src + x + d)));
                            }
                            _mm256_storeu_si256((__m256i*)(dst + x), acc);
                        }
#elif defined(__aarch64__)
                        // NEON interior where all offsets are in bounds (16-wide)
                        for (; x <= src_cols - 16 - half; x += 16) {
                            uint8x16_t acc = vld1q_u8(src + x);
                            for (int d = 1; d <= half; ++d) {
                                if (x - d >= 0)
                                    acc = vorrq_u8(acc, vld1q_u8(src + x - d));
                                if (x + d <= src_cols - 16)
                                    acc = vorrq_u8(acc, vld1q_u8(src + x + d));
                            }
                            vst1q_u8(dst + x, acc);
                        }
#endif
                        // Scalar for remaining / borders
                        for (; x < src_cols; ++x) {
                            uchar val = src[x];
                            for (int d = 1; d <= half; ++d) {
                                if (x - d >= 0) val |= src[x - d];
                                if (x + d < src_cols) val |= src[x + d];
                            }
                            dst[x] = val;
                        }
                        // Fix up any AVX2-skipped left border
                        for (x = 0; x < std::min(half, src_cols); ++x) {
                            uchar val = src[x];
                            for (int d = 1; d <= half; ++d) {
                                if (x - d >= 0) val |= src[x - d];
                                if (x + d < src_cols) val |= src[x + d];
                            }
                            dst[x] = val;
                        }
                    }

                    // --- Step 2b: Compute popcount discount for entire row ---
                    // This is orientation-independent, so compute once per row.
                    {
                        const uchar *sp = hv_spread_buf.data();
                        uchar *disc = discount_buf.data();
                        int x = 0;
#ifdef __AVX2__
                        alignas(16) static const uchar pop4_tbl[16] = {
                            0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
                        __m256i pop4_lut = _mm256_broadcastsi128_si256(
                            _mm_load_si128((const __m128i*)pop4_tbl));
                        alignas(16) static const uchar disc_tbl[16] = {
                            0, 4, 4, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                        __m256i disc_lut = _mm256_broadcastsi128_si256(
                            _mm_load_si128((const __m128i*)disc_tbl));
                        __m256i nmask = _mm256_set1_epi8(0x0F);

                        for (; x <= src_cols - 32; x += 32) {
                            __m256i sb = _mm256_loadu_si256((const __m256i*)(sp + x));
                            __m256i lo = _mm256_and_si256(sb, nmask);
                            __m256i hi = _mm256_and_si256(
                                _mm256_srli_epi16(sb, 4), nmask);
                            __m256i pcnt = _mm256_add_epi8(
                                _mm256_shuffle_epi8(pop4_lut, lo),
                                _mm256_shuffle_epi8(pop4_lut, hi));
                            __m256i dv = _mm256_shuffle_epi8(disc_lut, pcnt);
                            _mm256_storeu_si256((__m256i*)(disc + x), dv);
                        }
#elif defined(__aarch64__)
                        // NEON: hardware per-byte popcount (vcntq_u8) → discount LUT.
                        // popcount(byte) == pop4[lo] + pop4[hi], always < 9, so the
                        // 16-entry table indexed by the count reproduces the scalar
                        // (nbits < 9) ? discount_table[nbits] : 0 exactly.
                        alignas(16) static const uchar disc_tbl[16] = {
                            0, 4, 4, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                        uint8x16_t disc_lut = vld1q_u8(disc_tbl);
                        for (; x <= src_cols - 16; x += 16) {
                            uint8x16_t sb = vld1q_u8(sp + x);
                            uint8x16_t pcnt = vcntq_u8(sb);
                            vst1q_u8(disc + x, vqtbl1q_u8(disc_lut, pcnt));
                        }
#endif
                        for (; x < src_cols; ++x) {
                            uchar sb = sp[x];
                            int nbits = pop4[sb & 0x0F] + pop4[(sb >> 4) & 0x0F];
                            disc[x] = (nbits < 9) ? discount_table[nbits] : 0;
                        }
                    }

                    // --- Step 3: LUT + discount + write to linear memories ---
                    int grid_row = r % T;
                    int dec_r = r / T;
                    const uchar *spread_row = hv_spread_buf.data();
                    const uchar *disc_row = discount_buf.data();

                    for (int ori = 0; ori < 8; ++ori) {
                        const uchar *lut_ptr = SIMILARITY_LUT + 32 * ori;

#ifdef __AVX2__
                        // AVX2: apply LUT + popcount discount to full spread row,
                        // then strided-decimate to linear memories.
                        __m256i lut_lo_v = _mm256_broadcastsi128_si256(
                            _mm_loadu_si128((const __m128i*)lut_ptr));
                        __m256i lut_hi_v = _mm256_broadcastsi128_si256(
                            _mm_loadu_si128((const __m128i*)(lut_ptr + 16)));
                        __m256i nibble_mask = _mm256_set1_epi8(0x0F);

                        uchar *resp_row = response_buf.data();
                        int x = 0;
                        for (; x <= src_cols - 32; x += 32) {
                            __m256i sb = _mm256_loadu_si256((const __m256i*)(spread_row + x));
                            __m256i lo_nib = _mm256_and_si256(sb, nibble_mask);
                            __m256i hi_nib = _mm256_and_si256(
                                _mm256_srli_epi16(sb, 4), nibble_mask);
                            __m256i raw_score = _mm256_max_epu8(
                                _mm256_shuffle_epi8(lut_lo_v, lo_nib),
                                _mm256_shuffle_epi8(lut_hi_v, hi_nib));

                            // Apply popcount discount: result = raw * disc / 4
                            __m256i dv = _mm256_loadu_si256((const __m256i*)(disc_row + x));
                            // Unpack to 16-bit, multiply, shift right 2, pack
                            __m256i zero = _mm256_setzero_si256();
                            __m256i raw_lo16 = _mm256_unpacklo_epi8(raw_score, zero);
                            __m256i raw_hi16 = _mm256_unpackhi_epi8(raw_score, zero);
                            __m256i dsc_lo16 = _mm256_unpacklo_epi8(dv, zero);
                            __m256i dsc_hi16 = _mm256_unpackhi_epi8(dv, zero);
                            __m256i prod_lo = _mm256_srli_epi16(
                                _mm256_mullo_epi16(raw_lo16, dsc_lo16), 2);
                            __m256i prod_hi = _mm256_srli_epi16(
                                _mm256_mullo_epi16(raw_hi16, dsc_hi16), 2);
                            __m256i result = _mm256_packus_epi16(prod_lo, prod_hi);

                            _mm256_storeu_si256((__m256i*)(resp_row + x), result);
                        }
                        for (; x < src_cols; ++x) {
                            uchar sb = spread_row[x];
                            uchar raw = std::max(
                                lut_ptr[sb & 0x0F],
                                lut_ptr[(sb >> 4) + 16]);
                            resp_row[x] = (uchar)(raw * disc_row[x] / 4);
                        }

                        // Strided decimate from response row to linear memories.
                        // SSE paths for T=2 and T=4 extract every Nth byte via shuffles.
                        if (T == 2) {
                            // T=2: pick every 2nd byte. Each 16-byte load → 8 output bytes.
                            // Two offsets (even/odd), 2 loads → 16 outputs.
                            alignas(16) static const uint8_t shuf_t2[2][16] = {
                                {0,2,4,6,8,10,12,14, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
                                {1,3,5,7,9,11,13,15, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
                            };
                            for (int c_start = 0; c_start < 2; ++c_start) {
                                int grid_index = grid_row * 2 + c_start;
                                uchar *mem_ptr = memories[ori].ptr(grid_index) + dec_r * mem_w;
                                __m128i mask = _mm_load_si128((const __m128i*)shuf_t2[c_start]);
                                int c = 0;
                                for (; c + 32 <= src_cols; c += 32) {
                                    const uchar *p = resp_row + c;
                                    __m128i v0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p)),     mask);
                                    __m128i v1 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p + 16)), mask);
                                    __m128i out = _mm_unpacklo_epi64(v0, v1);
                                    _mm_storeu_si128((__m128i*)mem_ptr, out);
                                    mem_ptr += 16;
                                }
                                // Scalar tail
                                for (int cc = c + c_start; cc < src_cols; cc += 2) {
                                    *mem_ptr++ = resp_row[cc];
                                }
                            }
                        } else if (T == 8) {
                            // T=8: per-offset SSE pass, same pattern as T=4.
                            // Each 32-byte AVX2 load → 4 output bytes (every 8th byte).
                            // 4 loads → 16 output bytes per offset.
                            // Shuffle picks bytes {0,8,16,24} from 32-byte AVX2 lane —
                            // but pshufb works per 16-byte lane, so use 16-byte SSE:
                            // each 16-byte load → 2 bytes, 8 loads → 16 bytes.
                            alignas(16) static const uint8_t shuf_t8[16] = {
                                0, 8, 0x80,0x80,0x80,0x80,0x80,0x80,
                                0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80
                            };
                            __m128i mask = _mm_load_si128((const __m128i*)shuf_t8);
                            for (int c_start = 0; c_start < 8; ++c_start) {
                                int grid_index = grid_row * 8 + c_start;
                                uchar *mem_ptr = memories[ori].ptr(grid_index) + dec_r * mem_w;
                                int c = c_start;
                                for (; c + 128 <= src_cols; c += 128) {
                                    const uchar *p = resp_row + c;
                                    __m128i v0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p)),      mask);
                                    __m128i v1 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p + 16)),  mask);
                                    __m128i v2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p + 32)),  mask);
                                    __m128i v3 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p + 48)),  mask);
                                    __m128i v4 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p + 64)),  mask);
                                    __m128i v5 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p + 80)),  mask);
                                    __m128i v6 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p + 96)),  mask);
                                    __m128i v7 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p + 112)), mask);
                                    __m128i v01 = _mm_unpacklo_epi16(v0, v1);
                                    __m128i v23 = _mm_unpacklo_epi16(v2, v3);
                                    __m128i v45 = _mm_unpacklo_epi16(v4, v5);
                                    __m128i v67 = _mm_unpacklo_epi16(v6, v7);
                                    __m128i v0123 = _mm_unpacklo_epi32(v01, v23);
                                    __m128i v4567 = _mm_unpacklo_epi32(v45, v67);
                                    __m128i out = _mm_unpacklo_epi64(v0123, v4567);
                                    _mm_storeu_si128((__m128i*)mem_ptr, out);
                                    mem_ptr += 16;
                                }
                                for (int cc = c; cc < src_cols; cc += 8) {
                                    *mem_ptr++ = resp_row[cc];
                                }
                            }
                        } else if (T == 4) {
                            // T=4: pick every 4th byte. Each 16-byte load → 4 output bytes.
                            // Four offsets, 4 loads → 16 outputs.
                            alignas(16) static const uint8_t shuf_t4[4][16] = {
                                {0,4,8,12, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80},
                                {1,5,9,13, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80},
                                {2,6,10,14, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80},
                                {3,7,11,15, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80},
                            };
                            for (int c_start = 0; c_start < 4; ++c_start) {
                                int grid_index = grid_row * 4 + c_start;
                                uchar *mem_ptr = memories[ori].ptr(grid_index) + dec_r * mem_w;
                                __m128i mask = _mm_load_si128((const __m128i*)shuf_t4[c_start]);
                                int c = 0;
                                for (; c + 64 <= src_cols; c += 64) {
                                    const uchar *p = resp_row + c;
                                    __m128i v0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p)),      mask);
                                    __m128i v1 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p + 16)),  mask);
                                    __m128i v2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p + 32)),  mask);
                                    __m128i v3 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p + 48)),  mask);
                                    __m128i v01 = _mm_unpacklo_epi32(v0, v1);
                                    __m128i v23 = _mm_unpacklo_epi32(v2, v3);
                                    __m128i out = _mm_unpacklo_epi64(v01, v23);
                                    _mm_storeu_si128((__m128i*)mem_ptr, out);
                                    mem_ptr += 16;
                                }
                                // Scalar tail
                                for (int cc = c + c_start; cc < src_cols; cc += 4) {
                                    *mem_ptr++ = resp_row[cc];
                                }
                            }
                        } else {
                            // General T: scalar strided copy
                            for (int c_start = 0; c_start < T; ++c_start) {
                                int grid_index = grid_row * T + c_start;
                                uchar *mem_ptr = memories[ori].ptr(grid_index) + dec_r * mem_w;
                                for (int c = c_start; c < src_cols; c += T) {
                                    *mem_ptr++ = resp_row[c];
                                }
                            }
                        }
#elif defined(__aarch64__)
                        // NEON mirror of the AVX2 path: LUT + popcount discount over
                        // the full spread row into response_buf, then strided decimate
                        // into the linear memories. Bit-identical to the scalar #else.
                        uint8x16_t lut_lo_v = vld1q_u8(lut_ptr);
                        uint8x16_t lut_hi_v = vld1q_u8(lut_ptr + 16);
                        uint8x16_t nibble_mask = vdupq_n_u8(0x0F);

                        uchar *resp_row = response_buf.data();
                        int x = 0;
                        for (; x <= src_cols - 16; x += 16) {
                            uint8x16_t sb = vld1q_u8(spread_row + x);
                            uint8x16_t lo_nib = vandq_u8(sb, nibble_mask);
                            uint8x16_t hi_nib = vandq_u8(vshrq_n_u8(sb, 4), nibble_mask);
                            uint8x16_t raw = vmaxq_u8(vqtbl1q_u8(lut_lo_v, lo_nib),
                                                      vqtbl1q_u8(lut_hi_v, hi_nib));
                            // result = raw * disc / 4, widening to u16 to avoid overflow.
                            uint8x16_t dv = vld1q_u8(disc_row + x);
                            uint16x8_t prod_lo = vshrq_n_u16(
                                vmull_u8(vget_low_u8(raw),  vget_low_u8(dv)),  2);
                            uint16x8_t prod_hi = vshrq_n_u16(
                                vmull_u8(vget_high_u8(raw), vget_high_u8(dv)), 2);
                            vst1q_u8(resp_row + x,
                                     vcombine_u8(vqmovn_u16(prod_lo), vqmovn_u16(prod_hi)));
                        }
                        for (; x < src_cols; ++x) {
                            uchar sb = spread_row[x];
                            uchar raw = std::max(lut_ptr[sb & 0x0F], lut_ptr[(sb >> 4) + 16]);
                            resp_row[x] = (uchar)(raw * disc_row[x] / 4);
                        }

                        // Strided decimate response_buf → linear memories.
                        // vld4q/vld2q deinterleave the T offsets in one pass.
                        if (T == 4) {
                            uchar *mp[4];
                            for (int k = 0; k < 4; ++k)
                                mp[k] = memories[ori].ptr(grid_row * 4 + k) + dec_r * mem_w;
                            int c = 0;
                            for (; c + 64 <= src_cols; c += 64) {
                                uint8x16x4_t q = vld4q_u8(resp_row + c);
                                vst1q_u8(mp[0], q.val[0]); mp[0] += 16;
                                vst1q_u8(mp[1], q.val[1]); mp[1] += 16;
                                vst1q_u8(mp[2], q.val[2]); mp[2] += 16;
                                vst1q_u8(mp[3], q.val[3]); mp[3] += 16;
                            }
                            for (int k = 0; k < 4; ++k)
                                for (int cc = c + k; cc < src_cols; cc += 4)
                                    *mp[k]++ = resp_row[cc];
                        } else if (T == 2) {
                            uchar *mp[2];
                            for (int k = 0; k < 2; ++k)
                                mp[k] = memories[ori].ptr(grid_row * 2 + k) + dec_r * mem_w;
                            int c = 0;
                            for (; c + 32 <= src_cols; c += 32) {
                                uint8x16x2_t q = vld2q_u8(resp_row + c);
                                vst1q_u8(mp[0], q.val[0]); mp[0] += 16;
                                vst1q_u8(mp[1], q.val[1]); mp[1] += 16;
                            }
                            for (int k = 0; k < 2; ++k)
                                for (int cc = c + k; cc < src_cols; cc += 2)
                                    *mp[k]++ = resp_row[cc];
                        } else {
                            for (int c_start = 0; c_start < T; ++c_start) {
                                int grid_index = grid_row * T + c_start;
                                uchar *mem_ptr = memories[ori].ptr(grid_index) + dec_r * mem_w;
                                for (int c = c_start; c < src_cols; c += T)
                                    *mem_ptr++ = resp_row[c];
                            }
                        }
#else
                        for (int c_start = 0; c_start < T; ++c_start) {
                            int grid_index = grid_row * T + c_start;
                            uchar *mem_ptr = memories[ori].ptr(grid_index) + dec_r * mem_w;

                            for (int c = c_start; c < src_cols; c += T) {
                                uchar sb = spread_row[c];
                                uchar raw = std::max(
                                    lut_ptr[sb & 0x0F],
                                    lut_ptr[(sb >> 4) + 16]);
                                *mem_ptr++ = (uchar)(raw * disc_row[c] / 4);
                            }
                        }
#endif
                    }
                }
            }

                if (g_profile.enabled)
                    g_profile.fused_spread_lut_ms += std::chrono::duration<double, std::milli>(
                        PClock::now() - fused_t0).count();
            }
        }

        sizes.push_back(quantized.size());
    }

    timer.out("construct response map");

    {
        using PClock = std::chrono::high_resolution_clock;
        auto match_t0 = PClock::now();

        if (class_ids.empty())
        {
            TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
            for (; it != itend; ++it)
                matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
        }
        else
        {
            for (int i = 0; i < (int)class_ids.size(); ++i)
            {
                TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
                if (it != class_templates.end())
                    matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
            }
        }

        if (g_profile.enabled) {
            double match_ms = std::chrono::duration<double, std::milli>(PClock::now() - match_t0).count();
            // Split coarse vs refine is inside matchClass, just record total here
            g_profile.coarse_match_ms += match_ms;
        }

        auto sort_t0 = PClock::now();
        std::sort(matches.begin(), matches.end());
        std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
        matches.erase(new_end, matches.end());
        if (g_profile.enabled)
            g_profile.sort_nms_ms += std::chrono::duration<double, std::milli>(PClock::now() - sort_t0).count();
    }

    timer.out("templ match");

    return matches;
}

// Used to filter out weak matches
struct MatchPredicate
{
    MatchPredicate(float _threshold) : threshold(_threshold) {}
    bool operator()(const Match &m) { return m.similarity < threshold; }
    float threshold;
};

// Score a single template at a single position in linear memory space.
// Returns raw similarity sum (not normalized to 0-100).
static int scoreAtPosition(const std::vector<Mat> &linear_memories,
                           const Template &templ, Size size, int T,
                           int match_x, int match_y) {
    int W = size.width / T;
    int score = 0;
    for (auto& f : templ.features) {
        // Feature position in image coords
        int fx = f.x + match_x;
        int fy = f.y + match_y;
        if (fx < 0 || fy < 0) continue;

        const Mat &memory_grid = linear_memories[f.label];
        int grid_x = fx % T;
        int grid_y = fy % T;
        int grid_index = grid_y * T + grid_x;
        if (grid_index >= memory_grid.rows) continue;

        int lm_x = fx / T;
        int lm_y = fy / T;
        int lm_index = lm_y * W + lm_x;
        if (lm_index < 0 || lm_index >= memory_grid.cols) continue;

        score += memory_grid.ptr(grid_index)[lm_index];
    }
    return score;
}

void Detector::refineOrientations(std::vector<Match> &matches, float angle_step,
                                  int num_templates) const {
    if (matches.empty() || num_templates < 3) return;

    // We need the linear memory pyramid that was built during match().
    // Since we don't cache it, we re-score using the stored templates.
    // However, the linear memories are not stored after match() returns.
    //
    // Alternative approach: use the scores of neighboring template_ids
    // that were already computed during matching. But those aren't stored either.
    //
    // Simplest approach: parabolic interpolation using the match scores
    // from templates at template_id-1, template_id, template_id+1.
    // We search through the raw matches list for neighbors at the same position.

    // Group matches by approximate position (within T pixels)
    // For each position, find the best template_id and its neighbors' scores.

    // Actually, the matches list already has all template scores above threshold.
    // For each NMS-surviving match, find its angular neighbors in the raw list.

    // Simplest correct approach: for each match, just do parabolic interpolation
    // using the assumption that similarity varies smoothly with angle.
    // We need scores at template_id-1 and template_id+1.
    // These might exist in the matches list, or we approximate.

    // Build a lookup: for each (class_id, approximate position), store all (template_id, score)
    struct PosKey {
        int x, y;
        bool operator==(const PosKey& o) const {
            return abs(x - o.x) < 16 && abs(y - o.y) < 16;
        }
    };

    // For each match, search for neighbors with template_id +/- 1 nearby
    for (auto& m : matches) {
        int tid = m.template_id;
        int tid_prev = (tid - 1 + num_templates) % num_templates;
        int tid_next = (tid + 1) % num_templates;

        float s_center = m.similarity;
        float s_prev = -1, s_next = -1;

        // Search in the full matches list for neighboring template scores at similar position
        // (matches is sorted by similarity descending, but we need spatial+angular neighbors)
        // This is O(N) per match but N is small after NMS.

        // Default: assume symmetric falloff if neighbor not found
        float default_neighbor = s_center * 0.9f;

        for (auto& other : matches) {
            if (other.class_id != m.class_id) continue;
            int dx = abs(other.x - m.x), dy = abs(other.y - m.y);
            if (dx > 16 || dy > 16) continue;  // not same object

            if (other.template_id == tid_prev && s_prev < other.similarity)
                s_prev = other.similarity;
            if (other.template_id == tid_next && s_next < other.similarity)
                s_next = other.similarity;
        }

        if (s_prev < 0) s_prev = default_neighbor;
        if (s_next < 0) s_next = default_neighbor;

        // Parabolic interpolation: fit y = a*x^2 + b*x + c through
        // (-1, s_prev), (0, s_center), (1, s_next)
        // Peak at x = -b/(2a) where a = (s_prev + s_next)/2 - s_center
        //                            b = (s_next - s_prev)/2
        float a = (s_prev + s_next) / 2.0f - s_center;
        float b = (s_next - s_prev) / 2.0f;

        float offset = 0;
        if (a < -0.001f) {  // concave (has a maximum)
            offset = -b / (2.0f * a);
            offset = std::max(-0.5f, std::min(0.5f, offset));  // clamp to [-0.5, 0.5]
        }

        m.refined_angle = (tid + offset) * angle_step;
        // Wrap to [0, 360)
        if (m.refined_angle < 0) m.refined_angle += 360.0f;
        if (m.refined_angle >= 360.0f) m.refined_angle -= 360.0f;
    }
}


void enableProfiling(bool enable) { g_profile.enabled = enable; }
void resetProfiling() { g_profile.reset(); }
void printProfiling() { g_profile.print(); }

void Detector::matchClass(const LinearMemoryPyramid &lm_pyramid,
                          const std::vector<Size> &sizes,
                          float threshold, std::vector<Match> &matches,
                          const std::string &class_id,
                          const std::vector<TemplatePyramid> &template_pyramids) const
{
    // MSVC OpenMP only supports 2.0 (no custom reductions, no unsigned loop var).
    // Use critical section for thread-safe match collection instead.
    int num_templates = static_cast<int>(template_pyramids.size());
#pragma omp parallel for schedule(dynamic)
    for (int template_id_i = 0; template_id_i < num_templates; ++template_id_i)
    {
        size_t template_id = static_cast<size_t>(template_id_i);
        const TemplatePyramid &tp = template_pyramids[template_id];
        // First match over the whole image at the lowest pyramid level
        /// @todo Factor this out into separate function
        const std::vector<LinearMemories> &lowest_lm = lm_pyramid.back();

        std::vector<Match> candidates;
        {
            // Compute similarity maps for each ColorGradient at lowest pyramid level
            Mat similarities;
            int lowest_start = static_cast<int>(tp.size() - 1);
            int lowest_T = T_at_level.back();
            int num_features = 0;

            {
                const Template &templ = tp[lowest_start];
                num_features += static_cast<int>(templ.features.size());

                if (templ.features.size() < 64){
                    similarity_64(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
                    similarities.convertTo(similarities, CV_16U);
                }else if (templ.features.size() < 8192){
                    similarity(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
                }else{
                    CV_Error(Error::StsBadArg, "feature size too large");
                }
            }

            // Find initial matches
            for (int r = 0; r < similarities.rows; ++r)
            {
                ushort *row = similarities.ptr<ushort>(r);
                for (int c = 0; c < similarities.cols; ++c)
                {
                    int raw_score = row[c];
                    float score = (raw_score * 100.f) / (4 * num_features);

                    if (score > threshold)
                    {
                        // Local-max prefilter: emit only the peak of each score blob,
                        // collapsing the above-threshold flood BEFORE cap/refine/NMS.
                        if (candidate_local_max) {
                            bool ismax = true;
                            for (int dr = -1; dr <= 1 && ismax; ++dr) {
                                int rr = r + dr;
                                if (rr < 0 || rr >= similarities.rows) continue;
                                const ushort* nrow = similarities.ptr<ushort>(rr);
                                for (int dc = -1; dc <= 1; ++dc) {
                                    int cc = c + dc;
                                    if (cc < 0 || cc >= similarities.cols || (dr == 0 && dc == 0)) continue;
                                    if (nrow[cc] > (ushort)raw_score) { ismax = false; break; }
                                }
                            }
                            if (!ismax) continue;
                        }
                        int offset = lowest_T / 2 + (lowest_T % 2 - 1);
                        int x = c * lowest_T + offset;
                        int y = r * lowest_T + offset;
                        candidates.push_back(Match(x, y, score, class_id, static_cast<int>(template_id)));
                    }
                }
            }
        }


        // Cap coarse candidates to top-K by score to prevent noise flooding.
        // Maximum coarse candidates per template before pyramid refinement.
        // On noisy images, thousands of false candidates pass the threshold.
        // Keeping only the best ones prevents the refinement stage from exploding.
        {
            const int MAX_CANDIDATES = 256;
            if ((int)candidates.size() > MAX_CANDIDATES) {
                std::partial_sort(candidates.begin(),
                                  candidates.begin() + MAX_CANDIDATES,
                                  candidates.end());
                // Match::operator< sorts by descending similarity
                candidates.resize(MAX_CANDIDATES);
            }
        }

        // Locally refine each match by marching up the pyramid
        for (int l = pyramid_levels - 2; l >= 0; --l)
        {
            const std::vector<LinearMemories> &lms = lm_pyramid[l];
            int T = T_at_level[l];
            int start = static_cast<int>(l);
            Size size = sizes[l];
            int border = 8 * T;
            int offset = T / 2 + (T % 2 - 1);
            int max_x = size.width - tp[start].width - border;
            int max_y = size.height - tp[start].height - border;

            Mat similarities2;
            for (int m = 0; m < (int)candidates.size(); ++m)
            {
                Match &match2 = candidates[m];
                int x = match2.x * 2 + 1; /// @todo Support other pyramid distance
                int y = match2.y * 2 + 1;

                // Require 8 (reduced) row/cols to the up/left
                x = std::max(x, border);
                y = std::max(y, border);

                // Require 8 (reduced) row/cols to the down/left, plus the template size
                x = std::min(x, max_x);
                y = std::min(y, max_y);

                // Compute local similarity maps for each ColorGradient
                int numFeatures = 0;

                {
                    const Template &templ = tp[start];
                    numFeatures += static_cast<int>(templ.features.size());

                    if (templ.features.size() < 64){
                        similarityLocal_64(lms[0], templ, similarities2, size, T, Point(x, y));
                        similarities2.convertTo(similarities2, CV_16U);
                    }else if (templ.features.size() < 8192){
                        similarityLocal(lms[0], templ, similarities2, size, T, Point(x, y));
                    }else{
                        CV_Error(Error::StsBadArg, "feature size too large");
                    }
                }

                // Find best local adjustment — find max ushort first, convert once
                int best_r = -1, best_c = -1;
                {
#ifdef __AVX2__
                    // AVX2: scan 16x16 ushort grid (each row = one 256-bit register)
                    __m256i global_max = _mm256_setzero_si256();
                    __m256i col_indices = _mm256_setr_epi16(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
                    __m256i best_col_vec = _mm256_setzero_si256();
                    int best_row_scalar = 0;
                    // Track per-column max to find row later
                    for (int r = 0; r < 16; ++r) {
                        __m256i row_v = _mm256_loadu_si256((const __m256i*)similarities2.ptr<ushort>(r));
                        __m256i gt = _mm256_cmpgt_epi16(row_v, global_max); // signed compare ok for ushort < 32768
                        global_max = _mm256_max_epu16(global_max, row_v);
                        best_col_vec = _mm256_blendv_epi8(best_col_vec, col_indices, gt);
                        // Check if this row had the new global max
                        // We need to reduce global_max to find the single winner
                    }
                    // Horizontal reduction to find max value and its position
                    alignas(32) uint16_t max_arr[16], col_arr[16];
                    _mm256_store_si256((__m256i*)max_arr, global_max);
                    _mm256_store_si256((__m256i*)col_arr, best_col_vec);
                    uint16_t best_val = 0;
                    int best_lane = 0;
                    for (int i = 0; i < 16; ++i) {
                        if (max_arr[i] > best_val) { best_val = max_arr[i]; best_lane = i; }
                    }
                    // Now find which row produced the max in this lane
                    best_c = best_lane;
                    best_r = 0;
                    for (int r = 0; r < 16; ++r) {
                        if (similarities2.ptr<ushort>(r)[best_lane] == best_val) {
                            best_r = r;
                            break;
                        }
                    }
#else
                    ushort best_val = 0;
                    for (int r = 0; r < similarities2.rows; ++r) {
                        ushort *row = similarities2.ptr<ushort>(r);
                        for (int c = 0; c < similarities2.cols; ++c) {
                            if (row[c] > best_val) {
                                best_val = row[c]; best_r = r; best_c = c;
                            }
                        }
                    }
#endif
                }
                float best_score = (best_r >= 0) ?
                    (similarities2.ptr<ushort>(best_r)[best_c] * 100.f) / (4 * numFeatures) : 0;
                // Update current match
                match2.similarity = best_score;
                match2.x = (x / T - 8 + best_c) * T + offset;
                match2.y = (y / T - 8 + best_r) * T + offset;
            }

            // Filter out any matches that drop below the similarity threshold
            std::vector<Match>::iterator new_end = std::remove_if(candidates.begin(), candidates.end(),
                                                                  MatchPredicate(threshold));
            candidates.erase(new_end, candidates.end());
        }

        #pragma omp critical
        matches.insert(matches.end(), candidates.begin(), candidates.end());
    }
}

int Detector::addTemplate(const Mat source, const std::string &class_id,
                          const Mat &object_mask, int num_features)
{
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    TemplatePyramid tp;
    tp.resize(pyramid_levels);

    if (scale_pyramid_features) {
        // Extract features only at level 0, scale coordinates for coarser levels.
        // Orientation labels from full-res are valid at all scales (proven empirically).
        Ptr<ColorGradientPyramid> qp = modality->process(source, object_mask);
        if (num_features > 0) qp->num_features = num_features;

        bool success = qp->extractTemplate(tp[0]);
        if (!success) return -1;

        for (int l = 1; l < pyramid_levels; ++l) {
            tp[l].pyramid_level = l;
            tp[l].angle = tp[0].angle;
            float s = 1.0f / (1 << l);  // 0.5 for l=1, 0.25 for l=2, etc.
            tp[l].features.clear();
            for (auto& f : tp[0].features) {
                Feature fn;
                fn.x = (int)(f.x * s + 0.5f);
                fn.y = (int)(f.y * s + 0.5f);
                fn.label = f.label;
                fn.theta = f.theta;
                tp[l].features.push_back(fn);
            }
            tp[l].tl_x = (int)(tp[0].tl_x * s + 0.5f);
            tp[l].tl_y = (int)(tp[0].tl_y * s + 0.5f);
            tp[l].width = (int)(tp[0].width * s + 0.5f);
            tp[l].height = (int)(tp[0].height * s + 0.5f);
        }
    } else {
        // Original: extract independently at each pyramid level
        Ptr<ColorGradientPyramid> qp = modality->process(source, object_mask);
        if (num_features > 0) qp->num_features = num_features;

        for (int l = 0; l < pyramid_levels; ++l) {
            if (l > 0) qp->pyrDown();
            bool success = qp->extractTemplate(tp[l]);
            if (!success) return -1;
        }
    }

    cropTemplates(tp);
    template_pyramids.push_back(tp);
    return template_id;
}

static cv::Point2f rotate2d(const cv::Point2f inPoint, const double angRad)
{
    cv::Point2f outPoint;
    outPoint.x = std::cos(angRad)*inPoint.x - std::sin(angRad)*inPoint.y;
    outPoint.y = std::sin(angRad)*inPoint.x + std::cos(angRad)*inPoint.y;
    return outPoint;
}

static cv::Point2f rotatePoint(const cv::Point2f inPoint, const cv::Point2f center, const double angRad)
{
    return rotate2d(inPoint - center, angRad) + center;
}

// Rotate a TemplatePyramid's features by theta degrees around center.
// Creates a new TemplatePyramid with rotated coordinates + orientation labels.
static std::vector<Template> rotateTemplatePyramid(const std::vector<Template>& src,
                                              float theta, cv::Point2f center,
                                              int pyramid_levels) {
    std::vector<Template> tp;
    tp.resize(pyramid_levels);
    // Negative angle for CW rotation in image coordinates (y-axis points down).
    // warpAffine with -angle rotates the object CW, so feature rotation must match.
    float angRad = -theta * (float)CV_PI / 180.0f;

    for (int l = 0; l < pyramid_levels; ++l) {
        cv::Point2f lvl_center = center;
        for (int i = 0; i < l; ++i) lvl_center *= 0.5f;

        for (auto& f : src[l].features) {
            cv::Point2f p((float)(f.x + src[l].tl_x),
                          (float)(f.y + src[l].tl_y));
            cv::Point2f p_rot = rotatePoint(p, lvl_center, angRad);

            Feature f_new;
            f_new.x = (int)(p_rot.x + 0.5f);
            f_new.y = (int)(p_rot.y + 0.5f);

            // Rotate orientation to match the rotated edge direction
            f_new.theta = f.theta - theta;
            while (f_new.theta >= 360) f_new.theta -= 360;
            while (f_new.theta < 0) f_new.theta += 360;
            // Round to nearest bin to match comparison-based scene quantization
            f_new.label = (int)(f_new.theta * 16.0f / 360.0f + 0.5f) & 7;

            tp[l].features.push_back(f_new);
        }
        tp[l].pyramid_level = l;
        tp[l].angle = theta;
    }
    return tp;
}

int Detector::addTemplate_rotate(const string &class_id, int zero_id,
                                 float theta, cv::Point2f center)
{
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());
    const auto& src = template_pyramids[zero_id];

    auto tp = rotateTemplatePyramid(src, theta, center, pyramid_levels);
    cropTemplates(tp);
    template_pyramids.push_back(tp);
    return template_id;
}

int Detector::addRotatedTemplates(const cv::Mat& templ_gray, const cv::Mat& object_mask,
                                   const std::string& class_id,
                                   float angle_start, float angle_end,
                                   float angle_step) {
    // Step 1: Extract features once at 0 degrees
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];

    // Use addTemplate for the base extraction (handles quantize + NMS + feature selection)
    int zero_id = addTemplate(templ_gray, class_id, object_mask);
    if (zero_id < 0) return -1;

    // Copy base template (the reference would be invalidated by push_back)
    auto base_tp = template_pyramids[zero_id];
    cv::Point2f center(templ_gray.cols / 2.0f, templ_gray.rows / 2.0f);

    // Step 2: Rotate features for all other angles
    int num_angles = 0;
    for (float a = angle_start + angle_step; a < angle_end; a += angle_step)
        ++num_angles;
    template_pyramids.reserve(template_pyramids.size() + num_angles);

    int count = 1;  // already have the 0-degree template
    for (float angle = angle_start + angle_step; angle < angle_end; angle += angle_step) {
        auto tp = rotateTemplatePyramid(base_tp, angle, center, pyramid_levels);
        cropTemplates(tp);
        template_pyramids.push_back(tp);
        count++;
    }

    return count;
}
const std::vector<Template> &Detector::getTemplates(const std::string &class_id, int template_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    CV_Assert(i != class_templates.end());
    CV_Assert(i->second.size() > size_t(template_id));
    return i->second[template_id];
}

int Detector::numTemplates() const
{
    int ret = 0;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
        ret += static_cast<int>(i->second.size());
    return ret;
}

int Detector::numTemplates(const std::string &class_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    if (i == class_templates.end())
        return 0;
    return static_cast<int>(i->second.size());
}

std::vector<std::string> Detector::classIds() const
{
    std::vector<std::string> ids;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
    {
        ids.push_back(i->first);
    }

    return ids;
}

void Detector::read(const FileNode &fn)
{
    class_templates.clear();
    pyramid_levels = fn["pyramid_levels"];
    fn["T"] >> T_at_level;

    modality = makePtr<ColorGradient>();
}

void Detector::write(FileStorage &fs) const
{
    fs << "pyramid_levels" << pyramid_levels;
    fs << "T" << T_at_level;

    modality->write(fs);
}

std::string Detector::readClass(const FileNode &fn, const std::string &class_id_override)
{
    // Detector should not already have this class
    String class_id;
    if (class_id_override.empty())
    {
        String class_id_tmp = fn["class_id"];
        CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
        class_id = class_id_tmp;
    }
    else
    {
        class_id = class_id_override;
    }

    TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
    std::vector<TemplatePyramid> &tps = v.second;
    int expected_id = 0;

    FileNode tps_fn = fn["template_pyramids"];
    tps.resize(tps_fn.size());
    FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
    for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
    {
        int template_id = (*tps_it)["template_id"];
        CV_Assert(template_id == expected_id);
        FileNode templates_fn = (*tps_it)["templates"];
        tps[template_id].resize(templates_fn.size());

        FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
        int idx = 0;
        for (; templ_it != templ_it_end; ++templ_it)
        {
            tps[template_id][idx++].read(*templ_it);
        }
    }

    class_templates.insert(v);
    return class_id;
}

void Detector::writeClass(const std::string &class_id, FileStorage &fs) const
{
    TemplatesMap::const_iterator it = class_templates.find(class_id);
    CV_Assert(it != class_templates.end());
    const std::vector<TemplatePyramid> &tps = it->second;

    fs << "class_id" << it->first;
    fs << "pyramid_levels" << pyramid_levels;
    fs << "template_pyramids"
       << "[";
    for (size_t i = 0; i < tps.size(); ++i)
    {
        const TemplatePyramid &tp = tps[i];
        fs << "{";
        fs << "template_id" << int(i); //TODO is this cast correct? won't be good if rolls over...
        fs << "templates"
           << "[";
        for (size_t j = 0; j < tp.size(); ++j)
        {
            fs << "{";
            tp[j].write(fs);
            fs << "}"; // current template
        }
        fs << "]"; // templates
        fs << "}"; // current pyramid
    }
    fs << "]"; // pyramids
}

void Detector::readClasses(const std::vector<std::string> &class_ids,
                           const std::string &format)
{
    for (size_t i = 0; i < class_ids.size(); ++i)
    {
        const String &class_id = class_ids[i];
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::READ);
        readClass(fs.root());
    }
}

void Detector::writeClasses(const std::string &format) const
{
    TemplatesMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
    for (; it != it_end; ++it)
    {
        const String &class_id = it->first;
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::WRITE);
        writeClass(class_id, fs);
    }
}

} // namespace line2Dup
