#include "line2Dup.h"
#include <iostream>

#ifdef __AVX2__
#include <immintrin.h>
#elif defined(_MSC_VER) && defined(__AVX2__)
#include <immintrin.h>
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
        std::cout << message << "\nelasped time:" << t << "s\n" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

namespace line2Dup
{
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
                static const int NEIGHBOR_THRESHOLD = 5;
                if (max_votes >= NEIGHBOR_THRESHOLD)
                    quantized_angle.at<uchar>(r, c) = uchar(1 << index);
            }
        }
    }
}

static void quantizedOrientations(const Mat &src, Mat &magnitude,
                                  Mat &angle, Mat& angle_ori, float threshold)
{
    Mat smoothed;
    // Compute horizontal and vertical image derivatives on all color channels separately
    static const int KERNEL_SIZE = 7;
    // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
    GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

    if(src.channels() == 1){
        // FAST PATH: integer Sobel + comparison-based 8-bin quantization.
        // Avoids cv::phase() (atan2 on every pixel) which dominates preprocessing.
        // Instead, determine orientation bin from dx/dy sign + magnitude comparisons.
        Mat sobel_dx_16s, sobel_dy_16s;
        Sobel(smoothed, sobel_dx_16s, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
        Sobel(smoothed, sobel_dy_16s, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);

        // Squared magnitude (float) for threshold compatibility
        magnitude.create(src.size(), CV_32F);
        magnitude.setTo(0);

        // angle_ori: compute per-feature atan2 during training, not per-pixel.
        // Store dx/dy as float for the few features that need theta.
        angle_ori.create(src.size(), CV_32F);
        angle_ori.setTo(0);

        // Direct 8-bin quantization from dx, dy (no atan2).
        // Undirected gradients: [0,180) mapped to 8 bins of 22.5 deg each.
        // Bin centers: 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5
        // Fixed-point tan boundaries * 10000:
        static const int TAN_B[4] = {1989, 6682, 14966, 50273};

        // Step 1: Compute magnitude + unfiltered 8-bin quantization
        // AVX2 fast path: compute L1 magnitude for 16 pixels at once,
        // then scalar bin computation only for above-threshold pixels.
        Mat quantized_unfiltered = Mat::zeros(src.size(), CV_8U);
        int threshold_i = (int)threshold;  // L1 threshold (not squared)
        float threshold_sq = threshold * threshold;  // for magnitude Mat compatibility

        for (int r = 1; r < src.rows - 1; ++r) {
            const short *dx = sobel_dx_16s.ptr<short>(r);
            const short *dy = sobel_dy_16s.ptr<short>(r);
            float *mag_r = magnitude.ptr<float>(r);
            uchar *qr = quantized_unfiltered.ptr<uchar>(r);

            int c = 1;
#ifdef __AVX2__
            // AVX2: compute magnitude for 16 pixels, threshold, then
            // only process above-threshold pixels with scalar bin logic.
            const __m256i thresh_v = _mm256_set1_epi16((short)threshold_i);
            for (; c <= src.cols - 1 - 16; c += 16) {
                __m256i vdx = _mm256_loadu_si256((const __m256i*)(dx + c));
                __m256i vdy = _mm256_loadu_si256((const __m256i*)(dy + c));
                __m256i adx = _mm256_abs_epi16(vdx);
                __m256i ady = _mm256_abs_epi16(vdy);
                __m256i mag_l1 = _mm256_add_epi16(adx, ady);

                // Store squared magnitude as float for compatibility
                // Widen to int32 and convert
                __m256i lo16 = _mm256_unpacklo_epi16(vdx, _mm256_setzero_si256());
                __m256i hi16 = _mm256_unpackhi_epi16(vdx, _mm256_setzero_si256());
                // Actually, just compute and store mag_sq with scalar below
                // since the float store is not the bottleneck

                // Check which pixels are above threshold
                __m256i above = _mm256_cmpgt_epi16(mag_l1, thresh_v);
                int mask = _mm256_movemask_epi8(above);

                if (mask == 0) {
                    // All below threshold: just store zero magnitudes
                    for (int i = 0; i < 16; ++i) {
                        int gx = dx[c+i], gy = dy[c+i];
                        mag_r[c+i] = (float)(gx*gx + gy*gy);
                    }
                    continue;
                }

                // Some pixels above threshold: scalar bin computation
                for (int i = 0; i < 16; ++i) {
                    int gx = dx[c+i], gy = dy[c+i];
                    int mag_sq_i = gx*gx + gy*gy;
                    mag_r[c+i] = (float)mag_sq_i;

                    int abs_gx = abs(gx), abs_gy = abs(gy);
                    if (abs_gx + abs_gy <= threshold_i) continue;

                    int ugx = gx, ugy = gy;
                    if (ugy < 0) { ugx = -ugx; ugy = -ugy; }
                    if (ugy == 0 && ugx < 0) ugx = -ugx;

                    int bin;
                    if (ugx >= 0) {
                        long long test_y = (long long)ugy * 10000;
                        if (test_y < (long long)ugx * TAN_B[0]) bin = 0;
                        else if (test_y < (long long)ugx * TAN_B[1]) bin = 1;
                        else if (test_y < (long long)ugx * TAN_B[2]) bin = 2;
                        else if (test_y < (long long)ugx * TAN_B[3]) bin = 3;
                        else bin = 4;
                    } else {
                        int agx = -ugx;
                        long long test_y = (long long)ugy * 10000;
                        if (test_y < (long long)agx * TAN_B[0]) bin = 0;
                        else if (test_y < (long long)agx * TAN_B[1]) bin = 7;
                        else if (test_y < (long long)agx * TAN_B[2]) bin = 6;
                        else if (test_y < (long long)agx * TAN_B[3]) bin = 5;
                        else bin = 4;
                    }
                    qr[c+i] = (uchar)bin;
                }
            }
#endif
            // Scalar tail
            for (; c < src.cols - 1; ++c) {
                int gx = dx[c], gy = dy[c];
                int mag_sq_i = gx*gx + gy*gy;
                mag_r[c] = (float)mag_sq_i;

                int abs_gx = abs(gx), abs_gy = abs(gy);
                if (abs_gx + abs_gy <= threshold_i) continue;

                int ugx = gx, ugy = gy;
                if (ugy < 0) { ugx = -ugx; ugy = -ugy; }
                if (ugy == 0 && ugx < 0) ugx = -ugx;

                int bin;
                if (ugx >= 0) {
                    long long test_y = (long long)ugy * 10000;
                    if (test_y < (long long)ugx * TAN_B[0]) bin = 0;
                    else if (test_y < (long long)ugx * TAN_B[1]) bin = 1;
                    else if (test_y < (long long)ugx * TAN_B[2]) bin = 2;
                    else if (test_y < (long long)ugx * TAN_B[3]) bin = 3;
                    else bin = 4;
                } else {
                    int agx = -ugx;
                    long long test_y = (long long)ugy * 10000;
                    if (test_y < (long long)agx * TAN_B[0]) bin = 0;
                    else if (test_y < (long long)agx * TAN_B[1]) bin = 7;
                    else if (test_y < (long long)agx * TAN_B[2]) bin = 6;
                    else if (test_y < (long long)agx * TAN_B[3]) bin = 5;
                    else bin = 4;
                }
                qr[c] = (uchar)bin;
            }
        }

        // Step 2: Fast 3x3 neighborhood voting.
        // Instead of full 8-bin histogram, count how many of the 8 neighbors
        // share the center pixel's bin. Cheaper: 8 equality checks vs 9 loads +
        // histogram + find-max. Semantically equivalent when center bin wins.
        angle = Mat::zeros(src.size(), CV_8U);
        static const int NEIGHBOR_THRESHOLD = 5; // 5 of 9 (center + 4 neighbors)
        int q_step = static_cast<int>(quantized_unfiltered.step1());

        for (int r = 1; r < src.rows - 1; ++r) {
            float *mag_r = magnitude.ptr<float>(r);
            const uchar *q_prev = quantized_unfiltered.ptr<uchar>(r-1);
            const uchar *q_curr = quantized_unfiltered.ptr<uchar>(r);
            const uchar *q_next = quantized_unfiltered.ptr<uchar>(r+1);
            uchar *angle_r = angle.ptr<uchar>(r);

            for (int c = 1; c < src.cols - 1; ++c) {
                if (mag_r[c] > threshold_sq) {
                    uchar center_bin = q_curr[c];
                    // Count center + 8 neighbors matching center_bin
                    int votes = 1; // center always matches itself
                    votes += (q_prev[c-1] == center_bin);
                    votes += (q_prev[c]   == center_bin);
                    votes += (q_prev[c+1] == center_bin);
                    votes += (q_curr[c-1] == center_bin);
                    votes += (q_curr[c+1] == center_bin);
                    votes += (q_next[c-1] == center_bin);
                    votes += (q_next[c]   == center_bin);
                    votes += (q_next[c+1] == center_bin);

                    if (votes >= NEIGHBOR_THRESHOLD)
                        angle_r[c] = (uchar)(1 << center_bin);
                }
            }
        }

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
                                           float _strong_threshold)
    : src(_src),
      mask(_mask),
      pyramid_level(0),
      weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
    update();
}

void ColorGradientPyramid::update()
{
    quantizedOrientations(src, magnitude, angle, angle_ori, weak_threshold);
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
    dst = Mat::zeros(angle.size(), CV_8U);
    angle.copyTo(dst, mask);
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
            std::cout << "too few features, abort" << std::endl;
            return false;
        }
        std::cout << "have no enough features, exaustive mode" << std::endl;
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
    // Max score per feature = 4. In a batch of 63 features: 63*4=252 < 255.
    // Accumulate in uint8 at FULL SIMD width (32 bytes on AVX2 = 2x throughput).
    // Widen to int16 only between batches.
    const int BATCH = 63;

    for (int batch_start = 0; batch_start < num_valid; batch_start += BATCH) {
        int batch_end = std::min(batch_start + BATCH, num_valid);

#ifdef __AVX2__
        // AVX2 path: accumulate batch in uint8 temp buffer, then widen
        std::vector<uint8_t> acc8(template_positions, 0);

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
    quantizers.push_back(modality->process(source, mask));

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

            // FULLY FUSED: spread + computeResponseMaps + linearize in one pass.
            // Instead of creating WxH spread buffer then reading it back,
            // compute the spread on-the-fly per row using two small temp buffers.
            // Eliminates both h_spread and spread_quantized intermediate allocations.
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

                // Popcount discount table: fewer bits set = more discriminative.
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
                        // For T=4: use SSE shuffles to extract every 4th byte.
                        // Each 16-byte load yields 4 output bytes; 4 loads → 16 outputs.
                        if (T == 4) {
                            // Shuffle masks: pick every 4th byte at each offset
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
        }

        sizes.push_back(quantized.size());
    }

    timer.out("construct response map");

    if (class_ids.empty())
    {
        // Match all templates
        TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
        for (; it != itend; ++it)
            matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
    }
    else
    {
        // Match only templates for the requested class IDs
        for (int i = 0; i < (int)class_ids.size(); ++i)
        {
            TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
            if (it != class_templates.end())
                matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
        }
    }

    // Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
    std::sort(matches.begin(), matches.end());
    std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
    matches.erase(new_end, matches.end());

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
                        int offset = lowest_T / 2 + (lowest_T % 2 - 1);
                        int x = c * lowest_T + offset;
                        int y = r * lowest_T + offset;
                        candidates.push_back(Match(x, y, score, class_id, static_cast<int>(template_id)));
                    }
                }
            }
        }


        // Cap coarse candidates to top-K by score to prevent noise flooding.
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

                // Find best local adjustment
                float best_score = 0;
                int best_r = -1, best_c = -1;
                for (int r = 0; r < similarities2.rows; ++r)
                {
                    ushort *row = similarities2.ptr<ushort>(r);
                    for (int c = 0; c < similarities2.cols; ++c)
                    {
                        int score_int = row[c];
                        float score = (score_int * 100.f) / (4 * numFeatures);

                        if (score > best_score)
                        {
                            best_score = score;
                            best_r = r;
                            best_c = c;
                        }
                    }
                }
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

    {
        // Extract a template at each pyramid level
        Ptr<ColorGradientPyramid> qp = modality->process(source, object_mask);

        if(num_features > 0)
        qp->num_features = num_features;

        for (int l = 0; l < pyramid_levels; ++l)
        {
            /// @todo Could do mask subsampling here instead of in pyrDown()
            if (l > 0)
                qp->pyrDown();

            bool success = qp->extractTemplate(tp[l]);
            if (!success)
                return -1;
        }
    }

    //    Rect bb =
    cropTemplates(tp);

    /// @todo Can probably avoid a copy of tp here with swap
    template_pyramids.push_back(tp);
    return template_id;
}

static cv::Point2f rotate2d(const cv::Point2f inPoint, const double angRad)
{
    cv::Point2f outPoint;
    //CW rotation
    outPoint.x = std::cos(angRad)*inPoint.x - std::sin(angRad)*inPoint.y;
    outPoint.y = std::sin(angRad)*inPoint.x + std::cos(angRad)*inPoint.y;
    return outPoint;
}

static cv::Point2f rotatePoint(const cv::Point2f inPoint, const cv::Point2f center, const double angRad)
{
    return rotate2d(inPoint - center, angRad) + center;
}

int Detector::addTemplate_rotate(const string &class_id, int zero_id,
                                 float theta, cv::Point2f center)
{
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    const auto& to_rotate_tp = template_pyramids[zero_id];

    TemplatePyramid tp;
    tp.resize(pyramid_levels);

    for (int l = 0; l < pyramid_levels; ++l)
    {
        if(l>0) center /= 2;

        for(auto& f: to_rotate_tp[l].features){
            Point2f p;
            p.x = f.x + to_rotate_tp[l].tl_x;
            p.y = f.y + to_rotate_tp[l].tl_y;
            Point2f p_rot = rotatePoint(p, center, -theta/180*CV_PI);

            Feature f_new;
            f_new.x = int(p_rot.x + 0.5f);
            f_new.y = int(p_rot.y + 0.5f);

            f_new.theta = f.theta - theta;
            while(f_new.theta > 360) f_new.theta -= 360;
            while(f_new.theta < 0) f_new.theta += 360;

            f_new.label = int(f_new.theta * 16 / 360 + 0.5f);
            f_new.label &= 7;


            tp[l].features.push_back(f_new);
        }
        tp[l].pyramid_level = l;
    }

    cropTemplates(tp);

    template_pyramids.push_back(tp);
    return template_id;
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
