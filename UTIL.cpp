#include "line2Dup.h"
#include "UTIL.hpp"
using namespace cv;
// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

inline void cv_dnn::GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

template <typename BoxType>
inline void cv_dnn::NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&,void* ctx),void* ctx)
{
    CV_Assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        // printf("w:%d  h:%d\n", bboxes[i].width, bboxes[i].height);
        const int idx = score_index_vec[i].second;
        bool keep = true;
        float overlap=-1;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            overlap = computeOverlap(bboxes[idx], bboxes[kept_idx],ctx);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
            indices.push_back(idx);
        else
        {
            // printf("overlap:%.2f thres:%f\n",overlap,adaptive_threshold);
        }
        if (keep && eta < 1 && adaptive_threshold > 0.1) {
          adaptive_threshold *= eta;
        }
    }
}



// copied from opencv 3.4, not exist in 3.0
template<typename _Tp> static inline
double  cv_dnn::jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    float Aab = (a & b).area();
    float dist=1.0f - Aab / (Aa + Ab - Aab);

    return dist;
}

template <typename T>
inline float cv_dnn::rectOverlap(const T& a, const T& b,void* ctx)
{
    float angle_diff = fmod(fabs(a.angle_deg - b.angle_deg), 360.0f);
    if (angle_diff > 180.0f) {
        angle_diff = 360.0f - angle_diff;
    }
    // if(angle_diff>10)//large angle diff, consider as no overlap
    // {
    //     return 0.f;
    // }




    
    return 1.f - static_cast<float>(jaccardDistance__(a.rect, b.rect));
}
//Non Maximum Suppression
void cv_dnn::NMSBoxes(const std::vector<NMSBoxesStruct>& bboxes, // Vector of bounding boxes
                      const std::vector<float>& scores, // Vector of scores corresponding to each bounding box
                      const float score_threshold, // Minimum score required to keep a bounding box
                      const float nms_threshold, // Threshold for non-maximum suppression
                      std::vector<int>& indices, // Output vector of indices of kept bounding boxes
                      const float eta, // Soft NMS parameter
                      const int top_k) // Maximum number of bounding boxes to keep
{
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap,NULL);
}
