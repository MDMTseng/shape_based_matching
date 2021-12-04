#ifndef CXX_SBM_IF_HPP
#define CXX_SBM_IF_HPP
#include "line2Dup.h"
using namespace cv;
class SBM_if{
  public:
  std::string class_id = "test";
  std::string prefix = "test/";
  line2Dup::Detector detector;
  SBM_if();
  SBM_if(int num_features, std::vector<int> T, float weak_thresh = 30.0f, float strong_thresh = 60.0f);
  int TemplateFeatureExtraction (const Mat source,
                          const Mat &object_mask, int num_features,line2Dup::TemplatePyramid &ret_tp);
  void train(Mat &img,float scaleN=1,Mat *mask=NULL);
  std::vector<line2Dup::Match> test(Mat &img);

};


void MIPP_test();

#endif

