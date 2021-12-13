#ifndef CXX_SBM_IF_HPP
#define CXX_SBM_IF_HPP
#include "line2Dup.h"
using namespace cv;
class SBM_if{
  public:
  std::string class_id = "test";


  struct anchorInfo{
    cv::Point2f offset;
    bool flip;
  };
  std::map<std::string, struct anchorInfo> template_Offset;
  std::string prefix = "test/";
  line2Dup::Detector detector;
  SBM_if();
  SBM_if(int num_features, std::vector<int> T, float weak_thresh = 30.0f, float strong_thresh = 60.0f);

  void regTemplateOffset(std::string class_id,struct anchorInfo anchor_info);

  struct anchorInfo fetchTemplateOffset(std::string class_id);

  int TemplateFeatureExtraction (const Mat source,
                          const Mat &object_mask, int num_features,line2Dup::TemplatePyramid &ret_tp);
  void train(Mat &img,float scaleN=1,Mat *mask=NULL);

  void train(std::string name,line2Dup::TemplatePyramid &_tp,cv::Point2f rotateCenter,bool y_flip=false,float scaleN=1,float angleFrom=0,float angleTo=360,int angleSegments=360);
  // void train(std::string name,line2Dup::TemplatePyramid &tp);
  std::vector<line2Dup::Match> test(Mat &img);

};


void MIPP_test();

#endif

