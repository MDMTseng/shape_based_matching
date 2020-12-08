#ifndef CXX_SBM_IF_HPP
#define CXX_SBM_IF_HPP
#include "line2Dup.h"
using namespace cv;
class SBM_if{

  std::string prefix = "test/";
  line2Dup::Detector detector;
  SBM_if();

  void train(Mat &img,int roi_x,int roi_y,int roi_w,int roi_h);
  void test(Mat &test_img);
};


void MIPP_test();

#endif

