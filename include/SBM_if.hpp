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

  void train(Mat &img);
  std::vector<line2Dup::Match> test(Mat &img);

};


void MIPP_test();

#endif

