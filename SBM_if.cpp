#include "SBM_if.hpp"
#include <memory>
#include <UTIL.hpp>
#include <iostream>
#include <assert.h>
#include <chrono>
using namespace std;
using namespace cv;
 
SBM_if::SBM_if(): detector(60, {4,8},30,80)
{

}

void SBM_if::train(Mat &img)
{
  Mat mask = Mat(img.size(), CV_8UC1, {255});

  line2Dup::TemplatePyramid tp;
  detector.TemplateFeatureExtraction (img, mask, 128,tp);






  auto center = cv::Point2f(img.cols/2.0,img.rows/2.0);
  detector.addTemplate_rotate(class_id,tp,center);

  for(int i=0;i<tp.size();i++)
  {
    // printf("pyLevel[%d]: xy:%d %d wh:%d  %d\n",i,tp[i].tl_x,tp[i].tl_y,tp[i].width,tp[i].height);
    
    int minY=999;
    int maxY=0;
    
    for (auto& f : tp[i].features)
    {
      int trueY=tp[i].tl_y+f.y;
      if(minY>trueY)
      {
        minY=trueY;
      }
      if(maxY>trueY)
      {
        maxY=trueY;
      }
      f.y=trueY;
    }

    for (auto& f : tp[i].features)
    {
      f.y=maxY-f.y;
      f.theta*=-1;
    }


    // printf("pyLevel[%d]: xy:%d %d wh:%d  %d\n",i,tp[i].tl_x,tp[i].tl_y,tp[i].width,tp[i].height);
    
  }

  detector.addTemplate_rotate(class_id+"_f",tp,center);
}
// only support gray img now
std::vector<line2Dup::Match> SBM_if::test(Mat &img)
{
  return detector.match(img, 60,180, {class_id,class_id+"_f"});
}

void MIPP_test(){
    std::cout << "MIPP tests" << std::endl;
    std::cout << "----------" << std::endl << std::endl;

    std::cout << "Instr. type:       " << mipp::InstructionType                  << std::endl;
    std::cout << "Instr. full type:  " << mipp::InstructionFullType              << std::endl;
    std::cout << "Instr. version:    " << mipp::InstructionVersion               << std::endl;
    std::cout << "Instr. size:       " << mipp::RegisterSizeBit       << " bits" << std::endl;
    std::cout << "Instr. lanes:      " << mipp::Lanes                            << std::endl;
    std::cout << "64-bit support:    " << (mipp::Support64Bit    ? "yes" : "no") << std::endl;
    std::cout << "Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no") << std::endl;

#ifndef has_max_int8_t
        std::cout << "in this SIMD, int8 max is not inplemented by MIPP" << std::endl;
#endif

#ifndef has_shuff_int8_t
        std::cout << "in this SIMD, int8 shuff is not inplemented by MIPP" << std::endl;
#endif

    std::cout << "----------" << std::endl << std::endl;
}

// int main(){

//     MIPP_test();
//     angle_test("train"); // test or train
//     return 0;
// }
