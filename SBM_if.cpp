#include "SBM_if.hpp"
#include <memory>
#include <UTIL.hpp>
#include <iostream>
#include <assert.h>
#include <chrono>
using namespace std;
using namespace cv;
 
SBM_if::SBM_if(): detector(6, {4, 8})
{

}

void SBM_if::train(Mat &img)
{
  Mat mask = Mat(img.size(), CV_8UC1, {255});

  Timer train_timer;
  shape_based_matching::shapeInfo_producer shapes(img, mask);
  shapes.angle_range = {0, 360};
  shapes.angle_step = 1;
  shapes.produce_infos();
  
  detector.addTemplate_rotate(class_id, shapes);

}
// only support gray img now
std::vector<line2Dup::Match> SBM_if::test(Mat &img)
{
  std::vector<std::string> ids;
  string class_id = "test";
  ids.push_back(class_id);
  return detector.match(img, 80, ids);
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
