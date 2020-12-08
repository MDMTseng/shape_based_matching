#include "SBM_if.hpp"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
using namespace std;
using namespace cv;

SBM_if::SBM_if(): detector(128, {4, 8})
{

}

void SBM_if::train(Mat &img,int roi_x,int roi_y,int roi_w,int roi_h)
{
  assert(!img.empty() && "check your img path");

  Rect roi(roi_x, roi_y, roi_w, roi_h);
  img = img(roi).clone();
  Mat mask = Mat(img.size(), CV_8UC1, {255});

  // padding to avoid rotating out
  int padding = 100;
  cv::Mat padded_img = cv::Mat(img.rows + 2*padding, img.cols + 2*padding, img.type(), cv::Scalar::all(0));
  img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

  cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar::all(0));
  mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

  shape_based_matching::shapeInfo_producer shapes(padded_img, padded_mask);
  shapes.angle_range = {0, 360};
  shapes.angle_step = 1;
  shapes.produce_infos();
  std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
  string class_id = "test";
  for(auto& info: shapes.infos){
      imshow("train", shapes.src_of(info));
      waitKey(1);

      std::cout << "\ninfo.angle: " << info.angle << std::endl;
      int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
      std::cout << "templ_id: " << templ_id << std::endl;
      if(templ_id != -1){
          infos_have_templ.push_back(info);
      }
  }
  detector.writeClasses(prefix+"case1/%s_templ.yaml");
  shapes.save_infos(infos_have_templ, prefix + "case1/test_info.yaml");
  std::cout << "train end" << std::endl << std::endl;
}

// only support gray img now
void SBM_if::test(Mat &test_img)
{
  std::vector<std::string> ids;
  ids.push_back("test");
  detector.readClasses(ids, prefix+"case1/%s_templ.yaml");

  // angle & scale are saved here, fetched by match id
  auto infos = shape_based_matching::shapeInfo_producer::load_infos(prefix + "case1/test_info.yaml");

  assert(!test_img.empty() && "check your img path");

  int padding = 500;
  cv::Mat padded_img = cv::Mat(test_img.rows + 2*padding,
                                test_img.cols + 2*padding, test_img.type(), cv::Scalar::all(0));
  test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));

  // no need for stride now
  cv::Mat img = padded_img;
  std::cout << "test img size: " << img.rows * img.cols << std::endl << std::endl;

  Timer timer;
  detector.set_produce_dxy = true; // produce dxy, for icp purpose maybe
  std::vector<line2Dup::Match> matches = detector.match(img, 90, ids);
  timer.out("match total time");

  // test dx, dy;
  cv::Mat canny_edge;
  cv::Canny(detector.dx_, detector.dy_, canny_edge, 30, 60);
  cv::imshow("canny edge", canny_edge);
  // cv::waitKey();

  if(img.channels() == 1) cvtColor(img, img, cv::COLOR_GRAY2BGR);

  std::cout << "matches.size(): " << matches.size() << std::endl;
  size_t top5 = 1;
  if(top5>matches.size()) top5=matches.size();
  for(size_t i=0; i<top5; i++){
      auto match = matches[i];
      auto templ = detector.getTemplates("test",
                                          match.template_id);

      // 270 is width of template image
      // 100 is padding when training
      // tl_x/y: template croping topleft corner when training

      float r_scaled = 270/2.0f*infos[match.template_id].scale;

      // scaling won't affect this, because it has been determined by warpAffine
      // cv::warpAffine(src, dst, rot_mat, src.size()); last param
      float train_img_half_width = 270/2.0f + 100;

      // center x,y of train_img in test img
      float x =  match.x - templ[0].tl_x + train_img_half_width;
      float y =  match.y - templ[0].tl_y + train_img_half_width;

      cv::Vec3b randColor;
      randColor[0] = rand()%155 + 100;
      randColor[1] = rand()%155 + 100;
      randColor[2] = rand()%155 + 100;
      for(int i=0; i<templ[0].features.size(); i++){
          auto feat = templ[0].features[i];
          cv::circle(img, {feat.x+match.x, feat.y+match.y}, 3, randColor, -1);
      }

      cv::putText(img, to_string(int(round(match.similarity))),
                  Point(match.x+r_scaled-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);

      cv::RotatedRect rotatedRectangle({x, y}, {2*r_scaled, 2*r_scaled}, -infos[match.template_id].angle);

      cv::Point2f vertices[4];
      rotatedRectangle.points(vertices);
      for(int i=0; i<4; i++){
          int next = (i+1==4) ? 0 : (i+1);
          cv::line(img, vertices[i], vertices[next], randColor, 2);
      }

      std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
      std::cout << "match.similarity: " << match.similarity << std::endl;
  }

  imshow("img", img);
  waitKey(0);

  std::cout << "test end" << std::endl << std::endl;
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
