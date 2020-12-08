#include "line2Dup.h"
#include "SBM_if.hpp"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <UTIL.hpp>
using namespace std;
using namespace cv;

static std::string prefix = "test/";

void circle_gen(){
    Mat bg = Mat(800, 800, CV_8UC3, {0, 0, 0});
    cv::circle(bg, {400, 400}, 200, {255,255,255}, -1);
    cv::imshow("test", bg);
    waitKey(0);
}

void scale_test(string mode = "test"){
    int num_feature = 150;

    // feature numbers(how many ori in one templates?)
    // two pyramids, lower pyramid(more pixels) in stride 4, lower in stride 8
    line2Dup::Detector detector(num_feature, {4, 8});

//    mode = "test";
    if(mode == "train"){
        Mat img = cv::imread(prefix+"case0/templ/circle.png", cv::IMREAD_GRAYSCALE);
        assert(!img.empty() && "check your img path");
        shape_based_matching::shapeInfo_producer shapes(img);

        shapes.scale_range = {0.1f, 1};
        shapes.scale_step = 0.01f;
        shapes.produce_infos();

        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "circle";
        for(auto& info: shapes.infos){

            // template img, id, mask,
            //feature numbers(missing it means using the detector initial num)
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info),
                                                int(num_feature*info.scale));
            std::cout << "templ_id: " << templ_id << std::endl;

            // may fail when asking for too many feature_nums for small training img
            if(templ_id != -1){  // only record info when we successfully add template
                infos_have_templ.push_back(info);
            }
        }

        // save templates
        detector.writeClasses(prefix+"case0/%s_templ.yaml");

        // save infos,
        // in this simple case infos are not used
        shapes.save_infos(infos_have_templ, prefix + "case0/circle_info.yaml");
        std::cout << "train end" << std::endl << std::endl;

    }else if(mode=="test"){
        std::vector<std::string> ids;

        // read templates
        ids.push_back("circle");
        detector.readClasses(ids, prefix+"case0/%s_templ.yaml");

        Mat test_img = imread(prefix+"case0/1.jpg", cv::IMREAD_GRAYSCALE);
        assert(!test_img.empty() && "check your img path");

        // no need stride now
        Mat img = test_img.clone();

        if(test_img.channels() == 1) cvtColor(test_img, test_img, COLOR_GRAY2BGR);

        Timer timer;
        // match, img, min socre, ids
        auto matches = detector.match(img, 75, ids);
        // one output match:
        // x: top left x
        // y: top left y
        // template_id: used to find templates
        // similarity: scores, 100 is best
        timer.out();

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 5;
        if(top5>matches.size()) top5=matches.size();
        for(size_t i=0; i<top5; i++){
            auto match = matches[i];
            auto templ = detector.getTemplates("circle",
                                               match.template_id);
            // template:
            // nums: num_pyramids * num_modality (modality, depth or RGB, always 1 here)
            // template[0]: lowest pyrimad(more pixels)
            // template[0].width: actual width of the matched template
            // template[0].tl_x / tl_y: topleft corner when cropping templ during training
            // In this case, we can regard width/2 = radius
            int x =  templ[0].width/2 + match.x;
            int y = templ[0].height/2 + match.y;
            int r = templ[0].width/2;
            Scalar color(255, rand()%255, rand()%255);

            cv::putText(img, to_string(int(round(match.similarity))),
                        Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, color);
            cv::circle(img, {x, y}, r, color, 2);
        }

        imshow("img", img);
        waitKey(0);

        std::cout << "test end" << std::endl << std::endl;
    }
}

void angle_test(string mode = "test"){
    line2Dup::Detector detector(128, {4, 8});

//    mode = "test";
    if(mode == "train"){
        Mat img = imread(prefix+"case1/train.png");
        assert(!img.empty() && "check your img path");

        Rect roi(130, 110, 270, 270);
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
    }else if(mode=="test"){
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(ids, prefix+"case1/%s_templ.yaml");

        // angle & scale are saved here, fetched by match id
        auto infos = shape_based_matching::shapeInfo_producer::load_infos(prefix + "case1/test_info.yaml");

        // only support gray img now
        Mat test_img = imread(prefix+"case1/test.png");
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



}

void noise_test(string mode = "test"){
    line2Dup::Detector detector(64, {4,8});

//    mode = "test";
    if(mode == "train"){
        Mat img = imread(prefix+"case2/train.png");
        assert(!img.empty() && "check your img path");
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        shape_based_matching::shapeInfo_producer shapes(img, mask);
        shapes.angle_range = {0, 360};
        shapes.angle_step = 1;
        shapes.produce_infos();
        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "test";
        
        detector.addTemplate_rotate(class_id, shapes);

        
        // addTemplate_rotate(class_id, int zero_id, float theta, cv::Point2f center);
        for(auto& info: shapes.infos){
            // auto mat = shapes.src_of(info);
            // imshow("train", mat);
            // waitKey(1);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            // int templ_id = detector.addTemplate(mat, class_id, shapes.mask_of(info));
            // std::cout << "templ_id: " << templ_id << std::endl;
            // if(templ_id != -1)
                infos_have_templ.push_back(info);
            
        }
        detector.writeClasses(prefix+"case2/%s_templ.yaml");
        shapes.save_infos(infos_have_templ, prefix + "case2/test_info.yaml");
        std::cout << "train end" << std::endl << std::endl;
    }else if(mode=="test"){
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(ids, prefix+"case2/%s_templ.yaml");

        Mat test_img = imread(prefix+"case2/test.png", cv::IMREAD_GRAYSCALE);
        assert(!test_img.empty() && "check your img path");

        Timer timer;
        auto matches = detector.match(test_img, 80, ids);
        timer.out();

        if(test_img.channels() == 1) cvtColor(test_img, test_img, COLOR_GRAY2BGR);

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 500;
        if(top5>matches.size()) top5=matches.size();

        vector<Rect> boxes;
        vector<float> scores;
        vector<int> idxs;
        for(auto match: matches){
            Rect box;
            box.x = match.x;
            box.y = match.y;

            auto templ = detector.getTemplates("test",
                                               match.template_id);

            box.width = templ[0].width;
            box.height = templ[0].height;
            boxes.push_back(box);
            scores.push_back(match.similarity);
        }
        cv_dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs);

        for(auto idx: idxs){
            auto match = matches[idx];
            auto templ = detector.getTemplates("test",
                                               match.template_id);

            int x =  templ[0].width + match.x;
            int y = templ[0].height + match.y;
            int r = templ[0].width/2;
            cv::Vec3b randColor;
            randColor[0] = rand()%155 + 100;
            randColor[1] = rand()%155 + 100;
            randColor[2] = rand()%155 + 100;

            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(test_img, {feat.x+match.x, feat.y+match.y}, 2, randColor, -1);
            }

            cv::putText(test_img, to_string(int(round(match.similarity))),
                        Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);
            cv::rectangle(test_img, {match.x, match.y}, {x, y}, randColor, 2);

            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }

        imshow("img", test_img);
        waitKey(0);

        std::cout << "test end" << std::endl << std::endl;
    }
    else if(mode=="traintest")
    {
        std::vector<std::string> ids;
        string class_id = "test";
        ids.push_back(class_id);

        SBM_if sbmif;
        {
            Mat img = imread(prefix+"case2/train.png");
            assert(!img.empty() && "check your img path");

            Mat mask = Mat(img.size(), CV_8UC1, {255});

            // padding to avoid rotating out
            int padding = 0;
            cv::Mat padded_img = img;

            cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar::all(0));
            mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));



            Timer train_timer;
            shape_based_matching::shapeInfo_producer shapes(padded_img, padded_mask);
            shapes.angle_range = {0, 360};
            shapes.angle_step = 1;
            shapes.produce_infos();

            
            detector.addTemplate_rotate(class_id, shapes);

            train_timer.out("Training OK!!");





        }








        Mat test_img = imread(prefix+"case2/test.png", cv::IMREAD_GRAYSCALE);
        assert(!test_img.empty() && "check your img path");
 
        Timer timer;
        auto matches = detector.match(test_img, 80, ids);
        timer.out();

        if(test_img.channels() == 1) cvtColor(test_img, test_img, COLOR_GRAY2BGR);

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 500;
        if(top5>matches.size()) top5=matches.size();

        vector<Rect> boxes;
        vector<float> scores;
        vector<int> idxs;
        for(auto match: matches){
            Rect box;
            box.x = match.x;
            box.y = match.y;

            auto templ = detector.getTemplates("test",
                                               match.template_id);

            box.width = templ[0].width;
            box.height = templ[0].height;
            boxes.push_back(box);
            scores.push_back(match.similarity);
        }
        cv_dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs);

        for(auto idx: idxs){
            auto match = matches[idx];

            auto templ = detector.getTemplates("test",
                                            match.template_id);
            int x =  templ[0].width + match.x;
            int y = templ[0].height + match.y;
            int r = templ[0].width/2;


            cv::Vec3b randColor;
            randColor[0] = rand()%155 + 100;
            randColor[1] = rand()%155 + 100; 
            randColor[2] = rand()%155 + 100;

            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(test_img, {feat.x+match.x, feat.y+match.y}, 5, randColor, -1);
            }

            cv::putText(test_img, to_string(int(round(match.similarity))),
                        Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);

            printf("xy:%d,%d  wh:%d,%d\n",x,y,templ[0].width,templ[0].height);
            cv::RotatedRect rotatedRectangle(
              {(float)(x+match.x)/2, (float)(y+match.y)/2}, 
              {(float)templ[0].width, (float)templ[0].height}, 
              -templ[0].angle);

            cv::Point2f vertices[4];
            rotatedRectangle.points(vertices);
            for(int i=0; i<4; i++){
                int next = (i+1)%4;
                cv::line(test_img, vertices[i], vertices[next], randColor, 2);
            }





            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }

        imshow("img", test_img);
        waitKey(0);

        std::cout << "test end" << std::endl << std::endl;
    }

}
 
void view_angle(){
    float weak_thresh = 30.0f;

    // default params for detector
    line2Dup::Detector detector(63, {4, 8}, weak_thresh, 60.0f);
    // last two: magnitude thresh to extract angle in test image;
    //magnitude thresh to extract template points in train image;

    Mat img = cv::imread(prefix+"case0/templ/circle.png");
    assert(!img.empty() && "check your img path");
    imshow("img", img);
    cv::Mat gray;
    cv::cvtColor(img, gray, COLOR_GRAY2BGR);

    GaussianBlur(gray, gray, {5, 5}, 0);

    Mat grad1,grad2,angle;
    Sobel(gray, grad1, CV_32FC1, 1, 0);
    Sobel(gray, grad2, CV_32FC1, 0, 1);
    phase(grad1, grad2, angle, true);
    for(int r=0; r<angle.rows; r++){
        for(int c=0; c<angle.cols; c++){
            if(angle.at<float>(r, c) > 180)
            angle.at<float>(r, c) -= 180;
        }
    }
    angle.convertTo(angle,CV_8UC1);
    Mat grad_mask = (grad1.mul(grad1) + grad2.mul(grad2)) > weak_thresh*weak_thresh;
    Mat angle_masked;
    angle.copyTo(angle_masked, grad_mask);
    imshow("mask", grad_mask);
    imshow("angle", angle_masked);
    // angle masked is what we use for shape based matching
    cv::waitKey(0);
}

int main(){

    MIPP_test();
    noise_test("traintest"); // test or train
    return 0;
}
