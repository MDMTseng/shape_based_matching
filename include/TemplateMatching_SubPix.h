#pragma once

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)




cv::Point TemplateMatching_Pix(cv::Mat &img,cv::Mat &templ,cv::Mat &result,bool &isResForMax);
cv::Point2f TemplateMatching_SubPix(cv::Mat &img,cv::Mat &templ,cv::Mat &result,bool &isResForMax);



//CV_EXPORTS_W 
int  minMaxLocSubPix(CV_OUT cv::Point2d* SubPixLoc,
						cv::InputArray src,						
                           CV_IN_OUT cv::Point* LocIn,						   
						   CV_IN_OUT const int Method = 0
                           );
