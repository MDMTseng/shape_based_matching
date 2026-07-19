// feat_count.cpp — how many features can a template actually yield? Extract at
// rising num_features and print the returned finest-level count; the ceiling =
// the candidate supply (edge content). Distinguishes "saturated because simple"
// from "saturated because small".
//
//   feat_count <shape|png> [size]     shape: star|blocks|arrow, or a PNG path

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <string>

static cv::Mat builtin(const std::string& s, int S) {
    cv::Mat m(S,S,CV_8U,cv::Scalar(40)); int c=S/2;
    if(s=="blocks"){cv::rectangle(m,{S/5,S/5},{S*7/16,S*13/16},220,cv::FILLED);
        cv::rectangle(m,{S/5,S*3/5},{S*13/16,S*13/16},220,cv::FILLED);}
    else if(s=="arrow"){cv::rectangle(m,{c-5,S/5},{c+5,S*5/6},220,cv::FILLED);
        std::vector<cv::Point> h{{c,S/10},{c-16,S/4},{c+16,S/4}};cv::fillConvexPoly(m,h,220);}
    else{std::vector<cv::Point> st;for(int i=0;i<10;i++){double a=CV_PI/2+i*CV_PI/5;double r=(i&1)?S*0.15:S*0.34;
        st.push_back({(int)(c+r*cos(a)),(int)(c-r*sin(a))});}cv::polylines(m,st,true,225,3);}
    return m;
}

int main(int argc,char**argv){
    sbm::setLogLevel(sbm::LogLevel::Error);
    std::string arg = argc>1?argv[1]:"star";
    int S = argc>2?std::atoi(argv[2]):160;
    cv::Mat t;
    if(arg.find('.')!=std::string::npos) t=cv::imread(arg,cv::IMREAD_GRAYSCALE);
    if(t.empty()) t=builtin(arg,S);

    std::printf("feat_count | %s | %dx%d\n", arg.c_str(), t.cols, t.rows);
    std::printf("  %-8s %-s\n","req nf","-> returned finest features (ceiling = candidate supply)");
    for(int nf : {32,64,128,256,512,1024}){
        auto fs = sbm::extractFeatures(t, cv::Mat(), nf);
        int n = fs.levels.empty()?0:(int)fs.levels[0].features.size();
        std::printf("  %-8d %d%s\n", nf, n, n<nf?"   (SATURATED — fewer than requested)":"");
    }
    return 0;
}
