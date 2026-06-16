// test_roi_confidence.cpp
// Validate MatchResult.refine_residual as a per-result confidence signal for
// detecting completely-off / unreliable ROI matches. A trustworthy match has the
// sample points agreeing on one pose (residual ~0); occlusion / partial / gross
// mismatch makes points disagree (residual large). Show the residual separates
// good matches from bad ones so a threshold can flag "off" matches.

#include "shape_matcher.h"
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace cv;

static void boxInto(Mat& t,double cx,double cy,double s,double x0,double x1,double y0,double y1){
    for(double y=y0;y<=y1;y+=0.4) for(double x=x0;x<=x1;x+=0.4){
        int px=(int)lround(cx+x*s), py=(int)lround(cy+y*s);
        if(px>=0&&px<t.cols&&py>=0&&py<t.rows) t.at<uchar>(py,px)=200; }
}
static Mat make_flag(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-6,6,-40,40);
    for(double y=-40;y<=-8;y+=0.4){ double f=(y+40)/32.0, xr=6+(40-2)*(1.0-f);
        for(double x=6;x<=xr;x+=0.4){ int px=(int)lround(c+x*s),py=(int)lround(c+y*s);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; } } return t; }

static void place(Mat& scene, const Mat& templ, double X, double Y, double deg){
    double cx=templ.cols/2.0, cy=templ.rows/2.0;
    Mat R = getRotationMatrix2D(Point2f((float)cx,(float)cy), -deg, 1.0);
    R.at<double>(0,2) += X - cx;  R.at<double>(1,2) += Y - cy;
    Mat warped; warpAffine(templ, warped, R, scene.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    scene = max(scene, warped);
}

int main(){
    const int TW=140; Mat flag=make_flag(TW);
    auto feat=sbm::extractFeatures(flag); feat.setOrigin(TW/2.0f,TW/2.0f);
    sbm::MatchConfig cfg; cfg.min_score=40; cfg.nms_radius=60; cfg.refine=sbm::RefineMode::ROI;
    sbm::ShapeMatcher m(cfg);
    sbm::ModelConfig mc; mc.angle={0,360,1}; mc.flip=false;
    m.addModel("flag", feat, mc);

    const int W=400,H=400; double X=200,Y=200;
    auto matchResidual=[&](const Mat& scene)->std::pair<float,float>{ // {residual, origin_err}
        auto rs=m.match(scene);
        const sbm::MatchResult* b=nullptr; float bs=-1;
        for(auto& r:rs){ float d=(float)std::hypot(r.x-X,r.y-Y); if(d<90&&r.score>bs){bs=r.score;b=&r;} }
        if(!b) return {-1,-1};
        return {b->refine_residual, (float)std::hypot(b->x-X,b->y-Y)};
    };

    printf("MatchResult.refine_residual as off-match detector (flag, ROI)\n\n");
    printf("%-26s | residual(px) | origin-err(px)\n","case");
    printf("------------------------------------------------------------\n");

    // GOOD: clean
    { Mat s(H,W,CV_8U,Scalar(0)); place(s,flag,X,Y,33); auto r=matchResidual(s);
      printf("%-26s | %10.2f   | %.2f\n","good clean", r.first, r.second); }
    // GOOD: noise 30
    { Mat s(H,W,CV_8U,Scalar(0)); place(s,flag,X,Y,33); Mat n(H,W,CV_8U); RNG(1).fill(n,RNG::NORMAL,0,30); s+=n;
      auto r=matchResidual(s); printf("%-26s | %10.2f   | %.2f\n","good + noise30", r.first, r.second); }
    // OFF: occlude right half (paint black box over the flag triangle)
    { Mat s(H,W,CV_8U,Scalar(0)); place(s,flag,X,Y,33);
      rectangle(s, Rect((int)X, (int)Y-80, 120, 100), Scalar(0), FILLED);
      auto r=matchResidual(s); printf("%-26s | %10.2f   | %.2f\n","occluded (triangle gone)", r.first, r.second); }
    // OFF: occlude lower pole
    { Mat s(H,W,CV_8U,Scalar(0)); place(s,flag,X,Y,33);
      rectangle(s, Rect((int)X-60, (int)Y, 120, 90), Scalar(0), FILLED);
      auto r=matchResidual(s); printf("%-26s | %10.2f   | %.2f\n","occluded (lower half)", r.first, r.second); }
    // OFF: heavy clutter lines crossing the shape
    { Mat s(H,W,CV_8U,Scalar(0)); place(s,flag,X,Y,33);
      for(int i=0;i<6;i++) line(s, Point(120,150+i*12), Point(300,160+i*12), Scalar(200), 3);
      auto r=matchResidual(s); printf("%-26s | %10.2f   | %.2f\n","heavy clutter", r.first, r.second); }
    // OFF: extreme noise 60
    { Mat s(H,W,CV_8U,Scalar(0)); place(s,flag,X,Y,33); Mat n(H,W,CV_8U); RNG(2).fill(n,RNG::NORMAL,0,60); s+=n;
      auto r=matchResidual(s); printf("%-26s | %10.2f   | %.2f\n","extreme noise60", r.first, r.second); }

    printf("\nIf residual is low for the good rows and clearly higher for the\n");
    printf("occluded/clutter rows, a residual threshold flags off matches.\n");
    return 0;
}
