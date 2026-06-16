// test_roi_iters.cpp
// Does ROI iteration need to RE-MATCH, or just re-solve? Compare, with edge-only points:
//   - solve iters 1 / 2 / 3 / 5 (match ONCE, Gauss-Newton re-solve only)
//   - + iterative_rematch (re-match every iteration, ICP-style)
// If solve-iters>1 helps, the Gauss-Newton iteration matters. If iterative_rematch adds
// little over solve-iters, re-matching is unnecessary for normal init error.

#include "shape_matcher.h"
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace cv;
static float angDiff(float a, float b){ float d=a-b; while(d>180)d-=360; while(d<-180)d+=360; return d; }

static void boxInto(Mat& t,double cx,double cy,double s,double x0,double x1,double y0,double y1){
    for(double y=y0;y<=y1;y+=0.4) for(double x=x0;x<=x1;x+=0.4){
        int px=(int)lround(cx+x*s), py=(int)lround(cy+y*s);
        if(px>=0&&px<t.cols&&py>=0&&py<t.rows) t.at<uchar>(py,px)=200; }
}
static Mat make_F(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-10,-2,-34,34); boxInto(t,c,c,s,-10,26,-34,-26); boxInto(t,c,c,s,-10,16,-6,2); return t; }
static Mat make_L(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-30,-18,-34,34); boxInto(t,c,c,s,-30,30,22,34); return t; }
static Mat make_T(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-34,34,-34,-22); boxInto(t,c,c,s,-6,6,-34,34); return t; }

static void place(Mat& scene, const Mat& templ, double X, double Y, double deg){
    double cx=templ.cols/2.0, cy=templ.rows/2.0;
    Mat R = getRotationMatrix2D(Point2f((float)cx,(float)cy), -deg, 1.0);
    R.at<double>(0,2) += X - cx;  R.at<double>(1,2) += Y - cy;
    Mat warped; warpAffine(templ, warped, R, scene.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    scene = max(scene, warped);
}
struct Acc { double wo=0,so=0; int n=0; void add(double o){ wo=std::max(wo,o); so+=o; n++; } };

static Acc run(const Mat& templ, int iters, bool rematch, int noise){
    int TW=templ.cols;
    auto feat=sbm::extractFeatures(templ); feat.setOrigin(TW/2.0f,TW/2.0f);
    sbm::MatchConfig cfg; cfg.min_score=40; cfg.nms_radius=60; cfg.refine=sbm::RefineMode::ROI;
    cfg.roi_edge_only_points=true; cfg.roi_min_spacing=12.0f;
    cfg.roi_max_iters=iters; cfg.roi_iterative_rematch=rematch;
    sbm::ShapeMatcher m(cfg); sbm::ModelConfig mc; mc.angle={0,360,1}; mc.flip=false; m.addModel("s",feat,mc);
    Acc a; const int W=400,H=400; cv::RNG rng(7);
    for(int deg=0; deg<360; deg+=15){
        Mat clean(H,W,CV_8U,Scalar(0)); double X=200,Y=200; place(clean,templ,X,Y,deg);
        int reps=noise>0?3:1;
        for(int r=0;r<reps;r++){
            Mat sc=clean.clone(); if(noise>0){ Mat nz(H,W,CV_8U); rng.fill(nz,cv::RNG::NORMAL,0,noise); sc+=nz; }
            auto rs=m.match(sc); const sbm::MatchResult* b=nullptr; float bs=-1;
            for(auto& rr:rs){ float d=(float)std::hypot(rr.x-X,rr.y-Y); if(d<60&&rr.score>bs){bs=rr.score;b=&rr;} }
            if(b) a.add(std::hypot(b->x-X,b->y-Y));
        }
    }
    return a;
}

int main(){
    const int TW=140;
    struct S{ const char* nm; Mat im; };
    std::vector<S> shapes = {{"F",make_F(TW)},{"L",make_L(TW)},{"T",make_T(TW)}};
    printf("Edge-only. origin err mean|worst (px). Does iteration need re-matching?\n\n");
    printf("%-5s | %-13s %-13s %-13s %-13s | %-15s\n","shape","solve x1","solve x2","solve x3","solve x5","x3 + rematch");
    printf("----------------------------------------------------------------------------------------\n");
    for(int noise : {0, 30}){
      printf("[noise=%d]\n", noise);
      for(auto& s : shapes){
        Acc a1=run(s.im,1,false,noise), a2=run(s.im,2,false,noise), a3=run(s.im,3,false,noise),
            a5=run(s.im,5,false,noise), ar=run(s.im,3,true,noise);
        printf("%-5s | %5.2f|%-7.2f %5.2f|%-7.2f %5.2f|%-7.2f %5.2f|%-7.2f | %5.2f|%-7.2f\n", s.nm,
               a1.so/a1.n,a1.wo, a2.so/a2.n,a2.wo, a3.so/a3.n,a3.wo, a5.so/a5.n,a5.wo, ar.so/ar.n,ar.wo);
      }
    }
    return 0;
}
