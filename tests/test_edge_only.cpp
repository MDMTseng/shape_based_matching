// test_edge_only.cpp
// Validate the library edge-only ROI point selection (MatchConfig.roi_edge_only_points)
// with the ORIGINAL 2D matchTemplate refine unchanged. Compares default (corner+edge)
// vs edge-only (+12px spacing) on origin error, clean and under noise. Expectation:
// edge-only fixes the corner-clustering worst-case (e.g. L) at the same 2D speed.

#include "shape_matcher.h"
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>   // getRotationMatrix2D moved here in OpenCV 5
#endif
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
static Mat make_flag(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-6,6,-40,40);
    for(double y=-40;y<=-8;y+=0.4){ double f=(y+40)/32.0, xr=6+(40-2)*(1.0-f);
        for(double x=6;x<=xr;x+=0.4){ int px=(int)lround(c+x*s),py=(int)lround(c+y*s);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; } } return t; }
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

struct Acc { double wo=0,wa=0,so=0,sa=0; int n=0;
    void add(double o,double a){ wo=std::max(wo,o); wa=std::max(wa,a); so+=o; sa+=a; n++; } };

static Acc run(const Mat& templ, bool edge_only, int noise_sigma){
    int TW=templ.cols;
    auto feat=sbm::extractFeatures(templ); feat.setOrigin(TW/2.0f,TW/2.0f);
    sbm::MatchConfig cfg; cfg.min_score=40; cfg.nms_radius=60; cfg.refine=sbm::RefineMode::ROI;
    cfg.roi_edge_only_points = edge_only;
    cfg.roi_min_spacing = edge_only ? 12.0f : 0.0f;
    sbm::ShapeMatcher m(cfg); sbm::ModelConfig mc; mc.angle={0,360,1}; mc.flip=false;
    m.addModel("s",feat,mc);
    Acc acc; const int W=400,H=400; cv::RNG rng(7);
    for(int deg=0; deg<360; deg+=15){
        Mat clean(H,W,CV_8U,Scalar(0)); double X=200,Y=200; place(clean,templ,X,Y,deg);
        int reps=noise_sigma>0?3:1;
        for(int r=0;r<reps;r++){
            Mat sc=clean.clone(); if(noise_sigma>0){ Mat nz(H,W,CV_8U); rng.fill(nz,cv::RNG::NORMAL,0,noise_sigma); sc+=nz; }
            auto rs=m.match(sc); const sbm::MatchResult* b=nullptr; float bs=-1;
            for(auto& rr:rs){ float d=(float)std::hypot(rr.x-X,rr.y-Y); if(d<60&&rr.score>bs){bs=rr.score;b=&rr;} }
            if(!b) continue; acc.add(std::hypot(b->x-X,b->y-Y), std::fabs(angDiff(b->angle,(float)deg)));
        }
    }
    return acc;
}

int main(){
    const int TW=140;
    struct S{ const char* nm; Mat im; };
    std::vector<S> shapes = {{"F",make_F(TW)},{"flag",make_flag(TW)},{"L",make_L(TW)},{"T",make_T(TW)}};
    printf("Library edge-only ROI points (original 2D refine). origin err mean|worst (px).\n\n");
    printf("%-6s %-12s | clean (mean|worst) | noise30 (mean|worst)\n","shape","selection");
    printf("------------------------------------------------------------\n");
    for(auto& s : shapes){
        Acc d0=run(s.im,false,0), d3=run(s.im,false,30);
        Acc e0=run(s.im,true,0),  e3=run(s.im,true,30);
        printf("%-6s %-12s | %6.2f | %-9.2f | %6.2f | %-9.2f\n", s.nm,"default",
               d0.so/d0.n,d0.wo, d3.so/d3.n,d3.wo);
        printf("%-6s %-12s | %6.2f | %-9.2f | %6.2f | %-9.2f\n", "","edge-only",
               e0.so/e0.n,e0.wo, e3.so/e3.n,e3.wo);
    }
    return 0;
}
