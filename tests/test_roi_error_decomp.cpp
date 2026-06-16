// test_roi_error_decomp.cpp
// Decompose ROI refine ORIGIN error into its sources:
//   (1) coarse-angle quantization: ROI matches at the coarse (1-deg-stepped) angle,
//       warping each template patch to that angle. If the true pose sits between
//       template angles, the patch is mis-rotated -> match offset. Tested by placing
//       instances at ALIGNED angles (integer, coarse-exact) vs MISALIGNED (integer+0.5).
//   (2) per-point subpixel template-match floor: TM_CCORR_NORMED + parabolic peak.
//       Whatever remains at ALIGNED angles with no anti-aliasing is this floor.
//   (3) scene anti-aliasing: INTER_LINEAR (AA) vs INTER_NEAREST (crisp).
// 2x2: {aligned, misaligned} x {LINEAR, NEAREST}. Origin error px (worst | mean).

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
static Mat make_flag(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-6,6,-40,40);
    for(double y=-40;y<=-8;y+=0.4){ double f=(y+40)/32.0, xr=6+(40-2)*(1.0-f);
        for(double x=6;x<=xr;x+=0.4){ int px=(int)lround(c+x*s),py=(int)lround(c+y*s);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; } } return t; }

static void place(Mat& scene, const Mat& templ, double X, double Y, double deg, int interp){
    double cx=templ.cols/2.0, cy=templ.rows/2.0;
    Mat R = getRotationMatrix2D(Point2f((float)cx,(float)cy), -deg, 1.0);
    R.at<double>(0,2) += X - cx;  R.at<double>(1,2) += Y - cy;
    Mat warped; warpAffine(templ, warped, R, scene.size(), interp, BORDER_CONSTANT, Scalar(0));
    scene = max(scene, warped);
}

struct Acc { double wo=0,wa=0,so=0,sa=0; int n=0;
    void add(double o,double a){ wo=std::max(wo,o); wa=std::max(wa,a); so+=o; sa+=a; n++; } };

// Sweep base angles; add `off` (0.0 = aligned to 1-deg grid, 0.5 = worst misalignment).
static Acc run(const Mat& templ, double off, int interp){
    int TW=templ.cols;
    auto feat = sbm::extractFeatures(templ);
    feat.setOrigin(TW/2.0f, TW/2.0f); feat.setAngleOffset(0);
    sbm::MatchConfig cfg; cfg.min_score=50; cfg.nms_radius=60; cfg.refine=sbm::RefineMode::ROI;
    sbm::ShapeMatcher m(cfg);
    sbm::ModelConfig mc; mc.angle={0,360,1}; mc.flip=false;   // 1-deg template grid (default)
    m.addModel("s", feat, mc);
    Acc acc;
    const int W=400,H=400; double X=200, Y=200;
    for(int b=0; b<90; b+=6){
        double deg = b + off;
        Mat scene(H,W,CV_8U,Scalar(0));
        place(scene, templ, X, Y, deg, interp);
        auto rs = m.match(scene);
        const sbm::MatchResult* best=nullptr; float bs=-1;
        for(auto& r: rs){ float d=(float)std::hypot(r.x-X,r.y-Y); if(d<60 && r.score>bs){bs=r.score;best=&r;} }
        if(!best) continue;
        acc.add(std::hypot(best->x-X,best->y-Y), std::fabs(angDiff(best->angle,(float)deg)));
    }
    return acc;
}

int main(){
    const int TW=140;
    Mat flag = make_flag(TW);
    printf("ROI error decomposition (flag, TW=%d, 1-deg template grid, 8 pts, roi_half=15)\n", TW);
    printf("origin px (worst|mean), angle deg (worst|mean), over base angles 0..84 step6\n\n");

    struct Cond{ const char* nm; double off; int interp; };
    Cond conds[] = {
        {"aligned   + NEAREST(crisp)", 0.0, INTER_NEAREST},
        {"aligned   + LINEAR (AA)   ", 0.0, INTER_LINEAR},
        {"misalign  + NEAREST(crisp)", 0.5, INTER_NEAREST},
        {"misalign  + LINEAR (AA)   ", 0.5, INTER_LINEAR},
    };
    printf("%-28s | o-worst o-mean | a-worst a-mean\n","condition");
    printf("-------------------------------------------------------------\n");
    for(auto& c : conds){
        Acc a = run(flag, c.off, c.interp);
        printf("%-28s | %6.2f %6.2f | %6.2f %6.2f\n",
               c.nm, a.wo, a.so/a.n, a.wa, a.sa/a.n);
    }
    printf("\nReading:\n");
    printf("  aligned+NEAREST   = pure subpixel template-match floor (no AA, no angle-quant)\n");
    printf("  aligned->misalign = extra error from 1-deg coarse-angle quantization\n");
    printf("  NEAREST->LINEAR   = extra error from scene anti-aliasing\n");
    return 0;
}
