// test_roi_spacing.cpp
// Sweep the ROI point min-spacing (MatchConfig.roi_min_spacing) to find the knee
// between reducing ROI window overlap and preserving solve conditioning. Reports,
// per shape: min pairwise distance among selected points (overlap proxy) and ROI
// origin error (clean + noise). Spacing too large excludes clustered discriminative
// points and wrecks conditioning (e.g. flag); too small leaves heavy overlap.

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

static float minPair(const std::vector<Point2f>& p){
    float m=1e9; for(size_t i=0;i<p.size();i++) for(size_t j=i+1;j<p.size();j++)
        m=std::min(m,(float)std::hypot(p[i].x-p[j].x,p[i].y-p[j].y));
    return p.size()<2?0:m;
}

// returns {min_pair, npts, o_mean_clean, o_worst_clean, o_mean_n30, o_worst_n30}
static void run(const Mat& templ, float spacing, float out[6]){
    int TW=templ.cols;
    sbm::MatchConfig cfg; cfg.min_score=40; cfg.nms_radius=60; cfg.refine=sbm::RefineMode::ROI;
    cfg.roi_min_spacing = spacing;
    // pre-extract + report selected points at this spacing
    auto feat = sbm::extractFeatures(templ); feat.setOrigin(TW/2.0f,TW/2.0f);
    auto pts = feat.selectOptimizedPoints(8, spacing);
    out[0]=minPair(pts); out[1]=(float)pts.size();
    sbm::ShapeMatcher m(cfg);
    sbm::ModelConfig mc; mc.angle={0,360,1}; mc.flip=false;
    m.addModel("s", feat, mc);
    const int W=400,H=400; cv::RNG rng(7);
    for(int pass=0; pass<2; pass++){
        int ns = pass==0?0:30; double so=0,wo=0; int n=0;
        for(int deg=0; deg<360; deg+=15){
            Mat clean(H,W,CV_8U,Scalar(0)); double X=200,Y=200; place(clean,templ,X,Y,deg);
            int reps=ns>0?3:1;
            for(int r=0;r<reps;r++){
                Mat sc=clean.clone(); if(ns>0){ Mat nz(H,W,CV_8U); rng.fill(nz,cv::RNG::NORMAL,0,ns); sc+=nz; }
                auto rs=m.match(sc); const sbm::MatchResult* b=nullptr; float bs=-1;
                for(auto& rr:rs){ float d=(float)std::hypot(rr.x-X,rr.y-Y); if(d<60&&rr.score>bs){bs=rr.score;b=&rr;} }
                if(!b) continue; double o=std::hypot(b->x-X,b->y-Y); so+=o; wo=std::max(wo,o); n++;
            }
        }
        out[2+pass*2]= n?so/n:-1; out[3+pass*2]=(float)wo;
    }
}

int main(){
    const int TW=140;
    struct S{ const char* nm; Mat im; };
    std::vector<S> shapes = {{"F",make_F(TW)},{"flag",make_flag(TW)},{"L",make_L(TW)},{"T",make_T(TW)}};
    float spacings[] = {0.0f, 6.0f, 8.0f, 10.0f, 12.0f, 15.0f};

    printf("ROI min-spacing sweep. minpair=closest selected pair(px); o=origin err mean|worst.\n\n");
    for(auto& s : shapes){
        printf("=== %s ===\n", s.nm);
        printf("%-9s | %-6s %-4s | clean o(mean|worst) | noise30 o(mean|worst)\n","spacing","minpr","npt");
        printf("-------------------------------------------------------------------------\n");
        for(float sp : spacings){
            float o[6]; run(s.im, sp, o);
            printf("%-9.0f | %5.1f  %3.0f | %6.2f | %-9.2f | %6.2f | %-9.2f\n",
                   sp, o[0], o[1], o[2], o[3], o[4], o[5]);
        }
        printf("\n");
    }
    return 0;
}
