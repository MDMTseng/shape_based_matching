// test_roi_reject.cpp
// Validate the per-point score-gate outlier rejection (roi_reject_low_score).
// Idea: at addModel each ROI point's "score floor" is the worst self-correlation
// under the coarse-error angle envelope; at match a point scoring below
// floor*pct matched the wrong place and is dropped. To stress "large initial
// angle error -> completely wrong match", use a COARSE angle grid (step 5deg, so
// coarse init is up to +-2.5deg off) plus noise, and compare baseline vs reject.

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

struct Acc { double wo=0,wa=0,so=0,sa=0; int n=0;
    void add(double o,double a){ wo=std::max(wo,o); wa=std::max(wa,a); so+=o; sa+=a; n++; } };

static Acc run(const Mat& templ, bool reject, int noise_sigma, float grid_step, float pct){
    int TW=templ.cols;
    auto feat = sbm::extractFeatures(templ);
    feat.setOrigin(TW/2.0f, TW/2.0f);
    sbm::MatchConfig cfg; cfg.min_score=40; cfg.nms_radius=60; cfg.refine=sbm::RefineMode::ROI;
    cfg.roi_reject_low_score = reject; cfg.roi_reject_angle_tol = grid_step; cfg.roi_reject_pct = pct;
    sbm::ShapeMatcher m(cfg);
    sbm::ModelConfig mc; mc.angle={0,360,grid_step}; mc.flip=false;   // COARSE grid -> larger init angle error
    m.addModel("s", feat, mc);
    Acc acc; const int W=400,H=400; cv::RNG rng(12345);
    for(double deg=2.3; deg<360; deg+=7.0){   // off-grid degs -> coarse init is mis-rotated
        Mat clean(H,W,CV_8U,Scalar(0)); double X=200,Y=200;
        place(clean, templ, X, Y, deg);
        int reps = noise_sigma>0?3:1;
        for(int rep=0; rep<reps; rep++){
            Mat scene=clean.clone();
            if(noise_sigma>0){ Mat n(H,W,CV_8U); rng.fill(n,cv::RNG::NORMAL,0,noise_sigma); scene+=n; }
            auto rs=m.match(scene);
            const sbm::MatchResult* b=nullptr; float bs=-1;
            for(auto& r:rs){ float d=(float)std::hypot(r.x-X,r.y-Y); if(d<60&&r.score>bs){bs=r.score;b=&r;} }
            if(!b) continue;
            acc.add(std::hypot(b->x-X,b->y-Y), std::fabs(angDiff(b->angle,(float)deg)));
        }
    }
    return acc;
}

int main(){
    const int TW=140;
    struct S{ const char* nm; Mat im; };
    std::vector<S> shapes = {{"F",make_F(TW)},{"flag",make_flag(TW)},{"L",make_L(TW)},{"T",make_T(TW)}};
    int noises[] = {0, 20, 30, 40};
    float grid = 5.0f;   // coarse angle grid -> init angle error up to +-2.5deg

    printf("Score-gate outlier rejection. COARSE %.0fdeg grid (large init angle err), off-grid poses.\n", grid);
    printf("origin err mean|worst (px): baseline vs reject(pct=0.8)\n\n");
    for(auto& s : shapes){
        printf("=== %s ===\n", s.nm);
        printf("%-6s | %-16s %-16s %-16s\n","sigma","baseline","reject0.8","reject0.95");
        printf("--------------------------------------------------------\n");
        for(int ns : noises){
            Acc a0=run(s.im,false,ns,grid,0.8f), a1=run(s.im,true,ns,grid,0.8f), a2=run(s.im,true,ns,grid,0.95f);
            printf("%-6d | %5.2f|%-9.2f %5.2f|%-9.2f %5.2f|%-9.2f\n", ns,
                   a0.so/a0.n,a0.wo, a1.so/a1.n,a1.wo, a2.so/a2.n,a2.wo);
        }
        printf("\n");
    }
    return 0;
}
