// test_roi_largescene.cpp
// Does ROI refine cost grow with SCENE size (5MP)? The refine only touches small
// regions around each match, but matchROI_subpixel reads the scene search window as
// a STRIDED view into the (large) scene -> rows far apart -> TLB / scattered loads.
// Measure refine cost (match - coarse) for a small vs a 5MP scene, same single L.

#include "shape_matcher.h"
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>   // getRotationMatrix2D moved here in OpenCV 5
#endif
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

using namespace cv;
using clk = std::chrono::high_resolution_clock;

static void boxInto(Mat& t,double cx,double cy,double s,double x0,double x1,double y0,double y1){
    for(double y=y0;y<=y1;y+=0.4) for(double x=x0;x<=x1;x+=0.4){
        int px=(int)lround(cx+x*s), py=(int)lround(cy+y*s);
        if(px>=0&&px<t.cols&&py>=0&&py<t.rows) t.at<uchar>(py,px)=200; }
}
static Mat make_L(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-30,-18,-34,34); boxInto(t,c,c,s,-30,30,22,34); return t; }

static void place(Mat& scene, const Mat& templ, double X, double Y, double deg){
    double cx=templ.cols/2.0, cy=templ.rows/2.0;
    Mat R = getRotationMatrix2D(Point2f((float)cx,(float)cy), -deg, 1.0);
    R.at<double>(0,2) += X - cx;  R.at<double>(1,2) += Y - cy;
    Mat w; warpAffine(templ, w, R, scene.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0)); scene=max(scene,w);
}

// Returns mean ms/match over reps, for a scene of size WxH with the L near its centre.
static int g_last_nmatch=0;
static double timeScene(int W, int H, bool edge_only, sbm::RefineMode rm, int reps, int max_results=0, float min_score=40){
    const int TW=140; Mat L=make_L(TW);
    auto feat=sbm::extractFeatures(L); feat.setOrigin(TW/2.0f,TW/2.0f);
    sbm::MatchConfig cfg; cfg.min_score=min_score; cfg.nms_radius=60; cfg.refine=rm; cfg.max_results=max_results;
    if(edge_only){ cfg.roi_edge_only_points=true; cfg.roi_min_spacing=12.0f; }
    sbm::ShapeMatcher m(cfg); sbm::ModelConfig mc; mc.angle={0,360,1}; mc.flip=false; m.addModel("L",feat,mc);
    Mat scene(H,W,CV_8U,Scalar(0)); place(scene,L,W/2.0,H/2.0,33);
    auto r0=m.match(scene); g_last_nmatch=(int)r0.size();
    auto t0=clk::now(); for(int i=0;i<reps;i++){ volatile auto r=m.match(scene); (void)r; } auto t1=clk::now();
    return std::chrono::duration<double,std::milli>(t1-t0).count()/reps;
}

int main(){
    int reps=60;
    struct Sz{ const char* nm; int W,H; };
    std::vector<Sz> sizes = { {"400x400 (0.16MP)",400,400}, {"2500x2000 (5MP)",2500,2000} };
    printf("ROI refine cost vs scene size (single L). refine ms = ROI - None. %d reps.\n\n", reps);
    printf("%-20s %8s %8s %10s %8s %12s\n","scene","#match","None","ROI","refine","refine/match");
    printf("---------------------------------------------------------------------------\n");
    for(auto& s : sizes){
        double none = timeScene(s.W,s.H,false,sbm::RefineMode::None,reps);
        double roi  = timeScene(s.W,s.H,false,sbm::RefineMode::ROI, reps); int nm=g_last_nmatch;
        printf("%-20s %8d %8.2f %10.2f %8.2f %12.3f\n", s.nm, nm, none, roi, roi-none, nm>0?(roi-none)/nm:0);
    }
    printf("\n-- max_results=1 (refine only the top match): --\n");
    for(auto& s : sizes){
        double none = timeScene(s.W,s.H,false,sbm::RefineMode::None,reps,1);
        double roi  = timeScene(s.W,s.H,false,sbm::RefineMode::ROI, reps,1); int nm=g_last_nmatch;
        printf("%-20s %8d %8.2f %10.2f %8.2f %12.3f\n", s.nm, nm, none, roi, roi-none, nm>0?(roi-none)/nm:0);
    }
    printf("\n=> refine total scales with #matches; per-match refine is ~constant.\n");
    return 0;
}
