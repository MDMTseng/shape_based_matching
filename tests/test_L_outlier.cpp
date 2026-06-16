// test_L_outlier.cpp
// Chase the L-shape ROI outlier (multi-template showed L=1.92px at deg=70 while the
// coarse step-15 sweep showed L worst ~0.48px). Sweep L ALONE at 1-deg resolution,
// report ROI vs coarse(None) origin error, and flag spikes. Then we know whether the
// spike is angle-specific (a fragile pose) or scene-context specific (multi-template).

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
static Mat make_L(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-30,-18,-34,34); boxInto(t,c,c,s,-30,30,22,34); return t; }

static void place(Mat& scene, const Mat& templ, double X, double Y, double deg){
    double cx=templ.cols/2.0, cy=templ.rows/2.0;
    Mat R = getRotationMatrix2D(Point2f((float)cx,(float)cy), -deg, 1.0);
    R.at<double>(0,2) += X - cx;  R.at<double>(1,2) += Y - cy;
    Mat warped; warpAffine(templ, warped, R, scene.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    scene = max(scene, warped);
}

struct Hit{ bool ok; float o,a,score; };
static Hit best(sbm::ShapeMatcher& m, const Mat& scene, double X,double Y,double deg){
    auto rs=m.match(scene); const sbm::MatchResult* b=nullptr; float bs=-1;
    for(auto& r:rs){ float d=(float)std::hypot(r.x-X,r.y-Y); if(d<70&&r.score>bs){bs=r.score;b=&r;} }
    if(!b) return {false,0,0,0};
    return {true,(float)std::hypot(b->x-X,b->y-Y),std::fabs(angDiff(b->angle,(float)deg)),b->score};
}

int main(){
    const int TW=140; Mat L=make_L(TW);
    auto feat=sbm::extractFeatures(L); feat.setOrigin(TW/2.0f,TW/2.0f);
    sbm::ModelConfig mc; mc.angle={0,360,1}; mc.flip=false;

    sbm::MatchConfig c_roi; c_roi.min_score=50; c_roi.nms_radius=60; c_roi.refine=sbm::RefineMode::ROI;
    sbm::MatchConfig c_non=c_roi; c_non.refine=sbm::RefineMode::None;
    sbm::MatchConfig c_sub=c_roi; c_sub.refine=sbm::RefineMode::ICP_Subpixel;
    sbm::ShapeMatcher mroi(c_roi), mnon(c_non), msub(c_sub);
    mroi.addModel("L",feat,mc); mnon.addModel("L",feat,mc); msub.addModel("L",feat,mc);

    printf("L alone, 1-deg sweep. Spikes (ROI o>0.8px) flagged.\n");
    printf("deg | coarse o,a | ROI o,a | ICP_Sub o,a\n");
    printf("----+------------+----------+------------\n");
    float worst_roi=0; int worst_deg=0;
    for(int deg=0; deg<180; deg++){
        const int W=400,H=400; double X=200,Y=200;
        Mat scene(H,W,CV_8U,Scalar(0)); place(scene,L,X,Y,deg);
        Hit n=best(mnon,scene,X,Y,deg), r=best(mroi,scene,X,Y,deg), s=best(msub,scene,X,Y,deg);
        if(r.ok && r.o>worst_roi){ worst_roi=r.o; worst_deg=deg; }
        if(r.ok && r.o>0.8)
            printf("%3d | %5.2f %4.1f | %5.2f %4.1f | %5.2f %4.1f   <-- SPIKE\n",
                   deg, n.o,n.a, r.o,r.a, s.o,s.a);
    }
    printf("\nworst ROI origin over 0..179: %.2fpx @deg=%d\n", worst_roi, worst_deg);
    return 0;
}
