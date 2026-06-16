// test_roi_overlap.cpp
// Quantify ROI sample-point overlap from auto selection (selectOptimizedPoints).
// ROI window half-size = 15 (kDefaultROIHalf) -> a 30px window. Two points closer
// than 30px have overlapping windows; closer than 15px overlap heavily. The grid
// distribution (5x5, 2/cell) does NOT enforce a pairwise min distance, so points
// can cluster. Report min pairwise distance + overlap counts per shape.

#include "shape_matcher.h"
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace cv;

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

static void report(const char* nm, const Mat& im){
    auto feat = sbm::extractFeatures(im);
    auto pts = feat.selectOptimizedPoints(8);
    int n = (int)pts.size();
    float mind = 1e9; int n_overlap=0, n_heavy=0; double sum_min_each=0;
    for(int i=0;i<n;i++){
        float mi=1e9;
        for(int j=0;j<n;j++) if(i!=j){
            float d=(float)std::hypot(pts[i].x-pts[j].x, pts[i].y-pts[j].y);
            mi=std::min(mi,d);
            if(j>i){ if(d<30) n_overlap++; if(d<15) n_heavy++; mind=std::min(mind,d); }
        }
        sum_min_each += mi;
    }
    printf("%-6s | pts=%2d | min-pair=%5.1f | mean-nn=%5.1f | pairs<30px(overlap)=%2d | pairs<15px(heavy)=%2d\n",
           nm, n, mind, n>0?sum_min_each/n:0, n_overlap, n_heavy);
}

int main(){
    const int TW=140;
    printf("ROI window half=15 (30px window). Overlap if pair<30px, heavy if <15px.\n\n");
    report("F", make_F(TW));
    report("flag", make_flag(TW));
    report("L", make_L(TW));
    report("T", make_T(TW));
    return 0;
}
