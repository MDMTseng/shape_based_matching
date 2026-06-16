// test_flag_viz.cpp
// Visualize the "flag" template that exhibits the rotation-angle score collapse
// (see test_flip_separate.cpp). Draws, on the template image:
//   - all refine_points (gray dots)
//   - level-0 match features with their gradient orientation (theta) as a short line
//   - selected optimized ROI points (boxes + normals)
//   - the template centre (origin) cross
// Also renders the template rotated to 25 deg (a collapse angle) to compare what
// the matcher's quantized features look like there.

#include "shape_matcher.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cstdio>
#include <cmath>

using namespace cv;
using namespace sbm;

static Mat make_flag(int TW){
    Mat t(TW,TW,CV_8U,Scalar(0)); double cx=TW/2.0, cy=TW/2.0, s=TW/120.0;
    auto box=[&](double x0,double x1,double y0,double y1){
        for(double y=y0;y<=y1;y+=0.4) for(double x=x0;x<=x1;x+=0.4){
            int px=(int)lround(cx+x*s), py=(int)lround(cy+y*s);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; } };
    box(-6,6,-40,40);
    for(double y=-40;y<=-8;y+=0.4){
        double frac=(y+40)/32.0;
        double xr=6 + (40-2)*(1.0-frac);
        for(double x=6;x<=xr;x+=0.4){
            int px=(int)lround(cx+x*s), py=(int)lround(cy+y*s);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; }
    }
    return t;
}

static Mat make_F(int TW){
    Mat t(TW,TW,CV_8U,Scalar(0)); double cx=TW/2.0, cy=TW/2.0, s=TW/120.0;
    auto box=[&](double x0,double x1,double y0,double y1){
        for(double y=y0;y<=y1;y+=0.4) for(double x=x0;x<=x1;x+=0.4){
            int px=(int)lround(cx+x*s), py=(int)lround(cy+y*s);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; } };
    box(-10,-2,-34,34); box(-10,26,-34,-26); box(-10,16,-6,2);
    return t;
}

// Draw features (level 0) with orientation, refine points, ROI boxes, centre.
static Mat draw(const Mat& templ, const char* title){
    auto fs = extractFeatures(templ);
    auto opt = fs.selectOptimizedPoints(8);
    float tcx = templ.cols/2.0f, tcy = templ.rows/2.0f;

    Mat vis; cvtColor(templ, vis, COLOR_GRAY2BGR);

    // refine points (dense) - gray
    for (auto& rp : fs.refine_points){
        int px=(int)(rp.px+tcx+0.5f), py=(int)(rp.py+tcy+0.5f);
        circle(vis, Point(px,py), 1, Scalar(90,90,90), -1);
    }

    // level-0 features with gradient orientation line.
    // Feature coords are relative to the level bounding box (tl_x,tl_y).
    if (!fs.levels.empty()){
        auto& lv = fs.levels[0];
        for (auto& f : lv.features){
            int px = lv.tl_x + f.x, py = lv.tl_y + f.y;
            // theta in degrees; draw a short line along the gradient direction
            float th = f.theta * (float)CV_PI/180.0f;
            int dx=(int)lround(5*std::cos(th)), dy=(int)lround(5*std::sin(th));
            line(vis, Point(px-dx,py-dy), Point(px+dx,py+dy), Scalar(255,180,0), 1);
            circle(vis, Point(px,py), 1, Scalar(0,0,255), -1);
        }
    }

    // optimized ROI points - boxes
    int roi_half=10;
    for (auto& pt : opt){
        int px=(int)(pt.x+tcx+0.5f), py=(int)(pt.y+tcy+0.5f);
        rectangle(vis, Rect(px-roi_half,py-roi_half,2*roi_half+1,2*roi_half+1), Scalar(0,255,0), 1);
        circle(vis, Point(px,py), 2, Scalar(0,200,255), -1);
    }

    // centre / origin cross
    drawMarker(vis, Point((int)tcx,(int)tcy), Scalar(255,0,255), MARKER_CROSS, 16, 2);

    char buf[128];
    snprintf(buf,sizeof(buf),"%s: %d feat (L0), %d refine pts, %d ROI",
             title, fs.numFeatures(), (int)fs.refine_points.size(), (int)opt.size());
    Mat vis_big; resize(vis, vis_big, Size(), 2,2, INTER_NEAREST);
    putText(vis_big, buf, Point(6,18), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255,255,255), 1);
    return vis_big;
}

int main(){
    const int TW=240;
    Mat flag = make_flag(TW);
    Mat F    = make_F(TW);

    // flag rotated to 25 deg (a score-collapse angle) - extract on the rotated raster
    Mat flag25; {
        Mat R = getRotationMatrix2D(Point2f(TW/2.0f,TW/2.0f), -25.0, 1.0);
        warpAffine(flag, flag25, R, flag.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    }

    Mat a = draw(flag,   "flag @0");
    Mat b = draw(flag25, "flag @25 (collapse)");
    Mat c = draw(F,      "F @0 (stable)");

    imwrite("output/flag_template.png",  a);
    imwrite("output/flag25_template.png", b);
    imwrite("output/F_template.png",      c);
    printf("Saved output/flag_template.png, output/flag25_template.png, output/F_template.png\n");
    return 0;
}
