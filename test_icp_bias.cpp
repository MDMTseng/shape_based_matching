// ICP bias per template shape: is the +0.4 deg / +0.4px bias shape-dependent?

#include "line2Dup.h"
#include "icp_refine.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
using namespace cv;
using namespace std;

static Mat pad16(const Mat& img) {
    int pw=(img.cols+15)&~15, ph=(img.rows+15)&~15;
    if(pw!=img.cols||ph!=img.rows){Mat p;copyMakeBorder(img,p,0,ph-img.rows,0,pw-img.cols,BORDER_CONSTANT,Scalar(0));return p;}
    return img;
}

static Mat make_L(int TW) {
    Mat img(TW,TW,CV_8U,Scalar(0));
    rectangle(img, Point(TW/2-15,TW/2-30), Point(TW/2-5,TW/2+30), Scalar(200), -1);
    rectangle(img, Point(TW/2-15,TW/2+20), Point(TW/2+25,TW/2+30), Scalar(200), -1);
    return img;
}
static Mat make_T(int TW) {
    Mat img(TW,TW,CV_8U,Scalar(0));
    rectangle(img, Point(TW/2-5,TW/2-25), Point(TW/2+5,TW/2+25), Scalar(200), -1);
    rectangle(img, Point(TW/2-25,TW/2-25), Point(TW/2+25,TW/2-15), Scalar(200), -1);
    return img;
}
static Mat make_arrow(int TW) {
    Mat img(TW,TW,CV_8U,Scalar(0));
    Point pts[3] = {Point(TW/2-20,TW/2+15), Point(TW/2-20,TW/2-15), Point(TW/2+20,TW/2)};
    fillConvexPoly(img, pts, 3, Scalar(200));
    return img;
}
static Mat make_wrench(int TW) {
    Mat img(TW,TW,CV_8U,Scalar(0));
    rectangle(img, Point(TW/2-30,TW/2-4), Point(TW/2+5,TW/2+4), Scalar(200), -1);
    circle(img, Point(TW/2+12,TW/2), 12, Scalar(200), -1);
    return img;
}

static void placeInScene(Mat& scene, const Mat& templ, int cx, int cy, double angle) {
    int TW = templ.cols;
    Mat M = getRotationMatrix2D(Point2f(TW/2.0f,TW/2.0f), -angle, 1.0);
    Mat rotated;
    warpAffine(templ, rotated, M, Size(TW,TW), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    int ox=cx-TW/2, oy=cy-TW/2;
    for(int r=0;r<TW;r++){int sy=oy+r;if(sy<0||sy>=scene.rows)continue;
        for(int c=0;c<TW;c++){int sx=ox+c;if(sx<0||sx>=scene.cols)continue;
            uchar v=rotated.at<uchar>(r,c);if(v>0)scene.at<uchar>(sy,sx)=v;}}
}

int main() {
    const int TW = 80, W = 640, H = 480;
    const int cx = W/2, cy = H/2;
    const float step = 2.0f, threshold = 50.0f;

    struct Shape { const char* name; Mat templ; };
    Shape shapes[] = {
        {"L-shape", make_L(TW)},
        {"T-shape", make_T(TW)},
        {"Arrow",   make_arrow(TW)},
        {"Wrench",  make_wrench(TW)},
    };

    fprintf(stderr, "%-10s  %6s  %12s  %16s\n", "Shape", "Feats", "ICP ang bias", "ICP pos bias");
    fprintf(stderr, "%-10s  %6s  %12s  %16s\n", "-----", "-----", "------------", "------------");

    for (auto& shape : shapes) {
        Mat mask = Mat::zeros(TW,TW,CV_8U);
        for(int r=0;r<TW;r++) for(int c=0;c<TW;c++)
            if(shape.templ.at<uchar>(r,c)>0) mask.at<uchar>(r,c)=255;
        dilate(mask, mask, Mat(), Point(-1,-1), 5);

        // Train
        line2Dup::Detector det(128,{4,8},30,60);
        for(int a=0;a<360;a+=(int)step){
            Mat rt,rm;
            Mat M=getRotationMatrix2D(Point2f(TW/2.0f,TW/2.0f),-(double)a,1.0);
            warpAffine(shape.templ,rt,M,Size(TW,TW));
            warpAffine(mask,rm,M,Size(TW,TW));
            det.addTemplate(rt,"S",rm);
        }

        // ICP model edges
        auto model_edges = icp_refine::extractModelEdges(shape.templ);
        int nfeats = (int)model_edges.size();

        // Sweep angles, measure signed ICP error
        float sum_ang=0, sum_dx=0, sum_dy=0;
        int n=0;

        for(int gt=0; gt<360; gt+=3) {
            Mat scene(H,W,CV_8U,Scalar(50));
            placeInScene(scene, shape.templ, cx, cy, gt);

            auto matches = det.match(pad16(scene), threshold);
            if(matches.empty()) continue;

            auto& m = matches[0];
            auto& ti = det.getTemplates(m.class_id, m.template_id);
            float mcx = m.x + TW/2.0f - ti[0].tl_x;
            float mcy = m.y + TW/2.0f - ti[0].tl_y;
            float coarse_angle = m.template_id * step;

            Mat ss, sdx, sdy;
            GaussianBlur(scene, ss, Size(7,7), 0);
            Sobel(ss, sdx, CV_16S, 1, 0, 3);
            Sobel(ss, sdy, CV_16S, 0, 1, 3);

            icp_refine::ICPConfig cfg;
            cfg.max_iterations = 30;
            cfg.max_dist = 10.0f;

            icp_refine::Pose2D init(mcx, mcy, coarse_angle);
            auto ref = icp_refine::refineWithNormals(
                model_edges, sdx, sdy, init, TW, 20, cfg);

            float ae = ref.angle - gt;
            if(ae>180)ae-=360; if(ae<-180)ae+=360;
            sum_ang += ae;
            sum_dx += ref.x - cx;
            sum_dy += ref.y - cy;
            n++;
        }

        fprintf(stderr, "%-10s  %5d   %+.2f deg    (%+.2f, %+.2f) px\n",
                shape.name, nfeats, sum_ang/n, sum_dx/n, sum_dy/n);
    }

    return 0;
}
