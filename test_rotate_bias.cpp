#include "line2Dup.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
using namespace cv;
static void draw_L(Mat& img, int cx, int cy, double angle, int color, double scale = 2.0) {
    double rad = angle * CV_PI / 180.0;
    double cs = cos(rad), sn = sin(rad);
    for (double ly = -15*scale; ly <= 15*scale; ly += 0.5)
        for (double lx = -5*scale; lx <= 5*scale; lx += 0.5) {
            int px = cx + (int)(lx*cs - ly*sn + 0.5); int py = cy + (int)(lx*sn + ly*cs + 0.5);
            if (px >= 0 && px < img.cols && py >= 0 && py < img.rows) img.at<uchar>(py, px) = (uchar)color;
        }
    for (double ly = 5*scale; ly <= 15*scale; ly += 0.5)
        for (double lx = 5*scale; lx <= 20*scale; lx += 0.5) {
            int px = cx + (int)(lx*cs - ly*sn + 0.5); int py = cy + (int)(lx*sn + ly*cs + 0.5);
            if (px >= 0 && px < img.cols && py >= 0 && py < img.rows) img.at<uchar>(py, px) = (uchar)color;
        }
}
static Mat pad16(const Mat& img) {
    int pw=(img.cols+15)&~15, ph=(img.rows+15)&~15;
    if(pw!=img.cols||ph!=img.rows){Mat p;copyMakeBorder(img,p,0,ph-img.rows,0,pw-img.cols,BORDER_CONSTANT,Scalar(0));return p;}
    return img;
}
// Template shapes using cv drawing (larger for reliable feature extraction)
static Mat make_templ_L(int TW) {
    Mat img(TW, TW, CV_8U, Scalar(0));
    rectangle(img, Point(TW/2-15, TW/2-30), Point(TW/2-5, TW/2+30), Scalar(200), -1);
    rectangle(img, Point(TW/2-15, TW/2+20), Point(TW/2+25, TW/2+30), Scalar(200), -1);
    return img;
}
static Mat make_templ_T(int TW) {
    Mat img(TW, TW, CV_8U, Scalar(0));
    rectangle(img, Point(TW/2-5, TW/2-25), Point(TW/2+5, TW/2+25), Scalar(200), -1);
    rectangle(img, Point(TW/2-25, TW/2-25), Point(TW/2+25, TW/2-15), Scalar(200), -1);
    return img;
}
static Mat make_templ_arrow(int TW) {
    Mat img(TW, TW, CV_8U, Scalar(0));
    Point pts[3] = {Point(TW/2-20, TW/2+15), Point(TW/2-20, TW/2-15), Point(TW/2+20, TW/2)};
    fillConvexPoly(img, pts, 3, Scalar(200));
    return img;
}
static Mat make_templ_wrench(int TW) {
    Mat img(TW, TW, CV_8U, Scalar(0));
    rectangle(img, Point(TW/2-30, TW/2-4), Point(TW/2+5, TW/2+4), Scalar(200), -1);
    circle(img, Point(TW/2+12, TW/2), 12, Scalar(200), -1);
    return img;
}

// Draw shape into scene at given angle using warpAffine from 0-degree template
static void drawShapeInScene(Mat& scene, const Mat& templ, int cx, int cy, double angle) {
    int TW = templ.cols;
    Mat M = getRotationMatrix2D(Point2f(TW/2.0f, TW/2.0f), -angle, 1.0);
    Mat rotated;
    warpAffine(templ, rotated, M, Size(TW, TW), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    int ox = cx - TW/2, oy = cy - TW/2;
    for (int r = 0; r < TW; ++r) {
        int sy = oy + r;
        if (sy < 0 || sy >= scene.rows) continue;
        for (int c = 0; c < TW; ++c) {
            int sx = ox + c;
            if (sx < 0 || sx >= scene.cols) continue;
            uchar v = rotated.at<uchar>(r, c);
            if (v > 0) scene.at<uchar>(sy, sx) = v;
        }
    }
}

int main() {
    int TW=80, W=640, H=480, cx=W/2, cy=H/2;

    struct Shape { const char* name; Mat templ; };
    Shape shapes[] = {
        {"L-shape", make_templ_L(TW)},
        {"T-shape", make_templ_T(TW)},
        {"Arrow",   make_templ_arrow(TW)},
        {"Wrench",  make_templ_wrench(TW)},
    };

    fprintf(stderr, "%-10s  %-8s  %10s  %10s\n", "Shape", "Step", "warpAffine", "FeatRotate");
    fprintf(stderr, "%-10s  %-8s  %10s  %10s\n", "-----", "----", "----------", "----------");

    for (auto& shape : shapes) {
        Mat mask = Mat::zeros(TW, TW, CV_8U);
        for (int r = 0; r < TW; ++r)
            for (int c = 0; c < TW; ++c)
                if (shape.templ.at<uchar>(r, c) > 0) mask.at<uchar>(r, c) = 255;
        dilate(mask, mask, Mat(), Point(-1,-1), 5);

        for (float step : {1.0f, 2.0f, 3.0f, 5.0f}) {
            // warpAffine method
            line2Dup::Detector dw(128,{4,8},30,60);
            for(int a=0;a<360;a+=(int)step){
                Mat rt,rm;
                Mat M=getRotationMatrix2D(Point2f(TW/2.0f,TW/2.0f),-(double)a,1.0);
                warpAffine(shape.templ,rt,M,Size(TW,TW));
                warpAffine(mask,rm,M,Size(TW,TW));
                dw.addTemplate(rt,"S",rm);
            }

            // Feature rotation method
            line2Dup::Detector dr(128,{4,8},30,60);
            dr.addRotatedTemplates(shape.templ, mask, "S", 0, 360, step);

            float sw=0, sr=0; int nw=0, nr=0;
            for(int gt=0;gt<360;gt+=5){
                Mat scene(H,W,CV_8U,Scalar(50));
                drawShapeInScene(scene, shape.templ, cx, cy, gt);
                Mat padded=pad16(scene);

                auto mw=dw.match(padded,50);
                if(!mw.empty()){
                    float e=mw[0].template_id*step-gt;
                    if(e>180)e-=360;if(e<-180)e+=360;
                    sw+=e; nw++;
                }
                auto mr=dr.match(padded,50);
                if(!mr.empty()){
                    float e=mr[0].template_id*step-gt;
                    if(e>180)e-=360;if(e<-180)e+=360;
                    sr+=e; nr++;
                }
            }
            fprintf(stderr, "%-10s  step=%1.0f     %+5.1f deg    %+5.1f deg\n",
                    shape.name, step, nw>0?sw/nw:0, nr>0?sr/nr:0);
        }
        fprintf(stderr, "\n");
    }

    return 0;
}
