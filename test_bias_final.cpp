// Final bias table: 4 shapes x 4 steps x 2 methods
// Scene created with warpAffine (consistent with template creation)

#include "line2Dup.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
using namespace cv;

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
    for(int r=0;r<TW;r++) { int sy=oy+r; if(sy<0||sy>=scene.rows) continue;
        for(int c=0;c<TW;c++) { int sx=ox+c; if(sx<0||sx>=scene.cols) continue;
            uchar v=rotated.at<uchar>(r,c); if(v>0) scene.at<uchar>(sy,sx)=v; }}
}

static float measureBias(line2Dup::Detector& det, const Mat& templ,
                         float step, int W, int H, int cx, int cy) {
    float sum=0; int n=0;
    for(int gt=0; gt<360; gt+=5) {
        Mat scene(H, W, CV_8U, Scalar(50));
        placeInScene(scene, templ, cx, cy, gt);
        auto m = det.match(pad16(scene), 50);
        if(m.empty()) continue;
        float e = m[0].template_id * step - gt;
        if(e>180) e-=360; if(e<-180) e+=360;
        sum += e; n++;
    }
    return n > 0 ? sum/n : 999;
}

int main() {
    const int TW=80, W=640, H=480, cx=W/2, cy=H/2;

    struct Shape { const char* name; Mat templ; };
    Shape shapes[] = {
        {"L-shape", make_L(TW)},
        {"T-shape", make_T(TW)},
        {"Arrow",   make_arrow(TW)},
        {"Wrench",  make_wrench(TW)},
    };
    float steps[] = {1, 2, 3, 5};

    FILE* f;
    fopen_s(&f, "C:/Users/TRS001/Documents/workspace/templmatch/test_imgs/bias_final.txt", "w");
    if(!f) return 1;

    // Header
    fprintf(f, "%-10s", "Shape");
    for(float s : steps) fprintf(f, " |   step=%1.0f         ", s);
    fprintf(f, "\n%-10s", "");
    for(int i=0;i<4;i++) fprintf(f, " |  warp    feat   ");
    fprintf(f, "\n%-10s", "-----");
    for(int i=0;i<4;i++) fprintf(f, " | ------  ------  ");
    fprintf(f, "\n");

    for(auto& shape : shapes) {
        Mat mask = Mat::zeros(TW,TW,CV_8U);
        for(int r=0;r<TW;r++) for(int c=0;c<TW;c++)
            if(shape.templ.at<uchar>(r,c)>0) mask.at<uchar>(r,c)=255;
        dilate(mask, mask, Mat(), Point(-1,-1), 5);

        // Count features at 0 degrees
        {
            line2Dup::Detector dtmp(128,{4,8},30,60);
            int id = dtmp.addTemplate(shape.templ, "tmp", mask);
            if(id >= 0) {
                auto& t = dtmp.getTemplates("tmp", id);
                fprintf(f, "%-10s (%3d feats)", shape.name, (int)t[0].features.size());
            } else {
                fprintf(f, "%-10s (??? feats)", shape.name);
            }
        }

        for(float step : steps) {
            // Method 1: warpAffine
            line2Dup::Detector dw(128,{4,8},30,60);
            for(int a=0;a<360;a+=(int)step) {
                Mat rt,rm;
                Mat M=getRotationMatrix2D(Point2f(TW/2.0f,TW/2.0f),-(double)a,1.0);
                warpAffine(shape.templ,rt,M,Size(TW,TW));
                warpAffine(mask,rm,M,Size(TW,TW));
                dw.addTemplate(rt,"S",rm);
            }
            float bw = measureBias(dw, shape.templ, step, W, H, cx, cy);

            // Method 2: feature rotation
            line2Dup::Detector dr(128,{4,8},30,60);
            dr.addRotatedTemplates(shape.templ, mask, "S", 0, 360, step);
            float br = measureBias(dr, shape.templ, step, W, H, cx, cy);

            fprintf(f, " | %+5.1f  %+5.1f  ", bw, br);
        }
        fprintf(f, "\n");
    }

    fprintf(f, "\nAll values in degrees. Scene created with warpAffine.\n");
    fclose(f);
    fprintf(stderr, "Done. See bias_final.txt\n");
    return 0;
}
