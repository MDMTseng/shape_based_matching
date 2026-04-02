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
int main() {
    int TW=80, W=640, H=480, cx=W/2, cy=H/2;
    Mat templ(TW,TW,CV_8U,Scalar(0));
    draw_L(templ,TW/2,TW/2,0,200);
    Mat mask=Mat::ones(TW,TW,CV_8U)*255;

    // Method 1: warpAffine
    line2Dup::Detector det1(128,{4,8},30,60);
    for(int a=0;a<360;a+=2){
        Mat rt,rm;
        Mat M=getRotationMatrix2D(Point2f(TW/2.0f,TW/2.0f),-a,1.0);
        warpAffine(templ,rt,M,Size(TW,TW)); warpAffine(mask,rm,M,Size(TW,TW));
        det1.addTemplate(rt,"L",rm);
    }

    // Method 2: addRotatedTemplates (new batch API)
    line2Dup::Detector det2(128,{4,8},30,60);
    int cnt = det2.addRotatedTemplates(templ, mask, "L", 0, 360, 2);
    fprintf(stderr, "addRotatedTemplates: %d templates\n", cnt);

    // Test with draw_L scenes
    float sum1=0, sum2=0; int n=0;
    for(int gt=0;gt<360;gt+=5){
        Mat scene(H,W,CV_8U,Scalar(50));
        draw_L(scene,cx,cy,gt,200);
        Mat padded=pad16(scene);

        auto m1=det1.match(padded,50);
        auto m2=det2.match(padded,50);
        if(m1.empty()||m2.empty()) continue;

        float e1=m1[0].template_id*2.0f-gt; if(e1>180)e1-=360;if(e1<-180)e1+=360;
        float e2=m2[0].template_id*2.0f-gt; if(e2>180)e2-=360;if(e2<-180)e2+=360;
        sum1+=e1; sum2+=e2; n++;
    }
    fprintf(stderr,"Method 1 (warpAffine):       bias = %+.1f deg  (n=%d)\n",sum1/n,n);
    fprintf(stderr,"Method 2 (rotate features):  bias = %+.1f deg  (n=%d)\n",sum2/n,n);

    // Bias vs step size: warpAffine vs feature rotation
    fprintf(stderr, "\n%-8s  %10s  %10s\n", "Step", "warpAffine", "FeatRotate");
    fprintf(stderr, "%-8s  %10s  %10s\n", "----", "----------", "----------");

    for (float step : {1.0f, 2.0f, 3.0f, 5.0f, 10.0f}) {
        // warpAffine method
        line2Dup::Detector dw(128,{4,8},30,60);
        for(int a=0;a<360;a+=(int)step){
            Mat rt,rm;
            Mat M=getRotationMatrix2D(Point2f(TW/2.0f,TW/2.0f),-a,1.0);
            warpAffine(templ,rt,M,Size(TW,TW)); warpAffine(mask,rm,M,Size(TW,TW));
            dw.addTemplate(rt,"L",rm);
        }

        // Feature rotation method
        line2Dup::Detector dr(128,{4,8},30,60);
        dr.addRotatedTemplates(templ, mask, "L", 0, 360, step);

        float sw=0, sr=0; int nw=0, nr=0;
        for(int gt=0;gt<360;gt+=5){
            Mat scene(H,W,CV_8U,Scalar(50));
            draw_L(scene,cx,cy,gt,200);
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
        fprintf(stderr, "step=%2.0f    %+5.1f deg    %+5.1f deg\n",
                step, nw>0?sw/nw:0, nr>0?sr/nr:0);
    }

    return 0;
}
