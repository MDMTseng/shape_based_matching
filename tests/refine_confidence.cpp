// refine_confidence.cpp — validate a REFINED confidence that re-evaluates the
// SBM-style orientation agreement AT the refined pose (same 0-100 scale as the
// coarse SBM score, but computed at the exact sub-pixel/sub-degree pose).
//
// refined_score: render the template at the match's (x,y,angle,scale), compute
// its quantized orientation, and count the fraction of its edge pixels whose
// orientation agrees (+-1 bin) with the scene at the same pixel. Same semantics
// as the coarse score; sharper (exact pose, full res); drops when refine diverges.
//
// Checks: (1) on-scale + ~tracks coarse for good matches, (2) sharp vs pose
// offset (peaks at the true pose), (3) drops under occlusion / bad fit,
// (4) inversely tracks refine_residual.
//
//   ./refine_confidence

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static const int NB=5, BK=7; static const float WK=30.f;
static inline int qbin(float gx,float gy){int x=(int)std::lround(gx),y=(int)std::lround(gy);
    if(y<0){x=-x;y=-y;}if(y==0&&x<0)x=-x;static const long T[4]={1989,6682,14966,50273};
    long ty=(long)y*10000;int b;if(x>=0){if(ty<(long)x*T[0])b=0;else if(ty<(long)x*T[1])b=1;
        else if(ty<(long)x*T[2])b=2;else if(ty<(long)x*T[3])b=3;else b=4;}
    else{long a=-x;if(ty<a*T[0])b=0;else if(ty<a*T[1])b=7;else if(ty<a*T[2])b=6;else if(ty<a*T[3])b=5;else b=4;}return b;}
static inline int bindiff(int a,int b){int d=std::abs(a-b);return std::min(d,8-d);}
static cv::Mat binMap(const cv::Mat&g){cv::Mat sm,gx,gy;cv::GaussianBlur(g,sm,{BK,BK},0);
    cv::Sobel(sm,gx,CV_32F,1,0,3);cv::Sobel(sm,gy,CV_32F,0,1,3);
    cv::Mat raw(g.size(),CV_8S,cv::Scalar(-1)),mag(g.size(),CV_32F);
    for(int r=0;r<g.rows;++r){const float*a=gx.ptr<float>(r);const float*b=gy.ptr<float>(r);
        schar*rr=raw.ptr<schar>(r);float*mr=mag.ptr<float>(r);
        for(int c=0;c<g.cols;++c){mr[c]=std::sqrt(a[c]*a[c]+b[c]*b[c]);rr[c]=(schar)qbin(a[c],b[c]);}}
    cv::Mat o(g.size(),CV_8S,cv::Scalar(-1));
    for(int r=1;r<g.rows-1;++r){const float*mr=mag.ptr<float>(r);schar*orow=o.ptr<schar>(r);
        for(int c=1;c<g.cols-1;++c){if(mr[c]<=WK)continue;int h[8]={0};
            for(int dr=-1;dr<=1;++dr){const schar*q=raw.ptr<schar>(r+dr);for(int dc=-1;dc<=1;++dc)h[q[c+dc]]++;}
            int mv=0,idx=-1;for(int i=0;i<8;i++)if(h[i]>mv){mv=h[i];idx=i;}if(mv>=NB)orow[c]=(schar)idx;}}
    return o;}

// Render the template at the match pose and score orientation agreement vs the
// scene over the template's edge pixels — same 0-100 semantics as the SBM score.
// `matchAngle` is the MatchResult angle (matcher convention is the negative of
// the OpenCV render rotation, so we render at -matchAngle). A small spatial
// SPREAD (+-S px) mirrors the coarse matcher's T-neighborhood tolerance so the
// number is comparable to the coarse score (not over-strict under noise).
static float refinedScore(const cv::Mat& sceneBin, const cv::Mat& tmpl,
                          float x, float y, float matchAngle, float scale, int S=2) {
    cv::Mat M = cv::getRotationMatrix2D({tmpl.cols/2.f, tmpl.rows/2.f}, -matchAngle, scale);
    M.at<double>(0,2) += x - tmpl.cols/2.0;    // template center -> (x,y)
    M.at<double>(1,2) += y - tmpl.rows/2.0;
    cv::Mat posed; cv::warpAffine(tmpl, posed, M, sceneBin.size(),
                                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(40));
    cv::Mat bt = binMap(posed);
    int agree=0, tot=0;
    for (int r=0;r<bt.rows;++r){ const schar* t=bt.ptr<schar>(r);
        for (int c=0;c<bt.cols;++c){ if(t[c]<0) continue; ++tot; bool ok=false;
            for(int dr=-S;dr<=S&&!ok;++dr){ int rr=r+dr; if(rr<0||rr>=sceneBin.rows)continue;
                const schar* s=sceneBin.ptr<schar>(rr);
                for(int dc=-S;dc<=S;++dc){ int cc=c+dc; if(cc<0||cc>=sceneBin.cols)continue;
                    if(s[cc]>=0 && bindiff(t[c],s[cc])<=1){ok=true;break;} } }
            if(ok) ++agree; } }
    return tot? 100.f*agree/tot : 0.f;
}

static cv::Mat arrow(){cv::Mat m(160,160,CV_8U,cv::Scalar(40));cv::rectangle(m,{75,38},{85,132},220,cv::FILLED);
    std::vector<cv::Point> h{{80,16},{64,44},{96,44}};cv::fillConvexPoly(m,h,220);return m;}

struct Scene{cv::Mat img;float cx,cy;};
static Scene place(const cv::Mat&t,float ang,double sig,int W,int H,float occl,cv::RNG&rng){
    cv::Mat R=cv::getRotationMatrix2D({t.cols/2.f,t.rows/2.f},ang,1.0),rot;
    cv::warpAffine(t,rot,R,t.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(40));
    cv::Mat scene(H,W,CV_8U,cv::Scalar(40));int px=(W-t.cols)/2,py=(H-t.rows)/2;
    rot.copyTo(scene(cv::Rect(px,py,t.cols,t.rows)));
    if(occl>0){ int cut=(int)(t.rows*occl);   // occlude the object's BOTTOM occl fraction
        cv::rectangle(scene,{px,py+t.rows-cut},{px+t.cols,py+t.rows},40,cv::FILLED); }
    if(sig>0){cv::Mat n(H,W,CV_8U);rng.fill(n,cv::RNG::NORMAL,0,sig);cv::add(scene,n,scene);}
    return {scene,px+t.cols/2.f,py+t.rows/2.f};}

int main(){
    sbm::setLogLevel(sbm::LogLevel::Error);
    cv::Mat tmpl=arrow(); const int W=1280,H=960; const float tol=12.f;
    sbm::FeatureSet fs=sbm::extractFeatures(tmpl,cv::Mat(),128);
    sbm::MatchConfig c; c.min_score=20; c.refine=sbm::RefineMode::ROI; c.skip_voting=true;
    sbm::ShapeMatcher M(c); sbm::ModelConfig mc; mc.angle={0,360,2}; mc.scale={1,1,0.1f}; M.addModel("m",fs,mc);

    std::printf("refine_confidence | arrow | refine=ROI | refined_score = SBM-style orientation\n"
                "agreement of the template rendered at the refined pose vs the scene.\n\n");

    struct Case{const char*name; float ang; double sig; float occl;};
    std::vector<Case> cases={{"clean 30",30,0,0},{"noise12 30",30,12,0},{"noise25 75",75,25,0},
                             {"noise12 140",140,12,0},{"occlude25% 200",200,12,0.25f},
                             {"occlude50% 300",300,12,0.50f}};
    std::printf("  %-16s %-8s %-9s %-9s  %-6s %-6s\n","case","coarse","refined","residual","@+3deg","@+10deg");
    std::printf("  ------------------------------------------------------------------------------\n");
    cv::RNG rng(7);
    for(auto&cs:cases){ Scene sc=place(tmpl,cs.ang,cs.sig,W,H,cs.occl,rng);
        auto rs=M.match(sc.img); float bx=0,by=0,ba=0,cScore=-1,resid=-1;
        for(auto&r:rs) if(std::abs(r.x-sc.cx)<tol&&std::abs(r.y-sc.cy)<tol && r.score>cScore){
            cScore=r.score; bx=r.x; by=r.y; ba=r.angle; resid=r.refine_residual; }
        if(cScore<0){ std::printf("  %-16s   MISS\n",cs.name); continue; }
        cv::Mat sBin=binMap(sc.img);
        float rScore=refinedScore(sBin,tmpl,bx,by,ba,1.0f);
        // discriminative check: refined_score at a WRONG pose (+-3 / +-10 deg).
        float w3=refinedScore(sBin,tmpl,bx,by,ba+3,1.0f), w10=refinedScore(sBin,tmpl,bx,by,ba+10,1.0f);
        std::printf("  %-16s %-8.1f %-9.1f %-9.2f  %5.1f  %5.1f\n",cs.name,cScore,rScore,resid,w3,w10);
    }
    std::printf("\n(refined tracks coarse for good matches (same scale); the +-4deg profile shows\n"
                " it PEAKS at the refined pose (sharpness); occlusion drops it; it should move\n"
                " opposite to residual. If refined is LOW while coarse is high -> refine diverged.)\n");
    return 0;
}
