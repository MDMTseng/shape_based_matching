// study_skew.cpp — realistic stress study: a random template matched against a
// scene with a GRADIENT background + noise + multiple instances that are rotated
// AND slightly SKEWED (affine shear — the matcher only models rigid rot+trans+
// scale, so skew is an un-modelled deformation). Studies how the confidence
// signals (coarse SBM score, refined re-render score, ROI residual) track the
// skew amount and survive the gradient background.
//
//   ./study_skew [seed] [out_scene.png]

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// ---- SBM-style refined re-render score (from refine_confidence) --------------
static const int NB=5,BK=7; static const float WK=30.f;
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
static float refinedScore(const cv::Mat& sBin,const cv::Mat& tmpl,float x,float y,float ang,float scale,int S=2){
    cv::Mat M=cv::getRotationMatrix2D({tmpl.cols/2.f,tmpl.rows/2.f},-ang,scale);
    M.at<double>(0,2)+=x-tmpl.cols/2.0; M.at<double>(1,2)+=y-tmpl.rows/2.0;
    cv::Mat posed; cv::warpAffine(tmpl,posed,M,sBin.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(40));
    cv::Mat bt=binMap(posed); int agree=0,tot=0;
    for(int r=0;r<bt.rows;++r){const schar*t=bt.ptr<schar>(r);
        for(int c=0;c<bt.cols;++c){if(t[c]<0)continue;++tot;bool ok=false;
            for(int dr=-S;dr<=S&&!ok;++dr){int rr=r+dr;if(rr<0||rr>=sBin.rows)continue;const schar*s=sBin.ptr<schar>(rr);
                for(int dc=-S;dc<=S;++dc){int cc=c+dc;if(cc<0||cc>=sBin.cols)continue;
                    if(s[cc]>=0&&bindiff(t[c],s[cc])<=1){ok=true;break;}}}
            if(ok)++agree;}}
    return tot?100.f*agree/tot:0.f;}

// ---- random distinctive template (a few overlapping shapes) -----------------
static cv::Mat randTemplate(cv::RNG& rng,int S=130){
    cv::Mat m(S,S,CV_8U,cv::Scalar(40)); int c=S/2;
    int k=rng.uniform(3,5);
    for(int i=0;i<k;i++){ int cx=rng.uniform(S/3,S*2/3),cy=rng.uniform(S/3,S*2/3);
        int bright=rng.uniform(150,235); double ang=rng.uniform(0.0,360.0);
        if(rng.uniform(0,2)){ int a=rng.uniform(S/8,S/4),b=rng.uniform(S/8,S/4);
            cv::ellipse(m,{cx,cy},{a,b},ang,0,360,bright,cv::FILLED,cv::LINE_AA);}
        else{ int n=rng.uniform(3,6);double R=rng.uniform(S/8.0,S/4.0);std::vector<cv::Point> p;
            for(int j=0;j<n;j++){double t=ang*CV_PI/180+2*CV_PI*j/n+rng.uniform(-0.2,0.2);
                double r=R*(1+rng.uniform(-0.2,0.2));p.push_back({(int)(cx+r*cos(t)),(int)(cy+r*sin(t))});}
            std::vector<std::vector<cv::Point>> pp{p};cv::fillPoly(m,pp,bright,cv::LINE_AA);}}
    (void)c; return m;}

// Gradient background: linear ramp + a radial bump, mid-gray base.
static cv::Mat gradientBG(int W,int H,cv::RNG& rng){
    cv::Mat bg(H,W,CV_32F); float a=rng.uniform(-0.06f,0.06f),b=rng.uniform(-0.06f,0.06f);
    float base=rng.uniform(35.f,60.f); cv::Point2f rc(rng.uniform(0,W),rng.uniform(0,H));
    float ramp=rng.uniform(20.f,45.f), rr=std::hypot(W,H);
    for(int y=0;y<H;y++){float*p=bg.ptr<float>(y);for(int x=0;x<W;x++){
        float g=base+a*x+b*y + ramp*(1.f-std::hypot(x-rc.x,y-rc.y)/rr);
        p[x]=g;}}
    cv::Mat o; bg.convertTo(o,CV_8U); return o;}

int main(int argc,char**argv){
    sbm::setLogLevel(sbm::LogLevel::Error);
    int seed=argc>1?std::atoi(argv[1]):7;
    cv::RNG rng(seed);
    const int W=1000,H=760; const double sigma=10.0; const float tol=16.f;

    cv::Mat tmpl=randTemplate(rng);
    sbm::FeatureSet fs=sbm::extractFeatures(tmpl,cv::Mat(),128);
    sbm::MatchConfig c; c.min_score=38; c.refine=sbm::RefineMode::ROI; c.skip_voting=true;
    c.roi_min_spacing=-1; c.roi_edge_only_points=true;
    c.roi_reject_low_score=true; c.roi_reject_pct=0.8f;   // enable the score gate (inlier_frac)
    sbm::ShapeMatcher M(c); sbm::ModelConfig mc; mc.angle={0,360,3}; mc.scale={1,1,0.1f}; M.addModel("m",fs,mc);

    // Scene: gradient bg + noise + a MATRIX of {skew} x {occlusion} conditions.
    cv::Mat bgClean=gradientBG(W,H,rng);   // keep a clean copy to occlude WITH (edge-free)
    cv::Mat scene=bgClean.clone();
    struct GT{float cx,cy,ang,skew,occ; const char*label;};
    std::vector<GT> gts;
    struct Cond{float skew,occ; const char*label;};
    std::vector<Cond> conds={{0.00f,0.f,"rigid clean"},{0.00f,0.30f,"occlusion only"},
        {0.12f,0.f,"skew only"},{0.12f,0.30f,"skew+occ"},
        {0.20f,0.f,"heavy skew"},{0.20f,0.30f,"heavy skew+occ"}};
    int cols=3,rows=2, i=0;
    for(auto&cd:conds){ int gx=(i%cols),gy=(i/cols);
        float cx=W*(gx+1.0f)/(cols+1)+rng.uniform(-20,20);
        float cy=H*(gy+1.0f)/(rows+1)+rng.uniform(-20,20);
        float ang=rng.uniform(0.f,360.f);
        cv::Mat R=cv::getRotationMatrix2D({tmpl.cols/2.f,tmpl.rows/2.f},ang,1.0);
        cv::Mat Rf=cv::Mat::eye(3,3,CV_64F); R.copyTo(Rf(cv::Rect(0,0,3,2)));
        cv::Mat Sh=(cv::Mat_<double>(3,3)<<1,cd.skew,0, cd.skew*0.5,1,0, 0,0,1);
        cv::Mat A=Sh*Rf; A.at<double>(0,2)+=cx-tmpl.cols/2.0; A.at<double>(1,2)+=cy-tmpl.rows/2.0;
        cv::Mat A23=A(cv::Rect(0,0,3,2)).clone();
        cv::Mat obj; cv::warpAffine(tmpl,obj,A23,scene.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(0));
        obj.copyTo(scene, obj>70);
        if(cd.occ>0){ int w=tmpl.cols,h=tmpl.rows;   // EDGE-FREE occlusion: clean bg shows through the right strip
            int x0=std::max(0,(int)(cx+w/2-cd.occ*w)), x1=std::min(W,(int)(cx+w/2));
            int y0=std::max(0,(int)(cy-h/2)), y1=std::min(H,(int)(cy+h/2));
            if(x1>x0&&y1>y0){ cv::Rect rc(x0,y0,x1-x0,y1-y0); bgClean(rc).copyTo(scene(rc)); } }
        gts.push_back({cx,cy,ang,cd.skew,cd.occ,cd.label}); ++i; }
    cv::Mat noise(H,W,CV_8U); rng.fill(noise,cv::RNG::NORMAL,0,sigma); cv::add(scene,noise,scene);
    if(argc>2) cv::imwrite(argv[2],scene);

    auto rs=M.match(scene);
    std::printf("study_skew | seed %d | random template %dx%d | gradient bg + noise%.0f | skew x occlusion matrix\n",
                seed,tmpl.cols,tmpl.rows,sigma);
    std::printf("  %-16s %-5s %-5s | %-7s %-9s %-11s %-s\n",
                "condition","skew","occ","coarse","residual","inlier_frac","min_ratio");
    std::printf("  ------------------------------------------------------------------------------\n");
    for(auto&g:gts){ float bc=-1,res=-1,inl=-1,mr=-1; bool found=false;
        for(auto&r:rs) if(std::abs(r.x-g.cx)<tol&&std::abs(r.y-g.cy)<tol && r.score>bc){
            bc=r.score; res=r.refine_residual; inl=r.refine_inlier_frac; mr=r.refine_min_ratio; found=true; }
        if(!found){ std::printf("  %-16s %-5.2f %-5.2f | MISS\n",g.label,g.skew,g.occ); continue; }
        std::printf("  %-16s %-5.2f %-5.2f | %-7.1f %-9.2f %-11.2f %-.2f\n",
                    g.label,g.skew,g.occ,bc,res,inl,mr);
    }
    std::printf("\n  (division of labour: residual should rise with SKEW (rigid-fit violation);\n"
                "   inlier_frac should drop with OCCLUSION (points lost); skew+occ hits BOTH.)\n");
    return 0;
}
