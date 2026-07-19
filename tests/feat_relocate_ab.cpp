// feat_relocate_ab.cpp — can a rotation-UNSTABLE feature be made robust by
// MOVING it (position) instead of dropping it? Tests feature relocation: slide
// each unstable feature along its edge tangent (perpendicular to its gradient)
// to a nearby MORE rotation-stable point, re-reading its orientation there.
// Maintains the feature count N (unlike selectRotationStable's drop; unlike the
// score-reweight which only reorders). The idea: a corner is orientation-
// ambiguous IN THE IMAGE (unfixable in place), but sliding OFF the junction onto
// a clean edge segment lands on a well-defined, rotation-faithful orientation.
//
//   ./feat_relocate_ab star 1 40    (shape, register step, num_features)

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static const int   NEIGHBOR_THRESHOLD=5, BLUR_KSIZE=7; static const float WEAK=30.f;
static inline int rs_qbin(float gxf,float gyf){int gx=(int)std::lround(gxf),gy=(int)std::lround(gyf);
    if(gy<0){gx=-gx;gy=-gy;}if(gy==0&&gx<0)gx=-gx;static const long T[4]={1989,6682,14966,50273};
    long ty=(long)gy*10000;int b;if(gx>=0){if(ty<(long)gx*T[0])b=0;else if(ty<(long)gx*T[1])b=1;
        else if(ty<(long)gx*T[2])b=2;else if(ty<(long)gx*T[3])b=3;else b=4;}
    else{long a=-gx;if(ty<a*T[0])b=0;else if(ty<a*T[1])b=7;else if(ty<a*T[2])b=6;else if(ty<a*T[3])b=5;else b=4;}return b;}
static inline int bindiff(int a,int b){int d=std::abs(a-b);return std::min(d,8-d);}
static cv::Mat votedBinMap(const cv::Mat&g){cv::Mat sm,gx,gy;cv::GaussianBlur(g,sm,{BLUR_KSIZE,BLUR_KSIZE},0);
    cv::Sobel(sm,gx,CV_32F,1,0,3);cv::Sobel(sm,gy,CV_32F,0,1,3);
    cv::Mat raw(g.size(),CV_8S,cv::Scalar(-1)),mag(g.size(),CV_32F);
    for(int r=0;r<g.rows;++r){const float*a=gx.ptr<float>(r);const float*b=gy.ptr<float>(r);
        schar*rr=raw.ptr<schar>(r);float*mr=mag.ptr<float>(r);
        for(int c=0;c<g.cols;++c){mr[c]=std::sqrt(a[c]*a[c]+b[c]*b[c]);rr[c]=(schar)rs_qbin(a[c],b[c]);}}
    cv::Mat o(g.size(),CV_8S,cv::Scalar(-1));
    for(int r=1;r<g.rows-1;++r){const float*mr=mag.ptr<float>(r);schar*orow=o.ptr<schar>(r);
        for(int c=1;c<g.cols-1;++c){if(mr[c]<=WEAK)continue;int h[8]={0};
            for(int dr=-1;dr<=1;++dr){const schar*q=raw.ptr<schar>(r+dr);for(int dc=-1;dc<=1;++dc)h[q[c+dc]]++;}
            int mv=0,idx=-1;for(int i=0;i<8;i++)if(h[i]>mv){mv=h[i];idx=i;}if(mv>=NEIGHBOR_THRESHOLD)orow[c]=(schar)idx;}}
    return o;}
static inline int binAt(const cv::Mat&m,float x,float y){int xi=(int)std::lround(x),yi=(int)std::lround(y);
    if(xi<0||yi<0||xi>=m.cols||yi>=m.rows)return -1;return m.at<schar>(yi,xi);}

// Relocate rotation-unstable finest-level features along their edge tangent to a
// nearby more-stable point (re-reading the label there). Count preserved.
static sbm::FeatureSet relocate(const sbm::FeatureSet& fs, const cv::Mat& tmpl,
                                int slide_max, int& moved) {
    const float probes[]={6,-6,12,-12}; cv::Point2f ctr(tmpl.cols/2.f,tmpl.rows/2.f);
    struct RM{cv::Mat bm,M;}; std::vector<RM> rms;
    cv::Mat base = votedBinMap(tmpl);
    for(float a:probes){ RM r; r.M=cv::getRotationMatrix2D(ctr,a,1.0);
        cv::Mat rot; cv::warpAffine(tmpl,rot,r.M,tmpl.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        r.bm=votedBinMap(rot); rms.push_back(r); }
    // Mean orientation DRIFT: how far the ACTUAL voted bin at the rotated position
    // deviates from the ANALYTIC-rotation PREDICTION (gradient rotates by the probe
    // angle). Faithful edges ~0; corners/off-structure high. Lower = more stable.
    auto drift=[&](float x,float y,int lbl){ float acc=0; int tot=0;
        for(size_t k=0;k<rms.size();++k){ double*m=(double*)rms[k].M.data;
            float rx=m[0]*x+m[1]*y+m[2], ry=m[3]*x+m[4]*y+m[5];
            int actual=binAt(rms[k].bm,rx,ry); ++tot;
            if(actual<0){ acc+=4; continue; }                     // rotated off structure = max drift
            float phi=(lbl*22.5f + probes[k])*(float)CV_PI/180.f; // analytic-rotated prediction
            int pred=rs_qbin(std::cos(phi)*100.f, std::sin(phi)*100.f);
            acc += bindiff(actual,pred); }
        return tot?acc/tot:4.f; };
    sbm::FeatureSet out=fs; moved=0;
    if(out.levels.empty()) return out;
    auto& lv=out.levels[0];
    for(auto& f:lv.features){ float ax=f.x+lv.tl_x, ay=f.y+lv.tl_y;
        float d0=drift(ax,ay,f.label); if(d0<=1.5f) continue;     // only genuine corners (high drift)
        float gdir=f.label*22.5f*(float)CV_PI/180.f; float tx=-std::sin(gdir),ty=std::cos(gdir);
        float bestD=d0,bx=ax,by=ay; int bl=f.label;
        for(int s=-slide_max;s<=slide_max;++s){ if(!s)continue;
            float nx=ax+tx*s, ny=ay+ty*s; int nl=binAt(base,nx,ny); if(nl<0)continue; // off-structure
            float dd=drift(nx,ny,nl); if(dd<bestD-0.3f){bestD=dd;bx=nx;by=ny;bl=nl;} }
        if(bx!=ax||by!=ay){ f.x=(int)std::lround(bx)-lv.tl_x; f.y=(int)std::lround(by)-lv.tl_y; f.label=bl; ++moved; }
    }
    return out;
}

static cv::Mat arrowShape(){cv::Mat m(160,160,CV_8U,cv::Scalar(40));cv::rectangle(m,{75,38},{85,132},220,cv::FILLED);
    std::vector<cv::Point> h{{80,16},{64,44},{96,44}};cv::fillConvexPoly(m,h,220);return m;}
static cv::Mat starShape(){cv::Mat m(160,160,CV_8U,cv::Scalar(40));const int c=80;std::vector<cv::Point> st;
    for(int i=0;i<10;i++){double a=CV_PI/2+i*CV_PI/5;double r=(i&1)?24:54;st.push_back({(int)(c+r*cos(a)),(int)(c-r*sin(a))});}
    cv::polylines(m,st,true,225,3);return m;}

struct TC{cv::Mat scene;float cx,cy;};
static TC make_case(const cv::Mat&t,float ang,double sig,int W,int H,cv::RNG&rng){
    cv::Mat R=cv::getRotationMatrix2D({t.cols/2.f,t.rows/2.f},ang,1.0),rot;
    cv::warpAffine(t,rot,R,t.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(40));
    cv::Mat scene(H,W,CV_8U,cv::Scalar(40));int px=(W-rot.cols)/2,py=(H-rot.rows)/2;
    rot.copyTo(scene(cv::Rect(px,py,rot.cols,rot.rows)));
    if(sig>0){cv::Mat n(H,W,CV_8U);rng.fill(n,cv::RNG::NORMAL,0,sig);cv::add(scene,n,scene);}
    return {scene,px+rot.cols/2.f,py+rot.rows/2.f};}
static float bestScore(sbm::ShapeMatcher&m,const TC&tc,float tol){float best=-1;
    for(auto&r:m.match(tc.scene))if(std::abs(r.x-tc.cx)<tol&&std::abs(r.y-tc.cy)<tol)best=std::max(best,r.score);return best;}
static void report(const char*tag,sbm::ShapeMatcher&m,const cv::Mat&t,double sig,int W,int H,float tol){
    cv::RNG rng(31);double sum=0;float mn=1e9,mx=0;int amin=0;std::vector<float> s;
    for(int a=0;a<360;++a){TC tc=make_case(t,(float)a,sig,W,H,rng);float v=bestScore(m,tc,tol);s.push_back(v);
        if(v>=0){sum+=v;mx=std::max(mx,v);if(v<mn){mn=v;amin=a;}}}
    double mean=sum/360,var=0;for(float v:s)var+=(v-mean)*(v-mean);
    std::printf("  %-24s mean %5.1f  min %5.1f @%3ddeg  peak-peak %4.1f  std %4.2f\n",
                tag,mean,mn>1e8?0:mn,amin,mx-(mn>1e8?0:mn),std::sqrt(var/360));}

int main(int argc,char**argv){
    sbm::setLogLevel(sbm::LogLevel::Warning);
    std::string shp=argc>1?argv[1]:"star"; float step=argc>2?(float)std::atof(argv[2]):1.f;
    int nf=argc>3?std::atoi(argv[3]):40;
    cv::Mat tmpl=(shp=="arrow")?arrowShape():starShape();
    const int W=1280,H=960;const float tol=12.f;

    sbm::FeatureSet f0=sbm::extractFeatures(tmpl,cv::Mat(),nf);
    int moved=0; sbm::FeatureSet fRel=relocate(f0,tmpl,6,moved);
    sbm::FeatureSet fDrop=sbm::selectRotationStable(f0,tmpl,9,3,0.8f);
    int n0=f0.levels[0].features.size(), nR=fRel.levels[0].features.size(), nD=fDrop.levels[0].features.size();

    auto cfg=[&](){sbm::MatchConfig c;c.min_score=20;c.refine=sbm::RefineMode::None;c.skip_voting=true;c.match_scale=1;return c;};
    auto mk=[&](const sbm::FeatureSet&fs){auto m=std::make_shared<sbm::ShapeMatcher>(cfg());
        sbm::ModelConfig mc;mc.angle={0,360,step};mc.scale={1,1,0.1f};m->addModel("m",fs,mc);return m;};
    auto N=mk(f0),Rl=mk(fRel),Dr=mk(fDrop);

    std::printf("feat_relocate_ab | %s | extract@0 register 360@%.1fdeg | "
                "counts: normal=%d relocated=%d(moved %d) selectRotStable=%d\n",
                shp.c_str(),step,n0,nR,moved,nD);
    std::printf("object 0..359 @1deg\n\n");
    for(double sig:{0.0,12.0}){ std::printf("[noise sigma=%.0f]\n",sig);
        report("normal (N)",*N,tmpl,sig,W,H,tol);
        report("RELOCATED (SAME N)",*Rl,tmpl,sig,W,H,tol);
        report("selectRotStable (fewer)",*Dr,tmpl,sig,W,H,tol);
        std::printf("\n"); }
    std::printf("(relocation slides unstable features along their edge tangent to a more\n"
                " rotation-stable point, keeping count N. If peak-peak/min beats normal AND\n"
                " ~matches the drop version, moving beats dropping — robustness at full count.)\n");
    return 0;
}
