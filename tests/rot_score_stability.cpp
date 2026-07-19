// rot_score_stability.cpp — score stability of the STANDARD workflow:
// extract features ONCE at 0deg, register 360deg variants by ANALYTIC rotation,
// then rotate the real object through 360deg and watch the match score.
//
// Even with a FINE angle step the score is not flat: the analytically-rotated
// 0deg features do not perfectly match the really-rotated object (a straight
// edge rotates rigidly, but corners / curves / anti-aliased edges / quantization
// boundaries do not) — that residual wobble IS the feature set's rotation
// stability. This isolates it (fine step + optional zero noise) and asks whether
// sbm::selectRotationStable (which keeps analytic-rotation-stable features)
// flattens it.
//
//   ./rot_score_stability          # arrow   ./rot_score_stability star   # star
//   ./rot_score_stability arrow 1  # register step 1deg (default)

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static cv::Mat arrowShape(){ cv::Mat m(160,160,CV_8U,cv::Scalar(40));
    cv::rectangle(m,{75,38},{85,132},220,cv::FILLED);
    std::vector<cv::Point> h{{80,16},{64,44},{96,44}}; cv::fillConvexPoly(m,h,220); return m; }
static cv::Mat starShape(){ cv::Mat m(160,160,CV_8U,cv::Scalar(40)); const int c=80;
    std::vector<cv::Point> st; for(int i=0;i<10;i++){double a=CV_PI/2+i*CV_PI/5;double r=(i&1)?24:54;
        st.push_back({(int)(c+r*cos(a)),(int)(c-r*sin(a))});} cv::polylines(m,st,true,225,3); return m; }

struct TC{ cv::Mat scene; float cx,cy; };
static TC make_case(const cv::Mat&t,float ang,double sig,int W,int H,cv::RNG&rng){
    // Pad before rotating so a large pattern's corners are not clipped.
    int P=(int)(std::max(t.cols,t.rows)*1.45);
    cv::Mat pad(P,P,CV_8U,cv::Scalar(40));
    t.copyTo(pad(cv::Rect((P-t.cols)/2,(P-t.rows)/2,t.cols,t.rows)));
    cv::Mat R=cv::getRotationMatrix2D({P/2.f,P/2.f},ang,1.0),rot;
    cv::warpAffine(pad,rot,R,{P,P},cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(40));
    cv::Mat scene(H,W,CV_8U,cv::Scalar(40));int px=(W-P)/2,py=(H-P)/2;
    rot.copyTo(scene(cv::Rect(px,py,P,P)));
    if(sig>0){cv::Mat n(H,W,CV_8U);rng.fill(n,cv::RNG::NORMAL,0,sig);cv::add(scene,n,scene);}
    return {scene,W/2.f,H/2.f}; }
static float bestScore(sbm::ShapeMatcher&m,const TC&tc,float tol){ float best=-1;
    for(auto&r:m.match(tc.scene)) if(std::abs(r.x-tc.cx)<tol&&std::abs(r.y-tc.cy)<tol) best=std::max(best,r.score);
    return best; }

// Sweep the object 0..359 at 1deg; return the per-angle correct-match score.
static std::vector<float> sweep(sbm::ShapeMatcher&m,const cv::Mat&t,double sig,int W,int H,float tol){
    std::vector<float> s; cv::RNG rng(31);
    for(int a=0;a<360;++a){ TC tc=make_case(t,(float)a,sig,W,H,rng); s.push_back(bestScore(m,tc,tol)); }
    return s; }
static void report(const char* tag, const std::vector<float>& s){
    double sum=0; float mn=1e9,mx=0; int amin=0, miss=0;
    for(int a=0;a<360;++a){ float v=s[a]; if(v<0){miss++;continue;} sum+=v; mx=std::max(mx,v);
        if(v<mn){mn=v;amin=a;} }
    int n=360-miss; double mean=n?sum/n:0; double var=0;
    for(float v:s) if(v>=0) var+=(v-mean)*(v-mean);
    double sd=n?std::sqrt(var/n):0;
    // worst score in each 30deg bucket — shows WHERE it dips.
    std::string prof="  ";
    for(int b=0;b<12;++b){ float bmn=1e9; for(int a=b*30;a<b*30+30;++a) if(s[a]>=0) bmn=std::min(bmn,s[a]);
        char c[8]; std::snprintf(c,sizeof c,"%4.0f",bmn>1e8?0:bmn); prof+=c; }
    std::printf("  %-22s mean %5.1f  min %5.1f @%3ddeg  peak-peak %4.1f  std %4.2f  miss %d\n",
                tag, mean, mn>1e8?0:mn, amin, mx-(mn>1e8?0:mn), sd, miss);
    std::printf("    per-30deg worst:%s\n", prof.c_str());
}

int main(int argc,char**argv){
    sbm::setLogLevel(sbm::LogLevel::Warning);
    std::string shp = argc>1?argv[1]:"arrow";
    float step = argc>2?(float)std::atof(argv[2]):1.f;
    int   nf   = argc>3?std::atoi(argv[3]):40;   // keep well below candidate supply so
                                                 // SELECTION actually chooses a subset
    const int W=1280,H=960; const float tol=12.f;
    cv::Mat tmpl;
    if (shp.find('.')!=std::string::npos) tmpl = cv::imread(shp, cv::IMREAD_GRAYSCALE);
    if (tmpl.empty()) tmpl = (shp=="star")?starShape():arrowShape();

    sbm::FeatureSet f0 = sbm::extractFeatures(tmpl,cv::Mat(),nf);          // normal: extract N @0
    sbm::FeatureSet fR = sbm::selectRotationStable(f0,tmpl,9.f,3.f,0.8f);  // subset (fewer than N)
    int n0=f0.levels.empty()?0:(int)f0.levels[0].features.size();
    int nR=fR.levels.empty()?0:(int)fR.levels[0].features.size();

    auto cfg=[&](){ sbm::MatchConfig c;c.min_score=20;c.refine=sbm::RefineMode::None;
        c.skip_voting=true;c.match_scale=1.0f;return c; };
    auto mk=[&](const sbm::FeatureSet&fs){ auto m=std::make_shared<sbm::ShapeMatcher>(cfg());
        sbm::ModelConfig mc; mc.angle={0,360,step}; mc.scale={1,1,0.1f}; m->addModel("m",fs,mc); return m; };
    auto N=mk(f0), R=mk(fR);

    std::printf("rot_score_stability | %s | extract@0, register 360 @ step %.1fdeg\n"
                "feature counts: normal=%d  selectRotationStable(drops to)=%d\n", shp.c_str(),step,n0,nR);
    std::printf("object rotated 0..359 @ 1deg; score = correct-match score (analytic-rotation "
                "vs real-rotation fidelity)\n\n");

    for (double sig : {0.0, 12.0}) {
        std::printf("[noise sigma=%.0f]\n", sig);
        report("normal (N feats)",       sweep(*N,tmpl,sig,W,H,tol));
        report("selectRotStable (~0.8N)", sweep(*R,tmpl,sig,W,H,tol));
        std::printf("\n");
    }
    std::printf("(peak-peak / std = the rotation wobble of a 0deg-extracted, analytically-\n"
                " rotated feature set; lower = more rotation-stable. per-30deg worst shows the\n"
                " dip pattern — quantization-boundary / pixel-grid periodicity.)\n");
    return 0;
}
