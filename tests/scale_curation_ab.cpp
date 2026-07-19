// scale_curation_ab.cpp — Phase-2 A/B: does dropping the noise-fragile feature
// tail AT THE SMALL-SCALE VARIANTS raise the downscale score?
//
// Root cause (feat_robustness_probe): the downscale cliff is a fragile TAIL of
// features whose 8-bin orientation is destroyed by noise once the structure is
// small; they contribute ~0 but stay in the count-normalized denominator
// (score = raw/(4*numFeatures)) and DILUTE the score. Prediction: curating them
// out of the small-scale variants shrinks the denominator to contributing
// features and RAISES worst-score at downscale — unlike rotation (uniform
// angular loss, undroppable).
//
// No library change: scale-adaptive curation is simulated by registering TWO
// models — a robust-subset model over the small-scale range and the full model
// over the rest — so NMS picks the best. Fragility is an ABSOLUTE gate
// (recovery-under-noise at a mild downscale < THRESH), so it auto-adapts: a
// solid shape drops nothing, a thin shape drops its fragile tail.
//
//   A  baseline      : full features, scale [0.6,1.4]
//   B  scale-adaptive: robust subset [0.6,0.8) + full [0.8,1.4]
//   C  global-curated: robust subset, scale [0.6,1.4]   (shows why adaptive wins)
//
//   ./scale_curation_ab           # thin star (4)   ./scale_curation_ab 3  # solid

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static const int   NEIGHBOR_THRESHOLD = 5;
static const int   BLUR_KSIZE = 7;
static const float WEAK_THRESH = 30.f;

static inline int rs_qbin(float gxf, float gyf) {
    int gx=(int)std::lround(gxf), gy=(int)std::lround(gyf);
    if (gy<0){gx=-gx;gy=-gy;} if (gy==0&&gx<0) gx=-gx;
    static const long TAN_B[4]={1989,6682,14966,50273};
    long ty=(long)gy*10000; int b;
    if (gx>=0){ if(ty<(long)gx*TAN_B[0])b=0; else if(ty<(long)gx*TAN_B[1])b=1;
        else if(ty<(long)gx*TAN_B[2])b=2; else if(ty<(long)gx*TAN_B[3])b=3; else b=4; }
    else { long a=-gx; if(ty<a*TAN_B[0])b=0; else if(ty<a*TAN_B[1])b=7;
        else if(ty<a*TAN_B[2])b=6; else if(ty<a*TAN_B[3])b=5; else b=4; }
    return b;
}
static inline int bindiff(int a,int b){int d=std::abs(a-b);return std::min(d,8-d);}

static cv::Mat votedBinMap(const cv::Mat& gray){
    cv::Mat sm,gx,gy; cv::GaussianBlur(gray,sm,cv::Size(BLUR_KSIZE,BLUR_KSIZE),0);
    cv::Sobel(sm,gx,CV_32F,1,0,3); cv::Sobel(sm,gy,CV_32F,0,1,3);
    cv::Mat raw(gray.size(),CV_8S,cv::Scalar(-1)), mag(gray.size(),CV_32F);
    for(int r=0;r<gray.rows;++r){const float*gxr=gx.ptr<float>(r);const float*gyr=gy.ptr<float>(r);
        schar*rr=raw.ptr<schar>(r);float*mr=mag.ptr<float>(r);
        for(int c=0;c<gray.cols;++c){mr[c]=std::sqrt(gxr[c]*gxr[c]+gyr[c]*gyr[c]);rr[c]=(schar)rs_qbin(gxr[c],gyr[c]);}}
    cv::Mat out(gray.size(),CV_8S,cv::Scalar(-1));
    for(int r=1;r<gray.rows-1;++r){const float*mr=mag.ptr<float>(r);schar*orow=out.ptr<schar>(r);
        for(int c=1;c<gray.cols-1;++c){ if(mr[c]<=WEAK_THRESH)continue; int h[8]={0};
            for(int dr=-1;dr<=1;++dr){const schar*q=raw.ptr<schar>(r+dr);for(int dc=-1;dc<=1;++dc)h[q[c+dc]]++;}
            int mv=0,idx=-1;for(int i=0;i<8;i++)if(h[i]>mv){mv=h[i];idx=i;}
            if(mv>=NEIGHBOR_THRESHOLD)orow[c]=(schar)idx; }}
    return out;
}
static inline int binAt(const cv::Mat&bm,float x,float y){int xi=(int)std::lround(x),yi=(int)std::lround(y);
    if(xi<0||yi<0||xi>=bm.cols||yi>=bm.rows)return -1;return bm.at<schar>(yi,xi);}

// Robust subset: drop features whose orientation-recovery under noise at a mild
// downscale (0.85) is below THRESH. Auto-adaptive per shape. Returns a curated
// copy + the drop count.
static sbm::FeatureSet curate(const sbm::FeatureSet& fs, const cv::Mat& tmpl,
                              float thresh, int& dropped) {
    const float mild = 0.85f; const int MC = 16; const double sigma = 12.0;
    cv::RNG rng(777);
    sbm::FeatureSet out = fs; dropped = 0;
    // Curate EACH pyramid level at its OWN resolution. A level-L feature's coord
    // (f.x+tl_x) is in (>>L) units; its full-res-scaled scene position is
    // c*2^L*mild, and the level-L match runs on pyrDown^L(scene), where that maps
    // to c*mild (the 2^L and the L pyrDowns cancel). So probe position is
    // (f.x+tl_x)*mild for every level; only the IMAGE differs — pyrDown^L of the
    // scaled+noised template, exactly the coarse image the matcher sees. This
    // sees coarse-level fragility the full-res probe cannot (finest curation was
    // null; the score dilution lives at the coarse T=8 level).
    for (auto& lv : out.levels) {
        int L = lv.level;
        std::vector<cv::Mat> maps;
        for (int m=0;m<MC;++m){
            cv::Mat img; cv::resize(tmpl,img,cv::Size(),mild,mild,cv::INTER_AREA);
            cv::Mat n(img.size(),CV_8U); rng.fill(n,cv::RNG::NORMAL,0,sigma); cv::add(img,n,img);
            for (int p=0;p<L;++p) cv::pyrDown(img,img);   // to this level's resolution
            maps.push_back(votedBinMap(img));
        }
        std::vector<sbm::FeatureSet::Feature> keep;
        for (auto& f : lv.features) {
            float ax=(f.x+lv.tl_x)*mild, ay=(f.y+lv.tl_y)*mild; int hit=0;
            for (auto& mp : maps){ int b=binAt(mp,ax,ay); if(b>=0&&bindiff(b,f.label)==0)hit++; }
            if ((float)hit/MC >= thresh) keep.push_back(f); else dropped++;
        }
        if (!keep.empty()) lv.features = keep;   // never empty a level
    }
    return out;
}

static cv::Mat builtin(int which){ cv::Mat m(160,160,CV_8U,cv::Scalar(40)); const int c=80;
    if(which==3){cv::rectangle(m,{35,35},{70,130},220,cv::FILLED);cv::rectangle(m,{35,95},{130,130},220,cv::FILLED);}
    else{std::vector<cv::Point> st;for(int i=0;i<10;i++){double a=CV_PI/2+i*CV_PI/5;double r=(i&1)?24:54;
        st.push_back({(int)(c+r*cos(a)),(int)(c-r*sin(a))});}cv::polylines(m,st,true,225,3);} return m; }

struct TC { cv::Mat scene; float cx,cy,scale; };
static TC make_case(const cv::Mat& t, float s, float ang, double sig, int W, int H, cv::RNG& rng){
    cv::Mat sc; cv::resize(t,sc,cv::Size(),s,s,s<1?cv::INTER_AREA:cv::INTER_LINEAR);
    cv::Mat R=cv::getRotationMatrix2D({sc.cols/2.f,sc.rows/2.f},ang,1.0), rot;
    cv::warpAffine(sc,rot,R,sc.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(40));
    cv::Mat scene(H,W,CV_8U,cv::Scalar(40)); int px=(W-rot.cols)/2,py=(H-rot.rows)/2;
    rot.copyTo(scene(cv::Rect(px,py,rot.cols,rot.rows)));
    cv::Mat n(H,W,CV_8U); rng.fill(n,cv::RNG::NORMAL,0,sig); cv::add(scene,n,scene);
    return {scene, px+rot.cols/2.f, py+rot.rows/2.f, s};
}

// Best correct-position score for a matcher on a case (0 if not found).
static float bestScore(sbm::ShapeMatcher& m, const TC& tc, float tol){
    float best=-1; for(auto& r: m.match(tc.scene))
        if(std::abs(r.x-tc.cx)<tol && std::abs(r.y-tc.cy)<tol) best=std::max(best,r.score);
    return best;
}

int main(int argc,char**argv){
    sbm::setLogLevel(sbm::LogLevel::Warning);
    cv::Mat tmpl; std::string arg=argc>1?argv[1]:"";
    if(!arg.empty()&&arg.find_first_not_of("0123456789")!=std::string::npos) tmpl=cv::imread(arg,cv::IMREAD_GRAYSCALE);
    if(tmpl.empty()) tmpl=builtin(arg.empty()?4:std::atoi(arg.c_str()));

    const int nf=128, W=1280,H=960; const double sigma=12.0; const float tol=12.f, SPLIT=0.8f;
    sbm::FeatureSet full = sbm::extractFeatures(tmpl, cv::Mat(), nf);
    int dropped=0; sbm::FeatureSet robust = curate(full, tmpl, 0.70f, dropped);
    int fcount = full.levels.empty()?0:(int)full.levels[0].features.size();
    int rcount = robust.levels.empty()?0:(int)robust.levels[0].features.size();
    std::printf("scale_curation_ab | template %dx%d | nf=%d | robust subset: %d/%d kept (%d fragile dropped)\n\n",
                tmpl.cols,tmpl.rows,nf,rcount,fcount,dropped);

    auto mkcfg=[&](){ sbm::MatchConfig c; c.min_score=30; c.refine=sbm::RefineMode::None;
        c.skip_voting=true; c.match_scale=1.0f; return c; };
    sbm::ModelConfig ang; ang.angle={0,360,4};
    auto rng_ss=[&](float a,float b){ sbm::ScaleRange s; s.min=a;s.max=b;s.step=0.1f; return s; };

    // A: full, all scales (baseline — native extract, analytic scale variants)
    sbm::ShapeMatcher A(mkcfg()); { sbm::ModelConfig m=ang; m.scale=rng_ss(0.6f,1.4f); A.addModel("m",full,m); }
    // B: finest-level curation < SPLIT, full >= SPLIT (scale-adaptive curation)
    sbm::ShapeMatcher B(mkcfg());
    { sbm::ModelConfig lo=ang; lo.scale=rng_ss(0.6f,SPLIT-0.001f); B.addModel("lo",robust,lo);
      sbm::ModelConfig hi=ang; hi.scale=rng_ss(SPLIT,1.4f);        B.addModel("hi",full,hi); }
    // R: REX — the user's heuristic as a per-variant scheme. Low-scale band uses
    // features RE-EXTRACTED from a 0.7x-downscaled + blurred template (orientation
    // defined AT a coarse resolution), full features for the high band. The
    // re-extracted template is 0.7x original, so to cover original scale s its
    // variant scale = s/0.7 (original [0.6,0.8] -> variant [0.857,1.143]).
    cv::Mat t07; cv::resize(tmpl,t07,cv::Size(),0.7,0.7,cv::INTER_AREA);
    cv::GaussianBlur(t07,t07,cv::Size(3,3),0);
    sbm::FeatureSet fx = sbm::extractFeatures(t07, cv::Mat(), nf);
    int xcount = fx.levels.empty()?0:(int)fx.levels[0].features.size();
    std::printf("REX low-band: re-extracted %d features from 0.7x+blur template\n\n", xcount);
    sbm::ShapeMatcher C(mkcfg());
    { sbm::ModelConfig lo=ang; lo.scale={0.6f/0.7f, (SPLIT-0.001f)/0.7f, 0.1f}; C.addModel("lo",fx,lo);
      sbm::ModelConfig hi=ang; hi.scale=rng_ss(SPLIT,1.4f);                     C.addModel("hi",full,hi); }

    const float scales[]={0.63f,0.71f,0.85f,1.15f,1.35f};
    const float angles[]={17,88,163,251,320};
    std::printf("worst-case correct-match score per scale (over %zu angles, sigma=%.0f):\n",
                sizeof(angles)/sizeof(float),sigma);
    std::printf("  %-7s | %-18s %-18s %-18s\n","scale","A baseline","B curation","R rex(0.7+blur)");
    std::printf("  %-7s | %-8s %-8s   %-8s %-8s   %-8s %-8s\n","","worst","mean","worst","mean","worst","mean");
    std::printf("  -------------------------------------------------------------------------\n");
    for(float s: scales){
        float wa=1e9,wb=1e9,wc=1e9,ma=0,mb=0,mc=0; int na=0,nb=0,nc=0;
        cv::RNG rng(9001);
        for(float a: angles){ TC tc=make_case(tmpl,s,a,sigma,W,H,rng);
            float xa=bestScore(A,tc,tol), xb=bestScore(B,tc,tol), xc=bestScore(C,tc,tol);
            if(xa>=0){wa=std::min(wa,xa);ma+=xa;na++;} if(xb>=0){wb=std::min(wb,xb);mb+=xb;nb++;}
            if(xc>=0){wc=std::min(wc,xc);mc+=xc;nc++;} }
        auto W_=[&](float w,int n){return n?w:0.f;};
        std::printf("  %-7.2f | %-8.1f %-8.1f   %-8.1f %-8.1f   %-8.1f %-8.1f\n",
            s, W_(wa,na),na?ma/na:0, W_(wb,nb),nb?mb/nb:0, W_(wc,nc),nc?mc/nc:0);
    }
    std::printf("\n(worst = lowest correct-match score across angles at that scale; 0 = a miss.\n"
                " Prediction: B raises worst@downscale vs A without hurting upscale; C helps\n"
                " downscale but hurts upscale — showing curation must be scale-ADAPTIVE.)\n");
    return 0;
}
