// auto_tune_featureset.cpp — automatic feature-set / scale-band tuner.
//
// Chains the root-cause findings into a predict->verify pipeline:
//
//  TIER 1 (predict, no matcher — milliseconds): build the per-feature
//    scale-space orientation-stability signature (recovery under noise, each
//    pyramid level at its OWN resolution). From it derive
//      - a CANDIDATE scale floor (where the dead-feature fraction crosses a
//        dilution budget — dead features contribute ~0 to score=raw/(4*numFeat)),
//      - a num_features suggestion (the stable core), and
//      - a curated feature set (drop the fragile tail; auto-adaptive: a solid
//        shape drops nothing).
//
//  TIER 2 (verify/refine — small matcher sweep): on alter-template ground truth
//    (scale x angle x noise), FIND the real floor = the lowest scale whose
//    WORST-CASE correct-match score still clears the target, and A/B the tuned
//    config (curated + floor) against a naive wide band. Emit the config +
//    templates saved.
//
//   ./auto_tune_featureset            # thin star (4)   ./auto_tune_featureset 3  # solid
//   ./auto_tune_featureset my.png

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using Clk = std::chrono::high_resolution_clock;
static double ms_since(Clk::time_point t){ return std::chrono::duration<double,std::milli>(Clk::now()-t).count(); }

static const int   NEIGHBOR_THRESHOLD = 5;
static const int   BLUR_KSIZE = 7;
static const float WEAK_THRESH = 30.f;

static inline int rs_qbin(float gxf, float gyf){ int gx=(int)std::lround(gxf),gy=(int)std::lround(gyf);
    if(gy<0){gx=-gx;gy=-gy;} if(gy==0&&gx<0)gx=-gx; static const long T[4]={1989,6682,14966,50273};
    long ty=(long)gy*10000;int b; if(gx>=0){if(ty<(long)gx*T[0])b=0;else if(ty<(long)gx*T[1])b=1;
        else if(ty<(long)gx*T[2])b=2;else if(ty<(long)gx*T[3])b=3;else b=4;}
    else{long a=-gx;if(ty<a*T[0])b=0;else if(ty<a*T[1])b=7;else if(ty<a*T[2])b=6;else if(ty<a*T[3])b=5;else b=4;} return b; }
static inline int bindiff(int a,int b){int d=std::abs(a-b);return std::min(d,8-d);}

static cv::Mat votedBinMap(const cv::Mat& gray){
    cv::Mat sm,gx,gy; cv::GaussianBlur(gray,sm,cv::Size(BLUR_KSIZE,BLUR_KSIZE),0);
    cv::Sobel(sm,gx,CV_32F,1,0,3); cv::Sobel(sm,gy,CV_32F,0,1,3);
    cv::Mat raw(gray.size(),CV_8S,cv::Scalar(-1)),mag(gray.size(),CV_32F);
    for(int r=0;r<gray.rows;++r){const float*a=gx.ptr<float>(r);const float*b=gy.ptr<float>(r);
        schar*rr=raw.ptr<schar>(r);float*mr=mag.ptr<float>(r);
        for(int c=0;c<gray.cols;++c){mr[c]=std::sqrt(a[c]*a[c]+b[c]*b[c]);rr[c]=(schar)rs_qbin(a[c],b[c]);}}
    cv::Mat out(gray.size(),CV_8S,cv::Scalar(-1));
    for(int r=1;r<gray.rows-1;++r){const float*mr=mag.ptr<float>(r);schar*o=out.ptr<schar>(r);
        for(int c=1;c<gray.cols-1;++c){if(mr[c]<=WEAK_THRESH)continue;int h[8]={0};
            for(int dr=-1;dr<=1;++dr){const schar*q=raw.ptr<schar>(r+dr);for(int dc=-1;dc<=1;++dc)h[q[c+dc]]++;}
            int mv=0,idx=-1;for(int i=0;i<8;i++)if(h[i]>mv){mv=h[i];idx=i;} if(mv>=NEIGHBOR_THRESHOLD)o[c]=(schar)idx;}}
    return out; }
static inline int binAt(const cv::Mat&m,float x,float y){int xi=(int)std::lround(x),yi=(int)std::lround(y);
    if(xi<0||yi<0||xi>=m.cols||yi>=m.rows)return -1;return m.at<schar>(yi,xi);}

// Per-feature orientation recovery under noise at scale s, each pyramid level at
// its own resolution (pyrDown^level of the scaled+noised template; probe pos =
// (x+tl)*s for every level). Returns a flat recovery vector (0..1) parallel to a
// flat feature list, plus the mean.
struct FeatRef { int level, idx; };
static std::vector<float> recoveryAtScale(const sbm::FeatureSet& fs, const cv::Mat& tmpl,
                                          float s, double sigma, int MC, cv::RNG& rng,
                                          std::vector<FeatRef>* refs=nullptr) {
    std::vector<float> rec;
    for (size_t L=0; L<fs.levels.size(); ++L) {
        const auto& lv = fs.levels[L]; int lvl = lv.level;
        std::vector<cv::Mat> maps;
        for (int m=0;m<MC;++m){ cv::Mat img; cv::resize(tmpl,img,cv::Size(),s,s,cv::INTER_AREA);
            cv::Mat n(img.size(),CV_8U); rng.fill(n,cv::RNG::NORMAL,0,sigma); cv::add(img,n,img);
            for (int p=0;p<lvl;++p) cv::pyrDown(img,img); maps.push_back(votedBinMap(img)); }
        for (size_t i=0;i<lv.features.size();++i){ const auto&f=lv.features[i];
            float ax=(f.x+lv.tl_x)*s, ay=(f.y+lv.tl_y)*s; int hit=0;
            for (auto&mp:maps){ int b=binAt(mp,ax,ay); if(b>=0&&bindiff(b,f.label)==0)hit++; }
            rec.push_back((float)hit/MC); if(refs) refs->push_back({(int)L,(int)i}); }
    }
    return rec;
}

// Curated copy: drop features whose recovery at the operating scale < thresh.
static sbm::FeatureSet curateAt(const sbm::FeatureSet& fs, const cv::Mat& tmpl,
                                float s, float thresh, double sigma, int MC, int& dropped) {
    cv::RNG rng(777); std::vector<FeatRef> refs;
    auto rec = recoveryAtScale(fs, tmpl, s, sigma, MC, rng, &refs);
    sbm::FeatureSet out = fs; dropped = 0;
    std::vector<std::vector<sbm::FeatureSet::Feature>> keep(out.levels.size());
    for (size_t k=0;k<refs.size();++k){ auto&r=refs[k];
        if (rec[k]>=thresh) keep[r.level].push_back(fs.levels[r.level].features[r.idx]); else dropped++; }
    for (size_t L=0;L<out.levels.size();++L) if(!keep[L].empty()) out.levels[L].features=keep[L];
    return out;
}

static cv::Mat builtin(int which){ cv::Mat m(160,160,CV_8U,cv::Scalar(40)); const int c=80;
    if(which==3){cv::rectangle(m,{35,35},{70,130},220,cv::FILLED);cv::rectangle(m,{35,95},{130,130},220,cv::FILLED);}
    else if(which==0){cv::rectangle(m,{30,30},{130,130},230,3);cv::line(m,{30,30},{130,130},180,2);
        cv::line(m,{130,30},{30,130},180,2);cv::rectangle(m,{55,55},{105,105},120,2);}
    else{std::vector<cv::Point> st;for(int i=0;i<10;i++){double a=CV_PI/2+i*CV_PI/5;double r=(i&1)?24:54;
        st.push_back({(int)(c+r*cos(a)),(int)(c-r*sin(a))});}cv::polylines(m,st,true,225,3);} return m; }

struct TC{ cv::Mat scene; float cx,cy; };
static TC make_case(const cv::Mat&t,float s,float ang,double sig,int W,int H,cv::RNG&rng){
    cv::Mat sc;cv::resize(t,sc,cv::Size(),s,s,s<1?cv::INTER_AREA:cv::INTER_LINEAR);
    cv::Mat R=cv::getRotationMatrix2D({sc.cols/2.f,sc.rows/2.f},ang,1.0),rot;
    cv::warpAffine(sc,rot,R,sc.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(40));
    cv::Mat scene(H,W,CV_8U,cv::Scalar(40));int px=(W-rot.cols)/2,py=(H-rot.rows)/2;
    rot.copyTo(scene(cv::Rect(px,py,rot.cols,rot.rows)));
    cv::Mat n(H,W,CV_8U);rng.fill(n,cv::RNG::NORMAL,0,sig);cv::add(scene,n,scene);
    return {scene,px+rot.cols/2.f,py+rot.rows/2.f}; }
static float bestScore(sbm::ShapeMatcher&m,const TC&tc,float tol){ float best=-1;
    for(auto&r:m.match(tc.scene)) if(std::abs(r.x-tc.cx)<tol&&std::abs(r.y-tc.cy)<tol) best=std::max(best,r.score);
    return best; }
// detect count + hi/lo/mean correct-match score over a set of angles at scale s.
struct Stat{ int det,n; float hi,lo,mean; };
static Stat statAt(sbm::ShapeMatcher&m,const cv::Mat&t,float s,const std::vector<float>&angs,
                   double sig,int W,int H,float tol,float mins){ cv::RNG rng(4242);
    int det=0; float hi=0,lo=1e9,sum=0;
    for(float a:angs){ TC tc=make_case(t,s,a,sig,W,H,rng); float x=bestScore(m,tc,tol);
        if(x>=mins){ det++; hi=std::max(hi,x); lo=std::min(lo,x); sum+=x; } }
    return {det,(int)angs.size(),det?hi:0.f,det?lo:0.f,det?sum/det:0.f}; }
// worst correct-score over a set of angles at scale s (0 = a miss on some angle).
static float worstAt(sbm::ShapeMatcher&m,const cv::Mat&t,float s,const std::vector<float>&angs,
                     double sig,int W,int H,float tol){ cv::RNG rng(9001); float worst=1e9;
    for(float a:angs){ TC tc=make_case(t,s,a,sig,W,H,rng); float x=bestScore(m,tc,tol);
        worst=std::min(worst, x<0?0.f:x); } return worst; }

int main(int argc,char**argv){
    sbm::setLogLevel(sbm::LogLevel::Warning);
    cv::Mat tmpl; std::string arg=argc>1?argv[1]:"";
    if(!arg.empty()&&arg.find_first_not_of("0123456789")!=std::string::npos) tmpl=cv::imread(arg,cv::IMREAD_GRAYSCALE);
    if(tmpl.empty()) tmpl=builtin(arg.empty()?4:std::atoi(arg.c_str()));

    const int nf=128, W=1280,H=960, MC=16;
    const double sigma = argc>2 ? std::atof(argv[2]) : 12.0;   // deployment noise (tuner adapts)
    const float tol=12.f;
    const float MIN_TARGET=60.f;          // worst-case score the band must hold
    const float DEAD=0.4f, TAU=0.70f, BUDGET=0.12f;  // dilution budget / curation gate

    sbm::FeatureSet full = sbm::extractFeatures(tmpl, cv::Mat(), nf);
    std::printf("auto_tune_featureset | template %dx%d | nf=%d | noise sigma=%.0f\n",
                tmpl.cols,tmpl.rows,nf,sigma);

    // ---- TIER 1: predictive signature ---------------------------------------
    auto t1 = Clk::now();
    std::printf("\n[Tier 1] scale-space stability signature (predict):\n");
    std::printf("  %-7s %-11s %-11s\n","scale","mean recov","dead frac(<0.4)");
    const float scan[]={1.0f,0.9f,0.8f,0.7f,0.6f,0.5f};
    float cand_floor = 0.5f; bool set=false;
    for (float s: scan){ cv::RNG rng((unsigned)(s*1000)+1);
        auto rec = recoveryAtScale(full,tmpl,s,sigma,MC,rng);
        double sum=0; int dead=0; for(float v:rec){sum+=v; if(v<DEAD)dead++;}
        float meanr=(float)(sum/rec.size()), df=(float)dead/rec.size();
        std::printf("  %-7.2f %-11.2f %.0f%%\n", s, meanr, df*100);
        if(!set && df>BUDGET){ cand_floor=s+0.1f; set=true; } }   // last scale under budget
    std::printf("  -> candidate scale floor (dead-frac budget %.0f%%): %.2f\n", BUDGET*100, cand_floor);

    // curated set + num_features suggestion at the candidate floor
    int dropped=0; sbm::FeatureSet curated = curateAt(full,tmpl,cand_floor,TAU,sigma,MC,dropped);
    int fc = full.levels.empty()?0:(int)full.levels[0].features.size();
    int cc = curated.levels.empty()?0:(int)curated.levels[0].features.size();
    std::printf("  curated finest level: %d/%d kept (%d fragile dropped @ s=%.2f)\n",
                cc,fc,dropped,cand_floor);

    double t1_ms = ms_since(t1);
    // ---- TIER 2: verify/refine the floor on ground truth --------------------
    auto t2 = Clk::now();
    std::printf("\n[Tier 2] verify on alter-template ground truth (find real floor):\n");
    auto mkcfg=[&](){ sbm::MatchConfig c;c.min_score=30;c.refine=sbm::RefineMode::None;
        c.skip_voting=true;c.match_scale=1.0f;return c; };
    sbm::ModelConfig ang; ang.angle={0,360,4};
    std::vector<float> angs={17,88,163,251,320};

    // Self-consistent floor search: for each candidate floor f, build the ACTUAL
    // banded matcher [f,1.4] and stress it at its smallest scale f (the hardest
    // point in that band). Floor = lowest f whose worst-case there clears target.
    std::printf("  %-7s %-16s\n","floor f","worst @ f (band [f,1.4])");
    float real_floor=0.9f;
    for (float f: {0.9f,0.8f,0.7f,0.6f,0.5f}){
        sbm::ShapeMatcher M(mkcfg()); sbm::ModelConfig m=ang; m.scale={f,1.4f,0.1f}; M.addModel("m",curated,m);
        float w=worstAt(M,tmpl,f,angs,sigma,W,H,tol);
        std::printf("  %-7.2f %-16.1f %s\n", f, w, w>=MIN_TARGET?"OK":"below target");
        if (w>=MIN_TARGET) real_floor=f; }
    std::printf("  -> real floor (worst >= %.0f): %.2f  (Tier-1 predicted %.2f)\n",
                MIN_TARGET, real_floor, cand_floor);

    // A/B: naive wide band [0.5,1.4] full vs tuned [real_floor,1.4] curated
    sbm::ShapeMatcher N(mkcfg()); int nt; { sbm::ModelConfig m=ang; m.scale={0.5f,1.4f,0.1f}; nt=N.addModel("m",full,m); }
    sbm::ShapeMatcher B(mkcfg()); int bt; { sbm::ModelConfig m=ang; m.scale={real_floor,1.4f,0.1f}; bt=B.addModel("m",curated,m); }

    // ---- rotation x downscale matching test: hi/lo, basic vs tuned ----------
    std::printf("\n[Rotation x downscale test] off-grid angles {13,47,101,163,229,293,341} x scales,\n");
    std::printf("  sigma=%.0f, min_score=%.0f.  basic = full features, wide band [0.5,1.4];\n", sigma, MIN_TARGET);
    std::printf("  tuned = curated features, auto floor [%.2f,1.4].\n\n", real_floor);
    std::vector<float> rangs={13,47,101,163,229,293,341};
    std::printf("  %-7s | %-22s | %-22s\n","scale","BASIC  det  hi    lo","TUNED  det  hi    lo");
    std::printf("  --------+------------------------+------------------------\n");
    const float tscales[]={0.55f,0.63f,0.71f,0.85f,1.00f,1.20f,1.35f};
    Stat bAll{0,0,0,1e9f,0}, tAll{0,0,0,1e9f,0}; float bhiA=0,thiA=0,bloA=1e9,tloA=1e9;
    for (float s: tscales){
        Stat bs=statAt(N,tmpl,s,rangs,sigma,W,H,tol,MIN_TARGET);
        Stat ts=statAt(B,tmpl,s,rangs,sigma,W,H,tol,MIN_TARGET);
        std::printf("  %-7.2f | %2d/%-2d  %5.1f %5.1f    | %2d/%-2d  %5.1f %5.1f\n",
            s, bs.det,bs.n,bs.hi,bs.lo, ts.det,ts.n,ts.hi,ts.lo);
        bAll.det+=bs.det; bAll.n+=bs.n; tAll.det+=ts.det; tAll.n+=ts.n;
        if(bs.det){bhiA=std::max(bhiA,bs.hi);bloA=std::min(bloA,bs.lo);}
        if(ts.det){thiA=std::max(thiA,ts.hi);tloA=std::min(tloA,ts.lo);}
    }
    std::printf("  --------+------------------------+------------------------\n");
    std::printf("  overall | %2d/%-2d  %5.1f %5.1f    | %2d/%-2d  %5.1f %5.1f\n",
        bAll.det,bAll.n,bhiA,bloA==1e9?0:bloA, tAll.det,tAll.n,thiA,tloA==1e9?0:tloA);
    std::printf("  (det = detections >= min_score across the 7 angles; hi/lo = best/worst\n"
                "   correct-match score among them; 0 = none cleared min_score at that scale.)\n");

    std::printf("\n=== RECOMMENDED ===\n");
    std::printf("  scale range   : [%.2f, 1.40] step 0.10   (naive floor 0.50 wasted %d templates below it)\n",
                real_floor, nt-bt);
    std::printf("  num_features  : %d  (curated from %d; %d fragile dropped)\n", cc, fc, dropped);
    std::printf("  curation      : %s (auto-adaptive; solids drop ~0)\n", dropped>0?"ON":"off (no fragile tail)");
    std::printf("  templates     : %d tuned vs %d naive  (%.0f%% fewer)\n", bt, nt, 100.0*(nt-bt)/nt);
    std::printf("  note          : below the floor, no feature treatment recovers the score "
                "(resolution floor) — don't generate those variants.\n");
    std::printf("\n[timing] Tier-1 predict %.0f ms | Tier-2 verify %.0f ms | total %.0f ms\n",
                t1_ms, ms_since(t2), t1_ms + ms_since(t2));
    return 0;
}
