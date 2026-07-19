// feat_robustness_probe.cpp — per-feature scale-space orientation-stability
// signature, and whether it explains feature robustness.
//
// ROOT-CAUSE HYPOTHESIS under test: a feature contributes to the match score
// iff its 8-bin quantized gradient ORIENTATION stays correct at its transformed
// position. So robustness == orientation-label stability across scale-space. The
// extract pipeline already gates on a single-scale 1px version (hysteresisGradient:
// keep a pixel iff >= NEIGHBOR_THRESHOLD of its 3x3 agree on the bin). This probe
// measures each feature's stability across a BLUR/DOWNSCALE/ROTATION sweep and
// tests two predictions:
//   (A) the signature predicts survival under EXTREME downscale (the measured
//       thin-shape cliff at s=0.63), and
//   (B) the "extract at 0.7x + blur, scale features back" heuristic keeps
//       exactly the high-signature features (i.e. it is a scale-space low-pass
//       select in disguise).
//
// Faithful replica of the pipeline's quantizer: GaussianBlur(ksize) -> Sobel3 ->
// rs_qbin (the exact line2Dup 8-bin fold) -> 3x3 majority >= NEIGHBOR_THRESHOLD
// with a magnitude gate. Same numbering as the stored feature label.
//
//   cmake --build build --target feat_robustness_probe
//   ./feat_robustness_probe            # thin star (index 4, the hard case)
//   ./feat_robustness_probe 3          # solid blocks (contrast)
//   ./feat_robustness_probe my.png

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static const int   NEIGHBOR_THRESHOLD = 5;   // == line2Dup.cpp
static const int   BLUR_KSIZE         = 7;   // == default pipeline blur
static const float WEAK_THRESH        = 30.f;

// --- the exact line2Dup 8-bin quantizer (copied from shape_matcher.cpp) -------
static inline int rs_qbin(float gxf, float gyf) {
    int gx = (int)std::lround(gxf), gy = (int)std::lround(gyf);
    if (gy < 0) { gx = -gx; gy = -gy; }
    if (gy == 0 && gx < 0) gx = -gx;
    static const long TAN_B[4] = {1989, 6682, 14966, 50273};
    long ty = (long)gy * 10000; int b;
    if (gx >= 0) {
        if (ty < (long)gx*TAN_B[0]) b=0; else if (ty < (long)gx*TAN_B[1]) b=1;
        else if (ty < (long)gx*TAN_B[2]) b=2; else if (ty < (long)gx*TAN_B[3]) b=3; else b=4;
    } else {
        long a=-gx;
        if (ty < a*TAN_B[0]) b=0; else if (ty < a*TAN_B[1]) b=7;
        else if (ty < a*TAN_B[2]) b=6; else if (ty < a*TAN_B[3]) b=5; else b=4;
    }
    return b;
}
static inline int bindiff(int a, int b) { int d = std::abs(a-b); return std::min(d, 8-d); }

// Voted-bin map of an image, mirroring hysteresisGradient: per pixel the 8-bin
// orientation, but only where mag > WEAK_THRESH and >= NEIGHBOR_THRESHOLD of the
// 3x3 patch share the dominant bin. -1 = no stable feature there.
static cv::Mat votedBinMap(const cv::Mat& gray) {
    cv::Mat sm, gx, gy;
    cv::GaussianBlur(gray, sm, cv::Size(BLUR_KSIZE, BLUR_KSIZE), 0);
    cv::Sobel(sm, gx, CV_32F, 1, 0, 3);
    cv::Sobel(sm, gy, CV_32F, 0, 1, 3);
    cv::Mat raw(gray.size(), CV_8S, cv::Scalar(-1));   // per-pixel bin (pre-vote)
    cv::Mat mag(gray.size(), CV_32F);
    for (int r = 0; r < gray.rows; ++r) {
        const float* gxr = gx.ptr<float>(r); const float* gyr = gy.ptr<float>(r);
        schar* rr = raw.ptr<schar>(r); float* mr = mag.ptr<float>(r);
        for (int c = 0; c < gray.cols; ++c) {
            mr[c] = std::sqrt(gxr[c]*gxr[c] + gyr[c]*gyr[c]);
            rr[c] = (schar)rs_qbin(gxr[c], gyr[c]);
        }
    }
    cv::Mat out(gray.size(), CV_8S, cv::Scalar(-1));
    for (int r = 1; r < gray.rows-1; ++r) {
        const float* mr = mag.ptr<float>(r);
        schar* orow = out.ptr<schar>(r);
        for (int c = 1; c < gray.cols-1; ++c) {
            if (mr[c] <= WEAK_THRESH) continue;
            int hist[8] = {0};
            for (int dr=-1; dr<=1; ++dr) {
                const schar* q = raw.ptr<schar>(r+dr);
                for (int dc=-1; dc<=1; ++dc) hist[q[c+dc]]++;
            }
            int mv=0, idx=-1;
            for (int i=0;i<8;i++) if (hist[i]>mv){mv=hist[i];idx=i;}
            if (mv >= NEIGHBOR_THRESHOLD) orow[c] = (schar)idx;
        }
    }
    return out;
}
static inline int binAt(const cv::Mat& bm, float x, float y) {
    int xi=(int)std::lround(x), yi=(int)std::lround(y);
    if (xi<0||yi<0||xi>=bm.cols||yi>=bm.rows) return -1;
    return bm.at<schar>(yi, xi);
}

struct Feat { float x, y; int label; float cornerness; };

// Native-resolution features at finest level (absolute template coords).
static std::vector<Feat> nativeFeatures(const cv::Mat& tmpl, int nf) {
    sbm::FeatureSet fs = sbm::extractFeatures(tmpl, cv::Mat(), nf);
    std::vector<Feat> out;
    if (fs.levels.empty()) return out;
    auto& lv = fs.levels[0];
    for (auto& f : lv.features)
        out.push_back({(float)(f.x+lv.tl_x), (float)(f.y+lv.tl_y), f.label, f.cornerness});
    return out;
}

int main(int argc, char** argv) {
    sbm::setLogLevel(sbm::LogLevel::Warning);

    // built-in shapes match tune_sweep_scale (4 = thin star, 3 = solid blocks)
    auto builtin = [](int which){
        cv::Mat m(160,160,CV_8U,cv::Scalar(40)); const int c=80;
        if (which==3){ cv::rectangle(m,{35,35},{70,130},220,cv::FILLED);
                       cv::rectangle(m,{35,95},{130,130},220,cv::FILLED); }
        else { std::vector<cv::Point> st; for(int i=0;i<10;i++){double a=CV_PI/2+i*CV_PI/5;
               double r=(i&1)?24:54; st.push_back({(int)(c+r*cos(a)),(int)(c-r*sin(a))});}
               cv::polylines(m,st,true,225,3); }
        return m; };

    cv::Mat tmpl; std::string arg = argc>1?argv[1]:"";
    if (!arg.empty() && arg.find_first_not_of("0123456789")!=std::string::npos)
        tmpl = cv::imread(arg, cv::IMREAD_GRAYSCALE);
    if (tmpl.empty()) tmpl = builtin(arg.empty()?4:std::atoi(arg.c_str()));

    const int nf = 128;
    const double sigma = 12.0;     // == tune_sweep_scale noise
    const int MC = 24;             // Monte-Carlo noise realizations
    auto feats = nativeFeatures(tmpl, nf);
    std::printf("feat_robustness_probe | template %dx%d | %zu native features (nf=%d) | "
                "noise sigma=%.0f, %d MC\n\n", tmpl.cols, tmpl.rows, feats.size(), nf, sigma, MC);
    if (feats.empty()) { std::printf("no features\n"); return 1; }

    // Deterministic noise bank (fixed seed for reproducibility across runs).
    cv::RNG rng(12345);
    auto noisy = [&](const cv::Mat& g){ cv::Mat n(g.size(),CV_8U); rng.fill(n,cv::RNG::NORMAL,0,sigma);
        cv::Mat o; cv::add(g,n,o); return o; };

    // --- CENTREPIECE: orientation-recovery-under-noise vs SCALE ---------------
    // For each downscale s: resize template, then over MC noise realizations
    // compute the voted-bin map and count, per native feature, the fraction of
    // realizations its EXACT 8-bin label is recovered at the scaled position.
    // This is the matcher's actual per-feature contribution condition. The mean
    // over features is the predicted score trend; per-feature it is the
    // robustness signature.
    const float scales[] = {1.0f, 0.85f, 0.75f, 0.63f};
    const int   NS = (int)(sizeof(scales)/sizeof(float));
    std::vector<std::vector<float>> recov(NS, std::vector<float>(feats.size(), 0.f));
    for (int si=0; si<NS; ++si){ float s=scales[si];
        cv::Mat base; cv::resize(tmpl,base,cv::Size(),s,s,cv::INTER_AREA);
        std::vector<int> hit(feats.size(),0);
        for (int m=0;m<MC;++m){ cv::Mat bm=votedBinMap(noisy(base));
            for (size_t i=0;i<feats.size();++i){ int b=binAt(bm, feats[i].x*s, feats[i].y*s);
                if (b>=0 && bindiff(b,feats[i].label)==0) hit[i]++; } }
        for (size_t i=0;i<feats.size();++i) recov[si][i]=(float)hit[i]/MC; }

    std::printf("orientation-recovery-under-noise vs scale (mean over features):\n");
    std::printf("  %-7s %-10s %-s\n","scale","mean recov","distribution lo<.4 / mid / hi>=.8");
    for (int si=0; si<NS; ++si){ double sum=0; int lo=0,mid=0,hi=0;
        for (float v:recov[si]){ sum+=v; if(v<0.4f)lo++; else if(v<0.8f)mid++; else hi++; }
        std::printf("  %-7.2f %-10.2f %d / %d / %d\n", scales[si], sum/feats.size(), lo,mid,hi); }
    std::printf("\n");

    // Signature = the CHEAP predictor: recovery at a mild scale (0.85) under
    // noise. Outcome = recovery at the extreme cliff scale (0.63). If the mild
    // signature predicts the extreme outcome, one cheap probe ranks robustness.
    const int MILD = 1, EXTREME = NS-1;   // indices into scales[] (0.85, 0.63)
    std::vector<float>& sig  = recov[MILD];
    std::vector<float>& out63 = recov[EXTREME];

    // --- (A) does the mild-scale signature predict extreme-scale survival? ----
    //   "survives" = recovered in a majority of noise realizations at 0.63.
    double sigSurv=0, sigDie=0; int nSurv=0, nDie=0;
    for (size_t i=0;i<feats.size();++i){ bool surv = out63[i] >= 0.5f;
        if (surv){ sigSurv+=sig[i]; nSurv++; } else { sigDie+=sig[i]; nDie++; } }
    std::printf("(A) mild-signature (s=0.85) vs extreme survival (s=0.63, recov>=0.5):\n");
    std::printf("    survived: %3d  mean mild-signature %.2f\n", nSurv, nSurv?sigSurv/nSurv:0);
    std::printf("    died    : %3d  mean mild-signature %.2f\n", nDie,  nDie?sigDie/nDie:0);
    std::printf("    -> signature %s predict survival\n\n",
        (nSurv&&nDie&&sigSurv/nSurv - sigDie/nDie > 0.1)?"DOES":"does NOT clearly");

    // --- (B) does the 0.7x+blur heuristic keep the robust (high-recov) feats? --
    cv::Mat t07; cv::resize(tmpl,t07,cv::Size(),0.7,0.7,cv::INTER_AREA);
    cv::GaussianBlur(t07,t07,cv::Size(3,3),0);
    auto f07 = nativeFeatures(t07, nf);
    std::vector<cv::Point2f> kept07;
    for (auto&f:f07) kept07.push_back({f.x/0.7f, f.y/0.7f});
    const float R=4.0f;
    // Score each native feature by its extreme-scale recovery (the real robustness).
    double robKept=0, robDrop=0; int nKept=0, nDrop=0;
    for (size_t i=0;i<feats.size();++i){ auto&f=feats[i]; bool kept=false;
        for (auto&p:kept07) if (std::abs(p.x-f.x)<R && std::abs(p.y-f.y)<R){ kept=true; break; }
        if (kept){ robKept+=out63[i]; nKept++; } else { robDrop+=out63[i]; nDrop++; } }
    std::printf("(B) 0.7x+blur+scale-back heuristic (%zu feats) vs feature robustness (recov@0.63):\n", f07.size());
    std::printf("    native feats it KEEPS  : %3d  mean recov@0.63 %.2f\n", nKept, nKept?robKept/nKept:0);
    std::printf("    native feats it DROPS  : %3d  mean recov@0.63 %.2f\n", nDrop, nDrop?robDrop/nDrop:0);
    std::printf("    -> heuristic %s select for robust features\n",
        (nKept&&nDrop&&robKept/nKept - robDrop/nDrop > 0.05)?"DOES":"does NOT clearly");
    return 0;
}
