// rot_blur_extract_ab.cpp — does ROTATIONAL-MOTION-BLUR extraction give a more
// rotation-robust feature set for elongated objects? (The rotation analog of the
// scale-side "downscale+blur+re-extract" rex trick.)
//
// Rotational motion blur about the object center = average the template over a
// small ±delta rotation. Each surviving feature's gradient becomes the
// tangentially-averaged (rotation-stable) orientation — an IMAGE-DOMAIN version
// of selectRotationStable that changes orientation VALUES, not just subsets. It
// was never tested by the rotation negative-result benches (those did per-angle
// real extraction / voting), so it is a genuinely open variant.
//
// Two hypotheses, two scene conditions:
//   CLEAN scene      : does blur-extract hold detection at OFF-GRID angles under
//                      a COARSE angle step better than normal extract? (pure
//                      robustness — averaged orientation vs periphery smear tradeoff)
//   MOTION-BLUR scene: if the SCENE object is itself rotationally motion-blurred
//                      (spinning part / conveyor), does a blur-extracted template
//                      match it better? (domain matching — expected win)
//
//   ./rot_blur_extract_ab            # elongated arrow, coarse step 10, blur ±5
//   ./rot_blur_extract_ab 12 6       # coarse step 12, blur half-angle 6

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstdio>
#include <vector>

// Elongated, asymmetric arrow of total length ~len, centered in a square canvas.
// Larger len => the tip sits at a larger radius, so a 1-deg angle-step error
// swings it by more pixels (r * dtheta) — the size-dependent effect under test.
static cv::Mat elongated(int len=120) {
    int S = (int)(len*1.35);                 // canvas
    cv::Mat m(S,S,CV_8U,cv::Scalar(40));
    int cx=S/2, cy=S/2, half=len/2, w=std::max(4,len/16);
    cv::rectangle(m, {cx-w, cy-half+len/6}, {cx+w, cy+half}, 220, cv::FILLED);  // shaft
    int hw=w*3, hh=len/6;                     // arrowhead
    std::vector<cv::Point> head{{cx, cy-half}, {cx-hw, cy-half+hh}, {cx+hw, cy-half+hh}};
    cv::fillConvexPoly(m, head, 220);
    return m;
}

// Rotational motion blur about the image center: average `steps` rotated copies
// spanning [-half, +half] degrees. Periphery smears more (arc = r*delta).
static cv::Mat rotBlur(const cv::Mat& img, float half_deg, int steps=9) {
    cv::Point2f c(img.cols/2.f, img.rows/2.f);
    cv::Mat acc(img.size(), CV_32F, cv::Scalar(0));
    for (int i=0;i<steps;++i){ float a = -half_deg + 2*half_deg*i/(steps-1);
        cv::Mat R=cv::getRotationMatrix2D(c,a,1.0), r; cv::warpAffine(img,r,R,img.size(),
            cv::INTER_LINEAR, cv::BORDER_REPLICATE); cv::Mat rf; r.convertTo(rf,CV_32F); acc+=rf; }
    acc/=steps; cv::Mat out; acc.convertTo(out,CV_8U); return out;
}

struct TC{ cv::Mat scene; float cx,cy; };
// Place the template at angle `ang`, scale 1.0; optionally motion-blur the object
// (scene-side rotational blur); add noise.
static TC make_case(const cv::Mat& t, float ang, float scene_blur, double sig,
                    int W, int H, cv::RNG& rng) {
    cv::Mat R=cv::getRotationMatrix2D({t.cols/2.f,t.rows/2.f},ang,1.0), rot;
    cv::warpAffine(t,rot,R,t.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(40));
    if (scene_blur>0.f) rot = rotBlur(rot, scene_blur);
    cv::Mat scene(H,W,CV_8U,cv::Scalar(40)); int px=(W-rot.cols)/2,py=(H-rot.rows)/2;
    rot.copyTo(scene(cv::Rect(px,py,rot.cols,rot.rows)));
    cv::Mat n(H,W,CV_8U); rng.fill(n,cv::RNG::NORMAL,0,sig); cv::add(scene,n,scene);
    return {scene, px+rot.cols/2.f, py+rot.rows/2.f};
}
static float bestScore(sbm::ShapeMatcher&m,const TC&tc,float tol){ float best=-1;
    for(auto&r:m.match(tc.scene)) if(std::abs(r.x-tc.cx)<tol&&std::abs(r.y-tc.cy)<tol) best=std::max(best,r.score);
    return best; }

struct Stat{ int det,n; float hi,lo,mean; };
static Stat run(sbm::ShapeMatcher&m,const cv::Mat&t,const std::vector<float>&angs,float scene_blur,
                double sig,int W,int H,float tol,float mins){ cv::RNG rng(2024);
    int det=0; float hi=0,lo=1e9,sum=0;
    for(float a:angs){ TC tc=make_case(t,a,scene_blur,sig,W,H,rng); float x=bestScore(m,tc,tol);
        if(x>=mins){det++;hi=std::max(hi,x);lo=std::min(lo,x);sum+=x;} }
    return {det,(int)angs.size(),det?hi:0.f,det?lo:0.f,det?sum/det:0.f}; }

int main(int argc,char**argv){
    sbm::setLogLevel(sbm::LogLevel::Warning);
    float coarse = argc>1?(float)std::atof(argv[1]):4.f;
    int   objlen = argc>2?std::atoi(argv[2]):120;
    // Blur half-angle defaults to step/2 — just enough to cover the off-grid
    // angular swing the discrete angle step leaves uncovered.
    float halfb  = argc>3?(float)std::atof(argv[3]):coarse/2.f;
    const int nf=128; const double sigma=12.0; const float tol=12.f, mins=55.f;

    cv::Mat tmpl = elongated(objlen);
    const int W=std::max(1280,(int)(tmpl.cols*1.6)), H=std::max(960,(int)(tmpl.rows*1.6));
    float tipR = objlen/2.f;                                   // tip radius
    float swing = tipR * (coarse/2.f) * (float)CV_PI/180.f;    // tip px error at worst off-grid angle
    cv::Mat tblur = rotBlur(tmpl, halfb);
    sbm::FeatureSet fN = sbm::extractFeatures(tmpl,  cv::Mat(), nf);   // normal
    sbm::FeatureSet fB = sbm::extractFeatures(tblur, cv::Mat(), nf);   // rot-blur extract
    int nN=fN.levels.empty()?0:(int)fN.levels[0].features.size();
    int nB=fB.levels.empty()?0:(int)fB.levels[0].features.size();

    std::printf("rot_blur_extract_ab | arrow len=%d (canvas %dx%d) | angle step=%.1f, "
                "rot-blur half=%.1f | features N=%d B=%d\n", objlen,tmpl.cols,tmpl.rows,coarse,halfb,nN,nB);
    std::printf("tip radius=%.0fpx -> worst off-grid tip swing = r*(step/2) = %.1f px "
                "(scene %dx%d, sigma=%.0f, min_score=%.0f)\n", tipR, swing, W, H, sigma, mins);

    auto cfg=[&](){ sbm::MatchConfig c;c.min_score=30;c.refine=sbm::RefineMode::None;
        c.skip_voting=true;c.match_scale=1.0f;return c; };
    auto mk=[&](const sbm::FeatureSet&fs,float step){ auto m=std::make_shared<sbm::ShapeMatcher>(cfg());
        sbm::ModelConfig mc; mc.angle={0,360,step}; mc.scale={1.0f,1.0f,0.1f}; m->addModel("m",fs,mc); return m; };

    auto A     = mk(fN, coarse);   // basic: normal extract, coarse step
    auto B     = mk(fB, coarse);   // rot-blur extract, coarse step
    auto Rref  = mk(fN, 2.0f);     // reference: normal extract, FINE step (gold)

    // Off-grid angles (avoid multiples of the coarse step).
    std::vector<float> angs={7,23,49,71,113,146,187,229,271,317,353};

    auto show=[&](const char* cond, float scene_blur){
        Stat a=run(*A,tmpl,angs,scene_blur,sigma,W,H,tol,mins);
        Stat b=run(*B,tmpl,angs,scene_blur,sigma,W,H,tol,mins);
        Stat r=run(*Rref,tmpl,angs,scene_blur,sigma,W,H,tol,mins);
        std::printf("\n[%s]\n", cond);
        std::printf("  %-26s %-6s %-6s %-6s %-6s\n","config","det","hi","lo","mean");
        std::printf("  A normal      (step %2.0f)     %d/%-2d  %5.1f %5.1f %5.1f\n",coarse,a.det,a.n,a.hi,a.lo,a.mean);
        std::printf("  B rot-blur ext(step %2.0f)     %d/%-2d  %5.1f %5.1f %5.1f\n",coarse,b.det,b.n,b.hi,b.lo,b.mean);
        std::printf("  ref normal    (step  2, gold) %d/%-2d  %5.1f %5.1f %5.1f\n",r.det,r.n,r.hi,r.lo,r.mean);
    };
    show("CLEAN scene (no motion blur) — pure rotation robustness at coarse step", 0.f);
    show("MOTION-BLUR scene (object rot-blurred ±", halfb);   // domain-matching case

    // ---- angle-step sweep (normal extract): does a finer step fix the tip swing? --
    std::printf("\n[angle-step sweep, normal extract, CLEAN scene] tip radius=%.0fpx:\n", tipR);
    std::printf("  %-6s %-11s %-7s %-6s %-6s %-8s\n","step","tip swing","det","hi","lo","#templ");
    for (float st : {8.f,4.f,2.f,1.5f,1.f}) {
        auto M = mk(fN, st);
        Stat s = run(*M, tmpl, angs, 0.f, sigma, W, H, tol, mins);
        float sw = tipR * (st/2.f) * (float)CV_PI/180.f;
        std::printf("  %-6.1f %-6.1f px   %d/%-2d  %5.1f %5.1f %-8d\n",
                    st, sw, s.det, s.n, s.hi, s.lo, M->numTemplates());
    }
    std::printf("  (finer step shrinks the tip swing r*(step/2); worst-case lo recovers when\n"
                "   the swing drops under the coarse-grid tolerance. Cost = template count.)\n");
    std::printf("\n(det = detections >= min_score over %zu off-grid angles; hi/lo = best/worst\n"
                " correct-match score. A vs B isolates rot-blur EXTRACTION; the gold ref shows\n"
                " what a fine angle step achieves. CLEAN tests pure robustness; MOTION-BLUR tests\n"
                " train/test domain matching.)\n", angs.size());
    return 0;
}
