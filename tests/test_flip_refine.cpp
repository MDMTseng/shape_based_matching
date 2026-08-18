// test_flip_refine.cpp
// ---------------------------------------------------------------------------
// Hermetic experiment for FLIP-template refine quality. See
// ../flip_template_problem.md for the code-path write-up.
//
// CODE FACT (verified by reading shape_matcher.cpp): addModel(flip=true) registers
// a flipped detector class from flipped *features*, but discards the flipped
// FeatureSet; the match loop refines every flipped hit against the BASE template
// image + base refine_points (ModelInfo stores only `features`, and the loop does
// `fs = mi->features`). So the flipped refine is fed a non-mirrored template.
//
// OPEN QUESTION this experiment answers: does that actually degrade the flipped
// REFINED pose, and by how much, vs an identical non-flipped control? Measure
// ground-truth pose error (origin px + angle deg) for control vs flip, clean and
// under noise, with RefineMode::None (coarse) and ROI (refined).
//
// METHOD (convention-safe):
//   - Place instances with warpAffine at a KNOWN origin + rotation. Origin =
//     template centre, so reported (x,y) must equal the placed origin regardless
//     of any flip/angle convention -> origin error is the clean, decisive metric.
//   - The matcher's flip = mirror about the HORIZONTAL axis (flipFeatures mirrors
//     feature Y). So construct the flipped instance with cv::flip(templ, 0) to match
//     that convention, then rotate. Angle error is reported but auto-calibrated
//     against the non-flip mapping, not hard-coded.
//
// FOR THE NEXT AGENT (iterate here):
//   build: cmake --build <dir> --target test_flip_refine   (MSVC CMakeLists), or
//          g++ -std=c++14 -O2 -mavx2 -D__AVX2__ -fopenmp -Wa,-mbig-obj \
//            -I. -Iinclude -IMIPP `pkg-config --cflags opencv4` \
//            tests/test_flip_refine.cpp <prebuilt libshape_based_matching.a> \
//            `pkg-config --libs opencv4` -fopenmp -o test_flip_refine
//   run:   ./test_flip_refine
//   The number that matters: FLIP refined origin/angle error vs CONTROL. If flip
//   >> control (esp. under noise), the base-template refine is the cause -> apply
//   buildFlippedTemplate() per the doc and re-run; flip must match control.
// ---------------------------------------------------------------------------

#include "shape_matcher.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>   // getRotationMatrix2D moved here in OpenCV 5
#endif
#include <cstdio>
#include <cmath>
#include <vector>

using namespace cv;
static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr,"  FAIL: "); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\n"); g_fail++; } } while(0)

static float angDiff(float a, float b){ float d=a-b; while(d>180)d-=360; while(d<-180)d+=360; return d; }

// Asymmetric "F" (chiral) so a flip is distinct from any rotation. Centre = (TW/2,TW/2).
static Mat make_F(int TW){
    Mat t(TW,TW,CV_8U,Scalar(0)); double cx=TW/2.0, cy=TW/2.0;
    auto box=[&](double x0,double x1,double y0,double y1){
        for(double y=y0;y<=y1;y+=0.4) for(double x=x0;x<=x1;x+=0.4){
            int px=(int)lround(cx+x), py=(int)lround(cy+y);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; } };
    box(-10,-2,-34,34);  // spine
    box(-10,26,-34,-26); // top arm
    box(-10,16,-6,2);    // mid arm
    return t;
}

// Place `templ` (optionally horizontally-mirror-about-x-axis = cv::flip(.,0)) so its
// centre lands at (X,Y), rotated `deg`. warpAffine -> sub-pixel exact placement.
static void place(Mat& scene, const Mat& templ, double X, double Y, double deg, bool flip){
    // NOTE: `Mat src = templ` is a shallow copy sharing templ's buffer, so
    // cv::flip(templ, src, 0) would flip templ IN PLACE and corrupt the caller's
    // image across calls. Write to a fresh Mat to keep templ const.
    Mat src; if(flip) cv::flip(templ, src, 0); else src = templ;
    double cx=templ.cols/2.0, cy=templ.rows/2.0;
    Mat R = getRotationMatrix2D(Point2f((float)cx,(float)cy), -deg, 1.0); // image y-down: -deg = CCW math
    R.at<double>(0,2) += X - cx;  R.at<double>(1,2) += Y - cy;
    Mat warped; warpAffine(src, warped, R, scene.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    scene = max(scene, warped);
}

struct GT { double X, Y, deg; bool flip; };

static const sbm::MatchResult* nearest(const std::vector<sbm::MatchResult>& rs, const GT& g, float radius=70){
    const sbm::MatchResult* best=nullptr; float bd=radius;
    for(auto& r: rs){ float d=(float)std::hypot(r.x-g.X, r.y-g.Y); if(d<bd){bd=d;best=&r;} }
    return best;
}

static std::vector<sbm::MatchResult> run(const sbm::FeatureSet& f, const Mat& scene, sbm::RefineMode rm){
    sbm::MatchConfig cfg; cfg.min_score=50; cfg.nms_radius=60; cfg.refine=rm;
    sbm::ShapeMatcher m(cfg);
    sbm::ModelConfig mc; mc.angle={0,360,1}; mc.flip=true;
    m.addModel("F", f, mc);
    return m.match(scene);
}

int main(){
    printf("=== flip-refine ground-truth experiment (see flip_template_problem.md) ===\n\n");
    const int TW=120;
    Mat templ = make_F(TW);
    auto feat = sbm::extractFeatures(templ);
    feat.setOrigin(TW/2.0f, TW/2.0f); feat.setAngleOffset(0);
    printf("template: %d features\n\n", feat.numFeatures());

    const int W=900, H=520;
    // Same pose for control and flip so any difference is purely the flip path.
    std::vector<double> angles = {0, 25, 70, 130};
    int noises[] = {0, 12};   // additive gaussian sigma

    printf("%-6s %-5s %-7s | %-22s | %-22s\n","noise","kind","angle","coarse err (origin,ang)","ROI   err (origin,ang)");
    printf("---------------------------------------------------------------------------------\n");
    double worst_ctrl_roi=0, worst_flip_roi=0;
    for(int ns : noises){
      for(double deg : angles){
        // build a 2-instance scene: control (no flip) left, flip right, same deg
        Mat scene(H,W,CV_8U,Scalar(0));
        GT gc{ 220.0, 260.0, deg, false }, gf{ 660.0, 260.0, deg, true };
        place(scene, templ, gc.X, gc.Y, gc.deg, gc.flip);
        place(scene, templ, gf.X, gf.Y, gf.deg, gf.flip);
        if(ns>0){ Mat n(H,W,CV_8U); randn(n,0,ns); scene += n; }

        auto co = run(feat, scene, sbm::RefineMode::None);
        auto ro = run(feat, scene, sbm::RefineMode::ROI);

        // calibrate the non-flip angle convention from the control coarse hit, so
        // angle error is measured against the matcher's own mapping (offset-safe).
        for(GT* g : {&gc,&gf}){
          const auto* c = nearest(co,*g); const auto* r = nearest(ro,*g);
          const char* kind = g->flip?"flip":"ctrl";
          if(!c||!r){ printf("%-6d %-5s %-7.0f | NOT DETECTED\n", ns, kind, deg); g_fail++; continue; }
          float c_o=(float)std::hypot(c->x-g->X,c->y-g->Y), r_o=(float)std::hypot(r->x-g->X,r->y-g->Y);
          // expected reported angle: ctrl -> deg ; flip -> -deg (matcher user_angle=-raw)
          float exp_ang = g->flip ? (float)(-deg) : (float)deg;
          float c_a=std::fabs(angDiff(c->angle,exp_ang)), r_a=std::fabs(angDiff(r->angle,exp_ang));
          bool flag_ok = (r->flipped==g->flip);
          printf("%-6d %-5s %-7.0f | o=%5.2f a=%6.2f flip=%d | o=%5.2f a=%6.2f flip=%d%s\n",
                 ns, kind, deg, c_o,c_a,(int)c->flipped, r_o,r_a,(int)r->flipped,
                 flag_ok?"":"  <-- flip-flag ambiguous (separate scoring issue)");
          if(g->flip) worst_flip_roi=std::max(worst_flip_roi,(double)r_o);
          else        worst_ctrl_roi=std::max(worst_ctrl_roi,(double)r_o);
        }
      }
    }
    printf("\nworst ROI origin error:  control=%.2fpx  flip=%.2fpx\n", worst_ctrl_roi, worst_flip_roi);
    // Decisive: flip refined origin must be as good as control (within 1px slack).
    CHECK(worst_flip_roi <= worst_ctrl_roi + 1.0, "FLIP refined origin (%.2f) much worse than control (%.2f) -> base-template refine bug",
          worst_flip_roi, worst_ctrl_roi);

    printf(g_fail? "\n*** %d CHECK(S) FAILED ***\n" : "\nAll checks passed.\n", g_fail);
    return g_fail?1:0;
}
