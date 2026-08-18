// test_flip_separate.cpp
// ---------------------------------------------------------------------------
// Counterpart to test_flip_refine.cpp. Instead of relying on
// ModelConfig.flip=true (which registers a flipped detector class but refines
// against the BASE template — see flip_template_problem.md), this test
// registers the ORIGINAL template and a MANUALLY-flipped template as two
// SEPARATE models, each with flip=false. Each model therefore carries its own
// correct templ_image + refine_points, so ROI refine for a flipped instance is
// fed a correctly-mirrored template. This is the manual equivalent of the
// proposed buildFlippedTemplate() fix.
//
// What it checks (detection accuracy):
//   1. The correct model wins for each instance (chirality): "F" for the
//      non-flipped instance, "F_flip" for the flipped instance.
//   2. Origin localisation error (the convention-safe metric: origin = centre).
//   3. Reported angle error vs the known ground-truth rotation.
//   4. Both stay accurate under ROI refine and under additive noise.
//
// Convention: the matcher's flip in test_flip_refine mirrors feature Y, i.e.
// cv::flip(., 0) (about the horizontal axis). We mirror the template image the
// same way to build the second model, and place the flipped scene instance the
// same way, so everything is on one consistent axis.
//
//   build: cmake --build <dir> --target test_flip_separate
//   run:   ./test_flip_separate
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
    Mat t(TW,TW,CV_8U,Scalar(0)); double cx=TW/2.0, cy=TW/2.0, s=TW/120.0;
    auto box=[&](double x0,double x1,double y0,double y1){
        for(double y=y0;y<=y1;y+=0.4) for(double x=x0;x<=x1;x+=0.4){
            int px=(int)lround(cx+x*s), py=(int)lround(cy+y*s);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; } };
    box(-10,-2,-34,34);  // spine
    box(-10,26,-34,-26); // top arm
    box(-10,16,-6,2);    // mid arm
    return t;
}

// Strongly chiral "flag": vertical pole + a right-pointing triangular flag at the
// TOP. The flag's diagonal edge reverses orientation under a horizontal mirror, so
// flip(shape) is unambiguously distinct from any rotation of shape. NOTE: this shape
// also exposes a separate rotation-angle scoring weakness (its own correct model
// scores poorly at ~25/130 deg), so it is offered as a stress case, not the default.
// Centre = (TW/2, TW/2).
static Mat make_flag(int TW){
    Mat t(TW,TW,CV_8U,Scalar(0)); double cx=TW/2.0, cy=TW/2.0, s=TW/120.0;
    auto box=[&](double x0,double x1,double y0,double y1){
        for(double y=y0;y<=y1;y+=0.4) for(double x=x0;x<=x1;x+=0.4){
            int px=(int)lround(cx+x*s), py=(int)lround(cy+y*s);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; } };
    box(-6,6,-40,40);    // vertical pole (full height)
    // Solid right triangle flag at the top: base at y=-40 (wide), apex at y=-8 (narrow).
    for(double y=-40;y<=-8;y+=0.4){
        double frac=(y+40)/32.0;            // 0 at top base, 1 at apex
        double xr=6 + (40-2)*(1.0-frac);    // right edge recedes left as we go down
        for(double x=6;x<=xr;x+=0.4){
            int px=(int)lround(cx+x*s), py=(int)lround(cy+y*s);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; }
    }
    return t;
}

// Place `templ` (optionally horizontally-mirror-about-x-axis = cv::flip(.,0)) so its
// centre lands at (X,Y), rotated `deg`. warpAffine -> sub-pixel exact placement.
static void place(Mat& scene, const Mat& templ, double X, double Y, double deg, bool flip){
    // NOTE: `Mat src = templ` is a shallow copy that SHARES templ's buffer, so
    // cv::flip(templ, src, 0) would flip templ IN PLACE and corrupt the caller's
    // image across calls. Clone (or write to a fresh Mat) to keep templ const.
    Mat src; if(flip) cv::flip(templ, src, 0); else src = templ;
    double cx=templ.cols/2.0, cy=templ.rows/2.0;
    Mat R = getRotationMatrix2D(Point2f((float)cx,(float)cy), -deg, 1.0); // image y-down: -deg = CCW math
    R.at<double>(0,2) += X - cx;  R.at<double>(1,2) += Y - cy;
    Mat warped; warpAffine(src, warped, R, scene.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    scene = max(scene, warped);
}

struct GT { double X, Y, deg; bool flip; const char* want_model; };

// Best-scoring result within `radius` of a ground-truth position.
static const sbm::MatchResult* nearest(const std::vector<sbm::MatchResult>& rs, const GT& g, float radius=70){
    const sbm::MatchResult* best=nullptr; float bs=-1;
    for(auto& r: rs){ float d=(float)std::hypot(r.x-g.X, r.y-g.Y); if(d<radius && r.score>bs){bs=r.score;best=&r;} }
    return best;
}

// Match ONE model in its own matcher (no cross-model NMS), return its best hit
// near a ground-truth position.
static const sbm::MatchResult bestHit(const sbm::FeatureSet& f, const char* name,
                                      const Mat& scene, const GT& g, bool& found){
    sbm::MatchConfig cfg; cfg.min_score=40; cfg.nms_radius=60; cfg.refine=sbm::RefineMode::ROI;
    sbm::ShapeMatcher m(cfg);
    sbm::ModelConfig mc; mc.angle={0,360,1}; mc.flip=false;
    m.addModel(name, f, mc);
    auto rs = m.match(scene);
    const auto* r = nearest(rs, g);
    found = (r!=nullptr);
    return found ? *r : sbm::MatchResult{};
}

int main(){
    printf("=== flip as SEPARATE model (original + manually-flipped) ===\n\n");
    const int TW=240;           // 2x template
    Mat templ   = make_flag(TW); // strong-chirality stress case (score-collapse probe); make_F(TW) for the clean case
    Mat templ_f; cv::flip(templ, templ_f, 0);  // horizontal-axis mirror, same as matcher's flip

    auto feat   = sbm::extractFeatures(templ);
    auto feat_f = sbm::extractFeatures(templ_f);
    feat.setOrigin(TW/2.0f, TW/2.0f);   feat.setAngleOffset(0);
    feat_f.setOrigin(TW/2.0f, TW/2.0f); feat_f.setAngleOffset(0);
    printf("template: %d features | flipped template: %d features\n\n",
           feat.numFeatures(), feat_f.numFeatures());

    std::vector<double> angles = {0, 25, 70, 130};
    int noises[] = {0, 12};

    // For each instance we report BOTH models' best score at that location
    // (each run in its own matcher, no cross-model NMS) so we can see whether the
    // correct model genuinely wins on score, plus the correct model's pose error.
    printf("%-6s %-7s | %-22s %-22s | %-26s\n",
           "noise","gt-kind","F-model (score)","F_flip-model (score)","correct-model pose err");
    printf("------------------------------------------------------------------------------------------------\n");

    double worst_ctrl_o=0, worst_flip_o=0, worst_ctrl_a=0, worst_flip_a=0;
    int margin_fail=0;

    for(int ns : noises){
      for(double deg : angles){
        const int W=900, H=520;
        Mat scene(H,W,CV_8U,Scalar(0));
        // control (no flip) left, flipped right, same rotation.
        GT gc{ 220.0, 260.0, deg, false, "F"      };
        GT gf{ 660.0, 260.0, deg, true,  "F_flip" };
        place(scene, templ, gc.X, gc.Y, gc.deg, gc.flip);
        place(scene, templ, gf.X, gf.Y, gf.deg, gf.flip);
        if(ns>0){ Mat n(H,W,CV_8U); randn(n,0,ns); scene += n; }


        for(GT* g : {&gc,&gf}){
          const char* kind = g->flip?"flip":"ctrl";
          bool f_ok=false, ff_ok=false;
          auto rF  = bestHit(feat,   "F",      scene, *g, f_ok);
          auto rFF = bestHit(feat_f, "F_flip", scene, *g, ff_ok);
          float sF  = f_ok ? rF.score  : -1;
          float sFF = ff_ok? rFF.score : -1;

          // The correct (ground-truth) model and the result we trust for pose.
          const sbm::MatchResult* correct = g->flip ? (ff_ok?&rFF:nullptr) : (f_ok?&rF:nullptr);
          bool winner_correct = g->flip ? (sFF >= sF) : (sF >= sFF);

          printf("%-6d %-7s | score=%-15.1f score=%-15.1f | ",
                 ns, kind, sF, sFF);
          if(!correct){ printf("CORRECT MODEL NOT DETECTED\n"); g_fail++; continue; }

          float o = (float)std::hypot(correct->x-g->X, correct->y-g->Y);
          float a = std::fabs(angDiff(correct->angle, (float)deg));
          printf("o=%5.2f a=%6.2f  win=%s\n", o, a, winner_correct?"correct":"WRONG");

          if(!winner_correct){ margin_fail++; continue; }  // detection failed -> pose is meaningless
          // Pose accuracy is asserted ONLY where the correct model actually won.
          if(g->flip){ worst_flip_o=std::max(worst_flip_o,(double)o); worst_flip_a=std::max(worst_flip_a,(double)a); }
          else       { worst_ctrl_o=std::max(worst_ctrl_o,(double)o); worst_ctrl_a=std::max(worst_ctrl_a,(double)a); }
        }
      }
    }

    printf("\nworst origin err (correct model, when detected):  control=%.2fpx  flip=%.2fpx\n", worst_ctrl_o, worst_flip_o);
    printf("worst angle  err (correct model, when detected):  control=%.2fdeg flip=%.2fdeg\n", worst_ctrl_a, worst_flip_a);

    // -------- HARD CONTRACT: pose accuracy of separate registration --------
    // When the correct (own-chirality) model is used, ROI refine with that model's
    // OWN template gives flip poses as accurate as the non-flipped control. This is
    // the property the "independent flipped template" fix must guarantee.
    CHECK(worst_flip_o <= 1.5, "flip origin err %.2fpx exceeds 1.5px", worst_flip_o);
    CHECK(worst_flip_a <= 3.0, "flip angle err %.2fdeg exceeds 3deg", worst_flip_a);
    CHECK(worst_flip_o <= worst_ctrl_o + 1.0, "flip origin (%.2f) much worse than control (%.2f)",
          worst_flip_o, worst_ctrl_o);

    // -------- Flip DETECTION margin --------
    // The matcher cleanly scores the correct chirality at 100 and the wrong one at
    // ~60 at every angle. An earlier apparent "flip-detection ambiguity at 25/130
    // deg" turned out to be a TEST-HARNESS bug: place() aliased templ's buffer
    // (`Mat src = templ; cv::flip(templ, src, 0)`), flipping the global template
    // in place and toggling which instance was mirrored every iteration (hence the
    // alternating 0-ok / 25-bad / 70-ok / 130-bad parity). With that fixed there is
    // no ambiguity. If this ever fires again, suspect a similar scene-construction
    // bug before blaming the matcher.
    CHECK(margin_fail==0, "%d case(s) where wrong-chirality model scored >= correct "
          "(unexpected; check scene construction)", margin_fail);
    if(!margin_fail)
        printf("\nflip detection: correct model out-scored wrong model in all cases.\n");

    printf(g_fail? "\n*** %d CHECK(S) FAILED ***\n" : "\nAll checks passed.\n", g_fail);
    return g_fail?1:0;
}
