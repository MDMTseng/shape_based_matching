// test_flag_sweep.cpp
// Decisive probe for the rotation-angle SELF-MATCH score collapse (see
// test_flip_separate.cpp). NO flip, NO cross-model: ONE flag model vs ONE flag
// instance, swept over rotation. For each angle we place the scene instance two
// ways:
//   AA   = INTER_LINEAR  (anti-aliased edges, like a real rotated raster)
//   hard = INTER_NEAREST (no anti-aliasing, crisp edges)
// If the collapse is caused by anti-aliasing weakening tilted edges, `hard` should
// stay near 100 while `AA` dips. If `hard` also dips, the degradation is in the
// matcher's per-angle rotated-template generation, not the scene raster.

#include "shape_matcher.h"
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>   // getRotationMatrix2D moved here in OpenCV 5
#endif
#include <cstdio>
#include <cmath>

using namespace cv;
using namespace sbm;

static Mat make_flag(int TW){
    Mat t(TW,TW,CV_8U,Scalar(0)); double cx=TW/2.0, cy=TW/2.0, s=TW/120.0;
    auto box=[&](double x0,double x1,double y0,double y1){
        for(double y=y0;y<=y1;y+=0.4) for(double x=x0;x<=x1;x+=0.4){
            int px=(int)lround(cx+x*s), py=(int)lround(cy+y*s);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; } };
    box(-6,6,-40,40);
    for(double y=-40;y<=-8;y+=0.4){
        double frac=(y+40)/32.0;
        double xr=6 + (40-2)*(1.0-frac);
        for(double x=6;x<=xr;x+=0.4){
            int px=(int)lround(cx+x*s), py=(int)lround(cy+y*s);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; }
    }
    return t;
}

static float angDiff(float a, float b){ float d=a-b; while(d>180)d-=360; while(d<-180)d+=360; return d; }

static void place(Mat& scene, const Mat& templ, double X, double Y, double deg, int interp){
    double cx=templ.cols/2.0, cy=templ.rows/2.0;
    Mat R = getRotationMatrix2D(Point2f((float)cx,(float)cy), -deg, 1.0);
    R.at<double>(0,2) += X - cx;  R.at<double>(1,2) += Y - cy;
    Mat warped; warpAffine(templ, warped, R, scene.size(), interp, BORDER_CONSTANT, Scalar(0));
    scene = max(scene, warped);
}

// best-scoring hit within radius of (X,Y)
struct Hit { bool found; float score, ox, oa; };
static Hit bestHit(ShapeMatcher& m, const Mat& scene, double X, double Y, double deg){
    auto rs = m.match(scene);
    const MatchResult* best=nullptr; float bs=-1;
    for(auto& r: rs){ float d=(float)std::hypot(r.x-X, r.y-Y); if(d<80 && r.score>bs){bs=r.score;best=&r;} }
    if(!best) return {false,0,0,0};
    return {true, best->score, (float)std::hypot(best->x-X,best->y-Y), std::fabs(angDiff(best->angle,(float)deg))};
}

int main(){
    const int TW=240;
    Mat templ = make_flag(TW);
    auto feat = extractFeatures(templ);
    feat.setOrigin(TW/2.0f, TW/2.0f);
    printf("flag template TW=%d, %d features\n\n", TW, feat.numFeatures());

    MatchConfig cfg; cfg.min_score=30; cfg.nms_radius=60; cfg.refine=RefineMode::ROI;
    ShapeMatcher m(cfg);
    ModelConfig mc; mc.angle={0,360,1}; mc.flip=false;
    m.addModel("flag", feat, mc);

    printf("deg | AA(linear): score  o-err  a-err | hard(nearest): score  o-err  a-err\n");
    printf("----+----------------------------------+----------------------------------\n");
    float min_aa=1e9, min_hard=1e9; int min_aa_deg=0, min_hard_deg=0;
    for(int deg=0; deg<=180; deg+=5){
        const int W=600, H=600;
        Mat sa(H,W,CV_8U,Scalar(0)), sh(H,W,CV_8U,Scalar(0));
        place(sa, templ, 300, 300, deg, INTER_LINEAR);
        place(sh, templ, 300, 300, deg, INTER_NEAREST);
        Hit a = bestHit(m, sa, 300, 300, deg);
        Hit h = bestHit(m, sh, 300, 300, deg);
        printf("%3d | %16.1f %5.2f %6.2f | %16.1f %5.2f %6.2f\n",
               deg, a.found?a.score:-1, a.ox, a.oa, h.found?h.score:-1, h.ox, h.oa);
        if(a.found && a.score<min_aa){ min_aa=a.score; min_aa_deg=deg; }
        if(h.found && h.score<min_hard){ min_hard=h.score; min_hard_deg=deg; }
    }
    printf("\nworst self-score:  AA=%.1f @%d deg   hard=%.1f @%d deg\n",
           min_aa, min_aa_deg, min_hard, min_hard_deg);
    printf("interpretation: if hard stays high but AA dips -> anti-aliasing; "
           "if both dip -> matcher per-angle template generation.\n");

    // ---------------------------------------------------------------------
    // Bridge to test_flip_separate: does adding a SECOND (opposite-chirality)
    // instance to the scene change the correct model's score at the first?
    // ---------------------------------------------------------------------
    printf("\n=== two-instance interference probe (mimics test_flip_separate) ===\n");
    Mat templ_f; flip(templ, templ_f, 0);
    auto feat_f = extractFeatures(templ_f); feat_f.setOrigin(TW/2.0f, TW/2.0f);
    ShapeMatcher mf(cfg); ModelConfig mc2; mc2.angle={0,360,1}; mc2.flip=false;
    mf.addModel("flag_flip", feat_f, mc2);

    auto bestAt=[&](ShapeMatcher& mm, const Mat& sc, double X, double Y, double deg)->Hit{ return bestHit(mm, sc, X, Y, deg); };

    printf("deg | scene          | flag@ctrl  flag@flip | flagflip@ctrl flagflip@flip\n");
    printf("----+----------------+----------------------+----------------------------\n");
    for(int deg : {0, 25, 70, 130}){
        const int W=900, H=520;
        // (a) BOTH instances present
        Mat both(H,W,CV_8U,Scalar(0));
        place(both, templ,   220,260, deg, INTER_LINEAR);          // control (non-flipped)
        place(both, templ_f, 660,260, deg, INTER_LINEAR);          // flipped
        // (b) ONLY the control instance present
        Mat onlyc(H,W,CV_8U,Scalar(0));
        place(onlyc, templ, 220,260, deg, INTER_LINEAR);

        Hit fc  = bestAt(m,  both, 220,260, deg);   // correct model at control, both present
        Hit ff  = bestAt(m,  both, 660,260, deg);   // correct model at flip instance
        Hit ffc = bestAt(mf, both, 220,260, deg);   // flipped model at control
        Hit fff = bestAt(mf, both, 660,260, deg);   // flipped model at flip instance
        Hit fc1 = bestAt(m,  onlyc,220,260, deg);   // correct model at control, ALONE
        printf("%3d | both           | %7.1f   %7.1f | %10.1f   %10.1f\n",
               deg, fc.score, ff.score, ffc.score, fff.score);
        printf("    | ctrl-only      | %7.1f   (n/a)   | %10s   %10s   <- correct model, control alone\n",
               fc1.score, "-", "-");
    }

    // Full dump at deg=25 to expose any harness divergence.
    printf("\n=== deg=25 full dump (correct model 'flag' on two-instance scene) ===\n");
    {
        int deg=25; const int W=900,H=520;
        Mat both(H,W,CV_8U,Scalar(0));
        place(both, templ,   220,260, deg, INTER_LINEAR);
        place(both, templ_f, 660,260, deg, INTER_LINEAR);
        auto rs = m.match(both);
        printf("model 'flag' returned %d hits:\n", (int)rs.size());
        for(auto& r : rs)
            printf("  x=%.1f y=%.1f score=%.1f angle=%.1f flipped=%d model=%s\n",
                   r.x, r.y, r.score, r.angle, (int)r.flipped, r.model_name.c_str());

        // Faithful replica of test_flip_separate's bestHit(): fresh matcher,
        // min_score=40, nearest = best score within radius 70.
        auto replica=[&](const FeatureSet& f, const char* nm, double X, double Y)->float{
            MatchConfig c; c.min_score=40; c.nms_radius=60; c.refine=RefineMode::ROI;
            ShapeMatcher mm(c); ModelConfig mcx; mcx.angle={0,360,1}; mcx.flip=false;
            mm.addModel(nm, f, mcx);
            auto r2 = mm.match(both);
            const MatchResult* best=nullptr; float bs=-1;
            for(auto& rr: r2){ float d=(float)std::hypot(rr.x-X, rr.y-Y); if(d<70 && rr.score>bs){bs=rr.score;best=&rr;} }
            return best?best->score:-1;
        };
        printf("replica bestHit (min_score=40, r=70): flag@ctrl=%.1f  flagflip@ctrl=%.1f\n",
               replica(feat,"flag",220,260), replica(feat_f,"flagflip",220,260));
    }
    return 0;
}
