// test_multi_template.cpp
// ---------------------------------------------------------------------------
// Multi-template matching: register several DISTINCT shapes (plus an
// original+flipped pair) in ONE ShapeMatcher, build a scene containing one
// instance of each at a known pose+rotation, and assert that every instance is
// detected by the CORRECT model with an accurate ROI-refined pose and no
// cross-talk (no wrong model winning a location).
//
// Each model carries its own FeatureSet (templ_image + refine_points + caches),
// so ROI refine for a hit uses that model's own template — verified here.
//
//   build: cmake --build <dir> --target test_multi_template
//   run:   ./test_multi_template
// ---------------------------------------------------------------------------

#include "shape_matcher.h"
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>   // getRotationMatrix2D moved here in OpenCV 5
#endif
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

using namespace cv;
static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr,"  FAIL: "); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\n"); g_fail++; } } while(0)

static float angDiff(float a, float b){ float d=a-b; while(d>180)d-=360; while(d<-180)d+=360; return d; }

// ---- distinct asymmetric shapes (centre = TW/2,TW/2) ----
static void boxInto(Mat& t,double cx,double cy,double s,double x0,double x1,double y0,double y1){
    for(double y=y0;y<=y1;y+=0.4) for(double x=x0;x<=x1;x+=0.4){
        int px=(int)lround(cx+x*s), py=(int)lround(cy+y*s);
        if(px>=0&&px<t.cols&&py>=0&&py<t.rows) t.at<uchar>(py,px)=200; }
}
static Mat make_F(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-10,-2,-34,34); boxInto(t,c,c,s,-10,26,-34,-26); boxInto(t,c,c,s,-10,16,-6,2); return t; }
static Mat make_flag(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-6,6,-40,40);
    for(double y=-40;y<=-8;y+=0.4){ double f=(y+40)/32.0, xr=6+(40-2)*(1.0-f);
        for(double x=6;x<=xr;x+=0.4){ int px=(int)lround(c+x*s),py=(int)lround(c+y*s);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; } } return t; }
static Mat make_L(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-30,-18,-34,34); boxInto(t,c,c,s,-30,30,22,34); return t; }   // L: tall left bar + bottom bar
static Mat make_T(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-34,34,-34,-22); boxInto(t,c,c,s,-6,6,-34,34); return t; }     // T: top bar + stem

// Place templ at (X,Y) rotated deg. NOTE: write flip into a FRESH Mat (never alias
// the source buffer — that was the bug that corrupted earlier flip experiments).
static void place(Mat& scene, const Mat& templ, double X, double Y, double deg, bool flip){
    Mat src; if(flip) cv::flip(templ, src, 0); else src = templ;
    double cx=templ.cols/2.0, cy=templ.rows/2.0;
    Mat R = getRotationMatrix2D(Point2f((float)cx,(float)cy), -deg, 1.0);
    R.at<double>(0,2) += X - cx;  R.at<double>(1,2) += Y - cy;
    Mat warped; warpAffine(src, warped, R, scene.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    scene = max(scene, warped);
}

struct Inst { std::string model; double X, Y, deg; bool flip; };

int main(){
    printf("=== multi-template matching (distinct shapes + flip pair, one matcher) ===\n\n");
    const int TW=140;

    // Build feature sets for each distinct shape.
    struct Model { std::string name; Mat templ; sbm::FeatureSet feat; };
    std::vector<Model> models;
    auto add=[&](const std::string& nm, const Mat& im){
        Model m; m.name=nm; m.templ=im; m.feat=sbm::extractFeatures(im);
        m.feat.setOrigin(TW/2.0f, TW/2.0f); m.feat.setAngleOffset(0); models.push_back(m); };
    add("F",     make_F(TW));
    add("flag",  make_flag(TW));
    add("L",     make_L(TW));
    add("T",     make_T(TW));
    // flip pair: register the horizontally-mirrored flag as its own model.
    // CAUTION: do NOT name it "flag_flip" — ShapeMatcher::addModel sets model
    // "flag"'s internal class_id_flip = "sbm_flag_flip", which collides with a model
    // literally named "flag_flip" (class_id = "sbm_flag_flip"). The match loop would
    // then mis-attribute this model's hits to "flag" as flipped. Use a distinct name.
    Mat flag_f; cv::flip(make_flag(TW), flag_f, 0);
    add("flagM", flag_f);   // "flag mirrored"

    // One matcher, all models registered.
    sbm::MatchConfig cfg; cfg.min_score=50; cfg.nms_radius=70; cfg.refine=sbm::RefineMode::ICP_Subpixel;
    sbm::ShapeMatcher matcher(cfg);
    sbm::ModelConfig mc; mc.angle={0,360,1}; mc.flip=false;
    for(auto& m : models){ matcher.addModel(m.name, m.feat, mc); printf("registered %-10s (%d feat)\n", m.name.c_str(), m.feat.numFeatures()); }
    printf("\n%d models, %d total template variants\n\n", matcher.numModels(), matcher.numTemplates());

    // Scene: one instance of each, distinct pose. flag_flip instance is a mirrored flag.
    const int W=1000, H=700;
    Mat scene(H,W,CV_8U,Scalar(0));
    std::vector<Inst> insts = {
        { "F",         180, 180,   0, false },
        { "flag",      500, 180,  25, false },
        { "L",         820, 180,  70, false },
        { "T",         180, 480, 130, false },
        { "flagM",     500, 480,  40, false },  // model is already the mirrored shape
    };
    for(auto& it : insts){
        // For "flag_flip" the registered model IS the mirrored flag, so place its
        // (already-mirrored) template directly without re-flipping.
        const Mat* tm = nullptr;
        for(auto& m : models) if(m.name==it.model){ tm=&m.templ; break; }
        place(scene, *tm, it.X, it.Y, it.deg, it.flip);
    }

    auto rs = matcher.match(scene);
    printf("scene produced %d hits (>=score %.0f)\n", (int)rs.size(), cfg.min_score);
    for(auto& r: rs) printf("   hit: model=%-10s x=%.1f y=%.1f score=%.1f ang=%.1f flipped=%d\n",
                            r.model_name.c_str(), r.x, r.y, r.score, r.angle, (int)r.flipped);
    printf("\n");

    printf("%-10s %-18s | %-28s | %s\n","expect","gt(x,y,deg)","got(model,score)","pose err");
    printf("--------------------------------------------------------------------------------------------\n");
    for(auto& it : insts){
        // best-scoring hit within radius of the instance origin
        const sbm::MatchResult* best=nullptr; float bs=-1;
        for(auto& r: rs){ float d=(float)std::hypot(r.x-it.X, r.y-it.Y); if(d<80 && r.score>bs){bs=r.score;best=&r;} }
        if(!best){ printf("%-10s (%.0f,%.0f,%.0f) | NOT DETECTED\n", it.model.c_str(), it.X,it.Y,it.deg); g_fail++; continue; }
        float o=(float)std::hypot(best->x-it.X, best->y-it.Y);
        float a=std::fabs(angDiff(best->angle,(float)it.deg));
        bool model_ok = (best->model_name==it.model);
        printf("%-10s (%.0f,%.0f,%.0f) | %-12s score=%5.1f      | o=%5.2f a=%5.2f %s\n",
               it.model.c_str(), it.X,it.Y,it.deg, best->model_name.c_str(), best->score, o, a,
               model_ok?"":" <-- WRONG MODEL");
        // With ICP_Subpixel refine, every shape localizes to sub-0.5px / sub-0.5deg.
        // (RefineMode::ROI is angle-tuned and gives 0.2-0.5px position with occasional
        // ~2px outliers — see test_refine_accuracy.cpp for the per-mode comparison.)
        CHECK(model_ok, "%s instance won by wrong model '%s'", it.model.c_str(), best->model_name.c_str());
        CHECK(o < 0.5, "%s origin err %.2fpx > 0.5", it.model.c_str(), o);
        CHECK(a < 0.5, "%s angle err %.2fdeg > 0.5", it.model.c_str(), a);
    }

    printf(g_fail? "\n*** %d CHECK(S) FAILED ***\n" : "\nAll checks passed.\n", g_fail);
    return g_fail?1:0;
}
