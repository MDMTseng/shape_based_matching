// negative_feature_ab.cpp — validate negative / exclusion points: a template
// region that must be EDGE-FREE, used to reject false matches that have extra
// structure there. Template has a shape in its LEFT half, empty RIGHT half;
// negative points sit in the empty right zone. Scene has a CLEAN instance
// (right zone empty) and a CONTAMINATED one (a blob drawn in the right zone).
// Sweeps the penalty from off -> soft -> near-hard -> hard-veto.
//
//   ./negative_feature_ab

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>

// Template: a 180-ASYMMETRIC arrow in the LEFT-CENTER (so the match pose is
// unambiguous — a symmetric shape would match at the mirrored 180 pose and the
// negative points would land on the wrong side). RIGHT zone deliberately empty.
static cv::Mat leftShape(int S=130){
    cv::Mat m(S,S,CV_8U,cv::Scalar(40));
    int cx=S*2/5, cy=S/2;                              // arrow centred left-of-middle
    cv::rectangle(m,{cx-5,cy-35},{cx+5,cy+30},220,cv::FILLED);   // shaft
    std::vector<cv::Point> head{{cx,cy-52},{cx-18,cy-28},{cx+18,cy-28}};
    cv::fillConvexPoly(m,head,220);                    // arrowhead up (breaks 180 symmetry)
    return m;   // right zone (x > S*3/5) is background
}

int main(){
    sbm::setLogLevel(sbm::LogLevel::Error);
    const int S=130, W=1000, H=760; const float tol=16.f; const double sigma=8.0;
    cv::Mat tmpl=leftShape(S);
    sbm::FeatureSet fs=sbm::extractFeatures(tmpl,cv::Mat(),128);

    // Negative points in the empty RIGHT zone, relative to template CENTER (S/2,S/2).
    // A small grid at x ~ +25..+45 (right of center), y spread.
    for(int dx : {22,32,42}) for(int dy : {-25,-8,8,25})
        fs.negative_points.push_back(cv::Point2f((float)dx,(float)dy));
    std::printf("negative_feature_ab | template %dx%d | %zu features, %zu negative points\n",
                S,S,fs.numFeatures(),fs.negative_points.size());

    // Build scene: clean instance @ left-center, contaminated @ right-center.
    cv::RNG rng(5);
    cv::Mat scene(H,W,CV_8U,cv::Scalar(45));
    auto place=[&](float cx,float cy,bool contaminate){
        cv::Mat obj; tmpl.copyTo(obj);
        if(contaminate){ // extra blob in the empty RIGHT zone (where negative points are)
            cv::circle(obj,{S/2+32,S/2},10,210,cv::FILLED); }
        int px=(int)(cx-S/2),py=(int)(cy-S/2);
        cv::Mat roi=scene(cv::Rect(px,py,S,S));
        obj.copyTo(roi, obj>70);
    };
    place(300,380,false);   // CLEAN
    place(700,380,true);    // CONTAMINATED (blob in the forbidden zone)
    cv::Mat noise(H,W,CV_8U); rng.fill(noise,cv::RNG::NORMAL,0,sigma); cv::add(scene,noise,scene);

    struct Cfg{const char*name; float pen; bool veto;};
    std::vector<Cfg> cfgs={{"off",0,false},{"soft (pen 20)",20,false},
                           {"near-hard (pen 100)",100,false},{"hard-veto",0,true}};
    std::printf("\n  %-20s | %-18s | %-18s\n","config","CLEAN @300","CONTAMINATED @700");
    std::printf("  %-20s | %-8s %-8s  | %-8s %-8s\n","","found","score","found","score");
    std::printf("  ---------------------+--------------------+-------------------\n");
    for(auto&cf:cfgs){
        sbm::MatchConfig c; c.min_score=50; c.refine=sbm::RefineMode::None; c.skip_voting=true;
        c.negative_penalty=cf.pen; c.negative_hard_veto=cf.veto; c.negative_min_violations=1;
        c.negative_mag_thresh=60.f;
        sbm::ShapeMatcher M(c); sbm::ModelConfig mc; mc.angle={0,360,4}; mc.scale={1,1,0.1f};
        M.addModel("m",fs,mc);
        auto rs=M.match(scene);
        // Locate the two real instances robustly by x-range (left=clean, right=contaminated);
        // ignore the coordinate-convention offset — the negative check follows the matched pose.
        auto best=[&](float xlo,float xhi,bool&f)->float{ float b=-1; f=false;
            for(auto&r:rs) if(r.x>=xlo&&r.x<xhi&&std::abs(r.y-380)<80&&r.score>b){b=r.score;f=true;} return b; };
        bool fc,fx; float sc=best(150,500,fc), sx=best(500,850,fx);
        (void)tol;
        std::printf("  %-20s | %-8s %-8.1f | %-8s %-8.1f\n", cf.name,
                    fc?"yes":"NO",fc?sc:0, fx?"yes":"NO ",fx?sx:0);
    }
    std::printf("\n  (CLEAN should stay found/high at every setting; CONTAMINATED should lose\n"
                "   score with penalty and be REJECTED at near-hard / hard-veto.)\n");
    return 0;
}
