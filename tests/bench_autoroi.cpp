// bench_autoroi.cpp — the ONLY effective way to exploit large empty regions in
// the current dense linear-memory architecture: shrink the processed EXTENT.
// A cheap edge-bbox pre-pass finds where content is, we crop to it (+margin),
// match on the crop, then offset results back to scene coords. Pose stays a
// full 0..360 register (SBM needed). Measures END-TO-END time INCLUDING the
// detect pass, vs full-scene match, and verifies detections land on the same
// ground-truth instances (so the crop didn't lose or misplace anything).
//
// For a non-rectangular mask: same idea — crop to the mask's bbox, then filter
// results whose center falls outside the mask. Here we demo the rectangular case.
//
//   ./bench_autoroi [seed]

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdio>
#include <vector>

using Clk = std::chrono::high_resolution_clock;

static cv::Mat randTemplate(cv::RNG& rng,int S=130){
    cv::Mat m(S,S,CV_8U,cv::Scalar(40));
    int k=rng.uniform(3,5);
    for(int i=0;i<k;i++){ int cx=rng.uniform(S/3,S*2/3),cy=rng.uniform(S/3,S*2/3);
        int bright=rng.uniform(150,235); double ang=rng.uniform(0.0,360.0);
        if(rng.uniform(0,2)){ int a=rng.uniform(S/8,S/4),b=rng.uniform(S/8,S/4);
            cv::ellipse(m,{cx,cy},{a,b},ang,0,360,bright,cv::FILLED,cv::LINE_AA);}
        else{ int n=rng.uniform(3,6);double R=rng.uniform(S/8.0,S/4.0);std::vector<cv::Point> p;
            for(int j=0;j<n;j++){double t=ang*CV_PI/180+2*CV_PI*j/n+rng.uniform(-0.2,0.2);
                double r=R*(1+rng.uniform(-0.2,0.2));p.push_back({(int)(cx+r*cos(t)),(int)(cy+r*sin(t))});}
            std::vector<std::vector<cv::Point>> pp{p};cv::fillPoly(m,pp,bright,cv::LINE_AA);}}
    return m;}

// Cheap content-bbox detector: downscale -> gradient magnitude -> threshold ->
// bounding box of above-threshold pixels, scaled back up + margin. Works for
// bright objects AND dark silhouettes (both have edges at their boundary).
static cv::Rect detectBBox(const cv::Mat& scene, int down, int magThresh, int margin){
    cv::Mat small; cv::resize(scene, small, cv::Size(), 1.0/down, 1.0/down, cv::INTER_AREA);
    cv::Mat gx, gy; cv::Sobel(small,gx,CV_16S,1,0,3); cv::Sobel(small,gy,CV_16S,0,1,3);
    cv::Mat ax, ay, mag; cv::convertScaleAbs(gx,ax); cv::convertScaleAbs(gy,ay);
    cv::add(ax, ay, mag);
    cv::Mat m; cv::threshold(mag, m, magThresh, 255, cv::THRESH_BINARY);
    std::vector<cv::Point> pts; cv::findNonZero(m, pts);
    if(pts.empty()) return cv::Rect(0,0,scene.cols,scene.rows);      // fallback: full
    cv::Rect r = cv::boundingRect(pts);
    // scale back to full res + margin, clamp to image
    int x0=std::max(0, r.x*down - margin), y0=std::max(0, r.y*down - margin);
    int x1=std::min(scene.cols, (r.x+r.width)*down + margin);
    int y1=std::min(scene.rows, (r.y+r.height)*down + margin);
    return cv::Rect(x0,y0,x1-x0,y1-y0);
}

int main(int argc,char**argv){
    sbm::setLogLevel(sbm::LogLevel::Error);
    int seed=argc>1?std::atoi(argv[1]):7;
    cv::RNG rng(seed);
    const int W=1000,H=760,S=130; const int NREP=10; const float tol=18.f;

    cv::Mat tmpl=randTemplate(rng);
    sbm::FeatureSet fs=sbm::extractFeatures(tmpl,cv::Mat(),128);

    std::printf("bench_autoroi | seed %d | %dx%d | template %dx%d | %d reps\n",
                seed,W,H,S,S,NREP);
    std::printf("  Full-scene match vs (detect bbox + crop + match + remap).\n");
    std::printf("  End-to-end incl. detect. Pose 0..360 register. det = GT instances hit.\n\n");
    std::printf("  %-10s | %-6s | %-8s | %-8s | %-9s | %-7s | %-s\n",
                "content","empty%","full ms","auto ms","detect ms","bbox%","speedup  det(full/auto)");
    std::printf("  -----------+--------+----------+----------+-----------+-------+----------------------\n");

    struct FR{const char*name; float f;};
    std::vector<FR> fracs={{"1/1 (all)",1.0f},{"1/4",0.5f},{"1/9",0.333f},{"1/16",0.25f}};

    for(auto& fr : fracs){
        int cw=(int)(W*fr.f), ch=(int)(H*fr.f);
        cw=std::max(cw,S+40); ch=std::max(ch,S+40);
        cv::Mat scene(H,W,CV_8U,cv::Scalar(45));
        std::vector<cv::Point2f> gts;
        int nx=std::max(1,cw/(S+10)), ny=std::max(1,ch/(S+10));
        for(int gy=0;gy<ny;gy++)for(int gx=0;gx<nx;gx++){
            float cx=(gx+0.5f)*cw/nx, cy=(gy+0.5f)*ch/ny;
            float ang=rng.uniform(0.f,360.f);
            cv::Mat R=cv::getRotationMatrix2D({S/2.f,S/2.f},ang,1.0),obj;
            R.at<double>(0,2)+=cx-S/2.0; R.at<double>(1,2)+=cy-S/2.0;
            cv::warpAffine(tmpl,obj,R,scene.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(0));
            obj.copyTo(scene, obj>70); gts.push_back({cx,cy});
        }
        cv::Mat noise(H,W,CV_8U); rng.fill(noise,cv::RNG::NORMAL,0,12.0); cv::add(scene,noise,scene);

        auto mkMatcher=[&](){ sbm::MatchConfig c; c.min_score=55; c.refine=sbm::RefineMode::None;
            c.skip_voting=true; auto M=std::make_shared<sbm::ShapeMatcher>(c);
            sbm::ModelConfig mc; mc.angle={0,360,2}; mc.scale={1,1,0.1f}; M->addModel("m",fs,mc); return M; };
        auto countHits=[&](const std::vector<sbm::MatchResult>& rs, float ox, float oy){
            int det=0; for(auto&g:gts) for(auto&r:rs)
                if(std::abs(r.x+ox-g.x)<tol&&std::abs(r.y+oy-g.y)<tol){det++;break;} return det; };

        // --- FULL: match whole scene ---
        auto Mf=mkMatcher(); std::vector<sbm::MatchResult> rf;
        double tf=0; for(int rep=0;rep<NREP;++rep){ auto t0=Clk::now(); rf=Mf->match(scene);
            tf+=std::chrono::duration<double,std::milli>(Clk::now()-t0).count(); }
        double ms_full=tf/NREP; int det_full=countHits(rf,0,0);

        // --- AUTO-ROI: detect bbox + crop + match + remap ---
        auto Ma=mkMatcher(); std::vector<sbm::MatchResult> ra; cv::Rect roi;
        double ta=0, td=0;
        for(int rep=0;rep<NREP;++rep){
            auto t0=Clk::now();
            roi=detectBBox(scene, /*down=*/8, /*magThresh=*/40, /*margin=*/S/2+8);
            auto t1=Clk::now();
            cv::Mat crop=scene(roi);           // view, no copy
            ra=Ma->match(crop);
            auto t2=Clk::now();
            td+=std::chrono::duration<double,std::milli>(t1-t0).count();
            ta+=std::chrono::duration<double,std::milli>(t2-t0).count();
        }
        double ms_auto=ta/NREP, ms_det=td/NREP;
        int det_auto=countHits(ra, (float)roi.x, (float)roi.y);   // remap crop->scene
        float bboxPct=100.f*(float)(roi.width*roi.height)/(W*H);
        float emptyPct=100.f-100.f*(cw*ch)/(float)(W*H);

        std::printf("  %-10s | %-6.1f | %-8.2f | %-8.2f | %-9.2f | %-5.1f | %.2fx     %d/%d %s\n",
                    fr.name, emptyPct, ms_full, ms_auto, ms_det, bboxPct, ms_full/ms_auto,
                    det_auto, det_full, det_auto==det_full?"":"[DET MISMATCH!]");
    }
    std::printf("\n  (auto ms INCLUDES the detect pass. bbox%% = cropped area / full area.\n"
                "   det must match full — proves the crop lost no instance and remap is correct.)\n");
    return 0;
}
