// bench_nms.cpp — measure the candidate local-max prefilter (line2Dup coarse
// candidate extraction) vs legacy, across NOISE levels. High noise floods the
// above-threshold candidate list, which the legacy path pushes wholesale (then
// caps/refines/NMS-es); the local-max prefilter collapses each blob to its peak
// at the source. Reports match() wall time + detections (correctness unchanged).
//
//   ./bench_nms [seed]

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
static cv::Mat gradientBG(int W,int H,cv::RNG& rng){
    cv::Mat bg(H,W,CV_32F); float a=rng.uniform(-0.05f,0.05f),b=rng.uniform(-0.05f,0.05f);
    float base=rng.uniform(35.f,55.f);
    for(int y=0;y<H;y++){float*p=bg.ptr<float>(y);for(int x=0;x<W;x++)p[x]=base+a*x+b*y;}
    cv::Mat o; bg.convertTo(o,CV_8U); return o;}

int main(int argc,char**argv){
    sbm::setLogLevel(sbm::LogLevel::Error);
    int seed=argc>1?std::atoi(argv[1]):7;
    cv::RNG rng(seed);
    const int W=1000,H=760; const float tol=18.f; const int NREP=8;

    cv::Mat tmpl=randTemplate(rng);
    sbm::FeatureSet fs=sbm::extractFeatures(tmpl,cv::Mat(),128);

    // Place 6 clean rotated instances (no skew) — the ground truth.
    struct GT{float cx,cy;};
    std::vector<GT> gts;
    cv::Mat base=gradientBG(W,H,rng);
    int cols=3,rows=2;
    for(int i=0;i<6;i++){ int gx=i%cols,gy=i/cols;
        float cx=W*(gx+1.0f)/(cols+1)+rng.uniform(-20,20), cy=H*(gy+1.0f)/(rows+1)+rng.uniform(-20,20);
        float ang=rng.uniform(0.f,360.f);
        cv::Mat R=cv::getRotationMatrix2D({tmpl.cols/2.f,tmpl.rows/2.f},ang,1.0),obj;
        cv::warpAffine(tmpl,obj,R,tmpl.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(0));
        R.at<double>(0,2)+=cx-tmpl.cols/2.0; R.at<double>(1,2)+=cy-tmpl.rows/2.0;
        cv::warpAffine(tmpl,obj,R,base.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(0));
        obj.copyTo(base, obj>70); gts.push_back({cx,cy}); }

    std::printf("bench_nms | seed %d | %dx%d | 6 instances | template %dx%d | %d reps\n\n",
                seed,W,H,tmpl.cols,tmpl.rows,NREP);
    std::printf("  %-7s | %-22s | %-22s | %-s\n","noise","LOCAL-MAX on","LEGACY (all>thresh)","speedup");
    std::printf("  %-7s | %-10s %-10s | %-10s %-10s |\n","sigma","ms","det","ms","det");
    std::printf("  --------+-----------------------+-----------------------+--------\n");

    for(double sig : {10.0, 25.0, 40.0}) {
        cv::Mat scene=base.clone();
        cv::Mat noise(H,W,CV_8U); rng.fill(noise,cv::RNG::NORMAL,0,sig); cv::add(scene,noise,scene);

        auto run=[&](bool lmax, double& ms, int& det){
            sbm::MatchConfig c; c.min_score=30; c.refine=sbm::RefineMode::None; c.skip_voting=true;
            c.candidate_local_max=lmax;   // min_score 30 + no refine → maximise the candidate flood
            sbm::ShapeMatcher M(c); sbm::ModelConfig mc; mc.angle={0,360,2}; mc.scale={1,1,0.1f};
            M.addModel("m",fs,mc);
            std::vector<sbm::MatchResult> rs;
            double t=0; for(int rep=0;rep<NREP;++rep){ auto t0=Clk::now(); rs=M.match(scene);
                t+=std::chrono::duration<double,std::milli>(Clk::now()-t0).count(); }
            ms=t/NREP;
            det=0; for(auto&g:gts) for(auto&r:rs) if(std::abs(r.x-g.cx)<tol&&std::abs(r.y-g.cy)<tol){det++;break;}
        };
        double ms_on,ms_off; int det_on,det_off;
        run(true,ms_on,det_on); run(false,ms_off,det_off);
        std::printf("  %-7.0f | %-10.1f %-10d | %-10.1f %-10d | %.2fx\n",
                    sig,ms_on,det_on,ms_off,det_off, ms_off/ms_on);
    }
    std::printf("\n  (local-max collapses each score blob to its peak at extraction; det should be\n"
                "   unchanged, ms should drop — most at high noise where the flood is worst.)\n");
    return 0;
}
