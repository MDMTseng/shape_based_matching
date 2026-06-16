// test_roi_speed_1d.cpp
// Measure the per-match refine cost of 1D edge matching vs 2D template matching.
// Times match() over many repeats; subtracts the refine=None baseline (coarse only)
// to isolate the refine cost per object. Same edge-Dopt point set for the ROI modes.

#include "shape_matcher.h"
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

using namespace cv;
using clk = std::chrono::high_resolution_clock;

static void boxInto(Mat& t,double cx,double cy,double s,double x0,double x1,double y0,double y1){
    for(double y=y0;y<=y1;y+=0.4) for(double x=x0;x<=x1;x+=0.4){
        int px=(int)lround(cx+x*s), py=(int)lround(cy+y*s);
        if(px>=0&&px<t.cols&&py>=0&&py<t.rows) t.at<uchar>(py,px)=200; }
}
static Mat make_L(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-30,-18,-34,34); boxInto(t,c,c,s,-30,30,22,34); return t; }
static Mat make_F(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-10,-2,-34,34); boxInto(t,c,c,s,-10,26,-34,-26); boxInto(t,c,c,s,-10,16,-6,2); return t; }
static Mat make_T(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-34,34,-34,-22); boxInto(t,c,c,s,-6,6,-34,34); return t; }
static Mat make_flag(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-6,6,-40,40);
    for(double y=-40;y<=-8;y+=0.4){ double f=(y+40)/32.0, xr=6+(40-2)*(1.0-f);
        for(double x=6;x<=xr;x+=0.4){ int px=(int)lround(c+x*s),py=(int)lround(c+y*s);
            if(px>=0&&px<TW&&py>=0&&py<TW) t.at<uchar>(py,px)=200; } } return t; }

static std::vector<Point2f> edgeDopt(const sbm::FeatureSet& feat, int N, float spacing){
    float hw=feat.templ_width/2.0f-5, hh=feat.templ_height/2.0f-5;
    struct E{ float px,py,j0,j1,j2; };
    std::vector<E> cand;
    for(auto& rp:feat.refine_points){
        if(rp.type!=sbm::FeatureSet::RefinePt::EDGE) continue;
        if(std::abs(rp.px)>hw||std::abs(rp.py)>hh) continue;
        float nn=std::sqrt(rp.nx*rp.nx+rp.ny*rp.ny); if(nn<1e-3f) continue;
        float nx=rp.nx/nn, ny=rp.ny/nn;
        cand.push_back({rp.px,rp.py,-rp.py*nx+rp.px*ny,nx,ny});
    }
    std::vector<Point2f> sel; std::vector<int> sidx; float I[3][3]={}; for(int i=0;i<3;i++)I[i][i]=1e-3f;
    auto det3=[](float A[3][3]){return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])-A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])+A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);};
    float sp2=spacing*spacing;
    while((int)sel.size()<N){ int best=-1; float bestd=-1e30f;
        for(int i=0;i<(int)cand.size();i++){ bool u=false; for(int s:sidx)if(s==i){u=true;break;} if(u)continue;
            if(spacing>0){bool c=false;for(auto&p:sel){float dx=cand[i].px-p.x,dy=cand[i].py-p.y;if(dx*dx+dy*dy<sp2){c=true;break;}}if(c)continue;}
            float T[3][3];for(int r=0;r<3;r++)for(int q=0;q<3;q++)T[r][q]=I[r][q];
            float j0=cand[i].j0,j1=cand[i].j1,j2=cand[i].j2;
            T[0][0]+=j0*j0;T[0][1]+=j0*j1;T[0][2]+=j0*j2;T[1][0]+=j1*j0;T[1][1]+=j1*j1;T[1][2]+=j1*j2;T[2][0]+=j2*j0;T[2][1]+=j2*j1;T[2][2]+=j2*j2;
            float d=det3(T); if(d>bestd){bestd=d;best=i;} }
        if(best<0)break; sidx.push_back(best); sel.push_back(Point2f(cand[best].px,cand[best].py));
        float j0=cand[best].j0,j1=cand[best].j1,j2=cand[best].j2;
        I[0][0]+=j0*j0;I[0][1]+=j0*j1;I[0][2]+=j0*j2;I[1][0]+=j1*j0;I[1][1]+=j1*j1;I[1][2]+=j1*j2;I[2][0]+=j2*j0;I[2][1]+=j2*j1;I[2][2]+=j2*j2; }
    return sel;
}

static double timeMode(const Mat& templ, sbm::RefineMode rm, bool edge, bool oned, bool iter, int reps, bool collapse=false, bool edge_only_lib=false){
    int TW=templ.cols;
    auto feat=sbm::extractFeatures(templ); feat.setOrigin(TW/2.0f,TW/2.0f);
    if(edge){ feat.cached_opt_points=edgeDopt(feat,8,12.0f); feat.cached_opt_max_points=8; }
    sbm::MatchConfig cfg; cfg.min_score=40; cfg.nms_radius=60; cfg.refine=rm;
    cfg.roi_edge_1d_match=oned; cfg.roi_iterative_rematch=iter; cfg.roi_edge_collapse=collapse;
    if(edge_only_lib){ cfg.roi_edge_only_points=true; cfg.roi_min_spacing=12.0f; }
    sbm::ShapeMatcher m(cfg); sbm::ModelConfig mc; mc.angle={0,360,1}; mc.flip=false;
    m.addModel("s",feat,mc);
    const int W=400,H=400; Mat scene(H,W,CV_8U,Scalar(0));
    { double cx=TW/2.0,cy=TW/2.0; Mat R=getRotationMatrix2D(Point2f((float)cx,(float)cy),-33.0,1.0);
      R.at<double>(0,2)+=200-cx; R.at<double>(1,2)+=200-cy; Mat w; warpAffine(templ,w,R,scene.size()); scene=max(scene,w); }
    m.match(scene); // warm up
    auto t0=clk::now();
    for(int i=0;i<reps;i++){ volatile auto r=m.match(scene); (void)r; }
    auto t1=clk::now();
    return std::chrono::duration<double,std::milli>(t1-t0).count()/reps;
}

int main(){
    const int TW=140; int reps=400;
    struct S{ const char* nm; Mat im; };
    std::vector<S> shapes = {{"F",make_F(TW)},{"flag",make_flag(TW)},{"L",make_L(TW)},{"T",make_T(TW)}};
    printf("Default-2D vs edge-only-2D refine cost (same 2D matchTemplate, different points).\n");
    printf("refine ms = match() - None. %d reps each.\n\n", reps);
    printf("%-6s %10s %10s %14s\n","shape","edgeOnly","eo+rematch","rematch ratio");
    printf("-------------------------------------------------\n");
    for(auto& s : shapes){
        double none = timeMode(s.im, sbm::RefineMode::None, false,false,false, reps);
        double eo   = timeMode(s.im, sbm::RefineMode::ROI,  false,false,false, reps, false, true);
        double eor  = timeMode(s.im, sbm::RefineMode::ROI,  false,false,true,  reps, false, true); // +iterative rematch (re-warps every iter)
        double re = eo-none, rr = eor-none;
        printf("%-6s %10.3f %10.3f %13.2fx\n", s.nm, re, rr, re>1e-6? rr/re : 0.0);
    }
    return 0;
}
