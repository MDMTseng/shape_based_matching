// test_edge_uniform.cpp
// Alternative ROI point selection: prefer EDGE features (not corners) and spread
// them as uniformly as possible via farthest-point sampling (FPS). Hypothesis:
// many uniformly-distributed edges give robust, well-conditioned, low-overlap
// constraints (each edge is 1D across its normal; spatial spread -> diverse normals),
// avoiding the corner-clustering degeneracy (flag) and heavy ROI overlap.
//
// Injects the chosen points via FeatureSet.cached_opt_points before addModel, so the
// matcher uses them without changing the library selector. Compares to the default
// (corner-priority) selection on origin error + point spacing, clean and under noise.

#include "shape_matcher.h"
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>   // getRotationMatrix2D moved here in OpenCV 5
#endif
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace cv;
static float angDiff(float a, float b){ float d=a-b; while(d>180)d-=360; while(d<-180)d+=360; return d; }

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
    boxInto(t,c,c,s,-30,-18,-34,34); boxInto(t,c,c,s,-30,30,22,34); return t; }
static Mat make_T(int TW){ Mat t(TW,TW,CV_8U,Scalar(0)); double c=TW/2.0,s=TW/120.0;
    boxInto(t,c,c,s,-34,34,-34,-22); boxInto(t,c,c,s,-6,6,-34,34); return t; }

static void place(Mat& scene, const Mat& templ, double X, double Y, double deg){
    double cx=templ.cols/2.0, cy=templ.rows/2.0;
    Mat R = getRotationMatrix2D(Point2f((float)cx,(float)cy), -deg, 1.0);
    R.at<double>(0,2) += X - cx;  R.at<double>(1,2) += Y - cy;
    Mat warped; warpAffine(templ, warped, R, scene.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    scene = max(scene, warped);
}

// Farthest-point sampling over EDGE refine points -> uniform spread, edge-preferring.
static std::vector<Point2f> edgeUniform(const sbm::FeatureSet& feat, int N){
    float hw = feat.templ_width/2.0f - 5, hh = feat.templ_height/2.0f - 5;
    std::vector<Point2f> cand;
    for(auto& rp : feat.refine_points){
        if(rp.type != sbm::FeatureSet::RefinePt::EDGE) continue;     // edges only
        if(std::abs(rp.px) > hw || std::abs(rp.py) > hh) continue;   // border margin
        cand.push_back(Point2f(rp.px, rp.py));
    }
    if((int)cand.size() < N){  // not enough edges -> fall back to all in-margin points
        cand.clear();
        for(auto& rp : feat.refine_points)
            if(std::abs(rp.px)<=hw && std::abs(rp.py)<=hh) cand.push_back(Point2f(rp.px,rp.py));
    }
    std::vector<Point2f> sel;
    if(cand.empty()) return sel;
    // seed: highest leverage (farthest from center)
    int seed=0; float bestR=-1;
    for(int i=0;i<(int)cand.size();i++){ float R=cand[i].x*cand[i].x+cand[i].y*cand[i].y; if(R>bestR){bestR=R;seed=i;} }
    sel.push_back(cand[seed]);
    std::vector<float> mind(cand.size(), 1e18f);
    while((int)sel.size() < N){
        Point2f last = sel.back();
        int best=-1; float bestd=-1;
        for(int i=0;i<(int)cand.size();i++){
            float d=(cand[i].x-last.x)*(cand[i].x-last.x)+(cand[i].y-last.y)*(cand[i].y-last.y);
            mind[i]=std::min(mind[i], d);
            if(mind[i]>bestd){ bestd=mind[i]; best=i; }
        }
        if(best<0 || bestd<=0) break;
        sel.push_back(cand[best]);
    }
    return sel;
}

// Edge-only D-optimal: greedily pick edges maximizing det(information matrix),
// optionally with a min spacing. Conditioning-aware (favours diverse normals +
// leverage) AND edge-preferring AND naturally spread. Uses stored normals (nx,ny).
static std::vector<Point2f> edgeDopt(const sbm::FeatureSet& feat, int N, float spacing){
    float hw = feat.templ_width/2.0f - 5, hh = feat.templ_height/2.0f - 5;
    struct E{ float px,py,nx,ny,j0,j1,j2; };
    std::vector<E> cand;
    for(auto& rp : feat.refine_points){
        if(rp.type != sbm::FeatureSet::RefinePt::EDGE) continue;
        if(std::abs(rp.px) > hw || std::abs(rp.py) > hh) continue;
        float nn = std::sqrt(rp.nx*rp.nx + rp.ny*rp.ny);
        if(nn < 1e-3f) continue;
        E e; e.px=rp.px; e.py=rp.py; e.nx=rp.nx/nn; e.ny=rp.ny/nn;
        e.j0 = -rp.py*e.nx + rp.px*e.ny; e.j1=e.nx; e.j2=e.ny;
        cand.push_back(e);
    }
    std::vector<Point2f> sel; std::vector<int> sidx;
    float I[3][3]={}; for(int i=0;i<3;i++) I[i][i]=1e-3f;
    auto det3=[](float A[3][3]){ return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
        - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]); };
    float sp2 = spacing*spacing;
    while((int)sel.size() < N){
        int best=-1; float bestd=-1e30f;
        for(int i=0;i<(int)cand.size();i++){
            bool used=false; for(int s:sidx) if(s==i){used=true;break;} if(used) continue;
            if(spacing>0){ bool close=false; for(auto& p:sel){ float dx=cand[i].px-p.x,dy=cand[i].py-p.y; if(dx*dx+dy*dy<sp2){close=true;break;} } if(close) continue; }
            float T[3][3]; for(int r=0;r<3;r++)for(int c=0;c<3;c++) T[r][c]=I[r][c];
            float j0=cand[i].j0,j1=cand[i].j1,j2=cand[i].j2;
            T[0][0]+=j0*j0;T[0][1]+=j0*j1;T[0][2]+=j0*j2;T[1][0]+=j1*j0;T[1][1]+=j1*j1;T[1][2]+=j1*j2;T[2][0]+=j2*j0;T[2][1]+=j2*j1;T[2][2]+=j2*j2;
            float d=det3(T); if(d>bestd){bestd=d;best=i;}
        }
        if(best<0) break;
        sidx.push_back(best); sel.push_back(Point2f(cand[best].px,cand[best].py));
        float j0=cand[best].j0,j1=cand[best].j1,j2=cand[best].j2;
        I[0][0]+=j0*j0;I[0][1]+=j0*j1;I[0][2]+=j0*j2;I[1][0]+=j1*j0;I[1][1]+=j1*j1;I[1][2]+=j1*j2;I[2][0]+=j2*j0;I[2][1]+=j2*j1;I[2][2]+=j2*j2;
    }
    return sel;
}

static float minPair(const std::vector<Point2f>& p){
    float m=1e9; for(size_t i=0;i<p.size();i++) for(size_t j=i+1;j<p.size();j++)
        m=std::min(m,(float)std::hypot(p[i].x-p[j].x,p[i].y-p[j].y));
    return p.size()<2?0:m;
}

// mode: 0=default 2D, 1=edge-FPS, 2=edgeDopt 2D, 3=edgeDopt 1D,
//       4=edgeDopt 1D + iterative rematch, 5=edgeDopt 2D + iterative rematch
// out[6] = {minpair, npts, o_mean_clean, o_worst_clean, o_mean_n30, o_worst_n30}
static void run(const Mat& templ, int mode, float out[6]){
    int TW=templ.cols;
    auto feat = sbm::extractFeatures(templ); feat.setOrigin(TW/2.0f,TW/2.0f);
    std::vector<Point2f> pts;
    if(mode==1){ pts = edgeUniform(feat, 8); feat.cached_opt_points=pts; feat.cached_opt_max_points=8; }
    else if(mode>=2){ pts = edgeDopt(feat, 8, 12.0f); feat.cached_opt_points=pts; feat.cached_opt_max_points=8; }
    else { pts = feat.selectOptimizedPoints(8); }
    out[0]=minPair(pts); out[1]=(float)pts.size();
    sbm::MatchConfig cfg; cfg.min_score=40; cfg.nms_radius=60; cfg.refine=sbm::RefineMode::ROI;
    cfg.roi_edge_1d_match = (mode==3 || mode==4);
    cfg.roi_iterative_rematch = (mode==4 || mode==5);
    cfg.roi_edge_collapse = (mode==6);
    sbm::ShapeMatcher m(cfg);
    sbm::ModelConfig mc; mc.angle={0,360,1}; mc.flip=false;
    m.addModel("s", feat, mc);
    const int W=400,H=400; cv::RNG rng(7);
    for(int pass=0; pass<2; pass++){
        int ns=pass==0?0:30; double so=0,wo=0; int n=0;
        for(int deg=0; deg<360; deg+=15){
            Mat clean(H,W,CV_8U,Scalar(0)); double X=200,Y=200; place(clean,templ,X,Y,deg);
            int reps=ns>0?3:1;
            for(int r=0;r<reps;r++){
                Mat sc=clean.clone(); if(ns>0){ Mat nz(H,W,CV_8U); rng.fill(nz,cv::RNG::NORMAL,0,ns); sc+=nz; }
                auto rs=m.match(sc); const sbm::MatchResult* b=nullptr; float bs=-1;
                for(auto& rr:rs){ float d=(float)std::hypot(rr.x-X,rr.y-Y); if(d<60&&rr.score>bs){bs=rr.score;b=&rr;} }
                if(!b) continue; double o=std::hypot(b->x-X,b->y-Y); so+=o; wo=std::max(wo,o); n++;
            }
        }
        out[2+pass*2]= n?so/n:-1; out[3+pass*2]=(float)wo;
    }
}

int main(){
    const int TW=140;
    struct S{ const char* nm; Mat im; };
    std::vector<S> shapes = {{"F",make_F(TW)},{"flag",make_flag(TW)},{"L",make_L(TW)},{"T",make_T(TW)}};
    printf("Edge-uniform (FPS over edges) vs default (corner-priority). o=origin err mean|worst (px).\n\n");
    printf("%-6s %-13s | %-5s %-4s | clean o(mean|worst) | noise30 o(mean|worst)\n","shape","selection","minpr","npt");
    printf("------------------------------------------------------------------------------------\n");
    for(auto& s : shapes){
        float d[6], g[6], h[6], c[6];
        run(s.im,0,d); run(s.im,2,g); run(s.im,3,h); run(s.im,6,c);
        printf("%-6s %-14s | %5.1f %4.0f | %6.2f | %-9.2f | %6.2f | %-9.2f\n",
               s.nm,"default(2D)",  d[0],d[1], d[2],d[3], d[4],d[5]);
        printf("%-6s %-14s | %5.1f %4.0f | %6.2f | %-9.2f | %6.2f | %-9.2f\n",
               "",   "edgeDopt-2D",  g[0],g[1], g[2],g[3], g[4],g[5]);
        printf("%-6s %-14s | %5.1f %4.0f | %6.2f | %-9.2f | %6.2f | %-9.2f\n",
               "",   "edgeDopt-1D",  h[0],h[1], h[2],h[3], h[4],h[5]);
        printf("%-6s %-14s | %5.1f %4.0f | %6.2f | %-9.2f | %6.2f | %-9.2f\n",
               "",   "edge-collapse", c[0],c[1], c[2],c[3], c[4],c[5]);
    }
    return 0;
}
