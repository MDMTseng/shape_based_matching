// bench_stablevote.cpp — repeatability-voted feature selection.
//
// bench_realrot showed per-angle real extraction makes BAD templates (extraction
// noise + lost point correspondence). This uses per-angle extraction the OTHER
// way — not as templates, but as VOTES to find which base-frame locations
// reliably re-appear under rotation:
//
//   for each rotation k in a set:
//       rot = rotate(image, k); extract features; rotate the points BACK by -k
//       into the base (upright) frame.
//   a base-frame location detected in MANY rotations = rotation-stable (a true
//   corner/junction). one detected in few = an edge point that slides, or noise.
//   keep the high-vote locations, then scatter-select topM among them (uniform
//   spatial spread). Build the templates with ANALYTIC rotation (proven best in
//   bench_realrot) from that curated base.
//
// This differs from selectRotationStable(): that keeps base points whose
// ANALYTICALLY-rotated ORIENTATION still matches the real rotated template
// (orientation consistency, fixed positions). This one keeps points that are
// actually RE-DETECTED at the same place across real rotations (detection
// repeatability), then spreads them. Orthogonal signals; this bench measures
// whether repeatability-voting beats plain scatter selection at equal feature
// count and at coarse angle steps.
//
//   cmake --build build --target bench_stablevote
//   ./bench_stablevote           # built-in feature-rich shape
//   ./bench_stablevote my.png

#include "shape_matcher.h"
#include "sbm_log.h"
#include <opencv2/opencv.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>   // getRotationMatrix2D moved here in OpenCV 5
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#if defined(__AVX2__)
#  define SBM_SIMD "AVX2"
#elif defined(__aarch64__)
#  define SBM_SIMD "NEON"
#else
#  define SBM_SIMD "scalar"
#endif

using Clk = std::chrono::high_resolution_clock;

static cv::Mat builtin_shape() {
    cv::Mat m(160, 160, CV_8U, cv::Scalar(40));
    cv::rectangle(m, {30,30}, {130,130}, 230, 3);
    cv::line(m, {30,30}, {130,130}, 180, 2);
    cv::line(m, {130,30}, {30,130}, 180, 2);
    cv::rectangle(m, {55,55}, {105,105}, 120, 2);
    cv::circle(m, {80,80}, 18, 200, 2);
    return m;
}

// ---- greedy scatter selection: pick M points, vote-priority + min spacing ----
// cands sorted by caller priority; grow the min-distance until <=M survive.
static std::vector<int> scatterSelect(const std::vector<cv::Point2f>& pts,
                                      const std::vector<int>& priority, int M) {
    if ((int)pts.size() <= M) {
        std::vector<int> all(pts.size()); for (size_t i=0;i<pts.size();++i) all[i]=(int)i;
        return all;
    }
    // order indices by priority desc (higher vote first)
    std::vector<int> order(pts.size());
    for (size_t i=0;i<pts.size();++i) order[i]=(int)i;
    std::stable_sort(order.begin(), order.end(),
                     [&](int a,int b){ return priority[a] > priority[b]; });
    // binary-search a spacing radius so exactly ~M survive greedy acceptance.
    float lo=0, hi=0;
    for (auto&p:pts) hi=std::max(hi, std::abs(p.x)+std::abs(p.y));
    std::vector<int> best;
    for (int it=0; it<24; ++it) {
        float r=(lo+hi)*0.5f;
        std::vector<int> sel;
        for (int idx : order) {
            bool ok=true;
            for (int s : sel) {
                float dx=pts[idx].x-pts[s].x, dy=pts[idx].y-pts[s].y;
                if (dx*dx+dy*dy < r*r) { ok=false; break; }
            }
            if (ok) sel.push_back(idx);
        }
        if ((int)sel.size() >= M) { best=sel; lo=r; } else hi=r;
    }
    if (best.empty()) best=order;
    best.resize(std::min((int)best.size(), M));
    return best;
}

// ---- build a repeatability-voted base FeatureSet ----
static sbm::FeatureSet buildStableVoted(const cv::Mat& img, const sbm::MatchConfig& cfg,
                                        int pool, int M, float voteStep, float voteRange,
                                        float radius, float keep_frac, bool verbose) {
    sbm::FeatureSet fs0 = sbm::extractFeatures(img, cv::Mat(), pool,
                          cfg.T_levels, cfg.weak_threshold, cfg.strong_threshold);
    cv::Point2f center(img.cols/2.f, img.rows/2.f);

    std::vector<float> angs;
    for (float a=-voteRange; a<=voteRange+1e-3f; a+=voteStep)
        if (std::abs(a) > 1e-3f) angs.push_back(a);

    // Per level: vote counts for each fs0 candidate.
    std::vector<std::vector<int>> votes(fs0.levels.size());
    for (size_t l=0; l<fs0.levels.size(); ++l)
        votes[l].assign(fs0.levels[l].features.size(), 0);

    for (float k : angs) {
        cv::Mat rot, R = cv::getRotationMatrix2D(center, k, 1.0);
        cv::warpAffine(img, rot, R, img.size(), cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, cv::Scalar(40));
        sbm::FeatureSet fsk = sbm::extractFeatures(rot, cv::Mat(), pool,
                              cfg.T_levels, cfg.weak_threshold, cfg.strong_threshold);
        for (size_t l=0; l<fs0.levels.size() && l<fsk.levels.size(); ++l) {
            cv::Point2f lc = center * (float)std::pow(0.5, (int)l);
            cv::Mat Rb = cv::getRotationMatrix2D(lc, -k, 1.0);   // map extracted pts home
            const auto& kl = fsk.levels[l];
            const auto& bl = fs0.levels[l];
            for (size_t j=0; j<kl.features.size(); ++j) {
                float gx=kl.features[j].x+kl.tl_x, gy=kl.features[j].y+kl.tl_y;
                cv::Point2f home(Rb.at<double>(0,0)*gx+Rb.at<double>(0,1)*gy+Rb.at<double>(0,2),
                                 Rb.at<double>(1,0)*gx+Rb.at<double>(1,1)*gy+Rb.at<double>(1,2));
                // does 'home' fall near a base candidate? (nearest within radius votes)
                int best=-1; float bd=radius*radius;
                for (size_t i=0;i<bl.features.size();++i){
                    float dx=(bl.features[i].x+bl.tl_x)-home.x;
                    float dy=(bl.features[i].y+bl.tl_y)-home.y;
                    float d=dx*dx+dy*dy; if (d<bd){bd=d;best=(int)i;}
                }
                if (best>=0) votes[l][best]++;
            }
        }
    }

    int needVotes = std::max(1, (int)(angs.size()*keep_frac));
    sbm::FeatureSet out = fs0;
    int total_kept=0, total_pool=0;
    for (size_t l=0; l<fs0.levels.size(); ++l) {
        const auto& bl = fs0.levels[l];
        std::vector<cv::Point2f> pts; std::vector<int> pri; std::vector<int> src;
        for (size_t i=0;i<bl.features.size();++i)
            if (votes[l][i] >= needVotes) {
                pts.push_back({(float)bl.features[i].x,(float)bl.features[i].y});
                pri.push_back(votes[l][i]); src.push_back((int)i);
            }
        total_pool += (int)bl.features.size();
        // scale M per level roughly by candidate share (finest level dominates).
        int Ml = (l==0)? M : std::max(4, M/(1<<l));
        std::vector<int> pick = scatterSelect(pts, pri, Ml);
        std::vector<sbm::FeatureSet::Feature> kept;
        for (int p : pick) kept.push_back(bl.features[src[p]]);
        total_kept += (int)kept.size();
        out.levels[l].features = std::move(kept);
    }
    if (verbose)
        std::printf("  [stablevote] pool=%d, votes>=%d/%zu -> kept %d features "
                    "(step %.0f, range +-%.0f, r=%.1f)\n",
                    total_pool, needVotes, angs.size(), total_kept, voteStep, voteRange, radius);
    return out;
}

struct TestCase { cv::Mat scene; float cx, cy; };
static TestCase make_case(const cv::Mat& tmpl, float angleDeg, double sigma,
                          int W,int H,int px,int py){
    cv::Mat rot, R=cv::getRotationMatrix2D({tmpl.cols/2.f,tmpl.rows/2.f}, angleDeg, 1.0);
    cv::warpAffine(tmpl,rot,R,tmpl.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(40));
    cv::Mat scene(H,W,CV_8U,cv::Scalar(40));
    rot.copyTo(scene(cv::Rect(px,py,tmpl.cols,tmpl.rows)));
    cv::Mat noise(H,W,CV_8U); cv::randn(noise,0,sigma); cv::add(scene,noise,scene);
    return {scene, px+tmpl.cols/2.f, py+tmpl.rows/2.f};
}

struct Eval { float detect,worst,mean; double match_ms; };
static Eval evaluate(sbm::ShapeMatcher& m, const std::vector<TestCase>& t, float tol){
    std::vector<float> raw(t.size(),-1.f); double tot=0;
    for (size_t i=0;i<t.size();++i){
        auto t0=Clk::now(); auto rs=m.match(t[i].scene);
        tot+=std::chrono::duration<double,std::milli>(Clk::now()-t0).count();
        for(auto&r:rs) if(std::abs(r.x-t[i].cx)<tol&&std::abs(r.y-t[i].cy)<tol)
            raw[i]=std::max(raw[i],r.score);
    }
    int det=0; float w=1e9f,s=0;
    for(float v:raw) if(v>0){det++;w=std::min(w,v);s+=v;}
    return {(float)det/t.size(), det?w:0.f, det?s/det:0.f, tot/t.size()};
}

int main(int argc, char** argv){
    sbm::setLogLevel(sbm::LogLevel::Warning);
    cv::Mat tmpl;
    if (argc>1) tmpl=cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
    if (tmpl.empty()) tmpl=builtin_shape();

    const int W=1280,H=960,px=560,py=400; const double sigma=10.0; const float tol=12.f;
    const int M = argc>2 ? std::atoi(argv[2]) : 100;
    const int POOL=400;

    std::vector<TestCase> tests;
    for(int i=0;i<24;++i) tests.push_back(make_case(tmpl,i*15.37f+2.6f,sigma,W,H,px,py));

    std::printf("bench_stablevote | %s | tmpl %dx%d | %zu off-grid poses @ %dx%d sigma=%.0f | M=%d\n",
                SBM_SIMD, tmpl.cols, tmpl.rows, tests.size(), W, H, sigma, M);
    std::printf("ORIGINAL = extractFeatures scatter-to-M | STABLEVOTE = repeatability-voted top-M\n"
                "both use ANALYTIC rotation for templates.\n\n");

    sbm::MatchConfig cfg; cfg.min_score=30; cfg.refine=sbm::RefineMode::None; cfg.skip_voting=true;

    // Build the voted base ONCE (independent of angle step).
    std::printf("building STABLEVOTE base:\n");
    sbm::FeatureSet voted = buildStableVoted(tmpl, cfg, POOL, M, /*step*/30, /*range*/150,
                                             /*radius*/3.5f, /*keep_frac*/0.5f, true);
    sbm::FeatureSet orig = sbm::extractFeatures(tmpl, cv::Mat(), M,
                           cfg.T_levels, cfg.weak_threshold, cfg.strong_threshold);
    auto fcount=[](const sbm::FeatureSet&f){int n=0;for(auto&l:f.levels)n+=l.features.size();return n;};
    std::printf("  ORIGINAL features=%d | STABLEVOTE features=%d\n\n", fcount(orig), fcount(voted));

    const float STEPS[]={1,5,15};
    std::printf("%-11s %-5s %-6s | %-8s %-7s %-7s | %-8s\n",
                "base","step","#tmpl","detect","worst","mean","match_ms");
    std::printf("------------------------------------------------------------------\n");
    for(float step:STEPS){
        for(int which=0;which<2;++which){
            const sbm::FeatureSet& fs = which? voted : orig;
            auto m=std::make_unique<sbm::ShapeMatcher>(cfg);
            sbm::ModelConfig mc; mc.angle={0,360,step};
            int nt=m->addModel("m",fs,mc);
            Eval e=evaluate(*m,tests,tol);
            std::printf("%-11s %-5.0f %-6d | %6.0f%%  %6.1f  %6.1f | %8.2f\n",
                        which?"STABLEVOTE":"ORIGINAL", step, nt,
                        e.detect*100, e.worst, e.mean, e.match_ms);
        }
        std::printf("------------------------------------------------------------------\n");
    }
    std::printf("\nRead: if STABLEVOTE holds worst/mean above ORIGINAL at equal #tmpl,\n"
                "repeatability-voting picks more rotation-robust points than plain\n"
                "scatter. Biggest expected gain at coarse steps (15) + off-grid poses.\n");
    return 0;
}
