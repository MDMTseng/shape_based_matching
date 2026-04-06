#ifndef CXXLINEMOD_H
#define CXXLINEMOD_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <map>

#include "mipp.h"  // for SIMD in different platforms

namespace line2Dup
{

struct Feature
{
    int x;
    int y;
    int label;
    float theta;

    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;

    Feature() : x(0), y(0), label(0) {}
    Feature(int x, int y, int label);
};
inline Feature::Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}

struct Template
{
    int width;
    int height;
    int tl_x;
    int tl_y;
    int pyramid_level;
    float angle = 0;  ///< Rotation angle in degrees (for feature-rotated templates)
    std::vector<Feature> features;

    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;
};

class ColorGradientPyramid
{
public:
    ColorGradientPyramid() : pyramid_level(0), weak_threshold(30), num_features(128),
                             strong_threshold(60), match_only(false) {}
    ColorGradientPyramid(const cv::Mat &src, const cv::Mat &mask,
                                             float weak_threshold, size_t num_features,
                                             float strong_threshold,
                                             bool match_only = false);

    void quantize(cv::Mat &dst) const;

    bool extractTemplate(Template &templ) const;

    void pyrDown();

public:
    bool match_only;  ///< Skip magnitude/angle_ori computation (matching only needs angle)
    void update();
    /// Candidate feature with a score
    struct Candidate
    {
        Candidate(int x, int y, int label, float score);

        /// Sort candidates with high score to the front
        bool operator<(const Candidate &rhs) const
        {
            return score > rhs.score;
        }

        Feature f;
        float score;
    };

    cv::Mat src;
    cv::Mat mask;

    int pyramid_level;
    cv::Mat angle;
    cv::Mat magnitude;
    cv::Mat angle_ori;

    float weak_threshold;
    size_t num_features;
    float strong_threshold;
    int blur_kernel_size;  ///< Gaussian blur kernel size before Sobel (default 7, increase for noise)
    bool skip_voting = false;  ///< Skip 3x3 neighborhood voting
    static bool selectScatteredFeatures(const std::vector<Candidate> &candidates,
                                                                            std::vector<Feature> &features,
                                                                            size_t num_features, float distance);
};
inline ColorGradientPyramid::Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}

class ColorGradient
{
public:
    ColorGradient();
    ColorGradient(float weak_threshold, size_t num_features, float strong_threshold);

    std::string name() const;

    float weak_threshold;
    size_t num_features;
    float strong_threshold;
    int blur_kernel_size = 7;  ///< Gaussian blur before Sobel (7=default, increase for noise e.g. 11, 15)
    bool skip_voting = true;   ///< Skip 3x3 neighborhood voting (spread already covers it)
    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;

    cv::Ptr<ColorGradientPyramid> process(const cv::Mat src, const cv::Mat &mask = cv::Mat(),
                                          bool match_only = false) const
    {
        // Create without update, set config, then update once with correct settings.
        // The constructor normally calls update() but with default skip_voting=false,
        // which prevents the fused Sobel+Quantize path from running.
        auto p = cv::Ptr<ColorGradientPyramid>(new ColorGradientPyramid());
        p->src = src;
        p->mask = mask;
        p->pyramid_level = 0;
        p->weak_threshold = weak_threshold;
        p->num_features = num_features;
        p->strong_threshold = strong_threshold;
        p->match_only = match_only;
        p->blur_kernel_size = blur_kernel_size;
        p->skip_voting = skip_voting;
        p->update();
        return p;
    }
};

struct Match
{
    Match()
    {
    }

    Match(int x, int y, float similarity, const std::string &class_id, int template_id);

    /// Sort matches with high similarity to the front
    bool operator<(const Match &rhs) const
    {
        // Secondarily sort on template_id for the sake of duplicate removal
        if (similarity != rhs.similarity)
            return similarity > rhs.similarity;
        else
            return template_id < rhs.template_id;
    }

    bool operator==(const Match &rhs) const
    {
        return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id;
    }

    int x;
    int y;
    float similarity;
    float refined_angle;  ///< Sub-step refined angle in degrees (-1 if not refined)
    std::string class_id;
    int template_id;
};

inline Match::Match(int _x, int _y, float _similarity, const std::string &_class_id, int _template_id)
        : x(_x), y(_y), similarity(_similarity), refined_angle(-1), class_id(_class_id), template_id(_template_id)
{
}

class Detector
{
public:
    /**
         * \brief Empty constructor, initialize with read().
         */
    Detector();

    Detector(std::vector<int> T);
    Detector(int num_features, std::vector<int> T, float weak_thresh = 30.0f, float strong_thresh = 60.0f);

    std::vector<Match> match(cv::Mat sources, float threshold,
                                                     const std::vector<std::string> &class_ids = std::vector<std::string>(),
                                                     const cv::Mat masks = cv::Mat()) const;

    /// Refine orientation of matches by parabolic interpolation between
    /// neighboring template angles. Call after match() and spatial NMS.
    /// @param angle_step  Degrees between consecutive template_ids.
    /// @param num_templates  Total number of rotation templates (for wraparound).
    void refineOrientations(std::vector<Match> &matches, float angle_step,
                            int num_templates) const;

    int addTemplate(const cv::Mat sources, const std::string &class_id,
                                    const cv::Mat &object_mask, int num_features = 0);

    int addTemplate_rotate(const std::string &class_id, int zero_id, float theta, cv::Point2f center);

    /// Batch-add rotated templates from a single 0-degree extraction.
    /// Extracts features once from templ_gray, then mathematically rotates
    /// feature coordinates + orientation labels for all angles.
    /// Eliminates warpAffine bias and is faster than per-angle extraction.
    /// @param templ_gray  Template image at 0 degrees.
    /// @param object_mask Template mask (255 = object, 0 = background).
    /// @param class_id    Template class name.
    /// @param angle_start First angle in degrees.
    /// @param angle_end   Last angle in degrees (exclusive).
    /// @param angle_step  Angle increment in degrees.
    /// @return Number of templates added, or -1 on failure.
    int addRotatedTemplates(const cv::Mat& templ_gray, const cv::Mat& object_mask,
                            const std::string& class_id,
                            float angle_start = 0, float angle_end = 360,
                            float angle_step = 2);

    const cv::Ptr<ColorGradient> &getModalities() const { return modality; }
    cv::Ptr<ColorGradient> &getModalities() { return modality; }

    int getT(int pyramid_level) const { return T_at_level[pyramid_level]; }

    int pyramidLevels() const { return pyramid_levels; }

    const std::vector<Template> &getTemplates(const std::string &class_id, int template_id) const;

    int numTemplates() const;
    int numTemplates(const std::string &class_id) const;
    int numClasses() const { return static_cast<int>(class_templates.size()); }

    /// Get mutable access to template pyramids for a class (for feature scaling).
    std::vector<std::vector<Template>>& getClassTemplates(const std::string& class_id) {
        return class_templates[class_id];
    }

    std::vector<std::string> classIds() const;

    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;

    std::string readClass(const cv::FileNode &fn, const std::string &class_id_override = "");
    void writeClass(const std::string &class_id, cv::FileStorage &fs) const;

    void readClasses(const std::vector<std::string> &class_ids,
                                     const std::string &format = "templates_%s.yml.gz");
    void writeClasses(const std::string &format = "templates_%s.yml.gz") const;

    /// When true, extract features only at level 0 and scale coordinates
    /// for coarser levels. Orientation labels from full-res stay valid.
    /// When false (default), re-extract features independently at each level.
    bool scale_pyramid_features = false;

protected:
    cv::Ptr<ColorGradient> modality;
    int pyramid_levels;
    std::vector<int> T_at_level;

    typedef std::vector<Template> TemplatePyramid;
    typedef std::map<std::string, std::vector<TemplatePyramid>> TemplatesMap;
    TemplatesMap class_templates;

    typedef std::vector<cv::Mat> LinearMemories;
    // Indexed as [pyramid level][ColorGradient][quantized label]
    typedef std::vector<std::vector<LinearMemories>> LinearMemoryPyramid;

    void matchClass(const LinearMemoryPyramid &lm_pyramid,
                                    const std::vector<cv::Size> &sizes,
                                    float threshold, std::vector<Match> &matches,
                                    const std::string &class_id,
                                    const std::vector<TemplatePyramid> &template_pyramids) const;
};

// Per-stage profiling control
void enableProfiling(bool enable);
void resetProfiling();
void printProfiling();

} // namespace line2Dup

namespace shape_based_matching {
class shapeInfo_producer{
public:
    cv::Mat src;
    cv::Mat mask;

    std::vector<float> angle_range;
    std::vector<float> scale_range;

    float angle_step = 15;
    float scale_step = 0.5;
    float eps = 0.00001f;

    class Info{
    public:
        float angle;
        float scale;

        Info(float angle_, float scale_){
            angle = angle_;
            scale = scale_;
        }
    };
    std::vector<Info> infos;

    shapeInfo_producer(cv::Mat src, cv::Mat mask = cv::Mat()){
        this->src = src;
        if(mask.empty()){
            // make sure we have masks
            this->mask = cv::Mat(src.size(), CV_8UC1, {255});
        }else{
            this->mask = mask;
        }
    }

    static cv::Mat transform(cv::Mat src, float angle, float scale){
        cv::Mat dst;

        cv::Point2f center(src.cols/2.0f, src.rows/2.0f);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
        cv::warpAffine(src, dst, rot_mat, src.size());

        return dst;
    }
    static void save_infos(std::vector<shapeInfo_producer::Info>& infos, std::string path = "infos.yaml"){
        cv::FileStorage fs(path, cv::FileStorage::WRITE);

        fs << "infos"
           << "[";
        for (int i = 0; i < infos.size(); i++)
        {
            fs << "{";
            fs << "angle" << infos[i].angle;
            fs << "scale" << infos[i].scale;
            fs << "}";
        }
        fs << "]";
    }
    static std::vector<Info> load_infos(std::string path = "info.yaml"){
        cv::FileStorage fs(path, cv::FileStorage::READ);

        std::vector<Info> infos;

        cv::FileNode infos_fn = fs["infos"];
        cv::FileNodeIterator it = infos_fn.begin(), it_end = infos_fn.end();
        for (int i = 0; it != it_end; ++it, i++)
        {
            infos.emplace_back(float((*it)["angle"]), float((*it)["scale"]));
        }
        return infos;
    }

    void produce_infos(){
        infos.clear();

        assert(angle_range.size() <= 2);
        assert(scale_range.size() <= 2);
        assert(angle_step > eps*10);
        assert(scale_step > eps*10);

        // make sure range not empty
        if(angle_range.size() == 0){
            angle_range.push_back(0);
        }
        if(scale_range.size() == 0){
            scale_range.push_back(1);
        }

        if(angle_range.size() == 1 && scale_range.size() == 1){
            float angle = angle_range[0];
            float scale = scale_range[0];
            infos.emplace_back(angle, scale);

        }else if(angle_range.size() == 1 && scale_range.size() == 2){
            assert(scale_range[1] > scale_range[0]);
            float angle = angle_range[0];
            for(float scale = scale_range[0]; scale <= scale_range[1]+eps; scale += scale_step){
                infos.emplace_back(angle, scale);
            }
        }else if(angle_range.size() == 2 && scale_range.size() == 1){
            assert(angle_range[1] > angle_range[0]);
            float scale = scale_range[0];
            for(float angle = angle_range[0]; angle <= angle_range[1]+eps; angle += angle_step){
                infos.emplace_back(angle, scale);
            }
        }else if(angle_range.size() == 2 && scale_range.size() == 2){
            assert(scale_range[1] > scale_range[0]);
            assert(angle_range[1] > angle_range[0]);
            for(float scale = scale_range[0]; scale <= scale_range[1]+eps; scale += scale_step){
                for(float angle = angle_range[0]; angle <= angle_range[1]+eps; angle += angle_step){
                    infos.emplace_back(angle, scale);
                }
            }
        }
    }

    cv::Mat src_of(const Info& info){
        return transform(src, info.angle, info.scale);
    }

    cv::Mat mask_of(const Info& info){
        return (transform(mask, info.angle, info.scale) > 0);
    }
};

}

#endif
