// Compiled ONCE PER ISA (see CMakeLists): -mavx2 -mfma with LINE2DUP_ISA=avx2,
// -msse4.2 with LINE2DUP_ISA=base. Each copy forwards IDetector to the Detector
// of its own namespace. Nothing here is ISA-specific by hand -- the flags on the
// translation unit are the whole difference.
#include "detector_iface.h"

namespace line2Dup {
namespace LINE2DUP_ISA {

namespace {
class Adapter final : public IDetector {
public:
    Adapter(int num_features, std::vector<int> T, float weak, float strong)
        : det_(num_features, std::move(T), weak, strong) {}

    int addTemplate(const cv::Mat sources, const std::string &class_id,
                    const cv::Mat &object_mask, int num_features) override {
        return det_.addTemplate(sources, class_id, object_mask, num_features);
    }
    const std::vector<Template> &getTemplates(const std::string &class_id,
                                              int template_id) const override {
        return det_.getTemplates(class_id, template_id);
    }
    std::vector<std::vector<Template>> &getClassTemplates(const std::string &class_id) override {
        return det_.getClassTemplates(class_id);
    }
    std::vector<Match> match(cv::Mat sources, float threshold,
                             const std::vector<std::string> &class_ids,
                             const cv::Mat masks) const override {
        return det_.match(sources, threshold, class_ids, masks);
    }
    void setBlurKernelSize(int k) override { det_.getModalities()->blur_kernel_size = k; }
    void setSkipVoting(bool v)   override { det_.getModalities()->skip_voting = v; }

private:
    Detector det_;
};
} // namespace

std::unique_ptr<IDetector> makeDetectorImpl(int num_features, std::vector<int> T,
                                            float weak_thresh, float strong_thresh) {
    return std::unique_ptr<IDetector>(
        new Adapter(num_features, std::move(T), weak_thresh, strong_thresh));
}

void enableProfilingImpl(bool enable) { enableProfiling(enable); }
void resetProfilingImpl()             { resetProfiling(); }
void printProfilingImpl()             { printProfiling(); }

} // namespace LINE2DUP_ISA
} // namespace line2Dup
