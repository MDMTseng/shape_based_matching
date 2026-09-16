#ifndef LINE2DUP_DETECTOR_IFACE_H
#define LINE2DUP_DETECTOR_IFACE_H
//
// One binary, two instruction-set builds of the matcher, chosen when the first
// Detector is created.
//
// Why this exists: line2Dup carries hand-written AVX2 in its hot loops, guarded
// by #ifdef __AVX2__ with scalar fallbacks. Compiling the library -mavx2 made
// those unconditional -- correct on the machines in the field, and a silent
// startup death on a CPU without AVX (a Pentium Gold 4415Y, Kaby Lake with AVX
// fused off, 2026-09-16). Compiling it -msse4.2 instead costs ~2.1x on the
// match path, measured. Neither is a good default, so neither is the default:
// line2Dup.cpp is compiled TWICE, into line2Dup::avx2 and line2Dup::base, and
// __builtin_cpu_supports picks one at runtime.
//
// Only the SIMD-carrying classes are duplicated. Feature, Template and Match
// are compiled once and shared, so results cross the ISA boundary as themselves
// -- there is no struct conversion anywhere in this design, and therefore no
// place for a field to go quietly missing.
//
#include "line2Dup.h"
#include <memory>
#include <string>
#include <vector>

namespace line2Dup {

/// The slice of Detector that shape_matcher.cpp actually uses. Kept to exactly
/// that: every method here is one an existing call site needs, and the two
/// getModalities() callers became the two setters, so ColorGradient -- which is
/// itself ISA-specific -- never has to cross this boundary.
struct IDetector {
    virtual ~IDetector() = default;

    virtual int addTemplate(const cv::Mat sources, const std::string &class_id,
                            const cv::Mat &object_mask, int num_features = 0) = 0;

    virtual const std::vector<Template> &getTemplates(const std::string &class_id,
                                                      int template_id) const = 0;

    virtual std::vector<std::vector<Template>> &getClassTemplates(const std::string &class_id) = 0;

    virtual std::vector<Match> match(cv::Mat sources, float threshold,
                                     const std::vector<std::string> &class_ids,
                                     const cv::Mat masks = cv::Mat()) const = 0;

    /// Stand in for getModalities()->blur_kernel_size / ->skip_voting.
    virtual void setBlurKernelSize(int k) = 0;
    virtual void setSkipVoting(bool v) = 0;
};

/// true when this CPU runs the AVX2 build. Cheap after the first call.
bool usingAVX2();

/// Name of the active build ("avx2" / "sse4.2"), for logs and diagnostics --
/// worth printing at startup, because "why is this machine half speed" is
/// otherwise invisible.
const char *activeISA();

std::unique_ptr<IDetector> makeDetector(int num_features, std::vector<int> T,
                                        float weak_thresh, float strong_thresh);

// Profiling, routed to whichever build is live.
void enableProfilingDispatch(bool enable);
void resetProfilingDispatch();
void printProfilingDispatch();

} // namespace line2Dup

#endif
