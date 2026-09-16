// Compiled ONCE, with no ISA flags: it must run on the oldest CPU we support,
// because it is what decides whether the AVX2 half may be entered at all.
#include "detector_iface.h"
#include <cstdlib>
#include <cstring>

namespace line2Dup {

// Declared, not included: these return ISA-independent types, so this file never
// has to see either build's Detector -- which is also why it cannot accidentally
// call one.
namespace avx2 {
std::unique_ptr<IDetector> makeDetectorImpl(int, std::vector<int>, float, float);
void enableProfilingImpl(bool); void resetProfilingImpl(); void printProfilingImpl();
}
namespace base {
std::unique_ptr<IDetector> makeDetectorImpl(int, std::vector<int>, float, float);
void enableProfilingImpl(bool); void resetProfilingImpl(); void printProfilingImpl();
}

bool usingAVX2() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    // Function-local static: computed once, thread-safe, and after libgcc's
    // __cpu_indicator_init has run. FMA is checked too because the AVX2 build
    // is compiled -mavx2 -mfma; every shipping AVX2 part has it, but a machine
    // that somehow does not must take the portable half rather than fault.
    static const bool yes = [] {
        // SBM_FORCE_ISA=base|avx2 overrides the probe. Two uses: proving the
        // two builds agree (run a fleet sweep each way and diff the judge
        // values), and answering "is this machine on the slow half?" in the
        // field without a rebuild. Forcing avx2 on a CPU without it WILL
        // crash -- that is the point of it being an explicit override.
        if (const char *e = std::getenv("SBM_FORCE_ISA")) {
            if (std::strcmp(e, "base") == 0) return false;
            if (std::strcmp(e, "avx2") == 0) return true;
        }
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    return yes;
#else
    return false;
#endif
}

const char *activeISA() { return usingAVX2() ? "avx2" : "sse4.2"; }

std::unique_ptr<IDetector> makeDetector(int num_features, std::vector<int> T,
                                        float weak_thresh, float strong_thresh) {
    return usingAVX2()
        ? avx2::makeDetectorImpl(num_features, std::move(T), weak_thresh, strong_thresh)
        : base::makeDetectorImpl(num_features, std::move(T), weak_thresh, strong_thresh);
}

void enableProfilingDispatch(bool e) { usingAVX2() ? avx2::enableProfilingImpl(e) : base::enableProfilingImpl(e); }
void resetProfilingDispatch()        { usingAVX2() ? avx2::resetProfilingImpl()   : base::resetProfilingImpl(); }
void printProfilingDispatch()        { usingAVX2() ? avx2::printProfilingImpl()   : base::printProfilingImpl(); }

} // namespace line2Dup
