#pragma once
/// @file test_utils.h
/// @brief Shared utilities for test programs.

#include <iostream>
#include <sstream>
#include <cstdio>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif

/// RAII guard that suppresses ALL output (cout + stderr) from library code.
/// - cout is suppressed via rdbuf swap (for any remaining C++ stream output)
/// - stderr is suppressed via fd-level dup2 (for sbm_log warnings/errors and printf)
/// - stdout printf from library is suppressed because sbm_log uses stderr by default;
///   legacy printf output (StageProfile) is now routed through sbm_log too.
struct OutputGuard {
    std::streambuf* orig_cout;
    std::ostringstream sink;
    int orig_stderr_fd;

    OutputGuard() : orig_cout(std::cout.rdbuf()), orig_stderr_fd(-1) {
        std::cout.rdbuf(sink.rdbuf());
        fflush(stderr);
#ifdef _WIN32
        orig_stderr_fd = _dup(_fileno(stderr));
        FILE* nul = nullptr;
        freopen_s(&nul, "NUL", "w", stderr);
#else
        orig_stderr_fd = dup(fileno(stderr));
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, fileno(stderr));
            close(devnull);
        }
#endif
    }

    ~OutputGuard() {
        std::cout.rdbuf(orig_cout);
        fflush(stderr);
#ifdef _WIN32
        if (orig_stderr_fd >= 0) {
            _dup2(orig_stderr_fd, _fileno(stderr));
            _close(orig_stderr_fd);
        }
#else
        if (orig_stderr_fd >= 0) {
            dup2(orig_stderr_fd, fileno(stderr));
            close(orig_stderr_fd);
        }
#endif
    }
};
