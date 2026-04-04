#pragma once
/// @file sbm_log.h
/// @brief Unified logging for shape_based_matching library.
/// Header-only. All library output goes through sbm::sbm_log().
///
/// Usage:
///   sbm::setLogLevel(sbm::LogLevel::Debug);         // show everything
///   sbm::setLogFile("sbm.log");                     // also log to file
///   sbm::setLogCallback(myCallback);                // custom handler
///   sbm::sbm_log(sbm::LogLevel::Info, "match", "found %d objects", n);

#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sbm {

enum class LogLevel { Debug = 0, Info = 1, Warning = 2, Error = 3, Off = 4 };

/// Callback signature: (level, module, message)
using LogCallback = void(*)(LogLevel level, const char* module, const char* msg);

namespace detail {
    inline LogCallback& callback() {
        static LogCallback cb = nullptr;
        return cb;
    }
    inline LogLevel& min_level() {
        static LogLevel lv = LogLevel::Warning;  // default: only warnings and errors
        return lv;
    }
    inline FILE*& log_file() {
        static FILE* f = nullptr;
        return f;
    }
    inline bool& show_timestamp() {
        static bool ts = false;
        return ts;
    }
    inline bool& show_thread() {
        static bool th = false;
        return th;
    }
    inline std::chrono::steady_clock::time_point& start_time() {
        static auto t0 = std::chrono::steady_clock::now();
        return t0;
    }

    inline const char* level_str(LogLevel lv) {
        switch (lv) {
            case LogLevel::Debug:   return "DBG";
            case LogLevel::Info:    return "INF";
            case LogLevel::Warning: return "WRN";
            case LogLevel::Error:   return "ERR";
            default:                return "???";
        }
    }

    inline void default_handler(LogLevel level, const char* module, const char* msg) {
        char prefix[128] = "";
        int pos = 0;

        if (show_timestamp()) {
            auto now = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(now - start_time()).count();
            pos += snprintf(prefix + pos, sizeof(prefix) - pos, "[%8.1fms]", ms);
        }
        if (show_thread()) {
#ifdef _OPENMP
            pos += snprintf(prefix + pos, sizeof(prefix) - pos, "[T%d]", omp_get_thread_num());
#endif
        }
        pos += snprintf(prefix + pos, sizeof(prefix) - pos, "[%s]", level_str(level));
        if (module && module[0])
            pos += snprintf(prefix + pos, sizeof(prefix) - pos, "[%s]", module);

        // Console output (stderr for warnings/errors, suppress debug/info by default)
        if (level >= LogLevel::Warning)
            fprintf(stderr, "%s %s\n", prefix, msg);

        // File output (everything at or above min_level)
        if (log_file())
            fprintf(log_file(), "%s %s\n", prefix, msg);
    }
}

/// Set minimum log level. Messages below this level are suppressed.
/// Default: Warning (only warnings and errors shown)
inline void setLogLevel(LogLevel level) {
    detail::min_level() = level;
}

/// Set log file. All messages at or above min_level are written here.
/// Pass nullptr to disable file logging. The file is NOT closed automatically.
inline void setLogFile(FILE* f) {
    detail::log_file() = f;
}

/// Convenience: open a log file by path.
/// Returns the FILE* (caller should close it when done).
inline FILE* openLogFile(const char* path) {
    FILE* f = fopen(path, "w");
    detail::log_file() = f;
    return f;
}

/// Enable/disable timestamp prefix (milliseconds since first log call)
inline void setLogTimestamp(bool enable) {
    detail::show_timestamp() = enable;
}

/// Enable/disable OpenMP thread ID prefix
inline void setLogThread(bool enable) {
    detail::show_thread() = enable;
}

/// Set a custom log callback. Overrides default handler entirely.
/// Pass nullptr to restore default behavior.
inline void setLogCallback(LogCallback cb) {
    detail::callback() = cb;
}

/// Main log function. Module is a short tag like "coarse", "icp", "roi", "feat".
inline void sbm_log(LogLevel level, const char* module, const char* fmt, ...) {
    if (level < detail::min_level()) return;

    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    auto cb = detail::callback();
    if (cb)
        cb(level, module, buf);
    else
        detail::default_handler(level, module, buf);
}

/// Convenience overload without module tag
inline void sbm_log(LogLevel level, const char* fmt, ...) {
    if (level < detail::min_level()) return;

    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    auto cb = detail::callback();
    if (cb)
        cb(level, "", buf);
    else
        detail::default_handler(level, "", buf);
}

} // namespace sbm
