# Unified Logging Plan

## Current State

### Problem Summary
The project has 3 independent output systems that conflict:

1. **Library prints** (meiqua line2Dup.cpp):
   - `std::cout` — "have no enough features" (unconditional, line 816/819)
   - `printf` — timing breakdown (guarded by `enabled`, lines 47-57)
   - `std::cout` — Timer::out() (user-called, line 24)

2. **Refinement debug** (roi_refine.cpp, shape_matcher.cpp):
   - `fprintf(stderr)` — ROI per-iteration detail (guarded by `config.verbose`)
   - `fprintf(stderr)` — FeatureSet load errors

3. **Test output** (5 test files):
   - `printf` to stdout — results, tables, progress
   - `fprintf` to files — detailed logs, CSV data
   - Various CoutSuppressor variants to mute library

### The Core Issue
`CoutSuppressor` only redirects `std::cout.rdbuf()`. The library also uses `printf`
(C stdio) which bypasses this entirely. No test file suppresses `printf`-to-stdout.

### Impact
- Library timing lines appear interleaved with test output
- `grep` filters fail because library and test output mix on same lines
- Test result parsing is unreliable
- Debugging is confusing — unclear which output is library vs test

## Proposed Solution

### 1. Library Side: Add a log callback (non-breaking)

Add a global log function pointer that all library output goes through.
Default: prints to stdout/stderr as before. User can redirect or mute.

```cpp
// In line2Dup.h or a new logging.h
namespace sbm {
    enum class LogLevel { Debug, Info, Warning, Error };
    using LogCallback = void(*)(LogLevel level, const char* msg);

    void setLogCallback(LogCallback cb);  // nullptr = default (stdout/stderr)
    void log(LogLevel level, const char* fmt, ...);
}
```

Replace ALL library prints:
- line2Dup.cpp:816,819 → `sbm::log(Warning, "have no enough features, exhaustive mode")`
- line2Dup.cpp:47-57 → `sbm::log(Debug, "  %-28s %7.1fms", label, ms)` (only if profiling)
- line2Dup.cpp:24 → `sbm::log(Info, "elapsed time: %.7fs", t)`
- roi_refine.cpp:389-443 → `sbm::log(Debug, "[ROI] %d constraints...", n)` (only if verbose)
- shape_matcher.cpp:153 → `sbm::log(Error, "FeatureSet::load() read error")`

### 2. Test Side: One universal suppressor

Replace all 3 variants (CoutSup, CoutSuppressor, inline rdbuf) with one:

```cpp
// In a shared test_utils.h
struct OutputGuard {
    // Suppresses ALL output: cout, printf, stderr
    std::streambuf* orig_cout;
    std::ostringstream sink;
    int orig_stdout_fd, orig_stderr_fd;

    OutputGuard() : orig_cout(std::cout.rdbuf()) {
        std::cout.rdbuf(sink.rdbuf());
        fflush(stdout); fflush(stderr);
#ifdef _WIN32
        orig_stdout_fd = _dup(_fileno(stdout));
        orig_stderr_fd = _dup(_fileno(stderr));
        FILE* nul1 = nullptr; freopen_s(&nul1, "NUL", "w", stdout);
        FILE* nul2 = nullptr; freopen_s(&nul2, "NUL", "w", stderr);
#else
        orig_stdout_fd = dup(fileno(stdout));
        orig_stderr_fd = dup(fileno(stderr));
        int devnull = open("/dev/null", O_WRONLY);
        dup2(devnull, fileno(stdout));
        dup2(devnull, fileno(stderr));
        close(devnull);
#endif
    }
    ~OutputGuard() {
        std::cout.rdbuf(orig_cout);
        fflush(stdout); fflush(stderr);
#ifdef _WIN32
        _dup2(orig_stdout_fd, _fileno(stdout));
        _dup2(orig_stderr_fd, _fileno(stderr));
        _close(orig_stdout_fd); _close(orig_stderr_fd);
#else
        dup2(orig_stdout_fd, fileno(stdout));
        dup2(orig_stderr_fd, fileno(stderr));
        close(orig_stdout_fd); close(orig_stderr_fd);
#endif
    }
};
```

Usage in ALL test files:
```cpp
{ OutputGuard guard; matcher.match(scene); }  // ALL library output suppressed
printf("Result: %.2f\n", result);             // test output goes through
```

### 3. Test Output: Use fprintf to file, not stdout

All test results should go to a file, not stdout. Stdout only for
human-readable progress/summary. This avoids the interleaving problem entirely.

```cpp
// Pattern for all tests:
FILE* log = fopen("output/test_results.txt", "w");
fprintf(log, "obj[%d] gt=(...) -> (...) err=...\n", ...);  // detailed data
printf("Section 10: 20/20 matched, 0.1deg, 0.07px\n");      // summary only
fclose(log);
```

### 4. Migration Plan

| Step | Files | Change |
|------|-------|--------|
| 1 | Create `logging.h` | sbm::log() with callback, default stdout |
| 2 | line2Dup.cpp | Replace cout/printf with sbm::log() |
| 3 | roi_refine.cpp | Replace fprintf(stderr) with sbm::log() |
| 4 | shape_matcher.cpp | Replace fprintf(stderr) with sbm::log() |
| 5 | Create `test_utils.h` | OutputGuard, shared helpers |
| 6 | test_regression.cpp | Replace CoutSuppressor with OutputGuard |
| 7 | test_simple.cpp | Replace inline rdbuf swaps with OutputGuard |
| 8 | test_20mp_noise.cpp | Replace CoutSup with OutputGuard |
| 9 | test_variant.cpp | Replace CoutSup with OutputGuard |
| 10 | test_selection_compare.cpp | Replace CoutSup with OutputGuard |
| 11 | All tests | Move detailed output to files, keep stdout for summary |

### 5. API for Users

After migration, users can control library output:

```cpp
// Silent (production)
sbm::setLogCallback(nullptr);  // or a no-op callback

// Verbose (debugging)
sbm::setLogCallback([](sbm::LogLevel lv, const char* msg) {
    if (lv >= sbm::LogLevel::Warning)
        fprintf(stderr, "[SBM %s] %s\n", lv == sbm::LogLevel::Error ? "ERR" : "WARN", msg);
});

// Log to file
FILE* f = fopen("sbm.log", "w");
sbm::setLogCallback([f](sbm::LogLevel lv, const char* msg) {
    fprintf(f, "[%d] %s\n", (int)lv, msg);
});
```

### 6. Non-Breaking Compatibility

- Default behavior unchanged (prints to stdout/stderr as before)
- Existing code that ignores output still works
- New code can opt-in to callback-based logging
- CoutSuppressor/CoutSup still work for cout-based prints
- OutputGuard works for everything including printf
