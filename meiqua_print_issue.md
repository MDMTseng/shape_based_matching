# Meiqua shape_based_matching: Debug Print Interference Analysis

## Summary

The library mixes THREE different output mechanisms across four files. The
`CoutSuppressor` in test programs redirects `std::cout.rdbuf()` but this does
**not** suppress C `printf()` calls, which bypass the C++ streambuf entirely and
write directly to the C `FILE* stdout`. This is the root cause.

---

## Every Print Statement Found

### line2Dup.cpp

| Line | Function / Context | Stream | Mechanism |
|------|--------------------|--------|-----------|
| 24   | `Timer::out()` — "elasped time" | **stdout** | `std::cout <<` |
| 47   | `StageProfile::print()` — GaussianBlur timing | **stdout** | `printf()` |
| 48   | `StageProfile::print()` — Sobel timing | **stdout** | `printf()` |
| 49   | `StageProfile::print()` — Quantize timing | **stdout** | `printf()` |
| 50   | `StageProfile::print()` — 3x3 voting timing | **stdout** | `printf()` |
| 51   | `StageProfile::print()` — Fused spread+LUT timing | **stdout** | `printf()` |
| 52   | `StageProfile::print()` — Coarse similarity timing | **stdout** | `printf()` |
| 53   | `StageProfile::print()` — Pyramid refinement timing | **stdout** | `printf()` |
| 54   | `StageProfile::print()` — Sort + NMS timing | **stdout** | `printf()` |
| 57   | `StageProfile::print()` — TOTAL timing | **stdout** | `printf()` |
| 816  | `extractFeatures` — "too few features, abort" | **stdout** | `std::cout <<` |
| 819  | `extractFeatures` — "have no enough features, exaustive mode" | **stdout** | `std::cout <<` |

### line2Dup.h

No print statements found.

### shape_matcher.cpp

| Line | Function / Context | Stream | Mechanism |
|------|--------------------|--------|-----------|
| 153  | `FeatureSet::load()` — read error warning | **stderr** | `fprintf(stderr, ...)` |

### roi_refine.cpp

| Line | Function / Context | Stream | Mechanism |
|------|--------------------|--------|-----------|
| 389  | ICP iteration — constraint count | **stderr** | `fprintf(stderr, ...)` |
| 395  | ICP iteration — per-point details | **stderr** | `fprintf(stderr, ...)` |
| 418  | ICP iteration — outlier rejection count | **stderr** | `fprintf(stderr, ...)` |
| 443  | ICP iteration — per-iteration pose update | **stderr** | `fprintf(stderr, ...)` |

### icp_refine.cpp

No print statements found.

---

## Why CoutSuppressor Fails to Catch Everything

### The core problem: cout vs printf are independent

`CoutSuppressor` works by swapping `std::cout.rdbuf()` to a
`std::ostringstream` sink. This only affects data written via
`std::cout << ...`. It does **not** affect:

1. **`printf()` calls** (line2Dup.cpp lines 47-57) — `printf` writes to the C
   runtime's `FILE* stdout`, which has its own file descriptor. Redirecting the
   C++ streambuf does not touch the C `FILE*` at all. These 9 printf lines in
   `StageProfile::print()` will appear on terminal stdout even when
   CoutSuppressor is active.

2. **`fprintf(stderr, ...)` calls** (roi_refine.cpp, shape_matcher.cpp) — These
   write to stderr. The updated CoutSuppressor in test_regression.cpp does now
   handle stderr via `_dup2`/`freopen_s("NUL")`, so these are suppressed when
   the suppressor is used. However, simpler test programs (test_20mp_noise.cpp,
   test_variant.cpp) only suppress cout, not stderr.

### Why shell `2>/dev/null` doesn't catch printf

Shell `2>/dev/null` redirects file descriptor 2 (stderr). The `printf()` calls
write to fd 1 (stdout), so `2>/dev/null` has no effect on them. You would need
`>/dev/null` or `1>/dev/null` to suppress printf output.

### Guard status of each print site

- **StageProfile::print()** (lines 47-57): Guarded by `if (!enabled) return;`
  at line 46. The `enabled` flag defaults to `false`, so these only fire if
  `enableProfiling(true)` is called explicitly. **Not a problem in normal use.**

- **Timer::out()** (line 24): Only fires when user code calls `timer.out()`.
  Not called internally by the library. **Not a problem in normal use.**

- **"too few features" / "exaustive mode"** (lines 816, 819): These fire
  unconditionally during `extractFeatures()` when the template has few edge
  features. These are the **primary offenders** — they fire during `addModel()`
  and cannot be suppressed by CoutSuppressor because they use `std::cout`
  (which IS caught by rdbuf redirect). Wait — actually these ARE caught.

  **Correction**: Lines 816/819 use `std::cout`, so CoutSuppressor does catch
  them. The real unguarded problem is printf in StageProfile, but that is
  guarded by `enabled`.

- **roi_refine.cpp** (lines 389-443): Guarded by `config.verbose` which
  defaults to `false`. Only fires if user sets verbose. **Not a problem in
  normal use.**

- **shape_matcher.cpp** (line 153): Only fires on file load errors. Uses
  stderr. Rarely triggered.

### Revised Root Cause

After careful analysis, the actual interference scenario is:

1. **When profiling is enabled**: `StageProfile::print()` uses `printf()`
   (stdout), which is NOT caught by CoutSuppressor's rdbuf redirect. This is
   the primary bug — if any test enables profiling, 9 lines of printf output
   leak through the suppressor.

2. **When verbose ROI is enabled**: `fprintf(stderr, ...)` in roi_refine.cpp
   leaks through in test programs that only suppress cout (test_20mp_noise.cpp,
   test_variant.cpp) but is caught by the full CoutSuppressor in
   test_regression.cpp.

3. **Lines 816/819** (`std::cout`): These are caught by CoutSuppressor since
   they use `std::cout`. However, they still interfere with programs that do
   NOT use CoutSuppressor (e.g., simple test programs that just parse stdout).

---

## Definitive Fix

### Option A: Replace all printf with std::cout (minimal change)

In `line2Dup.cpp`, replace the `StageProfile::print()` method to use
`std::cout` instead of `printf`. Then CoutSuppressor catches everything:

```cpp
// line2Dup.cpp line 45-58: replace printf with cout
void print() const {
    if (!enabled) return;
    auto fmt = [](const char* label, double ms) {
        std::cout << "  " << std::left << std::setw(28) << label
                  << std::right << std::setw(7) << std::fixed
                  << std::setprecision(1) << ms << "ms\n";
    };
    fmt("GaussianBlur 7x7", blur_ms);
    // ... etc for each stage
}
```

### Option B: Suppress C stdout too (robust, recommended)

Extend CoutSuppressor to also redirect the C `FILE* stdout` via
`_dup2`/`freopen_s`, the same way it already handles stderr:

```cpp
CoutSuppressor() : orig_cout(std::cout.rdbuf()), ... {
    std::cout.rdbuf(sink.rdbuf());
    // Suppress C stdout (catches printf)
    fflush(stdout);
    orig_stdout_fd = _dup(_fileno(stdout));
    freopen_s(&nul_stdout, "NUL", "w", stdout);
    // Suppress C stderr (catches fprintf(stderr,...))
    fflush(stderr);
    orig_stderr_fd = _dup(_fileno(stderr));
    freopen_s(&nul_stderr, "NUL", "w", stderr);
}
```

### Option C: Guard all prints behind a flag (cleanest)

Wrap lines 816 and 819 in `if (verbose)` or remove them entirely (they are
diagnostic messages from the original meiqua code that serve no purpose in
production). The StageProfile prints are already guarded.

### Recommended approach

**Option B + Option C combined**: Make CoutSuppressor suppress both C stdout
and C stderr (so it is truly comprehensive), AND guard/remove the unconditional
cout prints at lines 816/819. This makes the library silent by default and the
suppressor robust against any future print additions.
