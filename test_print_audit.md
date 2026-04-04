# Test Print Audit

Audit of all output mechanisms and library-print suppression in 5 test files.

---

## Library print sources (what tests are trying to suppress)

| File | Line | Stream | Content |
|------|------|--------|---------|
| line2Dup.cpp | 24 | cout | Timer: `"message\nelasped time:" << t << "s\n"` |
| line2Dup.cpp | 47-57 | stdout (printf) | Per-stage timing breakdown (7 lines) |
| line2Dup.cpp | 816 | cout | `"too few features, abort"` |
| line2Dup.cpp | 819 | cout | `"have no enough features, exaustive mode"` |
| shape_matcher.cpp | 153 | stderr (fprintf) | `"Warning: FeatureSet::load(...) read error"` |
| roi_refine.cpp | 389 | stderr (fprintf) | `"[ROI] %d constraints..."` (when `config.verbose`) |
| roi_refine.cpp | 395 | stderr (fprintf) | Per-point matching detail (when `config.verbose`) |
| roi_refine.cpp | 418 | stderr (fprintf) | `"outlier rejection: ..."` (when `config.verbose`) |
| roi_refine.cpp | 443 | stderr (fprintf) | `"iter %d: angle=..."` (when `config.verbose`) |

Key observation: The library uses **two separate output channels**:
- `std::cout` and `printf` (both go to stdout) for general timing info
- `fprintf(stderr, ...)` for ROI debug output (gated by `config.verbose`)

---

## CoutSup (lightweight, cout-only suppressor)

Found in: test_20mp_noise.cpp:35-39, test_variant.cpp:31-36, test_selection_compare.cpp:27-32

```cpp
struct CoutSup {
    std::streambuf* ob;
    std::ostringstream sink;
    CoutSup() : ob(std::cout.rdbuf()) { std::cout.rdbuf(sink.rdbuf()); }
    ~CoutSup() { std::cout.rdbuf(ob); }
};
```

**What it redirects:** Only `std::cout` (C++ stream).
**Does it handle printf?** NO. printf writes directly to the C stdout FILE*, bypassing cout's streambuf entirely. The line2Dup.cpp timing printfs at lines 47-57 will **not** be suppressed.
**Does it handle stderr?** NO.
**RAII safe?** Yes, destructor restores original rdbuf.

### Critical flaw
CoutSup only intercepts `std::cout << ...` calls. It does **not** intercept:
- `printf(...)` calls (C stdio to stdout fd 1)
- `fprintf(stderr, ...)` calls (C stdio to stderr fd 2)

The library's timing breakdown (line2Dup.cpp:47-57) uses `printf`, so CoutSup does NOT suppress it. The cout-based prints at lines 24, 816, 819 ARE suppressed.

---

## CoutSuppressor (full suppressor, cout + stderr)

Found in: test_regression.cpp:306-345

```cpp
struct CoutSuppressor {
    std::streambuf* orig_cout;
    std::ostringstream sink;
    FILE* orig_stderr_copy;
    int orig_stderr_fd;
    int devnull_fd;
    ...
};
```

**What it redirects:**
1. `std::cout` via rdbuf swap (same as CoutSup)
2. `stderr` via `_dup`/`_dup2` + `freopen_s("NUL")` on Windows, or `dup`/`dup2` + `/dev/null` on POSIX

**Does it handle cout?** Yes.
**Does it handle printf to stdout?** NO. It only swaps cout's rdbuf. printf to stdout is untouched.
**Does it handle stderr?** Yes, via fd-level dup2 redirect to NUL/dev/null.
**Does it handle fprintf(stderr,...)?** Yes, because it redirects the stderr fd itself.
**RAII safe?** Yes, destructor restores both cout rdbuf and stderr fd.

### Gap in CoutSuppressor
It suppresses cout (C++ stream) and stderr (fd-level), but it does **not** suppress printf-to-stdout. The library's `printf` timing output in line2Dup.cpp:47-57 will leak through.

---

## Per-file analysis

### 1. test_simple.cpp

**Own output:**
| Line | Function | Stream | Purpose |
|------|----------|--------|---------|
| 15 | fprintf(stderr, ...) | stderr | CHECK macro failure messages |
| 39 | printf | stdout | "Template: L-shape" |
| 47 | printf | stdout | "Template: V-shape" |
| 52 | printf | stdout | "Template: parallel lines" |
| 57 | printf | stdout | "Template: long pole" |
| 61 | printf | stdout | "Template: single horizontal line" |
| 68 | printf | stdout | "Template: flat triangle" |
| 75 | printf | stdout | "Template: triangle" |
| 82 | printf | stdout | "Template: right triangle" |
| 89 | printf | stdout | Feature count |
| 122-126 | printf | stdout | Match info, GT info |
| 129-130 | printf | stdout | Robustness header |
| 167 | printf | stdout | PCA analysis header |
| 195 | printf | stdout | Per-point PCA results |
| 216-237 | printf | stdout | Various results |
| 239-241 | printf | stdout | Table headers |
| 282-366 | printf | stdout | Per-entry results |
| 374-399 | printf | stdout | Match mode results |
| 414-452 | printf | stdout | Ablation results |
| 458-567 | printf | stdout | FHD speed benchmark |
| 568-569 | printf | stdout | Separator line |
| 587-607 | printf | stdout | ICP detail |
| 694-699 | printf | stdout | Row-at-once results |
| 715-838 | printf, fprintf(bf,...) | stdout + file | FHD 20-object section |
| 845-1578 | printf, fprintf(cf/rf/ef/sumf,...) | stdout + files | Sub-pixel sweep, charts |
| 1594-1603 | printf | stdout | Skew test header |
| 1694-1709 | fprintf(sf,...) | file only | Skew results to file |
| 1783 | printf | stdout | "saved" message |
| 1788-1898 | printf | stdout | Isolated per-angle section |
| 1903 | printf | stdout | Final pass/fail summary |

**Suppression mechanism:** Inline `std::cout.rdbuf()` swap — no CoutSup struct.
- Line 582: `std::ostringstream ns; auto ob = std::cout.rdbuf(ns.rdbuf());`
- Line 585: `std::cout.rdbuf(ob);`
- Line 611-659: `std::streambuf* orig_cout = std::cout.rdbuf();` ... manual swap around each match call
- Throughout (lines 645, 652, 771, 780, 911, 959, 1096, 1106, 1345, 1369, 1378, 1662, 1830, 1841, 1886): pairs of `std::cout.rdbuf(null_stream.rdbuf())` / `std::cout.rdbuf(orig_cout)`

**Handles printf?** NO. Only cout rdbuf swaps.
**Handles stderr?** NO. No stderr redirection at all.

**File output:**
- Lines 759+: `fprintf(bf, ...)` to `output/fhd20_n30.txt`
- Lines 854+: `fprintf(cf, ...)` to `output/coarse_vs_noise.txt`
- Lines 987+: `fprintf(rf, ...)` to `output/roi_size_sweep.txt`
- Lines 1151+: `fprintf(sumf, ...)` to `output/subpixel_summary.txt`
- Lines 1571+: `fprintf(ef, ...)` to `output/angle_errors.txt`
- Lines 1700+: `fprintf(sf, ...)` to `output/skew_results.txt`

### 2. test_regression.cpp

**Own output:**
| Line | Function | Stream | Purpose |
|------|----------|--------|---------|
| 38 | LOG macro → fprintf(g_log,...) | file (g_log) | Log to output/regression_log.txt |
| 42-43 | CHECK macro → printf | stdout | PASS/FAIL lines |
| 47-49 | WARN_IF macro → printf | stdout | WARN/OK lines |
| 85-172 | printf | stdout | CSV load errors/info |
| 186-214 | printf | stdout | Threshold evaluation |
| 351-2297 | printf | stdout | Section headers, results, summary |
| 2228 | g_log = stderr | stderr fallback | If log file can't open |
| 2299 | fclose(g_log) | file | Close log |

**Suppression mechanism:** `CoutSuppressor` struct (lines 306-345).
Used at lines: 380, 437, 478, 531, 601, 631, 686, 750, 785, 816, 844, 1022, 1086, 1111, 1144, 1199, 1203, 1413, 1417, 1424, 1550, 1617, 1630, 1673, 1694, 1715, 1744, 1766, 1792, 1811, 1840, 1858, 1873, 1894, 1908, 1925, 1955, 1976, 2013, 2070, 2093

**Handles cout?** Yes (rdbuf swap).
**Handles printf?** NO.
**Handles stderr?** Yes (fd-level dup2 to NUL).
**File output:** LOG macro writes to `output/regression_log.txt` via g_log FILE*.

### 3. test_20mp_noise.cpp

**Own output:**
| Line | Function | Stream | Purpose |
|------|----------|--------|---------|
| 100 | printf | stdout | Header: "20MP ... 20 objects" |
| 101-106 | printf | stdout | Column headers and separator |
| 130 | printf | stdout | Per-noise-level row prefix |
| 182-184 | fprintf(stderr,...) | stderr | N40 per-object detail (uses `ae`, `pe` before they're declared — likely a bug) |
| 193-194 | fprintf(stderr,...) | stderr | N40 MISSING object detail |
| 201-202 | printf | stdout | Per-mode results in row |
| 206 | printf | stdout | Newline end of row |
| 211 | printf | stdout | "saved" message |

**File output:** Lines 108-210 use `fprintf(rf, ...)` to `output/20mp_noise_sweep.txt`. Mirrors stdout table content.

**Suppression mechanism:** `CoutSup` struct (lines 35-39).
Used at:
- Line 142: `{ CoutSup s; matcher.addModel(...); }`
- Line 145: `{ CoutSup s; matcher.match(scene); }` (warmup)
- Line 153: `{ CoutSup s; last_results = matcher.match(scene); }` (timed runs)

**Handles cout?** Yes.
**Handles printf?** NO.
**Handles stderr?** NO.

**Why fprintf(stderr,...) at lines 182-194 would not appear with `2>&1 | grep`:**
These lines are only emitted when `ni == 6` (noise=40 specifically). If running a different noise level, they simply don't execute. However, if they do execute and `2>&1` is used, they WILL appear in the combined stream. The only scenario where they'd be invisible is:
1. The program is run without `2>&1` (stderr goes to terminal, pipe only captures stdout)
2. The `ae` and `pe` variables at line 182 are used before declaration (they are declared at line 185-187), so on some compilers this block may not compile or may behave unexpectedly

### 4. test_variant.cpp

**Own output:**
| Line | Function | Stream | Purpose |
|------|----------|--------|---------|
| 62-63 | printf | stdout | Feature counts |
| 104 | printf | stdout | "Single variant" header |
| 119 | printf("") | stdout | Force flush (no-op) |
| 125-127 | printf | stdout | Single-variant results |
| 130 | printf | stdout | "Multi-variant" header |
| 154-156 | printf | stdout | Multi-variant results |
| 169 | snprintf → putText | image only | Score label (not console) |
| 212 | printf | stdout | "saved" message |
| 214-218 | printf | stdout | Summary |

**Suppression mechanism:** `CoutSup` struct (lines 31-36).
Used at:
- Lines 117-124: `{ CoutSup s; matcher.addModel(...); ... matcher.match(scene); }`
- Lines 145-152: `{ CoutSup s; matcher.addModel(...); ... matcher.match(scene); }`

**Handles cout?** Yes.
**Handles printf?** NO.
**Handles stderr?** NO.
**File output:** None (only image via imwrite).

### 5. test_selection_compare.cpp

**Own output:**
| Line | Function | Stream | Purpose |
|------|----------|--------|---------|
| 668 | fprintf(stderr,...) | stderr | "Cannot open output file" error |
| 674 | printf | stdout | Table header |
| 678 | printf | stdout | Shape section header |
| 696 | printf | stdout | Old vs New point counts |
| 701 | printf | stdout | Condition label |
| 721 | printf | stdout | Result line |
| 739 | printf | stdout | "Results saved" message |

**File output:** Lines 668-736 use `fprintf(fout, ...)` to `output/selection_compare.txt`. Mirrors stdout.

**Suppression mechanism:** `CoutSup` struct (lines 27-32).
Used at:
- Line 587: `{ CoutSup s; matcher.addModel(...); }`
- Line 592: `{ CoutSup s; results = matcher.match(...); }`
- Line 621: `{ CoutSup s; matcher.addModel(...); }`
- Line 626: `{ CoutSup s; results = matcher.match(...); }`

**Handles cout?** Yes.
**Handles printf?** NO.
**Handles stderr?** NO.

---

## Root cause: why test output gets "eaten" or interleaved

### Problem 1: printf leak-through (all files except test_regression.cpp)
The library's timing breakdown in line2Dup.cpp (lines 47-57) uses `printf`, which writes directly to the C stdout file descriptor. All `CoutSup` / inline `rdbuf()` suppression only intercepts C++ `std::cout`. The `printf` calls bypass this entirely and write directly to fd 1.

**Result:** Library timing lines (`"GaussianBlur 7x7 ...ms"`, `"Sobel dx+dy ...ms"`, etc.) appear interleaved with test output even when CoutSup is active.

### Problem 2: stdout buffering causes interleaving
`printf` (C stdio) and `std::cout` (C++ streams) maintain separate buffers. Even when both target fd 1, their flush timing differs. This means library printf output can appear in the middle of a test's printf output, especially when the test prints a partial line (e.g., `printf("%-14s", cond.name)`) then calls the library, then prints more.

test_simple.cpp explicitly acknowledges this at line 693:
```
// Print entire row at once (avoids meiqua cout interleaving)
```

### Problem 3: stderr not suppressed by CoutSup
Only test_regression.cpp's `CoutSuppressor` redirects stderr (via dup2). The other four files use `CoutSup` which ignores stderr completely. If ROI verbose mode is enabled, `fprintf(stderr,...)` calls from roi_refine.cpp (lines 389-443) will appear unsuppressed in test_20mp_noise.cpp, test_variant.cpp, and test_selection_compare.cpp.

### Problem 4: test_simple.cpp manual rdbuf swap is fragile
test_simple.cpp manually swaps `std::cout.rdbuf()` around every library call (dozens of times). If an exception occurs between swap-out and swap-back, cout remains redirected — no RAII safety. This is also verbose and error-prone.

### Problem 5: CoutSup in test_20mp_noise.cpp doesn't cover its own fprintf(stderr) output
Lines 182-194 write per-object diagnostics to stderr. Since CoutSup doesn't touch stderr, these always go to the terminal. When piping with `| grep` (without `2>&1`), stderr output goes directly to the terminal (not through the pipe), so grep never sees it. With `2>&1 | grep`, it would appear but interleaved with stdout.

---

## Summary table

| File | Suppressor | Blocks cout | Blocks printf | Blocks stderr | RAII |
|------|-----------|-------------|---------------|---------------|------|
| test_simple.cpp | inline rdbuf swap | Yes | **NO** | **NO** | **NO** |
| test_regression.cpp | CoutSuppressor | Yes | **NO** | Yes | Yes |
| test_20mp_noise.cpp | CoutSup | Yes | **NO** | **NO** | Yes |
| test_variant.cpp | CoutSup | Yes | **NO** | **NO** | Yes |
| test_selection_compare.cpp | CoutSup | Yes | **NO** | **NO** | Yes |

**No test file suppresses printf-to-stdout from the library.** This is the primary cause of output interleaving.

To fully suppress library output, a suppressor would need to:
1. Swap `std::cout.rdbuf()` (for C++ stream output)
2. Redirect fd 1 via `dup2` to `/dev/null` or `NUL` (for C printf to stdout)
3. Redirect fd 2 via `dup2` (for C fprintf to stderr)

Only test_regression.cpp does step 1 + step 3. No file does step 2.
