# Test Improvement Plan

## 1. Performance Boundary Tracking

Currently `test_regression` evaluates thresholds as pass/fail but discards actual values after the run. No trend detection exists.

### 1a. Metrics Report with Margin Analysis
- **File**: `test_regression.cpp` (modify `evaluate_thresholds()`)
- **Change**: After each threshold evaluation, compute `margin% = 100 * (threshold - actual) / threshold` (adjusted for op direction). Print a summary table sorted by tightest margin.
- **Output**: `output/metrics_report.csv` with columns: `id, metric, actual, threshold, op, margin_pct, status`
- **Priority**: P0 | **Effort**: 2h

### 1b. Historical Metrics Tracking
- **File**: `test_regression.cpp` (new function `append_history()`)
- **Change**: After evaluation, append one row per metric to `output/metrics_history.csv` with columns: `timestamp, git_hash, id, actual, threshold, margin_pct`. File grows over time.
- **Priority**: P1 | **Effort**: 2h

### 1c. Trend Alerting
- **File**: `test_regression.cpp` (new function `check_trends()`)
- **Change**: After appending history, read last 10 entries per metric. If margin < 10% or margin has decreased 3 consecutive runs, print `TREND WARNING`. Exit code remains pass/fail based on thresholds only.
- **Priority**: P1 | **Effort**: 3h

---

## 2. Parameter Sweep Tests

All sweeps output CSV to `output/sweep_*.csv` for offline analysis. Each sweep is a standalone executable or a section in a new `test_sweeps.cpp`.

### 2a. Rotation Sweep (360 x 1 deg)
- **File**: `tests/test_sweep_rotation.cpp`
- **Metrics**: angle_error, pos_error per angle, per method (coarse/ICP/ROI)
- **Parameters**: angle 0-359 at 1-degree steps, single L-shape template, clean scene
- **Output**: `output/sweep_rotation.csv` (angle, method, ang_err, pos_err)
- **Thresholds**: `sweep_rot_roi_ang_mean,check,<,0.2` and `sweep_rot_roi_worst_ang,check,<,1.0`
- **Priority**: P0 | **Effort**: 4h

### 2b. Sub-pixel Position Sweep
- **File**: `tests/test_sweep_rotation.cpp` (second section)
- **Metrics**: pos_error vs sub-pixel offset, detect systematic bias
- **Parameters**: X offset 0.0-0.9 step 0.1, Y offset 0.0-0.9 step 0.1 (100 combinations)
- **Output**: `output/sweep_subpixel.csv` (dx, dy, method, pos_err_x, pos_err_y)
- **Thresholds**: `sweep_subpx_roi_mean,check,<,0.15` and `sweep_subpx_bias,check,<,0.05`
- **Priority**: P0 | **Effort**: 3h

### 2c. Noise Sweep
- **File**: `tests/test_sweep_noise.cpp`
- **Metrics**: detection_rate, ang_err_mean, pos_err_mean, speed_ms
- **Parameters**: sigma = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}, 10 objects, FHD
- **Output**: `output/sweep_noise.csv` (sigma, method, found, ang_mean, pos_mean, ms)
- **Thresholds**: `sweep_noise_roi_break,warn,>,40` (sigma where ROI ang_mean > 1 deg)
- **Priority**: P0 | **Effort**: 3h

### 2d. Blur Sweep
- **File**: `tests/test_sweep_noise.cpp` (second section)
- **Metrics**: same as 2c
- **Parameters**: kernel = {3, 5, 7, 9, 11, 15, 21, 31}
- **Output**: `output/sweep_blur.csv`
- **Priority**: P1 | **Effort**: 2h (reuses 2c infrastructure)

### 2e. Template Size Sweep
- **File**: `tests/test_sweep_scale.cpp`
- **Metrics**: ang_err, pos_err, feature_count, speed_ms
- **Parameters**: template side = {50, 100, 150, 200, 300, 400} px
- **Output**: `output/sweep_template_size.csv`
- **Priority**: P1 | **Effort**: 3h

### 2f. Object Count Sweep
- **File**: `tests/test_sweep_scale.cpp` (second section)
- **Metrics**: speed_ms, found_count, false_positive_count
- **Parameters**: count = {1, 5, 10, 20, 50}, FHD scene
- **Output**: `output/sweep_object_count.csv`
- **Priority**: P1 | **Effort**: 3h

### 2g. Resolution Sweep
- **File**: `tests/test_sweep_scale.cpp` (third section)
- **Metrics**: speed_ms (coarse/ICP/ROI), detection_rate
- **Parameters**: VGA (640x480), HD (1280x720), FHD (1920x1080), 4K (3840x2160), 20MP (5472x3648)
- **Output**: `output/sweep_resolution.csv`
- **Priority**: P1 | **Effort**: 2h (partially covered by section 10, extend it)

### 2h. Combined Stress Test
- **File**: `tests/test_sweep_stress.cpp`
- **Metrics**: detection_rate, ang_err, pos_err
- **Parameters**: noise x blur x object_count matrix: {n=0,20,40} x {b=1,5,11} x {obj=1,10} = 18 combos
- **Output**: `output/sweep_stress.csv`
- **Thresholds**: `sweep_stress_n40b11_roi_found,check,>=,8` (out of 10)
- **Priority**: P2 | **Effort**: 3h

---

## 3. Diagnostic Tests for Algorithm Development

These produce rich output for debugging, not CI pass/fail. Each writes visualization or detailed CSV.

### 3a. Feature Selection Quality
- **File**: `tests/test_diag_features.cpp`
- **Metrics**: per-feature sensitivity (d_ang, d_pos), spatial coverage, corner ratio
- **Output**: `output/diag_feature_overlay.png` (features on template), `output/diag_sensitivity_heatmap.png`
- **Pathological templates**: circle, parallel lines, thin pole, single line
- **Priority**: P1 | **Effort**: 4h

### 3b. Coarse Matching Score Analysis
- **File**: `tests/test_diag_coarse.cpp`
- **Metrics**: score histogram (true match vs false), score_margin (best - 2nd best), NMS candidate count before/after
- **Output**: `output/diag_score_histogram.csv`, `output/diag_nms_stats.csv`
- **Priority**: P1 | **Effort**: 4h

### 3c. Refinement Convergence Curves
- **File**: `tests/test_diag_convergence.cpp`
- **Metrics**: per-iteration angle_error, pos_error, residual for both ICP and ROI
- **Output**: `output/diag_convergence_icp.csv`, `output/diag_convergence_roi.csv`
- **Highlight cases**: convergence plateau, oscillation, divergence (if any)
- **Priority**: P1 | **Effort**: 3h

### 3d. Inverse ICP Validation
- **File**: `tests/test_diag_convergence.cpp` (section 2)
- **Metrics**: forward vs inverse ICP ang_err and pos_err at 72 angles; per-angle divergence flag
- **Output**: `output/diag_icp_forward_vs_inverse.csv`
- **Priority**: P2 | **Effort**: 2h (partially exists in test_icp_accuracy.cpp, formalize it)

### 3e. ROI matchTemplate Quality
- **File**: `tests/test_diag_roi.cpp`
- **Metrics**: per-point correlation peak value, peak sharpness (2nd derivative), PCA eigenvalue ratio
- **Output**: `output/diag_roi_peaks.csv`, `output/diag_roi_pca_accuracy.csv`
- **Priority**: P2 | **Effort**: 4h

---

## 4. A/B Testing Framework

### 4a. Dual-Run Comparison Script
- **File**: `tests/scripts/ab_compare.py` (Python, calls test_regression twice)
- **Workflow**: Build version A, run, save `metrics_report_A.csv`. Checkout version B, build, run, save `metrics_report_B.csv`. Diff.
- **Output**: `output/ab_diff.csv` (metric, val_A, val_B, delta, delta_pct, improved/regressed)
- **Priority**: P1 | **Effort**: 4h

### 4b. Statistical Significance
- **File**: `tests/scripts/ab_compare.py` (--repeat N flag)
- **Workflow**: Run each version N times (default 5). Per metric, Welch's t-test on distributions. Flag significant regressions (p < 0.05).
- **Output**: `output/ab_stats.csv` (metric, mean_A, std_A, mean_B, std_B, p_value, significant)
- **Priority**: P2 | **Effort**: 3h

---

## 5. Continuous Integration Readiness

### 5a. Unified Entry Point
- **Status**: Already exists: `test_regression all` returns exit code 0/1.
- **Change needed**: Add `--json` flag to emit `output/regression_results.json` with structure: `{pass, fail, warn, metrics: {key: {actual, threshold, status}}}`.
- **Priority**: P0 | **Effort**: 2h

### 5b. Time Budget Enforcement
- **File**: `test_regression.cpp`
- **Change**: Record wall-clock start time. After all sections, if elapsed > 300s (5 min), print `TIMEOUT WARNING` and exit 1.
- **Thresholds**: `ci_total_time,check,<,300.0`
- **Priority**: P0 | **Effort**: 1h

### 5c. Selective Section Running
- **Status**: Already exists: `test_regression 3` runs section 3 only, `test_regression 6 10` runs sections 6-10.
- **Change needed**: Add `--fast` mode that skips sections 9 (noise/blur, slow) and 10 (multi-res, slow) for quick pre-commit checks. Target: < 30s.
- **Priority**: P1 | **Effort**: 1h

---

## Implementation Roadmap

### Phase 1 — CI Foundation (P0, ~9h)
1. Metrics report with margin (1a)
2. JSON output (5a)
3. Time budget (5b)
4. Rotation sweep (2a) and sub-pixel sweep (2b)
5. Noise sweep (2c)

### Phase 2 — Coverage Expansion (P1, ~25h)
6. Historical tracking + trend alerts (1b, 1c)
7. Blur, template size, object count, resolution sweeps (2d-2g)
8. Diagnostic tests: features, coarse scores, convergence (3a-3c)
9. A/B comparison script (4a)
10. Fast mode (5c)

### Phase 3 — Advanced Diagnostics (P2, ~12h)
11. Combined stress test (2h)
12. Forward vs inverse ICP formalized (3d)
13. ROI peak quality diagnostics (3e)
14. Statistical A/B testing (4b)

**Total estimated effort: ~46h across all phases.**
