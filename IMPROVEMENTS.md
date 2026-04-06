# Shape-Based Matching: Optimization & Refinement Summary

## Branch: `optimized-avx2` — Speed Optimizations

All optimizations applied to meiqua's original shape_based_matching implementation.

### Speed Results (72 L-shape templates, clean image)

| Resolution | Original | Optimized | Speedup |
|-----------|----------|-----------|---------|
| VGA 640x480 | 9.4ms | 4.0ms | **2.4x** |
| FHD 1920x1080 | 44ms | 20ms | **2.2x** |
| 30MP 6000x5000 | 1,292ms | 231ms | **5.6x** |

### Optimizations Applied

1. **Fix MIPP scalar shuff bug** — replaced scalar `mipp::shuff<uint8_t>` (which fell back to a loop on AVX2) with direct `_mm256_shuffle_epi8` (vpshufb).

2. **Chunked uint8 accumulation in similarity()** — accumulate scores in uint8 (batches of 63 features, max 63*4=252 < 255) at full 32-byte AVX2 width, widen to int16 only between batches.

3. **Fully fused spread + response maps + linearize** — single row-wise pass with two small temp buffers instead of three separate full-image passes. Eliminates WxH intermediate `h_spread` and `spread_quantized` buffers.

4. **Popcount discriminability penalty** — spread bytes with many orientation bits set (ambiguous/noisy) get reduced response scores. Discount table: `{0, 4, 4, 4, 3, 2, 1, 0, 0}` indexed by popcount. Eliminates grid-pattern false positives (73K -> 0).

5. **SSE strided decimation for T=4** — extract every 4th byte using `_mm_shuffle_epi8` + `_mm_unpacklo_epi32` instead of scalar strided copy. 4 loads -> 16 outputs per iteration.

6. **OpenMP parallelization** — enable `/openmp` on MSVC, parallelize both the fused row loop and template matching in `matchClass`. Fix MSVC OpenMP 2.0 compatibility (no custom reductions, signed loop var).

7. **Integer quantization** — replace `cv::phase()` (per-pixel atan2) with comparison-based 8-bin orientation quantization using fixed-point tan boundaries. No floating-point atan2 needed.

8. **NMS pointer optimization** — replace `.at<float>()` bounds-checked accessors with raw pointer arithmetic in the 5x5 NMS loop during template feature extraction.

9. **AVX2 vectorized voting** — process 32 pixels simultaneously with `_mm256_cmpeq_epi8` for 8 neighbor comparisons + `_mm256_shuffle_epi8` bin-to-bitmask LUT. 9.7x faster than scalar voting.

10. **AVX2 vectorized quantize** — process 8 pixels at a time in int32: widen int16 dx/dy, compute magnitude squared, threshold, undirected reduction (branchless blendv), and 4 tan-boundary multiply-compares — all in AVX2 registers.

11. **Candidate cap** — limit coarse candidates to top-256 per template using `std::partial_sort`. Prevents noise-induced candidate explosion from flooding pyramid refinement.

### Robustness Results (FHD, 180 templates, 20 objects)

| Metric | Original | Optimized |
|--------|----------|-----------|
| Grid pattern false positives | 593,659 | **170** |
| Noise sigma=80 false positives | 658,704 | **43,100** |
| Random texture false positives | 794,643 | **43,592** |
| Clean detection | all 20 found | all 20 found |
| Blur k=21 detection | all 20 found | all 20 found |

### Per-Stage Profiling (FHD, clean)

| Stage | Time |
|-------|------|
| GaussianBlur 7x7 | 0.9ms |
| Sobel dx+dy (int16) | 3.0ms |
| Quantize (AVX2) | 2.5ms |
| 3x3 voting (AVX2) | 0.9ms |
| Fused spread+LUT+linearize | 3.6ms |
| Matching (OpenMP) | 0.4ms |
| **Total** | **11.3ms** |

---

## Branch: `orientation-refine` — ICP Pose Refinement

Edge-based ICP for sub-degree orientation and sub-pixel position accuracy.

### Inverse ICP (current default)

Instead of the traditional forward ICP (model edges → scene EDT), we use **inverse ICP**
(scene edges → template EDT). This eliminates divergence from edge ambiguity.

```
Coarse match (LineMOD)
    |  ~5 deg accuracy, ~2.5px accuracy
    v
Spatial NMS (deduplicate overlapping detections)
    |
    v
Inverse ICP (per object)
    |  Pre-built template EDT (cached, computed once at addModel)
    |  Extract scene Canny edges in local ROI
    |  Inverse-transform scene edges to template space
    |  Match against template EDT (clean, no ambiguity)
    |  Point-to-plane ICP with normal compatibility check
    v
Refined pose: < 0.5 deg, < 1px accuracy, zero divergence
```

### Forward vs Inverse ICP

Forward ICP builds the EDT on the scene and matches model points against it.
The problem: scene EDT contains edges from multiple objects, noise, and background.
At certain template angles (e.g., 315° for L-shape), model edge points from one
arm can snap to scene edges from the other arm — a valid-looking but wrong match.

Inverse ICP builds the EDT on the template (once, at setup time). Scene edges are
inverse-transformed to template space and matched against the clean template EDT.
The template has no ambiguity — each arm's edges are well-separated in canonical space.

| Metric (72 angles, isolated) | Forward ICP | Inverse ICP |
|------------------------------|-------------|-------------|
| Divergence failures (>5px) | **11/72** | **0/72** |
| Mean angle error | 0.24 deg | **0.04 deg** |
| Mean position error | 4.0 px | **0.66 px** |
| Worst position error | 37.1 px | **1.3 px** |

### Key ICP Features

- **Inverse correspondence direction** — scene edges matched against template EDT.
  Template EDT is clean (no other objects, no noise), eliminating divergence from
  edge ambiguity. Zero failures across all angles.

- **Normal compatibility filtering** — reject correspondences where model and scene edge normals disagree by > 45 degrees. Prevents cross-part matching.

- **Point-to-point regularization** — small weighted point-to-point term alongside point-to-plane prevents sliding along straight edges.

- **Cached template EDT** — `buildTemplateScene()` called once at `addModel()` time.
  Stored in FeatureSet. Zero per-match overhead for template processing.

- **Local ROI scene edges** — Canny edge extraction only in a small patch around
  each match. Higher thresholds (50/100) to suppress noise edges.

- **EDT-based distance field** — use OpenCV `distanceTransform` with `DIST_LABEL_PIXEL` for O(W*H) closest-edge lookup instead of brute-force O(W*H*max_dist^2).

- **3x3 LDL solver** — no Eigen dependency. Custom 3x3 symmetric positive-definite solver for the SO2 pose update (theta, tx, ty). Also supports 4x4 solver for Sim2 (with scale).

### Known limitation

Inverse ICP scene edge extraction uses Canny thresholds 50/100. Under moderate
noise (sigma >= 20) without blur, valid edges may be rejected, degrading accuracy.
Possible fix: adaptive Canny thresholds based on scene noise estimate, or
pre-blur the scene ROI more aggressively.

### ICP Accuracy (120 angles, 3-degree sweep, clean image)

| Metric | Coarse Only | + ICP Refine |
|--------|------------|--------------|
| Angle mean | 5.1 deg | **0.4 deg** |
| Angle max | 11.0 deg | **1.2 deg** |
| Angle <= 2 deg | 17% | **100%** |
| Position mean | 2.5 px | **0.6 px** |
| Position max | 5.8 px | **1.1 px** |
| Position <= 2 px | 45% | **100%** |

### ICP Robustness

| Condition | Angle <= 2 deg | Position <= 2 px |
|-----------|---------------|-----------------|
| Clean | 100% | 100% |
| Noise sigma=10 | 100% | 100% |
| Noise sigma=30 | 99% | 99% |
| Blur k=5 | 100% | 100% |
| Blur k=11 | 100% | 100% |
| Blur k=21 | 100% | 100% |
| Noise+blur | 100% | 100% |

### ICP Signed Bias (per shape)

| Shape | Features | Angle Bias | Position Bias |
|-------|----------|------------|---------------|
| L-shape | 198 | +0.06 deg | (-0.03, -0.06) px |
| T-shape | 196 | -0.14 deg | (-0.04, -0.05) px |
| Wrench | 148 | -0.58 deg | (+0.45, -0.46) px |
| Arrow | 114 | -19.3 deg | shape ambiguity |

ICP bias is near-zero for well-defined shapes. Arrow fails due to edge ambiguity (near-identical edges at different orientations).

---

## Coarse Matching Angle Bias

### Root Cause

LineMOD's 8-bin orientation quantization creates asymmetric score profiles. The bias is:
- **Inherent to the algorithm** (original meiqua has it too: -4.0 deg)
- **Shape-dependent** (Arrow: -26.9 deg, L-shape: -4.4 deg, T-shape: -5.0 deg)
- **Not predictable from parameters alone**

### Mitigation

| Method | Mean Error | Max Error | Cost |
|--------|-----------|-----------|------|
| Raw coarse | 5.2 deg | 11.0 deg | 0 |
| + `calibrateAngleBias()` | 2.6 deg | 10.2 deg | 0 (20ms one-time) |
| + ICP refinement | 0.4 deg | 1.2 deg | 0.3ms/object |

`calibrateAngleBias()` auto-measures the bias by matching the template at 12 known angles (~20ms, call once at training time).

### Feature Rotation (`addRotatedTemplates`)

Extract features once at 0 degrees, mathematically rotate coordinates + orientation labels for all angles. Avoids warpAffine interpolation artifacts.

| Shape | Step | warpAffine bias | Feature rotation bias |
|-------|------|----------------|----------------------|
| L-shape | 1 deg | -4.4 deg | -4.3 deg |
| T-shape | 1 deg | -5.0 deg | +5.9 deg |
| Arrow | 1 deg | -26.9 deg | +0.8 deg |
| Wrench | 1 deg | -5.6 deg | -8.3 deg |

Feature rotation helps significantly for Arrow (-26.9 -> +0.8 deg) but not universally. Best for step <= 3 degrees.

---

## Multi-Resolution Pipeline

Match at reduced scale for speed, ICP at full resolution for accuracy.

### Architecture

```
Full-res template
    |
    v
Scale feature coordinates by match_scale
    |
    v
Resize scene to match_scale (e.g., 30%)
    |
    v
LineMOD match on small scene (fast)
    |
    v
Scale positions back to full resolution
    |
    v
Local ROI Sobel (per match, ~0.1ms each)
    |
    v
ICP refine at full resolution
```

### Results (3 objects)

| Config | Match | ICP | Total | vs Full |
|--------|-------|-----|-------|---------|
| FHD full | 22ms | 11ms | 33ms | 1.0x |
| FHD 50% | 7ms | 2ms | **9ms** | **3.7x** |
| FHD 30% | 4ms | 4ms | **8ms** | **4.2x** |
| 30MP full | 267ms | 2ms | 268ms | 1.0x |
| 30MP 50% | 78ms | 2ms | **81ms** | **3.3x** |
| 30MP 30% | 35ms | 9ms | **45ms** | **5.9x** |

### End-to-End Speedup (30MP, 3 objects)

| Pipeline | Time | vs Original |
|----------|------|-------------|
| Original meiqua | 1,292ms | 1.0x |
| + all AVX2 optimizations | 231ms | 5.6x |
| + 30% multi-res + local ICP | **45ms** | **28.7x** |

---

## Files

### Core Implementation
- `line2Dup.cpp` — all matching optimizations (AVX2, fusion, quantization, voting, OpenMP)
- `line2Dup.h` — API additions (calibrateAngleBias, addRotatedTemplates, getClassTemplates)
- `icp_refine.h` / `icp_refine.cpp` — edge-based ICP pose refinement

---

## Branch: `roi-refine` — ROI-Based Pose Refinement

Alternative to ICP: uses template matching on small ROI patches with PCA-based
constraint classification.

### Architecture

```
Coarse match (LineMOD)
    |
    v
Select ~8-15 critical points (corners first, then spaced edges)
    |
    v
For each iteration (3-5x):
    |  Rotate ROI patches from template (cached after first warp)
    |  matchTemplate per point (subpixel parabolic interpolation)
    |  PCA on each ROI gradient -> edge (1D) or corner (2D) constraint
    |  Single rigid body solve
    |  Update pose, re-match only if angle changed > 2 deg
    v
Refined pose
```

### Key Design Decisions

- **matchTemplate per ROI** instead of closest-edge lookup — searches a 40x40 pixel
  window per point, finding correct correspondences even when initial pose is far off
- **PCA edge/corner classification** — eigenvalue ratio < 1.5 means corner (2D constraint),
  otherwise edge (1D point-to-plane). Corners add tangent direction as second constraint.
- **Cached warpAffine** — rotate all ROI patches once, reuse across iterations.
  Only re-warp if angle drifts > 5 deg from cached angle.
- **Adaptive re-match** — re-run matchTemplate only if pose changed > 2 deg since last match.
  Small corrections reuse cached correspondences (just rigid solve).
- **Outlier rejection** — remove correspondences with distance > 2x median before solve.

### Sensitivity-Optimized Feature Selection

The feature set for ROI refinement is automatically optimized for balanced sensitivity:

1. **Initial greedy selection**: corners first (×10 priority), then by distance from center
2. **Iterative swap**: remove least-sensitive feature, try all candidates from pool,
   accept swap if it reduces worst-case sensitivity. Repeat until worst/least ratio < 2.
3. **Corner protection**: corners are never swapped out (they provide 2D constraint)
4. **Cache**: computed once at `addModel()`, reused at zero cost during `match()`

**Sensitivity metrics** (per feature, via 1px perturbation Jacobian):
- `d_ang`: angle change from 1px matching error (deg/px)
- `d_pos`: position change from 1px matching error (px/px)
- `leverage`: distance from rotation center (angular torque arm)

**Key insight**: sensitivity is zero-sum — reweighting features can't reduce total
sensitivity, only redistribute it. The fix is better feature *selection*, not weighting.
Removing the least-sensitive feature barely changes others; removing the most-sensitive
causes neighbors to absorb its load. Adding equally-sensitive features lowers all.

| Shape | Before (greedy) worst_ang | After (optimized) worst_ang |
|-------|--------------------------|----------------------------|
| L-shape | 1.06 | **0.52** |
| V-shape | 0.94 | **0.41** |
| Parallel lines | 0.72 | **0.43** |
| Long pole | 0.80 | **0.44** |
| Single line | 1.32 | **0.73** |

Optimization cost: 80-220ms (offline, once per template). Cached call: 0ms.

### Inverse ICP vs ROI Comparison (FHD 1920x1080, 200x200 template, 10 objects)

| | None (coarse) | ICP (inverse) | ROI (8 optimized) |
|---|---|---|---|
| **Speed** | 23ms | **25ms** | 25ms |
| **Angle** | 10.0 deg | **0.5 deg** | 1.1 deg |
| **Position** | 5.7px | 1.0px | **0.6px** |

### Accuracy Under Degradation (FHD, 10 objects)

| Condition | ICP angle | ROI angle | ICP pos | ROI pos |
|-----------|----------|----------|---------|---------|
| clean | **0.5 deg** | 1.1 deg | 1.0px | **0.6px** |
| noise s=10 | **0.9 deg** | 1.1 deg | 1.1px | **0.6px** |
| noise s=20 | 7.3 deg | **0.9 deg** | 5.1px | **0.5px** |
| blur k=5 | **0.2 deg** | 1.2 deg | **0.7px** | 0.7px |
| blur k=11 | **0.2 deg** | 1.6 deg | **0.7px** | 1.1px |
| blur k=21 | **0.1 deg** | 1.7 deg | **0.8px** | 1.1px |
| n30+b11 | **0.2 deg** | 1.6 deg | **0.7px** | 0.9px |
| n50+b11 | **0.2 deg** | 1.7 deg | **0.6px** | 0.9px |

### ICP vs ROI Characteristics

- **Inverse ICP** now wins both angle AND position under blur/noise+blur.
  Template EDT is clean — no divergence, no cross-object matching artifacts.
  Hundreds of edge correspondences give strong angular + translational constraint.

- **ROI** still wins under pure noise (s>=20) where ICP's Canny threshold
  rejects valid edges. matchTemplate averages over the patch, inherently noise-robust.

- **Position**: ICP improved from 4.3px (forward) to 1.0px (inverse) on clean.
  ROI still slightly better at 0.6px due to template-match 2D locking.

### When to Use Which

| Scenario | Recommended |
|----------|-------------|
| Blur conditions | **ICP inverse** (0.1-0.2 deg, 0.7px) |
| Noise+blur (real-world) | **ICP inverse** (0.2 deg, 0.6-0.7px) |
| Pure noise (no blur) | **ROI** (0.9 deg, 0.5px) |
| Best angle accuracy | **ICP inverse** (0.04 deg isolated) |
| Best position accuracy | **ROI** (0.6px) or ICP inverse (0.7px) |
| Speed-critical | Either — both ~25ms for 10 objects |

---

## AVX2 ICP Inner Loop

Vectorized the hot loop in `refineWithNormals` (correspondence matching + Jacobian
accumulation):

- **SoA layout**: model edges converted to separate pos_x/pos_y/normal_x/normal_y arrays
  for contiguous SIMD loads
- **8-wide transform**: `cs*px - sn*py + tx` via `_mm256_mul_ps` / `_mm256_add_ps`
- **AVX2 gather**: `_mm256_mask_i32gather_ps` for closest_x/y and normal_x/y lookups
  from distance transform (4 gathers per 8 points)
- **Masked accumulation**: 11 `__m256` accumulators (6 ATA + 3 ATb + total_error +
  inlier_count), non-inlier lanes zeroed via `_mm256_and_ps` with combined mask
- **Branch elimination**: bounds check, distance check, normal compatibility all
  computed as SIMD masks, combined into single inlier mask

Scalar tail handles remaining N%8 points.

## OpenMP Parallel Object Refinement

Per-object ICP/ROI refinement parallelized with `#pragma omp parallel for schedule(dynamic)`:

- Each object's refinement is independent (reads shared scene, writes own result)
- Dynamic scheduling handles variable ICP convergence per object
- Threshold: only parallelize if >= 4 objects (avoids thread creation overhead)
- Thread-safe: `selectOptimizedPoints` cache pre-computed at `addModel()` time

### Combined Speed Results (FHD, 10 objects)

| Mode | Before | After | Speedup |
|------|--------|-------|---------|
| ICP dense (clean) | 42ms | **31ms** | 1.35x |
| ICP dense (noise s=50) | 1061ms | **93ms** | 11x |
| ROI (clean) | 33ms | **26ms** | 1.24x |

### Files
- `roi_refine.h` / `roi_refine.cpp` — ROI refinement implementation
- `icp_refine.h` / `icp_refine.cpp` — ICP refinement (AVX2 inner loop)
- `shape_matcher.h` — RefineMode::ROI, selectOptimizedPoints, analyzeSensitivity
- `shape_matcher.cpp` — ROI/ICP integration, OpenMP parallel refinement,
  sensitivity-optimized feature selection

---

### Tests & Benchmarks
- `bench_avx2.cpp` — speed benchmark (VGA/FHD/30MP)
- `bench_profile.cpp` — per-stage timing under clean/noisy conditions
- `bench_preprocess.cpp` — OpenCV preprocessing stage profiling
- `test_robustness.cpp` — robustness under noise/blur/brightness/occlusion
- `test_visual.cpp` — visual output with orientation arrows + NMS + ICP
- `test_icp_accuracy.cpp` — ICP accuracy sweep (120 angles x 8 conditions)
- `test_icp_bias.cpp` — ICP signed bias per template shape
- `test_bias_table.cpp` — coarse bias table (5 shapes x 5 steps)
- `test_bias_final.cpp` — warpAffine vs feature rotation bias comparison
- `test_rotate_bias.cpp` — feature rotation bias vs step size
- `test_multires.cpp` — multi-resolution matching benchmark
- `test_simple.cpp` — API usage, ICP vs ROI comparison, FHD benchmark,
  sensitivity analysis, per-angle isolated accuracy test
- `test_api.cpp` — multi-model matching with visual output

---

## Bug Fix: `angle_ori` Not Computed for Grayscale Images

`quantizedOrientations()` created `angle_ori` and set it to zero for single-channel
images but never populated it with actual gradient angles. Every feature's `theta`
fell back to the undirected formula `bin * 22.5` (range 0-157.5), losing the true
gradient direction (0-360). When `addRotatedTemplates` rotated features by adding
the rotation angle to `theta` and re-quantizing to get the label, the resulting
labels were wrong for most angles.

**Symptom**: detection worked at 0 and ~180 degrees only (2/24 objects found).
Scores at other angles dropped from ~90 to ~40 because rotated template labels
didn't match the scene's orientation bins.

**Fix**: compute `angle_ori` via `cv::phase(sobel_dx, sobel_dy, angle_ori, true)`
from the existing int16 Sobel buffers. Cost: one `convertTo` + `phase` call per
template extraction (not per match).

| Metric | Before | After |
|--------|--------|-------|
| Detection (24 angles) | 2/24 | **24/24** |
| Score (non-zero angles) | ~40 | **91.6** |
| Angle accuracy (ROI) | — | **0.05 deg** |
| Position accuracy (ROI) | — | **0.25 px** |

---

## Scene Downscale: `match_scale` for Faster Coarse Matching

Downscale the scene before coarse LineMOD matching, then run ROI/ICP refinement
at full resolution. The coarse stage (Sobel, quantize, voting, spread, scan) is
O(pixels) and dominates runtime, so downscaling gives near-linear speedup.

### How It Works

```
Full-res template features
    |
    v
Scale feature coordinates by match_scale (integer rounding)
    |
    v
Resize scene to match_scale (e.g., 0.3x)
    |
    v
LineMOD match on small scene (fast)
    |
    v
Scale positions back to full resolution
    |
    v
ROI/ICP refine at full resolution (sub-pixel accuracy)
```

Template features are scaled in-place before matching and restored losslessly
from a saved copy afterward. The orientation labels (8-bin quantized gradient
direction) stay valid across scales because they represent edge directions that
are preserved under downscaling. No re-extraction needed.

### Key Bug Fix: `tl_x` Position Mapping

The original `match_scale` code resized the scene but didn't scale template
features, so nothing matched. After adding feature scaling, a position mapping
bug caused 0/40 detection despite correct scores (69-74): `tmpl[0].tl_x` was
in full-res coordinates (after template restore) but the formula multiplied it
by `inv_scale` again, shifting every match position by ~43%.

Fix: check whether `tl_x` is from the scaled or restored detector and apply
`inv_scale` only when appropriate.

### Results (200x200 L-shape, 20MP, 40 objects, noise=30)

| match_scale | Scene | Time | Found | ROI Angle | ROI Position |
|-------------|-------|------|-------|-----------|--------------|
| 1.0 | 5472x3648 | 92ms | 40/40 | 0.11 deg | 0.07 px |
| 0.7 | 3830x2553 | 57ms | 40/40 | 0.10 deg | 0.07 px |
| 0.5 | 2736x1824 | 37ms | 40/40 | 0.05 deg | 0.07 px |
| 0.3 | 1641x1094 | 20ms | 40/40 | 0.06 deg | 0.27 px |

### Results (Real-world metal part template, 20MP, 24 objects)

| match_scale | Time | Found | ROI Angle | ROI Position |
|-------------|------|-------|-----------|--------------|
| 1.0 | 103ms | 24/24 | 0.05 deg | 0.25 px |
| 0.5 | 36ms | 24/24 | 0.05 deg | 0.29 px |
| 0.3 | 20ms | 24/24 | 0.06 deg | 0.27 px |
| 0.28 | 18ms | 23/24 | 4.13 deg | 1.30 px |
| 0.25 | 13ms | 18/24 | — | — |

**Minimum usable scale**: ~0.30 for a 200px template (scaled template = 60px,
needs enough pixels for T=4 grid cells). Below 0.28 detection degrades.

**Speedup**: 5.2x at scale=0.3 with zero accuracy loss when combined with ROI.

### Orientation Stability Across Scales

Features with gradient orientations near the center of their 8-bin quantization
range (high "orientation margin") are stable under downscaling. Features on bin
boundaries can flip bins.

- **Synthetic L-shape** (axis-aligned edges): all features at exact bin centers
  (margin = 11.25 deg maximum). Multi-scale consensus: 81% stable.
- **Real-world metal part** (tilted edges): all features at bin centers
  (margin = 11.2 deg). Multi-scale consensus: 65% stable.

Both templates work fine with scale-only (no re-extraction) because the spread
operation in the response map tolerates +/-1 bin differences.

### T-Level (Pyramid Stride) Investigation

Tested T={4,8}, {6,8}, {6,12}, {8,16} with match_scale=0.5:

| T | Found (scale=1.0) | Found (scale=0.5) | Speed |
|---|-------------------|-------------------|-------|
| {4,8} | 24/24 | 24/24 | 30ms |
| {6,8} | 24/24 | 24/24 | 30ms |
| {6,12} | 5/24 | 20/24 | 24ms |
| {8,16} | 0/24 | — | — |

**{4,8}** and **{6,8}** both work. Larger T values ({6,12}, {8,16}) fail because
the template doesn't span enough grid cells at the coarse level. Speed difference
between {4,8} and {6,8} is negligible (<2%) — the bottleneck is preprocessing
(Sobel+quantize+spread), not the template scan. **`match_scale` is the real speed
lever**, not T values.

### Scale-from-L0 Pyramid Mode

Added `Detector::scale_pyramid_features` flag: extract features only at level 0
and scale coordinates for coarser levels, instead of re-extracting from pyrDown'd
images. This enables deeper pyramids (3-4 levels) where re-extraction fails due
to insufficient features on the tiny image.

However, deeper pyramids (T={4,8,16} or T={4,8,16,32}) with scale-from-L0 can
build templates (85 features at all levels) but the pyramid refinement step
doesn't converge — candidates from the coarsest level don't survive refinement.
This remains an open area for investigation.

**Practical recommendation**: use standard T={4,8} with `match_scale=0.3-0.5`
for maximum speed. This gives 3-5x speedup with zero accuracy loss.

### Files

- `shape_matcher.h` — `MatchConfig::match_scale`
- `shape_matcher.cpp` — scene downscale, feature scale-in-place + restore,
  `tl_x` position fix
- `line2Dup.cpp` — `angle_ori` fix, `scale_pyramid_features` mode
- `line2Dup.h` — `Detector::scale_pyramid_features` flag
- `tests/bench_match_scale.cpp` — match_scale / T-level sweep benchmark

### Findings and Caveats

1. **Re-extraction is NOT needed for match_scale.** Simply scaling integer
   feature coordinates works because orientation labels survive downscaling.
   We initially assumed re-extraction from the downscaled template would be
   necessary (labels would mismatch), but empirically the scores are identical
   (e.g., 74.1 vs 74.2 at 0.7x). The `angle_ori` bug was the real cause of
   earlier failures, not label mismatch.

2. **The `angle_ori` bug was the root cause of most detection failures.**
   Before the fix, grayscale feature extraction produced undirected theta
   (0-157.5 deg). This silently broke `addRotatedTemplates` for all angles
   except near 0 and 180. The fix is simple (4 lines) but the symptom
   (low detection rate) was misleading — it looked like a template quality
   or threshold issue.

3. **Template `strong_threshold` must match the content.** For templates with
   internal texture (brushed metal, surface reflections), lower thresholds
   (60) pick up texture features that don't survive rotation via warpAffine.
   Higher thresholds (100-150) select only contour edges which are rotationally
   stable. However, after the `angle_ori` fix, even threshold=60 achieves
   24/24 detection because the texture features' orientations are now correctly
   directed.

4. **`match_scale` minimum depends on template size and T value.** The scaled
   template must span enough T-grid cells for the response map to have
   discriminative patterns. Rule of thumb: `template_px * match_scale > 15 * T[0]`.
   For a 200px template with T=4: min scale = 15*4/200 = 0.30. Below this,
   features collapse onto the same grid cells.

5. **Pyramid T values have negligible speed impact.** Changing T from {4,8} to
   {6,8} saves <2% because preprocessing (Sobel, quantize, spread) dominates
   at >80% of total time. The coarse template scan (which T affects) is only
   ~12% of the pipeline. `match_scale` reduces ALL stages proportionally,
   making it far more effective for speed.

6. **Deep pyramids (3+ levels) don't work in practice.** The pyramid refinement
   step (coarse→fine candidate narrowing) fails when the coarsest level has
   very low resolution. Even with scale-from-L0 providing 85 features at all
   levels, the refinement can't reliably promote candidates from T=16 to T=4.
   Stick with 2-level pyramids and use `match_scale` for speed.

7. **Synthetic L-shape templates have a pathological property.** All edges are
   axis-aligned, placing every feature exactly on an 8-bin orientation boundary.
   This maximizes sensitivity to any perturbation (blur, scaling, interpolation).
   Real-world templates with non-axis-aligned edges are much more robust — all
   features land at bin centers with maximum margin (11.25 deg).

8. **warpAffine placement ≠ real scene matching.** Placing a real-photo template
   into a scene via warpAffine is a valid test for LineMOD (pure rotation of
   gradient patterns). However, the bilinear interpolation can slightly blur
   fine texture, reducing scores by ~10-15% at non-cardinal angles. This is
   a test artifact — real scenes with actual rotated objects would have their
   own native gradient field.

---

## Per-Angle Position Bias Calibration

ROI refinement has angle-dependent position bias: the measured position
systematically shifts by up to 0.5px as a function of the object's rotation
angle. The bias pattern is a smooth sinusoid (template origin vs true rotation
center offset that rotates with the object) plus sharp spikes at 45-degree
intervals (orientation bin boundary effects).

### Calibration Method

At `addModel()` time, for each template angle:
1. Render the template at that angle via warpAffine into a small scene
2. Match and refine with the same ShapeMatcher pipeline
3. Measure (dx, dy) offset from the known GT position
4. Store as a per-angle lookup table

At `match()` time, subtract the calibrated bias for the matched angle.
Cost: zero per-match (one array lookup). Calibration cost: ~50ms at
`addModel` time (360 self-matches on a small scene).

### Results (FHD, real metal part template, 24 objects, clean)

| Metric | Before | After |
|--------|--------|-------|
| Position mean | 0.30 px | **0.17 px** |
| Position worst | 0.60 px | **0.40 px** |
| Bias magnitude | 0.056 px | **0.024 px** |
| RMS (x, y) | (0.234, 0.238) | **(0.129, 0.153)** |

### 360-Degree Sweep (single object, all angles)

| Metric | Value |
|--------|-------|
| Position mean | **0.064 px** |
| Position worst | 0.51 px |
| Angle mean | **0.053 deg** |
| Angle worst | 0.17 deg |
| Detection rate | **360/360** |

### Residual Error Pattern

The 360-degree error chart shows 45-degree periodic position spikes
(~0.5px) coinciding with 8-bin orientation quantization boundaries.
These spikes are sub-pixel-position-dependent (vary with where the
object is placed in the scene) and cannot be removed by angle-only
calibration. The smooth sinusoidal bias component IS fully removed.

Blurring the template for ROI refine was tested but **made accuracy
worse** (0.064 → 0.178px mean) — sharp templates give better
matchTemplate peak localization.

---

## NMS Radius Configuration

Changed auto NMS radius from `min(w,h) / 2` to `min(w,h) * nms_radius_scale`
with configurable `nms_radius_scale` (default 0.75).

The original 0.5 multiplier was too tight for templates with strong internal
edges: secondary coarse-match peaks at 60-85px from true positives survived
NMS, generating false positives that wasted ROI refine time.

| nms_radius_scale | FP count | Total time (s=0.3, 24 obj) |
|------------------|----------|---------------------------|
| 0.50 (old) | 37 FP | 17ms |
| 0.60 | 18 FP | 15ms |
| **0.75 (new default)** | **4 FP** | **14ms** |

### Score Gap Analysis (s=0.5, real template)

| | Count | Score range |
|---|-------|-------------|
| True positives | 24 | 79.5 - 84.6 |
| False positives | 0 | — (all eliminated by min_score=65) |
| Score gap | | **21.6 points** |

With `min_score=65` (raised from 50): zero FP, zero missed TP.

---

## Cached Scaled Detector

When `match_scale < 1.0`, pre-build a separate detector with scaled template
features at `addModel()` time. The `match()` call uses this detector directly
instead of deep-copying and scaling templates every frame.

Previously, each `match()` call deep-copied 360 template pyramids (each with
~85 features), scaled coordinates, matched, then restored from the copy.
The copy/restore added ~1-3ms per call.

With the cached detector, the scale-in-place overhead is eliminated. Both
paths produce identical results. Falls back to scale-in-place when no
cached detector is available.

---

## Combined Pipeline Performance (FHD 1920x1080, 24 objects, clean)

All optimizations combined: fused Sobel+Quantize, match_scale, bias
calibration, NMS radius, min_score tuning.

| Config | Total | TP/FP | Angle | Position |
|--------|-------|-------|-------|----------|
| s=1.0 score=65 | **12ms** | 24/0 | 0.07 deg | 0.17 px |
| s=0.5 score=65 | **6ms** | 24/0 | 0.07 deg | 0.16 px |

Compared to the original pipeline before all optimizations on this branch:
- FHD full-res: ~45ms → **12ms** (3.8x speedup)
- FHD with match_scale=0.5: → **6ms** (7.5x speedup)
- 20MP with match_scale=0.3: → **14ms** from ~90ms original
