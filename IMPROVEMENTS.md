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

### ICP Architecture

```
Coarse match (LineMOD)
    |  ~5 deg accuracy, ~2.5px accuracy
    v
Spatial NMS (deduplicate overlapping detections)
    |
    v
Local ROI ICP (per object, ~0.3ms each)
    |  Extract 160x160 patch around match
    |  Canny + EDT + normals on patch only
    |  Point-to-plane ICP with normal compatibility check
    v
Refined pose: < 1 deg, < 1px accuracy
```

### Key ICP Features

- **Normal compatibility filtering** — reject correspondences where model and scene edge normals disagree by > 45 degrees. Prevents cross-part matching (e.g., vertical arm matching horizontal arm edges). This single fix eliminated all ICP divergence cases.

- **Point-to-point regularization** — small weighted point-to-point term alongside point-to-plane prevents sliding along straight edges.

- **Local ROI processing** — compute Canny + distance transform + normals only in a small patch around each match. ~0.3ms per object vs 40ms for full-scene processing.

- **EDT-based distance field** — use OpenCV `distanceTransform` with `DIST_LABEL_PIXEL` for O(W*H) closest-edge lookup instead of brute-force O(W*H*max_dist^2).

- **3x3 LDL solver** — no Eigen dependency. Custom 3x3 symmetric positive-definite solver for the SO2 pose update (theta, tx, ty). Also supports 4x4 solver for Sim2 (with scale).

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

### ICP vs ROI Comparison

| | ICP (dense) | ROI 8pt x 3 | ROI 15pt x 5 |
|---|---|---|---|
| **Speed** | 1.3ms | **0.7ms** | 2-5ms |
| **Accuracy** | **<0.2 deg** | <1 deg | <0.5 deg |
| **Robustness** | +-8px/+-8deg | **+-20px/+-20deg** | **+-30px/+-30deg** |

### Robustness Under Degradation

| Condition | ICP | ROI 8pt x 3 |
|---|---|---|
| Noise sigma=10 | @-0.2 deg 0.1px | @-0.1 deg 0.1px |
| Noise sigma=30 | @-0.2 deg 0.1px | @-0.1 deg 0.2px |
| Noise sigma=50 | @+0.1 deg 0.2px | @+0.1 deg 0.2px |
| Blur k=5 | @-0.1 deg 0.2px | @+0.1 deg 0.1px |
| Blur k=11 | @+0.0 deg 0.1px | @+0.2 deg 0.3px |
| Blur k=21 | **@-0.0 deg 0.1px** | @+0.9 deg 1.1px |
| +10px +10deg | @+2.8 deg FAIL | **@+0.3 deg 0.1px** |
| +20px +20deg | @+8.5 deg FAIL | **@+1.0 deg 1.4px** |
| +10px+10deg+noise+blur | @-0.2 deg 0.2px | @+0.6 deg 0.7px |

### When to Use Which

| Scenario | Recommended |
|----------|-------------|
| Good coarse init, any noise/blur | **ICP** (1.3ms, <0.2 deg) |
| Bad coarse init, moderate conditions | **ROI 8pt x 3** (0.7ms, <1 deg) |
| Bad coarse init, need sub-degree | **ROI 15pt x 5** (2-5ms, <0.5 deg) |
| Heavy blur (k > 15) | **ICP** (edge-based, blur-invariant) |
| Real-time tracking (frame-to-frame) | **ROI 8pt x 3** (0.6ms with cache) |

### Files
- `roi_refine.h` / `roi_refine.cpp` — ROI refinement implementation
- `shape_matcher.h` — RefineMode::ROI added
- `shape_matcher.cpp` — ROI integration, template image serialization

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
- `test_simple.cpp` — minimal API usage + ICP vs ROI robustness comparison
- `test_api.cpp` — multi-model matching with visual output
