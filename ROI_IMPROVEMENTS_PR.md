# ROI refine: edge-only point selection + match confidence (+ investigation knobs)

## Summary
Adds an **edge-only ROI point selection** mode and a **per-result ROI confidence**
signal to the shape matcher, plus several opt-in experiment knobs. **Default
behavior is unchanged** — every new feature is opt-in (default off / 0 / false) and
the original tuned constants are untouched. `--fast` regression: **73/73 green**;
ROI section: **18/18 green** (per-call refine still 0.83 ms).

The two changes worth adopting:

| Feature | Flag | Effect |
|---|---|---|
| **Edge-only points** | `MatchConfig.roi_edge_only_points` (+ `roi_min_spacing≈12`) | Selects EDGE points only via pure D-optimal + spacing, keeping the original 2D matchTemplate refine. Fixes corner-clustering worst-case. **~same speed.** |
| **Refine confidence** | `MatchResult.refine_residual` (output) | Mean \|point-to-line\| fit residual (px) of the matched points. Detects completely-off matches. |

## 1. Edge-only ROI point selection (recommended)
`selectOptimizedPoints(max, spacing, edge_only=true)` takes a clean self-contained
branch: unweighted **D-optimal over EDGE refine points** (their stored normals) with
a **min pairwise spacing**, no grid / no leverage-priority corner phase. The
**per-point 2D matching is identical** to today — only the point set differs.

Why: the default selector grabs high-leverage **corners** first, which can **cluster**
on shapes whose discriminative features are concentrated (e.g. an L's corner, a flag's
triangle). Under a slightly-off coarse pose this clustering yields a large worst-case
error. Spreading well-conditioned **edges** removes it.

Measured (synthetic F/flag/L/T, origin error mean|worst px, clean / noise σ=30):

| shape | default worst (clean/noise) | edge-only worst |
|---|---|---|
| F | 0.24 / 0.25 | 0.20 / 0.23 |
| flag | 0.27 / 0.41 | 0.27 / 0.31 |
| **L** | 0.48 / **2.56** | 0.39 / **0.41** |
| T | 0.33 / 0.38 | 0.33 / 0.35 |

All shapes ≤ 0.41 px worst (clean **and** noisy); the L noise outlier 2.56 → 0.41.
Speed: ~1.0× default for L/T, ~1.1–1.7× for flag (measurement-noisy); selection is
one-time at `addModel`, per-match cost unchanged.

`roi_min_spacing` (default 0 = off) also caps ROI-window overlap on its own; ~10–12 px
is safe, larger values are a per-shape trade-off (can exclude clustered points).

## 2. `MatchResult.refine_residual` — per-result confidence (recommended)
ROI refine now reports the mean |point-to-line| residual of the sample points at the
final pose. Trustworthy matches agree on one pose (residual ~0); occlusion / gross
mismatch / off-init makes them disagree (large). Use a threshold (~1–2 px) to flag
**completely-off matches** at the result level.

Measured (flag, ROI): good clean 0.14 px, good+noise30 0.15 px, occluded **7.4 px**,
heavy clutter 0.27 px, extreme noise60 0.17 px — a ~25× separation tracking the actual
origin error. (-1 = not computed, i.e. refine != ROI.)

## 3. Opt-in investigation knobs (default off; kept for tuning, not recommended on by default)
- `roi_iterative_rematch` — ICP-style re-match every iteration. +~30 % accuracy on
  edge-only (re-warps the template patch to the refined angle, not just the search),
  costs ~1.2–2× refine. The plain solve already converges in ~2 Gauss-Newton iters.
- `roi_max_iters` (0 = default 3) — solve iterations. **2 is enough** (×2=×3=×5);
  exposed for tuning.
- `roi_edge_1d_match` — 1D profile match along the edge normal (tangent-averaged).
  Accurate + noise-robust, but **slower than 2D** (per-match scene resampling), so not
  a speed win for arbitrary-angle edges.
- `roi_edge_collapse` — narrow-search matchTemplate → direct 1D result. Same story:
  accurate but not faster (still needs scene resampling).
- `roi_distinct_pct` / `roi_weight_by_distinct` — filter/weight points by clean-template
  self-distinctiveness. Helps some shapes, hurts others' worst-case → **not a safe
  default**.
- `roi_reject_low_score` (+ `roi_reject_angle_tol`, `roi_reject_pct`) — per-point score
  gate. **Inert** with TM_CCORR_NORMED (scores too compressed) and the worst cases are
  systematic, not per-point — use `refine_residual` instead.

## Tried and reverted (with evidence; do not re-attempt blindly)
- **2D facet / TM_CCOEFF_NORMED subpixel** — no change to the ~0.12–0.15 px ROI floor;
  the floor is the warp-and-correlate match, not the peak interpolation.
- **rewarp tuning** (rematch 0.5 / rewarp 1 / iters 6) — ~2× ROI cost for a
  shape-specific benefit; reverted to original constants.
- **contiguous template-block cache** — a 30×30 block already fits in L1; cloning adds
  a copy with no warp speedup.
- **contiguous search-window cache (5MP)** — no measurable per-call speedup; at 5 MP the
  **coarse** stage dominates, the refine stays ~0.8 ms/call.

## Bug fix: flipped-match refine (behavior change for `ModelConfig.flip = true`)
`ModelConfig.flip` previously refined every flipped hit against the **non-flipped**
base template (it kept only the flipped *features*, discarding the flipped template).
Now `addModel` builds a **complete independent flipped template**
(`Impl::buildFlippedTemplate`: `cv::flip(templ_image, ., 0)`, mirrored `refine_points`
& `origin`, freshly recomputed caches) stored as `ModelInfo::features_flip`, and the
match loop refines flipped hits against it (dropping the now-unneeded `oy=-oy` patch).
**flip ROI origin 1.36 px → 0.08 px** (= control), `tests/test_flip_refine.cpp`. The
addModel cache precompute was refactored into `Impl::precomputeFeatureCaches` (used for
base and flip). This is a **fix**, not opt-in — flipped matches were inaccurate before.
(See `flip_template_problem.md`, now marked resolved.)

## Investigation byproducts (FYI)
- The flip experiments had a **test-harness aliasing bug** in `place()`
  (`Mat src = templ; cv::flip(templ, src, 0)` flips in place). Fixed in
  `tests/test_flip_refine.cpp`; doc "Experimental status" corrected. The SAME trap
  bit `buildFlippedTemplate` (flip into a fresh Mat, not a shallow copy).
- `ShapeMatcher::addModel`: a model named `"<X>_flip"` collides with model `"<X>"`'s
  internal `class_id_flip` and mis-attributes hits. (Naming caveat; not fixed here.)

## Files
- `include/shape_matcher.h`, `shape_matcher.cpp`, `roi_refine.{h,cpp}` — the features
  above (all opt-in; default path untouched).
- `tests/` — new hermetic tests + `CMakeLists.txt` targets (test_edge_only,
  test_roi_confidence, test_refine_accuracy, test_roi_overlap/spacing, test_roi_iters,
  test_roi_speed_1d, test_multi_template, test_flip_separate, …).

## Risk
Low. Default code path and tuned constants unchanged; new behavior is gated behind
default-off flags and one new (-1-by-default) output field. Regression green.
