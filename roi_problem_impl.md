# ROI Feature Selection: Current Implementation

## Algorithm Overview

Two-phase greedy forward selection: corners first by leverage, edges by D-optimal det(J^T J).

```
Input:  M candidate features on template contour, target count N, min spacing d_min
Output: S = selected set of N features for ROI refinement

Phase 1 — Corners (2D anchors):
    Sort corners by R = distance from center, descending
    Greedily select well-spaced corners (skip if within d_min of any selected)

Phase 2 — Edges (fill geometric gaps):
    For each remaining slot:
        For each edge candidate not blocked by spacing:
            Trial: compute det(J^T J) with current set + this candidate
        Select the candidate with highest det(J^T J)
```

## Why Two Phases

The Jacobian-based information matrix `J^T J` captures geometric constraint structure
(normal directions, angular leverage) but is blind to matching quality (how precisely
matchTemplate can localize each feature).

- **Corner**: matchTemplate gives sharp 2D peak → reliable position in both X and Y.
  But J^T J sees only one row per corner — it doesn't know the tangent is also constrained.

- **Edge**: matchTemplate gives elongated 1D peak → reliable only along normal, slides
  along tangent. J^T J sees the same row structure as a corner — it can't tell the difference.

If we let det(J^T J) alone pick features, it selects edges with high gradient (strong
normal constraint) and ignores corners (weaker per-direction gradient). Result: all edges,
no 2D anchoring, 10x worse accuracy.

Phase 1 forces corner selection based on match quality (cornerness = 2D peak sharpness).
Phase 2 then optimally fills the remaining geometric constraint gaps.

## Unified Approach: Why It Failed

We attempted `max log det(I_w)` with `I_w = sum w_i J_i^T J_i`, `w_i = 1/sigma_i^2`,
where `sigma_i = kappa / sqrt(n^T M n)` from the gradient structure tensor.

**Result**: 0.064° → 0.62° accuracy (10x worse).

**Root cause**: `n^T M n` for an edge is ~lambda_1 (large), for a corner is ~lambda_2
(smaller). So edges get HIGHER weight than corners, dominating det(I_w). But edges only
provide 1D constraint — the high weight is misleading.

The fix would require modeling corners as contributing TWO Jacobian rows (normal + tangent),
each weighted by respective precision. This changes the problem structure and is left for
future work.

## Phase 1: Corner Selection

### What Qualifies as a Corner

A feature point where the PCA eigenvalue ratio of the gradient structure tensor is below
a threshold (currently `kCornerThreshold = 0.3`). Low ratio means the gradient has
comparable magnitude in two directions → the matchTemplate peak is sharp in 2D.

### Selection Criteria

Corners are sorted by `R = sqrt(px^2 + py^2)` — distance from template center.

**Why R matters**: the angular Jacobian component is `a_i = -py*nx + px*ny = |p| * sin(phi)`,
where phi is the angle between position vector and normal. Farther corners contribute more
angular constraint per unit of matching error.

### Spacing

Greedy selection with minimum distance `d_min = max(templ_width, templ_height) / 16 * 1.5`.
Skip any candidate within d_min of an already-selected feature.

### Typical Result

For a 200×200 L-shape: 4 corners selected at the bend points and arm endpoints.
These provide 2D anchoring that prevents positional sliding.

## Phase 2: Edge Selection via D-Optimal

### The Information Matrix

For the current selected set S, the 3×3 information matrix is:

```
J^T J = sum_{i in S} [ a_i^2       a_i*nx_i    a_i*ny_i  ]
                      [ a_i*nx_i    nx_i^2      nx_i*ny_i  ]
                      [ a_i*ny_i    nx_i*ny_i   ny_i^2     ]
```

where `a_i = -py_i * nx_i + px_i * ny_i` (angular Jacobian = position cross normal).

### Selection Criterion

At each step, pick the edge candidate that maximizes `det(J^T J)` when added to the
current set. This is the D-optimal criterion — it maximizes the total information volume.

```
det(J^T J) = J00*(J11*J22 - J12*J21)
           - J01*(J10*J22 - J12*J20)
           + J02*(J10*J21 - J11*J20)
```

### What det Naturally Prefers

1. **Novel normal directions**: an edge whose normal is perpendicular to existing edges
   adds a new dimension to the information matrix. An edge parallel to existing ones
   barely increases det (nearly redundant information).

2. **High angular leverage**: edges far from center with tangential normals
   (`a_i = R * sin(phi)` large) increase the `[0,0]` element of J^T J.

3. **Balanced directions**: det is the product of eigenvalues, so it penalizes
   lopsided information (much constraint in one direction, little in another).

### Typical Result

For the L-shape after 4 corners: 4 edges selected — typically one from each arm
segment, with diverse normal directions (horizontal and vertical normals on the
two arms), maximizing the remaining angular and translational constraint.

## Performance

### Accuracy (72-angle sweep, 200×200 L-shape, clean image)

| Metric      | Value   |
|-------------|---------|
| Angle mean  | 0.064°  |
| Angle worst | 0.27°   |
| Pos mean    | 0.054px |
| Pos worst   | 0.14px  |

### Speed

| Step                 | Time   |
|----------------------|--------|
| Feature selection    | 40ms   |
| Cached (subsequent)  | 0ms    |
| Per-match refinement | 0.8ms  |

Feature selection runs once at `addModel()` time and is cached.

## Implementation Details

### Source Location

`shape_matcher.cpp`, function `FeatureSet::selectOptimizedPoints(int max_points)`

### Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `kCornerThreshold` | 0.3 | Cornerness above this = corner |
| `kSolverRegularization` | 0.001 | Tikhonov reg for det computation |
| `kDefaultOptPoints` | 8 | Default N for match() |
| min_dist | templ_size/16*1.5 | Minimum feature spacing |

### Data Flow

```
extractFeatures()
    → refine_points[] with (px, py, nx, ny, cornerness)

addModel()
    → selectOptimizedPoints(8)     // triggers computation + cache
    → cached_opt_points stored     // reused by all match() calls

match()
    → cached_opt_points → ROI refine
```

### Normal Computation

Each candidate's normal is computed via Sobel on a 30×30 ROI patch centered at the
feature position in the template image. The direction of maximum gradient magnitude
within the patch determines the normal. This is done by `buildConstraint()`.

### Degenerate Template Detection

After selection, `analyzeSensitivity()` computes the hat matrix diagonal (leverage)
for each selected feature. If `worst_angle_sens > threshold`, the template has poor
angular constraint. This is reported to the user — the algorithm cannot fix geometric
degeneracy (e.g., circle, parallel lines), only detect it.

## Known Limitations

1. **Two-phase is a heuristic** — it's not provably optimal. The phase 1 corner
   selection may not be the best set of corners for the specific phase 2 edge complement.

2. **Corner detection depends on gradient structure** — template preprocessing
   (blur, contrast) affects which points are classified as corners.

3. **Fixed spacing** — d_min doesn't adapt to template size vs feature density.
   A template with features only in one region may waste the spacing budget.

4. **No match precision weighting** — all edges are treated equally in det(J^T J).
   High-contrast edges provide more precise matches but this is not captured.

5. **The sensitivity metric (worst_ang) can drift** — the threshold was relaxed from
   1.1 to 2.0 during development. Original value tracked in comments.

## Comparison of Approaches Tried

| Approach | Angle | Position | Notes |
|----------|-------|----------|-------|
| Greedy cornerness+leverage | 0.075° | 0.053px | Original heuristic |
| Fedorov exchange (swap) | 0.068° | 0.054px | Iterative improvement |
| Hat matrix leverage swap | 0.068° | 0.054px | Analytic, 18x faster |
| Corners first + det edges | **0.064°** | **0.054px** | Current best |
| Pure det(J^T J) all features | 0.82° | 1.10px | No corners selected → bad |
| Unified w_i = 1/sigma^2 | 0.62° | 1.76px | Edges over-weighted → bad |
