# ROI Feature Selection Problem: Optimal N-Point Selection for Rigid Pose Estimation

## Problem Statement

Given a template contour with M candidate feature points (M >> N), select exactly N points to serve as ROI matching anchors for rigid body pose refinement. The selected set must minimize pose estimation error under realistic matching noise.

## System Model

### Pose Estimation

The system estimates a 2D rigid transform (theta, tx, ty) by:
1. For each selected feature point, extract a small ROI patch from the template
2. Match each patch against the scene using `matchTemplate` to find the correspondence
3. Solve the over-determined system of point-to-plane constraints for (theta, tx, ty)

The linearized constraint from feature i is:

```
(R * p_i + t - dst_i) . n_i = 0
```

where:
- `p_i = (px_i, py_i)` — feature position relative to template center
- `n_i = (nx_i, ny_i)` — edge normal direction (unit vector)
- `dst_i` — matched position in scene (from matchTemplate)

The Jacobian row for feature i is:

```
J_i = [ -py_i * nx_i + px_i * ny_i,   nx_i,   ny_i ]
        \_________________________/    \___________/
           angular component            translation
           = p_i x n_i (cross product)
```

The normal equations are `J^T J x = J^T b`, where `J^T J` is a 3x3 information matrix.

### Two Types of Features

#### Corner Features

A corner has a sharp correlation peak in **both** directions. The matchTemplate response surface has two large eigenvalues (eigenvalue ratio close to 1).

- Provides **2D positional constraint** — the match locks position in both X and Y
- The matched position `dst_i` is reliable in all directions
- Effective matching uncertainty: isotropic, small sigma in both axes

#### Edge Features

An edge has a sharp correlation peak only along the **normal** direction. The tangential direction has a flat, elongated peak (eigenvalue ratio >> 1).

- Provides **1D positional constraint** — only along the edge normal
- The matched position `dst_i` is reliable along normal, unreliable along tangent
- Effective matching uncertainty: anisotropic — small sigma_normal, large sigma_tangent
- The point-to-plane formulation `(error . n_i)` correctly projects out the unreliable tangent component

### Key Parameters Per Feature

For each candidate feature i:

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Position | `(px_i, py_i)` | Relative to template center |
| Normal | `(nx_i, ny_i)` | Edge gradient direction (unit vector) |
| Distance from center | `R_i = \|p_i\|` | Leverage for angular constraint |
| Position angle | `theta_i = atan2(py_i, px_i)` | Angular position on template |
| Normal angle | `phi_i = atan2(ny_i, nx_i)` | Edge orientation |
| Cornerness | `c_i` | 0 = pure edge, 1 = strong corner |
| PCA eigenvalue ratio | `lambda_1 / lambda_2` | Match peak shape (low = corner, high = edge) |
| Angular Jacobian | `a_i = -py_i * nx_i + px_i * ny_i` | = `R_i * sin(angle between p_i and n_i)` |
| Match precision | `sigma_i` | Expected 1-sigma matching error (px) |

## The Optimization Problem

### Objective

Select S subset of {1, ..., M} with |S| = N to minimize the worst-case pose estimation error under 1-pixel matching noise in each selected feature.

### Information Matrix

The 3x3 information matrix for a selected set S is:

```
I(S) = J_S^T J_S = sum_{i in S} J_i^T J_i
```

Expanded:

```
I(S) = sum_i [ a_i^2       a_i*nx_i    a_i*ny_i  ]
              [ a_i*nx_i    nx_i^2      nx_i*ny_i  ]
              [ a_i*ny_i    nx_i*ny_i   ny_i^2     ]
```

where `a_i = -py_i * nx_i + px_i * ny_i`.

### What Makes I(S) Well-Conditioned

The information matrix must have three large, comparable eigenvalues:

1. **Angular information** (top-left block): `sum a_i^2` must be large
   - Requires features with large `|a_i| = |p_i x n_i|`
   - Maximized when normal is **tangential** to the circle around center (perpendicular to radial direction)
   - Proportional to R_i — farther features contribute more
   - Features with normal pointing toward/away from center contribute **zero** angular information

2. **Translational information** (bottom-right 2x2 block): `sum [nx^2 nx*ny; nx*ny ny^2]` must have two large eigenvalues
   - Requires **diverse normal directions** — normals spanning at least two non-parallel orientations
   - If all normals point the same way, one translation axis is unconstrained
   - Optimally: normals uniformly distributed in angle, making this block ~ (N/2) * I

3. **Decoupling** (off-diagonal blocks): `sum a_i*nx_i` and `sum a_i*ny_i` ideally near zero
   - Achieved with symmetric feature placement
   - Reduces cross-sensitivity between rotation and translation estimates

## Complications

### 1. Match Precision Varies by Feature Type

The Jacobian treats all features equally, but their matching quality differs:

- **Corner**: sigma ~ 0.05 px (sharp 2D peak, parabolic subpixel works well)
- **Edge along normal**: sigma ~ 0.1 px (1D peak, reasonable subpixel)
- **Edge along tangent**: sigma ~ 5-50 px (flat peak, essentially random)

The point-to-plane formulation projects out the tangent error, but the feature's effective information contribution depends on its match precision. A corner at R=10 may provide more reliable information than an edge at R=50.

The weighted information matrix should be:

```
I_w(S) = sum_{i in S} (1/sigma_i^2) * J_i^T J_i
```

where sigma_i reflects the actual matching uncertainty along the constraint direction.

### 2. Corner vs Edge Trade-off is Non-Obvious

Consider two candidates:
- Corner A: at center (R=5), cornerness=1.0, sigma=0.05 px
- Edge B: far from center (R=80), cornerness=0, sigma=0.1 px (along normal)

Edge B has 16x more angular leverage (`a_i ~ R`), but Corner A locks position in 2D. Which is better depends on the **existing set** — if we already have angular information, the corner adds more; if we lack angular constraint, the far edge is critical.

### 3. Normal Direction Diversity

Two edge features with the same normal direction are nearly redundant — adding the second barely increases det(I). The information gain per feature depends on what's already selected.

Example: a rectangle has edges with only 2 normal directions (horizontal and vertical). Adding more horizontal edges after the first few gives diminishing returns — the system needs vertical edges to constrain the other translation axis.

### 4. Spatial Clustering

Features too close together provide correlated information (their ROI patches overlap, and their Jacobian rows are nearly identical). A minimum spacing constraint is needed, but how to balance spacing vs information content?

### 5. Template-Dependent Difficulty

| Template | Difficulty | Why |
|----------|-----------|-----|
| L-shape | Easy | 4 corners, 2 edge directions, asymmetric |
| Rectangle | Medium | 4 corners, but 2 edge directions only |
| Circle | Hard | 0 corners, all normals are radial (zero angular Jacobian!) |
| Parallel lines | Degenerate | 1 edge direction, 0 corners |
| Single line | Impossible | 1D constraint only |

### 6. The N Budget Problem

With N features, each taking ~0.1ms for matchTemplate:
- N=8: fast (0.8ms) but fewer constraints
- N=15: slower (1.5ms) but more averaging
- Diminishing returns: error decreases as ~1/sqrt(N) beyond a minimum N

The optimal N depends on the template complexity and the desired accuracy/speed trade-off.

## Formal Problem Definition

```
Given:
  M candidate features with properties {p_i, n_i, c_i, sigma_i}
  Target count N
  Minimum spacing d_min

Find:
  S* = argmax_{S, |S|=N} f(S)

Subject to:
  ||p_i - p_j|| >= d_min  for all i, j in S, i != j

Where f(S) is one of:
  (a) D-optimal:  max det(I_w(S))         — maximize total information
  (b) A-optimal:  min trace(I_w(S)^{-1})  — minimize average variance
  (c) G-optimal:  min max_i h_ii          — minimize worst-case leverage
  (d) E-optimal:  max lambda_min(I_w(S))  — maximize weakest direction

With:
  I_w(S) = sum_{i in S} w_i * J_i^T J_i
  w_i = 1/sigma_i^2 (match precision weight)
  sigma_i estimated from PCA eigenvalue ratio of the ROI gradient
```

By the Kiefer-Wolfowitz equivalence theorem, criteria (a) and (c) yield the same optimal design for continuous (relaxed) problems. For the discrete subset selection problem, they may differ but greedy algorithms give good approximations for both.

## Open Questions

1. Can sigma_i be estimated cheaply without running matchTemplate? (PCA eigenvalue ratio is a proxy but not exact)
2. Is there a closed-form solution for special template geometries (convex, star-shaped)?
3. How to handle templates with very few corners (< 3)? Fall back to edge-only with diversity?
4. Should the selection adapt to the scene (e.g., if scene is noisy, prefer corners more)?
5. Is there a unified objective that naturally balances corner reliability vs edge leverage without a two-phase approach?
