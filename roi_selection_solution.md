# Unified Greedy Feature Selection for Rigid Pose Estimation

## 1. Unified Objective: Weighted D-Optimal Design

Use a single objective throughout:

```
S* = argmax_{S, |S|=N} log det(I_w(S))
```

where `I_w(S) = sum_{i in S} w_i * J_i^T * J_i` and `w_i = 1 / sigma_i^2`.

This is correct because each `w_i * J_i^T * J_i` is a rank-1 (or rank-0) contribution to the 3x3 information matrix. Corners and edges are *not* different types -- they are the same type with different weights. A corner with sigma=0.05 gets w=400; an edge with sigma=0.1 gets w=100. The Jacobian structure already encodes the geometric contribution; the weight encodes match quality. No two-phase approach is needed.

## 2. Estimating sigma_i from Template Data Alone

### The Structure Tensor Approach

For each candidate point i, compute the gradient structure tensor over the ROI patch (e.g., 16x16):

```
M_i = sum_{(x,y) in patch} w(x,y) * [gx^2, gx*gy; gx*gy, gy^2]
```

where `w(x,y)` is a Gaussian window. Let `lambda_1 >= lambda_2 >= 0` be its eigenvalues.

**Key insight**: the auto-correlation surface of the patch is locally quadratic with Hessian proportional to `M_i`. The matching precision along direction `d` is `sigma_d ~ C / sqrt(d^T M_i d)`, where C is a constant depending on noise level.

For the point-to-plane formulation, only the normal-direction precision matters:

```
sigma_i = C / sqrt(n_i^T * M_i * n_i)
```

where `n_i` is the edge normal. For a corner, `n_i^T M_i n_i ~ lambda_2` (the smaller eigenvalue, still large). For an edge, `n_i^T M_i n_i ~ lambda_1` (the large eigenvalue, since the normal aligns with the gradient).

**Practical formula**:

```
sigma_i = kappa / sqrt(n_i^T * M_i * n_i + epsilon)
```

- `kappa ~ 0.5` (calibrate once empirically: run matchTemplate on a few patches, measure actual sigma, fit kappa)
- `epsilon = 1.0` (prevents division by zero for featureless patches)

**Cost**: one structure tensor per candidate, O(patch_size^2) per point. For M=500 candidates with 16x16 patches, this is ~2M multiply-adds -- negligible.

### Why This Works Without matchTemplate

The Cramer-Rao bound for template matching precision is exactly `sigma = noise_std / sqrt(d^T M_i d)`. The structure tensor *is* the Fisher information of the patch. We are not approximating -- this is the theoretical lower bound, and NCC-based matching with parabolic subpixel interpolation approaches it closely.

## 3. Edge Contribution to I_w

An edge feature has anisotropic uncertainty: `sigma_normal` (small) and `sigma_tangent` (large). The point-to-plane constraint `(error . n_i) = 0` projects the observation onto the normal, so only `sigma_normal` enters:

```
w_i = 1 / sigma_normal_i^2 = n_i^T * M_i * n_i / kappa^2
```

The Jacobian row `J_i = [a_i, nx_i, ny_i]` is already a 1D constraint. The rank-1 matrix `w_i * J_i^T * J_i` is the complete contribution. No special edge/corner logic is needed -- the weight automatically gives corners ~4x more per-feature contribution (since their sigma is ~2x smaller) while the Jacobian geometry handles the rest.

**An edge does NOT contribute a 2D constraint with different sigmas.** It contributes exactly one scalar constraint (the normal-direction residual), weighted by how precisely that constraint can be measured. This is already what the point-to-plane Jacobian encodes.

## 4. Greedy Forward Selection Algorithm

```
function SELECT_FEATURES(candidates[1..M], N, d_min):
    // Precompute per-candidate quantities
    for i = 1 to M:
        M_i = structure_tensor(patch_i)       // 2x2
        sigma_i = kappa / sqrt(n_i^T * M_i * n_i + eps)
        w_i = 1.0 / (sigma_i * sigma_i)
        J_i = [a_i, nx_i, ny_i]              // 1x3
        // Precompute weighted outer product
        H_i = w_i * J_i^T * J_i              // 3x3, rank-1

    // Initialize information matrix with small regularizer
    // (prevents singular matrix before 3 features are selected)
    I_w = delta * I_3x3     // delta = 1e-6

    S = {}
    blocked = {}   // set of indices too close to selected points

    for k = 1 to N:
        best_score = -inf
        best_idx = -1

        for i = 1 to M:
            if i in S or i in blocked: continue

            // Greedy gain = log det(I_w + H_i) - log det(I_w)
            // By matrix determinant lemma (rank-1 update):
            // det(I_w + w*J^T*J) = det(I_w) * (1 + w * J * I_w^{-1} * J^T)
            // So gain = log(1 + w_i * J_i * I_w^{-1} * J_i^T)
            v = I_w^{-1} * J_i^T              // 3x1, solve 3x3 system
            gain = log(1 + w_i * dot(J_i, v)) // scalar

            if gain > best_score:
                best_score = gain
                best_idx = i

        if best_idx == -1: break   // no more valid candidates

        // Select feature
        S = S + {best_idx}
        I_w = I_w + H_{best_idx}

        // Block nearby candidates
        for i = 1 to M:
            if ||p_i - p_{best_idx}|| < d_min:
                blocked = blocked + {i}

    return S
```

**Complexity**: O(N * M) with O(1) per candidate evaluation (3x3 solve is constant-time). The `I_w^{-1}` is maintained incrementally via the Sherman-Morrison update:

```
I_w_inv_new = I_w_inv - (I_w_inv * H_i * I_w_inv) / (1 + w_i * J_i * I_w_inv * J_i^T)
```

This is O(9) multiply-adds per update (3x3 matrix).

**Near-optimality guarantee**: `log det` is a monotone submodular function of the selected set. Greedy forward selection achieves a `(1 - 1/e) ~ 63%` approximation ratio for submodular maximization under matroid constraints. The spacing constraint is a partition matroid (points in the same spatial cell conflict), so the guarantee holds.

### Why This Naturally Handles the Corner vs Edge Trade-off

- Early iterations: `I_w` is nearly singular. The greedy gain `log(1 + w_i * J_i * I_w^{-1} * J_i^T)` is huge for features that fill missing information directions. If no corners exist, far-apart edges with diverse normals will be selected first because they span the 3D information space.
- Later iterations: `I_w` is well-conditioned. The gain favors features with large `w_i` (high precision), since the geometric contribution is already covered. Corners win here due to lower sigma.

## 5. Special Cases

### Zero corners (circle)

For a circle, all normals are radial: `n_i = p_i / |p_i|`. The angular Jacobian `a_i = p_i x n_i = 0` for all features. The information matrix has zero angular information regardless of selection.

**Detection**: after selecting N features, check `I_w[0,0] < threshold`. If so, the template is rotationally ambiguous -- report this to the caller. No feature selection can fix a geometric degeneracy.

**Mitigation**: if the circle has any texture (even slight), the matchTemplate ROI will have non-radial gradient components. Use the dominant gradient direction of the patch (eigenvector of `M_i`) as `n_i` instead of the contour normal. This recovers some angular information from texture.

### Single normal direction (parallel lines)

Two parallel lines have normals `+n` and `-n` (equivalent for the outer product). The translational block `sum [nx^2, nx*ny; nx*ny, ny^2]` is rank-1. One translation axis is unconstrained.

**Detection**: after selection, check `lambda_min(I_w[1:2, 1:2]) < threshold`. Report the degenerate axis to the caller.

**The algorithm still does the best it can**: it will select features spread along the lines (maximizing angular leverage) and from both lines (if available), which is the optimal strategy for this degenerate geometry.

### Very few candidates (< N)

Select all valid candidates (after spacing filter). Return `|S| < N` with a flag. The caller can reduce `d_min` and retry, or accept fewer features.

## 6. Answers to the 5 Open Questions

**Q1: Can sigma_i be estimated cheaply?**
Yes. `sigma_i = kappa / sqrt(n_i^T * M_i * n_i + eps)` where `M_i` is the structure tensor. This is the Cramer-Rao bound for template matching. Cost: one 2x2 eigendecomposition per candidate.

**Q2: Closed-form for special geometries?**
For a regular N-gon with uniform edge weight: select one feature per edge, equally spaced along each edge, choosing the point with maximum `|p_i x n_i|` (farthest from center projected onto normal). For a circle: any N equally-spaced points are equivalent (and degenerate for rotation). No closed form exists for general shapes, but the greedy algorithm is O(NM) which is fast enough.

**Q3: Templates with < 3 corners?**
No special handling needed. The greedy algorithm automatically selects the best mix of edges and corners. With 0 corners, it selects edges with diverse normals and high leverage. With 1-2 corners, it selects those first (high gain on near-singular `I_w`), then fills with edges. The information-theoretic framework handles this seamlessly.

**Q4: Should selection adapt to the scene?**
No, for two reasons: (1) the selection happens once at template registration time, before any scene is available; (2) the Cramer-Rao bound already accounts for the signal-to-noise ratio through the structure tensor (higher contrast edges have larger `M_i` eigenvalues, hence smaller sigma). If scene noise is anticipated to be high, increase `kappa` globally -- this does not change the relative ranking.

**Q5: Unified objective without two-phase?**
Yes, this is exactly what `max log det(I_w(S))` with `w_i = 1/sigma_i^2` provides. The weight encodes match quality; the Jacobian encodes geometric contribution. The greedy gain `log(1 + w_i * J_i * I_w^{-1} * J_i^T)` automatically balances both in a single scalar score. No corner-first or edge-second logic is needed.

## Implementation Attempt: Results

The unified approach was implemented and tested. **It failed in practice:**

- ROI accuracy degraded from 0.064° to 0.62° (10x worse)
- Only 4 features selected (instead of 8) due to blocking
- All selected features were edges with same normal direction

### Root Cause

The weight `w_i = n^T M n / kappa^2` gives edges **higher** weight than corners because:
- Edge: `n` aligns with the gradient direction → `n^T M n ≈ lambda_1` (large)
- Corner: `n` is at an angle to both gradient directions → `n^T M n ≈ lambda_2` (smaller)

This causes edges to dominate the det(I_w), but they only provide 1D constraint.
The Jacobian `J_i = [a_i, nx, ny]` is already 1D (one row), so weighting by `1/sigma_normal^2`
correctly measures the normal-direction precision but OVERWEIGHTS the total information
contribution of edges relative to corners.

### The Fundamental Issue

The point-to-plane formulation already handles the anisotropy by projecting error onto the
normal. The `sigma_i` from the structure tensor measures precision *along that projection*,
which is correct. But det(I_w) doesn't know that an edge feature is "missing" the tangential
constraint — it sees a well-weighted 1D constraint and is satisfied.

A corner provides the same normal constraint PLUS an implicit tangential constraint that
prevents the matched position from sliding. This implicit constraint is not captured by
the Jacobian rows (only one row per feature) but is real in practice (matchTemplate pins
the position in 2D).

### Practical Conclusion

The two-phase approach (corners first by leverage, edges by det) works because:
1. Phase 1 ensures corners are selected — they provide 2D position anchoring
2. Phase 2 fills geometric gaps — diverse normals and high angular leverage

The unified `det(I_w)` fails because the Jacobian formulation is structurally blind to
the 2D vs 1D matching quality difference. A fix would require modeling corners as
contributing TWO Jacobian rows (normal + tangent), but this changes the problem structure.

**Current best: two-phase selection with 0.064° / 0.054px accuracy.**
