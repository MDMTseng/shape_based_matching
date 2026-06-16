# Flip handling is incomplete: refine uses the non-flipped template

> **RESOLVED (2026-06-16) — `buildFlippedTemplate` implemented.** `ModelConfig.flip`
> now builds a complete independent flipped template (`Impl::buildFlippedTemplate`:
> flipped `templ_image` via `cv::flip(.,0)`, flipped `refine_points`, mirrored
> `origin`, freshly recomputed caches), stored as `ModelInfo::features_flip`, and the
> match loop refines flipped hits against it (`fs = features_flip`; the `oy=-oy` patch
> is dropped — the mirrored origin already encodes it). Verified by
> `tests/test_flip_refine.cpp`: **flip ROI origin 1.36px → 0.08px** (= control 0.06px).
> Caveat found during implementation: `cv::flip` into a shallow-copied Mat aliases the
> base buffer — write to a fresh Mat (same trap as the test harness's `place()`).
> The remaining visSele symptom is the **consumer flip/angle convention** (separate),
> and there is **no** matcher-side flip-detection ambiguity (that was a test bug).

## Summary

When a model is added with `ModelConfig.flip = true`, the matcher refines every
flipped hit against the **non-flipped** template image + non-flipped refine points:
it flips only the *gradient features* into a second detector class
(`class_id_flip`) and discards the flipped `FeatureSet`. This is a real
architectural smell — the refine is fed the wrong template for flipped matches.

**Experimental status (UPDATED 2026-06-15 — earlier numbers were a test bug).**
The hermetic experiment `tests/test_flip_refine.cpp` measures ground-truth pose
error for a flipped object vs an identical non-flipped control.

> CORRECTION: the original writeup here claimed the flipped origin "still localizes
> to <1px (≈0.83px, on par with the control)" and that the base-template reuse does
> "not produce a dramatic refine failure", and reported a "flip-detection ambiguity"
> for weakly-chiral shapes. **All three claims were artifacts of an aliasing bug in
> the test harness's `place()` helper** (`Mat src = templ; cv::flip(templ, src, 0)`
> shares templ's buffer → flips the template IN PLACE, toggling which instance is
> mirrored every iteration). Fixed in `test_flip_refine.cpp` and proven in
> `tests/test_flag_sweep.cpp` / `tests/test_flip_separate.cpp`.

With the harness fixed, the **base-template-refine bug is real and measurable**: on
the clean synthetic "F", control ROI origin error is ~**0.06px** while the flipped
ROI origin error is ~**1.36px** (≈20× worse), with flip angle ~2.2° vs control ~1.3°.
So the base-template reuse for flipped matches DOES degrade the refined pose — the
earlier "on par with control" reading was wrong. It is degraded but not catastrophic
on a clean F (1.36px, not tens of px), so the severe visSele symptom (~half the
calipers miss, angle off ~150°) is still likely **dominated by the consumer's
flip/angle convention** (visSele also fails flip with `RefineMode::None`, which
bypasses refine) — but this refine path is a genuine, separate contributor, not a
non-issue. There is **no** matcher-side flip-detection ambiguity: a chiral shape's
correct-chirality model scores 100 and the wrong chirality ~60 at every angle
(`test_flag_sweep.cpp`); that earlier claim was the same harness bug.

Separately, `tests/test_flip_separate.cpp` shows that giving the flipped match its
OWN correct template (registering original + manually-flipped as two models) brings
the flipped refined pose back to control quality (~0.12px) — i.e. the fix below works.

So this document describes a **correct architectural fix** (independent flipped
template) that is both the right design AND now backed by clean measurements: it
should bring the flipped ROI origin from ~1.36px down to control's ~0.06px. The
proposed fix: build a **complete, independent flipped template** (its own flipped
`templ_image`, `refine_points`, `origin`, recomputed refine caches) and use it for
every flipped-match stage — not just flip the feature geometry under a new class name.

## How flip works today

`ShapeMatcher::addModel()` (shape_matcher.cpp ~1691) registers two detector
classes per model:

- `class_id`      — the base features.
- `class_id_flip` — `Impl::flipFeatures(features)` (shape_matcher.cpp:1574).

`flipFeatures()` produces a `FeatureSet` that flips **only**:

```cpp
static FeatureSet flipFeatures(const FeatureSet& fs) {
    FeatureSet flipped = fs;                 // shallow copy of EVERYTHING
    for (auto& lv : flipped.levels)
        for (auto& f : lv.features) {
            f.y = (2*center_y - true_y) - lv.tl_y;   // mirror feature Y about center
            f.theta = -f.theta; ...                  // mirror gradient orientation
        }
    flipped.origin.y = fs.templ_height - fs.origin.y; // mirror origin Y
    return flipped;
}
```

That flipped `FeatureSet` is converted to detector templates and added under
`class_id_flip` — **then discarded**. `ModelInfo` (shape_matcher.cpp:1522) stores
only the **base** `FeatureSet features;` — there is no `features_flip`.

At match time (shape_matcher.cpp ~1968):

```cpp
for (auto& model : impl_->models) {
    if (m.class_id == model.class_id)      { mi = &model; break; }
    if (m.class_id == model.class_id_flip) { mi = &model; is_flip = true; break; }
}
auto& fs = mi->features;        // <-- ALWAYS the BASE FeatureSet, even when is_flip
```

`is_flip` is then used for exactly two small corrections:

- `if (is_flip) oy = -oy;`                               (origin Y, line ~2003)
- `if (is_flip) user_angle = -user_angle + 2*offset;`    (reported angle, line ~2011/2042/2087)

Everything else — **the refine** — uses the base `fs` unchanged.

## Why the pose is wrong for a flipped match

`FeatureSet` carries several assets that the refine consumes. Here is what
`flipFeatures()` mirrors vs. what the flipped match actually needs:

| FeatureSet member        | used by                         | flipped by `flipFeatures`? | flipped variant reaches refine? |
|--------------------------|---------------------------------|----------------------------|----------------------------------|
| `levels[].features`      | coarse line2Dup match           | **yes** (Y + θ)            | yes (added as `class_id_flip`)   |
| `origin`                 | result x/y mapping              | yes (`origin.y`)           | partial — refine uses base `fs.origin`, then `oy=-oy` patch |
| `templ_image`            | **ROI refine** (`refineROI`)    | **NO**                     | **NO — base image used** ✗       |
| `refine_points`          | **ROI refine** sample points    | **NO**                     | **NO — base points used** ✗      |
| `cached_opt_points`      | ROI refine sample positions     | NO                         | NO ✗                             |
| `cached_lock_info`       | ROI refine per-point constraint | NO                         | NO ✗                             |
| `cached_templ_scene`     | **ICP refine**                  | NO                         | NO ✗                             |

The two refine paths (shape_matcher.cpp):

```cpp
// ROI refine (~2048)
auto opt_points = fs.selectOptimizedPoints(...);     // base points
roi_refine::refineROI(fs.templ_image, scene, sample_pts, init_pose, roi_cfg);
//                    ^^^^^^^^^^^^^^^ base (non-flipped) template image

// ICP refine (~2019)
icp_refine::refineInverse(fs.cached_templ_scene, ...); // base template edges
```

So for a flipped match the refine is asked to align the **non-mirrored** template
image / edge set to a **mirrored** scene region. It cannot converge correctly:
it drives the pose to whatever minimizes a mismatched template, corrupting
`scene_x/scene_y/raw_angle`, and that corrupted pose is what gets reported.

Symptom in practice (visSele B5S_dddddn2, horizontally-mirrored input): the part
is detected with `flipped=true` at the correct mirrored centre, but downstream
caliper placement lands wrong (≈half the calipers miss) and the measured angle is
off by ~150°. With `RefineMode::None` it is still wrong (origin/angle convention),
but ROI refine makes it worse, confirming the refine half of the bug.

> Note: the **angle/flip convention** at the *consumer* side (how `result.angle`
> + `result.flipped` map to a measurement pose) is a **separate** bug tracked on
> the consumer (visSele). This document is only about the submodule's
> responsibility: returning an **accurate refined pose** for a flipped match.

## The fix: an independent, complete flipped template

Replace "flip the feature geometry into a new class name" with "build a full
flipped template once, store it, and use it for the flipped match's refine."

1. **`buildFlippedTemplate(const FeatureSet& base) -> FeatureSet`** — a superset of
   today's `flipFeatures()` that mirrors the **whole** template about the same
   horizontal-mirror axis used for the features:
   - `levels[].features`     — as today (Y about center, θ negated).
   - `templ_image`           — `cv::flip(base.templ_image, 0)` (mirror rows; the
     feature mirror is about the horizontal axis → flip Y).
   - `refine_points`         — for each `RefinePt`: `py' = (H-1) - py` (in the same
     center-relative frame the features use), and `ny' = -ny` (mirror the normal
     Y); `px,nx,cornerness,type` unchanged.
   - `origin`                — `origin.y' = templ_height - origin.y` (as today).
   - `cached_opt_points` / `cached_lock_info` / `cached_templ_scene` /
     `templ_scene_valid` — **clear** them so they are recomputed lazily from the
     flipped `templ_image`/`refine_points`, OR recompute them eagerly here. Do NOT
     copy the base caches (that is the current latent trap even if `templ_image`
     were flipped).
   - `angle_offset`, `templ_width/height` — unchanged.

2. **Store it.** Add `FeatureSet features_flip; bool has_flip = false;` to
   `ModelInfo`. In `addModel`, when `config.flip`, build it once and keep it.

3. **Use it.** In the match loop, when `is_flip`, set `fs = mi->features_flip`
   (instead of `mi->features`). Then **delete** the `if (is_flip) oy = -oy;` patch
   — the flipped `fs.origin` already encodes it, so the existing
   `ox/oy = fs.origin - templ_center` math is correct without the patch. The
   `user_angle = -user_angle + 2*offset` mapping stays (that is the matched-angle
   convention for a mirrored template, independent of refine).

4. **Confirm the mirror axis is internally consistent.** `flipFeatures` mirrors Y
   (rows). `templ_image` must be flipped on the same axis (`cv::flip(...,0)`), and
   `refine_points.py`/`ny` mirrored on the same axis. If any one of the three uses
   a different axis, refine still diverges — so the test (below) must check the
   refined pose, not just detection.

### Cost

`buildFlippedTemplate` runs once per model at `addModel` time (one `cv::flip` + one
pass over `refine_points` + the same cache precompute the base already does). Zero
per-match cost. Memory: one extra `templ_image` + `refine_points` per flipped model.

## Minimal standalone test (no visSele)

Goal: prove that a flipped match's **refined** pose is accurate, on a purely
synthetic image, using only the submodule's public API. Put it in
`tests/test_flip_refine.cpp` and add an `add_executable` in `CMakeLists.txt`
(mirror `test_robustness`). It must use an **asymmetric** template so a flip is
geometrically distinguishable from a rotation (e.g. an "F"/"R"/"L" glyph or an
L-bracket with a notch).

### Procedure

1. **Template.** Render an asymmetric shape (e.g. white "F" on black, ~200×200) into
   `templ`. `FeatureSet feat = extractFeatures(templ);` `feat.setOrigin(cx, cy);`
   pick an origin that is NOT the template centre (so origin-mapping errors show up).
2. **Model.** `ModelConfig mcfg; mcfg.angle = {0,360,1}; mcfg.flip = true;`
   `MatchConfig cfg; cfg.refine = RefineMode::ROI;` `ShapeMatcher m(cfg);`
   `m.addModel("F", feat, mcfg);`
3. **Flipped scene at a KNOWN pose.** Take the template, `cv::flip(templ, sceneT, 1)`
   (horizontal mirror = the flip the matcher must recover), rotate it by a known
   `θ_gt` about a known placement, and paste onto a larger blank `scene` at known
   `(x_gt, y_gt)` (the origin's true scene location after the same transform).
4. **Match.** `auto r = m.match(scene);` take the top result.
5. **Assert (this is what currently FAILS):**
   - `r.flipped == true`
   - `hypot(r.x - x_gt, r.y - y_gt) < 1.0` px   (origin localisation)
   - `angDiff(r.angle, θ_gt_expected) < 0.5°`   (orientation)
   where `θ_gt_expected` is the reported-angle the matcher *should* produce for the
   constructed flipped pose (derive it from the same `user_angle` convention, or
   — more robustly — also build the **non-flipped** scene at the mirror-image pose,
   read its reported angle, and assert the flipped report is consistent with it).
6. **Control cases** (must keep passing): a non-flipped scene at a known pose →
   `r.flipped == false`, sub-px / sub-° pose. And the same flipped test with
   `cfg.refine = RefineMode::None` → looser tolerance (coarse), to separate "refine
   broke it" from "coarse/convention broke it".

### What good looks like

- **Before the fix:** flip control passes detection (`flipped==true`) but the ROI
  tolerance (`<1px`, `<0.5°`) FAILS — the refined pose is off because the refine
  used the base template. (The `RefineMode::None` variant may pass or fail on the
  convention alone; that isolates the refine contribution.)
- **After the fix:** flipped + ROI passes the same sub-px / sub-° tolerance as the
  non-flipped control.

### Why no visSele

Everything needed (`extractFeatures`, `ShapeMatcher`, `ModelConfig.flip`,
`RefineMode::ROI`, `MatchResult.{x,y,angle,flipped}`) is in the submodule's public
header. The test renders its own synthetic template + scene, so it is hermetic and
fast, and it pins the contract — "a flipped match returns an accurate refined
pose" — independently of any downstream measurement convention.

## Files / line references (as of this writing)

- `shape_matcher.cpp:1522` `struct ModelInfo` — add `features_flip` + `has_flip`.
- `shape_matcher.cpp:1574` `flipFeatures` — extend to `buildFlippedTemplate`
  (templ_image / refine_points / cache clear).
- `shape_matcher.cpp:1691` `addModel` (~1792 flip branch) — build + store the
  flipped FeatureSet.
- `shape_matcher.cpp:1968-2103` match loop — `fs = is_flip ? mi->features_flip :
  mi->features;` and drop the `oy=-oy` patch.
- `include/shape_matcher.h:23` `struct FeatureSet` — members enumerated in the
  table above.
