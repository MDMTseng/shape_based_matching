# ICP refine does not receive the matched scale

**Status:** read from the code, NOT yet measured. Written down so it is not
re-discovered from scratch.

**Found:** 2026-09-22, while answering "does scaling SBM features break
something".

## What

`shape_matcher.cpp` decides the scale a hit was found at:

```cpp
float matched_scale = mi->config.scale.min + scale_idx * mi->config.scale.step;   // ~2524
```

**ROI refine uses it.** Around 2645 it resizes the template image to that scale,
multiplies every sample point position by it, and scales `roi_half`:

```cpp
cv::resize(fs.templ_image, templ_use, cv::Size(), matched_scale, matched_scale, ...);
p.pos *= matched_scale;
p.roi_half = std::max(5, (int)std::lround(p.roi_half * matched_scale));
```

**ICP refine does not.** The initial pose is built with three arguments, so
`Pose2D::scale` takes its default of 1.0:

```cpp
icp_refine::Pose2D init(scene_x, scene_y, raw_angle);                 // 2586
auto refined = icp_refine::refineInverse(fs.cached_templ_scene, ...); // 2587
```

and `ICPConfig::use_scale` (Sim2, solve for scale) defaults to false and is
never set anywhere in `shape_matcher.cpp` -- grep says the only readers are
inside `icp_refine.cpp` (340, 366, 392).

`fs.cached_templ_scene` is built once from the UNSCALED `fs.templ_image`
(~1939), so on a def with a scale range, ICP aligns a full-size template
against an object found at, say, 0.8.

## Why it would be hard to notice

The pose does not come back obviously wrong -- ICP converges to a compromise,
and what degrades is `rmse` and `fitness`. Those are exactly the fields the
trust gates read, so the symptom is "this part is less trustworthy", not
"refine is broken". A recipe would look marginal rather than faulty.

## Who is exposed

Only defs that BOTH use a scale range (`scale.min != scale.max`) and select an
ICP refine mode. This bench's recipes use ROI refine, which is why nothing has
been seen.

## The fix, when it is taken

Two candidates, and the conservative one is preferred:

1. Pass it: `Pose2D init(scene_x, scene_y, raw_angle, matched_scale);` -- the
   scale was already decided by the coarse stage.
2. Let ICP solve it: `icp_cfg.use_scale = true` -- adds a degree of freedom to
   a fit that already has the answer.

## Before changing anything, measure

Take a template, scale a scene to 0.8, run ICP refine twice -- init scale 1.0
against init scale 0.8 -- and compare the returned pose error and rmse. If the
difference is in the noise, this note is the whole of the work.
