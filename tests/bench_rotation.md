# Cross-platform match-throughput benches

Two benches for comparing SIMD backends (x86 **AVX2** vs ARM **NEON** vs scalar).
Both self-label their backend in the first output line, and use fixed synthetic
objects/scenes so match **counts and scores are bit-identical across backends** —
only timing differs. If your scores differ from the reference below, that's a
correctness bug, not a perf trade-off.

## Build & run

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release .      # Linux/macOS; on Windows use your OpenCV_DIR
cmake --build build --target bench_rotation --target bench_multiobj -j

./build/bench_rotation      # coarse-match throughput vs template count
./build/bench_multiobj      # 5 objects x 360deg = 1800 templates, 1.2/5/20 MP, ROI on/off
```

- `bench_rotation` — one object, {1, 72, 360} rotation variants, 640×480 & 1280×960,
  plus a per-stage profile (GaussianBlur / Sobel+quantize / fused spread+LUT+linearize /
  coarse similarity). Isolates where the coarse pipeline spends time.
- `tune_sweep` — **parameter auto-tuner**. Generates test scenes by *altering the
  template* (rotate to known off-grid angles + Gaussian noise → known pose), sweeps
  `num_features × match_scale × scaled_blur_ksize × min_score`, and ranks every combo by
  **robust detection** (detection rate, then worst-case score, then speed). Prints the full
  table and the best config. Prototype of an eventual `sbm::autoTune()`. Run:
  `./tune_sweep [shape_index | template.png]`. Example finds: a solid shape's best is a
  fast downscale (`match_scale 0.7 + blur`, 100% detect, ~8 ms), while a thin wireframe
  shape's best is full res (downscale drops it) — the tuner picks per-shape automatically.
- `bench_multiobj` — the realistic case: **5 distinct objects, each swept 0–360° at 1°
  (1800 template variants)**, matched against **1.2 / 5 / 20 MP** scenes with additive
  Gaussian noise (σ=10). Sweeps refine/scale: coarse-only at full res, then ROI refine
  at scene downscale `match_scale` ∈ {1.0, 0.7, 0.5}. Downscale shrinks the coarse-match
  cost (~scale²); ROI refine runs at full res to recover accuracy.

## Reference: aarch64 (Raspberry Pi 5 @ 2.4 GHz, 4 threads, gcc 14, `performance` governor, NEON)

### bench_multiobj — 1800 templates, σ=10 noise (match avg · fps · detected · weakest score)
`minscore` = score of the weakest **detected** object (headroom above `min_score`=50).
`crd` = coordinate-scale full-res features (`addModel(FeatureSet)`); `rex` = re-extract at
match_scale (`addModel(image)`), which keeps score under downscale.

| Scene | none s=1.0 | 0.7 crd | 0.7 **rex** | 0.5 crd | 0.5 **rex** |
|------:|----------:|--------:|------------:|--------:|------------:|
| 1.2 MP | 49 · 21 · 5/5 · **95** | 25 · 40 · 4/5 · 62 | 29 · 34 · 4/5 · **94** | 25 · 39 · 4/5 · 57 | 19 · 52 · 4/5 · **79** |
| 5.0 MP | 198 · 5.0 · 5/5 · **97** | 82 · 12 · 4/5 · 62 | 80 · 12 · 4/5 · **93** | 48 · 21 · 4/5 · 72 | 37 · 27 · 4/5 · **78** |
| 20.2 MP | 1234 · 0.8 · 5/5 · **95** | 403 · 2.5 · 5/5 · 52 | 316 · 3.2 · 5/5 · **54** | 202 · 5.0 · 4/5 · 52 | 115 · 8.7 · 4/5 · **79** |

*(cells: match avg ms · fps · detected · minscore)*

Notes:
- **Full-res finds all 5 at high score** (≈95–99.7, zero-noise or σ=10); the `minscore`
  column shows ~45 points of headroom over `min_score`=50. If your AVX2 run shows 5/5 at
  s=1.0 with matching `minscore`, the backends agree.
- **`minscore` quantifies the downscale recall margin.** Downscaling collapses headroom
  toward the threshold: 20 MP s=0.7 = 51.7 (right on the edge, still 5/5); push further and
  the weakest object drops under 50 and is pruned (4/5). It's the direct signal for how
  much `match_scale` you can afford before losing a detection.
- **ROI refine at full res adds ~nothing** (≈0.05 ms/object): at 1800 templates the coarse
  match dominates. Refine buys accuracy, not speed — *unless* paired with downscale:
- **Downscale + ROI refine is the win at high resolution.** `match_scale` shrinks the
  coarse-match cost ~scale²; ROI refine runs at full res to recover localization. 20 MP
  goes 0.82 → **5.7 fps (≈7×)** at s=0.5.
- **Downscale has a recall cost.** `min_score` is *also the coarse-pyramid prune threshold*
  — a candidate below it at the coarse T=8 level is dropped before fine matching. On a
  0.5-downscaled scene the coarse score of a marginal object dips under 50, so it's pruned.
  Full res keeps all 5. Lower `min_score` to keep marginal objects (but that refines more
  candidates → slower), or use `rex`:
- **`rex` (re-extract at scale) fixes the downscale score drop.** The `addModel(image, …)`
  overload re-extracts the model from a template resized by `match_scale`, so its features
  are selected/oriented at the resolution the resized scene is matched at — instead of
  crowding full-res features onto the coarse grid. It restores most of the lost `minscore`
  (0.7: 62→94; 0.5: 57→79) **and is often faster** (fewer, cleaner features → less coarse
  work; 20 MP s=0.5: 202→115 ms, 5.0→8.7 fps). Prefer `addModel(image)` whenever you use
  `match_scale < 1`.
- **Thin shapes: add `scaled_blur_ksize`.** For wireframe/sparse templates the optional
  pre-blur arg on `addModel(image, …, scaled_blur_ksize=3)` stabilises orientation
  quantization before the downscale re-extraction (star s=0.5: 94→97). It slightly hurts
  solid shapes, so it's off by default — enable per-model for thin ones only.
- **Below ~0.7 stops helping at small scenes**: 1.2 MP floors ~24 ms (s=0.7 ≈ s=0.5) —
  there the 1800-template fixed cost, not scene pixels, dominates. At 5/20 MP the scene
  still dominates, so 0.5 keeps paying (≈4× / 7×).
- `nmatch` is deterministic per (scale, backend) — use it as a cross-backend correctness
  check; it must match this reference on AVX2.

### bench_rotation — single object, coarse path @ 1280×960
| Templates | NEON match avg |
|----------:|---------------:|
| 1 (upright) | ~12 ms |
| 72 (5° sweep) | ~12 ms |
| 360 (1° sweep) | ~14 ms |

Per-stage @ 1280×960 / 360 templates (NEON): fused spread+LUT+linearize ≈ 5–6 ms,
GaussianBlur ≈ 3 ms, Sobel+quantize ≈ 3 ms, coarse similarity ≈ 3–4 ms (bandwidth-bound
across 4 threads at high template counts).

### NEON port history (this branch, `ct/dev`)
Same aarch64 box, 360-template sweep @ 1280×960, `bench_rotation`:

| Build | match avg | fps |
|------:|----------:|----:|
| scalar (pre-NEON) | 21.4 ms | 47 |
| + fused-loop NEON | 14.0 ms | 72 |
| + similarity NEON  | 12.7 ms | 79 |

## x86 team: please append your AVX2 numbers

Run both benches on your x86 box (build picks up AVX2 via `-march=native -mavx2`, or
`/arch:AVX2` on MSVC — the output header should say `SIMD backend: AVX2`) and drop the
tables here so we can compare AVX2 vs NEON on identical scenarios. Confirm `nmatch` and
scores match the reference (they must, if both backends are correct).
```
# paste bench_multiobj + bench_rotation output here
```
