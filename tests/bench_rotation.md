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
- `bench_multiobj` — the realistic case: **5 distinct objects, each swept 0–360° at 1°
  (1800 template variants)**, matched against **1.2 / 5 / 20 MP** scenes with additive
  Gaussian noise (σ=10). Sweeps refine/scale: coarse-only at full res, then ROI refine
  at scene downscale `match_scale` ∈ {1.0, 0.7, 0.5}. Downscale shrinks the coarse-match
  cost (~scale²); ROI refine runs at full res to recover accuracy.

## Reference: aarch64 (Raspberry Pi 5 @ 2.4 GHz, 4 threads, gcc 14, `performance` governor, NEON)

### bench_multiobj — 1800 templates, σ=10 noise (match avg, fps)
| Scene | none s=1.0 | ROI s=1.0 | ROI s=0.7 | ROI s=0.5 |
|------:|----------:|----------:|----------:|----------:|
| 1.2 MP (1280×960) | 47 ms · 21.1 | 49 ms · 20.2 | **25 ms · 40.0** | 26 ms · 38.5 |
| 5.0 MP (2592×1944) | 194 ms · 5.2 | 192 ms · 5.2 | 80 ms · 12.5 | **48 ms · 20.7** |
| 20.2 MP (5184×3888) | 1206 ms · 0.83 | 1219 ms · 0.82 | 394 ms · 2.5 | **180 ms · 5.5** |

Notes:
- **ROI refine at full res adds ~nothing** (≈0.05 ms/object): at 1800 templates the coarse
  match dominates. Refine buys accuracy, not speed — *unless* paired with downscale:
- **Downscale + ROI refine is the win at high resolution.** `match_scale` shrinks the
  coarse-match cost ~scale²; ROI refine at full res recovers localization. 20 MP goes
  0.83 → **5.5 fps (6.7×)** at s=0.5, detection maintained.
- **Below ~0.7 stops helping at small scenes**: 1.2 MP floors at ~25 ms (s=0.7 ≈ s=0.5) —
  there the 1800-template fixed/per-template cost, not scene pixels, is the bottleneck.
  At 5/20 MP the scene still dominates, so 0.5 keeps paying (4× / 6.7×).
- `nmatch` (4–5 of 5) is deterministic per (scale, backend) — one object hovers near
  `min_score`=50 under noise. Use it as a cross-backend correctness check, not a quality
  metric; it must match the reference on AVX2.

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
