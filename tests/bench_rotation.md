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
  Gaussian noise (σ=20), **with and without ROI refine** (`RefineMode::None` vs `ROI`).

## Reference: aarch64 (Raspberry Pi 5 @ 2.4 GHz, 4 threads, gcc 14, `performance` governor, NEON)

### bench_multiobj — 1800 templates, σ=20 noise
| Scene | refine=none | refine=ROI | detected |
|------:|------------:|-----------:|:--------:|
| 1.2 MP (1280×960) | 54 ms (18.5 fps) | 53 ms (18.7 fps) | 4/5 |
| 5.0 MP (2592×1944) | 198 ms (5.1 fps) | 201 ms (5.0 fps) | 4/5 |
| 20.2 MP (5184×3888) | 1238 ms (0.81 fps) | 1226 ms (0.82 fps) | 4/5 |

Notes:
- **ROI refine adds ~nothing here**: at 1800 templates the coarse match dominates
  entirely (ROI refine ≈ 0.05 ms/object). Refine-mode choice does not move throughput
  at this template count — it buys accuracy, not speed.
- Cost scales ≈ linearly with pixel count (coarse match is O(scene_pixels × templates)).
- 4/5 is deterministic (one object stays just under `min_score`=50 at σ=20); it should
  be identical on every backend — use it as a correctness check, not a quality metric.

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
