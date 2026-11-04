# Native C3 corridor generation

This module generates a continuously certified C² corridor directly from a seed
path. The RViz2 demo uses `generateNative()` with `NativePreset::Balanced`; see
[interactive commands](../../../docs/C3交互可视化.md). H5 remains available through
its separate `generate()` entry and as a projection quality reference.

Each cubic span represents `F(u,z) = c(u) + M(u)z`, with `||z|| <= 1`. Nonuniform
B-spline control variables are extracted exactly into cubic Bézier coefficients.
The output contains degree-three geometry and its original support planes.
Every accepted update satisfies continuous support and full-section Jacobian
certificates. Each volume-refinement update follows the same acceptance rule.

## Implementation

| File | Responsibility |
| --- | --- |
| `cubic_basis.hpp`, `geometry.*` | Exact extraction, geometry, degree-aware import/export |
| `cubic_generator.*` | Input validation, initialization, inflation and certified acceptance |
| `obstacle_cover.hpp`, `lazy_obstacles.hpp` | Cached local-box layout, BVH queries and dynamic separation |
| `cubic_optimizer.*` | Shared immutable geometric constraints, progress cones, grouped Newton kernel |
| `support_pruning.cpp` | Projected candidate selection with a sufficient 3D implication proof |
| `cubic_refinement.cpp` | Cached exact line model and certified swept-volume refinement |
| `cubic_diagnostics.cpp` | Independent dense derivatives and finite-difference checks |
| `runtime.hpp` | Optional stage profiling and generation deadline |
| `cubic_query.hpp` | Native cubic evaluation and section inversion |

A fixed knot/model layout owns one coverage layout and one bending-energy
matrix. Initialization, the first solve and progress repairs share immutable
geometric groups. Progress cones remain local to each problem. New separation
planes in a later inflation create new groups; knot refinement creates a new
layout. Obstacle separation always uses the current geometry.

Geometric constraints are stored once as groups. `Cut` only stores progress
cones. There is no second expanded copy of every geometric row. Projected
pruning witnesses are materialized only by explicit proof tests.

The solver uses exact grouped 9D Hessians, axis-aligned kernels, banded Cholesky,
and a scalar quadratic line model. The sufficient Armijo bound falls back to
an exact logarithmic cost difference. `Balanced` also uses proved support
pruning, barrier-path prediction and one certified volume refinement. `Fast`
and `Volume` retain their own energy weights and iteration budgets.

## Inputs, certificates and timing

`generateCubic()` checks external seed/options/bounds once. Internal routines
consume valid finite data and their established layout. Domain tests for log
barriers, factorization pivots, nondegenerate separator metrics and continuous
geometric acceptance belong to the numerical algorithm and remain in place.
The final exported state has already passed the acceptance certificate; it is
not recertified a second time merely for export.

Normal generation records total time. `OptimizerSettings::profile=true`
(`NATIVE_PROFILE=1` in the benchmark) enables detailed stage timers. With it off,
stage fields are zero/unmeasured, not estimates of stage cost. Deadline checks
occur at generation, span, subdivision and Newton boundaries. Inner obstacle
loops read the clock once per 1024 selected obstacles, rather than per 32.

`check_gradients` is an explicit diagnostic, enabled on untimed benchmark
warmups. The independent dense implementation is separated from the active
Newton kernel. `CubicQuery` is constructed once for certified geometry; that
geometry must outlive the query object.

## Build and compare

Builds and measurements are serial. The comparison runner limits address space
to 4 GiB, uses one numerical thread and pins the benchmark to CPU 2 by default.

```bash
cd /home/zdp/ForCodex/GCOPTER
bash -c 'ulimit -v 4194304
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cmake -S direct_corridor/experiments/native_splines \
  -B .scratch/native_splines/build -DCMAKE_BUILD_TYPE=Release
cmake --build .scratch/native_splines/build -j1
ctest --test-dir .scratch/native_splines/build --output-on-failure -j1'
```

`run.py` compares native C3, the H5 projection reference and optional H5
representation diagnostics. `run_cleanup.py` compares the current native path
against the frozen E3 binary at
`.scratch/native_splines/cleanup_baseline/native_spline_benchmark`:

```bash
python3 direct_corridor/experiments/native_splines/run_cleanup.py \
  --output .scratch/native_splines/cleanup/reproduce \
  --cases 0,1,2,3,4,5,6 --blocks 3 --repeats 7
```

The frozen baseline is commit `01c8a5b`, built in Release without
`-march=native`. Inputs and hashes are from
`docs/validation/native_splines_e1/inputs.tar.gz`. The runner verifies hashes at
entry and keeps every timing, failure, coefficient export and fresh-map audit.
Use `--configurations clean,profiled` to measure profiling overhead separately.

The optional trajectory replay backend still consumes native geometry, but is
not part of this generation-only comparison. Build it with
`-DNATIVE_SPLINE_TRAJECTORY=ON` when needed.

## Research history

The [cleanup report](../../../docs/原生三次走廊代码清理与热路径精简.md) records the active
architecture and its regression results. The [E3 report](../../../docs/原生三次走廊E3构造缓存与覆盖复用实验.md)
records construction/coverage reuse before cleanup.

E1 exact reach, E2 selective certificate refinement, working sets, point
exchange, stale separator reuse, adaptive-knot heuristics, frame recentering,
coordinate inflation, alternative barriers, weak Hessians, delayed Hessian
refresh and the older assembly/line-search variants were removed from active
code after the experiments. Construction caches and exact kernels now have one
implementation, without ablation toggles.

Historical reports, measurements and source snapshots remain in `docs/validation`.
The old drivers and configurations belong to those snapshots, not the current
API. Recover them from their recorded commit (E1 `5c6f860`, E2 `3a34a75`,
E3 `01c8a5b`) or the corresponding `source_snapshot.tar.gz`. This preserves the
experiments without carrying inactive alternatives through the generator.
