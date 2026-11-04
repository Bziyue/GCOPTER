# Native spline experiments

Research branch based on `ros2@15c70cd`. The RViz2 interactive demo now defaults
to native C3; see [interactive commands](../../../docs/C3交互可视化.md).
The production `generate()` library entry remains H5-S. The report, figures, full measurement sequences and compressed
geometry/trajectory coefficients are in
[`原生三次走廊冗余约束与Newton内核优化.md`](../../../docs/原生三次走廊冗余约束与Newton内核优化.md).
This fifth round selects configurations by corridor generation time, union
volume, geometric regularity and continuous safety. Trajectory optimization
is retained as historical data and is not a selection criterion.
The [next-stage improvement plan](../../../docs/原生三次走廊下一阶段改进方案.md)
reconciles the independent review with the current implementation. It prioritizes
tighter redundancy bounds, selective certificate replacement, incremental setup
and smoothness control, with staged cubic generation as a separate experiment.
The [E1 exact-reach experiment](../../../docs/原生三次走廊E1精确可达域实验.md)
implements and compares the tighter domain bound on 67 cases across 16 maps.
Use `OptimizerSettings::reach_bound=1` for the exact bound or `2` to try it only
after the original triangle bound fails (`NATIVE_REACH_BOUND` in the benchmark).
Both remain opt-in: extra pruning did not consistently improve generation time,
union volume and boundary smoothness together. The default remains `0`.

The [E2 certificate-replacement experiment](../../../docs/原生三次走廊E2选择性证书替换实验.md)
uses `OptimizerSettings::certificate_refinement=1` for exact degree-five
certificate elevation, or `2` for one de Casteljau bisection. Geometry and queries
remain cubic. `certificate_fraction` selects a fraction of interval groups using
barrier pressure times the support-bound gap, divided by added row count. Only
planes with a tighter bound are replaced. Selection starts in the second
inflation pass and does not add Newton iterations. The benchmark equivalents are
`NATIVE_CERTIFICATE_REFINEMENT` and `NATIVE_CERTIFICATE_FRACTION`.

Both E2 modes default to off. The solver rebuilds implication proofs for the
replacement rows, while the final certificate still covers every original
plane and the full-section Jacobian. `run_e2.py --blocks 3 --repeats 7` rotates
configuration order across three serial blocks and retains all 21 timings,
CPU time, coefficient hashes and certificate audits. It reuses the E1 input
manifest and enforces a 4 GiB address-space limit and one numerical thread.

The [E3 construction/coverage experiment](../../../docs/原生三次走廊E3构造缓存与覆盖复用实验.md)
keeps cubic coefficients, original support planes and Newton decisions exactly
unchanged. Across 67 frozen inputs, the sum of per-case generation medians falls
by 6.75% (6.83% for the 65 successful cases). All three native presets now enable
`reuse_construction`, `fast_construction` and `cache_coverage`. These reuse the
initial feasible problem and geometric rows during progress repairs, cache
interval extraction/group lookups, and retain the static local-box layout.
Dynamic obstacle separation still runs for the current geometry each pass.
The benchmark switches are `NATIVE_REUSE_CONSTRUCTION`,
`NATIVE_FAST_CONSTRUCTION` and `NATIVE_CACHE_COVERAGE`; setting all three to `0`
selects the instrumented legacy implementation. `run_e3.py` compares it with the
frozen `3a34a75` binary, individual ablations and the combined implementation,
serially under the same 4 GiB limit. `setup_ms` now includes initialization's
feasibility-problem construction; `initial_problem_ms` exposes that component.

The fourth-round report is
[`原生三次走廊连续约束与自适应优化实验.md`](../../../docs/原生三次走廊连续约束与自适应优化实验.md).
The [third-round report](../../../docs/原生三次走廊约束与求解加速.md) covers
shared supports, block assembly and scalar line searches.
The [second-round report](../../../docs/原生三次走廊性能优化实验.md) records
initialization, volume recovery and the native query implementation.
The [first-round report](../../../docs/原生三次与最小jerk走廊实验.md) retains the
original comparison and the broader research plan.

Implemented: exact nonuniform cubic extraction, degree-aware affine-disc
geometry, natural cubic and free-endpoint minimum-jerk diagnostics, continuous
sphere/support and full-disk Jacobian certificates, native and H5-projected
cubic initialization, five-variable control-point optimization, band Cholesky,
natural-endpoint projection ablation, and common-backend trajectory replay.

The current native path uses clearance-based initialization, grouped 9D
support Hessians, shared-support merging, occupied 5×5 band blocks, cached
quadratic line searches, shared fixed energy matrices,
cached construction and coverage layouts, lazy obstacle metric priorities, and
quadratic-derivative progress cones. It calls
neither H5 nor FIRI. `generateNative(seed, index,
bounds, options)` selects `NativePreset::Balanced`; `Fast` and `Volume` select
the measured alternatives. `generateCubic()` exposes individual research
settings and the optional H5 projection reference.

`Balanced` now also uses proved support pruning in both Hessian assembly and
step-boundary scans, axis-aligned plane kernels, derivative-only Newton assembly,
and a rational sufficient Armijo test with an exact logarithm fallback. It takes
8/3/3 Newton steps per inflation, uses the preceding factor for barrier-path
predictions, and finishes with one certified swept-volume refinement.
The other presets retain their previous solver budgets. Their exact kernels
also omit unused values and use the sufficient line-search test.

`CubicQuery` consumes the returned degree-three `Geometry` directly. It uses
cubic Horner evaluation, an analytical 2×2 Gram inverse, and exact-key caches.
Construct it once for certified geometry and reuse it for the trajectory
optimization. Geometry must outlive its query object.

The seven frozen cases are a development experiment. Q5-C joint optimization,
derivative elimination, native initialization with natural endpoints, full
parameter/equal-variable scans and held-out scene evaluation are not yet
implemented. `Q5-T` and `H5-C3-T` currently mean global jet-fraction controls;
`H5-Z-fixed` means zeroing shape jets at the current H5-S nodal values.

```bash
cd /home/zdp/ForCodex/GCOPTER
tar -xzf docs/validation/native_splines_optimized/frozen_inputs.tar.gz
cmake -S direct_corridor/experiments/native_splines \
  -B .scratch/native_splines/build -DCMAKE_BUILD_TYPE=Release
cmake --build .scratch/native_splines/build -j1
ctest --test-dir .scratch/native_splines/build --output-on-failure -j1
python3 direct_corridor/experiments/native_splines/run.py --native \
  --generation-only --method-count 1 --preset balanced --repeats 7 \
  --output .scratch/native_splines/round3/reproduce
```

For trajectory replay, reconfigure with `-DNATIVE_SPLINE_TRAJECTORY=ON` and the
installed OMPL prefix if needed, build `native_trajectory_comparison` serially,
then run `trajectory.py`. It verifies frozen geometry against the map and
uses the existing optimizer and identical initialization. Cubics use the
native query by default; `--quintic-replay` selects the original evaluator for
the same geometry. Exact elevation remains an independent audit adapter.
Primary timing has profiling off; `--profile` enables a separate diagnostic run.
The replay's `corridor_ms` is reading/audit/adaptation time, not generation time.

`-DNATIVE_SPLINE_NATIVE_ARCH=ON` enables the local CPU instruction set for all
benchmark arms, including H5. It does not enable fast-math; compare its timings
separately from portable builds.

`compare.py` archives measurements and draws figures using NumPy/Matplotlib.
`independent_audit.py` uses NumPy and SciPy for an additional nearest-sphere
sampling audit. The continuous certificates remain the output acceptance test.
`native_volume_refine` increases QMC samples without regenerating a corridor.
`analyze_optimized.py` archives the second-round measurements, exploratory
settings, and shared-domain QMC results, and renders the report figures.
`explore.py` runs explicit serial third-round ablations; `analyze_round3.py`
archives their outputs and the matched final measurements.
`explore_round4.py` screens continuation, adaptive knots and exact swept-volume
gradients under a 4 GiB address-space limit. `analyze_round4.py` archives the
fourth-round measurements and all rejected alternatives. These alternatives
remain opt-in: `NATIVE_SEPARATOR_REUSE_GAP`, `NATIVE_SUPPORT_EXCHANGE`,
`NATIVE_ADAPTIVE_KNOTS`, `NATIVE_PATH_PREDICTOR`, `NATIVE_VOLUME_REFINEMENTS`.
`NATIVE_CACHE_VALUES=1` enables the tested same-point cone cache; it is disabled
by default because the measured implementation was slower than recomputing the dot products.
The continuous oracle supports degree-six support slacks and degree-four fixed
normal progress slacks; only support-point exchange is connected to generation.
Exact knot insertion is tested independently; the current adaptive experiment
selects an initial nonuniform knot vector rather than performing online insertion.

Fifth-round generation experiments use `explore_round5.py`. It explicitly
freezes the old solver settings before applying each configuration, alternates
configuration order by case, uses CPU 2 and a 4 GiB address-space limit, and
records binary/source/input hashes. The `frozen` arm expects a separately built
`f72f3fe` benchmark at `.scratch/native_splines/round5_baseline/native_spline_benchmark`.
`pruned_predict3_volume1` is the new Balanced configuration; `projected_axis_bounded`
isolates pruning plus exact kernels at the old Newton budget.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 direct_corridor/experiments/native_splines/explore_round5.py \
  --configurations frozen,projected_axis_bounded,pruned_predict3_volume1 \
  --repeats 21 --output .scratch/native_splines/reproduce-round5
```

`evaluate_corridors_round5.py` checks the archived screening geometries with a
shared-domain QMC sequence and an independent geometry audit. It never runs a
trajectory optimizer. `evaluate_round5.py` retains the earlier comparison and
requires an explicit `--trajectories` to replay trajectories. `analyze_round5.py`
archives all screening/final data, verifies that the finally timed coefficients
match the measured geometries, and plots time, volume and curvature together.
The selected configuration improves time and union volume on the seven frozen
development cases; local curvature and shape bending can increase.

Experimental alternatives remain explicit settings: `projected_pruning=1`
prunes only assembly, `distance_barrier` uses the regularized support-distance
barrier, `initial_sweeps` inflates the initial shapes, `hessian_interval` reuses
a factor, and `weak_curvature_ratio` omits weak shape/cross curvature while
keeping the full gradient. These alternatives do not bypass final certificates.

The CSV v3 header begins `degree,u0,u1` followed by nine world-space geometric
channels. There are exactly four or six controls per interval. The legacy v2
production interface remains unchanged; `GeometryAccess` supplies an isolated
exact-elevation adapter for this experiment.
