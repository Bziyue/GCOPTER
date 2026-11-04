# Native spline experiments

Research branch based on `ros2@15c70cd`. Production generation and ROS defaults
remain H5-S. The report, figures, full measurement sequences and compressed
geometry/trajectory coefficients are in
[`原生三次走廊性能优化实验.md`](../../../docs/原生三次走廊性能优化实验.md).
The [first-round report](../../../docs/原生三次与最小jerk走廊实验.md) retains the
original comparison and the broader research plan.

Implemented: exact nonuniform cubic extraction, degree-aware affine-disc
geometry, natural cubic and free-endpoint minimum-jerk diagnostics, continuous
sphere/support and full-disk Jacobian certificates, native and H5-projected
cubic initialization, five-variable control-point optimization, band Cholesky,
natural-endpoint projection ablation, and common-backend trajectory replay.

The current native path uses clearance-based initialization, grouped 9D
support Hessians, banded assembly, analytical feasible steps, and proactive
progress cones. It calls neither H5 nor FIRI. `generateNative(seed, index,
bounds, options)` selects `NativePreset::Balanced`; `Fast` and `Volume` select
the measured alternatives. `generateCubic()` exposes individual research
settings and the optional H5 projection reference.

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
  --output .scratch/native_splines/round2/balanced
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

The CSV v3 header begins `degree,u0,u1` followed by nine world-space geometric
channels. There are exactly four or six controls per interval. The legacy v2
production interface remains unchanged; `GeometryAccess` supplies an isolated
exact-elevation adapter for this experiment.
