# SplineTrajectory integration

Source: SplineTrajectory main, commit `d5041355f08e28a9a62d088388f27af147d1a16c` (2026-09-12).
The three public headers are byte-identical to that source. See [protocols](https://github.com/Bziyue/SplineTrajectory/blob/d5041355f08e28a9a62d088388f27af147d1a16c/include/SplineOptimizerProtocols.md),
[usage](https://github.com/Bziyue/SplineTrajectory/blob/d5041355f08e28a9a62d088388f27af147d1a16c/README.md) and [measured validation](https://github.com/Bziyue/SplineTrajectory/blob/d5041355f08e28a9a62d088388f27af147d1a16c/docs/refactor_validation.md).
MIT license and MINVO notices are preserved alongside this file.

| Header | SHA-256 |
| --- | --- |
| `SplineTrajectory.hpp` | `7038992eadbc72f0a821ea849a1e16a0397606416c22f33a254becfe42885364` |
| `SplineOptimizer.hpp` | `e476a73f313a5a219dc06d451a0c3e622dec0e034eb5fae3350ed53f26573af9` |
| `SplineConvexHull.hpp` | `84ba9dbc060b8d24ff4c1803a0c8c5d8201e497f4871ef4e7e658aa62efcb47c` |

Configuration is supplied to `prepareContext`; structured sample callbacks declare their required derivative order.
Coefficient costs bind statically. The internal quadrature call chain uses a local GCC/Clang flatten annotation
because GCC 9 otherwise loses flatness inlining. Other compilers retain the ordinary C++ path.
Independent samplers borrow immutable polynomials; rebuild them after source mutation or move. No shared prewarming is needed.
`backwardGradInto` writes preallocated spatial gradients; domain penalties add to existing gradients.
The regression reproduces the previous domain-penalty overwrite and checks complete mapped finite differences.

To synchronize, copy the three source headers and license notices from the recorded upstream commit,
update these hashes, migrate adapters if protocols changed, and run the mapped-gradient and planner regressions.
Do not restore an older snapshot over the merged numerical failure, interval or convex-hull functionality.

Validation: Ubuntu 20.04 / GCC 9.4 / Noetic Release global_planning builds, `gcopter_spline_map_regression` passes,
and `tests/spline_solve_benchmark.cpp` completes three fixed corridors. Medians improve by about 6%--7% against the old integration.
The benchmark also compiles against the historical MINCO implementation; see the upstream report for its const-comparator compatibility include.
