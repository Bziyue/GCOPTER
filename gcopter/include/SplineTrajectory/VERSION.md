# SplineTrajectory integration

Source: SplineTrajectory main, commit `126525e49a43b0548bc6960e4979dd6b6258f289` (2026-09-12).
The three public headers, including comments and formatting, are byte-identical to upstream.
[Optimizer protocols](https://github.com/Bziyue/SplineTrajectory/blob/126525e49a43b0548bc6960e4979dd6b6258f289/docs/optimizer.md) · [Migration guide](https://github.com/Bziyue/SplineTrajectory/blob/126525e49a43b0548bc6960e4979dd6b6258f289/docs/api_migration.md) ·
[Examples](https://github.com/Bziyue/SplineTrajectory/blob/126525e49a43b0548bc6960e4979dd6b6258f289/examples/README.md). MIT and MINVO notices are preserved alongside this file.

| Header | SHA-256 |
| --- | --- |
| `SplineTrajectory.hpp` | `9d5c3eca996a13104e13bcc668cf73e50367db5cb53faf4713f0cd1c041106dc` |
| `SplineOptimizer.hpp` | `6f4ae9c547903264b726f7bf1f28a683a02426cdf8197ee4719b04b9d541e7bf` |
| `SplineConvexHull.hpp` | `221a30e2247f6c71218d7e02e7228e802a60c820574f11f24b442da9bbb09c33` |

The optimizer owns preparation, layout, polynomial and evaluation buffers. Concrete parameterizations
borrow immutable planner maps for one solve; ordinary named objective structs bind duration,
integral and decision costs. Preparation fixes integration options and encodes the initial coordinates once; `initialGuess()`
returns a copy without repeating inverse-map solves, and callbacks reuse storage.
`backwardInto()` writes spatial gradients directly and domain penalties accumulate in those same
slices. The public polynomial and diagnostic parameter views are read-only. Independent samplers
own their traversal state and require no source warm-up.

To synchronize, copy the three original headers from the recorded upstream commit, verify their
hashes, update this record and migrate application adapters. Preserve the existing numerical-failure,
interval and convex-hull behavior; do not introduce downstream header variants.

## Planner integration

`SplineSFCOptimizer::optimize()` returns a `Polynomial` snapshot, avoiding copies of solver
factorization and adjoint scratch. The historical returned cost still excludes the mapped norm
penalty that L-BFGS minimizes. The complete callback includes that penalty through the named
`decision` stage, so finite checks and gradient propagation have one evaluation path.
Visualization samples sequentially; state publication obtains P/V/A/J in one polynomial query.
The optional gradient diagnostic uses relative tolerance `1e-6` for strongly weighted penalties.

## Validation (2026-09-12)

GCC 9.4, C++17 Release and ROS1 Noetic: `global_planning` and the mapped-gradient regression
built successfully; the mapped-gradient regression passed. The three fixed corridor perturbations
passed complete, duration, integral and energy finite differences (maximum relative error `5.49e-9`).
The upstream optimizer regressions passed 3/3, including failures, ownership and one-time initial encoding.

## Paired integration benchmark

Baseline: GCOPTER `9f999be`, using the preceding SplineTrajectory integration. Both versions use the
same [fixed corridor benchmark](../../tests/spline_solve_benchmark.cpp), compiler, flags and inputs.
AMD Ryzen 9 7945HX, one pinned logical CPU; three alternating blocks of 21 trials per perturbation.
The table gives medians in microseconds, each cell **before / after**.

| Perturbation | Prepare | Solve | Prepare + solve |
| --- | ---: | ---: | ---: |
| -1 | 20.18 / 32.06 | 356.19 / 339.57 | 383.50 / 374.00 |
| +0 | 19.00 / 32.06 | 407.25 / 395.37 | 426.24 / 425.06 |
| +1 | 19.00 / 30.87 | 358.57 / 343.13 | 376.38 / 372.81 |

Final costs and durations matched exactly at printed double precision in all paired trials.
Mapped initial validation now occurs during preparation. Initial coordinates are reused by the solver,
so compare the total column when assessing the complete operation. Total latency stayed within
roughly 0–3% of baseline; this small workload does not establish a general speedup.

From the repository root, build and run the standalone benchmark with Eigen available:

```sh
c++ -std=c++17 -O3 -DNDEBUG -fno-fast-math $(pkg-config --cflags eigen3) \
    -Igcopter/include gcopter/tests/spline_solve_benchmark.cpp -o /tmp/gcopter-spline-benchmark
GCOPTER_GRAD_CHECK=1 /tmp/gcopter-spline-benchmark 1
/tmp/gcopter-spline-benchmark 21
```

Raw logs and paired CSV files are retained in the integration workspace's
`.scratch/downstream-sync/paired-final/`; they are local artifacts, not repository attachments.
This validation covers optimization and ROS compilation; no flight or collision simulation was run.
