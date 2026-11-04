# SweepCorridor

A C++17 / Eigen header-only library for continuously certified affine-disc
sweep corridors:

`F(u,z) = c(u) + M(u) z`, with `||z||₂ <= 1`.

`Generator<3>` uses a native cubic C² B-spline kernel. `Generator<5>` uses the
quintic Hermite smooth kernel. Both return `Corridor<Degree>` with identical
section and differential interfaces. Only degrees 3 and 5 are supported.

## Use

```cpp
#include <SweepCorridor.hpp>
namespace sc = SweepCorridor;
sc::Options options;
sc::ObstacleIndex map(obstacles); // Owned immutable sphere BVH; reusable.
auto result = sc::Generator<3>(options).generate(seed, map, bounds);
if (result) {
    sc::Query<3> query(result.corridor);
    auto section = query.section(0.5);
    auto first = query.evaluateFirst(0.5, position); // Value and gradient only.
    auto differential = query.evaluate(0.5, position); // Includes Hessian.
}
```

Change the template argument to `5` for H5. There are no runtime algorithm
presets. Geometry, safety margins, resource limits and optional stage profiling
are configurable; optimizer constants and continuous safety checks are fixed.

`Corridor` owns knots and degree-specific Bézier controls. Each control is nine
scalars: center followed by the column-major 3x2 shape matrix. `section()` returns
center and shape plus their first and second derivatives in normalized u.
`Query` precomputes power coefficients and uses Horner evaluation with a span
cursor; it owns its cache and can outlive the source corridor. Each evaluation
thread uses its own Query. Corridor itself is immutable and shareable.

`evaluate(u,x)` returns `q = ||M⁺(x-c)||² - 1`, gradient in `[u,x,y,z]`, Hessian
and signed section-plane residual. Membership in that section requires both
`q <= 0` and zero plane residual. This is not a signed distance function or a
global inverse parameterization. For distant overlapping spans, u is not
necessarily unique. Query inputs must be finite, u in [0,1], and the corridor
nonempty; generation and map construction validate external data once.

For trajectory optimization, include `<SweepCorridor/InverseQuery.hpp>`:

```cpp
sc::InverseQuery<3> inverse(result.corridor);
auto residual = inverse(position, progress_hint);
// residual.radial, residual.plane: residuals in metres
// residual.grad_radial, residual.grad_plane: world-position gradients
```

`InverseQuery<Degree>` uses the same first-order implementation for both degrees.
It solves the section-plane equation from the supplied hint and differentiates
through the root. Outside the ellipse it returns a radial violation; inside it
returns zero. At endpoint extensions it retains the signed cap-plane residual.
`valid` reports whether a root or endpoint was obtained. This is a local inverse,
not a global nearest-point search across overlapping corridor sections.

Exact-key bounded caches reuse section geometry, Gram inverses, dual maps and
complete answers. Cache collisions evict entries without changing root selection;
roots are never warm-started from previous queries. The query owns its polynomial
coefficients and caches; use one instance per thread. `reset()` clears the caches.
The kernel contains no clocks or profiling counters. `first(u)` and
`evaluateFirst(u,x)` skip second derivatives and Hessian work entirely.

Obstacles are closed spheres. A voxel of edge a can be represented by its
circumsphere of radius `sqrt(3)*a/2`. Bounds and obstacles are inflated using the
same robot and safety margins. Successful generation includes continuous
support and full-section Jacobian certification. An unsuccessful result has no
usable corridor. `certify()` is an explicit certificate recheck, outside normal
generation timing. Local Jacobian positivity does not imply global injectivity.

## Integration

Only Eigen >= 3.3 is required. No ROS, GCOPTER, OMPL, Boost, filesystem, environment
variables or trajectory optimizer are needed by the headers.

Copy `include/` and add it and Eigen to your include paths, or use:

```cmake
add_subdirectory(path/to/SweepCorridor)
target_link_libraries(your_target PRIVATE SweepCorridor::SweepCorridor)
```

Installation supports `find_package(SweepCorridor REQUIRED)` with the same target.
The GCOPTER repository's `direct_corridor` folder is now an adapter only; it is
not needed when copying this library to another repository.

## Build and test

```bash
cmake -S SweepCorridor -B build/sweep -DCMAKE_BUILD_TYPE=Release
cmake --build build/sweep -j1
ctest --test-dir build/sweep --output-on-failure -j1
cmake --install build/sweep --prefix /your/prefix
```

Tests cover both generators, continuous certificates, identical query interfaces,
Horner derivatives, constraint gradients/Hessians and multiple translation units.
The `tests/consumer` project tests consumption of an installed package. Frozen
input benchmarks are opt-in via `SWEEP_BUILD_BENCHMARKS=ON`; their file and report
I/O lives outside the library headers.

The implementation separates shared geometry/certificates from the C3 banded
solver and H5 node solver. `detail/` is private implementation, not a supported
configuration or extension API. Historical projection, Minkowski, jet-fitting
and tuning experiments remain available in GCOPTER commit `92da988` and archived
reports, not in this portable package.
