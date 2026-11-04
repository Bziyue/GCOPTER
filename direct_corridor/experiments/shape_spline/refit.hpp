// SPDX-License-Identifier: MIT
#pragma once
#include "gcopter/direct_corridor.hpp"
#include <ostream>
namespace gcopter::direct::experiment {
struct ShapeFitMetrics {
  double total_ms = 0, planes_ms = 0, setup_ms = 0, optimize_ms = 0, verify_ms = 0;
  double initial_scale = 0, fit_error = 0, max_support_residual = 0;
  double c2_error = 0, shape_acceleration_energy = 0, baseline_acceleration_energy = 0;
  double max_knot_first = 0, max_knot_second = 0, gradient_error = 0, hessian_error = 0;
  int degree = 0, controls = 0, constraints = 0, iterations = 0, regularity_backtracks = 0;
  int reused_planes = 0;
  bool direct_initialization = false;
};
struct ShapeFitResult {
  Corridor corridor;
  ShapeFitMetrics metrics;
};
// Controlled shape-only experiment. Reuses the baseline center and builds a
// continuous separating-plane cover before optimizing a C2 matrix spline.
// Degree 5 uses interior knot multiplicity 3 (free Hermite derivative DOFs),
// degree 3 uses simple interior knots. Output is elevated to the common degree 5 API.
ShapeFitResult refitShape(const Corridor& baseline, const std::vector<Vec3>& seed,
                          const ObstacleIndex& map, const Bounds& bounds, const Options& options,
                          int degree, bool check_derivatives = false);
// Runs the same center inflation pipeline, then consumes its accepted cover
// directly. Input validation and the generation deadline belong to that pipeline.
Result generateIntegratedShape(const std::vector<Vec3>& seed, const ObstacleIndex& map,
                               const Bounds& bounds, const Options& options,
                               ShapeFitMetrics& metrics, bool check_derivatives = false);
Result generateFastShape(const std::vector<Vec3>& seed, const ObstacleIndex& map,
                         const Bounds& bounds, const Options& options,
                         ShapeFitMetrics& metrics, int inflation_rounds = 1,
                         bool check_derivatives = false);
Result generateCoordinateShape(const std::vector<Vec3>& seed, const ObstacleIndex& map,
                               const Bounds& bounds, const Options& options,
                               ShapeFitMetrics& metrics, int warmup_limit = 8,
                               bool check_derivatives = false,
                               bool fraction_to_boundary = false);
inline void writeShapeMetrics(std::ostream& s, const ShapeFitMetrics& m){
  s << ",\"shape_refit_ms\":" << m.total_ms << ",\"shape_planes_ms\":" << m.planes_ms
    << ",\"shape_setup_ms\":" << m.setup_ms << ",\"shape_optimize_ms\":" << m.optimize_ms
    << ",\"shape_verify_ms\":" << m.verify_ms << ",\"shape_degree\":" << m.degree
    << ",\"shape_controls\":" << m.controls << ",\"shape_constraints\":" << m.constraints
    << ",\"shape_newton_steps\":" << m.iterations << ",\"shape_initial_scale\":" << m.initial_scale
    << ",\"shape_projection_control_error\":" << m.fit_error << ",\"shape_support_residual\":" <<
    m.max_support_residual
    << ",\"shape_c2_error\":" << m.c2_error << ",\"shape_acceleration_energy\":" <<
    m.shape_acceleration_energy
    << ",\"baseline_shape_acceleration_energy\":" << m.baseline_acceleration_energy
    << ",\"max_shape_knot_first\":" << m.max_knot_first << ",\"max_shape_knot_second\":" <<
    m.max_knot_second
    << ",\"shape_gradient_error\":" << m.gradient_error << ",\"shape_hessian_error\":" <<
    m.hessian_error
    << ",\"shape_regularity_backtracks\":" << m.regularity_backtracks
    << ",\"shape_reused_planes\":" << m.reused_planes
    << ",\"shape_direct_initialization\":" << (m.direct_initialization ? "true" : "false");
}
}
