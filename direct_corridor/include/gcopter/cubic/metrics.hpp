// SPDX-License-Identifier: MIT
#pragma once
#include "gcopter/direct_corridor.hpp"
#include <cstdint>
namespace gcopter::direct::native {
struct OptimizerMetrics {
  double setup_ms=0,assembly_ms=0,solve_ms=0,line_search_ms=0,certificate_ms=0;
  double initial_problem_ms=0,initial_pruning_ms=0;
  int construction_reuses=0;
  int basis_cache_hits=0,basis_cache_misses=0,group_lookup_reuses=0;
  double gradient_error=0,hessian_error=0,initial_scale=1;
  double gradient_reference_error=0,hessian_reference_error=0,derivative_min_step=1;
  int steps=0,backtracks=0,damping_retries=0,regularity_backtracks=0;
  int regularity_repairs=0;
  int active_constraints=0;
  double final_decrement=0;
  int controls=0,variables=0,constraints=0,hessian_nonzero=0;
  int merged_supports=0,support_groups=0,pruned_supports=0;
  double pruning_ms=0;
  int bound_line_evaluations=0,bound_line_accepts=0,exact_line_evaluations=0;
  int axis_supports=0;
  std::uint64_t gradient_support_rows=0,hessian_support_rows=0,boundary_support_rows=0,line_support_rows=0;
  struct Pass {int geometric,retained,boundary,progress,steps;double pruning_ms;};
  std::vector<Pass> passes;
  int full_step_accepts=0,feasible_scans=0;
  int energy_builds=0;
  int predictor_steps=0,predictor_rejections=0,volume_steps=0;
  double predictor_ms=0,volume_ms=0,volume_gain=0;
  double minimum_step=1,step_sum=0;
  std::string termination="not_started";
};
// One fixed algorithm configuration. Only explicit diagnostics are selectable.
struct OptimizerSettings {
  static constexpr double center_energy=1e-5,shape_energy=1e-5,offset_weight=.001;
  static constexpr double gradient_tolerance=1e-7;
  static constexpr double initial_radius_fraction=.65,fraction_boundary=.99,progress_threshold=.03;
  static constexpr double volume_trust=.1;
  static constexpr int iterations=8,corrector_iterations=3,inflation_iterations=2;
  bool profile=false,check_gradients=false;
};
struct CoverMetrics {
  double query_ms=0, separation_ms=0, total_ms=0;
  double topology_ms=0;
  int topology_builds=0,reused_cover_spans=0;
  size_t priority_evaluations=0;
  Metrics primitives;
};
}
