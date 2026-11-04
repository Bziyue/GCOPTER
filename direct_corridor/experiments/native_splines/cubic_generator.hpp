// SPDX-License-Identifier: MIT
#pragma once
#include "cubic_optimizer.hpp"
#include "obstacle_cover.hpp"
namespace gcopter::direct::native {
enum class NativePreset { Fast, Balanced, Volume };
inline void applyNativePreset(NativePreset preset,Options& options,OptimizerSettings& solver) {
  solver=OptimizerSettings{};
  solver.reuse_construction=true;
  solver.fast_construction=true;
  solver.cache_coverage=true;
  solver.initial_radius_fraction=.65;
  solver.fraction_boundary=.99;
  solver.progress_threshold=.03;
  solver.tensor_assembly=false;
  solver.cached_line=true;
  solver.axis_kernels=true;
  solver.skip_unused_values=true;
  solver.bounded_line=true;
  solver.progress_depth=0;
  solver.center_energy=solver.shape_energy=preset==NativePreset::Volume?1e-6:1e-5;
  solver.iterations=preset==NativePreset::Balanced?8:6;
  solver.corrector_iterations=preset==NativePreset::Volume?0:4;
  if(preset==NativePreset::Balanced) {
    solver.projected_pruning=2;
    solver.path_predictor=1;
    solver.corrector_iterations=3;
    solver.volume_refinements=1;
  }
  options.initial_iterations=solver.iterations;
  options.inflation_iterations=2;
}
struct CubicResult {
  Geometry corridor;
  std::vector<detail::SupportPlane> cover;
  OptimizerMetrics optimizer;
  CoverMetrics cover_metrics;
  double total_ms=0,initialization_ms=0,verification_ms=0;
  int refinements=0,initialization_attempts=0;
  std::string status="not_started",message;
};
CubicResult generateCubic(const std::vector<Vec3>& seed,const ObstacleIndex& index,
                          const Bounds& bounds,const Options& options,const OptimizerSettings& solver,
                          const Geometry* projected_reference=nullptr);
inline CubicResult generateNative(const std::vector<Vec3>& seed,const ObstacleIndex& index,
    const Bounds& bounds,Options options,NativePreset preset=NativePreset::Balanced) {
  OptimizerSettings solver;applyNativePreset(preset,options,solver);
  return generateCubic(seed,index,bounds,options,solver);
}
}
