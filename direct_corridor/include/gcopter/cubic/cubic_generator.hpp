// SPDX-License-Identifier: MIT
#pragma once
#include "geometry.hpp"
#include "metrics.hpp"
#include "options.hpp"
#include "gcopter/support_plane.hpp"
namespace gcopter::direct::native {
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
                          const Bounds& bounds,const CubicOptions& options,const OptimizerSettings& solver,
                          const Geometry* projected_reference=nullptr);
inline CubicResult generateNative(const std::vector<Vec3>& seed,const ObstacleIndex& index,
    const Bounds& bounds,CubicOptions options) {
  OptimizerSettings solver;
  return generateCubic(seed,index,bounds,options,solver);
}
}
