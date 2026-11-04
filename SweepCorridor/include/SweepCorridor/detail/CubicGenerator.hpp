// SPDX-License-Identifier: MIT
#pragma once
#include "GeometryData.hpp"
#include "CubicMetrics.hpp"
#include "CubicOptions.hpp"
#include "SupportPlane.hpp"
namespace SweepCorridor::internal::cubic {
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
                          const Bounds& bounds,const CubicOptions& options,const OptimizerSettings& solver);
inline CubicResult generateNative(const std::vector<Vec3>& seed,const ObstacleIndex& index,
    const Bounds& bounds,CubicOptions options) {
  OptimizerSettings solver;
  return generateCubic(seed,index,bounds,options,solver);
}
}
