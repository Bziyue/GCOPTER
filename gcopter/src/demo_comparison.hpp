// SPDX-License-Identifier: MIT
#pragma once
#include <SweepCorridor.hpp>
#include <atomic>
#include "TrajectoryOptComponents/SFCCommonTypes.hpp"
#include <memory>
#include <string>
namespace demo_comparison {
using Vec3=SweepCorridor::Vec3;
struct FiriCorridor {
  traj_opt_components::PolyhedraH polys;
  std::vector<Vec3> triangles,edges;
  double milliseconds=0;
};
FiriCorridor buildFiri(const std::vector<Vec3>& seed,const SweepCorridor::ObstacleIndex& map,
                      const SweepCorridor::Bounds& bounds,const SweepCorridor::Options& options);
struct Trajectory {
  std::vector<Vec3> points;
  double optimize_ms=0,duration=0,length=0,jerk=0,max_violation=0,penetration=0;
  int status=0,iterations=0;
  bool feasible=false;
};
struct Comparison {Trajectory firi,c3;};
Comparison optimize(const std::vector<Vec3>& seed,const SweepCorridor::Corridor& corridor,
                    const FiriCorridor& firi,const SweepCorridor::ObstacleIndex& map,
                    const SweepCorridor::Bounds& bounds,double padding,
                    const std::shared_ptr<std::atomic_bool>& cancel);
} // namespace demo_comparison
