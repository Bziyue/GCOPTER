// SPDX-License-Identifier: MIT
#pragma once
#include <SweepCorridor.hpp>
namespace gcopter::direct {
using SweepCorridor::Vec3;
using SweepCorridor::Mat32;
using SweepCorridor::Sphere;
using SweepCorridor::Bounds;
using SweepCorridor::ObstacleIndex;
using SweepCorridor::Section;
using SweepCorridor::Differential;
using SweepCorridor::Corridor;
using SweepCorridor::Options;
using SweepCorridor::Status;
using SweepCorridor::Result;
using SweepCorridor::Metrics;
using SweepCorridor::statusName;
inline Result generate(const std::vector<Vec3>& seed,const ObstacleIndex& map,
                       const Bounds& bounds,const Options& options={}) {
  return SweepCorridor::Generator(options).generate(seed,map,bounds);
}
}
