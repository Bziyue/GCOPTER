// SPDX-License-Identifier: MIT
#pragma once
#include "gcopter/direct_corridor.hpp"
#include <functional>

namespace gcopter::direct::detail {
// A continuous support plane on a local interval of one Bernstein span.
struct SupportPlane {
  int span;
  double begin, end;
  Vec3 normal;
  double bound;
};
// Optional final shape stage. The cover belongs to the accepted inflation
// iterate, and is consumed before the generator's deadline/return boundary.
using ShapeStage = std::function<void(Corridor&, const std::vector<SupportPlane>&,
                                     double, const std::function<void()>&)>;
Result generateWithShapeStage(const std::vector<Vec3>& seed,
                              const ObstacleIndex& index, const Bounds& bounds,
                              const Options& options, const ShapeStage& stage,
                              int first_iteration_limit = 12,
                              bool fraction_to_boundary = false);
}
