// SPDX-License-Identifier: MIT
#pragma once
#include "gcopter/support_plane.hpp"
#include <functional>

namespace gcopter::direct::detail {
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
