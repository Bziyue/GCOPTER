// SPDX-License-Identifier: MIT
#pragma once
#include "gcopter/direct_corridor.hpp"
namespace gcopter::direct::detail {
// A continuous support plane on a local interval of one Bernstein span.
struct SupportPlane {
  int span;
  double begin, end;
  Vec3 normal;
  double bound;
};
}
