// SPDX-License-Identifier: MIT
#pragma once
#include "Types.hpp"
namespace SweepCorridor::internal::detail {
// A continuous support plane on a local interval of one Bernstein span.
struct SupportPlane {
  int span;
  double begin, end;
  Vec3 normal;
  double bound;
};
}
