// SPDX-License-Identifier: MIT
#pragma once
#include "Types.hpp"
namespace SweepCorridor::internal::cubic {
struct CubicOptions : GeometryOptions {
  CubicOptions() { frame_smoothing=1.5; }
  explicit CubicOptions(const GeometryOptions& geometry):GeometryOptions(geometry) {}
};
}
