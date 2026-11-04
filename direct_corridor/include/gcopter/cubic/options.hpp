// SPDX-License-Identifier: MIT
#pragma once
#include "gcopter/direct_corridor.hpp"
namespace gcopter::direct::native {
struct CubicOptions : GeometryOptions {
  CubicOptions() { frame_smoothing=1.5; }
  explicit CubicOptions(const GeometryOptions& geometry):GeometryOptions(geometry) {}
};
}
