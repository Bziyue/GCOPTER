// SPDX-License-Identifier: MIT
#pragma once
#include <Eigen/Core>

namespace gcopter::direct::native {
// Exact supremum of sum_i v_i^T S_i z for one shared ||z|| <= 1 and
// lower I <= S_i <= upper I. Domain bounds are established at construction.
// Taking the norm after summation preserves cancellation between frames.
inline double exactShapeReach(double norm_sum,const Eigen::Vector2d& sum,
                              double lower,double upper) noexcept {
  return .5*(upper-lower)*norm_sum+.5*(upper+lower)*sum.norm();
}
}
