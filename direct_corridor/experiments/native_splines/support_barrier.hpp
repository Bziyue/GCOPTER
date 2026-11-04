// SPDX-License-Identifier: MIT
#pragma once
#include <Eigen/Core>
#include <cmath>

namespace gcopter::direct::native {
// Fixed regularization, not a state-dependent norm clamp. For unit map-cover
// normals, the smooth support exceeds ||w|| by at most 1 nm.
inline constexpr double support_epsilon=1e-9;
inline double supportRadius(const Eigen::Vector3d& value) {
  return std::sqrt(value.tail<2>().squaredNorm()+support_epsilon*support_epsilon);
}
inline void distanceBarrierJet(const Eigen::Vector3d& value,double mu,double radius,double gap,
                               Eigen::Vector3d& gradient,Eigen::Matrix3d& hessian) {
  gradient<<1,value[1]/radius,value[2]/radius;
  hessian=(mu/(gap*gap))*gradient*gradient.transpose();
  hessian.bottomRightCorner<2,2>().noalias()+=(mu/(gap*radius))*
      (Eigen::Matrix2d::Identity()-gradient.tail<2>()*gradient.tail<2>().transpose());
  gradient*=mu/gap;
}
}
