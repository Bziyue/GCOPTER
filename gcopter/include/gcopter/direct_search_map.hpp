// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cmath>
#include <SweepCorridor.hpp>

namespace gcopter::direct {
using SweepCorridor::Vec3;
using SweepCorridor::Bounds;
using SweepCorridor::Sphere;
using SweepCorridor::ObstacleIndex;
/** Read-only adapter for GCOPTER's OMPL search. The reference clearance is
 * reserved during search only; the corridor uses the physical padding itself.
 * All closed search edges are checked against the same solid sphere snapshot.
 */
class SearchMap {
 public:
  SearchMap(const ObstacleIndex& index, Bounds bounds, double padding)
      : index_(index), bounds_(std::move(bounds)), padding_(padding) {}

  bool query(const Vec3& p) const { return !segmentFree(p, p); }
  bool segmentFree(const Vec3& a, const Vec3& b) const {
    if (!a.allFinite() || !b.allFinite() || !std::isfinite(padding_) ||
        padding_ < 0)
      return false;
    for (const auto& p : {a, b})
      if ((p.array() <= bounds_.lower.array() + padding_).any() ||
          (p.array() >= bounds_.upper.array() - padding_).any())
        return false;
    std::vector<int> ids;
    const Vec3 expand = Vec3::Constant(padding_);
    index_.query(a.cwiseMin(b) - expand, a.cwiseMax(b) + expand, ids);
    const Vec3 d = b - a;
    const double squared = d.squaredNorm();
    for (int id : ids) {
      const auto& sphere = index_.obstacles()[id];
      const double t =
          squared > 1e-24
              ? std::clamp((sphere.center - a).dot(d) / squared, 0., 1.)
              : 0.;
      if ((a + t * d - sphere.center).squaredNorm() <=
          std::pow(sphere.radius + padding_, 2))
        return false;
    }
    return true;
  }

 private:
  const ObstacleIndex& index_;
  Bounds bounds_;
  double padding_;
};
}  // namespace gcopter::direct
