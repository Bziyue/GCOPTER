// SPDX-License-Identifier: MIT
#pragma once
#include <cmath>
#include <SweepCorridor.hpp>

namespace gcopter::direct {
using SweepCorridor::Vec3;
using SweepCorridor::Bounds;
using SweepCorridor::Sphere;
using SweepCorridor::ObstacleIndex;
/** Read-only adapter for GCOPTER's OMPL search. The reference clearance is
 * reserved during search only; the corridor uses the physical padding itself.
 * All closed search edges are checked against the declared obstacle geometry.
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
    return index_.segmentFree(a,b,padding_);
  }

 private:
  const ObstacleIndex& index_;
  Bounds bounds_;
  double padding_;
};
}  // namespace gcopter::direct
