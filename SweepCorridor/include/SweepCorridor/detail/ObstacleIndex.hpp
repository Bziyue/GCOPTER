// SPDX-License-Identifier: MIT
#pragma once
#include "Types.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace SweepCorridor::internal {
inline ObstacleIndex::ObstacleIndex(std::vector<Sphere> obstacles)
    : obstacles_(std::move(obstacles)) {
  if (obstacles_.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max() / 2))
    throw std::invalid_argument("too many obstacles");
  for (const auto& s : obstacles_)
    if (!s.center.allFinite() || !std::isfinite(s.radius) || s.radius < 0 ||
        !(s.center.array().abs() + s.radius).isFinite().all())
      throw std::invalid_argument("invalid obstacle sphere");
  order_.resize(obstacles_.size());
  std::iota(order_.begin(), order_.end(), 0);
  nodes_.reserve(obstacles_.size() / 4 + 1);
  if (!order_.empty()) build(0, static_cast<int>(order_.size()));
}
inline int ObstacleIndex::build(int begin, int end) {
  Node n;
  n.begin = begin;
  n.end = end;
  n.lower = Vec3::Constant(std::numeric_limits<double>::infinity());
  n.upper = -n.lower;
  for (int i = begin; i < end; ++i) {
    const auto& s = obstacles_[order_[i]];
    n.lower = n.lower.cwiseMin(s.center - Vec3::Constant(s.radius));
    n.upper = n.upper.cwiseMax(s.center + Vec3::Constant(s.radius));
  }
  const int id = static_cast<int>(nodes_.size());
  nodes_.push_back(n);
  if (end - begin > 8) {
    Eigen::Index axis;
    (n.upper - n.lower).maxCoeff(&axis);
    int mid = begin + (end - begin) / 2;
    std::nth_element(order_.begin() + begin, order_.begin() + mid,
                     order_.begin() + end, [&](int a, int b) {
                       const double x = obstacles_[a].center[axis],
                                    y = obstacles_[b].center[axis];
                       return x == y ? a < b : x < y;
                     });
    int left = build(begin, mid), right = build(mid, end);
    nodes_[id].left = left;
    nodes_[id].right = right;
  }
  return id;
}
inline void ObstacleIndex::query(const Vec3& lower, const Vec3& upper,
                          std::vector<int>& ids) const {
  ids.clear();
  if (nodes_.empty()) return;
  // Balanced median BVH over at most INT_MAX/2 leaves needs <64 stack entries.
  std::array<int, 64> stack{};
  int size = 1;
  stack[0] = 0;
  while (size) {
    const auto& n = nodes_[stack[--size]];
    if ((n.lower.array() > upper.array()).any() ||
        (n.upper.array() < lower.array()).any())
      continue;
    if (n.left < 0) {
      for (int k = n.begin; k < n.end; ++k) {
        int id = order_[k];
        const auto& s = obstacles_[id];
        if ((s.center.array() - s.radius <= upper.array()).all() &&
            (s.center.array() + s.radius >= lower.array()).all())
          ids.push_back(id);
      }
    } else {
      stack[size++] = n.left;
      stack[size++] = n.right;
    }
  }
}

}
