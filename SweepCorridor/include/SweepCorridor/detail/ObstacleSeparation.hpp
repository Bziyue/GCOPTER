// SPDX-License-Identifier: MIT
#pragma once
#include "Types.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <map>
namespace SweepCorridor::internal::detail {
struct Failure {Status status;std::string message;};
struct LocalBox {double begin,end;Bounds bounds;};
struct ObstacleGroups {
  struct Block {
    Vec3 center, half;
    double min_radius, max_radius;
    unsigned long long alive;
    int begin, end, nearest;
  };
  std::vector<Block> blocks;
  bool uniform_radius = true;
};
inline ObstacleGroups obstacleGroups(const std::vector<int>& ids,
                                     const ObstacleIndex& map) {
  ObstacleGroups result;
  result.blocks.reserve((ids.size() + 31) / 32);
  for (int begin = 0; begin < int(ids.size()); begin += 32) {
    const int end = std::min(begin + 32, int(ids.size()));
    Vec3 lo = map.obstacles()[ids[begin]].center, hi = lo;
    double rmin = INFINITY, rmax = 0;
    for (int j = begin; j < end; ++j) {
      const auto& sphere = map.obstacles()[ids[j]];
      lo = lo.cwiseMin(sphere.center);
      hi = hi.cwiseMax(sphere.center);
      rmin = std::min(rmin, sphere.radius);
      rmax = std::max(rmax, sphere.radius);
      result.uniform_radius &= sphere.radius == map.obstacles()[ids.front()].radius;
    }
    result.blocks.push_back({.5 * lo + .5 * hi, .5 * hi - .5 * lo, rmin, rmax,
                            (1ull << (end-begin))-1, begin, end, -1});
  }
  return result;
}

struct QueryCache {
  static constexpr std::size_t byte_limit = 16 * 1024 * 1024;
  std::vector<std::vector<int>> ids;
  std::vector<ObstacleGroups> groups;
  std::vector<int> spans;
  std::map<std::array<double, 6>, int> boxes;
  std::size_t bytes = 0;

  const std::vector<int>& query(int span, const Vec3& lower, const Vec3& upper,
                               const ObstacleIndex& map,
                               std::vector<int>& scratch) {
    if (span < int(spans.size()) && spans[span] >= 0) return ids[spans[span]];
    const std::array<double, 6> key{lower[0], lower[1], lower[2],
                                    upper[0], upper[1], upper[2]};
    int entry = -1;
    const auto found = boxes.find(key);
    if (found != boxes.end()) {
      entry = found->second;
    } else {
      map.query(lower, upper, scratch);
      const std::size_t index_bytes = scratch.capacity() * sizeof(int);
      const std::size_t group_bytes = ((scratch.size() + 31) / 32) * sizeof(ObstacleGroups::Block);
      if (index_bytes + group_bytes <= byte_limit - bytes) {
        auto grouped = obstacleGroups(scratch, map);
        const std::size_t size = index_bytes + grouped.blocks.capacity() * sizeof(ObstacleGroups::Block);
        if (size <= byte_limit - bytes) {
          bytes += size;
          entry = int(ids.size());
          ids.push_back(std::move(scratch));
          groups.push_back(std::move(grouped));
          boxes.emplace(key, entry);
        }
      }
    }
    if (span == int(spans.size())) spans.push_back(entry);
    return entry >= 0 ? ids[entry] : scratch;
  }
};

// Closest point of a sphere in an ellipsoidal metric. In principal axes the
// KKT equation is sum((t*delta_i/(axes2_i+t))^2)=radius^2, with a monotone
// bracket. Eight safeguarded Newton steps only propose a normal: the caller
// still certifies the whole Bezier patch against the original sphere.
inline Vec3 metricSphereNormal(const Vec3& delta, const Vec3& axes2,
                               double radius) {
  const double distance = delta.norm();
  if (distance <= radius) return delta;
  double lo = radius * axes2.minCoeff() / (distance - radius);
  double hi = radius * axes2.maxCoeff() / (distance - radius);
  double t = lo;
  for (int it = 0; it < 8; ++it) {
    const Vec3 inverse = (axes2.array() + t).inverse().matrix();
    const Vec3 v = delta.cwiseProduct(inverse);
    const double f = t * t * v.squaredNorm() - radius * radius;
    if (std::abs(f) <= 1e-10 * radius * radius) break;
    if (f < 0)
      lo = t;
    else
      hi = t;
    const double derivative =
        2 * t * v.cwiseAbs2().dot(axes2.cwiseProduct(inverse));
    const double next = t - f / derivative;
    t = next > lo && next < hi ? next : .5 * (lo + hi);
  }
  return delta.cwiseQuotient(axes2 + Vec3::Constant(t));
}

}
