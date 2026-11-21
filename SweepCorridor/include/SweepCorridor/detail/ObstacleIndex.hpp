// SPDX-License-Identifier: MIT
#pragma once
#include "Types.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace SweepCorridor::internal {
inline ObstacleIndex ObstacleIndex::fromPoints(const std::vector<Vec3>& points) {
  std::vector<Sphere> obstacles;
  obstacles.reserve(points.size());
  for (const auto& point : points) obstacles.push_back({point, 0.});
  return ObstacleIndex(std::move(obstacles));
}

inline ObstacleIndex ObstacleIndex::fromOccupiedVoxels(
    const std::vector<Vec3>& centers, double resolution) {
  const double half = .5 * resolution;
  const double radius = std::sqrt(3.) * half;
  if (!(half > 0) || !std::isfinite(radius))
    throw std::invalid_argument("invalid voxel resolution");
  std::vector<Sphere> spheres;
  spheres.reserve(centers.size());
  for (const auto& center : centers) spheres.push_back({center, radius});
  return ObstacleIndex(std::move(spheres),half);
}

inline ObstacleIndex::ObstacleIndex(std::vector<Sphere> obstacles)
    : ObstacleIndex(std::move(obstacles),0) {}

inline ObstacleIndex::ObstacleIndex(std::vector<Sphere> obstacles,double voxel_half_width)
    : obstacles_(std::move(obstacles)),voxel_half_width_(voxel_half_width) {
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
  if (!order_.empty()) {
    if(voxel_half_width_>0) orderVoxels();
    build(0, static_cast<int>(order_.size()));
  }
}

// Distance from a finite segment to a solid cube. Between face crossings the
// squared distance is a quadratic in t with a fixed set of clamped axes.
inline double segmentVoxelDistanceSquared(const Vec3& a,const Vec3& b,
                                          const Vec3& center,double half) {
  const Vec3 start=a-center, direction=b-a;
  std::array<double,8> breaks;
  breaks.fill(1.);
  breaks[0]=0;
  int count=2;
  for(int axis=0;axis<3;++axis) if(direction[axis]!=0) {
    for(const double face:{-half,half}) {
      const double t=(face-start[axis])/direction[axis];
      if(t>0 && t<1) breaks[count++]=t;
    }
  }
  // Unused face crossings sort behind the interior knots at the endpoint.
  std::sort(breaks.begin(),breaks.end());
  double best=std::numeric_limits<double>::infinity();
  for(int i=1;i<count;++i) {
    const double mid=.5*(breaks[i-1]+breaks[i]);
    const Vec3 point=start+mid*direction;
    const Vec3 offset=point-point.cwiseMax(-half).cwiseMin(half);
    Vec3 slope=direction;
    for(int axis=0;axis<3;++axis) if(offset[axis]==0) slope[axis]=0;
    const double speed=slope.squaredNorm();
    const double step=speed>0
        ? std::clamp(-offset.dot(slope)/speed,breaks[i-1]-mid,breaks[i]-mid) : 0;
    best=std::min(best,(offset+step*slope).squaredNorm());
  }
  return best;
}

inline double ObstacleIndex::pointDistance(int id, const Vec3& point) const {
  const auto& obstacle = obstacles_[id];
  const Vec3 delta = point - obstacle.center;
  if (voxel_half_width_ > 0)
    return (delta.cwiseAbs().array() - voxel_half_width_).max(0.).matrix().norm();
  return std::max(0., delta.norm() - obstacle.radius);
}

inline double ObstacleIndex::supportExtent(int id, const Vec3& direction) const {
  return voxel_half_width_>0 ? voxel_half_width_*direction.cwiseAbs().sum()
                             : obstacles_[id].radius*direction.norm();
}

inline bool ObstacleIndex::segmentFree(const Vec3& a, const Vec3& b, double padding) const {
  if(!a.allFinite() || !b.allFinite() || !(padding>=0) || !std::isfinite(padding)) return false;
  std::vector<int> ids;
  query(a.cwiseMin(b) - Vec3::Constant(padding), a.cwiseMax(b) + Vec3::Constant(padding), ids);
  const Vec3 direction = b - a;
  const double length2 = direction.squaredNorm();
  for (int id : ids) {
    const auto& obstacle = obstacles_[id];
    if (voxel_half_width_ > 0) {
      if (segmentVoxelDistanceSquared(a, b, obstacle.center, voxel_half_width_) <= padding * padding)
        return false;
    } else {
      const double t = length2 > 0 ? std::clamp((obstacle.center-a).dot(direction)/length2, 0., 1.) : 0.;
      const double radius = obstacle.radius + padding;
      if ((a+t*direction-obstacle.center).squaredNorm() <= radius*radius) return false;
    }
  }
  return true;
}

// Quantization controls only the ordering. Each leaf still encloses the
// original double-precision obstacles. Stable radix sorting three ten-bit
// digits gives spatial packets and bottom-up BVH bounds in linear time.
inline void ObstacleIndex::orderVoxels() {
  Vec3 lo=obstacles_.front().center, hi=lo;
  for(const auto& s:obstacles_) { lo=lo.cwiseMin(s.center);hi=hi.cwiseMax(s.center); }
  const double half_size=(.5*hi-.5*lo).maxCoeff();
  if(half_size==0) return;
  auto interleave=[](std::uint32_t v) {
    v=(v|(v<<16))&0x030000ffu;
    v=(v|(v<<8))&0x0300f00fu;
    v=(v|(v<<4))&0x030c30c3u;
    return (v|(v<<2))&0x09249249u;
  };
  std::vector<std::uint32_t> keys(order_.size());
  for(size_t i=0;i<keys.size();++i) {
    const Vec3 relative=(.5*obstacles_[i].center-.5*lo)/half_size;
    for(int j=0;j<3;++j) {
      const auto value=static_cast<std::uint32_t>(std::clamp(1024*relative[j],0.,1023.));
      keys[i]|=interleave(value)<<j;
    }
  }
  std::vector<int> scratch(order_.size());
  for(int shift:{0,10,20}) {
    std::array<int,1024> offsets{};
    auto digit=[&](int id){return (keys[id]>>shift)&1023u;};
    for(int id:order_) ++offsets[digit(id)];
    int sum=0;
    for(int& offset:offsets) {const int count=offset;offset=sum;sum+=count;}
    for(int id:order_) scratch[offsets[digit(id)]++]=id;
    order_.swap(scratch);
  }
}

inline int ObstacleIndex::build(int begin, int end) {
  Node n;
  n.begin = begin;
  n.end = end;
  n.lower = Vec3::Constant(std::numeric_limits<double>::infinity());
  n.upper = -n.lower;
  if(voxel_half_width_==0 || end-begin<=8) for (int i = begin; i < end; ++i) {
    const auto& s = obstacles_[order_[i]];
    n.lower = n.lower.cwiseMin(s.center - Vec3::Constant(s.radius));
    n.upper = n.upper.cwiseMax(s.center + Vec3::Constant(s.radius));
  }
  const int id = static_cast<int>(nodes_.size());
  nodes_.push_back(n);
  if (end - begin > 8) {
    int mid = begin + (end - begin) / 2;
    if(voxel_half_width_==0) {
      Eigen::Index axis;
      (n.upper - n.lower).maxCoeff(&axis);
      std::nth_element(order_.begin() + begin, order_.begin() + mid,
                     order_.begin() + end, [&](int a, int b) {
                       const double x = obstacles_[a].center[axis],
                                    y = obstacles_[b].center[axis];
                       return x == y ? a < b : x < y;
                     });
    }
    int left = build(begin, mid), right = build(mid, end);
    nodes_[id].left = left;
    nodes_[id].right = right;
    if(voxel_half_width_>0) {
      nodes_[id].lower=nodes_[left].lower.cwiseMin(nodes_[right].lower);
      nodes_[id].upper=nodes_[left].upper.cwiseMax(nodes_[right].upper);
    }
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
