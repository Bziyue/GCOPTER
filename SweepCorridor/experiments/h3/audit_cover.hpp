// SPDX-License-Identifier: MIT
#pragma once
#include <SweepCorridor.hpp>

namespace SweepCorridor::experiments::h3 {

// Offline check against the original solids. Query the AABB of the *exported
// control disks*, independently of the seed boxes used by the separator.
inline bool auditObstacleCover(const internal::cubic::Geometry& geometry,
                              const std::vector<SupportPlane>& cover,
                              const ObstacleIndex& index,double padding) {
  using Plane=internal::detail::SupportPlane;
  struct Row {const Plane* plane; double threshold;};
  std::vector<std::vector<Row>> rows(geometry.pieces());
  const double half=index.voxelHalfWidth();
  for(const auto& plane:cover) {
    double corner_support=0;
    for(int corner=0;corner<8 && half>0;++corner) {
      Vec3 offset;
      for(int axis=0;axis<3;++axis) offset[axis]=(corner&(1<<axis))?half:-half;
      corner_support=std::max(corner_support,plane.normal.dot(offset));
    }
    if(std::abs(plane.normal.norm()-1)>1e-12) return false;
    rows[plane.span].push_back({&plane,plane.bound+padding+corner_support});
  }
  std::vector<int> ids;
  for(int span=0;span<int(geometry.pieces());++span) {
    auto& ordered=rows[span];
    std::sort(ordered.begin(),ordered.end(),[](const Row& a,const Row& b) {
      return a.plane->begin<b.plane->begin;
    });
    Vec3 lo=Vec3::Constant(INFINITY), hi=-lo;
    for(const auto& v:geometry.curve(span)) {
      const Vec3 reach=internal::cubic::shape(v).rowwise().norm();
      lo=lo.cwiseMin(v.head<3>()-reach);
      hi=hi.cwiseMax(v.head<3>()+reach);
    }
    index.query(lo-Vec3::Constant(padding),hi+Vec3::Constant(padding),ids);
    for(int id:ids) {
      const auto& s=index.obstacles()[id];
      double covered=0;
      for(const auto& row:ordered) {
        const auto& p=*row.plane;
        if(p.begin>covered) break;
        if(p.end>covered && p.normal.dot(s.center)>=row.threshold+(half>0?0:s.radius)-1e-9)
          covered=p.end;
        if(covered==1) break;
      }
      if(covered<1) return false;
    }
  }
  return true;
}
}
