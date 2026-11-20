// SPDX-License-Identifier: MIT
#pragma once
#include "model.hpp"

namespace SweepCorridor::experiments::h3 {

// Exact midpoint knot insertion. Measure offsets from the original seed
// reference and keep its endpoint anchors, even after a free cap has moved.
inline Model bisectModel(const Model& coarse,const Geometry& reference) {
  const auto g=coarse.geometry();
  Geometry fine;
  fine.breaks.reserve(2*g.pieces()+1);
  fine.controls.reserve(8*g.pieces());
  fine.breaks.push_back(g.breaks.front());
  for(size_t i=0;i<g.pieces();++i) {
    const auto halves=c3::split<3>(g.curve(i));
    fine.controls.insert(fine.controls.end(),halves.first.begin(),halves.first.end());
    fine.controls.insert(fine.controls.end(),halves.second.begin(),halves.second.end());
    fine.breaks.push_back(.5*(g.breaks[i]+g.breaks[i+1]));
    fine.breaks.push_back(g.breaks[i+1]);
  }
  Model out(fine);
  for(int i=0;i<out.basis.count;++i) {
    const Vec3 base=reference.section(out.basis.breaks[i]).center;
    out.x.segment<3>(6*i)=out.reference[i]-base;
    out.reference[i]=base;
  }
  out.anchor_parameters=coarse.anchor_parameters;
  return out;
}

inline std::vector<SupportPlane> bisectCover(const std::vector<SupportPlane>& old) {
  std::vector<SupportPlane> out;
  out.reserve(2*old.size());
  for(size_t first=0;first<old.size();) {
    size_t end=first+1;
    while(end<old.size() && old[end].span==old[first].span) ++end;
    for(int child=0;child<2;++child) for(size_t i=first;i<end;++i) {
      auto p=old[i];
      const double lo=std::max(0.,2*p.begin-child), hi=std::min(1.,2*p.end-child);
      if(lo>hi || (lo==hi && p.begin!=p.end)) continue;
      p.span=2*p.span+child;
      p.begin=lo; p.end=hi;
      out.push_back(p);
    }
    first=end;
  }
  return out;
}

inline int coverGroupEnd(const c3::CoverLayout& layout,int first,int group_size) {
  const auto& domain=layout.spans[first];
  int end=first+1;
  while(end<int(layout.spans.size()) && end-first<group_size &&
        layout.spans[end].lower==domain.lower && layout.spans[end].upper==domain.upper) ++end;
  return end;
}

inline double groupChordRadius(const Geometry& g,int first,int end) {
  return .5*(g.controls[4*end-1]-g.controls[4*first]).head<3>().norm();
}

inline void separateInflationSpan(c3::MapCover<3>& builder,const c3::GeometryPatch<3>& patch,
                                  const c3::CoverLayout& layout,double longitudinal_floor) {
  const auto checkpoint=builder.planes.size();
  builder.longitudinal_floor=longitudinal_floor;
  try { builder.obstaclesForSpan(patch,layout); }
  catch(const c3::Failure& error) {
    if(error.status!=Status::InsufficientClearance || longitudinal_floor<=0) throw;
    builder.planes.resize(checkpoint);
    builder.longitudinal_floor=0;
    builder.obstaclesForSpan(patch,layout);
  }
}

// A full-span separator excludes obstacles independently of the patch that
// proposed it. Share it only within exactly the same BVH query domain and
// only when every receiving Bezier control disk is strictly on its safe side.
inline std::vector<internal::detail::SupportPlane> inflationCover(
    const Geometry& g,const ObstacleIndex& index,const Bounds& bounds,
    const GeometryOptions& options,c3::QueryCache& cache,c3::CoverMetrics& metrics,
    Deadline& deadline,const c3::CoverLayout& layout,const OptimizerSettings& settings) {
  StageTimer total(settings.profile);
  c3::MapCover<3> builder(index,bounds,options,cache,metrics,deadline,settings.profile);
  const int group_size=settings.cover_group_size;
  builder.prefer_faces=group_size>1 || index.voxelHalfWidth()>0;
  builder.group_axis_limit=settings.group_axis_limit;
  using Patch=c3::GeometryPatch<3>;
  using Planes=std::vector<internal::detail::SupportPlane>;
  Planes out;
  for(int first=0;first<int(g.pieces());) {
    const int end=coverGroupEnd(layout,first,group_size);
    const int middle=first+(end-first)/2;
    const Patch representative{middle,0,1,g.curve(middle)};
    const double group_floor=groupChordRadius(g,first,end);
    const auto floor=[&](int i) {
      return settings.separator_scale==SeparatorScale::Span ? groupChordRadius(g,i,i+1) : group_floor;
    };
    deadline.check();
    separateInflationSpan(builder,representative,layout,floor(middle));
    Planes proposal;
    proposal.swap(builder.planes);
    const bool full=std::all_of(proposal.begin(),proposal.end(),
        [](const auto& p){return p.begin==0 && p.end==1;});
    for(int i=first;i<end;++i) {
      deadline.check();
      const Patch patch{i,0,1,g.curve(i)};
      builder.boundsForSpan(patch,layout);
      const bool fits=i==middle || (settings.share_cover_planes && full && std::all_of(proposal.begin(),proposal.end(),
          [&](const auto& p){return c3::supportBound<3>(patch.controls,p.normal,p.bound)<-1e-9;}));
      if(fits) {
        if(i!=middle && !proposal.empty()) ++metrics.shared_obstacle_spans;
        for(auto p:proposal) { p.span=i; builder.planes.push_back(p); }
      } else {
        separateInflationSpan(builder,patch,layout,floor(i));
      }
      out.insert(out.end(),builder.planes.begin(),builder.planes.end());
      builder.planes.clear();
    }
    first=end;
  }
  metrics.total_ms+=total.elapsed();
  return out;
}
}
