// SPDX-License-Identifier: MIT
#pragma once
#include "cubic_certificates.hpp"
#include <map>
#include <numeric>

namespace gcopter::direct::native {
struct SupportRefinement {
  std::vector<unsigned char> selected;
  int intervals=0,selected_intervals=0,selected_planes=0,extra_rows=0;
  double gap_sum=0;
};

// Select complete parameter intervals, then replace only the planes whose
// hull bound gets tighter. The pressure is a finite-barrier sensitivity, not
// a converged KKT multiplier. Its common mu factor cancels in the ranking.
inline SupportRefinement selectSupportRefinement(const Geometry& geometry,
    const std::vector<detail::SupportPlane>& planes,int mode,double fraction) {
  SupportRefinement result;result.selected.resize(planes.size());
  struct Interval {
    double score=0,cost=0,measure=0;
    Curve<3,V9> coarse;
    Curve<5,V9> elevated;
    std::pair<Curve<3,V9>,Curve<3,V9>> halves;
    std::vector<size_t> planes;
  };
  std::vector<Interval> intervals;
  std::map<std::tuple<int,double,double>,size_t> ids;
  std::vector<double> gaps(planes.size());
  for(size_t i=0;i<planes.size();++i) {
    const auto& plane=planes[i];if(plane.begin==plane.end)continue;
    const auto inserted=ids.emplace(std::make_tuple(plane.span,plane.begin,plane.end),intervals.size());
    if(inserted.second) {
      intervals.emplace_back();auto& interval=intervals.back();
      interval.coarse=restrictCurve<3>(geometry.curve<3>(plane.span),plane.begin,plane.end);
      if(mode==1)interval.elevated=elevate<3,5>(interval.coarse);
      else interval.halves=split<3>(interval.coarse);
      interval.measure=(geometry.breaks[plane.span+1]-geometry.breaks[plane.span])*(plane.end-plane.begin);
    }
    auto& interval=intervals[inserted.first->second];
    const auto& curve=interval.coarse;
    const double scale=plane.normal.norm();
    const Vec3 normal=plane.normal/scale;const double bound=plane.bound/scale;
    const double coarse=supportBound<3>(curve,normal,bound);
    double fine;
    if(mode==1)fine=supportBound<5>(interval.elevated,normal,bound);
    else fine=std::max(supportBound<3>(interval.halves.first,normal,bound),supportBound<3>(interval.halves.second,normal,bound));
    const double gap=coarse-fine;
    if(gap<=1e-9)continue;
    double pressure=0;
    for(const auto& y:curve) {
      const double d=bound-normal.dot(y.head<3>()),r=(shape(y).transpose()*normal).norm();
      pressure+=2*d/std::max(1e-18,(d-r)*(d+r));
    }
    interval.score+=.25*pressure*gap*interval.measure;
    interval.cost+=mode==1?2:3;
    interval.planes.push_back(i);gaps[i]=gap;
  }
  result.intervals=intervals.size();
  std::vector<size_t> order(intervals.size());std::iota(order.begin(),order.end(),0);
  auto score=[&](size_t i){return intervals[i].cost?intervals[i].score/intervals[i].cost:0.;};
  std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return score(a)>score(b);});
  const size_t budget=std::min(order.size(),size_t(std::ceil(fraction*order.size())));
  for(size_t k=0;k<budget&&score(order[k])>0;++k) {
    ++result.selected_intervals;
    for(size_t i:intervals[order[k]].planes) {
      result.selected[i]=1;++result.selected_planes;result.gap_sum+=gaps[i];
      result.extra_rows+=mode==1?2:3;
    }
  }
  return result;
}
}
