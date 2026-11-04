#pragma once
#include "GeometryData.hpp"
namespace SweepCorridor::internal::cubic {
template<int D> inline Section evaluate(const Curve<D,V9>& p,double t,double h) {
  Curve<D-1,V9> d;Curve<D-2,V9> dd;
  for(int j=0;j<D;++j) d[j]=D/h*(p[j+1]-p[j]);
  for(int j=0;j<D-1;++j) dd[j]=(D-1)/h*(d[j+1]-d[j]);
  const V9 y=value<D>(p,t),dy=value<D-1>(d,t),ddy=value<D-2>(dd,t);
  return {y.head<3>(),dy.head<3>(),ddy.head<3>(),shape(y),shape(dy),shape(ddy)};
}
inline Section Geometry::section(double u) const {
  const int i=std::clamp(int(std::upper_bound(breaks.begin(),breaks.end(),u)-breaks.begin())-1,0,int(pieces())-1);
  const double h=breaks[i+1]-breaks[i],t=(u-breaks[i])/h;
  return evaluate<3>(curve(i),t,h);
}
inline Geometry Geometry::fromCubic(const CubicBasis& b,const std::vector<V9>& values) {
  Geometry g;g.breaks=b.breaks;g.controls.resize(4*b.spans.size());
  // For simple cubic knots, extraction rows have 3,2,2,3 nonzero entries.
  // Retain their accumulation order while skipping six algebraic zeros.
  constexpr int begin[4]={0,1,1,1},end[4]={3,3,3,4};
  for(std::size_t i=0;i<b.spans.size();++i) {
    const auto& s=b.spans[i];
    for(int j=0;j<4;++j) {
      auto& v=g.controls[4*i+j];v.setZero();
      for(int k=begin[j];k<end[j];++k)v+=s.extraction(j,k)*values[s.ids[k]];
    }
  }
  return g;
}
}
