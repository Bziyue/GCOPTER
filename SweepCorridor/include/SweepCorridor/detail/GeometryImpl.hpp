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
  return degree==3?evaluate<3>(curve<3>(i),t,h):evaluate<5>(curve<5>(i),t,h);
}
inline Geometry Geometry::fromCubic(const CubicBasis& b,const std::vector<V9>& values) {
  Geometry g;g.breaks=b.breaks;
  for(const auto& s:b.spans) for(int j=0;j<4;++j) {
    V9 v=V9::Zero();for(int k=0;k<4;++k) v+=s.extraction(j,k)*values[s.ids[k]];
    g.controls.push_back(v);
  }
  return g;
}
}
