// SPDX-License-Identifier: MIT
#pragma once
#include "Types.hpp"
#include "CubicBasis.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <array>

namespace SweepCorridor::internal::cubic {
using V9 = Eigen::Matrix<double,9,1>;
inline V9 pack(const Vec3& c,const Mat32& m) {
  V9 v;v.head<3>()=c;v.tail<6>()=Eigen::Map<const Eigen::Matrix<double,6,1>>(m.data());return v;
}
inline Mat32 shape(const V9& v) { return Eigen::Map<const Mat32>(v.data()+3); }
template<int D,class V> using Curve = std::array<V,D+1>;
template<int D,class V> inline V value(Curve<D,V> p,double t) {
  for(int k=D;k>0;--k) for(int j=0;j<k;++j) p[j]=(1-t)*p[j]+t*p[j+1];
  return p[0];
}
template<int D,class V> inline std::pair<Curve<D,V>,Curve<D,V>> split(Curve<D,V> p,double t=.5) {
  Curve<D,V> a,b;a[0]=p[0];b[D]=p[D];
  for(int k=1;k<=D;++k) {
    for(int j=0;j<=D-k;++j) p[j]=(1-t)*p[j]+t*p[j+1];
    a[k]=p[0];b[D-k]=p[D-k];
  }
  return {a,b};
}
template<int D,class V> inline Curve<D,V> restrictCurve(Curve<D,V> p,double a,double b) {
  if(b<1) p=split<D>(p,b).first;
  if(a>0) p=split<D>(p,a/b).second;
  return p;
}
inline double binomial(int n,int k) { double v=1;for(int j=1;j<=k;++j) v*=double(n+1-j)/j;return v; }
struct Geometry {
  static constexpr int degree=3;
  std::vector<double> breaks;
  std::vector<V9> controls; // Exactly degree+1 rows per span, world c / M.
  size_t pieces() const { return breaks.empty()?0:breaks.size()-1; }
  template<int D> Curve<D,V9> curve(int i) const {
    Curve<D,V9> p;std::copy_n(controls.begin()+(D+1)*i,D+1,p.begin());return p;
  }
  template<int D> void append(const Curve<D,V9>& p) { controls.insert(controls.end(),p.begin(),p.end()); }
  Section section(double u) const;
  static Geometry fromCubic(const CubicBasis& basis,const std::vector<V9>& controls);
};
}
