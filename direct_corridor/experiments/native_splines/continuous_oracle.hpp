// SPDX-License-Identifier: MIT
#pragma once
#include "cubic_certificates.hpp"
#include <limits>

namespace gcopter::direct::native {
struct ConeOracleResult {
  bool safe=false,witnessed=false;
  double parameter=.5,violation=-INFINITY;
  int nodes=0,depth=0;
};
template<int D> Curve<2*D,double> squaredCone(const Curve<D,Vec3>& p,double tolerance) {
  static const auto weights=[] {
    std::array<std::array<double,D+1>,D+1> w{};
    for(int i=0;i<=D;++i)for(int j=0;j<=D;++j)
      w[i][j]=binomial(D,i)*binomial(D,j)/binomial(2*D,i+j);
    return w;
  }();
  Curve<2*D,double> q{};
  for(int i=0;i<=D;++i)for(int j=0;j<=D;++j)
    q[i+j]+=weights[i][j]*((p[i][0]+tolerance)*(p[j][0]+tolerance)-p[i].template tail<2>().dot(p[j].template tail<2>()));
  return q;
}
template<int D> bool coneOracleNode(const Curve<D,Vec3>& p,const Curve<2*D,double>& q,
    double tolerance,double begin,double end,int depth,ConeOracleResult& out) {
  ++out.nodes;out.depth=std::max(out.depth,depth);
  double dmin=INFINITY,qmin=INFINITY,scale=0;
  for(const auto& v:p){dmin=std::min(dmin,v[0]+tolerance);scale=std::max(scale,v.squaredNorm());}
  for(double v:q)qmin=std::min(qmin,v);
  const double roundoff=64*std::numeric_limits<double>::epsilon()*scale;
  if(dmin>=0&&qmin>=roundoff)return true;
  for(double t:{0.,.5,1.}) {
    const auto v=value<D>(p,t);const double residual=v.template tail<2>().norm()-v[0];
    if(residual>out.violation){out.violation=residual;out.parameter=begin+t*(end-begin);}
    if(residual>tolerance){out.witnessed=true;return false;}
  }
  if(depth==20||out.nodes>=4096)return false;
  const auto halves=split<D>(p);const auto squares=split<2*D>(q);
  const double mid=.5*(begin+end);
  return coneOracleNode<D>(halves.first,squares.first,tolerance,begin,mid,depth+1,out)&&
         coneOracleNode<D>(halves.second,squares.second,tolerance,mid,end,depth+1,out);
}
// p=(positive side, two transverse components). Squaring alone is insufficient:
// certify both the positive side and its degree-2D squared SOC slack.
template<int D> ConeOracleResult coneOracle(const Curve<D,Vec3>& p,double tolerance=1e-9) {
  ConeOracleResult result;
  bool hull=true;for(const auto& v:p)hull&=v[0]+tolerance>=v.template tail<2>().norm();
  if(hull){result.safe=true;result.nodes=1;return result;}
  const auto q=squaredCone<D>(p,tolerance);
  result.safe=coneOracleNode<D>(p,q,tolerance,0.,1.,0,result);
  return result;
}
inline ConeOracleResult supportOracle(const Geometry& g,const detail::SupportPlane& plane,
                                      double tolerance=1e-9) {
  const auto controls=restrictCurve<3>(g.curve<3>(plane.span),plane.begin,plane.end);
  Curve<3,Vec3> cone;
  for(int i=0;i<4;++i)cone[i]<<plane.bound-plane.normal.dot(controls[i].head<3>()),
                             shape(controls[i]).transpose()*plane.normal;
  auto result=coneOracle<3>(cone,tolerance);
  result.parameter=plane.begin+result.parameter*(plane.end-plane.begin);
  return result;
}
} // namespace gcopter::direct::native
