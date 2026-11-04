// SPDX-License-Identifier: MIT
#pragma once
#include "geometry.hpp"
#include "../../src/generation_pipeline.hpp"
#include <limits>

namespace gcopter::direct::native {
struct Certificate {
  bool support=true, jacobian=true;
  int support_failed=0, jacobian_failed=0, support_leaves=0, jacobian_leaves=0;
  int support_depth=0, jacobian_depth=0;
  double support_residual=-INFINITY, cone_relative_margin=INFINITY;
  bool safe() const { return support&&jacobian; }
};
template<int D> inline double supportBound(const Curve<D,V9>& p,const Vec3& normal,double bound) {
  double violation=-INFINITY;
  for(const auto& v:p)violation=std::max(violation,normal.dot(v.template head<3>())+(shape(v).transpose()*normal).norm()-bound);
  return violation;
}
template<int D> inline bool supportCertificate(const Curve<D,V9>& p,const Vec3& n,double b,
    int budget,Certificate& r,int depth=0) {
  const double violation=supportBound<D>(p,n,b);
  r.support_depth=std::max(r.support_depth,depth);
  if(violation<=1e-9 || depth==budget) {
    ++r.support_leaves;r.support_residual=std::max(r.support_residual,violation);
    return violation<=1e-9;
  }
  // A violating endpoint is a witness; subdivision cannot fix it.
  if(n.dot(p.front().template head<3>())+(shape(p.front()).transpose()*n).norm()>b+1e-9 ||
     n.dot(p.back().template head<3>())+(shape(p.back()).transpose()*n).norm()>b+1e-9) {
    ++r.support_leaves;r.support_residual=std::max(r.support_residual,violation);return false;
  }
  const auto halves=split<D>(p);
  const bool a=supportCertificate<D>(halves.first,n,b,budget,r,depth+1);
  const bool c=supportCertificate<D>(halves.second,n,b,budget,r,depth+1);
  return a&&c;
}
template<int D> inline Curve<3*D-1,Vec3> jacobianCoefficients(const Curve<D,V9>& p) {
  static const auto weights=[] {
    std::array<std::array<double,D+1>,D+1> a{};
    std::array<std::array<double,2*D+1>,D> b{};
    for(int j=0;j<=D;++j)for(int k=0;k<=D;++k)a[j][k]=binomial(D,j)*binomial(D,k)/binomial(2*D,j+k);
    for(int j=0;j<D;++j)for(int k=0;k<=2*D;++k)b[j][k]=D*binomial(D-1,j)*binomial(2*D,k)/binomial(3*D-1,j+k);
    return std::make_pair(a,b);
  }();
  Curve<2*D,Vec3> normals;Curve<3*D-1,Vec3> cone;
  for(auto& v:normals)v.setZero();
  for(auto& v:cone)v.setZero();
  for(int j=0;j<=D;++j)for(int k=0;k<=D;++k)
    normals[j+k]+=weights.first[j][k]*shape(p[j]).col(0).cross(shape(p[k]).col(1));
  for(int j=0;j<D;++j)for(int k=0;k<=2*D;++k) {
    const Vec3 n=weights.second[j][k]*normals[k];
    cone[j+k][0]+=(p[j+1]-p[j]).template head<3>().dot(n);
    cone[j+k].template tail<2>()+=(shape(p[j+1])-shape(p[j])).transpose()*n;
  }
  return cone;
}
template<int D> inline bool coneCertificate(const Curve<D,Vec3>& p,double epsilon,double scale,
                                           Certificate& r,int depth=0) {
  double gap=INFINITY;for(const auto& v:p)gap=std::min(gap,v[0]-v.template tail<2>().norm());
  r.jacobian_depth=std::max(r.jacobian_depth,depth);
  if(gap>epsilon) {++r.jacobian_leaves;r.cone_relative_margin=std::min(r.cone_relative_margin,gap/scale);return true;}
  if(depth==12 || p.front()[0]-p.front().template tail<2>().norm()<=epsilon ||
     p.back()[0]-p.back().template tail<2>().norm()<=epsilon) {
    ++r.jacobian_leaves;r.cone_relative_margin=std::min(r.cone_relative_margin,gap/scale);return false;
  }
  const auto halves=split<D>(p);
  const bool a=coneCertificate<D>(halves.first,epsilon,scale,r,depth+1);
  const bool b=coneCertificate<D>(halves.second,epsilon,scale,r,depth+1);
  return a&&b;
}
template<int D> inline bool jacobianCertificate(const Curve<D,V9>& p,Certificate& r) {
  const auto cone=jacobianCoefficients<D>(p);
  double scale=0;for(const auto& v:cone)scale=std::max(scale,v.norm());
  return scale>1e-18 && coneCertificate<3*D-1>(cone,std::max(1e-18,scale*1e-9),scale,r);
}
inline Certificate certify(const Geometry& g,const std::vector<detail::SupportPlane>& cover,int support_depth=9) {
  Certificate out;
  for(const auto& c:cover) {
    const bool ok=g.degree==3?
      supportCertificate<3>(restrictCurve<3>(g.curve<3>(c.span),c.begin,c.end),c.normal,c.bound,support_depth,out):
      supportCertificate<5>(restrictCurve<5>(g.curve<5>(c.span),c.begin,c.end),c.normal,c.bound,support_depth,out);
    out.support_failed+=!ok;
  }
  for(int i=0;i<int(g.pieces());++i) {
    const bool ok=g.degree==3?jacobianCertificate<3>(g.curve<3>(i),out):jacobianCertificate<5>(g.curve<5>(i),out);
    out.jacobian_failed+=!ok;
  }
  out.support=out.support_failed==0;out.jacobian=out.jacobian_failed==0;return out;
}
// Global interpolation of shared jets preserves C2 in the QUINTIC space.
// This is a diagnostic control, deliberately not labelled constrained cubic.
inline Geometry safeJetProposal(const Geometry& source,const Geometry& target,
    const std::vector<detail::SupportPlane>& cover,double& fraction) {
  Geometry q=target.degree==3?elevateGeometry(target):target;
  fraction=1.;
  const auto target_controls=q.controls;
  for(int iteration=0;iteration<=40;++iteration) {
    if(certify(q,cover).safe())return q;
    fraction*=.5;
    for(size_t j=0;j<q.controls.size();++j)q.controls[j]=source.controls[j]+fraction*(target_controls[j]-source.controls[j]);
  }
  fraction=0;return source;
}
}
