// SPDX-License-Identifier: MIT
#pragma once
#include "GeometryData.hpp"
#include "SupportPlane.hpp"
#include <limits>

namespace SweepCorridor::internal::cubic {
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
template<int D> inline Curve<3*D-1,Vec3> jacobianCoefficients(const Curve<D,V9>& p,Curve<2*D,Vec3>* section_normals=nullptr) {
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
  if(section_normals)*section_normals=normals;
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
// The historical span-relative check remains available for C3. In the
// pointwise test each Bernstein coefficient certifies its own disk/axis ratio.
struct JacobianRequirement {
  double margin;
  bool pointwise;
  double min_axis_travel;
  JacobianRequirement(double relative=1e-9,bool disk_relative=false,double travel=0)
      :margin(relative),pointwise(disk_relative),min_axis_travel(travel){}
  JacobianRequirement onSpan(double width) const { return {margin,pointwise,min_axis_travel*width}; }
};

template<int D> inline bool diskConeCertificate(const Curve<D,Vec3>& cone,double epsilon,
    double margin,Certificate& r,int depth=0) {
  double max_ratio2=0;
  bool interior=true;
  auto valid=[&](const Vec3& v) {
    const double axis=v[0], tilt2=v.tail<2>().squaredNorm();
    const double allowed=std::min(axis-epsilon,(1-margin)*axis);
    return allowed>0 && tilt2<allowed*allowed;
  };
  for(const auto& v:cone) {
    interior=valid(v) && interior;
    max_ratio2=std::max(max_ratio2,v[0]>0 ? v.template tail<2>().squaredNorm()/(v[0]*v[0]) : INFINITY);
  }
  r.jacobian_depth=std::max(r.jacobian_depth,depth);
  if(interior || depth==12 || !valid(cone.front()) || !valid(cone.back())) {
    ++r.jacobian_leaves;
    r.cone_relative_margin=std::min(r.cone_relative_margin,1-std::sqrt(max_ratio2));
    return interior;
  }
  const auto halves=split<D>(cone);
  const bool left=diskConeCertificate<D>(halves.first,epsilon,margin,r,depth+1);
  const bool right=diskConeCertificate<D>(halves.second,epsilon,margin,r,depth+1);
  return left && right;
}
template<int D> inline bool certifyJacobianCone(const Curve<D,Vec3>& cone,
    Certificate& r,JacobianRequirement requirement) {
  double scale2=0;
  for(const auto& v:cone) scale2=std::max(scale2,v.squaredNorm());
  const double scale=std::sqrt(scale2);
  if(!(scale>1e-18) || !std::isfinite(scale)) return false;
  if(requirement.pointwise)
    return diskConeCertificate<D>(cone,std::max(1e-18,1e-9*scale),requirement.margin,r);
  return coneCertificate<D>(cone,std::max(1e-18,scale*requirement.margin),scale,r);
}

// q0/(|M1 x M2|) is normal center travel per local parameter. Scaling
// the requested travel by the seed arc length of this span fixes the units.
template<int D> inline bool centerProgressCertificate(const Curve<3*D-1,Vec3>& cone,
    const Curve<2*D,Vec3>& normals,double travel,int depth=0) {
  double min_axis=INFINITY,max_area2=0;
  for(const auto& q:cone) min_axis=std::min(min_axis,q[0]);
  for(const auto& n:normals) max_area2=std::max(max_area2,n.squaredNorm());
  const double scale2=travel*travel;
  if(min_axis>0 && min_axis*min_axis>scale2*max_area2) return true;
  auto valid=[&](const Vec3& q,const Vec3& n) {
    return q[0]>0 && q[0]*q[0]>scale2*n.squaredNorm();
  };
  if(depth==12 || !valid(cone.front(),normals.front()) || !valid(cone.back(),normals.back())) return false;
  const auto q=split<3*D-1>(cone), n=split<2*D>(normals);
  return centerProgressCertificate<D>(q.first,n.first,travel,depth+1) &&
         centerProgressCertificate<D>(q.second,n.second,travel,depth+1);
}

template<int D> inline bool jacobianCertificate(const Curve<D,V9>& p,Certificate& r,
                                               JacobianRequirement requirement={}) {
  Curve<2*D,Vec3> normals;
  const auto cone=jacobianCoefficients<D>(p,requirement.min_axis_travel>0?&normals:nullptr);
  return certifyJacobianCone<3*D-1>(cone,r,requirement) &&
         (requirement.min_axis_travel<=0 || centerProgressCertificate<D>(cone,normals,requirement.min_axis_travel));
}
inline Certificate certify(const Geometry& g,const std::vector<detail::SupportPlane>& cover,
                           int support_depth=9,JacobianRequirement requirement={}) {
  Certificate out;
  for(const auto& c:cover) {
    const bool ok=supportCertificate<3>(restrictCurve<3>(g.curve(c.span),c.begin,c.end),c.normal,c.bound,support_depth,out);
    out.support_failed+=!ok;
  }
  for(int i=0;i<int(g.pieces());++i) {
    const bool ok=jacobianCertificate<3>(g.curve(i),out,requirement.onSpan(g.breaks[i+1]-g.breaks[i]));
    out.jacobian_failed+=!ok;
  }
  out.support=out.support_failed==0;out.jacobian=out.jacobian_failed==0;return out;
}
}
