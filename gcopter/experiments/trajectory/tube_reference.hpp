// SPDX-License-Identifier: MIT
#pragma once
#include "gcopter/direct_corridor.hpp"
#include <Eigen/Geometry>
#include <Eigen/Cholesky>
#include <algorithm>
#include <cstdint>
using namespace gcopter::direct;
struct TubeValue {
  double radial=0, plane=0, u=0;
  Vec3 grad_radial=Vec3::Zero(), grad_plane=Vec3::Zero();
  bool valid=true;
};
// Local inversion of n(u).(p-c(u))=0. Progress is only a root-selection hint,
// not a constraint on the physical trajectory. Multiple roots may exist.
template<class CorridorType,class Getter>
inline TubeValue referenceTubeValue(const CorridorType& corridor, const Vec3& p, double hint,Getter&& getter,uint64_t* fallbacks=nullptr) {
  double u=hint,f=0,fu=0;Section section;Vec3 normal,normal_u;
  auto plane=[&](double x) {
    section=getter(x);
    Vec3 cross=section.shape.col(0).cross(section.shape.col(1));
    Vec3 dcross=section.shape_first.col(0).cross(section.shape.col(1))+
               section.shape.col(0).cross(section.shape_first.col(1));
    normal=cross.normalized();normal_u=(dcross-normal*normal.dot(dcross))/cross.norm();
    f=normal.dot(p-section.center);
    fu=normal_u.dot(p-section.center)-normal.dot(section.first);
    return f;
  };
  for(int i=0;i<16;++i) {
    plane(u);if(std::abs(f)<1e-8)break;
    if(std::abs(fu)<1e-10)break;
    double next=std::clamp(u-std::clamp(f/fu,-.1,.1),0.,1.);
    if(next==u)break;u=next;
  }
  plane(u);
  if(std::abs(f)>1e-7 && u>0 && u<1) {
    if(fallbacks)++*fallbacks;
    // Rare fallback: choose the nearest sign-changing knot interval, then bisect.
    double best=1e10,lo=0,hi=0,fl=0;
    const auto& knots=corridor.knots();double previous=plane(knots.front());
    for(size_t i=1;i<knots.size();++i) {
      double current=plane(knots[i]);double dist=std::abs(.5*(knots[i-1]+knots[i])-hint);
      if(previous*current<=0 && dist<best){best=dist;lo=knots[i-1];hi=knots[i];fl=previous;}
      previous=current;
    }
    if(best<1e10) {
      for(int i=0;i<40;++i){u=.5*(lo+hi);double fm=plane(u);if(std::abs(fm)<1e-9)break;if(fl*fm<=0)hi=u;else{lo=u;fl=fm;}}
    }
    plane(u);
  }
  TubeValue out;out.u=u;out.plane=f;out.grad_plane=normal;
  const bool root=std::abs(f)<1e-7&&std::abs(fu)>1e-10;
  out.valid=root||u==0||u==1;
  Vec3 d=p-section.center;
  const Mat32& M=section.shape;const Mat32& Md=section.shape_first;
  Eigen::Matrix2d gram=M.transpose()*M;
  Eigen::Matrix<double,2,3> inv=gram.ldlt().solve(M.transpose());
  Eigen::Vector2d z=inv*d;
  double q=z.squaredNorm();
  if(q>1) {
    // Radius in the section plane. This also handles endpoint cap extensions.
    Vec3 tangent=d-normal*normal.dot(d);double r=tangent.norm(),sq=std::sqrt(q);
    out.radial=r*(1-1/sq);
    out.grad_radial=(1-1/sq)*tangent/r+r/(q*sq)*(inv.transpose()*z);
    if(root) {
      Eigen::Vector2d zu=gram.ldlt().solve(Md.transpose()*d-M.transpose()*section.first-
          (Md.transpose()*M+M.transpose()*Md)*z);
      double ru=-tangent.dot(section.first)/r;
      double derivative=ru*(1-1/sq)+r/(q*sq)*z.dot(zu);
      out.grad_radial-=normal*(derivative/fu);
    }
  }
  if(root){out.plane=0;out.grad_plane.setZero();}
  return out;
}

inline TubeValue tubeValue(const Corridor& corridor,const Vec3& p,double hint){return referenceTubeValue(corridor,p,hint,[&](double u){return corridor.section(u);});}
