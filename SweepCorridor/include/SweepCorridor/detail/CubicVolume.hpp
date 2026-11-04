// SPDX-License-Identifier: MIT
#pragma once
#include "CubicOptimizer.hpp"

namespace SweepCorridor::internal::cubic {
// Degree eight in the path parameter, cubic in the affine control variables.
// This is oriented swept volume; a globally overlapping sweep has multiplicity.
inline double sweptVolume(const CubicModel& model,const Eigen::VectorXd& x,Eigen::VectorXd* gradient=nullptr) {
  constexpr double points[5]={.046910077030668,.230765344947158,.5,.769234655052842,.953089922969332};
  constexpr double weights[5]={.118463442528095,.239314335249683,.284444444444444,.239314335249683,.118463442528095};
  const double pi=std::acos(-1.);
  std::vector<V9> controls(model.basis.count);
  for(int i=0;i<model.basis.count;++i) {
    const auto v=x.segment<6>(6*i);Eigen::Matrix2d s;s<<v[3],v[4],v[4],v[5];
    controls[i]=pack(model.reference[i]+v.head<3>(),model.frames[i]*s);
  }
  if(gradient)gradient->setZero(x.size());
  double volume=0;
  for(int span=0;span<int(model.basis.spans.size());++span) {
    const auto& extraction=model.basis.spans[span].extraction;
    const double h=model.basis.breaks[span+1]-model.basis.breaks[span];
    for(int q=0;q<5;++q) {
      const double t=points[q],r=1-t;
      Eigen::Vector4d b,db;b<<r*r*r,3*r*r*t,3*r*t*t,t*t*t;
      db<<-3*r*r,3*r*r-6*r*t,6*r*t-3*t*t,3*t*t;
      const Eigen::Vector4d basis=extraction.transpose()*b,derivative=extraction.transpose()*db/h;
      V9 y=V9::Zero();Vec3 tangent=Vec3::Zero();
      for(int j=0;j<4;++j){y+=basis[j]*controls[span+j];tangent+=derivative[j]*controls[span+j].head<3>();}
      const auto m=shape(y);const Vec3 normal=m.col(0).cross(m.col(1));
      const double w=pi*h*weights[q];volume+=w*normal.dot(tangent);
      if(gradient) {
        const Vec3 first=m.col(1).cross(tangent),second=tangent.cross(m.col(0));
        for(int j=0;j<4;++j) {
          const int node=span+j,offset=6*node;const auto& frame=model.frames[node];
          gradient->segment<3>(offset)+=w*derivative[j]*normal;
          (*gradient)[offset+3]+=w*basis[j]*frame.col(0).dot(first);
          (*gradient)[offset+4]+=w*basis[j]*(frame.col(1).dot(first)+frame.col(0).dot(second));
          (*gradient)[offset+5]+=w*basis[j]*frame.col(1).dot(second);
        }
      }
    }
  }
  return volume;
}
// Exact cubic change along a control-space direction. Compute all coefficients
// in one quadrature pass instead of evaluating the two displaced geometries.
inline Vec3 volumeLine(const CubicModel& model,const Eigen::VectorXd& x,const Eigen::VectorXd& direction) {
  constexpr double points[5]={.046910077030668,.230765344947158,.5,.769234655052842,.953089922969332};
  constexpr double weights[5]={.118463442528095,.239314335249683,.284444444444444,.239314335249683,.118463442528095};
  std::vector<V9> controls(model.basis.count),deltas(model.basis.count);
  for(int i=0;i<model.basis.count;++i){controls[i]=pack(model.reference[i]+x.segment<3>(6*i),model.frames[i]*matrix(x,i));
    deltas[i]=pack(direction.segment<3>(6*i),model.frames[i]*matrix(direction,i));}
  Vec3 result=Vec3::Zero();
  for(int span=0;span<int(model.basis.spans.size());++span){const auto& e=model.basis.spans[span].extraction;
    const double h=model.basis.breaks[span+1]-model.basis.breaks[span];
    for(int q=0;q<5;++q){const double t=points[q],r=1-t;
      const Eigen::Vector4d b=e.transpose()*Eigen::Vector4d(r*r*r,3*r*r*t,3*r*t*t,t*t*t);
      const Eigen::Vector4d db=e.transpose()*Eigen::Vector4d(-3*r*r,3*r*r-6*r*t,6*r*t-3*t*t,3*t*t)/h;
      V9 y=V9::Zero(),dy=V9::Zero();Vec3 tangent=Vec3::Zero(),dtangent=Vec3::Zero();
      for(int j=0;j<4;++j){y+=b[j]*controls[span+j];dy+=b[j]*deltas[span+j];
        tangent+=db[j]*controls[span+j].head<3>();dtangent+=db[j]*deltas[span+j].head<3>();}
      const auto m=shape(y),d=shape(dy);const Vec3 n=m.col(0).cross(m.col(1));
      const Vec3 dn=d.col(0).cross(m.col(1))+m.col(0).cross(d.col(1)),ddn=d.col(0).cross(d.col(1));
      result+=std::acos(-1.)*h*weights[q]*Vec3(n.dot(dtangent)+dn.dot(tangent),dn.dot(dtangent)+ddn.dot(tangent),ddn.dot(dtangent));
    }
  }return result;
}
} // namespace SweepCorridor::internal::cubic
