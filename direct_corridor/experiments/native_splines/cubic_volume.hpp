// SPDX-License-Identifier: MIT
#pragma once
#include "cubic_optimizer.hpp"

namespace gcopter::direct::native {
// Degree eight in the path parameter, cubic in the affine control variables.
// This is oriented swept volume; a globally overlapping sweep has multiplicity.
inline double sweptVolume(const CubicModel& model,const Eigen::VectorXd& x,Eigen::VectorXd* gradient=nullptr) {
  constexpr double points[5]={.046910077030668,.230765344947158,.5,.769234655052842,.953089922969332};
  constexpr double weights[5]={.118463442528095,.239314335249683,.284444444444444,.239314335249683,.118463442528095};
  const double pi=std::acos(-1.);
  std::vector<V9> controls(model.basis.count);
  for(int i=0;i<model.basis.count;++i) {
    const auto v=x.segment<5>(5*i);Eigen::Matrix2d s;s<<v[2],v[3],v[3],v[4];
    controls[i]=pack(model.reference[i]+model.frames[i]*v.head<2>(),model.frames[i]*s);
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
          const int node=span+j,offset=5*node;const auto& frame=model.frames[node];
          gradient->segment<2>(offset)+=w*derivative[j]*frame.transpose()*normal;
          (*gradient)[offset+2]+=w*basis[j]*frame.col(0).dot(first);
          (*gradient)[offset+3]+=w*basis[j]*(frame.col(1).dot(first)+frame.col(0).dot(second));
          (*gradient)[offset+4]+=w*basis[j]*frame.col(1).dot(second);
        }
      }
    }
  }
  return volume;
}
} // namespace gcopter::direct::native
