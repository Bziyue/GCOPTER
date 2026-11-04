// SPDX-License-Identifier: MIT
#pragma once
#include "CubicOptimizer.hpp"
#include <Eigen/LU>

namespace SweepCorridor::internal::cubic {
// Eliminate (u,z) from p=c(u)+M(u)z by local plane inversion. Each root and
// its implicit derivative touch only four cubic controls: no dense KKT block.
struct Anchor {
  double u=0,plane=0;
  int span=0;
  Eigen::Vector2d z=Eigen::Vector2d::Zero();
  Eigen::Vector4d weights;
  Section section;
  Eigen::Matrix<double,2,24> derivative;
};
inline Section modelSection(const CubicModel& model,const Eigen::VectorXd& x,double u,
                            int& span,Eigen::Vector4d& weights) {
  const auto& breaks=model.basis.breaks;
  span=std::min(int(breaks.size())-2,int(std::upper_bound(breaks.begin(),breaks.end(),u)-breaks.begin())-1);
  const double h=breaks[span+1]-breaks[span],t=(u-breaks[span])/h,r=1-t;
  const auto& e=model.basis.spans[span].extraction;
  weights=e.transpose()*Eigen::Vector4d(r*r*r,3*r*r*t,3*r*t*t,t*t*t);
  const Eigen::Vector4d dw=e.transpose()*Eigen::Vector4d(-3*r*r,3*r*r-6*r*t,6*r*t-3*t*t,3*t*t)/h;
  const Eigen::Vector4d ddw=e.transpose()*Eigen::Vector4d(6*r,18*t-12,6-18*t,6*t)/(h*h);
  V9 y=V9::Zero(),dy=V9::Zero(),ddy=V9::Zero();
  for(int k=0;k<4;++k){const int i=span+k;
    const V9 control=pack(model.reference[i]+x.segment<3>(6*i),model.frames[i]*matrix(x,i));
    y+=weights[k]*control;dy+=dw[k]*control;ddy+=ddw[k]*control;}
  return {y.head<3>(),dy.head<3>(),ddy.head<3>(),shape(y),shape(dy),shape(ddy)};
}
inline bool locateAnchor(const CubicModel& model,const Eigen::VectorXd& x,const Vec3& point,
                         Anchor& anchor,bool derivatives) {
  double u=std::clamp(anchor.u,0.,1.),fu=0;Vec3 n,nd,delta;
  for(int iteration=0;iteration<20;++iteration){
    anchor.section=modelSection(model,x,u,anchor.span,anchor.weights);
    const auto& s=anchor.section;
    n=s.shape.col(0).cross(s.shape.col(1));
    nd=s.shape_first.col(0).cross(s.shape.col(1))+s.shape.col(0).cross(s.shape_first.col(1));
    delta=point-s.center;const double f=n.dot(delta);fu=nd.dot(delta)-n.dot(s.first);
    anchor.plane=f/n.norm();anchor.u=u;
    if(std::abs(anchor.plane)<=1e-12)break;
    if(std::abs(fu)<1e-15)return false;
    const double next=std::clamp(u-f/fu,0.,1.);
    if(next==u)return false;u=next;
  }
  if(std::abs(anchor.plane)>1e-8||std::abs(fu)<1e-15)return false;
  const auto& s=anchor.section;const auto& m=s.shape;const auto& md=s.shape_first;
  const Eigen::Matrix2d gram=m.transpose()*m;
  if(gram.determinant()<=1e-24)return false;
  const Eigen::Matrix2d inverse=gram.inverse();
  anchor.z=inverse*m.transpose()*delta;
  if(!derivatives)return anchor.z.squaredNorm()<1;
  const Eigen::Vector2d zu=inverse*(md.transpose()*delta-m.transpose()*s.first-
      (md.transpose()*m+m.transpose()*md)*anchor.z);
  for(int k=0;k<4;++k){const Eigen::Matrix<double,9,6> j=model.jacobian(anchor.span+k)*anchor.weights[k];
    for(int d=0;d<6;++d){const Vec3 dc=j.col(d).head<3>();const Mat32 dm=shape(j.col(d));
      const Vec3 dn=dm.col(0).cross(m.col(1))+m.col(0).cross(dm.col(1));
      const double du=-(dn.dot(delta)-n.dot(dc))/fu;
      anchor.derivative.col(6*k+d)=inverse*(dm.transpose()*delta-m.transpose()*dc-
          (dm.transpose()*m+m.transpose()*dm)*anchor.z)+zu*du;
    }
  }
  return anchor.z.squaredNorm()<1;
}
inline double anchorObjective(const CubicModel& model,const Eigen::VectorXd& x,
    std::array<Anchor,2>& anchors,double mu,Eigen::VectorXd* gradient=nullptr,BandedSPD* hessian=nullptr) {
  double cost=0;
  for(int end=0;end<2;++end){auto& a=anchors[end];
    if(!locateAnchor(model,x,model.reference[end?model.basis.count-1:0],a,gradient!=nullptr))return INFINITY;
    const double slack=1-a.z.squaredNorm();cost-=mu*std::log(slack);
    if(gradient){const Eigen::Matrix<double,24,1> d=2*a.derivative.transpose()*a.z;
      gradient->segment<24>(6*a.span)+=mu/slack*d;
      if(hessian){
        // Positive Gauss-Newton curvature; nonlinear root jets are not frozen
        // in the gradient or line-search objective.
        const Eigen::Matrix<double,24,24> h=2*mu/slack*a.derivative.transpose()*a.derivative+mu/(slack*slack)*d*d.transpose();
        for(int j=0;j<24;++j)for(int k=0;k<=j;++k)hessian->add(6*a.span+j,6*a.span+k,h(j,k));
      }
    }
  }
  return anchors[0].u<anchors[1].u?cost:INFINITY;
}
// Since the endpoint frames are fixed and S is positive definite, the cap
// planes have constant normals. These are exact linear domain inequalities,
// not prescribed endpoint positions or extension lengths.
inline Vec3 inwardCapNormal(const CubicModel& model,int end) {
  const auto& f=model.frames[end?model.basis.count-1:0];
  return (end?1.:-1.)*f.col(0).cross(f.col(1)).normalized();
}
inline double capStep(const CubicModel& model,const Eigen::VectorXd& x,const Eigen::VectorXd& step) {
  double alpha=1;
  for(int end=0;end<2;++end){const int offset=end?int(x.size())-6:0;const Vec3 n=inwardCapNormal(model,end);
    const double slack=n.dot(x.segment<3>(offset)),d=n.dot(step.segment<3>(offset));
    if(d < -1e-12)alpha=std::min(alpha,std::max(0.,slack)/-d);
  }return alpha;
}
} // namespace SweepCorridor::internal::cubic
