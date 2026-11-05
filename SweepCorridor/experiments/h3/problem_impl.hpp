// SPDX-License-Identifier: MIT
#pragma once
#include "problem.hpp"
namespace SweepCorridor::experiments::h3 {
inline double Problem::objective(const Eigen::VectorXd& x,double mu,Eigen::VectorXd* g,
                               BlockTridiagonal* h,bool evaluate_cost)const {
  const bool derivatives=g,curvature=h;
  const auto qx=quadratic.multiply(x);
  double cost=evaluate_cost?.5*x.dot(qx)+linear.dot(x):0;
  if(derivatives)*g=qx+linear;
  if(curvature){
    // Preserve allocated factor/scale buffers across Newton steps.
    h->assign(quadratic);
  }
  auto logdet=[&](int i,const Eigen::Matrix2d& p,double weight,double sign) {
    const double det=p.determinant();if(p(0,0)<=0 || det<=0)return false;
    if(evaluate_cost)cost-=weight*std::log(det);
    if(derivatives) {
      const double a=p(1,1)/det,b=-p(0,1)/det,c=p(0,0)/det;
      g->segment<3>(6*i+3)-=weight*sign*Vec3(a,2*b,c);
      if(curvature) {
        Eigen::Matrix3d v;v<<a*a,2*a*b,b*b,2*a*b,2*(a*c+b*b),2*b*c,b*b,2*b*c,c*c;
        for(int j=0;j<3;++j)for(int k=0;k<=j;++k)h->add(6*i+3+j,6*i+3+k,weight*v(j,k));
      }
    }
    return true;
  };
  for(int i=0;i<model.basis.count;++i) {
    const auto s=matrix(x,i);
    if(!logdet(i,s,settings.area_weight*model.basis.weight(i),1) ||
       !logdet(i,s-options.min_radius*Eigen::Matrix2d::Identity(),mu,1) ||
       !logdet(i,options.max_radius*Eigen::Matrix2d::Identity()-s,mu,-1))return INFINITY;
    const Vec3 d=x.segment<3>(6*i);const double gap=options.max_offset*options.max_offset-d.squaredNorm();
    if(gap<=0)return INFINITY;
    if(evaluate_cost)cost-=mu*std::log(gap);
    if(derivatives) {
      g->segment<3>(6*i)+=2*mu/gap*d;
      if(curvature) {
        const Eigen::Matrix3d v=2*mu/gap*Eigen::Matrix3d::Identity()+4*mu/(gap*gap)*d*d.transpose();
        for(int j=0;j<3;++j)for(int k=0;k<=j;++k)h->add(6*i+j,6*i+k,v(j,k));
      }
    }
  }
  for(const auto& group:geometry->groups) {
#ifndef H3_WORLD_ASSEMBLY
    if(curvature&&group.count==1){
      const bool feasible=evaluate_cost?group.singleJets<true>(x,mu,*g,*h,cost):group.singleJets<false>(x,mu,*g,*h,cost);
      if(!feasible)return INFINITY;
      continue;
    }
#endif
    const V9 y=group.geometry(x);
    V9 gy;Eigen::Matrix<double,9,9> hy;
    if(derivatives)gy.setZero();
    if(curvature) {
      hy.setZero();
      const bool feasible=evaluate_cost?c3::exactSupportJets<true>(group,y,mu,gy,hy,cost):
                                        c3::exactSupportJets<false>(group,y,mu,gy,hy,cost);
      if(!feasible)return INFINITY;
      group.accumulate(gy,hy,*g,*h);continue;
    }
    for(size_t id=0;id<group.retained;++id) {
      const auto& support=group.supports[id];const Vec3 v=support.project(y);
      const double slack=v[0]*v[0]-v.tail<2>().squaredNorm();
      if(v[0]<=0||slack<=0)return INFINITY;
      if(evaluate_cost)cost-=mu*std::log(slack);
      if(derivatives)support.addGradient((2*mu/slack)*v,gy);
    }
    if(derivatives)group.accumulateGradient(gy,*g);
  }
  // Keep longitudinal spacing away from collapse even after the feasibility
  // barrier is reduced. This reuses the same local Lorentz-cone derivatives.
  const double cut_mu=mu+settings.progress_weight/model.basis.count;
  for(const auto& c:cuts) {
    const Vec3 v=c.value(x);
    const double slack=v[0]*v[0]-v.tail<2>().squaredNorm();
    if(v[0]<=0||slack<=0)return INFINITY;
    if(evaluate_cost)cost-=cut_mu*std::log(slack);
    if(curvature)c.accumulate(v,cut_mu,slack,*g,*h);
    else if(derivatives) {
      const Vec3 signed_v(-v[0],v[1],v[2]);
      g->segment<12>(6*c.node).noalias()+=(2*cut_mu/slack)*c.map.transpose()*signed_v;
    }
  }
  return cost;
}
inline Problem::LineModel Problem::prepareLine(const Eigen::VectorXd& x,
    const Eigen::VectorXd& direction,double mu)const {
  LineModel line;line.terms.reserve(geometry->count+cuts.size()+4*model.basis.count);
  line.linear=direction.dot(quadratic.multiply(x)+linear);
  line.quadratic=.5*direction.dot(quadratic.multiply(direction));
  double limit=INFINITY;
  auto append=[&](double a,double b,double c,double weight) {
    line.terms.push_back({a/c,b/c,weight});
  };
  auto boundary=[&](double a,double b,double c,double value,double derivative) {
    const double alpha=std::min(1.,limit);
    if((a*alpha+b)*alpha+c>0&&value+alpha*derivative>0)return;
    if(derivative<0)limit=std::min(limit,-value/derivative);
    limit=std::min(limit,c3::positiveRoot(a,b,c));
  };
  auto spd=[&](const Eigen::Matrix2d& s,const Eigen::Matrix2d& d,double weight,bool constrain) {
    const double a=d.determinant(),b=s(0,0)*d(1,1)+s(1,1)*d(0,0)-2*s(0,1)*d(0,1),c=s.determinant();
    append(a,b,c,weight);
    if(constrain)boundary(a,b,c,s(0,0),d(0,0));
  };
  for(int i=0;i<model.basis.count;++i) {
    const Eigen::Matrix2d s=c3::shapeMatrix(x,6*i),d=c3::shapeMatrix(direction,6*i);
    spd(s,d,settings.area_weight*model.basis.weight(i),false);
    spd(s-options.min_radius*Eigen::Matrix2d::Identity(),d,mu,true);
    spd(options.max_radius*Eigen::Matrix2d::Identity()-s,-d,mu,true);
    const auto c=x.segment<3>(6*i),v=direction.segment<3>(6*i);
    const double a=-v.squaredNorm(),b=-2*c.dot(v),gap=options.max_offset*options.max_offset-c.squaredNorm();
    append(a,b,gap,mu);boundary(a,b,gap,1,0);
  }
  auto cone=[&](const Vec3& v,const Vec3& d,double weight) {
    const double a=d[0]*d[0]-d.tail<2>().squaredNorm();
    const double b=2*(v[0]*d[0]-v.tail<2>().dot(d.tail<2>()));
    const double c=v[0]*v[0]-v.tail<2>().squaredNorm();
    boundary(a,b,c,v[0],d[0]);
    append(a,b,c,weight);
  };
  for(const auto& group:geometry->groups) {
    const V9 y=group.geometry(x);
    const V9 d=group.direction(direction);
    for(size_t j=0;j<group.retained;++j) {
      const auto& p=group.supports[j];
      const Vec3 v=p.project(y);
      cone(v,p.projectDirection(d),mu);
    }
  }
  for(const auto& c:cuts) {
    const Vec3 v=c.value(x),d=c.map*direction.segment<12>(6*c.node);
    cone(v,d,mu+settings.progress_weight/model.basis.count);
  }
  line.step=std::min(1.,settings.fraction_boundary*limit);
  return line;
}

}
