// SPDX-License-Identifier: MIT
#pragma once
#include "jets.hpp"

namespace SweepCorridor::experiments::h3 {
using M9=Eigen::Matrix<double,9,9>;
inline Eigen::Matrix3d skew(const Vec3& x){
  Eigen::Matrix3d a;a<<0,-x.z(),x.y(),x.z(),0,-x.x(),-x.y(),x.x(),0;return a;
}
// A single shared jet adds b(u)*delta to all nine geometry channels.
// The cubic volume term integrates to det(delta)*[b^3]/3 = 0: b is zero
// at both ends of each Hermite span. The local volume model is EXACT quadratic.
struct HermitePatch {
  int begin,end,node;
  double scale;
  std::array<c3::Curve<3,V9>,2> curves;
  std::array<c3::Curve<3,double>,2> basis;
  HermitePatch(const Model& m,const Geometry& g,int i)
      :begin(std::max(0,i-1)),end(std::min(i+1,m.basis.count-1)),node(i),scale(3/m.basis.weight(i)){
    for(int span=begin;span<end;++span){const int k=span-begin;curves[k]=g.curve(span);basis[k].fill(0);
      basis[k][span==node?1:2]=(span==node?1.:-1.)*(m.basis.breaks[span+1]-m.basis.breaks[span])*scale/3;}
  }
};
struct JetPatch : HermitePatch {
  V9 volume_gradient=V9::Zero(),energy_gradient=V9::Zero(),energy_diagonal=V9::Zero();
  M9 volume_hessian=M9::Zero();
  struct Support {Vec3 value,normal;};
  std::vector<Support> supports;
  JetPatch(const Model& m,const Geometry& g,int i,double length,double energy_weight)
      :HermitePatch(m,g,i){
    constexpr double t[5]={.046910077030668,.230765344947158,.5,.769234655052842,.953089922969332};
    constexpr double w[5]={.118463442528095,.239314335249683,.284444444444444,.239314335249683,.118463442528095};
    for(int span=begin;span<end;++span){const int k=span-begin;const double h=m.basis.breaks[span+1]-m.basis.breaks[span];
      const auto& p=curves[k];const auto& b=basis[k];
      const V9 d0=6*(p[2]-2*p[1]+p[0])/(h*h),d1=6*(p[3]-2*p[2]+p[1])/(h*h);
      const double a=6*(b[2]-2*b[1]+b[0])/(h*h),c=6*(b[3]-2*b[2]+b[1])/(h*h);
      V9 weight;weight.head<3>().setConstant(OptimizerSettings::center_energy);weight.tail<6>().setConstant(OptimizerSettings::shape_energy);
      weight*=energy_weight*h/(3*length*length*length);
      energy_gradient+=weight.cwiseProduct((2*a+c)*d0+(a+2*c)*d1);
      energy_diagonal+=2*(a*a+a*c+c*c)*weight;
      for(int q=0;q<5;++q){const double u=t[q],r=1-u;
        const double s=b[1]*3*r*r*u+b[2]*3*r*u*u;
        const double ds=(b[1]*3*r*(1-3*u)+b[2]*3*u*(2-3*u))/h;
        const auto section=c3::evaluate<3>(p,u,h);const auto& a=section.shape;const auto& v=section.first;
        const double factor=std::acos(-1.)*h*w[q];const Vec3 n=a.col(0).cross(a.col(1));
        volume_gradient.head<3>()+=factor*ds*n;
        volume_gradient.segment<3>(3)+=factor*s*a.col(1).cross(v);
        volume_gradient.tail<3>()+=factor*s*v.cross(a.col(0));
        volume_hessian.block<3,3>(0,3)-=factor*ds*s*skew(a.col(1));
        volume_hessian.block<3,3>(0,6)+=factor*ds*s*skew(a.col(0));
        volume_hessian.block<3,3>(3,6)-=factor*s*s*skew(v);
      }
    }
    volume_hessian.block<3,3>(3,0)=volume_hessian.block<3,3>(0,3).transpose();
    volume_hessian.block<3,3>(6,0)=volume_hessian.block<3,3>(0,6).transpose();
    volume_hessian.block<3,3>(6,3)=volume_hessian.block<3,3>(3,6).transpose();
  }
  double volumeDelta(const V9& d)const{return volume_gradient.dot(d)+.5*d.dot(volume_hessian*d);}
  double energyDelta(const V9& d)const{return energy_gradient.dot(d)+.5*energy_diagonal.dot(d.cwiseProduct(d));}
  void addSupports(const std::vector<internal::detail::SupportPlane>& cover,const std::vector<std::vector<int>>& ids){
    for(int span=begin;span<end;++span)for(int id:ids[span]){const auto& p=cover[id];
      const auto a=c3::restrictCurve<3>(curves[span-begin],p.begin,p.end);
      const auto b=c3::restrictCurve<3>(basis[span-begin],p.begin,p.end);
      for(int j=0;j<(p.begin==p.end?1:4);++j)if(std::abs(b[j])>1e-15)
        supports.push_back({Vec3(p.bound-p.normal.dot(a[j].head<3>()),p.normal.dot(a[j].segment<3>(3)),p.normal.dot(a[j].tail<3>())),b[j]*p.normal});
    }
  }
  double objective(const V9& d,double volume,double mu,V9* gradient=nullptr,M9* hessian=nullptr)const{
    const double total=volume+volumeDelta(d);if(total<=0)return INFINITY;
    double cost=energyDelta(d)-std::log(total/volume);
    if(gradient)*gradient=energy_gradient+energy_diagonal.cwiseProduct(d)-(volume_gradient+volume_hessian*d)/total;
    if(hessian){*hessian=energy_diagonal.asDiagonal();const V9 vg=(volume_gradient+volume_hessian*d)/total;hessian->noalias()+=vg*vg.transpose();}
    for(const auto& p:supports){const Vec3& n=p.normal;
      const Vec3 v=p.value+Vec3(-n.dot(d.head<3>()),n.dot(d.segment<3>(3)),n.dot(d.tail<3>()));
      const double slack=v[0]*v[0]-v.tail<2>().squaredNorm();if(v[0]<=0||slack<=0)return INFINITY;
      cost-=mu*std::log(slack);
      if(gradient){V9 a;a<<v[0]*n,v[1]*n,v[2]*n;const double f=2*mu/slack;*gradient+=f*a;
        if(hessian){hessian->noalias()+=(2*f/slack)*a*a.transpose();const Eigen::Matrix3d gram=f*n*n.transpose();
          hessian->topLeftCorner<3,3>()-=gram;hessian->block<3,3>(3,3)+=gram;hessian->bottomRightCorner<3,3>()+=gram;}
      }
    }return cost;
  }
  c3::Curve<3,V9> curve(int span,const V9& d)const{
    auto p=curves[span-begin];for(int j=0;j<4;++j)p[j]+=basis[span-begin][j]*d;return p;
  }
};

inline void optimizeVolumeJets(Model& m,const std::vector<internal::detail::SupportPlane>& cover,
    double length,const OptimizerSettings& settings,JetMetrics& metrics,const Deadline& deadline){
  StageTimer timer(settings.profile);auto geometry=m.geometry();double volume=sweptVolume(m,m.x);
  std::vector<std::vector<int>> planes(geometry.pieces());for(int k=0;k<int(cover.size());++k)planes[cover[k].span].push_back(k);
  std::array<Anchor,2> anchors;for(int end=0;end<2;++end)anchors[end].u=m.anchor_parameters[end];
  anchorObjective(m,m.x,anchors,1.);
  const double mu=1e-5/m.basis.count;
  for(int node=0;node<m.basis.count;++node){deadline.check();
    JetPatch patch(m,geometry,node,length,settings.volume_jet_energy);patch.addSupports(cover,planes);
    const V9 old=m.jets[node];V9 d=V9::Zero();bool changed=false;
    const bool touches_anchor=(anchors[0].span>=patch.begin&&anchors[0].span<patch.end)||
                             (anchors[1].span>=patch.begin&&anchors[1].span<patch.end);
    for(int iteration=0;iteration<settings.volume_jet_steps;++iteration){
      V9 gradient;M9 hessian;const double cost=patch.objective(d,volume,mu,&gradient,&hessian);
      if(!std::isfinite(cost))break;
      const V9 scale=hessian.diagonal().cwiseMax(1e-20).cwiseSqrt().cwiseInverse();
      M9 balanced=scale.asDiagonal()*hessian*scale.asDiagonal();balanced.diagonal().array()+=1e-8;
      Eigen::LLT<M9> factor(balanced);if(factor.info()!=Eigen::Success)break;
      const V9 step=scale.cwiseProduct(factor.solve(-scale.cwiseProduct(gradient)));
      const double decrease=gradient.dot(step);if(decrease>=-1e-10)break;
      bool accepted=false;
      for(double alpha=1;alpha>=1./4096;alpha*=.5){const V9 next=d+alpha*step;
        if(patch.objective(next,volume,mu)>cost+1e-4*alpha*decrease)continue;
        bool regular=true;for(int span=patch.begin;span<patch.end;++span){c3::Certificate certificate;
          if(!c3::jacobianCertificate<3>(patch.curve(span,next),certificate)){regular=false;break;}}
        if(!regular)continue;
        m.jets[node]=old+patch.scale*next;auto trial=anchors;
        if(touches_anchor&&!std::isfinite(anchorObjective(m,m.x,trial,1.))){m.jets[node]=old+patch.scale*d;continue;}
        anchors=trial;d=next;accepted=changed=true;break;
      }
      if(!accepted)break;
    }
    if(changed){volume+=patch.volumeDelta(d);for(int span=patch.begin;span<patch.end;++span){const auto p=patch.curve(span,d);
        std::copy(p.begin(),p.end(),geometry.controls.begin()+4*span);}}
    else m.jets[node]=old;
  }
  for(int end=0;end<2;++end)m.anchor_parameters[end]=anchors[end].u;
  m.cacheQuadrature();metrics.milliseconds+=timer.elapsed();
}
} // namespace SweepCorridor::experiments::h3
