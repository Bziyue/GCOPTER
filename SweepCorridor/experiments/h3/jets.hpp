// SPDX-License-Identifier: MIT
#pragma once
#include "anchors.hpp"

namespace SweepCorridor::experiments::h3 {
struct JetMetrics {
  int accepted=0,rejected=0,joint_steps=0,joint_backtracks=0,joint_groups=0,joint_supports=0;
  double milliseconds=0,fraction_sum=0,joint_setup_ms=0,joint_assembly_ms=0,joint_solve_ms=0,joint_line_ms=0,joint_refresh_ms=0;
  int end_steps=0,end_trials=0;
  double end_ms=0,end_volume_gain=0;
  int refresh_attempts=0,refresh_spans=0,refresh_failures=0,refresh_cache_hits=0;
  double refresh_ms=0,local_setup_ms=0,local_joint_ms=0;
  int joint_budget_skips=0;
};
struct JointFeedback {
  double volume=0;
  struct Node {double gain=0,reward=0,decrement=0,pressure=0;};
  std::vector<Node> nodes;
};

// Nine channels share a scalar tridiagonal factorization. The solution is a
// proposal only; safety clipping below generally leaves a C1 Hermite spline.
inline std::vector<V9> naturalJets(const Model& m){
  const int n=m.basis.count;
  std::vector<V9> y(n),z(n,V9::Zero()),d(n);std::vector<double> upper(n,0);
  for(int i=0;i<n;++i)y[i]=m.node(m.x,i);
  for(int i=1;i<n-1;++i){const double a=m.basis.breaks[i]-m.basis.breaks[i-1],b=m.basis.breaks[i+1]-m.basis.breaks[i];
    const double diagonal=2*(a+b)-a*upper[i-1];upper[i]=b/diagonal;
    z[i]=(6*((y[i+1]-y[i])/b-(y[i]-y[i-1])/a)-a*z[i-1])/diagonal;}
  for(int i=n-2;i>0;--i)z[i]-=upper[i]*z[i+1];
  for(int i=0;i<n-1;++i){const double h=m.basis.breaks[i+1]-m.basis.breaks[i];d[i]=(y[i+1]-y[i])/h-h*(2*z[i]+z[i+1])/6;}
  const double h=m.basis.breaks[n-1]-m.basis.breaks[n-2];d[n-1]=(y[n-1]-y[n-2])/h+h*(z[n-2]+2*z[n-1])/6;
  return d;
}
inline double patchBending(const c3::Curve<3,V9>& p,double h){
  const V9 a=6*(p[2]-2*p[1]+p[0]),b=6*(p[3]-2*p[2]+p[1]);
  return (a.squaredNorm()+a.dot(b)+b.squaredNorm())/(3*h*h*h);
}

// A knot jet affects exactly B2 of its left span and B1 of its right span.
// Clip against the reused support planes (including subdivided intervals),
// certify only these two patches, then update the two anchor roots.
inline void updateJets(Model& m,const std::vector<internal::detail::SupportPlane>& cover,
    JetMetrics& metrics,const Deadline& deadline,bool profile){
  StageTimer timer(profile);const auto target=naturalJets(m);auto g=m.geometry();
  std::vector<std::vector<int>> planes(g.pieces());for(int k=0;k<int(cover.size());++k)planes[cover[k].span].push_back(k);
  std::array<Anchor,2> anchors;for(int end=0;end<2;++end)anchors[end].u=m.anchor_parameters[end];
  for(int node=0;node<m.basis.count;++node){
    deadline.check();const V9 delta=target[node]-m.jets[node];
    const int begin=std::max(0,node-1),end=std::min(node+1,m.basis.count-1);
    std::array<c3::Curve<3,V9>,2> before,change;double alpha=1,old_energy=0;
    for(int span=begin;span<end;++span){const int local=span-begin;const double h=m.basis.breaks[span+1]-m.basis.breaks[span];
      before[local]=g.curve(span);for(auto& v:change[local])v.setZero();
      change[local][span==node?1:2]=(span==node?1.:-1.)*(h/3)*delta;
      old_energy+=patchBending(before[local],h);
      for(int id:planes[span]){const auto& p=cover[id];const auto a=c3::restrictCurve<3>(before[local],p.begin,p.end),b=c3::restrictCurve<3>(change[local],p.begin,p.end);
        for(int j=0;j<(p.begin==p.end?1:4);++j){const Vec3 v(p.bound-p.normal.dot(a[j].head<3>()),p.normal.dot(a[j].segment<3>(3)),p.normal.dot(a[j].tail<3>()));
          const Vec3 d(-p.normal.dot(b[j].head<3>()),p.normal.dot(b[j].segment<3>(3)),p.normal.dot(b[j].tail<3>()));
          const double A=d[0]*d[0]-d.tail<2>().squaredNorm(),B=2*(v[0]*d[0]-v.tail<2>().dot(d.tail<2>())),C=v[0]*v[0]-v.tail<2>().squaredNorm();
          if((A*alpha+B)*alpha+C>0&&v[0]+alpha*d[0]>0)continue;
          if(d[0]<0)alpha=std::min(alpha,.99*-v[0]/d[0]);alpha=std::min(alpha,.99*c3::positiveRoot(A,B,C));
        }
      }
    }
    const V9 old=m.jets[node];bool accepted=false;
    for(int trial=0;trial<16&&alpha>1e-9;++trial,alpha*=.5){bool valid=true;double energy=0;
      for(int span=begin;span<end;++span){auto p=before[span-begin];for(int j=0;j<4;++j)p[j]+=alpha*change[span-begin][j];
        energy+=patchBending(p,m.basis.breaks[span+1]-m.basis.breaks[span]);c3::Certificate cert;
        if(!c3::jacobianCertificate<3>(p,cert)){valid=false;break;}}
      if(!valid||energy>old_energy+1e-10*(1+old_energy))continue;
      m.jets[node]=old+alpha*delta;auto trial_anchors=anchors;
      if(!std::isfinite(anchorObjective(m,m.x,trial_anchors,1.))){m.jets[node]=old;continue;}
      for(int span=begin;span<end;++span)for(int j=0;j<4;++j)g.controls[4*span+j]=before[span-begin][j]+alpha*change[span-begin][j];
      anchors=trial_anchors;accepted=true;break;
    }
    if(accepted){++metrics.accepted;metrics.fraction_sum+=alpha;}else{++metrics.rejected;m.jets[node]=old;}
  }
  for(int end=0;end<2;++end)m.anchor_parameters[end]=anchors[end].u;
  m.cacheQuadrature();metrics.milliseconds+=timer.elapsed();
}
}
