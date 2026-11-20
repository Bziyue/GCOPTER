// SPDX-License-Identifier: MIT
#pragma once
#include "jets.hpp"

namespace SweepCorridor::experiments::h3 {
struct CoreQuality { double volume=0, bending=0; };

// Integral of (k1^2 + k2^2) dA on the lateral wall between the anchors.
// Center and shape derivatives below use the local Bezier parameter.
inline CoreQuality coreQuality(const Model& model,const Deadline& deadline) {
  static const std::array<Eigen::Vector2d,16> circle=[] {
    std::array<Eigen::Vector2d,16> points;
    for(size_t i=0;i<points.size();++i) {
      const double angle=2*std::acos(-1.)*i/points.size();
      points[i]={std::cos(angle),std::sin(angle)};
    }
    return points;
  }();
  CoreQuality result;
  for(int i=0;i<model.basis.count-1;++i) {
    deadline.check();
    const double lo=model.basis.breaks[i],h=model.basis.breaks[i+1]-lo;
    const double a=std::max(0.,(model.anchor_parameters[0]-lo)/h);
    const double b=std::min(1.,(model.anchor_parameters[1]-lo)/h);
    if(b<=a)continue;
    const auto curve=model.curve(model.x,i);
    for(size_t q=0;q<gaussNodes.size();++q) {
      const auto s=c3::evaluate<3>(curve,a+(b-a)*gaussNodes[q],1.);
      const double weight=std::acos(-1.)*(b-a)*gaussWeights[q];
      result.volume+=weight*s.shape.col(0).cross(s.shape.col(1)).dot(s.first);
      for(const auto& z:circle) {
        const Eigen::Vector2d dz(-z.y(),z.x());
        const Vec3 u=s.first+s.shape_first*z, t=s.shape*dz, normal=u.cross(t);
        const double area2=normal.squaredNorm(),area=std::sqrt(area2);
        if(!(area>0))return {0,INFINITY};
        const Vec3 n=normal/area;
        const double e=n.dot(s.second+s.shape_second*z),f=n.dot(s.shape_first*dz),g=-n.dot(s.shape*z);
        const double trace=(e*t.squaredNorm()-2*f*u.dot(t)+g*u.squaredNorm())/area2;
        const double k2=trace*trace-2*(e*g-f*f)/area2;
        result.bending+=2*weight*area*std::max(0.,k2)/circle.size();
      }
    }
  }
  return result;
}

inline void smoothCore(Model& model,const std::vector<SupportPlane>& cover,double length,
                       const OptimizerSettings& settings,JetMetrics& metrics,const Deadline& deadline) {
  Model candidate=model;
  JetMetrics trial_metrics;
  updateJets(candidate,cover,trial_metrics,deadline,settings.profile,settings.regularity(length));
  if(!trial_metrics.accepted)return;
  const auto before=coreQuality(model,deadline), after=coreQuality(candidate,deadline);
  if(after.volume>=(1-settings.smoothing_volume_tolerance)*before.volume && after.bending<before.bending) {
    model=std::move(candidate);
    metrics.core_smoothed=true;
  }
}
}
