// SPDX-License-Identifier: MIT
#pragma once
#include "minkowski_geometry.hpp"
namespace gcopter::direct::minkowski {
struct AffineShape {
  static constexpr bool needs_projection=true;
  Templates bank;
  Weights weights=Weights::Zero();
  double inner_radius;
  explicit AffineShape(const Options& settings):bank(settings) {
    for(int k=0;k<bank.count;++k) weights[k]=1./bank.count;
    // Each template is in the unit disc, hence their weighted sum is too.
    // Its support function is 1-Lipschitz in unit angle. Sampling minus half
    // the angular grid spacing gives a rigorous (floating guarded) inner disc.
    constexpr int samples=2048;
    inner_radius=INFINITY;
    for(int j=0;j<samples;++j) {
      const double angle=2*std::acos(-1.)*j/samples;
      inner_radius=std::min(inner_radius,support(V2(std::cos(angle),std::sin(angle))));
    }
    inner_radius-=std::acos(-1.)/samples+1e-12;
  }
  double support(const V2& z) const {
    return weights.dot(bank.support(z));
  }
  detail::SupportJet jet(const V2& z) const {
    detail::SupportJet out{0,0,V2::Zero(),V2::Zero()};
    const double length=z.norm();
    if(length<=1e-14) return out;
    out.radial=z/length;
    for(int k=0;k<bank.count;++k) {
      const V2 qz=bank.q[k]*z;
      const double h=std::sqrt(z.dot(qz));
      out.length+=weights[k]*h;
      out.gradient+=weights[k]/h*qz;
      out.curvature+=weights[k]*bank.q[k].determinant()*length*length/(h*h*h);
    }
    return out;
  }
  Vec3 direction(const Mat32& frame,const Vec3& delta) const {
    return SectionProjection(frame,bank,weights).direction(delta);
  }
};
Result generateAffine(const std::vector<Vec3>& seed,const ObstacleIndex& index,
                      const Bounds& bounds,const Options& options);
}  // namespace gcopter::direct::minkowski
