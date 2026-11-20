// SPDX-License-Identifier: MIT
#pragma once
#include "anchors.hpp"

namespace SweepCorridor::experiments::h3 {

// Integral over the two freely extended ends. A gradient callback receives
// derivatives w.r.t. the four world-space Bezier controls of each span.
// Only spans outside the anchors and the two cut spans are visited.
template<class Gradient=std::nullptr_t>
inline double capVolume(const Model& model,const Eigen::VectorXd& x,
                        const std::array<Anchor,2>& anchors,Gradient add=nullptr) {
  constexpr bool differentiate=!std::is_same_v<Gradient,std::nullptr_t>;
  const double pi=std::acos(-1.);
  double volume=0;
  for(int end=0;end<2;++end) {
    const auto& anchor=anchors[end];
    const int first=end?anchor.span:0;
    const int last=end?model.basis.count-2:anchor.span;
    for(int span=first;span<=last;++span) {
      const double h=model.basis.breaks[span+1]-model.basis.breaks[span];
      const double cut=std::clamp((anchor.u-model.basis.breaks[span])/h,0.,1.);
      const double a=end && span==anchor.span ? cut : 0.;
      const double b=!end && span==anchor.span ? cut : 1.;
      c3::Curve<3,V9> gradient;
      if constexpr(differentiate) for(auto& g:gradient) g.setZero();
      if(b>a) {
        const auto p=model.curve(x,span);
        for(size_t q=0;q<gaussNodes.size();++q) {
          const double t=a+(b-a)*gaussNodes[q],r=1-t,weight=pi*(b-a)*gaussWeights[q];
          const std::array<double,4> basis{r*r*r,3*r*r*t,3*r*t*t,t*t*t};
          const Mat32 m=shape(basis[0]*p[0]+basis[1]*p[1]+basis[2]*p[2]+basis[3]*p[3]);
          const Vec3 tangent=3*(r*r*(p[1]-p[0])+2*r*t*(p[2]-p[1])+t*t*(p[3]-p[2])).head<3>();
          const Vec3 normal=m.col(0).cross(m.col(1));
          volume+=weight*normal.dot(tangent);
          if constexpr(differentiate) {
            const std::array<double,4> db{-3*r*r,3*r*(1-3*t),3*t*(2-3*t),3*t*t};
            Mat32 dm;
            dm.col(0)=m.col(1).cross(tangent);
            dm.col(1)=tangent.cross(m.col(0));
            const V9 shape_gradient=pack(Vec3::Zero(),dm);
            for(int j=0;j<4;++j) {
              gradient[j]+=weight*basis[j]*shape_gradient;
              gradient[j].head<3>()+=weight*db[j]*normal;
            }
          }
        }
      }
      if constexpr(differentiate) {
        // An anchor may be on a cap even when the omitted volume is zero.
        // Implicit differentiation of c(t)+M(t)z=p supplies the moving limit.
        if(span==anchor.span) {
          const Section& s=anchor.section;
          const Vec3 normal=s.shape.col(0).cross(s.shape.col(1));
          const double longitudinal=normal.dot(s.first+s.shape_first*anchor.z);
          const double factor=(end?pi:-pi)*normal.dot(s.first)/longitudinal;
          const V9 boundary=factor*pack(normal,normal*anchor.z.transpose());
          const double r=1-cut;
          const std::array<double,4> basis{r*r*r,3*r*r*cut,3*r*cut*cut,cut*cut*cut};
          for(int j=0;j<4;++j) gradient[j]+=basis[j]*boundary;
        }
        add(span,gradient);
      }
    }
  }
  return volume;
}

}
