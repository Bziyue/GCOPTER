// SPDX-License-Identifier: MIT
#pragma once
#include "cubic_optimizer.hpp"
#include <Eigen/Eigenvalues>

namespace gcopter::direct::native {
inline std::vector<double> adaptiveBreaks(const CubicModel& model,int mode) {
  const double angle=mode==1?.06:.15,ratio=mode==1?1.2:1.6,floor=mode==1?.5:.3;
  const auto& breaks=model.basis.breaks;
  std::vector<double> selected{breaks.front()};bool removed=false;
  for(int k=1;k<int(breaks.size())-1;++k) {
    bool merge=!removed&&k>1&&k+3<int(breaks.size());
    if(merge) {
      const Vec3 normal=model.frames[k-1].col(0).cross(model.frames[k-1].col(1));
      double minimum=INFINITY,maximum=0;
      for(int i=k-1;i<=k+3;++i) {
        merge&=(normal-model.frames[i].col(0).cross(model.frames[i].col(1))).norm()<angle;
        const auto s=model.x.segment<3>(5*i+2);
        const double mean=.5*(s[0]+s[2]),spread=std::hypot(.5*(s[0]-s[2]),s[1]);
        minimum=std::min(minimum,mean-spread);maximum=std::max(maximum,mean+spread);
      }
      merge&=minimum>floor&&maximum<ratio*minimum;
    }
    removed=merge;if(!merge)selected.push_back(breaks[k]);
  }
  selected.push_back(breaks.back());return selected;
}
// Exact insertion operates on the nine world channels. Polar reconstruction
// gives each new matrix an orthonormal frame and a symmetric positive shape.
// The caller checks nodal bounds for its optimization model after insertion.
inline CubicModel insertKnot(const CubicModel& model,double parameter) {
  auto breaks=model.basis.breaks;
  auto position=std::upper_bound(breaks.begin(),breaks.end(),parameter);
  if(position==breaks.begin()||position==breaks.end()||*(position-1)==parameter)
    throw std::invalid_argument("insert a new simple interior knot");
  const int span=int(position-breaks.begin())-1,k=span+3;
  breaks.insert(position,parameter);CubicModel out(breaks);out.x.setZero();
  std::vector<V9> old(model.basis.count);
  for(int i=0;i<model.basis.count;++i) {
    const auto x=model.x.segment<5>(5*i);Eigen::Matrix2d s;s<<x[2],x[3],x[3],x[4];
    old[i]=pack(model.reference[i]+model.frames[i]*x.head<2>(),model.frames[i]*s);
  }
  for(int i=0;i<out.basis.count;++i) {
    V9 y;
    if(i<=k-3)y=old[i];
    else if(i<=k) {
      const double alpha=(parameter-model.basis.knots[i])/(model.basis.knots[i+3]-model.basis.knots[i]);
      y=(1-alpha)*old[i-1]+alpha*old[i];
    }else y=old[i-1];
    const Mat32 m=shape(y);Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen(m.transpose()*m);
    if(eigen.eigenvalues()[0]<=1e-14)throw std::runtime_error("inserted control shape is rank deficient");
    const Eigen::Matrix2d s=eigen.eigenvectors()*eigen.eigenvalues().cwiseSqrt().asDiagonal()*eigen.eigenvectors().transpose();
    out.reference[i]=y.head<3>();out.frames[i]=m*s.inverse();
    out.x.segment<3>(5*i+2)<<s(0,0),s(0,1),s(1,1);
  }
  return out;
}
} // namespace gcopter::direct::native
