// SPDX-License-Identifier: MIT
#include "geometry.hpp"
#include "SplineTrajectory/SplineTrajectory.hpp"
#include <cassert>
#include <iostream>
using namespace gcopter::direct::native;
int main() {
  CubicBasis b({0,.07,.21,.39,.62,.8,1.});std::vector<V9> p(b.count);
  for(int i=0;i<b.count;++i)for(int j=0;j<9;++j)p[i][j]=std::sin(.7*i+.31*j)+i*.3;
  auto g=minimumJerk(Geometry::fromCubic(b,p));
  using Spline=SplineTrajectory::QuinticSplineND<9>;
  Spline::CoefficientMatrix values(g.breaks.size(),9);
  for(int i=0;i<int(g.breaks.size());++i){const auto s=g.section(g.breaks[i]);values.row(i)=pack(s.center,s.shape);}
  const auto a=g.section(0),z=g.section(1);
  SplineTrajectory::BoundaryConditions<9> boundary(pack(a.first,a.shape_first),pack(a.second,a.shape_second),
                                                 pack(z.first,z.shape_first),pack(z.second,z.shape_second));
  Spline spline(g.breaks,values,boundary);assert(spline.isValid());
  int i=0;double error=0;
  for(auto piece:spline.polynomial()) {
    for(int j=0;j<=10;++j){const double t=piece.duration()*j/10.;const auto q=g.section(g.breaks[i]+t);
      error=std::max(error,(piece.evaluateLocal(t)-pack(q.center,q.shape)).norm());}
    ++i;
  }
  assert(error<1e-10);
  assert(std::abs(spline.energy()-quality(g,1).jerk_energy)<1e-8*std::max(1.,spline.energy()));
  // Free endpoint velocity AND acceleration imply zero third and fourth
  // derivatives at both ends, rather than zero first/second derivatives.
  for(int i:{0,int(g.pieces())-1}) {
    auto c=g.curve<5>(i);const double h=g.breaks[i+1]-g.breaks[i];
    const int k=i==0?0:2;
    const V9 third=60/std::pow(h,3)*(c[k+3]-3*c[k+2]+3*c[k+1]-c[k]);
    const int l=i==0?0:1;
    const V9 fourth=120/std::pow(h,4)*(c[l+4]-4*c[l+3]+6*c[l+2]-4*c[l+1]+c[l]);
    assert(third.norm()<1e-7);assert(fourth.norm()<2e-6);
  }
  std::cout<<"free-jet variational solution agrees with migrated QuinticSplineND<9>: "<<error<<'\n';
}
