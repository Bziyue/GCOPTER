// SPDX-License-Identifier: MIT
#include "cubic_query.hpp"
#include "cubic_optimizer.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
using namespace gcopter::direct;
using namespace gcopter::direct::native;
// Finite differences use an independently refined physical root. The fast
// query's 1e-8 plane stopping tolerance otherwise dominates a 1e-6 position
// perturbation, equally for the legacy and native implementations.
static double oracleRadial(const Geometry& g,const Vec3& p,double u) {
  for(int it=0;it<32;++it) {
    const auto s=g.section(u);
    const Vec3 n=s.shape.col(0).cross(s.shape.col(1));
    const Vec3 dn=s.shape_first.col(0).cross(s.shape.col(1))+s.shape.col(0).cross(s.shape_first.col(1));
    const double f=n.dot(p-s.center),df=dn.dot(p-s.center)-n.dot(s.first);
    const double next=std::clamp(u-f/df,0.,1.);
    if(std::abs(next-u)<1e-14){u=next;break;}u=next;
  }
  const auto s=g.section(u);const Vec3 d=p-s.center;
  const Vec3 n=s.shape.col(0).cross(s.shape.col(1)).normalized();
  const Eigen::Vector2d z=(s.shape.transpose()*s.shape).ldlt().solve(s.shape.transpose()*d);
  const double norm=z.norm();return norm>1?(d-n*n.dot(d)).norm()*(1-1/norm):0.;
}
int main(int argc,char** argv) {
  CubicBasis uneven({0,.08,.25,.66,1});std::vector<V9> controls;
  for(int i=0;i<uneven.count;++i) {
    const double u=uneven.greville(i);Mat32 m;m<<0,0,.4+.1*u,0,0,.3+.2*u;
    controls.push_back(pack(Vec3(8*u,.3*std::sin(3*u),1+.2*u),m));
  }
  const auto nonuniform=Geometry::fromCubic(uneven,controls);PreparedCubic nonuniform_query(nonuniform);
  size_t nonuniform_cursor=0;
  for(int i=0;i<=200;++i) {
    const double u=i/200.;const auto s=nonuniform.section(u);
    const auto a=nonuniform_query.firstUnchecked(u,nonuniform_cursor);
    assert((s.center-a.center).norm()<1e-12);assert((s.first-a.first).norm()<1e-10);
    assert((s.shape-a.shape).norm()<1e-12);assert((s.shape_first-a.shape_first).norm()<1e-10);
  }
  Geometry g;
  if(argc>1)g=Geometry::read(argv[1]);
  else g=CubicModel::fromSeed({{0,0,1},{2,0,1.5},{4,2,2},{4,5,2}},.4,.3,1.5).geometry();
  const auto legacy=g.quinticAdapter();PreparedCubic prepared(g);
  CubicQuery query(g);tube_query::Query reference(legacy,true);
  double value_error=0,gradient_error=0,section_error=0,finite_difference_error=0;
  std::vector<Vec3> points;std::vector<double> hints;
  size_t cursor=0;
  for(int i=0;i<2000;++i) {
    const double u=(i+.3)/2000.,angle=i*.173;
    const auto s=g.section(u);const auto a=prepared.firstUnchecked(u,cursor);
    section_error=std::max({section_error,(a.center-s.center).norm(),(a.first-s.first).norm(),(a.shape-s.shape).norm(),(a.shape_first-s.shape_first).norm()});
    const double radius=i%2?1.2:.8;
    const Vec3 p=s.center+radius*s.shape*Eigen::Vector2d(std::cos(angle),std::sin(angle));
    const double hint=std::clamp(u+.02*std::sin(i*.13),0.,1.);
    points.push_back(p);hints.push_back(hint);
    const auto actual=query(p,hint),expected=reference(p,hint);
    assert(actual.valid&&expected.valid);
    value_error=std::max({value_error,std::abs(actual.radial-expected.radial),std::abs(actual.plane-expected.plane),std::abs(actual.u-expected.u)});
    gradient_error=std::max({gradient_error,(actual.grad_radial-expected.grad_radial).norm(),(actual.grad_plane-expected.grad_plane).norm()});
    if(i%2 && i%13==0)for(int k=0;k<3;++k) {
      const Vec3 d=1e-6*Vec3::Unit(k);
      const double numerical=(oracleRadial(g,p+d,actual.u)-oracleRadial(g,p-d,actual.u))/2e-6;
      finite_difference_error=std::max(finite_difference_error,std::abs(numerical-actual.grad_radial[k]));
      assert(std::abs(numerical-actual.grad_radial[k])<1e-5);
    }
  }
  assert(section_error<1e-8);assert(value_error<1e-7);assert(gradient_error<1e-5);
  std::cout<<"section_error="<<section_error<<", value_error="<<value_error<<", gradient_error="<<gradient_error<<", finite_difference_error="<<finite_difference_error<<'\n';
  auto timed=[&](auto& q) {
    q.reset();double sum=0;const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<int(points.size());++i)sum+=q(points[i],hints[i]).radial;
    const double time=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    assert(std::isfinite(sum));return time;
  };
  for(int i=0;i<5;++i){double a,b;if(i%2){a=timed(query);b=timed(reference);}else{b=timed(reference);a=timed(query);}std::cout<<"native_ms="<<a<<", elevated_ms="<<b<<'\n';}
}
