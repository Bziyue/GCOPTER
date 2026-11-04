// SPDX-License-Identifier: MIT
#include "cubic_generator.hpp"
#include <cassert>
#include <iostream>
using namespace gcopter::direct;
using namespace gcopter::direct::native;
int main() {
  BandedSPD band(70,19);Eigen::MatrixXd dense=Eigen::MatrixXd::Zero(70,70);
  for(int i=0;i<70;++i)for(int j=std::max(0,i-19);j<=i;++j){double v=i==j?10:.01*std::sin(i*2.3+j);band.add(i,j,v);dense(i,j)=dense(j,i)=v;}
  Eigen::VectorXd rhs(70);for(int j=0;j<70;++j)rhs[j]=std::cos(j*.57);
  assert(band.factor());assert((band.solve(rhs)-dense.llt().solve(rhs)).norm()<1e-13);
  assert((band.multiply(rhs)-dense*rhs).norm()<1e-12);
  const std::vector<Vec3> seed{{-3,0,1},{0,0,1},{3,1,1}};
  auto model=CubicModel::fromSeed(seed,.4,.4,1.5);
  std::vector<detail::SupportPlane> planes;
  for(int i=0;i<int(model.basis.spans.size());++i)for(int axis=0;axis<3;++axis)
    for(double sign:{-1.,1.})planes.push_back({i,0,1,sign*Vec3::Unit(axis),8.});
  Options o;o.max_radius=3;o.max_offset=3;o.min_radius=.03;o.budget_ms=10000;o.local_range=3;
  OptimizerSettings settings;settings.check_gradients=true;
  CubicProblem problem(model,planes,6.2,o,settings);OptimizerMetrics metrics;
  auto oracle_settings=settings;oracle_settings.reference_assembly=true;
  CubicProblem oracle(model,planes,6.2,o,oracle_settings);
  Eigen::VectorXd ga,gb;BandedSPD ha,hb;
  const double fa=problem.objective(model.x,.003,&ga,&ha),fb=oracle.objective(model.x,.003,&gb,&hb);
  assert(std::abs(fa-fb)<1e-12);
  assert((ga-gb).norm()<1e-12);
  assert((ha.data-hb.data).norm()<1e-11);
  problem.checkDerivatives(model.x,metrics);
  std::cout<<"gradient "<<metrics.gradient_error<<", Hv "<<metrics.hessian_error<<'\n';
  assert(metrics.gradient_error<1e-7);assert(metrics.hessian_error<1e-7);
  auto feasible_settings=settings;feasible_settings.fraction_boundary=.99;
  CubicProblem feasible_problem(model,planes,6.2,o,feasible_settings);
  for(int trial=0;trial<32;++trial) {
    Eigen::VectorXd direction(model.x.size());
    for(int k=0;k<direction.size();++k)direction[k]=3*std::sin((k+1)*(trial+1)*.197);
    const double alpha=feasible_problem.feasibleStep(model.x,direction);
    assert(alpha>0&&alpha<=1);
    assert(std::isfinite(oracle.objective(model.x+alpha*direction,.003)));
  }
  auto natural=naturalCubic(model.geometry());const auto projected=CubicModel::project(natural).geometry();
  for(int i=0;i<100;++i){auto a=natural.section(i/99.),b=projected.section(i/99.);assert((pack(a.center,a.shape)-pack(b.center,b.shape)).norm()<1e-12);}
  ObstacleIndex map({});Bounds bounds{Vec3(-5,-5,-2),Vec3(5,5,5)};
  auto result=generateCubic(seed,map,bounds,o,settings);
  std::cout<<result.status<<" "<<result.message<<", "<<result.total_ms<<" ms\n";
  assert(result.status=="SUCCESS");assert(certify(result.corridor,result.cover).safe());
  assert(result.corridor.degree==3);assert(quality(result.corridor,6.2).c2_error<1e-8);
  assert((result.corridor.section(0).center-seed.front()).norm()<1e-12);
  assert((result.corridor.section(1).center-seed.back()).norm()<1e-12);
  settings.natural_endpoints=true;
  auto source=Geometry::fromCorridor(generate(seed,map,bounds,o).corridor);
  auto nbc=generateCubic(seed,map,bounds,o,settings,&source);
  std::cout<<"NBC "<<nbc.status<<" "<<nbc.message<<'\n';
  assert(nbc.status=="SUCCESS");
  for(double u:{0.,1.}){const auto s=nbc.corridor.section(u);assert(pack(s.second,s.shape_second).norm()<1e-6);}
}
