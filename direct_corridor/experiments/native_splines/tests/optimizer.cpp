// SPDX-License-Identifier: MIT
#include "cubic_generator.hpp"
#include "cubic_volume.hpp"
#include <cassert>
#include <iostream>
#include <set>
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
  for(int i=0;i<int(model.basis.spans.size());++i) {
    planes.push_back({i,0,1,Vec3(1,2,3).normalized(),8.});
    planes.push_back({i,.13,.83,Vec3(-2,3,1).normalized(),8.});
  }
  Options o;o.max_radius=3;o.max_offset=3;o.min_radius=.03;o.budget_ms=10000;o.local_range=3;
  OptimizerSettings settings;settings.check_gradients=true;
  CubicProblem problem(model,planes,6.2,o,settings);OptimizerMetrics metrics;
  CubicProblem shared(model,planes,6.2,o,settings,problem.energy);
  assert(&shared.quadratic==&problem.quadratic);
  assert(shared.objective(model.x,.003)==problem.objective(model.x,.003));
  std::set<int> group_sizes;for(const auto& group:problem.groups)group_sizes.insert(group.count);
  assert(group_sizes==std::set<int>({1,2,3,4}));
  auto oracle_settings=settings;oracle_settings.reference_assembly=true;
  CubicProblem oracle(model,planes,6.2,o,oracle_settings);
  Eigen::VectorXd ga,gb;BandedSPD ha,hb;
  const double fa=problem.objective(model.x,.003,&ga,&ha),fb=oracle.objective(model.x,.003,&gb,&hb);
  assert(std::abs(fa-fb)<1e-12);
  assert((ga-gb).norm()<1e-12);
  assert((ha.data-hb.data).norm()<1e-11);
  auto full_tensor_settings=settings;full_tensor_settings.tensor_assembly=false;
  CubicProblem full_tensor(model,planes,6.2,o,full_tensor_settings);
  assert(std::abs(full_tensor.objective(model.x,.003,&gb,&hb)-fa)<1e-12);
  assert((ga-gb).norm()<1e-12);assert((ha.data-hb.data).norm()<1e-11);
  auto unmerged_settings=settings;unmerged_settings.merge_shared_supports=false;
  CubicProblem unmerged(model,planes,6.2,o,unmerged_settings);
  assert(problem.groups.size()<unmerged.groups.size());
  assert(problem.cuts.size()<unmerged.cuts.size());
  assert(problem.merged_supports>0);
  // Merging shared values must preserve support feasibility for arbitrary
  // states, including nonuniform spans and both sides of the boundary.
  CubicModel uneven({0,.07,.23,.62,1});uneven.x.setZero();
  for(int i=0;i<uneven.basis.count;++i) {
    const double u=uneven.basis.greville(i);
    uneven.reference[i]=Vec3(6*u,.3*std::sin(4*u),1+.2*u);
    const Vec3 tangent=Vec3(1,.2*std::cos(4*u),.1).normalized();
    uneven.frames[i].col(0)=tangent.unitOrthogonal();uneven.frames[i].col(1)=tangent.cross(uneven.frames[i].col(0));
    uneven.x[5*i+2]=.4;uneven.x[5*i+4]=.5;
  }
  std::vector<detail::SupportPlane> uneven_planes;
  for(int i=0;i<4;++i)for(int axis=0;axis<3;++axis)for(double sign:{-1.,1.})
    uneven_planes.push_back({i,0,1,sign*Vec3::Unit(axis),8.});
  CubicProblem merged_uneven(uneven,uneven_planes,6.,o,settings),raw_uneven(uneven,uneven_planes,6.,o,unmerged_settings);
  auto minimum_support=[&](const CubicProblem& p,const Eigen::VectorXd& x) {
    double minimum=INFINITY;
    for(const auto& c:p.cuts){const auto v=c.apply(x);minimum=std::min(minimum,c.bound-v[0]-v.tail<2>().norm());}
    return minimum;
  };
  for(int trial=0;trial<24;++trial) {
    auto x=uneven.x;
    for(int j=0;j<x.size();++j)x[j]+=3*std::sin((trial+1)*(j+1)*.351);
    assert(std::abs(minimum_support(merged_uneven,x)-minimum_support(raw_uneven,x))<1e-11);
  }
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
    assert(std::isfinite(unmerged.objective(model.x+alpha*direction,.003)));
    const auto line=feasible_problem.prepareLine(model.x,direction,.003);
    CubicProblem::Evaluation cached;
    feasible_problem.objective(model.x,.003,nullptr,nullptr,&cached);
    const auto cached_line=feasible_problem.prepareLine(model.x,direction,.003,&cached);
    assert(cached_line.step==line.step);
    assert(std::abs(line.step-alpha)<1e-10);
    for(double fraction:{.1,.5,1.}) {
      const double t=fraction*line.step;
      const double actual=feasible_problem.objective(model.x+t*direction,.003)-feasible_problem.objective(model.x,.003);
      assert(std::abs(line.delta(t)-actual)<1e-10*std::max(1.,std::abs(actual)));
      assert(cached_line.delta(t)==line.delta(t));
    }
  }
  auto inflation_settings=settings;inflation_settings.coordinate_sweeps=2;
  CubicProblem inflation(model,planes,6.2,o,inflation_settings);
  auto inflated=model.x;OptimizerMetrics inflation_metrics;
  const auto original_cost=inflation.objective(inflated,0);
  inflation.inflate(inflated,inflation_metrics);
  assert(inflation_metrics.coordinate_updates>0);
  assert(inflation.objective(inflated,0)<=original_cost+1e-10);
  assert(std::isfinite(unmerged.objective(inflated,.003)));
  Eigen::VectorXd volume_gradient;
  sweptVolume(uneven,uneven.x,&volume_gradient);
  double volume_error=0;
  for(int trial=0;trial<8;++trial) {
    Eigen::VectorXd direction(uneven.x.size());
    for(int j=0;j<direction.size();++j)direction[j]=std::sin((trial+1)*(j+1)*.731);
    direction.normalize();const double step=1e-6;
    const double fd=(sweptVolume(uneven,uneven.x+step*direction)-sweptVolume(uneven,uneven.x-step*direction))/(2*step);
    volume_error=std::max(volume_error,std::abs(fd-volume_gradient.dot(direction))/std::max(1.,std::abs(fd)));
  }
  std::cout<<"swept-volume gradient "<<volume_error<<'\n';assert(volume_error<1e-8);
  auto volume_settings=settings;volume_settings.volume_refinements=2;
  CubicProblem volume_problem(model,planes,6.2,o,volume_settings);
  auto enlarged=model.x;OptimizerMetrics volume_metrics;
  volume_problem.refineVolume(enlarged,volume_metrics,[&](const Eigen::VectorXd& x){return certify(model.geometry(x),planes).safe();});
  assert(volume_metrics.volume_steps>0);assert(sweptVolume(model,enlarged)>sweptVolume(model,model.x));
  assert(certify(model.geometry(enlarged),planes).safe());
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
