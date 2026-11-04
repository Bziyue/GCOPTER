// SPDX-License-Identifier: MIT
#include "cubic_generator.hpp"
#include "cubic_optimizer.hpp"
#include "obstacle_cover.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace gcopter::direct;
using namespace gcopter::direct::native;
static void sameCover(const std::vector<detail::SupportPlane>& a,const std::vector<detail::SupportPlane>& b) {
  assert(a.size()==b.size());
  for(size_t i=0;i<a.size();++i) {
    assert(a[i].span==b[i].span);
    assert(!std::memcmp(&a[i].begin,&b[i].begin,sizeof(double)));
    assert(!std::memcmp(&a[i].end,&b[i].end,sizeof(double)));
    assert(!std::memcmp(a[i].normal.data(),b[i].normal.data(),3*sizeof(double)));
    assert(!std::memcmp(&a[i].bound,&b[i].bound,sizeof(double)));
  }
}
static void sameGeometry(const Geometry& a,const Geometry& b) {
  assert(a.degree==b.degree&&a.breaks==b.breaks&&a.controls.size()==b.controls.size());
  for(size_t i=0;i<a.controls.size();++i)assert(!std::memcmp(a.controls[i].data(),b.controls[i].data(),9*sizeof(double)));
}
int main() {
  const std::vector<Vec3> seed{{-4,0,0},{-1,.3,.1},{2,-.2,.4},{4,0,.6}};
  std::vector<Sphere> obstacles;
  for(double x=-5;x<=5;x+=.5)for(double z=-1;z<=2;z+=.5)for(double sign:{-1.,1.})
    obstacles.push_back({Vec3(x,2*sign,z),.2});
  ObstacleIndex index(obstacles);Bounds bounds{Vec3(-8,-6,-5),Vec3(8,6,5)};
  CubicOptions options;OptimizerSettings settings;
  options.max_radius=3;options.max_offset=3;options.local_range=3;options.local_progress=3;
  options.robot_radius=.2;options.safety_margin=.05;options.budget_ms=3000;
  auto model=CubicModel::fromSeed(seed,.5,.3,1.5,{0,.07,.13,.31,.4,.63,.89,1});
  const CoverLayout layout(model.basis.breaks,seed,bounds,options);
  QueryCache cache;CoverMetrics metrics;Deadline deadline(10000);
  for(int pass=0;pass<3;++pass) {
    const auto geometry=model.geometry();
    const auto reused=mapCover(geometry,index,bounds,options,cache,metrics,deadline,layout);
    const CoverLayout fresh_layout(model.basis.breaks,seed,bounds,options);
    QueryCache fresh_cache;CoverMetrics fresh_metrics;
    const auto cover=mapCover(geometry,index,bounds,options,fresh_cache,fresh_metrics,deadline,fresh_layout);
    sameCover(cover,reused);assert(certify(geometry,reused).safe());
    auto duplicates=cover;
    for(size_t i=0;i<std::min(size_t(6),cover.size());++i) {
      auto p=cover[i];p.bound+=.25;
      for(int j=0;j<3;++j)if(p.normal[j]==0)p.normal[j]=-p.normal[j];
      duplicates.insert(duplicates.begin(),p);
    }
    std::reverse(duplicates.begin(),duplicates.end());
    CubicProblem prototype(model,duplicates,9,options,settings);
    ProgressConstraints progress{{0,2},{2}};
    CubicProblem fresh(model,duplicates,9,options,settings,prototype.energy,nullptr,progress);
    CubicProblem shared(model,duplicates,9,options,settings,prototype.energy,&prototype,progress);
    assert(shared.reused_construction&&shared.pruning_ms==0);
    assert(shared.geometry.get()==prototype.geometry.get());
    assert(fresh.geometry->count==shared.geometry->count&&fresh.cuts.size()==shared.cuts.size());
    for(int trial=0;trial<4;++trial) {
      auto x=model.x;for(int i=0;i<x.size();++i)x[i]+=.001*std::sin((trial+1)*(i+1)*.12);
      Eigen::VectorXd a,b;BandedSPD h,k;
      assert(fresh.objective(x,.003,&a,&h)==shared.objective(x,.003,&b,&k));
      assert((a.array()==b.array()).all()&&(h.data.array()==k.data.array()).all());
    }
    for(int i=0;i<model.basis.count;++i)model.x.segment<3>(5*i+2)*=1.15;
  }
  assert(metrics.reused_cover_spans==3*int(model.basis.spans.size()));
  const auto baseline=generateCubic(seed,index,bounds,options,settings);
  settings.profile=true;
  const auto profiled=generateCubic(seed,index,bounds,options,settings);
  assert(profiled.status==baseline.status&&profiled.status=="SUCCESS");
  sameGeometry(profiled.corridor,baseline.corridor);sameCover(profiled.cover,baseline.cover);
  assert(baseline.optimizer.assembly_ms==0&&baseline.cover_metrics.query_ms==0);
  assert(profiled.optimizer.assembly_ms>0&&profiled.cover_metrics.query_ms>0);
  assert(baseline.optimizer.construction_reuses>0&&baseline.cover_metrics.topology_builds==1);
  auto invalid=options;invalid.local_progress=INFINITY;
  assert(generateCubic(seed,index,bounds,invalid,settings).status=="INVALID_INPUT");
  invalid=options;invalid.control_spacing=std::nan("");
  assert(generateCubic(seed,index,bounds,invalid,settings).status=="INVALID_INPUT");
  auto bad_seed=seed;bad_seed[1][0]=INFINITY;
  assert(generateCubic(bad_seed,index,bounds,options,settings).status=="INVALID_INPUT");
  invalid=options;invalid.max_controls=3;
  assert(generateCubic(seed,index,bounds,invalid,settings).status==statusName(Status::Capacity));
  invalid=options;invalid.budget_ms=1e-6;
  assert(generateCubic(seed,index,bounds,invalid,settings).status=="TIMEOUT");
  std::cout<<"Shared constraints, coverage lifetime, profiling equivalence and input/deadline contracts passed\n";
}
