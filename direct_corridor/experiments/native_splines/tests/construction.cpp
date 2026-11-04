// SPDX-License-Identifier: MIT
#include "cubic_generator.hpp"
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
  Options options;OptimizerSettings settings;applyNativePreset(NativePreset::Balanced,options,settings);
  assert(settings.reuse_construction&&settings.fast_construction&&settings.cache_coverage);
  settings.reuse_construction=settings.fast_construction=settings.cache_coverage=false;
  options.max_radius=3;options.max_offset=3;options.local_range=3;options.local_progress=2;
  options.robot_radius=.2;options.safety_margin=.05;options.budget_ms=3000;
  auto model=CubicModel::fromSeed(seed,.5,.3,1.5,{0,.07,.13,.31,.4,.63,.89,1});
  const CoverLayout layout(model.basis.breaks,seed,bounds,options);
  QueryCache original_cache,cached_cache;CoverMetrics original_metrics,cached_metrics;
  const std::function<void()> stop=[]{};
  for(int pass=0;pass<3;++pass) {
    const auto geometry=model.geometry();
    const auto cover=mapCover(geometry,seed,index,bounds,options,original_cache,original_metrics,stop,nullptr,0,true);
    const auto reused=mapCover(geometry,seed,index,bounds,options,cached_cache,cached_metrics,stop,nullptr,0,true,&layout);
    sameCover(cover,reused);assert(certify(geometry,reused).safe());
    auto duplicates=cover;
    for(size_t i=0;i<std::min(size_t(6),cover.size());++i) {
      auto p=cover[i];p.bound+=.25;
      for(int j=0;j<3;++j)if(p.normal[j]==0)p.normal[j]=-p.normal[j];
      duplicates.insert(duplicates.begin(),p);
    }
    std::reverse(duplicates.begin(),duplicates.end());
    for(int mode:{0,1,2}) {
      auto base=settings;base.certificate_refinement=mode;base.certificate_fraction=.25;
      CubicProblem prototype(model,duplicates,9,options,base);
      auto next=base;next.iterations=4;next.progress_spans={0,2};next.strict_progress_spans={2};
      CubicProblem fresh(model,duplicates,9,options,next,prototype.energy);
      next.fast_construction=true;
      CubicProblem flat(model,duplicates,9,options,next,prototype.energy);
      CubicProblem cached(model,duplicates,9,options,next,prototype.energy,&prototype);
      assert(cached.reused_construction&&cached.pruning_ms==0);
      assert(cached.geometric_cuts==fresh.geometric_cuts&&cached.cuts.size()==fresh.cuts.size());
      for(int trial=0;trial<4;++trial) {
        auto x=model.x;for(int i=0;i<x.size();++i)x[i]+=.001*std::sin((trial+1)*(i+1)*.12);
        Eigen::VectorXd a,b;BandedSPD h,k;
        assert(fresh.objective(x,.003,&a,&h)==cached.objective(x,.003,&b,&k));
        assert((a.array()==b.array()).all()&&(h.data.array()==k.data.array()).all());
        assert(fresh.objective(x,.003,&a,&h)==flat.objective(x,.003,&b,&k));
        assert((a.array()==b.array()).all()&&(h.data.array()==k.data.array()).all());
      }
      assert(flat.cuts.size()==fresh.cuts.size());
      for(size_t i=0;i<flat.cuts.size();++i) {
        const auto& a=flat.cuts[i];const auto& b=fresh.cuts[i];
        assert(a.first==b.first&&a.count==b.count&&a.bound==b.bound&&a.redundant==b.redundant);
        for(int k=0;k<a.count;++k)assert(!std::memcmp(a.normals[k].data(),b.normals[k].data(),2*sizeof(double)));
      }
    }
    for(int i=0;i<model.basis.count;++i)model.x.segment<3>(5*i+2)*=1.15;
  }
  assert(cached_metrics.reused_cover_spans==3*int(model.basis.spans.size()));
  for(int mode:{0,1,2}) {
    settings.certificate_refinement=mode;settings.certificate_fraction=.1;
    const auto baseline=generateCubic(seed,index,bounds,options,settings);
    auto cached=settings;cached.reuse_construction=true;cached.cache_coverage=true;cached.fast_construction=true;
    const auto result=generateCubic(seed,index,bounds,options,cached);
    assert(result.status==baseline.status&&result.status=="SUCCESS");
    sameGeometry(result.corridor,baseline.corridor);sameCover(result.cover,baseline.cover);
    assert(result.optimizer.construction_reuses>0&&result.cover_metrics.topology_builds==1);
  }
  std::cout<<"Exact construction/cover reuse: nonuniform knots, changed geometry, progress repairs and E2 modes passed\n";
}
