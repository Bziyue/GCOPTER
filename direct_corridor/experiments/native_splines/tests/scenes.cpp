// SPDX-License-Identifier: MIT
#include "cubic_generator.hpp"
#include <cassert>
#include <iostream>
using namespace gcopter::direct;
using namespace gcopter::direct::native;
int main() {
  struct Scene{std::string name;std::vector<Vec3> seed;std::vector<Sphere> spheres;bool require_success=false,require_failure=false;};
  std::vector<Scene> scenes={
    {"straight",{{-3,0,0},{3,0,0}},{},true},
    {"right_angle",{{-3,0,0},{0,0,0},{0,3,0}},{},true},
    {"u_turn",{{-3,0,0},{0,0,0},{0,2,0},{-3,2,0}},{},false},
    {"vertical",{{0,0,-3},{0,0,3}},{},true},
    {"backtracking_cusp",{{-3,0,0},{0,0,0},{-3,0,0}},{},false,true},
    {"narrow_003m",{{-2,0,0},{2,0,0}},{},false,true}};
  const double r=std::sqrt(3)*.25/2;
  for(double x=-1;x<=1;x+=.25)for(double sign:{-1.,1.})scenes.back().spheres.push_back({Vec3(x,sign*(r+.25+.015),0),r});
  Scene helix;helix.name="helix";
  for(int i=0;i<=24;++i){const double t=2*std::acos(-1.)*i/24;helix.seed.push_back(Vec3(2*std::cos(t),2*std::sin(t),.5*t));}
  scenes.push_back(helix);
  Scene overlap;overlap.name="close_nonadjacent_legs";overlap.seed={{-3,0,0},{0,0,0},{0,.4,0},{-3,.4,0}};scenes.push_back(overlap);
  for(int reach_mode:{0,1,2})for(const auto& scene:scenes) {
    Options o;o.control_spacing=.4;o.min_radius=.03;o.max_radius=3;o.max_offset=3;
    o.robot_radius=.2;o.safety_margin=.05;
    o.local_range=3;o.frame_smoothing=1.5;o.budget_ms=3000;o.max_refinements=3;o.initial_iterations=8;
    ObstacleIndex index(scene.spheres);Bounds bounds{Vec3::Constant(-10),Vec3::Constant(10)};
    OptimizerSettings solver;applyNativePreset(NativePreset::Balanced,o,solver);solver.reach_bound=reach_mode;
    auto r=generateCubic(scene.seed,index,bounds,o,solver);
    std::cout<<scene.name<<" reach="<<reach_mode<<": "<<r.status<<", "<<r.total_ms<<" ms, refinements="<<r.refinements<<"\n";
    if(scene.require_success)assert(r.status=="SUCCESS");
    if(scene.require_failure)assert(r.status!="SUCCESS");
    if(r.status=="SUCCESS") {
      assert(certify(r.corridor,r.cover).safe());
      assert((r.corridor.section(0).center-scene.seed.front()).norm()<1e-10);
      assert((r.corridor.section(1).center-scene.seed.back()).norm()<1e-10);
    }else assert(r.corridor.pieces()==0);
  }
  // Reuse obstacle-side masks while refreshing the changed patch side.
  const std::vector<Vec3> path{{-2,0,0},{2,0,0}};
  auto model=CubicModel::fromSeed(path,.4,.4,1.5);
  std::vector<Sphere> walls;
  for(double x=-3;x<=3;x+=.25)for(double z=-2;z<=2;z+=.25)for(double sign:{-1.,1.})
    walls.push_back({Vec3(x,2*sign,z),.1});
  ObstacleIndex map(walls);Bounds bounds{Vec3::Constant(-5),Vec3::Constant(5)};
  Options options;options.local_range=3;options.robot_radius=.2;options.safety_margin=.05;
  QueryCache cache;SeparatorCache separated;CoverMetrics metrics;
  const std::function<void()> stop=[]{};
  auto cover=mapCover(model.geometry(),path,map,bounds,options,cache,metrics,stop,&separated,.2);
  assert(certify(model.geometry(),cover).safe());
  for(int i=0;i<model.basis.count;++i)model.x.segment<3>(5*i+2)*=1.2;
  cover=mapCover(model.geometry(),path,map,bounds,options,cache,metrics,stop,&separated,.2);
  assert(metrics.reused_planes>0&&metrics.reused_spheres>0);
  assert(certify(model.geometry(),cover).safe());
  QueryCache independent;CoverMetrics audit;
  assert(certify(model.geometry(),mapCover(model.geometry(),path,map,bounds,options,independent,audit,stop)).safe());
  for(int i=0;i<model.basis.count;++i){model.x[5*i+2]=2.1;model.x[5*i+4]=2.1;}
  bool rejected=false;
  try{mapCover(model.geometry(),path,map,bounds,options,cache,metrics,stop,&separated,.2);}
  catch(const detail::Failure&){rejected=true;}
  assert(rejected);
}
