// SPDX-License-Identifier: MIT
#include "cubic_adaptation.hpp"
#include "lazy_obstacles.hpp"
#include <cassert>
#include <numeric>
#include <iostream>
using namespace gcopter::direct;
using namespace gcopter::direct::native;
int main() {
  for(bool uniform:{false,true}) {
    std::vector<Sphere> spheres;
    for(int i=0;i<768;++i)spheres.push_back({Vec3(.3*(i%32),.4*((i/32)%8),.7*(i/256)),uniform?.1:.05+.01*(i%13)});
    ObstacleIndex index(spheres);std::vector<int> ids(spheres.size());std::iota(ids.begin(),ids.end(),0);
    auto grouped=detail::obstacleGroups(ids,index);auto& blocks=grouped.blocks;
    Eigen::Matrix3d metric;metric<<1,.2,.1, .3,4,.2, .1,.3,.4;
    const Vec3 center(3.78,.73,-.31);const double scale=.3,pad=.25;
    LazyMetricOrder order(blocks,ids,index,metric,center,scale,pad,uniform);
    for(int step=0;step<200;++step) {
      int expected=-1;double score=INFINITY;
      for(const auto& block:blocks)for(auto bits=block.alive;bits;bits&=bits-1) {
        const int j=block.begin+__builtin_ctzll(bits);const auto& sphere=spheres[ids[j]];
        double distance=(metric*(sphere.center-center)).squaredNorm();
        if(!uniform)distance=std::sqrt(distance)-(sphere.radius+pad)/scale;
        if(distance<score||(distance==score&&(expected<0||ids[j]<ids[expected]))){score=distance;expected=j;}
      }
      const int selected=order.next();assert(selected==expected);if(selected<0)break;
      const int block=selected/32;blocks[block].alive&=~(1ull<<(selected-blocks[block].begin));order.changed(block);
      if(step%5==0) {
        const int removed=(step*7+11)%blocks.size();blocks[removed].alive=0;order.changed(removed);
      }
    }
    assert(order.evaluations<=spheres.size());
  }
  auto model=CubicModel::fromSeed({Vec3(-3,0,0),Vec3(0,.2,.3),Vec3(3,1,.7)},.7,.4,1.5);
  const auto original=model.geometry();
  for(double u:{.031,.21,.743})model=insertKnot(model,u);
  const auto refined=model.geometry();
  for(int i=0;i<=200;++i) {
    const auto a=original.section(i/200.),b=refined.section(i/200.);
    assert((pack(a.center,a.shape)-pack(b.center,b.shape)).norm()<2e-13);
    assert((pack(a.first,a.shape_first)-pack(b.first,b.shape_first)).norm()<2e-11);
    assert((pack(a.second,a.shape_second)-pack(b.second,b.shape_second)).norm()<2e-9);
  }
  auto straight=CubicModel::fromSeed({Vec3(-5,0,0),Vec3(5,0,0)},.4,1,1.5);
  assert(adaptiveBreaks(straight,1).size()<straight.basis.breaks.size());
  std::cout<<"lazy order matches exhaustive search; world-space knot insertion preserves geometry and C2\n";
}
