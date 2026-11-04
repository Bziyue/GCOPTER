// SPDX-License-Identifier: MIT
#pragma once
#include "../../src/hermite_optimizer.hpp"
#include <memory>
#include <queue>

namespace gcopter::direct::native {
// The box key is a lower bound on every surviving sphere's ordering score.
// Exact scores are computed only when a block can contain the next obstacle.
class LazyMetricOrder {
  struct Entry {double score;int obstacle,block;unsigned version;};
  struct Later {bool operator()(const Entry& a,const Entry& b)const {
    if(a.score!=b.score)return a.score>b.score;
    if(a.obstacle!=b.obstacle)return a.obstacle>b.obstacle;
    return a.block>b.block;
  }};
  std::priority_queue<Entry,std::vector<Entry>,Later> heap;
  std::vector<detail::ObstacleGroups::Block>& blocks;
  const std::vector<int>& ids;
  std::unique_ptr<double[]> scores;
  const ObstacleIndex& map;
  Eigen::Matrix3d metric;
  Vec3 center;
  double inverse_scale,pad;
  bool uniform;
  std::vector<unsigned> versions;
  std::vector<unsigned char> evaluated;
public:
  size_t evaluations=0;
  LazyMetricOrder(std::vector<detail::ObstacleGroups::Block>& b,const std::vector<int>& obstacle_ids,
                  const ObstacleIndex& m,const Eigen::Matrix3d& a,const Vec3& origin,
                  double scale,double padding,bool equal_radius)
      :blocks(b),ids(obstacle_ids),scores(new double[ids.size()]),map(m),metric(a),center(origin),inverse_scale(1/scale),
       pad(padding),uniform(equal_radius),versions(b.size()),evaluated(b.size()) {
    const auto absolute=metric.cwiseAbs().eval();
    for(size_t i=0;i<blocks.size();++i)if(blocks[i].alive) {
      const auto& block=blocks[i];
      const Vec3 midpoint=metric*(block.center-center),extent=absolute*block.half;
      const double squared=(midpoint.cwiseAbs()-extent).cwiseMax(0.).squaredNorm();
      const double guard=64*std::numeric_limits<double>::epsilon()*std::max(1.,midpoint.squaredNorm()+extent.squaredNorm());
      double lower=std::max(0.,squared-guard);
      if(!uniform)lower=std::sqrt(lower)-(block.max_radius+pad)*inverse_scale;
      heap.push({lower,-1,int(i),0});
    }
  }
  void changed(int i) {
    auto& block=blocks[i];
    if(!block.alive){++versions[i];block.nearest=-1;return;}
    if(!evaluated[i])return;
    int nearest=-1;
    for(auto bits=block.alive;bits;bits&=bits-1) {
      const int j=block.begin+__builtin_ctzll(bits);
      if(nearest<0||scores[j]<scores[nearest]||
         (scores[j]==scores[nearest]&&ids[j]<ids[nearest]))nearest=j;
    }
    block.nearest=nearest;++versions[i];
    heap.push({scores[nearest],ids[nearest],i,versions[i]});
  }
  int next() {
    while(!heap.empty()) {
      const auto entry=heap.top();auto& block=blocks[entry.block];
      if(entry.version!=versions[entry.block]||!block.alive){heap.pop();continue;}
      if(evaluated[entry.block])return block.nearest;
      heap.pop();
      for(auto bits=block.alive;bits;bits&=bits-1) {
        const int j=block.begin+__builtin_ctzll(bits);
        const auto& sphere=map.obstacles()[ids[j]];
        double distance=(metric*(sphere.center-center)).squaredNorm();
        if(!uniform)distance=std::sqrt(distance)-(sphere.radius+pad)*inverse_scale;
        scores[j]=distance;++evaluations;
      }
      // Only evaluated blocks may read scores; removed candidates never
      // return. Avoid initializing an array that is mostly never consulted.
      evaluated[entry.block]=1;changed(entry.block);
    }
    return -1;
  }
};
} // namespace gcopter::direct::native
