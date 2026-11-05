// SPDX-License-Identifier: MIT
#pragma once
#include "end_blocks.hpp"

namespace SweepCorridor::experiments::h3 {
// Query domains and obstacle packets belong to the frozen seed layout. Only
// the separating normals of selected spans are rebuilt at the current shape.
inline std::vector<unsigned char> refreshCover(const Geometry& geometry,
    std::vector<internal::detail::SupportPlane>& cover,const std::vector<unsigned char>& selected,
    const ObstacleIndex& map,const Bounds& bounds,const GeometryOptions& options,
    c3::QueryCache& cache,c3::CoverMetrics& timings,const c3::CoverLayout& layout,
    JetMetrics& metrics,Deadline& deadline,bool profile){
  StageTimer timer(profile);c3::MapCover<3> builder(map,bounds,options,cache,timings,deadline,profile);
  const double pad=options.robot_radius+options.safety_margin+options.certificate_margin;
  std::vector<unsigned char> rebuilt(geometry.pieces(),0);
  for(int i=0;i<int(geometry.pieces());++i)if(selected[i]){
    deadline.check();const auto checkpoint=builder.planes.size();++metrics.refresh_attempts;
    try{
      const c3::GeometryPatch<3> patch{i,0,1,geometry.curve(i)};
      for(int axis=0;axis<3;++axis){builder.limit(patch,Vec3::Unit(axis),bounds.upper[axis]-pad);builder.limit(patch,-Vec3::Unit(axis),-bounds.lower[axis]-pad);}
      const auto& span=layout.spans[i];
      for(size_t j=span.begin;j<span.end;++j){const auto& part=layout.parts[j];
        const c3::GeometryPatch<3> p{i,part.begin,part.end,c3::restrictCurve<3>(patch.controls,part.begin,part.end)};
        for(int axis=0;axis<3;++axis){builder.limit(p,Vec3::Unit(axis),part.bounds.upper[axis]);builder.limit(p,-Vec3::Unit(axis),-part.bounds.lower[axis]);}}
      StageTimer query(profile);std::vector<int> scratch;
      metrics.refresh_cache_hits+=i<int(cache.spans.size())&&cache.spans[i]>=0;
      const auto& ids=cache.query(i,span.lower-Vec3::Constant(pad),span.upper+Vec3::Constant(pad),map,scratch);
      timings.query_ms+=query.elapsed();query.restart();builder.pairCuts(patch,ids,0);timings.separation_ms+=query.elapsed();
      rebuilt[i]=1;++metrics.refresh_spans;
    }catch(const c3::Failure&){
      // A new separating proposal may fail even though the existing one is
      // certified. Retain that complete old span; never publish a partial cover.
      builder.planes.resize(checkpoint);++metrics.refresh_failures;
    }
  }
  std::vector<internal::detail::SupportPlane> merged;merged.reserve(cover.size()+builder.planes.size());
  size_t old=0,fresh=0;
  for(int i=0;i<int(geometry.pieces());++i){
    while(old<cover.size()&&cover[old].span==i){if(!rebuilt[i])merged.push_back(cover[old]);++old;}
    while(fresh<builder.planes.size()&&builder.planes[fresh].span==i)merged.push_back(builder.planes[fresh++]);
  }
  cover=std::move(merged);const double elapsed=timer.elapsed();timings.total_ms+=elapsed;metrics.refresh_ms+=elapsed;return rebuilt;
}

inline void refineLocal(Model& model,std::vector<internal::detail::SupportPlane>& cover,double length,
    const ObstacleIndex& map,const Bounds& bounds,const GeometryOptions& options,const OptimizerSettings& settings,
    c3::QueryCache& cache,c3::CoverMetrics& timings,const c3::CoverLayout& layout,JetMetrics& metrics,Deadline& deadline,
    JointFeedback& feedback){
  const int n=model.basis.count;
  for(int round=0;round<settings.refresh_rounds;++round){
    std::vector<unsigned char> spans(n-1,0);
    // Fixed box faces cannot improve by changing their separating normals.
    // Reuse the previous Newton assembly's obstacle pressure to avoid empty work.
    auto pressure=[&](int span){return std::max(feedback.nodes[span].pressure,feedback.nodes[span+1].pressure);};
    for(int i=0;i<std::min(settings.end_nodes,n-1);++i){spans[i]=pressure(i)>0;spans[n-2-i]=pressure(n-2-i)>0;}
    if(settings.refresh_fraction>0){
      double gain=0;for(const auto& node:feedback.nodes)gain+=node.gain;
      auto nodeScore=[&](int i){const auto& f=feedback.nodes[i];
        return f.pressure/(1+f.gain/(model.basis.weight(i)*std::max(1e-12,gain)));};
      std::vector<std::pair<double,int>> order;for(int i=0;i<n-1;++i)if(!spans[i]){
        const double score=std::max(nodeScore(i),nodeScore(i+1));if(score>0)order.push_back({score,i});}
      std::stable_sort(order.begin(),order.end(),[](const auto& a,const auto& b){return a.first>b.first;});
      const int count=std::min(int(order.size()),int(std::ceil((n-1)*settings.refresh_fraction)));
      for(int i=0;i<count;++i)spans[order[i].second]=1;
    }
    if(std::none_of(spans.begin(),spans.end(),[](unsigned char x){return x!=0;}))break;
    const double old_volume=feedback.volume,old_end_gain=metrics.end_volume_gain;
    const auto rebuilt=refreshCover(model.geometry(),cover,spans,map,bounds,options,cache,timings,layout,metrics,deadline,settings.profile);
    std::vector<unsigned char> nodes(n,0),support_spans(n-1,0);
    for(int i=0;i<n-1;++i)if(rebuilt[i])nodes[i]=nodes[i+1]=1;
    for(int i=0;i<n-1;++i)if(nodes[i]||nodes[i+1])support_spans[i]=1;
    // End center blocks also need the unchanged support rows of their neighbors.
    if(settings.end_steps)for(int i=0;i<std::min(settings.end_nodes,n-1);++i)support_spans[i]=support_spans[n-2-i]=1;
    StageTimer setup(settings.profile);GroupBuildMetrics grouping;auto constraints=buildSupportGroups(model,cover,grouping,&support_spans);
    metrics.local_setup_ms+=setup.elapsed();
    if(settings.refresh_shape){auto local=settings;local.joint_sweeps=1;
      const double before=metrics.milliseconds;
      optimizeJointJets(model,cover,length,options,local,metrics,deadline,constraints.get(),&nodes,&feedback);
      metrics.local_joint_ms+=metrics.milliseconds-before;}
    if(settings.end_steps)optimizeEnds(model,options,length,settings,*constraints,metrics,deadline);
    feedback.volume+=metrics.end_volume_gain-old_end_gain;
    if(feedback.volume-old_volume<1e-4*old_volume)break;
  }
}
}
