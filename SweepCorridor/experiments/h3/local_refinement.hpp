// SPDX-License-Identifier: MIT
#pragma once
#include "end_blocks.hpp"
#include "inflation_cover.hpp"

namespace SweepCorridor::experiments::h3 {
// Query domains and obstacle packets belong to the frozen seed layout. Only
// the separating normals of selected spans are rebuilt at the current shape.
inline std::vector<unsigned char> refreshCover(const Geometry& geometry,
    std::vector<internal::detail::SupportPlane>& cover,const std::vector<unsigned char>& selected,
    const ObstacleIndex& map,const Bounds& bounds,const GeometryOptions& options,
    c3::QueryCache& cache,c3::CoverMetrics& timings,const c3::CoverLayout& layout,
    JetMetrics& metrics,Deadline& deadline,const OptimizerSettings& settings){
  StageTimer timer(settings.profile);c3::MapCover<3> builder(map,bounds,options,cache,timings,deadline,settings.profile);
  builder.prefer_faces=settings.cover_group_size>1 || map.voxelHalfWidth()>0;
  builder.group_axis_limit=settings.group_axis_limit;
  const int group_size=settings.separator_scale==SeparatorScale::Group ? settings.cover_group_size :
                       settings.separator_scale==SeparatorScale::Span ? 1 : 0;
  std::vector<unsigned char> rebuilt(geometry.pieces(),0);
  int group_end=0;
  double chord=0;
  for(int i=0;i<int(geometry.pieces());++i) {
    if(group_size>0 && i==group_end) {
      group_end=coverGroupEnd(layout,i,group_size);
      chord=groupChordRadius(geometry,i,group_end);
    }
    if(!selected[i])continue;
    deadline.check();const auto checkpoint=builder.planes.size();++metrics.refresh_attempts;
    try{
      const c3::GeometryPatch<3> patch{i,0,1,geometry.curve(i)};
      builder.boundsForSpan(patch,layout);
      const auto cached=cache.ids.size();
      separateInflationSpan(builder,patch,layout,chord);
      metrics.refresh_cache_hits+=cache.spans[i]>=0 && size_t(cache.spans[i])<cached;
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

// In disk coordinates, translate by s*v and stretch by I+s*v*v^T.
// For a halfspace a.z <= b the exact upper bound on s is
// (b^2-|a|^2)/(2*k*(b+k)), k=a.v>0. The opposite faces stay feasible.
// Its area ratio is 1+s. Rank this feasible gain inside the current cover.
inline std::vector<double> sectionGains(const Geometry& g,const std::vector<SupportPlane>& cover,
                                        const std::array<double,2>& anchors) {
  std::vector<double> gain(g.pieces(),0.);
  size_t first=0;
  constexpr double diagonal=.7071067811865475244;
  for(int span=0;span<int(g.pieces());++span) {
    size_t end=first;
    while(end<cover.size() && cover[end].span==span) ++end;
    const double a=std::max(g.breaks[span],anchors[0]),b=std::min(g.breaks[span+1],anchors[1]);
    if(b>a && end>first) {
      const double h=g.breaks[span+1]-g.breaks[span],t=(.5*(a+b)-g.breaks[span])/h;
      const auto section=c3::evaluate<3>(g.curve(span),t,h);
      std::array<double,8> limits;
      limits.fill(INFINITY);
      for(size_t j=first;j<end;++j) {
        const auto& p=cover[j];
        if(t<p.begin || t>p.end) continue;
        const Eigen::Vector2d n=section.shape.transpose()*p.normal;
        const double gap=p.bound-p.normal.dot(section.center);
        const double slack=std::max(0.,gap*gap-n.squaredNorm());
        const std::array<double,4> projection{n.x(),n.y(),diagonal*(n.x()+n.y()),diagonal*(n.x()-n.y())};
        for(int k=0;k<4;++k) {
          const double mag=std::abs(projection[k]);
          if(mag<=1e-12) continue;
          auto& limit=limits[2*k+(projection[k]<0)];
          limit=std::min(limit,slack/(2*mag*(gap+mag)));
        }
      }
      const double stretch=*std::max_element(limits.begin(),limits.end());
      const double density=section.shape.col(0).cross(section.shape.col(1)).dot(section.first);
      if(std::isfinite(stretch)) gain[span]=std::max(0.,stretch*density*(b-a));
    }
    first=end;
  }
  return gain;
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
    const bool has_pressure=std::any_of(spans.begin(),spans.end(),[](unsigned char v){return v!=0;});
    if(!has_pressure && !settings.use_section_gain && (!settings.end_steps || settings.require_refresh_for_ends))break;
    const auto geometry=model.geometry();
    const double old_volume=feedback.volume,old_end_gain=metrics.end_volume_gain;
    const auto rebuilt=has_pressure ? refreshCover(geometry,cover,spans,map,bounds,options,cache,timings,layout,
                                                   metrics,deadline,settings) : spans;
    std::vector<unsigned char> nodes(n,0),support_spans(n-1,0);
    for(int i=0;i<n-1;++i)if(rebuilt[i])nodes[i]=nodes[i+1]=1;
    if(settings.use_section_gain && settings.refresh_shape && settings.refresh_fraction>0) {
      const auto gain=sectionGains(geometry,cover,model.anchor_parameters);
      std::vector<int> order;
      for(int i=0;i<n-1;++i) if(gain[i]>0 && !(nodes[i] && nodes[i+1]))order.push_back(i);
      const int count=std::min(int(order.size()),int(std::ceil((n-1)*settings.refresh_fraction)));
      std::partial_sort(order.begin(),order.begin()+count,order.end(),[&](int i,int j) {
        return gain[i]==gain[j]?i<j:gain[i]>gain[j];
      });
      for(int j=0;j<count;++j)nodes[order[j]]=nodes[order[j]+1]=1;
      metrics.area_selected_spans+=count;
    }
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
