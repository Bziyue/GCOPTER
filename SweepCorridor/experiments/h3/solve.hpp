// SPDX-License-Identifier: MIT
#pragma once
#include "problem_impl.hpp"
#include "cap_volume.hpp"
namespace SweepCorridor::experiments::h3 {
inline bool regularGeometry(const Model& model,const Eigen::VectorXd& x,c3::JacobianRequirement relative_margin) {
  const auto g=model.geometry(x);c3::Certificate result;
  for(int i=0;i<int(g.pieces());++i)if(!c3::jacobianCertificate<3>(g.curve(i),result,relative_margin.onSpan(g.breaks[i+1]-g.breaks[i])))return false;
  return true;
}
template<class Solve>
inline void constrainCapStep(const Model& model,const Eigen::VectorXd& x,Eigen::VectorXd& step,Solve&& solve) {
  std::array<Vec3,2> normals{{inwardCapNormal(model,0),inwardCapNormal(model,1)}};
  const std::array<int,2> offsets{{0,int(x.size())-6}};
  Eigen::Vector2d gaps;
  for(int i=0;i<2;++i)gaps[i]=normals[i].dot(x.segment<3>(offsets[i]));
  const auto slack=[&](int i,const Eigen::VectorXd& d){return gaps[i]+normals[i].dot(d.segment<3>(offsets[i]));};
  if(!projectCapStep(2,offsets,normals,step,slack,solve))throw std::runtime_error("cap constrained Newton direction");
}
inline void Problem::optimize(Eigen::VectorXd& x,OptimizerMetrics& metrics,const Deadline& deadline,int first_stage,int last_stage)const {
  metrics.construction_reuses+=reused_construction;
  metrics.basis_cache_hits+=basis_cache_hits;metrics.basis_cache_misses+=basis_cache_misses;
  metrics.group_lookup_reuses+=group_lookup_reuses;
  metrics.controls=model.basis.count;metrics.variables=x.size();metrics.constraints=geometry->count+cuts.size();
  metrics.merged_supports=geometry->merged;metrics.support_groups=geometry->groups.size();
  metrics.pruned_supports=geometry->count-geometry->retained;metrics.axis_supports=geometry->axes;
  metrics.pruning_ms+=pruning_ms;metrics.active_constraints=geometry->retained+cuts.size();
  metrics.passes.push_back({geometry->count,geometry->retained,geometry->retained,int(cuts.size()),0,pruning_ms});
  metrics.hessian_nonzero=(model.basis.count*21+(model.basis.count-1)*36);
  std::array<Anchor,2> anchors;
  for(int end=0;end<2;++end)anchors[end].u=model.anchor_parameters[end];
  constexpr std::array<double,3> barriers{{.003,.0001,.00001}};
  Eigen::VectorXd g,step,vg,inverse_v,next;BlockTridiagonal h;
  for(int stage=first_stage;stage<last_stage;++stage){const double anchor_mu=barriers[stage],mu=anchor_mu/model.basis.count;
    bool converged=false,stalled=false;const int limit=stage?settings.corrector_iterations:settings.iterations;
    for(int it=0;it<limit;++it){deadline.check();StageTimer timer(settings.profile);
      const double base=objective(x,mu,&g,&h,false);
      double volume=sweptVolume(model,x,&vg);
      const double anchor_cost=anchorObjective(model,x,anchors,anchor_mu,&g,&h);
      if(!std::isfinite(base))throw std::runtime_error("free-boundary support interior");
      if(!std::isfinite(anchor_cost))throw std::runtime_error("free-boundary anchor inversion / inclusion");
      const double discount=1-settings.cap_volume_weight;
      double cap=0;
      if(discount>0) {
        cap=capVolume(model,x,anchors,[&](int span,const c3::Curve<3,V9>& control) {
          vg.segment<6>(6*span)-=discount*model.jacobian(span).transpose()*(control[0]+control[1]);
          vg.segment<6>(6*span+6)-=discount*model.jacobian(span+1).transpose()*(control[2]+control[3]);
        });
        volume-=discount*cap;
      }
      if(!(volume>0))throw std::runtime_error("free-boundary nonpositive volume");
      vg/=volume;g-=vg;
      metrics.assembly_ms+=timer.elapsed();metrics.gradient_support_rows+=metrics.active_constraints;metrics.hessian_support_rows+=metrics.active_constraints;
      timer.restart();double damping=0;
      while(!h.factor(damping)){damping=damping==0?1e-8:damping*10;++metrics.damping_retries;
        if(damping>1e6)throw std::runtime_error("free-boundary band factorization");}
      // The positive rank-one part of Hessian(-log V) is handled exactly by
      // Sherman-Morrison. No dense matrix or wider band is materialized.
      h.solveInto(vg,inverse_v);const double denominator=1+vg.dot(inverse_v);
      auto solve=[&](const Eigen::VectorXd& rhs,Eigen::VectorXd& answer){h.solveInto(rhs,answer);answer-=inverse_v*(vg.dot(answer)/denominator);};
      solve(-g,step);
      constrainCapStep(model,x,step,solve);
      metrics.solve_ms+=timer.elapsed();const double decrement=-g.dot(step);metrics.final_decrement=decrement;
      if(decrement<settings.gradient_tolerance){converged=true;break;}
      timer.restart();const auto line=prepareLine(x,step,mu);double alpha=std::min(line.step,capStep(model,x,step));
      ++metrics.feasible_scans;metrics.boundary_support_rows+=metrics.active_constraints;
      const Vec3 polynomial=volumeLine(model,x,step);
      bool accepted=false;std::array<Anchor,2> trial_anchors;
      for(int trial=0;trial<28&&alpha>1e-12;++trial,alpha*=.5){
        double dv=alpha*(polynomial[0]+alpha*(polynomial[1]+alpha*polynomial[2]));
        next=x+alpha*step;trial_anchors=anchors;
        const double ca=anchorObjective(model,next,trial_anchors,anchor_mu);
        if(!std::isfinite(ca))continue;
        if(discount>0) dv-=discount*(capVolume(model,next,trial_anchors)-cap);
        if(!(dv>-volume))continue;
        const double other=-std::log1p(dv/volume)+ca-anchor_cost,decrease=-1e-4*alpha*decrement;
        ++metrics.bound_line_evaluations;metrics.line_support_rows+=metrics.active_constraints;
        bool sufficient=line.upperBound(alpha)+other<=decrease;
        if(sufficient)++metrics.bound_line_accepts;
        else {++metrics.exact_line_evaluations;metrics.line_support_rows+=metrics.active_constraints;sufficient=line.delta(alpha)+other<=decrease;}
        if(sufficient){
          StageTimer cert_timer(settings.profile);const bool regular=regularGeometry(model,next,regularity);metrics.certificate_ms+=cert_timer.elapsed();
          if(regular){accepted=true;break;}++metrics.regularity_backtracks;
        }++metrics.backtracks;
      }
      metrics.line_search_ms+=timer.elapsed();
      if(!accepted){stalled=true;break;}
      x.swap(next);anchors=trial_anchors;
      for(int end=0;end<2;++end)model.anchor_parameters[end]=anchors[end].u;
      ++metrics.steps;++metrics.passes.back().steps;
      if(alpha==1)++metrics.full_step_accepts;
      metrics.minimum_step=std::min(metrics.minimum_step,alpha);metrics.step_sum+=alpha;
    }
    metrics.termination=converged?"newton_decrement":(stalled?"line_search_stalled":"iteration_limit");
  }
}
}
