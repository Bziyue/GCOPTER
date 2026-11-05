// SPDX-License-Identifier: MIT
#pragma once
#include "problem_impl.hpp"
#include "anchors.hpp"
namespace SweepCorridor::experiments::h3 {
inline bool regularGeometry(const Model& model,const Eigen::VectorXd& x) {
  const auto g=model.geometry(x);c3::Certificate result;
  for(int i=0;i<int(g.pieces());++i)if(!c3::jacobianCertificate<3>(g.curve(i),result))return false;
  return true;
}
template<class Solve>
inline void constrainCapStep(const Model& model,const Eigen::VectorXd& x,Eigen::VectorXd& step,Solve&& solve) {
  std::array<Vec3,2> normals{{inwardCapNormal(model,0),inwardCapNormal(model,1)}};
  const std::array<int,2> offsets{{0,int(x.size())-6}};
  Eigen::Vector2d gaps,rhs;for(int i=0;i<2;++i){gaps[i]=normals[i].dot(x.segment<3>(offsets[i]));rhs[i]=gaps[i]+normals[i].dot(step.segment<3>(offsets[i]));}
  const int violated=(rhs[0]<0?1:0)|(rhs[1]<0?2:0);if(!violated)return;
  const Eigen::VectorXd unconstrained=step;Eigen::Matrix<double,Eigen::Dynamic,2> inverse_normals(x.size(),2);
  std::array<bool,2> ready{{false,false}};
  auto attempt=[&](int mask){Eigen::Matrix2d schur=Eigen::Matrix2d::Identity();Eigen::Vector2d b=Eigen::Vector2d::Zero();
    for(int i=0;i<2;++i)if((mask>>i)&1){
      if(!ready[i]){Eigen::VectorXd v=Eigen::VectorXd::Zero(x.size()),answer;v.segment<3>(offsets[i])=normals[i];solve(v,answer);inverse_normals.col(i)=answer;ready[i]=true;}
      b[i]=rhs[i];}
    for(int i=0;i<2;++i)if((mask>>i)&1)for(int j=0;j<2;++j)
      schur(i,j)=((mask>>j)&1)?normals[i].dot(inverse_normals.col(j).segment<3>(offsets[i])):0;
    const Eigen::Vector2d multipliers=schur.ldlt().solve(b);
    if(multipliers.maxCoeff()>1e-10)return false;
    Eigen::VectorXd candidate=unconstrained;for(int i=0;i<2;++i)if((mask>>i)&1)candidate-=multipliers[i]*inverse_normals.col(i);
    for(int i=0;i<2;++i)if(gaps[i]+normals[i].dot(candidate.segment<3>(offsets[i])) < -1e-9)return false;
    step=std::move(candidate);return true;
  };
  if(attempt(violated))return;
  for(int mask:{3,1,2})if(mask!=violated&&attempt(mask))return;
  throw std::runtime_error("cap constrained Newton direction");
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
      const double base=objective(x,mu,&g,&h,false),volume=sweptVolume(model,x,&vg);
      const double anchor_cost=anchorObjective(model,x,anchors,anchor_mu,&g,&h);
      if(!std::isfinite(base))throw std::runtime_error("free-boundary support interior");
      if(!std::isfinite(anchor_cost))throw std::runtime_error("free-boundary anchor inversion / inclusion");
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
        const double dv=alpha*(polynomial[0]+alpha*(polynomial[1]+alpha*polynomial[2]));if(dv<=-volume)continue;
        next=x+alpha*step;trial_anchors=anchors;
        const double ca=anchorObjective(model,next,trial_anchors,anchor_mu);
        const double other=-std::log1p(dv/volume)+ca-anchor_cost,decrease=-1e-4*alpha*decrement;
        ++metrics.bound_line_evaluations;metrics.line_support_rows+=metrics.active_constraints;
        bool sufficient=line.upperBound(alpha)+other<=decrease;
        if(sufficient)++metrics.bound_line_accepts;
        else {++metrics.exact_line_evaluations;metrics.line_support_rows+=metrics.active_constraints;sufficient=line.delta(alpha)+other<=decrease;}
        if(sufficient){
          StageTimer cert_timer(settings.profile);const bool regular=regularGeometry(model,next);metrics.certificate_ms+=cert_timer.elapsed();
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
