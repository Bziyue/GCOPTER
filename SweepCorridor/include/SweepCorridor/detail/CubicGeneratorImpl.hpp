#pragma once
// SPDX-License-Identifier: MIT
#include "CubicGenerator.hpp"
#include "CubicOptimizer.hpp"
#include "ObstacleCover.hpp"
#include <chrono>
namespace SweepCorridor::internal::cubic {
inline void clearanceInitialization(CubicModel& model,const ObstacleIndex& index,
    const Bounds& bounds,const GeometryOptions& o,double fraction) {
  const double pad=o.robot_radius+o.safety_margin+o.certificate_margin;
  std::vector<int> ids;
  for(int i=0;i<model.basis.count;++i) {
    const auto& c=model.reference[i];
    double radius=std::min({o.local_range,.9*o.max_radius,
        (c-bounds.lower).minCoeff()-pad,(bounds.upper-c).minCoeff()-pad});
    const double reach=radius+pad;
    index.query(c-Vec3::Constant(reach),c+Vec3::Constant(reach),ids);
    for(int id:ids){const auto& sphere=index.obstacles()[id];radius=std::min(radius,(sphere.center-c).norm()-sphere.radius-pad);}
    model.x[6*i+3]=model.x[6*i+5]=std::max(o.min_radius*1.1,fraction*radius);
  }
}
inline CubicResult generateCubic(const std::vector<Vec3>& input_seed,const ObstacleIndex& index,
                          const Bounds& bounds,const GeometryOptions& o,const OptimizerSettings& solver) {
  CubicResult result;const auto start=Clock::now();
  // Validate at this boundary; the model, objective and Newton loop consume
  // the resulting finite, nonduplicate seed without repeating these checks.
  if(!bounds.lower.allFinite()||!bounds.upper.allFinite()||!(bounds.upper.array()>bounds.lower.array()).all()||
     !std::isfinite(o.control_spacing)||!(o.control_spacing>0)||!std::isfinite(o.max_radius)||!(o.min_radius>0)||
     !(o.max_radius>o.min_radius)||!std::isfinite(o.max_offset)||!(o.max_offset>0)||
     !std::isfinite(o.local_range)||!(o.local_range>0)||!std::isfinite(o.local_progress)||!(o.local_progress>0)||
     !std::isfinite(o.budget_ms)||!(o.budget_ms>0)||o.max_refinements<0||o.max_refinements>20||
     o.certificate_depth<0||o.certificate_depth>20||
     !std::isfinite(o.robot_radius)||!(o.robot_radius>=0)||!std::isfinite(o.safety_margin)||!(o.safety_margin>=0)||
     !std::isfinite(o.certificate_margin)||!(o.certificate_margin>=0)||!std::isfinite(o.frame_smoothing)||!(o.frame_smoothing>=0)) {
    result.status=Status::InvalidInput;result.message="invalid cubic generator options/bounds";return result;
  }
  std::vector<Vec3> seed;
  for(const auto& p:input_seed) {
    if(!p.allFinite()){result.status=Status::InvalidInput;return result;}
    if(seed.empty()||(p-seed.back()).norm()>1e-10)seed.push_back(p);
  }
  double length=0;for(size_t i=1;i<seed.size();++i)length+=(seed[i]-seed[i-1]).norm();
  if(seed.size()<2 || !std::isfinite(length) || length<=0){result.status=Status::InvalidInput;return result;}
  const int refinements=o.max_refinements;
  Deadline deadline(o.budget_ms,start);
  try {
    const double pad=o.robot_radius+o.safety_margin;
    for(size_t i=1;i<seed.size();++i) {
      const Vec3 a=seed[i-1],b=seed[i],d=b-a;
      std::vector<int> ids;index.query(a.cwiseMin(b)-Vec3::Constant(pad),a.cwiseMax(b)+Vec3::Constant(pad),ids);
      for(int id:ids){const auto& sphere=index.obstacles()[id];const double t=std::clamp((sphere.center-a).dot(d)/d.squaredNorm(),0.,1.);
        if((sphere.center-a-t*d).norm()<=sphere.radius+pad)throw Failure{Status::SeedCollision,"seed intersects an inflated obstacle"};}
    }
    for(int refinement=0;refinement<=refinements;++refinement) {
      result.refinements=refinement;deadline.check();StageTimer timer(solver.profile);
      const double controls=std::ceil(length/std::ldexp(o.control_spacing,-refinement))+3;
      if(controls>o.max_controls)
        throw Failure{Status::Capacity,"cubic control capacity"};
      CubicModel model=CubicModel::fromSeed(seed,std::ldexp(o.control_spacing,-refinement),o.min_radius*1.1,o.frame_smoothing);
      clearanceInitialization(model,index,bounds,o,solver.initial_radius_fraction);
      result.initialization_ms+=timer.elapsed();
      QueryCache cache;std::vector<detail::SupportPlane> cover;bool initialized=false;
      timer.restart();const CoverLayout layout(model.basis.breaks,seed,bounds,o);
      const double layout_ms=timer.elapsed();result.cover_metrics.topology_ms+=layout_ms;
      result.cover_metrics.total_ms+=layout_ms;++result.cover_metrics.topology_builds;
      timer.restart();auto energy=std::make_shared<CubicEnergy>(model,length,solver);
      result.optimizer.setup_ms+=timer.elapsed();++result.optimizer.energy_builds;
      std::unique_ptr<CubicProblem> initial_problem;
      for(int attempt=0;attempt<2;++attempt) {
        ++result.initialization_attempts;deadline.check();
        if(attempt) {
          for(int i=0;i<model.basis.count;++i) {
            model.x.segment<3>(6*i+3)<<o.min_radius*1.1,0,o.min_radius*1.1;
          }
        }
        try {
          const auto g=model.geometry();
          cover=mapCover(g,index,bounds,o,cache,result.cover_metrics,deadline,layout,solver.profile);
          if(!certify(g,cover).safe())throw Failure{Status::VerificationFailed,"initial full-disk certificate"};
          StageTimer preparation(solver.profile);
          auto test=std::make_unique<CubicProblem>(model,cover,length,o,solver,energy);
          const double preparation_ms=preparation.elapsed();
          result.optimizer.initial_problem_ms+=preparation_ms;result.optimizer.setup_ms+=preparation_ms;
          result.optimizer.initial_pruning_ms+=test->pruning_ms;result.optimizer.pruning_ms+=test->pruning_ms;
          result.optimizer.basis_cache_hits+=test->basis_cache_hits;result.optimizer.basis_cache_misses+=test->basis_cache_misses;
          result.optimizer.group_lookup_reuses+=test->group_lookup_reuses;
          if(!std::isfinite(test->objective(model.x,.003/model.basis.count)))throw Failure{Status::InsufficientClearance,"initial SOC interior"};
          initial_problem=std::move(test);initialized=true;break;
        }catch(const Failure& error){result.message=error.message;}
      }
      if(!initialized)continue;
      ProgressConstraints progress;
      // Longitudinal center motion also needs ordering in straight spans.
      // The old transverse-only model guaranteed this there by construction.
      for(int i=0;i<int(model.basis.spans.size());++i)progress.spans.push_back(i);
      for(int iteration=0;iteration<OptimizerSettings::inflation_iterations;++iteration) {
        deadline.check();
        if(iteration)cover=mapCover(model.geometry(),index,bounds,o,cache,result.cover_metrics,deadline,layout,solver.profile);
        timer.restart();
        CubicProblem problem(model,cover,length,o,solver,energy,initial_problem.get(),progress);
        initial_problem.reset();result.optimizer.setup_ms+=timer.elapsed();
        auto candidate=model.x;
        problem.optimize(candidate,result.optimizer,deadline);
        model.x=std::move(candidate);
      }
      // Every accepted iterate satisfies the support and full-disk certificates.
      // Export that state without repeating the same full certificate.
      result.corridor=model.geometry();result.cover=std::move(cover);result.status=Status::Success;result.message.clear();break;
    }
    if(result.status!=Status::Success)result.status=Status::InitializationFailed;
  }catch(const DeadlineExceeded&){result.status=Status::Timeout;result.message="cubic generation deadline";}
   catch(const Failure& error){result.status=error.status;result.message=error.message;}
   catch(const std::exception& error){result.status=Status::NumericalFailure;result.message=error.what();}
  result.total_ms=elapsedMs(start);return result;
}
}
