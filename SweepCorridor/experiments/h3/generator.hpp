// SPDX-License-Identifier: MIT
#pragma once
#include "solve.hpp"
#include "jets.hpp"
namespace SweepCorridor::experiments::h3 {
using c3::QueryCache;using c3::CoverLayout;using c3::mapCover;using c3::certify;
using c3::Failure;using c3::DeadlineExceeded;
struct Result : c3::CubicResult { JetMetrics jets; };
inline Result generate(const std::vector<Vec3>& input_seed,const ObstacleIndex& index,
                          const Bounds& bounds,const GeometryOptions& o,const OptimizerSettings& solver) {
  const bool refresh_jets=solver.jet_schedule!=JetSchedule::Frozen,early_jets=solver.jet_schedule==JetSchedule::BeforeTightening;
  Result result;auto& jm=result.jets;const auto start=Clock::now();
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
    result.status=Status::InvalidInput;result.message="invalid H3 generator options/bounds";return result;
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
        throw Failure{Status::Capacity,"H3 node capacity"};
      c3::CubicModel reference=c3::CubicModel::fromSeed(seed,std::ldexp(o.control_spacing,-refinement),o.min_radius*1.1,o.frame_smoothing);
      c3::clearanceInitialization(reference,index,bounds,o,solver.initial_radius_fraction);
      Model model(reference.geometry());
      result.initialization_ms+=timer.elapsed();
      QueryCache cache;std::vector<internal::detail::SupportPlane> cover;bool initialized=false;
      timer.restart();const CoverLayout layout(model.basis.breaks,seed,bounds,o);
      const double layout_ms=timer.elapsed();result.cover_metrics.topology_ms+=layout_ms;
      result.cover_metrics.total_ms+=layout_ms;++result.cover_metrics.topology_builds;
      timer.restart();auto energy=std::make_shared<Energy>(model,length,solver);
      result.optimizer.setup_ms+=timer.elapsed();++result.optimizer.energy_builds;
      std::unique_ptr<Problem> initial_problem;
      for(int attempt=0;attempt<2;++attempt) {
        ++result.initialization_attempts;deadline.check();
        if(attempt) {
          for(int i=0;i<reference.basis.count;++i)reference.x.segment<3>(6*i+3)<<o.min_radius*1.1,0,o.min_radius*1.1;
          model=Model(reference.geometry());
          energy=std::make_shared<Energy>(model,length,solver);
        }
        try {
          const auto g=model.geometry();
          cover=mapCover(g,index,bounds,o,cache,result.cover_metrics,deadline,layout,solver.profile);
          if(!certify(g,cover).safe())throw Failure{Status::VerificationFailed,"initial full-disk certificate"};
          StageTimer preparation(solver.profile);
          auto test=std::make_unique<Problem>(model,cover,length,o,solver,energy,nullptr,false);
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
      for(int iteration=0;iteration<solver.inflation_iterations;++iteration) {
        deadline.check();
        if(iteration)cover=mapCover(model.geometry(),index,bounds,o,cache,result.cover_metrics,deadline,layout,solver.profile);
        timer.restart();
        Problem problem(model,cover,length,o,solver,energy,initial_problem.get(),true);
        initial_problem.reset();result.optimizer.setup_ms+=timer.elapsed();
        auto candidate=model.x;
        problem.optimize(candidate,result.optimizer,deadline,0,early_jets?1:3);
        model.x=std::move(candidate);
        if(refresh_jets){
          updateJets(model,cover,jm,deadline,solver.profile);
          if(early_jets){
            timer.restart();energy=std::make_shared<Energy>(model,length,solver);
            Problem corrector(model,cover,length,o,solver,energy);
            result.optimizer.setup_ms+=timer.elapsed();++result.optimizer.energy_builds;
            corrector.optimize(model.x,result.optimizer,deadline,1,3);
            updateJets(model,cover,jm,deadline,solver.profile);
          }
          if(iteration+1<solver.inflation_iterations){timer.restart();energy=std::make_shared<Energy>(model,length,solver);
            result.optimizer.setup_ms+=timer.elapsed();++result.optimizer.energy_builds;}
        }
      }
      if(solver.polish){
        timer.restart();energy=std::make_shared<Energy>(model,length,solver);
        Problem final_problem(model,cover,length,o,solver,energy);
        result.optimizer.setup_ms+=timer.elapsed();++result.optimizer.energy_builds;
        final_problem.optimize(model.x,result.optimizer,deadline,2,3);
        updateJets(model,cover,jm,deadline,solver.profile);
      }
      // Every accepted iterate satisfies the support and full-disk certificates.
      // Export that state without repeating the same full certificate.
      result.corridor=model.geometry();result.cover=std::move(cover);result.status=Status::Success;result.message.clear();break;
    }
    if(result.status!=Status::Success)result.status=Status::InitializationFailed;
  }catch(const DeadlineExceeded&){result.status=Status::Timeout;result.message="H3 generation deadline";}
   catch(const Failure& error){result.status=error.status;result.message=error.message;}
   catch(const std::exception& error){result.status=Status::NumericalFailure;result.message=error.what();}
  result.total_ms=c3::elapsedMs(start);return result;
}
}
