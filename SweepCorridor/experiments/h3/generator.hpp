// SPDX-License-Identifier: MIT
#pragma once
#include "solve.hpp"
#include "jets.hpp"
#include "volume_jets.hpp"
#include "joint_jets.hpp"
#include "end_blocks.hpp"
#include "local_refinement.hpp"
#include "wall_quality.hpp"
#include "inflation_cover.hpp"
namespace SweepCorridor::experiments::h3 {
using c3::QueryCache;using c3::CoverLayout;using c3::mapCover;using c3::certify;
using c3::Failure;using c3::DeadlineExceeded;
struct Result : c3::CubicResult { JetMetrics jets; bool coarse_start_used=false,intermediate_radius_used=false; };
inline Result generate(const std::vector<Vec3>& input_seed,const ObstacleIndex& index,
                          const Bounds& bounds,const GeometryOptions& o,const OptimizerSettings& configuration) {
  auto solver=configuration;
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
     !std::isfinite(o.certificate_margin)||!(o.certificate_margin>=0)||!std::isfinite(o.frame_smoothing)||!(o.frame_smoothing>=0)||
     (solver.refresh_rounds>0&&(!solver.joint_final_only||solver.joint_jet_steps<=0))||
     !(solver.initial_radius_fraction>0 && solver.initial_radius_fraction<=1)||solver.fine_iterations<=0||
     !(solver.group_axis_limit>0)||!(solver.bending_weight>=0)||!std::isfinite(solver.bending_weight)||
     !(solver.smoothing_volume_tolerance>=0 && solver.smoothing_volume_tolerance<1)||
     !(solver.min_center_progress>=0 && solver.min_center_progress<1)||!(solver.cap_volume_weight>=0 && solver.cap_volume_weight<=1)||solver.cover_group_size<1||solver.inflation_iterations<1||(solver.end_steps>0&&solver.end_nodes<=0)||
     !(solver.relative_jacobian_margin>=1e-9 && solver.relative_jacobian_margin<1)) {
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
      deadline.check();
      if(!index.segmentFree(seed[i-1],seed[i],pad))
        throw Failure{Status::SeedCollision,"seed intersects an inflated obstacle"};
    }
    const bool coarse_start=solver.coarse_start && refresh_jets && !early_jets &&
                            solver.inflation_iterations>=2 && length>2*o.control_spacing;
    for(int refinement=coarse_start?-1:0;refinement<=refinements;++refinement) {
      solver.bending_weight=configuration.bending_weight;
      solver.corrector_iterations=configuration.corrector_iterations;
      result.refinements=std::max(0,refinement);deadline.check();StageTimer timer(solver.profile);
      const double controls=std::ceil(length/std::ldexp(o.control_spacing,-result.refinements))+3;
      if(controls>o.max_controls)
        throw Failure{Status::Capacity,"H3 node capacity"};
      c3::CubicModel reference=c3::CubicModel::fromSeed(seed,std::ldexp(o.control_spacing,-refinement),o.min_radius*1.1,o.frame_smoothing);
      c3::clearanceInitialization(reference,index,bounds,o,solver.initial_radius_fraction);
      Model model(reference.geometry());
      result.initialization_ms+=timer.elapsed();
      QueryCache cache;std::vector<internal::detail::SupportPlane> cover;bool initialized=false,used_intermediate=false;
      timer.restart();CoverLayout layout(model.basis.breaks,seed,bounds,o);
      const double layout_ms=timer.elapsed();result.cover_metrics.topology_ms+=layout_ms;
      result.cover_metrics.total_ms+=layout_ms;++result.cover_metrics.topology_builds;
      timer.restart();auto energy=std::make_shared<Energy>(model,length,solver);
      result.optimizer.setup_ms+=timer.elapsed();++result.optimizer.energy_builds;
      std::unique_ptr<Problem> initial_problem;
      std::shared_ptr<const Problem::Constraints> final_constraints;
      for(int attempt=0;attempt<(solver.intermediate_radius?3:2);++attempt) {
        ++result.initialization_attempts;deadline.check();
        if(attempt) {
          const Vec3 minimum(o.min_radius*1.1,0,o.min_radius*1.1);
          const double fraction=solver.intermediate_radius && attempt==1 ? .5 : 0.;
          // A reduced-radius start permits much sharper walls during inflation.
          // Use stronger bending regularization for this certified candidate.
          solver.bending_weight=fraction>0 ? configuration.bending_weight/fraction : configuration.bending_weight;
          solver.corrector_iterations=fraction>0 && configuration.corrector_iterations>0
              ? std::max(3,configuration.corrector_iterations) : configuration.corrector_iterations;
          for(int i=0;i<reference.basis.count;++i) {
            auto radii=reference.x.segment<3>(6*i+3);
            radii=minimum+fraction*(radii-minimum);
          }
          model=Model(reference.geometry());
          energy=std::make_shared<Energy>(model,length,solver);
        }
        try {
          const auto g=model.geometry();
          cover=inflationCover(g,index,bounds,o,cache,result.cover_metrics,deadline,layout,solver);
          if(!certify(g,cover,o.certificate_depth,solver.regularity(length)).safe())throw Failure{Status::VerificationFailed,"initial full-disk certificate"};
          StageTimer preparation(solver.profile);
          auto test=std::make_unique<Problem>(model,cover,length,o,solver,energy,nullptr,false);
          const double preparation_ms=preparation.elapsed();
          result.optimizer.initial_problem_ms+=preparation_ms;result.optimizer.setup_ms+=preparation_ms;
          result.optimizer.initial_pruning_ms+=test->pruning_ms;result.optimizer.pruning_ms+=test->pruning_ms;
          result.optimizer.basis_cache_hits+=test->basis_cache_hits;result.optimizer.basis_cache_misses+=test->basis_cache_misses;
          result.optimizer.group_lookup_reuses+=test->group_lookup_reuses;
          if(!std::isfinite(test->objective(model.x,.003/model.basis.count)))throw Failure{Status::InsufficientClearance,"initial SOC interior"};
          initial_problem=std::move(test);initialized=true;used_intermediate=solver.intermediate_radius && attempt==1;break;
        }catch(const Failure& error){result.message=error.message;}
      }
      if(!initialized)continue;
      bool retry_fine=false;
      const int passes=solver.inflation_iterations+(refinement<0);
      for(int iteration=0;iteration<passes;++iteration) {
        deadline.check();
        const bool lift=refinement<0 && iteration+1==passes;
        if(iteration) {
          try {
            cover=inflationCover(model.geometry(),index,bounds,o,cache,result.cover_metrics,deadline,layout,solver);
          } catch(const Failure& error) {
            // Every accepted update has a certificate in the previous complete
            // cover. A failed change of separating planes can keep using it.
            if(error.status!=Status::InsufficientClearance) throw;
          }
        }
        if(lift) {
          timer.restart();
          model=bisectModel(model,reference.geometry());
          cover=bisectCover(cover);
          if(!model.x.allFinite() || !certify(model.geometry(),cover,o.certificate_depth,solver.regularity(length)).safe()) {
            retry_fine=true;break;
          }
          cache.spans.clear(); // Candidate packets remain keyed by the physical query box.
          layout=CoverLayout(model.basis.breaks,seed,bounds,o);
          ++result.cover_metrics.topology_builds;
          energy=std::make_shared<Energy>(model,length,solver);
          result.optimizer.setup_ms+=timer.elapsed();
          ++result.optimizer.energy_builds;
        }
        timer.restart();
        auto pass_settings=solver;
        if(!lift && iteration+1<passes) pass_settings.cap_volume_weight=1.;
        if(lift) {
          pass_settings.iterations=std::min(solver.fine_iterations,solver.iterations);
          pass_settings.corrector_iterations=std::min(2,solver.corrector_iterations);
        }
        Problem problem(model,cover,length,o,pass_settings,energy,initial_problem.get(),true);
        if(lift && !std::isfinite(problem.objective(model.x,.003/model.basis.count))) {
          retry_fine=true;break;
        }
        final_constraints=problem.geometry;
        initial_problem.reset();result.optimizer.setup_ms+=timer.elapsed();
        auto candidate=model.x;
        problem.optimize(candidate,result.optimizer,deadline,0,early_jets?1:3);
        model.x=std::move(candidate);
        if(refresh_jets){
          updateJets(model,cover,jm,deadline,solver.profile,solver.regularity(length));
          if(solver.volume_jet_steps)optimizeVolumeJets(model,cover,length,solver,jm,deadline);
          if(solver.joint_jet_steps&&!solver.joint_final_only)optimizeJointJets(model,cover,length,o,solver,jm,deadline,solver.reuse_joint_supports?problem.geometry.get():nullptr);
          if(early_jets){
            timer.restart();energy=std::make_shared<Energy>(model,length,solver);
            Problem corrector(model,cover,length,o,solver,energy);
            result.optimizer.setup_ms+=timer.elapsed();++result.optimizer.energy_builds;
            corrector.optimize(model.x,result.optimizer,deadline,1,3);
            updateJets(model,cover,jm,deadline,solver.profile,solver.regularity(length));
          }
          if(iteration+1<passes && !(refinement<0 && iteration+2==passes)){timer.restart();energy=std::make_shared<Energy>(model,length,solver);
            result.optimizer.setup_ms+=timer.elapsed();++result.optimizer.energy_builds;}
        }
      }
      if(retry_fine)continue;
      JointFeedback feedback;
      if(solver.joint_jet_steps&&solver.joint_final_only)optimizeJointJets(model,cover,length,o,solver,jm,deadline,solver.reuse_joint_supports?final_constraints.get():nullptr,
        nullptr,(solver.refresh_rounds||solver.joint_budget_fraction<1||solver.joint_budget_share<1)?&feedback:nullptr);
      if(solver.refresh_rounds)refineLocal(model,cover,length,index,bounds,o,solver,cache,result.cover_metrics,layout,jm,deadline,feedback);
      else if(solver.end_steps)optimizeEnds(model,o,length,solver,*final_constraints,jm,deadline);
      if(solver.polish){
        timer.restart();energy=std::make_shared<Energy>(model,length,solver);
        Problem final_problem(model,cover,length,o,solver,energy);
        result.optimizer.setup_ms+=timer.elapsed();++result.optimizer.energy_builds;
        final_problem.optimize(model.x,result.optimizer,deadline,2,3);
        updateJets(model,cover,jm,deadline,solver.profile,solver.regularity(length));
      }
      if(used_intermediate && solver.smoothing_volume_tolerance>0)
        smoothCore(model,cover,length,solver,jm,deadline);
      // Every accepted iterate satisfies the support and full-disk certificates.
      // Export that state without repeating the same full certificate.
      result.coarse_start_used=refinement<0;result.intermediate_radius_used=used_intermediate;
      result.corridor=model.geometry();result.cover=std::move(cover);result.status=Status::Success;result.message.clear();break;
    }
    if(result.status!=Status::Success)result.status=Status::InitializationFailed;
  }catch(const DeadlineExceeded&){result.status=Status::Timeout;result.message="H3 generation deadline";}
   catch(const Failure& error){result.status=error.status;result.message=error.message;}
   catch(const std::exception& error){result.status=Status::NumericalFailure;result.message=error.what();}
  result.total_ms=c3::elapsedMs(start);return result;
}
}
