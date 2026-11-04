// SPDX-License-Identifier: MIT
#include "cubic_generator.hpp"
#include <chrono>
namespace gcopter::direct::native {
static void clearanceInitialization(CubicModel& model,const ObstacleIndex& index,
    const Bounds& bounds,const Options& o,double fraction) {
  const double pad=o.robot_radius+o.safety_margin+o.certificate_margin;
  std::vector<int> ids;
  for(int i=0;i<model.basis.count;++i) {
    const auto& c=model.reference[i];
    double radius=std::min({o.local_range,.9*o.max_radius,
        (c-bounds.lower).minCoeff()-pad,(bounds.upper-c).minCoeff()-pad});
    const double reach=radius+pad;
    index.query(c-Vec3::Constant(reach),c+Vec3::Constant(reach),ids);
    for(int id:ids){const auto& sphere=index.obstacles()[id];radius=std::min(radius,(sphere.center-c).norm()-sphere.radius-pad);}
    model.x[5*i+2]=model.x[5*i+4]=std::max(o.min_radius*1.1,fraction*radius);
  }
}
CubicResult generateCubic(const std::vector<Vec3>& input_seed,const ObstacleIndex& index,
                          const Bounds& bounds,const Options& o,const OptimizerSettings& solver,
                          const Geometry* projected_reference) {
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
    result.status="INVALID_INPUT";result.message="invalid cubic generator options/bounds";return result;
  }
  std::vector<Vec3> seed;
  for(const auto& p:input_seed) {
    if(!p.allFinite()){result.status="INVALID_INPUT";return result;}
    if(seed.empty()||(p-seed.back()).norm()>1e-10)seed.push_back(p);
  }
  double length=0;for(size_t i=1;i<seed.size();++i)length+=(seed[i]-seed[i-1]).norm();
  if(seed.size()<2 || !std::isfinite(length) || length<=0){result.status="INVALID_INPUT";return result;}
  const int refinements=projected_reference?0:o.max_refinements;
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
      const double controls=projected_reference?projected_reference->pieces()+3:
          std::ceil(length/std::ldexp(o.control_spacing,-refinement))+3;
      if(controls>o.max_controls)
        throw Failure{Status::Capacity,"cubic control capacity"};
      CubicModel model=projected_reference?CubicModel::project(naturalCubic(*projected_reference)):
        CubicModel::fromSeed(seed,std::ldexp(o.control_spacing,-refinement),o.min_radius*1.1,o.frame_smoothing);
      if(!projected_reference)clearanceInitialization(model,index,bounds,o,solver.initial_radius_fraction);
      result.initialization_ms+=timer.elapsed();
      QueryCache cache;std::vector<detail::SupportPlane> cover;bool initialized=false;
      timer.restart();const CoverLayout layout(model.basis.breaks,seed,bounds,o);
      const double layout_ms=timer.elapsed();result.cover_metrics.topology_ms+=layout_ms;
      result.cover_metrics.total_ms+=layout_ms;++result.cover_metrics.topology_builds;
      timer.restart();auto energy=std::make_shared<CubicEnergy>(model,length,solver);
      result.optimizer.setup_ms+=timer.elapsed();++result.optimizer.energy_builds;
      const auto original=model.x;
      std::unique_ptr<CubicProblem> initial_problem;
      for(int attempt=0;attempt<(projected_reference?20:2);++attempt) {
        ++result.initialization_attempts;deadline.check();
        if(attempt) {
          const double fraction=std::pow(.8,attempt);
          for(int i=0;i<model.basis.count;++i) {
            if(projected_reference)model.x.segment<3>(5*i+2)=fraction*original.segment<3>(5*i+2);
            else model.x.segment<3>(5*i+2)<<o.min_radius*1.1,0,o.min_radius*1.1;
          }
          result.optimizer.initial_scale=fraction;
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
      if(solver.progress_threshold>=0) {
        const auto g=model.geometry();
        for(int i=0;i<int(g.pieces());++i) {
          const auto a=g.section(g.breaks[i]),b=g.section(g.breaks[i+1]);
          const Vec3 na=a.shape.col(0).cross(a.shape.col(1)).normalized(),nb=b.shape.col(0).cross(b.shape.col(1)).normalized();
          if((nb-na).norm()>solver.progress_threshold)progress.spans.push_back(i);
        }
      }
      for(int iteration=0;iteration<OptimizerSettings::inflation_iterations;++iteration) {
        deadline.check();
        if(iteration)cover=mapCover(model.geometry(),index,bounds,o,cache,result.cover_metrics,deadline,layout,solver.profile);
        timer.restart();
        auto current=solver;
        current.check_gradients=solver.check_gradients&&!iteration;
        CubicProblem problem(model,cover,length,o,current,energy,initial_problem.get(),progress);
        initial_problem.reset();result.optimizer.setup_ms+=timer.elapsed();
        const Eigen::VectorXd initial=model.x;
        auto candidate=model.x;
        problem.optimize(candidate,result.optimizer,deadline);
        for(int repair=0;repair<3;++repair) {
          timer.restart();
          const auto g=model.geometry(candidate);std::vector<int> failed;
          for(int i=0;i<int(g.pieces());++i){Certificate c;if(!jacobianCertificate<3>(g.curve<3>(i),c))failed.push_back(i);}
          result.optimizer.certificate_ms+=timer.elapsed();
          if(failed.empty())break;
          progress.spans.insert(progress.spans.end(),failed.begin(),failed.end());
          std::sort(progress.spans.begin(),progress.spans.end());
          progress.spans.erase(std::unique(progress.spans.begin(),progress.spans.end()),progress.spans.end());
          progress.strict_spans.insert(progress.strict_spans.end(),failed.begin(),failed.end());
          std::sort(progress.strict_spans.begin(),progress.strict_spans.end());
          progress.strict_spans.erase(std::unique(progress.strict_spans.begin(),progress.strict_spans.end()),progress.strict_spans.end());
          current.check_gradients=false;++result.optimizer.regularity_repairs;
          timer.restart();
          CubicProblem repaired(model,cover,length,o,current,energy,&problem,progress);
          result.optimizer.setup_ms+=timer.elapsed();
          auto feasible=[&](double alpha) {
            const Eigen::VectorXd value=initial+alpha*(candidate-initial);
            for(const auto& cut:repaired.cuts) {
              const auto v=cut.apply(value);
              if(cut.bound-v[0]<=v.tail<2>().norm()+1e-10)return false;
            }
            return true;
          };
          double lo=0,hi=1;
          if(feasible(1))lo=1;
          else for(int i=0;i<20;++i){const double mid=.5*(lo+hi);if(feasible(mid))lo=mid;else hi=mid;}
          candidate=initial+(.99*lo)*(candidate-initial);
          repaired.optimize(candidate,result.optimizer,deadline);
        }
        timer.restart();
        for(int backtrack=0;;++backtrack) {
          model.x=initial+std::ldexp(1.,-backtrack)*(candidate-initial);
          if(certify(model.geometry(),cover).safe())break;
          ++result.optimizer.regularity_backtracks;
          if(backtrack==32){model.x=initial;break;}
        }
        result.optimizer.certificate_ms+=timer.elapsed();
        if(iteration+1==OptimizerSettings::inflation_iterations)
          problem.refineVolume(model.x,result.optimizer,[&](const Eigen::VectorXd& x){deadline.check();return certify(model.geometry(x),cover).safe();});
      }
      // Every accepted iterate (including volume refinement) is certified.
      // Export that state without repeating the same full certificate.
      result.corridor=model.geometry();result.cover=std::move(cover);result.status="SUCCESS";result.message.clear();break;
    }
    if(result.status!="SUCCESS")result.status="INITIALIZATION_FAILED";
  }catch(const DeadlineExceeded&){result.status="TIMEOUT";result.message="cubic generation deadline";}
   catch(const Failure& error){result.status=statusName(error.status);result.message=error.message;}
   catch(const std::exception& error){result.status="NUMERICAL_FAILURE";result.message=error.what();}
  result.total_ms=elapsedMs(start);return result;
}
}
