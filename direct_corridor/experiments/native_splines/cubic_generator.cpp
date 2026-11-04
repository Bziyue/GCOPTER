// SPDX-License-Identifier: MIT
#include "cubic_generator.hpp"
#include <chrono>
namespace gcopter::direct::native {
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
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
static bool recenterFrames(CubicModel& model,const std::vector<detail::SupportPlane>& cover,
    double length,const Options& o,const OptimizerSettings& settings) {
  const auto geometry=model.geometry();
  std::vector<Vec3> centers(model.basis.count);
  std::vector<Eigen::AngleAxisd> rotations;
  for(int i=0;i<model.basis.count;++i) {
    centers[i]=model.reference[i]+model.frames[i]*model.x.segment<2>(5*i);
    const Vec3 tangent=geometry.section(model.basis.greville(i)).first.normalized();
    const Vec3 normal=model.frames[i].col(0).cross(model.frames[i].col(1));
    rotations.emplace_back(Eigen::Quaterniond::FromTwoVectors(normal,tangent));
  }
  for(double fraction:{1.,.5,.25,.125,.0625}) {
    CubicModel candidate=model;
    for(int i=0;i<model.basis.count;++i) {
      candidate.reference[i]=centers[i];candidate.x.segment<2>(5*i).setZero();
      candidate.frames[i]=Eigen::AngleAxisd(fraction*rotations[i].angle(),rotations[i].axis()).toRotationMatrix()*model.frames[i];
      candidate.x.segment<3>(5*i+2)*=.95;
    }
    if(!certify(candidate.geometry(),cover).safe())continue;
    CubicProblem problem(candidate,cover,length,o,settings);
    if(!std::isfinite(problem.objective(candidate.x,.003/candidate.basis.count)))continue;
    model=std::move(candidate);return true;
  }
  return false;
}
CubicResult generateCubic(const std::vector<Vec3>& input_seed,const ObstacleIndex& index,
                          const Bounds& bounds,const Options& o,const OptimizerSettings& solver,
                          const Geometry* projected_reference) {
  CubicResult result;const auto start=Clock::now();
  const std::function<void()> stop=[&]{if(ms(start)>o.budget_ms)throw Failure{Status::Timeout,"cubic generation deadline"};};
  // Validate at this boundary; the model, objective and Newton loop consume
  // the resulting finite, nonduplicate seed without repeating these checks.
  if(!bounds.lower.allFinite()||!bounds.upper.allFinite()||!(bounds.upper.array()>bounds.lower.array()).all()||
     !(o.control_spacing>0)||!(o.min_radius>0)||!(o.max_radius>o.min_radius)||!(o.local_range>0)||!(o.local_progress>0)||
     !(o.budget_ms>0)||o.max_refinements<0||o.max_refinements>20) {
    result.status="INVALID_INPUT";result.message="invalid cubic generator options/bounds";return result;
  }
  std::vector<Vec3> seed;
  for(const auto& p:input_seed) {
    if(!p.allFinite()){result.status="INVALID_INPUT";return result;}
    if(seed.empty()||(p-seed.back()).norm()>1e-10)seed.push_back(p);
  }
  double length=0;for(size_t i=1;i<seed.size();++i)length+=(seed[i]-seed[i-1]).norm();
  if(seed.size()<2 || length<=0){result.status="INVALID_INPUT";return result;}
  const int refinements=projected_reference?0:o.max_refinements;
  try {
    const double pad=o.robot_radius+o.safety_margin;
    for(size_t i=1;i<seed.size();++i) {
      const Vec3 a=seed[i-1],b=seed[i],d=b-a;
      std::vector<int> ids;index.query(a.cwiseMin(b)-Vec3::Constant(pad),a.cwiseMax(b)+Vec3::Constant(pad),ids);
      for(int id:ids){const auto& sphere=index.obstacles()[id];const double t=std::clamp((sphere.center-a).dot(d)/d.squaredNorm(),0.,1.);
        if((sphere.center-a-t*d).norm()<=sphere.radius+pad)throw Failure{Status::SeedCollision,"seed intersects an inflated obstacle"};}
    }
    for(int refinement=0;refinement<=refinements;++refinement) {
      result.refinements=refinement;stop();auto timer=Clock::now();
      if(!projected_reference && std::ceil(length/std::ldexp(o.control_spacing,-refinement))+3>o.max_controls)
        throw Failure{Status::Capacity,"cubic control capacity"};
      CubicModel model=projected_reference?CubicModel::project(naturalCubic(*projected_reference)):
        CubicModel::fromSeed(seed,std::ldexp(o.control_spacing,-refinement),o.min_radius*1.1,o.frame_smoothing);
      if(!projected_reference && solver.initial_radius_fraction>0)clearanceInitialization(model,index,bounds,o,solver.initial_radius_fraction);
      result.initialization_ms+=ms(timer);
      if(size_t(model.basis.count)>o.max_controls)throw Failure{Status::Capacity,"cubic control capacity"};
      QueryCache cache;std::vector<detail::SupportPlane> cover;bool initialized=false;
      const auto original=model.x;
      for(int attempt=0;attempt<(projected_reference?20:solver.initial_radius_fraction>0?2:1);++attempt) {
        ++result.initialization_attempts;stop();
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
          cover=mapCover(g,seed,index,bounds,o,cache,result.cover_metrics,stop);
          if(!certify(g,cover).safe())throw Failure{Status::VerificationFailed,"initial full-disk certificate"};
          CubicProblem test(model,cover,length,o,solver);
          if(!std::isfinite(test.objective(model.x,.003/model.basis.count)))throw Failure{Status::InsufficientClearance,"initial SOC interior"};
          initialized=true;break;
        }catch(const Failure& error){result.message=error.message;}
      }
      if(!initialized)continue;
      std::vector<int> known_progress=solver.progress_spans;
      if(solver.progress_threshold>=0) {
        const auto g=model.geometry();
        for(int i=0;i<int(g.pieces());++i) {
          const auto a=g.section(g.breaks[i]),b=g.section(g.breaks[i+1]);
          const Vec3 na=a.shape.col(0).cross(a.shape.col(1)).normalized(),nb=b.shape.col(0).cross(b.shape.col(1)).normalized();
          if((nb-na).norm()>solver.progress_threshold)known_progress.push_back(i);
        }
      }
      for(int iteration=0;iteration<o.inflation_iterations;++iteration) {
        stop();
        if(iteration) {
          if(solver.recenter_frames) {
            timer=Clock::now();recenterFrames(model,cover,length,o,solver);
            result.initialization_ms+=ms(timer);
          }
          cover=mapCover(model.geometry(),seed,index,bounds,o,cache,result.cover_metrics,stop);
        }
        timer=Clock::now();
        auto current=solver;current.progress_spans=known_progress;if(!iteration)current.iterations=o.initial_iterations;
        current.check_gradients=solver.check_gradients&&!iteration;
        CubicProblem problem(model,cover,length,o,current);result.optimizer.setup_ms+=ms(timer);
        const Eigen::VectorXd initial=model.x;
        auto candidate=model.x;
        problem.optimize(candidate,result.optimizer,stop);
        for(int repair=0;repair<3;++repair) {
          const auto g=model.geometry(candidate);std::vector<int> failed;
          for(int i=0;i<int(g.pieces());++i){Certificate c;if(!jacobianCertificate<3>(g.curve<3>(i),c))failed.push_back(i);}
          if(failed.empty())break;
          current.progress_spans.insert(current.progress_spans.end(),failed.begin(),failed.end());
          std::sort(current.progress_spans.begin(),current.progress_spans.end());
          current.progress_spans.erase(std::unique(current.progress_spans.begin(),current.progress_spans.end()),current.progress_spans.end());
          current.check_gradients=false;++result.optimizer.regularity_repairs;
          timer=Clock::now();CubicProblem repaired(model,cover,length,o,current);result.optimizer.setup_ms+=ms(timer);
          // The old and proposed points already obey the geometric SOCs.
          // Find an interior warm start using only the new progress cones.
          auto feasible=[&](double alpha) {
            const Eigen::VectorXd value=initial+alpha*(candidate-initial);
            for(size_t j=repaired.geometric_cuts;j<repaired.cuts.size();++j) {
              const auto& cut=repaired.cuts[j];const auto v=cut.apply(value);
              if(cut.bound-v[0]<=v.tail<2>().norm()+1e-10)return false;
            }
            return true;
          };
          double lo=0,hi=1;
          if(feasible(1))lo=1;
          else for(int i=0;i<20;++i){const double mid=.5*(lo+hi);if(feasible(mid))lo=mid;else hi=mid;}
          candidate=initial+(.99*lo)*(candidate-initial);
          repaired.optimize(candidate,result.optimizer,stop);
        }
        known_progress=current.progress_spans;
        timer=Clock::now();
        for(int backtrack=0;;++backtrack) {
          model.x=initial+std::ldexp(1.,-backtrack)*(candidate-initial);
          if(certify(model.geometry(),cover).safe())break;
          ++result.optimizer.regularity_backtracks;
          if(backtrack==32){model.x=initial;break;}
        }
        result.optimizer.certificate_ms+=ms(timer);
      }
      timer=Clock::now();auto g=model.geometry();const auto final=certify(g,cover);
      result.verification_ms+=ms(timer);
      if(!final.safe())throw Failure{Status::VerificationFailed,"final cubic support/Jacobian audit"};
      if(solver.natural_endpoints) {
        double residual=0;
        for(double u:{0.,1.}){const auto s=g.section(u);residual=std::max(residual,pack(s.second,s.shape_second).norm()/(length*length));}
        if(residual>1e-7)throw Failure{Status::VerificationFailed,"natural endpoint residual"};
      }
      result.corridor=std::move(g);result.cover=std::move(cover);result.status="SUCCESS";result.message.clear();break;
    }
    if(result.status!="SUCCESS")result.status="INITIALIZATION_FAILED";
  }catch(const Failure& error){result.status=statusName(error.status);result.message=error.message;}
   catch(const std::exception& error){result.status="NUMERICAL_FAILURE";result.message=error.what();}
  result.total_ms=ms(start);return result;
}
}
