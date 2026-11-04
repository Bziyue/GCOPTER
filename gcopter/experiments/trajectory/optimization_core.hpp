// SPDX-License-Identifier: MIT
// Shared costs for offline benchmarks and the interactive comparison.
#pragma once
#include <functional>
#include "corridor_benchmark_common.hpp"
#include "SplineTrajectory/SplineOptimizer.hpp"
#include "SplineTrajectory/SplineTrajectory.hpp"
#include "TrajectoryOptAdapters/GcopterPenaltyIntegralCostAdapter.hpp"
#include "TrajectoryOptAdapters/BezierPenaltyCost.hpp"
#include "TrajectoryOptComponents/LinearTimeCost.hpp"
#include <cstdint>
#include <cstring>
#include "sweep_query.hpp"
namespace {
using BenchmarkTubeQuery=sweep_benchmark::Query;

using Spline = SplineTrajectory::QuinticSplineND<3>;
using Opt = SplineTrajectory::SplineOptimizer<Spline>;
double polyDistance(const Polys& polys, const Vec3& p, Vec3& grad) {
  double best=std::numeric_limits<double>::infinity();
  for (const auto& h: polys) {
    double worst=-std::numeric_limits<double>::infinity(); int row=0;
    for(int k=0;k<h.rows();++k) {
      double v=h.row(k).head<3>().dot(p)+h(k,3);
      if(v>worst) {worst=v;row=k;}
    }
    if(worst<best) {best=worst;grad=h.row(row).head<3>().transpose();}
  }
  return best; // union of H-polytopes, piecewise differentiable
}
struct Cost {
  traj_opt_adapters::GcopterPenaltyIntegralCostAdapter dynamics;
  const Polys* polys=nullptr;
  int steps=16,pieces=1;
  double weight=1e5;
  BenchmarkTubeQuery* query=nullptr;
  bool ellipse=false;
  bool bezier=false;
  bool keep_tube_integral=false;
  double positionWeight() const {return weight*(bezier&&ellipse&&keep_tube_integral?0.5:1.0);}
  static constexpr int kDerivativeOrder = 3;
  double operator()(const SplineTrajectory::IntegralPointInfo& point,
      const SplineTrajectory::SampleState<3>& state,
      SplineTrajectory::SampleGradient<3>& partial) const {
    auto& stats=query->stats;
    tube_query::Timer integral_timer(stats.enabled,stats.integral_ms);
    double cost;{tube_query::Timer dynamics_timer(stats.enabled,stats.dynamics_ms);cost=dynamics(point,state,partial);}
    if(bezier&&!(ellipse&&keep_tube_integral)) return cost;
    const double position_weight=positionWeight();
    const int seg=point.segment_index,step=point.step_index;
    const auto& p=state.p;auto& gp=partial.p;
    Vec3 gradient;double violation;
    if(ellipse) {
      auto tube=(*query)(p,(seg+double(step)/steps)/pieces);
      cost+=position_weight*tube.plane*tube.plane;gp+=2*position_weight*tube.plane*tube.grad_plane;
      violation=tube.radial;gradient=tube.grad_radial;
    } else violation=polyDistance(*polys,p,gradient);
    if(violation>0){cost+=position_weight*violation*violation;gp+=2*position_weight*violation*gradient;}
    return cost;
  }
};
struct ControlPosition {
  Cost* cost=nullptr;
  std::vector<int> leaf_poly;
  // 一次初始化固定片→凸区域关联，禁止每个控制点各自选择不同多面体。
  void assign(const Spline::Polynomial& polynomial,int depth) {
    if(cost->ellipse)return;
    auto hull=SplineTrajectory::ConvexHullWorkspace<3>::fromPPoly(
        polynomial,SplineTrajectory::ConvexHullBasis::Bezier,0,depth);
    leaf_poly.resize(hull.numPieces());
    for(int piece=0;piece<hull.numPieces();++piece) {
      double best=std::numeric_limits<double>::infinity();int chosen=0;
      for(int poly=0;poly<int(cost->polys->size());++poly) {
        const auto& h=(*cost->polys)[poly];double worst=-std::numeric_limits<double>::infinity();
        for(int cp=0;cp<Spline::kCoefficientCount;++cp)
          worst=std::max(worst,(h.leftCols<3>()*hull.controls().row(piece*Spline::kCoefficientCount+cp).transpose()+h.col(3)).maxCoeff());
        if(worst<best){best=worst;chosen=poly;}
      }
      leaf_poly[piece]=chosen;
    }
  }
  double operator()(int piece,int segment,double alpha,const Vec3& p,Vec3& gradient,double& violation) const {
    double value=0;
    if(cost->ellipse) {
      const double weight=cost->positionWeight();
      const auto tube=(*cost->query)(p,(segment+alpha)/cost->pieces);
      violation=std::max(std::abs(tube.plane),std::max(0.0,tube.radial));
      value=weight*tube.plane*tube.plane;
      gradient=2*weight*tube.plane*tube.grad_plane;
      if(tube.radial>0){value+=weight*tube.radial*tube.radial;gradient+=2*weight*tube.radial*tube.grad_radial;}
    } else {
      const auto& h=(*cost->polys)[leaf_poly[piece]];
      for(int plane=0;plane<h.rows();++plane) {
        const Vec3 n=h.row(plane).head<3>().transpose();const double v=n.dot(p)+h(plane,3);
        violation=std::max(violation,v);
        if(v>0){value+=cost->weight*v*v;gradient+=2*cost->weight*v*n;}
      }
    }
    return value;
  }
};
struct Run {
  Opt optimizer;
  std::unique_ptr<BenchmarkTubeQuery> query;

  traj_opt_components::LinearTimeCost time{20};
  Cost cost;
  traj_opt_adapters::BezierPenaltyCost<Spline,ControlPosition> bezier_cost;
  struct Coefficient {
    Run* run;
    double operator()(const SplineTrajectory::ParameterView<Spline>& p,SplineTrajectory::CoefficientGradient<Spline>& g) {
      auto& stats=run->query->stats;
      tube_query::Timer timer(stats.enabled,stats.coefficient_ms);
      return run->cost.bezier?run->bezier_cost(p,g):0;
    }
  };
  struct Objective { traj_opt_components::LinearTimeCost& duration; Cost& integral; Coefficient coefficient; };
  std::function<bool()> cancelled;
  int evaluations=0,iterations=0;
  static double evaluate(void* ptr,const Eigen::VectorXd& x,Eigen::VectorXd& g) {
    auto& r=*static_cast<Run*>(ptr); ++r.evaluations;
    tube_query::Timer timer(r.query->stats.enabled,r.query->stats.objective_ms);
    Objective objective{r.time,r.cost,{&r}};
    auto value=r.optimizer.evaluate(x,g,objective);
    return value.cost;
  }
  static int progress(void* ptr,const Eigen::VectorXd&,const Eigen::VectorXd&,
      double,double,int k,int) {auto& r=*static_cast<Run*>(ptr);r.iterations=k;return r.cancelled&&r.cancelled()?1:0;}
};

} // namespace
