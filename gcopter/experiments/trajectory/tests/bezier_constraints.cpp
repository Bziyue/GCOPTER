#include "TrajectoryOptAdapters/BezierPenaltyCost.hpp"
#include <iostream>
#include <stdexcept>

using Spline=SplineTrajectory::QuinticSplineND<3>;
using Hull=SplineTrajectory::ConvexHullWorkspace<3>;
void require(bool good,const char* message){if(!good)throw std::runtime_error(message);}

int main() {
  SplineTrajectory::SplineOptimizer<Spline> optimizer;
  decltype(optimizer)::Problem problem;
  problem.durations={1.1,1.3};
  problem.waypoints.resize(3,3);
  problem.waypoints<<0,0,1, 1,0.8,1.5, 2,0,1;
  require(bool(optimizer.prepare(problem)),"prepare");
  traj_opt_components::PolyhedraH polys(1);
  polys[0].resize(3,4);polys[0]<<0,1,0,-0.2, 0,-1,0,-0.2, 0,0,1,-1.1;
  Eigen::VectorXi indices=Eigen::VectorXi::Zero(2);
  traj_opt_adapters::BezierPenaltyCost<Spline,traj_opt_adapters::SegmentPolytopePenalty> cost;
  cost.position={&polys,&indices,0.01,13};
  cost.configure(2,2,0.3,7,0.01);
  struct Objective{decltype(cost)& coefficient;} objective{cost};
  auto x=optimizer.initialGuess();
  const auto checked=SplineTrajectory::checkGradients(optimizer,x,objective);
  std::cout<<"active position + speed gradient error "<<checked.relative_error<<'\n';
  require(checked.relative_error<2e-6,"active Bézier penalty full gradient");
  require(cost.max_position_violation>0&&cost.max_speed_violation>0,"test constraints must be active");

  // 两端位于 y=0，端点采样看不到中段越界；控制点约束必须捕获它。
  SplineTrajectory::BoundaryConditions<3> bc;
  bc.start_velocity=Eigen::Vector3d(1,6,0);bc.end_velocity=Eigen::Vector3d(1,-6,0);
  Spline::CoefficientMatrix points(2,3);points<<0,0,0, 1,0,0;
  Spline curve({1.0},points,0.0,bc);
  require(curve.isValid(),"overshoot spline");
  auto coarse=Hull::fromPPoly(curve.polynomial(),SplineTrajectory::ConvexHullBasis::Bezier,0,0);
  auto fine=Hull::fromPPoly(curve.polynomial(),SplineTrajectory::ConvexHullBasis::Bezier,0,2);
  const double midpoint=curve.polynomial().evaluate(0.5).y();
  require(midpoint>0.2,"midpoint violates y <= 0.2");
  require(fine.controls().col(1).maxCoeff()>=midpoint-1e-10,"hull detects between-node overshoot");
  require(fine.controls().col(1).maxCoeff()<coarse.controls().col(1).maxCoeff(),"subdivision tightens bound");
  for(int piece=0;piece<fine.numPieces();++piece) {
    const auto& info=fine.pieceInfo(piece);
    for(int k=0;k<=16;++k) {
      const double alpha=k/16.0;
      Eigen::Matrix<double,6,3> cp=fine.controls().middleRows(piece*6,6);
      for(int degree=5;degree>0;--degree)
        for(int i=0;i<degree;++i)cp.row(i)=(1-alpha)*cp.row(i)+alpha*cp.row(i+1);
      auto p=curve.polynomial().evaluate(info.start_time+alpha*info.duration);
      require((cp.row(0).transpose()-p).norm()<1e-10,"de Casteljau preserves curve");
    }
  }
  std::cout<<"PASS: overshoot, subdivision, reconstruction and active constraints\n";
}
