// SPDX-License-Identifier: MIT
#include "demo_comparison.hpp"
#include "../experiments/trajectory/optimization_core.hpp"
namespace demo_comparison {
FiriCorridor buildFiri(const std::vector<Vec3>& seed,const SweepCorridor::ObstacleIndex& map,
                      const SweepCorridor::Bounds& bounds,const SweepCorridor::Options& options) {
  std::vector<Vec3> points;points.reserve(map.obstacles().size());
  for(const auto& s:map.obstacles())points.push_back(s.center);
  auto raw=generateFiri(seed,points,bounds,map.obstacles().front().radius,
                       options.robot_radius+options.safety_margin,true,options.local_range,
                       options.local_progress,options.certificate_margin);
  FiriCorridor result;result.polys=std::move(raw.polys);result.milliseconds=raw.total_ms;
  for(const auto& h:result.polys) {
    Eigen::Matrix3Xd vertices;
    if(!geo_utils::enumerateVs(h,vertices)||vertices.cols()<4)
      throw std::runtime_error("FIRI polytope has no interior");
    quickhull::QuickHull<double> hull;
    const auto mesh=hull.getConvexHull(vertices.data(),vertices.cols(),false,true);
    const auto& indices=mesh.getIndexBuffer();
    for(size_t i=0;i<indices.size();i+=3) {
      for(int k=0;k<3;++k)result.triangles.push_back(vertices.col(indices[i+k]));
    }
    // Draw true polytope edges; omit the diagonals used to triangulate faces.
    for(int i=0;i<vertices.cols();++i)for(int j=i+1;j<vertices.cols();++j) {
      int shared=0;
      for(int k=0;k<h.rows();++k)
        if(std::abs(h.row(k).head<3>().dot(vertices.col(i))+h(k,3))<1e-5 &&
           std::abs(h.row(k).head<3>().dot(vertices.col(j))+h(k,3))<1e-5)++shared;
      if(shared>=2){result.edges.push_back(vertices.col(i));result.edges.push_back(vertices.col(j));}
    }
  }
  return result;
}
Comparison optimize(const std::vector<Vec3>& seed,const SweepCorridor::Corridor& corridor,
                    const FiriCorridor& firi,const SweepCorridor::ObstacleIndex& map,
                    const SweepCorridor::Bounds& bounds,double padding,
                    const std::shared_ptr<std::atomic_bool>& cancel) {
  std::vector<double> arc{0};for(size_t i=1;i<seed.size();++i)arc.push_back(arc.back()+(seed[i]-seed[i-1]).norm());
  const int pieces=std::max(2,int(std::ceil(arc.back())));
  Opt::Problem problem;problem.waypoints.resize(pieces+1,3);
  for(int i=0;i<=pieces;++i) {
    double s=arc.back()*i/pieces;
    int k=std::clamp(int(std::upper_bound(arc.begin(),arc.end(),s)-arc.begin())-1,0,int(seed.size())-2);
    problem.waypoints.row(i)=(seed[k]+(seed[k+1]-seed[k])*((s-arc[k])/(arc[k+1]-arc[k]))).transpose();
  }
  problem.durations.assign(pieces,arc.back()/pieces/2.0);
  problem.boundary.start_velocity=problem.boundary.end_velocity=Vec3::Zero();
  problem.boundary.start_acceleration=problem.boundary.end_acceleration=Vec3::Zero();
  Comparison comparison;Eigen::VectorXd initial;
  for(int method=0;method<2;++method) {
    if(cancel->load())return comparison;
    const bool ellipse=method==1;auto& result=ellipse?comparison.c3:comparison.firi;
    Run run;SplineTrajectory::OptimizerOptions config;config.energy_weight=1;config.integration_steps=16;
    if(!run.optimizer.prepare(problem,config))throw std::runtime_error("trajectory initialization failed");
    run.query=std::make_unique<BenchmarkTubeQuery>(corridor,false);
    run.cost.query=run.query.get();
    Polys empty(1);empty[0].resize(0,4);Eigen::VectorXi ids=Eigen::VectorXi::Zero(pieces);
    Eigen::VectorXd limits(5),weights(5);limits<<4,2.1,1.05,2,12;weights<<0,1e4,1e4,1e4,1e5;
    flatness::FlatnessMap flat;flat.reset(.61,9.8,.70,.80,.01,.0001);
    run.cost.dynamics.reset(&empty,&ids,.01,limits,weights,&flat);
    run.cost.pieces=pieces;run.cost.polys=&firi.polys;run.cost.steps=16;run.cost.ellipse=ellipse;run.cost.weight=1e5;
    run.cost.bezier=true;run.cost.keep_tube_integral=true;
    Eigen::VectorXd x=run.optimizer.initialGuess(),g(x.size());
    if(method==0)initial=x;else if(x!=initial)throw std::runtime_error("initial trajectory mismatch");
    if(!run.optimizer.updateAccepted(x))throw std::runtime_error("initial spline failed");
    run.cost.dynamics.penalty_weights(1)=0;
    run.bezier_cost.position.cost=&run.cost;
    run.bezier_cost.position.assign(run.optimizer.polynomial(),2);
    run.bezier_cost.configure(pieces,2,limits(0),weights(1),.01);
    run.cancelled=[cancel]{return cancel->load();};
    lbfgs::lbfgs_parameter_t params;params.mem_size=64;params.past=3;params.delta=1e-6;
    params.g_epsilon=0;params.min_step=1e-32;params.max_iterations=800;
    double objective=0;auto start=Clock::now();
    result.status=lbfgs::lbfgs_optimize(x,objective,Run::evaluate,nullptr,Run::progress,&run,params);
    result.optimize_ms=ms(start);result.iterations=run.iterations;
    if(cancel->load())return comparison;
    Run::evaluate(&run,x,g);
    const auto& trajectory=run.optimizer.polynomial();
    result.duration=trajectory.duration();result.jerk=run.optimizer.energy();
    Vec3 previous=trajectory.evaluate(0);std::vector<int> near;
    bool dynamics_ok=true,roots_ok=true;double t0=0;
    for(int seg=0;seg<pieces;++seg) {
      double T=run.optimizer.parameters().durations[seg];int n=std::max(64,int(std::ceil(T/.005)));
      for(int k=0;k<=n;++k) {
        if(cancel->load())return comparison;
        double t=t0+T*k/n,u=(seg+double(k)/n)/pieces;
        Vec3 p=trajectory.evaluate(t),v=trajectory.evaluate(t,1),a=trajectory.evaluate(t,2),j=trajectory.evaluate(t,3),gradient;
        if(!p.allFinite()||!v.allFinite()||!a.allFinite()||!j.allFinite())throw std::runtime_error("nonfinite trajectory");
        result.length+=(p-previous).norm();previous=p;
        if(k%4==0||k==n)result.points.push_back(p);
        if(ellipse){auto q=tubeValue(corridor,p,u);roots_ok&=q.valid;result.max_violation=std::max({result.max_violation,q.radial,std::abs(q.plane)});}
        else result.max_violation=std::max(result.max_violation,polyDistance(firi.polys,p,gradient));
        map.query(p-Vec3::Constant(padding),p+Vec3::Constant(padding),near);
        for(int id:near){const auto& s=map.obstacles()[id];result.penetration=std::max(result.penetration,s.radius+padding-(p-s.center).norm());}
        result.penetration=std::max({result.penetration,(bounds.lower.array()+padding-p.array()).maxCoeff(),(p.array()-bounds.upper.array()+padding).maxCoeff()});
        const auto state=traj_opt_components::evaluateFlatnessPenaltyState(&flat,v,a,j);
        double tilt=std::acos(std::clamp(1-2*(state.quaternion(1)*state.quaternion(1)+state.quaternion(2)*state.quaternion(2)),-1.,1.));
        dynamics_ok&=v.norm()<=4.04&&state.angular_rate.norm()<=2.121&&tilt<=1.0605&&state.thrust>=1.98&&state.thrust<=12.12;
      }
      t0+=T;
    }
    result.feasible=roots_ok&&dynamics_ok&&result.penetration<=1e-6&&result.max_violation<=.005;
  }
  return comparison;
}
} // namespace demo_comparison
