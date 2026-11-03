// SPDX-License-Identifier: MIT
#include "corridor_benchmark_common.hpp"
#include "optimizer.hpp"
#include <cstdint>
using uz_experiment::Optimizer;
int main(int argc,char** argv){try{
  if(argc!=19)throw std::invalid_argument("use trajectory/run.py --binary uz_experiment");
  const auto seed=read(argv[1]),points=read(argv[2]);
  Bounds bounds{Vec3(std::stod(argv[4]),std::stod(argv[5]),std::stod(argv[6])),Vec3(std::stod(argv[7]),std::stod(argv[8]),std::stod(argv[9]))};
  const double radius=std::sqrt(3)*std::stod(argv[10])/2;
  const int repeats=std::stoi(argv[11]),quadrature=std::stoi(argv[14]),max_iterations=std::stoi(argv[16]);const bool check=std::stoi(argv[15]);
  const bool scan=!std::getenv("GCOPTER_UZ_TIME_SCAN")||std::string(std::getenv("GCOPTER_UZ_TIME_SCAN"))!="0";
  if(seed.size()<2||repeats<1||max_iterations<1||!std::isfinite(radius)||radius<=0)throw std::invalid_argument("invalid experiment inputs");
  std::vector<Sphere> spheres;for(auto p:points)spheres.push_back({p,radius});ObstacleIndex index(std::move(spheres));
  Options options;options.control_spacing=.4;options.min_radius=.03;options.budget_ms=10000;
  options.robot_radius=.2;options.safety_margin=.05;options.local_range=3;options.local_progress=7;
  double diagonal=0;for(auto b:expansionBoxes(seed,bounds,3,.252))diagonal=std::max(diagonal,(b.bounds.upper-b.bounds.lower).norm());
  options.max_radius=.5*diagonal+.001;options.max_offset=diagonal+.001;
  auto direct=generate(seed,index,bounds,options);if(!direct)throw std::runtime_error(direct.message);
  const std::filesystem::path dir=argv[3];std::filesystem::create_directories(dir);
  std::ofstream out(dir/"runs.jsonl");out<<std::setprecision(14);
  double length=0;for(size_t i=1;i<seed.size();++i)length+=(seed[i]-seed[i-1]).norm();
  for(int rep=-1;rep<repeats;++rep){
    auto clock=Clock::now();Optimizer optimizer(direct.corridor,seed.front(),seed.back(),quadrature);
    Eigen::VectorXd x=optimizer.initial(length/2),g(11);
    int time_seed_factor=1;
    if(scan){double best=INFINITY;auto candidate=x;
      for(int k=0;k<=8;++k){candidate(0)=std::log(length/2)+k*std::log(2.0);double value=optimizer.evaluate(candidate,g);
        if(value<best){best=value;x=candidate;time_seed_factor=1<<k;}}
    }
    const double prepare_ms=ms(clock);
    if(rep==-1){std::ofstream file(dir/"initial_variables.csv");file<<std::setprecision(17);for(int i=0;i<x.size();++i)file<<x(i)<<'\n';}
    double gradient_error=-1;
    if(check&&rep==-1){
      optimizer.evaluate(x,g);const auto analytic=g;Eigen::VectorXd numeric(11);
      for(int j=0;j<11;++j){const double base=x(j);x(j)=base+1e-6;double a=optimizer.evaluate(x,g);x(j)=base-1e-6;double b=optimizer.evaluate(x,g);x(j)=base;numeric(j)=(a-b)/2e-6;}
      gradient_error=(analytic-numeric).norm()/std::max(1.0,analytic.norm());
      std::cerr<<"u/z gradient "<<gradient_error<<'\n';if(!(gradient_error<2e-5))throw std::runtime_error("u/z gradient check failed");
    }
    optimizer.evaluations=optimizer.root_iterations=0;lbfgs::lbfgs_parameter_t params;
    params.mem_size=64;params.past=3;params.delta=1e-6;params.g_epsilon=0;params.max_iterations=max_iterations;
    double objective=0;clock=Clock::now();int status=lbfgs::lbfgs_optimize(x,objective,Optimizer::callback,nullptr,Optimizer::progressCallback,&optimizer,params);
    const double optimize_ms=ms(clock);const auto roots=optimizer.root_iterations;const auto evaluations=optimizer.evaluations;
    optimizer.evaluate(x,g);const double gradient_inf=g.lpNorm<Eigen::Infinity>();double final_gradient_error=-1;
    if(check&&rep==-1){const auto analytic=g;auto candidate=x;Eigen::VectorXd numeric(11);
      // Fourth-order Richardson difference avoids O(h^2) error at the sharply
      // curved smoothed penalties near their final active thresholds.
      for(int j=0;j<11;++j){
        const double h=1e-6*std::max(1.0,std::abs(x(j)));
        candidate(j)=x(j)+h;double a=optimizer.evaluate(candidate,g);
        candidate(j)=x(j)-h;double b=optimizer.evaluate(candidate,g);
        candidate(j)=x(j)+h/2;double c=optimizer.evaluate(candidate,g);
        candidate(j)=x(j)-h/2;double d=optimizer.evaluate(candidate,g);candidate(j)=x(j);
        numeric(j)=(4*(c-d)/h-(a-b)/(2*h))/3;
      }
      final_gradient_error=(analytic-numeric).norm()/std::max(1.0,analytic.norm());
      std::cerr<<"final u/z Richardson gradient "<<final_gradient_error<<'\n';
      if(!(final_gradient_error<2e-5))throw std::runtime_error("final u/z gradient check failed");
    }
    const double duration=std::exp(x(0));double path_length=0,jerk_energy=0,penetration=0,max_speed=0,max_acc=0,max_rate=0,max_tilt=0,min_thrust=INFINITY,max_thrust=0;
    double geometry_error=0,max_z=0,previous_u=-1;bool monotone=true;std::vector<int> near;
    flatness::FlatnessMap flat;flat.reset(.61,9.8,.70,.80,.01,.0001);
    // Bounded audit storage for very slow candidates; additionally sample every
    // geometry piece in u so localized high derivatives cannot hide in long flights.
    const int count=std::max(2000,int(std::min(20000.0,std::ceil(duration/.005))));Vec3 previous=seed.front();
    std::vector<double> audit;for(int k=0;k<=count;++k)audit.push_back(k/double(count));
    uz_experiment::Progress progress(x(1),x(2));uint64_t audit_roots=0;
    for(double u:direct.corridor.knots())if(u>0&&u<1){
      const double s=progress.inverse(u,audit_roots).first;
      audit.push_back(std::max(0.0,s-1e-9));audit.push_back(std::min(1.0,s+1e-9));
    }
    const auto& knots=direct.corridor.knots();
    for(size_t i=0;i+1<knots.size();++i)for(int k=1;k<64;++k)
      audit.push_back(progress.inverse(knots[i]+(knots[i+1]-knots[i])*k/64.0,audit_roots).first);
    std::sort(audit.begin(),audit.end());double previous_s=0,previous_jerk=optimizer.at(x,0).j.squaredNorm();
    std::ofstream csv;if(rep==0){csv.open(dir/"uz_trajectory.csv");csv<<std::setprecision(14)<<"t,u,z1,z2,x,y,z,vx,vy,vz,ax,ay,az\n";}
    for(double s:audit){
      auto state=optimizer.at(x,s);const Vec3 p=state.p;
      monotone=monotone&&state.u>=previous_u-1e-12;previous_u=state.u;max_z=std::max(max_z,state.z.norm());
      geometry_error=std::max(geometry_error,(direct.corridor.section(state.u).point(state.z)-p).norm());
      path_length+=(p-previous).norm();previous=p;
      const double squared_jerk=state.j.squaredNorm();jerk_energy+=0.5*(previous_jerk+squared_jerk)*(s-previous_s)*duration;
      previous_jerk=squared_jerk;previous_s=s;
      max_speed=std::max(max_speed,state.v.norm());max_acc=std::max(max_acc,state.a.norm());
      auto f=traj_opt_components::evaluateFlatnessPenaltyState(&flat,state.v,state.a,state.j);
      max_rate=std::max(max_rate,f.angular_rate.norm());max_tilt=std::max(max_tilt,std::acos(std::clamp(1-2*(f.quaternion(1)*f.quaternion(1)+f.quaternion(2)*f.quaternion(2)),-1.,1.)));
      min_thrust=std::min(min_thrust,f.thrust);max_thrust=std::max(max_thrust,f.thrust);
      index.query(p-Vec3::Constant(.25),p+Vec3::Constant(.25),near);
      for(int id:near)penetration=std::max(penetration,radius+.25-(p-index.obstacles()[id].center).norm());
      penetration=std::max({penetration,(bounds.lower.array()+.25-p.array()).maxCoeff(),(p.array()-bounds.upper.array()+.25).maxCoeff()});
      if(csv)csv<<s*duration<<','<<state.u<<','<<state.z.x()<<','<<state.z.y()<<','<<p.x()<<','<<p.y()<<','<<p.z()<<','<<state.v.x()<<','<<state.v.y()<<','<<state.v.z()<<','<<state.a.x()<<','<<state.a.y()<<','<<state.a.z()<<'\n';
    }
    // Independent higher-order quadrature catches under-resolved objective integration.
    Optimizer refined(direct.corridor,seed.front(),seed.back(),std::min(64,2*quadrature));
    const double refined_cost=refined.evaluate(x,g);const double quadrature_error=std::abs(refined_cost-objective)/std::max(1.0,std::abs(refined_cost));
    const double max_control=optimizer.maxControlNorm(x);
    const bool feasible=penetration<=1e-6&&geometry_error<=1e-7&&monotone&&max_control<=1+1e-12&&max_speed<=4.04&&max_rate<=2.121&&max_tilt<=1.0605&&min_thrust>=1.98&&max_thrust<=12.12;
    out<<"{\"method\":\"uz\",\"parameterization\":\"global_degree7_progress_degree5_transverse\",\"repeat\":"<<rep<<",\"dimension\":11,\"status\":"<<status
      <<",\"iterations\":"<<optimizer.iterations<<",\"evaluations\":"<<evaluations<<",\"progress_inverse_iterations\":"<<roots<<",\"quadrature_samples\":"<<optimizer.sampleCount()
      <<",\"time_seed_factor\":"<<time_seed_factor<<",\"audit_points\":"<<audit.size()<<",\"uniform_audit_dt_s\":"<<duration/count
      <<",\"prepare_ms\":"<<prepare_ms<<",\"optimize_ms\":"<<optimize_ms<<",\"corridor_ms\":"<<direct.metrics.total_ms<<",\"objective\":"<<objective
      <<",\"duration_s\":"<<duration<<",\"length_m\":"<<path_length<<",\"jerk_energy\":"<<jerk_energy<<",\"max_speed\":"<<max_speed<<",\"max_acceleration\":"<<max_acc
      <<",\"max_body_rate\":"<<max_rate<<",\"max_tilt\":"<<max_tilt<<",\"min_thrust\":"<<min_thrust<<",\"max_thrust\":"<<max_thrust
      <<",\"max_penetration_m\":"<<penetration<<",\"reconstruction_error_m\":"<<geometry_error<<",\"max_z_norm\":"<<max_z<<",\"max_z_control_norm\":"<<max_control
      <<",\"progress_monotone\":"<<(monotone?"true":"false")<<",\"refined_objective\":"<<refined_cost<<",\"quadrature_relative_error\":"<<quadrature_error
      <<",\"gradient_relative_error\":"<<gradient_error<<",\"final_gradient_relative_error\":"<<final_gradient_error<<",\"final_gradient_inf_norm\":"<<gradient_inf
      <<",\"sampled_feasible\":"<<(feasible?"true":"false")<<"}\n";out.flush();
    if(rep==0){std::ofstream file(dir/"final_variables.csv");file<<std::setprecision(17);for(int i=0;i<11;++i)file<<x(i)<<'\n';}
    std::cerr<<"u/z "<<rep<<" "<<optimize_ms<<" ms, feasible="<<feasible<<", integration error="<<quadrature_error<<'\n';
  }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
