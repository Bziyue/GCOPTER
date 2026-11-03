// SPDX-License-Identifier: MIT
// Controlled common-backend experiment, not the native GCOPTER pipeline.
#include "corridor_benchmark_common.hpp"
#include "SplineTrajectory/SplineOptimizer.hpp"
#include "SplineTrajectory/SplineTrajectory.hpp"
#include "TrajectoryOptAdapters/GcopterPenaltyIntegralCostAdapter.hpp"
#include "TrajectoryOptAdapters/BezierPenaltyCost.hpp"
#include "TrajectoryOptComponents/LinearTimeCost.hpp"
#include <cstdint>
#include <cstring>

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
struct TubeValue {
  double radial=0, plane=0, u=0;
  Vec3 grad_radial=Vec3::Zero(), grad_plane=Vec3::Zero();
  bool valid=true;
};
// Local inversion of n(u).(p-c(u))=0. Progress is only a root-selection hint,
// not a constraint on the physical trajectory. Multiple roots may exist.
TubeValue tubeValue(const Corridor& corridor, const Vec3& p, double hint) {
  double u=hint,f=0,fu=0;Section section;Vec3 normal,normal_u;
  auto plane=[&](double x) {
    section=corridor.section(x);
    Vec3 cross=section.shape.col(0).cross(section.shape.col(1));
    Vec3 dcross=section.shape_first.col(0).cross(section.shape.col(1))+
               section.shape.col(0).cross(section.shape_first.col(1));
    normal=cross.normalized();normal_u=(dcross-normal*normal.dot(dcross))/cross.norm();
    f=normal.dot(p-section.center);
    fu=normal_u.dot(p-section.center)-normal.dot(section.first);
    return f;
  };
  for(int i=0;i<16;++i) {
    plane(u);if(std::abs(f)<1e-8)break;
    if(std::abs(fu)<1e-10)break;
    double next=std::clamp(u-std::clamp(f/fu,-.1,.1),0.,1.);
    if(next==u)break;u=next;
  }
  plane(u);
  if(std::abs(f)>1e-7 && u>0 && u<1) {
    // Rare fallback: choose the nearest sign-changing knot interval, then bisect.
    double best=1e10,lo=0,hi=0,fl=0;
    const auto& knots=corridor.knots();double previous=plane(knots.front());
    for(size_t i=1;i<knots.size();++i) {
      double current=plane(knots[i]);double dist=std::abs(.5*(knots[i-1]+knots[i])-hint);
      if(previous*current<=0 && dist<best){best=dist;lo=knots[i-1];hi=knots[i];fl=previous;}
      previous=current;
    }
    if(best<1e10) {
      for(int i=0;i<40;++i){u=.5*(lo+hi);double fm=plane(u);if(std::abs(fm)<1e-9)break;if(fl*fm<=0)hi=u;else{lo=u;fl=fm;}}
    }
    plane(u);
  }
  TubeValue out;out.u=u;out.plane=f;out.grad_plane=normal;
  const bool root=std::abs(f)<1e-7&&std::abs(fu)>1e-10;
  out.valid=root||u==0||u==1;
  Vec3 d=p-section.center;
  const Mat32& M=section.shape;const Mat32& Md=section.shape_first;
  Eigen::Matrix2d gram=M.transpose()*M;
  Eigen::Matrix<double,2,3> inv=gram.ldlt().solve(M.transpose());
  Eigen::Vector2d z=inv*d;
  double q=z.squaredNorm();
  if(q>1) {
    // Radius in the section plane. This also handles endpoint cap extensions.
    Vec3 tangent=d-normal*normal.dot(d);double r=tangent.norm(),sq=std::sqrt(q);
    out.radial=r*(1-1/sq);
    out.grad_radial=(1-1/sq)*tangent/r+r/(q*sq)*(inv.transpose()*z);
    if(root) {
      Eigen::Vector2d zu=gram.ldlt().solve(Md.transpose()*d-M.transpose()*section.first-
          (Md.transpose()*M+M.transpose()*Md)*z);
      double ru=-tangent.dot(section.first)/r;
      double derivative=ru*(1-1/sq)+r/(q*sq)*z.dot(zu);
      out.grad_radial-=normal*(derivative/fu);
    }
  }
  if(root){out.plane=0;out.grad_plane.setZero();}
  return out;
}
struct Cost {
  traj_opt_adapters::GcopterPenaltyIntegralCostAdapter dynamics;
  const Corridor* corridor=nullptr;
  const Polys* polys=nullptr;
  int steps=16,pieces=1;
  double weight=1e5;
  bool ellipse=false;
  bool bezier=false;
  bool keep_tube_integral=false;
  double positionWeight() const {return weight*(bezier&&ellipse&&keep_tube_integral?0.5:1.0);}
  static constexpr int kDerivativeOrder = 3;
  double operator()(const SplineTrajectory::IntegralPointInfo& point,
      const SplineTrajectory::SampleState<3>& state,
      SplineTrajectory::SampleGradient<3>& partial) const {
    double cost=dynamics(point,state,partial);
    if(bezier&&!(ellipse&&keep_tube_integral)) return cost;
    const double position_weight=positionWeight();
    const int seg=point.segment_index,step=point.step_index;
    const auto& p=state.p;auto& gp=partial.p;
    Vec3 gradient;double violation;
    if(ellipse) {
      auto tube=tubeValue(*corridor,p,(seg+double(step)/steps)/pieces);
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
      const auto tube=tubeValue(*cost->corridor,p,(segment+alpha)/cost->pieces);
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

  traj_opt_components::LinearTimeCost time{20};
  Cost cost;
  traj_opt_adapters::BezierPenaltyCost<Spline,ControlPosition> bezier_cost;
  struct Coefficient {
    Run* run;
    double operator()(const SplineTrajectory::ParameterView<Spline>& p,SplineTrajectory::CoefficientGradient<Spline>& g) {
      return run->cost.bezier?run->bezier_cost(p,g):0;
    }
  };
  struct Objective { traj_opt_components::LinearTimeCost& duration; Cost& integral; Coefficient coefficient; };
  int evaluations=0,iterations=0;
  static double evaluate(void* ptr,const Eigen::VectorXd& x,Eigen::VectorXd& g) {
    auto& r=*static_cast<Run*>(ptr); ++r.evaluations;
    Objective objective{r.time,r.cost,{&r}};
    auto value=r.optimizer.evaluate(x,g,objective);
    return value.cost;
  }
  static int progress(void* ptr,const Eigen::VectorXd&,const Eigen::VectorXd&,
      double,double,int k,int) {static_cast<Run*>(ptr)->iterations=k;return 0;}
};
std::string hash(const Eigen::VectorXd& x) {
  uint64_t h=14695981039346656037ull;
  auto bytes=reinterpret_cast<const unsigned char*>(x.data());
  for(size_t i=0;i<size_t(x.size())*sizeof(double);++i) h=(h^bytes[i])*1099511628211ull;
  std::ostringstream out;out<<std::hex<<h;return out.str();
}
int main(int argc,char**argv) {try {
  if(argc!=19) throw std::runtime_error("seed map output xmin ymin zmin xmax ymax zmax resolution repeats spacing weight steps gradient_check max_iterations constraint depth");
  auto seed=read(argv[1]),points=read(argv[2]);
  Bounds bounds{Vec3(std::stod(argv[4]),std::stod(argv[5]),std::stod(argv[6])),
                Vec3(std::stod(argv[7]),std::stod(argv[8]),std::stod(argv[9]))};
  double radius=std::sqrt(3)*std::stod(argv[10])/2, spacing=std::stod(argv[12]),weight=std::stod(argv[13]);
  int repeats=std::stoi(argv[11]),steps=std::stoi(argv[14]); bool check=std::stoi(argv[15]);int max_iterations=std::stoi(argv[16]);
  std::string constraint=argv[17];int depth=std::stoi(argv[18]);
  if((constraint!="bezier"&&constraint!="sampled"&&constraint!="bezier-controls")||depth<0||depth>5)throw std::runtime_error("invalid constraint mode or depth (0..5)");
  const bool bezier=constraint!="sampled";
  if(seed.size()<2||points.empty()||repeats<1||!std::isfinite(spacing)||spacing<=0||!std::isfinite(weight)||weight<=0||!std::isfinite(radius)||radius<=0||max_iterations<1||steps<2||!bounds.lower.allFinite()||!bounds.upper.allFinite()||!(bounds.upper.array()>bounds.lower.array()).all())
    throw std::runtime_error("invalid benchmark inputs");
  for(size_t i=1;i<seed.size();++i)if((seed[i]-seed[i-1]).norm()<1e-10)throw std::runtime_error("duplicate seed point");
  std::filesystem::path dir=argv[3];std::filesystem::create_directories(dir);
  std::vector<Sphere> spheres;for(auto p:points)spheres.push_back({p,radius});
  auto clock=Clock::now();ObstacleIndex index(std::move(spheres));double index_ms=ms(clock);
  Options options;options.control_spacing=.4;options.min_radius=.03;options.budget_ms=10000;
  options.robot_radius=.2;options.safety_margin=.05;options.local_range=3;options.local_progress=7;
  double diagonal=0;for(auto b:expansionBoxes(seed,bounds,3,.252))diagonal=std::max(diagonal,(b.bounds.upper-b.bounds.lower).norm());
  options.max_radius=.5*diagonal+.001;options.max_offset=diagonal+.001;
  Result direct=generate(seed,index,bounds,options);if(!direct)throw std::runtime_error(direct.message);
  auto firi=generateFiri(seed,points,bounds,radius,.25,true,3);
  int disconnected=0,empty_polys=0;Vec3 interior;
  for(const auto& h:firi.polys)if(!geo_utils::findInterior(h,interior))++empty_polys;
  for(size_t i=1;i<firi.polys.size();++i)if(!geo_utils::overlap(firi.polys[i-1],firi.polys[i]))++disconnected;
  double seed_outside=0;Vec3 grad_seed;
  for(size_t i=1;i<seed.size();++i)for(int k=0;k<=100;++k)
    seed_outside=std::max(seed_outside,polyDistance(firi.polys,seed[i-1]+(seed[i]-seed[i-1])*(k/100.),grad_seed));
  std::vector<double> arc{0};for(size_t i=1;i<seed.size();++i)arc.push_back(arc.back()+(seed[i]-seed[i-1]).norm());
  const int pieces=std::max(2,int(std::ceil(arc.back()/spacing)));
  Opt::Problem problem;problem.waypoints.resize(pieces+1,3);
  for(int i=0;i<=pieces;++i) {
    double s=arc.back()*i/pieces;auto it=std::upper_bound(arc.begin(),arc.end(),s);
    int k=std::clamp(int(it-arc.begin())-1,0,int(seed.size())-2);
    problem.waypoints.row(i)=(seed[k]+(seed[k+1]-seed[k])*((s-arc[k])/(arc[k+1]-arc[k]))).transpose();
  }
  problem.durations.assign(pieces,arc.back()/pieces/2.0);
  problem.boundary.start_velocity=problem.boundary.end_velocity=Vec3::Zero();
  problem.boundary.start_acceleration=problem.boundary.end_acceleration=Vec3::Zero();
  Polys empty(1);empty[0].resize(0,4);Eigen::VectorXi ids=Eigen::VectorXi::Zero(pieces);
  Eigen::VectorXd limits(5),weights(5);limits<<4,2.1,1.05,2,12;weights<<0,1e4,1e4,1e4,1e5;
  std::ofstream out(dir/"runs.jsonl");out<<std::setprecision(12);
  std::string expected_hash;
  for(int rep=-1;rep<repeats;++rep) for(int order=0;order<2;++order) {
    bool ellipse=(rep+order+2)%2;std::string name=ellipse?"hermite":"firi";
    clock=Clock::now();Run run;SplineTrajectory::OptimizerOptions config;config.energy_weight=1;config.integration_steps=steps;
    if(!run.optimizer.prepare(problem,config))throw std::runtime_error("prepare failed");
    flatness::FlatnessMap flat;flat.reset(.61,9.8,.70,.80,.01,.0001);
    run.cost.dynamics.reset(&empty,&ids,.01,limits,weights,&flat);
    run.cost.corridor=&direct.corridor;run.cost.pieces=pieces;run.cost.polys=&firi.polys;run.cost.steps=steps;run.cost.ellipse=ellipse;run.cost.weight=weight;
    Eigen::VectorXd x=run.optimizer.initialGuess(),g(x.size());
    run.cost.bezier=bezier;
    run.cost.keep_tube_integral=constraint=="bezier";
    if(bezier) {
      if(!run.optimizer.updateAccepted(x))throw std::runtime_error("initial spline failed");
      run.cost.dynamics.penalty_weights(1)=0;
      run.bezier_cost.position.cost=&run.cost;
      run.bezier_cost.position.assign(run.optimizer.polynomial(),depth);
      run.bezier_cost.configure(pieces,depth,limits(0),weights(1),.01);
    }
    std::string init_hash=hash(x);if(expected_hash.empty())expected_hash=init_hash;
    if(expected_hash!=init_hash)throw std::runtime_error("initial vector mismatch");
    double prepare_ms=ms(clock),gradient_error=-1;
    if(rep==-1 && order==0) {
      std::ofstream initial(dir/"initial_waypoints.csv");initial<<std::setprecision(17)<<"x,y,z,duration_to_next\n";
      for(int i=0;i<=pieces;++i)initial<<problem.waypoints(i,0)<<','<<problem.waypoints(i,1)<<','<<problem.waypoints(i,2)<<','<<(i<pieces?problem.durations[i]:0)<<'\n';
    }
    if(check&&rep==-1) {
      Run::Objective objective{run.time,run.cost,{&run}};
      auto result=SplineTrajectory::checkGradients(run.optimizer,x,objective);
      gradient_error=result.relative_error;
      std::cerr<<name<<" gradient relative error "<<gradient_error<<'\n';
      if(!(gradient_error<1e-4))throw std::runtime_error("gradient check failed");
    }
    lbfgs::lbfgs_parameter_t params;params.mem_size=64;params.past=3;params.delta=1e-6;
    params.g_epsilon=0;params.min_step=1e-32;params.max_iterations=max_iterations;
    double objective=0;clock=Clock::now();
    int status=lbfgs::lbfgs_optimize(x,objective,Run::evaluate,nullptr,Run::progress,&run,params);
    double optimize_ms=ms(clock);Run::evaluate(&run,x,g);
    const auto& trajectory=run.optimizer.polynomial();
    double optimization_sample_violation=0;
    for(int seg=0;seg<pieces;++seg) {
      double start=0;for(int k=0;k<seg;++k)start+=run.optimizer.parameters().durations[k];
      for(int k=0;k<=steps;++k) {
        Vec3 point=trajectory.evaluate(start+run.optimizer.parameters().durations[seg]*k/steps),grad;
        double violation=ellipse?tubeValue(direct.corridor,point,(seg+double(k)/steps)/pieces).radial:polyDistance(firi.polys,point,grad);
        optimization_sample_violation=std::max(optimization_sample_violation,violation);
      }
    }
    double duration=trajectory.duration(),length=0,max_speed=0,max_acc=0,max_rate=0,max_tilt=0,min_thrust=1e10,max_thrust=0,max_plane=0,max_corridor=0,penetration=0;
    int inverse_failures=0;
    std::ofstream csv;if(rep==0){csv.open(dir/(name+"_trajectory.csv"));csv<<"t,x,y,z,vx,vy,vz,ax,ay,az\n";csv<<std::setprecision(12);}
    Vec3 previous=trajectory.evaluate(0);std::vector<int> near;double t0=0;
    // Audit >=4x optimization samples and <=5 ms time step; not a continuous certificate.
    for(int seg=0;seg<pieces;++seg) {
      double T=run.optimizer.parameters().durations[seg];int n=std::max(4*steps,int(std::ceil(T/.005)));
      for(int k=0;k<=n;++k) {
        double t=t0+T*k/n,u=(seg+double(k)/n)/pieces;
        Vec3 p=trajectory.evaluate(t),v=trajectory.evaluate(t,1),a=trajectory.evaluate(t,2),j=trajectory.evaluate(t,3);
        length+=(p-previous).norm();previous=p;max_speed=std::max(max_speed,v.norm());max_acc=std::max(max_acc,a.norm());
        auto state=traj_opt_components::evaluateFlatnessPenaltyState(&flat,v,a,j);
        max_rate=std::max(max_rate,state.angular_rate.norm());
        max_tilt=std::max(max_tilt,std::acos(std::clamp(1-2*(state.quaternion(1)*state.quaternion(1)+state.quaternion(2)*state.quaternion(2)),-1.,1.)));
        min_thrust=std::min(min_thrust,state.thrust);max_thrust=std::max(max_thrust,state.thrust);
        Vec3 grad;
        if(ellipse) {
          auto tube=tubeValue(direct.corridor,p,u);
          inverse_failures+=!tube.valid;
          max_plane=std::max(max_plane,std::abs(tube.plane));
          max_corridor=std::max(max_corridor,tube.radial);
        } else max_corridor=std::max(max_corridor,polyDistance(firi.polys,p,grad));
        index.query(p-Vec3::Constant(.25),p+Vec3::Constant(.25),near);
        for(int id:near)penetration=std::max(penetration,radius+.25-(p-index.obstacles()[id].center).norm());
        penetration=std::max({penetration,(bounds.lower.array()+.25-p.array()).maxCoeff(),(p.array()-bounds.upper.array()+.25).maxCoeff()});
        if(csv)csv<<t<<','<<p.x()<<','<<p.y()<<','<<p.z()<<','<<v.x()<<','<<v.y()<<','<<v.z()<<','<<a.x()<<','<<a.y()<<','<<a.z()<<'\n';
      }t0+=T;
    }
    bool feasible=inverse_failures==0&&penetration<=1e-6&&max_plane<=.005&&max_corridor<=.005&&max_speed<=4.04&&max_rate<=2.121&&max_tilt<=1.0605&&min_thrust>=1.98&&max_thrust<=12.12;
    out<<"{\"method\":\""<<name<<"\",\"repeat\":"<<rep<<",\"pieces\":"<<pieces<<",\"initial_hash\":\""<<init_hash<<"\",\"status\":"<<status
       <<",\"constraint\":\""<<constraint<<"\",\"bezier_depth\":"<<depth
       <<",\"tube_integral_retained\":"<<(bezier&&ellipse&&run.cost.keep_tube_integral?"true":"false")
       <<",\"control_position_violation_m\":"<<run.bezier_cost.max_position_violation
       <<",\"control_speed_violation_mps\":"<<run.bezier_cost.max_speed_violation
       <<",\"iterations\":"<<run.iterations<<",\"evaluations\":"<<run.evaluations<<",\"index_ms\":"<<index_ms<<",\"corridor_ms\":"<<(ellipse?direct.metrics.total_ms:firi.total_ms)
       <<",\"prepare_ms\":"<<prepare_ms<<",\"optimize_ms\":"<<optimize_ms<<",\"objective\":"<<objective
       <<",\"duration_s\":"<<duration<<",\"length_m\":"<<length<<",\"jerk_energy\":"<<run.optimizer.energy()<<",\"max_speed\":"<<max_speed<<",\"max_acceleration\":"<<max_acc
       <<",\"max_body_rate\":"<<max_rate<<",\"max_tilt\":"<<max_tilt<<",\"min_thrust\":"<<min_thrust<<",\"max_thrust\":"<<max_thrust
       <<",\"optimization_sample_violation_m\":"<<optimization_sample_violation<<",\"max_plane_error_m\":"<<max_plane<<",\"max_corridor_error_m\":"<<max_corridor<<",\"max_penetration_m\":"<<penetration
       <<",\"firi_disconnected_neighbors\":"<<disconnected<<",\"firi_empty_polys\":"<<empty_polys<<",\"firi_seed_outside_m\":"<<seed_outside
       <<",\"inverse_failures\":"<<inverse_failures<<",\"gradient_relative_error\":"<<gradient_error<<",\"sampled_feasible\":"<<(feasible?"true":"false")<<"}\n";out.flush();
    std::cerr<<name<<" rep "<<rep<<" "<<optimize_ms<<"ms feasible="<<feasible<<" plane="<<max_plane<<" corridor="<<max_corridor<<" collision="<<penetration<<'\n';
  }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
