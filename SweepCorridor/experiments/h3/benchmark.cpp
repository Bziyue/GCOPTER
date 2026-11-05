#include "../free_boundary/common.hpp"
#ifndef H3_BASELINE
#include "generator.hpp"
#endif
#include "../../../gcopter/tools/corridor_volume.hpp"
#include <sys/resource.h>
double anchorParameter(const c3::Geometry& g,const sc::Vec3& p,double u) {
  for(int iteration=0;iteration<30;++iteration){const auto s=g.section(u);const sc::Vec3 d=p-s.center;
    const sc::Vec3 n=s.shape.col(0).cross(s.shape.col(1));
    const sc::Vec3 nd=s.shape_first.col(0).cross(s.shape.col(1))+s.shape.col(0).cross(s.shape_first.col(1));
    const double f=n.dot(d),fu=nd.dot(d)-n.dot(s.first);
    if(std::abs(f)/n.norm()<1e-8){const Eigen::Vector2d z=(s.shape.transpose()*s.shape).inverse()*s.shape.transpose()*d;
      if(z.squaredNorm()<=1+1e-8)return u;break;}
    const double next=std::clamp(u-f/fu,0.,1.);if(next==u)break;u=next;
  }
  // Independent fallback: inspect all span brackets rather than relying on
  // the optimizer's cached local root. Curved sweeps can have other plane roots.
  auto plane=[&](double t){const auto s=g.section(t);return (p-s.center).dot(s.shape.col(0).cross(s.shape.col(1)));};
  for(size_t i=0;i<g.pieces();++i){double lo=g.breaks[i],hi=g.breaks[i+1],fl=plane(lo),fh=plane(hi);
    if(fl*fh>0)continue;
    for(int k=0;k<45;++k){const double mid=.5*(lo+hi),fm=plane(mid);if(fl*fm<=0){hi=mid;fh=fm;}else{lo=mid;fl=fm;}}
    const double t=.5*(lo+hi);const auto s=g.section(t);const Eigen::Vector2d z=(s.shape.transpose()*s.shape).inverse()*s.shape.transpose()*(p-s.center);
    if(z.squaredNorm()<=1+1e-8)return t;
  }return -1;
}
int main(int argc,char** argv){try{
  if(argc!=11)throw std::runtime_error("seed map output xmin ymin zmin xmax ymax zmax repeats");
  const auto seed=readPoints(argv[1]),cloud=readPoints(argv[2]);std::vector<sc::Sphere> spheres;
  for(const auto& p:cloud)spheres.push_back({p,std::sqrt(3)*.25/2});
  const auto index_start=c3::Clock::now();const sc::ObstacleIndex map(std::move(spheres));const double index_ms=c3::elapsedMs(index_start);
  const sc::Bounds bounds{{std::stod(argv[4]),std::stod(argv[5]),std::stod(argv[6])},{std::stod(argv[7]),std::stod(argv[8]),std::stod(argv[9])}};
  auto options=optionsFor(seed,bounds);c3::OptimizerSettings solver;solver.profile=std::getenv("SWEEP_PROFILE");
  const fs::path output=argv[3];fs::create_directories(output);std::ofstream log(output/"rows.jsonl");log<<std::setprecision(17);
  const double length=seedLength(seed);
  for(int repeat=-1;repeat<std::stoi(argv[10]);++repeat){
#ifndef H3_BASELINE
    const std::string mode=std::getenv("H3_MODE")?std::getenv("H3_MODE"):"h3_new";
    sc::experiments::h3::OptimizerSettings h3_solver(solver);
    // Named ablations retain their original settings as the demo default evolves.
    if(mode.rfind("h3_new",0)!=0){
      h3_solver.volume_jet_energy=1;h3_solver.joint_jet_steps=0;
      h3_solver.joint_final_only=false;h3_solver.joint_sweeps=1;}
    if(mode=="h3")h3_solver.jet_schedule=sc::experiments::h3::JetSchedule::Frozen;
    if(mode=="h3_mid")h3_solver.jet_schedule=sc::experiments::h3::JetSchedule::BeforeTightening;
    if(mode=="h3_polish")h3_solver.polish=true;
    if(mode=="h3_more")h3_solver.iterations=12;
    if(mode=="h3_three"){h3_solver.inflation_iterations=3;h3_solver.iterations=4;h3_solver.corrector_iterations=2;}
    if(mode=="h3_volume"||mode=="h3_vmid"||mode=="h3_vreg")h3_solver.volume_jet_steps=2;
    if(mode=="h3_vmid")h3_solver.jet_schedule=sc::experiments::h3::JetSchedule::BeforeTightening;
    if(mode=="h3_vreg")h3_solver.volume_jet_energy=10;
    if(mode=="h3_joint"||mode=="h3_joint4"||mode=="h3_jointreg")h3_solver.joint_jet_steps=2;
    if(mode=="h3_joint4")h3_solver.joint_jet_steps=4;
    if(mode=="h3_jointreg")h3_solver.volume_jet_energy=10;
    if(mode=="h3_final"||mode=="h3_final4"||mode=="h3_relaxed_joint"){
      h3_solver.joint_jet_steps=2;h3_solver.joint_final_only=true;}
    if(mode=="h3_final4")h3_solver.joint_jet_steps=4;
    if(mode=="h3_relaxed"||mode=="h3_relaxed_joint")h3_solver.bending_weight=.25;
    if(mode=="h3_relaxed1")h3_solver.bending_weight=.1;
    if(mode=="h3_two"||mode=="h3_two_half"||mode=="h3_two_soft"||mode=="h3_four"){
      h3_solver.joint_final_only=true;h3_solver.joint_jet_steps=2;h3_solver.joint_sweeps=2;}
    if(mode=="h3_four"){h3_solver.joint_jet_steps=1;h3_solver.joint_sweeps=4;}
    if(mode=="h3_two_half"||mode=="h3_two_soft")h3_solver.bending_weight=.5;
    if(mode=="h3_two_soft")h3_solver.volume_jet_energy=3;
    if(mode=="h3_balanced"||mode=="h3_balanced6"||mode=="h3_balanced2"){
      h3_solver.joint_final_only=true;h3_solver.joint_jet_steps=2;h3_solver.bending_weight=.5;h3_solver.volume_jet_energy=3;}
    if(mode=="h3_balanced6")h3_solver.iterations=6;
    if(mode=="h3_balanced2")h3_solver.corrector_iterations=2;
    if(mode=="h3_half")h3_solver.bending_weight=.5;
    if(mode=="h3_halving"){h3_solver.joint_final_only=true;h3_solver.joint_jet_steps=2;h3_solver.joint_fast_line=false;}
    if(mode=="h3_rebuild"){h3_solver.joint_final_only=true;h3_solver.joint_jet_steps=2;h3_solver.bending_weight=.5;h3_solver.volume_jet_energy=3;h3_solver.corrector_iterations=2;h3_solver.reuse_joint_supports=false;}
    if(mode=="h3_two_reg3"||mode=="h3_final_reg3"){
      h3_solver.joint_final_only=true;h3_solver.joint_jet_steps=2;h3_solver.volume_jet_energy=3;
      h3_solver.joint_sweeps=mode=="h3_two_reg3"?2:1;}
    if(mode=="h3_new_rebuild")h3_solver.reuse_joint_supports=false;
    if(mode=="h3_new_halving")h3_solver.joint_fast_line=false;
#endif
    const auto start=c3::Clock::now();auto r=
#ifdef H3_BASELINE
      c3::generateCubic(seed,map,bounds,options,solver);
#else
      sc::experiments::h3::generate(seed,map,bounds,options,h3_solver);
#endif
const double total=c3::elapsedMs(start);
    log<<"{\"repeat\":"<<repeat<<",\"status\":\""<<sc::statusName(r.status)<<"\",\"message\":\""<<r.message<<"\",\"wall_ms\":"<<total
       <<",\"index_ms\":"<<index_ms<<",\"steps\":"<<r.optimizer.steps<<",\"termination\":\""<<r.optimizer.termination<<"\",\"controls\":"<<r.optimizer.controls
       <<",\"constraints\":"<<r.optimizer.constraints<<",\"active_constraints\":"<<r.optimizer.active_constraints<<",\"refinements\":"<<r.refinements
       <<",\"backtracks\":"<<r.optimizer.backtracks<<",\"jacobian_backtracks\":"<<r.optimizer.regularity_backtracks
       <<",\"decrement\":"<<r.optimizer.final_decrement<<",\"minimum_step\":"<<r.optimizer.minimum_step<<",\"mean_step\":"<<(r.optimizer.steps?r.optimizer.step_sum/r.optimizer.steps:0)
       <<",\"variables\":"<<r.optimizer.variables<<",\"hessian_nonzero\":"<<r.optimizer.hessian_nonzero<<",\"initialization_ms\":"<<r.initialization_ms
       <<",\"cover_ms\":"<<r.cover_metrics.total_ms<<",\"setup_ms\":"<<r.optimizer.setup_ms<<",\"assembly_ms\":"<<r.optimizer.assembly_ms
       <<",\"solve_ms\":"<<r.optimizer.solve_ms<<",\"line_ms\":"<<r.optimizer.line_search_ms<<",\"certificate_ms\":"<<r.optimizer.certificate_ms;
#ifndef H3_BASELINE
    log<<",\"jet_ms\":"<<r.jets.milliseconds<<",\"jet_accepted\":"<<r.jets.accepted<<",\"jet_rejected\":"<<r.jets.rejected<<",\"jet_fraction\":"<<(r.jets.accepted?r.jets.fraction_sum/r.jets.accepted:0);
    log<<",\"joint_setup_ms\":"<<r.jets.joint_setup_ms<<",\"joint_assembly_ms\":"<<r.jets.joint_assembly_ms<<",\"joint_solve_ms\":"<<r.jets.joint_solve_ms
       <<",\"joint_refresh_ms\":"<<r.jets.joint_refresh_ms<<",\"joint_line_ms\":"<<r.jets.joint_line_ms<<",\"joint_steps\":"<<r.jets.joint_steps
       <<",\"joint_backtracks\":"<<r.jets.joint_backtracks<<",\"joint_groups\":"<<r.jets.joint_groups<<",\"joint_supports\":"<<r.jets.joint_supports;
#endif
    if(r.status==sc::Status::Success){const auto cert=c3::certify(r.corridor,r.cover);
      const double begin=anchorParameter(r.corridor,seed.front(),0),end=anchorParameter(r.corridor,seed.back(),1);
      if(!cert.safe()||begin<0||end<begin){writeGeometry(r.corridor,output/"failed_geometry.csv");
        log<<",\"verification_pass\":false}\n";log.flush();
        std::cerr<<"support="<<cert.support<<" jacobian="<<cert.jacobian<<" anchors="<<begin<<','<<end<<'\n';
        throw std::runtime_error("independent certificate / anchor failure");}
      double c1_error=0;for(int i=1;i<int(r.corridor.pieces());++i){const auto a=c3::evaluate<3>(r.corridor.curve(i-1),1,r.corridor.breaks[i]-r.corridor.breaks[i-1]),b=c3::evaluate<3>(r.corridor.curve(i),0,r.corridor.breaks[i+1]-r.corridor.breaks[i]);c1_error=std::max({c1_error,(a.center-b.center).norm(),(a.shape-b.shape).norm(),(a.first-b.first).norm()/length,(a.shape_first-b.shape_first).norm()/length});}
      if(c1_error>1e-7)throw std::runtime_error("C1 continuity verification failed");
      log<<",\"c1_error\":"<<c1_error;
      const auto core=restrictGeometry(r.corridor,begin,end);const auto q=quality(core,length),full=quality(r.corridor,length/(end-begin));
      log<<",\"support_pass\":true,\"jacobian_pass\":true,\"anchors_pass\":true,\"anchor_begin\":"<<begin<<",\"anchor_end\":"<<end
         <<",\"swept_volume\":"<<full.swept_volume<<",\"boundary_bending\":"<<q.boundary_bending<<",\"c2_error\":"<<full.c2_error
         <<",\"start_distance\":"<<(r.corridor.section(0).center-seed.front()).norm()<<",\"end_distance\":"<<(r.corridor.section(1).center-seed.back()).norm();
      if(repeat==0){writeGeometry(r.corridor,output/"corridor.csv");writeGeometry(core,output/"core.csv");
        std::ofstream planes(output/"support_planes.csv");planes<<std::setprecision(17)<<"span,begin,end,nx,ny,nz,bound\n";
        for(const auto& p:r.cover)planes<<p.span<<','<<p.begin<<','<<p.end<<','<<p.normal[0]<<','<<p.normal[1]<<','<<p.normal[2]<<','<<p.bound<<'\n';}
    }log<<"}\n";log.flush();
  }rusage usage{};getrusage(RUSAGE_SELF,&usage);log<<"{\"max_rss_kib\":"<<usage.ru_maxrss<<"}\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
