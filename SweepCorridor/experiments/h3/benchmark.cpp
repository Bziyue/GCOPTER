#include "../free_boundary/common.hpp"
#ifndef H3_BASELINE
#include "generator.hpp"
#include "audit_cover.hpp"
#include "benchmark_presets.hpp"
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
  if(argc!=11 && argc!=12)throw std::runtime_error("seed map output xmin ymin zmin xmax ymax zmax repeats [voxel_resolution]");
  const double resolution=argc==12?std::stod(argv[11]):.25;
  if(!(resolution>0) || !std::isfinite(resolution))throw std::invalid_argument("invalid resolution");
  const auto seed=readPoints(argv[1]),cloud=readPoints(argv[2]);
  const std::string mode=std::getenv("H3_MODE")?std::getenv("H3_MODE"):"h3_new";
  c3::OptimizerSettings solver;solver.profile=std::getenv("SWEEP_PROFILE");
#ifndef H3_BASELINE
  const auto preset=sc::experiments::h3::benchmarkPreset(mode,solver);
  const auto& h3_solver=preset.solver;
#endif
  const auto index_start=c3::Clock::now();
  const sc::ObstacleIndex map=[&] {
#ifndef H3_BASELINE
    using sc::experiments::h3::MapInput;
    if(preset.input==MapInput::Points)return sc::ObstacleIndex::fromPoints(cloud);
    if(preset.input==MapInput::Voxels)
      return sc::ObstacleIndex::fromOccupiedVoxels(cloud,resolution);
#endif
    std::vector<sc::Sphere> spheres;
    spheres.reserve(cloud.size());
    for(const auto& p:cloud)spheres.push_back({p,std::sqrt(3)*resolution/2});
    return sc::ObstacleIndex(std::move(spheres));
  }();
  const double index_ms=c3::elapsedMs(index_start);
  const sc::Bounds bounds{{std::stod(argv[4]),std::stod(argv[5]),std::stod(argv[6])},{std::stod(argv[7]),std::stod(argv[8]),std::stod(argv[9])}};
  auto options=optionsFor(seed,bounds);
  const fs::path output=argv[3];fs::create_directories(output);std::ofstream log(output/"rows.jsonl");log<<std::setprecision(17);
  const double length=seedLength(seed);
  for(int repeat=-1;repeat<std::stoi(argv[10]);++repeat){
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
    log<<",\"intermediate_radius_used\":"<<(r.intermediate_radius_used?"true":"false")
       <<",\"core_smoothed\":"<<(r.jets.core_smoothed?"true":"false")<<",\"area_selected_spans\":"<<r.jets.area_selected_spans;
    log<<",\"coarse_start_used\":"<<(r.coarse_start_used?"true":"false");
    log<<",\"shared_obstacle_spans\":"<<r.cover_metrics.shared_obstacle_spans;
    log<<",\"jet_ms\":"<<r.jets.milliseconds<<",\"jet_accepted\":"<<r.jets.accepted<<",\"jet_rejected\":"<<r.jets.rejected<<",\"jet_fraction\":"<<(r.jets.accepted?r.jets.fraction_sum/r.jets.accepted:0);
    log<<",\"joint_setup_ms\":"<<r.jets.joint_setup_ms<<",\"joint_assembly_ms\":"<<r.jets.joint_assembly_ms<<",\"joint_solve_ms\":"<<r.jets.joint_solve_ms
       <<",\"joint_refresh_ms\":"<<r.jets.joint_refresh_ms<<",\"joint_line_ms\":"<<r.jets.joint_line_ms<<",\"joint_steps\":"<<r.jets.joint_steps
       <<",\"joint_backtracks\":"<<r.jets.joint_backtracks<<",\"joint_groups\":"<<r.jets.joint_groups<<",\"joint_supports\":"<<r.jets.joint_supports;
    log<<",\"end_steps\":"<<r.jets.end_steps<<",\"end_trials\":"<<r.jets.end_trials<<",\"end_volume_gain\":"<<r.jets.end_volume_gain<<",\"end_ms\":"<<r.jets.end_ms;
    log<<",\"refresh_ms\":"<<r.jets.refresh_ms<<",\"refresh_spans\":"<<r.jets.refresh_spans<<",\"refresh_failures\":"<<r.jets.refresh_failures<<",\"refresh_cache_hits\":"<<r.jets.refresh_cache_hits;
    log<<",\"local_setup_ms\":"<<r.jets.local_setup_ms<<",\"local_joint_ms\":"<<r.jets.local_joint_ms;
    log<<",\"joint_budget_skips\":"<<r.jets.joint_budget_skips;
#endif
    if(r.status==sc::Status::Success){
#ifndef H3_BASELINE
      const auto cert=c3::certify(r.corridor,r.cover,options.certificate_depth,h3_solver.regularity(length));
      log<<",\"jacobian_relative_lower_bound\":"<<cert.cone_relative_margin;
      if(repeat==0) {
        const double pad=options.robot_radius+options.safety_margin+options.certificate_margin;
        if(!sc::experiments::h3::auditObstacleCover(r.corridor,r.cover,map,pad))
          throw std::runtime_error("incomplete obstacle separation in exported corridor");
        log<<",\"obstacle_audit\":true";
      }
#else
      const auto cert=c3::certify(r.corridor,r.cover);
#endif
      const double begin=anchorParameter(r.corridor,seed.front(),0),end=anchorParameter(r.corridor,seed.back(),1);
      if(!cert.safe()||begin<0||end<=begin){writeGeometry(r.corridor,output/"failed_geometry.csv");
        log<<",\"verification_pass\":false}\n";log.flush();
        std::cerr<<"support="<<cert.support<<" jacobian="<<cert.jacobian<<" anchors="<<begin<<','<<end<<'\n';
        throw std::runtime_error("independent certificate / anchor failure");}
      double c1_error=0;for(int i=1;i<int(r.corridor.pieces());++i){const auto a=c3::evaluate<3>(r.corridor.curve(i-1),1,r.corridor.breaks[i]-r.corridor.breaks[i-1]),b=c3::evaluate<3>(r.corridor.curve(i),0,r.corridor.breaks[i+1]-r.corridor.breaks[i]);c1_error=std::max({c1_error,(a.center-b.center).norm(),(a.shape-b.shape).norm(),(a.first-b.first).norm()/length,(a.shape_first-b.shape_first).norm()/length});}
      if(c1_error>1e-7)throw std::runtime_error("C1 continuity verification failed");
      log<<",\"c1_error\":"<<c1_error;
      const auto core=restrictGeometry(r.corridor,begin,end);const auto q=quality(core,length),full=quality(r.corridor,length/(end-begin));
      log<<",\"support_pass\":true,\"jacobian_pass\":true,\"anchors_pass\":true,\"anchor_begin\":"<<begin<<",\"anchor_end\":"<<end
         <<",\"swept_volume\":"<<full.swept_volume<<",\"core_swept_volume\":"<<q.swept_volume
         <<",\"boundary_bending\":"<<q.boundary_bending<<",\"c2_error\":"<<full.c2_error
         <<",\"start_distance\":"<<(r.corridor.section(0).center-seed.front()).norm()<<",\"end_distance\":"<<(r.corridor.section(1).center-seed.back()).norm();
      if(repeat==0){writeGeometry(r.corridor,output/"corridor.csv");writeGeometry(core,output/"core.csv");
        std::ofstream planes(output/"support_planes.csv");planes<<std::setprecision(17)<<"span,begin,end,nx,ny,nz,bound\n";
        for(const auto& p:r.cover)planes<<p.span<<','<<p.begin<<','<<p.end<<','<<p.normal[0]<<','<<p.normal[1]<<','<<p.normal[2]<<','<<p.bound<<'\n';}
    }log<<"}\n";log.flush();
  }rusage usage{};getrusage(RUSAGE_SELF,&usage);log<<"{\"max_rss_kib\":"<<usage.ru_maxrss<<"}\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
