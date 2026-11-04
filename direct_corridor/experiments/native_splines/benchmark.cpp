// SPDX-License-Identifier: MIT
#include "cubic_certificates.hpp"
#include "obstacle_cover.hpp"
#include "cubic_generator.hpp"
#include "corridor_volume.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <sys/resource.h>
using namespace gcopter::direct;
namespace ns=gcopter::direct::native;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
static std::vector<Vec3> read(const char* path) {
  std::ifstream f(path);if(!f)throw std::runtime_error("missing XYZ input");
  std::vector<Vec3> out;std::string line;
  while(std::getline(f,line)) {
    if(line.empty()||line[0]=='#')continue;
    std::istringstream row(line);Vec3 p;
    if(!(row>>p[0]>>p[1]>>p[2])||!p.allFinite())throw std::runtime_error("invalid XYZ input");
    out.push_back(p);
  }
  return out;
}
static void number(std::ostream& o,double x){if(std::isfinite(x))o<<x;else o<<"null";}
static void certificate(std::ostream& o,const ns::Certificate& c) {
  o<<",\"support_pass\":"<<c.support<<",\"jacobian_pass\":"<<c.jacobian
   <<",\"support_failed\":"<<c.support_failed<<",\"jacobian_failed\":"<<c.jacobian_failed
   <<",\"support_leaves\":"<<c.support_leaves<<",\"jacobian_leaves\":"<<c.jacobian_leaves
   <<",\"support_depth\":"<<c.support_depth<<",\"jacobian_depth\":"<<c.jacobian_depth
   <<",\"support_residual\":";number(o,c.support_residual);
  o<<",\"cone_relative_margin\":";number(o,c.cone_relative_margin);
}
static void quality(std::ostream& o,const ns::Quality& q) {
  o<<",\"c2_error\":"<<q.c2_error<<",\"center_energy\":"<<q.center_energy
   <<",\"shape_energy\":"<<q.shape_energy<<",\"jerk_energy\":"<<q.jerk_energy
   <<",\"min_axis\":"<<q.min_axis<<",\"max_curvature\":";number(o,q.max_curvature);
  o<<",\"normalized_jacobian\":";number(o,q.normalized_jacobian);
  o<<",\"sampled_folded_spans\":"<<q.folded_spans;
}
template<int D> static double fixedBound(const ns::Curve<D,ns::V9>& p,const Vec3& n,double b,int depth) {
  if(!depth)return ns::supportBound<D>(p,n,b);
  const auto halves=ns::split<D>(p);
  return std::max(fixedBound<D>(halves.first,n,b,depth-1),fixedBound<D>(halves.second,n,b,depth-1));
}
struct OtherPatch:corridor_volume::Patch {
  bool valid=true,active=true;std::size_t* unresolved=nullptr;
  bool contains(const Vec3& p) const {return active&&corridor_volume::Patch::contains(p,*unresolved);}
};
static double sampledPenetration(const ns::Geometry& g,const ObstacleIndex& map,const Bounds& bounds) {
  double violation=0;std::vector<int> ids;
  for(int i=0;i<int(g.pieces());++i)for(int j=0;j<=12;++j) {
    const auto s=g.section(g.breaks[i]+(g.breaks[i+1]-g.breaks[i])*j/12.);
    for(int k=0;k<32;++k)for(double radius:{0.,.5,1.}) {
      const double angle=2*std::acos(-1.)*k/32;
      const Vec3 p=s.point(radius*Eigen::Vector2d(std::cos(angle),std::sin(angle)));
      violation=std::max({violation,(bounds.lower.array()+.25-p.array()).maxCoeff(),(p.array()+.25-bounds.upper.array()).maxCoeff()});
      map.query(p-Vec3::Constant(.25),p+Vec3::Constant(.25),ids);
      for(int id:ids){const auto& sphere=map.obstacles()[id];violation=std::max(violation,sphere.radius+.25-(sphere.center-p).norm());}
    }
  }
  return violation;
}
int main(int argc,char** argv) {
  try {
    if(argc!=12)throw std::runtime_error("seed map output xmin ymin zmin xmax ymax zmax repeats samples");
    const auto seed=read(argv[1]),points=read(argv[2]);
    if(seed.size()<2)throw std::runtime_error("seed needs two points");
    const Bounds bounds{Vec3(std::stod(argv[4]),std::stod(argv[5]),std::stod(argv[6])),Vec3(std::stod(argv[7]),std::stod(argv[8]),std::stod(argv[9]))};
    const std::filesystem::path dir=argv[3];std::filesystem::create_directories(dir);
    const int repeats=std::stoi(argv[10]),samples=std::stoi(argv[11]);
    double length=0;for(size_t i=1;i<seed.size();++i)length+=(seed[i]-seed[i-1]).norm();
    std::vector<Sphere> spheres;for(const auto& p:points)spheres.push_back({p,std::sqrt(3)*.25/2});
    auto start=Clock::now();ObstacleIndex index(std::move(spheres));const double index_ms=ms(start);
    Options o;o.control_spacing=.4;o.min_radius=.03;o.budget_ms=10000;
    o.robot_radius=.2;o.safety_margin=.05;o.local_range=3;o.local_progress=7;
    o.frame_smoothing=1.5;o.section_smoothing=.2;o.jet_smoothing=.01;o.certify_jacobian=true;o.initial_iterations=8;
    double diagonal=0;Vec3 a,b=seed.front();
    for(size_t i=1;i<seed.size();) {
      a=b;if((a-seed[i]).norm()>7)b=a+(seed[i]-a).normalized()*7;else b=seed[i++];
      const Vec3 lo=(a.cwiseMin(b)-Vec3::Constant(3)).cwiseMax(bounds.lower+Vec3::Constant(.252));
      const Vec3 hi=(a.cwiseMax(b)+Vec3::Constant(3)).cwiseMin(bounds.upper-Vec3::Constant(.252));
      diagonal=std::max(diagonal,(hi-lo).norm());
    }
    o.max_radius=.5*diagonal+.001;o.max_offset=diagonal+.001;
    std::ofstream log(dir/"rows.jsonl");log<<std::setprecision(17);
    log<<"{\"kind\":\"input\",\"seed_length\":"<<length<<",\"obstacles\":"<<points.size()<<",\"index_ms\":"<<index_ms<<"}\n";
    std::vector<detail::SupportPlane> cover;
    std::vector<double> baseline_times;
    Result baseline;
    for(int rep=-1;rep<repeats;++rep) {
      start=Clock::now();
      auto r=rep==-1?detail::generateWithShapeStage(seed,index,bounds,o,
        [&](Corridor&,const auto& p,double,const auto&){cover=p;},8,true):generate(seed,index,bounds,o);
      const double wall=ms(start);
      log<<"{\"kind\":\"generation\",\"model\":\"H5-S\",\"repeat\":"<<rep<<",\"status\":\""<<statusName(r.status)<<"\",\"wall_ms\":"<<wall
       <<",\"total_ms\":"<<r.metrics.total_ms<<",\"certificate_ms\":"<<r.metrics.certificate_ms
       <<",\"optimization_ms\":"<<r.metrics.optimization_ms<<",\"smoothing_ms\":"<<r.metrics.smoothing_ms
       <<",\"nodes\":"<<r.metrics.nodes<<",\"constraints\":"<<r.metrics.constraints
       <<",\"active_constraints\":"<<r.metrics.active_constraints
       <<",\"newton_steps\":"<<r.metrics.newton_steps<<",\"refinements\":"<<r.metrics.refinements<<"}\n";log.flush();
      if(!r)throw std::runtime_error(r.message);
      baseline_times.push_back(r.metrics.total_ms);
      baseline=std::move(r);
    }
    ns::Geometry base=ns::Geometry::fromCorridor(baseline.corridor);
    // The cover is generated from the same deterministic accepted inflation
    // iterate before smoothing; final baseline must satisfy it independently.
    if(!ns::certify(base,cover).safe())throw std::runtime_error("captured H5-S cover audit failed");
    std::ofstream planes(dir/"support_planes.csv");planes<<std::setprecision(17)<<"span,begin,end,nx,ny,nz,bound\n";
    for(const auto& p:cover)planes<<p.span<<','<<p.begin<<','<<p.end<<','<<p.normal[0]<<','<<p.normal[1]<<','<<p.normal[2]<<','<<p.bound<<'\n';
    const bool generation_only=std::getenv("NATIVE_GENERATION_ONLY");
    std::vector<std::string> names={"H5-S","C3-I","Q5-U","H5-C3-T","Q5-T","H5-Z-fixed"};
    if(generation_only)names={"H5-S"};
    std::vector<ns::Geometry> variants(names.size());variants[0]=base;
    for(int model=0;model<int(names.size());++model)for(int rep=-1;rep<repeats;++rep) {
      double fraction=1;start=Clock::now();ns::Geometry g;
      if(model==0)g=base;
      if(model==1)g=ns::naturalCubic(base);
      if(model==2)g=ns::minimumJerk(base);
      if(model==3)g=ns::safeJetProposal(base,ns::naturalCubic(base),cover,fraction);
      if(model==4)g=ns::safeJetProposal(base,ns::minimumJerk(base),cover,fraction);
      if(model==5) {
        g=base;
        for(int i=0;i<int(g.pieces());++i)for(int j=0;j<6;++j)
          g.controls[6*i+j].tail<6>()=base.controls[6*i+(j<3?0:5)].tail<6>();
      }
      const double construction=ms(start);
      start=Clock::now();const auto cert=ns::certify(g,cover);const double cert_ms=ms(start);
      start=Clock::now();const auto jac=ns::certify(g,{});const double jac_ms=ms(start);(void)jac;
      log<<"{\"kind\":\"representation\",\"model\":\""<<names[model]<<"\",\"repeat\":"<<rep
         <<",\"degree\":"<<g.degree<<",\"spans\":"<<g.pieces()<<",\"control_rows\":"<<g.controls.size()
         <<",\"construction_ms\":"<<construction<<",\"certificate_ms\":"<<cert_ms
         <<",\"jacobian_ms\":"<<jac_ms<<",\"jet_fraction\":"<<fraction;
      certificate(log,cert);
      if(rep==0){quality(log,ns::quality(g,length));g.write((dir/(names[model]+".csv")).string());}
      log<<"}\n";log.flush();variants[model]=std::move(g);
    }
    if(std::getenv("NATIVE_CANDIDATES"))for(int method=0;method<(std::getenv("NATIVE_METHOD_COUNT")?std::stoi(std::getenv("NATIVE_METHOD_COUNT")):3);++method) {
      const std::string name=method==0?"C3-OPT-native":method==1?"C3-OPT-project":"C3-NBC-project";
      ns::OptimizerSettings solver;solver.natural_endpoints=method==2;
      if(method==0)if(const char* preset=std::getenv("NATIVE_PRESET")) {
        const std::string value=preset;
        if(value!="balanced"&&value!="fast"&&value!="volume")throw std::runtime_error("unknown native preset");
        ns::applyNativePreset(value=="balanced"?ns::NativePreset::Balanced:value=="fast"?ns::NativePreset::Fast:ns::NativePreset::Volume,o,solver);
      }
      if(const char* v=std::getenv("NATIVE_ENERGY"))solver.center_energy=solver.shape_energy=std::stod(v);
      if(const char* v=std::getenv("NATIVE_CENTER_ENERGY"))solver.center_energy=std::stod(v);
      if(const char* v=std::getenv("NATIVE_SHAPE_ENERGY"))solver.shape_energy=std::stod(v);
      if(const char* v=std::getenv("NATIVE_RADIUS_FRACTION"))solver.initial_radius_fraction=std::stod(v);
      if(const char* v=std::getenv("NATIVE_BARRIER_STAGES"))solver.barrier_stages=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_CORRECTOR_ITERATIONS"))solver.corrector_iterations=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_PROGRESS_THRESHOLD"))solver.progress_threshold=std::stod(v);
      if(const char* v=std::getenv("NATIVE_PROGRESS_SAMPLES"))solver.progress_samples=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_PROGRESS_DEPTH"))solver.progress_depth=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_POLAR_PRUNING"))solver.polar_pruning=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_PROJECTED_PRUNING"))solver.projected_pruning=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_DISTANCE_BARRIER"))solver.distance_barrier=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_INITIAL_SWEEPS"))solver.initial_sweeps=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_INITIAL_FRACTION"))solver.initial_fraction=std::stod(v);
      if(const char* v=std::getenv("NATIVE_HESSIAN_INTERVAL"))solver.hessian_interval=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_WEAK_CURVATURE_RATIO"))solver.weak_curvature_ratio=std::stod(v);
      if(const char* v=std::getenv("NATIVE_MERGE_SUPPORTS"))solver.merge_shared_supports=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_TENSOR_ASSEMBLY"))solver.tensor_assembly=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_GROUPED_ASSEMBLY"))solver.grouped_assembly=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_AXIS_KERNELS"))solver.axis_kernels=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_TRIAL_FIRST"))solver.trial_first=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_WARM_STAGE"))solver.warm_start_stage=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_WARM_RESERVE"))solver.warm_reserve=std::stod(v);
      if(const char* v=std::getenv("NATIVE_CACHED_LINE"))solver.cached_line=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_SKIP_UNUSED_VALUES"))solver.skip_unused_values=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_BOUNDED_LINE"))solver.bounded_line=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_REUSE_ENERGY"))solver.reuse_energy=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_CACHE_VALUES"))solver.cache_values=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_SEPARATOR_REUSE_GAP"))solver.separator_reuse_gap=std::stod(v);
      if(const char* v=std::getenv("NATIVE_SUPPORT_EXCHANGE"))solver.support_exchange=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_EXCHANGE_ROUNDS"))solver.exchange_rounds=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_EXCHANGE_ITERATIONS"))solver.exchange_iterations=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_LAZY_PRIORITIES"))solver.lazy_priorities=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_ADAPTIVE_KNOTS"))solver.adaptive_knots=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_PATH_PREDICTOR"))solver.path_predictor=std::stod(v);
      if(const char* v=std::getenv("NATIVE_VOLUME_REFINEMENTS"))solver.volume_refinements=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_VOLUME_TRUST"))solver.volume_trust=std::stod(v);
      if(const char* v=std::getenv("NATIVE_COORDINATE_SWEEPS"))solver.coordinate_sweeps=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_WORKING_SET"))solver.working_set=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_FRACTION_BOUNDARY"))solver.fraction_boundary=std::stod(v);
      if(const char* v=std::getenv("NATIVE_RECENTER"))solver.recenter_frames=std::stoi(v)!=0;
      if(const char* v=std::getenv("NATIVE_ITERATIONS"))solver.iterations=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_INITIAL_ITERATIONS"))o.initial_iterations=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_INFLATIONS"))o.inflation_iterations=std::stoi(v);
      if(const char* v=std::getenv("NATIVE_SPACING"))o.control_spacing=std::stod(v);
      bool success=false;ns::Geometry last;
      log<<"{\"kind\":\"configuration\",\"model\":\""<<name<<"\",\"control_spacing\":"<<o.control_spacing
         <<",\"initial_iterations\":"<<o.initial_iterations<<",\"iterations\":"<<solver.iterations
         <<",\"corrector_iterations\":"<<solver.corrector_iterations<<",\"inflations\":"<<o.inflation_iterations
         <<",\"center_energy\":"<<solver.center_energy<<",\"shape_energy\":"<<solver.shape_energy
         <<",\"progress_samples\":"<<solver.progress_samples<<",\"progress_depth\":"<<solver.progress_depth
         <<",\"merge_shared_supports\":"<<solver.merge_shared_supports<<",\"tensor_assembly\":"<<solver.tensor_assembly
         <<",\"grouped_assembly\":"<<solver.grouped_assembly
         <<",\"axis_kernels\":"<<solver.axis_kernels
         <<",\"skip_unused_values\":"<<solver.skip_unused_values
         <<",\"bounded_line\":"<<solver.bounded_line
         <<",\"cached_line\":"<<solver.cached_line<<",\"trial_first\":"<<solver.trial_first
         <<",\"reuse_energy\":"<<solver.reuse_energy<<",\"cache_values\":"<<solver.cache_values
         <<",\"separator_reuse_gap\":"<<solver.separator_reuse_gap
         <<",\"support_exchange\":"<<solver.support_exchange<<",\"exchange_rounds\":"<<solver.exchange_rounds<<",\"exchange_iterations\":"<<solver.exchange_iterations
         <<",\"lazy_priorities\":"<<solver.lazy_priorities
         <<",\"adaptive_knots\":"<<solver.adaptive_knots
         <<",\"path_predictor\":"<<solver.path_predictor<<",\"volume_refinements\":"<<solver.volume_refinements<<",\"volume_trust\":"<<solver.volume_trust
         <<",\"warm_start_stage\":"<<solver.warm_start_stage<<",\"warm_reserve\":"<<solver.warm_reserve
         <<",\"coordinate_sweeps\":"<<solver.coordinate_sweeps<<",\"polar_pruning\":"<<solver.polar_pruning
         <<",\"projected_pruning\":"<<solver.projected_pruning
         <<",\"distance_barrier\":"<<solver.distance_barrier<<",\"initial_sweeps\":"<<solver.initial_sweeps<<",\"initial_fraction\":"<<solver.initial_fraction
         <<",\"hessian_interval\":"<<solver.hessian_interval
         <<",\"weak_curvature_ratio\":"<<solver.weak_curvature_ratio
         <<",\"working_set\":"<<solver.working_set<<"}\n";
      for(int rep=-1;rep<repeats;++rep) {
        solver.check_gradients=rep==-1;
        auto result=ns::generateCubic(seed,index,bounds,o,solver,method?&base:nullptr);
        const auto& m=result.optimizer;const auto& c=result.cover_metrics;
        log<<"{\"kind\":\"generation\",\"model\":\""<<name<<"\",\"repeat\":"<<rep
           <<",\"status\":\""<<result.status<<"\",\"message\":\""<<result.message<<"\""
           <<",\"total_ms\":"<<result.total_ms+(method?baseline_times[rep+1]:0.)
           <<",\"own_ms\":"<<result.total_ms<<",\"h5_initialization_ms\":"<<(method?baseline_times[rep+1]:0.)
           <<",\"initialization_ms\":"<<result.initialization_ms<<",\"refinements\":"<<result.refinements
           <<",\"initialization_attempts\":"<<result.initialization_attempts<<",\"cover_ms\":"<<c.total_ms
           <<",\"query_ms\":"<<c.query_ms<<",\"separation_ms\":"<<c.separation_ms
           <<",\"reused_planes\":"<<c.reused_planes<<",\"reused_spheres\":"<<c.reused_spheres<<",\"separator_cache_bytes\":"<<c.separator_cache_bytes
           <<",\"priority_evaluations\":"<<c.priority_evaluations
           <<",\"setup_ms\":"<<m.setup_ms<<",\"assembly_ms\":"<<m.assembly_ms<<",\"solve_ms\":"<<m.solve_ms
           <<",\"line_search_ms\":"<<m.line_search_ms<<",\"jacobian_check_ms\":"<<m.certificate_ms
           <<",\"verification_ms\":"<<result.verification_ms<<",\"variables\":"<<m.variables<<",\"controls\":"<<m.controls
           <<",\"active_constraints\":"<<m.active_constraints<<",\"final_decrement\":"<<m.final_decrement
           <<",\"merged_supports\":"<<m.merged_supports<<",\"support_groups\":"<<m.support_groups<<",\"pruned_supports\":"<<m.pruned_supports
           <<",\"pruning_ms\":"<<m.pruning_ms
           <<",\"initial_inflation_ms\":"<<m.initial_inflation_ms
           <<",\"hessian_refreshes\":"<<m.hessian_refreshes
           <<",\"bound_line_evaluations\":"<<m.bound_line_evaluations
           <<",\"bound_line_accepts\":"<<m.bound_line_accepts
           <<",\"exact_line_evaluations\":"<<m.exact_line_evaluations
           <<",\"weak_curvature_rows\":"<<m.weak_curvature_rows
           <<",\"axis_supports\":"<<m.axis_supports
           <<",\"gradient_support_rows\":"<<m.gradient_support_rows<<",\"hessian_support_rows\":"<<m.hessian_support_rows
           <<",\"boundary_support_rows\":"<<m.boundary_support_rows<<",\"line_support_rows\":"<<m.line_support_rows
           <<",\"full_step_accepts\":"<<m.full_step_accepts<<",\"feasible_scans\":"<<m.feasible_scans
           <<",\"warm_reserves\":"<<m.warm_reserves
           <<",\"energy_builds\":"<<m.energy_builds<<",\"minimum_step\":"<<m.minimum_step<<",\"mean_step\":"<<(m.steps?m.step_sum/m.steps:0)
           <<",\"predictor_steps\":"<<m.predictor_steps<<",\"predictor_rejections\":"<<m.predictor_rejections<<",\"predictor_ms\":"<<m.predictor_ms
           <<",\"volume_steps\":"<<m.volume_steps<<",\"volume_ms\":"<<m.volume_ms<<",\"volume_gain\":"<<m.volume_gain
           <<",\"exchange_steps\":"<<m.exchange_steps<<",\"exchange_witnesses\":"<<m.exchange_witnesses<<",\"oracle_nodes\":"<<m.oracle_nodes<<",\"oracle_unresolved\":"<<m.oracle_unresolved<<",\"oracle_ms\":"<<m.oracle_ms
           <<",\"coordinate_ms\":"<<m.coordinate_ms<<",\"coordinate_updates\":"<<m.coordinate_updates<<",\"coordinate_rejections\":"<<m.coordinate_rejections
           <<",\"constraints\":"<<m.constraints<<",\"newton_steps\":"<<m.steps<<",\"termination\":\""<<m.termination<<"\""
           <<",\"backtracks\":"<<m.backtracks<<",\"regularity_backtracks\":"<<m.regularity_backtracks
           <<",\"regularity_repairs\":"<<m.regularity_repairs
           <<",\"damping_retries\":"<<m.damping_retries<<",\"equality_rank\":"<<m.equality_rank
           <<",\"initial_scale\":"<<m.initial_scale<<",\"gradient_error\":"<<m.gradient_error<<",\"hessian_error\":"<<m.hessian_error;
        log<<",\"optimization_passes\":[";
        for(size_t p=0;p<m.passes.size();++p) {
          const auto& pass=m.passes[p];if(p)log<<',';
          log<<"{\"geometric\":"<<pass.geometric<<",\"retained\":"<<pass.retained
             <<",\"boundary\":"<<pass.boundary<<",\"progress\":"<<pass.progress
             <<",\"steps\":"<<pass.steps<<",\"pruning_ms\":"<<pass.pruning_ms<<'}';
        }
        log<<']';
        success=result.status=="SUCCESS";
        if(success) {
          certificate(log,ns::certify(result.corridor,result.cover));
          if(rep==0){
            quality(log,ns::quality(result.corridor,length));result.corridor.write((dir/(name+".csv")).string());
            std::ofstream output(dir/(name+"_support_planes.csv"));output<<std::setprecision(17)<<"span,begin,end,nx,ny,nz,bound\n";
            for(const auto& p:result.cover)output<<p.span<<','<<p.begin<<','<<p.end<<','<<p.normal[0]<<','<<p.normal[1]<<','<<p.normal[2]<<','<<p.bound<<'\n';
          }
          last=std::move(result.corridor);
        }
        log<<"}\n";log.flush();
      }
      if(success){names.push_back(name);variants.push_back(std::move(last));}
    }
    for(int i=0;i<int(names.size());++i) {
      ns::QueryCache cache;ns::CoverMetrics timing;start=Clock::now();
      const std::function<void()> stop=[&]{if(ms(start)>10000)throw detail::Failure{Status::Timeout,"map audit timeout"};};
      log<<"{\"kind\":\"fresh_map_certificate\",\"model\":\""<<names[i]<<"\"";
      try {
        const auto fresh=ns::mapCover(variants[i],seed,index,bounds,o,cache,timing,stop);
        const auto c=ns::certify(variants[i],fresh);
        log<<",\"status\":\""<<(c.safe()?"SUCCESS":"CERTIFICATE_FAILED")<<"\"";certificate(log,c);
      }catch(const detail::Failure& error){log<<",\"status\":\""<<statusName(error.status)<<"\",\"message\":\""<<error.message<<"\"";}
      log<<",\"wall_ms\":"<<ms(start)<<",\"query_ms\":"<<timing.query_ms<<",\"separation_ms\":"<<timing.separation_ms;
      log<<",\"sampled_penetration_m\":"<<sampledPenetration(variants[i],index,bounds)<<"}\n";log.flush();
    }
    // Same curve, same planes, native versus exact elevation, fixed subdivision.
    if(!generation_only) {
    const auto cubic=variants[1],quintic=ns::elevateGeometry(cubic);
    for(int degree:{3,5})for(int depth=0;depth<=3;++depth)for(int rep=-1;rep<repeats;++rep) {
      const auto& g=degree==3?cubic:quintic;
      start=Clock::now();double residual=-INFINITY,sum=0;int failed=0;
      for(const auto& p:cover) {
        const double bound=degree==3?fixedBound<3>(ns::restrictCurve<3>(g.curve<3>(p.span),p.begin,p.end),p.normal,p.bound,depth):
          fixedBound<5>(ns::restrictCurve<5>(g.curve<5>(p.span),p.begin,p.end),p.normal,p.bound,depth);
        residual=std::max(residual,bound);sum+=bound;failed+=bound>1e-9;
      }
      const double elapsed=ms(start);
      log<<"{\"kind\":\"elevation\",\"degree\":"<<degree<<",\"depth\":"<<depth<<",\"repeat\":"<<rep
         <<",\"support_ms\":"<<elapsed<<",\"residual\":"<<residual<<",\"mean_bound\":"<<sum/cover.size()<<",\"failed\":"<<failed<<"}\n";
    }
    }
    if(samples>0) {
      std::vector<std::vector<corridor_volume::Patch>> patches;
      std::vector<corridor_volume::Box> domain;
      for(const auto& g:variants){patches.push_back(corridor_volume::patches(g.quinticAdapter()));for(const auto& p:patches.back())domain.push_back(p.box);}
      for(int i=1;i<int(names.size());++i) {
        std::size_t unresolved=0;std::vector<OtherPatch> other;
        for(const auto& p:patches[i]){OtherPatch a;static_cast<corridor_volume::Patch&>(a)=p;a.unresolved=&unresolved;other.push_back(a);}
        // These empty membership patches impose the identical common sampling
        // cells for EVERY comparison, including unsafe diagnostic geometries.
        for(const auto& box:domain){OtherPatch a;a.box=box;a.active=false;other.push_back(a);}
        start=Clock::now();const auto v=corridor_volume::measure(patches[0],other,seed,samples);
        log<<"{\"kind\":\"volume\",\"model\":\""<<names[i]<<"\",\"baseline\":"<<v.volume[0].mean
           <<",\"volume\":"<<v.volume[1].mean<<",\"paired_delta\":"<<v.paired_difference.mean
           <<",\"paired_ci95\":"<<v.paired_difference.ci95<<",\"samples\":"<<v.samples<<",\"cells\":"<<v.cells
           <<",\"unresolved\":"<<v.unresolved+unresolved<<",\"measurement_ms\":"<<ms(start)<<"}\n";log.flush();
      }
    }
    rusage usage{};getrusage(RUSAGE_SELF,&usage);log<<"{\"kind\":\"resource\",\"peak_rss_kib\":"<<usage.ru_maxrss<<"}\n";
    std::cout<<dir<<": complete\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
