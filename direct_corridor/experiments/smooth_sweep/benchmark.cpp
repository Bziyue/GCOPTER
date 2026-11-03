// SPDX-License-Identifier: MIT
#include "refit.hpp"
#include "corridor_benchmark_common.hpp"
#include "../../src/hermite_optimizer.hpp"
using gcopter::direct::experiment::ShapeFitMetrics;
struct OtherPatch : corridor_volume::Patch {
  std::size_t* unresolved;
  bool valid = true;
  bool contains(const Vec3& p) const { return corridor_volume::Patch::contains(p,*unresolved); }
};
static void exportGeometry(const Corridor& c, const std::filesystem::path& dir,
                           const std::string& name, double seed_length, std::ostream& out) {
  std::ofstream file(dir/(name+"_controls.csv"));
  file << std::setprecision(17) << "u0,u1,cx,cy,cz,m00,m10,m20,m01,m11,m21\n";
  int folds=0, certified=0;
  double ratio=0, curvature=0, angle=0, energy=0,boundary_energy=0;
  for (size_t i=0;i<c.pieces();++i) {
    certified += detail::regularTube(c.centerControls().data()+6*i,c.shapeControls().data()+6*i);
    for (int j=0;j<6;++j) {
      const auto& cp=c.centerControls()[6*i+j];
      const auto& mp=c.shapeControls()[6*i+j];
      file<<c.knots()[i]<<','<<c.knots()[i+1]<<','<<cp[0]<<','<<cp[1]<<','<<cp[2];
      for(int k=0;k<6;++k) file<<','<<mp.data()[k];
      file<<'\n';
    }
    bool folded=false;
    for(int j=0;j<=128;++j) {
      const double h=c.knots()[i+1]-c.knots()[i];
      const auto s=c.section(c.knots()[i]+h*j/128.);
      const Vec3 n=s.shape.col(0).cross(s.shape.col(1));
      const double j0=n.dot(s.first), g=(s.shape_first.transpose()*n).norm();
      folded |= j0<=g;
      ratio=std::max(ratio,g/j0);
      curvature=std::max(curvature,s.first.cross(s.second).norm()/std::pow(s.first.norm(),3));
      angle=std::max(angle,std::acos(std::clamp(n.normalized().dot(s.first.normalized()),-1.,1.))*180/std::acos(-1.));
      // Center curvature energy, integrated in physical center arc length.
      const double k=s.first.cross(s.second).norm()/std::pow(s.first.norm(),3);
      energy+=(j==0||j==128?.5:1.)*h/128.*s.first.norm()*k*k;
      boundary_energy+=(j==0||j==128?.5:1.)*h/128.*
          (s.second.squaredNorm()+.5*s.shape_second.squaredNorm())/std::pow(seed_length,3);
    }
    folds+=folded;
  }
  out<<",\"spans\":"<<c.pieces()<<",\"folded_spans\":"<<folds
     <<",\"certified_spans\":"<<certified<<",\"max_jacobian_ratio\":"<<ratio
     <<",\"max_curvature\":"<<curvature<<",\"max_angle_deg\":"<<angle
     <<",\"curvature_energy\":"<<energy<<",\"boundary_acceleration_energy\":"<<boundary_energy;
}
static double samplePenetration(const Corridor& c,const ObstacleIndex& map,const Bounds& bounds) {
  double violation=0;
  std::vector<int> ids;
  for(size_t i=0;i<c.pieces();++i) for(int k=0;k<=8;++k) {
    const auto s=c.section(c.knots()[i]+(c.knots()[i+1]-c.knots()[i])*k/8.);
    for(int j=0;j<24;++j) for(double radius:{0.,.5,1.}) {
      const double angle=2*std::acos(-1.)*j/24.;
      const Vec3 p=s.point(radius*Eigen::Vector2d(std::cos(angle),std::sin(angle)));
      violation=std::max({violation,(bounds.lower.array()+.25-p.array()).maxCoeff(),
                           (p.array()+.25-bounds.upper.array()).maxCoeff()});
      map.query(p-Vec3::Constant(.25),p+Vec3::Constant(.25),ids);
      for(int id:ids) {
        const auto& obstacle=map.obstacles()[id];
        violation=std::max(violation,obstacle.radius+.25-(obstacle.center-p).norm());
      }
    }
  }
  return violation;
}
int main(int argc,char**argv) {
  try {
    if(argc!=15) throw std::runtime_error("seed map output frame_smoothing section_smoothing certify repeats samples xmin ymin zmin xmax ymax zmax");
    auto seed=read(argv[1]), points=read(argv[2]);
    double seed_length=0;for(size_t i=1;i<seed.size();++i) seed_length+=(seed[i]-seed[i-1]).norm();
    std::filesystem::path dir=argv[3]; std::filesystem::create_directories(dir);
    Bounds bounds{Vec3(std::stod(argv[9]),std::stod(argv[10]),std::stod(argv[11])),
                  Vec3(std::stod(argv[12]),std::stod(argv[13]),std::stod(argv[14]))};
    std::vector<Sphere> spheres; for(const auto& p:points) spheres.push_back({p,std::sqrt(3)*.25/2});
    ObstacleIndex index(std::move(spheres));
    Options o; o.control_spacing=.4; o.min_radius=.03; o.budget_ms=10000;
    o.robot_radius=.2; o.safety_margin=.05; o.local_range=3; o.local_progress=7;
    double diagonal=0;
    for(auto box:expansionBoxes(seed,bounds,3,.252)) diagonal=std::max(diagonal,(box.bounds.upper-box.bounds.lower).norm());
    o.max_radius=.5*diagonal+.001; o.max_offset=diagonal+.001;
    std::array<Result,2> results;
    std::ofstream log(dir/"results.jsonl"); log<<std::setprecision(12);
    for(int repeat=-1;repeat<std::stoi(argv[7]);++repeat) for(int mode=0;mode<2;++mode) {
      auto settings=o; ShapeFitMetrics m;
      if(mode) {
        settings.frame_smoothing=std::stod(argv[4]); settings.section_smoothing=std::stod(argv[5]);
        settings.certify_jacobian=std::stoi(argv[6]);
        if(const char* v=std::getenv("SWEEP_JETS")) settings.jet_smoothing=std::stod(v);
        if(const char* v=std::getenv("SWEEP_ROUNDS")) settings.inflation_iterations=std::stoi(v);
      }
      const char* warmup=std::getenv("SWEEP_WARMUP");
      auto r=mode?detail::generateWithShapeStage(seed,index,bounds,settings,{},warmup?std::stoi(warmup):12,true):
        gcopter::direct::experiment::generateCoordinateShape(seed,index,bounds,settings,m,12,false,true);
      if(!r) throw std::runtime_error(r.message);
      log<<"{\"mode\":\""<<(mode?"smooth":"boundary-12")<<"\",\"repeat\":"<<repeat
         <<",\"generation_ms\":"<<r.metrics.total_ms<<",\"certificate_ms\":"<<r.metrics.certificate_ms
         <<",\"optimization_ms\":"<<r.metrics.optimization_ms;
      log<<",\"smoothing_ms\":"<<r.metrics.smoothing_ms<<",\"mean_jet_fraction\":"<<r.metrics.mean_jet_fraction;
      log<<",\"regularity_repairs\":"<<r.metrics.regularity_repairs;
      if(repeat==0) {
        exportGeometry(r.corridor,dir,mode?"smooth":"baseline",seed_length,log);
        const double penetration=samplePenetration(r.corridor,index,bounds);
        log<<",\"max_sample_penetration_m\":"<<penetration;
        if(penetration>1e-6) throw std::runtime_error("independent collision audit failed");
      }
      log<<"}\n"; log.flush(); results[mode]=std::move(r);
    }
    if(std::stoi(argv[8])>0) {
      std::size_t unresolved=0;
      std::vector<OtherPatch> other;
      for(const auto& p:corridor_volume::patches(results[1].corridor)) {
        OtherPatch q; static_cast<corridor_volume::Patch&>(q)=p; q.unresolved=&unresolved; other.push_back(q);
      }
      auto v=corridor_volume::measure(corridor_volume::patches(results[0].corridor),other,seed,std::stoi(argv[8]));
      log<<"{\"baseline_volume\":"<<v.volume[0].mean<<",\"smooth_volume\":"<<v.volume[1].mean
         <<",\"paired_delta\":"<<v.paired_difference.mean<<",\"paired_ci95\":"<<v.paired_difference.ci95
         <<",\"unresolved\":"<<v.unresolved+unresolved<<"}\n";
    }
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
