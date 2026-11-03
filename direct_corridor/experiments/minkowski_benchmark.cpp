// SPDX-License-Identifier: MIT
#include "minkowski.hpp"
#include "minkowski_volume.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace gcopter::direct;
namespace mk=gcopter::direct::minkowski;
namespace cv=corridor_volume;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
std::vector<Vec3> read(const char* path) {
  std::ifstream stream(path);
  if(!stream) throw std::runtime_error("input missing");
  std::vector<Vec3> values; std::string line;
  while(std::getline(stream,line)) {
    if(line.empty() || line[0]=='#') continue;
    std::istringstream row(line); Vec3 p; std::string extra;
    if(!(row>>p[0]>>p[1]>>p[2]) || row>>extra || !p.allFinite()) throw std::runtime_error("invalid xyz");
    values.push_back(p);
  }
  return values;
}
struct Expansion {double begin,end; Bounds bounds;};
std::vector<Expansion> expansion(const std::vector<Vec3>& seed,const Bounds& world,double pad) {
  std::vector<Expansion> out; double arc=0; Vec3 a=seed.front();
  for(std::size_t i=1;i<seed.size();) {
    const double length=(seed[i]-a).norm();
    const Vec3 b=length>7 ? Vec3(a+7*(seed[i]-a)/length):seed[i++];
    const double end=arc+(b-a).norm();
    out.push_back({arc,end,{(a.cwiseMin(b)-Vec3::Constant(3)).cwiseMax(world.lower+Vec3::Constant(pad)),
                           (a.cwiseMax(b)+Vec3::Constant(3)).cwiseMin(world.upper-Vec3::Constant(pad))}});
    arc=end; a=b;
  }
  return out;
}
double median(std::vector<double> v) {
  std::sort(v.begin(),v.end());
  return .5*(v[(v.size()-1)/2]+v[v.size()/2]);
}
void list(std::ostream& out,const std::vector<double>& v) {
  out<<'['; for(std::size_t i=0;i<v.size();++i) out<<(i?",":"")<<v[i]; out<<']';
}
int main(int argc,char** argv) {
  try {
    if(argc!=24 && argc!=25) throw std::runtime_error("usage: benchmark seed map out xmin ymin zmin xmax ymax zmax resolution repeats samples templates subdivisions steps thinness angle spacing inflation adapt_orientation product_bounds affine orientation_smoothing [ellipse_spacing]");
    const auto seed=read(argv[1]), points=read(argv[2]);
    const std::filesystem::path dir=argv[3];
    const Bounds bounds{Vec3(std::stod(argv[4]),std::stod(argv[5]),std::stod(argv[6])),
                        Vec3(std::stod(argv[7]),std::stod(argv[8]),std::stod(argv[9]))};
    const double radius=std::sqrt(3.)*.5*std::stod(argv[10]);
    const int repeats=std::stoi(argv[11]), samples=std::stoi(argv[12]);
    if(seed.size()<2 || repeats<1 || repeats>100) throw std::runtime_error("invalid benchmark input");
    mk::Options options;
    options.templates=std::stoi(argv[13]); options.subdivisions=std::stoi(argv[14]);
    options.newton_iterations=std::stoi(argv[15]); options.thinness=std::stod(argv[16]);
    options.angle=std::stod(argv[17]); options.geometry.control_spacing=std::stod(argv[18]);
    options.geometry.inflation_iterations=std::stoi(argv[19]);
    options.adapt_orientation=std::stoi(argv[20])!=0;
    options.product_bounds=std::stoi(argv[21])!=0;
    options.affine=std::stoi(argv[22])!=0;
    options.orientation_smoothing=std::stod(argv[23]);
    auto& o=options.geometry;
    o.min_radius=.03; o.robot_radius=.2; o.safety_margin=.05; o.certificate_margin=.002;
    o.budget_ms=10000; o.local_range=3; o.local_progress=7;
    const auto boxes=expansion(seed,bounds,.252);
    double diagonal=0;
    for(const auto& box:boxes) diagonal=std::max(diagonal,(box.bounds.upper-box.bounds.lower).norm());
    o.max_radius=.5*diagonal+.001; o.max_offset=diagonal+.001;
    std::vector<Sphere> spheres; for(const auto& p:points) spheres.push_back({p,radius});
    const auto timer=Clock::now(); ObstacleIndex map(std::move(spheres)); const double index_ms=ms(timer);
    // Ellipse reference always retains its previous production defaults.
    auto ellipse_options=o;
    ellipse_options.control_spacing=argc==25?std::stod(argv[24]):.4;
    ellipse_options.inflation_iterations=2;
    Result ellipse; mk::Result minkowski;
    std::vector<double> et,mt;
    for(int rep=-1;rep<repeats;++rep) {
      auto e=[&]{ellipse=generate(seed,map,bounds,ellipse_options);};
      auto m=[&]{minkowski=mk::generate(seed,map,bounds,options);};
      if(rep%2==0) {e();m();} else {m();e();}
      if(!ellipse || !minkowski) throw std::runtime_error(!ellipse ? ellipse.message:minkowski.message);
      if(rep>=0) {et.push_back(ellipse.metrics.total_ms);mt.push_back(minkowski.metrics.total_ms);}
    }
    std::size_t audit_samples=0;
    double min_clearance=INFINITY, max_box_violation=0;
    for(int k=0;k<=256;++k) {
      const double u=k/256.,arc=u*boxes.back().end;
      auto at=std::upper_bound(boxes.begin(),boxes.end(),arc,[](double s,const Expansion& b){return s<b.end;});
      if(at==boxes.end()) --at;
      const auto section=minkowski.corridor.section(u);
      for(int j=0;j<32;++j) {
        const Vec3 edge=minkowski.corridor.boundary(u,2*std::acos(-1.)*j/32).point;
        for(double rho:{0.,.5,1.}) {
          ++audit_samples;
          const Vec3 p=section.center+rho*(edge-section.center);
          max_box_violation=std::max({max_box_violation,(at->bounds.lower-p).maxCoeff(),(p-at->bounds.upper).maxCoeff()});
          std::vector<int> ids; const Vec3 reach=Vec3::Constant(.252);
          map.query(p-reach,p+reach,ids);
          for(int id:ids) {
            const auto& sphere=map.obstacles()[id];
            const double clearance=(p-sphere.center).norm()-sphere.radius-.25;
            min_clearance=std::min(min_clearance,clearance);
            if(clearance<.002-1e-8) throw std::runtime_error("independent Minkowski sphere audit failed");
          }
          if((p.array()<bounds.lower.array()+.252-1e-8).any() ||
             (p.array()>bounds.upper.array()-.252+1e-8).any()) throw std::runtime_error("world bound audit failed");
        }
      }
    }
    if(max_box_violation>1e-7) throw std::runtime_error("local expansion box audit failed");
    if((minkowski.corridor.section(0).center-seed.front()).norm()>1e-10 ||
       (minkowski.corridor.section(1).center-seed.back()).norm()>1e-10) throw std::runtime_error("endpoint drift");
    std::size_t unresolved=0;
    const auto volume_start=Clock::now();
    const auto volume=cv::measure(cv::patches(ellipse.corridor),minkowski_volume::patches(minkowski.corridor,unresolved),seed,samples,8,1.,3.);
    const double volume_ms=ms(volume_start);
    if(volume.unresolved || unresolved) throw std::runtime_error("unresolved union volume roots");
    std::filesystem::create_directories(dir);
    std::ofstream out(dir/"metrics.json"); out<<std::setprecision(15);
    out<<"{\"status\":\"SUCCESS\",\"templates\":"<<options.templates<<",\"subdivisions\":"<<options.subdivisions
       <<",\"steps\":"<<options.newton_iterations<<",\"thinness\":"<<options.thinness<<",\"angle\":"<<options.angle
       <<",\"spacing\":"<<o.control_spacing<<",\"inflation_iterations\":"<<o.inflation_iterations
       <<",\"adapt_orientation\":"<<(options.adapt_orientation?"true":"false")
       <<",\"product_bounds\":"<<(options.product_bounds?"true":"false")
       <<",\"affine\":"<<(options.affine?"true":"false")
       <<",\"orientation_smoothing\":"<<options.orientation_smoothing
       <<",\"ellipse_spacing\":"<<ellipse_options.control_spacing
       <<",\"index_ms\":"<<index_ms<<",\"ellipse_median_ms\":"<<median(et)<<",\"minkowski_median_ms\":"<<median(mt)
       <<",\"ellipse_times_ms\":"; list(out,et); out<<",\"minkowski_times_ms\":"; list(out,mt);
    out<<",\"ellipse_union_m3\":"<<volume.volume[0].mean<<",\"ellipse_ci95\":"<<volume.volume[0].ci95
       <<",\"minkowski_union_m3\":"<<volume.volume[1].mean<<",\"minkowski_ci95\":"<<volume.volume[1].ci95
       <<",\"volume_delta_m3\":"<<volume.paired_difference.mean<<",\"volume_delta_ci95\":"<<volume.paired_difference.ci95
       <<",\"volume_samples\":"<<volume.samples<<",\"volume_ms\":"<<volume_ms<<",\"unresolved\":0"
       <<",\"audit_samples\":"<<audit_samples<<",\"max_box_violation\":"<<max_box_violation
       <<",\"min_sample_clearance\":";
    if(std::isfinite(min_clearance)) out<<min_clearance; else out<<"null";
    for(const auto& item:{std::make_pair("ellipse",&ellipse.metrics),std::make_pair("minkowski",&minkowski.metrics)}) {
      const auto& m=*item.second;
      out<<",\""<<item.first<<"_metrics\":{\"nodes\":"<<m.nodes<<",\"constraints\":"<<m.constraints
         <<",\"active_constraints\":"<<m.active_constraints<<",\"optimization_ms\":"<<m.optimization_ms
         <<",\"certificate_ms\":"<<m.certificate_ms<<",\"obstacle_pairs\":"<<m.obstacle_pairs
         <<",\"newton_steps\":"<<m.newton_steps<<",\"inflation_steps\":"<<m.inflation_steps
         <<",\"refinements\":"<<m.refinements<<"}";
    }
    out<<"}\n";
    std::ofstream surface(dir/"surface.csv"); surface<<std::setprecision(17)<<"u,angle,x,y,z\n";
    for(int k=0;k<=400;++k) for(int j=0;j<64;++j) {
      const double u=k/400.,angle=j*2*std::acos(-1.)/64;
      const auto p=minkowski.corridor.boundary(u,angle).point;
      surface<<u<<','<<angle<<','<<p[0]<<','<<p[1]<<','<<p[2]<<'\n';
    }
    if(!out || !surface) throw std::runtime_error("output failure");
    std::cout<<"ellipse "<<median(et)<<" ms / "<<volume.volume[0].mean<<" m3; Minkowski "<<median(mt)
             <<" ms / "<<volume.volume[1].mean<<" m3; delta CI95 "<<volume.paired_difference.ci95<<'\n';
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
