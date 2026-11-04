#include <SweepCorridor.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <sys/resource.h>
namespace sc=SweepCorridor;
namespace fs=std::filesystem;
#include "query_measurements.hpp"
std::vector<sc::Vec3> read(const char* path){std::ifstream f(path);if(!f)throw std::runtime_error("missing input");
  std::vector<sc::Vec3> points;std::string line;while(std::getline(f,line)){if(line.empty()||line[0]=='#')continue;
    std::istringstream row(line);sc::Vec3 p;if(!(row>>p.x()>>p.y()>>p.z()))throw std::runtime_error("invalid point");points.push_back(p);}return points;}
template<int D> void run(const std::vector<sc::Vec3>& seed,const sc::ObstacleIndex& map,const sc::Bounds& bounds,
    const sc::Options& options,int repeats,const fs::path& dir,std::ostream& log) {
  const std::string name=D==3?"C3-OPT-native":"H5-S";sc::Generator<D> generator(options);
  for(int repeat=-1;repeat<repeats;++repeat) {
    const auto start=std::chrono::steady_clock::now();auto r=generator.generate(seed,map,bounds);
    const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    log<<"{\"kind\":\"generation\",\"model\":\""<<name<<"\",\"repeat\":"<<repeat<<",\"status\":\""<<sc::statusName(r.status)
       <<"\",\"total_ms\":"<<r.metrics.total_ms<<",\"wall_ms\":"<<wall<<",\"newton_steps\":"<<r.metrics.iterations;
    if constexpr(D==3){const auto& m=r.metrics.solver;
      log<<",\"regularity_repairs\":"<<m.regularity_repairs<<",\"regularity_backtracks\":"<<m.regularity_backtracks
         <<",\"backtracks\":"<<m.backtracks<<",\"termination\":\""<<m.termination<<"\",\"predictor_steps\":"<<m.predictor_steps
         <<",\"predictor_rejections\":"<<m.predictor_rejections<<",\"volume_steps\":"<<m.volume_steps;}
    if(r){const auto certificate=sc::certify(r.corridor,r.support_planes);if(!certificate.safe())throw std::runtime_error("certificate failed");
      log<<",\"support_pass\":true,\"jacobian_pass\":true";
      if(repeat==0){std::ofstream out(dir/(name+".csv"));out<<std::setprecision(17)<<"degree,u0,u1,cx,cy,cz,m00,m10,m20,m01,m11,m21\n";
        for(std::size_t k=0;k<r.corridor.controls().size();++k){out<<D<<','<<r.corridor.knots()[k/(D+1)]<<','<<r.corridor.knots()[k/(D+1)+1];
          for(double v:r.corridor.controls()[k])out<<','<<v;out<<'\n';}
        std::ofstream planes(dir/(name+"_support_planes.csv"));planes<<std::setprecision(17)<<"span,begin,end,nx,ny,nz,bound\n";
        for(const auto& p:r.support_planes)planes<<p.span<<','<<p.begin<<','<<p.end<<','<<p.normal[0]<<','<<p.normal[1]<<','<<p.normal[2]<<','<<p.bound<<'\n';
      }}
    log<<"}\n";log.flush();
    if(r&&repeat==0&&std::getenv("SWEEP_QUERY_MEASURE"))queryMeasurements(r.corridor,log);
  }
}
int main(int argc,char** argv){try{if(argc!=12)throw std::runtime_error("seed map output xmin ymin zmin xmax ymax zmax repeats samples");
 auto seed=read(argv[1]),cloud=read(argv[2]);std::vector<sc::Sphere> spheres;for(auto p:cloud)spheres.push_back({p,std::sqrt(3)*.25/2});
 sc::ObstacleIndex map(std::move(spheres));sc::Bounds bounds{{std::stod(argv[4]),std::stod(argv[5]),std::stod(argv[6])},{std::stod(argv[7]),std::stod(argv[8]),std::stod(argv[9])}};
 sc::Options o;o.budget_ms=10000;double diagonal=0;sc::Vec3 a,b=seed.front();
 for(std::size_t i=1;i<seed.size();){a=b;if((a-seed[i]).norm()>7)b=a+(seed[i]-a).normalized()*7;else b=seed[i++];
   const sc::Vec3 lo=(a.cwiseMin(b)-sc::Vec3::Constant(3)).cwiseMax(bounds.lower+sc::Vec3::Constant(.252));
   const sc::Vec3 hi=(a.cwiseMax(b)+sc::Vec3::Constant(3)).cwiseMin(bounds.upper-sc::Vec3::Constant(.252));diagonal=std::max(diagonal,(hi-lo).norm());}
 o.max_radius=.5*diagonal+.001;o.max_offset=diagonal+.001;
 const fs::path dir=argv[3];fs::create_directories(dir);std::ofstream log(dir/"rows.jsonl");log<<std::setprecision(17);
 if(!std::getenv("NATIVE_SKIP_H5"))run<5>(seed,map,bounds,o,std::stoi(argv[10]),dir,log);
 if(std::getenv("NATIVE_CANDIDATES"))run<3>(seed,map,bounds,o,std::stoi(argv[10]),dir,log);
 rusage usage{};getrusage(RUSAGE_SELF,&usage);log<<"{\"kind\":\"resource\",\"max_rss_kib\":"<<usage.ru_maxrss<<"}\n";
 return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
