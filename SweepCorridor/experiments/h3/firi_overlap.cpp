// SPDX-License-Identifier: MIT
#include "../free_boundary/common.hpp"
#include "../../../gcopter/tools/corridor_benchmark_common.hpp"

// Compare several exported Hermite corridors to one RViz FIRI result, including
// its normal solid-sphere erosion, local boxes and shortcut. All QMC and file
// operations are outside the generator timings.
int main(int argc,char** argv) {
  try {
    if(argc<12) throw std::runtime_error(
        "seed map xmin ymin zmin xmax ymax zmax resolution repeats samples corridor.csv ...");
    const auto seed=readPoints(argv[1]), cloud=readPoints(argv[2]);
    if(seed.size()<2) throw std::invalid_argument("seed requires at least two points");
    const sc::Bounds bounds{{std::stod(argv[3]),std::stod(argv[4]),std::stod(argv[5])},
                            {std::stod(argv[6]),std::stod(argv[7]),std::stod(argv[8])}};
    const double resolution=std::stod(argv[9]);
    const int repeats=std::stoi(argv[10]), samples=std::stoi(argv[11]);
    if(!(resolution>0) || !std::isfinite(resolution) || repeats<1 || samples<1)
      throw std::invalid_argument("invalid resolution or repeat/sample count");
    const auto o=optionsFor(seed,bounds);
    FiriResult firi;
    std::vector<double> times;
    for(int rep=-1;rep<repeats;++rep) {
      firi=generateFiri(seed,cloud,bounds,std::sqrt(3.)*.5*resolution,
                        o.robot_radius+o.safety_margin,true,o.local_range,
                        o.local_progress,o.certificate_margin);
      if(rep>=0) times.push_back(firi.total_ms);
    }
    std::vector<corridor_volume::Poly> polys;
    for(const auto& h:firi.polys) polys.emplace_back(h);
    bool connected=!polys.empty() && polys.front().contains(seed.front()) &&
                   polys.back().contains(seed.back());
    for(size_t i=0;i<polys.size();++i) {
      connected=connected && polys[i].valid;
      if(i && connected) {
        Eigen::MatrixX4d both(polys[i-1].h.rows()+polys[i].h.rows(),4);
        both<<polys[i-1].h,polys[i].h;
        connected=corridor_volume::Poly(std::move(both)).valid;
      }
    }
    std::cout<<std::setprecision(17)<<"{\"firi_ms\":";
    list(std::cout,times);
    std::cout<<",\"connected\":"<<std::boolalpha<<connected<<",\"comparisons\":[";
    for(int i=12;i<argc;++i) {
      const auto patches=corridor_volume::patches(readGeometry(argv[i]).controls);
      // Sampling just the grid cells containing FIRI fixes the domain and
      // Sobol points across all candidates. The intersection is unchanged.
      const auto r=corridor_volume::measure(patches,polys,seed,samples,8,1.,o.local_range,true);
      if(r.unresolved) throw std::runtime_error("unresolved cubic membership");
      if(i>12) std::cout<<',';
      std::cout<<"{\"firi\":"<<r.volume[1].mean
               <<",\"intersection\":"<<r.volume[4].mean
               <<",\"intersection_ci95\":"<<r.volume[4].ci95<<",\"intersection_shifts\":";
      list(std::cout,r.intersection_shifts);
      std::cout<<'}';
    }
    std::cout<<"]}\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
