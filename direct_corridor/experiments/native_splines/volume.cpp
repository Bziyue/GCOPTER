// SPDX-License-Identifier: MIT
// Re-measure frozen geometry without repeating/timing corridor generation.
#include "geometry.hpp"
#include "corridor_volume.hpp"
#include <filesystem>
#include <iostream>
#include <sstream>
using namespace gcopter::direct;
namespace ns=gcopter::direct::native;
struct OtherPatch:corridor_volume::Patch {
  bool valid=true,active=true;size_t* unresolved=nullptr;
  bool contains(const Vec3& p)const{return active&&corridor_volume::Patch::contains(p,*unresolved);}
};
int main(int argc,char**argv){try{
  if(argc!=4 && argc!=5)throw std::runtime_error("seed geometry_directory samples_per_shift [model_names_comma_separated]");
  std::ifstream file(argv[1]);std::vector<Vec3> seed;std::string line;
  while(std::getline(file,line)){if(line.empty()||line[0]=='#')continue;Vec3 p;std::istringstream row(line);if(!(row>>p[0]>>p[1]>>p[2]))throw std::runtime_error("invalid seed");seed.push_back(p);}
  const std::filesystem::path directory=argv[2];const int samples=std::stoi(argv[3]);
  std::vector<std::string> names{"H5-S","C3-I","Q5-U","H5-C3-T","Q5-T","H5-Z-fixed","C3-OPT-native","C3-OPT-project","C3-NBC-project"};
  if(argc==5){names.clear();std::istringstream input(argv[4]);std::string name;while(std::getline(input,name,','))names.push_back(name);}
  std::vector<std::vector<corridor_volume::Patch>> patches;std::vector<corridor_volume::Box> boxes;
  for(const auto& name:names){patches.push_back(corridor_volume::patches(ns::Geometry::read((directory/(name+".csv")).string()).quinticAdapter()));for(const auto& p:patches.back())boxes.push_back(p.box);}
  std::ofstream out(directory/"volume_refined.jsonl");out<<std::setprecision(17);
  for(size_t i=1;i<names.size();++i) {
    size_t unresolved=0;std::vector<OtherPatch> other;
    for(const auto& p:patches[i]){OtherPatch a;static_cast<corridor_volume::Patch&>(a)=p;a.unresolved=&unresolved;other.push_back(a);}
    for(const auto& box:boxes){OtherPatch a;a.box=box;a.active=false;other.push_back(a);}
    const auto v=corridor_volume::measure(patches[0],other,seed,samples);
    out<<"{\"kind\":\"volume\",\"model\":\""<<names[i]<<"\",\"baseline\":"<<v.volume[0].mean
       <<",\"volume\":"<<v.volume[1].mean<<",\"paired_delta\":"<<v.paired_difference.mean
       <<",\"paired_ci95\":"<<v.paired_difference.ci95<<",\"samples\":"<<v.samples<<",\"cells\":"<<v.cells
       <<",\"unresolved\":"<<v.unresolved+unresolved<<"}\n";
  }
  std::cout<<directory<<": volume refined to 8 x "<<samples<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
