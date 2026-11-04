#include "common.hpp"
#include "../../../gcopter/tools/corridor_volume.hpp"
struct Other : corridor_volume::Patch {
  bool valid=true;size_t* unresolved=nullptr;
  bool contains(const sc::Vec3& p)const{return corridor_volume::Patch::contains(p,*unresolved);}
};
template<class T> std::vector<T> patches(const c3::Geometry& g) {
  std::vector<T> out(g.pieces());for(size_t i=0;i<out.size();++i){for(int j=0;j<4;++j){out[i].c[j]=g.controls[4*i+j].head<3>();out[i].m[j]=c3::shape(g.controls[4*i+j]);}out[i].prepare();}return out;
}
int main(int argc,char** argv){try{if(argc!=5)throw std::runtime_error("baseline candidate seed samples");
  const auto a=patches<corridor_volume::Patch>(readGeometry(argv[1]));auto b=patches<Other>(readGeometry(argv[2]));
  size_t unresolved=0;for(auto& p:b)p.unresolved=&unresolved;
  const auto r=corridor_volume::measure(a,b,readPoints(argv[3]),std::stoi(argv[4]));
  std::cout<<std::setprecision(17)<<"{\"baseline\":"<<r.volume[0].mean<<",\"candidate\":"<<r.volume[1].mean
    <<",\"paired_delta\":"<<r.paired_difference.mean<<",\"paired_ci95\":"<<r.paired_difference.ci95
    <<",\"samples\":"<<r.samples<<",\"unresolved\":"<<r.unresolved+unresolved<<"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
