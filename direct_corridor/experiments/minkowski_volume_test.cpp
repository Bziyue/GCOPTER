// SPDX-License-Identifier: MIT
#include "minkowski_volume.hpp"
#include <iostream>
using namespace gcopter::direct;
namespace mk=gcopter::direct::minkowski;
namespace cv=corridor_volume;
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
int main() {
  try {
    for(int mode=0;mode<3;++mode) {
      mk::Options o;
      o.geometry.robot_radius=0; o.geometry.safety_margin=0; o.geometry.certificate_margin=1e-8;
      o.geometry.max_radius=5; o.geometry.max_offset=3;
      o.affine=mode==2; o.thinness=mode==1?.15:.05;
      const ObstacleIndex empty({});
      const auto result=mk::generate({Vec3(-2,0,0),Vec3(2,0,0)},empty,{Vec3(-5,-1,-1),Vec3(5,1,1)},o);
      require(bool(result),result.message.c_str());
      const auto& c=result.corridor;
      // Independent oriented polygon integral of the analytic support boundary.
      double area=0;
      const auto frame=c.section(.5).frame;
      const Vec3 normal=frame.col(0).cross(frame.col(1)).normalized();
      const Vec3 center=c.section(.5).center;
      Vec3 previous=c.boundary(.5,0).point-center;
      for(int j=1;j<=8192;++j) {
        const Vec3 point=c.boundary(.5,2*std::acos(-1.)*j/8192).point-center;
        area+=previous.cross(point).dot(normal)/2;
        previous=point;
      }
      require(std::abs(area-c.area(.5))<2e-5,"mixed area vs independent polygon integral");
      std::size_t unresolved=0;
      auto p=minkowski_volume::patches(c,unresolved);
      require(p.front().contains(c.section(.01).center),"Minkowski volume interior");
      for(const auto& patch:p) require(!patch.contains(Vec3(2.1,0,0)),"Minkowski end cap");
      auto duplicates=p; duplicates.insert(duplicates.end(),p.begin(),p.end());
      const auto volume=cv::measure({},duplicates,{Vec3(-2,0,0),Vec3(2,0,0)},16384);
      require(std::abs(volume.volume[1].mean-4*area)<.035,"straight sweep union volume, duplicates excluded");
      require(unresolved==0 && volume.unresolved==0,"resolved plane roots");
      std::cout<<"mode "<<mode<<" analytic="<<4*area<<" measured="<<volume.volume[1].mean<<'\n';
    }
    std::cout<<"PASS: Minkowski area, union membership, duplicates and end caps\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
