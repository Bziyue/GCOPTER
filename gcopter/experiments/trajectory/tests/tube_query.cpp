#include "../tube_query.hpp"
#include <iostream>
#include <stdexcept>

void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){
  ObstacleIndex obstacles({});Bounds bounds{Vec3(-8,-8,-4),Vec3(8,8,8)};
  // A narrow, regular tube is required for a meaningful local implicit-gradient
  // check. Wide curved tubes can switch inverse branches under perturbation.
  Options options;options.max_radius=0.2;options.max_offset=0.1;options.budget_ms=10000;
  auto result=generate({Vec3(-3,0,1),Vec3(-1,1,1.5),Vec3(1,0,2),Vec3(3,1,2)},obstacles,bounds,options);
  require(bool(result),"generate curved test corridor");
  PreparedCorridor prepared(result.corridor);size_t cursor=0;
  tube_query::Query fast(result.corridor,true),reference(result.corridor,false);
  double section_error=0,query_error=0,gradient_error=0;
  std::vector<double> parameters=result.corridor.knots();
  for(int k=0;k<=1000;++k)parameters.push_back(k/1000.0);
  for(double u:parameters){
    auto a=prepared.firstUnchecked(u,cursor);auto b=result.corridor.section(u);
    section_error=std::max({section_error,(a.center-b.center).norm(),(a.first-b.first).norm(),(a.shape-b.shape).norm(),(a.shape_first-b.shape_first).norm()});
    for(double radius:{0.5,1.2}){
      const Vec3 p=b.point(Eigen::Vector2d(radius*0.8,radius*0.6));
      const double hint=std::clamp(u+0.015,0.0,1.0);
      const auto x=fast(p,hint),y=reference(p,hint);
      query_error=std::max({query_error,std::abs(x.radial-y.radial),std::abs(x.plane-y.plane),std::abs(x.u-y.u),(x.grad_radial-y.grad_radial).norm(),(x.grad_plane-y.grad_plane).norm()});
      require(x.valid==y.valid,"root validity unchanged");
      const auto again=fast(p,hint);
      require(again.radial==x.radial&&(again.grad_radial.array()==x.grad_radial.array()).all(),"exact memoization");
      if(radius>1&&u>0.1&&u<0.9){
        Vec3 finite;
        // The unchanged root stopping tolerance is 1e-8: use a larger difference
        // step so root stopping error does not dominate this spatial check.
        for(int j=0;j<3;++j){Vec3 plus=p,minus=p;plus(j)+=1e-4;minus(j)-=1e-4;
          finite(j)=(fast(plus,hint).radial-fast(minus,hint).radial)/2e-4;}
        gradient_error=std::max(gradient_error,(finite-x.grad_radial).norm());
      }
    }
  }
  std::cout<<"section "<<section_error<<", inverse "<<query_error<<", gradient "<<gradient_error<<std::endl;
  require(section_error<1e-8,"prepared section equivalence");
  require(query_error<1e-7,"reference/fast inverse equivalence");
  require(gradient_error<1e-4,"implicit spatial gradient finite difference");
  require(fast.stats.query_hits>0&&fast.stats.section_hits>0,"caches exercised");
  std::cout<<"section "<<section_error<<", inverse "<<query_error<<", gradient "<<gradient_error
           <<", query hits "<<fast.stats.query_hits<<", section hits "<<fast.stats.section_hits<<'\n';
}
