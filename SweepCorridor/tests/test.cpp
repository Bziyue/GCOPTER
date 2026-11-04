#include <SweepCorridor.hpp>
#include <cassert>
#include <iostream>
int odr();
void banded();
void queries(const SweepCorridor::Corridor&);
void failures(){
 namespace sc=SweepCorridor;sc::ObstacleIndex empty({});
 sc::Bounds b{sc::Vec3::Constant(-10),sc::Vec3::Constant(10)};
 auto invalid=sc::Generator().generate({sc::Vec3::Zero()},empty,b);
 assert(invalid.status==sc::Status::InvalidInput&&invalid.corridor.empty());
 sc::ObstacleIndex blocked({{sc::Vec3::Zero(),.4}});
 auto collision=sc::Generator().generate({{-2,0,0},{2,0,0}},blocked,b);
 assert(collision.status==sc::Status::SeedCollision&&collision.corridor.empty());
 sc::Options options;options.budget_ms=-1;
 auto budget=sc::Generator(options).generate({{-2,0,0},{2,0,0}},empty,b);
 assert(budget.status==sc::Status::InvalidInput&&budget.corridor.empty());
}
int main(){banded();failures();namespace sc=SweepCorridor;sc::ObstacleIndex map({});sc::Bounds b{sc::Vec3::Constant(-10),sc::Vec3::Constant(10)};
  std::vector<sc::Vec3> seed{{-3,0,0},{0,.5,0},{3,1,0}};
  auto a=sc::Generator().generate(seed,map,b);
  std::cout<<sc::statusName(a.status)<<'\n';assert(a);assert(odr()==3);queries(a.corridor);
  assert(sc::certify(a.corridor,a.support_planes).safe());
}
