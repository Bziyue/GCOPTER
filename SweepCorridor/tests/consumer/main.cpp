#include <SweepCorridor/InverseQuery.hpp>
#include <iostream>
int main() {
  namespace sc=SweepCorridor;
  sc::ObstacleIndex map({});
  sc::Bounds bounds{sc::Vec3::Constant(-10),sc::Vec3::Constant(10)};
  std::vector<sc::Vec3> seed{{-3,0,0},{0,.5,0},{3,1,0}};
  sc::Options options;
  auto result=sc::Generator<3>(options).generate(seed,map,bounds);
  if(!result){std::cerr<<sc::statusName(result.status)<<'\n';return 1;}
  sc::Query<3> query(result.corridor);
  const auto section=query.section(.5);
  const auto constraint=query.evaluateFirst(.5,section.point({.2,.3}));
  sc::InverseQuery<3> inverse(result.corridor);
  const auto residual=inverse(section.point({1.2,.3}),.5);
  if(!residual.valid||residual.radial<=0)return 2;
  std::cout<<result.corridor.pieces()<<" spans; q="<<constraint.value
           <<"; plane residual="<<constraint.plane_residual<<'\n';
}
