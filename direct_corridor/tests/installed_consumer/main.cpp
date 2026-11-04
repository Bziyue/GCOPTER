#include <gcopter/cubic/cubic_generator.hpp>
#include <type_traits>
#include <iostream>
using namespace gcopter::direct;
static_assert(!std::is_convertible_v<Options,native::CubicOptions>);
template<class T,class=void> struct has_jets:std::false_type{};
template<class T> struct has_jets<T,std::void_t<decltype(T::jet_smoothing)>>:std::true_type{};
static_assert(has_jets<Options>::value);
static_assert(!has_jets<native::CubicOptions>::value);
int main() {
  GeometryOptions geometry;geometry.local_range=3;geometry.frame_smoothing=1.5;
  geometry.budget_ms=10000;
  const std::vector<Vec3> seed{{-3,0,0},{0,.5,0},{3,1,0}};
  ObstacleIndex map({});Bounds bounds{Vec3::Constant(-10),Vec3::Constant(10)};
  const auto c3=native::generateNative(seed,map,bounds,native::CubicOptions(geometry));
  const auto h5=generate(seed,map,bounds,Options(geometry));
  std::cout<<"C3 "<<c3.status<<", H5 "<<statusName(h5.status)<<'\n';
  return c3.status=="SUCCESS"&&h5?0:1;
}
