#include <iostream>
#include <stdexcept>

#include "gcopter/direct_search_map.hpp"
#include "gcopter/sfc_gen.hpp"
using namespace gcopter::direct;
void require(bool x, const char* why) {
  if (!x) throw std::runtime_error(why);
}
int main() {
  try {
    ompl::RNG::setSeed(23);
    Bounds bounds{Vec3(-2, -2, -2), Vec3(2, 2, 2)};
    ObstacleIndex obstacles({{Vec3::Zero(), .3}});
    SearchMap map(obstacles, bounds, .1);
    require(!map.query({-1, 0, 0}) && !map.query({1, 0, 0}), "endpoints free");
    require(!map.segmentFree({-1, 0, 0}, {1, 0, 0}),
            "continuous edge catches between-endpoint obstacle");
    std::vector<Vec3> path;
    int checks = 0;
    double cost = sfc_gen::planPath(
        Vec3(-1, 0, 0), Vec3(1, 0, 0), bounds.lower, bounds.upper, &map, .2,
        path,
        [&](const Vec3& a, const Vec3& b) {
          ++checks;
          return map.segmentFree(a, b);
        },
        .6);
    require(std::isfinite(cost) && path.size() > 2 && checks > 0,
            "original RRTstar exact detour");
    require((path.front() - Vec3(-1, 0, 0)).norm() < 1e-9 &&
                (path.back() - Vec3(1, 0, 0)).norm() < 1e-9,
            "exact endpoints");
    for (std::size_t k = 1; k < path.size(); ++k)
      require(map.segmentFree(path[k - 1], path[k]),
              "all search edges solid-free");
    Bounds narrow{Vec3(-2, -.4, -.4), Vec3(2, .4, .4)};
    ObstacleIndex wall({{Vec3::Zero(), .7}});
    SearchMap blocked(wall, narrow, .05);
    cost = sfc_gen::planPath(
        Vec3(-1.5, 0, 0), Vec3(1.5, 0, 0), narrow.lower, narrow.upper, &blocked,
        .03, path,
        [&](const Vec3& a, const Vec3& b) { return blocked.segmentFree(a, b); },
        .4);
    require(!std::isfinite(cost) && path.empty(),
            "approximate solution rejected and previous path cleared");
    std::cout << "PASS: original RRTstar, solid edges, exact endpoints, "
                 "approximate rejection\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
