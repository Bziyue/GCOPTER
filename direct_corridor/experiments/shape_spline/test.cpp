#include "refit.hpp"
#include <iostream>
#include <stdexcept>
using namespace gcopter::direct;
int main(){
  try{
    std::vector<Vec3> seed{{0, 0, 1}, {2, .1, 1}, {4, 0, 1}, {6, 1, 1}};
    ObstacleIndex map({{Vec3(3, 1, 1), .3}, {Vec3(1, -1.3, 1), .25}});
    Bounds bounds{Vec3(-1, -3, 0), Vec3(8, 4, 3)};
    Options o;
    o.control_spacing = .7;
    o.max_radius = 1.2;
    o.max_offset = .5;
    o.robot_radius = .1;
    o.safety_margin = .02;
    o.budget_ms = 10000;
    auto base = generate(seed, map, bounds, o);
    if (!base){
      throw std::runtime_error(base.message);
    }
    for (int d:{5, 3}){
      auto r = experiment::refitShape(base.corridor, seed, map, bounds, o, d, true);
      if (r.corridor.knots() != base.corridor.knots() ||
          r.corridor.centerControls() != base.corridor.centerControls()) {
        throw std::runtime_error("shape-only fit changed the reference center");
      }
      if (d == 5 && r.metrics.fit_error > 1e-9) {
        throw std::runtime_error("quintic basis does not reproduce the baseline");
      }
      if (r.metrics.c2_error > 1e-6||r.metrics.max_support_residual > 1e-9||
          r.metrics.max_knot_first < 1e-5||r.metrics.max_knot_second < 1e-5){
        throw std::runtime_error("C2/nonzero derivative/support regression");
      }
      for (int k = 0; k <= 1000; ++k){
        auto s = r.corridor.section(k / 1000.);
        for (int j = 0; j < 32; ++j){
          double t = j * 2 * std::acos(-1.) / 32;
          Vec3 p = s.point({std::cos(t), std::sin(t)});
          if ((p.array() < bounds.lower.array() + .12 - 1e-7).any()||
              (p.array() > bounds.upper.array() - .12 + 1e-7).any()){
            throw std::runtime_error("independent map bound");
          }
          for (const auto& obstacle:map.obstacles()){
            if ((p - obstacle.center).norm() < obstacle.radius + .12 - 1e-7){
              throw std::runtime_error("independent sphere audit");
            }
          }
        }
      }
      if (d == 3){
        const auto& m = r.corridor.shapeControls();
        for (size_t i = 0; i < r.corridor.pieces(); ++i){
          for (int j = 0; j < 2; ++j){
            if ((m[6 * i + j + 4] - 4 * m[6 * i + j + 3] + 6 * m[6 * i + j + 2] - 4 *
                 m[6 * i + j + 1] + m[6 * i + j]).norm() > 1e-9){
              throw std::runtime_error("cubic elevation is not cubic");
            }
          }
        }
      }
      std::cout << d << " degree: grad=" << r.metrics.gradient_error << " Hessian=" <<
        r.metrics.hessian_error << " C2=" << r.metrics.c2_error << " first=" <<
        r.metrics.max_knot_first << " second=" << r.metrics.max_knot_second << " cost_ms=" <<
      r.metrics.total_ms << '\n';
    }
  }
  catch (const std::exception& e){
    std::cerr << e.what() << '\n';
    return 1;
  }
}
