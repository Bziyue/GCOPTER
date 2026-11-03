#include "refit.hpp"
#include "../../src/hermite_optimizer.hpp"
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
    for (int mode : {5, 3, 0, -1, -2, -6, -8, -112, -212}) {
      int d = mode <= 0 ? 5 : mode;
      auto reference = base.corridor;
      if (mode == -1) {
        auto one = o; one.inflation_iterations = 1;
        auto result = generate(seed, map, bounds, one);
        if (!result) throw std::runtime_error(result.message);
        reference = std::move(result.corridor);
      }
      experiment::ShapeFitResult r;
      if (mode == 0) {
        auto integrated = experiment::generateIntegratedShape(seed, map, bounds, o, r.metrics, true);
        if (!integrated) throw std::runtime_error(integrated.message);
        r.corridor = std::move(integrated.corridor);
        if (r.metrics.reused_planes == 0 || r.metrics.planes_ms != 0 ||
            !r.metrics.direct_initialization)
          throw std::runtime_error("integrated stage did not reuse its cover");
      } else if (mode == -112 || mode == -212) {
        auto warm = detail::generateWithShapeStage(seed, map, bounds, o, {}, 12, true, mode == -212);
        if (!warm) throw std::runtime_error(warm.message);
        reference = std::move(warm.corridor);
        auto fast = experiment::generateCoordinateShape(seed, map, bounds, o, r.metrics, 12, true, true, mode == -212);
        if (!fast) throw std::runtime_error(fast.message);
        r.corridor = std::move(fast.corridor);
      } else if (mode < -2) {
        auto warm = detail::generateWithShapeStage(seed, map, bounds, o, {}, -mode);
        if (!warm) throw std::runtime_error(warm.message);
        reference = std::move(warm.corridor);
        auto fast = experiment::generateCoordinateShape(seed, map, bounds, o, r.metrics, -mode, true);
        if (!fast) throw std::runtime_error(fast.message);
        r.corridor = std::move(fast.corridor);
      } else if (mode < 0) {
        auto fast = experiment::generateFastShape(seed, map, bounds, o, r.metrics, -mode, true);
        if (!fast) throw std::runtime_error(fast.message);
        r.corridor = std::move(fast.corridor);
      } else {
        r = experiment::refitShape(base.corridor, seed, map, bounds, o, d, true);
      }
      if (r.corridor.knots() != reference.knots() ||
          r.corridor.centerControls() != reference.centerControls()) {
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
      std::cout << (mode <= 0 ? "integrated " : "offline ") << d << " degree: grad=" << r.metrics.gradient_error << " Hessian=" <<
        r.metrics.hessian_error << " C2=" << r.metrics.c2_error << " first=" <<
        r.metrics.max_knot_first << " second=" << r.metrics.max_knot_second << " cost_ms=" <<
      r.metrics.total_ms << '\n';
    }
    // An analytic cone/SPD boundary step stays feasible, and crossing its
    // undamped boundary exits the same objective's domain.
    detail::Stop no_stop = [] {};
    detail::Optimizer optimizer(o, no_stop);
    std::vector<detail::Node> nodes(2);
    for (int i = 0; i < 2; ++i) { nodes[i].s = i; nodes[i].x = detail::circle(.5); }
    optimizer.step.resize(2);
    optimizer.step[0] << 2, -1, .6, -.7, -.4;
    optimizer.step[1] << -1, 2, -.3, .2, .8;
    std::vector<detail::Cut> cuts{{0, { .5, .1 }, { .2, .3 }, 1.1, false}};
    std::vector<int> active{0};
    const double alpha = optimizer.feasibleStep(nodes, cuts, active);
    auto trial = nodes;
    for (int i = 0; i < 2; ++i) trial[i].x += alpha * optimizer.step[i];
    if (!(alpha > 0 && alpha < 1) || !std::isfinite(optimizer.objective(trial, cuts, active, .01, false)))
      throw std::runtime_error("analytic boundary step is not feasible");
    for (int i = 0; i < 2; ++i) trial[i].x = nodes[i].x + alpha / .995 * 1.000001 * optimizer.step[i];
    if (std::isfinite(optimizer.objective(trial, cuts, active, .01, false)))
      throw std::runtime_error("analytic boundary step missed its limiting constraint");
    // The integrated stage shares the generator's cancellation boundary.
    int calls = 0;
    o.cancelled = [&] { ++calls; return false; };
    if (!generate(seed, map, bounds, o)) throw std::runtime_error("cancellation fixture");
    const int baseline_calls = calls;
    calls = 0;
    o.cancelled = [&] { return ++calls > baseline_calls; };
    experiment::ShapeFitMetrics metrics;
    auto cancelled = experiment::generateIntegratedShape(seed, map, bounds, o, metrics);
    if (cancelled.status != Status::Cancelled || cancelled.corridor.pieces() != 0)
      throw std::runtime_error("integrated stage ignored cancellation");
    o.cancelled = {};
    auto invalid = experiment::generateIntegratedShape({}, map, bounds, o, metrics);
    if (invalid.status != Status::InvalidInput)
      throw std::runtime_error("integrated stage bypassed entry validation");
  }
  catch (const std::exception& e){
    std::cerr << e.what() << '\n';
    return 1;
  }
}
