// SPDX-License-Identifier: MIT
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "gcopter/direct_corridor.hpp"
using namespace gcopter::direct;
namespace {
void check(bool pass, const std::string& message) {
  if (!pass) throw std::runtime_error(message);
}
void success(const Result& r) {
  check(bool(r), std::string(statusName(r.status)) + ": " + r.message);
}
Bounds box() { return {Vec3(-30, -30, -30), Vec3(30, 30, 30)}; }
Options opts() {
  Options o;
  o.robot_radius = .05;
  o.safety_margin = .01;
  o.budget_ms = 10000;
  return o;
}
void sampleSafety(const Result& r, const ObstacleIndex& map,
                  const Bounds& bounds, const Options& o) {
  success(r);
  const double pad = o.robot_radius + o.safety_margin;
  for (int k = 0; k <= 501; ++k) {
    const auto s = r.corridor.section(k / 501.);
    for (int a = 0; a < 24; ++a)
      for (double radius : {0., .5, 1.}) {
        const double angle = a * 2 * std::acos(-1.) / 24;
        const Vec3 p =
            s.point(radius * Eigen::Vector2d(std::cos(angle), std::sin(angle)));
        check((p.array() >= bounds.lower.array() + pad - 1e-8).all() &&
                  (p.array() <= bounds.upper.array() - pad + 1e-8).all(),
              "sample outside eroded map");
        for (const auto& obstacle : map.obstacles())
          check((p - obstacle.center).norm() > obstacle.radius + pad - 1e-8,
                "independent sample hits obstacle");
      }
  }
}
void derivatives(const Corridor& c) {
  for (double u : {.137, .361, .719}) {
    const auto s = c.section(u);
    const Vec3 p = s.point({.3, -.2}) + Vec3(.03, -.02, .01);
    const auto value = c.evaluate(u, p);
    const double h = 1e-6;
    for (int k = 0; k < 4; ++k) {
      Vec3 plus = p, minus = p;
      double up = u, um = u;
      if (k == 0) {
        up += h;
        um -= h;
      } else {
        plus[k - 1] += h;
        minus[k - 1] -= h;
      }
      const auto dp = c.evaluate(up, plus), dm = c.evaluate(um, minus);
      check(std::abs((dp.value - dm.value) / (2 * h) - value.gradient[k]) <
                2e-4 * (1 + std::abs(value.gradient[k])),
            "gradient finite difference");
      check(((dp.gradient - dm.gradient) / (2 * h) - value.hessian.col(k))
                    .norm() < 2e-3 * (1 + value.hessian.col(k).norm()),
            "Hessian finite difference");
    }
  }
  for (std::size_t k = 1; k < c.pieces(); ++k) {
    const double left = c.knots()[k] - c.knots()[k - 1],
                 right = c.knots()[k + 1] - c.knots()[k];
    const auto& p = c.centerControls();
    const auto& m = c.shapeControls();
    const std::size_t i = 6 * k;
    check((p[i - 1] - p[i]).norm() < 1e-12, "C0 centre join");
    check((5 * (p[i - 1] - p[i - 2]) / left - 5 * (p[i + 1] - p[i]) / right)
                  .norm() < 1e-8,
          "C1 centre join");
    check((20 * (p[i - 1] - 2 * p[i - 2] + p[i - 3]) / (left * left) -
           20 * (p[i + 2] - 2 * p[i + 1] + p[i]) / (right * right))
                  .norm() < 1e-6,
          "C2 centre join");
    check((m[i - 1] - m[i]).norm() < 1e-12, "C0 shape join");
    check((5 * (m[i - 1] - m[i - 2]) / left - 5 * (m[i + 1] - m[i]) / right)
                  .norm() < 1e-10,
          "C1 shape join");
    check((20 * (m[i - 1] - 2 * m[i - 2] + m[i - 3]) / (left * left) -
           20 * (m[i + 2] - 2 * m[i + 1] + m[i]) / (right * right))
                  .norm() < 1e-8,
          "C2 shape join");
  }
}
}  // namespace
int main() {
  try {
    Options o = opts();
    ObstacleIndex empty({});
    auto straight = generate({Vec3(-2, 0, 0), Vec3(2, 0, 0)}, empty, box(), o);
    success(straight);
    check((straight.corridor.section(0).center - Vec3(-2, 0, 0)).norm() < 1e-12,
          "start preserved");
    check((straight.corridor.section(1).center - Vec3(2, 0, 0)).norm() < 1e-12,
          "goal preserved");
    check(std::abs(straight.corridor.section(.5).shape.col(0).norm() -
                   o.max_radius) < 1e-7,
          "open-space maximum radius");
    derivatives(straight.corridor);
    auto denseSeed =
        generate({Vec3(-2, 0, 0), Vec3(-2, 0, 0), Vec3(0, 0, 0), Vec3(2, 0, 0)},
                 empty, box(), o);
    success(denseSeed);
    // A multi-turn, non-monotone, varying-height seed exercises transport and
    // joins.
    std::vector<Vec3> route = {{-5, -3, -1}, {-2, -3, 0},  {-2, 1, 1},
                               {1, 1, 2},    {1, -2, 2.5}, {4, -2, 3}};
    auto curved = generate(route, empty, box(), o);
    success(curved);
    derivatives(curved.corridor);
    std::reverse(route.begin(), route.end());
    auto reverse = generate(route, empty, box(), o);
    success(reverse);
    check((reverse.corridor.section(0).center - route.front()).norm() < 1e-12,
          "reverse endpoint");
    auto vertical = generate({Vec3(0, 0, -3), Vec3(0, 0, 3)}, empty, box(), o);
    success(vertical);
    derivatives(vertical.corridor);
    // Two acute turns separated by a short edge used to invalidate the
    // length-weighted reference tangent despite a collision-free seed.
    const double turn = 70 * std::acos(-1.) / 180;
    Vec3 cornerA(1, 0, 0),
        cornerB = cornerA + .08 * Vec3(std::cos(turn), std::sin(turn), 0);
    auto shortEdge =
        generate({Vec3::Zero(), cornerA, cornerB,
                  cornerB + Vec3(std::cos(2 * turn), std::sin(2 * turn), 0)},
                 empty, box(), o);
    success(shortEdge);
    derivatives(shortEdge.corridor);
    // Rotated geometry must not depend on a special x/z path assumption.
    Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(.7, Vec3(1, 2, 3).normalized()).toRotationMatrix();
    for (auto& p : route) p = rotation * p;
    success(generate(route, empty, box(), o));
    std::vector<Sphere> obstacles;
    for (int i = -20; i <= 20; ++i)
      for (double sign : {-1., 1.})
        for (int z = -15; z <= 15; ++z)
          obstacles.push_back({Vec3(i * .2, sign * .7, z * .2), .11});
    ObstacleIndex wall(obstacles);
    auto tunnel = generate({Vec3(-3, 0, 0), Vec3(3, 0, 0)}, wall, box(), o);
    sampleSafety(tunnel, wall, box(), o);
    derivatives(tunnel.corridor);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(
        tunnel.corridor.section(.5).shape.transpose() *
        tunnel.corridor.section(.5).shape);
    check(eig.eigenvalues()[1] > 4 * eig.eigenvalues()[0],
          "anisotropic passage retains tall ellipse");
    // Regressions for the two freedoms missing from the original generator.
    std::vector<Sphere> oneSide, rotatedWalls;
    const Eigen::Matrix3d roll =
        Eigen::AngleAxisd(.55, Vec3::UnitX()).toRotationMatrix();
    for (const auto& obstacle : obstacles) {
      if (obstacle.center.y() > 0)
        oneSide.push_back({obstacle.center - Vec3(0, .3, 0), obstacle.radius});
      rotatedWalls.push_back({roll * obstacle.center, obstacle.radius});
    }
    ObstacleIndex asymmetric(oneSide), oblique(rotatedWalls);
    const auto shifted =
        generate({Vec3(-3, 0, 0), Vec3(3, 0, 0)}, asymmetric, box(), o);
    sampleSafety(shifted, asymmetric, box(), o);
    const auto shiftedSection = shifted.corridor.section(.5);
    check(shiftedSection.center.y() < -.8,
          "center moves into available free space");
    check((shiftedSection.shape.transpose() * shiftedSection.shape)
                  .determinant() > 2.,
          "off-center corridor recovers wide free space");
    const auto oriented =
        generate({Vec3(-3, 0, 0), Vec3(3, 0, 0)}, oblique, box(), o);
    sampleSafety(oriented, oblique, box(), o);
    const Mat32 rotatedShape = oriented.corridor.section(.5).shape;
    check(std::abs((rotatedShape * rotatedShape.transpose())(1, 2)) > .4,
          "ellipse principal axes rotate to match oblique walls");
    // A local bottleneck must not shrink distant open sections through global
    // coefficients.
    ObstacleIndex local({{Vec3(0, .45, 0), .1}, {Vec3(0, -.45, 0), .1}});
    auto bottle = generate({Vec3(-8, 0, 0), Vec3(8, 0, 0)}, local, box(), o);
    sampleSafety(bottle, local, box(), o);
    const auto far = bottle.corridor.section(.1).shape,
               near = bottle.corridor.section(.5).shape;
    check((far.transpose() * far).determinant() >
              4 * (near.transpose() * near).determinant(),
          "local support retains distant area");
    // Near-boundary extrapolated controls trigger refinement, not clipped
    // endpoints.
    Bounds narrow{Vec3(-1, -1, -1), Vec3(1, 1, 1)};
    o.max_radius = .25;
    auto edge = generate({Vec3(-.9, 0, 0), Vec3(.9, 0, 0)}, empty, narrow, o);
    sampleSafety(edge, empty, narrow, o);
    check((edge.corridor.section(0).center - Vec3(-.9, 0, 0)).norm() < 1e-12,
          "Hermite endpoint does not require extrapolated controls");
    // Exact backtracking cannot be advertised as a regular parametric corridor.
    auto cusp = generate({Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(-1, 0, 0)}, empty,
                         box(), o);
    check(!cusp && cusp.status == Status::DegeneratePath &&
              cusp.corridor.pieces() == 0,
          "cusp rejected");
    ObstacleIndex blocking({{Vec3(0, 0, 0), .2}});
    auto blocked =
        generate({Vec3(-1, 0, 0), Vec3(1, 0, 0)}, blocking, box(), o);
    check(blocked.status == Status::SeedCollision &&
              blocked.corridor.pieces() == 0,
          "between-vertex collision rejected");
    auto invalid = generate(
        {Vec3::Zero(), Vec3(std::numeric_limits<double>::quiet_NaN(), 0, 0)},
        empty, box(), o);
    check(invalid.status == Status::InvalidInput, "NaN rejected");
    check(generate({Vec3::Zero(), Vec3::Zero()}, empty, box(), o).status ==
              Status::DegeneratePath,
          "duplicate-only path");
    o.cancelled = [] { return true; };
    check(generate(route, wall, box(), o).status == Status::Cancelled,
          "cancellation");
    o.cancelled = {};
    o.budget_ms = 1e-10;
    check(generate(route, wall, box(), o).status == Status::Timeout,
          "deadline");
    o.budget_ms = 10000;
    o.max_controls = 4;
    check(generate(route, wall, box(), o).status == Status::Capacity,
          "capacity");
    o.max_controls = 20000;
    o = opts();
    o.max_constraints = 1;
    check(generate({Vec3(-3, 0, 0), Vec3(3, 0, 0)}, wall, box(), o).status ==
              Status::Capacity,
          "constraint capacity");
    o = opts();
    o.min_radius = .5;
    auto tooNarrow = generate({Vec3(-3, 0, 0), Vec3(3, 0, 0)}, local, box(), o);
    check(!tooNarrow && tooNarrow.corridor.pieces() == 0,
          "positive minimum size is not bypassed");
    o = opts();
    o.max_radius = -1;
    check(generate(route, empty, box(), o).status == Status::InvalidInput,
          "invalid options");
    bool rejected = false;
    try {
      ObstacleIndex bad({{Vec3::Zero(), -1}});
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check(rejected, "invalid solid radius");
    // BVH must return the same candidate set as a brute-force AABB oracle.
    std::mt19937 rng(73);
    std::uniform_real_distribution<double> rnd(-10, 10);
    std::vector<Sphere> random;
    for (int i = 0; i < 1000; ++i)
      random.push_back({Vec3(rnd(rng), rnd(rng), rnd(rng)), .3});
    ObstacleIndex index(random);
    for (int k = 0; k < 100; ++k) {
      Vec3 lo(rnd(rng), rnd(rng), rnd(rng)), hi = lo + Vec3::Constant(2);
      std::vector<int> ids, truth;
      index.query(lo, hi, ids);
      for (int i = 0; i < 1000; ++i)
        if ((random[i].center.array() - .3 <= hi.array()).all() &&
            (random[i].center.array() + .3 >= lo.array()).all())
          truth.push_back(i);
      std::sort(ids.begin(), ids.end());
      check(ids == truth, "BVH versus brute force");
    }
    // Slender safe polyline whose coarse rounded control hull approaches an
    // obstacle.
    o = opts();
    o.control_spacing = 1.;
    o.max_radius = .3;
    ObstacleIndex corner({{Vec3(.65, .35, 0), .1}});
    auto bend = generate({Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0)}, corner,
                         box(), o);
    sampleSafety(bend, corner, box(), o);
    // Long path is not restricted to one global polynomial or one travel
    // direction.
    std::vector<Vec3> longPath;
    for (int k = 0; k < 1001; ++k)
      longPath.emplace_back(k * .03, 2 * std::sin(k * .01), std::sin(k * .007));
    Bounds longBox{Vec3(-5, -10, -10), Vec3(40, 10, 10)};
    auto longResult = generate(longPath, empty, longBox, o);
    success(longResult);
    check((longResult.corridor.section(1).center - longPath.back()).norm() <
              1e-10,
          "dense long seed endpoint retained after resampling");
    // Independent random cross-section oracle; generated obstacles are not used
    // to tune expectations, and any returned success must exclude every input
    // sphere.
    o = opts();
    o.max_radius = .7;
    int randomSuccesses = 0;
    for (int trial = 0; trial < 24; ++trial) {
      std::vector<Vec3> wave;
      for (int k = 0; k <= 12; ++k)
        wave.emplace_back(-3 + k * .5, .3 * std::sin(k * .5 + trial),
                          .2 * std::cos(k * .4));
      std::vector<Sphere> randomObstacles;
      for (int k = 0; k < 100; ++k) {
        Vec3 p(rnd(rng) * .35, rnd(rng) * .3, rnd(rng) * .2);
        if (std::hypot(p.y(), p.z()) > .7)
          randomObstacles.push_back({p, .05 + .03 * (k % 4)});
      }
      ObstacleIndex randomMap(randomObstacles);
      auto r = generate(wave, randomMap, box(), o);
      if (r) {
        ++randomSuccesses;
        sampleSafety(r, randomMap, box(), o);
      } else
        check(r.corridor.pieces() == 0, "random failure has no output");
    }
    check(randomSuccesses >= 20, "random known-clear path success rate");
    // q alone must not silently count axial displacement as membership.
    auto s = straight.corridor.section(.4);
    auto axial = straight.corridor.evaluate(.4, s.center + Vec3(1, 0, 0));
    check(axial.value < 0 && std::abs(axial.plane_residual) > .9,
          "plane membership contract");
    std::cout << "PASS: derivatives, C2 joins, exact endpoints, "
                 "reverse/vertical/rotated/long paths, "
                 "anisotropy, locality, continuous certificates plus "
                 "independent samples, BVH, failures, budgets\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
