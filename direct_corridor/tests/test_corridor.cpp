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
#include "../src/hermite_optimizer.hpp"
using namespace gcopter::direct;
namespace {
void check(bool pass, const std::string& message) {
  if (!pass) throw std::runtime_error(message);
}
void success(const Result& r) {
  check(bool(r), std::string(statusName(r.status)) + ": " + r.message);
}
Bounds box() { return {Vec3(-30, -30, -30), Vec3(30, 30, 30)}; }
// Explicit historical fixture for the low-level Hermite regression cases.
Options legacyOptions() {
  Options o;
  o.frame_smoothing=0;o.section_smoothing=0;o.jet_smoothing=0;
  o.certify_jacobian=false;o.initial_iterations=12;
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
  {
    Options defaults;
    auto explicit_smooth=defaults;
    explicit_smooth.frame_smoothing=1.5;explicit_smooth.section_smoothing=.2;
    explicit_smooth.jet_smoothing=.01;explicit_smooth.certify_jacobian=true;
    explicit_smooth.initial_iterations=8;
    const std::vector<Vec3> seed{{-3,0,0},{0,.5,0},{3,1,0}};
    ObstacleIndex map({});
    const auto a=generate(seed,map,box(),defaults),b=generate(seed,map,box(),explicit_smooth);
    success(a);success(b);
    check(a.corridor.centerControls()==b.corridor.centerControls(),"H5 default smooth centers");
    check(a.corridor.shapeControls()==b.corridor.shapeControls(),"H5 default smooth shapes");
  }

  {
    const std::vector<Vec3> seed{{-3,0,0},{0,.5,0},{3,1,0}};
    ObstacleIndex map({});auto options=legacyOptions();
    const auto plain=generate(seed,map,box(),options);success(plain);
    options.profile=true;
    const auto profiled=generate(seed,map,box(),options);success(profiled);
    check(plain.corridor.centerControls()==profiled.corridor.centerControls(),"profile changes centers");
    check(plain.corridor.shapeControls()==profiled.corridor.shapeControls(),"profile changes shapes");
    check(plain.metrics.certificate_ms==0,"disabled stage timer");
    check(profiled.metrics.certificate_ms>0,"enabled stage timer");
  }

  try {
    // Query identity is independent of the changing section shapes. Check
    // cross-span sharing, round reuse and the memory-cap fallback explicitly.
    {
      ObstacleIndex map({{Vec3(0, 0, 0), .1}, {Vec3(2, 0, 0), .2}});
      detail::QueryCache cache;
      std::vector<int> scratch, expected;
      const Vec3 lo(-1, -1, -1), hi(1, 1, 1);
      map.query(lo, hi, expected);
      check(cache.query(0, lo, hi, map, scratch) == expected,
            "first cached query matches BVH");
      check(cache.query(1, lo, hi, map, scratch) == expected &&
                cache.ids.size() == 1 && cache.spans[0] == cache.spans[1],
            "identical boxes share candidate storage across spans");
      check(cache.query(0, lo, hi, map, scratch) == expected,
            "later inflation round reuses candidate indices");
      cache.bytes = detail::QueryCache::byte_limit;
      const Vec3 wider(3, 1, 1);
      map.query(lo, wider, expected);
      check(cache.query(2, lo, wider, map, scratch) == expected &&
                cache.ids.size() == 1 && cache.spans[2] == -1,
            "full cache falls back without omitting obstacles");
      check(cache.query(2, lo, wider, map, scratch) == expected,
            "uncached span remains correct in later rounds");
    }
    // A packet encloses sphere SUPPORT values, including unequal radii. Check
    // its lower/upper interval against every original sphere in many planes;
    // an incorrect bound could silently discard an entire obstacle packet.
    {
      std::mt19937 random(20261103);
      std::uniform_real_distribution<double> coordinate(-6, 6), radius(0, .8);
      for (int count : {1, 31, 32, 33, 97}) {
        for (double translation : {0., 1e7}) {
          std::vector<Sphere> spheres;
          for (int i = 0; i < count; ++i)
            spheres.push_back({Vec3(coordinate(random), coordinate(random), coordinate(random)) +
                                   Vec3::Constant(translation), radius(random)});
          ObstacleIndex map(std::move(spheres));
          detail::QueryCache cache;
          std::vector<int> scratch;
          const Vec3 lo = Vec3::Constant(translation - 8), hi = Vec3::Constant(translation + 8);
          const auto& ids = cache.query(0, lo, hi, map, scratch);
          const auto& groups = cache.groups[cache.spans[0]];
          check(int(ids.size()) == count, "packet query includes every sphere");
          int covered = 0;
          for (const auto& block : groups.blocks) {
            covered += block.end - block.begin;
            check(__builtin_popcountll(block.alive) == block.end - block.begin,
                  "packet mask covers its exact candidate range");
            for (int trial = 0; trial < 80; ++trial) {
              const Vec3 n = Vec3(coordinate(random), coordinate(random), coordinate(random)).normalized();
              const double middle = n.dot(block.center), spread = n.cwiseAbs().dot(block.half);
              const double lower = middle - spread - block.max_radius - .3;
              const double upper = middle + spread - block.min_radius - .3;
              const double tolerance = 64 * std::numeric_limits<double>::epsilon() *
                  (1 + std::abs(translation));
              for (int j = block.begin; j < block.end; ++j) {
                const auto& sphere = map.obstacles()[ids[j]];
                const double exact = n.dot(sphere.center) - sphere.radius - .3;
                check(lower <= exact + tolerance && exact <= upper + tolerance,
                      "packet interval encloses every sphere's supporting plane");
              }
            }
          }
          check(covered == count, "packets form a disjoint full partition");
          cache.query(1, lo, hi, map, scratch);
          check(cache.groups.size() == 1 && cache.spans[0] == cache.spans[1],
                "identical query boxes reuse immutable packet geometry");
          check(cache.bytes <= detail::QueryCache::byte_limit,
                "candidate indices and packet capacities share the memory cap");
        }
      }
    }
    // Independent bisection checks the sphere-metric KKT direction, including
    // point obstacles and isotropic metrics. Actual safety uses support planes.
    {
      std::mt19937 random(19);
      std::uniform_real_distribution<double> value(-2, 2), ratio(.05, .8);
      for (int k = 0; k < 200; ++k) {
        const Vec3 delta(value(random), value(random), value(random));
        const Vec3 axes(std::exp(value(random)), std::exp(value(random)),
                        std::exp(value(random)));
        const double radius = k == 0 ? 0 : ratio(random) * delta.norm();
        double lo = 0, hi = radius * axes.maxCoeff() / (delta.norm() - radius);
        for (int it = 0; it < 100; ++it) {
          const double t = .5 * (lo + hi);
          if ((t * delta.array() / (axes.array() + t)).square().sum() <
              radius * radius)
            lo = t;
          else
            hi = t;
        }
        const Vec3 expected = delta.cwiseQuotient(axes + Vec3::Constant(hi));
        const Vec3 actual = detail::metricSphereNormal(delta, axes, radius);
        check((actual.normalized() - expected.normalized()).norm() < 1e-7,
              "sphere metric normal matches independent bisection");
      }
      const Vec3 delta(1, 2, -3);
      check((detail::metricSphereNormal(delta, Vec3::Constant(.7), .4)
                 .normalized() - delta.normalized()).norm() < 1e-12,
            "isotropic metric gives radial sphere normal");
    }
    // Independent finite differences of the full barrier objective check the
    // closed-form logdet derivatives together with SOC and coupling terms.
    {
      Options options = legacyOptions();
      options.section_smoothing = .2;
      detail::Stop stop = [] {};
      detail::Optimizer optimizer(options, stop);
      std::vector<detail::Node> nodes(2);
      nodes[0].s = 0;
      nodes[1].s = .7;
      nodes[0].x << .03, -.04, .5, .07, .7;
      nodes[1].x << -.02, .05, .6, -.03, .4;
      const std::vector<detail::Cut> cuts{
          {0, Eigen::Vector2d(.3, .4), Eigen::Vector2d(.1, -.2), 1.2},
          {0, Eigen::Vector2d(-.2, .4), Eigen::Vector2d(.2, -.1), .9, true}};
      const std::vector<int> active{0, 1};
      optimizer.objective(nodes, cuts, active, .01, true);
      Eigen::Matrix<double, 10, 1> gradient;
      gradient << optimizer.gradient[0], optimizer.gradient[1];
      Eigen::Matrix<double, 10, 10> hessian;
      hessian << optimizer.diagonal[0], optimizer.upper[0],
                 optimizer.upper[0].transpose(), optimizer.diagonal[1];
      for (int j = 0; j < 10; ++j) {
        const double delta = 1e-6;
        auto plus = nodes, minus = nodes;
        plus[j / 5].x[j % 5] += delta;
        minus[j / 5].x[j % 5] -= delta;
        const double vp = optimizer.objective(plus, cuts, active, .01, true);
        Eigen::Matrix<double, 10, 1> gp;
        gp << optimizer.gradient[0], optimizer.gradient[1];
        const double vm = optimizer.objective(minus, cuts, active, .01, true);
        Eigen::Matrix<double, 10, 1> gm;
        gm << optimizer.gradient[0], optimizer.gradient[1];
        check(std::abs((vp - vm) / (2 * delta) - gradient[j]) < 1e-6,
              "optimizer gradient matches scalar finite difference");
        check(((gp - gm) / (2 * delta) - hessian.col(j)).norm() < 1e-5,
              "optimizer Hessian matches gradient finite difference");
      }
    }
    // Polar redundancy is invariant to translating the feasible center, and
    // removes loose faces while retaining every edge of a square plus SOC
    // rows coupling two different nodes.
    for (const Eigen::Vector2d offset :
         {Eigen::Vector2d(0, 0), Eigen::Vector2d(3, -2)}) {
      std::vector<detail::Node> nodes(2);
      for (auto& n : nodes) {
        n.x = detail::circle(.1);
        n.x.head<2>() = offset;
      }
      std::vector<detail::Cut> cuts;
      for (int j = 0; j < 4; ++j) {
        const Eigen::Vector2d a =
            (j < 2 ? Eigen::Vector2d::UnitX() : Eigen::Vector2d::UnitY()) *
            (j % 2 ? -1 : 1);
        cuts.push_back({0, a, Eigen::Vector2d::Zero(), 1 + a.dot(offset)});
      }
      for (int j = 0; j < 40; ++j) {
        const double angle = j * 2 * std::acos(-1.) / 40;
        const Eigen::Vector2d a(std::cos(angle), std::sin(angle));
        cuts.push_back({0, a, Eigen::Vector2d::Zero(), 2 + a.dot(offset)});
      }
      cuts.push_back({0, Eigen::Vector2d::UnitX(), Eigen::Vector2d::UnitY(),
                      4 + offset.sum()});
      const auto active = detail::irredundant(nodes, cuts);
      check(active == std::vector<int>({0, 1, 2, 3, 44}),
            "polar removes only redundant support faces");
      // Right-end-only constraints use the same translation, while rows very
      // close to the dual hull boundary are conservatively retained.
      cuts.push_back({0, Eigen::Vector2d(1, 0), Eigen::Vector2d::Zero(),
                      1 + 1e-14 + offset.x()});
      for (int j = 0; j < 44; ++j) {
        auto c = cuts[j];
        c.b = c.a;
        c.a.setZero();
        cuts.push_back(c);
      }
      const auto both = detail::irredundant(nodes, cuts);
      check(both == std::vector<int>({0, 1, 2, 3, 44, 45, 46, 47, 48, 49}),
            "polar keeps ambiguous boundary and handles right endpoints");
    }
    Options o = legacyOptions();
    ObstacleIndex empty({});
    // Positive center Jacobian alone misses a fold inside the disk. Here
    // J(u,z)=1+1.875*a*z0 at the span midpoint, while J(u,0)=1 everywhere.
    {
      std::array<Vec3,6> c;
      std::array<Mat32,6> m;
      for(double amplitude:{.4,.8}) {
        for(int j=0;j<6;++j) {
          c[j]=Vec3(0,0,j/5.);
          m[j].col(0)=Vec3(1,0,j<3?0:amplitude);
          m[j].col(1)=Vec3::UnitY();
        }
        check(detail::regularTube(c.data(),m.data())==(amplitude<.5),
              "full disk detects interior fold despite positive center Jacobian");
      }
    }
    {
      auto smooth=o;
      smooth.frame_smoothing=1.5;smooth.section_smoothing=.2;
      smooth.jet_smoothing=.01;smooth.certify_jacobian=true;smooth.initial_iterations=8;
      const std::vector<Vec3> path{{-5,0,0},{-2,0,0},{0,1.4,0},{2,1,1},{5,0,1}};
      const ObstacleIndex map({{Vec3(0,-1,.2),.5},{Vec3(1,3,.5),.5}});
      const auto r=generate(path,map,box(),smooth);
      sampleSafety(r,map,box(),smooth);
      derivatives(r.corridor);
      check((r.corridor.section(0).center-path.front()).norm()<1e-12 &&
            (r.corridor.section(1).center-path.back()).norm()<1e-12,"smooth endpoints preserved");
      bool nonzero=false;
      for(const auto& n:r.corridor.hermiteNodes()) nonzero|=n.state.shape_first.norm()>1e-3;
      check(nonzero,"shared shape jets remain free");
      for(size_t i=0;i<r.corridor.pieces();++i)
        check(detail::regularTube(r.corridor.centerControls().data()+6*i,
                                  r.corridor.shapeControls().data()+6*i),"smooth whole disk certificate");
      for(const auto& m:r.corridor.shapeControls()) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(m.transpose()*m);
        check(eig.eigenvalues().maxCoeff()<=smooth.max_radius*smooth.max_radius+1e-9,
              "smoothing and refill respect radius cap");
      }
      const auto full=generate({Vec3(-2,0,0),Vec3(2,0,0)},empty,box(),smooth);
      success(full);
      check(full.corridor.section(.5).shape.col(0).norm()>smooth.max_radius*(1-1e-5),
            "radius reserve is refilled in open space");
    }
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
    // Analytic empty rectangular tunnel with a 2000:1 axis ratio. Constraint
    // curvature has very different scales, but the large axis must still grow.
    Options slender = legacyOptions();
    slender.robot_radius = slender.safety_margin = 0;
    slender.numerical_guard = 1e-10;
    slender.certificate_margin = 1e-9;
    slender.min_radius = 1e-5;
    slender.max_radius = 2;
    const Bounds slit{Vec3(-3, -.001, -3), Vec3(3, .001, 3)};
    const auto scaled = generate({Vec3(-2,0,0), Vec3(2,0,0)}, empty, slit, slender);
    sampleSafety(scaled, empty, slit, slender);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> slitAxes(
        scaled.corridor.section(.5).shape.transpose() *
        scaled.corridor.section(.5).shape);
    check(slitAxes.eigenvalues()[1] > 1.8*1.8 &&
              slitAxes.eigenvalues()[0] > .0008*.0008,
          "equilibrated solver expands highly anisotropic section");
    // Local range is a world-axis box expansion, not an ellipse-radius cap.
    // Exercise a 7 m box transition inside a nonaligned Hermite interval.
    Options ranged = legacyOptions();
    ranged.max_radius = 10;
    ranged.max_offset = 20;
    ranged.local_range = 3;
    ranged.local_progress = 7;
    ranged.control_spacing = 2.3;
    auto expanded =
        generate({Vec3(0, 0, 0), Vec3(10, 0, 0)}, empty, box(), ranged);
    success(expanded);
    derivatives(expanded.corridor);
    check(expanded.corridor.section(.5).shape.col(0).norm() > 2.9,
          "local range removes the old 1.5 m cutoff");
    for (int k = 0; k <= 500; ++k) {
      const double u = k / 500.;
      const auto section = expanded.corridor.section(u);
      const Bounds local = u * 10 < 7 ? Bounds{Vec3(-3, -3, -3), Vec3(10, 3, 3)}
                                      : Bounds{Vec3(4, -3, -3), Vec3(13, 3, 3)};
      const Vec3 extent = section.shape.rowwise().norm();
      check((section.center - extent - local.lower).minCoeff() >= -1e-8 &&
                (local.upper - section.center - extent).minCoeff() >= -1e-8,
            "entire section stays in its local expansion box");
    }
    ranged.local_range = -1;
    check(
        generate({Vec3::Zero(), Vec3(10, 0, 0)}, empty, box(), ranged).status ==
            Status::InvalidInput,
        "negative local expansion rejected");
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
    // A farther sphere may protrude in front of a smaller near sphere. The
    // greedy separator must test its whole inflated radius before skipping it.
    std::vector<Sphere> varied;
    for (int i = -3; i <= 3; ++i)
      for (int sign : {-1, 1}) {
        varied.push_back({Vec3(i * .4, sign * 1.4, 0), .1});
        varied.push_back({Vec3(i * .4, sign * 2.8, 0), 1.8});
        varied.push_back({Vec3(i * .4, sign * 3.5, 0), .2});
      }
    ObstacleIndex differentRadii(varied);
    const auto occluded =
        generate({Vec3(-3, 0, 0), Vec3(3, 0, 0)}, differentRadii, box(), o);
    sampleSafety(occluded, differentRadii, box(), o);
    derivatives(occluded.corridor);
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
    o = legacyOptions();
    o.max_constraints = 1;
    check(generate({Vec3(-3, 0, 0), Vec3(3, 0, 0)}, wall, box(), o).status ==
              Status::Capacity,
          "constraint capacity");
    o = legacyOptions();
    o.min_radius = .5;
    auto tooNarrow = generate({Vec3(-3, 0, 0), Vec3(3, 0, 0)}, local, box(), o);
    check(!tooNarrow && tooNarrow.corridor.pieces() == 0,
          "positive minimum size is not bypassed");
    o = legacyOptions();
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
    o = legacyOptions();
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
    o = legacyOptions();
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
