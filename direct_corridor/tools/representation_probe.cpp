// SPDX-License-Identifier: MIT
// Offline representation/certificate experiment, NOT a corridor generator.
// Compile with the reference project's unmodified SplineTrajectory headers.
#include <Eigen/SVD>
#include <SplineConvexHull.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <gcopter/direct_corridor.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace gcopter::direct;
using Q = Eigen::Matrix<double, 9, 1>;
using Piece = std::vector<Q>;
using Curve = std::vector<Piece>;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
Q control(const Vec3& p) {
  Q q = Q::Zero();
  q.head<3>() = p;
  return q;
}
Mat32 shape(const Q& q) { return Eigen::Map<const Mat32>(q.data() + 3); }
std::vector<Q> read(const std::string& file, int columns) {
  std::ifstream input(file);
  if (!input) throw std::runtime_error("cannot open " + file);
  std::string line;
  std::vector<Q> data;
  while (std::getline(input, line)) {
    if (line.empty() || line.rfind("cx,", 0) == 0) continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    Q q = Q::Zero();
    for (int i = 0; i < columns; ++i)
      if (!(row >> q[i])) throw std::runtime_error("invalid row in " + file);
    if (!q.allFinite()) throw std::runtime_error("nonfinite input");
    data.push_back(q);
  }
  if (data.empty()) throw std::runtime_error("empty input");
  return data;
}
Q evaluate(Piece p, double t) {
  for (std::size_t n = p.size() - 1; n; --n)
    for (std::size_t i = 0; i < n; ++i) p[i] = (1 - t) * p[i] + t * p[i + 1];
  return p[0];
}
std::pair<Piece, Piece> split(Piece p) {
  Piece left(p.size()), right(p.size());
  left.front() = p.front();
  right.back() = p.back();
  for (std::size_t k = 1; k < p.size(); ++k) {
    for (std::size_t i = 0; i < p.size() - k; ++i)
      p[i] = .5 * (p[i] + p[i + 1]);
    left[k] = p[0];
    right[p.size() - 1 - k] = p[p.size() - 1 - k];
  }
  return {left, right};
}
Curve subdivide(Curve pieces, int depth) {
  for (int d = 0; d < depth; ++d) {
    Curve next;
    for (const auto& p : pieces) {
      const auto halves = split(p);
      next.push_back(halves.first);
      next.push_back(halves.second);
    }
    pieces = std::move(next);
  }
  return pieces;
}
// Derivatives are with respect to the COMMON path parameter, with span h.
Piece hermite(const Q& p0, const Q& v0, const Q& a0, const Q& p1, const Q& v1,
              const Q& a1, double h) {
  return {p0,
          p0 + h * v0 / 5.,
          p0 + 2 * h * v0 / 5. + h * h * a0 / 20.,
          p1 - 2 * h * v1 / 5. + h * h * a1 / 20.,
          p1 - h * v1 / 5.,
          p1};
}
Piece derivative(const Piece& p, double h) {
  Piece out;
  for (std::size_t i = 1; i < p.size(); ++i)
    out.push_back((p.size() - 1) * (p[i] - p[i - 1]) / h);
  return out;
}
double c2Error(const Curve& curve, const std::vector<double>& durations) {
  double error = 0;
  for (std::size_t i = 1; i < curve.size(); ++i) {
    Piece a = curve[i - 1], b = curve[i];
    for (int order = 0; order <= 2; ++order) {
      error = std::max(error, (a.back() - b.front()).norm());
      a = derivative(a, durations[i - 1]);
      b = derivative(b, durations[i]);
    }
  }
  return error;
}

struct Environment {
  ObstacleIndex index;
  Bounds bounds;
  double padding;
  // A sufficient lower bound on the clearance of the entire Bezier piece.
  // Each obstacle gets its own direction. Negative means inconclusive, not
  // necessarily collision. The cap keeps BVH queries local and remains valid.
  double clearanceBound(const Piece& p) const {
    constexpr double cap = 1.5;
    Vec3 lo = p[0].head<3>(), hi = lo, mean = Vec3::Zero();
    for (const auto& q : p) {
      lo = lo.cwiseMin(q.head<3>());
      hi = hi.cwiseMax(q.head<3>());
      mean += q.head<3>() / double(p.size());
    }
    double result = std::min({cap, (lo - bounds.lower).minCoeff() - padding,
                              (bounds.upper - hi).minCoeff() - padding});
    std::vector<int> ids;
    index.query(lo - Vec3::Constant(cap + padding),
                hi + Vec3::Constant(cap + padding), ids);
    for (int id : ids) {
      const auto& s = index.obstacles()[id];
      const Vec3 delta = mean - s.center;
      if (delta.norm() < 1e-14) return -s.radius - padding;
      const Vec3 n = delta.normalized();
      for (const auto& q : p)
        result = std::min(result,
                          n.dot(q.head<3>() - s.center) - s.radius - padding);
    }
    return result;
  }
  double pointClearance(const Vec3& p) const {
    double r = std::min({1.5, (p - bounds.lower).minCoeff() - padding,
                         (bounds.upper - p).minCoeff() - padding});
    std::vector<int> ids;
    index.query(p - Vec3::Constant(1.5 + padding),
                p + Vec3::Constant(1.5 + padding), ids);
    for (int id : ids) {
      const auto& s = index.obstacles()[id];
      r = std::min(r, (p - s.center).norm() - s.radius - padding);
    }
    return r;
  }
  // Certifies a constant multiplier of M throughout ONE interval; it does not
  // construct smooth joins with neighboring intervals or optimize any radii.
  double scaleBound(const Piece& p) const {
    constexpr double cap = 8.;
    Vec3 lo = p[0].head<3>(), hi = lo, mean = Vec3::Zero();
    double bound = cap, max_shape = 0;
    auto cut = [&](const Q& q, const Vec3& n, double gap) {
      const double support = (shape(q).transpose() * n).norm();
      if (gap < 0)
        bound = 0;
      else if (support > 1e-14)
        bound = std::min(bound, gap / support);
    };
    for (const auto& q : p) {
      lo = lo.cwiseMin(q.head<3>());
      hi = hi.cwiseMax(q.head<3>());
      mean += q.head<3>() / double(p.size());
      max_shape = std::max(max_shape, shape(q).norm());
      for (int axis = 0; axis < 3; ++axis) {
        cut(q, Vec3::Unit(axis), q[axis] - bounds.lower[axis] - padding);
        cut(q, Vec3::Unit(axis), bounds.upper[axis] - q[axis] - padding);
      }
    }
    const Vec3 expand = Vec3::Constant(cap * max_shape + padding);
    std::vector<int> ids;
    index.query(lo - expand, hi + expand, ids);
    for (int id : ids) {
      const auto& s = index.obstacles()[id];
      const Vec3 delta = mean - s.center;
      if (delta.norm() < 1e-14) return 0;
      const Vec3 n = delta.normalized();
      for (const auto& q : p)
        cut(q, n, n.dot(q.head<3>() - s.center) - s.radius - padding);
    }
    return bound;
  }
};

int main(int argc, char** argv) {
  try {
    if (argc != 12)
      throw std::runtime_error(
          "usage: probe map.xyz seed.xyz controls.csv voxel_resolution padding "
          "lx ly lz ux uy uz");
    auto map = read(argv[1], 3), seed = read(argv[2], 3),
         controls = read(argv[3], 9);
    if (seed.size() < 2 || controls.size() < 4)
      throw std::runtime_error("short input");
    const double voxel = std::stod(argv[4]), padding = std::stod(argv[5]);
    Bounds bounds{
        Vec3(std::stod(argv[6]), std::stod(argv[7]), std::stod(argv[8])),
        Vec3(std::stod(argv[9]), std::stod(argv[10]), std::stod(argv[11]))};
    if (!std::isfinite(voxel) || !std::isfinite(padding) || !(voxel > 0) ||
        !(padding >= 0) || !bounds.lower.allFinite() ||
        !bounds.upper.allFinite() ||
        (bounds.upper.array() <= bounds.lower.array()).any())
      throw std::runtime_error("invalid geometry");
    std::vector<Sphere> spheres;
    for (const auto& q : map)
      spheres.push_back({q.head<3>(), std::sqrt(3.) * voxel / 2});
    Environment env{ObstacleIndex(std::move(spheres)), bounds, padding};
    Curve cubic, equivalent;
    std::size_t bottleneck = 0;
    double min_axis = 1e9, identity = 0;
    for (std::size_t i = 0; i + 3 < controls.size(); ++i) {
      Piece b{(controls[i] + 4 * controls[i + 1] + controls[i + 2]) / 6.,
              (2 * controls[i + 1] + controls[i + 2]) / 3.,
              (controls[i + 1] + 2 * controls[i + 2]) / 3.,
              (controls[i + 1] + 4 * controls[i + 2] + controls[i + 3]) / 6.};
      cubic.push_back(b);
      equivalent.push_back(
          hermite(b[0], 3 * (b[1] - b[0]), 6 * (b[2] - 2 * b[1] + b[0]), b[3],
                  3 * (b[3] - b[2]), 6 * (b[3] - 2 * b[2] + b[1]), 1.));
      for (int sample = 0; sample <= 20; ++sample) {
        const double t = sample / 20.;
        const auto q = evaluate(b, t);
        identity =
            std::max(identity, (q - evaluate(equivalent.back(), t)).norm());
        const double minor = shape(q).jacobiSvd().singularValues()[1];
        if (minor < min_axis) {
          min_axis = minor;
          bottleneck = i;
        }
      }
    }
    if (identity > 1e-10)
      throw std::runtime_error("cubic/Hermite identity failed");
    double split_error = 0;
    const auto split_pieces = subdivide({cubic[bottleneck]}, 4);
    for (std::size_t i = 0; i < split_pieces.size(); ++i)
      for (double t : {0., .37, 1.})
        split_error = std::max(
            split_error,
            (evaluate(split_pieces[i], t) -
             evaluate(cubic[bottleneck], (i + t) / split_pieces.size()))
                .norm());
    if (split_error > 1e-10)
      throw std::runtime_error("subdivision identity failed");
    std::cout << std::setprecision(12);
    std::cout << "{\"kind\":\"identity\",\"max_error\":" << identity
              << ",\"subdivision_error\":" << split_error
              << ",\"bottleneck_span\":" << bottleneck
              << ",\"min_axis\":" << min_axis << "}\n";
    auto scale = [&](const std::string& method, const Curve& pieces,
                     int depth) {
      const auto begin = Clock::now();
      double value = 8.;
      const auto leaves = subdivide(pieces, depth);
      for (const auto& leaf : leaves)
        value = std::min(value, env.scaleBound(leaf));
      // Pull back by 1 ppm for the later independent dense audit.
      value *= (1. - 1e-6);
      double audit = 1.5;
      for (int i = 0; i <= 200; ++i) {
        const Q q = evaluate(cubic[bottleneck], i / 200.);
        for (int j = 0; j < 48; ++j) {
          const double angle = 2 * std::acos(-1.) * j / 48.;
          const Vec3 p = q.head<3>() +
                         value * shape(q) *
                             Eigen::Vector2d(std::cos(angle), std::sin(angle));
          audit = std::min(audit, env.pointClearance(p));
        }
      }
      if (audit < -1e-8) throw std::runtime_error("local scale audit failed");
      std::cout << "{\"kind\":\"local_scale\",\"method\":\"" << method
                << "\",\"depth\":" << depth << ",\"factor\":" << value
                << ",\"sampled_clearance\":" << audit
                << ",\"check_and_audit_ms\":" << ms(begin) << "}\n";
    };
    scale("raw_bspline_hull",
          {Piece(controls.begin() + bottleneck,
                 controls.begin() + bottleneck + 4)},
          0);
    for (int depth : {0, 1, 2, 3, 4}) {
      scale("cubic_bezier", {cubic[bottleneck]}, depth);
      scale("same_curve_quintic_hermite", {equivalent[bottleneck]}, depth);
    }

    std::vector<double> durations;
    SplineTrajectory::QuinticSpline3D::WaypointMatrix points(seed.size(), 3);
    for (std::size_t i = 0; i < seed.size(); ++i) {
      points.row(i) = seed[i].head<3>().transpose();
      if (i) {
        const double h = (seed[i] - seed[i - 1]).norm();
        if (h < 1e-8) throw std::runtime_error("duplicate seed point");
        durations.push_back(h);
      }
    }
    // Chord length as parameter span; these are not flight durations.
    SplineTrajectory::BoundaryConditions<3> boundary;
    boundary.start.v = (seed[1] - seed[0]).head<3>() / durations.front();
    boundary.end.v =
        (seed.back() - seed[seed.size() - 2]).head<3>() / durations.back();
    auto begin = Clock::now();
    SplineTrajectory::QuinticSpline3D minimum_jerk;
    if (!minimum_jerk.update(durations, points, 0., boundary))
      throw std::runtime_error("reference minimum-jerk solve failed");
    const auto hull = SplineTrajectory::toBezier(minimum_jerk.polynomial());
    Curve minimum, local;
    for (int i = 0; i < hull.numPieces(); ++i) {
      Piece p;
      for (int j = 0; j < 6; ++j)
        p.push_back(control(hull.pieceControls(i).row(j).transpose()));
      minimum.push_back(std::move(p));
    }
    const double minimum_ms = ms(begin);
    begin = Clock::now();
    std::vector<Q> velocities(seed.size());
    velocities.front() = (seed[1] - seed[0]) / durations.front();
    velocities.back() =
        (seed.back() - seed[seed.size() - 2]) / durations.back();
    for (std::size_t i = 1; i + 1 < seed.size(); ++i)
      velocities[i] =
          (seed[i + 1] - seed[i - 1]) / (durations[i - 1] + durations[i]);
    for (std::size_t i = 0; i + 1 < seed.size(); ++i)
      local.push_back(hermite(seed[i], velocities[i], Q::Zero(), seed[i + 1],
                              velocities[i + 1], Q::Zero(), durations[i]));
    const double local_ms = ms(begin);
    auto reference = [&](const std::string& method, const Curve& curve,
                         const std::vector<double>& spans, double fit_ms) {
      const double c2 = c2Error(curve, spans);
      if (c2 > 1e-8) throw std::runtime_error("C2 join check failed");
      double sampled = 1.5;
      for (const auto& p : curve)
        for (int i = 0; i <= 40; ++i)
          sampled = std::min(
              sampled, env.pointClearance(evaluate(p, i / 40.).head<3>()));
      for (int depth : {0, 1, 2, 3, 4}) {
        const auto start = Clock::now();
        const auto leaves = subdivide(curve, depth);
        double bound = 1.5;
        for (const auto& p : leaves)
          bound = std::min(bound, env.clearanceBound(p));
        if (bound > sampled + 1e-9)
          throw std::runtime_error("bound exceeds sampled clearance");
        std::cout << "{\"kind\":\"reference\",\"method\":\"" << method
                  << "\",\"depth\":" << depth << ",\"leaves\":" << leaves.size()
                  << ",\"clearance_bound\":" << bound
                  << ",\"sampled_clearance\":" << sampled
                  << ",\"c2_error\":" << c2 << ",\"fit_ms\":" << fit_ms
                  << ",\"check_ms\":" << ms(start) << "}\n";
      }
    };
    reference("current_cubic", cubic, std::vector<double>(cubic.size(), 1.),
              0.);
    reference("minimum_jerk_quintic", minimum, durations, minimum_ms);
    reference("local_quintic_hermite", local, durations, local_ms);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
