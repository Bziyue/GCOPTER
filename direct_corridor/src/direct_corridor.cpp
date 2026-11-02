// SPDX-License-Identifier: MIT
#include "gcopter/direct_corridor.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "hermite_optimizer.hpp"
namespace gcopter::direct {
namespace {
using detail::Failure;
Vec3 segmentClosest(const Vec3& p, const Vec3& a, const Vec3& b) {
  const Vec3 d = b - a;
  const double t = d.squaredNorm() > 1e-24
                       ? std::clamp((p - a).dot(d) / d.squaredNorm(), 0., 1.)
                       : 0.;
  return a + t * d;
}
double bernstein(int n, int k, double t) {
  return detail::choose(n, k) * std::pow(t, k) * std::pow(1 - t, n - k);
}
}  // namespace
ObstacleIndex::ObstacleIndex(std::vector<Sphere> obstacles)
    : obstacles_(std::move(obstacles)) {
  if (obstacles_.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max() / 2))
    throw std::invalid_argument("too many obstacles");
  for (const auto& s : obstacles_)
    if (!s.center.allFinite() || !std::isfinite(s.radius) || s.radius < 0 ||
        !(s.center.array().abs() + s.radius).isFinite().all())
      throw std::invalid_argument("invalid obstacle sphere");
  order_.resize(obstacles_.size());
  std::iota(order_.begin(), order_.end(), 0);
  nodes_.reserve(obstacles_.size() / 4 + 1);
  if (!order_.empty()) build(0, static_cast<int>(order_.size()));
}
int ObstacleIndex::build(int begin, int end) {
  Node n;
  n.begin = begin;
  n.end = end;
  n.lower = Vec3::Constant(std::numeric_limits<double>::infinity());
  n.upper = -n.lower;
  for (int i = begin; i < end; ++i) {
    const auto& s = obstacles_[order_[i]];
    n.lower = n.lower.cwiseMin(s.center - Vec3::Constant(s.radius));
    n.upper = n.upper.cwiseMax(s.center + Vec3::Constant(s.radius));
  }
  const int id = static_cast<int>(nodes_.size());
  nodes_.push_back(n);
  if (end - begin > 8) {
    Eigen::Index axis;
    (n.upper - n.lower).maxCoeff(&axis);
    int mid = begin + (end - begin) / 2;
    std::nth_element(order_.begin() + begin, order_.begin() + mid,
                     order_.begin() + end, [&](int a, int b) {
                       const double x = obstacles_[a].center[axis],
                                    y = obstacles_[b].center[axis];
                       return x == y ? a < b : x < y;
                     });
    int left = build(begin, mid), right = build(mid, end);
    nodes_[id].left = left;
    nodes_[id].right = right;
  }
  return id;
}
void ObstacleIndex::query(const Vec3& lower, const Vec3& upper,
                          std::vector<int>& ids) const {
  ids.clear();
  if (nodes_.empty()) return;
  // Balanced median BVH over at most INT_MAX/2 leaves needs <64 stack entries.
  std::array<int, 64> stack{};
  int size = 1;
  stack[0] = 0;
  while (size) {
    const auto& n = nodes_[stack[--size]];
    if ((n.lower.array() > upper.array()).any() ||
        (n.upper.array() < lower.array()).any())
      continue;
    if (n.left < 0) {
      for (int k = n.begin; k < n.end; ++k) {
        int id = order_[k];
        const auto& s = obstacles_[id];
        if ((s.center.array() - s.radius <= upper.array()).all() &&
            (s.center.array() + s.radius >= lower.array()).all())
          ids.push_back(id);
      }
    } else {
      stack[size++] = n.left;
      stack[size++] = n.right;
    }
  }
}

Section Corridor::section(double u) const {
  if (!std::isfinite(u) || u < 0 || u > 1 || pieces() == 0)
    throw std::invalid_argument("invalid corridor parameter");
  const auto at = std::upper_bound(knots_.begin(), knots_.end(), u);
  const std::size_t i =
      std::min(pieces() - 1, std::size_t(at - knots_.begin() - 1));
  const double h = knots_[i + 1] - knots_[i], t = (u - knots_[i]) / h;
  Section s;
  s.center.setZero();
  s.first.setZero();
  s.second.setZero();
  s.shape.setZero();
  s.shape_first.setZero();
  s.shape_second.setZero();
  for (int j = 0; j < 6; ++j) {
    const auto k = 6 * i + j;
    double b = bernstein(5, j, t);
    s.center += b * centers_[k];
    s.shape += b * shapes_[k];
    if (j < 5) {
      double d = 5 / h * bernstein(4, j, t);
      s.first += d * (centers_[k + 1] - centers_[k]);
      s.shape_first += d * (shapes_[k + 1] - shapes_[k]);
    }
    if (j < 4) {
      double d = 20 / (h * h) * bernstein(3, j, t);
      s.second += d * (centers_[k + 2] - 2 * centers_[k + 1] + centers_[k]);
      s.shape_second += d * (shapes_[k + 2] - 2 * shapes_[k + 1] + shapes_[k]);
    }
  }
  return s;
}
Differential Corridor::evaluate(double u, const Vec3& x) const {
  if (!x.allFinite()) throw std::invalid_argument("nonfinite evaluation point");
  const auto s = section(u);
  const auto& m = s.shape;
  const auto& d = s.shape_first;
  const auto& dd = s.shape_second;
  const Eigen::Matrix2d g = m.transpose() * m;
  const Eigen::Matrix2d gd = d.transpose() * m + m.transpose() * d;
  const Eigen::Matrix2d gdd =
      dd.transpose() * m + 2 * d.transpose() * d + m.transpose() * dd;
  const Eigen::Matrix2d inverse = g.inverse();
  const Eigen::Matrix2d id = -inverse * gd * inverse;
  const Eigen::Matrix2d idd =
      2 * inverse * gd * inverse * gd * inverse - inverse * gdd * inverse;
  const Eigen::Matrix<double, 2, 3> p = inverse * m.transpose();
  const Eigen::Matrix<double, 2, 3> pd =
      id * m.transpose() + inverse * d.transpose();
  const Eigen::Matrix<double, 2, 3> pdd =
      idd * m.transpose() + 2 * id * d.transpose() + inverse * dd.transpose();
  const Vec3 r = x - s.center;
  const Eigen::Vector2d z = p * r, zu = pd * r - p * s.first;
  const Eigen::Vector2d zuu = pdd * r - 2 * pd * s.first - p * s.second;
  Differential out;
  out.value = z.squaredNorm() - 1;
  out.gradient[0] = 2 * z.dot(zu);
  out.gradient.tail<3>() = 2 * p.transpose() * z;
  out.hessian(0, 0) = 2 * (zu.squaredNorm() + z.dot(zuu));
  out.hessian.block<3, 1>(1, 0) = 2 * (pd.transpose() * z + p.transpose() * zu);
  out.hessian.block<1, 3>(0, 1) = out.hessian.block<3, 1>(1, 0).transpose();
  out.hessian.bottomRightCorner<3, 3>() = 2 * p.transpose() * p;
  out.plane_residual = r.dot(m.col(0).cross(m.col(1)).normalized());
  return out;
}
struct GeneratorAccess {
  static Corridor make(const std::vector<detail::Node>& nodes) {
    Corridor c;
    const double total = nodes.back().s;
    for (const auto& n : nodes) {
      HermiteNode h;
      h.u = n.s / total;
      h.state.center = detail::center(n);
      h.state.first = total * n.velocity;
      h.state.second.setZero();
      h.state.shape = detail::shape(n);
      h.state.shape_first.setZero();
      h.state.shape_second.setZero();
      c.nodes_.push_back(h);
      c.knots_.push_back(h.u);
    }
    c.knots_.back() = 1.;
    c.nodes_.back().u = 1.;
    for (int i = 0; i + 1 < int(nodes.size()); ++i) {
      std::array<Vec3, 6> cp;
      std::array<Mat32, 6> mp;
      detail::controls(detail::patch(nodes, i), nodes, cp, mp);
      c.centers_.insert(c.centers_.end(), cp.begin(), cp.end());
      c.shapes_.insert(c.shapes_.end(), mp.begin(), mp.end());
    }
    return c;
  }
};
const char* statusName(Status s) {
  switch (s) {
    case Status::Success:
      return "SUCCESS";
    case Status::InvalidInput:
      return "INVALID_INPUT";
    case Status::SeedCollision:
      return "SEED_COLLISION";
    case Status::InsufficientClearance:
      return "INSUFFICIENT_CLEARANCE";
    case Status::DegeneratePath:
      return "DEGENERATE_PATH";
    case Status::Capacity:
      return "CAPACITY";
    case Status::Timeout:
      return "TIMEOUT";
    case Status::Cancelled:
      return "CANCELLED";
    case Status::VerificationFailed:
      return "VERIFICATION_FAILED";
  }
  return "UNKNOWN";
}

Result generate(const std::vector<Vec3>& seed, const ObstacleIndex& index,
                const Bounds& bounds, const Options& o) {
  using namespace detail;
  Result result;
  const auto begin = std::chrono::steady_clock::now();
  auto elapsed = [&] {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - begin)
        .count();
  };
  const Stop stop = [&] {
    if (o.cancelled && o.cancelled())
      throw Failure{Status::Cancelled, "request cancelled"};
    if (elapsed() >= o.budget_ms)
      throw Failure{Status::Timeout, "generation budget exhausted"};
  };
  try {
    for (double v : {o.control_spacing, o.min_radius, o.max_radius,
                     o.max_offset, o.robot_radius, o.safety_margin,
                     o.numerical_guard, o.certificate_margin, o.budget_ms})
      if (!std::isfinite(v))
        throw Failure{Status::InvalidInput, "nonfinite generator option"};
    if (seed.size() < 2 || !bounds.lower.allFinite() ||
        !bounds.upper.allFinite() ||
        (bounds.upper.array() <= bounds.lower.array()).any() ||
        o.control_spacing <= 0 || o.min_radius <= 0 ||
        o.max_radius <= o.min_radius || o.max_offset <= 0 ||
        o.robot_radius < 0 || o.safety_margin < 0 || o.numerical_guard <= 0 ||
        o.certificate_margin < o.numerical_guard || o.budget_ms <= 0 ||
        o.max_refinements < 0 || o.max_refinements > 20 || o.max_controls < 2 ||
        o.max_controls > 1000000 || o.max_constraints < 1 ||
        o.inflation_iterations < 1 || o.inflation_iterations > 5 ||
        o.certificate_depth < 1 || o.certificate_depth > 16)
      throw Failure{Status::InvalidInput,
                    "invalid seed, geometry or Hermite options"};
    stop();
    const double pad = o.robot_radius + o.safety_margin;
    if (!std::isfinite(pad + o.max_radius + o.max_offset +
                       o.certificate_margin))
      throw Failure{Status::InvalidInput, "geometry scale overflows"};
    std::vector<Vec3> clean;
    std::vector<double> arc;
    for (const auto& p : seed) {
      if (!p.allFinite()) throw Failure{Status::InvalidInput, "nonfinite seed"};
      if ((p.array() <= bounds.lower.array() + pad + o.numerical_guard).any() ||
          (p.array() >= bounds.upper.array() - pad - o.numerical_guard).any())
        throw Failure{Status::SeedCollision,
                      "seed touches eroded map boundary"};
      if (clean.empty() || (p - clean.back()).norm() > 1e-10) {
        const double length = clean.empty() ? 0 : (p - clean.back()).norm();
        arc.push_back(arc.empty() ? 0 : arc.back() + length);
        clean.push_back(p);
      }
      if (clean.size() > o.max_controls)
        throw Failure{Status::Capacity, "seed exceeds node capacity"};
    }
    if (clean.size() < 2)
      throw Failure{Status::DegeneratePath, "seed has no length"};
    for (std::size_t i = 1; i < clean.size(); ++i) {
      stop();
      std::vector<int> ids;
      const Vec3 e = Vec3::Constant(pad + o.numerical_guard);
      index.query(clean[i - 1].cwiseMin(clean[i]) - e,
                  clean[i - 1].cwiseMax(clean[i]) + e, ids);
      for (int id : ids) {
        const auto& s = index.obstacles()[id];
        if ((segmentClosest(s.center, clean[i - 1], clean[i]) - s.center)
                .norm() <= s.radius + pad + o.numerical_guard)
          throw Failure{Status::SeedCollision,
                        "seed segment intersects an inflated sphere"};
      }
      if (i + 1 < clean.size() &&
          (clean[i] - clean[i - 1])
                  .normalized()
                  .dot((clean[i + 1] - clean[i]).normalized()) < -.999999)
        throw Failure{Status::DegeneratePath, "seed reverses through a cusp"};
    }
    Failure last{Status::InsufficientClearance,
                 "no certifiable Hermite reference"};
    for (int refinement = 0; refinement <= o.max_refinements; ++refinement) {
      stop();
      result.metrics.refinements = refinement;
      try {
        const double spacing = std::ldexp(o.control_spacing, -refinement),
                     total = arc.back();
        const double count = std::ceil(total / spacing);
        if (!std::isfinite(count) || count > o.max_controls)
          throw Failure{Status::Capacity, "Hermite node capacity exceeded"};
        std::vector<std::pair<double, bool>> samples{{0, true}, {total, true}};
        for (int k = 1; k < int(count); ++k)
          samples.push_back({k * total / count, false});
        for (std::size_t i = 1; i + 1 < clean.size(); ++i)
          if ((clean[i] - clean[i - 1])
                  .normalized()
                  .dot((clean[i + 1] - clean[i]).normalized()) < std::cos(.2))
            samples.push_back({arc[i], true});
        std::sort(samples.begin(), samples.end());
        std::vector<std::pair<double, bool>> selected;
        for (auto sample : samples) {
          if (!selected.empty() &&
              sample.first - selected.back().first < spacing * .12) {
            if (!sample.second) continue;
            if (!selected.back().second) {
              selected.back() = sample;
              continue;
            }
            if (sample.first - selected.back().first < 1e-8) continue;
          }
          selected.push_back(sample);
        }
        if (selected.size() > o.max_controls)
          throw Failure{Status::Capacity, "Hermite node capacity exceeded"};
        std::vector<Node> nodes(selected.size());
        const double initial_radius =
            o.min_radius +
            std::min(.1 * o.min_radius, .1 * (o.max_radius - o.min_radius));
        for (std::size_t i = 0; i < nodes.size(); ++i) {
          auto& n = nodes[i];
          n.s = selected[i].first;
          const auto at = std::upper_bound(arc.begin(), arc.end(), n.s);
          const auto k =
              std::min(clean.size() - 2, std::size_t(at - arc.begin() - 1));
          n.reference = clean[k] + (n.s - arc[k]) / (arc[k + 1] - arc[k]) *
                                       (clean[k + 1] - clean[k]);
          n.x = circle(initial_radius);
        }
        Vec3 previous = Vec3::Zero(), axis = Vec3::Zero();
        for (std::size_t i = 0; i < nodes.size(); ++i) {
          auto& n = nodes[i];
          // Unit-edge bisector stays forward on both sides of a short/sharp
          // junction; a length-weighted secant can overshoot that junction.
          if (i == 0)
            n.velocity = (nodes[1].reference - n.reference).normalized();
          else if (i + 1 == nodes.size())
            n.velocity = (n.reference - nodes[i - 1].reference).normalized();
          else
            n.velocity =
                .5 * ((n.reference - nodes[i - 1].reference).normalized() +
                      (nodes[i + 1].reference - n.reference).normalized());
          if (n.velocity.norm() < 1e-8)
            throw Failure{Status::DegeneratePath,
                          "Hermite reference has a stationary tangent"};
          const Vec3 tangent = n.velocity.normalized();
          if (i == 0)
            axis = tangent.unitOrthogonal();
          else {
            if (previous.dot(tangent) < -.999999)
              throw Failure{Status::DegeneratePath,
                            "antiparallel Hermite frames"};
            axis = Eigen::Quaterniond::FromTwoVectors(previous, tangent) * axis;
            axis -= tangent * tangent.dot(axis);
            axis.normalize();
          }
          n.frame.col(0) = axis;
          n.frame.col(1) = tangent.cross(axis);
          previous = tangent;
        }
        std::vector<Node> best;
        for (int iteration = 0; iteration < o.inflation_iterations;
             ++iteration) {
          if (iteration && elapsed() > .5 * o.budget_ms) break;
          try {
            double timer = elapsed();
            Certificates certificates(index, bounds, o, stop, result.metrics);
            certificates.build(nodes);
            result.metrics.certificate_ms += elapsed() - timer;
            for (const auto& c : certificates.cuts)
              if (value(c, nodes) >= c.h - 1e-11)
                throw Failure{Status::InsufficientClearance,
                              "initial Hermite control hull needs refinement"};
            auto full = nodes;
            for (auto& n : full) n.x = circle(o.max_radius);
            bool unconstrained = true;
            for (const auto& c : certificates.cuts)
              if (value(c, full) >= c.h - 1e-9) {
                unconstrained = false;
                break;
              }
            if (unconstrained)
              nodes = std::move(full);
            else {
              timer = elapsed();
              Optimizer optimizer(o, stop);
              optimizer.optimize(nodes, certificates.cuts, result.metrics);
              result.metrics.optimization_ms += elapsed() - timer;
            }
            // Full SOC residual audit, covering every obstacle certificate.
            for (const auto& c : certificates.cuts)
              if (value(c, nodes) > c.h + o.numerical_guard * .01)
                throw Failure{Status::VerificationFailed,
                              "full Hermite support residual failed"};
            for (int i = 0; i + 1 < int(nodes.size()); ++i)
              if (!regular(patch(nodes, i), nodes))
                throw Failure{Status::DegeneratePath,
                              "Hermite tube transversality certificate failed"};
            stop();
            best = nodes;
            ++result.metrics.inflation_steps;
            if (unconstrained) break;
          } catch (const Failure& f) {
            if (!best.empty() && (f.status == Status::InsufficientClearance ||
                                  f.status == Status::DegeneratePath))
              break;
            throw;
          }
        }
        if (best.empty())
          throw Failure{Status::VerificationFailed,
                        "no accepted Hermite iterate"};
        result.corridor = GeneratorAccess::make(best);
        result.metrics.nodes = best.size();
        result.metrics.controls = result.corridor.centerControls().size();
        for (const auto& n : best)
          result.metrics.max_center_shift =
              std::max(result.metrics.max_center_shift, n.x.head<2>().norm());
        result.status = Status::Success;
        result.message =
            "quintic Hermite, offset/rotation optimization and continuous "
            "Bezier certificates passed";
        break;
      } catch (const Failure& f) {
        if (f.status != Status::InsufficientClearance &&
            f.status != Status::DegeneratePath)
          throw;
        last = f;
        if (refinement == o.max_refinements) throw last;
      }
    }
  } catch (const Failure& f) {
    result.status = f.status;
    result.message = f.message;
    result.corridor = Corridor();
  }
  result.metrics.total_ms = elapsed();
  return result;
}
}  // namespace gcopter::direct
