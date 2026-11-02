// SPDX-License-Identifier: MIT
// Private implementation: convex node-shape optimization and local
// certificates.
#pragma once
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <tuple>

#include "gcopter/direct_corridor.hpp"

namespace gcopter::direct::detail {
using V5 = Eigen::Matrix<double, 5, 1>;
using H5 = Eigen::Matrix<double, 5, 5>;
using M2 = Eigen::Matrix2d;
using V2 = Eigen::Vector2d;
using Stop = std::function<void()>;
struct Failure {
  Status status;
  std::string message;
};
struct Node {
  double s;
  Vec3 reference, velocity;
  Mat32 frame;
  V5 x;
};
struct Patch {
  int span = 0;
  std::array<Vec3, 6> reference;
  std::array<double, 6> w{{0, 0, 0, 1, 1, 1}};
};
struct Cut {
  int span;
  V2 a, b;
  double h;
  bool linear = false;
};
inline M2 matrix(const V5& x) {
  M2 m;
  m << x[2], x[3], x[3], x[4];
  return m;
}
inline V5 circle(double r) {
  V5 x;
  x << 0, 0, r, 0, r;
  return x;
}
inline Vec3 center(const Node& n) {
  return n.reference + n.frame * n.x.head<2>();
}
inline Mat32 shape(const Node& n) { return n.frame * matrix(n.x); }
inline Patch patch(const std::vector<Node>& n, int i) {
  Patch p;
  p.span = i;
  const auto &a = n[i], &b = n[i + 1];
  double h = b.s - a.s;
  p.reference = {a.reference,
                 a.reference + h * a.velocity / 5.,
                 a.reference + 2 * h * a.velocity / 5.,
                 b.reference - 2 * h * b.velocity / 5.,
                 b.reference - h * b.velocity / 5.,
                 b.reference};
  return p;
}
inline std::pair<Patch, Patch> split(Patch p) {
  Patch a, b;
  a.span = b.span = p.span;
  a.reference[0] = p.reference[0];
  b.reference[5] = p.reference[5];
  a.w[0] = p.w[0];
  b.w[5] = p.w[5];
  for (int k = 1; k <= 5; ++k) {
    for (int j = 0; j <= 5 - k; ++j) {
      p.reference[j] = .5 * (p.reference[j] + p.reference[j + 1]);
      p.w[j] = .5 * (p.w[j] + p.w[j + 1]);
    }
    a.reference[k] = p.reference[0];
    b.reference[5 - k] = p.reference[5 - k];
    a.w[k] = p.w[0];
    b.w[5 - k] = p.w[5 - k];
  }
  return {a, b};
}
inline void controls(const Patch& p, const std::vector<Node>& n,
                     std::array<Vec3, 6>& c, std::array<Mat32, 6>& m) {
  const auto &a = n[p.span], &b = n[p.span + 1];
  const Vec3 ca = a.frame * a.x.head<2>(), cb = b.frame * b.x.head<2>();
  const Mat32 ma = shape(a), mb = shape(b);
  for (int j = 0; j < 6; ++j) {
    c[j] = p.reference[j] + (1 - p.w[j]) * ca + p.w[j] * cb;
    m[j] = (1 - p.w[j]) * ma + p.w[j] * mb;
  }
}
template <class T>
inline T middle(std::array<T, 6> v) {
  for (int k = 5; k > 0; --k)
    for (int j = 0; j < k; ++j) v[j] = .5 * (v[j] + v[j + 1]);
  return v[0];
}
inline double choose(int n, int k) {
  double v = 1;
  for (int j = 1; j <= k; ++j) v *= double(n + 1 - j) / j;
  return v;
}
inline bool positive(std::array<double, 15> v, double eps, int depth = 0) {
  if (*std::min_element(v.begin(), v.end()) > eps) return true;
  if (depth == 12 || v.front() <= eps || v.back() <= eps) return false;
  std::array<double, 15> a, b;
  a[0] = v[0];
  b[14] = v[14];
  for (int k = 1; k <= 14; ++k) {
    for (int j = 0; j <= 14 - k; ++j) v[j] = .5 * (v[j] + v[j + 1]);
    a[k] = v[0];
    b[14 - k] = v[14 - k];
  }
  return positive(a, eps, depth + 1) && positive(b, eps, depth + 1);
}
// Degree-14 Bernstein certificate for c' dot (M0 cross M1), including knots.
inline bool regular(const Patch& p, const std::vector<Node>& nodes) {
  std::array<Vec3, 6> c;
  std::array<Mat32, 6> m;
  controls(p, nodes, c, m);
  std::array<double, 15> d{};
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 6; ++j)
      for (int k = 0; k < 6; ++k)
        d[i + j + k] +=
            5 * (c[i + 1] - c[i]).dot(m[j].col(0).cross(m[k].col(1))) *
            choose(4, i) * choose(5, j) * choose(5, k) / choose(14, i + j + k);
  double scale = 0;
  for (double v : d) scale = std::max(scale, std::abs(v));
  return scale > 1e-18 && positive(d, std::max(1e-18, scale * 1e-10));
}

inline double value(const Cut& c, const std::vector<Node>& nodes) {
  const auto &a = nodes[c.span].x, &b = nodes[c.span + 1].x;
  double v = c.a.dot(a.head<2>()) + c.b.dot(b.head<2>());
  if (!c.linear) v += (matrix(a) * c.a + matrix(b) * c.b).norm();
  return v;
}

struct Certificates {
  const ObstacleIndex& map;
  const Bounds& bounds;
  const Options& options;
  const Stop& stop;
  Metrics& metrics;
  std::vector<Cut> cuts;
  // Fixed directions are used only when they already separate the current
  // entire patch. Otherwise retain the actual nearest-ellipse direction or
  // subdivide this obstacle/patch pair; no obstacle is discarded by clustering.
  std::array<Vec3, 729> directions;
  Certificates(const ObstacleIndex& m, const Bounds& b, const Options& o,
               const Stop& s, Metrics& me)
      : map(m), bounds(b), options(o), stop(s), metrics(me) {
    for (int x = -4; x <= 4; ++x)
      for (int y = -4; y <= 4; ++y)
        for (int z = -4; z <= 4; ++z) {
          Vec3 n(x, y, z);
          directions[(x + 4) * 81 + (y + 4) * 9 + z + 4] =
              n.norm() > 0 ? n.normalized() : Vec3::Zero();
        }
  }
  void add(const Patch& p, const Vec3& normal, double bound,
           const std::vector<Node>& nodes) {
    const auto &a = nodes[p.span], &b = nodes[p.span + 1];
    // Repeated Hermite value coefficients give identical variable rows: keep
    // only the strongest of them instead of paying for duplicate constraints.
    for (int j = 0; j < 6;) {
      const double w = p.w[j];
      double h = bound - normal.dot(p.reference[j]);
      int k = j + 1;
      while (k < 6 && std::abs(p.w[k] - w) < 1e-14) {
        h = std::min(h, bound - normal.dot(p.reference[k]));
        ++k;
      }
      Cut cut{p.span, (1 - w) * (a.frame.transpose() * normal),
              w * (b.frame.transpose() * normal), h};
      const double reach = (options.max_offset + options.max_radius) *
                           (cut.a.norm() + cut.b.norm());
      if (h < reach + options.numerical_guard) {
        if (cuts.size() >= options.max_constraints)
          throw Failure{Status::Capacity,
                        "certificate constraint capacity exceeded"};
        cuts.push_back(cut);
      }
      j = k;
    }
  }
  void pairCuts(const Patch& p, const std::vector<int>& ids,
                const std::vector<Node>& nodes, int depth) {
    stop();
    std::array<Vec3, 6> c;
    std::array<Mat32, 6> m;
    controls(p, nodes, c, m);
    const Vec3 cm = middle(c);
    const Mat32 mm = middle(m);
    Eigen::SelfAdjointEigenSolver<M2> eig(mm.transpose() * mm);
    const V2 lambda = eig.eigenvalues();
    const M2 u = eig.eigenvectors();
    const Eigen::Matrix<double, 2, 3> transform =
        u.transpose() * mm.transpose();
    auto support = [&](const Vec3& n) {
      double v = -std::numeric_limits<double>::infinity();
      for (int j = 0; j < 6; ++j)
        v = std::max(v, n.dot(c[j]) + (m[j].transpose() * n).norm());
      return v;
    };
    std::array<double, 729> planes, cached_support;
    planes.fill(std::numeric_limits<double>::infinity());
    cached_support.fill(std::numeric_limits<double>::quiet_NaN());
    std::vector<int> failed;
    for (int id : ids) {
      ++metrics.obstacle_pairs;
      const auto& s = map.obstacles()[id];
      const double radius = s.radius + options.robot_radius +
                            options.safety_margin + options.certificate_margin;
      const Vec3 delta = s.center - cm;
      // Interior map voxels often lie behind an already accepted separator.
      // Reuse that exact half-space certificate before solving another closest
      // ellipse problem. The test covers the whole inflated sphere.
      if (delta.cwiseAbs().maxCoeff() > 1e-12) {
        const Eigen::Vector3i radial = (4 * delta / delta.cwiseAbs().maxCoeff())
                                           .array()
                                           .round()
                                           .cast<int>();
        const int radial_key =
            (radial.x() + 4) * 81 + (radial.y() + 4) * 9 + radial.z() + 4;
        if (planes[radial_key] <= directions[radial_key].dot(s.center) - radius)
          continue;
      }
      const V2 rhs = transform * delta;
      V2 v = rhs.cwiseQuotient(lambda);
      if (v.squaredNorm() > 1) {
        if (lambda[1] < 1.001 * lambda[0])
          v.normalize();
        else {
          double lo = 0, hi = rhs.norm();
          for (int it = 0; it < 18; ++it) {
            double t = .5 * (lo + hi);
            if ((rhs.array() / (lambda.array() + t)).matrix().squaredNorm() > 1)
              lo = t;
            else
              hi = t;
          }
          v = (rhs.array() / (lambda.array() + hi)).matrix();
        }
      }
      Vec3 n = delta - mm * (u * v);
      if (n.norm() < 1e-12) {
        failed.push_back(id);
        continue;
      }
      n.normalize();
      Eigen::Vector3i q =
          (4 * n / n.cwiseAbs().maxCoeff()).array().round().cast<int>();
      int key = (q.x() + 4) * 81 + (q.y() + 4) * 9 + q.z() + 4;
      const Vec3 nq = directions[key];
      const double h = nq.dot(s.center) - radius;
      if (std::isnan(cached_support[key])) cached_support[key] = support(nq);
      if (cached_support[key] < h - 1e-10)
        planes[key] = std::min(planes[key], h);
      else {
        const double exact = n.dot(s.center) - radius;
        if (support(n) < exact - 1e-10)
          add(p, n, exact, nodes);
        else
          failed.push_back(id);
      }
    }
    for (int key = 0; key < 729; ++key)
      if (std::isfinite(planes[key]))
        add(p, directions[key], planes[key], nodes);
    ++metrics.certificate_leaves;
    if (!failed.empty()) {
      if (depth >= options.certificate_depth)
        throw Failure{
            Status::InsufficientClearance,
            "Hermite patch cannot be separated from an inflated obstacle"};
      auto halves = split(p);
      pairCuts(halves.first, failed, nodes, depth + 1);
      pairCuts(halves.second, failed, nodes, depth + 1);
    }
  }
  void build(const std::vector<Node>& nodes) {
    const double pad = options.robot_radius + options.safety_margin +
                       options.certificate_margin;
    for (int i = 0; i + 1 < int(nodes.size()); ++i) {
      const auto p = patch(nodes, i);
      for (int axis = 0; axis < 3; ++axis) {
        add(p, Vec3::Unit(axis), bounds.upper[axis] - pad, nodes);
        add(p, -Vec3::Unit(axis), -bounds.lower[axis] - pad, nodes);
      }
      // Linear forward-progress constraints at derivative Bezier controls.
      // They prevent center offsets from reversing the parameterization.
      for (const Vec3& normal :
           {nodes[i].velocity.normalized(), nodes[i + 1].velocity.normalized()})
        for (int j = 0; j < 5; ++j) {
          double dw = p.w[j + 1] - p.w[j];
          double h = normal.dot(p.reference[j + 1] - p.reference[j]) -
                     .000001 * (nodes[i + 1].s - nodes[i].s);
          Cut cut{i, dw * (nodes[i].frame.transpose() * normal),
                  -dw * (nodes[i + 1].frame.transpose() * normal), h, true};
          if (h <= 0)
            throw Failure{Status::DegeneratePath,
                          "reference Hermite derivative needs refinement"};
          if (cut.a.squaredNorm() + cut.b.squaredNorm() > 1e-20) {
            if (cuts.size() >= options.max_constraints)
              throw Failure{Status::Capacity,
                            "certificate constraint capacity exceeded"};
            cuts.push_back(cut);
          }
        }
      Vec3 lo = p.reference[0], hi = lo, extent;
      for (const auto& v : p.reference) {
        lo = lo.cwiseMin(v);
        hi = hi.cwiseMax(v);
      }
      for (int axis = 0; axis < 3; ++axis)
        extent[axis] = (options.max_radius + options.max_offset) *
                           std::max(nodes[i].frame.row(axis).norm(),
                                    nodes[i + 1].frame.row(axis).norm()) +
                       pad;
      std::vector<int> ids;
      map.query(lo - extent, hi + extent, ids);
      pairCuts(p, ids, nodes, 0);
    }
    // Merge identical node-only rows shared by adjacent patches. Scaling a
    // row does not change its feasible set; duplicate barriers otherwise also
    // bias the optimization towards directions with denser obstacle samples.
    for (auto& cut : cuts)
      if (!cut.linear) {
        if (cut.a.squaredNorm() == 0) {
          ++cut.span;
          cut.a = cut.b;
          cut.b.setZero();
        }
        if (cut.b.squaredNorm() == 0) {
          double norm = cut.a.norm();
          cut.a /= norm;
          cut.h /= norm;
        }
      }
    auto key = [](const Cut& c) {
      return std::make_tuple(c.span, c.linear, c.a[0], c.a[1], c.b[0], c.b[1]);
    };
    std::sort(cuts.begin(), cuts.end(),
              [&](const Cut& a, const Cut& b) { return key(a) < key(b); });
    std::vector<Cut> unique;
    unique.reserve(cuts.size());
    for (const auto& cut : cuts) {
      if (!unique.empty() && key(cut) == key(unique.back()))
        unique.back().h = std::min(unique.back().h, cut.h);
      else
        unique.push_back(cut);
    }
    // Keep the two-node storage contract also for the final node.
    for (auto& cut : unique)
      if (cut.span == int(nodes.size()) - 1) {
        --cut.span;
        cut.b = cut.a;
        cut.a.setZero();
      }
    cuts.swap(unique);
    metrics.constraints = cuts.size();
  }
};

// Five variables per knot: offset x/y and the full symmetric positive shape
// matrix xx/xy/yy. The off-diagonal entry optimizes ellipse orientation.
// Neighboring knots are coupled by SOC support constraints; the Newton matrix
// is block tridiagonal and solved in O(number of knots) memory and arithmetic.
struct Optimizer {
  const Options& o;
  const Stop& stop;
  std::vector<H5> diagonal, upper, inverse;
  std::vector<V5> gradient, rhs, step;
  std::array<M2, 3> basis;
  Optimizer(const Options& options, const Stop& s) : o(options), stop(s) {
    basis[0] << 1, 0, 0, 0;
    basis[1] << 0, 1, 1, 0;
    basis[2] << 0, 0, 0, 1;
  }
  double objective(const std::vector<Node>& n, const std::vector<Cut>& cuts,
                   const std::vector<int>& active, double mu,
                   bool derivatives) {
    const int count = n.size();
    if (derivatives) {
      diagonal.assign(count, H5::Zero());
      upper.assign(count - 1, H5::Zero());
      gradient.assign(count, V5::Zero());
    }
    double cost = 0;
    auto logdet = [&](int i, const M2& p, double weight, double sign) {
      if (p(0, 0) <= 0 || p.determinant() <= 0) return false;
      cost -= weight * std::log(p.determinant());
      if (derivatives) {
        const M2 inv = p.inverse();
        for (int a = 0; a < 3; ++a) {
          gradient[i][a + 2] -= weight * sign * (inv * basis[a]).trace();
          for (int b = 0; b < 3; ++b)
            diagonal[i](a + 2, b + 2) +=
                weight * (inv * basis[a] * inv * basis[b]).trace();
        }
      }
      return true;
    };
    for (int i = 0; i < count; ++i) {
      const auto& x = n[i].x;
      const M2 s = matrix(x);
      if (!logdet(i, s, 1., 1.) ||
          !logdet(i, s - o.min_radius * M2::Identity(), mu, 1.) ||
          !logdet(i, o.max_radius * M2::Identity() - s, mu, -1.))
        return INFINITY;
      const double gap =
          o.max_offset * o.max_offset - x.head<2>().squaredNorm();
      if (gap <= 0) return INFINITY;
      cost -= mu * std::log(gap);
      const double anchor = .02 / (o.max_radius * o.max_radius);
      cost += anchor * x.head<2>().squaredNorm();
      if (derivatives) {
        gradient[i].head<2>() += (2 * mu / gap + 2 * anchor) * x.head<2>();
        diagonal[i].topLeftCorner<2, 2>() +=
            (2 * mu / gap + 2 * anchor) * M2::Identity() +
            4 * mu / (gap * gap) * x.head<2>() * x.head<2>().transpose();
      }
    }
    for (int i = 0; i + 1 < count; ++i) {
      const double smoothing =
          .01 * std::min(4., o.control_spacing / (n[i + 1].s - n[i].s));
      const V5 delta = n[i + 1].x - n[i].x;
      cost += smoothing * delta.squaredNorm();
      if (derivatives) {
        gradient[i] -= 2 * smoothing * delta;
        gradient[i + 1] += 2 * smoothing * delta;
        diagonal[i].diagonal().array() += 2 * smoothing;
        diagonal[i + 1].diagonal().array() += 2 * smoothing;
        upper[i].diagonal().array() -= 2 * smoothing;
      }
    }
    for (int id : active) {
      const Cut& c = cuts[id];
      const int i = c.span;
      const V2 z = c.linear
                       ? V2::Zero()
                       : V2(matrix(n[i].x) * c.a + matrix(n[i + 1].x) * c.b);
      const double length = z.norm();
      const double gap = c.h - c.a.dot(n[i].x.head<2>()) -
                         c.b.dot(n[i + 1].x.head<2>()) - length;
      if (gap <= 0) return INFINITY;
      cost -= mu * std::log(gap);
      if (derivatives) {
        // A 2-D norm has a rank-one Hessian. Accumulate two rank-one
        // contributions directly into 5x5 blocks; most rows touch one knot.
        const V2 unit = length > 1e-14 ? V2(z / length) : V2::Zero();
        auto row = [&](const V2& a) {
          V5 v;
          v << a[0], a[1], unit[0] * a[0], unit[0] * a[1] + unit[1] * a[0],
              unit[1] * a[1];
          return v;
        };
        auto normal = [&](const V2& a) {
          V5 v;
          v << 0, 0, -unit[1] * a[0], -unit[1] * a[1] + unit[0] * a[0],
              unit[0] * a[1];
          return v;
        };
        const V5 ga = row(c.a), gb = row(c.b), wa = normal(c.a),
                 wb = normal(c.b);
        const double first = mu / gap, second = first / gap,
                     curvature = length > 1e-14 ? first / length : 0;
        const bool left = c.a.squaredNorm() > 0, right = c.b.squaredNorm() > 0;
        if (left) {
          gradient[i] += first * ga;
          diagonal[i].noalias() += second * (ga * ga.transpose()) +
                                   curvature * (wa * wa.transpose());
        }
        if (right) {
          gradient[i + 1] += first * gb;
          diagonal[i + 1].noalias() += second * (gb * gb.transpose()) +
                                       curvature * (wb * wb.transpose());
        }
        if (left && right)
          upper[i].noalias() += second * (ga * gb.transpose()) +
                                curvature * (wa * wb.transpose());
      }
    }
    return cost;
  }
  void solve(std::vector<Node>& nodes, const std::vector<Cut>& cuts,
             const std::vector<int>& active, Metrics& metrics) {
    const int count = nodes.size();
    inverse.resize(count);
    rhs.resize(count);
    step.resize(count);
    for (double mu : {.003, .0001, .00001}) {
      for (int iteration = 0; iteration < 8; ++iteration) {
        stop();
        ++metrics.newton_steps;
        const double before = objective(nodes, cuts, active, mu, true);
        if (!std::isfinite(before))
          throw Failure{Status::VerificationFailed,
                        "noninterior Hermite optimizer initialization"};
        // Selected endpoints remain exactly the center of the end sections.
        for (int i : {0, count - 1})
          for (int k = 0; k < 2; ++k) {
            gradient[i][k] = 0;
            diagonal[i].row(k).setZero();
            diagonal[i].col(k).setZero();
            diagonal[i](k, k) = 1;
            if (i < count - 1) upper[i].row(k).setZero();
            if (i > 0) upper[i - 1].col(k).setZero();
          }
        for (int i = 0; i < count; ++i) {
          H5 h = diagonal[i];
          rhs[i] = -gradient[i];
          if (i) {
            const H5 l = upper[i - 1].transpose() * inverse[i - 1];
            h -= l * upper[i - 1];
            rhs[i] -= l * rhs[i - 1];
          }
          h.diagonal().array() += 1e-10;
          Eigen::LDLT<H5> ldlt(h);
          if (ldlt.info() != Eigen::Success || !ldlt.isPositive())
            throw Failure{Status::VerificationFailed,
                          "Hermite Newton factorization failed"};
          inverse[i] = ldlt.solve(H5::Identity());
        }
        double slope = 0;
        for (int i = count - 1; i >= 0; --i) {
          step[i] =
              inverse[i] * (rhs[i] - (i + 1 < count ? V5(upper[i] * step[i + 1])
                                                    : V5::Zero()));
          slope += gradient[i].dot(step[i]);
        }
        if (-slope < 1e-7 * count) break;
        double alpha = 1;
        bool accepted = false;
        auto trial = nodes;
        for (int ls = 0; ls < 28; ++ls) {
          for (int i = 0; i < count; ++i)
            trial[i].x = nodes[i].x + alpha * step[i];
          const double after = objective(trial, cuts, active, mu, false);
          if (std::isfinite(after) && after <= before + 1e-3 * alpha * slope) {
            nodes.swap(trial);
            accepted = true;
            break;
          }
          alpha *= .5;
        }
        if (!accepted) break;
      }
    }
  }
  void optimize(std::vector<Node>& nodes, const std::vector<Cut>& cuts,
                Metrics& metrics) {
    // Feasible local circle warm start. This only initializes the optimizer;
    // offsets and the full symmetric shape remain free during the solve.
    bool centered = true;
    for (const auto& node : nodes) centered &= node.x.head<2>().norm() < 1e-12;
    if (centered) {
      std::vector<double> radius(nodes.size(), .9 * o.max_radius);
      for (const auto& c : cuts)
        if (!c.linear) {
          const double a = c.a.norm(), b = c.b.norm(), r = .85 * c.h / (a + b);
          if (a > 0) radius[c.span] = std::min(radius[c.span], r);
          if (b > 0) radius[c.span + 1] = std::min(radius[c.span + 1], r);
        }
      auto candidate = nodes;
      for (std::size_t i = 0; i < nodes.size(); ++i)
        candidate[i].x = circle(std::max(nodes[i].x[2], radius[i]));
      bool feasible = true;
      for (const auto& c : cuts)
        if (value(c, candidate) >= c.h - 1e-10) {
          feasible = false;
          break;
        }
      if (feasible) nodes.swap(candidate);
    }
    std::vector<int> active(cuts.size());
    std::iota(active.begin(), active.end(), 0);
    ++metrics.optimization_passes;
    solve(nodes, cuts, active, metrics);
    metrics.active_constraints = active.size();
  }
};
}  // namespace gcopter::direct::detail
