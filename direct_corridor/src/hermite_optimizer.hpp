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
struct LocalBox {
  double begin, end;  // seed arc length interval
  Bounds bounds;
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
inline std::pair<Patch, Patch> split(Patch p, double t = .5) {
  Patch a, b;
  a.span = b.span = p.span;
  a.reference[0] = p.reference[0];
  b.reference[5] = p.reference[5];
  a.w[0] = p.w[0];
  b.w[5] = p.w[5];
  for (int k = 1; k <= 5; ++k) {
    for (int j = 0; j <= 5 - k; ++j) {
      p.reference[j] = (1 - t) * p.reference[j] + t * p.reference[j + 1];
      p.w[j] = (1 - t) * p.w[j] + t * p.w[j + 1];
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
  const std::vector<LocalBox>& local_boxes;
  const Stop& stop;
  Metrics& metrics;
  std::vector<Cut> cuts;
  Certificates(const ObstacleIndex& m, const Bounds& b, const Options& o,
               const Stop& s, Metrics& me, const std::vector<LocalBox>& boxes)
      : map(m),
        bounds(b),
        options(o),
        local_boxes(boxes),
        stop(s),
        metrics(me) {}
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
    if (ids.empty()) {
      ++metrics.certificate_leaves;
      return;
    }
    std::array<Vec3, 6> c;
    std::array<Mat32, 6> m;
    controls(p, nodes, c, m);
    const Vec3 cm = middle(c);
    const Mat32 mm = middle(m);
    Eigen::SelfAdjointEigenSolver<M2> eig(mm.transpose() * mm);
    const V2 lambda = eig.eigenvalues();
    if (eig.info() != Eigen::Success || !lambda.allFinite() ||
        lambda[0] <= 1e-24)
      throw Failure{Status::DegeneratePath,
                    "degenerate section in separator metric"};
    const M2 u = eig.eigenvectors();
    const Eigen::Matrix<double, 2, 3> transform = u.transpose() * mm.transpose();
    const Vec3 normal = mm.col(0).cross(mm.col(1)).normalized();
    const double longitudinal =
        std::max(.5 * (c[5] - c[0]).norm(), options.min_radius);
    const double min_scale = std::min(std::sqrt(lambda[0]), longitudinal);
    const double pad = options.robot_radius + options.safety_margin +
                       options.certificate_margin;
    // Whitening is constant over this patch. Hoist all divisions out of the
    // obstacle loop. Equal-radius spheres have the same additive priority
    // offset, so ordering by the squared metric needs neither sqrt nor radius
    // division. These scores affect order only, never geometric acceptance.
    Eigen::Matrix3d metric;
    metric.topRows<2>() = lambda.cwiseInverse().asDiagonal() * transform;
    metric.row(2) = normal.transpose() / longitudinal;
    const double first_radius = map.obstacles()[ids.front()].radius;
    bool uniform_radius = true;
    struct Candidate {
      int id;
      double distance;
    };
    std::vector<Candidate> remaining;
    remaining.reserve(ids.size());
    // The ellipsoidal metric determines only the processing order. Every
    // accepted plane is checked against all six control ellipses, and every
    // discarded obstacle must lie wholly behind such an accepted plane.
    for (int id : ids) {
      const auto& sphere = map.obstacles()[id];
      uniform_radius &= sphere.radius == first_radius;
      remaining.push_back({id, (metric * (sphere.center - cm)).squaredNorm()});
    }
    if (!uniform_radius) {
      const double inverse_scale = 1 / min_scale;
      for (auto& candidate : remaining)
        candidate.distance = std::sqrt(candidate.distance) -
                             (map.obstacles()[candidate.id].radius + pad) *
                                 inverse_scale;
    }
    auto nearest = [&] {
      return std::min_element(
                 remaining.begin(), remaining.end(),
                 [](const Candidate& a, const Candidate& b) {
                   return a.distance < b.distance ||
                          (a.distance == b.distance && a.id < b.id);
                 }) - remaining.begin();
    };
    std::vector<int> failed;
    std::size_t next = nearest();
    while (!remaining.empty()) {
      stop();
      const int id = remaining[next].id;
      const auto& sphere = map.obstacles()[id];
      const Vec3 delta = sphere.center - cm;
      const V2 rhs = transform * delta;
      V2 v = rhs.cwiseQuotient(lambda);
      if (v.squaredNorm() > 1) {
        if (lambda[1] < 1.001 * lambda[0]) {
          v.normalize();
        } else {
          double lo = 0, hi = rhs.norm();
          for (int it = 0; it < 18; ++it) {
            const double t = .5 * (lo + hi);
            if ((rhs.array() / (lambda.array() + t)).matrix().squaredNorm() > 1)
              lo = t;
            else
              hi = t;
          }
          v = (rhs.array() / (lambda.array() + hi)).matrix();
        }
      }
      Vec3 n = delta - mm * (u * v);
      double h = -INFINITY, support = INFINITY;
      if (n.norm() > 1e-12) {
        n.normalize();
        h = n.dot(sphere.center) - sphere.radius - pad;
        support = -INFINITY;
        for (int j = 0; j < 6; ++j)
          support =
              std::max(support, n.dot(c[j]) + (m[j].transpose() * n).norm());
      }
      ++metrics.obstacle_pairs;
      if (support < h - 1e-10) {
        add(p, n, h, nodes);
        std::size_t count = 0;
        double best = INFINITY;
        for (const auto& candidate : remaining) {
          if (candidate.id == id) continue;
          ++metrics.obstacle_pairs;
          const auto& other = map.obstacles()[candidate.id];
          if (n.dot(other.center) - other.radius - pad >= h) continue;
          remaining[count] = candidate;
          if (count == 0 || candidate.distance < best) {
            best = candidate.distance;
            next = count;
          }
          ++count;
        }
        remaining.resize(count);
      } else {
        failed.push_back(id);
        remaining[next] = remaining.back();
        remaining.pop_back();
        if (!remaining.empty()) next = nearest();
      }
    }
    ++metrics.certificate_leaves;
    if (!failed.empty()) {
      if (depth >= options.certificate_depth)
        throw Failure{Status::InsufficientClearance,
                      "Hermite patch cannot be separated from an inflated obstacle"};
      const auto halves = split(p);
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
      Vec3 local_lo = Vec3::Constant(INFINITY), local_hi = -local_lo;
      if (!local_boxes.empty()) {
        const double start = nodes[i].s, end = nodes[i + 1].s, h = end - start;
        auto at = std::upper_bound(
            local_boxes.begin(), local_boxes.end(), start,
            [](double s, const LocalBox& box) { return s < box.end; });
        for (; at != local_boxes.end() && at->begin < end; ++at) {
          const double from = std::clamp((at->begin - start) / h, 0., 1.);
          const double to = std::clamp((at->end - start) / h, 0., 1.);
          if (to <= from) continue;
          Patch part = p;
          if (to < 1) part = split(part, to).first;
          if (from > 0) part = split(part, from / to).second;
          for (int axis = 0; axis < 3; ++axis) {
            add(part, Vec3::Unit(axis), at->bounds.upper[axis], nodes);
            add(part, -Vec3::Unit(axis), -at->bounds.lower[axis], nodes);
          }
          local_lo = local_lo.cwiseMin(at->bounds.lower);
          local_hi = local_hi.cwiseMax(at->bounds.upper);
        }
        if (!local_lo.allFinite())
          throw Failure{Status::VerificationFailed,
                        "missing local expansion interval"};
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
      Vec3 query_lo = lo - extent, query_hi = hi + extent;
      if (!local_boxes.empty()) {
        // The local support constraints already certify the entire curve is
        // in these boxes. This keeps loose algebraic radius/offset bounds
        // from turning each BVH query into a full-map scan.
        query_lo = query_lo.cwiseMax(local_lo - Vec3::Constant(pad));
        query_hi = query_hi.cwiseMin(local_hi + Vec3::Constant(pad));
      }
      map.query(query_lo, query_hi, ids);
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

// A node-only support row says its ellipse lies in a 2-D half-plane.
// Translate by a strictly feasible center d, then polarize:
//   a.y <= h  <=>  q.(y-d) <= 1, q=a/(h-a.d).
// A q strictly inside conv({0,q_j}) is redundant for EVERY ellipse, not just
// the current one. Keep coupled rows and numerically ambiguous boundary rows.
// Original rows remain available for the full residual audit after solving.
inline std::vector<int> irredundant(const std::vector<Node>& nodes,
                                  const std::vector<Cut>& cuts) {
  struct Point {
    V2 q;
    int id;
  };
  std::vector<std::vector<Point>> groups(nodes.size());
  std::vector<bool> keep(cuts.size(), true);
  for (int id = 0; id < int(cuts.size()); ++id) {
    const auto& c = cuts[id];
    if (c.linear || (c.a.squaredNorm() > 0 && c.b.squaredNorm() > 0)) continue;
    const bool left = c.a.squaredNorm() > 0;
    const V2 a = left ? c.a : c.b;
    const int i = c.span + !left;
    const double h = c.h - a.dot(nodes[i].x.head<2>());
    if (h > 0 && (a / h).allFinite()) groups[i].push_back({a / h, id});
  }
  auto cross = [](const V2& a, const V2& b, const V2& c) {
    return (static_cast<long double>(b[0]) - a[0]) *
               (static_cast<long double>(c[1]) - a[1]) -
           (static_cast<long double>(b[1]) - a[1]) *
               (static_cast<long double>(c[0]) - a[0]);
  };
  for (auto& points : groups) {
    if (points.size() < 4) continue;
    points.push_back({V2::Zero(), -1});
    std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
      return std::make_tuple(a.q[0], a.q[1], a.id) <
             std::make_tuple(b.q[0], b.q[1], b.id);
    });
    std::vector<Point> hull;
    for (const auto& p : points) {
      while (hull.size() >= 2 &&
             cross(hull[hull.size() - 2].q, hull.back().q, p.q) <= 0)
        hull.pop_back();
      hull.push_back(p);
    }
    const auto lower = hull.size();
    for (int j = int(points.size()) - 2; j >= 0; --j) {
      const auto& p = points[j];
      while (hull.size() > lower &&
             cross(hull[hull.size() - 2].q, hull.back().q, p.q) <= 0)
        hull.pop_back();
      hull.push_back(p);
    }
    if (hull.size() < 4) continue;
    hull.pop_back();
    for (const auto& p : points) {
      if (p.id < 0) continue;
      bool inside = true;
      for (std::size_t j = 0; j < hull.size(); ++j) {
        const V2& a = hull[j].q;
        const V2& b = hull[(j + 1) % hull.size()].q;
        const double guard = 1e-12 *
            (1 + (b - a).lpNorm<1>() * (p.q - a).lpNorm<1>());
        if (cross(a, b, p.q) <= guard) {
          inside = false;
          break;
        }
      }
      if (inside) keep[p.id] = false;
    }
  }
  std::vector<int> active;
  for (int id = 0; id < int(cuts.size()); ++id)
    if (keep[id]) active.push_back(id);
  return active;
}

// Five variables per knot: offset x/y and the full symmetric positive shape
// matrix xx/xy/yy. The off-diagonal entry optimizes ellipse orientation.
// Neighboring knots are coupled by SOC support constraints; the Newton matrix
// is block tridiagonal and solved in O(number of knots) memory and arithmetic.
struct Optimizer {
  const Options& o;
  const Stop& stop;
  std::vector<H5> diagonal, upper;
  std::vector<Eigen::LLT<H5>> factors;
  std::vector<V5> gradient, rhs, step, scaling;
  Optimizer(const Options& options, const Stop& s) : o(options), stop(s) {}
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
      const double determinant = p.determinant();
      if (p(0, 0) <= 0 || determinant <= 0) return false;
      cost -= weight * std::log(determinant);
      if (derivatives) {
        // For P^-1=[[a,b],[b,c]], expand -log det P in the symmetric
        // coordinates (P00,P01,P11), avoiding nine tiny matrix products.
        const double a = p(1, 1) / determinant, b = -p(0, 1) / determinant,
                     c = p(0, 0) / determinant;
        gradient[i].tail<3>() -= weight * sign * Vec3(a, 2 * b, c);
        Eigen::Matrix3d h;
        h << a * a, 2 * a * b, b * b,
             2 * a * b, 2 * (a * c + b * b), 2 * b * c,
             b * b, 2 * b * c, c * c;
        diagonal[i].bottomRightCorner<3, 3>() += weight * h;
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
    factors.resize(count);
    rhs.resize(count);
    step.resize(count);
    scaling.resize(count);
    for (double mu : {.003, .0001, .00001}) {
      for (int iteration = 0; iteration < 12; ++iteration) {
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
        // Equilibrate the Newton system before block elimination. Near-active
        // support barriers can differ by many orders of magnitude; an absolute
        // 1e-10 diagonal shift does not stabilize their Schur complements.
        for (int i = 0; i < count; ++i)
          scaling[i] = diagonal[i].diagonal().cwiseMax(1e-30).cwiseSqrt();
        for (int i = 0; i < count; ++i) {
          diagonal[i] = scaling[i].cwiseInverse().asDiagonal() * diagonal[i] *
                        scaling[i].cwiseInverse().asDiagonal();
          if (i + 1 < count)
            upper[i] = scaling[i].cwiseInverse().asDiagonal() * upper[i] *
                       scaling[i + 1].cwiseInverse().asDiagonal();
        }
        bool factored = false;
        // The convex barrier Hessian is positive definite; use Cholesky rather
        // than pivoted LDLT, and solve through factors without forming inverses.
        // If roundoff still
        // destroys a Schur pivot, restart the SAME system with larger relative
        // damping. The original objective and all support constraints remain
        // unchanged; the line search must accept the resulting descent step.
        for (double damping : {1e-10, 1e-8, 1e-6, 1e-4}) {
          factored = true;
          for (int i = 0; i < count; ++i) {
            H5 h = diagonal[i];
            rhs[i] = -gradient[i].cwiseQuotient(scaling[i]);
            if (i) {
              h -= upper[i - 1].transpose() *
                   factors[i - 1].solve(upper[i - 1]);
              rhs[i] -= upper[i - 1].transpose() *
                        factors[i - 1].solve(rhs[i - 1]);
            }
            h.diagonal().array() += damping;
            factors[i].compute(h);
            if (!h.allFinite() || factors[i].info() != Eigen::Success) {
              factored = false;
              break;
            }
          }
          if (factored) break;
        }
        if (!factored)
          throw Failure{Status::VerificationFailed,
                        "Hermite Newton factorization failed after scaling"};
        for (int i = count - 1; i >= 0; --i) {
          step[i] = factors[i].solve(
              rhs[i] - (i + 1 < count ? V5(upper[i] * step[i + 1])
                                     : V5::Zero()));
        }
        double slope = 0;
        for (int i = 0; i < count; ++i) {
          step[i] = step[i].cwiseQuotient(scaling[i]);
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
    const auto active = irredundant(nodes, cuts);
    ++metrics.optimization_passes;
    solve(nodes, cuts, active, metrics);
    metrics.active_constraints = active.size();
  }
};
}  // namespace gcopter::direct::detail
