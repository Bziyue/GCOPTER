// SPDX-License-Identifier: MIT
#include "refit.hpp"
#include "../../src/hermite_optimizer.hpp"
#include <Eigen/SparseCholesky>
#include <Eigen/LU>
#include <chrono>
#include <cassert>
#include <stdexcept>
namespace gcopter::direct::experiment {
using Clock = std::chrono::steady_clock;
using V = Eigen::VectorXd;
using M = Eigen::MatrixXd;
using V3 = Eigen::Vector3d;
using H3 = Eigen::Matrix3d;
using J23 = Eigen::Matrix<double, 2, 3>;
using Sparse = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
using Plane = detail::Certificates::Plane;
static double ms(Clock::time_point t){
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
struct ShapeAccess {
  static Corridor make(const Corridor& base, const std::vector<Mat32>& shapes) {
    Corridor out = base;
    out.shapes_ = shapes;
    for (size_t i = 0; i < out.nodes_.size(); ++i) {
      out.nodes_[i].state = out.section(out.knots_[i]);
    }
    return out;
  }
};
namespace {
// Lower symmetric band; no dense N-by-N allocation in the Newton loop.
struct Band {
  int n, width;
  M data;
  Band(int nn, int w) : n(nn), width(w), data(M::Zero(nn, w)) {
  }
  void zero() {
    data.setZero();
  }
  void add(int i, int j, double x) {
    if (i < j){
      std::swap(i, j);
    }
    // Local spline support fixes this bandwidth during setup.
    assert(i - j < width);
    data(i, i - j) += x;
  }
  Sparse sparse()const {
    std::vector<Triplet> t;
    t.reserve(n * width);
    for (int i = 0; i < n; ++i){
      for (int k = 0; k < std::min(width, i + 1); ++k){
        t.emplace_back(i, i - k, data(i, k));
      }
    }
    Sparse s(n, n);
    s.setFromTriplets(t.begin(), t.end());
    return s;
  }
  V multiply(const V& x)const {
    V y = V::Zero(n);
    for (int i = 0; i < n; ++i){
      for (int k = 0; k < std::min(width, i + 1); ++k){
        int j = i - k;
        y[i] += data(i, k) * x[j];
        if (k){
          y[j] += data(i, k) * x[i];
        }
      }
    }
    return y;
  }
};
struct Term {
  int id;
  J23 jac;
};
struct Constraint {
  std::vector<Term> terms;
  double bound;
};
struct Span {
  std::vector<int> ids;
  M extraction;
};
static Eigen::Matrix2d matrix(const V& x, int i) {
  Eigen::Matrix2d s;
  s << x[3 * i], x[3 * i + 1], x[3 * i + 1], x[3 * i + 2];
  return s;
}
static J23 jacobian(const Eigen::Vector2d& a){
  J23 j;
  j << a[0], a[1], 0, 0, a[0], a[1];
  return j;
}
static M restrictControls(M a, double begin, double end) {
  auto split = [](M p, double t, bool left){
                 M q(6, p.cols());
                 if (left){
                   q.row(0) = p.row(0);
                 }
                 else{
                   q.row(5) = p.row(5);
                 }
                 for (int k = 1; k <= 5; ++k){
                   for (int j = 0; j <= 5 - k; ++j){
                     p.row(j) = (1 - t) * p.row(j) + t * p.row(j + 1);
                   }
                   if (left){
                     q.row(k) = p.row(0);
                   }
                   else{
                     q.row(5 - k) = p.row(5 - k);
                   }
                 }
                 return q;
               };
  if (end < 1) {
    a = split(a, end, true);
  }
  if (begin > 0){
    a = split(a, begin / end, false);
  }
  return a;
}
struct Basis {
  int degree, count;
  std::vector<double> knots;
  std::vector<Span> spans;
  Basis(const std::vector<double>& breaks, int d, bool local = false) : degree(d) {
    knots.insert(knots.end(), d + 1, 0.);
    for (size_t i = 1; i + 1 < breaks.size(); ++i){
      knots.insert(knots.end(), d - 2, breaks[i]);
    }
    knots.insert(knots.end(), d + 1, 1.);
    count = int(knots.size()) - d - 1;
    M coll(d + 1, d + 1);
    for (int k = 0; k <= d; ++k){
      double t = double(k) / d;
      for (int j = 0; j <= d; ++j){
        coll(k, j) = detail::choose(d, j) * std::pow(t, j) * std::pow(1 - t, d - j);
      }
    }
    const M inverse = coll.inverse();
    M elevate = M::Zero(6, d + 1);
    for (int j = 0; j < 6; ++j){
      for (int k = 0; k <= d; ++k){
        if (j - k >= 0&&j - k <= 5 - d){
          elevate(j, k) = detail::choose(d, k) * detail::choose(5 - d, j - k) / detail::choose(5,
                                                                                               j);
        }
      }
    }
    for (size_t i = 0; i + 1 < breaks.size(); ++i) {
      if (local) {
        // The polynomial on this element has only d+1 nonzero basis functions.
        // Evaluate them on the fixed knot span, including its endpoint limits.
        const int at = d + (d - 2) * i;
        M samples(d + 1, d + 1);
        for (int k = 0; k <= d; ++k) {
          const double u = breaks[i] + (breaks[i + 1] - breaks[i]) * double(k) / d;
          std::array<double, 6> b{}, left{}, right{};
          b[0] = 1;
          for (int j = 1; j <= d; ++j) {
            left[j] = u - knots[at + 1 - j];
            right[j] = knots[at + j] - u;
            double saved = 0;
            for (int r = 0; r < j; ++r) {
              const double temp = b[r] / (right[r + 1] + left[j - r]);
              b[r] = saved + right[r + 1] * temp;
              saved = left[j - r] * temp;
            }
            b[j] = saved;
          }
          for (int j = 0; j <= d; ++j) samples(k, j) = b[j];
        }
        Span span;
        for (int j = 0; j <= d; ++j) span.ids.push_back(at - d + j);
        span.extraction = elevate * inverse * samples;
        spans.push_back(std::move(span));
        continue;
      }
      M samples(d + 1, count);
      for (int k = 0; k <= d; ++k){
        samples.row(k) = values(breaks[i] + (breaks[i + 1] - breaks[i]) * double(k) /
                                d).transpose();
      }
      M all = elevate * inverse * samples;
      Span s;
      for (int j = 0; j < count; ++j){
        if (all.col(j).cwiseAbs().maxCoeff() > 1e-11){
          s.ids.push_back(j);
        }
      }
      s.extraction.resize(6, s.ids.size());
      for (size_t j = 0; j < s.ids.size(); ++j){
        s.extraction.col(j) = all.col(s.ids[j]);
      }
      spans.push_back(std::move(s));
    }
  }
  V values(double u)const {
    V b = V::Zero(knots.size() - 1);
    if (u >= 1){
      b[count - 1] = 1;
      return b.head(count);
    }
    for (int i = 0; i < b.size(); ++i) {
      if (knots[i] <= u&&u < knots[i + 1]){
        b[i] = 1;
      }
    }
    for (int k = 1; k <= degree; ++k) {
      for (int i = 0; i < int(knots.size()) - k - 1; ++i){
        double x = 0, den = knots[i + k] - knots[i];
        if (den > 0){
          x += (u - knots[i]) / den * b[i];
        }
        den = knots[i + k + 1] - knots[i + 1];
        if (den > 0){
          x += (knots[i + k + 1] - u) / den * b[i + 1];
        }
        b[i] = x;
      }
    }
    return b.head(count);
  }
};
static constexpr double gx[] = {-.9324695142031521, -.6612093864662645, -.2386191860831969,
                                .2386191860831969, .6612093864662645, .9324695142031521};
static constexpr double gw[] = {.1713244923791704, .3607615730481386, .4679139345726910,
                                .4679139345726910, .3607615730481386, .1713244923791704};
static Eigen::Matrix<double, 6, 1> bern(double t) {
  Eigen::Matrix<double, 6, 1>b;
  for (int k = 0; k < 6; ++k){
    b[k] = detail::choose(5, k) * std::pow(t, k) * std::pow(1 - t, 5 - k);
  }
  return b;
}
static Eigen::Matrix<double, 6, 1> second(double t, double h) {
  Eigen::Matrix<double, 6, 1>b = Eigen::Matrix<double, 6, 1>::Zero();
  for (int k = 0; k < 4; ++k){
    double v = 20 / (h * h) * detail::choose(3, k) * std::pow(t, k) * std::pow(1 - t, 3 - k);
    b[k] += v;
    b[k + 1] -= 2 * v;
    b[k + 2] += v;
  }
  return b;
}
static std::vector<Plane> planes(const Corridor& base, const std::vector<Vec3>& seed,
                                 const ObstacleIndex& map, const Bounds& bounds,
                                 const Options& options, double length) {
  std::vector<detail::Node> nodes;
  for (const auto& h:base.hermiteNodes()){
    detail::Node n;
    n.s = h.u * length;
    n.reference = h.state.center;
    n.velocity = h.state.first / length;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(h.state.shape.transpose() * h.state.shape);
    Eigen::Matrix2d s = eig.eigenvectors() * eig.eigenvalues().cwiseSqrt().asDiagonal() *
                        eig.eigenvectors().transpose();
    n.frame = h.state.shape * s.inverse();
    n.x << 0, 0, s(0, 0), s(0, 1), s(1, 1);
    nodes.push_back(n);
  }
  std::vector<detail::LocalBox> boxes;
  if (options.local_range > 0){
    Vec3 a, b = seed.front();
    double arc = 0, pad = options.robot_radius + options.safety_margin + options.certificate_margin;
    for (size_t i = 1; i < seed.size();){
      a = b;
      if ((a - seed[i]).norm() > options.local_progress){
        b = a + (seed[i] - a).normalized() * options.local_progress;
      }
      else{
        b = seed[i++];
      }
      double end = arc + (b - a).norm();
      boxes.push_back({arc, end,
                       {(a.cwiseMin(b) -
                         Vec3::Constant(options.local_range)).cwiseMax(bounds.lower +
                                                                       Vec3::Constant(pad)),
                        (a.cwiseMax(b) +
                         Vec3::Constant(options.local_range)).cwiseMin(bounds.upper -
                                                                       Vec3::Constant(pad))}});
      arc = end;
    }
  }
  detail::Stop stop = []{
                      };
  Metrics metrics;
  detail::QueryCache cache;
  detail::Certificates cert(map, bounds, options, stop, metrics, boxes, cache);
  std::vector<Plane> out;
  cert.exported_planes = &out;
  cert.use_shape_metric = true;
  cert.build(nodes);
  return out;
}
struct Problem {
  Basis basis;
  std::vector<Mat32> control;
  std::vector<Constraint> constraints;
  V weights;
  Band energy;
  double energy_weight = 0;
  V identity;
  Problem(const Corridor& base, int degree, double length, bool direct_initialization)
      : basis(base.knots(), degree),
    weights(V::Zero(basis.count)), energy(3 * basis.count, 3 * (degree + 1)),
    identity(V::Zero(3 * basis.count)) {
    const int n = basis.count;
    if (direct_initialization) {
      // A degree-5 C2 spline with triple internal knots represents a zero-jet
      // Hermite curve by three identical control matrices per node. No Gram
      // matrix, quadrature projection or global linear solve is needed.
      control.resize(n);
      for (int i = 0; i < n; ++i) {
        control[i] = base.hermiteNodes()[i / 3].state.shape;
        weights[i] = (basis.knots[i + degree + 1] - basis.knots[i]) / (degree + 1);
        identity[3 * i] = identity[3 * i + 2] = 1;
      }
    } else {
      std::vector<Triplet> ts;
      M rhs = M::Zero(n, 6);
      for (size_t i = 0; i < base.pieces(); ++i) {
        const auto& e = basis.spans[i];
        double du = base.knots()[i + 1] - base.knots()[i];
        for (int k = 0; k < 6; ++k){
          double t = .5 * (1 + gx[k]), w = .5 * gw[k] * du;
          V b = e.extraction.transpose() * bern(t);
          Mat32 target = Mat32::Zero();
          for (int j = 0; j < 6; ++j){
            target += bern(t)[j] * base.shapeControls()[6 * i + j];
          }
          for (int j = 0; j < b.size(); ++j){
            int a = e.ids[j];
            weights[a] += w * b[j];
            for (int z = 0; z < 6; ++z){
              rhs(a, z) += w * b[j] * target.data()[z];
            }
            for (int l = 0; l < b.size(); ++l){
              ts.emplace_back(a, e.ids[l], w * b[j] * b[l]);
            }
          }
        }
      }
      Sparse gram(n, n);
      gram.setFromTriplets(ts.begin(), ts.end());
      Eigen::SimplicialLDLT<Sparse> solve;
      solve.compute(gram);
      if (solve.info() != Eigen::Success){
        throw std::runtime_error("shape projection factorization");
      }
      M coeff = solve.solve(rhs);
      control.resize(n);
      for (int i = 0; i < n; ++i){
        for (int z = 0; z < 6; ++z){
          control[i].data()[z] = coeff(i, z);
        }
        identity[3 * i] = identity[3 * i + 2] = 1;
      }
    }
    // Physical arc-length acceleration regularizer, common to both degrees.
    for (size_t i = 0; i < base.pieces(); ++i) {
      const auto& e = basis.spans[i];
      double h = (base.knots()[i + 1] - base.knots()[i]) * length;
      for (int k = 0; k < 6; ++k){
        double t = .5 * (1 + gx[k]), w = .5 * gw[k] * h;
        V dd = e.extraction.transpose() * second(t, h);
        std::vector<Eigen::Matrix<double, 6, 3> > maps(e.ids.size());
        for (size_t j = 0; j < e.ids.size(); ++j){
          const Mat32& a = control[e.ids[j]];
          Mat32 m0 = Mat32::Zero(), m1, m2 = Mat32::Zero();
          m0.col(0) = a.col(0);
          m1.col(0) = a.col(1);
          m1.col(1) = a.col(0);
          m2.col(1) = a.col(1);
          maps[j].col(0) = Eigen::Map<const Eigen::Matrix<double, 6, 1> >(m0.data()) * dd[j];
          maps[j].col(1) = Eigen::Map<const Eigen::Matrix<double, 6, 1> >(m1.data()) * dd[j];
          maps[j].col(2) = Eigen::Map<const Eigen::Matrix<double, 6, 1> >(m2.data()) * dd[j];
        }
        for (size_t j = 0; j < e.ids.size(); ++j) {
          for (size_t l = 0; l <= j; ++l){
            H3 hess = 2 * w * maps[j].transpose() * maps[l];
            for (int a = 0; a < 3; ++a){
              for (int b = 0; b < 3; ++b){
                if (j != l||a >= b){
                  energy.add(3 * e.ids[j] + a, 3 * e.ids[l] + b, hess(a, b));
                }
              }
            }
          }
        }
      }
    }
  }
  void constrain(const Corridor& base, const std::vector<Plane>& planes) {
    for (const auto& plane:planes) {
      const auto& span = basis.spans[plane.span];
      M e = restrictControls(span.extraction, plane.begin, plane.end);
      M c(6, 3);
      for (int j = 0; j < 6; ++j){
        c.row(j) = base.centerControls()[6 * plane.span + j];
      }
      c = restrictControls(c, plane.begin, plane.end);
      for (int k = 0; k < 6; ++k){
        Constraint cut;
        cut.bound = plane.bound - c.row(k).dot(plane.normal);
        for (int j = 0; j < e.cols(); ++j){
          if (std::abs(e(k, j)) > 1e-12){
            cut.terms.push_back({span.ids[j],
                                 e(k,
                                   j) * jacobian(control[span.ids[j]].transpose() * plane.normal)});
          }
        }
        if (cut.bound <= 0){
          throw std::runtime_error("fixed center outside certificate");
        }
        constraints.push_back(std::move(cut));
      }
    }
  }
  double eval(const V& x, double mu, V* gradient = nullptr, Band* hessian = nullptr)const {
    const bool deriv = gradient;
    double cost = 0;
    if (deriv){
      *gradient = V::Zero(x.size());
    }
    if (hessian){
      hessian->zero();
    }
    const Eigen::Matrix2d bases[3] = {(Eigen::Matrix2d() << 1, 0, 0, 0).finished(),
                                      (Eigen::Matrix2d() << 0, 1, 1, 0).finished(),
                                      (Eigen::Matrix2d() << 0, 0, 0, 1).finished()};
    for (int i = 0; i < basis.count; ++i) {
      auto s = matrix(x, i);
      double det = s.determinant();
      if (s(0, 0) <= 0||det <= 1e-16){
        return INFINITY;
      }
      double w = weights[i];
      cost -= w * std::log(det);
      const auto inv = s.inverse().eval();
      if (deriv){
        for (int a = 0; a < 3; ++a){
          (*gradient)[3 * i + a] -= w * (inv * bases[a]).trace();
        }
      }
      if (hessian){
        for (int a = 0; a < 3; ++a){
          for (int b = 0; b <= a; ++b){
            hessian->add(3 * i + a, 3 * i + b, w * (inv * bases[a] * inv * bases[b]).trace());
          }
        }
      }
      V3 delta = x.segment<3>(3 * i) - identity.segment<3>(3 * i);
      cost += .005 * w * delta.squaredNorm();
      if (deriv){
        gradient->segment<3>(3 * i) += .01 * w * delta;
      }
      if (hessian){
        for (int a = 0; a < 3; ++a){
          hessian->add(3 * i + a, 3 * i + a, .01 * w);
        }
      }
    }
    V ex = energy.multiply(x);
    cost += .5 * energy_weight * x.dot(ex);
    if (deriv){
      *gradient += energy_weight * ex;
    }
    if (hessian){
      hessian->data += energy_weight * energy.data;
    }
    const double weight = mu / constraints.size();
    for (const auto& c:constraints) {
      Eigen::Vector2d q = Eigen::Vector2d::Zero();
      for (const auto& t:c.terms){
        q += t.jac * x.segment<3>(3 * t.id);
      }
      double r = q.norm(), g = c.bound - r;
      if (g <= 0){
        return INFINITY;
      }
      cost -= weight * std::log(g / c.bound);
      if (!deriv&&!hessian){
        continue;
      }
      Eigen::Vector2d u = r > 1e-14?(q / r).eval():Eigen::Vector2d::Zero();
      if (deriv){
        for (const auto& t:c.terms){
          gradient->segment<3>(3 * t.id) += weight / g * t.jac.transpose() * u;
        }
      }
      if (hessian) {
        Eigen::Matrix2d curvature = weight *
                                    (u * u.transpose() / (g * g) +
                                     (Eigen::Matrix2d::Identity() - u * u.transpose()) /
                                     (g * std::max(r, 1e-14)));
        for (size_t j = 0; j < c.terms.size(); ++j){
          for (size_t l = 0; l <= j; ++l){
            const auto &a = c.terms[j], &b = c.terms[l];
            H3 block = a.jac.transpose() * curvature * b.jac;
            for (int v = 0; v < 3; ++v){
              for (int w = 0; w < 3; ++w){
                if (j != l||v >= w){
                  hessian->add(3 * a.id + v, 3 * b.id + w, block(v, w));
                }
              }
            }
          }
        }
      }
    }
    return cost;
  }
  std::vector<Mat32> output(const V& x)const {
    std::vector<Mat32> out;
    out.reserve(6 * basis.spans.size());
    for (const auto& e:basis.spans){
      for (int k = 0; k < 6; ++k){
        Mat32 m = Mat32::Zero();
        for (int j = 0; j < e.extraction.cols(); ++j){
          m += e.extraction(k, j) * control[e.ids[j]] * matrix(x, e.ids[j]);
        }
        out.push_back(m);
      }
    }
    return out;
  }
};
static bool regular(const Corridor& base, const std::vector<Mat32>& shape) {
  for (size_t span = 0; span < base.pieces(); ++span) {
    std::array<double, 15>d{};
    for (int i = 0; i < 5; ++i){
      for (int j = 0; j < 6; ++j){
        for (int k = 0; k < 6; ++k){
          d[i + j + k] += 5 *
                          (base.centerControls()[6 * span + i + 1] -
                           base.centerControls()[6 * span + i]).dot(shape[6 * span +
                                                                          j].col(0).cross(shape[6 *
                                                                                                span
                                                                                                + k]
                                                                                          .col(
                                                                                            1))) *
                          detail::choose(4, i) * detail::choose(5, j) * detail::choose(5,
                                                                                       k) /
                          detail::choose(14, i + j + k);
        }
      }
    }
    double scale = 0;
    for (double x:d){
      scale = std::max(scale, std::abs(x));
    }
    if (scale <= 1e-18||!detail::positive(d, std::max(1e-18, scale * 1e-10))){
      return false;
    }
  }
  return true;
}
}
static void verifyShapeControls(const Corridor& base, const std::vector<Plane>& cover,
                                double length, const std::vector<Mat32>& shapes,
                                ShapeFitMetrics& metrics) {
  metrics.max_support_residual = -INFINITY;
  // Audit the exported coefficients directly, independently of the sparse
  // optimization rows (including any tiny extraction weights).
  for (const auto& plane : cover) {
    M centers(6, 3), matrices(6, 6);
    for (int k = 0; k < 6; ++k) {
      centers.row(k) = base.centerControls()[6 * plane.span + k];
      for (int j = 0; j < 6; ++j) {
        matrices(k, j) = shapes[6 * plane.span + k].data()[j];
      }
    }
    centers = restrictControls(centers, plane.begin, plane.end);
    matrices = restrictControls(matrices, plane.begin, plane.end);
    for (int k = 0; k < 6; ++k) {
      Mat32 shape;
      for (int j = 0; j < 6; ++j) {
        shape.data()[j] = matrices(k, j);
      }
      const double support = centers.row(k).dot(plane.normal) +
                             (shape.transpose() * plane.normal).norm();
      metrics.max_support_residual = std::max(metrics.max_support_residual,
                                               support - plane.bound);
    }
  }
  if (metrics.max_support_residual > 1e-9){
    throw std::runtime_error("shape support certificate");
  }
  for (size_t i = 1; i < base.pieces(); ++i) {
    double left = (base.knots()[i] - base.knots()[i - 1]) * length,
           right = (base.knots()[i + 1] - base.knots()[i]) * length;
    const int k = 6 * i;
    Mat32 dl = 5 * (shapes[k - 1] - shapes[k - 2]) / left,
          dr = 5 * (shapes[k + 1] - shapes[k]) / right;
    Mat32 al = 20 * (shapes[k - 1] - 2 * shapes[k - 2] + shapes[k - 3]) / (left * left),
          ar = 20 * (shapes[k + 2] - 2 * shapes[k + 1] + shapes[k]) / (right * right);
    metrics.c2_error = std::max({metrics.c2_error, (shapes[k - 1] - shapes[k]).norm(),
                                 (dl - dr).norm(), (al - ar).norm()});
    metrics.max_knot_first = std::max(metrics.max_knot_first, dr.norm());
    metrics.max_knot_second = std::max(metrics.max_knot_second, ar.norm());
  }
  if (metrics.c2_error > 1e-6) {
    throw std::runtime_error("shape C2 join");
  }
}
#include "scalar_inflation.hpp"

static ShapeFitResult fitWithCover(const Corridor& base, const std::vector<Plane>& cover,
                                    double length, int degree, bool check,
                                    bool direct_initialization, const detail::Stop& stop) {
  auto start = Clock::now(), timer = start;
  ShapeFitResult out;
  auto& metrics = out.metrics;
  metrics.degree = degree;
  metrics.direct_initialization = direct_initialization;
  Problem p(base, degree, length, direct_initialization);
  p.constrain(base, cover);
  metrics.controls = p.basis.count;
  metrics.constraints = p.constraints.size();
  double energy = 0;
  for (size_t i = 0; i < base.pieces(); ++i){
    double h = (base.knots()[i + 1] - base.knots()[i]) * length;
    for (int k = 0; k < 6; ++k){
      auto b = second(.5 * (1 + gx[k]), h);
      Mat32 m = Mat32::Zero();
      for (int j = 0; j < 6; ++j){
        m += b[j] * base.shapeControls()[6 * i + j];
      }
      energy += .5 * gw[k] * h*m.squaredNorm();
    }
  }
  metrics.baseline_acceleration_energy = energy;
  p.energy_weight = .05 / std::max(energy, 1e-6);
  auto unscaled = p.output(p.identity);
  for (size_t i = 0; i < unscaled.size(); ++i){
    metrics.fit_error = std::max(metrics.fit_error, (unscaled[i] - base.shapeControls()[i]).norm());
  }
  double scale = 1;
  for (const auto& c:p.constraints){
    Eigen::Vector2d q = Eigen::Vector2d::Zero();
    for (const auto& t:c.terms){
      q += t.jac * p.identity.segment<3>(3 * t.id);
    }
    if (q.norm() > 1e-14){
      scale = std::min(scale, c.bound / q.norm());
    }
  }
  metrics.initial_scale = .8 * scale;
  V x = metrics.initial_scale * p.identity;
  const V initial = x;
  if (!regular(base, p.output(x))) {
    throw std::runtime_error("initial shape spline fails continuous regularity");
  }
  Band h(x.size(), 3 * (degree + 1));
  V g;
  metrics.setup_ms = ms(timer);
  timer = Clock::now();
  if (check) {
    p.eval(x, .1, &g, &h);
    V direction(x.size());
    for (int i = 0; i < direction.size(); ++i){
      direction[i] = std::sin(i * .71 + .2);
    }
    direction.normalize();
    double eps = 1e-5;
    V gp, gm;
    double fp = p.eval(x + eps * direction, .1, &gp), fm = p.eval(x - eps * direction, .1, &gm);
    double fd = (fp - fm) / (2 * eps);
    metrics.gradient_error = std::abs(fd - g.dot(direction)) / std::max(1., std::abs(fd));
    const V exact_hv = h.multiply(direction);
    metrics.hessian_error = ((gp - gm) / (2 * eps) - exact_hv).norm() /
                            std::max(1., exact_hv.norm());
    if (metrics.gradient_error > 1e-5||metrics.hessian_error > 1e-4){
      throw std::runtime_error("shape objective derivative check");
    }
  }
  Eigen::SimplicialLDLT<Sparse> solver;
  bool analyzed = false;
  for (double mu:{1., .1, .01, .001}) {
    for (int iteration = 0; iteration < 20; ++iteration){
      if (stop) stop();
      double f = p.eval(x, mu, &g, &h);
      auto sparse = h.sparse();
      if (!analyzed){
        solver.analyzePattern(sparse);
        analyzed = true;
      }
      solver.factorize(sparse);
      if (solver.info() != Eigen::Success){
        throw std::runtime_error("shape Newton factorization");
      }
      V step = solver.solve(-g);
      double descent = g.dot(step);
      if (!std::isfinite(descent)||descent > 0){
        throw std::runtime_error("shape Newton direction");
      }
      if (-descent < 1e-10){
        break;
      }
      double alpha = 1;
      bool accepted = false;
      for (int ls = 0; ls < 40; ++ls){
        V candidate = x + alpha * step;
        double next = p.eval(candidate, mu);
        if (std::isfinite(next)&&next <= f + 1e-4 * alpha * descent){
          x = std::move(candidate);
          accepted = true;
          break;
        }
        alpha *= .5;
      }
      ++metrics.iterations;
      if (!accepted){
        break;
      }
    }
  }
  metrics.optimize_ms = ms(timer);
  timer = Clock::now();
  auto shapes = p.output(x);
  while (!regular(base, shapes)){
    if (++metrics.regularity_backtracks > 20){
      throw std::runtime_error("shape regularity restoration");
    }
    x = .5 * (x + initial);
    shapes = p.output(x);
  }
  verifyShapeControls(base, cover, length, shapes, metrics);
  out.corridor = ShapeAccess::make(base, shapes);
  metrics.shape_acceleration_energy = .5 * x.dot(p.energy.multiply(x));
  metrics.verify_ms = ms(timer);
  metrics.total_ms = ms(start);
  return out;
}
ShapeFitResult refitShape(const Corridor& base, const std::vector<Vec3>& seed,
                         const ObstacleIndex& map, const Bounds& bounds,
                         const Options& options, int degree, bool check) {
  if ((degree != 3 && degree != 5) || base.pieces() == 0 || seed.size() < 2)
    throw std::invalid_argument("shape experiment degree/input");
  auto start = Clock::now();
  double length = 0;
  for (size_t i = 1; i < seed.size(); ++i)
    length += (seed[i] - seed[i - 1]).norm();
  auto cover = planes(base, seed, map, bounds, options, length);
  double planes_ms = ms(start);
  auto out = fitWithCover(base, cover, length, degree, check, false, {});
  out.metrics.planes_ms = planes_ms;
  out.metrics.total_ms = ms(start);
  return out;
}

Result generateIntegratedShape(const std::vector<Vec3>& seed, const ObstacleIndex& map,
                               const Bounds& bounds, const Options& options,
                               ShapeFitMetrics& metrics, bool check) {
  metrics = {};
  return detail::generateWithShapeStage(seed, map, bounds, options,
      [&](Corridor& base, const std::vector<Plane>& cover, double length,
          const detail::Stop& stop) {
        auto fit = fitWithCover(base, cover, length, 5, check, true, stop);
        fit.metrics.reused_planes = static_cast<int>(cover.size());
        metrics = fit.metrics;
        base = std::move(fit.corridor);
      });
}

Result generateFastShape(const std::vector<Vec3>& seed, const ObstacleIndex& map,
                         const Bounds& bounds, const Options& options,
                         ShapeFitMetrics& metrics, int inflation_rounds, bool check) {
  if (inflation_rounds < 1 || inflation_rounds > 2)
    throw std::invalid_argument("fast shape inflation rounds");
  metrics = {};
  Options o = options;
  o.inflation_iterations = inflation_rounds;
  return detail::generateWithShapeStage(seed, map, bounds, o,
      [&](Corridor& base, const std::vector<Plane>& cover, double length,
          const detail::Stop& stop) {
        auto fit = fitScalarShape(base, cover, length, stop, check);
        metrics = fit.metrics;
        base = std::move(fit.corridor);
      });
}

Result generateCoordinateShape(const std::vector<Vec3>& seed, const ObstacleIndex& map,
                               const Bounds& bounds, const Options& options,
                               ShapeFitMetrics& metrics, int warmup_limit, bool check, bool fraction_to_boundary, bool continue_barrier) {
  if (warmup_limit < 1 || warmup_limit > 12)
    throw std::invalid_argument("coordinate warmup limit");
  metrics = {};
  return detail::generateWithShapeStage(seed, map, bounds, options,
      [&](Corridor& base, const std::vector<Plane>& cover, double length,
          const detail::Stop& stop) {
        auto fit = fitScalarShape(base, cover, length, stop, check, true);
        metrics = fit.metrics;
        base = std::move(fit.corridor);
      }, warmup_limit, fraction_to_boundary, continue_barrier);
}

}
