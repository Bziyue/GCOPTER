// SPDX-License-Identifier: MIT
// Private implementation included by refit.cpp. One positive scale per local
// quintic control matrix; continuous SOC constraints retain their exact norm.
namespace {
struct ScalarCut {
  std::array<int, 6> ids{};
  std::array<Eigen::Vector2d, 6> a;
  int size = 0;
  double bound = 0;
};
struct ScalarProblem {
  Basis basis;
  std::vector<Mat32> control;
  V weights;
  std::vector<ScalarCut> cuts;

  ScalarProblem(const Corridor& base, const std::vector<Plane>& planes)
      : basis(base.knots(), 5, true), control(basis.count), weights(basis.count) {
    for (int i = 0; i < basis.count; ++i) {
      control[i] = base.hermiteNodes()[i / 3].state.shape;
      weights[i] = (basis.knots[i + 6] - basis.knots[i]) / 6;
    }
    cuts.reserve(planes.size() * 6);
    for (const auto& plane : planes) {
      const auto& span = basis.spans[plane.span];
      M e = restrictControls(span.extraction, plane.begin, plane.end);
      M c(6, 3);
      for (int j = 0; j < 6; ++j) c.row(j) = base.centerControls()[6 * plane.span + j];
      c = restrictControls(c, plane.begin, plane.end);
      for (int k = 0; k < 6; ++k) {
        ScalarCut cut;
        cut.bound = plane.bound - c.row(k).dot(plane.normal);
        for (int j = 0; j < e.cols(); ++j) {
          if (std::abs(e(k, j)) <= 1e-12) continue;
          const int at = cut.size++;
          cut.ids[at] = span.ids[j];
          cut.a[at] = e(k, j) * control[span.ids[j]].transpose() * plane.normal;
        }
        if (cut.bound <= 0) throw std::runtime_error("scalar shape center outside cover");
        cuts.push_back(cut);
      }
    }
  }
  double eval(const V& x, double mu, V* g = nullptr, Band* h = nullptr) const {
    if (g) g->setZero(x.size());
    if (h) h->zero();
    double value = 0;
    for (int i = 0; i < x.size(); ++i) {
      if (x[i] <= 0) return INFINITY;
      value -= 2 * weights[i] * std::log(x[i]);
      if (g) (*g)[i] = -2 * weights[i] / x[i];
      if (h) h->add(i, i, 2 * weights[i] / (x[i] * x[i]));
    }
    const double w = mu / cuts.size();
    for (const auto& c : cuts) {
      Eigen::Vector2d z = Eigen::Vector2d::Zero();
      for (int j = 0; j < c.size; ++j) z += c.a[j] * x[c.ids[j]];
      const double r = z.norm(), gap = c.bound - r;
      if (gap <= 0) return INFINITY;
      value -= w * std::log(gap / c.bound);
      if (!g) continue;
      const Eigen::Vector2d unit = r > 1e-14 ? Eigen::Vector2d(z / r) : Eigen::Vector2d::Zero();
      const Eigen::Vector2d perpendicular(-unit.y(), unit.x());
      double radial[6], tangential[6];
      for (int j = 0; j < c.size; ++j) {
        radial[j] = c.a[j].dot(unit);
        tangential[j] = c.a[j].dot(perpendicular);
        (*g)[c.ids[j]] += w / gap * radial[j];
      }
      if (h) {
        const double a = w / (gap * gap), b = r > 1e-14 ? w / (gap * r) : 0;
        for (int j = 0; j < c.size; ++j)
          for (int k = 0; k <= j; ++k)
            h->add(c.ids[j], c.ids[k], a * radial[j] * radial[k] + b * tangential[j] * tangential[k]);
      }
    }
    return value;
  }
  std::vector<Mat32> output(const V& x) const {
    std::vector<Mat32> out;
    out.reserve(6 * basis.spans.size());
    for (const auto& span : basis.spans)
      for (int k = 0; k < 6; ++k) {
        Mat32 m = Mat32::Zero();
        for (int j = 0; j < span.extraction.cols(); ++j)
          m += span.extraction(k, j) * x[span.ids[j]] * control[span.ids[j]];
        out.push_back(m);
      }
    return out;
  }
};

// Six-wide equilibrated Cholesky with reusable factor/RHS workspaces and no
// sparse assembly. Local basis support guarantees the band structure.
static V scalarNewton(const Band& h, const V& g, M& factor, V& rhs, V& scale) {
  const int n = h.n, width = h.width;
  scale = h.data.col(0).cwiseMax(1e-30).cwiseSqrt();
  factor.setZero(n, width);
  for (int i = 0; i < n; ++i) {
    for (int j = std::max(0, i - width + 1); j <= i; ++j) {
      double v = h.data(i, i - j) / (scale[i] * scale[j]);
      if (i == j) v += 1e-10;
      for (int k = std::max(0, i - width + 1); k < j; ++k)
        v -= factor(i, i - k) * factor(j, j - k);
      if (i == j) {
        if (v <= 0) throw std::runtime_error("scalar band Cholesky pivot");
        factor(i, 0) = std::sqrt(v);
      } else factor(i, i - j) = v / factor(j, 0);
    }
    double v = -g[i] / scale[i];
    for (int j = std::max(0, i - width + 1); j < i; ++j) v -= factor(i, i - j) * rhs[j];
    rhs[i] = v / factor(i, 0);
  }
  V step(n);
  for (int i = n - 1; i >= 0; --i) {
    double v = rhs[i];
    for (int j = i + 1; j < std::min(n, i + width); ++j) v -= factor(j, j - i) * step[j];
    step[i] = v / factor(i, 0);
  }
  return step.cwiseQuotient(scale);
}

static double scalarEnergy(const Corridor& base, const std::vector<Mat32>& shapes, double length) {
  double value = 0;
  for (size_t i = 0; i < base.pieces(); ++i) {
    const double h = (base.knots()[i + 1] - base.knots()[i]) * length;
    for (int k = 0; k < 6; ++k) {
      const auto b = second(.5 * (1 + gx[k]), h);
      Mat32 m = Mat32::Zero();
      for (int j = 0; j < 6; ++j) m += b[j] * shapes[6 * i + j];
      value += .5 * gw[k] * h * m.squaredNorm();
    }
  }
  return value;
}
}

static ShapeFitResult fitScalarShape(const Corridor& base, const std::vector<Plane>& cover,
                                    double length, const detail::Stop& stop, bool check,
                                    bool coordinate = false) {
  auto start = Clock::now(), timer = start;
  ShapeFitResult out;
  auto& metrics = out.metrics;
  metrics.degree = 5;
  metrics.direct_initialization = true;
  metrics.reused_planes = cover.size();
  ScalarProblem p(base, cover);
  metrics.controls = p.basis.count;
  metrics.constraints = p.cuts.size();
  V x = V::Ones(p.basis.count);
  auto reproduction = p.output(x);
  for (size_t i = 0; i < reproduction.size(); ++i)
    metrics.fit_error = std::max(metrics.fit_error, (reproduction[i] - base.shapeControls()[i]).norm());
  double scale = 1;
  for (const auto& c : p.cuts) {
    Eigen::Vector2d z = Eigen::Vector2d::Zero();
    for (int j = 0; j < c.size; ++j) z += c.a[j];
    if (z.norm() > 1e-14) scale = std::min(scale, c.bound / z.norm());
  }
  metrics.initial_scale = .95 * scale;
  x *= metrics.initial_scale;
  const V initial = x;
  Band h(x.size(), 6);
  V g, rhs(x.size()), scaling(x.size());
  M factor(x.size(), 6);
  metrics.setup_ms = ms(timer);
  timer = Clock::now();
  if (check) {
    p.eval(x, .01, &g, &h);
    V direction(x.size());
    for (int i = 0; i < x.size(); ++i) direction[i] = std::sin(i * .71 + .2);
    direction.normalize();
    const double eps = 1e-5;
    V gp, gm;
    const double fp = p.eval(x + eps * direction, .01, &gp), fm = p.eval(x - eps * direction, .01, &gm);
    const double fd = (fp - fm) / (2 * eps);
    metrics.gradient_error = std::abs(fd - g.dot(direction)) / std::max(1., std::abs(fd));
    metrics.hessian_error = ((gp - gm) / (2 * eps) - h.multiply(direction)).norm() /
                            std::max(1., h.multiply(direction).norm());
    if (metrics.gradient_error > 1e-5 || metrics.hessian_error > 1e-4)
      throw std::runtime_error("scalar shape gradient/Hessian check");
    const V step = scalarNewton(h, g, factor, rhs, scaling);
    if ((h.multiply(step) + g).norm() / std::max(1., g.norm()) > 1e-5)
      throw std::runtime_error("scalar band solve residual");
  }
  if (coordinate) {
    // Start from the accepted tube. Each coordinate is increased to the exact
    // boundary of its incident SOCs; no global Hessian or barrier path.
    x.setConstant(std::min(1., scale * (1 - 1e-10)));
    metrics.initial_scale = x[0];
    struct Incident { int cut, term; };
    std::vector<std::vector<Incident>> incident(x.size());
    std::vector<Eigen::Vector2d> projected(p.cuts.size(), Eigen::Vector2d::Zero());
    for (int k = 0; k < int(p.cuts.size()); ++k) {
      const auto& c = p.cuts[k];
      for (int j = 0; j < c.size; ++j) {
        incident[c.ids[j]].push_back({k, j});
        projected[k] += c.a[j] * x[c.ids[j]];
      }
    }
    for (int pass = 0; pass < 2; ++pass) {
      stop();
      for (int at = 0; at < x.size(); ++at) {
        const int i = pass == 0 ? at : x.size() - 1 - at;
        double limit = INFINITY;
        for (const auto& item : incident[i]) {
          const auto& c = p.cuts[item.cut];
          const auto& a = c.a[item.term];
          const auto& z = projected[item.cut];
          const double aa = a.squaredNorm();
          if (aa <= 1e-28) continue;
          const double r = z.norm(), slack = std::max(0., (c.bound - r) * (c.bound + r));
          const double az = a.dot(z), root = std::sqrt(az * az + aa * slack);
          const double delta = az >= 0 ? slack / std::max(root + az, 1e-30) : (root - az) / aa;
          limit = std::min(limit, delta);
        }
        if (!std::isfinite(limit)) throw std::runtime_error("unbounded scalar coordinate");
        const double delta = .995 * limit;
        x[i] += delta;
        for (const auto& item : incident[i])
          projected[item.cut] += p.cuts[item.cut].a[item.term] * delta;
      }
      ++metrics.iterations;
    }
  } else for (double mu : {.03, .003, .0003}) {
    for (int iteration = 0; iteration < 12; ++iteration) {
      stop();
      const double f = p.eval(x, mu, &g, &h);
      V step = scalarNewton(h, g, factor, rhs, scaling);
      const double slope = g.dot(step);
      if (!std::isfinite(slope) || slope > 0) throw std::runtime_error("scalar Newton direction");
      if (-slope < 1e-8) break;
      double alpha = 1;
      bool accepted = false;
      for (int ls = 0; ls < 30; ++ls) {
        V trial = x + alpha * step;
        const double after = p.eval(trial, mu);
        if (std::isfinite(after) && after <= f + 1e-4 * alpha * slope) {
          x = std::move(trial);
          accepted = true;
          break;
        }
        alpha *= .5;
      }
      ++metrics.iterations;
      if (!accepted) break;
    }
  }
  metrics.optimize_ms = ms(timer);
  timer = Clock::now();
  auto shapes = p.output(x);
  while (!regular(base, shapes)) {
    if (++metrics.regularity_backtracks > 20) throw std::runtime_error("scalar regularity restoration");
    x = .5 * (x + initial);
    shapes = p.output(x);
  }
  verifyShapeControls(base, cover, length, shapes, metrics);
  metrics.baseline_acceleration_energy = scalarEnergy(base, base.shapeControls(), length);
  metrics.shape_acceleration_energy = scalarEnergy(base, shapes, length);
  out.corridor = ShapeAccess::make(base, shapes);
  metrics.verify_ms = ms(timer);
  metrics.total_ms = ms(start);
  return out;
}
