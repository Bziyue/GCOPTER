  // Experimental fraction-to-boundary step. The current point is interior,
  // established by objective(); alpha is capped on its convex feasible ray.
  static void clipPolynomial(double c, double b, double a, double& alpha) {
    if (c + alpha * (b + alpha * a) > 0) return;
    if (a == 0) {
      alpha = std::min(alpha, -c / b);
      return;
    }
    const double discriminant = std::max(0., b * b - 4 * a * c);
    const double q = -.5 * (b + std::copysign(std::sqrt(discriminant), b));
    const double first = q / a, second = c / q;
    if (first > 0) alpha = std::min(alpha, first);
    if (second > 0) alpha = std::min(alpha, second);
  }
  double feasibleStep(const std::vector<Node>& nodes, const std::vector<Cut>& cuts,
                      const std::vector<int>& active) const {
    constexpr double fraction = .99;
    double alpha = 1 / fraction;
    auto positiveMatrix = [&](const M2& p, const M2& d) {
      if (p(0, 0) + alpha * d(0, 0) <= 0)
        alpha = std::min(alpha, -p(0, 0) / d(0, 0));
      clipPolynomial(p.determinant(),
                     p(0, 0) * d(1, 1) + p(1, 1) * d(0, 0) -
                         2 * p(0, 1) * d(0, 1),
                     d.determinant(), alpha);
    };
    for (int i = 0; i < int(nodes.size()); ++i) {
      const auto& x = nodes[i].x;
      const auto& d = step[i];
      const M2 s = matrix(x), ds = matrix(d);
      positiveMatrix(s - o.min_radius * M2::Identity(), ds);
      positiveMatrix(o.max_radius * M2::Identity() - s, -ds);
      clipPolynomial(o.max_offset * o.max_offset - x.head<2>().squaredNorm(),
                     -2 * x.head<2>().dot(d.head<2>()), -d.head<2>().squaredNorm(), alpha);
    }
    for (int id : active) {
      const auto& c = cuts[id];
      const int i = c.span;
      const double t = c.h - c.a.dot(nodes[i].x.head<2>()) -
                                  c.b.dot(nodes[i + 1].x.head<2>());
      const double dt = -c.a.dot(step[i].head<2>()) - c.b.dot(step[i + 1].head<2>());
      if (t + alpha * dt <= 0) alpha = std::min(alpha, -t / dt);
      if (c.linear) continue;
      const V2 z = matrix(nodes[i].x) * c.a + matrix(nodes[i + 1].x) * c.b;
      const V2 dz = matrix(step[i]) * c.a + matrix(step[i + 1]) * c.b;
      const double next_t = t + alpha * dt;
      if (next_t * next_t > (z + alpha * dz).squaredNorm()) continue;
      const double length = z.norm();
      clipPolynomial((t - length) * (t + length),
                     2 * (t * dt - z.dot(dz)), dt * dt - dz.squaredNorm(), alpha);
    }
    return std::min(1., fraction * alpha);
  }
