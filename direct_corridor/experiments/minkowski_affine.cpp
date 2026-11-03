// SPDX-License-Identifier: MIT
#include "minkowski_affine_shape.hpp"
#include <chrono>
namespace gcopter::direct::minkowski {
namespace {
Vec3 segmentClosest(const Vec3& p,const Vec3& a,const Vec3& b) {
  const Vec3 d=b-a;
  const double t=d.squaredNorm()>1e-24 ? std::clamp((p-a).dot(d)/d.squaredNorm(),0.,1.):0.;
  return a+t*d;
}
}
struct AffineAccess {
  static Corridor make(const std::vector<detail::Node>& nodes,const AffineShape& shape) {
    Corridor c;
    c.templates_=shape.bank.count; c.q_=shape.bank.q; c.mixed_=shape.bank.mixed;
    for(const auto& n:nodes) c.knots_.push_back(n.s/nodes.back().s);
    c.knots_.back()=1;
    for(int i=0;i+1<int(nodes.size());++i) {
      std::array<Vec3,6> cp;
      std::array<Mat32,6> mp;
      detail::controls(detail::patch(nodes,i),nodes,cp,mp);
      for(int j=0;j<6;++j) {
        c.centers_.push_back(cp[j]); c.frames_.push_back(mp[j]);
        c.weights_.push_back(shape.weights);
      }
    }
    return c;
  }
};
Result generateAffine(const std::vector<Vec3>& seed, const ObstacleIndex& index,
                const Bounds& bounds, const Options& settings) {
  using namespace detail;
  auto o=settings.geometry;
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
    for (double v :
         {o.control_spacing, o.min_radius, o.max_radius, o.max_offset,
          o.robot_radius, o.safety_margin, o.numerical_guard,
          o.certificate_margin, o.budget_ms, o.local_range, o.local_progress})
      if (!std::isfinite(v))
        throw Failure{Status::InvalidInput, "nonfinite generator option"};
    if (seed.size() < 2 || !bounds.lower.allFinite() ||
        !bounds.upper.allFinite() ||
        (bounds.upper.array() <= bounds.lower.array()).any() ||
        o.control_spacing <= 0 || o.min_radius <= 0 ||
        o.max_radius <= o.min_radius || o.max_offset <= 0 ||
        o.local_range < 0 || o.local_progress < 1e-6 || o.robot_radius < 0 ||
        o.safety_margin < 0 || o.numerical_guard <= 0 ||
        o.certificate_margin < o.numerical_guard || o.budget_ms <= 0 ||
        o.max_refinements < 0 || o.max_refinements > 20 || o.max_controls < 2 ||
        o.max_controls > 1000000 || o.max_constraints < 1 ||
        o.inflation_iterations < 1 || o.inflation_iterations > 5 ||
        o.certificate_depth < 1 || o.certificate_depth > 16)
      throw Failure{Status::InvalidInput,
                    "invalid seed, geometry or Hermite options"};
    if((settings.templates!=2 && settings.templates!=4) || !std::isfinite(settings.thinness) ||
       settings.thinness<.01 || settings.thinness>1 || !std::isfinite(settings.angle) ||
       settings.newton_iterations<1 || settings.newton_iterations>36 ||
       settings.orientation_smoothing!=0 ||
       settings.adapt_orientation || settings.product_bounds || settings.subdivisions!=0)
      throw Failure{Status::InvalidInput,"invalid affine Minkowski options"};
    const AffineShape shape(settings);
    o.max_radius/=shape.inner_radius;
    o.min_radius/=shape.inner_radius;
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
    std::vector<LocalBox> local_boxes;
    if (o.local_range > 0) {
      const double clearance = pad + o.certificate_margin;
      for (std::size_t i = 1; i < clean.size(); ++i) {
        Vec3 a = clean[i - 1];
        double start = arc[i - 1];
        while (start < arc[i]) {
          stop();
          const double remaining = (clean[i] - a).norm();
          const bool last = remaining <= o.local_progress;
          const Vec3 b =
              last ? clean[i]
                   : Vec3(a + (clean[i] - a) * (o.local_progress / remaining));
          const double end = last ? arc[i] : start + (b - a).norm();
          if (end <= start)
            throw Failure{Status::InvalidInput,
                          "local progress below coordinate resolution"};
          if (local_boxes.size() >= o.max_controls)
            throw Failure{Status::Capacity,
                          "local expansion box capacity exceeded"};
          local_boxes.push_back(
              {start,
               end,
               {(a.cwiseMin(b) - Vec3::Constant(o.local_range))
                    .cwiseMax(bounds.lower + Vec3::Constant(clearance)),
                (a.cwiseMax(b) + Vec3::Constant(o.local_range))
                    .cwiseMin(bounds.upper - Vec3::Constant(clearance))}});
          a = b;
          start = end;
        }
      }
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
        QueryCache query_cache;
        for (int iteration = 0; iteration < o.inflation_iterations;
             ++iteration) {
          if (iteration && elapsed() > .5 * o.budget_ms) break;
          try {
            double timer = elapsed();
            ShapeCertificates<AffineShape> certificates(index, bounds, o, stop, result.metrics,
                                      local_boxes, query_cache, shape);
            // The tiny initial circles carry no useful free-space shape.
            // Use the learned metric only after the first inflation solve.
            certificates.use_shape_metric = iteration > 0;
            certificates.build(nodes);
            result.metrics.certificate_ms += elapsed() - timer;
            for (const auto& c : certificates.cuts)
              if (detail::value(c, nodes, shape) >= c.h - 1e-11)
                throw Failure{Status::InsufficientClearance,
                              "initial Hermite control hull needs refinement"};
            auto full = nodes;
            for (auto& n : full) n.x = circle(o.max_radius);
            bool unconstrained = true;
            for (const auto& c : certificates.cuts)
              if (detail::value(c, full, shape) >= c.h - 1e-9) {
                unconstrained = false;
                break;
              }
            if (unconstrained)
              nodes = std::move(full);
            else {
              timer = elapsed();
              ShapeOptimizer<AffineShape> optimizer(o, stop, shape);
              optimizer.iteration_limit=settings.newton_iterations;
              optimizer.optimize(nodes, certificates.cuts, result.metrics);
              result.metrics.optimization_ms += elapsed() - timer;
            }
            // Full SOC residual audit, covering every obstacle certificate.
            for (const auto& c : certificates.cuts)
              if (detail::value(c, nodes, shape) > c.h + o.numerical_guard * .01)
                throw Failure{Status::VerificationFailed,
                              "full Hermite support residual failed"};
            for (int i = 0; i + 1 < int(nodes.size()); ++i)
              if (!detail::regular(detail::patch(nodes, i), nodes))
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
        result.corridor = AffineAccess::make(best,shape);
        result.metrics.nodes = best.size();
        result.metrics.controls = result.corridor.centerControls().size();
        for (const auto& n : best)
          result.metrics.max_center_shift =
              std::max(result.metrics.max_center_shift, n.x.head<2>().norm());
        result.status = Status::Success;
        result.message =
            "affine Minkowski sweep, convex logdet solve and continuous Bezier certificates passed";
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
}
