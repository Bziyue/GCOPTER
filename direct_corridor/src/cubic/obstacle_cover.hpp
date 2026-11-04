// SPDX-License-Identifier: MIT
#pragma once
#include "gcopter/cubic/cubic_certificates.hpp"
#include "../hermite_optimizer.hpp"
#include "lazy_obstacles.hpp"
#include "runtime.hpp"
#include "gcopter/cubic/metrics.hpp"
#include "gcopter/cubic/options.hpp"
#include <memory>
#include <chrono>

namespace gcopter::direct::native {
using detail::QueryCache;
using detail::ObstacleGroups;
using detail::obstacleGroups;
using detail::metricSphereNormal;
using detail::Failure;
using M2=Eigen::Matrix2d;
using V2=Eigen::Vector2d;
template<int D> struct GeometryPatch {
  int span;double begin=0,end=1;Curve<D,V9> controls;
  std::pair<GeometryPatch,GeometryPatch> halves() const {
    auto p=split<D>(controls);double mid=.5*(begin+end);
    return {{span,begin,mid,p.first},{span,mid,end,p.second}};
  }
};
// Valid for one seed/knots/bounds/options tuple. The generator owns a fresh
// layout per knot refinement, while changing geometry reuses these intervals.
// Dynamic support tests and separation planes are still computed each pass.
struct CoverLayout {
  struct Part {double begin,end;Bounds bounds;};
  struct Span {size_t begin,end;Vec3 lower,upper;};
  std::vector<Span> spans;
  std::vector<Part> parts;
  CoverLayout(const std::vector<double>& breaks,const std::vector<Vec3>& seed,const Bounds& bounds,const GeometryOptions& options) {
    const double pad=options.robot_radius+options.safety_margin+options.certificate_margin;
    std::vector<detail::LocalBox> boxes;
    double length=0;Vec3 a,b=seed.front();
    for(size_t j=1;j<seed.size();) {
      a=b;if((a-seed[j]).norm()>options.local_progress)b=a+(seed[j]-a).normalized()*options.local_progress;else b=seed[j++];
      const double end=length+(b-a).norm();
      boxes.push_back({length,end,{(a.cwiseMin(b)-Vec3::Constant(options.local_range)).cwiseMax(bounds.lower+Vec3::Constant(pad)),
                                   (a.cwiseMax(b)+Vec3::Constant(options.local_range)).cwiseMin(bounds.upper-Vec3::Constant(pad))}});
      length=end;
    }
    spans.resize(breaks.size()-1);
    parts.reserve(spans.size()+boxes.size());
    for(size_t i=0;i<spans.size();++i) {
      auto& span=spans[i];span.begin=parts.size();span.lower=Vec3::Constant(INFINITY);span.upper=-span.lower;
      const double s=breaks[i]*length,e=breaks[i+1]*length,h=e-s;
      for(const auto& box:boxes) {
        const double from=std::clamp((box.begin-s)/h,0.,1.),to=std::clamp((box.end-s)/h,0.,1.);
        if(to<=from)continue;
        parts.push_back({from,to,box.bounds});
        span.lower=span.lower.cwiseMin(box.bounds.lower);span.upper=span.upper.cwiseMax(box.bounds.upper);
      }
      span.end=parts.size();
    }
  }
};
// The sphere separator uses the production BVH, metricSphereNormal and spatial
// packets. Only the polynomial control count and emission of raw planes differ.
template<int D> struct MapCover {
  using Patch=GeometryPatch<D>;
  const ObstacleIndex& map;const Bounds& bounds;const GeometryOptions& options;
  QueryCache& query_cache;CoverMetrics& timings;Metrics& metrics;
  Deadline& deadline;
  bool profile;
  std::vector<detail::SupportPlane> planes;
  MapCover(const ObstacleIndex& m,const Bounds& b,const GeometryOptions& o,QueryCache& cache,
           CoverMetrics& t,Deadline& budget,bool profiling)
    :map(m),bounds(b),options(o),query_cache(cache),timings(t),metrics(t.primitives),deadline(budget),profile(profiling){}
  void add(const Patch& p,const Vec3& n,double h) {planes.push_back({p.span,p.begin,p.end,n,h});}
  void limit(const Patch& p,const Vec3& n,double h,int depth=0) {
    if(supportBound<D>(p.controls,n,h)<=1e-9){add(p,n,h);return;}
    if(depth==options.certificate_depth)throw Failure{Status::InsufficientClearance,"continuous local/world bounds failed"};
    const auto halves=p.halves();limit(halves.first,n,h,depth+1);limit(halves.second,n,h,depth+1);
  }
  void pairCuts(const Patch& p, const std::vector<int>& ids,
                int depth) {
    if (ids.empty()) {
      ++metrics.certificate_leaves;
      return;
    }
    if(depth)deadline.check(); // The outer span already checked depth zero.
    Curve<D,Vec3> c;
    Curve<D,Mat32> m;
    for(int j=0;j<=D;++j){c[j]=p.controls[j].template head<3>();m[j]=shape(p.controls[j]);}
    const Vec3 cm = value<D>(c,.5);
    const Mat32 mm = value<D>(m,.5);
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
        std::max(.5 * (c[D] - c[0]).norm(), options.min_radius);
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
    Eigen::Matrix3d rotation;
    rotation.topRows<2>() = lambda.cwiseSqrt().cwiseInverse().asDiagonal() *
                           transform;
    rotation.row(2) = normal.transpose();
    const Vec3 axes2(lambda[0], lambda[1], longitudinal * longitudinal);
    // Keep spatial groups intact while planes remove individual spheres.
    // Bounds can discard/retain a whole group; only straddling groups need
    // per-sphere predicates. The eager path scans group minima; the lazy path
    // orders box lower bounds before computing individual metric priorities.
    constexpr int block_size = 32;
    ObstacleGroups temporary;
    const ObstacleGroups* grouped;
    if (depth == 0 && query_cache.spans[p.span] >= 0)
      grouped = &query_cache.groups[query_cache.spans[p.span]];
    else {
      temporary = obstacleGroups(ids, map);
      grouped = &temporary;
    }
    auto blocks = grouped->blocks;
    const bool uniform_radius = grouped->uniform_radius;
    LazyMetricOrder lazy(blocks,ids,map,metric,cm,min_scale,pad,uniform_radius);
    int next=lazy.next();
    std::vector<int> failed;
    while (next >= 0) {
      deadline.obstacle();
      const int selected_block=next/block_size;
      auto& selected = blocks[selected_block];
      selected.alive &= ~(1ull << (next - selected.begin));
      lazy.changed(selected_block);
      const int id = ids[next];
      const auto& sphere = map.obstacles()[id];
      const Vec3 delta = sphere.center - cm;
      Vec3 n;
      double h;
      auto separate = [&](const Vec3& direction) {
        const double length = direction.norm();
        if (length <= 1e-12) return false;
        n = direction / length;
        h = n.dot(sphere.center) - sphere.radius - pad;
        double support = -INFINITY;
        for (int j = 0; j <= D; ++j)
          support =
              std::max(support, n.dot(c[j]) + (m[j].transpose() * n).norm());
        return support < h - 1e-10;
      };
      bool separated =
          separate(rotation.transpose() * metricSphereNormal(
              rotation * delta, axes2, sphere.radius + pad));
      if (!separated) {
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
        separated = separate(delta - mm * (u * v));
      }
      ++metrics.obstacle_pairs;
      next = -1;
      if (separated) {
        add(p, n, h);
        const Vec3 absolute = n.cwiseAbs();
        for (size_t block_id=0;block_id<blocks.size();++block_id) {
          auto& b=blocks[block_id];
          if (!b.alive) continue;
          const auto before=b.alive;
          const double projection = n.dot(b.center), extent = absolute.dot(b.half);
          // Only a whole-group exclusion needs this accumulated roundoff
          // allowance; ambiguous groups use the original sphere predicate.
          const double guard = 1e-10 + 32 * std::numeric_limits<double>::epsilon() *
              (absolute.dot(b.center.cwiseAbs() + b.half) + b.max_radius + pad + std::abs(h));
          if (projection - extent - b.max_radius - pad >= h + guard) {
            b.alive = 0;
            b.nearest = -1;
            lazy.changed(int(block_id));
            continue;
          }
          if (projection + extent - b.min_radius - pad >= h - guard) {
            for (auto bits = b.alive; bits; bits &= bits - 1) {
              const int j = b.begin + __builtin_ctzll(bits);
              const auto& other = map.obstacles()[ids[j]];
              ++metrics.obstacle_pairs;
              if (n.dot(other.center) - other.radius - pad >= h)
                b.alive &= ~(1ull << (j - b.begin));
            }
          }
          if(before!=b.alive)lazy.changed(int(block_id));
        }
      } else {
        failed.push_back(id);
      }
      next=lazy.next();
    }
    timings.priority_evaluations+=lazy.evaluations;
    ++metrics.certificate_leaves;
    if (!failed.empty()) {
      if (depth >= options.certificate_depth)
        throw Failure{Status::InsufficientClearance,
                      "polynomial patch cannot be separated from an inflated obstacle"};
      const auto halves = p.halves();
      pairCuts(halves.first, failed, depth + 1);
      pairCuts(halves.second, failed, depth + 1);
    }
  }
  void build(const Geometry& g,const CoverLayout& layout) {
    StageTimer total(profile);
    const double pad=options.robot_radius+options.safety_margin+options.certificate_margin;
    for(int i=0;i<int(g.pieces());++i) {
      deadline.check();const Patch p{i,0,1,g.curve<D>(i)};
      for(int axis=0;axis<3;++axis){limit(p,Vec3::Unit(axis),bounds.upper[axis]-pad);limit(p,-Vec3::Unit(axis),-bounds.lower[axis]-pad);}
      const auto& span=layout.spans[i];
      for(size_t j=span.begin;j<span.end;++j) {
        const auto& box=layout.parts[j];
        const Patch part{i,box.begin,box.end,restrictCurve<D>(p.controls,box.begin,box.end)};
        for(int axis=0;axis<3;++axis){limit(part,Vec3::Unit(axis),box.bounds.upper[axis]);limit(part,-Vec3::Unit(axis),-box.bounds.lower[axis]);}
      }
      ++timings.reused_cover_spans;
      StageTimer timer(profile);std::vector<int> scratch;
      const auto& ids=query_cache.query(i,span.lower-Vec3::Constant(pad),span.upper+Vec3::Constant(pad),map,scratch);
      timings.query_ms+=timer.elapsed();timer.restart();pairCuts(p,ids,0);timings.separation_ms+=timer.elapsed();
    }
    timings.total_ms+=total.elapsed();
  }
};
inline std::vector<detail::SupportPlane> mapCover(const Geometry& g,const ObstacleIndex& map,
    const Bounds& bounds,const GeometryOptions& options,QueryCache& cache,CoverMetrics& metrics,
    Deadline& deadline,const CoverLayout& layout,bool profile=false) {
  if(g.degree==3){MapCover<3> c(map,bounds,options,cache,metrics,deadline,profile);c.build(g,layout);return c.planes;}
  MapCover<5> c(map,bounds,options,cache,metrics,deadline,profile);c.build(g,layout);return c.planes;
}
}
