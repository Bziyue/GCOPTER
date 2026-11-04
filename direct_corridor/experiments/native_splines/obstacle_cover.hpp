// SPDX-License-Identifier: MIT
#pragma once
#include "cubic_certificates.hpp"
#include "../../src/hermite_optimizer.hpp"
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
struct CoverMetrics {
  double query_ms=0, separation_ms=0, total_ms=0;
  Metrics primitives;
};
// The sphere separator uses the production BVH, metricSphereNormal and spatial
// packets. Only the polynomial control count and emission of raw planes differ.
template<int D> struct MapCover {
  using Patch=GeometryPatch<D>;
  const ObstacleIndex& map;const Bounds& bounds;const Options& options;
  QueryCache& query_cache;CoverMetrics& timings;Metrics& metrics;
  const std::function<void()>& stop;
  std::vector<detail::SupportPlane> planes;
  MapCover(const ObstacleIndex& m,const Bounds& b,const Options& o,QueryCache& cache,
           CoverMetrics& t,const std::function<void()>& check)
    :map(m),bounds(b),options(o),query_cache(cache),timings(t),metrics(t.primitives),stop(check){}
  void add(const Patch& p,const Vec3& n,double h) {planes.push_back({p.span,p.begin,p.end,n,h});}
  void limit(const Patch& p,const Vec3& n,double h,int depth=0) {
    if(supportBound<D>(p.controls,n,h)<=1e-9){add(p,n,h);return;}
    if(depth==options.certificate_depth)throw Failure{Status::InsufficientClearance,"continuous local/world bounds failed"};
    const auto halves=p.halves();limit(halves.first,n,h,depth+1);limit(halves.second,n,h,depth+1);
  }
  void pairCuts(const Patch& p, const std::vector<int>& ids,
                int depth) {
    stop();
    if (ids.empty()) {
      ++metrics.certificate_leaves;
      return;
    }
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
    // per-sphere predicates. Per-group minima avoid a heap and array compaction.
    struct Candidate { int id; double distance; };
    using Block = ObstacleGroups::Block;
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
    std::vector<Candidate> remaining;
    remaining.reserve(ids.size());
    for (int id : ids)
      remaining.push_back({id, (metric * (map.obstacles()[id].center - cm)).squaredNorm()});
    if (!uniform_radius) {
      const double inverse_scale = 1 / min_scale;
      for (auto& candidate : remaining)
        candidate.distance = std::sqrt(candidate.distance) -
            (map.obstacles()[candidate.id].radius + pad) * inverse_scale;
    }
    auto better = [&](int a, int b) {
      return b < 0 || remaining[a].distance < remaining[b].distance ||
          (remaining[a].distance == remaining[b].distance && remaining[a].id < remaining[b].id);
    };
    auto nearest = [&](Block& b) {
      b.nearest = -1;
      for (auto bits = b.alive; bits; bits &= bits - 1) {
        const int j = b.begin + __builtin_ctzll(bits);
        if (better(j, b.nearest)) b.nearest = j;
      }
    };
    int next = -1;
    for (auto& b : blocks) {
      nearest(b);
      if (better(b.nearest, next)) next = b.nearest;
    }
    std::vector<int> failed;
    unsigned processed = 0;
    while (next >= 0) {
      if ((++processed & 31u) == 0) stop();
      auto& selected = blocks[next / block_size];
      selected.alive &= ~(1ull << (next - selected.begin));
      nearest(selected);
      const int id = remaining[next].id;
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
        for (auto& b : blocks) {
          if (!b.alive) continue;
          const double projection = n.dot(b.center), extent = absolute.dot(b.half);
          // Only a whole-group exclusion needs this accumulated roundoff
          // allowance; ambiguous groups use the original sphere predicate.
          const double guard = 1e-10 + 32 * std::numeric_limits<double>::epsilon() *
              (absolute.dot(b.center.cwiseAbs() + b.half) + b.max_radius + pad + std::abs(h));
          if (projection - extent - b.max_radius - pad >= h + guard) {
            b.alive = 0;
            b.nearest = -1;
            continue;
          }
          if (projection + extent - b.min_radius - pad >= h - guard) {
            b.nearest = -1;
            for (auto bits = b.alive; bits; bits &= bits - 1) {
              const int j = b.begin + __builtin_ctzll(bits);
              const auto& other = map.obstacles()[remaining[j].id];
              ++metrics.obstacle_pairs;
              if (n.dot(other.center) - other.radius - pad >= h)
                b.alive &= ~(1ull << (j - b.begin));
              else if (better(j, b.nearest)) b.nearest = j;
            }
          }
          if (b.nearest >= 0 && better(b.nearest, next)) next = b.nearest;
        }
      } else {
        failed.push_back(id);
        for (const auto& b : blocks)
          if (b.nearest >= 0 && better(b.nearest, next)) next = b.nearest;
      }
    }
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
  void build(const Geometry& g,const std::vector<Vec3>& seed) {
    using Clock=std::chrono::steady_clock;
    auto elapsed=[](auto t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();};
    const auto start=Clock::now();
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
    for(int i=0;i<int(g.pieces());++i) {
      stop();const Patch p{i,0,1,g.curve<D>(i)};
      for(int axis=0;axis<3;++axis){limit(p,Vec3::Unit(axis),bounds.upper[axis]-pad);limit(p,-Vec3::Unit(axis),-bounds.lower[axis]-pad);}
      const double s=g.breaks[i]*length,e=g.breaks[i+1]*length,h=e-s;
      Vec3 lo=Vec3::Constant(INFINITY),hi=-lo;
      for(const auto& box:boxes) {
        const double from=std::clamp((box.begin-s)/h,0.,1.),to=std::clamp((box.end-s)/h,0.,1.);
        if(to<=from)continue;
        const Patch part{i,from,to,restrictCurve<D>(p.controls,from,to)};
        for(int axis=0;axis<3;++axis){limit(part,Vec3::Unit(axis),box.bounds.upper[axis]);limit(part,-Vec3::Unit(axis),-box.bounds.lower[axis]);}
        lo=lo.cwiseMin(box.bounds.lower);hi=hi.cwiseMax(box.bounds.upper);
      }
      auto timer=Clock::now();std::vector<int> scratch;
      const auto& ids=query_cache.query(i,lo-Vec3::Constant(pad),hi+Vec3::Constant(pad),map,scratch);
      timings.query_ms+=elapsed(timer);timer=Clock::now();pairCuts(p,ids,0);timings.separation_ms+=elapsed(timer);
    }
    timings.total_ms+=elapsed(start);
  }
};
inline std::vector<detail::SupportPlane> mapCover(const Geometry& g,const std::vector<Vec3>& seed,
    const ObstacleIndex& map,const Bounds& bounds,const Options& options,QueryCache& cache,
    CoverMetrics& metrics,const std::function<void()>& stop) {
  if(g.degree==3){MapCover<3> c(map,bounds,options,cache,metrics,stop);c.build(g,seed);return c.planes;}
  MapCover<5> c(map,bounds,options,cache,metrics,stop);c.build(g,seed);return c.planes;
}
}
