// SPDX-License-Identifier: MIT
// Experimental linear-support variant of the production sphere separator.
#pragma once
#include "minkowski_geometry.hpp"
namespace gcopter::direct::minkowski {
using detail::ObstacleGroups;
using detail::obstacleGroups;
using detail::metricSphereNormal;
using detail::middle;
struct Certificates {
  const Templates& bank;
  const Options& settings;
  const std::vector<V6>& parameters;
  const ObstacleIndex& map;
  const Bounds& bounds;
  const direct::Options& options;
  const std::vector<LocalBox>& local_boxes;
  const Stop& stop;
  Metrics& metrics;
  QueryCache& query_cache;
  bool use_shape_metric = false;
  std::vector<LinearCut> cuts;
  struct Plane { Patch patch; Vec3 normal; double bound; };
  std::vector<Plane> planes;
  Certificates(const ObstacleIndex& m, const Bounds& b, const Options& o,
               const Stop& s, Metrics& me, const std::vector<LocalBox>& boxes,
               QueryCache& cache, const Templates& t, const std::vector<V6>& x)
      : bank(t), settings(o), parameters(x), map(m), bounds(b), options(o.geometry),
        local_boxes(boxes), stop(s), metrics(me), query_cache(cache) {}
  void append(LinearCut cut) {
    if (cuts.size() >= options.max_constraints)
      throw Failure{Status::Capacity, "Minkowski constraint capacity exceeded"};
    cuts.push_back(std::move(cut));
  }
  void add(const Patch& p, const Vec3& normal, double bound,
           const std::vector<Node>& nodes, int level = -1) {
    if (level < 0) {
      if(settings.adapt_orientation && !use_shape_metric)
        planes.push_back({p,normal,bound});
      level = settings.subdivisions;
    }
    if (level && (nodes[p.span].frame-nodes[p.span+1].frame).squaredNorm()>1e-20) {
      const auto halves = detail::split(p);
      add(halves.first, normal, bound, nodes, level-1);
      add(halves.second, normal, bound, nodes, level-1);
      return;
    }
    const V2 na = nodes[p.span].frame.transpose()*normal;
    const V2 nb = nodes[p.span+1].frame.transpose()*normal;
    if(settings.product_bounds && (nodes[p.span].frame-nodes[p.span+1].frame).squaredNorm()>1e-20) {
      for(auto cut:productRows(p,nodes,bank,normal,bound)) append(cut);
      return;
    }
    const Weights h = supportCoefficients(p,nodes,bank,normal);
    for (int j=0;j<6;) {
      const double w=p.w[j];
      double rhs=bound-normal.dot(p.reference[j]);
      int k=j+1;
      while(k<6 && std::abs(p.w[k]-w)<1e-14) {
        rhs=std::min(rhs,bound-normal.dot(p.reference[k]));
        ++k;
      }
      LinearCut cut;
      cut.span=p.span;
      cut.a << (1-w)*na, (1-w)*h;
      cut.b << w*nb, w*h;
      cut.h=rhs;
      const double reach=options.max_offset*(cut.a.head<2>().norm()+cut.b.head<2>().norm())+
                         options.max_radius*(cut.a.tail<4>().sum()+cut.b.tail<4>().sum());
      if(rhs<reach+options.numerical_guard) append(cut);
      j=k;
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
    const SectionProjection projection(p,nodes,parameters,bank);
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
        const double support = settings.product_bounds
            ? productSupportBound(p,nodes,parameters,bank,n)
            : supportBound(p,nodes,parameters,bank,n);
        return support < h - 1e-10;
      };
      bool separated = use_shape_metric &&
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
      if (!separated) separated = separate(projection.direction(delta));
      ++metrics.obstacle_pairs;
      next = -1;
      if (separated) {
        add(p, n, h, nodes);
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
          LinearCut cut;
          cut.span=i;
          cut.a.head<2>()=dw*(nodes[i].frame.transpose()*normal);
          cut.b.head<2>()=-dw*(nodes[i+1].frame.transpose()*normal);
          cut.h=h;
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
        extent[axis] = (bank.count * options.max_radius + options.max_offset) *
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
      pairCuts(p, query_cache.query(i, query_lo, query_hi, map, ids), nodes, 0);
    }
    canonicalize(nodes.size());
  }
  void canonicalize(std::size_t node_count) {
    // Normalize and merge identical linear rows; unlike the ellipse polar
    // filter this remains valid when support coefficients bound a whole frame
    // interval, not just one physical section.
    for(auto& cut:cuts) {
      if(cut.a.squaredNorm()==0) {
        ++cut.span; cut.a=cut.b; cut.b.setZero();
      }
      const double scale=std::sqrt(cut.a.squaredNorm()+cut.b.squaredNorm());
      cut.a/=scale; cut.b/=scale; cut.h/=scale;
    }
    auto key=[](const LinearCut& c) {
      std::array<double,13> k;
      k[0]=c.span;
      for(int j=0;j<6;++j) { k[1+j]=c.a[j]; k[7+j]=c.b[j]; }
      return k;
    };
    std::sort(cuts.begin(),cuts.end(),[&](const LinearCut& a,const LinearCut& b){ return key(a)<key(b); });
    std::vector<LinearCut> unique;
    unique.reserve(cuts.size());
    for(const auto& cut:cuts) {
      if(!unique.empty() && key(cut)==key(unique.back()))
        unique.back().h=std::min(unique.back().h,cut.h);
      else unique.push_back(cut);
    }
    for(auto& cut:unique) if(cut.span==int(node_count)-1) {
      --cut.span; cut.b=cut.a; cut.a.setZero();
    }
    cuts.swap(unique);
    metrics.constraints=cuts.size();
  }
  void adapt(std::vector<Node>& nodes) {
    // An origin-centered inscribed ellipse has Q=(2*M)^-1, where M is the
    // D-optimal moment matrix of polar half-plane normals. Only its direction
    // is needed: this small local solve does not constrain later center shifts.
    std::vector<std::vector<V2>> polar(nodes.size());
    for(const auto& plane:planes) {
      const auto range=std::minmax_element(plane.patch.w.begin(),plane.patch.w.end());
      for(int side=0;side<2;++side) {
        if((side==0 && *range.first>1e-12) || (side==1 && *range.second<1-1e-12)) continue;
        const int i=plane.patch.span+side;
        const double gap=plane.bound-plane.normal.dot(nodes[i].reference);
        const V2 n=nodes[i].frame.transpose()*plane.normal;
        if(gap>0 && n.squaredNorm()>1e-20) polar[i].push_back(n/gap);
      }
    }
    std::vector<M2> rotations(nodes.size(),M2::Identity());
    std::vector<double> angles(nodes.size(),0);
    const double period=std::acos(-1.)/bank.count;
    double previous=0;
    for(std::size_t i=0;i<nodes.size();++i) {
      auto& q=polar[i];
      std::sort(q.begin(),q.end(),[](const V2& a,const V2& b){return std::make_pair(a[0],a[1])<std::make_pair(b[0],b[1]);});
      q.erase(std::unique(q.begin(),q.end(),[](const V2& a,const V2& b){return a==b;}),q.end());
      if(q.size()<3) continue;
      M2 moment=M2::Zero();
      for(const auto& v:q) moment+=v*v.transpose();
      moment/=q.size();
      for(int it=0;it<24;++it) {
        const M2 inverse=moment.inverse();
        double maximum=0; int selected=0;
        for(int j=0;j<int(q.size());++j) {
          const double score=q[j].dot(inverse*q[j]);
          if(score>maximum) {maximum=score;selected=j;}
        }
        if(maximum<=2.001) break;
        const double alpha=(maximum-2)/(2*(maximum-1));
        moment=(1-alpha)*moment+alpha*(q[selected]*q[selected].transpose());
      }
      Eigen::SelfAdjointEigenSolver<M2> eig(moment);
      double angle=previous;
      if(eig.eigenvalues()[1]>1.1*eig.eigenvalues()[0]) {
        const V2 axis=eig.eigenvectors().col(0);
        const double raw=std::atan2(axis[1],axis[0])-settings.angle;
        angle=raw+std::round((previous-raw)/period)*period;
      }
      previous=angle;
      angles[i]=angle;
    }
    if(settings.orientation_smoothing>0) {
      // Solve min ||theta-target||^2 + lambda*sum (theta[i+1]-theta[i])^2.
      // This reduces the frame variation paid by the continuous support bound.
      const double lambda=settings.orientation_smoothing;
      std::vector<double> diagonal(nodes.size(),1+2*lambda);
      diagonal.front()=diagonal.back()=1+lambda;
      for(std::size_t i=1;i<nodes.size();++i) {
        const double multiplier=-lambda/diagonal[i-1];
        diagonal[i]-=multiplier*(-lambda);
        angles[i]-=multiplier*angles[i-1];
      }
      angles.back()/=diagonal.back();
      for(int i=int(nodes.size())-2;i>=0;--i) angles[i]=(angles[i]+lambda*angles[i+1])/diagonal[i];
    }
    for(std::size_t i=0;i<nodes.size();++i) {
      rotations[i]=Eigen::Rotation2Dd(angles[i]).toRotationMatrix();
      nodes[i].frame=(nodes[i].frame*rotations[i]).eval();
    }
    std::vector<LinearCut> progress;
    for(auto cut:cuts) if(cut.a.tail<4>().squaredNorm()+cut.b.tail<4>().squaredNorm()==0) {
      cut.a.head<2>()=rotations[cut.span].transpose()*cut.a.head<2>();
      cut.b.head<2>()=rotations[cut.span+1].transpose()*cut.b.head<2>();
      progress.push_back(cut);
    }
    cuts.clear();
    for(const auto& plane:planes)
      add(plane.patch,plane.normal,plane.bound,nodes,settings.subdivisions);
    cuts.insert(cuts.end(),progress.begin(),progress.end());
    canonicalize(nodes.size());
  }
};
}
