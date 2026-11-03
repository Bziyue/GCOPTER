// SPDX-License-Identifier: MIT
#pragma once
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>
#include "minkowski.hpp"
#include "../src/hermite_optimizer.hpp"

namespace gcopter::direct::minkowski {
using V6 = Eigen::Matrix<double, 6, 1>;
using H6 = Eigen::Matrix<double, 6, 6>;
using M2 = Eigen::Matrix2d;
using V2 = Eigen::Vector2d;
using detail::Node;
using detail::Patch;
using detail::Failure;
using detail::Stop;
using detail::LocalBox;
using detail::QueryCache;

struct Templates {
  int count;
  std::array<M2, 4> q, root;
  Eigen::Matrix4d mixed = Eigen::Matrix4d::Zero();
  explicit Templates(const Options& o) : count(o.templates) {
    const double pi = std::acos(-1.);
    const std::array<double, 4> angles{0, pi/2, pi/4, 3*pi/4};
    for (int k = 0; k < 4; ++k) {
      const M2 rot = Eigen::Rotation2Dd(angles[k] + o.angle).toRotationMatrix();
      root[k] = rot * V2(1., o.thinness).asDiagonal() * rot.transpose();
      q[k] = root[k] * root[k];
    }
    // Mixed area is independent of a common template rotation. Periodic
    // quadrature is verified independently at doubled resolution in tests.
    constexpr int samples = 4096;
    for (int j = 0; j < samples; ++j) {
      const double angle = 2*pi*j/samples;
      const V2 n(std::cos(angle), std::sin(angle)), t(-n[1], n[0]);
      Weights h = Weights::Zero(), d = h;
      for (int k = 0; k < count; ++k) {
        h[k] = std::sqrt(n.dot(q[k] * n));
        d[k] = t.dot(q[k] * n)/h[k];
      }
      mixed.noalias() += (pi/samples) * (h*h.transpose()-d*d.transpose());
    }
  }
  Weights support(const V2& n) const {
    Weights h = Weights::Zero();
    for (int k = 0; k < count; ++k) h[k] = std::sqrt(n.dot(q[k]*n));
    return h;
  }
  void proxy(std::vector<Node>& nodes, const std::vector<V6>& x) const {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      M2 m = M2::Zero();
      for (int k = 0; k < count; ++k) m += x[i][2+k]*root[k];
      nodes[i].x << x[i][0], x[i][1], m(0,0), m(0,1), m(1,1);
    }
  }
};

struct LinearCut {
  int span;
  V6 a = V6::Zero(), b = V6::Zero();
  double h;
};
inline double value(const LinearCut& c, const std::vector<V6>& x) {
  return c.a.dot(x[c.span])+c.b.dot(x[c.span+1]);
}

// A convex support function attains its maximum over the affine frame path
// at an endpoint. Nonnegative weights turn this into a linear bound on every
// Bernstein control. Subdivision tightens the fixed direction coefficients.
inline Weights supportCoefficients(const Patch& p, const std::vector<Node>& nodes,
                                   const Templates& bank, const Vec3& n) {
  const Mat32& a = nodes[p.span].frame;
  const Mat32& b = nodes[p.span+1].frame;
  const auto range = std::minmax_element(p.w.begin(), p.w.end());
  const V2 na = a.transpose()*n, nb = b.transpose()*n;
  return bank.support((1-*range.first)*na+*range.first*nb).cwiseMax(
      bank.support((1-*range.second)*na+*range.second*nb));
}
inline double supportBound(const Patch& p, const std::vector<Node>& nodes,
                           const std::vector<V6>& x, const Templates& bank,
                           const Vec3& normal) {
  const auto& a = nodes[p.span];
  const auto& b = nodes[p.span+1];
  const Weights h = supportCoefficients(p, nodes, bank, normal);
  const double va = normal.dot(a.frame*x[p.span].head<2>()) + h.dot(x[p.span].tail<4>());
  const double vb = normal.dot(b.frame*x[p.span+1].head<2>()) + h.dot(x[p.span+1].tail<4>());
  double upper = -INFINITY;
  for (int j=0;j<6;++j)
    upper = std::max(upper, normal.dot(p.reference[j])+(1-p.w[j])*va+p.w[j]*vb);
  return upper;
}

inline std::array<LinearCut,11> productRows(const Patch& p,const std::vector<Node>& nodes,
                                          const Templates& bank,const Vec3& normal,double bound) {
  const V2 na=nodes[p.span].frame.transpose()*normal, nb=nodes[p.span+1].frame.transpose()*normal;
  std::array<Weights,6> supports;
  for(int i=0;i<6;++i) supports[i]=bank.support((1-p.w[i])*na+p.w[i]*nb);
  std::array<LinearCut,11> rows;
  for(auto& row:rows) {row.span=p.span;row.h=bound;}
  constexpr double c5[]={1,5,10,10,5,1}, c10[]={1,10,45,120,210,252,210,120,45,10,1};
  for(int i=0;i<6;++i) for(int j=0;j<6;++j) {
    const double coefficient=c5[i]*c5[j]/c10[i+j];
    auto& row=rows[i+j];
    row.h-=coefficient*normal.dot(p.reference[i]);
    row.a.head<2>()+=coefficient*(1-p.w[i])*na;
    row.b.head<2>()+=coefficient*p.w[i]*nb;
    row.a.tail<4>()+=coefficient*(1-p.w[j])*supports[i];
    row.b.tail<4>()+=coefficient*p.w[j]*supports[i];
  }
  return rows;
}
inline double productSupportBound(const Patch& p,const std::vector<Node>& nodes,
                                  const std::vector<V6>& x,const Templates& bank,const Vec3& normal) {
  double upper=-INFINITY;
  for(const auto& row:productRows(p,nodes,bank,normal,0)) upper=std::max(upper,value(row,x)-row.h);
  return upper;
}

// Exact smooth boundary support jets in an orthonormal section coordinate.
// Also used with transformed template matrices by the separator.
struct SupportJet {
  double h=0, first=0, radius=0;
  V2 point=V2::Zero();
};
inline SupportJet supportJet(const std::array<M2,4>& q,int count,
                             const Weights& weights,double angle) {
  const V2 n(std::cos(angle),std::sin(angle)),t(-n[1],n[0]);
  SupportJet out;
  for(int k=0;k<count;++k) {
    const V2 a=q[k]*n;
    const double h=std::sqrt(n.dot(a)), weight=weights[k]/h;
    out.h+=weights[k]*h;
    out.first+=weight*t.dot(a);
    out.radius+=weight*q[k].determinant()/(h*h);
    out.point+=weight*a;
  }
  return out;
}
struct SectionProjection {
  Mat32 basis;
  std::array<M2,4> q;
  Weights weights;
  int count;
  SectionProjection(const Mat32& f,const Templates& bank,const Weights& r)
      :weights(r),count(bank.count) {
    basis.col(0)=f.col(0).normalized();
    basis.col(1)=f.col(0).cross(f.col(1)).normalized().cross(basis.col(0));
    const M2 transform=basis.transpose()*f;
    for(int k=0;k<count;++k) q[k]=transform*bank.q[k]*transform.transpose();
  }
  SectionProjection(const Patch& p,const std::vector<Node>& nodes,
                    const std::vector<V6>& x,const Templates& bank) :count(bank.count) {
    const double w=detail::middle(p.w);
    const Mat32 f=(1-w)*nodes[p.span].frame+w*nodes[p.span+1].frame;
    basis.col(0)=f.col(0).normalized();
    basis.col(1)=f.col(0).cross(f.col(1)).normalized().cross(basis.col(0));
    const M2 transform=basis.transpose()*f;
    for(int k=0;k<count;++k) q[k]=transform*bank.q[k]*transform.transpose();
    weights=(1-w)*x[p.span].tail<4>()+w*x[p.span+1].tail<4>();
  }
  Vec3 direction(const Vec3& delta) const {
    const V2 z=basis.transpose()*delta;
    if(z.squaredNorm()<1e-28) return delta;
    const double center=std::atan2(z[1],z[0]), halfpi=.5*std::acos(-1.);
    double lo=center-halfpi,hi=center+halfpi,angle=center;
    // First locate the supporting normal of the ray through z. This provides
    // either an interior projection or a strictly separating starting normal.
    for(int it=0;it<64;++it) {
      const V2 n(std::cos(angle),std::sin(angle)),t(-n[1],n[0]);
      const auto s=supportJet(q,count,weights,angle);
      const double a=n.dot(z),g=t.dot(z)*s.h-a*s.first;
      if(std::abs(g)<2e-13*z.norm()*(s.h+std::abs(s.first)) || hi-lo<1e-13) break;
      if(g>0) lo=angle; else hi=angle;
      const double next=angle+g/(a*s.radius);
      angle=std::isfinite(next) && next>lo && next<hi ? next:.5*(lo+hi);
    }
    auto s=supportJet(q,count,weights,angle);
    if(V2(std::cos(angle),std::sin(angle)).dot(z)<=s.h)
      return delta-basis*z;
    // Euclidean projection maximizes d=n.z-h(n). On the visible arc d>0,
    // d''=-(d+curvature_radius)<0. Backtracking stays on that concave arc.
    for(int it=0;it<32;++it) {
      const V2 n(std::cos(angle),std::sin(angle)),t(-n[1],n[0]);
      const double d=n.dot(z)-s.h,g=t.dot(z)-s.first;
      if(std::abs(g)<1e-12*(1+z.norm())) break;
      const double step=g/(d+s.radius);
      double alpha=1;
      bool accepted=false;
      for(int ls=0;ls<20;++ls) {
        const double next=angle+alpha*step;
        const auto candidate=supportJet(q,count,weights,next);
        const double after=V2(std::cos(next),std::sin(next)).dot(z)-candidate.h;
        if(after>=d+1e-3*alpha*step*g) {
          angle=next; s=candidate; accepted=true; break;
        }
        alpha*=.5;
      }
      if(!accepted) break;
    }
    // This direction is only a proposal. The caller checks the whole patch
    // against the original closed sphere, so solver accuracy cannot weaken it.
    return delta-basis*s.point;
  }
};

inline bool regular(const Patch& p, const std::vector<Node>& nodes) {
  std::array<Vec3, 6> c;
  std::array<Mat32, 6> f;
  const auto& a = nodes[p.span];
  const auto& b = nodes[p.span+1];
  for (int j=0;j<6;++j) {
    c[j] = p.reference[j]+(1-p.w[j])*a.frame*a.x.head<2>()+p.w[j]*b.frame*b.x.head<2>();
    f[j] = (1-p.w[j])*a.frame+p.w[j]*b.frame;
  }
  std::array<double,15> d{};
  for(int i=0;i<5;++i) for(int j=0;j<6;++j) for(int k=0;k<6;++k)
    d[i+j+k] += 5*(c[i+1]-c[i]).dot(f[j].col(0).cross(f[k].col(1)))*
      detail::choose(4,i)*detail::choose(5,j)*detail::choose(5,k)/detail::choose(14,i+j+k);
  double scale=0;
  for(double v:d) scale=std::max(scale,std::abs(v));
  return scale>1e-18 && detail::positive(d,std::max(1e-18,scale*1e-10));
}
}  // namespace gcopter::direct::minkowski
