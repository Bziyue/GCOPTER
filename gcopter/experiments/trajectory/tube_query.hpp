// SPDX-License-Identifier: MIT
#pragma once
#include "tube_reference.hpp"
#include "gcopter/corridor_evaluator.hpp"
#include <chrono>
#include <cstring>
#include <cstdint>
#include <limits>

namespace tube_query {
using Clock=std::chrono::steady_clock;
struct Stats {
  bool enabled=false;
  double objective_ms=0,integral_ms=0,dynamics_ms=0,coefficient_ms=0,tube_ms=0,section_ms=0;
  uint64_t queries=0,query_hits=0,section_calls=0,section_hits=0,plane_evaluations=0,fallbacks=0;
};
struct Timer {
  double* destination=nullptr;Clock::time_point start;
  Timer(bool enabled,double& value){if(enabled){destination=&value;start=Clock::now();}}
  ~Timer(){if(destination)*destination+=std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
};
inline uint64_t bits(double x){uint64_t result;std::memcpy(&result,&x,sizeof(x));return result;}
inline uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}

/** Fixed-size exact memoization, no warm-started roots or tolerance-based cache keys.
 * Cache collisions only replace entries; neither query order nor optimizer history changes the function.
 * New candidate positions are still inverted from the original progress hint.
 */
class Query {
  const Corridor& corridor_;
  PreparedCorridor prepared_;
  bool fast_,horner_;
  size_t cursor_=0;
  struct Geometry {bool valid=false;double u=0;FirstSection section;Vec3 normal,normal_first;};
  struct Answer {bool valid=false;std::array<double,4> key{};TubeValue value;};
  std::vector<Geometry> geometry_{2048};
  std::vector<Answer> answers_{1024};
  const Geometry& geometryAt(double u) {
    Timer timer(stats.enabled,stats.section_ms);++stats.section_calls;
    auto& item=geometry_[mix(bits(u))&(geometry_.size()-1)];
    if(item.valid&&item.u==u){++stats.section_hits;return item;}
    item.section=horner_?prepared_.firstUnchecked(u,cursor_):prepared_.firstCompatibleUnchecked(u,cursor_);item.valid=true;item.u=u;
    const auto& s=item.section;
    const Vec3 cross=s.shape.col(0).cross(s.shape.col(1));
    const Vec3 dcross=s.shape_first.col(0).cross(s.shape.col(1))+s.shape.col(0).cross(s.shape_first.col(1));
    const double norm=cross.norm();item.normal=cross/norm;
    item.normal_first=(dcross-item.normal*item.normal.dot(dcross))/norm;
    return item;
  }
TubeValue compute(const Vec3& p, double hint) {
  double u=hint,f=0,fu=0;FirstSection section;double last_u=std::numeric_limits<double>::quiet_NaN();Vec3 normal,normal_u;
  auto plane=[&](double x) {
    last_u=x;const auto& frame=geometryAt(x);section=frame.section;++stats.plane_evaluations;
    normal=frame.normal;normal_u=frame.normal_first;
    f=normal.dot(p-section.center);
    fu=normal_u.dot(p-section.center)-normal.dot(section.first);
    return f;
  };
  for(int i=0;i<16;++i) {
    if(last_u!=u)plane(u);if(std::abs(f)<1e-8)break;
    if(std::abs(fu)<1e-10)break;
    double next=std::clamp(u-std::clamp(f/fu,-.1,.1),0.,1.);
    if(next==u)break;u=next;
  }
  if(last_u!=u)plane(u);
  if(std::abs(f)>1e-7 && u>0 && u<1) {
    ++stats.fallbacks;
    // Rare fallback: choose the nearest sign-changing knot interval, then bisect.
    double best=1e10,lo=0,hi=0,fl=0;
    const auto& knots=corridor_.knots();double previous=plane(knots.front());
    for(size_t i=1;i<knots.size();++i) {
      double current=plane(knots[i]);double dist=std::abs(.5*(knots[i-1]+knots[i])-hint);
      if(previous*current<=0 && dist<best){best=dist;lo=knots[i-1];hi=knots[i];fl=previous;}
      previous=current;
    }
    if(best<1e10) {
      for(int i=0;i<40;++i){u=.5*(lo+hi);double fm=plane(u);if(std::abs(fm)<1e-9)break;if(fl*fm<=0)hi=u;else{lo=u;fl=fm;}}
    }
    if(last_u!=u)plane(u);
  }
  TubeValue out;out.u=u;out.plane=f;out.grad_plane=normal;
  const bool root=std::abs(f)<1e-7&&std::abs(fu)>1e-10;
  out.valid=root||u==0||u==1;
  Vec3 d=p-section.center;
  const Mat32& M=section.shape;const Mat32& Md=section.shape_first;
  Eigen::Matrix2d gram=M.transpose()*M;
  Eigen::LDLT<Eigen::Matrix2d> factor(gram);
  Eigen::Matrix<double,2,3> inv=factor.solve(M.transpose());
  Eigen::Vector2d z=inv*d;
  double q=z.squaredNorm();
  if(q>1) {
    // Radius in the section plane. This also handles endpoint cap extensions.
    Vec3 tangent=d-normal*normal.dot(d);double r=tangent.norm(),sq=std::sqrt(q);
    out.radial=r*(1-1/sq);
    out.grad_radial=(1-1/sq)*tangent/r+r/(q*sq)*(inv.transpose()*z);
    if(root) {
      Eigen::Vector2d zu=factor.solve(Md.transpose()*d-M.transpose()*section.first-
          (Md.transpose()*M+M.transpose()*Md)*z);
      double ru=-tangent.dot(section.first)/r;
      double derivative=ru*(1-1/sq)+r/(q*sq)*z.dot(zu);
      out.grad_radial-=normal*(derivative/fu);
    }
  }
  if(root){out.plane=0;out.grad_plane.setZero();}
  return out;
}

 public:
  Stats stats;
  Query(const Corridor& corridor,bool fast,bool profile=false,bool horner=false):corridor_(corridor),prepared_(corridor),fast_(fast),horner_(horner){stats.enabled=profile;}
  void reset(){const bool enabled=stats.enabled;stats=Stats{};stats.enabled=enabled;for(auto& v:geometry_)v.valid=false;for(auto& v:answers_)v.valid=false;}
  TubeValue operator()(const Vec3& p,double hint) {
    Timer timer(stats.enabled,stats.tube_ms);++stats.queries;
    if(!fast_)return referenceTubeValue(corridor_,p,hint,[&](double u){
      Timer section_timer(stats.enabled,stats.section_ms);++stats.section_calls;++stats.plane_evaluations;return corridor_.section(u);
    },&stats.fallbacks);
    std::array<double,4> key{p.x(),p.y(),p.z(),hint};
    uint64_t h=0;for(double value:key)h=mix(h^bits(value));
    auto& item=answers_[h&(answers_.size()-1)];
    if(item.valid&&item.key==key){++stats.query_hits;return item.value;}
    auto value=compute(p,hint);item.valid=true;item.key=key;item.value=value;return value;
  }
};
} // namespace tube_query
