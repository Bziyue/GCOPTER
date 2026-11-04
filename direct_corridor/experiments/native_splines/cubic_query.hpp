// SPDX-License-Identifier: MIT
#pragma once
#include "geometry.hpp"
#include "../../../gcopter/experiments/trajectory/tube_query.hpp"
#include <memory>
namespace gcopter::direct::native {
// Four native coefficients per span, with cubic Horner and a constant-time
// span lookup for the uniform native generator. No quintic elevation here.
class PreparedCubic {
  using Matrix=Eigen::Matrix3d;
  struct Piece {std::array<Matrix,4> power;double inverse_width;};
  std::vector<double> knots_;
  std::vector<Piece> pieces_;
  bool uniform_=true;
public:
  explicit PreparedCubic(const Geometry& g):knots_(g.breaks),pieces_(g.pieces()) {
    if(g.degree!=3||g.pieces()==0)throw std::invalid_argument("native cubic query requires degree three");
    for(size_t i=0;i<g.pieces();++i) {
      std::array<Matrix,4> b;
      for(int j=0;j<4;++j)b[j]=Eigen::Map<const Matrix>(g.controls[4*i+j].data());
      auto& p=pieces_[i];p.power={b[0],3*(b[1]-b[0]),3*(b[2]-2*b[1]+b[0]),b[3]-3*b[2]+3*b[1]-b[0]};
      p.inverse_width=1/(knots_[i+1]-knots_[i]);
      uniform_&=std::abs(knots_[i]-double(i)/g.pieces())<1e-12;
    }
  }
  FirstSection firstUnchecked(double u,size_t& cursor)const {
    if(!(u>=knots_[cursor] && (u<knots_[cursor+1]||(cursor+1==pieces_.size()&&u==1)))) {
      if(uniform_) {
        cursor=std::min(pieces_.size()-1,size_t(u*pieces_.size()));
        if(u<knots_[cursor])--cursor;
        else if(cursor+1<pieces_.size()&&u>=knots_[cursor+1])++cursor;
      } else cursor=std::min(pieces_.size()-1,size_t(std::upper_bound(knots_.begin(),knots_.end(),u)-knots_.begin()-1));
    }
    const auto& p=pieces_[cursor];const double t=(u-knots_[cursor])*p.inverse_width;
    const Matrix v=((p.power[3]*t+p.power[2])*t+p.power[1])*t+p.power[0];
    const Matrix d=((3*p.power[3]*t+2*p.power[2])*t+p.power[1])*p.inverse_width;
    return {v.col(0),d.col(0),v.rightCols<2>(),d.rightCols<2>()};
  }
};
using tube_query::Stats;using tube_query::Timer;using tube_query::bits;using tube_query::mix;
class CubicQuery {
  const native::Geometry& corridor_;
  PreparedCubic prepared_;
  size_t cursor_=0;
  struct Geometry {bool valid=false;double u=0;FirstSection section;Vec3 normal,normal_first;Eigen::Matrix2d inverse_gram;Eigen::Matrix<double,2,3> dual;};
  struct Answer {bool valid=false;std::array<double,4> key{};TubeValue value;};
  std::vector<Geometry> geometry_{4096};
  std::vector<Answer> answers_{8192};
  const Geometry& geometryAt(double u) {
    Timer timer(stats.enabled,stats.section_ms);++stats.section_calls;
    auto& item=geometry_[mix(bits(u))&(geometry_.size()-1)];
    if(item.valid&&item.u==u){++stats.section_hits;return item;}
    item.section=prepared_.firstUnchecked(u,cursor_);item.valid=true;item.u=u;
    const auto& s=item.section;
    const Vec3 cross=s.shape.col(0).cross(s.shape.col(1));
    const Vec3 dcross=s.shape_first.col(0).cross(s.shape.col(1))+s.shape.col(0).cross(s.shape_first.col(1));
    const double norm=cross.norm();item.normal=cross/norm;
    item.normal_first=(dcross-item.normal*item.normal.dot(dcross))/norm;
    const double a=s.shape.col(0).squaredNorm(),b=s.shape.col(0).dot(s.shape.col(1)),c=s.shape.col(1).squaredNorm();
    item.inverse_gram<<c,-b,-b,a;
    item.inverse_gram/=cross.squaredNorm();
    item.dual.noalias()=item.inverse_gram*s.shape.transpose();
    return item;
  }
TubeValue compute(const Vec3& p, double hint) {
  double u=hint,f=0,fu=0;FirstSection section;double last_u=std::numeric_limits<double>::quiet_NaN();Vec3 normal,normal_u;const Geometry* frame_pointer=nullptr;
  auto plane=[&](double x) {
    last_u=x;const auto& frame=geometryAt(x);frame_pointer=&frame;section=frame.section;++stats.plane_evaluations;
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
    const auto& knots=corridor_.breaks;double previous=plane(knots.front());
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
  const auto& inv=frame_pointer->dual;
  Eigen::Vector2d z=inv*d;
  double q=z.squaredNorm();
  if(q>1) {
    // Radius in the section plane. This also handles endpoint cap extensions.
    Vec3 tangent=d-normal*normal.dot(d);double r=tangent.norm(),sq=std::sqrt(q);
    out.radial=r*(1-1/sq);
    out.grad_radial=(1-1/sq)*tangent/r+r/(q*sq)*(inv.transpose()*z);
    if(root) {
      Eigen::Vector2d zu=frame_pointer->inverse_gram*(Md.transpose()*d-M.transpose()*section.first-
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
  explicit CubicQuery(const native::Geometry& corridor,bool profile=false):corridor_(corridor),prepared_(corridor){stats.enabled=profile;}
  void reset(){const bool enabled=stats.enabled;stats=Stats{};stats.enabled=enabled;for(auto& v:geometry_)v.valid=false;for(auto& v:answers_)v.valid=false;}
  TubeValue operator()(const Vec3& p,double hint) {
    Timer timer(stats.enabled,stats.tube_ms);++stats.queries;
    std::array<double,4> key{p.x(),p.y(),p.z(),hint};
    uint64_t h=0;for(double value:key)h=mix(h^bits(value));
    auto& item=answers_[h&(answers_.size()-1)];
    if(item.valid&&item.key==key){++stats.query_hits;return item.value;}
    auto value=compute(p,hint);item.valid=true;item.key=key;item.value=value;return value;
  }
};

// The benchmark can still run the unchanged H5 evaluator as its control arm.
// Selection is once per query; all native inversion iterations use cubic data.
class BenchmarkQuery {
  std::unique_ptr<tube_query::Query> reference_;
  std::unique_ptr<CubicQuery> cubic_;
public:
  Stats& stats;
  BenchmarkQuery(const Corridor& legacy,const native::Geometry* geometry,bool fast,bool profile,bool horner)
    :reference_(geometry&&geometry->degree==3?nullptr:std::make_unique<tube_query::Query>(legacy,fast,profile,horner)),
     cubic_(geometry&&geometry->degree==3?std::make_unique<CubicQuery>(*geometry,profile):nullptr),
     stats(cubic_?cubic_->stats:reference_->stats){}
  void reset(){if(cubic_)cubic_->reset();else reference_->reset();}
  TubeValue operator()(const Vec3& p,double hint){return cubic_?(*cubic_)(p,hint):(*reference_)(p,hint);}
};
} // namespace gcopter::direct::native
