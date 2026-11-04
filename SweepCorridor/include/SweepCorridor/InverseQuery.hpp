// SPDX-License-Identifier: MIT
#pragma once
#include <SweepCorridor.hpp>
#include <cstdint>
#include <cstring>
namespace SweepCorridor {
// Local section-plane inversion. The hint selects a root, not a global union
// projection. Endpoint results retain their cap residual. No root warm starts.
struct InverseResult {
  double radial=0,plane=0,u=0;
  Vec3 grad_radial=Vec3::Zero(),grad_plane=Vec3::Zero();
  bool valid=true;
};
namespace detail {
inline uint64_t queryBits(double x){uint64_t r;std::memcpy(&r,&x,sizeof(x));return r;}
inline uint64_t queryMix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
}
// Owns coefficients and bounded exact caches. One instance per thread.
// Preconditions are the same as Query; hint is finite and in [0,1].
template<int Degree> class InverseQuery {
  Query<Degree> prepared_;
  struct Geometry {bool valid=false;double u=0;FirstSection section;Vec3 normal,normal_first;Eigen::Matrix2d inverse_gram;Eigen::Matrix<double,2,3> dual;};
  struct Answer {bool valid=false;std::array<double,4> key{};InverseResult value;};
  std::vector<Geometry> geometry_{4096};
  std::vector<Answer> answers_{8192};
  const Geometry& geometryAt(double u) {

    auto& item=geometry_[detail::queryMix(detail::queryBits(u))&(geometry_.size()-1)];
    if(item.valid&&item.u==u){return item;}
    item.section=prepared_.first(u);item.valid=true;item.u=u;
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
  InverseResult compute(const Vec3& p, double hint) {
    double u=hint,f=0,fu=0;FirstSection section;double last_u=std::numeric_limits<double>::quiet_NaN();Vec3 normal,normal_u;const Geometry* frame_pointer=nullptr;
    auto plane=[&](double x) {
      last_u=x;const auto& frame=geometryAt(x);frame_pointer=&frame;section=frame.section;
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

      // Rare fallback: choose the nearest sign-changing knot interval, then bisect.
      double best=1e10,lo=0,hi=0,fl=0;
      const auto& knots=prepared_.knots();double previous=plane(knots.front());
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
    InverseResult out;out.u=u;out.plane=f;out.grad_plane=normal;
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
  explicit InverseQuery(const Corridor<Degree>& corridor):prepared_(corridor){}
  void reset(){for(auto& v:geometry_)v.valid=false;for(auto& v:answers_)v.valid=false;}
  InverseResult operator()(const Vec3& p,double hint) {

    std::array<double,4> key{p.x(),p.y(),p.z(),hint};
    uint64_t h=0;for(double value:key)h=detail::queryMix(h^detail::queryBits(value));
    auto& item=answers_[h&(answers_.size()-1)];
    if(item.valid&&item.key==key){return item.value;}
    auto value=compute(p,hint);item.valid=true;item.key=key;item.value=value;return value;
  }
};

} // namespace SweepCorridor
