// SPDX-License-Identifier: MIT
#include "minkowski_geometry.hpp"
#include <stdexcept>

namespace gcopter::direct::minkowski {
namespace {
double bernstein(int n,int k,double t) {
  return detail::choose(n,k)*std::pow(t,k)*std::pow(1-t,n-k);
}
struct Support {
  Weights h=Weights::Zero(), d=h, rho=h, rho_first=h;
};
Support support(const std::array<M2,4>& q,int count,double angle) {
  const V2 n(std::cos(angle),std::sin(angle)), t(-n[1],n[0]);
  Support out;
  for(int k=0;k<count;++k) {
    const V2 a=q[k]*n;
    out.h[k]=std::sqrt(n.dot(a));
    out.d[k]=t.dot(a)/out.h[k];
    out.rho[k]=q[k].determinant()/std::pow(out.h[k],3);
    out.rho_first[k]=-3*out.rho[k]*out.d[k]/out.h[k];
  }
  return out;
}
}
Section Corridor::section(double u) const {
  if(!std::isfinite(u) || u<0 || u>1 || !pieces())
    throw std::invalid_argument("invalid Minkowski parameter");
  const auto at=std::upper_bound(knots_.begin(),knots_.end(),u);
  const std::size_t i=std::min(pieces()-1,std::size_t(at-knots_.begin()-1));
  const double h=knots_[i+1]-knots_[i], t=(u-knots_[i])/h;
  Section s;
  s.center.setZero(); s.first.setZero(); s.second.setZero();
  s.frame.setZero(); s.frame_first.setZero(); s.frame_second.setZero();
  s.weights.setZero(); s.weights_first.setZero(); s.weights_second.setZero();
  for(int j=0;j<6;++j) {
    const auto k=6*i+j;
    const double b=bernstein(5,j,t);
    s.center+=b*centers_[k]; s.frame+=b*frames_[k]; s.weights+=b*weights_[k];
    if(j<5) {
      const double d=5/h*bernstein(4,j,t);
      s.first+=d*(centers_[k+1]-centers_[k]);
      s.frame_first+=d*(frames_[k+1]-frames_[k]);
      s.weights_first+=d*(weights_[k+1]-weights_[k]);
    }
    if(j<4) {
      const double d=20/(h*h)*bernstein(3,j,t);
      s.second+=d*(centers_[k+2]-2*centers_[k+1]+centers_[k]);
      s.frame_second+=d*(frames_[k+2]-2*frames_[k+1]+frames_[k]);
      s.weights_second+=d*(weights_[k+2]-2*weights_[k+1]+weights_[k]);
    }
  }
  return s;
}
Boundary Corridor::boundary(double u,double angle) const {
  if(!std::isfinite(angle)) throw std::invalid_argument("nonfinite boundary angle");
  const auto s=section(u);
  const auto h=support(q_,templates_,angle);
  const V2 n(std::cos(angle),std::sin(angle)), t(-n[1],n[0]);
  const V2 b=s.weights.dot(h.h)*n+s.weights.dot(h.d)*t;
  const V2 bu=s.weights_first.dot(h.h)*n+s.weights_first.dot(h.d)*t;
  const V2 buu=s.weights_second.dot(h.h)*n+s.weights_second.dot(h.d)*t;
  const V2 bt=s.weights.dot(h.rho)*t;
  Boundary out;
  out.point=s.center+s.frame*b;
  out.first=s.first+s.frame_first*b+s.frame*bu;
  out.second=s.second+s.frame_second*b+2*s.frame_first*bu+s.frame*buu;
  out.angle_first=s.frame*bt;
  out.mixed=s.frame_first*bt+s.frame*(s.weights_first.dot(h.rho)*t);
  out.angle_second=s.frame*(s.weights.dot(h.rho_first)*t-s.weights.dot(h.rho)*n);
  return out;
}
double Corridor::normalAngle(const V2& z,const Weights& r) const {
  const double center=std::atan2(z[1],z[0]), half_pi=.5*std::acos(-1.);
  double lo=center-half_pi, hi=center+half_pi, angle=center;
  // Alternate with bisection: even an unhelpful Newton proposal halves the
  // bracket every two steps, including extremely anisotropic sections.
  for(int it=0;it<96;++it) {
    const V2 n(std::cos(angle),std::sin(angle)), t(-n[1],n[0]);
    const auto h=support(q_,templates_,angle);
    const double a=n.dot(z), b=t.dot(z), H=r.dot(h.h), D=r.dot(h.d);
    const double residual=b*H-a*D;
    if(std::abs(residual)<=2e-14*z.norm()*(H+std::abs(D)) || hi-lo<2e-14) return angle;
    if(residual>0) lo=angle; else hi=angle;
    const double next=angle+residual/(a*r.dot(h.rho));
    angle=it%2==0 && std::isfinite(next) && next>lo && next<hi ? next:.5*(lo+hi);
  }
  throw std::runtime_error("Minkowski gauge root did not converge");
}
double Corridor::gauge(const V2& z,const Weights& r) const {
  if(z.squaredNorm()==0) return 0;
  const double angle=normalAngle(z,r);
  const V2 n(std::cos(angle),std::sin(angle));
  return n.dot(z)/r.dot(support(q_,templates_,angle).h);
}
double Corridor::area(double u) const {
  const auto s=section(u);
  return s.weights.dot(mixed_*s.weights)*s.frame.col(0).cross(s.frame.col(1)).norm();
}
Differential Corridor::evaluate(double u,const Vec3& x) const {
  if(!x.allFinite()) throw std::invalid_argument("nonfinite evaluation point");
  const auto s=section(u);
  const auto& f=s.frame; const auto& d=s.frame_first; const auto& dd=s.frame_second;
  const M2 g=f.transpose()*f, gd=d.transpose()*f+f.transpose()*d,
           gdd=dd.transpose()*f+2*d.transpose()*d+f.transpose()*dd;
  const M2 inverse=g.inverse(), id=-inverse*gd*inverse,
           idd=2*inverse*gd*inverse*gd*inverse-inverse*gdd*inverse;
  const Eigen::Matrix<double,2,3> p=inverse*f.transpose(),
      pd=id*f.transpose()+inverse*d.transpose(),
      pdd=idd*f.transpose()+2*id*d.transpose()+inverse*dd.transpose();
  const Vec3 delta=x-s.center;
  const V2 z=p*delta, zu=pd*delta-p*s.first,
           zuu=pdd*delta-2*pd*s.first-p*s.second;
  Differential out;
  out.plane_residual=delta.dot(f.col(0).cross(f.col(1)).normalized());
  out.value=-1;
  if(z.squaredNorm()==0) return out;
  const double angle=normalAngle(z,s.weights);
  const V2 n(std::cos(angle),std::sin(angle)), t(-n[1],n[0]);
  const auto h=support(q_,templates_,angle);
  const double a=n.dot(z), at=t.dot(z), H=s.weights.dot(h.h),
               Ht=s.weights.dot(h.d), Htt=s.weights.dot(h.rho-h.h);
  Eigen::Vector4d av, avtheta, hv=Eigen::Vector4d::Zero(), hvtheta=hv;
  av << n.dot(zu), p.transpose()*n;
  avtheta << t.dot(zu), p.transpose()*t;
  hv[0]=s.weights_first.dot(h.h); hvtheta[0]=s.weights_first.dot(h.d);
  Eigen::Matrix4d ah=Eigen::Matrix4d::Zero(), hh=ah;
  ah(0,0)=n.dot(zuu); hh(0,0)=s.weights_second.dot(h.h);
  ah.block<3,1>(1,0)=pd.transpose()*n;
  ah.block<1,3>(0,1)=ah.block<3,1>(1,0).transpose();
  const double eta=a/H;
  const Eigen::Vector4d grad=av/H-a*hv/(H*H),
      mixed=avtheta/H-(at*hv+Ht*av+a*hvtheta)/(H*H)+2*a*Ht*hv/(H*H*H);
  const double curvature=-a/H-(2*at*Ht+a*Htt)/(H*H)+2*a*Ht*Ht/(H*H*H);
  const Eigen::Matrix4d hessian=ah/H-(av*hv.transpose()+hv*av.transpose()+a*hh)/(H*H)+
      2*a*(hv*hv.transpose())/(H*H*H)-(mixed*mixed.transpose())/curvature;
  out.value=std::pow(eta,4)-1;
  out.gradient=4*std::pow(eta,3)*grad;
  out.hessian=12*eta*eta*(grad*grad.transpose())+4*std::pow(eta,3)*hessian;
  return out;
}
}  // namespace gcopter::direct::minkowski
