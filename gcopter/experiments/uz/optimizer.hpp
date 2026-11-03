// SPDX-License-Identifier: MIT
#pragma once
#include "gcopter/corridor_evaluator.hpp"
#include "TrajectoryOptAdapters/GcopterPenaltyIntegralCostAdapter.hpp"
#include "gcopter/lbfgs.hpp"
#include <Eigen/Cholesky>
#include <array>
#include <cmath>
#include <limits>

namespace uz_experiment {
using namespace gcopter::direct;
using V2=Eigen::Vector2d;
using Basis=Eigen::Matrix<double,4,8>;

inline double choose(int n,int k){double a=1;for(int j=1;j<=k;++j)a*=double(n+1-j)/j;return a;}
/** Values and first three physical-u derivatives of Bernstein bases. */
inline Basis basis(int degree,double u){
  Basis result=Basis::Zero();double factor=1;
  for(int r=0;r<=3;++r){
    if(r)factor*=degree-r+1;
    std::array<double,8> low{};low[0]=1;
    for(int n=1;n<=degree-r;++n){for(int j=n;j>0;--j)low[j]=(1-u)*low[j]+u*low[j-1];low[0]*=1-u;}
    for(int i=0;i<=degree;++i)for(int j=0;j<=r;++j)
      if(i-j>=0&&i-j<=degree-r)result(r,i)+=factor*((r-j)%2?-1:1)*choose(r,j)*low[i-j];
  }
  return result;
}
inline std::array<double,5> horner(const std::array<double,8>& a,double s){
  std::array<double,5> d{};d[0]=a[7];
  for(int i=6;i>=0;--i){for(int r=4;r>0;--r)d[r]=d[r]*s+r*d[r-1];d[0]=d[0]*s+a[i];}
  return d;
}
/** Monotone degree-seven progress, zero endpoint velocity/acceleration.
 * Three positive control gaps >=0.05; two logits change the progress shape.
 */
struct Progress {
  std::array<double,8> controls{},power{},reverse_power{};
  Eigen::Matrix2d control_jacobian;
  Progress(double a,double b){
    const double shift=std::max({0.0,a,b});
    Eigen::Vector3d q(std::exp(a-shift),std::exp(b-shift),std::exp(-shift));q/=q.sum();
    constexpr double gap=0.05,scale=1-3*gap;
    controls={0,0,0,gap+scale*q(0),2*gap+scale*(q(0)+q(1)),1,1,1};
    for(int j=0;j<2;++j){
      control_jacobian(0,j)=scale*q(0)*((j==0?1.0:0.0)-q(j));
      control_jacobian(1,j)=scale*q(2)*q(j);
    }
    auto forward=controls;std::array<double,8> reverse;
    for(int i=0;i<8;++i)reverse[i]=1-controls[7-i];
    for(int k=0;k<8;++k){
      power[k]=choose(7,k)*forward[0];reverse_power[k]=choose(7,k)*reverse[0];
      for(int j=0;j<7-k;++j){forward[j]=forward[j+1]-forward[j];reverse[j]=reverse[j+1]-reverse[j];}
    }
  }
  std::array<double,5> at(double s) const {
    if(s<=0.5)return horner(power,s);
    auto v=horner(reverse_power,1-s);v[0]=1-v[0];v[2]=-v[2];v[4]=-v[4];return v;
  }
  /** Unique scalar time-map inversion at fixed geometric quadrature nodes.
   * This is not a nearest-section inversion. Reflection avoids cancellation at u=1.
   */
  std::pair<double,std::array<double,5>> inverse(double u,uint64_t& iterations) const {
    const bool reverse=u>0.5;const double target=reverse?1-u:u;
    const auto& a=reverse?reverse_power:power;
    double lo=0,hi=1,s=std::clamp(std::cbrt(target/a[3]),1e-8,0.99);
    std::array<double,5> v{};
    for(int k=0;k<60;++k){
      ++iterations;v=horner(a,s);const double f=v[0]-target;
      if(std::abs(f)<=2e-14*std::max(target,1e-6))break;
      if(f>0)hi=s;else lo=s;
      const double next=s-f/v[1];s=(next>lo&&next<hi)?next:0.5*(lo+hi);
    }
    // Always synchronize derivatives with the returned root.
    v=horner(a,s);
    if(reverse){s=1-s;v[0]=1-v[0];v[2]=-v[2];v[4]=-v[4];}
    return {s,v};
  }
};

struct Geometry {
  std::array<Vec3,4> center;
  std::array<std::array<Mat32,6>,4> map;
};
struct Sample {double u,weight;Geometry geometry;};
struct State {Vec3 p,v,a,j;V2 z;double u;};

/** Separate reduced experiment: 11 variables, not the world-space minimum-jerk parameterization.
 * x = log(T), two progress logits, four free 2D transverse Bezier controls.
 * z(t)=z(u(t)); fixed endpoint controls reproduce the same physical endpoints.
 */
class Optimizer {
  const Corridor& corridor_;
  PreparedCorridor prepared_;
  V2 start_,end_;
  flatness::FlatnessMap flat_;
  traj_opt_components::PolyhedraH empty_{1};
  Eigen::VectorXi ids_=Eigen::VectorXi::Zero(1);
  traj_opt_adapters::GcopterPenaltyIntegralCostAdapter dynamics_;
  double time_weight_=20;
  std::vector<Sample> samples_;
  std::array<V2,6> controls_;
  std::array<double,6> disk_scales_{};

  Geometry geometry(double u,size_t& cursor) const {
    cursor=prepared_.locate(u,cursor);const auto& piece=prepared_.piece(cursor);
    const double s=(u-prepared_.knots()[cursor])*piece.inverse_width;
    std::array<Eigen::Matrix3d,4> d;
    d[0]=piece.power[5];for(int r=1;r<4;++r)d[r].setZero();
    for(int k=4;k>=0;--k){for(int r=3;r>0;--r)d[r]=d[r]*s+r*d[r-1];d[0]=d[0]*s+piece.power[k];}
    double factor=1;for(int r=0;r<4;++r){d[r]*=factor;factor*=piece.inverse_width;}
    const Basis b=basis(5,u);Geometry result;
    for(int r=0;r<4;++r){
      result.center[r]=d[r].col(0);
      for(int i=0;i<6;++i){result.map[r][i].setZero();for(int k=0;k<=r;++k)
        result.map[r][i]+=choose(r,k)*d[k].rightCols<2>()*b(r-k,i);}
    }
    return result;
  }
  void decode(const Eigen::VectorXd& x){
    controls_[0]=start_;controls_[5]=end_;
    for(int i=1;i<5;++i){const V2 q=x.segment<2>(3+2*(i-1));const double scale=std::hypot(1.0,q.norm());
      controls_[i]=q/scale;disk_scales_[i]=scale;}
  }
  std::array<Vec3,4> spatial(const Geometry& geometry) const {
    auto f=geometry.center;
    for(int r=0;r<4;++r)for(int i=0;i<6;++i)f[r]+=geometry.map[r][i]*controls_[i];
    return f;
  }
 public:
  uint64_t evaluations=0,root_iterations=0;
  int iterations=0;
  Optimizer(const Corridor& corridor,const Vec3& start,const Vec3& end,int quadrature=12,bool dynamics=true)
      :corridor_(corridor),prepared_(corridor){
    for(int i=0;i<2;++i){auto s=corridor.section(i);const Vec3 p=i?end:start;
      V2 z=(s.shape.transpose()*s.shape).ldlt().solve(s.shape.transpose()*(p-s.center));
      if((s.point(z)-p).norm()>1e-7||z.norm()>1)throw std::invalid_argument("endpoint outside corridor section");
      (i?end_:start_)=z;}
    flat_.reset(.61,9.8,.70,.80,.01,.0001);empty_[0].resize(0,4);
    Eigen::VectorXd limits(5),weights(5);limits<<4,2.1,1.05,2,12;weights<<0,1e4,1e4,1e4,1e5;
    if(!dynamics)weights.setZero();
    dynamics_.reset(&empty_,&ids_,.01,limits,weights,&flat_);
    if(quadrature<2||quadrature>64)throw std::invalid_argument("quadrature must be 2..64");
    std::vector<std::pair<double,double>> gauss;
    for(int i=0;i<quadrature;++i){
      double z=std::cos(3.141592653589793*(i+.75)/(quadrature+.5)),derivative=0;
      for(int k=0;k<30;++k){double a=1,b=z;
        for(int n=2;n<=quadrature;++n){double c=((2*n-1)*z*b-(n-1)*a)/n;a=b;b=c;}
        derivative=quadrature*(z*b-a)/(z*z-1);double step=b/derivative;z-=step;if(std::abs(step)<1e-15)break;}
      gauss.emplace_back((1+z)/2,1/((1-z*z)*derivative*derivative));
    }
    std::sort(gauss.begin(),gauss.end());size_t cursor=0;
    const auto& knots=corridor.knots();
    for(size_t k=0;k+1<knots.size();++k)for(auto [s,w]:gauss){
      double r=s,jacobian=1;
      // Cancel the integrable dt/du singularities from zero endpoint velocity/acceleration.
      if(knots.size()==2){const double a=s*s*s,b=std::pow(1-s,3),sum=a+b;r=a/sum;jacobian=3*s*s*(1-s)*(1-s)/(sum*sum);}
      else if(k==0){r=s*s*s;jacobian=3*s*s;}
      else if(k+2==knots.size()){r=1-std::pow(1-s,3);jacobian=3*(1-s)*(1-s);}
      const double width=knots[k+1]-knots[k],u=knots[k]+width*r;
      samples_.push_back({u,w*width*jacobian,geometry(u,cursor)});
    }
  }
  Eigen::VectorXd initial(double duration) const {
    Eigen::VectorXd x=Eigen::VectorXd::Zero(11);x(0)=std::log(duration);
    for(int i=1;i<5;++i){V2 z=(1-i/5.0)*start_+i/5.0*end_;if(z.norm()>=0.99)z*=0.99/z.norm();
      x.segment<2>(3+2*(i-1))=z/std::sqrt(1-z.squaredNorm());}
    return x;
  }
  double evaluate(const Eigen::VectorXd& x,Eigen::VectorXd& gradient){
    ++evaluations;gradient.setZero(11);const double T=std::exp(x(0));
    if(!x.allFinite()||!std::isfinite(T)||T<=0)return std::numeric_limits<double>::infinity();
    decode(x);Progress progress(x(1),x(2));const double inv=1/T,inv2=inv*inv,inv3=inv2*inv;
    double cost=time_weight_*T;gradient(0)=cost;
    std::array<V2,6> g_controls;for(auto& g:g_controls)g.setZero();V2 g_progress=V2::Zero();
    for(const auto& sample:samples_){
      const auto [s,U]=progress.inverse(sample.u,root_iterations);
      const double U1=U[1],U2=U[2],U3=U[3];const auto f=spatial(sample.geometry);
      SplineTrajectory::SampleState<3> state;state.p=f[0];state.v=f[1]*(U1*inv);
      state.a=(f[2]*U1*U1+f[1]*U2)*inv2;
      state.j=(f[3]*U1*U1*U1+3*f[2]*U1*U2+f[1]*U3)*inv3;
      SplineTrajectory::SampleGradient<3> partial;
      SplineTrajectory::IntegralPointInfo info{0,1,0,1,s,T,T,s*T,s*T};
      const double density=state.j.squaredNorm()+dynamics_(info,state,partial);
      partial.j+=2*state.j;
      const double measure=sample.weight*T/U1;cost+=measure*density;
      gradient(0)+=measure*(density-partial.v.dot(state.v)-2*partial.a.dot(state.a)-3*partial.j.dot(state.j));
      std::array<Vec3,4> gf{partial.p,
        partial.v*(U1*inv)+partial.a*(U2*inv2)+partial.j*(U3*inv3),
        partial.a*(U1*U1*inv2)+partial.j*(3*U1*U2*inv3),partial.j*(U1*U1*U1*inv3)};
      for(int i=1;i<5;++i)for(int r=0;r<4;++r)g_controls[i].noalias()+=measure*sample.geometry.map[r][i].transpose()*gf[r];
      std::array<double,4> gu{};
      gu[1]=partial.v.dot(f[1])*inv+partial.a.dot(2*f[2]*U1)*inv2+
          partial.j.dot(3*f[3]*U1*U1+3*f[2]*U2)*inv3-density/U1;
      gu[2]=partial.a.dot(f[1])*inv2+partial.j.dot(3*f[2]*U1)*inv3;
      gu[3]=partial.j.dot(f[1])*inv3;
      const Basis b=basis(7,s);
      for(int k=0;k<2;++k)for(int r=1;r<=3;++r)
        g_progress(k)+=measure*gu[r]*(b(r,k+3)-U[r+1]*b(0,k+3)/U1);
    }
    gradient.segment<2>(1)=progress.control_jacobian.transpose()*g_progress;
    for(int i=1;i<5;++i)gradient.segment<2>(3+2*(i-1))=
      (g_controls[i]-controls_[i]*controls_[i].dot(g_controls[i]))/disk_scales_[i];
    return cost;
  }
  State at(const Eigen::VectorXd& x,double s){
    decode(x);Progress progress(x(1),x(2));const double T=std::exp(x(0));auto U=progress.at(s);
    U[0]=std::clamp(U[0],0.0,1.0);size_t cursor=0;auto f=spatial(geometry(U[0],cursor));
    V2 z=V2::Zero();const Basis b=basis(5,U[0]);for(int i=0;i<6;++i)z+=b(0,i)*controls_[i];
    return {f[0],f[1]*U[1]/T,(f[2]*U[1]*U[1]+f[1]*U[2])/(T*T),
      (f[3]*U[1]*U[1]*U[1]+3*f[2]*U[1]*U[2]+f[1]*U[3])/(T*T*T),z,U[0]};
  }
  double maxControlNorm(const Eigen::VectorXd& x){decode(x);double value=0;for(auto z:controls_)value=std::max(value,z.norm());return value;}
  size_t sampleCount() const{return samples_.size();}
  static double callback(void* ptr,const Eigen::VectorXd& x,Eigen::VectorXd& g){return static_cast<Optimizer*>(ptr)->evaluate(x,g);}
  static int progressCallback(void* ptr,const Eigen::VectorXd&,const Eigen::VectorXd&,double,double,int k,int){static_cast<Optimizer*>(ptr)->iterations=k;return 0;}
};
} // namespace uz_experiment
