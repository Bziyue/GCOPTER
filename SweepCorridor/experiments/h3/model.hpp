// SPDX-License-Identifier: MIT
#pragma once
#include <SweepCorridor.hpp>
#include "block_tridiagonal.hpp"

namespace SweepCorridor::experiments::h3 {
namespace c3=internal::cubic;
using internal::GeometryOptions;
using c3::V9;using c3::Geometry;using c3::pack;using c3::shape;
using c3::OptimizerMetrics;
enum class JetSchedule { Frozen, AfterPass, BeforeTightening };
struct OptimizerSettings : c3::OptimizerSettings {
  int iterations=8,corrector_iterations=3,inflation_iterations=2;
  JetSchedule jet_schedule=JetSchedule::AfterPass;
  bool polish=false;
  int volume_jet_steps=0;
  double volume_jet_energy=3;
  int joint_jet_steps=2;
  bool joint_final_only=true;
  int joint_sweeps=2;
  bool joint_fast_line=true;
  bool reuse_joint_supports=true;
  double bending_weight=1;
  int end_nodes=4,end_steps=4;
  double end_energy=1;
  int refresh_rounds=2;
  bool refresh_shape=true;
  double joint_budget_fraction=1,refresh_fraction=.05;
  double joint_budget_share=.99;
  OptimizerSettings(const c3::OptimizerSettings& baseline={}):c3::OptimizerSettings(baseline){}
};
using c3::Clock;using c3::StageTimer;using c3::Deadline;
using c3::matrix;
using Map = Eigen::Matrix<double,9,6>;

struct Model {
  struct Basis {
    std::vector<double> breaks;
    int count=0;
    double weight(int i)const{return .5*((i?breaks[i]-breaks[i-1]:0)+(i+1<count?breaks[i+1]-breaks[i]:0));}
  } basis;
  struct Sample {double left,derivative,weight;V9 offset;Vec3 tangent;};
  std::vector<Vec3> reference;
  std::vector<Mat32> frames;
  std::vector<V9> jets;
  std::vector<std::array<Sample,5>> quadrature;
  Eigen::VectorXd x;
  mutable std::array<double,2> anchor_parameters{{0.,1.}};

  // This is an exact change of basis of a cheap, unoptimized seed reference.
  // No C3 corridor inflation or projection solve is hidden in initialization.
  explicit Model(const Geometry& reference_geometry){
    basis.breaks=reference_geometry.breaks;basis.count=basis.breaks.size();
    reference.resize(basis.count);frames.resize(basis.count);jets.resize(basis.count);x.setZero(6*basis.count);
    for(int i=0;i<basis.count;++i){const auto s=reference_geometry.section(basis.breaks[i]);reference[i]=s.center;
      const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> e(s.shape.transpose()*s.shape);
      const Eigen::Matrix2d root=e.eigenvectors()*e.eigenvalues().cwiseSqrt().asDiagonal()*e.eigenvectors().transpose();
      frames[i]=s.shape*root.inverse();x.segment<3>(6*i+3)<<root(0,0),root(0,1),root(1,1);
      jets[i]=pack(s.first,s.shape_first);
    }
    cacheQuadrature();
  }
  Map jacobian(int i)const{
    Map j=Map::Zero();j.topLeftCorner<3,3>().setIdentity();
    j.block<3,1>(3,3)=frames[i].col(0);j.block<3,1>(3,4)=frames[i].col(1);
    j.block<3,1>(6,4)=frames[i].col(0);j.block<3,1>(6,5)=frames[i].col(1);return j;
  }
  V9 node(const Eigen::VectorXd& values,int i)const{return pack(reference[i]+values.segment<3>(6*i),frames[i]*matrix(values,i));}
  V9 nodeDirection(const Eigen::VectorXd& values,int i)const{return pack(values.segment<3>(6*i),frames[i]*matrix(values,i));}
  c3::Curve<3,V9> offsets(int i)const{
    const double h=(basis.breaks[i+1]-basis.breaks[i])/3;
    return {V9::Zero(),V9(h*jets[i]),V9(-h*jets[i+1]),V9::Zero()};
  }
  Geometry geometry(const Eigen::VectorXd& values)const{
    Geometry g;g.breaks=basis.breaks;g.controls.resize(4*(basis.count-1));
    V9 left=node(values,0);
    for(int i=0;i<basis.count-1;++i){const V9 right=node(values,i+1);const double h=(basis.breaks[i+1]-basis.breaks[i])/3;
      g.controls[4*i]=left;g.controls[4*i+1]=left+h*jets[i];g.controls[4*i+2]=right-h*jets[i+1];g.controls[4*i+3]=right;left=right;}
    return g;
  }
  Geometry geometry()const{return geometry(x);}
  Section section(const Eigen::VectorXd& values,double u,int& span,Eigen::Vector2d& weights)const{
    span=std::min(basis.count-2,int(std::upper_bound(basis.breaks.begin(),basis.breaks.end(),u)-basis.breaks.begin())-1);
    const double h=basis.breaks[span+1]-basis.breaks[span],t=(u-basis.breaks[span])/h,r=1-t;
    weights<<r*r*(1+2*t),t*t*(3-2*t);
    const V9 left=node(values,span),right=node(values,span+1),delta=right-left;
    const V9 y=weights[0]*left+weights[1]*right+h*(t*r*r*jets[span]-t*t*r*jets[span+1]);
    const V9 dy=(6*t*r/h)*delta+(1-4*t+3*t*t)*jets[span]+(3*t*t-2*t)*jets[span+1];
    const V9 ddy=(6*(1-2*t)/(h*h))*delta+((-4+6*t)/h)*jets[span]+((6*t-2)/h)*jets[span+1];
    return {y.head<3>(),dy.head<3>(),ddy.head<3>(),shape(y),shape(dy),shape(ddy)};
  }
  void cacheQuadrature(){
    quadrature.resize(basis.count-1);
    cacheQuadrature(0,basis.count-1);
  }
  void cacheQuadrature(int begin,int end){
    constexpr double t[5]={.046910077030668,.230765344947158,.5,.769234655052842,.953089922969332};
    constexpr double w[5]={.118463442528095,.239314335249683,.284444444444444,.239314335249683,.118463442528095};
    for(int i=begin;i<end;++i){const double h=basis.breaks[i+1]-basis.breaks[i];
      for(int q=0;q<5;++q){const double u=t[q],r=1-u;auto& s=quadrature[i][q];
        s.left=r*r*(1+2*u);s.derivative=-6*u*r/h;s.weight=std::acos(-1.)*h*w[q];
        s.offset=h*(u*r*r*jets[i]-u*u*r*jets[i+1]);
        s.tangent=(1-4*u+3*u*u)*jets[i].head<3>()+(3*u*u-2*u)*jets[i+1].head<3>();}
    }
  }
};

struct Energy {
  BlockTridiagonal quadratic;
  Eigen::VectorXd linear;
  Energy(const Model& m,double length,const OptimizerSettings& settings):quadratic(m.x.size()),linear(Eigen::VectorXd::Zero(m.x.size())){
    for(int i=0;i<m.basis.count-1;++i){const double h=m.basis.breaks[i+1]-m.basis.breaks[i];
      for(double t:{.5-.5/std::sqrt(3.),.5+.5/std::sqrt(3.)}){
        const double d=6*(1-2*t)/(h*h*length*length);
        Eigen::Matrix<double,9,12> a;a.leftCols<6>()=-d*m.jacobian(i);a.rightCols<6>()=d*m.jacobian(i+1);
        V9 b=((-4+6*t)*m.jets[i]+(6*t-2)*m.jets[i+1])/(h*length*length);
        b.head<3>()+=d*(m.reference[i+1]-m.reference[i]);
        const double wc=std::sqrt(settings.bending_weight*settings.center_energy*.5*h*length),wm=std::sqrt(settings.bending_weight*settings.shape_energy*.5*h*length);
        a.topRows<3>()*=wc;a.bottomRows<6>()*=wm;b.head<3>()*=wc;b.tail<6>()*=wm;
        const Eigen::Matrix<double,12,12> gram=2*a.transpose()*a;
        linear.segment<12>(6*i).noalias()+=2*a.transpose()*b;
        quadratic.diagonal[i]+=gram.topLeftCorner<6,6>();quadratic.diagonal[i+1]+=gram.bottomRightCorner<6,6>();
        quadratic.lower[i]+=gram.bottomLeftCorner<6,6>();
      }
    }
    for(int i=0;i<m.basis.count;++i)quadratic.diagonal[i].diagonal().head<3>().array()+=2*settings.offset_weight*m.basis.weight(i);
  }
};

inline double sweptVolume(const Model& m,const Eigen::VectorXd& x,Eigen::VectorXd* gradient=nullptr){
  if(gradient)gradient->setZero(x.size());double volume=0;V9 left=m.node(x,0);
  for(int i=0;i<m.basis.count-1;++i){const V9 right=m.node(x,i+1);
    V9 gl=V9::Zero(),gr=V9::Zero();
    for(const auto& s:m.quadrature[i]){const V9 y=s.left*left+(1-s.left)*right+s.offset;
      const Vec3 tangent=s.derivative*(left.head<3>()-right.head<3>())+s.tangent;
      const Mat32 a=shape(y);const Vec3 n=a.col(0).cross(a.col(1));volume+=s.weight*n.dot(tangent);
      if(gradient){Mat32 mg;mg.col(0)=a.col(1).cross(tangent);mg.col(1)=tangent.cross(a.col(0));const V9 g=pack(Vec3::Zero(),mg);
        gl+=s.weight*s.left*g;gr+=s.weight*(1-s.left)*g;
        gl.head<3>()+=s.weight*s.derivative*n;gr.head<3>()-=s.weight*s.derivative*n;}
    }
    if(gradient){gradient->segment<6>(6*i).noalias()+=m.jacobian(i).transpose()*gl;
      gradient->segment<6>(6*i+6).noalias()+=m.jacobian(i+1).transpose()*gr;}
    left=right;
  }return volume;
}
inline Vec3 volumeLine(const Model& m,const Eigen::VectorXd& x,const Eigen::VectorXd& direction){
  Vec3 out=Vec3::Zero();V9 left=m.node(x,0),dl=m.nodeDirection(direction,0);
  for(int i=0;i<m.basis.count-1;++i){const V9 right=m.node(x,i+1),dr=m.nodeDirection(direction,i+1);
    for(const auto& s:m.quadrature[i]){
      const Mat32 a=shape(s.left*left+(1-s.left)*right+s.offset),d=shape(s.left*dl+(1-s.left)*dr);
      const Vec3 tangent=s.derivative*(left.head<3>()-right.head<3>())+s.tangent;
      const Vec3 dt=s.derivative*(dl.head<3>()-dr.head<3>());
      const Vec3 n=a.col(0).cross(a.col(1)),dn=d.col(0).cross(a.col(1))+a.col(0).cross(d.col(1)),dd=d.col(0).cross(d.col(1));
      out+=s.weight*Vec3(n.dot(dt)+dn.dot(tangent),dn.dot(dt)+dd.dot(tangent),dd.dot(dt));
    }left=right;dl=dr;
  }return out;
}
}
