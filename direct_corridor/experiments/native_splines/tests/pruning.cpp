// SPDX-License-Identifier: MIT
#include "cubic_optimizer.hpp"
#include "cubic_certificates.hpp"
#include <Eigen/Eigenvalues>
#include <cassert>
#include <iostream>
#include <random>
using namespace gcopter::direct;
using namespace gcopter::direct::native;

static void checkProofs(const CubicProblem::Group& group,const Vec3& center,
                        const std::vector<CubicProblem::Group::Proof>& proofs,double offset,double radius) {
  auto find=[&](int id)->const CubicProblem::Support& {
    const auto p=std::find_if(group.supports.begin(),group.supports.end(),
                            [&](const auto& s){return s.original_id==id;});
    assert(p!=group.supports.end());return *p;
  };
  auto dual=[&](int id) {
    if(id<0)return Vec3::Zero().eval();
    const auto& p=find(id);return Vec3(p.normal/(p.bound-p.normal.dot(center)));
  };
  for(const auto& proof:proofs) {
    assert(find(proof.removed).redundant);
    if(proof.left>=0)assert(!find(proof.left).redundant);
    if(proof.right>=0)assert(!find(proof.right).redundant);
    assert(proof.left_weight>=0&&proof.right_weight>=0&&proof.left_weight+proof.right_weight<=1);
    const Vec3 residual=dual(proof.removed)-proof.left_weight*dual(proof.left)-proof.right_weight*dual(proof.right);
    long double upper=proof.left_weight+proof.right_weight;
    upper+=residual.cast<long double>().dot((group.reference.head<3>()-center).cast<long double>());
    for(int k=0;k<group.count;++k) {
      const Eigen::Matrix<long double,2,1> v=group.maps[k].cast<long double>().transpose()*residual.cast<long double>();
      upper+=(offset+radius)*v.norm();
    }
    assert(upper<1-1e-11);
    assert(std::abs(upper-proof.upper)<1e-11);
  }
}

int main() {
  for(int i=0;i<=1000;++i)for(double q:{std::expm1(-30.+60.*i/1000),1e-14*std::cos(double(i))}) {
    CubicProblem::LineModel line;line.terms.push_back({0,q,1});
    const long double exact=-std::log1p(static_cast<long double>(q));
    assert(static_cast<long double>(line.upperBound(1))>=exact);
  }
  for(double tilt:{0.,1e-9,.05,.5})for(bool one_sided:{false,true}) {
    CubicProblem::Group group;group.first=0;group.count=2;group.reference.setZero();
    group.maps[0].col(0)=.5*Vec3::UnitY();group.maps[0].col(1)=.5*Vec3::UnitZ();
    group.maps[1].col(0)=.5*Vec3(std::sin(tilt),std::cos(tilt),0);
    group.maps[1].col(1)=.5*Vec3::UnitZ();
    Eigen::VectorXd initial=Eigen::VectorXd::Zero(10);
    for(int i=0;i<2;++i){initial[5*i+2]=.15;initial[5*i+4]=.15;}
    initial.segment<2>(0)<<.04,-.03;initial.segment<2>(5)<<-.02,.01;
    auto add=[&](Vec3 n,double b){group.supports.push_back({n,b,false,-1,int(group.supports.size())});};
    for(int j=0;j<4;++j) {
      const double theta=(one_sided?j*.35:j*1.5707963267948966);
      add(Vec3(0,std::cos(theta),std::sin(theta)),1.);
    }
    for(int j=0;j<12;++j) {
      const double theta=(one_sided?j*.08:j*.5235987755982988);
      add(Vec3(0,std::cos(theta),std::sin(theta)),2.);
    }
    add(Vec3(0,1,1e-12),1.+1e-11); // Nearly parallel, nearly tight.
    add(Vec3(1,.1,.1),.1); // A 2-D projection alone cannot justify deleting this.
    const Vec3 center=group.geometry(initial).head<3>();
    std::vector<CubicProblem::Group::Proof> proofs;
    pruneProjectedSupports(group,initial,1.,1.,&proofs);
    assert(proofs.size()>=6);checkProofs(group,center,proofs,1.,1.);
    int feasible=0;
    for(int trial=0;trial<400;++trial) {
      Eigen::VectorXd x(10);
      for(int i=0;i<2;++i) {
        const double a=.17*(trial+1)*(i+1),b=.31*(trial+2)*(i+1);
        x.segment<2>(5*i)<<.7*std::sin(a),.7*std::cos(a);
        Eigen::Matrix2d rotation;rotation<<std::cos(b),-std::sin(b),std::sin(b),std::cos(b);
        const Eigen::Matrix2d shape=rotation*Eigen::Vector2d(.15,.8).asDiagonal()*rotation.transpose();
        x.segment<3>(5*i+2)<<shape(0,0),shape(0,1),shape(1,1);
      }
      const auto y=group.geometry(x);bool safe=true;
      auto value=[&](const auto& p) {
        return p.normal.dot(y.head<3>())+std::hypot(p.normal.dot(y.segment<3>(3)),p.normal.dot(y.tail<3>()))-p.bound;
      };
      for(const auto& p:group.supports)if(!p.redundant&&value(p)>0)safe=false;
      if(safe){++feasible;for(const auto& p:group.supports)assert(value(p)<1e-10);}
    }
    assert(feasible>0);
  }
  std::cout<<"Projected implication proofs and sufficient line bound passed\n";
}
