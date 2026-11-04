// SPDX-License-Identifier: MIT
#pragma once
#include <Eigen/Core>
#include <cmath>
#include <algorithm>
namespace SweepCorridor::internal::cubic {
// Symmetric lower-band storage. No dense N*N intermediate in assembly/solve.
struct BandedSPD {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> data,lower;
  Eigen::VectorXd scale;
  int n=0;
  static constexpr int width=19; // Four cubic controls, five variables per control.
  BandedSPD()=default;
  explicit BandedSPD(int size):data(Eigen::MatrixXd::Zero(size,width+1)),n(size){}
  void add(int i,int j,double v){if(i<j)std::swap(i,j);data(i,i-j)+=v;}
  double get(int i,int j)const{if(i<j)std::swap(i,j);return i-j<=width?data(i,i-j):0.;}
  Eigen::VectorXd multiply(const Eigen::VectorXd& x)const {
    Eigen::VectorXd y=Eigen::VectorXd::Zero(n);
    for(int i=0;i<n;++i){y[i]+=data(i,0)*x[i];for(int d=1;d<=std::min(i,width);++d){y[i]+=data(i,d)*x[i-d];y[i-d]+=data(i,d)*x[i];}}
    return y;
  }
  void fix(int i) {
    for(int j=std::max(0,i-width);j<=std::min(n-1,i+width);++j) {
      const int row=std::max(i,j),col=std::min(i,j);data(row,row-col)=i==j?1:0;
    }
  }
  bool factor(double damping=0) {
    scale.resize(n);lower.resize(n,width+1);
    for(int i=0;i<n;++i)scale[i]=1/std::sqrt(data(i,0)+damping);
    for(int i=0;i<n;++i) {
      const auto* row=data.data()+i*(width+1);
      auto* li=lower.data()+i*(width+1);
      const int begin=std::max(0,i-width);
      for(int j=begin;j<=i;++j) {
        const auto* lj=lower.data()+j*(width+1);
        double v=(row[i-j]+(i==j?damping:0))*scale[i]*scale[j];
        // Every accessed entry has already been written in this factorization.
        // Keep the subtraction order identical to the generic banded solver.
        for(int k=begin;k<j;++k)v-=li[i-k]*lj[j-k];
        if(i==j){if(!(v>1e-15))return false;li[0]=std::sqrt(v);}
        else li[i-j]=v/lj[0];
      }
    }
    return true;
  }
  void solveInto(const Eigen::VectorXd& rhs,Eigen::VectorXd& x,double sign=1)const {
    x=sign*rhs.cwiseProduct(scale);
    for(int i=0;i<n;++i){for(int d=1;d<=std::min(i,width);++d)x[i]-=lower(i,d)*x[i-d];x[i]/=lower(i,0);}
    for(int i=n-1;i>=0;--i){for(int d=1;d<=std::min(n-1-i,width);++d)x[i]-=lower(i+d,d)*x[i+d];x[i]/=lower(i,0);}
    x.array()*=scale.array();
  }
  Eigen::VectorXd solve(const Eigen::VectorXd& rhs)const {
    Eigen::VectorXd x;solveInto(rhs,x);return x;
  }
};
}
