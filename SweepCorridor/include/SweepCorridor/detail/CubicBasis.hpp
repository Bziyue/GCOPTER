// SPDX-License-Identifier: MIT
#pragma once
#include <Eigen/Core>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace SweepCorridor::internal::cubic {
// A clamped, nonuniform cubic with SIMPLE internal knots. Extraction uses
// endpoint value/derivative identities, without collocation inversion/fitting.
struct CubicBasis {
  struct Span { std::array<int,4> ids; Eigen::Matrix4d extraction; };
  std::vector<double> breaks, knots;
  std::vector<Span> spans;
  int count = 0;
  explicit CubicBasis(const std::vector<double>& b) : breaks(b) {
    if(b.size()<2 || b.front()!=0 || b.back()!=1)
      throw std::invalid_argument("cubic breaks must span [0,1]");
    for(size_t i=1;i<b.size();++i)
      if(!std::isfinite(b[i]) || b[i]<=b[i-1])
        throw std::invalid_argument("cubic breaks must increase strictly");
    knots.insert(knots.end(),4,0.);
    knots.insert(knots.end(),b.begin()+1,b.end()-1);
    knots.insert(knots.end(),4,1.);
    count=int(knots.size())-4;
    for(int k=0;k<int(b.size())-1;++k) {
      Span s; for(int j=0;j<4;++j) s.ids[j]=k+j;
      const auto left=jet(k,b[k]), right=jet(k,b[k+1]);
      const double h=b[k+1]-b[k];
      s.extraction.row(0)=left.row(0);
      s.extraction.row(1)=left.row(0)+h/3*left.row(1);
      s.extraction.row(2)=right.row(0)-h/3*right.row(1);
      s.extraction.row(3)=right.row(0);
      // Simple cubic knots: the two interior Bezier controls depend on
      // precisely two B-spline controls. These are algebraic zeros, even
      // when endpoint/derivative cancellation leaves a rounding residue.
      s.extraction(0,3)=s.extraction(1,0)=s.extraction(1,3)=0;
      s.extraction(2,0)=s.extraction(2,3)=s.extraction(3,0)=0;
      spans.push_back(s);
    }
  }
  // Local polynomial limits at BOTH ends of the chosen nonzero span.
  // Column j belongs to global control k+j; rows are derivative orders 0..2.
  Eigen::Matrix<double,3,4> jet(int k,double u) const {
    const int at=k+3;
    std::array<std::array<double,8>,4> v{};
    std::array<std::array<double,8>,4> d{}, dd{};
    v[0][3]=1.;
    for(int p=1;p<=3;++p) for(int j=3-p;j<=3;++j) {
      const int i=at-3+j;
      const double a=knots[i+p]-knots[i], b=knots[i+p+1]-knots[i+1];
      if(a>0) {
        v[p][j]+=(u-knots[i])/a*v[p-1][j];
        d[p][j]+=p/a*v[p-1][j];
        dd[p][j]+=p/a*d[p-1][j];
      }
      if(b>0) {
        v[p][j]+=(knots[i+p+1]-u)/b*v[p-1][j+1];
        d[p][j]-=p/b*v[p-1][j+1];
        dd[p][j]-=p/b*d[p-1][j+1];
      }
    }
    Eigen::Matrix<double,3,4> out;
    for(int j=0;j<4;++j) { out(0,j)=v[3][j];out(1,j)=d[3][j];out(2,j)=dd[3][j]; }
    return out;
  }
  double greville(int i) const { return (knots[i+1]+knots[i+2]+knots[i+3])/3.; }
  double weight(int i) const { return (knots[i+4]-knots[i])/4.; }
};
}
