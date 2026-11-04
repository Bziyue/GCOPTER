// SPDX-License-Identifier: MIT
#include "cubic_certificates.hpp"
#include <cassert>
#include <iostream>
#include <random>
using namespace gcopter::direct::native;
using namespace gcopter::direct;
static double deBoor(const CubicBasis& b,const std::vector<double>& p,int span,double u) {
  std::array<double,4> d;for(int j=0;j<4;++j)d[j]=p[span+j];
  const int k=span+3;
  for(int r=1;r<=3;++r)for(int j=3;j>=r;--j) {
    const int i=k-3+j;const double a=(u-b.knots[i])/(b.knots[i+4-r]-b.knots[i]);
    d[j]=(1-a)*d[j-1]+a*d[j];
  }
  return d[3];
}
int main() {
  CubicBasis basis({0,.002,.037,.18,.181,.61,1});
  std::vector<V9> p(basis.count);
  std::vector<double> scalar;
  for(int i=0;i<basis.count;++i){p[i].setConstant(std::sin(1.27*i));scalar.push_back(p[i][0]);}
  Geometry g=Geometry::fromCubic(basis,p);
  for(int i=0;i<int(g.pieces());++i)for(int j=0;j<=20;++j) {
    const double t=j/20.,u=g.breaks[i]+t*(g.breaks[i+1]-g.breaks[i]);
    assert(std::abs(value<3>(g.curve<3>(i),t)[0]-deBoor(basis,scalar,i,u))<2e-13);
    const auto halves=split<3>(g.curve<3>(i),.37);
    const V9 split_value=t<.37?value<3>(halves.first,t/.37):value<3>(halves.second,(t-.37)/.63);
    assert((split_value-value<3>(g.curve<3>(i),t)).norm()<3e-13);
    assert((value<5>(elevate<3,5>(g.curve<3>(i)),t)-value<3>(g.curve<3>(i),t)).norm()<3e-13);
    const auto b=basis.jet(i,u);double d=0,dd=0;
    for(int k=0;k<4;++k){d+=b(1,k)*scalar[i+k];dd+=b(2,k)*scalar[i+k];}
    const auto q=g.curve<3>(i);const double h=g.breaks[i+1]-g.breaks[i];
    assert(std::abs(d-value<2>(Curve<2,V9>{3/h*(q[1]-q[0]),3/h*(q[2]-q[1]),3/h*(q[3]-q[2])},t)[0])<1e-9);
    assert(std::abs(dd-value<1>(Curve<1,V9>{6/(h*h)*(q[2]-2*q[1]+q[0]),6/(h*h)*(q[3]-2*q[2]+q[1])},t)[0])<1e-5);
  }
  double weights=0;for(int i=0;i<basis.count;++i)weights+=basis.weight(i);assert(std::abs(weights-1)<1e-14);
  assert(quality(g,1).c2_error<1e-5);
  bool rejected=false;try{CubicBasis b({0,.5,.5,1});}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
  // Constant cross section on a nonuniform straight path: exact positive J.
  for(int i=0;i<basis.count;++i){p[i].setZero();p[i][0]=10*basis.greville(i);p[i][4]=1;p[i][8]=2;}
  g=Geometry::fromCubic(basis,p);
  assert(certify(g,{}).safe());
  auto c=naturalCubic(g),q=minimumJerk(g);
  assert(quality(c,10).c2_error<1e-7);assert(quality(q,10).c2_error<1e-6);
  for(int j=0;j<100;++j) {
    const double u=j/99.;assert((g.section(u).center-c.section(u).center).norm()<1e-10);
    assert((g.section(u).center-q.section(u).center).norm()<2e-6);
  }
  // Minimum-jerk variational property on fixed arbitrary nodal values.
  CubicBasis regular({0,.1,.23,.4,.61,.83,1});
  std::vector<V9> pc(regular.count);
  for(int i=0;i<regular.count;++i){pc[i]=V9::Zero();pc[i][0]=regular.greville(i)*10;pc[i][1]=std::sin(i*.7);pc[i][4]=1+.2*std::sin(i);pc[i][8]=2;}
  g=Geometry::fromCubic(regular,pc);c=naturalCubic(g);q=minimumJerk(g);
  assert(quality(c,10).center_energy+quality(c,10).shape_energy<=quality(g,10).center_energy+quality(g,10).shape_energy+1e-10);
  assert(quality(q,10).jerk_energy<=quality(g,10).jerk_energy+1e-10);
  assert(quality(q,10).jerk_energy<=quality(c,10).jerk_energy+1e-10);
  for(double u:g.breaks){assert((pack(g.section(u).center,g.section(u).shape)-pack(q.section(u).center,q.section(u).shape)).norm()<1e-10);}
  // Full-disk certificate rejects twist/expansion folding missed at z=0.
  Curve<3,V9> fold;
  for(int i=0;i<4;++i){fold[i]=V9::Zero();fold[i][0]=i/3.;fold[i][3]=2*i/3.;fold[i][4]=1;fold[i][8]=1;}
  Certificate report;assert(!jacobianCertificate<3>(fold,report));
  // Analytic cone polynomial versus direct derivatives at independent points.
  const auto curve=g.curve<3>(2);const auto cone=jacobianCoefficients<3>(curve);
  for(double t:{.013,.31,.779,.99}) {
    const V9 y=value<3>(curve,t);
    const V9 dy=value<2>(Curve<2,V9>{3*(curve[1]-curve[0]),3*(curve[2]-curve[1]),3*(curve[3]-curve[2])},t);
    const Vec3 n=shape(y).col(0).cross(shape(y).col(1));Vec3 expected;
    expected<<n.dot(dy.head<3>()),(shape(dy).transpose()*n);
    assert((expected-value<8>(cone,t)).norm()<1e-11);
  }
  std::cout<<"nonuniform extraction, derivatives, C2, subdivision, elevation, energies and full-disk certificates passed\n";
}
