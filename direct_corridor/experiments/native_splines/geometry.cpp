// SPDX-License-Identifier: MIT
#include "geometry.hpp"
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>
#include <Eigen/Eigenvalues>
#include <sstream>

namespace gcopter::direct::native {
// The legacy query/trajectory interface stays quintic. Cubic outputs enter it
// through exact degree elevation, never through a curve fit.
struct GeometryAccess {
  static Corridor make(const native::Geometry& g) {
    const auto q=g.degree==3?native::elevateGeometry(g):g;
    Corridor c;c.knots_=q.breaks;
    for(const auto& v:q.controls) {c.centers_.push_back(v.head<3>());c.shapes_.push_back(native::shape(v));}
    for(double u:q.breaks) c.nodes_.push_back({u,q.section(u)});
    return c;
  }
};
}
namespace gcopter::direct::native {
template<int D> static Section evaluate(const Curve<D,V9>& p,double t,double h) {
  Curve<D-1,V9> d;Curve<D-2,V9> dd;
  for(int j=0;j<D;++j) d[j]=D/h*(p[j+1]-p[j]);
  for(int j=0;j<D-1;++j) dd[j]=(D-1)/h*(d[j+1]-d[j]);
  const V9 y=value<D>(p,t),dy=value<D-1>(d,t),ddy=value<D-2>(dd,t);
  return {y.head<3>(),dy.head<3>(),ddy.head<3>(),shape(y),shape(dy),shape(ddy)};
}
Section Geometry::section(double u) const {
  const int i=std::clamp(int(std::upper_bound(breaks.begin(),breaks.end(),u)-breaks.begin())-1,0,int(pieces())-1);
  const double h=breaks[i+1]-breaks[i],t=(u-breaks[i])/h;
  return degree==3?evaluate<3>(curve<3>(i),t,h):evaluate<5>(curve<5>(i),t,h);
}
Corridor Geometry::quinticAdapter() const {return GeometryAccess::make(*this);}
Geometry Geometry::fromCorridor(const Corridor& c) {
  Geometry g;g.degree=5;g.breaks=c.knots();
  for(size_t j=0;j<c.centerControls().size();++j) g.controls.push_back(pack(c.centerControls()[j],c.shapeControls()[j]));
  return g;
}
Geometry Geometry::fromCubic(const CubicBasis& b,const std::vector<V9>& values) {
  Geometry g;g.breaks=b.breaks;
  for(const auto& s:b.spans) for(int j=0;j<4;++j) {
    V9 v=V9::Zero();for(int k=0;k<4;++k) v+=s.extraction(j,k)*values[s.ids[k]];
    g.controls.push_back(v);
  }
  return g;
}
void Geometry::write(const std::string& path) const {
  std::ofstream f(path);f<<std::setprecision(17);
  f<<"degree,u0,u1,cx,cy,cz,m00,m10,m20,m01,m11,m21\n";
  for(size_t i=0;i<pieces();++i) for(int j=0;j<=degree;++j) {
    f<<degree<<','<<breaks[i]<<','<<breaks[i+1];
    for(double v:controls[(degree+1)*i+j]) f<<','<<v;
    f<<'\n';
  }
}
Geometry Geometry::read(const std::string& path) {
  std::ifstream file(path);if(!file)throw std::runtime_error("geometry file not found");
  std::string line;std::getline(file,line);
  if(line!="degree,u0,u1,cx,cy,cz,m00,m10,m20,m01,m11,m21")throw std::runtime_error("unknown native geometry format");
  Geometry g;double last_begin=-1,last_end=-1;int rows=0;
  while(std::getline(file,line)) {
    std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
    int degree;double a,b;V9 v;
    if(!(row>>degree>>a>>b)||(degree!=3&&degree!=5)||!(a>=0&&b>a&&b<=1))throw std::runtime_error("invalid native geometry interval");
    for(double& x:v)if(!(row>>x)||!std::isfinite(x))throw std::runtime_error("invalid native geometry coefficient");
    if(g.controls.empty()){g.degree=degree;g.breaks.push_back(a);}
    if(degree!=g.degree)throw std::runtime_error("mixed geometry degrees");
    if(rows%(degree+1)==0){if(a!=g.breaks.back())throw std::runtime_error("geometry gap");g.breaks.push_back(b);last_begin=a;last_end=b;}
    else if(a!=last_begin||b!=last_end)throw std::runtime_error("incomplete native control block");
    g.controls.push_back(v);++rows;
  }
  if(rows==0||rows%(g.degree+1)||g.breaks.front()!=0||g.breaks.back()!=1)throw std::runtime_error("incomplete native geometry");
  return g;
}
Geometry elevateGeometry(const Geometry& c) {
  Geometry g;g.degree=5;g.breaks=c.breaks;
  for(int i=0;i<int(c.pieces());++i) g.append<5>(elevate<3,5>(c.curve<3>(i)));
  return g;
}
Geometry naturalCubic(const Geometry& src) {
  const auto& b=src.breaks;const int n=b.size();
  std::vector<V9> y(n),dd(n,V9::Zero()),d(n);
  std::vector<double> diag(n,1.),upper(n,0.);
  for(int i=0;i<n;++i) {auto s=src.section(b[i]);y[i]=pack(s.center,s.shape);}
  for(int i=1;i<n-1;++i) {
    const double a=b[i]-b[i-1],h=b[i+1]-b[i];
    diag[i]=2*(a+h)-a*upper[i-1]/diag[i-1];upper[i]=h;
    dd[i]=6*((y[i+1]-y[i])/h-(y[i]-y[i-1])/a)-a*dd[i-1]/diag[i-1];
  }
  for(int i=n-2;i>0;--i) dd[i]=(dd[i]-upper[i]*dd[i+1])/diag[i];
  for(int i=0;i<n-1;++i) d[i]=(y[i+1]-y[i])/(b[i+1]-b[i])-(b[i+1]-b[i])*(2*dd[i]+dd[i+1])/6;
  const double h=b.back()-b[n-2];d.back()=(y.back()-y[n-2])/h+h*(dd[n-2]+2*dd.back())/6;
  Geometry g;g.breaks=b;
  for(int i=0;i<n-1;++i) {const double h=b[i+1]-b[i];g.append<3>({y[i],y[i]+h*d[i]/3,y[i+1]-h*d[i+1]/3,y[i+1]});}
  return g;
}
// Quintic Hermite controls from [value0,velocity0,accel0,value1,velocity1,accel1].
static Eigen::Matrix<double,6,6> hermite(double h) {
  Eigen::Matrix<double,6,6> t=Eigen::Matrix<double,6,6>::Zero();
  t(0,0)=t(1,0)=t(2,0)=t(3,3)=t(4,3)=t(5,3)=1;
  t(1,1)=h/5;t(2,1)=2*h/5;t(2,2)=h*h/20;
  t(3,4)=-2*h/5;t(3,5)=h*h/20;t(4,4)=-h/5;return t;
}
Geometry minimumJerk(const Geometry& src) {
  const int n=src.breaks.size();
  Geometry out;out.degree=5;out.breaks=src.breaks;
  if(n==2) return elevateGeometry(naturalCubic(src)); // Straight minimizer of the nonunique two-node problem.
  std::vector<V9> y(n);for(int i=0;i<n;++i){const auto s=src.section(src.breaks[i]);y[i]=pack(s.center,s.shape);}
  // Work in average-span coordinates to balance velocity/acceleration units.
  const double scale=n-1;
  std::vector<Eigen::Triplet<double>> entries;
  Eigen::Matrix<double,Eigen::Dynamic,9> rhs=Eigen::Matrix<double,Eigen::Dynamic,9>::Zero(2*n,9);
  Eigen::Matrix3d gram;for(int i=0;i<3;++i)for(int j=0;j<3;++j)gram(i,j)=binomial(2,i)*binomial(2,j)/(5*binomial(4,i+j));
  Eigen::Matrix<double,3,6> third=Eigen::Matrix<double,3,6>::Zero();
  for(int j=0;j<3;++j)third.row(j).segment<4>(j)<<-60,180,-180,60;
  for(int i=0;i<n-1;++i) {
    const double h=(src.breaks[i+1]-src.breaks[i])*scale;
    const Eigen::Matrix<double,3,6> t=third*hermite(h);
    const Eigen::Matrix<double,6,6> q=t.transpose()*gram*t/std::pow(h,5);
    const int a[]={1,2,4,5},ids[]={2*i,2*i+1,2*i+2,2*i+3};
    for(int j=0;j<4;++j) {
      rhs.row(ids[j])-=q(a[j],0)*y[i].transpose()+q(a[j],3)*y[i+1].transpose();
      for(int k=0;k<4;++k)entries.emplace_back(ids[j],ids[k],q(a[j],a[k]));
    }
  }
  Eigen::SparseMatrix<double> a(2*n,2*n);a.setFromTriplets(entries.begin(),entries.end());
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solve(a);
  if(solve.info()!=Eigen::Success)throw std::runtime_error("minimum-jerk factorization failed");
  const Eigen::Matrix<double,Eigen::Dynamic,9> jets=solve.solve(rhs);
  for(int i=0;i<n-1;++i) {
    Eigen::Matrix<double,6,9> data;
    data.row(0)=y[i];data.row(1)=jets.row(2*i);data.row(2)=jets.row(2*i+1);
    data.row(3)=y[i+1];data.row(4)=jets.row(2*i+2);data.row(5)=jets.row(2*i+3);
    Eigen::Matrix<double,6,9> p=hermite((src.breaks[i+1]-src.breaks[i])*scale)*data;
    for(int j=0;j<6;++j)out.controls.push_back(p.row(j));
  }
  return out;
}
Quality quality(const Geometry& g,double length) {
  Quality q;
  // Physical arc reference s=L*u makes energies and C2 errors comparable.
  for(int i=0;i<int(g.pieces());++i) {
    const double h=g.breaks[i+1]-g.breaks[i];
    bool folded=false;
    for(int j=0;j<=64;++j) {
      const double t=j/64.;
      const Section s=g.degree==3?evaluate<3>(g.curve<3>(i),t,h):evaluate<5>(g.curve<5>(i),t,h);
      const Vec3 n=s.shape.col(0).cross(s.shape.col(1));
      const double margin=n.dot(s.first)-(s.shape_first.transpose()*n).norm();
      const double scale=n.norm()*s.first.norm();
      q.normalized_jacobian=std::min(q.normalized_jacobian,margin/std::max(1e-30,scale));
      folded|=margin<=0;
      const auto gram=(s.shape.transpose()*s.shape).eval();
      const double disc=std::hypot(gram(0,0)-gram(1,1),2*gram(0,1));
      q.min_axis=std::min(q.min_axis,std::sqrt(std::max(0.,.5*(gram.trace()-disc))));
      q.max_curvature=std::max(q.max_curvature,s.first.cross(s.second).norm()/std::pow(s.first.norm(),3));
    }
    q.folded_spans+=folded;
    if(i) for(int order=0;order<3;++order) {
      const auto l=g.degree==3?evaluate<3>(g.curve<3>(i-1),1,g.breaks[i]-g.breaks[i-1]):evaluate<5>(g.curve<5>(i-1),1,g.breaks[i]-g.breaks[i-1]);
      const auto r=g.degree==3?evaluate<3>(g.curve<3>(i),0,h):evaluate<5>(g.curve<5>(i),0,h);
      const V9 a=order==0?pack(l.center,l.shape):order==1?pack(l.first,l.shape_first):pack(l.second,l.shape_second);
      const V9 b=order==0?pack(r.center,r.shape):order==1?pack(r.first,r.shape_first):pack(r.second,r.shape_second);
      q.c2_error=std::max(q.c2_error,(a-b).norm()/std::pow(length,order));
    }
    constexpr double x[]={-.9324695142031521,-.6612093864662645,-.2386191860831969,.2386191860831969,.6612093864662645,.9324695142031521};
    constexpr double w[]={.1713244923791704,.3607615730481386,.4679139345726910,.4679139345726910,.3607615730481386,.1713244923791704};
    for(int k=0;k<6;++k) {
      const auto s=g.section(g.breaks[i]+h*(x[k]+1)/2);
      q.center_energy+=.5*h*w[k]*s.second.squaredNorm()/std::pow(length,3);
      q.shape_energy+=.5*h*w[k]*s.shape_second.squaredNorm()/std::pow(length,3);
      V9 third;
      if(g.degree==3) {const auto p=g.curve<3>(i);third=6/std::pow(h,3)*(p[3]-3*p[2]+3*p[1]-p[0]);}
      else {const auto p=g.curve<5>(i);Curve<2,V9> d;for(int j=0;j<3;++j)d[j]=60/std::pow(h,3)*(p[j+3]-3*p[j+2]+3*p[j+1]-p[j]);third=value<2>(d,(x[k]+1)/2);}
      q.jerk_energy+=.5*h*w[k]*third.squaredNorm()/std::pow(length,5);
    }
  }
  return q;
}
}
