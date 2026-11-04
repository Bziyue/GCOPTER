#pragma once
// SPDX-License-Identifier: MIT
#include "CubicOptimizer.hpp"
#include "CubicVolume.hpp"
#include <chrono>
#include <limits>

namespace SweepCorridor::internal::cubic {
using M2=Eigen::Matrix2d;
inline M2 shapeMatrix(const Eigen::VectorXd& x,int offset) {
  M2 s;s<<x[offset+3],x[offset+4],x[offset+4],x[offset+5];return s;
}
inline double positiveRoot(double a,double b,double c) {
  if(a==0)return b<0?-c/b:INFINITY;
  const double scale=std::max({std::abs(a),std::abs(b),std::abs(c)});
  a/=scale;b/=scale;c/=scale;
  const double discriminant=b*b-4*a*c;
  if(discriminant<0)return INFINITY;
  const double q=-.5*(b+std::copysign(std::sqrt(discriminant),b));
  double result=INFINITY;
  for(double r:{q/a,q!=0?c/q:INFINITY})if(r>0)result=std::min(result,r);
  return result;
}
inline double CubicProblem::LineModel::delta(double alpha)const {
  double out=alpha*(linear+alpha*quadratic);
  for(const auto& t:terms) {
    const double change=alpha*(t.b+alpha*t.a);
    if(change<=-1)return INFINITY;
    out-=t.weight*std::log1p(change);
  }
  return out;
}
inline double CubicProblem::LineModel::upperBound(double alpha)const {
  double out=alpha*(linear+alpha*quadratic),magnitude=std::abs(out);
  for(const auto& t:terms) {
    const double change=alpha*(t.b+alpha*t.a);
    if(change<=-1)return INFINITY;
    // For q >= 0: log(1+q) >= 2q/(2+q).
    // For -1 < q < 0: log(1+q) >= q(2+q)/(2(1+q)).
    // Both bounds match the linear and quadratic Taylor terms at zero.
    const double lower=change>=0?change/(1+.5*change):
                                 change*(2+change)/(2*(1+change));
    const double term=-t.weight*lower;
    out+=term;magnitude+=std::abs(term);
  }
  // Cover accumulation/roundoff before using this sufficient acceptance test.
  return out+8*std::numeric_limits<double>::epsilon()*(terms.size()+2)*(1+magnitude);
}
inline CubicProblem::LineModel CubicProblem::prepareLine(const Eigen::VectorXd& x,
    const Eigen::VectorXd& direction,double mu)const {
  LineModel line;line.terms.reserve(geometry->count+cuts.size()+4*model.basis.count);
  line.linear=direction.dot(quadratic.multiply(x)+linear);
  line.quadratic=.5*direction.dot(quadratic.multiply(direction));
  double limit=INFINITY;
  auto append=[&](double a,double b,double c,double weight) {
    line.terms.push_back({a/c,b/c,weight});
  };
  auto boundary=[&](double a,double b,double c,double value,double derivative) {
    const double alpha=std::min(1.,limit);
    if((a*alpha+b)*alpha+c>0&&value+alpha*derivative>0)return;
    if(derivative<0)limit=std::min(limit,-value/derivative);
    limit=std::min(limit,positiveRoot(a,b,c));
  };
  auto spd=[&](const M2& s,const M2& d,double weight,bool constrain) {
    const double a=d.determinant(),b=s(0,0)*d(1,1)+s(1,1)*d(0,0)-2*s(0,1)*d(0,1),c=s.determinant();
    append(a,b,c,weight);
    if(constrain)boundary(a,b,c,s(0,0),d(0,0));
  };
  for(int i=0;i<model.basis.count;++i) {
    const M2 s=shapeMatrix(x,6*i),d=shapeMatrix(direction,6*i);
    spd(s,d,settings.area_weight*model.basis.weight(i),false);
    spd(s-options.min_radius*M2::Identity(),d,mu,true);
    spd(options.max_radius*M2::Identity()-s,-d,mu,true);
    const auto c=x.segment<3>(6*i),v=direction.segment<3>(6*i);
    const double a=-v.squaredNorm(),b=-2*c.dot(v),gap=options.max_offset*options.max_offset-c.squaredNorm();
    append(a,b,gap,mu);boundary(a,b,gap,1,0);
  }
  auto cone=[&](const Vec3& v,const Vec3& d,double weight) {
    const double a=d[0]*d[0]-d.tail<2>().squaredNorm();
    const double b=2*(v[0]*d[0]-v.tail<2>().dot(d.tail<2>()));
    const double c=v[0]*v[0]-v.tail<2>().squaredNorm();
    boundary(a,b,c,v[0],d[0]);
    append(a,b,c,weight);
  };
  for(const auto& group:geometry->groups) {
    const V9 y=group.geometry(x);
    const V9 d=group.direction(direction);
    for(size_t j=0;j<group.retained;++j) {
      const auto& p=group.supports[j];
      const Vec3 v=p.project(y);
      cone(v,p.projectDirection(d),mu);
    }
  }
  for(const auto& c:cuts) {
    Vec3 v=c.apply(x);v[0]=c.bound-v[0];
    auto d=c.apply(direction);d[0]=-d[0];cone(v,d,mu+settings.progress_weight/model.basis.count);
  }
  line.step=std::min(1.,settings.fraction_boundary*limit);
  return line;
}

} // namespace SweepCorridor::internal::cubic
