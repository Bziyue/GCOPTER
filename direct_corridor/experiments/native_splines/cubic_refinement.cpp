// SPDX-License-Identifier: MIT
#include "cubic_optimizer.hpp"
#include "cubic_volume.hpp"
#include "support_barrier.hpp"
#include <chrono>
#include <limits>

namespace gcopter::direct::native {
using M2=Eigen::Matrix2d;
static M2 shapeMatrix(const Eigen::VectorXd& x,int offset) {
  M2 s;s<<x[offset+2],x[offset+3],x[offset+3],x[offset+4];return s;
}
static double positiveRoot(double a,double b,double c) {
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
double CubicProblem::LineModel::delta(double alpha)const {
  double out=alpha*(linear+alpha*quadratic);
  for(const auto& t:terms) {
    const double change=alpha*(t.b+alpha*t.a);
    if(change<=-1)return INFINITY;
    out-=t.weight*std::log1p(change);
  }
  for(const auto& t:distances) {
    const double square_change=alpha*(t.b+alpha*t.a);
    const double radius=std::sqrt(std::max(support_epsilon*support_epsilon,t.c+square_change));
    const double change=(alpha*t.derivative-square_change/(radius+t.radius))/t.gap;
    if(change<=-1)return INFINITY;
    out-=t.weight*std::log1p(change);
  }
  return out;
}
double CubicProblem::LineModel::upperBound(double alpha)const {
  if(!distances.empty())return INFINITY;
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
CubicProblem::LineModel CubicProblem::prepareLine(const Eigen::VectorXd& x,
    const Eigen::VectorXd& direction,double mu,const Evaluation* cache)const {
  LineModel line;line.terms.reserve(cuts.size()+4*model.basis.count);
  if(settings.distance_barrier)line.distances.reserve(geometric_cuts);
  line.linear=cache?direction.dot(cache->energy_gradient):direction.dot(quadratic.multiply(x)+linear);
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
    const M2 s=shapeMatrix(x,5*i),d=shapeMatrix(direction,5*i);
    spd(s,d,model.basis.weight(i),false);
    spd(s-options.min_radius*M2::Identity(),d,mu,true);
    spd(options.max_radius*M2::Identity()-s,-d,mu,true);
    const auto c=x.segment<2>(5*i),v=direction.segment<2>(5*i);
    const double a=-v.squaredNorm(),b=-2*c.dot(v),gap=options.max_offset*options.max_offset-c.squaredNorm();
    append(a,b,gap,mu);boundary(a,b,gap,1,0);
  }
  auto cone=[&](const Vec3& v,const Vec3& d,bool active,bool smooth=false) {
    const double a=d[0]*d[0]-d.tail<2>().squaredNorm();
    const double b=2*(v[0]*d[0]-v.tail<2>().dot(d.tail<2>()));
    const double c=v[0]*v[0]-v.tail<2>().squaredNorm()-(smooth?support_epsilon*support_epsilon:0);
    boundary(a,b,c,v[0],d[0]);
    if(active) {
      if(smooth) {
        const double radius=supportRadius(v);
        line.distances.push_back({d.tail<2>().squaredNorm(),2*v.tail<2>().dot(d.tail<2>()),
          radius*radius,radius,v[0]-radius,d[0],mu});
      } else append(a,b,c,mu);
    }
  };
  size_t id=0;
  for(const auto& group:groups) {
    V9 y;if(!cache)y=group.geometry(x);
    const V9 d=group.direction(direction);
    for(size_t j=0;j<group.boundary_count;++j) {
      const auto& p=group.supports[j];
      const Vec3 v=cache?cache->cones[id+j]:p.project(y);
      cone(v,p.projectDirection(d),
           !p.redundant&&(!settings.grouped_assembly||settings.reference_assembly||p.active),
           settings.distance_barrier&&!p.redundant);
    }
    id+=group.supports.size();
  }
  for(size_t j=geometric_cuts;j<cuts.size();++j) {
    const auto& c=cuts[j];Vec3 v;
    if(cache)v=cache->cones[j];else{v=c.apply(x);v[0]=c.bound-v[0];}
    auto d=c.apply(direction);d[0]=-d[0];cone(v,d,true);
  }
  line.step=std::min(1.,(settings.fraction_boundary>0?settings.fraction_boundary:.99)*limit);
  return line;
}

void CubicProblem::inflate(Eigen::VectorXd& x,OptimizerMetrics& metrics,int sweeps,double fraction)const {
  const auto start=std::chrono::steady_clock::now();
  struct Incident {int cut,local;};
  std::vector<std::vector<Incident>> incident(model.basis.count);
  std::vector<Vec3> values(cuts.size());
  for(size_t j=0;j<cuts.size();++j) {
    const auto& c=cuts[j];values[j]=c.apply(x);values[j][0]=c.bound-values[j][0];
    for(int k=0;k<c.count;++k)if(!c.normals[k].isZero(0))incident[c.first/5+k].push_back({int(j),k});
  }
  Eigen::VectorXd energy_gradient=quadratic.multiply(x)+linear;
  for(int sweep=0;sweep<(sweeps<0?settings.coordinate_sweeps:sweeps);++sweep)for(int order=0;order<model.basis.count;++order) {
    const int node=sweep%2?model.basis.count-1-order:order,offset=5*node;
    const Vec3 s=x.segment<3>(offset+2);
    const double largest=.5*(s[0]+s[2])+std::hypot(.5*(s[0]-s[2]),s[1]);
    double alpha=fraction*(options.max_radius/largest-1);
    for(const auto& entry:incident[node]) {
      const auto& n=cuts[entry.cut].normals[entry.local];const auto& v=values[entry.cut];
      const Eigen::Vector2d d(s[0]*n[0]+s[1]*n[1],s[1]*n[0]+s[2]*n[1]);
      const double a=d.squaredNorm();if(a==0)continue;
      const double b=v.tail<2>().dot(d),gap=v[0]*v[0]-v.tail<2>().squaredNorm();
      const double root=std::sqrt(b*b+a*gap);
      alpha=std::min(alpha,fraction*(b>=0?gap/(root+b):(root-b)/a));
    }
    // Minimize the exact logdet-plus-bending objective along this radius ray.
    const double l=s.dot(energy_gradient.segment<3>(offset+2));
    double q=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)q+=s[i]*quadratic.get(offset+2+i,offset+2+j)*s[j];
    q=std::max(0.,q);const double c=l-2*model.basis.weight(node);
    if(c>=0)continue;
    const double optimum=q>0?positiveRoot(q,q+l,c):(l>0?-c/l:INFINITY);
    alpha=std::min(alpha,optimum);if(alpha<=1e-10)continue;
    x.segment<3>(offset+2)+=alpha*s;++metrics.coordinate_updates;
    for(const auto& entry:incident[node]) {
      const auto& n=cuts[entry.cut].normals[entry.local];
      values[entry.cut][1]+=alpha*(s[0]*n[0]+s[1]*n[1]);
      values[entry.cut][2]+=alpha*(s[1]*n[0]+s[2]*n[1]);
    }
    for(int row=std::max(0,offset+2-quadratic.width);row<=std::min(int(x.size())-1,offset+4+quadratic.width);++row)
      for(int j=0;j<3;++j)energy_gradient[row]+=alpha*quadratic.get(row,offset+2+j)*s[j];
  }
  metrics.coordinate_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
void CubicProblem::refineVolume(Eigen::VectorXd& x,OptimizerMetrics& metrics,
    const std::function<bool(const Eigen::VectorXd&)>& certificate)const {
  const auto start=std::chrono::steady_clock::now();
  for(int iteration=0;iteration<settings.volume_refinements;++iteration) {
    Eigen::VectorXd gradient,unused,direction;BandedSPD hessian;
    const double before=sweptVolume(model,x,&gradient);
    objective(x,.00001/model.basis.count,&unused,&hessian);
    for(int id:{0,1,int(x.size())-5,int(x.size())-4}){gradient[id]=0;hessian.fix(id);}
    double damping=0;
    while(!hessian.factor(damping)){damping=damping==0?1e-8:damping*10;if(damping>1e6)break;}
    if(damping>1e6)break;
    hessian.solveInto(gradient,direction);
    if(equalities.rows()) {
      Eigen::MatrixXd inverse(x.size(),equalities.rows());
      for(int j=0;j<equalities.rows();++j)inverse.col(j)=hessian.solve(equalities.row(j).transpose());
      const Eigen::MatrixXd schur=equalities*inverse;
      direction-=inverse*schur.ldlt().solve(equalities*direction);
    }
    double movement=0;
    for(int i=0;i<model.basis.count;++i) {
      const auto p=direction.segment<5>(5*i);
      movement=std::max(movement,p.head<2>().norm()+std::sqrt(p[2]*p[2]+2*p[3]*p[3]+p[4]*p[4]));
    }
    if(movement<=1e-12)break;
    direction*=std::min(1.,settings.volume_trust/movement);
    const double first=gradient.dot(direction),plus=sweptVolume(model,x+direction)-before,
                 minus=sweptVolume(model,x-direction)-before;
    const double second=.5*(plus+minus),third=.5*(plus-minus)-first;
    const double upper=feasibleStep(x,direction);double alpha=upper;
    auto gain=[&](double t){return t*(first+t*(second+t*third));};
    const double a=3*third,b=2*second,c=first,discriminant=b*b-4*a*c;
    if(a!=0&&discriminant>=0) {
      const double q=-.5*(b+std::copysign(std::sqrt(discriminant),b));
      for(double root:{q/a,q!=0?c/q:INFINITY})if(root>0&&root<upper&&gain(root)>gain(alpha))alpha=root;
    }else if(a==0&&b<0){const double root=-c/b;if(root>0&&root<alpha)alpha=root;}
    bool accepted=false;
    for(int trial=0;trial<10&&gain(alpha)>1e-10;++trial,alpha*=.5) {
      const Eigen::VectorXd candidate=x+alpha*direction;
      if(certificate(candidate)) {
        const double after=sweptVolume(model,candidate);
        if(after>before){x=candidate;metrics.volume_gain+=after-before;++metrics.volume_steps;accepted=true;}
        break;
      }
    }
    if(!accepted)break;
  }
  metrics.volume_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
} // namespace gcopter::direct::native
