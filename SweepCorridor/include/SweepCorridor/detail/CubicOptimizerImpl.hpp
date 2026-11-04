#pragma once
// SPDX-License-Identifier: MIT
#include "CubicOptimizer.hpp"
#include "Certificates.hpp"
#include <Eigen/Eigenvalues>
#include <chrono>
#include <map>
namespace SweepCorridor::internal::cubic {
inline Eigen::Matrix2d matrix(const Eigen::VectorXd& x,int i) {
  Eigen::Matrix2d s;s<<x[6*i+3],x[6*i+4],x[6*i+4],x[6*i+5];return s;
}
inline Eigen::Matrix<double,9,6> CubicModel::jacobian(int i)const {
  Eigen::Matrix<double,9,6> j=Eigen::Matrix<double,9,6>::Zero();
  j.topLeftCorner<3,3>().setIdentity();
  j.block<3,1>(3,3)=frames[i].col(0);j.block<3,1>(3,4)=frames[i].col(1);
  j.block<3,1>(6,4)=frames[i].col(0);j.block<3,1>(6,5)=frames[i].col(1);return j;
}
inline Geometry CubicModel::geometry(const Eigen::VectorXd& values)const {
  std::vector<V9> controls(basis.count);
  for(int i=0;i<basis.count;++i)controls[i]=pack(reference[i]+values.segment<3>(6*i),frames[i]*matrix(values,i));
  return Geometry::fromCubic(basis,controls);
}
inline CubicModel CubicModel::fromSeed(const std::vector<Vec3>& seed,double spacing,double radius,double smoothing,
                                 const std::vector<double>& requested_breaks) {
  std::vector<double> arc(seed.size(),0);std::vector<Vec3> moment(seed.size(),Vec3::Zero());
  for(size_t i=1;i<seed.size();++i){arc[i]=arc[i-1]+(seed[i]-seed[i-1]).norm();moment[i]=moment[i-1]+.5*(arc[i]+arc[i-1])*(seed[i]-seed[i-1]);}
  const double length=arc.back();const int segments=std::max(1,int(std::ceil(length/spacing)));
  auto breaks=requested_breaks;
  if(breaks.empty())for(int i=0;i<=segments;++i)breaks.push_back(double(i)/segments);
  CubicModel out(breaks);out.x.setZero();
  auto segment=[&](double s){return std::clamp(int(std::upper_bound(arc.begin(),arc.end(),s)-arc.begin())-1,0,int(seed.size())-2);};
  auto integral=[&](double s) {
    s=std::clamp(s,0.,length);const int k=segment(s);
    const Vec3 displacement=(s-arc[k])/(arc[k+1]-arc[k])*(seed[k+1]-seed[k]);
    return std::make_pair(Vec3(seed[k]-seed[0]+displacement),Vec3(moment[k]+.5*(s+arc[k])*displacement));
  };
  Vec3 previous=Vec3::Zero(),axis=Vec3::Zero();
  for(int i=0;i<out.basis.count;++i) {
    const double s=length*out.basis.greville(i);const int k=segment(s);
    out.reference[i]=seed[k]+(s-arc[k])/(arc[k+1]-arc[k])*(seed[k+1]-seed[k]);
    Vec3 tangent=(seed[k+1]-seed[k]).normalized();
    if(smoothing>0) {
      const auto a=integral(s-smoothing),b=integral(s),c=integral(s+smoothing);
      const Vec3 average=(1-s/smoothing)*(b.first-a.first)+(b.second-a.second)/smoothing+
                         (1+s/smoothing)*(c.first-b.first)-(c.second-b.second)/smoothing;
      if(average.dot(tangent)>.15*average.norm())tangent=average.normalized();
    }
    if(!i)axis=tangent.unitOrthogonal();else {axis=Eigen::Quaterniond::FromTwoVectors(previous,tangent)*axis;axis-=tangent*tangent.dot(axis);axis.normalize();}
    out.frames[i].col(0)=axis;out.frames[i].col(1)=tangent.cross(axis);previous=tangent;
    out.x[6*i+3]=out.x[6*i+5]=radius;
  }
  return out;
}
inline Vec3 CubicProblem::Cut::apply(const Eigen::VectorXd& x)const {
  Vec3 v=Vec3::Zero();
  for(int k=0;k<count;++k) {
    const double* p=x.data()+first+6*k;const auto& n=normals[k];
    v[0]+=center_sign*center_normals[k].dot(Eigen::Map<const Vec3>(p));
    v[1]+=n[0]*p[3]+n[1]*p[4];v[2]+=n[0]*p[4]+n[1]*p[5];
  }return v;
}
inline bool CubicProblem::Cut::compact() {
  auto zero=[&](int i){return normals[i].isZero(0)&&center_normals[i].isZero(0);};
  while(count&&zero(count-1))--count;
  int start=0;while(start<count&&zero(start))++start;
  for(int k=start;k<count;++k){normals[k-start]=normals[k];center_normals[k-start]=center_normals[k];}
  first+=6*start;count-=start;return count>0;
}
inline void CubicProblem::Cut::accumulate(const Vec3& v,double mu,double slack,
    Eigen::VectorXd& gradient,BandedSPD& hessian)const {
  Eigen::Matrix<double,24,1> reverse;const int size=6*count;const double factor=mu/slack;
  for(int k=0;k<count;++k){const auto& n=normals[k];const int j=6*k;
    for(int a=0;a<3;++a)reverse[size-1-j-a]=-2*center_sign*v[0]*center_normals[k][a];
    reverse[size-4-j]=-2*v[1]*n[0];reverse[size-5-j]=-2*(v[1]*n[1]+v[2]*n[0]);reverse[size-6-j]=-2*v[2]*n[1];}
  for(int j=0;j<size;++j){const double d=reverse[size-1-j];gradient[first+j]-=factor*d;
    hessian.data.row(first+j).head(j+1).noalias()+=(factor/slack*d)*reverse.segment(size-1-j,j+1).transpose();}
  for(int j=0;j<3*count;++j){const int col=6*(j/3)+j%3;const double v=-2*factor*center_normals[j/3][j%3];
    for(int k=0;k<=j;++k)hessian.add(first+col,first+6*(k/3)+k%3,v*center_normals[k/3][k%3]);}
  for(int row=0;row<2;++row)for(int j=0;j<2*count;++j){const int col=6*(j/2)+3+row+j%2;const double v=2*factor*normals[j/2][j%2];
    for(int k=0;k<=j;++k)hessian.add(first+col,first+6*(k/2)+3+row+k%2,v*normals[k/2][k%2]);}
}
inline V9 CubicProblem::Group::direction(const Eigen::VectorXd& x)const {
  V9 out=V9::Zero();
  for(int k=0;k<count;++k){const auto v=x.segment<6>(first+6*k);const auto& f=maps[k];
    out.head<3>()+=center_weights[k]*v.head<3>();
    out.segment<3>(3).noalias()+=f*v.segment<2>(3);out.tail<3>().noalias()+=f*v.tail<2>();}
  return out;
}
inline void CubicProblem::Group::accumulateGradient(const V9& g,Eigen::VectorXd& out)const {
  for(int k=0;k<count;++k){const auto& f=maps[k];const int offset=first+6*k;
    out.segment<3>(offset)+=center_weights[k]*g.head<3>();
    out[offset+3]+=f.col(0).dot(g.segment<3>(3));
    out[offset+4]+=f.col(1).dot(g.segment<3>(3))+f.col(0).dot(g.tail<3>());
    out[offset+5]+=f.col(1).dot(g.tail<3>());}
}
inline void CubicProblem::Group::accumulate(const V9& g,const Eigen::Matrix<double,9,9>& h,
    Eigen::VectorXd& result_gradient,BandedSPD& result_hessian)const {
  accumulateGradient(g,result_gradient);
  for(int b=0;b<count;++b){const auto& f=maps[b];Eigen::Matrix<double,9,6> product;
    product.leftCols<3>()=center_weights[b]*h.leftCols<3>();
    product.col(3).noalias()=h.middleCols<3>(3)*f.col(0);
    product.col(4).noalias()=h.middleCols<3>(3)*f.col(1)+h.rightCols<3>()*f.col(0);
    product.col(5).noalias()=h.rightCols<3>()*f.col(1);
    for(int a=b;a<count;++a){const auto& left=maps[a];Eigen::Matrix<double,6,6> block;
      block.topRows<3>()=center_weights[a]*product.topRows<3>();
      block.row(3).noalias()=left.col(0).transpose()*product.middleRows<3>(3);
      block.row(4).noalias()=left.col(1).transpose()*product.middleRows<3>(3)+left.col(0).transpose()*product.bottomRows<3>();
      block.row(5).noalias()=left.col(1).transpose()*product.bottomRows<3>();
      for(int j=0;j<6;++j){auto row=result_hessian.data.row(first+6*a+j);
        if(a==b)row.head(j+1)+=block.row(j).head(j+1).reverse();
        else row.segment(6*(a-b)+j-5,6)+=block.row(j).reverse();}
    }
  }
}
inline CubicEnergy::CubicEnergy(const CubicModel& m,double length,const OptimizerSettings& settings)
    :quadratic(6*m.basis.count),linear(Eigen::VectorXd::Zero(6*m.basis.count)) {
  // Exact two-point Gauss integration of squared cubic second derivatives.
  for(int i=0;i<int(m.basis.spans.size());++i) {
    const double h=m.basis.breaks[i+1]-m.basis.breaks[i];
    for(double t:{.5-.5/std::sqrt(3.),.5+.5/std::sqrt(3.)}) {
      const auto basis=m.basis.jet(i,m.basis.breaks[i]+h*t);
      Eigen::Matrix<double,9,24> map;V9 reference=V9::Zero();
      for(int k=0;k<4;++k) {
        const double d=basis(2,k)/(length*length);
        map.middleCols<6>(6*k)=d*m.jacobian(i+k);reference.head<3>()+=d*m.reference[i+k];
      }
      const double wc=std::sqrt(settings.center_energy*.5*h*length),wm=std::sqrt(settings.shape_energy*.5*h*length);
      map.topRows<3>()*=wc;map.bottomRows<6>()*=wm;reference.head<3>()*=wc;
      const Eigen::Matrix<double,24,24> gram=2*map.transpose()*map;
      linear.segment<24>(6*i)+=2*map.transpose()*reference;
      for(int a=0;a<24;++a)for(int b=0;b<=a;++b)quadratic.add(6*i+a,6*i+b,gram(a,b));
    }
  }
  for(int i=0;i<m.basis.count;++i)for(int k=0;k<3;++k)quadratic.add(6*i+k,6*i+k,2*settings.offset_weight*m.basis.weight(i));
}
inline CubicProblem::CubicProblem(const CubicModel& m,const std::vector<detail::SupportPlane>& cover,
    double length,const GeometryOptions& o,const OptimizerSettings& s,
    std::shared_ptr<const CubicEnergy> shared_energy,const CubicProblem* geometric_template,
    const ProgressConstraints& progress)
    :model(m),options(o),settings(s),
     energy(shared_energy?std::move(shared_energy):std::make_shared<CubicEnergy>(m,length,s)),
     quadratic(energy->quadratic),linear(energy->linear) {
  if(geometric_template) {
    geometry=geometric_template->geometry;reused_construction=true;
  }else {
    auto data=std::make_shared<GeometricConstraints>();auto& groups=data->groups;
    auto key=[](const auto& p){return std::make_tuple(p.span,p.begin,p.end,p.normal[0],p.normal[1],p.normal[2]);};
    auto planes=cover;
    std::stable_sort(planes.begin(),planes.end(),[&](const auto& a,const auto& b){return key(a)<key(b);});
    size_t count=0;
    for(const auto& p:planes) {
      if(count&&key(p)==key(planes[count-1])){if(p.bound<planes[count-1].bound)planes[count-1]=p;}
      else planes[count++]=p;
    }
    planes.resize(count);
    std::map<std::tuple<int,double,double,int>,size_t> group_ids;
    struct IntervalRows {
      Curve<3,Eigen::Vector4d> weights;
      std::array<int,4> groups{{-1,-1,-1,-1}};
    };
    std::map<std::tuple<int,double,double>,IntervalRows> interval_rows;
    for(const auto& plane:planes) {
      const auto& span=m.basis.spans[plane.span];
      const auto found=interval_rows.try_emplace(std::make_tuple(plane.span,plane.begin,plane.end));
      auto& cached=found.first->second;
      if(found.second) {
        Curve<3,Eigen::Vector4d> basis;
        for(int j=0;j<4;++j)basis[j]=span.extraction.row(j);
        cached.weights=restrictCurve<3>(basis,plane.begin,plane.end);++basis_cache_misses;
      }else ++basis_cache_hits;
      for(int j=0;j<(plane.begin==plane.end?1:4);++j) {
        int local_span=plane.span;
        Eigen::Vector4d weights=cached.weights[j];
        auto group_key=std::make_tuple(plane.span,plane.begin,plane.end,j);
        if(j==0||j==3) {
          const double t=j==0?plane.begin:plane.end;
          // Adjacent cubic spans share this VALUE, with the same three global
          // controls. Give it a single identity, also at subdivision boundaries.
          group_key=t==1?std::make_tuple(plane.span+1,0.,0.,0):std::make_tuple(plane.span,t,t,0);
          if(t==1&&plane.span+1<int(m.basis.spans.size())) {
            local_span=plane.span+1;weights=m.basis.spans[local_span].extraction.row(0);
          }
        }
        const auto& local=m.basis.spans[local_span];
        Cut c;c.first=6*local.ids[0];c.bound=plane.bound;
        for(int k=0;k<4;++k) {
          const int id=local.ids[k];const Eigen::Vector2d n=weights[k]*m.frames[id].transpose()*plane.normal;
          c.bound-=weights[k]*m.reference[id].dot(plane.normal);
          c.normals[k]=n;c.center_normals[k]=weights[k]*plane.normal;
        }
        // Skip constraints which are identically constant (e.g. a straight end cap).
        if(!c.compact()){if(c.bound < -1e-9)throw std::runtime_error("inconsistent constant support plane");continue;}
        size_t group_id;bool create=false;
        if(cached.groups[j]>=0){group_id=cached.groups[j];++group_lookup_reuses;}
        else {
          const auto inserted=group_ids.emplace(group_key,groups.size());group_id=inserted.first->second;create=inserted.second;
          cached.groups[j]=int(group_id);
        }
        if(create) {
          Group group;group.first=6*local.ids[0];group.count=4;
          group.reference.setZero();
          for(int k=0;k<4;++k) {
            group.reference.head<3>()+=weights[k]*m.reference[local.ids[k]];
            group.maps[k]=weights[k]*m.frames[local.ids[k]];group.center_weights[k]=weights[k];
          }
          while(group.count && weights[group.count-1]==0)--group.count;
          int start=0;while(start<group.count && weights[start]==0)++start;
          group.first+=6*start;group.count-=start;
          for(int k=0;k<group.count;++k){group.maps[k]=group.maps[k+start];group.center_weights[k]=group.center_weights[k+start];}
          groups.push_back(std::move(group));
        }
        groups[group_id].supports.push_back({plane.normal,plane.bound});
      }
    }
    for(auto& group:groups) {
      {
        auto key=[](const Support& p){return std::make_tuple(p.normal[0],p.normal[1],p.normal[2]);};
        auto& supports=group.supports;
        std::sort(supports.begin(),supports.end(),[&](const Support& a,const Support& b){return key(a)<key(b);});
        size_t count=0;
        for(const auto& p:supports) {
          if(count&&key(p)==key(supports[count-1]))supports[count-1].bound=std::min(supports[count-1].bound,p.bound);
          else supports[count++]=p;
        }
        data->merged+=supports.size()-count;supports.resize(count);
      }
      for(size_t j=0;j<group.supports.size();++j) {
        auto& p=group.supports[j];p.original_id=int(j);
        for(int axis=0;axis<3;++axis)
          if(p.normal[axis]!=0&&p.normal[(axis+1)%3]==0&&p.normal[(axis+2)%3]==0)p.axis=axis;
      }
    }
    StageTimer pruning(settings.profile);
    for(auto& group:groups)
      pruneProjectedSupports(group,m.x,o.max_offset,o.max_radius);
    pruning_ms=pruning.elapsed();
    for(auto& group:groups) {
      const auto last=std::stable_partition(group.supports.begin(),group.supports.end(),
                                            [](const Support& p){return !p.redundant;});
      group.retained=last-group.supports.begin();
      data->count+=group.supports.size();data->retained+=group.retained;
      for(size_t i=0;i<group.retained;++i)data->axes+=group.supports[i].axis>=0;
    }
    geometry=std::move(data);
  }
  // Local convex progress cones preserve ordering while all centers move.
  // They use fixed normals and a degree-two derivative hull. The nonlinear
  // full-disk certificate remains the acceptance test for every step.
  if(progress.spans.empty())return;
  const auto initial=m.geometry();
  for(int span:progress.spans) {
    const int samples=3;
    for(int sample=0;sample<samples;++sample) {
    const double t=.5*sample;
    const double u=m.basis.breaks[span]+t*(m.basis.breaks[span+1]-m.basis.breaks[span]);
    const auto section=initial.section(u);
    const Vec3 normal=section.shape.col(0).cross(section.shape.col(1)).normalized();
    const auto& e=m.basis.spans[span].extraction;
    Curve<2,Eigen::Vector4d> derivative;
    for(int j=0;j<3;++j)derivative[j]=3*(e.row(j+1)-e.row(j));
    auto emit=[&](const auto& part) {for(int control=0;control<3;++control) {
      const auto& weight=part[control];
      Cut c;c.first=6*span;c.center_sign=-1;c.bound=-1e-5*length*(m.basis.breaks[span+1]-m.basis.breaks[span]);
      for(int k=0;k<4;++k) {
        const Eigen::Vector2d v=weight[k]*m.frames[span+k].transpose()*normal;
        c.bound+=weight[k]*m.reference[span+k].dot(normal);
        c.normals[k]=v;c.center_normals[k]=weight[k]*normal;
      }
      if(!c.compact())continue;
      const Vec3 value=c.apply(m.x);
      if(c.bound-value[0]>value.tail<2>().norm()+1e-9)cuts.push_back(c);
    }};
    emit(derivative);
    }
  }
}
template<bool WithValue>
inline bool exactSupportJets(const CubicProblem::Group& group,const V9& y,double mu,
                             V9& gradient,Eigen::Matrix<double,9,9>& hessian,double& cost) {
  const size_t count=group.retained;
  for(size_t id=0;id<count;++id) {
    const auto& p=group.supports[id];
    const Vec3 v=p.project(y);
    const double slack=v[0]*v[0]-v.tail<2>().squaredNorm();
    if(v[0]<=0||slack<=0)return false;
    if constexpr(WithValue)cost-=mu*std::log(slack);
    const double f=2*mu/slack;
    p.addGradient(f*v,gradient);
    Eigen::Matrix3d hc=(2*f/slack)*v*v.transpose();
    hc(0,0)-=f;hc(1,1)+=f;hc(2,2)+=f;
    if(p.axis>=0) {
      const int axis=p.axis;const double n2=p.normal[axis]*p.normal[axis];
      for(int a=0;a<3;++a)for(int b=0;b<=a;++b)hessian(3*a+axis,3*b+axis)+=hc(a,b)*n2;
    } else {
      const Eigen::Matrix3d nn=p.normal*p.normal.transpose();
      for(int a=0;a<3;++a)for(int b=0;b<=a;++b)hessian.block<3,3>(3*a,3*b).noalias()+=hc(a,b)*nn;
    }
  }
  for(int a=0;a<3;++a)for(int b=0;b<a;++b)hessian.block<3,3>(3*b,3*a)=hessian.block<3,3>(3*a,3*b).transpose();
  return true;
}
inline double CubicProblem::objective(const Eigen::VectorXd& x,double mu,Eigen::VectorXd* g,
                               BandedSPD* h,bool evaluate_cost)const {
  const bool derivatives=g,curvature=h;
  const auto qx=quadratic.multiply(x);
  double cost=evaluate_cost?.5*x.dot(qx)+linear.dot(x):0;
  if(derivatives)*g=qx+linear;
  if(curvature){
    // Preserve allocated factor/scale buffers across Newton steps.
    h->n=quadratic.n;h->data=quadratic.data;
  }
  auto logdet=[&](int i,const Eigen::Matrix2d& p,double weight,double sign) {
    const double det=p.determinant();if(p(0,0)<=0 || det<=0)return false;
    if(evaluate_cost)cost-=weight*std::log(det);
    if(derivatives) {
      const double a=p(1,1)/det,b=-p(0,1)/det,c=p(0,0)/det;
      g->segment<3>(6*i+3)-=weight*sign*Vec3(a,2*b,c);
      if(curvature) {
        Eigen::Matrix3d v;v<<a*a,2*a*b,b*b,2*a*b,2*(a*c+b*b),2*b*c,b*b,2*b*c,c*c;
        for(int j=0;j<3;++j)for(int k=0;k<=j;++k)h->add(6*i+3+j,6*i+3+k,weight*v(j,k));
      }
    }
    return true;
  };
  for(int i=0;i<model.basis.count;++i) {
    const auto s=matrix(x,i);
    if(!logdet(i,s,settings.area_weight*model.basis.weight(i),1) ||
       !logdet(i,s-options.min_radius*Eigen::Matrix2d::Identity(),mu,1) ||
       !logdet(i,options.max_radius*Eigen::Matrix2d::Identity()-s,mu,-1))return INFINITY;
    const Vec3 d=x.segment<3>(6*i);const double gap=options.max_offset*options.max_offset-d.squaredNorm();
    if(gap<=0)return INFINITY;
    if(evaluate_cost)cost-=mu*std::log(gap);
    if(derivatives) {
      g->segment<3>(6*i)+=2*mu/gap*d;
      if(curvature) {
        const Eigen::Matrix3d v=2*mu/gap*Eigen::Matrix3d::Identity()+4*mu/(gap*gap)*d*d.transpose();
        for(int j=0;j<3;++j)for(int k=0;k<=j;++k)h->add(6*i+j,6*i+k,v(j,k));
      }
    }
  }
  for(const auto& group:geometry->groups) {
    const V9 y=group.geometry(x);
    V9 gy;Eigen::Matrix<double,9,9> hy;
    if(derivatives)gy.setZero();
    if(curvature) {
      hy.setZero();
      const bool feasible=evaluate_cost?exactSupportJets<true>(group,y,mu,gy,hy,cost):
                                        exactSupportJets<false>(group,y,mu,gy,hy,cost);
      if(!feasible)return INFINITY;
      group.accumulate(gy,hy,*g,*h);continue;
    }
    for(size_t id=0;id<group.retained;++id) {
      const auto& support=group.supports[id];const Vec3 v=support.project(y);
      const double slack=v[0]*v[0]-v.tail<2>().squaredNorm();
      if(v[0]<=0||slack<=0)return INFINITY;
      if(evaluate_cost)cost-=mu*std::log(slack);
      if(derivatives)support.addGradient((2*mu/slack)*v,gy);
    }
    if(derivatives)group.accumulateGradient(gy,*g);
  }
  // Keep longitudinal spacing away from collapse even after the feasibility
  // barrier is reduced. This reuses the same local Lorentz-cone derivatives.
  const double cut_mu=mu+settings.progress_weight/model.basis.count;
  for(const auto& c:cuts) {
    Vec3 v=c.apply(x);v[0]=c.bound-v[0];
    const double slack=v[0]*v[0]-v.tail<2>().squaredNorm();
    if(v[0]<=0||slack<=0)return INFINITY;
    if(evaluate_cost)cost-=cut_mu*std::log(slack);
    if(curvature)c.accumulate(v,cut_mu,slack,*g,*h);
    else if(derivatives) {
      const double f=2*cut_mu/slack;
      for(int k=0;k<c.count;++k) {
        const auto& n=c.normals[k];const int offset=c.first+6*k;
        g->segment<3>(offset)+=(f*c.center_sign*v[0])*c.center_normals[k];
        g->segment<3>(offset+3)+=f*Vec3(v[1]*n[0],v[1]*n[1]+v[2]*n[0],v[2]*n[1]);
      }
    }
  }
  return cost;
}
} // namespace SweepCorridor::internal::cubic
