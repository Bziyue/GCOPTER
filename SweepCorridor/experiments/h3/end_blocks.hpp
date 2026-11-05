// SPDX-License-Identifier: MIT
#pragma once
#include "joint_jets.hpp"

namespace SweepCorridor::experiments::h3 {
// With M fixed, V = pi integral (M0 x M1).c' is affine in the center
// values and shared center derivatives. A local block therefore needs only
// six variables per node and reuses the H3 block-tridiagonal factorization.
struct CenterBlock {
  Model& model;
  const GeometryOptions& options;
  int first,last,span_begin,span_end,size;
  Geometry geometry;
  Eigen::VectorXd origin,linear,volume_gradient;
  BlockTridiagonal quadratic;
  double volume;
  struct NodeWeight {int node;double value,jet;};
  struct Support {Vec3 normal;double minus,plus;};
  struct Group {std::vector<NodeWeight> weights;std::vector<Support> rows;};
  struct Jacobian {
    c3::Curve<8,Vec3> base;
    Eigen::Matrix<double,9,Eigen::Dynamic> change;
  };
  std::vector<Group> groups;
  std::vector<Jacobian> jacobians;

  double scale(int node)const{return 3/model.basis.weight(node);}
  std::vector<NodeWeight> weights(int span,const Eigen::Vector4d& b)const{
    const double h=model.basis.breaks[span+1]-model.basis.breaks[span];
    std::vector<NodeWeight> result;
    if(span>=first&&span<last)result.push_back({span-first,b[0]+b[1],h*scale(span)*b[1]/3});
    if(span+1>=first&&span+1<last)result.push_back({span+1-first,b[2]+b[3],-h*scale(span+1)*b[2]/3});
    return result;
  }
  static Vec3 direction(const std::vector<NodeWeight>& weights,const Eigen::VectorXd& d){
    Vec3 value=Vec3::Zero();for(const auto& w:weights)value+=w.value*d.segment<3>(6*w.node)+w.jet*d.segment<3>(6*w.node+3);return value;
  }
  static void addGradient(const std::vector<NodeWeight>& weights,const Vec3& g,Eigen::VectorXd& result){
    for(const auto& w:weights){result.segment<3>(6*w.node)+=w.value*g;result.segment<3>(6*w.node+3)+=w.jet*g;}
  }
  static void addHessian(const std::vector<NodeWeight>& weights,const Eigen::Matrix3d& h,BlockTridiagonal& result){
    for(size_t i=0;i<weights.size();++i)for(size_t j=0;j<=i;++j){const auto& a=weights[i];const auto& b=weights[j];Block v;
      v.topLeftCorner<3,3>()=a.value*b.value*h;v.topRightCorner<3,3>()=a.value*b.jet*h;
      v.bottomLeftCorner<3,3>()=a.jet*b.value*h;v.bottomRightCorner<3,3>()=a.jet*b.jet*h;
      result.addBlock(a.node,b.node,v);}
  }
  CenterBlock(Model& m,int begin,int end,const GeometryOptions& o,double length,double energy_weight,
              const Problem::Constraints& constraints)
      :model(m),options(o),first(begin),last(end),span_begin(std::max(0,begin-1)),span_end(std::min(end,m.basis.count-1)),
       size(6*(end-begin)),geometry(m.geometry()),origin(size),linear(Eigen::VectorXd::Zero(size)),
       volume_gradient(Eigen::VectorXd::Zero(size)),quadratic(size),volume(sweptVolume(m,m.x)){
    for(int i=first;i<last;++i){origin.segment<3>(6*(i-first))=m.x.segment<3>(6*i);
      origin.segment<3>(6*(i-first)+3)=m.jets[i].head<3>()/scale(i);}
    constexpr double t[5]={.046910077030668,.230765344947158,.5,.769234655052842,.953089922969332};
    constexpr double w[5]={.118463442528095,.239314335249683,.284444444444444,.239314335249683,.118463442528095};
    for(int span=span_begin;span<span_end;++span){const auto p=geometry.curve(span);const double h=m.basis.breaks[span+1]-m.basis.breaks[span];
      for(int q=0;q<5;++q){const double u=t[q],r=1-u;Eigen::Vector4d db;db<<-3*r*r,3*r*(1-3*u),3*u*(2-3*u),3*u*u;db/=h;
        const auto section=c3::evaluate<3>(p,u,h);const Vec3 normal=section.shape.col(0).cross(section.shape.col(1));
        addGradient(weights(span,db),std::acos(-1.)*h*w[q]*normal,volume_gradient);}
      for(double u:{.5-.5/std::sqrt(3.),.5+.5/std::sqrt(3.)}){
        Eigen::Vector4d ddb;ddb<<6*(1-u),18*u-12,6-18*u,6*u;ddb/=h*h;
        const auto a=weights(span,ddb);const double coefficient=energy_weight*OptimizerSettings::center_energy*h/(length*length*length);
        addGradient(a,coefficient*c3::evaluate<3>(p,u,h).second,linear);
        addHessian(a,coefficient*Eigen::Matrix3d::Identity(),quadratic);
      }
      Jacobian jc;jc.base=c3::jacobianCoefficients<3>(p);jc.change.setZero(9,size);
      c3::Curve<6,Vec3> normals;for(auto& n:normals)n.setZero();
      for(int a=0;a<4;++a)for(int b=0;b<4;++b)
        normals[a+b]+=c3::binomial(3,a)*c3::binomial(3,b)/c3::binomial(6,a+b)*shape(p[a]).col(0).cross(shape(p[b]).col(1));
      for(int j=0;j<3;++j){const auto mapping=weights(span,Eigen::Vector4d::Unit(j+1)-Eigen::Vector4d::Unit(j));
        for(int k=0;k<7;++k){const Vec3 n=3*c3::binomial(2,j)*c3::binomial(6,k)/c3::binomial(8,j+k)*normals[k];
          for(const auto& a:mapping){jc.change.block<1,3>(j+k,6*a.node)+=a.value*n.transpose();jc.change.block<1,3>(j+k,6*a.node+3)+=a.jet*n.transpose();}}}
      jacobians.push_back(std::move(jc));
    }
    for(int i=first;i<last;++i){const double coefficient=2*OptimizerSettings::offset_weight*m.basis.weight(i);
      linear.segment<3>(6*(i-first))+=coefficient*m.x.segment<3>(6*i);
      quadratic.diagonal[i-first].topLeftCorner<3,3>().diagonal().array()+=coefficient;}
    for(const auto& source:constraints.groups){if(source.span<span_begin||source.span>=span_end)continue;
      Group g;g.weights=weights(source.span,source.bernstein);if(g.weights.empty())continue;
      V9 y=V9::Zero();for(int j=0;j<4;++j)y+=source.bernstein[j]*geometry.controls[4*source.span+j];
      for(const auto& p:source.supports){const double r=p.bound-p.normal.dot(y.head<3>()),s=(shape(y).transpose()*p.normal).norm();
        g.rows.push_back({p.normal,r-s,r+s});}
      groups.push_back(std::move(g));
    }
  }
  void assign(const Eigen::VectorXd& d){
    for(int i=first;i<last;++i){model.x.segment<3>(6*i)=origin.segment<3>(6*(i-first))+d.segment<3>(6*(i-first));
      model.jets[i].head<3>()=scale(i)*(origin.segment<3>(6*(i-first)+3)+d.segment<3>(6*(i-first)+3));}
  }
  double objective(const Eigen::VectorXd& d,double mu,Eigen::VectorXd* gradient=nullptr,BlockTridiagonal* hessian=nullptr)const{
    const double total=volume+volume_gradient.dot(d);if(total<=0)return INFINITY;
    const Eigen::VectorXd qd=quadratic.multiply(d);double cost=linear.dot(d)+.5*d.dot(qd)-std::log(total/volume);
    if(gradient)*gradient=linear+qd-volume_gradient/total;
    // The affine-volume rank-one Hessian is applied by Sherman-Morrison.
    if(hessian)hessian->assign(quadratic);
    for(int i=0;i<size/6;++i){const Vec3 center=origin.segment<3>(6*i)+d.segment<3>(6*i);
      const double gap=options.max_offset*options.max_offset-center.squaredNorm();if(gap<=0)return INFINITY;
      cost-=mu*std::log(gap);if(gradient)gradient->segment<3>(6*i)+=2*mu/gap*center;
      if(hessian){hessian->diagonal[i].topLeftCorner<3,3>().diagonal().array()+=2*mu/gap;
        hessian->diagonal[i].topLeftCorner<3,3>().noalias()+=4*mu/(gap*gap)*center*center.transpose();}}
    for(const auto& group:groups){const Vec3 delta=direction(group.weights,d);Vec3 g=Vec3::Zero();Eigen::Matrix3d h=Eigen::Matrix3d::Zero();
      for(const auto& p:group.rows){const double shift=p.normal.dot(delta),a=p.minus-shift,b=p.plus-shift;
        if(a<=0||b<=0)return INFINITY;cost-=mu*(std::log(a)+std::log(b));
        if(gradient)g+=mu*(1/a+1/b)*p.normal;
        if(hessian)h.noalias()+=mu*(1/(a*a)+1/(b*b))*p.normal*p.normal.transpose();}
      if(gradient)addGradient(group.weights,g,*gradient);
      if(hessian)addHessian(group.weights,h,*hessian);
    }return cost;
  }
  double anchorCost(std::array<Anchor,2>& anchors,double mu,Eigen::VectorXd* gradient=nullptr,BlockTridiagonal* hessian=nullptr)const{
    double cost=0;
    for(int end=0;end<2;++end){auto& a=anchors[end];if(!locateAnchor(model,model.x,model.reference[end?model.basis.count-1:0],a,false))return INFINITY;
      const double gap=1-a.z.squaredNorm();cost-=mu*std::log(gap);if(!gradient)continue;
      const double h=model.basis.breaks[a.span+1]-model.basis.breaks[a.span],t=(a.u-model.basis.breaks[a.span])/h,r=1-t;
      Eigen::Vector4d b;b<<r*r*r,3*r*r*t,3*r*t*t,t*t*t;const auto mapping=weights(a.span,b);
      Eigen::Matrix3d inverse;inverse.col(0)=a.section.first+a.section.shape_first*a.z;inverse.rightCols<2>()=a.section.shape;
      const Eigen::Matrix<double,2,3> dz=-inverse.inverse().bottomRows<2>();
      const Vec3 g=2*mu/gap*dz.transpose()*a.z;addGradient(mapping,g,*gradient);
      if(hessian){const Eigen::Matrix3d curvature=2*mu/gap*dz.transpose()*dz+4*mu/(gap*gap)*(dz.transpose()*a.z)*(a.z.transpose()*dz);
        addHessian(mapping,curvature,*hessian);}
    }return anchors[0].u<anchors[1].u?cost:INFINITY;
  }
  c3::CubicProblem::LineModel prepareLine(const Eigen::VectorXd& d,const Eigen::VectorXd& step,double mu)const{
    c3::CubicProblem::LineModel line;line.linear=step.dot(linear+quadratic.multiply(d));line.quadratic=.5*step.dot(quadratic.multiply(step));
    double limit=INFINITY;
    for(const auto& group:groups){const Vec3 delta=direction(group.weights,d),change=direction(group.weights,step);
      for(const auto& p:group.rows){const double shift=p.normal.dot(delta),s=p.normal.dot(change);
        for(double gap:{p.minus-shift,p.plus-shift}){line.terms.push_back({0,-s/gap,mu});if(s>0)limit=std::min(limit,gap/s);}}}
    for(int i=0;i<size/6;++i){const Vec3 center=origin.segment<3>(6*i)+d.segment<3>(6*i),v=step.segment<3>(6*i);
      const double a=-v.squaredNorm(),b=-2*center.dot(v),c=options.max_offset*options.max_offset-center.squaredNorm();
      line.terms.push_back({a/c,b/c,mu});if(a+b+c<=0)limit=std::min(limit,c3::positiveRoot(a,b,c));}
    line.step=std::min(1.,.99*limit);return line;
  }
  bool regular(const Eigen::VectorXd& d)const{
    for(const auto& jac:jacobians){auto cone=jac.base;const Eigen::Matrix<double,9,1> change=jac.change*d;double scale=0;
      for(int i=0;i<9;++i){cone[i][0]+=change[i];scale=std::max(scale,cone[i].norm());}
      c3::Certificate certificate;if(scale<=1e-18||!c3::coneCertificate<8>(cone,std::max(1e-18,1e-9*scale),scale,certificate))return false;
    }return true;
  }
  template<class Solve> void constrainCaps(const Eigen::VectorXd& d,Eigen::VectorXd& step,Solve&& solve)const{
    std::vector<std::pair<int,Vec3>> caps;if(first==0)caps.push_back({0,inwardCapNormal(model,0)});
    if(last==model.basis.count)caps.push_back({size-6,inwardCapNormal(model,1)});
    const int n=caps.size();Eigen::Vector2d rhs=Eigen::Vector2d::Zero();int mask=0;
    for(int i=0;i<n;++i){rhs[i]=caps[i].second.dot(origin.segment<3>(caps[i].first)+d.segment<3>(caps[i].first)+step.segment<3>(caps[i].first));if(rhs[i]<0)mask|=1<<i;}
    if(!mask)return;
    const Eigen::VectorXd unconstrained=step;std::array<Eigen::VectorXd,2> inverse;
    for(int i=0;i<n;++i){Eigen::VectorXd g=Eigen::VectorXd::Zero(size);g.segment<3>(caps[i].first)=caps[i].second;solve(g,inverse[i]);}
    auto attempt=[&](int active){Eigen::Matrix2d a=Eigen::Matrix2d::Identity();Eigen::Vector2d b=Eigen::Vector2d::Zero();
      for(int i=0;i<n;++i)if(active&(1<<i)){b[i]=rhs[i];for(int j=0;j<n;++j)a(i,j)=(active&(1<<j))?caps[i].second.dot(inverse[j].segment<3>(caps[i].first)):0;}
      const Eigen::Vector2d lambda=a.ldlt().solve(b);if(lambda.maxCoeff()>1e-10)return false;
      Eigen::VectorXd candidate=unconstrained;for(int i=0;i<n;++i)if(active&(1<<i))candidate-=lambda[i]*inverse[i];
      for(const auto& cap:caps)if(cap.second.dot(origin.segment<3>(cap.first)+d.segment<3>(cap.first)+candidate.segment<3>(cap.first)) < -1e-9)return false;
      step=std::move(candidate);return true;};
    if(attempt(mask))return;for(int active=1;active<(1<<n);++active)if(active!=mask&&attempt(active))return;
    step.setZero();
  }
};

inline void optimizeEnds(Model& model,const GeometryOptions& options,double length,
    const OptimizerSettings& settings,const Problem::Constraints& constraints,JetMetrics& metrics,const Deadline& deadline){
  StageTimer timer(settings.profile);const int n=model.basis.count,k=std::min(settings.end_nodes,n);
  std::vector<std::pair<int,int>> blocks;
  if(n<=2*k)blocks.push_back({0,n});else{blocks.push_back({0,k});blocks.push_back({n-k,n});}
  std::array<Anchor,2> anchors;for(int i=0;i<2;++i)anchors[i].u=model.anchor_parameters[i];
  for(const auto& range:blocks){deadline.check();CenterBlock patch(model,range.first,range.second,options,length,settings.end_energy,constraints);
    Eigen::VectorXd d=Eigen::VectorXd::Zero(patch.size),gradient,step,inverse_v;BlockTridiagonal hessian;
    const double mu=1e-5/n,anchor_mu=1e-5;
    for(int iteration=0;iteration<settings.end_steps;++iteration){deadline.check();
      const double cost=patch.objective(d,mu,&gradient,&hessian),anchor=patch.anchorCost(anchors,anchor_mu,&gradient,&hessian);
      if(!std::isfinite(cost)||!std::isfinite(anchor))break;
      if(!hessian.factor(1e-8))break;
      const double total=patch.volume+patch.volume_gradient.dot(d);const Eigen::VectorXd v=patch.volume_gradient/total;
      hessian.solveInto(v,inverse_v);const double denominator=1+v.dot(inverse_v);
      auto solve=[&](const Eigen::VectorXd& rhs,Eigen::VectorXd& answer){hessian.solveInto(rhs,answer);answer-=inverse_v*(v.dot(answer)/denominator);};
      solve(-gradient,step);patch.constrainCaps(d,step,solve);const double descent=gradient.dot(step);if(descent>=-1e-10)break;
      const auto line=patch.prepareLine(d,step,mu);const double dv=patch.volume_gradient.dot(step);bool accepted=false;
      for(double alpha=line.step;alpha>=1e-10;alpha*=.5){++metrics.end_trials;const Eigen::VectorXd next=d+alpha*step;
        if(alpha*dv<=-total)continue;
        patch.assign(next);auto trial=anchors;const double anchor_next=patch.anchorCost(trial,anchor_mu);
        if(line.delta(alpha)-std::log1p(alpha*dv/total)+anchor_next-anchor>1e-4*alpha*descent)continue;
        if(!patch.regular(next))continue;
        anchors=trial;d=next;accepted=true;++metrics.end_steps;break;
      }
      patch.assign(d);if(!accepted)break;
    }
    metrics.end_volume_gain+=patch.volume_gradient.dot(d);model.cacheQuadrature(patch.span_begin,patch.span_end);
  }
  for(int i=0;i<2;++i)model.anchor_parameters[i]=anchors[i].u;
  metrics.end_ms+=timer.elapsed();
}
}
