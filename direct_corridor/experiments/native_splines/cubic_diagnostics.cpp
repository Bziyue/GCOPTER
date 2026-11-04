// SPDX-License-Identifier: MIT
#include "cubic_optimizer.hpp"
namespace gcopter::direct::native {
Eigen::Matrix<double,3,20> CubicProblem::Cut::dense()const {
  Eigen::Matrix<double,3,20> a=Eigen::Matrix<double,3,20>::Zero();
  for(int k=0;k<count;++k) {
    const auto& n=normals[k];
    a.block<1,2>(0,5*k)=center_sign*n.transpose();
    a(1,5*k+2)=n[0];a(1,5*k+3)=n[1];a(2,5*k+3)=n[0];a(2,5*k+4)=n[1];
  }
  return a;
}
static Eigen::Matrix2d matrix(const Eigen::VectorXd& x,int i) {
  Eigen::Matrix2d s;s<<x[5*i+2],x[5*i+3],x[5*i+3],x[5*i+4];return s;
}
double CubicProblem::referenceObjective(const Eigen::VectorXd& x,double mu,Eigen::VectorXd* g,BandedSPD* h)const {
  constexpr bool evaluate_cost=true;
  const bool derivatives=g,curvature=h;
  const auto qx=quadratic.multiply(x);
  double cost=evaluate_cost?.5*x.dot(qx)+linear.dot(x):0;
  if(derivatives)*g=qx+linear;
  if(curvature){
    // Preserve allocated factor/scale buffers across Newton steps.
    h->n=quadratic.n;h->width=quadratic.width;h->data=quadratic.data;
  }
  auto logdet=[&](int i,const Eigen::Matrix2d& p,double weight,double sign) {
    const double det=p.determinant();if(p(0,0)<=0 || det<=0)return false;
    if(evaluate_cost)cost-=weight*std::log(det);
    if(derivatives) {
      const double a=p(1,1)/det,b=-p(0,1)/det,c=p(0,0)/det;
      g->segment<3>(5*i+2)-=weight*sign*Vec3(a,2*b,c);
      if(curvature) {
        Eigen::Matrix3d v;v<<a*a,2*a*b,b*b,2*a*b,2*(a*c+b*b),2*b*c,b*b,2*b*c,c*c;
        for(int j=0;j<3;++j)for(int k=0;k<=j;++k)h->add(5*i+2+j,5*i+2+k,weight*v(j,k));
      }
    }
    return true;
  };
  for(int i=0;i<model.basis.count;++i) {
    const auto s=matrix(x,i);
    if(!logdet(i,s,model.basis.weight(i),1) ||
       !logdet(i,s-options.min_radius*Eigen::Matrix2d::Identity(),mu,1) ||
       !logdet(i,options.max_radius*Eigen::Matrix2d::Identity()-s,mu,-1))return INFINITY;
    const Eigen::Vector2d d=x.segment<2>(5*i);const double gap=options.max_offset*options.max_offset-d.squaredNorm();
    if(gap<=0)return INFINITY;
    if(evaluate_cost)cost-=mu*std::log(gap);
    if(derivatives) {
      g->segment<2>(5*i)+=2*mu/gap*d;
      if(curvature) {
        const Eigen::Matrix2d v=2*mu/gap*Eigen::Matrix2d::Identity()+4*mu/(gap*gap)*d*d.transpose();
        for(int j=0;j<2;++j)for(int k=0;k<=j;++k)h->add(5*i+j,5*i+k,v(j,k));
      }
    }
  }
  auto accumulate=[&](const Cut& c) {
    Vec3 v=c.apply(x);v[0]=c.bound-v[0];
    const double slack=v[0]*v[0]-v.tail<2>().squaredNorm();
    if(v[0]<=0||slack<=0)return false;
    cost-=mu*std::log(slack);
    const auto map=c.dense();const int n=5*c.count;
    const Eigen::Matrix<double,20,1> ds=-2*map.transpose()*v;
    if(derivatives)g->segment(c.first,n)-=mu/slack*ds.head(n);
    if(curvature) {
      const double first=mu/slack,second=first/slack;
      Eigen::Matrix<double,20,20> local=second*ds*ds.transpose();
      local.noalias()+=2*first*map.bottomRows<2>().transpose()*map.bottomRows<2>();
      local.noalias()-=2*first*map.row(0).transpose()*map.row(0);
      for(int j=0;j<n;++j)for(int k=0;k<=j;++k)h->add(c.first+j,c.first+k,local(j,k));
    }
    return true;
  };
  for(const auto& group:geometry->groups)for(size_t id=0;id<group.retained;++id) {
    const auto& p=group.supports[id];Cut c;
    c.first=group.first;c.count=group.count;c.bound=p.bound-p.normal.dot(group.reference.head<3>());
    for(int k=0;k<c.count;++k)c.normals[k]=group.maps[k].transpose()*p.normal;
    c.compact();if(!accumulate(c))return INFINITY;
  }
  for(const auto& c:cuts)if(!accumulate(c))return INFINITY;
  return cost;
}
void CubicProblem::checkDerivatives(const Eigen::VectorXd& x,OptimizerMetrics& metrics)const {
  const double mu=.003/model.basis.count;
  Eigen::VectorXd g;BandedSPD h;const double f=objective(x,mu,&g,&h);(void)f;
  Eigen::VectorXd reference_gradient;BandedSPD reference_hessian;
   {
    referenceObjective(x,mu,&reference_gradient,&reference_hessian);
    metrics.gradient_reference_error=std::max(metrics.gradient_reference_error,
        (g-reference_gradient).norm()/std::max({1.,g.norm(),reference_gradient.norm()}));
  }
  for(int trial=0;trial<8;++trial) {
    Eigen::VectorXd v(x.size());for(int j=0;j<v.size();++j)v[j]=std::sin((j+1)*(trial+1)*.381);
    v.normalize();
    // Later inflation passes start close to active cones. Scale the diagnostic
    // stencil to their two-sided margin and cancel its O(step^2) error.
    // This checker only runs on untimed benchmark warmups / explicit tests.
    const double step=std::min(1e-6,.01*std::min(feasibleStep(x,v),feasibleStep(x,-v)));
    metrics.derivative_min_step=std::min(metrics.derivative_min_step,step);
    const double wide=(objective(x+step*v,mu)-objective(x-step*v,mu))/(2*step);
    const double narrow=(objective(x+.5*step*v,mu)-objective(x-.5*step*v,mu))/step;
    const double fd=(4*narrow-wide)/3;
    metrics.gradient_error=std::max(metrics.gradient_error,std::abs(fd-g.dot(v))/std::max({1.,std::abs(fd),std::abs(g.dot(v))}));
    Eigen::VectorXd gp,gm,hp,hm;BandedSPD scratch;
    objective(x+step*v,mu,&gp,&scratch);objective(x-step*v,mu,&gm,&scratch);
    objective(x+.5*step*v,mu,&hp,&scratch);objective(x-.5*step*v,mu,&hm,&scratch);
    const Eigen::VectorXd hv=h.multiply(v),num=(4*(hp-hm)/step-(gp-gm)/(2*step))/3;
    metrics.hessian_error=std::max(metrics.hessian_error,(num-hv).norm()/std::max({1.,num.norm(),hv.norm()}));
     {
      const auto expected=reference_hessian.multiply(v);
      metrics.hessian_reference_error=std::max(metrics.hessian_reference_error,
          (expected-hv).norm()/std::max({1.,expected.norm(),hv.norm()}));
    }
  }
}
}
