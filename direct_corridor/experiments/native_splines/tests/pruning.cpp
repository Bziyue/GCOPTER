// SPDX-License-Identifier: MIT
#include "cubic_optimizer.hpp"
#include "support_barrier.hpp"
#include "support_reach.hpp"
#include "cubic_certificates.hpp"
#include <Eigen/Eigenvalues>
#include <cassert>
#include <iostream>
#include <random>
using namespace gcopter::direct;
using namespace gcopter::direct::native;

static void checkProofs(const CubicProblem::Group& group,double offset,double lower,double radius) {
  auto find=[&](int id)->const CubicProblem::Support& {
    const auto p=std::find_if(group.supports.begin(),group.supports.end(),
                            [&](const auto& s){return s.original_id==id;});
    assert(p!=group.supports.end());return *p;
  };
  auto dual=[&](int id) {
    if(id<0)return Vec3::Zero().eval();
    const auto& p=find(id);return Vec3(p.normal/(p.bound-p.normal.dot(group.proof_center)));
  };
  for(const auto& proof:group.proofs) {
    assert(find(proof.removed).redundant);
    if(proof.left>=0)assert(!find(proof.left).redundant);
    if(proof.right>=0)assert(!find(proof.right).redundant);
    assert(proof.left_weight>=0&&proof.right_weight>=0&&proof.left_weight+proof.right_weight<=1);
    const Vec3 residual=dual(proof.removed)-proof.left_weight*dual(proof.left)-proof.right_weight*dual(proof.right);
    long double upper=proof.left_weight+proof.right_weight;
    const Vec3 origin=proof.exact?group.proof_fixed_center:Vec3(group.reference.head<3>());
    upper+=residual.cast<long double>().dot((origin-group.proof_center).cast<long double>());
    Eigen::Matrix<long double,2,1> sum=Eigen::Matrix<long double,2,1>::Zero();
    long double norms=0;
    for(int k=0;k<group.count;++k) {
      const Eigen::Matrix<long double,2,1> v=group.maps[k].cast<long double>().transpose()*residual.cast<long double>();
      if(proof.exact){upper+=group.proof_offsets[k]*v.norm();norms+=v.norm();sum+=v;}
      else upper+=(offset+radius)*v.norm();
    }
    if(proof.exact)upper+=.5L*(radius-lower)*norms+.5L*(radius+lower)*sum.norm();
    assert(upper<1-1e-11);
    assert(std::abs(upper-proof.upper)<1e-11);
  }
}

int main() {
  // Construct maximizing matrices independently using a symmetric eigensolve.
  // Include exact cancellation, zero projections, equal spectral bounds and
  // widely different scales, rather than sampling only generic positive norms.
  std::mt19937_64 random(20261104);std::normal_distribution<double> normal;
  double worst_reach_error=0;
  for(int trial=0;trial<100000;++trial) {
    const int count=1+trial%4;const double scale=std::pow(10.,trial%13-6);
    const double lower=trial%5==0?0:.1,upper=trial%7==0?lower:1.7;
    std::array<Eigen::Vector2d,4> vectors;
    Eigen::Vector2d sum=Eigen::Vector2d::Zero();double norms=0;
    for(int i=0;i<count;++i) {
      vectors[i]=scale*Eigen::Vector2d(normal(random),normal(random));
      if(trial%11==0)vectors[i].setZero();
      if(trial%13==0&&i==1)vectors[i]=-vectors[0];
      sum+=vectors[i];norms+=vectors[i].norm();
    }
    const Eigen::Vector2d z=sum.norm()>0?Eigen::Vector2d(sum.normalized()):Eigen::Vector2d::UnitX();
    double attained=0;
    for(int i=0;i<count;++i) {
      const Eigen::Matrix2d symmetric=.5*(z*vectors[i].transpose()+vectors[i]*z.transpose());
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen(symmetric);
      const Eigen::Vector2d values((eigen.eigenvalues()[0]>0?upper:lower),(eigen.eigenvalues()[1]>0?upper:lower));
      const Eigen::Matrix2d matrix=eigen.eigenvectors()*values.asDiagonal()*eigen.eigenvectors().transpose();
      attained+=vectors[i].dot(matrix*z);
    }
    const double reach=exactShapeReach(norms,sum,lower,upper);
    worst_reach_error=std::max(worst_reach_error,std::abs(reach-attained)/(1+std::abs(reach)));
    assert(std::abs(reach-attained)<1e-11*(1+std::abs(reach)));
    assert(reach<=upper*norms+1e-11*(1+upper*norms));
  }
  std::cout<<"exact reach: 100000 extrema, relative error "<<worst_reach_error<<'\n';
  // New deletions must also come from shared-z cancellation, independently
  // of fixed endpoint offsets. Use two/four interior controls and nonuniform
  // weights, with transverse frame tilts cancelling in the residual direction.
  for(int count:{2,4}) {
    CubicProblem::Group group;group.first=5;group.count=count;group.reference.setZero();
    Eigen::VectorXd x=Eigen::VectorXd::Zero(5*(count+2));
    for(int k=0;k<count;++k) {
      const double weight=count==2?.5:(k<2?.1:.4),sign=k%2?-1.:1.;
      group.maps[k].col(0)=weight*Vec3(sign*std::sin(.6),std::cos(.6),0);
      group.maps[k].col(1)=weight*Vec3::UnitZ();
      x[5*(k+1)+2]=x[5*(k+1)+4]=.2;
    }
    auto add=[&](Vec3 n){group.supports.push_back({n,1.,true,false,-1,int(group.supports.size())});};
    for(int axis:{1,2})for(double sign:{-1.,1.})add(sign*Vec3::Unit(axis));
    add(Vec3(1/std::sin(.6),.1,0));
    auto triangle=group,exact=group,hybrid=group;
    pruneProjectedSupports(triangle,x,.2,.03,1.,0);
    pruneProjectedSupports(exact,x,.2,.03,1.,1);
    pruneProjectedSupports(hybrid,x,.2,.03,1.,2);
    assert(!triangle.supports.back().redundant);
    assert(exact.supports.back().redundant&&hybrid.supports.back().redundant);
    assert(exact.exact_reach_additional>0&&hybrid.exact_reach_additional>0);
    checkProofs(exact,.2,.03,1.);checkProofs(hybrid,.2,.03,1.);
  }
  for(int i=0;i<=1000;++i)for(double q:{std::expm1(-30.+60.*i/1000),1e-14*std::cos(double(i))}) {
    CubicProblem::LineModel line;line.terms.push_back({0,q,1});
    const long double exact=-std::log1p(static_cast<long double>(q));
    assert(static_cast<long double>(line.upperBound(1))>=exact);
  }
  for(const Vec3 y:{Vec3(0,0,0),Vec3(.2,.1,.3),Vec3(.2,1e-10,-2e-10)}) {
    auto jet=[&](const Vec3& p,Vec3& g,Eigen::Matrix3d& h) {
      const Vec3 v(1-p[0],p[1],p[2]);const double r=supportRadius(v),gap=v[0]-r;
      distanceBarrierJet(v,.003,r,gap,g,h);return -.003*std::log(gap);
    };
    Vec3 g;Eigen::Matrix3d h;jet(y,g,h);
    assert(g.allFinite()&&h.allFinite());
    for(int axis=0;axis<3;++axis) {
      const double step=axis>0&&y.tail<2>().norm()<1e-8?1e-13:1e-6;
      const Vec3 d=Vec3::Unit(axis);Vec3 gp,gm;Eigen::Matrix3d unused;
      jet(y+step*d,gp,unused);jet(y-step*d,gm,unused);
      assert(((gp-gm)/(2*step)-h.col(axis)).norm()<1e-6*std::max(1.,h.col(axis).norm()));
    }
  }
  for(int mode:{0,1,2})for(double tilt:{0.,1e-9,.05,.5})for(bool one_sided:{false,true}) {
    CubicProblem::Group group;group.first=0;group.count=2;group.reference.setZero();
    group.maps[0].col(0)=.5*Vec3::UnitY();group.maps[0].col(1)=.5*Vec3::UnitZ();
    group.maps[1].col(0)=.5*Vec3(std::sin(tilt),std::cos(tilt),0);
    group.maps[1].col(1)=.5*Vec3::UnitZ();
    Eigen::VectorXd initial=Eigen::VectorXd::Zero(10);
    for(int i=0;i<2;++i){initial[5*i+2]=.15;initial[5*i+4]=.15;}
    initial.segment<2>(0)<<.04,-.03;initial.segment<2>(5)<<-.02,.01;
    auto add=[&](Vec3 n,double b){group.supports.push_back({n,b,true,false,-1,int(group.supports.size())});};
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
    pruneProjectedSupports(group,initial,1.,.03,1.,mode);
    assert(group.proofs.size()>=6);checkProofs(group,1.,.03,1.);
    int feasible=0;
    for(int trial=0;trial<400;++trial) {
      Eigen::VectorXd x(10);
      for(int i=0;i<2;++i) {
        const double a=.17*(trial+1)*(i+1),b=.31*(trial+2)*(i+1);
        x.segment<2>(5*i)<<.7*std::sin(a),.7*std::cos(a);
        if(mode)x.segment<2>(5*i)=initial.segment<2>(5*i); // Fixed, possibly nonzero endpoints.
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

  auto model=CubicModel::fromSeed({Vec3(-2,0,0),Vec3(2,0,0)},.5,.2,1.5);
  Options options;options.max_offset=2;options.max_radius=3;
  std::vector<detail::SupportPlane> planes;
  for(int span=0;span<int(model.basis.spans.size());++span) {
    for(int axis:{1,2})for(double sign:{-1.,1.})planes.push_back({span,0,1,sign*Vec3::Unit(axis),1.});
    for(int j=0;j<12;++j) {
      const double a=j*.5235987755982988;
      planes.push_back({span,0,1,Vec3(0,std::cos(a),std::sin(a)),1.7});
    }
  }
  for(bool distance:{false,true}) {
  OptimizerSettings settings;settings.tensor_assembly=false;settings.distance_barrier=distance;
  CubicProblem original(model,planes,4.,options,settings);
  settings.reach_bound=1;
  settings.projected_pruning=1;CubicProblem assembly(model,planes,4.,options,settings);
  settings.projected_pruning=2;CubicProblem complete(model,planes,4.,options,settings);
  settings.reference_assembly=true;CubicProblem dense(model,planes,4.,options,settings);
  for(const auto& group:complete.groups){assert(group.boundary_count<group.supports.size());checkProofs(group,2.,options.min_radius,3.);}
  Eigen::VectorXd ga,gb;BandedSPD ha,hb;
  const double fa=complete.objective(model.x,.003,&ga,&ha),fb=dense.objective(model.x,.003,&gb,&hb);
  assert(std::abs(fa-fb)<1e-12);assert((ga-gb).norm()<1e-11);assert((ha.data-hb.data).norm()<1e-10);
  Eigen::VectorXd gradient_only;
  assert(complete.objective(model.x,.003,&gradient_only)==fa);
  assert((gradient_only-ga).norm()<1e-12);
  assert(std::abs(dense.objective(model.x,.003,&gradient_only)-fb)<1e-12);
  assert((gradient_only-gb).norm()<1e-12);
  auto axis_settings=settings;axis_settings.reference_assembly=false;axis_settings.axis_kernels=true;
  CubicProblem axial(model,planes,4.,options,axis_settings);
  Eigen::VectorXd axis_gradient;BandedSPD axis_hessian;
  assert(axial.objective(model.x,.003,&axis_gradient,&axis_hessian)==fa);
  assert((axis_gradient-ga).norm()<1e-12);assert((axis_hessian.data-ha.data).norm()<1e-11);
  Eigen::VectorXd derivatives_only;BandedSPD derivatives_hessian;
  assert(complete.objective(model.x,.003,&derivatives_only,&derivatives_hessian,nullptr,0,nullptr,false)==0);
  assert((derivatives_only-ga).squaredNorm()==0);
  assert((derivatives_hessian.data-ha.data).squaredNorm()==0);
  auto infeasible=model.x;infeasible[2]=-1;
  assert(!std::isfinite(complete.objective(infeasible,.003,&derivatives_only,&derivatives_hessian,nullptr,0,nullptr,false)));
  if(!distance)for(double ratio:{.15,.3,.5,.7}) {
    BandedSPD approximation,reference;std::uint64_t weak_rows=0;
    Eigen::VectorXd gradient,reference_gradient;
    assert(complete.objective(model.x,.003,&gradient,&approximation,nullptr,ratio,&weak_rows)==fa);
    assert((gradient-ga).norm()<1e-12);
    assert(std::abs(dense.objective(model.x,.003,&reference_gradient,&reference,nullptr,ratio)-fa)<1e-12);
    assert((approximation.data-reference.data).norm()<1e-10);
    assert(approximation.factor(0));
  }
  OptimizerMetrics metrics;complete.checkDerivatives(model.x,metrics);
  assert(metrics.gradient_error<1e-7&&metrics.hessian_error<1e-7);
  for(int trial=0;trial<32;++trial) {
    Eigen::VectorXd direction(model.x.size());
    for(int j=0;j<direction.size();++j)direction[j]=3*std::sin((j+1)*(trial+1)*.193);
    const auto a=assembly.prepareLine(model.x,direction,.003),b=complete.prepareLine(model.x,direction,.003);
    const auto fast=axial.prepareLine(model.x,direction,.003);
    assert(fast.step==b.step);
    assert(std::abs(a.step-b.step)<1e-12);
    assert(std::abs(b.step-complete.feasibleStep(model.x,direction))<1e-12);
    assert(std::isfinite(original.objective(model.x+b.step*direction,.003)));
    CubicProblem::Evaluation values;complete.objective(model.x,.003,nullptr,nullptr,&values);
    const auto cached=complete.prepareLine(model.x,direction,.003,&values);
    assert(cached.step==b.step);
    for(double fraction:{.1,.5,1.}) {
      const double alpha=b.step*fraction;
      const double actual=complete.objective(model.x+alpha*direction,.003)-fa;
      assert(std::abs(b.delta(alpha)-actual)<1e-10*std::max(1.,std::abs(actual)));
      assert(cached.delta(alpha)==b.delta(alpha));
      assert(fast.delta(alpha)==b.delta(alpha));
      assert(b.upperBound(alpha)>=b.delta(alpha));
    }
  }
  auto lag_settings=settings;lag_settings.reference_assembly=false;lag_settings.hessian_interval=2;
  CubicProblem lagged(model,planes,4.,options,lag_settings);
  auto optimized=model.x;OptimizerMetrics lag_metrics;
  lagged.optimize(optimized,lag_metrics,[]{});
  assert(lag_metrics.hessian_refreshes<lag_metrics.steps);
  assert(std::isfinite(original.objective(optimized,.003)));
  assert(certify(model.geometry(optimized),planes).safe());
  auto exact_settings=settings;exact_settings.reference_assembly=false;
  CubicProblem with_value(model,planes,4.,options,exact_settings);
  exact_settings.skip_unused_values=true;
  CubicProblem without_value(model,planes,4.,options,exact_settings);
  Eigen::VectorXd a=model.x,b=model.x;OptimizerMetrics ma,mb;
  with_value.optimize(a,ma,[]{});without_value.optimize(b,mb,[]{});
  assert((a-b).squaredNorm()==0);assert(ma.steps==mb.steps&&ma.backtracks==mb.backtracks);
  exact_settings.bounded_line=true;
  CubicProblem bounded(model,planes,4.,options,exact_settings);
  auto c=model.x;OptimizerMetrics mc;bounded.optimize(c,mc,[]{});
  assert((a-c).squaredNorm()==0);assert(ma.steps==mc.steps&&ma.backtracks==mc.backtracks);
  if(!distance)assert(mc.bound_line_accepts>0);
  }
  std::cout<<"projected proofs retain witnesses; original constraints, derivatives and line bounds verified\n";
}
