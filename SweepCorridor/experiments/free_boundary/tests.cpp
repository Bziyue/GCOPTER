#include "common.hpp"
#include <cassert>
void capDirections(){
  // Correlated cap directions exercise both adding an initially feasible
  // constraint and dropping an initially violated one from the active set.
  auto model=c3::CubicModel::fromSeed({{-2,0,0},{2,0,0}},1,.4,1.5);
  const int n=model.x.size();Eigen::MatrixXd a=Eigen::MatrixXd::Zero(2,n);
  a.block<1,3>(0,0)=c3::inwardCapNormal(model,0).transpose();
  a.block<1,3>(1,n-6)=c3::inwardCapNormal(model,1).transpose();
  for(double rho:{-.8,.8})for(Eigen::Vector2d proposed:{Eigen::Vector2d(-.5,.1),Eigen::Vector2d(-1,-.1)}){
    const Eigen::MatrixXd inverse=Eigen::MatrixXd::Identity(n,n)+rho*(a.row(0).transpose()*a.row(1)+a.row(1).transpose()*a.row(0));
    auto solve=[&](const Eigen::VectorXd& rhs,Eigen::VectorXd& answer){answer=inverse*rhs;};
    Eigen::VectorXd step=a.transpose()*proposed;const auto original=step;
    c3::constrainCapStep(model,model.x,step,solve);
    const Eigen::Vector2d gaps=a*step;assert(gaps.minCoeff()>-1e-12);
    const Eigen::VectorXd force=inverse.ldlt().solve(step-original);
    const Eigen::Vector2d multipliers=a*force;
    assert(multipliers.minCoeff()>-1e-12);
    assert((force-a.transpose()*multipliers).norm()<1e-12);
    assert((gaps.array()*multipliers.array()).abs().maxCoeff()<1e-12);
  }
}
int main(){
  capDirections();
  const std::vector<sc::Vec3> seed{{-2,0,0},{0,.4,.2},{2,.3,.5}};
  sc::Options options;const sc::Bounds bounds{{-10,-10,-10},{10,10,10}};sc::ObstacleIndex empty({});
  auto model=c3::CubicModel::fromSeed(seed,.5,.4,1.5);auto x=model.x;
  for(int i=0;i<model.basis.count;++i)x.segment<3>(6*i)=.003*sc::Vec3(std::sin(i),std::cos(i),std::sin(2*i));
  const auto g=model.geometry(x);const double u=.371;const auto section=g.section(u);
  const sc::Vec3 point=section.point(Eigen::Vector2d(.3,.2));c3::Anchor anchor;anchor.u=u+.01;
  assert(c3::locateAnchor(model,x,point,anchor,true));double anchor_error=0;
  for(int k=0;k<24;++k){const int index=6*anchor.span+k;auto plus=x,minus=x;plus[index]+=1e-6;minus[index]-=1e-6;
    auto ap=anchor,am=anchor;assert(c3::locateAnchor(model,plus,point,ap,false)&&c3::locateAnchor(model,minus,point,am,false));
    anchor_error=std::max(anchor_error,((ap.z-am.z)/2e-6-anchor.derivative.col(k)).norm());}
  std::cerr<<"anchor_error="<<anchor_error<<'\n';assert(anchor_error<2e-6);
  Eigen::VectorXd vg;const double volume=c3::sweptVolume(model,x,&vg);assert(volume>0);double volume_error=0;
  for(int i=0;i<x.size();++i){auto plus=x,minus=x;plus[i]+=1e-6;minus[i]-=1e-6;
    const double fd=(c3::sweptVolume(model,plus)-c3::sweptVolume(model,minus))/2e-6;
    volume_error=std::max(volume_error,std::abs(fd-vg[i]));}assert(volume_error<2e-6);
  c3::QueryCache cache;c3::CoverMetrics metrics;c3::Deadline deadline(10000,c3::Clock::now());
  c3::CoverLayout layout(model.basis.breaks,seed,bounds,options);
  auto planes=c3::mapCover(g,empty,bounds,options,cache,metrics,deadline,layout,false);
  c3::ProgressConstraints progress;progress.spans={1};c3::OptimizerSettings settings;
  c3::CubicProblem problem(model,planes,seedLength(seed),options,settings,{},nullptr,progress);
  const double mu=.001;Eigen::VectorXd gradient;c3::BandedSPD hessian;
  const double value=problem.objective(x,mu,&gradient,&hessian);assert(std::isfinite(value));
  double gradient_error=0,hessian_error=0;
  for(int i=0;i<x.size();++i){auto plus=x,minus=x;plus[i]+=1e-6;minus[i]-=1e-6;Eigen::VectorXd gp,gm;
    const double fp=problem.objective(plus,mu,&gp),fm=problem.objective(minus,mu,&gm);
    gradient_error=std::max(gradient_error,std::abs((fp-fm)/2e-6-gradient[i]));
    Eigen::VectorXd unit=Eigen::VectorXd::Zero(x.size());unit[i]=1;
    hessian_error=std::max(hessian_error,((gp-gm)/2e-6-hessian.multiply(unit)).norm());}
  assert(gradient_error<1e-5&&hessian_error<1e-4);
  Eigen::VectorXd direction=Eigen::VectorXd::LinSpaced(x.size(),-.01,.01);const auto line=problem.prepareLine(x,direction,mu);
  for(double alpha:{0.,.2,.7,1.})assert(std::abs(line.delta(alpha)-(problem.objective(x+alpha*direction,mu)-value))<1e-10);
  const auto polynomial=c3::volumeLine(model,x,direction);
  for(double alpha:{-.8,.2,.7,1.})assert(std::abs(alpha*(polynomial[0]+alpha*(polynomial[1]+alpha*polynomial[2]))-(c3::sweptVolume(model,x+alpha*direction)-volume))<1e-10);
  const std::vector<sc::Vec3> straight{{-2,0,0},{2,0,0}};
  auto result=c3::generateCubic(straight,empty,bounds,options,settings);
  std::cout<<sc::statusName(result.status)<<' '<<result.message<<'\n';assert(result.status==sc::Status::Success);
  assert(c3::certify(result.corridor,result.cover).safe());assert(quality(result.corridor,4).c2_error<1e-7);
  assert(result.corridor.section(0).center.x() < -2.1 && result.corridor.section(1).center.x()>2.1);
  // Only 8 mm of longitudinal space exists outside this start. Expansion
  // must stop at the same world boundary, without a special endpoint rule.
  const sc::Bounds near_wall{{-2.26,-10,-10},{10,10,10}};
  auto bounded=c3::generateCubic(straight,empty,near_wall,options,settings);
  assert(bounded.status==sc::Status::Success&&c3::certify(bounded.corridor,bounded.cover).safe());
  assert(bounded.corridor.section(0).center.x()>=-2.008-1e-8);
  assert(bounded.corridor.section(0).center.x()<=-2+1e-8);
  std::cout<<"anchor_derivative_error="<<anchor_error<<" volume_gradient_error="<<volume_error
           <<" support_gradient_error="<<gradient_error<<" support_hessian_error="<<hessian_error
           <<" start="<<result.corridor.section(0).center.transpose()<<" end="<<result.corridor.section(1).center.transpose()<<'\n';
}
