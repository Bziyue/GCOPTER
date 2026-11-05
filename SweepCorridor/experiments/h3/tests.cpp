#include "../free_boundary/common.hpp"
#include "generator.hpp"
namespace h3=SweepCorridor::experiments::h3;
#include <cassert>
void capDirections(){
  // Correlated cap directions exercise both adding an initially feasible
  // constraint and dropping an initially violated one from the active set.
  h3::Model model(c3::CubicModel::fromSeed({{-2,0,0},{2,0,0}},1,.4,1.5).geometry());
  const int n=model.x.size();Eigen::MatrixXd a=Eigen::MatrixXd::Zero(2,n);
  a.block<1,3>(0,0)=h3::inwardCapNormal(model,0).transpose();
  a.block<1,3>(1,n-6)=h3::inwardCapNormal(model,1).transpose();
  for(double rho:{-.8,.8})for(Eigen::Vector2d proposed:{Eigen::Vector2d(-.5,.1),Eigen::Vector2d(-1,-.1)}){
    const Eigen::MatrixXd inverse=Eigen::MatrixXd::Identity(n,n)+rho*(a.row(0).transpose()*a.row(1)+a.row(1).transpose()*a.row(0));
    auto solve=[&](const Eigen::VectorXd& rhs,Eigen::VectorXd& answer){answer=inverse*rhs;};
    Eigen::VectorXd step=a.transpose()*proposed;const auto original=step;
    h3::constrainCapStep(model,model.x,step,solve);
    const Eigen::Vector2d gaps=a*step;assert(gaps.minCoeff()>-1e-12);
    const Eigen::VectorXd force=inverse.ldlt().solve(step-original);
    const Eigen::Vector2d multipliers=a*force;
    assert(multipliers.minCoeff()>-1e-12);
    assert((force-a.transpose()*multipliers).norm()<1e-12);
    assert((gaps.array()*multipliers.array()).abs().maxCoeff()<1e-12);
  }
}
void blockSolve(){
  h3::BlockTridiagonal a(6*10);Eigen::MatrixXd dense=Eigen::MatrixXd::Zero(a.n,a.n);
  for(int i=0;i<10;++i){const h3::Block b=h3::Block::Random();a.diagonal[i]=b*b.transpose()+10*h3::Block::Identity();dense.block<6,6>(6*i,6*i)=a.diagonal[i];
    if(i){a.lower[i-1]=h3::Block::Random();dense.block<6,6>(6*i,6*i-6)=a.lower[i-1];dense.block<6,6>(6*i-6,6*i)=a.lower[i-1].transpose();}}
  assert(a.factor());const Eigen::VectorXd rhs=Eigen::VectorXd::Random(a.n);Eigen::VectorXd answer;a.solveInto(rhs,answer);
  assert((dense*answer-rhs).norm()<1e-12);assert((a.multiply(rhs)-dense*rhs).norm()<1e-12);
}
void exactRepresentation(){
  const auto reference=c3::CubicModel::fromSeed({{0,0,0},{1,.4,0},{2,1,.3}},.4,.4,1.5,{0,.09,.4,.71,1}).geometry();
  h3::Model m(reference);const auto h=m.geometry();double error=0;
  for(int k=0;k<101;++k){const double u=k/100.;const auto a=reference.section(u),b=h.section(u);int span;Eigen::Vector2d weights;const auto c=m.section(m.x,u,span,weights);
    error=std::max({error,(a.center-b.center).norm(),(a.shape-b.shape).norm(),(a.first-b.first).norm(),(a.second-b.second).norm(),(a.shape_second-b.shape_second).norm(),(b.first-c.first).norm(),(b.shape_second-c.shape_second).norm()});}
  assert(error<1e-10);
  for(int i=0;i<m.basis.count;++i)m.x.segment<3>(6*i)+=.003*sc::Vec3::Random();
  const auto moved=m.geometry();double c1=0,c2=0;
  for(int i=1;i<m.basis.count-1;++i){const auto a=c3::evaluate<3>(moved.curve(i-1),1,m.basis.breaks[i]-m.basis.breaks[i-1]),b=c3::evaluate<3>(moved.curve(i),0,m.basis.breaks[i+1]-m.basis.breaks[i]);
    const sc::Vec3 point=a.center+.2*a.shape.col(0);
    const auto qa=sc::detail::differential(a,point),qb=sc::detail::differential(b,point);
    assert((qa.gradient-qb.gradient).norm()<1e-9);
    c1=std::max({c1,(a.first-b.first).norm(),(a.shape_first-b.shape_first).norm()});c2=std::max(c2,(a.second-b.second).norm());}
  assert(c1<1e-11&&c2>.01);std::cout<<"representation_error="<<error<<" c1_error="<<c1<<" c2_jump="<<c2<<'\n';
}
void affineOffsetPruning(){
  c3::CubicProblem::Group g;g.first=0;g.count=1;g.reference.setZero();g.reference[3]=.49;
  g.maps[0]<<0,0,1,0,0,1;g.center_weights[0]=1;
  for(const sc::Vec3& n:{sc::Vec3(0,1,0),sc::Vec3(0,-1,0),sc::Vec3(0,0,1),sc::Vec3(0,0,-1),sc::Vec3(2,.1,.1)}){
    c3::CubicProblem::Support p;p.normal=n;p.bound=1;p.original_id=g.supports.size();g.supports.push_back(p);}
  Eigen::VectorXd x=Eigen::VectorXd::Zero(6);x[3]=x[5]=.05;
  auto incorrect=g;c3::pruneProjectedSupportsImpl<false>(incorrect,x,.02,1,nullptr);
  c3::pruneProjectedSupportsImpl<true>(g,x,.02,1,nullptr);
  assert(incorrect.supports.back().redundant&&!g.supports.back().redundant);
  x[0]=.01;x[3]=x[5]=.5;
  const auto y=g.geometry(x);for(int i=0;i<4;++i){const auto v=g.supports[i].project(y);assert(v[0]>v.tail<2>().norm());}
  const auto violation=g.supports.back().project(y);assert(violation[0]<violation.tail<2>().norm());
}
int main(){
  capDirections();
  blockSolve();
  affineOffsetPruning();
  exactRepresentation();
  const std::vector<sc::Vec3> seed{{-2,0,0},{0,.4,.2},{2,.3,.5}};
  sc::Options options;const sc::Bounds bounds{{-10,-10,-10},{10,10,10}};sc::ObstacleIndex empty({});
  h3::Model model(c3::CubicModel::fromSeed(seed,.5,.4,1.5).geometry());auto x=model.x;
  for(int i=0;i<model.basis.count;++i)x.segment<3>(6*i)=.003*sc::Vec3(std::sin(i),std::cos(i),std::sin(2*i));
  const auto g=model.geometry(x);const double u=.371;const auto section=g.section(u);
  const sc::Vec3 point=section.point(Eigen::Vector2d(.3,.2));h3::Anchor anchor;anchor.u=u+.01;
  assert(h3::locateAnchor(model,x,point,anchor,true));double anchor_error=0;
  for(int k=0;k<12;++k){const int index=6*anchor.span+k;auto plus=x,minus=x;plus[index]+=1e-6;minus[index]-=1e-6;
    auto ap=anchor,am=anchor;assert(h3::locateAnchor(model,plus,point,ap,false)&&h3::locateAnchor(model,minus,point,am,false));
    anchor_error=std::max(anchor_error,((ap.z-am.z)/2e-6-anchor.derivative.col(k)).norm());}
  std::cerr<<"anchor_error="<<anchor_error<<'\n';assert(anchor_error<2e-6);
  Eigen::VectorXd vg;const double volume=h3::sweptVolume(model,x,&vg);assert(volume>0);double volume_error=0;
  for(int i=0;i<x.size();++i){auto plus=x,minus=x;plus[i]+=1e-6;minus[i]-=1e-6;
    const double fd=(h3::sweptVolume(model,plus)-h3::sweptVolume(model,minus))/2e-6;
    volume_error=std::max(volume_error,std::abs(fd-vg[i]));}assert(volume_error<2e-6);
  c3::QueryCache cache;c3::CoverMetrics metrics;c3::Deadline deadline(10000,c3::Clock::now());
  c3::CoverLayout layout(model.basis.breaks,seed,bounds,options);
  auto planes=c3::mapCover(g,empty,bounds,options,cache,metrics,deadline,layout,false);
  c3::OptimizerSettings settings;
  h3::Problem problem(model,planes,seedLength(seed),options,settings,{},nullptr,true);
  // Force two-node subdivision rows in addition to one-node endpoint rows.
  auto subplanes=planes;for(auto& p:subplanes){p.begin=.13;p.end=.81;}
  planes.insert(planes.end(),subplanes.begin(),subplanes.end());
  h3::Problem subdivided(model,planes,seedLength(seed),options,settings);
  const double mu=.001;Eigen::VectorXd gradient;h3::BlockTridiagonal hessian;
  const double value=subdivided.objective(x,mu,&gradient,&hessian);assert(std::isfinite(value));
  double gradient_error=0,hessian_error=0;
  for(int i=0;i<x.size();++i){auto plus=x,minus=x;plus[i]+=1e-6;minus[i]-=1e-6;Eigen::VectorXd gp,gm;
    const double fp=subdivided.objective(plus,mu,&gp),fm=subdivided.objective(minus,mu,&gm);
    gradient_error=std::max(gradient_error,std::abs((fp-fm)/2e-6-gradient[i]));
    Eigen::VectorXd unit=Eigen::VectorXd::Zero(x.size());unit[i]=1;
    hessian_error=std::max(hessian_error,((gp-gm)/2e-6-hessian.multiply(unit)).norm());}
  assert(gradient_error<1e-5&&hessian_error<1e-4);
  Eigen::VectorXd direction=Eigen::VectorXd::LinSpaced(x.size(),-.01,.01);const auto line=subdivided.prepareLine(x,direction,mu);
  for(double alpha:{0.,.2,.7,1.})assert(std::abs(line.delta(alpha)-(subdivided.objective(x+alpha*direction,mu)-value))<1e-10);
  const auto polynomial=h3::volumeLine(model,x,direction);
  for(double alpha:{-.8,.2,.7,1.})assert(std::abs(alpha*(polynomial[0]+alpha*(polynomial[1]+alpha*polynomial[2]))-(h3::sweptVolume(model,x+alpha*direction)-volume))<1e-10);
  const auto natural=h3::naturalJets(model);const auto saved=model.jets;model.jets=natural;model.cacheQuadrature();
  assert(quality(model.geometry(),seedLength(seed)).c2_error<1e-8);model.jets=saved;model.cacheQuadrature();
  h3::JetMetrics jm;h3::updateJets(model,planes,jm,deadline,false);assert(c3::certify(model.geometry(),planes).safe());
  const std::vector<sc::Vec3> straight{{-2,0,0},{2,0,0}};
  auto result=h3::generate(straight,empty,bounds,options,settings);
  std::cout<<sc::statusName(result.status)<<' '<<result.message<<'\n';assert(result.status==sc::Status::Success);
  assert(c3::certify(result.corridor,result.cover).safe());
  assert(result.corridor.section(0).center.x() < -2.1 && result.corridor.section(1).center.x()>2.1);
  // Only 8 mm of longitudinal space exists outside this start. Expansion
  // must stop at the same world boundary, without a special endpoint rule.
  const sc::Bounds near_wall{{-2.26,-10,-10},{10,10,10}};
  auto bounded=h3::generate(straight,empty,near_wall,options,settings);
  assert(bounded.status==sc::Status::Success&&c3::certify(bounded.corridor,bounded.cover).safe());
  assert(bounded.corridor.section(0).center.x()>=-2.008-1e-8);
  assert(bounded.corridor.section(0).center.x()<=-2+1e-8);
  std::cout<<"anchor_derivative_error="<<anchor_error<<" volume_gradient_error="<<volume_error
           <<" support_gradient_error="<<gradient_error<<" support_hessian_error="<<hessian_error
           <<" start="<<result.corridor.section(0).center.transpose()<<" end="<<result.corridor.section(1).center.transpose()<<'\n';
}
