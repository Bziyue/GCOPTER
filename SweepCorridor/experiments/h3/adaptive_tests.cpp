#include "../free_boundary/common.hpp"
#include "generator.hpp"
#include <cassert>
namespace h3=SweepCorridor::experiments::h3;
int main(){
  std::vector<sc::Vec3> seed{{0,0,0},{2,0,0},{4,0,0}};
  h3::Model m(c3::CubicModel::fromSeed(seed,.4,.4,1.5,{0,.09,.4,.71,1}).geometry());
  for(int i=0;i<m.basis.count;++i){m.x[6*i]=.3*(2*m.basis.breaks[i]-1);m.jets[i][0]+=.6;}
  m.cacheQuadrature();sc::Options options;const sc::Bounds bounds{{-10,-10,-10},{10,10,10}};sc::ObstacleIndex empty({});
  c3::QueryCache cache;c3::CoverMetrics metrics;c3::Deadline deadline(10000,c3::Clock::now());
  c3::CoverLayout layout(m.basis.breaks,seed,bounds,options);const double length=seedLength(seed);
  auto cover=c3::mapCover(m.geometry(),empty,bounds,options,cache,metrics,deadline,layout,false);
  for(int span=0;span<m.basis.count-1;++span)cover.push_back({span,.13,.89,sc::Vec3(1,.7,.3).normalized(),8});
  h3::Problem constraints(m,cover,length,options,h3::OptimizerSettings{}, {},nullptr,false);
  double volume_error=0,gradient_error=0,hessian_error=0,anchor_error=0,jacobian_error=0;
  for(const auto range:{std::pair<int,int>{0,3},{1,m.basis.count-1},{m.basis.count-3,m.basis.count},{0,m.basis.count}}){
    h3::CenterBlock patch(m,range.first,range.second,options,length,3,*constraints.geometry);
    const auto original=m.geometry();const Eigen::VectorXd d=.002*Eigen::VectorXd::Random(patch.size);
    patch.assign(d);m.cacheQuadrature();volume_error=std::max(volume_error,std::abs(h3::sweptVolume(m,m.x)-patch.volume-patch.volume_gradient.dot(d)));
    const auto changed=m.geometry();
    for(int span=patch.span_begin;span<patch.span_end;++span){const auto reference=c3::jacobianCoefficients<3>(changed.curve(span));
      const auto& affine=patch.jacobians[span-patch.span_begin];const Eigen::Matrix<double,9,1> delta=affine.change*d;
      for(int j=0;j<9;++j){auto v=affine.base[j];v[0]+=delta[j];jacobian_error=std::max(jacobian_error,(reference[j]-v).norm());}}
    Eigen::VectorXd g;h3::BlockTridiagonal h;const double mu=.001;
    assert(std::isfinite(patch.objective(d,mu,&g,&h)));const double total=patch.volume+patch.volume_gradient.dot(d);
    for(int j=0;j<patch.size;++j){Eigen::VectorXd plus=d,minus=d,gp,gm;plus[j]+=1e-6;minus[j]-=1e-6;
      const double fp=patch.objective(plus,mu,&gp),fm=patch.objective(minus,mu,&gm);
      gradient_error=std::max(gradient_error,std::abs((fp-fm)/2e-6-g[j]));
      for(int k=0;k<patch.size;++k)hessian_error=std::max(hessian_error,std::abs((gp[k]-gm[k])/2e-6-h.get(k,j)-patch.volume_gradient[k]*patch.volume_gradient[j]/(total*total)));
    }
    std::array<h3::Anchor,2> anchors;anchors[1].u=1;
    Eigen::VectorXd ag=Eigen::VectorXd::Zero(patch.size);h3::BlockTridiagonal ah(patch.size);
    const double ac=patch.anchorCost(anchors,mu,&ag,&ah);assert(std::isfinite(ac));
    for(int j=0;j<patch.size;++j){Eigen::VectorXd plus=d,minus=d;plus[j]+=1e-6;minus[j]-=1e-6;
      auto a=anchors,b=anchors;patch.assign(plus);const double fp=patch.anchorCost(a,mu);patch.assign(minus);const double fm=patch.anchorCost(b,mu);
      anchor_error=std::max(anchor_error,std::abs((fp-fm)/2e-6-ag[j]));}
    patch.assign(d);assert(patch.regular(d));
    const Eigen::VectorXd step=.001*Eigen::VectorXd::Random(patch.size);const auto line=patch.prepareLine(d,step,mu);
    for(double alpha:{0.,.2,.7,1.}){
      const double approximate=line.delta(alpha)-std::log1p(alpha*patch.volume_gradient.dot(step)/total);
      const double exact=patch.objective(d+alpha*step,mu)-patch.objective(d,mu);assert(std::abs(approximate-exact)<1e-11);}
    patch.assign(Eigen::VectorXd::Zero(patch.size));m.cacheQuadrature();
  }
  std::cout<<"center_volume_error="<<volume_error<<" gradient_error="<<gradient_error<<" hessian_error="<<hessian_error
           <<" anchor_gradient_error="<<anchor_error<<" affine_jacobian_error="<<jacobian_error<<std::endl;
  assert(volume_error<1e-10&&gradient_error<1e-7&&hessian_error<1e-6&&anchor_error<1e-7&&jacobian_error<1e-10);
  h3::OptimizerSettings settings;settings.refresh_rounds=0;settings.end_steps=4;
  const auto result=h3::generate(seed,empty,bounds,options,settings);
  assert(result.status==sc::Status::Success&&c3::certify(result.corridor,result.cover).safe());
  std::cout<<"end_steps="<<result.jets.end_steps<<" end_volume_gain="<<result.jets.end_volume_gain<<std::endl;
  sc::ObstacleIndex obstacles({{{1,1.5,.1},.2},{{2,-1.6,-.2},.25},{{3,1.7,.3},.2}});
  c3::QueryCache local_cache;c3::CoverMetrics local_metrics;
  auto before=c3::mapCover(m.geometry(),obstacles,bounds,options,local_cache,local_metrics,deadline,layout,false);
  m.x[7]+=.03;const auto current=m.geometry();
  const auto complete=c3::mapCover(current,obstacles,bounds,options,local_cache,local_metrics,deadline,layout,false);
  std::vector<unsigned char> selected(current.pieces(),0);selected[0]=selected[1]=1;h3::JetMetrics jm;
  auto updated=before;const auto refreshed=h3::refreshCover(current,updated,selected,obstacles,bounds,options,local_cache,local_metrics,layout,jm,deadline,false);
  assert(jm.refresh_spans==2&&jm.refresh_cache_hits==2&&jm.refresh_failures==0);
  for(int span=0;span<int(current.pieces());++span){std::vector<sc::internal::detail::SupportPlane> a,b;
    for(const auto& p:updated)if(p.span==span)a.push_back(p);
    for(const auto& p:(selected[span]?complete:before))if(p.span==span)b.push_back(p);
    assert(a.size()==b.size());for(size_t j=0;j<a.size();++j)assert(a[j].begin==b[j].begin&&a[j].end==b[j].end&&a[j].normal==b[j].normal&&a[j].bound==b[j].bound);
  }
  assert(c3::certify(current,updated).safe());
  h3::GroupBuildMetrics grouping;auto local_groups=h3::buildSupportGroups(m,updated,grouping,&selected);
  std::vector<unsigned char> nodes(m.basis.count,0);nodes[1]=1;settings.joint_sweeps=1;
  h3::optimizeJointJets(m,updated,length,options,settings,jm,deadline,local_groups.get(),&nodes);
  const auto refined=m.geometry();assert(c3::certify(refined,updated).safe());
  for(int span=2;span<int(current.pieces());++span)for(int j=0;j<4;++j)assert(current.controls[4*span+j]==refined.controls[4*span+j]);
  settings.joint_sweeps=2;settings.joint_budget_fraction=.5;settings.joint_budget_share=1;settings.refresh_fraction=.2;settings.refresh_rounds=1;
  const auto adaptive=h3::generate(seed,obstacles,bounds,options,settings);
  assert(adaptive.status==sc::Status::Success&&c3::certify(adaptive.corridor,adaptive.cover).safe());
  assert(adaptive.jets.joint_budget_skips>0&&adaptive.jets.refresh_spans>0&&adaptive.jets.refresh_spans==adaptive.jets.refresh_cache_hits);
  std::cout<<"partial_cover_exact=1 unchanged_spans_exact=1 adaptive_skips="<<adaptive.jets.joint_budget_skips
           <<" refreshed_spans="<<adaptive.jets.refresh_spans<<std::endl;
  h3::OptimizerSettings tiny_settings;sc::ObstacleIndex nearby({{{.1,1.5,.1},.2}});
  const auto tiny=h3::generate({{0,0,0},{.2,0,0}},nearby,bounds,options,tiny_settings);
  assert(tiny.status==sc::Status::Success&&c3::certify(tiny.corridor,tiny.cover).safe()&&tiny.jets.end_steps>0);
  tiny_settings.joint_jet_steps=0;
  assert(h3::generate(seed,empty,bounds,options,tiny_settings).status==sc::Status::InvalidInput);
  std::cout<<"merged_short_end_block=1 invalid_schedule_rejected_at_entry=1"<<std::endl;
}
