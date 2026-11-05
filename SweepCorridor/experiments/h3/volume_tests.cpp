#include "../free_boundary/common.hpp"
#include "generator.hpp"
#include <cassert>
namespace h3=SweepCorridor::experiments::h3;
int main(){
  std::vector<sc::Vec3> seed{{0,0,0},{1,.4,.2},{2,1,.3},{3,1,1}};
  h3::Model m(c3::CubicModel::fromSeed(seed,.4,.4,1.5,{0,.09,.4,.71,1}).geometry());
  const double length=seedLength(seed);auto geometry=m.geometry();const double volume=h3::sweptVolume(m,m.x);
  sc::Options options;const sc::Bounds bounds{{-10,-10,-10},{10,10,10}};sc::ObstacleIndex empty({});
  c3::QueryCache cache;c3::CoverMetrics metrics;c3::Deadline deadline(10000,c3::Clock::now());
  c3::CoverLayout layout(m.basis.breaks,seed,bounds,options);
  auto cover=c3::mapCover(geometry,empty,bounds,options,cache,metrics,deadline,layout,false);
  std::vector<std::vector<int>> ids(geometry.pieces());
  const auto original=cover;for(auto p:original){p.begin=.17;p.end=.83;cover.push_back(p);}
  for(int i=0;i<int(cover.size());++i)ids[cover[i].span].push_back(i);
  double max_volume=0,max_gradient=0,max_hessian=0;
  for(int node=0;node<m.basis.count;++node){h3::JetPatch patch(m,geometry,node,length,1.);patch.addSupports(cover,ids);
    const h3::V9 old=m.jets[node],direction=.01*h3::V9::Random();
    for(double alpha:{-2.,-.4,.2,1.,2.}){m.jets[node]=old+alpha*patch.scale*direction;m.cacheQuadrature();
      const double error=std::abs(h3::sweptVolume(m,m.x)-volume-patch.volumeDelta(alpha*direction));max_volume=std::max(error,max_volume);}
    m.jets[node]=old;m.cacheQuadrature();
    h3::V9 gradient;h3::M9 hessian;const double mu=.001;
    assert(std::isfinite(patch.objective(direction,volume,mu,&gradient,&hessian)));
    hessian-=patch.volume_hessian/(volume+patch.volumeDelta(direction));
    for(int k=0;k<9;++k){auto plus=direction,minus=direction;plus[k]+=1e-6;minus[k]-=1e-6;h3::V9 gp,gm;
      const double fp=patch.objective(plus,volume,mu,&gp),fm=patch.objective(minus,volume,mu,&gm);
      max_gradient=std::max(max_gradient,std::abs((fp-fm)/2e-6-gradient[k]));
      max_hessian=std::max(max_hessian,((gp-gm)/2e-6-hessian.col(k)).norm());}
  }
  std::cout<<"jet_volume_exact_error="<<max_volume<<" gradient_error="<<max_gradient<<" hessian_error="<<max_hessian<<std::endl;
  assert(max_volume<1e-10&&max_gradient<1e-7&&max_hessian<1e-6);
  double energy_error=0;
  for(int node=0;node<m.basis.count;++node){
    h3::JointPatch patch(m,geometry,node,length,options,1.,cover,ids);
    const h3::V15 d=.01*h3::V15::Random();double actual=0;
    for(int span=patch.patch.begin;span<patch.patch.end;++span){
      const double h=m.basis.breaks[span+1]-m.basis.breaks[span];
      for(double t:{.5-.5/std::sqrt(3.),.5+.5/std::sqrt(3.)}){
        const auto a=c3::evaluate<3>(geometry.curve(span),t,h),b=c3::evaluate<3>(patch.curve(span,d),t,h);
        actual+=.5*h/(length*length*length)*(h3::OptimizerSettings::center_energy*(b.second.squaredNorm()-a.second.squaredNorm())+
          h3::OptimizerSettings::shape_energy*(b.shape_second.squaredNorm()-a.shape_second.squaredNorm()));
      }
    }
    const sc::Vec3 x=m.x.segment<3>(6*node);
    actual+=h3::OptimizerSettings::offset_weight*m.basis.weight(node)*((x+d.head<3>()).squaredNorm()-x.squaredNorm());
    energy_error=std::max(energy_error,std::abs(actual-patch.energy_gradient.dot(d)-.5*d.dot(patch.energy_hessian*d)));
  }
  std::cout<<"joint_closed_energy_error="<<energy_error<<std::endl;assert(energy_error<1e-10);
  h3::JointPatch joint(m,geometry,2,length,options,1.,cover,ids);
  const h3::V15 direction=.005*h3::V15::Random();h3::V15 gradient;h3::M15 hessian;
  joint.objective(direction,volume,.001,&gradient,&hessian);
  double joint_gradient_error=0,joint_hessian_error=0;
  for(int k=0;k<15;++k){auto plus=direction,minus=direction;plus[k]+=1e-6;minus[k]-=1e-6;h3::V15 gp,gm,vp,vm;
    const double fp=joint.objective(plus,volume,.001,&gp),fm=joint.objective(minus,volume,.001,&gm);
    joint.volume(plus,&vp);joint.volume(minus,&vm);
    joint_gradient_error=std::max(joint_gradient_error,std::abs((fp-fm)/2e-6-gradient[k]));
    joint_hessian_error=std::max(joint_hessian_error,((gp-gm)/2e-6-hessian.col(k)+(vp-vm)/(2e-6*(volume+joint.volume(direction)-joint.old_volume))).norm());}
  std::cout<<"joint_gradient_error="<<joint_gradient_error<<" joint_hessian_error="<<joint_hessian_error<<std::endl;
  assert(joint_gradient_error<1e-7&&joint_hessian_error<1e-6);
  m.x.segment<3>(6)+=sc::Vec3(.003,-.002,.001);m.jets[3]+=.01*h3::V9::Random();m.cacheQuadrature();
  const auto moved=m.geometry();joint.refresh(moved);
  h3::JointPatch rebuilt(m,moved,2,length,options,1.,cover,ids);h3::V15 ga,gb;h3::M15 ha,hb;
  const double ca=joint.objective(direction,volume,.001,&ga,&ha),cb=rebuilt.objective(direction,volume,.001,&gb,&hb);
  std::cout<<"joint_refresh_error="<<std::max({std::abs(ca-cb),(ga-gb).norm(),(ha-hb).norm()})<<std::endl;
  assert(std::abs(ca-cb)<1e-11&&(ga-gb).norm()<1e-10&&(ha-hb).norm()<1e-10);
  h3::Problem assembled(m,cover,length,options,h3::OptimizerSettings{}, {},nullptr,false);
  std::vector<const h3::NodalGroup*> shared;
  for(const auto& g:assembled.geometry->groups)if(g.first/6<=2&&2<g.first/6+g.count)shared.push_back(&g);
  h3::JointPatch reused(m,moved,2,length,options,1.,cover,ids,&shared);h3::V15 gc;h3::M15 hc;
  const double cc=reused.objective(direction,volume,.001,&gc,&hc);
  std::cout<<"joint_reuse_error="<<std::max({std::abs(cc-cb),(gc-gb).norm(),(hc-hb).norm()})<<std::endl;
  assert(std::abs(cc-cb)<1e-11&&(gc-gb).norm()<1e-10&&(hc-hb).norm()<1e-10);
  const h3::V15 search=.002*h3::V15::Random();const auto line=joint.prepareLine(direction,search,volume,.001);
  for(double alpha:{0.,.2,.7,1.})assert(std::abs(line.delta(alpha)-(joint.objective(direction+alpha*search,volume,.001)-ca))<1e-10);
  h3::OptimizerSettings settings;settings.volume_jet_steps=2;settings.joint_jet_steps=0;settings.refresh_rounds=0;settings.end_steps=0;
  auto result=h3::generate(seed,empty,bounds,options,settings);
  assert(result.status==sc::Status::Success&&c3::certify(result.corridor,result.cover).safe());
  settings.volume_jet_steps=0;settings.joint_jet_steps=2;
  result=h3::generate(seed,empty,bounds,options,settings);
  assert(result.status==sc::Status::Success&&c3::certify(result.corridor,result.cover).safe());
}
