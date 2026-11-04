// SPDX-License-Identifier: MIT
#include "cubic_generator.hpp"
#include "support_refinement.hpp"
#include <cassert>
#include <iostream>
using namespace gcopter::direct;
using namespace gcopter::direct::native;
int main() {
  auto model=CubicModel::fromSeed({Vec3(0,0,0),Vec3(3,0,0)},4,.5,1.5);
  assert(model.basis.count==4);
  for(int i:{1,2})model.x[5*i+2]=model.x[5*i+4]=1.5;
  std::vector<detail::SupportPlane> cover;
  for(int axis=0;axis<2;++axis)for(double sign:{-1.,1.})
    cover.push_back({0,0,1,sign*model.frames[0].col(axis),1.51});
  Options options;options.min_radius=.03;options.max_radius=3;options.max_offset=3;
  OptimizerSettings base;base.projected_pruning=2;base.axis_kernels=true;
  CubicProblem coarse(model,cover,3,options,base);
  auto expanded=model.x;
  for(int i:{1,2})expanded[5*i+2]=expanded[5*i+4]=1.6;
  assert(!std::isfinite(coarse.objective(expanded,.003)));
  for(int mode:{1,2}) {
    auto settings=base;settings.certificate_refinement=mode;settings.certificate_fraction=1;
    const auto selection=selectSupportRefinement(model.geometry(),cover,mode,1);
    assert(selection.intervals==1&&selection.selected_intervals==1&&selection.selected_planes==4);
    auto rescaled=cover;for(auto& p:rescaled){p.normal*=17;p.bound*=17;}
    const auto same=selectSupportRefinement(model.geometry(),rescaled,mode,1);
    assert(same.selected==selection.selected);
    CubicProblem fine(model,cover,3,options,settings);
    // Coarse rows must be replaced: retaining them would reject this geometry.
    assert(std::isfinite(fine.objective(expanded,.003)));
    assert(certify(model.geometry(expanded),cover).safe());
    assert(fine.refined_planes==4&&fine.refinement_extra_rows==4*(mode==1?2:3));
    OptimizerMetrics metrics;fine.checkDerivatives(model.x,metrics);
    assert(metrics.gradient_error<1e-7&&metrics.hessian_error<1e-7);
    assert(metrics.gradient_reference_error<1e-12&&metrics.hessian_reference_error<1e-12);
    // A strict coarse interior stays feasible for the replacement and its
    // step-boundary kernels, including after projected pruning is rebuilt.
    for(int trial=0;trial<40;++trial) {
      Eigen::VectorXd direction(model.x.size());
      for(int j=0;j<direction.size();++j)direction[j]=std::sin((j+1)*(trial+1)*.173);
      const auto x=(model.x+coarse.feasibleStep(model.x,direction)*direction).eval();
      assert(std::isfinite(fine.objective(x,.003)));
      const auto alpha=fine.feasibleStep(model.x,direction);
      const auto y=(model.x+alpha*direction).eval();
      for(size_t j=0;j<fine.geometric_cuts;++j) {
        const auto& c=fine.cuts[j];const auto v=c.apply(y);
        assert(c.bound-v[0]>v.tail<2>().norm());
      }
    }
    std::cout<<"mode "<<mode<<" gradient "<<metrics.gradient_error<<" Hessian "<<metrics.hessian_error<<'\n';
  }
  // Nonuniform knots and subinterval endpoints: compare the sparse rows
  // with certificates extracted independently from the world-space geometry.
  auto uneven=CubicModel::fromSeed({Vec3(0,0,0),Vec3(2,.3,0),Vec3(4,0,.3)},.5,.4,1.5,{0,.13,.55,1});
  for(int i=0;i<uneven.basis.count;++i)uneven.x[5*i+2]=uneven.x[5*i+4]=.5+.15*std::sin(2*i);
  std::vector<detail::SupportPlane> planes;
  for(int span=0;span<int(uneven.basis.spans.size());++span)for(double a:{0.,.5})for(int axis=0;axis<3;++axis)
    planes.push_back({span,a,a+.5,Vec3::Unit(axis),8.});
  for(int mode:{1,2}) {
    auto settings=base;settings.certificate_refinement=mode;settings.certificate_fraction=1;
    CubicProblem p(uneven,planes,4,options,settings);
    const auto selection=selectSupportRefinement(uneven.geometry(),planes,mode,1);
    for(int trial=0;trial<10;++trial) {
      auto x=uneven.x;
      for(int j=0;j<x.size();++j)x[j]+=.005*std::sin((trial+1)*(j+1));
      const auto geometry=uneven.geometry(x);double physical=-INFINITY,sparse=-INFINITY;
      for(size_t i=0;i<planes.size();++i) {
        const auto& plane=planes[i];
        const auto curve=restrictCurve<3>(geometry.curve<3>(plane.span),plane.begin,plane.end);
        double bound;
        if(!selection.selected[i])bound=supportBound<3>(curve,plane.normal,plane.bound);
        else if(mode==1)bound=supportBound<5>(elevate<3,5>(curve),plane.normal,plane.bound);
        else {
          const auto halves=split<3>(curve);
          bound=std::max(supportBound<3>(halves.first,plane.normal,plane.bound),supportBound<3>(halves.second,plane.normal,plane.bound));
        }
        physical=std::max(physical,bound);
      }
      for(size_t i=0;i<p.geometric_cuts;++i) {
        const auto& cut=p.cuts[i];const auto value=cut.apply(x);
        sparse=std::max(sparse,value[0]+value.tail<2>().norm()-cut.bound);
      }
      assert(std::abs(physical-sparse)<1e-12);
    }
    OptimizerMetrics metrics;p.checkDerivatives(uneven.x,metrics);
    assert(metrics.gradient_error<1e-7&&metrics.hessian_error<1e-7);
    assert(metrics.gradient_reference_error<1e-12&&metrics.hessian_reference_error<1e-12);
  }
}
