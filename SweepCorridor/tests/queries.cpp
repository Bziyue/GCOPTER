#include <SweepCorridor.hpp>
#include <cassert>
namespace sc=SweepCorridor;
template<int D> void check(const sc::Corridor<D>& c){
 sc::Query<D> query(c);
 for(int j=1;j<39;++j){const double u=j/40.;const auto s=c.section(u),q=query.section(u);
  assert((s.center-q.center).norm()<1e-10);assert((s.first-q.first).norm()<1e-8);assert((s.second-q.second).norm()<1e-6);
  assert((s.shape-q.shape).norm()<1e-10);assert((s.shape_first-q.shape_first).norm()<1e-8);assert((s.shape_second-q.shape_second).norm()<1e-6);
  const sc::Vec3 x=s.point(Eigen::Vector2d(.3,.2))+sc::Vec3(.03,-.01,.02);const auto d=c.evaluate(u,x);
  const double h=1e-6;
  for(int k=0;k<4;++k){double a=u,b=u;sc::Vec3 p=x,m=x;if(k==0){a+=h;b-=h;}else {p[k-1]+=h;m[k-1]-=h;}
   const auto plus=c.evaluate(a,p),minus=c.evaluate(b,m);
   const double g=(plus.value-minus.value)/(2*h);assert(std::abs(g-d.gradient[k])<2e-5*(1+std::abs(g)));
   const Eigen::Vector4d hv=(plus.gradient-minus.gradient)/(2*h);assert((hv-d.hessian.col(k)).norm()<5e-4*(1+hv.norm()));
  }
 }
}
void queries(const sc::Corridor<3>& a,const sc::Corridor<5>& b){check(a);check(b);}
