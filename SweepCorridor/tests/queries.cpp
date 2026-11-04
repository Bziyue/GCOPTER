#include <SweepCorridor/InverseQuery.hpp>
#include <cassert>
namespace sc=SweepCorridor;
template<int D> void check(const sc::Corridor<D>& c){
 sc::Query<D> query(c); sc::InverseQuery<D> inverse(c);
 for(int j=1;j<39;++j){const double u=j/40.;const auto s=c.section(u),q=query.section(u);
  assert((s.center-q.center).norm()<1e-10);assert((s.first-q.first).norm()<1e-8);assert((s.second-q.second).norm()<1e-6);
  assert((s.shape-q.shape).norm()<1e-10);assert((s.shape_first-q.shape_first).norm()<1e-8);assert((s.shape_second-q.shape_second).norm()<1e-6);
  const sc::Vec3 x=s.point(Eigen::Vector2d(.3,.2))+sc::Vec3(.03,-.01,.02);const auto d=c.evaluate(u,x);
  const auto fast=query.evaluateFirst(u,x);
  assert(std::abs(fast.value-d.value)<1e-9);
  assert((fast.gradient-d.gradient).norm()<1e-7*(1+d.gradient.norm()));
  assert(std::abs(fast.plane_residual-d.plane_residual)<1e-9);
  // Off-boundary points exercise implicit differentiation through inversion.
  const sc::Vec3 outside=s.point(Eigen::Vector2d(1.2,.3));
  const auto inv=inverse(outside,u+.001);
  assert(inv.valid&&std::abs(inv.u-u)<1e-7&&inv.radial>0);
  // Independent de Casteljau/LDLT reference, with a tighter Newton tolerance.
  auto reference=[&](const sc::Vec3& p) {
    double t=u;
    for(int it=0;it<30;++it){auto a=c.section(t);
      sc::Vec3 n=a.shape.col(0).cross(a.shape.col(1));
      sc::Vec3 nd=a.shape_first.col(0).cross(a.shape.col(1))+a.shape.col(0).cross(a.shape_first.col(1));
      double f=n.dot(p-a.center),fd=nd.dot(p-a.center)-n.dot(a.first);
      if(std::abs(f)<1e-13)break;t-=f/fd;
    }
    auto a=c.section(t);sc::Vec3 delta=p-a.center;
    Eigen::Vector2d z=(a.shape.transpose()*a.shape).ldlt().solve(a.shape.transpose()*delta);
    sc::Vec3 normal=a.shape.col(0).cross(a.shape.col(1)).normalized();
    return (delta-normal*normal.dot(delta)).norm()*(1-1/z.norm());
  };
  assert(std::abs(reference(outside)-inv.radial)<1e-7);
  for(int k=0;k<3;++k){sc::Vec3 p=outside,m=outside;p[k]+=1e-5;m[k]-=1e-5;
    double fd=(reference(p)-reference(m))/2e-5;
    assert(std::abs(fd-inv.grad_radial[k])<2e-5*(1+std::abs(fd)));
  }
  auto cached=inverse(outside,u+.001);inverse.reset();auto fresh=inverse(outside,u+.001);
  assert(cached.radial==fresh.radial&&cached.u==fresh.u&&(cached.grad_radial.array()==fresh.grad_radial.array()).all());
  const double h=1e-6;
  for(int k=0;k<4;++k){double a=u,b=u;sc::Vec3 p=x,m=x;if(k==0){a+=h;b-=h;}else {p[k-1]+=h;m[k-1]-=h;}
   const auto plus=c.evaluate(a,p),minus=c.evaluate(b,m);
   const double g=(plus.value-minus.value)/(2*h);assert(std::abs(g-d.gradient[k])<2e-5*(1+std::abs(g)));
   const Eigen::Vector4d hv=(plus.gradient-minus.gradient)/(2*h);assert((hv-d.hessian.col(k)).norm()<5e-4*(1+hv.norm()));
  }
 }
 // Endpoint caps retain a plane residual; cache eviction cannot alter results.
 for(double u:{0.,1.}){auto s=c.section(u);sc::Vec3 n=s.shape.col(0).cross(s.shape.col(1)).normalized();
   sc::Vec3 p=s.center+(u==0?-.2:.2)*n;auto a=inverse(p,u);
   assert(a.valid&&std::abs(std::abs(a.plane)-.2)<1e-9);
   for(int i=0;i<10000;++i){double t=(i+.5)/10000.;inverse(c.section(t).center,t);}
   auto b=inverse(p,u);assert(a.plane==b.plane&&a.u==b.u);
 }
}
void queries(const sc::Corridor<3>& a,const sc::Corridor<5>& b){check(a);check(b);}
