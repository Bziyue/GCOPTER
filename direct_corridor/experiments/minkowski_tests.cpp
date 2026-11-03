// SPDX-License-Identifier: MIT
#include "minkowski_optimizer.hpp"
#include "minkowski_affine_shape.hpp"
#include <iostream>
#include <random>
#include <stdexcept>
using namespace gcopter::direct;
namespace mk=gcopter::direct::minkowski;
void check(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
void audit(const mk::Corridor& c,const ObstacleIndex& map,const Bounds& b,double pad) {
  for(int j=0;j<=80;++j) for(int k=0;k<48;++k) {
    const double u=j/80., angle=2*std::acos(-1.)*k/48;
    const auto boundary=c.boundary(u,angle);
    const auto s=c.section(u);
    check(std::abs(c.evaluate(u,boundary.point).value)<2e-8,"boundary gauge");
    for(double radius:{0.,.5,1.}) {
      const Vec3 p=s.center+radius*(boundary.point-s.center);
      check((p.array()>=b.lower.array()+pad-1e-8).all() &&
            (p.array()<=b.upper.array()-pad+1e-8).all(),"world bounds");
      for(const auto& obstacle:map.obstacles())
        check((p-obstacle.center).norm()>obstacle.radius+pad-1e-8,"independent sphere collision");
    }
  }
  for(double u:{.137,.537}) {
    const auto s=c.section(u);
    const Vec3 p=s.center+s.frame*Eigen::Vector2d(.2,.1);
    const auto value=c.evaluate(u,p);
    const double eps=1e-6;
    for(int k=0;k<4;++k) {
      Vec3 a=p,b=p; double ua=u,ub=u;
      if(k==0) { ua+=eps; ub-=eps; } else { a[k-1]+=eps; b[k-1]-=eps; }
      const auto va=c.evaluate(ua,a), vb=c.evaluate(ub,b);
      check(std::abs((va.value-vb.value)/(2*eps)-value.gradient[k])<2e-4*(1+std::abs(value.gradient[k])),"gauge gradient");
      check(((va.gradient-vb.gradient)/(2*eps)-value.hessian.col(k)).norm()<2e-3*(1+value.hessian.col(k).norm()),"gauge Hessian");
    }
    const auto at_center=c.evaluate(u,s.center);
    check(at_center.value==-1 && at_center.gradient.norm()==0 && at_center.hessian.norm()==0,"C2 gauge center");
    const double theta=.37;
    const auto v=c.boundary(u,theta), a=c.boundary(u+eps,theta), b=c.boundary(u-eps,theta),
               l=c.boundary(u,theta-eps),r=c.boundary(u,theta+eps);
    check(((a.point-b.point)/(2*eps)-v.first).norm()<1e-4*(1+v.first.norm()),"boundary first");
    check(((a.first-b.first)/(2*eps)-v.second).norm()<1e-3*(1+v.second.norm()),"boundary second");
    check(((r.point-l.point)/(2*eps)-v.angle_first).norm()<1e-5*(1+v.angle_first.norm()),"angle derivative");
    check(((r.angle_first-l.angle_first)/(2*eps)-v.angle_second).norm()<1e-4*(1+v.angle_second.norm()),"angle Hessian");
    check(((r.first-l.first)/(2*eps)-v.mixed).norm()<1e-4*(1+v.mixed.norm()),"mixed derivative");
  }
  for(std::size_t i=1;i+1<c.knots().size();++i) {
    const double u=c.knots()[i], eps=1e-8;
    const auto a=c.boundary(u-eps,.37),b=c.boundary(u+eps,.37);
    check((a.point-b.point).norm()<1e-5,"C0 join");
    check((a.first-b.first).norm()<1e-4,"C1 join");
    check((a.second-b.second).norm()<.2,"C2 join");
  }
}
int main() {
  try {
    mk::Options o;
    o.geometry.robot_radius=0; o.geometry.safety_margin=0;
    o.geometry.certificate_margin=1e-8;
    o.geometry.max_radius=5; o.geometry.max_offset=3;
    o.geometry.budget_ms=10000;
    for(int mode=0;mode<6;++mode) {
      const int count=mode%2==0?2:4;
      o.templates=count;
      o.affine=mode==2 || mode==3;
      o.adapt_orientation=mode>=4;
      o.product_bounds=mode==5;
      const ObstacleIndex empty({});
      const Bounds square{Vec3(-5,-1,-1),Vec3(5,1,1)};
      auto straight=mk::generate({Vec3(-2,0,0),Vec3(2,0,0)},empty,square,o);
      check(bool(straight),straight.message.c_str());
      check(straight.corridor.area(.5)/4>(o.affine && count==4 ?.82:.975),"square area gain");
      audit(straight.corridor,empty,square,0);
      const Bounds world{Vec3(-5,-4,-4),Vec3(5,4,4)};
      const ObstacleIndex map({{Vec3(0,1,0),.3},{Vec3(1,-.7,.4),.2},{Vec3(-1,-1,.8),.15}});
      auto curve=mk::generate({Vec3(-2,0,0),Vec3(0,0,.1),Vec3(2,.3,.5)},map,world,o);
      check(bool(curve),curve.message.c_str());
      audit(curve.corridor,map,world,0);
      std::cout<<"mode "<<mode<<", "<<count<<" templates: square="<<straight.corridor.area(.5)/4
               <<" curve="<<curve.metrics.total_ms<<" ms\n";
    }
    o.affine=false; o.adapt_orientation=false; o.product_bounds=false;
    o.adapt_orientation=true; o.orientation_smoothing=4;
    const auto smooth=mk::generate({Vec3(-2,0,0),Vec3(0,.3,.1),Vec3(2,.4,.5)},
        ObstacleIndex({{Vec3(0,1,0),.3}}),{Vec3(-5,-4,-4),Vec3(5,4,4)},o);
    check(bool(smooth),smooth.message.c_str());
    audit(smooth.corridor,ObstacleIndex({{Vec3(0,1,0),.3}}),{Vec3(-5,-4,-4),Vec3(5,4,4)},0);
    o.adapt_orientation=false; o.orientation_smoothing=0;
    // Continuous support bound: independent dense samples on random frames,
    // asymmetric weights, centers and subintervals.
    std::mt19937_64 random(41);
    std::uniform_real_distribution<double> uniform(-1,1);
    mk::Templates bank(o);
    for(int repeat=0;repeat<80;++repeat) {
      std::vector<detail::Node> nodes(2);
      std::vector<mk::V6> x(2);
      for(int i=0;i<2;++i) {
        nodes[i].s=i; nodes[i].reference=Vec3(i,0,0); nodes[i].velocity=Vec3::UnitX();
        nodes[i].frame=Eigen::AngleAxisd(uniform(random),Vec3(1,2,3).normalized()).toRotationMatrix().leftCols<2>();
        for(int j=0;j<6;++j) x[i][j]=j<2 ?.2*uniform(random):.4+.3*uniform(random);
      }
      bank.proxy(nodes,x);
      auto p=detail::patch(nodes,0);
      if(repeat%2) p=detail::split(detail::split(p,.8).first,.25).second;
      const Vec3 n=Vec3(uniform(random),uniform(random),uniform(random)).normalized();
      const double bound=mk::supportBound(p,nodes,x,bank,n);
      const double product=mk::productSupportBound(p,nodes,x,bank,n);
      check(product<=bound+1e-12,"product tightens frame bound");
      for(int j=0;j<=100;++j) {
        const double t=j/100.; auto ref=p.reference; auto w=p.w;
        for(int k=5;k>0;--k) for(int l=0;l<k;++l) {ref[l]=(1-t)*ref[l]+t*ref[l+1];w[l]=(1-t)*w[l]+t*w[l+1];}
        const Mat32 frame=(1-w[0])*nodes[0].frame+w[0]*nodes[1].frame;
        const Vec3 c=ref[0]+(1-w[0])*nodes[0].frame*x[0].head<2>()+w[0]*nodes[1].frame*x[1].head<2>();
        const mk::Weights weights=(1-w[0])*x[0].tail<4>()+w[0]*x[1].tail<4>();
        check(n.dot(c)+weights.dot(bank.support(frame.transpose()*n))<=product+1e-12,"whole-interval product support bound");
      }
    }
    // Optimizer objective gradients and both neighboring Hessian blocks.
    std::vector<detail::Node> nodes(3);
    std::vector<mk::V6> x(3);
    for(int i=0;i<3;++i) {nodes[i].s=i; x[i]<<.03,-.02,.3,.2,.25,.17;}
    std::vector<mk::LinearCut> cuts(2);
    for(int i=0;i<2;++i) {cuts[i].span=i; cuts[i].a<<.1,.2,.3,.2,.2,.3;cuts[i].b=.5*cuts[i].a;cuts[i].h=2;}
    detail::Stop stop=[]{};
    mk::Optimizer solver(o,bank,stop,nodes);
    solver.objective(x,cuts,.01,true);
    const auto gradient=solver.gradient; const auto diagonal=solver.diagonal; const auto upper=solver.upper;
    const double eps=1e-6;
    for(int i=0;i<3;++i) for(int j=0;j<6;++j) {
      auto a=x,b=x; a[i][j]+=eps; b[i][j]-=eps;
      const double va=solver.objective(a,cuts,.01,true); const auto ga=solver.gradient;
      const double vb=solver.objective(b,cuts,.01,true); const auto gb=solver.gradient;
      check(std::abs((va-vb)/(2*eps)-gradient[i][j])<1e-6,"optimizer gradient");
      for(int k=0;k<3;++k) {
        mk::V6 expected=mk::V6::Zero();
        if(k==i) expected=diagonal[k].col(j);
        if(k==i-1) expected=upper[k].col(j);
        if(k==i+1) expected=upper[i].row(j).transpose();
        check(((ga[k]-gb[k])/(2*eps)-expected).norm()<1e-5,"optimizer Hessian");
      }
    }
    std::cout<<"PASS: Minkowski geometry, continuous bounds, derivatives and optimizer\n";
  } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
