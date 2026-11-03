#include "optimizer.hpp"
#include <iostream>
#include <stdexcept>
using namespace uz_experiment;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
double checkGradient(Optimizer& optimizer,Eigen::VectorXd x){
  Eigen::VectorXd analytic(11),numeric(11),scratch(11);optimizer.evaluate(x,analytic);
  for(int j=0;j<x.size();++j){const double base=x(j),h=1e-6;x(j)=base+h;double a=optimizer.evaluate(x,scratch);
    x(j)=base-h;double b=optimizer.evaluate(x,scratch);x(j)=base;numeric(j)=(a-b)/(2*h);}
  double error=(analytic-numeric).norm()/std::max(1.0,analytic.norm());
  std::cout<<"gradient relative error "<<error<<std::endl;require(error<2e-5,"u/z full analytic gradient");return error;
}
int main(){
  ObstacleIndex obstacles({});Bounds bounds{Vec3(-10,-10,-5),Vec3(10,10,5)};
  Options options;options.max_radius=.5;options.max_offset=.1;options.budget_ms=10000;
  const Vec3 start(-2,0,0),end(2,0,0);
  auto line=generate({start,end},obstacles,bounds,options);require(bool(line),"line corridor");
  Optimizer optimizer(line.corridor,start,end,16,false);auto x=optimizer.initial(4);
  x(1)=.2;x(2)=-.3;x(3)=.1;x(6)=-.05;checkGradient(optimizer,x);
  x=optimizer.initial(4);double cost=0;lbfgs::lbfgs_parameter_t params;params.past=3;params.delta=1e-9;params.g_epsilon=0;params.max_iterations=500;
  int status=lbfgs::lbfgs_optimize(x,cost,Optimizer::callback,nullptr,Optimizer::progressCallback,&optimizer,params);
  require(status>=0,"straight trajectory convergence");
  const double exact=std::pow(180*(end-start).squaredNorm(),1.0/6);
  std::cout<<"straight duration "<<std::exp(x(0))<<", exact "<<exact<<", cost "<<cost<<'\n';
  require(std::abs(std::exp(x(0))-exact)<2e-4,"analytic minimum jerk/time solution");
  require(optimizer.maxControlNorm(x)<1,"transverse control disk");
  for(double s:{0.0,1.0}){auto state=optimizer.at(x,s);require(state.v.norm()<1e-10&&state.a.norm()<1e-9,"zero endpoint PVA derivatives");}
  auto curve=generate({Vec3(-3,0,1),Vec3(-1,1,1),Vec3(1,0,1),Vec3(3,1,1)},obstacles,bounds,options);require(bool(curve),"curved corridor");
  Optimizer curved(curve.corridor,Vec3(-3,0,1),Vec3(3,1,1),16,true);
  auto y=curved.initial(3);y(1)=.15;y(2)=-.2;y(3)=.07;y(8)=-.1;
  checkGradient(curved,y);
  Progress progress(y(1),y(2));uint64_t iterations=0;
  for(int k=1;k<100;++k){double u=k/100.;auto [s,jet]=progress.inverse(u,iterations);require(std::abs(progress.at(s)[0]-u)<1e-12,"monotone inverse");}
  double previous=-1;
  for(int k=0;k<=1000;++k){auto state=curved.at(y,k/1000.0);require(state.u>=previous-1e-14&&state.z.norm()<=1,"progress and transverse membership");previous=state.u;}
  std::cout<<"PASS: u/z gradients, containment construction and straight optimum\n";
}
