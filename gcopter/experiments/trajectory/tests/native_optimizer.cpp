#include "gcopter/spline_sfc_optimizer.hpp"
#include <stdexcept>
int main() {
  gcopter::SplineSFCOptimizer optimizer;
  gcopter::PolyhedronH box(6,4);
  box << 1,0,0,-5, -1,0,0,-5, 0,1,0,-3, 0,-1,0,-3, 0,0,1,-4, 0,0,-1,0;
  Eigen::Matrix3d head=Eigen::Matrix3d::Zero(),tail=head;
  head.col(0)<<-2,0,2;tail.col(0)<<2,0,2;
  Eigen::VectorXd bounds(5),weights(5),physical(6);
  bounds<<4,2.1,1.05,2,12;weights<<1e5,1e4,1e4,1e4,1e5;physical<<1,9.81,0.7,0.7,0.01,0.001;
  if(!optimizer.setup(20,head,tail,{box},2,0.01,16,bounds,weights,physical))
    throw std::runtime_error("native setup");
  gcopter::SplineSFCOptimizer::Polynomial polynomial;
  const double cost=optimizer.optimize(polynomial,1e-6);
  if(!std::isfinite(cost)||!polynomial.isValid())throw std::runtime_error("native optimize");
  for(int i=0;i<=1000;++i) {
    auto p=polynomial.evaluate(polynomial.duration()*i/1000.0);
    if((box.leftCols<3>()*p+box.col(3)).maxCoeff()>1e-4)throw std::runtime_error("box violation");
  }
  if((polynomial.evaluate(polynomial.endTime())-tail.col(0)).norm()>1e-7)
    throw std::runtime_error("native end point");
  std::cout<<"PASS: native optimizer, cost="<<cost<<", duration="<<polynomial.duration()<<'\n';
}
