#include "SplineTrajectory/SplineOptimizer.hpp"
#include "SplineTrajectory/SplineConvexHull.hpp"
#include <iostream>
#include <stdexcept>

void require(bool good, const char* message) { if (!good) throw std::runtime_error(message); }

template<class Spline> void check() {
  SplineTrajectory::SplineOptimizer<Spline> optimizer;
  typename decltype(optimizer)::Problem problem;
  problem.durations = {1.1, 1.4, 0.9};
  problem.waypoints.resize(4,3);
  problem.waypoints << 0,0,1, 1,0.8,1.2, 2,-0.4,1.6, 3,0,1;
  require(bool(optimizer.prepare(problem)), "prepare");
  SplineTrajectory::ConvexHullWorkspace<3> hull;
  hull.configure(3, Spline::kCoefficientCount, {SplineTrajectory::ConvexHullBasis::Bezier, 0, 2});
  typename Spline::CoefficientMatrix cp_gradient(hull.controls().rows(),3);
  auto hull_cost = [&](const SplineTrajectory::ParameterView<Spline>& p,
                          SplineTrajectory::CoefficientGradient<Spline>& g) {
    hull.update(p.polynomial);
    const int count = hull.controls().rows()/3;
    double cost = 0;
    for(int r=0;r<hull.controls().rows();++r) {
      const int seg=r/count;
      const double density=0.5*hull.controls().row(r).squaredNorm()/count;
      cost += p.durations[seg]*density;
      g.durations(seg) += density;
      cp_gradient.row(r)=p.durations[seg]/count*hull.controls().row(r);
    }
    hull.backwardAdd(cp_gradient,g.coefficients,g.durations);
    return cost;
  };
  struct Objective { decltype(hull_cost)& coefficient; } objective{hull_cost};
  auto x=optimizer.initialGuess();
  auto result=SplineTrajectory::checkGradients(optimizer,x,objective);
  std::cout << Spline::kCoefficientCount << " coefficients: gradient error " << result.relative_error << '\n';
  require(result.relative_error<2e-6,"Bézier → coefficient/time → spline adjoint gradient");
  Eigen::VectorXd g(x.size());
  const double good=optimizer.costAndGradient(x,g,objective);
  auto bad=x; bad(0)=std::numeric_limits<double>::infinity();
  require(!std::isfinite(optimizer.costAndGradient(bad,g,objective)),"nonfinite candidate rejected");
  require(std::abs(optimizer.costAndGradient(x,g,objective)-good)<1e-9,"recovery after rejection");
  require(bool(optimizer.updateAccepted(x)),"accepted point");
  auto polynomial=optimizer.copyPolynomial();
  require((polynomial.evaluate(polynomial.startTime())-problem.waypoints.row(0).transpose()).norm()<1e-9,"start endpoint");
}
int main() {
  check<SplineTrajectory::CubicSplineND<3>>();
  check<SplineTrajectory::QuinticSplineND<3>>();
  check<SplineTrajectory::SepticSplineND<3>>();
  std::cout << "PASS: library migration regression\n";
}
