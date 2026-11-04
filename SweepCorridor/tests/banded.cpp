#include <SweepCorridor/detail/BandedSPD.hpp>
#include <Eigen/Cholesky>
#include <cassert>
void banded() {
  using SweepCorridor::internal::cubic::BandedSPD;
  for(int n:{20,41,103}) {
    BandedSPD band(n);Eigen::MatrixXd dense=Eigen::MatrixXd::Zero(n,n);
    for(int i=0;i<n;++i)for(int d=1;d<=std::min(i,BandedSPD::width);++d) {
      double v=.01*std::sin(i*31+d*13);band.data(i,d)=v;dense(i,i-d)=dense(i-d,i)=v;
    }
    for(int i=0;i<n;++i)band.data(i,0)=dense(i,i)=1+dense.row(i).cwiseAbs().sum();
    Eigen::VectorXd rhs=Eigen::VectorXd::LinSpaced(n,-1,2),x;
    for(double damping:{0.,.01,0.}) {
      assert(band.factor(damping));band.solveInto(rhs,x);
      Eigen::MatrixXd a=dense+damping*Eigen::MatrixXd::Identity(n,n);
      assert((a*x-rhs).norm()<1e-12);
      assert((x-a.llt().solve(rhs)).norm()<1e-12);
    }
    // A failed factorization may leave partial data; the retry must overwrite
    // every coefficient it reads without requiring a full zero-fill.
    band.data(0,0)=dense(0,0)=-1;
    assert(!band.factor());assert(band.factor(2));band.solveInto(rhs,x);
    assert(((dense+2*Eigen::MatrixXd::Identity(n,n))*x-rhs).norm()<1e-12);
  }
}
