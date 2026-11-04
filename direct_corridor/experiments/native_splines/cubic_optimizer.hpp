// SPDX-License-Identifier: MIT
#pragma once
#include "geometry.hpp"
#include "banded_spd.hpp"
#include "../../src/generation_pipeline.hpp"
namespace gcopter::direct::native {
struct CubicModel {
  CubicBasis basis;
  std::vector<Vec3> reference;
  std::vector<Mat32> frames;
  Eigen::VectorXd x;
  explicit CubicModel(const std::vector<double>& breaks):basis(breaks),reference(basis.count),frames(basis.count),x(5*basis.count){}
  Geometry geometry(const Eigen::VectorXd& values)const;
  Geometry geometry()const{return geometry(x);}
  Eigen::Matrix<double,9,5> jacobian(int i)const;
  static CubicModel project(const Geometry& natural);
  static CubicModel fromSeed(const std::vector<Vec3>& seed,double spacing,double radius,double frame_smoothing);
};
struct OptimizerMetrics {
  double setup_ms=0,assembly_ms=0,solve_ms=0,line_search_ms=0,certificate_ms=0;
  double gradient_error=0,hessian_error=0,initial_scale=1;
  int steps=0,backtracks=0,damping_retries=0,regularity_backtracks=0;
  int regularity_repairs=0;
  int active_constraints=0;
  double final_decrement=0;
  int controls=0,variables=0,constraints=0,hessian_nonzero=0,equality_rank=0;
  std::string termination="not_started";
};
struct OptimizerSettings {
  double center_energy=1e-4,shape_energy=1e-4,offset_weight=.001;
  double gradient_tolerance=1e-7;
  bool natural_endpoints=false,check_gradients=false;
  bool reference_assembly=false; // Dense oracle for equivalence tests only.
  bool grouped_assembly=true;
  double initial_radius_fraction=0;
  bool recenter_frames=false;
  double fraction_boundary=0;
  int barrier_stages=3;
  bool working_set=false;
  bool polar_pruning=false;
  double progress_threshold=-1;
  int iterations=12;
  int corrector_iterations=0;
  std::vector<int> progress_spans;
};
struct CubicProblem {
  struct Cut {
    int first,count=4;double bound,center_sign=1;
    std::array<Eigen::Vector2d,4> normals;
    Eigen::Matrix<double,3,20> dense()const;
    Vec3 apply(const Eigen::VectorXd& x)const;
    bool compact();
    void accumulate(const Vec3& v,double mu,double slack,Eigen::VectorXd& gradient,BandedSPD& hessian)const;
  };
  const CubicModel& model;const Options& options;OptimizerSettings settings;
  std::vector<Cut> cuts;
  struct Support {Vec3 normal;double bound;mutable bool active=true;bool redundant=false;};
  struct Group {
    int first,count;
    V9 reference;
    Eigen::Matrix<double,9,20> map;
    std::vector<Support> supports;
  };
  std::vector<Group> groups;
  size_t geometric_cuts=0;
  BandedSPD quadratic;Eigen::VectorXd linear;
  Eigen::MatrixXd equalities;
  CubicProblem(const CubicModel& m,const std::vector<detail::SupportPlane>& cover,
               double length,const Options& o,const OptimizerSettings& s);
  double objective(const Eigen::VectorXd& x,double mu,Eigen::VectorXd* gradient=nullptr,BandedSPD* hessian=nullptr)const;
  void checkDerivatives(const Eigen::VectorXd& x,OptimizerMetrics& metrics)const;
  double feasibleStep(const Eigen::VectorXd& x,const Eigen::VectorXd& step)const;
  int activateSupports(const Eigen::VectorXd& x)const;
  void optimize(Eigen::VectorXd& x,OptimizerMetrics& metrics,const std::function<void()>& stop)const;
};
}
