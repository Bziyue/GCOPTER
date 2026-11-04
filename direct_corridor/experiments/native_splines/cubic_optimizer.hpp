// SPDX-License-Identifier: MIT
#pragma once
#include "geometry.hpp"
#include "banded_spd.hpp"
#include "../../src/generation_pipeline.hpp"
#include <memory>
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
  static CubicModel fromSeed(const std::vector<Vec3>& seed,double spacing,double radius,double frame_smoothing,
                             const std::vector<double>& requested_breaks={});
};
struct OptimizerMetrics {
  double setup_ms=0,assembly_ms=0,solve_ms=0,line_search_ms=0,certificate_ms=0;
  double gradient_error=0,hessian_error=0,initial_scale=1;
  int steps=0,backtracks=0,damping_retries=0,regularity_backtracks=0;
  int regularity_repairs=0;
  int active_constraints=0;
  double final_decrement=0;
  int controls=0,variables=0,constraints=0,hessian_nonzero=0,equality_rank=0;
  int merged_supports=0,support_groups=0,pruned_supports=0;
  int full_step_accepts=0,feasible_scans=0;
  int warm_reserves=0;
  int energy_builds=0;
  int exchange_steps=0,exchange_witnesses=0,oracle_nodes=0,oracle_unresolved=0;
  double oracle_ms=0;
  int predictor_steps=0,predictor_rejections=0,volume_steps=0;
  double predictor_ms=0,volume_ms=0,volume_gain=0;
  double minimum_step=1,step_sum=0;
  double coordinate_ms=0;
  int coordinate_updates=0,coordinate_rejections=0;
  std::string termination="not_started";
};
struct OptimizerSettings {
  double center_energy=1e-4,shape_energy=1e-4,offset_weight=.001;
  double gradient_tolerance=1e-7;
  bool natural_endpoints=false,check_gradients=false;
  bool reference_assembly=false; // Dense oracle for equivalence tests only.
  bool grouped_assembly=true;
  bool merge_shared_supports=true;
  bool tensor_assembly=true,trial_first=false;
  bool cached_line=true;
  bool reuse_energy=true,cache_values=false;
  double separator_reuse_gap=-1;
  int support_exchange=0,exchange_rounds=4,exchange_iterations=4;
  bool lazy_priorities=true;
  int adaptive_knots=0;
  double path_predictor=0,volume_trust=.1;
  int volume_refinements=0;
  int coordinate_sweeps=0;
  int barrier_begin=0,warm_start_stage=0;
  double warm_reserve=0;
  double initial_radius_fraction=0;
  bool recenter_frames=false;
  double fraction_boundary=0;
  int barrier_stages=3;
  bool working_set=false;
  bool polar_pruning=false;
  double progress_threshold=-1;
  int progress_samples=3,progress_depth=1;
  int iterations=12;
  int corrector_iterations=0;
  std::vector<int> progress_spans;
  std::vector<int> strict_progress_spans;
};
struct CubicEnergy {
  BandedSPD quadratic;
  Eigen::VectorXd linear;
  CubicEnergy(const CubicModel& model,double length,const OptimizerSettings& settings);
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
    std::array<Mat32,4> maps;
    std::vector<Support> supports;
    // Kept out of Support so value/step scans do not load Hessian-only data.
    std::vector<Eigen::Matrix<double,6,1>> normal_products;
    V9 direction(const Eigen::VectorXd& x)const;
    V9 geometry(const Eigen::VectorXd& x)const{return reference+direction(x);}
    void accumulate(const V9& gradient,const Eigen::Matrix<double,9,9>& hessian,
                    Eigen::VectorXd& result_gradient,BandedSPD& result_hessian)const;
  };
  std::vector<Group> groups;
  size_t geometric_cuts=0;
  int merged_supports=0;
  std::shared_ptr<const CubicEnergy> energy;
  const BandedSPD& quadratic;
  const Eigen::VectorXd& linear;
  Eigen::MatrixXd equalities;
  CubicProblem(const CubicModel& m,const std::vector<detail::SupportPlane>& cover,
               double length,const Options& o,const OptimizerSettings& s,
               std::shared_ptr<const CubicEnergy> shared_energy={});
  struct Evaluation {
    Eigen::VectorXd energy_gradient;
    std::vector<Vec3> cones;
  };
  double objective(const Eigen::VectorXd& x,double mu,Eigen::VectorXd* gradient=nullptr,
                   BandedSPD* hessian=nullptr,Evaluation* evaluation=nullptr)const;
  void checkDerivatives(const Eigen::VectorXd& x,OptimizerMetrics& metrics)const;
  double feasibleStep(const Eigen::VectorXd& x,const Eigen::VectorXd& step)const;
  struct LineModel {
    struct Term {double a,b,weight;};
    std::vector<Term> terms;
    double linear=0,quadratic=0,step=1;
    double delta(double alpha)const;
  };
  LineModel prepareLine(const Eigen::VectorXd& x,const Eigen::VectorXd& step,double mu,
                        const Evaluation* evaluation=nullptr)const;
  void inflate(Eigen::VectorXd& x,OptimizerMetrics& metrics)const;
  void refineVolume(Eigen::VectorXd& x,OptimizerMetrics& metrics,const std::function<bool(const Eigen::VectorXd&)>& certify)const;
  int activateSupports(const Eigen::VectorXd& x)const;
  void optimize(Eigen::VectorXd& x,OptimizerMetrics& metrics,const std::function<void()>& stop)const;
};
}
