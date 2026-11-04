// SPDX-License-Identifier: MIT
#pragma once
#include "geometry.hpp"
#include "banded_spd.hpp"
#include "runtime.hpp"
#include <cstdint>
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
  double initial_problem_ms=0,initial_pruning_ms=0;
  int construction_reuses=0;
  int basis_cache_hits=0,basis_cache_misses=0,group_lookup_reuses=0;
  double gradient_error=0,hessian_error=0,initial_scale=1;
  double gradient_reference_error=0,hessian_reference_error=0,derivative_min_step=1;
  int steps=0,backtracks=0,damping_retries=0,regularity_backtracks=0;
  int regularity_repairs=0;
  int active_constraints=0;
  double final_decrement=0;
  int controls=0,variables=0,constraints=0,hessian_nonzero=0;
  int merged_supports=0,support_groups=0,pruned_supports=0;
  double pruning_ms=0;
  int bound_line_evaluations=0,bound_line_accepts=0,exact_line_evaluations=0;
  int axis_supports=0;
  std::uint64_t gradient_support_rows=0,hessian_support_rows=0,boundary_support_rows=0,line_support_rows=0;
  struct Pass {int geometric,retained,boundary,progress,steps;double pruning_ms;};
  std::vector<Pass> passes;
  int full_step_accepts=0,feasible_scans=0;
  int energy_builds=0;
  int predictor_steps=0,predictor_rejections=0,volume_steps=0;
  double predictor_ms=0,volume_ms=0,volume_gain=0;
  double minimum_step=1,step_sum=0;
  std::string termination="not_started";
};
// One fixed algorithm configuration. Only explicit diagnostics are selectable.
struct OptimizerSettings {
  static constexpr double center_energy=1e-5,shape_energy=1e-5,offset_weight=.001;
  static constexpr double gradient_tolerance=1e-7;
  static constexpr double initial_radius_fraction=.65,fraction_boundary=.99,progress_threshold=.03;
  static constexpr double volume_trust=.1;
  static constexpr int iterations=8,corrector_iterations=3,inflation_iterations=2;
  bool profile=false,check_gradients=false;
};
struct ProgressConstraints {
  std::vector<int> spans,strict_spans;
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
  std::vector<Cut> cuts; // Progress cones only; geometric rows live in groups.
  struct Support {
    Vec3 normal;double bound;bool redundant=false;
    std::int8_t axis=-1;int original_id=-1;
    Vec3 project(const V9& y)const {
      if(axis>=0){const double n=normal[axis];return Vec3(bound-n*y[axis],n*y[axis+3],n*y[axis+6]);}
      return Vec3(bound-normal.dot(y.head<3>()),normal.dot(y.segment<3>(3)),normal.dot(y.tail<3>()));
    }
    Vec3 projectDirection(const V9& y)const {
      if(axis>=0){const double n=normal[axis];return Vec3(-n*y[axis],n*y[axis+3],n*y[axis+6]);}
      return Vec3(-normal.dot(y.head<3>()),normal.dot(y.segment<3>(3)),normal.dot(y.tail<3>()));
    }
    void addGradient(const Vec3& values,V9& out)const {
      if(axis>=0){for(int a=0;a<3;++a)out[3*a+axis]+=values[a]*normal[axis];}
      else for(int a=0;a<3;++a)out.segment<3>(3*a)+=values[a]*normal;
    }
  };
  struct Group {
    int first,count;
    V9 reference;
    std::array<Mat32,4> maps;
    std::vector<Support> supports;
    // Retained rows form a contiguous prefix. Original IDs stay stable after
    // partitioning so each deletion can be checked against retained witnesses.
    size_t retained=0;
    struct Proof {int removed,left,right;double left_weight,right_weight,upper;};
    V9 direction(const Eigen::VectorXd& x)const;
    V9 geometry(const Eigen::VectorXd& x)const{return reference+direction(x);}
    void accumulateGradient(const V9& gradient,Eigen::VectorXd& result_gradient)const;
    void accumulate(const V9& gradient,const Eigen::Matrix<double,9,9>& hessian,
                    Eigen::VectorXd& result_gradient,BandedSPD& result_hessian)const;
  };
  struct GeometricConstraints {
    std::vector<Group> groups;
    int count=0,retained=0,merged=0,axes=0;
  };
  // Shared across initial feasibility, first solve and progress repairs. It is
  // immutable and rebuilt when model frames, cover or pruning point change.
  std::shared_ptr<const GeometricConstraints> geometry;
  bool reused_construction=false;
  int basis_cache_hits=0,basis_cache_misses=0,group_lookup_reuses=0;
  double pruning_ms=0;
  std::shared_ptr<const CubicEnergy> energy;
  const BandedSPD& quadratic;
  const Eigen::VectorXd& linear;
  CubicProblem(const CubicModel& m,const std::vector<detail::SupportPlane>& cover,
               double length,const Options& o,const OptimizerSettings& s,
               std::shared_ptr<const CubicEnergy> shared_energy={},
               const CubicProblem* geometric_template=nullptr,
               const ProgressConstraints& progress={});
  double objective(const Eigen::VectorXd& x,double mu,Eigen::VectorXd* gradient=nullptr,
                   BandedSPD* hessian=nullptr,bool evaluate_cost=true)const;
  // Independent dense assembly used by explicit diagnostics and tests only.
  double referenceObjective(const Eigen::VectorXd& x,double mu,Eigen::VectorXd* gradient,
                            BandedSPD* hessian)const;
  void checkDerivatives(const Eigen::VectorXd& x,OptimizerMetrics& metrics)const;
  double feasibleStep(const Eigen::VectorXd& x,const Eigen::VectorXd& step)const;
  struct LineModel {
    struct Term {double a,b,weight;};
    std::vector<Term> terms;
    double linear=0,quadratic=0,step=1;
    double delta(double alpha)const;
    double upperBound(double alpha)const;
  };
  LineModel prepareLine(const Eigen::VectorXd& x,const Eigen::VectorXd& step,double mu)const;
  void refineVolume(Eigen::VectorXd& x,OptimizerMetrics& metrics,const std::function<bool(const Eigen::VectorXd&)>& certify)const;
  void optimize(Eigen::VectorXd& x,OptimizerMetrics& metrics,const Deadline& deadline)const;
};
void pruneProjectedSupports(CubicProblem::Group& group,const Eigen::VectorXd& x,
                            double max_offset,double max_radius,
                            std::vector<CubicProblem::Group::Proof>* proofs=nullptr);
}
