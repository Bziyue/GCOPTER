// SPDX-License-Identifier: MIT
#pragma once
#include "GeometryData.hpp"
#include "CubicMetrics.hpp"
#include "Types.hpp"
#include "BandedSPD.hpp"
#include "Runtime.hpp"
#include <cstdint>
#include <memory>
namespace SweepCorridor::internal::cubic {
struct CubicModel {
  CubicBasis basis;
  std::vector<Vec3> reference;
  std::vector<Mat32> frames;
  Eigen::VectorXd x;
  mutable std::array<double,2> anchor_parameters{{0.,1.}};
  explicit CubicModel(const std::vector<double>& breaks):basis(breaks),reference(basis.count),frames(basis.count),x(6*basis.count){}
  Geometry geometry(const Eigen::VectorXd& values)const;
  Geometry geometry()const{return geometry(x);}
  Eigen::Matrix<double,9,6> jacobian(int i)const;
  static CubicModel fromSeed(const std::vector<Vec3>& seed,double spacing,double radius,double frame_smoothing,
                             const std::vector<double>& requested_breaks={});
};
struct ProgressConstraints {
  std::vector<int> spans;
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
    std::array<Vec3,4> center_normals;
    Vec3 apply(const Eigen::VectorXd& x)const;
    bool compact();
    void accumulate(const Vec3& v,double mu,double slack,Eigen::VectorXd& gradient,BandedSPD& hessian)const;
  };
  const CubicModel& model;const GeometryOptions& options;OptimizerSettings settings;
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
    std::array<double,4> center_weights;
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
               double length,const GeometryOptions& o,const OptimizerSettings& s,
               std::shared_ptr<const CubicEnergy> shared_energy={},
               const CubicProblem* geometric_template=nullptr,
               const ProgressConstraints& progress={});
  double objective(const Eigen::VectorXd& x,double mu,Eigen::VectorXd* gradient=nullptr,
                   BandedSPD* hessian=nullptr,bool evaluate_cost=true)const;
  struct LineModel {
    struct Term {double a,b,weight;};
    std::vector<Term> terms;
    double linear=0,quadratic=0,step=1;
    double delta(double alpha)const;
    double upperBound(double alpha)const;
  };
  LineModel prepareLine(const Eigen::VectorXd& x,const Eigen::VectorXd& step,double mu)const;
  void optimize(Eigen::VectorXd& x,OptimizerMetrics& metrics,const Deadline& deadline)const;
};
void pruneProjectedSupports(CubicProblem::Group& group,const Eigen::VectorXd& x,
                            double max_offset,double max_radius,
                            std::vector<CubicProblem::Group::Proof>* proofs=nullptr);
}
