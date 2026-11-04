// SPDX-License-Identifier: MIT
#pragma once
#include "SweepCorridor/detail/QuinticGeneratorImpl.hpp"
#include "SweepCorridor/detail/GeometryImpl.hpp"
#include "SweepCorridor/detail/CubicOptimizerImpl.hpp"
#include "SweepCorridor/detail/CubicRefinement.hpp"
#include "SweepCorridor/detail/SupportPruning.hpp"
#include "SweepCorridor/detail/CubicGeneratorImpl.hpp"
#include <utility>

namespace SweepCorridor {
using Vec3 = internal::Vec3;
using Mat32 = internal::Mat32;
using Sphere = internal::Sphere;
using Bounds = internal::Bounds;
using Section = internal::Section;
using Differential = internal::Differential;
using ObstacleIndex = internal::ObstacleIndex;
using SupportPlane = internal::detail::SupportPlane;
using Control = Eigen::Matrix<double,9,1>;

struct Options : internal::GeometryOptions {
  Options() { control_spacing=.4;min_radius=.03;safety_margin=.05;
              local_range=3;frame_smoothing=1.5; }
  bool profile=false;
};
enum class Status { Success, InvalidInput, SeedCollision, InsufficientClearance,
  DegeneratePath, Capacity, Timeout, VerificationFailed, InitializationFailed,
  NumericalFailure };
inline const char* statusName(Status s) {
  switch(s) {
#define SWEEP_STATUS(x,y) case Status::x:return y;
    SWEEP_STATUS(Success,"SUCCESS") SWEEP_STATUS(InvalidInput,"INVALID_INPUT")
    SWEEP_STATUS(SeedCollision,"SEED_COLLISION") SWEEP_STATUS(InsufficientClearance,"INSUFFICIENT_CLEARANCE")
    SWEEP_STATUS(DegeneratePath,"DEGENERATE_PATH") SWEEP_STATUS(Capacity,"CAPACITY")
    SWEEP_STATUS(Timeout,"TIMEOUT") SWEEP_STATUS(VerificationFailed,"VERIFICATION_FAILED")
    SWEEP_STATUS(InitializationFailed,"INITIALIZATION_FAILED") SWEEP_STATUS(NumericalFailure,"NUMERICAL_FAILURE")
#undef SWEEP_STATUS
  }
  return "NUMERICAL_FAILURE";
}
namespace detail {
inline Status status(const std::string& name) {
  for(auto s:{Status::Success,Status::InvalidInput,Status::SeedCollision,Status::InsufficientClearance,
      Status::DegeneratePath,Status::Capacity,Status::Timeout,Status::VerificationFailed,
      Status::InitializationFailed})if(name==statusName(s))return s;
  return Status::NumericalFailure;
}
// Shared exact differential of the affine-disc section, with world and u jets.
inline Differential differential(const Section& s,const Vec3& x) {
  const auto& m = s.shape;
  const auto& d = s.shape_first;
  const auto& dd = s.shape_second;
  const Eigen::Matrix2d g = m.transpose() * m;
  const Eigen::Matrix2d gd = d.transpose() * m + m.transpose() * d;
  const Eigen::Matrix2d gdd =
      dd.transpose() * m + 2 * d.transpose() * d + m.transpose() * dd;
  const Eigen::Matrix2d inverse = g.inverse();
  const Eigen::Matrix2d id = -inverse * gd * inverse;
  const Eigen::Matrix2d idd =
      2 * inverse * gd * inverse * gd * inverse - inverse * gdd * inverse;
  const Eigen::Matrix<double, 2, 3> p = inverse * m.transpose();
  const Eigen::Matrix<double, 2, 3> pd =
      id * m.transpose() + inverse * d.transpose();
  const Eigen::Matrix<double, 2, 3> pdd =
      idd * m.transpose() + 2 * id * d.transpose() + inverse * dd.transpose();
  const Vec3 r = x - s.center;
  const Eigen::Vector2d z = p * r, zu = pd * r - p * s.first;
  const Eigen::Vector2d zuu = pdd * r - 2 * pd * s.first - p * s.second;
  Differential out;
  out.value = z.squaredNorm() - 1;
  out.gradient[0] = 2 * z.dot(zu);
  out.gradient.tail<3>() = 2 * p.transpose() * z;
  out.hessian(0, 0) = 2 * (zu.squaredNorm() + z.dot(zuu));
  out.hessian.block<3, 1>(1, 0) = 2 * (pd.transpose() * z + p.transpose() * zu);
  out.hessian.block<1, 3>(0, 1) = out.hessian.block<3, 1>(1, 0).transpose();
  out.hessian.bottomRightCorner<3, 3>() = 2 * p.transpose() * p;
  out.plane_residual = r.dot(m.col(0).cross(m.col(1)).normalized());
  return out;
}
} // namespace detail
template<int Degree> class Generator;
// Immutable after generation. Query preconditions: nonempty corridor, finite
// u in [0,1], finite x. No repeated input scans in numerical evaluation.
template<int Degree> class Corridor {
  static_assert(Degree==3||Degree==5,"SweepCorridor supports degrees 3 and 5");
  std::vector<double> knots_;
  std::vector<Control> controls_;
  friend class Generator<Degree>;
public:
  static constexpr int degree=Degree;
  std::size_t pieces()const{return knots_.empty()?0:knots_.size()-1;}
  bool empty()const{return pieces()==0;}
  const std::vector<double>& knots()const{return knots_;}
  const std::vector<Control>& controls()const{return controls_;}
  Section section(double u)const {
    const auto i=std::min(pieces()-1,std::size_t(std::upper_bound(knots_.begin(),knots_.end(),u)-knots_.begin()-1));
    const double h=knots_[i+1]-knots_[i];
    internal::cubic::Curve<Degree,Control> p;
    std::copy_n(controls_.begin()+(Degree+1)*i,Degree+1,p.begin());
    return internal::cubic::evaluate<Degree>(p,(u-knots_[i])/h,h);
  }
  Differential evaluate(double u,const Vec3& x)const{return detail::differential(section(u),x);}
};
// Optional power-basis cache. Owns coefficients, so no borrowed lifetime or
// mutable state in Corridor. Use a separate Query per evaluation thread.
template<int Degree> class Query {
  struct Piece {std::array<Control,Degree+1> power;double inverse_width;};
  std::vector<double> knots_;
  std::vector<Piece> pieces_;
  mutable std::size_t cursor_=0;
public:
  explicit Query(const Corridor<Degree>& corridor):knots_(corridor.knots()),pieces_(corridor.pieces()) {
    if(corridor.empty())throw std::invalid_argument("empty corridor query");
    for(std::size_t i=0;i<pieces_.size();++i) {
      auto& p=pieces_[i];p.inverse_width=1/(knots_[i+1]-knots_[i]);
      for(int k=0;k<=Degree;++k) {
        p.power[k].setZero();
        for(int j=0;j<=k;++j)p.power[k]+=internal::cubic::binomial(Degree,k)*
          internal::cubic::binomial(k,j)*((k-j)%2?-1.:1.)*corridor.controls()[(Degree+1)*i+j];
      }
    }
  }
  Section section(double u)const {
    if(!(u>=knots_[cursor_]&&(u<knots_[cursor_+1]||(u==1&&cursor_+1==pieces_.size()))))
      cursor_=std::min(pieces_.size()-1,std::size_t(std::upper_bound(knots_.begin(),knots_.end(),u)-knots_.begin()-1));
    const auto& p=pieces_[cursor_];const double t=(u-knots_[cursor_])*p.inverse_width;
    Control y=p.power[Degree],d=Degree*p.power[Degree],dd=Degree*(Degree-1)*p.power[Degree];
    for(int k=Degree-1;k>=0;--k)y=y*t+p.power[k];
    for(int k=Degree-1;k>=1;--k)d=d*t+k*p.power[k];
    for(int k=Degree-1;k>=2;--k)dd=dd*t+k*(k-1)*p.power[k];
    d*=p.inverse_width;dd*=p.inverse_width*p.inverse_width;
    return {y.head<3>(),d.head<3>(),dd.head<3>(),internal::cubic::shape(y),internal::cubic::shape(d),internal::cubic::shape(dd)};
  }
  Differential evaluate(double u,const Vec3& x)const{return detail::differential(section(u),x);}
};
template<int Degree> struct Metrics {
  double total_ms=0;
  std::size_t controls=0,constraints=0;
  int iterations=0,refinements=0;
  using Solver=std::conditional_t<Degree==3,internal::cubic::OptimizerMetrics,internal::Metrics>;
  Solver solver;
};
template<int Degree> struct Result {
  Status status=Status::InvalidInput;
  std::string message;
  Corridor<Degree> corridor;
  std::vector<SupportPlane> support_planes;
  Metrics<Degree> metrics;
  explicit operator bool()const{return status==Status::Success;}
};
template<int Degree> class Generator {
  static_assert(Degree==3||Degree==5,"SweepCorridor supports degrees 3 and 5");
  Options options_;
public:
  explicit Generator(Options options={}):options_(std::move(options)){}
  Result<Degree> generate(const std::vector<Vec3>& seed,const ObstacleIndex& map,const Bounds& bounds)const {
    Result<Degree> out;
    if constexpr(Degree==3) {
      internal::cubic::OptimizerSettings settings;settings.profile=options_.profile;
      auto r=internal::cubic::generateCubic(seed,map,bounds,internal::cubic::CubicOptions(options_),settings);
      out.status=detail::status(r.status);out.message=std::move(r.message);
      out.metrics.total_ms=r.total_ms;out.metrics.controls=r.optimizer.controls;
      out.metrics.constraints=r.optimizer.constraints;out.metrics.iterations=r.optimizer.steps;
      out.metrics.refinements=r.refinements;out.metrics.solver=std::move(r.optimizer);
      if(out){out.corridor.knots_=std::move(r.corridor.breaks);out.corridor.controls_=std::move(r.corridor.controls);
        out.support_planes=std::move(r.cover);}
    }else {
      internal::Options settings(options_);settings.profile=options_.profile;
      auto r=internal::generate(seed,map,bounds,settings);
      out.status=detail::status(internal::statusName(r.status));out.message=std::move(r.message);
      out.metrics.total_ms=r.metrics.total_ms;out.metrics.controls=r.metrics.nodes;
      out.metrics.constraints=r.metrics.constraints;out.metrics.iterations=r.metrics.newton_steps;
      out.metrics.refinements=r.metrics.refinements;out.metrics.solver=r.metrics;
      if(out){out.corridor.knots_=r.corridor.knots();out.corridor.controls_.reserve(r.corridor.centerControls().size());
        for(std::size_t i=0;i<r.corridor.centerControls().size();++i)
          out.corridor.controls_.push_back(internal::cubic::pack(r.corridor.centerControls()[i],r.corridor.shapeControls()[i]));
        out.support_planes=std::move(r.cover);}
    }
    return out;
  }
};
template<int Degree> inline internal::cubic::Certificate certify(const Corridor<Degree>& corridor,
    const std::vector<SupportPlane>& planes) {
  internal::cubic::Geometry g;g.degree=Degree;g.breaks=corridor.knots();g.controls=corridor.controls();
  return internal::cubic::certify(g,planes);
}
} // namespace SweepCorridor
