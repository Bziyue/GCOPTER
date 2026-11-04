// SPDX-License-Identifier: MIT
#pragma once
#include "SweepCorridor/detail/ObstacleIndex.hpp"
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
struct FirstSection { Vec3 center,first; Mat32 shape,shape_first; };
struct FirstDifferential {
  double value=0,plane_residual=0;
  Eigen::Vector4d gradient=Eigen::Vector4d::Zero();
};
using Control = Eigen::Matrix<double,9,1>;

struct Options : internal::GeometryOptions {
  Options() { control_spacing=.4;min_radius=.03;safety_margin=.05;
              local_range=3;frame_smoothing=1.5; }
  bool profile=false;
};
using Status = internal::Status;
using internal::statusName;
namespace detail {
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
class Generator;
// Immutable after generation. Query preconditions: nonempty corridor, finite
// u in [0,1], finite x. No repeated input scans in numerical evaluation.
class Corridor {
  std::vector<double> knots_;
  std::vector<Control> controls_;
  friend class Generator;
public:
  static constexpr int degree=3;
  std::size_t pieces()const{return knots_.empty()?0:knots_.size()-1;}
  bool empty()const{return pieces()==0;}
  const std::vector<double>& knots()const{return knots_;}
  const std::vector<Control>& controls()const{return controls_;}
  Section section(double u)const {
    const auto i=std::min(pieces()-1,std::size_t(std::upper_bound(knots_.begin(),knots_.end(),u)-knots_.begin()-1));
    const double h=knots_[i+1]-knots_[i];
    internal::cubic::Curve<3,Control> p;
    std::copy_n(controls_.begin()+(4)*i,4,p.begin());
    return internal::cubic::evaluate<3>(p,(u-knots_[i])/h,h);
  }
  Differential evaluate(double u,const Vec3& x)const{return detail::differential(section(u),x);}
};
// Optional power-basis cache. Owns coefficients, so no borrowed lifetime or
// mutable state in Corridor. Use a separate Query per evaluation thread.
class Query {
  struct Piece {std::array<Control,4> power;double inverse_width;};
  std::vector<double> knots_;
  std::vector<Piece> pieces_;
  mutable std::size_t cursor_=0;
  bool uniform_=true;
  std::size_t locate(double u)const {
    if(!(u>=knots_[cursor_]&&(u<knots_[cursor_+1]||(u==1&&cursor_+1==pieces_.size())))) {
      if(uniform_) {
        cursor_=std::min(pieces_.size()-1,std::size_t(u*pieces_.size()));
        if(u<knots_[cursor_])--cursor_;
        else if(cursor_+1<pieces_.size()&&u>=knots_[cursor_+1])++cursor_;
      } else cursor_=std::min(pieces_.size()-1,std::size_t(std::upper_bound(knots_.begin(),knots_.end(),u)-knots_.begin()-1));
    }
    return cursor_;
  }
public:
  explicit Query(const Corridor& corridor):knots_(corridor.knots()),pieces_(corridor.pieces()) {
    if(corridor.empty())throw std::invalid_argument("empty corridor query");
    for(std::size_t i=0;i<pieces_.size();++i) {
      uniform_ &= std::abs(knots_[i]-double(i)/pieces_.size())<1e-12;
      auto& p=pieces_[i];p.inverse_width=1/(knots_[i+1]-knots_[i]);
      const auto* b=corridor.controls().data()+4*i;
      p.power[0]=b[0];
      p.power[1]=-3*b[0];p.power[1]+=3*b[1];
      p.power[2]=3*b[0];p.power[2]-=6*b[1];p.power[2]+=3*b[2];
      p.power[3]=-b[0];p.power[3]+=3*b[1];p.power[3]-=3*b[2];p.power[3]+=b[3];
    }
  }
  Section section(double u)const {
    locate(u);
    const auto& p=pieces_[cursor_];const double t=(u-knots_[cursor_])*p.inverse_width;
    const Control y=((p.power[3]*t+p.power[2])*t+p.power[1])*t+p.power[0];
    const Control d=((3*p.power[3]*t+2*p.power[2])*t+p.power[1])*p.inverse_width;
    const Control dd=(6*p.power[3]*t+2*p.power[2])*(p.inverse_width*p.inverse_width);
    return {y.head<3>(),d.head<3>(),dd.head<3>(),internal::cubic::shape(y),internal::cubic::shape(d),internal::cubic::shape(dd)};
  }
  const std::vector<double>& knots()const{return knots_;}
  FirstSection first(double u)const {
    const auto i=locate(u);const auto& p=pieces_[i];
    const double t=(u-knots_[i])*p.inverse_width;
    const Control y=((p.power[3]*t+p.power[2])*t+p.power[1])*t+p.power[0];
    const Control d=((3*p.power[3]*t+2*p.power[2])*t+p.power[1])*p.inverse_width;
    return {y.head<3>(),d.head<3>(),internal::cubic::shape(y),internal::cubic::shape(d)};
  }
  FirstDifferential evaluateFirst(double u,const Vec3& x)const {
    const auto s=first(u);const auto& m=s.shape;const auto& md=s.shape_first;
    const Eigen::Matrix2d inverse=(m.transpose()*m).inverse();
    const Eigen::Matrix<double,2,3> dual=inverse*m.transpose();
    const Vec3 r=x-s.center;const Eigen::Vector2d z=dual*r;
    const Eigen::Vector2d zu=inverse*(md.transpose()*r-m.transpose()*s.first-
        (md.transpose()*m+m.transpose()*md)*z);
    FirstDifferential out;out.value=z.squaredNorm()-1;
    out.gradient[0]=2*z.dot(zu);out.gradient.tail<3>()=2*dual.transpose()*z;
    out.plane_residual=r.dot(m.col(0).cross(m.col(1)).normalized());return out;
  }
  Differential evaluate(double u,const Vec3& x)const{return detail::differential(section(u),x);}
};
struct Metrics {
  double total_ms=0;
  std::size_t controls=0,constraints=0;
  int iterations=0,refinements=0;
  using Solver=internal::cubic::OptimizerMetrics;
  Solver solver;
};
struct Result {
  Status status=Status::InvalidInput;
  std::string message;
  Corridor corridor;
  std::vector<SupportPlane> support_planes;
  Metrics metrics;
  explicit operator bool()const{return status==Status::Success;}
};
class Generator {
  Options options_;
public:
  explicit Generator(Options options={}):options_(std::move(options)){}
  Result generate(const std::vector<Vec3>& seed,const ObstacleIndex& map,const Bounds& bounds)const {
    Result out;
      internal::cubic::OptimizerSettings settings;settings.profile=options_.profile;
      auto r=internal::cubic::generateCubic(seed,map,bounds,options_,settings);
      out.status=r.status;out.message=std::move(r.message);
      out.metrics.total_ms=r.total_ms;out.metrics.controls=r.optimizer.controls;
      out.metrics.constraints=r.optimizer.constraints;out.metrics.iterations=r.optimizer.steps;
      out.metrics.refinements=r.refinements;out.metrics.solver=std::move(r.optimizer);
      if(out){out.corridor.knots_=std::move(r.corridor.breaks);out.corridor.controls_=std::move(r.corridor.controls);
        out.support_planes=std::move(r.cover);}
    return out;
  }
};
inline internal::cubic::Certificate certify(const Corridor& corridor,
    const std::vector<SupportPlane>& planes) {
  internal::cubic::Geometry g;g.breaks=corridor.knots();g.controls=corridor.controls();
  return internal::cubic::certify(g,planes);
}
} // namespace SweepCorridor
