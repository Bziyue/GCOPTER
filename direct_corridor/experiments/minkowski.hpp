// SPDX-License-Identifier: MIT
#pragma once
#include "gcopter/direct_corridor.hpp"

namespace gcopter::direct::minkowski {
using Weights = Eigen::Vector4d;
struct Options {
  direct::Options geometry;
  int templates = 4;                 // 2: orthogonal; 4: plus two diagonals
  double thinness = .05;              // minor/major ratio of each ellipse
  int subdivisions = 0;             // extra support-bound splits, 0..2
  int newton_iterations = 12;        // per barrier stage
  double angle = 0;                 // rotation in the transported frame
  bool adapt_orientation = false;   // centered polar-ellipsoid orientation
  bool product_bounds = false;      // degree-10 positive Bernstein products
  bool affine = false;             // fixed Minkowski body, free SPD transforms
  double orientation_smoothing = 0; // banded quadratic angle regularization
};
struct Section {
  Vec3 center, first, second;
  Mat32 frame, frame_first, frame_second;
  Weights weights, weights_first, weights_second;
};
struct Boundary {
  Vec3 point, first, second, angle_first, mixed, angle_second;
};
/** Experimental C2 sweep of planar smooth Minkowski sums.
 * The shared rank-two frame is polynomial and need not be orthonormal between
 * knots. All templates live in that same plane. No global injectivity claim.
 */
class Corridor {
 public:
  std::size_t pieces() const { return knots_.empty() ? 0 : knots_.size()-1; }
  Section section(double u) const;
  Boundary boundary(double u, double normal_angle) const;
  /** q=gauge^4-1, C2 also at the center; exact u/xyz derivatives.
   * Membership additionally requires plane_residual=0. */
  Differential evaluate(double u, const Vec3& x) const;
  double area(double u) const;
  int templates() const { return templates_; }
  const std::array<Eigen::Matrix2d, 4>& templateMatrices() const { return q_; }
  const std::vector<double>& knots() const { return knots_; }
  const std::vector<Vec3>& centerControls() const { return centers_; }
  const std::vector<Mat32>& frameControls() const { return frames_; }
  const std::vector<Weights>& weightControls() const { return weights_; }
  // Low-level query for an already evaluated section, used by offline volume
  // measurement. Inputs here come from section()/the stored polynomial data.
  double gauge(const Eigen::Vector2d& z, const Weights& r) const;

 private:
  friend struct Access;
  friend struct AffineAccess;
  int templates_ = 0;
  std::array<Eigen::Matrix2d, 4> q_;
  Eigen::Matrix4d mixed_;
  std::vector<double> knots_;
  std::vector<Vec3> centers_;
  std::vector<Mat32> frames_;
  std::vector<Weights> weights_;
  double normalAngle(const Eigen::Vector2d& z, const Weights& r) const;
};
struct Result {
  Status status = Status::InvalidInput;
  std::string message;
  Corridor corridor;
  Metrics metrics;
  explicit operator bool() const { return status == Status::Success; }
};
Result generate(const std::vector<Vec3>& seed, const ObstacleIndex& index,
                const Bounds& bounds, const Options& options = {});
}  // namespace gcopter::direct::minkowski
