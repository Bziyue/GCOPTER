// SPDX-License-Identifier: MIT
#pragma once
#include <Eigen/Core>
#include <array>
#include <functional>
#include <string>
#include <vector>

namespace gcopter::direct {
using Vec3 = Eigen::Vector3d;
using Mat32 = Eigen::Matrix<double, 3, 2>;

/** Closed solid sphere in the caller's Cartesian frame, in metres. */
struct Sphere {
  Vec3 center = Vec3::Zero();
  double radius = 0;
};
/** Finite world bounds; robot clearance is subtracted by generate(). */
struct Bounds {
  Vec3 lower, upper;
};

/** Immutable reusable sphere BVH. Construction validates inputs and may throw.
 * Concurrent read-only queries/generations are safe. No borrowed map storage.
 * Represent a voxel by its centre and radius sqrt(3)*resolution/2; a bare point
 * is NOT a solid voxel and unobserved space is NOT inferred from this API.
 */
class ObstacleIndex {
 public:
  explicit ObstacleIndex(std::vector<Sphere> obstacles);
  const std::vector<Sphere>& obstacles() const { return obstacles_; }
  /** Find all spheres whose AABB intersects [lower, upper]; output is replaced.
   */
  void query(const Vec3& lower, const Vec3& upper, std::vector<int>& ids) const;

 private:
  struct Node {
    Vec3 lower, upper;
    int begin = 0, end = 0, left = -1, right = -1;
  };
  int build(int begin, int end);
  std::vector<Sphere> obstacles_;
  std::vector<int> order_;
  std::vector<Node> nodes_;
};

/** Geometric output, NOT a timed or dynamically feasible trajectory. */
struct Section {
  Vec3 center, first, second;
  Mat32 shape, shape_first, shape_second;
  /** x(u,v) = center(u) + shape(u)*v, ||v||<=1. All u derivatives use [0,1]. */
  Vec3 point(const Eigen::Vector2d& v) const { return center + shape * v; }
};
struct Differential {
  double value = 0;
  Eigen::Vector4d gradient = Eigen::Vector4d::Zero();  // [u, world x,y,z]
  Eigen::Matrix4d hessian = Eigen::Matrix4d::Zero();
  double plane_residual =
      0;  // metres; q<=0 alone describes an infinite cylinder!
};

struct HermiteNode {
  double u = 0;   // common normalized path parameter
  Section state;  // value / first / second derivatives, in world coordinates
};

/** C2 world-coordinate quintic Hermite affine-disc sweep.
 * Sections are ellipses, not necessarily exactly normal to center'(u).
 * No global injectivity/self-intersection or full-tube Jacobian guarantee.
 * Read-only evaluation is thread-safe. No nearest-path inversion is performed.
 */
class Corridor {
 public:
  std::size_t pieces() const { return knots_.empty() ? 0 : knots_.size() - 1; }
  Section section(double u) const;
  /** q=||shape^+(x-center)||^2-1 with exact u/xyz gradient and Hessian.
   * Membership also requires plane_residual=0. Throws for invalid u/x or empty
   * output.
   */
  Differential evaluate(double u, const Vec3& x) const;
  const std::vector<Vec3>& centerControls() const { return centers_; }
  const std::vector<Mat32>& shapeControls() const { return shapes_; }
  /** Piece-major equivalent Bezier coefficients: SIX rows per interval.
   * knots()[i:i+2] gives the interval; these are NOT B-spline controls. */
  const std::vector<double>& knots() const { return knots_; }
  const std::vector<HermiteNode>& hermiteNodes() const { return nodes_; }

 private:
  friend struct GeneratorAccess;
  std::vector<Vec3> centers_;
  std::vector<Mat32> shapes_;
  std::vector<double> knots_;
  std::vector<HermiteNode> nodes_;
};

struct Options {
  double control_spacing = 0.5;
  double min_radius =
      0.02;  // control-disc semiaxis floor, not a swept-section floor
  double max_radius = 1.5;
  double robot_radius = 0.2;
  double safety_margin = 0.02;
  double numerical_guard = 1e-8;
  double budget_ms = 1000;
  double max_offset = 1.5;  // transverse center displacement at Hermite nodes
  // Optional seed-segment AABBs, expanded by local_range in each world axis.
  // Zero disables these limits. Long seed edges are split at local_progress.
  double local_range = 0;
  double local_progress = 7;
  double certificate_margin =
      .002;  // additional metric margin, not a radius scale
  int inflation_iterations = 2;
  int certificate_depth = 9;
  int max_refinements = 6;
  std::size_t max_controls =
      20000;  // limit on Hermite nodes, not exported rows
  std::size_t max_constraints = 2000000;
  std::function<bool()> cancelled;
};
enum class Status {
  Success,
  InvalidInput,
  SeedCollision,
  InsufficientClearance,
  DegeneratePath,
  Capacity,
  Timeout,
  Cancelled,
  VerificationFailed
};
const char* statusName(Status status);
struct Metrics {
  std::size_t controls = 0, constraints = 0, obstacle_pairs = 0;
  std::size_t nodes = 0, certificate_leaves = 0, active_constraints = 0;
  int inflation_steps = 0, newton_steps = 0, optimization_passes = 0;
  double max_center_shift = 0, optimization_ms = 0, certificate_ms = 0;
  int refinements = 0;
  double total_ms = 0;
};
struct Result {
  Status status = Status::InvalidInput;
  std::string message;
  Corridor corridor;  // empty on every failure; no stale/partial output
  Metrics metrics;
  explicit operator bool() const { return status == Status::Success; }
};

/** Direct seed -> certified quintic Hermite ellipse sweep, with no FIRI/SFC
 * generation. Seed order and endpoints are preserved, interior corners may be
 * smoothed. Closed sphere obstacles are inflated by robot_radius+safety_margin.
 * Bounds are eroded by the same amount. All inputs must share one immutable
 * world snapshot. Consecutive duplicates are removed. Backtracking/cusps,
 * insufficient clearance, budget/capacity limits fail explicitly; arbitrary
 * valid seed length is supported within the configured capacity, but arbitrary
 * invalid paths are not repairable. Success checks entire spline intervals
 * using nonnegative basis weights, not sampled rings. Certificates are
 * floating-point with guards, not formal intervals. index construction time is
 * excluded from metrics.total_ms. No persistent cache.
 */
Result generate(const std::vector<Vec3>& seed, const ObstacleIndex& index,
                const Bounds& bounds, const Options& options = {});
}  // namespace gcopter::direct
