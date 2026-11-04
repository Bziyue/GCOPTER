// SPDX-License-Identifier: MIT
#pragma once
#include <Eigen/Core>
#include <array>
#include <functional>
#include <string>
#include <vector>

namespace SweepCorridor::internal {
namespace detail { struct SupportPlane; }
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

// Geometry and resource inputs. Lengths in metres; budget in milliseconds.
struct GeometryOptions {
  double control_spacing = 0.5;
  double min_radius = 0.02; // Optimizer floor.
  double max_radius = 1.5;
  double robot_radius = 0.2;
  double safety_margin = 0.02;
  double certificate_margin = 0.002; // Additional metric margin.
  double budget_ms = 1000;
  double max_offset = 1.5; // Transverse center displacement.
  // Seed-segment AABBs expand by local_range; long edges split at local_progress.
  double local_range = 0;
  double local_progress = 7;
  double frame_smoothing = 0; // Arc-length filter scale; zero uses seed tangents.
  int certificate_depth = 9;
  int max_refinements = 6;
  std::size_t max_controls = 20000; // B-spline control points, not export rows.
};
enum class Status {
  Success,
  InvalidInput,
  SeedCollision,
  InsufficientClearance,
  DegeneratePath,
  Capacity,
  Timeout,
  VerificationFailed
};
const char* statusName(Status status);
} // namespace SweepCorridor::internal
