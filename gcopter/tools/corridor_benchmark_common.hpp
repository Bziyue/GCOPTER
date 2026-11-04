#pragma once
// SPDX-License-Identifier: MIT
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "corridor_volume.hpp"
#include "gcopter/sfc_gen.hpp"

namespace {
using namespace SweepCorridor;
using Clock = std::chrono::steady_clock;
using Polys = traj_opt_components::PolyhedraH;
double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
std::vector<Vec3> read(const char* path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("input not found");
  std::vector<Vec3> values;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    Vec3 p;
    if (!(row >> p.x() >> p.y() >> p.z()) || !p.allFinite())
      throw std::runtime_error("invalid XYZ");
    values.push_back(p);
  }
  return values;
}
// Same segment stepping, full-cloud filtering and gap insertion as the
// repository's convexCover. Only fix empty-cloud indexing and check return
// values; firi::firi itself and its four iterations are unchanged.
Polys cover(const std::vector<Vec3>& path, const std::vector<Vec3>& points,
            const Bounds& bounds, double progress, double range) {
  Polys result;
  Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
  for (int j = 0; j < 3; ++j) {
    bd(2 * j, j) = 1;
    bd(2 * j + 1, j) = -1;
  }
  Vec3 a, b = path.front();
  std::vector<Vec3> selected;
  selected.reserve(points.size());
  for (std::size_t i = 1; i < path.size();) {
    a = b;
    if ((a - path[i]).norm() > progress)
      b = a + (path[i] - a).normalized() * progress;
    else
      b = path[i++];
    for (int j = 0; j < 3; ++j) {
      bd(2 * j, 3) = -std::min(std::max(a[j], b[j]) + range, bounds.upper[j]);
      bd(2 * j + 1, 3) =
          std::max(std::min(a[j], b[j]) - range, bounds.lower[j]);
    }
    selected.clear();
    for (const auto& p : points)
      if ((bd.leftCols<3>() * p + bd.rightCols<1>()).maxCoeff() < 0)
        selected.push_back(p);
    Eigen::Map<const Eigen::Matrix3Xd> pc(
        selected.empty() ? nullptr : selected.front().data(), 3,
        selected.size());
    Eigen::MatrixX4d hp, gap;
    if (!firi::firi(bd, pc, a, b, hp) || !hp.allFinite())
      throw std::runtime_error("FIRI returned invalid polytope");
    if (!result.empty()) {
      const Eigen::Vector4d ah(a[0], a[1], a[2], 1);
      if (3 <= ((hp * ah).array() > -1e-6).cast<int>().sum() +
                   ((result.back() * ah).array() > -1e-6).cast<int>().sum()) {
        if (!firi::firi(bd, pc, a, a, gap, 1) || !gap.allFinite())
          throw std::runtime_error("FIRI gap failure");
        result.push_back(gap);
      }
    }
    result.push_back(hp);
  }
  return result;
}
struct FiriResult {
  Polys polys;
  double cover_ms = 0, erosion_ms = 0, shortcut_ms = 0, total_ms = 0;
  int before_shortcut = 0;
};
struct ExpansionBox {
  double begin, end;
  Bounds bounds;
};
std::vector<ExpansionBox> expansionBoxes(const std::vector<Vec3>& seed,
                                         const Bounds& bounds, double range,
                                         double pad) {
  std::vector<ExpansionBox> boxes;
  Vec3 a, b = seed.front();
  double arc = 0;
  for (std::size_t i = 1; i < seed.size();) {
    a = b;
    if ((a - seed[i]).norm() > 7)
      b = a + (seed[i] - a).normalized() * 7;
    else
      b = seed[i++];
    const double end = arc + (b - a).norm();
    boxes.push_back({arc,
                     end,
                     {(a.cwiseMin(b) - Vec3::Constant(range))
                          .cwiseMax(bounds.lower + Vec3::Constant(pad)),
                      (a.cwiseMax(b) + Vec3::Constant(range))
                          .cwiseMin(bounds.upper - Vec3::Constant(pad))}});
    arc = end;
  }
  return boxes;
}
FiriResult generateFiri(const std::vector<Vec3>& seed,
                        const std::vector<Vec3>& points, Bounds bounds,
                        double obstacle_radius, double pad, bool solid,
                        double range, double progress=7., double certificate_margin=.002) {
  const auto start = Clock::now();
  FiriResult out;
  const double erosion = solid ? obstacle_radius + pad + certificate_margin : 0;
  // Expand artificial local limits before erosion, so their final nominal
  // range remains 3 m; adjust world bounds to end at the same robot clearance.
  if (solid) {
    bounds.lower.array() -= obstacle_radius;
    bounds.upper.array() += obstacle_radius;
  }
  out.polys = cover(seed, points, bounds, progress, range + erosion);
  out.cover_ms = ms(start);
  auto timer = Clock::now();
  for (auto& h : out.polys)
    for (int j = 0; j < h.rows(); ++j) {
      h.row(j) /= h.row(j).head<3>().norm();
      h(j, 3) += erosion;
    }
  out.erosion_ms = ms(timer);
  out.before_shortcut = out.polys.size();
  timer = Clock::now();
  sfc_gen::shortCut(out.polys);
  out.shortcut_ms = ms(timer);
  out.total_ms = ms(start);
  return out;
}
double quantile(std::vector<double> values, double q) {
  std::sort(values.begin(), values.end());
  return values[std::min(values.size() - 1,
                         std::size_t(std::ceil(q * values.size()) - 1))];
}
void list(std::ostream& out, const std::vector<double>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i)
    out << (i ? "," : "") << values[i];
  out << ']';
}
} // namespace
