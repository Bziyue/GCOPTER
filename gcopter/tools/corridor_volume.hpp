// SPDX-License-Identifier: MIT
// Offline union-volume measurement; never included in generation timings.
#pragma once
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <array>
#include <boost/random/sobol.hpp>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "gcopter/direct_corridor.hpp"

namespace corridor_volume {
using namespace gcopter::direct;
inline double choose(int n, int k) {
  double a = 1;
  for (int j = 1; j <= k; ++j) a *= double(n + 1 - j) / j;
  return a;
}
struct Box {
  Vec3 lo = Vec3::Constant(INFINITY), hi = Vec3::Constant(-INFINITY);
  bool contains(const Vec3& p) const {
    return (p.array() >= lo.array()).all() && (p.array() <= hi.array()).all();
  }
};
struct Poly {
  Eigen::MatrixX4d h;
  Box box;
  bool valid = false;
  explicit Poly(Eigen::MatrixX4d planes) : h(std::move(planes)) {
    // Independent, bounded-memory vertex enumeration. Do not let an unchecked
    // dual-hull conversion determine the allocation extent of the volume grid.
    if (!h.allFinite() || h.rows() > 256)
      throw std::runtime_error("invalid or oversized volume polytope");
    if (h.rows() < 4) return;
    Vec3 sum = Vec3::Zero();
    int count = 0;
    for (int i = 0; i < h.rows(); ++i)
      for (int j = i + 1; j < h.rows(); ++j)
        for (int k = j + 1; k < h.rows(); ++k) {
          const Vec3 a = h.row(i).head<3>(), b = h.row(j).head<3>(),
                     c = h.row(k).head<3>();
          const Vec3 bc = b.cross(c);
          const double det = a.dot(bc);
          if (std::abs(det) < 1e-10) continue;
          const Vec3 v =
              (-h(i, 3) * bc - h(j, 3) * c.cross(a) - h(k, 3) * a.cross(b)) /
              det;
          if (!v.allFinite()) continue;
          bool inside = true;
          for (int row = 0; row < h.rows(); ++row)
            if (h.row(row).head<3>().dot(v) + h(row, 3) > 1e-7) {
              inside = false;
              break;
            }
          if (!inside) continue;
          box.lo = box.lo.cwiseMin(v);
          box.hi = box.hi.cwiseMax(v);
          sum += v;
          ++count;
        }
    if (count < 4) return;
    const Vec3 center = sum / count;
    for (int row = 0; row < h.rows(); ++row)
      if (h.row(row).head<3>().dot(center) + h(row, 3) >= -1e-9) return;
    box.lo.array() -= 1e-7;
    box.hi.array() += 1e-7;
    valid = true;
  }
  bool contains(const Vec3& p) const {
    if (!valid || !box.contains(p)) return false;
    for (int i = 0; i < h.rows(); ++i)
      if (h.row(i).head<3>().dot(p) + h(i, 3) > 1e-9) return false;
    return true;
  }
};
struct Patch {
  std::array<Vec3, 4> c;
  std::array<Mat32, 4> m;
  // The section-plane equation is degree 9 in the local parameter.
  std::array<Eigen::Vector4d, 10> plane;
  Box box;
  void prepare() {
    std::array<Vec3, 7> normal;
    for (auto& v : normal) v.setZero();
    for (int i = 0; i < 4; ++i) {
      const Vec3 e = m[i].rowwise().norm();
      box.lo = box.lo.cwiseMin(c[i] - e);
      box.hi = box.hi.cwiseMax(c[i] + e);
      for (int j = 0; j < 4; ++j)
        normal[i + j] += m[i].col(0).cross(m[j].col(1)) * choose(3, i) *
                         choose(3, j) / choose(6, i + j);
    }
    box.lo.array() -= 1e-9;
    box.hi.array() += 1e-9;
    for (auto& p : plane) p.setZero();
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 7; ++j) {
        const double w = choose(3, i) * choose(6, j) / choose(9, i + j);
        plane[i + j].head<3>() += w * normal[j];
        plane[i + j][3] -= w * c[i].dot(normal[j]);
      }
  }
  bool disc(double t, const Vec3& point) const {
    auto cp = c;
    auto mp = m;
    for (int k = 3; k > 0; --k)
      for (int j = 0; j < k; ++j) {
        cp[j] = (1 - t) * cp[j] + t * cp[j + 1];
        mp[j] = (1 - t) * mp[j] + t * mp[j + 1];
      }
    const Eigen::Matrix2d g = mp[0].transpose() * mp[0];
    const Eigen::Vector2d rhs = mp[0].transpose() * (point - cp[0]);
    if (g.determinant() <= 1e-20) return false;
    const Eigen::Vector2d v = g.inverse() * rhs;
    return v.squaredNorm() <= 1 + 1e-9 &&
           (point - cp[0] - mp[0] * v).norm() < 1e-6;
  }
  static double evaluate(std::array<double, 10> v, double t,
                         double& derivative) {
    for (int k = 9; k > 1; --k)
      for (int j = 0; j < k; ++j) v[j] = (1 - t) * v[j] + t * v[j + 1];
    derivative = 9 * (v[1] - v[0]);
    return (1 - t) * v[0] + t * v[1];
  }
  bool roots(const std::array<double, 10>& v, const Vec3& p, double a, double b,
             int depth, std::size_t& unresolved) const {
    const auto mm = std::minmax_element(v.begin(), v.end());
    if (*mm.first > 0 || *mm.second < 0) return false;
    int variations = 0, last = 0;
    for (double f : v) {
      int sign = (f > 0) - (f < 0);
      if (sign) {
        if (last && sign != last) ++variations;
        last = sign;
      }
    }
    if (v.front() == 0 && disc(a, p)) return true;
    if (v.back() == 0 && disc(b, p)) return true;
    if (variations == 0) return false;
    if (variations == 1 && v.front() * v.back() < 0) {
      double lo = 0, hi = 1,
             t = std::abs(v.front()) /
                 (std::abs(v.front()) + std::abs(v.back()));
      for (int k = 0; k < 40; ++k) {
        double derivative = 0, f = evaluate(v, t, derivative);
        if (std::abs(f) < 1e-11 || hi - lo < 1e-11) break;
        if ((f > 0) == (v.front() > 0))
          lo = t;
        else
          hi = t;
        double next = t - f / derivative;
        t = std::isfinite(next) && next > lo && next < hi ? next
                                                          : .5 * (lo + hi);
      }
      return disc(a + (b - a) * t, p);
    }
    if (depth == 32) {
      ++unresolved;
      return disc(.5 * (a + b), p);
    }
    auto x = v;
    std::array<double, 10> left, right;
    left[0] = x[0];
    right[9] = x[9];
    for (int k = 1; k <= 9; ++k) {
      for (int j = 0; j <= 9 - k; ++j) x[j] = .5 * (x[j] + x[j + 1]);
      left[k] = x[0];
      right[9 - k] = x[9 - k];
    }
    return roots(left, p, a, .5 * (a + b), depth + 1, unresolved) ||
           roots(right, p, .5 * (a + b), b, depth + 1, unresolved);
  }
  bool contains(const Vec3& p, std::size_t& unresolved) const {
    if (!box.contains(p)) return false;
    std::array<double, 10> values;
    for (int i = 0; i < 10; ++i)
      values[i] = plane[i].head<3>().dot(p) + plane[i][3];
    return roots(values, p, 0, 1, 0, unresolved);
  }
};
inline std::vector<Patch> patches(const Corridor& corridor) {
  std::vector<Patch> result(corridor.pieces());
  for (std::size_t i = 0; i < result.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      result[i].c[j] = corridor.controls()[4 * i + j].head<3>();
      result[i].m[j] = Eigen::Map<const Mat32>(corridor.controls()[4 * i + j].data()+3);
    }
    result[i].prepare();
  }
  return result;
}
struct Estimate {
  double mean = 0, ci95 = 0;
};
struct Measurement {
  // Cubic union, FIRI union, same 3-m seed ROI for each, intersection.
  std::array<Estimate, 5> volume;
  Estimate paired_difference;  // second union minus first, shared samples
  std::size_t unresolved = 0, cells = 0, samples = 0;
};
struct Cell {
  std::array<int, 3> key;
  std::vector<int> patches, polys;
};
template <class Other = Poly>
inline Measurement measure(const std::vector<Patch>& patches,
                           const std::vector<Other>& polys,
                           const std::vector<Vec3>& seed, int samples = 65536,
                           int replicates = 8, double cell_size = 1.,
                           double roi_radius = 3.) {
  if (replicates != 8)
    throw std::invalid_argument(
        "eight random shifts required for the reported t interval");
  if (samples < 1 || samples > 1048576 || !std::isfinite(cell_size) ||
      cell_size <= 0)
    throw std::invalid_argument("invalid volume sample/grid capacity");
  std::map<std::array<int, 3>, Cell> bins;
  std::size_t associations = 0;
  auto insert = [&](const Box& b, int index, bool patch) {
    if (!b.lo.allFinite() || !b.hi.allFinite() ||
        (b.hi.array() < b.lo.array()).any() ||
        b.lo.cwiseAbs().maxCoeff() / cell_size > 1e8 ||
        b.hi.cwiseAbs().maxCoeff() / cell_size > 1e8)
      throw std::runtime_error("invalid volume bounding box");
    const Eigen::Vector3i lo = (b.lo / cell_size).array().floor().cast<int>();
    const Eigen::Vector3i hi = (b.hi / cell_size).array().floor().cast<int>();
    const double required =
        (hi.cast<double>() - lo.cast<double>() + Vec3::Ones()).prod();
    if (required > 50000 || associations + required > 2000000)
      throw std::runtime_error(
          "volume grid capacity exceeded before allocation");
    associations += std::size_t(required);
    for (int x = lo[0]; x <= hi[0]; ++x)
      for (int y = lo[1]; y <= hi[1]; ++y)
        for (int z = lo[2]; z <= hi[2]; ++z) {
          if (bins.size() >= 50000)
            throw std::runtime_error("volume cell capacity exceeded");
          const std::array<int, 3> key{x, y, z};
          auto& cell = bins[key];
          cell.key = key;
          (patch ? cell.patches : cell.polys).push_back(index);
        }
  };
  for (int i = 0; i < int(patches.size()); ++i) insert(patches[i].box, i, true);
  for (int i = 0; i < int(polys.size()); ++i)
    if (polys[i].valid) insert(polys[i].box, i, false);
  std::vector<Cell> cells;
  for (auto& entry : bins) cells.push_back(std::move(entry.second));
  Measurement result;
  result.cells = cells.size();
  result.samples = std::size_t(samples) * replicates;
  if (cells.empty()) return result;
  const double domain = cells.size() * std::pow(cell_size, 3);
  std::array<std::vector<double>, 5> estimates;
  std::mt19937_64 random(20261102);
  for (int rep = 0; rep < replicates; ++rep) {
    boost::random::sobol sobol(4);
    std::array<uint64_t, 4> shift;
    for (auto& s : shift) s = random();
    std::array<std::size_t, 5> hits{};
    for (int k = 0; k < samples; ++k) {
      std::array<double, 4> u;
      for (int d = 0; d < 4; ++d)
        u[d] = double((sobol() ^ shift[d]) >> 11) * 0x1.0p-53;
      const auto& cell =
          cells[std::min(cells.size() - 1, std::size_t(u[0] * cells.size()))];
      const Vec3 p = cell_size * Vec3(cell.key[0] + u[1], cell.key[1] + u[2],
                                      cell.key[2] + u[3]);
      bool h = false, f = false;
      for (int i : cell.patches)
        if (patches[i].contains(p, result.unresolved)) {
          h = true;
          break;
        }
      for (int i : cell.polys)
        if (polys[i].contains(p)) {
          f = true;
          break;
        }
      hits[0] += h;
      hits[1] += f;
      hits[4] += h && f;
      if (h || f) {
        bool in_roi = false;
        for (std::size_t j = 1; j < seed.size(); ++j) {
          const Vec3 d = seed[j] - seed[j - 1];
          const double t =
              d.squaredNorm() > 0
                  ? std::clamp((p - seed[j - 1]).dot(d) / d.squaredNorm(), 0.,
                               1.)
                  : 0;
          if ((p - seed[j - 1] - t * d).squaredNorm() <=
              roi_radius * roi_radius) {
            in_roi = true;
            break;
          }
        }
        hits[2] += h && in_roi;
        hits[3] += f && in_roi;
      }
    }
    for (int d = 0; d < 5; ++d)
      estimates[d].push_back(domain * hits[d] / samples);
  }
  for (int d = 0; d < 5; ++d) {
    auto& e = result.volume[d];
    e.mean = std::accumulate(estimates[d].begin(), estimates[d].end(), 0.) /
             replicates;
    double variance = 0;
    for (double v : estimates[d]) variance += (v - e.mean) * (v - e.mean);
    e.ci95 =
        2.364624251 * std::sqrt(variance / (replicates * (replicates - 1)));
  }
  auto& delta = result.paired_difference;
  delta.mean = result.volume[1].mean - result.volume[0].mean;
  double delta_variance = 0;
  for (int rep = 0; rep < replicates; ++rep) {
    const double error = estimates[1][rep] - estimates[0][rep] - delta.mean;
    delta_variance += error * error;
  }
  delta.ci95 = 2.364624251 *
      std::sqrt(delta_variance / (replicates * (replicates - 1)));
  return result;
}
}  // namespace corridor_volume
