// SPDX-License-Identifier: MIT
// Read-only use of the user's SplineTrajectory reference implementation.
// This is a parameterization experiment, not a collision-certified generator.
#include <SplineConvexHull.hpp>
#include <SplineTrajectory.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
using Points = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

template <class Matrix>
void jsonMatrix(std::ostream& out, const Matrix& a) {
  out << '[';
  for (int i = 0; i < a.rows(); ++i) {
    if (i) out << ',';
    out << '[';
    for (int j = 0; j < a.cols(); ++j) {
      if (j) out << ',';
      out << a(i, j);
    }
    out << ']';
  }
  out << ']';
}

// Six-point Gauss integration is exact for squared second/third derivatives
// of a degree-five polynomial; it avoids cancellation in power antiderivatives.
template <class Poly>
double energy(const Poly& p, int derivative) {
  constexpr double x[] = {-.9324695142031521, -.6612093864662645, -.2386191860831969,
                            .2386191860831969, .6612093864662645, .9324695142031521};
  constexpr double w[] = {.1713244923791704, .3607615730481386, .4679139345726910,
                          .4679139345726910, .3607615730481386, .1713244923791704};
  double sum = 0;
  for (auto piece : p)
    for (int j = 0; j < 6; ++j)
      sum += .5 * piece.duration() * w[j] *
             piece.evaluateLocal(.5 * piece.duration() * (1 + x[j]), derivative).squaredNorm();
  return sum;
}

template <class Spline>
void probe(const std::filesystem::path& output, std::ostream& summary,
           const std::string& name, const std::vector<double>& knots,
           const Points& points, bool save) {
  Spline spline(knots, points);
  if (!spline.isValid()) throw std::runtime_error("reference spline construction failed");
  const int count = int(knots.size()) - 1, degree = Spline::kDegree;
  std::vector<double> times;
  const int batch = std::max(20, 12000 / count);
  volatile double checksum = 0;
  for (int repeat = 0; repeat < 7; ++repeat) {
    auto start = Clock::now();
    for (int i = 0; i < batch; ++i) {
      spline.update(knots, points);
      checksum += spline.polynomial().coefficients()(0, 0);
    }
    times.push_back(std::chrono::duration<double, std::micro>(Clock::now()-start).count()/batch);
  }
  std::sort(times.begin(), times.end());
  const auto& poly = spline.polynomial();
  double residual = 0, jump = 0, min_width = INFINITY;
  Eigen::Matrix<double, 3, 3> previous;
  int i = 0;
  for (auto piece : poly) {
    residual = std::max(residual, (piece.evaluateLocal(0) - points.row(i).transpose()).norm());
    for (int order = 0; order <= 2; ++order) {
      if (i) jump = std::max(jump, (previous.col(order)-piece.evaluateLocal(0, order)).norm());
      previous.col(order) = piece.evaluateLocal(piece.duration(), order);
    }
    for (int j = 0; j <= 128; ++j)
      min_width = std::min(min_width,
          piece.evaluateLocal(piece.duration() * j / 128.).template tail<2>().minCoeff());
    ++i;
  }
  residual = std::max(residual, (previous.col(0) - points.bottomRows(1).transpose()).norm());
  if (residual > 1e-9 || jump > 1e-8) throw std::runtime_error("interpolation / C2 residual failed");
  const double acc = energy(poly, 2), jerk = energy(poly, 3);
  const double expected = degree == 3 ? acc : jerk;
  if (std::abs(spline.energy() - expected) > 1e-8 * std::max(1., expected))
    throw std::runtime_error("reference energy differs from independent Gauss quadrature");
  auto hull = SplineTrajectory::toBezier(poly);
  auto start = Clock::now();
  for (int repeat = 0; repeat < batch; ++repeat) hull.update(poly);
  const double hull_us = std::chrono::duration<double, std::micro>(Clock::now()-start).count()/batch;
  summary << name << ',' << degree << ',' << count << ',' << times[3] << ','
          << hull_us << ',' << acc << ',' << jerk << ',' << residual << ',' << jump << ','
          << min_width << '\n';
  if (save) {
    std::ofstream file(output / (name + "_degree" + std::to_string(degree) + ".json"));
    file << std::setprecision(17) << "{\"degree\":" << degree << ",\"knots\":[";
    for (std::size_t k = 0; k < knots.size(); ++k) file << (k ? "," : "") << knots[k];
    file << "],\"values\":";
    jsonMatrix(file, points);
    file << ",\"coefficients\":";
    jsonMatrix(file, poly.coefficients());
    file << ",\"bezier\":";
    jsonMatrix(file, hull.controls());
    file << "}\n";
  }
  (void)checksum;
}

int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: spline_probe output_directory\n"; return 2; }
  const std::filesystem::path output(argv[1]);
  std::filesystem::create_directories(output);
  std::ofstream summary(output / "spline.csv");
  summary << std::setprecision(17)
          << "case,degree,segments,update_us,bezier_update_us,acc_energy,jerk_energy,knot_error,c2_jump,sampled_min_halfwidth\n";
  for (int count : {16, 64, 256, 1024}) {
    std::vector<double> knots(count + 1);
    Points points(count + 1, 3);
    for (int i = 0; i <= count; ++i) {
      const double s = knots[i] = .4 * i;
      points.row(i) << .15 * std::sin(.16 * s), .2 + .55 * (1+std::cos(.4*s)),
                       .3 + .45 * (1+std::sin(.23*s));
    }
    const auto name = "smooth_" + std::to_string(count);
    probe<SplineTrajectory::CubicSplineND<3>>(output, summary, name, knots, points, count == 16);
    probe<SplineTrajectory::QuinticSplineND<3>>(output, summary, name, knots, points, count == 16);
  }
  std::vector<double> knots{0, 1, 2, 3, 4, 5};
  Points throat(6, 3);
  throat << 0,.8,.5, 0,.8,.5, 0,.03,.5, 0,.03,.5, 0,.8,.5, 0,.8,.5;
  probe<SplineTrajectory::CubicSplineND<3>>(output, summary, "throat", knots, throat, true);
  probe<SplineTrajectory::QuinticSplineND<3>>(output, summary, "throat", knots, throat, true);
  // Export source matrices rather than copying third-party constants into this
  // repository. Their provenance and hashes are recorded by the experiment runner.
  std::ofstream basis(output / "basis.json");
  basis << std::setprecision(17) << '{';
  for (int degree : {3, 5}) {
    if (degree == 5) basis << ',';
    basis << '"' << degree << "\":{\"bezier\":";
    jsonMatrix(basis, SplineTrajectory::detail::HullKernelFactory::bezierPowerToControlMatrix(degree));
    basis << ",\"minvo\":";
    jsonMatrix(basis, SplineTrajectory::detail::HullKernelFactory::minvoPowerToControlMatrix(degree));
    basis << '}';
  }
  basis << "}\n";
  std::cout << "Cubic/quintic interpolation, C2 and independent energy checks passed.\n";
}
