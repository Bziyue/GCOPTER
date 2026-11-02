// SPDX-License-Identifier: MIT
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#include "gcopter/direct_corridor.hpp"
using namespace gcopter::direct;
int main(int argc, char** argv) {
  const int repeats = argc > 1 ? std::max(1, std::stoi(argv[1])) : 15;
  std::cout << "scenario,obstacles,seed_points,index_ms,median_ms,p95_ms,"
               "controls,constraints,min_axis,max_axis,successes,repeats\n";
  bool all = true;
  for (int scenario = 0; scenario < 4; ++scenario) {
    double length = scenario == 3 ? 200 : 30;
    std::vector<Vec3> seed;
    const int n = scenario == 3 ? 4001 : 121;
    for (int i = 0; i < n; ++i) {
      double x = length * i / (n - 1);
      seed.emplace_back(x, 2 * std::sin(x * .25), .5 * std::sin(x * .4));
    }
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> x(-5, length + 5), y(-10, 10),
        z(-5, 5);
    std::vector<Sphere> spheres;
    const int count = scenario == 0 ? 0 : (scenario == 1 ? 10000 : 100000);
    while (int(spheres.size()) < count) {
      Vec3 p(x(rng), y(rng), z(rng));
      // Recorded synthetic free ribbon: not an upstream real-world map
      // benchmark.
      double dy = p.y() - 2 * std::sin(p.x() * .25),
             dz = p.z() - .5 * std::sin(p.x() * .4);
      if (std::hypot(dy, dz) > .9) spheres.push_back({p, .08});
    }
    const auto begin = std::chrono::steady_clock::now();
    ObstacleIndex map(std::move(spheres));
    double index_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
    Options o;
    o.control_spacing = .35;
    o.max_radius = 1.;
    o.robot_radius = .15;
    o.budget_ms = 10000;
    Bounds bounds{Vec3(-5, -10, -5), Vec3(length + 5, 10, 5)};
    std::vector<double> times;
    Result last;
    int successes = 0;
    double minaxis = 1e30, maxaxis = 0;
    for (int r = 0; r < repeats + 1; ++r) {
      auto result = generate(seed, map, bounds, o);
      if (r == 0) continue;
      times.push_back(result.metrics.total_ms);
      if (result) {
        ++successes;
        last = std::move(result);
      } else
        std::cerr << statusName(result.status) << ": " << result.message
                  << '\n';
    }
    if (last)
      for (int k = 0; k <= 1000; ++k) {
        auto m = last.corridor.section(k / 1000.).shape;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> e(m.transpose() * m);
        minaxis = std::min(minaxis, std::sqrt(e.eigenvalues()[0]));
        maxaxis = std::max(maxaxis, std::sqrt(e.eigenvalues()[1]));
      }
    std::sort(times.begin(), times.end());
    const char* names[] = {"open_30m", "sparse_30m", "dense_30m", "long_200m"};
    std::cout << names[scenario] << ',' << count << ',' << seed.size() << ','
              << index_ms << ',' << times[times.size() / 2] << ','
              << times[std::min(times.size() - 1,
                                std::size_t(std::ceil(times.size() * .95) - 1))]
              << ',' << last.metrics.controls << ',' << last.metrics.constraints
              << ',' << minaxis << ',' << maxaxis << ',' << successes << ','
              << repeats << '\n';
    all &= successes == repeats;
  }
  return all ? 0 : 1;
}
