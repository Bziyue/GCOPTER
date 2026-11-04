// SPDX-License-Identifier: MIT
#include "../tools/corridor_volume.hpp"

#include <iostream>
using namespace corridor_volume;
void require(bool ok, const char* reason) {
  if (!ok) throw std::runtime_error(reason);
}
Poly cube(double lo, double hi) {
  Eigen::MatrixX4d h = Eigen::MatrixX4d::Zero(6, 4);
  for (int i = 0; i < 3; ++i) {
    h(2 * i, i) = 1;
    h(2 * i + 1, i) = -1;
    h(2 * i, 3) = -(i == 0 ? hi : 1);
    h(2 * i + 1, 3) = i == 0 ? lo : 0;
  }
  return Poly(h);
}
int main() {
  try {
    Patch cylinder;
    for (int j = 0; j < 4; ++j) {
      cylinder.c[j] = Vec3(4. * j / 3, 0, 0);
      cylinder.m[j] << 0, 0, .5, 0, 0, .5;
    }
    cylinder.prepare();
    std::size_t unresolved = 0;
    require(cylinder.contains(Vec3(2, .2, .1), unresolved),
            "cylinder interior");
    require(!cylinder.contains(Vec3(2, .5, .5), unresolved),
            "cylinder radial exterior");
    require(!cylinder.contains(Vec3(4.1, 0, 0), unresolved),
            "cylinder axial exterior");
    auto v = measure({cylinder, cylinder}, {}, {Vec3(0, 0, 0), Vec3(4, 0, 0)},
                     16384);
    require(std::abs(v.volume[0].mean - std::acos(-1.)) < .025,
            "analytic cylinder union volume, duplicates excluded");
    const auto a = cube(0, 1), b = cube(.5, 1.5);
    v = measure({}, {a, a, b}, {Vec3::Zero(), Vec3(1, 0, 0)}, 16384);
    require(std::abs(v.volume[1].mean - 1.5) < .025,
            "overlapping cube union volume");
    // Two roots per section-plane equation: naive signed Jacobian integration
    // gives zero, whereas the geometric union is a cylinder of length one.
    Patch folded;
    for (int j = 0; j < 4; ++j) {
      folded.c[j] = Vec3(4. * j / 3 - 4. * j * (j - 1) / 6, 0, 0);
      folded.m[j] = cylinder.m[j];
    }
    folded.prepare();
    require(folded.contains(Vec3(.75, .2, .1), unresolved),
            "multiple plane roots");
    v = measure({folded}, {}, {Vec3::Zero(), Vec3(1, 0, 0)}, 16384);
    require(std::abs(v.volume[0].mean - std::acos(-1.) / 4) < .012,
            "folded sweep counted once");
    require(v.unresolved == 0 && unresolved == 0, "all roots resolved");
    auto oversized = cylinder;
    oversized.box.hi = Vec3::Constant(1e6);
    bool capacity = false;
    try {
      measure({oversized}, {}, {Vec3::Zero(), Vec3(1, 0, 0)}, 1);
    } catch (const std::runtime_error&) {
      capacity = true;
    }
    require(capacity, "oversized volume grid rejected before allocation");
    std::cout << "PASS: cylinder, overlapping polyhedra, multiple roots, "
                 "folded sweep union\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
